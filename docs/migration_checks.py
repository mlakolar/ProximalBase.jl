"""Small, deterministic preservation checks for documentation migration #1019.

This is a provenance check, not a Markdown renderer or a semantic equivalence
proof. Documenter owns rendering; independent reviewers own approved rewrites.
"""
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit
import csv
import json
import posixpath
import re
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "docs/audit/docs-tracks-1019"
BASELINE = "7ae86e2093060eb124211e2f21f043c6379f97ae"
NORMALIZER_VERSION = 2
FIELDS = ("block_id", "sha", "kind", "action", "dest", "dest_sha", "clause_ids", "reviewer")
ACTIONS = {"stay", "move", "split", "reword", "dedupe", "supersede", "archive", "upstream"}
REVIEWED = {"split", "reword", "dedupe", "supersede"}
CONTRACTS = {"docs/design/" + name for name in (
    "block-prox-0.6.md", "iterate-family-0.6.md", "symmetric-sparse-iterate-0.6.md")}
EXTRA = {"README.md", "CONTRIBUTING.md", "examples/README.md", "test/TESTING.md",
         "benchmark/README.md", "docs/repository_hygiene.md"}
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
LIST = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
LINK = re.compile(r"(\[[^\]\n]*\]\()([^\s)]+)(\))")
PROTECTED = re.compile(r"`+[^`]*`+|https?://[^\s)]+|\[[^\]]*\]\([^)]*\)", re.S)
INLINE_LITERAL = re.compile(
    r"(?<!`)(`+)(?!`)(.*?)\1(?!`)"
    r"|(?<!\\)\$\$.*?(?<!\\)\$\$"
    r"|(?<!\\)\$(?!\$)[^\n$]*?(?<!\\)\$(?!\$)"
    r"|\\\(.*?\\\)|\\\[.*?\\\]", re.S)


def git(*args, root=ROOT):
    return subprocess.check_output(["git", *args], cwd=root, text=True)


def baseline_paths(root=ROOT, baseline=BASELINE):
    paths = git("ls-tree", "-r", "--name-only", baseline, root=root).splitlines()
    return sorted(p for p in paths if p in EXTRA or p in CONTRACTS or
                  (p.startswith("docs/src/") and p.endswith(".md") and
                   "/examples/" not in p and "/contracts/" not in p))


def read_current(path, root=ROOT):
    file = root / path
    if file.is_file():
        return file.read_text(encoding="utf-8")
    # Sparse checkouts omit clean tracked files. A real deletion must still fail.
    if git("ls-files", "-v", "--", path, root=root).startswith("S "):
        return git("show", ":" + path, root=root)
    raise FileNotFoundError(path)


def canonical_links(text, path):
    def replace(match):
        target = match[2]
        url = urlsplit(target)
        if url.scheme or url.netloc or target.startswith(("@", "/")):
            return match[0]
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), url.path)) if url.path else path
        return match[1] + urlunsplit(("", "", resolved, url.query, url.fragment)) + match[3]
    return LINK.sub(replace, text)


def normalized(text, path, kind="prose"):
    text = text.replace("\r\n", "\n").strip()
    if kind in {"code", "math"}:
        return text
    # Mask literals before either whitespace normalization or link rebasing.
    # A Julia string containing spaces or Markdown-looking text is still code.
    literals = []
    marker = "\0literal:"
    while marker in text:
        marker += "_"

    def protect(value):
        literals.append(value)
        return f"{marker}{len(literals) - 1}\0"

    chunks, outside, fenced = [], [], []
    opened = None
    for line in text.splitlines(keepends=True):
        fence = FENCE.match(line)
        if opened is not None:
            fenced.append(line)
            if fence and fence[1][0] == opened[0] and len(fence[1]) >= len(opened) and not fence[2].strip():
                chunks.append(protect("".join(fenced)))
                fenced, opened = [], None
        elif fence:
            chunks.append("".join(outside))
            outside, fenced, opened = [], [line], fence[1]
        else:
            outside.append(line)
    if fenced:
        # Keep malformed/unclosed fences literal too; Markdown lint owns errors.
        chunks.append(protect("".join(fenced)))
    chunks.append("".join(outside))
    masked = INLINE_LITERAL.sub(lambda match: protect(match[0]), "".join(chunks))
    result = " ".join(canonical_links(masked, path).split())
    for index, literal in enumerate(literals):
        result = result.replace(f"{marker}{index}\0", literal)
    return result


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def sentences(text):
    """Split outside code, math and links; punctuation stays with its sentence.

    Abbreviations can yield extra units, but deterministic units are used on both
    sides. The checker does not claim linguistic sentence understanding.
    """
    mask = list(text)
    for pattern in (PROTECTED, INLINE_LITERAL):
        for match in pattern.finditer(text):
            mask[match.start():match.end()] = "x" * (match.end() - match.start())
    start = 0
    for match in re.finditer(r"[.!?](?=\s+\S)", "".join(mask)):
        end = match.end()
        if text[start:end].strip():
            yield text[start:end].strip()
        start = end
    if text[start:].strip():
        yield text[start:].strip()


@dataclass(frozen=True)
class Block:
    id: str
    path: str
    kind: str
    text: str
    line: int
    heading: str

    @property
    def norm(self):
        return normalized(self.text, self.path, self.kind)

    @property
    def sha(self):
        return digest(self.norm)


def blocks(text, path):
    """Inventory headings, fences, table rows, list items and prose paragraphs.

    Numerical prose is sentence-granular. Admonitions stay whole, including
    nested fences. Blank lines are formatting, not content blocks.
    """
    lines = text.splitlines()
    heading = []
    ordinal = 0
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        start = i
        line = lines[i]
        h = HEADING.match(line)
        f = FENCE.match(line)
        if h:
            depth = len(h[1])
            heading = heading[:depth - 1] + [h[2]]
            kind = "heading"
            i += 1
        elif f:
            kind = "math" if f[2].strip() == "math" else "code"
            i += 1
            while i < len(lines):
                closing = FENCE.match(lines[i])
                i += 1
                if closing and closing[1][0] == f[1][0] and len(closing[1]) >= len(f[1]) and not closing[2].strip():
                    break
        elif re.match(r"^\s*(?:!!!|\?\?\?)\s", line):
            kind = "admonition"
            indent = len(line) - len(line.lstrip())
            i += 1
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
                i += 1
        elif line.lstrip().startswith("|"):
            kind = "table-row"
            i += 1
        else:
            kind = "list-item" if LIST.match(line) else "prose"
            i += 1
            while i < len(lines) and lines[i].strip() and not (
                    HEADING.match(lines[i]) or FENCE.match(lines[i]) or
                    LIST.match(lines[i]) or lines[i].lstrip().startswith(("|", "!!!", "???"))):
                i += 1
        value = "\n".join(lines[start:i]).rstrip()
        heading_path = " / ".join(heading)
        numerical = (path == "docs/src/numerics.md" or path.startswith("docs/src/spec/") or
                     (path in {"docs/src/penalties.md", "docs/src/group_penalty.md", "docs/src/smooth.md", "docs/src/calculus.md"}
                      and re.search(r"numerical|endpoint|Exceptional|[Ww]eights|[Ss]teps|[Dd]omain|Unsupported|[Aa]lias|zero scale", heading_path)))
        units = sentences(value) if numerical and kind == "prose" else [value]
        for unit in units:
            ordinal += 1
            # The file and full heading path are stable provenance, not current routing.
            identifier = f"{path}::{heading_path}::{ordinal}"
            yield Block(identifier, path, kind, unit, start + 1, heading_path)


def baseline_blocks(root=ROOT, baseline=BASELINE):
    return [block for path in baseline_paths(root, baseline)
            for block in blocks(git("show", baseline + ":" + path, root=root), path)]


def write_tsv(path, fields, rows, baseline=BASELINE):
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write(f"# baseline\t{baseline}\n")
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader((line for line in stream if not line.startswith("# baseline\t")), delimiter="\t"))


class ContentAnchors(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.article_depth = 0
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            if self.article_depth:
                self.article_depth += 1
            elif attrs.get("id") == "documenter-page":
                self.article_depth = 1
        if self.article_depth and "id" in attrs:
            self.ids.add(attrs["id"])

    def handle_endtag(self, tag):
        if tag == "article" and self.article_depth:
            self.article_depth -= 1


def page_key(path):
    path = str(path).replace("\\", "/")
    if path == "index.html":
        return ""
    if path.endswith("/index.html"):
        return path[:-11]
    return path.removesuffix(".html")


def html_anchors(build):
    return {page_key(path.relative_to(build)): ContentAnchors(path.read_text(encoding="utf-8")).ids
            for path in sorted(build.rglob("*.html"))}


def anchor_snapshot(metadata, profile="local", audit=AUDIT):
    """Load one explicitly selected, complete runtime-specific baseline.

    Generated admonition fragments use Julia's version-dependent Base.hash.
    Profiles preserve every exact page/fragment pair; they are never fallback
    sets, suffix stripping, or a user-supplied path that could waive failures.
    """
    files = {"local": "anchors.tsv", "ci": "anchors-ci.tsv"}
    if profile not in files or profile not in metadata.get("anchor_profiles", {}):
        raise ValueError(f"unknown or unavailable anchor profile: {profile}")
    descriptor = metadata["anchor_profiles"][profile]
    if descriptor.get("file") != files[profile]:
        raise ValueError(f"unexpected snapshot filename for anchor profile {profile}")
    path = audit / files[profile]
    content = path.read_text(encoding="utf-8")
    if not content.startswith(f"# baseline\t{metadata['baseline']}\n"):
        raise ValueError(f"anchor profile {profile} names a different baseline")
    if digest(content) != descriptor.get("sha256"):
        raise ValueError(f"anchor profile {profile} snapshot hash mismatch")
    rows = read_tsv(path)
    pairs = {(row["page"], row["fragment"]) for row in rows}
    pages = {row["page"] for row in rows if not row["fragment"]}
    anchors = sum(bool(row["fragment"]) for row in rows)
    if not rows or len(pairs) != len(rows):
        raise ValueError(f"anchor profile {profile} is empty or has duplicate rows")
    if pages != {row["page"] for row in rows} or len(pages) != metadata["rendered_pages"]:
        raise ValueError(f"anchor profile {profile} has incomplete baseline page records")
    if anchors != metadata["content_anchor_count"]:
        raise ValueError(f"anchor profile {profile} has an incomplete baseline anchor count")
    return rows


def initial_ledger(base_blocks):
    return [dict(block_id=b.id, sha=b.sha, kind=b.kind, action="stay", dest=b.path,
                 dest_sha=b.sha, clause_ids="", reviewer="mechanical") for b in base_blocks]


def baseline_measurements(root=ROOT, baseline=BASELINE):
    paths = git("ls-tree", "-r", "--name-only", baseline, root=root).splitlines()
    page_paths = [p for p in baseline_paths(root, baseline) if p.startswith("docs/src/")]
    pages = {p: git("show", baseline + ":" + p, root=root) for p in page_paths}
    source = "\n".join(git("show", baseline + ":" + p, root=root)
                       for p in paths if p.startswith("src/") and p.endswith(".jl"))
    examples = [git("show", baseline + ":" + p, root=root)
                for p in paths if p.startswith("examples/") and p.endswith(".jl")]
    joined = "\n".join(pages.values())
    return dict(
        authored_page_words=sum(len(text.split()) for text in pages.values()),
        authored_page_headings=sum(b.kind == "heading" for path, text in pages.items() for b in blocks(text, path)),
        explicit_page_anchor_ids=len(re.findall(r"\(@id\s+[^)]+\)", joined)),
        cross_page_fragment_links=len(re.findall(r"\]\([^)]*\.md#[^)]*\)", joined)),
        page_ref_links=joined.count("(@ref"),
        executed_page_example_fences=len(re.findall(r"^\s*```@example\b", joined, re.M)),
        page_and_source_doctest_fences=len(re.findall(r"^\s*```jldoctest\b", joined + "\n" + source, re.M)),
        page_assertion_lines=sum("@assert" in line for line in joined.splitlines()),
        example_script_assertion_lines=sum("@assert" in line for text in examples for line in text.splitlines()),
        example_script_count=len(examples),
        source_numerics_citation_lines=sum("numerics.md" in line for p in paths
            if p.startswith(("src/", "test/")) and p.endswith(".jl")
            for line in git("show", baseline + ":" + p, root=root).splitlines()),
    )


def initialize(build, root=ROOT, audit=AUDIT, baseline=BASELINE):
    audit.mkdir(parents=True, exist_ok=True)
    baseline_data = baseline_blocks(root, baseline)
    if (audit / "ledger.tsv").exists():
        raise ValueError("Refusing to overwrite an existing ledger; move or restore it explicitly")
    write_tsv(audit / "ledger.tsv", FIELDS, initial_ledger(baseline_data), baseline)
    inventory = []
    all_paths = git("ls-tree", "-r", "--name-only", baseline, root=root).splitlines()
    scope = set(baseline_paths(root, baseline))
    inventoried = sorted(scope | {p for p in all_paths if p.startswith("docs/design/") and p.endswith(".md")})
    for path in inventoried:
        text = git("show", baseline + ":" + path, root=root)
        inventory.append(dict(path=path, bytes=len(text.encode()), sha=digest(text), words=len(text.split()),
                              scope="blocks" if path in scope else "historical-file"))
    write_tsv(audit / "inventory.tsv", ("path", "bytes", "sha", "words", "scope"), inventory, baseline)
    pages = html_anchors(build)
    if not pages or not any(pages.values()):
        raise ValueError("Baseline build has no content anchors; build the unchanged site first")
    anchor_rows = [dict(page=page, fragment=fragment) for page, ids in pages.items() for fragment in ["", *sorted(ids)]]
    write_tsv(audit / "anchors.tsv", ("page", "fragment"), anchor_rows, baseline)
    examples = {p: digest(git("show", baseline + ":" + p, root=root)) for p in all_paths
                if p.startswith("examples/") and p.endswith(".jl")}
    metadata = dict(schema=1, normalizer_version=NORMALIZER_VERSION, baseline=baseline, issue=1019, mode="report-only",
                    block_count=len(baseline_data), authored_pages=sum(p.startswith("docs/src/") for p in scope),
                    rendered_pages=len(pages), content_anchor_count=sum(map(len, pages.values())),
                    example_scripts=examples, measurements=baseline_measurements(root, baseline))
    metadata["anchor_profiles"] = {"local": {
        "file": "anchors.tsv", "sha256": digest((audit / "anchors.tsv").read_text()),
        "source": "Explicit unchanged HTML supplied to --baseline-html",
    }}
    (audit / "baseline.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def destination_parts(value):
    return [item.strip() for item in value.split(";") if item.strip()]


def split_destination(value):
    path, _, anchor = value.partition("#")
    if not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ValueError("destination must be a repository-relative path")
    return path, anchor


def heading_anchor(text):
    """Documenter's slug for a bounded heading: plain text, code, or an @id.

    Other inline markup is deliberately rejected instead of guessing its rendered
    text. The slug filter follows Documenter.utilities.slugify (letters,
    punctuation, decimal digits; whitespace and ampersands become hyphens/and).
    """
    heading = HEADING.fullmatch(text.strip())
    if not heading:
        raise ValueError("not a single Markdown heading")
    title = heading[2]
    explicit = re.fullmatch(r"\[([^\]]+)\]\(@id\s+([^\s)]+)\)", title)
    if explicit:
        title = explicit[2]
    else:
        literals = []

        def code(match):
            if match[1] is None:
                raise ValueError("unsupported formatted heading; cannot prove its fragment ownership")
            literals.append(match[2])
            return f"\0{len(literals) - 1}\0"

        title = INLINE_LITERAL.sub(code, title)
        if (any(mark in title for mark in ("`", "*", "[", "]", "<", ">", "$", "\\", "@id")) or
                re.search(r"(?<!\w)_(?!\s)|(?<!\s)_(?!\w)|&[A-Za-z#0-9]+;", title)):
            raise ValueError("unsupported formatted heading; cannot prove its fragment ownership")
        for index, value in enumerate(literals):
            title = title.replace(f"\0{index}\0", value)
    slug = re.sub(r"\s+", "-", title).replace("&", "-and-")
    slug = "".join(char for char in slug if unicodedata.category(char)[0] in {"L", "P"} or char.isdecimal())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        raise ValueError("heading has no nonempty fragment")
    return slug, bool(explicit)


def heading_page(block):
    return page_key(PurePosixPath(block.path).relative_to("docs/src").with_suffix(".html"))


def section_text(text, anchor):
    if not anchor:
        return text
    lines = text.splitlines()
    start = None
    level = 0
    opened = None
    for i, line in enumerate(lines):
        fence = FENCE.match(line)
        if opened is not None:
            if fence and fence[1][0] == opened[0] and len(fence[1]) >= len(opened) and not fence[2].strip():
                opened = None
            continue
        if fence:
            opened = fence[1]
            continue
        heading = HEADING.match(line)
        if heading:
            explicit = re.search(r"\(@id\s+([^\s)]+)\)", heading[2])
            title = re.sub(r"\[([^\]]+)\]\(@id[^)]*\)", r"\1", heading[2]).replace("`", "")
            slug = re.sub(r"\s+", "-", title)
            if start is not None and len(heading[1]) <= level:
                return "\n".join(lines[start:i])
            if (explicit and explicit[1] == anchor) or slug == anchor:
                start, level = i, len(heading[1])
    if start is None:
        raise ValueError(f"missing destination heading #{anchor}")
    return "\n".join(lines[start:])


def source_citations(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "numerics.md" not in line:
            continue
        tail = " ".join(re.sub(r"^\s*#\s*", "", s) for s in lines[i:i + 3]).split("numerics.md", 1)[1]
        match = re.match(r'[\s,(]*"([^\"]+)"', tail)
        yield i + 1, " ".join(match[1].split()) if match else ""


def heading_titles(text):
    return {re.sub(r"\[([^\]]+)\]\(@id[^)]*\)", r"\1", match[2]).replace("`", "")
            for line in text.splitlines() if (match := HEADING.match(line))}


def protected_tokens(block):
    text = block.norm
    tokens = Counter()
    for match in INLINE_LITERAL.finditer(text):
        tokens["literal:" + match[0]] += 1
    for pattern, prefix in ((r"(?<!\w)#\d+", "issue:"),
                            (r"(?<![\w.])(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?!\w)", "number:"),
                            (r"https?://[^\s)]+", "url:"),
                            (r"\[[^\]]*\]\(([^)]+)\)", "link:")):
        for match in re.finditer(pattern, text):
            tokens[prefix + (match[1] if prefix == "link:" else match[0])] += 1
    if block.kind in {"code", "math"}:
        tokens[block.kind + ":" + digest(text)] += 1
    return tokens


def example_tokens(block):
    if block.kind not in {"code", "admonition"}:
        return Counter()
    return Counter(line.strip() for line in block.text.splitlines()
                   if "@example" in line or "jldoctest" in line or "@assert" in line)


def upstream_records(rows, root=ROOT):
    """Read the reviewed receipt; Git checks below verify its content provenance.

    The merged-PR status and reviewer identity are review attestations, not facts
    an offline Git repository can authenticate. They never replace blob checks.
    """
    required = {row["block_id"] for row in rows if row.get("action") == "upstream"}
    path = root / "docs/audit/docs-tracks-1019/upstream.json"
    if not required and not path.exists():
        return {}
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != 2 or not isinstance(receipt.get("changes"), list):
        raise ValueError("upstream receipt requires schema 2 and a changes list")
    records = {}
    for record in receipt["changes"]:
        if not isinstance(record, dict) or not isinstance(record.get("block_id"), str):
            raise ValueError("upstream receipt has no block_id")
        identifier = record["block_id"]
        if identifier in records:
            raise ValueError(f"duplicate upstream receipt: {identifier}")
        records[identifier] = record
    if set(records) != required:
        raise ValueError("upstream receipt IDs must exactly match upstream ledger rows")
    return records


def upstream_hashes(record, name):
    """Accept one scalar or one ordered list, never both spellings."""
    scalar, plural = name + "_sha", name + "_shas"
    if (scalar in record) == (plural in record):
        raise ValueError(f"upstream requires exactly one of {scalar} and {plural}")
    hashes = [record[scalar]] if scalar in record else record[plural]
    if (not isinstance(hashes, list) or not hashes or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)):
        raise ValueError(f"upstream requires nonempty exact {name} hashes")
    return hashes


def sequence_starts(values, sequence):
    return [i for i in range(len(values) - len(sequence) + 1)
            if values[i:i + len(sequence)] == sequence]


def validate_upstream(old, row, record, root=ROOT):
    """Bind one original block to an exact sequence of reviewed upstream diffs.

    Each event replaces a contiguous span in its original file. Hashes must join
    exactly; unrelated intervening commits or file edits need not be adjacent.
    The final hash remains bound to the ledger's current destination.
    """
    if record is None:
        raise ValueError("missing valid upstream receipt")
    expected = dict(block_id=old.id, baseline_sha=old.sha, dest=row["dest"],
                    dest_sha=row["dest_sha"], reviewer=row["reviewer"])
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("upstream receipt does not match original hash, destination, or reviewer")
    destinations, hashes = destination_parts(row["dest"]), destination_parts(row["dest_sha"])
    if not destinations or len(destinations) != len(hashes) or len(set(destinations)) != 1:
        raise ValueError("upstream requires one repeated destination scope for its ordered span")
    path, anchor = split_destination(destinations[0])
    if path != old.path or hashes == [old.sha]:
        raise ValueError("upstream requires a changed block in its original file")
    reviewer, _, revision = row["reviewer"].partition("@")
    if not reviewer or reviewer == "mechanical" or not revision:
        raise ValueError("upstream requires independent review identity@revision")
    events = record.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("upstream requires a nonempty events sequence")
    source = [old]
    for index, event in enumerate(events, 1):
        try:
            source = validate_upstream_event(source, event, root)
        except (ValueError, subprocess.CalledProcessError) as error:
            raise ValueError(f"upstream event {index}: {error}") from error
    if [block.sha for block in source] != hashes:
        raise ValueError("upstream final replacement does not match the destination hash")
    if events[-1]["reviewer"] != row["reviewer"]:
        raise ValueError("upstream final event reviewer does not match the destination review")
    current = list(blocks(section_text(read_current(path, root), anchor), path))
    starts = sequence_starts([block.sha for block in current], hashes)
    if len(starts) != 1:
        raise ValueError("no block with expected hash sequence uniquely contiguous at the current destination")
    return current[starts[0]:starts[0] + len(hashes)]


def validate_upstream_event(old, record, root=ROOT):
    """Verify one complete ordered transition using actual Git line changes."""
    if not isinstance(record, dict):
        raise ValueError("upstream event must be an object")
    source_hashes = upstream_hashes(record, "source")
    replacement_hashes = upstream_hashes(record, "replacement")
    if source_hashes != [block.sha for block in old]:
        raise ValueError("upstream source hash breaks event continuity")
    if replacement_hashes == source_hashes:
        raise ValueError("upstream event requires a different exact replacement hash sequence")
    reviewer = record.get("reviewer")
    if not isinstance(reviewer, str):
        raise ValueError("upstream event requires independent review identity@revision")
    identity, _, revision = reviewer.partition("@")
    if not identity or identity == "mechanical" or not revision:
        raise ValueError("upstream event requires independent review identity@revision")
    number = record.get("pull_request")
    if (type(number) is not int or number <= 0 or record.get("merged") is not True or
            record.get("pull_request_url") != f"https://github.com/mlakolar/ProximalBase.jl/pull/{number}"):
        raise ValueError("upstream requires a reviewed merged pull request in this repository")
    for key in ("before_commit", "after_commit", "before_blob", "after_blob"):
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", record[key]):
            raise ValueError(f"upstream requires an exact Git object ID: {key}")
    before, after = record["before_commit"], record["after_commit"]
    # Read raw commit headers: HEAD^ / revision walking stops at shallow roots.
    git("cat-file", "commit", before, root=root)
    header = git("cat-file", "commit", after, root=root).partition("\n\n")[0]
    parents = re.findall(r"^parent ([0-9a-f]{40})$", header, re.M)
    if not parents or parents[0] != before:
        raise ValueError("upstream before_commit is not after_commit's first parent")
    path = old[0].path
    texts = []
    for commit, key in ((before, "before_blob"), (after, "after_blob")):
        actual = git("rev-parse", "--verify", f"{commit}:{path}", root=root).strip()
        if actual != record[key]:
            raise ValueError(f"upstream file blob mismatch: {key}")
        texts.append(git("cat-file", "blob", actual, root=root))
    if record["before_blob"] == record["after_blob"]:
        raise ValueError("upstream file did not change")
    before_blocks, after_blocks = (list(blocks(text, path)) for text in texts)
    before_hashes, after_hashes = ([block.sha for block in page] for page in (before_blocks, after_blocks))
    old_starts, new_starts = sequence_starts(before_hashes, source_hashes), sequence_starts(after_hashes, replacement_hashes)
    if len(old_starts) != 1 or len(new_starts) != 1:
        raise ValueError("upstream diff must contain unique original and replacement blocks")
    originals = before_blocks[old_starts[0]:old_starts[0] + len(source_hashes)]
    replacements = after_blocks[new_starts[0]:new_starts[0] + len(replacement_hashes)]
    if [block.kind for block in originals] != [block.kind for block in old]:
        raise ValueError("upstream source block kinds changed outside the recorded sequence")
    if sequence_starts(after_hashes, source_hashes) or sequence_starts(before_hashes, replacement_hashes):
        raise ValueError("upstream diff must remove the original and introduce the replacement")

    raw_lines = [text.splitlines(keepends=True) for text in texts]
    offsets, character_spans = [], []
    for text, lines, page in zip(texts, raw_lines, (before_blocks, after_blocks)):
        positions = [0]
        for line in lines:
            positions.append(positions[-1] + len(line))
        offsets.append(positions)
        located, cursor = {}, 0
        for block in page:
            start = text.index(block.text, max(cursor, positions[block.line - 1]))
            cursor = start + len(block.text)
            located[id(block)] = (start, cursor)
        character_spans.append(located)

    def span(block, side):
        start, end = character_spans[side][id(block)]
        return texts[side].count("\n", 0, start), texts[side].count("\n", 0, end - 1) + 1

    def touches(span, start, end):
        first, last = span
        # A pure insertion/deletion has an empty range on one side.
        return first <= start <= last if start == end else max(first, start) < min(last, end)

    old_span = (span(originals[0], 0)[0], span(originals[-1], 0)[1])
    new_span = (span(replacements[0], 1)[0], span(replacements[-1], 1)[1])
    changes = []
    for tag, i, j, k, l in SequenceMatcher(None, texts[0].splitlines(), texts[1].splitlines(), autojunk=False).get_opcodes():
        # Adjacent edited table rows remain separate when their unchanged keys
        # prove a unique, ordered one-to-one pairing. Splits/reorders do not fit.
        keys = []
        for side, start, end in ((0, i, j), (1, k, l)):
            matches = [re.fullmatch(r"\s*\|([^|]+)\|.*\|\s*", line.rstrip("\r\n"))
                       for line in raw_lines[side][start:end]]
            keys.append([match[1].strip() for match in matches] if all(matches) else [])
        if (tag == "replace" and j - i == l - k > 1 and keys[0] == keys[1] and
                len(keys[0]) == j - i and all(keys[0]) and len(set(keys[0])) == len(keys[0])):
            changes.extend((tag, i + n, i + n + 1, k + n, k + n + 1) for n in range(j - i))
        else:
            changes.append((tag, i, j, k, l))
    owned = [(i, j, k, l) for tag, i, j, k, l in changes if tag != "equal" and touches(old_span, i, j)]
    if not any(touches(new_span, k, l) for i, j, k, l in owned):
        raise ValueError("upstream original and replacement do not share a changed diff hunk")
    # An equal hash elsewhere does not make a newly introduced copy unchanged.
    # Pair equal blocks by position, corroborated by exact character mapping
    # inside the existing line hunks. An unchanged sentence can share a changed
    # line with its neighbor; a copy elsewhere still supplies no proof.
    equal_characters = []
    for tag, i, j, k, l in changes:
        if tag not in {"equal", "replace"}:
            continue
        before_start, after_start = offsets[0][i], offsets[1][k]
        a, b = texts[0][before_start:offsets[0][j]], texts[1][after_start:offsets[1][l]]
        matches = [(0, 0, len(a))] if a == b else SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
        for bf, af, length in matches:
            if not length:
                continue
            bf, af = before_start + bf, after_start + af
            if equal_characters and equal_characters[-1][0] + equal_characters[-1][2] == bf and equal_characters[-1][1] + equal_characters[-1][2] == af:
                previous = equal_characters.pop()
                equal_characters.append((previous[0], previous[1], previous[2] + length))
            else:
                equal_characters.append((bf, af, length))
    unchanged = {}
    for tag, i, j, k, l in SequenceMatcher(None, before_hashes, after_hashes, autojunk=False).get_opcodes():
        if tag != "equal":
            continue
        for before_index, after_index in zip(range(i, j), range(k, l)):
            bf, bl = character_spans[0][id(before_blocks[before_index])]
            af, al = character_spans[1][id(after_blocks[after_index])]
            if any(i <= bf and bl <= i + length and af == k + bf - i and al == k + bl - i
                   for i, k, length in equal_characters):
                unchanged[after_index] = before_index
    # Project owned changes and retained source members, excluding only proven
    # unchanged neighbors. The API heading and @docs edit remain distinct.
    required = [index for index, block in enumerate(after_blocks)
                if (index in unchanged and old_starts[0] <= unchanged[index] < old_starts[0] + len(source_hashes)) or
                (index not in unchanged and
                 any(touches(span(block, 1), k, l) for i, j, k, l in owned))]
    if (not required or required != list(range(required[0], required[-1] + 1)) or
            [after_hashes[index] for index in required] != replacement_hashes):
        raise ValueError("upstream replacement must cover the complete ordered changed span")
    return replacements


def coverage(base_blocks, rows, root=ROOT):
    """Return errors and validated baseline-to-destination mappings.

    Approved transformations are tied to replacement hashes, never an unchecked
    exemption flag. Their meaning still requires the recorded independent review.
    """
    errors, mappings = [], {}
    try:
        upstream = upstream_records(rows, root)
    except (FileNotFoundError, ValueError) as error:
        errors.append(f"upstream evidence: {error}")
        upstream = {}
    by_id = {b.id: b for b in base_blocks}
    row_counts = Counter(row["block_id"] for row in rows)
    for identifier in sorted(set(by_id) - set(row_counts)):
        errors.append(f"missing ledger row: {identifier}")
    for identifier, count in row_counts.items():
        if identifier not in by_id:
            errors.append(f"unknown baseline block: {identifier}")
        if count != 1:
            errors.append(f"duplicate ledger row: {identifier}")
    cached, candidates_by_dest, counts_by_path = {}, {}, {}
    used = Counter()
    for row in rows:
        old = by_id.get(row["block_id"])
        if old is None:
            continue
        action = row.get("action", "")
        if row.get("sha") != old.sha or row.get("kind") != old.kind:
            errors.append(f"baseline hash/kind mismatch: {old.id}")
            continue
        if action not in ACTIONS:
            errors.append(f"unknown action {action!r}: {old.id}")
            continue
        reviewer = row.get("reviewer", "")
        if not reviewer or (action in REVIEWED | {"upstream"} and (reviewer == "mechanical" or "@" not in reviewer)):
            errors.append(f"missing independent review identity@revision for {action}: {old.id}")
            continue
        destinations = destination_parts(row.get("dest", ""))
        hashes = destination_parts(row.get("dest_sha", ""))
        if not destinations or len(destinations) != len(hashes):
            errors.append(f"destination/hash count mismatch: {old.id}")
            continue
        if action == "upstream":
            try:
                upstream_replacements = validate_upstream(old, row, upstream.get(old.id), root)
            except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
                errors.append(f"{old.id}: {error}")
                continue
        replacements = []
        for index, (dest, expected) in enumerate(zip(destinations, hashes)):
            try:
                path, anchor = split_destination(dest)
                if path not in cached:
                    cached[path] = read_current(path, root)
                    page_blocks = list(blocks(cached[path], path))
                    candidates_by_dest[path] = page_blocks
                    counts_by_path[path] = Counter(b.sha for b in page_blocks)
                if dest not in candidates_by_dest:
                    candidates_by_dest[dest] = list(blocks(section_text(cached[path], anchor), path))
                candidates = candidates_by_dest[dest]
                matches = [upstream_replacements[index]] if action == "upstream" else [b for b in candidates if b.sha == expected]
                if not matches:
                    raise ValueError(f"no block with expected hash {expected[:12]} at {dest}")
                if action in {"stay", "move", "archive"} and expected != old.sha:
                    raise ValueError(f"{action} must preserve the original normalized hash")
                if action == "stay" and path != old.path:
                    raise ValueError("stay must retain its source path")
                if action == "archive":
                    if not path.startswith("docs/design/"):
                        raise ValueError("archive must have a docs/design destination")
                    origin = canonical_links(read_current(old.path, root), old.path)
                    if not re.search(r"\]\(" + re.escape(path) + r"(?:[#)]|\s)", origin):
                        raise ValueError("archive requires a forwarding link at its source")
                if action in {"stay", "move", "archive", "upstream"}:
                    key = (path, expected)
                    used[key] += 1
                    # Count across the whole page, so two anchor scopes cannot
                    # accidentally reuse one occurrence for two original blocks.
                    available = counts_by_path[path][expected]
                    if used[key] > available:
                        raise ValueError("destination occurrence reused without a reviewed dedupe action")
                replacements.append(matches[0])
            except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
                errors.append(f"{old.id}: {error}")
        if len(replacements) == len(destinations):
            if action == "split":
                old_words = Counter(old.norm.split())
                new_words = Counter(" ".join(b.norm for b in replacements).split())
                if old_words - new_words:
                    errors.append(f"split loses original words; classify/review as reword: {old.id}")
                    continue
            mappings[old.id] = (action, replacements)
    return errors, mappings


def accounting(base_blocks, mappings, kind):
    errors = []
    for old in base_blocks:
        if old.id not in mappings:
            errors.append(f"unaccounted baseline block: {old.id}")
            continue
        action, replacements = mappings[old.id]
        # coverage has independently checked the exact reviewed upstream Git
        # change. These are inherited changes, not preservation-equivalent edits.
        if action == "upstream":
            continue
        # Reviewed replacement hashes are the acceptance evidence for semantic
        # transformations. Every check uses this same action policy.
        if action in REVIEWED:
            continue
        if kind == "tokens":
            before, after = protected_tokens(old), sum((protected_tokens(b) for b in replacements), Counter())
        elif kind == "sentences":
            before = Counter(sentences(old.norm))
            after = sum((Counter(sentences(b.norm)) for b in replacements), Counter())
        elif kind == "examples":
            before, after = example_tokens(old), sum((example_tokens(b) for b in replacements), Counter())
        else:
            raise ValueError(kind)
        if before - after:
            errors.append(f"lost {kind}: {old.id}")
    return errors


def anchor_check(rows, build, *, retirements=(), base_blocks=(), mappings=None, report=None):
    pages = html_anchors(build)
    errors = []
    if not pages:
        return ["no built HTML; run docs/make.jl first"]
    frozen = {(row["page"], row["fragment"]) for row in rows}
    originals = {block.id: block for block in base_blocks}
    approved = {}
    retired_blocks = set()
    if not isinstance(retirements, (list, tuple)):
        errors.append("anchor_retirements must be an array")
        retirements = ()
    for record in retirements:
        try:
            if not isinstance(record, dict):
                raise ValueError("anchor retirement must be an object")
            page, fragments, target = record.get("page"), record.get("fragments"), record.get("target")
            if not isinstance(page, str) or not isinstance(target, str) or not target:
                raise ValueError("anchor retirement requires a page and nonempty target")
            if not isinstance(fragments, list) or not fragments or not all(isinstance(item, str) and item for item in fragments):
                raise ValueError("anchor retirement requires named fragments")
            if len(fragments) != len(set(fragments)):
                raise ValueError("duplicate anchor retirement fragment")
            pairs = {(page, fragment) for fragment in fragments}
            if not pairs <= frozen:
                raise ValueError("anchor retirement is not in the frozen snapshot")
            identifier = record.get("block_id")
            if not isinstance(identifier, str):
                raise ValueError("anchor retirement original block/hash mismatch")
            if pairs & approved.keys() or identifier in retired_blocks:
                raise ValueError("duplicate anchor retirement record")
            old = originals.get(identifier)
            if old is None or record.get("baseline_sha") != old.sha:
                raise ValueError("anchor retirement original block/hash mismatch")
            if old.kind != "heading" or "@id" in old.text or heading_page(old) != page:
                raise ValueError("anchor retirement requires the original automatic heading, without @id")
            slug, _ = heading_anchor(old.text)
            owned = {slug, slug + "-1"}  # First heading's primary and legacy IDs.
            if not set(fragments) <= owned:
                raise ValueError("anchor retirement fragments do not belong to the original heading")
            for other in base_blocks:
                if other.kind != "heading" or other.id == old.id or other.path != old.path:
                    continue
                other_slug, _ = heading_anchor(other.text)
                if owned & {other_slug, other_slug + "-1"}:
                    raise ValueError("ambiguous duplicate or numbered heading fragment ownership")
            reviewer = record.get("reviewer", "")
            if not isinstance(reviewer, str) or "@" not in reviewer:
                raise ValueError("anchor retirement requires independent review identity@revision")
            identity, _, revision = reviewer.partition("@")
            if not identity or identity == "mechanical" or not revision:
                raise ValueError("anchor retirement requires independent review identity@revision")
            if not isinstance(record.get("evidence"), str) or not record["evidence"].strip():
                raise ValueError("anchor retirement requires the authored-inbound-reference audit evidence")
            action, replacements = (mappings or {}).get(old.id, (None, []))
            if (action != "upstream" or len(replacements) != 1 or replacements[0].kind != "heading" or
                    replacements[0].path != old.path or heading_anchor(replacements[0].text)[0] != target):
                raise ValueError("anchor retirement requires a validated upstream mapping to the exact successor heading")
            if target not in pages.get(page, set()):
                raise ValueError(f"missing anchor retirement target on its original page: {page}#{target}")
            approved.update({pair: target for pair in pairs})
            retired_blocks.add(identifier)
        except ValueError as error:
            errors.append(str(error))
    for row in rows:
        page, fragment = row["page"], row["fragment"]
        if page not in pages:
            errors.append(f"missing baseline page: {page or '/'}")
        elif fragment and fragment not in pages[page] and (page, fragment) not in approved:
            errors.append(f"missing original page+fragment: {page or '/'}#{fragment}")
    if report is not None:
        present = {(page, fragment) for page, fragment in frozen if fragment and fragment in pages.get(page, set())}
        retired = approved.keys() - present
        report.update(baseline_fragment_count=sum(bool(fragment) for page, fragment in frozen),
                      present_fragment_count=len(present), retired_fragment_count=len(retired),
                      approved_retirement_count=len(approved),
                      approved_retirements=[dict(page=page, fragment=fragment, target=approved[page, fragment])
                                            for page, fragment in sorted(approved)])
    return errors


def citations_check(root=ROOT, baseline=BASELINE, scope_base=None):
    errors, evidence = [], []
    numerics = read_current("docs/src/numerics.md", root)
    titles = heading_titles(numerics)
    paths = git("ls-tree", "-r", "--name-only", baseline, "src", "test", root=root).splitlines()
    for path in paths:
        if not path.endswith(".jl"):
            continue
        for line, heading in source_citations(git("show", baseline + ":" + path, root=root)):
            evidence.append(dict(path=path, line=line, heading=heading))
            if heading and heading not in titles:
                errors.append(f"{path}:{line}: missing cited numerics.md heading {heading!r}")
    current_paths = git("ls-files", "src", "test", root=root).splitlines()
    for path in current_paths:
        if path.endswith(".jl"):
            for line, heading in source_citations(read_current(path, root)):
                if heading and heading not in titles:
                    errors.append(f"{path}:{line}: missing currently cited numerics.md heading {heading!r}")
    if scope_base:
        changed = git("diff", "--name-only", scope_base, "--", "src", "test", root=root).splitlines()
        errors.extend(f"documentation PR changes protected source/test path: {path}" for path in changed)
    return errors, evidence


def examples_check(base_blocks, mappings, metadata, root=ROOT):
    errors = accounting(base_blocks, mappings, "examples")
    for path, expected in metadata["example_scripts"].items():
        try:
            if digest(read_current(path, root)) != expected:
                errors.append(f"baseline example script changed: {path}")
        except FileNotFoundError:
            errors.append(f"baseline example script missing: {path}")
    return errors


def apply_moves(manifest, root=ROOT, audit=AUDIT, baseline=BASELINE):
    """Apply explicit, verbatim source spans to ledger destinations.

    The input is a JSON list of source/text/destination/anchor records. Every
    source span must be unique in the frozen file; destination proof is left to
    coverage(), which must pass before accepting the updated ledger.
    """
    rows = read_tsv(audit / "ledger.tsv")
    by_id = {b.id: b for b in baseline_blocks(root, baseline)}
    source_cache, touched = {}, set()
    for move in manifest:
        source = move["source"]
        if source not in source_cache:
            source_cache[source] = git("show", baseline + ":" + source, root=root)
        original = source_cache[source]
        text = move["text"]
        if original.count(text) != 1:
            raise ValueError(f"move span is not unique in frozen {source}")
        start = original.index(text)
        start_line = original[:start].count("\n") + 1
        end_line = start_line + text.count("\n")
        found = 0
        for row in rows:
            block = by_id[row["block_id"]]
            if block.path != source or not start_line <= block.line <= end_line:
                continue
            if block.norm not in normalized(text, source, block.kind):
                continue
            if block.id in touched:
                raise ValueError(f"overlapping move spans: {block.id}")
            row.update(action="move", dest=move["destination"] + ("#" + move["anchor"] if move.get("anchor") else ""),
                       dest_sha=block.sha, reviewer="mechanical")
            touched.add(block.id)
            found += 1
        if not found:
            raise ValueError(f"move span covers no whole baseline block: {source}:{start_line}")
    errors, _ = coverage(list(by_id.values()), rows, root)
    if errors:
        raise ValueError("move ledger did not validate:\n" + "\n".join(errors))
    write_tsv(audit / "ledger.tsv", FIELDS, rows, baseline)
    return len(touched)
