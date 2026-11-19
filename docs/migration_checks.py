"""Small, deterministic preservation checks for documentation migration #1019.

This is a provenance check, not a Markdown renderer or a semantic equivalence
proof. Documenter owns rendering; independent reviewers own approved rewrites.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit
import csv
import io
import json
import posixpath
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "docs/audit/docs-tracks-1019"
BASELINE = "7ae86e2093060eb124211e2f21f043c6379f97ae"
FIELDS = ("block_id", "sha", "kind", "action", "dest", "dest_sha", "clause_ids", "reviewer")
ACTIONS = {"stay", "move", "split", "reword", "dedupe", "supersede", "archive"}
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
        if url.scheme or url.netloc or target.startswith(("@", "#", "/")):
            return match[0]
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), url.path))
        return match[1] + urlunsplit(("", "", resolved, url.query, url.fragment)) + match[3]
    return LINK.sub(replace, text)


def normalized(text, path, kind="prose"):
    text = text.replace("\r\n", "\n").strip()
    if kind in {"code", "math"}:
        return text
    return " ".join(canonical_links(text, path).split())


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def sentences(text):
    """Split outside code, math and links; punctuation stays with its sentence.

    Abbreviations can yield extra units, but deterministic units are used on both
    sides. The checker does not claim linguistic sentence understanding.
    """
    mask = list(text)
    for match in PROTECTED.finditer(text):
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
        self.article = False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article" and attrs.get("id") == "documenter-page":
            self.article = True
        if self.article and "id" in attrs:
            self.ids.add(attrs["id"])

    def handle_endtag(self, tag):
        if tag == "article":
            self.article = False


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


def initial_ledger(base_blocks):
    return [dict(block_id=b.id, sha=b.sha, kind=b.kind, action="stay", dest=b.path,
                 dest_sha=b.sha, clause_ids="", reviewer="mechanical") for b in base_blocks]


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
    metadata = dict(schema=1, baseline=baseline, issue=1019, mode="report-only",
                    block_count=len(baseline_data), authored_pages=sum(p.startswith("docs/src/") for p in scope),
                    rendered_pages=len(pages), content_anchor_count=sum(map(len, pages.values())),
                    example_scripts=examples)
    (audit / "baseline.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def destination_parts(value):
    return [item.strip() for item in value.split(";") if item.strip()]


def split_destination(value):
    path, _, anchor = value.partition("#")
    if not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ValueError("destination must be a repository-relative path")
    return path, anchor


def section_text(text, anchor):
    if not anchor:
        return text
    lines = text.splitlines()
    start = None
    level = 0
    for i, line in enumerate(lines):
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
    for match in re.finditer(r"(`+)(.*?)\1", text, re.S):
        tokens["span:" + match[2]] += 1
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


def coverage(base_blocks, rows, root=ROOT):
    """Return errors and validated baseline-to-destination mappings.

    Approved transformations are tied to replacement hashes, never an unchecked
    exemption flag. Their meaning still requires the recorded independent review.
    """
    errors, mappings = [], {}
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
        if not reviewer or (action in REVIEWED and (reviewer == "mechanical" or "@" not in reviewer)):
            errors.append(f"missing independent review identity@revision for {action}: {old.id}")
            continue
        destinations = destination_parts(row.get("dest", ""))
        hashes = destination_parts(row.get("dest_sha", ""))
        if not destinations or len(destinations) != len(hashes):
            errors.append(f"destination/hash count mismatch: {old.id}")
            continue
        replacements = []
        for dest, expected in zip(destinations, hashes):
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
                matches = [b for b in candidates if b.sha == expected]
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
                if action in {"stay", "move", "archive"}:
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


def anchor_check(rows, build):
    pages = html_anchors(build)
    errors = []
    if not pages:
        return ["no built HTML; run docs/make.jl first"]
    for row in rows:
        page, fragment = row["page"], row["fragment"]
        if page not in pages:
            errors.append(f"missing baseline page: {page or '/'}")
        elif fragment and fragment not in pages[page]:
            errors.append(f"missing original page+fragment: {page or '/'}#{fragment}")
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
