"""Small, deterministic preservation checks for documentation migration #1019.

This is a provenance check, not a Markdown renderer or a semantic equivalence
proof. Documenter owns rendering; independent reviewers own approved rewrites.
"""
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit, urlunsplit
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
REFINEMENT_PATHS = {"docs/src/smooth.md", "docs/src/group_penalty.md", "docs/src/calculus.md"}
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


def sentence_children(block):
    """Stable child identities without renumbering any frozen provenance root."""
    if block.kind != "prose":
        return [block]
    values = list(sentences(block.text))
    if len(values) == 1:
        return [block]
    return [Block(f"{block.id}::sentence:{i}", block.path, block.kind, text,
                  block.line, block.heading) for i, text in enumerate(values, 1)]


def refinement_records(base_blocks):
    records = []
    for block in base_blocks:
        children = sentence_children(block)
        if block.path in REFINEMENT_PATHS and len(children) > 1:
            if normalized(" ".join(child.text for child in children), block.path) != block.norm:
                raise ValueError(f"sentence children do not reconstruct their parent: {block.id}")
            records.append(dict(block_id=block.id, sha=block.sha,
                                children=[dict(block_id=child.id, sha=child.sha) for child in children]))
    return records


def validated_refinements(base_blocks, root=ROOT):
    path = root / "docs/audit/docs-tracks-1019/refinements.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = refinement_records(base_blocks)
    if (not isinstance(data, dict) or data.get("schema") != 1 or data.get("baseline") != BASELINE or
            data.get("parents") != expected):
        raise ValueError("refinement map differs from the complete ordered frozen parent-to-sentence map")
    originals = {block.id: block for block in base_blocks}
    return {record["block_id"]: sentence_children(originals[record["block_id"]]) for record in expected}


def refinement_inventory(base_blocks, refinements):
    """Effective accounting units, with every child bound to its original root."""
    return [child for block in base_blocks for child in refinements.get(block.id, [block])]


def destination_blocks(text, path):
    """Offer sentence units alongside whole paragraphs for reviewed refinements.

    The frozen parser remains unchanged. Destination occurrence accounting below
    also checks overlapping text, so a paragraph and one of its children cannot
    be used twice to cover different roots.
    """
    for block in blocks(text, path):
        yield block
        if path in REFINEMENT_PATHS and block.kind == "prose":
            children = sentence_children(block)
            if len(children) > 1:
                yield from children


def write_tsv(path, fields, rows, baseline=BASELINE):
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write(f"# baseline\t{baseline}\n")
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_tsv(lines):
    return list(csv.DictReader((line for line in lines if not line.startswith("# baseline\t")), delimiter="\t"))


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return parse_tsv(stream)


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


class AdmonitionContent(HTMLParser):
    """Extract exact rendered admonition titles and distinguish self-permalinks."""
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.stack, self.admonitions, self.links = [], [], []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        inside = any(item[1].get("id") == "documenter-page" for item in self.stack)
        classes = attrs.get("class", "").split()
        node = None
        if inside and tag in {"div", "details"} and "admonition" in classes:
            node = dict(fragment=attrs.get("id", ""), classes=classes, title="")
            self.admonitions.append(node)
        owner = next((item[2] for item in reversed(self.stack) if item[2] is not None), None)
        if inside and tag == "a" and attrs.get("href"):
            self.links.append((attrs["href"], "admonition-anchor" in classes,
                               owner["fragment"] if owner else None))
        if tag not in self.VOID:
            self.stack.append((tag, attrs, node))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if not any("admonition-header" in item[1].get("class", "").split() for item in self.stack):
            return
        owner = next((item[2] for item in reversed(self.stack) if item[2] is not None), None)
        if owner is not None:
            owner["title"] += data


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


def anchor_snapshot(metadata, audit=AUDIT):
    """Load the complete frozen snapshot; runtime IDs use bounded identities.

    The old CI snapshot and runtime descriptors remain immutable audit evidence.
    They no longer select different acceptance rules for different Julia builds.
    """
    descriptor = metadata.get("anchor_snapshot", {})
    if descriptor.get("file") != "anchors.tsv":
        raise ValueError("a complete frozen anchors.tsv descriptor is required")
    path = audit / "anchors.tsv"
    content = path.read_text(encoding="utf-8")
    if not content.startswith(f"# baseline\t{metadata['baseline']}\n"):
        raise ValueError("anchor snapshot names a different baseline")
    if digest(content) != descriptor.get("sha256"):
        raise ValueError("anchor snapshot hash mismatch")
    rows = read_tsv(path)
    pairs = {(row["page"], row["fragment"]) for row in rows}
    pages = {row["page"] for row in rows if not row["fragment"]}
    anchors = sum(bool(row["fragment"]) for row in rows)
    if not rows or len(pairs) != len(rows):
        raise ValueError("anchor snapshot is empty or has duplicate rows")
    if pages != {row["page"] for row in rows} or len(pages) != metadata["rendered_pages"]:
        raise ValueError("anchor snapshot has incomplete baseline page records")
    if anchors != metadata["content_anchor_count"]:
        raise ValueError("anchor snapshot has an incomplete baseline anchor count")
    return rows


def runtime_anchor_identities(root=ROOT):
    """The three approved uncited runtime IDs, not a general hash exemption."""
    path = root / "docs/audit/docs-tracks-1019/anchor-identities.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        ("function_library", "note", "Which squared-loss object should I use?", "docs/src/function_library.md"),
        ("getting_started", "details", "Degenerate data and step-size estimates", "docs/src/getting_started.md"),
        ("api/iterates", "note", "Non-finite vectors", "src/symmetric_sparse_iterate.jl"),
    }
    records = data.get("identities") if isinstance(data, dict) else None
    if (not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(records, list) or len(records) != 3 or
            any(not isinstance(record, dict) for record in records) or
            {(r.get("page"), r.get("category"), r.get("title"), r.get("source")) for r in records} != expected):
        raise ValueError("runtime anchor identities must name exactly the three approved page/category/title/source tuples")
    audit = path.parent
    metadata = json.loads((audit / "baseline.json").read_text(encoding="utf-8"))
    local = {(row["page"], row["fragment"]) for row in anchor_snapshot(metadata, audit)}
    archived = audit / "anchors-ci.tsv"
    if digest(archived.read_text(encoding="utf-8")) != metadata["anchor_profiles"]["ci"]["sha256"]:
        raise ValueError("historical CI anchor snapshot hash mismatch")
    ci = {(row["page"], row["fragment"]) for row in read_tsv(archived)}
    differing = local ^ ci
    represented = {(record["page"], fragment) for record in records for fragment in record.get("fragments", [])}
    if differing != represented:
        raise ValueError("runtime identities must account for exactly the frozen local/CI fragment differences")
    return records


def authored_link_destinations(text):
    """Extract the Markdown/HTML destination forms used by authored sources."""
    targets = []
    destination = r"(?:<([^<>\n]+)>|([^\s)<>]+))"
    for pattern in (r"\]\(\s*" + destination,
                    r"(?m)^[ \t]*(?:>[ \t]*)*\[[^\]\n]+\]:\s*" + destination):
        targets.extend(left or right for left, right in re.findall(pattern, text))
    targets.extend(re.findall(r"<([A-Za-z][A-Za-z0-9+.-]*:[^<>\s]+)>", text))
    targets.extend("#" + value for value in re.findall(r"\(@ref\s+([^\s)]+)\)", text))

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "a" and (href := dict(attrs).get("href")):
                targets.append(href)

    # HTMLParser handles quoted/unquoted hrefs and HTML character references.
    Links(convert_charrefs=True).feed(text)
    return [unescape(target.strip()) for target in targets]


def check_runtime_identities(records, frozen, build, root):
    """Return exact baseline pairs accounted for by rendered content identity."""
    errors, accounted, identities = [], set(), []
    parsed = {}
    for path in build.rglob("*.html"):
        html = AdmonitionContent(path.read_text(encoding="utf-8"))
        html.relative_path = path.relative_to(build).as_posix()
        parsed[page_key(html.relative_path)] = html
    dynamic_fragments = set()
    for record in records:
        try:
            page, category, title, source = (record[key] for key in ("page", "category", "title", "source"))
            fragments = record["fragments"]
            if (not isinstance(fragments, list) or len(fragments) != 2 or len(set(fragments)) != 2 or
                    any(not isinstance(fragment, str) or not fragment for fragment in fragments)):
                raise ValueError("runtime anchor identity requires both exact historical fragments")
            pairs = {(page, fragment) for fragment in fragments} & frozen
            if len(pairs) != 1 or pairs & accounted:
                raise ValueError("runtime anchor identity does not own one distinct frozen fragment")
            source_text = read_current(source, root)
            pattern = r'^\s*(?:!!!|\?\?\?)\s+' + re.escape(category) + r'\s+"' + re.escape(title) + r'"\s*$'
            if len(re.findall(pattern, source_text, re.M)) != 1:
                raise ValueError(f"runtime anchor source identity must occur exactly once: {source} / {category} / {title}")
            rendered_category = {"note": "is-info", "details": "is-details"}[category]
            matches = [node for node in parsed.get(page, AdmonitionContent("")).admonitions
                       if rendered_category in node["classes"] and node["title"] == title]
            if len(matches) != 1 or not matches[0]["fragment"]:
                raise ValueError(f"runtime anchor rendered identity must occur exactly once: {page} / {category} / {title}")
            current = matches[0]["fragment"]
            dynamic_fragments.update([*fragments, current])
            accounted.update(pairs)
            identities.append(dict(page=page, category=category, title=title, fragment=current))
            # Protect current and historical dynamic fragment links. A generated
            # self-permalink is the sole permitted incoming link.
            targets = {(page, fragment) for fragment in [*fragments, current]}
            for from_page, html in parsed.items():
                for href, permalink, owner in html.links:
                    url = urlsplit(href)
                    if url.scheme or url.netloc or not url.fragment:
                        continue
                    target_path = posixpath.normpath(posixpath.join(posixpath.dirname(html.relative_path), unquote(url.path))) if url.path else html.relative_path
                    target_page = page_key(target_path.rstrip("/"))
                    if target_page.endswith("/index"):
                        target_page = target_page[:-6]
                    pair = (target_page, unquote(url.fragment))
                    if pair in targets and not (from_page == page and not url.path and permalink and owner == current and pair[1] == current):
                        errors.append(f"authored/non-self link to runtime-generated fragment: {from_page} -> {href}")
        except (KeyError, FileNotFoundError, ValueError) as error:
            errors.append(str(error))
    sources = {path for directory in ("docs/src", "docs/design", "src", "test", "examples")
               for path in (root / directory).rglob("*")
               if path.is_file() and path.suffix in {".md", ".jl", ".html"}}
    sources.update(root / filename for filename in ("README.md", "CONTRIBUTING.md", "AGENTS.md")
                   if (root / filename).is_file())
    for path in sources:
        text = path.read_text(encoding="utf-8")
        # Markdown destinations can be padded with whitespace or enclosed in
        # angle brackets. Reference definitions need checking even when their
        # owning README/design/source file is not rendered into the manual.
        for target in authored_link_destinations(text):
            if unquote(urlsplit(target).fragment) in dynamic_fragments:
                errors.append(f"authored link to runtime-generated fragment in {path.relative_to(root)}: {target}")
    return errors, accounted, identities


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
    metadata["anchor_snapshot"] = {
        "file": "anchors.tsv", "sha256": digest((audit / "anchors.tsv").read_text()),
        "source": "Explicit unchanged HTML supplied to --baseline-html",
    }
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


class CheckedMappings(dict):
    """Validated mappings plus exact accounting bases and bounded allowances."""
    def __init__(self):
        super().__init__()
        self.bases = {}
        self.allowances = {}
        self.refinements = {}


def receipt_records(root, filename, key, *, schema=1, commit=None):
    path = AUDIT.relative_to(ROOT) / filename
    if commit is None:
        if not (root / path).exists():
            return {}
        text = (root / path).read_text(encoding="utf-8")
    else:
        if not git("ls-tree", "--full-tree", commit, "--", str(path), root=root).strip():
            return {}
        text = git("show", f"{commit}:{path}", root=root)
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("schema") != schema or not isinstance(data.get(key), list):
        raise ValueError(f"{filename} requires schema {schema} and a {key} list")
    result = {}
    for record in data[key]:
        if not isinstance(record, dict) or not isinstance(record.get("block_id"), str):
            raise ValueError(f"{filename} record requires a block_id")
        identifier = record["block_id"]
        if identifier in result:
            raise ValueError(f"duplicate {filename} record: {identifier}")
        result[identifier] = record
    return result


def receipt_digest(record):
    return digest(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def validate_migration_basis(old, record, receipt, source):
    expected = dict(block_id=old.id, baseline_sha=old.sha,
                    upstream_sha=receipt_digest(receipt), source_shas=[block.sha for block in source])
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("migration basis does not match the exact reviewed upstream terminal span")
    if not isinstance(record.get("evidence"), str) or not record["evidence"].strip():
        raise ValueError("migration basis requires the historical review evidence")


def token_delta(before, after):
    return dict(removed=dict(sorted((before - after).items())),
                added=dict(sorted((after - before).items())))


def validate_allowance(old, row, basis, replacements, record):
    """Require a reviewed, current and fully used exact token/marker change."""
    if row["action"] not in {"reword", "split"}:
        raise ValueError("token allowance is unused by a reword or split action")
    expected = dict(block_id=old.id, baseline_sha=old.sha, action=row["action"],
                    source_shas=[block.sha for block in basis], dest=row["dest"],
                    dest_sha=row["dest_sha"], reviewer=row["reviewer"])
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("token allowance is stale: original, source, destination, action or reviewer differs")
    if not isinstance(record.get("evidence"), str) or not record["evidence"].strip():
        raise ValueError("token allowance requires independent review evidence")
    actual = {}
    for name, counter in (("tokens", protected_tokens), ("examples", example_tokens)):
        before = sum((counter(block) for block in basis), Counter())
        after = sum((counter(block) for block in replacements), Counter())
        delta = token_delta(before, after)
        if delta["removed"]:
            actual[name] = delta
    if not actual:
        raise ValueError("token allowance is unused: no protected token or marker was removed")
    if record.get("changes") != actual:
        raise ValueError("token allowance does not exactly match every removed and added token/marker")
    return actual


def located_destination_blocks(text, path):
    located, cursor = [], 0
    for block in blocks(text, path):
        start = text.index(block.text, cursor)
        end = start + len(block.text)
        cursor = end
        located.append((block, start, end))
        if path in REFINEMENT_PATHS and block.kind == "prose":
            children = sentence_children(block)
            if len(children) > 1:
                child_cursor = start
                for child in children:
                    child_start = text.index(child.text, child_cursor, end)
                    child_end = child_start + len(child.text)
                    located.append((child, child_start, child_end))
                    child_cursor = child_end
    return located


def upstream_records(rows, root=ROOT, migration_bases=None):
    """Read the reviewed receipt; Git checks below verify its content provenance.

    The merged-PR status and reviewer identity are review attestations, not facts
    an offline Git repository can authenticate. They never replace blob checks.
    """
    required = {row["block_id"] for row in rows if row.get("action") == "upstream"}
    required.update(migration_bases or {})
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


def tracked_ledger_matches(old, commit, root, **expected):
    path = "docs/audit/docs-tracks-1019/ledger.tsv"
    if not git("ls-tree", "--full-tree", commit, "--", path, root=root).strip():
        return False
    ledger = git("show", f"{commit}:{path}", root=root)
    rows = [row for row in parse_tsv(ledger.splitlines(keepends=True)) if row.get("block_id") == old.id]
    return (len(rows) == 1 and rows[0].get("sha") == old.sha and rows[0].get("kind") == old.kind and
            all(rows[0].get(key) == value for key, value in expected.items()))


def tracked_move_matches(old, commit, destination, hashes, root):
    return tracked_ledger_matches(old, commit, root, action="move", dest_sha=";".join(hashes),
                                  dest=";".join([destination] * len(hashes)))


def validate_upstream_prior_move(old, destination, first_event, root, source_hashes):
    """Find the exact tracked move of the source span in the event's parent."""
    before = first_event.get("before_commit") if isinstance(first_event, dict) else None
    if not isinstance(before, str) or not re.fullmatch(r"[0-9a-f]{40}", before):
        raise ValueError("upstream requires an exact Git object ID: before_commit")
    if not tracked_move_matches(old, before, destination, source_hashes, root):
        raise ValueError("upstream outside the original file requires a matching move in the first event's parent ledger")
    return before


def git_span_count(commit, path, anchor, hashes, root):
    if not git("ls-tree", "--full-tree", commit, "--", path, root=root).strip():
        return 0
    text = git("show", f"{commit}:{path}", root=root)
    hashes_at_scope = [block.sha for block in blocks(section_text(text, anchor), path)]
    return len(sequence_starts(hashes_at_scope, hashes))


def validate_reviewed_move(old, destination, move, receipt, source, root):
    """Verify a reviewed span's actual relocation across the landed move."""
    if not isinstance(move, dict) or set(move) != {"before_commit", "after_commit"}:
        raise ValueError("reviewed move requires exact before_commit and after_commit")
    before, after = move["before_commit"], move["after_commit"]
    validate_first_parent(before, after, "reviewed move", root)
    for commit in (before, after):
        historical = receipt_records(root, "upstream.json", "changes", schema=2, commit=commit).get(old.id)
        if receipt_digest(historical) != receipt_digest(receipt):
            raise ValueError("reviewed move requires the unchanged historical upstream receipt")
    if not tracked_ledger_matches(old, before, root, action="upstream", dest=receipt["dest"],
                                  dest_sha=receipt["dest_sha"], reviewer=receipt["reviewer"]):
        raise ValueError("reviewed move requires the tracked historical upstream row")
    basis = receipt_records(root, "migration-bases.json", "bases", commit=after).get(old.id, {})
    validate_migration_basis(old, basis, receipt, source)
    hashes = [block.sha for block in source]
    if not tracked_move_matches(old, after, destination, hashes, root):
        raise ValueError("reviewed move requires the exact tracked terminal-span move")
    original_path, original_anchor = split_destination(destination_parts(receipt["dest"])[0])
    path, anchor = split_destination(destination)
    if (git_span_count(before, original_path, original_anchor, hashes, root) != 1 or
            git_span_count(before, original_path, "", hashes, root) != 1 or
            git_span_count(after, original_path, "", hashes, root) != 0 or
            git_span_count(before, path, "", hashes, root) != 0 or
            git_span_count(after, path, anchor, hashes, root) != 1 or
            git_span_count(after, path, "", hashes, root) != 1):
        raise ValueError("reviewed move must remove and relocate the exact unique terminal span")


def validate_upstream_moved_basis(old, destination, record, move_after, root):
    """Verify the prior original-file events and the reviewed span they supplied to a move."""
    prefix = record["events"][:move_after]
    first_event = record["events"][move_after]
    if not isinstance(first_event, dict):
        raise ValueError("upstream event must be an object")
    source_hashes = upstream_hashes(first_event, "source")
    before = validate_upstream_prior_move(old, destination, first_event, root, source_hashes)
    receipt = receipt_records(root, "upstream.json", "changes", schema=2, commit=before).get(old.id)
    basis = receipt_records(root, "migration-bases.json", "bases", commit=before).get(old.id)
    if receipt is None or basis is None:
        raise ValueError("post-move upstream requires a historical receipt and migration basis in the parent")
    if receipt_digest(receipt.get("events")) != receipt_digest(prefix):
        raise ValueError("post-move upstream prefix differs from the historical receipt in the parent")
    if any(not isinstance(receipt.get(key), str) for key in ("dest", "dest_sha", "reviewer")):
        raise ValueError("historical upstream receipt requires a destination, hashes and reviewer")
    source = validate_upstream(old, receipt, receipt, root, current=False)
    validate_migration_basis(old, basis, receipt, source)
    if [block.sha for block in source] != source_hashes:
        raise ValueError("post-move upstream source must match the historical terminal span")
    if "reviewed_move" in record:
        validate_reviewed_move(old, destination, record["reviewed_move"], receipt, source, root)
    return source


def validate_upstream(old, row, record, root=ROOT, *, current=True):
    """Bind one original block to an exact sequence of reviewed upstream diffs.

    Each event replaces a contiguous span in one file. The tracked parent ledger
    can establish a prior exact move to that file. A historical receipt for a
    moved frozen block uses the same proof and returns the Git-verified span.
    Hashes must join exactly across events.
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
    if hashes == [old.sha]:
        raise ValueError("upstream requires a changed block")
    reviewer, _, revision = row["reviewer"].partition("@")
    if not reviewer or reviewer == "mechanical" or not revision:
        raise ValueError("upstream requires independent review identity@revision")
    events = record.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("upstream requires a nonempty events sequence")
    move_after = record.get("move_after_event", 0)
    if "move_after_event" in record and (type(move_after) is not int or not 0 < move_after < len(events)
                                         or not current or path == old.path or not anchor):
        raise ValueError("move_after_event requires original-file events followed by a move to a named section")
    if "reviewed_move" in record and not move_after:
        raise ValueError("reviewed_move requires move_after_event")
    source = [old]
    scope = None
    if path != old.path:
        if not current and not anchor:
            raise ValueError("a post-move migration basis requires a named section")
        if current and row.get("action") != "upstream":
            raise ValueError("post-move receipts require an upstream ledger action")
        if move_after:
            source = validate_upstream_moved_basis(old, destinations[0], record, move_after, root)
        else:
            validate_upstream_prior_move(old, destinations[0], events[0], root, [old.sha])
        scope = (path, anchor)
    for index, event in enumerate(events[move_after:], move_after + 1):
        try:
            source = validate_upstream_event(source, event, root, scope=scope)
        except (ValueError, subprocess.CalledProcessError) as error:
            raise ValueError(f"upstream event {index}: {error}") from error
    if [block.sha for block in source] != hashes:
        raise ValueError("upstream final replacement does not match the destination hash")
    if events[-1]["reviewer"] != row["reviewer"]:
        raise ValueError("upstream final event reviewer does not match the destination review")
    if not current:
        return source
    current = list(blocks(section_text(read_current(path, root), anchor), path))
    starts = sequence_starts([block.sha for block in current], hashes)
    if len(starts) != 1:
        raise ValueError("no block with expected hash sequence uniquely contiguous at the current destination")
    return current[starts[0]:starts[0] + len(hashes)]


def validate_first_parent(before, after, label, root):
    for commit in (before, after):
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"{label} requires exact Git commit IDs")
    # Shallow roots retain raw parent headers even when revision walking stops.
    git("cat-file", "commit", before, root=root)
    header = git("cat-file", "commit", after, root=root).partition("\n\n")[0]
    parents = re.findall(r"^parent ([0-9a-f]{40})$", header, re.M)
    if not parents or parents[0] != before:
        raise ValueError(f"{label} before_commit is not after_commit's first parent")


def validate_upstream_event(old, record, root=ROOT, *, scope=None):
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
    validate_first_parent(before, after, "upstream", root)
    path, anchor = scope if scope is not None else (old[0].path, "")
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
    if anchor:
        for text, hashes in zip(texts, (source_hashes, replacement_hashes)):
            scoped_hashes = [block.sha for block in blocks(section_text(text, anchor), path)]
            if len(sequence_starts(scoped_hashes, hashes)) != 1:
                raise ValueError("post-move upstream span must stay inside its recorded heading")
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
    """Validate original identities, destinations, provenance and exact allowances.

    The original ledger never changes its source hashes. A separately bound
    upstream migration basis changes the *comparison input* only after its full
    historical chain passes, then ordinary preservation rules apply to that input.
    """
    errors, mappings = [], CheckedMappings()
    try:
        bases = receipt_records(root, "migration-bases.json", "bases")
        allowances = receipt_records(root, "token-allowances.json", "allowances")
        upstream = upstream_records(rows, root, bases)
        mappings.refinements = validated_refinements(base_blocks, root)
    except (FileNotFoundError, ValueError) as error:
        errors.append(f"migration evidence: {error}")
        bases, allowances, upstream = {}, {}, {}
    by_id = {b.id: b for b in base_blocks}
    row_counts = Counter(row["block_id"] for row in rows)
    for identifier in sorted(set(by_id) - set(row_counts)):
        errors.append(f"missing ledger row: {identifier}")
    for identifier, count in row_counts.items():
        if identifier not in by_id:
            errors.append(f"unknown baseline block: {identifier}")
        if count != 1:
            errors.append(f"duplicate ledger row: {identifier}")
    for filename, records in (("migration basis", bases), ("token allowance", allowances)):
        for identifier in records.keys() - by_id.keys():
            errors.append(f"unused {filename} for unknown baseline block: {identifier}")
    cached, candidates_by_dest, used = {}, {}, {}
    used_allowances, used_bases = set(), set()
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
        identity, _, revision = reviewer.partition("@")
        if not reviewer or (action in REVIEWED | {"upstream"} and
                            (not identity or identity == "mechanical" or not revision)):
            errors.append(f"missing independent review identity@revision for {action}: {old.id}")
            continue
        destinations = destination_parts(row.get("dest", ""))
        hashes = destination_parts(row.get("dest_sha", ""))
        if not destinations or len(destinations) != len(hashes):
            errors.append(f"destination/hash count mismatch: {old.id}")
            continue
        basis = [old]
        try:
            if action == "upstream":
                if old.id in bases:
                    raise ValueError("unused migration basis on an upstream action")
                validate_upstream(old, row, upstream.get(old.id), root)
            elif old.id in bases:
                record, receipt = bases[old.id], upstream.get(old.id)
                if receipt is None:
                    raise ValueError("migration basis has no historical upstream receipt")
                basis = validate_upstream(old, receipt, receipt, root, current=False)
                validate_migration_basis(old, record, receipt, basis)
                used_bases.add(old.id)
            if action in {"stay", "move", "archive"} and hashes != [block.sha for block in basis]:
                raise ValueError(f"{action} must preserve the original normalized hash sequence")
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
            errors.append(f"{old.id}: {error}")
            continue
        replacements = []
        for dest, expected in zip(destinations, hashes):
            try:
                path, anchor = split_destination(dest)
                if path not in cached:
                    cached[path] = read_current(path, root)
                    candidates_by_dest[path] = located_destination_blocks(cached[path], path)
                    used[path] = []
                if dest not in candidates_by_dest:
                    section = section_text(cached[path], anchor)
                    start = cached[path].index(section)
                    end = start + len(section)
                    candidates_by_dest[dest] = [(b, i, j) for b, i, j in candidates_by_dest[path]
                                                if start <= i and j <= end]
                matches = [(b, i, j) for b, i, j in candidates_by_dest[dest] if b.sha == expected]
                if not matches:
                    raise ValueError(f"no block with expected hash {expected[:12]} at {dest}")
                if action == "stay" and path != old.path:
                    raise ValueError("stay must retain its source path")
                if action == "archive":
                    if not path.startswith("docs/design/"):
                        raise ValueError("archive must have a docs/design destination")
                    origin = canonical_links(read_current(old.path, root), old.path)
                    if not re.search(r"\]\(" + re.escape(path) + r"(?:[#)]|\s)", origin):
                        raise ValueError("archive requires a forwarding link at its source")
                if action not in {"dedupe", "supersede"}:
                    free = [(b, i, j) for b, i, j in matches
                            if not any(max(i, a) < min(j, z) for a, z in used[path])]
                    if not free:
                        raise ValueError("destination occurrence reused without a reviewed dedupe action")
                    block, start, end = free[0]
                    used[path].append((start, end))
                else:
                    block = matches[0][0]
                replacements.append(block)
            except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
                errors.append(f"{old.id}: {error}")
        if len(replacements) != len(destinations):
            continue
        if action == "split":
            old_words = Counter(" ".join(b.norm for b in basis).split())
            new_words = Counter(" ".join(b.norm for b in replacements).split())
            if old_words - new_words:
                errors.append(f"split loses original words; classify/review as reword: {old.id}")
                continue
        if old.id in allowances:
            try:
                mappings.allowances[old.id] = validate_allowance(old, row, basis, replacements, allowances[old.id])
                used_allowances.add(old.id)
            except ValueError as error:
                errors.append(f"{old.id}: {error}")
                continue
        mappings[old.id] = (action, replacements)
        mappings.bases[old.id] = basis
    for identifier in sorted(allowances.keys() - used_allowances):
        errors.append(f"unused or invalid token allowance: {identifier}")
    for identifier in sorted(bases.keys() - used_bases):
        errors.append(f"unused or invalid migration basis: {identifier}")
    return errors, mappings


def accounting(base_blocks, mappings, kind):
    errors = []
    for old in base_blocks:
        if old.id not in mappings:
            errors.append(f"unaccounted baseline block: {old.id}")
            continue
        action, replacements = mappings[old.id]
        # Historical changes are accepted only after exact Git receipt checks.
        # Only dedupe/supersede may otherwise reduce ordinary preservation counts.
        if action in {"upstream", "dedupe", "supersede"}:
            continue
        if kind == "sentences" and action in {"reword", "split"}:
            continue
        basis = getattr(mappings, "bases", {}).get(old.id, [old])
        if kind == "tokens":
            counter = protected_tokens
        elif kind == "sentences":
            counter = lambda block: Counter(sentences(block.norm))
        elif kind == "examples":
            counter = example_tokens
        else:
            raise ValueError(kind)
        before = sum((counter(block) for block in basis), Counter())
        after = sum((counter(block) for block in replacements), Counter())
        allowed = getattr(mappings, "allowances", {}).get(old.id, {}).get(kind, {})
        before.subtract(allowed.get("removed", {}))
        if before - after:
            errors.append(f"lost {kind}: {old.id}")
    return errors


def anchor_check(rows, build, *, retirements=(), base_blocks=(), mappings=None, report=None,
                 identities=(), root=ROOT):
    pages = html_anchors(build)
    errors = []
    if not pages:
        return ["no built HTML; run docs/make.jl first"]
    frozen = {(row["page"], row["fragment"]) for row in rows}
    identity_errors, identity_pairs, rendered_identities = check_runtime_identities(identities, frozen, build, root) if identities else ([], set(), [])
    errors.extend(identity_errors)
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
        elif fragment and fragment not in pages[page] and (page, fragment) not in approved and (page, fragment) not in identity_pairs:
            errors.append(f"missing original page+fragment: {page or '/'}#{fragment}")
    if report is not None:
        present = {(page, fragment) for page, fragment in frozen if fragment and fragment in pages.get(page, set())}
        retired = approved.keys() - present
        report.update(baseline_fragment_count=sum(bool(fragment) for page, fragment in frozen),
                      present_fragment_count=len(present), retired_fragment_count=len(retired),
                      approved_retirement_count=len(approved),
                      approved_retirements=[dict(page=page, fragment=fragment, target=approved[page, fragment])
                                            for page, fragment in sorted(approved)])
        if identities:
            report.update(exact_fragment_count=len(present - identity_pairs),
                          runtime_identity_count=len(identity_pairs), runtime_identities=rendered_identities,
                          accounted_fragment_count=len(present | set(retired) | identity_pairs))
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
