#!/usr/bin/env python3
"""Check a committed rewrite of the published SparseGroupNormL2 text."""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys

import migration_checks as m


REFERENCE = m.AUDIT / "post-baseline-sparsegroupnorml2.json"


@dataclass(frozen=True)
class Span:
    scope: str
    text: str

    @property
    def sha(self):
        return m.digest(self.text)


def authored_path(path):
    if not re.fullmatch(r"docs/src/(?:[a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+\.md", path):
        raise ValueError(f"expected an authored documentation path: {path}")
    return path


def authored_blob(revision, path, root):
    data = subprocess.check_output(["git", "show", revision + ":" + path], cwd=root)
    text = data.decode("utf-8")
    if "\r" in text:
        raise ValueError(f"expected LF line endings in the authored Git blob: {path}")
    return text


def authored_section(page, path, anchor):
    if not anchor:
        raise ValueError(f"expected a named section in {path}")
    section = m.section_text(page, anchor)
    # A fenced copy can contain the same section text as the authored heading.
    layout = m.located_destination_blocks(page, path)
    heading = section.split("\n", 1)[0].rstrip()
    headings = [first for block, first, _ in layout if block.kind == "heading" and block.text == heading]
    if len(headings) != 1 or not page.startswith(section, headings[0]):
        raise ValueError(f"cannot locate a unique authored section for {path}#{anchor}")
    return section, headings[0], layout


def locate(span, page):
    path, anchor = m.split_destination(span.scope)
    section, offset, layout = authored_section(page, path, anchor)
    start = section.find(span.text)
    if not span.text or start < 0 or section.find(span.text, start + 1) >= 0:
        raise ValueError(f"passage {span.sha[:12]} must occur once in {span.scope}")
    start += offset
    end = start + len(span.text)
    blocks = [(block, first, last) for block, first, last in layout
              if first < end and start < last]
    if not blocks or blocks[0][1] != start or blocks[-1][2] != end:
        raise ValueError(f"passage must occupy complete Markdown blocks in {span.scope}")
    return start, end, [block for block, _, _ in blocks]


def published_spans(reference, root):
    revision = reference["commit"]
    if reference["schema"] != 1 or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("reference requires schema 1 and an exact Git commit")
    parent = m.git("rev-parse", "--verify", revision + "^1", root=root).strip()
    manifest_name = reference["move_manifest"]
    if not re.fullmatch(r"moves-[a-z0-9_]+\.json", manifest_name):
        raise ValueError("invalid reference move manifest name")
    manifest_path = "docs/audit/docs-tracks-1019/" + manifest_name
    manifest = json.loads(m.git("show", revision + ":" + manifest_path, root=root))
    if manifest["schema"] != 1 or manifest["source_revision"] != parent or not manifest["moves"]:
        raise ValueError("move manifest must bind passages to the published commit's first parent")

    spans = {}
    for move in manifest["moves"]:
        path = authored_path(move["source"])
        original = authored_blob(parent, path, root)
        start, end = move["source_start"], move["source_end"]
        text = move["text"]
        if (move["source_revision"] != parent or not 0 <= start < end <= len(original)
                or original[start:end] != text or m.digest(text) != move["source_sha256"]
                or original.count(text) != 1 or original.count("\n", 0, start) + 1 != move["source_line"]):
            raise ValueError(f"invalid historical passage at {path}:{move['source_line']}")
        destination = authored_path(move["destination"])
        span = Span(destination + "#" + move["anchor"], move["destination_text"])
        if span.text != text or span.sha != move["destination_sha256"] or span.sha in spans:
            raise ValueError(f"invalid or duplicate published passage at {span.scope}")
        spans[span.sha] = span
    return spans


def declared_rewrites(plan, sources, path):
    if plan["schema"] != 1 or not isinstance(plan["changes"], list):
        raise ValueError("rewrites require schema 1 and a changes list")
    rewrites = {}
    for change in plan["changes"]:
        sha = change["source_sha256"]
        action = change["action"]
        identity, _, revision = change["reviewer"].partition("@")
        if (sha not in sources or sha in rewrites or action not in {"reword", "split"}
                or not identity or identity == "mechanical" or not revision):
            raise ValueError(f"invalid, duplicate or unreviewed mapping for passage {sha}")
        outputs = []
        for item in change["destinations"]:
            span = Span(item["scope"], item["text"])
            if m.split_destination(span.scope)[0] != path or span.sha != item["sha256"]:
                raise ValueError(f"destination must match the reviewed hash in {path}")
            outputs.append(span)
        if not outputs or (action == "split" and len(outputs) < 2):
            raise ValueError(f"missing replacement pieces for passage {sha}")
        rewrites[sha] = action, outputs
    return rewrites


def tokens(blocks, count):
    total = Counter()
    for block in blocks:
        total.update(count(block))
    return total


def check(reference, candidate, plan, root=m.ROOT):
    if not re.fullmatch(r"[0-9a-f]{40}", candidate):
        raise ValueError("candidate requires an exact Git commit")
    sources = published_spans(reference, root)
    revision = reference["commit"]
    paths = {m.split_destination(span.scope)[0] for span in sources.values()}
    if len(paths) != 1 or not next(iter(paths)).startswith("docs/src/spec/"):
        raise ValueError("the reference manifest must cover one family contract")
    path = paths.pop()
    rewrites = declared_rewrites(plan, sources, path)
    original = authored_blob(revision, path, root)
    current = authored_blob(candidate, path, root)

    findings = []
    occupied = []
    for sha, source in sources.items():
        _, _, before = locate(source, original)
        action, destinations = rewrites.get(sha, ("stay", [source]))
        try:
            ranges = [locate(span, current) for span in destinations]
        except ValueError as error:
            findings.append(f"{source.scope}: {error}")
            continue
        after = []
        previous_end = 0
        for start, end, blocks in ranges:
            if start < previous_end or any(start < stop and begin < end for begin, stop in occupied):
                findings.append(f"{source.scope}: replacement passages overlap or are out of order")
            occupied.append((start, end))
            previous_end = end
            after.extend(blocks)
        if action == "split":
            old_words = Counter(word for block in before for word in block.norm.split())
            new_words = Counter(word for block in after for word in block.norm.split())
            if old_words - new_words:
                findings.append(f"{source.scope}: split loses original words")
        for name, count in (("protected tokens", m.protected_tokens), ("example markers", m.example_tokens)):
            if tokens(before, count) != tokens(after, count):
                findings.append(f"{source.scope}: changed {name} in passage {sha[:12]}")

    examples = reference["guide_examples"]
    guide, anchor = m.split_destination(examples["scope"])
    authored_path(guide)
    original_guide = authored_blob(revision, guide, root)
    current_guide = authored_blob(candidate, guide, root)
    guide_section, _, _ = authored_section(original_guide, guide, anchor)
    opener = "```@example " + examples["sandbox"]
    example_count = sum(block.kind == "code" and block.text.splitlines()[0] == opener
                        for block in m.blocks(guide_section, guide))
    if not example_count or example_count != examples["count"]:
        raise ValueError("reference guide example count differs from the published inventory")
    # Markdown before the section can change whether the examples execute.
    if original_guide != current_guide:
        findings.append(f"{examples['scope']}: retained examples or their guide context changed")
    return dict(reference=revision, candidate=candidate, passages=len(sources),
                guide_examples=example_count, findings=findings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="Git commit to check")
    parser.add_argument("--rewrites", help="path to the reviewed mapping in the candidate commit")
    args = parser.parse_args()
    try:
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        candidate = m.git("rev-parse", "--verify", "--end-of-options", args.candidate + "^{commit}").strip()
        plan = (json.loads(m.git("show", candidate + ":" + args.rewrites))
                if args.rewrites else dict(schema=1, changes=[]))
        result = check(reference, candidate, plan)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Passage preservation could not run: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 1 if result["findings"] else 0


if __name__ == "__main__":
    sys.exit(main())
