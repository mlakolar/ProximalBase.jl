#!/usr/bin/env python3
"""Check authored family-page headings and permanent clause identifiers."""
from pathlib import Path
import re
import sys

SPEC = Path(__file__).resolve().parent / "src/spec"
HEADINGS = (
    "Identity", "Definition and domain", "Traits", "Supported arguments",
    "Preconditions", "Results", "Ordinary-path clauses", "Range clauses",
    "Endpoint clauses", "Failure behavior", "Limitations and non-guarantees",
    "Clauses owned elsewhere", "Evidence", "History",
)
IDENTIFIER = re.compile(r"[a-z][a-z0-9-]*\.[a-z][a-z0-9-]*\.[a-z][a-z0-9-]*")


def main():
    errors, identifiers = [], {}
    pages = sorted(path for path in SPEC.glob("*.md") if path.name != "index.md")
    for path in pages:
        headings = []
        fenced = False
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.startswith("```"):
                fenced = not fenced
                continue
            if fenced:
                continue
            if line.startswith("## "):
                title = re.sub(r"\[([^\]]+)\]\(@id\s+[^)]+\)", r"\1", line[3:])
                headings.append(title.strip())
            if line.startswith("|"):
                first = line.split("|", 2)[1].strip().strip("`")
                if IDENTIFIER.fullmatch(first):
                    where = f"{path.name}:{number}"
                    if first in identifiers:
                        errors.append(f"{where}: duplicate clause {first} ({identifiers[first]})")
                    identifiers[first] = where
        if tuple(headings) != HEADINGS:
            errors.append(f"{path.name}: expected the fourteen family headings in their documented order")
    if not pages:
        errors.append("No authored family contract pages found")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Contract schema passed: {len(pages)} family page(s), {len(identifiers)} assigned clause identifiers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
