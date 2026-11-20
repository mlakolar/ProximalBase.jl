#!/usr/bin/env python3
"""Check the family map, family-page headings and permanent clause identifiers."""
from pathlib import Path
import re
import sys

from check_markdown import FENCE

DOCS_SRC = Path(__file__).resolve().parent / "src"
SPEC = DOCS_SRC / "spec"
HEADINGS = (
    "Identity", "Definition and domain", "Traits", "Supported arguments",
    "Preconditions", "Results", "Ordinary-path clauses", "Range clauses",
    "Endpoint clauses", "Failure behavior", "Limitations and non-guarantees",
    "Clauses owned elsewhere", "Evidence", "History",
)
IDENTIFIER = re.compile(r"[a-z][a-z0-9-]*\.[a-z][a-z0-9-]*\.[a-z][a-z0-9-]*")
EXISTING_PAGE_FAMILIES = frozenset({
    "GroupNormL2", "GroupNormLinf", "IndBallL2", "IndBox", "IndNonnegative",
    "LeastSquaresLoss", "LinearFunction", "LogDetLoss", "LogisticLoss", "NormL0",
    "NormL2", "NormLinf", "QuadraticFunction", "Scaled", "SparseGroupNormL2",
    "SqrNormL2", "SqrtLeastSquaresLoss", "ZeroFunction", "kkt_violation",
})
FAMILY_MAP_HEADER = ["Family or operation", "Current entry point", "Migration state"]


def markdown_content(text):
    """Yield lines with their fence directive, or None for unfenced text."""
    fence = directive = None
    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if not match:
            yield number, line, directive
            continue
        marker, suffix = match.groups()
        if fence is None:
            fence, directive = marker, suffix.strip()
        elif marker.startswith(fence) and not suffix.strip():
            fence = directive = None


def table_cells(line):
    cells = re.split(r"(?<!\\)\|", line.strip())
    if len(cells) == 1:
        return []
    if not cells[0]:
        cells.pop(0)
    if not cells[-1]:
        cells.pop()
    return [cell.strip() for cell in cells]

def family_map_rows(index):
    lines = iter(markdown_content(index))
    for _, line, directive in lines:
        if directive is None and line.strip() == "## Family map":
            break
    else:
        return
    for number, line, directive in lines:
        if directive is not None:
            continue
        if line.lstrip().startswith(("# ", "## ")):
            return
        if table_cells(line) == FAMILY_MAP_HEADER:
            break
    else:
        return

    separator_number, separator, directive = next(lines, (0, "", None))
    alignment = table_cells(separator)
    if (separator_number != number + 1 or directive is not None or len(alignment) != 3
            or any(len(cell) < 3 or set(cell) - {"-", ":"} for cell in alignment)):
        return
    previous = separator_number
    for number, line, directive in lines:
        cells = table_cells(line)
        if number != previous + 1 or directive is not None or not cells:
            return
        yield number, cells
        previous = number


def family_map_errors(reference, index, page_names):
    errors, mapped, documented = [], set(), set()
    for _, line, directive in markdown_content(reference):
        if directive is not None and re.match(r"@docs(?=[\s;]|$)", directive):
            name = re.split(r"[(\{]", line, maxsplit=1)[0].strip().rsplit(".", 1)[-1]
            if name.isidentifier() and name[0].isupper():
                documented.add(name)

    for number, cells in family_map_rows(index):
        where = f"spec/index.md:{number}"
        if len(cells) != 3:
            errors.append(f"{where}: expected a family, entry point and migration state")
            continue
        family, _, state = cells
        names = set(re.findall(r"`(\w+)`", family))
        if not names:
            errors.append(f"{where}: expected a code-formatted family or operation name")
            continue
        mapped.update(names)
        if state == "Existing pages":
            for name in sorted(names - EXISTING_PAGE_FAMILIES):
                errors.append(f"{where}: {name} cannot enter the family map as Existing pages")
        elif state != "Thin constructor" and not re.fullmatch(r"Thin constructor of `\w+`", state):
            for name in sorted(names):
                if f"{name.lower()}.md" not in page_names:
                    errors.append(f"{where}: {name} needs a family contract page or a thin-constructor entry")

    for name in sorted(documented - mapped):
        errors.append(f"api/functions.md: {name} is missing from the family map in spec/index.md")
    return errors


def main():
    pages = sorted(path for path in SPEC.glob("*.md") if path.name != "index.md")
    errors = family_map_errors(
        (DOCS_SRC / "api/functions.md").read_text(encoding="utf-8"),
        (SPEC / "index.md").read_text(encoding="utf-8"),
        {path.name for path in pages},
    )
    identifiers = {}
    for path in pages:
        headings = []
        fenced = False
        clause_table = False
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
                if first.lower() in {"id", "identifier", "clause id"}:
                    clause_table = True
                    continue
                if re.fullmatch(r"[:\-\s]+", first):
                    continue
                if clause_table and not IDENTIFIER.fullmatch(first):
                    errors.append(f"{path.name}:{number}: invalid clause identifier {first!r}")
                elif IDENTIFIER.fullmatch(first):
                    where = f"{path.name}:{number}"
                    if first in identifiers:
                        errors.append(f"{where}: duplicate clause {first} ({identifiers[first]})")
                    identifiers[first] = where
            else:
                clause_table = False
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
