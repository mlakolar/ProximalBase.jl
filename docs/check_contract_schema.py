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
CONTRACT_LINK = re.compile(r"\[[^\]]+\]\((?:\./)?([^/\s()#]+\.md)(?:#[^\s)]+)?\)")
API_OWNER = re.compile(r"\[`(\w+)`\]\(@ref(?:\s+(?:ProximalBase\.)?\1)?\)")
LINK_OR_CODE = re.compile(r"(?<!`)(?P<code>`+)(?!`).*?(?<!`)(?P=code)(?!`)|\[[^\]]*\]\([^)]*\)")


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


def markdown_links(line):
    for token in LINK_OR_CODE.finditer(line):
        if token["code"]:
            continue
        prefix = line[:token.start()]
        escapes = len(prefix) - len(prefix.rstrip("\\"))
        if not prefix.endswith("!") and escapes % 2 == 0:
            yield token[0]


def contract_owners(text):
    """Read declared families from the visible title and Identity API links."""
    names = set()
    identity = title_seen = False
    for _, line, directive in markdown_content(text):
        if directive is not None:
            continue
        if line.startswith("# ") and not title_seen:
            title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line[2:])
            names.update(re.findall(r"\w+", title))
            title_seen = True
        elif line.startswith("## "):
            title = re.sub(r"\[([^\]]+)\]\(@id\s+[^)]+\)", r"\1", line[3:])
            identity = title.strip() == "Identity"
        elif identity:
            for link in markdown_links(line):
                owner = API_OWNER.fullmatch(link)
                if owner:
                    names.add(owner[1])
    return names


def family_map_errors(reference, index, pages):
    errors, mapped, documented = [], set(), set()
    owners = {filename: contract_owners(text) for filename, text in pages.items()}
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
        family, entry_point, state = cells
        names = set(re.findall(r"`(\w+)`", family))
        if not names:
            errors.append(f"{where}: expected a code-formatted family or operation name")
            continue
        mapped.update(names)
        if state == "Existing pages":
            for name in sorted(names - EXISTING_PAGE_FAMILIES):
                errors.append(f"{where}: {name} cannot enter the family map as Existing pages")
        elif state != "Thin constructor" and not re.fullmatch(r"Thin constructor of `\w+`", state):
            linked_owners = set()
            for token in markdown_links(entry_point):
                link = CONTRACT_LINK.fullmatch(token)
                if link:
                    linked_owners.update(owners.get(link[1], ()))
            for name in sorted(names - linked_owners):
                errors.append(f"{where}: {name} needs a family contract page or a thin-constructor entry")

    for name in sorted(documented - mapped):
        errors.append(f"api/functions.md: {name} is missing from the family map in spec/index.md")
    return errors


def main():
    pages = {path.name: path.read_text(encoding="utf-8")
             for path in sorted(SPEC.glob("*.md")) if path.name != "index.md"}
    errors = family_map_errors(
        (DOCS_SRC / "api/functions.md").read_text(encoding="utf-8"),
        (SPEC / "index.md").read_text(encoding="utf-8"),
        pages,
    )
    identifiers = {}
    for filename, source in pages.items():
        headings = []
        fenced = False
        clause_table = False
        for number, line in enumerate(source.splitlines(), 1):
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
                    errors.append(f"{filename}:{number}: invalid clause identifier {first!r}")
                elif IDENTIFIER.fullmatch(first):
                    where = f"{filename}:{number}"
                    if first in identifiers:
                        errors.append(f"{where}: duplicate clause {first} ({identifiers[first]})")
                    identifiers[first] = where
            else:
                clause_table = False
        if tuple(headings) != HEADINGS:
            errors.append(f"{filename}: expected the fourteen family headings in their documented order")
    if not pages:
        errors.append("No authored family contract pages found")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Contract schema passed: {len(pages)} family page(s), {len(identifiers)} assigned clause identifiers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
