#!/usr/bin/env python3
"""Lint active documentation Markdown, including sources of generated contracts.

Checks that API names in prose/tables are formatted as code or links and that
fenced code blocks and inline backticks are balanced. Historical audits are
intentionally excluded: they preserve the original wording of older designs.

The check is deliberately conservative: only known Julia symbols, not English
words or mathematical variable names, are treated as API identifiers.
"""
from pathlib import Path
import re
import sys

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
CONTRACTS = (
    "block-prox-0.6.md",
    "iterate-family-0.6.md",
    "symmetric-sparse-iterate-0.6.md",
)

API_NAMES = (
    "AbstractFunction", "SeparableFunction", "ScalarLoss",
    "LeastSquaresLoss", "SqrNormL2", "QuadraticFunction", "ZeroFunction",
    "LinearFunction", "NormL1", "Scaled", "SparseIterate",
    "SymmetricSparseIterate", "BlockIterate",
    "Float32", "Float64", "AbstractArray", "AbstractVector", "AbstractMatrix",
    "value_and_gradient!", "value_and_gradient", "value_and_prox!",
    "gradient!", "prox_block!", "prox_block", "prox!",
    "value_entry", "kkt_violation", "lipschitz", "opnorm2",
    "reset_cache!", "dropzeros!", "activate!", "isactive",
    "activecoordinates", "activevalues", "nactive", "ncoordinates",
    "blockview", "blockindices", "matrixview",
    "tril_index", "tril_cartesian", "check_symmetric_parameters",
    "check_parameters", "entry_parameters", "issmooth", "isproximable",
    "isseparable", "isconvex", "iszeropreserving", "samegroups",
    "copyto!", "nnz",
    "deriv2", "deriv", "value_and_deriv",
)
NAME_RE = re.compile(
    r"(?<![\w])(?:" +
    "|".join(re.escape(name) for name in sorted(API_NAMES, key=len, reverse=True)) +
    r")(?![\w])"
)
PROTECTED = re.compile(
    r"\[[^\]\n]*\]\([^)\n]*\)"    # Markdown links, including @ref targets
    r"|(?<!\\)\x60+[^\x60\n]*\x60+"  # Inline code and Documenter math
    r"|https?://\S+"                 # URLs
    r"|\$[^$\n]+\$"                 # GitHub inline mathematics
)
FENCE = re.compile(r"^\s*(\x60{3,}|~{3,})(.*)$")


def sources():
    files = [
        p for p in (DOCS / "src").rglob("*.md")
        if "examples" not in p.parts and "contracts" not in p.parts
    ]
    files.extend(DOCS / "design" / name for name in CONTRACTS)
    files.extend((ROOT / "README.md", ROOT / "CONTRIBUTING.md"))
    return sorted(files)


def check_file(path):
    errors = []
    opened = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fence = FENCE.match(line)
        if fence:
            marker, suffix = fence.groups()
            if opened is None:
                if not suffix.strip():
                    errors.append((line_number, "Unlabeled code fence; specify a language or Documenter directive"))
                opened = (marker[0], len(marker), line_number)
            elif marker[0] == opened[0] and len(marker) >= opened[1] and not suffix.strip():
                opened = None
            continue
        if opened is not None:
            continue

        # Headings are navigational labels; API names there may intentionally
        # appear without code formatting. Their bodies still require formatting.
        if line.lstrip().startswith("#"):
            continue

        # Mask links, code/math spans, and URLs before scanning remaining prose.
        masked = PROTECTED.sub(lambda m: " " * len(m.group()), line)
        # An @ref target following a styled label is not prose.
        masked = re.sub(r"\(@ref(?:\s+[^)]+)?\)", "", masked)
        for match in NAME_RE.finditer(masked):
            errors.append((
                line_number,
                f"Unformatted Julia API name '{match.group()}'; use backticks or a Markdown link"
            ))

        # A line should not end with an unmatched inline code/math delimiter.
        backticks = re.findall(r"(?<!\\)\x60+", line)
        if len(backticks) % 2:
            errors.append((line_number, "Unmatched inline backtick delimiter"))
        elif any(backticks[j] != backticks[j + 1] for j in range(0, len(backticks), 2)):
            errors.append((line_number, "Mismatched inline backtick delimiter lengths"))

    if opened is not None:
        errors.append((opened[2], "Unclosed fenced code or math block"))
    return errors


def main():
    errors = []
    files = sources()
    for path in files:
        if not path.is_file():
            errors.append(f"{path}: expected documentation source is missing")
            continue
        for line_number, message in check_file(path):
            errors.append(f"{path.relative_to(ROOT)}:{line_number}: {message}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(
            f"Markdown formatting audit failed: {len(errors)} error(s) "
            f"across {len(files)} active source files.", file=sys.stderr
        )
        return 1
    print(f"Markdown formatting audit passed for {len(files)} active source files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
