#!/usr/bin/env python3
"""Check the inventory citations of docs/design/overlapping-group-penalties-963.md.

Each row of the two inventory tables names one or more symbols (first column, in
backticks) and as many `path:line` citations (second column, in the same order). For each row, the file is read at the pinned
commit of its table with `git show <sha>:<path>`, and the check fails when the
cited line does not contain the symbol. Run from anywhere inside the repository:

    python3 docs/audit/overlapping-group-penalties-963/check_inventory.py
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NOTE = ROOT / "docs" / "design" / "overlapping-group-penalties-963.md"
COMMITS = {
    "master": "13891ccfd5c33a042c5df82d670abf1f2d786a0f",
    "legacy/full": "eff21b6d7d33c622313122f46c6ee71693c101ee",
}
CODE = re.compile(r"`([^`]+)`")
CITE = re.compile(r"^(src/[\w/.-]+\.jl):(\d+)$")


def file_lines(sha, path, cache={}):
    key = (sha, path)
    if key not in cache:
        out = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{path}"],
                             capture_output=True, text=True, check=True).stdout
        cache[key] = out.splitlines()
    return cache[key]


def main():
    text = NOTE.read_text(encoding="utf-8").splitlines()
    in_inventory = False
    sha = None
    rows = {name: 0 for name in COMMITS}
    failures = 0
    for number, line in enumerate(text, 1):
        if line.startswith("## "):
            in_inventory = line.startswith("## 3.")
            sha = None
            continue
        if not in_inventory:
            continue
        if line.startswith("### "):
            sha = next((name for name in COMMITS if name in line), None)
            continue
        if sha is None or not line.startswith("|") or set(line.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        symbols = CODE.findall(cells[0])
        cites = CODE.findall(cells[1]) if len(cells) > 1 else []
        if not symbols and not cites:
            continue                      # the header row
        matches = [CITE.match(c) for c in cites]
        if not symbols or len(symbols) != len(cites) or not all(matches):
            print(f"FAIL note line {number}: a row needs as many symbols as path:line citations")
            failures += 1
            continue
        for symbol, match in zip(symbols, matches):
            path, lineno = match.group(1), int(match.group(2))
            name = symbol.split("(")[0]
            lines = file_lines(COMMITS[sha], path)
            ok = 1 <= lineno <= len(lines) and name in lines[lineno - 1]
            rows[sha] += 1
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {sha} {path}:{lineno} {name}")
    for name, count in rows.items():
        if count == 0:
            print(f"FAIL no rows for {name}")
            failures += 1
    print(f"{sum(rows.values())} citations, {failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
