#!/usr/bin/env python3
"""Verdict for the nightly full-validation matrix.

Reads the per-cell logs downloaded from the `full-validation-log-<os>-<julia>-<shard>`
artifacts and checks them against the full test selection (the union of all four
tiers in `test/tiers.toml`).  The expected cells below mirror the matrix in
`.github/workflows/full-validation.yml`; `test_full_validation_verdict.py` compares
the two texts.  Standard library only; Python 3.11 or newer (`tomllib`).
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

OSES = ("ubuntu-latest", "windows-latest", "macos-latest")
JULIAS = ("1.10", "1.13")
SHARDS = (1, 2, 3)
SHARD_COUNT = 3
TIERS = ("quick", "standard", "thorough", "full")
ARTIFACT_PREFIX = "full-validation-log-"

HEADER_RE = re.compile(r"^PB_FULL_VALIDATION(?P<fields>(?:\s+\S+=\S*)*)\s*$")
RUNNER_RE = re.compile(
    r"^ProximalBase minimal tests: tier (?P<tier>\S+); shard (?P<shard>\S+); "
    r"(?P<count>\d+) files; Julia (?P<version>\S+) on (?P<machine>.+?)\s*$"
)
FILE_TIME_RE = re.compile(
    r"^PB_TEST_FILE_TIME file=(?P<file>\S+) elapsed_s=(?P<elapsed>\S+) status=(?P<status>\S+)\s*$"
)


class TiersError(Exception):
    """The tier manifest does not define a full selection."""


def expected_cells():
    return [(o, j, s) for o in OSES for j in JULIAS for s in SHARDS]


def artifact_name(os_name, julia, shard):
    return f"{ARTIFACT_PREFIX}{os_name}-{julia}-{shard}"


def load_selection(tiers_path):
    """Return the sorted union of the four tiers; error on a missing tier or a duplicate."""
    with open(tiers_path, "rb") as handle:
        manifest = tomllib.load(handle)
    tiers = manifest.get("tiers")
    if not isinstance(tiers, dict) or set(tiers) != set(TIERS):
        raise TiersError("test manifest must declare all four tiers")
    names = [str(name) for tier in TIERS for name in tiers[tier]]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise TiersError(f"duplicate test file in manifest: {duplicates}")
    return sorted(names)


class Cell:
    """What one log file says about one OS x Julia x shard cell."""

    def __init__(self, os_name, julia, shard):
        self.os, self.julia, self.shard = os_name, julia, shard
        self.present = False
        self.problems = []
        self.results = {}  # file -> (elapsed_s, status)
        self.version = None
        self.machine = None

    @property
    def label(self):
        return f"{self.os} / Julia {self.julia} / shard {self.shard}/{SHARD_COUNT}"

    def problem(self, text):
        self.problems.append(f"{self.label}: {text}")


def parse_log(cell, text, expected_commit):
    cell.present = True
    headers, runners = [], []
    for line in text.splitlines():
        line = line.rstrip("\r")
        match = HEADER_RE.match(line)
        if match:
            headers.append(dict(field.split("=", 1) for field in match["fields"].split()))
            continue
        match = RUNNER_RE.match(line)
        if match:
            runners.append(match)
            continue
        match = FILE_TIME_RE.match(line)
        if match:
            name = match["file"]
            if name in cell.results:
                cell.problem(f"test file {name} is reported twice in one log")
            try:
                elapsed = float(match["elapsed"])
            except ValueError:
                elapsed = float("nan")
                cell.problem(f"test file {name} has a malformed elapsed time {match['elapsed']!r}")
            cell.results[name] = (elapsed, match["status"])

    if len(headers) != 1:
        cell.problem(f"expected one PB_FULL_VALIDATION header, found {len(headers)}")
    else:
        header = headers[0]
        wanted = {
            "commit": expected_commit,
            "os": cell.os,
            "julia": cell.julia,
            "shard": f"{cell.shard}/{SHARD_COUNT}",
        }
        for key, value in wanted.items():
            if header.get(key) != value:
                cell.problem(f"header {key}={header.get(key)!r}, expected {value!r}")

    if len(runners) != 1:
        cell.problem(f"expected one 'ProximalBase minimal tests' line, found {len(runners)}")
    else:
        runner = runners[0]
        cell.version, cell.machine = runner["version"], runner["machine"]
        if not cell.version.startswith(cell.julia + "."):
            cell.problem(f"ran Julia {cell.version}, expected {cell.julia}.x")
        if runner["tier"] != "full":
            cell.problem(f"ran tier {runner['tier']}, expected full")
        if runner["shard"] != f"{cell.shard}/{SHARD_COUNT}":
            cell.problem(f"runner reports shard {runner['shard']}, expected {cell.shard}/{SHARD_COUNT}")


def find_cell_logs(logs_dir):
    """Map artifact directory name -> list of files inside it; also return stray entries."""
    found, stray = {}, []
    root = Path(logs_dir)
    if not root.is_dir():
        return found, stray
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith(ARTIFACT_PREFIX):
            found[entry.name] = sorted(p for p in entry.rglob("*") if p.is_file())
        else:
            stray.append(entry.name)
    return found, stray


def pair_label(os_name, julia):
    return f"{os_name} / Julia {julia}"


def evaluate(logs_dir, selection, expected_commit):
    """Return (problems, rows, found_any) for the logs under `logs_dir`."""
    problems = []
    found, stray = find_cell_logs(logs_dir)
    if not Path(logs_dir).is_dir():
        problems.append(f"log directory {logs_dir} does not exist; no test artifacts were downloaded")
    elif not found:
        problems.append(f"log directory {logs_dir} holds no {ARTIFACT_PREFIX}* artifact")

    expected_names = {artifact_name(*cell): cell for cell in expected_cells()}
    for name in found:
        if name not in expected_names:
            problems.append(f"log artifact {name} is outside the expected cells")
    for name in stray:
        problems.append(f"unexpected entry {name} in the log directory")

    cells = {}
    for key in expected_cells():
        cell = Cell(*key)
        cells[key] = cell
        name = artifact_name(*key)
        files = found.get(name)
        if files is None:
            cell.problem("no log artifact (missing cell)")
        elif len(files) != 1:
            cell.problem(f"expected exactly one log file in {name}, found {len(files)}")
        else:
            parse_log(cell, files[0].read_text(encoding="utf-8", errors="replace"), expected_commit)
        problems.extend(cell.problems)

    selected = set(selection)
    rows = []
    for os_name in OSES:
        for julia in JULIAS:
            pair = [cells[(os_name, julia, s)] for s in SHARDS]
            label = pair_label(os_name, julia)
            owner = {}
            failing = []
            reported = set()
            for cell in pair:
                for name, (_, status) in cell.results.items():
                    reported.add(name)
                    if name in owner and owner[name] != cell.shard:
                        problems.append(
                            f"{label}: test file {name} ran in shards {owner[name]} and {cell.shard}"
                        )
                    owner.setdefault(name, cell.shard)
                    if status != "pass":
                        failing.append(f"{name} ({status})")
                        problems.append(
                            f"{label} / shard {cell.shard}/{SHARD_COUNT}: test file {name} has status {status}"
                        )
            unexpected = sorted(reported - selected)
            if unexpected:
                problems.append(f"{label}: files outside the full selection ran: {unexpected}")
            if all(cell.present for cell in pair):
                missing = sorted(selected - reported)
                if missing:
                    problems.append(f"{label}: files in the full selection did not run: {missing}")
            passed = {n for c in pair for n, (_, st) in c.results.items() if st == "pass"} & selected
            times = {}
            for cell in pair:
                for name, (elapsed, _) in cell.results.items():
                    times[name] = max(elapsed, times.get(name, elapsed))
            slowest = sorted(times.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            rows.append(
                {
                    "os": os_name,
                    "julia": ", ".join(sorted({c.version for c in pair if c.version})) or "n/a",
                    "machine": ", ".join(sorted({c.machine for c in pair if c.machine})) or "n/a",
                    "shards": sum(1 for c in pair if c.present and not c.problems),
                    "passed": len(passed),
                    "expected": len(selected),
                    "failing": sorted(failing),
                    "slowest": slowest,
                }
            )
    return problems, rows


def cell_text(value):
    return str(value).replace("|", "\\|")


def render(commit, rows, problems):
    lines = [
        "## Full validation",
        "",
        f"Commit: `{commit}`",
        "",
        "| OS | Julia | Machine | Shards completed | Files passed | Failing files | Slowest 5 files |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        failing = ", ".join(row["failing"]) or "none"
        slowest = ", ".join(f"{name} {seconds:.1f} s" for name, seconds in row["slowest"]) or "n/a"
        lines.append(
            "| " + " | ".join(
                cell_text(x)
                for x in (
                    row["os"],
                    row["julia"],
                    row["machine"],
                    f"{row['shards']}/{SHARD_COUNT}",
                    f"{row['passed']}/{row['expected']}",
                    failing,
                    slowest,
                )
            ) + " |"
        )
    lines.append("")
    if problems:
        lines.append(f"FULL VALIDATION: FAIL ({len(problems)} problems)")
        lines.append("")
        lines.extend(f"- {problem}" for problem in problems)
    else:
        lines.append("FULL VALIDATION: PASS")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", required=True, help="directory holding the downloaded artifact directories")
    parser.add_argument("--tiers", required=True, help="path of test/tiers.toml")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--summary", help="Markdown file to append the report to")
    args = parser.parse_args(argv)

    try:
        selection = load_selection(args.tiers)
        problems, rows = evaluate(args.logs, selection, args.expected_commit)
    except (TiersError, OSError, tomllib.TOMLDecodeError) as error:
        problems, rows = [f"cannot read the test selection: {error}"], []
    report = render(args.expected_commit, rows, problems)
    print(report)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(report)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
