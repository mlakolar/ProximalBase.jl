#!/usr/bin/env python3
"""Verdict for the nightly full-validation matrix.

Reads the per-cell logs downloaded from the
`full-validation-log-<os>-<julia>-<shard>-attempt-<n>` artifacts and the job
conclusions of the same workflow run attempt. The job conclusions are checked
against the expected matrix job names, and the passing timing records of each
OS and Julia pair against the full test selection (the union of all four tiers
in `test/tiers.toml`).  The expected
cells below mirror the matrix in `.github/workflows/full-validation.yml`;
`test_full_validation_verdict.py` compares the two texts.  It imports only the
standard library and needs Python 3.11 or newer (`tomllib`).
"""
from __future__ import annotations

import argparse
import json
import math
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
JOB_PREFIX = "Full / "
# Substrings of the platform triple that `Sys.MACHINE` reports for each runner OS.
MACHINE_MARKERS = {
    "ubuntu-latest": ("linux",),
    "windows-latest": ("mingw", "windows"),
    "macos-latest": ("apple-darwin",),
}

HEADER_RE = re.compile(r"^PB_FULL_VALIDATION(?P<fields>(?:\s+\S+=\S*)*)\s*$")
RUNNER_RE = re.compile(
    r"^ProximalBase minimal tests: tier (?P<tier>\S+); shard (?P<shard>\S+); "
    r"(?P<count>\d+) files; Julia (?P<version>\S+) on (?P<machine>.+?)\s*$"
)
FILE_TIME_RE = re.compile(
    r"^PB_TEST_FILE_TIME file=(?P<file>\S+) elapsed_s=(?P<elapsed>\S+) status=(?P<status>\S+)\s*$"
)
SHARD_TIME_RE = re.compile(
    r"^PB_TEST_SHARD_TIME shard=(?P<shard>\S+) files=(?P<count>\d+) elapsed_s=(?P<elapsed>\S+)\s*$"
)
NAME_RE = re.compile(r"\S+")


class TiersError(Exception):
    """The tier manifest does not define a full selection."""


def expected_cells():
    return [(o, j, s) for o in OSES for j in JULIAS for s in SHARDS]


def artifact_name(os_name, julia, shard, attempt):
    return f"{ARTIFACT_PREFIX}{os_name}-{julia}-{shard}-attempt-{attempt}"


def job_name(os_name, julia, shard):
    """The job name in the exact format of the workflow's `name:` for the test job."""
    return f"{JOB_PREFIX}{os_name} / Julia {julia} (shard {shard}/{SHARD_COUNT})"


def load_selection(tiers_path):
    """Return the sorted union of the four tiers.

    Each tier must be an array of non-empty strings without whitespace, no file may
    appear twice, and the union needs at least one file for each shard.
    """
    with open(tiers_path, "rb") as handle:
        manifest = tomllib.load(handle)
    tiers = manifest.get("tiers")
    if not isinstance(tiers, dict) or set(tiers) != set(TIERS):
        raise TiersError("test manifest must declare all four tiers")
    names = []
    for tier in TIERS:
        entries = tiers[tier]
        if not isinstance(entries, list):
            raise TiersError(f"tier {tier} must be an array of file names")
        for entry in entries:
            if not isinstance(entry, str) or not NAME_RE.fullmatch(entry):
                raise TiersError(f"tier {tier} holds an entry that is not a non-empty name: {entry!r}")
            names.append(entry)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise TiersError(f"duplicate test file in manifest: {duplicates}")
    if len(names) < SHARD_COUNT:
        raise TiersError(
            f"the full selection has {len(names)} files, fewer than the {SHARD_COUNT} shards"
        )
    return sorted(names)


def parse_seconds(text):
    """Return a finite, non-negative float, or None."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


class Cell:
    """What one log file says about one OS x Julia x shard cell."""

    def __init__(self, os_name, julia, shard):
        self.os, self.julia, self.shard = os_name, julia, shard
        self.present = False
        self.problems = []
        self.results = {}  # file -> (elapsed_s or None, status)
        self.version = None
        self.machine = None

    @property
    def label(self):
        return f"{self.os} / Julia {self.julia} / shard {self.shard}/{SHARD_COUNT}"

    def problem(self, text):
        self.problems.append(f"{self.label}: {text}")


def parse_log(cell, text, expected_commit):
    cell.present = True
    shard_label = f"{cell.shard}/{SHARD_COUNT}"
    headers, runners, trailers = [], [], []
    runner_index = None
    last_file_index = -1
    lines = [line.rstrip("\r") for line in text.splitlines()]
    for index, line in enumerate(lines):
        match = HEADER_RE.match(line)
        if match:
            fields = [field.split("=", 1) for field in match["fields"].split()]
            keys = [key for key, _ in fields]
            if len(keys) != len(set(keys)):
                cell.problem(f"header repeats a field: {line.strip()!r}")
            headers.append(dict(fields))
            continue
        match = RUNNER_RE.match(line)
        if match:
            runners.append(match)
            runner_index = index
            continue
        match = FILE_TIME_RE.match(line)
        if match:
            name = match["file"]
            elapsed = parse_seconds(match["elapsed"])
            if name in cell.results:
                cell.problem(f"test file {name} is reported twice in one log")
            if elapsed is None:
                cell.problem(f"test file {name} has elapsed_s={match['elapsed']}, not a finite non-negative number")
            cell.results[name] = (elapsed, match["status"])
            last_file_index = index
            continue
        match = SHARD_TIME_RE.match(line)
        if match:
            trailers.append((index, match))

    if len(headers) != 1:
        cell.problem(f"expected one PB_FULL_VALIDATION header, found {len(headers)}")
    else:
        header = headers[0]
        wanted = {
            "commit": expected_commit,
            "os": cell.os,
            "julia": cell.julia,
            "shard": shard_label,
        }
        for key, value in wanted.items():
            if header.get(key) != value:
                cell.problem(f"header {key}={header.get(key)!r}, expected {value!r}")

    announced_count = None
    if len(runners) != 1:
        cell.problem(f"expected one 'ProximalBase minimal tests' line, found {len(runners)}")
    else:
        runner = runners[0]
        cell.version, cell.machine = runner["version"], runner["machine"]
        announced_count = int(runner["count"])
        if not re.fullmatch(re.escape(cell.julia) + r"\.\d+", cell.version):
            cell.problem(f"ran Julia {cell.version}, expected a release of {cell.julia}.<patch>")
        if runner["tier"] != "full":
            cell.problem(f"ran tier {runner['tier']}, expected full")
        if runner["shard"] != shard_label:
            cell.problem(f"runner reports shard {runner['shard']}, expected {shard_label}")
        markers = MACHINE_MARKERS[cell.os]
        if not any(marker in cell.machine.lower() for marker in markers):
            cell.problem(f"machine {cell.machine} does not look like {cell.os} (expected {' or '.join(markers)})")
        check_announced_files(cell, lines, runner_index, announced_count)

    if len(trailers) != 1:
        cell.problem(f"expected one PB_TEST_SHARD_TIME line, found {len(trailers)}")
    else:
        index, trailer = trailers[0]
        if index < last_file_index:
            cell.problem("PB_TEST_SHARD_TIME does not follow the last PB_TEST_FILE_TIME record")
        if trailer["shard"] != shard_label:
            cell.problem(f"PB_TEST_SHARD_TIME reports shard {trailer['shard']}, expected {shard_label}")
        if announced_count is not None and int(trailer["count"]) != announced_count:
            cell.problem(
                f"PB_TEST_SHARD_TIME reports {trailer['count']} files, the runner line announced {announced_count}"
            )
        if parse_seconds(trailer["elapsed"]) is None:
            cell.problem(f"PB_TEST_SHARD_TIME has elapsed_s={trailer['elapsed']}, not a finite non-negative number")


def check_announced_files(cell, lines, runner_index, announced_count):
    """The line after the runner line lists the shard's files; compare it with the records."""
    following = [line for line in lines[runner_index + 1 :] if line.strip()]
    if not following:
        cell.problem("no file list follows the 'ProximalBase minimal tests' line")
        return
    names = following[0].split()
    if announced_count < 1:
        cell.problem("the runner line announces no files")
    if len(names) != announced_count:
        cell.problem(f"the runner line announces {announced_count} files, the list that follows has {len(names)}")
    if len(names) != len(set(names)):
        cell.problem("the announced file list repeats a file")
    timed = set(cell.results)
    listed = set(names)
    if timed != listed:
        cell.problem(
            f"announced files without a timing record: {sorted(listed - timed)}; "
            f"timing records for unannounced files: {sorted(timed - listed)}"
        )
    elif len(cell.results) != announced_count:
        cell.problem(f"the runner line announces {announced_count} files, found {len(cell.results)} timing records")


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


def load_jobs(jobs_path):
    """Read JSON lines {"name", "conclusion"}; return (records, problems)."""
    records, problems = [], []
    for number, line in enumerate(Path(jobs_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            name = record["name"]
            conclusion = record.get("conclusion")
            if not isinstance(name, str):
                raise TypeError("name is not a string")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            problems.append(f"{jobs_path} line {number} is not a job record: {error}")
            continue
        records.append((name, conclusion))
    return records, problems


def check_jobs(records):
    """Return (problems, conclusions) with `conclusions` keyed by (os, julia, shard)."""
    problems = []
    expected = {job_name(*cell): cell for cell in expected_cells()}
    seen = {}
    for name, conclusion in records:
        if not name.startswith(JOB_PREFIX):
            continue
        if name not in expected:
            problems.append(f"job {name!r} is outside the expected matrix")
            continue
        seen.setdefault(name, []).append(conclusion)
    conclusions = {}
    for name, cell in expected.items():
        found = seen.get(name, [])
        if not found:
            problems.append(f"job {name!r} is missing from this run attempt")
            conclusions[cell] = "missing"
        elif len(found) > 1:
            problems.append(f"job {name!r} appears {len(found)} times")
            conclusions[cell] = "duplicate"
        else:
            conclusions[cell] = str(found[0])
            if found[0] != "success":
                problems.append(f"job {name!r} concluded {found[0]}, expected success")
    return problems, conclusions


def pair_label(os_name, julia):
    return f"{os_name} / Julia {julia}"


def evaluate(logs_dir, selection, expected_commit, attempt, job_records):
    """Return (problems, rows) for the logs under `logs_dir` and the job records."""
    problems = []
    job_problems, job_conclusions = check_jobs(job_records)
    problems.extend(job_problems)

    found, stray = find_cell_logs(logs_dir)
    if not Path(logs_dir).is_dir():
        problems.append(f"log directory {logs_dir} does not exist; no test artifacts were downloaded")
    elif not found:
        problems.append(f"log directory {logs_dir} holds no {ARTIFACT_PREFIX}* artifact")

    expected_names = {artifact_name(*cell, attempt) for cell in expected_cells()}
    for name in found:
        if name not in expected_names:
            problems.append(f"log artifact {name} is outside the expected cells of attempt {attempt}")
    for name in stray:
        problems.append(f"unexpected entry {name} in the log directory")

    cells = {}
    for key in expected_cells():
        cell = Cell(*key)
        cells[key] = cell
        name = artifact_name(*key, attempt)
        files = found.get(name)
        if files is None:
            cell.problem(f"no log artifact {name} (missing cell)")
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
                    if elapsed is not None:
                        times[name] = max(elapsed, times.get(name, elapsed))
            slowest = sorted(times.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            rows.append(
                {
                    "os": os_name,
                    "julia": ", ".join(sorted({c.version for c in pair if c.version})) or "n/a",
                    "machine": ", ".join(sorted({c.machine for c in pair if c.machine})) or "n/a",
                    "jobs": " ".join(f"{c.shard}:{job_conclusions[(os_name, julia, c.shard)]}" for c in pair),
                    "shards": sum(
                        1
                        for c in pair
                        if c.present and not c.problems and job_conclusions[(os_name, julia, c.shard)] == "success"
                    ),
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
        "| OS | Julia | Machine | Job conclusions (shard:conclusion) | Shards completed | Files passed | Failing files | Slowest 5 files |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
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
                    row["jobs"],
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
    parser.add_argument("--jobs", required=True, help="JSON lines with the name and conclusion of each job of this run attempt")
    parser.add_argument("--attempt", required=True, help="run attempt that the artifact names must carry")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--aggregate", required=True, help="aggregate result of the matrix job (needs.test.result)")
    parser.add_argument("--summary", help="Markdown file to append the report to")
    args = parser.parse_args(argv)
    if not args.attempt.isdigit():
        parser.error("--attempt must be a number")

    try:
        selection = load_selection(args.tiers)
        records, job_file_problems = load_jobs(args.jobs)
        problems, rows = evaluate(args.logs, selection, args.expected_commit, args.attempt, records)
        problems = job_file_problems + problems
        if args.aggregate != "success":
            problems.append(f"the aggregate result of the matrix job is {args.aggregate!r}, expected 'success'")
    except (TiersError, OSError, tomllib.TOMLDecodeError) as error:
        problems, rows = [f"cannot read the validation inputs: {error}"], []
    report = render(args.expected_commit, rows, problems)
    print(report)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(report)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
