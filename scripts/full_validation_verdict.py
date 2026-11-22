#!/usr/bin/env python3
"""Verdict for the full-validation matrix: `full` (the default without `--plan`), `focused` or `release`.

Reads the cell logs of one run attempt (one directory per artifact, or one flat log file) and the
job list of that attempt.  With `--plan plan.json` the cells, mode and requested files come from
the plan; without it the script expects the 18 cells of the nightly run.  A `focused` verdict
requires the requested files to pass in each planned cell.  A `release` verdict also takes
`--ci-runs` and requires a successful CI.yml run on the commit with the jobs in `CI_REQUIRED_JOBS`.
When the script renders a failed verdict, `--signature` writes its failure signature.

The script imports only the standard library and needs Python 3.11 or newer (`tomllib`).
"""
from __future__ import annotations

import argparse
import hashlib
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
FOCUSED_PREFIX = "Focused / "
MODES = ("full", "focused", "release")
MAX_FOCUSED_FILES = 20
# The CI.yml jobs that a release needs on the candidate commit: `PR required jobs`, the
# frozen benchmark, the strict documentation build and the six Julia 1.13 test shards.
CI_EVENTS = ("push", "workflow_dispatch")
CI_REQUIRED_JOBS = (
    "PR required jobs",
    "Frozen benchmark differential / Linux",
    "Strict documentation / Linux",
    *(f"Julia 1.13 / Linux (tests {n}/6)" for n in range(1, 7)),
)
SIGNATURE_HASH_LENGTH = 12
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
# The log file of a cell, as the test step names it.
LOG_FILE_RE = re.compile(r"full-validation-(?P<os>[a-z]+-latest)-(?P<julia>1\.[0-9]+)-(?P<shard>[0-9]+)\.log")


class TiersError(Exception):
    """The tier manifest does not define a full selection."""


class PlanError(Exception):
    """The plan file is not a valid plan for this mode."""

    def __init__(self, problems, mode=None):
        super().__init__("; ".join(problems))
        self.problems = list(problems)
        self.mode = mode  # the mode the plan file declares, when it declares a known one


def expected_cells():
    return [(o, j, s) for o in OSES for j in JULIAS for s in SHARDS]


def artifact_name(os_name, julia, shard, attempt):
    return f"{ARTIFACT_PREFIX}{os_name}-{julia}-{shard}-attempt-{attempt}"


def job_name(os_name, julia, shard):
    """The job name in the exact format of the workflow's `name:` for the test job."""
    return f"{JOB_PREFIX}{os_name} / Julia {julia} (shard {shard}/{SHARD_COUNT})"


def focused_job_name(os_name, julia):
    """The job name in the exact format of the workflow's `name:` for a focused cell."""
    return f"{FOCUSED_PREFIX}{os_name} / Julia {julia}"


class Plan:
    """The cells and files that one run is expected to cover."""

    def __init__(self, mode, files, cells, shard_count):
        self.mode = mode
        self.files = list(files)
        self.cells = list(cells)
        self.shard_count = shard_count
        self.from_file = False

    @property
    def pairs(self):
        return list(dict.fromkeys(cell[:2] for cell in self.cells))

    @property
    def job_prefix(self):
        return FOCUSED_PREFIX if self.mode == "focused" else JOB_PREFIX

    def job_name(self, cell):
        if self.mode == "focused":
            return focused_job_name(cell[0], cell[1])
        return job_name(*cell)

    @property
    def title(self):
        return {"focused": "Focused validation", "release": "Release validation", "full": "Full validation"}.get(
            self.mode, "Validation"
        )

    @property
    def verdict_label(self):
        return self.title.upper()


def default_plan():
    return Plan("full", [], expected_cells(), SHARD_COUNT)


def load_plan(plan_path, selection):
    """Read and validate `plan.json`; raise PlanError with the problems found."""
    try:
        data = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except ValueError as error:
        raise PlanError([f"{plan_path} is not valid JSON: {error}"])
    if not isinstance(data, dict):
        raise PlanError([f"{plan_path} does not hold a JSON object"])
    mode = data.get("mode")
    if mode not in MODES:
        raise PlanError([f"plan mode {mode!r} is not one of {', '.join(MODES)}"])
    problems = []
    files = data.get("files")
    if not isinstance(files, list) or not all(isinstance(name, str) for name in files):
        raise PlanError(["plan files must be a list of names"], mode)
    matrix = data.get("matrix")
    include = matrix.get("include") if isinstance(matrix, dict) else None
    if not isinstance(include, list) or not include:
        raise PlanError(["plan matrix must hold a non-empty include list"], mode)
    shard_count = SHARD_COUNT if mode in ("full", "release") else 1
    cells, labels = [], {}
    for entry in include:
        if not isinstance(entry, dict):
            problems.append(f"matrix entry {entry!r} is not an object")
            continue
        os_name, julia = entry.get("os"), entry.get("julia")
        shard, shards, label = entry.get("shard"), entry.get("shards"), entry.get("label")
        if os_name not in OSES or julia not in JULIAS:
            problems.append(f"matrix entry {entry!r} names an OS or Julia version outside the allowed sets")
            continue
        integers = all(isinstance(n, int) and not isinstance(n, bool) for n in (shard, shards))
        if not integers or shards != shard_count or not 1 <= shard <= shards:
            problems.append(f"matrix entry {entry!r} needs shard in 1..{shard_count} of {shard_count} for mode {mode}")
            continue
        cell = (os_name, julia, shard)
        cells.append(cell)
        labels[cell] = label
    if problems:
        raise PlanError(problems, mode)
    plan = Plan(mode, files, cells, shard_count)
    if len(set(cells)) != len(cells):
        problems.append("the plan repeats a cell")
    if mode in ("full", "release"):
        if sorted(cells) != sorted(expected_cells()):
            problems.append(f"the cells of a {mode} plan differ from the {len(expected_cells())} expected cells")
        if files:
            problems.append(f"a {mode} plan must not list files")
    else:
        if len({cell[:2] for cell in cells}) != len(cells):
            problems.append("a focused plan repeats an OS and Julia pair")
        if not files:
            problems.append("a focused plan needs at least one file")
        if len(files) > MAX_FOCUSED_FILES:
            problems.append(f"a focused plan lists {len(files)} files, at most {MAX_FOCUSED_FILES} are allowed")
        if len(files) != len(set(files)):
            problems.append("the plan repeats a file")
        unknown = sorted(set(files) - set(selection))
        if unknown:
            problems.append(f"the plan lists files outside the full selection: {unknown}")
    for cell, label in labels.items():
        if label != plan.job_name(cell):
            problems.append(f"matrix label {label!r} differs from the job name {plan.job_name(cell)!r}")
    if problems:
        raise PlanError(problems, mode)
    return plan


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

    def __init__(self, os_name, julia, shard, shards=SHARD_COUNT):
        self.os, self.julia, self.shard, self.shards = os_name, julia, shard, shards
        self.present = False
        self.problems = []
        self.kinds = set()  # problem classes, for the failure signature
        self.results = {}  # file -> (elapsed_s or None, status)
        self.version = None
        self.machine = None

    @property
    def label(self):
        return f"{self.os} / Julia {self.julia} / shard {self.shard}/{self.shards}"

    def problem(self, text, kind="log-invalid"):
        self.problems.append(f"{self.label}: {text}")
        self.kinds.add(kind)


def parse_log(cell, text, expected_commit, header_extra=None):
    cell.present = True
    shard_label = f"{cell.shard}/{cell.shards}"
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
            **(header_extra or {}),
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
    """The first nonblank line after the runner line lists the shard's files; compare it with the records."""
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


def find_cell_logs(logs_dir, attempt):
    """Map artifact name -> list of log files; also return stray entries.

    The download action writes one directory per artifact when several artifacts
    match, and the files of a single match directly into the target directory.  A
    top-level `full-validation-<os>-<julia>-<shard>.log` is therefore filed under the
    artifact name of its cell; the download pattern selects the current attempt.
    """
    found, stray = {}, []
    root = Path(logs_dir)
    if not root.is_dir():
        return found, stray
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith(ARTIFACT_PREFIX):
            found.setdefault(entry.name, []).extend(sorted(p for p in entry.rglob("*") if p.is_file()))
            continue
        match = LOG_FILE_RE.fullmatch(entry.name) if entry.is_file() else None
        if match:
            name = artifact_name(match["os"], match["julia"], int(match["shard"]), attempt)
            found.setdefault(name, []).append(entry)
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


def check_jobs(records, plan):
    """Return (problems, conclusions, unexpected) with `conclusions` keyed by cell.

    `unexpected` is True when a job of the plan's name prefix is outside the plan.
    """
    problems = []
    expected = {plan.job_name(cell): cell for cell in plan.cells}
    seen = {}
    unexpected = False
    for name, conclusion in records:
        if not name.startswith(plan.job_prefix):
            continue
        if name not in expected:
            problems.append(f"job {name!r} is outside the expected matrix")
            unexpected = True
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
    return problems, conclusions, unexpected


def pair_label(os_name, julia):
    return f"{os_name} / Julia {julia}"


def evaluate(logs_dir, selection, expected_commit, attempt, job_records, plan=None):
    """Return (problems, rows, global_classes) for the logs under `logs_dir` and the job records.

    A row's `classes` and the global classes name the kinds of problem, for the
    failure signature; the problem texts are for people.
    """
    plan = plan or default_plan()
    problems = []
    global_classes = set()
    job_problems, job_conclusions, unexpected_job = check_jobs(job_records, plan)
    problems.extend(job_problems)
    if unexpected_job:
        global_classes.add("unexpected-job")

    found, stray = find_cell_logs(logs_dir, attempt)
    if not Path(logs_dir).is_dir():
        problems.append(f"log directory {logs_dir} does not exist; no test artifacts were downloaded")
        global_classes.add("no-logs")
    elif not found:
        problems.append(f"log directory {logs_dir} holds no {ARTIFACT_PREFIX}* artifact")
        global_classes.add("no-logs")

    expected_names = {artifact_name(*cell, attempt) for cell in plan.cells}
    for name in found:
        if name not in expected_names:
            problems.append(f"log artifact {name} is outside the expected cells of attempt {attempt}")
            global_classes.add("unexpected-artifact")
    for name in stray:
        problems.append(f"unexpected entry {name} in the log directory")
        global_classes.add("unexpected-artifact")

    # With a plan file the log header also names the mode and the shard count.
    header_extra = {"mode": plan.mode, "shards": str(plan.shard_count)} if plan.from_file else None

    cells = {}
    for key in plan.cells:
        cell = Cell(*key, plan.shard_count)
        cells[key] = cell
        name = artifact_name(*key, attempt)
        files = found.get(name)
        if files is None:
            cell.problem(f"no log artifact {name} (missing cell)", kind="missing-log")
        elif len(files) != 1:
            cell.problem(f"expected exactly one log file in {name}, found {len(files)}")
        else:
            parse_log(cell, files[0].read_text(encoding="utf-8", errors="replace"), expected_commit, header_extra)
        problems.extend(cell.problems)

    selected = set(selection)
    if plan.mode == "focused":
        target = set(plan.files)
        outside_text = "files outside the requested list ran"
        missing_text = "requested files did not run"
    else:
        target = selected
        outside_text = "files outside the full selection ran"
        missing_text = "files in the full selection did not run"
    rows = []
    for os_name, julia in plan.pairs:
        pair = [cells[c] for c in plan.cells if c[:2] == (os_name, julia)]
        label = pair_label(os_name, julia)
        owner = {}
        failing = []
        failing_names = set()
        reported = set()
        classes = set()
        for cell in pair:
            classes |= cell.kinds
            if job_conclusions[(os_name, julia, cell.shard)] != "success":
                classes.add("job-not-success")
            for name, (_, status) in cell.results.items():
                reported.add(name)
                if name in owner and owner[name] != cell.shard:
                    problems.append(
                        f"{label}: test file {name} ran in shards {owner[name]} and {cell.shard}"
                    )
                    classes.add("coverage")
                owner.setdefault(name, cell.shard)
                if status != "pass":
                    failing.append(f"{name} ({status})")
                    if name in selected:
                        failing_names.add(name)
                    else:
                        classes.add("coverage")
                    problems.append(
                        f"{label} / shard {cell.shard}/{plan.shard_count}: test file {name} has status {status}"
                    )
        unexpected = sorted(reported - target)
        if unexpected:
            problems.append(f"{label}: {outside_text}: {unexpected}")
            classes.add("coverage")
        if all(cell.present for cell in pair):
            missing = sorted(target - reported)
            if missing:
                problems.append(f"{label}: {missing_text}: {missing}")
                classes.add("coverage")
        passed = {n for c in pair for n, (_, st) in c.results.items() if st == "pass"} & target
        times = {}
        for cell in pair:
            for name, (elapsed, _) in cell.results.items():
                if elapsed is not None:
                    times[name] = max(elapsed, times.get(name, elapsed))
        slowest = sorted(times.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        rows.append(
            {
                "os": os_name,
                "julia_minor": julia,
                "julia": ", ".join(sorted({c.version for c in pair if c.version})) or "n/a",
                "machine": ", ".join(sorted({c.machine for c in pair if c.machine})) or "n/a",
                "jobs": " ".join(f"{c.shard}:{job_conclusions[(os_name, julia, c.shard)]}" for c in pair),
                "shards": sum(
                    1
                    for c in pair
                    if c.present and not c.problems and job_conclusions[(os_name, julia, c.shard)] == "success"
                ),
                "passed": len(passed),
                "expected": len(target),
                "failing": sorted(failing),
                "failing_names": sorted(failing_names),
                "classes": classes,
                "slowest": slowest,
            }
        )
    return problems, rows, global_classes


def load_ci_runs(path):
    """Read the CI.yml runs of the commit: a JSON list of run objects with their jobs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(run, dict) for run in data):
        raise ValueError("the CI run list is not a JSON list of objects")
    return data


def ci_run_rejection(run, commit):
    """Return why `run` does not qualify as the CI evidence for `commit`, or None if it does."""
    reasons = []
    if run.get("head_sha") != commit:
        reasons.append(f"head_sha {run.get('head_sha')!r} is not {commit}")
    if run.get("event") not in CI_EVENTS:
        reasons.append(f"event {run.get('event')!r} is not {' or '.join(CI_EVENTS)}")
    if run.get("conclusion") != "success":
        reasons.append(f"conclusion {run.get('conclusion')!r} is not success")
    jobs = run.get("jobs")
    jobs = jobs if isinstance(jobs, list) else []
    for required in CI_REQUIRED_JOBS:
        conclusions = [j.get("conclusion") for j in jobs if isinstance(j, dict) and j.get("name") == required]
        if not conclusions:
            reasons.append(f"job {required!r} is absent")
        elif any(c != "success" for c in conclusions):
            reasons.append(f"job {required!r} concluded {conclusions}, expected success")
    return "; ".join(reasons) or None


def check_ci_runs(runs, commit):
    """Return (problems, evidence) for the release check; `evidence` is the qualifying run or None."""
    rejected = []
    for run in runs:
        reason = ci_run_rejection(run, commit)
        if reason is None:
            return [], run
        rejected.append(f"run {run.get('id')!r}: {reason}")
    problems = [
        f"no CI.yml run on candidate commit {commit} qualifies: it needs event push or workflow_dispatch, "
        f"conclusion success and the jobs {', '.join(repr(j) for j in CI_REQUIRED_JOBS)} all successful. "
        "Dispatch CI.yml on a ref that points at the candidate commit with `gh workflow run CI.yml --ref <ref>`, "
        "wait for it, then rerun all release matrix jobs or dispatch a fresh release run."
    ]
    problems.extend(rejected[:5])
    return problems, None


def signature_lines(rows, global_classes):
    """The sorted failure-signature lines: per OS and Julia minor, the failing files or the problem classes."""
    lines = []
    for row in rows:
        if row["failing_names"]:
            what = ", ".join(row["failing_names"])
        elif row["classes"]:
            what = ", ".join(sorted(row["classes"]))
        else:
            continue
        lines.append(f"{row['os']} {row['julia_minor']}: {what}")
    if global_classes:
        lines.append("global: " + ", ".join(sorted(global_classes)))
    return sorted(lines)


def signature_text(lines):
    """The lines followed by `signature-hash: <12 hex>`, from the SHA-256 of the lines, each followed by a newline."""
    body = "".join(line + "\n" for line in lines)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:SIGNATURE_HASH_LENGTH]
    return f"{body}signature-hash: {digest}\n"


def cell_text(value):
    return str(value).replace("|", "\\|")


def render(commit, rows, problems, plan=None, notes=()):
    plan = plan or default_plan()
    lines = [
        f"## {plan.title}",
        "",
        f"Commit: `{commit}`",
        "",
    ]
    if plan.mode == "focused":
        lines += [f"Requested files: {' '.join(f'`{name}`' for name in plan.files)}", ""]
    lines += [
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
                    f"{row['shards']}/{plan.shard_count}",
                    f"{row['passed']}/{row['expected']}",
                    failing,
                    slowest,
                )
            ) + " |"
        )
    lines.append("")
    lines.extend(notes)
    if notes:
        lines.append("")
    if problems:
        lines.append(f"{plan.verdict_label}: FAIL ({len(problems)} problems)")
        lines.append("")
        lines.extend(f"- {problem}" for problem in problems)
    else:
        lines.append(f"{plan.verdict_label}: PASS")
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
    parser.add_argument("--plan", help="plan.json of this run; without it the script checks the full validation")
    parser.add_argument("--ci-runs", help="JSON list of the CI.yml runs on the commit with their jobs; required in release mode")
    parser.add_argument("--signature", help="file to write the failure signature to when there are problems")
    parser.add_argument("--summary", help="Markdown file to append the report to")
    args = parser.parse_args(argv)
    if not args.attempt.isdigit():
        parser.error("--attempt must be a number")

    plan = default_plan()
    notes = []
    try:
        selection = load_selection(args.tiers)
        if args.plan:
            plan = load_plan(args.plan, selection)
            plan.from_file = True
        if args.ci_runs and plan.mode != "release":
            parser.error("--ci-runs is only used in release mode")
        if plan.mode == "release" and not args.ci_runs:
            parser.error("release mode needs --ci-runs")
        records, job_file_problems = load_jobs(args.jobs)
        problems, rows, classes = evaluate(args.logs, selection, args.expected_commit, args.attempt, records, plan)
        problems = job_file_problems + problems
        if job_file_problems:
            classes.add("job-list-invalid")
        if args.aggregate != "success":
            problems.append(f"the aggregate result of the matrix job is {args.aggregate!r}, expected 'success'")
            classes.add("aggregate-not-success")
        if plan.mode == "release":
            ci_problems, evidence = check_ci_runs(load_ci_runs(args.ci_runs), args.expected_commit)
            problems.extend(ci_problems)
            if evidence is None:
                classes.add("ci-run-missing")
            else:
                attempt_note = f" (attempt {evidence['run_attempt']})" if evidence.get("run_attempt") else ""
                notes.append(f"CI.yml run on this commit: {evidence.get('html_url') or evidence.get('id')}{attempt_note}")
        # A problem class that adds nothing to a more specific one is dropped from the signature.
        if any(row["failing_names"] or row["classes"] for row in rows):
            classes.discard("aggregate-not-success")
    except PlanError as error:
        # An invalid plan that still names its mode reports under that mode's label; one that
        # does not gets the neutral label.
        plan = Plan(error.mode or "unknown", [], [], 1 if error.mode == "focused" else SHARD_COUNT)
        problems, rows, classes = [f"invalid plan: {p}" for p in error.problems], [], {"plan-invalid"}
    except (TiersError, OSError, ValueError, tomllib.TOMLDecodeError) as error:
        if args.plan and not plan.from_file:
            plan = Plan("unknown", [], [], SHARD_COUNT)
        problems, rows, classes = [f"cannot read the validation inputs: {error}"], [], {"inputs-unreadable"}
    report = render(args.expected_commit, rows, problems, plan, notes)
    print(report)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(report)
    if args.signature and problems:
        lines = signature_lines(rows, classes) or ["global: unclassified"]
        Path(args.signature).write_text(signature_text(lines), encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
