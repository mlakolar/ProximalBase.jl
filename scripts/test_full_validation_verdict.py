#!/usr/bin/env python3
"""Tests for the full-validation verdict script, on synthetic log trees."""
from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import full_validation_plan as plan_script
import full_validation_verdict as v

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "full-validation.yml"
COMMIT = "a" * 40
ATTEMPT = "2"
TIERS_TEXT = """\
[tiers]
quick = ["alpha", "beta"]
standard = ["gamma"]
thorough = ["delta", "epsilon"]
full = ["zeta"]

[not_tests]
files = ["helper"]
"""
SELECTION = ["alpha", "beta", "delta", "epsilon", "gamma", "zeta"]
MACHINES = {
    "ubuntu-latest": "x86_64-linux-gnu",
    "windows-latest": "x86_64-w64-mingw32",
    "macos-latest": "aarch64-apple-darwin24.0.0",
}


def shard_files(shard, names=SELECTION):
    return names[shard - 1 :: 3]


def log_text(os_name, julia, shard, files, *, commit=COMMIT, header_os=None, header_shard=None,
             version=None, statuses=None, announced=None, count=None, machine=None,
             trailer=True, trailer_count=None, elapsed="1.5", extra_header=None, headers=1,
             runner_count=1, shards=3, mode=None, runner_shard=None):
    """One cell log.  `files` are the timing records; `announced` is the printed file list."""
    statuses = statuses or {}
    version = version or f"{julia}.9"
    announced = files if announced is None else announced
    count = len(announced) if count is None else count
    header = (
        f"PB_FULL_VALIDATION commit={commit} os={header_os or os_name} julia={julia} "
        f"shard={header_shard or f'{shard}/{shards}'}"
        + (f" mode={mode} shards={shards}" if mode else "")
    )
    runner = (
        f"ProximalBase minimal tests: tier full; shard {runner_shard or f'{shard}/{shards}'}; {count} files; "
        f"Julia {version} on {machine or MACHINES[os_name]}"
    )
    lines = [header + (extra_header or "")] * headers
    lines += ["Julia Version " + version]
    lines += [runner, "  " + " ".join(announced)] * runner_count
    for i, name in enumerate(files):
        lines.append(
            f"PB_TEST_FILE_TIME file={name} elapsed_s={elapsed if i == 0 else f'{10 + i}.5'} "
            f"status={statuses.get(name, 'pass')}"
        )
    if trailer:
        lines.append(f"PB_TEST_SHARD_TIME shard={shard}/{shards} files={len(files) if trailer_count is None else trailer_count} elapsed_s=99.0")
    return "\r\n".join(lines) + "\r\n"


class Harness(unittest.TestCase):
    """Writes synthetic log trees, job lists and plans, and runs the verdict script on them."""

    label = "FULL VALIDATION"
    aggregate = "success"
    plan_path = None
    ci_runs_path = None
    signature_path = None
    log_defaults = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.tiers = self.root / "tiers.toml"
        self.tiers.write_text(TIERS_TEXT)
        self.jobs = self.root / "jobs.jsonl"
        self.cells = v.expected_cells()
        self.job_name = lambda cell: v.job_name(*cell)
        self.default_files = shard_files
        self.write_jobs()

    def configure(self, **inputs):
        """Use the plan script to write plan.json for these inputs and adopt its cells."""
        plan = plan_script.make_plan(inputs, COMMIT, SELECTION)
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(plan))
        self.cells = [(c["os"], c["julia"], c["shard"]) for c in plan["matrix"]["include"]]
        labels = {(c["os"], c["julia"], c["shard"]): c["label"] for c in plan["matrix"]["include"]}
        self.job_name = labels.__getitem__
        shards = plan["matrix"]["include"][0]["shards"]
        self.log_defaults = {"shards": shards, "mode": plan["mode"]}
        if plan["mode"] == "focused":
            self.default_files = lambda shard: list(plan["files"])
        self.write_jobs()
        return plan

    def write_jobs(self, conclusions=None, skip=(), extra=(), repeat=()):
        conclusions = conclusions or {}
        records = []
        for cell in self.cells:
            if cell in skip:
                continue
            record = {"name": self.job_name(cell), "conclusion": conclusions.get(cell, "success")}
            records += [record] * (2 if cell in repeat else 1)
        records += [{"name": "Full validation verdict", "conclusion": None}, {"name": "Other job", "conclusion": "failure"}]
        records += list(extra)
        self.jobs.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def write_cell(self, os_name, julia, shard, text=None, files=None, attempt=ATTEMPT, **kw):
        directory = self.logs / v.artifact_name(os_name, julia, shard, attempt)
        directory.mkdir(parents=True, exist_ok=True)
        if files is None:
            files = self.default_files(shard)
        kw = {**self.log_defaults, **kw}
        name = f"full-validation-{os_name}-{julia}-{shard}.log"
        (directory / name).write_text(text if text is not None else log_text(os_name, julia, shard, files, **kw))

    def write_all(self, overrides=None, skip=()):
        overrides = overrides or {}
        for cell in self.cells:
            if cell in skip:
                continue
            self.write_cell(*cell, **overrides.get(cell, {}))

    def run_verdict(self, summary=None):
        argv = ["--logs", str(self.logs), "--tiers", str(self.tiers), "--jobs", str(self.jobs),
                "--attempt", ATTEMPT, "--expected-commit", COMMIT, "--aggregate", self.aggregate]
        if summary:
            argv += ["--summary", str(summary)]
        if self.plan_path:
            argv += ["--plan", str(self.plan_path)]
        if self.ci_runs_path:
            argv += ["--ci-runs", str(self.ci_runs_path)]
        if self.signature_path:
            argv += ["--signature", str(self.signature_path)]
        out = io.StringIO()
        with redirect_stdout(out):
            code = v.main(argv)
        return code, out.getvalue()

    def assertFails(self, *fragments):
        code, out = self.run_verdict()
        self.assertEqual(code, 1, out)
        self.assertIn(f"{self.label}: FAIL", out)
        for fragment in fragments:
            self.assertIn(fragment, out)
        return out

    def only_cell_problem(self, overrides, *fragments):
        self.write_all(overrides)
        return self.assertFails(*fragments)

    def logs_reset(self):
        for path in sorted(self.logs.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()


class VerdictTests(Harness):
    def test_expected_cells(self):
        self.assertEqual(len(v.expected_cells()), 18)

    def test_all_cells_pass(self):
        self.write_all()
        summary = self.root / "summary.md"
        summary.write_text("earlier\n")
        code, out = self.run_verdict(summary)
        self.assertEqual(code, 0, out)
        self.assertIn("FULL VALIDATION: PASS", out)
        self.assertIn(COMMIT, out)
        self.assertIn("| ubuntu-latest | 1.10.9 | x86_64-linux-gnu | 1:success 2:success 3:success | 3/3 | 6/6 | none |", out)
        text = summary.read_text()
        self.assertTrue(text.startswith("earlier\n"))
        self.assertIn("FULL VALIDATION: PASS", text)

    def test_missing_cell(self):
        self.write_all(skip={("windows-latest", "1.13", 2)})
        self.assertFails("windows-latest / Julia 1.13 / shard 2/3: no log artifact")

    def test_extra_cell(self):
        self.write_all()
        (self.logs / "full-validation-log-freebsd-1.10-1-attempt-2").mkdir()
        (self.logs / "full-validation-log-freebsd-1.10-1-attempt-2" / "x.log").write_text("x")
        self.assertFails("full-validation-log-freebsd-1.10-1-attempt-2 is outside the expected cells")

    def test_stale_attempt_directory(self):
        self.write_all(skip={("macos-latest", "1.10", 3)})
        self.write_cell("macos-latest", "1.10", 3, attempt="1")
        self.assertFails("macos-latest / Julia 1.10 / shard 3/3: no log artifact",
                         "full-validation-log-macos-latest-1.10-3-attempt-1 is outside the expected cells of attempt 2")

    def test_stray_entry(self):
        self.write_all()
        (self.logs / "notes.txt").write_text("x")
        self.assertFails("unexpected entry notes.txt")

    def test_commit_mismatch(self):
        self.only_cell_problem({("macos-latest", "1.10", 1): {"commit": "b" * 40}}, "header commit='" + "b" * 40 + "'")

    def test_header_os_and_shard_mismatch(self):
        out = self.only_cell_problem({
            ("ubuntu-latest", "1.10", 1): {"header_os": "windows-latest"},
            ("ubuntu-latest", "1.10", 2): {"header_shard": "1/3"},
        }, "header os='windows-latest'", "header shard='1/3'")
        self.assertIn("2 problems", out)

    def test_missing_header(self):
        self.write_all()
        path = next((self.logs / v.artifact_name("ubuntu-latest", "1.10", 1, ATTEMPT)).iterdir())
        path.write_text("\n".join(l for l in path.read_text().splitlines() if not l.startswith("PB_FULL")))
        self.assertFails("expected one PB_FULL_VALIDATION header, found 0")

    def test_duplicate_and_conflicting_headers(self):
        self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"headers": 2}}, "found 2")
        self.logs_reset()
        self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"extra_header": " commit=" + "c" * 40}}, "header repeats a field")

    def test_machine_must_match_os(self):
        self.only_cell_problem({("windows-latest", "1.10", 1): {"machine": "x86_64-linux-gnu"}},
                               "machine x86_64-linux-gnu does not look like windows-latest")

    def test_failing_file(self):
        out = self.only_cell_problem({("ubuntu-latest", "1.13", 3): {"statuses": {"zeta": "fail", "delta": "error"}}},
                                     "test file zeta has status fail", "test file delta has status error")
        self.assertIn("delta (error), zeta (fail)", out)
        self.assertIn("| 4/6 |", out)

    def test_missing_file_in_one_pair(self):
        files = [n for n in shard_files(2) if n != "gamma"]
        self.only_cell_problem({("windows-latest", "1.10", 2): {"files": files}},
                               "windows-latest / Julia 1.10: files in the full selection did not run: ['gamma']")

    def test_file_in_two_shards(self):
        files = shard_files(2) + ["alpha"]
        self.only_cell_problem({("macos-latest", "1.13", 2): {"files": files}},
                               "macos-latest / Julia 1.13: test file alpha ran in shards 1 and 2")

    def test_unexpected_file(self):
        files = shard_files(1) + ["omega"]
        self.only_cell_problem({("macos-latest", "1.13", 1): {"files": files}},
                               "files outside the full selection ran: ['omega']")

    def test_duplicate_report_in_one_log(self):
        files = shard_files(1) + [shard_files(1)[0]]
        self.only_cell_problem({("ubuntu-latest", "1.13", 1): {"files": files}}, "is reported twice in one log")

    def test_wrong_julia_minor(self):
        self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"version": "1.11.2"}},
                               "ran Julia 1.11.2, expected a release of 1.10.<patch>")

    def test_julia_version_shapes(self):
        for version in ("1.100.0", "1.10.x", "1.10.9-rc1", "1.10", "1.10.9+1"):
            with self.subTest(version=version):
                self.logs_reset()
                self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"version": version}},
                                       f"ran Julia {version}, expected a release of 1.10.<patch>")

    def test_elapsed_must_be_finite_and_non_negative(self):
        for elapsed in ("nan", "inf", "-1.0", "abc"):
            with self.subTest(elapsed=elapsed):
                self.logs_reset()
                self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"elapsed": elapsed}},
                                       "not a finite non-negative number")

    def test_two_log_files_in_one_cell(self):
        self.write_all()
        directory = self.logs / v.artifact_name("ubuntu-latest", "1.10", 1, ATTEMPT)
        (directory / "second.log").write_text("x")
        self.assertFails("expected exactly one log file")

    def test_empty_logs_directory(self):
        out = self.assertFails("holds no full-validation-log-* artifact")
        self.assertIn("(19 problems)", out)

    def test_absent_logs_directory(self):
        self.logs.rmdir()
        self.assertFails("does not exist")

    # Per-shard consistency.
    def test_timing_record_moved_to_sibling_shard(self):
        moved = shard_files(1)[0]
        out = self.only_cell_problem({
            ("ubuntu-latest", "1.10", 1): {"files": shard_files(1)[1:], "announced": shard_files(1)},
            ("ubuntu-latest", "1.10", 2): {"files": shard_files(2) + [moved], "announced": shard_files(2)},
        }, f"announced files without a timing record: ['{moved}']",
           f"timing records for unannounced files: ['{moved}']")
        self.assertNotIn("did not run", out)

    def test_missing_trailer(self):
        self.only_cell_problem({("macos-latest", "1.13", 3): {"trailer": False}}, "expected one PB_TEST_SHARD_TIME line, found 0")

    def test_trailer_before_last_file_record(self):
        self.write_all()
        path = next((self.logs / v.artifact_name("ubuntu-latest", "1.10", 1, ATTEMPT)).iterdir())
        lines = path.read_text().splitlines()
        lines.insert(-2, lines.pop())
        path.write_text("\n".join(lines) + "\n")
        self.assertFails("PB_TEST_SHARD_TIME does not follow the last PB_TEST_FILE_TIME record")

    def test_trailer_count_mismatch(self):
        self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"trailer_count": 5}},
                               "PB_TEST_SHARD_TIME reports 5 files, the runner line announced 2")

    def test_announced_count_mismatch(self):
        self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"count": 3, "trailer_count": 3}},
                               "announces 3 files, the list that follows has 2")

    def test_announced_list_repeats_a_file(self):
        files = shard_files(1)
        self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"announced": files + files[:1], "count": 3}},
                               "the announced file list repeats a file")

    def test_two_runner_lines(self):
        self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"runner_count": 2}},
                               "expected one 'ProximalBase minimal tests' line, found 2")

    def test_no_file_list(self):
        text = log_text("ubuntu-latest", "1.10", 1, [])
        text = "\r\n".join(l for l in text.splitlines() if l.strip() != "")
        self.write_all({("ubuntu-latest", "1.10", 1): {"text": text}})
        self.assertFails("announces no files")

    def test_runner_line_without_list(self):
        lines = log_text("ubuntu-latest", "1.10", 1, shard_files(1)).splitlines()
        lines = [l for l in lines if not l.startswith("  ")]
        self.write_all({("ubuntu-latest", "1.10", 1): {"text": "\n".join(lines) + "\n"}})
        self.assertFails("the list that follows has")

    # Job outcomes.
    def test_job_failure_with_complete_logs(self):
        self.write_all()
        self.write_jobs({("windows-latest", "1.10", 2): "cancelled", ("macos-latest", "1.13", 1): "failure"})
        out = self.assertFails(
            "job 'Full / windows-latest / Julia 1.10 (shard 2/3)' concluded cancelled, expected success",
            "job 'Full / macos-latest / Julia 1.13 (shard 1/3)' concluded failure, expected success",
        )
        self.assertIn("1:success 2:cancelled 3:success | 2/3 |", out)

    def test_aggregate_result_must_be_success(self):
        self.write_all()
        for result in ("failure", "cancelled", "skipped", ""):
            with self.subTest(result=result):
                self.aggregate = result
                self.assertFails(f"the aggregate result of the matrix job is {result!r}, expected 'success'")

    def test_missing_job(self):
        self.write_all()
        self.write_jobs(skip={("ubuntu-latest", "1.13", 3)})
        self.assertFails("job 'Full / ubuntu-latest / Julia 1.13 (shard 3/3)' is missing from this run attempt")

    def test_duplicate_job(self):
        self.write_all()
        self.write_jobs(repeat={("ubuntu-latest", "1.13", 3)})
        self.assertFails("job 'Full / ubuntu-latest / Julia 1.13 (shard 3/3)' appears 2 times")

    def test_unexpected_full_job(self):
        self.write_all()
        self.write_jobs(extra=[{"name": "Full / freebsd / Julia 1.10 (shard 1/3)", "conclusion": "success"}])
        self.assertFails("job 'Full / freebsd / Julia 1.10 (shard 1/3)' is outside the expected matrix")

    def test_other_jobs_are_ignored(self):
        self.write_all()
        self.write_jobs(extra=[{"name": "Another workflow job", "conclusion": "failure"}])
        code, out = self.run_verdict()
        self.assertEqual(code, 0, out)

    def test_malformed_jobs_file(self):
        self.write_all()
        self.jobs.write_text(self.jobs.read_text() + "not json\n")
        self.assertFails("is not a job record")

    def test_absent_jobs_file(self):
        self.write_all()
        self.jobs.unlink()
        self.assertFails("cannot read the validation inputs")

    # Tier validation.
    def tiers_fail(self, text, fragment):
        self.tiers.write_text(text)
        self.write_all()
        self.assertFails(fragment)

    def test_duplicate_test_name_in_tiers(self):
        self.tiers_fail(TIERS_TEXT.replace('full = ["zeta"]', 'full = ["zeta", "alpha"]'), "duplicate test file in manifest")

    def test_missing_tier_in_tiers(self):
        self.tiers_fail(TIERS_TEXT.replace('full = ["zeta"]\n', ""), "all four tiers")

    def test_four_empty_tiers(self):
        self.tiers_fail("[tiers]\nquick = []\nstandard = []\nthorough = []\nfull = []\n", "fewer than the 3 shards")

    def test_two_file_selection(self):
        self.tiers_fail('[tiers]\nquick = ["a"]\nstandard = []\nthorough = []\nfull = ["b"]\n', "fewer than the 3 shards")

    def test_scalar_tier(self):
        self.tiers_fail(TIERS_TEXT.replace('quick = ["alpha", "beta"]', 'quick = "alpha"'), "tier quick must be an array")

    def test_non_string_entry(self):
        self.tiers_fail(TIERS_TEXT.replace('standard = ["gamma"]', 'standard = ["gamma", 3]'), "not a non-empty name: 3")

    def test_empty_string_entry(self):
        self.tiers_fail(TIERS_TEXT.replace('standard = ["gamma"]', 'standard = ["gamma", ""]'), "not a non-empty name: ''")


class FocusedTests(Harness):
    label = "FOCUSED VALIDATION"

    def setUp(self):
        super().setUp()
        self.configure(mode="focused", files="alpha gamma")

    def test_pass_in_every_cell(self):
        self.write_all()
        code, out = self.run_verdict()
        self.assertEqual(code, 0, out)
        self.assertIn("FOCUSED VALIDATION: PASS", out)
        self.assertNotIn("FULL VALIDATION", out)
        self.assertIn("## Focused validation", out)
        self.assertIn("Requested files: `alpha` `gamma`", out)
        self.assertIn("| windows-latest | 1.13.9 | x86_64-w64-mingw32 | 1:success | 1/1 | 2/2 | none |", out)
        self.assertEqual(len(self.cells), 6)

    def test_restricted_platform_and_version(self):
        self.configure(mode="focused", files="beta", platforms="windows", julia="1.13")
        self.write_all()
        code, out = self.run_verdict()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cells, [("windows-latest", "1.13", 1)])
        self.assertIn("Focused / windows-latest / Julia 1.13", self.jobs.read_text())

    def test_missing_requested_file(self):
        self.only_cell_problem({("macos-latest", "1.10", 1): {"files": ["alpha"], "announced": ["alpha"]}},
                               "macos-latest / Julia 1.10: requested files did not run: ['gamma']")

    def test_extra_file(self):
        out = self.only_cell_problem({("ubuntu-latest", "1.13", 1): {"files": ["alpha", "gamma", "beta"]}},
                                     "ubuntu-latest / Julia 1.13: files outside the requested list ran: ['beta']")
        self.assertNotIn("FULL VALIDATION", out)

    def test_failing_requested_file(self):
        out = self.only_cell_problem({("windows-latest", "1.10", 1): {"statuses": {"gamma": "fail"}}},
                                     "test file gamma has status fail")
        self.assertIn("| 1/2 |", out)

    def test_shard_mismatch(self):
        self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"runner_shard": "1/3"}},
                               "runner reports shard 1/3, expected 1/1")

    def test_header_mode_mismatch(self):
        self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"mode": "full"}},
                               "header mode='full', expected 'focused'")

    def test_header_without_mode(self):
        self.log_defaults = {"shards": 1}
        self.write_all()
        self.assertFails("header mode=None, expected 'focused'", "header shards=None, expected '1'")

    def test_header_shard_count_mismatch(self):
        self.only_cell_problem({("ubuntu-latest", "1.10", 1): {"shards": 3}}, "header shard='1/3', expected '1/1'")

    def test_job_conclusion_and_name(self):
        self.write_all()
        self.write_jobs({("macos-latest", "1.13", 1): "failure"})
        self.assertFails("job 'Focused / macos-latest / Julia 1.13' concluded failure, expected success")
        self.write_jobs(skip={("macos-latest", "1.13", 1)}, extra=[{"name": "Full / macos-latest / Julia 1.13 (shard 1/3)", "conclusion": "success"}])
        out = self.assertFails("job 'Focused / macos-latest / Julia 1.13' is missing from this run attempt")
        self.assertNotIn("outside the expected matrix", out)

    def test_missing_log(self):
        self.write_all(skip={("macos-latest", "1.13", 1)})
        self.assertFails("macos-latest / Julia 1.13 / shard 1/1: no log artifact")

    def test_requested_name_outside_the_selection(self):
        data = json.loads(self.plan_path.read_text())
        data["files"] = ["alpha", "omega"]
        self.plan_path.write_text(json.dumps(data))
        self.write_all()
        self.assertFails("invalid plan: the plan lists files outside the full selection: ['omega']")

    def test_focused_plan_needs_files(self):
        data = json.loads(self.plan_path.read_text())
        data["files"] = []
        self.plan_path.write_text(json.dumps(data))
        self.write_all()
        self.assertFails("a focused plan needs at least one file")

    def test_full_label_in_a_focused_plan(self):
        data = json.loads(self.plan_path.read_text())
        data["matrix"]["include"][0]["label"] = "Full / ubuntu-latest / Julia 1.10 (shard 1/3)"
        self.plan_path.write_text(json.dumps(data))
        self.write_all()
        self.assertFails("differs from the job name 'Focused / ubuntu-latest / Julia 1.10'")

    def test_focused_plan_with_three_shards(self):
        data = json.loads(self.plan_path.read_text())
        for entry in data["matrix"]["include"]:
            entry["shards"] = 3
        self.plan_path.write_text(json.dumps(data))
        self.write_all()
        self.assertFails("needs shard in 1..1 of 1 for mode focused")


class PlanFileTests(Harness):
    """Plan files in full mode, and plans that are not valid."""

    def setUp(self):
        super().setUp()
        self.configure(mode="full")

    def test_full_plan_passes_and_headers_carry_mode(self):
        self.write_all()
        code, out = self.run_verdict()
        self.assertEqual(code, 0, out)
        self.assertIn("FULL VALIDATION: PASS", out)
        self.assertIn(" mode=full shards=3", next((self.logs / v.artifact_name("ubuntu-latest", "1.10", 1, ATTEMPT)).iterdir()).read_text())

    def test_header_must_carry_mode_with_a_plan(self):
        self.log_defaults = {}
        self.write_all()
        self.assertFails("header mode=None, expected 'full'")

    def test_plan_with_fewer_cells(self):
        data = json.loads(self.plan_path.read_text())
        data["matrix"]["include"].pop()
        self.plan_path.write_text(json.dumps(data))
        self.write_all()
        self.assertFails("invalid plan: the cells of a full plan differ from the 18 expected cells")

    def test_full_plan_with_files(self):
        data = json.loads(self.plan_path.read_text())
        data["files"] = ["alpha"]
        self.plan_path.write_text(json.dumps(data))
        self.write_all()
        self.assertFails("a full plan must not list files")

    def test_unknown_mode_and_unreadable_plan(self):
        data = json.loads(self.plan_path.read_text())
        data["mode"] = "nightly"
        self.plan_path.write_text(json.dumps(data))
        self.write_all()
        self.label = "VALIDATION"  # without a known mode the report does not claim to be a full validation
        out = self.assertFails("plan mode 'nightly' is not one of full, focused, release")
        self.assertNotIn("FULL VALIDATION", out)
        self.plan_path.write_text("{")
        self.assertFails("is not valid JSON")
        self.plan_path.unlink()
        out = self.assertFails("cannot read the validation inputs")
        self.assertNotIn("FULL VALIDATION", out)

    def test_plan_with_an_unknown_os(self):
        data = json.loads(self.plan_path.read_text())
        data["matrix"]["include"][0]["os"] = "freebsd"
        self.plan_path.write_text(json.dumps(data))
        self.write_all()
        self.assertFails("outside the allowed sets")


def ci_run(**overrides):
    """A CI.yml run that qualifies as release evidence, with its jobs; overrides replace fields."""
    run = {
        "id": 101, "run_attempt": 1, "head_sha": COMMIT, "event": "push", "conclusion": "success",
        "html_url": "https://github.com/example/repo/actions/runs/101",
        "jobs": [{"name": name, "conclusion": "success"} for name in v.CI_REQUIRED_JOBS]
                + [{"name": "Change classification / Linux", "conclusion": "success"}],
    }
    run.update(overrides)
    return run


def with_job(run, name, conclusion):
    jobs = [dict(job) for job in run["jobs"] if job["name"] != name]
    if conclusion is not None:
        jobs.append({"name": name, "conclusion": conclusion})
    return {**run, "jobs": jobs}


class ReleaseTests(Harness):
    label = "RELEASE VALIDATION"

    def setUp(self):
        super().setUp()
        self.configure(mode="release", expected_sha=COMMIT)
        self.ci_runs_path = self.root / "ci_runs.json"
        self.write_ci_runs([ci_run()])

    def write_ci_runs(self, runs):
        self.ci_runs_path.write_text(json.dumps(runs))

    def test_pass_with_a_qualifying_run(self):
        self.write_all()
        code, out = self.run_verdict()
        self.assertEqual(code, 0, out)
        self.assertIn("RELEASE VALIDATION: PASS", out)
        self.assertIn("## Release validation", out)
        self.assertIn("CI.yml run on this commit: https://github.com/example/repo/actions/runs/101 (attempt 1)", out)
        self.assertNotIn("FULL VALIDATION", out)

    def test_workflow_dispatch_run_qualifies(self):
        self.write_ci_runs([ci_run(event="workflow_dispatch")])
        self.write_all()
        self.assertEqual(self.run_verdict()[0], 0)

    def test_one_good_run_among_rejected_ones(self):
        self.write_ci_runs([ci_run(event="pull_request"), ci_run(head_sha="b" * 40), ci_run(id=7)])
        self.write_all()
        self.assertEqual(self.run_verdict()[0], 0)

    def test_no_ci_run(self):
        self.write_ci_runs([])
        self.write_all()
        self.assertFails("no CI.yml run on candidate commit " + COMMIT + " qualifies",
                         "gh workflow run CI.yml --ref <ref>",
                         "rerun all release matrix jobs or dispatch a fresh release run")

    def test_ci_run_on_another_commit(self):
        self.write_ci_runs([ci_run(head_sha="b" * 40)])
        self.write_all()
        self.assertFails("run 101: head_sha '" + "b" * 40 + "' is not " + COMMIT)

    def test_skipped_benchmark(self):
        self.write_ci_runs([with_job(ci_run(), "Frozen benchmark differential / Linux", "skipped")])
        self.write_all()
        self.assertFails("job 'Frozen benchmark differential / Linux' concluded ['skipped'], expected success")

    def test_failed_docs_job(self):
        self.write_ci_runs([with_job(ci_run(), "Strict documentation / Linux", "failure")])
        self.write_all()
        self.assertFails("job 'Strict documentation / Linux' concluded ['failure'], expected success")

    def test_missing_ci_job(self):
        self.write_ci_runs([with_job(ci_run(), "Julia 1.13 / Linux (tests 2/6)", None), with_job(ci_run(id=2), "PR required jobs", None)])
        self.write_all()
        self.assertFails("job 'Julia 1.13 / Linux (tests 2/6)' is absent", "job 'PR required jobs' is absent")

    def test_pull_request_run_does_not_qualify(self):
        self.write_ci_runs([ci_run(event="pull_request")])
        self.write_all()
        self.assertFails("event 'pull_request' is not push or workflow_dispatch")

    def test_unsuccessful_run(self):
        self.write_ci_runs([ci_run(conclusion="failure")])
        self.write_all()
        self.assertFails("conclusion 'failure' is not success")

    def test_matrix_failure_fails_the_release(self):
        self.write_all({("windows-latest", "1.10", 2): {"statuses": {"beta": "fail"}}})
        out = self.assertFails("test file beta has status fail")
        self.assertNotIn("FULL VALIDATION", out)

    def test_malformed_ci_runs(self):
        self.ci_runs_path.write_text("{")
        self.write_all()
        self.assertFails("cannot read the validation inputs")
        self.ci_runs_path.write_text('{"id": 1}')
        self.assertFails("not a JSON list of objects")

    def test_release_needs_the_ci_runs_option(self):
        self.ci_runs_path = None
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.run_verdict()

    def test_ci_runs_option_is_for_release_only(self):
        self.configure(mode="full")
        self.ci_runs_path = self.root / "ci_runs.json"
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.run_verdict()

    def test_release_plan_with_a_missing_cell(self):
        data = json.loads(self.plan_path.read_text())
        data["matrix"]["include"].pop()
        self.plan_path.write_text(json.dumps(data))
        self.write_all()
        self.assertFails("the cells of a release plan differ from the 18 expected cells")


class SignatureTests(Harness):
    def setUp(self):
        super().setUp()
        self.signature_path = self.root / "signature.txt"

    def signature(self):
        return self.signature_path.read_text()

    def test_no_signature_when_passing(self):
        self.write_all()
        self.assertEqual(self.run_verdict()[0], 0)
        self.assertFalse(self.signature_path.exists())

    def test_failing_files_sorted_with_hash(self):
        self.write_all({
            ("windows-latest", "1.13", 3): {"statuses": {"zeta": "fail", "delta": "error"}},
            ("macos-latest", "1.10", 1): {"statuses": {"epsilon": "fail", "alpha": "fail"}},
        })
        self.assertEqual(self.run_verdict()[0], 1)
        lines = ["macos-latest 1.10: alpha, epsilon", "windows-latest 1.13: delta, zeta"]
        body = "".join(line + "\n" for line in lines)
        digest = hashlib.sha256(body.encode()).hexdigest()[:12]
        self.assertEqual(self.signature(), body + f"signature-hash: {digest}\n")

    def test_signature_is_deterministic_and_ignores_the_failure_detail(self):
        self.write_all({("windows-latest", "1.13", 3): {"statuses": {"zeta": "fail"}}})
        self.run_verdict()
        first = self.signature()
        self.logs_reset()
        self.write_all({("windows-latest", "1.13", 3): {"statuses": {"zeta": "error"}, "elapsed": "9.5"}})
        self.run_verdict()
        self.assertEqual(self.signature(), first)

    def test_other_failures_change_the_hash(self):
        self.write_all({("windows-latest", "1.13", 3): {"statuses": {"zeta": "fail"}}})
        self.run_verdict()
        first = self.signature().splitlines()[-1]
        self.logs_reset()
        self.write_all({("windows-latest", "1.13", 3): {"statuses": {"delta": "fail"}}})
        self.run_verdict()
        self.assertNotEqual(self.signature().splitlines()[-1], first)

    def test_job_failure_without_a_failing_file(self):
        self.write_all()
        self.write_jobs({("windows-latest", "1.10", 2): "cancelled"})
        self.run_verdict()
        self.assertIn("windows-latest 1.10: job-not-success\n", self.signature())

    def test_missing_cell_class(self):
        self.write_all(skip={("macos-latest", "1.13", 2)})
        self.run_verdict()
        self.assertIn("macos-latest 1.13: missing-log\n", self.signature())

    def test_failing_file_hides_the_job_class(self):
        self.write_all({("ubuntu-latest", "1.10", 1): {"statuses": {"alpha": "fail"}}})
        self.write_jobs({("ubuntu-latest", "1.10", 1): "failure"})
        self.aggregate = "failure"
        self.run_verdict()
        text = self.signature()
        self.assertIn("ubuntu-latest 1.10: alpha\n", text)
        self.assertNotIn("job-not-success", text)
        self.assertNotIn("aggregate", text)

    def test_names_outside_the_selection_stay_out(self):
        self.write_all({("ubuntu-latest", "1.10", 1): {"files": ["alpha", "epsilon", "evil_name"], "statuses": {"evil_name": "fail"}}})
        self.run_verdict()
        text = self.signature()
        self.assertNotIn("evil", text)
        self.assertIn("ubuntu-latest 1.10: coverage\n", text)

    def test_aggregate_alone(self):
        self.write_all()
        self.aggregate = "cancelled"
        self.run_verdict()
        self.assertIn("global: aggregate-not-success\n", self.signature())

    def test_global_classes(self):
        self.run_verdict()
        text = self.signature()
        self.assertIn("global: no-logs\n", text)

    def test_unreadable_inputs_and_invalid_plan(self):
        self.write_all()
        self.jobs.unlink()
        self.run_verdict()
        self.assertTrue(self.signature().startswith("global: inputs-unreadable\n"))
        self.configure(mode="focused", files="alpha")
        data = json.loads(self.plan_path.read_text())
        data["mode"] = "x"
        self.plan_path.write_text(json.dumps(data))
        self.run_verdict()
        self.assertTrue(self.signature().startswith("global: plan-invalid\n"))

    def test_signature_lines_match_the_report_job_pattern(self):
        workflow = WORKFLOW.read_text()
        pattern = re.search(r"^\s+pattern='(.*)'$", workflow, re.M).group(1)
        # grep -E and Python agree on this expression: no backreferences or classes beyond POSIX ones.
        regex = re.compile(pattern)
        scenarios = [
            {("windows-latest", "1.13", 3): {"statuses": {"zeta": "fail", "delta": "error"}}},
            {("macos-latest", "1.10", 1): {"files": ["alpha", "epsilon", "x_y"], "statuses": {"x_y": "fail"}}},
        ]
        for overrides in scenarios:
            self.logs_reset()
            self.write_all(overrides)
            self.run_verdict()
            for line in self.signature().splitlines():
                self.assertRegex(line, regex)
        self.logs_reset()
        self.write_all(skip={("macos-latest", "1.13", 2)})
        self.aggregate = "failure"
        self.run_verdict()
        for line in self.signature().splitlines():
            self.assertRegex(line, regex)


class LogLayoutTests(Harness):
    """One matching artifact downloads into the log directory itself; several into one directory each."""

    label = "FOCUSED VALIDATION"

    def flatten(self):
        """Move each log file of the per-artifact directories to the top level, as a single match does."""
        for directory in sorted(self.logs.iterdir()):
            for path in directory.iterdir():
                path.rename(self.logs / path.name)
            directory.rmdir()

    def test_several_matches_use_one_directory_each(self):
        self.configure(mode="focused", files="alpha")
        self.write_all()
        self.assertTrue(all(entry.is_dir() for entry in self.logs.iterdir()))
        code, out = self.run_verdict()
        self.assertEqual(code, 0, out)

    def test_one_match_sits_at_the_top_level(self):
        self.configure(mode="focused", files="alpha", platforms="linux", julia="1.13")
        self.write_all()
        self.flatten()
        self.assertEqual([p.name for p in self.logs.iterdir()], ["full-validation-ubuntu-latest-1.13-1.log"])
        code, out = self.run_verdict()
        self.assertEqual(code, 0, out)
        self.assertIn("FOCUSED VALIDATION: PASS", out)
        self.assertIn("| ubuntu-latest | 1.13.9 | x86_64-linux-gnu | 1:success | 1/1 | 1/1 | none |", out)

    def test_one_match_of_a_full_plan_with_one_cell_uploaded(self):
        self.configure(mode="full")
        self.write_all(skip=set(self.cells) - {("macos-latest", "1.13", 2)})
        self.flatten()
        out = self.assertFails_as("FULL VALIDATION")
        self.assertNotIn("unexpected entry", out)
        self.assertNotIn("macos-latest / Julia 1.13 / shard 2/3: no log artifact", out)
        self.assertIn("macos-latest / Julia 1.13 / shard 1/3: no log artifact", out)
        self.assertEqual(out.count("(missing cell)"), 17)

    def test_multi_cell_plan_with_one_uploaded_log(self):
        self.configure(mode="focused", files="alpha")
        keep = ("windows-latest", "1.10", 1)
        self.write_all(skip=set(self.cells) - {keep})
        self.flatten()
        out = self.assertFails("windows-latest / Julia 1.13 / shard 1/1: no log artifact")
        self.assertEqual(out.count("(missing cell)"), 5)
        self.assertNotIn("unexpected entry", out)
        self.assertIn("| windows-latest | 1.10.9 | x86_64-w64-mingw32 | 1:success | 1/1 | 1/1 | none |", out)

    def test_zero_matches(self):
        self.configure(mode="focused", files="alpha", platforms="linux", julia="1.13")
        out = self.assertFails("ubuntu-latest / Julia 1.13 / shard 1/1: no log artifact")
        self.assertIn("holds no full-validation-log-* artifact", out)
        self.logs.rmdir()
        self.assertFails("does not exist")

    def test_top_level_log_of_an_unplanned_cell(self):
        self.configure(mode="focused", files="alpha", platforms="linux", julia="1.13")
        self.write_all()
        self.write_cell("windows-latest", "1.10", 1)
        self.flatten()
        self.assertFails("full-validation-log-windows-latest-1.10-1-attempt-2 is outside the expected cells of attempt 2")

    def test_top_level_log_and_directory_for_one_cell(self):
        self.configure(mode="focused", files="alpha", platforms="linux", julia="1.13")
        self.write_all()
        (self.logs / "full-validation-ubuntu-latest-1.13-1.log").write_text("x")
        self.assertFails("expected exactly one log file")

    def test_unrecognized_top_level_entries_are_stray(self):
        self.configure(mode="focused", files="alpha", platforms="linux", julia="1.13")
        self.write_all()
        self.flatten()
        (self.logs / "notes.txt").write_text("x")
        (self.logs / "full-validation-ubuntu-latest-1.13-1.log.bak").write_text("x")
        self.assertFails("unexpected entry notes.txt", "unexpected entry full-validation-ubuntu-latest-1.13-1.log.bak")

    def assertFails_as(self, label):
        self.label = label
        return self.assertFails()


class RoundTripTests(Harness):
    """The plan the planner accepts, with complete synthetic logs and jobs, gives PASS in each mode."""

    def passes(self, label, **inputs):
        self.setUp()
        self.configure(**inputs)
        self.label = label
        if inputs.get("mode") == "release":
            self.ci_runs_path = self.root / "ci_runs.json"
            self.ci_runs_path.write_text(json.dumps([ci_run()]))
        self.write_all()
        code, out = self.run_verdict()
        self.assertEqual(code, 0, out)
        self.assertIn(f"{label}: PASS", out)

    def test_full(self):
        self.passes("FULL VALIDATION", mode="full")
        self.assertEqual(len(self.cells), 18)

    def test_release(self):
        self.passes("RELEASE VALIDATION", mode="release", expected_sha=COMMIT)

    def test_focused_in_every_restriction(self):
        for platform in plan_script.PLATFORM_CHOICES:
            for julia in plan_script.JULIA_CHOICES:
                with self.subTest(platform=platform, julia=julia):
                    self.passes("FOCUSED VALIDATION", mode="focused", files="alpha gamma", platforms=platform, julia=julia)

    def test_focused_layout_with_one_cell(self):
        self.passes("FOCUSED VALIDATION", mode="focused", files="beta", platforms="macos", julia="1.10")
        self.assertEqual(len(self.cells), 1)


# The workflow tests below are source-text checks. They read the block forms that this workflow
# uses (two-space job keys, `run:` with a `|` block or on one line, `env:` entries) and
# representative names and headers. They are not a YAML validator, and they do not run the
# issue lookup and creation of the report job.
def uncommented(text):
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def job_blocks(text):
    """Map job id -> the job's text, for the jobs of the workflow (two-space-indented keys)."""
    _, jobs = text.split("\njobs:\n")
    blocks, current = {}, None
    for line in jobs.splitlines():
        match = re.fullmatch(r"  ([a-z]+):", line)
        if match:
            current = match.group(1)
            blocks[current] = []
        elif current:
            blocks[current].append(line)
    return {name: "\n".join(lines) for name, lines in blocks.items()}


def run_blocks(text):
    """The text of each `run:` step written as a one-line command or as a `|` or `>` block."""
    blocks, lines = [], text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)(?:- )?run:\s*(.*)$", line)
        if not match:
            continue
        indent, rest = len(match.group(1)), match.group(2)
        if rest.strip() not in ("|", ">", "|-", ">-"):
            blocks.append(rest)
            continue
        body = []
        for following in lines[index + 1 :]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following)
        blocks.append("\n".join(body))
    return blocks


class WorkflowTextTests(unittest.TestCase):
    def setUp(self):
        self.full = WORKFLOW.read_text()
        self.text = uncommented(self.full)
        self.jobs = job_blocks(self.text)

    def test_jobs(self):
        self.assertEqual(list(self.jobs), ["plan", "test", "verdict", "report"])

    def test_inputs_match_the_plan_script(self):
        def options(name):
            match = re.search(rf"^      {name}:\n(?:        .*\n)*?        options: \[(.*)\]$", self.text, re.M)
            self.assertIsNotNone(match, name)
            return [item.strip().strip("'") for item in match.group(1).split(",")]

        self.assertEqual(options("mode"), list(v.MODES))
        self.assertEqual(options("platforms"), list(plan_script.PLATFORM_CHOICES))
        self.assertEqual(options("julia"), list(plan_script.JULIA_CHOICES))
        for name, default in (("mode", "full"), ("platforms", "all"), ("julia", "both")):
            self.assertRegex(self.text, rf"      {name}:\n(?:        .*\n)*?        default: {default}\n")
        for name in ("files", "expected_sha"):
            self.assertRegex(self.text, rf"      {name}:\n(?:        .*\n)*?        type: string\n        default: ''\n")
        self.assertIn("cron: '23 9 * * *'", self.text)

    def test_inputs_reach_the_scripts_through_env(self):
        for block in run_blocks(self.text):
            for needle in ("${{ inputs.", "${{ github.event.inputs."):
                self.assertNotIn(needle, block)
            # The run blocks of this workflow hold no expression at all.
            self.assertNotIn("${{", block)
        references = [line for line in self.text.splitlines() if "inputs." in line and "${{" in line]
        # The five plan inputs; the verdict job output `needs.verdict.outputs.attempt` is not an input.
        self.assertEqual(len(references), 5)
        for line in references:
            self.assertRegex(line, r"^          PB_[A-Z_]+: \$\{\{ inputs\.[a-z_]+ \}\}$")
        self.assertNotIn("github.event.inputs", self.text)
        for name, key in (("MODE", "mode"), ("PLATFORMS", "platforms"), ("JULIA", "julia"), ("FILES", "files"), ("EXPECTED_SHA", "expected_sha")):
            self.assertIn(f"PB_{name}: ${{{{ inputs.{key} }}}}", self.jobs["plan"])

    def test_plan_job(self):
        plan = self.jobs["plan"]
        self.assertIn("persist-credentials: false", plan)
        self.assertLess(plan.index("actions/checkout"), plan.index("full_validation_plan.py --tiers"))
        for name in ("matrix", "mode", "files"):
            self.assertIn(f"{name}: ${{{{ steps.plan.outputs.{name} }}}}", plan)
        self.assertIn("name: full-validation-plan", plan)
        self.assertNotIn("permissions:", plan)

    def test_test_job_wiring(self):
        test = self.jobs["test"]
        self.assertIn("needs: [plan]", test)
        self.assertIn("matrix: ${{ fromJSON(needs.plan.outputs.matrix) }}", test)
        self.assertIn("fail-fast: false", test)
        self.assertIn("name: ${{ matrix.label }}", test)
        self.assertIn("--check-bounds=yes", test)
        self.assertIn('"--tier=full", "--shard=" * ENV["PB_SHARD"] * "/" * ENV["PB_SHARDS"]', test)
        # Focused names reach the runner as exact selectors built from the environment.
        self.assertIn('ENV["PB_MODE"] == "focused" && append!(args, ["--exact"; String.(split(ENV["PB_FILES"]))])', test)
        for needle in ("PB_MODE: ${{ needs.plan.outputs.mode }}", "PB_FILES: ${{ needs.plan.outputs.files }}",
                       "PB_SHARDS: ${{ matrix.shards }}", "shard=%s/%s mode=%s shards=%s"):
            self.assertIn(needle, test)
        self.assertNotIn("permissions:", test)

    def test_artifact_name_matches_script_format(self):
        match = re.search(r"^          name: (full-validation-log-.*)$", self.text, re.M)
        self.assertIsNotNone(match)
        rendered = (match.group(1).replace("${{ matrix.os }}", "ubuntu-latest")
                    .replace("${{ matrix.julia }}", "1.10")
                    .replace("${{ matrix.shard }}", "1")
                    .replace("${{ github.run_attempt }}", "7"))
        self.assertEqual(rendered, v.artifact_name("ubuntu-latest", "1.10", 1, 7))
        self.assertIn("pattern: full-validation-log-*-attempt-${{ github.run_attempt }}", self.text)
        # The upload path names the file that the verdict script recognizes at the top level.
        path = re.search(r"^          path: (full-validation-logs/.*)$", self.text, re.M).group(1)
        rendered_path = (path.replace("${{ matrix.os }}", "ubuntu-latest").replace("${{ matrix.julia }}", "1.10")
                         .replace("${{ matrix.shard }}", "1"))
        self.assertTrue(v.LOG_FILE_RE.fullmatch(rendered_path.split("/")[-1]))

    def test_header_format_matches_the_script(self):
        # A header built from the test step's printf format satisfies the verdict script's header pattern.
        match = re.search(r"printf '(PB_FULL_VALIDATION [^']*?)\\n'", self.text)
        self.assertIsNotNone(match)
        template = match.group(1)
        self.assertEqual(template.count("%s"), 7)
        rendered = template.replace("%s", "{}").format(COMMIT, "ubuntu-latest", "1.10", 1, 3, "full", 3)
        self.assertEqual(rendered, f"PB_FULL_VALIDATION commit={COMMIT} os=ubuntu-latest julia=1.10 shard=1/3 mode=full shards=3")
        self.assertRegex(rendered, v.HEADER_RE)

    def test_verdict_job_wiring(self):
        verdict = self.jobs["verdict"]
        self.assertIn("needs: [plan, test]", verdict)
        self.assertIn("if: always()", verdict)
        self.assertIn("attempt: ${{ github.run_attempt }}", verdict)
        self.assertIn("PLAN_RESULT: ${{ needs.plan.result }}", verdict)
        self.assertIn('"$PLAN_RESULT" != "success"', verdict)
        for needle in ("--jobs jobs.jsonl", '--attempt "$GITHUB_RUN_ATTEMPT"', '--aggregate "$TEST_RESULT"',
                       "--plan plan/plan.json", "--signature signature.txt", '--expected-commit "$GITHUB_SHA"',
                       "name: full-validation-plan", "persist-credentials: false"):
            self.assertIn(needle, verdict)
        self.assertIn("name: full-validation-signature-attempt-${{ github.run_attempt }}", verdict)

    def test_release_evidence_step(self):
        verdict = self.jobs["verdict"]
        self.assertIn("if: needs.plan.outputs.mode == 'release'", verdict)
        # The run list is paginated, and the event and conclusion filter precedes the jobs requests.
        self.assertRegex(verdict, r'gh api --paginate "repos/\$GITHUB_REPOSITORY/actions/workflows/CI\.yml/runs\?head_sha=\$GITHUB_SHA&per_page=100"')
        self.assertIn('select((.event == "push" or .event == "workflow_dispatch") and .conclusion == "success")', verdict)
        self.assertIn("run_attempt}", verdict)
        # The jobs come from the attempt that the run record reports.
        self.assertIn('actions/runs/$id/attempts/$attempt/jobs?per_page=100"', verdict)
        self.assertLess(verdict.index("select((.event"), verdict.index("attempts/$attempt/jobs"))
        self.assertIn('if [ "$PB_MODE" = "release" ]; then\n            extra=(--ci-runs ci_runs.json)', verdict)
        self.assertIn('"${extra[@]}"', verdict)

    def test_report_job(self):
        report = self.jobs["report"]
        self.assertIn("needs: [verdict]", report)
        self.assertIn("if: ${{ always() && needs.verdict.result == 'failure' && github.event_name == 'schedule' }}", report)
        self.assertEqual(self.text.count("github.event_name == 'schedule'"), 2)  # the concurrency group and this job
        self.assertIn("GH_TOKEN: ${{ github.token }}", report)
        self.assertIn("Nightly full validation failing [$hash]", report)
        self.assertIn("verdict-incomplete", report)
        self.assertIn("--body-file", report)
        # The artifact and the links name the attempt that produced the signature.
        self.assertIn("pattern: full-validation-signature-attempt-${{ needs.verdict.outputs.attempt }}", report)
        self.assertIn("PRODUCING_ATTEMPT: ${{ needs.verdict.outputs.attempt }}", report)
        self.assertIn('link="$RUN_URL/attempts/$PRODUCING_ATTEMPT"', report)
        self.assertNotIn("github.run_attempt", report)

    def test_report_job_deduplicates_through_the_labeled_issue_list(self):
        report = self.jobs["report"]
        self.assertIn("label=nightly-full-validation", report)
        self.assertIn('gh label create "$label"', report)
        self.assertIn("2>/dev/null || true", report)
        self.assertIn('gh api --paginate "repos/$GITHUB_REPOSITORY/issues?state=all&labels=$label&per_page=100"', report)
        self.assertIn("select(.pull_request == null)", report)
        self.assertIn('.title == $title and .state == "open"', report)
        self.assertIn('.title == $title and .state == "closed"', report)
        self.assertIn('gh issue reopen "$closed"', report)
        self.assertIn('--label "$label"', report)
        self.assertNotIn("gh issue list", report)
        self.assertNotIn("--search", report)

    def test_permissions_are_confined(self):
        head = self.text.split("\njobs:\n")[0]
        self.assertIn("permissions:\n  contents: read\n", head)
        self.assertNotIn("actions:", head)
        self.assertNotIn("issues:", head)
        for name in ("plan", "test"):
            self.assertNotIn("permissions:", self.jobs[name])
        self.assertIn("permissions:\n      contents: read\n      actions: read\n", self.jobs["verdict"])
        self.assertNotIn("issues:", self.jobs["verdict"])
        # The report job requests issues: write and nothing else.
        self.assertIn("permissions:\n      issues: write\n    steps:", self.jobs["report"])
        self.assertNotIn("contents:", self.jobs["report"])
        self.assertNotIn("actions: read", self.jobs["report"])
        self.assertEqual(self.text.count("actions: read"), 1)
        self.assertEqual(self.text.count("issues: write"), 1)
        self.assertEqual(self.text.count("contents: write"), 0)

    def test_commented_snippets_do_not_count(self):
        self.assertEqual(uncommented("  # if: always()\nx: 1"), "x: 1")

    def test_run_block_scan_finds_blocks(self):
        sample = "      - name: x\n        run: |\n          a ${{ inputs.mode }}\n          b\n      - run: echo ${{ inputs.files }}\n"
        found = run_blocks(sample)
        self.assertEqual(len(found), 2)
        self.assertIn("${{ inputs.mode }}", found[0])
        self.assertIn("${{ inputs.files }}", found[1])

    def test_forbidden_strings(self):
        for needle in ("continue-on-error", "pull_request_target"):
            self.assertNotIn(needle, self.full)


if __name__ == "__main__":
    unittest.main()
