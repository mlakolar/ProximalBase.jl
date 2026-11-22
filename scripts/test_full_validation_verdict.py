#!/usr/bin/env python3
"""Tests for the full-validation verdict script, on synthetic log trees."""
from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

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
             runner_count=1):
    """One cell log.  `files` are the timing records; `announced` is the printed file list."""
    statuses = statuses or {}
    version = version or f"{julia}.9"
    announced = files if announced is None else announced
    count = len(announced) if count is None else count
    header = (
        f"PB_FULL_VALIDATION commit={commit} os={header_os or os_name} julia={julia} "
        f"shard={header_shard or f'{shard}/3'}"
    )
    runner = (
        f"ProximalBase minimal tests: tier full; shard {shard}/3; {count} files; "
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
        lines.append(f"PB_TEST_SHARD_TIME shard={shard}/3 files={len(files) if trailer_count is None else trailer_count} elapsed_s=99.0")
    return "\r\n".join(lines) + "\r\n"


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.tiers = self.root / "tiers.toml"
        self.tiers.write_text(TIERS_TEXT)
        self.jobs = self.root / "jobs.jsonl"
        self.write_jobs()

    def write_jobs(self, conclusions=None, skip=(), extra=(), repeat=()):
        conclusions = conclusions or {}
        records = []
        for cell in v.expected_cells():
            if cell in skip:
                continue
            record = {"name": v.job_name(*cell), "conclusion": conclusions.get(cell, "success")}
            records += [record] * (2 if cell in repeat else 1)
        records += [{"name": "Full validation verdict", "conclusion": None}, {"name": "Other job", "conclusion": "failure"}]
        records += list(extra)
        self.jobs.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def write_cell(self, os_name, julia, shard, text=None, files=None, attempt=ATTEMPT, **kw):
        directory = self.logs / v.artifact_name(os_name, julia, shard, attempt)
        directory.mkdir(parents=True, exist_ok=True)
        if files is None:
            files = shard_files(shard)
        name = f"full-validation-{os_name}-{julia}-{shard}.log"
        (directory / name).write_text(text if text is not None else log_text(os_name, julia, shard, files, **kw))

    def write_all(self, overrides=None, skip=()):
        overrides = overrides or {}
        for cell in v.expected_cells():
            if cell in skip:
                continue
            self.write_cell(*cell, **overrides.get(cell, {}))

    def run_verdict(self, summary=None):
        argv = ["--logs", str(self.logs), "--tiers", str(self.tiers), "--jobs", str(self.jobs),
                "--attempt", ATTEMPT, "--expected-commit", COMMIT]
        if summary:
            argv += ["--summary", str(summary)]
        out = io.StringIO()
        with redirect_stdout(out):
            code = v.main(argv)
        return code, out.getvalue()

    def assertFails(self, *fragments):
        code, out = self.run_verdict()
        self.assertEqual(code, 1, out)
        self.assertIn("FULL VALIDATION: FAIL", out)
        for fragment in fragments:
            self.assertIn(fragment, out)
        return out

    def only_cell_problem(self, overrides, *fragments):
        self.write_all(overrides)
        return self.assertFails(*fragments)

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

    def logs_reset(self):
        for path in sorted(self.logs.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()

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


def uncommented(text):
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


class WorkflowTextTests(unittest.TestCase):
    def setUp(self):
        self.full = WORKFLOW.read_text()
        self.text = uncommented(self.full)

    def test_matrix_matches_script_constants(self):
        oses = ", ".join(v.OSES)
        julias = ", ".join(f"'{j}'" for j in v.JULIAS)
        shards = ", ".join(str(s) for s in v.SHARDS)
        for line in (f"os: [{oses}]", f"julia: [{julias}]", f"shard: [{shards}]"):
            self.assertIn(line, self.text)
        self.assertEqual(v.SHARD_COUNT, len(v.SHARDS))

    def test_job_name_matches_script_format(self):
        match = re.search(r"^    name: (Full / .*)$", self.text, re.M)
        self.assertIsNotNone(match)
        template = match.group(1)
        for cell in v.expected_cells():
            rendered = (template.replace("${{ matrix.os }}", cell[0])
                        .replace("${{ matrix.julia }}", cell[1])
                        .replace("${{ matrix.shard }}", str(cell[2])))
            self.assertEqual(rendered, v.job_name(*cell))

    def test_artifact_name_matches_script_format(self):
        match = re.search(r"^          name: (full-validation-log-.*)$", self.text, re.M)
        self.assertIsNotNone(match)
        rendered = (match.group(1).replace("${{ matrix.os }}", "ubuntu-latest")
                    .replace("${{ matrix.julia }}", "1.10")
                    .replace("${{ matrix.shard }}", "1")
                    .replace("${{ github.run_attempt }}", "7"))
        self.assertEqual(rendered, v.artifact_name("ubuntu-latest", "1.10", 1, 7))
        self.assertIn("pattern: full-validation-log-*-attempt-${{ github.run_attempt }}", self.text)

    def test_required_strings(self):
        for needle in ("if: always()", "persist-credentials: false", "--check-bounds=yes", "--tier=full",
                       "actions: read", "--jobs jobs.jsonl", '--attempt "$GITHUB_RUN_ATTEMPT"',
                       'test "$TEST_RESULT" = success'):
            self.assertIn(needle, self.text)

    def test_actions_read_is_confined_to_the_verdict_job(self):
        self.assertEqual(self.text.count("actions: read"), 1)
        head, verdict = self.text.split("  verdict:\n")
        self.assertNotIn("actions: read", head)
        self.assertIn("permissions:\n  contents: read\n", head)

    def test_commented_snippets_do_not_count(self):
        self.assertEqual(uncommented("  # if: always()\nx: 1"), "x: 1")

    def test_forbidden_strings(self):
        for needle in ("continue-on-error", "pull_request_target"):
            self.assertNotIn(needle, self.full)


if __name__ == "__main__":
    unittest.main()
