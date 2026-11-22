#!/usr/bin/env python3
"""Tests for the full-validation verdict script, on synthetic log trees."""
from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import full_validation_verdict as v

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "full-validation.yml"
COMMIT = "a" * 40
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


def shard_files(shard, names=SELECTION):
    return names[shard - 1 :: 3]


def log_text(os_name, julia, shard, files, *, commit=COMMIT, header_os=None, header_shard=None,
             version=None, statuses=None):
    statuses = statuses or {}
    version = version or f"{julia}.9"
    lines = [
        f"PB_FULL_VALIDATION commit={commit} os={header_os or os_name} julia={julia} "
        f"shard={header_shard or f'{shard}/3'}",
        "Julia Version " + version,
        f"ProximalBase minimal tests: tier full; shard {shard}/3; {len(files)} files; "
        f"Julia {version} on x86_64-test",
        "  " + " ".join(files),
    ]
    for i, name in enumerate(files):
        lines.append(f"PB_TEST_FILE_TIME file={name} elapsed_s={10 + i}.5 status={statuses.get(name, 'pass')}")
    lines.append(f"PB_TEST_SHARD_TIME shard={shard}/3 files={len(files)} elapsed_s=99.0")
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

    def write_cell(self, os_name, julia, shard, text=None, files=None, **kw):
        directory = self.logs / v.artifact_name(os_name, julia, shard)
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
        argv = ["--logs", str(self.logs), "--tiers", str(self.tiers), "--expected-commit", COMMIT]
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
        self.assertIn("| ubuntu-latest | 1.10.9 | x86_64-test | 3/3 | 6/6 | none |", out)
        text = summary.read_text()
        self.assertTrue(text.startswith("earlier\n"))
        self.assertIn("FULL VALIDATION: PASS", text)

    def test_missing_cell(self):
        self.write_all(skip={("windows-latest", "1.13", 2)})
        self.assertFails("windows-latest / Julia 1.13 / shard 2/3: no log artifact")

    def test_extra_cell(self):
        self.write_all()
        (self.logs / "full-validation-log-freebsd-1.10-1").mkdir()
        (self.logs / "full-validation-log-freebsd-1.10-1" / "x.log").write_text("x")
        self.assertFails("full-validation-log-freebsd-1.10-1 is outside the expected cells")

    def test_stray_entry(self):
        self.write_all()
        (self.logs / "notes.txt").write_text("x")
        self.assertFails("unexpected entry notes.txt")

    def test_commit_mismatch(self):
        self.write_all({("macos-latest", "1.10", 1): {"commit": "b" * 40}})
        self.assertFails("header commit='" + "b" * 40 + "'")

    def test_header_os_and_shard_mismatch(self):
        self.write_all({
            ("ubuntu-latest", "1.10", 1): {"header_os": "windows-latest"},
            ("ubuntu-latest", "1.10", 2): {"header_shard": "1/3"},
        })
        out = self.assertFails("header os='windows-latest'", "header shard='1/3'")
        self.assertIn("2 problems", out)

    def test_missing_header(self):
        self.write_all()
        path = next((self.logs / v.artifact_name("ubuntu-latest", "1.10", 1)).iterdir())
        path.write_text("\n".join(l for l in path.read_text().splitlines() if not l.startswith("PB_FULL")))
        self.assertFails("expected one PB_FULL_VALIDATION header, found 0")

    def test_failing_file(self):
        self.write_all({("ubuntu-latest", "1.13", 3): {"statuses": {"zeta": "fail", "delta": "error"}}})
        out = self.assertFails("test file zeta has status fail", "test file delta has status error")
        self.assertIn("delta (error), zeta (fail)", out)
        self.assertIn("| 4/6 |", out)

    def test_missing_file_in_one_pair(self):
        files = [n for n in shard_files(2) if n != "gamma"]
        self.write_all({("windows-latest", "1.10", 2): {"files": files}})
        self.assertFails("windows-latest / Julia 1.10: files in the full selection did not run: ['gamma']")

    def test_file_in_two_shards(self):
        files = shard_files(2) + ["alpha"]
        self.write_all({("macos-latest", "1.13", 2): {"files": files}})
        self.assertFails("macos-latest / Julia 1.13: test file alpha ran in shards 1 and 2")

    def test_unexpected_file(self):
        files = shard_files(1) + ["omega"]
        self.write_all({("macos-latest", "1.13", 1): {"files": files}})
        self.assertFails("files outside the full selection ran: ['omega']")

    def test_duplicate_report_in_one_log(self):
        files = shard_files(1) + [shard_files(1)[0]]
        self.write_all({("ubuntu-latest", "1.13", 1): {"files": files}})
        self.assertFails("is reported twice in one log")

    def test_wrong_julia_minor(self):
        self.write_all({("ubuntu-latest", "1.10", 1): {"version": "1.11.2"}})
        self.assertFails("ran Julia 1.11.2, expected 1.10.x")

    def test_julia_prefix_is_not_a_substring_match(self):
        self.write_all({("ubuntu-latest", "1.10", 1): {"version": "1.100.0"}})
        self.assertFails("ran Julia 1.100.0, expected 1.10.x")

    def test_two_log_files_in_one_cell(self):
        self.write_all()
        directory = self.logs / v.artifact_name("ubuntu-latest", "1.10", 1)
        (directory / "second.log").write_text("x")
        self.assertFails("expected exactly one log file")

    def test_empty_logs_directory(self):
        out = self.assertFails("holds no full-validation-log-* artifact")
        self.assertIn("(19 problems)", out)

    def test_absent_logs_directory(self):
        self.logs.rmdir()
        self.assertFails("does not exist")

    def test_duplicate_test_name_in_tiers(self):
        self.tiers.write_text(TIERS_TEXT.replace('full = ["zeta"]', 'full = ["zeta", "alpha"]'))
        self.write_all()
        self.assertFails("duplicate test file in manifest")

    def test_missing_tier_in_tiers(self):
        self.tiers.write_text(TIERS_TEXT.replace('full = ["zeta"]\n', ""))
        self.write_all()
        self.assertFails("all four tiers")


class WorkflowTextTests(unittest.TestCase):
    def setUp(self):
        self.text = WORKFLOW.read_text()

    def test_matrix_matches_script_constants(self):
        oses = ", ".join(v.OSES)
        julias = ", ".join(f"'{j}'" for j in v.JULIAS)
        shards = ", ".join(str(s) for s in v.SHARDS)
        for line in (f"os: [{oses}]", f"julia: [{julias}]", f"shard: [{shards}]"):
            self.assertIn(line, self.text)
        self.assertEqual(v.SHARD_COUNT, len(v.SHARDS))
        self.assertIn(f"(shard ${{{{ matrix.shard }}}}/{v.SHARD_COUNT})", self.text)

    def test_required_strings(self):
        for needle in ("if: always()", "persist-credentials: false", "--check-bounds=yes", "--tier=full"):
            self.assertIn(needle, self.text)

    def test_forbidden_strings(self):
        for needle in ("continue-on-error", "pull_request_target"):
            self.assertNotIn(needle, self.text)


if __name__ == "__main__":
    unittest.main()
