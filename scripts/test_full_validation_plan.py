#!/usr/bin/env python3
"""Tests for the full-validation plan script."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import full_validation_plan as plan
import full_validation_verdict as v

ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 40
OTHER_SHA = "b" * 40
SELECTION = sorted(
    ["alpha", "beta", "gamma", "elastic", "elastic_extremes", "kkt", "kkt_block", "solo"]
    + [f"bulk_{n:02d}" for n in range(1, 25)]
)


def make(**inputs):
    return plan.make_plan(inputs, SHA, SELECTION)


def problems_of(**inputs):
    try:
        plan.make_plan(inputs, SHA, SELECTION)
    except plan.PlanInputError as error:
        return error.problems
    raise AssertionError(f"inputs were accepted: {inputs}")


def cells(result):
    return [(c["os"], c["julia"], c["shard"]) for c in result["matrix"]["include"]]


class PlanTests(unittest.TestCase):
    def assertProblem(self, fragment, **inputs):
        found = problems_of(**inputs)
        self.assertTrue(any(fragment in p for p in found), f"{fragment!r} not in {found}")
        return found

    # Defaults and the schedule.
    def test_defaults(self):
        result = make()
        self.assertEqual(result["mode"], "full")
        self.assertEqual(result["files"], [])
        self.assertEqual(sorted(cells(result)), sorted(v.expected_cells()))
        self.assertEqual(len(result["matrix"]["include"]), 18)

    def test_schedule_inputs_are_empty_or_absent(self):
        for inputs in ({}, {"mode": "", "platforms": "", "julia": "", "files": "", "expected_sha": ""},
                       {"mode": None, "platforms": None, "julia": None, "files": None, "expected_sha": None}):
            with self.subTest(inputs=inputs):
                result = make(**inputs)
                self.assertEqual((result["mode"], result["platforms"], result["julia"]), ("full", "all", "both"))
                self.assertEqual(sorted(cells(result)), sorted(v.expected_cells()))

    def test_full_matrix_equals_the_verdict_constants(self):
        for mode, extra in (("full", {}), ("release", {"expected_sha": SHA})):
            with self.subTest(mode=mode):
                result = make(mode=mode, **extra)
                self.assertEqual(cells(result), v.expected_cells())
                for entry in result["matrix"]["include"]:
                    self.assertEqual(entry["shards"], v.SHARD_COUNT)
                    self.assertEqual(entry["label"], v.job_name(entry["os"], entry["julia"], entry["shard"]))
        self.assertEqual(v.SHARD_COUNT, len(v.SHARDS))

    # Platforms, Julia versions and modes.
    def test_every_platform_and_julia_combination(self):
        oses = {"linux": ["ubuntu-latest"], "windows": ["windows-latest"], "macos": ["macos-latest"],
                "all": list(v.OSES)}
        julias = {"1.10": ["1.10"], "1.13": ["1.13"], "both": ["1.10", "1.13"]}
        for platform, os_names in oses.items():
            for julia, versions in julias.items():
                with self.subTest(platform=platform, julia=julia):
                    full = make(mode="full", platforms=platform, julia=julia)
                    self.assertEqual(cells(full), [(o, j, s) for o in os_names for j in versions for s in (1, 2, 3)])
                    focused = make(mode="focused", files="alpha", platforms=platform, julia=julia)
                    self.assertEqual(cells(focused), [(o, j, 1) for o in os_names for j in versions])
                    for entry in focused["matrix"]["include"]:
                        self.assertEqual(entry["shards"], 1)
                        self.assertEqual(entry["label"], f"Focused / {entry['os']} / Julia {entry['julia']}")

    def test_full_labels_keep_the_job_name_format(self):
        result = make(mode="full", platforms="windows", julia="1.13")
        self.assertEqual([c["label"] for c in result["matrix"]["include"]],
                         [f"Full / windows-latest / Julia 1.13 (shard {n}/3)" for n in (1, 2, 3)])

    def test_julia_versions_are_strings(self):
        for entry in make()["matrix"]["include"]:
            self.assertIsInstance(entry["julia"], str)

    def test_invalid_choices(self):
        self.assertProblem("mode 'nightly' is not one of full, focused, release", mode="nightly")
        self.assertProblem("platforms 'bsd' is not one of all, linux, windows, macos", platforms="bsd")
        self.assertProblem("julia '1.11' is not one of both, 1.10, 1.13", julia="1.11")
        self.assertProblem("julia '1.1' is not one of", julia="1.1")  # a YAML float would lose the zero
        self.assertProblem("mode 'Full'", mode="Full")
        self.assertEqual(len(problems_of(mode="x", platforms="y", julia="z")), 3)

    # Files.
    def test_focused_files_are_normalized(self):
        result = make(mode="focused", files="  alpha\tgamma\n beta  ")
        self.assertEqual(result["files"], ["alpha", "gamma", "beta"])
        self.assertEqual(plan.outputs_text(result).splitlines()[-1], "files=alpha gamma beta")

    def test_focused_needs_files(self):
        self.assertProblem("focused mode needs files", mode="focused")
        self.assertProblem("focused mode needs files", mode="focused", files="   ")

    def test_files_only_for_focused(self):
        self.assertProblem("files must be empty in 'full' mode", mode="full", files="alpha")
        self.assertProblem("files must be empty in 'full' mode", files="alpha")
        self.assertProblem("files must be empty in 'release' mode", mode="release", files="alpha", expected_sha=SHA)

    def test_unknown_file(self):
        found = self.assertProblem("file 'omega' is not a test name in the full selection", mode="focused", files="alpha omega")
        self.assertEqual(len(found), 1)

    def test_path_traversal_and_paths(self):
        for token in ("../x", "..", "test/alpha", "alpha.jl", "/etc/passwd", "a\\b", "~/x", "alpha;ls"):
            with self.subTest(token=token):
                self.assertProblem("is not a plain test name", mode="focused", files=token)

    def test_wildcards_and_prefix_syntax(self):
        for token in ("*", "kkt*", "kkt_?", "[ab]lpha", "{alpha,beta}", "ALPHA", "-alpha", "--tier=full"):
            with self.subTest(token=token):
                self.assertProblem("is not a plain test name", mode="focused", files=token)

    def test_a_name_that_is_only_a_prefix_of_names_is_unknown(self):
        self.assertProblem("file 'alph' is not a test name", mode="focused", files="alph")

    def test_duplicate_file(self):
        self.assertProblem("file 'alpha' is listed more than once", mode="focused", files="alpha beta alpha")

    def test_at_most_20_files(self):
        names = [f"bulk_{n:02d}" for n in range(1, 21)]
        self.assertEqual(make(mode="focused", files=" ".join(names))["files"], names)
        self.assertProblem("lists 21 names, at most 20 are allowed", mode="focused", files=" ".join(names + ["bulk_21"]))

    def test_prefix_collision_with_an_unrequested_name(self):
        self.assertProblem("file 'elastic' is a prefix of ['elastic_extremes']", mode="focused", files="elastic")
        self.assertProblem("file 'kkt' is a prefix of ['kkt_block']", mode="focused", files="kkt alpha")
        # Requesting every longer name as well removes the collision.
        self.assertEqual(make(mode="focused", files="elastic elastic_extremes")["files"], ["elastic", "elastic_extremes"])
        self.assertEqual(make(mode="focused", files="kkt_block")["files"], ["kkt_block"])

    def test_untrusted_text_cannot_start_a_workflow_command_or_run_long(self):
        found = problems_of(mode="focused", files="::error::x " + "y" * 500)
        for problem in found:
            self.assertTrue(problem.startswith("file '"), problem)
            self.assertLess(len(problem), 400)
        self.assertIn("...", " ".join(found))

    # The expected SHA.
    def test_expected_sha_cases(self):
        self.assertEqual(make(expected_sha=SHA)["expected_sha"], SHA)
        self.assertEqual(make(mode="focused", files="alpha", expected_sha=SHA)["expected_sha"], SHA)
        for bad in ("abc", "A" * 40, SHA + "\n", " " + SHA, SHA[:-1] + "g", SHA + "0", "main"):
            with self.subTest(bad=bad):
                self.assertProblem("is not 40 lowercase hexadecimal characters", expected_sha=bad)

    def test_expected_sha_mismatch_prints_both(self):
        found = self.assertProblem("does not match the commit of this run", expected_sha=OTHER_SHA)
        self.assertIn(OTHER_SHA, found[0])
        self.assertIn(SHA, found[0])
        self.assertProblem("does not match", mode="focused", files="alpha", expected_sha=OTHER_SHA)

    def test_release_needs_the_sha(self):
        self.assertProblem("release mode needs expected_sha", mode="release")
        self.assertEqual(make(mode="release", expected_sha=SHA)["mode"], "release")

    def test_release_needs_all_platforms_and_both_versions(self):
        self.assertProblem("release mode needs platforms=all", mode="release", expected_sha=SHA, platforms="windows")
        self.assertProblem("release mode needs julia=both", mode="release", expected_sha=SHA, julia="1.13")
        self.assertEqual(len(problems_of(mode="release", expected_sha=SHA, platforms="linux", julia="1.10")), 2)

    def test_problems_are_all_reported(self):
        found = problems_of(mode="focused", files="omega ../x", expected_sha="zz", platforms="bsd")
        self.assertGreaterEqual(len(found), 4)

    # The outputs and the command line.
    def test_outputs_text(self):
        result = make(mode="focused", files="alpha", platforms="macos", julia="1.10")
        lines = plan.outputs_text(result).splitlines()
        self.assertEqual(lines[0], "matrix=" + json.dumps({"include": result["matrix"]["include"]}, separators=(",", ":")))
        self.assertNotIn('", "', lines[0])
        self.assertNotIn('": ', lines[0])
        self.assertEqual(lines[1:], ["mode=focused", "files=alpha"])
        self.assertEqual(plan.outputs_text(make()).splitlines()[-2:], ["mode=full", "files="])


class MainTests(unittest.TestCase):
    def run_main(self, environ, tiers=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        argv = ["--tiers", str(tiers or ROOT / "test" / "tiers.toml"), "--plan-out", str(root / "plan.json"),
                "--github-output", str(root / "output")]
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = plan.main(argv, environ)
        output = (root / "output").read_text() if (root / "output").exists() else None
        plan_file = json.loads((root / "plan.json").read_text()) if (root / "plan.json").exists() else None
        return code, output, plan_file, err.getvalue()

    def test_scheduled_run_with_the_real_manifest(self):
        code, output, plan_file, err = self.run_main({"GITHUB_SHA": SHA})
        self.assertEqual((code, err), (0, ""))
        self.assertIn("mode=full\n", output)
        self.assertEqual(len(plan_file["matrix"]["include"]), 18)
        self.assertEqual(plan_file["commit"], SHA)

    def test_focused_run_with_the_real_manifest(self):
        env = {"GITHUB_SHA": SHA, "PB_MODE": "focused", "PB_FILES": "kkt_block minimal_math", "PB_PLATFORMS": "windows",
               "PB_JULIA": "1.13", "PB_EXPECTED_SHA": SHA}
        code, output, plan_file, err = self.run_main(env)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("files=kkt_block minimal_math\n", output)
        self.assertEqual(plan_file["files"], ["kkt_block", "minimal_math"])
        self.assertEqual(cells(plan_file), [("windows-latest", "1.13", 1)])

    def test_real_manifest_prefix_pairs_are_rejected_when_incomplete(self):
        names = v.load_selection(ROOT / "test" / "tiers.toml")
        pairs = [(a, b) for a in names for b in names if a != b and b.startswith(a)]
        self.assertTrue(pairs, "the manifest has no prefix pair; the check would be untested")
        a, b = pairs[0]
        code, output, _, err = self.run_main({"GITHUB_SHA": SHA, "PB_MODE": "focused", "PB_FILES": a})
        self.assertEqual(code, 1)
        self.assertIsNone(output)
        self.assertIn(f"file '{a}' is a prefix of", err)
        code, _, _, err = self.run_main({"GITHUB_SHA": SHA, "PB_MODE": "focused", "PB_FILES": " ".join(sorted({a} | {n for n in names if n.startswith(a)}))})
        self.assertEqual((code, err), (0, ""))

    def test_invalid_inputs_write_nothing(self):
        code, output, plan_file, err = self.run_main({"GITHUB_SHA": SHA, "PB_MODE": "focused", "PB_FILES": "../x"})
        self.assertEqual(code, 1)
        self.assertIsNone(output)
        self.assertIsNone(plan_file)
        self.assertIn("full-validation plan: file '../x' is not a plain test name", err)

    def test_unreadable_manifest(self):
        code, output, _, err = self.run_main({"GITHUB_SHA": SHA}, tiers="/nonexistent/tiers.toml")
        self.assertEqual(code, 1)
        self.assertIn("cannot read", err)

    def test_mismatched_sha_fails_the_run(self):
        code, _, _, err = self.run_main({"GITHUB_SHA": SHA, "PB_EXPECTED_SHA": OTHER_SHA})
        self.assertEqual(code, 1)
        self.assertIn(OTHER_SHA, err)
        self.assertIn(SHA, err)


if __name__ == "__main__":
    unittest.main()
