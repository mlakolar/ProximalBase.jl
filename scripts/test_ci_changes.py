#!/usr/bin/env python3
"""Network-free tests for the documentation-only change classifier."""
from __future__ import annotations

import contextlib
import io
import os
import secrets
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ci_changes as c


class PathTests(unittest.TestCase):
    def test_documentation_paths(self):
        for path in (
            "README.md",
            "CONTRIBUTING.md",
            "AGENTS.md",
            "docs/src/numerics.md",
            "docs/design/block-prox-0.6.md",
            "docs/audit/docs-tracks-1019/ledger.tsv",
            "docs/audit/x/summary.json",
            "docs/audit/x/table.csv",
            "docs/audit/x/notes.txt",
            "docs/src/assets/logo.png",
            "docs/src/assets/figure.svg",
            "docs/src/diagram.mmd",
        ):
            with self.subTest(path=path):
                self.assertTrue(c.is_docs_path(path, "100644"))
        self.assertTrue(c.is_docs_path("docs/src/a.md", "100755"))
        self.assertTrue(c.is_docs_path("docs/src/a.md", "000000"))

    def test_code_paths(self):
        for path in (
            "docs/make.jl",
            "docs/check_html.py",
            "docs/Project.toml",
            "docs/src/foo.jl",
            "docs/audit/x/bench.jl",
            "docs/audit/x/run.py",
            "docs/audit/x/env.toml",
            "docs/design/x.patch",
            "docs/src/data.csv",
            "docs/a.md.jl",
            "docs/src/README",
            "test/TESTING.md",
            "examples/README.md",
            "benchmark/README.md",
            ".github/pull_request_template.md",
            ".github/workflows/CI.yml",
            "src/x.jl",
            "Project.toml",
            "Makefile",
            "foo.xyz",
            "README.MD",
        ):
            with self.subTest(path=path):
                self.assertFalse(c.is_docs_path(path, "100644"))

    def test_non_regular_modes(self):
        for mode in ("120000", "160000", "040000"):
            with self.subTest(mode=mode):
                self.assertFalse(c.is_docs_path("docs/src/link.md", mode))
                self.assertFalse(c.is_docs_path("README.md", mode))

    def test_empty_change_set(self):
        self.assertFalse(c.classify([]))


class RepoTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        for command in (
            ["git", "init", "-q", "-b", "main"],
            ["git", "config", "user.email", "ci@example.test"],
            ["git", "config", "user.name", "CI"],
        ):
            subprocess.run(command, check=True)
        for name in (
            "README.md",
            "docs/src/a.md",
            "docs/make.jl",
            "src/x.jl",
            "test/t.jl",
            "Project.toml",
        ):
            self.write(name, "start\n")
        self.commit("start")
        self.base = self.head()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def write(self, name, contents="changed\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    def commit(self, message):
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-qm", message, "--allow-empty"], check=True)

    def head(self, rev="HEAD"):
        return subprocess.check_output(["git", "rev-parse", rev], text=True).strip()

    def docs_only(self, base=None):
        return c.classify(c.changed_paths(base or self.base, "HEAD"))

    def run_main(self, env):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = c.main()
        return code, out.getvalue(), err.getvalue()


class ClassificationTests(RepoTestCase):
    def check(self, paths, expected):
        for name in paths:
            self.write(name)
        self.commit("change")
        self.assertEqual(self.docs_only(), expected)

    def test_markdown_only(self):
        self.check(["README.md", "docs/src/a.md", "docs/src/new.md"], True)

    def test_markdown_only_across_commits(self):
        self.write("README.md")
        self.commit("one")
        self.write("docs/src/a.md")
        self.commit("two")
        self.write("docs/audit/x/ledger.tsv")
        self.commit("three")
        self.assertEqual(self.docs_only(), True)

    def test_docs_plus_source(self):
        self.check(["docs/src/a.md", "src/x.jl"], False)

    def test_code_in_a_later_commit(self):
        self.write("docs/src/a.md")
        self.commit("one")
        self.write("src/x.jl")
        self.commit("two")
        self.assertFalse(self.docs_only())

    def test_single_code_paths(self):
        for name in (
            "src/x.jl",
            "test/t.jl",
            ".github/workflows/CI.yml",
            "docs/make.jl",
            "docs/check_html.py",
            "examples/lasso.jl",
            "Project.toml",
            "unknown.xyz",
        ):
            with self.subTest(path=name):
                self.write(name, f"change {name}\n")
                self.commit("change")
                self.assertFalse(self.docs_only())
                self.base = self.head()

    def test_rename_within_docs(self):
        subprocess.run(["git", "mv", "docs/src/a.md", "docs/src/b.md"], check=True)
        self.commit("rename")
        paths = c.changed_paths(self.base, "HEAD")
        self.assertEqual(sorted(p for p, _ in paths), ["docs/src/a.md", "docs/src/b.md"])
        self.assertTrue(self.docs_only())

    def test_rename_out_of_docs(self):
        (self.root / "src").mkdir(exist_ok=True)
        subprocess.run(["git", "mv", "docs/src/a.md", "src/a.jl"], check=True)
        self.commit("rename")
        self.assertFalse(self.docs_only())

    def test_rename_from_code_into_docs(self):
        subprocess.run(["git", "mv", "src/x.jl", "docs/src/x.md"], check=True)
        self.commit("rename")
        self.assertFalse(self.docs_only())

    def test_delete_documentation(self):
        subprocess.run(["git", "rm", "-q", "docs/src/a.md"], check=True)
        self.commit("delete")
        self.assertEqual(c.changed_paths(self.base, "HEAD"), [("docs/src/a.md", "000000")])
        self.assertTrue(self.docs_only())

    def test_delete_source(self):
        subprocess.run(["git", "rm", "-q", "src/x.jl"], check=True)
        self.commit("delete")
        self.assertFalse(self.docs_only())

    def test_symlink_named_as_documentation(self):
        os.symlink("a.md", self.root / "docs/src/link.md")
        self.commit("symlink")
        self.assertEqual(c.changed_paths(self.base, "HEAD"), [("docs/src/link.md", "120000")])
        self.assertFalse(self.docs_only())

    def test_file_replaced_by_symlink(self):
        (self.root / "docs/src/a.md").unlink()
        os.symlink("../../src/x.jl", self.root / "docs/src/a.md")
        self.commit("replace")
        self.assertFalse(self.docs_only())

    def test_empty_diff(self):
        self.commit("empty")
        self.assertEqual(c.changed_paths(self.base, "HEAD"), [])
        self.assertFalse(self.docs_only())

    def test_path_with_unusual_characters(self):
        self.write("docs/src/new\nline.md")
        self.write("docs/src/sp ace.md")
        self.commit("names")
        self.assertEqual(len(c.changed_paths(self.base, "HEAD")), 2)
        self.assertTrue(self.docs_only())


class BaseTests(RepoTestCase):
    def silent(self, event, env):
        with contextlib.redirect_stdout(io.StringIO()):
            return c.base_for_event(event, env)

    def test_push_with_zero_before(self):
        self.assertIsNone(self.silent("push", {"PUSH_BEFORE_SHA": "0" * 40}))

    def test_push_with_unreachable_before(self):
        env = {"PUSH_BEFORE_SHA": secrets.token_hex(20)}
        self.assertIsNone(self.silent("push", env))

    def test_push_with_reachable_before(self):
        self.write("README.md")
        self.commit("next")
        self.assertEqual(self.silent("push", {"PUSH_BEFORE_SHA": self.base}), self.base)

    def test_push_with_malformed_before(self):
        for value in ("", "abc", "G" * 40, self.base.upper(), self.base + "0"):
            with self.subTest(value=value):
                with self.assertRaises(c.ClassificationError):
                    self.silent("push", {"PUSH_BEFORE_SHA": value})
        with self.assertRaises(c.ClassificationError):
            self.silent("push", {})

    def test_workflow_dispatch(self):
        self.assertIsNone(self.silent("workflow_dispatch", {}))

    def test_unknown_event(self):
        for event in ("schedule", "pull_request_target", ""):
            with self.subTest(event=event):
                with self.assertRaises(c.ClassificationError):
                    self.silent(event, {})

    def test_pull_request_without_merge_commit(self):
        with self.assertRaises(c.ClassificationError):
            self.silent("pull_request", {})

    def test_pull_request_with_merge_commit(self):
        subprocess.run(["git", "checkout", "-q", "-b", "topic"], check=True)
        self.write("docs/src/a.md")
        self.commit("topic")
        subprocess.run(["git", "checkout", "-q", "main"], check=True)
        self.write("README.md")
        self.commit("main moves")
        main_tip = self.head()
        subprocess.run(["git", "merge", "-q", "--no-ff", "-m", "merge", "topic"], check=True)
        self.assertEqual(self.silent("pull_request", {}), main_tip)
        # Only the topic branch's change counts, not what main added before.
        paths = c.changed_paths(main_tip, "HEAD")
        self.assertEqual([p for p, _ in paths], ["docs/src/a.md"])


class MainTests(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.output = self.root.parent / (self.root.name + "-output.txt")
        self.addCleanup(lambda: self.output.unlink(missing_ok=True))

    def env(self, **extra):
        env = {"GITHUB_OUTPUT": str(self.output), "GITHUB_EVENT_NAME": "push",
               "PUSH_BEFORE_SHA": self.base}
        env.update(extra)
        return env

    def output_lines(self):
        return self.output.read_text().splitlines() if self.output.exists() else []

    def test_docs_only_writes_one_line(self):
        self.write("README.md")
        self.commit("docs")
        code, out, _ = self.run_main(self.env())
        self.assertEqual(code, 0)
        self.assertEqual(self.output_lines(), ["docs_only=true"])
        self.assertIn("README.md", out)
        self.assertIn("docs_only=true", out)

    def test_code_change_writes_false(self):
        self.write("src/x.jl")
        self.commit("code")
        code, _, _ = self.run_main(self.env())
        self.assertEqual(code, 0)
        self.assertEqual(self.output_lines(), ["docs_only=false"])

    def test_zero_before_writes_false(self):
        code, _, _ = self.run_main(self.env(PUSH_BEFORE_SHA="0" * 40))
        self.assertEqual(code, 0)
        self.assertEqual(self.output_lines(), ["docs_only=false"])

    def test_workflow_dispatch_writes_false(self):
        code, _, _ = self.run_main(self.env(GITHUB_EVENT_NAME="workflow_dispatch"))
        self.assertEqual(code, 0)
        self.assertEqual(self.output_lines(), ["docs_only=false"])

    def test_empty_diff_writes_false(self):
        self.commit("empty")
        code, _, _ = self.run_main(self.env())
        self.assertEqual(code, 0)
        self.assertEqual(self.output_lines(), ["docs_only=false"])

    def test_unknown_event_fails_without_output(self):
        code, _, err = self.run_main(self.env(GITHUB_EVENT_NAME="schedule"))
        self.assertNotEqual(code, 0)
        self.assertIn("unexpected event", err)
        self.assertEqual(self.output_lines(), [])

    def test_pull_request_without_merge_commit_fails_without_output(self):
        code, _, _ = self.run_main(self.env(GITHUB_EVENT_NAME="pull_request"))
        self.assertNotEqual(code, 0)
        self.assertEqual(self.output_lines(), [])

    def test_failed_git_diff_fails_without_output(self):
        # A base that names no object makes git diff fail.
        with mock.patch.object(c, "base_for_event", return_value="f" * 40):
            code, _, _ = self.run_main(self.env())
        self.assertNotEqual(code, 0)
        self.assertEqual(self.output_lines(), [])

    def test_step_summary(self):
        summary = self.root.parent / (self.root.name + "-summary.md")
        self.addCleanup(lambda: summary.unlink(missing_ok=True))
        self.write("README.md")
        self.commit("docs")
        code, _, _ = self.run_main(self.env(GITHUB_STEP_SUMMARY=str(summary)))
        self.assertEqual(code, 0)
        self.assertIn("docs_only=true", summary.read_text())

    def test_prints_without_output_file(self):
        self.write("README.md")
        self.commit("docs")
        env = self.env()
        env.pop("GITHUB_OUTPUT")
        with mock.patch.dict(os.environ):
            os.environ.pop("GITHUB_OUTPUT", None)
            code, out, _ = self.run_main(env)
        self.assertEqual(code, 0)
        self.assertIn("docs_only=true", out)
        self.assertEqual(self.output_lines(), [])


if __name__ == "__main__":
    unittest.main()
