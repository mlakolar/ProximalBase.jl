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


MODES = {"A": ("000000", "100644"), "M": ("100644", "100644"), "D": ("100644", "000000")}


def ch(path, status="A", old=None, new=None):
    default_old, default_new = MODES[status]
    return c.Change(path, old or default_old, new or default_new, status)


class PathTests(unittest.TestCase):
    def test_documentation_paths(self):
        for path in (
            "README.md",
            "CONTRIBUTING.md",
            "AGENTS.md",
            "docs/src/numerics.md",
            "docs/design/block-prox-0.6.md",
            "docs/audit/docs-tracks-1019/ledger.tsv",
            "docs/audit/docs-tracks-1019/x/summary.json",
            "docs/audit/docs-tracks-1019/table.csv",
            "docs/audit/docs-tracks-1019/notes.txt",
            "docs/audit/iterate-performance-852/notes.md",
            "docs/src/assets/logo.png",
            "docs/src/assets/figure.svg",
            "docs/src/diagram.mmd",
        ):
            for status in "AMD":
                with self.subTest(path=path, status=status):
                    self.assertTrue(c.is_docs_path(ch(path, status)))

    def test_code_paths(self):
        for path in (
            "docs/make.jl",
            "docs/check_html.py",
            "docs/Project.toml",
            "docs/src/foo.jl",
            "docs/audit/x/bench.jl",
            "docs/audit/x/run.py",
            "docs/audit/x/env.toml",
            "docs/audit/docs-tracks-1019/run.py",
            "docs/audit/docs-tracks-1019/env.toml",
            "docs/audit/docs-tracks-1019.json",
            "docs/audit/x/table.csv",
            "docs/audit/iterate-performance-852/research/runtime-hashes.json",
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
                self.assertFalse(c.is_docs_path(ch(path)))

    def test_mode_transitions(self):
        path = "docs/src/a.md"
        for status, old, new in (
            ("A", "000000", "100755"),
            ("A", "000000", "120000"),
            ("A", "000000", "160000"),
            ("M", "100644", "100755"),
            ("M", "100755", "100644"),
            ("M", "100755", "100755"),
            ("M", "120000", "120000"),
            ("D", "100755", "000000"),
            ("D", "120000", "000000"),
            ("D", "160000", "000000"),
            ("T", "120000", "100644"),
            ("T", "100644", "120000"),
            ("M", "000000", "100644"),
            ("A", "100644", "100644"),
            ("D", "100644", "100644"),
            ("A", "100644", "000000"),
            ("D", "000000", "100644"),
        ):
            with self.subTest(status=status, old=old, new=new):
                self.assertFalse(c.is_docs_path(c.Change(path, old, new, status)))

    def test_empty_change_set(self):
        self.assertFalse(c.classify([]))


class RepoTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY"):
            os.environ.pop(name, None)
        for command in (
            ["git", "init", "-q", "-b", "main"],
            ["git", "config", "user.email", "ci@example.test"],
            ["git", "config", "user.name", "CI"],
            ["git", "config", "core.fileMode", "true"],
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

    def commit_index(self, message):
        # Commit the index as it is: `git add -A` would drop a gitlink whose
        # directory is absent from the work tree.
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
        self.write("docs/audit/docs-tracks-1019/ledger.tsv")
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
        self.assertEqual(sorted(x.path for x in paths), ["docs/src/a.md", "docs/src/b.md"])
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
        self.assertEqual(c.changed_paths(self.base, "HEAD"), [c.Change("docs/src/a.md", "100644", "000000", "D")])
        self.assertTrue(self.docs_only())

    def test_delete_source(self):
        subprocess.run(["git", "rm", "-q", "src/x.jl"], check=True)
        self.commit("delete")
        self.assertFalse(self.docs_only())

    def test_symlink_named_as_documentation(self):
        os.symlink("a.md", self.root / "docs/src/link.md")
        self.commit("symlink")
        self.assertEqual(c.changed_paths(self.base, "HEAD"), [c.Change("docs/src/link.md", "000000", "120000", "A")])
        self.assertFalse(self.docs_only())

    def test_file_replaced_by_symlink(self):
        (self.root / "docs/src/a.md").unlink()
        os.symlink("../../src/x.jl", self.root / "docs/src/a.md")
        self.commit("replace")
        self.assertFalse(self.docs_only())

    def test_audit_manifest_outside_the_docs_checks(self):
        self.write("docs/audit/iterate-performance-852/research/runtime-hashes.json")
        self.commit("manifest")
        self.assertFalse(self.docs_only())

    def test_audit_data_read_by_the_docs_checks(self):
        self.write("docs/audit/docs-tracks-1019/ledger.tsv")
        self.commit("ledger")
        self.assertTrue(self.docs_only())

    def commit_gitlink(self, path):
        subprocess.run(["git", "update-index", "--add", "--cacheinfo",
                        f"160000,{self.base},{path}"], check=True)
        self.commit_index("gitlink")

    def test_delete_symlink_named_as_documentation(self):
        os.symlink("a.md", self.root / "docs/link.md")
        self.commit("symlink")
        self.base = self.head()
        subprocess.run(["git", "rm", "-q", "docs/link.md"], check=True)
        self.commit("delete")
        self.assertEqual(c.changed_paths(self.base, "HEAD"),
                         [c.Change("docs/link.md", "120000", "000000", "D")])
        self.assertFalse(self.docs_only())

    def test_replace_symlink_with_regular_file(self):
        os.symlink("a.md", self.root / "docs/link.md")
        self.commit("symlink")
        self.base = self.head()
        (self.root / "docs/link.md").unlink()
        self.write("docs/link.md")
        self.commit("replace")
        self.assertFalse(self.docs_only())

    def test_delete_gitlink_named_as_documentation(self):
        self.commit_gitlink("docs/module.md")
        self.base = self.head()
        subprocess.run(["git", "rm", "-q", "--cached", "docs/module.md"], check=True)
        self.commit_index("delete")
        self.assertEqual(c.changed_paths(self.base, "HEAD"),
                         [c.Change("docs/module.md", "160000", "000000", "D")])
        self.assertFalse(self.docs_only())

    def test_replace_gitlink_with_regular_file(self):
        self.commit_gitlink("docs/module.md")
        self.base = self.head()
        subprocess.run(["git", "rm", "-q", "--cached", "docs/module.md"], check=True)
        self.write("docs/module.md")
        self.commit("replace")
        self.assertFalse(self.docs_only())

    def test_add_gitlink_named_as_documentation(self):
        self.commit_gitlink("docs/module.md")
        self.assertFalse(self.docs_only())

    def test_execute_bit_change(self):
        (self.root / "docs/src/a.md").chmod(0o755)
        self.commit("chmod +x")
        self.assertEqual(c.changed_paths(self.base, "HEAD"),
                         [c.Change("docs/src/a.md", "100644", "100755", "M")])
        self.assertFalse(self.docs_only())
        self.base = self.head()
        (self.root / "docs/src/a.md").chmod(0o644)
        self.commit("chmod -x")
        self.assertFalse(self.docs_only())

    def test_add_and_delete_executable_markdown(self):
        self.write("docs/src/run.md")
        (self.root / "docs/src/run.md").chmod(0o755)
        self.commit("add executable")
        self.assertFalse(self.docs_only())
        self.base = self.head()
        subprocess.run(["git", "rm", "-q", "docs/src/run.md"], check=True)
        self.commit("delete executable")
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
        for value in ("abc", "G" * 40, self.base.upper(), self.base + "0"):
            with self.subTest(value=value):
                with self.assertRaises(c.ClassificationError):
                    self.silent("push", {"PUSH_BEFORE_SHA": value})

    def test_push_with_missing_before(self):
        self.assertIsNone(self.silent("push", {}))
        self.assertIsNone(self.silent("push", {"PUSH_BEFORE_SHA": ""}))

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
        self.assertEqual([x.path for x in paths], ["docs/src/a.md"])


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
        self.assertEqual(self.output_lines(), ["docs_only=true", f"base={self.base}"])
        self.assertIn("README.md", out)
        self.assertIn("docs_only=true", out)

    def test_code_change_writes_false(self):
        self.write("src/x.jl")
        self.commit("code")
        code, _, _ = self.run_main(self.env())
        self.assertEqual(code, 0)
        self.assertEqual(self.output_lines(), ["docs_only=false", f"base={self.base}"])

    def test_zero_before_writes_false(self):
        code, _, _ = self.run_main(self.env(PUSH_BEFORE_SHA="0" * 40))
        self.assertEqual(code, 0)
        self.assertEqual(self.output_lines(), ["docs_only=false", "base="])

    def test_missing_before_writes_false(self):
        for value in (None, ""):
            self.output.unlink(missing_ok=True)
            env = self.env()
            if value is None:
                env.pop("PUSH_BEFORE_SHA")
            else:
                env["PUSH_BEFORE_SHA"] = value
            with mock.patch.dict(os.environ):
                os.environ.pop("PUSH_BEFORE_SHA", None)
                code, _, _ = self.run_main(env)
            self.assertEqual(code, 0)
            self.assertEqual(self.output_lines(), ["docs_only=false", "base="])

    def test_malformed_before_fails_without_output(self):
        code, _, _ = self.run_main(self.env(PUSH_BEFORE_SHA="not-a-sha"))
        self.assertNotEqual(code, 0)
        self.assertEqual(self.output_lines(), [])

    def test_workflow_dispatch_writes_false(self):
        code, _, _ = self.run_main(self.env(GITHUB_EVENT_NAME="workflow_dispatch"))
        self.assertEqual(code, 0)
        self.assertEqual(self.output_lines(), ["docs_only=false", "base="])

    def test_empty_diff_writes_false(self):
        self.commit("empty")
        code, _, _ = self.run_main(self.env())
        self.assertEqual(code, 0)
        self.assertEqual(self.output_lines(), ["docs_only=false", f"base={self.base}"])

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

    def test_control_characters_in_paths_do_not_start_lines(self):
        self.write("docs/src/a\n::error::injected.md")
        self.write("docs/src/b\r::notice::x.md")
        self.commit("names")
        code, out, _ = self.run_main(self.env())
        self.assertEqual(code, 0)
        self.assertFalse([line for line in out.splitlines() if line.startswith("::")])
        self.assertTrue(all(line.startswith(("event:", "base:", "head:", "  path ", "docs_only="))
                            for line in out.splitlines()))
        self.assertIn("docs_only=true", out)

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
