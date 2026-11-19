#!/usr/bin/env python3
"""Focused negative fixtures for preservation checks; no package tests changed."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import json
import unittest

import migration_checks as m


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = "docs/src/numerics.md"
        self.original = "# Numerical contract\n\nWith a positive `λ`, the result is `0`. NaN stays NaN.\n"
        self.write(self.source, self.original)
        self.baseline = list(m.blocks(self.original, self.source))
        self.rows = m.initial_ledger(self.baseline)

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def check(self, rows=None):
        return m.coverage(self.baseline, self.rows if rows is None else rows, self.root)

    def test_unchanged_content_passes(self):
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        for name in ("tokens", "sentences", "examples"):
            self.assertEqual(m.accounting(self.baseline, mappings, name), [])

    def test_missing_ledger_row_is_not_masked_by_counts(self):
        errors, mappings = self.check(self.rows[:-1])
        self.assertTrue(any("missing ledger row" in error for error in errors))
        self.assertTrue(m.accounting(self.baseline, mappings, "sentences"))

    def test_changed_baseline_hash_fails(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]["sha"] = "0" * 64
        self.assertTrue(any("baseline hash" in error for error in self.check(rows)[0]))

    def test_mutated_condition_and_token_fail(self):
        self.write(self.source, self.original.replace("positive", "negative").replace("`0`", "`1`"))
        errors, mappings = self.check()
        self.assertTrue(errors)
        self.assertTrue(m.accounting(self.baseline, mappings, "tokens"))

    def test_wrong_destination_hash_fails(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]["dest_sha"] = "1" * 64
        self.assertTrue(self.check(rows)[0])

    def test_duplicate_rows_fail(self):
        errors, _ = self.check(self.rows + [self.rows[-1]])
        self.assertTrue(any("duplicate ledger row" in error for error in errors))

    def test_one_occurrence_cannot_cover_two_original_blocks(self):
        old = "# Rules\n\nSame sentence.\n\nSame sentence.\n"
        self.baseline = list(m.blocks(old, self.source))
        self.rows = m.initial_ledger(self.baseline)
        self.write(self.source, "# Rules\n\nSame sentence.\n")
        self.assertTrue(any("occurrence reused" in error for error in self.check()[0]))

    def test_verbatim_move_accepts_equivalent_relative_link(self):
        old = "# Rules\n\nSee [precision](numerical_usage.md#Ownership-and-aliases).\n"
        self.baseline = list(m.blocks(old, self.source))
        self.rows = m.initial_ledger(self.baseline)
        self.write(self.source, "# Rules\n\nForwarding link.\n")
        self.write("docs/src/spec/sample.md", "# Rules\n\nSee [precision](../numerical_usage.md#Ownership-and-aliases).\n")
        self.rows[-1].update(action="move", dest="docs/src/spec/sample.md#Rules")
        self.assertEqual(self.check()[0], [])

    def test_wrong_relative_link_fails(self):
        old = "# Rules\n\nSee [precision](numerical_usage.md).\n"
        self.baseline = list(m.blocks(old, self.source))
        self.rows = m.initial_ledger(self.baseline)
        self.write(self.source, "# Rules\n")
        self.write("docs/src/spec/sample.md", "# Rules\n\nSee [precision](numerical_usage.md).\n")
        self.rows[-1].update(action="move", dest="docs/src/spec/sample.md")
        self.assertTrue(self.check()[0])

    def test_fragment_only_link_requires_original_page_after_move(self):
        old = "# Rules\n\nSee [details](#details).\n"
        self.baseline = list(m.blocks(old, self.source))
        self.rows = m.initial_ledger(self.baseline)
        self.write(self.source, "# Rules\n")
        target = "docs/src/spec/sample.md"
        self.write(target, "# Rules\n\nSee [details](#details).\n")
        self.rows[-1].update(action="move", dest=target)
        self.assertTrue(self.check()[0])
        self.write(target, "# Rules\n\nSee [details](../numerics.md#details).\n")
        self.assertEqual(self.check()[0], [])

    def test_reword_requires_independent_review_and_current_hash(self):
        rewritten = "# Numerical contract\n\nA positive `λ` gives `0`. NaN stays NaN.\n"
        self.write(self.source, rewritten)
        replacements = list(m.blocks(rewritten, self.source))
        self.rows[1].update(action="reword", dest_sha=replacements[1].sha)
        self.assertTrue(any("independent review" in error for error in self.check()[0]))
        self.rows[1]["reviewer"] = "independent-reviewer@fixture-revision"
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        for name in ("tokens", "sentences", "examples"):
            self.assertEqual(m.accounting(self.baseline, mappings, name), [])
        self.write(self.source, rewritten.replace("positive", "negative"))
        self.assertTrue(self.check()[0])

    def test_all_reviewed_actions_use_same_acceptance_policy(self):
        # These fixtures exercise provenance, not the truth of an equivalence.
        # Actual rows require independent rendered semantic review.
        for action in ("reword", "dedupe", "supersede"):
            with self.subTest(action=action):
                replacement = "# Replacement\n\nReplacement text.\n"
                self.write("docs/src/replacement.md", replacement)
                candidate = list(m.blocks(replacement, "docs/src/replacement.md"))[-1]
                rows = copy.deepcopy(self.rows)
                rows[1].update(action=action, dest="docs/src/replacement.md", dest_sha=candidate.sha,
                               reviewer="independent-reviewer@fixture-revision")
                errors, mappings = self.check(rows)
                self.assertEqual(errors, [])
                for name in ("tokens", "sentences", "examples"):
                    self.assertEqual(m.accounting(self.baseline, mappings, name), [])

    def test_split_rejects_missing_words(self):
        self.write("docs/src/replacement.md", "Replacement.\n")
        candidate = list(m.blocks("Replacement.\n", "docs/src/replacement.md"))[0]
        self.rows[1].update(action="split", dest="docs/src/replacement.md", dest_sha=candidate.sha,
                            reviewer="independent-reviewer@fixture-revision")
        self.assertTrue(any("split loses original words" in error for error in self.check()[0]))

    def test_reviewed_split_retains_original_words(self):
        original = "alpha beta gamma delta\n"
        self.baseline = list(m.blocks(original, self.source))
        self.rows = m.initial_ledger(self.baseline)
        target = "docs/src/replacement.md"
        replacement = "alpha beta\n\ngamma delta\n"
        self.write(target, replacement)
        pieces = list(m.blocks(replacement, target))
        self.rows[0].update(action="split", dest=target + ";" + target,
                            dest_sha=";".join(b.sha for b in pieces),
                            reviewer="independent-reviewer@fixture-revision")
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(m.accounting(self.baseline, mappings, "sentences"), [])

    def test_archive_requires_forwarding_link(self):
        target = "docs/design/history.md"
        self.write(target, "NaN stays NaN.\n")
        self.rows[-1].update(action="archive", dest=target)
        self.assertTrue(any("forwarding link" in error for error in self.check()[0]))
        self.write(self.source, self.original + "\nSee [history](../design/history.md).\n")
        self.assertEqual(self.check()[0], [])

    def test_replaced_assertion_is_not_hidden_by_same_total_count(self):
        original = "```@example sample\n@assert x == 0\n```\n"
        self.baseline = list(m.blocks(original, self.source))
        self.rows = m.initial_ledger(self.baseline)
        self.write(self.source, original.replace("x == 0", "y == 1"))
        errors, mappings = self.check()
        self.assertTrue(errors)
        self.assertTrue(m.accounting(self.baseline, mappings, "examples"))

    def test_sentence_split_respects_code_and_links(self):
        text = "Use `x.y` and [a.b](../a.b.md). Then return `0.5`."
        self.assertEqual(list(m.sentences(text)), ["Use `x.y` and [a.b](../a.b.md).", "Then return `0.5`."])
        text = "Use $a.  b$. Then finish."
        self.assertEqual(list(m.sentences(text)), ["Use $a.  b$.", "Then finish."])

    def test_inline_code_literal_spaces_are_preserved(self):
        original = 'Call `replace("a  b", "  " => " ")`.\n'
        changed = 'Call `replace("a b", " " => " ")`.\n'
        self.baseline = list(m.blocks(original, self.source))
        self.rows = m.initial_ledger(self.baseline)
        self.write(self.source, changed)
        self.assertNotEqual(m.normalized(original, self.source), m.normalized(changed, self.source))
        self.assertTrue(self.check()[0])
        changed_block = list(m.blocks(changed, self.source))[0]
        self.assertTrue(m.protected_tokens(self.baseline[0]) - m.protected_tokens(changed_block))

    def test_code_string_that_looks_like_link_is_not_rebased(self):
        text = 'Call `f("[label](other.md)")` and see [real](other.md).'
        result = m.normalized(text, self.source)
        self.assertIn('`f("[label](other.md)")`', result)
        self.assertIn('[real](docs/src/other.md)', result)
        # A code span used as the display label must not prevent real-link repair.
        result = m.normalized('[`f("a  b")`](other.md)', self.source)
        self.assertEqual(result, '[`f("a  b")`](docs/src/other.md)')

    def test_math_literal_spaces_survive_prose_normalization(self):
        for text in ('Use ``a  b``.', 'Use $a  b$.', 'Use \\(a  b\\).'):
            with self.subTest(text=text):
                self.assertNotEqual(m.normalized(text, self.source), m.normalized(text.replace('  ', ' '), self.source))
        self.assertEqual(m.normalized('One  word\n then\tmore.', self.source), 'One word then more.')

    def test_admonition_fence_preserves_julia_string_spaces(self):
        original = '!!! note\n\n    ```julia\n    s = "a  b"\n    ```\n'
        changed = original.replace('"a  b"', '"a b"')
        self.baseline = list(m.blocks(original, self.source))
        self.rows = m.initial_ledger(self.baseline)
        self.write(self.source, changed)
        self.assertNotEqual(m.normalized(original, self.source, 'admonition'),
                            m.normalized(changed, self.source, 'admonition'))
        self.assertTrue(self.check()[0])

    def test_section_ignores_heading_looking_code_comments(self):
        text = ('## [Results](@id result)\n\n```@example demo\n'
                '# A meaningful comment\n@assert 1 == 1\n```\n\n## Next\n')
        selected = m.section_text(text, 'result')
        self.assertIn('@assert 1 == 1\n```', selected)
        self.assertNotIn('## Next', selected)
        with self.assertRaisesRegex(ValueError, 'missing destination heading'):
            m.section_text(text, 'A-meaningful-comment')

    def test_fence_and_math_are_atomic(self):
        text = "# Code\n\n```@example demo\n# Not a heading\n@assert x == 1\n```\n\n```math\nx^2\n```\n"
        result = list(m.blocks(text, self.source))
        self.assertEqual([b.kind for b in result], ["heading", "code", "math"])
        self.assertEqual(m.example_tokens(result[1]), Counter({"```@example demo": 1, "@assert x == 1": 1}))

    def test_anchor_must_stay_on_original_page(self):
        build = self.root / "build"
        self.write("build/old.html", '<article id="documenter-page"><h1 id="other">Old</h1></article>')
        self.write("build/new.html", '<article id="documenter-page"><h1 id="required">New</h1></article>')
        errors = m.anchor_check([dict(page="old", fragment="required")], build)
        self.assertEqual(len(errors), 1)
        self.assertIn("old#required", errors[0])

    def test_sidebar_anchor_changes_are_excluded(self):
        page = '<nav><input id="collapse-1"></nav><article id="documenter-page"><h1 id="kept">Text</h1></article>'
        self.assertEqual(m.ContentAnchors(page).ids, {"documenter-page", "kept"})

    def anchor_profile_fixture(self):
        audit = self.root / "audit"
        audit.mkdir()
        metadata = dict(baseline=m.BASELINE, rendered_pages=1, content_anchor_count=1, anchor_profiles={})
        for profile, filename, fragment in (("local", "anchors.tsv", "original-local-hash"),
                                             ("ci", "anchors-ci.tsv", "original-ci-hash")):
            rows = [dict(page="old", fragment=""), dict(page="old", fragment=fragment)]
            m.write_tsv(audit / filename, ("page", "fragment"), rows)
            metadata["anchor_profiles"][profile] = dict(file=filename, sha256=m.digest((audit / filename).read_text()))
        return audit, metadata

    def test_anchor_profiles_are_explicit_and_wrong_runtime_fails(self):
        audit, metadata = self.anchor_profile_fixture()
        self.write("build/old.html", '<article id="documenter-page"><h1 id="original-local-hash">Old</h1></article>')
        build = self.root / "build"
        self.assertEqual(m.anchor_check(m.anchor_snapshot(metadata, "local", audit), build), [])
        self.assertTrue(m.anchor_check(m.anchor_snapshot(metadata, "ci", audit), build))
        with self.assertRaisesRegex(ValueError, "unknown or unavailable"):
            m.anchor_snapshot(metadata, "other", audit)
        del metadata["anchor_profiles"]["ci"]
        with self.assertRaisesRegex(ValueError, "unknown or unavailable"):
            m.anchor_snapshot(metadata, "ci", audit)

    def test_empty_or_truncated_anchor_profile_is_not_accepted(self):
        audit, metadata = self.anchor_profile_fixture()
        path = audit / "anchors.tsv"
        for rows, error in (([], "empty"), ([dict(page="old", fragment="")], "anchor count"),
                            ([dict(page="old", fragment="original-local-hash")], "page records")):
            with self.subTest(rows=rows):
                m.write_tsv(path, ("page", "fragment"), rows)
                metadata["anchor_profiles"]["local"]["sha256"] = m.digest(path.read_text())
                with self.assertRaisesRegex(ValueError, error):
                    m.anchor_snapshot(metadata, "local", audit)

    def test_anchor_profile_snapshot_hash_is_checked(self):
        audit, metadata = self.anchor_profile_fixture()
        path = audit / "anchors.tsv"
        path.write_text(path.read_text().replace("original-local-hash", "different-hash"))
        with self.assertRaisesRegex(ValueError, "snapshot hash mismatch"):
            m.anchor_snapshot(metadata, "local", audit)

    def test_nested_docstring_articles_keep_following_page_anchors(self):
        html = ('<article id="documenter-page"><h1 id="page-title"></h1>'
                '<article><details id="docstring"><article id="nested-note">'
                '</article></details></article><h2 id="following-section"></h2>'
                '</article><article id="outside-page"></article>')
        self.assertEqual(m.ContentAnchors(html).ids,
                         {"documenter-page", "page-title", "docstring",
                          "nested-note", "following-section"})

    def test_flat_and_directory_urls_share_page_identity(self):
        self.assertEqual(m.page_key("old.html"), m.page_key("old/index.html"))
        self.assertEqual(m.page_key("index.html"), "")
        self.assertEqual(m.page_key("developer/index.html"), "developer")

    def test_wrapped_source_heading_reference(self):
        text = '# numerics.md ("Supported precision\n# and oracle agreement")\n'
        self.assertEqual(list(m.source_citations(text)), [(1, "Supported precision and oracle agreement")])


class UpstreamTests(unittest.TestCase):
    """Use real Git objects, not mocked hashes, for inherited-change receipts."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = "README.md"
        self.original = ("# Guide\n\nOriginal `0` rule.\n\n## API\n\n"
                         "```@docs\nIndNonnegative\n```\n\n## Other\n\nAn unrelated rule.\n")
        self.changed = (self.original.replace("Original `0`", "Updated `1`")
                        .replace("```@docs\n", "```@docs\nIndBox\n")
                        .replace("An unrelated rule.", "A different unrelated rule."))
        m.git("init", "--quiet", root=self.root)
        self.before = self.commit(self.original, "before")
        self.after = self.commit(self.changed, "upstream")
        self.receipt = self.root / "docs/audit/docs-tracks-1019/upstream.json"
        self.receipt.parent.mkdir(parents=True)
        self.select("prose")

    def commit(self, text, message):
        (self.root / self.path).write_text(text)
        m.git("add", self.path, root=self.root)
        m.git("-c", "user.name=Migration fixture", "-c", "user.email=fixture@example.invalid",
              "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", message, root=self.root)
        return m.git("rev-parse", "HEAD", root=self.root).strip()

    def select(self, kind):
        old = next(b for b in m.blocks(self.original, self.path) if b.kind == kind)
        new = next(b for b in m.blocks(self.changed, self.path) if b.kind == kind)
        self.baseline = [old]
        self.rows = m.initial_ledger(self.baseline)
        self.rows[0].update(action="upstream", dest_sha=new.sha, reviewer="fixture-reviewer@upstream")
        self.record = dict(block_id=old.id, baseline_sha=old.sha, dest=self.path,
                           dest_sha=new.sha, reviewer=self.rows[0]["reviewer"],
                           pull_request=1020, pull_request_url="https://github.com/mlakolar/ProximalBase.jl/pull/1020",
                           merged=True, before_commit=self.before, after_commit=self.after,
                           before_blob=m.git("rev-parse", f"{self.before}:{self.path}", root=self.root).strip(),
                           after_blob=m.git("rev-parse", f"{self.after}:{self.path}", root=self.root).strip())
        self.save()

    def save(self, records=None):
        self.receipt.write_text(json.dumps(dict(schema=1, changes=[self.record] if records is None else records)))

    def check(self):
        return m.coverage(self.baseline, self.rows, self.root)

    def assert_rejected(self, message):
        errors, mappings = self.check()
        self.assertTrue(any(message in error for error in errors), errors)
        self.assertNotIn(self.baseline[0].id, mappings)
        for kind in ("tokens", "sentences", "examples"):
            self.assertTrue(m.accounting(self.baseline, mappings, kind))

    def test_reviewed_upstream_change_passes_all_accounting(self):
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        for kind in ("tokens", "sentences", "examples"):
            self.assertEqual(m.accounting(self.baseline, mappings, kind), [])

    def test_upstream_insertion_inside_code_block_passes_with_shallow_roots(self):
        self.select("code")
        # Match depth-one fetches of both exact commits: their parent objects
        # exist, but Git revision walking treats each commit as a shallow root.
        (self.root / ".git/shallow").write_text(self.before + "\n" + self.after + "\n")
        self.assertEqual(m.git("rev-parse", "--is-shallow-repository", root=self.root).strip(), "true")
        self.assertEqual(self.check()[0], [])

    def test_upstream_requires_receipt_review_and_merged_pr(self):
        self.receipt.unlink()
        self.assert_rejected("upstream evidence")
        for field, value, error in (("merged", False, "merged pull request"),
                                     ("reviewer", "mechanical", "independent review")):
            with self.subTest(field=field):
                self.select("prose")
                self.record[field] = value
                if field == "reviewer":
                    self.rows[0][field] = value
                self.save()
                self.assert_rejected(error)

    def test_upstream_rejects_duplicate_and_unused_receipts(self):
        self.save([self.record, self.record])
        self.assert_rejected("duplicate upstream receipt")
        extra = dict(self.record, block_id="unrelated")
        self.save([self.record, extra])
        self.assert_rejected("exactly match upstream ledger rows")

    def test_upstream_rejects_bad_hash_blob_and_parent(self):
        for field, value, error in (("baseline_sha", "0" * 64, "does not match"),
                                     ("dest_sha", "0" * 64, "does not match"),
                                     ("before_blob", self.record["after_blob"], "blob mismatch"),
                                     ("before_commit", self.after, "first parent")):
            with self.subTest(field=field):
                self.select("prose")
                self.record[field] = value
                self.save()
                self.assert_rejected(error)

    def test_upstream_rejects_unrelated_changed_block_as_replacement(self):
        unrelated = list(m.blocks(self.changed, self.path))[-1]
        self.rows[0]["dest_sha"] = self.record["dest_sha"] = unrelated.sha
        self.save()
        self.assert_rejected("share a changed diff hunk")

    def test_upstream_rejects_hash_not_present_in_recorded_after_blob(self):
        mutation = self.changed.replace("Updated `1`", "Unreviewed `2`")
        (self.root / self.path).write_text(mutation)
        new = next(b for b in m.blocks(mutation, self.path) if b.kind == "prose")
        self.rows[0]["dest_sha"] = self.record["dest_sha"] = new.sha
        self.save()
        self.assert_rejected("unique original and replacement blocks")

    def test_upstream_rejects_current_mutation_after_review(self):
        (self.root / self.path).write_text(self.changed.replace("Updated `1`", "Unreviewed `2`"))
        self.assert_rejected("no block with expected hash")

    def test_upstream_rejects_unchanged_original_and_ambiguous_replacement(self):
        for suffix, error in (("\nOriginal `0` rule.\n", "remove the original"),
                               ("\nUpdated `1` rule.\n", "unique original and replacement")):
            with self.subTest(suffix=suffix):
                m.git("checkout", "--quiet", "--detach", self.before, root=self.root)
                self.after = self.commit(self.changed + suffix, "invalid upstream")
                self.select("prose")
                self.assert_rejected(error)


if __name__ == "__main__":
    unittest.main()
