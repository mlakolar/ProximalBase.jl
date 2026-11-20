#!/usr/bin/env python3
"""Focused fixtures for documentation preservation and the family map."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import json
import subprocess
import sys
import unittest

from check_contract_schema import family_map_errors
import check_migration_lines as line_report
import migration_checks as m


class FamilyMapTests(unittest.TestCase):
    def check(self, reference, rows, pages=None, other_sections=""):
        index = (
            "## Family map\n\n"
            "| Family or operation | Current entry point | Migration state |\n"
            "|:--|:--|:--|\n" + rows + "\n" + other_sections
        )
        return family_map_errors(reference, index, pages or {})

    def test_requires_exact_row_in_family_map_for_qualified_type(self):
        reference = "```@docs\nProximalBase.NewFamily{T}\nProximalBase.some_oracle!\n```\n"
        errors = self.check(
            reference, "| `NewFamilyVariant` | [Guide](newfamilyvariant.md) | Authored contract |",
            pages={"newfamilyvariant.md": "# NewFamilyVariant numerical contract"},
            other_sections="## Other table\n\n| `NewFamily` | [Guide](newfamily.md) | Authored contract |\n",
        )
        self.assertEqual(errors, ["api/functions.md: NewFamily is missing from the family map in spec/index.md"])

    def test_example_text_does_not_count_as_api_docs(self):
        reference = "Mentions `NewFamily`.\n\n````markdown\n```@docs\nNewFamily\n```\n````\n"
        self.assertEqual(self.check(reference, ""), [])

    def test_docs_directive_with_options_counts_as_api_docs(self):
        for directive in ("@docs; canonical=false", "@docs ; canonical=false"):
            with self.subTest(directive=directive):
                errors = self.check(f"```{directive}\nNewFamily\n```\n", "")
                self.assertEqual(len(errors), 1)
                self.assertIn("NewFamily is missing", errors[0])

    def test_detached_row_is_outside_the_family_map_table(self):
        reference = "```@docs\nLinearFunction\nZeroFunction\n```\n"
        rows = "| `LinearFunction` | Guide | Existing pages |\n\n| `ZeroFunction` | Guide | Existing pages |"
        errors = self.check(reference, rows)
        self.assertEqual(len(errors), 1)
        self.assertIn("ZeroFunction is missing", errors[0])

    def test_table_row_may_omit_its_outer_pipes(self):
        reference = "```@docs\nNormL0\n```\n"
        self.assertEqual(self.check(reference, "`NormL0` | Guide | Existing pages"), [])

    def test_recognizes_both_names_in_grandfathered_row(self):
        reference = "```@docs\nLinearFunction\nZeroFunction\n```\n"
        self.assertEqual(self.check(reference, "| `LinearFunction` and `ZeroFunction` | Guide | Existing pages |"), [])

    def test_new_name_cannot_enter_existing_pages_even_in_a_grouped_row(self):
        errors = self.check(
            "```@docs\nNewFamily\n```\n",
            "| `LinearFunction` and `NewFamily` | [Guide](newfamily.md) | Existing pages |",
            pages={"newfamily.md": "# NewFamily numerical contract"},
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("NewFamily cannot enter the family map as Existing pages", errors[0])

    def test_requires_own_contract_page_or_thin_constructor_marker(self):
        reference = "```@docs\nNewFamily\n```\n"
        row = "| `NewFamily` | [Contract](newfamily.md) | Authored contract |"
        errors = self.check(reference, row, pages={"otherfamily.md": "# OtherFamily numerical contract"})
        self.assertEqual(len(errors), 1)
        self.assertIn("NewFamily needs a family contract page", errors[0])
        self.assertEqual(self.check(reference, row, pages={"newfamily.md": "# NewFamily numerical contract"}), [])
        self.assertEqual(self.check(reference, "| `NewFamily` | [Owner](@ref IndBox) | Thin constructor of `IndBox` |"), [])

    def test_linked_contract_may_use_a_short_filename_and_declare_two_families(self):
        reference = "```@docs\nLinearFunction\nZeroFunction\n```\n"
        row = "| `LinearFunction` and `ZeroFunction` | [Contract](linear.md) | Verbatim migration |"
        linear = "# [Affine contracts](@id spec-linear)\n\n## [Identity](@id spec-linear-identity)\n\n"
        linear += "[`LinearFunction`](@ref) and [`ZeroFunction`](@ref ProximalBase.ZeroFunction)."
        self.assertEqual(self.check(reference, row, pages={"linear.md": linear}), [])

    def test_contract_must_link_to_a_page_that_declares_the_family(self):
        reference = "```@docs\nNewFamily\n```\n"
        pages = {"newfamily.md": "# NewFamily contract", "other.md": (
            "# [NewFamilyVariant contract](@id NewFamily)\n\n"
            "## Identity\n\n```markdown\n[`NewFamily`](@ref)\n```\n\n"
            "``[`NewFamily`](@ref)``\n\n"
            "## Definition and domain\n\n[`NewFamily`](@ref) is a related type.\n"
        )}
        for entry in ("[Contract](other.md)", "[Guide](../smooth.md)", "`[Contract](newfamily.md)`",
                      "![Contract](newfamily.md)", r"\[Contract](newfamily.md)"):
            with self.subTest(entry=entry):
                row = f"| `NewFamily` | {entry} | Authored contract |"
                errors = self.check(reference, row, pages=pages)
                self.assertEqual(len(errors), 1)
                self.assertIn("NewFamily needs a family contract page", errors[0])


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

    def test_only_dedupe_and_supersede_may_reduce_protected_counts(self):
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
                self.assertEqual(m.accounting(self.baseline, mappings, "sentences"), [])
                self.assertEqual(m.accounting(self.baseline, mappings, "examples"), [])
                if action == "reword":
                    self.assertTrue(m.accounting(self.baseline, mappings, "tokens"))
                else:
                    self.assertEqual(m.accounting(self.baseline, mappings, "tokens"), [])

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

    def anchor_retirement_fixture(self):
        page, path = "api/functions", "docs/src/api/functions.md"
        identifier = path + "::Function reference / Nonnegative orthant::13"
        old = m.Block(identifier, path, "heading", "## Nonnegative orthant", 1, "Nonnegative orthant")
        replacement = next(m.blocks("## Indicators\n", path))
        fragments = ["Nonnegative-orthant", "Nonnegative-orthant-1"]
        record = dict(page=page, fragments=fragments, target="Indicators", block_id=old.id,
                      baseline_sha=old.sha, reviewer="fixture-reviewer@fixture-revision",
                      evidence="Fixture: reviewed repository-authored inbound references are absent.")
        rows = [dict(page=page, fragment=fragment) for fragment in ["", *fragments, "Still-required"]]
        self.write("build/api/functions.html", '<article id="documenter-page"><h2 id="Indicators"></h2><h2 id="Still-required"></h2></article>')
        return rows, dict(retirements=[record], base_blocks=[old], mappings={old.id: ("upstream", [replacement])})

    def test_named_anchor_retirement_reports_present_and_retired_separately(self):
        rows, options = self.anchor_retirement_fixture()
        report = {}
        self.assertEqual(m.anchor_check(rows, self.root / "build", report=report, **options), [])
        self.assertEqual(report["baseline_fragment_count"], 3)
        self.assertEqual(report["present_fragment_count"], 1)
        self.assertEqual(report["retired_fragment_count"], 2)
        self.assertEqual(report["approved_retirement_count"], 2)
        self.assertEqual({r["fragment"] for r in report["approved_retirements"]},
                         {"Nonnegative-orthant", "Nonnegative-orthant-1"})
        self.assertEqual(len(m.anchor_check(rows, self.root / "build")), 2)

    def test_anchor_retirement_cannot_remove_an_explicit_id_heading(self):
        rows, options = self.anchor_retirement_fixture()
        old = options["base_blocks"][0]
        explicit = m.Block(old.id, old.path, "heading", "## [Nonnegative orthant](@id Nonnegative-orthant)", 1, old.heading)
        options["base_blocks"] = [explicit]
        options["retirements"][0]["baseline_sha"] = explicit.sha
        errors = m.anchor_check(rows, self.root / "build", **options)
        self.assertTrue(any("automatic heading, without @id" in error for error in errors), errors)

    def test_anchor_retirement_rejects_unknown_duplicate_and_nonbaseline_pairs(self):
        for case, message in (("unknown", "do not belong to the original heading"), ("duplicate", "duplicate anchor retirement"),
                               ("not-frozen", "not in the frozen snapshot")):
            with self.subTest(case=case):
                rows, options = self.anchor_retirement_fixture()
                record = options["retirements"][0]
                if case == "unknown":
                    record["fragments"][0] = "ProximalBase.IndNonnegative"
                    rows.append(dict(page="api/functions", fragment="ProximalBase.IndNonnegative"))
                elif case == "duplicate":
                    options["retirements"].append(copy.deepcopy(record))
                else:
                    rows = [row for row in rows if row["fragment"] != "Nonnegative-orthant"]
                errors = m.anchor_check(rows, self.root / "build", **options)
                self.assertTrue(any(message in error for error in errors), errors)

    def test_anchor_retirement_requires_target_and_original_page(self):
        rows, options = self.anchor_retirement_fixture()
        self.write("build/api/functions.html", '<article id="documenter-page"><h2 id="Still-required"></h2></article>')
        self.write("build/other.html", '<article id="documenter-page"><h2 id="Indicators"></h2></article>')
        errors = m.anchor_check(rows, self.root / "build", **options)
        self.assertTrue(any("target on its original page" in error for error in errors), errors)
        (self.root / "build/api/functions.html").unlink()
        errors = m.anchor_check(rows, self.root / "build", **options)
        self.assertTrue(any("missing baseline page: api/functions" in error for error in errors), errors)

    def test_anchor_retirement_keeps_unrelated_missing_fragments_blocking(self):
        rows, options = self.anchor_retirement_fixture()
        self.write("build/api/functions.html", '<article id="documenter-page"><h2 id="Indicators"></h2></article>')
        errors = m.anchor_check(rows, self.root / "build", **options)
        self.assertEqual(errors, ["missing original page+fragment: api/functions#Still-required"])

    def test_anchor_retirement_requires_validated_mapping_and_review_evidence(self):
        for case, message in (("mapping", "validated upstream mapping"),
                               ("wrong-source-page", "validated upstream mapping"),
                               ("reviewer", "independent review"), ("evidence", "audit evidence")):
            with self.subTest(case=case):
                rows, options = self.anchor_retirement_fixture()
                if case == "mapping":
                    options["mappings"] = {}
                elif case == "wrong-source-page":
                    old = options["base_blocks"][0]
                    replacement = next(m.blocks("## Indicators\n", "docs/src/other.md"))
                    options["mappings"] = {old.id: ("upstream", [replacement])}
                else:
                    options["retirements"][0][case] = ""
                errors = m.anchor_check(rows, self.root / "build", **options)
                self.assertTrue(any(message in error for error in errors), errors)

    def test_anchor_retirement_accepts_a_second_unrelated_heading(self):
        rows, options = self.anchor_retirement_fixture()
        old = next(m.blocks("## Input `Float32` & ℓ₂\n", "docs/src/other.md"))
        replacement = next(m.blocks("## More input rules\n", old.path))
        record = dict(options["retirements"][0], page="other", block_id=old.id,
                      baseline_sha=old.sha, fragments=["Input-Float32-and-ℓ", "Input-Float32-and-ℓ-1"],
                      target="More-input-rules")
        options["retirements"].append(record)
        options["base_blocks"].append(old)
        options["mappings"][old.id] = ("upstream", [replacement])
        rows.extend(dict(page="other", fragment=fragment) for fragment in ["", *record["fragments"]])
        self.write("build/other.html", '<article id="documenter-page"><h2 id="More-input-rules"></h2></article>')
        report = {}
        self.assertEqual(m.anchor_check(rows, self.root / "build", report=report, **options), [])
        self.assertEqual(report["approved_retirement_count"], 4)

    def test_anchor_retirement_accepts_index_pages_and_explicit_successor(self):
        for page, path, html in (("developer", "docs/src/developer/index.md", "build/developer/index.html"),
                                  ("", "docs/src/index.md", "build/index.html")):
            with self.subTest(path=path):
                old = next(m.blocks("## Release policy\n", path))
                replacement = next(m.blocks("## [Updated release policy](@id current-release-policy)\n", path))
                record = dict(page=page, fragments=["Release-policy", "Release-policy-1"],
                              target="current-release-policy", block_id=old.id, baseline_sha=old.sha,
                              reviewer="fixture-reviewer@fixture", evidence="Fixture authored-reference audit.")
                rows = [dict(page=page, fragment=fragment) for fragment in ["", *record["fragments"]]]
                self.write(html, '<article id="documenter-page"><h2 id="current-release-policy"></h2></article>')
                self.assertEqual(m.anchor_check(rows, self.root / "build", retirements=[record],
                                               base_blocks=[old], mappings={old.id: ("upstream", [replacement])}), [])

    def test_anchor_retirement_rejects_ambiguous_duplicate_and_numbered_titles(self):
        for text in ("### Nonnegative orthant", "### Nonnegative-orthant-1",
                     "### [Separate explicit heading](@id Nonnegative-orthant)"):
            with self.subTest(text=text):
                rows, options = self.anchor_retirement_fixture()
                old = options["base_blocks"][0]
                options["base_blocks"].append(m.Block("other-heading", old.path, "heading", text, 4, text))
                errors = m.anchor_check(rows, self.root / "build", **options)
                self.assertTrue(any("ambiguous duplicate or numbered" in error for error in errors), errors)

    def test_anchor_retirement_rejects_malformed_records_and_unparsed_markup(self):
        for case in ("missing-block", "empty-fragments", "non-string-target", "formatted-heading"):
            with self.subTest(case=case):
                rows, options = self.anchor_retirement_fixture()
                record = options["retirements"][0]
                if case == "missing-block":
                    del record["block_id"]
                elif case == "empty-fragments":
                    record["fragments"] = []
                elif case == "non-string-target":
                    record["target"] = []
                else:
                    old = options["base_blocks"][0]
                    formatted = m.Block(old.id, old.path, "heading", "## **Nonnegative orthant**", 1, old.heading)
                    options["base_blocks"] = [formatted]
                    record["baseline_sha"] = formatted.sha
                errors = m.anchor_check(rows, self.root / "build", **options)
                self.assertTrue(errors)
                self.assertTrue(any("missing original page+fragment" in error for error in errors))

    def test_sidebar_anchor_changes_are_excluded(self):
        page = '<nav><input id="collapse-1"></nav><article id="documenter-page"><h1 id="kept">Text</h1></article>'
        self.assertEqual(m.ContentAnchors(page).ids, {"documenter-page", "kept"})

    def anchor_snapshot_fixture(self):
        audit = self.root / "audit"
        audit.mkdir()
        rows = [dict(page="old", fragment=""), dict(page="old", fragment="original-fragment")]
        m.write_tsv(audit / "anchors.tsv", ("page", "fragment"), rows)
        metadata = dict(baseline=m.BASELINE, rendered_pages=1, content_anchor_count=1,
                        anchor_snapshot=dict(file="anchors.tsv", sha256=m.digest((audit / "anchors.tsv").read_text())))
        return audit, metadata

    def test_one_complete_anchor_snapshot_has_no_runtime_profile_selection(self):
        audit, metadata = self.anchor_snapshot_fixture()
        self.write("build/old.html", '<article id="documenter-page"><h1 id="original-fragment">Old</h1></article>')
        self.assertEqual(m.anchor_check(m.anchor_snapshot(metadata, audit), self.root / "build"), [])
        del metadata["anchor_snapshot"]
        with self.assertRaisesRegex(ValueError, "complete frozen"):
            m.anchor_snapshot(metadata, audit)

    def test_empty_or_truncated_anchor_snapshot_is_not_accepted(self):
        audit, metadata = self.anchor_snapshot_fixture()
        path = audit / "anchors.tsv"
        for rows, error in (([], "empty"), ([dict(page="old", fragment="")], "anchor count"),
                            ([dict(page="old", fragment="original-fragment")], "page records")):
            with self.subTest(rows=rows):
                m.write_tsv(path, ("page", "fragment"), rows)
                metadata["anchor_snapshot"]["sha256"] = m.digest(path.read_text())
                with self.assertRaisesRegex(ValueError, error):
                    m.anchor_snapshot(metadata, audit)

    def test_anchor_snapshot_hash_is_checked(self):
        audit, metadata = self.anchor_snapshot_fixture()
        path = audit / "anchors.tsv"
        path.write_text(path.read_text().replace("original-fragment", "different-fragment"))
        with self.assertRaisesRegex(ValueError, "snapshot hash mismatch"):
            m.anchor_snapshot(metadata, audit)

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


    def example_reword_fixture(self):
        original = '```@example old\ns = "a  b"\n@assert s == "a  b"\n```\n'
        self.baseline = list(m.blocks(original, self.source))
        self.rows = m.initial_ledger(self.baseline)
        changed = original.replace('@example old', '@example new')
        self.write(self.source, changed)
        replacement = next(m.blocks(changed, self.source))
        self.rows[0].update(action='reword', dest_sha=replacement.sha, reviewer='independent@reviewed-revision')
        record = dict(block_id=self.baseline[0].id, baseline_sha=self.baseline[0].sha,
                      source_shas=[self.baseline[0].sha], action='reword', dest=self.source,
                      dest_sha=replacement.sha, reviewer=self.rows[0]['reviewer'],
                      evidence='Fixture independent rendered review of exactly the sandbox-label change.',
                      changes={name: m.token_delta(counter(self.baseline[0]), counter(replacement))
                               for name, counter in [('tokens', m.protected_tokens), ('examples', m.example_tokens)]})
        return original, changed, record

    def save_allowance(self, record):
        self.write('docs/audit/docs-tracks-1019/token-allowances.json',
                   json.dumps(dict(schema=1, allowances=[record])))

    def test_reword_does_not_exempt_example_label_without_exact_allowance(self):
        self.example_reword_fixture()
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        self.assertTrue(m.accounting(self.baseline, mappings, 'tokens'))
        self.assertTrue(m.accounting(self.baseline, mappings, 'examples'))

    def test_exact_reviewed_sandbox_label_allowance_retains_assertions(self):
        _, _, record = self.example_reword_fixture()
        self.save_allowance(record)
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        for kind in ('tokens', 'sentences', 'examples'):
            self.assertEqual(m.accounting(self.baseline, mappings, kind), [])

    def test_label_allowance_rejects_changed_assertion_even_with_refreshed_destination(self):
        _, changed, record = self.example_reword_fixture()
        changed = changed.replace('@assert s == "a  b"', '@assert s != "a  b"')
        self.write(self.source, changed)
        self.rows[0]['dest_sha'] = record['dest_sha'] = next(m.blocks(changed, self.source)).sha
        self.save_allowance(record)
        self.assertTrue(any('exactly match every removed' in e for e in self.check()[0]))

    def test_label_allowance_rejects_removed_assertion_and_changed_literal_whitespace(self):
        for mutation in ('remove-assertion', 'literal-space'):
            with self.subTest(mutation=mutation):
                _, changed, record = self.example_reword_fixture()
                changed = changed.replace('@assert s == "a  b"\n', '') if mutation == 'remove-assertion' else changed.replace('a  b', 'a b')
                self.write(self.source, changed)
                self.rows[0]['dest_sha'] = record['dest_sha'] = next(m.blocks(changed, self.source)).sha
                self.save_allowance(record)
                self.assertTrue(any('exactly match every removed' in e for e in self.check()[0]))

    def test_reword_changed_numeric_bound_is_not_covered_by_review_string(self):
        text = 'The error is at most `8 eps(T)`.\n'
        self.baseline = list(m.blocks(text, self.source))
        self.rows = m.initial_ledger(self.baseline)
        changed = text.replace('8 eps', '16 eps')
        self.write(self.source, changed)
        self.rows[0].update(action='reword', dest_sha=next(m.blocks(changed, self.source)).sha,
                            reviewer='independent@fixture')
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        self.assertTrue(m.accounting(self.baseline, mappings, 'tokens'))

    def test_split_does_not_normalize_away_literal_whitespace(self):
        text = 'Keep ``a  b`` then finish.\n'
        self.baseline = list(m.blocks(text, self.source))
        self.rows = m.initial_ledger(self.baseline)
        changed = text.replace('a  b', 'a b')
        self.write(self.source, changed)
        self.rows[0].update(action='split', dest_sha=next(m.blocks(changed, self.source)).sha,
                            reviewer='independent@fixture')
        errors, mappings = self.check()
        self.assertEqual(errors, [])  # Whitespace-word counting alone is insufficient.
        self.assertTrue(m.accounting(self.baseline, mappings, 'tokens'))

    def test_allowance_rejects_stale_unlisted_unused_and_missing_evidence(self):
        for mutation in ('hash', 'missing-kind', 'extra-loss', 'unused', 'evidence', 'unknown'):
            with self.subTest(mutation=mutation):
                original, _, record = self.example_reword_fixture()
                if mutation == 'hash':
                    record['dest_sha'] = '0' * 64
                elif mutation == 'missing-kind':
                    del record['changes']['examples']
                elif mutation == 'extra-loss':
                    record['changes']['tokens']['removed']['number:999'] = 1
                elif mutation == 'unused':
                    self.write(self.source, original)
                    self.rows[0]['dest_sha'] = record['dest_sha'] = self.baseline[0].sha
                elif mutation == 'evidence':
                    record['evidence'] = ''
                else:
                    record['block_id'] = 'no-such-root'
                self.save_allowance(record)
                self.assertTrue(self.check()[0])

    def test_reword_and_split_cannot_reuse_one_destination_occurrence(self):
        self.baseline = list(m.blocks('First `0` rule.\n\nSecond `0` rule.\n', self.source))
        self.rows = m.initial_ledger(self.baseline)
        self.write(self.source, 'Equivalent `0` rule.\n')
        candidate = next(m.blocks('Equivalent `0` rule.\n', self.source))
        for row in self.rows:
            row.update(action='reword', dest_sha=candidate.sha, reviewer='independent@fixture')
        self.assertTrue(any('occurrence reused' in e for e in self.check()[0]))

    def test_refinement_children_reconstruct_parent_and_preserve_other_ids(self):
        source = 'docs/src/smooth.md'
        original = '# Smooth\n\nFirst `0` statement. Second `1` statement.\n\nLast.\n'
        self.baseline = list(m.blocks(original, source))
        expected = m.refinement_records(self.baseline)
        self.assertEqual(len(expected), 1)
        self.write('docs/audit/docs-tracks-1019/refinements.json',
                   json.dumps(dict(schema=1, baseline=m.BASELINE, parents=expected)))
        result = m.validated_refinements(self.baseline, self.root)
        self.assertEqual(len(result[self.baseline[1].id]), 2)
        self.assertEqual([b.id for b in result[self.baseline[1].id]],
                         [self.baseline[1].id + '::sentence:1', self.baseline[1].id + '::sentence:2'])
        expected[0]['children'].reverse()
        self.write('docs/audit/docs-tracks-1019/refinements.json',
                   json.dumps(dict(schema=1, baseline=m.BASELINE, parents=expected)))
        with self.assertRaisesRegex(ValueError, 'complete ordered'):
            m.validated_refinements(self.baseline, self.root)

    def test_parent_and_child_occurrences_cannot_cover_two_roots(self):
        target = 'docs/src/smooth.md'
        paragraph = 'First `0` statement. Second `1` statement.'
        parent = next(m.blocks(paragraph, target))
        child = m.Block('separate-root', self.source, 'prose', 'First `0` statement.', 1, '')
        self.baseline = [parent, child]
        self.rows = m.initial_ledger(self.baseline)
        self.rows[1].update(action='move', dest=target)
        self.write(target, paragraph)
        self.assertTrue(any('occurrence reused' in e for e in self.check()[0]))

    def runtime_identity_fixture(self, pretty=False):
        title = 'Exact title'
        source = 'docs/src/old.md'
        self.write(source, '!!! note "Exact title"\n    Retained body.\n')
        identity = dict(page='old', category='note', title=title, source=source,
                        fragments=['Exact-title-old-runtime', 'Exact-title-other-runtime'])
        html = ('<article id="documenter-page"><div class="admonition is-info" id="Exact-title-new-runtime">'
                '<header class="admonition-header">Exact title<a class="admonition-anchor" '
                'href="#Exact-title-new-runtime"></a></header><div class="admonition-body">Retained body.</div>'
                '</div><h2 id="ordinary-required">Other</h2></article>')
        path = 'build/old/index.html' if pretty else 'build/old.html'
        self.write(path, html)
        rows = [dict(page='old', fragment=value) for value in ['', 'Exact-title-old-runtime', 'ordinary-required']]
        return rows, identity, path, html

    def test_runtime_identity_accepts_new_runtime_but_keeps_ordinary_anchors(self):
        rows, identity, path, html = self.runtime_identity_fixture()
        report = {}
        self.assertEqual(m.anchor_check(rows, self.root/'build', identities=[identity], root=self.root, report=report), [])
        self.assertEqual(report['runtime_identity_count'], 1)
        self.assertEqual(report['exact_fragment_count'], 1)
        self.assertEqual(report['accounted_fragment_count'], 2)
        self.write(path, html.replace('id="ordinary-required"', 'id="changed"'))
        self.assertTrue(any('ordinary-required' in e for e in m.anchor_check(rows, self.root/'build', identities=[identity], root=self.root)))

    def test_runtime_identity_rejects_missing_duplicate_and_wrong_category(self):
        for mutation in ('missing', 'duplicate', 'category', 'title', 'source-category'):
            with self.subTest(mutation=mutation):
                rows, identity, path, html = self.runtime_identity_fixture()
                if mutation == 'missing':
                    html = '<article id="documenter-page"><h2 id="ordinary-required"></h2></article>'
                elif mutation == 'duplicate':
                    html = html.replace('</article>', html[html.index('<div'):html.index('<h2')]+'</article>')
                elif mutation == 'category':
                    html = html.replace('is-info', 'is-warning')
                elif mutation == 'title':
                    html = html.replace('Exact title', 'Another title')
                else:
                    self.write(identity['source'], '!!! warning "Exact title"\n    Retained body.\n')
                self.write(path, html)
                self.assertTrue(m.anchor_check(rows, self.root/'build', identities=[identity], root=self.root))

    def test_runtime_identity_rejects_nonself_rendered_and_authored_links(self):
        for pretty in (False, True):
            with self.subTest(pretty=pretty):
                rows, identity, path, html = self.runtime_identity_fixture(pretty)
                href = '../old/#Exact-title-new-runtime' if pretty else 'old.html#Exact-title-new-runtime'
                other = 'build/other/index.html' if pretty else 'build/other.html'
                self.write(other, '<article id="documenter-page"><a href="'+href+'">Authored</a></article>')
                errors = m.anchor_check(rows, self.root/'build', identities=[identity], root=self.root)
                self.assertTrue(any('non-self link' in e for e in errors), errors)
                (self.root/other).unlink()
        rows, identity, _, _ = self.runtime_identity_fixture()
        self.write('README.md', '[Old note](docs/old.html#Exact-title-old-runtime)\n')
        errors = m.anchor_check(rows, self.root/'build', identities=[identity], root=self.root)
        self.assertTrue(any('authored link' in e for e in errors), errors)

    def test_runtime_identity_rejects_reference_and_padded_markdown_destinations(self):
        target = 'docs/old.html#Exact-title-old-runtime'
        for text in (
                '[See note][old-note]\n\n[old-note]: ' + target + '\n',
                '[See note][old-note]\n\n[old-note]: <' + target + '> "Title"\n',
                '[See note][old-note]\n\n[old-note]:\n    <' + target + '>\n',
                '[See note]( ' + target + ' )\n',
                '[See note]( <' + target + '> "Title" )\n',
                '[See note](\n  <' + target + '>\n)\n'):
            with self.subTest(text=text):
                rows, identity, _, _ = self.runtime_identity_fixture()
                self.write('README.md', text)
                errors = m.anchor_check(rows, self.root/'build', identities=[identity], root=self.root)
                self.assertTrue(any('authored link' in e for e in errors), errors)

    def test_runtime_identity_rejects_autolinks_and_unquoted_html(self):
        for text in (
                '<https://mlakolar.github.io/ProximalBase.jl/stable/old/#Exact-title-old-runtime>',
                '<a href=docs/old.html#Exact-title-old-runtime>See note</a>',
                '<A HREF="docs/old.html&#35;Exact-title-old-runtime">See note</A>',
                '[See note](docs/old.html&#35;Exact-title-old-runtime)'):
            with self.subTest(text=text):
                rows, identity, _, _ = self.runtime_identity_fixture()
                self.write('README.md', text)
                errors = m.anchor_check(rows, self.root/'build', identities=[identity], root=self.root)
                self.assertTrue(any('authored link' in e for e in errors), errors)



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

    def select(self, kind, marker=None):
        old = next(b for b in m.blocks(self.original, self.path) if b.kind == kind and (marker is None or marker in b.text))
        new = next(b for b in m.blocks(self.changed, self.path) if b.kind == kind and (marker is None or marker in b.text))
        self.baseline = [old]
        self.rows = m.initial_ledger(self.baseline)
        self.rows[0].update(action="upstream", dest_sha=new.sha, reviewer="fixture-reviewer@upstream")
        self.event = self.make_event(self.before, self.after, old.sha, new.sha)
        self.record = dict(block_id=old.id, baseline_sha=old.sha, dest=self.path,
                           dest_sha=new.sha, reviewer=self.rows[0]["reviewer"],
                           events=[self.event])
        self.save()

    def make_event(self, before, after, source_sha, replacement_sha, number=1020):
        return dict(source_sha=source_sha, replacement_sha=replacement_sha,
                    pull_request=number, pull_request_url=f"https://github.com/mlakolar/ProximalBase.jl/pull/{number}",
                    merged=True, before_commit=before, after_commit=after,
                    before_blob=m.git("rev-parse", f"{before}:{self.path}", root=self.root).strip(),
                    after_blob=m.git("rev-parse", f"{after}:{self.path}", root=self.root).strip(),
                    reviewer="fixture-reviewer@upstream")

    def save(self, records=None):
        self.receipt.write_text(json.dumps(dict(schema=2, changes=[self.record] if records is None else records)))

    def check(self):
        return m.coverage(self.baseline, self.rows, self.root)

    def assert_rejected(self, message):
        errors, mappings = self.check()
        self.assertTrue(any(message in error for error in errors), errors)
        self.assertNotIn(self.baseline[0].id, mappings)
        for kind in ("tokens", "sentences", "examples"):
            self.assertTrue(m.accounting(self.baseline, mappings, kind))

    def migrate_reviewed_upstream_fixture(self):
        old = self.baseline[0]
        terminal = next(b for b in m.blocks(self.changed, self.path) if b.sha == self.record['dest_sha'])
        target = 'docs/src/spec/family.md'
        file = self.root / target
        file.parent.mkdir(parents=True)
        file.write_text(terminal.text + '\n')
        (self.root / self.path).write_text(self.changed.replace(terminal.text, 'See the family contract.'))
        self.rows[0].update(action='move', dest=target, dest_sha=terminal.sha, reviewer='mechanical')
        basis = dict(block_id=old.id, baseline_sha=old.sha, upstream_sha=m.receipt_digest(self.record),
                     source_shas=[terminal.sha], evidence='Fixture: retain the independently reviewed upstream chain before the move.')
        self.basis_file = self.receipt.parent / 'migration-bases.json'
        self.basis_file.write_text(json.dumps(dict(schema=1, bases=[basis])))
        return terminal, target, basis

    def test_upstream_terminal_span_can_move_without_erasing_historical_receipt(self):
        terminal, _, _ = self.migrate_reviewed_upstream_fixture()
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(mappings.bases[self.baseline[0].id][0].sha, terminal.sha)
        for kind in ('tokens', 'sentences', 'examples'):
            self.assertEqual(m.accounting(self.baseline, mappings, kind), [])
        self.assertEqual(json.loads(self.receipt.read_text())['changes'], [self.record])

    def test_upstream_migration_checks_terminal_tokens_not_obsolete_original_tokens(self):
        terminal, target, _ = self.migrate_reviewed_upstream_fixture()
        changed = terminal.text.replace('`1`', '`0`')
        (self.root/target).write_text(changed + '\n')
        self.rows[0].update(action='reword', dest_sha=next(m.blocks(changed, target)).sha,
                            reviewer='independent@migration')
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        self.assertTrue(m.accounting(self.baseline, mappings, 'tokens'))

    def test_upstream_migration_rejects_stale_basis_and_invalid_historical_chain(self):
        _, _, basis = self.migrate_reviewed_upstream_fixture()
        basis['upstream_sha'] = '0' * 64
        self.basis_file.write_text(json.dumps(dict(schema=1, bases=[basis])))
        self.assert_rejected('exact reviewed upstream terminal span')
        basis['upstream_sha'] = m.receipt_digest(self.record)
        self.basis_file.write_text(json.dumps(dict(schema=1, bases=[basis])))
        self.event['merged'] = False
        self.save()
        self.assert_rejected('merged pull request')

    def test_unused_migration_basis_does_not_become_an_exemption(self):
        old = self.baseline[0]
        basis = dict(block_id=old.id, baseline_sha=old.sha, upstream_sha=m.receipt_digest(self.record),
                     source_shas=[self.record['dest_sha']], evidence='Fixture.')
        (self.receipt.parent/'migration-bases.json').write_text(json.dumps(dict(schema=1, bases=[basis])))
        self.assert_rejected('unused migration basis')

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
        self.assert_rejected("migration evidence")
        for field, value, error in (("merged", False, "merged pull request"),
                                     ("reviewer", "mechanical", "independent review")):
            with self.subTest(field=field):
                self.select("prose")
                if field == "reviewer":
                    self.rows[0][field] = self.record[field] = value
                else:
                    self.event[field] = value
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
                                     ("before_blob", self.event["after_blob"], "blob mismatch"),
                                     ("before_commit", self.after, "first parent")):
            with self.subTest(field=field):
                self.select("prose")
                (self.event if field in {"before_blob", "before_commit"} else self.record)[field] = value
                self.save()
                self.assert_rejected(error)

    def test_upstream_rejects_unrelated_changed_block_as_replacement(self):
        unrelated = list(m.blocks(self.changed, self.path))[-1]
        self.rows[0]["dest_sha"] = self.record["dest_sha"] = self.event["replacement_sha"] = unrelated.sha
        self.save()
        self.assert_rejected("share a changed diff hunk")

    def test_upstream_rejects_hash_not_present_in_recorded_after_blob(self):
        mutation = self.changed.replace("Updated `1`", "Unreviewed `2`")
        (self.root / self.path).write_text(mutation)
        new = next(b for b in m.blocks(mutation, self.path) if b.kind == "prose")
        self.rows[0]["dest_sha"] = self.record["dest_sha"] = self.event["replacement_sha"] = new.sha
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

    def extend_chain(self, intervening=False):
        before, text = self.after, self.changed
        if intervening:
            text += "\nAn intervening unrelated paragraph.\n"
            before = self.commit(text, "unrelated intervening file edit")
        self.final = text.replace("Updated `1` rule.", "Final `2` rule.").replace(
            "A different unrelated rule.", "A final unrelated rule.")
        after = self.commit(self.final, "second upstream change")
        new = next(b for b in m.blocks(self.final, self.path) if b.kind == "prose")
        event = self.make_event(before, after, self.record["dest_sha"], new.sha, number=1022)
        event["reviewer"] = "second-fixture-reviewer@second-upstream"
        self.record["events"].append(event)
        self.rows[0]["dest_sha"] = self.record["dest_sha"] = new.sha
        self.rows[0]["reviewer"] = self.record["reviewer"] = event["reviewer"]
        self.save()
        return event

    def test_upstream_chain_accepts_intervening_unrelated_file_edits(self):
        second = self.extend_chain(intervening=True)
        self.assertNotEqual(self.event["after_commit"], second["before_commit"])
        self.assertNotEqual(self.event["after_blob"], second["before_blob"])
        self.assertEqual(self.event["replacement_sha"], second["source_sha"])
        self.assertNotEqual(self.event["reviewer"], second["reviewer"])
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        for kind in ("tokens", "sentences", "examples"):
            self.assertEqual(m.accounting(self.baseline, mappings, kind), [])

    def test_upstream_chain_rejects_empty_skipped_reordered_and_duplicate_events(self):
        self.extend_chain()
        first, second = copy.deepcopy(self.record["events"])
        for events, error in (([], "nonempty events"),
                               ([second], "source hash breaks event continuity"),
                               ([first], "final replacement"),
                               ([second, first], "source hash breaks event continuity"),
                               ([first, first, second], "source hash breaks event continuity")):
            with self.subTest(events=events):
                self.record["events"] = events
                self.save()
                self.assert_rejected(error)

    def test_upstream_chain_validates_later_event_evidence(self):
        second = self.extend_chain(intervening=True)
        original = copy.deepcopy(second)
        for field, value, error in (("source_sha", self.baseline[0].sha, "event continuity"),
                                     ("before_blob", self.event["before_blob"], "blob mismatch"),
                                     ("before_commit", self.before, "first parent"),
                                     ("replacement_sha", "0" * 64, "unique original and replacement"),
                                     ("merged", False, "merged pull request"),
                                     ("reviewer", "mechanical", "independent review")):
            with self.subTest(field=field):
                second.clear()
                second.update(original)
                second[field] = value
                self.save()
                self.assert_rejected(error)

    def test_upstream_chain_binds_final_reviewer_and_current_destination(self):
        second = self.extend_chain()
        self.rows[0]["reviewer"] = self.record["reviewer"] = "other-reviewer@other-revision"
        self.save()
        self.assert_rejected("final event reviewer")
        self.rows[0]["reviewer"] = self.record["reviewer"] = second["reviewer"]
        self.save()
        (self.root / self.path).write_text(self.final.replace("Final `2` rule.", "Unreviewed `3` rule."))
        self.assert_rejected("no block with expected hash")

    def test_upstream_chain_rejects_unrelated_later_diff_hunk(self):
        second = self.extend_chain()
        unrelated = list(m.blocks(self.final, self.path))[-1]
        self.rows[0]["dest_sha"] = self.record["dest_sha"] = second["replacement_sha"] = unrelated.sha
        self.save()
        self.assert_rejected("share a changed diff hunk")

    def test_upstream_chain_keeps_original_file_restriction(self):
        self.extend_chain()
        target = self.root / "docs/src/other.md"
        target.parent.mkdir(parents=True)
        target.write_text(self.final)
        self.rows[0]["dest"] = self.record["dest"] = "docs/src/other.md"
        self.save()
        self.assert_rejected("original file")

    def split_fixture(self):
        self.path = "docs/src/numerics.md"
        (self.root / self.path).parent.mkdir(parents=True, exist_ok=True)
        self.original = "# Rules\n\nApply the group rule and norm rule; no other rule is claimed.\n\n## Other\n\nUnrelated text.\n"
        self.changed = "# Rules\n\nApply the group rule. Apply the norm rule.\nNo other rule is claimed.\n\n## Other\n\nUnrelated text.\n"
        self.before = self.commit(self.original, "before split")
        self.after = self.commit(self.changed, "upstream sentence split")
        self.select("prose")
        self.pieces = [block for block in m.blocks(self.changed, self.path) if block.kind == "prose"][:3]
        del self.event["replacement_sha"]
        self.set_split_hashes([block.sha for block in self.pieces])

    def set_split_hashes(self, hashes):
        self.event["replacement_shas"] = hashes
        self.rows[0]["dest"] = self.record["dest"] = ";".join([self.path] * len(hashes))
        self.rows[0]["dest_sha"] = self.record["dest_sha"] = ";".join(hashes)
        self.save()

    def test_upstream_split_preserves_all_three_ordered_pieces(self):
        self.split_fixture()
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        self.assertEqual([b.sha for b in mappings[self.baseline[0].id][1]], [b.sha for b in self.pieces])
        for kind in ("tokens", "sentences", "examples"):
            self.assertEqual(m.accounting(self.baseline, mappings, kind), [])

    def test_upstream_split_rejects_missing_reordered_and_extra_receipt_pieces(self):
        self.split_fixture()
        hashes = [block.sha for block in self.pieces]
        extra = next(block.sha for block in m.blocks(self.changed, self.path) if block.text == "## Other")
        for changed, error in ((hashes[:-1], "complete ordered changed span"),
                                ([hashes[0], hashes[2]], "unique original and replacement"),
                                (list(reversed(hashes)), "unique original and replacement"),
                                (hashes + [extra], "complete ordered changed span")):
            with self.subTest(hashes=changed):
                self.set_split_hashes(changed)
                self.assert_rejected(error)
        self.set_split_hashes(hashes[:1])
        self.event["replacement_sha"] = self.event.pop("replacement_shas")[0]
        self.save()
        self.assert_rejected("complete ordered changed span")

    def test_upstream_span_hash_keys_are_exclusive_and_nonempty(self):
        self.split_fixture()
        original = copy.deepcopy(self.event)
        for name in ("source", "replacement"):
            with self.subTest(name=name):
                self.event.clear()
                self.event.update(original)
                self.event[name + "_sha"] = self.baseline[0].sha
                self.event[name + "_shas"] = [self.baseline[0].sha]
                self.save()
                self.assert_rejected("exactly one of")
                del self.event[name + "_sha"]
                self.event[name + "_shas"] = []
                self.save()
                self.assert_rejected("nonempty exact")

    def test_upstream_split_current_pieces_must_remain_contiguous_and_ordered(self):
        self.split_fixture()
        texts = [block.text for block in self.pieces]
        for changed in (list(reversed(texts)), [texts[0], "An unrelated insertion.", *texts[1:]]):
            with self.subTest(text=changed):
                (self.root / self.path).write_text("# Rules\n\n" + " ".join(changed) + "\n")
                self.assert_rejected("uniquely contiguous at the current destination")

    def test_upstream_split_continues_through_later_partial_span_change(self):
        self.split_fixture()
        final = self.changed.replace("Apply the norm rule.", "Apply the newer norm rule.")
        after = self.commit(final, "later change to one split piece")
        hashes = [block.sha for block in m.blocks(final, self.path) if block.kind == "prose"][:3]
        second = self.make_event(self.after, after, self.pieces[0].sha, hashes[0], number=1022)
        del second["source_sha"], second["replacement_sha"]
        second.update(source_shas=[block.sha for block in self.pieces], replacement_shas=hashes)
        self.record["events"].append(second)
        self.rows[0]["dest_sha"] = self.record["dest_sha"] = ";".join(hashes)
        self.save()
        self.assertEqual(self.check()[0], [])
        second["source_shas"].reverse()
        self.save()
        self.assert_rejected("source hash breaks event continuity")

    def test_upstream_split_cannot_omit_new_copy_of_text_existing_elsewhere(self):
        self.original = "# Guide\n\nOriginal combined rule.\n\n## Elsewhere\n\nRepeated obligation.\n"
        self.changed = self.original.replace("Original combined rule.", "First replacement rule.\n\nRepeated obligation.")
        self.before = self.commit(self.original, "before duplicated split piece")
        self.after = self.commit(self.changed, "upstream split introduces an existing phrase")
        self.select("prose")
        # The incomplete scalar receipt must not hide the introduced second copy.
        self.assert_rejected("complete ordered changed span")
        pieces = [block for block in m.blocks(self.changed, self.path) if block.kind == "prose"][:2]
        del self.event["replacement_sha"]
        self.set_split_hashes([block.sha for block in pieces])
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        self.assertEqual([block.line for block in mappings[self.baseline[0].id][1]], [block.line for block in pieces])

    def table_fixture(self, variant=None):
        self.path = "docs/src/api/coverage.md"
        (self.root / self.path).parent.mkdir(parents=True, exist_ok=True)
        old = ["| `First` | Original first rule. |", "| `Second` | Original second rule. |"]
        new = [line.replace("Original", "Updated") for line in old]
        if variant == "reordered":
            new.reverse()
        elif variant == "duplicate":
            old[1], new[1] = old[1].replace("Second", "First"), new[1].replace("Second", "First")
        elif variant == "renamed":
            new[1] = new[1].replace("Second", "Third")
        elif variant == "added":
            new.append("| `Third` | An additional obligation. |")
        elif variant == "deleted":
            new.pop()
        prefix = "# API\n\n| Name | Rule |\n|:--|:--|\n"
        self.original, self.changed = prefix + "\n".join(old) + "\n", prefix + "\n".join(new) + "\n"
        self.before = self.commit(self.original, "before adjacent table changes")
        self.after = self.commit(self.changed, "upstream adjacent table changes")
        self.select("table-row", marker="`First`")

    def test_upstream_adjacent_table_rows_keep_separate_ownership(self):
        self.table_fixture()
        first, row, record = self.baseline[0], self.rows[0], self.record
        self.select("table-row", marker="`Second`")
        self.baseline.insert(0, first)
        self.rows.insert(0, row)
        self.save([record, self.record])
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        self.assertTrue(all(len(replacements) == 1 for _, replacements in mappings.values()))
        self.assertNotEqual(mappings[first.id][1][0].sha, mappings[self.baseline[1].id][1][0].sha)

    def test_upstream_table_row_cannot_claim_its_edited_neighbor(self):
        self.table_fixture()
        other = next(b for b in m.blocks(self.changed, self.path) if "`Second`" in b.text)
        self.rows[0]["dest_sha"] = self.record["dest_sha"] = self.event["replacement_sha"] = other.sha
        self.save()
        self.assert_rejected("share a changed diff hunk")

    def test_upstream_table_pairing_rejects_reordered_duplicate_renamed_and_added_keys(self):
        for variant in ("reordered", "duplicate", "renamed", "added"):
            with self.subTest(variant=variant):
                self.table_fixture(variant)
                self.assert_rejected("complete ordered changed span")

    def test_upstream_deleted_table_row_cannot_reuse_surviving_destination(self):
        self.table_fixture("deleted")
        second = next(b for b in m.blocks(self.original, self.path) if "`Second`" in b.text)
        row = m.initial_ledger([second])[0]
        row.update(action="upstream", dest_sha=self.rows[0]["dest_sha"], reviewer=self.rows[0]["reviewer"])
        event = self.make_event(self.before, self.after, second.sha, row["dest_sha"])
        record = dict(block_id=second.id, baseline_sha=second.sha, dest=self.path,
                      dest_sha=row["dest_sha"], reviewer=row["reviewer"], events=[event])
        self.baseline.append(second)
        self.rows.append(row)
        self.save([self.record, record])
        self.assertTrue(any("destination occurrence reused" in error for error in self.check()[0]))

    def test_upstream_unchanged_neighbors_on_a_changed_line_keep_their_ownership(self):
        self.path = "docs/src/numerics.md"
        (self.root / self.path).parent.mkdir(parents=True, exist_ok=True)
        self.original = "# Rules\n\nAn unchanged leading rule\ncontinues here. Original `0` rule. A retained trailing rule.\n"
        self.changed = self.original.replace("Original `0`", "Original `1`")
        self.before = self.commit(self.original, "before same-line edit")
        self.after = self.commit(self.changed, "upstream same-line edit")
        self.select("prose", marker="Original")
        changed_row = self.rows[0]
        self.baseline = list(m.blocks(self.original, self.path))
        self.rows = [changed_row if row["block_id"] == changed_row["block_id"] else row
                     for row in m.initial_ledger(self.baseline)]
        errors, mappings = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(len(mappings[changed_row["block_id"]][1]), 1)
        for kind in ("tokens", "sentences", "examples"):
            self.assertEqual(m.accounting(self.baseline, mappings, kind), [])

    def test_upstream_character_mapping_cannot_hide_a_new_duplicate_split_piece(self):
        self.path = "docs/src/numerics.md"
        (self.root / self.path).parent.mkdir(parents=True, exist_ok=True)
        for elsewhere in ("Repeated obligation.", "Repeated\nobligation."):
            with self.subTest(elsewhere=elsewhere):
                self.original = "# Rules\n\nOriginal combined rule. Unchanged neighbor.\n\n## Elsewhere\n\n" + elsewhere + "\n"
                self.changed = self.original.replace("Original combined rule.", "First replacement rule. Repeated obligation.")
                self.before = self.commit(self.original, "before same-line duplicated piece")
                self.after = self.commit(self.changed, "upstream same-line duplicated piece")
                self.select("prose")
                self.assert_rejected("complete ordered changed span")
                pieces = [b for b in m.blocks(self.changed, self.path) if b.kind == "prose"][:2]
                del self.event["replacement_sha"]
                self.set_split_hashes([block.sha for block in pieces])
                self.assertEqual(self.check()[0], [])


class RemovedLineTests(unittest.TestCase):
    """Compare real Git trees with staged, unstaged and untracked doc files."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        m.git("init", "--quiet", root=self.root)
        self.write("docs/src/guide.md", "# Guide\n\nAn original rule.\n")
        self.base = self.commit()

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)

    def commit(self):
        m.git("add", "-A", root=self.root)
        m.git("-c", "user.name=Line report fixture", "-c", "user.email=fixture@example.invalid",
              "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture", root=self.root)
        return m.git("rev-parse", "HEAD", root=self.root).strip()

    def report(self, base=None):
        return line_report.removed_lines(self.base if base is None else base, self.root)

    def test_new_post_baseline_text_deletion_is_reported(self):
        self.write("docs/src/guide.md", "# Guide\n\nAn original rule.\nA later rule.\n")
        later_base = self.commit()
        self.write("docs/src/guide.md", "# Guide\n\nAn original rule.\n")
        report = self.report(later_base)
        self.assertEqual(report["unmatched"], [
            {"path": "docs/src/guide.md", "line": 4, "text": "A later rule."}])
        self.assertEqual((report["removed"], report["added"], report["matched"]), (1, 0, 0))

    def test_verbatim_move_to_untracked_destination_preserves_deleted_file(self):
        source = self.root / "docs/src/guide.md"
        self.write("docs/src/new/guide.md", source.read_bytes())
        source.unlink()
        report = self.report()
        self.assertEqual(report["changed_paths"], ["docs/src/guide.md", "docs/src/new/guide.md"])
        self.assertEqual((report["removed"], report["added"], report["matched"]), (2, 2, 2))
        self.assertEqual(report["unmatched"], [])

    def test_only_added_occurrences_credit_duplicate_removals(self):
        self.write("docs/src/guide.md", "# Guide\n\nRepeated rule.\nRepeated rule.\n")
        self.write("docs/src/other.md", "# Other\n\nRepeated rule.\n")
        self.base = self.commit()
        self.write("docs/src/guide.md", "# Guide\n")
        self.write("docs/src/other.md", "# Other\n\nRepeated rule.\nAn unrelated addition.\n")
        self.write("docs/src/new.md", "Repeated rule.\n")
        report = self.report()
        self.assertEqual((report["removed"], report["added"], report["matched"]), (2, 2, 1))
        self.assertEqual([item["text"] for item in report["unmatched"]], ["Repeated rule."])

    def test_current_tree_includes_staged_and_unstaged_changes(self):
        self.write("docs/src/guide.md", "# Guide\n\nStaged removal.\nUnstaged removal.\n")
        self.base = self.commit()
        self.write("docs/src/guide.md", "# Guide\n\nUnstaged removal.\n")
        m.git("add", "docs/src/guide.md", root=self.root)
        self.write("docs/src/guide.md", "# Guide\n")
        self.write("docs/src/new.md", "Staged removal.\n")
        report = self.report()
        self.assertEqual(report["matched"], 1)
        self.assertEqual([item["text"] for item in report["unmatched"]], ["Unstaged removal."])

    def test_literal_whitespace_is_not_normalized(self):
        old = ['s = "a  b"', "\tIndented", "Trailing  ", "Windows\r"]
        self.write("docs/src/guide.md", "\n".join(old) + "\n")
        self.base = self.commit()
        self.write("docs/src/guide.md", 's = "a b"\n Indented\nTrailing\nWindows\n')
        report = self.report()
        self.assertEqual([item["text"] for item in report["unmatched"]], old)

    def test_ignored_generated_destinations_cannot_credit_moves(self):
        self.write(".gitignore", "docs/src/generated/\n")
        self.base = self.commit()
        source = self.root / "docs/src/guide.md"
        self.write("docs/src/generated/guide.md", source.read_bytes())
        source.unlink()
        report = self.report()
        self.assertEqual(report["added"], 0)
        self.assertEqual(report["changed_paths"], ["docs/src/guide.md"])
        self.assertEqual(len(report["unmatched"]), 2)

    def test_reflow_and_relative_link_repairs_remain_reviewable(self):
        self.write("docs/src/guide.md", "A rule on\ntwo lines.\nSee [details](other.md).\n")
        self.base = self.commit()
        (self.root / "docs/src/guide.md").unlink()
        self.write("docs/src/new/guide.md", "A rule on two lines.\nSee [details](../other.md).\n")
        report = self.report()
        self.assertEqual([item["text"] for item in report["unmatched"]],
                         ["A rule on", "two lines.", "See [details](other.md)."])

    def test_empty_tree_and_missing_base_differ_from_no_changes(self):
        self.assertEqual(self.report()["changed_paths"], [])
        empty = subprocess.check_output(["git", "-C", str(self.root), "mktree"], input=b"").decode().strip()
        report = self.report(empty)
        self.assertEqual(report["base_tree"], empty)
        self.assertEqual((report["removed"], report["added"]), (0, 2))
        with self.assertRaises(ValueError):
            self.report("nonexistent-comparison-base")

    def test_clean_sparse_omission_is_not_a_deletion(self):
        m.git("update-index", "--skip-worktree", "docs/src/guide.md", root=self.root)
        (self.root / "docs/src/guide.md").unlink()
        self.assertEqual(self.report()["changed_paths"], [])
        self.assertEqual(self.report()["unmatched"], [])

    def test_changed_binary_and_non_utf8_files_fail_explicitly(self):
        for data, message in ((b"binary\0data", "binary"), (b"bad\xfftext", "UTF-8")):
            with self.subTest(data=data):
                self.write("docs/src/new.md", data)
                with self.assertRaisesRegex(ValueError, message):
                    self.report()

    def test_cli_report_only_blocking_and_configuration_exit_codes(self):
        self.write("docs/check_migration_lines.py", Path(line_report.__file__).read_bytes())
        self.write("docs/src/guide.md", "# Guide\n")
        script = self.root / "docs/check_migration_lines.py"
        for arguments, code in (([self.base], 0), ([self.base, "--blocking"], 1), (["missing-base"], 2)):
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, "-B", str(script), *arguments],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                if code != 2:
                    self.assertEqual(len(json.loads(result.stdout)["unmatched"]), 1)
                else:
                    self.assertIn("could not run", result.stderr)


if __name__ == "__main__":
    unittest.main()
