# NormL1 pilot review

## Independent review

Reviewer: `plan_consistency`, an agent that did not author the migration.

The reviewer compared all fourteen source spans with the frozen baseline and
the destination, inspected the relevant implementation and independent tests,
and compared the rendered old sections with the rendered pilot. All numerical
wording is preserved, with two relative-link repairs. The original
`penalties.html#l1-endpoints` target remains and forwards to the new page.

Two checker defects found during review were repaired before final acceptance:
significant whitespace inside code/math literals is preserved, and heading-like
Julia comments inside fences do not terminate a selected destination section.
The reviewer independently reran all 27 fixtures successfully.

## Reviewed example metadata repair

After the verbatim move, the destination's example name changed in a separate
commit to avoid a Julia module-replacement warning across pages. The entire
executable body and its assertions are unchanged. The exact change is:

```text
old: ```@example numerical_endpoints
new: ```@example norml1_numerical_endpoints
```

The reviewer fetched the destination directly from GitHub at
`127d2dfea08f0eb471b7e991e1289300e540a398` and verified its Git blob.
The single affected ledger row uses the reviewed `reword` action to account for
the changed fence metadata; no numerical rule was reworded.

| Receipt field | Value |
|:--|:--|
| Reviewer and revision | `plan_consistency@127d2dfea08f0eb471b7e991e1289300e540a398` |
| Destination | `docs/src/spec/norml1.md#spec-norml1-endpoints` |
| Original block SHA-256 | `396796c7300e4af881c0508a86148e1c30f391fece93f234e5a9dc89424c6b46` |
| Destination block SHA-256 | `f96cb594134cda45e80ce5192e2d2f845ed17c52c547fb02d7dbb8c70b249fc8` |
| Destination Git blob | `a8edcfeba2e44aaf513672a7fb2cbaaf7f36f4a1` |

This independent review does not substitute for the owner pilot review required
by Phase 2 step 3. See [open questions](open-questions.md).

## Baseline build records

The exact frozen tree was built locally with Julia 1.10.10, including all nine
isolated scripts, Documenter examples, doctests and API coverage. Its HTML check
reported 54 pages, 5,258 checked internal links/assets, and no broken or orphan
pages. All six migration checks passed on the unchanged all-stay ledger.

The initial PR checkpoint also passed the strict documentation CI job, before
any manual text changed. Its normal documentation artifact preserves that
baseline documentation:

- [CI run 44443622327](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44443622327).
- Artifact name: `documentation-linux-current-julia`.
- Artifact ID: `14492512149`.
- Artifact SHA-256: `35c8aa068d12a170ad9f43007b62fa701fcbea499dbefca60abf9cef45183af1`.
- PR checkpoint: `db20ba7564992abadf0c1293a6ff4d10acd07ad5`.

PR CI builds the GitHub merge ref. Upstream commit `2a55362` advanced only the
least-squares implementation and its tests; the documentation sources in that
baseline artifact match the frozen documentation baseline. The implementation
branch incorporates that upstream change without changing its code or tests.
Documentation-only scope is checked against the PR merge base, not against the
older documentation-provenance commit.

## Final local acceptance of the pilot

The final strict build on Julia 1.10.10 ran all nine isolated scripts again,
executed the manual examples and doctests, and passed API coverage. The unique
example name removed the new module-replacement warning. The search-index size
advisory also present in the baseline remains non-failing.

| Check | Result |
|:--|:--|
| Markdown lint | 49 active documentation sources pass |
| Contract schema | One family page with all fourteen headings; no clause statuses or identifiers assigned yet |
| Preservation checker fixtures | 27 pass |
| Block, token, sentence, anchor, citation and example checks | All six pass in blocking acceptance mode |
| Rendered HTML | 56 pages; 5,637 checked internal links/assets; no broken links or orphan pages |
| Baseline URLs | All 54 original pages and 621 original content anchors retained |
| Source/test scope | No changes relative to upstream `2a55362d374fa4299d29db7684659fce8cc8e149` |
| Example-script provenance | All nine original scripts retain their frozen hashes |

The pilot has 44 authored manual pages, 97 executed example fences, 31
page/source doctest fences, 266 manual assertion lines and 92 example-script
assertion lines. The extra example fence supplies the import needed by the
endpoint blocks remaining in the common page. All original examples survive.

The ledger now contains 2,123 unchanged blocks, 38 exact moved blocks and one
independently reviewed example-label repair. CI continues to run the migration
checks in report-only mode until the final lock in Phase 5.
