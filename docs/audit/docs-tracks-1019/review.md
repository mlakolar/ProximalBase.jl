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
| Preservation checker fixtures | 30 pass, including explicit anchor-profile regressions |
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

## Exact anchors across the two build environments

The first CI comparison exposed two generated admonition IDs that differ
between Julia 1.10.10 and 1.13.4. Documenter 1.20.0 constructs these IDs with
Julia's `hash` of the flattened admonition text; that hash is not a stable
cross-version identifier. The authored text and page locations did not change.

Both original snapshots are retained: `anchors.tsv` for the exact local baseline
and `anchors-ci.tsv` from the SHA-verified baseline CI artifact above. The CLI
selects `local` or `ci` explicitly, and CI passes `--anchor-profile ci`.
Each profile is pinned to its file hash, 54 page records and 621 content anchors.
There is no hash stripping, omission of admonition anchors, or fallback to a
profile that happens to pass.

The baseline CI snapshot was compared independently with the final CI artifact
`14493262087` from run `44445716820` (ZIP SHA-256
`e5b6ff2ebbdf98733db612b8799f367d540b9368d544c9a83f63c104adfcdc8a`).
That exact comparison loses no original page or fragment. All six preservation
checks pass with the local profile against the final local build, and with the
CI profile against that final CI build. The profile correction changes no
authored documentation page or numerical behavior.

## Owner review of the pilot

Mladen reviewed PR #1021 at `1e35bcf` against master at `2a55362` on
18 November 2026. The review confirmed that no numerical statement was lost
or altered and approved the fourteen-section layout for the remaining
verbatim family moves, with the stated adjustments. Merge approval is
conditional on the five fixes listed in that review; clause conversion
remains held until the preservation-checker gap is closed.

The three documentation fixes from that review are applied here: the
import-only block in `numerics.md` uses `@setup numerical_endpoints`, the
five forwarding rows become one family row, and the contract page's one
source link and seven test links follow `blob/master`. The five original
numerical endpoint rows remain verbatim on `spec/norml1.md`. The import-only
block was added during the migration, so making it hidden setup removes no
baseline example or assertion. The earlier local acceptance counts above
record the checkpoint before this rendering repair.

The original source and evidence inspection was at
`7ae86e2093060eb124211e2f21f043c6379f97ae` (`7ae86e2`). It covered
`src/functions/norml1.jl` and the seven files named in the evidence table:
`test/core_functions.jl`, `test/minimal_math.jl`, `test/infinite_weights.jl`,
`test/numerical_recovery.jl`, `test/kkt_point.jl`, `test/one_based_axes.jl`,
and `test/symmetric_parameters.jl`. This immutable inspection provenance
belongs to the audit; the rendered manual links to the current files.

The two CI corrections use a depth-one checkout and fetch the exact frozen
baseline commit. A change to `ledger.tsv` activates a blocking source/test
scope check on any branch. The comparison is the tested PR merge tree against
its first parent, or an entire push against its `before` revision; only that
exact comparison commit is fetched when missing. Benchmark and deployment
history settings are unchanged. The six preservation checks remain report-only.

The corrected local strict build passes all nine isolated examples, executable
manual examples, doctests and API coverage. All 30 preservation-checker fixtures
pass, and the six checks pass in blocking mode with the local anchor profile.
The rendered checker finds 56 pages, 5,633 internal links/assets, 135 repository
source/edit links, and no broken links or orphan pages. The original 54 pages
and 621 content anchors still pass exact comparison. No baseline ledger row,
source file, test file or example script changes in these five fixes.

The reviewed-row token/example exemption remains a known preparation gap.
The pilot's sole `reword` row was independently confirmed to change only its
example sandbox name. Clause conversion must not use that exemption; the
required accounting fix and regression cases are tracked in #1019.

The independent `plan_consistency` review confirmed all five corrections in
the source and rendered output. Workflow validation parsed YAML, checked shell
syntax, and exercised nine cases with the exact extracted workflow scripts.
A separate real depth-one Git fixture began with only a PR merge commit:
the baseline and comparison commits were initially absent. The scripts fetched
only those two exact commits at depth one, retained a shallow repository with
three commit objects, and could read the frozen baseline with `git show`.
A docs-only migration passed despite unrelated upstream source changes;
a migration that added its own source change failed the scope guard.
