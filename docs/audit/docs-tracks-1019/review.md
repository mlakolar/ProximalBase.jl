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

This independent review does not substitute for the owner acceptance required
by Phase 2 step 3. See [open questions](open-questions.md).

## Baseline build records

The lead implementation agent built the exact frozen tree locally with Julia
1.10.10, including all nine
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

The lead implementation agent's final strict build on Julia 1.10.10 ran all nine isolated scripts again,
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

## External review accepted by the owner

An external Claude review prepared for Mladen covered PR #1021 at `1e35bcf`
against master at `2a55362`, dated 18 November 2026. The owner accepted that
review and its recommendation to use the fourteen-section layout for the
remaining verbatim family moves, with the stated adjustments. The review found
no numerical statement lost or altered. The owner approved merge subject to
the five requested corrections.

The external reviewer read the diff, ran the Python checks and inspected the
published preview. It did not run the Julia documentation build or anchor
check, or verify CI results. The build and CI evidence recorded in this file
was checked by the implementation team. Clause conversion remains held until
the preservation-checker gap is closed. Owner acceptance does not mean that
the owner personally performed the review or reran those checks.

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

## Upstream reconciliation and complete anchor inventory

While the corrections from the accepted external review were in CI, master merged
[PR #1020](https://github.com/mlakolar/ProximalBase.jl/pull/1020), restoring
`IndBox`, at `fc95196390b8f1a164621625362984f03f728ab0`. GitHub reported the PR
merged at that revision. Its actual first parent is
`2a55362d374fa4299d29db7684659fce8cc8e149`; the PR API's older base SHA is not
the landed diff boundary. The tested merge tree therefore inherited four
changed baseline blocks, which the report-only checks correctly reported.

The `upstream` action now records three of these changes in `upstream.json`:
the README function inventory, the API `@docs` fence, and the migration scope
paragraph. `norml1_pilot@fc95196390b8f1a164621625362984f03f728ab0` independently
verified the original and replacement text, actual file blobs, and merged PR.
The original block IDs and hashes remain unchanged. The checker recomputes
the parent, blob, uniqueness and diff-hunk evidence and requires the current
replacement to match the external change exactly. It does not authenticate
GitHub merge status or reviewer identity; those are reviewed attestations.
The action currently covers one changed block in the same file, without a
general rename, move or deletion exemption.

The fourth changed block was the old `Nonnegative orthant` heading. It is
restored verbatim as a forwarding stub to `Box and nonnegative orthant`,
preserving both original generated fragments while retaining the new upstream
heading and single API fence. The family index adds an `IndBox` entry pointing
to its existing API and numerical owners. No new family contract is migrated.

This inspection exposed a separate anchor-parser defect: the Boolean article
flag stopped at the first nested API docstring article. The earlier claim that
621 IDs covered the complete rendered baseline was therefore incomplete.
Article-depth parsing recovers 92 additional IDs: 28 in `api/functions`,
33 in `api/interface`, and 31 in `api/iterates`. Both profiles were re-extracted
from the same frozen HTML, with the original CI ZIP checksum verified again.
They add those IDs and remove none. The corrected baseline is **54 pages and
713 content IDs**, superseding the earlier completeness count. Former snapshot
hashes remain in `baseline.json` for provenance.

The independent reviewer used a different article-ancestor stack parser
directly on the original local HTML and CI ZIP. Both corrected snapshots match
their complete source artifacts exactly: 767 page/fragment pairs, consisting
of 54 page records and 713 IDs. The nested-article regression checks that an API
docstring does not hide later headings and that unrelated outside articles
remain excluded. A third runtime-dependent admonition ID, `Non-finite vectors`
on `api/iterates`, is now covered alongside the two already known IDs. No
fragment is waived by this correction.

The reconciled strict local build passes all nine isolated scripts, executable
examples, doctests and API coverage. All **40 fixtures** and all six preservation
checks pass in blocking mode. Rendered checks find **56 pages, 5,649 internal
links/assets and 136 repository source/edit links**, with no broken links or
orphan pages. The corrected 713-ID baseline passes. The ledger remains at
2,162 original rows: 2,120 stay, 38 move, one reviewed example-label repair,
and three reviewed upstream changes. Source, tests and example scripts have
no PR diff against the current upstream base.

The updated real shallow-Git fixture starts with only the tested PR merge
commit. It fetches exactly three distinct provenance revisions at depth one:
the frozen baseline, upstream before, and upstream after (also the scope base).
Only those four commit objects are present afterward; the checkout remains
shallow. Provenance reads succeed, inherited source changes pass the scope
check, and a PR-owned source change fails. No full history is fetched.

The independent `plan_consistency` review passed the upstream checker,
all three receipts and the completed rendering. Its separate parser found
762 IDs on the current 56 pages and confirmed every one of the 713 original
IDs, including both historical and both new API heading fragments exactly once.

## Successive upstream changes

The actual master ref advanced again before merge, through
[PR #1022](https://github.com/mlakolar/ProximalBase.jl/pull/1022), restoring
`IndBallL2` at `9acd0a5d3a806f7254b02d9b92faf866e6b3c429`. Its actual first
parent is `fc95196390b8f1a164621625362984f03f728ab0`. The PR metadata for #1021
still reported its previously tested base, so the final merge check must read
the current master ref rather than treating that metadata as current.

This second restore changes the same three blocks and renames the API group to
`Indicators`. Both `Nonnegative orthant` and `Box and nonnegative orthant`
remain as forwarding headings, each pointing directly to `Indicators`. The
three indicator docstrings remain in the one upstream API fence. An additive
`IndBallL2` index row links to its existing API, range and endpoint owners.

Schema 2 receipts retain the complete first event for #1020 and append #1022,
reviewed as `norml1_pilot@9acd0a5d3a806f7254b02d9b92faf866e6b3c429`. Every event
has exact source/replacement hashes, before/after commits and blobs, and its own
review identity. The original hash starts the chain, each intermediate hash
joins two events, and the final hash/reviewer match the ledger. Unrelated changes
to other parts of a file may occur between events; the checker does not require
adjacent commits or whole-file blob continuity. It does require exact block
continuity and each event's independently recomputed changed-hunk evidence.

All **46 fixtures** pass, including nonadjacent upstream events, missing or
reordered transitions, stale later evidence, final reviewer mismatches, and
later current-content changes. Independent `plan_consistency` review confirms
that the first receipts are retained, both transitions match actual Git data,
all 2,162 original units resolve, and source/tests/example scripts have no PR
diff against the actual `9acd0a5` master. Clause conversion remains held.

The strict local rebuild against that master passes all nine isolated scripts,
executable manual examples, doctests and API coverage. All six preservation
checks pass in blocking mode. Rendered validation finds 56 pages, 5,663 internal
links/assets and 137 repository source/edit links, with no broken links or
orphan pages. An independent ancestor-stack extraction finds 765 current content
IDs and retains all 713 original IDs. Both historical API headings and the new
`Indicators` heading have their two generated fragments exactly once; both
forwarding paragraphs resolve directly to `Indicators`, and each indicator
has one docstring target.

The real shallow-Git fixture fetches exactly four distinct provenance revisions:
the frozen baseline and the before/after boundaries of both upstream events.
Starting from the tested merge commit, it ends with five commit objects and
remains shallow. All provenance blobs are readable; inherited source changes
pass the scope guard and a PR-owned source change fails.

## Upstream sentence split

[PR #1023](https://github.com/mlakolar/ProximalBase.jl/pull/1023) subsequently
restored `NormL0` at `fc13ace163d219da0668b7f18be54ce6043edbf7`, with actual first
parent `9acd0a5d3a806f7254b02d9b92faf866e6b3c429`. GitHub confirms that exact
merged revision. It changes two more frozen units: the `value_entry` coverage
row and the shared infinite-level family-summary sentence. The latter becomes
three consecutive sentences, adding `NormL0` to the applicable families.

`norml1_pilot@fc13ace163d219da0668b7f18be54ce6043edbf7` independently verified
the merged PR, actual parent and file blobs, and all replacement hashes. Each
replacement is unique and byte-identical to the upstream text. The three
numerical sentences remain consecutive at their original shared-rule heading.
The original row now references all three exact hashes in order; no baseline
unit is regenerated or discarded. The earlier three two-event chains remain
unchanged. The ledger totals are 2,118 stay, 38 move, one reviewed example-label
repair, and five upstream dispositions, covering all 2,162 original units.

Schema 2 permits singular or plural source/replacement hash fields, exclusively,
so a bounded contiguous span can survive later upstream edits. Each transition
still verifies actual Git commit, blob and diff evidence. This is external-change
accounting, not permission for migration rewrites; the `reword`/`split` hardening
gate before clause conversion remains in force. The new cardinality endpoint
section stays in its upstream location, with an additive family-index link.
The NormL1 contract and all existing API forwarding headings are unchanged.

The strict local build against `fc13ace` passes all nine isolated scripts,
executable manual examples, doctests and API coverage (67 bindings). Markdown
lint passes for all 49 active sources; the pilot retains all fourteen headings
and assigns no clause identifiers. Rendered validation finds 56 pages, 5,676
internal links/assets and 138 repository source/edit links, with no broken links
or orphan pages. The independent ancestor-stack review finds 770 current IDs
and confirms all 713 original IDs, both historical API forwarding headings,
and exactly one target for each indicator and the new `NormL0` docstring.

`norml1_pilot@fc13ace163d219da0668b7f18be54ce6043edbf7` also checked every one
of the seven Evidence-table descriptions against its referenced test file and
found no mismatch. This was author verification: that agent authored the pilot
page and its evidence table. Those NormL1 source/test files are unchanged at this base.
The core-functions description's exact input/output identity denotes aliasing
of the same object; its approximate value comparison is not a claim of bitwise
numerical equality. No clause status was inferred from this review.

All **52 fixtures** and all six preservation checks pass in blocking mode at
this revision. The new span fixtures reject missing, reordered, extra, stale
or nonconsecutive pieces and support a later change to one member of a prior
span. Independent review reproduced an omission case where a replacement piece
duplicated text elsewhere in the old file. The correction uses positional
block and raw-line alignment; the regression rejects the incomplete receipt
and accepts the complete span. Every newly introduced copy remains accounted
for even when its normalized text already exists elsewhere.

The final real shallow-Git fixture fetches exactly five distinct provenance
commits at depth one. Including the starting merge commit, exactly six commit
objects are present; no unrelated history is fetched. All event blobs are
readable, and the scope comparison needs no further fetch. Inherited source
changes pass while a PR-owned source change fails. The final branch has no
source, test or example-script diff against `fc13ace`.

## Second-round decisions before the next family

The external Claude review of merge commit `b0c4c8e` against its master parent
`fc13ace` confirmed the five requested fixes, 2,162 accounted-for ledger rows,
52 passing fixtures and five preservation checks. It counted the 713 IDs in
the frozen snapshots, but did not rerun the rendered anchor check or Julia
build and did not inspect CI results. Those validation results remain the
implementation team's evidence, not independent findings of that review.

The follow-up adopts four decisions:

1. Require forwarding stubs for explicit IDs and authored incoming links or
   source citations. The two uncited automatic API headings are removed.
   The original heading's rename is recorded through its existing external
   feature transitions, and its two baseline fragments are explicitly retired
   to `Indicators`. The complete frozen snapshots remain unchanged.
2. Route new families directly to `docs/src/spec/<family>.md` with the fourteen
   headings, source/test evidence and links to shared owners. The writing guide
   now states this rule; the contributor workflow and `CONTRIBUTING.md` link to it.
   The three already-restored families leave 16 existing family/operation entries
   to migrate, instead of the initial 13.
3. Retain receipts for the four inventory blocks. API presence checks do not
   protect the README and migration scope claims or the descriptions and evidence
   inside coverage cells. Batch several reconciliations when practical; keep
   the existing retirement of migration checks after Phase 5 acceptance.
4. Attribute the supplied review to its external author and record the owner's
   acceptance separately. The audit README defines agent roles by the change
   reviewed. The pilot author's later evidence-table verification is labelled
   as author verification, while its external-feature receipts remain independent.

No numerical clause is converted in this follow-up. The `reword`/`split` token
and example accounting fix remains required before clause conversion.

For the heading decision, `plan_consistency@b0c4c8edc0ae1498b7eaf30faa71f542c97b4a45`
audited 162 tracked text files covering the active manual, developer/design
documents, README/CONTRIBUTING, examples, source/docstrings and tests, plus nine
directly referenced benchmark/audit resources at that revision. It found no
authored incoming link, reference definition, `@ref` or raw HTML link to either
heading or its generated `-1` fragment. Both headings lack an explicit `@id`.
Generated navigation and self-permalinks do not count as authored references;
external bookmarks are outside this audit's scope.

The same reviewer verified the original heading's exact `Nonnegative orthant`
to `Box and nonnegative orthant` to `Indicators` transitions in merged PRs
#1020 and #1022. The added heading receipt keeps the original block hash and
uses the existing upstream mechanism. The ledger now has 2,117 stay, 38 move,
one reviewed example-label repair, and six upstream dispositions with ten
verified events. No original unit or frozen snapshot has been regenerated.

The lead implementation agent's strict local build passes all nine isolated
scripts, executable manual examples, doctests and API coverage for 67 bindings.
All six preservation checks pass in blocking mode, with 58 passing checker
fixtures. HTML validation checks 56 pages, 5,675 internal links/assets and
138 repository source/edit links, with no broken links or orphan pages.

`plan_consistency` independently reviewed the follow-up code, documentation and
rendered output. Its separate extraction found 768 current content IDs and
exactly the two approved baseline omissions; the other 711 remain. Both later
box-heading fragments are also absent, `Indicators` and each API docstring
target remain singletons, and no rendered link targets a removed fragment.
The new-family guidance and contributor-workflow link render correctly. Both
frozen snapshot files are byte-identical to `b0c4c8e`, and there is no source,
test or example-script diff against that base. The independent review found
no blocking issue; it did not author these repository changes.

## Third-round preparation corrections

The external Claude review of merge commit `5df0776` against `b0c4c8e` confirmed
the four second-round decisions, 58 fixtures, five preservation checks and no
source, test or example-script changes. It did not rerun the Julia build or
rendered anchor check, inspect CI results, or load the preview pages. Those
implementation-validation results are not attributed to the external review.

This follow-up makes automatic-heading retirement records data-driven while
retaining their original-heading, frozen-fragment, upstream-mapping, reviewer,
audit-evidence and same-page-successor validations. It introduces no new
retirement and changes neither frozen snapshot.

A separate removed-lines report compares each change with its own base under
`docs/src`. It covers later additions without adding them to the frozen ledger,
preserves line multiplicity and literal whitespace, and leaves deliberate
rewrites or link repairs visible for review. CI retains the report even when
the ledger is unchanged. The audit now requires pending upstream changes to be
reconciled to an all-pass report before migration work begins.

The Phase 5 cleanup includes the supplemental report and the temporary
writing-guide sentence about retirement records and the frozen inventory.
The durable forwarding and new-family routing rules remain. Clause conversion
is still held pending the separate `reword`/`split` token and example fix.

### Reconciliation before the next migration

Actual master advanced through three external changes after `5df0776`. Their
PR metadata reported older base SHAs; the receipts use each landed commit's
actual first parent instead:

| Merged PR | Landed commit | Actual first parent | Reconciled original units |
|:--|:--|:--|:--|
| #1024, `IndBallLinf` | `87c8ee066e313ed2ecc540068d5ce3080717951d` | `5df07768395d2225e2bc1d45368cc08e91dfe1e9` | API indicator `@docs` fence and migration scope paragraph |
| #1026, least-squares test descriptions | `b9120c1374fb5ef379339499bf0e282518fa3606` | `87c8ee066e313ed2ecc540068d5ce3080717951d` | `test/TESTING.md` test-inventory paragraph |
| #1025, scaled indicator screens | `a4b682ffbb41d6ec66bdc8fa886c82f04d418d2c` | `b9120c1374fb5ef379339499bf0e282518fa3606` | The scaling guide's existing unsupported-forms list item |

`norml1_pilot@a4b682ffbb41d6ec66bdc8fa886c82f04d418d2c` independently
verified those externally authored changes and their exact Git transitions.
`plan_consistency` separately confirmed the merged PRs and actual parent chain.
Two existing receipts gained one event each; two original units gained new
receipts. The ledger retains all 2,162 original IDs, hashes and kinds:
2,115 stay, 38 move, one reviewed example-label repair, and eight upstream
dispositions covering fourteen events. No source, test or example file is
edited by this reconciliation.

The restored `IndBallLinf` adds a fourth family since the frozen baseline. Its
existing endpoint rule remains in the numerical hub pending its own migration;
the current existing-page backlog is seventeen family/operation entries. The
removed-lines report covers this later text as well as the three restores
identified in the external review.

Before adding that family-map entry or making another migration change, the
lead implementation agent built the reconciled tree with Julia 1.10.10 and
ran all six checks in blocking mode against `a4b682f`. All passed, including
711 original fragments present and the same two approved retirements. The
strict build ran the nine isolated examples, manual examples and doctests;
API coverage included 68 bindings. The reconciliation was committed separately
as `cf809016b30f749f05410d6eca8aa7bfe19f998e` before the third-round
preparation changes.

### Local validation before the concurrent IndBallL1 merge

The strict Julia 1.10.10 build on base `a4b682f` passes with the added family-map entry.
All six preservation checks pass in blocking mode, and all 73 checker fixtures
pass. The new report compares the final tree with actual base `a4b682f`: one
added family-map line, zero removed lines and zero unmatched lines. Its fixtures
reject deletion of post-baseline text, a lost duplicate occurrence and literal
whitespace changes; they accept exact moves to new untracked destinations.
They also distinguish staged and unstaged edits, deleted files, clean sparse
omissions, deliberate reflow/link repairs, invalid bases and CLI exit modes.

Local rendered validation checks 56 pages, 5,683 local links/assets and 139
repository source/edit links, with no broken links or orphan pages.
`plan_consistency` independently extracted 769 current content IDs, with all
711 retained original IDs and exactly the same two approved retirements.
The indicator targets remain singletons, the added `IndBallLinf` map link
resolves, and no rendered link targets a retired fragment.

The bounded heading helper supports every one of the 304 heading units in the
frozen ledger. Independent comparison of all 256 authored manual headings
against the retained baseline HTML confirms their exact rendered slugs; the
installed Documenter source confirms the slug filter and explicit-ID handling.
No frozen heading needs a special-case Python entry. Ambiguous ownership is
still rejected. The frozen snapshots and retirement metadata are unchanged.

`plan_consistency` independently reviewed the checker, report, fixtures and
integration, reran all 73 fixtures, and exercised additional real-Git loss,
move, ignored-output and sparse-checkout cases without a ledger. It found no
blocking issue and authored no repository implementation. Its reviewed blobs
were `f103002e730febcc22543c6216c2bc9089accf8a` (`migration_checks.py`),
`49089d04ee3afa6f69e74313a3ddd017d81451b7` (`check_migration_lines.py`), and
`101ee66d7430f93477751086927bb499373beb82` (`test_migration_checks.py`).

### Concurrent IndBallL1 integration

PR #1031 at `ee90029` passed every required lane, both gates and preview
publication in CI run `44471487157`. Its actual preservation and removed-lines
reports were also clear. Before merge, the actual master ref had advanced to
`b07983a88bf0d67a5b626721f58f06b057b6f91d`, the landed commit of PR #1027
(`IndBallL1`). Its actual first parent is `a4b682f`; the feature PR metadata
reported the older base `87c8ee0`.

The integration preserves both family-map additions: the migration's pending
`IndBallLinf` entry and the feature's authored `IndBallL1` contract. This was
the only textual merge conflict. `IndBallL1` follows the new-family routing
rule with its own fourteen-section `spec/indballl1.md` page, so the existing-page
backlog remains seventeen. The feature's source, tests, numerical text,
examples, API references and navigation are inherited unchanged.

The original indicator `@docs` block has one further upstream event for the
new API entry, independently verified by
`norml1_pilot@b07983a88bf0d67a5b626721f58f06b057b6f91d`. The other frozen
units retain their existing dispositions. All 2,162 roots remain accounted for;
the eight upstream receipts now contain fifteen events.
Final acceptance must use the integrated head, actual base `b07983a`, and its
new CI reports and preview; the earlier green run does not establish that
the combined tree passed. These final results are recorded on PR #1031.
