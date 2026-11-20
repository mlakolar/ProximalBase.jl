# Documentation reader-track migration

Tracking issue: [#1019](https://github.com/mlakolar/ProximalBase.jl/issues/1019).
The frozen documentation provenance baseline is
`7ae86e2093060eb124211e2f21f043c6379f97ae`.
The original planning snapshot was `6c487c4`; the inventory below was measured
from the later implementation baseline, not copied from the planning estimates.

## Scope and files

`inventory.tsv` records path, UTF-8 byte size, SHA-256, whitespace-delimited word
count, and scope. The block ledger covers all 42 authored pages under
`docs/src/`, the three design contracts rendered into the manual, `README.md`,
`CONTRIBUTING.md`, `examples/README.md`, `test/TESTING.md`,
`benchmark/README.md`, and `docs/repository_hygiene.md`: 51 documents in total.
The other top-level Markdown design records have file inventory rows only;
historical audit data and generated copies are excluded from the prose ledger.

`ledger.tsv` has one row per baseline block. Baseline text is read with
`git show`, so this folder does not duplicate the old manual. `anchors.tsv`
records each baseline rendered page and its content anchors. The old
`anchors-ci.tsv` is retained as immutable evidence from the other build runtime;
acceptance now uses one snapshot and the three bounded identities below. Each snapshot
contains 54 rendered pages, including generated examples and contracts, and
713 content anchors. Sidebar control IDs are excluded because navigation
regrouping intentionally changes those controls. `source-citations.tsv` records
the baseline source/test references to `numerics.md`; the check also reads
current citations. `baseline.json` records the baseline and immutable example
script hashes. These nine scripts are preserved exactly.

The baseline HTML itself is a build artifact, not repository content. In CI,
the exact baseline and recorded upstream-provenance commits must be available
locally; the workflow fetches those commits at depth one. The regular documentation build
still runs the nine isolated scripts, doctests, and all executable manual
examples; provenance checks do not replace their execution.

The initial parser correction for fragment-only links changed the normalized
hashes of two unchanged `smooth.md` blocks. Only those ledger hashes were
refreshed from the same frozen source; no baseline text, block ID, or action
changed. A `#fragment` link now resolves to its original page, so an unrepaired
move to another page is detected.

Normalizer version 2 preserves literal whitespace in inline code, inline math,
and fenced content nested in admonitions before normalizing surrounding prose
or rebasing actual Markdown links. Section selection also ignores heading-like
comments inside fences. The independent review supplied regressions for both
defects. Recomputing every block from the same frozen source confirmed that
this correction changes no hash in the present baseline or its exact moved
destinations; all 2,162 block IDs and their actions are unchanged.

## Running the six checks

```sh
python3 -m unittest discover -s docs -p 'test_migration_checks.py'
python3 docs/check_migration.py --check all --build docs/build
```

Local and CI builds use the same command and the same acceptance rules.

The default is report-only: every check prints `PASS` or `FAIL` and its findings,
but preservation findings do not change the command's exit status. Missing or
unreadable configuration is an execution error, not a successful report. Use
`--blocking` for an explicit acceptance run; the planned final lock will make
that invocation mandatory in CI. A single check can be selected with
`--check blocks`, `tokens`, `sentences`, `anchors`, `citations`, or `examples`.

For the docs-only scope check, supply the PR's merge base explicitly:

```sh
python3 docs/check_migration.py --blocking --scope-base "$PR_MERGE_BASE"
```

This rejects changes in `src/` and `test/` relative to that PR base. The frozen
documentation baseline is deliberately not the source-code comparison base:
unrelated upstream implementation changes may continue during this migration.
The source/test trees and source docstrings are therefore protected by PR scope,
while their links to old manual locations are protected by the anchor and
citation checks.

## Exact anchors and runtime-generated identities

The complete frozen `anchors.tsv` still contains all 54 original pages and
713 content IDs. Its hash is verified through `baseline.json`. Every page remains
required. Of its IDs, 708 active fragments are compared exactly, the two already
approved automatic-heading retirements require their checked successor, and
only these three uncited runtime-generated admonitions use a content identity:

| Page | Category | Exact title |
|:--|:--|:--|
| `function_library` | `note` | Which squared-loss object should I use? |
| `getting_started` | `details` | Degenerate data and step-size estimates |
| `api/iterates` | `note` | Non-finite vectors |

`anchor-identities.json` records both historical fragments and the original
source for each identity. The checker requires the exact category/title once in
that source and once on the original rendered page. A missing or duplicated
admonition, a changed category/title, an unrelated missing fragment, or an
authored link to a current or historical dynamic fragment fails. The sole
permitted link is the generated self-permalink inside that same admonition.
This is not suffix stripping or a general exemption for generated IDs.

Documenter 1.20.0 derives those fragments from Julia's runtime string hash.
The original local Julia 1.10.10 and CI Julia 1.13.4 builds therefore produced
different hashes for unchanged text. Their complete snapshots and the original
artifact/commit/SHA-256 provenance remain in this folder and `baseline.json` as
historical evidence. They no longer select runtime-specific acceptance profiles.

The anchor parser retains the original nested-article correction: IDs after
API docstring articles count too. The original incomplete 621-ID inventory was
corrected to 713 by re-extracting the same frozen HTML; all 92 recovered IDs
remain protected. Acceptance reports exact fragments, content identities and
approved retirements separately, together accounting for all 713 targets.

## Block identity and destinations

A block is a heading, paragraph, table row, list item, fenced code or math block,
or complete admonition. Prose in `numerics.md`, new `spec/` pages, and the
identified endpoint sections of the family guides is divided into deterministic
sentence units. Code, math, and admonitions are not split into prose sentences.
The parser is intentionally bounded; it is not a full Markdown implementation.
Fixtures cover its actual assumptions, and Documenter remains the renderer.

`block_id` contains the original path, heading path, and ordinal. A hash is over
normalized content: surrounding prose whitespace is collapsed and actual
relative Markdown links are resolved against the source document. Inline code,
inline math, and all fences retain their literal contents; a string containing
Markdown-looking text is not treated as a link. This permits a correct relative-link repair during
a move without disguising it as a numerical rewrite. Headings are recorded as
their own kind so that navigation labels are not silently lost.

`dest` is a repository-relative path, optionally followed by a heading anchor.
Page-level destinations are sufficient for unchanged blocks; moved blocks
should name their owning section. `dest_sha` identifies the exact current
replacement block. Semicolon-separated destinations and matching hashes support
a reviewed split. The checker validates the replacement hashes and preserves
occurrence counts: two identical original statements cannot reuse one remaining
copy unless the ledger explicitly records a reviewed deduplication.

## Action acceptance rules

| Action | Required evidence |
|:--|:--|
| `stay` | Original normalized hash at the original path, with enough distinct occurrences |
| `move` | Original normalized hash at the named destination, with enough distinct occurrences |
| `split` | Independently reviewed replacement hashes; the pieces retain every original word token |
| `reword` | Independently reviewed replacement hashes and equivalent meaning confirmed in rendered review |
| `dedupe` | Independently reviewed surviving replacement hash and equivalent meaning confirmed in rendered review |
| `supersede` | Independently reviewed replacement hashes for routing text deliberately replaced by navigation |
| `archive` | Original normalized hash under `docs/design/` and a forwarding link from the original page |
| `upstream` | Independently reviewed merged external PR receipt; exact before/after commits, file blobs and changed blocks verified against Git; current replacement matches that external change |

`reviewer=mechanical` means only that the checker can verify unchanged text; it
is not a human or agent review attestation. Reviewed actions require
an independent identity and revision in the form `reviewer@revision`. Record a
real review, never a placeholder. Reviewers compare the old rendered section
with the rendered destination and check the current source/tests where the
wording describes numerical behavior. The review revision and replacement
hashes make the attestation refer to specific content.

All six checks consume the same action policy. Exact moves preserve the
protected code/math/numeric/link tokens, sentence fingerprints, and executed
example/assertion markers. Reviewed transformations use their specific
replacement hashes and independent review as semantic acceptance evidence.
`reword` and `split` still preserve protected tokens and executed-example markers;
only exact reviewed allowances can account for intentional changes. A reviewed
`dedupe` or `supersede` may deliberately reduce ordinary migration counts. Aggregate counts are
diagnostics, not a mathematical or semantic proof that the meaning survived.
The checker cannot authenticate a review identity or decide whether two
statements mean the same thing.

### Review identities and verification scope

These labels identify agents in the implementation session, not human names or
authenticated GitHub identities. Independence applies to the particular change
being attested. A reviewer label does not by itself establish owner approval.

| Label or role | Work and independence |
|:--|:--|
| `plan_consistency` | Independently reviewed the pilot migration and checker changes without editing or committing those repository files. Compared moved content, the example-label repair, frozen artifacts and rendered anchors; reran checks and supplied defect reproductions and design suggestions. |
| `norml1_pilot` | Authored the NormL1 destination page, migration and evidence table, and later the standalone removed-lines report. Its signed upstream receipts independently review external changes in PRs #1020, #1022, #1023, #1024, #1026, #1025 and #1027, which it did not author. Its later seven-row evidence-table check was author verification. |
| `migration_checks` | Authored checker changes and regression fixtures, coordinated the removed-lines report, and ran the real shallow-Git workflow fixtures. `plan_consistency` independently reviewed that implementation. |
| Lead implementation agent | Ran strict local builds and inspected CI results, artifacts and preview publication. These are implementation-validation records. |
| External Claude reviewer | Prepared the reviews supplied to and accepted by the owner, with the verification limits stated in each review. The first three reviews did not independently verify the Julia build, rendered anchor check or CI results; the third could not load the preview pages either. |
| Owner | Accepted the external review and made scope and progression decisions. Acceptance does not imply personally performing the review or rerunning its checks. |

## Documentation changed by unrelated pull requests

Ordinary feature pull requests may update the documentation they affect without
editing this migration ledger. After such a pull request merges, reconcile its
changed baseline blocks in a separate documentation-only change. Keep every
original `block_id` and `sha`; set its action to `upstream` and record the exact
current destination and exact replacement hash or ordered hashes.

Reconcile first: before starting a migration change, reconcile pending upstream
changes and run all six checks in blocking mode so the starting report is all-pass.

`upstream.json` uses schema 2. Each record binds the original block ID/hash to
its final destination/hash and reviewer, followed by an ordered `events` list.
Every event records the external PR number and URL, reviewed merged status,
actual before/after commits, file blob IDs, source/replacement block hashes,
and its own reviewer. A single block uses `source_sha` or `replacement_sha`;
an ordered span uses `source_shas` or `replacement_shas`. Each side has exactly
one of those forms. Use the actual first parent of the landed commit, which
can differ from the PR API's recorded base. The checker recomputes each event's
Git provenance and requires its source and replacement to share a changed diff
hunk. The first source is the frozen original; each later source is the complete
preceding replacement span. The final ordered hashes and reviewer match the
ledger. Multiple destinations repeat one exact same-file scope, paired in order
with their hashes. Current pieces must remain consecutive in that scope.

The checker derives the complete replacement span from the source-owned changed
line regions. An unchanged neighboring block is excluded only when positional
block matching and unchanged raw-line alignment agree. Matching text elsewhere
in the old file does not exempt a newly introduced copy from the receipt.

Events need not be adjacent commits or have identical whole-file blobs between
them: unrelated upstream work may edit other sections. Exact block-hash
continuity is mandatory. The action supports exact contiguous replacement spans
in the same file, without a general move, rename or deletion exemption. Each PR's
merged status is independently verified and recorded in `review.md`; an offline
checker cannot authenticate it. CI fetches the exact before/after revisions for
every recorded event at depth one, deduplicating revisions shared by events.

All accounting checks accept the upstream disposition only after that evidence
passes. A later unrecorded edit to the replacement still fails. All original
block identities and hashes remain frozen, including a heading renamed by an
external feature change.

Retain a forwarding stub for an explicit `@id` or a heading with an authored
incoming link or source citation. Audit manual sources, docstrings, repository
entry documents and examples before retiring an automatic heading. Generated
self-permalinks and navigation are not authored incoming references. This
repository audit cannot establish whether an external bookmark exists.

Record each retirement in `baseline.json`: its original block ID/hash, page,
automatic fragments, same-page successor, reviewer and authored-reference audit.
The checker validates those data against the frozen heading and fragment inventory,
the checked upstream mapping and the rendered successor. A later valid retirement
uses the same record format without changing Python code. Explicit IDs, unrelated
fragments and missing pages remain protected.

The two uncited API stubs are removed under this narrower rule. The original
`Nonnegative orthant` heading has an upstream receipt for its two renames to
`Indicators`. Its two original automatic fragments have named retirement
records in `baseline.json`, pointing to that successor on the same page. The
later `Box and nonnegative orthant` heading was never in the frozen baseline.
Both complete 713-ID snapshots remain unchanged. Acceptance accounts for
708 exact fragments, three checked content identities and two approved
retirements; it does not report all 713 original fragments as still present.
Every other baseline ID and every original page must remain. Ordinary strict rendering and HTML link checks still reject broken
authored links.

### Inventory reconciliation decision

Keep upstream receipts for the four current inventory blocks: the README
function list, migration scope paragraph, grouped API `@docs` fence, and
`value_entry` coverage row. `check_api.jl` checks that required names appear in
the API reference and coverage text. It does not validate the README or migration
scope claims, supported-argument descriptions, or evidence references inside
coverage cells. An exemption for the whole blocks would also exempt those
claims. This temporary migration therefore accepts the reconciliation cost;
batch several reviewed external changes in one documentation update when
practical. Numerical text always retains its exact provenance. The migration
checks still retire after Phase 5 acceptance.

## Reviewed rewrites and exact allowances

Token and example accounting remains active for every `reword` and `split`.
A review string is not an exemption: changing a numeric bound, dropping an
assertion or changing literal whitespace fails even when the replacement hash
and reviewer field are updated. Sentence fingerprints may change during a
reviewed rewrite, but the protected code/math/numeric/link tokens and complete
example/assertion lines must still be accounted for.

`token-allowances.json` uses schema 1 and an `allowances` list. Each record binds
the original block ID/hash, the comparison-source hashes, action, ordered
destinations and hashes, and the row's independent reviewer. Its `changes`
object lists the exact removed and added multisets under `tokens` and/or
`examples`; `evidence` identifies the bounded independent review. The checker
recomputes those multisets. Missing, stale, extra, duplicate or unused allowances
fail. The six inherited sandbox-label-only repairs have explicit receipts;
combining one with an altered assertion does not pass that receipt.

Reviewers still compare meaning and the rendered result. These mechanical
checks cannot authenticate a reviewer identity or decide equivalence. The owner
controls acceptance of each family review; closing the accounting gap does not
authorize changing numerical policy.

## Sentence refinement and later upstream moves

`refinements.json` is an immutable map from 64 original prose roots in
`smooth.md`, `group_penalty.md` and `calculus.md` to 313 ordered sentence children.
Every child has an ID formed from its original root plus `::sentence:N` and an
exact normalized hash. Rejoining those children must reproduce the original
normalized parent. No unrelated ID or original hash changes. This produces
2,411 effective inventory units while retaining all 2,162 provenance roots in
the root ledger.

A root's reviewed `split` row lists every child destination/hash in source order,
including children retained in their original guide. The original root still
binds the complete mapping; there is no second independently editable copy of
its provenance. Destination matching recognizes exact child sentences in the
three guides. Occurrence checks reject reuse across overlapping whole-paragraph
and child destinations, as well as reuse by different `reword` or `split` rows.

The read-only helper `docs/refine_migration_ledger.py` prepares replacement root
rows from a JSON list of `block_id` and ordered `destinations` arrays. It requires
an actual independent `reviewer@revision`, prints its proposed rows, and never
overwrites the ledger. Review the output, apply it, then run every preservation
check. For programmatic use, `sentence_children`, `refinement_inventory` and
`validated_refinements` expose the same frozen units.

Six refined roots already have reviewed upstream changes. Their original
receipts must survive the subsequent move. `migration-bases.json` has schema 1
and a `bases` list; each record binds `block_id`, `baseline_sha`, the canonical
`upstream_sha` digest of the entire retained receipt, its exact terminal
`source_shas`, and review `evidence`. The root ledger can then use an ordinary
`move`, `split` or `reword` destination. The checker verifies every historical
Git transition first, then applies ordinary token/example accounting to the
final reviewed upstream text. It does not compare to obsolete pre-feature words
or exempt the later migration. Stale and unused basis records fail.

## Updating the ledger

Initialization is one-time and refuses to overwrite the ledger:

```sh
python3 docs/check_migration.py --init --baseline-html /path/to/unchanged/build
```

For mechanical moves, prepare a JSON list with `source`, `text`, `destination`,
and optional `anchor` fields. `text` must be a unique verbatim span in the frozen
source file. The helper updates only baseline blocks fully contained in those
spans, validates the complete resulting ledger, and writes only on success:

```sh
python3 docs/check_migration.py --apply-moves /path/to/moves.json --blocking
```

Do not regenerate the baseline to make findings disappear. New authored content
does not need a baseline ledger row; existing content must always retain its
original provenance. When a parser limitation affects a real move, add a
focused fixture and make the limitation visible in review. Do not broaden the
parser speculatively or replace a missing block with an aggregate-count check.

The supplemental removed-lines report covers text added after the frozen baseline:

```sh
python3 docs/check_migration_lines.py "$PR_MERGE_BASE"
```

It compares the change's actual base with the current tree under `docs/src`,
including new destination files. Every removed nonblank line must reappear among
the added lines, with identical literal whitespace and enough occurrences. It
does not credit unchanged copies elsewhere or use ledger exemptions. Reflow,
link repairs and deliberate rewrites remain visible for review. The default is
report-only; `--blocking` is available for a verbatim move. A clean line report
supplements the six checks and independent review, not a claim of semantic
equivalence. CI retains this report beside the preservation log for every change,
including changes that do not touch the ledger.

After the final Phase 5 blocking acceptance, retire the six temporary migration
checks, the removed-lines report and the migration-only source/test scope guard
from routine CI. Remove the temporary retirement-record/frozen-inventory sentence
from `developer/writing_documentation.md`, retaining its forwarding and new-family
routing rules. Retain this frozen audit and its final acceptance evidence.
Ordinary strict builds, links, API coverage and executable-example checks
continue to protect the manual.

The `NormL1Nonneg` and `ElasticNet` mechanical moves temporarily keep the old
automatic subsection headings as forwarding entries so the frozen page/fragment
pairs remain checkable, including the duplicated weights heading. This does not
make an uncited automatic heading a permanent forwarding requirement. Review
those temporary entries during the final compatibility cleanup; retain explicit
IDs and authored link/citation targets under the writing guide's durable rule.

Two whole `NormL1Nonneg` KKT paragraphs parse into three and four units at their
new `spec/` destination. Their reviewed `split` rows record that parser
granularity only: each original paragraph remains a contiguous, ordered span,
with only its relative Markdown link rebased. The exact source/replacement
hashes and independent public-revision review are in `norml1nonneg-review.json`.
The removed-lines report and independent raw-text comparison establish
preservation for those two historical mappings. They now also pass protected-token
and example accounting without an allowance. The split word-count check remains
an additional check; it is not used as a substitute for literal preservation or
independent review.
