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
and `anchors-ci.tsv` record each baseline rendered page and its content anchors
for the two explicitly selected build runtimes described below. Each snapshot
contains 54 rendered pages, including generated examples and contracts, and
621 content anchors. Sidebar control IDs are excluded because navigation
regrouping intentionally changes those controls. `source-citations.tsv` records
the baseline source/test references to `numerics.md`; the check also reads
current citations. `baseline.json` records the baseline and immutable example
script hashes. These nine scripts are preserved exactly.

The baseline HTML itself is a build artifact, not repository content. In CI,
the baseline commit must be available locally. The regular documentation build
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

CI selects its original baseline runtime explicitly:

```sh
python3 docs/check_migration.py --anchor-profile ci --check all --build docs/build
```

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

## Exact anchors and build runtimes

Documenter 1.20.0 constructs an admonition fragment from its title and
`string(hash(strip(mdflatten(node))), base=16)`. That uses Julia's runtime
string hash rather than a portable content digest. The unchanged two
admonitions therefore have different fragments under the local Julia 1.10.10
build and the CI Julia 1.13.4 build, although their content is identical. The
flattened input contains the admonition text, title, and category, not its file
path or link destinations.

The `local` profile is the original `anchors.tsv`; it remains the default and
is unchanged. The `ci` profile is `anchors-ci.tsv`, captured from the retained
Phase 0 CI artifact **14492512149** at commit
`db20ba7564992abadf0c1293a6ff4d10acd07ad5`, before any authored documentation
change. Its ZIP SHA-256 is
`35c8aa068d12a170ad9f43007b62fa701fcbea499dbefca60abf9cef45183af1`.
Both builds used Documenter 1.20.0. Runtime, source-commit, artifact, and
snapshot SHA-256 provenance is recorded in `baseline.json`.

The CI job log records the actual merge-ref checkout as
`f08c07bbfb6d36dd9096cd94754e12a30300cef6`, merging that checkpoint into
`2a55362d374fa4299d29db7684659fce8cc8e149`. The latter upstream commit changes
only least-squares implementation and tests, so the authored documentation
still matches the frozen provenance baseline. Both parents and the actual
build revision are recorded separately in the profile metadata.

| Page | Local fragment | CI fragment |
|:--|:--|:--|
| `function_library` | `Which-squared-loss-object-should-I-use?-4d7e89b086da14ac` | `Which-squared-loss-object-should-I-use?-7961aee47a1e55` |
| `getting_started` | `Degenerate-data-and-step-size-estimates-431e86eee387de1e` | `Degenerate-data-and-step-size-estimates-a1d8f97b86f5e430` |

These are the only differences between the two baseline snapshots. Each is
validated as a complete, nonempty snapshot of 54 original page records and
621 exact content anchors, with no duplicate rows and a verified snapshot hash.
Every anchor remains bound to its original page. Selection is explicit via
`--anchor-profile local` or `--anchor-profile ci`; unknown or unavailable
profiles are configuration errors. The checker never removes hash suffixes,
ignores the two IDs, accepts a reduced subset, or falls back to another profile.
Use a baseline and final build from the same declared runtime. A future runtime
change requires an explicit provenance and compatibility decision.

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

`reviewer=mechanical` means only that the checker can verify unchanged text; it
is not a human or agent review attestation. The four reviewed actions require
an independent identity and revision in the form `reviewer@revision`. Record a
real review, never a placeholder. Reviewers compare the old rendered section
with the rendered destination and check the current source/tests where the
wording describes numerical behavior. The review revision and replacement
hashes make the attestation refer to specific content.

All six checks consume the same action policy. Exact moves preserve the
protected code/math/numeric/link tokens, sentence fingerprints, and executed
example/assertion markers. Reviewed transformations use their specific
replacement hashes and independent review as the acceptance evidence, including
when an approved deduplication reduces a total count. Aggregate counts are
diagnostics, not a mathematical or semantic proof that the meaning survived.
The checker cannot authenticate a review identity or decide whether two
statements mean the same thing.

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
