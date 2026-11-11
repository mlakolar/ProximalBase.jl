# #844 implementation and validation

## PB-only performance follow-up (2026-11-10; decision pending)

The [block-update investigation](block-update-performance.md) records the
baseline, exact source revisions, call paths, rejected prototypes, paired
measurements and allocation evidence for PB #787. The bounded candidate
`2fb4d9aa` passes 30,513 targeted assertions on both Julia 1.10.13 and 1.13.3.
Its ten-pair Julia 1.10 focus run improves the target combined update
69→48 ns and the independent size-5 public mathematical call 32→29 ns, but
confirms inactive-zero copy regressions of about 127→219 ns. This candidate
**fails the performance gate**. Optimization attempts stop for a maintainer
decision; no measured regression is approved. Raw evidence and narrow
alternatives are linked from the investigation.

All performance work is PB-only. It changes neither ProximalSolvers nor its
pins, tests, documentation, CI or PR #237. The historical whole-solver figures
remain historical evidence and the downstream gate stays pending under
PS #238 / PB #527. The following naming validation predates this performance
candidate and must not be reported as its final full-suite validation.

## Completed PB-only naming follow-up (2026-11-10)

The [maintainer ruling](https://github.com/mlakolar/ProximalBase.jl/pull/848#issuecomment-6779277418)
requires `BlockIterate`, `blockview` and `blockindices` in PB 0.6, with no
compatibility aliases. Counts, active IDs, layout semantics, storage operations,
group penalty names and real LMO atoms are retained. This change does not edit
ProximalSolvers, PR #237, its pin, tests, docs or CI. PS #238 owns the later rename.

The [re-review](https://github.com/mlakolar/ProximalBase.jl/pull/848#pullrequestreview-5849239904)
correctly identified an ownership gap: public `activevalues` views can alias
iterate storage through a plain vector parent. The partial `_block_owner`
guard is removed. Output independent of iterate storage and model caches is
now an explicit caller precondition, not a universal type-based guarantee.
Public storage views can be read into independent numerical outputs; their
owners remain unchanged. Existing shape/type/group/overlap and parameter checks
remain. Mathematical kernels have no iterate representation dispatch.

Current checks include an exact 190-export comparison (only the three approved
replacements), comparison of normalized executable text across 54 source and
extension files (only the rename and guard removal), 5,616 byte-identical
numerical differential results, 7,344 passing benchmark-harness assertions,
and a successful documentation build with executable examples on Julia 1.10.10.
The full Julia 1.13.3 run passed all **220 files: 1,960,015 assertions passed,
368 marked broken**, no failures or errors. The targeted Julia 1.10.10 storage,
block-oracle and failure suites passed **6,933 assertions**. The stale generated
coverage cell is corrected to `wrappers / prox_block`. The final source
`19ae85bcdebc33dd1246ddd6bb1597ad3cb323c3` only renames the private group-view
helper since the full-suite-tested source. Its storage/conformance run passed
7,369 assertions (220 marked broken) on Julia 1.13.3, and its storage suites
passed another 6,059 assertions on Julia 1.10.10. Fresh conformance reports at
that final source pass snapshot checks on both runtimes. See [the public inventory](PB-storage-public-api.txt),
[validation summary](PB-storage-validation.json) and
[PB-only performance data](PB-storage-performance.json).

All 108 measured PB-only math/writer cases allocate zero bytes and preserve
identical numerical/activation metadata. Paired measurements include a separate
A/A control. Five isolated writer cases above 5% in batched measurements were
within 2.4% with single evaluations; combined math+write cases were within
4.7% in this run. The raw data retain every case, including the larger component
deltas; these are exploratory timings, not confidence bounds. The previous
+21 ns / 1.056× whole-solve performance gate **remains pending**, as instructed.
No PS solve is rerun or reinterpreted in this PB-only scope.

The pre-rename PB/PS benchmark program is preserved verbatim in
`measurements/block_prox_pre_rename.jl`. Existing historical audit/baseline
artifacts and exact dependency pins are retained. The records below describe
their stated pre-rename commits; their earlier naming and ownership decisions
are historical and superseded by this follow-up.

## Pre-rename implementation and validation history

The implementation is published for review in
[draft PR #848](https://github.com/mlakolar/ProximalBase.jl/pull/848) and remains
unmerged. The mathematical API is committed at
PB `8fb174454a5b0c431376decf5a3c87830353a562` (tree
`58fdb820ad84d9c30d317ccab58190dcd5ba0e04`). The numerical PS companion was validated at
`75894887d7272df71c2c263feb4ea0b6d211fd48`, based on
`f9b7a26663b499b6f881bcc26c56d1acced06749`.

[ProximalSolvers.patch](ProximalSolvers.patch) is a `git format-patch` artifact;
apply it to that PS baseline with `git am`. The tested checkout is also in
`/tmp/pb844-solvers`. Its root project, Julia 1.10 manifest and docs project
all pin the PB implementation commit above. The PB commit must be available
from the chosen Git remote before those remote pins can be instantiated;
local integration uses `Pkg.develop` with the two exact checked-out sources.
The numerical and performance evidence below was recorded before publication.

## PR branch integration

The review branch incorporates master `8166613c039af17c7dcc4d7e8a61f4f7f3bb70b4`
and preserves its function-first documentation structure. Moved guides use the
pure block API, and the block-prox contract is rendered alongside the iterate
contracts. Production files in `src/`, `ext/` and `Project.toml` are identical
to the validated implementation commit, which remains reachable in the PR
history and continues to be the companion's dependency pin.

After integration, Julia 1.10.10 passed **2,524 assertions** across
`test/testing.jl`, `test/ci_plan.jl` and `test/tier_selection.jl`, with native
package images disabled. This includes migrating the new `WrongAtomKKT`
fixture from master to the `prox_block` author hook.
The reorganized documentation also built successfully on Julia 1.10.10,
including executable examples and generated contracts, with deployment disabled.

## Review follow-up (#848)

The [review at `46d318df`](https://github.com/mlakolar/ProximalBase.jl/pull/848#pullrequestreview-5847928791)
identified contradictory live requirements in the four design/acceptance
documents. They now specify pure block math followed by a separate public
storage write: FA-1–FA-7, FL-4, symmetric IDX-3 and ORC-5–ORC-7, the migration
recipe, FT15/FT16/FT23/FT28, T04/T05 and performance protocols. Rule IDs are
preserved. The old FD-12 sign-off is linked as an archived decision; retired
call spellings survive only in explicitly historical audit/evidence sections.

A vocabulary table distinguishes entry, iterate coordinate, penalty block,
group, active slot and LMO atom. The storage-name change remains a separate
maintainer decision; this migration retains `AtomIterate` and its accessors.
The developer guide explains the intentional `_block_owner` preflight safety
dependency and the need to update its guard/tests for new representations.

The companion is now published as
[ProximalSolvers #237](https://github.com/mlakolar/ProximalSolvers.jl/pull/237),
still pinned to PB `8fb17445`. Its required CI must pass on the exact final
pair before merge. The previously matched **269 failures and 83 errors** are
regression evidence, not green integration; PS #230 and PB #527 remain open.
Its initial [Linux CI run](https://github.com/mlakolar/ProximalSolvers.jl/actions/runs/42885546237/job/130073189488)
reported **120,044 passes, 269 failed assertions and 83 errors**. Equal failure
counts alone do not assert that every failing case was independently compared
again in CI.

The published PS head `8e98442bf6a16bf765182c3fdfcdc8472cc3511d` adds a docs-only
follow-up: replace the example's removed `nnz` alias with `nactive`, render the
pinned block docstrings locally, and resolve their existing PB references
without requiring publication of the new API first. The documentation built
locally on Julia 1.10.10 with deployment disabled. PS source, tests and dependency
pins are unchanged from the numerical validation commit. The refreshed
two-commit patch applied with `git am` to baseline `f9b7a266`; its resulting
tree exactly matches the published companion tree
`5bf1dd810c142a6c0e0910e7fd12edb4730f67d0`.

**Performance gate: pending.** On 2026-11-10 the maintainer explicitly chose
to keep this gate pending. The +21 ns small-block case and up to 1.056×
whole-solve cost have not been approved. No new timing claim or bulk-writer
redesign is made in this documentation follow-up; #787 remains separate.

Local review validation on Julia 1.10.10: **3,325 assertions passed** across
`block_step`, `block_step_failures` and `symmetric_oracles`; the documentation
build, including executable examples and all rendered contracts, passed with
deployment disabled. A separate content audit found no retired oracle in live
requirements/examples/acceptance cases and verified that normative rule IDs
are preserved. Production source and extensions are unchanged by this follow-up.

## Contract and inventory

- [Pure block contract](../../design/block-prox-0.6.md): signatures, penalty
  versus iterate index spaces, representation, aliasing and failure behavior.
- [Pinned call-site inventory](../block-prox-844-call-sites.tsv): 1,065 matches
  across PB and PS source, tests, extensions, examples, docs and benchmarks.
- [Public API audit](../block-prox-844-public-api.txt): removed bindings
  `prox_entry`, `prox_coordinate!`, `cdprox!`; added exports `prox_block`,
  `prox_block!`; every other export unchanged.

The independent storage-type rename is excluded. Full-array oracles and the
KKT, curvature, diagonal-metric and zero-curvature semantics are unchanged.
PS retains the existing public group-layout check, KKT skip, lazy curvature,
zero-curvature branches and model/cache updates. The three specialized group
shrink factors now use PB's numerical kernels. The added PS regression checks
that trial `[1e155,0]`, level `1e155` and curvature `2` yield `[5e154,0]`.

## Local checks

The retired-versus-new differential script prints 5,616 scalar/group results,
including wrappers, Float32/Float64, zero/Inf levels, weights and steps,
NaN/Inf inputs and extreme finite magnitudes. The baseline and candidate
outputs matched byte for byte. The group oracle also has direct comparisons
with full `prox!`, independent extreme references, exact in-place tests,
public custom-function extensions, buffer failures, symmetric entry mapping,
and explicit public-writer/active-set tests.

Julia 1.10.13: the affected PB suites passed after retrying three files whose
initial compilation ran out of local disk space. The affected PS suites
(`coordinate_descent_basics`, `generic_newton`, `group_curvature`, `group_lasso`,
`group_momentum`, `group_penalties`, `multiresponse`, `nonconvex_penalties`)
passed 90,094 assertions. No mathematical assertion failed in the LTS run.
The documentation build, including executable examples, passed. The benchmark
harness checks, catalog selection fingerprints and references passed.

Julia 1.13.3: all **220 PB test files passed**, with **1,959,875 passed
assertions and 368 marked broken**. Files that emitted cache-write warnings
during disk pressure were rerun with native package images disabled; all
**88,589 assertions passed** without those warnings. The full file inventory
was checked for omissions and duplicates. The final docs build on Julia
1.10.13 also passed with deployment disabled.

The full PS baseline on Julia 1.13.3 had **120,045 passes, 269 failed
assertions and 83 errors**. The companion had **120,047 passes, the same 269
failed assertions and the same 83 errors**. All **352 failing test cases**
match by testset, location and assertion expression after normalizing checkout
paths and the approved scalar rename. The two additional passes cover the
independent extreme group-shrink reference. See
[the comparison record](downstream-baseline.json) and
[the complete local test summary](test-results.json).

[Performance results](performance.md) include raw Float32/Float64 scalar,
group, public-writer and whole-solver measurements on Julia 1.10.13 and
1.13.3. Warmed scalar/group/public-writer hot paths allocate zero bytes; solve
allocations and pass counts match. The largest observed whole-solve median
ratio is **1.056×**. One small group/write case has median timings **48 → 69 ns** and a
**1.417× median paired ratio**;
that measured tradeoff needs a maintainer decision before merging. No private
storage writer or legacy proximal shim was added to avoid that cost.

The exploratory Julia 1.10.10 full run exposed sparse-dot shape failures also
reproduced on unchanged PB `8eaf8f812fd0720e180f35cfc75bc7b090e8ea97`.
The latest LTS patch passed that coverage. Results from the older runtime and
from local disk/worker interruptions are not reported as successful runs.

## Reproduction

Use the PB implementation commit and companion PS patch above, with Julia
1.10.13 or 1.13.3. Test environments were isolated under `/tmp`, with the PB
`Project.toml` extras and compatibility bounds and explicit `Pkg.develop` paths.
`JULIA_DEPOT_PATH=/tmp/issue843-depot`, `JULIA_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` were set; runtime downloads were checked against the
official Julia binary SHA-256 catalog.

```sh
# Full PB run in the resolved test environment; bounds checks enabled.
julia --startup-file=no --check-bounds=yes --project=/tmp/pb844-test-113-env \
    test/runtests.jl --tier=full --jobs=6

# Full PS integration run in the resolved combined environment.
julia --startup-file=no --check-bounds=yes --project=/tmp/pb844-integration-113-env \
    /tmp/pb844-solvers/test/runtests.jl --tier=full --jobs=2

# Baseline environment uses PB 8eaf8f81 and PS f9b7a266, with matched dependencies.
julia --project=<baseline-env> benchmark/block_prox_differential.jl > old.tsv
julia --project=<candidate-env> benchmark/block_prox_differential.jl > new.tsv
cmp old.tsv new.tsv

# The benchmark environments also contain BenchmarkTools and the appropriate PS source.
PB_BLOCK_BENCH_REPEATS=4 PB_BLOCK_BENCH_OUTPUT=<prefix> \
    taskset -c 2 julia --startup-file=no --compiled-modules=no \
    --check-bounds=no --project=<env> benchmark/block_prox.jl
```

The benchmark harness is the script on the completed review branch; the
package mathematics remains exactly that of the implementation pin. Its
SHA-256 is recorded in `performance.json`. The first repetition warms the
complete case set; repetitions 2–4 provide the paired medians. All final
measurements use fresh source loading and explicit bounds-check settings.
Earlier timings with randomized coordinate order or differently bounds-checked
cached code were discarded. Optimized Float64 least-squares/NormL1 scalar CD IR from fresh source matched
byte for byte after normalizing the approved scalar rename.

For a docs-only local build, omit the final `deploydocs` call from
`docs/make.jl` and run the remaining build from `docs/`. No documentation was
deployed. On a machine with limited space, `--pkgimages=no` avoids large
native package compilation caches; final benchmarks instead disable compiled
modules on both sides to ensure identical lowering settings.

## Review and release gates

PB #527 and PS #230 still track the outstanding owner/release integration.
The companion changes do not implement the separate symmetric-owner or
SparseArrays-alias migration. Baseline and candidate downstream full suites
are compared explicitly so those existing failures cannot be mistaken for
#844 regressions. Green remote PB/PS CI and resolution of the existing release
blockers and the measured writer tradeoff remain review gates before closing
#844 or merging. Nothing was merged.
