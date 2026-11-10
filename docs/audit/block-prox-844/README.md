# #844 implementation and validation

The implementation is local and unmerged. The mathematical API is committed at
PB `8fb174454a5b0c431376decf5a3c87830353a562` (tree
`58fdb820ad84d9c30d317ccab58190dcd5ba0e04`). The companion PS commit is
`75894887d7272df71c2c263feb4ea0b6d211fd48`, based on
`f9b7a26663b499b6f881bcc26c56d1acced06749`.

[ProximalSolvers.patch](ProximalSolvers.patch) is a `git format-patch` artifact;
apply it to that PS baseline with `git am`. The tested checkout is also in
`/tmp/pb844-solvers`. Its root project, Julia 1.10 manifest and docs project
all pin the PB implementation commit above. The PB commit must be available
from the chosen Git remote before those remote pins can be instantiated;
local integration uses `Pkg.develop` with the two exact checked-out sources.
Nothing was pushed, merged, or posted to the issue trackers.

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
