# Test tiers: baseline, placement and measurements

This file records why the test suite is split into tiers, where each file sits and why, which
tests moved between files, and what the tiers cost on GitHub-hosted runners.
[CONTRIBUTING.md](../CONTRIBUTING.md) is the user guide; [`tiers.toml`](tiers.toml) lists the
files of each tier and [`.github/tier-rules.toml`](../.github/tier-rules.toml) says which tier CI
runs for a change.

## Baseline (before tiers)

On 2026-11-01 every pull request ran the core tier (`--core`: every file except
`precision_mixed`, `large_inputs`, `spectral_overflow` and `conformance/*`) on Julia lts and
Julia 1, Ubuntu, 4 workers each; Julia 1 also collected coverage. Master ran all 135 files on
five jobs. Times from the job logs of run 41348877210 (PR) and 41341537016 (master):

| Run | Job | Files | Job time | Worker time (sum over files) |
|---|---|---:|---:|---:|
| PR, core | Julia lts, Ubuntu | 117 | 20.1 min | 4252 s |
| PR, core | Julia 1, Ubuntu, coverage | 117 | 29.2 min | 6306 s |
| master, all | Julia lts, Ubuntu | 135 | 30.3 min | 6518 s |
| master, all | Julia 1, Ubuntu, coverage | 135 | 33.0 min | 6968 s |
| master, all | Julia pre, Ubuntu | 135 | 37.1 min | 8013 s |
| master, all | Julia 1, Windows | 135 | 37.2 min | 7734 s |
| master, all | Julia 1, macOS (2 workers) | 135 | 46.0 min | 4458 s |

A PR therefore waited about 29 minutes for its slowest required job (other PR runs that day:
26–30 min). With 4 workers the wall time is bounded below by a quarter of the worker time, so
balancing the files better could not have helped much: the work itself had to shrink.

**Compilation dominates.** Every test file was run twice in one fresh process on the Discovery
cluster (AMD EPYC 7542, one file per core, `--check-bounds=yes`, commit 83a2eef9), first cold
and then again in a new sandbox module, which reuses the package methods compiled by the first
run:

| Julia | Cold, sum over 135 files | Compilation in the cold runs | Warm rerun | Execution (warm minus its compilation) |
|---|---:|---:|---:|---:|
| 1.10.12 | 5856 s | 5457 s (93%) | 892 s | 98 s (1.7%) |
| 1.13.3 | 6662 s | 6057 s (91%) | 1032 s | 102 s (1.5%) |

The warm reruns still compile the closures and types each test file defines. Executing the
assertions takes under 2% of the time: the cost is compiling specializations. ParallelTestRunner's
`--verbose` output, now on in CI, shows the same on GitHub runners: 94–98% compilation per tier
(Julia 1.13). A test file's cost grows with the distinct method
specializations it reaches (function types × element types × shapes × wrappers × iterate
types), not with its number of assertions. Coverage on Julia 1 made it worse: code under
coverage is compiled without the package image, which is why the Julia 1 PR job took 6306
worker seconds against 4252 on lts.

The most expensive files on the Julia 1 PR job (worker seconds, lts / Julia 1):
`gpu` 377 / 573, `conjugate_projections` 343 / 465, `constrained` 112 / 187,
`blockchecks_wrappers` 146 / 186, `symmetric_groups` 109 / 168, `blockchecks_steps` 128 / 162,
`metric` 103 / 155, `blockchecks_values` 81 / 139, `stepchecks` 68 / 137, `hyperslab` 84 / 129,
`blockchecks` 97 / 109, `smooth` 71 / 107. `allocations` (50 / 80) ran serially after the
parallel batch.

## How the tiers were chosen

- **Quick** (every PR) holds cheap files that each test a basic invariant across the public
  API: loading, Aqua, the generic interface and traits; iterates (active sets, value writes
  versus state copies, `dropzeros!`, symmetric coordinates and exact symmetry, aliasing,
  read-only handles); values, gradients and proximal operators of representative separable
  and nonseparable functions; conjugates as projections; constraints; spectral functions;
  precision, overflow and non-finite boundary cases; validation before writes; zero
  allocations on hot paths; and the guard against ambiguities with the GPU extension
  (`gpu_extension_ambiguities`). The device-array suites on JLArrays are in Full (below).
  The five most expensive files contributed representative cases, moved into new files
  (below), so that Quick keeps their main identities and regressions without their grids.
  Five of the Quick files, `atom_iterate`, `conjugate_basics`, `lmo_atoms`, `nested_scaled`
  and `sparse_iterate_basics`, keep the Float64 pass of each operation family and a Float32
  representative of it; the cases that repeat those (Float32 and Float16 passes, one mixed-type
  product, the atomicity cases of `GroupNormLinf` and `GroupSCAD`) are in the Full-tier `_grids`
  files (below). A sixth, `constrained_basics`, keeps the Float64 passes, the Float32 pass of the
  vector, and for the refusal of views and representations three of the eighteen forms; its
  Float32 passes of the other representations, five destination element types and fifteen of the
  eighteen forms are in `constrained_basics_grids`.
- **Standard** adds the files of most function families, wrappers, constraints, smooth losses
  and the examples: what an ordinary implementation change can break.
- **Thorough** adds what shared or numerical changes can break: precision and overflow
  recovery, non-finite values, spectral domains and tolerances, the conformance suite,
  symmetric writes, products and parameters, block steps, metrics, the wrapper calculus,
  the remaining iterate and constraint grids, and the specialized families (cones, graphs,
  fibers, polyhedra). Three Thorough files, `scaled_infinite_level_underflow`,
  `zero_level_infinite_step` and `symmetric_groups`, keep the Float64 cases (all except the
  wrapper variants of 16 indicators in `scaled_infinite_level_underflow`) and Float32
  representatives; the Float32 passes and indicator-wrapper variants that repeat those are in
  `scaled_infinite_level_underflow_grids`, `zero_level_infinite_step_grids` and
  `symmetric_groups_grids`, which are in Full (below). Three more, `extreme_normals`,
  `constrained` and `metric`, keep Float64 representatives, with two exceptions listed per file
  below: `constrained` moves six Float64 vector-box cells of *a separable function on a box*, and
  `metric` moves the Float64 main-loop cases of two of its five penalties. The Float32 passes and
  type grids that repeat those are in `extreme_normals_grids`, `constrained_grids` and
  `metric_grids`, also in Full (below). Four more, `value_overflow_glm`, `symmetric_wrappers` and `block_step`
  keep the Float64 passes, and `blockchecks` runs the flat sum and the `SliceSum` of each of its
  40 cases, the `FiberSum` with the function as the off-diagonal part for the cases of a vector and
  a proximable function, and the wrapper variants of nine of the 40; the cases are listed per file
  below. The Float32 storage cases, Float32 passes and wrapper variants that repeat those are in
  `value_overflow_glm_grids`, `symmetric_wrappers_grids`, `blockchecks_grids` and
  `block_step_grids`, also in Full (below).
  `test/calculus.jl` stays whole in Thorough and has no companion. Its profiled hotspots
  (failure atomicity before writes of the tuple wrappers, zero preservation, and buffers and
  allocations) run mostly on distinct Float64 wrapper types. The one measured candidate for a
  split is *Tilt and Translate on tuples refuse restricted outputs before writing* (eleven
  restricted wrapper builders at `test/calculus.jl:1295-1306`, one shared union dispatch at
  `src/calculus.jl:885-890`, about 3 s in isolation), which was not moved.
- **Full** adds the exhaustive catalogs whose cases repeat paths the tiers above already reach:
  the device-array suites on JLArrays (`gpu`, `gpu_smoke`, `conformance/gpu_arrays`), `conjugate_projections` and
  `allocations`; `precision_mixed`; the block
  checks on steps, values, wrappers and workspaces; the frozen weight-step formulas; the
  complex-input rejection catalogs (`real_*`); the symmetric mutator and parameter inventories;
  separable block traversal; the wrapper rejected-step catalog (`stepchecks`); and the family
  files `hyperslab`, `image`, `distl2` and `precompose_array`.

Moving a file to a later tier never removes it from pre-merge testing of its own code: the
rules in `.github/tier-rules.toml` run a family's files beside any change to that family
(`focus`), for example `hyperslab` with `src/functions/sets.jl` and `stepchecks` with
`src/calculus.jl`, and every changed test file runs beside the tier.

Every file is listed once in `tiers.toml`; full runs all of them, so the full tier runs
exactly the files the old complete suite ran, plus the new files split out of them.
`test/tier_selection.jl` checks this, the cumulative inclusion, and the CI risk rules.

Files added since then sit with the files of the code they test. `infinite_level_values`
(standard) holds the values of `NormL2`, `NormLinf`, `NuclearNorm` and `Scaled` with an infinite
level or factor: family behaviour beside `norms` and `infinite_weights`, a few seconds of small
cases, with no grid that would need the thorough tier. `sparsegroup_infinite_prox` (thorough)
holds the proximal map of `SparseGroupNormL2` with an infinite ℓ1 level: a grid over group levels,
weights, steps and non-finite entries on vectors, `AtomIterate`, `SymmetricSparseIterate`
outputs, JLArrays and the wrappers, about 60,000 assertions, beside its siblings
`sparsegroup_infinite_level` and `l2_nan_blocks`.
`nested_scaled_values` (standard) holds the values of nested `Scaled` chains whose product does not
fold into a normal level, a grid over the levelled families and the wrappers that fold them in both
directions and in Float32 and Float64, and the fused values at the stored output, next to
`nested_scaled`, which takes a minute of compilation in a fresh process (its Float32 repeats are in
`nested_scaled_grids`, below); the values add no grid that would need the thorough tier. `bigfloat_integer_operands` (standard) holds the exact integer operands of BigFloat
arithmetic in `NormL1Fused` and the pooling of `IndIsotonic` and `SLOPE`, the segment lengths of the total-variation kernel of
`TotalVariation1D` and `FusedLasso`, and the value at the stored output of `NormL1Fused`, at precisions of two to five bits where the
platform decides whether an integer such as 5 is rounded: a few seconds, with no large grid.

## Measured times on GitHub

End to end is from the run's creation to its last job (the gate); the test jobs run in
parallel after the plan job (about 20 s). Ubuntu runners for this public repository have 4
cores; the CPU model varies between runs (AMD EPYC 7763, 9V74, Intel Xeon 8370C, 8573C), and the
same files take up to 1.8 times longer on a slow one, so each tier was run three times. Julia 1
has no coverage below full, and is then often faster than Julia lts. A cold cache would add
package installation and precompilation of the dependencies (a few minutes once per cache key).

| Tier | Target | End to end, runs 1 / 2 / 3 | Test job, Julia lts | Test job, Julia 1 |
|---|---:|---|---|---|
| quick | 3–5 min | 3.0 / 4.6 / 4.9 min | 2.4 / 4.1 / 4.4 min | 2.4 / 3.4 / 3.2 min |
| standard | 8–10 min | 11.7 / 9.6 / 9.5 min | 11.2 / 6.9 / 7.9 min | 8.9 / 9.1 / 9.0 min |
| thorough | 15–20 min | 26.4 / 21.2 / 16.7 min | 25.1 / 20.3 / 15.8 min | 25.9 / 20.7 / 16.2 min |
| full (five jobs) | about 30 min | 41.0 / 34.9 min | 31.3 / 24.4 min | 31.4 / 34.4 min (coverage) |

Runs 1 used the first placement (quick 34 files, standard 78, thorough 135); runs 2 and 3 the
measured placement (quick 37, standard 64, thorough 119 and 118). After run 3, `nonfinite_inputs`
moved from quick back to thorough (about 50 worker seconds on a slow runner), leaving quick at 36
files. Full: macOS took 40.5 and 23.5 min, Windows 37.1 and 23.3, Julia pre 30.1 and 29.7.

At the final placement (commit fafb1189): quick took 4.5 min end to end (run 41369981986; Julia
lts 4.0 min on an EPYC 7763, Julia 1 2.7 min on an EPYC 9V45; the runner itself 2.9 and 2.0
min). Full took 43.0 min as this PR's run (41369978673: macOS 41.1, Windows 36.3, Julia pre 35.0,
Julia 1 with coverage 33.4, Julia lts 20.5) and 35.1 min as a manual run without inputs
(41369984297). Both full runs and the thorough runs checked the conformance coverage map and
uploaded the reports.
Inside the job, about 50–70 s go to checkout, the Julia cache, the test environment and
precompiling ProximalBase with bounds checking (about 25 s); the rest is the test runner.

**Before and after.** PR runs on 2026-11-01 took 26–30 min end to end (median 28.4 min over
eight runs, all on the core tier). An ordinary PR now waits 3–5 min (quick), one that changes a
function family 8–10 min (standard), and one that changes shared code 17–21 min (thorough). The
full suite after merge takes 25–41 min, as master's complete runs did before (21–46 min per job).

## Tests moved into quick-tier files

Each split moved whole testsets, or split one loop's parameter tuple into a representative
subset and its complement, so that the original file plus the new file run exactly the
assertions the original ran. No assertion, tolerance, input or expected error changed and
nothing was deleted. Assertion counts (from the runner) confirm it.

| New file (quick) | From (later tier) | What moved | Assertions: before = after + new |
|---|---|---|---|
| `sparse_iterate_basics` | `sparse_iterate` (thorough) | 17 whole testsets: construction, `dropzeros!`/nnz, `activate!`, active coordinates, converting writes, `fill!`/norm, copies and the conversion-failure regression (#416), stored zeros in `mul!` (Float64), the symmetric matrix (coordinates, mirrored writes, exact symmetry, `copyto!`), triple products (Float64 in quick, Float32 in `sparse_iterate_basics_grids`, below), aliasing of views | 2897 = 2577 + 320 |
| `allocations_basics` | `allocations` (full) | Loop entries: block steps on a `SparseIterate` and with `NormL1`/`GroupNormL2`; `GroupNormL2` with distinct group layouts; `NormL1` on vectors, sparse, atom and symmetric iterates; `GroupNormL2` prox; `LeastSquaresLoss` gradients | 2910 = 2766 + 144 |
| `constrained_basics` | `constrained` (thorough) | `NormL1` and centered `SqrNormL2` on four boxes with the brute-force optimality check; coordinate steps on a box; `Scaled(NormL1)` wrappers; simplices, closed forms, the zero-sum lasso KKT check, errors, general normals with a box, Float32 cases; Float64 open-domain boxes | 4108 = 3526 + 581 (+1, see below) |
| `conjugate_basics` | `conjugate_projections` (full) | Seven direct conjugate families (clipping, check loss, orthant, l2 ball, group l2 ball, huge·tiny scale, shift) in Float64, and in Float32 (four of the seven families stay in quick; the other three are in `conjugate_basics_grids`, below); the Float64 Moreau identity with infinite weights; nested mixed sums and failure atomicity; output aliasing of parameters; biconjugates; Float64 l1-ball radius accuracy and group l1 balls | 265693 = 259835 + 5858 |
| `gpu_smoke` (with `gpu_prelude`, `gpu_suite_basics`; full tier since #771) | `gpu` (full) | `NormL1` (plain and weighted) and the user bits type; `NormL2`, `IndSimplex`; `GroupNormL2` on host and device layouts; weighted norms at device overflow; least-squares and logistic losses with a design matrix; infinite weights; Moreau array steps | 21767 = 21469 + 298 |

The ambiguity guard for the GPU extension moved from `gpu` to `gpu_smoke` (through `gpu_suite_basics`;
since #771 it is in the helper `gpu_ambiguities`, which the quick file `gpu_extension_ambiguities` runs too),
and `gpu_suite_basics` gained the `_maximum_abs` tests for a `Diagonal` of a device vector. The
runner's counts for `gpu` and `gpu_smoke`: 21469 and 319 at the base (21788 together, the guard
failing there), 21468 and 412 at the head (21880 together): the guard's assertion moved, and 92 are new.

`constrained`: one check in `ProximalBase.Testing.test_prox` runs only for random candidate
points inside the constraint set, so its count depends on the random draws (110–112 for the
moved block under seeds 1–12); the moved code changed which draws later testsets see.

Moving cases changes the global random stream for the testsets that follow them in
`sparse_iterate`, `constrained` and `allocations` (each test file starts from `Random.seed!(1)`).
Every affected testset passes; none depended on its position in the stream. In `gpu`, every
remaining testset starts in the same random state as before. `test/cuda/runtests.jl` now runs
`gpu_prelude`, `gpu_suite_basics` and then, after reseeding, `gpu_suite`, so the physical GPU
covers exactly what it covered before, with the same random inputs as `test/gpu.jl`.

**Coverage.** The full tier's Codecov upload (Julia 1, Ubuntu, run 41361146694 at dad59957)
covers 14160 of 14618 lines (96.86%), the same 14160 lines as master at the base commit 83a2eef9.
**Assertions.** At the same base, every full job passes exactly 116 more assertions than master's
run 41341879373 (Julia 1: 639818 against 639702; broken counts equal): the 117 of
`tier_selection` less the one random-draw-dependent check in `constrained`.

### The `_grids` files

Six Quick files and ten Thorough files have a Full-tier companion with the cases that repeat one
that stays in the base file for another element type, wrapper or view form. Each companion is a self-contained module with the fixtures it needs
copied, and its name starts with the name of the base file, so `julia scripts/test.jl atom_iterate`
runs both. The cases are moved with their assertions; the loop guards and the copied fixtures were
adjusted. Quick
keeps the Float64 pass of each operation family, a Float32 representative of each, and an atomicity,
aliasing or NaN case of each kind for a representative function (listed per file below). The Float64
cases that moved are listed per file in the table below.

| Base file + companion (Full) | What the companion holds | Assertions: before = base file + grids |
|---|---|---|
| `nested_scaled` + `nested_scaled_grids` | the Float32 conjugates of the nested scales (#333), five `check` calls of the steps outside the range (#566) and the explicit infinite factor under the fold (#566); the Float32 allocation check of `Conjugate(Scaled(...))` stays in Quick | 1192 = 1057 + 135 |
| `atom_iterate` + `atom_iterate_grids` | the Float32 x Float64 -> Float64 product and the Float32 strided views of the five-argument products (the Float64 x Float32 -> Float64 triple stays in Quick); the failed-step atomicity cases of the group wrappers for `GroupNormLinf` and `GroupSCAD`, in Float64 (`GroupNormL2`, `GroupMCP` and `SparseGroupNormL2` stay in Quick); the Float32 pass of the group view norms and dots | 3998 = 3752 + 246 |
| `lmo_atoms` + `lmo_atoms_grids` | the Float32 passes of the density matrices, the Krylov method, the `Symmetric` and `Hermitian` wrappers and the NaN and Inf support values | 3673 = 3029 + 644 |
| `conjugate_basics` + `conjugate_basics_grids` | the Float32 `CheckLoss`, `IndNonnegative` and `Translate` families of the direct conjugate projections | 5858 = 5675 + 183 |
| `sparse_iterate_basics` + `sparse_iterate_basics_grids` | the Float16 pass and the Float32 `lmul!`, `ldiv!` cases of the scaling by a number (FN-9); the Float32 sparse triple products | 2840 = 1604 + 1236 |
| `constrained_basics` + `constrained_basics_grids` | the Float32 passes of five of the six representations of the fused value (the Float32 vector stays in Quick) and of the parameter types on a vector (levels, weights, bounds and steps); the destinations of element type `Rational`, `Real`, `AbstractFloat`, `BigFloat` and `Bool`; the refusal in each constructible role of fifteen of the eighteen forms of views, reshapes and other representations (a range view, a `UnitRange{Union}` view and a `Union`-element `Array` stay in Quick) | 10857 = 7756 + 3101 |
| `scaled_infinite_level_underflow` (Thorough) + `scaled_infinite_level_underflow_grids` | in "the listed indicators are accepted": the Float32 pass of seventeen of the twenty indicators (alone, under `Scaled` and under `Translate(Scaled)`) and the `Scaled` and `Translate(Scaled)` variants of sixteen indicators in Float64; the base file keeps all twenty indicators alone in Float64, the wrapper variants of `IndBox`, `IndSimplex`, `IndSOC` and `IndPattern` in Float64, and `IndBox`, `IndSimplex` and `IndPattern` with their wrappers in Float32 | 115533 = 115184 + 349 |
| `zero_level_infinite_step` (Thorough) + `zero_level_infinite_step_grids` | the Float32 pass of `Regularize` with a zero level for six of the eleven functions of the function loop, three of the six of the array steps and three of the four of the per-group steps and the coordinate step; Float64 stays whole in the base file | 14528 = 11277 + 3251 |
| `symmetric_groups` (Thorough) + `symmetric_groups_grids` | the Float32 pass of the symmetric group writes for `GroupNormLinf` (its levels, weights and zero-level variants and its `rejects_group_steps` cases) and for the radial `GroupSCAD`; Float64 stays whole in the base file, and Float32 keeps `NormL2`, `GroupNormL2`, `SparseGroupNormL2` and `GroupMCP` | 12327 = 10491 + 1836 |
| `extreme_normals` (Thorough) + `extreme_normals_grids` | in the overflow catalog, the Float32 pass of *scalar normals*, *rescaled normals*, *tiny entries keep their bits*, *the halfspace is judged by the residual*, *iterates* and *the neighbourhood of the thresholds*, and the destinations of Float32 data into a Float64 output; the base file keeps the Float64 pass, the Float32 *array normals* and *steps whose weighted squared norm or product overflows*, and the destinations of a Float64 model into a Float32 output | 24834 = 15658 + 9176 |
| `constrained` (Thorough) + `constrained_grids` | in *a separable function on a box*, the `IndBox(l, u)` cells of six of the eight functions, and the Float32 pass of *variable fixing with extreme normals*; the base file keeps `NormL1` and `ElasticNet` on that box (the other three boxes build types that the scalar box already builds) and the Float64 pass | 4935 = 4366 + 569 |
| `metric` (Thorough) + `metric_grids` | the `GroupNormLinf` and `GroupSCAD` cases of the main loop of *per-group steps for regularization*; the base file keeps `GroupNormL2`, `SparseGroupNormL2` and `GroupMCP` there and all five penalties in the reordered-groups loop, and the kernels of the two moved penalties stay checked in *a step per group for group norms* and *for MCP and SCAD* | 1814 = 1566 + 248 |
| `value_overflow_glm` (Thorough) + `value_overflow_glm_grids` | in *GLMLoss values and gradients that leave the range*, the Float32 cases with a sparse `X`, a `SparseIterate` coefficient (also with zero entries) and an `AtomIterate` coefficient of Logistic, Huber, SmoothCheck and the two Conv losses, and the first three of those four storages of Poisson; the base file keeps the Float64 pass, the dense Float32 cases of these losses, the `SparseIterate` coefficient with zero entries of Poisson, and all the Float32 cases of Squared and SquaredHinge | 28847 = 28521 + 326 |
| `symmetric_wrappers` (Thorough) + `symmetric_wrappers_grids` | the Float32 pass of *a separable sum with a rejection-only owner block writes no block* except its *supported blocks* testsets, and the Float32 entries of three composed `NuclearNorm` combinations of *the exact check applies to every other combination* (`Conjugate(Translate)`, `Conjugate(Regularize)` and `MoreauEnvelope(Conjugate)`); the base file keeps the Float64 pass of both testsets, the Float32 supported blocks and the Float32 entries of the other combinations | 4815 = 4233 + 582 |
| `blockchecks` (Thorough) + `blockchecks_grids` | in *Block checks before writes*, the `Scaled`, nested `SeparableSum` and `FiberSum` variants (the function as the diagonal or with no diagonal, for a vector and a proximable function) of 31 of the 40 cases; the base file keeps those variants of nine cases and, for each of the 40, the `@test E <: ...` check, the flat `SeparableSum`, the `SliceSum` and, for a vector and a proximable function, the `FiberSum` variant with the function as the off-diagonal part, and its other testsets | 5077 = 3277 + 1800 |
| `block_step` (Thorough) + `block_step_grids` | the Float32 pass of *block steps / references* except seven of its 83 (function, iterate) combinations; the base file keeps the Float64 pass, those seven Float32 combinations and the other testsets of *block steps* | 1378 = 958 + 420 |

The counts of the last eleven rows are the passes of each file, measured with the profile driver on Julia 1.13.3;
the first number of a row is the base file before the move (10857 for `constrained_basics`). Six companions repeat cases of a Quick file and ten repeat cases of a Thorough
file; a companion is selected in the same way for either, and the table in `test/tier_selection.jl` has the
tier of the base file of each companion.

The base file does not run these cases (Quick for six companions, Thorough for the ten others). They run in these places:

- In a pull request, for a change to the source of their own family, beside the tier (`focus` lists
  in `.github/tier-rules.toml`; the rules add focus files and do not change a tier):
  - `atom_iterate_grids`: `src/atom_iterate.jl`, `src/functions/norms.jl`,
    `src/functions/sparsegroup.jl`, `src/functions/nonconvex.jl`;
  - `sparse_iterate_basics_grids`: `src/sparse_iterate.jl`;
  - `nested_scaled_grids`: `src/calculus.jl`, the sources shared by every function (`src/metric.jl`,
    `src/utils.jl`, `src/interface.jl`, ...), `src/functions/singular.jl`;
  - `conjugate_basics_grids`: `src/calculus.jl`, the shared sources, `src/functions/losses.jl`;
  - `lmo_atoms_grids`: `src/lmo.jl`, `src/lmo_atoms.jl`, the shared sources, `src/functions/singular.jl`;
  - `constrained_basics_grids`: `src/constrained.jl` (before this rule, a change to that file matched the
    rule for `src/**`, tier standard, which includes the cases that moved);
  - `block_step_grids`: `src/calculus.jl`, the shared sources, `src/atom_iterate.jl` and
    `src/sparse_iterate.jl` (`block_step` is a focus file of the rules for those two paths, and each of
    them lists `block_step_grids` beside it, so a pull request on one of them runs the Float32
    combinations that moved out of `block_step`);
  - `scaled_infinite_level_underflow_grids`, `zero_level_infinite_step_grids`, `symmetric_groups_grids`,
    `extreme_normals_grids`, `constrained_grids`, `metric_grids`, `value_overflow_glm_grids`,
    `symmetric_wrappers_grids`, `blockchecks_grids`:
    `src/calculus.jl` and the shared sources. Their base files are Thorough-tier files, as is that of
    `block_step_grids`. The two thorough
    rules, the rule for `src/calculus.jl` and the rule for the shared sources, list these ten companions
    (these nine and `block_step_grids`), so a
    pull request on one of those paths runs them beside thorough, and the base files run the Float64 cases
    and the Float32 representatives that stay. No standard-tier rule lists these nine, and the two
    iterate rules above list `block_step_grids` and none of the other nine: the rule of another standard-tier path alone does not run the base files, and a
    companion would run the Float32 repeats without the Float64 baseline that the base file keeps. A pull request that is
    thorough for another reason (the label `test:thorough`, which selects none of them, or a path whose rule does not list them) runs the
    base files without these companions when no changed file or other rule adds one (`test:full` selects all of them, a changed companion
    selects itself, and the shared rule and the rule for `src/calculus.jl` list all ten), and the cases that
    moved run in the nightly full tier.
  With the label `test:iterate-narrow`, a pull request that changes a single `AtomIterate` or
  `SparseIterate` implementation source (`src/atom_iterate.jl` or `src/sparse_iterate.jl`) and no path
  that needs a higher tier gives quick and that iterate's own grids file. `src/symmetric_parameters.jl`
  gives quick with no grids file, two iterate implementations give standard, and a simultaneous change
  to shared code such as `src/utils.jl` gives thorough.
- In the full tier: the nightly run, tag runs, and a pull request whose tier is full (the label
  `test:full`, or a path whose rule needs full, such as `src/lapack.jl`).
- When requested: `julia scripts/test.jl <name>_grids`, or a change to the grids file itself.

Other source paths do not add the grids files as automatic focus files in a pull request. A master
push that changes package tests or code runs the thorough tier, which has the Thorough base files
and none of the companions (a documentation-only push runs no package tests).
`test/tier_selection.jl` keeps one table of the companions (the companion, its base file, the tier of the
base file and the source paths of its family with their tiers), and from it checks that each companion is in
Full and in no lower tier, that its base file is in the tier of the table, the map above with `classify`
(the direct-family paths, with some of the shared sources as representatives: `src/interface.jl`, for
example, also gets the thirteen grids files of the shared rule), that the named unrelated paths (`entropy.jl`, `deprecated.jl`, `sets.jl`, ...) get no grids file as
focus, and what a master push, the nightly run and `test:full` select.

In the nightly triage (`[[triage.deferred]]` in `.github/tier-rules.toml`), a failure of
`extreme_normals_grids` is deferred in the group "extreme precision and exponent ranges", beside
`extreme_normals` (the globs of that table are anchored, so the entry of `extreme_normals` does not
match the companion, and `test/ci_triage.jl` checks the group of both). The other nine companions of
Thorough files (`scaled_infinite_level_underflow_grids`, `zero_level_infinite_step_grids`,
`symmetric_groups_grids`, `constrained_grids`, `metric_grids`, `value_overflow_glm_grids`,
`symmetric_wrappers_grids`, `blockchecks_grids` and `block_step_grids`) are in no deferred group.

## JLArrays and CUDA

`gpu_smoke`, `gpu` and `conformance/gpu_arrays` (all full tier) run on JLArrays, GPUArrays' reference
backend on the CPU, with scalar indexing disallowed: they catch code that indexes device arrays element
by element or does not adapt its parameters. They do not test a physical device. `test/cuda` runs the
same shared cases (`gpu_prelude`, `gpu_suite_basics`, `gpu_suite`) on an NVIDIA GPU by hand (see its
README); no tier and no CI job runs it, and `test/oracles` (JuMP and conic solvers) is likewise run by
hand.

### The device-array suites leave the default tiers (#771, part 1)

Policy of #769: the pull-request tiers keep what a change to the package can break in minutes; a
suite whose cases repeat paths that the CPU files already reach runs nightly. The device-array suites
are such a suite: they exercise the same functions through `JLArray`, and a regression found only there
comes from the GPU extension, the GPU test files or code that dispatches on device arrays.

| File | Before | After | Worker time, Julia 1.13.3 on one cluster node, one sample each |
|---|---|---|---:|
| `gpu_smoke` | quick | full | 90.4 s in the full quick tier at the base (16 workers), 83.8 s in standard (82% compilation); 58.4 s alone |
| `conformance/gpu_arrays` | thorough | full | 67.1 s in the thorough tier at the base (16 workers); 42.2 s alone (3 workers on 4 cores) |
| `gpu` | full | full | not on the pull-request path |
| `gpu_extension_ambiguities` | (new) | quick | 11.0 s alone (same run as the two above) |

The ambiguity guard stays in quick: it moved into the helper `gpu_ambiguities`, which
`gpu_suite_basics` includes (so full `gpu_smoke` and the physical CUDA run still execute it) and which
the new quick file `gpu_extension_ambiguities` includes, after asserting that the extension is loaded so
that it cannot pass without it. A quick run therefore saves the time of `gpu_smoke` less that of the
new file (about 80 s of worker time in these samples) and a thorough run the time of
`conformance/gpu_arrays`. The samples were taken under different loads, on one machine, and are not an
end-to-end CI time; the CI before and after is taken from three
comparable runs.

A pull request still runs the device-array files when it changes the GPU test files or helpers
(`.github/tier-rules.toml`), the conformance suite, its scripts or the coverage map (which add
`conformance/gpu_arrays`, so that the map check has the complete inventory), the GPU extension (`ext/`
is full), or when it carries the label `test:gpu`. A change that affects device arrays without any of
these is first seen by the nightly run. 31 test files load JLArrays as a second array type for CPU
catalogs; they stay in their tiers, so device arrays remain exercised by `src/` changes.

In CI, a focus run of `conformance/gpu_arrays` below thorough executes its assertions and writes no
report (the harness writes reports as rows are emitted when `CONFORMANCE_REPORT` and
`CONFORMANCE_RUN_ID` are both nonempty; CI sets the run id for every tier and gives `CONFORMANCE_REPORT`
a value only for thorough and full runs). Thorough without it writes 13 of the 14 report files, and
`scripts/conformance_map.jl --check --tier thorough --extra <focus files>` then compares a partial
inventory (CONTRIBUTING.md, `docs/design/numerical-contract/README.md`); the full check, with all
14 files, is unchanged.

## CI scenarios

Checked on GitHub with draft PRs against this branch (#431–#437, closed unmerged), so that each
diff held only the scenario's change:

| Scenario | PR | Plan | Outcome | PR merge gate |
|---|---|---|---|---|
| a test file in quick changed (`test/lla.jl`) | #431 | quick | passed, 4.1 min | passed |
| label `test:standard` added | #431 | standard | the next label cancelled it | **failed** (cancelled) |
| label `test:full` added | #431 | full, five platform jobs | cancelled by hand | **failed** (cancelled) |
| a test file outside quick changed (`test/hyperslab.jl`) | #437 | quick and `hyperslab` | passed, 4.6 min | passed |
| family code without a test change (`src/functions/norms.jl`) | #432 | standard | passed, 8.4 min | passed |
| shared code (`src/utils.jl`) | #433 | thorough | passed, 27.1 min | passed |
| documentation only (`docs/src/index.md`) | #434 | none | no test jobs, 0.4 min | passed |
| a failing assertion in a quick file | #435 | quick | both test jobs failed | **failed** |
| a test file missing from `tiers.toml` | #436 | the plan job failed | no test jobs | **failed** |
| the runner, CI and manifest (this PR) | #430 | full, five platform jobs | passed, 41 min | passed |
| manual runs of quick, standard and thorough | — | as chosen | passed | (named "CI result") |

Every check ran on the PR's head commit; the plan, test and gate jobs print it. The gate reads
the conclusion of each expected test job of its own run, so a skipped, cancelled, missing or
failed job fails it. Pushes to master and the nightly run can only be observed after the merge.
These scenarios ran with the first gate, a job after the tests; the next section describes the
gate that replaced it and the scenarios that checked it.

## The merge gate, the tier lists and the report upload (review of #430)

Three review comments on #430 changed the gate, the rule for `tiers.toml` and the conformance
report upload. Scenario PRs #443–#453 checked them on GitHub, against `ci/gate-scenarios` (a
scratch copy of this branch at f76a24d8) with a temporary ruleset that required `PR merge gate`
on that branch only, as branch protection will on master; both were deleted afterwards.

**The gate starts with the run.** The first gate had `needs: [plan, test, analysis]`, and GitHub
creates a job's check only when the job is queued. A label on an unchanged commit started a new
run, and until its tests finished the commit showed only the earlier, green gate (#431). The
gate job now has no `needs`, so its check exists, pending, from the start of the run. It polls
this run's jobs through the API (`scripts/ci_gate.jl`, read-only token) and passes only when
**Required jobs** (the former gate check, still after the plan, the tests and the benchmark
controls) passed in its own attempt. A failed, cancelled, skipped, missing or duplicated
Required jobs fails it, as do a run for another commit or event, 15 minutes of API errors, every
other job finished for 90 s without a Required jobs, and 170 minutes without a decision.
`test/ci_gate.jl` checks these decisions on job states, including re-runs and API errors. The
issue for a failing master run is opened by its own job, the only one with `issues: write`,
which runs no repository code and is skipped on pull requests.

Branch protection counts the newest check of a name: on #443, with the ruleset in place, the
earlier green gate and the new pending one were both listed as required and the PR was BLOCKED
(rollup PENDING) until the new run passed.

| Scenario | PR (run) | Plan | What GitHub showed | Gate |
|---|---|---|---|---|
| quick | #443 (41384664046) | quick | gate in progress 3 s after the run was created; tests passed | passed, 5.2 min |
| label `test:full` on the passing commit | #443 (41385185212) | full, five jobs | the label request returned at 03:50:00.9, the new gate was listed (queued) and the PR unstable at 03:50:02.5; BLOCKED once the ruleset existed; CLEAN between 04:30:06 and 04:30:27, after the gate passed at 04:30:18 | passed, 40.3 min |
| the same, with the ruleset from the start; full fails on Windows | #453 | quick, then full | CLEAN after quick; label request 04:22:46.7–04:22:48.8, CLEAN at 04:22:48.7, BLOCKED at 04:22:50.6; the Windows job failed and the PR stayed BLOCKED | failed |
| label `test:thorough`, then removed | #444 | quick, thorough, quick | each label change listed a new pending gate within 6 s; thorough passed | passed (21.2 min for thorough) |
| superseded by a new commit | #450 | standard | the first run was cancelled with its gate | cancelled |
| the new run cancelled by hand | #450 | standard | gate cancelled, PR BLOCKED | cancelled |
| a test failing on Julia 1 only (lts passed) | #445, attempt 1 | quick | Required jobs: Julia 1 failure | failed |
| Re-run failed jobs | #445, attempt 2 | | only Julia 1 re-ran; the gate waited for the new Required jobs | passed |
| a re-run of the gate alone | #445, attempt 3 | | the attempt lists the passed Required jobs under its own number; gate passed in 16 s | passed |
| Re-run all jobs | #445, attempt 4 | | gate pending within 4 s; the old Required jobs was not listed | passed |
| Re-run job on one test job of a passed run | manual quick run 41390953488, attempt 2 | | the job and Required jobs re-ran; the gate did not and kept its result (a job without `needs` is no dependent) | unchanged |
| a test file missing from `tiers.toml` | #446 | the plan failed | | failed |
| no Windows job (stub workflow, no tests) | #448 | full | Required jobs: "Julia 1 - windows-latest - x64 - full: missing" | failed |
| no Required jobs (stub workflow, renamed) | #449 | full | gate failed 95 s after the last job | failed |
| an invalid workflow file (first push of #448 and #449) | — | no run | no gate check at all; the ruleset blocks a missing required check | none |
| documentation only | #447 | none | no test jobs | passed, 21 s |
| `tiers.toml` only: `utils` moved from quick to standard | #451 | full, five jobs | | passed, 42.0 min |
| a row removed from the coverage map | #452 | thorough | the map check on Julia lts failed | failed |
| this PR merged with master (#413 added `test/symmetric_sphere.jl`) | #430 (41384482255) | the plan failed on the unlisted file | | failed; the file is now in standard |

**What remains.** Between a label (or push) and GitHub creating the new run, the previous gate is
still the newest check: on #453 the PR was BLOCKED less than 4 s after the label request
started, and on #443 and #444 the new gate was listed 2–6 s after the label. No workflow can
close that window. **Re-run job** on a single job does not re-run the gate (it depends on no
job), so a failure in such a re-run of a passed run leaves the gate green; the next section
makes the verdict job a second required check for this. The gate job holds one Ubuntu runner for the length of its run (about 5
minutes for quick, 40 for full); runners are free for this public repository but count towards
its limit of concurrent jobs. It makes about three API requests a minute, well within the
token's 1000 an hour per repository.

**Time after the tests.** From the last required test job to the gate's result took 6–9 s with
the first gate (runs 41369981986, 41369978673, 41369984297) and 12–22 s now (quick 12, 14, 21,
22 s; thorough 15 s; full 15, 21 s): Required jobs takes about 5–8 s to start and finish, and the
gate polls every 20 s until it sees Required jobs or every other job finished, then every 5 s.
Quick took 4.6–5.3 min end to end in these scenarios (5.0, 4.6, 5.2 and 5.3 min, with up to eight
runs at a time) against 4.5 min for the final quick run of the first round.

**`tiers.toml` gets full.** It decides what every tier runs, so a valid edit can move a regression
test out of the tiers a PR runs and still pass `tier_selection`, which checks that each file is
listed once. Any change to it, including the line of a new test file, now plans full (#451);
ordinary test files keep quick and run beside it.

**Conformance report upload.** A full run's macOS job passed its tests and then failed on
`Failed to CreateArtifact: Unable to make request: ETIMEDOUT` (run 41373888409). Every job now
checks that the test step wrote the reports (required), retries a failed upload once after 30 s
(`overwrite: true`), and reports the outcome in its summary from the artifact id GitHub returned.
The Julia lts, Linux upload, from which the coverage map is regenerated, stays required; on the
other jobs a second failure is a warning. Two manual thorough runs on scenario branches whose
first upload pointed at a missing directory checked this: in 41385714788 the lts upload
succeeded on the retry (notice) and Julia 1 failed twice (warning, no artifact, job passed);
in 41385717877 the lts upload failed twice, which failed the job and the gate, and Julia 1
succeeded on the retry.

## Two required checks (second review of #430)

The second review found two gaps. **Re-run job** on one test job re-runs that job and the
verdict job, but not the gate (it depends on no job), so with only `PR merge gate` required a
failed re-run of a passed run left the gate green (manual run 41390953488). And the upload result
step wrote `::error::` for an unconfirmed required upload without exiting nonzero itself.

**The verdict job is the second required check.** It is named **PR required jobs** on pull
requests and **CI required jobs** on pushes, nightly and manual runs (the gate likewise `PR merge
gate` or `CI result`), so a manual run on a PR branch, whose checks land on the same commit,
cannot supply a passing check of the required name. The gate waits for the verdict of its own
event; `test/ci_gate.jl` checks that another name never counts. After the merge, branch
protection requires both `PR merge gate` and `PR required jobs` from GitHub Actions.

Scenario PRs #464–#472 checked this on GitHub against `ci/gate-scenarios` (this branch at
cfbe152a) with a temporary ruleset requiring both checks, and #472 against
`ci/gate-scenarios-gateonly` with a ruleset requiring only `PR merge gate`, the earlier advice.
Both rulesets and branches were deleted afterwards. The failures are real `@test`s that read
`GITHUB_RUN_ATTEMPT` and fail on Julia 1 in one attempt only; #470 stubs the tests. A watcher
polled the PR's merge state and the commit's check runs every 3–4 s (times UTC).

| Scenario | PR (run, attempt) | What GitHub showed | Merge state |
|---|---|---|---|
| A: quick | #464 (41448846473, 1) | both checks passed; 4.9 min end to end | CLEAN |
| D: Re-run job on Julia 1, which passes | #464 (2) | request 13:02:40–41; the new Julia 1 check queued; PR required jobs absent from the PR's checks until the job finished; created and passed at 13:06:48 | CLEAN at 13:02:33.6, BLOCKED from 13:02:41.5, CLEAN at 13:06:48.8 |
| B: label `test:full` on the passing commit | #464 (41450588756) | label request 13:09:45.1–47.4; new gate queued; full, five jobs, passed in 42.8 min; both checks passed at 13:52:24 and 13:52:38 | CLEAN at 13:09:48.0, BLOCKED from 13:09:50.4 to 13:52:17.7 (polled every 30 s after the first minute), CLEAN at 13:52:49.5 |
| C: Re-run job on Julia 1, which fails | #466 (41448860044, 2) | request 13:04:26–27; the gate was copied into the attempt as passed; Julia 1 failed; PR required jobs failed at 13:08:43 | CLEAN at 13:04:16.4, BLOCKED from 13:04:27.4 on |
| E: Re-run failed jobs after C | #466 (3) | Julia 1 and PR required jobs re-ran, the gate was copied | BLOCKED until PR required jobs passed; CLEAN at 13:14:00 |
| E: Re-run all jobs | #466 (4) | request 13:16:18–20; a new gate queued at once and every job re-ran; the gate passed 12 s after PR required jobs | CLEAN at 13:16:13.9, BLOCKED from 13:16:21.3, CLEAN at 13:21:25.2 |
| E: Re-run failed jobs after a failed run | #467 (41448872959, 1–2) | Julia 1 failed in attempt 1, failing both checks; the re-run gate was queued within 2 s of the request and waited | BLOCKED; CLEAN at 13:08:49.8 |
| E: Re-run job after a failed run, then the gate | #468 (41448883485, 1–3) | attempt 2 re-ran Julia 1 and PR required jobs, which passed, but the gate kept its failure; attempt 3 (Re-run failed jobs) re-ran only the gate, which passed 27 s after the request | BLOCKED until 13:10:08, then CLEAN |
| E: the gate's API requests fail, then the gate alone is re-run | #470 (41449027214, 1–2; stub tests, full matrix) | attempt 1: the gate failed on a nonexistent repository while PR required jobs passed; attempt 2: Re-run job on the gate, passed 29 s after the request | BLOCKED; CLEAN at 13:05:15.7 |
| control: C with only `PR merge gate` required | #472 (41450008718, 2) | request 13:11:16–17; the gate was copied into the attempt as passed at about 13:11:31; Julia 1 failed at 13:15:23 and PR required jobs at 13:15:34 | BLOCKED for 14 s, then UNSTABLE (mergeable) through the failure and after |

**Why a single-job re-run blocks at once.** The PR's checks show the newest attempt, and
GitHub lists in it only the jobs it has created there: jobs that are not re-run are copied with
their earlier results 10–15 s after the request, and re-run jobs appear when they are queued.
PR required jobs is created only when the re-run test job finishes, so until then the required
check is missing and the PR is BLOCKED (the commit's check-run listing still shows the earlier,
passed PR required jobs; branch protection does not use it). With only the gate required, the
copied gate satisfied protection while the job ran and after it failed (#472).

**What remains.** Between a label, push or re-run and GitHub creating the run or attempt, the
earlier checks are still the newest: 3–5 s on #464 for a label (2–6 s in the first round).
For a re-run the PR was BLOCKED at the first poll after the request returned (within a second
of it, on #464, #466 and #472), so any window is shorter than that, but polling cannot exclude
one. The block
*during* a single-job re-run rests on the observed listing above, which GitHub does not
document; if it changed, both checks would show their earlier results while the job runs, and
the PR would still be blocked once the re-run's PR required jobs is created (queued, then
failed). A stronger guarantee would need a second, trusted workflow that sets a pending commit
status when a re-run is requested (`workflow_run` with `statuses: write`), with its own start-up
delay and failure modes; for a window of seconds it is not worth it.

**The upload result step fails on its own.** `scripts/ci_upload_result.jl` replaces the inline
step: an upload counts only when its step succeeded and returned an artifact id, an unconfirmed
first upload (also a success without an id) is retried, and when the required Julia lts, Linux
upload is unconfirmed the step writes the summary line and exits 1. On the other jobs it warns,
and it names a failed test step instead of saying the tests passed. `test/ci_gate.jl` covers the
cases in a table (required or not; first upload with or without an id; retry; failed tests) and
the exit status and summary of the step. A manual thorough run, 41449072966, replaced the tests
and both upload steps by stubs that succeed without an artifact id: on Julia lts the result
step failed with the error, failing the job, CI required jobs and CI result; on Julia 1, whose
stub test step failed, the warning named the failed test step.

## Iterate changes (third review of #430)

The 0.6 work changes `SparseIterate`, `AtomIterate` and `SymmetricSparseIterate` in many small
PRs, and every change to their files ran thorough (17–21 min). The three files now have rules of
their own in `.github/tier-rules.toml`: **standard and a focus group** by default, and **quick
and the same focus group** with the label `test:iterate-narrow`. The label applies only when
exactly one of the three rules matches, lowers nothing else (shared code stays thorough,
infrastructure and unknown paths full, other package code without tests standard), and leaves
the focus files unchanged; `test:thorough` and `test:full` raise the tier as before. Whether a
change is narrow, shared or architectural is the author's declaration, checked in review
(CONTRIBUTING.md, "Iterate changes", and the PR template); no rule reads it from the diff.

**Focus groups**, chosen by what the files protect (worker seconds on Julia lts from the full
run 41364805347), beside the iterate files of quick (`active_handle`, `atom_iterate`,
`iterate_invariants`, `iterate_queries`, `iterate_writes`, `matrixview`,
`sparse_iterate_basics`, `symmetric_parameters`, `symmetric_products`, `wrapper_buffers`,
`allocations_basics`, `block_step_failures`). The grids file of each of the first two rules
(`atom_iterate_grids`, `sparse_iterate_basics_grids`, above) is a focus file of it, and
`block_step_grids` (above) is a focus file of both, beside `block_step` (the worker seconds below
are from the run named above and do not include it):

| Rule | Focus files | Why | Worker s |
|---|---|---|---:|
| `src/atom_iterate.jl` | `atom_iterate_grids`, `real_iterates`, `block_step`, `block_step_grids`, `norms`, `sparsegroup`, `infinite_group_weights` | constructors and products; block steps over groups (activation order, staging, pre-write errors, aliased trial points); the group norms and sparse group on atoms; active-group visits with infinite weights | 165 |
| `src/sparse_iterate.jl` (both stored iterates and their shared code) | `sparse_iterate`, `sparse_iterate_basics_grids`, `real_iterates`, `block_step`, `block_step_grids`, `sparse_value_cost`, `conformance/failures_storage`, `symmetric_mul`, `symmetric_argument` | the complete iterate tests; products and dispatch; block steps; no dense pass in value oracles; storage after failures; the products of the symmetric owner, the symmetric part read by `prox!`/`gradient!` | 532 |
| `src/symmetric_parameters.jl` | `symmetric_parameters_paths`, `_trait`, `_writes`, `_merge`, `_extensions` | the check paths, the trait, writes, merged and extension parameters; the catalogs (`_centers`, `_inventory`, `_inventory_walk`) stay in full | 133 |

Left out on purpose and still in thorough or full: `symmetric_groups` (group-norm writes into a
symmetric iterate, 118 s), `allocations` (the full catalog; `allocations_basics` in quick has
the iterate cases), `gradient_alias` (function outputs), the spectral, constrained and
conformance families. `src/sparse_iterate.jl` holds both stored iterates and the code they
share, so its group is the union: the file name cannot tell a narrow `SparseIterate` fix from a
symmetric change. A new quick file, `iterate_invariants` (94 checks, about 3 s), runs one
sequence on all four iterate kinds: the active list, reverse map and values agree and inactive
coordinates are zero; zero writes neither activate nor deactivate; `activate!` and `dropzeros!`;
independent copies; `copyto!` replaces the whole state; failed conversions leave the destination
unchanged; one symmetric coordinate per pair.

**Additive tier lists.** A change to `test/tiers.toml` needed full. The plan now compares the
parsed lists at the base and the head (`TestTiers.manifest_additions`): when the head only adds
names of test files that the PR adds (absent at the base, present at the head) and nothing
outside `[tiers]` changed, the lists decide quick and the new files run as changed test files.
A removed, moved or renamed file, an added name that is not a new file, or a change to the
helpers, the environments or `[not_tests.run_through]` (the test files that run each helper)
still needs full; a listed name without a file stops the plan. Order within the tier lists and
comments alone change nothing (quick); reordering a `helpers` or `run_through` array is a change
outside `[tiers]` and needs full.

**Measured on GitHub** (scenario PRs #477–#483 against `ci/iterate-scenarios`, this branch at
5bd6d95a; each changes a comment, so the time is the tests' alone; end to end from the run's
creation to `PR merge gate`):

| Scenario | PR (run) | Plan | Files | Test jobs (lts / Julia 1) | End to end |
|---|---|---|---:|---|---:|
| I1: narrow `AtomIterate` change | #478 (41463898074) | quick + atom group | 43 | 5.2 / 5.9 min | **6.7 min** |
| I2: narrow `sparse_iterate.jl` change | #479 (41463913825) | quick + sparse/symmetric group | 48 | 7.1 / 6.9 min | **8.0 min** |
| I3: narrow `symmetric_parameters.jl` change | #480 (41463926290) | quick + parameter group | 43 | 5.0 / 3.8 min | **5.8 min** |
| I4: `sparse_iterate.jl` and `atom_iterate.jl`, no label | #477 (41463854961) | standard + both groups | 76 | 12.5 / 7.8 min | **13.8 min** |
| I5: `sparse_iterate.jl` with `test:thorough` | #483 (41464422896) | thorough + the focus files not in thorough | 124 | 21.1 / 22.2 min | **23.3 min** |
| I6: a new test file and its `tiers.toml` line | #481 (41464205413) | quick + the new file | 39 | 3.5 / 3.8 min | **4.7 min** |
| I7: `utils` moved from quick to standard | #482 (41464229437) | full: "`utils` moves from `quick` to `standard`" | — | cancelled after the plan | — |

Before this change I1–I4 ran thorough: 23.3 min for I5 in the same conditions, 16.7–26.4 min
in the first round. The
scenarios ran up to six at a time beside a full run, on AMD EPYC 7763 runners (I6 on an Intel
Xeon 8573C). The slowest files of I2 and I4 on Julia lts were `sparse_iterate` (172–174 s),
`symmetric_argument` (117–122 s) and `block_step` (106–111 s);
if the narrow groups grow past the target, `sparse_iterate.jl`'s two testsets (sparse,
symmetric) can be split into separate files without losing a test.

`test/tier_selection.jl` checks the policy (the table of CONTRIBUTING, label precedence, that
the label never lowers another path or changes the focus files, order independence, that every
focus file is in full and only the iterate rules have a `narrow` tier) and the additive rule
(additions, order and comments, a removed, moved or renamed file, an unknown added name, helper,
environment and `run_through` changes, unreadable TOML, and the plan with and without
additions). It also checks the problems of `[not_tests.run_through]` and that the table agrees
with the sources both ways, on the real catalog and on synthetic sources: each recorded test names
the helper's file, directly or through one other helper, and each test that names a helper's file
is recorded for it, except for the helpers that the runner loads for every test file (`testutils`,
`conformance/common`). The planner
was also run on synthetic commits: a new file and its line gave quick and the file; a moved file
gave full with the reason; a listed name without a file stopped the plan.

**The nightly plan.** While this policy was written, the planner's `recent_full_run`, which the
nightly run calls to skip a commit that a full run already passed that day, was deleted by
accident (d1b9a268) and restored (97b62c2d); nothing tested that path. `recent_full_run` now
takes the GitHub CLI as an argument and checks the answer itself (not a pull request,
concluded `success`, the same commit, created in the 24 hours before the plan, test jobs of the
full tier), and `test/tier_selection.jl` runs the schedule path of the planner with a fake CLI:
no earlier run, a qualifying nightly, push or manual run (skip), quick runs only, another
commit, 25 hours old or in the future, failed, cancelled or unfinished runs, a pull-request run,
an API error and malformed answers (all full). An API error or malformed answer used to fail
the plan job, and with it the nightly run; it now runs the full tier, with the reason in the
plan. Deleting or renaming the function, dropping the fallback or the 24-hour window each fails
the test. The times come from the CLI's jq as Unix time, so the planner no longer loads
`Dates`, which the test environment does not have.

## A test made independent of its worker

The ambiguity check for the GPU extension is in `gpu_ambiguities.jl`, which `gpu_suite_basics.jl`
includes at its end, so `gpu_smoke` and the physical CUDA run execute it (it was in
`gpu_suite.jl`), and which `gpu_extension_ambiguities.jl` runs in the quick tier. It calls
`Test.detect_ambiguities(ProximalBase)`, which also sees
methods that other test files add to ProximalBase functions in the same worker process. With
`symmetric_parameters_trait` (its `MyL1` adds an `lla_weight` method) earlier on the worker, the
check failed: on Discovery with Julia 1.13.3 and no test history (files in random order), and
locally by running the two files in one process. Which files share a worker depends on the
tier and on the timing history, so the check ignores methods defined in test modules. The
known package ambiguities are unchanged; the assertion now prints the offending pairs when it
fails, on every Julia version. When the tiers were introduced, this was the only assertion whose
input changed.

## Focused platform runs (#772)

`platform_focus` (quick) tests `scripts/platform_focus.jl` and the text of
`.github/workflows/platform-focused.yml`, the workflow that runs one test file on native Windows and
macOS (CONTRIBUTING.md, "Focused runs on Windows and macOS"). `scripts/test.jl <name>` runs every
test whose name starts with `<name>`; some declared names are a prefix of another, so the
workflow passes the name and `!<other>` for each longer name. The test feeds that selector for each
declared name to the runner's own `parse_args` and `filter_tests!` on the real catalog and requires
exactly that file, and `check_selectors` returns no error and no warning for it (the runner calls it
first, and its warnings reach the log). It also checks that each refusal of the plan job says why
(unknown name, undeclared prefix, directory, helper, environment, a form that is not a name, an over-long name, a
listed name without a file, a Julia version below the bound of `Project.toml`, an `os`, a SHA), that
the plan reports the detected dispatch-input validation failures together rather than stopping at the first bad field, that a hostile input cannot add a workflow command of its own, that
`check-log` accepts the header of one file and rejects two files, another file, none and no header,
and that the workflow has dispatch as its only trigger, reads inputs in its shell `run:` steps only
through `env:` (`run-name` and `concurrency` read them directly), uses the action versions of `CI.yml`, and reads exactly the outputs the script writes. The workflow text is
read with CRLF endings turned into LF, since the quick tier also runs on Windows in the nightly run.
The native runs themselves are not part of the test; the PR carries that evidence.

## Nightly triage and release validation (#773)

The nightly run, tags and release validation now check more than the verdict of each test
job, and four quick-tier test files cover the scripts and the workflow wiring behind that.
The nightly and release runs themselves can only be observed on GitHub, after the merge for the
scheduled run; these tests use logs cut from a real failed run (a macOS and a Windows job that
both failed `three_factor_product`) and an in-memory GitHub, never the network or a real issue.

- **`ci_result`** tests `scripts/ci_result.jl`, the reader of a test job's log. It extracts the
  files the runner announced, what became of each (passed, failed, crashed, started and never
  finished) and every failure with its file, test set, kind and a signature that stays the same from
  night to night (digits and addresses replaced or removed, and absolute POSIX paths without
  whitespace and with at least two components and Windows drive paths without whitespace removed; a
  path with a space keeps the part after the space and a single-component path such as `/artifact`
  stays, so a different checkout path can change the signature in those cases), from the real macOS and Windows logs in each
  shape the log comes in (raw, colored, the API's timestamps, `gh run view --log`, Windows paths
  and carriage returns). It also covers an exception, a test that threw, a crash, a bounded
  excerpt and a log that is not a test log (never silence: it is reported as unrecognized). The
  inventory check compares the header's files with the tier's files and the header's count, and
  the command line step the workflow runs ("Check the test inventory") is run on logs that pass
  and fail it: a job that ran fewer or other files must not count as a pass.
- **`ci_triage`** tests `scripts/ci_triage.jl`, the reporter behind the `alert`, `triage` and
  `stale` jobs, against an in-memory GitHub that records every call. The cases are those the
  issue asks for: a new failure on Windows next to a Linux failure of the same night (separate
  issues, with different severity), a repeat (one comment per run, a changed signature said so),
  recovery (resolved when every Julia alias recorded for the issue has a job that succeeded or whose
  recognized log lists the file as passed, so a later failed step does not block it; a missing,
  cancelled or unreadable job blocks it for its alias), closed issues (reopened, except one closed as
  not planned; flaky never closed), jobs that failed without a test failure (infrastructure), API
  outages (a failed mandatory read stops the run before any write; for the alert only the lines per
  failed file are dropped) and failed writes (reported with exit status 1, a state change only after
  its comment exists, repaired by the next run), the alert of a failed push to
  master, a stale nightly, a story of several nights, the dry run and the rehearsal labels, text
  safety (mentions, fences, size cap), the adapter against a fake `gh` and the `[triage]` table
  (every deferred pattern must match a test file of `test/tiers.toml`, so a rename cannot leave a
  stale entry).
- **`release_record`** tests `scripts/release_record.jl`. The record is a pure function of its
  data, and the test pins what makes it say PASSED: the plan, the tests, the verdict and the gate
  all succeeded, the run is a manual run of the exact commit (the full lowercase SHA that equals the
  run's, not a short, uppercase, empty or different one), the four required test jobs each
  succeeded and the jobs and artifacts could be listed. Each other case is FAILED, with the reason;
  an unreadable job or artifact listing is a FAILED record, never a PASSED one. Julia pre is listed as
  advisory and a failure of it does not change the heading. The text from the inputs is reduced to
  a safe set of characters (no markup, mention or reference injection; plain URLs are kept). `main`
  is run with a fake `gh` and an environment dictionary: it writes the file and the job summary,
  exits 0 for a FAILED record as for a PASSED one (the verdict decides the run's conclusion; the
  record is its evidence; a malformed `expected_sha`, a non-empty but malformed `GITHUB_SHA` and an
  unrecognized result such as `PLAN=bogus` are FAILED records), 1 when it cannot write and 2 for a
  missing or empty `GITHUB_SHA` or result variable or a malformed repository, run id, attempt or
  server URL. A dispatch that the planner rejected has no
  release-record job and so no record.
- **`ci_workflow`** reads `.github/workflows/CI.yml` as text, cutting out each job by its key (no YAML
  library), and pins the wiring that no other test sees: the old `report` job is gone, only
  `alert`, `triage` and `stale` have `issues: write` (and no job of the test code does), the `if` of
  each reporting job and of the release record (none runs for a pull request, and the reporters not
  for a manual run; the triage job uses `!cancelled()`, so a cancelled run stops it), separate concurrency groups (a running job of these groups is not cancelled by a newer run; a pending one can be replaced), no `${{ ... }}` expression in
  any `run:` of these jobs (so no event field or input reaches a script line), the check names
  (the PR names appear only in expressions guarded by `pull_request`, and the nightly's names are
  not the required ones), the step "Run package tests" with `id: tests` equal to the reporter's
  `TEST_STEP` and the inventory step after it with its condition, the Codecov step that fails the
  nightly and a release run, the `release-record` artifact kept 90 days, and the files that the
  release record compares with the default branch (every `scripts/ci_*.jl`, the gate, the test
  runner and the test selection files). The checks read the file without its full-line comments, and
  a test shows that the Codecov line, the inventory condition and the retention each fail when
  only commented out.

New CI behavior that these tests pin: the nightly, tag and release test jobs fail unless their log
announced exactly the files of the tier (the step "Check the test inventory"); the verdict of those
runs also needs the `conformance-reports-...` artifact of each required job (`test/ci_verdict.jl`); a
lost Codecov upload fails the nightly and a release run (the nightly's reporter files it as an
infrastructure failure; a release run has no reporter, so it is a red run and a red record); and a
release run writes the record.

What the tests do not show: a pull request run has the inventory check off, uploads no coverage,
runs no reporter and writes no release record, so the PR's own run exercises none of this. The
tests use fixtures and read the workflow text. The live evidence is separate and outstanding until
it has run: a release run of the exact SHA (and one with a wrong SHA, which fails in the plan) and
the first scheduled triage after the merge.

## Per-file times

From the full run 41364805347 (Julia lts on an AMD EPYC 9V74, Windows on an EPYC 9V45). A file's
time includes compiling what earlier files on its worker had not compiled, so it depends on
the order: `gpu_smoke` takes 9 s on Windows because `gpu` had already run on that worker. The
tiers shown are those of that run; `gpu_smoke` and `conformance/gpu_arrays` have been in the full tier
since #771.
Files added after that run have no time in the table: `spectral_fused_values` (thorough) checks
that the value `value_and_prox!` returns for the counting functions of `SingularValue` and
`Spectral`, and for `IndBallRank`, is the standalone value at the matrix it stored.

<details><summary>Every test file, its tier and its time</summary>

| File | Tier | Julia lts, Ubuntu (s) | Julia 1, Windows (s) |
|---|---|---:|---:|
| `gpu_smoke` | quick (full since #771) | 51 | 9 |
| `conjugate_basics` | quick | 28 | 26 |
| `losses` | quick | 23 | 27 |
| `wrapper_buffers` | quick | 20 | 14 |
| `atom_iterate` | quick | 20 | 21 |
| `spectral` | quick | 19 | 14 |
| `constrained_basics` | quick | 19 | 18 |
| `aqua` | quick | 12 | 14 |
| `spectralfunction` | quick | 12 | 10 |
| `sparse_iterate_basics` | quick | 11 | 10 |
| `sets` | quick | 10 | 11 |
| `interface` | quick | 10 | 8 |
| `symmetric_products` | quick | 10 | 9 |
| `scaled_overflow` | quick | 10 | 7 |
| `allocations_basics` | quick | 10 | 3 |
| `entropy` | quick | 9 | 10 |
| `intersection` | quick | 8 | 9 |
| `utils` | quick | 8 | 5 |
| `nested_box_values` | quick | 8 | 7 |
| `deprecated` | quick | 6 | 6 |
| `matrixview` | quick | 6 | 4 |
| `active_handle` | quick | 5 | 2 |
| `lla` | quick | 5 | 4 |
| `caches` | quick | 5 | 5 |
| `block_step_failures` | quick | 4 | 8 |
| `iterate_queries` | quick | 3 | 3 |
| `spectral_mask` | quick | 3 | 4 |
| `tier_selection` | quick | 2 | 2 |
| `lapack` | quick | 2 | 2 |
| `symmetric_parameters` | quick | 1 | 4 |
| `conformance/tolerances` | quick | 1 | 1 |
| `readme` | quick | 1 | 1 |
| `model_precision` | quick | 1 | 1 |
| `spectral_box_tolerance` | quick | 1 | 2 |
| `docstrings` | quick | 0 | 0 |
| `compat` | quick | 0 | 0 |
| `smooth` | standard | 83 | 75 |
| `separable` | standard | 53 | 51 |
| `nonconvex` | standard | 50 | 46 |
| `distance` | standard | 49 | 40 |
| `constrained_nonconvex` | standard | 38 | 32 |
| `examples` | standard | 32 | 30 |
| `second_order` | standard | 31 | 27 |
| `ballconstrained` | standard | 29 | 28 |
| `norms` | standard | 28 | 30 |
| `singular` | standard | 28 | 26 |
| `sqrhinge` | standard | 24 | 21 |
| `sumlargest` | standard | 24 | 23 |
| `binary` | standard | 22 | 21 |
| `cubenorm` | standard | 22 | 18 |
| `minimum` | standard | 21 | 21 |
| `ordered` | standard | 21 | 20 |
| `composition` | standard | 20 | 21 |
| `metric_steps` | standard | 20 | 18 |
| `soc` | standard | 18 | 15 |
| `affine` | standard | 18 | 16 |
| `envelope` | standard | 18 | 16 |
| `infinite_weights` | standard | 18 | 15 |
| `stiefel` | standard | 18 | 13 |
| `matrixsets` | standard | 17 | 12 |
| `bernoulli` | standard | 15 | 15 |
| `sparse_value_cost` | standard | 14 | 12 |
| `sparsegroup` | standard | 11 | 19 |
| `sparse_iterate` | thorough | 128 | 64 |
| `symmetric_groups` | thorough | 118 | 104 |
| `metric` | thorough | 114 | 105 |
| `blockchecks` | thorough | 104 | 71 |
| `constrained` | thorough | 100 | 107 |
| `symmetric_argument` | thorough | 79 | 44 |
| `block_step` | thorough | 64 | 73 |
| `spectral_nonfinite_values` | thorough | 61 | 60 |
| `calculus` | thorough | 57 | 62 |
| `symmetric_mul` | thorough | 47 | 19 |
| `spectral_overflow_wrappers` | thorough | 42 | 30 |
| `spectral_overflow` | thorough | 40 | 41 |
| `graph` | thorough | 40 | 28 |
| `symmetric_parameters_paths` | thorough | 39 | 26 |
| `gradient_alias` | thorough | 39 | 33 |
| `symmetric_parameters_trait` | thorough | 36 | 25 |
| `extreme_normals` | thorough | 35 | 27 |
| `fibers` | thorough | 29 | 22 |
| `conformance/spectral` | thorough | 29 | 19 |
| `expcone` | thorough | 28 | 27 |
| `symmetric_parameters_writes` | thorough | 28 | 19 |
| `polyhedron` | thorough | 28 | 26 |
| `box_bounds_overflow` | thorough | 28 | 23 |
| `testing` | thorough | 28 | 31 |
| `infinite_group_weights` | thorough | 27 | 37 |
| `value_overflow_fused` | thorough | 25 | 25 |
| `converted_weights` | thorough | 25 | 25 |
| `spectral_overflow_prox` | thorough | 24 | 13 |
| `weighted_overflow` | thorough | 23 | 19 |
| `partial` | thorough | 23 | 16 |
| `precision` | thorough | 23 | 35 |
| `rotatedsoc` | thorough | 21 | 18 |
| `conformance/element_types` | thorough | 21 | 14 |
| `spectral_domain` | thorough | 19 | 12 |
| `conformance/harness` | thorough | 18 | 18 |
| `conformance/gpu_arrays` | thorough (full since #771) | 17 | 44 |
| `nonfinite_inputs` | thorough | 17 | 14 |
| `spectral_tolerance` | thorough | 16 | 20 |
| `symmetric_parameters_merge` | thorough | 15 | 13 |
| `symmetric_parameters_extensions` | thorough | 15 | 11 |
| `conformance/linear_sets` | thorough | 15 | 12 |
| `conformance/data_functions_2` | thorough | 15 | 10 |
| `large_inputs` | thorough | 13 | 15 |
| `conjugate_steps` | thorough | 11 | 9 |
| `conformance/special_values` | thorough | 10 | 12 |
| `value_overflow_sqrnorm` | thorough | 9 | 11 |
| `conformance/kernels` | thorough | 9 | 11 |
| `conformance/data_functions` | thorough | 9 | 8 |
| `conformance/steps` | thorough | 9 | 8 |
| `conformance/failures_storage` | thorough | 8 | 10 |
| `conformance/wrappers` | thorough | 6 | 8 |
| `conformance/linear_sets_2` | thorough | 5 | 5 |
| `conformance/second_order` | thorough | 4 | 4 |
| `logdet_overflow` | thorough | 2 | 4 |
| `gpu` | full | 390 | 355 |
| `conjugate_projections` | full | 359 | 299 |
| `precision_mixed` | full | 318 | 306 |
| `blockchecks_wrappers` | full | 147 | 100 |
| `blockchecks_steps` | full | 129 | 110 |
| `blockchecks_values` | full | 85 | 88 |
| `hyperslab` | full | 83 | 89 |
| `weight_steps` | full | 82 | 48 |
| `stepchecks` | full | 68 | 88 |
| `allocations` | full | 62 | 52 |
| `symmetric_parameters_centers` | full | 61 | 43 |
| `image` | full | 54 | 56 |
| `distl2` | full | 48 | 50 |
| `real_wrappers` | full | 44 | 50 |
| `real_functions` | full | 39 | 32 |
| `precompose_array` | full | 37 | 44 |
| `real_iterates` | full | 35 | 27 |
| `separable_blocks` | full | 29 | 33 |
| `symmetric_parameters_inventory` | full | 28 | 27 |
| `blockchecks_workspace` | full | 20 | 26 |
| `symmetric_parameters_inventory_walk` | full | 1 | 1 |

</details>
