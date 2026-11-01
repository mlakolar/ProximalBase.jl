# Test tiers: baseline, placement and measurements

This file records why the test suite is split into tiers, where each file sits and why, which
tests moved between files, and what the tiers cost on GitHub-hosted runners.
[CONTRIBUTING.md](../CONTRIBUTING.md) is the user guide; [`tiers.toml`](tiers.toml) is the
manifest that the runner and CI read.

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

**Compilation dominates.** ParallelTestRunner's `--verbose` output gives each file's compile
share on Julia 1.11 and later: 92–99% for nearly every file, locally and on CI (the Quick tier
on CI: 94% of 373 worker seconds). A test file's cost grows with the distinct method
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
  allocations on hot paths; and device arrays (JLArrays) with scalar indexing disallowed.
  The five most expensive files contributed representative cases, moved into new files
  (below), so that Quick keeps their main identities and regressions without their grids.
- **Standard** adds the files of most function families, wrappers, constraints, losses and
  the examples: what an ordinary implementation change can break.
- **Thorough** adds what shared or numerical changes can break: precision and overflow
  recovery, non-finite values, spectral domains and tolerances, the conformance suite,
  symmetric writes and parameters, steps and metrics, the wrapper and block checks, and the
  remaining iterate grids.
- **Full** adds the exhaustive catalogs: the rest of `gpu`, `conjugate_projections` and
  `allocations`, and the catalog files whose every case repeats a path an earlier tier
  already reaches.

Every file is listed once in `tiers.toml`; full runs all of them, so the full tier runs
exactly the files the old complete suite ran, plus the new files split out of them.
`test/tier_selection.jl` checks this, the cumulative inclusion, and the CI risk rules.

## Tests moved into quick-tier files

Each split moved whole testsets, or split one loop's parameter tuple into a representative
subset and its complement, so that the original file plus the new file run exactly the
assertions the original ran. No assertion, tolerance, input or expected error changed and
nothing was deleted. Assertion counts (from the runner) confirm it.

| New file (quick) | From (later tier) | What moved | Assertions: before = after + new |
|---|---|---|---|
| `sparse_iterate_basics` | `sparse_iterate` (thorough) | 17 whole testsets: construction, `dropzeros!`/nnz, `activate!`, active coordinates, converting writes, `fill!`/norm, copies and the conversion-failure regression (#416), stored zeros in `mul!` (Float64), the symmetric matrix (coordinates, mirrored writes, exact symmetry, `copyto!`), triple products (Float32, Float64), aliasing of views | 2897 = 2577 + 320 |
| `allocations_basics` | `allocations` (full) | Loop entries: block steps on a `SparseIterate` and with `NormL1`/`GroupNormL2`; `GroupNormL2` with distinct group layouts; `NormL1` on vectors, sparse, atom and symmetric iterates; `GroupNormL2` prox; `LeastSquaresLoss` gradients | 2910 = 2766 + 144 |
| `constrained_basics` | `constrained` (thorough) | `NormL1` and centered `SqrNormL2` on four boxes with the brute-force optimality check; coordinate steps on a box; `Scaled(NormL1)` wrappers; simplices, closed forms, the zero-sum lasso KKT check, errors, general normals with a box, Float32 cases; Float64 open-domain boxes | 4108 = 3526 + 581 (+1, see below) |
| `conjugate_basics` | `conjugate_projections` (full) | Seven direct conjugate families (clipping, check loss, orthant, l2 ball, group l2 ball, huge·tiny scale, shift) in Float64 and Float32; the Float64 Moreau identity with infinite weights; nested mixed sums and failure atomicity; output aliasing of parameters; biconjugates; Float64 l1-ball radius accuracy and group l1 balls | 265693 = 259835 + 5858 |
| `gpu_smoke` (with `gpu_prelude`, `gpu_suite_basics`) | `gpu` (full) | `NormL1` (plain and weighted) and the user bits type; `NormL2`, `IndSimplex`; `GroupNormL2` on host and device layouts; weighted norms at device overflow; least-squares and logistic losses with a design matrix; infinite weights; Moreau array steps | 21767 = 21469 + 298 |

`constrained`: one check in `ProximalBase.Testing.test_prox` runs only for random candidate
points inside the constraint set, so its count depends on the random draws (110–112 for the
moved block under seeds 1–12); the moved code changed which draws later testsets see.

Moving cases changes the global random stream for the testsets that follow them in
`sparse_iterate`, `constrained` and `allocations` (each test file starts from `Random.seed!(1)`).
Every affected testset passes; none depended on its position in the stream. In `gpu`, every
remaining testset starts in the same random state as before. `test/cuda/runtests.jl` now runs
`gpu_prelude`, `gpu_suite_basics` and then, after reseeding, `gpu_suite`, so the physical GPU
covers exactly what it covered before, with the same random inputs as `test/gpu.jl`.

## JLArrays and CUDA

`gpu_smoke` (quick) and `gpu` (full) run on JLArrays, GPUArrays' reference backend on the CPU,
with scalar indexing disallowed: they catch code that indexes device arrays element by element
or does not adapt its parameters. They do not test a physical device. `test/cuda` runs the same
shared cases on an NVIDIA GPU by hand (see its README); no tier and no CI job runs it, and
`test/oracles` (JuMP and conic solvers) is likewise run by hand.
