# SLOPE sorting and least-squares backend studies

These ten-run comparisons support keeping the current package defaults.
`benchmark/changes/remaining.jl` contains reusable prototypes; it changes no
library dispatch or public API. Sorting measurements cover the ordering kernel,
not a complete SLOPE prox. Least-squares measurements compare the current thin
SVD with a Gram/Cholesky prototype for vector responses.

## Design and validation

Discovery job **13422745** completed in **2m57s**. Package source:
`0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a`; measured harness:
`42d6119b0c3f4ad3dc63b4574b3f6466516458b5`. Ten fresh processes ran per Julia
version (1.13.2 and 1.10.12), using common seeded inputs across alternatives
and shuffled case order within a process. There are 102 cases per process and
**2,040 valid case/run measurements**. Each worker asserts its loaded package
root. All numerical and raw-record checks passed.

Times below are arithmetic means of ten run means. Paired speedups use geometric
within-process ratios, with nominal t intervals on log ratios. Full per-case
statistics and allocation observations are in [summary.csv](summary.csv),
[contrasts.csv](contrasts.csv) and [STUDY.md](STUDY.md). No runs or outliers are
discarded. Intervals are not corrected for inspecting multiple cases.

Workers collect up to 40 timing samples or 0.05 seconds per case, plus ten
separate warmed Julia heap allocation observations. Preparation, validation
and compilation are excluded from latency; constructor cases explicitly include
construction. Native allocation and physical GPU execution are not measured.

## SLOPE ordering

Float64, n=100,000, Julia 1.13.2:

| Input | Current default | QuickSort | MergeSort |
|---|---:|---:|---:|
| Ordinary magnitudes | 4.836 ms | 6.760 ms | 8.291 ms |
| Already ordered | 0.222 ms | 1.134 ms | 1.209 ms |
| Many ties | 3.199 ms | 12.704 ms | 2.815 ms |

MergeSort improves the large tied-input case by 1.138× [1.099, 1.177], but
allocates about 400 kB per call despite supplying the tested scratch buffer.
QuickSort is allocation free and wins on the 4,096-entry ordinary cases
(Float64: 1.260× [1.204, 1.317] on current Julia), but loses on the larger
ordinary, tied and already ordered inputs. LTS shows similar tradeoffs. These
results do not justify a global algorithm replacement or a size-only switch.
An allocation-free stable ordering method remains a candidate for future work.

## Least squares

The prototype forms the smaller Gram matrix and factors its shifted system.
It uses the dual formula for wide matrices and reuses a factor for an unchanged
step. Numerical checks compare against the package prox on the tested seeded
well-conditioned designs; rank-deficient and ill-conditioned designs, matrix
responses and broader API behavior are outside this prototype's coverage.

Float64 design 1,024×256, Julia 1.13.2:

| Workload | Current thin SVD | Cholesky prototype | Paired speedup [95% CI] |
|---|---:|---:|---:|
| Construction + first prox | 31.245 ms | 2.117 ms | 14.793 [13.707, 15.964] |
| Warmed, fixed step | 14.703 μs | 38.459 μs | 0.381 [0.360, 0.404] |
| Four changing steps | 59.858 μs | 1,096.301 μs | 0.054 [0.050, 0.058] |

The construction case allocates about 1.06 MB in the Cholesky prototype.
Both prototypes have zero observed warmed Julia heap allocation. Cholesky
removes much of the first-use cost but loses on repeated calls, especially
when the step changes. The smaller and wide shapes, both floating-point types
and LTS results are retained in the full table. A future opt-in cold-use backend
would need broader numerical and API coverage; this study does not add it.

## Recreate

Use `benchmark/changes/remaining.jl` with the [campaign protocol](../../../campaign/README.md).
Create the test project with `Pkg.develop(path=...)` and set
`PB_EXPECTED_PACKAGE_ROOT`. `launch/` preserves the exact Discovery commands,
account and paths used here; adapt those for another run. The job used at most
four CPUs/16 GiB, one worker on each AMD EPYC 7513 node and one Julia/BLAS thread.

Raw data, profiles, source snapshots and logs are retained in
`/home1/mkolar/juliaPkg/runs/20261025-pb-optimizations` and its local counterpart
under `.slurm/runs/20261025-pb-optimizations/results`. The canonical registry is
`/Users/mladen.kolar/projects/juliaPkg/.slurm/runs/20261025-pb-optimizations.json`.
