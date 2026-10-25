# Partial-SVD cutoff reuse

A small `NuclearNorm(partial=true)` call could accept its partial result, then
switch to the full SVD on the next nearby call. The next allocation target
includes guard pairs and oversampling and can cross the quarter-dimension
cutoff even though the existing block has enough room for the learned rank and
its guards.

The change reuses that block only when the preceding partial result passed its
acceptance checks, the block is warm, dimensions match, and the learned rank
plus guards fits. The actual block still obeys the quarter-dimension limit.
Residual tests, guard tests, iteration limits and ordinary allocation/growth
policy are unchanged. Full fallback clears the acceptance flag, avoiding
repeated retries after a failed partial solve.

## Ten-run comparison

Discovery job **13422553** completed in **7m43s**. The package baseline is
`0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a`; the measured implementation is
`50ce5b52f6d3b58f07927bfa4500159c120a8d2f`. The common harness is
`5e92a5d6eac823b627afa2a3b06b5cb68f59eaf0`.

Each Julia version has ten fresh before/after process pairs. A pair runs on the
same CPU and node, alternating which implementation runs first. Inputs are
keyed by scenario and repetition; case order is shuffled. Each case collects
up to 40 samples or 0.05 seconds, plus ten separate allocation observations per
process. The mean is the arithmetic mean of ten run means. Ratios are paired
geometric speedups with nominal t intervals on log ratios.

| Float64 nearby rank-four input | Julia | Before / after mean | Paired speedup [95% CI] |
|---|---|---:|---:|
| 128×64 | 1.13.2 | 696.9 / 185.7 µs | 3.76 [3.62, 3.92] |
| 64×128 | 1.13.2 | 765.2 / 180.2 µs | 4.25 [4.17, 4.33] |
| 128×64 | 1.10.12 | 664.6 / 178.0 µs | 3.73 [3.70, 3.77] |

All **14,800 case/run measurements** passed the numerical and raw-record
checks. There are 370 cases per variant/version: Float32/Float64, eight shapes,
ranks 2/4/6/8/12, nearby and random-restarted bases, clustered/high-rank spectra,
unrelated cached inputs, construction/first calls, and full-SVD controls. No
unexpected warmed Julia heap allocation was observed. This does not measure
native library allocation.

No case has both a paired slowdown above 10% and an interval excluding one. It has
two nominally slower groups on LTS and eight on Julia 1.13.2; those rows remain
in the [full comparison](comparison.csv) and [report](REPORT.md). The lowest
paired speedup is 0.932 [0.784, 1.108]. Its arithmetic mean is about 12% higher
after the change; the two summaries describe different aspects of variation.
Some intervals still allow larger slowdowns. Intervals
are per-case diagnostics without a multiple-comparison or correlated-machine-noise
correction. No blanket speedup across sizes, ranks or spectra is claimed.

See `before/` and `after/` for means, between-run SD, intervals, allocation
summaries, coverage and reconciliation with every raw observation. The focused
partial-decomposition test file, including small rectangular repeat calls,
high-rank fallback and warmed allocation checks, passed on both versions.

## Rejected first candidate

An initial patch removed guard padding from every predicted-rank allocation.
Its ten-run study, job **13422385**, passed numerical checks but roughly doubled
the Float64 192×128 clustered-rank-20 cost on both Julia versions. On 1.13.2
it changed 2.647 ms to 5.425 ms, with a paired ratio of 0.489 [0.478, 0.499].
That version is **rejected**, despite its gains on the motivating small case.
Its source/harness is `9fa780ae33b00dfd9c3cd86e1cd12eb712b97ced`; its full
comparison remains in `rejected-initial/`. The final study uses fresh pairs
and keeps the original allocation policy. Expanded canaries also motivated
the acceptance flag, so failed partial solves cannot cause repeated retries
at the cutoff.

## Recreation and retained evidence

Use `benchmark/changes/partial_svd.jl` with version-specific before/after
projects and the [campaign protocol](../../../campaign/README.md). The
runtime [source audit](source-audit.txt) verified the actual baseline and final
candidate package roots on both Julia versions. Use `Pkg.develop(path=...)`
and `PB_EXPECTED_PACKAGE_ROOT` for fresh runs; the portable harness now records
and can assert that root before generating cases. The
`launch/` files preserve Discovery settings for this run; adapt account/paths
before reuse. Workers use one Julia and one BLAS thread, and four workers on
distinct AMD EPYC 7513 nodes at most. Requested peak resources are 4 CPUs and
16 GiB. CPU profiles include preparation; latency and allocation observations
exclude it.

The local run registry is
`/Users/mladen.kolar/projects/juliaPkg/.slurm/runs/20261025-pb-optimizations.json`.
Its run directory retains raw samples, profiles, numerical diagnostics, source
snapshots and receipts. The corresponding Discovery directory is
`/home1/mkolar/juliaPkg/runs/20261025-pb-optimizations`.
`partial2-before`/`partial2-after` are the final comparison;
`partial-before`/`partial-after` are the rejected initial comparison.
