# A Taylor start for Wright omega

For `-2 ≤ u < 2`, start the existing Halley iteration with a quadratic Taylor
expansion at zero. If `a=ω(0)`, implicit differentiation of `ω+log(ω)=u` gives
`ω′=ω/(1+ω)` and `ω″=ω/(1+ω)^3`. The start is
`a + u*(a/(1+a) + u*a/(2*(1+a)^3))`.

This removes an `exp` and two `log1p` evaluations from initialization. The
Halley iteration, convergence test, negative-u series and large-u asymptotic
start remain unchanged. Tests cover a dense Float32/Float64 central grid and
both branch boundaries, alongside the existing Lambert-W reference, extreme
arguments, scalar objective, layout and allocation tests.

## Source-verified ten-run comparison

Baseline: `0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a`.
Measured implementation/harness: `f69cbe5733314c0cd80ab194c1a9fad377425dc1`.
There are 72 cases per variant/version: Float32/Float64, lengths 256 and
100,000, entropy/KL, ordinary inputs, inputs concentrated in the modified
branch, outside-branch controls, prox/value-prox and aliasing.

Ten fresh before/after process pairs run per Julia version. Within each pair,
both variants use the same CPU/node and seeded input; first/second order
alternates. All **2,880 case/run measurements** and raw-record checks passed.
No unexpected warmed Julia heap allocation was observed.

| Float64 prox, n=100,000 | Julia | Before / after mean | Paired speedup [95% CI] |
|---|---|---:|---:|
| NegEntropy, ordinary inputs | 1.13.2 | 9.589 / 8.560 ms | 1.121 [1.105, 1.137] |
| NegEntropy, central branch | 1.13.2 | 13.084 / 8.203 ms | 1.596 [1.576, 1.616] |
| KLDivergence, ordinary inputs | 1.13.2 | 10.077 / 9.023 ms | 1.117 [1.110, 1.125] |
| NegEntropy, ordinary inputs | 1.10.12 | 10.106 / 9.018 ms | 1.121 [1.111, 1.130] |
| NegEntropy, central branch | 1.10.12 | 13.453 / 8.206 ms | 1.639 [1.621, 1.657] |

All 48 modified-branch/ordinary case groups per version have paired intervals
above one. No group has an interval entirely below one. Outside-branch controls
remain in the [full comparison](comparison.csv) and [report](REPORT.md). These
per-case nominal intervals are not corrected for multiple comparisons or
correlated machine noise; they do not establish a universal speedup.

Times are arithmetic means of ten run means. Between-run SD, sample counts,
nominal t intervals and allocation observations are in `before/` and `after/`.
Speedup intervals use paired log ratios. Each process collects up to 40 samples
or 0.05 seconds and ten separate warmed allocation observations per case.
Compilation, validation and preparation are excluded from latency. CPU profiles
are separate sampling diagnostics. These are CPU measurements and Julia heap
allocation observations; native allocation and GPU execution are not measured.

## Corrected environment and retained invalid attempt

The initial job, **13422635**, ran a valid ten-pair LTS comparison but loaded
the baseline on Julia 1.13. A copied Project.toml `[sources]` entry overrode the
candidate Manifest path when adding the test reference dependency. Its initial
Julia 1.13 timings are **invalid as a candidate comparison** and are excluded.

Runtime source audit **13422706** detected the wrong path, verified every other
candidate environment, then used `Pkg.develop(path=...)` to repair a separate
Julia 1.13 environment. The complete entropy test file and 72-case canary were
rerun against the actual candidate and passed. Job **13422729** reran only the
ten invalid Julia 1.13 pairs, asserting the loaded package path in every
process. It completed in **1m55s**. The ten verified LTS pairs were reused
unchanged. No valid run was discarded or replaced by a faster draw.

The [source audit](source-audit.txt), original invalid data/environment and
corrected worker logs are retained. The portable harness now records its loaded
package root and rejects a conflicting `PB_EXPECTED_PACKAGE_ROOT` before case
generation. Recreate projects with `Pkg.develop`; do not update a Manifest path
while leaving a conflicting Project `[sources]` entry.

Use `benchmark/changes/entropy.jl` and the [campaign protocol](../../../campaign/README.md)
for a fresh comparison. `launch/` records the correction/reuse recipe for this
run and its Discovery paths; adapt paths/account before reuse. Workers use one
Julia and one BLAS thread, at most four workers on distinct AMD EPYC 7513 nodes,
with a peak request of four CPUs/16 GiB.

The canonical registry is
`/Users/mladen.kolar/projects/juliaPkg/.slurm/runs/20261025-pb-optimizations.json`.
Raw observations, profiles, diagnostics, snapshots and launchers are retained
in its corresponding local run directory and
`/home1/mkolar/juliaPkg/runs/20261025-pb-optimizations` on Discovery.
