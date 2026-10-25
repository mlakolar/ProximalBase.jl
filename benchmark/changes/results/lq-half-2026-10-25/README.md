# Half-thresholding arithmetic

For `NormLq(λ, 1/2)`, the threshold is `3μ^(2/3)/2`, where `μ=γλwᵢ`.
Use `cbrt(μ)^2` there. In the trigonometric root, evaluate
`(μ/t)/sqrt(t) * (3sqrt(3)/4)` instead of multiplying μ by a reciprocal
fractional power. This avoids overflow-prone intermediate values and removes
general powers from the half-thresholding path. The final multiplication is
scaled by a factor of at most one, rather than forming `2t` first.

Other q values retain their formulas. New tests compare with independent
256-bit Newton roots across Float32/Float64 scales, including subnormal μ.
The existing nonconvex objective, layout, inference and allocation checks also
passed on both Julia versions.

## Ten-run evidence

Discovery job **13422634** completed in **4m29s**. Baseline:
`0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a`; implementation/harness:
`1132145cf8ac216562176640aca99d8eed49add6`. Later changes add benchmark
source-path guards and reporting; the measured library arithmetic is unchanged.

There are 108 cases per variant/version: q=1/2, q=2/3 and q=0.7; Float32 and
Float64; lengths 256 and 100,000; dense, weighted, diagonal-step and sparse
calls; prox, combined value/prox and aliasing. Ten fresh before/after process
pairs run per Julia version on the same CPU/node within a pair. Order
alternates, inputs use matching scenario/repetition keys, and case order is
shuffled. All 4,320 case/run measurements and raw-record checks passed. No
unexpected warmed Julia heap allocation was observed.

| Float64 q=1/2 prox, n=100,000 | Before / after mean | Paired speedup [95% CI] |
|---|---:|---:|
| Julia 1.13.2 | 7.693 / 3.866 ms | 1.991 [1.966, 2.016] |
| Julia 1.10.12 | 6.558 / 3.607 ms | 1.818 [1.806, 1.830] |

Every q=1/2 case has a paired interval above one. The q=2/3 LTS controls have
several roughly 3% slowdowns, despite retaining their formulas. These measured
code-generation/timing effects are retained, not dismissed as zero. There are
17 nominally slower case/version groups on LTS and one on Julia 1.13.2; none
has both a paired slowdown above 10% and an interval excluding one. Read the
[complete comparison](comparison.csv) and [report](REPORT.md), including the
unchanged-q controls, before applying the headline speedup to another workload.

Times are arithmetic means of ten run means. Between-run SD, nominal t
intervals, sample counts and byte observations are in `before/` and `after/`.
Paired ratios use geometric within-seed speedups and t intervals on log ratios.
There is no fastest-run selection or correction for inspecting many cases.
Allocation observations track Julia heap bytes, not native allocation.

Each process collects up to 40 samples or 0.05 seconds per case, plus ten
separate warmed allocation observations. Compilation, validation and preparation
are excluded from latency. Representative CPU profiles are retained separately.
GPU execution and exhaustive q/weight/scale performance coverage are outside
this CPU study.

Runtime audit **13422706** verified the actual baseline and candidate package
roots on both Julia versions; see [source receipts](source-audit.txt). It also
identified a separate entropy-environment setup error, which does not affect
this comparison. The portable harness now records and can assert its loaded
package root before case generation.

Use `benchmark/changes/lq_half.jl` and the [campaign protocol](../../../campaign/README.md)
for a fresh comparison. Create projects with `Pkg.develop(path=...)`, set
`PB_EXPECTED_PACKAGE_ROOT`, and use identical instrumentation for both variants.
`launch/` preserves this run's Discovery settings and absolute paths; adapt them
before reuse. Peak request was four CPUs/16 GiB, one worker per AMD EPYC 7513
node, with one Julia and one BLAS thread.

Raw observations, source snapshots, profiles and logs are retained in
`/home1/mkolar/juliaPkg/runs/20261025-pb-optimizations` and the corresponding
local `.slurm/runs/20261025-pb-optimizations/results` directory. The canonical
registry is `/Users/mladen.kolar/projects/juliaPkg/.slurm/runs/20261025-pb-optimizations.json`.
