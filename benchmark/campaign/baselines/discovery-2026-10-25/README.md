# Discovery CPU baseline — 25 October 2026

The campaign measured 1,054 cases on each of Julia 1.13.2 and 1.10.12, with ten
fresh processes per version. All 21,080 case/run checks passed. The retained
records contain 811,426 timing samples and 210,800 allocation observations.
There were no missing repetitions, duplicate measurements, or unexpected warmed
Julia heap allocations. The package implementation was unchanged.

The primary statistic is the arithmetic mean of ten run means, accompanied by
between-run SD and a nominal Student-t 95% interval. Inputs vary by a stable
scenario key and repetition; case order is shuffled. Nothing is selected by
fastest run. See the [protocol](../../README.md), [full summary](summary.csv),
[automatic report](REPORT.md), [phase contrasts](CONTRASTS.md), and
[raw-record validation](validation.json).

## What the measurements establish

Each version has 1,038 warmed cases with an allocation-free expectation, plus
16 construction/first-prox cases. All 100 separate warmed allocation calls per
case/version returned zero Julia heap bytes. This says nothing about native
library allocations or allocations when dimensions, ranks or cache states change.

Variation is material. The median between-run coefficient of variation was
8.3% on Julia 1.13.2 and 4.4% on 1.10.12. Respectively 252 and 68 case/version
groups exceeded 15% CV. These are separate within-version summaries; they do not
establish a Julia-version regression. Input-dependent work, shared-node load,
and very short calls all contribute. For example, a random 1%-sparse vector of
length 256 has few stored entries, whose count changes between draws. The raw
metrics retain this work count. Such variation must not be interpreted as an
implementation defect or removed by discarding slow draws.

Representative Float64 timings on Julia 1.13.2 follow. Each row uses ten runs.
Times are mean ± between-run SD, rather than best-case latency.

| Operation | Shape | Mean ± SD |
|---|---|---:|
| `NegEntropy` prox | 100,000 | 9.797 ± 0.484 ms |
| `KLDivergence` prox | 100,000 | 10.188 ± 0.509 ms |
| `NormLq(q=1/2)` prox | 100,000 | 7.665 ± 0.362 ms |
| SLOPE prox | 100,000 | 6.759 ± 1.079 ms |

The representative CPU traces show transcendental arithmetic in the scalar
nonconvex/entropy kernels and LAPACK work in the spectral kernels. These traces
identify work to inspect; they do not certify a particular replacement as faster.
Preparation callbacks appear in CPU profiles and are excluded from latency and
allocation measurements.

## Candidates for future improvements

1. **Scalar arithmetic in entropy/KL and `NormLq`.** Their costs are substantial
   and reasonably stable on these inputs. Investigate invariant arithmetic,
   special cases and the scalar iteration before changing vector storage.
   Preserve behavior near thresholds and on extreme arguments. The present
   `NormLq` timing covers `q=1/2`, not its generic Newton branch.
2. **The small-matrix partial-SVD cutoff.** At 128×64, Float64 `NuclearNorm`
   with `partial=true` costs 0.706 ms versus 0.689 ms for the full path. At
   512×256 it costs 1.520 ms versus 20.453 ms; the paired full/partial ratio
   is 13.46 with interval [13.11, 13.82]. The small case has no advantage.
   Source inspection explains the fallback: a learned rank of four plus three
   guard pairs requests a 17-column block; the quarter-dimension rule permits
   only 16 columns. `IndBallRank(4)` requests a smaller block and benefits on
   the same input family. Benchmark shape/rank/gap sweeps before adjusting this
   conservative heuristic. This is a candidate, not a demonstrated bad cutoff.
3. **SLOPE's ordering work.** Its large-vector cost is visible, but its 16% CV
   on Julia 1.13.2 makes small speedup claims unreliable. Inspect sorting and
   order restoration, then confirm changes with interleaved paired runs.
4. **First-use least-squares cost.** For a 1024×256 Float64 design, construction
   plus first prox costs 32.998 ms, versus 14.91 µs for a warmed prox. The CPU
   traces show decomposition work. Solvers should retain the function/cache.
   A different first-use backend could help one-shot workloads, but these
   measurements do not compare such a backend or justify changing the default.

The existing partial path has large benefits on the well-separated rank-four
512-dimensional cases. These results support retaining it. They do not support
a blanket partial/full speedup claim across ranks and spectra.

## Provenance and recreation

- Package source: `2eab0955b4e33121582f2abbb73a7f87803b1631`.
- Frozen measurement harness: `c47dd77c599baf57642ed81dac0d79aa3a508e6f`.
- Final aggregation, raw-record validation and analytic controls:
  `85de74dd196ea417b96cf347dfa647df579e023a`.
- Discovery jobs: production `13422265`, completed in 18m21s; analysis
  `13422278`, completed in 6s. Every worker exited zero.
- Hardware: AMD EPYC 7513, nodes b10-17 through b10-20. Four workers at most,
  one per node, one CPU/4 GiB each; one Julia and one BLAS thread. Nodes were
  shared with other users. See [hardware](metadata/hardware.txt) and the twenty
  per-process metadata receipts in `metadata/`.
- BenchmarkTools 1.8.0; up to 40 samples or 0.1 seconds per case/run, with actual
  counts retained. The slowest cases collected as few as three samples per run.
- Six preparation/diagnostic attempts, including one canceled initial
  production attempt, used 18m00s before the corrected baseline. Total allocated
  job running time was 36m27s, or 36m57s conservatively including a canceled
  step's 30-second cleanup. Peak request was 4 CPUs/16 GiB. This stayed within
  the standing 8-CPU/16-GiB/two-hour task allowance.

The run directory is
`/home1/mkolar/juliaPkg/runs/20261025-pb-performance-baseline` on Discovery.
The local registry is
`/Users/mladen.kolar/projects/juliaPkg/.slurm/runs/20261025-pb-performance-baseline.json`.
The corresponding local run directory retains source snapshots, launchers,
version-specific manifests, all raw samples, profiles, worker logs and prior
diagnostics. `full-v2` is the corrected baseline; `full-v1` and the canaries are
excluded from its statistics. The compressed transfer files are approximately
7.5 MiB for production and 160 KiB for the analysis summary.

The launcher recipes are included in `launch/`. Their account and absolute
paths describe this run and must be adapted for another checkout. For a new
implementation, keep the measured `cases.jl`/`run.jl` instrumentation fixed,
develop the new package into separate version-specific projects, and rerun
repetitions 1–10. Follow the protocol's Slurm instructions and compare retained
raw directories with `compare.py`. `contrasts.py RESULTS SUMMARY_DIRECTORY`
recreates the phase contrasts after `summarize.py` has populated the summary.

The analytic controls checked a known mean, SD, t interval and paired ratio,
then checked rejection of missing/wrong repetitions, corrupted raw observations
and mismatched settings. Raw-record validation reconciled every reported mean
and extreme with its timing/allocation observations. Package checks include
dense references, returned-value agreement, directional gradients/Hessians,
an analytic Huber Hessian, partial/full agreement, a Dykstra/simplex comparison,
and residual/objective checks for a five-sweep coordinate trace. They complement
the unit tests; they are not independent proofs of every function.

The earlier diagnostics exposed two harness issues: a finite-difference Huber
check crossing curvature boundaries, and dynamic datatype conversions in timed
closures that boxed on LTS. Both were corrected and checked before the fresh
production run. Their failed/adverse records remain separate. They are not
package defects, and no valid corrected-production run was excluded.

GPU execution, native allocator instrumentation, compilation/load latency,
exhaustive parameter regimes, long solver trajectories, and large/high-rank or
nearly degenerate partial spectra remain outside this baseline. Prioritize such
extensions when a proposed implementation change depends on them.
