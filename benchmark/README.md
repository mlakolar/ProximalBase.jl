# Benchmarks

This suite tracks the runtime and Julia heap allocations of ProximalBase
operations. Keep reusable cases and measurement tools here. Save generated
results with the pull request or release that uses them.

## Setup and selections

From the package root, with Julia 1.10 or later and Python 3.11 or later:

```sh
julia scripts/instantiate.jl benchmark
JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/run.jl
python3 benchmark/summarize.py benchmark/results benchmark/results/summary
```

The default is one quick run on small inputs, mostly Float64. It is an exploratory
check, with no between-run uncertainty estimate. Use a fresh output directory
for each measurement series; existing repetitions are never overwritten.

| Setting | Use |
|---|---|
| `PB_SUITE=quick` | Small representative cases; the default. |
| `PB_SUITE=extended` | Float32/Float64, larger inputs and broader cases. |
| `PB_GROUPS=vectors,smooth` | Select groups before constructing their inputs. |
| `PB_REPETITION=1` | Seeded process/input draw, numbered 1–10. |
| `PB_OUTPUT=/absolute/path` | Output directory; defaults to `benchmark/results`. |
| `PB_LIST=true` | List selected case IDs without measuring them. |
| `PB_PROFILE_SECONDS=0.2` | Enable representative profiles; disabled by default. |

Groups are `vectors`, `smooth`, `matrices`, `constraints`, `tensors`, `products`, `blocks` and
`owner`. They cover dense/sparse/atom calls, values and derivatives,
coordinate operations, ordered penalties, constraints and workspace states.
The blocks group covers flat and nested tuple oracles, wrappers and long block sums
on Float32/Float64 inputs, with the same cases in both suites.
The matrix group includes partial/full decompositions and fallback inputs.
Construction plus first use is measured separately from workspace reuse.

The `owner` group measures `SymmetricSparseIterate` (reads, writes, first activation,
coordinate sweeps, `mul!`, `Matrix`, construction) beside a `SparseMatrixCSC` and a dense
`Symmetric` product. It uses only calls that exist in 0.5.x and 0.6, so the same harness
gives the before and after of the redesign in
[#391](https://github.com/mlakolar/ProximalBase.jl/issues/391). The pre-refactor
numbers are recorded in `baselines/symmetric-owner-0.5.md`.

For example, select matrix operations across the extended input sizes:

```sh
PB_SUITE=extended PB_GROUPS=matrices PB_OUTPUT=benchmark/results/matrices \
  JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/run.jl
```

The optional [GPU solver suite](gpu_solver/README.md) measures CPU/CUDA time
to independently checked accuracy on physical GPU hardware.

## Compare revisions

Keep one copy of this harness fixed while measuring two package checkouts.
Prepare fresh environments for the starting and proposed revisions with the
same Julia executable, then run that same harness with each environment:

```sh
julia benchmark/setup.jl /absolute/path/to/before benchmark/results/env-before
julia benchmark/setup.jl /absolute/path/to/after benchmark/results/env-after
PB_EXPECTED_PACKAGE_ROOT=/absolute/path/to/before \
  PB_OUTPUT=benchmark/results/before JULIA_NUM_THREADS=1 \
  julia --project=benchmark/results/env-before benchmark/run.jl
PB_EXPECTED_PACKAGE_ROOT=/absolute/path/to/after \
  PB_OUTPUT=benchmark/results/after JULIA_NUM_THREADS=1 \
  julia --project=benchmark/results/env-after benchmark/run.jl
python3 benchmark/compare.py benchmark/results/before benchmark/results/after \
  benchmark/results/comparison
```

The runner verifies `PB_EXPECTED_PACKAGE_ROOT` before building cases. Use the
same suite, groups, CPU, Julia/BLAS versions and thread settings on both sides. The runner uses one BLAS thread. For stronger evidence, repeat in fresh
processes with `PB_REPETITION=1,...,5` (or up to 10), alternating which revision
runs first. Pass `--expected-runs 5` to the summarizer, validator and comparator.
Repetitions belong in the same directory; different Julia versions need separate
environments. Versions are discovered from metadata, or explicitly required by
`validate.py` and `summarize.py` with `--versions`.

The comparison produces `REPORT.md`, `comparison.csv` and `coverage.json`.
It reports timing variation, paired speedup intervals, mean and maximum bytes,
and unexpected warmed allocations. Single-run ratios are exploratory. Failed
checks, missing repetitions, unmatched cases or incompatible settings prevent
a comparison and retain diagnostics. An interval containing one is inconclusive.

## Measurement and evidence

Compilation, numerical checks and input/cache preparation are outside timing.
Cases without a prepare step use automatic evaluations per sample, tuned within
`PB_SECONDS`. Cases with a prepare step use one evaluation and reset their
starting state before each sample.
`PB_SAMPLES` and `PB_SECONDS` override the sample/time limits; keep them matched
between revisions. Intervals describe variation across process/input draws,
including machine noise; they do not correct for correlated noise or multiple
comparisons. Julia heap bytes exclude native library allocations and peak memory.
Profiles include preparation callbacks and are diagnostic, not latency estimates.

Each process retains a `-calibration.tsv` file with raw no-op timer readings at
`start` and `end`. Metadata records each phase's median, minimum and timer
step: the smallest positive reading after rounding to whole nanoseconds, which
is one tick on a coarse timer. It also records `timer_floor_ns`: the maximum
of the medians and usable steps. Fewer than 100 readings in either phase, or two sub-nanosecond medians
with no steps, gives an unusable calibration (NaN floor).

A measurement is timer-limited when `median_ns * evals < K * timer_floor_ns`
(strict). With a missing or NaN floor, only `evals == 1` is limited. Missing
`evals` is also timer-limited. Older records can be summarized but fail record
validation. Both analysis tools accept `--timer-limit-multiple K` (positive and finite; default 10). Any limited
repetition marks the case/version's `timer_limited` column; comparisons check
both sides. A ratio gate must filter on `timer_limited`. Comparisons retain
ratios, intervals and allocation warnings, with a `timer-limited` assessment.
Their CSV also records `timer_limited_by` (for example,
`before:rep01,rep03;after:rep02`), `timer_limit_multiple`, and distinct
`before_evals` / `after_evals` joined with `/`. Summary reports give the timer
floor range and unusable calibration count per Julia version.

Results retain all samples, checks, allocation observations, settings, package
revision, dirty status and source/harness content hashes. Preserve manifests and
any uncommitted source changes with evidence supporting a PR claim. Do not
select the fastest draw or discard adverse runs. The default suite is not a
substitute for the package tests or broad coverage of every parameter regime.

```sh
python3 benchmark/validate.py benchmark/results benchmark/results/validation.json
python3 benchmark/check_analysis.py
```

Generated results, environments and profiles are ignored by Git. Put the concise
finding and relevant limitations in the PR, and use durable attachments when
detailed records are needed. Historical experiments remain recoverable in the
[pre-cleanup snapshot](https://github.com/mlakolar/ProximalBase.jl/tree/3ec6f5513d3d8c91beec010f156c533f09cbf739/benchmark).
The repeated CPU baseline and its raw-data locations are documented with
[PR #102](https://github.com/mlakolar/ProximalBase.jl/pull/102) and its
[original report](https://github.com/mlakolar/ProximalBase.jl/blob/0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a/benchmark/campaign/baselines/discovery-2026-10-25/README.md).
