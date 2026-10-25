# Repeated CPU benchmarks and profiles

This harness measures a broad catalog of package operations and retains a baseline
for future implementation changes. It does not change the package implementation.
It covers Float32/Float64; small/large, dense/sparse/atom/symmetric-sparse/strided
inputs; scalar/diagonal steps; first-use and warmed caches; proximal/value/gradient
and second-order operations; constraints, spectral and partial decompositions,
stacked-array functions, sparse products, and a fixed-work lasso coordinate trace.
The catalog is in `cases.jl`. GPU hardware and exhaustive parameter coverage are
outside this CPU campaign.

## Measurement design

- Run ten fresh Julia processes per supported Julia version, with repetitions
  numbered 1–10. Each repetition draws inputs from stable scenario keys and its
  repetition number, and shuffles measurement order deterministically.
- Use one benchmark worker per node, one Julia thread and one BLAS thread on
  a fixed CPU model. Independent workers may run on different nodes, never
  concurrently on the same node. Record the machine, versions, source/harness commits and
  thread settings. Separate project manifests for different Julia versions.
- Check correctness and warm compilation outside the measured region. Each
  case collects up to 40 BenchmarkTools samples or 0.1 seconds; actual counts
  are retained. Small stateless calls batch 25 evaluations; stateful and aliasing
  calls use one. Mutated inputs and nearby-call cache states are restored by
  preparation callbacks outside the clock.
- Report the arithmetic mean of ten **run means**, between-run standard deviation,
  and a Student-t 95% interval. Keep run medians, tails, GC time, all raw sample
  times and all runs. Do not report the fastest draw as a typical time. Input
  variability and shared-node machine noise are both present in these intervals;
  ten draws are a useful baseline, not a guarantee of narrow intervals.
- Measure allocated bytes in ten separate warmed calls per case per process.
  Retain maximum and mean bytes, so occasional allocations are visible.
  Constructor-plus-first-prox costs have their own phase and allocation policy.
- Collect representative CPU profiles, including preparation callbacks for
  stateful traces, separately from benchmark timings. Allocation stack profiles
  isolate the timed operation when a supposedly allocation-free call allocates.
  Profiles are sampling diagnostics, not another source of latency measurements.

Validation compares optimized layouts with dense calls, checks returned values,
uses directional derivatives for Float64 gradients and Hessian products, compares
partial decompositions with full decompositions, checks a specialized simplex
projection against Dykstra, and checks residual/objective invariants in the lasso
trace. These checks complement the unit tests; they are not an independent proof
of every operator's correctness. Failed cases retain their errors and are excluded
from timing conclusions; missing repetitions are reported explicitly.

## Run and analyze

Create a project from `Project.toml`, develop the package checkout into it and
instantiate it under the Julia version being measured. Set the three source and
output variables, then run the harness ten times, serially:

```sh
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PB_SOURCE_SHA=<package-commit> PB_HARNESS_SHA=<harness-commit>
export PB_OUTPUT=/absolute/path/to/fresh-results
PB_REPETITION=1 julia --project=/path/to/version-specific-project run.jl
# Repeat PB_REPETITION=2,...,10 in new processes, using the same output directory.
python3 summarize.py "$PB_OUTPUT" /path/to/summary --expected-runs=10
```

Use Python 3.11 or later for the paired comparison utility. For the user's
Discovery runs, all execution, analysis and profiling runs under Slurm using
the task's saved launchers. GNU Parallel pins its worker slots to separate nodes
with exclusive, CPU-bound `srun` steps. Cluster account, paths and resource settings stay in the
run record rather than this portable harness.

`PB_MODE=canary` reduces samples and profile duration for launcher validation.
`PB_SAMPLES`, `PB_SECONDS` and `PB_PROFILE_SECONDS` override measurement settings;
retain identical settings when comparing changes. Existing repetition files are
never silently overwritten.

To compare a later implementation using the same case IDs, keyed seeds, Julia
version, CPU and settings:

```sh
python3 compare.py /path/to/before-results /path/to/after-results /path/to/comparison
```

The primary before/after times remain arithmetic means. Paired speedups use the
geometric mean of the ten within-seed ratios with an interval on log ratios;
an interval spanning one is inconclusive. Review allocation counts, numerical
checks and variability alongside speed. The comparator checks recorded settings;
also confirm the actual CPU model and matching input-generation code before
interpreting its output.
