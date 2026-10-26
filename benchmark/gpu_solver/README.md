# GPU solver time to a common accuracy

Optional benchmark for issue #110. It uses ordinary proximal gradient with
`LeastSquaresLoss`, `gradient!`, `NormL1`, and `prox!`. CUDA remains optional;
production implementations and ordinary JLArrays tests are unchanged.

The [Discovery report](https://github.com/mlakolar/ProximalBase.jl/blob/3ec6f5513d3d8c91beec010f156c533f09cbf739/benchmark/gpu_solver/results/discovery-2026-10-26/README.md) retains the
measured V100 comparisons, raw samples, environment and diagnostic receipts.

## Practical recommendation

Benchmark the intended workload on its target hardware at matched, independently
verified accuracy. Include its actual transfer and data-reuse pattern. The V100
study favors CPU execution for small problems and GPU execution for the largest
dense problems, particularly when data remain on the device. Transfers can erase
a modest GPU advantage. The crossover also depends on CPU/BLAS configuration,
precision and stopping checks; these results establish no portable size threshold.
Keep GPU execution an explicit choice. Automatic backend selection and production
optimizations require additional evidence for the intended workload.

## Run the benchmark

From a clean committed checkout on a physical CUDA device:

```sh
julia --project=test/cuda test/cuda/instantiate.jl
julia --project=test/cuda test/cuda/runtests.jl
julia --project=test/cuda benchmark/gpu_solver/check.jl
PB_MODE=small PB_OUTPUT=benchmark/gpu_solver/results/small julia --project=test/cuda benchmark/gpu_solver/run.jl
python3 benchmark/gpu_solver/summarize.py benchmark/gpu_solver/results/small
PB_MODE=full PB_OUTPUT=benchmark/gpu_solver/results/full julia --project=test/cuda benchmark/gpu_solver/run.jl
python3 benchmark/gpu_solver/summarize.py benchmark/gpu_solver/results/full
PB_N=400 PB_P=40 julia --project=test/cuda benchmark/gpu_solver/profile.jl > benchmark/gpu_solver/results/profile-small.log 2>&1
PB_N=10000 PB_P=1000 julia --project=test/cuda benchmark/gpu_solver/profile.jl > benchmark/gpu_solver/results/profile-large.log 2>&1
```

A missing CUDA device is an error. `check.jl` also runs with the package's CPU
project and requires no CUDA installation. Output directories must be fresh.
Save the generated environment, project, manifest, setup files, raw CSV, and
profiles. The Discovery launcher pins CUDA toolkit 12.8 by default, independently of the
installed driver, so V100 devices remain supported. Override `PB_CUDA_RUNTIME`
only with a toolkit supported by the selected device. A manifest alone does not
capture this selection: retain `LocalPreferences.toml` alongside it. To select
12.8 manually after instantiation, run the following command and restart Julia
before validation:

```sh
julia --project=test/cuda -e 'using CUDA; CUDA.set_runtime_version!(v"12.8")'
```

The manifest contains a local package path: develop the recreated
checkout at that path, or explicitly run `Pkg.develop(path=...)` before replay.
The runner verifies the loaded package path. Source SHA and tracked dirty status
are in `environment.txt`; use a committed checkout to capture the harness itself.

## Design and acceptance

The saved-record validators and both reporters share complete-pair acceptance.
Run their synthetic controls without CUDA with:

```sh
python3 benchmark/gpu_solver/check_analysis.py
```

Failed repetitions remain in failure counts and suppress the affected speedup.
The validator also rejects empty designs, fewer than five repetitions, and
nonfinite or negative reference residuals.

The small run uses seed 103, shape 400 × 40, both precisions, and target 1e-4.
The full run first repeats that gate, then measures shapes 400 × 40,
4,000 × 400, and 10,000 × 1,000; seeds 103–105; Float32/Float64;
check frequencies 1 and 10; five repetitions; and both timing scenarios.
Float64 adds target 1e-6 only after the initial gate passes. Set
`PB_REPETITIONS` (at least five), `PB_MAXITER` (default 20,000), and
`PB_BLAS_THREADS` (default four) to change recorded execution settings.

Inputs follow issue #110: seeded Gaussian design divided by sqrt(n), random
signs on the first ceil(0.05p) coefficients, and Gaussian noise of scale 0.1.
Generate in Float64 once and cast before computing the step. The step is the
selected precision's rounding of 1/(1.01 times the squared Float64 spectral
norm of the rounded design). The penalty is the selected precision's rounding
of 0.1 times the maximum absolute Float64 design-response correlation.
CPU and GPU use identical rounded inputs, step, penalty, and initialization.
These shared scalar values and the normalization denominator are in every row.

Preallocated arrays hold coefficients, gradients, proximal candidates and
intermediate vectors. Stopping uses the infinity norm of the gradient mapping,
divided by max(1, maximum absolute design-response correlation, penalty).
The mapping is checked at the returned iterate, including iteration zero and
the cap. Iteration counts mean accepted updates. Less frequent checks can add
updates. All mapping computations and stopping reductions are inside the clock.
There is no line search or acceptance test in this fixed-step algorithm.

Every returned solution is independently checked on the CPU in Float64 using
the rounded inputs. Success requires the requested residual bound and an
objective gap of at most 10 times the tolerance times max(1, reference objective).
The reference uses the same rounded problem and a 1e-10 residual target, with a
100,000-update cap. Its residual and objective are retained. This objective-gap
threshold is a diagnostic acceptance criterion, not a derived error bound.
Failure to construct a reference or pass the initial gate stops expansion;
failed timed solves remain CSV rows and never contribute successful speedups.
Runtime exceptions abort the process and remain in its log; partial CSV output
must not be described as the full design. Coefficient equality is not required.

## Timing boundaries

Data generation, spectral step, reference solve, workspace construction and
first-call compilation/execution have separate setup receipts. The first-call
cost combines compilation and execution; it does not isolate pure compilation.
Every selected timing path is warmed before sampling. CPU/GPU runs alternate
their first backend within each pair. Natural GC is enabled throughout.

Resident timings begin with data and workspace already present, after resetting
the coefficients outside the clock. Transfer timings include fresh workspace
construction, initial input upload, zero initialization, solve, and final
coefficient download. The CPU counterpart includes fresh workspace construction
and the same solve; its Array conversion needs no device transfer. Spectral setup
and compilation are excluded from both scenarios. GPU work is synchronized
before the clock starts and before it stops. Independent verification is outside
the clock. The summary retains per-seed medians and ranges and counts failures.
Raw rows include execution order; slow samples are retained.
The summarizer checks exact design coverage, duplicate rows, source identity,
and recorded numerical success claims before producing a report. Run
`python3 benchmark/gpu_solver/validate.py full` to repeat these saved-record checks.

## Profiles and the Float64 anomaly

Run `profile.jl` separately, choosing an observed CPU win and GPU win from the
summary. `PB_N`, `PB_P`, `PB_SEED` and `PB_PRECISION` select the case. It compares
gradient/value-and-gradient and prox/value-and-prox, and isolates the two GEMVs.
Host allocation counts and CUDA host/device allocation diagnostics are separate.
Live/pool memory receipts before and after repeated solves distinguish collected
live memory from transient allocation and reserved memory.

The profiler captures operation and full-solve CUDA traces. Plain synchronized
wall time, Julia-instrumented time, CUDA-instrumented time, and explicit-GC wall
time run separately on identical data. Frequencies alternate their order.
Profile launch counts, reductions, GEMVs, transfers and host waits before
proposing production changes. Scalar GEMV allocations alone do not establish a
material solver bottleneck. The launcher records clock/utilization/power samples;
these coarse samples cannot explain every short outlier. Do not infer a general
check-frequency recommendation from two instruments or a single device.

## Discovery recreation

After compute authorization, stage a committed checkout at `RUN/source` and
run `sbatch --parsable source/benchmark/gpu_solver/discovery.sbatch` from `RUN`.
The example requests one L40S, four CPUs, 16 GB, and one hour with the existing
Discovery account and Julia installation. Site paths belong in this launcher,
not the portable solver. Keep the job ID, submission command, source revision,
resource allocation, validation logs and outputs in the project's lightweight
run registry. The launcher validates hardware and the solver, runs the small gate,
expands the full design, then profiles two shapes separately. Preserve failed
logs and partial rows. Inspect numerical failures even if the scheduler exits zero.
