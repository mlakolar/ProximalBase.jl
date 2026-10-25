# Discovery common-accuracy comparison, 26 October 2026

One Tesla V100-PCIE-32GB, Intel Xeon Gold 6130, four allocated CPU cores and
four OpenBLAS threads; one Julia thread. Julia 1.10.12, CUDA.jl 5.11.3,
CUDA runtime 12.8, compiler 12.9, driver 580.159.04. This date uses the project
owner's Asia/Dubai timezone; the environment receipts retain UTC timestamps.

Job **13447752** completed in 11m16s with exit 0. Physical CUDA validation passed
**1,182 assertions** and the solver checks passed **33 assertions**. The small gate
has 80 timing records; the full design has **1,080 records and zero failures**.
Both downloaded datasets passed exact coverage and recorded-success validation.
The largest independent residuals were 9.911e-5 for target 1e-4 and 9.594e-7 for
target 1e-6. The maximum absolute objective gap from the tighter CPU reference
was 2.622e-7. These are finite-sample checks, not a convergence theorem.

## CPU/GPU comparison

The [compact tables](tables.md) give the ranges of seed-specific medians at target
1e-4, checked every update. The [full summary](full/summary.md) retains every seed,
target and check frequency, including the within-seed minimum and maximum of
five timed repetitions. All individual samples remain in [timings.csv](full/timings.csv).

The CPU wins the 400 × 40 cases. For 4,000 × 400, Float32 still favors the CPU.
Resident Float64 GPU solves are 1.33–1.40 times faster, but including input upload,
workspace construction and coefficient download reverses that result: GPU times
are 5.34–5.42 ms versus CPU times of 3.46–3.70 ms.

At 10,000 × 1,000, the resident GPU speedup is 6.57–6.72 for Float32 and
10.53–10.72 for Float64. With transfers and workspace setup included, these
ranges fall to 1.74–1.81 and 2.25–2.39. The extra GPU scenario cost is about
8.4 ms and 16.7 ms respectively; it includes allocation and initialization,
so it is not a pure PCIe transfer measurement. Across all 108 paired
seed/configuration groups, 51 favor the GPU and 57 favor the CPU.
These observations apply to this device, CPU allocation, workload and process.
They do not locate a general crossover or provide independent-process uncertainty.

Both timing scenarios exclude input generation, the spectral step calculation,
reference fitting, and compilation. Resident resets and verification downloads
are outside the clock. Transfer-inclusive solves construct a fresh workspace,
upload inputs and download the final coefficients inside the clock. Stopping
reductions are included; there is no line search in this fixed-step method.
Setup receipts record construction and first calls separately. For example,
the first full-run Float64 small-case GPU construction took 0.258 s and its
first solve took 3.906 s. These first-call costs combine compilation and execution;
they are not pure compiler timings or steady-state samples. Process startup and
package loading are excluded. The source order can affect such first-use costs.

## Allocation and timing diagnostics

The original [small](profile-400.log) and [large](profile-10000.log) diagnostics
use Float64, seed 103. Each gradient allocated four device scalars (32 bytes);
each isolated GEMV allocated two (16 bytes). `prox!` allocated no device memory.
The value-returning variants introduced reductions: value-and-gradient allocated
5/6 device objects (40/168 bytes, small/large), and value-and-prox allocated
1/2 objects (8/24 bytes). Host allocation observations are recorded separately
and do not measure device allocation or peak native memory.

For the large case, pool-used memory was 76.477 MiB after collection both before
and after 20 repeated solves; reserved memory was 96 MiB. The small case returned
to 132.500 KiB used, with 32 MiB reserved. This limited check found no retained
live-memory growth. It does not establish zero transient allocation. A separate
instrumented large solve allocated 840 device bytes with checks every iteration,
or 744 bytes with checks every ten iterations.

The diagnostic timing instruments are separate from the main timing samples.
For the large case, synchronized natural-GC wall time had a median of 4.539 ms
(range 4.491–4.692) with checks every iteration, versus 5.095 ms
(5.085–5.432) with checks every ten. The latter used 20 updates rather than 14.
Julia-instrumented medians were 4.503/5.084 ms; explicit-GC wall medians were
4.707/5.285 ms. CUDA-instrumented samples and every slow sample are retained in
the logs and compact tables. Julia-instrumented samples reported zero host GC time.
These V100 measurements do not resolve the earlier L40S bimodality. Coarse
one-second clock/utilization telemetry covers the diagnostics, not the main
benchmark, and cannot assign a clock state to each millisecond solve.

## Rendered traces and implementation decision

Profile-only job **13447957** completed in 1m23s on V100 node d13-04.
It used the same package and saved CUDA environment as the timing run, with
profile source `68eea64f88780aad1749c01c7e42ef03b0ba2814`.
The [small CPU-win trace](profile-v5/profile-400.log) and
[large GPU-win trace](profile-v5/profile-10000.log) include operation and
whole-solve host/API and device activity. They run on a second node of the same
GPU model; their instrumented durations are separate diagnostics and do not
replace the main timings. Profile job receipts and telemetry are in
[profile-v5](profile-v5).

| Float64 case | Check frequency | Profile wall ms | Device busy ms | Kernel launches | Host scalar downloads | GEMV-named kernels |
|---|---|---|---|---|---|---|
| 400 × 40 | 1 | 3.73 | 0.318 | 92 | 13 | 26 |
| 400 × 40 | 10 | 4.19 | 0.390 | 112 | 3 | 42 |
| 10,000 × 1,000 | 1 | 5.20 | 3.53 | 136 | 15 | 30 |
| 10,000 × 1,000 | 10 | 5.83 | 4.73 | 136 | 3 | 42 |

Kernel launch counts combine driver `cuLaunchKernel` and runtime
`cudaLaunchKernel` API calls. Host scalar downloads are the recorded
`cuMemcpyDtoHAsync` calls for stopping reductions. Neither count includes an
initial input upload or final coefficient download: these are resident solves.

In the small every-update solve, the GPU was busy for only 8.52% of the trace;
CUDA API calls occupied 1.59 ms, or 42.68%. The trace shows many short launches,
scalar transfers and gaps between GPU operations. In the large every-update solve,
GPU activity occupied 67.87%; GEMV-named kernels accounted for 3.135 ms of the
3.53 ms device-busy interval. Host API time was 2.41 ms and can overlap device
work; these percentages must not be added. Matrix-vector work matters much more
in the large case, while host submission remains visible.

The gradient-only trace contains four 8-byte pool allocations and four host-to-device
scalar copies. Each isolated GEMV contains exactly two of each. This localizes the
four-scalar allocation observation to the GEMV calls. It does not by itself
identify the wrapper's scalar variables or prove a reusable-buffer speedup.
The value-returning gradient adds one stopping-independent scalar download and
one/two reduction kernels for the small/large problem. The prox-only call is one
broadcast; value-and-prox adds one/two reduction kernels, a scalar download and
host synchronization. Both full-solve traces retain the reductions and host waits
required by the selected stopping rule.

Checking every ten updates reduces host scalar downloads from 13/15 to three,
but increases GEMV work from 26/30 named kernels to 42. It therefore did not
improve these seed-103 diagnostic solves. Neither the count comparison nor the
separate timing instruments justify a general check-frequency recommendation.
The [parsed trace summary](profile-v5/trace-summary.json) retains counts and
summed displayed activity durations; the full printed traces remain authoritative.

Leave package implementations unchanged. Small cases favor CPU execution, and
matrix-vector work dominates the large device trace. The scalar allocations are
now localized, but their removal has no measured end-to-end benefit yet. Any
future scalar-buffer or launch-fusion patch should be separate, preserve stream
and numerical behavior, and demonstrate repeated before/after improvement to
the same verified accuracy. Individual `CUDA.@time` call latencies can include
instrumentation compilation and must not be used to rank operations.

## Provenance and recreation

Measured package and timing harness source:
`30d96c9d965cd062b9decf72afcae8076b716bce`. The only tracked dirty file was
`test/cuda/Project.toml`: selecting the runtime added the CUDA runtime JLL to
its extras and reordered TOML tables. The exact [patch](environment-project.patch),
resolved project, manifest and runtime preferences are retained in both result
folders. No package implementation or benchmark code was dirty.

Use a checkout of that commit with the saved `full/Project.toml`,
`full/Manifest.toml` and `full/LocalPreferences.toml`. The manifest's package path
is local to the original run: develop the recreated checkout into that environment
before instantiation, then keep the toolkit preference when restarting Julia.
Follow the [benchmark commands](../../README.md); fresh output paths are required.
The saved [launcher](launch.sbatch), [submission](submission.txt),
[accounting](accounting.tsv), [hardware receipt](hardware.txt),
[environment](full/environment.txt), and [run record](run-record.txt) identify
the actual run. The source launcher requested an L40S/one hour by default;
the recorded submission overrode it to one generic GPU and 20 minutes.

The earlier V100 attempt used the automatically selected CUDA toolkit 13.2,
which could not compile for sm70. It was cancelled after 5m23s and collected no
solver timings. Its [compressed failure log](failed-attempt-v3-validation.log.gz)
is preserved. Selecting runtime 12.8 before starting a fresh Julia process fixed
the incompatibility. Keep the saved preferences: the manifest alone is insufficient.
The original diagnostic logs are retained as collected; the later profile-only
job supplies their missing rendered traces without rerunning valid timing samples.

Recheck the downloaded records and regenerate the compact tables from the repository root:

```sh
python3 benchmark/gpu_solver/validate.py benchmark/gpu_solver/results/discovery-2026-10-26/small
python3 benchmark/gpu_solver/validate.py benchmark/gpu_solver/results/discovery-2026-10-26/full
python3 benchmark/gpu_solver/report_tables.py benchmark/gpu_solver/results/discovery-2026-10-26
```
