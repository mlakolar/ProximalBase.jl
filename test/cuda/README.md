# Physical CUDA validation

This optional environment runs the shared GPU tests on a physical NVIDIA GPU.
CUDA is not a runtime dependency, and ordinary CI continues to run `test/gpu.jl`
with JLArrays on the CPU. Julia 1.10 or later and a CUDA 5 compatible device and
driver are required. No device is an error, rather than a skipped test run.

From the repository root:

```sh
julia --project=test/cuda test/cuda/instantiate.jl
julia --project=test/cuda test/cuda/runtests.jl 2>&1 | tee cuda-validation.log
julia --project=test/cuda -e 'include("test/gpu.jl")'
```

Use `set -o pipefail` when piping the output so a failed test remains a failed
command. Retain the generated `test/cuda/Manifest.toml` alongside the log to
recreate the resolved versions. The log records the source commit and tracked
dirty status, device, CUDA runtime and driver, Julia and package versions, seed,
tolerances, test counts and warmed allocation diagnostics. Run from a clean
committed checkout for an unambiguous source reference.

The shared suite covers Float32 and Float64, separable functions including a
user-defined bits type, array weights and bounds, norms, projections, supported
group penalties with host and device layouts, least-squares and GLM values and
gradients, matrix responses, in-place calls and `value_and_prox!`. Losses revisit
both a changed input and the original input after initializing their buffers.
Scalar indexing is disabled. Device work is synchronized before copying results,
comparing floating-point values and measuring allocations. Float32 comparisons
use `rtol=5e-5, atol=5e-6`; Float64 comparisons use `rtol=1e-10, atol=1e-12`.
Indicator values and discrete selections are compared exactly where appropriate.

The warmed diagnostics use `CUDA.@time`, which reports **host allocations and
device allocations separately**, including allocations served by the device
memory pool. They exclude setup and compilation; they are observations, not
zero-allocation assertions or a performance benchmark. See the
[CUDA profiling documentation](https://cuda.juliagpu.org/stable/development/profiling/).

## Supported boundary

The existing GPU extension covers scalar-step operations in this suite. Diagonal
steps currently use CPU loops and are not supported by this extension. The
least-squares proximal operator, spectral functions, `LogDetLoss`, sparse
iterates and GLM second-order methods remain CPU-only. `IndBallGroupL0` selects
groups on the host; imposing an additional within-group entry bound `k` makes its
projection CPU-only. The tests check the explicit errors for least-squares prox
and the bounded group projection. Other GPU backends still have reference-backend
coverage only; a CUDA run does not validate their device compilers.

## Discovery

Stage a committed checkout at `RUN/source`, then run from `RUN`:

```sh
sbatch --parsable source/test/cuda/discovery.sbatch
```

The launcher requests one A40, four CPUs, 16 GB host memory and 45 minutes under
`mkolar_1314`. It uses the installed Julia 1.10.12 and a run-specific depot layered
over the existing shared depot. Adjust site paths and resource settings for other
clusters. Submitting the launcher requires compute authorization.

The [Discovery report from 2026-10-25](validation/discovery-2026-10-25/README.md)
records the first physical A40 run, including its raw output and manifest.
