# Discovery physical-GPU validation, 2026-10-25

**Passed:** 1,182/1,182 physical CUDA assertions and 1,182/1,182 JLArrays
assertions. No backend failures or production-code changes were needed.

| Field | Recorded value |
|:--|:--|
| Tested commit | `097e788440da1aba71d22d531c209f5e4ecf1ea7` (clean tracked tree) |
| Cluster / node | USC Discovery / `b04-10` |
| Slurm job | `13423570`, `COMPLETED`, exit `0:0` |
| Device | NVIDIA A40, compute capability 8.6, 44.988 GiB |
| NVIDIA driver | `580.159.04` (`nvidia-smi`) |
| CUDA toolchain | Runtime 13.2, compiler 13.3, artifact installations |
| Julia / ProximalBase | 1.10.12 / 0.5.3 |
| CUDA.jl / GPUArrays / GPUArraysCore | 5.11.3 / 11.5.16 / 0.2.1 |
| Adapt / JLArrays | 4.7.4 / 0.3.3 |
| Seed | 1 |
| Float32 tolerance | Relative `5e-5`, absolute `5e-6` |
| Float64 tolerance | Relative `1e-10`, absolute `1e-12` |
| Resources | One A40, four CPUs, 16 GiB host memory; 45-minute limit |
| Elapsed allocation | 5m25s, including environment setup and both suites |
| Physical suite time | 1m24.8s, including kernel compilation |

The launcher was submitted from
`/home1/mkolar/juliaPkg/runs/20261025-pb-gpu103` with:

```sh
sbatch --parsable --job-name=pb-gpu103-20261025 source/test/cuda/discovery.sbatch
```

It ran the CUDA environment setup, then
`julia --project=source/test/cuda source/test/cuda/runtests.jl`, then the shared
JLArrays entry point. The tested source was staged from a Git archive and the
exact commit restored from a small Git bundle; its tree identity was checked
before the harness printed its commit. This report and the evidence files were
added afterward, without changing the tested harness or package code.

The reference-backend totals were 264 separable, 227 norms/projections, 504 group
and 187 loss assertions. Both precisions passed with scalar indexing disabled.
The suite includes weighted/bounded parameters on the device, a user-defined
separable function, aliasing, `value_and_prox!`, and repeated loss calls after
buffer initialization. See [supported boundaries](../../README.md#supported-boundary)
for CPU-only operations, including diagonal steps and least-squares prox.

## Warmed allocation observations

These are single synchronized diagnostic calls after two warm-ups, not timing
benchmarks. CPU allocation includes Julia/CUDA launch overhead; device allocation
counts include the device memory pool. No zero-allocation guarantee is inferred.

| Operation | Precision | Host allocations / KiB | Device allocations / bytes |
|:--|:--|--:|--:|
| `prox!` | Float64 | 76 / 2.141 | 0 / 0 |
| `value_and_prox!` | Float64 | 124 / 3.469 | 1 / 8 |
| Least-squares `gradient!` | Float64 | 179 / 4.047 | 4 / 32 |
| GLM `value_and_gradient!` | Float64 | 257 / 5.906 | 5 / 40 |
| `prox!` | Float32 | 77 / 2.078 | 0 / 0 |
| `value_and_prox!` | Float32 | 125 / 3.406 | 1 / 4 |
| Least-squares `gradient!` | Float32 | 179 / 4.047 | 4 / 16 |
| GLM `value_and_gradient!` | Float32 | 257 / 5.906 | 5 / 20 |

## Evidence and recreation

- [Raw output](slurm-13423570.out): driver output, source identity, full package
  versions, device metadata, test summaries and separate allocation diagnostics.
- [Resolved manifest](Manifest.toml): exact dependency versions used by the job.
- [Slurm accounting](accounting.tsv): exact allocation and successful exit.

From the repository root on a machine with a supported NVIDIA device:

```sh
cp test/cuda/validation/discovery-2026-10-25/Manifest.toml test/cuda/Manifest.toml
julia --project=test/cuda test/cuda/instantiate.jl
julia --project=test/cuda test/cuda/runtests.jl
julia --project=test/cuda -e 'include("test/gpu.jl")'
```

The setup rewrites the manifest's original absolute ProximalBase development
path to the current checkout. Other dependency versions are retained by the
manifest. For the exact tested source, use commit `097e788` and copy the saved
manifest from this report before checking out that commit. No other physical
device or backend was tested in this run.
