# Weighted L1 investigation (issue #129)

The manual `Weighted L1 investigation` workflow reproduces the historical
comparison on one Julia 1.13.3 x86 runner. It sets up five pinned revisions
and the selected repair, runs five alternating fresh-process probes, checks
random-exponent products and compares five pairs of extended vector-suite
runs. It has no automatic pull-request trigger or timing gate.

## Findings

`6319e9f` first routed weighted L1 entries through compensated products.
Its SIMD loops include extra range/reordering work. `f48d630` still produces
vector loops on the measured AVX2 runner, but computes inlined exponent and
significand recovery speculatively in ordinary lanes. `e61e32d` outlines
recovery but leaves scalar loops with conditional calls.

Testing only the intermediate product range lets LLVM unswitch the shared
proximal check in the existing generic loop. A CPU `Array` value kernel checks
all intermediates while summing. A zero-aware SIMD retry handles unpenalized
coordinates and zero levels without slowing the ordinary loop with masks.
It tests the actual factors, since a rounded zero can hide underflow. A NaN
in that speculative retry selects the compensated fallback to preserve zero
factor and nonnegative-domain priorities for Inf/NaN inputs. Every pass still
checks individual nonzero intermediates; a finite total alone is insufficient.

Views, mixed precision and device arrays retain their existing value paths.
In-place L1 retains a scalar overlap loop. No public API, coefficient cache
or proximal specialization was added. Physical GPU performance and end-to-end
HighDimensionalModels solver performance were not measured.

## Original measurements

These results describe the original repair at `5051836`, before the review's
zero-factor refinement. Fixed-input Float64, 100,000 entries, one AMD EPYC 7763
runner, Julia 1.13.3, BenchmarkTools 1.8.0; means of five process minima:

| Operation | `e61e32d`, microseconds | `5051836`, microseconds |
|---|---:|---:|
| Weighted L1 prox! | 128.9 | 31.3 |
| Weighted L1 value | 123.8 | 21.6 |
| Weighted nonnegative L1 prox! | 110.5 | 31.6 |
| Weighted nonnegative L1 value | 156.0 | 31.1 |

The original extended suite retained 488 cases: 77 faster, 404 inconclusive
and seven slower estimates, with no unexpected allocations. Its paired
Float64 L1 mean speedups were 2.52x for prox!, 1.41x in place, and 3.40x for
value. Small control differences do not establish unrelated regressions.
Detailed tables, slower controls, source identities and review-follow-up
measurements belong in [PR #131](https://github.com/mlakolar/ProximalBase.jl/pull/131).
The [original run](https://github.com/mlakolar/ProximalBase.jl/actions/runs/40465118863)
retains raw samples, manifests and compiler dumps until its artifact expires.

## Reproduce

Run the same scripts against separate package environments on the same host:

```sh
julia benchmark/setup.jl /path/to/revision /path/to/fresh-environment
PB_COMPILER=true JULIA_NUM_THREADS=1 julia --project=/path/to/fresh-environment \
  benchmark/weighted_l1/probe.jl /path/to/fresh-output
julia --project=/path/to/fresh-environment benchmark/weighted_l1/products.jl
PB_COMPILER=true JULIA_NUM_THREADS=1 julia --project=/path/to/fresh-environment \
  benchmark/weighted_l1/zero_weights.jl /path/to/fresh-zero-output
```

`zero_weights.jl` compares ordinary, 10% zero-weight and zero-level values for
both penalties and precisions, retaining all warmed samples and allocations.
`PB_COMPILER=true` also saves LLVM. Alternate revision order across five fresh
process pairs. The extended suite includes zero-weight and zero-level cases;
use `PB_SUITE=extended PB_GROUPS=vectors` with `benchmark/run.jl` and the
workflow's repetition/output settings. Keep generated output with the PR,
not in the package source tree.
