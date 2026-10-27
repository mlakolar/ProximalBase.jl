# Weighted L1 investigation (issue #129)

The `Weighted L1 investigation` workflow runs Julia 1.13.3 on one x86
runner. It retains all samples and generated LLVM/native/type output in the
`weighted-l1-investigation` artifact, including failures. Nothing in this
workflow is a timing gate in ordinary package CI.

`probe.jl` compares `4dc00b0`, `b7b0659`, `6319e9f`, `f48d630`, `e61e32d`
and the repair in five fresh processes per revision, reversing their order
on alternate repetitions. All use the issue's fixed-seed input, 100,000
Float64 entries, level 0.5 and step 1. Minima and means are warmed
BenchmarkTools measurements; first invocation and compilation are separate
columns. First invocation uses `invokelatest` to keep compilation inside its
timer; it includes that dispatch overhead and excludes package loading.
Ordinary-input ablations compare entrywise calls, plain arithmetic,
a fused range check, an extra weight pass, and Boolean/integer reductions.
These ablations are diagnostics, not numerically safe replacement APIs.

The workflow also runs the existing extended vector suite against `e61e32d`
and the repair, five times in alternating order, with Float32/Float64 and
sizes 256/100,000. `compare.py --expected-runs 5` checks the retained samples,
allocations, matching settings and coverage, and reports paired mean-time
ratios and intervals. This broader comparison retains adverse controls.
`products.jl` repeats the original 200,000-triple random-exponent probe for
each precision against a 512-bit reference; the workflow requires identical
before/after maxima and mismatch counts.

For local reproduction, prepare separate environments with the fixed harness:

```sh
julia benchmark/setup.jl /path/to/revision /path/to/fresh-environment
PB_COMPILER=true JULIA_NUM_THREADS=1 julia --project=/path/to/fresh-environment \
  benchmark/weighted_l1/probe.jl /path/to/fresh-output
julia --project=/path/to/fresh-environment benchmark/weighted_l1/products.jl
```

For the vector suite, use `PB_SUITE=extended PB_GROUPS=vectors` with
`benchmark/run.jl` and the repetition/output settings in the workflow.
Performance conclusions apply to the measured ordinary dense inputs. The
proximal operator retains its generic entrywise implementation: LLVM can
unswitch the simplified product check when the step and level are shared.
Only values on ordinary Float32/Float64 CPU arrays have a specialized loop;
views, mixed precision and device arrays retain their existing value paths.
Exceptional coefficients (including zero weights) select the compensated
value fallback, so the ordinary-input speedup is not promised for them.
