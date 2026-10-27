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

## Compiler findings

The first responsible change is `6319e9f`: it sends weighted scalar entries
through `_positive_product`. Its loops still vectorize on Julia 1.13.3/AVX2,
but their vector bodies include the range tests, factor sorting and selected
reordered products. This is extra work, not type instability.

`f48d630` also produces four-wide vector loops on this runner. Contrary to
the original hypothesis, the mere presence of vector code does not make
this revision fast: LLVM if-converts the inlined exceptional arithmetic.
The ordinary vector body computes exponent extraction, variable shifts,
`llvm.ctlz.v4i64`, significand products and exponent reconstruction before
selecting the ordinary result. Native code contains many spills and reloads.
The expensive recovery arithmetic is therefore paid even on ordinary inputs.

`e61e32d` removes that work from the hot loop by outlining recovery, but both
weighted L1 loops then become scalar. Their LLVM has a per-element conditional
call to `_exceptional_product` and no `vector.body`; the value has a scalar
loop-carried floating-point sum. The return types remain concrete and warmed
allocations remain zero.

The repair tests only the two-factor intermediate. A normal intermediate has
not lost information to intermediate underflow or overflow; multiplying it
by the final factor needs no subsequent range recovery. This keeps the same
arithmetic on ordinary normal results and leaves significand/exponent recovery
for exceptional intermediates. It does not claim correctly rounded exact
three-factor products or bitwise agreement with the old double-rounded
exceptional route for every subnormal final result.

For the proximal operator, `step*level` is loop invariant. LLVM unswitches
that test and produces an ordinary vector loop plus an exceptional scalar
loop using the existing generic implementation. An explicit dense proximal
specialization was tested and removed because the entrywise loop already
has the same vector code and measured performance. There is no weight
prepass, new coefficient buffer or new proximal dispatch.

For values, `level*w[i]` varies by entry. The small CPU-array specialization
sums ordinary terms and AND-reduces their intermediate-range tests in one
vector loop. Only after that read-only pass can it call the compensated
entrywise fallback. A finite final sum cannot stand in for those term tests:
the regression test with two terms would otherwise lose half its value.

## Standalone and package loops

The probe separates unchecked arithmetic, fused range checking, a separate
weight-checking pass, and the package calls. The reconstructed prepass does
add memory traffic and work, but it does not reproduce the full historical
66–79 versus 41 microsecond gap. The uncommitted package prototype is not
available, so that historical ratio cannot be attributed more precisely.
The current package value and the equivalent standalone checked reduction
both vectorize and agree in timing on the same runner. The integer-counter
and Boolean reductions both vectorize in these probes; the proposed
explanation for the historical value gap is not supported here.

The new proximal path needs neither of those checking passes; removing its
data-dependent final-product test is enough for the compiler to move the
remaining guard outside the existing loop. This is the concrete reason a
second dense proximal implementation is unnecessary.

## Final measurements

[Run 40465118863](https://github.com/mlakolar/ProximalBase.jl/actions/runs/40465118863)
completed successfully on an AMD EPYC 7763, Julia 1.13.3, BenchmarkTools 1.8.0,
with one Julia/BLAS thread. These are means of the five **process minima**, in
microseconds, on the fixed input described above. All individual minima,
means, allocations and first-call/compilation times are retained in
[probe-metrics.md](evidence/probe-metrics.md).

| Operation | `4dc00b0` | `b7b0659` | `6319e9f` | `f48d630` | `e61e32d` | Repair |
|---|---:|---:|---:|---:|---:|---:|
| Weighted L1 prox! | 37.1 | 37.3 | 59.5 | 672.9 | 128.9 | 31.3 |
| Weighted nonnegative L1 prox! | 37.4 | 35.4 | 48.2 | 642.7 | 110.5 | 31.6 |
| Weighted L1 value | 24.7 | 24.8 | 81.6 | 843.9 | 123.8 | 21.6 |
| Weighted nonnegative L1 value | 92.9 | 92.9 | 222.2 | 210.3 | 156.0 | 31.1 |
| Coordinate sweep | 108.3 | 108.3 | 170.5 | 185.6 | 142.3 | 108.3 |
| Unweighted L1 prox! control | 25.6 | 25.7 | 24.0 | 24.0 | 24.4 | 24.0 |

The ordinary standalone proximal loop is 31.3 microseconds against 31.4 for
the repaired entrywise loop. Fused coefficient checking is 42.4, and a
separate checking pass plus proximal loop is 48.8. The package value is 21.6
against 21.3 for the standalone Boolean reduction and 20.9 for the integer
counter. This quantifies the extra checking/pass costs and the absence of a
systematic package-wrapper penalty in the repaired calls. Small differences
should not be interpreted as parity or superiority claims: even the unchanged
standalone controls vary across revision processes.

The extended suite uses different seeded inputs, level 0.3, step 0.7, weights
in [0.2, 1.2), and its existing preparation/evaluation protocol. Its means
therefore should not be compared directly with the microbenchmark minima.
Five paired processes per revision completed **4,880 measurements** over 488
cases with no case failures and no unexpected allocations. All 32 weighted L1
cases (two penalties, two precisions, two sizes, four operations) were faster
under the comparator's stated intervals. For Float64 and 100,000 entries:

| Operation | Before / after mean, microseconds | Paired speedup [95% interval] |
|---|---:|---:|
| L1 prox! | 133.6 / 53.5 | 2.52 [2.11, 3.01] |
| L1 prox!, in place | 134.1 / 95.4 | 1.41 [1.39, 1.42] |
| L1 value | 126.6 / 37.4 | 3.40 [2.98, 3.88] |
| L1 value_and_prox! | 481.4 / 88.1 | 5.48 [4.73, 6.35] |
| Nonnegative L1 prox! | 116.2 / 51.9 | 2.24 [2.02, 2.48] |
| Nonnegative L1 prox!, in place | 116.8 / 53.0 | 2.20 [2.05, 2.37] |
| Nonnegative L1 value | 167.3 / 37.5 | 4.47 [4.07, 4.91] |
| Nonnegative L1 value_and_prox! | 1092.1 / 89.0 | 12.31 [11.20, 13.53] |

In-place L1 retains a material cost: LLVM's existing memory-overlap check
selects a scalar loop for `y === x`, so its improvement is smaller. No alias
promise was added to the generic separable interface. Combined value/proximal
calls benefit especially from removing unnecessary exceptional-product calls
for the zero outputs of thresholding.

The [full 488-case comparison](evidence/comparison.md) retains all controls:
77 faster, 404 inconclusive and **7 slower**. The largest slower estimates are
NormL2 prox! (0.883x) and IndHalfspace value_and_prox! (0.886x); the others are
GroupNormLinf, FusedLasso, HingeLoss, GroupMCP and ElasticNet controls, with
ratios from 0.918x to 0.998x. These intervals are not corrected for multiple
comparisons and do not establish unrelated kernel regressions. They are
retained without claiming a package-wide speedup. No end-to-end
HighDimensionalModels solver speedup was measured. The separate q=2/3 NormLq
algorithmic tradeoff is not addressed by this repair.

## Numerical validation and provenance

The original 200,000-triple probe per precision gives identical before/after
normal-result maxima: **1.41867704 ulp (Float32)** and **1.37145183 ulp
(Float64)**, with no mismatches in zero or overflow classifications. This is empirical coverage,
not a universal rounding guarantee. Existing tolerances/allocation assertions
were unchanged. New tests cover normal intermediates with subnormal final
results, a normal aggregate that would hide an underflowed contribution,
in-place and weight/output aliasing, zero and infinite factors, matrices,
views and mixed precision. JLArrays retains its existing device implementation.
CPU array specialization uses `Array`, since `DenseArray` also includes
device arrays and caused a dispatch ambiguity in the initial prototype.

The complete local suites passed on Julia 1.10.12 and 1.13.3. Final-code
[package CI](https://github.com/mlakolar/ProximalBase.jl/actions/runs/40465118945)
passed on Linux LTS/current, macOS and Windows, including naming and benchmark
analysis controls; [documentation CI](https://github.com/mlakolar/ProximalBase.jl/actions/runs/40465119052)
also passed.

The measured head is `50518360c2e7108f94ce7b5f37e8aaaa8a3229ac`; Actions
checked out its merge tree at `7f192ad3c5d0f940b5eaef2256d99d8f5cef7298`.
The current production files match the recorded source fingerprint
`654730a599a0a1642222b08562e385d1379e3191`. The suite harness fingerprint is
`3b3a0704f86bcbe6849ca53a9cdb930be60129a6`, and the probe SHA-256 is
`f27e42b6a34015b8c1868200cf2d2ddc1f49de0ae43717b5bccbba01b0208c78`.
Subsequent report-only commits do not change those production files.

The [workflow artifact](https://github.com/mlakolar/ProximalBase.jl/actions/runs/40465118863/artifacts/12716876976)
contains every timing sample, allocation observation, manifest, metadata file
and compiler dump (`probes/REVISION/rep-1/package_{prox,value}.{ll,native,types}`).
Its archive SHA-256 is
`c10567a720de68ac5db393fc31b57d0489b1acc5820dfde7746239b679686f8a`;
GitHub currently retains it until 25 January 2027. Durable process metrics and
the complete comparison are committed beside this report; the workflow and
pinned sources provide the recreation recipe after artifact expiry.

An [earlier run](https://github.com/mlakolar/ProximalBase.jl/actions/runs/40463841528)
was canceled when the redundant proximal specialization was removed. Its
completed historical probes and partial vector-suite results were preserved
by the artifact-upload step, and were not pooled into this final comparison.
