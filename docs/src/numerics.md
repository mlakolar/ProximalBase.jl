# [Numerical contract](@id numerics)

This page states the retained numerical behavior and its limits. The
[historical decisions](https://github.com/mlakolar/ProximalBase.jl/tree/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design/numerical-contract)
cover a much larger package; they are historical evidence rather than a live
capability list. Current source and tests govern the minimal core.

## Supported precision and oracle agreement

The supported CPU numerical targets are homogeneous `Float64` and
homogeneous `Float32`. In the supported configuration, the floating-point
design/response data, iterate or proximal input point, function parameters (such as
levels, weights and centers), proximal step, and floating output buffers
all use the same type `T`. `Float64` is the primary performance and
accuracy target; `Float32` is also tested. Individual methods may accept
ordinary integer parameters; this is not a promise to coerce arbitrary
mixed floating-point arguments.

**Mixed `Float32`/`Float64` is not a guaranteed mode.** Some current
methods still accept it and perform internal promotion, but their return
types, numerical accuracy, exceptional-range behavior and performance
are implementation-specific. They are not required to reject such calls.
A wider data or parameter type does not establish a general promotion rule
for the result. Other real types, including `BigFloat`, are not guaranteed
production types. `BigFloat` remains useful for independent reference tests.

For supported homogeneous array-function calls:

- `f(x)` returns the value in `T = float(eltype(x))`.
- Allocating gradient and proximal operations produce their documented
  representations in `T`; `SymmetricSparseIterate`-specific results are described in the
  symmetric iterate guide.
- Mutating operations respect the destination's element type and never
  change the caller's storage type.
- `value_and_gradient!(out,f,x)` returns the standalone value at the
  input `x` and stores the corresponding gradient in `out`.
- `value_and_prox!(out,f,x,γ)` returns the value at the **stored**
  proximal output, including its rounding, not at an unrounded temporary.
- Exact input/output proximal identity is supported; built-in gradients
  reject input/output identity. Other partial/derived-view overlap is
  unsupported.

Some implementations use promoted private scratch or widened arithmetic
to improve numerical stability. Those internals do not widen the supported
set of caller precision combinations or imply a generic model-precision
override for custom functions.

For `Float32` calls, explicitly construct floating parameters and steps
in `Float32`, rather than assuming Julia's `Float64` literal `0.5`
will be treated as a supported `Float32` parameter. The
[array and precision conventions](numerical_usage.md) include a
homogeneous example for both supported types.

## Overflow and underflow of values

Ordinary finite calls use efficient numerical paths. Exceptional paths
selectively recover representable values when an intermediate product or sum
would overflow or underflow. These include scaled square sums, finite-data
affine/quadratic values and products of nonnegative level/step/weight factors.
`SqrNormL2` also recovers gradient differences and proximal convex-combination
coefficients. `SqrNormL2` and `LeastSquaresLoss` share one squared-residual
recomputation: each residual is formed before scaling (as `x/2 - y/2`, one
binade up, when `x - y` itself overflows) and the squares are summed in units
of the largest one. Genuine NaN/Inf data or parameters keep their ordinary
IEEE result; only finite inputs whose intermediates leave the range are
recomputed. Recovery is worthwhile when it changes a materially inaccurate
answer into a representable result in the working precision.

These are practical numerical safeguards, not a promise of exhaustive
range recovery or correctly rounded arbitrary reductions, gradients, or
matrix products. When the final mathematical value itself is out of range,
an appropriate infinity or underflow to zero may be unavoidable, subject
to the function's endpoint semantics. There is no cross-platform or
cross-version bitwise arithmetic-order guarantee.

```@example numerical_range
using ProximalBase
λ = 2.0^-1022
f = SqrNormL2(λ,[-2.0^1023])
x = [2.0^1023]
@assert gradient(f,x) == [4.0]
z = prox(SqrNormL2(2.0^800),[2.0^900],2.0^800)
@assert z == [2.0^-700]
reference = setprecision(BigFloat,256) do
    Float64(big(2.0)^900 / (1 + big(2.0)^1600))
end
@assert z[1] == reference
z
```

Nested positive finite `Scaled` factors can retain their exponent sum so an
overflowed/underflowed intermediate scale does not discard a recoverable leaf
result. This mechanism is implemented for retained kernels, not synthesized
for custom functions. Original `Squared()` scalar arithmetic has no such
array-value recovery.

## NaN, infinities and zeros

There is no universal non-finite endpoint rule or universal step validator.
The mathematical prox uses positive finite γ. Many retained closed forms accept
zero, negative, infinite or NaN steps and apply their established arithmetic;
acceptance does not establish a meaningful proximal minimizer in those cases.
Decomposition-based methods are intended for finite data and ordinary valid
steps. Do not pass arbitrary non-finite data into them expecting a uniform
all-NaN result or a particular exception.

| Case | Retained behavior |
|:--|:--|
| `ZeroFunction` value | Zero even at NaN/Inf input entries |
| Unweighted `NormL1`, zero level | Zero value on dense float arrays, including NaN entries |
| `NormL1`, zero weight | Unpenalized entry at every level |
| `NormL1`, infinite weight | Finite input values map to zero, even with zero level or step. No fixed-zero guarantee for `NaN`/`±Inf` inputs with finite level; `shrink` can return `NaN`. Value is `Inf` away from zero. See [#853](https://github.com/mlakolar/ProximalBase.jl/issues/853). |
| `NormL1`, infinite level and positive step | Positive-weight entries prox to zero, including non-finite inputs |
| `NormL1`, finite factors whose product overflows | Distinguished from genuinely infinite weights/levels; recovery handles representable finite results where supported. With an input of `±Inf`, an overflowed finite threshold can yield `NaN`; [#866](https://github.com/mlakolar/ProximalBase.jl/issues/866) is a documented, deferred limitation, not an authorized hot-loop fix. |
| `SqrNormL2`, zero level | Zero standalone value; gradient/prox can still encounter IEEE zero-times-nonfinite arithmetic |
| `SqrNormL2`, infinite level | Gradient is the IEEE `λ(x - y)`: ±Inf for every nonzero residual, subnormal ones included, and NaN at a zero residual |
| Generic custom value/prox | Author-defined arithmetic; traits do not grant recovery or NaN policy |
| Paired symmetric parameters, in `check_symmetric_parameters` | Exact equality, NaN matches NaN, signed zeros agree; the numerical oracles assume it and do not compare |
| Opposing infinities in a symmetric iterate KKT gradient pair | ArgumentError at that chosen pair |

```@example numerical_endpoints
using ProximalBase
@assert NormL1(0.0)([NaN,Inf]) == 0
@assert prox(NormL1(Inf),[NaN,Inf],1.0) == [0,0]
@assert prox(NormL1(0.0,[Inf,0.0]),[2.0,3.0]) == [0,3]
@assert isnan(shrink(NaN,1.0))
@assert prox(NormL1(1.0,[Inf]),[2.0],1.0) == [0.0]  # finite input fixed at zero
@assert prox(NormL1(0.0,[Inf]),[2.0],0.0) == [0.0]  # zero level/step
@assert isnan(prox(NormL1(1.0,[Inf]),[Inf],1.0)[1])  # non-finite limitation (#853)
@assert isnan(prox(NormL1(1e300,[1e10]),[Inf],1.0)[1])  # final factor overflows (#866)
@assert isnan(prox(NormL1(1e10,[1.0]),[Inf],1e300)[1]) # intermediate product overflows (#866)
prox(NormL1(Inf),[NaN,Inf],1.0)
```

**Decision on `NormL1` non-finite inputs (#853 and #866).** A genuinely infinite
coordinate weight maps a *finite input value* to zero, even at zero level and step.
For inputs of `NaN` or `±Inf` with finite level, that fixed-zero guarantee is
intentionally not extended: retained soft-threshold arithmetic may return
`NaN`. Separately, a threshold made from *finite* factors can overflow to
`Inf`; with an input of `±Inf`, the current result can likewise be `NaN` even
though the mathematical threshold is finite. These examples characterize
present behavior, not additional non-finite-output guarantees. A genuinely
infinite **level** with positive step retains its distinct fixed-zero rule
for positive weights, including non-finite inputs (historical decision 49).

The endpoint extensions were deferred to protect ordinary proximal throughput:
PR [#874](https://github.com/mlakolar/ProximalBase.jl/pull/874) measured
slowdowns of **+5.9% (`Float32`) and +9.9% (`Float64`)** for weighted array steps
at 65,536 entries after optimization, and reported **+19–31%** for weighted
scalar steps on a newer local CPU. Unlike finite-input overflow/underflow
recovery, these cases require a particular convention for a genuinely
non-finite input. No additional checks or scans are authorized in the hot
loop for #853 or #866. Reconsider #866 if a practically important finite-input
failure is demonstrated or an implementation preserves ordinary-path speed.

Signed zeros do not activate absent coordinates; explicit active signed zeros
can be retained in storage. Symmetric-iterate `isequal` distinguishes stored negative zero
from absent positive zero; `==` does not. `dropzeros!` removes either signed zero
and preserves NaN/Inf.

## Sparse product semantics

The specialized products in [the operation table](iterates.md#Supported-products)
ignore zero iterate coefficients, stored or absent. No term is formed even if
the other operand at that coordinate is NaN/Inf. A nonzero NaN/Inf coefficient
contributes ordinarily. A zero in a dense factor remains an IEEE zero.
A coordinate view uses ordinary dense-vector semantics instead.

```@example numerical_sparse_product
using ProximalBase, LinearAlgebra
x = SparseIterate(2)
x[1] = 2
activate!(x,2)
v = [3.0,Inf]
@assert dot(v,x) == 6
@assert isnan(dot(v,coordinates(x)))
dropzeros!(x)
@assert dot(v,x) == 6
dot(v,x)
```

Five-argument products with β=0 overwrite the destination without reading its
old numerical entries; α=0 scales/clears output without reading right-operand
values, after required type/shape/alias checks. Finite summation can depend on
activation order; compaction can therefore change final rounding.

The direct `A*X*B` kernels for matching `Float32`/`Float64` Symmetric-strided factors
and matrix iterate X recover the range of individual finite three-factor terms.
They promise accuracy within one neighboring floating value for a normal term,
or one spacing for a subnormal term, not correct rounding. A non-finite sum
is recomputed in scaled units only when all contributing factors are finite;
true NaN/Inf data keep ordinary IEEE outcomes. Tiny terms can be rounded away
beside large ones. Other mixed/generic product routes do not inherit this
specific guarantee.

## Validation, aliasing and evidence

Predictable type/shape/parameter errors precede built-in oracle output writes.
A read, conversion or numerical failure can leave output changed. Storage
state copies and matrix imports have distinct guarantees documented in
[views and aliasing](developer/views_aliasing.md). Parameter arrays are not
made immutable or copied by validation. Call `reset_cache!` after same-shape
cached data changes.

Retained evidence includes `test/ridge_precision.jl` (independent BigFloat
ridge references), `test/minimal_math.jl`, `test/core_functions.jl`,
`test/sparse_iterate.jl`, `test/sparse_iterate_basics.jl`,
`test/symmetric_products.jl`, `test/matrixview.jl` and
`test/numerical_kernels.jl` (BigFloat references for the shared recovery
kernels and their `SqrNormL2`/`LeastSquaresLoss`/`LinearFunction` consumers). These are concrete tested
domains; a historical acceptance goal for a removed family is not evidence for
this implementation. New numerical claims need independent calculations,
FP32/FP64 checks, exceptional inputs and regression tests.
