# [Numerical contract](@id numerics)

This page states the retained numerical behavior and its limits. The
[historical decisions](https://github.com/mlakolar/ProximalBase.jl/tree/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design/numerical-contract)
cover a much larger package; they are historical evidence rather than a live
capability list. Current source and tests govern the minimal core.

## Result precision and oracle agreement

The supported validation targets are `Float32` and `Float64` on CPU. Real input
is required; other real element types can dispatch, with less validation
coverage. Retained array values are delivered in `float(eltype(x))`.
Allocating gradient/prox buffers start in the argument's floating element type.
Mutating methods convert to the destination's element type.

The contracts of the fused calls are:

- `value_and_gradient!(out,f,x)` returns the standalone value at x, and writes
  the standalone gradient in out's precision.
- `value_and_prox!(out,f,x,γ)` returns the value at the **stored** proximal
  output, including its rounding, rather than at an unrounded temporary.
- Exact input/output proximal identity is supported; gradient identity is
  rejected for built-ins. Other partial/derived-view overlap is unsupported.

`SqrNormL2`'s model precision for gradient/prox is the wider of argument and
output floats. Its level, center and step do not widen ordinary calculations;
parameters are rounded to that model where representable. Least-squares scratch
uses floating promotion of X and Y; quadratic scratch also promotes b and c.
A wide parameter/data type is not permission to assume every result is returned
in that wider type. The source's internal `_model_precision` commentary is
not a generic precision override for custom functions.

## Overflow and underflow of values

Ordinary arithmetic order is retained at ordinary magnitudes. Exceptional
paths recover representable values when an intermediate product or sum would
leave range. These include scaled square sums, finite-data affine/quadratic
values and products of nonnegative level/step/weight factors. `SqrNormL2` also
recovers gradient differences and proximal convex-combination coefficients.
This does not guarantee correctly rounded arbitrary reductions or recovery of
every gradient/matrix product.

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
| `NormL1`, infinite weight | Finite trial thresholds to zero even at zero level; NaN/Inf trial can return NaN with finite level; infinite value away from zero |
| `NormL1`, infinite level and positive step | Positive-weight entries prox to zero, including non-finite trials |
| `NormL1`, finite factors whose product overflows | Distinguished from a genuinely infinite level; bounded-factor recovery is used |
| `SqrNormL2`, zero level | Zero standalone value; gradient/prox can still encounter IEEE zero-times-nonfinite arithmetic |
| Generic custom value/prox | Author-defined arithmetic; traits do not grant recovery or NaN policy |
| Paired symmetric parameters | Exact equality, NaN matches NaN, signed zeros agree |
| Opposing infinities in an owner KKT gradient pair | ArgumentError at that chosen pair |

```@example numerical_endpoints
using ProximalBase
@assert NormL1(0.0)([NaN,Inf]) == 0
@assert prox(NormL1(Inf),[NaN,Inf],1.0) == [0,0]
@assert prox(NormL1(0.0,[Inf,0.0]),[2.0,3.0]) == [0,3]
@assert isnan(shrink(NaN,1.0))
@assert isnan(prox(NormL1(1.0,[Inf]),[Inf],1.0)[1])
prox(NormL1(Inf),[NaN,Inf],1.0)
```

Signed zeros do not activate absent coordinates; explicit active signed zeros
can be retained in storage. Owner `isequal` distinguishes stored negative zero
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
`test/symmetric_products.jl` and `test/matrixview.jl`. These are concrete tested
domains; a historical acceptance goal for a removed family is not evidence for
this implementation. New numerical claims need independent calculations,
FP32/FP64 checks, exceptional inputs and regression tests.
