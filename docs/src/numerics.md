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
recomputed. For the `LinearFunction` value at a matrix `SparseIterate`, the finite
inputs are `d`, the nonzero stored entries and the coefficients at them (for a `Diagonal`
coefficient array, the nonzero stored diagonal entries and their coefficients: an off-diagonal
`NaN` or `Inf` of the iterate is ignored): the
recomputation sums their products as exponent-separated terms (as the `QuadraticFunction`
recovery does), collapses the sum and adds `d` to it in its own units, and a nonfinite coefficient at a zero of the
iterate does not prevent it. It reads the nonzero stored entries, `O(nactive(x))`; for a
`Diagonal` coefficient array it scans the `p` diagonal positions, `O(p)`, whatever
`nactive(x)` is. Recovery is worthwhile when it changes a materially inaccurate
answer into a representable result in the working precision.

The accuracy of a recovered value is an absolute bound `B = 4 n eps (sum |terms| + |d|)` for
`n` terms. It is a statement about the computed value when the exact value plus `B` is within
the range (`|exact| + B <= floatmax`). When `B`
exceeds the representable range, there is no guarantee of a finite output. Extreme cancellation
among products far beyond the output range (about `2^1077` for `Float64`, `2^150` for `Float32`
data) can return an infinity although the exact result is a finite, normal number, in the
recovery of a dense argument and in the recovery of the `LinearFunction` value at a matrix
`SparseIterate` alike ([#907](https://github.com/mlakolar/ProximalBase.jl/issues/907)). A small
`d` beside a dot that cancels exactly is returned as `d` when the compensated sum of the
products cancels to zero, which is not guaranteed for products of very different sizes.

`NormL1` values also recover terms `λ0*w[i]*abs(x[i])` that round into the
subnormal range, which loses accuracy even when the total is normal
([#373](https://github.com/mlakolar/ProximalBase.jl/issues/373)). Each such
rounding loses at most half the smallest subnormal spacing,
`eps(T)*floatmin(T)/2`, so an ordinary sum `s ≥ n*floatmin(T)` of `n` terms
loses at most `eps(T)/2` relative to `s` and is returned from the unchanged
vectorized loop. A smaller sum, a zero one included, is recomputed: the
products `w[i]*abs(x[i])` are accumulated with their exponents, and `λ0` is
applied before the single conversion to `T`. This holds for dense arrays and
for the stored entries of `SparseIterate`, `BlockIterate` and
`SymmetricSparseIterate`, and for `value_and_prox!` at its stored output.
Zero and infinite levels, infinite weights and NaN/Inf entries keep the rules
below. Under an infinite `Scaled` factor a recovered positive value gives `Inf`,
as the folded level `NormL1(Inf, w)` does; before #373 a value whose terms all
rounded to zero gave `NaN` there. An exactly zero value still gives `0`.

`ElasticNet` values are the value of the ℓ₁ part, computed as `NormL1`
computes it, plus the value of the ridge part, computed as `SqrNormL2`
computes it, each with the recovery above, then one addition; on sparse
and symmetric iterates both parts read the stored entries only, and on a
`BlockIterate` its storage. On a
dense `Float32` or `Float64` `Array` with parameters of its type, a single
pass over the entries is returned instead when a test on its sum bounds its
relative error by `(n + 2) eps(T)` for `n ≤ 1/(2 eps(T)) - 2` entries; the bound is proved in a
comment next to the code (`src/functions/elasticnet.jl`). With both levels
positive and finite, a sum of the two parts below `2 floatmin(T)` (each part
rounded on its own) is recomputed: the terms of both parts are added in one
exponent-sum accumulator, as the `NormL1` recomputation adds its terms, and
the sum is rounded once. For `N` nonzero terms (`λ1*w[i]*abs(x[i])` and
`λ2*x[i]^2/2`, up to two per entry and three per off-diagonal coordinate of a
symmetric iterate) the result is within
`(γ_3 + (γ_{N-1}^2 + N 2^-1070)(1 + γ_2)) V` of the exact value `V` plus half
a spacing of `T` at the result (`γ_m = m u/(1 - m u)`, `u` half the epsilon of
the accumulator, `Float64` for `Float32` data), as proved next to the code, which
is below the worst case of the separate parts. On Julia 1.10 `ldexp` returns `0`
for a `Float64` result in `[δ/2, δ)`, `δ` the smallest subnormal
([#896](https://github.com/mlakolar/ProximalBase.jl/issues/896)), and the final
term of the bound is a whole spacing;
`ElasticNet(2.0^-538, 1.0)([2.0^-537])` is `2.0^-1074`. Only a value whose
stored entries are all zero keeps its sum. Under `Scaled`, the value of an
`ElasticNet` is recovered as for any function with levels; only where that
gives no finite value, or `0` at a point where the function is not exactly `0`,
for a finite positive factor, is it the scaled value of the ℓ₁ part plus that of
the ridge part, each rounded on its own.
Where one part
dominates, the test can pass although the other part alone would be
recomputed (ℓ₁ terms below `floatmin(T)` beside a normal ridge part, or
squares below `floatmin(T)` beside a normal ℓ₁ part). Its proximal point soft thresholds
by the threshold of the `NormL1` kernel and multiplies by the ridge factor
`1/(1 + γλ2)` of the `SqrNormL2` kernel, both formed from the unrounded
factors, so a factor `γλ2` outside the range still acts on a representable
result.

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

A known limitation ([#567](https://github.com/mlakolar/ProximalBase.jl/issues/567)):
for `Scaled(SqrNormL2(λ2), α)` and `Scaled(ElasticNet(λ1, λ2, w), α)` with a
finite positive `α` and a scalar step `γ` for which `αγ` exceeds `floatmax`
while `αγλ2` is in range, the folded ridge level overflows, and `prox`,
`prox!`, `prox_block` and `value_and_prox!` return `0` where the result is
representable. For `ElasticNet` the same step passed as an array of steps
gives the representable result (`SqrNormL2` takes no array of steps):

```julia
s = Scaled(ElasticNet(0.0, 2.0^100), 2.0^600)
prox(s, [2.0^1000], 2.0^600)      # [0.0]
prox(s, [2.0^1000], [2.0^600])    # [4.909093465297727e-91], about 2^-300
```

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
| `NormL1`, infinite level, nonnegative step | Positive-weight entries prox to `+0`, including non-finite inputs; zero-weight entries stay free. At `γ = 0` this is the fixed-function limit `γ → 0+`, the projection onto the closure of the domain ([#665](https://github.com/mlakolar/ProximalBase.jl/issues/665)) |
| `NormL1`, finite factors whose product overflows | Distinguished from genuinely infinite weights/levels; recovery handles representable finite results where supported. With an input of `±Inf`, an overflowed finite threshold can yield `NaN`; [#866](https://github.com/mlakolar/ProximalBase.jl/issues/866) is a documented, deferred limitation, not an authorized hot-loop fix. |
| `SqrNormL2`, zero level | Zero standalone value; gradient/prox can still encounter IEEE zero-times-nonfinite arithmetic |
| `SqrNormL2`, infinite level | Gradient is the IEEE `λ(x - y)`: ±Inf for every nonzero residual, subnormal ones included, and NaN at a zero residual |
| `ElasticNet`, negative or `NaN` scalar step | `prox`, `prox!` and `value_and_prox!` throw `ArgumentError` before any write. The entries of an array of steps are not checked: nonnegative entries are a precondition, an exception to the requirement of [#908](https://github.com/mlakolar/ProximalBase.jl/issues/908) that negative and `NaN` steps fail before writes (no scan of the steps, as for #874). `prox_block` and `Scaled` routes do not check |
| `ElasticNet`, infinite ridge level `λ2`, nonnegative step | The indicator of `{0}` (#908): every entry proxes to `+0`, including at `γ = 0` and for non-finite inputs; value `0` at the zero vector, `NaN` at a `NaN` entry, `Inf` otherwise; `kkt_violation` is `-Inf`. `SqrNormL2(Inf)` keeps its own rule |
| `ElasticNet`, infinite `λ1` or infinite weight, overflowing finite threshold | The `NormL1` rows above (decision 49, #665, #853, #866) |
| `ElasticNet`, zero `λ2` | Values of `NormL1(λ1, w)`, and at `γ ≥ 0` with one element type its proximal points, bit for bit; a zero `λ2` adds `0` also at `NaN`/`±Inf` entries |
| Generic custom value/prox | Author-defined arithmetic; traits do not grant recovery or NaN policy |
| Paired symmetric parameters, in `check_symmetric_parameters` | Exact equality, NaN matches NaN, signed zeros agree; the numerical oracles assume it and do not compare |
| Opposing infinities in a symmetric iterate KKT gradient pair | ArgumentError at that chosen pair |

```@example numerical_endpoints
using ProximalBase
@assert NormL1(0.0)([NaN,Inf]) == 0
@assert prox(NormL1(Inf),[NaN,Inf],1.0) == [0,0]
@assert prox(NormL1(Inf),[1.0,2.0],0.0) == [0,0]  # zero step: projection onto {0} (#665)
@assert isequal(prox(NormL1(Inf,[1.0,0.0,2.0]),[3.0,NaN,Inf],0.0), [0,NaN,0])
@assert prox(NormL1(1.0),[1.0,2.0],0.0) == [1,2]  # finite level: zero threshold
@assert prox(NormL1(0.0,[Inf,0.0]),[2.0,3.0]) == [0,3]
@assert isnan(shrink(NaN,1.0))
@assert prox(NormL1(1.0,[Inf]),[2.0],1.0) == [0.0]  # finite input fixed at zero
@assert prox(NormL1(0.0,[Inf]),[2.0],0.0) == [0.0]  # zero level/step
@assert isnan(prox(NormL1(1.0,[Inf]),[Inf],1.0)[1])  # non-finite limitation (#853)
@assert isnan(prox(NormL1(1e300,[1e10]),[Inf],1.0)[1])  # final factor overflows (#866)
@assert isnan(prox(NormL1(1e10,[1.0]),[Inf],1e300)[1]) # intermediate product overflows (#866)
prox(NormL1(Inf),[NaN,Inf],1.0)
```

```@example numerical_endpoints
@assert prox(ElasticNet(1.0,Inf),[3.0,NaN,-Inf],0.0) == [0,0,0]  # infinite ridge level: {0}
@assert ElasticNet(1.0,Inf)([0.0,0.0]) == 0 && ElasticNet(1.0,Inf)([0.0,1e-300]) == Inf
@assert isnan(ElasticNet(1.0,Inf)([NaN,0.0]))
@assert prox(ElasticNet(Inf,1.0,[1.0,0.0]),[NaN,3.0],0.0) == [0,3]  # the NormL1 rule (#665)
@assert try prox(ElasticNet(1.0,1.0),[1.0],-1.0); false catch e; e isa ArgumentError end
prox(ElasticNet(1.0,Inf),[3.0,NaN,-Inf],0.0)
```

**Decision on `NormL1` non-finite inputs (#853 and #866).** A genuinely infinite
coordinate weight maps a *finite input value* to zero, even at zero level and step.
For inputs of `NaN` or `±Inf` with finite level, that fixed-zero guarantee is
intentionally not extended: retained soft-threshold arithmetic may return
`NaN`. Separately, a threshold made from *finite* factors can overflow to
`Inf`; with an input of `±Inf`, the current result can likewise be `NaN` even
though the mathematical threshold is finite. These examples characterize
present behavior, not additional non-finite-output guarantees. A genuinely
infinite **level** at any nonnegative step retains its distinct fixed-zero rule
for positive weights, including non-finite inputs (historical decision 49,
extended to `γ = 0` by #665 below).

**Zero step at an infinite `NormL1` level (#665, Option A).** For
`NormL1(Inf, w)`, the `γ = 0` endpoint is `lim_{γ→0+} prox_{γf}(x)` with the
function `f` held fixed. That function is the indicator of `{0}` in each
coordinate with `w[i] > 0` (an infinite `w[i]` included) and is zero in each
coordinate with `w[i] = 0`, so the limit is the projection onto the closure of
its domain: positive-weight coordinates become exactly `+0`, even for `NaN` or
`±Inf` inputs, and zero-weight coordinates are returned as soft thresholding by
a zero threshold returns them (`NaN` and `±Inf` unchanged). Unweighted
`NormL1(Inf)` therefore maps every coordinate to `+0`. The order of limits
matters: for a fixed *finite* level, `shrink(x, γλ) → x` as `γ → 0+`, but
`NormL1(Inf)` is already the constraint at every positive step, so the result
is not computed as `shrink(x, 0⋅Inf)`. Before #665 the zero step returned the
identity; results at `γ = 0` with an infinite level changed, and every other
case is unchanged. This rule is specific to the infinite *overall level* of
`NormL1`. An infinite *coordinate weight* at a finite level keeps the narrower
#853 rule above (finite inputs fixed; non-finite inputs may give `NaN`), and
finite factors whose product overflows are never treated as an infinite level
(#866). Finite levels at `γ = 0` keep the identity. A finite positive `Scaled`
factor passes the zero step through, so `2 * NormL1(Inf)` projects as well; an
infinite factor at a zero step forms the step `Inf⋅0 = NaN` in the wrapper
before `NormL1` sees it, and that wrapper step rule is unchanged (#311).
`ElasticNet` follows this rule for an infinite `λ1`, and implements the
same zero-step projection for an infinite ridge level `λ2`, the indicator
of `{0}` in every coordinate (#908); no other function is claimed to
implement a zero-step projection rule. For `NormL1` the test is made only on the
exceptional branch of the kernel (the product `0⋅Inf` is not an ordinary
number), so ordinary positive finite steps run the unchanged loop.

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

The `LinearFunction` value at a matrix `SparseIterate` follows the same rule, as
its own computation and not through `dot(c, x)`: a position that is not stored, or is
stored as `+0` or `-0`, contributes nothing and `c` is not read there, so a `NaN` or
`±Inf` coefficient at it does not reach the value, while a nonzero entry keeps
ordinary arithmetic. For a dense `c` that is not a `Diagonal`, `f(Matrix(x))` takes the
dense IEEE dot and gives `NaN` for such a coefficient.

```@example numerical_linear_sparse
using ProximalBase, LinearAlgebra
x = SparseIterate(2, 2)
x[1, 1] = 2
activate!(x, 4)                    # the entry (2, 2) is stored as a zero
c = [1.0 NaN; Inf 1.0]
f = LinearFunction(c, 0.25)
@assert f(x) == 2.25
@assert isnan(f(Matrix(x)))
f(x)
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
`test/symmetric_products.jl`, `test/matrixview.jl`, `test/linear_sparse_value.jl`
(BigFloat references and the strong-zero examples of the `LinearFunction` value at a matrix
`SparseIterate`), `test/sparse_value_cost.jl` (counts of the zeros formed and of the
coefficient entries read by that value) and
`test/numerical_kernels.jl` (BigFloat references for the shared recovery
kernels and their `SqrNormL2`/`LeastSquaresLoss`/`LinearFunction` consumers),
and `test/elasticnet.jl` and `test/elasticnet_extremes.jl` (`ElasticNet`
against 4096-bit BigFloat references and its endpoint rules). These are concrete tested
domains; a historical acceptance goal for a removed family is not evidence for
this implementation. New numerical claims need independent calculations,
FP32/FP64 checks, exceptional inputs and regression tests.
