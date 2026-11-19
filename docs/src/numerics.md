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

## [Overflow and underflow of values](@id Overflow-and-underflow-of-values)

The following range guarantees concern finite-data computations. Non-finite
inputs and penalty endpoints have separate rules below.

### [Range guarantees](@id numerical-value-range)

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
`nactive(x)` is. At a dense array, a vector `SparseIterate` and a `SymmetricSparseIterate`, the
recomputation sums the same exponent-separated terms when `d`, all coefficients and all entries of
the argument are finite. Recovery is worthwhile when it changes a materially inaccurate
answer into a representable result in the working precision.

The squared-residual recomputation that `SqrNormL2` and the least-squares oracles share has the same kind of
boundary uncertainty as the `NormL2` recomputation below: within a few eps (the relative error bound of the sum) of
the overflow threshold `floatmax(T) + ulp/2`, where `ulp` is the spacing between `floatmax(T)` and its predecessor, it
does not decide on which side the exact value lies, so a finite answer there is within the error bound of the exact
value and `floatmax(T)` or `Inf` can be returned for a value that rounds to the other. This was observed for
`SqrNormL2`, `Scaled(LeastSquaresLoss, α)(β)` and the oracles of the
[least-squares section](smooth.md#Least-squares-at-a-predictor-or-residual-that-the-caller-holds).

The `LeastSquaresLoss` value has one more recovery, which needs `β`: where the product `Xβ` overflows
although `β`, `X` and `Y` are finite, `f(β)` and `value_and_gradient!` attempt to form `Xβ` again in units of a power of
two, without recovering every representable value (the smooth-functions guide gives a counterexample). The oracles at a caller-supplied predictor or residual (`ProximalBase.value_from_predictor` and the others
described with the [smooth functions](smooth.md#Least-squares-at-a-predictor-or-residual-that-the-caller-holds))
take the predictor `z` as given, without `β`: they recover a value whose squares overflow or underflow and, for the
value alone, a difference `z - Y` that overflows, and a predictor with a non-finite entry gives a non-finite value.
For strided `Float32` and `Float64` vectors, these oracles, `f(β)` and `value_and_gradient!` also recompute a value
whose ordinary computation `α s / 2` is above `floatmax(T)/2`, so that the error of the ordinary sum of squares, which grows with
the length, does not decide between a finite value and `Inf`.

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

`NormL1Nonneg` values take the `NormL1` loops with an `Inf` term in place of the
term of a negative entry; the rules for `NaN` and `Inf` entries are in the table below, and at a
feasible point the recovery above applies to the same terms. At such a point the
value of a dense array agrees with that of `NormL1` within the error bound of the sum, and on the
stored entries of `SparseIterate`, `BlockIterate` and `SymmetricSparseIterate` the loops are
sequential and the values are equal bit for bit.
Where a positive finite level and weight have a product that rounds to zero and the entry is
`+Inf`, the value is `NaN`, as for `NormL1` (the retained product forms `0*Inf`,
[#655](https://github.com/mlakolar/ProximalBase.jl/issues/655)).

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
symmetric iterate), with `u` half the epsilon of the accumulator (`Float64` for
`Float32` data) and `N u < 1`, the hypothesis of the summation bound used, the
result is within `(γ_3 + (γ_{N-1}^2 + N 2^-1070)(1 + γ_2)) V` of the exact
value `V` plus half a spacing of `T` at the result (`γ_m = m u/(1 - m u)`), as
proved next to the code, which is below the worst case of the separate parts. On Julia 1.10 `ldexp` returns `0`
for a `Float64` result in `[δ/2, δ)`, `δ` the smallest subnormal
([#896](https://github.com/mlakolar/ProximalBase.jl/issues/896)), and the final
term of the bound is a whole spacing;
`ElasticNet(2.0^-538, 1.0)([2.0^-537])` is `2.0^-1074`. Only a value whose
stored entries are all zero keeps its sum. Under `Scaled`, the value of an
`ElasticNet` is recovered as for any function with levels. For a finite
positive factor, where that gives no finite value, or `0` at a point where the
function is not exactly `0`, the scaled values of the ℓ₁ part and of the ridge
part, each rounded on its own, are added; the sum replaces that result if it is
finite after conversion to its type, and the first result is kept otherwise.
Where one part
dominates, the test can pass although the other part alone would be
recomputed (ℓ₁ terms below `floatmin(T)` beside a normal ridge part, or
squares below `floatmin(T)` beside a normal ℓ₁ part). Its proximal point soft thresholds
by the threshold of the `NormL1` kernel and multiplies by the ridge factor
`1/(1 + γλ2)` of the `SqrNormL2` kernel, both formed from the unrounded
factors, so a factor `γλ2` outside the range still acts on a representable
result.

`NormL2` values recompute a sum of squares that overflows or falls below the normal
range from the squared-residual kernel above, in units of the largest stored entry.
On a `SparseIterate` and a `SymmetricSparseIterate` both the ordinary sum and that
recomputation visit the stored entries only, each off-diagonal symmetric coordinate
twice ([#808](https://github.com/mlakolar/ProximalBase.jl/issues/808)).
`NormL2` and `GroupNormL2` values, and their fused values, recompute a value at or above `floatmax(T)/2`
in power-of-two units, with sums accumulated in at least `Float64`. Within a few eps of
the overflow threshold `floatmax(T) + ulp/2` that recomputation does not decide on
which side the exact value lies, so a value that rounds to `floatmax(T)` may be
returned as `Inf`, or the reverse: the same limit as the other recoveries above
([#904](https://github.com/mlakolar/ProximalBase.jl/issues/904)).

For `IndBallL2`, the value and projection compare the norm with the radius in
scaled units when the ordinary sum of squares overflows or is too small.
The domain has no fixed feasibility tolerance. Within the rounding error of
the norm, the comparison can fall on either side of the boundary.
`value_and_prox!` evaluates the stored point. Its value can be `Inf` when
rounding of the projection puts the stored point outside the computed
boundary. A positive finite radius projects a NaN input to an all-NaN output,
including implicit zeros of a `SparseIterate`.

For `IndBallL1`, the value and projection compare the computed sum of
magnitudes with the radius rounded to the floating type `T` of the argument.
There is no feasibility tolerance. Rounding can decide a point at the
boundary. For example, the computed sum for `[1.0, 2.0^-54]` equals `1.0`,
and the projection at radius `1.0` copies that point.

Finite sums above `floatmax(T)/2` are compared in power-of-two units. The
shared ℓ₁-ball threshold from `NormLinf` selects the soft-threshold level.
A tiny radius can be lost when that level rounds to the largest magnitude.
The fallback sorts the magnitudes and forms the output directly from their
gaps at `O(n log n)` cost. The exceptional path is selected when the
radius is at most `64 eps(T)` times the computed sum in the working units
or when a level search is needed at a subnormal radius. A radius tiny
relative to the largest magnitude goes directly to the fallback. The
other cases on this path check the tentative projected mass before writing.
On the sorted path for a subnormal radius, the gaps and remaining radius
are combined in scaled units and rounded back when storing the output.
With `M = floatmax(Float64)` and radius `1`, the projection of `[M, -M]`
is `[0.5, -0.5]`.

For an exact normal threshold, rounding the threshold alone changes the
projected mass by at most `u S/(1-u)`, where `u = eps(T)/2` and `S` is the
exact sum of the magnitudes in the working units. The displacement is at
most `u` times the exact threshold per affected coordinate, and each
affected magnitude exceeds the smaller of the two thresholds.
Outside the mass screen, this contribution relative to the radius is bounded
by `(S/s)/(128(1-u))` for the computed sum `s`. The bound isolates threshold
rounding. The selector's absolute error, sum rounding, and subnormal
rounding retain their own effects.

`value_and_prox!` evaluates the stored point. Its value can be `Inf` when
rounding puts that point outside the computed boundary. For example, with
`δ = nextfloat(0.0)`, radius `3δ` projects `[M, -M]` to `[2δ, -2δ]`.

`GroupNormL2` values apply the same check to their group terms `λ0*w[k]*‖x[G_k]‖₂`,
and recompute a sum below `K*floatmin(T)` for `K` groups from the terms' exponents.
Its proximal operators recompute, for finite inputs, a group norm whose sum of
squares overflows or underflows and a threshold `γ*λ0*w[k]` outside the range, in
power-of-two units; the ordinary path needs a normal threshold and a normal sum of
squares, and is otherwise unchanged.

The clip level `τ` of `NormLinf` has an **absolute** error. On the ordinary finite path, for `n ≥ 1` entries in the
floating type `T` of the output with `n eps(T) ≤ 1/2` (up to `2^22` entries in `Float32` and `2^51` in `Float64`), a
hypothesis that the code does not check, the level returned by the threshold satisfies
`|τ - τ*| ≤ 2.25 n eps(T) max|xᵢ| + 2nδ`, where `τ*` is the root of `Σ(|xᵢ| - τ)₊ = a` for the entries
`abs(convert(T, xᵢ))` and the radius `a` that the prox forms (the product `γλ` rounded in the promoted type of `γ`, `λ`
and `T`, then to `T`), and `δ = nextfloat(zero(T))` (the storage of a subnormal radius or output adds `δ`; `x = [δ]`,
`λ = δ`, `γ = 0.5` gives `[δ]` where the exact point is `δ/2`). The prox clips at the larger of the level and zero,
which is within the same bound of `max(τ*, 0)`. The derivation, in `src/functions/normlinf.jl`, bounds the excess
`Σ(|xᵢ| - τ)₊ - a` for each of the four ways the threshold search ends, and it does not assume that the entries it found
above the level are the entries above the exact level. Unlike the relative bounds of the `NormL2` and `GroupNormL2`
kernels, the bound is absolute: an entry far below `max|xᵢ|` is not clipped to a relative accuracy. A sum of magnitudes
that overflows or exceeds `floatmax(T)/2` is processed in power-of-two units: the statement above then applies first to
the scaled entries and the scaled radius, as they are rounded, and converting back carries the scaling allowances
described here (the radius `γλ` rounded to `T` can itself be infinite there, as for `x = [M, M]`, `λ = 1.5`, `γ = M`
with `M = floatmax(T)`, which is not the reference of the bound). The subnormal losses of the scaling are below the
nominal scale `n eps(T) max|xᵢ|` under the same hypothesis on `n`, and the scaling of an entry or of the radius changes
it by at most one subnormal spacing (the allowance covers the `ldexp` of Julia 1.10 noted above), a second hypothesis of
that argument that the code does not check.
Through `Scaled`, the wrapper's recovery of a radius `γλα` beyond the range has a limit: for
`Scaled(NormLinf(2.0 * 2.0^-1000), 2.0^1000)` at the step `0.75 floatmax(Float64)` on two entries equal to
`floatmax(Float64)`, even the folded level at half the scale is out of range, the wrapper passes an infinite step, and the
output is `+0` where the exact result is about `floatmax/4`. For positive
finite level and step, an infinite entry without `NaN` gives the identity; on the finite clipping branch the value is
`λ τ`, computed from the largest stored magnitude `τ`, and `value_and_prox!` returns the value of the stored output,
computed as `f(y)` computes it. Endpoint precedence is given in the table below.

These are practical numerical safeguards, not a promise of exhaustive
range recovery or correctly rounded arbitrary reductions, gradients, or
matrix products. When the final mathematical value itself is out of range,
an appropriate infinity or underflow to zero may be unavoidable, subject
to the function's endpoint semantics. There is no cross-platform or
cross-version bitwise arithmetic-order guarantee.

Routes that compute the same proximal point, such as the array `prox!` and the scalar `prox_block`, are not guaranteed
to agree bit for bit. For homogeneous `Float32` or `Float64` inputs with `γλ` between `floatmin(T)` and `2^125`
(`Float32`) or `2^1021` (`Float64`), the centered `SqrNormL2` proximal point evaluates the coordinates whose argument
and center entries have the same sign bit with `muladd`, which the compiler may or may not contract into one rounding,
depending on how it optimizes each call site and on the CPU. On Julia 1.13.4 and an x86-64 CPU with fused multiply-add,
with `γλ < 1`, `prox_block` was observed to contract it and the vectorized loop of `prox!` not to, so the two differed
in the last place. Both are rounded evaluations of the same formula; the tests compare the two routes within one unit in
the last place on selected inputs.

The order in which an ordinary value adds its terms is not part of the contract either. The unweighted
`NormL1` value of a dense `Float32` or `Float64` array of 16 to 1024 entries, and the `NormL2` value of such an
array or strided view, add their terms in one vectorized loop. Julia's `sum` is also one vectorized loop up to
1024 entries, so the rounding error is of the same kind and grows with the length in the same way, but the result
can differ from `λ0 * sum(abs, x)` or `λ * sqrt(sum(abs2, x))` in the last place. The tests bound these values
against a high-precision sum and do not compare their bits; shorter and longer arrays use `sum`.

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
for custom functions.

A known limitation ([#567](https://github.com/mlakolar/ProximalBase.jl/issues/567)):
for `Scaled(ElasticNet(λ1, λ2, w), α)` with finite levels, a finite positive `α`
and a positive finite scalar step `γ` whose product `αγ` is above `floatmax`,
`prox`, `prox!`, `prox_block` and `value_and_prox!` can return `0` where the
exact result is representable. A representable result does not need a
representable coefficient. With `λ1 = 0`, `α = γ = 2.0^600` and `x = [2.0^1000]`,
the coefficient `αγλ2` is `2^1300` for `λ2 = 2.0^100`, above `floatmax`,
while the quotient `x/(1 + αγλ2)`, about `2^-300`, is representable, and the
result is `0`. For `λ2 = 2.0^-1000` and `x = [4.0]` the coefficient is `2^200`
and the result is the representable `4/(1 + 2^200)`.
`Scaled(SqrNormL2(λ2), α)` also returns `0` in the first case.
In some cases the same step passed as an array of steps gives the
representable result, for example in the first case and in its `Float32`
analogue (`λ2 = 2.0f0^80`, `α = γ = 2.0f0^80`, `x = [2.0f0^127]`); for
`λ2 = 2.0^880` and `x = [2.0^1023]`, with the same `α` and `γ`, both steps
give `0` where the result is `2^-1057`:

```julia
s = Scaled(ElasticNet(0.0, 2.0^100), 2.0^600)
prox(s, [2.0^1000], 2.0^600)      # [0.0]
prox(s, [2.0^1000], [2.0^600])    # [4.909093465297727e-91], about 2^-300
t = Scaled(ElasticNet(0.0, 2.0^880), 2.0^600)
prox(t, [2.0^1023], 2.0^600)      # [0.0]
prox(t, [2.0^1023], [2.0^600])    # [0.0]; the result is 2^-1057
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
| `NormL1`, infinite level, nonnegative step | Positive-weight entries prox to `+0`, including non-finite inputs; zero-weight entries stay free. See [the zero-step rule](@ref infinite-penalty-at-zero-step) |
| `NormL1`, finite factors whose product overflows | Distinguished from genuinely infinite weights/levels; recovery handles representable finite results where supported. With an input of `±Inf`, an overflowed finite threshold can yield `NaN`; [#866](https://github.com/mlakolar/ProximalBase.jl/issues/866) is a documented, deferred limitation, not an authorized hot-loop fix. |
| `NormL2` | The rules of one `GroupNormL2` group with weight one (rows below); value `0` at zero level; no proximal operator for a `SymmetricSparseIterate` (`ArgumentError` before any write) |
| `NormLinf`, negative or `NaN` step | Checked first: `ArgumentError` before any write, ahead of the rows below, also through `Scaled` (a size mismatch is a `DimensionMismatch`); no proximal operator for a `SymmetricSparseIterate` (`ArgumentError` before any write, also through `Scaled`) |
| `NormLinf`, infinite level, nonnegative step | `+0` at every entry, `γ = 0` and `γ = Inf` included (#665 reading, #946), also for `NaN`/`±Inf` inputs; value `0` at the zero vector (an empty one included), `Inf` at any other argument without a `NaN`, `NaN` at a `NaN` entry |
| `NormLinf`, finite level, zero level or zero step | After the rows above: the identity (`-0`, `NaN` and `±Inf` entries unchanged), also at `γ = Inf` with a zero level; value `0` at a zero level, `λ maxᵢ abs(xᵢ)` at a zero step of a positive level |
| `NormLinf`, positive finite level, infinite step | `+0` for every input (decided before the entries are classified), `NaN` and `±Inf` entries included |
| `NormLinf`, positive finite level and positive finite step | A `NaN` entry makes every output `NaN`; a `±Inf` entry (no `NaN`) leaves the argument unchanged, also when the finite product `γλ` overflows; otherwise the clip at `τ` with `Σ(abs(xᵢ) - τ)₊ = γλ`, and `+0` when `γλ` is at least `‖x‖₁` or overflows while `‖x‖₁` is finite; a sum of magnitudes above `floatmax(T)/2` is processed in power-of-two units |
| `GroupNormL2`, zero weight | Unpenalized group: returned unchanged (`NaN`/`±Inf` included), zero value |
| `GroupNormL2`, infinite weight, or infinite level with a positive weight | Group fixed at `+0` at every nonnegative step, `γ = 0` included (see [zero-step rule](@ref infinite-penalty-at-zero-step)), also for `NaN`/`±Inf` inputs; value `0` at zero, `Inf` elsewhere (`NaN` at a `NaN` entry) |
| `GroupNormL2`, positive finite parameters and step | A `NaN` entry makes the group `NaN`; a `±Inf` entry (no `NaN`) leaves the group unchanged, also when the finite threshold's product overflows; `γ = Inf` fixes the group at `+0` |
| `GroupNormL2`, negative or `NaN` step | `ArgumentError` before any write, in `prox!`, `value_and_prox!` and `prox_block!` |
| `SqrNormL2`, zero level | Zero standalone value; gradient/prox can still encounter IEEE zero-times-nonfinite arithmetic |
| `SqrNormL2`, infinite level | Gradient is the IEEE `λ(x - y)`: ±Inf for every nonzero residual, subnormal ones included, and NaN at a zero residual; the four-argument and full-point `kkt_violation` scores of `SqrNormL2` take their gradient entry from the ridge-gradient helpers and follow this rule for finite `v` and `x` (the non-finite rules of the `kkt_violation` rows below take precedence, at an infinite level too: `SqrNormL2(Inf)`, `v = [Inf]`, `x = [0.0]` scores `+Inf`), while the three-argument screen does not use the gradient (`kkt_violation(SqrNormL2(Inf), [1.0], 1)` is `1.0`) |
| `ElasticNet`, negative or `NaN` scalar step | `prox`, `prox!` and `value_and_prox!` throw `ArgumentError` before any write. The entries of an array of steps are not checked: nonnegative entries are a caller precondition, a documented exception to [#908](https://github.com/mlakolar/ProximalBase.jl/issues/908)'s preflight rule. `prox_block` and `Scaled` routes do not check. The historical reasoning is in [numerical endpoint history](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-endpoint-history-921.md) |
| `IndBox` | Exact closed intervals, with no feasibility tolerance. A `NaN` entry gives value `NaN` and stays `NaN` under projection. An infinite entry is feasible only at an unbounded endpoint in its direction. The projection clamps each entry independently of the step and the fused value is evaluated at the stored result. Public scalar-step array oracles reject negative and `NaN` steps before writes. |
| `IndNonnegative` | Value `0` at a point with no negative and no `NaN` entry (`-0.0` and `+Inf` included), `Inf` at a negative entry (`-Inf` included) when no entry is `NaN`, `NaN` at a `NaN` entry. The projection is `max(v, 0)` at a step `γ ≥ 0`: `-0.0` and `-Inf` give `+0`, `NaN` stays `NaN`. A negative or `NaN` scalar step throws `ArgumentError` before any write in `prox`, `prox!` and `value_and_prox!`; array-step entries, `prox_block` and `Scaled` routes do not check and project. `value_and_prox!` returns the value at the stored output: `NaN` at a `NaN` output entry, otherwise `0`. The tolerance of the archived value (`v >= -_tol`) is not kept: a negative entry of any size gives `Inf` |
| `IndBallL2` | Projects at any nonnegative scalar step, including zero and infinity. A negative or NaN step throws before writes, including through scalar `Scaled`. A NaN entry gives value NaN at any radius. Zero radius projects to positive zeros. Infinite radius copies the input, and the value is zero if the input has no NaN. At positive finite radius, a single infinite entry gives the signed radius at that coordinate and zeros elsewhere. Multiple infinite entries or any NaN give an all-NaN projection. |
| `IndBallL1` | The same step, radius and nonfinite rules as `IndBallL2`: any nonnegative scalar step projects, a negative or NaN step throws before writes (including through scalar `Scaled`), a NaN entry gives value NaN, zero radius projects to positive zeros, infinite radius copies the input, and at positive finite radius a single infinite entry gives the signed radius there and zeros elsewhere, while multiple infinite entries or any NaN give an all-NaN projection. A feasible point is copied with its signed zeros. Entries at or below the threshold level become `+0`. |
| `ElasticNet`, infinite ridge level `λ2`, nonnegative step | The indicator of `{0}` (#908): every entry proxes to `+0`, including at `γ = 0` and for non-finite inputs; value `0` at the zero vector, `NaN` at a `NaN` entry, `Inf` otherwise; the screen `kkt_violation(f,v,i)` is `-Inf`, as is `kkt_violation(f,v,x,i)` at a zero entry of `x` with a finite `v[i]`. `SqrNormL2(Inf)` keeps its own rule |
| `ElasticNet`, infinite `λ1` or infinite weight, overflowing finite threshold | The `NormL1` rows above and [zero-step rule](@ref infinite-penalty-at-zero-step) |
| `ElasticNet`, zero `λ2` | Values of `NormL1(λ1, w)`, and at `γ ≥ 0` with one element type its proximal points, bit for bit; a zero `λ2` adds `0` also at `NaN`/`±Inf` entries |
| `NormL1Nonneg`, domain | The domain is `x ≥ 0` exactly: `-0.0` and `+Inf` are feasible, any negative number, a subnormal one included, is not. The value is `Inf` where an entry that is read is negative and `NaN` where one is `NaN` (`NaN` takes precedence); a `+Inf` entry whose positive finite coefficient `λ0 w` rounds to zero also gives `NaN`, as for `NormL1` ([#655](https://github.com/mlakolar/ProximalBase.jl/issues/655)), and then takes precedence over the `Inf` of a negative entry (`NormL1Nonneg(5.0e-324,[5.0e-324,1.0])([Inf,-1.0])` is `NaN`); a zero level or weight keeps the domain. The value is `0` at a feasible point, `+Inf` entries included, for a zero weight and for a zero level with a finite weight; an infinite weight at a zero level has the value of its own row below (`NormL1Nonneg(0.0,[Inf])([1.0])` is `Inf`). A proximal point is `+0`, positive or `NaN`: `-0.0` gives `+0.0` |
| `NormL1Nonneg`, zero weight; zero level with a finite weight; zero step with a finite level and weight | The projection `max(v, 0)` onto the orthant, not the identity: `-Inf` and `-0.0` give `+0`, `+Inf` and `NaN` are kept. A zero weight projects at every level and step, an infinite level and an infinite step included. An infinite weight, and an infinite level with a positive weight, follow their own rows |
| `NormL1Nonneg`, infinite level, nonnegative step | Positive-weight entries prox to `+0` at every step, `γ = 0` included, for non-finite inputs too (the `NormL1` rule, #665); zero-weight entries are projected. The value of a positive-weight entry is `0` at `0`, `NaN` at `NaN` and `Inf` otherwise |
| `NormL1Nonneg`, infinite weight, finite level | Finite inputs and `-Inf` map to `+0`, also at a zero level or step; `+Inf` and `NaN` map to `NaN`, as for `NormL1` ([#853](https://github.com/mlakolar/ProximalBase.jl/issues/853)). The value is `0` at `0`, `Inf` at any other number, `NaN` at `NaN`, also at a zero level |
| `NormL1Nonneg`, finite factors whose threshold overflows, or an infinite step with a positive finite level and a positive finite weight | Finite inputs and `-Inf` give `+0`; `+Inf` gives `NaN` (as `NormL1`, deferred #866); `NaN` gives `NaN`. An infinite step with a zero level or a zero weight is the projection: `prox(NormL1Nonneg(0.0),[3.0],Inf)` is `[3.0]` |
| `NormL1Nonneg`, negative or `NaN` scalar step | `prox`, `prox!` and `value_and_prox!` throw `ArgumentError` before any write; the entries of an array of steps are not checked, and `prox_block` and `Scaled` routes do not check. A negative step through `Scaled` violates the precondition `γ ≥ 0`, and its result can depend on the representation |
| `kkt_violation(f,v,x,k)` and `kkt_violation(f,v,x)`, non-finite `v` or `x` (`NormL1`, `NormL1Nonneg`, `ElasticNet`, `ZeroFunction`, `LinearFunction`, `SqrNormL2`) | The entries of `v`, and the entries of `x` that the formula reads, are converted to the working type `W` first (see the element-types row), and the rules apply to the converted entries. For the entries of `v` and the entries of `x` that the formula reads (`ZeroFunction` and `LinearFunction` read none of `x`): a `NaN` gives `NaN`; otherwise an infinite entry gives `+Inf`. In the full-point form a `NaN` takes precedence over an infinite entry at another position. These rules replace the value of the three-argument screen where that would be `-Inf` (an infinite `NormL1`, `NormL1Nonneg` or `ElasticNet` coefficient, an infinite `ElasticNet` ridge level, or `v[i] = +Inf` for `NormL1Nonneg`) or `NaN` (`Inf - Inf`) for a non-finite `v[i]`, so that a call that reads the point does not report slack or zero where stationarity is undefined |
| `kkt_violation(f,v,x,k)` and `kkt_violation(f,v,x)`, finite `v` and `x` (after conversion to `W`), exceptional parameters | The formula is evaluated in floating point in `W`. For `NormL1`, `NormL1Nonneg` and `ElasticNet`, a zero weight gives a zero coefficient at any level, and an infinite coefficient (an infinite weight, an infinite level with a positive weight, or a coefficient whose product overflows in `W`, as in `NormL1(floatmax(Float64), [2.0])`) gives `-Inf` at a zero entry and `+Inf` at a nonzero entry; the full-point form takes the maximum with `0` over the entries, so a `-Inf` score of a zero entry does not raise it (the full-point value is `0` when no other entry has a positive score, as in that example). `NormL1Nonneg` scores a negative entry `x[i]` `+Inf` at any coefficient, since the point is outside the domain. An infinite `ElasticNet` ridge level does the same, and a ridge product `λ2 x[i]` that overflows at a nonzero entry gives `+Inf`. For `LinearFunction` and `SqrNormL2` the score is `abs(v[i] + c[i])` or `abs(v[i] + d[i])` in floating point, so an infinite `LinearFunction` coefficient gives `+Inf` (`kkt_violation(LinearFunction([Inf]), [0.0], [0.0], 1)` is `+Inf`), a `NaN` coefficient of a `LinearFunction` or a `NaN` center of a `SqrNormL2` gives `NaN`, and `SqrNormL2` with an infinite level follows the gradient oracle row above. `ZeroFunction` has no parameter |
| `kkt_violation(f,v,x,k)` and `kkt_violation(f,v,x)`, element types | The working type `W` is `float(promote_type(eltype(v), eltype(x)))` for `NormL1`, `NormL1Nonneg`, `ElasticNet`, `ZeroFunction` and a `SqrNormL2` without a center, and `float(promote_type(eltype(v), eltype(x), eltype(c)))` for a `LinearFunction` with coefficients `c` and a `SqrNormL2` with a center `c` (`float` is applied after the promotion: integer `v` and `x` with a `Float32` `c` give `Float32`). The levels and weights of `NormL1`, `NormL1Nonneg` and `ElasticNet` and a scalar `SqrNormL2` level do not enter `W`. The entries of `v`, and the entries of `x` that the formula reads, are converted to `W` before the zero test, the non-finite rules and the formulas, so each statement about a zero, nonzero, finite or infinite entry is about the converted value: `kkt_violation(NormL1(0.5), Float16[0], [1//10^12], 1)` is `-0.5` because the entry converts to zero, and an integer `x = [10^6]` with a `Float16` `v` converts to `Inf` and scores `+Inf`. For `NormL1`, `NormL1Nonneg` and `ElasticNet` the coefficient `λ w[i]` (`λ1 w[i]` for `ElasticNet`) is formed by the helper of the three-argument screen and rounded once to `W`, not as a product of factors rounded to `W` first (`NormL1(1.00000006, [1.00000006])`, `Float32` `v = [0]` and `x = [1]` gives `1.0000001f0`, while the product of the two factors rounded to `Float32` first gives `1.0000002f0`); `Bool` factors, such as an omitted weight, do not widen the type in which the helper forms it. When `eltype(v)` is a floating type, the three-argument screen forms its coefficient the same way with `W = eltype(v)`, so with a wider `x` the coefficient of the point-aware forms can differ from it (`NormL1(1/3)`, `Float32` `v`, `Float64` `x`). For another real element type of `v` (integers), the three-argument screen keeps the coefficient computed from the level and the weight in their own types (`kkt_violation(NormL1(0.5), [0], 1)` is the `Float64` `-0.5`, and with `NormL1(0.5f0)` it is a `Float32`); the same holds for `NormL1Nonneg` and `ElasticNet`. The level `λ2` of `ElasticNet` is converted to `W` with no further range recovery. For finite converted inputs the score of `NormL1Nonneg` is `-v[i] - c[i]` at a zero entry, `abs(v[i] + c[i])` at a positive entry and `Inf` at a negative one. Its three-argument screen forms `-v[i] - c` in `P = promote_type(typeof(v[i]), typeof(c))`, or in `float(P)` when `P` is an unsigned integer type; otherwise the arithmetic of `P` applies (a signed integer type wraps at `typemin`). At a nonzero entry the arithmetic is `abs(v[i] + c[i] sign(x[i]))` for `NormL1` and `abs((v[i] + λ2 x[i]) + c[i] sign(x[i]))` for `ElasticNet`, with the term `λ2 x[i]` left out when `λ2` is zero. A `SqrNormL2` gradient entry computed in a wider type (a level or a center that does not fit `W`) is rounded back to `W` before it is added to `v[i]`. For `Float32` or `Float64` arrays and parameters of one type and a finite `v[i]`, the score at a zero entry equals the three-argument screen bit for bit (a `SqrNormL2` with an infinite level excepted). Near a stationary point the terms cancel, so the error of the score need not be small relative to the score |
| `kkt_violation(f,v,x,k)` and `kkt_violation(f,v,x)`, `GroupNormL2` and `NormL2`: element types, zero test, coefficient | The working type `W` is `float(promote_type(eltype(v), eltype(x)))`, and the result has type `W` in each branch, an entry in no group included (`abs` of the entry converted to `W`). The entries of `v` and `x` are converted to `W` before they are classified or measured: a block is zero when all its entries are zeros of either sign after the conversion (`-0.0` is zero, a subnormal entry is not), and an entry that does not fit `W` is infinite there (an integer `10^6` with a `Float16` `v` scores `+Inf`). The coefficient `c = λ0 w[k]` (`λ w` with the weight one for `NormL2`) is formed by the helper of the three-argument screen and rounded once to `W`: it is `0` for a zero weight at any level, and `Inf` for an infinite weight, for an infinite level with a positive weight, and where the product overflows in `W`. A nonzero block uses the coefficient rounded to `W`. A zero block reuses the existing screen on `v` read in `W`, including the recovery of differences that involve finite out-of-range products; for `v`, `x` and the parameters of one floating type and a finite `v`, its value equals the screen of earlier versions bit for bit |
| `kkt_violation(f,v,x,k)` and `kkt_violation(f,v,x)`, `GroupNormL2` and `NormL2`: non-finite entries and coefficients | In this order, for the entries of `v_G` and `x_G` converted to `W`: a `NaN` gives `NaN`; otherwise an infinite entry gives `+Inf`. Then, at a nonzero block, an infinite `c` gives `+Inf` and `c = 0` gives `norm(v_G)`. At a zero block the rules of the screen stay, with one exception: an infinite entry of `v_G` with an infinite `c` is `+Inf`, where the screen is `NaN` (`Inf - Inf`). For a finite `v_G`, an infinite level or weight (with a positive weight) gives `-Inf` at a zero block; where `c` overflowed to infinity from finite parameters the screen recovers the difference from the finite factors, and the value can be finite (`GroupNormL2(floatmax(Float64), [[1]], [1.5])`, `v = [floatmax(Float64)]` and `x = [0.0]` give about `-floatmax(Float64)/2`). The full-point form keeps a `NaN` in its running maximum, so a `NaN` takes precedence over `+Inf` at another group or entry, and takes the maximum with zero of `W` otherwise (`-Inf` scores of zero blocks do not raise it). An entry in no group has `abs` of the converted entry, with `NaN` giving `NaN` and an infinite entry `+Inf` |
| `kkt_violation(f,v,x,k)` and `kkt_violation(f,v,x)`, `GroupNormL2` and `NormL2`: accuracy | In the ordinary range, which is when the sums of squares of `x_G` and of the residual are finite and at least `n floatmin(W)/eps(W)` for a block of `n` entries and `c/norm(x_G)` is finite, a nonzero block is computed after the scan for a zero block by two arithmetic passes: the norm of the point (`t = c/√Σxⱼ²`), then the norm of the residual (`√Σ(vⱼ + t xⱼ)²`). To first order in `n u`, with `u = eps(W)/2`, its error is at most `(n + 8) u (norm(v_G) + c)`, where `c` is the coefficient as rounded to `W`. Rounding `λ0 w` to `W` adds a relative `u` for a normal finite product and an absolute term of the subnormal spacing where the product is subnormal or rounds to zero. The bound is relative to the size of the terms that cancel near stationarity, not to the score. The tests use this bound, plus the subnormal spacing of `W`, for zero blocks as well. The test of the ordinary range is on the sums, not on single entries: a block with a subnormal entry whose sums are ordinary stays on the ordinary path. Outside it (sums that overflow or underflow, a very large `c`) the block is computed in the recovery type (at least `Float64`) from the entries read in `W`, with the norm of `x_G` and the factor `c/norm(x_G)` kept as a significand and an exponent: this extends the range of the computation, not its precision, and the result is rounded once to `W`. The residuals are scaled by the exponent of the largest one so that the accumulated squares do not overflow; a scaled square far below the largest can underflow below the rounding scale of the norm. The `Float32` and `Float64` calls on arrays allocate nothing in either range (measured by the tests) |
| `kkt_violation(f,v,x,k)` and `kkt_violation(f,v,x)`, `Scaled` | For a finite positive `α` the call evaluates the existing score of the function on the lazy gradient `v/α`, with the rounding and range limits of that evaluation (the leaf's gradient entry or its sums can overflow or underflow although `1/α` and `v[i]/α` do not), and multiplies it by `α`; underflow, overflow and `NaN` are not recovered. The reciprocal of the factor takes part in the type promotion (an `Int` factor introduces `Float64`; `BigFloat` data stay `BigFloat`). Nested wrappers whose product of factors is not a normal number, an infinite factor, and the functions without a score are listed in the [scaling guide](@ref scaled-kkt). A stack that nests an infinite factor with another `Scaled` wrapper (a direct call of the parameterized constructor `Scaled{F,T}` builds it; `Scaled(f, α)` merges the factors) has unspecified three-argument and zero-coordinate values and throws an `ArgumentError` at a nonzero coordinate or group and in the full-point form |
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

```@example numerical_endpoints
@assert isequal(prox(NormL1Nonneg(0.0),[-2.0,-0.0,3.0,Inf,-Inf,NaN],1.0), [0.0,0.0,3.0,Inf,0.0,NaN])  # the projection
@assert prox(NormL1Nonneg(Inf,[1.0,0.0]),[NaN,-2.0],0.0) == [0,0]   # the NormL1 rule (#665), a zero weight projects
@assert NormL1Nonneg(0.0)([Inf,1.0]) == 0 && NormL1Nonneg(0.0)([-1.0]) == Inf
@assert isnan(NormL1Nonneg(2.0)([NaN,-1.0])) && NormL1Nonneg(2.0)([1.0,-1e-300]) == Inf
@assert try prox(NormL1Nonneg(1.0),[1.0],-1.0); false catch e; e isa ArgumentError end
prox(NormL1Nonneg(0.0),[-2.0,-0.0,3.0,Inf,-Inf,NaN],1.0)
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
for positive weights, including non-finite inputs; see [the zero-step rule](@ref infinite-penalty-at-zero-step).

### Cardinality penalty endpoints

For `NormL0(λ)` at positive finite `λ` and `γ`, compare the squared
distance cost of setting a finite input `v` to zero with the penalty cost
of keeping it. The comparison uses the exact products of the represented
homogeneous `Float32` or `Float64` inputs. A tie gives positive zero,
including when the candidate costs fall outside the floating-point range.

| Level and valid step | Proximal entry |
|:--|:--|
| `λ = 0`, including at `γ = Inf` | Identity |
| `λ = Inf`, including at `γ = 0` | Positive zero |
| `0 < λ < Inf`, `γ = 0` | Identity |
| `0 < λ < Inf`, `γ = Inf` | Positive zero |
| `0 < λ < Inf`, `0 < γ < Inf` | Hard thresholding for finite inputs. NaN and signed infinities pass through |

At a zero level the value is zero, including for NaN input. At a positive
level, a NaN entry gives a NaN value ([E2, #560](https://github.com/mlakolar/ProximalBase.jl/issues/560)).
Either infinity is a nonzero entry. With fixed positive finite level and
step, the signed infinity output follows the limit from finite inputs
([E3, #469](https://github.com/mlakolar/ProximalBase.jl/issues/469)).
The constant-zero map at an infinite step is the endpoint convention.
The value sums the entry contributions in the result precision, which
can incur rounding.

Direct array oracles reject negative and NaN steps before writing.
Array steps are matched by linear index and checked in full, with work
proportional to their length. A scalar step is required for a symmetric
sparse iterate output. Value and scalar-step proximal calls between
same-kind sparse iterates traverse stored entries, with the value
counting each symmetric off-diagonal entry twice. A sparse iterate can
omit a zero from storage and read it back as positive zero. Fused calls
return the value at the stored output. Use a nonnegative, non-NaN step
with `prox_block` and in calls through `Scaled`.

### [Infinite penalties at zero step](@id infinite-penalty-at-zero-step)

For `NormL1(Inf, w)`, the `γ = 0` endpoint is `lim_{γ→0+} prox_{γf}(x)` with the
function `f` held fixed. That function is the indicator of `{0}` in each
coordinate with `w[i] > 0` (an infinite `w[i]` included) and is zero in each
coordinate with `w[i] = 0`, so the limit is the projection onto the closure of
its domain: positive-weight coordinates become exactly `+0`, even for `NaN` or
`±Inf` inputs, and zero-weight coordinates are returned as soft thresholding by
a zero threshold returns them (`NaN` and `±Inf` unchanged). Unweighted
`NormL1(Inf)` therefore maps every coordinate to `+0`. The order of limits
matters: for a fixed *finite* level, `shrink(x, γλ) → x` as `γ → 0+`, but
`NormL1(Inf)` is already the constraint at every positive step, so the result
is not computed as `shrink(x, 0⋅Inf)`. Among the `NormL1` parameters, this rule
is specific to the infinite *overall level*. An infinite *coordinate weight* at a finite level keeps the narrower
#853 rule above (finite inputs fixed; non-finite inputs may give `NaN`), and
finite factors whose product overflows are never treated as an infinite level
(the finite-threshold limitation above). For finite levels and finite weights,
`γ = 0` keeps the identity. A finite positive `Scaled`
factor passes the zero step through, so `2 * NormL1(Inf)` projects as well; an
infinite factor at a zero step forms the step `Inf⋅0 = NaN` in the wrapper
before `NormL1` sees it, and follows the wrapper's existing step arithmetic.
`ElasticNet` follows this rule for an infinite `λ1`, and implements the
same zero-step projection for an infinite ridge level `λ2`, the indicator
of `{0}` in every coordinate. `NormL1Nonneg` applies it to an infinite level (positive-weight entries become `+0`; zero-weight entries are projected onto the orthant, not left unchanged). `GroupNormL2` applies the same reading at `γ = 0` to an infinite level (groups with
a positive weight become `+0`) and to an infinite group weight. `NormL0`, `NormL2` and `NormLinf` apply
it to an infinite level. No other function
is claimed to implement a zero-step projection rule.

The [endpoint history](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-endpoint-history-921.md)
records the design decisions and performance evidence.

Signed zeros do not activate absent coordinates; explicit active signed zeros
can be retained in storage. Symmetric-iterate `isequal` distinguishes stored negative zero
from absent positive zero; `==` does not. `dropzeros!` removes either signed zero
and preserves NaN/Inf.

## [Sparse product semantics](@id Sparse-product-semantics)

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
`test/elasticnet.jl` and `test/elasticnet_extremes.jl` (`ElasticNet`
against 4096-bit BigFloat references and its endpoint rules),
`test/norml1nonneg.jl` and `test/norml1nonneg_extremes.jl` (`NormL1Nonneg` against
golden-section and 4096-bit BigFloat references, its endpoint rules and its `Scaled` routes), `test/indnonnegative.jl` (`IndNonnegative`
against references written with comparisons: the value rule, the projection, the steps, the fused value at the stored
output, `Scaled`, and a counting element type for the stored entries of sparse and symmetric iterates), and
`test/kkt_point.jl` (the scores of `kkt_violation` for the five entrywise families against
BigFloat references written from the definition, and the non-finite rules above), `test/kkt_block.jl` (the block scores of
`GroupNormL2` and `NormL2` and the scores of `Scaled` against BigFloat references of the definition, with the bound above),
`test/groupnorml2.jl` and `test/norml2.jl` (BigFloat references for `GroupNormL2`
and `NormL2` values and proximal points, and for the `GroupNormL2` KKT screens,
including the exceptional cases above), `test/normlinf.jl` (a sort-based ℓ1-ball reference in `BigFloat` for the
`NormLinf` clip and its KKT conditions, the endpoint rows above, overflow of `γλ` and of `‖x‖₁`, and the
fallback of the threshold), and `test/sparse_value_cost.jl` (a counting element type that shows the
`NormL2` value visits no unstored entry of a sparse or symmetric iterate). These are concrete tested
domains; a historical acceptance goal for a removed family is not evidence for
this implementation. New numerical claims need independent calculations,
FP32/FP64 checks, exceptional inputs and regression tests.
