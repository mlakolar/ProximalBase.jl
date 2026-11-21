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

Conversion can lose the range of a mixed-type parameter. The gradient and
point-aware KKT scores of `SqrNormL2` convert the level and center entries
to their working type. The proximal map also converts the center entries to
its working type. The value rounds a `Float64` center to `Float32` for
`Float32` data, and a center of `BigFloat` precision to the data type for
`Float32` or `Float64` data. The same center rounding applies during
value recovery. The dense unweighted sum paths of `NormL1` and
`NormL1Nonneg` convert the level to the data type. Their entrywise
fallbacks use the supplied level when the converted level and entries are
finite and the product with the sum is nonfinite. An input that rounds to
zero, a subnormal, or infinity can therefore change the result even when
the exact result is representable.

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
See [SqrNormL2 range clauses](@ref spec-sqrnorml2-range) for its gradient and proximal recovery. `SqrNormL2` and `LeastSquaresLoss` share one squared-residual
recomputation: each residual is formed before scaling (as `x/2 - y/2`, one
binade up, when `x - y` itself overflows) and the squares are summed in units
of the largest one. Genuine NaN/Inf data or parameters keep their ordinary
IEEE result; only finite inputs whose intermediates leave the range are
recomputed. See [LinearFunction range clauses](@ref spec-linear-range) for its finite-input recovery.  Recovery is worthwhile when it changes a materially inaccurate
answer into a representable result in the working precision.

The squared-residual recomputation that `SqrNormL2` and the least-squares oracles share has the same kind of
boundary uncertainty as the `NormL2` recomputation below: within a few eps (the relative error bound of the sum) of
the overflow threshold `floatmax(T) + ulp/2`, where `ulp` is the spacing between `floatmax(T)` and its predecessor, it
does not decide on which side the exact value lies, so a finite answer there is within the error bound of the exact
value and `floatmax(T)` or `Inf` can be returned for a value that rounds to the other. This was observed for
`SqrNormL2`, `Scaled(LeastSquaresLoss, α)(β)` and the oracles of the
[least-squares section](smooth.md#Least-squares-at-a-predictor-or-residual-that-the-caller-holds).

See the [LeastSquaresLoss range clauses](@ref spec-leastsquares-range).

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

For `NormL1` value recovery, including subnormal terms, see the
[NormL1 range clauses](@ref spec-norml1-range).

For nonnegative ℓ₁ value recovery, see the
[NormL1Nonneg range clauses](@ref spec-norml1nonneg-range).

For `ElasticNet` value recovery, its stated bounds and the retained implementation details,
see [ElasticNet range clauses](@ref spec-elasticnet-range).

See the [NormL2 range clauses](@ref spec-norml2-range).

`NormL2`, `GroupNormL2` and `GroupNormLinf` values, and their fused values, may recompute a value at or above `floatmax(T)/2`
in power-of-two units, with sums accumulated in at least `Float64`. Within a few eps of
the overflow threshold `floatmax(T) + ulp/2` that recomputation does not decide on
which side the exact value lies, so a value that rounds to `floatmax(T)` may be
returned as `Inf`, or the reverse: the same limit as the other recoveries above
([#904](https://github.com/mlakolar/ProximalBase.jl/issues/904)).

See the [IndBallL2 range clauses](@ref spec-indballl2-range).

See the [IndBallL1 numerical contract](@ref spec-indballl1) for ℓ₁-ball
feasibility, power-of-two scaling, and tiny-radius projection.

See the [GroupNormL2 range clauses](@ref spec-groupnorml2-range).

`GroupNormLinf` values apply the same check to their group terms `λ0*w[k]*max|x[G_k]|`, each formed in floating point
as `(λ0*w[k])*m` with two rounded products when `λ0*w[k]` is a normal number (and by the three-factor kernel otherwise)
and then converted to the output type, and may recompute from the exponents of `w[k]*max|x[G_k]|` with `λ0` applied once.
See the [GroupNormL2 range clauses](@ref spec-groupnorml2-range).

See the [NormLinf range clauses](@ref spec-normlinf-range).

### [GroupNormLinf clip accuracy](@id groupnormlinf-clip-accuracy)

For `GroupNormLinf`, let `T` be the common type of the input, output, level, weights and step (`Float32` or `Float64`).
For finite data and positive finite parameters in round-to-nearest arithmetic with gradual underflow, each stored entry
is within `8 n eps(T) M + (2n + 1)δ` of the exact proximal entry for `r = γλ0w[k]`. Here `n ≥ 1` is the size of the
group, `M` its largest magnitude and `δ = nextfloat(zero(T))`. The bound assumes `n eps(T) ≤ 1/2`, which the code does
not check. This applies to the direct array and compact block calls. The parameter precedence for zero and infinite
factors is given in the table below.

The exact nonnegative level at radius `r` is the maximum of zero and the levels
`(sum(abs(x[i]) for i in A) - r) / length(A)` over nonempty subsets `A` of the group. Thus it changes by at most the
largest entry perturbation and the absolute radius perturbation. A relative perturbation of the radius by at most `η`
changes the level by at most `η M / (1 - η)`: at the smaller radius a subset with a positive level has radius per entry
at most `M`. The radius uses two products in the working type, or two products of significands followed by exponent
scaling if the normal-range products cannot be used. With `u = eps(T)/2`, the two products contribute at most `4u M`
to the level error and subnormal radius rounding at most `δ`, including the Julia 1.10 `ldexp` allowance above.

If the computed radius is below the computed sum, the shared threshold contributes at most `4.5 n u M + 2nδ` before
accounting for the radius error. If the computed radius reaches the computed sum, that sum is at least
`(1 - (n - 1)u)` times the exact sum. The subset formula then bounds the level for the computed radius by `(n - 1)u M`.
If the rounded radius overflows in these units, the exact radius exceeds `floatmax(T)/(1 + u)^2`. The exact sum is
at most `2 floatmax(T)/3` when its computed value is at most `floatmax(T)/2`, so zero is then the exact result.

If the first sum exceeds `F/2`, where `F = floatmax(T)`, the group is read in units of `2^-k` with
`k = ceil(log2(n)) + 2`. Under the size hypothesis the rounded scaled sum is at most `(F/4 + nδ)/(1 - n u) < F/2`,
so the first scaling suffices. The scaled entries incur at most `δ` of perturbation in those units. Applying the
unscaled bound there and multiplying back gives `(4.5n + 4)u M` plus at most
`2^k [(2n + 2) + (4.5n + 4)u]δ`. Since `2^k < 8n`, the latter term is at most `48 n^2 δ`.
The scaling trigger requires `M > F/(3n)`, and `144 n^2 δ ≤ u F` for `Float32` and `Float64` under the size
hypothesis, so this term is at most `n u M`. Scaling the level back is exact where finite. Capping it at `M`, including
when it overflows, can only decrease the error. Clipping selects an entry or the signed level, both already in `T`,
so same-type storage adds no rounding error. With a different input or output type, entries are read in the floating
type of the destination and the clipped values are converted to its element type on write.

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

For `NormL1` and `NormL1Nonneg` under `Scaled` with a positive finite scalar step (`Float32` or `Float64` factors, step
and level), the product `αγ` of the factors and the step can be below `floatmin(T)` or above `floatmax(T)` while the
threshold `αγλ0wᵢ` of an entry is representable, because its weight brings the product back into range. The level that
holds the factors and the exponent of the step, which is `αγλ0` divided by the significand of `γ`, is not a normal
number when it is zero, when it is subnormal, and when it overflows, which can happen although `αγλ0` is at most
`floatmax(T)`. In these cases the exponent is split between the level and the step when the step and the level of that
split are both normal numbers (the first limit below says when they are not), and the leaf forms the threshold of each
entry from them and its weight, so that the weight can bring it back into range.
For `Scaled(Scaled(NormL1(1.3 * 2.0^20, [2.0^1013]), 2.0^-531), 2.0^-531)`, the step `2.0^-30` and `x = [3 * 2.0^-50]`,
the threshold is `2.2551405187698493e-18` and the scalar step gives `[2.662280118581606e-15]`, as the array step does.
The tests compare the proximal point with the point given by the exact threshold, within four units in the last place of
the point, on selected inputs and on a seeded sweep over the exponents of the factors, the step, the level and the
weight. Two limits, each on its own:

- When the step or the level of the split is not a normal number, which happens for `αγλ0` near or beyond the square of
  a range limit of the type (`floatmin(T)^2`, `floatmax(T)^2`), the exponent is not split and the step is passed rounded
  (zero, a subnormal number or `Inf`), as before.
- `ElasticNet` keeps the earlier route. Under the same conditions, on either side of the range (`αγ` below `floatmin(T)`
  or above `floatmax(T)`), also with `λ2 = 0`, and with a single factor, a scalar step can lose the ℓ₁ threshold where
  the same step passed as an array of steps gives it. With `ElasticNet(1.3 * 2.0^20, 0.0, [2.0^1013])` in place of the
  `NormL1` above, the scalar step gives `[2.6623668547554047e-15]` and the array step `[2.662280118581606e-15]`.

The dedicated `Scaled(ElasticNet(...), α)` limitation and its examples are retained in
[ElasticNet limitations](@ref spec-elasticnet-limitations).

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
| `LinearFunction and ZeroFunction` | See the [LinearFunction and ZeroFunction endpoint clauses](@ref spec-linear-endpoints) |
| `NormL1` | See the [NormL1 endpoint clauses](@ref spec-norml1-endpoints) |
| `NormL2` | See the [NormL2 endpoint clauses](@ref spec-norml2-endpoints) |
| `NormLinf` | See the [NormLinf endpoint clauses](@ref spec-normlinf-endpoints) |
| `GroupNormL2` | See the [GroupNormL2 endpoint clauses](@ref spec-groupnorml2-endpoints) |
| `SparseGroupNormL2` | See the [sparse-group endpoint rules](@ref sparse-group-l2-endpoints). |
| `GroupNormLinf`, zero weight | Unpenalized group: returned unchanged (`NaN`/`±Inf` included), also at an infinite level or step; zero value |
| `GroupNormLinf`, infinite weight, or infinite level with a positive weight | Group fixed at `+0` at every nonnegative step, `γ = 0` and `γ = Inf` included, also for `NaN`/`±Inf` inputs, and also for an infinite weight with a zero level; value `0` at a zero group, `Inf` at a nonzero group without `NaN`, `NaN` at a `NaN` entry |
| `GroupNormLinf`, zero level or zero step, finite positive weight | After the rows above: the identity (`-0`, `NaN` and `±Inf` entries unchanged), also when `λ0*w[k]` overflows and at `γ = Inf` with a zero level; value `0` at a zero level (`NaN` entries included), `λ0 w[k] max abs(x_G)` at a zero step of a positive level |
| `GroupNormLinf`, positive finite level and weight, infinite step | `+0` for every input, `NaN` and `±Inf` entries included |
| `GroupNormLinf`, positive finite level, weight and step | A `NaN` entry makes the group `NaN` and leaves the other groups to their own rules; a `±Inf` entry (no `NaN`) leaves the group unchanged, also when the finite product `γ*λ0*w[k]` overflows; otherwise the clip of the group at `τ` with `Σ(abs(xᵢ) - τ)₊ = γ λ0 w[k]`, and `+0` when the computed radius reaches the computed sum of magnitudes or overflows while that sum is finite |
| `GroupNormLinf`, negative or `NaN` step | `ArgumentError` before any write, in `prox!`, `value_and_prox!` and `prox_block!`; no proximal operator for a `SymmetricSparseIterate` (`ArgumentError` before any write) |
| `SqrNormL2` | See the [SqrNormL2 endpoint clauses](@ref spec-sqrnorml2-endpoints) |
| `ElasticNet` | See the [ElasticNet endpoint clauses](@ref spec-elasticnet-endpoints) |
| `IndBox` | See the [IndBox endpoint clauses](@ref spec-indbox-endpoints) |
| `IndNonnegative` | See the [IndNonnegative endpoint clauses](@ref spec-indnonnegative-endpoints) |
| `IndBallL2` | See the [IndBallL2 endpoint clauses](@ref spec-indballl2-endpoints) |
| `IndBallL1` | See [ℓ₁-ball endpoint rules](@ref spec-indballl1-endpoints). |
| `NormL1Nonneg` | See the [NormL1Nonneg endpoint clauses](@ref spec-norml1nonneg-endpoints) |
| `kkt_violation` | See the [kkt_violation endpoint clauses](@ref spec-kkt_violation-endpoints) |
| Generic custom value/prox | Author-defined arithmetic; traits do not grant recovery or NaN policy |
| Paired symmetric parameters, in `check_symmetric_parameters` | Exact equality, NaN matches NaN, signed zeros agree; the numerical oracles assume it and do not compare |
| Opposing infinities in a symmetric iterate KKT gradient pair | ArgumentError at that chosen pair |

The `NormL1` assertions are retained in its
[executed endpoint example](@ref spec-norml1-endpoints).

The `ElasticNet` assertions are retained in its
[executed endpoint examples](@ref spec-elasticnet-endpoints).

The `NormL1Nonneg` assertions are retained in its
[executed endpoint example](@ref spec-norml1nonneg-endpoints).

The decision on `NormL1` non-finite inputs is retained with the
[NormL1 endpoint clauses](@ref spec-norml1-endpoints).

### [Sparse-group endpoints](@id sparse-group-l2-endpoints)

The following rules apply to each covered group in the array and compact
proximal methods, in the order shown. Uncovered entries pass through the
array prox and are ignored by the value.

| Condition | Proximal result |
|:--|:--|
| Negative or NaN step | `ArgumentError` before writes |
| Infinite stored ℓ₁ level, infinite weight, or infinite stored ℓ₂ level with positive weight | Fill the group with `+0`, including at a zero step and with NaN or infinite inputs |
| Zero step or zero penalty after the hard constraints | Identity |
| Infinite step and positive penalty after the hard constraints | Fill the group with `+0` |
| Finite positive step and a positive group ℓ₂ term, with a NaN input | NaN throughout the group |
| Finite positive step and a positive group ℓ₂ term, with an infinite input and no NaN | Keep the entrywise soft-threshold result without a further group shrink |
| Other finite positive steps | Apply the entrywise soft threshold. A zero group term leaves that result in place. A positive group term then applies the group shrink. |

The entrywise threshold at a finite step propagates NaN at the corresponding
entry when the ℓ₁ term is positive. When finite `γλ1` overflows in the
working type, a coupled group retains genuine infinite inputs. An ℓ₁-only
group follows the [NormL1 endpoint rule](@ref spec-norml1-endpoints)
and returns NaN at such an entry. The group shrink distinguishes a product
of finite factors outside the working range from a stored infinite parameter.
The supported homogeneous precision and general range limits above apply.
For mixed-width calls, the stored intermediate and destination conversions
can change the result. The fused value uses the stored destination.

A zero group has value zero under a hard constraint. An entry outside the
hard domain gives an infinite value unless a penalized entry yields NaN.
Zero penalty terms contribute zero. NaN from a penalized entry takes
precedence over infinite contributions.

For the point-aware KKT methods, NaN among the entries read by a group gives
NaN. Otherwise an infinite entry gives `+Inf`. For finite entries, a
hard-constrained zero group gives `-Inf` and an active group outside
the hard domain gives `+Inf`. An active group whose finite group-level
product rounds to infinity also gives `+Inf` in the working type. At a
zero group the signed screen retains the finite factors for range recovery.
The full-point maximum propagates NaN
across groups and includes the magnitudes of the gradient entries at
uncovered coordinates. See the [sparse group lasso guide](@ref sparse-group-l2-guide)
for the finite-data scores.

### Cardinality penalty endpoints

See the [IndBallL0 endpoint rules](@ref spec-indballl0-endpoints).

See the [NormL0 endpoint clauses](@ref spec-norml0-endpoints).

### [Minimax concave penalty](@id spec-mcp)

See the [MCP numerical contract](@ref mcp-contract) for step rules,
value endpoints and the supported array and iterate routes.

### [Off-diagonal cardinality penalty](@id spec-norml0offdiag-numerics)

See the [numerical contract of `NormL0(λ; diagonal = false)`](@ref spec-norml0offdiag) for the
input domain, the proximal and value rules, refusals and storage.

### SCAD endpoints

See the [SCAD endpoint clauses](@ref scad-endpoints) for the proximal and
value rules at zero and infinite levels, steps and shape.

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
of `{0}` in every coordinate. `NormL1Nonneg` applies it to an infinite level (positive-weight entries become `+0`; zero-weight entries are projected onto the orthant, not left unchanged). `GroupNormL2` and `GroupNormLinf` apply the same reading at `γ = 0` to an infinite level (groups with
a positive weight become `+0`) and to an infinite group weight. `SparseGroupNormL2` also projects
onto its hard domain at zero step. An infinite ℓ₁ level fixes all covered entries, and an infinite ℓ₂
level or weight follows the group rules in the [sparse-group endpoint table](@ref sparse-group-l2-endpoints).
`NormL0`, `NormL2` and `NormLinf` apply
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

The [LinearFunction sparse-value rules and example](@ref spec-linear-endpoints) retain the corresponding strong-zero semantics.

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
`test/kkt_linf.jl` (the full-point `NormLinf` score and its `Scaled` forms against BigFloat references of the definition with
sort-based levels, within the bound of the `NormLinf` accuracy row: six families of data, both branches of the face, ties, the zero point, the exceptional
levels and entries, the ranges, no allocation and no write, and the refused forms),
`test/groupnorml2.jl` and `test/norml2.jl` (BigFloat references for `GroupNormL2`
and `NormL2` values and proximal points, and for the `GroupNormL2` KKT screens,
including the exceptional cases above), `test/normlinf.jl` (a sort-based ℓ1-ball reference in `BigFloat` for the
`NormLinf` clip and its KKT conditions, the endpoint rows above, overflow of `γλ` and of `‖x‖₁`, and the
fallback of the threshold), `test/groupnormlinf.jl` (the same reference for each group of `GroupNormLinf`, its equality with `NormLinf` for one
group and with `NormL1` for singleton groups, the endpoint rows above, the radius that overflows or underflows in two of its factors,
the recomputed values, the margin at a zero group, and the scores of `kkt_violation` against sort-based BigFloat references within the bound above or the allowances listed in `test/TESTING.md`), and `test/sparse_value_cost.jl` (a counting element type that shows the
`NormL2` value visits no unstored entry of a sparse or symmetric iterate). These are concrete tested
domains; a historical acceptance goal for a removed family is not evidence for
this implementation. New numerical claims need independent calculations,
FP32/FP64 checks, exceptional inputs and regression tests.
