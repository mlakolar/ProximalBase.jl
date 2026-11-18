# [Scaling a function](@id scaling-guide)

*Assumes: scalar multiplication and gradients · You will learn: how `Scaled(f, α)` changes function values, gradients and proximal steps.*

The Julia type [`Scaled`](@ref) represents a nonnegative multiple of
an existing mathematical function. This **changes the scale** of a function;
it does not add two functions together or create a solver.

## Nonnegative scaling

For ``h=\alpha f`` with ``\alpha>0`` and finite ordinary arithmetic,

```math
h(x)=\alpha f(x),\qquad \nabla h(x)=\alpha\nabla f(x),\qquad
\operatorname{prox}_{\gamma h}(x)=\operatorname{prox}_{(\gamma\alpha)f}(x).
```

The symbols map to Julia as follows: start with a function object `f`,
choose a nonnegative scale `α`, and construct `h = Scaled(f, α)`.
The expressions `α * f` and `f * α` have the same meaning.
Call `h(x)` to evaluate the scaled function. Thus **`f` is the function
being scaled, `α` is the multiplier, and `x` is the evaluation point**.
Negative or NaN scales throw `ArgumentError`; infinite scaling has
function-specific exceptional arithmetic described in
[Array and precision conventions](numerical_usage.md).

Smoothness, convexity, proximability and separability traits come from the
original function. Zero preservation additionally requires a finite scale
and an original function that preserves zero. Scaling only works with
operations already supported by the original function.
For example, scaling does not create a gradient for a function without one.
Details about per-entry steps, special storage and extension hooks are in
[Operations on functions](operations.md) and [Writing your own function](extending.md).

For example, a [proximal point](glossary.md#Proximal-operator) for positive finite `α` and `γ` uses the
original function's proximal map with step `α * γ`, as in the equation above.
The `lipschitz` bound also scales by `α`. Other supported operations, such
as scalar coordinate updates, coordinate restriction, and cache invalidation,
are forwarded to the original function where methods exist.
This wrapper does **not** automatically provide every possible extension,
including `value_entry` or a compact-vector `prox_block!` implementation.
See [Writing your own function](extending.md) for extension contracts.

```@example library_scaled
using ProximalBase
f = SqrNormL2(2.0)
h = Scaled(f,0.25)
x = [2.0,-1]
@assert h(x) == 0.25 * f(x)
@assert gradient(h,x) == 0.25 * gradient(f,x)
@assert prox(h,x,0.4) ≈ prox(f,x,0.1)
@assert lipschitz(h) == 0.5
@assert reset_cache!(h) === h
prox(h,x,0.4)
```

## [Scoring optimality](@id scaled-kkt)

`kkt_violation(h, v, x, k)` and `kkt_violation(h, v, x)` of `h = Scaled(f, α)` with a
finite positive `α` evaluate the existing score of `f` on the lazy gradient `v/α` and multiply it
by `α`: in exact arithmetic the score of ``\alpha f`` is positively homogeneous in ``\alpha`` and `v`
together. `f` is `NormL1`, `NormL1Nonneg`, `ElasticNet`, `ZeroFunction`, `LinearFunction`, `SqrNormL2`,
`GroupNormL2` or, for the full-point form, `NormL2`. For one wrapper with a finite positive factor, on the
numerical domain below with finite entries of the lazily scaled gradient `v/α`, at a zero coordinate and for arrays and
parameters of one floating type, the signed score agrees bit for bit with the scaled three-argument
screen, except for the endpoint cases of the [numerical contract](numerics.md) and for `SqrNormL2` with an infinite level,
whose score takes its gradient entry from the ridge-gradient helpers.
The scores of the function are in [scoring a coordinate](@ref kkt-screens).

```@example library_scaled
h1 = Scaled(NormL1(0.5), 2.0)             # the function NormL1(1.0)
@assert kkt_violation(h1, [3.0, 0.5], [0.0, 1.0], 1) == 2.0      # 2 (|3/2| - 0.5)
@assert kkt_violation(h1, [3.0, 0.5], [0.0, 1.0], 2) == 1.5      # 2 |0.5/2 + 0.5|
kkt_violation(h1, [3.0, 0.5], [0.0, 1.0])
```

**Domain.** The call evaluates the existing score of the leaf on the lazy gradient `v/α`, with the
rounding and the range limits of that evaluation, and then multiplies by `α`; underflow, overflow
and `NaN` are not recovered. The value agrees with the score of the scaled function up to
rounding when `1/α`, the entries `v[i]/α` and the arithmetic of the leaf stay in the range of the working type. The
limits are those of the leaf: for `Scaled(SqrNormL2(2.0^1000), 2.0^-1000)` at `v = [0.0]`,
`x = [2.0^600]` the quotients are representable but the gradient entry of the leaf overflows, and the call returns
`+Inf` where the score of the scaled function is `2.0^600`. Outside the range of `v/α` the value is the one the
evaluation gives, which need not be an infinity or `NaN`: `Scaled(SqrNormL2(1.0), 1e-300)` at `v = [1e10]`,
`x = [1.0]` gives `+Inf` (the score of the scaled function is `1e10`), and `Scaled(ZeroFunction(), 2.0)` at
`v = [nextfloat(0.0)]` gives `0.0` (the quotient underflows).

The reciprocal of the factor takes part in the type promotion. An ordinary `Int` factor, as in `2 * f`,
introduces `Float64`: data of type `Float16` or `Float32` then widen to `Float64`, while `BigFloat` data
remain `BigFloat`.

```@example library_scaled
@assert kkt_violation(Scaled(SqrNormL2(1.0), 1e-300), [1e10], [1.0], 1) == Inf
@assert kkt_violation(Scaled(SqrNormL2(1.0), 1e-30), [1e10], [1.0], 1) ≈ 1e10
@assert kkt_violation(Scaled(SqrNormL2(2.0^1000), 2.0^-1000), [0.0], [2.0^600], 1) == Inf
@assert kkt_violation(Scaled(ZeroFunction(), 2.0), [nextfloat(0.0)], [0.0], 1) == 0.0
@assert kkt_violation(2 * NormL1(0.5f0), Float32[1, 2], Float32[0, 3], 2) isa Float64
@assert kkt_violation(2 * NormL1(big"0.5"), [big"1.0", big"2.0"], [big"0.0", big"3.0"], 2) isa BigFloat
```

### Unsupported cases

* **Nested wrappers whose level does not fold.** For two positive finite factors, the
  constructor merges the wrappers when the product is a normal number and keeps them nested
  otherwise; zero and infinite factors follow the endpoint rules of the constructor. For a chain around `NormL1`, `NormL1Nonneg`, `ElasticNet`, `GroupNormL2` or `NormL2` the product of the
  factors is folded into the level (both levels of `ElasticNet`, an infinite one being kept as it is) when the result is a nonzero finite
  number that the type of the level accepts, or the level is zero: a normal number, or a subnormal one when the folded level is a
  `Float32` or a `Float64` (a `Float16` level folds only to a normal number). The chain then has the score of the function with
  that level, in each form. A subnormal folded level has fewer significant bits than a normal one, and the coefficient of an entry is that
  rounded level times the weight. Where
  it does not fold, the three-argument screen of `NormL1`, `NormL1Nonneg` and `ElasticNet`, and the four-argument
  form of `NormL1`, `NormL1Nonneg`, `ElasticNet` and `GroupNormL2` at a zero coordinate or group, are computed from the coefficient
  ``a\,\lambda\,w`` of the entry or group, with ``a`` the product of the factors, ``\lambda`` the level and ``w`` the
  weight. For a positive finite ``\lambda`` and ``w`` the significands of the three are multiplied and their exponents are
  applied in one last step, without forming ``a\lambda`` or dividing `v` by ``a``. A zero weight gives the coefficient 0 (also
  under an infinite level), and an infinite weight or level gives `Inf`. For finite entries of `v` the value is
  ``\lvert v_i\rvert-a\lambda w_i`` for a coordinate (``-v_i-a\lambda w_i`` for `NormL1Nonneg`) and ``\lVert v_G\rVert_2-a\lambda w_G`` for a group, up to the rounding
  of the coefficient and of the difference in the type of the call. A `NaN` entry gives `NaN`. An infinite entry gives what the
  three-argument screen gives for `NormL1` and `ElasticNet` (`Inf - Inf` is `NaN`), and `Inf` for `GroupNormL2`. An
  `ElasticNet` with an infinite `λ2` gives `-Inf` at a zero coordinate, whatever `v` holds.
  `Scaled(Scaled(NormL1(1.0), 2.0^-600), 2.0^-600)` at `v = [0.25]` returns `0.25`, the
  coefficient ``2^{-1200}`` rounding to zero in `Float64`. The four-argument form at a
  nonzero coordinate or group (an entry in no group of a `GroupNormL2` counts as nonzero when `x`
  is nonzero there) and the full-point form throw an `ArgumentError` for such a chain, and the three-argument and four-argument
  forms of `NormL2` throw an `ArgumentError`.
* **An infinite factor.** `Scaled(f, Inf)` and `Inf * f` give one wrapper with an infinite factor, also when `f` is
  itself scaled, because the constructor merges the factors. `kkt_violation(h, v, x)` and the four-argument form at a
  nonzero coordinate or group throw an `ArgumentError` for it (an entry in no group of a `GroupNormL2` counts as
  nonzero when `x` is nonzero there). For a finite supplied `v`, the lazy gradient `v/α` is zero and a zero point has
  these values: `-Inf` for a penalized coordinate with a positive coefficient, and `NaN` (`Inf * 0`) where the coefficient
  is zero and for an entry in no group of a `GroupNormL2`; for the smooth functions it is `α` times the score at `v = 0`.
* **A stack that nests an infinite factor with another `Scaled` wrapper.** `Scaled(f, α)` and `α * f` do not produce it,
  because they merge the infinite factor; a direct call of the parameterized constructor `Scaled{F,T}(f, α)` does. For such a stack the
  three-argument screen and the four-argument value at a zero coordinate or group are **unspecified**: they can differ from
  version 0.5, and no particular value, `NaN` included, is promised. The four-argument form at a nonzero coordinate or
  group and the full-point form throw an `ArgumentError`.
* **Chains around `LinearFunction` and `SqrNormL2`.** The product of the factors is applied
  to the gradient entry of the function, as the gradient of the chain applies it, and the score is
  ``\lvert v_i+a\,d_i\rvert`` in all three forms (an uncentered `SqrNormL2` at a zero coordinate,
  and `ZeroFunction`, have ``\lvert v_i\rvert``). The three-argument screen of such a chain
  around `LinearFunction` and a centered `SqrNormL2` was incorrect before 0.6: it
  returned the screen of the function at the unscaled `v`.
* **Functions without the forms.** `NormLinf`, `GroupNormLinf`, the losses and `IndNonnegative` have no
  four-argument or full-point score under `Scaled` (`GroupNormLinf` has its zero-group margin unwrapped only); the call on
  the `Scaled` function throws a `MethodError` (the call names the `Scaled` function). `IndNonnegative` has the
  three-argument screen `-v[i]`. Under a single
  `Scaled(IndNonnegative(), α)` wrapper with a finite positive `α`, the screen is `α` times the screen of
  `IndNonnegative` at the entry `inv(α) * v[i]`. Each of the two products can round, underflow or
  overflow, so the result need not be `-v[i]`: `kkt_violation(Scaled(f, 2.0), [nextfloat(0.0)], 1)` is
  `-0.0`, and `kkt_violation(Scaled(f, 0.25), [floatmax(Float64)], 1)` is `-Inf`. An infinite `α` gives
  `NaN` at a finite `v[i]`. A function defined outside the package is treated as in
  [Writing your own function](extending.md).

```@example library_scaled
lin = Scaled(Scaled(LinearFunction([1.0]), 1e-200), 1e-200)    # the product of the factors underflows Float64
@assert lin.f isa Scaled
@assert kkt_violation(lin, [0.25], 1) == 0.25 && kkt_violation(lin, [0.25], [0.7]) == 0.25
nl = Scaled(Scaled(NormL1(1.0), 2.0^-600), 2.0^-600)
@assert kkt_violation(nl, [0.25], 1) == 0.25                # the coefficient 2^-1200 is not representable; the exponent is applied last
gl = Scaled(Scaled(GroupNormL2(1.0, [[1, 2]], [1.0]), 2.0^-600), 2.0^-600)
@assert kkt_violation(gl, [0.3, 0.4], [0.0, 0.0], 1) == 0.5   # a zero group: the norm of v, less the coefficient
refused(f) = try f(); false catch e; e isa ArgumentError end
@assert refused(() -> kkt_violation(nl, [0.25], [1.0], 1)) && refused(() -> kkt_violation(nl, [0.25], [1.0]))
inf = Scaled(NormL1(1.0), Inf)
@assert kkt_violation(inf, [3.0], [0.0], 1) == -Inf
@assert refused(() -> kkt_violation(inf, [3.0], [1.0], 1)) && refused(() -> kkt_violation(inf, [3.0], [0.0]))
t = 2.0^-600
@assert !(Scaled(Scaled(LinearFunction([1.0]), 2.0), Inf).f isa Scaled)             # the constructor merges the infinite factor
@assert !(Scaled(Scaled(Scaled(NormL1(1.0), t), t), Inf).f isa Scaled)
@assert !((Inf * (t * (t * NormL1(1.0)))).f isa Scaled)
raw(f, a) = Scaled{typeof(f), typeof(a)}(f, a)
handbuilt = raw(Scaled(NormL1(1.0), t), Inf)                    # two wrappers, one of them infinite: only a direct call builds it
@assert handbuilt.f isa Scaled
@assert refused(() -> kkt_violation(handbuilt, [0.25], [1.0], 1)) && refused(() -> kkt_violation(handbuilt, [0.25], [0.0]))
@assert isnan(kkt_violation(Scaled(NormL1(0.0), Inf), [3.0], [0.0], 1))
nomethod = try kkt_violation(Scaled(NormLinf(1.0), 2.0), [1.0, 2.0], [0.0, 1.0], 1); false catch e; e isa MethodError end
@assert nomethod
kkt_violation(lin, [0.25], 1)
```

## Allocation, aliases and validation

The wrapper follows the original function's supported input shapes and
allocation conventions. It retains the function object by reference, not as
an independent copy. See [Operations on functions](operations.md) for mutating
calls and [Array and precision conventions](numerical_usage.md) for aliasing and cache rules.

Nested positive finite scales retain their exponents when multiplying them
would overflow or underflow. Built-in recovery paths can fold the total scale
into the level of `NormL1` or `NormL1Nonneg` or into both levels of `ElasticNet`, or recover a
representable squared-norm value. When the fold takes the ridge level of an
`ElasticNet` below `floatmin` while folding a step, that level is dropped,
since its ridge factor `1/(1 + γλ2)` rounds to one. An infinite level of
`NormL1`, `NormL1Nonneg` or `ElasticNet` stays a hard constraint when the factor and the step
leave the range. The folds of an `ElasticNet` keep an infinite level as it is
and fold the other level as they fold it beside a finite level; a fold that
overflows is not used. With `λ1 = Inf`, a folded ridge level acts on the
entries with a zero weight; with `λ2 = Inf` and a nonnegative step, the
proximal point is `0` at every entry, whatever `λ1`. The scaled value of an `ElasticNet` is recovered as for any
function with levels. For a finite positive factor, where that gives no finite
value (for example `ElasticNet(2.0^-1000, 2.0^1000)` under the factor
`2.0^-600` at `[2.0^300]`, whose levels do not fold by one factor), or `0` at a
point where the function is not exactly `0`, the scaled values of the ℓ₁ part
and of the ridge part, each recovered as for `NormL1` and `SqrNormL2` and
rounded on its own, are added; the sum replaces that result if it is finite
after conversion to its type, and the first result is kept otherwise. This is
not a promise that arbitrary user-defined functions acquire numerical range
recovery. `GroupNormLinf` has no fold of its level: under `Scaled` its proximal operator receives the step `αγ` as the wrapper rounds it,
and an `αγ` that overflows reaches it as `Inf` and gives `+0` (see the [group ℓ∞ norm](group_penalty.md#group-linf-guide)).

## A zero scale can change the returned type

`Scaled(f,0)` simplifies only when a domain-preserving representation is known.
For finite-data built-in losses, quadratics, centered squared norms and affine
functions it returns `ZeroFunction()`. A zero-scaled finite-level `NormL1` with
finite weights also returns `ZeroFunction`. Infinite weights are retained in
`NormL1(0,w)` so their fixed-zero restrictions survive. A zero-scaled
`ElasticNet` with finite levels likewise returns `ZeroFunction` for finite
weights and `ElasticNet(0,0,w)` for infinite ones. A zero-scaled finite-level
`NormL1Nonneg` is `NormL1Nonneg(0,w)` for any weights, which keeps its domain
`x ≥ 0` and the restrictions of infinite weights. With finite weights it is the
indicator of the orthant, whose proximal operator is the projection `max(v, 0)`, not the
identity; an infinite weight keeps its fixed zero (`0 * NormL1Nonneg(1.0, [Inf])` maps
a finite input to `+0`). Infinite `NormL1`, `NormL1Nonneg` and
`ElasticNet` levels and non-finite parameters of other function families can make construction
throw `ArgumentError`. A custom function's zero scale has no generic supported
representation and throws. The indicator `IndNonnegative` does not depend on its scale:
`Scaled(IndNonnegative(), 0)` and `0 * IndNonnegative()` return the indicator itself.
Inspect the constructed object rather than assume it is always a `Scaled`.

```@example scaled_zero
using ProximalBase
@assert 0 * SqrNormL2(2.0) isa ZeroFunction
h = 0 * NormL1(1.0,[Inf,1.0])
@assert h isa NormL1
@assert prox(h,[2.0,3.0]) == [0,3]
prox(h,[2.0,3.0])
```

```@example scaled_zero
z = 0 * NormL1Nonneg(2.0)
@assert z isa NormL1Nonneg && z.λ0 == 0
@assert z([1.0, 0.0]) == 0 && z([1.0, -1.0]) == Inf
@assert prox(z,[-2.0,3.0]) == [0,3]
```

There is no general function-object addition or composition API.
`f(x)+g(x)` evaluates two functions then adds their scalar values. To implement
another mathematical object, use the [extension interface](extending.md).
