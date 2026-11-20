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
`GroupNormL2` or, for the full-point form, `NormL2`, `NormLinf`, `LeastSquaresLoss`
or `QuadraticFunction`. For one wrapper with a finite positive factor, on the
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

The full-point form of `NormLinf` is scored the same way. `Scaled(NormLinf(0.5), 2.0)` is the function
`NormLinf(1.0)`, and for a power of two as `α` the quotient `v/α` is exact where it neither overflows nor underflows.

```@example library_scaled
h2 = Scaled(NormLinf(0.5), 2.0)           # the function NormLinf(1.0)
@assert kkt_violation(h2, [3.0, 1.0], [0.0, 0.0]) ≈ sqrt(5.0)                 # 2 × the score at v/2 = (1.5, 0.5)
@assert kkt_violation(h2, [3.0, 1.0], [0.0, 0.0]) ≈ kkt_violation(NormLinf(1.0), [3.0, 1.0], [0.0, 0.0])
kkt_violation(h2, [-0.4, 0.3], [2.0, 1.0])
```

See the [Scaled range clauses](@ref spec-scaled-range).

### Unsupported cases

See the [Scaled limitations and non-guarantees](@ref spec-scaled-limitations).

## Allocation, aliases and validation

See the [Scaled results](@ref spec-scaled-results).

See the [Scaled range clauses](@ref spec-scaled-range).

## A zero scale can change the returned type

See the [Scaled endpoint clauses](@ref spec-scaled-endpoints).

There is no general function-object addition or composition API.
`f(x)+g(x)` evaluates two functions then adds their scalar values. To implement
another mathematical object, use the [extension interface](extending.md).
