# Scaling a function

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
[Numerical usage](numerical_usage.md).

Smoothness, convexity, proximability and separability traits come from the
original function. Zero preservation additionally requires a finite scale
and an original function that preserves zero. Scaling only works with
operations already supported by the original function.
For example, scaling does not create a gradient for a function without one.
Details about per-entry steps, special storage and extension hooks are in
[Operations on functions](operations.md) and [Writing your own function](extending.md).

For example, a proximal point for positive finite `α` and `γ` uses the
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

## Allocation, aliases and validation

The wrapper follows the original function's supported input shapes and
allocation conventions. It retains the function object by reference, not as
an independent copy. See [Operations on functions](operations.md) for mutating
calls and [Numerical usage](numerical_usage.md) for aliasing and cache rules.

Nested positive finite scales retain their exponents when multiplying them
would overflow or underflow. Built-in recovery paths can fold the total scale
into NormL1's level or recover a representable squared-norm value. This is not
a promise that arbitrary user-defined functions acquire numerical range recovery.

## A zero scale can change the returned type

`Scaled(f,0)` simplifies only when a domain-preserving representation is known.
For finite-data built-in losses, quadratics, centered squared norms and affine
functions it returns `ZeroFunction()`. A zero-scaled finite-level NormL1 with
finite weights also returns ZeroFunction. Infinite weights are retained in
`NormL1(0,w)` so their fixed-zero restrictions survive. Infinite NormL1 levels
and non-finite parameters of other function families can make construction
throw `ArgumentError`. A custom function's zero scale has no generic supported
representation and throws. Inspect the constructed object rather than assume
it is always a `Scaled`.

```@example scaled_zero
using ProximalBase
@assert 0 * SqrNormL2(2.0) isa ZeroFunction
h = 0 * NormL1(1.0,[Inf,1.0])
@assert h isa NormL1
@assert prox(h,[2.0,3.0]) == [0,3]
prox(h,[2.0,3.0])
```

There is no general function-object addition or composition API.
`f(x)+g(x)` evaluates two functions then adds their scalar values. To implement
another mathematical object, use the [extension interface](extending.md).
