# Functions built from functions

A function can be multiplied by a nonnegative constant using [`Scaled`](@ref).
For example, `0.5 * f` represents the function whose value is half of `f(x)`.
When minimizing a loss plus a penalty, keep them as separate function objects
and combine their operations in the optimization algorithm.

## Nonnegative scaling

For ``h=\alpha f`` with ``\alpha>0`` and finite ordinary arithmetic,

```math
h(x)=\alpha f(x),\qquad \nabla h(x)=\alpha\nabla f(x),\qquad
\operatorname{prox}_{\gamma h}(x)=\operatorname{prox}_{(\gamma\alpha)f}(x).
```

Construct `Scaled(f,α)`, `α*f` or `f*α`, with `f::AbstractFunction` and a real
nonnegative scale. Negative or NaN scales throw `ArgumentError`; `Inf` is
accepted according to the wrapped function's exceptional-value rules. The inner function is stored
by reference, including its parameters, mutable scratch and caches.

Smoothness, convexity, proximability and separability traits forward from the
inner type. Zero preservation requires a finite scale and a zero-preserving
inner instance. Scaling only works with operations already supported by the original function.
For example, scaling does not create a gradient for a function without one.
Details about per-entry steps, special storage and extension hooks are in
[The interface](interface.md) and [Writing your own function](extending.md).

Scalar `prox_block` forwards with the scaled step, and `restrict` wraps the
restricted function. The zero-coordinate screen for positive ordinary scale is
``\alpha\,v_f(\mathrm{grad}/\alpha)``. `lipschitz` scales the inner bound.
There is no Scaled `value_entry` or Scaled compact-vector `prox_block!` fallback.
`reset_cache!` also invalidates relevant caches of the wrapped function.

```@example library_scaled
using ProximalBase
f = SqrNormL2(2.0)
h = Scaled(f,0.25)
x = [2.0,-1]
@assert h(x) == 0.25f(x)
@assert gradient(h,x) == 0.25gradient(f,x)
@assert prox(h,x,0.4) ≈ prox(f,x,0.1)
@assert lipschitz(h) == 0.5
@assert reset_cache!(h) === h
prox(h,x,0.4)
```

## Allocation, aliases and validation

The allocating APIs follow the wrapped function's supported input types.
The mutating APIs take the output first; exact proximal input/output identity
is supported. The combined calls return the value at their respective input
or stored output. Shapes, types and parameters are checked according to the
inner operation before predictable writes. Other overlapping input/output
views remain unsupported. Use independent data/parameter/output arrays.

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
