# The interface

This page explains **how to call** ProximalBase's mathematical operations:
which methods allocate, which write into an output buffer, what they return,
and which argument and storage combinations are supported. For the mathematical
meaning of gradients, proximal maps, coordinate updates, KKT checks and
step-size bounds, start with [Operations on functions](operations.md).

A solver combines these operations; the function object does not maintain a
solver iteration count, working set or convergence policy. Numerical precision,
parameter ownership, cache validity and uncommon aliasing behavior are described
separately in the [developer numerical usage guide](numerical_usage.md).

## Values and gradients

A smooth ``f`` has differential ``df(x)[h]=\langle\nabla f(x),h\rangle``.
`f(x)` returns its value. `gradient(f,x)` allocates a dense gradient even
when `x` is an array iterate. `gradient!(out,f,x)` writes a supplied destination.
`value_and_gradient!` returns the value at the **input** and stores its gradient;
`value_and_gradient` returns the pair `(value, gradient)`.

```@example interface_gradients
using ProximalBase
f = SqrNormL2(2.0, [1.0, 0])
x = [3.0, -1]
out = similar(x)
v = value_and_gradient!(out, f, x)
@assert v == 5 && out == [4, -2]
@assert value_and_gradient(f, x) == (v, out)
(v, out)
```

The destination must have the matching shape and real element type.
Built-in gradients reject `out === x` with `ArgumentError`. Do not use other
overlapping views of `x` as a gradient destination either. Parameter/output
aliases are function-specific; prefer independent buffers.

## Proximal maps

For positive finite ``\gamma``, the convention is

```math
\operatorname{prox}_{\gamma g}(x)=\operatorname*{argmin}_z
\left\{g(z)+\frac{\|z-x\|^2}{2\gamma}\right\}.
```

For a proper closed convex ``g``, this minimizer is unique. For NormL1,
minimization separates by entry and yields soft thresholding. If
``g(z)=\lambda|z|`` in one dimension, the condition
``0\in\lambda\partial|z|+(z-x)/\gamma`` gives ``z=0`` when
``|x|\le\gamma\lambda`` and otherwise
``z=x-\gamma\lambda\operatorname{sign}(x)``.

```@example interface_prox
using ProximalBase
g = NormL1(0.5)
x = [2.0, -0.25]
out = similar(x)
@assert prox!(out, g, x, 1.0) === out
@assert out == [1.5, 0]
v = value_and_prox!(out, g, x, 1.0)
@assert v == g(out) == 0.75
prox!(x, g, x, 1.0)
@assert x == out
out
```

`prox(f,x,γ)` uses a fresh `similar(x,float(eltype(x)))` for ordinary arrays,
so SparseIterate and BlockIterate retain their storage kind. A `Symmetric` or
`Hermitian` array wrapper gets a dense result. Supported symmetric-owner prox
calls return an independent owner. `value_and_prox!` evaluates the stored output,
including its rounding and working-set semantics.

Exact proximal input/output aliasing is supported. Other shared-memory views
can return wrong values without an error; use `copy(x)` as the source when in
doubt. Predictable shape/type/parameter checks precede writes, but failures
while reading, converting or computing can leave partial output.

## Steps per entry and diagonal metrics

For entry-separable ``g`` and positive finite steps ``\gamma_i``, the diagonal
metric problem is

```math
\operatorname*{argmin}_z g(z)+\sum_i\frac{(z_i-x_i)^2}{2\gamma_i}.
```

The generic `SeparableFunction` array kernels and `ZeroFunction` accept an
array of steps with the same **number of entries** as `x`, paired in linear
order. Scaled forwards this capability only when its inner function supports
it. This does not supply array-step methods for SqrNormL2, LinearFunction,
LeastSquaresLoss or QuadraticFunction. A symmetric-owner destination rejects
an array step with `ArgumentError` before writing.

```@example interface_steps
using ProximalBase
x = [2.0, -1]
γ = [0.5, 2.0]
z = prox(NormL1(0.5), x, γ)
@assert z == [1.75, 0]
z
```

The positive finite step is the mathematical setting. Retained closed forms
also have family-specific zero/Inf/NaN arithmetic; the public interface does
not perform a uniform positivity check. See [numerics](numerics.md) before
using endpoints. A diagonal of coordinate curvatures is not generally a
full-matrix majorizer for simultaneous updates.

## Coordinates, optimality and symmetric geometry

A scalar `prox_block(g,v,i,γ)` receives an already constructed numerical
trial point and a global entry index. It neither reads an iterate nor activates
anything. Compact-vector `prox_block!` is a user extension; implementing it
alone does not define `f(x)`, full-array `prox!`, KKT screens or traits.

At a convex optimum, ``0\in\nabla f(x)+\partial g(x)``. For weighted NormL1,
a zero coordinate obeys ``|\nabla_i f(x)|\le\lambda_0w_i`` and its signed
screen is ``|\nabla_i f(x)|-\lambda_0w_i``. The
[coordinate guide](coordinate_descent.md) explains the nonzero condition and
working-set checks.

A symmetric owner represents a matrix in Frobenius geometry. Each
off-diagonal independent coordinate occurs twice in a function value and
squared distance. Supported allocating gradients at an owner are dense
matrices; SqrNormL2 and LinearFunction support them, while ZeroFunction has
owner value/prox methods but no owner gradient method. A prox into an owner
reads a nonsymmetric matrix argument through its symmetric part. Parameter
arrays must already be exactly symmetric. See
[symmetric parameters](symmetric_parameters.md) for the checked domains.

## Executable scalar contract check

```jldoctest
julia> using ProximalBase

julia> f = SqrNormL2(1.0); g = NormL1(0.5); x = [2.0, -1.0];

julia> (f(x), Tuple(gradient(f,x)), Tuple(prox(g,x)))
(2.5, (2.0, -1.0), (1.5, -0.5))

julia> out = similar(x); v = value_and_prox!(out,g,x,1.0); (v, Tuple(out))
(1.0, (1.5, -0.5))
```
