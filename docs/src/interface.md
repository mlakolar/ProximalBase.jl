
# The interface

This chapter explains the mathematics of gradient and proximal operations,
including per-entry and diagonal-metric steps. It is a **conceptual
explanation** of how these computations are used, not a list of every
exported function. For a searchable overview of the public calls,
start with [Operations on functions](operations.md); for capability
traits, start with [Function objects and capabilities](function_objects.md).

The mathematical function objects represent losses, penalties, constraints
and their compositions. All mutating function operations take their
**output first**, for example `gradient!(grad, f, x)` or
`prox!(out, g, x, γ)`, with the documented exceptions and aliasing rules
in [Array and numerical conventions](numerical_usage.md).
For a full optimization loop, read
[Proximal gradient methods](proximal_gradient.md) or
[Coordinate descent](coordinate_descent.md).

## Gradients and function values

A smooth function's value is obtained by `f(x)`. Its gradient can be
computed into a caller-allocated array by [`gradient!`](@ref), or
allocated with [`gradient`](@ref). When both the value and gradient
are needed, [`value_and_gradient!`](@ref) writes the gradient and
**returns the value at the input**. A smooth function need not have a
finite globally valid Lipschitz constant; check [`lipschitz`](@ref)
before selecting a fixed step.

```@example interface_gradients
using ProximalBase

f = SqrNormL2(1.0)
x = [2.0, -1.0]
out = similar(x)
v = value_and_gradient!(out, f, x)
@assert v ≈ f(x) && out ≈ gradient(f, x)
(value = v, gradient = out)
```

## Proximal operators

The proximal operator of a function ``g`` with step ``\gamma > 0`` is

```math
\operatorname{prox}_{\gamma g}(x) = \operatorname*{argmin}_z \; g(z) + \frac{1}{2\gamma}\|z - x\|^2 .
```

The proximal oracles for linear sets, `IndPolyhedron`, `SumLargest`, the
spectral functions (`NuclearNorm`, `IndBallRank`, `IndStiefel`, `IndPSD`,
`TracePSD`, `Spectral` and `SingularValue`) and `LogDetLoss` do not reject
non-finite input values: closed forms use IEEE arithmetic, and algorithms that
cannot process them return `NaN` in every coupled output entry. A NaN entry
propagates to every result that depends on it, values included, and an
infinite entry follows the function's formula or limit; see
[NaN, infinities and degenerate parameters](@ref numerics-special). The
functions that use LAPACK detect non-finite entries and never pass them to
LAPACK.

It balances decreasing ``g`` against staying near ``x``, and ``\gamma`` sets the
balance. Two cases explain the name and its uses.

- **Projection.** When ``g`` is the indicator function of a set ``C`` (``0`` on
  ``C`` and ``+\infty`` outside), the proximal operator is the Euclidean
  projection onto ``C``, for every ``\gamma``.
- **Shrinkage.** When ``g = \lambda\|\cdot\|_1``, the problem separates over the
  coordinates. Each is solved by soft thresholding,
  ``\operatorname{sign}(x_i)\max(|x_i| - \gamma\lambda, 0)``, which sets small
  coordinates exactly to zero.

For a convex ``g`` (closed and proper) the minimizer is unique, and the map is
*firmly nonexpansive*:

```math
\|\operatorname{prox}_{\gamma g}(x) - \operatorname{prox}_{\gamma g}(y)\|^2
\le \langle \operatorname{prox}_{\gamma g}(x) - \operatorname{prox}_{\gamma g}(y),\, x - y \rangle .
```

The fixed points of ``\operatorname{prox}_{\gamma g}`` are the minimizers of
``g``. For a nonconvex ``g``, such as the ℓ0 penalty, MCP and SCAD with large
steps, or a rank constraint, the minimizer may not be unique. ProximalBase
selects one solution when there are ties. The function pages describe the
closed forms and iterative procedures used to compute the operators.

**The Moreau envelope.** ``M_{\gamma g}(x) = \min_z g(z) + \|z - x\|^2/(2\gamma)``
is a smooth approximation of ``g`` for convex ``g``. Its gradient,

```math
\nabla M_{\gamma g}(x) = \frac{x - \operatorname{prox}_{\gamma g}(x)}{\gamma},
```

is Lipschitz with constant ``1/\gamma``.

**Moreau's decomposition.** With the convex conjugate
``g^*(y) = \sup_x \langle x, y\rangle - g(x)``,

```math
x = \operatorname{prox}_{\gamma g}(x) + \gamma\, \operatorname{prox}_{g^*/\gamma}(x/\gamma).
```

Therefore, the proximal operator of a function gives that of its conjugate.
[`Conjugate`](@ref) uses this, and so do the ℓ∞ norms, through the projection
onto the ℓ1 ball. A check:

```@example interface
using ProximalBase, LinearAlgebra
x, γ = [3.0, -0.5, 1.2], 0.7
f = NormL1(1.0)
@assert prox(f, x, γ) + γ * prox(Conjugate(f), x / γ, 1 / γ) ≈ x # hide
prox(f, x, γ) + γ * prox(Conjugate(f), x / γ, 1 / γ) ≈ x
```

## Steps per entry and diagonal metrics

A scalar step is set by the worst-scaled coordinate. In a diagonal metric
``D = \operatorname{diag}(d)``, the proximal operator of a separable ``g`` is

```math
\operatorname*{argmin}_z\; g(z) + \sum_i \frac{d_i}{2}(z_i - x_i)^2 ,
```

the scalar proximal operator with the step ``\gamma_i = 1/d_i`` in every entry.
So [`prox!`](@ref) and [`value_and_prox!`](@ref) of a [`SeparableFunction`](@ref)
take an array of steps of the size of ``x`` as well as a scalar, on arrays and on
the three iterates. Group norms and the nonconvex [`GroupMCP`](@ref) and
[`GroupSCAD`](@ref) take a vector with one step per group, a metric constant on
each group, which keeps the closed form. `Scaled`, `Tilt`, `Translate` and
`Conjugate` pass the steps on.

The projections onto a hyperplane, a halfspace, a hyperslab, a simplex, an
ℓ1 ball and the monotone vectors ([`IndIsotonic`](@ref), weighted isotonic
regression) take an array of steps as well, as projections in the metric. The hyperplane has the
closed form ``x - \gamma \odot a\,(\langle a, x\rangle - b)/\sum_i \gamma_i a_i^2``.
The simplex and the ℓ1 ball are ``\max(x_i - \gamma_i\nu, 0)`` and the soft
thresholding of ``x_i`` at ``\gamma_i\nu`` for a level ``\nu`` found by Newton's
method from the left, which ends at the exact root. [`Constrained`](@ref) on a
box, a hyperplane, a halfspace, a hyperslab or a simplex takes the steps into
its multiplier search, with ``z_i(\nu) = \operatorname{prox}_{\gamma_i f_i}(x_i - \gamma_i\nu a_i)``.

The step contract takes a limit only where it is unique; otherwise steps must
be positive and finite. For the coupling sets above and the multiplier searches
of `Constrained` on a hyperplane, halfspace, hyperslab or simplex, every array
step must be positive and finite. Invalid steps raise an `ArgumentError` before
writing caller-owned arrays in these methods. This is a chosen contract:
degenerate diagonal metrics need not have a unique limiting projection, since
different sequences of steps can give different limits. `IndIsotonic` also
requires a representable ratio of the largest to the smallest step. Scalar
steps on indicators leave the projection unchanged. The existing behavior of
separable functions, box constraints and per-group steps is unchanged.

The curvature of a smooth ``f`` along each coordinate, a Lipschitz constant of
``\partial f/\partial x_k``, comes from [`coordinate_lipschitz`](@ref) and
[`coordinate_lipschitz!`](@ref): ``\lVert X_{:,j}\rVert^2`` for least squares,
the diagonal of a quadratic, a bound for a GLM. These are the steps of coordinate
descent. A method that moves every coordinate at once can use them as the metric,
but the diagonal of a Hessian does not majorize it, so the steps
``\gamma_i = t/L_i`` need a scalar ``t`` from a line search, or
``t = 1/\lambda_{\max}(D^{-1/2}\nabla^2 f\, D^{-1/2})``, which standardizing the
columns of ``X`` amounts to.

## Numerical conventions and devices

Input/output aliasing, caching, numeric precision, non-finite data,
spectral calculations and GPU support are documented separately in
[Array, precision, and device conventions](numerical_usage.md).
The [numerical contract](@ref numerics) is the authoritative policy
when precise exceptional behavior matters.
