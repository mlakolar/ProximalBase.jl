# Writing your own function

A function is a subtype of [`AbstractFunction`](@ref) that declares its traits
and implements the oracles they promise. Solvers written against the interface
then accept it like the built-in functions.

## Any function

| Trait | Implement | You get |
|:------|:----------|:--------|
| always | the value, `(f::MyF)(x)`, `Inf` outside the domain | |
| `isproximable` | [`prox!`](@ref)`(y, f, x, γ)`; `y` may be `x` | [`prox`](@ref), and [`value_and_prox!`](@ref) as `prox!` then `f(y)` |
| `issmooth` | [`value_and_gradient!`](@ref)`(g, f, x)` | [`gradient!`](@ref), [`gradient`](@ref), [`value_and_gradient`](@ref) |
| `isconvex` | | |

Override `value_and_prox!` when the proximal computation gives the value more
cheaply, and define [`lipschitz`](@ref) when a smooth function's constant is
known. The traits are defined on the type:

```@example extending
using ProximalBase, LinearAlgebra

# the indicator of the box [-1, 1]ⁿ intersected with the hyperplane Σᵢ xᵢ = 0
struct CenteredBox <: AbstractFunction end
ProximalBase.isproximable(::Type{CenteredBox}) = true
ProximalBase.isconvex(::Type{CenteredBox}) = true
(::CenteredBox)(x) = all(abs.(x) .<= 1 + 1e-12) && abs(sum(x)) <= 1e-9 ? 0.0 : Inf

# the projection: x = clamp.(v .- τ, -1, 1), with τ chosen by bisection so that Σᵢ xᵢ = 0
function ProximalBase.prox!(y::AbstractVector, ::CenteredBox, v::AbstractVector, ::Real)
    lo, hi = minimum(v) - 1, maximum(v) + 1
    for _ in 1:100
        τ = (lo + hi) / 2
        sum(t -> clamp(t - τ, -1, 1), v) > 0 ? (lo = τ) : (hi = τ)
    end
    y .= clamp.(v .- (lo + hi) / 2, -1, 1)
end

y = prox(CenteredBox(), [3.0, 0.2, -0.4, 0.1], 1.0)
(y, sum(y))
```

The projection uses the same idea as the projection onto the simplex: the
constraint ``\sum_i x_i = 0`` has a multiplier ``\tau``, and for a fixed
``\tau`` the box projection is separable. Here a bisection on ``\tau`` replaces
Condat's exact search.

## Separable functions

A separable function ``f(x) = \sum_i f_i(x_i)`` subtypes
[`SeparableFunction`](@ref) and implements two scalar functions:

- [`value_entry`](@ref)`(f, v, i)`, the value ``f_i(v)``;
- [`prox_entry`](@ref)`(f, v, i, γ)`, the proximal operator of ``\gamma f_i`` at
  ``v``.

Here ``i`` is the linear index of the entry, which lets parameters such as
weights depend on it. The value, [`prox!`](@ref) and
[`value_and_prox!`](@ref) then work on any array and on the three iterates, and
[`prox_coordinate!`](@ref) works on the iterates. Declare
[`iszeropreserving`](@ref) when ``f_i(0) = 0`` and the proximal operator maps
``0`` to ``0``. Then the value and the proximal operator on a sparse iterate
touch only its stored entries.

```@example extending
# Σᵢ wᵢ xᵢ² / 2: separable, with weights
struct WeightedSquare <: SeparableFunction
    w::Vector{Float64}
end
ProximalBase.value_entry(f::WeightedSquare, v, i) = f.w[i] * v^2 / 2
ProximalBase.prox_entry(f::WeightedSquare, v, i, γ) = v / (1 + γ * f.w[i])
ProximalBase.iszeropreserving(::WeightedSquare) = true
ProximalBase.isconvex(::Type{WeightedSquare}) = true

f = WeightedSquare([1.0, 2.0, 4.0])
x = [1.0, 1.0, 1.0]
(f(x), prox(f, x, 0.5))
```

```@example extending
xs = SparseIterate(3)                          # a coordinate step on a sparse iterate
xs[2] = 1.0
prox_coordinate!(xs, f, 2, 0.5), nnz(xs)
```

On a `SymmetricSparseIterate`, the index ``i`` of entry ``(r, c)`` is
``(c - 1)p + r``. A function whose parameters depend on the entry must give
``(r, c)`` and ``(c, r)`` the same parameters, so that its value and its
proximal operator are those of a symmetric matrix. Active-set solvers can also
use [`kkt_violation`](@ref) and [`restrict`](@ref), which you define for your
type when it has the structure they need.

## Testing a new function

- **The proximal operator.** Check the optimality condition
  ``0 \in \partial f(z) + (z - x)/\gamma`` at ``z = \operatorname{prox}_{\gamma f}(x)``,
  or compare with a brute-force minimization, which for a scalar problem is a
  fine grid. Check that `prox!(y, f, y, γ)` gives the same result in place.
- **The gradient.** Compare with central finite differences along a few random
  directions.
- **Allocations.** Measure `@allocated` inside a function, after a warm-up call.
  The hot calls of the built-in functions allocate nothing, and the test suite
  checks them on Julia 1.10 and the current release.
