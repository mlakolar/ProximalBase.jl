# ProximalBase.jl

Building blocks for proximal, coordinate-descent and active-set optimization in
Julia. Most problems in high-dimensional statistics have the form

```math
\underset{x}{\text{minimize}} \quad f(x) + g(x),
```

with a smooth loss ``f`` (least squares, a generalized linear model, a Gaussian
log-likelihood) and a nonsmooth penalty or constraint ``g`` (the ℓ1 norm, a
group norm, the nuclear norm, the indicator function of a set). First-order
methods solve them with two oracles: the gradient of ``f``, and the proximal
operator of ``g``. ProximalBase provides these oracles for a catalog of
functions, and the sparse iterates on which coordinate descent and active-set
methods work.

ProximalBase provides the pieces, not complete solvers: you write the
optimization loop, or use
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl).

ProximalBase is one of five Julia packages for high-dimensional statistics,
three for estimation and inference and two for the optimization beneath them:

| Package | |
|---|---|
| [CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl) | the lasso and its paths; the group, square-root, scaled and heteroscedastic lasso; varying-coefficient models |
| [CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl) | Gaussian graphical models: the graphical lasso, CLIME and ACLIME, precision matrices by the scaled lasso, the fused graphical lasso, and the difference of two precision matrices |
| [DiffPrecTest.jl](https://github.com/mlakolar/DiffPrecTest.jl) | confidence intervals and tests for the difference of two precision matrices |
| [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl) | the solvers: coordinate descent with penalty paths, screening and active sets; proximal gradient methods, proximal Newton and ADMM |
| **ProximalBase.jl** (this package) | the building blocks: losses, penalties and constraints with their gradients and proximal operators, and sparse iterates; [documentation](https://mlakolar.github.io/ProximalBase.jl/dev/) with the mathematics |

Most users need one of the first three. To solve a problem of your own, use
ProximalSolvers; to write your own algorithm, ProximalBase.

## Installation

```julia
using Pkg
Pkg.add("ProximalBase")
```

ProximalBase requires Julia 1.10 or later.

## A first example

The lasso, ``\tfrac12\|y - X\beta\|^2 + \lambda\|\beta\|_1``, by proximal
gradient descent: a gradient step on the loss, then the proximal operator of the
penalty, which is soft thresholding.

```@example index
using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(1)
X = randn(rng, 100, 20)
y = X * [ones(3); zeros(17)] + 0.1 * randn(rng, 100)

f = LeastSquaresLoss(y, X)            # f(β) = ‖y - Xβ‖² / 2
g = NormL1(5.0)                       # g(β) = 5 ‖β‖₁
L = lipschitz(f)                      # ‖X‖₂², by the Lanczos process

β, grad = zeros(20), zeros(20)
for _ in 1:500
    gradient!(grad, f, β)             # ∇f(β)
    β .-= grad ./ L
    prox!(β, g, β, 1 / L)             # β = prox_{g/L}(β - ∇f(β)/L)
end
round.(β, digits = 3)
```

The three nonzero coefficients are recovered and the others are exactly zero.
The gradient mapping, zero at the solution, confirms it:

```@example index
norm(gradient_mapping(f, g, β, 1 / L))
```

## Principles

- **Functions are objects with oracles.** A function is callable, its value
  being `f(x)`. Traits say which oracles it provides: `gradient!` for a smooth
  function, `prox!` for a proximable one, `prox_entry` for a separable one. A
  function can be both smooth and proximable, so the same object serves a
  gradient method and ADMM.
- **Output first.** Mutating functions take their output first, as `mul!` does:
  `prox!(y, f, x, γ)`, `gradient!(g, f, x)`.
- **No allocations in the hot calls.** Values, gradients, proximal operators and
  coordinate steps allocate nothing once a function has seen its input. The
  spectral functions call LAPACK through workspaces they own, and the tests check
  every hot call on Julia 1.10 and the current release.
- **Iterates for coordinate descent.** A `SparseIterate` gives O(1) access to any
  entry and O(nnz) products. A `SymmetricSparseIterate` is a sparse symmetric
  matrix whose coordinates are the entries of its lower triangle. An
  `AtomIterate` is a vector whose coordinates are groups of entries.

## The documentation

- [The interface](@ref) defines proximal operators and the oracles, with the
  facts about them that solvers use.
- The function pages give each function's definition, gradient, Lipschitz
  constant and proximal operator, with the derivations:
  [Smooth functions](@ref), [Penalties](@ref), [Sets](@ref), and
  [Functions built from functions](@ref).
- [Coordinate descent and iterates](@ref) derives the coordinate step and
  explains the three iterates.
- [Writing your own function](@ref) shows how to add a function, separable or
  not.
- The examples are fifteen complete programs, each checking its result.
