# ProximalBase.jl

[![CI](https://github.com/mlakolar/ProximalBase.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/ProximalBase.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/ProximalBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/ProximalBase.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Building blocks for proximal and coordinate-descent optimization in Julia:

- smooth loss functions with in-place gradients,
- proximal operators of common penalties (ℓ1, group lasso, nuclear norm, ...),
- sparse iterate types with O(1) coordinate access for coordinate descent.

ProximalBase provides the pieces, not complete solvers: you write the
optimization loop.

## Installation

Requires Julia 1.10 or later.

```julia
using Pkg
Pkg.add("ProximalBase")
```

## Example: lasso by proximal gradient descent

Minimize `‖Y - Xβ‖² / 2 + λ‖β‖₁`:

```julia
using ProximalBase, LinearAlgebra

n, p = 100, 20
X = randn(n, p)
β_true = [ones(3); zeros(p - 3)]
Y = X * β_true + 0.1 * randn(n)

f = LeastSquaresLoss(Y, X)   # f(β) = ‖Y - Xβ‖² / 2
g = ProxL1(5.0)              # g(β) = 5 ‖β‖₁
γ = 1 / opnorm(X)^2          # step size 1/L, L = Lipschitz constant of ∇f

β = zeros(p)
grad = similar(β)
for iter in 1:500
    value_and_gradient!(f, grad, β)   # grad = ∇f(β)
    prox!(g, β, β - γ * grad, γ)      # β = prox_{γg}(β - γ∇f(β))
end
```

`β` recovers the three nonzero coefficients and sets the rest to exactly zero.

## Overview

All proximal operators follow the convention

```
prox_{γg}(x) = argmin_z  g(z) + ‖z - x‖² / (2γ)
```

**Smooth functions** (`DifferentiableFunction`): `value(f, x)`,
`gradient!(f, grad, x)`, `value_and_gradient!(f, grad, x)`

| Type | Function |
|---|---|
| `L2Loss(y)` | `‖x - y‖² / 2` |
| `QuadraticFunction(A, b, c)` | `x'Ax / 2 + b'x + c` |
| `LeastSquaresLoss(Y, X)` | `‖Y - Xβ‖² / 2` |

**Proximal operators** (`ProximableFunction`): `value(g, x)`,
`prox!(g, out, x, γ)`, `prox(g, x, γ)`, and `cdprox!(g, x, k, γ)` for
coordinate-wise updates

| Type | Penalty |
|---|---|
| `ProxZero()` | `0` |
| `ProxL1(λ0)`, `ProxL1(λ0, λ)` | `λ0 Σⱼ λⱼ \|xⱼ\|` |
| `ProxL2(λ0)`, `ProxL2(λ0, λ)` | `λ0 ‖x‖₂`, or group lasso `λ0 Σₖ λₖ ‖xₖ‖₂` on an `AtomIterate` |
| `ProxGroupL2(λ0, groups)`, `ProxGroupL2(λ0, groups, λ)` | group lasso `λ0 Σₖ λₖ ‖x[Gₖ]‖₂` over disjoint groups of any sizes, with `λₖ = √\|Gₖ\|` by default |
| `ProxL2Sq(λ)` | `λ ‖x‖₂²` |
| `ProxNuclear(λ)` | `λ ‖X‖_*` |
| `ProxL1Fused(λ1, λ2)` | `λ1 (\|x1\| + \|x2\|) + λ2 \|x1 - x2\|` |
| `ProxGaussLikelihood(S)` | `tr(SX) - log det X` |

**Iterates**

| Type | Use |
|---|---|
| `SparseIterate` | Sparse vector or matrix with O(1) entry access and O(nnz) multiplication |
| `SymmetricSparseIterate` | Sparse symmetric matrix storing only the lower triangle |
| `AtomIterate` | Dense array split into equal-sized groups, for group penalties |

**Active sets**: the stored entries of a `SparseIterate` can serve as the
active set of an active-set solver. `activate!(x, i)` stores coordinate `i` as
an explicit zero, `restrict(f, S)` and `restrict(g, S)` give the smooth function
and the penalty on the coordinates `S` (for `QuadraticFunction`, `L2Loss`,
`LeastSquaresLoss`, `ProxZero`, `ProxL1` and `ProxGroupL2`), and
`kkt_violation(g, grad, x, i)` measures how much a coordinate at zero violates
optimality. For the separable penalties `ProxZero` and `ProxL1`, the shorter
`kkt_violation(g, grad, i)` gives the same value.

**Utilities**: `shrink`, `shrinkL2!`, `proxL1Fused`, `A_mul_B_row`,
`At_mul_B_row`, `A_mul_X_mul_B`, `A_mul_UUt_mul_B`, `norm_diff`, and index
helpers for lower-triangular storage.

Every exported function and type has a docstring; use `?name` in the Julia REPL,
for example `?ProxL1`.

## License

MIT, see [LICENSE](LICENSE).
