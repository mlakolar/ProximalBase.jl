# ProximalBase.jl

[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mlakolar.github.io/ProximalBase.jl/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlakolar.github.io/ProximalBase.jl/dev/)
[![CI](https://github.com/mlakolar/ProximalBase.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/ProximalBase.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/ProximalBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/ProximalBase.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Building blocks for proximal, coordinate-descent and active-set optimization in
Julia:

- functions with gradients and proximal operators: losses, convex and
  nonconvex penalties, indicator functions of sets, and functions built from
  other functions;
- sparse iterates with O(1) coordinate access, whose coordinates are entries,
  the entries of a symmetric matrix, or groups of entries.

ProximalBase provides the pieces, not complete solvers: you write the
optimization loop, or use [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl).
Its hot calls allocate nothing.

The [documentation](https://mlakolar.github.io/ProximalBase.jl/dev/) has the
mathematics:
- the proximal operators and the facts solvers use;
- each function's closed forms, with their derivations;
- the coordinate step on symmetric matrices;
- the examples, run and rendered.

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

Requires Julia 1.10 or later.

```julia
using Pkg
Pkg.add("ProximalBase")
```

## The interface

A function `f` is callable, `f(x)` being its value (`Inf` outside its domain),
and its traits say which oracles it provides:

| Trait | Oracles |
|:------|:--------|
| `issmooth(f)` | `gradient!(g, f, x)`, `value_and_gradient!(g, f, x)` |
| `isproximable(f)` | `prox!(y, f, x, γ)`, `value_and_prox!(y, f, x, γ)` |
| `isseparable(f)` | `prox_entry(f, v, i, γ)`, and `prox_coordinate!(x, f, k, γ)` on any iterate |
| `isconvex(f)` | |

Proximal operators follow the convention

```
prox_{γf}(x) = argmin_z  f(z) + ‖z - x‖² / (2γ)
```

Mutating functions take their output first, as `mul!` does, and the
allocating versions `prox(f, x, γ)`, `gradient(f, x)` and
`value_and_gradient(f, x)` return new arrays. `value_and_prox!` returns `f` at
the result, which many functions get from the computation of the proximal
operator itself: a solver that tracks the objective should call it instead of
`prox!` followed by `f(y)`. A function can be both smooth and proximable, as
`LeastSquaresLoss` and `LogDetLoss` are, so the same object serves a proximal
gradient method and ADMM.

`lipschitz(f)` is a Lipschitz constant of `∇f`, for step sizes; for a loss of a
matrix it computes the largest eigenvalue by the Lanczos process, without
forming `X'X`. `gradient_mapping(f, g, x, γ)` is zero exactly at the minimizers
of `f + g`, so its norm is a stopping rule. A separable function of your own
subtypes `SeparableFunction` and implements `value_entry` and `prox_entry`; the
value, `prox!`, `value_and_prox!` and coordinate steps on every iterate come
with it (see [examples/15](examples/15_calculus_and_custom_functions.jl)).

## Example: lasso by proximal gradient descent

Minimize `‖Y - Xβ‖² / 2 + λ‖β‖₁`:

```julia
using ProximalBase, LinearAlgebra

n, p = 100, 20
X = randn(n, p)
β_true = [ones(3); zeros(p - 3)]
Y = X * β_true + 0.1 * randn(n)

f = LeastSquaresLoss(Y, X)   # f(β) = ‖Y - Xβ‖² / 2
g = NormL1(5.0)              # g(β) = 5 ‖β‖₁
γ = 1 / opnorm(X)^2          # step size 1/L, L = Lipschitz constant of ∇f

β = zeros(p)
grad = similar(β)
for iter in 1:500
    gradient!(grad, f, β)             # grad = ∇f(β)
    β .-= γ .* grad
    prox!(β, g, β, γ)                 # β = prox_{γg}(β - γ∇f(β))
end
```

`β` recovers the three nonzero coefficients and sets the rest to exactly zero.

To fit the lasso, use [CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl):
`lasso(X, Y, 5 / n)` computes the same `β` by coordinate descent (its loss is
divided by `n`), and `LassoPath(X, Y)` the solutions along a path of penalty
levels.

## Example: group lasso by block coordinate descent

Minimize `‖Y - Xβ‖² / 2 + λ Σₖ √|Gₖ| ‖β[Gₖ]‖₂`, one group at a time. An
`AtomIterate` is a vector whose coordinates are the groups:

```julia
using ProximalBase, LinearAlgebra

n, p = 100, 20
groups = [1:5, 6:10, 11:15, 16:20]
X = randn(n, p)
β_true = [ones(5); zeros(15)]
Y = X * β_true + 0.1 * randn(n)

g = GroupNormL2(50.0, groups)        # g(β) = 50 Σₖ √5 ‖β[Gₖ]‖₂
β = AtomIterate(p, g)                # its coordinates are the groups of g
r = copy(Y)                          # the residual Y - Xβ
L = [opnorm(X[:, G])^2 for G in groups]
u, old = zeros(5), zeros(5)
for sweep in 1:100, k in 1:ncoordinates(β)
    G = β.groups[k]
    old .= atom(β, k)
    mul!(u, view(X, :, G)', r)                   # u = X[:, G]' r = -∇_G f(β)
    atom(β, k) .+= u ./ L[k]                     # a gradient step on group k,
    prox_coordinate!(β, g, k, 1 / L[k])          # then its proximal step
    mul!(r, view(X, :, G), old .- atom(β, k), 1, 1)
end
dropzeros!(β)
```

`β` keeps the first group, and its only active group is that one: the other
three are exactly zero. `prox_coordinate!` activates a group that it leaves
nonzero, so the active groups track the support, and an active-set solver would
sweep over them only.

## Example: the graphical lasso by ADMM

Minimize `tr(SΩ) - log det Ω + λ Σᵢ≠ⱼ |Ωᵢⱼ|`, with both terms through their
proximal operators:

```julia
using ProximalBase, LinearAlgebra

p = 30
Ω0 = SymTridiagonal(fill(1.0, p), fill(0.4, p - 1))     # a sparse precision matrix
A = randn(500, p) / cholesky(Matrix(Ω0)).U'             # 500 draws from N(0, Ω0⁻¹)
S = Symmetric(A'A / 500)

f = LogDetLoss(S)                         # tr(SΩ) - log det Ω, Inf unless Ω ≻ 0
g = NormL1(0.2, ones(p, p) - I)           # the diagonal is not penalized
Ω, Z, U = Matrix(1.0I, p, p), zeros(p, p), zeros(p, p)
ρ = 1.0
for iter in 1:200
    prox!(Ω, f, Z .- U, 1 / ρ)            # an eigendecomposition, in a workspace of f
    prox!(Z, g, Ω .+ U, 1 / ρ)            # soft thresholding
    U .+= Ω .- Z
end
```

`Z` recovers the pattern of `Ω0`: every entry next to the diagonal is nonzero,
and few others are. `f` and `g` own their workspaces, so the proximal operators
allocate nothing; the temporaries `Z .- U` and `Ω .+ U` are those of this short
loop.

To fit the graphical lasso, use [CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl):
`covselpath(Matrix(S), [0.2, 0.1]; penalize_diag = false)` runs this ADMM to a
stopping rule at each penalty level, warm starting each level from the last.

## More examples

The [examples](examples/) directory has seventeen complete scripts, each of which
checks its own result:
- the lasso by ISTA, FISTA and ADMM, and by coordinate descent along a path;
- nonconvex penalties and the group lasso;
- logistic, robust and Poisson regression;
- constrained least squares, and penalties with constraints;
- the graphical lasso, and covariance and correlation matrices;
- matrix completion, robust PCA, sparse PCA and multitask regression;
- a differential network;
- a working-set solver;
- functions of your own.

## Functions

**Smooth functions**

| Function | Value | Also proximable |
|:---------|:------|:---------------:|
| `ZeroFunction()` | `0` | ✓ |
| `LinearFunction(c, d)` | `⟨c, x⟩ + d` | ✓ |
| `SqrNormL2(λ, y)` | `(λ/2) ‖x - y‖²` | ✓ |
| `QuadraticFunction(A, b, c)` | `x'Ax / 2 + b'x + c` | ✓ |
| `KronQuadratic(A, B, C)` | `⟨Θ, AΘB⟩ / 2 + ⟨C, Θ⟩`, without forming `B ⊗ A` | |
| `LeastSquaresLoss(Y, X)` | `‖Y - Xβ‖² / 2`, `Y` a vector or a matrix | ✓ |
| `LogDetLoss(S, w)` | `w (tr(SX) - log det X)` | ✓ |
| `LogisticLoss(y, X)` | `Σᵢ log(1 + exp(ηᵢ)) - yᵢηᵢ`, `η = Xβ` | |
| `PoissonLoss(y, X)` | `Σᵢ exp(ηᵢ) - yᵢηᵢ` | |
| `HuberLoss(y, X, δ)` | `Σᵢ h_δ(ηᵢ - yᵢ)` | |
| `GLMLoss(ℓ, y, X; weights, offset)` | `Σᵢ wᵢ ℓ(ηᵢ, yᵢ)`, `η = Xβ + offset` | |

The scalar losses `ℓ` of `GLMLoss` (`ProximalBase.Squared`, `Logistic`,
`Poisson`, `Huber`, and `SquaredHinge`, `SmoothCheck` and `ConvolutionCheck`,
smooth versions of the hinge and quantile losses) expose `ProximalBase.deriv`,
`deriv2` and a `curvature` bound, for Newton and coordinate steps.

**Penalties**

| Function | Value | Separable |
|:---------|:------|:---------:|
| `NormL1(λ0, w)` | `λ0 Σᵢ wᵢ \|xᵢ\|` | ✓ |
| `NormL1Nonneg(λ0, w)` | `λ0 Σᵢ wᵢ xᵢ` on `x ≥ 0` | ✓ |
| `ElasticNet(λ1, λ2, w)` | `λ1 Σᵢ wᵢ \|xᵢ\| + (λ2/2) ‖x‖²` | ✓ |
| `NormL0(λ)` | `λ ⋅ #{xᵢ ≠ 0}`, nonconvex | ✓ |
| `MCP(λ, a)`, `SCAD(λ, a)` | the minimax concave and SCAD penalties, nonconvex | ✓ |
| `NormL2(λ)` | `λ ‖x‖₂` | |
| `NormLinf(λ)` | `λ maxᵢ \|xᵢ\|` | |
| `GroupNormL2(λ0, groups, w)` | `λ0 Σₖ wₖ ‖x[Gₖ]‖₂`, the group lasso | |
| `GroupNormLinf(λ0, groups, w)` | `λ0 Σₖ wₖ ‖x[Gₖ]‖∞` | |
| `NormL1Fused(λ1, λ2)` | `Σᵢ λ1 (\|x1ᵢ\| + \|x2ᵢ\|) + λ2 \|x1ᵢ - x2ᵢ\|` on pairs `(x1, x2)` | |
| `NuclearNorm(λ)` | `λ ‖X‖_*` | |
| `TracePSD(λ)` | `λ tr(X)` on `X ⪰ 0` | |

**Nonsmooth losses**, separable, for splitting methods

| Function | Value |
|:---------|:------|
| `CheckLoss(τ, w)` | `Σᵢ wᵢ ρ_τ(xᵢ)`, `ρ_τ(u) = max(τu, (τ - 1)u)`, the quantile loss |
| `HingeLoss(y, w)` | `Σᵢ wᵢ max(0, 1 - yᵢxᵢ)` |
| `EpsInsensitiveLoss(ε, w)` | `Σᵢ wᵢ max(0, \|xᵢ\| - ε)` |

**Indicator functions of sets**

| Function | Set |
|:---------|:----|
| `IndBox(l, u)`, `IndNonnegative()`, `IndBallLinf(r)` | `l ≤ x ≤ u`, `x ≥ 0`, `‖x‖∞ ≤ r` (separable) |
| `IndBallL0(s; diagonal)` | at most `s` nonzero entries, nonconvex; on a symmetric matrix, entries of one triangle, with or without the diagonal |
| `IndBallL1(r)`, `IndBallL2(r)` | `‖x‖₁ ≤ r`, `‖x‖₂ ≤ r` |
| `IndSphereL2(r)` | `‖x‖₂ = r`, nonconvex |
| `IndSimplex(r; equality)` | `x ≥ 0, Σᵢ xᵢ = r` (or `≤ r`) |
| `IndHyperplane(a, b)`, `IndHalfspace(a, b)` | `⟨a, x⟩ = b`, `⟨a, x⟩ ≤ b`; a scalar `a` multiplies `Σᵢ xᵢ` |
| `IndPSD()`, `IndPSD(rank = r)` | `X ⪰ 0`, and of rank at most `r` (nonconvex) |
| `IndBallRank(r)` | rank at most `r`, nonconvex |

**Functions built from functions**

| Function | Value |
|:---------|:------|
| `Scaled(f, α)`, also `α * f` | `α f(x)` |
| `Tilt(f, c, d)` | `f(x) + ⟨c, x⟩ + d` |
| `Translate(f, b)` | `f(x - b)` |
| `Conjugate(f)` | `f*(y)`, with the proximal operator by Moreau's identity |
| `SeparableSum(f1, f2, ...)` | `Σₖ fₖ(xₖ)` on a tuple of blocks |
| `Sum(f1, f2, ...)` | `Σₖ fₖ(x)`, smooth |
| `Constrained(f, C)` | a convex separable `f` on a box, a hyperplane, a halfspace or a simplex `C`; a norm on an ℓ2 ball or sphere |
| `Spectral(f)` | `f(λ(X))` of the eigenvalues of a symmetric `X`, for an `f` that does not depend on the order of its entries: the Fantope, bounds on the eigenvalues, the trace norm |

## Coordinate descent and active sets

**Iterates**

| Type | Coordinates |
|:-----|:------------|
| `SparseIterate` | its entries; a sparse vector or matrix with O(1) entry access and O(nnz) products |
| `SymmetricSparseIterate` | the entries of its lower triangle; a sparse symmetric `p × p` matrix |
| `AtomIterate` | groups of entries that partition `1:p`, with products over the active groups |

`prox_coordinate!(x, f, k, γ)` applies the proximal operator of `γf` to
coordinate `k` of the iterate `x`; for a separable `f` it uses
`prox_entry(f, v, i, γ)`, the scalar proximal operator of entry `i`, which a
coordinate-descent loop can also call directly. Entries index the array and
coordinates the iterate; they differ for a `SymmetricSparseIterate` and an
`AtomIterate`. `ncoordinates(x)`, `nactive(x)`, `activecoordinates(x)`,
`isactive(x, k)`, `activate!(x, k)` and `dropzeros!(x)` maintain the active
set, `coordinates(x)` is a vector view of the coordinates (the lower triangle of
a `SymmetricSparseIterate`), and `tril_index` and `tril_cartesian` convert
between a coordinate and its entry `(r, c)`. A `SymmetricSparseIterate` is an
ordinary `p × p` matrix for indexing, iteration and broadcasting, in which
writing `x[r, c]` also writes `x[c, r]`.

`restrict(f, S)` is the function of `x[S]` on the vectors supported on `S`, and
`kkt_violation(f, g, x, k)` measures how much a coordinate at zero violates
optimality given the gradient `g` of the smooth part.

## Performance

The hot calls (`f(x)`, `prox!`, `value_and_prox!`, `gradient!`,
`value_and_gradient!`, `prox_entry`, `prox_coordinate!`) allocate nothing once a
function has seen its input; the tests check this on Julia 1.10 and the current
release. The spectral functions and `LogDetLoss` call LAPACK through workspaces
that they own, sized for their input, and the proximal operators that need
scratch space (`NormLinf`, `IndBallL1`, `IndSimplex`, `IndBallL0`, ...) keep it.
Projections onto the simplex and the ℓ1 ball take O(n) expected time
(Condat's algorithm), and `IndBallL0` uses an in-place quickselect. A function
that owns a workspace must not be shared between threads; give each thread a
`deepcopy`.

The common calls on `Float64` arrays are compiled when the package is
precompiled, so their first use takes milliseconds, not the tenths of a second
that compilation would take.

Every exported function and type has a docstring; use `?name` in the Julia REPL,
for example `?NormL1`.

## License

MIT, see [LICENSE](LICENSE).
