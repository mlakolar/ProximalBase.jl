
# Getting started

Start by calling operations on a mathematical function. Then solve a lasso
problem in two different ways using the **same loss and penalty**:
coordinate descent and proximal gradient. The different algorithms
reuse the same ProximalBase function objects.

## Installation

Install ProximalBase on Julia 1.10 or later:

```julia
using Pkg
Pkg.add("ProximalBase")
```

## First look at function objects

Consider a simple composite optimization problem in two variables,
\(x = (x_1,x_2) \in \mathbb{R}^2\):

```math
\min_{x \in \mathbb{R}^2} F(x) = f(x) + g(x),
\qquad
f(x) = \frac{1}{2}\|x\|_2^2
     = \frac{1}{2}(x_1^2+x_2^2),
\qquad
g(x) = \frac{1}{2}\|x\|_1
     = \frac{1}{2}(|x_1|+|x_2|).
```

The two functions play different roles. The smooth term \(f\) penalizes
large coordinates quadratically, and its gradient is

```math
\nabla f(x) = x.
```

The \(\ell_1\) term \(g\) encourages sparsity and is not differentiable
at a zero coordinate. Instead of a gradient, an optimization method can
use its **proximal operator**, defined for a step \(\gamma>0\) by

```math
\operatorname{prox}_{\gamma g}(v)
= \operatorname*{argmin}_{z \in \mathbb{R}^2}
\left\{g(z) + \frac{\|z-v\|_2^2}{2\gamma}\right\}.
```

For this particular \(g\), the minimization separates by coordinate:
\(\operatorname{prox}_{\gamma g}(v)_i =
\operatorname{sign}(v_i)\max(|v_i|-\gamma/2,0)\).
At \(v=(2,-1)\) and \(\gamma=1\), it gives
\((1.5,-0.5)\).

In ProximalBase, **function objects** represent these mathematical
functions. `SqrNormL2(1.0)` represents \(f\), and `NormL1(0.5)`
represents \(g\). We can evaluate a function, calculate an available
gradient, or request a proximal point without writing those formulas
ourselves:

```@example getting_started_oracles
using ProximalBase

f0 = SqrNormL2(1.0)
g0 = NormL1(0.5)
x0 = [2.0, -1.0]

@assert issmooth(f0) && isproximable(g0)
@assert f0(x0) ≈ 2.5
@assert gradient(f0, x0) ≈ [2.0, -1.0]
@assert prox(g0, x0, 1.0) ≈ [1.5, -0.5]

(value = f0(x0), gradient = gradient(f0, x0), prox = prox(g0, x0, 1.0))
```

Here `issmooth` and `isproximable` describe capabilities, while
`gradient` and `prox` perform the computations. These calls are
**building blocks**, not a complete optimization algorithm: a solver
still chooses where to evaluate them, how large a step to take, and when
to stop. The example uses `f0` and `g0` to distinguish these toy
functions from the lasso loss and penalty introduced next.

See [Function objects and capabilities](function_objects.md) for
the common abstraction and [Operations on functions](operations.md)
for the available methods.

## Lasso problem setup

We consider the objective

```math
F(\beta) = \frac12\|y-X\beta\|_2^2 + \lambda\|\beta\|_1,
```

where ``X\in\mathbb{R}^{n\times p}`` is the design matrix,
``y\in\mathbb{R}^n`` is the response and ``\lambda>0`` controls the penalty.
[`LeastSquaresLoss`](@ref) uses the sum of squared residuals divided by two.
For the alternative convention ``\|y-X\beta\|_2^2/(2n)+\lambda\|\beta\|_1``,
use `NormL1(n * λ)` with this unscaled loss, or scale the loss by `1 / n`.

```@example getting_started
using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(1)
n, p = 100, 20
X = randn(rng, n, p)
y = X * [ones(3); zeros(p - 3)] + 0.1 * randn(rng, n)
λ = 5.0
f = LeastSquaresLoss(y, X)
g = NormL1(λ)
nothing # hide
```

## Coordinate descent for the lasso

Let ``r=y-X\beta`` and ``a_j=\|X_{:,j}\|_2^2>0``. Holding the other
coefficients fixed gives the update

```math
\beta_j^+
= S_{\lambda/a_j}\left(\beta_j+\frac{X_{:,j}^\top r}{a_j}\right),
\qquad
r^+ = r-X_{:,j}(\beta_j^+-\beta_j),
```

where ``S_t(v)=\operatorname{sign}(v)\max(|v|-t,0)`` is soft thresholding.
The residual update costs ``O(n)`` and avoids recomputing ``X\beta`` after
each coordinate. We compute the scalar thresholding step with
[`prox_entry`](@ref), using the curvature ``a_j`` supplied by
[`coordinate_lipschitz`](@ref).

We use a [`SparseIterate`](@ref) to store the coefficients and a full gradient
mapping to check convergence after each sweep. The
[Coordinate descent](@ref) chapter derives the update and gives the coordinate
optimality conditions used by active-set procedures.

```@example getting_started
function lasso_cd(f, g; tol = 1e-8, maxsweeps = 1000)
    X, y = f.X, f.Y
    β = SparseIterate(size(X, 2))
    r = copy(y)
    a = [coordinate_lipschitz(f, j) for j in axes(X, 2)]
    all(>(0), a) || error("Remove zero columns before fitting")
    γ = 1 / opnorm(X)^2
    G = zeros(size(X, 2))

    for sweep in 1:maxsweeps
        for j in axes(X, 2)
            old = β[j]
            z = old + dot(view(X, :, j), r) / a[j]
            new = prox_entry(g, z, j, 1 / a[j])
            if new != old
                β[j] = new
                axpy!(old - new, view(X, :, j), r)
            end
        end
        gradient_mapping!(G, f, g, β, γ)
        norm(G, Inf) <= tol && return dropzeros!(β)
    end
    error("Coordinate descent did not converge within maxsweeps")
end

β_cd = lasso_cd(f, g)
round.(Vector(β_cd), digits = 3)
```

Note that a coordinate that stays at zero is never stored, while an entry
that returns to zero remains active until `dropzeros!` is called.
In particular, storage records the working set during a sweep, and may contain
more entries than the current support.

## Proximal gradient

Next, we update all coefficients together. The loss has gradient
``\nabla f(\beta)=X^\top(X\beta-y)`` with Lipschitz constant
``L=\|X\|_2^2``. A proximal gradient step with ``\gamma=1/L`` is

```math
\beta^+ = \operatorname{prox}_{\gamma g}
\bigl(\beta-\gamma\nabla f(\beta)\bigr).
```

[`gradient!`](@ref) stores the gradient in a supplied array, and
[`prox!`](@ref) applies the proximal operator to the full vector.
Here we use the same stopping rule as for coordinate descent, and overwrite
the coefficients with `prox!(β, g, β, γ)` after taking the gradient step.

```@example getting_started
function lasso_pg(f, g; tol = 1e-8, maxiter = 10_000)
    β = zeros(size(f.X, 2))
    grad, G = similar(β), similar(β)
    γ = 1 / opnorm(f.X)^2
    for iter in 1:maxiter
        gradient!(grad, f, β)
        β .-= γ .* grad
        prox!(β, g, β, γ)
        gradient_mapping!(G, f, g, β, γ)
        norm(G, Inf) <= tol && return β
    end
    error("Proximal gradient did not converge within maxiter")
end

β_pg = lasso_pg(f, g)
@assert norm(Vector(β_cd) - β_pg, Inf) < 1e-7
(coordinate_descent = f(β_cd) + g(β_cd),
 proximal_gradient = f(β_pg) + g(β_pg))
```

## Checking the result

For convex ``f`` and ``g``, the gradient mapping is zero exactly at a
minimizer. A small coefficient change alone need not imply that the
optimality conditions hold, especially when only a working set has been updated.
Here we check both the gradient mapping and the lasso conditions directly.

```math
\nabla_j f(\beta)+\lambda\operatorname{sign}(\beta_j)=0
\quad\text{if }\beta_j\ne0,
\qquad
|\nabla_j f(\beta)|\le\lambda
\quad\text{if }\beta_j=0.
```

```@example getting_started
grad = gradient(f, β_cd)
violation = maximum(eachindex(grad)) do j
    iszero(β_cd[j]) ? max(abs(grad[j]) - λ, 0.0) :
                     abs(grad[j] + λ * sign(β_cd[j]))
end
@assert violation < 1e-7
@assert norm(gradient_mapping(f, g, β_pg, 1 / opnorm(X)^2), Inf) < 1e-8
(kkt_residual = violation, nonzeros = nactive(β_cd))
```

The loops above illustrate how we use ProximalBase to construct two optimization
procedures from the same loss and penalty. For penalty paths, screening and
complete solvers, see
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl). Next,
continue with [Coordinate descent](@ref) for entry and block updates, or
[The interface](@ref) for the mathematics of full proximal steps.
