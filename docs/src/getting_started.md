# Getting started

*Assumes: gradients, convexity and basic Julia · You will learn: how to assemble a lasso objective, take a proximal-gradient step and run a short iteration.*

Start with a loss and a regularizer, evaluate their operations, and assemble a
proximal-gradient method.

## Installation

To install the development version shown in this manual in Julia 1.10 or later:

```julia
using Pkg
Pkg.add(url="https://github.com/mlakolar/ProximalBase.jl", rev="master")
```

For a local checkout, use `Pkg.develop(path="/path/to/ProximalBase.jl")` instead.
A registry release may have a different API; see [migration](migration.md).
The documentation and CI are validated on Linux with current Julia 1.x.

## A lasso objective

A lasso objective combines squared-error data fitting with an ℓ₁ regularizer,
which can set some coefficients exactly to zero.

With design matrix ``X\in\mathbb R^{n\times p}``, response ``y\in\mathbb R^n``
and coefficient vector ``\beta\in\mathbb R^p``, consider

```math
\min_\beta\;\underbrace{\frac{1}{2n}\|y-X\beta\|_2^2}_{f(\beta)}
+\underbrace{\lambda\|\beta\|_1}_{g(\beta)},\qquad\lambda\ge0.
```

[`LeastSquaresLoss(y,X)`](@ref LeastSquaresLoss) uses ``\|y-X\beta\|^2/2``.
Multiply it by `1/n` to obtain the mean-loss convention above. The penalty
level stays ``\lambda``. Scaling only the penalty by ``n`` gives the same
minimizers with the unnormalized loss, but different objective values and
step sizes.

```@example getting_started
using ProximalBase, LinearAlgebra
X = [1.0 2; 3 1; 0 2; 2 -1]
y = [1.0, 2, 3, -0.5]
n, p = size(X)
λ = 0.2
f = (1 / n) * LeastSquaresLoss(y, X)
g = NormL1(λ)
β = [0.3, -0.4]
@assert f(β) ≈ sum(abs2, X * β - y) / (2n)
@assert g(β) ≈ λ * sum(abs, β)
(loss = f(β), penalty = g(β), objective = f(β) + g(β))
```

The smooth loss has gradient

```math
\nabla f(\beta)=\frac1n X^\top(X\beta-y).
```

The regularizer is not differentiable at zero. Its
[proximal operator](glossary.md#Proximal-operator) uses
[soft thresholding](glossary.md#Soft-thresholding), with threshold
``\gamma\lambda``:

```math
\operatorname{prox}_{\gamma g}(v)_j=
\operatorname{sign}(v_j)\max(|v_j|-\gamma\lambda,0).
```

```@example getting_started
grad = gradient(f, β)
γ = 0.1
thresholded = prox(g, β, γ)
@assert grad ≈ X' * (X * β - y) / n
@assert thresholded ≈ [0.28, -0.38]
(gradient = grad, proximal_point = thresholded)
```

## A proximal-gradient step

Choose ``0<\gamma\le1/L``, where ``L=\|X\|_2^2/n`` bounds the change in the
gradient. Take a gradient step on ``f``, then apply the proximal map of ``g``:

```math
\beta^+=\operatorname{prox}_{\gamma g}
\bigl(\beta-\gamma\nabla f(\beta)\bigr).
```

```@example getting_started
L = lipschitz(f)
γ = 1 / L
βnext = prox(g, β - γ * gradient(f, β), γ)
@assert f(βnext) + g(βnext) <= f(β) + g(β)
@assert βnext ≈ sign.(β - γ * grad) .* max.(abs.(β - γ * grad) .- γ * λ, 0)
βnext
```

### A short proximal-gradient run

To see sparsity develop over several iterations, reuse the same design and
response but choose a stronger regularization level ``\lambda=1.5``.
Starting from the current ``\beta``, run 100 steps of size ``1/L`` and
record the objective after every step.

```@example getting_started
g_sparse = NormL1(1.5)
β_run = copy(β)
objective_history = [f(β_run) + g_sparse(β_run)]
for _ in 1:100
    β_run = prox(g_sparse, β_run - γ * gradient(f, β_run), γ)
    push!(objective_history, f(β_run) + g_sparse(β_run))
end
# For a convex smooth loss and a valid step, the exact objective
# is nonincreasing. The tolerance accounts for floating-point rounding.
@assert all(diff(objective_history) .<= 1e-12)
@assert any(iszero, β_run)
(final_coefficients = β_run,
 zero_coordinates = findall(iszero, β_run),
 initial_objective = first(objective_history),
 final_objective = last(objective_history))
```

The first coefficient becomes exactly zero. This is the effect of
soft thresholding rather than an explicit equality constraint.

!!! details "Degenerate data and step-size estimates"
    If the design matrix is zero, then `lipschitz(f)` can be zero.
    Do not divide by it; choose a positive step separately.

    [`opnorm2`](@ref) uses an iterative estimate of a squared spectral
    norm. Such an estimate can **underestimate** the true norm, so it is
    not interchangeable with the bound from `lipschitz` when choosing a
    guaranteed descent step. See [numerical usage](numerical_usage.md).



## Reusing output buffers

The Julia convention `!` means that an operation may **mutate** an
argument; it does not promise zero allocations. ProximalBase puts the
**destination first** in mutating mathematical operations.
A gradient needs an output buffer separate from its input, while a proximal
map permits exact in-place use. Other overlapping views are not supported.
See [aliasing](glossary.md#Aliasing) and
[the numerical usage contract](numerical_usage.md) for the precise rules.

```@example getting_started
gradbuf = similar(β)
v = value_and_gradient!(gradbuf, f, β)
@assert v ≈ f(β) && gradbuf ≈ grad
trial = β - γ * gradbuf
vp = value_and_prox!(trial, g, trial, γ)
@assert trial ≈ βnext && vp ≈ g(trial)
(input_value = v, value_at_proximal_point = vp)
```

`gradient!` and `prox!` return the output; `value_and_gradient!` and
`value_and_prox!` return a scalar value. Allocating `value_and_gradient`
returns `(value, gradient)`. See [Operations on functions](operations.md#Calling-conventions,-results-and-supported-inputs) for per-entry steps,
shape, aliasing and return-type details.

## Optimality and storage

For this convex problem, the
[Karush–Kuhn–Tucker (KKT) conditions](glossary.md#KKT-conditions-for-nonsmooth-objectives)
take a slightly different form from stationarity for a smooth constrained
problem. The [subdifferential](glossary.md#Subgradient-and-subdifferential)
of the scalar absolute value is

```math
\partial |t|=
\begin{cases}
\{1\}, & t>0,\\
[-1,1], & t=0,\\
\{-1\}, & t<0.
\end{cases}
```

Here ``\partial |t|`` contains the slopes of supporting lines to the
convex function ``|t|``. Since
``g(\beta)=\lambda\sum_j|\beta_j|``, first-order optimality reads

```math
0\in \nabla f(\beta)+\lambda\partial\|\beta\|_1.
```

Examining one coordinate at a time gives

```math
\begin{cases}
\nabla_j f(\beta)+\lambda\operatorname{sign}(\beta_j)=0,
&\beta_j\ne0,\\[3pt]
|\nabla_j f(\beta)|\le\lambda, &\beta_j=0.
\end{cases}
```

For a nonzero coordinate, the subgradient has the single value
``\operatorname{sign}(\beta_j)``, giving the first equality.
At zero, the subgradient can be **any number in** ``[-1,1]``;
thus the equation ``0=\nabla_j f(\beta)+\lambda s_j`` can hold
precisely when ``|\nabla_j f(\beta)|\le\lambda``.
This is the nonsmooth analogue of the multiplier-based KKT conditions:
if you introduce ``t_j\ge\beta_j`` and ``t_j\ge-\beta_j`` to
represent ``|\beta_j|``, the constraint multipliers lead to the
same two conditions.

For a coordinate assumed to be zero, `kkt_violation(g, grad, j)`
returns ``|\mathrm{grad}_j|-\lambda``. A **positive** result means
that keeping that coefficient at zero violates its stationarity condition;
a **nonpositive** result means zero passes this one check. It does not
evaluate the stationarity condition for a nonzero coefficient, and
checking zero coordinates alone is not a global optimality certificate.
See [coordinate descent](coordinate_descent.md) for the full screening
procedure.

A solver can store coefficients in a [`SparseIterate`](@ref) and use
[`activate!`](@ref), [`isactive`](@ref), and [`dropzeros!`](@ref)
to maintain a [working set](glossary.md#Working-set).
Here **active** means the coordinate is retained in storage or visited
by the solver, **not** that an inequality constraint holds with equality
as in the KKT terminology from a constraints course.
A coefficient can be exactly zero yet remain active so that the solver
checks it again on its next sweep. [Compaction](glossary.md#Compaction)
removes stored zeros when the solver no longer wants to keep them.

Continue with [proximal-gradient methods](proximal_gradient.md),
[coordinate descent](coordinate_descent.md), or the executable
[lasso primitives](examples/lasso.md).
A production solver also needs stopping rules, acceleration and possibly
line search; those belong in
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl).
