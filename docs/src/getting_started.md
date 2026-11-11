# Getting started

Start with a loss and a penalty, evaluate their operations, then assemble one
proximal-gradient step. The example uses functions available in this version.

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

The penalty is not differentiable at zero. Its proximal operator is
soft thresholding, with threshold ``\gamma\lambda``:

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

For an all-zero design, ``L=0``; choose a positive step separately rather
than take its reciprocal. [`opnorm2`](@ref) is an estimate and may miss an
eigenspace; it is not interchangeable with the bound from `lipschitz`.

## Reusing output buffers

Mutating mathematical operations put the destination first. A gradient needs
an output separate from its input. A proximal map permits exact in-place use.

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

For this convex problem, the KKT conditions are

```math
\nabla_j f(\beta)+\lambda\operatorname{sign}(\beta_j)=0\quad(\beta_j\ne0),
\qquad |\nabla_j f(\beta)|\le\lambda\quad(\beta_j=0).
```

`kkt_violation(g, grad, j)` computes ``|\mathrm{grad}_j|-\lambda`` for a
coordinate assumed to be zero. A nonpositive value checks that zero condition;
it does not certify a nonzero coefficient's stationarity.

A solver can store coefficients in a [`SparseIterate`](@ref) and use
[`activate!`](@ref), [`isactive`](@ref) and [`dropzeros!`](@ref) to manage its
working set. Mathematical zeros and active coordinates are different concepts:
a thresholded coefficient can remain active until explicitly compacted.

Continue with [proximal-gradient methods](proximal_gradient.md),
[coordinate descent](coordinate_descent.md), or the executable
[lasso primitives](examples/lasso.md). Those examples illustrate caller loops;
complete solver implementations belong in ProximalSolvers.
