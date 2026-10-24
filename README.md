# ProximalBase.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlakolar.github.io/ProximalBase.jl/dev/)
[![CI](https://github.com/mlakolar/ProximalBase.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/ProximalBase.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/ProximalBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/ProximalBase.jl)

ProximalBase provides building blocks for writing optimization algorithms in
Julia, with an emphasis on problems in high-dimensional statistics. We provide
loss functions, penalties and constraints, together with the operations needed
by coordinate descent, proximal methods and active-set procedures.

The package has three parts:

- Functions with values, gradients and proximal operators, including least
  squares, generalized linear models, sparsity penalties and matrix constraints.
- Coordinate updates for individual entries and groups, with curvature bounds
  and optimality checks for constructing coordinate-descent procedures.
- Sparse vector, symmetric matrix and group iterates that track active
  coordinates and support repeated updates.

Use ProximalBase when you want to write or extend an optimization procedure,
or use the complete solvers in
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl).

## Installation

Requires Julia 1.10 or later.

```julia
using Pkg
Pkg.add("ProximalBase")
```

## One problem, two procedures

We consider the lasso,

$$
\min_\beta\; \frac12\|y-X\beta\|_2^2 + \lambda\|\beta\|_1,
$$

where the loss is a sum over observations, without division by the sample size.
In particular, we use the same loss and penalty in both procedures below,
so that only the optimization loop changes.

```julia
using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(1)
X = randn(rng, 100, 20)
y = X * [ones(3); zeros(17)] + 0.1 * randn(rng, 100)
f = LeastSquaresLoss(y, X)
g = NormL1(5.0)
```

### Coordinate descent

For least squares, we minimize over one coefficient at a time and maintain the
residual `r = y - Xβ`, so each update uses one column of `X`.

```julia
β = SparseIterate(size(X, 2))
r = copy(y)
a = [coordinate_lipschitz(f, j) for j in axes(X, 2)]

for sweep in 1:100, j in axes(X, 2)
    old = β[j]
    z = old + dot(view(X, :, j), r) / a[j]
    new = prox_entry(g, z, j, 1 / a[j])
    if new != old
        β[j] = new
        axpy!(old - new, view(X, :, j), r)
    end
end
```

Here `prox_entry` applies soft thresholding with threshold `5 / a[j]`, while
the iterate stores coordinates that become nonzero during the updates.
Use `dropzeros!(β)` to remove stored entries that have returned to zero.

### Proximal gradient

Next, we update all coefficients together, using the gradient of the loss and
the proximal operator of the penalty.

```julia
β_pg, grad = zeros(size(X, 2)), zeros(size(X, 2))
γ = 1 / opnorm(X)^2
for iter in 1:500
    gradient!(grad, f, β_pg)
    β_pg .-= γ .* grad
    prox!(β_pg, g, β_pg, γ)
end
```

Mutating calls take their output first. The
[getting-started guide](https://mlakolar.github.io/ProximalBase.jl/dev/getting_started/)
derives both updates, adds stopping rules and checks that their solutions agree.

## Documentation

Start with the [guide](https://mlakolar.github.io/ProximalBase.jl/dev/getting_started/),
then read the derivations for
[coordinate descent](https://mlakolar.github.io/ProximalBase.jl/dev/coordinate_descent/)
or [proximal operators](https://mlakolar.github.io/ProximalBase.jl/dev/interface/).
The documentation includes function definitions and an API reference, while
the [examples](examples/README.md) give complete programs for regression,
graphical models, matrix estimation and constrained optimization.

## License

MIT, see [LICENSE](LICENSE).
