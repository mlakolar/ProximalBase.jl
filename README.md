# ProximalBase.jl

[![Stable documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://mlakolar.github.io/ProximalBase.jl/stable/)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlakolar.github.io/ProximalBase.jl/dev/)
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
- Sparse data structures for vectors, symmetric matrices and groups of
  coefficients that track active coordinates and support repeated updates.

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

```math
\min_{\beta} \left\{
\frac12\lVert y-X\beta\rVert_2^2 + \lambda\lVert\beta\rVert_1
\right\},
```

where the loss is a sum over observations, without division by the sample size.
In particular, we use the same loss and penalty in both procedures below,
so that only the optimization loop changes.

```julia
using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(1)
X = randn(rng, 100, 20)
y = X * [ones(3); zeros(17)] + 0.1 * randn(rng, 100)
λ = 5.0
f = LeastSquaresLoss(y, X)
g = NormL1(λ)
```

### Coordinate descent

For least squares, we minimize over one coefficient at a time and maintain the
residual `r = y - Xβ`, so each update uses one column of `X`. With
`a[j] = sum(abs2, X[:, j])`, the exact coordinate minimizer is

```math
\beta_j^{\mathrm{new}} = S_{\lambda/a_j}
\left(\beta_j^{\mathrm{old}} + \frac{X_{:,j}^{\mathsf T}r}{a_j}\right),
```

where $S_t(v)=\mathrm{sign}(v)\max(|v|-t,0)$. The old coefficient
adds back this column's contribution to the residual before fitting its new
value; it is needed because `r` includes the current coefficient.

```julia
β = SparseIterate(size(X, 2))
r = copy(y)
a = [sum(abs2, view(X, :, j)) for j in axes(X, 2)]
all(>(0), a) || error("Remove zero columns before fitting")

for sweep in 1:100
    for j in axes(X, 2)
        xj = view(X, :, j)
        old = β[j]
        z = old + dot(xj, r) / a[j]
        new = prox_entry(g, z, j, 1 / a[j])
        if new != old
            β[j] = new
            axpy!(old - new, xj, r)  # keep r = y - Xβ for the next coordinate
        end
    end
end
```

Here `prox_entry` applies soft thresholding with threshold `λ / a[j]`, while
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

### Check both solutions

The iteration counts above illustrate the two procedures; a fixed iteration
budget alone does not certify convergence. For the lasso, a nonzero coefficient
must satisfy `grad[j] + λ * sign(β[j]) = 0`, and a zero coefficient must satisfy
`abs(grad[j]) <= λ`. Check these conditions and agreement for this example:

```julia
for solution in (β, β_pg)
    grad_solution = gradient(f, solution)
    violation = maximum(eachindex(solution)) do j
        iszero(solution[j]) ? max(abs(grad_solution[j]) - λ, 0.0) :
                             abs(grad_solution[j] + λ * sign(solution[j]))
    end
    @assert violation < 1e-7
end
@assert norm(Vector(β) - β_pg, Inf) < 1e-7
```

`coordinate_lipschitz(f, j)` provides the coordinate curvature used above.

Functions that modify an output array take that array as their first argument.
The
[getting-started guide](https://mlakolar.github.io/ProximalBase.jl/stable/getting_started/)
derives both updates, adds stopping rules and checks that their solutions agree.

## Documentation

Start with the [guide](https://mlakolar.github.io/ProximalBase.jl/stable/getting_started/),
then read the derivations for
[coordinate descent](https://mlakolar.github.io/ProximalBase.jl/stable/coordinate_descent/)
or [proximal operators](https://mlakolar.github.io/ProximalBase.jl/stable/interface/).
The documentation includes function definitions and an API reference, while
the [examples](examples/README.md) give complete programs for regression,
graphical models, matrix estimation and constrained optimization.

## Development and tests

ProximalBase can be developed on its own; local clones of ProximalSolvers
or other related packages are not required. From the ProximalBase directory,
install the declared dependencies and run the complete suite with:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

To use the same test script as continuous integration (CI):

```sh
julia scripts/test.jl
```

This calls `Pkg.test()` with array bounds checking enabled. Julia prepares
an environment containing ProximalBase and its test dependencies automatically.
There is no need to run `scripts/instantiate.jl` first.

To build the documentation or run benchmarks, use `scripts/instantiate.jl`
to install the tools for that task and select your local ProximalBase code.
For example, `julia scripts/instantiate.jl docs` sets up the documentation
build. This script does not clone other package repositories or require any
particular Git branch. See [CONTRIBUTING.md](CONTRIBUTING.md) for the commands.

The [compatibility guide](docs/src/migration.md)
lists the 0.4 names retained through 0.5, their replacements, and the planned
0.6 removal policy.

## License

MIT, see [LICENSE](LICENSE).
