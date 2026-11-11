# ProximalBase.jl

**Mathematical function objects and working-set storage for optimization in Julia.**

Many optimization problems can be written as

$$
\min_{\beta\in\mathbb R^p}\bigl\{f(\beta)+g(\beta)\bigr\},
$$

where $f$ may be a smooth data-fitting function and $g$ may
be a regularizer. ProximalBase implements function evaluation, gradients,
proximal operators and specialized iterate storage. The choice of
optimization algorithm, step sizes and stopping rules belongs to a solver
such as [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl).

Read the [manual](https://mlakolar.github.io/ProximalBase.jl/dev/),
start with [Getting started](docs/src/getting_started.md), or browse the
[function library](docs/src/function_library.md).

## Example: least squares with an ℓ₁ regularizer

For a design matrix $X\in\mathbb R^{n\times p}$, response
$y\in\mathbb R^n$ and regularization level $\lambda\geq0$,
consider

$$
\min_\beta\left\{
\underbrace{\frac{1}{2n}\|y-X\beta\|_2^2}_{f(\beta)}
+\underbrace{\lambda\|\beta\|_1}_{g(\beta)}
\right\}.
$$

The mathematical components correspond directly to the following Julia
objects:

```julia
using ProximalBase

X = [1.0 2.0; 3.0 1.0; 0.0 2.0]
y = [1.0, 2.0, 3.0]
β = [0.3, -0.4]
λ = 0.2

f = (1 / size(X, 1)) * LeastSquaresLoss(y, X)
g = NormL1(λ)

objective_value = f(β) + g(β)
grad = gradient(f, β)
proximal_point = prox(g, β, 0.1)
```

`LeastSquaresLoss(y,X)` is **half the sum** of squared residuals, so
the multiplier `1/size(X,1)` produces the `1/(2n)` formula above.
`NormL1(λ)` supplies the second term. These operations are building
blocks; the code does not solve the full optimization problem.

## Available functionality

The [function library](docs/src/function_library.md) covers
`LeastSquaresLoss`, `SqrNormL2`, `QuadraticFunction`,
`ZeroFunction`, `LinearFunction`, `NormL1` and `Scaled`.
A separate [Scalar-loss interface](docs/src/scalar_losses.md) handles
individual prediction/observation pairs, with
[`ProximalBase.Squared()`](docs/src/scalar_squared.md) as a built-in example. The [iterate guide](docs/src/iterates.md) covers
`SparseIterate`, `BlockIterate` and `SymmetricSparseIterate`.

See [Operations on functions](docs/src/operations.md) for formulas and
Julia methods, and [Writing your own function](docs/src/extending.md) for
extension examples.

## Testing and development

Run `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`.
CI validates Linux with the current Julia 1.x release. See
[CONTRIBUTING.md](CONTRIBUTING.md) for focused tests, documentation and
benchmarks.

The documented `master` interface is under development and may differ
from a tagged or registered release. See
[Compatibility and migration](docs/src/migration.md) for the
pre-rebuild implementation and its archival references.
