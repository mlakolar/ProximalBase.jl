# Scalar losses: interface and operations

*Assumes: derivatives and basic Julia · You will learn: observation-level losses, qualified method calls, and how to build a regression objective.*

A **scalar loss** measures how well a single prediction ``\eta`` agrees
with a single observed outcome ``y``:

```math
\ell:\mathbb R\times\mathbb R\to\mathbb R,
\qquad (\eta,y)\mapsto\ell(\eta,y).
```

Its derivatives are taken with respect to the **prediction** ``\eta``,
not the observation ``y``. Scalar losses are useful for constructing
statistical objectives with one term per observation.

The Julia abstraction is `ProximalBase.ScalarLoss`. The
[qualified prefix](glossary.md#Exported-and-qualified-names) means these
names are not exported and must be called through `ProximalBase.`. It is **separate
from `AbstractFunction`**: scalar losses do not automatically provide
the five `AbstractFunction` traits, array `gradient` methods, or
array `prox` methods. The only built-in concrete scalar loss currently
provided is [`ProximalBase.Squared()`](scalar_squared.md).

## All scalar-loss operations

Every operation is listed below. These function names are **not exported**,
so call them with the `ProximalBase.` prefix.

| Julia call | Mathematical meaning | Result |
|:--|:--|:--|
| `ℓ(η, y)` | ``\ell(\eta,y)`` | Scalar loss value |
| `ProximalBase.deriv(ℓ, η, y)` | ``\partial\ell(\eta,y)/\partial\eta`` | First derivative with respect to the prediction |
| `ProximalBase.deriv2(ℓ, η, y)` | ``\partial^2\ell(\eta,y)/\partial\eta^2`` | Second derivative with respect to the prediction |
| `ProximalBase.curvature(ℓ)` | ``C\geq\sup_{\eta,y}\partial^2_\eta\ell(\eta,y)`` | Global upper bound on second derivative; `Inf` if unbounded |
| `ProximalBase.value_and_deriv(ℓ, η, y)` | ``(\ell(\eta,y),\partial_\eta\ell(\eta,y))`` | Tuple `(value, derivative)` |

The `value_and_deriv` generic default evaluates the value and derivative
separately. A concrete loss may provide a specialized implementation.

These are **scalar** operations: they do not take an array destination and
are not mutating methods. In particular, `deriv2` is not a matrix Hessian
and `curvature` is not automatically a Lipschitz constant for an entire
regression problem.

## How a regression objective uses a scalar loss

Given rows ``x_i^\mathsf{T}`` of a design matrix ``X``, a
coefficient vector ``\beta``, and responses ``y_i``, one can
construct an empirical objective

```math
F(\beta)=\sum_{i=1}^{n}w_i\,\ell(x_i^\mathsf{T}\beta,y_i),
\qquad w_i\geq0.
```

Write ``\eta=X\beta`` and let
``d_i=\partial_\eta\ell(\eta_i,y_i)``. Then

```math
\nabla F(\beta)=X^\mathsf{T}(w\odot d).
```

For a twice-differentiable scalar loss the corresponding Hessian is

```math
\nabla^2F(\beta)=
X^\mathsf{T}\operatorname{Diag}\!\left(
w_i\,\partial_\eta^2\ell(\eta_i,y_i)
\right)X.
```

The scalar interface supplies the observation-level building blocks.
**It does not automatically construct `F`, its gradient, or a regression
solver.** The `LeastSquaresLoss` object is a separate built-in
array-function implementation for the squared-error case.

For a twice-differentiable convex scalar loss with
``\partial_\eta^2\ell\leq C``, a useful gradient Lipschitz bound
for this unnormalized objective is
``C\,(\max_i w_i)\|X\|_2^2``. The scalar
`curvature(ℓ)` supplies ``C``; the model and design-matrix
factors are the caller's responsibility.

## Implementing a new scalar loss

A custom type subtypes `ProximalBase.ScalarLoss` and implements
the callable loss, first derivative, second derivative, and a global
curvature bound. For example, the following loss doubles the conventional
squared loss:

```math
\ell(\eta,y)=(\eta-y)^2,\quad
\partial_\eta\ell=2(\eta-y),\quad
\partial_\eta^2\ell=2.
```

```@example custom_scalar_loss
using ProximalBase

struct DoubleSquared <: ProximalBase.ScalarLoss end

(::DoubleSquared)(η::Real, y::Real) = (η - y)^2
ProximalBase.deriv(::DoubleSquared, η::Real, y::Real) = 2 * (η - y)
ProximalBase.deriv2(::DoubleSquared, η::Real, y::Real) = 2.0
ProximalBase.curvature(::DoubleSquared) = 2.0

ℓ = DoubleSquared()
@assert ℓ(3.0, 1.0) == 4.0
@assert ProximalBase.deriv(ℓ, 3.0, 1.0) == 4.0
@assert ProximalBase.deriv2(ℓ, 3.0, 1.0) == 2.0
@assert ProximalBase.curvature(ℓ) == 2.0
@assert ProximalBase.value_and_deriv(ℓ, 3.0, 1.0) == (4.0, 4.0)
```

No extra method is needed for `value_and_deriv` because the default
already calls `ℓ(η,y)` and `deriv(ℓ,η,y)`. Specializing it is
optional when a combined computation is useful.

See [Squared loss](scalar_squared.md) for the built-in implementation and
[Scalar-loss API reference](api/scalar_losses.md) for complete docstrings.
