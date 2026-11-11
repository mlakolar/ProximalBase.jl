# Squared scalar loss

The built-in `ProximalBase.Squared()` represents the **squared error of
one prediction and one observation**. It is a concrete subtype of
`ProximalBase.ScalarLoss`, not an array-function object.

## Definition and constructor

For a real prediction ``\eta`` and observed response ``y``,

```math
\ell(\eta,y)=\frac12(\eta-y)^2.
```

The Julia constructor takes **no parameters**:

```julia
ℓ = ProximalBase.Squared()
```

Call `ℓ(η, y)` to evaluate the squared error for a prediction `η`
and observation `y`. The two inputs are given **at evaluation time**.

## Mathematical derivatives and Julia operations

All derivatives are with respect to ``\eta``:

```math
\frac{\partial\ell}{\partial\eta}=\eta-y,\qquad
\frac{\partial^2\ell}{\partial\eta^2}=1,\qquad
\sup_{\eta,y}\frac{\partial^2\ell}{\partial\eta^2}=1.
```

| Julia call | Mathematical value |
|:--|:--|
| `ℓ(η, y)` | ``(\eta-y)^2/2`` |
| `ProximalBase.deriv(ℓ, η, y)` | ``\eta-y`` |
| `ProximalBase.deriv2(ℓ, η, y)` | ``1`` |
| `ProximalBase.curvature(ℓ)` | ``1`` |
| `ProximalBase.value_and_deriv(ℓ, η, y)` | ``((\eta-y)^2/2,\eta-y)`` |

Here is a complete numerical check:

```@example scalar_squared
using ProximalBase

ℓ = ProximalBase.Squared()
η = 3.0
y = 1.0

@assert ℓ(η, y) == 2.0
@assert ProximalBase.deriv(ℓ, η, y) == 2.0
@assert ProximalBase.deriv2(ℓ, η, y) == 1.0
@assert ProximalBase.curvature(ℓ) == 1.0
@assert ProximalBase.value_and_deriv(ℓ, η, y) == (2.0, 2.0)

# The second derivative uses the floating type of the prediction.
@assert ProximalBase.deriv2(ℓ, 3f0, 1f0) === 1f0
(value = ℓ(η, y), derivative = ProximalBase.deriv(ℓ, η, y))
```

## Relation to LeastSquaresLoss

For design matrix ``X\in\mathbb R^{n\times p}``,
response ``y\in\mathbb R^n``, coefficients ``\beta``, and
predictions ``\eta=X\beta``,

```math
\sum_{i=1}^{n}\ell(\eta_i,y_i)
=\frac12\|y-X\beta\|_2^2.
```

The same objective can be represented by the array-function object
`LeastSquaresLoss(y, X)`:

```@example scalar_squared
X = [1.0 0.0; 0.0 2.0]
responses = [1.0, -1.0]
β = [0.5, 0.25]
predictions = X * β

observation_loss = sum(ℓ(predictions[i], responses[i])
                       for i in eachindex(responses))
f = LeastSquaresLoss(responses, X)
@assert observation_loss ≈ f(β)
@assert gradient(f, β) ≈ X' * (predictions - responses)
observation_loss
```

**The interfaces are different.** `ProximalBase.Squared()` handles one
`(prediction, observation)` pair. `LeastSquaresLoss(y, X)`
handles a full coefficient vector, design matrix and response, and
supplies array `gradient` and `prox` operations. Calling
`ProximalBase.Squared()` does not automatically construct a regression model.

The scalar implementation works with real arguments and performs
ordinary promoted scalar arithmetic. Large residuals can overflow;
it does not inherit the special overflow-recovery paths of
`LeastSquaresLoss`.

For the abstraction and all extension requirements, see the
[Scalar-loss interface](scalar_losses.md) and
[Scalar-loss API reference](api/scalar_losses.md).
