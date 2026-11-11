# Scalar-loss API reference

Scalar losses measure individual predictions against observations. They
subtype `ProximalBase.ScalarLoss`, **not** `AbstractFunction`,
and use qualified calls instead of array-function `gradient`
or `prox` operations.

For mathematical definitions, return values and examples, see
[Scalar losses: interface and operations](../scalar_losses.md) and
[Squared scalar loss](../scalar_squared.md).

```@docs
ProximalBase.ScalarLoss
ProximalBase.Squared
ProximalBase.deriv
ProximalBase.deriv2
ProximalBase.curvature
ProximalBase.value_and_deriv
```
