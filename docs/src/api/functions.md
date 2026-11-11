# Function reference

The [function library](../function_library.md) gives dimensions, errors,
numerical examples and capability limits for every retained concrete type.

## Smooth array functions

```@docs
LeastSquaresLoss
SqrNormL2
QuadraticFunction
ZeroFunction
LinearFunction
```

## Entry penalty

```@docs
NormL1
```

## Qualified scalar observation-loss interface

These names are not exported. ScalarLoss is separate from AbstractFunction;
Squared does not have array gradient/prox methods or construct an automatic
regression model. See [the scalar guide](../smooth.md#Scalar-squared-loss).

```@docs
ProximalBase.ScalarLoss
ProximalBase.Squared
ProximalBase.deriv
ProximalBase.deriv2
ProximalBase.curvature
ProximalBase.value_and_deriv
```
