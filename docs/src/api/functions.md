# Function reference

The [function library](../function_library.md) gives dimensions, errors,
numerical examples and capability limits for the available concrete function types.

## Smooth array functions

```@docs
LeastSquaresLoss
SqrNormL2
QuadraticFunction
ZeroFunction
LinearFunction
```

## Entrywise ℓ₁ norm and elastic net

```@docs
NormL1
ElasticNet
```

Scalar observation losses belong to a separate mathematical interface.
See the [Scalar-loss API reference](scalar_losses.md).
