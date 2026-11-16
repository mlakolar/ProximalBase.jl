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

## Least-squares oracles at a supplied predictor or residual

The qualified public names below evaluate a vector-response `LeastSquaresLoss`, or one `Scaled` of it,
at a linear predictor or a residual that the caller has already formed. They are not exported.

```@docs
ProximalBase.value_from_predictor
ProximalBase.value_and_gradient_from_predictor!
ProximalBase.value_from_residual
ProximalBase.leastsquares_data
```

## Entrywise ℓ₁ norm and elastic net

```@docs
NormL1
ElasticNet
```

## ℓ₂ and group ℓ₂ norms

```@docs
NormL2
GroupNormL2
```

Scalar observation losses belong to a separate mathematical interface.
See the [Scalar-loss API reference](scalar_losses.md).
