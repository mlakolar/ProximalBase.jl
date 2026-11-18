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

## Square-root least squares

`SqrtLeastSquaresLoss` is convex and not smooth, with no gradient and no proximal operator. It is evaluated by the
value oracles of the next group.

```@docs
SqrtLeastSquaresLoss
```

## Least-squares oracles at a supplied predictor or residual

The qualified public names below evaluate a vector-response `LeastSquaresLoss` or `SqrtLeastSquaresLoss`, or one
`Scaled` of either, at a linear predictor or a residual that the caller has already formed
(`value_and_gradient_from_predictor!` serves `LeastSquaresLoss` only). They are not exported.

```@docs
ProximalBase.value_from_predictor
ProximalBase.value_and_gradient_from_predictor!
ProximalBase.value_from_residual
ProximalBase.leastsquares_data
```

## Entrywise ℓ₁ norm, its nonnegative form and elastic net

```@docs
NormL1
NormL1Nonneg
ElasticNet
```

## Nonnegative orthant

```@docs
IndNonnegative
```

## ℓ₂ and group ℓ₂ norms

```@docs
NormL2
GroupNormL2
```

## Maximum-magnitude norm

```@docs
NormLinf
```
