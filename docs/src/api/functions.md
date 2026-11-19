# Function reference

The [function library](../function_library.md) gives dimensions, errors,
numerical examples and capability limits for the available concrete function types.

## Smooth array functions

```@docs
LeastSquaresLoss
LogDetLoss
LogisticLoss
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

## [Oracles at a supplied predictor or residual](@id Least-squares-oracles-at-a-supplied-predictor-or-residual)

Both predictor operations support `LeastSquaresLoss` and `LogisticLoss`.
`SqrtLeastSquaresLoss` supports the predictor value operation. The residual
value and data accessor serve both least-squares losses. See the individual
operations for the supported domains and scaling rules.

```@docs
ProximalBase.value_from_predictor
ProximalBase.value_and_gradient_from_predictor!
ProximalBase.value_from_residual
ProximalBase.leastsquares_data
```

## Separable nonconvex penalties

```@docs
NormL0
SCAD
```

## Off-diagonal cardinality penalty

```@docs
NormL0OffDiag
```

## Entrywise ℓ₁ norm, its nonnegative form and elastic net

```@docs
NormL1
NormL1Nonneg
ElasticNet
```

## Indicators

```@docs
IndBox
IndBallLinf
IndNonnegative
IndBallL1
IndBallL2
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

## Group maximum-magnitude norm

```@docs
GroupNormLinf
```
