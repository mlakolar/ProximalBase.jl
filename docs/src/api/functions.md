# Function reference

```@autodocs
Modules = [ProximalBase]
Pages = ["functions/smooth.jl", "functions/leastsquares.jl", "functions/glm.jl", "functions/logdet.jl",
         "functions/separable.jl", "functions/losses.jl", "functions/entropy.jl", "functions/norms.jl", "functions/fibers.jl", "functions/spectral.jl", "functions/sets.jl", "functions/ordered.jl", "functions/nonconvex.jl", "functions/singular.jl", "functions/matrixsets.jl", "functions/affine.jl", "lla.jl"]
Private = false
Filter = t -> !(t in (SeparableFunction, value_entry, iszeropreserving))
```

```@docs
SparseGroupNormL2
```

## Scalar losses of `GLMLoss`

These are not exported; use them qualified, as `ProximalBase.Logistic()`.

```@docs
ProximalBase.ScalarLoss
ProximalBase.Squared
ProximalBase.Logistic
ProximalBase.Poisson
ProximalBase.Huber
ProximalBase.SquaredHinge
ProximalBase.SmoothCheck
ProximalBase.ConvolutionCheck
ProximalBase.Smoothed
ProximalBase.deriv
ProximalBase.deriv2
ProximalBase.curvature
ProximalBase.value_and_deriv
```

## Intersections of sets

```@docs
IndIntersection
```
