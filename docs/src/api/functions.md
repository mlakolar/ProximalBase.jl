# Function reference

```@autodocs
Modules = [ProximalBase]
Pages = ["functions/smooth.jl", "functions/leastsquares.jl", "functions/glm.jl", "functions/logdet.jl",
         "functions/separable.jl", "functions/norms.jl", "functions/spectral.jl", "functions/sets.jl"]
Private = false
Filter = t -> !(t in (SeparableFunction, value_entry, iszeropreserving))
```

## Scalar losses of `GLMLoss`

These are not exported; use them qualified, as `ProximalBase.Logistic()`.

```@docs
ProximalBase.ScalarLoss
ProximalBase.Squared
ProximalBase.Logistic
ProximalBase.Poisson
ProximalBase.Huber
ProximalBase.Smoothed
ProximalBase.deriv
ProximalBase.deriv2
ProximalBase.curvature
ProximalBase.value_and_deriv
```
