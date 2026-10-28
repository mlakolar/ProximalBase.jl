# Interface reference

```@docs
ProximalBase
```

```@autodocs
Modules = [ProximalBase]
Pages = ["interface.jl", "second_order.jl", "metric.jl", "caches.jl"]
Private = false
```

## Testing

```@docs
ProximalBase.Testing
ProximalBase.Testing.test_prox
ProximalBase.Testing.test_gradient
ProximalBase.Testing.test_separable
ProximalBase.Testing.test_conjugate
ProximalBase.Testing.test_kkt
ProximalBase.Testing.test_traits
ProximalBase.Testing.allocations
ProximalBase.Testing.brute_force_prox
```

## Separable functions

```@docs
SeparableFunction
value_entry
iszeropreserving
ProximalBase.check_parameters
```

## Matrix norm estimate

```@docs
opnorm2
```
