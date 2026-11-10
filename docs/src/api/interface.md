# Interface reference

For an organized user-facing inventory of these operations, see
[Operations on functions](../operations.md), and for their meaning and
capability traits see [Function objects and capabilities](../function_objects.md).
The generated material below is the method-level API reference.


```@docs
ProximalBase
```

```@autodocs
Modules = [ProximalBase]
Pages = ["interface.jl", "lmo_atoms.jl", "second_order.jl", "metric.jl", "caches.jl"]
Private = false
Filter = t -> !(t in (iszeropreserving, ProximalBase.check_parameters,
  ProximalBase.entry_parameters, ProximalBase.check_symmetric_parameters))
```

## Testing

```@docs
ProximalBase.Testing
ProximalBase.Testing.test_prox
ProximalBase.Testing.test_gradient
ProximalBase.Testing.test_separable
ProximalBase.Testing.test_conjugate
ProximalBase.Testing.test_kkt
ProximalBase.Testing.test_iterate
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
ProximalBase.entry_parameters
ProximalBase.check_symmetric_parameters
```

## Matrix norm estimate

```@docs
opnorm2
```
