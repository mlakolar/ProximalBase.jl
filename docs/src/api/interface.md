# Interface reference

For mathematical meaning, Julia calling conventions, and allocation/return
behavior, see [Operations on functions](../operations.md).
The docstrings describe generic declarations as well as concrete methods;
consult the concrete guides for supported storage/argument combinations.

```@docs
ProximalBase
AbstractFunction
SeparableFunction
```

## Capabilities

```@docs
issmooth
isproximable
isseparable
isconvex
iszeropreserving
```

## Values, gradients and proximal maps

```@docs
gradient
gradient!
value_and_gradient
value_and_gradient!
prox
prox!
value_and_prox!
value_entry
prox_block
prox_block!
```

## Coordinates, bounds and caches

```@docs
kkt_violation
restrict
lipschitz
opnorm2
reset_cache!
```

## Qualified extension validation

These supported names are public but not exported. Custom public-oracle
overrides must implement their own checked boundary; declaring traits alone
does not run validation. See [extending](../extending.md).

```@docs
ProximalBase.check_parameters
ProximalBase.entry_parameters
ProximalBase.check_symmetric_parameters
```
