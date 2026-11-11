# Iterate and utility reference

```@meta
CurrentModule = ProximalBase
```

```@autodocs
Modules = [ProximalBase]
Pages = ["sparse_iterate.jl", "block_iterate.jl", "utils.jl", "lapack.jl", "deprecated.jl"]
Private = false
Filter = t -> !(t isa Function && nameof(t) in (:dropzeros!, :samegroups))
```

```@docs
dropzeros!(::Union{SparseIterate, SymmetricSparseIterate})
dropzeros!(::BlockIterate)
```

```@docs
ProximalBase.samegroups
```
