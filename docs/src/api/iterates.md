# Iterate and utility reference

```@meta
CurrentModule = ProximalBase
```

```@autodocs
Modules = [ProximalBase]
Pages = ["sparse_iterate.jl", "atom_iterate.jl", "utils.jl", "lapack.jl", "deprecated.jl"]
Private = false
Filter = t -> !(t isa Function && nameof(t) in (:nnz, :nonzeros, :findnz, :dropzeros!, :samegroups))
```

```@docs
dropzeros!(::Union{SparseIterate, SymmetricSparseIterate})
dropzeros!(::AtomIterate)
```

```@docs
ProximalBase.samegroups
```
