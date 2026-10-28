# Iterate and utility reference

```@meta
CurrentModule = ProximalBase
```

```@autodocs
Modules = [ProximalBase]
Pages = ["sparse_iterate.jl", "atom_iterate.jl", "utils.jl", "lapack.jl", "deprecated.jl"]
Private = false
Filter = t -> !(t isa Function && nameof(t) in (:nnz, :nonzeros, :findnz, :dropzeros!, :opnorm2))
```

```@docs
nnz(::Union{SparseIterate, SymmetricSparseIterate})
nonzeros(::SparseIterate)
findnz(::SparseIterate{<:Any, 1})
dropzeros!(::Union{SparseIterate, SymmetricSparseIterate})
dropzeros!(::AtomIterate)
```
