# Iterate and utility reference

```@autodocs
Modules = [ProximalBase]
Pages = ["sparse_iterate.jl", "atom_iterate.jl", "utils.jl", "lapack.jl", "deprecated.jl"]
Private = false
Filter = t -> !(t in (nnz, dropzeros!))
```

```@docs
nnz(::Union{SparseIterate, SymmetricSparseIterate})
dropzeros!(::Union{SparseIterate, SymmetricSparseIterate})
dropzeros!(::AtomIterate)
```
