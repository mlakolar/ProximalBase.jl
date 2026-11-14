# Iterate and utility reference

The [storage overview](../iterates.md) distinguishes entries, independent
symmetric variables, groups and active slots. The
[product domain table](../iterates.md#Supported-products) documents supported
Base/LinearAlgebra methods in addition to the named operations below.

## Storage types

```@docs
SparseIterate
SymmetricSparseIterate
BlockIterate
```

## Working sets and views

```@docs
ncoordinates
nactive
coordinates
activecoordinates
activevalues
isactive
activate!
matrixview
blockview
blockindices
ProximalBase.samegroups
```

## Compaction and sparse-array binding

There is no `nnz` method for iterates. Count activity or mathematical nonzeros
explicitly as explained in [the guide](../iterates.md).

```@docs
dropzeros!(::Union{SparseIterate, SymmetricSparseIterate})
dropzeros!(::BlockIterate)
nnz
```

## Numerical utilities

The triangular helpers assume valid lower-triangle inputs and do not promise
bounds validation; norm_diff supports p=2 or Inf, not arbitrary norm orders.

```@docs
shrink
norm_diff
tril_index
tril_cartesian
```

## Selected three-factor product docstrings

These methods require matching `Float32`/`Float64` and `Symmetric` wrappers of strided
factors for their direct numerical guarantee. Mixed/generic methods have the
narrower guarantees described in [numerics](../numerics.md).

```@docs
Base.:*(::LinearAlgebra.Symmetric{T, <:StridedMatrix{T}}, ::SparseIterate{T, 2}, ::LinearAlgebra.Symmetric{T, <:StridedMatrix{T}}) where {T<:Union{Float32, Float64}}
Base.:*(::LinearAlgebra.Symmetric{T, <:StridedMatrix{T}}, ::SymmetricSparseIterate{T}, ::LinearAlgebra.Symmetric{T, <:StridedMatrix{T}}) where {T<:Union{Float32, Float64}}
```
