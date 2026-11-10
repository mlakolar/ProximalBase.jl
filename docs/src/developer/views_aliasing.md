# Coordinate views, active handles, and aliasing

This is **implementation documentation**, not a requirement for users to call private Base hooks. See the [Iterates overview](../iterates.md) and the [0.6 design contracts](index.md) for the supported surface.


The two coordinate-wrapper types and their internal aliasing hooks are explained here, rather than on separate developer pages. For normative requirements, see the [iterate-family contract](contracts/iterate-family-0.6.md)
and [symmetric-owner contract](contracts/symmetric-sparse-iterate-0.6.md).

## Coordinate IDs, storage slots and matrix positions

There are **three different integers** that are easy to confuse:

| Name | Meaning | How it is used |
|:--|:--|:--|
| Coordinate ID `k` | A stable optimization-variable identifier | `coordinates(x)[k]`, `isactive(x,k)`, `activate!(x,k)` |
| Active slot `a` | Offset into the packed active-value/ID storage | `activecoordinates(x)[a]`, `activevalues(x)[a]` |
| Matrix linear index `i` | Ordinary Julia column-major position in a full matrix | `x[i]` for matrix `SparseIterate`, `matrixview(X)[i]` for a symmetric owner |

For `SparseIterate`, coordinate ID `k` **equals** the array linear index. For `SymmetricSparseIterate`, it indexes the independent lower triangle, *not* the `p²`-entry matrix. Slots have neither meaning: their order is driven by activation and compaction.

The internal two-way mapping is `coordinate ID -> slot` and `slot -> coordinate ID`. The implementation reserves capacity for all coordinates at construction, so membership and first activation are O(1). These are private fields, not public accessors.

## SparseCoordinates: vector view of an ordinary sparse iterate

Source: `src/sparse_iterate.jl`, definitions of `SparseCoordinates` and `coordinates(x::SparseIterate)`.

```julia
struct SparseCoordinates{T,N} <: AbstractVector{T}
    x::SparseIterate{T,N}
end

Base.size(c::SparseCoordinates) = (length(c.x),)
Base.IndexStyle(::Type{<:SparseCoordinates}) = IndexLinear()
# The actual implementations include propagation of bounds checks.
Base.getindex(c::SparseCoordinates, i::Int) = _getstored(c.x, i)
Base.setindex!(c::SparseCoordinates, v, i::Int) = (_setstored!(c.x, v, i); c)

coordinates(x::SparseIterate{T,N}) where {T,N} =
    SparseCoordinates{T,N}(x)
```

This is a **schematic excerpt**; consult the source for the precise bounds-check path. It holds a reference to `x` rather than copying entries or reverse-map arrays.

For both vector and matrix `SparseIterate`, `coordinates(x)` is one-dimensional. In particular, `coordinates(x)[k]` and `x[k]` read and write the same entry in O(1). The wrapper exists for a uniform solver-facing *coordinate vector* and ordinary `AbstractVector` arithmetic semantics, **not** because the underlying iterate needs another lookup mechanism.

It is distinct from `activevalues(x)`: the former has `ncoordinates(x)` logical entries, including implicit zeros; the latter has `nactive(x)` packed values. A generic product or reduction on `SparseCoordinates` may visit **all** coordinates and has ordinary vector arithmetic; it does not inherit `SparseIterate`'s specialized strong-zero sparse products.

## SymmetricCoordinates: independent variables of a non-array owner

Source: `src/sparse_iterate.jl`, `SymmetricCoordinates`.

```julia
struct SymmetricCoordinates{T} <: AbstractVector{T}
    x::SymmetricSparseIterate{T}
end

Base.size(c::SymmetricCoordinates) =
    (length(c.x.ind2nzval),)
Base.IndexStyle(::Type{<:SymmetricCoordinates}) = IndexLinear()
Base.getindex(c::SymmetricCoordinates, k::Int) = _getstored(c.x, k)
Base.setindex!(c::SymmetricCoordinates, v, k::Int) =
    (_setstored!(c.x, v, k); c)

coordinates(x::SymmetricSparseIterate) = SymmetricCoordinates(x)
```

For a `p × p` owner, this view has `q = p(p+1)/2` entries. The owner deliberately rejects `x[k]`, because this might mean either coordinate `k` or full-matrix linear position `k`. `coordinates(x)[k]` resolves the ambiguity without giving the owner generic array iteration or broadcasting.

A zero fill of the symmetric coordinate vector is specialized to visit the *active* slots only; a nonzero fill visits all `q` coordinates. Both preserve the value-write rule that already active zeros remain active.

**The wrapper is not a Frobenius view.** `norm(coordinates(X))` is the Euclidean norm of `q` independent values. The full matrix Frobenius norm has each off-diagonal contribution twice. The owner and its matrix view provide the appropriate matrix-oriented operations.

## SymmetricMatrixView: a separate read-only full-matrix view

The `SymmetricMatrixView <: AbstractMatrix` returned by `matrixview(X)` has shape `(p,p)`, normal matrix indexing and `p²` logical entries. Both triangles read the same independent storage. The view is live and read-only, whereas `Matrix(X)` allocates an independent dense snapshot.

The `SymmetricCoordinates` and `SymmetricMatrixView` wrappers answer different questions. Do not replace one with `reshape` or `vec` of the other: they enumerate different numbers of values.

## _ActiveIDs: protect working-set metadata

`activecoordinates(x)` for `SparseIterate`, `SymmetricSparseIterate` and `BlockIterate` returns the same private handle type:

```julia
struct _ActiveIDs <: AbstractVector{Int}
    ids::SubArray{Int,1,Vector{Int},Tuple{UnitRange{Int}},true}
end

Base.size(h::_ActiveIDs) = size(h.ids)
Base.IndexStyle(::Type{_ActiveIDs}) = IndexLinear()
Base.getindex(h::_ActiveIDs, i::Int) = h.ids[i]
Base.setindex!(::_ActiveIDs, v, i::Int) =
    throw(ArgumentError("activecoordinates(x) is read-only"))
```

The actual error message names the supported ways to change values. `_ActiveIDs` wraps the active-ID list **without copying it**. Preventing `ids[1] = ...` matters: a write would make the ID list disagree with the coordinate-to-slot reverse map, potentially corrupting the iterate. The original pre-0.6 `activecoordinates` returned a writable `view`; PR #414 implemented this wrapper to prevent that problem.

This is a read-only **API guarantee**, not a security boundary. Julia permits introspection of private struct fields. Users must not mutate `h.ids` or other internal references.

## The aliasing methods: why they exist

Two separate Base hooks are implemented for the coordinate wrappers:

```julia
Base.dataids(c::SparseCoordinates) = Base.dataids(c.x)
Base.unaliascopy(c::SparseCoordinates) = coordinates(copy(c.x))

Base.dataids(c::SymmetricCoordinates) = Base.dataids(c.x)
Base.unaliascopy(c::SymmetricCoordinates) = coordinates(copy(c.x))
```

The owner types themselves also implement `Base.dataids(x)`, returning identities for **all their backing arrays** (values, active IDs and reverse maps). `_ActiveIDs` similarly defines `dataids` and `unaliascopy`, preserving the read-only wrapper when a copy is needed.

- **`Base.dataids`** lets Julia recognize that a view, its iterate and another view of the same iterate may share underlying storage. It is used by alias-detection machinery such as `Base.mightalias` and, indirectly, array copying and broadcasting.
- **`Base.unaliascopy`** produces a logically equivalent, nonaliasing source when Julia needs to protect a read/write operation from overlap. `coordinates(copy(c.x))` copies the *iterate state* and then presents the new state's coordinate vector. This can allocate O(n) or O(q) storage when actually executed.

Ordinary indexed reads/writes of a coordinate view do **not** call `unaliascopy` in their hot path. The extra copy is part of the aliasing fallback, not the cost of constructing a coordinate handle.

**These are internal Julia Base hooks.** They are implementation details that should be tested against each supported Julia version; we do not promise their names as ProximalBase public API.

## Handle lifetimes

| Handle | What can change underneath it? | Reacquisition rule |
|:--|:--|:--|
| `coordinates(x)` | Scalar values and active membership | Stable coordinate-ID mapping for the lifetime of `x`; reads remain live |
| `matrixview(X)` | Symmetric matrix values and membership | Stable matrix-position mapping for the lifetime of `X` |
| `activecoordinates(x)` | Active IDs/slot order, extent | Reacquire after any activation, compaction or same-kind state replacement |
| `activevalues(x)` | Active values/slot order, extent | Reacquire after the same operations |

A stale active handle is not a snapshot of the previous working set. Its old length cannot reflect new activations, and a compaction can move slot `a` to a different coordinate. Do not retain slot-indexed momentum history across these events without a corresponding remapping.

## Performance and conformance checks

The `SparseCoordinates` allocation tests in `test/allocations.jl` exercise wrapper creation, scalar reads/writes and zero fill on Float32/Float64 vector and matrix iterates. `test/iterate_queries.jl` exercises live views, size and aliasing behavior. [#694](https://github.com/mlakolar/ProximalBase.jl/issues/694) tracks remaining ordinary-vector conformance tests. The symmetric-owner acceptance plan has independent coverage of its two index spaces and read-only matrix view.

A useful performance comparison is `x[k]` versus `coordinates(x)[k]` for `SparseIterate`, and `X[i,j]` versus `coordinates(X)[k]` for the corresponding independent coordinate on the symmetric owner. These comparisons should be made on warm Float32/Float64 kernels; **no measured zero-time-overhead claim is made merely because the wrappers allocate nothing**.
