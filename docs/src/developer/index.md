# Developer documentation and contracts

This page is for contributors implementing or reviewing **storage types, mathematical function kernels, Julia integration methods and numerical behavior** in ProximalBase 0.6. The user-facing guides explain how to call the APIs; this single developer page explains *which behavior is required*, *why*, and *how it is tested*.

## Contract index

| Contract or record | Authoritative source | Full authoritative text |
|:--|:--|:--|
| SparseIterate / AtomIterate family | `docs/design/iterate-family-0.6.md` | [Full 0.6 iterate-family contract](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/iterate-family-0.6.md) |
| Family acceptance and downstream migration | `docs/design/iterate-family-0.6-acceptance.md` | [Full iterate acceptance plan](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/iterate-family-0.6-acceptance.md) |
| SymmetricSparseIterate independent-coordinate owner | `docs/design/symmetric-sparse-iterate-0.6.md` | [Full symmetric owner contract](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/symmetric-sparse-iterate-0.6.md) |
| Symmetric owner acceptance and migration | `docs/design/symmetric-sparse-iterate-0.6-acceptance.md` | [Full symmetric owner acceptance plan](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/symmetric-sparse-iterate-0.6-acceptance.md) |
| Package-wide numerical behavior | `docs/src/numerics.md` | [Numerical contract](../numerics.md) (already published) |
| SparseIterate source-level audit | `docs/audit/sparse-iterate-06-method-inventory.md` | [Method-by-method source inventory](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/audit/sparse-iterate-06-method-inventory.md) |

**Source of truth:** The files in `docs/design/` remain the authoritative versions of the four iterate-family documents. This developer page links directly to their full text on GitHub instead of duplicating them into additional documentation pages. **Edit only the authoritative files**. The numerical contract already lives in `docs/src/numerics.md` and is published directly. The source audit is historical evidence of definitions, not an alternative public API contract.

To inspect the numerical decisions and implementation evidence behind the public numerical contract, use the following canonical design records in GitHub:

- [Numerical-contract design record](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/README.md)
- [Maintainer decisions and rationale](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/decisions.md)
- [Numerical conformance coverage map](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/coverage.md)
- [Step endpoints and infinite-argument design](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/step-infinity-interface-311-469.md)
- [Historical issue impact review](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/impact-on-open-issues.md)

These **design records** are informative history and tracking evidence; they are not automatically up-to-date tests or an independent normative specification. The public numerical contract governs when a historical discussion differs from it.

## Data structures versus mathematical functions

The storage interfaces are defined by what an algorithm can do with its **iterate**: index, write, activate, copy, traverse the active set and materialize a dense value.

| Iterate | Shape / index space | Main accessors | Distinguishing rule |
|:--|:--|:--|:--|
| `SparseIterate` | `AbstractVector` or `AbstractMatrix`; coordinate = array linear index | `x[k]`, `coordinates(x)`, `activecoordinates(x)`, `activevalues(x)` | O(1) scalar access, active-slot traversal O(s) |
| `SymmetricSparseIterate` | Non-array owner; coordinate = independent lower-triangle entry | `X[i,j]`, `coordinates(X)`, `matrixview(X)`, active handles | `X[k]` is forbidden; symmetric entries represent one variable |
| `AtomIterate` | `AbstractVector`; coordinate = group | `x[j]`, `atom(x,k)`, `atomindices(x,k)`, `activecoordinates(x)` | Active sets contain groups, not scalar entries |

See the [SparseIterate guide](../sparse_iterate.md), [symmetric owner guide](../symmetric_sparse_iterate.md), and the coordinate-view implementation notes below.

The following APIs belong to **functions or algorithms**, not to the storage type's minimum contract: `restrict`, `prox_block`, `prox_block!`, `kkt_violation`, `gradient!`, `prox!`, `quadratic_model` and `value_and_prox!`. They may accept an iterate, but a new iterate does not need to define those oracles itself. Consult [the function interface](../interface.md) and [coordinate descent](../coordinate_descent.md).

## How to read the 0.6 contracts

Each iterate-family contract contains a **requirements section** with stable rule IDs, followed by design decisions and historical evidence. The requirements and any explicit later amendments govern; past proposals and experiments are not promises. For the current SparseIterate decision, **FC-11 is amended**: `nnz`, `nonzeros`, `findnz` and `issparse` are no longer specialized on `SparseIterate`. The underlying independent-coordinate owner also does not provide these aliases, and `dropzeros!` remains in both types.

The numerical contract explicitly separates promised behavior from conformance-suite gaps. Do not infer that a published rule has passed every backend test: consult the coverage map and the relevant CI run or tracker.

## Contributor workflow for a contract change

1. **State the need and scope.** Is this a storage interface, Julia `AbstractArray` compatibility, a kernel optimization, a function-oracle capability or a numerical policy? Avoid moving one into another merely for convenience.
2. **Edit the authoritative requirement** with rule IDs, semantics, input validation, aliasing, explicit-zero behavior, complexity and performance targets. If replacing an earlier decision, state explicitly which rule was amended.
3. **Update the acceptance plan and tests**, including independent references, exception behavior, and the supported Julia versions. For owner changes, test the distinction between coordinate IDs, active slots and full matrix indices.
4. **Update the user-facing guide, API reference, migration guide and source docstrings.** Keep the developer documentation linked to the authoritative record, rather than cloning the contract into another independently editable file.
5. **Audit the downstream solver consumers.** In particular, changes in copy semantics, group layouts, owner indexing or activation may require ProximalSolvers migration and issue #527's release ledger.
6. **Measure ordinary CPU hot paths** on Float32 and Float64 before and after changes, with the pre-registered benchmark method and stated acceptance threshold. Do not silently accept a regression in the name of interface cleanliness.
7. **Require reviewed, green CI** before merging. Generated documentation appearing in the navigation is not evidence that the numerical or iterate conformance gates have passed.

## Maintaining the contract references

This is **one developer documentation page**. The complete iterate-family and owner contracts remain in `docs/design/` and are available through the links above; they are not copied or generated as extra navigation pages. The numerical contract is already published directly at [Numerical contract](../numerics.md). The source-level audit remains in `docs/audit/`.

When amending a contract, update the source document, relevant tests and acceptance plan, plus the user-facing guide and this index when necessary. The historical discussion in a contract can describe earlier policies; the current normative requirement and later explicit amendments govern.

Current implementation/release tracking: [iterate-family umbrella #456](https://github.com/mlakolar/ProximalBase.jl/issues/456), [symmetric-owner umbrella #391](https://github.com/mlakolar/ProximalBase.jl/issues/391), [numerical contract #366](https://github.com/mlakolar/ProximalBase.jl/issues/366), and [downstream migration #527](https://github.com/mlakolar/ProximalBase.jl/issues/527).


## Coordinate views and aliasing

The two coordinate-wrapper types and their internal aliasing hooks are explained here, rather than on separate developer pages. For the authoritative rules, see the iterate-family and symmetric-owner contracts above.

### Coordinate IDs, storage slots and matrix positions

There are **three different integers** that are easy to confuse:

| Name | Meaning | How it is used |
|:--|:--|:--|
| Coordinate ID `k` | A stable optimization-variable identifier | `coordinates(x)[k]`, `isactive(x,k)`, `activate!(x,k)` |
| Active slot `a` | Offset into the packed active-value/ID storage | `activecoordinates(x)[a]`, `activevalues(x)[a]` |
| Matrix linear index `i` | Ordinary Julia column-major position in a full matrix | `x[i]` for matrix `SparseIterate`, `matrixview(X)[i]` for a symmetric owner |

For `SparseIterate`, coordinate ID `k` **equals** the array linear index. For `SymmetricSparseIterate`, it indexes the independent lower triangle, *not* the `p²`-entry matrix. Slots have neither meaning: their order is driven by activation and compaction.

The internal two-way mapping is `coordinate ID -> slot` and `slot -> coordinate ID`. The implementation reserves capacity for all coordinates at construction, so membership and first activation are O(1). These are private fields, not public accessors.

### SparseCoordinates: vector view of an ordinary sparse iterate

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

### SymmetricCoordinates: independent variables of a non-array owner

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

### SymmetricMatrixView: a separate read-only full-matrix view

The `SymmetricMatrixView <: AbstractMatrix` returned by `matrixview(X)` has shape `(p,p)`, normal matrix indexing and `p²` logical entries. Both triangles read the same independent storage. The view is live and read-only, whereas `Matrix(X)` allocates an independent dense snapshot.

The `SymmetricCoordinates` and `SymmetricMatrixView` wrappers answer different questions. Do not replace one with `reshape` or `vec` of the other: they enumerate different numbers of values.

### _ActiveIDs: protect working-set metadata

`activecoordinates(x)` for `SparseIterate`, `SymmetricSparseIterate` and `AtomIterate` returns the same private handle type:

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

### The aliasing methods: why they exist

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

### Handle lifetimes

| Handle | What can change underneath it? | Reacquisition rule |
|:--|:--|:--|
| `coordinates(x)` | Scalar values and active membership | Stable coordinate-ID mapping for the lifetime of `x`; reads remain live |
| `matrixview(X)` | Symmetric matrix values and membership | Stable matrix-position mapping for the lifetime of `X` |
| `activecoordinates(x)` | Active IDs/slot order, extent | Reacquire after any activation, compaction or same-kind state replacement |
| `activevalues(x)` | Active values/slot order, extent | Reacquire after the same operations |

A stale active handle is not a snapshot of the previous working set. Its old length cannot reflect new activations, and a compaction can move slot `a` to a different coordinate. Do not retain slot-indexed momentum history across these events without a corresponding remapping.

### Performance and conformance checks

The `SparseCoordinates` allocation tests in `test/allocations.jl` exercise wrapper creation, scalar reads/writes and zero fill on Float32/Float64 vector and matrix iterates. `test/iterate_queries.jl` exercises live views, size and aliasing behavior. [#694](https://github.com/mlakolar/ProximalBase.jl/issues/694) tracks remaining ordinary-vector conformance tests. The symmetric-owner acceptance plan has independent coverage of its two index spaces and read-only matrix view.

A useful performance comparison is `x[k]` versus `coordinates(x)[k]` for `SparseIterate`, and `X[i,j]` versus `coordinates(X)[k]` for the corresponding independent coordinate on the symmetric owner. These comparisons should be made on warm Float32/Float64 kernels; **no measured zero-time-overhead claim is made merely because the wrappers allocate nothing**.
