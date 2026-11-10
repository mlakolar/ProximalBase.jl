# SparseIterate: storage, access, and working-set interface

This page describes **only the `SparseIterate` data structure**: what a solver can do with it, the meaning of active coordinates, and the cost of each operation. It does not define a function's value, gradient, proximal operator, restriction, or KKT screen. Those belong to [the function interface](interface.md) and [coordinate descent](coordinate_descent.md).

## Purpose and representation

A `SparseIterate{T,N}` is a mutable real-valued Julia `AbstractArray{T,N}`, with `N = 1` (vector) or `2` (matrix). It is intended for coordinate-descent and active-set algorithms that need **constant-time lookup and insertion by coordinate**, alongside **fast traversal of the active working set**.

The implementation reserves capacity for every logical coordinate, even though only active entries are traversed during a sparse sweep. Its private storage layout is explained in the [developer guide](developer/sparse_storage.md), not part of the interface a solver needs to access.

**Sparse work, not sparse memory.** The reverse map and active-value capacity are allocated for **all** `n = length(x)` coordinates. Construction and retained storage cost `O(n)`; active traversal costs `O(s)`, where `s = nactive(x)`. Unlike a `SparseVector` or `SparseMatrixCSC`, this is not a memory-compressed representation of a huge dimension with tiny support. Its design favors cheap coordinate updates without searching or reallocating.

For a matrix, a coordinate ID is its standard Julia column-major linear index, `k = (j - 1) * size(x, 1) + i`; there is no separate matrix-coordinate numbering.

## Public interface and costs

The costs below describe the `SparseIterate` operation itself, after construction. Obtaining a view is `O(1)`; iterating over its elements costs the number visited. A constant-time claim is not a claim that every generic Julia operation is constant time.

| Purpose | Public method | Cost | Notes |
|:--|:--|:--|:--|
| Shape and element type | `size(x)`, `length(x)`, `eltype(x)` | `O(1)` | Vector or matrix; ordinary Julia array shape |
| Total coordinates | `ncoordinates(x)` | `O(1)` | Equal to `length(x)` |
| Scalar value | `x[k]`, `x[i,j]` | `O(1)` | Inactive coordinates read as zero; matrix uses normal Julia indexing |
| Scalar assignment | `x[k] = v`, `x[i,j] = v` | `O(1)` | Converts to `T`; a nonzero value activates; a zero does not deactivate |
| Membership | `isactive(x,k)` | `O(1)` | Tests stored membership, including explicitly active zero |
| Explicit insertion | `activate!(x,k)` | `O(1)` | If absent, inserts an active zero; otherwise no change |
| Counts | `nactive(x)` | `O(1)` | Counts **active stored** entries, even those equal to zero |
| Active IDs | `activecoordinates(x)` | `O(1)` to obtain; `O(s)` to traverse | Read-only handle in active-slot order, **not sorted** |
| Active values | `activevalues(x)` | `O(1)` to obtain; `O(s)` to traverse | Writable handles; pair with active IDs at the **same slot** |
| All coordinates | `coordinates(x)` | `O(1)` to obtain; `O(n)` to traverse | Live writable `AbstractVector`, even when `x` is a matrix |
| Remove active zeros | `dropzeros!(x)` | `O(s)` | May reorder active slots; does not change mathematical values |
| Dense snapshot | `Vector(x)`, `Matrix(x)` | `O(n)` | Allocates independent dense output |
| Allocate/copy iterate | `similar(x)`, `copy(x)` | `O(n)` storage | `similar` is empty; `copy` preserves active state |
| State replacement | `copyto!(x,y)` for same-size `SparseIterate` objects | `O(n)` worst case | Preserves source active IDs, slot order and explicit zeros |
| Ordinary value write | `copyto!(x,A)`, `fill!(x,v)`, broadcasting | Usually `O(n)` | Does not automatically remove previously active coordinates |

Here `s = nactive(x)`. The data structure provides direct access to coordinates and fast traversal of its working set. The implementation also supplies sparse-aware linear-algebra kernels (below), but they are **not** a replacement for this storage/working-set API.

## Active coordinates are not mathematical nonzeros

`nactive(x)` counts entries deliberately retained in the working set. An active entry can contain `0.0`, including after an update. This is necessary when solvers keep a stable set of coordinates throughout a sweep.

```julia
using ProximalBase
using SparseArrays: dropzeros!

x = SparseIterate(5)
x[2] = 3.0
activate!(x, 4)             # active, although its value is zero

ncoordinates(x)             # 5
nactive(x)                  # 2, even though only one value is nonzero
activecoordinates(x)        # [2, 4]
activevalues(x)             # [3.0, 0.0]

x[2] = 0.0                 # coordinate 2 remains active
nactive(x)                  # 2
dropzeros!(x)               # remove active zero entries
nactive(x)                  # 0
```

A missing coordinate is always zero. Writing a nonzero value to a missing coordinate makes it active; writing zero to a missing coordinate does not. `activate!` is the intentional way to store a zero. `NaN` and `Inf` count as nonzero for activation and are never removed by `dropzeros!`.

**Do not treat the active IDs as sorted.** New entries append to the slot list. Compaction removes a zero by moving another active slot into its place. After activation, `dropzeros!` or whole-state replacement, reacquire `activecoordinates(x)` and `activevalues(x)`. Older views have their old extent and may no longer correspond to the current slot mapping.

For efficient active traversal, use the two handles **together**:

```julia
ids = activecoordinates(x)
vals = activevalues(x)
for a in eachindex(ids)
    k = ids[a]
    v = vals[a]
    # operate on coordinate k and its stored value v
end
```

This loop is `O(s)`; `for k in eachindex(x)` is the ordinary `O(n)` array traversal.

## Ordinary array behavior and copying

`SparseIterate` remains an `AbstractArray`, so ordinary iteration, indexing, broadcasting, `fill!`, `copy!`, `copyto!` and equality describe its **mathematical values**. Equality and hashing do not depend on which zeros happen to be active. The elements must be real.

The distinction between a value write and a solver-state copy is deliberate:

| Call | Meaning for the working set |
|:--|:--|
| `x[k] = v`, `x .= A`, `fill!(x,v)` | Value writes: never deactivate an existing coordinate |
| `copyto!(x, A)` for an ordinary array `A` | Elementwise value write; previously active zeros remain active |
| `copyto!(x, y)` for a same-size `SparseIterate` `y` | **State copy**: replace the active IDs, order, stored zeros and values |
| `copy(x)` | Independent state copy |
| `similar(x)` | New zero-valued iterate with no active coordinates |
| `dropzeros!(x)` | Explicit compaction of active zeros |

For a `SparseIterate`, the supported clear is `fill!(x, 0); dropzeros!(x)`. The first step changes values; the second removes the active zeros. To replace the working set with the nonzero pattern of an ordinary dense vector, first clear the old working set and then copy the new values, or stage a new iterate and state-copy it when aliasing or failure atomicity matters.

```julia
x = SparseIterate(3)
activate!(x, 1)

copyto!(x, [0.0, 4.0, 0.0])
activecoordinates(x)       # [1, 2]: the old active zero remains

dropzeros!(x)
activecoordinates(x)       # [2]

y = SparseIterate(3)
activate!(y, 3)
copyto!(x, y)
activecoordinates(x)       # [3]: source state replaces destination state
```

Value writes through `coordinates(x)`, views, `vec` and `reshape` follow ordinary Julia conversion and indexing conventions; unlike the iterate's own bulk-write preflight, such wrappers need not reject an invalid value before earlier entries have been written. A failed same-kind state copy, by contrast, must leave the destination unchanged if a stored value cannot convert to its element type.

## Sparse linear algebra: an additional, narrower guarantee

ProximalBase implements selected sparse-aware methods for `SparseIterate`, including `dot`, `mul!`, `A*x`, and matrix bilinear/three-factor products in their documented dispatch domains. Those methods walk active entries and skip mathematically zero coefficients. For a vector `x` with `s` active entries, supported `dot(v,x)` is `O(s)`, and `mul!(out,A,x)` is approximately `O(size(A,1) * s)`.

A matrix `SparseIterate` has selected kernels with stored-entry or column-wise traversal; a three-factor `A * X * B` fast path has a more restrictive supported type/domain contract. **Do not assume every generic Julia product is sparse-aware.** For instance, a product through `coordinates(x)` is a product of an ordinary vector view and can traverse all `n` entries. See [the interface](interface.md) and [the numerical contract](@ref numerics) for the supported operation-specific rules.

## Implementation details

The [SparseIterate developer guide](developer/sparse_storage.md) covers the internal representation, invariants, Julia dispatch and aliasing hooks, method inventory, and remaining internal issues. For coordinate-level proximal and screening operations, use the [coordinate oracles guide](iterate_oracles.md).
