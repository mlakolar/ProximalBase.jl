# SparseIterate: storage, access, and working-set interface

This page describes **only the `SparseIterate` data structure**: what a solver can do with it, the meaning of active coordinates, and the cost of each operation. It does not define a function's value, gradient, proximal operator, restriction, or KKT screen. Those belong to [the function interface](@ref) and [coordinate descent](@ref).

## Purpose and representation

A `SparseIterate{T,N}` is a mutable real-valued Julia `AbstractArray{T,N}`, with `N = 1` (vector) or `2` (matrix). It is intended for coordinate-descent and active-set algorithms that need **constant-time lookup and insertion by coordinate**, alongside **fast traversal of the active working set**.

The essential design is three arrays, reserved at construction:

- An integer map from each logical coordinate ID to its active storage slot, or zero if inactive.
- A packed list of active coordinate IDs, in slot order.
- A packed list of active values, in that same order.

The internal fields are **private**. Solvers use the public methods below rather than `x.nzval`, `x.nzval2ind`, `x.ind2nzval` or `x.nnz`.

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
| Counts | `nactive(x)`, `nnz(x)` | `O(1)` | Both count **stored** entries, even those equal to zero |
| Active IDs | `activecoordinates(x)` | `O(1)` to obtain; `O(s)` to traverse | Read-only handle in active-slot order, **not sorted** |
| Active values | `activevalues(x)`, `nonzeros(x)` | `O(1)` to obtain; `O(s)` to traverse | Writable handles; pair with active IDs at the **same slot** |
| All coordinates | `coordinates(x)` | `O(1)` to obtain; `O(n)` to traverse | Live writable `AbstractVector`, even when `x` is a matrix |
| Remove active zeros | `dropzeros!(x)` | `O(s)` | May reorder active slots; does not change mathematical values |
| Export stored entries | `findnz(x)` | `O(s \log s)` | Allocates sorted index/value copies; includes stored zeros |
| Dense snapshot | `Vector(x)`, `Matrix(x)` | `O(n)` | Allocates independent dense output |
| Allocate/copy iterate | `similar(x)`, `copy(x)` | `O(n)` storage | `similar` is empty; `copy` preserves active state |
| State replacement | `copyto!(x,y)` for same-size `SparseIterate` objects | `O(n)` worst case | Preserves source active IDs, slot order and explicit zeros |
| Ordinary value write | `copyto!(x,A)`, `fill!(x,v)`, broadcasting | Usually `O(n)` | Does not automatically remove previously active coordinates |

Here `s = nactive(x)`. The lookup and insertion fast paths use an already allocated reverse map and packed storage. The implementation also supplies sparse-aware linear-algebra kernels (below), but they are **not** a replacement for this storage/working-set API.

## Active coordinates are not mathematical nonzeros

`nactive(x)` counts entries deliberately retained in the working set. An active entry can contain `0.0`, including after an update. This is necessary when solvers keep a stable set of coordinates throughout a sweep.

```julia
using ProximalBase
using SparseArrays: nnz, dropzeros!

x = SparseIterate(5)
x[2] = 3.0
activate!(x, 4)             # active, although its value is zero

ncoordinates(x)             # 5
nactive(x)                  # 2
nnz(x)                      # 2, not 1
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

A matrix `SparseIterate` has selected kernels with stored-entry or column-wise traversal; a three-factor `A * X * B` fast path has a more restrictive supported type/domain contract. **Do not assume every generic Julia product is sparse-aware.** For instance, a product through `coordinates(x)` is a product of an ordinary vector view and can traverse all `n` entries. See [the interface](@ref) and [the numerical contract](@ref) for the supported operation-specific rules.

## Implementation and efficiency checklist

The data structure's essential invariants and performance targets are:

1. **Direct reverse lookup:** `coordinate ID -> slot` permits `O(1)` reads, writes, membership checks and first activation, with no per-update search or resizing.
2. **Packed active traversal:** `slot -> coordinate ID, value` permits `O(s)` working-set passes and compaction. Active slots may move, but coordinate IDs retain their meaning.
3. **Dense-sized reserved storage:** `O(n)` construction and memory are the price of predictable `O(1)` operations. Do not replace this with a hash-table or compressed sparse layout without a motivating, measured workload.
4. **No accidental deactivation:** value writes preserve active status; `dropzeros!` removes zeros, and same-kind state copies explicitly replace state.
5. **No hidden full traversal:** use active handles for sparse sweeps; only traverse all `n` entries for genuinely dense-array operations.
6. **Allocation behavior:** indexing, activation, membership and obtaining handles should be allocation-free after warm-up on ordinary Float32/Float64 CPU use. Copies, construction, sorted exports and dense snapshots allocate intentionally.

This page is about the representation, not which mathematical **functions** accept it. The latter compatibility belongs to the function-oracle inventory and its tests, not to the `SparseIterate` contract.


For the definition-by-definition inventory, including internal aliasing hooks, wrapper methods, conditional `copyto!` overloads, kernels and out-of-file specializations, see [the 0.6 source-level SparseIterate audit](../audit/sparse-iterate-06-method-inventory.md). This documents the source at the time of the audit; it does not imply that Julia method-table probes or tests have run.

## Inventory: implemented versus inherited methods

This table is an index of the **data-structure API**, not a declaration that every method is separately specialized or sparse-optimized. "Explicit" refers to methods in `src/sparse_iterate.jl` (including shared stored-iterate helpers); "Julia generic" means ordinary `AbstractArray` behavior. Costs for generic array methods may be `O(length(x))`.

| Category | Methods | Implementation | Sparse-specific guarantee |
|:--|:--|:--|:--|
| Construction | `SparseIterate(n)`, `SparseIterate(T,n)`, `SparseIterate(n,m)`, `SparseIterate(T,n,m)` | Explicit | Reserves `O(length(x))` storage |
| Construction from data | `SparseIterate(::Array)`, `SparseIterate(::SparseVector)`, `SparseIterate(::SparseMatrixCSC)` | Explicit | Creates state from stored/nonzero values |
| Shape and indexing | `size`, `length`, `IndexStyle`, scalar `getindex`, scalar `setindex!` | Explicit | `O(1)` scalar access and insertion |
| Other array metadata | `eltype`, `ndims`, `axes`, `eachindex`, `firstindex`, `lastindex` | Julia generic | Ordinary array conventions |
| General array operations | Iteration, ranged/indexed access, `view`, `reshape`, `vec`, `map`, `map!`, array reductions | Julia generic | **Not** generally `O(nactive)` |
| Coordinates | `ncoordinates`, `coordinates` and the view's `size`, `getindex`, `setindex!`, `fill!` | Explicit | No-copy `O(1)` view creation, `O(1)` entry access |
| Active-set management | `nactive`, `activecoordinates`, `activevalues`, `isactive`, `activate!`, `dropzeros!` | Explicit | `O(1)` lookup/activation, `O(nactive)` traversal/compaction |
| SparseArrays interface | `issparse`, `nnz`, `nonzeros`, `findnz` | Explicit | `findnz` allocates sorted copies |
| State copy | `copy`, `similar`, `similar(x,T)`, same-kind `copyto!`, `copy!` | Julia generic plus explicit state-copy kernels | Active coordinates and explicit zeros preserved for compatible same-kind sources |
| Value writes | `copyto!` from `AbstractArray`, `fill!`, broadcasting and broadcast assignment | Explicit dispatch and Julia indexing | Old active entries remain until `dropzeros!` |
| Special array sources | Matrix `copyto!` from `UniformScaling`, supported sparse wrappers and `AbstractQ` orthogonal factors | Explicit | Compatibility paths, not core solver fast paths |
| Dense snapshots | `Vector(x)`, `Matrix(x)`, `Array(x)` | Explicit | `O(length(x))` |
| Comparison and display | `==`, `isequal`, `hash`; `show` | Julia generic for comparisons; explicit `show` | Comparisons depend on values, not active state |
| Norms/differences | `norm`; `norm_diff` | Explicit sparse norm; generic difference utility | Do not claim all differences cost `O(nactive)` |
| Sparse vector products | `dot(v,x)`, `dot(x,v)`, `dot(x,y)`, `A*x`, `mul!(out,A,x)`, five-argument `mul!` | Explicit for documented operand domains | Sparse-aware active-entry work |
| Sparse matrix products | `dot(a,X,b)`, selected `A*X`, `mul!(out,A,X)`, selected `A*X*B` | Explicit for documented operand domains | Not every general matrix product is specialized |
| Scalar scaling | `rmul!`, `lmul!`, `rdiv!`, `ldiv!` | Explicit | Active-only fast path where scalar maps zero to zero |
| Aliasing infrastructure | `Base.dataids(x)`, coordinate-view `dataids`/`unaliascopy` | Explicit | Prevents aliased array copies from reading overwritten data |

**Not a separate iterate interface:** `restrict`, `kkt_violation`, `prox_coordinate!`, `prox!`, `value_and_prox!`, `gradient!` and `hessian_vec!`. Their APIs belong to function oracles; individual methods may accept `SparseIterate` as an input or destination.

**Not required for 0.6:** shape resizing (`resize!`, `push!`), a new generic iterate hierarchy, an `O(nactive)` implementation of every Julia array reduction, full specialization of all matrix operators, dynamic sparse-capacity allocation, or new mixed-precision/GPU fast paths.

### Remaining SparseIterate-specific follow-ups

- [#694](https://github.com/mlakolar/ProximalBase.jl/issues/694): test the `SparseCoordinates` view against ordinary-vector indexing, `sortperm`, and display conventions on supported Julia versions. This is a **conformance test gap**; it does not establish a missing method.
- [#697](https://github.com/mlakolar/ProximalBase.jl/issues/697): correct documentation for complex values converted through wrappers and same-shape state copies.
- [#797](https://github.com/mlakolar/ProximalBase.jl/issues/797): optional optimization of the already implemented, correctness-first orthogonal-factor copy path.
- [#341](https://github.com/mlakolar/ProximalBase.jl/issues/341): outstanding sparse *function-value* implementations, tracked separately from the iterate's data-structure API.
- [#456](https://github.com/mlakolar/ProximalBase.jl/issues/456): the shared SparseIterate/AtomIterate release umbrella; open status alone does not imply a missing SparseIterate method.

The former all-zero clear optimization [#644](https://github.com/mlakolar/ProximalBase.jl/issues/644) is **closed**; the specialized `dropzeros!(::SparseIterate)` implementation is on the current branch.

