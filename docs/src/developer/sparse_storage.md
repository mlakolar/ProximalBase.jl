# SparseIterate: storage internals and method audit

**Contributor material.** See the [public SparseIterate guide](../sparse_iterate.md) for construction, the working-set API, examples, and user-visible costs. This page records the implementation and 0.6 compatibility decisions; the [iterate-family contract](contracts/iterate-family-0.6.md) is authoritative for normative requirements.

## Implementation and efficiency checklist

The data structure's essential invariants and performance targets are:

1. **Direct reverse lookup:** `coordinate ID -> slot` permits `O(1)` reads, writes, membership checks and first activation, with no per-update search or resizing.
2. **Packed active traversal:** `slot -> coordinate ID, value` permits `O(s)` working-set passes and compaction. Active slots may move, but coordinate IDs retain their meaning.
3. **Dense-sized reserved storage:** `O(n)` construction and memory are the price of predictable `O(1)` operations. Do not replace this with a hash-table or compressed sparse layout without a motivating, measured workload.
4. **No accidental deactivation:** value writes preserve active status; `dropzeros!` removes zeros, and same-kind state copies explicitly replace state.
5. **No hidden full traversal:** use active handles for sparse sweeps; only traverse all `n` entries for genuinely dense-array operations.
6. **Allocation behavior:** indexing, activation, membership and obtaining handles should be allocation-free after warm-up on ordinary Float32/Float64 CPU use. Copies, construction, explicitly sorted snapshots of active entries, and dense snapshots allocate intentionally.

This page is about the representation, not which mathematical **functions** accept it. The latter compatibility belongs to the function-oracle inventory and its tests, not to the `SparseIterate` contract. For `SparseCoordinates`, `SymmetricCoordinates` and the private aliasing hooks in detail, see [Developer documentation](index.md).


For the definition-by-definition inventory, including internal aliasing hooks, wrapper methods, conditional `copyto!` overloads, kernels and out-of-file specializations, see [the 0.6 source-level SparseIterate audit](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/audit/sparse-iterate-06-method-inventory.md). This documents the source at the time of the audit; it does not imply that Julia method-table probes or tests have run.

## Storage invariants and update implementation

The representation is a dense-sized direct-address table plus a compact active list:

```julia
# Conceptual internal fields, NOT a public access API:
nzval::Vector{T}          # active values occupy slots 1:s
nzval2ind::Vector{Int}    # slot a -> coordinate ID k
ind2nzval::Array{Int,N}   # coordinate ID k -> slot a; 0 when inactive
nnz::Int                  # internal active-slot count, s
```

The internal field name `nnz` **is not** the removed public `SparseArrays.nnz(x)` method. It may remain private without exposing a SparseArrays API.

The critical invariant, for every active slot `a`, is `ind2nzval[nzval2ind[a]] == a`. Every inactive coordinate has a zero reverse-map entry and reads as `zero(T)`. The packed ID list and packed values must always have the same active length. The allocated tails beyond `s` are not logical values and must not affect correctness.

A scalar read is one direct-address lookup plus either `zero(T)` or the stored value. A scalar write converts to `T` before deciding activation, then either changes an existing slot, appends a newly nonzero coordinate, or leaves an inactive zero absent. `activate!(x,k)` deliberately appends a **zero-valued** active slot if absent. These paths require no search or resize.

`dropzeros!(x)` scans active slots only. When a slot contains zero, it can move the last surviving slot to the vacancy and repair the reverse map. Consequently the active list is not sorted or insertion-order stable after compaction. For zero-valued full clearing, the optimized all-zero path avoids unnecessary slot movement. Do not retain slot-number-indexed momentum history across compaction or same-kind state replacement without remapping it.

### Value semantics versus working-set semantics

An inactive zero and an explicitly active zero have the same numerical value but different working-set meaning. Updating an active coefficient to zero does not remove it. A working-set-aware algorithm therefore has to choose when to compact, instead of equating `iszero(x[k])` with `!isactive(x,k)`.

The following code is a safe active-set sweep **only when it does not activate or compact during traversal**:

```julia
ids = activecoordinates(x)
for a in eachindex(ids)
    k = ids[a]
    x[k] = update_coefficient(x[k], k)  # updates existing active slots
end
dropzeros!(x)  # after the sweep, if compaction is desired
```

A read-only `activecoordinates(x)` view prevents accidental edits to internal ID metadata. `activevalues(x)` is writable because changing a stored value preserves the group of active coordinates. Use both views in matching slot order; reacquire them after membership-changing operations.

## Conversion, array compatibility and aliasing

A `SparseIterate` is intentionally an `AbstractVector` or `AbstractMatrix`. Shape, range indexing, views, iteration, broadcasting and reductions follow Julia's array conventions. These **generic** methods often traverse all `n` logical entries. For example, `sum(x)` is not promised to be `O(s)` just because `x` has few active coordinates.

The explicit `SparseCoordinates` view differs from the iterate even for a vector: `coordinates(x)` is always an `AbstractVector` of length `ncoordinates(x)`, with ordinary dense-vector arithmetic dispatch. Its O(1) linear access directly invokes the same stored-index primitives as `x[k]`. This provides shared coordinate-descent code with `SymmetricSparseIterate`, whose `x[k]` intentionally has no scalar-coordinate meaning. **The view is not a packed active-values array.**

The implementation also supplies three internal aliasing hooks that a code review must not overlook:

```julia
Base.dataids(x::SparseIterate)
Base.dataids(c::SparseCoordinates)
Base.unaliascopy(c::SparseCoordinates)
```

The first identifies all three storage arrays as potentially aliased with public views. The second forwards the wrapper's alias identity. The third creates `coordinates(copy(c.x))`, which is independent of the original iterate but preserves wrapper semantics when Julia copies an overlapping broadcast or indexed source. The read-only `_ActiveIDs` wrapper has its own `size`, `IndexStyle`, `getindex`, rejecting `setindex!`, `dataids` and `unaliascopy` methods. These are **private Julia integration details**, not additional user-facing storage APIs.

The iterate's own bulk writes reject complex-typed sources before mutation where the contract promises preflight. Writes through ordinary Julia views instead use Base's scalar conversion rules; converting a complex number with **nonzero imaginary part** to a real element type raises `InexactError`, and an earlier element may already have been changed. A zero imaginary part can convert successfully.

### Copying and clearing: important dispatch distinctions

For equal-sized `SparseIterate` objects, `copyto!(dest,src)` performs a state copy: source active IDs, explicit zeros and active-slot order replace the destination's. Conversion checks are staged when needed so that a failing conversion leaves the destination unchanged. When shapes differ but lengths agree, this is a linear-order value write; unequal lengths are rejected.

`copyto!(dest,A::AbstractArray)` from an ordinary array and `dest .= rhs` perform value writes, including when `rhs` is an iterate. They retain any previously active zero slots, and new nonzero values activate their coordinates. The explicit broadcast implementation prevents Base's identity shortcut from turning `dest .= rhs` into a state-copy operation.

For a fresh working set equal to the converted nonzero pattern of an ordinary array `A`, the cheap spelling is `copyto!(dropzeros!(fill!(x, 0)), A)` **only when** the source does not alias `x` and conversion/read cannot fail. Otherwise construct a temporary `t = similar(x)`, populate it from `A` before touching `x`, then use `copyto!(x,t)`. These choices are intentionally distinct from elementwise copying.

## Why the four SparseArrays aliases are removed

`SparseIterate` is an optimization iterate with **O(n) reserved memory** and independent active-state metadata, not a CSC-compressed sparse array. The 0.6 interface deliberately does not implement `SparseArrays.nnz(x)`, `nonzeros(x)`, `findnz(x)`, or `issparse(x)` for this type. No intrinsic algorithmic requirement for these aliases was identified. Their removal avoids suggesting a broader `AbstractSparseArray` contract.

| Former specialized call | Replacement |
|:--|:--|
| `nnz(x)` | `nactive(x)`, including explicitly active zero slots |
| `nonzeros(x)` | `activevalues(x)`, a writable active-slot view |
| `findnz(x)` | `collect(zip(activecoordinates(x), activevalues(x)))` for unsorted stored entries; sort explicitly when sorted output is needed |
| `issparse(x)` | Check for a specific iterate type or use ordinary `AbstractArray` dispatch; do not branch on this trait |
| `dropzeros!(x)` | **Unchanged**: required for active-set compaction |

These are removed **only for `SparseIterate`**. Ordinary `SparseVector` / `SparseMatrixCSC` APIs remain unchanged. This is a breaking 0.6 API change, including the removal of behavior that the earlier signed-off FC-11 contract had required.

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
| SparseArrays compatibility | `dropzeros!` only; no `SparseIterate` specializations for `issparse`, `nnz`, `nonzeros`, `findnz` | Deliberate 0.6 policy | `nactive`, `activevalues`, `activecoordinates` are canonical |
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
