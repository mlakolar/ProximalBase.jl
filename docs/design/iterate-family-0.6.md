# Iterate-family contract

This contract defines the public storage behavior of `SparseIterate` and
`BlockIterate`. The independent-coordinate behavior of
`SymmetricSparseIterate` is documented separately. These requirements
describe the current implementation and supersede the older full-family
design preserved in the [historical snapshot](https://github.com/mlakolar/ProximalBase.jl/tree/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design).
They do not introduce new performance acceptance targets.

## Representations and coordinate spaces

| Type | Julia representation | One optimization coordinate |
|:--|:--|:--|
| `SparseIterate` | Real `AbstractArray` (vector or matrix) | One scalar array entry, in column-major order |
| `BlockIterate` | Real `AbstractVector` with an ordered partition | One group of entries |

Both representations reserve storage proportional to the **full entry or
coordinate domain**, not just the number of active variables. They are
designed for sparse *work*, not compressed sparse *memory*.

Coordinate reads, writes, membership tests and first activation use
direct-address structures. An **active coordinate** is a position retained
in the working set. It may contain an exact numerical zero.

## Activity and zero values

The observable behavior is:

- Writing a converted **nonzero** value to an inactive entry/group activates
  it. Writing zero to an inactive entry/group does not.
- Writing zero to an **already active** entry/group does not deactivate it.
  `activate!(x, k)` can explicitly activate a zero-valued coordinate/group.
- `nactive(x)` counts stored coordinates or groups, including active zeros.
- `dropzeros!(x)` removes entries equal to zero (or groups whose entries are
  all zero). It does not remove `NaN` or `Inf`.
- Positive and negative signed zeros are numerical zeros for activation and
  compaction; `NaN` and `Inf` are nonzero.
- An active coordinate **ID** is not its position in the packed active list.
  Packed slots may change when activity is modified.

For example:

```julia
x = SparseIterate(4)
activate!(x, 2)       # keep coordinate 2 active, even though x[2] == 0
@assert isactive(x, 2)
@assert nactive(x) == 1
dropzeros!(x)         # remove the active zero
@assert nactive(x) == 0
```

The `activecoordinates(x)` result is a read-only, unsorted **live handle**
to active IDs; for `SparseIterate`, `activevalues(x)` is a writable
handle to values in the matching packed-slot order. Reacquire both
handles after activation, compaction, or replacing storage state.

## Views, groups and copying

- `coordinates(x)` for a `SparseIterate` is a live writable vector.
  A `BlockIterate` instead exposes `blockindices(x, k)` and
  `blockview(x, k)`; it does not expose `coordinates` or `activevalues`.
- Groups are validated, ordered, and immutable **in use**. Vector group
  inputs are copied. Copies and `similar` may share immutable layout
  metadata.
- `ProximalBase.samegroups(x, y)` compares ordered group partitions.
  Any cached equality result must be synchronized when objects are shared
  concurrently.
- `copy(x)` makes independent mutable values and active state.
  `similar(x)` creates zero-valued empty storage.
- For matching same-kind layouts, `copyto!(dest, src)` replaces active
  state, preconverting potentially failing stored-value conversions.
  For different compatible shapes or group layouts with the same entry
  length, copying instead proceeds by numerical entries.
- Ordinary value writes and broadcasting preserve existing activity.
  Clearing active state requires setting values to zero followed by
  `dropzeros!(x)`.
- The exact `copyto!` behavior for a `SymmetricSparseIterate` matrix
  import is documented in the symmetric-owner contract.
- `blockview(x, k)` writes through public entry setters. Copying an entire
  block follows Julia's prefix, alias and conversion behavior; it is not
  guaranteed to be an atomic state copy.

## Validation and partial writes

Direct bulk writers on array iterates reject predictable complex-type and
shape errors before writing. Ordinary views, abstractly typed inputs, and
fallible per-element conversions can still fail after partial writes.
**The iterate's storage invariants must remain consistent even on failure.**

`nnz` is re-exported for standard `SparseArrays` types, but it is
**not** a method on these iterates. Use `nactive(x)` for working-set
cardinality; explicitly stored zeros mean that this need not equal a
mathematical count of nonzero entries.

## Mathematical operations and storage

Storage methods do not automatically provide a mathematical function
operation. A scalar `prox_block(f, v, i, γ)` returns a numerical entry
result. The compact-vector extension `prox_block!(out, f, u, k, γ)`
writes a numerical output buffer. Neither automatically updates
`SparseIterate` or `BlockIterate` storage.

The caller constructs the trial, computes the proximal result in
independent scratch, writes the result through public storage, and updates
any residuals. Do not access private metadata to synthesize a storage-aware
proximal update.

Supported sparse-aware products skip **zero iterate coefficients** even
when the corresponding entry of another operand is `NaN` or `Inf`.
In contrast, an ordinary coordinate view uses the usual IEEE expression
``0\cdot\infty``, which can yield `NaN`. The exact supported
matrix/vector combinations and numerical accuracy promises are specified
in the user product table and numerical contract. No universal
zero-allocation, atomicity, or sparse-time guarantee applies to arbitrary
inherited `AbstractArray` methods.

## Internal representation and tests

Current implementations use packed active value/ID lists and reverse maps
for `SparseIterate`, and a dense value vector plus group/position maps
for `BlockIterate`. Fields, wrapper types, thresholds and cache identities
are private implementation details, not public extension hooks.

Relevant test suites include `test/iterate_invariants.jl`,
`test/iterate_queries.jl`, `test/active_handle.jl`,
`test/iterate_writes.jl`, `test/real_iterates.jl`,
`test/block_copy.jl`, `test/sparse_iterate_basics.jl`,
`test/sparse_iterate_basics_grids.jl`,
`test/symmetric_products.jl` and `test/matrixview.jl`.
Run the test inventory specified by `test/tiers.toml`.
Historical acceptance plans describe a broader package and are not
current method inventories or evidence of tests run today.
