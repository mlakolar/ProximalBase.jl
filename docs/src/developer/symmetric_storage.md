# SymmetricSparseIterate: owner implementation

This page explains the implementation decisions behind the
[user-facing symmetric iterate](../symmetric_sparse_iterate.md).
The [owner contract](contracts/symmetric-sparse-iterate-0.6.md) and
[acceptance plan](contracts/symmetric-sparse-iterate-0.6-acceptance.md)
are authoritative; this page is a contributor map, not a second specification.

## Why it is a non-array owner

`SymmetricSparseIterate{T}` represents a real `p × p` symmetric matrix but
is **not** an `AbstractMatrix`: independent optimization variables number
`q = p(p+1)/2`, whereas matrix entries number `p²`. Exposing one-index
owner operations would leave `X[k]` ambiguous. Consequently the owner
provides `X[i,j]`, `size(X)`, and explicit representations, but deliberately
does not expose ordinary array `length`, `axes`, iteration or broadcast.
The defined `axes(X)` throws to stop accidental generic array dispatch.

A user must choose the intended space:

| Representation | Index | Number | Mutation |
|:--|:--|:--|:--|
| `X[i,j]` | Matrix pair | `p²` addressable positions | Yes; mirrored pair is one coordinate |
| `coordinates(X)[k]` | Independent lower-triangle coordinate | `q` | Yes |
| `matrixview(X)[i,j]` | Full matrix position | `p²` | No |
| `activecoordinates(X)[a]` | Packed active slot to coordinate ID | `nactive(X)` | No; use active values |

`tril_index` and `tril_cartesian` define the stable lower-triangle mapping.
Changing the active-set storage order must not change those coordinate IDs.

## Internal representation and shared kernels

`src/sparse_iterate.jl` uses the same stored-coordinate primitives as
`SparseIterate`: `nzval`, `nzval2ind`, `ind2nzval` and `nnz` form a direct
coordinate-to-slot map and a packed active list. The symmetric owner adds `p`.
These are private fields; solvers must use public accessors.

For every active slot `a`, the reverse mapping to its coordinate ID must
return `a`; inactive coordinates have a zero map entry. `_getstored`,
`_setstored!`, `activate!`, and compaction act on *independent* coordinates,
not both mirrored positions. A scalar pair write maps `(i,j)` and `(j,i)`
to the same ID; it never performs two independent writes.

The shared `_ActiveIDs` wrapper protects active-set membership metadata.
`activevalues` is writable and paired with the ID handle, but both handles
must be reacquired after activation, `dropzeros!`, and any state replacement.
The two stable live views are covered in [Coordinate views and aliasing](views_aliasing.md).

## Public and private matrix views

The public `SymmetricMatrixView <: AbstractMatrix` is read-only and live.
Its array integration provides ordinary matrix read semantics, and selected
products and conversions exploit the active set. The internal `_OwnerMatrix`
is a *different* mutable wrapper for trusted kernels that must write matrix
positions into owner storage; it is not a public escape hatch.

Read-only behavior must be tested through derived `view`, `reshape`,
`transpose`, and `Symmetric` wrappers, as well as the direct view. Aliasing
hooks must identify the owner as sharing data with both coordinate and
matrix views. `Matrix(X)` is an independent dense snapshot, not another handle.

## Multiplicity, dispatch, and numerical policy

A nonzero off-diagonal independent variable represents two matrix entries.
Matrix-oriented values, gradients and Frobenius proximal steps account for
that multiplicity; the coordinate vector intentionally does not. A careless
call to generic vector algorithms on `coordinates(X)` may therefore solve a
different optimization problem.

Built-in function oracles which explicitly accept the owner follow the
mathematical function's interface. This is distinct from the storage
operations; never add generic array methods to make an unsupported oracle
appear to work. For pairwise parameter rules and the symmetric part of
proximal input points see [Symmetric parameters](../symmetric_parameters.md).
For result precision, strong-zero products and non-finite behavior see
the [numerical contract](@ref numerics).

## Review and tests

Consult the owner contract's `OWN`, `IDX`, `CRD`, `MAT`, `CPY`, `ACT`,
`ORC`, `VAL`, and `PERF` requirements and the companion acceptance plan.
Relevant tests include `test/symmetric_iterate_indexing.jl`,
`test/symmetric_iterate_copies.jl`, `test/matrixview.jl`,
`test/symmetric_oracles.jl`, and conformance cases. In particular verify:

1. Mirrored writes refer to one coordinate and one activation.
2. The owner refuses accidental one-index, axis, and array traversal.
3. Coordinate and matrix views use the correct index spaces, with no
   write through any read-only wrapper.
4. State copy, dense import, failure-before-write and alias rejection
   have their distinct semantics.
5. Matrix operations count off-diagonal entries twice where appropriate
   and remain consistent across compaction.
