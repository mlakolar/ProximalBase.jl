# [Overview and working sets](@id Sparse-iterates)

*Assumes: vectors, matrices and basic optimization iterations · You will learn: when to use specialized storage and what active coordinates mean.*

For full-vector methods such as ordinary proximal gradient, a plain Julia
`Vector` (or `Matrix`) is often all the storage you need.
Use `SparseIterate`, `BlockIterate` or `SymmetricSparseIterate`
when an algorithm repeatedly visits only a selected
[working set](glossary.md#Working-set) of coordinates, groups or independent
symmetric-matrix entries. These types make working-set access and updates
convenient, but they are not compressed-memory sparse matrices.

An [iterate](glossary.md#Iterate) stores the current optimization variable.
A coordinate is **[active](glossary.md#Active-coordinate)** here when it is
explicitly stored or chosen for visits, **not** when an inequality constraint
holds with equality as in standard KKT terminology.
A zero coordinate may remain active so a solver checks it on its next sweep.

## Choosing an iterate

| Storage | Array interface | One optimization coordinate | Capacity |
|:--|:--|:--|:--|
| [`SparseIterate`](@ref) | Real vector or matrix | One scalar entry | O(number of entries) |
| [`SymmetricSparseIterate`](@ref) | Object representing a symmetric p×p matrix (not an `AbstractArray`) | One lower-triangle variable | O(p(p+1)/2) |
| [`BlockIterate`](@ref) | Real vector | One ordered group of entries | O(entries + groups) |

These provide sparse **work**, with direct lookup tables and reserved slots;
they are not compressed sparse formats for avoiding memory proportional to the
full coordinate domain. For an algorithm that updates every coordinate at each step, a regular
array is often simpler.

## Entries, coordinates and active slots

For `SparseIterate`, coordinates are Julia linear entry indices. For symmetric matrix storage, coordinates number the independent lower triangle, while matrix indices
include both mirrored entries. For `BlockIterate`, `x[j]` still indexes an entry,
but coordinate `k` names a group. An [active **slot**](glossary.md#Active-slot) is the position in a packed
list of coordinate IDs. It is not a coordinate ID and can move on compaction.

| Call | `SparseIterate` | `SymmetricSparseIterate` | `BlockIterate` |
|:--|:--|:--|:--|
| `ncoordinates(x)` | length(x) | p(p+1)/2 | number of groups |
| `nactive(x)` | stored entries, zeros included | stored independent variables, zeros included | stored groups, even all-zero groups |
| `coordinates(x)` | live writable vector in linear order | live writable lower-triangle vector | no method |
| `activecoordinates(x)` | read-only active entry IDs | read-only independent-coordinate IDs | read-only active group IDs |
| `activevalues(x)` | writable packed values | writable packed values | no method; use `blockview` |
| `matrixview(x)` | no method | read-only live matrix | no method |
| `blockview(x,k)` / `blockindices(x,k)` | no method | no method | live entry view / read-only ordered index list |

The exported `nnz` is SparseArrays' binding for standard sparse arrays. There
is no `nnz`, `nonzeros` or `findnz` method for these iterates. Count a
`SparseIterate`'s nonzeros using `count(!iszero,activevalues(x))`; for a symmetric
iterate that counts independent nonzeros, not full-matrix entries. For blocks,
sum `count(!iszero,blockview(x,k))` over active group IDs.

## Active coordinates are deliberate state

Writing a nonzero converted value activates its coordinate. Writing zero to
an absent coordinate leaves it absent; writing zero to an active coordinate
keeps it active. `activate!(x,k)` explicitly retains a zero coordinate, and
[`dropzeros!(x)`](glossary.md#Compaction) removes zero coordinates/all-zero
groups. NaN/Inf are nonzero
for this rule; signed zeros are zero.

```@example iterate_working_set
using ProximalBase
x = SparseIterate(4)
activate!(x,3)
x[1] = 2
x[1] = 0
@assert nactive(x) == 2 && count(!iszero,activevalues(x)) == 0
saved_ids = collect(activecoordinates(x))
dropzeros!(x)
@assert nactive(x) == 0 && saved_ids == [3,1]
nactive(x)
```

Acquire `activecoordinates` and `activevalues` again after any activation,
compaction or whole-state replacement. Handles retain their old extent and
refer to live slots; they are not snapshots. Coordinate and matrix views keep
their stable index mapping for the iterate's lifetime. Never modify private
metadata through fields or a view's parent indices.

## Copies and mathematical operations

`copy(x)` copies values and working-set state independently. `similar(x)` is
empty storage of the same kind/shape/layout. Same-kind state copies replace
the destination's active set when shapes/layouts match. Ordinary array writes,
broadcasts and most mathematical proximal writes preserve already active
coordinates. A fresh allocating prox can omit the source's active zeros. `ZeroFunction` is a
copying exception: its distinct array-iterate output uses `copyto!`, replacing
state for matching `SparseIterate` shapes/`BlockIterate` layouts. Its symmetric-iterate route
instead uses coordinate value writes.

For `BlockIterate`, same-layout `copyto!` replaces state; different layouts of
the same entry length copy by value. A symmetric iterate's `copyto!(x,A)` is a
checked import, replacing state with nonzero coordinates in increasing order.
See the individual guides for validation and failure behavior.

## Supported products

The following sparse-aware operations are available for these iterate types. Real types and compatible
dimensions are required. Other Julia fallbacks can be dense, slower, or absent.
Use independent product destinations: ordinary `mul!` does not permit arbitrary
aliasing. The symmetric iterate methods explicitly reject overlapping output/input storage.

| Call | Supported domain / cost |
|:--|:--|
| `A*x`, `mul!(out,A,x[,α,β])` | Strided dense matrix and vector `SparseIterate` or `BlockIterate`; O(rows(A)·active entries) |
| `mul!(out,A,B[,α,β])` | Strided matrix A and matrix `SparseIterate` B, strided matrix output; active-entry/column traversal |
| `A*x`, `A*B` and mul! | Symmetric wrapper of a strided matrix, vector/matrix `SparseIterate` or vector `BlockIterate`; respects selected triangle |
| `A'*x`, `transpose(A)*x`, mul! | Strided A and vector `SparseIterate`, strided output |
| `dot(v,x)` in either order | Strided vector and vector `SparseIterate`/`BlockIterate`; also combinations of those vector iterates |
| `dot(a,X,b)` | Strided vectors and matrix `SparseIterate`; arbitrary one-based real vectors and symmetric iterate |
| `X*v`, `X*B`, mul! | Symmetric iterate and real vector/matrix right operand; O((p+s)·number of columns) |
| `mul!(out,matrixview(X),v[,α,β])` | Symmetric-iterate view and real vector; O(p+s) |
| `dot(A,X)` in either order | Strided matrix and symmetric iterate, full Frobenius inner product |
| `A*X*B` | Symmetric wrappers of strided matrices and matrix `SparseIterate`/`SymmetricSparseIterate` of matching `Float32`/`Float64`; direct sparse terms, O(s·rows(X)·cols(X)) |

Here `s` counts stored coefficients; active blocks can contain several entries.
The direct three-factor guarantee requires all three element types to match.
Other matching-type Symmetric wrappers use a restricted/dense route; mixed
types use generic/dense multiplication. See [numerics](numerics.md) for the
three-factor range and summation guarantees.

`dot(c, X)` for a matrix `SparseIterate` `X` and a dense `c` is the generic `dot`
over every position, `O(mn)`; it is not in the table. The value of a
[`LinearFunction`](smooth.md#Linear-function) at a matrix `SparseIterate` is
computed separately from the stored nonzero entries, with the strong-zero rule below.

Specialized products skip zero iterate coefficients, including stored zeros,
even against NaN/Inf in the other operand. This differs from an ordinary
dense IEEE `0*Inf`. A coordinate view has ordinary vector products and does
not inherit sparse work or strong-zero semantics. See
[array conventions](numerical_usage.md).
