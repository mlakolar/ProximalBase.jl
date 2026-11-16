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

## [Real values and write validation](@id iterate-write-validation)

`SparseIterate` and `BlockIterate` store real values. Their constructors and
direct bulk writes reject a known complex element type, even when every
imaginary part is zero. This applies to two-argument `copyto!` from an array,
`copy!`, broadcasting into the iterate and `fill!`; matrix `SparseIterate`
also checks `UniformScaling` and orthogonal-factor sources. The rejection is
an `ArgumentError` before any destination write.

Broadcasts check both their input leaves and their inferred result. A known
complex result is rejected; a result of an uncertain non-real type, such as
`Any`, `Number` or a union with a complex member, is evaluated before writing
and rejected if any element is complex. The destination is unchanged.

These checks do not make every write atomic. Two-argument `copyto!` and
`copy!` inspect source types, so tuples, generators and arrays such as
`Vector{Any}` pass this preflight and convert values one at a time. Indexed
assignment, writes through views or `coordinates`, offset/index-range
`copyto!` and `map!` also use ordinary element conversion. A complex value
then raises `InexactError`, possibly after earlier entries have been written.
For a vector iterate, `vec(x)` and `reshape(x, length(x))` are the iterate
itself and retain its direct-write checks. A matrix's `vec`, or a reshape to
a different shape, is a wrapper and follows elementwise conversion behavior.

`copy!` from a source of another size first uses Base's own mismatch behavior
(which depends on the array shape), except for iterate-to-iterate copies:
`SparseIterate` or `BlockIterate` sources of
different entry length raise `DimensionMismatch`. Equal-length copies
between different shapes or block layouts write values in linear order;
matching shapes/layouts of the same kind copy working-set state.

For failure-atomic conversion, stage an independent correctly typed result
and then use a supported state copy. A same-kind state copy preconverts
fallible stored values. A symmetric iterate's matrix import has a separate
[validation and conversion policy](@ref symmetric-copy-import),
and an orthogonal factor uses the [staged matrix-copy path](@ref sparse-orthogonal-copy).

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
types use generic/dense multiplication. See
[three-factor products](@ref three-factor-products) for the direct method's
range and summation guarantees.

`dot(c, X)` for a matrix `SparseIterate` `X` and a dense `c` that is not a `Diagonal` is the
generic `dot`, which visits all `mn` positions; it is not in the table. The value of a
[`LinearFunction`](smooth.md#Linear-function) at a matrix `SparseIterate` is
computed separately from the stored nonzero entries, with the strong-zero rule below.

Specialized products skip zero iterate coefficients, including stored zeros,
even against NaN/Inf in the other operand. This differs from an ordinary
dense IEEE `0*Inf`. A coordinate view has ordinary vector products and does
not inherit sparse work or strong-zero semantics. See
[array conventions](numerical_usage.md).

## [Three-factor products](@id three-factor-products)

For matching `Float32` or `Float64` operands, `A * X * B` has a direct
stored-entry method when `A` and `B` are `Symmetric` wrappers of strided
matrices and `X` is a matrix `SparseIterate` or a `SymmetricSparseIterate`.
The result is an ordinary dense matrix. Each entry is the sum of
`A[i,k] * X[k,l] * B[l,j]` over the mathematically nonzero entries of `X`.
For an `m × n` matrix iterate the cost is `O(nactive(X) * m * n)` operations
and `O(m * n)` memory. For a symmetric iterate both dimensions are `p`;
each stored off-diagonal coordinate contributes its two mirrored entries.

**Zeros belong to the sparse coefficient.** A stored zero and an absent
entry of `X` contribute no term, even if the corresponding entry of `A` or
`B` is non-finite. An iterate with no nonzeros produces the zero matrix
without reading numerical entries of either factor. A nonzero coefficient,
including `NaN` or an infinity, always contributes. Zeros in `A` and `B`
have ordinary IEEE behavior, so a zero factor multiplied by an infinite
coefficient produces `NaN`. Consequently, this operation need not equal
`A * Matrix(X) * B` on non-finite data.

**Range and rounding.** A term with three finite nonzero factors differs
from the correctly rounded exact product by at most one neighboring
floating-point number for a normal result (relative error below about
`2eps(T)`), or one spacing for a subnormal result. Correct rounding is not
promised. Within that margin of the overflow threshold or of half the smallest
subnormal, a term can fall on either side of the boundary: the neighbor of
`floatmax(T)` is `Inf`, and the neighbor of zero is the smallest subnormal.
Beyond that margin, whether a term overflows or underflows is determined by
its exact value, not by an intermediate partial product. For example,
`2^600 * 2^600 * 2^-600` and its underflow mirror are evaluated exactly
when the powers are `Float64` values.

**Summation.** Terms are added in the storage order of `X`, in precision `T`.
For a symmetric coordinate `(r,c)`, its `(r,c)` term precedes the `(c,r)`
term. Tiny terms beside large ones can round away. A non-finite ordinary sum
is recomputed in scaled units only when every factor of every contributing
term is finite. This recovers a representable result after an overflowing
partial sum followed by cancellation; a result beyond the range remains
`±Inf`. If any contributing factor is `NaN` or infinite, the ordinary IEEE
result stands. An exactly zero sum is `+0.0`.

Other matching element types or non-strided wrappers use restricted/dense
methods, and mixed types use generic/dense multiplication. These paths do
not inherit the direct method's range and rounding guarantees. The general
zero, precision and failure rules are in the [numerical contract](@ref numerics).
