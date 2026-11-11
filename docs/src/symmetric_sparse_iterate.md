# SymmetricSparseIterate: independent coordinates and matrix views

A [`SymmetricSparseIterate`](@ref) owns a real symmetric ``p\times p`` matrix
with one stored value per independent lower-triangular coordinate. It supports
constant-time pair access, membership and activation, and packed active
traversal. Capacity is reserved for all ``q=p(p+1)/2`` coordinates, so memory
is O(q) even when the working set is small.

**It is not an `AbstractArray` or `AbstractMatrix`.** It has `size(x)==(p,p)` and
`eltype(x)`, but no array length, axes, iteration or broadcasting. This prevents
an algorithm from accidentally treating a lower-triangle coordinate ID as a
full-matrix linear position.

## Three index spaces

| Expression | Meaning |
|:--|:--|
| `x[r,c]` | Matrix pair; either triangle names the same variable |
| `coordinates(x)[k]` | Independent lower-triangle variable k, in column order |
| `matrixview(x)[i]` | Ordinary full-matrix linear index i |
| `activecoordinates(x)[a]` | Coordinate ID in active slot a |
| `x[k]` | Rejected with ArgumentError because the index space is ambiguous |

For ``p=3``, coordinates are `(1,1),(2,1),(3,1),(2,2),(3,2),(3,3)`.
`tril_index(p,r,c)` requires a valid lower-triangle pair and returns its ID;
`tril_cartesian(p,k)` converts a valid ID back. These arithmetic helpers do
not provide a general bounds-validation layer; validate external indices first.

```@example symmetric_indices
using ProximalBase, LinearAlgebra
x = SymmetricSparseIterate(3)
x[2,2] = 7
coordinates(x)[3] = 2
@assert x[3,1] == x[1,3] == 2
@assert tril_index(3,2,2) == 4 && tril_cartesian(3,3) == (3,1)
@assert coordinates(x)[4] == 7 && matrixview(x)[4] == 0
@assert collect(activecoordinates(x)) == [4,3]
Matrix(x)
```

Pair bounds are checked before narrowing other integer types, with a
`BoundsError` for an invalid pair. `ncoordinates(x)` is q, `nactive(x)` counts
stored independent variables, and `isactive/activate!` take independent IDs.
The direct constructor `SymmetricSparseIterate([T,]p)` accepts nonnegative
Int dimensions, with `Float64` as the default element type.

## The live views and dense copy

`coordinates(x)` is a writable length-q vector; every write addresses exactly
one variable. An inactive coordinate reads zero. A converted nonzero write
(including NaN/Inf) activates it; zero leaves an inactive coordinate absent
and an active one active. `fill!(coordinates(x),0)` touches active values
only and preserves membership; following it with `dropzeros!(x)` empties the
working set in O(s), where `s = nactive(x)`.

`matrixview(x)` is a read-only live ``p\times p`` `AbstractMatrix`. Both triangles
read the same variable. Writes through the view or derived views/wrappers
are refused; use the owner or its coordinate view to write. `Matrix(x)` is
an independent dense snapshot. Allocating broadcasts and `similar` on the
matrix view produce ordinary writable dense arrays.

```@example symmetric_indices
A = matrixview(x)
x[1,1] = 1
@assert A[1,1] == 1
B = Matrix(x)
B[1,1] = 10
@assert x[1,1] == 1
activate!(x,2)
@assert nactive(x) == 4
snapshot = copy(x)
fill!(coordinates(x),0)
@assert nactive(x) == 4
dropzeros!(x)
@assert nactive(x) == 0 && nactive(snapshot) == 4
Matrix(snapshot)
```

`activecoordinates` is read-only and `activevalues` is writable. They pair by
slot, keep their old extent and must be reacquired after activation,
compaction or whole-state replacement. Coordinate/matrix views instead keep
a stable index mapping for the owner's lifetime.

## Copying state versus importing a matrix

`copy(x)` preserves active zeros and slot order independently. `similar(x)`
and `similar(x,T)` allocate an empty owner of the same dimension.
`copyto!(dest,src)` between owners replaces values and active state;
self-copy does nothing. Fallible conversion of stored values is staged
before state writes; dimension mismatch leaves the destination unchanged.

`SymmetricSparseIterate(A)` and `copyto!(x,A)` import a matrix. A source must
be real, square/of the destination shape, one-based and **exactly symmetric**
before conversion. Paired NaNs agree; signed zeros agree; integer comparisons
are exact. The import replaces activity with coordinates whose converted
value is nonzero, in increasing coordinate order. It does not preserve
explicit active zeros from the old state. `copyto!(x,matrixview(x))` rejects
shared memory with `ArgumentError`; `copyto!(x,Matrix(x))` is independent.
Shape/type/symmetry/alias checks precede state writes, but an element conversion
can fail after the old state has been cleared. Stage an independent typed
matrix first when conversion failure must be atomic.

Use `Symmetric(A,:L)` to deliberately select the lower triangle of a matrix
whose stored triangles differ. Import does not average them. This differs
from a mathematical **prox into an owner**, which reads a nonsymmetric
argument through its symmetric part.

## The multiplicity of an off-diagonal variable

For the matrix represented by coordinates ``v_k``,

```math
\|X\|_F^2=\sum_{k\text{ diagonal}}v_k^2+
2\sum_{k\text{ off diagonal}}v_k^2.
```

The independent vector's Euclidean norm counts each value once. `norm(x)`
and `norm(matrixview(x))` use Frobenius geometry; `norm(x,1)` sums absolute
matrix entries with multiplicity two, and `norm(x,Inf)` is the largest absolute
entry, not an operator norm.

```@example symmetric_multiplicity
using ProximalBase, LinearAlgebra
x = SymmetricSparseIterate(2)
x[2,1] = 3
@assert norm(coordinates(x)) == 3
@assert norm(x)^2 ≈ 18 && NormL1()(x) == 6
@assert norm(x,1) == 6 && norm(x,Inf) == 3
@assert Matrix(prox(NormL1(1.0),x,1.0)) == [0 2;2 0]
Matrix(x)
```

For a symmetric entry penalty and Frobenius prox, the two copies of an
off-diagonal term cancel the two copies of its squared distance, so the scalar
threshold is unchanged. If instead solving a Euclidean coordinate-vector
problem with a loss differentiated in coordinate space, both the coordinate
gradient and curvature carry the corresponding multiplicity. Do not mix the
two geometries. Owner KKT screens use the symmetric **matrix** gradient,
without doubling the reported entry screen.

## Supported numerical operations

| Operation | Meaning |
|:--|:--|
| `x*v`, `x*B`, `mul!(out,x,v/B[,α,β])` | Real right operand; sparse stored traversal; independent output required |
| `mul!(out,matrixview(x),v[,α,β])` | O(p+s) vector product, with `s = nactive(x)` |
| `dot(a,x,b)` | Bilinear form; one-based real vectors; off-diagonal pair contributes twice |
| `dot(A,x)` / `dot(x,A)` | Frobenius inner product for strided real matrix A |
| `transpose(x)` / `adjoint(x)` | The same owner |
| `norm_diff(x,y[,p])` | Full-matrix difference norm, p=2 or Inf; owner/array combinations |
| `sparse(matrixview(x))`, `tril/triu(matrixview(x),k)` | SparseMatrixCSC containing mathematical nonzero entries |
| `A*x*B` | Selected Symmetric-factor kernels; see [product domains](iterates.md#Supported-products) |

Zero owner coefficients are skipped even against a non-finite right operand;
stored zero and absent zero have the same product contribution. Other generic
operations on the matrix view can visit all p² entries. Generic coordinate
vector products have ordinary IEEE arithmetic.

`SqrNormL2`, `LinearFunction` and `SeparableFunction` have explicit owner value/prox
methods; supported `Scaled` leaves forward them. `SqrNormL2` and `LinearFunction`
also have owner gradients, allocating a dense symmetric matrix. `ZeroFunction`
has owner value/prox, but no owner gradient. `LeastSquaresLoss` and
`QuadraticFunction` have no owner interface. A true capability trait is not
sufficient to accept an owner.

A symmetric-owner prox destination accepts a scalar step only. Array steps
throw `ArgumentError` before writes. Full-entry parameter arrays must satisfy
the [exact symmetric-parameter contract](symmetric_parameters.md).

## Deliberately unsupported owner operations

There is no `length(x)`, `axes(x)`, `iterate(x)`, `eachindex(x)`, `vec(x)`,
`reshape(x,...)`, `view(x,...)`, broadcast, `fill!(x,v)`, `copy!(x,A)` or
`mul!(x,A,B)` owner-array interface. Use `coordinates` for variables or
`matrixview`/`Matrix` for ordinary matrix computations, then import an exactly
symmetric result if necessary. `axes` and one-index access explicitly throw
`ArgumentError`; many other absent calls throw `MethodError`.

Two owners compare by represented matrix values under `==` and `isequal`,
independently of active order. NaNs and signed zeros follow those operators'
ordinary distinctions; equal owners have equal hashes. There is no owner-array
value comparison or `isapprox` owner method: compare `Matrix(x)` or the view.

See [the complete example](examples/symmetric_storage.md) and
[developer invariants](developer/symmetric_storage.md).
