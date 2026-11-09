# SymmetricSparseIterate: independent coordinates and matrix views

A `SymmetricSparseIterate{T}` represents a real, symmetric `p × p` matrix using **one stored value per independent lower-triangular coordinate**, with O(1) entry lookup and active-coordinate operations. Its main purpose is to let algorithms update a symmetric variable exactly once while retaining a sparse *working set*.

**It is deliberately not an `AbstractArray`.** This distinguishes it from [`SparseIterate`](sparse_iterate.md), which remains a Julia vector or matrix with ordinary linear indexing. The full normative specification is the [symmetric owner 0.6 contract](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/symmetric-sparse-iterate-0.6.md); the companion [acceptance plan](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/symmetric-sparse-iterate-0.6-acceptance.md) specifies its tests.

## Three different index spaces

For a symmetric matrix of order `p`, there are `q = p(p+1)/2` independent coordinates.

| Expression | Index space | Meaning |
|:--|:--|:--|
| `X[i,j]` | Matrix pair | Symmetric value at row `i`, column `j`; `X[i,j] == X[j,i]` |
| `coordinates(X)[k]` | Lower-triangle coordinate ID | Exactly one independent variable; `1 ≤ k ≤ q` |
| `matrixview(X)[k]` | Full-matrix linear index | The position `k` in the ordinary `p × p` column-major matrix, including mirrored entries |
| `activecoordinates(X)[a]` | Active storage slot | The coordinate ID occupying slot `a` in the packed working set |
| `X[k]` | **Unsupported** | Ambiguous between the previous two index spaces; throws `ArgumentError` |

The lower-triangle coordinate IDs run **down each lower-triangle column**. For `p = 3`:

| Coordinate ID `k` | Matrix position | Mirrored position |
|:--|:--|:--|
| 1 | (1,1) | (1,1) |
| 2 | (2,1) | (1,2) |
| 3 | (3,1) | (1,3) |
| 4 | (2,2) | (2,2) |
| 5 | (3,2) | (2,3) |
| 6 | (3,3) | (3,3) |

The helpers `tril_index(p,r,c)` and `tril_cartesian(p,k)` convert between these spaces when `r ≥ c`.

```julia
X = SymmetricSparseIterate(3)
c = coordinates(X)

c[2] = 5.0
X[2,1]                  # 5.0
X[1,2]                  # 5.0, same independent coordinate
X[2]                    # ArgumentError: use coordinates(X)[k] or matrixview(X)[k]

c[4] = 7.0             # diagonal at (2,2)
matrixview(X)[4]        # 5.0: full-matrix linear index = (1,2), NOT coordinate 4
```

## Public storage interface and costs

Let `s = nactive(X)`, the number of active independent variables, **including stored zeros**. The owner reserves `q` slots and a direct coordinate-to-slot map at construction. It is optimized for sparse *work*, not memory compression.

| Operation | Available method | Cost |
|:--|:--|:--|
| Construct empty owner | `SymmetricSparseIterate(p)`, `SymmetricSparseIterate(T,p)` | O(q) time and storage |
| Import symmetric matrix | `SymmetricSparseIterate(M)` | O(p²) validation/reading; O(q) stored capacity |
| Matrix shape/type | `size(X)`, `size(X,d)`, `eltype(X)` | O(1) |
| Independent-coordinate count | `ncoordinates(X)` | O(1), equals `q` |
| Matrix-pair read/write | `X[i,j]`, `X[i,j] = v` | O(1) |
| Coordinate-vector handle | `coordinates(X)` | O(1) construction; O(1) scalar read/write |
| Read-only full-matrix handle | `matrixview(X)` | O(1) construction |
| Working-set size and membership | `nactive(X)`, `isactive(X,k)` | O(1) |
| Explicit activation | `activate!(X,k)` | O(1); creates an active zero when absent |
| Active IDs | `activecoordinates(X)` | O(1) handle creation; O(s) traversal |
| Active values | `activevalues(X)` | O(1) handle creation; O(s) traversal |
| Remove stored zeros | `dropzeros!(X)` | O(s), may reorder active slots |
| Copy state | `copy(X)`, `copyto!(X,Y::SymmetricSparseIterate)` | O(q) worst case |
| Create empty same-size owner | `similar(X)`, `similar(X,T)` | O(q) |
| Readable dense snapshot | `Matrix(X)`, `Array(X)` | O(p² + s) |
| Checked whole-matrix import | `copyto!(X,M::AbstractMatrix)` | O(p²) validation/read, state replacement |

These methods do **not** imply `length(X)`, `axes(X)`, `eachindex(X)`, iteration, `X .= rhs`, generic `sum(X)` or `map(X)`. Those operations belong on `coordinates(X)` when working with independent variables, or `matrixview(X)` when reading all matrix positions.

There is **no `nnz(X)`, `nonzeros(X)`, `findnz(X)` or `issparse(X)` specialization** for the owner. Its `dropzeros!` method remains an essential working-set operation, not a promise of the `SparseArrays.AbstractSparseArray` API.

## Working-set semantics

An inactive independent coordinate is zero. A coordinate can be active while its value is zero. A nonzero scalar write activates a missing coordinate; setting an active coordinate to zero does not deactivate it. `activate!(X,k)` is the explicit way to retain a zero in the working set.

```julia
X = SymmetricSparseIterate(3)
activate!(X, 5)
nactive(X)                   # 1
coordinates(X)[5]           # 0.0

coordinates(X)[5] = 3.0
coordinates(X)[5] = 0.0
nactive(X)                   # 1: still active

dropzeros!(X)
nactive(X)                   # 0
```

`activecoordinates(X)` is a **read-only handle** of active coordinate IDs in storage-slot order, not a sorted list. `activevalues(X)` is a writable handle of the corresponding values. Their extents are fixed when obtained. Reacquire both after activation, compaction or state replacement; the old slot mapping may no longer be valid.

A safe sweep that does not change membership may iterate over `activecoordinates(X)`, update those coordinate values, and then call `dropzeros!` after the loop. A full scan of `coordinates(X)` visits all `q` variables, whether stored or not.

Clearing the working set is:

```julia
fill!(coordinates(X), 0)   # O(s): preserve active IDs while zeroing values
dropzeros!(X)              # O(s): remove them
```

## The two views have different semantics

`coordinates(X)` returns a live, writable `SymmetricCoordinates <: AbstractVector`. It is the vector of `q` independent variables, with ordinary vector operations. It does **not** compute the full matrix Frobenius norm or count off-diagonal entries twice.

`matrixview(X)` returns a live, **read-only** `SymmetricMatrixView <: AbstractMatrix`. It has `p²` logical entries, including both triangles. Reading sees later writes to `X`. Writing through it or derived standard array wrappers is unsupported; use `X[i,j]` or `coordinates(X)[k]` to mutate the owner. `Matrix(X)` creates an independent dense snapshot.

```julia
c = coordinates(X)
A = matrixview(X)

size(c)            # (6,)
size(A)            # (3, 3)
c[2] = 4.0
A[1,2]             # 4.0
A[1,2] = 9.0       # rejected: view is read-only
```

A full-matrix inner product or Frobenius norm counts off-diagonal entries **twice**; an ordinary vector norm on `coordinates(X)` counts every independent coordinate once. For mathematical kernels, use the appropriate matrix/owner operation rather than treating coordinate-space arithmetic as Frobenius arithmetic.

For the wrapper types' exact Julia method definitions and the private `Base.dataids`/`Base.unaliascopy` hooks, see [Coordinate views and aliasing](developer/index.md).

## Copying, imports, and exceptional cases

`copy(X)` preserves the independent values, all active IDs, their slot order and explicit active zeros. `similar(X)` allocates an empty owner. A same-kind `copyto!(X,Y)` **replaces X's state** with Y's, including active zeros.

Unlike ordinary elementwise array copying, `copyto!(X,M::AbstractMatrix)` is a **checked import and replacement**: it requires the correct size, real elements and exact symmetry and makes the working set the nonzeros of the converted matrix. An input sharing memory with `X` (such as `matrixview(X)`) is rejected before writing. `copyto!(X,X)` is a no-op.

**There is no automatic symmetrization by ordinary matrix import.** To work with a specific triangle, use a deliberate `Symmetric(M,:L)` or another explicit model representation.

## Performance boundary

Scalar access and activation take O(1) after O(q) construction. Active-set sweeps take O(s). The specialized `mul!` owner/matrix-view product with a dense vector can perform O(p+s) work in its supported domain. Generic operations on `matrixview(X)` may inspect O(p²) entries.

These guarantees concern the *data structure*. KKT screening, function values, gradients, proximal maps, validation of external parameters and matrix factorizations belong to their own function or solver contracts; they are not hidden in the coordinate handles.

For contributor-facing invariants, full acceptance tests and the relation to `SparseIterate`, follow the [developer documentation](developer/index.md).
