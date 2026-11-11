# SymmetricSparseIterate: independent coordinates and matrix views

A `SymmetricSparseIterate{T}` represents a real, symmetric `p × p` matrix using **one stored value per independent lower-triangular coordinate**, with O(1) entry lookup and active-coordinate operations. Its main purpose is to let algorithms update a symmetric variable exactly once while retaining a sparse *working set*.

**It is deliberately not an `AbstractArray`.** This distinguishes it from [`SparseIterate`](sparse_iterate.md), which remains a Julia vector or matrix with ordinary linear indexing. For the normative 0.6 design and acceptance requirements, see the [symmetric-owner contract](developer/contracts/symmetric-sparse-iterate-0.6.md) and its [acceptance plan](developer/contracts/symmetric-sparse-iterate-0.6-acceptance.md) in Developer Documentation.

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

## Index-space example

```@example owner_index_spaces
using ProximalBase

p = 3
x = SymmetricSparseIterate(p)
x[2, 2] = 7.0                                        # the pair (2, 2) is variable 4
x[3, 1] = 5.0                                        # the pairs (3, 1) and (1, 3) are variable 3
@assert x[1, 3] == 5.0 && coordinates(x)[3] == 5.0
@assert tril_index(p, 2, 2) == 4 && coordinates(x)[4] == 7.0
@assert collect(activecoordinates(x)) == [4, 3]      # variable ids, in the order of activation
@assert collect(activevalues(x)) == [7.0, 5.0]       # their values, slot by slot
@assert matrixview(x)[4] == 0.0                      # linear index 4 is the position (1, 2), a different variable
@assert matrixview(x)[5] == 7.0                      # and linear index 5 is (2, 2)
try
    x[3]
    error("expected an ArgumentError")
catch e
    @assert e isa ArgumentError
    showerror(stdout, e)
    println()
end
```

## Public storage interface and costs

Let `s = nactive(X)`, the number of active independent variables, **including stored zeros**. The iterate reserves capacity proportional to `q`, and it is optimized for sparse *work*, not memory compression.

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

For the wrapper types' exact Julia method definitions and the private `Base.dataids`/`Base.unaliascopy` hooks, see [Coordinate views and aliasing](developer/views_aliasing.md).

## Copying, imports, and exceptional cases

`copy(X)` preserves the independent values, all active IDs, their slot order and explicit active zeros. `similar(X)` allocates an empty owner. A same-kind `copyto!(X,Y)` **replaces X's state** with Y's, including active zeros.

Unlike ordinary elementwise array copying, `copyto!(X,M::AbstractMatrix)` is a **checked import and replacement**: it requires the correct size, real elements and exact symmetry and makes the working set the nonzeros of the converted matrix. An input sharing memory with `X` (such as `matrixview(X)`) is rejected before writing. `copyto!(X,X)` is a no-op.

**There is no automatic symmetrization by ordinary matrix import.** To work with a specific triangle, use a deliberate `Symmetric(M,:L)` or another explicit model representation.

## Checked import and state-copy example

```@example owner_copies
using ProximalBase, LinearAlgebra

x = SymmetricSparseIterate(3)
x[2, 1] = 1.0
activate!(x, 6)                                      # an explicit zero at (3, 3)
y = copy(x)
@assert collect(activecoordinates(y)) == [2, 6] && isactive(y, 6)

w = SymmetricSparseIterate(3)
copyto!(w, Matrix(x))                                # an import: the explicit zero is not carried
@assert collect(activecoordinates(w)) == [2]

A = [1.0 2.0 0.0; 2.5 1.0 0.0; 0.0 0.0 1.0]          # not symmetric
try
    copyto!(w, A)
    error("expected an ArgumentError")
catch e
    @assert e isa ArgumentError
end
@assert collect(activecoordinates(w)) == [2]         # nothing was written
copyto!(w, Symmetric(A, :L))                         # the lower triangle, chosen by the caller
@assert w[1, 2] == 2.5

try
    copyto!(w, matrixview(w))                        # the source shares memory with w
    error("expected an ArgumentError")
catch e
    @assert e isa ArgumentError
end
prox!(w, NormL1(0.1), matrixview(w), 1.0)            # a proximal map from its own view is fine
@assert w[2, 1] ≈ 2.4
```

## Performance boundary

Scalar access and activation take O(1) after O(q) construction. Active-set sweeps take O(s). The specialized `mul!` owner/matrix-view product with a dense vector can perform O(p+s) work in its supported domain. Generic operations on `matrixview(X)` may inspect O(p²) entries.

These guarantees concern the *data structure*. KKT screening, function values, gradients, proximal maps, validation of external parameters and matrix factorizations belong to their own function or solver contracts; they are not hidden in the coordinate handles.

For contributor-facing invariants, full acceptance tests and the relation to `SparseIterate`, follow the [developer documentation](developer/views_aliasing.md).

## Equality and snapshots

Two owners are `==` and `isequal` when their matrices are: `isequal(x, y)` is
`isequal(Matrix(x), Matrix(y))`, whatever the active sets and their order. An active
`-0.0` is therefore different under `isequal` from an absent coordinate, which is
`+0.0`, and is equal to it under `==`; two `NaN`s are `isequal` and not `==`. Equal
matrices have equal hashes. There is no comparison of an owner with an array, and
there is no `isapprox` for owners: compare `Matrix(x)` or `matrixview(x)`, which are
ordinary matrices.

```@example owner_equality
using ProximalBase

x = SymmetricSparseIterate(3)
x[2, 1] = 1.0
activate!(x, 6)                                      # an active zero at (3, 3)
y = SymmetricSparseIterate(3)
y[2, 1] = 1.0
@assert isequal(x, y) && x == y && hash(x) == hash(y) # an active zero is an absent coordinate
y[2, 1] = 1.0 + 1e-9
@assert !isequal(x, y)
@assert isapprox(matrixview(x), matrixview(y); atol = 1e-6)   # an approximation compares the matrices
try
    isapprox(x, y)
    error("expected a MethodError")
catch e
    @assert e isa MethodError
end
```

## The multiplicity of an off-diagonal variable

A variable ``(r, c)`` with ``r \ne c`` stands for two entries of the matrix.
Everything that is a function of the matrix counts both: `norm(x)` is the Frobenius norm
of `Matrix(x)`, `dot(A, x)` and `dot(a, x, b)` are the inner products of the matrix, a
penalty such as `NormL1` takes the sum over both triangles, `IndBallL0` counts the entries of the
matrix, so a nonzero off-diagonal pair counts two, the gradient and the curvature
along a coordinate are those of the direction ``E_{rc} + E_{cr}``, and a proximal step is
the Frobenius one. The values of the variables themselves, `x[r, c]`, are not rescaled.
`kkt_violation` screens a coordinate at zero: a positive value says the coordinate should
enter the working set, and a value that is not positive does not certify that a returned
solution is stationary.

```@example owner_multiplicity
using ProximalBase, LinearAlgebra

x = SymmetricSparseIterate(3)
x[2, 1] = 3.0
x[3, 3] = 4.0
@assert norm(x) ≈ sqrt(2 * 3.0^2 + 4.0^2)           # (2, 1) counts twice
@assert norm(x) ≈ norm(Matrix(x))
@assert NormL1(1.0)(x) == 2 * 3.0 + 4.0
@assert dot(ones(3, 3), x) == 2 * 3.0 + 4.0
@assert x[1, 2] == 3.0                               # the variable itself is not rescaled
```

An owner as the output of a proximal map takes a *scalar* Frobenius step ``\gamma``. An array
of steps, or one step per group, is not supported on an owner: it throws an `ArgumentError`
before anything is written, and the dense and other iterate outputs are unchanged. Use a
scalar step, or `Matrix(x)` as the output.

## Array operations and the calls to use instead

An array function is not defined for the owner. Each of the following says what to
call for the question that the array function would have answered.

| To get | Use |
|:-------|:----|
| a variable by its id | `coordinates(x)[k]` |
| a matrix read or iteration: `x[:, j]`, `sum`, `map`, `view`, broadcasting | the same call on `matrixview(x)` |
| ``p`` | `size(x, 1)` |
| ``q`` | `ncoordinates(x)`, or `length(coordinates(x))` |
| the size of the working set | `nactive(x)` |
| the number of nonzeros of the matrix | `count(!iszero, Matrix(x))`, or `nnz(sparse(matrixview(x)))` |
| to clear the owner | `fill!(coordinates(x), 0); dropzeros!(x)` |
| the stored values | `activevalues(x)` |
| writable dense storage | `Matrix(x)` |
| the matrix as a sparse matrix, or a triangle of it | `sparse(matrixview(x))`, `triu(matrixview(x), k)`, `tril(matrixview(x), k)` |
| the product of two matrices stored in the owner | compute `C = A * B` into a dense matrix, and `copyto!(x, C)` if a symmetric import is meant |

[Compatibility and migration](@ref) lists what each of these was in 0.5.

Its coordinates are the
``p(p+1)/2`` entries ``(r, c)``, ``r \ge c``, of the lower triangle, numbered
column by column:

```math
k = \texttt{tril\_index}(p, r, c) = \frac{(c - 1)(2p - c + 2)}{2} + (r - c + 1).
```

[`tril_cartesian`](@ref) inverts this by solving a quadratic equation. Its
floating square root is exact for ``p < 2^{24}``, far beyond any matrix that
fits in memory, and it switches to an integer square root beyond that.
[`coordinates`](@ref)`(x)` is a vector view of the coordinates.

**The coordinate step on a symmetric matrix.** A coordinate ``(r, c)`` with
``r \ne c`` moves two entries of ``\Theta`` together. Take a penalty that
counts both triangles, ``\lambda\sum_{i,j}|\Theta_{ij}|``, as `NormL1` does on
the matrix. Let ``G`` be the gradient of ``f`` over symmetric matrices, and
``a`` the curvature along the direction ``E_{rc} + E_{cr}``. Along the
coordinate,

```math
\varphi(t) = 2G_{rc}(t - t_0) + \frac{a}{2}(t - t_0)^2 + 2\lambda|t| + \text{const},
\qquad
t^+ = S_{2\lambda/a}\Big(t_0 - \frac{2G_{rc}}{a}\Big).
```

That is `prox_block(NormL1(λ), t₀ - (2/a) G[r, c], i, 2/a)`. On the diagonal
the factors of 2 disappear and the step is ``1/a``. So, as the
[`KronQuadratic`](@ref) example of the D-trace loss shows, off-diagonal
coordinates take the step ``2/a`` where the diagonal takes ``1/a``.

## Sparse matrix conversion

`matrixview(X)` can be supplied to standard matrix readers. In the supported
sparse conversion and triangle-extraction paths, it can also be converted to
a `SparseMatrixCSC` without scanning all `p²` matrix positions. A stored zero
is not included in the result. Unlike the live view, a converted matrix is
an independent object.
```@example owner_view
using ProximalBase, SparseArrays, LinearAlgebra

x = SymmetricSparseIterate(4)
x[3, 1] = 2.0
x[4, 4] = 5.0
activate!(x, 2)                                      # an active zero at (2, 1)
A = matrixview(x)
S = sparse(A)
@assert S isa SparseMatrixCSC && S == sparse(Matrix(x))
@assert nnz(S) == 3                                  # (3, 1), (1, 3) and (4, 4); the active zero is no entry
@assert triu(A) == triu(Matrix(x)) && tril(A, -1) == tril(Matrix(x), -1)
@assert nnz(triu(A, 1)) == 1 && nnz(tril(A, -1)) == 1
try
    A[1, 1] = 1.0
    error("expected the write to be refused")
catch e
    @assert occursin(r"setindex!|read-only|not defined", sprint(showerror, e))
end
@assert x[1, 1] == 0.0

ids = activecoordinates(x)
@assert length(ids) == nactive(x) == 3
x[4, 3] = 1.0                                        # a new active variable
@assert length(ids) == 3 && length(activecoordinates(x)) == 4   # the old handle did not follow
fill!(coordinates(x), 0)                             # writes the active variables only
@assert nactive(x) == 4 && all(iszero, Matrix(x))
dropzeros!(x)
@assert nactive(x) == 0
```

For the distinction between the public read-only view and private mutable integration wrappers, see [symmetric owner internals](developer/symmetric_storage.md).
