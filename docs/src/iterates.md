# Sparse iterates

ProximalBase provides three iterate types for coordinate descent and active-set
procedures. We separate the choice of storage from the update formulas,
which are derived in [Coordinate descent](@ref).

| Iterate | Array represented | One coordinate |
|:--------|:------------------|:---------------|
| [`SparseIterate`](@ref) | A sparse vector or matrix | One entry |
| [`SymmetricSparseIterate`](@ref) | A sparse symmetric matrix | One lower-triangular entry and its symmetric counterpart |
| [`AtomIterate`](@ref) | A vector partitioned into groups | One group of entries |

The entries of an iterate are real. A complex element type, or a complex
operand of an iterate product, raises an `ArgumentError`; see
[Precision](@ref).

## Entries and coordinates

[`prox_entry`](@ref)`(g, v, i, γ)` computes the scalar proximal operator of
entry ``i`` without writing the iterate. In contrast,
[`prox_coordinate!`](@ref)`(x, g, k, γ)` reads coordinate ``k`` of the iterate,
applies its proximal operator and writes the result. The calling procedure
must supply the gradient step before applying this proximal operation.

The two calls index different spaces:
- `prox_entry` takes an entry ``i`` of the array, the index of the weights and
  bounds.
- `prox_coordinate!` takes a coordinate ``k`` of the iterate.

The two coincide for a `SparseIterate`. A coordinate is an entry of the lower
triangle for a `SymmetricSparseIterate`, and a group for an `AtomIterate`.

## SparseIterate

A [`SparseIterate`](@ref) is a vector or matrix that stores its nonzero
entries in a list. Three arrays hold them: the values, their indices, and for
every index the position of its value in the list (zero when it is not stored).
Reading or writing any entry is therefore ``O(1)``, and a product ``Xx`` costs
``O(n\cdot\mathrm{nnz})``. The stored entries are the active coordinates
([`activecoordinates`](@ref), [`nactive`](@ref)). Setting an entry to zero keeps
it stored, so that an active set does not change during a sweep. `dropzeros!`
removes the zeros between sweeps. `nnz(x)` counts stored zeros too.
`nonzeros(x)` gives a view of the stored values, paired with
`activecoordinates(x)` in storage order; the view is invalidated when the
stored set changes. `findnz(x)` returns copies of the stored indices and
values in sorted order, including stored zeros.

## SymmetricSparseIterate

A [`SymmetricSparseIterate`](@ref) is a sparse symmetric ``p \times p`` matrix
that stores its lower triangle. For indexing, length, iteration and
broadcasting it is an ordinary ``p \times p`` matrix, except that writing
``x_{rc}`` also writes ``x_{cr}``.

Read and write entries through the lower triangle, ``r \ge c``, for example
with `for c in 1:p, r in c:p`. The position `x[c, r]` is the same entry as
`x[r, c]`, so writing either position writes both. A loop over every position
writes each off-diagonal entry twice, and the second write wins.

Whole-value writes are checked. These include the constructor, `copyto!`,
`copy!`, broadcasting into ``x``, `map!`, `axpy!`, `axpby!`, `transpose!`,
`adjoint!`, `permutedims!` and `mul!`. Indexed array assignment of every
position in order (`x[:, :] = A`, `x[:] = v`), and broadcasting and `fill!`
into full, in-order views and reshapes (`x[:, :] .= A`, `vec(x) .= v`), use
the same checks. They cover a view whose parent is ``x``, a reshape whose
parent is ``x``, and a view of such a reshape. Use `:` or integer vectors
matching each axis in order for a full two-index assignment or a full matrix
view. A full single-index assignment accepts `:`, the integers `1:length(x)`
in order, or `CartesianIndices(x)` with its shape. `copyto!(x, Q)` from an orthogonal factor is checked too.

Matrix values must have the axes of ``x``, with equal entries ``(r, c)`` and
``(c, r)`` and `NaN` equal to `NaN`. These comparisons precede conversion to
the element type of ``x``. Linear full writes reshape their values in
column-major order for the check; `x[:] = v` needs ``p^2`` values. If the
axes differ or symmetry fails, the checked writers throw a
`DimensionMismatch` or an `ArgumentError` before writing anything.
`Symmetric(A, :L)` stores the lower triangle of a matrix ``A`` that is not
symmetric, and `Symmetric(A, :U)` its upper triangle. Each entry of the lower
triangle is written once. Sources that are symmetric by their type, such as
another `SymmetricSparseIterate`, a `Symmetric` or a `Diagonal`, skip the
symmetry check. A broadcast whose arguments do not show that its value is
symmetric is first evaluated at both positions of each off-diagonal entry.
A function with side effects can therefore run more often and give a value
that is not symmetric.

With two array operands, `mul!` into ``x`` checks the unscaled product for
exact symmetry in a buffer promoted from the destination and operand types,
applies scalars and conversion at the final write, and leaves the product
out when α = 0, as LinearAlgebra does. For `mul!` by a number, the array
operand must have the axes of ``x``. The seven-argument
`mul!(x, tA, tB, A, B, α, β)`, when available, uses the same checks with
flags `'N'`, `'T'` and `'C'` for unchanged, transposed and adjoint operands;
other flags throw before writing. `fillband!(x, v, l, u)` checks that `v` is
real and writes a symmetric band (`l == -u`) through the lower triangle
once. An empty band (`l > u`) returns ``x`` unchanged.

Indexed array assignment, broadcasting and `fill!` into part of ``x``, such
as a row, a column, a block or a partial mask, throw an `ArgumentError`
before writing. Logical-mask indexed assignment is rejected even for an
all-true mask. `copyto!` with offsets or index ranges (including transpose
flags), `copy_transpose!` and `copy_adjoint!` with index ranges, `copyto!` from a
tuple, a generator or a number, and the indexed `axpy!` also throw before
writing. Out-of-bounds indices still throw a `BoundsError`.

`reverse!`, `sort!`, `accumulate!`, `kron!` and `copytrito!` (when available)
throw an `ArgumentError` before writing. This also rejects `reverse(x)`
with or without `dims`, `cumsum!`, `cumprod!`, and allocating `cumsum`,
`cumprod` and `accumulate` with `dims`. `circshift!` and `circshift` of a
nonempty ``x`` throw too.
`triu!(x, k)` returns ``x`` unchanged when ``k \le 1-p``, and `tril!(x, k)`
when ``k \ge p-1``. Both return ``x`` for every `k` when ``p=0``. Otherwise
they throw an `ArgumentError` before writing.

`lmul!`, `rmul!`, `ldiv!` and `rdiv!` by a matrix, a factorization, an
orthogonal factor or a rotation throw an `ArgumentError` before writing.
So does `ldiv!(x, A, B)` for such an operator ``A``. Reductions into ``x``
(`sum!`, `prod!`, `maximum!`, `minimum!`, `count!`, `all!`, `any!`,
`extrema!`, `findmax!` and `findmin!` with ``x`` as either destination),
in-place factorizations (`lu!`, `qr!`, `cholesky!` and `bunchkaufman!`),
`get!` and `sortperm!` also throw before writing. Nonempty bands of `fillband!` that are not symmetric,
`read!`, `readbytes!` and `hex2bytes!` also throw before writing; these
reading and decoding calls do not consume their input. Apply these
operations to `Matrix(x)`, or compute into a dense matrix and `copyto!` a
symmetric result.

`fill!`, and `rmul!`, `lmul!`, `rdiv!` and `ldiv!` by a number or a
`UniformScaling`, keep ``x`` symmetric. `replace` and `replace!` visit each
entry of the lower triangle once, and their `count` counts those entries,
column by column. `triu`, `tril` and `sparse` return a `SparseMatrixCSC`.
`sort(x; dims)` and `accumulate(op, x)` without `dims` keep their dense
results. Allocating `lu`, `qr`, `cholesky` and `bunchkaufman` use dense copies.

Other functions that write an arbitrary array position by position are not
checked; apply them to `Matrix(x)`. Loops of scalar writes over every
position stay unchecked. Other writes through a view or reshape, such as
`copyto!` into a view and scalar writes through a view, stay unchecked too. Broadcasts and fills through other
nested wrappers, such as `vec(view(x, :, :))`, are also unchecked.
Two-argument `copyto!(x, itr)` from other non-array iterators is unchecked.
A Base function can write part of ``x`` before reaching a check. `rand!`
and `randn!` give a symmetric matrix but draw each off-diagonal entry twice.
`shuffle!` and `shuffle` give wrong values; use `Matrix(x)` instead.

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

That is `prox_entry(NormL1(λ), t₀ - (2/a) G[r, c], i, 2/a)`. On the diagonal
the factors of 2 disappear and the step is ``1/a``. So, as the
[`KronQuadratic`](@ref) example of the D-trace loss shows, off-diagonal
coordinates take the step ``2/a`` where the diagonal takes ``1/a``. Per-entry
parameters, such as the weights of `NormL1`, are indexed like the dense
``p \times p`` matrix and must be symmetric.

## AtomIterate

An [`AtomIterate`](@ref) is a vector whose coordinates are groups of entries,
partitioning ``1{:}p``. Its active groups play the role of the stored entries
of a `SparseIterate`, and [`atom`](@ref)`(x, k)` is a view of group ``k``.
Block coordinate descent for the group lasso takes, for group ``k``, a gradient
step of size ``1/L_k`` (with ``L_k = \|X_{G_k}\|_2^2`` for least squares). It
then applies the group's proximal step, `prox_coordinate!(x, g, k, 1/L_k)`,
which is block soft thresholding, and which activates the group when it ends
nonzero. `AtomIterate(p, g)` builds an iterate that shares the groups of a
group penalty `g`.

An iterate and a group penalty whose groups are equal but were built separately
also take the active-group paths. The first check compares the two layouts in
full; the iterate then remembers the penalty's group array, so later checks take
constant time. `copy` and `similar` keep what the iterate remembers, `copyto!`
keeps the destination's, and an iterate built from a restricted penalty starts
with nothing remembered.

Group arrays and the indices inside them must not be mutated after a penalty
or an iterate is built. This also applies to arrays retained by inner
constructors.

## Tracking an active set

The active coordinates form a stored working set, which can contain entries
or groups that have returned to zero. We keep them during a sweep so that
the set being traversed does not change.

| Operation | Meaning |
|:----------|:--------|
| [`ncoordinates`](@ref)`(x)` | Number of possible coordinates |
| [`nactive`](@ref)`(x)` | Number of stored active coordinates |
| [`activecoordinates`](@ref)`(x)` | Indices of the active coordinates |
| [`isactive`](@ref)`(x, k)` | Whether coordinate `k` is active |
| [`activate!`](@ref)`(x, k)` | Add a coordinate to the active set |
| `dropzeros!(x)` | Remove stored zero entries or zero groups |

Use [`kkt_violation`](@ref) to test excluded zero coordinates and
[`restrict`](@ref) to construct a supported restricted problem. The
[Coordinate descent](@ref) chapter explains the full optimality checks that
must accompany these storage operations.
