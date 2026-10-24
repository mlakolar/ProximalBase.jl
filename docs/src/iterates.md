# Sparse iterates

ProximalBase provides three iterate types for coordinate descent and active-set
procedures. We separate the choice of storage from the update formulas,
which are derived in [Coordinate descent](@ref).

| Iterate | Array represented | One coordinate |
|:--------|:------------------|:---------------|
| [`SparseIterate`](@ref) | A sparse vector or matrix | One entry |
| [`SymmetricSparseIterate`](@ref) | A sparse symmetric matrix | One lower-triangular entry and its symmetric counterpart |
| [`AtomIterate`](@ref) | A vector partitioned into groups | One group of entries |

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
removes the zeros between sweeps.

## SymmetricSparseIterate

A [`SymmetricSparseIterate`](@ref) is a sparse symmetric ``p \times p`` matrix
that stores its lower triangle. For indexing, length, iteration and
broadcasting it is an ordinary ``p \times p`` matrix, except that writing
``x_{rc}`` also writes ``x_{cr}``. Its coordinates are the
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
