# Coordinate descent and iterates

## The coordinate step

Coordinate descent minimizes
``F(x) = f(x) + \sum_i g_i(x_i)``, with ``f`` smooth and ``g`` separable, one
coordinate at a time. Fix every coordinate but ``x_i``. Let ``a_i`` bound the
curvature of ``f`` along ``x_i``; it is exact for a quadratic, and
``a_i = \|X_i\|^2`` for least squares. Minimizing the quadratic model of ``f``
plus ``g_i`` gives

```math
x_i^+ = \operatorname*{argmin}_t \; \nabla_i f(x)\,(t - x_i) + \frac{a_i}{2}(t - x_i)^2 + g_i(t)
= \operatorname{prox}_{g_i/a_i}\Big(x_i - \frac{\nabla_i f(x)}{a_i}\Big).
```

For least squares, with the residual ``r = y - Xx`` kept up to date, the
gradient is ``\nabla_i f(x) = -X_i^\top r``. The step is then
``x_i^+ = \operatorname{prox}_{g_i/a_i}(x_i + X_i^\top r/a_i)``, followed by
``r \leftarrow r - X_i(x_i^+ - x_i)``. For a convex ``F`` with separable
``g``, cyclic coordinate descent converges to a minimizer (Tseng, 2001).

[`prox_entry`](@ref)`(g, v, i, γ)` is ``\operatorname{prox}_{\gamma g_i}(v)``.
It computes the new value without writing the iterate. So a solver writes only
the coordinates that change, and a coordinate that stays at zero is never
stored. [`prox_coordinate!`](@ref)`(x, g, k, γ)` applies the proximal step to
coordinate ``k`` of an iterate in place. It reads the current value, maps the
coordinate to its entry, writes the result and keeps the active set up to date.

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

## Active sets

Most coordinates of a sparse solution are zero, so active-set and working-set
solvers sweep only the coordinates that may be nonzero. Two oracles support
them.

- **[`kkt_violation`](@ref)`(g, ∇f(x), i)`** measures how much a coordinate at
  zero violates optimality, given the gradient of the smooth part. It is
  ``|\nabla_i f| - \lambda w_i`` for the ℓ1 norm, and
  ``\|\nabla_{G_k} f\|_2 - \lambda w_k`` for a group, so a positive value means
  the coordinate should join the active set.
- **[`restrict`](@ref)`(f, S)`** is the function of ``x_S`` that ``f`` becomes on
  the vectors supported on ``S``. For a loss it keeps the columns ``S`` of
  ``X``.

A working-set solver alternates between solving the problem restricted to
``S`` and adding the worst violators. The working-set example solves a lasso
with ``p = 5000`` that way, about 40 times faster than proximal gradient descent on
all the coordinates.
