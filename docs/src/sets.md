# Sets

The indicator function ``\iota_C`` of a set ``C`` is ``0`` on ``C`` and
``+\infty`` outside. Its proximal operator is the Euclidean projection
``P_C(v) = \operatorname{argmin}_{z \in C}\|z - v\|``, for every step, so
projected gradient descent is proximal gradient descent with an indicator.
[`value_and_prox!`](@ref) returns ``0``. The projections return a point within
a relative ``\sqrt{\varepsilon}`` of the set, so the result is feasible up to
rounding.

| Function | Set | Projection |
|:---------|:----|:-----------|
| [`IndBox`](@ref)`(l, u)` | ``l \le x \le u`` | ``\operatorname{clamp}(v, l, u)`` |
| [`IndNonnegative`](@ref)`()` | ``x \ge 0`` | ``\max(v, 0)`` |
| [`IndBallLinf`](@ref)`(r)` | ``\lVert x\rVert_\infty \le r`` | ``\operatorname{clamp}(v, -r, r)`` |
| [`IndBallL2`](@ref)`(r)` | ``\lVert x\rVert_2 \le r`` | ``\min(1, r/\lVert v\rVert_2)\,v`` |
| [`IndSphereL2`](@ref)`(r)` | ``\lVert x\rVert_2 = r``, nonconvex | ``r\,v/\lVert v\rVert_2`` |
| [`IndBallL1`](@ref)`(r)` | ``\lVert x\rVert_1 \le r`` | ``\operatorname{sign}(v)\odot P_{\Delta_r}(\lvert v\rvert)``, by Condat's algorithm |
| [`IndSimplex`](@ref)`(r)` | ``x \ge 0,\ \sum_i x_i = r`` | ``\max(v - \tau, 0)`` |
| [`IndBallL0`](@ref)`(s)` | at most ``s`` nonzero entries | keep the ``s`` largest in magnitude |
| [`IndHyperplane`](@ref)`(a, b)` | ``\langle a, x\rangle = b`` | ``v - \frac{\langle a, v\rangle - b}{\lVert a\rVert^2}\,a`` |
| [`IndHalfspace`](@ref)`(a, b)` | ``\langle a, x\rangle \le b`` | the same when ``\langle a, v\rangle > b`` |
| [`IndPSD`](@ref)`()` | ``X \succeq 0`` | ``U\max(\Lambda, 0)U^\top`` |
| [`IndPSD`](@ref)`(rank = r)` | ``X \succeq 0``, rank at most ``r`` | the ``r`` largest positive eigenpairs |
| [`IndBallRank`](@ref)`(r)` | rank at most ``r`` | the truncated SVD |

## Separable sets

The box, the nonnegative orthant and the ℓ∞ ball are products of intervals.
Their projections clamp each coordinate, and they have scalar proximal
operators for coordinate descent. A box whose bounds exclude zero does not
preserve zeros (see [`iszeropreserving`](@ref)), so on a sparse iterate it
visits every entry. [`Constrained`](@ref) restricts a convex separable function
to a box, and the result is again separable.

## The simplex and the ℓ1 ball

The projection onto the simplex ``\Delta_r = \{x \ge 0 : \sum_i x_i = r\}`` is
``x_i = \max(v_i - \tau, 0)``. Here ``\tau`` is the multiplier of the equality
constraint, the unique solution of

```math
\sum_i \max(v_i - \tau, 0) = r .
```

The left side is continuous, piecewise linear and decreasing in ``\tau``, so
``\tau`` lies between two sorted values of ``v``. Sorting finds it in
``O(n\log n)``. Condat's algorithm (2016) finds it in ``O(n)`` expected time
with one pass and a scratch buffer, which the function keeps.

When ``r`` is large, the level of Condat's algorithm starts far below the
entries, so many entries become candidates and are then dropped one by one.
When more than an eighth of the entries, and more than 1024, become candidates,
Michelot's algorithm (1986) takes over:
the level of all the entries, then of those above it, until that set stops
changing. It is Newton's method on the left side of the equation above, from
below, and each of its steps is a vectorized pass. With
`equality = false` the set is ``\sum_i x_i \le r``, and ``\tau = 0`` when ``v``
already satisfies the inequality.

Projecting onto the ℓ1 ball ``\{\|x\|_1 \le r\}`` is trivial when
``\|v\|_1 \le r``. Otherwise the projection has the signs of ``v`` and solves
the simplex problem for ``|v|``. The ℓ∞ norms use this projection through
Moreau's decomposition.

## Sparsity

The projection onto ``\{x : \#\{i : x_i \ne 0\} \le s\}`` keeps the ``s``
entries of largest magnitude, a nonconvex set with an exact projection.
[`IndBallL0`](@ref) selects them with Floyd and Rivest's algorithm in ``O(n)``
expected time, in place, and in ``O(\mathrm{nnz})`` on a `SparseIterate`. It
keeps exactly ``s`` entries when there are ties. Projected gradient descent with
this set is iterative hard thresholding. On a symmetric matrix, ``s`` counts the
entries of one triangle, so that an entry and its transpose are kept or dropped
together. Dropping an off-diagonal pair costs ``2x_{ij}^2`` in the Frobenius
norm and a diagonal entry ``x_{ii}^2``, so the projection compares
``\sqrt{2}\,|x_{ij}|`` with ``|x_{ii}|``. With `diagonal = false` the diagonal
does not count and is kept, as for the off-diagonal sparsity of a precision
matrix.

## Hyperplanes and halfspaces

Minimizing ``\|z - v\|^2`` subject to ``\langle a, z\rangle = b`` gives
``z = v - \mu a`` with ``\mu = (\langle a, v\rangle - b)/\|a\|^2``. For the
halfspace ``\langle a, z\rangle \le b``, the same step applies when ``v``
violates the inequality, and ``z = v`` otherwise.

A scalar `a` stands for an array of `a`s of the size of the argument, so
`IndHyperplane(1, r)` is the set of arrays whose entries sum to ``r``, of any
size. [`Constrained`](@ref) restricts a convex separable function to a
hyperplane or a halfspace.

## Positive semidefinite and low-rank matrices

The projection of a symmetric ``V = U\Lambda U^\top`` onto the positive
semidefinite cone, in the Frobenius norm, sets its negative eigenvalues to zero.
This holds because the Frobenius norm is unitarily invariant, and the nearest
diagonal matrix with nonnegative entries clips the negative ones.
The sets that bound the eigenvalues, such as the Fantope
``\{X : 0 \preceq X \preceq I,\ \operatorname{tr}(X) = k\}``, are [`Spectral`](@ref) of
the corresponding sets of vectors.
[`IndPSD`](@ref) reads the upper triangle of its argument. It computes all the
eigenpairs with LAPACK's divide-and-conquer `syevd`, or with `rank = r` only the
``r`` largest with `syevr`, keeping the positive ones. It rebuilds the result
from the eigenvectors it keeps with one symmetric rank-``k`` update. [`IndBallRank`](@ref)`(r)` keeps the ``r`` largest singular values
of a general matrix, which by the Eckart–Young theorem gives the nearest matrix
of rank at most ``r``. Both rank sets are nonconvex with exact projections.
Projected gradient with them is hard-impute in matrix completion.
