# Sets

The indicator function ``\iota_C`` of a set ``C`` is ``0`` on ``C`` and
``+\infty`` outside. Its proximal operator is the Euclidean projection
``P_C(v) = \operatorname{argmin}_{z \in C}\|z - v\|``, for every step, so
projected gradient descent is proximal gradient descent with an indicator.
[`value_and_prox!`](@ref) returns ``0`` without checking membership. Closed-form
projections use exact formulas, but computed results have rounding errors that
grow with the conditioning of the problem. Projections that determine a numerical
rank, such as [`IndAffine`](@ref), project onto the set of the truncated data
when the data are nearly rank deficient. Iterative projections, such as
[`IndIntersection`](@ref), use a stopping rule that need not bound the distance
from the returned point to the set; a returned value of ``0`` is not a
feasibility certificate.

| Function | Set | Projection |
|:---------|:----|:-----------|
| [`IndBox`](@ref)`(l, u)` | ``l \le x \le u`` | ``\operatorname{clamp}(v, l, u)`` |
| [`IndNonnegative`](@ref)`()` | ``x \ge 0`` | ``\max(v, 0)`` |
| [`IndBinary`](@ref)`(l, u)` | ``x_i \in \{l_i, u_i\}``, nonconvex | nearer endpoint, ties to ``l_i`` |
| [`IndBallLinf`](@ref)`(r)` | ``\lVert x\rVert_\infty \le r`` | ``\operatorname{clamp}(v, -r, r)`` |
| [`IndBallL2`](@ref)`(r)` | ``\lVert x\rVert_2 \le r`` | ``\min(1, r/\lVert v\rVert_2)\,v`` |
| [`IndSOC`](@ref)`()` | ``\lVert x_{2:n}\rVert_2 \le x_1`` | closed form in the head and the tail norm |
| [`IndExpPrimal`](@ref)`()` | ``\operatorname{cl}\{(r,s,t):s>0,\ s e^{r/s}\le t\}`` | a bracketed univariate root problem |
| [`IndExpDual`](@ref)`()` | the dual exponential cone | Moreau decomposition with the primal cone |
| [`IndSphereL2`](@ref)`(r)` | ``\lVert x\rVert_2 = r``, nonconvex | ``r\,v/\lVert v\rVert_2`` |
| [`IndBallL1`](@ref)`(r)` | ``\lVert x\rVert_1 \le r`` | ``\operatorname{sign}(v)\odot P_{\Delta_r}(\lvert v\rvert)``, by Condat's algorithm |
| [`IndSimplex`](@ref)`(r)` | ``x \ge 0,\ \sum_i x_i = r`` | ``\max(v - \tau, 0)`` |
| [`IndBallL0`](@ref)`(s)` | at most ``s`` nonzero entries | keep the ``s`` largest in magnitude |
| [`IndBallGroupL0`](@ref)`(s, groups)` | at most ``s`` nonzero groups | keep the ``s`` groups of largest norm |
| [`IndHyperplane`](@ref)`(a, b)` | ``\langle a, x\rangle = b`` | ``v - \frac{\langle a, v\rangle - b}{\lVert a\rVert^2}\,a`` |
| [`IndHalfspace`](@ref)`(a, b)` | ``\langle a, x\rangle \le b`` | the same when ``\langle a, v\rangle > b`` |
| [`IndIsotonic`](@ref)`()` | ``x_1 \le x_2 \le \dots \le x_n`` | isotonic regression, by pool adjacent violators |
| [`IndIntersection`](@ref)`(C₁, C₂, …)` | ``C_1 \cap C_2 \cap \cdots`` | Dykstra's algorithm, iterative |
| [`IndAffine`](@ref)`(A, b)` | ``Ax = b`` | ``v - A^\top(AA^\top)^{-1}(Av - b)``, from a factorization cached at construction |
| [`IndPSD`](@ref)`()` | ``X \succeq 0`` | ``U\max(\Lambda, 0)U^\top`` |
| [`IndPSD`](@ref)`(rank = r)` | ``X \succeq 0``, rank at most ``r`` | the ``r`` largest positive eigenpairs |
| [`IndBallRank`](@ref)`(r)` | rank at most ``r`` | the truncated SVD |
| [`SingularValue`](@ref)`(IndBallL1(r))` | ``\lVert X\rVert_* \le r`` | the ℓ1-ball projection of the singular values |
| [`SingularValue`](@ref)`(IndBallLinf(r))` | ``\lVert X\rVert_2 \le r`` | ``U\min(\Sigma, r)V^\top`` |
| [`IndPattern`](@ref)`(P)` | ``x_i = 0`` off the pattern ``P`` | zero the entries off ``P`` |
| [`IndBanded`](@ref)`(k, p)` | ``X_{ij} = 0`` for ``\lvert i - j\rvert > k`` | zero the entries outside the band |
| [`IndToeplitz`](@ref)`()` | ``X_{ij} = c_{j - i}`` | the mean of each diagonal |

## Separable sets

The box, the nonnegative orthant and the ℓ∞ ball are products of intervals.
Their projections clamp each coordinate, and they have scalar proximal
operators for coordinate descent. A box whose bounds exclude zero does not
preserve zeros (see [`iszeropreserving`](@ref)), so on a sparse iterate it
visits every entry. [`Constrained`](@ref) restricts a convex separable function
to a box, and the result is again separable.

[`IndBinary`](@ref)`(l, u)` instead allows two finite values per entry, with
``l_i < u_i``. Its projection selects the nearer endpoint, independently of the
step, with ties going to the lower endpoint. Alternating multiplication by a
matrix and projection onto ``\{-1, 1\}^n`` gives the projected power method
for community detection and synchronization (Chen and Candès, 2018). Zero
projects to ``-1``, unlike `sign(0) = 0`. Bounds may be scalars or arrays,
indexed linearly as for `IndBox`. Array bounds are stored without copying;
construct a new `IndBinary` after changing them. Zero is preserved exactly when
each pair contains zero, including mixed pairs such as ``\{0, 1\}`` and
``\{-1, 0\}``; then sparse iterates visit only stored entries, and `restrict`
is supported. A proximal gradient update from zero projects ``-\gamma g``,
so whether it stays zero depends on the step, and no gradient-only
`kkt_violation` is provided. With scalar bounds,
`Spectral(IndBinary(0, 1))` projects onto the orthogonal projection matrices.

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

## Second-order cones

[`IndSOC`](@ref)`()` is the epigraph of the Euclidean norm: the first entry
is a variable bound on the norm of the remaining entries. With head ``t`` and
tail norm ``s``, the projection keeps the point when ``s \le t``, returns zero
when ``s \le -t``, and otherwise returns
``(\alpha, (\alpha/s)x_{2:n})`` with ``\alpha = (t+s)/2``.
Arrays are read by one-based linear index, including matrices. An empty array
has no head and is rejected. Only scalar steps are supported.

`IndSOC(groups)` projects each disjoint group onto its own cone. The first
listed index is the head, even when the indices are out of order; a group of
one index imposes nonnegativity. For example, `IndSOC([[3, 1, 2], [5, 4]])`
bounds the norm of entries 1 and 2 by entry 3, and the magnitude of entry 4 by
entry 5. Groups are copied and must be nonempty. Entries in no group are free
and remain unchanged; an empty collection of groups leaves all entries free.
[`Conjugate`](@ref) gives the polar indicator, which requires the negative
cone on each group and zero on every free entry, within the indicator's
tolerance. With a single cone the polar is its negative, since the
second-order cone is self-dual.

## Exponential cones

[`IndExpPrimal`](@ref)`()` projects a triple in the order ``(r,s,t)`` onto
``K_{\exp} = \operatorname{cl}\{(r,s,t):s>0,\ s\exp(r/s)\le t\}``.
The face at ``s=0`` is ``r\le0,\ t\ge0``.
[`IndExpDual`](@ref)`()` projects onto
``K_{\exp}^* = \operatorname{cl}\{(u,v,w):u<0,\ -u\exp(v/u)\le e w\}``,
whose face at ``u=0`` is ``v\ge0,\ w\ge0``.
Arrays use one-based linear indexing and must have exactly three entries.

Both constructors accept disjoint triples of indices, for example
`IndExpPrimal([[3, 1, 2], [6, 4, 5]])`. Each triple uses its listed order.
Groups are copied. Entries outside the groups are free and are copied by the
projection; an empty collection leaves all entries free. Only scalar steps
are supported. The two grouped constructors are not duals when free entries
exist: [`Conjugate`](@ref) requires zero on those entries and the polar cone
on each triple. The docstrings specify the tolerant membership inequalities.
Projection branches use exact inequalities, separately from that tolerance.

The projection follows Section 5 and Algorithm 1 of
[Friberg (2023)](https://doi.org/10.1080/10556788.2021.2022147).
Closed-form and heuristic candidates handle the faces and extreme root
parameters. Otherwise, a finite bracket bounds damped Newton steps, followed
by bisection after twenty steps. The search stops at floating-point resolution
or after one hundred steps and compares the last candidate with the heuristics
by distance. The paper writes the coordinates as ``(t,s,r)``.
The dual projection uses ``P_{K^*}(x)=x+P_K(-x)``; direct reconstruction of
the polar component avoids cancellation. Scaled equations and exponential
products avoid intermediate overflow, including when ``\exp(r/s)`` alone
is unrepresentable. A component can overflow if the projection itself is
unrepresentable in the output type. Non-finite entries give value `Inf` and
project to `NaN` within their triple; free entries are copied unchanged.

These cones represent exponential and relative-entropy constraints in conic
splitting methods. Direct proximal operators such as [`NegEntropy`](@ref),
[`KLDivergence`](@ref), [`NegLog`](@ref), and the GLM losses are usually cheaper
when the model admits them directly.

## Intersections

The projection onto a nonempty intersection ``C_1 \cap \cdots \cap C_m`` of
closed convex sets need not have a closed form even when each ``C_k`` has one:
the correlation matrices, for instance, are the positive semidefinite matrices
with unit diagonal.
Alternating the projections converges to a point of the intersection, which
need not be the projection. Dykstra's algorithm (Boyle and Dykstra, 1986) keeps
a correction ``p_k`` for each set and repeats, for ``k = 1, \dots, m``,

```math
y \leftarrow P_{C_k}(y + p_k), \qquad p_k \leftarrow (y_{\mathrm{old}} + p_k) - y ,
```

from ``y = v`` and ``p_k = 0``. Throughout, ``v = y + \sum_k p_k``, and the
iterates converge to ``P_C(v)``. The algorithm is block coordinate ascent on the
dual of the projection problem, whose variables are the corrections (Tibshirani,
2017), so it converges from any corrections. [`IndIntersection`](@ref) uses this
to start each call from the corrections of the previous one, which helps inside
a solver, where successive inputs are close. It stops when the sum of squared
step lengths in a round is at most
``\max(\mathtt{tol}, 10\varepsilon)^2\max(1, \|y\|)^2``, or after `maxiter`
rounds. Only the first non-converged call in a session warns; later calls are
silent. `f.converged` reports only whether the step-length test
was met, and `iterations` counts the rounds of the last call. Neither stopping
condition bounds the distance to the intersection: nearly parallel sets can
meet the step-length test far from it, and `maxiter` can also stop before
feasibility. Linear convergence requires geometric assumptions, as for a
nonempty intersection of affine subspaces; there is no uniform linear rate for
arbitrary closed convex sets, and small angles can make convergence slow.

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

[`IndBallGroupL0`](@ref) is the same for groups: it keeps the ``s`` groups of
largest Euclidean norm, and projected gradient descent with it is iterative hard
thresholding of groups. With `k`, each group also keeps at most ``k`` nonzero
entries, the doubly sparse set of group IHT with sparsity within the groups.
The projection ranks the groups by the norm of their ``k`` largest entries and
keeps those entries of the ``s`` best groups.

## Affine sets

[`IndAffine`](@ref)`(A, b)` projects onto ``\{x : Ax = b\}`` with several
equality constraints at once. It factors ``AA^\top`` once, by Cholesky when
``A`` has full row rank. Otherwise it caches a thin SVD
``A = U\Sigma V^\top`` of the rank of ``A``, and the projection is
``v - V(V^\top v - \Sigma^{-1}U^\top b)``, defined when ``b`` is in the range of
``A``. A tight frame, ``AA^\top = \mu I``, needs no factorization
(`IndAffine(A, b; μ)`). Each call costs two products with ``A`` and allocates
nothing. With ADMM, it gives basis pursuit, ``\min \|x\|_1`` subject to
``Ax = b``. [`Constrained`](@ref)`(f, IndAffine(A, b))` restricts a
least-squares loss or a quadratic to the set, the ``f`` of ADMM for
equality-constrained estimators such as log-contrast regression with several
subcompositions.

## Hyperplanes and halfspaces

Minimizing ``\|z - v\|^2`` subject to ``\langle a, z\rangle = b`` gives
``z = v - \mu a`` with ``\mu = (\langle a, v\rangle - b)/\|a\|^2``. For the
halfspace ``\langle a, z\rangle \le b``, the same step applies when ``v``
violates the inequality, and ``z = v`` otherwise.

A scalar `a` stands for an array of `a`s of the size of the argument, so
`IndHyperplane(1, r)` is the set of arrays whose entries sum to ``r``, of any
size. [`Constrained`](@ref) restricts a convex separable function to a
hyperplane or a halfspace.

## Monotone vectors

The projection onto the nondecreasing vectors, [`IndIsotonic`](@ref), is
isotonic regression. Pool adjacent violators scans the entries, pooling each
entry with the block before it while their means are out of order. Each block
then takes its mean, the projection of the block onto the constants. With a
stack of blocks in buffers that the function owns, it runs in O(n) time.
`IndIsotonic(decreasing = true)` projects onto the nonincreasing vectors, and
the same pooling gives the proximal operator of [`SLOPE`](@ref).

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

With `partial = true`, [`IndPSD`](@ref) and [`IndBallRank`](@ref) compute only
the eigenpairs or singular triplets they keep; see [Partial decompositions](@ref).

## Structured matrices

[`IndPattern`](@ref)`(P)` is the set of arrays supported on a pattern ``P``, and
[`IndBanded`](@ref) the set of banded matrices. Each is a product of one set per
entry, ``\{0\}`` off the pattern or the band and ``\mathbb{R}`` on it, so the
projection zeroes the entries outside, and both are separable: they preserve
zeros and have scalar proximal operators for coordinate descent. `IndBanded`
holds the size of its matrices, which gives the row and the column of an entry
from its linear index, instead of a pattern. Banding a sample covariance is its
projection (Bickel and Levina, 2008). With [`LogDetLoss`](@ref) in ADMM,
`IndPattern` gives the refit on a known graph: the Gaussian maximum likelihood
estimate whose precision matrix is zero off the graph (see the graphical lasso
example).

The Toeplitz matrices, constant along each diagonal, form a subspace, so the
projection onto them is orthogonal. On each diagonal ``D_d = \{(i, j) : j - i = d\}``
it is the mean of the entries of ``V``:

```math
P(V)_{ij} = \frac{1}{\lvert D_{j-i}\rvert} \sum_{(k, l) \in D_{j-i}} V_{kl} .
```

With `symmetric = true`, [`IndToeplitz`](@ref) projects onto the symmetric
Toeplitz matrices and averages the diagonals ``d`` and ``-d`` together. This is
a structural constraint of covariance matrices of a stationary series.
Covariance matrices must also be positive semidefinite, which requires a
separate constraint such as [`IndPSD`](@ref).
