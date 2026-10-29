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
| [`IndRotatedSOC`](@ref)`()` | ``\lVert x_{3:n}\rVert_2^2 \le 2x_1x_2``, ``x_1,x_2 \ge 0`` | an orthogonal change of coordinates to the second-order cone |
| [`IndExpPrimal`](@ref)`()` | ``\operatorname{cl}\{(r,s,t):s>0,\ s e^{r/s}\le t\}`` | a bracketed univariate root problem |
| [`IndExpDual`](@ref)`()` | the dual exponential cone | Moreau decomposition with the primal cone |
| [`IndSphereL2`](@ref)`(r)` | ``\lVert x\rVert_2 = r``, nonconvex | ``r\,v/\lVert v\rVert_2`` |
| [`IndBallL1`](@ref)`(r)` | ``\lVert x\rVert_1 \le r`` | ``\operatorname{sign}(v)\odot P_{\Delta_r}(\lvert v\rvert)``, by Condat's algorithm |
| [`IndSimplex`](@ref)`(r)` | ``x \ge 0,\ \sum_i x_i = r`` | ``\max(v - \tau, 0)`` |
| [`IndBallL0`](@ref)`(s)` | at most ``s`` nonzero entries | keep the ``s`` largest in magnitude |
| [`IndBallGroupL0`](@ref)`(s, groups)` | at most ``s`` nonzero groups | keep the ``s`` groups of largest norm |
| [`IndHyperplane`](@ref)`(a, b)` | ``\langle a, x\rangle = b`` | ``v - \frac{\langle a, v\rangle - b}{\lVert a\rVert^2}\,a`` |
| [`IndHalfspace`](@ref)`(a, b)` | ``\langle a, x\rangle \le b`` | the same when ``\langle a, v\rangle > b`` |
| [`IndHyperslab`](@ref)`(a, l, u)` | ``l \le \langle a, x\rangle \le u`` | project onto the violated bound, or keep ``v`` inside |
| [`IndIsotonic`](@ref)`()` | ``x_1 \le x_2 \le \dots \le x_n`` | isotonic regression, by pool adjacent violators |
| [`IndIntersection`](@ref)`(C₁, C₂, …)` | ``C_1 \cap C_2 \cap \cdots`` | Dykstra's algorithm, iterative |
| [`IndAffine`](@ref)`(A, b)` | ``Ax = b`` | ``v - A^\top(AA^\top)^{-1}(Av - b)``, from a factorization cached at construction |
| [`IndPolyhedron`](@ref)`(A, l, u, xmin, xmax)` | ``l \le Ax \le u,\ x_{\min} \le x \le x_{\max}`` | a dual active-set method, with a cached working set |
| [`IndGraph`](@ref)`(A)` | ``y = Ax``, on pairs ``(x, y)`` | factor the smaller of ``I + A^\top A`` and ``I + AA^\top`` once |
| [`IndPSD`](@ref)`()` | ``X \succeq 0`` | ``U\max(\Lambda, 0)U^\top`` |
| [`IndPSD`](@ref)`(rank = r)` | ``X \succeq 0``, rank at most ``r`` | the ``r`` largest positive eigenpairs |
| [`IndBallRank`](@ref)`(r)` | rank at most ``r`` | the truncated SVD |
| [`IndStiefel`](@ref)`()` | ``X^\top X = I_k``, ``k \le p``, nonconvex | ``UV^\top`` from the thin SVD |
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

[`IndRotatedSOC`](@ref)`()` uses the order ``(u,v,z)`` for the cone
``\{(u,v,z) : \|z\|_2^2 \le 2uv,\ u,v \ge 0\}``. The orthogonal map
``Q(u,v,z) = ((u+v)/\sqrt{2}, (u-v)/\sqrt{2}, z)`` takes it to the
second-order cone, so its projection is ``Q^\top P_K Q``. The implementation
computes this projection without forming ``Q`` and preserves small heads
when the two heads have very different magnitudes. An array needs at least
two entries. `IndRotatedSOC(groups)` uses the first two listed indices of
each group as its heads. Each group needs at least two indices; a group of
two is the nonnegative quadrant. Group ownership, free entries, linear
indexing, non-finite entries and scalar steps follow `IndSOC`. Its conjugate
is the polar indicator, with the negative cone on each group and zero on
free entries. The single rotated cone is self-dual.

For the value oracle, `tol = √eps(float(eltype(x)))`. The heads may be as
small as ``-\mathtt{tol}\max(1,|u|,|v|)``. In a frame divided by ``2^k``,
write ``a=u/2^k``, ``b=v/2^k`` and ``s=\|z/2^k\|_2``. The quadratic
inequality is checked without squaring as
``\operatorname{hypot}(a-b,\sqrt{2}s) \le a+b+\mathtt{tol}\max(2^{-k},|a|,|b|)``.

The cone is the epigraph of the perspective ``\|z\|_2^2/(2u) \le v``
for ``u>0``, with its closed extension at ``u=0``. It represents a
quadratic divided by a scale, as in concomitant scale estimation, or by a
selection variable shared by several coefficients. For a scalar coefficient,
``\beta^2 \le as`` is `IndRotatedSOC()` on `(a, s/2, β)`, stored as an
array. When the selection variables are uncoupled, minimizing
``\lambda_0 a + \lambda_2\beta^2/a`` over ``0 \le a \le 1`` gives
[`Berhu`](@ref)`(2√(λ₀λ₂), √(λ₀/λ₂))`. With coupled selection variables,
such as a shared budget, the cone retains their joint constraint.

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

## Polyhedra

[`IndPolyhedron`](@ref)`(A, l, u)` projects onto several linear equalities
and inequalities at once. Optional coordinate bounds give
`IndPolyhedron(A, l, u, xmin, xmax)`. Bounds may be scalars or vectors;
infinite sides are omitted. Equal finite bounds are equalities. Dense and
sparse matrices are supported, and the constructor copies the data.

This projection adds linear constraints to a loss through splitting. For
example, constrained lasso combines a least-squares loss and an ℓ1 penalty
with equalities and inequalities on the coefficients. Ordered effects,
nonnegativity with a budget, and sign constraints on compositional coefficients
all have this form. For order restrictions on a partial order, a row with
``1`` at ``i`` and ``-1`` at ``j``, with upper bound zero, enforces
``\theta_i \le \theta_j``. [`IndIsotonic`](@ref) provides a faster specialized
projection when the order is a chain.

The dual active-set method adds violated sides and drops inequalities when
their multipliers reach zero. Native equalities have free multipliers and
are never dropped. A thin orthogonal factorization avoids forming ``AA^\top``
and handles redundant rows. Successive calls reuse the working set;
[`reset_cache!`](@ref) clears it. Calls on one function share buffers and must
be sequential. The workspace uses ``O(n^2)`` storage. Each search costs
``O(mn)`` for dense data or ``O(\operatorname{nnz}(A) + m + n)`` for sparse data,
and factor updates cost ``O(n|S|)``. A large working set can be expensive.

Membership uses a relative ``\sqrt{\varepsilon}`` tolerance, with an absolute
floor in the original constraint units. Projection decisions use the tighter
``8\varepsilon`` scale of normalized rows. Thus a point accepted by the value
can still move under projection. The dependence threshold is
``8n\varepsilon`` times the normal's norm. Rows are scaled by powers of two;
scaled nonzero coefficients, finite offsets and the projection arithmetic
must remain representable. Accuracy is limited by numerical rank and output
precision.

An infeasibility certificate, checked to ``64n\varepsilon`` relative to its
normal and offset scales and verified exactly on the stored data, causes an
`ArgumentError` identifying an empty set. This exceptional verification
allocates. It preserves genuine small directions when the rank test is
inconclusive, rather than declaring a feasible set empty.
Numerical failure and the iteration limit (`maxiter`, default
`100 * (sum(size(A)) + 1)`) instead cause distinct errors. As for
[`IndIntersection`](@ref), `value_and_prox!` returns zero without a feasibility
certificate. It does not replace checking the constraints of an ill-conditioned
problem. `Constrained(f, IndPolyhedron(...))` is not provided.

## Graphs of linear maps

[`IndGraph`](@ref)`(A)` projects a tuple `(c, d)` onto ``\{(x, y) : y = Ax\}``.
For an ``m \times n`` matrix, it factors ``I + A^\top A`` when ``n \le m`` and
``I + AA^\top`` otherwise. Dense projections allocate nothing after
construction. Sparse matrices keep a sparse factorization, but each solve
allocates and may compute in `Float64` before conversion to the data's floating
type. The Gram matrix must be representable and retain positive definiteness
after adding the identity; unsupported numerical ranges throw at construction.
An operator with `size`, a real `eltype`, and products with itself and its
adjoint uses conjugate gradients on the smaller system. Its `rtol` bounds the
relative true residual of that system, and `maxiter` can stop before convergence;
check `f.converged` and `f.iterations`. The matrix or operator is stored by
reference and must stay unchanged. Construct a new function after changing it.

Graph-form splitting writes ``\min_x \ell(Ax) + r(x)`` as
``\min_{x,y} \ell(y) + r(x)`` subject to ``y = Ax`` (Parikh and Boyd, 2014;
Fougner and Boyd, 2018). ADMM then uses the proximal operators of the loss and
penalty and this graph projection. The same factorization serves squared,
Huber, check, hinge, logistic and Poisson losses of the predictor ``X\beta``,
different penalties, and every regularization parameter on a path.
Row and column equilibration belongs in the solver.

## Hyperplanes, halfspaces and hyperslabs

Minimizing ``\|z - v\|^2`` subject to ``\langle a, z\rangle = b`` gives
``z = v - \mu a`` with ``\mu = (\langle a, v\rangle - b)/\|a\|^2``. For the
halfspace ``\langle a, z\rangle \le b``, the same step applies when ``v``
violates the inequality, and ``z = v`` otherwise.

A scalar `a` stands for an array of `a`s of the size of the argument, so
`IndHyperplane(1, r)` is the set of arrays whose entries sum to ``r``, of any
size. [`Constrained`](@ref) restricts a convex separable function to a
hyperplane, a halfspace or a hyperslab.

[`IndHyperslab`](@ref)`(a, l, u)` bounds one linear functional on both sides.
Its projection keeps a point inside the slab and otherwise projects onto the
violated bound, in one pass over the entries for the correction. Equal bounds
give a hyperplane, one infinite bound gives a halfspace, and two infinite
bounds give the whole space. Membership checks each finite bound with its own
floating-point tolerance, so a distant bound does not hide a violation of the
other. The normal and each finite bound use the overflow-safe scaling of
`IndHyperplane`; see the docstring for its representability conditions.

For example, `IndHyperslab(1, l, u)` bounds the sum of the coefficients, and
`IndHyperslab(a, -c, c)` bounds the magnitude of a contrast. For a linear model,
`a` can be the difference between two groups' mean covariates to bound their
gap in mean predictions. `Spectral(IndHyperslab(1, l, u))` bounds the trace of
a symmetric matrix.

## Monotone vectors

The projection onto the nondecreasing vectors, [`IndIsotonic`](@ref), is
isotonic regression. Pool adjacent violators scans the entries, pooling each
entry with the block before it while their means are out of order. Each block
then takes its mean, the projection of the block onto the constants. With a
stack of blocks in buffers that the function owns, it runs in O(n) time.
`IndIsotonic(decreasing = true)` projects onto the nonincreasing vectors, and
the same pooling gives the proximal operator of [`SLOPE`](@ref).
An array of steps `γ` gives weighted isotonic regression, minimizing
`Σᵢ wᵢ(θᵢ − vᵢ)²` with `wᵢ = 1/γᵢ`, by the same pooling with weighted means.
See the [weighted isotonic regression example](examples/19_weighted_isotonic_regression.md).

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

## Orthonormal columns

[`IndStiefel`](@ref)`()` projects a real ``p \times k`` matrix onto the
matrices with orthonormal columns, for ``k \le p``. If
``X = U\Sigma V^\top`` is a thin SVD, the Frobenius projection is
``UV^\top``. All ``k`` singular-vector columns are used, including an
orthonormal completion at rank-deficient inputs. The projection is unique
at full column rank. Otherwise the SVD chooses a minimizer, deterministically
for a fixed input, LAPACK backend and configuration. The choice need not be
continuous across rank-deficient inputs. With ``k = p`` the set is the
orthogonal group, including reflections; with ``k = 1`` it is the unit
sphere written as a column matrix.

This projection gives gradient projection steps for factor rotation and
k-component PCA. Projecting ``A^\top B`` gives a minimizer of the orthogonal
Procrustes problem ``\min_{Q^\top Q = I}\|AQ - B\|_F``. The step size does
not affect the projection. The alternative formula
``X(X^\top X)^{-1/2}`` requires full column rank and squares the condition
number when forming the Gram matrix, so this implementation uses the SVD.
There is no general closed form for `Constrained(f, IndStiefel())`.

The value checks ``\|X^\top X - I\|_F \le \sqrt{\varepsilon}\sqrt{k}``
in the argument's floating precision. A value of zero only reports this
tolerance check. Nonfinite inputs have value `Inf` and cannot be projected.
The proximal workspace follows the output's precision, which must be
`Float32` or `Float64`; scaling precedes conversion to preserve direction
at extreme magnitudes. Buffers are reused for fixed types and sizes.

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
