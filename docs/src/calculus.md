# Functions built from functions

These wrappers build a function from others and carry over its oracles, so a
new function inherits a proximal operator, a gradient and a Lipschitz constant
from known ones.

| Function | Value | Proximal operator | Gradient |
|:---------|:------|:------------------|:---------|
| [`Scaled`](@ref)`(f, α)`, also `α * f` | ``\alpha f(x)`` | ``\operatorname{prox}_{(\gamma\alpha) f}(x)`` | ``\alpha\nabla f(x)`` |
| [`Tilt`](@ref)`(f, c, d)` | ``f(x) + \langle c, x\rangle + d`` | ``\operatorname{prox}_{\gamma f}(x - \gamma c)`` | ``\nabla f(x) + c`` |
| [`Translate`](@ref)`(f, b)` | ``f(x - b)`` | ``b + \operatorname{prox}_{\gamma f}(x - b)`` | ``\nabla f(x - b)`` |
| [`Conjugate`](@ref)`(f)` | ``f^*(x) = \sup_z \langle z, x\rangle - f(z)`` | ``x - \gamma\operatorname{prox}_{f/\gamma}(x/\gamma)`` | |
| [`SeparableSum`](@ref)`(f1, f2, ...)` | ``\sum_k f_k(x_k)`` on a tuple of blocks | block by block | block by block |
| [`Sum`](@ref)`(f1, f2, ...)` | ``\sum_k f_k(x)`` | | ``\sum_k \nabla f_k(x)`` |
| [`Constrained`](@ref)`(f, C)` | ``f(x)`` on a box, hyperplane, halfspace or simplex ``C`` | on a box, ``\operatorname{clamp}(\operatorname{prox}_{\gamma f}(x), l, u)``; on a hyperplane, ``\operatorname{prox}_{\gamma f}(x - \gamma\nu a)`` | |
| [`Spectral`](@ref)`(f)` | ``f(\lambda(X))`` for a symmetric ``X`` | ``V\operatorname{diag}(\operatorname{prox}_{\gamma f}(\lambda))V^\top`` | |
| [`MoreauEnvelope`](@ref)`(f, μ)` | ``\min_z f(z) + \tfrac{1}{2\mu}\lVert z - x\rVert^2`` | ``x + \tfrac{\gamma}{\mu + \gamma}\big(\operatorname{prox}_{(\mu+\gamma) f}(x) - x\big)`` | ``\big(x - \operatorname{prox}_{\mu f}(x)\big)/\mu`` |
| [`Precompose`](@ref)`(f, A, μ)`, ``AA^\top = \mu I`` | ``f(Ax)`` | ``x + \tfrac1\mu A^\top\big(\operatorname{prox}_{\mu\gamma f}(Ax) - Ax\big)`` | ``A^\top\nabla f(Ax)`` |
| [`Regularize`](@ref)`(f, ρ, c)` | ``f(x) + \tfrac{\rho}{2}\lVert x - c\rVert^2`` | ``\operatorname{prox}_{\frac{\gamma}{1+\gamma\rho} f}\big(\tfrac{x + \gamma\rho c}{1 + \gamma\rho}\big)`` | ``\nabla f(x) + \rho(x - c)`` |

## Derivations

Each rule follows from substituting into the definition of the proximal
operator.

- **Scaling.** ``\gamma(\alpha f) = (\gamma\alpha) f``, so the proximal operator
  of ``\alpha f`` with step ``\gamma`` is that of ``f`` with step
  ``\gamma\alpha``. The Lipschitz constant scales by ``\alpha``.
- **Tilting.** Completing the square,
  ``f(z) + \langle c, z\rangle + \frac{1}{2\gamma}\|z - x\|^2 = f(z) + \frac{1}{2\gamma}\|z - (x - \gamma c)\|^2 + \text{const}``.
- **Translation.** Substituting ``u = z - b`` turns the problem for
  ``f(\cdot - b)`` at ``x`` into the problem for ``f`` at ``x - b``.
- **Conjugation.** Moreau's decomposition (see [The interface](@ref)), with
  ``f/\gamma`` in place of ``f``.
- **Blocks.** For ``\sum_k f_k(x_k)`` the problem defining the proximal operator
  separates over the blocks, so a solver can update a tuple of variables, such
  as the sparse and low-rank parts of robust PCA, with one call.
- **Envelopes.** For the Moreau envelope ``e`` of ``f`` with parameter ``\mu``,
  let ``P = \operatorname{prox}_{(\mu+\gamma) f}(x)`` and
  ``y = x + \frac{\gamma}{\mu+\gamma}(P - x)``. Then
  ``(y - P)/\mu = (x - P)/(\mu + \gamma) \in \partial f(P)``, so
  ``P = \operatorname{prox}_{\mu f}(y)``, and ``(x - y)/\gamma = (y - P)/\mu = \nabla e(y)``:
  ``y`` is the proximal point of ``\gamma e``. The value ``e(y)`` comes from
  ``f(P)``, with no second evaluation.
- **Composition with ``A``, ``AA^\top = \mu I``.** Write ``z = x + A^\top u``
  plus a component in the null space of ``A``; the problem for ``f(Az)`` at
  ``x`` becomes the problem for ``f`` at ``Ax`` with step ``\mu\gamma``, and
  ``A\operatorname{prox}_{\gamma f\circ A}(x) = \operatorname{prox}_{\mu\gamma f}(Ax)``.
- **A ridge term.** ``f(z) + \frac{\rho}{2}\|z - c\|^2 + \frac{1}{2\gamma}\|z - x\|^2``
  equals ``f(z) + \frac{1 + \gamma\rho}{2\gamma}\big\|z - \frac{x + \gamma\rho c}{1 + \gamma\rho}\big\|^2``
  plus a constant.

```@example calculus
using ProximalBase, LinearAlgebra
x, c, γ = [2.0, -1.0, 0.5], [0.3, 0.3, -0.3], 0.7
@assert prox(Tilt(NormL1(1.0), c, 0.0), x, γ) ≈ prox(NormL1(1.0), x - γ * c, γ) # hide
prox(Tilt(NormL1(1.0), c, 0.0), x, γ) ≈ prox(NormL1(1.0), x - γ * c, γ)
```

## Conjugates

The conjugate of a convex function gives a second function for free, with its
proximal operator from Moreau's decomposition. The conjugate of a norm is the
indicator of the unit ball of the dual norm, and the conjugate of a ball's
indicator is the dual norm scaled by the radius:

| ``f`` | ``f^*`` |
|:------|:--------|
| ``\lambda\lVert x\rVert_1`` | the indicator of ``\lVert y\rVert_\infty \le \lambda`` |
| ``\lambda\lVert x\rVert_2`` | the indicator of ``\lVert y\rVert_2 \le \lambda`` |
| ``\lambda\lVert x\rVert_\infty`` | the indicator of ``\lVert y\rVert_1 \le \lambda`` |
| the indicator of ``l \le x \le u`` | ``\sum_i \max(l_i y_i, u_i y_i)`` |
| ``\tfrac{\lambda}{2}\lVert x - c\rVert^2``, ``\lambda > 0`` | ``\tfrac{1}{2\lambda}\lVert y\rVert^2 + \langle c, y\rangle`` |
| ``\langle c, x\rangle + d`` | the indicator of ``\{c\}``, minus ``d`` |

For ``\lambda = 0``, the squared-norm conjugate is the indicator of ``\{0\}``.
Centering the function at ``c`` adds ``\langle c, y\rangle`` to its conjugate;
at ``\lambda = 0`` this leaves the indicator unchanged, since the linear term
vanishes at ``y = 0``.

[`Conjugate`](@ref) knows these values, and those of the other norms and balls,
so `value_and_prox!` reports them. The proximal operator of any convex,
proximable function's conjugate is available. Where the standalone value is
not implemented, `value_and_prox!` computes it at the proximal point from
the Fenchel equality ``f^*(y) = \langle p, y\rangle - f(p)``, with ``p`` the corresponding
proximal point of ``f``. Subtraction can lose accuracy when these terms nearly
cancel, and errors in the inner proximal point also affect the value. For a
conjugate that is an indicator, the result can be a small nonzero number
rather than zero; it is not a feasibility certificate.

```@example calculus
h = Conjugate(NormL2(1.0))                     # the indicator of the unit ℓ2 ball
@assert prox(h, [3.0, 4.0], 0.5) ≈ [0.6, 0.8] # hide
prox(h, [3.0, 4.0], 0.5) ≈ [0.6, 0.8]
```

## Separable functions with constraints

[`Constrained`](@ref)`(f, C)` is ``f + \iota_C`` for a convex separable
``f(x) = \sum_i f_i(x_i)`` and a set ``C`` that is a box, a hyperplane, a
halfspace or a simplex. Its proximal operator reduces to the scalar proximal
operators of ``f``.

**On a box.** The problem that defines the proximal operator separates over the
entries, into the scalar problems
```math
\underset{l_i \le z \le u_i}{\text{minimize}} \quad f_i(z) + \frac{1}{2\gamma}(z - v_i)^2 .
```
The objective is strictly convex, so it decreases up to its unconstrained
minimizer ``\operatorname{prox}_{\gamma f_i}(v_i)`` and increases after it. Its
minimizer on the interval is therefore the point of the interval nearest to that
minimizer:
```math
\operatorname{prox}_{\gamma(f + \iota_C)}(v)_i = \operatorname{clamp}\big(\operatorname{prox}_{\gamma f_i}(v_i),\, l_i,\, u_i\big).
```
The constrained function is again separable, so coordinate descent applies to
it. For a nonconvex ``f_i`` the objective can have several local minimizers, and
the clamp can pick the wrong one, so `Constrained` requires a convex ``f``,
except for [`MCP`](@ref), [`SCAD`](@ref) and [`NormL0`](@ref). Their scalar
objectives are piecewise quadratic, convex or concave between the kinks of
``f_i``, so the minimum on the interval is at one of a few candidates: the ends
of the interval, the kinks inside it, and the clamped vertices of the convex
pieces. When the step is small enough for the objective to be convex everywhere
(``\gamma < a`` for MCP, ``\gamma < a - 1`` for SCAD), the clamp is exact again.
For example, the ℓ0 penalty with ``\lambda = \gamma = 1`` on ``[-1, 0.5]`` at
``v = 2``: hard thresholding keeps ``2``, whose clamp ``0.5`` costs
``1 + 1.5^2/2 = 2.125``, while ``0`` costs ``2^2/2 = 2``.

**On a hyperplane.** For ``C = \{z : \langle a, z\rangle = b\}``, the Lagrangian
```math
f(z) + \frac{1}{2\gamma}\|z - v\|^2 + \nu\,(\langle a, z\rangle - b)
```
also separates over the entries. Completing the square, its minimizer for a
fixed multiplier ``\nu`` is
```math
z_i(\nu) = \operatorname{prox}_{\gamma f_i}(v_i - \gamma\nu a_i).
```
A scalar proximal operator is nondecreasing and 1-Lipschitz, so each
``a_i z_i(\nu)`` is a continuous, nonincreasing function of ``\nu``, and so is
``\varphi(\nu) = \langle a, z(\nu)\rangle - b``. The proximal point is
``z(\nu^\star)`` for a root ``\nu^\star`` of ``\varphi``, and it is the same for
every root. For the halfspace ``\langle a, z\rangle \le b`` the multiplier is
nonnegative and vanishes when the constraint is inactive: ``\nu^\star = 0`` when
``\varphi(0) \le 0``.

`Constrained` finds the root by a search over the scalar ``\nu``, and each step
of the search is one pass over the entries.

- **The start.** A solver calls the proximal operator at points that change
  little from one call to the next, and so does the multiplier. The search
  starts from the multiplier of the last call, which the function keeps.
- **The first step.** It is at least ``|\varphi|/(\gamma\|a\|^2)``, since the
  slope of ``\varphi`` is at least ``-\gamma\|a\|^2``. From the last
  multiplier, it goes to the root of the line through the start with the slope
  of ``\varphi`` there. Where the scalar proximal operators ``P_i`` of ``f``
  are piecewise linear, as for every function below, that slope is exact,
  ``\varphi'(\nu) = -\gamma\sum_i a_i^2 P_i'``, and from a start on the linear
  piece of the root the first step lands on the root. Otherwise the slope is
  that of the last secant. At a first call, from zero, the step is the lower
  bound: computing the slope costs half a pass, and pays only when the root
  lies near zero.
- **The bracket.** Secant steps bracket the root, and regula falsi with the
  Illinois rule shrinks the bracket. For the piecewise linear ``\varphi`` of the
  ℓ1 norm and boxes, the search is exact once the bracket lies on one piece.
- **The cost.** On simplices, capped simplices and the zero-sum lasso with
  ``10^3`` to ``10^5`` entries, the search took 7 to 16 passes from zero, and
  three to five from the last multiplier on slowly changing points.
- **Breakpoint searches.** For the continuous quadratic knapsack problem they
  are exact in ``O(n)`` time (Brucker, 1984; Kiwiel, 2008), but a single
  selection of the median of the breakpoints costs more than the whole search.

A box enters through the scalar proximal operators, so ``f`` may itself be
`Constrained(f, box)`; the simplex is the box ``x \ge 0`` with the hyperplane
``\sum_i x_i = r``. The caller must ensure that the set meets the domain of
``f``. A box clamps the unconstrained proximal point without checking this
intersection; for an indicator, `value_and_prox!` returns zero even when the
intersection is empty. The multiplier search on a hyperplane or halfspace
throws an `ArgumentError` if it cannot bracket a multiplier. It is not a
general feasibility test: rounding can reach the closure of an open domain,
and a finite multiplier need not exist at the boundary of a closed domain.

| Call | Function |
|:-----|:---------|
| `Constrained(NormL1(λ), IndHyperplane(1, 0))` | the lasso penalty on coefficients that sum to zero, as in regression on compositional data |
| `Constrained(NormL1(λ), IndBox(0, u))` | the nonnegative lasso penalty with upper bounds |
| `Constrained(IndBox(0, 1), IndHyperplane(1, k))` | the indicator of the capped simplex ``\{x : 0 \le x \le 1,\ \sum_i x_i = k\}`` |
| `Constrained(SqrNormL2(λ), IndSimplex(1))` | the ridge penalty on the probability simplex |

```@example calculus
v = [0.9, -0.4, 0.3, 1.6]
z = prox(Constrained(NormL1(0.2), IndHyperplane(1, 0)), v, 1.0)   # entries that sum to zero
(z, sum(z))
```

```@example calculus
prox(Constrained(IndBox(0, 1), IndHyperplane(1, 2)), v)            # onto the capped simplex
```

**On an ℓ2 ball or sphere.** For a norm ``f`` on the ball ``\{z : \|z\|_2 \le r\}``,
the proximal operator is the proximal point of ``f`` scaled onto the ball:
```math
\operatorname{prox}_{\gamma(f + \iota_{B_r})}(v) = \min\Big(1, \frac{r}{\|p\|_2}\Big)\, p,
\qquad p = \operatorname{prox}_{\gamma f}(v).
```
A positively homogeneous convex ``f`` has ``\partial f(tp) = \partial f(p)`` for
``t > 0``, so ``v - p \in \gamma\,\partial f(p)`` gives
``v - tp = (v - p) + (1 - t)p``, a subgradient plus a multiple of ``tp``, which
lies in the normal cone of the ball at ``tp = rp/\|p\|``. The same holds for the
indicator of a cone, also a nonconvex one such as the ℓ0 ball or the matrices of
rank at most ``r``. Among the points of the ball on one face of the cone, the
best is the projection there scaled onto the ball, at a distance that decreases
with the norm of that projection, so the projection onto the cone already picks
the face.

On the sphere ``\|z\|_2 = r``, the quadratic term is ``-\langle z, v\rangle`` plus
a constant, so the proximal operator maximizes ``\langle z, v\rangle - \gamma f(z)``.
For ``f = \lambda\|\cdot\|_1`` this is
``\sum_i |z_i|\,(|v_i| - \gamma\lambda)`` for ``z`` with the signs of ``v``, so
the maximizer is the soft thresholding of ``v`` scaled onto the sphere when some
``|v_i| > \gamma\lambda``. Otherwise every coefficient is at most zero, and since
``\sum_i |z_i| \ge \|z\|_2`` the maximizer puts all the mass on the entry where
``|v_i| - \gamma\lambda`` is largest. The group norm works the same way with the
margins ``\|v_G\| - \gamma\lambda w_G``, and the ℓ0 ball keeps the ``s`` largest
entries. These are the steps of the penalized matrix decomposition (Witten,
Tibshirani and Hastie, 2009) and of the truncated power method (Yuan and Zhang,
2013) for sparse principal components.

```@example calculus
u = prox(Constrained(NormL1(0.5), IndSphereL2(1)), [2.0, -0.3, 1.0, 0.2])   # a sparse unit vector
(u, norm(u))
```

**A quadratic on an affine set.** For a [`LeastSquaresLoss`](@ref) or a
[`QuadraticFunction`](@ref) ``f`` with Hessian ``H`` on
``\{z : Az = b\}`` for an ``m \times n`` matrix ``A``, the proximal problem is a
ridge regression with equality constraints. Its optimality conditions
```math
(H + I/\gamma)\, z + A^\top\lambda = c, \qquad Az = b, \qquad c = v/\gamma - \nabla f(0),
```
give, with ``M = (H + I/\gamma)^{-1}``,
```math
z = M(c - A^\top\lambda), \qquad (AMA^\top)\,\lambda = AMc - b .
```
`Constrained(f, IndAffine(A, b))` applies ``M`` through the decomposition that
``f`` keeps for its own proximal operator, the thin SVD ``X = U\Sigma V^\top`` of
the least-squares loss or the eigendecomposition of the quadratic:
``M = V(\Sigma^2 + I/\gamma)^{-1}V^\top + \gamma(I - VV^\top)``. The ``m \times m``
matrix ``AMA^\top`` changes with ``\gamma``, and the function keeps its Cholesky
factor for the last ``\gamma``, which ADMM changes rarely. A call then costs the
unconstrained step plus two products with an ``m \times n`` matrix. With ADMM,
``f`` as this function and ``g = \lambda\|\cdot\|_1`` give the lasso under
equality constraints, such as log-contrast regression with several
subcompositions (Shi, Zhang and Li, 2016).

```@example calculus
X = [1.0 0.0 2.0; 0.5 1.0 0.0; 0.0 1.0 1.0; 1.0 1.0 1.0]
Y = [1.0, 2.0, 0.5, 1.5]
f = Constrained(LeastSquaresLoss(Y, X), IndAffine([1.0 1.0 1.0], [0.0]))   # coefficients that sum to zero
z = prox(f, zeros(3), 10.0)
(z, sum(z))
```

## Functions of the eigenvalues

[`Spectral`](@ref)`(f)` is ``F(X) = f(\lambda(X))`` on the symmetric matrices,
for a function ``f`` of a vector that does not depend on the order of its
entries. Its proximal operator acts on the eigenvalues (Lewis, 1996): for
``X = V\operatorname{diag}(\lambda)V^\top``,
```math
\operatorname{prox}_{\gamma F}(X) = V\operatorname{diag}\big(\operatorname{prox}_{\gamma f}(\lambda)\big)V^\top .
```
For a symmetric ``Z`` with eigenvalues ``\mu``, sorted in the same order as
``\lambda``, the trace inequality ``\operatorname{tr}(ZX) \le \langle \mu, \lambda\rangle``
(von Neumann; Fan, 1949) gives ``\|Z - X\|_F \ge \|\mu - \lambda\|_2``. So the
objective that defines the proximal operator is at least
``f(\mu) + \|\mu - \lambda\|_2^2/(2\gamma)`` at ``Z``. The matrix above, with the
eigenvectors of ``X``, attains the minimum of that bound over ``\mu``. ``F`` is
convex exactly when ``f`` is.

| ``f`` | `Spectral(f)` |
|:------|:--------------|
| `IndNonnegative()` | the positive semidefinite cone, [`IndPSD`](@ref)`()` |
| `NormL1Nonneg(λ)` | ``\lambda\operatorname{tr}(X)`` on the cone, [`TracePSD`](@ref)`(λ)` |
| `NormL1(λ)` | ``\lambda\|X\|_*``, the trace norm |
| `NormL2(λ)` | ``\lambda\|X\|_F`` |
| `NormLinf(λ)` | ``\lambda\|X\|_2``, the spectral norm |
| `IndBox(l, u)` | ``lI \preceq X \preceq uI`` |
| `IndBallL1(r)` | ``\|X\|_* \le r`` |
| `IndSimplex(1)` | the density matrices: ``X \succeq 0`` and ``\operatorname{tr}(X) = 1`` |
| `Constrained(IndBox(0, 1), IndHyperplane(1, k))` | the Fantope: ``0 \preceq X \preceq I`` and ``\operatorname{tr}(X) = k`` |
| `Conjugate(Constrained(IndBox(0, 1), IndHyperplane(1, k)))` | ``\lambda_1(X) + \dots + \lambda_k(X)``, the sum of the ``k`` largest eigenvalues |
| `NormL0(λ)` | ``\lambda\operatorname{rank}(X)``, nonconvex |
| `IndBallL0(r)` | the matrices of rank at most ``r``, nonconvex |

The Fantope ``\mathcal{F}^k`` is the convex hull of the projections onto
``k``-dimensional subspaces (Fan, 1949; Overton and Womersley, 1992), and its
projection projects the eigenvalues onto the capped simplex. Vu, Cho, Lei and
Rohe (2013) estimate a sparse principal subspace by maximizing
``\langle S, X\rangle - \lambda\|X\|_1`` over ``\mathcal{F}^k``, and the example on
sparse principal component analysis does this by ADMM. A function with weights,
bounds or normals that differ between entries depends on their order, and
`Spectral` rejects it.

```@example calculus
B = randn(5, 5)
P = prox(Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, 2))), (B + B') / 2)   # onto the Fantope
(eigvals(Symmetric(P)), tr(P))
```

## Smoothing

[`MoreauEnvelope`](@ref)`(f, μ)` replaces a nonsmooth convex ``f`` by a smooth
function with a ``1/\mu``-Lipschitz gradient, from the same proximal operator:
the envelope of ``\delta|\cdot|`` is the Huber function, that of an indicator
function the squared distance to the set divided by ``2\mu``. For the losses of
a linear predictor, `ProximalBase.Smoothed(φ, μ)` is the envelope of a scalar
function of the residual as a scalar loss of [`GLMLoss`](@ref), with curvature
at most ``1/\mu``, so that coordinate descent and proximal Newton apply to a
smoothed check loss (quantile regression) or hinge.

```@example calculus
const PB = ProximalBase
@assert PB.Smoothed(NormL1(0.5), 1.0)(1.3, 0.2) ≈ PB.Huber(0.5)(1.3, 0.2) # hide
PB.Smoothed(NormL1(0.5), 1.0)(1.3, 0.2) ≈ PB.Huber(0.5)(1.3, 0.2)
```

