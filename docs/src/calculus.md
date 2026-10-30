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
| [`PointwiseMinimum`](@ref)`(f1, f2, ...)` | ``\min_k f_k(x)`` | the candidate with the smallest proximal objective | |
| [`Constrained`](@ref)`(f, C)` | ``f(x)`` on a box, hyperplane, halfspace or simplex ``C`` | on a box, ``\operatorname{clamp}(\operatorname{prox}_{\gamma f}(x), l, u)``; on a hyperplane, ``\operatorname{prox}_{\gamma f}(x - \gamma\nu a)`` | |
| [`Spectral`](@ref)`(f)` | ``f(\lambda(X))`` for a symmetric ``X`` | ``V\operatorname{diag}(\operatorname{prox}_{\gamma f}(\lambda))V^\top`` | |
| [`MoreauEnvelope`](@ref)`(f, μ)` | ``\min_z f(z) + \tfrac{1}{2\mu}\lVert z - x\rVert^2`` | ``x + \tfrac{\gamma}{\mu + \gamma}\big(\operatorname{prox}_{(\mu+\gamma) f}(x) - x\big)`` | ``\big(x - \operatorname{prox}_{\mu f}(x)\big)/\mu`` |
| [`SqrDistL2`](@ref)`(C, λ)` | ``\tfrac{\lambda}{2}\operatorname{dist}(x,C)^2`` | ``(x + \gamma\lambda P_C(x))/(1 + \gamma\lambda)`` | ``\lambda(x - P_C(x))`` where the projection is unique |
| [`DistL2`](@ref)`(C, λ)` | ``\lambda\operatorname{dist}(x,C)`` | ``p`` if ``d \le \gamma\lambda``, otherwise ``x + (\gamma\lambda/d)(p-x)``, with ``p=P_C(x)``, ``d=\|x-p\|_2`` | |
| [`Precompose`](@ref)`(f, A, μ)`, ``AA^\top = \mu I`` | ``f(Ax)`` | ``x + \tfrac1\mu A^\top\big(\operatorname{prox}_{\mu\gamma f}(Ax) - Ax\big)`` | ``A^\top\nabla f(Ax)`` |
| [`Precompose`](@ref)`(f, a)`, array `a`, separable `f` | ``f(a \odot x)`` | ``\operatorname{prox}_{\gamma a_i^2 f_i}(a_i x_i)/a_i`` in entry ``i`` | ``a \odot \nabla f(a \odot x)`` |
| [`LinearImage`](@ref)`(f, L)` | ``\inf_{Lx=y} f(x)`` | ``L\operatorname*{argmin}_x\{f(x) + \lVert Lx-y\rVert^2/(2\gamma)\}`` in the supported cases | |
| [`Regularize`](@ref)`(f, ρ, c)` | ``f(x) + \tfrac{\rho}{2}\lVert x - c\rVert^2`` | ``\operatorname{prox}_{\frac{\gamma}{1+\gamma\rho} f}\big(\tfrac{x + \gamma\rho c}{1 + \gamma\rho}\big)`` | ``\nabla f(x) + \rho(x - c)`` |

The allocating forms `prox`, `gradient`, `value_and_gradient` and `hessian_vec`
of a `SeparableSum` act on tuples of blocks, as do `Scaled` and `Conjugate` of
a `SeparableSum`, for the oracles each provides.

## Derivations

Each rule follows from substituting into the definition of the proximal
operator.

- **Scaling.** ``\gamma(\alpha f) = (\gamma\alpha) f``, so the proximal operator
  of ``\alpha f`` with step ``\gamma`` is that of ``f`` with step
  ``\gamma\alpha``. The Lipschitz constant scales by ``\alpha``.
  Zero scaling removes finite penalty terms and keeps domain constraints.
  For `NormL1`, `NormL1Nonneg`, `ElasticNet`, `CappedL1`, `LogPenalty`, `NormLq`,
  `Berhu`, `GroupNormL2`, `GroupNormLinf`, `SparseGroupNormL2`, `GroupMCP` and
  `GroupSCAD`, infinite weights retain the same family with all levels zero.
  With finite weights, `NormL1Nonneg` becomes `IndNonnegative()` and the others
  become `ZeroFunction()`. Non-finite levels or shape parameters are rejected.
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

## Entrywise scaling

[`Precompose`](@ref)`(f, a)` with a real array `a` represents
``h(x) = f(a \odot x)``. This is entrywise scaling even for a matrix `a`;
use the three-argument form for a linear map. The scale must have the same
size as `x`, with finite, nonzero entries. It is stored by reference, so
construct a new function after changing it.

For a separable ``f``, substituting ``u = a_i z_i`` in each scalar proximal
problem gives
```math
\operatorname{prox}_{\gamma h}(x)_i
= \frac{\operatorname{prox}_{\gamma a_i^2 f_i}(a_i x_i)}{a_i}.
```
The wrapper supports diagonal steps, coordinate updates, optimality tests
and restriction. Its coordinate curvatures are ``a_i^2 L_i``, and its gradient
has Lipschitz constant at most ``\max_i a_i^2 L_f``. Entrywise values require
`f` to be a `SeparableFunction`. On a `SymmetricSparseIterate`, both `a` and
the inner parameters must be symmetric. The image uses the promoted floating
type of `x` and `a`; values are computed at that image and have the
argument's precision, as in [Precision](@ref).
The scaled inputs, effective steps and results must fit
in that precision and in the range supported by the inner oracle; effective
steps are formed without first rounding ``a_i^2``.

For a group penalty, `a` must be constant on each group. Full-array values
and proximal operators then use the step ``\gamma c_k^2`` for group ``k``,
where ``c_k`` is its scale. Indices outside the groups remain unchanged.
The four-argument `kkt_violation` works on arrays and on matching groups of an
`AtomIterate`; group coordinate updates and `restrict` are not supported.
Other nonseparable functions are rejected, and the array wrapper supplies no
Hessian products.

Penalizing standardized regression coefficients can be done without
rescaling the design: with column scales `s`, use `Precompose(f, s)` on
the coefficients. For the ℓ1 norm this gives weights `s`. For MCP it changes
both the level and the concavity, entry by entry:
```math
\operatorname{MCP}_{\lambda, b}(s\beta)
= \operatorname{MCP}_{s\lambda, b/s^2}(\beta), \qquad s > 0.
```
Similarly, `Precompose(IndBox(l, u), s)` imposes a box on the standardized
coefficients.

Known residual scales also enter this way. The separable Huber function
`MoreauEnvelope(NormL1(1), δ)` has quadratic part ``t^2/(2\delta)`` for
``|t| \le \delta`` and linear part ``|t| - \delta/2`` otherwise.
Thus `Precompose(MoreauEnvelope(NormL1(1), δ), 1 ./ σ)` gives
``\sum_i \rho(r_i/\sigma_i)``. Both the quadratic curvature and the transition
point change with ``\sigma_i``.

```@example calculus
s = [0.5, 2.0, 1.5]
h = Precompose(MCP(0.3), s)
prox(h, [2.0, -1.0, 0.2], 0.4)
```

## Pointwise minima and unions

[`PointwiseMinimum`](@ref)`(f1, f2, ...)` takes one or more proximable functions
on arrays. If ``z_i = \operatorname{prox}_{\gamma f_i}(x)``, then
```math
\min_z\left\{\min_i f_i(z) + \frac{\|z-x\|^2}{2\gamma}\right\}
= \min_i\left\{f_i(z_i) + \frac{\|z_i-x\|^2}{2\gamma}\right\}.
```
The two minima commute, so the proximal operator evaluates every candidate
and keeps the one with the smallest objective. Ties go to the first function.
A `NaN` score never beats a number; if all scores are `NaN`, the first
candidate is kept and `value_and_prox!` returns `NaN`. Comparisons preserve the
range of the distance term even when its square overflows or underflows.
The minimum is generally nonconvex and has no smooth or separable oracles.

For indicators, the minimum is the indicator of the union, and its proximal
operator is the nearest of the individual projections. Isotonic regression
with an unknown direction is the union of the two monotone cones:

```@example calculus
monotone = PointwiseMinimum(IndIsotonic(), IndIsotonic(decreasing = true))
prox(monotone, [3.0, 1.0, 2.0, 0.0])
```

A block of effects whose sign is shared but unknown lies in the union of the
nonnegative and nonpositive orthants:

```@example calculus
shared_sign = PointwiseMinimum(IndNonnegative(), IndBox(-Inf, 0))
prox(shared_sign, [2.0, -0.5, 1.0])
```

For several prior guesses ``\beta_k``, the penalty
``\min_k \lambda\|\beta-\beta_k\|_1`` shrinks towards the best source model:

```@example calculus
prior = PointwiseMinimum(Translate(NormL1(0.5), [0.0, 1.0]),
                         Translate(NormL1(0.5), [2.0, 0.0]))
prox(prior, [1.5, 0.2], 1.0)
```

The cost is one proximal operator per component. For unions with many sets,
such as unimodal regression with an unknown mode, a dedicated algorithm can
be preferable (Stout, *Unimodal regression via prefix isotonic regression*,
2008). Two buffers follow the output's shape and element type; the output may
alias the input. Warmed calls on fixed-shape CPU arrays allocate nothing when
the components allocate nothing. Use a separate function object per thread.
Steps must be positive scalars; tuple inputs and arrays of steps are unsupported.

The standalone value `g(x)` evaluates every component's value in order, and
propagates `NaN` as `minimum` does. Every component must provide a value oracle;
`Conjugate(ElasticNet(0.5, 0.2))`, for example, has no standalone value. The proximal
operator instead obtains each candidate value from `value_and_prox!`, which
has the output's precision, and compares these values. It is exact when
every component's proximal operator and reported value are exact. Approximate
projections, partial decompositions and cancellation in `Conjugate` values can
change the selected branch. In particular, the package's indicators return
zero from `value_and_prox!` without checking membership. This is correct for
exact projections, but an inexact `IndIntersection` projection can win a
comparison it should lose. The zero returned for a union is not a feasibility
certificate.

## Images under linear maps

[`LinearImage`](@ref) builds the image, or infimal postcomposition,
``g(y) = \inf\{f(x) : Lx = y\}``. It is convex when ``f`` is convex and
has value ``+\infty`` outside the range of ``L``. Its proximal point is
``Lx^\star``, where

```math
x^\star \in \operatorname*{argmin}_x\; f(x) + \frac{\lVert Lx-y\rVert^2}{2\gamma}.
```

Two cases are supported. With ``L^\top L = \mu I`` and a convex proximable
``f``, use `LinearImage(f, L, μ)`. Then
``x^\star = \operatorname{prox}_{(\gamma/\mu) f}(L^\top y/\mu)``. A matrix is
checked unless `check = false`; an operator needs `mul!` and its adjoint and
is trusted to satisfy the identity. This gives a penalty on coefficients in
an orthogonal basis together with the constraint that the signal lies in
the basis's span. The standalone value needs the inner value oracle;
`value_and_prox!` also works with a proximal-only inner function.

With `f = QuadraticFunction(Q, q, c)`, use `LinearImage(f, L)` for a matrix
`L`. The matrix ``Q`` must be symmetric positive semidefinite and positive
definite on ``\operatorname{null}(L)``. The proximal point follows from

```math
(\gamma Q + L^\top L)x^\star = L^\top y - \gamma q.
```

A Cholesky factorization is cached for the last step. Dense solves allocate
nothing after warming the buffers and factorization. Sparse data and the
structured band matrices `Diagonal`, `Bidiagonal`, `Tridiagonal` and
`SymTridiagonal` use sparse factorizations; these solves allocate and compute
in `Float64` on Julia 1.10. Other matrix types are converted to dense matrices
for factorization. Very small or large steps can make the system numerically
singular or ill-conditioned. Squared Cholesky pivots no greater than
``64n\operatorname{eps}(T)\max_j|G_{jj}|`` for the assembled system
``G = \gamma Q + L^\top L`` are rejected as numerically singular. The factor
64 allows for rounding in both the matrix products and factorization.
The standalone quadratic value
uses a cached SVD of `L`, with the rank tolerance of [`IndAffine`](@ref), and
minimizes on its null space. Tall maps use a thin left factor; only wide maps
need a full decomposition to retain the complete right null space. Its reduced
factor uses the same pivot test with the original dimension of `Q` and the
larger maximum absolute diagonal of `Q` and ``N^\top QN``, where ``N`` is
the null-space basis. Retaining the scale of `Q` prevents projection rounding
from making an exact null direction appear positive definite.
This path densifies sparse data and accepts dependent
rows and rank-zero maps. Range membership uses a scaled residual and the tolerance
``\sqrt{\operatorname{eps}(T)}\max(1,\lVert y\rVert)``.

Quadratic buffers use the promoted floating type of `Q` and `L`.
Orthogonal-image buffers use the promoted floating type of `L` and the
argument. Scalar values have the argument's precision, as in
[Precision](@ref). When the transformed step falls outside its
normal range, the orthogonal proximal computation widens its inner workspace;
exceptional `BigFloat` computations may allocate. Powers of two carry extreme
scales without first narrowing them to the workspace type. Outputs keep their
element type. The function owns its buffers and is not safe to share between
threads. Data are stored
by reference: call [`reset_cache!`](@ref) on the outermost wrapper after
changing them, including before a standalone quadratic value call.

Three applications illustrate the distinction from [`Precompose`](@ref).

- **Penalties on fitted values.** Let ``x`` describe a function on a fine
  grid, let ``D`` take second differences, and let ``L`` evaluate the function
  at the design points. With ``Q = D^\top D``, the image is the least
  roughness among grid functions with fitted values ``y``. Observations must
  identify the unpenalized affine null space. Combining this image with a
  quantile, Huber or Poisson loss penalizes roughness without forming a dense
  precision on the fitted values. For ``Q \succ 0``, the Gaussian prior
  interpretation is proper and the induced covariance is
  ``LQ^{-1}L^\top``; singular roughness corresponds to an intrinsic prior
  with an unpenalized null space.
- **Profiling nuisance parameters.** With ``L = [I\;0]`` selecting ``\theta``
  from ``(\theta,\eta)``, the image is ``\min_\eta f(\theta,\eta)``. For
  ``f(\theta,\eta) = \lVert r-X\theta-Z\eta\rVert^2/2`` and full-column-rank
  ``Z``, it equals ``\lVert M_Z(r-X\theta)\rVert^2/2``, where ``M_Z`` projects
  orthogonally off the columns of ``Z``. This is the Frisch–Waugh–Lovell
  residualization identity. Convert the least-squares loss to its equivalent
  `QuadraticFunction` for this constructor.
- **A truncated basis.** With orthogonal columns of equal norm, `LinearImage`
  of `NormL1` penalizes the basis coefficients and constrains the signal to
  their span. Outside the span its value is `Inf`.

```@example calculus
L = [1.0 0.0 0.0; 0.0 1.0 0.0]
f = QuadraticFunction([2.0 0.0 1.0; 0.0 2.0 1.0; 1.0 1.0 2.0])
g = LinearImage(f, L)
@assert prox(g, [1.0, -1.0], 0.5) ≈ [0.5, -0.5] # hide
prox(g, [1.0, -1.0], 0.5)
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
rather than zero; it is not a feasibility certificate. For a positively
homogeneous ``f`` or a cone, ``\gamma\operatorname{prox}_{f/\gamma}(x/\gamma) =
\operatorname{prox}_f(x)``, and for the ℓ2 ball and box indicators, their
translations and positive rescalings, it is the projection onto ``\gamma C``,
so these conjugates take every step without dividing by ``\gamma``, with the
accuracy of a unit step, about ``\epsilon`` times the largest input entry
(with array steps and in `prox_entry`, only for the conjugates that are
indicators of separable functions, the box itself and the double conjugate); for
the others, `prox!` and `value_and_prox!` throw an `ArgumentError` when a finite
input gives a result that is not finite in working precision.

```@example calculus
h = Conjugate(NormL2(1.0))                     # the indicator of the unit ℓ2 ball
@assert prox(h, [3.0, 4.0], 0.5) ≈ [0.6, 0.8] # hide
prox(h, [3.0, 4.0], 0.5) ≈ [0.6, 0.8]
```

## Separable functions with constraints

[`Constrained`](@ref)`(f, C)` is ``f + \iota_C`` for a convex separable
``f(x) = \sum_i f_i(x_i)`` and a set ``C`` that is a box, a hyperplane, a
halfspace, a hyperslab or a simplex. Its proximal operator reduces to the scalar
proximal operators of ``f``.

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

For a hyperslab [`IndHyperslab`](@ref)`(a, l, u)`, the multiplier vanishes
when the unconstrained proximal point satisfies both bounds. Otherwise the
search meets the violated bound, with a negative multiplier at the lower bound
or a positive one at the upper bound. The selected bound stays fixed throughout
that call. A warm start from the other side, or from an inactive call, is
discarded. This also applies when `f` is itself constrained to a box.

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
intersection is empty. The multiplier search on a hyperplane, halfspace or
hyperslab throws an `ArgumentError` if it cannot bracket a multiplier. It is not a
general feasibility test: rounding can reach the closure of an open domain,
and a finite multiplier need not exist at the boundary of a closed domain.

| Call | Function |
|:-----|:---------|
| `Constrained(NormL1(λ), IndHyperplane(1, 0))` | the lasso penalty on coefficients that sum to zero, as in regression on compositional data |
| `Constrained(NormL1(λ), IndHyperslab(a, l, u))` | the lasso penalty with a bounded contrast ``l \le \langle a, x\rangle \le u`` |
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

## Distances to sets

[`DistL2`](@ref)`(C, λ)` is ``\lambda\operatorname{dist}(x,C)`` for finite
``\lambda \ge 0``. With one projection ``p \in P_C(x)`` and
``d = \|x-p\|_2``, its proximal operator is
```math
\operatorname{prox}_{\gamma f}(x) =
\begin{cases}
p, & d \le \gamma\lambda,\\
x + (\gamma\lambda/d)(p-x), & d > \gamma\lambda.
\end{cases}
```
The value in exact arithmetic is ``\lambda(d-\gamma\lambda)_+``. The
implementation measures the distance from the written output to the retained
projection, so its returned value accounts for rounding, with no second
projection. Distances and steps are scaled before comparison and interpolation
to preserve finite results even when ``d`` or ``\gamma\lambda`` exceeds the
floating-point range.

This penalty moves an estimate toward a structured submodel and can select it
exactly, to the projection's accuracy. For ``C=\{0\}`` it is
[`NormL2`](@ref)`(λ)`. For ``C=\operatorname{span}\{\mathbf 1\}`` it is
``\lambda\|x-\bar x\mathbf 1\|_2``, which fuses all coefficients at once.
[`IndIsotonic`](@ref) shrinks toward a monotone pattern, and
[`IndNonnegative`](@ref) toward the nonnegative orthant. For a convex set and
``\lambda>0``, the distance is nonsmooth on the boundary of ``C`` and zero,
hence smooth, in its interior. The function is not separable, even when the
set is. Its `issmooth` trait is false, no gradient oracles are provided, and
`lipschitz` throws because that interface concerns the gradient's constant.

For an ``L``-Lipschitz loss with a constrained minimizer on ``C``,
``\lambda>L`` puts every minimizer of the penalized problem in ``C``. Indeed,
projecting an infeasible ``x`` reduces its penalty by ``\lambda d`` and
increases its loss by at most ``Ld``. Equality ``\lambda=L`` need not suffice:
with ``C=\{0\}`` and loss ``-Lx``, every ``x\ge0`` minimizes the penalty at
equality. The squared distance below generally only pulls toward the set.

The proximal formula also holds for a nonconvex nonempty closed set with an
exact projection; `isconvex` follows the set. For fixed ``c\in C``, minimizing
``\lambda\|z-c\|_2+\|z-x\|_2^2/(2\gamma)`` over ``z`` gives the reduced
objective ``\phi(\|x-c\|_2)``, where ``\phi(r)=r^2/(2\gamma)`` for
``r\le\gamma\lambda`` and ``\phi(r)=\lambda r-\gamma\lambda^2/2`` otherwise
(Beck, *First-Order Methods in Optimization*, 2017, Chapter 6). This is
nondecreasing in ``r``, so any nearest point minimizes it. The projection
remains nearest along the segment from ``x`` to ``p`` by the triangle
inequality. At ties, such as sparse supports or the sphere at zero, the oracle
uses the projection selected by the set.

At zero scale the value vanishes and the proximal operator is the identity,
after structural checks, without a projection. Only scalar steps are
supported. The function owns its projection buffer and must not be shared
between threads. The buffer follows the argument's precision for values and
the output's for proximal operators; the output may alias the input.
`Conjugate(DistL2(C, λ))` provides a proximal operator for convex ``C``
through Moreau's identity, without a standalone conjugate value.
`Spectral(DistL2(...))`, `Constrained(DistL2(...), box)`, scalar proximal
operators, restriction and coordinate curvatures are unsupported. The
projection qualifications stated below for `SqrDistL2` apply to `DistL2` too.

```@example calculus
d = DistL2(IndNonnegative(), 2.0)
x = [-3.0, 1.0, -4.0]
@assert d(x) == 10 # hide
@assert prox(d, x, 3.0) == [0.0, 1.0, 0.0] # hide
(d(x), prox(d, x, 1.0), prox(d, x, 3.0))
```

[`SqrDistL2`](@ref)`(C, λ)` is ``(\lambda/2)\operatorname{dist}(x,C)^2``
for finite ``\lambda \ge 0`` and a proximable indicator function ``C``. It
uses one projection ``p \in P_C(x)`` for its value, gradient or proximal
operator. For a convex set and ``\lambda > 0``, it equals
[`MoreauEnvelope`](@ref)`(C, 1/λ)` and has a ``\lambda``-Lipschitz gradient.
It also supports nonconvex sets such as [`IndBallL0`](@ref),
[`IndBallGroupL0`](@ref), [`IndBallRank`](@ref) and [`IndSphereL2`](@ref).
For these sets, `issmooth` is false and `lipschitz` throws: the function can
fail to be differentiable at points with several nearest projections. The
gradient oracles still return ``\lambda(x - p)``. This is the gradient where
the projection is unique, almost everywhere, and the residual of the chosen
projection at ties.

The proximal formula holds for every nonempty closed set with an exact
projection. Minimizing
``(\lambda/2)\|z-c\|^2 + \|z-x\|^2/(2\gamma)`` first over ``z`` gives
``z = (x + \gamma\lambda c)/(1 + \gamma\lambda)`` and the reduced objective
``\lambda\|x-c\|^2/(2(1 + \gamma\lambda))``. Minimizing this over ``c \in C``
selects a nearest point ``p``. It remains nearest along the segment from
``x`` to ``p``: a closer point at an intermediate point would, by the
triangle inequality, also be closer to ``x``. Thus `value_and_prox!` uses the
distance from its returned point to the retained projection, with no second
projection.

At zero scale, the value and gradient vanish and the proximal operator is the
identity, after structural checks, with no projection. Separable sets also
provide scalar proximal operators, KKT tests and, when convex, coordinate
curvatures ``\lambda``. `restrict` is supported only for separable sets.
Only scalar steps are supported, and `Spectral(SqrDistL2(...))` is not
supported. The projection buffer belongs to the function, so do not share it
between threads. It follows the argument's precision for values and gradients,
and the output's precision for proximal operators. The proximal output may
alias the input; the gradient output must not.

The set must be nonempty and closed, and its proximal operator an exact
projection in the argument space. An inexact [`IndIntersection`](@ref),
partial spectral decomposition or truncated [`IndAffine`](@ref) system gives
approximate distances and oracles. Nonemptiness of a [`Constrained`](@ref)
set is the caller's responsibility. For [`Spectral`](@ref) sets, supply
symmetric matrices: they read one triangle and the distance is within the
symmetric matrices.

Distance penalties allow sparsity, rank and shape constraints to enter a
smooth objective through their projection residuals. Increasing ``\lambda``
in a sequence of problems is the distance-to-set penalty method of Xu, Chi
and Lange (2017) and the proximal distance method of Keys, Zhou and Lange
(2019), following distance majorization (Chi, Zhou and Lange, 2014).

```@example calculus
d = SqrDistL2(IndBallL0(1), 2.0)
x = [3.0, -1.0, 0.5]
@assert d(x) ≈ 1.25 # hide
(d(x), gradient(d, x), prox(d, x, 0.5))
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
`prox_entry`, `kkt_violation` and `restrict` of a `MoreauEnvelope` are supported
only for a separable function.

```@example calculus
const PB = ProximalBase
@assert PB.Smoothed(NormL1(0.5), 1.0)(1.3, 0.2) ≈ PB.Huber(0.5)(1.3, 0.2) # hide
PB.Smoothed(NormL1(0.5), 1.0)(1.3, 0.2) ≈ PB.Huber(0.5)(1.3, 0.2)
```

