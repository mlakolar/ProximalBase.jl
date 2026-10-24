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

```@example calculus
using ProximalBase, LinearAlgebra
x, c, γ = [2.0, -1.0, 0.5], [0.3, 0.3, -0.3], 0.7
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
| ``\tfrac{\lambda}{2}\lVert x - c\rVert^2`` | ``\tfrac{1}{2\lambda}\lVert y\rVert^2 + \langle c, y\rangle`` |
| ``\langle c, x\rangle + d`` | the indicator of ``\{c\}``, minus ``d`` |

[`Conjugate`](@ref) knows these values, and those of the other norms and balls,
so `value_and_prox!` reports them. The proximal operator of any convex,
proximable function's conjugate is available, even where the value is not.

```@example calculus
h = Conjugate(NormL2(1.0))                     # the indicator of the unit ℓ2 ball
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
the clamp can pick the wrong one, so `Constrained` requires a convex ``f``.

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
of the search is one pass over the entries. The slope of ``\varphi`` is at least
``-\gamma\|a\|^2``, so the first step, ``|\varphi(0)|/(\gamma\|a\|^2)``, does not
pass the root. Secant steps then bracket the root, and regula falsi with the
Illinois rule shrinks the bracket. For the ℓ1 norm and boxes ``\varphi`` is
piecewise linear, and the search is exact once the bracket lies on one piece.
On simplices, capped simplices and the zero-sum lasso, with ``10^3`` to ``10^5``
entries, it took 6 to 17 passes. A box enters through the scalar proximal
operators, so ``f`` may itself be `Constrained(f, box)`; the simplex is the box
``x \ge 0`` with the hyperplane ``\sum_i x_i = r``.

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
