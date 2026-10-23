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
PB.Smoothed(NormL1(0.5), 1.0)(1.3, 0.2) ≈ PB.Huber(0.5)(1.3, 0.2)
```

