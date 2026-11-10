
# Proximal gradient methods

The algorithm combines a gradient step on a smooth function `f` with the
proximal operator of a penalty or indicator `g`. ProximalBase provides
the **operations**, not an automatic iteration scheme:
[`gradient!`](@ref) evaluates the gradient, [`prox!`](@ref) applies
the proximal operator, and [`gradient_mapping!`](@ref) provides an
optimality measure. See [Getting started](getting_started.md) for a
complete, executable lasso solver and
[Operations on functions](operations.md) for signatures and caveats.

## Proximal gradient descent and the gradient mapping

For ``F = f + g`` with ``f`` smooth, the proximal gradient step is

```math
x^+ = \operatorname{prox}_{\gamma g}\big(x - \gamma \nabla f(x)\big),
```

and the *gradient mapping* is ``G_\gamma(x) = (x - x^+)/\gamma``, so that
``x^+ = x - \gamma G_\gamma(x)``. When ``g = 0`` it is the gradient. For convex
``f`` and ``g``, the optimality condition ``0 \in \nabla f(x) + \partial g(x)``
is equivalent, for any ``\gamma > 0``, to

```math
x = \operatorname{prox}_{\gamma g}\big(x - \gamma\nabla f(x)\big)
\iff G_\gamma(x) = 0 .
```

Therefore, ``\|G_\gamma(x)\|`` is a stopping rule that needs neither the optimal value nor
the form of ``g``. [`gradient_mapping`](@ref) computes it, and
[`gradient_mapping!`](@ref) does so without allocating.

**Step sizes.** Suppose that ``f`` is convex with an ``L``-Lipschitz gradient,
``g`` is proper, closed and convex, and ``F`` has a minimizer ``x^\star``.
For an initial point with finite objective and a fixed ``0<\gamma\le 1/L``,
the descent lemma and proximal optimality give

```math
F(x^+) \le F(x) - \frac{\gamma}{2}\|G_\gamma(x)\|^2 ,
```

so the objective does not increase. In particular, after ``k\ge1`` proximal
gradient steps,

```math
F(x^k)-F(x^\star)\le\frac{\|x^0-x^\star\|^2}{2\gamma k}.
```

Under the same convexity and smoothness conditions, the accelerated procedure
FISTA has objective error ``O(1/k^2)``. These global objective guarantees do
not extend to arbitrary nonconvex penalties. [`lipschitz`](@ref) returns
``L`` for the smooth functions. For
functions whose gradient is not globally Lipschitz, such as the Gaussian
log-likelihood and the Poisson loss, it returns ``\infty``, and a solver needs
a line search.

## A single step in Julia

```@example proximal_gradient_intro
using ProximalBase

f = SqrNormL2(1.0)
g = NormL1(0.5)
x = [2.0, -1.0]
γ = 0.5

x_next = prox(g, x - γ * gradient(f, x), γ)
G = gradient_mapping(f, g, x, γ)
@assert x_next ≈ x - γ * G
x_next
```

The solver is responsible for the step size and stopping rule; the
function objects provide the gradient and proximal map. For diagonal
metrics and per-entry steps, read [The interface](interface.md).
