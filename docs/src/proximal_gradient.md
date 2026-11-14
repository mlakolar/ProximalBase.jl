# Proximal gradient methods

*Assumes: gradient descent, Lipschitz gradients and a 1/L step size · You will learn: why a proximal step extends projection, how to test optimality and how to reuse buffers.*

Projected gradient descent first moves along ``-\nabla f(x)`` and then projects
the trial point onto a feasible set. Replace that projection by a
[proximal operator](glossary.md#Proximal-operator) for a
[proximable](glossary.md#Proximable) function ``g``, and the same pattern gives

```math
x^+=\operatorname{prox}_{\gamma g}(x-\gamma\nabla f(x)).
```

If ``\nabla f`` is L-Lipschitz, then
``f(z)\le f(x)+\langle\nabla f(x),z-x\rangle+L\|z-x\|^2/2``.
For ``0<\gamma\le1/L``, minimizing the corresponding quadratic model plus
``g`` gives this proximal step. With convex ``f``, [proper closed convex](glossary.md#Proper-closed-convex-function) ``g``
and a minimizer, it gives descent and the usual O(1/k) objective-error bound
``F(x^k)-F(x^\star)\le\|x^0-x^\star\|^2/(2\gamma k)``.
Acceleration requires its own schedule and assumptions.

## Gradient mapping and optimality

The [gradient mapping](glossary.md#Gradient-mapping) generalizes the
usual gradient as a measure of first-order stationarity in a proximal-gradient
method. It is the difference between the starting point and one full
proximal-gradient update, divided by the step size:


```math
G_\gamma(x)=\frac{x-\operatorname{prox}_{\gamma g}(x-\gamma\nabla f(x))}{\gamma}.
```

For convex ``g``, ``G_\gamma(x)=0`` is equivalent to the
[subgradient condition](glossary.md#Subgradient-and-subdifferential)
``0\in\nabla f(x)+\partial g(x)``; with convex ``f`` it characterizes global
optimality. A small coefficient change alone is not a complete convergence
test. ProximalBase does not provide a gradient-mapping helper; calculate this
expression in caller code.

```@example proximal_gradient_intro
using ProximalBase, LinearAlgebra
f = SqrNormL2(1.0)
g = NormL1(0.5)
x = [2.0,-1]
γ = 0.5
next = prox(g,x-γ*gradient(f,x),γ)
G = (x-next)/γ
@assert next == [0.75,-0.25]
@assert f(next)+g(next) <= f(x)+g(x)
(next=next, mapping=G)
```

## Buffer reuse and step selection

`gradient!(grad,f,x)` requires independent gradient scratch; `prox!(x,g,x,γ)`
then permits an exact in-place threshold. The full
[iteration example](examples/proximal_gradient.md) checks monotonic objective,
explicit KKT conditions and the closed-form lasso solution for orthogonal
columns. The [getting-started tutorial](getting_started.md) uses a general
design and verifies one descent step.

`lipschitz` returns a bound for supported smooth functions; with a zero bound,
choose a positive step explicitly. Dense least squares uses the full squared
spectral norm. Other matrix formats use a safe one/infinity bound, which can
be conservative. `opnorm2` is a deterministic numerical estimate and can miss
a leading eigenspace; use an independent bound or a line search if relying
on an estimate. A vector of coordinate curvatures does not automatically
majorize the full Hessian for simultaneous updates.

Storage is separate from the mathematical step. A
[working set](glossary.md#Working-set) controls which coordinates a solver visits. A full proximal write can
leave zeros active in an iterate. Whether to compact them after a step is
the solver's working-set policy, not a proximal optimality condition.
