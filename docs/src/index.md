# ProximalBase.jl

*Assumes: gradients, convexity, and basic Julia syntax · You will learn: the function operations provided by ProximalBase and where to begin.*

**Mathematical function objects and working-set storage for optimization in Julia.**

Many statistical estimation problems take the form

```math
\min_{x\in\mathbb R^p}F(x)=f(x)+g(x),
```

where ``f`` is a smooth data-fitting loss and ``g`` may be a regularizer.
For example, least squares measures prediction error and an ℓ₁ regularizer
can encourage exactly zero coefficients.

In projected gradient descent, we take a gradient step and project onto a
convex constraint set. A [proximal operator](glossary.md#Proximal-operator)
generalizes the projection: when ``g`` is the mathematical indicator of a
nonempty closed convex set ``C`` (zero inside ``C``, infinite outside),
its proximal operator is projection onto ``C``. For other functions, such as
an ℓ₁ regularizer, the proximal operator can shrink values rather than just
project onto a set. ProximalBase supplies the following operations:

```math
\nabla f(x),\qquad
\operatorname{prox}_{\gamma g}(v)
=\operatorname*{argmin}_z
\left\{g(z)+\frac{\|z-v\|^2}{2\gamma}\right\},\quad\gamma>0.
```

Here ``\gamma`` is a positive step size. For
``g(x)=0.5\|x\|_1``, its proximal map at step ``\gamma=1``
is [soft thresholding](glossary.md#Soft-thresholding): reduce each
coordinate's magnitude by ``0.5``, clipping it at zero.
Thus ``(2,-1)`` becomes ``(1.5,-0.5)``.

```@example homepage_intro
using ProximalBase
f = SqrNormL2(1.0)       # f(x) = ||x||²/2
g = NormL1(0.5)          # g(x) = 0.5*||x||₁
x = [2.0, -1.0]
@assert f(x) == 2.5
@assert gradient(f, x) == x
@assert prox(g, x, 1.0) == [1.5, -0.5]
(value = f(x), gradient = gradient(f, x), proximal_point = prox(g, x, 1.0))
```

ProximalBase supplies the function operations and storage of
[iterates](glossary.md#Iterate). A solver chooses steps, update order,
stopping criteria, and the [working set](glossary.md#Working-set) of
coordinates to visit. [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl)
implements algorithms that combine such operations; ProximalBase itself
does not provide a `solve()` function.

## Four pieces that fit together

| Part | Responsibility | Start here |
|:--|:--|:--|
| Function objects | Functions such as losses, regularizers and their nonnegative multiples | [Function objects and their properties](function_objects.md) |
| Mathematical operations | Function values, gradients, proximal maps and checks on whether a zero coordinate should be updated | [Operations on functions](operations.md) |
| Iterates | Dense or specialized storage for coordinates, symmetric variables and blocks | [Overview and working sets](iterates.md) |
| Optimization algorithms | Combine the operations, maintain residuals and select coordinates | [Building optimization algorithms](optimization_algorithms.md) |

For convex ``f`` and ``g``, optimality can be expressed using a
[subdifferential](glossary.md#Subgradient-and-subdifferential).
A vector ``s`` is a subgradient of convex ``g`` at ``x`` when
``g(z)\geq g(x)+\langle s,z-x\rangle`` for all ``z``.
The set of such vectors is denoted ``\partial g(x)``, and the
first-order condition becomes

```math
0\in\nabla f(x)+\partial g(x).
```

For instance, ``\partial |t|=[-1,1]`` at ``t=0``, which is why
a zero coefficient can satisfy the lasso optimality condition even
when the corresponding gradient entry is not zero.
The [coordinate descent guide](coordinate_descent.md) explains how this
leads to deciding which zero coordinates require further updates.

## [Learning paths](@id Where-to-start)

**Use operations in your own code.** Read
[Getting started](getting_started.md) →
[Function objects and their properties](function_objects.md) →
[Operations on functions](operations.md) →
[Function library](function_library.md). Stop there if you only
need the mathematical building blocks.

**Write an optimization algorithm.** Read
[Getting started](getting_started.md) →
[Proximal gradient methods](proximal_gradient.md) →
[Coordinate descent](coordinate_descent.md) →
[Iterates and working sets](iterates.md). Stop there if you are
implementing a solver using existing functions.

**Add a mathematical function.** Read
[Function objects and their properties](function_objects.md) →
[Writing your own function](extending.md) →
[Developer Documentation](developer/index.md). Stop after the extension
guide unless you need to change ProximalBase internals.

The [Glossary](glossary.md) defines unfamiliar terminology.
The [API reference](api/interface.md) lists signatures and qualified names.
For changes from earlier releases, see [Compatibility and migration](migration.md).
