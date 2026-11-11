# ProximalBase.jl

**Mathematical function objects and working-set storage for optimization in Julia.**

Many statistical estimation problems take the form

```math
\min_{x\in\mathbb R^p} F(x)=f(x)+g(x),
```

where ``f`` is a smooth data-fitting loss and ``g`` is a penalty. For example,
least squares measures prediction error, while an ``\ell_1`` penalty encourages
zero coefficients. An algorithm needs to evaluate these functions, differentiate
the loss and apply the penalty's proximal map:

```math
\nabla f(x),\qquad
\operatorname{prox}_{\gamma g}(v)=\operatorname*{argmin}_z
\left\{g(z)+\frac{\|z-v\|^2}{2\gamma}\right\},\quad\gamma>0.
```

```@example homepage_intro
using ProximalBase
f = SqrNormL2(1.0)       # f(x) = ||x||²/2
g = NormL1(0.5)          # g(x) = ||x||₁/2
x = [2.0, -1.0]
@assert f(x) == 2.5
@assert gradient(f, x) == x
@assert prox(g, x, 1.0) == [1.5, -0.5]
(value = f(x), gradient = gradient(f, x), proximal_point = prox(g, x))
```

ProximalBase supplies these mathematical operations and storage for iterates.
A solver chooses the steps, update order, stopping criterion and working set.
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl) implements
optimization algorithms using such operations.

## Four pieces that fit together

| Part | Responsibility | Start here |
|:--|:--|:--|
| Function objects | Losses, penalties and nonnegative scaling | [Function objects and capabilities](function_objects.md) |
| Mathematical operations | Values, gradients, proximal maps and zero-coordinate KKT screens | [Operations on functions](operations.md) |
| Iterates | Entries, independent symmetric coordinates or ordered blocks, with active-zero state | [Overview and working sets](iterates.md) |
| Optimization algorithms | Combine the operations; maintain residuals and select coordinates | [Building optimization algorithms](optimization_algorithms.md) |

For convex ``f`` and ``g``, optimality is characterized by
``0\in\nabla f(x)+\partial g(x)``. Coordinate methods apply this condition to
individual entries or blocks. A working set records which coordinates a solver
currently visits; it can include zeros deliberately retained for the next sweep.
The [coordinate guide](coordinate_descent.md) connects these conditions to
[`prox_block`](@ref) and [`kkt_violation`](@ref).

## Where to start

| Goal | Read |
|:--|:--|
| Fit the mathematical pieces of a lasso objective together | [Getting started](getting_started.md) |
| Find a retained function and its exact scaling | [Function library](function_library.md) |
| Understand allocating and mutating calls | [Proximal operators and gradients](interface.md) |
| Choose sparse, grouped or symmetric storage | [Iterates](iterates.md) |
| Define your own scalar or block proximal operation | [Writing your own function](extending.md) |
| Run small complete programs | [Lasso primitives](examples/lasso.md) and the expandable Examples category |
| Extend the implementation safely | [Developer Documentation](developer/index.md) |
| Look up signatures | [API reference](api/interface.md) |

This manual describes the minimal core after PR #850. The
[migration page](migration.md) identifies the historical full implementation and
removed interfaces. CPU Float32 and Float64 are the numerical validation targets.
