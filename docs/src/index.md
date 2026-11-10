
# ProximalBase.jl

**Reusable mathematical function objects and computational operations
for building optimization algorithms in Julia.**

ProximalBase provides losses, regularizers and constraint indicators,
together with gradients, proximal mappings, coordinate updates,
Hessian-vector products, linear minimization oracles and optimality
checks. The goal is to let an optimization algorithm ask a function
for the computation it needs, without implementing the same mathematics
again for every new loss or penalty.

The package is particularly useful for composite optimization,
including problems common in high-dimensional statistics:

```math
min_x F(x) = f(x) + g(x),
```

where `f` may be a smooth data-fitting loss and `g` a sparsity penalty
or the indicator function of a constraint set.

## A first example: functions and their operations

```@example homepage_intro
using ProximalBase

f = SqrNormL2(1.0)                  # a smooth objective
g = NormL1(0.5)                     # a proximable penalty
x = [2.0, -1.0]

value = f(x)                       # evaluate the function
grad = gradient(f, x)              # its gradient
point = prox(g, x, 0.5)           # the penalty's proximal map

γ = 0.5
next_x = prox(g, x - γ * gradient(f, x), γ)
@assert grad ≈ x
(value = value, gradient = grad, proximal_point = point, one_step = next_x)
```

The same `NormL1` function object can support both this full-vector
proximal step and a scalar entry update in coordinate descent. Which
algorithm runs is the caller's choice.

## Four pieces that fit together

| Part | Responsibility | Start here |
|:--|:--|:--|
| **Function objects** | Represent losses, penalties, constraints, or compositions of these | [Function objects and capabilities](function_objects.md) |
| **Operations (oracles)** | Evaluate values, gradients, proximal maps, curvature, restrictions, linear minimization and optimality conditions | [Operations on functions](operations.md) |
| **Iterates** | Represent entries, independent symmetric variables or groups, with optional sparse working-set behavior | [Iterates](iterates.md) |
| **Optimization algorithms** | Choose steps, update schedules, working sets, and stopping criteria from the available operations | [Building optimization algorithms](optimization_algorithms.md) |

ProximalBase is **not** a general solver that automatically selects
an algorithm or stopping rule. It does contain numerical algorithms
*inside individual oracles* (for example certain projections or matrix
computations). Complete solver implementations built on these
operations are available in
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl).

The available mathematical objects are collected in the
[Function library](function_library.md), including smooth losses,
penalties, indicator functions and function-composition wrappers.
For numerical types, array aliasing and GPU considerations, consult
[Array, precision, and device conventions](numerical_usage.md).

## Where to start

| Goal | Read |
|:--|:--|
| See two algorithms solve the same lasso problem | [Getting started](getting_started.md) |
| Understand `AbstractFunction` and capability traits | [Function objects and capabilities](function_objects.md) |
| Discover `gradient!`, `prox!`, `lmo!`, coordinate and second-order operations | [Operations on functions](operations.md) |
| Find a loss, regularizer or constraint set | [Function library](function_library.md) |
| Understand proximal gradient and coordinate descent | [Building optimization algorithms](optimization_algorithms.md) |
| Pick a sparse, grouped or symmetric variable representation | [Iterates](iterates.md) |
| Create or extend your own function | [Writing your own function](extending.md) |
| Look up exact signatures and exceptions | [API reference](api/interface.md) |
| Work on internal Julia dispatch, contracts or implementation | [Developer Documentation](developer/index.md) |

The [Examples](examples/01_lasso_three_ways.md) section contains complete
programs for regression, matrix estimation, graphical models, and constrained
optimization. The examples are executed when the documentation is built.

## Related packages

For statistical estimation and inference, see
[HighDimensionalModels.jl](https://github.com/mlakolar/HighDimensionalModels.jl) for
penalized regression,
[CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl)
for Gaussian graphical models, and
[DiffPrecTest.jl](https://github.com/mlakolar/DiffPrecTest.jl) for inference on
differences of precision matrices.
