
# Function objects and capabilities

ProximalBase represents a mathematical objective, penalty, or constraint by a
**function object**. Built-in function types are subtypes of
[`AbstractFunction`](@ref); an object `f` is callable as `f(x)`. The object
contains the mathematical parameters and, where needed, reusable computation
state. A solver interacts with it through a common collection of
[operations](operations.md), rather than a separate solver implementation for
each loss or penalty.

For example, `LeastSquaresLoss(y, X)` represents a data-fitting loss,
`NormL1(λ)` represents a sparsity penalty, and `IndBox(l, u)` represents the
indicator function of a feasible set. The same regularizer can be used in
proximal gradient, coordinate descent or another method that requests its
supported oracles. **The function does not itself choose the algorithm.**

## Values, parameters, and composition

A function object specifies *what is being computed*, not how the solver
iterates. For a composite problem,

```math
min_x F(x) = f(x) + g(x),
```

`f` may be a smooth data-fitting loss and `g` a penalty or an indicator of
constraints. The solver selects `x`, a step size, and the stopping condition;
ProximalBase provides operations such as `gradient!` and `prox!`.

Built-in types fall into several broad roles:

| Role | Examples | Read next |
|:--|:--|:--|
| Smooth losses and quadratic functions | [`LeastSquaresLoss`](@ref), [`SqrNormL2`](@ref), [`LogisticLoss`](@ref), [`QuadraticFunction`](@ref) | [Smooth functions](smooth.md) |
| Penalties and regularizers | [`NormL1`](@ref), [`ElasticNet`](@ref), [`GroupNormL2`](@ref), [`NuclearNorm`](@ref) | [Penalties](penalties.md) |
| Constraints as indicator functions | [`IndBox`](@ref), [`IndSimplex`](@ref), [`IndPSD`](@ref) | [Sets](sets.md) |
| Functions built from other functions | [`Scaled`](@ref), [`Translate`](@ref), [`Constrained`](@ref), [`SeparableSum`](@ref) | [Functions built from functions](calculus.md) |

An indicator function is zero at feasible points and infinite outside the
set. Its proximal map is a projection. Some mathematical functions are both
smooth and proximable: these categories are *capabilities*, not an exclusive
classification. Composition wrappers can preserve, transform, or restrict the
oracles of their component functions; compatibility must be checked for the
actual wrapper and operands.

## Traits and oracles

The public **traits** answer whether a function has a mathematical or
computational property. They do not perform the computation themselves.

| Trait | Meaning | Relevant operations |
|:--|:--|:--|
| [`issmooth(f)`](@ref issmooth) | Differentiable on the interior of its domain | [`gradient!`](@ref), [`value_and_gradient!`](@ref) |
| [`isproximable(f)`](@ref isproximable) | Has a supported, efficiently computed proximal map | [`prox!`](@ref), [`value_and_prox!`](@ref) |
| [`isseparable(f)`](@ref isseparable) | Expressible as a sum of independent entry functions | [`value_entry`](@ref), [`prox_block`](@ref) |
| [`isconvex(f)`](@ref isconvex) | Convexity, used to justify optimization guarantees | Property, not an oracle |
| [`hashessian(f)`](@ref hashessian) | Supports the second-order interface | [`hessian_vec!`](@ref), [`quadratic_model`](@ref) |
| [`haslmo(C)`](@ref haslmo) | A supported linear minimization oracle for this set instance | [`lmo!`](@ref), [`lmo`](@ref); sometimes [`lmo_atom`](@ref) |
| [`iszeropreserving(f)`](@ref iszeropreserving) | Separable zero entries can be skipped under its documented conditions | Sparse-aware evaluation and proximal kernels |

Most capability traits depend on the function's **type**. `haslmo` and
`iszeropreserving` are **instance-dependent**: the bounds or radius of a set,
or the parameters of a penalty, can change the answer.

A true trait is not a guarantee of every combination of argument type,
backend, or numerical parameter. For instance, a smooth function may lack
a *finite known* [`lipschitz`](@ref) constant, and a proximable function
need not have a special kernel for every array or sparse iterate. The
[operations guide](operations.md) and individual docstrings give the actual
supported domains.

```@example function_object_intro
using ProximalBase

f = SqrNormL2(1.0)
g = NormL1(0.5)
x = [2.0, -1.0]

@assert issmooth(f) && isproximable(f)
@assert isproximable(g) && isseparable(g) && isconvex(g)
@assert gradient(f, x) ≈ x
@assert prox(g, x, 1.0) ≈ [1.5, -0.5]
(value = f(x), gradient = gradient(f, x), proximal_point = prox(g, x, 1.0))
```

## Three distinct interfaces

1. **Function operations** describe values, derivatives, proximal maps,
   restrictions, and optimality checks for a mathematical function.
2. **Iterates** describe how an algorithm stores and changes optimization
   coordinates: entries in a [`SparseIterate`](@ref), independent symmetric
   variables in a [`SymmetricSparseIterate`](@ref), or groups in an
   [`BlockIterate`](@ref).
3. **Solvers** choose an update schedule, steps, momentum, line searches,
   active-set management, and convergence tests. ProximalBase supplies
   primitives for these choices, not a general solver interface.

`prox_block(f, u, k, γ)` computes one mathematical penalty block without
writing an iterate. For a separable function, `u` is a scalar and `k` indexes
its parameter arrays. For a group penalty, `u` is a compact vector and `k`
identifies the penalty group. The solver maps that result into an iterate:
a grouped iterate coordinate is a group, and a symmetric iterate coordinate
is an independent variable rather than a full-matrix linear index. See
[Entries and coordinates](@ref) for the required index conversion.

To implement your own mathematical function, read
[Writing your own function](extending.md). To understand Julia dispatch and
the private implementation, read [Developer Documentation](developer/index.md).
