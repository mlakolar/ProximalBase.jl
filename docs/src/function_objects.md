# Function objects and their properties

ProximalBase.jl represents mathematical functions as objects. An
optimization algorithm can ask a function which properties it has before
choosing operations such as a gradient or proximal step.

## The five function traits

A **trait** is a Julia function returning `true` or `false` to describe
a capability or mathematical property. ProximalBase defines **five**
function traits:

| Trait | What it tells you | What an algorithm can do with it |
|:--|:--|:--|
| [`issmooth(f)`](@ref issmooth) | A gradient is available on the interior of the function's domain | Consider gradient-based updates |
| [`isproximable(f)`](@ref isproximable) | A proximal operator is efficiently computable | Consider proximal updates |
| [`isseparable(f)`](@ref isseparable) | The function is a sum of separate scalar-coordinate terms | Consider coordinate-wise computation |
| [`isconvex(f)`](@ref isconvex) | The function satisfies the convexity property under its stated assumptions | Apply suitable convex-optimization guarantees |
| [`iszeropreserving(f)`](@ref iszeropreserving) | Each scalar term is zero at zero and its proximal map sends zero to zero | Skip inactive zeros in supported sparse computations |

These are **all five traits in the current function interface**, not an
illustrative subset. They describe different properties: a function can
be convex without being smooth, or separable without preserving zeros.

Here are two functions with different properties:

```@example properties
using ProximalBase
f = SqrNormL2(1.0)       # f(x) = ||x||² / 2
g = NormL1(0.5)          # g(x) = 0.5 * ||x||₁
x = [2.0, -1.0]
(f(x), g(x))
```

The following sections explain each trait mathematically, show examples,
and discuss what its value does *and does not* guarantee.
## Convexity

A function ``f`` is **convex** if, for any points ``x`` and ``y`` in its
domain and ``0 \leq t \leq 1``,

```math
f(tx+(1-t)y) \leq t f(x) + (1-t)f(y).
```

Geometrically, the line segment joining two points on the graph lies above
the graph. For a convex function, every local minimum is a global minimum
(under the usual meaning of local minimum relative to its domain).
This is why convexity helps establish guarantees for optimization algorithms.

Both the squared Euclidean norm and the absolute-value function are convex:

```math
f(x)=\tfrac12\|x\|_2^2,\qquad
g(x)=\lambda\|x\|_1,\quad \lambda\geq0.
```

A function can be convex without being differentiable: ``|x|`` has a corner
at zero. Thus **convex** and **smooth** describe different properties.

For the functions provided here, `isconvex(f)` reports the declared mathematical
assumption; it is **not** a runtime proof about supplied data. For example,
`QuadraticFunction` assumes its matrix is symmetric positive semidefinite;
its constructor does not certify that assumption.

## Smoothness and gradients

In optimization, a **smooth** function generally means one that can be
differentiated and for which gradient-based computations are appropriate.
For a differentiable function, the gradient is the vector of partial
derivatives:

```math
\nabla f(x)=
\begin{bmatrix}
\partial f(x)/\partial x_1\\
\vdots\\
\partial f(x)/\partial x_p
\end{bmatrix}.
```

It describes the first-order change in the function:

```math
f(x+h)\approx f(x)+\langle\nabla f(x),h\rangle.
```

For the squared norm, ``f(x)=\|x\|_2^2/2``, the gradient is simply
``\nabla f(x)=x``. A gradient-descent step therefore has the form

```math
x^{+}=x-\gamma\nabla f(x).
```

```@example properties
gradient(f, x)                  # [2.0, -1.0]
x - 0.1 * gradient(f, x)        # one gradient-descent step
```

By contrast, the ``\ell_1`` norm is not differentiable at coordinates equal
to zero, so `NormL1` is not smooth. It is still convex and proximable.

**What the trait promises:** `issmooth(f)` indicates that the function
supplies gradient and fused value/gradient operations on the interior of its
domain. It does **not**, by itself, establish a globally Lipschitz gradient
or an acceptable step size. Where available, consult [`lipschitz`](@ref)
and use an algorithm with an appropriate step-selection rule.

## Proximability and proximal operators

A function is **proximable** when its *proximal operator* can be computed
efficiently. For step ``\gamma>0``, this operator is

```math
\operatorname{prox}_{\gamma g}(x)
=\mathop{\arg\min}_z
\left\{
g(z)+\frac{1}{2\gamma}\|z-x\|_2^2
\right\}.
```

The proximal operator balances two goals: reducing ``g(z)`` while staying
near the input ``x``. Unlike a gradient step, it can handle many
**nondifferentiable** penalties directly.

For ``g(x)=\lambda\|x\|_1``, the proximal operator is **soft
thresholding**, applied independently to each coordinate:

```math
[\operatorname{prox}_{\gamma g}(x)]_i
=\operatorname{sign}(x_i)\max(|x_i|-\gamma\lambda,0).
```

```@example properties
prox(g, x, 1.0)                 # [1.5, -0.5]
isproximable(g)                # true
issmooth(g)                    # false
```

This illustrates why a nonsmooth penalty can still be convenient in a
solver. For example, a proximal-gradient algorithm for ``f+g`` can use

```math
x^{+}=\operatorname{prox}_{\gamma g}
\bigl(x-\gamma\nabla f(x)\bigr).
```

ProximalBase provides the mathematical operations; choosing step sizes,
iterations and stopping criteria belongs to an optimization algorithm
(such as one implemented using ProximalSolvers.jl). A proximable trait
does not mean a prox method supports every possible input shape or storage
type. See [Operations on functions](operations.md).

## Separability and coordinate operations

A function is **separable** when it can be written as a sum of functions
of individual coordinates:

```math
g(x)=\sum_{i=1}^{p}g_i(x_i).
```

For example,

```math
\lambda\|x\|_1=\sum_{i=1}^{p}\lambda|x_i|,
\qquad
\tfrac12\|x\|_2^2=\sum_{i=1}^{p}\tfrac12 x_i^2.
```

For a separable function, the proximal minimization splits into independent
scalar problems. This means each coordinate can be processed without solving
a coupled ``p``-dimensional proximal problem. That is useful for coordinate
descent, block-oriented extensions and sparse storage.

```@example properties
isseparable(g)                 # true
prox_block(g, 2.0, 1, 1.0)    # 1.5, scalar proximal operation at entry 1
```

The quadratic form ``x^\mathsf{T}Ax/2`` is **not** generally separable because
off-diagonal entries of `A` couple different coordinates. Even when a
function is mathematically separable, its Julia implementation does not
automatically inherit every scalar hook: `isseparable(f)` describes the
mathematics, whereas implementing the appropriate Julia methods makes coordinate operations
available. See [Writing your own function](extending.md) for the methods
needed to define a new separable function.

## Zero preservation and sparse iterates

Zero preservation is a stronger property relevant to sparse computation.
For a separable function ``g(x)=\sum_i g_i(x_i)``, ProximalBase considers it
**zero preserving** when, at every coordinate ``i``,

```math
g_i(0)=0,\qquad
\operatorname{prox}_{\gamma g_i}(0)=0
\quad\text{for every finite }\gamma>0.
```

The second condition means that applying a proximal update to a zero
coordinate leaves it at zero. The first means that omitted zeros also
contribute nothing to the function value.

For the ``\ell_1`` penalty, soft thresholding maps zero to zero. Therefore,
when a sparse iterate has many unstored coordinates, ProximalBase
can skip those coordinates rather than scanning
the entire dense array.

```@example properties
iszeropreserving(g)             # true
prox_block(g, 0.0, 1, 1.0)    # 0.0
```

Zero preservation is **not** the same as separability. A shifted squared
penalty such as ``(x_i-a_i)^2/2`` is separable but, for nonzero ``a_i``,
its proximal operator generally moves zero away from zero. Such a function
cannot safely skip inactive zeros just because it is separable.

Unlike the other four traits, `iszeropreserving(f)` depends on the
**particular function instance and its parameters**. This contract is mathematical (exact-arithmetic); extreme
floating-point inputs may still produce different exceptional behavior on
dense and stored-only paths. See the [Numerical contract](numerics.md) and
[Iterates](iterates.md).

## Properties of the built-in functions

The following table applies the five traits to the functions provided by
ProximalBase. For a parameterized function, zero preservation can depend on
the parameters of the particular object.


| Function | Smooth | Proximable | Separable | Convex | Zero preserving |
|:--|:--:|:--:|:--:|:--:|:--|
| `ZeroFunction` | yes | yes | yes | yes | yes |
| `LinearFunction` | yes | yes | yes | yes | no |
| `SqrNormL2` | yes | yes | yes | yes | uncentered and finite level only |
| `QuadraticFunction` | yes | yes | no | yes¹ | no |
| `LeastSquaresLoss` | yes | yes | no | yes | no |
| `NormL1` | no | yes | yes | yes | yes |
| `Scaled` | forwards | forwards | forwards | forwards | finite scale and zero-preserving inner function |

¹ `QuadraticFunction` assumes a symmetric positive semidefinite matrix.
Its constructor does not verify these assumptions, and the trait is not a
matrix validator.

**Important:** A true trait describes a mathematical capability but does
not manufacture Julia methods or guarantee support for every array layout,
precision or storage representation. For example, `LeastSquaresLoss`
accepts vector or matrix coefficients, whereas `QuadraticFunction`
accepts vectors; symmetric matrix storage has its own supported operations.
See [SymmetricSparseIterate](symmetric_sparse_iterate.md) for details.

## How an algorithm uses traits

The point of querying a trait is to make a decision **before calling an
operation**. For example, suppose an algorithm implements a
proximal-gradient step for a smooth loss `f` and a proximable penalty `g`.
Instead of assuming those capabilities, it can check them:

```@example properties
f = SqrNormL2(1.0)          # smooth squared-norm loss
g = NormL1(0.5)             # nonsmooth, proximable l1 penalty
x = [2.0, -1.0]
γ = 0.1

if issmooth(f) && isproximable(g)
    x_next = prox(g, x - γ * gradient(f, x), γ)
else
    error("This step requires a smooth loss and a proximable penalty")
end
x_next
```

The conditions matter: `issmooth(g)` is `false`, so the algorithm should
not attempt to use `gradient(g, x)` as if this were a differentiable
penalty. But `isproximable(g)` is `true`, so it can apply `prox(g, ..., γ)`.
Conversely, `issmooth(f)` is `true`, so the algorithm can compute its gradient.

Other traits guide other choices. A coordinate algorithm may check
`isseparable(g)` before attempting independent scalar updates, while a
sparse algorithm may also check `iszeropreserving(g)` to decide whether
it can skip coordinates currently equal to zero. An algorithm relying on
convexity may check `isconvex(f)` and `isconvex(g)` before invoking a
convex-specific convergence argument.

```@example properties
(smooth_loss = issmooth(f),
 proximable_penalty = isproximable(g),
 separable_penalty = isseparable(g),
 zero_preserving_penalty = iszeropreserving(g),
 convex_objective_parts = isconvex(f) && isconvex(g))
```

These names make clear **which function** is being queried and **why**.
More importantly, traits are not an automatic solver: they do not select
a valid step size or prove that arbitrary inputs satisfy a function's
assumptions. A `true` trait also does not create Julia methods for every
array shape or iterate representation. Consult
[Operations on functions](operations.md) and [The interface](interface.md)
for the methods actually supported.

## Declaring traits for a new function

For most properties (`issmooth`, `isproximable`, `isseparable`, and
`isconvex`), the trait is typically defined on the Julia **type**;
calling it on an object forwards to that type. Zero preservation is
queried on the **instance**, because its parameters (such as a center)
can change whether zero remains zero.

Undefined traits default to `false`. Declaring one says what the
function supports; it does not implement the corresponding computation.
See [Writing your own function](extending.md) for the methods a new
function must provide.
