# Function objects and their properties

*Assumes: convexity, gradients, and basic Julia · You will learn: all five function traits and how to use them when selecting operations.*

ProximalBase.jl represents mathematical functions as objects. An
optimization algorithm can ask a function which properties it has before
choosing operations such as a gradient or
[proximal step](glossary.md#Proximal-operator).

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

You already know the definition of convexity. In this package,
`isconvex(f)` reports that the function **declares** this property,
which an algorithm may use when choosing its convergence guarantees.
This is a declaration, not a runtime verification of the parameters:
for example, `QuadraticFunction` assumes a symmetric positive-semidefinite
coefficient matrix and does not certify that condition at construction.
A convex function need not be differentiable, as ``|t|`` illustrates.

```@example properties
@assert isconvex(f) && isconvex(g)
```

## Smoothness and gradients

The trait `issmooth(f)` indicates that the function supplies gradient
and combined value/gradient operations on the interior of its domain.
It does **not** promise a particular step size or a globally Lipschitz
gradient. When available, use
[`lipschitz`](@ref) to obtain a bound appropriate to the algorithm.

```@example properties
gradient(f, x)                  # [2.0, -1.0]
@assert issmooth(f) && !issmooth(g)
```

## Proximability and proximal operators

A **[proximal operator](glossary.md#Proximal-operator)** generalizes
something you already know: Euclidean projection.
If ``g`` is the mathematical **indicator** of a nonempty closed convex
set ``C`` (zero on ``C`` and ``+\infty`` outside),
then the proximal point of ``g`` is the projection of the input onto
``C``. Thus proximal gradient extends projected gradient descent.
This indicator is a mathematical illustration, not a claim that the
current function library includes every indicator function.

For a general function ``g`` and positive step ``\gamma``,
the proximal operator instead solves

```math
\operatorname{prox}_{\gamma g}(v)=
\operatorname*{argmin}_z\left\{g(z)+\frac{\|z-v\|_2^2}{2\gamma}\right\}.
```

The first term rewards lowering ``g``; the second discourages moving too
far from ``v``. A function is **[proximable](glossary.md#Proximable)**
when its proximal point can be computed efficiently.
In particular, it may be proximable without being differentiable.

For the ℓ₁ function ``g(x)=\lambda\|x\|_1``, the proximal operator is
[soft thresholding](glossary.md#Soft-thresholding):
reduce each magnitude by ``\gamma\lambda`` and set it exactly to zero
when it starts in ``[-\gamma\lambda,\gamma\lambda]``.

```math
[\operatorname{prox}_{\gamma g}(v)]_i
=\operatorname{sign}(v_i)\max(|v_i|-\gamma\lambda,0).
```

For our earlier example, ``v=(2,-1)``, ``\lambda=0.5`` and
``\gamma=1``. The threshold is ``0.5``, giving
``(2-0.5,-(1-0.5))=(1.5,-0.5)``.

```@example properties
@assert prox(g, x, 1.0) == [1.5, -0.5]
@assert isproximable(g) && !issmooth(g)
```

**Why does a solver combine a gradient and a proximal map?**
You know that an ``L``-smooth loss obeys the quadratic upper bound

```math
f(z)\le f(x)+\langle\nabla f(x),z-x\rangle
+\frac{L}{2}\|z-x\|_2^2.
```

Add ``g(z)`` to that upper model and minimize it over ``z``.
Completing the square yields the proximal-gradient step, with
``\gamma=1/L``:

```math
x^+=\operatorname{prox}_{\gamma g}
\bigl(x-\gamma\nabla f(x)\bigr).
```

A solver chooses the step size, repeats the update, and checks a stopping
condition; ProximalBase only supplies the mathematical operations.
The trait `isproximable(g)` does not imply that every possible array
shape or iterate type has a corresponding `prox` method.
See [Operations on functions](operations.md) for supported calls.

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
