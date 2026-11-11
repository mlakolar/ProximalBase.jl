# Function objects and capabilities

A **function object** represents a mathematical function together with its
parameters. For example, `SqrNormL2(1.0)` represents the function
``f(x) = \|x\|_2^2/2``, while `NormL1(0.5)` represents
``g(x) = 0.5\|x\|_1``. Both are callable:

```@example properties
using ProximalBase
x = [2.0, -1.0]
f = SqrNormL2(1.0)
g = NormL1(0.5)
(f(x), g(x))
```

Every retained vector/array function is an [`AbstractFunction`](@ref).
[`Squared`](@ref ProximalBase.Squared) belongs to a separate scalar
observation-loss interface and is used as `ProximalBase.Squared()`.

The benefit of representing functions as objects is that an optimization
algorithm can request the operations it needs, such as a **gradient** or a
**proximal map**, without knowing the precise formula of the function.
The properties below explain when such operations are useful.

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

For the retained functions, `isconvex(f)` reports the declared mathematical
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
mathematics, whereas subtyping [`SeparableFunction`](@ref) and defining
`value_entry` and `prox_block` enables the *generic separable kernels*.
See [Writing your own function](extending.md).

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
when a sparse iterate has many unstored coordinates, the generic
zero-preserving kernels can skip those coordinates rather than scanning
the entire dense array.

```@example properties
iszeropreserving(g)             # true
prox_block(g, 0.0, 1, 1.0)    # 0.0
```

Zero preservation is **not** the same as separability. A shifted squared
penalty such as ``(x_i-a_i)^2/2`` is separable but, for nonzero ``a_i``,
its proximal operator generally moves zero away from zero. Such a function
cannot safely skip inactive zeros just because it is separable.

Unlike the four type-level properties below,
`iszeropreserving(f)` depends on the **particular function instance and its
parameters**. This contract is mathematical (exact-arithmetic); extreme
floating-point inputs may still produce different exceptional behavior on
dense and stored-only paths. See the [Numerical contract](numerics.md) and
[Iterates](iterates.md).

## Traits in Julia

A **trait** is a query that tells an algorithm or generic kernel which
properties an object advertises. In ProximalBase, the main queries are:

| Trait | Mathematical meaning | Why it is useful |
|:--|:--|:--|
| [`isconvex`](@ref) | Satisfies the convexity inequality | Supports global-optimality reasoning |
| [`issmooth`](@ref) | Supplies gradients on the domain interior | Enables gradient-based operations |
| [`isproximable`](@ref) | Has an efficiently computable proximal map | Enables proximal steps |
| [`isseparable`](@ref) | Decomposes as a sum over scalar entries | Enables coordinate-wise operations |
| [`iszeropreserving`](@ref) | Zero entry values and zero-preserving scalar prox | Enables skipping inactive zeros |

The first four are usually declared on a Julia **type**; an instance query
forwards to that type. Zero preservation is declared for an **instance**,
because weights, centers, levels or scales may change it. The default for
an `AbstractFunction` is `false` unless a specific implementation
declares otherwise.

```@example properties
(issmooth(f), isproximable(f), isseparable(f), isconvex(f),
 iszeropreserving(f))
```

### Retained built-in functions

| Retained function | Smooth | Proximable | Separable | Convex | Zero preserving |
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
accepts vectors; symmetric owners require explicitly supported dispatch.

## Function objects, parameters and extension hooks

Objects retain array parameters by reference unless a constructor
explicitly copies them. Losses and quadratics may also own scratch buffers
and lazy decomposition caches. Review [Array and precision conventions](numerical_usage.md)
before modifying referenced data or sharing a function object across
concurrent calls.

`NormL1` is a [`SeparableFunction`](@ref): its `value_entry` and
scalar `prox_block` hooks feed generic array value/proximal kernels.
`SqrNormL2`, `LinearFunction` and `ZeroFunction` are also
mathematically separable and have scalar `prox_block` methods, but
are not `SeparableFunction` subtypes and do not provide
`value_entry` methods. A custom `SeparableFunction` automatically
declares `isseparable=true` and `isproximable=true`, but its author
must still implement the required scalar hooks. Smoothness is independent.

For the concrete [function library](function_library.md), implemented
[operations](operations.md), and [extension guide](extending.md), follow
the respective guides. The [coverage matrix](api/coverage.md) lists every
export and essential qualified extension hook.
