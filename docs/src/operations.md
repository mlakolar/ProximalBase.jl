# [Operations on functions](@id operation-guide)

*Assumes: gradients and basic Julia · You will learn: the complete function-operation catalog, including proximal maps, mutation and coordinate checks.*

ProximalBase provides **operations on mathematical functions**. They answer
questions such as “What is the objective value?”, “What is its gradient?”,
or “What is the [proximal point](glossary.md#Proximal-operator)?” An optimization algorithm combines
these operations to update its variables and check optimality.

## Available operations at a glance

This is the complete set of exported **function operations** in the current
package, grouped by purpose. Most operations are supported by specific
function families, not every function object. The table is a starting point;
each operation is explained below, including its mathematics and Julia usage.

| Purpose | Julia operations | What they compute |
|:--|:--|:--|
| Evaluate | `f(x)` | Value of the function at `x` |
| Differentiate | `gradient`, `gradient!` | Gradient at `x` |
| Value and derivative together | `value_and_gradient`, `value_and_gradient!` | Function value and gradient at `x` |
| Proximal point | `prox`, `prox!` | Minimizer of the proximal subproblem |
| Value and proximal point | `value_and_prox!` | Proximal point and its function value |
| Individual entries | `value_entry` | Contribution of one scalar entry to a separable function |
| Scalar or block proximal update | `prox_block`, `prox_block!` | Proximal point for one entry or an explicitly implemented block |
| Check first-order optimality | `kkt_violation` | Screen of a coordinate assumed zero, score of the actual coordinate, or the nonnegative full-point measure |
| Make a reduced function | `restrict` | Function on selected coordinates, normally fixing others at zero |
| Gradient bound | `lipschitz` | Bound on how rapidly the gradient can change |
| Squared spectral norm estimate | `opnorm2` | Estimate of the squared matrix operator norm |
| Soft-threshold a scalar | `shrink` | Apply the scalar soft-threshold utility |
| Update after data changes | `reset_cache!` | Invalidate cached decompositions |

`f(x)` is ordinary callable-function syntax rather than a separately
exported named operation. There is no allocating `value_and_prox` method.
The scalar utility `shrink(v,c)` returns
``\operatorname{sign}(v)\max(|v|-c,0)`` for nonnegative threshold ``c``;
see [The ℓ₁ norm](penalties.md).

Validation hooks used when defining new functions are described under
[Writing your own function](extending.md). Storage operations on
`SparseIterate`, `BlockIterate` and `SymmetricSparseIterate` are cataloged
in the [Iterates guide](iterates.md).

## Mathematical setting

Many optimization problems combine a smooth loss and a penalty:

```math
\min_x\;F(x)=f(x)+g(x).
```

The algorithm can evaluate `f` and `g`, differentiate `f`, and apply a
proximal operator to `g`. The available operations depend on the function's
[traits and concrete methods](function_objects.md).
## Evaluating a function

**Question:** What is the value of a function at the current point?

For example, the squared Euclidean norm with scale one is

```math
f(x)=\frac12\sum_{i=1}^p x_i^2=\frac12\|x\|_2^2.
```

```@example operations_basics
using ProximalBase
x = [2.0, -1.0]
f = SqrNormL2(1.0)
f(x)                         # 2.5
```

The result of `f(x)` is a **scalar**. To evaluate a composite objective,
add the values of its constituent functions:

```@example operations_basics
g = NormL1(0.5)
f(x) + g(x)                  # 4.0
```

Values are useful for comparing candidate solutions and monitoring progress.

## Computing a gradient

**Question:** In which direction does a differentiable function increase
most rapidly near the current point?

The gradient of ``f:\mathbb R^p\to\mathbb R`` collects its partial
derivatives:

```math
\nabla f(x)=\begin{bmatrix}
\partial f(x)/\partial x_1\\
\vdots\\
\partial f(x)/\partial x_p
\end{bmatrix}.
```

For ``f(x)=\|x\|_2^2/2``, the gradient is ``\nabla f(x)=x``.
Moving a short distance in the negative-gradient direction gives a
gradient-descent step:

```math
x^+=x-\gamma\nabla f(x),\qquad\gamma>0.
```

```@example operations_basics
grad = gradient(f, x)        # [2.0, -1.0]
trial = x - 0.1 * grad       # [1.8, -0.9]
```

`gradient(f,x)` returns a new gradient array. `gradient!(out,f,x)` computes
the same gradient in an existing destination. The destination goes first:

```@example operations_basics
out = similar(x)
gradient!(out, f, x)
@assert out == grad
```

Use the in-place form when an algorithm repeatedly needs a gradient and
can reuse its buffer. Not every function has a gradient: for example,
``\|x\|_1`` is nondifferentiable at zero.

## Evaluating a value and gradient together

**Question:** Can we obtain both the objective value and gradient at
the same point without making two separate oracle calls?

`value_and_gradient(f,x)` returns the pair ``(f(x),\nabla f(x))``.
The mutating form writes the gradient to `out` and **returns the value at
the input** `x`:

```@example operations_basics
value, grad2 = value_and_gradient(f, x)
buf = similar(x)
value2 = value_and_gradient!(buf, f, x)
@assert value == value2 == f(x)
@assert grad2 == buf == gradient(f, x)
```

This fused operation is convenient in methods that evaluate both the
objective and a descent direction at every iteration. It does not
necessarily promise fewer allocations or a particular speedup.

## Computing a proximal point

**Question:** What point balances decreasing a penalty with staying
close to the current trial point?

For a function ``g`` and a positive step ``\gamma``, the proximal operator
solves

```math
\operatorname{prox}_{\gamma g}(v)
=\operatorname*{argmin}_z\left\{g(z)+\frac{1}{2\gamma}\|z-v\|_2^2\right\}.
```

Unlike a gradient step, a proximal map can handle nonsmooth penalties.
For ``g(x)=\lambda\|x\|_1``, it is soft thresholding:

```math
[\operatorname{prox}_{\gamma g}(v)]_i
=\operatorname{sign}(v_i)\max(|v_i|-\gamma\lambda,0).
```

```@example operations_basics
prox(g, x, 1.0)              # [1.5, -0.5]
```

`prox!(out,g,x,γ)` writes the proximal point into `out` and returns that
destination. `value_and_prox!(out,g,x,γ)` writes the point but instead
returns **the value of `g` at the computed output**:

```@example operations_basics
point = similar(x)
prox!(point, g, x, 1.0)
@assert point == [1.5, -0.5]
penalty_at_point = value_and_prox!(point, g, x, 1.0)
@assert penalty_at_point == g(point)
```

The distinction matters: `value_and_gradient!` returns the value at
the **input**, whereas `value_and_prox!` returns the value at the
**stored proximal result**.

For [`IndBallL1`](@ref) and [`IndBallL2`](@ref), the proximal point is the
Euclidean projection at any nonnegative scalar step, including `0` and
`Inf`. A negative or NaN step throws before writing. The fused oracle
evaluates the stored point, which rounding can leave just outside the ball.

```@example operations_basics
ball = IndBallL1(3.0)
projected = prox(ball, [3.0, -1.0])   # soft threshold at level 0.5
@assert projected == [2.5, -0.5]
@assert prox(ball, [3.0, -1.0], 10.0) == projected
(ball([3.0, -1.0]), ball(projected))
```

## Putting gradients and proximal maps together

When minimizing ``f+g`` with a smooth ``f`` and a proximable ``g``,
a proximal-gradient step first moves using the gradient of ``f`` and
then applies the proximal map of ``g``:

```math
x^+=\operatorname{prox}_{\gamma g}\bigl(x-\gamma\nabla f(x)\bigr).
```

```@example operations_basics
γ = 0.1
x_next = prox(g, x - γ * gradient(f, x), γ)
@assert x_next == [1.75, -0.85]
x_next
```

This is **one step**, not a complete solver. A solver chooses suitable
steps, repeats the update and checks convergence. See
[proximal-gradient methods](proximal_gradient.md) for that context.

## Operating on individual coordinates or blocks

**Question:** Can the algorithm update only one coordinate or small block
instead of processing the whole vector?

If ``g(x)=\sum_i g_i(x_i)`` is separable, its proximal operator
can act independently on scalar coordinates. The scalar hook
`prox_block(g,v,i,γ)` returns

```math
\operatorname{prox}_{\gamma g_i}(v)
=\operatorname*{argmin}_z\left\{g_i(z)+\frac{(z-v)^2}{2\gamma}\right\}.
```

Here `v` is the proposed scalar point and `i` identifies the **global
entry**, so coordinate-dependent parameters can be selected.

```@example operations_basics
prox_block(g, 2.0, 1, 1.0)  # 1.5
```

`value_entry(g,v,i)` evaluates the scalar contribution ``g_i(v)``
where this operation is implemented (not for every mathematically
separable function). `GroupNormL2`, `GroupNormLinf` and `SparseGroupNormL2` implement
`prox_block!(out,g,u,k,γ)` on compact vector buffers; custom group functions
can implement the same hook. These hooks
compute mathematical answers; **they do not update an iterate's active
set or write back to solver state**. See [Coordinate oracles](iterate_oracles.md)
and [Writing your own function](extending.md) for precise contracts.

## Checking optimality with KKT conditions

This uses the [nonsmooth KKT conditions](glossary.md#KKT-conditions-for-nonsmooth-objectives),
not just stationarity equations involving ordinary derivatives.

**Question:** Is a coordinate that we have kept at zero allowed to
remain zero at the optimum?

For a convex differentiable loss ``f`` and an ``\ell_1`` penalty
``g(x)=\lambda\|x\|_1``, the optimality condition for a **zero**
coordinate ``i`` is

```math
|\nabla_i f(x)|\leq\lambda.
```

`kkt_violation` has three forms. In each, `v` is a vector that the caller
supplies, usually the gradient of the smooth part at the point `x`; it is not
recomputed. The result is the raw score of the stationarity condition.

| Call | What it evaluates | Sign |
|:--|:--|:--|
| `kkt_violation(g,v,i)` | The screen of coordinate `i` **assumed zero**; it reads no `x` | Signed |
| `kkt_violation(g,v,x,k)` | The score of the **actual** coordinate `k` of `x` (the group `k` for a `BlockIterate`) | The signed margin at a zero coordinate; a nonnegative mismatch at a nonzero one |
| `kkt_violation(g,v,x)` | The full-point measure ``\max(0,\max_k s_k)`` over the entries of `x`, with ``s_k`` the score of entry ``k``; zero of the result type for an empty `x`. See below for block, norm and smooth-function scores | Nonnegative |

The sign says the following.

* A **positive** score at a zero coordinate says that zero violates its
  condition: the coordinate is a candidate for a working set.
* A **positive** score at a nonzero coordinate is a stationarity mismatch. It
  is not an instruction to set the coordinate to zero.
* A **negative** score occurs only at a zero coordinate (or at a zero block)
  and is slack. The full-point form drops it through the maximum with `0`.

For an entry `i` with ``c_i=\lambda_0w_i`` (``\lambda_1w_i`` for `ElasticNet`),
the six families with these forms return the following scalars.

| Function | Entry at zero | Entry ``x_i\neq0`` | Full-point value, for finite inputs |
|:--|:--|:--|:--|
| `NormL1` | ``\lvert v_i\rvert-c_i`` | ``\lvert v_i+c_i\operatorname{sign}(x_i)\rvert`` | ``\ell_\infty`` distance from ``-v`` to ``\partial g(x)``; zero when ``0\in v+\partial g(x)`` |
| `NormL1Nonneg` | ``-v_i-c_i``, for finite converted inputs | ``\lvert v_i+c_i\rvert`` for ``x_i>0``, `Inf` for ``x_i<0``, for finite converted inputs | ``\ell_\infty`` distance from ``-v`` to ``\partial g(x)`` for ``x\geq0``, and `Inf` where an entry of ``x`` is negative |
| `ElasticNet` | ``\lvert v_i\rvert-c_i`` (`-Inf` for an infinite ridge level) | ``\lvert (v_i+\lambda_2x_i)+c_i\operatorname{sign}(x_i)\rvert`` | As `NormL1` |
| `ZeroFunction` | ``\lvert v_i\rvert`` | ``\lvert v_i\rvert`` | ``\lVert v\rVert_\infty`` |
| `LinearFunction` | ``\lvert v_i+c_i\rvert`` | ``\lvert v_i+c_i\rvert`` | ``\lVert v+c\rVert_\infty`` |
| `SqrNormL2` | ``\lvert v_i+d_i\rvert`` | ``\lvert v_i+d_i\rvert`` | ``\lVert v+\nabla g(x)\rVert_\infty`` |

For `LinearFunction`, ``c_i`` is its coefficient entry. For `SqrNormL2`, the
three-argument screen takes ``x_i=0`` (``\lvert v_i\rvert``, or ``\lvert v_i-\lambda y_i\rvert`` with a
center ``y``), and the four-argument and full-point forms use ``d_i``, entry ``i`` of ``\nabla g(x)`` as the
ridge-gradient helpers form it in the working type of the call: ``\lambda x_i``, or
``\lambda(x_i-y_i)`` with a center. For finite `v` and `x`, the full-point value equals the ``\ell_\infty`` norm of `v + gradient(g,x)` when
the gradient is computed in the same precision; the rules for non-finite entries take precedence over this formula,
at an infinite level too. The subdifferential of a smooth function is its gradient, so its score is
nonnegative at a zero coordinate as well. The norm of the full-point value is the ``\ell_\infty`` norm for these
functions; [`GroupNormL2`](group_penalty.md) measures block 2-norms, so a
tolerance chosen for one is not a tolerance for the other.

`GroupNormL2` has a score per block, with ``c=\lambda_0w_k``. `NormL2` (with ``c=\lambda``) and `NormLinf` have the
full-point form and no form for a block:

| Function | Block at zero | Block ``x_G\neq0`` | Full-point value, for finite inputs |
|:--|:--|:--|:--|
| `GroupNormL2` | ``\lVert v_G\rVert_2-c`` | ``\lVert v_G+c\,x_G/\lVert x_G\rVert_2\rVert_2`` | ``\max(0,\text{block scores},\lvert v_i\rvert\text{ for the entries in no group})``, a maximum of block 2-norms |
| `NormL2` | (no form for a block) | (no form for a block) | the positive part of that score with ``G`` all of `x`, the 2-norm distance from ``-v`` to ``\partial g(x)`` |
| `NormLinf` | (no form for a block) | (no form for a block) | the 2-norm distance from ``-v`` to ``\partial g(x)`` for ``g(x)=\lambda\max_i\lvert x_i\rvert``: at ``x=0``, the distance from ``v`` to the ℓ₁ ball of radius ``\lambda`` (``0`` when ``\sum_i\lvert v_i\rvert\le\lambda``); at ``x\ne0``, the distance with the face of the entries of largest magnitude, compared for equality after conversion to the working type |

`GroupNormLinf` scores a group at zero with the margin ``\lVert v_G\rVert_1-c`` of its dual (ℓ₁) norm, with ``c=\lambda_0w_k``, and has no
score at a nonzero group, no full-point form and no three-argument form (`ArgumentError`): see [the group ℓ∞ norm](group_penalty.md#group-linf-guide).

The local form of `GroupNormL2` takes an entry of an array point, which selects its group, or the group
`k` of a `BlockIterate`; an entry in no group scores ``\lvert v_i\rvert``. `NormL2` and `NormLinf` have no form for an entry or
a group (an `ArgumentError`); the value of `NormLinf` is a 2-norm distance to the subdifferential and not a maximum of scores,
and its definition, ties and an example are in [the ℓ∞ norm guide](@ref norm-linf-kkt).
`LeastSquaresLoss` and `QuadraticFunction` give the infinity norm of
`v + ∇g(x)` in the full-point form, with the gradient stored in the function's
type by `gradient!`. See
[smooth stationarity scores](@ref smooth-kkt) for precision and scratch ownership.
`Scaled` of the functions with a full-point score evaluates that score on `v/α` and multiplies it by `α`, with
the limits and unsupported cases of the [scaling guide](@ref scaled-kkt).

The worked example uses ``g=\lambda\|x\|_1`` with ``\lambda=0.5``, a point with
two zero coordinates and two nonzero ones, and a supplied `v`:

```@example operations_basics
xk = [0.0, 0.0, 2.0, -1.0]
vk = [0.8, 0.2, 0.0, 0.5]
screen = [kkt_violation(g, vk, i) for i in 1:4]
local_scores = [kkt_violation(g, vk, xk, i) for i in 1:4]
(screen = screen, local_scores = local_scores, full_point = kkt_violation(g, vk, xk))
```

The two forms agree at the zero coordinates 1 and 2: ``0.8-0.5>0`` says that
coordinate 1 violates the zero condition, and ``0.2-0.5<0`` is slack at
coordinate 2. At coordinate 3 the screen assumes ``x_3=0`` and returns
``\lvert 0\rvert-0.5=-0.5``; the four-argument form reads ``x_3=2`` and returns
``\lvert 0+0.5\operatorname{sign}(2)\rvert=0.5``, a mismatch. At coordinate 4
the condition holds and the score is ``0``. The full-point value is ``0.5``.
The values at coordinates 1 and 2 are computed in floating point, so compare
them with `≈` and not with `==`:

```@example operations_basics
@assert screen[3] == -0.5 && local_scores[3] == 0.5
@assert local_scores[1] ≈ 0.3 && local_scores[2] ≈ -0.3 && local_scores[4] == 0
@assert kkt_violation(g, vk, xk) == 0.5
```

ProximalBase returns the raw score. Scaling by the size of the problem, the
tolerance, when to evaluate the score, and the decision that a solver has
converged belong to the caller. The score is a statement about first-order
stationarity of the supplied `v`, not a claim of global optimality, and its accuracy depends on the
accuracy of `v`. `kkt_violation(g,v,x,k)` interprets `k` in the
iterate's coordinate or block space; see [Coordinate oracles](iterate_oracles.md)
and [Coordinate descent](coordinate_descent.md).

## Restricting a function to selected coordinates

`restrict(f, S)` constructs a smaller function on selected coordinates.
It **does not require separability**: quadratics and vector-response least
squares are supported alongside separable functions. Coordinates outside
`S` are normally fixed at **zero**, rather than their current iterate
values; centered squared norms additionally omit a constant from the
excluded coordinates.

See [Restricting functions](restrict.md) for the precise mathematics,
nonseparable examples, supported methods, and the distinction between
zero-fixed and nonzero-fixed subproblems.

## Gradient bounds and step sizes

**Question:** How large a gradient step can the algorithm safely try?

A differentiable function has an ``L``-Lipschitz gradient if

```math
\|\nabla f(x)-\nabla f(y)\|_2\leq L\|x-y\|_2.
```

Such a bound helps guide step sizes: for many convex smooth objectives,
a step no larger than ``1/L`` is a standard starting choice.
`lipschitz(f)` returns a bound for supported smooth function families.

```@example operations_basics
L = lipschitz(f)
@assert L == 1.0
```

For a real matrix ``X``, the squared spectral norm is
``\|X\|_2^2=\lambda_{\max}(X^\mathsf{T}X)``.
`opnorm2(X)` returns an **iterative estimate** of this quantity:

```@example operations_norm
using ProximalBase
Xnorm = [3.0 0.0; 0.0 4.0]
estimate = opnorm2(Xnorm)
# The exact squared spectral norm is 4² = 16.
(estimate = estimate, exact = 16.0)
```

An estimate is **not a certified upper bound**. Do not rely on it alone
for step-size safety; use an independently valid bound or backtracking.

## Evaluating at a caller-held predictor

`ProximalBase.value_from_predictor(f, η)` evaluates a loss at a supplied
predictor. `ProximalBase.value_and_gradient_from_predictor!(g, d, f, η)`
also writes its observation-space derivative and coefficient gradient. For
[`LogisticLoss`](@ref), the predictor includes the offset. See
[logistic regression](smooth.md#Logistic-regression-loss) and
[least-squares predictor and residual operations](smooth.md#Least-squares-at-a-predictor-or-residual-that-the-caller-holds).

## Invalidating cached calculations

**Question:** What if I modify data already stored by reference
inside a function object?

Some function objects reuse matrix decompositions to speed up
repeated proximal evaluations. If the data change but their shape stays
the same, `reset_cache!(f)` invalidates any such cached decomposition
so a later computation can rebuild it. This is a **lifecycle operation**,
not a new mathematical function.

For least squares, the proximal point at ``v`` solves
``\min_z\{\|y-Xz\|_2^2/2+\|z-v\|_2^2/(2\gamma)\}``.
If ``X`` changes, the cached matrix decomposition no longer describes
the current problem.

```@example operations_cache
using ProximalBase
Xcache = [1.0 0.0; 0.0 1.0]
ycache = zeros(2)
loss = LeastSquaresLoss(ycache, Xcache)
v = ones(2)
_ = prox(loss, v, 1.0)   # may build and cache a decomposition
Xcache[1, 1] = 2.0       # same dimensions; different mathematical loss
reset_cache!(loss)
z = prox(loss, v, 1.0)
@assert z ≈ [1 / 5, 1 / 2]
z
```

Changing data dimensions requires constructing a new function.
See [Array and precision conventions](numerical_usage.md) for
ownership details.

## Calling conventions, results and supported inputs

The mathematical definitions above are the same for allocating and mutating
calls. In Julia, the `!` suffix indicates mutation, **not** a guarantee
that the implementation performs no allocations.

| Mathematical operation | Allocating call | Mutating call | Mutating return |
|:--|:--|:--|:--|
| Gradient | `gradient(f,x)` | `gradient!(out,f,x)` | `out` |
| Value and gradient | `value_and_gradient(f,x)` | `value_and_gradient!(out,f,x)` | Value at input `x` |
| Proximal point | `prox(g,x,γ)` | `prox!(out,g,x,γ)` | `out` |
| Value and proximal point | none | `value_and_prox!(out,g,x,γ)` | Value at the stored proximal point |

Mutating calls put the **destination first**. For ordinary arrays,
`gradient(f,x)` allocates a dense gradient and `prox(f,x,γ)` allocates
a floating-point result compatible with the supported input type.
Some specialized symmetric-storage calls have different result types;
see the [symmetric iterate guide](symmetric_sparse_iterate.md).

### In-place calls and overlapping memory

Built-in gradient operations require a destination separate from their
input: `gradient!(x,f,x)` is not supported. Proximal maps support the
exact same object for input and output, as in `prox!(x,g,x,γ)`.
Other partially overlapping views are not supported and may produce
incorrect results without warning. Use an independent input copy when
in doubt. Parameter and scratch ownership details are in
[Array and precision conventions](numerical_usage.md).

### Different steps for different entries

For separable penalties, some proximal methods accept a vector or array
of positive steps `γ`, one for each input entry. The mathematical
problem then becomes

```math
\operatorname*{argmin}_z\left\{
g(z)+\sum_i\frac{(z_i-x_i)^2}{2\gamma_i}
\right\}.
```

```@example operations_steps
using ProximalBase
x_steps = [2.0, -1.0]
steps = [0.5, 2.0]
z_steps = prox(NormL1(0.5), x_steps, steps)
@assert z_steps == [1.75, 0.0]
z_steps
```

An array of steps is **not** universally supported. Generic
`SeparableFunction` array proximal operations and `ZeroFunction`
support it; `Scaled` forwards it only when the wrapped function does.
Do not assume the same for `SqrNormL2`, `LinearFunction`,
`LeastSquaresLoss`, or `QuadraticFunction`. A
`SymmetricSparseIterate` output accepts scalar steps only.

### Shapes, errors and precision

Each concrete function defines which vector, matrix and storage inputs
it supports; a `true` trait does not guarantee every argument shape.
Built-in methods check predictable shape, type and parameter errors
before output writes, but an error during computation may leave a
partially changed destination. The built-in functions, their constructors,
the screens, `norm_diff`, `opnorm2` and the products of the iterates require
one-based arrays. Such a call checks the axes of its array arguments when it
starts, reading axes metadata and no entries of the arrays it checks: an array
with offset axes raises an `ArgumentError` before the numerical kernels read the
arrays by position and before any write to an output or a cache. The group
constructors read the list of groups to reach each group's axes. `restrict`
indexes the stored parameter arrays with the index set, and the constructor of
the restricted function rejects a result with offset axes; a function with no
stored parameter array, such as `ElasticNet(λ1, λ2)` without weights, does not
inspect the index set. Offset axes are reported as a `DimensionMismatch`
by the point argument of `BlockIterate(x, groups)` and `SymmetricSparseIterate(X)`,
by `copyto!` into a `SymmetricSparseIterate`, and by an offset `z`, `r` or `d` at the
supported strided arguments of the least-squares predictor and residual oracles
at a one-based response; an offset gradient output `g` of those oracles, and the
group arguments of the group constructors, raise an `ArgumentError`, and a
representation the oracles do not support is a `MethodError`. For a function
defined outside this package, the generic `value_and_prox!` and `gradient!`
hand the arrays to its own methods without a one-based check, except where this
package symmetrizes the argument for a `SymmetricSparseIterate` output, which
checks the arrays that it reads; `Scaled`, the kernels of a `SeparableFunction`
and the allocating calls `prox`, `gradient` and `value_and_gradient` check the
arrays of such a function. Positive finite steps are the
standard mathematical setting. Exceptional `NaN`, infinite and zero
steps are function-specific rather than universally validated.
See [Array and precision conventions](numerical_usage.md),
[Numerical contract](numerics.md), and the
[API reference](api/interface.md) for precise limits.

### Symmetric matrices and coordinate updates

Proximal operators on symmetric iterates use the Frobenius geometry
of the full represented matrix. Off-diagonal independent coordinates
appear twice in the matrix norm. Supported functions have explicit
methods for this storage type; not every gradient or proximal
operation accepts it. See [SymmetricSparseIterate](symmetric_sparse_iterate.md)
and [Symmetric parameters](symmetric_parameters.md).

Scalar `prox_block` works on a supplied trial value, not an iterate:
the caller must separately write the returned result to the appropriate
entry or block and update its own solver state. For precise indexing and
ownership rules, see [Coordinate oracles](iterate_oracles.md).
