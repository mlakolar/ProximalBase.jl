# Operations on functions

ProximalBase functions are mathematical objects. An **operation** asks a
function to compute something useful for an optimization algorithm: its value,
gradient, proximal point, a coordinate-level update, or information about
optimality and step sizes. This page explains what each operation means.

Here is a common problem with a smooth loss and a possibly nonsmooth penalty:

```math
\min_x\; F(x)=f(x)+g(x).
```

An algorithm may evaluate the objective to monitor progress, use a gradient
of ``f`` to move downhill, and use a proximal map of ``g`` to encourage
sparsity. ProximalBase computes these ingredients; it does not choose an
algorithm or stopping rule.

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
separable function). A custom group penalty can instead implement
`prox_block!(out,g,u,k,γ)` on compact vector buffers. These hooks
compute mathematical answers; **they do not update an iterate's active
set or write back to solver state**. See [Coordinate oracles](iterate_oracles.md)
and [Writing your own function](extending.md) for precise contracts.

## Checking optimality with KKT conditions

**Question:** Is a coordinate that we have kept at zero allowed to
remain zero at the optimum?

For a convex differentiable loss ``f`` and an ``\ell_1`` penalty
``g(x)=\lambda\|x\|_1``, the optimality condition for a **zero**
coordinate ``i`` is

```math
|\nabla_i f(x)|\leq\lambda.
```

The scalar `kkt_violation(g,grad,i)` measures
``|\mathrm{grad}_i|-\lambda`` for `NormL1` (with its
appropriate coordinate weight). A **positive** result signals that
the zero-coordinate condition is violated:

```@example operations_basics
gradient_at_zero = [0.8, 0.2]
(kkt_violation(g, gradient_at_zero, 1),
 kkt_violation(g, gradient_at_zero, 2))
```

For this penalty with ``\lambda=0.5``, the result is
``(0.3,-0.3)``: the first zero coordinate fails the test; the second
passes it. This is **not** a general optimality certificate for a
nonzero coordinate or the entire vector. The iterate-aware variant
`kkt_violation(g,grad,x,k)` interprets `k` in the iterate's
coordinate or block space. See [Coordinate descent](coordinate_descent.md).

## Restricting a function to selected coordinates

**Question:** Can an algorithm form a smaller function involving only
selected variables?

For supported functions, `restrict(f,S)` constructs the corresponding
reduced mathematical object for an integer index set `S`.
For example, a separable penalty restricted to coordinates
``S\subseteq\{1,\ldots,p\}`` keeps their penalty terms:

```math
g_S(u)=\sum_{j=1}^{|S|}g_{S_j}(u_j).
```

```@example operations_basics
gr = restrict(g, [1])
@assert gr([2.0]) == g([2.0, 0.0])
```

This is useful for working-set algorithms that solve smaller subproblems.
However, `restrict` is **not** a universal way to construct
``f(x_0+U_Su)`` around a current point ``x_0``: constants and
offsets may need to be accounted for by the caller. Support is
function-specific; see [The interface](interface.md) for the details.

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

`opnorm2(X)` estimates the **squared spectral norm**
``\|X\|_2^2``, related to a least-squares gradient Lipschitz
constant. It is an **iterative estimate**, not a certified upper
bound: do not rely on it alone to guarantee step-size safety.
Use a verified bound or backtracking when necessary.

## Invalidating cached calculations

**Question:** What if I modify data already stored by reference
inside a function object?

Some function objects reuse matrix decompositions to speed up
repeated proximal evaluations. If the data change but their shape stays
the same, `reset_cache!(f)` invalidates any such cached decomposition
so a later computation can rebuild it. This is a **lifecycle operation**,
not a new mathematical function.

```julia
# If X is referenced by a LeastSquaresLoss and its entries change:
# X[1, 1] = new_value
# reset_cache!(loss)
```

Changing data dimensions requires constructing a new function.
See [Array and precision conventions](numerical_usage.md) for
ownership details.

## Julia calling conventions and technical reference

The mathematics above is independent of whether a call is allocating
or mutating. Here is a quick comparison:

| Mathematical task | Allocating call | Reuse an existing output |
|:--|:--|:--|
| Gradient | `gradient(f,x)` | `gradient!(out,f,x)` |
| Value and gradient | `value_and_gradient(f,x)` | `value_and_gradient!(out,f,x)` |
| Proximal point | `prox(g,x,γ)` | `prox!(out,g,x,γ)` |
| Value and proximal point | No allocating counterpart | `value_and_prox!(out,g,x,γ)` |

All mutating oracles take the **destination first**. Julia's `!`
indicates mutation, not zero allocation. Supported shapes, returned values,
input/output overlap and per-coordinate steps are specified in
[The interface](interface.md). For lower-level extension and numerical
contracts, see [Developer Documentation](developer/index.md).
