# Coordinate descent

Coordinate descent updates one entry or one group of entries at a time.
We use the scalar and block proximal operations, coordinate curvatures,
sparse iterates and optimality checks supplied by ProximalBase to construct
these procedures. In this section, we derive the updates and explain how they enter
an active-set procedure.

## A scalar coordinate update

We consider

```math
\min_x\; F(x)=f(x)+\sum_{j=1}^p g_j(x_j),
```

where ``f`` is differentiable and the penalty is separable. Fix all entries
except ``x_j``, and choose ``a_j>0`` such that

```math
f\bigl(x+(t-x_j)e_j\bigr)
\le f(x)+\nabla_j f(x)(t-x_j)+\frac{a_j}{2}(t-x_j)^2
```

for the candidate values of ``t``. A coordinate Lipschitz constant gives this
bound along the whole coordinate line. Minimizing the bound plus ``g_j`` and
completing the square gives

```math
\begin{aligned}
x_j^+
&=\operatorname*{argmin}_t\left\{
  \nabla_j f(x)(t-x_j)+\frac{a_j}{2}(t-x_j)^2+g_j(t)\right\}\\
&=\operatorname{prox}_{g_j/a_j}
  \left(x_j-\frac{\nabla_j f(x)}{a_j}\right).
\end{aligned}
```

For a quadratic loss with exact coordinate curvature, this minimizes the
original objective along that coordinate. For a general smooth loss, we
instead minimize a quadratic upper bound, which gives a proximal coordinate step.
In either case, a valid upper bound and an exact solution of the scalar
problem give ``F(x^+)\le F(x)``.

Using [`prox_entry`](@ref)`(g, v, j, γ)`, we compute the new scalar value
before deciding whether to write it to the iterate.
[`coordinate_lipschitz`](@ref)`(f, j)` returns a bound for entry ``j``, and
[`coordinate_lipschitz!`](@ref) fills an array of them.
For least squares we have ``a_j=\|X_{:,j}\|_2^2``, while a loss without a
finite global bound needs a local upper bound or a line search.
If a bound is zero, choose a positive majorizing curvature or handle that
coordinate separately before taking its reciprocal.

## Least squares and residual updates

For ``f(\beta)=\|y-X\beta\|_2^2/2``, let ``r=y-X\beta``. Then

```math
\nabla_j f(\beta)=-X_{:,j}^\top r,
\qquad a_j=\|X_{:,j}\|_2^2,
```

so the weighted lasso penalty ``g_j(t)=\lambda w_j|t|`` gives

```math
\beta_j^+=S_{\lambda w_j/a_j}
\left(\beta_j+\frac{X_{:,j}^\top r}{a_j}\right),
\qquad
r\leftarrow r-X_{:,j}(\beta_j^+-\beta_j).
```

Here ``S_t(v)=\operatorname{sign}(v)\max(|v|-t,0)``. We maintain the residual
after every update, so later coordinates in the sweep use the current
coefficients. A dense column costs ``O(n)`` per update, and a full sweep costs
``O(np)``. See [Getting started](@ref) for an executable implementation and
the [lasso-path example](examples/02_lasso_path_coordinate_descent.md) for
warm starts across penalty levels.

## Block coordinate updates

Let ``G_1,\ldots,G_K`` partition the entries, and write
``g(x)=\sum_k g_k(x_{G_k})``. Suppose that changing block ``G_k`` by ``d``
satisfies

```math
f(x+U_kd)\le f(x)+\langle\nabla_{G_k}f(x),d\rangle
                  +\frac{L_k}{2}\|d\|_2^2,
```

where ``U_k`` inserts the block into the full vector and ``L_k>0``.
The block proximal update is

```math
x_{G_k}^+=\operatorname{prox}_{g_k/L_k}
\left(x_{G_k}-\frac{\nabla_{G_k}f(x)}{L_k}\right).
```

For least squares, take ``L_k=\|X_{:,G_k}\|_2^2``. The group lasso has
``g_k(u)=\lambda w_k\|u\|_2``, so with
``v=x_{G_k}-\nabla_{G_k}f(x)/L_k`` we obtain

```math
x_{G_k}^+=\left(1-\frac{\lambda w_k}{L_k\|v\|_2}\right)_+v,
```

with value zero when ``v=0``. We use [`GroupNormL2`](@ref) to represent this
penalty, whose default weights are ``w_k=\sqrt{|G_k|}``.
An [`AtomIterate`](@ref) has one coordinate
per group, and `prox_coordinate!(x, g, k, u, 1 / Lk)` applies the block
thresholding at the gradient step `u`, which the calling procedure computes
without writing it into `x`.

Note that this quadratic bound generally differs from the exact block
least-squares loss, whose Hessian is ``X_{:,G_k}^\top X_{:,G_k}``.
When that Hessian is ``L_k I``, the thresholding formula minimizes the exact
block objective, while otherwise it gives one proximal block step.
The [group-lasso example](examples/04_group_lasso.md) implements this step
and compares its solution with proximal gradient.
The scalar entry curvatures from `coordinate_lipschitz` are not
block curvature bounds.

## Optimality and working sets

Suppose that ``f`` is convex and differentiable, and each ``g_j`` is proper,
closed and convex. At a point in the domain, the conditions

```math
0\in\nabla_j f(x)+\partial g_j(x_j),\qquad j=1,\ldots,p,
```

characterize a global minimizer. For the weighted lasso they become

```math
\begin{cases}
\nabla_j f(x)+\lambda w_j\operatorname{sign}(x_j)=0,&x_j\ne0,\\
|\nabla_j f(x)|\le\lambda w_j,&x_j=0.
\end{cases}
```

For a group lasso, a zero group must satisfy
``\|\nabla_{G_k}f(x)\|_2\le\lambda w_k``. A nonzero group must satisfy
``\nabla_{G_k}f(x)+\lambda w_k x_{G_k}/\|x_{G_k}\|_2=0``.
[`kkt_violation`](@ref) evaluates the signed violation at a coordinate assumed
to be zero:

```math
v_j=|\nabla_j f(x)|-\lambda w_j,
\qquad
v_k^{\mathrm{group}}=\|\nabla_{G_k}f(x)\|_2-\lambda w_k.
```

A positive value indicates that the coordinate should enter the working set.
A nonpositive value checks only the condition at zero, rather than the
stationarity of a nonzero coordinate. For entry indices, we use the
three-argument form `kkt_violation(g, grad, j)`. For group coordinates, we call
`kkt_violation(g, grad, x, k)` with an `AtomIterate` sharing the penalty's groups.

A simple working-set procedure has the following steps:

1. Choose a set ``S`` and solve the problem restricted to ``S``, using
   [`restrict`](@ref)`(f, S)` and `restrict(g, S)` where supported.
2. Compute the gradient at the full iterate and check the zero coordinates
   outside ``S`` with `kkt_violation`.
3. Add violating coordinates and repeat until both the restricted solution
   and the excluded coordinates satisfy the optimality conditions.

The [working-set example](examples/14_active_set_lasso.md) implements this
procedure for the lasso, starting from an empty set of predictors.
Note that stored coordinates are candidates for ``S``, and may include zeros
retained from earlier sweeps. See [Sparse iterates](@ref) for the storage operations and
the distinction between entry and coordinate indices, including symmetric
matrix updates.

## Scope of the updates

The scalar and block formulas describe individual updates, so a convergence
claim must also specify the objective and the coordinate schedule.
The least-squares lasso with ``\lambda>0``, nonzero columns and
cyclic exact coordinate minimization is a convex setting in which the
optimality conditions above provide a stopping criterion.

For nonconvex penalties, exact scalar minimization still gives descent when
the quadratic upper bound is valid. A coordinatewise or stationary point may
fail to be a global minimizer. Furthermore, coupled penalties such as total
variation and the nuclear norm require a compatible block update or a
splitting procedure, even though they have full proximal operators.
