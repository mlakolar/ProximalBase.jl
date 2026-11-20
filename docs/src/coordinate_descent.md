# Coordinate descent

*Assumes: derivatives, quadratic upper models and KKT conditions · You will learn: scalar and block updates, residual maintenance and working-set screening.*

Coordinate methods update one entry or one ordered block at a time.
A [working set](glossary.md#Working-set) is the subset selected for
updates and screening; it is not necessarily the nonzero support. The caller
constructs a trial, requests a pure proximal result, writes the result and
updates its residual/cache. ProximalBase supplies the function and storage
operations; the schedule and curvature calculations below are caller code.

## A scalar coordinate update

For ``F(x)=f(x)+\sum_jg_j(x_j)``, suppose ``a_j>0`` satisfies

```math
f(x+(t-x_j)e_j)\le f(x)+\nabla_j f(x)(t-x_j)+\frac{a_j}{2}(t-x_j)^2.
```

Completing the square in the model gives

```math
x_j^+=\operatorname{prox}_{g_j/a_j}
\left(x_j-\frac{\nabla_j f(x)}{a_j}\right).
```

The scalar call is `prox_block(g,trial,j,1/a[j])`. It accepts a
numerical trial and the global entry index, and does not mutate storage.
ProximalBase does not provide a coordinate-curvature helper; compute a valid
curvature from the model. Handle a zero curvature explicitly before inversion.

## Least squares and residual updates

For ``f(\beta)=\|y-X\beta\|^2/(2n)``, maintain ``r=y-X\beta``. Then

```math
\nabla_j f(\beta)=-X_{:,j}^\top r/n,\qquad
 a_j=\|X_{:,j}\|^2/n,
```

and for weighted `NormL1`,

```math
\beta_j^+=S_{\lambda_0w_j/a_j}
\left(\beta_j+\frac{X_{:,j}^\top r}{n a_j}\right),\qquad
r^+=r-X_{:,j}(\beta_j^+-\beta_j).
```

For a nonzero column this exactly minimizes the quadratic objective along
that coordinate. Each update costs O(n) for a dense column. The residual must
be updated immediately so later coordinates see the current point.

```@example cd_single
using ProximalBase, LinearAlgebra
X = [1.0 0;0 2;1 1]
y = [2.0,-1,1]
n = length(y)
f = (1/n)*LeastSquaresLoss(y,X)
g = NormL1(0.2)
β = SparseIterate(2)
r = copy(y)
j = 1
a = sum(abs2,view(X,:,j))/n
old = β[j]
trial = old+dot(view(X,:,j),r)/(n*a)
β[j] = prox_block(g,trial,j,1/a)
r .-= view(X,:,j).*(β[j]-old)
@assert r ≈ y-X*β
@assert f(β)+g(β) <= f(zeros(2))
Vector(β)
```

See [the full coordinate example](examples/coordinate_operations.md) for
repeated sweeps checked against the KKT conditions.

## Optimality and working sets

For convex differentiable ``f`` and
[proper closed convex](glossary.md#Proper-closed-convex-function) entry functions,
``0\in\nabla_jf(x)+\partial g_j(x_j)`` for every j characterizes a minimizer.
Here [the subdifferential](glossary.md#Subgradient-and-subdifferential)
replaces the ordinary derivative at nonsmooth points.
The weighted lasso conditions are

```math
\begin{cases}
\nabla_jf(x)+\lambda_0w_j\operatorname{sign}(x_j)=0,&x_j\ne0,\\
|\nabla_jf(x)|\le\lambda_0w_j,&x_j=0.
\end{cases}
```

`kkt_violation` has three forms, with `v` the vector that the caller supplies,
usually the gradient `grad` of the smooth part at `x`; it is not recomputed.

| Call | What it evaluates | Sign |
|:--|:--|:--|
| `kkt_violation(g,v,j)` | The screen of coordinate `j` **assumed zero**: for `NormL1`, ``\lvert v_j\rvert-\lambda_0w_j``; it reads no `x` | Signed |
| `kkt_violation(g,v,x,k)` | The score of the **actual** coordinate `k` of `x` (the group `k` for a `BlockIterate`) | The signed margin at a zero coordinate; a nonnegative mismatch at a nonzero one |
| `kkt_violation(g,v,x)` | ``\max(0,\max_k s_k)`` over the entries of `x`, with ``s_k`` the score of entry ``k`` | Nonnegative |

A **positive** score at a zero coordinate says that zero violates its
condition, so the coordinate is a working-set candidate. A **positive** score
at a nonzero coordinate is a stationarity mismatch; it is not an instruction
to set the coordinate to zero. A **negative** score occurs only at a zero
coordinate (or block) and is slack.

For an entry with ``c_j=\lambda_0w_j``, `NormL1` returns ``\lvert v_j\rvert-c_j`` at
a zero coordinate and ``\lvert v_j+c_j\operatorname{sign}(x_j)\rvert`` at a nonzero one.
`ElasticNet` returns the same at a zero coordinate with ``c_j=\lambda_1w_j``, and
``\lvert(v_j+\lambda_2x_j)+c_j\operatorname{sign}(x_j)\rvert`` at a nonzero one. For these two
the full-point value is the ``\ell_\infty`` distance from ``-v`` to the
subdifferential ``\partial g(x)`` when the inputs are finite. For finite inputs, `NormL1Nonneg` returns
``-v_j-c_j`` at a zero coordinate and ``\lvert v_j+c_j\rvert`` at a positive one; the rules for infeasible and
non-finite points are in the [numerical contract](numerics.md). For `ZeroFunction`,
`LinearFunction` and `SqrNormL2` the score is ``\lvert v_j+d_j\rvert`` at zero and
nonzero coordinates alike, with ``d_j`` the entry of ``\nabla g(x)`` (``0`` for
`ZeroFunction`, the coefficient for `LinearFunction`), and the full-point value
is ``\lVert v+\nabla g(x)\rVert_\infty``. The table of
[Mathematical operations](operations.md#Checking-optimality-with-KKT-conditions)
lists each scalar.

```@example cd_kkt
using ProximalBase
g = NormL1(0.5)
x = [0.0, 0.0, 2.0, -1.0]
v = [0.8, 0.2, 0.0, 0.5]
screen = [kkt_violation(g, v, j) for j in 1:4]
scores = [kkt_violation(g, v, x, j) for j in 1:4]
(screen = screen, scores = scores, full_point = kkt_violation(g, v, x))
```

At coordinate 3 the screen assumes ``x_3=0`` and returns ``-0.5``; the
four-argument form reads ``x_3=2`` and returns ``0.5``. At coordinates 1 and 2
the two agree, ``0.8-0.5>0`` (a candidate) and ``0.2-0.5<0`` (slack), up to
rounding, so compare them with `≈`:

```@example cd_kkt
@assert screen[3] == -0.5 && scores[3] == 0.5 && scores[4] == 0
@assert scores[1] ≈ 0.3 && screen[1] ≈ 0.3
@assert scores[2] ≈ -0.3 && screen[2] ≈ -0.3
@assert kkt_violation(g, v, x) == 0.5
```

ProximalBase returns the raw score. Scaling, the tolerance, when to evaluate
the score and the decision that a solver has converged belong to the caller.
A solver that ranks inactive coordinates calls the three-argument or the
four-argument form at coordinates of value zero; both give the signed screen
there when `x`, `v` and the function's parameters (levels, weights, centers and
coefficients) are `Float32` or `Float64` arrays and values of one type, `v` is finite and a `SqrNormL2` level is not infinite. On a `BlockIterate`, the
four-argument form takes the maximum of the entry scores of the group for the
functions above, and `GroupNormL2` scores the group (see
[The ℓ₂ and group ℓ₂ norms](group_penalty.md)). `GroupNormLinf` gives the ℓ₁ screen `‖v_G‖₁ - λ0 w[k]` of its dual norm at a zero group
([#634](https://github.com/mlakolar/ProximalBase.jl/issues/634)), not the Euclidean screen of `GroupNormL2`, and the Euclidean distance from `-v_G` to
the subdifferential at a nonzero group; the positive part of the screen is not a stationarity distance, and the full-point form scores a zero group by its
distance to the ℓ₁ ball and not by its screen.

A working-set procedure optimizes selected coordinates, computes a full
gradient to screen excluded zeros, adds violations, and repeats until both
included and excluded coordinates satisfy their conditions. A restricted solve
alone cannot certify the full problem. Stored coordinates include explicit
zeros and need not equal the current mathematical support.

`restrict(f,S)` and `restrict(g,S)` slice supported function parameters, with
local entry order given by S. For least squares and quadratics, they represent omitted variables fixed at
zero. Centered `SqrNormL2` slices its center and drops the omitted-coordinate
constant ``(λ/2)\sum_{j\notin S}y_j^2``. When that omitted contribution is finite, this
changes objective values but not the restricted minimizers, gradients or prox;
add the constant back when comparing full and restricted values. Nonzero fixed
variables require a separately derived residual/linear term.
Only vector-response `LeastSquaresLoss` has a restriction method.
See [Restricting functions](restrict.md) for full mathematical definitions,
nonseparable examples and the distinction between zero-fixed and nonzero-fixed
subproblems, or the [working-set example](examples/sparse_working_sets.md).

## Block updates

For a partition ``G_k`` and a block curvature ``L_k>0`` majorizing changes in
that block, a compatible decomposed penalty gives

```math
x_{G_k}^+=\operatorname{prox}_{g_k/L_k}
\left(x_{G_k}-\nabla_{G_k}f(x)/L_k\right).
```

Least squares can use ``L_k=\|X_{:,G_k}\|_2^2/n``. This generally gives one
upper-model step, rather than an exact block least-squares solution. Scalar
column curvatures alone are not a block spectral bound.

`BlockIterate` supplies an ordered partition and public block views. A separable
penalty uses scalar `prox_block` at each global member index. A custom block
penalty supplies the compact-vector extension hook; the built-in
[`GroupNormL2`](@ref), [`GroupNormLinf`](@ref) and [`SparseGroupNormL2`](@ref) implement it.
See [coordinate oracles](iterate_oracles.md),
the [block extension](examples/block_prox_extension.md) and the
[group-lasso building blocks](examples/group_lasso_prox.md). Matching layouts,
constructing trials and updating caches are the caller's responsibilities.
