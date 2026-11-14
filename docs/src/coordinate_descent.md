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

`kkt_violation(g,grad,j)` reports ``|\mathrm{grad}_j|-\lambda_0w_j`` for a
coordinate assumed zero. It is signed and can be negative. It is not the
stationarity residual at a nonzero coordinate. `kkt_violation(g,grad,x,k)`
uses x's coordinate interpretation and adds specified validation; on a
`BlockIterate` it takes the maximum scalar screen over the chosen group.

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
[`GroupNormL2`](group_penalty.md) implements it. See [coordinate oracles](iterate_oracles.md),
the [block extension](examples/block_prox_extension.md) and the
[group-lasso building blocks](examples/group_lasso_prox.md). Matching layouts,
constructing trials and updating caches are the caller's responsibilities.
