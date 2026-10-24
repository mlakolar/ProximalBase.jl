# Penalties

Below, ``S_t(v) = \operatorname{sign}(v)\max(|v| - t, 0)`` is soft thresholding
and ``\gamma`` is the step of the proximal operator.

| Function | Value | Proximal operator | Separable |
|:---------|:------|:------------------|:---------:|
| [`NormL1`](@ref)`(λ0, w)` | ``\lambda_0\sum_i w_i\lvert x_i\rvert`` | ``S_{\gamma\lambda_0 w_i}(v_i)`` | ✓ |
| [`NormL1Nonneg`](@ref)`(λ0, w)` | ``\lambda_0\sum_i w_i x_i`` on ``x \ge 0`` | ``\max(v_i - \gamma\lambda_0 w_i, 0)`` | ✓ |
| [`ElasticNet`](@ref)`(λ1, λ2, w)` | ``\lambda_1\sum_i w_i\lvert x_i\rvert + \tfrac{\lambda_2}{2}\lVert x\rVert^2`` | ``S_{\gamma\lambda_1 w_i}(v_i)/(1 + \gamma\lambda_2)`` | ✓ |
| [`NormL0`](@ref)`(λ)` | ``\lambda\,\#\{i : x_i \ne 0\}`` | hard thresholding at ``\sqrt{2\gamma\lambda}`` | ✓ |
| [`MCP`](@ref)`(λ, a)` | the minimax concave penalty | firm thresholding | ✓ |
| [`SCAD`](@ref)`(λ, a)` | the smoothly clipped absolute deviation | SCAD thresholding | ✓ |
| [`NormL2`](@ref)`(λ)` | ``\lambda\lVert x\rVert_2`` | ``\max(1 - \gamma\lambda/\lVert v\rVert_2, 0)\,v`` | |
| [`NormLinf`](@ref)`(λ)` | ``\lambda\lVert x\rVert_\infty`` | ``v - P_{\lVert\cdot\rVert_1 \le \gamma\lambda}(v)`` | |
| [`GroupNormL2`](@ref)`(λ0, groups, w)` | ``\lambda_0\sum_k w_k\lVert x_{G_k}\rVert_2`` | block soft thresholding of each group | |
| [`GroupNormLinf`](@ref)`(λ0, groups, w)` | ``\lambda_0\sum_k w_k\lVert x_{G_k}\rVert_\infty`` | ``v_{G_k} - P_{\lVert\cdot\rVert_1 \le \gamma\lambda_0 w_k}(v_{G_k})`` | |
| [`NormL1Fused`](@ref)`(λ1, λ2)` | ``\sum_i \lambda_1(\lvert a_i\rvert + \lvert b_i\rvert) + \lambda_2\lvert a_i - b_i\rvert`` | fuse each pair, then soft threshold | |
| [`TotalVariation1D`](@ref)`(λ)` | ``\lambda\sum_i\lvert x_{i+1} - x_i\rvert`` | Condat's direct algorithm | |
| [`FusedLasso`](@ref)`(λ1, λ2)` | ``\lambda_1\lVert x\rVert_1 + \lambda_2\sum_i\lvert x_{i+1} - x_i\rvert`` | ``S_{\gamma\lambda_1}`` of the total-variation denoising | |
| [`SLOPE`](@ref)`(λ)` | ``\sum_i\lambda_i\lvert x\rvert_{(i)}`` | pool adjacent violators on the sorted magnitudes | |
| [`NuclearNorm`](@ref)`(λ)` | ``\lambda\sum_i \sigma_i(X)`` | ``U\max(\Sigma - \gamma\lambda, 0)V^\top`` | |
| [`TracePSD`](@ref)`(λ)` | ``\lambda\operatorname{tr}(X)`` on ``X \succeq 0`` | ``U\max(\Lambda - \gamma\lambda, 0)U^\top`` | |

## The ℓ1 norm and its relatives

For [`NormL1`](@ref), ``g(x) = \lambda_0\sum_i w_i|x_i|``, the problem defining
the proximal operator separates over the coordinates. Each scalar problem
``\min_z \gamma\lambda_0 w_i|z| + \tfrac12(z - v_i)^2`` is solved by soft
thresholding at ``\gamma\lambda_0 w_i``. A weight ``w_i = 0`` leaves the
coordinate unpenalized, as for an intercept. The subdifferential of
``|\cdot|`` at zero is ``[-1, 1]``, so given the gradient ``g_i`` of the smooth
part, the coordinate ``x_i = 0`` is optimal exactly when
``|g_i| \le \lambda_0 w_i``. [`kkt_violation`](@ref) returns
``|g_i| - \lambda_0 w_i``: positive when the coordinate should leave zero.

[`NormL1Nonneg`](@ref) adds the constraint ``x \ge 0``, which turns soft
thresholding one-sided. [`ElasticNet`](@ref) adds ``\tfrac{\lambda_2}{2}\|x\|^2``.
Completing the square, its proximal operator is soft thresholding followed by
the shrinkage ``1/(1 + \gamma\lambda_2)``.

## Nonconvex penalties

**ℓ0.** The scalar problem ``\min_z \gamma\lambda\,\mathbf{1}[z \ne 0] + \tfrac12(z - v)^2``
compares ``z = 0``, of value ``\tfrac12 v^2``, with ``z = v``, of value
``\gamma\lambda``. So the proximal operator keeps ``v`` when
``|v| > \sqrt{2\gamma\lambda}`` and sets it to zero otherwise.

**MCP** (Zhang, 2010). The penalty is
``p(t) = \lambda|t| - t^2/(2a)`` for ``|t| \le a\lambda`` and ``a\lambda^2/2``
beyond, with ``a > 1``. For ``\gamma < a`` the scalar problem is convex, and its
solution is *firm thresholding*:

```math
\operatorname{prox}_{\gamma p}(v) =
\begin{cases}
0 & |v| \le \gamma\lambda,\\
\operatorname{sign}(v)\,\dfrac{|v| - \gamma\lambda}{1 - \gamma/a} & \gamma\lambda < |v| \le a\lambda,\\
v & |v| > a\lambda .
\end{cases}
```

Large values pass untouched, so MCP does not shrink large coefficients the way
the ℓ1 norm does.

**SCAD** (Fan and Li, 2001). The penalty is ``\lambda|t|`` for ``|t| \le \lambda``,
``(2a\lambda|t| - t^2 - \lambda^2)/(2(a - 1))`` for ``\lambda < |t| \le a\lambda``,
and ``(a + 1)\lambda^2/2`` beyond, with ``a > 2``. For ``\gamma < a - 1``,

```math
\operatorname{prox}_{\gamma p}(v) =
\begin{cases}
S_{\gamma\lambda}(v) & |v| \le (1 + \gamma)\lambda,\\
\dfrac{(a - 1)v - \operatorname{sign}(v)\,a\gamma\lambda}{a - 1 - \gamma} & (1 + \gamma)\lambda < |v| \le a\lambda,\\
v & |v| > a\lambda ,
\end{cases}
```

which for ``\gamma = 1`` is Fan and Li's thresholding rule. Each piece comes
from setting the derivative of ``\gamma p(z) + \tfrac12(z - v)^2`` to zero on
the corresponding interval.

For larger steps the scalar problems of MCP and SCAD are not convex.
[`prox_entry`](@ref) still returns an exact minimizer: it compares the candidate
minimizers of the pieces. So these penalties work in coordinate descent and
proximal gradient methods with any step. At zero their subdifferential is that
of ``\lambda|\cdot|``, so `kkt_violation` is ``|g_i| - \lambda``.

## Norms of vectors and groups

**ℓ2.** For ``g = \lambda\|\cdot\|_2``, the proximal point lies on the segment
from ``0`` to ``v``. Minimizing over its length gives *block soft thresholding*,
``\max(1 - \gamma\lambda/\|v\|_2, 0)\,v``, which sets the whole vector to zero
when ``\|v\|_2 \le \gamma\lambda``.

**ℓ∞.** The conjugate of ``\lambda\|\cdot\|_\infty`` is the indicator of the
ℓ1 ball of radius ``\lambda``. So by Moreau's decomposition its proximal operator
is ``v - P_{\|\cdot\|_1 \le \gamma\lambda}(v)``, through the ℓ1-ball projection
of [Sets](@ref).

**Groups.** [`GroupNormL2`](@ref)`(λ0, groups, w)` is
``\lambda_0\sum_k w_k\|x_{G_k}\|_2`` over disjoint groups, the group lasso
penalty. By default ``w_k = \sqrt{|G_k|}``, and indices in no group are
unpenalized. Its proximal operator applies block soft thresholding with
threshold ``\gamma\lambda_0 w_k`` to each group. A group at zero is optimal when
``\|g_{G_k}\|_2 \le \lambda_0 w_k``. [`GroupNormLinf`](@ref) uses the ℓ∞ norm of
each group, with the ℓ1-ball projection, and its optimality test is
``\|g_{G_k}\|_1 \le \lambda_0 w_k``. On an `AtomIterate` that shares their
groups, these penalties update one group at a time.

The rows of a ``p \times k`` matrix are the groups
`[i:p:p*k for i in 1:p]` of its linear indices. So `GroupNormL2` with these
groups is the row-sparse penalty of multitask regression; see the multitask
example.

**Fused pairs.** [`NormL1Fused`](@ref)`(λ1, λ2)` acts on pairs of arrays
``(a, b)``: ``\sum_i \lambda_1(|a_i| + |b_i|) + \lambda_2|a_i - b_i|``, the
penalty of the fused graphical lasso for two groups. Its proximal operator first
solves the fusion part, moving ``a_i`` and ``b_i`` toward each other by
``\gamma\lambda_2`` or to their mean. It then soft thresholds both by
``\gamma\lambda_1``. Composing the two proximal operators this way is exact
(Friedman, Hastie, Höfling and Tibshirani, 2007).

## Penalties across stacked matrices

The joint estimation of ``K`` related matrices, such as the precision matrices
of ``K`` classes or time points, works on an ``m \times n \times K`` array
``x``. The fiber ``x_{ij} = (x^1_{ij}, \dots, x^K_{ij})`` holds entry
``(i, j)`` of every matrix. [`FiberSum`](@ref)`(f; diagonal = f)` is
``\sum_{ij} f(x_{ij})``, and its proximal operator applies that of ``f`` to
each fiber. `diagonal` gives the fibers ``x_{ii}`` of square slices their own
function, and `nothing` leaves them unpenalized.

| penalty on ``x_{ij}`` | model | `f` |
|---|---|---|
| ``\lambda\|x_{ij}\|_2`` | a shared sparsity pattern, the group graphical lasso | `NormL2(λ)` |
| ``\lambda\|x_{ij}\|_\infty`` | simultaneous selection | `NormLinf(λ)` |
| ``\lambda_1(|x^1_{ij}| + |x^2_{ij}|) + \lambda_2|x^1_{ij} - x^2_{ij}|`` | the fused graphical lasso of two classes | `NormL1Fused(λ1, λ2)` |

[`SliceSum`](@ref)`(f₁, …, f_K)` is ``\sum_k f_k(x^k)`` for the slices
``x^k = x[:, :, k]``, and its value, gradient and proximal operator act slice by
slice. With ``f_k`` the log-likelihood `LogDetLoss(Sₖ, nₖ/n)` of class ``k``,
the joint graphical lasso minimizes `SliceSum(f₁, …, f_K)` plus a `FiberSum`.
ADMM splits the two: its ``x``-update takes ``K`` eigendecompositions, and its
``z``-update is the proximal operator of the penalty, fiber by fiber. In the
fused graphical lasso of Danaher, Wang and Witten (2014), with the diagonal
fused but not shrunk, the penalty is
`FiberSum(NormL1Fused(λ1, λ2); diagonal = NormL1Fused(0, λ2))`.

On a matrix, the fibers are the rows and the slices the columns.
`FiberSum(NormL2(λ))` is then the row-sparse penalty of multitask regression,
without the index vectors of `GroupNormL2`.

## Total variation, the fused lasso and SLOPE

These penalties depend on the order of the entries, and their proximal
operators pool adjacent entries into blocks of equal values.

**Total variation.** The proximal operator of
``\lambda\sum_i |x_{i+1} - x_i|`` is total-variation denoising. Its solution
``z`` is piecewise constant. Its optimality conditions say that the cumulative
sums of the residual ``v - z`` stay within ``\pm\gamma\lambda`` and end at zero, so
the cumulative sums of ``z`` are a taut string in a tube around those of ``v``.
Condat's algorithm (2013) scans ``v`` once, keeping bounds on the value of the
current segment, and goes back only when a segment ends. It is exact, works in
place, and takes O(n) time on most inputs, O(n²) at worst.

**The fused lasso.** For ``\lambda_1\|x\|_1 + \lambda_2\sum_i|x_{i+1} - x_i|``,
soft thresholding the denoised vector by ``\gamma\lambda_1`` gives the proximal
operator (Friedman, Hastie, Höfling and Tibshirani, 2007): thresholding keeps
the blocks of equal values, so their optimality conditions still hold.

**SLOPE.** The sorted ℓ1 norm ``J_\lambda(x) = \sum_i \lambda_i|x|_{(i)}``, with
``\lambda_1 \ge \dots \ge \lambda_n \ge 0`` and ``|x|_{(1)} \ge |x|_{(2)} \ge \cdots``,
is the ℓ1 norm when the ``\lambda_i`` are equal and the ℓ∞ norm when
``\lambda = (\lambda_1, 0, \dots, 0)``. With the Benjamini–Hochberg sequence
``\lambda_i = \sigma\Phi^{-1}(1 - iq/(2p))``, SLOPE controls the false discovery
rate of the selected variables for orthogonal designs (Bogdan, van den Berg,
Sabatti, Su and Candès, 2015). The proximal operator keeps the signs and the
order of the magnitudes of ``v``. With ``|v|`` sorted in decreasing order, it is
the nonincreasing, nonnegative sequence nearest to ``|v|_{(i)} - \gamma\lambda_i``:
isotonic regression by pool adjacent violators, clipped at zero, in
O(n log n) time for the sort. Entries pooled into a block come out with equal
magnitudes, which is how SLOPE groups correlated predictors.

```@example penalties
using ProximalBase
v = [3.0, -1.0, 0.5, 2.0]
prox(SLOPE([1.0, 0.8, 0.5, 0.2]), v, 1.0)    # magnitudes 3, 2, 1, 0.5 minus λ
```

## Spectral penalties

For a function of the singular values, ``F(X) = f(\sigma(X))``, with ``f``
absolutely symmetric, the proximal operator acts on the singular values:
``\operatorname{prox}_{\gamma F}(X) = U\operatorname{diag}(\operatorname{prox}_{\gamma f}(\sigma))V^\top``
for ``X = U\operatorname{diag}(\sigma)V^\top`` (Lewis, 1995). For
[`NuclearNorm`](@ref), ``f = \lambda\|\cdot\|_1``, which gives *singular value
thresholding*. [`TracePSD`](@ref), ``\lambda\operatorname{tr}(X)`` on the positive
semidefinite matrices, is the analogous function of the eigenvalues: it
subtracts ``\gamma\lambda`` from each eigenvalue and clips at zero. It is the
penalty on the low-rank part of latent-variable graphical models.
[`value_and_prox!`](@ref) gets the value from the thresholded spectrum. The
decompositions run in LAPACK workspaces that the functions own.
[`Spectral`](@ref)`(f)` builds the function of the eigenvalues of a symmetric
matrix from any function ``f`` of a vector that does not depend on the order of
its entries.
