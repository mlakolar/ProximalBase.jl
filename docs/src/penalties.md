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
| [`Berhu`](@ref)`(λ, M, w)` | ``\lambda w_i\lvert x_i\rvert`` up to ``M``, ``\lambda w_i(x_i^2 + M^2)/(2M)`` beyond | soft thresholding, then scaling | ✓ |
| [`NegEntropy`](@ref)`(λ)` | ``\lambda\sum_i x_i\log x_i`` on ``x \ge 0`` | ``\mu\,\omega(v_i/\mu - 1 - \log\mu)``, ``\mu = \gamma\lambda`` | ✓ |
| [`KLDivergence`](@ref)`(q; λ)` | ``\lambda\sum_i x_i\log(x_i/q_i) - x_i + q_i`` on ``x \ge 0`` | ``\mu\,\omega(v_i/\mu + \log(q_i/\mu))`` | ✓ |
| [`NegLog`](@ref)`(λ)` | ``-\lambda\sum_i\log x_i`` on ``x > 0`` | ``(v_i + \sqrt{v_i^2 + 4\gamma\lambda})/2`` | ✓ |
| [`CappedL1`](@ref)`(λ, a, w)` | ``\lambda\sum_i w_i\min(\lvert x_i\rvert, a)`` | the better of two clipped candidates | ✓ |
| [`LogPenalty`](@ref)`(λ, ε, w)` | ``\lambda\sum_i w_i\log(1 + \lvert x_i\rvert/\varepsilon)`` | zero or the root of a quadratic | ✓ |
| [`NormLq`](@ref)`(λ, q, w)` | ``\lambda\sum_i w_i\lvert x_i\rvert^q``, ``0 < q < 1`` | ℓq thresholding | ✓ |
| [`NormL2`](@ref)`(λ)` | ``\lambda\lVert x\rVert_2`` | ``\max(1 - \gamma\lambda/\lVert v\rVert_2, 0)\,v`` | |
| [`NormLinf`](@ref)`(λ)` | ``\lambda\lVert x\rVert_\infty`` | ``v - P_{\lVert\cdot\rVert_1 \le \gamma\lambda}(v)`` | |
| [`GroupNormL2`](@ref)`(λ0, groups, w)` | ``\lambda_0\sum_k w_k\lVert x_{G_k}\rVert_2`` | block soft thresholding of each group | |
| [`SparseGroupNormL2`](@ref)`(λ1, λ2, groups, w)` | ``\lambda_1\lVert x\rVert_1 + \lambda_2\sum_k w_k\lVert x_{G_k}\rVert_2`` | soft thresholding, then block soft thresholding | |
| [`GroupNormLinf`](@ref)`(λ0, groups, w)` | ``\lambda_0\sum_k w_k\lVert x_{G_k}\rVert_\infty`` | ``v_{G_k} - P_{\lVert\cdot\rVert_1 \le \gamma\lambda_0 w_k}(v_{G_k})`` | |
| [`GroupMCP`](@ref)`(λ, groups, w)`, [`GroupSCAD`](@ref) | ``\sum_k p_{\lambda w_k}(\lVert x_{G_k}\rVert_2)`` with the MCP or SCAD ``p`` | the scalar proximal operator of ``p`` at ``\lVert v_{G_k}\rVert_2``, along ``v_{G_k}`` | |
| [`NormL1Fused`](@ref)`(λ1, λ2)` | ``\sum_i \lambda_1(\lvert a_i\rvert + \lvert b_i\rvert) + \lambda_2\lvert a_i - b_i\rvert`` | fuse each pair, then soft threshold | |
| [`TotalVariation1D`](@ref)`(λ)` | ``\lambda\sum_i\lvert x_{i+1} - x_i\rvert`` | Condat's direct algorithm | |
| [`FusedLasso`](@ref)`(λ1, λ2)` | ``\lambda_1\lVert x\rVert_1 + \lambda_2\sum_i\lvert x_{i+1} - x_i\rvert`` | ``S_{\gamma\lambda_1}`` of the total-variation denoising | |
| [`SLOPE`](@ref)`(λ)` | ``\sum_i\lambda_i\lvert x\rvert_{(i)}`` | pool adjacent violators on the sorted magnitudes | |
| [`NuclearNorm`](@ref)`(λ)` | ``\lambda\sum_i \sigma_i(X)`` | ``U\max(\Sigma - \gamma\lambda, 0)V^\top`` | |
| [`TracePSD`](@ref)`(λ)` | ``\lambda\operatorname{tr}(X)`` on ``X \succeq 0`` | ``U\max(\Lambda - \gamma\lambda, 0)U^\top`` | |
| [`CheckLoss`](@ref)`(τ, w)` | ``\sum_i w_i\rho_\tau(x_i)`` | see below | ✓ |
| [`HingeLoss`](@ref)`(y, w)` | ``\sum_i w_i\max(0, 1 - y_i x_i)`` | see below | ✓ |
| [`EpsInsensitiveLoss`](@ref)`(ε, w)` | ``\sum_i w_i\max(0, \lvert x_i\rvert - \varepsilon)`` | see below | ✓ |
| [`SingularValue`](@ref)`(f)` | ``f(\sigma(X))`` | ``U\operatorname{diag}(\operatorname{prox}_{\gamma f}(\sigma))V^\top`` | |

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

**Capped ℓ1** (Zhang, 2010): ``p(t) = \lambda\min(|t|, a)``. On ``|z| \le a``
the scalar problem is the lasso's, so its minimizer is soft thresholding clipped
at ``a``; on ``|z| \ge a`` the penalty is constant, so the minimizer is
``\max(|v|, a)``. The proximal operator is the better of the two.

**Log penalty**: ``p(t) = \lambda\log(1 + |t|/\varepsilon)``, whose linearization
gives reweighted ℓ1 minimization (Candès, Wakin and Boyd, 2008). With
``c = \gamma\lambda`` and ``t = |v|``, the stationary points of
``c\log(1 + z/\varepsilon) + \tfrac12(z - t)^2`` on ``z > 0`` are the roots of
``z^2 + (\varepsilon - t)z + c - \varepsilon t = 0``. The larger is a local
minimum, compared with ``z = 0``. Its slope at zero is ``\lambda/\varepsilon``.

**ℓq**, ``0 < q < 1``: ``p(t) = \lambda|t|^q``. With ``\mu = \gamma\lambda`` and
``\beta = (2\mu(1 - q))^{1/(2 - q)}``, the proximal operator is zero for
``|v| \le \tau = \beta(2 - q)/(2 - 2q)``, where zero and ``\beta`` tie, and above
it the root ``z > \beta`` of ``z + \mu q z^{q - 1} = |v|``, with the sign of ``v``.
The root has a closed form for ``q = 1/2`` (half thresholding: Xu, Chang, Xu and
Zhang, 2012) and ``q = 2/3`` (Cao, Sun and Xu, 2013), and comes from Newton's
method, which converges monotonically from ``|v|``, otherwise. The slope at zero
is infinite, so zero is a local minimizer of every problem and `kkt_violation`
is ``-\infty``.

**Group MCP and group SCAD** (Huang, Breheny and Ma, 2012) apply the MCP or SCAD
with level ``\lambda w_k`` to the Euclidean norm of each group. For any ``p``,
``p(\lVert z\rVert) + \lVert z - v\rVert^2/(2\gamma)`` is smallest, among the ``z``
of a given norm ``r``, at ``z = r\,v/\lVert v\rVert``, so the proximal operator is
the scalar one at ``\lVert v_{G_k}\rVert``, along ``v_{G_k}``: exact for every step.

## Entropy, divergence and barrier functions

The negative entropy, the Kullback–Leibler divergence and the log barrier are
separable, convex, and finite only on the nonnegative (or positive) orthant. For
the first two, the optimality condition of the scalar problem with
``\mu = \gamma\lambda`` is ``z/\mu + \log z = c`` for a constant ``c``. Writing
``z = \mu s`` gives ``s + \log s = c - \log\mu``, whose solution is the Wright
omega function ``\omega(u) = W_0(e^u)``, the Lambert ``W`` function at ``e^u``.
ProximalBase computes ``\omega`` without forming ``e^u``, which overflows when
the step is small and ``u = v/\mu - 1 - \log\mu`` is large, so the proximal
operators stay accurate there. The log barrier's scalar problem is the
quadratic ``z^2 - vz - \gamma\lambda = 0``, whose positive root is the one the
proximal operator of [`LogDetLoss`](@ref) takes for each eigenvalue.

**The reverse Huber penalty** [`Berhu`](@ref) (Owen, 2007) is the ℓ1 norm up to
``M`` and a ridge penalty beyond, with matching slopes at ``\pm M``. It is
convex and keeps the sparsity of the lasso near zero while shrinking large,
correlated coefficients together. Its proximal operator soft-thresholds up to
``|v| = M + \gamma\lambda`` and scales by ``M/(M + \gamma\lambda)`` beyond.

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

**Sparse groups.** [`SparseGroupNormL2`](@ref)`(λ1, λ2, groups, w)` adds an ℓ1
term, ``\lambda_1\|x\|_1 + \lambda_2\sum_k w_k\|x_{G_k}\|_2``, the sparse
group lasso of Simon, Friedman, Hastie and Tibshirani (2013): it selects groups,
and entries within a selected group. Its proximal operator soft thresholds each
entry at ``\gamma\lambda_1`` and then applies block soft thresholding at
``\gamma\lambda_2 w_k`` to each group,

```math
z_{G_k} = \max\Bigl(1 - \frac{\gamma\lambda_2 w_k}{\|S_{\gamma\lambda_1}(v_{G_k})\|_2}, 0\Bigr) S_{\gamma\lambda_1}(v_{G_k}) .
```

Composing them is exact, because soft thresholding preserves the signs and
only shrinks, so the subdifferential of the group norm at its output contains
that at ``z``. A group at zero is optimal when
``\|S_{\lambda_1}(g_{G_k})\|_2 \le \lambda_2 w_k``, and an entry at zero inside
a nonzero group when ``|g_i| \le \lambda_1``. With one group,
`SparseGroupNormL2(λ1, λ2, [1:K], [1.0])`, it is the sparse group lasso of a
single vector: as the fiber function of a [`FiberSum`](@ref) it gives the
penalty of the group graphical lasso (see below).

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
| ``\lambda_1\|x_{ij}\|_1 + \lambda_2\|x_{ij}\|_2`` | the group graphical lasso of Danaher, Wang and Witten (2014), sparse within a fiber too | `SparseGroupNormL2(λ1, λ2, [1:K], [1.0])` |
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

## Losses of residuals and margins

Quantile regression, support vector machines and support vector regression
have losses that are nonsmooth but separable, with closed-form proximal
operators. So splitting methods such as ADMM apply them to ``z = X\beta`` exactly.

**The check loss.** [`CheckLoss`](@ref)`(τ)` is
``\sum_i \rho_\tau(x_i)`` with ``\rho_\tau(u) = \max(\tau u, (\tau - 1)u)``, for
``\tau \in (0, 1)``. Its proximal operator is

```math
\operatorname{prox}_{\gamma\rho_\tau}(v) =
\begin{cases}
v - \gamma\tau & v > \gamma\tau,\\
v + \gamma(1 - \tau) & v < -\gamma(1 - \tau),\\
0 & \text{otherwise}.
\end{cases}
```

Since ``\rho_\tau(-u) = \rho_{1-\tau}(u)``, the loss
``\sum_i \rho_\tau(y_i - z_i)`` of predictions ``z`` is
`Translate(CheckLoss(1 - τ), y)`. The conjugate of ``\rho_\tau`` is the indicator
of ``[\tau - 1, \tau]``, the dual feasible set of quantile regression, and
[`Conjugate`](@ref) gives its value.

**The hinge loss.** [`HingeLoss`](@ref)`(y)` is ``\sum_i \max(0, 1 - y_i x_i)``
for labels ``y_i = \pm 1``. At the margin ``u = y_i v``, the proximal operator of
``\gamma\max(0, 1 - u)`` is ``u + \gamma`` for ``u < 1 - \gamma``, ``1`` for
``1 - \gamma \le u \le 1``, and ``u`` above. Without labels, ``x`` holds the margins.

**The ε-insensitive loss.** [`EpsInsensitiveLoss`](@ref)`(ε)` is
``\sum_i \max(0, |x_i| - \varepsilon)``. Its proximal operator keeps ``v`` for
``|v| \le \varepsilon``, puts it at ``\pm\varepsilon`` for
``\varepsilon < |v| \le \varepsilon + \gamma``, and moves it toward zero by
``\gamma`` beyond. With ``\varepsilon = 0`` it is soft thresholding.

All three take weights ``w_i``, which multiply the step. For coordinate descent
and Newton steps, the scalar losses `ProximalBase.SquaredHinge`,
`ProximalBase.SmoothCheck` and `ProximalBase.ConvolutionCheck` of
[`GLMLoss`](@ref) are smooth versions with bounded curvature; see
[Smooth functions](@ref).

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

[`SingularValue`](@ref)`(f)` is the general function of the singular values, for
any absolutely symmetric vector function ``f`` with a proximal operator: one
thin SVD, then the proximal operator of ``f`` on the singular values. The
nuclear norm is `SingularValue(NormL1(λ))`. The others:

| `SingularValue(f)` | ``F(X)`` | TFOCS |
|:-------------------|:---------|:------|
| `SingularValue(NormLinf(λ))` | the spectral norm ``\lambda\lVert X\rVert_2`` | `prox_spectral` |
| `SingularValue(IndBallL1(r))` | the indicator of the nuclear-norm ball ``\lVert X\rVert_* \le r`` | `proj_nuclear` |
| `SingularValue(IndBallLinf(r))` | the indicator of the spectral-norm ball ``\lVert X\rVert_2 \le r`` | `proj_spectral` |
| `SingularValue(IndBallL0(r))` | the indicator of rank at most ``r`` | |
| `SingularValue(ElasticNet(λ1, λ2))` | ``\lambda_1\lVert X\rVert_* + \tfrac{\lambda_2}{2}\lVert X\rVert_F^2`` | |
| `SingularValue(Conjugate(Constrained(IndBox(0, λ), IndHalfspace(1, λ * k))))` | the Ky Fan ``k``-norm ``\lambda(\sigma_1 + \dots + \sigma_k)`` | |
| `SingularValue(MCP(λ, a))`, `SCAD`, `NormL0` | nonconvex penalties of the singular values | |

For a nonconvex separable ``f`` whose scalar proximal operator is nondecreasing
on ``[0, \infty)``, applying it to the singular values still gives a global
minimizer (Lu, Zhu, Xu, Yan and Lin, 2015). So MCP and SCAD thresholding of the
singular values reduce the bias of singular value thresholding, as they do for
the lasso. The conjugate of ``f \circ \sigma`` is ``f^* \circ \sigma``, so the
spectral norm and the nuclear-norm ball are a conjugate pair under
[`Conjugate`](@ref). The Ky Fan ``k``-norm, the sum of the ``k`` largest singular
values, is the support function of
``\{w : 0 \le w_i \le \lambda,\ \sum_i w_i \le \lambda k\}`` at ``\sigma``. It
needs the halfspace: with the hyperplane, the conjugate of the capped simplex is
the sum of the ``k`` largest entries, which is not absolutely symmetric.

### Partial decompositions

[`NuclearNorm`](@ref), [`TracePSD`](@ref), [`IndPSD`](@ref) and
[`IndBallRank`](@ref) keep only the singular triplets or eigenpairs above a
threshold, or the ``r`` largest, but by default compute the whole decomposition,
in ``O(p^3)`` for a ``p \times p`` matrix. With the keyword `partial = true` they
compute only the pairs they keep, by randomized subspace iteration with a
Rayleigh–Ritz step (Halko, Martinsson and Tropp, 2011): for a result of rank
``k``, each iteration costs ``O(p^2\ell)`` on a block of ``\ell \approx k + 10``
vectors.

- **Warm starts.** A call starts from the block of the previous call, after one
  more power step, with two fresh random vectors. A solver calls the proximal
  operator on nearby matrices, so a call takes one to a few iterations, and a
  direction that enters the leading subspace is amplified from call to call.
- **Accuracy.** A result is accepted when the residuals
  ``\|Xv_j - \sigma_j u_j\|`` (or ``\|Au_j - \theta_j u_j\|``) of the pairs kept
  are below ``\epsilon^{3/4}`` times the largest value, and the next three pairs
  are certified to lie below the threshold: their value plus their residual is
  below it. The result then agrees with the full decomposition to about
  ``10^{-12}`` in `Float64`.
- **Unknown ranks.** For the thresholds of [`NuclearNorm`](@ref),
  [`TracePSD`](@ref) and [`IndPSD`](@ref)`()`, the block starts from the rank of
  the previous call and doubles while too few of its values fall below the
  threshold. For symmetric matrices, it iterates on ``A + cI`` with a shift
  ``c`` when negative eigenvalues are large enough to crowd out the ones kept.
- **Fallback.** The function takes the full decomposition when the block would
  exceed a quarter of the dimension, or when the iteration does not converge,
  and that decomposition starts the next call. `partial = true` is therefore
  safe to use on any input, and pays when the rank of the result is small.
- **What stays full.** The value ``f(X)`` alone needs every singular value or
  eigenvalue, and still takes the full decomposition. As for the full
  decompositions, the workspaces belong to the function object, so do not share
  one between threads.

On a low-rank matrix plus noise, one call on a slightly changed input, as in a
solver, takes (Julia 1.13, one thread of an AMD EPYC 7513; `benchmark/partial.jl`):

| rank 10 | p = 500 | p = 1000 | p = 2000 | p = 5000 |
|:--|--:|--:|--:|--:|
| `NuclearNorm` | 3.39 ms (15×) | 11.7 ms (27×) | 46.2 ms (51×) | 343 ms (125×) |
| `IndBallRank(10)` | 2.69 ms (18×) | 9.20 ms (34×) | 37.4 ms (63×) | 243 ms (177×) |
| `IndPSD(rank = 10)` | 2.42 ms (4×) | 9.63 ms (6×) | 31.8 ms (11×) | 303 ms (24×) |
| `TracePSD` | 3.14 ms (8×) | 10.5 ms (14×) | 40.2 ms (25×) | 311 ms (54×) |
| `IndPSD()` | 3.12 ms (8×) | 12.2 ms (12×) | 47.7 ms (21×) | 363 ms (46×) |

In parentheses, the factor by which this is faster than the full decomposition;
at ``p = 5000`` the full SVD and eigendecomposition were timed once, for
`NuclearNorm` and `TracePSD`, and `IndBallRank` and `IndPSD()` take the same.

Over ranks ``k = 2, 10, 50`` and ``p`` from 200 to 5000, a warm call is faster
than the full decomposition wherever the partial path applies, and so is a call
from a random block, except for [`TracePSD`](@ref) and [`IndPSD`](@ref)`()` at
``p = 500`` with ``k = 50``, about 7% slower. Told the exact rank,
[ARPACK](https://github.com/JuliaLinearAlgebra/Arpack.jl) and
[KrylovKit](https://github.com/Jutho/KrylovKit.jl) compute the leading pairs of
one matrix up to twice as fast for ``p \le 2000``, and up to three times as
fast for [`IndPSD`](@ref)`(rank = k)`, but their SVDs are slower than the
partial one at ``p = 5000``; they allocate at every call, need the rank in
advance, and do not start from the previous call. The Lanczos SVD of TSVD is
often faster but accurate to about ``10^{-7}``, and randomized SVDs without
power iterations to a few percent (see `benchmark/README.md`).
