# Smooth functions

| Function | Value | Gradient | [`lipschitz`](@ref) | Proximable |
|:---------|:------|:---------|:------|:----------:|
| [`ZeroFunction`](@ref) | ``0`` | ``0`` | ``0`` | ✓ |
| [`LinearFunction`](@ref)`(c, d)` | ``\langle c, x\rangle + d`` | ``c`` | ``0`` | ✓ |
| [`SqrNormL2`](@ref)`(λ, y)` | ``\tfrac{\lambda}{2}\lVert x - y\rVert ^2`` | ``\lambda(x - y)`` | ``\lambda`` | ✓ |
| [`QuadraticFunction`](@ref)`(A, b, c)` | ``\tfrac12 x^\top A x + b^\top x + c`` | ``Ax + b`` | ``\lVert A\rVert _2`` | ✓ |
| [`KronQuadratic`](@ref)`(A, B, C)` | ``\tfrac12\langle\Theta, A\Theta B\rangle + \langle C, \Theta\rangle`` | ``A\Theta B + C`` | ``\lVert A\rVert _2\lVert B\rVert _2`` | |
| [`LeastSquaresLoss`](@ref)`(Y, X)` | ``\tfrac12\lVert Y - X\beta\rVert ^2`` | ``X^\top(X\beta - Y)`` | ``\lVert X\rVert _2^2`` | ✓ |
| [`GLMLoss`](@ref)`(ℓ, y, X)` | ``\sum_i w_i\,\ell(\eta_i, y_i)``, ``\eta = X\beta + o`` | ``X^\top(w \odot \ell'(\eta, y))`` | ``c_\ell\,\lVert W^{1/2}X\rVert _2^2`` | |
| [`LogDetLoss`](@ref)`(S, w)` | ``w(\operatorname{tr}(SX) - \log\det X)`` | ``w(S - X^{-1})`` | ``\infty`` | ✓ |

## Linear and quadratic functions

The proximal operator of ``f(x) = \langle c, x\rangle + d`` is ``x - \gamma c``.
For ``f(x) = \tfrac{\lambda}{2}\|x - y\|^2`` the optimality condition
``\lambda(z - y) + (z - x)/\gamma = 0`` gives

```math
\operatorname{prox}_{\gamma f}(x) = \frac{x + \gamma\lambda y}{1 + \gamma\lambda}.
```

Both are separable, so they also have scalar proximal operators for
coordinate descent.

For [`QuadraticFunction`](@ref), ``f(x) = \tfrac12 x^\top A x + b^\top x + c``
with ``A`` symmetric positive semidefinite, the proximal point solves
``(I + \gamma A) z = x - \gamma b``. The first call computes the
eigendecomposition ``A = Q\Lambda Q^\top``, once. After that, for any
``\gamma``,

```math
\operatorname{prox}_{\gamma f}(x) = Q\,(I + \gamma\Lambda)^{-1} Q^\top (x - \gamma b)
```

costs two products with ``Q``.

## The Kronecker quadratic

[`KronQuadratic`](@ref) is the quadratic of a ``p \times q`` matrix ``\Theta``

```math
f(\Theta) = \tfrac12\langle \Theta, A\Theta B\rangle + \langle C, \Theta\rangle
= \tfrac12 \operatorname{vec}(\Theta)^\top (B \otimes A)\operatorname{vec}(\Theta)
+ \operatorname{vec}(C)^\top\operatorname{vec}(\Theta),
```

for symmetric positive semidefinite ``A`` and ``B``. The identity uses
``\operatorname{vec}(A\Theta B) = (B^\top \otimes A)\operatorname{vec}(\Theta)``.
The ``pq \times pq`` matrix ``B \otimes A`` is never formed. The gradient is
``A\Theta B + C``, and since the eigenvalues of ``B \otimes A`` are the products
of those of ``A`` and ``B``, ``L = \|A\|_2\|B\|_2``.

On a `SymmetricSparseIterate` the variable is a symmetric matrix, and the
gradient is the one over symmetric matrices: the projection of ``A\Theta B + C``
onto them. Since ``(A\Theta B)^\top = B\Theta A`` for symmetric ``\Theta``, it is

```math
\tfrac12\,(A\Theta B + B\Theta A) + \tfrac12\,(C + C^\top).
```

A solver that steps along one triangle needs this symmetric gradient.

*Example: the D-trace loss.* For two covariance matrices ``\Sigma_x`` and
``\Sigma_y``, the loss
``\tfrac12\operatorname{tr}(\Sigma_x\Delta\Sigma_y\Delta) - \operatorname{tr}(\Delta(\Sigma_y - \Sigma_x))``
(Zhao, Cai and Li, 2014) is `KronQuadratic(Σx, Σy, Σx - Σy)`. Its gradient
vanishes at ``\Delta = \Sigma_x^{-1} - \Sigma_y^{-1}``, since then
``\Sigma_x\Delta\Sigma_y = \Sigma_y - \Sigma_x``. So minimizing it with an ℓ1
penalty estimates the difference of two precision matrices directly; see the
differential-network example.

## Least squares

For [`LeastSquaresLoss`](@ref), ``f(\beta) = \tfrac12\|Y - X\beta\|^2``, with
``Y`` a vector or a matrix. The gradient is ``X^\top(X\beta - Y)`` and
``L = \sigma_1(X)^2``. The proximal point solves
``(X^\top X + I/\gamma) z = X^\top Y + x/\gamma``. With the thin singular value
decomposition ``X = U\Sigma V^\top``, computed once, the solution is

```math
\operatorname{prox}_{\gamma f}(x) = x + V\,\operatorname{diag}\!\Big(\frac{\gamma}{1 + \gamma\sigma_i^2}\Big)\big(\Sigma U^\top Y - \Sigma^2 V^\top x\big),
```

for every ``\gamma``. To derive it, split ``z`` into its components in and
orthogonal to the row space of ``X``; the orthogonal component of ``x`` passes
through unchanged. Each call costs products with ``U`` and ``V``. That is why
ADMM, whose penalty parameter changes, gets the least-squares step at the same
cost every iteration.

## Generalized linear models

[`GLMLoss`](@ref)`(ℓ, y, X; weights, offset)` is the loss
``f(\beta) = \sum_i w_i\,\ell(\eta_i, y_i)`` of the linear predictor
``\eta = X\beta + o``. With ``\ell'`` and ``\ell''`` the derivatives in ``\eta``,

```math
\nabla f(\beta) = X^\top\big(w \odot \ell'(\eta, y)\big), \qquad
\nabla^2 f(\beta) = X^\top \operatorname{diag}\big(w \odot \ell''(\eta, y)\big) X
\preceq c_\ell\, X^\top W X ,
```

where ``c_\ell = \sup \ell''`` is the curvature bound of the scalar loss and
``W = \operatorname{diag}(w)``. So ``L = c_\ell\|W^{1/2}X\|_2^2``, and
[`lipschitz`](@ref) computes the spectral norm for dense matrices. For sparse
and other indexed matrices it uses the conservative bound
``\lVert W^{1/2}X\rVert_1\lVert W^{1/2}X\rVert_\infty`` instead of forming
``X^\top W X`` or densifying the design. [`opnorm2`](@ref) is a separate
iterative estimate and needs backtracking or an independent bound when used
to select a step. One value and gradient cost one product with ``X``, one pass
over the observations and one product with ``X^\top``.

| Scalar loss | ``\ell(\eta, y)`` | ``\ell'(\eta, y)`` | ``c_\ell`` | Constructor |
|:-----|:-----|:-----|:-----|:-----|
| `Squared()` | ``\tfrac12(\eta - y)^2`` | ``\eta - y`` | ``1`` | `GLMLoss(ProximalBase.Squared(), y, X)` |
| `Logistic()` | ``\log(1 + e^\eta) - y\eta`` | ``\sigma(\eta) - y`` | ``1/4`` | [`LogisticLoss`](@ref)`(y, X)` |
| `Poisson()` | ``e^\eta - y\eta`` | ``e^\eta - y`` | ``\infty`` | [`PoissonLoss`](@ref)`(y, X)` |
| `Huber(δ)` | ``\tfrac12 r^2`` if ``\lvert r\rvert \le \delta``, else ``\delta(\lvert r\rvert - \delta/2)``, with ``r = \eta - y`` | ``\operatorname{clamp}(r, -\delta, \delta)`` | ``1`` | [`HuberLoss`](@ref)`(y, X, δ)` |
| `SquaredHinge()` | ``\max(0, 1 - y\eta)^2``, ``y = \pm 1`` | ``-2y\max(0, 1 - y\eta)`` | ``2`` | `GLMLoss(ProximalBase.SquaredHinge(), y, X)` |
| `SmoothCheck(τ, h)` | the Moreau envelope of ``\rho_\tau`` at ``u = y - \eta`` | ``-\operatorname{clamp}(u/h, \tau - 1, \tau)`` | ``1/h`` | `GLMLoss(ProximalBase.SmoothCheck(τ, h), y, X)` |
| `ConvolutionCheck(τ, h; kernel)` | ``\int \rho_\tau(u - t)K(t/h)/h\,dt`` at ``u = y - \eta`` | ``-(\tau - \tfrac12) - \tfrac12 m'(u/h)`` for ``\lvert u\rvert < h`` | ``\max K/h`` | `GLMLoss(ProximalBase.ConvolutionCheck(τ, h), y, X)` |
| `Smoothed(φ, μ)` | ``\min_t \varphi(t) + \tfrac{1}{2\mu}(t - r)^2``, with ``r = \eta - y`` | ``(r - \operatorname{prox}_{\mu\varphi}(r))/\mu`` | ``1/\mu`` | `GLMLoss(ProximalBase.Smoothed(φ, μ), y, X)` |

Here ``\sigma(\eta) = 1/(1 + e^{-\eta})``, and ``\log(1 + e^\eta)`` is
computed so that it neither overflows nor loses precision. The scalar losses
expose `ProximalBase.deriv`, `deriv2` and `curvature`, for Newton and
coordinate steps.

`SmoothCheck` and `ConvolutionCheck` smooth the check loss
``\rho_\tau(u) = \max(\tau u, (\tau - 1)u)`` of
quantile regression, so that coordinate descent and proximal Newton fit it
without a linear program. `SmoothCheck(τ, h)` is its Moreau envelope
``\min_s \rho_\tau(s) + (s - u)^2/(2h)``: an asymmetric Huber loss that is
quadratic on ``-h(1 - \tau) < u < h\tau`` and within ``h/2`` of ``\rho_\tau``.
`ConvolutionCheck(τ, h)` convolves ``\rho_\tau`` with a kernel of bandwidth ``h``, the
smoothing of the conquer estimator (He, Pan, Tan and Zhou, 2023). With the
uniform or the Epanechnikov kernel (`kernel = :uniform` or `:epanechnikov`) it
equals ``\rho_\tau`` for ``|u| \ge h`` and is a polynomial in between: with
``m(s) = \int|s - t|K(t)\,dt``, it is ``(\tau - \tfrac12)u + \tfrac{h}{2}m(u/h)``.

## The Gaussian log-likelihood

[`LogDetLoss`](@ref)`(S, w)` is ``f(X) = w\,(\operatorname{tr}(SX) - \log\det X)``
on the positive definite matrices and ``+\infty`` elsewhere: the loss of the
graphical lasso, for a sample covariance ``S``. One Cholesky factorization gives
its value and its gradient ``w(S - X^{-1})``. The value never throws, so a line
search can reject a step that leaves the cone. The gradient is not Lipschitz,
since ``X^{-1}`` blows up at the boundary.

Its proximal operator has a closed form. The optimality condition for
``Z = \operatorname{prox}_{\gamma f}(X)`` is
``w(S - Z^{-1}) + (Z - X)/\gamma = 0``, which with ``\rho = 1/(\gamma w)``
reads

```math
\rho Z - Z^{-1} = \rho X - S .
```

Take the eigendecomposition ``\rho X - S = U\operatorname{diag}(t)U^\top``. The
matrix ``Z = U\operatorname{diag}(\mu)U^\top`` solves the equation when
``\rho\mu_i - 1/\mu_i = t_i``, that is, when

```math
\mu_i = \frac{t_i + \sqrt{t_i^2 + 4\rho}}{2\rho} = \frac{2}{\sqrt{t_i^2 + 4\rho} - t_i} > 0 .
```

The first form is used for ``t_i \ge 0`` and the second for ``t_i < 0``, which
avoids cancellation. ``Z`` is positive definite by construction, and
[`value_and_prox!`](@ref) gets ``f(Z)`` from the ``\mu_i``. This is the
``\Omega``-step of ADMM for the graphical lasso.

## Sums

[`Sum`](@ref)`(f1, f2, ...)` is ``\sum_k f_k``, a smooth function when all the
``f_k`` are. Its gradient and its Lipschitz constant are the sums of theirs. A
sum has no proximal operator in general.

## Second-order models

The smooth functions with [`hashessian`](@ref) provide two more oracles, for
Newton-type methods. [`hessian_vec!`](@ref)`(Hv, f, x, v)` computes
``\nabla^2 f(x)\,v`` at about the cost of a gradient, and
[`quadratic_model`](@ref)`(f, x)` returns the second-order model of ``f`` at
``x``,

```math
q(z) = f(x) + \langle \nabla f(x), z - x\rangle
+ \tfrac12\langle z - x, \nabla^2 f(x)(z - x)\rangle \quad (\text{up to a constant}),
```

as another function of ProximalBase. The GLM construction below floors very
small observation curvatures, so its model is a regularized curvature model
where that floor is active, while preserving the gradient at the model point.
Minimizing ``q + g`` with any solver and
penalty ``g`` is a proximal Newton step, and [`quadratic_model!`](@ref) moves
the model to the next point in place.

The quadratics are their own models. For a [`GLMLoss`](@ref) the model is the
weighted least-squares loss of iteratively reweighted least squares,

```math
q(\beta) = \tfrac12 \sum_i w_i\,\ell''_i\,\big(\eta_i(\beta) - z_i\big)^2 + \text{const},
\qquad z_i = \eta_i - \frac{\ell'_i}{\ell''_i},
```

with ``\ell'_i`` and ``\ell''_i`` evaluated at the current linear predictor. The
curvature ``\ell''_i`` is floored at ``\sqrt{\varepsilon}``, so that the working
responses stay finite where it vanishes (a Huber loss beyond ``\delta``); the
gradient of ``q`` at the current point is still exact.

For [`LogDetLoss`](@ref)`(S, w)`, with ``W = X^{-1}``, the expansion of
``w(\operatorname{tr}(SZ) - \log\det Z)`` at ``X`` in the step ``\Delta = Z - X`` is,
up to a constant,
``w\operatorname{tr}((S - W)\Delta) + \tfrac{w}{2}\operatorname{tr}(W\Delta W\Delta)``.
Written in the new point ``Z``, and using ``WXW = W``, it is

```math
q(Z) = \tfrac{w}{2}\operatorname{tr}(WZWZ) + w\operatorname{tr}\big((S - 2W)Z\big) + \text{const},
```

the [`KronQuadratic`](@ref)`(wW, W, w(S - 2W))`. Minimizing ``q(Z) + \lambda\|Z\|_1``
over symmetric matrices, by coordinate descent on a `SymmetricSparseIterate`, is
the Newton step of QUIC for the graphical lasso (Hsieh, Sustik, Dhillon and
Ravikumar, 2014). Without the penalty, its minimizer is ``2X - XSX``, Newton's
iteration for ``S^{-1}``.
