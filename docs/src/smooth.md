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
[`lipschitz`](@ref) computes it by the Lanczos process without forming
``X^\top W X``. One value and gradient cost one product with ``X``, one pass
over the observations and one product with ``X^\top``.

| Scalar loss | ``\ell(\eta, y)`` | ``\ell'(\eta, y)`` | ``c_\ell`` | Constructor |
|:-----|:-----|:-----|:-----|:-----|
| `Squared()` | ``\tfrac12(\eta - y)^2`` | ``\eta - y`` | ``1`` | `GLMLoss(ProximalBase.Squared(), y, X)` |
| `Logistic()` | ``\log(1 + e^\eta) - y\eta`` | ``\sigma(\eta) - y`` | ``1/4`` | [`LogisticLoss`](@ref)`(y, X)` |
| `Poisson()` | ``e^\eta - y\eta`` | ``e^\eta - y`` | ``\infty`` | [`PoissonLoss`](@ref)`(y, X)` |
| `Huber(δ)` | ``\tfrac12 r^2`` if ``\lvert r\rvert \le \delta``, else ``\delta(\lvert r\rvert - \delta/2)``, with ``r = \eta - y`` | ``\operatorname{clamp}(r, -\delta, \delta)`` | ``1`` | [`HuberLoss`](@ref)`(y, X, δ)` |
| `Smoothed(φ, μ)` | ``\min_t \varphi(t) + \tfrac{1}{2\mu}(t - r)^2``, with ``r = \eta - y`` | ``(r - \operatorname{prox}_{\mu\varphi}(r))/\mu`` | ``1/\mu`` | `GLMLoss(ProximalBase.Smoothed(φ, μ), y, X)` |

Here ``\sigma(\eta) = 1/(1 + e^{-\eta})``, and ``\log(1 + e^\eta)`` is
computed so that it neither overflows nor loses precision. The scalar losses
expose `ProximalBase.deriv`, `deriv2` and `curvature`, for Newton and
coordinate steps.

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
