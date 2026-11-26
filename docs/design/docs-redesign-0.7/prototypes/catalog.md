# [Function catalog](@id function-catalog)

Every function is a Julia object: construct it with its parameters, then evaluate it,
take its gradient or its proximal step at any point. Find the function by the role
it plays in your model. What each one supports is in the
[capability tables](@ref capability-tables).

## If your model needs…

| If your model needs… | use |
|:--|:--|
| squared-error regression | [`LeastSquaresLoss`](@ref entry-leastsquaresloss) |
| binary outcomes (logistic regression) | `LogisticLoss` |
| counts (Poisson regression) | `PoissonLoss` |
| a sparse inverse covariance (graphical lasso) | `LogDetLoss`, with a weighted `NormL1` |
| sparse coefficients (lasso) | [`NormL1`](@ref entry-norml1) |
| sparse coefficients among correlated predictors | `ElasticNet` |
| sparse, nonnegative coefficients | `NormL1Nonneg` |
| less shrinkage of large coefficients | `MCP` or `SCAD` (nonconvex) |
| a cost per nonzero coefficient, or at most ``s`` nonzeros | `NormL0`, or the constraint `IndBallL0` (nonconvex) |
| whole groups selected together (group lasso) | [`GroupNormL2`](@ref entry-groupnorml2) |
| groups, and sparsity within groups | `SparseGroupNormL2` |
| group selection with less shrinkage | `GroupMCP` or `GroupSCAD` (nonconvex) |
| a ridge penalty | `SqrNormL2` |
| bounds on each coefficient | `IndBox`, `IndNonnegative` |
| a norm-ball constraint | [`IndBallL1`](@ref entry-indballl1), `IndBallL2`, `IndBallLinf` |
| a multiple ``\alpha f`` of a function | `α * f` |

Several functions can play more than one role: `SqrNormL2` is a ridge penalty or a
squared distance, and `NormL1` is a penalty while `IndBallL1` is the corresponding
constraint. To combine a loss and a penalty, evaluate them separately and add the values,
`f(x) + g(x)`. There is no object for a sum of functions.

## Losses

| Function | Definition | Constructor | Convex |
|:--|:--|:--|:-:|
| [least squares](@ref entry-leastsquaresloss) | ``\tfrac12\|y-X\beta\|_2^2`` | `LeastSquaresLoss(y, X)` | ✓ |
| logistic | ``\sum_i w_i[\log(1+e^{\eta_i})-y_i\eta_i]``, ``\eta=X\beta+o`` | `LogisticLoss(y, X; weights, offset)` | ✓ |
| Poisson | ``\sum_i[e^{\eta_i}-y_i\eta_i]``, ``\eta=X\beta`` | `PoissonLoss(y, X)` | ✓ |
| square-root least squares | ``\|y-X\beta\|_2`` | `SqrtLeastSquaresLoss(y, X)` | ✓ |
| log-determinant | ``w[\operatorname{tr}(S\Theta)-\log\det\Theta]``, ``\Theta\succ0`` | `LogDetLoss(S, w = 1)` | ✓ |

The regression losses sum over observations; multiply by `1/n` for the mean.

## Sparsity penalties

![Three sparsity penalties: ℓ₁, MCP and SCAD.](../figures/penalties.svg)

| Function | Definition | Constructor | Convex |
|:--|:--|:--|:-:|
| [weighted ℓ₁ norm](@ref entry-norml1) | ``\lambda\sum_i w_i\lvert x_i\rvert`` | `NormL1(λ, w)` | ✓ |
| elastic net | ``\lambda_1\sum_i w_i\lvert x_i\rvert+\tfrac{\lambda_2}2\|x\|_2^2`` | `ElasticNet(λ1, λ2, w)` | ✓ |
| nonnegative ℓ₁ | ``\lambda\sum_i w_ix_i`` on ``x\ge0`` | `NormL1Nonneg(λ, w)` | ✓ |
| minimax concave (MCP) | ``\sum_i p_{\lambda,a}(x_i)`` | `MCP(λ, a = 3)` | — |
| SCAD | ``\sum_i p_{\lambda,a}(x_i)`` | `SCAD(λ, a = 3.7)` | — |
| ℓ₀ count | ``\lambda\cdot\#\{i: x_i\ne0\}`` | `NormL0(λ = 1)` | — |
| off-diagonal ℓ₀ count | ``\lambda\cdot\#\{i\ne j: X_{ij}\ne0\}`` | `NormL0(λ; diagonal = false)` | — |

Weights ``w`` are optional (all one by default), and `NormL1(λ)` and `NormL1Nonneg(λ)` default to ``\lambda=1``. The MCP and SCAD penalties
``p_{\lambda,a}`` are defined in their entries.

## Norm and group penalties

| Function | Definition | Constructor | Convex |
|:--|:--|:--|:-:|
| Euclidean norm | ``\lambda\|x\|_2`` | `NormL2(λ = 1)` | ✓ |
| maximum norm | ``\lambda\max_i\lvert x_i\rvert`` | `NormLinf(λ = 1)` | ✓ |
| [group lasso](@ref entry-groupnorml2) | ``\lambda\sum_k w_k\|x_{G_k}\|_2`` | `GroupNormL2(λ, groups, w)` | ✓ |
| sparse group lasso | ``\lambda_1\sum_{i\in\cup G_k}\lvert x_i\rvert+\lambda_2\sum_k w_k\|x_{G_k}\|_2`` | `SparseGroupNormL2(λ1, λ2, groups, w)` | ✓ |
| group maximum norm | ``\lambda\sum_k w_k\max_{i\in G_k}\lvert x_i\rvert`` | `GroupNormLinf(λ, groups, w)` | ✓ |
| group MCP | ``\sum_k p_{\lambda w_k,a}(\|x_{G_k}\|_2)`` | `GroupMCP(λ, groups, w; a = 3)` | — |
| group SCAD | ``\sum_k p_{\lambda w_k,a}(\|x_{G_k}\|_2)`` | `GroupSCAD(λ, groups, w; a = 3.7)` | — |

Groups are disjoint; indices in no group are not penalized. Default weights are
``\sqrt{\lvert G_k\rvert}``, except ``1`` for `GroupNormLinf`.

## Constraints

![Projections of one point onto three unit balls.](../figures/projections.svg)

| Constraint | Set | Constructor | Convex |
|:--|:--|:--|:-:|
| box | ``l_i\le x_i\le u_i`` | `IndBox(l, u)` | ✓ |
| ℓ∞ ball | ``\max_i\lvert x_i\rvert\le r`` | `IndBallLinf(r)` | ✓ |
| nonnegative orthant | ``x_i\ge0`` | `IndNonnegative()` | ✓ |
| [ℓ₁ ball](@ref entry-indballl1) | ``\|x\|_1\le r`` | `IndBallL1(r = 1)` | ✓ |
| Euclidean ball | ``\|x\|_2\le r`` | `IndBallL2(r = 1)` | ✓ |
| at most ``s`` nonzeros | ``\#\{i: x_i\ne0\}\le s`` | `IndBallL0(s)` | — |

A constraint is represented by its indicator, zero on the set and ``+\infty``
outside; its proximal step is the projection onto the set.

## Quadratic, linear and scaled functions

| Function | Definition | Constructor | Convex |
|:--|:--|:--|:-:|
| squared distance, ridge | ``\tfrac\lambda2\|x-c\|_2^2`` | `SqrNormL2(λ, c)` | ✓ |
| quadratic | ``\tfrac12x^\top Ax+b^\top x+c`` | `QuadraticFunction(A, b, c)` | ◐ if ``A\succeq0`` |
| Kronecker quadratic | ``\tfrac12\langle X,AXB\rangle+\langle C,X\rangle`` | `KronQuadratic(A, B, C)` | ◐ if ``A,B\succeq0`` or ``A,B\preceq0`` |
| linear | ``\langle c,x\rangle+d`` | `LinearFunction(c, d = 0)` | ✓ |
| zero | ``0`` | `ZeroFunction()` | ✓ |
| scaled function | ``\alpha f(x)``, ``\alpha\ge0`` | `α * f` or `Scaled(f, α)` | as ``f`` |

The center ``c`` of `SqrNormL2` is optional (zero by default). For matrices, ``\|\cdot\|_2`` is the Frobenius norm, and ``\langle\cdot,\cdot\rangle``
the Frobenius inner product.

## Not what you need?

- To write a function that is not here, see [Writing your own function](@ref extending).
- To fit a model without writing an algorithm, see [ProximalSolvers](@ref with-proximalsolvers).
