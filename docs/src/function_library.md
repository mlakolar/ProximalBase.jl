# Browse available functions

*Assumes: gradients and matrix notation · You will learn: which constructor matches each mathematical function and which squared loss to use.*

ProximalBase provides mathematical functions for building optimization
algorithms. Each Julia object represents a **specific mathematical formula**.
The table below pairs the formula with the constructor: you should be able
to see exactly how the symbols in a definition become Julia arguments.

For a task-oriented starting point, use [Choose a function or operation](choosing.md).

## Functions of a vector or matrix

| Julia constructor | Mathematical function | Arguments and interpretation | Detailed guide |
|:--|:--|:--|:--|
| [`LeastSquaresLoss(Y, X)`](@ref LeastSquaresLoss) | ``f(B)=\frac12\lVert Y-XB\rVert_F^2`` | ``X``: design matrix; ``Y``: response; ``B``: argument supplied to `f(B)` | [Least squares](smooth.md#Least-squares-loss) |
| [`LogisticLoss(y, X; weights, offset)`](@ref LogisticLoss) | ``f(\beta)=\sum_i w_i[\log(1+\exp(\eta_i))-y_i\eta_i]`` | ``\eta=X\beta+\mathrm{offset}``; convex and smooth | [Logistic regression](smooth.md#Logistic-regression-loss) |
| [`PoissonLoss(y, X)`](@ref PoissonLoss) | ``f(\beta)=\sum_i[\exp(\eta_i)-y_i\eta_i]`` | ``\eta=X\beta``; ``y_i\geq0``; dense `Float64` data; convex and smooth | [Poisson regression](smooth.md#Poisson-regression-loss) |
| [`SqrtLeastSquaresLoss(Y, X)`](@ref SqrtLeastSquaresLoss) | ``f(\beta)=\lVert Y-X\beta\rVert_2`` | ``X``: design matrix; ``Y``: response vector; ``\beta``: argument supplied to `f(β)`; convex, not smooth, no proximal operator | [Square-root least squares](smooth.md#Square-root-least-squares-is-not-smooth) |
| [`LogDetLoss(S, w)`](@ref LogDetLoss) | ``w[\operatorname{tr}(SX)-\log\det(X)]`` on SPD matrices | ``S``: finite symmetric coefficient matrix; ``w>0``: weight (one when omitted) | [Log-determinant loss](smooth.md#Log-determinant-loss) |
| [`SqrNormL2(λ, y)`](@ref SqrNormL2) | ``f(x)=\frac{\lambda}{2}\lVert x-y\rVert^2`` | ``\lambda\geq0``: level; ``y``: center (zero when omitted) | [Squared Euclidean norm](smooth.md#Squared-Euclidean-norm) |
| [`QuadraticFunction(A, b, c)`](@ref QuadraticFunction) | ``f(x)=\frac12x^\mathsf{T}Ax+b^\mathsf{T}x+c`` | ``A``: symmetric matrix (caller precondition), positive semidefinite not required; ``b``: linear coefficient (zero when omitted); ``c``: constant (zero when omitted) | [Quadratic function](smooth.md#Quadratic-function) |
| [`ZeroFunction()`](@ref ZeroFunction) | ``f(x)=0`` | No parameters | [Zero function](smooth.md#Zero-function) |
| [`LinearFunction(c, d)`](@ref LinearFunction) | ``f(x)=\langle c,x\rangle+d`` | ``c``: coefficient array; ``d``: constant (zero when omitted) | [Linear function](smooth.md#Linear-function) |
| [`NormL0(λ)`](@ref NormL0) | Level times the number of nonzero entries | ``\lambda\geq0``: level; nonconvex with a hard-thresholding proximal operator | [Numerical contract](numerics.md#Cardinality-penalty-endpoints) |
| [`NormL0(λ; diagonal = false)`](@ref NormL0) | Level times the number of nonzero off-diagonal entries of a matrix | finite ``\lambda\geq0``: level; nonconvex, for finite strided square `Float32` or `Float64` matrices that are symmetric by precondition, under the [storage preconditions](@ref spec-norml0offdiag-arguments), with a hard-thresholding proximal operator that keeps the diagonal | [Off-diagonal cardinality penalty](@ref norml0offdiag-guide) |
| [`MCP(λ, a)`](@ref MCP) | Minimax concave penalty applied entrywise | ``\lambda\geq0`` is the level and ``a>0`` is the shape. The proximal map uses firm or hard thresholding depending on the step. | [Minimax concave penalty](@ref mcp-guide) |
| [`SCAD(λ, a)`](@ref SCAD) | Sum of smoothly clipped absolute deviations | ``\lambda\geq0``: level, ``a>2``: shape (default 3.7) | [SCAD penalty](penalties.md#SCAD-penalty) |
| [`NormL1(λ0, w)`](@ref NormL1) | ``g(x)=\lambda_0\sum_i w_i\lvert x_i\rvert`` | ``\lambda_0\geq0``: overall level; ``w``: nonnegative weights (all one when omitted) | [The ℓ₁ norm](penalties.md) |
| [`NormL1Nonneg(λ0, w)`](@ref NormL1Nonneg) | ``g(x)=\lambda_0\sum_i w_ix_i`` for ``x\geq0``, ``+\infty`` otherwise | ``\lambda_0\geq0``: overall level; ``w``: nonnegative weights (all one when omitted) | [The nonnegative ℓ₁ penalty](penalties.md#The-nonnegative-l1-penalty) |
| [`ElasticNet(λ1, λ2, w)`](@ref ElasticNet) | ``g(x)=\lambda_1\sum_i w_i\lvert x_i\rvert+\frac{\lambda_2}{2}\sum_i x_i^2`` | ``\lambda_1,\lambda_2\geq0``: ℓ₁ and ridge levels; ``w``: nonnegative ℓ₁ weights (all one when omitted) | [The elastic-net penalty](penalties.md#The-elastic-net-penalty) |
| [`IndBox(l, u)`](@ref IndBox) | Indicator of coordinatewise intervals. Its proximal point clamps each entry to its interval. | `l`, `u` are scalar or per-entry bounds | [Numerical contract](numerics.md) |
| [`IndBallLinf(r)`](@ref IndBallLinf) | Indicator of ``\lVert x\rVert_\infty\leq r``, the box ``[-r, r]``, with clipping projection | ``r\geq 0`` is the radius. A positive unsigned radius is rejected | [Numerical contract](numerics.md) |
| [`IndNonnegative()`](@ref IndNonnegative) | ``g(x)=0`` if every entry of ``x`` is nonnegative, ``\infty`` if an entry is negative (``\mathrm{NaN}`` at a ``\mathrm{NaN}`` entry); the proximal point is ``\max(x_i,0)`` | No parameters | [Numerical contract](numerics.md) |
| [`IndBallL0(s; diagonal = true)`](@ref IndBallL0) | Cardinality indicator with Euclidean projection. Vectors and ordinary matrices select individual entries. Explicit symmetric representations select whole pairs (two entries) and, when counted, the best combination of diagonals and pairs | Nonnegative integer budget. Dense vectors and matrices, vector and matrix `SparseIterate`, dense `Symmetric`, and `SymmetricSparseIterate` with homogeneous `Float32` or `Float64`. Use `diagonal = false` to preserve a matrix diagonal | [API entry](@ref IndBallL0) |
| [`IndBallL1(r)`](@ref IndBallL1) | Indicator of ``\lVert x\rVert_1\leq r``; the proximal point is the Euclidean projection, a soft threshold at a common level | ``r\geq 0`` is the radius and defaults to one. Matrices sum the magnitudes of all entries | [Numerical contract](@ref spec-indballl1) |
| [`IndBallL2(r)`](@ref IndBallL2) | Indicator of ``\lVert x\rVert_2\leq r`` with radial projection | ``r\geq 0`` is the radius and defaults to one. Matrices use the Frobenius norm | [Numerical contract](numerics.md) |
| [`NormL2(λ)`](@ref NormL2) | ``g(x)=\lambda\lVert x\rVert_2`` | ``\lambda\geq0``: level; the Frobenius norm for a matrix | [The ℓ₂ and group ℓ₂ norms](group_penalty.md) |
| [`NormLinf(λ)`](@ref NormLinf) | ``g(x)=\lambda\max_i\lvert x_i\rvert`` | ``\lambda\geq0``: level; the largest magnitude of a matrix | [The ℓ∞ norm](group_penalty.md#The-NormLinf-norm) |
| [`GroupNormL2(λ0, groups, w)`](@ref GroupNormL2) | ``g(x)=\lambda_0\sum_k w_k\lVert x_{G_k}\rVert_2`` | ``\lambda_0\geq0``: overall level; ``G_k``: disjoint groups of indices; ``w``: nonnegative weights (``\sqrt{\lvert G_k\rvert}`` when omitted) | [The ℓ₂ and group ℓ₂ norms](group_penalty.md) |
| [`SparseGroupNormL2(λ1, λ2, groups, w)`](@ref SparseGroupNormL2) | ``g(x)=\lambda_1\sum_{i\in U}\lvert x_i\rvert+\lambda_2\sum_k w_k\lVert x_{G_k}\rVert_2`` | ``U`` is the union of the disjoint groups. Default weights are ``\sqrt{\lvert G_k\rvert}``. | [Sparse group lasso](group_penalty.md#sparse-group-l2-guide) |
| [`GroupNormLinf(λ0, groups, w)`](@ref GroupNormLinf) | ``g(x)=\lambda_0\sum_k w_k\max_{i\in G_k}\lvert x_i\rvert`` | ``\lambda_0\geq0``: overall level; ``G_k``: disjoint groups of indices; ``w``: nonnegative weights (``1`` when omitted) | [The group ℓ∞ norm](group_penalty.md#group-linf-guide) |
| [`Scaled(f, α)`](@ref Scaled) | ``h(x)=\alpha f(x)`` | ``f``: existing function object; ``\alpha\geq0``: scale | [Scaling a function](calculus.md) |

For `SqrNormL2`, the norm is the usual Euclidean norm for vectors and
the **Frobenius norm** for matrices. It is not the spectral norm of a matrix.

The variable **being optimized** (such as ``x`` or ``B``) is
passed when the function is *evaluated*, not when the function object is
constructed. For example:

```@example function_catalog
using ProximalBase

X = [1.0 0.0; 0.0 2.0]
y = [1.0, -1.0]
β = [0.5, 0.25]

f = LeastSquaresLoss(y, X)          # f(β) = ||y-Xβ||²/2
g = NormL1(0.2)                     # g(β) = 0.2*sum(abs,β)
@assert f(β) ≈ sum(abs2, y - X*β)/2
@assert g(β) ≈ 0.2 * sum(abs, β)
(f(β), g(β), f(β) + g(β))
```

This example uses `g` as a **regularizer** in `f(β)+g(β)`. The
role of a function is determined by the optimization problem: the
same `SqrNormL2`, for instance, can be used as a data-fitting function
or a ridge regularizer. The categories above describe the mathematics,
not separate solver types.

To find which functions provide gradients, proximal operators,
separable-coordinate operations, or convexity, see
[Function objects and their properties](function_objects.md).
To learn what these operations compute, see
[Operations on functions](operations.md).

!!! note "Which squared-loss object should I use?"
    - `SqrNormL2(λ, y)` represents ``\lambda\|x-y\|^2/2`` for an array
      argument ``x``. In particular, `SqrNormL2(1.0)` uses
      ``\lambda=1`` and no center, so its value is ``\|x\|^2/2``.
      It supplies array gradients and proximal operations.
    - `LeastSquaresLoss(y, X)` represents
      ``\|y-X\beta\|^2/2`` as a function of coefficients ``\beta``.
      It also supplies array gradients and proximal operations.

## Combining functions

A nonnegative multiple ``\alpha f`` can be constructed with
`Scaled(f, α)` or `α*f`. To evaluate an objective consisting of
several functions, write `f(x) + g(x)` to **add their numerical values**.
There is no general `f + g` function-object constructor in this version.
For algorithmic examples, start with [Getting started](getting_started.md).

Each function supports a documented set of argument shapes and operations.
Consult its detailed guide rather than assuming every trait holds for
every iterate representation.
