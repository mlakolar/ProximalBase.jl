# Browse available functions

ProximalBase provides mathematical functions for building optimization
algorithms. Each Julia object represents a **specific mathematical formula**.
The table below pairs the formula with the constructor: you should be able
to see exactly how the symbols in a definition become Julia arguments.

## Functions of a vector or matrix

| Julia constructor | Mathematical function | Arguments and interpretation | Detailed guide |
|:--|:--|:--|:--|
| [`LeastSquaresLoss(Y, X)`](@ref LeastSquaresLoss) | ``f(B)=\frac12\lVert Y-XB\rVert_F^2`` | ``X``: design matrix; ``Y``: response; ``B``: argument supplied to `f(B)` | [Least squares](smooth.md#Least-squares-loss) |
| [`SqrNormL2(λ, y)`](@ref SqrNormL2) | ``f(x)=\frac{\lambda}{2}\lVert x-y\rVert^2`` | ``\lambda\geq0``: level; ``y``: center (zero when omitted) | [Squared Euclidean norm](smooth.md#Squared-Euclidean-norm) |
| [`QuadraticFunction(A, b, c)`](@ref QuadraticFunction) | ``f(x)=\frac12x^\mathsf{T}Ax+b^\mathsf{T}x+c`` | ``A``: symmetric positive semidefinite matrix (caller assumption); ``b``: linear coefficient (zero when omitted); ``c``: constant (zero when omitted) | [Quadratic function](smooth.md#Quadratic-function) |
| [`ZeroFunction()`](@ref ZeroFunction) | ``f(x)=0`` | No parameters | [Zero function](smooth.md#Zero-function) |
| [`LinearFunction(c, d)`](@ref LinearFunction) | ``f(x)=\langle c,x\rangle+d`` | ``c``: coefficient array; ``d``: constant (zero when omitted) | [Linear function](smooth.md#Linear-function) |
| [`NormL1(λ0, w)`](@ref NormL1) | ``g(x)=\lambda_0\sum_i w_i\lvert x_i\rvert`` | ``\lambda_0\geq0``: overall level; ``w``: nonnegative weights (all one when omitted) | [The ℓ₁ norm](penalties.md) |
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

## Scalar squared loss

For functions of an individual **prediction** ``\eta`` and
**observation** ``y``, ProximalBase also provides
[`ProximalBase.Squared()`](@ref ProximalBase.Squared), representing

```math
\ell(\eta,y)=\frac12(\eta-y)^2.
```

It is called as `loss(η, y)`, not `f(x)`, and has the
specialized `deriv`, `deriv2` and `value_and_deriv` operations.
It is not a general array loss constructor or an automatic regression
model. See [Scalar losses](scalar_losses.md) for the separate interface and
[Squared loss](scalar_squared.md) for its mathematical definition and Julia example.

## Combining functions

A nonnegative multiple ``\alpha f`` can be constructed with
`Scaled(f, α)` or `α*f`. To evaluate an objective consisting of
several functions, write `f(x) + g(x)` to **add their numerical values**.
There is no general `f + g` function-object constructor in this version.
For algorithmic examples, start with [Getting started](getting_started.md).

Each function supports a documented set of argument shapes and operations.
Consult its detailed guide rather than assuming every trait holds for
every iterate representation.
