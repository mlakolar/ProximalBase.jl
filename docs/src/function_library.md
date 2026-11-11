# Browse available functions

These are the mathematical functions provided by ProximalBase.jl. Each entry
links to its definition, how to construct it, and a working example.
The same function may have more than one use: for example, a squared norm
can serve as either a smooth objective or a regularization penalty.

| Object | Definition | Gradient | Proximal map | Guide |
|:--|:--|:--:|:--:|:--|
| [`LeastSquaresLoss`](@ref) | ``\|Y-XB\|_F^2/2`` | yes | yes | [Smooth functions](smooth.md#Least-squares-loss) |
| [`SqrNormL2`](@ref) | ``\lambda\|x-y\|^2/2`` | yes | yes | [Smooth functions](smooth.md#Squared-Euclidean-norm) |
| [`QuadraticFunction`](@ref) | ``x^\top A x/2+b^\top x+c`` | yes | yes | [Smooth functions](smooth.md#Quadratic-function) |
| [`ZeroFunction`](@ref) | ``0`` | yes¹ | identity | [Smooth functions](smooth.md#Zero-function) |
| [`LinearFunction`](@ref) | ``\langle c,x\rangle+d`` | yes | shift | [Smooth functions](smooth.md#Linear-function) |
| [`NormL1`](@ref) | ``\lambda_0\sum_iw_i|x_i|`` | no | soft thresholding | [Penalties](penalties.md) |
| [`Scaled`](@ref) | ``\alpha f(x)`` | if supported | if supported | [Functions built from functions](calculus.md) |

¹ `ZeroFunction` provides gradients for arrays, but not for
`SymmetricSparseIterate`. See [symmetric storage](symmetric_sparse_iterate.md).

Use a function in more than one role when its mathematics permits it:
SqrNormL2 is a smooth objective or a ridge penalty; LeastSquaresLoss has both
a gradient and an SVD-based prox. Trait categories overlap. See
[capabilities](function_objects.md) for the full trait table.

The functions operating on arrays use real element types. Ordinary dense
arrays and the array iterates are accepted within each function's shape
contract. Symmetric matrix storage supports a more specific collection of
operations, described in the [symmetric guide](symmetric_sparse_iterate.md).

## Scalar observation loss

For developers implementing statistical models, `ProximalBase.Squared()`
provides the squared loss of a single prediction and observation,
``\ell(\eta,y)=(\eta-y)^2/2``. It uses `loss(prediction, observation)`
rather than the `f(x)` interface of the array functions above.
See [Scalar squared loss](smooth.md#Scalar-squared-loss) for its derivative
operations and use cases.

## Combining functions

ProximalBase supports multiplying a function by a nonnegative scalar.
For a composite objective, keep `f` and `g` separately and evaluate
`f(x) + g(x)` when you need the total objective value. This adds two
numbers; it does not create a new function object. See
[Functions built from functions](calculus.md) for scaling.
