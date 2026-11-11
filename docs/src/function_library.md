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
| [`ProximalBase.Squared`](@ref) | ``(\eta-y)^2/2`` | scalar deriv | no array prox | [Scalar squared loss](smooth.md#Scalar-squared-loss) |

¹ ZeroFunction gradients accept arrays; its owner value/prox methods do not
include gradients on a SymmetricSparseIterate.

Use a function in more than one role when its mathematics permits it:
SqrNormL2 is a smooth objective or a ridge penalty; LeastSquaresLoss has both
a gradient and an SVD-based prox. Trait categories overlap. See
[capabilities](function_objects.md) for the full trait table.

The functions operating on arrays use real element types. Ordinary dense
arrays and the array iterates are accepted within each function's shape
contract; symmetric owners have a narrower explicit method surface described
in [the symmetric guide](symmetric_sparse_iterate.md). `ProximalBase.Squared` is a specialized **scalar loss**: it measures the
squared discrepancy between a prediction and an observation. Unlike the
other functions in the table, it is called with two scalar arguments,
`loss(prediction, observation)`. It is described in detail in the
[scalar squared-loss section](smooth.md#Scalar-squared-loss).

ProximalBase also supports multiplying a function by a nonnegative scalar. Construct a composite
objective by keeping `f` and `g` separately and evaluating `f(x)+g(x)`; this
expression adds two scalar values. It does not construct a function object
with automatic gradient/prox dispatch.
