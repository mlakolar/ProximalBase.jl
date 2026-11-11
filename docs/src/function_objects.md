# Function objects and capabilities

A function object stores the parameters of a mathematical function. Every
retained vector/array function is an [`AbstractFunction`](@ref), callable as
`f(x)`. For example, `NormL1(0.5)` represents half the entrywise ``\ell_1``
norm; `LeastSquaresLoss(y,X)` represents half the squared residual norm.
[`Squared`](@ref ProximalBase.Squared) belongs to the separate scalar observation
loss interface and is used qualified as `ProximalBase.Squared()`.

Objects retain array parameters by reference unless a constructor explicitly
copies. Losses and quadratics also own scratch buffers and lazy decomposition
caches. See [ownership](numerical_usage.md) before reusing one object concurrently.

## Traits describe properties

| Trait | Meaning | Default for AbstractFunction |
|:--|:--|:--|
| [`issmooth`](@ref) | Supplies gradient operations on the interior of its domain | false |
| [`isproximable`](@ref) | Supplies an efficiently computable proximal map | false |
| [`isseparable`](@ref) | Sum over independent scalar entries | false |
| [`isconvex`](@ref) | Convex under the function's mathematical assumptions | false |
| [`iszeropreserving`](@ref) | Each entry has value zero at zero and prox maps zero to zero, for every positive finite step | false |

Define the first four on a type; the instance query forwards to that type.
`iszeropreserving` is an instance property: centers, levels and wrapper scales
can affect it. Declaring a trait does not create Julia methods.

| Retained function | Smooth | Proximable | Separable | Convex | Zero preserving |
|:--|:--:|:--:|:--:|:--:|:--|
| ZeroFunction | yes | yes | yes | yes | yes |
| LinearFunction | yes | yes | yes | yes | false |
| SqrNormL2 | yes | yes | yes | yes | uncentered and finite level only |
| QuadraticFunction | yes | yes | no | yes¹ | false |
| LeastSquaresLoss | yes | yes | no | yes | false |
| NormL1 | no | yes | yes | yes | yes |
| Scaled | forwards | forwards | forwards | forwards | finite scale and zero-preserving inner function |

¹ `QuadraticFunction` assumes a symmetric positive semidefinite matrix; its
constructor does not verify those properties. The trait is unconditional and
is not a matrix validator.

## Capabilities and supported signatures

`NormL1` is a [`SeparableFunction`](@ref); its scalar `value_entry` and
`prox_block` generate generic array values and proximal maps. `SqrNormL2`,
`LinearFunction` and `ZeroFunction` are mathematically separable and have scalar
`prox_block` methods, but are not `SeparableFunction` subtypes and have no
`value_entry` method. `isseparable(f)` alone does not promise that method.

A true trait also does not imply support for every argument shape or storage:
least squares accepts vector or matrix coefficients, quadratics accept vectors,
and symmetric owners require explicitly implemented dispatch. A custom
`SeparableFunction` automatically has `isproximable=true` and `isseparable=true`,
but its author must still implement both scalar hooks. Smoothness is independent.

The concrete [function library](function_library.md),
[operation catalog](operations.md) and [extension guide](extending.md) describe
actual combinations. The [coverage matrix](api/coverage.md) lists every export
and essential qualified extension hook.
