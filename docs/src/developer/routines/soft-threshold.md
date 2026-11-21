# [Scalar soft thresholding](@id routine-soft-threshold)

Soft thresholding is the scalar proximal map used by `NormL1`. Its
nonnegative variant and its composition with a ridge factor give the
ordinary entry maps of `NormL1Nonneg` and `ElasticNet`, respectively.
The family rules for [NormL1](../../spec/norml1.md#spec-norml1-endpoints),
[NormL1Nonneg](../../spec/norml1nonneg.md#spec-norml1nonneg-endpoints),
and [ElasticNet](../../spec/elasticnet.md#spec-elasticnet-endpoints)
describe endpoint and storage behavior.

## Ordinary formulas

For finite input ``v`` and nonnegative finite threshold ``c=\gamma\lambda w``,
`NormL1` minimizes ``\tfrac12(z-v)^2+c\lvert z\rvert``. The right and left
derivatives at zero are ``-v+c`` and ``-v-c``, so zero solves the problem
when ``-c\le v\le c``. Outside this interval the derivative on the
corresponding half-line gives

```math
S_c(v)=\operatorname{sign}(v)\max(\lvert v\rvert-c,0).
```

For `NormL1Nonneg`, minimizing on the nonnegative half-line gives
``\max(v-c,0)``. With a finite ridge level ``\lambda_2``, the extra
``\tfrac12\gamma\lambda_2 z^2`` term of `ElasticNet` leaves the zero interval
unchanged and multiplies the soft-thresholded result by
``1/(1+\gamma\lambda_2)``. Its threshold uses the ℓ₁ level ``\lambda_1``.
For example, at ``v=-3`` and ``c=1``, soft thresholding gives ``-2``, the
nonnegative map gives zero, and a ridge factor of ``1/2`` gives ``-1``.

## Implementation

The common `shrink` routine adds the positive part of `v - c` to the
negative part of `v + c`. On the ordinary `NormL1` route, some loops use
`_shrink_nonneg`. The `Base.IEEEFloat` method subtracts the projection of
`v` onto `[-c, c]` with two conditional selects. Adding positive zero
gives positive zero at a zero result, including when the input is
negative zero and the threshold is zero. The general `AbstractFloat`
method calls `shrink`.

The [threshold product](@ref routine-scaled-products) determines the
arithmetic path. `NormL1` and `NormL1Nonneg` use the weighted product
when the range flag fails and apply the infinite-level rule on that
branch. `ElasticNet` uses the ℓ₁ threshold result as the input to its
ordinary or scaled ridge multiplication. The contract links above give
the cases in which endpoint handling precedes ordinary thresholding.

## Source and tests

- [Scalar kernels](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/numerical_kernels.jl#L14-L42) and
  [NormL1 threshold dispatch](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/norml1.jl#L76-L100).
- [Nonnegative entry map](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/norml1nonneg.jl#L102-L114),
  [elastic-net composition](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/elasticnet.jl#L69-L97),
  and [ridge coefficients](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/sqrnorml2.jl#L268-L276).
- [Scalar threshold comparisons](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/numerical_kernels.jl#L1305-L1342),
  [constrained minima](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/norml1nonneg.jl#L231-L254), and
  [elastic-net point comparisons](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/elasticnet.jl#L820-L861).
