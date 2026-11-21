# [Scaled products](@id routine-scaled-products)

Proximal thresholds and weighted values combine factors such as a step,
a penalty level, and a weight. The shared product routines can keep a
representable result from being lost when the first multiplication leaves
the normal range. The [shared range rules](../../numerics.md#numerical-value-range),
[NormL1 range rules](../../spec/norml1.md#spec-norml1-range), and
[Scaled range rules](../../spec/scaled.md#spec-scaled-range) describe the
behavior of the public calls.

## Ordinary path

`_positive_product` keeps `(a * b) * c` when the first product is normal. A
range change in the final multiplication has no later factor to bring it
back into range. For proximal thresholds, `_threshold_product` forms
`(γ * λ) * w` in its working type, converts it to the requested threshold
type, and reports whether `γ * λ` was normal in the working type. A false
flag sends the `NormL1` threshold calculation to its exceptional path. The flag describes the first
product, so a true flag can coexist with an infinite final threshold.

## Exponent scaling

For nonzero finite factors, write each one as a significand times a power
of two. The three-factor kernel multiplies the bounded significands and
applies the sum of the exponents with `ldexp`. For instance, multiplying
``2^{800}`` and ``2^{400}`` first overflows in `Float64`, even though their
product with ``2^{-300}`` is ``2^{900}``. Keeping the exponents separate
also avoids starting the last multiplication with a rounded subnormal
intermediate.

The weighted threshold path handles a stored infinite weight before
using the finite-product recovery. `NormL1` also checks its infinite-level
rule in the exceptional branch. See the
[NormL1 endpoint rules](../../spec/norml1.md#spec-norml1-endpoints).

When `Scaled` applies a factor to a value held as significand and exponent
parts, the retained-exponent variant adds the exponents in `Int`. The
stored exponent can already have wrapped. This addition recovers the
intended combined exponent when it fits in `Int`.

## Source and tests

- [Product helpers](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/numerical_kernels.jl#L44-L107),
  [exponent-scaled products](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/numerical_kernels.jl#L124-L171),
  [NormL1 threshold caller](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/norml1.jl#L88-L100), and
  [application of retained parts](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/scaled.jl#L559-L565).
- [Bounded-product tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/numerical_kernels.jl#L70-L111)
  cover range changes and zero and non-finite factors in `Float32` and
  `Float64`. [Threshold tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/numerical_kernels.jl#L438-L467)
  exercise the flag and conversion separately.
  [Retained-exponent tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/numerical_kernels.jl#L725-L749)
  exercise the wrapped exponent with `BigFloat`.
