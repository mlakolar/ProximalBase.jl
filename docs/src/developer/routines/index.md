# [Numerical routines](@id numerical-routines)

These pages explain how selected routines compute their results in the current
implementation. Each page identifies the source revision and tests behind its
description. For supported inputs and numerical behavior, follow the
[owning numerical contracts](../../spec/index.md).

| Calculation | Routine |
|:--|:--|
| Products of steps, levels and weights | [Scaled products](@ref routine-scaled-products) |
| Sums of squares at large and small magnitudes | [Squared residuals in power-of-two units](@ref routine-squared-residuals) |
| The level used for projection and clipping | [The shared ℓ₁-ball level](@ref routine-l1-ball-level) |
