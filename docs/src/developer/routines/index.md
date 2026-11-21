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
| Scalar absolute-value penalties | [Scalar soft thresholding](@ref routine-soft-threshold) |
| Shrink a norm or project onto a Euclidean ball | [Euclidean shrinkage and radial projection](@ref routine-euclidean-shrink) |
| Select support under penalties and budgets | [Cardinality decisions and selection](@ref routine-cardinality-selection) |
| Predictor-side loss and cached least-squares solves | [Least-squares predictors and proximal solves](@ref routine-predictor-and-factorizations) |
| Products using stored coordinates | [Sparse products](@ref routine-sparse-products) |
| Signed screens and scores at the supplied point | [Point-aware KKT scores](@ref routine-point-aware-scores) |
| Scalar, array and compact proximal steps | [Proximal step dispatch](@ref routine-compact-steps) |
