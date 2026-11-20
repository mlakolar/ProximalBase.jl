# [LogisticLoss numerical contract](@id spec-logisticloss)

This page gathers the existing numerical rules for `LogisticLoss`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-logisticloss-identity)

**Links only until clause conversion.** Constructors and API help: [`LogisticLoss`](@ref).

Source: [`src/functions/logisticloss.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/logisticloss.jl).
The [API coverage matrix](../api/coverage.md#Coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-logisticloss-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [the function guide](../smooth.md#Logistic-regression-loss).

## [Traits](@id spec-logisticloss-traits)

**Links only until clause conversion.** Trait declarations are described by [`LogisticLoss`](@ref) and [the function guide](../smooth.md#Logistic-regression-loss).

## [Supported arguments](@id spec-logisticloss-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`LogisticLoss`](@ref) and [the function guide](../smooth.md#Logistic-regression-loss).

## [Preconditions](@id spec-logisticloss-preconditions)

The constructor borrows model arrays. Same-shape edits between calls must
preserve the validated data domain. Reconstruct the loss for structural changes
and refresh a caller-held predictor when its defining inputs change.
Ordinary calls share private scratch and require synchronization across tasks.
Predictor calls can share a loss across tasks when model data stay read-only
and the outputs are independent. Before writes, the oracles reject nonempty
output overlaps reported by `Base.mightalias` with the input, other output,
model arrays (including CSC values and indices) or scratch. An input predictor or
coefficient vector must also be independent of the scratch.

## [Results](@id spec-logisticloss-results)

**Links only until clause conversion.** Return values are described by [`LogisticLoss`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-logisticloss-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../smooth.md#Logistic-regression-loss).

## [Range clauses](@id spec-logisticloss-range)

The scalar formulas preserve the small tail for saturated responses.
For a response of one half and `0 < abs(η) < 1/1024`, the derivative uses a
centered expression to retain the slope through weighting. A
zero-weight row contributes zero to the value and observation derivative, even
at a nonfinite predictor. At active rows, NaN propagates. Infinite predictors
give the logistic limit: zero loss for a matching binary response, and infinity
otherwise. Standard matrix products may still propagate nonfinite design
entries, including through a zero derivative. For large weights the scalar
loop retains a weighted tail when the unweighted exponential becomes subnormal
or zero. The matrix products, final value sum and outer scaling use ordinary
floating-point arithmetic with their usual rounding and range limits.
The loss supplies first-order oracles. The historical generic scalar-loss,
coordinate and second-order interfaces remain deferred.

## [Endpoint clauses](@id spec-logisticloss-endpoints)

**Links only until clause conversion.** Shared endpoint conventions remain in [NaN, infinities and zeros](../numerics.md#NaN,-infinities-and-zeros).

## [Failure behavior](@id spec-logisticloss-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-logisticloss-arguments), [Preconditions](@ref spec-logisticloss-preconditions) and [Endpoint clauses](@ref spec-logisticloss-endpoints), together with the failure behavior in [`LogisticLoss`](@ref).

## [Limitations and non-guarantees](@id spec-logisticloss-limitations)

**Links only until clause conversion.** Shared limits on range recovery and arithmetic-order agreement remain in [Range guarantees](@ref numerical-value-range).

## [Clauses owned elsewhere](@id spec-logisticloss-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-logisticloss-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/logisticloss.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/logisticloss.jl)

## [History](@id spec-logisticloss-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
