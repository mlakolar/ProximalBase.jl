# [IndBallL2 numerical contract](@id spec-indballl2)

This page gathers the existing numerical rules for `IndBallL2`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-indballl2-identity)

**Links only until clause conversion.** Constructors and API help: [`IndBallL2`](@ref).

Source: [`src/functions/indballl2.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/indballl2.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-indballl2-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [`IndBallL2`](@ref).

## [Traits](@id spec-indballl2-traits)

**Links only until clause conversion.** Trait declarations are described by [`IndBallL2`](@ref).

## [Supported arguments](@id spec-indballl2-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`IndBallL2`](@ref).

## [Preconditions](@id spec-indballl2-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-indballl2-arguments).

## [Results](@id spec-indballl2-results)

**Links only until clause conversion.** Return values are described by [`IndBallL2`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-indballl2-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [`IndBallL2`](@ref).

## [Range clauses](@id spec-indballl2-range)

For `IndBallL2`, the value and projection compare the norm with the radius in
scaled units when the ordinary sum of squares overflows or is too small.
The domain has no fixed feasibility tolerance. Within the rounding error of
the norm, the comparison can fall on either side of the boundary.
`value_and_prox!` evaluates the stored point. For dense vectors of four
entries with matching `Float32` or `Float64` input, output and radius, the
projection checks the rounded coefficients when the input sum of squares is
ordinary and the finite radius is at least
`2sqrt(4floatmin(T) / eps(T))`. If the rounded coefficients fail the
comparison, it scales them once more toward zero. The stored point in that
case passes the same norm comparison. Other projections can return a stored
point outside the computed boundary, giving `Inf` in the fused call.
A positive finite radius projects a NaN input to an all-NaN output,
including implicit zeros of a `SparseIterate`.

## [Endpoint clauses](@id spec-indballl2-endpoints)

| Case | Retained behavior |
|:--|:--|
| `IndBallL2` | Projects at any nonnegative scalar step, including zero and infinity. A negative or NaN scalar step throws before writes, also for `α * IndBallL2(r)` with `α ≥ 0`, which is the indicator itself. A NaN entry gives value NaN at any radius. Zero radius projects to positive zeros. Infinite radius copies the input, and the value is zero if the input has no NaN. At positive finite radius, a single infinite entry gives the signed radius at that coordinate and zeros elsewhere, and multiple infinite entries or any NaN give an all-NaN projection. |

## [Failure behavior](@id spec-indballl2-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-indballl2-arguments), [Preconditions](@ref spec-indballl2-preconditions) and [Endpoint clauses](@ref spec-indballl2-endpoints), together with the failure behavior in [`IndBallL2`](@ref).

## [Limitations and non-guarantees](@id spec-indballl2-limitations)

**Links only until clause conversion.** Shared limits on range recovery and arithmetic-order agreement remain in [Range guarantees](@ref numerical-value-range).

## [Clauses owned elsewhere](@id spec-indballl2-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-indballl2-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/indballl2.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/indballl2.jl)

## [History](@id spec-indballl2-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
