# [LeastSquaresLoss numerical contract](@id spec-leastsquares)

This page gathers the existing numerical rules for `LeastSquaresLoss`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-leastsquares-identity)

**Links only until clause conversion.** Constructors and API help: [`LeastSquaresLoss`](@ref).

Source: [`src/functions/leastsquares.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/leastsquares.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-leastsquares-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [the function guide](../smooth.md#Least-squares-loss).

## [Traits](@id spec-leastsquares-traits)

**Links only until clause conversion.** Trait declarations are described by [`LeastSquaresLoss`](@ref) and [the function guide](../smooth.md#Least-squares-loss).

## [Supported arguments](@id spec-leastsquares-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`LeastSquaresLoss`](@ref) and [the function guide](../smooth.md#Least-squares-loss).

## [Preconditions](@id spec-leastsquares-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-leastsquares-arguments).

## [Results](@id spec-leastsquares-results)

**Links only until clause conversion.** Return values are described by [`LeastSquaresLoss`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-leastsquares-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../smooth.md#Least-squares-loss).

## [Range clauses](@id spec-leastsquares-range)

The [shared range guarantees](@ref numerical-value-range) describe the squared-residual recovery referenced here.

The `LeastSquaresLoss` value has one more recovery, which needs `β`: where the product `Xβ` overflows
although `β`, `X` and `Y` are finite, `f(β)` and `value_and_gradient!` attempt to form `Xβ` again in units of a power of
two, without recovering every representable value (the smooth-functions guide gives a counterexample). The oracles at a caller-supplied predictor or residual (`ProximalBase.value_from_predictor` and the others
described with the [smooth functions](../smooth.md#Least-squares-at-a-predictor-or-residual-that-the-caller-holds))
take the predictor `z` as given, without `β`: they recover a value whose squares overflow or underflow and, for the
value alone, a difference `z - Y` that overflows, and a predictor with a non-finite entry gives a non-finite value.
For strided `Float32` and `Float64` vectors, these oracles, `f(β)` and `value_and_gradient!` also recompute a value
whose ordinary computation `α s / 2` is above `floatmax(T)/2`, so that the error of the ordinary sum of squares, which grows with
the length, does not decide between a finite value and `Inf`.

The complete [predictor and residual oracle section](../smooth.md#Least-squares-at-a-predictor-or-residual-that-the-caller-holds) retains its domain, rounding, range, gradient and buffer rules. It remains one shared owner for these related operations.

## [Endpoint clauses](@id spec-leastsquares-endpoints)

**Links only until clause conversion.** Shared endpoint conventions remain in [NaN, infinities and zeros](../numerics.md#NaN,-infinities-and-zeros).

## [Failure behavior](@id spec-leastsquares-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-leastsquares-arguments), [Preconditions](@ref spec-leastsquares-preconditions) and [Endpoint clauses](@ref spec-leastsquares-endpoints), together with the failure behavior in [`LeastSquaresLoss`](@ref).

## [Limitations and non-guarantees](@id spec-leastsquares-limitations)

SVD-based
proximal evaluation is intended for finite data; no general non-finite-data
recovery is promised for the decomposition. Value range recovery is separate
from the ordinary matrix-product gradient arithmetic.

## [Clauses owned elsewhere](@id spec-leastsquares-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-leastsquares-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/core_functions.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/core_functions.jl)
- [`test/leastsquares_predictor.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/leastsquares_predictor.jl)
- [`test/kkt_smooth.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/kkt_smooth.jl)

## [History](@id spec-leastsquares-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
