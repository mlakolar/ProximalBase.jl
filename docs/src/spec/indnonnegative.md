# [IndNonnegative numerical contract](@id spec-indnonnegative)

This page gathers the existing numerical rules for `IndNonnegative`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-indnonnegative-identity)

**Links only until clause conversion.** Constructors and API help: [`IndNonnegative`](@ref).

Source: [`src/functions/indnonnegative.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/indnonnegative.jl).
The [API coverage matrix](../api/coverage.md#Coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-indnonnegative-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [`IndNonnegative`](@ref).

## [Traits](@id spec-indnonnegative-traits)

**Links only until clause conversion.** Trait declarations are described by [`IndNonnegative`](@ref) and [`IndNonnegative`](@ref).

## [Supported arguments](@id spec-indnonnegative-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`IndNonnegative`](@ref) and [`IndNonnegative`](@ref).

## [Preconditions](@id spec-indnonnegative-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-indnonnegative-arguments).

## [Results](@id spec-indnonnegative-results)

**Links only until clause conversion.** Return values are described by [`IndNonnegative`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-indnonnegative-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [`IndNonnegative`](@ref).

## [Range clauses](@id spec-indnonnegative-range)

**Links only until clause conversion.** Shared range and accuracy qualifications remain in [Range guarantees](@ref numerical-value-range).

## [Endpoint clauses](@id spec-indnonnegative-endpoints)

| Case | Retained behavior |
|:--|:--|
| `IndNonnegative` | Value `0` at a point with no negative and no `NaN` entry (`-0.0` and `+Inf` included), `Inf` at a negative entry (`-Inf` included) when no entry is `NaN`, `NaN` at a `NaN` entry. The projection is `max(v, 0)` at a step `γ ≥ 0`: `-0.0` and `-Inf` give `+0`, `NaN` stays `NaN`. A negative or `NaN` scalar step throws `ArgumentError` before any write in `prox`, `prox!` and `value_and_prox!`; array-step entries, `prox_block` and `Scaled` routes do not check and project. `value_and_prox!` returns the value at the stored output: `NaN` at a `NaN` output entry, otherwise `0`. The tolerance of the archived value (`v >= -_tol`) is not kept: a negative entry of any size gives `Inf` |

## [Failure behavior](@id spec-indnonnegative-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-indnonnegative-arguments), [Preconditions](@ref spec-indnonnegative-preconditions) and [Endpoint clauses](@ref spec-indnonnegative-endpoints), together with the failure behavior in [`IndNonnegative`](@ref).

## [Limitations and non-guarantees](@id spec-indnonnegative-limitations)

**Links only until clause conversion.** Shared limits on range recovery and arithmetic-order agreement remain in [Range guarantees](@ref numerical-value-range).

## [Clauses owned elsewhere](@id spec-indnonnegative-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-indnonnegative-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/indnonnegative.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/indnonnegative.jl)

## [History](@id spec-indnonnegative-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
