# [IndBox numerical contract](@id spec-indbox)

This page gathers the existing numerical rules for `IndBox`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-indbox-identity)

**Links only until clause conversion.** [`IndBox`](@ref) owns the box projection rules. [`IndBallLinf`](@ref) is its thin constructor with bounds `-r` and `r`; its cases are retained in [Endpoint clauses](@ref spec-indbox-endpoints).

Source: [`src/functions/indbox.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/indbox.jl). The [API coverage matrix](../api/coverage.md#Coverage-matrix) lists the interface.

## [Definition and domain](@id spec-indbox-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [`IndBox`](@ref).

## [Traits](@id spec-indbox-traits)

**Links only until clause conversion.** Trait declarations are described by [`IndBox`](@ref) and [`IndBox`](@ref).

## [Supported arguments](@id spec-indbox-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`IndBox`](@ref) and [`IndBox`](@ref).

## [Preconditions](@id spec-indbox-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-indbox-arguments).

## [Results](@id spec-indbox-results)

**Links only until clause conversion.** Return values are described by [`IndBox`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-indbox-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [`IndBox`](@ref).

## [Range clauses](@id spec-indbox-range)

**Links only until clause conversion.** Shared range and accuracy qualifications remain in [Range guarantees](@ref numerical-value-range).

## [Endpoint clauses](@id spec-indbox-endpoints)

| Case | Retained behavior |
|:--|:--|
| `IndBox` | Exact closed intervals, with no feasibility tolerance. A `NaN` entry gives value `NaN` and stays `NaN` under projection. An infinite entry is feasible only at an unbounded endpoint in its direction. The projection clamps each entry independently of the step and the fused value is evaluated at the stored result. Public scalar-step array oracles reject negative and `NaN` steps before writes. |
| `IndBallLinf` | Inherits the `IndBox` contract with bounds `-r` and `r`. At zero radius, non-NaN entries project to zero with the sign inherited from clipping. At infinite radius both infinite endpoints are feasible. Negative, `NaN` and positive unsigned radii throw `ArgumentError`. |

## [Failure behavior](@id spec-indbox-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-indbox-arguments), [Preconditions](@ref spec-indbox-preconditions) and [Endpoint clauses](@ref spec-indbox-endpoints), together with the failure behavior in [`IndBox`](@ref).

## [Limitations and non-guarantees](@id spec-indbox-limitations)

**Links only until clause conversion.** Shared limits on range recovery and arithmetic-order agreement remain in [Range guarantees](@ref numerical-value-range).

## [Clauses owned elsewhere](@id spec-indbox-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-indbox-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/indbox.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/indbox.jl)

## [History](@id spec-indbox-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
