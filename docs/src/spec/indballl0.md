# [IndBallL0 numerical contract](@id spec-indballl0)

This page carries the existing endpoint rules for `IndBallL0` verbatim.
See [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-indballl0-identity)

**Links only.** See the [`IndBallL0`](@ref) constructor and the
[API coverage matrix](../api/coverage.md#API-coverage-matrix).

## [Definition and domain](@id spec-indballl0-definition)

**Links only.** The cardinality budget and the role of the diagonal are described in the
[`IndBallL0`](@ref) API entry.

## [Traits](@id spec-indballl0-traits)

**Links only.** See [the five function traits](../function_objects.md#The-five-function-traits)
and the [source declarations](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/indballl0.jl).

## [Supported arguments](@id spec-indballl0-arguments)

**Links only.** The [`IndBallL0`](@ref) API entry describes the supported representations
and precision. See the [symmetric iterate guide](@ref symmetric-iterate-guide) for that storage layout.

## [Preconditions](@id spec-indballl0-preconditions)

**Links only.** See [Endpoint clauses](@ref spec-indballl0-endpoints) and the shared
[shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation).

## [Results](@id spec-indballl0-results)

**Links only.** See [Endpoint clauses](@ref spec-indballl0-endpoints) for the value and
stored-output rules and the [`IndBallL0`](@ref) API entry for how output storage is handled.

## [Ordinary-path clauses](@id spec-indballl0-ordinary)

**Links only.** The [`IndBallL0`](@ref) API entry contains the projection rules and examples.

## [Range clauses](@id spec-indballl0-range)

**Links only.** See [Endpoint clauses](@ref spec-indballl0-endpoints) for the comparison of
magnitudes. General bounds are in [Range guarantees](@ref numerical-value-range).

## [Endpoint clauses](@id spec-indballl0-endpoints)

[`IndBallL0`](@ref) requires finite counted data for projection. The kernel
classifies counted entries during the support count. A counted nonfinite
entry throws before the first write. A free diagonal is copied regardless
of its values. In the standalone indicator, a counted NaN returns NaN for
any budget, and either infinity counts as nonzero. In a symmetric input
the selected triangle determines the entries used by both oracles.

The counted-diagonal symmetric projection ranks magnitudes and compares
the savings of adjacent feasible splits using their exchanged entries.
It compares binary significand squares in aligned integer units. This
avoids overflow and underflow of squared magnitudes and preserves exact
ties even with subnormal inputs. The projection writes original values
or positive zero. Both signs of zero count as zero. The fused call returns
the typed zero value directly for the feasible stored result. Steps must
be finite and nonnegative.

## [Failure behavior](@id spec-indballl0-failures)

**Links only.** See [Endpoint clauses](@ref spec-indballl0-endpoints) and
[Predictable failures and partial writes](../numerical_usage.md#Predictable-failures-and-partial-writes).

## [Limitations and non-guarantees](@id spec-indballl0-limitations)

**Links only.** The [`IndBallL0`](@ref) API entry describes storage, overlap, and
scaling limits.

## [Clauses owned elsewhere](@id spec-indballl0-elsewhere)

**Links only.**

| Subject | Owning section |
|:--|:--|
| Precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Shared range limits | [Range guarantees](@ref numerical-value-range) |

## [Evidence](@id spec-indballl0-evidence)

- Endpoint behavior, argument validation, and signed zeros. [`test/indballl0.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/indballl0.jl)
- Exhaustive small-case support comparisons and storage cases. [`test/indballl0_reference.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/indballl0_reference.jl)

## [History](@id spec-indballl0-history)

The [restored functions table](../migration.md#Restored-functions) describes differences from the archive.
The projection was completed in [PR #1062](https://github.com/mlakolar/ProximalBase.jl/pull/1062).
