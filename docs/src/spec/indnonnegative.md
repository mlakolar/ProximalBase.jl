# [IndNonnegative numerical contract](@id spec-indnonnegative)

This page carries the existing endpoint rules for `IndNonnegative` verbatim.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-indnonnegative-identity)

**Links only.** See the [`IndNonnegative`](@ref) constructor and API help.

The [source file](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/indnonnegative.jl)
and [API coverage matrix](../api/coverage.md#API-coverage-matrix) record the interface.

## [Definition and domain](@id spec-indnonnegative-definition)

**Links only.** The mathematical definition and ordinary examples are in [`IndNonnegative`](@ref).

## [Traits](@id spec-indnonnegative-traits)

**Links only.** See the [`IndNonnegative`](@ref) API entry and the
[built-in trait table](../function_objects.md#Properties-of-the-built-in-functions).

## [Supported arguments](@id spec-indnonnegative-arguments)

**Links only.** See the [`IndNonnegative`](@ref) API entry for supported representations and the
[operation signatures](../operations.md#Available-operations-at-a-glance).

## [Preconditions](@id spec-indnonnegative-preconditions)

**Links only.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation)
and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with
the family conditions linked in [Supported arguments](@ref spec-indnonnegative-arguments).

## [Results](@id spec-indnonnegative-results)

**Links only.** Return values are described by [`IndNonnegative`](@ref). Shared output and aliasing
rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-indnonnegative-ordinary)

**Links only.** The ordinary mathematical formulas and examples remain in [`IndNonnegative`](@ref).

## [Range clauses](@id spec-indnonnegative-range)

**Links only.** Shared range and accuracy qualifications remain in [Range guarantees](@ref numerical-value-range).

## [Endpoint clauses](@id spec-indnonnegative-endpoints)

| Case | Retained behavior |
|:--|:--|
| `IndNonnegative` | Value `0` at a point with no negative and no `NaN` entry (`-0.0` and `+Inf` included), `Inf` at a negative entry (`-Inf` included) when no entry is `NaN`, `NaN` at a `NaN` entry. The projection is `max(v, 0)` at a step `γ ≥ 0`: `-0.0` and `-Inf` give `+0`, `NaN` stays `NaN`. A negative or `NaN` scalar step throws `ArgumentError` before any write in `prox`, `prox!` and `value_and_prox!`; array-step entries and `prox_block` do not check and project. `Scaled(f, α)`, `α * f` and `f * α` return `f` for a nonnegative `α`, so the same step check applies through them. `value_and_prox!` returns the value at the stored output: `NaN` at a `NaN` output entry, otherwise `0`. The tolerance of the archived value (`v >= -_tol`) is not kept: a negative entry of any size gives `Inf` |

## [Failure behavior](@id spec-indnonnegative-failures)

**Links only.** See [Endpoint clauses](@ref spec-indnonnegative-endpoints), the [`IndNonnegative`](@ref)
API entry and [Predictable failures and partial writes](../numerical_usage.md#Predictable-failures-and-partial-writes).

## [Limitations and non-guarantees](@id spec-indnonnegative-limitations)

**Links only.** See [Range guarantees](@ref numerical-value-range) for shared limits and
the [`IndNonnegative`](@ref) API entry for the available optimality screen.

## [Clauses owned elsewhere](@id spec-indnonnegative-elsewhere)

**Links only.**

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |
| Zero scaling | [A zero scale can change the returned type](../calculus.md#A-zero-scale-can-change-the-returned-type) |

## [Evidence](@id spec-indnonnegative-evidence)

The tests cover endpoint values and projections, step checks, destination storage,
scaling (the indicator returned by `Scaled`) and the optimality screen. [`test/indnonnegative.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/indnonnegative.jl)

## [History](@id spec-indnonnegative-history)

The [restored functions table](../migration.md#Restored-functions) describes differences from the archive.
See the [restoration issue #948](https://github.com/mlakolar/ProximalBase.jl/issues/948) and the
[documentation migration #1019](https://github.com/mlakolar/ProximalBase.jl/issues/1019).
