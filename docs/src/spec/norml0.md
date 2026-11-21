# [NormL0 numerical contract](@id spec-norml0)

This page gathers the existing numerical rules for `NormL0`.
For `diagonal = false`, see the [diagonal-free contract](norml0offdiag.md).
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-norml0-identity)

**Links only until clause conversion.** Constructors and API help: [`NormL0`](@ref).

Source: [`src/functions/norml0.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/norml0.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-norml0-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [`NormL0`](@ref).

## [Traits](@id spec-norml0-traits)

**Links only until clause conversion.** Trait declarations are described by [`NormL0`](@ref).

## [Supported arguments](@id spec-norml0-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`NormL0`](@ref).

## [Preconditions](@id spec-norml0-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-norml0-arguments).

## [Results](@id spec-norml0-results)

**Links only until clause conversion.** Return values are described by [`NormL0`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-norml0-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [`NormL0`](@ref).

## [Range clauses](@id spec-norml0-range)

**Links only until clause conversion.** Shared range and accuracy qualifications remain in [Range guarantees](@ref numerical-value-range).

## [Endpoint clauses](@id spec-norml0-endpoints)

### [Exact threshold comparison](@id norm-l0.prox.exact-threshold)

For `NormL0(λ)` at positive finite `λ` and `γ`, compare the squared
distance cost of setting a finite input `v` to zero with the penalty cost
of keeping it. The comparison uses the exact products of the represented
homogeneous `Float32` or `Float64` inputs. A tie gives positive zero,
including when the candidate costs fall outside the floating-point range.

### Level and step endpoints

| Level and valid step | Proximal entry |
|:--|:--|
| `λ = 0`, including at `γ = Inf` | Identity |
| `λ = Inf`, including at `γ = 0` | Positive zero |
| `0 < λ < Inf`, `γ = 0` | Identity |
| `0 < λ < Inf`, `γ = Inf` | Positive zero |
| `0 < λ < Inf`, `0 < γ < Inf` | Hard thresholding for finite inputs. NaN and signed infinities pass through |

At a zero level the value is zero, including for NaN input. At a positive
level, a NaN entry gives a NaN value ([E2, #560](https://github.com/mlakolar/ProximalBase.jl/issues/560)).
Either infinity is a nonzero entry. With fixed positive finite level and
step, the signed infinity output follows the limit from finite inputs
([E3, #469](https://github.com/mlakolar/ProximalBase.jl/issues/469)).
The constant-zero map at an infinite step is the endpoint convention.
The value sums the entry contributions in the result precision, which
can incur rounding.

Direct array oracles reject negative and NaN steps before writing.
Array steps are matched by linear index and checked in full, with work
proportional to their length. A scalar step is required for a symmetric
sparse iterate output. Value and scalar-step proximal calls between
same-kind sparse iterates traverse stored entries, with the value
counting each symmetric off-diagonal entry twice. A sparse iterate can
omit a zero from storage and read it back as positive zero. Fused calls
return the value at the stored output. Use a nonnegative, non-NaN step
with `prox_block` and in calls through `Scaled`.

## [Failure behavior](@id spec-norml0-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-norml0-arguments), [Preconditions](@ref spec-norml0-preconditions) and [Endpoint clauses](@ref spec-norml0-endpoints), together with the failure behavior in [`NormL0`](@ref).

## [Limitations and non-guarantees](@id spec-norml0-limitations)

**Links only until clause conversion.** Shared limits on range recovery and arithmetic-order agreement remain in [Range guarantees](@ref numerical-value-range).

## [Clauses owned elsewhere](@id spec-norml0-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-norml0-evidence)

| Clause ID | Status | Evidence |
|:--|:--|:--|
| `norm-l0.prox.exact-threshold` | `guaranteed` | [Scalar comparison and prepared cutoff](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/norml0.jl#L52-L126), [Exact objective tests](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/norml0.jl#L34-L75), and [Prepared scalar step tests](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/norml0.jl#L113-L132) |

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/norml0.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/norml0.jl)

## [History](@id spec-norml0-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
