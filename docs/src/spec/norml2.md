# [NormL2 numerical contract](@id spec-norml2)

This page gathers the existing numerical rules for `NormL2`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-norml2-identity)

**Links only until clause conversion.** Constructors and API help: [`NormL2`](@ref).

Source: [`src/functions/norml2.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/norml2.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-norml2-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [the function guide](../group_penalty.md#norm-l2-guide).

## [Traits](@id spec-norml2-traits)

**Links only until clause conversion.** Trait declarations are described by [`NormL2`](@ref) and [the function guide](../group_penalty.md#norm-l2-guide).

## [Supported arguments](@id spec-norml2-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`NormL2`](@ref) and [the function guide](../group_penalty.md#norm-l2-guide).

## [Preconditions](@id spec-norml2-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-norml2-arguments).

## [Results](@id spec-norml2-results)

**Links only until clause conversion.** Return values are described by [`NormL2`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-norml2-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../group_penalty.md#norm-l2-guide).

## [Range clauses](@id spec-norml2-range)

The [shared range guarantees](@ref numerical-value-range) describe the squared-residual kernel referenced here.

`NormL2` values recompute a sum of squares that overflows or falls below the normal
range from the squared-residual kernel above, in units of the largest stored entry.
**Implementation note.** On a `SparseIterate` or `SymmetricSparseIterate`, the
ordinary sum and recovery work from stored entries only. Each stored
off-diagonal symmetric coordinate occupies one slot and contributes with
multiplicity two to the squared norm ([#808](https://github.com/mlakolar/ProximalBase.jl/issues/808)).

## [Endpoint clauses](@id spec-norml2-endpoints)

| Case | Retained behavior |
|:--|:--|
| `NormL2` | The rules of one `GroupNormL2` group with weight one (rows below); value `0` at zero level; no proximal operator for a `SymmetricSparseIterate` (`ArgumentError` before any write) |

### [Zero-level value](@id norm-l2.value.zero-level)

A zero level gives value zero.

### Other level and step cases

An infinite level gives zero at the zero argument (including an empty array),
`NaN` at a `NaN` entry, and `Inf` otherwise. With unit group weight, the
[endpoint table](@ref group-l2-endpoints) also describes the norm's prox;
in particular zero level takes precedence over an infinite step.

The “rows below” in the retained table refer to the [GroupNormL2 endpoint table](@ref group-l2-endpoints).

## [Failure behavior](@id spec-norml2-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-norml2-arguments), [Preconditions](@ref spec-norml2-preconditions) and [Endpoint clauses](@ref spec-norml2-endpoints), together with the failure behavior in [`NormL2`](@ref).

## [Limitations and non-guarantees](@id spec-norml2-limitations)

**Links only until clause conversion.** Shared limits on range recovery and arithmetic-order agreement remain in [Range guarantees](@ref numerical-value-range).

## [Clauses owned elsewhere](@id spec-norml2-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-norml2-evidence)

| Clause ID | Status | Evidence |
|:--|:--|:--|
| `norm-l2.value.zero-level` | `guaranteed` | [Zero-level value branch](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/numerical_kernels.jl#L1017-L1026) and [Zero-level value and non-finite inputs](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/norml2.jl#L358-L369) |

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/norml2.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/norml2.jl)
- [`test/kkt_block.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/kkt_block.jl)
- [`test/sparse_value_cost.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/sparse_value_cost.jl)

## [History](@id spec-norml2-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
