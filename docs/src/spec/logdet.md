# [LogDetLoss numerical contract](@id spec-logdet)

This page gathers the existing numerical rules for `LogDetLoss`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-logdet-identity)

**Links only until clause conversion.** Constructors and API help: [`LogDetLoss`](@ref).

Source: [`src/functions/logdet.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/logdet.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-logdet-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [the function guide](../smooth.md#Log-determinant-loss).

## [Traits](@id spec-logdet-traits)

**Links only until clause conversion.** Trait declarations are described by [`LogDetLoss`](@ref) and [the function guide](../smooth.md#Log-determinant-loss).

## [Supported arguments](@id spec-logdet-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`LogDetLoss`](@ref) and [the function guide](../smooth.md#Log-determinant-loss).

## [Preconditions](@id spec-logdet-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-logdet-arguments).

## [Results](@id spec-logdet-results)

**Links only until clause conversion.** Return values are described by [`LogDetLoss`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-logdet-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../smooth.md#Log-determinant-loss).

## [Range clauses](@id spec-logdet-range)

**Links only until clause conversion.** Shared range and accuracy qualifications remain in [Range guarantees](@ref numerical-value-range).

## [Endpoint clauses](@id spec-logdet-endpoints)

The value uses Cholesky. It returns `NaN` at a logical NaN entry and `Inf`
for other domain failures, with an all-NaN gradient in both cases. A zero step
is the identity, including outside the domain. A positive finite step accepts
finite inputs, including asymmetric or indefinite matrices. It diagonalizes
``(X+X^\mathsf{T})/2-\gamma wS`` and replaces each eigenvalue ``d`` with the
positive root of ``z^2-dz-\gamma w=0``.

### [Invalid proximal steps](@id log-det-loss.prox.invalid-step)

Negative, NaN and infinite steps throw
before writing.

### Non-finite inputs at positive steps

Non-finite entries at a positive step also throw before
factorization and writes.

## [Failure behavior](@id spec-logdet-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-logdet-arguments), [Preconditions](@ref spec-logdet-preconditions) and [Endpoint clauses](@ref spec-logdet-endpoints), together with the failure behavior in [`LogDetLoss`](@ref).

## [Limitations and non-guarantees](@id spec-logdet-limitations)

At extreme scales the shifted matrix, its spectrum, or the reconstruction can
exceed the working range. The proximal point is positive definite in exact
arithmetic. Rounding can take the stored point outside the numerical domain,
in which case the fused value follows the standalone domain rule.

## [Clauses owned elsewhere](@id spec-logdet-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-logdet-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/logdet.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/logdet.jl)

| Clause ID | Status | Evidence |
|:--|:--|:--|
| `log-det-loss.prox.invalid-step` | `guaranteed` | [Proximal step check](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/logdet.jl#L121-L127), [Concrete fused and allocating methods](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/logdet.jl#L171-L189), and [invalid steps and untouched destinations](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/logdet.jl#L118-L147) |

## [History](@id spec-logdet-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
