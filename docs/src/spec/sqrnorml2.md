# [SqrNormL2 numerical contract](@id spec-sqrnorml2)

This page gathers the existing numerical rules for `SqrNormL2`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-sqrnorml2-identity)

**Links only until clause conversion.** Constructors and API help: [`SqrNormL2`](@ref).

Source: [`src/functions/sqrnorml2.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/sqrnorml2.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-sqrnorml2-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [the function guide](../smooth.md#Squared-Euclidean-norm).

## [Traits](@id spec-sqrnorml2-traits)

**Links only until clause conversion.** Trait declarations are described by [`SqrNormL2`](@ref) and [the function guide](../smooth.md#Squared-Euclidean-norm).

## [Supported arguments](@id spec-sqrnorml2-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`SqrNormL2`](@ref) and [the function guide](../smooth.md#Squared-Euclidean-norm).

## [Preconditions](@id spec-sqrnorml2-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-sqrnorml2-arguments).

## [Results](@id spec-sqrnorml2-results)

**Links only until clause conversion.** Return values are described by [`SqrNormL2`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-sqrnorml2-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../smooth.md#Squared-Euclidean-norm).

## [Range clauses](@id spec-sqrnorml2-range)

For supported homogeneous `Float32` or `Float64` calls, the value and
numerical outputs use that working precision. The implementation may compute
in a wider type for heterogeneous argument/destination calls, but this is
not a supported mixed-precision contract. In the current kernels, the value
uses the argument's precision; gradients and proximal points use the wider of
the argument and destination precisions. Parameters are rounded to that working
precision when representable, and proximal products are formed from their
unrounded factors before rounding. Combined gradient calls return the standalone
input value and gradient; combined proximal calls return the value at the stored
output. These implementation details do not extend the supported configurations.
Range recovery avoids losing a
representable result to overflow/underflow in a coefficient or difference.
For the supported mode, supply the level, center, step and destination in
the same floating-point type as the argument. See
[numerics](../numerics.md) for FP32/FP64 and zero/non-finite cases.
Shape mismatch is `DimensionMismatch`; complex center, argument or output is
`ArgumentError`.

### [Exact gradient aliasing](@id sqr-norm-l2.gradient.exact-aliasing)

Exact gradient aliasing with the input is rejected.

### Parameter and output buffers

Use
independent parameter/output buffers even where a particular kernel can safely
read an aliased center.

`SqrNormL2` also recovers gradient differences and proximal convex-combination
coefficients.

## [Endpoint clauses](@id spec-sqrnorml2-endpoints)

The [SqrNormL2 guide](@ref Squared-Euclidean-norm) gives the score formula.
See the [shared non-finite KKT rules](../numerics.md#NaN,-infinities-and-zeros) for the referenced rows.

| Case | Retained behavior |
|:--|:--|
| `SqrNormL2`, zero level | Zero standalone value; gradient/prox can still encounter IEEE zero-times-nonfinite arithmetic |
| `SqrNormL2`, infinite level | Gradient is the IEEE `λ(x - y)`: ±Inf for every nonzero residual, subnormal ones included, and NaN at a zero residual; the four-argument and full-point `kkt_violation` scores of `SqrNormL2` take their gradient entry from the ridge-gradient helpers and follow this rule for finite `v` and `x` (the non-finite rules of the `kkt_violation` rows below take precedence, at an infinite level too: `SqrNormL2(Inf)`, `v = [Inf]`, `x = [0.0]` scores `+Inf`), while the three-argument screen does not use the gradient (`kkt_violation(SqrNormL2(Inf), [1.0], 1)` is `1.0`) |

The rules for non-finite entries of `v` and `x` take precedence over this formula
(for `f = SqrNormL2(1.0)`, `v = [-Inf]` and `x = [Inf]` the score is `+Inf` while `norm(v + gradient(f,x), Inf)` is `NaN`).
The precision can differ: for `f = SqrNormL2(1.0, [nextfloat(1.0)])`, `x = Float32[1]` and
`v = Float32[0]` the score works in `Float64` and is `eps(Float64)`, while `gradient(f,x)` rounds the center to
`Float32` and is zero. With an infinite level and finite `v` and `x` the four-argument and full-point scores follow the gradient
oracle (`NaN` at a zero residual, `Inf` elsewhere); a non-finite entry of `v` or `x` still follows the non-finite rules
(for `f = SqrNormL2(Inf)`, `v = [Inf]` and `x = [0.0]` the score is `+Inf`, while the gradient is `NaN`).

A zero level makes the value zero, but does not make every
gradient or endpoint operation immune to IEEE `0*Inf`/`0*NaN` arithmetic.

## [Failure behavior](@id spec-sqrnorml2-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-sqrnorml2-arguments), [Preconditions](@ref spec-sqrnorml2-preconditions) and [Endpoint clauses](@ref spec-sqrnorml2-endpoints), together with the failure behavior in [`SqrNormL2`](@ref).

## [Limitations and non-guarantees](@id spec-sqrnorml2-limitations)

**Links only until clause conversion.** Shared limits on range recovery and arithmetic-order agreement remain in [Range guarantees](@ref numerical-value-range).

## [Clauses owned elsewhere](@id spec-sqrnorml2-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-sqrnorml2-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/ridge_precision.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/ridge_precision.jl)
- [`test/numerical_kernels.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/numerical_kernels.jl)
- [`test/kkt_point.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/kkt_point.jl)

| Clause ID | Status | Evidence |
|:--|:--|:--|
| `sqr-norm-l2.gradient.exact-aliasing` | `guaranteed` | [Gradient and fused entry points](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/sqrnorml2.jl#L143-L156), [family preflight](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/sqrnorml2.jl#L229-L236), [alias guards](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/interface.jl#L241-L249), and [alias rejection and preserved storage tests](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/symmetric_parameters.jl#L302-L309) |

## [History](@id spec-sqrnorml2-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
