# [SparseGroupNormL2 numerical contract](@id spec-sparsegroupnorml2)

This page gathers the existing numerical rules for `SparseGroupNormL2`.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-sparsegroupnorml2-identity)

**Links only until clause conversion.** See the [SparseGroupNormL2 API](@ref SparseGroupNormL2),
[implementation](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/sparsegroupnorml2.jl)
and [API coverage matrix](../api/coverage.md#API-coverage-matrix).

## [Definition and domain](@id spec-sparsegroupnorml2-definition)

The [function guide](../group_penalty.md#sparse-group-l2-guide) gives the penalty
formula and defines its index set `U`.

Entries outside ``U`` are free in both terms and retain their values in the
proximal output. A zero weight removes the group ℓ₂ term while the ℓ₁ term
still acts on that group. The constructor keeps the order of the groups and
indices and copies the supplied groups and weights. The levels and weights
are converted to one promoted floating type and checked for nonnegativity
and NaN. The default is ``w_k=\sqrt{|G_k|}``. With no groups, the value is
zero and the proximal map is the identity.

## [Traits](@id spec-sparsegroupnorml2-traits)

**Links only until clause conversion.** See the [SparseGroupNormL2 API](@ref SparseGroupNormL2).

## [Supported arguments](@id spec-sparsegroupnorml2-arguments)

`prox` allocates, `prox!` accepts exact in-place use, and
`value_and_prox!` returns the value at the stored destination. Array
oracles accept real one-based CPU arrays, including `SparseIterate` and
`BlockIterate`. Groups index matrices linearly. Full-array partial
overlap is unsupported. Full-array writes can change output iterate activity.
When copying free or identity entries into a floating-point `SparseIterate`,
the array oracle stores negative zeros that would otherwise be read as positive.
Absent positive zeros stay inactive. The soft-thresholded intermediate is written in the
destination type before the group shrink. An explicit integer buffer can
fail conversion here even when the final exact result is integral. Allocating
`prox` uses floating storage for integer input. See the [endpoint rules](@ref sparse-group-l2-endpoints)
for zero or infinite steps, nonfinite entries and infinite parameters.

## [Preconditions](@id spec-sparsegroupnorml2-preconditions)

`prox_block!(out, f, u, k, γ)` uses compact vectors in `groups[k]` order,
returns `out` and accepts `out === u`. It checks the group, lengths,
one-based real buffers, step and detected nonidentical overlap before
writing. The caller supplies an output buffer independent of iterate
storage. `prox_block(f, u, k, γ)` allocates that buffer. To store the
result, the caller writes it into the matching block as a separate step.

## [Results](@id spec-sparsegroupnorml2-results)

The [scoring derivation](../group_penalty.md#Zero-group-screening-and-active-group-mismatch)
gives the finite-data score formulas.

On an ordinary array `k` is an entry index and selects its group.
An uncovered index has score `abs(v[k])`. On a `BlockIterate`,
`k` is a group number and the selected ordered layout must agree.
The full-point `kkt_violation(f, v, x)` checks the full block layout
and returns the maximum across zero, the block scores and the
uncovered-entry magnitudes. The result measures the worst block
or free-coordinate mismatch. The three-argument assumed-zero
form with an integer index throws `ArgumentError`.

## [Ordinary-path clauses](@id spec-sparsegroupnorml2-ordinary)

**Links only until clause conversion.** See the [proximal formula and example](../group_penalty.md#sparse-group-l2-guide)
and the [compact group example](../group_penalty.md#Compact-groups).

## [Range clauses](@id spec-sparsegroupnorml2-range)

See [supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement)
and the [shared range guarantees](@ref numerical-value-range) for the
precision and limits referenced below. The [endpoint table](@ref spec-sparsegroupnorml2-endpoints)
gives the precedence of the proximal cases.

The entrywise threshold at a finite step propagates NaN at the corresponding
entry when the ℓ₁ term is positive. When finite `γλ1` overflows in the
working type, a coupled group retains genuine infinite inputs. An ℓ₁-only
group follows the [NormL1 endpoint rule](@ref spec-norml1-endpoints)
and returns NaN at such an entry. The group shrink distinguishes a product
of finite factors outside the working range from a stored infinite parameter.
The supported homogeneous precision and general range limits above apply.
For mixed-width calls, the stored intermediate and destination conversions
can change the result. The fused value uses the stored destination.

## [Endpoint clauses](@id spec-sparsegroupnorml2-endpoints)

The following rules apply to each covered group in the array and compact
proximal methods, in the order shown. Uncovered entries pass through the
array prox and are ignored by the value.

| Condition | Proximal result |
|:--|:--|
| Negative or NaN step | `ArgumentError` before writes |
| Infinite stored ℓ₁ level, infinite weight, or infinite stored ℓ₂ level with positive weight | Fill the group with `+0`, including at a zero step and with NaN or infinite inputs |
| Zero step or zero penalty after the hard constraints | Identity |
| Infinite step and positive penalty after the hard constraints | Fill the group with `+0` |
| Finite positive step and a positive group ℓ₂ term, with a NaN input | NaN throughout the group |
| Finite positive step and a positive group ℓ₂ term, with an infinite input and no NaN | Keep the entrywise soft-threshold result without a further group shrink |
| Other finite positive steps | Apply the entrywise soft threshold. A zero group term leaves that result in place. A positive group term then applies the group shrink. |

A zero group has value zero under a hard constraint. An entry outside the
hard domain gives an infinite value unless a penalized entry yields NaN.
Zero penalty terms contribute zero. NaN from a penalized entry takes
precedence over infinite contributions.

For the point-aware KKT methods, NaN among the entries read by a group gives
NaN. Otherwise an infinite entry gives `+Inf`. For finite entries, a
hard-constrained zero group gives `-Inf` and an active group outside
the hard domain gives `+Inf`. An active group whose finite group-level
product rounds to infinity also gives `+Inf` in the working type. At a
zero group the signed screen retains the finite factors for range recovery.
The full-point maximum propagates NaN
across groups and includes the magnitudes of the gradient entries at
uncovered coordinates. See the [sparse group lasso guide](@ref sparse-group-l2-guide)
for the finite-data scores.

## [Failure behavior](@id spec-sparsegroupnorml2-failures)

**Links only until clause conversion.** See [preconditions](@ref spec-sparsegroupnorml2-preconditions),
[endpoint clauses](@ref spec-sparsegroupnorml2-endpoints), and
[predictable failures and partial writes](../numerical_usage.md#Predictable-failures-and-partial-writes).

## [Limitations and non-guarantees](@id spec-sparsegroupnorml2-limitations)

`BlockIterate(p, f)`, `BlockIterate(T, p, f)` and `BlockIterate(x, f)`
require the groups to partition the iterate. A `SymmetricSparseIterate`
is unsupported for prox and KKT. The family has no scalar entry prox,
step vectors, or `restrict`. Positive finite `Scaled` supports generic
array value and prox with the scale applied to the step. A scaled product
outside the working range can round before the sparse-group oracle sees it.
For example, a tiny negative step can round to `-0.0` in the wrapper
before the leaf validates the step.
A zero scale throws `ArgumentError` through the generic constructor.
`Scaled` compact-block and KKT calls are unsupported.

## [Clauses owned elsewhere](@id spec-sparsegroupnorml2-elsewhere)

**Links only until clause conversion.** The shared rules are available in the following sections.

| Subject | Owning section |
|:--|:--|
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Compact proximal buffers | [Compact block buffers](@ref block-proximal-oracle) |
| Coordinate and block score routes | [Scoring a coordinate](@ref kkt-screens) |
| Zero-step hard constraints | [Infinite penalties at zero step](@ref infinite-penalty-at-zero-step) |

## [Evidence](@id spec-sparsegroupnorml2-evidence)

The [SparseGroupNormL2 tests](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/sparsegroupnorml2.jl)
cover construction, array and compact proximal calls, endpoint ordering, extreme
ranges, layouts, and point-aware scores.

## [History](@id spec-sparsegroupnorml2-history)

The [SparseGroupNormL2 restore](https://github.com/mlakolar/ProximalBase.jl/pull/1076)
added the original numerical text. The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019)
records its relocation.
