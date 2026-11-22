# [GroupNormL2 numerical contract](@id spec-groupnorml2)

This page gathers the existing numerical rules for `GroupNormL2`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-groupnorml2-identity)

**Links only until clause conversion.** Constructors and API help: [`GroupNormL2`](@ref).

Source: [`src/functions/groupnorml2.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/groupnorml2.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-groupnorml2-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [the function guide](../group_penalty.md#group-l2-guide).

## [Traits](@id spec-groupnorml2-traits)

**Links only until clause conversion.** Trait declarations are described by [`GroupNormL2`](@ref) and [the function guide](../group_penalty.md#group-l2-guide).

## [Supported arguments](@id spec-groupnorml2-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`GroupNormL2`](@ref) and [the function guide](../group_penalty.md#group-l2-guide).

## [Preconditions](@id spec-groupnorml2-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-groupnorml2-arguments).

## [Results](@id spec-groupnorml2-results)

**Links only until clause conversion.** Return values are described by [`GroupNormL2`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-groupnorml2-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../group_penalty.md#group-l2-guide).

## [Range clauses](@id spec-groupnorml2-range)

The [shared range guarantees](@ref numerical-value-range) describe the checks referenced here.

`GroupNormL2` values apply the same check to their group terms `λ0*w[k]*‖x[G_k]‖₂`,
and recompute a sum below `K*floatmin(T)` for `K` groups from the terms' exponents.

The proximal operators of `GroupNormL2` recompute, for finite inputs, a group norm whose sum of
squares overflows or underflows and a threshold `γ*λ0*w[k]` outside the range, in
power-of-two units; the ordinary path needs a normal threshold and a normal sum of
squares, and is otherwise unchanged.

## [Endpoint clauses](@id spec-groupnorml2-endpoints)

| Case | Retained behavior |
|:--|:--|
| `GroupNormL2`, zero weight | Unpenalized group: returned unchanged (`NaN`/`±Inf` included), zero value |
| `GroupNormL2`, infinite weight, or infinite level with a positive weight | Group fixed at `+0` at every nonnegative step, `γ = 0` included (see [zero-step rule](@ref infinite-penalty-at-zero-step)), also for `NaN`/`±Inf` inputs; value `0` at zero, `Inf` elsewhere (`NaN` at a `NaN` entry) |
| `GroupNormL2`, positive finite level and weight, `γ = Inf` | [Fill the group with `+0`](@ref group-norm-l2.prox.infinite-step) |
| `GroupNormL2`, positive finite parameters and step | A `NaN` entry makes the group `NaN`; a `±Inf` entry (no `NaN`) leaves the group unchanged, also when the finite threshold's product overflows |
| `GroupNormL2`, negative or `NaN` step | `ArgumentError` before any write, in `prox!`, `value_and_prox!` and `prox_block!` |

| Case | Result for group ``k`` |
|:--|:--|
| ``w_k=0`` | Unpenalized: returned unchanged, `NaN` and `±Inf` included; value zero |
| ``w_k=\infty``, or ``\lambda_0=\infty`` with ``w_k>0`` | Fixed at `+0` at every nonnegative step, ``\gamma=0`` included, also for `NaN` and `±Inf` inputs (see [infinite penalties at zero step](@ref infinite-penalty-at-zero-step)) |
| ``\lambda_0=0`` or ``\gamma=0``, finite parameters | The identity |
| ``\gamma=\infty``, positive finite level and weight | Fixed at `+0` |
| A `NaN` entry, positive finite level, weight and step | Every entry of the group becomes `NaN` |
| A `±Inf` entry and no `NaN`, positive finite level, weight and step | Returned unchanged: the norm is infinite and the threshold ``\gamma\lambda_0w_k`` is finite, also when its floating-point product overflows |
| ``\gamma<0`` or ``\gamma=`` `NaN` | `ArgumentError` before any write |

The common interpretation of [infinite penalties at zero step](@ref infinite-penalty-at-zero-step)
applies. A finite zero group becomes `+0` under positive finite parameters.

### [Proximal result at an infinite step](@id group-norm-l2.prox.infinite-step)

For a group with a positive finite level and weight, the infinite step
`γ = Inf` makes its proximal result `+0` throughout.

### [Values at infinite weights or levels](@id group-norm-l2.value.infinite-penalty)

An infinite-weight or infinite-level group's value is zero at a zero group, `NaN`
when it contains `NaN`, and `Inf` otherwise; a zero weight overrides these value
rules and contributes zero.

### Finite-input range recovery

For finite inputs, norms whose sums of squares overflow or underflow and
thresholds outside the floating-point range are recomputed in power-of-two
units, so that a representable proximal point is returned accurately. Values recover terms that round below `floatmin`: for `Float32` and `Float64`,
a sum below `K*floatmin`, where `K` is the total number of groups, is recomputed
using term exponents. Values at or above `floatmax/2` are recomputed in
power-of-two units and accumulated in at least `Float64`. Within a few eps of
the overflow threshold the result may be `floatmax` or `Inf`, regardless of
which one the exact value rounds to. See [value range limits](@ref numerical-value-range).

`GroupNormL2` does not apply to a `SymmetricSparseIterate` (its proximal operators
throw an `ArgumentError` before anything is written), and it has no
`restrict`, no per-group step and no compact block method under `Scaled`.
With a positive finite `α`, `Scaled(GroupNormL2(...), α)` multiplies the value by
`α`, computes the array proximal operator with the step `αγ`, and scores a group by evaluating the score of
`GroupNormL2` on `v/α` and multiplying it by `α` (see the [scaling guide](@ref scaling-guide)
for the limits and the unsupported cases); `test/kkt_block.jl` checks these scores
against the definition of the scaled function. A zero `α` throws an
`ArgumentError`, and an infinite `α` has no endpoint rule of its own. The
[group-lasso example](../examples/group_lasso_prox.md) checks the value, the
proximal operators, the block operations and the screen against their formulas.

## [Failure behavior](@id spec-groupnorml2-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-groupnorml2-arguments), [Preconditions](@ref spec-groupnorml2-preconditions) and [Endpoint clauses](@ref spec-groupnorml2-endpoints), together with the failure behavior in [`GroupNormL2`](@ref).

## [Limitations and non-guarantees](@id spec-groupnorml2-limitations)

**Links only until clause conversion.** Shared limits on range recovery and arithmetic-order agreement remain in [Range guarantees](@ref numerical-value-range).

## [Clauses owned elsewhere](@id spec-groupnorml2-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-groupnorml2-evidence)

| Clause ID | Status | Evidence |
|:--|:--|:--|
| `group-norm-l2.value.infinite-penalty` | `guaranteed` | [Group value route](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/groupnorml2.jl#L185-L195), [Zero-weight exit](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/numerical_kernels.jl#L960-L964), [Infinite penalty value classifier](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/numerical_kernels.jl#L997-L1010), and [Infinite parameter and zero-weight value tests](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/groupnorml2.jl#L629-L652) |
| `group-norm-l2.prox.infinite-step` | `guaranteed` | [Infinite-step exit](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/numerical_kernels.jl#L907-L923), [Array result](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/groupnorml2.jl#L216-L217), and [Compact positive-zero result](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/groupnorml2.jl#L271-L276) |

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/groupnorml2.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/groupnorml2.jl)
- [`test/kkt_block.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/kkt_block.jl)

## [History](@id spec-groupnorml2-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
