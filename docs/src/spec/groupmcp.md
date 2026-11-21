# [GroupMCP numerical contract](@id spec-groupmcp)

This page is the authored contract of `GroupMCP`.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-groupmcp-identity)

[`GroupMCP`](@ref GroupMCP) is the group minimax concave penalty. See the
[implementation](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/groupmcp.jl),
the [function guide](../group_penalty.md#group-mcp-guide) and the
[API coverage matrix](../api/coverage.md#API-coverage-matrix).

## [Definition and domain](@id spec-groupmcp-definition)

For disjoint nonempty groups ``G_k`` of linear indices and, in the formula below, finite positive ``\lambda_k`` and `a`,

```math
f(x)=\sum_{k=1}^{K}p_{\lambda_k,a}\bigl(\lVert x_{G_k}\rVert_2\bigr),\qquad \lambda_k=\lambda_0w_k,\qquad
p_{\lambda,a}(t)=\begin{cases}\lambda t-t^2/(2a), & 0\le t\le a\lambda,\\ a\lambda^2/2, & t>a\lambda.\end{cases}
```

Zero and infinite levels and weights follow the [endpoint table](@ref spec-groupmcp-endpoints). The weight multiplies
the level of the scalar [MCP](mcp.md), and the shape `a` is not rescaled. Entries in no group are unpenalized.
`GroupMCP(λ0, groups[, w]; a = 3)` converts `λ0`, `w` and `a` to
`T = float(promote_type(typeof(λ0), eltype(w), typeof(a)))`. The default weights ``w_k=\sqrt{\lvert G_k\rvert}``
are computed in `promote_type(T, Float64)` and converted to `T`. The level and the weights are nonnegative and not
`NaN`; `a` is positive and not `NaN`, and `a = Inf` is admitted: then ``p_{\lambda,\infty}(t)=\lambda t`` and `f` is
the penalty of [`GroupNormL2`](groupnorml2.md) with the level `λ0` and the weights `w`. The constructors, the typed
`GroupMCP{T, G}(λ0, a, groups, w)` included, copy the groups and the weights and keep their order; the typed
constructor converts its copied list of groups to `G`. With no groups the value is zero and the proximal map is the
identity.

## [Traits](@id spec-groupmcp-traits)

`GroupMCP` is proximable. Its convexity, smoothness and separability traits are false. The convexity trait describes
the type: with finite positive level, weight and shape the penalty of a group is not convex, while some instances are
convex (`a = Inf` gives a group norm, and a zero level gives the zero function).

## [Supported arguments](@id spec-groupmcp-arguments)

The value, `prox`, `prox!` and `value_and_prox!` take one-based real arrays: `Array` (a matrix is indexed linearly by
the groups), `SparseIterate` and `BlockIterate`, read and written by indexing their entries. The destination of `prox!`
and `value_and_prox!` has a floating-point element type; `prox` allocates one. The step is a nonnegative real scalar.
`prox_block!(out, f, u, k, γ)` and the allocating `prox_block(f, u, k, γ)` take compact vectors in the order of
`groups[k]` and do not change an iterate. `BlockIterate(p, f)`, `BlockIterate(T, p, f)` and `BlockIterate(x, f)` copy
the groups of `f`, which must then partition the entries.

The numerical target is homogeneous CPU `Float32` and `Float64`. For a finite shape, the proximal oracles try the
ordinary path when the input, the destination and the parameters have one of these types and the step converts to it
exactly; otherwise each group with positive finite parameters takes the scaled path below. The value tries its
ordinary path when `float(eltype(x))` is the parameter type (`Float32` or `Float64`), also for an integer array `x`.
At `a = Inf` the guards are those of the `GroupNormL2` kernels, which can take their ordinary path for other type
combinations. Mixed types have the range of the recovery type and no further mixed-precision guarantee.

## [Preconditions](@id spec-groupmcp-preconditions)

Exact in-place use (`y === x`, and `out === u` for compact blocks) is supported. Other overlaps that
`Base.mightalias` detects are rejected; an overlap that it does not detect is unsupported. For `prox_block!`, `out`
must be independent of iterate storage, as the [pure block contract](@ref block-proximal-oracle) states. A
`GroupMCP` has no buffer, and its oracles do not write to the object.

## [Results](@id spec-groupmcp-results)

`prox!` returns the destination, `prox_block!` returns `out`, and `value_and_prox!` returns the value at the stored
output. For a finite shape that value takes, for each group, the term that the value of the stored output takes: zero
for a group stored as zero or unpenalized, the plateau for a group kept on the ordinary path (formed by the same
expression as the value), and otherwise the term at the norm of the stored entries. For a group scaled on the ordinary
path, the squares of the stored entries are summed as they are written, so the fused value equals `f(y)` up to the
order of that sum (and, for different input and output array types, of the other sums of squares). At `a = Inf` each
term is that of `GroupNormL2` at the stored group.

## [Ordinary-path clauses](@id spec-groupmcp-ordinary)

For positive finite `λ0`, `w_k` and `γ`, a finite `a` and a finite nonzero group, let `r` be the computed norm of the
group and ``\lambda_k`` the level as formed. The group is replaced by the scalar MCP global proximal point ``t_*`` of
`r` along the group: for ``\gamma<a``,

- `+0` when ``r\le\gamma\lambda_k``;
- unchanged when ``r\ge a\lambda_k``;
- otherwise scaled by ``\min\bigl(1,\ (r-\gamma\lambda_k)\,a/((a-\gamma)\,r)\bigr)``;

and for ``\gamma\ge a``, unchanged when ``r^2>a\gamma\lambda_k^2`` and `+0` otherwise. A tie gives `+0`. The
transition at ``\gamma=a`` takes the hard rule, as for the scalar `MCP`.

On the `Float32` and `Float64` paths these comparisons are exact for the computed norm and the level as formed (on the
scaled path with a wider recovery type, such as `BigFloat`, the hard comparison is rounded in that type). They are not
exact in the entries, since the computed norm carries the rounding of the sum of squares. A group within that rounding
of a threshold can take either side.

On the ordinary path (homogeneous `Float32` or `Float64`), the level is ``\lambda_0 w_k`` rounded once in `T` and the
norm is the square root of the sum of squares in `T`. The path applies when the rounded level `λ0*w_k` and the rounded
kink `a*(λ0*w_k)` are normal numbers, the sum of squares passes the normal-range test of
[`GroupNormL2`](groupnorml2.md), and either ``\gamma<a`` with the rounded product `γ*(λ0*w_k)` at least
`2^(emin + p + 1)` (the bound of the scalar firm thresholds) or ``\gamma\ge a`` with a finite step. The comparisons with
``\gamma\lambda_k`` and ``a\lambda_k`` use the sign of a fused residual, and the hard comparison is that of the scalar
`MCP` (a fixed-width significand comparison when rounded products cannot decide).

The ordinary value of a group is ``r(\lambda_k-r/(2a))`` when `r` is below the rounded kink ``a\lambda_k`` and the
plateau ``a\lambda_k^2/2`` otherwise; ``r/(2a)`` is formed as `r*(inv(a)/2)` when `inv(a)/2` is a normal number and as
`(r/a)/2` otherwise. For a finite shape and a singleton group with unit weight, the proximal point agrees with `MCP` up to
the rounding of the firm formula: the decisions are those of the scalar comparisons, and the firm values can differ by
a few units in the last place. At `a = Inf` the singleton map is that of `GroupNormL2`, which decides on the threshold
formed as a rounded product and can give `+0` where the scalar `MCP`, deciding on a fused residual, keeps a tiny value
(``\lambda=1+\epsilon``, ``\gamma=1-\epsilon`` and input `1.0` in `Float64`).

## [Range clauses](@id spec-groupmcp-range)

For a finite shape, the scaled path takes the groups with positive finite parameters that the ordinary path does not.
It classifies the group (a `NaN` or an infinite entry, a zero group) and, for a finite nonzero group, forms the norm as
``n_s2^k`` in the recovery type `R` (`promote_type(Float64, ...)` over the input, destination, parameter and step
types, so `Float64` itself for `Float64` data), as `GroupNormL2` does; the scaling by ``2^k`` gives the range. The
level is the product of the significands of `λ0` and `w_k` rounded once in `R`, with its exponent kept apart. The
decisions of the scaled path do not use ``\lambda_k``, ``\gamma\lambda_k``, ``a\lambda_k`` or ``a\gamma\lambda_k^2``
formed as single numbers (the ordinary trial may have formed rounded products before it falls back; they are not
used): they use the signs of ``r-\gamma\lambda_k`` and ``r-a\lambda_k`` in units of `r`, and a comparison of ``r^2``
with ``a\gamma\lambda_k^2`` on the parts, exact for `R = Float64` and rounded in a wider `R`. A firm factor is formed
from the residual ``r-\gamma\lambda_k`` in the units of `r` and is at most one, so a firm output entry has at most the
magnitude of its input entry.

So an overflowing finite product ``\lambda_0w_k`` is not a stored infinity, and an underflowing one is not a stored
zero. Three `Float64` examples, each with its own parameters:

```@example group_mcp
using ProximalBase
# a = 3, λ = 2^1000; the level in units of r = 2^-30 (2^1030) overflows. γλ = 2^-60 < r < aλ: firm
g = GroupMCP(2.0^1000, [1:1], [1.0])
@assert prox_block(g, [2.0^-30], 1, 2.0^-1060) == [2.0^-30 - 2.0^-60]
# a = γ = 1.5 * 2^1023, λ = 2^-993; aγ overflows. r² = 2^60 ≤ aγλ² = 2.25 * 2^60: zero
h = GroupMCP(2.0^-993, [1:1], [1.0]; a = 1.5 * 2.0^1023)
@assert prox_block(h, [2.0^30], 1, 1.5 * 2.0^1023) == [0.0]
# a = 1e200, λ0 = w = 1e-200; λ0*w rounds to 0. γλ = 1e-300 > r: zero
u = GroupMCP(1e-200, [1:2], [1e-200]; a = 1e200)
@assert prox_block(u, [1e-310, 2e-310], 1, 1e100) == [0.0, 0.0]
```

On the scaled path the group norm is the scaled norm in `R`, not a sum of squares in the input type. Entries whose sum
of squares overflows the input type therefore give a representable output in a wider destination (`Float32` entries
near `floatmax(Float32)` and a `Float64` destination), and the homogeneous case recovers the decisions for such groups.

The value returns the ordinary sum of the group terms when it is below `floatmax(T)/2` and, for `Float32` and
`Float64`, at least `n floatmin(T)`. For a finite shape, `n` counts the terms rounded from positive finite parameters at
nonzero groups (a term that rounded to `Inf` included); at `a = Inf` it counts the groups with positive finite level and
weight, zero groups included. Otherwise the terms are recomputed as significands and exponents and summed in at least
`Float64`, with the plateau of a finite-shape group with an infinite entry formed in parts. The ordinary sum is kept
for a `NaN` sum, a `NaN` group, a nonzero group with an infinite weight or level and, at `a = Inf`, a group with an
infinite entry. Within a few units in the last place of the overflow threshold, a value can be `floatmax(T)` or `Inf`.

## [Endpoint clauses](@id spec-groupmcp-endpoints)

The rows apply per group in this order and test the stored parameters only (`iszero` and `isinf` of `λ0`, `w_k`, `a`
and `γ`), never their product. Other groups and the entries in no group are independent. Steps are nonnegative and
not `NaN`.

| Parameters | Proximal group | Group value |
|:--|:--|:--|
| `w_k = 0`, any `λ0` (`Inf` included) and step | Unchanged (`NaN`, `±Inf` and `-0` kept) | `0` |
| `w_k = Inf`, or `λ0 = Inf` and `w_k > 0` | `+0` at every step, `γ = 0` and `γ = Inf` included | `0` at a zero group, `NaN` at a `NaN` entry, `Inf` otherwise |
| `λ0 = 0`, finite `w_k` | Identity | `0`, also at nonfinite entries |
| Positive finite `λ0`, `w_k`; `γ = 0` | Identity | As the rows below |
| Positive finite `λ0`, `w_k`; `γ = Inf` | `+0` (the minimizer of `p` alone) | As the rows below; the stored `+0` has the value `0` |
| Positive finite `λ0`, `w_k`; finite `γ > 0`; a `NaN` entry | Every entry `NaN` | `NaN` |
| Same, a `±Inf` entry and no `NaN` | Unchanged (the norm is above `aλ_k`) | The plateau `aλ_k²/2` |
| Same, a zero group (`±0` entries) | `+0` | `0` |
| Same, finite nonzero group | The ordinary and range clauses above | ``p_{\lambda_k,a}`` at the norm |

At `a = Inf` the finite rows are those of `GroupNormL2` at the level ``\lambda_k``: soft group shrinkage and the value
``\lambda_k\lVert x_{G_k}\rVert_2`` (`Inf` at an infinite entry). The limits of the scalar penalty give these rows: an
infinite step minimizes ``p`` alone, at zero; with an infinite level zero is the only point of finite cost; and
``a\to\infty`` gives ``\lambda t``.

## [Failure behavior](@id spec-groupmcp-failures)

The constructor throws an `ArgumentError` for a negative or `NaN` level or weight (checked before conversion), a
nonpositive or `NaN` shape, an empty group, a nonpositive index, an index in two groups, complex weights and offset axes,
and a `DimensionMismatch` for a weight count that differs from the group count. The array oracles and `prox_block!`
throw before any write: an `ArgumentError` for a negative or `NaN` step, entries that are not real, a destination
whose element type is not floating point (integer and `Rational` destinations), a detected partial overlap and offset
axes; a `DimensionMismatch` for different shapes, groups beyond the argument and a compact length that differs from the
group; a `BoundsError` for a group number out of range or a `Bool`. A `SymmetricSparseIterate` argument or output of a
proximal oracle throws an `ArgumentError` before any write.

At a real array point (a `BlockIterate` and a `SparseIterate` included), `kkt_violation(f, v, x, k)` returns at a zero
group the signed screen of `GroupNormL2`, with the level and the weight passed as separate factors, so that an
overflowing finite product is recovered: ``\lVert v_{G_k}\rVert_2-\lambda_0w_k`` for finite data and coefficient, and
otherwise the rules of that screen (a `NaN` entry gives `NaN`; an infinite entry without a `NaN` gives `Inf`; with a
stored infinite level or weight, finite entries give `-Inf`). At an entry in no group it returns `abs(v[k])`. At such a point it throws an
`ArgumentError` at a nonzero group and in the full-point form `kkt_violation(f, v, x)`; `kkt_violation(f, v, i)` throws
the `ArgumentError` of a non-separable group function. Layout, count and extent errors are those of `GroupNormL2`. A
`SymmetricSparseIterate` point gives a `MethodError`.

## [Limitations and non-guarantees](@id spec-groupmcp-limitations)

With finite positive parameters the penalty is not convex, and the zero-group screen is a first-order score: a zero
value does not imply a global minimizer. The point-aware score at a nonzero group and the full-point form are not
implemented.

`Scaled(GroupMCP(...), α)` multiplies the value by `α` and calls the proximal operator with the effective step
``\alpha\gamma`` as rounded, as for the scalar `MCP`. A product of finite positive factors that leaves the normal range
is passed on as rounded: a nonzero subnormal step is used as it is, and a step that rounds to `Inf` or to zero follows
the infinite-step or zero-step row of the endpoint table (each penalized group with positive finite level and weight
becomes `+0` at `Inf` and is kept at zero, while a stored infinite level or weight still gives `+0`). A negative step
whose product with `α` rounds to `-0` reaches the function as a zero step. A zero `α` throws an `ArgumentError`. There
is no compact block method and no point-aware score under `Scaled` (`MethodError`).

```@example group_mcp
g = GroupMCP(2.0^-700, [1:2], [1.0])
@assert prox(Scaled(g, 2.0^600), [0.6, 0.8], 2.0^600) == [0.0, 0.0]     # αγ = 2^1200 rounds to Inf
@assert prox_block(g, [0.6, 0.8], 1, 2.0^1023) == [0.6, 0.8]            # a large finite step keeps the group
```

There is no `SymmetricSparseIterate` proximal operator, `restrict`, per-group step or array step. `restrict`, and array
steps with `Array`, `SparseIterate` and `BlockIterate` arguments, give a `MethodError`; the shared
`SymmetricSparseIterate` routes reject an array step with an `ArgumentError`. A `SparseIterate` destination is written
entry by entry, as for `GroupNormL2`.

## [Clauses owned elsewhere](@id spec-groupmcp-elsewhere)

| Subject | Owning section |
|:--|:--|
| The scalar penalty and its proximal map | [MCP numerical contract](@ref mcp-contract) |
| Group norms, the normal-range test and soft shrinkage at `a = Inf` | [GroupNormL2 numerical contract](groupnorml2.md) |
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Compact block buffers | [Pure block proximal contract](@ref block-proximal-oracle) |
| Scaling a function | [Scaling a function](@ref scaling-guide) |

## [Evidence](@id spec-groupmcp-evidence)

[`test/groupmcp.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/groupmcp.jl) compares the proximal
map with an independent BigFloat minimization of the radial objective: the scalar reference of
`test/mcp_reference.jl` (a 256-bit candidate comparison) at the group norm computed in 512-bit BigFloat. It compares the
objective at the computed output within a stated tolerance and, where the reference decision is stable under the
rounding of the norm, the output itself. It covers group sizes 1 to 512, permuted and noncontiguous groups with entries
in no group, exact ties, the endpoint rows, the scaled-path examples above and underflowing or overflowing products, a
tiny shape whose reciprocal overflows, scaling by powers of two across the two paths, singleton groups against `MCP`,
`Array`, `SparseIterate`, `BlockIterate` and compact blocks, refusals with the destinations compared to sentinel
copies, the fused value against the value at the stored output, zero allocations on the ordinary paths, `Scaled`, the
refused methods and the zero-group screen.

## [History](@id spec-groupmcp-history)

The [restoration issue](https://github.com/mlakolar/ProximalBase.jl/issues/1069) and its
[pull request](https://github.com/mlakolar/ProximalBase.jl/pull/1142) record the decisions: the endpoint table, the
refusal of the nonzero-group score, the rounded effective step under `Scaled` and the refusal of non-floating
destinations. The archived `legacy/full` implementation formed the group norm in the input type and could return
`NaN` for representable outputs ([#534](https://github.com/mlakolar/ProximalBase.jl/issues/534)); this implementation
forms it in the recovery type. The infinite-weight convention follows
[#250](https://github.com/mlakolar/ProximalBase.jl/issues/250).
