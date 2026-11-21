# [GroupMCP numerical contract](@id spec-groupmcp)

This page is the authored contract of `GroupMCP`.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-groupmcp-identity)

[`GroupMCP`](@ref GroupMCP) is the group minimax concave penalty. See the
[implementation](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/groupmcp.jl),
the [function guide](../group_penalty.md#group-mcp-guide) and the
[API coverage matrix](../api/coverage.md#API-coverage-matrix).

## [Definition and domain](@id spec-groupmcp-definition)

For disjoint nonempty groups ``G_k`` of linear indices,

```math
f(x)=\sum_{k=1}^{K}p_{\lambda_k,a}\bigl(\lVert x_{G_k}\rVert_2\bigr),\qquad \lambda_k=\lambda_0w_k,\qquad
p_{\lambda,a}(t)=\begin{cases}\lambda t-t^2/(2a), & 0\le t\le a\lambda,\\ a\lambda^2/2, & t>a\lambda.\end{cases}
```

The weight multiplies the level of the scalar [MCP](mcp.md), and the shape `a` is not rescaled. Entries in no
group are unpenalized. `GroupMCP(λ0, groups[, w]; a = 3)` converts `λ0`, `w` and `a` to
`T = float(promote_type(typeof(λ0), eltype(w), typeof(a)))`, with default weights ``w_k=\sqrt{\lvert G_k\rvert}``
computed in `T`. The level and the weights are nonnegative and not `NaN`; `a` is positive and not `NaN`, and
`a = Inf` is admitted: then ``p_{\lambda,\infty}(t)=\lambda t`` and `f` is the penalty of
[`GroupNormL2`](groupnorml2.md) with the level `λ0` and the weights `w`. The constructor copies the groups and the
weights and keeps their order. With no groups the value is zero and the proximal map is the identity.

## [Traits](@id spec-groupmcp-traits)

`GroupMCP` is proximable. Its convexity, smoothness and separability traits are false.

## [Supported arguments](@id spec-groupmcp-arguments)

The value, `prox`, `prox!` and `value_and_prox!` take one-based real arrays: `Array` (a matrix is indexed linearly by
the groups), `SparseIterate` and `BlockIterate`, read and written by indexing their entries. The destination of `prox!`
and `value_and_prox!` has a floating-point element type; `prox` allocates one. The step is a nonnegative real scalar.
`prox_block!(out, f, u, k, γ)` and the allocating `prox_block(f, u, k, γ)` take compact vectors in the order of
`groups[k]` and do not change an iterate. `BlockIterate(p, f)`, `BlockIterate(T, p, f)` and `BlockIterate(x, f)` copy
the groups of `f`, which must then partition the entries.

The numerical target is homogeneous CPU `Float32` and `Float64`: input, destination and parameters of one type, and a
step that converts to it exactly. Other combinations take the scaled path below, computed in at least `Float64`; they
have the range of that type and no further mixed-precision guarantee.

## [Preconditions](@id spec-groupmcp-preconditions)

Exact in-place use (`y === x`, and `out === u` for compact blocks) is supported. Other overlaps that
`Base.mightalias` detects are rejected; an overlap that it does not detect is unsupported. For `prox_block!`, `out`
must be independent of iterate storage, as the [pure block contract](@ref block-proximal-oracle) states. A
`GroupMCP` has no buffer, and its oracles do not write to the object.

## [Results](@id spec-groupmcp-results)

`prox!` returns the destination, `prox_block!` returns `out`, and `value_and_prox!` returns the value at the stored
output. That value takes, for each group, the term that the value of the stored output takes: zero for a group stored
as zero or unpenalized, the plateau for a group kept on the ordinary path (formed by the same expression as the
value), and otherwise the term at the norm of the stored entries. For a group scaled on the ordinary path, the squares
of the stored entries are summed as they are written, so the fused value equals `f(y)` up to the order of that sum
(and, for different input and output array types, of the other sums of squares).

## [Ordinary-path clauses](@id spec-groupmcp-ordinary)

For positive finite `λ0`, `w_k` and `γ`, a finite `a` and a finite nonzero group, let `r` be the computed norm of the
group and ``\lambda_k`` the level as formed. The group is replaced by the scalar MCP global proximal point ``t_*`` of
`r` along the group: for ``\gamma<a``,

- `+0` when ``r\le\gamma\lambda_k``;
- unchanged when ``r\ge a\lambda_k``;
- otherwise scaled by ``\min\bigl(1,\ (r-\gamma\lambda_k)\,a/((a-\gamma)\,r)\bigr)``;

and for ``\gamma\ge a``, unchanged when ``r^2>a\gamma\lambda_k^2`` and `+0` otherwise. A tie gives `+0`. The
transition at ``\gamma=a`` takes the hard rule, as for the scalar `MCP`.

These comparisons are exact for the computed norm and the level as formed; they are not exact in the entries, since
the computed norm carries the rounding of the sum of squares. A group within that rounding of a threshold can take
either side.

On the ordinary path (homogeneous `Float32` or `Float64`), the level is ``\lambda_0 w_k`` rounded once in `T` and the
norm is the square root of the sum of squares in `T`. The path applies when the level and ``a\lambda_k`` are normal
numbers, the sum of squares passes the normal-range test of [`GroupNormL2`](groupnorml2.md), and either ``\gamma<a``
with ``\gamma\lambda_k`` at least `2^(emin + p + 1)` (the bound of the scalar firm thresholds) or ``\gamma\ge a`` with a
finite step. The comparisons with ``\gamma\lambda_k`` and ``a\lambda_k`` use the sign of a fused residual, and the
hard comparison is that of the scalar `MCP` (a fixed-width significand comparison when rounded products cannot decide).

The value of a group is ``r(\lambda_k-r/(2a))`` when `r` is below the rounded kink ``a\lambda_k`` and the plateau
``a\lambda_k^2/2`` otherwise. For a singleton group with unit weight, the proximal point agrees with `MCP` up to the
rounding of the firm formula: the decisions are those of the scalar comparisons, and the firm values can differ by a
few units in the last place.

## [Range clauses](@id spec-groupmcp-range)

The scaled path takes the groups with positive finite parameters that the ordinary path does not. It forms the norm as ``n_s2^k`` in the recovery
type `R` (at least `Float64`, wide enough for the input, destination, parameters and step), as `GroupNormL2` does, and
the level as the product of the significands of `λ0` and `w_k` rounded once in `R`, with its exponent kept apart. No
product ``\lambda_k``, ``\gamma\lambda_k``, ``a\lambda_k`` or ``a\gamma\lambda_k^2`` is formed as one number: the
decisions use the signs of ``r-\gamma\lambda_k`` and ``r-a\lambda_k`` in units of `r`, and, for `R = Float64`, an exact
comparison of ``r^2`` with ``a\gamma\lambda_k^2``. For a wider `R` (such as `BigFloat`) that comparison is rounded in
`R`. A firm factor is formed from the residual ``r-\gamma\lambda_k`` in the units of `r`. A firm output entry has at
most the magnitude of its input entry.

So an overflowing finite product ``\lambda_0w_k`` is not a stored infinity, and an underflowing one is not a stored
zero. Under `Float64`, at `a = 3` and `w = [1.0]`:

```@example group_mcp
using ProximalBase
g = GroupMCP(2.0^1000, [1:1], [1.0])        # γλ = 2^-60 < r = 2^-30 < aλ: firm
@assert prox_block(g, [2.0^-30], 1, 2.0^-1060) == [2.0^-30 - 2.0^-60]
h = GroupMCP(2.0^-993, [1:1], [1.0]; a = 1.5 * 2.0^1023)   # r² = 2^60 ≤ aγλ² = 2.25 * 2^60: zero
@assert prox_block(h, [2.0^30], 1, 1.5 * 2.0^1023) == [0.0]
u = GroupMCP(1e-200, [1:2], [1e-200]; a = 1e200)   # λ rounds to 0 in Float64; γλ = 1e-300 > r
@assert prox_block(u, [1e-310, 2e-310], 1, 1e100) == [0.0, 0.0]
```

The group norm is formed in `R`, not in the input type. Entries whose sum of squares overflows the input type
therefore give a representable output in a wider destination (`Float32` entries near `floatmax(Float32)` and a
`Float64` destination), and the homogeneous case recovers the decisions for such groups.

The value returns the ordinary sum of the group terms when it is below `floatmax(T)/2` and, for `Float32` and
`Float64`, at least `n floatmin(T)` for `n` terms rounded from positive finite parameters at nonzero groups. Otherwise
the terms are recomputed as significands and exponents and summed in at least `Float64`, except for a `NaN` sum and an
infinite or `NaN` term, which keep the ordinary sum. A plateau or an interior term out of range is formed in parts. Within a few units in the last place of the overflow
threshold, a value can be `floatmax(T)` or `Inf`.

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

`kkt_violation(f, v, x, k)` returns the signed screen ``\lVert v_{G_k}\rVert_2-\lambda_0w_k`` at a zero group (that of
`GroupNormL2`, with the level and the weight passed as separate factors, so that an overflowing finite product is
recovered) and `abs(v[k])` at an entry in no group. It throws an `ArgumentError` at a nonzero group and in the
full-point form `kkt_violation(f, v, x)`, and `kkt_violation(f, v, i)` throws the `ArgumentError` of a non-separable
group function. Layout, count and extent errors are those of `GroupNormL2`.

## [Limitations and non-guarantees](@id spec-groupmcp-limitations)

`GroupMCP` is not convex. The zero-group screen is a first-order score: a zero value does not imply a global
minimizer. The point-aware score at a nonzero group and the full-point form are not implemented.

`Scaled(GroupMCP(...), α)` multiplies the value by `α` and calls the proximal operator with the effective step
``\alpha\gamma``. An effective step formed from finite positive factors that leaves the normal range is rounded, as for
the scalar `MCP`: an overflow reaches the function as `Inf` (each penalized group becomes `+0`), and an underflow as
zero (the identity). A zero `α` throws an `ArgumentError`. There is no compact block method and no point-aware score
under `Scaled` (`MethodError`).

```@example group_mcp
g = GroupMCP(2.0^-700, [1:2], [1.0])
@assert prox(Scaled(g, 2.0^600), [0.6, 0.8], 2.0^600) == [0.0, 0.0]     # αγ = 2^1200 rounds to Inf
@assert prox_block(g, [0.6, 0.8], 1, 2.0^1023) == [0.6, 0.8]            # a large finite step keeps the group
```

There is no `SymmetricSparseIterate` proximal operator, `restrict`, per-group step or array step (`MethodError` for the
last three). A `SparseIterate` destination is written entry by entry, as for `GroupNormL2`.

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
map with an independent BigFloat minimization of the radial objective (the scalar reference of `test/mcp_reference.jl`
at the exact group norm), by the objective at the computed output within a stated tolerance and, where the reference
decision is stable under the rounding of the norm, by the output itself. It covers group sizes 1 to 512, permuted and
noncontiguous groups with entries in no group, exact ties, the endpoint rows, the two scaled-path examples above and
underflowing or overflowing products, scaling by powers of two across the two paths, singleton groups against `MCP`,
`Array`, `SparseIterate`, `BlockIterate` and compact blocks, refusals before writes, the fused value against the value
at the stored output, zero allocations on the ordinary paths, `Scaled`, the refused methods and the zero-group screen.

## [History](@id spec-groupmcp-history)

The [restoration issue](https://github.com/mlakolar/ProximalBase.jl/issues/1069) and its
[pull request](https://github.com/mlakolar/ProximalBase.jl/pull/1142) record the decisions: the endpoint table, the
refusal of the nonzero-group score, the rounded effective step under `Scaled` and the refusal of non-floating
destinations. The archived `legacy/full` implementation formed the group norm in the input type and could return
`NaN` for representable outputs ([#534](https://github.com/mlakolar/ProximalBase.jl/issues/534)); this implementation
forms it in the recovery type. The infinite-weight convention follows
[#250](https://github.com/mlakolar/ProximalBase.jl/issues/250).
