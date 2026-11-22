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
f(x)=\sum_{k=1}^{K}p_{\lambda_k,a}\bigl(\lVert x_{G_k}\rVert_2\bigr),\qquad
p_{\lambda,a}(t)=\begin{cases}\lambda t-t^2/(2a), & 0\le t\le a\lambda,\\ a\lambda^2/2, & t>a\lambda,\end{cases}
```

where ``\lambda_k`` is the effective coefficient of group `k`: ``\lambda_0w_k`` when `λ0` and `w_k` are finite and
positive, and otherwise the [endpoint table](@ref spec-groupmcp-endpoints) gives the group's map and term. The formula
for ``p`` is that of finite positive ``\lambda`` and `a`; at `a = Inf` and finite positive ``\lambda`` it is
``\lambda t``. The shape `a` is not rescaled by the weights. Entries in no group are unpenalized.

**Zero weights.** A group with `w_k = 0` is unpenalized, whatever `λ0` (`Inf` included): the proximal map leaves its
entries unchanged and its term is zero. For the group's proximal map and term, this rule comes before the others on
this page.

`GroupMCP(λ0, groups, w; a = 3)` converts `λ0`, `w` and `a` to
`T = float(promote_type(typeof(λ0), eltype(w), typeof(a)))`. `GroupMCP(λ0, groups; a = 3)` uses `T = float(promote_type(typeof(λ0), typeof(a)))` and the weights
``w_k=\sqrt{\lvert G_k\rvert}``, computed in `promote_type(T, Float64)` and converted to `T`. The level and the weights
are nonnegative and not `NaN`; `a` is positive and not `NaN`. These two constructors copy the weights, the list of
groups and each group that is not a range, and keep their order; a range group can be kept as given (when every group
is a range of one concrete type) and must not be mutated afterwards. With no groups the value is zero and the proximal map is the identity.

## [Traits](@id spec-groupmcp-traits)

`GroupMCP` is proximable, and its convexity, smoothness and separability traits are false. Instances with a positively
weighted group of finite positive level and finite shape are generally nonconvex.

## [Supported arguments](@id spec-groupmcp-arguments)

The value, `prox`, `prox!` and `value_and_prox!` take one-based real arrays: `Array` (a matrix is indexed linearly by
the groups), `SparseIterate` and `BlockIterate`, read and written by indexing their entries. The destination of `prox!`
and `value_and_prox!` has a floating-point element type; `prox` allocates one. The step is a nonnegative real scalar.
`prox_block!(out, f, u, k, γ)` and the allocating `prox_block(f, u, k, γ)` take compact vectors in the order of
`groups[k]` and do not change an iterate. `BlockIterate(p, f)`, `BlockIterate(T, p, f)` and `BlockIterate(x, f)` take
the groups of `f`, which must then partition the entries, with the copying rule of the constructors above.

The numerical target is homogeneous CPU `Float32` and `Float64` input, destination, parameters and step. Some other type
combinations dispatch, without a precision guarantee.

## [Preconditions](@id spec-groupmcp-preconditions)

The typed constructor `GroupMCP{T, G}(λ0, a, groups, w)` validates its arguments and stores `groups` and `w` without
copying them: do not mutate the groups or the weights passed to it afterwards.

Exact in-place use (`y === x`, and `out === u` for compact blocks) is supported. Other overlaps that
`Base.mightalias` detects are rejected; an overlap that it does not detect is unsupported. For `prox_block!`, `out`
must be independent of iterate storage, as the [pure block contract](@ref block-proximal-oracle) states. A `GroupMCP`
has no buffer, and its oracles do not write to the object.

## [Results](@id spec-groupmcp-results)

`prox!` returns the destination, `prox_block!` returns `out`, and `value_and_prox!` returns the value at the stored
output: it equals `f(y)` up to the order in which sums of squares are accumulated.

## [Ordinary-path clauses](@id spec-groupmcp-ordinary)

For positive finite `λ0`, `w_k` and `γ`, a finite `a` and a finite nonzero group, let `r` be the computed norm of the
group and ``\lambda_k`` the level as formed. The group is replaced by a minimizer ``t_*`` of
``p_{\lambda_k,a}(t)+(t-r)^2/(2\gamma)`` over ``t\ge0``, along the group: for ``\gamma<a``,

- `+0` when ``r\le\gamma\lambda_k``;
- unchanged when ``r\ge a\lambda_k``;
- otherwise scaled by ``(r-\gamma\lambda_k)\,a/((a-\gamma)\,r)``, at most one;

and for ``\gamma\ge a``, unchanged when ``r^2>a\gamma\lambda_k^2`` and `+0` otherwise. At a tie the result is `+0`.
At ``\gamma=a`` the hard rule applies, as for the scalar `MCP`.

When the input, the destination, the parameters and the step are `Float32` or `Float64`, these comparisons are exact
for the computed norm and the level as formed. They are not exact in the entries: the computed norm carries the
rounding of the sum of squares, and a group within that rounding of a threshold can take either side.

The value of such a group is ``p_{\lambda_k,a}`` at its computed norm, up to rounding.

At `a = Inf` each group follows the rules of [`GroupNormL2`](groupnorml2.md) at the coefficient ``\lambda_k``: soft
group shrinkage and the value ``\lambda_k\lVert x_{G_k}\rVert_2``.

## [Range clauses](@id spec-groupmcp-range)

For positive finite stored `λ0` and `w_k`, the decisions above do not require the products ``\lambda_0w_k``,
``\gamma\lambda_k``, ``a\lambda_k`` or ``a\gamma\lambda_k^2`` to be representable: an overflowing finite product is not
treated as a stored infinity, and an underflowing one is not treated as a stored zero. A scaled norm is the recovery
route, when needed, for a group whose sum of squares overflows or underflows the input type. The stored result is limited by the range of the
destination type. Three `Float64` examples, each with its own parameters:

```@example group_mcp
using ProximalBase
# a = 3, λ = 2^1000; the level in units of r = 2^-30 (2^1030) overflows. γλ = 2^-60 < r < aλ: firm
g = GroupMCP(2.0^1000, [1:1], [1.0])
@assert prox_block(g, [2.0^-30], 1, 2.0^-1060) == [2.0^-30 - 2.0^-60]
# a = γ = 1.5 * 2^1023, λ = 2^-993; aγ overflows. r² = 2^60 ≤ aγλ² = 2.25 * 2^60: zero
h = GroupMCP(2.0^-993, [1:1], [1.0]; a = 1.5 * 2.0^1023)
@assert prox_block(h, [2.0^30], 1, 1.5 * 2.0^1023) == [0.0]
# a = 1e200, λ0 = w = 1e-200; λ0*w rounds to 0. γλ ≈ 1e-300 > r: zero
u = GroupMCP(1e-200, [1:2], [1e-200]; a = 1e200)
@assert prox_block(u, [1e-310, 2e-310], 1, 1e100) == [0.0, 0.0]
```

When the ordinary sum of the group terms is at or above `floatmax(T)/2` or, for `Float32` and `Float64`, small enough
that terms rounded into the subnormal range may have lost accuracy, the value is recomputed from scaled terms in at least `Float64`, unless the
endpoint table already makes the value `NaN` or `Inf`. Within a few units in the last place of the overflow threshold, a
value can be `floatmax(T)` or `Inf`.

## [Endpoint clauses](@id spec-groupmcp-endpoints)

The rows apply per group in this order and test the stored parameters only (`iszero` and `isinf` of `λ0`, `w_k`, `a`
and `γ`), never their product. Other groups and the entries in no group are independent. Steps are nonnegative and
not `NaN`.

| Parameters | Proximal group | Group value |
|:--|:--|:--|
| `w_k = 0` (see Zero weights above) | Unchanged | `0` |
| `w_k = Inf`, or `λ0 = Inf` and `w_k > 0` | `+0` at every step, `γ = 0` and `γ = Inf` included | `0` at a zero group, `NaN` at a `NaN` entry, `Inf` otherwise |
| `λ0 = 0`, finite `w_k > 0` | Identity | `0`, also at nonfinite entries |
| Positive finite `λ0`, `w_k`; `γ = 0` | Identity | As the rows below |
| Positive finite `λ0`, `w_k`; `γ = Inf` | `+0` | As the rows below |
| Positive finite `λ0`, `w_k`; finite `γ > 0`; a `NaN` entry | Every entry `NaN` | `NaN` |
| Same, a `±Inf` entry and no `NaN` | Unchanged | The plateau `aλ_k²/2` (`Inf` at `a = Inf`) |
| Same, a zero group (`±0` entries) | `+0` | `0` |
| Same, finite nonzero group | The ordinary and range clauses above | ``p_{\lambda_k,a}`` at the norm |

## [Failure behavior](@id spec-groupmcp-failures)

The constructors throw an `ArgumentError` for a negative or `NaN` level or weight, a nonpositive or `NaN` shape, an
empty group, a nonpositive index, an index in two groups and offset axes, and a `DimensionMismatch` for a weight count
that differs from the group count. The outer constructors check the level and the weights before conversion and also
throw an `ArgumentError` for complex weights. The array
oracles and `prox_block!` throw before any write: an `ArgumentError` for a negative or `NaN` step, entries that are not
real, a destination whose element type is not floating point, a detected partial overlap and offset axes; a
`DimensionMismatch` for different shapes, groups beyond the argument and a compact length that differs from the group;
a `BoundsError` for a group number out of range or a `Bool`. A `SymmetricSparseIterate` argument or output of a
proximal oracle with a scalar step throws an `ArgumentError` before any write.

At a real array point (`Array`, `SparseIterate` or `BlockIterate`), `kkt_violation(f, v, x, k)` returns at a zero group
the signed screen of `GroupNormL2` with the level and the weight as separate factors: ``\lVert v_{G_k}\rVert_2-\lambda_0w_k``
for finite converted gradient entries and finite `λ0` and `w_k`, up to rounding; ``\lVert v_{G_k}\rVert_2`` for a zero weight, whatever `λ0`; otherwise
the rules of that screen for `NaN` and infinite values. At an entry in no group it returns `abs(v[k])`. It throws an `ArgumentError` at a nonzero group, as do
`kkt_violation(f, v, x)` and `kkt_violation(f, v, i)`. Layout, count and extent errors are those of `GroupNormL2`. A
call with a `SymmetricSparseIterate` point that passes the shared argument checks throws a `MethodError`.

### [Working type of the zero-group screen](@id group-mcp.kkt.entry-classification)

The zero-group screen checks finiteness after converting the gradient
entries to `W = float(promote_type(eltype(v), eltype(x)))`. A finite stored
entry that overflows on conversion is treated as infinite.

## [Limitations and non-guarantees](@id spec-groupmcp-limitations)

The zero-group screen is a first-order score: a zero value does not imply a global minimizer. The score at a nonzero
group and the full-point form are not implemented.

`Scaled(GroupMCP(...), α)` multiplies the value by `α` and calls the proximal operator with the effective step
``\alpha\gamma`` as rounded, as for the scalar `MCP`; the rounded step then follows this page (a step that rounds to
zero or `Inf` takes the zero-step or infinite-step row). A zero `α` throws an `ArgumentError`. There is no compact block
method and no point-aware score under `Scaled` (`MethodError`).

```@example group_mcp
g = GroupMCP(2.0^-700, [1:2], [1.0])
@assert prox(Scaled(g, 2.0^600), [0.6, 0.8], 2.0^600) == [0.0, 0.0]     # αγ = 2^1200 rounds to Inf
@assert prox_block(g, [0.6, 0.8], 1, 2.0^1023) == [0.6, 0.8]            # a large finite step keeps the group
```

There is no `restrict` and no array (per-entry or per-group) step: calls that pass the shared argument checks throw a
`MethodError`, except where a
route shared with other functions rejects a `SymmetricSparseIterate` destination, or the allocating
`prox(f, S, steps)`, with an `ArgumentError`. The scalar-step refusal of a `SymmetricSparseIterate` is under Failure
behavior. A `SparseIterate` destination is written entry by entry, as for `GroupNormL2`.

## [Clauses owned elsewhere](@id spec-groupmcp-elsewhere)

**Links only.**

| Subject | Owning section |
|:--|:--|
| The scalar penalty and its proximal map | [MCP definition](@ref spec-mcp-definition) and [MCP ordinary-path clauses](@ref spec-mcp-ordinary) |
| Group norms, the screen and soft shrinkage at `a = Inf` | [GroupNormL2 range clauses](@ref spec-groupnorml2-range), [GroupNormL2 endpoint clauses](@ref spec-groupnorml2-endpoints), and [KKT endpoint clauses](@ref spec-kkt_violation-endpoints) |
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Compact block buffers | [Pure block proximal contract](@ref block-proximal-oracle) |
| Scaling a function | [Scaling a function](@ref scaling-guide) |

## [Evidence](@id spec-groupmcp-evidence)

| Clause ID | Status | Evidence |
|:--|:--|:--|
| `group-mcp.kkt.entry-classification` | `characterized` | [GroupMCP dispatch to the shared screen](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/src/functions/groupmcp.jl#L514-L533), [Working-type conversion in the shared screen](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/src/numerical_kernels.jl#L1134-L1145), and [Shared mixed-width checks](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/test/kkt_block.jl#L1233-L1241) |

[`test/groupmcp.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/groupmcp.jl) compares the proximal
map with a BigFloat minimization of the radial objective that does not use the package: the scalar reference of
`test/mcp_reference.jl` (candidate comparison at 256 bits, repeated at 4096 bits when two candidates are within its
uncertainty, ties included) at the group norm computed in 512-bit BigFloat. It checks the objective at the computed
output within a stated tolerance and, where the reference decision is stable under the rounding of the norm, the output
itself. The cases tested are listed in the file and in `test/TESTING.md`.

## [History](@id spec-groupmcp-history)

The [restoration issue](https://github.com/mlakolar/ProximalBase.jl/issues/1069) and its
[pull request](https://github.com/mlakolar/ProximalBase.jl/pull/1142) record the decisions: the endpoint table, the
refusal of the nonzero-group score, the rounded effective step under `Scaled`, the refusal of non-floating
destinations and the uncopied buffers of the typed constructor. The archived `legacy/full` implementation formed the
group norm in the input type and could return `NaN` for representable outputs
([#534](https://github.com/mlakolar/ProximalBase.jl/issues/534)). The infinite-weight convention follows
[#250](https://github.com/mlakolar/ProximalBase.jl/issues/250).
