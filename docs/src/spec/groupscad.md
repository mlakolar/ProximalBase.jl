# [GroupSCAD numerical contract](@id spec-groupscad)

This page is the authored contract of `GroupSCAD`.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-groupscad-identity)

[`GroupSCAD`](@ref GroupSCAD) is the group smoothly clipped absolute deviation penalty. See the
[implementation](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/groupscad.jl),
the [function guide](../group_penalty.md#group-scad-guide) and the
[API coverage matrix](../api/coverage.md#API-coverage-matrix).

## [Definition and domain](@id spec-groupscad-definition)

For disjoint nonempty groups ``G_k`` of linear indices,

```math
f(x)=\sum_{k=1}^{K}p_{\lambda_k,a}\bigl(\lVert x_{G_k}\rVert_2\bigr),\qquad
p_{\lambda,a}(t)=\begin{cases}\lambda t, & 0\le t\le\lambda,\\ \lambda t-(t-\lambda)^2/(2(a-1)), & \lambda<t<a\lambda,\\ (a+1)\lambda^2/2, & t\ge a\lambda,\end{cases}
```

where ``\lambda_k`` is the effective coefficient of group `k`: ``\lambda_0w_k`` when `λ0` and `w_k` are finite and
positive, and otherwise the [endpoint table](@ref spec-groupscad-endpoints) gives the group's map and term. The formula
for ``p`` is that of finite positive ``\lambda`` and finite `a`, as for the scalar [`SCAD`](@ref spec-scad); at `a = Inf` and
finite positive ``\lambda`` it is ``\lambda t``. The shape `a` is not rescaled by the weights. Entries in no group are
unpenalized.

**Zero weights.** A group with `w_k = 0` is unpenalized, whatever `λ0` (`Inf` included): the proximal map leaves its
entries unchanged (copied to the destination, see Storage below the endpoint table) and its term is zero. For the group's
proximal map and term, this rule comes before the others on this page.

**Parameter type.** `GroupSCAD(λ0, groups, w; a = 3.7)` uses `T = float(promote_type(typeof(λ0), eltype(w)))` and
converts `a` to `T`, so the default `a = 3.7` does not promote a `Float32` function to `Float64`.
`GroupSCAD(λ0, groups; a = 3.7)` uses `T = float(typeof(λ0))` and the weights ``w_k=\sqrt{\lvert G_k\rvert}``,
computed in `promote_type(T, Float64)` and converted to `T`. This differs from the scalar `SCAD`, whose one-argument
form promotes the level and the shape together. The level and the weights are nonnegative and not `NaN`; `a` is above
2 (`Inf` is admitted) and not `NaN`. `a` is checked before its conversion and again after it: a value above 2 whose
conversion to `T` is not above 2 (`a = 2.00000001` in `Float32`), and a finite value whose conversion to `T` is
infinite (`a = 1e39` in `Float32`), throw an `ArgumentError`. The two public constructors copy the weights, the list of
groups and each group that is not a range, and keep their order; a range group can be kept as given (when every group
is a range of one concrete type) and must not be mutated afterwards. With no groups the value is zero and the proximal
map is the identity.

## [Traits](@id spec-groupscad-traits)

`GroupSCAD` is proximable, and its convexity, smoothness and separability traits are false. Instances with a
positively weighted group of finite positive level and finite shape are nonconvex.

## [Supported arguments](@id spec-groupscad-arguments)

The value, `prox`, `prox!` and `value_and_prox!` take one-based real arrays: `Array` (a matrix is indexed linearly by
the groups), `SparseIterate` and `BlockIterate`, read and written by indexing their entries. The destination of `prox!`
and `value_and_prox!` has a concrete floating-point element type; `prox` allocates its destination with the element type
`float(eltype(x))`, which must pass the same check. The step is a nonnegative real scalar.
`prox_block!(out, f, u, k, γ)` and the allocating `prox_block(f, u, k, γ)` take compact vectors in the order of
`groups[k]` and do not change an iterate. `BlockIterate(p, f)`, `BlockIterate(T, p, f)` and `BlockIterate(x, f)` take
the groups of `f`, which must then partition the entries, with the copying rule of the constructors above.

The numerical target is homogeneous CPU `Float32` and `Float64` input, destination, parameters and step. Some other type
combinations dispatch, without a precision guarantee.

## [Preconditions](@id spec-groupscad-preconditions)

The typed constructor `GroupSCAD{T, G}(λ0, a, groups, w)` validates its arguments and stores `groups` and `w` without
copying them: do not mutate the groups or the weights passed to it afterwards.

Exact in-place use (`y === x`, and `out === u` for compact blocks) is supported. Other overlaps that
`Base.mightalias` detects are rejected; an overlap that it does not detect is unsupported. For `prox_block!`, `out`
must be independent of iterate storage, as the [pure block contract](@ref block-proximal-oracle) states. A `GroupSCAD`
has no buffer, and its oracles do not write to the object.

## [Results](@id spec-groupscad-results)

`prox!` returns the destination, `prox_block!` returns `out`, and `value_and_prox!` returns the value at the stored
output: it equals `f(y)` up to the order in which sums of squares are accumulated.

## [Ordinary-path clauses](@id spec-groupscad-ordinary)

**Exact-arithmetic target.** For positive finite `λ0`, `w_k` and `γ`, a finite `a` and a nonzero group with norm
``r=\lVert u_{G_k}\rVert_2``, the target replaces the group by ``(t_*/r)\,u_{G_k}``, where ``t_*`` is a global minimizer of
``p_{\lambda_k,a}(t)+(t-r)^2/(2\gamma)`` over ``t\ge0``, the one that the scalar `SCAD`
[proximal rule](@ref spec-scad-ordinary) selects at the input `r`: the group is unchanged when ``t_*=r``, `+0` when
``t_*=0``, and otherwise scaled by ``t_*/r``, which is below one. At a tie between minimizers the smaller magnitude is
taken, so a tie with zero gives `+0`. The middle piece has positive, zero or negative curvature as `γ` passes through
`a - 1`; the scalar contract owns the comparisons.

**Computed result.** When the input, the destination, the parameters and the step are `Float32` or `Float64`, the decisions
are the comparisons of the scalar kernel on the computed norm, the level as formed and the cutoffs, each rounded. A group
within that rounding of a cutoff can take either side: for `a = 3`, `γ = 5`, ``\lambda_k=1`` and `r = sqrt(20.0)` the
result is `+0`. The stored entries ``q\,u_i`` are rounded, so the direction of the group is not exact: for ``\lambda_k=1``,
`γ = 1` and `[1.5, nextfloat(0.0)]` the result is `[0.5, 0.0]`. ``t_*`` can round to `r` inside the middle piece, as for the
input `[1.2, 1.6]` with `λ_k = 1`, `γ = 1e-20` and `a = 3.7`: the group is then unchanged, and its value is
``p_{\lambda_k,a}`` at `r`, below the plateau.

The value of such a group is ``p_{\lambda_k,a}`` at its computed norm, up to rounding.

At `a = Inf` each group follows the rules of [`GroupNormL2`](groupnorml2.md) at the coefficient ``\lambda_k``: soft
group shrinkage and, for finite positive stored `λ0` and `w_k`, the value ``\lambda_k\lVert x_{G_k}\rVert_2``; the
[endpoint table](@ref spec-groupscad-endpoints) gives the other cases (a hard group at a zero input has the value zero).

## [Range clauses](@id spec-groupscad-range)

For positive finite stored `λ0` and `w_k`, the decisions above do not require the products ``\lambda_0w_k``,
``\gamma\lambda_k`` or ``a\lambda_k`` to be representable: an overflowing finite product is not treated as a stored
infinity, and an underflowing one is not treated as a stored zero. A scaled norm is the recovery route, when needed,
for a group whose sum of squares overflows or underflows the input type. For a finite shape and a finite positive step, that route writes
``r=m_r2^{e_r}`` and ``\lambda_k=m_\lambda2^{e_\lambda}`` (significands in ``[1/2,1)``): for ``r\le\lambda_k``, decided on these parts,
the group is soft thresholded; when ``e_r-e_\lambda`` is above `exponent(floatmax(R)) + 1` for the recovery type `R` the group is
unchanged; otherwise the scalar kernel decides the problem scaled by ``2^{-e_\lambda}``. For `a = 3.7`, `γ = 1.0`,
``\lambda_k=1/2`` and `r = floatmax(Float64)`, ``e_r-e_\lambda=1024``, which is not above that bound, although ``r/\lambda_k`` exceeds
`floatmax(Float64)`; the group is unchanged. The stored result is limited by the range
of the destination type. `Float64` examples, each with its own parameters:

```@example group_scad
using ProximalBase
# a = 3.7, λ = 2^1000; the level in units of r = 2^-30 (2^1030) overflows. γλ = 2^-60 < r ≤ λ: soft thresholding
g = GroupSCAD(2.0^1000, [1:1], [1.0])
@assert prox_block(g, [2.0^-30], 1, 2.0^-1060) == [2.0^-30 - 2.0^-60]
# λ0 = w = 1e-200: λ0*w rounds to 0, and r/λ ≈ 2.2e90. At a = 3.7 (nonconvex) the group is kept, since r/λ > √(γ(a+1))
u = [1e-310, 2e-310]
k = GroupSCAD(1e-200, [1:2], [1e-200])
@assert prox_block(k, u, 1, 1e100) == u
# at a = 1e200 (convex middle piece) r < γλ: zero
k2 = GroupSCAD(1e-200, [1:2], [1e-200]; a = 1e200)
@assert prox_block(k2, u, 1, 1e100) == [0.0, 0.0]
# the value on the middle piece with a near floatmax: p(0.9e308) ≈ 4.95e307
h = GroupSCAD(1.0, [1:1], [1.0]; a = 1e308)
@assert isapprox(h([0.9e308]), 4.95e307; rtol = 1e-14)
```

When the ordinary sum of the group terms is at or above `floatmax(T)/2` or, for `Float32` and `Float64`, small enough
that terms rounded into the subnormal range may have lost accuracy, the value is recomputed from scaled terms in at least `Float64`, unless the
endpoint table already makes the value `NaN` or `Inf`. Within a few units in the last place of the overflow threshold, a
value can be `floatmax(T)` or `Inf`.

## [Endpoint clauses](@id spec-groupscad-endpoints)

The rows apply per group in this order and are selected by the stored parameters (`iszero` and `isinf` of `λ0`, `w_k`, `a`
and `γ`), not by whether their product overflows or underflows. Other groups and the entries in no group are independent.
Steps are nonnegative and not `NaN`.

| Parameters | Proximal group | Group value |
|:--|:--|:--|
| `w_k = 0` (see Zero weights above) | Unchanged | `0` |
| `w_k = Inf`, or `λ0 = Inf` and `w_k > 0` | `+0` at every valid step, `γ = 0` and `γ = Inf` included | `0` at a zero group, `NaN` at a `NaN` entry, `Inf` otherwise |
| `λ0 = 0`, finite `w_k > 0` | Identity | `0`, also at nonfinite entries |
| Positive finite `λ0`, `w_k`; `γ = 0` | Identity, signed zeros kept (see Storage below) | As the rows below |
| Positive finite `λ0`, `w_k`; `γ = Inf` | `+0` | As the rows below |
| Positive finite `λ0`, `w_k`; finite `γ > 0`; a `NaN` entry | Every entry `NaN` | `NaN` |
| Same, a `±Inf` entry and no `NaN` | Unchanged | The plateau ``(a+1)\lambda_k^2/2`` for finite `a` (`Inf` at `a = Inf`) |
| Same, a zero group (`±0` entries) | `+0` | `0` |
| Same, finite nonzero group | The ordinary and range clauses above | ``p_{\lambda_k,a}`` at the norm |

At `a = Inf` the rows are those of `GroupNormL2` at the level `λ0` and the weight `w_k`, which the group map and the term of that function give.

**Storage.** "Unchanged", "Identity" and "signed zeros kept" mean that the entries, and the entries in no group, are copied to the
destination under the [storage rule](../iterates.md#Active-coordinates-are-deliberate-state) of the iterates and the conversion
to its element type. A dense destination of the same element type keeps a `-0.0`; a zero written to an absent coordinate of a
`SparseIterate` leaves it absent, and it reads `+0`. A route that sets a group to zero writes `+0`.

## [Failure behavior](@id spec-groupscad-failures)

The constructors throw an `ArgumentError` for a negative or `NaN` level or weight, a shape that is `NaN` or not above
2, a shape whose conversion to `T` is not above 2 or turns a finite value into `Inf`, an empty group, a nonpositive
index, an index in two groups and offset axes, and a `DimensionMismatch` for a weight count that differs from the group
count. The public constructors check the level, the weights and the shape before conversion and also throw an
`ArgumentError` for complex weights. The array
oracles and `prox_block!` throw before any write: an `ArgumentError` for a negative or `NaN` step (also when a group of
the function is fixed at `+0` by its parameters), entries that are not
real, a destination whose element type is not a concrete floating-point type (an integer or an abstract type such as
`AbstractFloat`), a detected partial overlap and offset axes; a
`DimensionMismatch` for different shapes, groups beyond the argument and a compact length that differs from the group;
a `BoundsError` for a group number out of range or a `Bool`. A `SymmetricSparseIterate` argument or output of a
proximal oracle with a scalar step throws an `ArgumentError` before any write.

At a real array point (`Array`, `SparseIterate` or `BlockIterate`), `kkt_violation(f, v, x, k)` returns at a zero group
the signed screen of `GroupNormL2` with the level and the weight as separate factors: ``\lVert v_{G_k}\rVert_2-\lambda_0w_k``
for finite `v`, `λ0` and `w_k`, up to rounding; ``\lVert v_{G_k}\rVert_2`` for a zero weight, whatever `λ0`; otherwise
the rules of that screen for `NaN` and infinite values. At an entry in no group it returns `abs(v[k])`. It throws an `ArgumentError` at a nonzero group, as do
`kkt_violation(f, v, x)` and `kkt_violation(f, v, i)`. Layout, count and extent errors are those of `GroupNormL2`. A
call with a `SymmetricSparseIterate` point that passes the shared argument checks throws a `MethodError`.

## [Limitations and non-guarantees](@id spec-groupscad-limitations)

An instance with a positively weighted group of finite positive level and finite shape is nonconvex (see Traits). The target of the proximal map is a global minimizer of the radial problem in exact arithmetic (see the
ordinary-path clauses for the computed result), and `GroupSCAD` carries no statement about the global minimization of an objective
that contains it. At a zero group the score is a rounded signed screen of the local condition ``\lVert v_{G_k}\rVert_2\le\lambda_k``
(for finite positive factors); a nonpositive score is not a proof of that condition and does not imply a global minimizer. The score
at a nonzero group and the full-point form are not implemented.

`Scaled(GroupSCAD(...), α)` multiplies the value by `α` and calls the proximal operator with the effective step
``\alpha\gamma`` as rounded, as for the scalar `SCAD`. The effective step is validated: a negative or `NaN` one (`Scaled(f, Inf)` at
`γ = 0` gives `NaN`) throws an `ArgumentError` before any write, and the ordered endpoint table applies to a valid effective step.
A zero `α` throws an `ArgumentError`. There is no compact block
method and no point-aware score under `Scaled` (`MethodError`).

```@example group_scad
g = GroupSCAD(2.0^-700, [1:2], [1.0])
@assert prox(Scaled(g, 2.0^600), [0.6, 0.8], 2.0^600) == [0.0, 0.0]     # αγ = 2^1200 rounds to Inf
@assert prox_block(g, [0.6, 0.8], 1, 2.0^1023) == [0.6, 0.8]            # a large finite step keeps the group
```

There is no `restrict` and no array (per-entry or per-group) step: calls that pass the shared argument checks throw a
`MethodError`, except where a
route shared with other functions rejects a `SymmetricSparseIterate` destination, or the allocating
`prox(f, S, steps)`, with an `ArgumentError`. The scalar-step refusal of a `SymmetricSparseIterate` is under Failure
behavior. A `SparseIterate` destination is written entry by entry, as for `GroupNormL2`.

For a finite shape, a `Float32` input whose norm exceeds `floatmax(Float32)`, with a `Float64` destination, is covered by the tests of
[#534](https://github.com/mlakolar/ProximalBase.jl/issues/534). Other mixed type combinations carry no precision guarantee; the
numerical targets are the homogeneous ones in Supported arguments.

## [Clauses owned elsewhere](@id spec-groupscad-elsewhere)

| Subject | Owning section |
|:--|:--|
| The scalar penalty and its proximal map | [SCAD numerical contract](@ref spec-scad) |
| Group norms, the screen and soft shrinkage at `a = Inf` | [GroupNormL2 numerical contract](groupnorml2.md) |
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Compact block buffers | [Pure block proximal contract](@ref block-proximal-oracle) |
| Scaling a function | [Scaling a function](@ref scaling-guide) |

## [Evidence](@id spec-groupscad-evidence)

[`test/groupscad.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/groupscad.jl) compares the proximal
map with a BigFloat minimization of the radial objective that does not use the package, at the group norm computed in 512-bit
BigFloat: for a finite shape the scalar reference of `test/scad_reference.jl` (candidate comparison at 512 bits, repeated at 4096 bits
when two candidates are within its uncertainty, ties included), and for `a = Inf` the closed-form soft threshold. It checks the objective at the computed
output within a stated tolerance and, where the reference decision is stable under the rounding of the norm, the output
itself. The cases tested are listed in the file and in `test/TESTING.md`.

## [History](@id spec-groupscad-history)

The [restoration issue](https://github.com/mlakolar/ProximalBase.jl/issues/1070) and its
[pull request](https://github.com/mlakolar/ProximalBase.jl/pull/1160) record the decisions: the endpoint table, the
parameter-type rule, the refusal of the nonzero-group score, the rounded effective step under `Scaled`, the refusal of
non-floating destinations and the uncopied buffers of the typed constructor, as for
[`GroupMCP`](@ref spec-groupmcp). The archived `legacy/full` implementation formed the
group norm in the input type and could return `NaN` for representable outputs
([#534](https://github.com/mlakolar/ProximalBase.jl/issues/534)); `GroupSCAD` forms the scaled norm in the recovery type
and has finite-shape tests for the reproducers. The infinite-weight convention follows
[#250](https://github.com/mlakolar/ProximalBase.jl/issues/250).
