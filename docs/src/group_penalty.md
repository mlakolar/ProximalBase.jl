# [The ℓ₂ and group ℓ₂ norms](@id l2-penalty-guide)

*Assumes: the ℓ₁ norm guide and grouped variables · You will learn: the Euclidean norm penalty, the group-lasso penalty, their proximal operators, the compact block form and the group optimality screen.*

## [The ℓ₂ norm](@id norm-l2-guide)

`NormL2(λ)` is ``g(x)=\lambda\lVert x\rVert_2`` (the Frobenius norm for a matrix)
with ``\lambda\ge0``. Its proximal operator shrinks the whole argument,
``\max(1-\gamma\lambda/\lVert x\rVert_2,0)\,x``. On a nonempty argument it is the
group ℓ₂ norm below with one group `1:length(x)` of weight one, and follows the same
rules, the exceptional cases included.

```@example group_penalty
using ProximalBase, LinearAlgebra
h = NormL2(0.5)
@assert h([3.0, 4.0]) == 2.5
@assert prox(h, [3.0, 4.0], 2.0) ≈ (1 - 1 / 5) * [3.0, 4.0]
@assert prox(h, [3.0, 4.0], 10.0) == [0.0, 0.0]          # ‖x‖ = 5 ≤ γλ
@assert prox(h, [3.0, 4.0], 1.0) == prox(GroupNormL2(0.5, [1:2], [1.0]), [3.0, 4.0], 1.0)
prox(h, [3.0, 4.0], 2.0)
```

The value takes arrays and the three iterates. On a `SymmetricSparseIterate` it is
the Frobenius norm of the symmetric matrix: each stored off-diagonal coordinate
counts twice in the squared norm. On a `SparseIterate` or a `SymmetricSparseIterate`
it reads only the stored entries, also when a sum of squares that overflows or
underflows is recomputed ([#808](https://github.com/mlakolar/ProximalBase.jl/issues/808)).
The proximal operators take arrays of the same size; a `BlockIterate` or a
`SparseIterate` is read and written by indexing its entries, O(length) in general, as
an array. A
`SymmetricSparseIterate` has no `NormL2` proximal operator: those calls throw an
`ArgumentError` before anything is written. A zero level gives value zero.
An infinite level gives zero at the zero argument (including an empty array),
`NaN` at a `NaN` entry, and `Inf` otherwise. With unit group weight, the
[endpoint table](@ref group-l2-endpoints) also describes the norm's prox;
in particular zero level takes precedence over an infinite step.

## The NormLinf norm

`NormLinf(λ)` is ``g(x)=\lambda\max_i\lvert x_i\rvert`` with ``\lambda\ge0``; it is not separable.
Its proximal operator clips the entries at a level ``\tau\ge0``,
``\operatorname{clamp}(x_i,-\tau,\tau)``, where ``\sum_i(\lvert x_i\rvert-\tau)_+=\gamma\lambda``,
and it is zero when ``\gamma\lambda\ge\lVert x\rVert_1``. The level comes from the
projection onto the ℓ₁ ball of radius ``\gamma\lambda`` (Condat's algorithm, with Michelot's as a fallback),
and the output is the clip itself, not ``x`` minus that projection. The error of ``\tau`` is absolute, of order
``n\,\varepsilon\max_i\lvert x_i\rvert+\delta`` with ``\delta`` the smallest subnormal number of the output type
(the storage of a subnormal radius or output adds ``\delta``). The value takes arrays and the three iterates, and a
`SymmetricSparseIterate` (its largest stored magnitude); the proximal operators take arrays and
the sparse and block iterates, and a `SymmetricSparseIterate` argument or output throws an `ArgumentError` before
anything is written. The endpoint rules (a zero or infinite level or step, `NaN` and infinite entries, overflow of
``\gamma\lambda``) are in the [numerical contract](numerics.md). A `NormLinf` object may keep a buffer of its own, so
that it must not be used from several tasks at once: construct a fresh `NormLinf(f.λ)` for each task, or synchronize
the calls ([ownership](numerical_usage.md#Ownership-and-aliases)); the full-point `kkt_violation` below does not
use that buffer.

```@example group_penalty
hinf = NormLinf(1.0)
@assert hinf([3.0, -1.0]) == 3.0
@assert prox(hinf, [3.0, -1.0], 1.0) == [2.0, -1.0]          # τ = 2: Σ(|xᵢ| - τ)₊ = 1
@assert prox(hinf, [2.0, 1.0], 2.0) == [0.5, 0.5]            # τ = 0.5
@assert prox(hinf, [3.0, -1.0], 4.0) == [0.0, 0.0]           # ‖x‖₁ = 4 ≤ γλ
@assert prox(NormLinf(1.0), [5.0], 2.0) == prox(NormL1(1.0), [5.0], 2.0)   # one entry: soft thresholding
prox(hinf, [3.0, -1.0], 1.0)
```

### [Optimality score](@id norm-linf-kkt)

For a smooth part `h` with gradient `v = ∇h(x)`, the point `x` is stationary for `h + g`, with
`g = NormLinf(λ)`, when ``-v\in\partial g(x)``. `kkt_violation(g, v, x)` is the Euclidean distance
from ``-v`` to ``\partial g(x)``. It is nonnegative, and it is zero when ``-v`` is a subgradient (in
floating point, zero to rounding). With ``J=\{i:\lvert x_i\rvert=\max_j\lvert x_j\rvert\}``, the subdifferential is
the set of ``s`` with ``\lVert s\rVert_1\le\lambda`` at ``x=0``. At ``x\ne0`` it is the set of ``s`` that vanish off ``J``,
satisfy ``s_i\operatorname{sign}(x_i)\ge0`` on ``J`` and have ``\sum_{i\in J}\lvert s_i\rvert=\lambda``.
With ``u_i=-\operatorname{sign}(x_i)\,v_i``, the value is the following.

| Point | Value | Level ``\tau`` |
|:--|:--|:--|
| ``x=0`` | ``0`` when ``\sum_i\lvert v_i\rvert\le\lambda``; otherwise ``\bigl(\sum_i\min(\lvert v_i\rvert,\tau)^2\bigr)^{1/2}``, the norm of `prox(g, v, 1)` | ``\tau\ge0`` with ``\sum_i(\lvert v_i\rvert-\tau)_+=\lambda`` |
| ``x\ne0`` | ``\bigl(\sum_{i\notin J}v_i^2+\sum_{i\in J}\min(u_i,\tau)^2\bigr)^{1/2}`` | ``\tau`` with ``\sum_{i\in J}(u_i-\tau)_+=\lambda``; it can be negative |

The face ``J`` is the set of entries whose magnitude equals the largest one after conversion to the
working type of the call, compared for equality, with no tolerance. An entry one unit in the last place below
the largest is not on the face, so for a positive finite level an exact tie can make the value discontinuous in `x`
(for finite entries of `v` and `x`, the value is ``\lVert v\rVert_2`` when ``\lambda=0`` and `Inf` at a nonzero `x` when the
level is infinite, ties or not). Where the
clip of `prox!` is active (a positive finite level and step with ``\gamma\lambda<\lVert x\rVert_1``), it writes the
same level ``\tau``, with the sign of the entry, into each clipped entry. The clipped entries of such a point
are therefore equal and form the face, which is the case the equality rule is made for: with
`v = (y - z)/γ` and `y = prox(g, z, γ)`, the vector ``-v`` is a subgradient at `y`. The example scores such a
point, with a tolerance, and then the same `v` at a point where one clipped entry is one unit in the last
place below the other:

```@example group_penalty
fl = NormLinf(1.0)
# the zero point: inside the ℓ1 ball of radius 1 (value 0), and outside it
@assert kkt_violation(fl, [0.3, -0.5], [0.0, 0.0]) == 0.0
@assert kkt_violation(fl, [3.0, 1.0], [0.0, 0.0]) ≈ sqrt(5.0)               # τ = 2: residual (2, 1)
@assert kkt_violation(fl, [3.0, 1.0], [0.0, 0.0]) ≈ norm(prox(fl, [3.0, 1.0], 1.0))
# a face of one entry: |v₁ + λ| there (τ = -0.6), and v₂ off it
@assert kkt_violation(fl, [-0.4, 0.3], [2.0, 1.0]) ≈ sqrt(0.6^2 + 0.3^2)
# a point computed by prox: the clipped entries tie, and the score is zero to rounding
z = [3.0, -2.5, 0.5]
y = prox(fl, z, 1.0)
v = y - z
@assert count(==(maximum(abs, y)), abs.(y)) == 2
@assert kkt_violation(fl, v, y) <= 1e-14
# the same v with one clipped entry one unit in the last place below the other: the face is entry 1
y1 = copy(y); y1[2] = -prevfloat(abs(y[2]))
@assert kkt_violation(fl, v, y1) ≈ sqrt(0.125) rtol = 1e-6                  # τ = u₁ - 1 = -0.25, and v₂ = 0.25 off the face
(stationary = kkt_violation(fl, v, y), one_ulp_below = kkt_violation(fl, v, y1))
```

For finite entries, a zero level gives ``\lVert v\rVert_2``, and an infinite level gives `0` at the zero point and
`Inf` at another point. A `NaN` entry of `v` or `x` gives `NaN`, also beside an infinite entry;
otherwise an infinite entry gives `Inf`, also at a zero or an infinite level. The working type is
`float(promote_type(eltype(v), eltype(x)))`. The rules for the levels, the non-finite entries, the
ranges and the accuracy are in the [numerical contract](numerics.md).

The norm is not separable, so it has the full-point form and no form for an entry or a group. The calls
`kkt_violation(g, v, i)` and `kkt_violation(g, v, x, k)` at an array, a `SparseIterate`, a `BlockIterate` or a
`SymmetricSparseIterate`, throw an `ArgumentError` that names the full-point form. The full-point form at a
`SymmetricSparseIterate` throws an `ArgumentError`. A `SparseIterate` or a `BlockIterate` in the full-point
form is read as one vector, by indexing. For `Array` arguments of one type, `Float32` or `Float64`, the tests
measure no allocation and no write to `v`, `x` or the buffer of the `NormLinf` object.
`Scaled(NormLinf(λ), α)` with a finite positive `α` scores `v/α` and multiplies the result by `α`; a chain of
wrappers whose factors fold into a normal level is scored with that level (the
[scaling guide](@ref scaled-kkt) has the limits and the refusals). `test/kkt_linf.jl` checks the value against
BigFloat references written from the definition.

## [The group ℓ₂ norm](@id group-l2-guide)

`GroupNormL2` represents the group-lasso penalty

```math
g(x)=\lambda_0\sum_{k=1}^{K}w_k\lVert x_{G_k}\rVert_2,
```

over disjoint groups ``G_k`` of linear indices of `x`. The groups may have any
sizes and orders, such as `1:5` or `[7, 2, 9]`; indices in no group are not
penalized. The level ``\lambda_0`` and the weights ``w_k`` are nonnegative, and
the weights default to ``w_k=\sqrt{\lvert G_k\rvert}``.

| Mathematical form | Julia constructor | Meaning |
|:--|:--|:--|
| ``\lambda_0\sum_k\sqrt{\lvert G_k\rvert}\,\lVert x_{G_k}\rVert_2`` | `GroupNormL2(λ0, groups)` | Default weights |
| ``\lambda_0\sum_k w_k\lVert x_{G_k}\rVert_2`` | `GroupNormL2(λ0, groups, w)` | One nonnegative weight per group |
| ``G_k=\{k\}`` for every ``k`` | `GroupNormL2(λ0, [[i] for i in 1:p], w)` | The weighted ℓ₁ norm `NormL1(λ0, w)` |

The constructor keeps its own copies of `groups` and `w`. A negative or `NaN`
level or weight, an empty group, a nonpositive index and an index in two groups
(or twice in one) throw an `ArgumentError`; a weight count that differs from the
group count throws a `DimensionMismatch`.
Groups retain their index order. Concrete range collections keep ranges;
otherwise groups are copied into integer vectors. The level and weights are
converted to their promoted floating-point type. These copies protect the
function from later edits to constructor inputs; they do not make arbitrary
changes to the function's fields valid. At an oracle call, every group index
must fit the argument, and arrays must be real and one-based.

```@example group_penalty
groups = [[1, 4, 2], [5, 3]]          # entry 6 is in no group
g = GroupNormL2(0.4, groups, [1.0, 3.0])
x = [1.0, -2.0, 0.5, 2.0, 0.0, 7.0]
@assert g(x) ≈ 0.4 * (norm(x[[1, 4, 2]]) + 3 * norm(x[[5, 3]]))
g(x)
```

### Proximal operator: group soft thresholding

For a step ``\gamma\ge0``, the proximal operator acts on each group separately:

```math
\operatorname{prox}_{\gamma g}(x)_{G_k}
=\max\left(1-\frac{\gamma\lambda_0w_k}{\lVert x_{G_k}\rVert_2},0\right)x_{G_k},
```

and leaves the entries in no group unchanged. A group whose norm is at most the
threshold ``\gamma\lambda_0w_k`` becomes zero; any other group keeps its
direction and loses ``\gamma\lambda_0w_k`` of its norm.

```@example group_penalty
γ = 0.5
p = prox(g, x, γ)
@assert p[[1, 4, 2]] ≈ (1 - γ * 0.4 / 3) * x[[1, 4, 2]]   # ‖x[G₁]‖ = 3
@assert p[[5, 3]] == [0, 0]                                 # ‖x[G₂]‖ = 0.5 ≤ γ⋅0.4⋅3
@assert p[6] == x[6]
p
```

`prox!(y, g, x, γ)` writes into `y` and accepts `y === x`. `value_and_prox!` also
returns the value at the stored output. The step is a nonnegative scalar: a
negative or `NaN` step throws an `ArgumentError` before anything is written.

### Compact block operations

`prox_block!(out, g, u, k, γ)` computes the result for **one group** from a
compact trial vector `u`, whose entries follow the order of `groups[k]`, and
writes it into `out`. It returns `out`, and `out === u` is supported. The group
index, the buffer lengths, the step and the overlap of `out` and `u` are checked
before any write. The buffers must be real, one-based vectors of the group's
length. Any nonidentical overlap reported by `Base.mightalias(out, u)` throws
`ArgumentError`. `prox_block(g, u, k, γ)` allocates the output.

The call does not write to or activate any iterate: `u` may be a view that reads
one, and `out` must be independent of iterate storage, the caller precondition of
the block contract. With a `BlockIterate` that has the groups of `g` (for example
`BlockIterate(p, g)`), the caller builds the trial,
computes the result in its own buffer and then writes it, which is the step that
activates the group. `BlockIterate(p, g)` and `BlockIterate(x, g)` require the groups
to partition every entry of the iterate, including entries unpenalized by a zero
weight:

```@example group_penalty
f = GroupNormL2(0.4, [[1, 4, 2], [5, 3], [6]])
z = BlockIterate(6, f)
u = [3.0, 4.0, 12.0]                        # ‖u‖ = 13
out = prox_block(f, u, 1, 1.0)
@assert out ≈ (1 - 0.4 * sqrt(3) / 13) * u
@assert nactive(z) == 0                     # the pure oracle changed nothing
copyto!(blockview(z, 1), out)
@assert collect(activecoordinates(z)) == [1]
out
```

The [block storage guide](block_iterate.md) describes the activation rule of
these writes, and the [pure block-prox contract](developer/contracts/block-prox-0.6.md)
the general extension hook.

### Optimality scores

For a smooth part `h` with gradient `G = ∇h(x)`, group ``k`` is optimal for `h + g` at
`x` when ``0\in G_{G_k}+\partial g_k(x_{G_k})``. With ``c_k=\lambda_0w_k``,
`kkt_violation(g, G, x, k)` returns the score of the group:

| Group of `x` | Score | Sign |
|:--|:--|:--|
| zero | ``\lVert G_{G_k}\rVert_2-c_k``, the screen of the group | signed; a positive value means that zero violates the condition |
| not zero | ``\lVert G_{G_k}+c_k\,x_{G_k}/\lVert x_{G_k}\rVert_2\rVert_2`` | nonnegative; a stationarity mismatch, not an instruction to set the group to zero |

A group is zero when its entries, converted to the working type of the call, are zeros
of either sign. On an array point `k` is an entry and selects its group; the score is
the same for each entry of the group. An entry in no group has the score
``\lvert G_k\rvert``, also when `x[k]` is nonzero. On a `BlockIterate`, `k` is the
group number and group `k` of the iterate must equal `groups[k]`. The three-argument
form throws an `ArgumentError`, because the penalty is not separable, and a
`SymmetricSparseIterate` has no `GroupNormL2` score: the call throws an `ArgumentError`.

`kkt_violation(g, G, x)` is the full-point measure: the maximum of `0`, of the positive
scores of the groups and of ``\lvert G_i\rvert`` over the entries in no group (including the
entries past the largest index of a group). The positive parts of the group scores are Euclidean
subgradient distances; the full-point measure takes their maximum together with the uncovered-entry
magnitudes. It is not an ``\ell_\infty``
distance, and a tolerance chosen for an ``\ell_\infty`` residual does not carry
over. It is zero of the working type for an empty `x`. It validates one-based axes, real
entries, `length(G) == length(x)` and the group extent once, and on a `BlockIterate`
the whole layout (its group count and each group), which is stricter than the local
form, which compares group `k`. A `NaN` in an entry that a group reads gives `NaN`,
and takes precedence over an infinite entry in another group.

In a score, a zero weight gives the coefficient `0` (also with an infinite level), and an
infinite weight, or an infinite level with a positive weight, gives `-Inf` at a zero
group and `+Inf` at a nonzero one (for finite entries). The working type `W` is
`float(promote_type(eltype(G), eltype(x)))`; the rules for non-finite entries and the
accuracy are in the [numerical contract](numerics.md).

```@example group_penalty
G = [0.3, -0.4, 1.0, 0.0, 0.0, 2.0]
z0 = BlockIterate(6, f)                     # every group is zero
@assert kkt_violation(f, G, z0, 1) ≈ 0.5 - 0.4 * sqrt(3)
@assert kkt_violation(f, G, z0, 3) ≈ 2.0 - 0.4
z1 = BlockIterate(6, f)
copyto!(blockview(z1, 1), [1.0, 1.0, 1.0])  # group 1 is nonzero
@assert kkt_violation(f, G, z1, 3) ≈ 2.0 - 0.4
# group 1 holds entries 1, 4, 2: G + 0.4 √3 x/‖x‖ = (0.7, 0.4, 0)
@assert kkt_violation(f, G, z1, 1) ≈ sqrt(0.7^2 + 0.4^2)
@assert kkt_violation(f, G, z1) ≈ 2.0 - 0.4     # the largest of the group scores
kkt_violation(f, G, z1, 1)
```

The group example of the [`kkt_violation` contract](@ref kkt-screens): with
`g = GroupNormL2(1.0, [[1, 2], [3, 4]], [1.0, 1.0])` and `x = [3, 4, 0, 0]`, the vector
`v = [-0.6, -0.8, 0.3, 0.4]` is stationary on the first group and has slack on the second.
The nonzero group is scored (see the [migration notes](migration.md) for the value that earlier
versions returned there), and its score is zero up to rounding, shown here with a tolerance:

```@example group_penalty
g0 = GroupNormL2(1.0, [[1, 2], [3, 4]], [1.0, 1.0])
x0 = [3.0, 4.0, 0.0, 0.0]
v0 = [-0.6, -0.8, 0.3, 0.4]
scores = [kkt_violation(g0, v0, x0, k) for k in 1:4]
@assert scores[1] == scores[2] && abs(scores[1]) <= 1e-15      # a mismatch at rounding level
@assert scores[3] ≈ -0.5 && scores[4] ≈ -0.5                   # the signed margin of the zero group
@assert 0 <= kkt_violation(g0, v0, x0) <= 1e-15                # the negative margin is dropped
scores
```

### [Exceptional parameters and values](@id group-l2-endpoints)

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
applies. A finite zero group becomes `+0` under positive finite parameters. An
infinite-weight or infinite-level group's value is zero at a zero group, `NaN`
when it contains `NaN`, and `Inf` otherwise; a zero weight overrides these value
rules and contributes zero.
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
[group-lasso example](examples/group_lasso_prox.md) checks the value, the
proximal operators, the block operations and the screen against their formulas.
