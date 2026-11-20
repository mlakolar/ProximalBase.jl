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
`ArgumentError` before anything is written. See the [NormL2 endpoint clauses](@ref spec-norml2-endpoints).

## The NormLinf norm

`NormLinf(λ)` is ``g(x)=\lambda\max_i\lvert x_i\rvert`` with ``\lambda\ge0``; it is not separable.
Its proximal operator clips the entries at a level ``\tau\ge0``,
``\operatorname{clamp}(x_i,-\tau,\tau)``, where ``\sum_i(\lvert x_i\rvert-\tau)_+=\gamma\lambda``,
and it is zero when ``\gamma\lambda\ge\lVert x\rVert_1``. The level comes from the
projection onto the ℓ₁ ball of radius ``\gamma\lambda`` (Condat's algorithm, with Michelot's as a fallback),
and the output is the clip itself, not ``x`` minus that projection. The error of ``\tau`` is absolute: on the ordinary
finite path, at most ``2.25\,n\,\varepsilon\max_i\lvert x_i\rvert+2n\delta`` for ``n\ge1`` entries with
``n\varepsilon\le 1/2``, relative to the radius ``a`` that the proximal operator forms from ``\gamma`` and ``\lambda``
and the entries read in the output type, with ``\varepsilon`` the machine epsilon and ``\delta`` the smallest subnormal
number of the output type (the storage of a subnormal radius or output adds ``\delta``). The [numerical
contract](numerics.md) states the hypotheses and the scaled units of the exceptional path. The value takes arrays and
the three iterates, and a
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
have the same magnitude and belong to the maximal face, and entries already at that magnitude belong to it as well
(`prox(NormLinf(1.0), [3.0, 2.0, 1.0], 1.0)` is `[2.0, 2.0, 1.0]`): this is the case the equality rule is made for. For
an exact proximal point `y = prox(g, z, γ)` and `v = (y - z)/γ`, the vector ``-v`` belongs to the subdifferential at `y`.
With the computed proximal point and difference, the score is checked against a rounding tolerance, as in the example, which
scores such a point and then the same `v` at a point where one clipped entry is one unit in the last place below the other:

```@example group_penalty
fl = NormLinf(1.0)
# the zero point: inside the ℓ1 ball of radius 1 (value 0), and outside it
@assert kkt_violation(fl, [0.3, -0.5], [0.0, 0.0]) == 0.0
@assert kkt_violation(fl, [3.0, 1.0], [0.0, 0.0]) ≈ sqrt(5.0)               # τ = 2: residual (2, 1)
@assert kkt_violation(fl, [3.0, 1.0], [0.0, 0.0]) ≈ norm(prox(fl, [3.0, 1.0], 1.0))
# a face of one entry: |v₁ + λ| there (τ = -0.6), and v₂ off it
@assert kkt_violation(fl, [-0.4, 0.3], [2.0, 1.0]) ≈ sqrt(0.6^2 + 0.3^2)
# a point computed by prox: the clipped entries tie in magnitude, and the score is zero to rounding
@assert prox(fl, [3.0, 2.0, 1.0], 1.0) == [2.0, 2.0, 1.0]                  # an entry already at the magnitude belongs to the face
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
form is read as one vector, by indexing. A length `n` for which `n + 1` is not a finite number of the working type
(`Float16` from `n = 65519`) throws an `ArgumentError`. For `Array` arguments of one type, `Float32` or `Float64`, with a level of a
hardware float type, the tests measure no allocation and no write to `v`, `x` or the buffer of the `NormLinf` object; wider
parameter arithmetic such as a `BigFloat` level can allocate.
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

## [Sparse group lasso](@id sparse-group-l2-guide)

`SparseGroupNormL2(λ1, λ2, groups[, w])` combines entry sparsity with group
sparsity. For nonempty disjoint groups ``G_1,\ldots,G_K`` of positive linear
indices, set ``U=\bigcup_k G_k``. The penalty is

```math
f(x)=\lambda_1\sum_{i\in U}|x_i|+\lambda_2\sum_{k=1}^K w_k\lVert x_{G_k}\rVert_2.
```

Entries outside ``U`` are free in both terms and retain their values in the
proximal output. A zero weight removes the group ℓ₂ term while the ℓ₁ term
still acts on that group. The constructor keeps the order of the groups and
indices and copies the supplied groups and weights. The levels and weights
are converted to one promoted floating type and checked for nonnegativity
and NaN. The default is ``w_k=\sqrt{|G_k|}``. With no groups, the value is
zero and the proximal map is the identity.

For finite levels and weights and a positive finite ``\gamma``, let ``S_a(t)=\operatorname{sign}(t)(|t|-a)_+``.
First soft-threshold each entry in the group, then shrink the resulting vector:

```math
s_i=S_{\gamma\lambda_1}(x_i),\quad c_k=\gamma\lambda_2 w_k,\qquad
y_{G_k}=\begin{cases}
0, & \lVert s\rVert_2\le c_k,\\
(1-c_k/\lVert s\rVert_2)s, & \lVert s\rVert_2>c_k.
\end{cases}
```

Soft thresholding projects the difference ``x_{G_k}-s`` onto the
coordinate box ``[-\gamma\lambda_1,\gamma\lambda_1]^{|G_k|}``.
When the result is nonzero, the group shrink preserves the signs and
zero entries of ``s``, so this difference is still an admissible
ℓ₁ subgradient contribution at ``y_{G_k}``. The shrink difference
``s-y_{G_k}`` gives the group ℓ₂ contribution. When the group shrinks
to zero, ``s`` lies inside the group ℓ₂ subgradient ball of radius ``c_k``,
and the original difference still lies in the box.

```@example sparse_group_penalty
using ProximalBase
f = SparseGroupNormL2(1.0, 2.0, [[3, 1]], [1.0])
x = [-5.0, 99.0, 4.0]
y = similar(x)
v = value_and_prox!(y, f, x, 1.0)
@assert y ≈ [-2.4, 99.0, 1.8]
@assert v ≈ f(y)
y
```

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

### Compact groups

`prox_block!(out, f, u, k, γ)` uses compact vectors in `groups[k]` order,
returns `out` and accepts `out === u`. It checks the group, lengths,
one-based real buffers, step and detected nonidentical overlap before
writing. The caller supplies an output buffer independent of iterate
storage. `prox_block(f, u, k, γ)` allocates that buffer. To store the
result, the caller writes it into the matching block as a separate step.

```@example sparse_group_penalty
p = SparseGroupNormL2(1.0, 2.0, [[3, 1], [2]], [1.0, 0.0])
z = BlockIterate(3, p)
out = prox_block(p, [4.0, -5.0], 1, 1.0)
@assert out ≈ [1.8, -2.4]
@assert nactive(z) == 0
copyto!(blockview(z, 1), out)
@assert collect(activecoordinates(z)) == [1]
out
```

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

### Zero-group screening and active-group mismatch

Let `v` be the supplied smooth gradient, and set ``c_k=\lambda_2w_k``.
At a zero group, `kkt_violation(f, v, x, k)` returns the signed margin

```math
\bigl\lVert S_{\lambda_1}(v_{G_k})\bigr\rVert_2-c_k.
```

The norm is the distance from the negative gradient to the ℓ₁ subgradient
box. A positive margin means the gradient is too far from the box to be
offset by the group ℓ₂ subgradient ball.

At a nonzero group the ℓ₂ subgradient is fixed. Set
``r_i=v_i+c_k x_i/\lVert x_{G_k}\rVert_2``. The block score is
the Euclidean norm of the residual whose entry is
``r_i+\lambda_1\operatorname{sign}(x_i)`` at a nonzero entry and
``S_{\lambda_1}(r_i)`` at a zero entry. This is the distance from
``-v_{G_k}`` to the subdifferential at the current group.

On an ordinary array `k` is an entry index and selects its group.
An uncovered index has score `abs(v[k])`. On a `BlockIterate`,
`k` is a group number and the selected ordered layout must agree.
The full-point `kkt_violation(f, v, x)` checks the full block layout
and returns the maximum across zero, the block scores and the
uncovered-entry magnitudes. The result measures the worst block
or free-coordinate mismatch. The three-argument assumed-zero
form with an integer index throws `ArgumentError`.

## [The group ℓ∞ norm](@id group-linf-guide)

`GroupNormLinf` represents the group ℓ₁/ℓ∞ penalty

```math
g(x)=\lambda_0\sum_{k=1}^{K}w_k\max_{i\in G_k}\lvert x_i\rvert,
```

over disjoint groups ``G_k`` of linear indices of `x`. Its constructors follow those of
[`GroupNormL2`](@ref group-l2-guide): groups of any sizes and orders, entries in no group unpenalized, the constructor's
own copies of `groups` and `w`, a `DimensionMismatch` for a weight count that differs from the group count, and an
`ArgumentError` for a converted level or weight that is negative or `NaN`, an empty group, a nonpositive index and an index in two groups.
The weights default to ``w_k=1``.

| Mathematical form | Julia constructor | Meaning |
|:--|:--|:--|
| ``\lambda_0\sum_k\max_{i\in G_k}\lvert x_i\rvert`` | `GroupNormLinf(λ0, groups)` | Default weights 1 |
| ``\lambda_0\sum_k w_k\max_{i\in G_k}\lvert x_i\rvert`` | `GroupNormLinf(λ0, groups, w)` | One nonnegative weight per group |
| one group `1:length(x)` of weight one | `GroupNormLinf(λ0, [1:length(x)])` | The norm `NormLinf(λ0)` |
| ``G_k=\{k\}`` for every ``k`` | `GroupNormLinf(λ0, [[i] for i in 1:p], w)` | The weighted ℓ₁ norm `NormL1(λ0, w)` |

### Proximal operator: group clipping

For a step ``\gamma\ge0`` the proximal operator acts on each group separately. With the radius
``r_k=\gamma\lambda_0w_k`` it clips the group at a level ``\tau_k\ge0``,

```math
\operatorname{prox}_{\gamma g}(x)_i=\operatorname{clamp}(x_i,-\tau_k,\tau_k),\qquad
\sum_{i\in G_k}\left(\lvert x_i\rvert-\tau_k\right)_+=r_k,\qquad i\in G_k,
```

and, for finite entries, the group becomes zero (`+0`) when the computed radius reaches the computed sum of magnitudes (see
below; the table of exceptional parameters and values below gives the other cases). In exact arithmetic this is ``x_{G_k}`` minus its projection onto
the ℓ₁ ball of radius ``r_k``, formed by the clip itself. The level comes from the same threshold as
`NormLinf`, applied to the entries of one group, read through its indices without a copy. With finite data and positive
finite parameters of the same type `Float32` or `Float64`, and input and output of that type, the absolute error in
each stored entry is at most ``8n\varepsilon\max_{i\in G_k}\lvert x_i\rvert+(2n+1)\delta`` under the assumptions in the
[numerical contract](numerics.md#groupnormlinf-clip-accuracy). Here ``n`` is the group size, ``\varepsilon`` is `eps(T)` and
``\delta`` is the smallest positive subnormal of `T`. The radius is formed as
``(\gamma\lambda_0)w_k`` in floating point and then converted to the output
type. When ``\gamma\lambda_0`` is not a normal number, or the product leaves the normal range of the output type, it is
formed from the significands and exponents of the three factors, so that an intermediate overflow or underflow does not
decide it. The accuracy bound includes the case where the computed radius reaches the computed sum and the entire group becomes `+0`.

```@example group_penalty
gi = GroupNormLinf(1.0, [[1, 4, 2], [5, 3]], [1.0, 2.0])      # entry 6 is in no group
xi = [3.0, -1.0, 0.5, 2.0, 4.0, 7.0]
@assert gi(xi) == 1.0 * 3.0 + 2.0 * 4.0
pg = prox(gi, xi, 0.5)
@assert pg == [2.5, -1.0, 0.5, 2.0, 3.0, 7.0]       # radii 0.5 (τ = 2.5) and 1 (τ = 3); entry 6 is unchanged
@assert prox(gi, xi, 10.0)[[1, 4, 2, 5, 3]] == zeros(5)    # the radii 10 and 20 are at least the sums of magnitudes, 6 and 4.5
@assert prox(GroupNormLinf(1.0, [1:2]), [3.0, -1.0], 1.0) == prox(NormLinf(1.0), [3.0, -1.0], 1.0)
@assert prox(GroupNormLinf(0.5, [[1], [2]]), [3.0, -1.0], 1.0) == prox(NormL1(0.5), [3.0, -1.0], 1.0)
pg
```

`prox!(y, g, x, γ)` writes into `y` and accepts `y === x`; `value_and_prox!` also returns the value at the stored
output, which it forms from the clip levels (for an output whose element type is not floating point, the value is
recomputed from the stored output, after the conversions of its entries succeed). The step is a nonnegative scalar: a negative or `NaN` step throws an
`ArgumentError` before anything is written. A `BlockIterate` or a `SparseIterate` is read and written by indexing its
entries, as for `GroupNormL2`; a `SymmetricSparseIterate` has no `GroupNormLinf` proximal operator (an `ArgumentError`
before any write).

`prox_block!(out, g, u, k, γ)` computes the result for one group from a compact trial vector `u` in the order of
`groups[k]`, with the checks, the aliasing rules and the commit through `copyto!(blockview(x, k), out)` of
[the group ℓ₂ norm](@ref group-l2-guide). `BlockIterate(p, g)` and `BlockIterate(x, g)` copy the groups of `g` and
require them to partition the iterate.

```@example group_penalty
gp = GroupNormLinf(1.0, [[1, 4, 2], [5, 3], [6]], [1.0, 2.0, 1.0])
zb = BlockIterate(6, gp)
out = prox_block(gp, [3.0, 2.0, -1.0], 1, 0.5)           # the group [1, 4, 2] in its own order
@assert out == [2.5, 2.0, -1.0] && nactive(zb) == 0
copyto!(blockview(zb, 1), out)
@assert collect(activecoordinates(zb)) == [1]
out
```

A `GroupNormLinf` object keeps a buffer for the threshold, with the length of its longest group, which the first group
that needs one allocates. The buffer is left untouched by the parameter exits, by `NaN` and infinite groups, by a zero
fill decided before the threshold (the computed radius reaches the computed sum of magnitudes; a threshold whose level is zero has
already used the buffer) and, in `prox_block!`, when `out` is an `Array` with a floating-point element type, `u` is an
`Array` or a `SubArray` of an `Array`, and `!Base.mightalias(out, u)` (so not when `out === u`): `out` is then the
buffer. The same object must not be used from several tasks at once while its buffer may be in use (`Scaled` wrappers of
one object share it, and so do the groups of a threaded loop of `prox_block!(u, g, u, k, γ)`), and it must not be used
reentrantly, also within one task: a nested call on the same object, from a `getindex` of an argument while the threshold
runs, can overwrite the candidates of the running threshold if it reaches the threshold with the shared buffer. Construct a fresh `GroupNormLinf` for each task, or synchronize
the calls ([ownership](numerical_usage.md#Ownership-and-aliases)).

### Optimality scores

The dual norm of the ℓ∞ norm is the ℓ₁ norm. Let ``c_k=\lambda_0w_k``, let `v` be the vector that the caller supplies, usually
``\nabla h(x)`` (`v` is not recomputed), and let ``J_k`` be the face of group ``k``: the entries of ``x_{G_k}`` whose magnitude equals the
largest one after conversion to the working type of the call, with no tolerance. The subdifferential of the penalty of the group
at ``x_{G_k}`` is, for a finite ``c_k``, the ℓ₁ ball of radius ``c_k`` when ``x_{G_k}=0``. Otherwise it is the set of the vectors ``s`` with ``s_i=0``
off ``J_k``, ``s_i=\operatorname{sign}(x_i)\sigma_i`` with ``\sigma_i\ge0`` on ``J_k``, and ``\sum_{i\in J_k}\sigma_i=c_k``.
`kkt_violation(g, v, x, k)` and `kkt_violation(g, v, x)` return the following numbers:

| Group of `x` | `kkt_violation(g, v, x, k)` | Term of `kkt_violation(g, v, x)` |
|:--|:--|:--|
| zero | the margin ``\lVert v_{G_k}\rVert_1-c_k``, signed; a positive value means that zero violates the condition | the distance from ``v_{G_k}`` to the ℓ₁ ball of radius ``c_k``, which is ``0`` for finite entries when ``\lVert v_{G_k}\rVert_1\le c_k`` (the call tests its computed sum of magnitudes) |
| not zero | the Euclidean distance from ``-v_{G_k}`` to the subdifferential at ``x_{G_k}`` | the same distance |
| an entry in no group | ``\lvert v_i\rvert`` | ``\lvert v_i\rvert`` |

At a nonzero group, with ``u_i=-\operatorname{sign}(x_i)v_i`` for ``i\in J_k``, the distance is
``\bigl(\sum_{i\in G_k\setminus J_k}v_i^2+\sum_{i\in J_k}\min(u_i,\tau)^2\bigr)^{1/2}`` where ``\sum_{i\in J_k}(u_i-\tau)_+=c_k``
(``\tau`` can be negative), and ``\lVert v_{G_k}\rVert_2`` when ``c_k=0``. It is the full-point score of [`NormLinf`](@ref norm-linf-kkt) with the
level ``c_k``, applied to the entries of the group, which are read through their indices without a copy; for a group of one entry the distance is, in exact arithmetic,
``\lvert v_i+c_k\operatorname{sign}(x_i)\rvert``, the score of `NormL1`. Because the face is decided by equality, an exact tie can make the
value discontinuous in `x`: an entry one unit in the last place below the largest is not on the face.

The group is selected as for `GroupNormL2`: on an array point `k` is an entry, which selects its group (an entry in no
group has the score ``\lvert v_k\rvert``, also where `x[k]` is nonzero), and on a `BlockIterate` it is the group number, whose layout must equal that
of `g`. `kkt_violation(g, v, x)` is the maximum, kept at `NaN` when a term is `NaN`, of zero, of ``\lvert v_i\rvert`` over the entries in no
group and of the distance of **every** group, a zero group included: a maximum of block 2-norm residuals and not an
``\ell_\infty`` distance. With one group that covers `x`, this value and the value of `NormLinf(λ0 w)`, at ``x=0`` and at ``x\ne0``, are each within the accuracy bound below of the same exact distance, so they agree to within twice the bound (not bit for bit: the group is read through its indices). The checks are those of
`GroupNormL2` (real one-based arrays of equal length, the group extent and, on a `BlockIterate`, the whole layout); a function without
groups scores ``\lvert v_i\rvert`` of every entry. The three-argument form `kkt_violation(g, v, k)` throws an `ArgumentError` because the penalty is not
separable, and so does a `SymmetricSparseIterate`, which has no groups.

At a zero group the four-argument score is the margin and the term of the full-point value is the distance, and they are in general different numbers (equal where ``v_{G_k}`` has one nonzero entry or the margin is zero; a negative margin has the term ``0``).
With ``d=v_{G_k}-\operatorname{proj}(v_{G_k})``, the residual of the projection onto the ℓ₁ ball of radius ``c_k``, a positive margin is
``\lVert d\rVert_1`` and the distance is ``\lVert d\rVert_2``, so in exact arithmetic ``\text{distance}\le\text{margin}\le\sqrt{n}\,\text{distance}`` for a group of
``n`` entries, and both are positive for the same groups (in exact arithmetic); computed values can break these inequalities by rounding near the boundary. For the gradient `(5/8, 5/8)` with ``\lambda_0=w=1`` the margin is ``0.25`` and the
distance is ``\sqrt2/8\approx0.177``. The sentence that `kkt_violation(f, v, x)` is `max(0, max_k score_k)` therefore has this exception
for the zero groups of `GroupNormLinf`; at nonzero groups and at entries in no group the two forms use the same distance. Neither is the Euclidean
screen of `GroupNormL2` ([#634](https://github.com/mlakolar/ProximalBase.jl/issues/634)), which is negative for `(5/8, 5/8)`, where the group must
enter, and a tolerance chosen for one of these scores does not carry over to another.

```@example group_penalty
v = [0.3, -0.4, 1.0, 0.5, 0.0, 2.0]
zb = BlockIterate(6, gp)                       # every group is zero
@assert kkt_violation(gp, v, zb, 1) ≈ 0.2      # ‖(0.3, 0.5, -0.4)‖₁ - 1
@assert kkt_violation(gp, v, zb, 2) == -1.0    # ‖(0.0, 1.0)‖₁ - 2
@assert kkt_violation(gp, v, zb, 3) == 1.0     # ‖(2.0,)‖₁ - 1
@assert kkt_violation(GroupNormLinf(1.0, [1:2]), [5/8, 5/8], zeros(2), 1) == 0.25
@assert kkt_violation(GroupNormL2(1.0, [1:2], [1.0]), [5/8, 5/8], zeros(2), 1) < 0
# the distance of a zero group to the ℓ₁ ball is smaller than its margin: ‖d‖₂ = √3/15 and ‖d‖₁ = 0.2 for v_G = (0.3, 0.5, -0.4)
g3 = GroupNormLinf(1.0, [1:3])
@assert kkt_violation(g3, [0.3, 0.5, -0.4], zeros(3), 1) ≈ 0.2
@assert kkt_violation(g3, [0.3, 0.5, -0.4], zeros(3)) ≈ sqrt(3) / 15
copyto!(blockview(zb, 1), [3.0, 2.0, -1.0])    # group 1 is nonzero: x[1] = 3, x[4] = 2, x[2] = -1
@assert kkt_violation(gp, v, zb, 1) ≈ sqrt(2.1)      # the face is x[1]: the distance from (-0.3, -0.5, 0.4) to (1, 0, 0)
@assert kkt_violation(gp, v, zb, 2) == -1.0          # the zero groups keep their margins
@assert kkt_violation(gp, v, zb) ≈ sqrt(2.1)         # the distance of group 1, against 0 and 1 for the zero groups 2 and 3
kkt_violation(gp, v, zb)
```

A group is zero when its entries, converted to the working type of the call
(the floating type of `promote_type(eltype(v), eltype(x))`), are zeros of either sign, so a `NaN` or infinite entry of
`x` makes its group nonzero. The exceptional values follow the form. The four-argument margin of a zero group: a `NaN` in ``v_{G_k}`` gives `NaN`,
and an infinite entry (with no `NaN`) or a computed sum of magnitudes that overflows gives `+Inf`, also when ``c_k`` is infinite (that branch
comes first); an infinite ``c_k`` gives `-Inf` when the computed sum of magnitudes is finite, also where ``\lambda_0w_k`` overflows in the
working type; a zero weight gives ``\lVert v_{G_k}\rVert_1``. The full-point term of a zero group follows the zero-point rules of `NormLinf`
instead, in this order for ``v_{G_k}`` converted to the working type: a `NaN` gives `NaN`; otherwise an infinite entry gives `+Inf`; then
``c_k=\infty`` gives ``0`` and ``c_k=0`` (a zero weight, a zero level with a finite weight, or a product ``\lambda_0w_k`` that rounds to zero in the working type) gives ``\lVert v_{G_k}\rVert_2``. A finite sum of
magnitudes that overflows does not give `+Inf` there: the distance is formed in units of a power of two and is `+Inf` where it is above `floatmax`.
At a nonzero group, in both forms and in this order for the entries of ``v_{G_k}`` and ``x_{G_k}`` converted to the working type: a
`NaN` gives `NaN`; otherwise an infinite entry gives `+Inf`; then an infinite ``c_k`` gives `+Inf` and ``c_k=0`` gives
``\lVert v_{G_k}\rVert_2``. In the full-point form a `NaN` term takes precedence over an infinite one, whatever the order of the groups.
A group of ``n`` entries for which ``n+1`` is not a finite number of the working type
(`Float16` from ``n=65519``) throws an `ArgumentError`: in the four-argument form at a nonzero group (the test for a zero group comes first,
so a zero group of that size keeps its margin), and in the full-point form at every group.

For `Float32` and `Float64` the error of a distance is, to first order in ``nu``, at most ``(2n+\sqrt n+5)\,u\,(\lVert v_{G_k}\rVert_2+c_k)``
with ``u=`` `eps(W)/2` for the working type `W`, ``n`` the group size and ``c_k`` as rounded to `W`, assuming ``nu\le1/4``, which is not
checked (at most 512 entries per group in `Float16`); out of the ordinary range the value is formed in units of a power of two or
recomputed in a wider type, which extends the range and not the precision (see the [numerical contract](numerics.md)). In the tests, the calls on `Array`
arguments of these types allocate nothing after the first call, and write neither `v` nor `x`. The four-argument call reads one group: a zero group has the zero test of ``x_{G_k}`` and one sum of ``\lvert v_i\rvert``, with no classification pass, and at a nonzero group the zero test comes before the passes below.
The full-point call reads every group, and each group first has two classification passes, one over ``x_{G_k}`` and one over ``v_{G_k}``, before any early exit. After them, apart from the early exits of the exceptional values and of a zero group inside the ball, a nonzero group runs the pass of the face, the
level iteration where the closed-form level is not certified, and the pass of the squares, and a zero group outside the ball runs the level iteration and the
pass of the squares; the paths for values out of the ordinary range add passes. The speed of the call is not tuned.

Under [`Scaled`](@ref) the scores have the routes of `GroupNormL2` (the [scaling guide](@ref scaled-kkt) has the limits). One wrapper with a finite positive factor
`α` evaluates the score at `v/α` and multiplies it by `α`. A chain whose product of factors is not a normal number gives the score of the function with the level that it folds
into where that level is a nonzero finite number of the type or zero; where it does not fold, a zero group has the margin ``\lVert v_{G_k}\rVert_1-a\lambda_0w_k`` with
the coefficient formed from the significands and the exponents of the three factors, and a nonzero group, a zero entry inside a nonzero
group, a nonzero entry in no group and the full-point form throw an `ArgumentError`. An infinite factor throws an `ArgumentError` at a nonzero group, at a nonzero entry in no group and in the full-point form. A
`SymmetricSparseIterate` throws the `ArgumentError` of the function, and `Scaled(g, 0)` has no representation for this function.

```@example group_penalty
gs = Scaled(GroupNormLinf(1.0, [1:2, 3:4], [1.0, 2.0]), 2.0)           # the function GroupNormLinf(2.0, [1:2, 3:4], [1.0, 2.0])
vs = [0.5, -0.25, 1.0, 2.0]
@assert kkt_violation(gs, vs, zeros(4), 1) ≈ -1.25                       # ‖(0.5, -0.25)‖₁ - 2
@assert kkt_violation(gs, vs, [1.0, -1.0, 0.0, 0.0], 1) ≈
        kkt_violation(GroupNormLinf(2.0, [1:2, 3:4], [1.0, 2.0]), vs, [1.0, -1.0, 0.0, 0.0], 1)
@assert kkt_violation(gs, vs, [1.0, -1.0, 0.0, 0.0]) ≈ 1.9445436482630056   # group 2 has the distance 0, since 3 ≤ 4
kkt_violation(gs, vs, [1.0, -1.0, 0.0, 0.0])
```

### Exceptional parameters and values

The rules apply to each group in this order; ``\lambda_0``, ``w_k`` and ``\gamma`` are decided separately, before a radius is formed,
so that the proximal operator and the value do not form a product such as ``\lambda_0w_k`` with ``\lambda_0=\infty`` and ``w_k=0``.

| Case | Result for group ``k`` |
|:--|:--|
| ``w_k=0`` | Unpenalized: returned unchanged, `NaN` and `±Inf` included, also at an infinite level or step; value zero |
| ``w_k=\infty``, or ``\lambda_0=\infty`` with ``w_k>0`` | Fixed at `+0` at every nonnegative step, ``\gamma=0`` and ``\gamma=\infty`` included, for `NaN` and `±Inf` inputs too; ``w_k=\infty`` with ``\lambda_0=0`` is included. Value `0` at a zero group, `Inf` at a nonzero group without `NaN`, `NaN` at a `NaN` entry |
| ``\lambda_0=0`` or ``\gamma=0``, finite positive ``w_k`` | The identity, also when ``\lambda_0w_k`` overflows and at ``\gamma=\infty`` with a zero level. Value `0` at a zero level, ``\lambda_0w_k\max\lvert x_{G_k}\rvert`` at a zero step |
| ``\gamma=\infty``, positive finite level and weight | Fixed at `+0` |
| A `NaN` entry, positive finite level, weight and step | Every entry of the group becomes `NaN`; the other groups are unaffected |
| A `±Inf` entry and no `NaN`, positive finite level, weight and step | Returned unchanged, also when the finite product ``\gamma\lambda_0w_k`` overflows |
| Finite entries, the computed radius reaches the computed sum of magnitudes | `+0`, also when the product ``\gamma\lambda_0w_k`` overflows while the sum is finite |
| ``\gamma<0`` or ``\gamma=`` `NaN` | `ArgumentError` before any write |

The value of a group is ``\lambda_0w_k\max\lvert x_{G_k}\rvert``, formed in floating point as
``(\lambda_0w_k)m`` with two rounded products when ``\lambda_0w_k`` is a normal number and by the three-factor
kernel in power-of-two units otherwise, and then converted to the output type; the sum over the groups may be recomputed in power-of-two units, accumulated in at least `Float64`, where it may
have lost accuracy (a sum below `K*floatmin(T)` for `K` groups, whose terms may have rounded into the subnormal range) or
class (at or above `floatmax(T)/2`), as for `GroupNormL2` ([value range limits](@ref numerical-value-range)). A sum of
magnitudes of one group above `floatmax(T)/2` is processed in power-of-two units. With a positive finite level and
weight, `prox!` computes the clip of a group with finite entries whose sum overflows from the entries scaled by a power of two:

```@example group_penalty
M = floatmax(Float64)
huge = GroupNormLinf(M, [1:2], [2.0])        # the radius γλ₀w = M/2 at γ = 0.25 is finite; λ₀w alone overflows
yh = prox(huge, [M, M], 0.25)
@assert yh[1] == yh[2] && isapprox(yh[1], 0.75M; rtol = 1e-14)
tiny = GroupNormLinf(2.0^-100, [1:2], [2.0^1000])   # γλ₀ underflows, γλ₀w = 2^-100 equals ‖x‖₁
@assert prox(tiny, [2.0^-100, 0.0], 2.0^-1000) == [0.0, 0.0]
yh
```

`GroupNormLinf` has no `SymmetricSparseIterate`, `restrict`, per-group step or compact block method under `Scaled`.
`Scaled(GroupNormLinf(...), α)` multiplies the value by `α` and computes the array proximal operator with the step
`αγ`; it has no folded level for the proximal operator, so a step `αγ` that overflows reaches the function as `Inf` (the scores of
`kkt_violation` fold the level of a chain, as above). A group that reaches the
infinite-step exit then gives `+0`, where the exact radius ``\alpha\gamma\lambda_0w_k`` can be below the ℓ₁ norm of the
group; a group with a zero weight or a zero level with a finite weight keeps the identity, because those exits come first.
A zero `α` throws an `ArgumentError`.

```@example group_penalty
scaled = Scaled(GroupNormLinf(2.0^-1020, [1:2]), 2.0^1000)
@assert prox(scaled, [2000.0, 500.0], 2.0^30) == [0.0, 0.0]           # the step 2^1030 overflows
@assert prox(GroupNormLinf(2.0^-20, [1:2]), [2000.0, 500.0], 2.0^30) == [976.0, 500.0]   # the radius 2^10 = 1024 keeps the group
prox(scaled, [2000.0, 500.0], 2.0^30)
```
