# The group ℓ₂ norm and group soft thresholding

*Assumes: the ℓ₁ norm guide and grouped variables · You will learn: the group-lasso penalty, its proximal operator, its compact block form and its optimality screen.*

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

```@example group_penalty
using ProximalBase, LinearAlgebra
groups = [[1, 4, 2], [5, 3]]          # entry 6 is in no group
g = GroupNormL2(0.4, groups, [1.0, 3.0])
x = [1.0, -2.0, 0.5, 2.0, 0.0, 7.0]
@assert g(x) ≈ 0.4 * (norm(x[[1, 4, 2]]) + 3 * norm(x[[5, 3]]))
g(x)
```

## Proximal operator: group soft thresholding

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

## Compact block operations

`prox_block!(out, g, u, k, γ)` computes the result for **one group** from a
compact trial vector `u`, whose entries follow the order of `groups[k]`, and
writes it into `out`. It returns `out`, and `out === u` is supported. The group
index, the buffer lengths, the step and the overlap of `out` and `u` are checked
before any write. `prox_block(g, u, k, γ)` allocates the output.

The call does not read or write any iterate. With a `BlockIterate` that has the
groups of `g` (for example `BlockIterate(p, g)`), the caller builds the trial,
computes the result in its own buffer and then writes it, which is the step that
activates the group:

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

## Optimality screen

For a smooth part `h` with gradient `G = ∇h(x)`, a zero group ``k`` is optimal
for `h + g` when ``\lVert G_{G_k}\rVert_2\le\lambda_0w_k``. On a `BlockIterate`,
`kkt_violation(g, G, x, k)` returns ``\lVert G_{G_k}\rVert_2-\lambda_0w_k``: a positive
value means that zero violates the condition. Group `k` of the iterate must equal
`groups[k]`. On an array point, `kkt_violation(g, G, x, i)` takes an entry `i`:
it is the same group screen when the group of `i` is zero in `x`, and
``\lvert G_i\rvert`` when `i` is in no group or its group is nonzero.

```@example group_penalty
G = [0.3, -0.4, 1.0, 0.0, 0.0, 2.0]
@assert kkt_violation(f, G, z, 1) ≈ 0.5 - 0.4 * sqrt(3)
@assert kkt_violation(f, G, z, 3) ≈ 2.0 - 0.4
kkt_violation(f, G, z, 3)
```

## Exceptional parameters and values

| Case | Result for group ``k`` |
|:--|:--|
| ``w_k=0`` | Unpenalized: returned unchanged, `NaN` and `±Inf` included; value zero |
| ``w_k=\infty``, or ``\lambda_0=\infty`` with ``w_k>0`` | Fixed at `+0` at every nonnegative step, ``\gamma=0`` included, also for `NaN` and `±Inf` inputs ([#665](https://github.com/mlakolar/ProximalBase.jl/issues/665)) |
| ``\lambda_0=0`` or ``\gamma=0``, finite parameters | The identity |
| ``\gamma=\infty``, positive finite level and weight | Fixed at `+0` |
| A `NaN` entry, positive finite parameters | Every entry of the group becomes `NaN` |
| A `±Inf` entry and no `NaN`, positive finite parameters | Returned unchanged: the norm is infinite and the threshold ``\gamma\lambda_0w_k`` is finite, also when its floating-point product overflows |
| ``\gamma<0`` or ``\gamma=`` `NaN` | `ArgumentError` before any write |

At ``\gamma=0`` an infinite level is the limit ``\gamma\to0^+`` with the function
fixed, the projection onto the closure of its domain, as for `NormL1(Inf)`.
For finite inputs, norms whose sums of squares overflow or underflow and
thresholds outside the floating-point range are recomputed in power-of-two
units, so that a representable proximal point is returned accurately. Values
recover terms that round below `floatmin` as `NormL1` values do
([numerical contract](numerics.md)).

`GroupNormL2` does not apply to a `SymmetricSparseIterate`, and it has no
`restrict`, no per-group step and no compact block method under `Scaled`.
`Scaled(GroupNormL2(...), α)` supports values, array proximal operators and
`kkt_violation`. The [group-lasso example](examples/group_lasso_prox.md) checks
these operations against their formulas.
