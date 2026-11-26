# Constraints (prototype: `IndBallL1` entry)

*In the implementation this entry is part of the new `constraints.md`, which opens
with a short explanation of indicator functions and projections and the figure
below, followed by `IndBox`, `IndBallLinf`, `IndNonnegative`, `IndBallL2` and `IndBallL0`.*

A constraint ``x\in C`` is written as its *indicator function*: zero on ``C`` and
``+\infty`` outside. Its proximal operator is the Euclidean projection onto ``C``, the
nearest point of ``C``, whatever the step.

![Projections of one point onto three unit balls.](../../figures/projections.svg)

## [ℓ₁ ball: `IndBallL1`](@id entry-indballl1)

The constraint ``\|x\|_1\le r``. Projecting onto it sets small entries exactly to zero.

```math
\iota_{\{\|x\|_1\le r\}}(x)=\begin{cases}0,&\sum_i|x_i|\le r,\\ +\infty,&\text{otherwise,}\end{cases}
\qquad r\ge0.
```

For a matrix the sum runs over all entries.

| Constructor | Meaning |
|:--|:--|
| `IndBallL1(r = 1)` | ball of radius `r` |

`r` must be nonnegative and not `NaN`; `0` and `Inf` are allowed.

```@example entry-l1ball
using ProximalBase

C = IndBallL1(1.0)
x = [2.0, -1.0, 0.5]
C(x)                    # outside the ball
```

```@example entry-l1ball
z = prox(C, x)          # the projection; a step may be given but does not matter
z, sum(abs, z)
```

### Projection

A point inside the ball is returned unchanged. A point outside is soft
thresholded, ``z_i=\operatorname{sign}(x_i)\max(|x_i|-\tau,0)``, at the level
``\tau>0`` for which ``\sum_i\max(|x_i|-\tau,0)=r``.

| Operation | Call | Supported |
|:--|:--|:--|
| value | `C(x)` | ✓ `0.0` or `Inf` |
| projection | `prox(C, x)`, `prox!(out, C, x)`, with an optional step ``\gamma\ge0`` | ✓ |
| value at the new point | `value_and_prox!(out, C, x)` | ✓ `0.0` at the projection |
| gradient, `lipschitz` | | — |
| one coordinate or group | `prox_block` | — the ball couples all entries |
| optimality score | `kkt_violation` | — |

### Inputs and storage

✓ vectors, matrices and strided views ·
✓ `SparseIterate` (visits stored entries) ·
✓ `BlockIterate` (uses the whole vector) ·
◐ `Symmetric` matrices: the projection is a dense `Matrix` ·
— `SymmetricSparseIterate`; use `Matrix(x)`.
Element types: `Float64` or `Float32`, the same type for data, parameters, step and
output ([precision rules](@ref numerical-usage)).

### Good to know

- Membership is decided with the computed sum of magnitudes, with no tolerance,
  so rounding can decide a point on the boundary.
- `α * C` for any `α ≥ 0` is `C` itself, because scaling an indicator does not change it.
- When the projection cannot use the output array as workspace, it uses a buffer
  stored in `C`. Use a separate `IndBallL1(r)` in each task that may run
  concurrently ([details](@ref IndBallL1)).
- An infinite entry or a `NaN` entry follows the
  [endpoint rules](@ref spec-indballl1-endpoints).

**See also:** [`NormL1`](@ref) (the penalty form), [`IndBallL2`](@ref),
[`IndBox`](@ref) and [`IndBallLinf`](@ref) (other balls),
[`IndBallL0`](@ref) (at most ``s`` nonzero entries) ·
[guarantees](@ref spec-indballl1) · [API](@ref IndBallL1)
