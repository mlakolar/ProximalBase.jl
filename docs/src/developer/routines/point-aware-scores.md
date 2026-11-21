# [Point-aware KKT scores](@id routine-point-aware-scores)

A KKT score measures stationarity using a smooth gradient `v` and
the subdifferential of a penalty or constraint `f`. The public forms
differ in which point information they use. The
[KKT forms and signs](../../operations.md#Checking-optimality-with-KKT-conditions)
and [KKT endpoint and type rules](../../spec/kkt_violation.md#spec-kkt_violation-endpoints)
own the contract for each supported route.

## Selecting the coordinate or group

The three-argument coordinate call `kkt_violation(f, v, i)` evaluates an
assumed-zero screen where the family provides it. For an entry on the
`NormL1` coordinate route, this is the signed margin ``|v_i|-c_i`` for
the working coefficient ``c_i``. The four-argument coordinate call
`kkt_violation(f, v, x, i)` reads the supplied point. In the ordinary
case it uses the signed margin in the point-aware working type at a
zero entry and ``|v_i+c_i\operatorname{sign}(x_i)|`` at a nonzero entry.
The `BlockIterate` route uses a group number and reduces entry scores.
[NormL1 coordinate formulas](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/norml1.jl#L170-L189).

For `GroupNormLinf`, an array index selects its group if it belongs
to one. An uncovered entry uses the gradient magnitude. A
`BlockIterate` index selects a group by its number, with the ordered
indices of the selected block checked against the function's group.
In ordinary finite arithmetic, a zero group takes the signed screen
``\lVert v_{G_k}\rVert_1-c_k`` with group coefficient ``c_k``.
A nonzero group uses the Euclidean distance from the negative
gradient to the group's subdifferential. Both `GroupNormLinf` and
`SparseGroupNormL2` reject the assumed-zero call. The sparse group
contract describes its
[point-aware forms](../../spec/sparsegroupnorml2.md#sparse-group-norm-l2.kkt.entry-classification).
[GroupNormLinf screen and group selection](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/groupnormlinf.jl#L422-L482)
and [SparseGroupNormL2 call routing](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/sparsegroupnorml2.jl#L230-L266).

## Reducing over a point

The full-point call `kkt_violation(f, v, x)` uses a family-specific
reduction. The `GroupNormLinf` implementation starts the maximum at
zero, includes the gradient magnitude on uncovered entries, and uses
the group distance even at a zero group. At zero this is the
Euclidean distance to the group's ℓ₁ ball. See the
[group ℓ∞ guide](../../group_penalty.md#group-linf-guide) for both forms.
[GroupNormLinf point reduction](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/groupnormlinf.jl#L490-L515)
and [distance at zero](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/normlinf.jl#L1080-L1094).

## Tests

- [NormL1 score tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/kkt_point.jl#L158-L188) exercise the
  signed screen and the full-point reduction.
- [GroupNormLinf zero-group test](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/groupnormlinf.jl#L1344-L1354)
  uses the gradient ``(5/8,5/8)`` and unit level. It checks the
  coordinate score ``1/4`` and the full-point distance ``\sqrt{2}/8``.
