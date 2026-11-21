# [Cardinality selection](@id routine-cardinality-selection)

`NormL0` makes a keep-or-drop decision at each penalized entry.
`IndBallL0` projects with a shared budget of counted nonzeros.
The [NormL0 contract](../../spec/norml0.md#spec-norml0-endpoints)
and the [IndBallL0 contract](../../spec/indballl0.md#spec-indballl0-endpoints)
own the exact comparison, input and endpoint rules.

## Scalar keep-or-drop decision

For finite ``v`` and positive finite level ``\lambda`` and step
``\gamma``, keeping a nonzero entry at ``v`` costs ``\gamma\lambda`` in
the proximal objective. Dropping it to zero costs ``v^2/2``. The minimizer
keeps ``v`` when ``v^2>2\gamma\lambda`` and returns zero on a tie.

With homogeneous `Float32` inputs, the scalar comparison widens both
products to `Float64`. On the homogeneous `Float64` route, the kernel can use the ordinary products when the right
product is normal and the rounded products differ. When that comparison
cannot decide, the kernel compares normalized exponents and aligns
integer significand products for the close cases. This preserves the
tie decision for the represented inputs even when the ordinary products
would overflow or underflow.

## Projection with a support budget

Keeping an independent coordinate saves its squared magnitude in
the squared-distance objective. Under a budget ``s``, a minimum keeps
the largest magnitudes up to that budget and copies the chosen
values unchanged. For ``x=(3,-2,1)``, a budget of two gives
``(3,-2,0)``. At unit step, the scalar penalty with ``\lambda=2``
gives ``(3,0,0)`` because dropping ``-2`` is a tie.

When a dense projection with all entries counted must drop entries and
the budget exceeds one, sampled quickselect finds the cutoff magnitude.
The selector keeps entries above the cutoff and fills the remaining
quota from equal magnitudes in logical column-major order. The
`SparseIterate` selector includes the logical coordinate in its key,
so a different active-slot order leaves the tie decision unchanged.

For a symmetric input with counted diagonals, a diagonal costs one
budget position, and an off-diagonal pair costs two. The implementation
selects and sorts the eligible magnitude prefixes of both classes. It
compares the gain from the next pair with the loss from the displaced
diagonals in aligned integer units. The smallest maximizing pair count
resolves a split tie. With free diagonals, the projection copies the
diagonal and retains up to ``\lfloor s/2\rfloor`` pairs.

## Source and tests

- [Scalar decisions and comparisons](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/norml0.jl#L52-L98),
  [dense selection](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/indballl0.jl#L191-L248),
  [sampled quickselect](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/indballl0.jl#L278-L320),
  and [sparse keys](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/indballl0.jl#L323-L355).
- [Symmetric selection](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/indballl0.jl#L503-L539)
  and [split comparison](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/indballl0.jl#L625-L680).
- [Scalar exact-objective tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/norml0.jl#L34-L75)
  and [independent support enumeration](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/indballl0_reference.jl#L60-L117)
  cover ties, subnormals, and magnitudes whose squares would overflow.
