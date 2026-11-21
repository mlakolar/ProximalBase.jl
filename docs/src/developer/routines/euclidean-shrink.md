# [Euclidean shrinkage and radial projection](@id routine-euclidean-shrink)

Euclidean shrinkage moves a vector toward zero along its own direction.
`NormL2` applies it over the argument, and `GroupNormL2` applies it to
each group with its own threshold. `IndBallL2` also uses a radial scale
when projecting onto a Euclidean ball. See the
[NormL2 range rules](../../spec/norml2.md#spec-norml2-range),
[GroupNormL2 range rules](../../spec/groupnorml2.md#spec-groupnorml2-range),
and [IndBallL2 range rules](../../spec/indballl2.md#spec-indballl2-range).

## Ordinary formulas

For a nonzero finite vector ``x``, the closest point having norm ``q``
lies in the direction of ``x`` by Cauchy–Schwarz. The proximal problem
for a nonnegative finite coefficient ``c`` thus minimizes
``\tfrac12(q-\lVert x\rVert_2)^2+cq`` over ``q\ge0``. The resulting norm
is ``\max(\lVert x\rVert_2-c,0)``. With ``n=\lVert x\rVert_2>c``, the
map stores ``(1-c/n)x``. When ``n\le c`` it gives zero. For a group
``G_k``, the same formula uses ``x[G_k]`` and ``c=\gamma\lambda_0w_k``.

For projection onto a ball of positive finite radius ``r``, the scalar
problem chooses ``q=\min(\lVert x\rVert_2,r)``. The projection copies
a point on or inside the ball and stores ``(r/n)x`` when ``n>r``. For example,
``x=(3,-4)`` has norm ``5``. Shrinkage at ``c=1`` gives
``(12/5,-16/5)``, and projection at ``r=5/2`` gives ``(3/2,-2)``.

## Range recovery

The shared shrinkage kernel uses the ordinary threshold and square sum
when their range checks pass. For a finite nonzero group and positive
finite factors on the recovery path, it represents the norm as
``n_s2^k``. The scaled norm ``n_s`` is the square root of the
[scaled sum of squares](@ref routine-squared-residuals) over the
selected indices. The shrinkage decision then uses the ratio
``(\gamma\lambda w\,2^{-k})/n_s``. The comparison and scaling can proceed
when an unscaled intermediate is out of range.

The general ball projection route can compare ``n_s`` with ``r2^{-k}``
when it has a scaled norm. Its recovered write also carries the radius
exponent separately. It divides the radius significand by ``n_s``
before restoring the exponent in each output product. This keeps a
tiny radius from rounding to zero during an intermediate division.
Non-finite cases use the family endpoint rules for
[NormL2](../../spec/norml2.md#spec-norml2-endpoints),
[GroupNormL2](../../spec/groupnorml2.md#spec-groupnorml2-endpoints),
and [IndBallL2](../../spec/indballl2.md#spec-indballl2-endpoints).

## Source and tests

- [Shared shrinkage and scaled ratio](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/numerical_kernels.jl#L857-L947),
  [NormL2 caller](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/norml2.jl#L78-L84),
  and [full-array and compact group callers](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/groupnorml2.jl#L204-L239).
- [Ball norm and comparison](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/indballl2.jl#L42-L68)
  and [radial writes](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/indballl2.jl#L171-L228).
- [Analytic shrinkage](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/groupnorml2.jl#L79-L90),
  [group range cases](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/groupnorml2.jl#L310-L339),
  [analytic projection](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/indballl2.jl#L98-L133),
  and [ball range cases](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/indballl2.jl#L203-L225).
