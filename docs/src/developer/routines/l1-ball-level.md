# [The shared ℓ₁-ball level](@id routine-l1-ball-level)

One level search serves the projection onto the boundary for
`IndBallL1` and the clipping proximal maps for `NormLinf` and
`GroupNormLinf`. The
family contracts describe the conditions on use and the accuracy of those
maps. See the [ball projection](../../spec/indballl1.md#spec-indballl1-ordinary),
[NormLinf range rules](../../spec/normlinf.md#spec-normlinf-range), and
[GroupNormLinf range rules](../../spec/groupnormlinf.md#spec-groupnormlinf-range).

## Ordinary path

For finite magnitudes ``y_i=\lvert x_i\rvert`` and positive radius
``a<\sum_i y_i``, the exact level ``\tau`` satisfies
``g(\tau)=\sum_i (y_i-\tau)_+=a``. Above the level, the projection
keeps magnitude ``y_i-\tau`` with the input sign. The complementary
clip keeps magnitude ``\min(y_i,\tau)`` with the input sign.

For ``x=(3,-2,1)`` and ``a=2``, the two magnitudes above the level
give ``(3-\tau)+(2-\tau)=2``. The level ``3/2`` gives projection
``(3/2,-1/2,0)`` and clip ``(3/2,-3/2,1)``.

The current selector starts with Condat's candidate pass. If the
candidate count grows large, it can hand over to a Michelot pass on
all entries. Both routes use the same refinement loop. At a trial level
with ``k>0`` magnitudes above it, the slope of ``g`` on that interval is
``-k``. The update adds ``(g(\tau)-a)/k`` to the level, with guards
for rounding and the next magnitude.

## Exceptional branches

If the sum of the magnitudes of finite entries overflows or exceeds
half the largest finite value in the working type, callers can read
both the magnitudes and the radius in power-of-two units, then rescale
the level. Non-finite inputs go through caller-specific endpoint paths.
For `IndBallL1`, a very small radius can make the computed level round
to the largest magnitude. The projection has a separate fallback that
sorts magnitudes and distributes the radius across gaps. Lowering the
level from the ``j``th magnitude ``b_j`` to the next spends
``j(b_j-b_{j+1})`` of the radius. When the remaining radius fits in the
current gap, the routine forms the output magnitudes from gaps and
shares of that remainder. This avoids subtracting a rounded global level
from the largest entries. See the
[IndBallL1 range rules](../../spec/indballl1.md#spec-indballl1-range).

## Source and tests

- [Candidate pass and handover](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/normlinf.jl#L583-L657),
  [level refinement](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/normlinf.jl#L739-L750),
  [NormLinf rescaling](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/normlinf.jl#L182-L216), and
  [group-level caller](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/groupnormlinf.jl#L238-L301).
- [Ball projection and cancellation check](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/indballl1.jl#L141-L200)
  and [sorted fallback](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/src/functions/indballl1.jl#L234-L274).
- [Threshold tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/normlinf.jl#L1024-L1053)
  cover the ordinary pass, a restart, and the handover.
  [Projection tests](https://github.com/mlakolar/ProximalBase.jl/blob/4e26f7324ced50544d9ab89ee3d32dbca2c3e830/test/indballl1.jl#L206-L269)
  compare against a reference at large magnitudes and tiny radii.
