# Coordinate oracles

A solver owns trial construction, penalty index conversion, storage writes,
cache updates and active-set policy. [`prox_block`](@ref) and
[`prox_block!`](@ref) supply the mathematical result; the four-argument
[`kkt_violation`](@ref) screens an iterate coordinate at zero.

## Steps on one coordinate

```@example coordinate_step
using ProximalBase
x = SparseIterate(4)
x[1] = 0.5
x[1] = prox_block(NormL1(1.0), x[1], 1, 1.0)
@assert iszero(x[1]) && isactive(x, 1)
x[2] = prox_block(NormL1(1.0), 3.0, 2, 1.0)
@assert x[2] == 2.0 && isactive(x, 2)
```

Writing zero preserves an existing active coordinate and leaves an absent
coordinate absent. A solver that needs a different working-set policy applies
it explicitly through `activate!` or `dropzeros!`.

## Block trials

Compute in compact numerical buffers before writing the iterate. This avoids
activating a group whose nonzero trial thresholds to zero. `out === u` is
supported; other overlapping buffers are rejected. The compact buffers have
the penalty group's length and order. Their shape, group ID and real element
types are checked before writes; a failing element read or conversion may
leave a partial numerical result, but the iterate has not been written.

```@example block_trials
using ProximalBase
f = GroupNormL2(1.0, [1:2, 3:4])
x = AtomIterate(4, f)
@assert ProximalBase.samegroups(x, f)
u, out = [3.0, 4.0], zeros(2)
prox_block!(out, f, u, 1, 1.0)
@assert nactive(x) == 0
copyto!(atom(x, 1), out)
@assert collect(atom(x, 1)) ≈ (1 - sqrt(2) / 5) .* u
@assert collect(activecoordinates(x)) == [1]
prox_block!(out, f, [0.5, 0.5], 2, 1.0)
copyto!(atom(x, 2), out)
@assert all(iszero, atom(x, 2)) && !isactive(x, 2)
```

For a separable penalty acting on a solver's group, call the scalar oracle
with each entry's global index. A separable penalty does not own that group
partition, so its scalar entry index cannot be replaced by an iterate group ID.

## Screening a zero coordinate

`kkt_violation(f, g, x, k)`, with `g` the gradient of the smooth part at `x`,
screens coordinate ``k`` at zero. A positive value means that the coordinate,
if it is zero, should enter the working set, and a value ``\le 0`` that zero
satisfies its optimality condition. It is not a stationarity residual of a
nonzero coordinate.

On an `AtomIterate`, ``k`` is a group. For a group penalty, the screen uses the
penalty's own dual norm and weights, and it does not read the values of `x` in
group ``k``:

| Function | Screen of group ``k`` |
|:---------|:----------------------|
| `GroupNormL2(λ0, groups, w)` | ``\lVert g_{G_k}\rVert_2 - \lambda_0 w_k`` |
| `GroupNormLinf(λ0, groups, w)` | ``\lVert g_{G_k}\rVert_1 - \lambda_0 w_k`` |
| `SparseGroupNormL2(λ1, λ2, groups, w)` | ``\lVert S_{\lambda_1}(g_{G_k})\rVert_2 - \lambda_2 w_k``, with ``S_{\lambda_1}`` soft thresholding at ``\lambda_1`` |
| `GroupMCP(λ, groups, w)`, `GroupSCAD(λ, groups, w)` | ``\lVert g_{G_k}\rVert_2 - \lambda w_k`` |
| `Scaled(f, α)` with ``\alpha > 0`` | ``\alpha`` times the screen of `f` at ``g/\alpha`` |
| `Regularize(f, ρ, c)` | the screen of `f` at ``g - \rho c``, or at ``g`` without `c` |
| a separable `f` | the largest `kkt_violation(f, g, i)` over the entries ``i`` of group ``k`` |

The weights ``w_k`` are those of the penalty, with the defaults of its
docstring: the square root of the group size, and 1 for `GroupNormLinf`.
`Scaled(f, 0)` is the function that [`Scaled`](@ref) builds at zero scale. For
a group penalty with finite levels, a finite shape parameter (`a` of `GroupMCP`
and `GroupSCAD`) and finite weights this is the zero function, whose screen on
an `AtomIterate` is `maximum(abs, view(g, atomindices(x, k)))`; a level or shape
parameter that is not finite makes `Scaled(f, 0)` throw an `ArgumentError`
when it is built. Otherwise, as
with an infinite weight or an indicator, the screen is that of the function
`Scaled` builds.

The table is not exhaustive. `IndBallGroupL0` has no screen and throws a
`MethodError`, and so do some functions without a screen of their own, such as
`NormL2`. Other functions behave differently: the array `Precompose` of a group
penalty has a screen on an `AtomIterate` (see [Entrywise scaling](@ref)), and
`MoreauEnvelope` of a function that is not separable throws an
`ArgumentError`. On an `AtomIterate`, the call throws a
`DimensionMismatch` when the groups of a group penalty differ from those of
`x`, and a `BoundsError` when ``k`` is not a group.

The screens differ in sign where it matters. At ``g = (5/8, 5/8)`` with
``\lambda_0 = 1``, the screen of `GroupNormLinf` is 0.25, so the group must
enter, while the ℓ₂ formula with weight 1 is about ``-0.116`` and would screen
the group out.

```@example screening
using ProximalBase

x2 = AtomIterate(2, [1:2])
g = [5 / 8, 5 / 8]
v = kkt_violation(GroupNormLinf(1.0, [1:2]), g, x2, 1)
@assert v == 0.25                                   # ‖g‖₁ − 1 > 0: the group must enter
@assert sqrt(sum(abs2, g)) - 1 ≈ -0.11611652351681556   # the ℓ₂ formula would screen it out
@assert kkt_violation(GroupNormL2(1.0, [1:2]), g, x2, 1) ≈ sqrt(sum(abs2, g)) - sqrt(2)
@assert kkt_violation(Scaled(GroupNormLinf(1.0, [1:2]), 2.0), g, x2, 1) == 2 * (sum(g) / 2 - 1)
@assert kkt_violation(Regularize(GroupNormL2(1.0, [1:2]), 1.0, [1.0, 1.0]), [3.0, 4.0], x2, 1) ≈
        sqrt(13) - sqrt(2)                          # the screen at (3, 4) − (1, 1)
v
```

On a `SparseIterate`, ``k`` is an entry. After the check of the real
element types, the call checks that ``k`` is one of `1:ncoordinates(x)` (a
`BoundsError` otherwise) and then that `g` has the length of `x` (a
`DimensionMismatch` otherwise); a matrix-shaped or flat `g` of that length is
accepted; these two checks do not scan `g`. For a separable `f` the call
then returns `kkt_violation(f, g, k)`. A group penalty screens
each entry with its group. When every entry of the group is zero in `x`, the
call returns the screen of the group in the table. Otherwise it returns the
screen of an entry at zero inside a nonzero group, `abs(g[k])`, or
`abs(g[k]) - λ1` for `SparseGroupNormL2`. An entry in no group gives
`abs(g[k])`.

```@example screening
f = GroupNormL2(1.0, [1:2, 3:4])
y = SparseIterate(4)
gs = [3.0, 4.0, 0.0, 0.0]
@assert kkt_violation(f, gs, y, 1) ≈ 5 - sqrt(2)    # group 1 of y is zero: the group screen
y[2] = 1.0
@assert kkt_violation(f, gs, y, 1) == 3.0           # entry 1 of a nonzero group: abs(gs[1])
```

For the distinction between entry-level function methods, coordinate-level iterate methods and the Julia dispatch design, see [oracle implementation](developer/oracle_dispatch.md).
