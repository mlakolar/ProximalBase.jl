# Coordinate oracles

[`prox_coordinate!`](@ref) and the four-argument [`kkt_violation`](@ref) act on
one coordinate ``k`` of an iterate: an entry of a `SparseIterate`, or a group of
an `AtomIterate` (see [Entries and coordinates](@ref)). This section states what
they do on these two iterates. For a `SymmetricSparseIterate`, see also
[What each call checks](@ref).

## Steps on one coordinate

`prox_coordinate!(x, f, k, γ)` applies the proximal operator of ``\gamma f`` to
the current value of coordinate ``k``. `prox_coordinate!(x, f, k, u, γ)` applies
it to a trial value `u` instead, without writing `u` into `x` first. The trial
is a number for a `SparseIterate`, and a vector of the group's length for an
`AtomIterate`. Both forms write the result into coordinate ``k`` and return
it: the new value of the entry for a `SparseIterate`, and `atom(x, k)` for an
`AtomIterate`.

Only coordinate ``k`` can join the working set:

- A coordinate that is absent and whose result is zero stays absent.
- A coordinate whose result is nonzero becomes active.
- An active coordinate stays active, even when the step sets it to zero. No
  coordinate is removed, and no other coordinate changes.

Both forms support separable functions on both iterates. On an `AtomIterate`
they also support the group penalties of the package (`GroupNormL2`,
`GroupNormLinf`, `SparseGroupNormL2`, `GroupMCP` and `GroupSCAD`) whose groups
are those of `x`, and their `Scaled` and `Regularize` versions. The arguments
are checked before anything is written, and after each of these errors `x` is
unchanged:

| Error | Cause |
|:------|:------|
| `BoundsError` | `k` is not a coordinate of `x` |
| `DimensionMismatch` | the groups of a group penalty differ from those of `x` (see [The groups of an iterate](@ref)) |
| `ArgumentError` | any other function, such as a group penalty on a `SparseIterate`, `IndBallGroupL0`, or `Precompose` of a group penalty with an array |
| `DimensionMismatch` | a parameter array, such as the weights of `NormL1` or the center of `Regularize`, whose length is not `length(x)` |
| `ArgumentError` | a trial `u` whose element type is not real |
| `DimensionMismatch` | for an `AtomIterate`, a trial `u` whose length is not the size of group ``k`` |

On an `AtomIterate`, an error outside this list, such as an entry of `u` whose
reading throws or a value `1.5` for an iterate of integers, can leave entries of
the group written. A group with a nonzero entry is then active.

```@example coordinate_steps
using ProximalBase

x = SparseIterate(4)
x[1] = 0.5
v = prox_coordinate!(x, NormL1(1.0), 1, 1.0)        # soft thresholding of 0.5 at 1
@assert v == 0 && isactive(x, 1)                    # an active entry stays active at zero
v = prox_coordinate!(x, NormL1(1.0), 2, 3.0, 1.0)   # the trial value 3.0 is not written first
@assert v == 2.0 == x[2] && isactive(x, 2)
prox_coordinate!(x, NormL1(1.0), 3, 0.5, 1.0)       # a zero result
@assert !isactive(x, 3)                             # leaves an absent entry absent
try
    prox_coordinate!(x, NormL1(1.0, ones(3)), 4, 1.0)   # three weights for four entries
    error("expected a DimensionMismatch")
catch e
    @assert e isa DimensionMismatch
end
@assert Vector(x) == [0.0, 2.0, 0.0, 0.0]
Vector(x)
```

A trial `u` that shares memory with `x`, such as `atom(x, j)` or a view of `x`
with its entries permuted, is copied before anything is written, so the result
does not depend on the order of the writes. Here the trial is group 1 itself in
reverse order. Writing its entries one at a time without the copy would give
the trial ``(4, 4)`` and the result ``(3, 3)``.

```@example coordinate_steps
f = GroupNormL2(1.0, [1:2, 3:4])                    # weights √2
a = AtomIterate([3.0, 4.0, 0.0, 0.0], f)
r = prox_coordinate!(a, f, 1, view(a, [2, 1]), 1.0) # the trial (4, 3)
@assert collect(r) ≈ (1 - sqrt(2) / 5) .* [4.0, 3.0]
@assert r == atom(a, 1)
```

## Block trials

Block coordinate descent computes a trial point ``u`` for group ``k`` (see
[AtomIterate](@ref)). The supported way to take the group step at ``u`` is
`prox_coordinate!(x, f, k, u, γ)`. It gives the values that writing `u` into
group ``k`` and then calling `prox_coordinate!(x, f, k, γ)` gives. The working
sets differ: writing a nonzero `u` activates the group, while the five-argument
form activates it only when the result is nonzero. A group whose step ends at
zero therefore leaves no trace in the working set. After an error of the list
in [Steps on one coordinate](@ref), `x` is unchanged.

```@example block_trials
using ProximalBase

f = GroupNormL2(1.0, [1:2, 3:4])
x = AtomIterate(4, f)
r = prox_coordinate!(x, f, 1, [3.0, 4.0], 1.0)
@assert collect(r) ≈ (1 - sqrt(2) / 5) .* [3.0, 4.0]
@assert collect(activecoordinates(x)) == [1]
prox_coordinate!(x, f, 2, [0.5, 0.5], 1.0)          # the norm of u is below √2
@assert iszero(atom(x, 2)) && !isactive(x, 2)

z = AtomIterate(4, f)                               # writing the trial first
z[3] = 0.5
z[4] = 0.5
prox_coordinate!(z, f, 2, 1.0)
@assert iszero(atom(z, 2)) && isactive(z, 2)        # same values, but the write activated group 2

before = (Vector(x), collect(activecoordinates(x)))
try
    prox_coordinate!(x, GroupNormL2(1.0, [[1, 3], [2, 4]]), 1, [9.0, 9.0], 1.0)
    error("expected a DimensionMismatch")
catch e
    @assert e isa DimensionMismatch
end
@assert (Vector(x), collect(activecoordinates(x))) == before
```

Do not bypass these methods by writing directly into the iterate's private storage. Use the five-argument `prox_coordinate!` form for block trial points; it keeps the working set consistent even when the trial is discarded.

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
