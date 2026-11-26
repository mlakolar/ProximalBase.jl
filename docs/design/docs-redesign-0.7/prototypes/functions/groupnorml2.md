# Norm and group penalties (prototype: `GroupNormL2` entry)

*In the implementation this entry sits in `group_penalty.md` after `NormL2` and
`NormLinf`, followed by `SparseGroupNormL2`, `GroupNormLinf`, `GroupMCP` and `GroupSCAD`.*

## [Group lasso: `GroupNormL2`](@id entry-groupnorml2)

The group-lasso penalty. It sets whole groups of coefficients to zero together,
for example all dummy variables of one categorical predictor.

```math
g(x)=\lambda\sum_{k}w_k\,\|x_{G_k}\|_2,\qquad \lambda\ge0,\ w_k\ge0,
```

where ``G_1,G_2,\dots`` are disjoint groups of indices. Indices in no group are
not penalized.

| Constructor | Meaning |
|:--|:--|
| `GroupNormL2(λ, groups)` | default weights ``w_k=\sqrt{\lvert G_k\rvert}`` |
| `GroupNormL2(λ, groups, w)` | one weight per group |

`groups` is a vector of index vectors or ranges, such as `[1:2, 3:5]` or
`[[1, 4], [2, 3]]`. The constructor checks that groups are nonempty, disjoint and
contain positive indices, and that `λ` and the weights are nonnegative and not `NaN`.
It copies `groups` and `w`.

```@example entry-gl
using ProximalBase

g = GroupNormL2(1.0, [1:2, 3:4])
g.w                                   # √2 for each group of two
```

```@example entry-gl
x = [3.0, 4.0, 0.1, 0.1]
prox(g, x, 1.0)                       # the second group becomes zero
```

### Proximal operator: group soft thresholding

For finite entries, finite parameters and a finite step ``\gamma>0``,

```math
\operatorname{prox}_{\gamma g}(v)_{G_k}=
\max\Bigl(1-\frac{\gamma\lambda w_k}{\|v_{G_k}\|_2},\ 0\Bigr)\,v_{G_k},
```

with the zero vector for a group whose norm is zero; entries in no group are unchanged.
A group whose norm is at most ``\gamma\lambda w_k`` becomes zero. Any other group keeps
its direction and its norm decreases by ``\gamma\lambda w_k``. Infinite levels or
weights, zero steps and non-finite entries follow the
[endpoint rules](@ref group-l2-endpoints); for example, an infinite weight sets its
group to zero even at a zero step.

![Group soft thresholding in two dimensions.](../../figures/group-shrink.svg)

| Operation | Call | Supported |
|:--|:--|:--|
| value | `g(x)` | ✓ |
| proximal step | `prox(g, x, γ)`, `prox!(out, g, x, γ)` | ✓ scalar step ``\gamma\ge0`` |
| value at the new point | `value_and_prox!(out, g, x, γ)` | ✓ returns `g(out)`, the documented value; the computed value can differ from `g(out)` in the last digit |
| one group | `prox_block(g, u, k, γ)`, `prox_block!(out, g, u, k, γ)` | ✓ on a separate buffer `u` holding the entries of group `k` |
| optimality score | `kkt_violation(g, v, x, k)`, `kkt_violation(g, v, x)` | ✓ for one group, or the whole point |
| one coordinate | `prox_block(g, xᵢ, i, γ)` | — groups are not separable |
| restriction | `restrict(g, S)` | — |
| gradient | `gradient(g, x)` | — |

In `kkt_violation(g, v, x, k)`, `k` is an entry index when `x` is an array (the score
of the group that contains entry `k`, or of entry `k` if it is in no group) and a
group number when `x` is a `BlockIterate`.

### Updating one group of a `BlockIterate`

`prox_block` and `prox_block!` compute on a separate buffer and do not write into an
iterate. Copy the group out, compute, and write the result back:

```@example entry-gl
xb = BlockIterate(copy(x), g)         # the iterate's groups are g's groups
u = collect(blockview(xb, 1))         # the entries of group 1
out = similar(u)
prox_block!(out, g, u, 1, 1.0)        # out must not share memory with xb
copyto!(blockview(xb, 1), out)
collect(xb)
```

### Inputs and storage

✓ vectors, matrices (groups use linear indices) and strided views ·
✓ `SparseIterate` ·
◐ `BlockIterate`: values and proximal steps work for any partition into groups. The
score of one group needs the iterate to have as many groups as `g` and the selected
group to list the same indices in the same order; the whole-point score needs every
group to match.
A `BlockIterate` must cover every index, so if `g` leaves entries in no group, either
score a plain array, or give `g` zero-weight singleton groups for those entries and
build the iterate with `BlockIterate(x, g)` ·
— `SymmetricSparseIterate`; use `Matrix(x)` instead.
Element types: `Float64` or `Float32`, the same type for data, parameters, step and
output ([precision rules](@ref numerical-usage)).

### Good to know

- The order of indices within a group sets the order of the buffer `u` for
  `prox_block`: for the group `[4, 1]`, `u = [x[4], x[1]]`.
- `prox_block!(u, g, u, k, γ)` may overwrite `u`; another overlap, or an output that
  shares memory with iterate storage, is not supported.
- `prox_block` is not available for `Scaled(g, α)`; scale ``\lambda`` instead.
- Negative and `NaN` steps throw `ArgumentError` before anything is written.

**See also:** [`SparseGroupNormL2`](@ref) (adds an ℓ₁ term within groups),
[`GroupMCP`](@ref) and [`GroupSCAD`](@ref) (nonconvex group penalties),
[`GroupNormLinf`](@ref), [`NormL2`](@ref) (one group containing everything) ·
[guarantees](@ref spec-groupnorml2) · [API](@ref GroupNormL2)
