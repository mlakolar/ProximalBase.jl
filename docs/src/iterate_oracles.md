# Coordinate and block operations

*Assumes: scalar proximal updates and coordinate indexing · You will learn: how to compute a coordinate result separately from writing it to storage.*

These are [proximal](glossary.md#Proximal-operator) and coordinate operations; they do not select a solver's [working set](glossary.md#Working-set).

The pure mathematical hook computes a proximal point at a supplied trial.
The solver then writes the result to storage. ProximalBase does not offer an operation that both computes a coordinate
proximal point and changes the iterate's active set.

## Entries, groups and independent variables

| Hook/call | Trial and index | Storage write afterward |
|:--|:--|:--|
| `prox_block(f,v,i,γ)` | Scalar v, global array-entry index i | `x[i]=result` for `SparseIterate` |
| `prox_block!(out,f,u,k,γ)` | Compact vectors in the selected group k's order | `copyto!(blockview(x,k),out)` after matching layouts |
| Scalar hook for a symmetric iterate | Scalar v, entry index `(c-1)p+r` derived from independent coordinate k | `coordinates(x)[k]=result` |

An active slot is never a coordinate or group index. `BlockIterate` has a storage partition,
not a penalty: `NormL1` has no groups and uses each global member index.
`ProximalBase.samegroups` accepts two BlockIterates; custom functions must
provide their own layout comparison or compare lists with `blockindices`.

```@example coordinate_trial
using ProximalBase
x = SparseIterate(4)
x[1] = 0.5
trial = x[1]  # an algorithm would include its gradient step here
new = prox_block(NormL1(1.0),trial,1,1.0)
@assert x[1] == 0.5
x[1] = new
@assert x[1] == 0 && isactive(x,1)
x[2] = prox_block(NormL1(1.0),3.0,2,1.0)
@assert x[2] == 2 && isactive(x,2)
Vector(x)
```

## [Compact block buffers](@id block-proximal-oracle)

The generic vector `prox_block` allocates a plain floating `Vector` of `length(u)`
and invokes `prox_block!`. [`GroupNormL2`](@ref), [`GroupNormLinf`](@ref), [`SparseGroupNormL2`](@ref)
and [`GroupMCP`](@ref) are the built-in group implementations.
`k` selects `f.groups[k]`, and `u` follows that group's ordered global indices.
These methods check the group, buffer lengths, real one-based storage, step and overlap
before writes. Custom extensions must provide these checks themselves. Exact `out === u` is supported by the
contract; other overlap must be rejected by the extension. The generic
allocating wrapper adds no such checks on the extension's behalf.

**Caller precondition:** output must be independent of iterate storage and
solver/model caches. A view such as `activevalues(x)`, `coordinates(x)` or
`blockview(x,k)` aliases storage even when its parent is a plain vector.
The mathematical hook cannot establish ownership from array types.
Compute in an independent numerical vector, then use the public storage writer.
An exception inside a custom numerical computation can leave partial numerical
output; keeping it independent protects the iterate until the separate write.

The [block extension](examples/block_prox_extension.md) is a complete checked
implementation. Implementing this hook does not automatically supply a full
array value, full-array prox, traits, `kkt_violation` method, symmetric-iterate interface or
`Scaled` compact-vector forwarding.

## [Scoring a coordinate](@id kkt-screens)

`kkt_violation(g,v,i)` reports the signed screen of coordinate `i` **assumed
zero**. `NormL1` returns ``|v_i|-\lambda_0w_i``. A positive value means zero fails
the condition. The routine does not inspect whether the coordinate is actually
zero, and it takes no iterate.

`kkt_violation(g,v,x,k)` reads `x` and interprets `k` in x's coordinate space.
For `NormL1`, `ElasticNet`, `ZeroFunction`, `LinearFunction` and `SqrNormL2` it
returns the signed screen at a zero coordinate and a nonnegative stationarity
mismatch at a nonzero one (see [Mathematical operations](operations.md) for each
scalar). For `NormL1Nonneg` its point-aware entry formulas and non-finite overrides are
described in [Mathematical operations](operations.md) and the [numerical contract](numerics.md).
`GroupNormL2` has the same two scores for a group, and `Scaled` forwards
them (below). `GroupNormLinf` has the signed margin ``\lVert v_G\rVert_1-c`` of its dual norm at a zero group, where `c = λ0 w[k]`, and,
at a nonzero group, the Euclidean distance from ``-v_G`` to the subdifferential of ``c\lVert\cdot\rVert_\infty`` at ``x_G``; its margin is the ℓ₁ screen of
[#634](https://github.com/mlakolar/ProximalBase.jl/issues/634), and its positive part is not a stationarity distance.
`kkt_violation(g,v,x)` is the nonnegative full-point measure
``\max(0,\max_k s_k)`` of the six scalar functions and `GroupNormL2` on a one-based array,
`SparseIterate` or `BlockIterate`: the scores ``s_k`` of the entries for the six scalar
functions, and for `GroupNormL2` the scores of the groups and ``\lvert v_i\rvert`` of the entries in no
group, a **maximum of block 2-norms** and not an ``\ell_\infty`` distance. For `GroupNormLinf` it is the maximum of zero,
of ``\lvert v_i\rvert`` of the entries in no group and of the distance of every group, a zero group included, which there is the
distance from ``v_G`` to the ℓ₁ ball of radius ``c`` and not the margin of the four-argument form. `NormL2` and `NormLinf` have the
full-point form and no other.

[`SparseGroupNormL2`](@ref sparse-group-l2-guide) also provides the two
point-aware forms. At a zero group it screens the soft-thresholded gradient.
At an active group it measures the block subgradient distance. The full-point
score takes the maximum with zero and includes the free-coordinate magnitudes.

[`GroupMCP`](@ref group-mcp-guide) has the four-argument form at a zero group, the screen
``\lVert v_G\rVert_2-\lambda_0w_k`` of `GroupNormL2`, and `abs(v[k])` at an entry in no group. It throws an
`ArgumentError` at a nonzero group and in the full-point form. Its function is not convex, so a zero score
does not imply a global minimizer.

| Iterate | Four-argument form (`k` is the index) |
|:--|:--|
| `SparseIterate` | Valid coordinate first, then gradient entry count equal to length(x); flat or matrix-shaped gradient of that length is accepted. The coordinate is read from `x` |
| `SymmetricSparseIterate` | Valid independent coordinate, then p² gradient entries; symmetric part of that gradient pair, the stored value of the coordinate, and the extent of the function's parameters (their mirrored entries are not compared). The score is in matrix-entry units. There is no full-point form: it throws `ArgumentError` |
| `BlockIterate`, scalar penalties | Group `k`. The maximum of the scores of the entries of the group: signed if every entry of the group is zero, nonnegative otherwise. A gradient whose length differs from `length(x)` throws `DimensionMismatch` |
| `BlockIterate`, `GroupNormL2` | Group `k`. The block score: the group screen at a zero group, the mismatch at a nonzero one. The group count, the layout of group `k` and the full gradient length are checked |
| `BlockIterate`, `SparseGroupNormL2` | Group `k`. Check the group count, ordered layout of group `k` and gradient length |
| `BlockIterate`, `GroupNormLinf` | Group `k`, with the checks of `GroupNormL2`. The margin `‖v_G‖₁ - c` at a zero group, the distance from `-v_G` to the subdifferential at a nonzero one |
| `BlockIterate`, `GroupMCP` | Group `k`, with the checks of `GroupNormL2`. The screen `‖v_G‖₂ - λ0 w[k]` at a zero group; `ArgumentError` at a nonzero one |
| Other arrays, scalar penalties | Real one-based axes, then gradient length equal to length(x) (`DimensionMismatch`); entry `k` is read from `x` |
| Other arrays, `GroupNormL2` | Entry index selects its group; point/gradient lengths and group extent checked. The block score of that group, `abs(v[k])` for an entry in no group (also when `x[k]` is nonzero) |
| Other arrays, `SparseGroupNormL2` | Entry `k` selects its group. An uncovered entry gives `abs(v[k])`. Check gradient length and group extent |
| Other arrays, `GroupNormLinf` | Entry index selects its group, with the checks of `GroupNormL2`. The margin at a zero group, the distance at a nonzero group, `abs(v[k])` for an entry in no group; the full-point form is the maximum of zero, of `abs(v[i])` over the entries in no group and of the distances of the groups, and the three-argument form throws `ArgumentError` |
| Other arrays, `GroupMCP` | Entry index selects its group, with the checks of `GroupNormL2`. The screen at a zero group, `abs(v[k])` for an entry in no group; `ArgumentError` at a nonzero group and in the full-point form |
| Any point, `NormL2`, `NormLinf`, `LeastSquaresLoss` and `QuadraticFunction` | No form for an entry or a group: `ArgumentError` that names the full-point form |

At an array that is not a `BlockIterate` and at a `SparseIterate`, where `k` is an entry index, a call of the four-argument
form for `NormL1` and `ElasticNet` reads `x[k]`, evaluates the score of a zero entry and that of a nonzero one, and selects
between them; `SqrNormL2` forms its gradient entry from `x[k]` (and from the center, when it has one); and each of the five
functions applies the rules for non-finite values to the entries of `v` and `x` that its formula uses. A call therefore does
more work than a call of the three-argument screen. For these five functions the caller chooses between the forms for a
coordinate that it knows to be zero: at those points, the three-argument form returns the same value as the four-argument form
with the same index when `v`, `x` and the parameters of the function (levels, weights, centers, coefficients) are `Float32` or
`Float64` arrays and values of one type, `v[k]` is finite and the function is not a `SqrNormL2` with an infinite level. A
`BlockIterate`, where `k` is a group and the form takes the maximum over the entries of the group, and a
`SymmetricSparseIterate`, where the form reads the symmetric gradient pair and the stored value, are not covered by this
statement.

For `SparseIterate`/`SymmetricSparseIterate`, invalid coordinate throws `BoundsError` before
extent mismatch throws `DimensionMismatch`, after checking one-based axes and
real element types. A flat or matrix-shaped gradient of the required entry
count is accepted. Opposing infinities in the selected symmetric gradient pair
throw `ArgumentError`. Parameter extents are checked for symmetric iterates;
mirrored parameters are a caller precondition, not an oracle check; the
three-argument form takes no iterate and makes no parameter-symmetry check.
`GroupNormL2` checks the number of groups, the selected ordered group
layout and `length(v) == length(x)`, then scores the group `Gₖ` with ``c=\lambda_0w_k``:
`norm(v[Gₖ]) - c` at a zero group and `norm(v[Gₖ] + c*x[Gₖ]/norm(x[Gₖ]))` at a nonzero one
(see [the group penalty](group_penalty.md)). Its three-argument form throws `ArgumentError`
because the penalty is not entrywise separable, as does a `SymmetricSparseIterate`
point. `NormL2` takes the whole of `x` as one block and has the full-point form
and no other: its three-argument and four-argument forms throw `ArgumentError`. `NormLinf` is not a block
score either: its full-point form is the Euclidean distance from ``-v`` to the subdifferential of
``\lambda\max_i\lvert x_i\rvert`` at `x` (at ``x=0``, the distance from `v` to the ℓ₁ ball of radius ``\lambda``;
at ``x\ne0``, with the face of the entries of largest magnitude, compared for equality after conversion to the
working type; see [the ℓ∞ norm](@ref norm-linf-kkt)), and its three-argument and four-argument
forms throw `ArgumentError`.

`LeastSquaresLoss` and `QuadraticFunction` use the full-point form to return
`‖v + ∇f(x)‖∞`. Inputs must have the same shape and must be independent of
the scratch the function writes. See [smooth stationarity scores](@ref smooth-kkt)
for accepted layouts, rounding, allocations and scratch ownership.

`Scaled` with a finite positive factor `α` evaluates the score of its function on `v/α` and multiplies it by `α`,
for the entrywise and group functions above (`GroupNormLinf` included) and, in the full-point form, `NormL2`, `NormLinf`,
`LeastSquaresLoss` and `QuadraticFunction` (the rounding and range limits
of that evaluation are in the [scaling guide](@ref scaled-kkt)).
Nested wrappers whose factors do not multiply to a normal number, an infinite factor,
and the functions without a score have unsupported cases, which are listed in
the [scaling guide](@ref scaled-kkt), and a function without a score gives a `MethodError` for the
call on the `Scaled` function.

For the penalty, zero, linear and squared-norm functions above, the full-point form checks real element types, one-based axes (`ArgumentError`),
`length(v) == length(x)` (`DimensionMismatch`) and the extent of the function's
parameters (for `GroupNormL2` the group extent and, on a `BlockIterate`, the whole layout:
the group count and each group; for `NormL2` and `NormLinf` no parameter). It makes a pass over the entries
(over the groups for `GroupNormL2`) and writes nothing. A second pass
that applies the rules for non-finite values entry by entry can follow it, for
instance when an entry of `v` or `x` is not finite or a score is `NaN` or infinite.
`NormLinf` makes several passes over `v` and `x` instead: one classification of each, the sums on the face,
the search for the level, which makes one pass for each level that the iteration visits, and the
sum of squares of the residual. The tests measure no allocation of this form for `NormLinf` on `Float32` or
`Float64` arrays of one type with a level of a hardware float type; a `BigFloat` level can allocate.
Dense `Float32` or `Float64` arrays of one type take the fast path of the other functions, where the tests
measure no allocation; other arrays, `SparseIterate` and `BlockIterate` included, are
read by entry, and element types such as `BigFloat` can allocate in their arithmetic. The
four-argument form on an array reads the parameters of entry `k`. The element
type, the rules for non-finite values and the rounding error are in the
[numerical contract](numerics.md).

A function defined outside the package gets its three-argument method's value from the
four-argument form at a zero coordinate (a group of zero entries) and an `ArgumentError` at
a nonzero one, unless it defines the four-argument method itself; under `Scaled` it has the
scaled three-argument value and that error also when it defines the method;
see [Writing your own function](extending.md). The built-in functions without a
score, such as `SqrtLeastSquaresLoss`, throw `MethodError`. A positive score at
a nonzero coordinate is a stationarity mismatch; see
[coordinate descent](coordinate_descent.md).

## Symmetric scalar steps

For symmetric-iterate coordinate k, compute `(r,c)=tril_cartesian(p,k)`, then use the
full-matrix entry index `i=(c-1)p+r`. The function's parameters must be symmetric;
validate the original function with `check_symmetric_parameters` before raw scalar hooks,
since the hook has no symmetric iterate to inspect and no numerical oracle
compares mirrored parameters.
Use a symmetric matrix gradient and the appropriate Frobenius curvature.

```@example symmetric_coordinate_trial
using ProximalBase
x = SymmetricSparseIterate(3)
k = tril_index(3,3,2)
g = NormL1(0.5,ones(3,3))
ProximalBase.check_symmetric_parameters(g,x)
r,c = tril_cartesian(3,k)
i = (c-1)*3+r
coordinates(x)[k] = prox_block(g,2.0,i,1.0)
@assert x[3,2] == x[2,3] == 1.5
Matrix(x)
```
