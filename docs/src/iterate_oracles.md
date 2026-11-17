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
| `prox_block!(out,f,u,k,γ)` | Compact vectors in custom group k's order | `copyto!(blockview(x,k),out)` after matching layouts |
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

## Compact block buffers

The generic vector `prox_block` allocates a floating Vector of `length(u)`
and invokes the custom `prox_block!`. No built-in function implements a compact group proximal update by default. An extension must validate group ID, dimensions, real one-based
buffers and overlap before writes. Exact `out === u` is supported by the
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

## Scoring a coordinate

`kkt_violation(g,v,i)` reports the signed screen of coordinate `i` **assumed
zero**. `NormL1` returns ``|v_i|-\lambda_0w_i``. A positive value means zero fails
the condition. The routine does not inspect whether the coordinate is actually
zero, and it takes no iterate.

`kkt_violation(g,v,x,k)` reads `x` and interprets `k` in x's coordinate space.
For `NormL1`, `ElasticNet`, `ZeroFunction`, `LinearFunction` and `SqrNormL2` it
returns the signed screen at a zero coordinate and a nonnegative stationarity
mismatch at a nonzero one (see [Mathematical operations](operations.md) for each
scalar). `kkt_violation(g,v,x)` is the nonnegative full-point measure
``\max(0,\max_k s_k)`` of these five functions on a one-based array,
`SparseIterate` or `BlockIterate`, read by entry.

| Iterate | Four-argument form (`k` is the index) |
|:--|:--|
| `SparseIterate` | Valid coordinate first, then gradient entry count equal to length(x); flat or matrix-shaped gradient of that length is accepted. The coordinate is read from `x` |
| `SymmetricSparseIterate` | Valid independent coordinate, then p² gradient entries; symmetric part of that gradient pair, the stored value of the coordinate, and the extent of the function's parameters (their mirrored entries are not compared). The score is in matrix-entry units. There is no full-point form: it throws `ArgumentError` |
| `BlockIterate` | Group `k`. For the five functions above, the maximum of the scores of the entries of the group: signed if every entry of the group is zero, nonnegative otherwise. For `GroupNormL2`, see [the group penalty](group_penalty.md) |
| Other arrays | Real one-based axes, then gradient length equal to length(x) (`DimensionMismatch`); entry `k` is read from `x` |

For `SparseIterate`/`SymmetricSparseIterate`, invalid coordinate throws `BoundsError` before
extent mismatch throws `DimensionMismatch`, after checking real element types.
For a `BlockIterate` and for other arrays, a gradient whose length differs from
`length(x)` throws `DimensionMismatch`.

A function defined outside the package that has only the three-argument method
gets that method's value from the four-argument form at a zero coordinate (a
group of zero entries) and an `ArgumentError` at a nonzero one;
see [Writing your own function](extending.md). The built-in functions without a
score, such as `SqrtLeastSquaresLoss`, throw `MethodError`. A positive score at
a nonzero coordinate is a stationarity mismatch; see
[coordinate descent](coordinate_descent.md).

## Symmetric scalar steps

For symmetric-iterate coordinate k, compute `(r,c)=tril_cartesian(p,k)`, then use the
full-matrix entry index `i=(c-1)p+r`. The function's parameters must be symmetric;
validate the original function with `check_symmetric_parameters` before raw scalar hooks,
since the hook has no symmetric iterate to inspect and no oracle checks it.
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
