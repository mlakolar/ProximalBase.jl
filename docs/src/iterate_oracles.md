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
array value, full-array prox, traits, KKT screen, symmetric-iterate interface or
`Scaled` compact-vector forwarding.

## Screening a zero coordinate

For the supported scalar functions, `kkt_violation(g,grad,i)` reports a signed
violation of the zero-coordinate condition. `NormL1` returns
``|\mathrm{grad}_i|-\lambda_0w_i``. A positive value means zero fails the
condition. The routine does not inspect whether the coordinate is actually zero.

`kkt_violation(g,grad,x,k)` interprets k in x's coordinate space:

| Iterate | Interpretation/checks |
|:--|:--|
| `SparseIterate` | Valid coordinate first, then gradient entry count equal to length(x); flat or matrix-shaped gradient of that length is accepted |
| `SymmetricSparseIterate` | Valid independent coordinate, then p² gradient entries; symmetric part of that gradient pair, and the extent of the function's parameters (their mirrored entries are not compared) |
| `BlockIterate` | Maximum scalar screen over global members of group k for supported scalar-screen functions |
| Other arrays | No extra point/gradient extent validation; indexing/function checks still apply |

For `SparseIterate`/`SymmetricSparseIterate`, invalid coordinate throws `BoundsError` before
extent mismatch throws `DimensionMismatch`, after checking real element types.
The `BlockIterate` path does not add that same explicit full-gradient extent
preflight; ensure the gradient is full-length. Nonzero coordinates need their
own stationarity test, as explained in [coordinate descent](coordinate_descent.md).

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
