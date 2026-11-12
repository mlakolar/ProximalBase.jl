# Symmetric sparse iterate contract

This document describes the current public `SymmetricSparseIterate`
representation, coordinate indexing, and mathematical operations.
It replaces the previous full-package design, which remains available in
the [preserved historical snapshot](https://github.com/mlakolar/ProximalBase.jl/blob/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design/symmetric-sparse-iterate-0.6-acceptance.md).

## Mathematical representation

A `SymmetricSparseIterate` represents a real symmetric
``p\times p`` matrix using

```math
q=\frac{p(p+1)}{2}
```

independent lower-triangular coordinates, numbered down columns. The
matrix positions ``(r,c)`` and ``(c,r)`` refer to the **same
variable**, so only one value is stored for each pair.

The symmetric iterate is **not** an `AbstractArray` or `AbstractMatrix`. Although
`size(x)` returns `(p,p)` and `eltype(x)` is defined, it does not
support ordinary array iteration, `length`, `axes`, `vec`, or
broadcasting. This prevents confusion between independent-coordinate IDs
and full-matrix linear indices.

### Three distinct forms of indexing

| Julia expression | Meaning |
|:--|:--|
| `x[r,c]` | Read/write one symmetric matrix variable; `x[c,r]` refers to the same value |
| `coordinates(x)[k]` | Read/write independent lower-triangle coordinate `k` |
| `matrixview(x)[i]` | Read full-matrix entry by ordinary Julia linear index `i` |
| `activecoordinates(x)[a]` | Active coordinate ID in packed slot `a` |
| `x[k]` | Deliberately rejected with `ArgumentError`: the index space is ambiguous |

For example:

```julia
x = SymmetricSparseIterate(3)
x[3, 1] = 2.0
@assert x[1, 3] == 2.0

k = tril_index(3, 3, 1)
@assert coordinates(x)[k] == 2.0
@assert matrixview(x)[1, 3] == 2.0
```

The public `matrixview(x)` is live but **read-only**, while
`Matrix(x)` creates an independent dense copy. No view derived
from the public matrix view may write to symmetric matrix storage. Bounds checks
for `x[r,c]` precede narrowing of other integer index types.

## Activity, copies and matrix imports

`SymmetricSparseIterate` follows the same working-set rules as the
iterate-family contract, but in **independent-coordinate space**.

- `activate!(x,k)` retains an active zero; `dropzeros!(x)` removes
  numerical zeros while preserving other entries and their values.
- `copy(x)` makes an independent state copy; `similar(x)` creates
  an empty symmetric iterate with the same dimensions.
- `copyto!(dest,src)` between symmetric iterates of the same size replaces both
  numerical values and activity, staging fallible stored conversions
  before updating the destination.
- Creating or importing a symmetric iterate from a matrix requires real data,
  compatible one-based ``p\times p`` dimensions and **exact**
  symmetry. Equal mirrored `NaN` values are accepted, and signed
  zeros compare equal.
- A source matrix overlapping symmetric matrix storage must be rejected. Matrix
  import replaces activity with converted nonzeros in increasing
  independent-coordinate order. A conversion failure after clearing
  destination state may leave a partially imported result.
- `==` compares the represented matrix values. `isequal` and
  `hash` also reflect signed-zero and `NaN` distinctions.
  There is no general symmetric-iterate/array `isapprox` or array equality method.

Symmetric-iterate matrix imports require exact symmetry; proximal calls with an
symmetric-iterate destination instead use the **symmetric part** of a matrix
argument when supported.

## Frobenius geometry

The symmetric iterate represents a full symmetric matrix ``X``, not just its
coordinate vector. Its squared Frobenius norm is

```math
\|X\|_F^2
=\sum_{i=1}^{p}X_{ii}^2
 +2\sum_{i>j}X_{ij}^2.
```

Each independent off-diagonal variable contributes **twice** to matrix
values and squared Frobenius distances. The norm of
`coordinates(x)` counts that independent coordinate only once.
For separable symmetric entry penalties, the factor of two appears in
both the function and the proximal squared distance; the scalar
threshold itself is therefore unchanged.

## Supported mathematical operations

Mathematical function methods accept a `SymmetricSparseIterate`
**only when that input/output combination is explicitly supported**.

| Operation | Supported function families and behavior |
|:--|:--|
| `f(x)`, `prox(f,x,γ)` | `SeparableFunction`, `SqrNormL2`, `LinearFunction`, `ZeroFunction`, and supported `Scaled` wrappers |
| `gradient(f,x)` | `SqrNormL2`, `LinearFunction`, and supported scaling; the allocated result is a dense symmetric matrix |
| `prox!(out,f,x,γ)` with a symmetric iterate `out` | An independent-coordinate symmetric iterate is written; only scalar steps `γ` are supported |
| `kkt_violation(f,grad,x,k)` | The chosen independent coordinate, using the symmetric matrix gradient and relevant parameter pair |

`ZeroFunction` supports symmetric-iterate values and proximal maps, but no symmetric iterate
gradient. `LeastSquaresLoss` and `QuadraticFunction` have no
direct symmetric-iterate methods.

The fused operations `value_and_gradient!` and
`value_and_prox!` return the value at the input or **stored output**,
respectively, just as for ordinary arrays. An allocating `prox`
returns a new independent symmetric iterate. Array-valued proximal steps with
a symmetric iterate destination are rejected before writes.

### Symmetric parameters and raw scalar hooks

Per-entry parameter arrays must have exactly equal mirrored entries on
the pairs examined by the relevant operation. Use
`ProximalBase.check_symmetric_parameters(f, x)` to validate **all**
declared pairs recursively before constructing raw scalar trials.

For independent coordinate `k`, first compute

```julia
r, c = tril_cartesian(p, k)
i = (c - 1) * p + r
```

and pass the **full-matrix entry index** `i` to
`prox_block(f, v, i, γ)` or `value_entry(f, v, i)`.
A raw scalar hook has no `SymmetricSparseIterate` argument and therefore does not verify
symmetric parameter pairs itself.

The `SymmetricSparseIterate`-specific `kkt_violation` checks its chosen mirrored pair,
uses the symmetric matrix gradient, and rejects an opposing-infinity
gradient pair. It does not double the reported scalar screen.

## Linear algebra and aliasing

Supported symmetric-iterate products reject outputs overlapping the symmetric iterate or a
numerical operand **before writing**. Zero iterate coefficients are
skipped even against `NaN` or `Inf` in another factor. The exact
product and three-factor numerical guarantees apply only to their
documented Julia dispatch domains.

## Internal implementation and evidence

The current `SymmetricSparseIterate` implementation shares packed active-list and reverse-map primitives
with `SparseIterate`, plus its matrix dimension `p`.
Public and private matrix-view types have different write permissions;
the internal `Base` alias integration hooks are **not** public
extension APIs. Test active zeros, mirrored pair writes, coordinate
mapping, blocked view writes, conversions, aliases and multiplicity.

Relevant test suites include `test/symmetric_iterate_interface.jl`,
`test/symmetric_iterate_indexing.jl`, `test/matrixview.jl`,
`test/real_iterates.jl`, `test/sparse_iterate_basics.jl`,
`test/symmetric_products.jl`, `test/core_functions.jl` and
`test/minimal_math.jl`. Historical acceptance evidence is kept
in the archived snapshot, not treated as a current test requirement.
