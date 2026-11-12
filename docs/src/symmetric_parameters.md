# Symmetric parameters

*Assumes: symmetric matrices · You will learn: when per-entry parameters must match across mirrored matrix positions.*

A symmetric [iterate](glossary.md#Iterate) stores one independent variable for a mirrored matrix pair.

A symmetric iterate stores one variable for `(r,c)` and `(c,r)`. Per-entry
parameters must give the same mathematical function at both positions.
ProximalBase symmetrizes supported computed matrix arguments and gradients;
it requires your parameter arrays to be **exactly symmetric**.

## Which parameters are checked

| Function | Per-entry array |
|:--|:--|
| `NormL1` | weights w |
| `SqrNormL2` | center y |
| `LinearFunction` | coefficients c |
| `Scaled` | recurses through its inner function |
| Custom `SeparableFunction` | every declared per-entry array |

Parameters refer to full matrix positions, usually a p×p array. An entrywise
weight vector of length p² can be paired in column-major order where the wrapped function
permits it; center/coefficient writers can impose the full matrix shape/axes.
Use matching matrices for centers and coefficients. Design matrices, response
data, block-indexed parameters and scalar levels are not per-entry matrix
parameters merely because they are arrays or numbers.

`ProximalBase.entry_parameters(f)` returns a tuple such as `(:w=>f.w,)`.
`nothing`/scalar numbers inside that tuple are skipped. A custom separable
function must explicitly declare the tuple or `()` when it has no arrays;
its default `nothing` means undeclared and symmetric use throws
`ArgumentError`. Pair names must be Symbols and values must be scalar,
`nothing` or arrays. Public-oracle overrides must call the validation hook
themselves; metadata alone does not execute a check.

## Exact equality and checked domains

Equality has no tolerance: paired NaNs match, `0.0` matches `-0.0`, and equal
integers/floats match. Constructors separately enforce real types and parameter
domains; declaring a parameter does not make a negative/NaN weight valid.

| Operation involving a symmetric iterate | Pairs checked |
|:--|:--|
| `f(x)` | Usually all; built-in stored-value paths such as `NormL1` compare stored pairs only |
| `prox!`, `value_and_prox!` with symmetric-iterate destination | All, except built-in zero-preserving stored paths with both arguments symmetric iterates compare pairs stored in input or output |
| Prox with dense destination and symmetric-iterate point | All declared pairs |
| Gradient/fused gradient, symmetric-iterate point or output | All declared pairs, when that method exists |
| Four-argument `kkt_violation` on a symmetric iterate | Chosen coordinate pair only |
| `check_symmetric_parameters(f,x)` | All declared pairs, recursively through `Scaled` |
| Raw `value_entry`, `prox_block`, three-argument KKT, `restrict` | No symmetric iterate; no pair check |

A successful stored-only call does not certify unvisited parameter pairs.
Custom separable functions use the conservative full pair check at generic
public boundaries, even when their numerical kernel skips inactive zeros.
There is no persistent validation cache; later parameter mutation can invalidate
a prior result. Arrays symmetric by construction, e.g. `Symmetric(W,:L)`,
need no pair scan because the wrapper defines the equal mirrored values.

```@example symmetric_parameter_checks
using ProximalBase, LinearAlgebra
x = SymmetricSparseIterate(3)
x[2,1] = 4
W = [1.0 2 1; 2 1 1; 1 1 1]
f = NormL1(1.0,W)
@assert ProximalBase.check_symmetric_parameters(f,x)
@assert prox(f,x)[2,1] == 2
Wbad = copy(W)
Wbad[1,3] = 3  # pair (3,1) is not stored yet
@assert prox(NormL1(1.0,Wbad),x)[2,1] == 2
try
    ProximalBase.check_symmetric_parameters(NormL1(1.0,Wbad),x)
    error("expected full pair validation to fail")
catch e
    @assert e isa ArgumentError
end
x[3,1] = 4
before = copy(x)
try
    prox!(x,NormL1(1.0,Wbad),x)
    error("expected stored-pair validation to fail")
catch e
    @assert e isa ArgumentError
end
@assert x == before && collect(activecoordinates(x)) == collect(activecoordinates(before))
Matrix(prox(NormL1(1.0,Symmetric(Wbad,:L)),x))
```

Selecting `Symmetric(Wbad,:L)` keeps the lower values. Averaging the triangles
changes the model; choose the intended definition explicitly. Do not average
labels or other nonnumerical metadata. Type/shape/parameter failures at built-in
symmetric-iterate boundaries are checked before predictable writes.

## Proximal arguments and computed gradients

For symmetric Z and any V,

```math
\|Z-V\|_F^2=\|Z-(V+V^\top)/2\|_F^2+\|(V-V^\top)/2\|_F^2.
```

Therefore a supported prox **into a symmetric iterate** at a nonsymmetric matrix reads
its symmetric part. Equal pairs are preserved; finite sums are averaged before
halving to retain subnormal digits, with halves added when the sum overflows.
Opposite infinities give NaN. A `Symmetric(V,:L)` argument deliberately selects
one triangle instead.

```@example symmetric_argument
using ProximalBase, LinearAlgebra
V = [0.0 1;3 0]
out = SymmetricSparseIterate(2)
prox!(out,NormL1(0.5),V,1.0)
@assert out[2,1] == 1.5
prox!(out,NormL1(0.5),Symmetric(V,:L),1.0)
@assert out[2,1] == 2.5
Matrix(out)
```

Symmetric-iterate gradients of supported smooth functions store the symmetric part.
Four-argument KKT uses the symmetric gradient pair, with no independent-variable
factor of two. An opposing `+Inf/-Inf` gradient pair has no symmetric part and
throws `ArgumentError` at the chosen coordinate. The routine does not scan
unrelated gradient pairs. Do not confuse this computed-gradient policy with
per-entry coefficient/center symmetry, which remains a caller requirement.
