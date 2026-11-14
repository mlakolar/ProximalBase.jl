# Symmetric parameters

*Assumes: symmetric matrices · You will learn: when per-entry parameters must match across mirrored matrix positions.*

A symmetric [iterate](glossary.md#Iterate) stores one independent variable for a mirrored matrix pair.

A symmetric iterate stores one variable for `(r,c)` and `(c,r)`. Per-entry
parameters must give the same mathematical function at both positions.
ProximalBase symmetrizes supported computed matrix arguments and gradients;
it requires your parameter arrays to be **exactly symmetric**.

Symmetric parameters are a **precondition of the model**, not something every
oracle verifies. Three parties share the work:

| Who | Responsibility |
|:--|:--|
| **Caller or solver** | Supplies exactly symmetric parameters and validates them with `check_symmetric_parameters` at the solve or workspace boundary, again after replacing a function or mutating a parameter array, and for any weights it creates later |
| **Numerical oracles** (`f(x)`, `prox!`, `value_and_prox!`, `gradient!`, `value_and_gradient!`, four-argument `kkt_violation`) | Assume the model is symmetric and compare no mirrored parameter entries. They still check shapes, parameter extents, element types, aliases and supported operations before writing |
| **`check_symmetric_parameters(f,x)`** | Compares every declared mirrored pair on demand, including pairs the iterate does not store |

An oracle given asymmetric parameters has **no guaranteed numerical meaning**:
its result may disagree with direct evaluation of the full-matrix function
(for example, a proximal operator reads the lower triangle of the weights while
the value sums both positions). No oracle averages or repairs a raw asymmetric array, and the explicit
check does not modify what it rejects. ProximalBase records no validation
certificate: a successful check says nothing about later mutation.

## Which parameters must be symmetric

| Function | Per-entry array |
|:--|:--|
| `NormL1` | weights w |
| `SqrNormL2` | center y |
| `LinearFunction` | coefficients c |
| `Scaled` | recurses through its inner function |
| Custom `SeparableFunction` | every declared per-entry array |

Parameters refer to full matrix positions, usually a p×p array. An entrywise
weight vector of length p² can be paired in column-major order where the wrapped function
permits it; center writers can impose the full matrix shape/axes, and the
coefficients of `LinearFunction` must be a p×p matrix for every oracle.
Use matching matrices for centers and coefficients. Design matrices, response
data, block-indexed parameters and scalar levels are not per-entry matrix
parameters merely because they are arrays or numbers.

`ProximalBase.entry_parameters(f)` returns a tuple such as `(:w=>f.w,)`.
`nothing`/scalar numbers inside that tuple are skipped. A custom separable
function must explicitly declare the tuple or `()` when it has no arrays;
its default `nothing` means undeclared and `check_symmetric_parameters` throws
`ArgumentError`. Pair names must be Symbols and values must be scalar,
`nothing` or arrays. Only the explicit checker reads the declaration; the
oracles use `check_parameters` for the extent of a custom function's arrays.

## Exact equality and the explicit check

Equality has no tolerance: paired NaNs match, `0.0` matches `-0.0`, and equal
integers/floats match. Constructors separately enforce real types and parameter
domains; declaring a parameter does not make a negative/NaN weight valid.

| Call | Compares mirrored parameter entries? |
|:--|:--|
| `check_symmetric_parameters(f,x)` on a symmetric iterate | Yes: every declared pair, recursively through `Scaled` |
| `f(x)`, `prox!`, `value_and_prox!`, `gradient!`, `value_and_gradient!` | No. They read the entries the mathematics needs, such as both positions of each stored off-diagonal coordinate in a value |
| Four-argument `kkt_violation` on a symmetric iterate | No. It checks the parameter extent, the coordinate and the gradient |
| Raw `value_entry`, `prox_block`, three-argument KKT, `restrict` | No: there is no symmetric iterate |

`check_symmetric_parameters` is not a substitute for the ordinary shape checks:
it skips the extent check for arrays that are symmetric by type, and a passing
check does not make a function's parameters fit an argument. Keep calling
`check_parameters` and the usual shape validation at the boundary.

Arrays symmetric by construction, e.g. `Symmetric(W,:L)`, pass the explicit
check without a scan because the wrapper defines the equal mirrored values.

```@example symmetric_parameter_checks
using ProximalBase, LinearAlgebra
x = SymmetricSparseIterate(3)
x[2,1] = 4
W = [1.0 2 1; 2 1 1; 1 1 1]
f = NormL1(1.0,W)
@assert ProximalBase.check_symmetric_parameters(f,x)
@assert prox(f,x)[2,1] == 2
Wbad = copy(W)
Wbad[1,3] = 3  # pair (3,1) is not stored in x
try
    ProximalBase.check_symmetric_parameters(NormL1(1.0,Wbad),x)
    error("expected full pair validation to fail")
catch e
    @assert e isa ArgumentError
end
@assert Wbad[3,1] == 1 && Wbad[1,3] == 3  # the checker does not repair what it rejects
@assert ProximalBase.check_symmetric_parameters(NormL1(1.0,Symmetric(Wbad,:L)),x)
Matrix(prox(NormL1(1.0,Symmetric(Wbad,:L)),x))
```

Selecting `Symmetric(Wbad,:L)` keeps the lower values. Averaging the triangles
changes the model; choose the intended definition explicitly. Do not average
labels or other nonnumerical metadata. Type, shape and parameter-extent failures
at built-in symmetric-iterate boundaries are checked before predictable writes.

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
per-entry coefficient/center symmetry, which is a precondition on the model.
