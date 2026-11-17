# Writing your own function

*Assumes: Julia structs and multiple dispatch · You will learn: how to implement mathematical function interfaces and check their inputs.*

Extend mathematical methods without reaching into [iterate](glossary.md#Iterate) fields. Choose the
smallest interface that describes the operations you can implement.

## Any mathematical function

Subtype `AbstractFunction`, define the callable value `(f::MyFunction)(x)` and
declare the appropriate type traits. Generic defaults are deliberately limited:

| Capability you implement | Required hooks | Generic result |
|:--|:--|:--|
| Smooth function | `value_and_gradient!(out,f,x)` and `issmooth` | `gradient!` delegates to it; allocating gradient/`value_and_gradient` use it |
| Proximable function | `prox!(out,f,x,γ)` and `isproximable` | allocating prox; default `value_and_prox!` computes prox then f(out) |
| Entry-separable function | Subtype `SeparableFunction`, `value_entry` and scalar `prox_block` | Array/iterate values, `prox!`, `value_and_prox!`; separability/proximability traits true |
| Compact block-only oracle | `prox_block!(out,f,u,k,γ)` | Allocating vector `prox_block` only |

`isconvex` defaults false and must be declared when justified. There is no
automatic gradient from a value, scalar prox from a full-array prox, `kkt_violation` method,
restriction, Lipschitz bound or full-array lifting of a compact block oracle.
Generic methods accepting tuples do not imply that the underlying function
supports tuple inputs; it still needs methods for the requested argument types. The generic `gradient!` default depends on
`value_and_gradient!`, so implement that hook if relying on the default.

An overriding public operation owns real-type, shape, parameter-extent and [aliasing](glossary.md#Aliasing)
preflight; parameter symmetry is a model precondition that the caller or solver validates. Destination comes first. Return the identical output for `gradient!`/
`prox!`/`prox_block!`, a scalar input value for `value_and_gradient!`, and a scalar
stored-output value for `value_and_prox!`. Proximal output may be exactly the
input; gradient output must be independent. Do not write parameters, input
buffers or solver storage while constructing a pure numerical result.

## Floating-point precision for extensions

For the supported CPU numerical targets, test custom functions with
homogeneous `Float64` arguments/parameters/outputs and, where supported
by the extension, homogeneous `Float32` arguments/parameters/outputs.
Custom hooks are not automatically required to handle mixed
`Float32`/`Float64` calls, and the generic interface does not supply
arbitrary-precision or exceptional-range recovery. If an extension
supports additional combinations, document their conversion and result
type rules separately.

Stable formulas and selective recovery for representable results are
encouraged. The common finite path should remain efficient; the
extension author decides which exceptional cases merit additional
arithmetic. See the [numerical contract](numerics.md).

## Separable functions

Suppose the function can be written as a sum of independent coordinate terms,

```math
f(x)=\sum_{i=1}^{p} f_i(x_i).
```

A Julia subtype of `SeparableFunction` supplies two scalar operations:
`value_entry(f, v, i)` evaluates ``f_i(v)``, while
`prox_block(f, v, i, γ)` computes the scalar proximal point of
``\gamma f_i`` at ``v``. Here i is the **linear entry index** of the underlying
argument, used to index parameters. Generic kernels visit arrays, `SparseIterate`,
`BlockIterate` and symmetric iterates. Declare `check_parameters(f,x)` to reject
wrong parameter extent before generic writes. Use `prod(size(x))` rather than
`length(x)` when accepting symmetric iterates.

Declare every per-entry array in `entry_parameters(f)` as a tuple of
`Symbol=>value` pairs; declare `()` if none. A custom `SeparableFunction`'s
default is `nothing`, which `check_symmetric_parameters` rejects. On a symmetric
iterate your per-entry arrays must be exactly symmetric: the generic kernels
assume it and compare no mirrored entries, so asymmetric arrays are an invalid
model with no guaranteed numerical meaning. Call
`check_symmetric_parameters(f,x)` once per model, at the solve or workspace
boundary and after any change to the arrays. Within ProximalBase it is the only reader of
`entry_parameters`; the extent of your arrays is the job of `check_parameters`.
See [symmetric parameters](symmetric_parameters.md).

`iszeropreserving(f)=true` is a mathematical promise that ``f_i(0)=0`` and
``prox_{γf_i}(0)=0`` for **all positive finite γ**. Only declare it when valid
for every parameter value the instance permits. It enables sparse traversal;
it does not provide an oracle or license arbitrary NaN suppression. Symmetric
values count off-diagonal entry terms twice, while coordinate vectors count
one independent value. Raw scalar calls do not check symmetric-iterate symmetry; validate
the original function before using them.

For example, the [WeightedSquare implementation](examples/custom_function.md)
represents

```math
f(x)=\frac12\sum_{i=1}^{p}w_i x_i^2,\qquad w_i\geq0.
```

Its scalar functions are

```math
f_i(v)=\frac{w_i v^2}{2},\qquad
\operatorname{prox}_{\gamma f_i}(v)=\frac{v}{1+\gamma w_i}.
```

The Julia methods correspond directly to these formulas:

```julia
ProximalBase.value_entry(f::WeightedSquare, v, i) = f.w[i] * v^2 / 2
ProximalBase.prox_block(f::WeightedSquare, v, i, γ) = v / (1 + γ * f.w[i])
```

The complete implementation owns its weights, checks finite nonnegative
parameters, defines both methods and the required parameter metadata, and
verifies dense, sparse and symmetric use and the explicit rejection of asymmetric weights.
Its definition and checks are included below from that executable source.

```@eval
using Markdown, ProximalBase
source = read(joinpath(pkgdir(ProximalBase), "examples", "custom_function.jl"), String)
Markdown.parse("```julia\n" * source * "\n```")
```

This example deliberately leaves `issmooth=false`; being a mathematical
quadratic does not synthesize a gradient method. Its simple scalar arithmetic
does not promise the built-in ridge kernel's extreme-range recovery.
To make it smooth, implement/check a full fused gradient and define the trait;
to support KKT/restrict/`lipschitz`, define and validate those independently.

### Optimality scores of your own function

`kkt_violation` has a three-argument form, the screen of a coordinate assumed zero,
a four-argument form `kkt_violation(f,v,x,k)` that evaluates the actual coordinate
`k` of `x`, and a full-point form `kkt_violation(f,v,x)`. For a function defined
outside the package, they behave as follows.

* With only the three-argument method
  `kkt_violation(f::MyFunction,v::AbstractArray,i::Integer)`, the four-argument form
  returns that method's value at a zero coordinate (on a `BlockIterate`, at a group
  whose entries are all zero, the maximum of the screens of its entries) and throws an
  `ArgumentError` at a nonzero coordinate or group. The error names the method to
  define. This holds on arrays, on a `BlockIterate` and on a `SymmetricSparseIterate`.
* Without the three-argument method, the four-argument form throws a `MethodError`.
* There is no generic full-point fallback: `kkt_violation(f,v,x)` is a `MethodError`
  until the function defines it.
* To support nonzero points, define the methods of the table. A method that you
  define is called as defined, and it owns the validation of its arguments.

| Form | Signature |
|:--|:--|
| Array point | `kkt_violation(f::MyFunction,v::AbstractArray,x::AbstractArray,k::Integer)` |
| `BlockIterate` | `kkt_violation(f::MyFunction,v::AbstractArray,x::BlockIterate,k::Integer)` |
| `SymmetricSparseIterate` | `kkt_violation(f::MyFunction,v::AbstractArray,x::SymmetricSparseIterate,k::Integer)` |
| Full point | `kkt_violation(f::MyFunction,v::AbstractArray,x::AbstractArray)` |

Type `f`, `v` and `x` as in the table. A `BlockIterate` is an `AbstractVector`, so a
method with `x::AbstractArray` and one with `x::BlockIterate` are both needed for a
function that is called on a `BlockIterate`: with only the first, a call with a
`BlockIterate` is ambiguous with the method of the package. A method that leaves `v`, `x`
or `k` untyped is ambiguous with the methods of the package for array points.

A `SymmetricSparseIterate` is not an `AbstractArray`, so a method with `x::AbstractArray`
does not apply to it and is not ambiguous with the method of the package for it. A call
with a symmetric iterate then follows the rules above: the value of the three-argument
method at a zero coordinate and an `ArgumentError` at a nonzero one, or a `MethodError`
without that method. Symmetric support needs a method typed `x::SymmetricSparseIterate`.

```@example custom_kkt
using ProximalBase
import ProximalBase: kkt_violation

struct UnitL1 <: AbstractFunction end        # the penalty ‖x‖₁, with only its zero screen
kkt_violation(::UnitL1, v::AbstractArray, i::Integer) = abs(v[i]) - 1

f = UnitL1()
x = [0.0, 0.0, 2.0]
v = [0.4, 1.5, -1.0]
@assert kkt_violation(f, v, x, 2) == kkt_violation(f, v, 2)    # a zero coordinate
refused = try kkt_violation(f, v, x, 3); false catch e; e isa ArgumentError end
@assert refused                                                # a nonzero coordinate
```

Its own methods give the score of the nonzero coordinate and the full-point value:

```@example custom_kkt
score(vk, xk) = iszero(xk) ? abs(vk) - 1 : abs(vk + sign(xk))
kkt_violation(::UnitL1, v::AbstractArray, x::AbstractArray, k::Integer) = score(v[k], x[k])
kkt_violation(::UnitL1, v::AbstractArray, x::AbstractArray) =
    max(0.0, maximum(score(v[k], x[k]) for k in eachindex(x); init = -Inf))
@assert kkt_violation(f, v, x, 1) ≈ -0.6
@assert kkt_violation(f, v, x, 3) ≈ 0.0
@assert kkt_violation(f, v, x) ≈ 0.5
```

## Compact-vector block extensions

A block function may act on a compact vector ``u`` representing one group.
For the [BlockSquare example](examples/block_prox_extension.md), group ``k``
has the mathematical form

```math
h_k(u)=\frac{a_k}{2}\|u\|_2^2,\qquad
\operatorname{prox}_{\gamma h_k}(u)=\frac{u}{1+\gamma a_k}.
```

Its Julia implementation stores a parameter `a[k]` for each group and
implements `prox_block!(out, f, u, k, γ)` by writing
`u ./ (1 + γ * f.a[k])` to `out`.

More generally, a custom block method computes ``\operatorname{prox}_{\gamma h_k}(u)``
from a numerical trial `u` and group index `k`. Buffers are in that group's **ordered compact index space**,
not the full vector. Validate one-based real vectors, group bounds, exact
length, parameters and overlap before writes. Support `out===u`; reject other
overlap. Return out. The allocating default creates a plain floating Vector
and calls your hook; it does not check your parameters or lift the hook into
full-array prox.

Output must be independent of iterate storage and solver/model caches. The
caller matches the partition, computes the trial, invokes the hook in scratch,
then writes `copyto!(blockview(x,k),out)` and updates its residual. A zero result
can leave an absent block inactive because the trial was never written there.
Parameter symmetry and independent-coordinate geometry are not automatically
defined for an arbitrary block decomposition on a symmetric iterate.

The complete BlockSquare example checks allocating, exact in-place and
rejected-overlap calls. Implementing only this block method does not by
itself define a full-array proximal operator or its associated traits.

```@eval
using Markdown, ProximalBase
source = read(joinpath(pkgdir(ProximalBase), "examples", "block_prox_extension.jl"), String)
Markdown.parse("```julia\n" * source * "\n```")
```

## Validation checklist

Verify independent formulas, source/destination order, identical-output return,
FP32/FP64 behavior you intend to support, stored zeros, wrong extent and bad
parameters, exact in-place prox, rejected overlap and parameter metadata.
Use arbitrary ordered/noncontiguous groups to catch accidental global indexing
of compact buffers. For symmetric support, test multiplicity, exact pair
checks and unchanged values/activity after preflight errors. The
[developer workflow](developer/change_process.md) describes where regression
tests and executable examples belong.
