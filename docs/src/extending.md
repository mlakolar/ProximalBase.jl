# Writing your own function

*Assumes: Julia structs and multiple dispatch · You will learn: how to implement mathematical function interfaces and check their inputs.*

Extend mathematical methods without reaching into [iterate](glossary.md#Iterate) fields. Choose the
smallest interface that describes the operations you can implement.

The `AbstractFunction` extension rules below apply to array functions.
For observation-level prediction losses, implement the separate
[`ProximalBase.ScalarLoss` interface](scalar_losses.md).

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
automatic gradient from a value, scalar prox from a full-array prox, KKT screen,
restriction, Lipschitz bound or full-array lifting of a compact block oracle.
Generic methods accepting tuples do not imply that the underlying function
supports tuple inputs; it still needs methods for the requested argument types. The generic `gradient!` default depends on
`value_and_gradient!`, so implement that hook if relying on the default.

An overriding public operation owns real-type, shape, parameter, [aliasing](glossary.md#Aliasing) and symmetric
preflight. Destination comes first. Return the identical output for `gradient!`/
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
default is `nothing`, which rejects symmetric use. Full symmetric operations
validate metadata, shape and exact mirrored pairs. Overriding a public oracle
replaces that generic boundary: call `check_symmetric_parameters(f,x)` yourself
when accepting a symmetric iterate.

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
verifies dense, sparse and symmetric use and failure-before-write behavior.
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
