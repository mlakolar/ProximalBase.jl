# Array and precision conventions

## Shapes and destination-first mutation

Use real arrays in each function's supported shape. Vector and matrix iterates
participate in array dispatch; a symmetric iterate requires an explicit method.
Gradients and proximal points must match their input shape. Per-entry weights
and array steps can use equal entry count in linear order, while centers,
coefficient writers and matrix products can impose matching shapes/axes.
The [function pages](smooth.md) give these distinctions for each family.

Mutating calls write the first argument. They can allocate scratch, stage an
alias, or build a decomposition; `!` is not a no-allocation promise. Allocating
gradients use dense buffers. Allocating proximal points use floating `similar`
arrays/iterates, except symmetric array wrappers return dense matrices and
supported symmetric-iterate calls return new symmetric iterates.

## Precision

The supported CPU configurations use the **same floating-point type**
throughout a problem: data, iterates, function parameters, steps and
floating destinations are all `Float64` or all `Float32`. This is
the guaranteed numerical contract, not a requirement that every other
combination be rejected by dispatch.

For those homogeneous calls, `f(x)` returns `float(eltype(x))` for
the supported array functions. Allocating oracles return their documented
representations in the working precision, and mutating oracles write into
the provided destination. Fused gradient calls return the value at the
input; fused proximal calls return the value at their stored output.

Use explicit conversions when creating a `Float32` problem. Julia
floating-point literals such as `0.5` are `Float64`; use
`Float32(0.5)` or `T(0.5)` for a working type `T`:

```@example precision_usage
using ProximalBase
for T in (Float32, Float64)
    x = T[1, -2]
    f = SqrNormL2(T(0.5))
    grad = zeros(T, length(x))
    value = value_and_gradient!(grad, f, x)
    point = prox(f, x, T(0.5))
    @assert value isa T
    @assert grad == T[0.5, -1.0]
    @assert eltype(grad) == T && eltype(point) == T
end
true
```

Existing kernels can use promoted scratch or accept mixed floating-point
arguments. These are implementation details, **not** a general
mixed-precision guarantee for accuracy, returned types or performance.
Independent high-precision reference calculations can still be used in
tests; `BigFloat` is not a supported production target.

The [numerical contract](numerics.md) describes selective
overflow/underflow recovery, non-finite values and aliasing. Integer
destinations may fail conversion, and custom functions define their own
arithmetic and recovery unless they explicitly adopt these conventions.

## Ownership and aliases

| Object or call | Ownership rule |
|:--|:--|
| Built-in function array parameters/data | Usually retained by reference; copy at construction if you need independence |
| `LeastSquaresLoss` / `QuadraticFunction` | Own scratch and lazy decomposition caches; no concurrent reuse without synchronization |
| `Scaled` | Retains the inner object, scratch and parameters |
| Iterate `copy` | Independent mutable values and active state; `BlockIterate` layout shared, immutable in use |
| Coordinate / block / matrix views | Live aliases; only `matrixview` is read-only |
| Active handles | Live old-extent slot handles; reacquire after structural change |

`prox!(x,f,x,γ)` supports exact in-place operation. Other overlapping
input/output views are unsupported and generally undetected. Copy the source
first. Built-in gradients reject input/output identity; other shared-memory
views also need independent scratch. Do not use function data or parameters
as writable output unless the specific method explicitly permits it.
`mul!` outputs should be independent of operands; symmetric-iterate kernels reject aliases.

After same-shape design/response/quadratic-matrix entry changes, invalidate
proximal decomposition caches with `reset_cache!(outermost_function)`.
Values/gradients read data directly. Reconstruct for dimension changes.
No reset is needed for a weight-only `NormL1` object, but it must retain valid
weights. Neither reset nor parameter validation copies caller data.

## Predictable failures and partial writes

Built-in oracles preflight predictable real-type, shape and parameter errors.
Symmetric pair checks occur before symmetric-iterate writes on implemented public boundaries.
A failing user element access, conversion or numerical computation may leave
partial output. A raw hook does not gain checks from a trait declaration.

Direct array-iterate bulk writes reject known complex source types, even with
zero imaginary parts, before writing. Broadcasts check leaves and inferred
results; uncertain non-real result types are staged before writes. Ordinary
views, indexed/offset writes and abstractly typed sources retain elementwise
conversion behavior and can fail partway. A same-kind state copy stages
fallible stored-value conversion; a symmetric iterate matrix import checks shape/symmetry/
aliases first but can fail converting after clearing old state.

For atomic conversion failure, prepare an independent correctly typed result
and validate it before the final state copy. See
[aliasing and failure contracts](developer/views_aliasing.md).

## Sparse products and ordinary views

Supported sparse products skip zero iterate coefficients, even against NaN/Inf.
A coordinate view is an ordinary vector: it can traverse all coordinates and
form IEEE `0*Inf`. Neither sparse work nor a sparse product's exceptional
semantics is implied by `AbstractVector` dispatch. Consult
[the product table](iterates.md#Supported-products) before choosing a kernel.

The supported build and examples use CPU Julia only. No device backend or
GPU extension is part of this minimal-core manual.
