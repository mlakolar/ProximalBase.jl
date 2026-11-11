# Pure block proximal contract: retained minimal core

**Status:** current description of retained mathematical extension hooks after
#850, adapted from the #844 design. No built-in group-penalty family remains.

## Scalar entries

prox_block(f,v,i,γ) computes the entry proximal point at an already constructed
scalar trial v. The global entry index i selects mathematical parameters.
Retained scalar methods cover NormL1, SqrNormL2, LinearFunction, ZeroFunction
and supported Scaled leaves. A custom SeparableFunction implements scalar
prox_block and value_entry; generic kernels supply its full-array values/prox.
No trait synthesizes raw value_entry for other mathematically separable types.

A raw entry hook does not access an owner or check mirrored parameter pairs.
For a symmetric coordinate k, the caller uses (r,c)=tril_cartesian(p,k), then
full-entry i=(c-1)p+r and validates original parameters before the scalar call.
Its curvature/gradient must use compatible symmetric geometry.

## Compact vectors

prox_block!(out,f,u,k,γ) is a pure user extension. It writes numerical out and
returns that identical object, with u/out in the selected custom group's
compact order. The generic allocating prox_block creates a floating Vector
of length(u), then delegates to that hook. It does not perform validation for
the custom implementation or infer a group layout.

An implementation conforming to this contract validates before predictable writes:
one-based real vectors, valid group k, exact compact lengths, valid parameters
and step for its mathematical domain. It supports out===u and rejects other
buffer overlap. Failed element reads/conversions/user arithmetic may leave
partial numerical output; the author documents any narrower numerical policy.
Built-in extreme-range recovery is not granted automatically to custom code.

**Caller precondition:** out is independent of iterate storage and solver/model
caches. coordinates, activevalues and blockview all alias their owners, including
views of a plain vector parent. The hook does not infer universal ownership.
Use independent scratch or the allocating wrapper.

The caller checks its custom function's ordered groups against blockindices,
constructs a trial, computes the proximal result, then writes
copyto!(blockview(x,k),out) and updates residuals/caches. A zero numerical result
can therefore leave an absent group inactive. A separable penalty instead loops
over global member indices using scalar prox_block; BlockIterate's group ID
is not an entry index.

## Unsupported automatic combinations

A compact hook alone does not add f(x), full-array prox, KKT, restrict, traits,
Scaled compact forwarding, symmetric block geometry or storage mutation APIs.
Implement those explicitly only with valid mathematics and checked dispatch.
samegroups compares two BlockIterates; it is not a generic custom-penalty query.

There are no retained storage-aware prox_entry!/prox_coordinate!/cdprox!
compatibility shims. Historical names here identify migration only.
The complete examples/custom_function.jl and examples/block_prox_extension.jl
are executed alone and through Literate. Retained tests/core_functions.jl
checks the generic extension and fresh allocating block result; fuller
buffer validation is demonstrated/tested by the documentation extension itself.

## Historical records

The [archived #844 contract](https://github.com/mlakolar/ProximalBase.jl/blob/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design/block-prox-0.6.md)
includes removed group kernels, downstream migration/performance discussion
and then-pending review gates. It is historical evidence, not a requirement
that this documentation revive those implementations or modify ProximalSolvers.
