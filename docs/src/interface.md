# Mathematical interface

A function is callable: `f(x)`. Declare its type's `issmooth`, `isproximable`,
`isseparable` and `isconvex` traits. Smooth functions provide `gradient!` and
`value_and_gradient!`; proximable functions provide `prox!` and
`value_and_prox!`. Allocating `gradient`, `value_and_gradient` and `prox` use
these hooks. Mutating calls put the destination first.

The convention is

```math
\operatorname{prox}_{\gamma f}(x)=\arg\min_z f(z)+\|z-x\|^2/(2\gamma).
```

A `SeparableFunction` implements `value_entry(f,v,i)` and scalar
`prox_block(f,v,i,γ)`. Generic kernels supply its array values and proximal
operators. `iszeropreserving` permits sparse kernels to skip inactive zeros.
`check_parameters` validates parameter shapes. `entry_parameters` declares
per-entry arrays for exact symmetric validation.

`prox_block!(out,f,u,k,γ)` is an extension hook for a compact vector block.
The generic allocating `prox_block(f,u,k,γ)` creates a fresh float vector and
forwards to it. Built-in NormL1 has a scalar block oracle and array `prox!`;
compact vector block semantics are defined by the custom function.

`kkt_violation` computes scalar-entry violations; on BlockIterate a scalar
penalty uses the maximum violation over that block's ordered index list.
`restrict` restricts retained per-entry parameters. `lipschitz` supplies a
bound where available. `opnorm2` is an iterative estimate and may miss a leading
eigenspace; step selection needs backtracking or an independent bound.

`reset_cache!` invalidates least-squares SVD or quadratic eigendecomposition
caches after referenced data entries change. It forwards through Scaled.
Array dimensions must stay unchanged; reconstruct a function for new shapes.
