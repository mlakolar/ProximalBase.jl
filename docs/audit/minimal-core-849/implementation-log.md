# Implementation record

## Phase B: extract and load

The approved declaration inventory drove exact frozen byte-segment extraction.
Storage source in `sparse_iterate.jl` and `block_iterate.jl` is unchanged at this
stage. Existing retained formulas, ordinary kernels, exceptional-scale kernels
and caches are copied without numerical rewrites. Shared unions and literal
method-generation loops are narrowed to the retained families as planned.
`_is_device_array(::AbstractArray)=false` moves from lapack.jl into utils.
The generic allocating vector `prox_block` remains with the custom `prox_block!`
hook, while built-in group kernels are inactive.

Load/independent validation exposed dispatch requirements that a lexical graph
cannot resolve. The policy and regenerated audit now include three shared
`_checkwrite` declarations from frozen constrained.jl:212–221 (generic five-argument
fallback and Scaled scalar/array preflights), plus both tuple `_check_own_arrays`
arms. These remain required by retained symmetric parameter/proximal checks;
no constrained function is retained. Explicit exported `opnorm2` and its helper
closure are also selected. The final selection at this stage is 407 storage
plus 907 extracted declarations; 4,162 are archived. The unchanged graph contains
5,476 frozen declarations across 52 files.

`PrecisionError` is not exported in the minimal module: its frozen docstring
states that only the deferred GLMLoss quadratic-model oracles raise it. Its
exception type and the second-order-only Float16 role helper remain archived.
Supported numerical validation targets Float32 and Float64.

The new precompile workload exercises all retained concrete functions and all
three iterates in Float32/64. The reduced module loads under Julia 1.13.3 with
PrecompileTools 1.4.0. A 52-assertion smoke checks all public bindings resolve,
`Test.detect_ambiguities(ProximalBase; recursive=true)` is empty, gradients and
proximal/value agreement for each retained function, and storage/products.
A focused symmetric NormL1 value/proximal reference passes for both float types.
Independent broader validation and final-source review are separate gates.

Deferred files remain on disk but are no longer included in Phase B. Phase C
will remove them from the active tree, with the frozen content preserved at the
baseline commit and archival refs. No ProximalSolvers changes are made.
