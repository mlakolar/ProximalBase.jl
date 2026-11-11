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

## Phase C: shrink active tree

Removed the 36 inactive Julia source files and both deferred extensions;
`removed-source.tsv` records every path, frozen blob and rationale. The module
now contains 16 source files. All removed content remains in the frozen full
commit, archive branches and annotated preservation tag. This commit changes
no retained numerical code. Historical tests/docs/benchmark catalogs are
migrated or removed separately so runnable resources match the reduced surface.

## Retained dispatch corrections during independent validation

Restored the two exact generic BlockIterate KKT dispatches from frozen
interface.jl. They reduce a scalar penalty's violation over the block's index
list; removing them would silently interpret the block number as an entry
index through the AbstractArray fallback. Reversed/noncontiguous groups,
weighted NormL1 and Scaled are protected by direct maximum-formula tests.
No built-in group penalty is restored.

Restored four exact `reset_cache!` methods: AbstractFunction fallback,
LeastSquaresLoss, QuadraticFunction and Scaled forwarding. The retained losses
store data by reference, so this hook remains necessary after data mutations
once a decomposition has been cached. Regression tests build the cache, mutate
referenced X/Y/A, reset it and compare a fresh function. The minimal caches.jl
contains only these methods. The unused `_ridge_weights` helper, whose only
consumers were deferred families, was removed. The regenerated final selection
is 407 storage declarations, 912 extracted declarations and 4,157 archived
ones; the frozen inventory still contains 5,476 declarations.

## Supported resources and documentation

The extracted files have coherent names: functions/scalar_loss.jl, scaled.jl
and array_steps.jl. There are 17 active source files totaling 333,106 bytes,
79.8% fewer bytes than the 52-file, 1,649,968-byte frozen tree. Numerical storage
ASTs remain identical after documentation-only link cleanup.

Historical tests, docs/design pages, examples, scripts, workflows and benchmark
catalogs that require retired features were removed from the active branch;
`retired-resources.txt` lists their original paths. Their content remains in the
frozen baseline and archival refs. The minimal runner enforces an explicit
18-test inventory and executes each file in a fresh Julia process, preserving
the former runner's fixture/dispatch isolation. Local TestUtils contains only
allocation measurement and the frozen generic storage-bookkeeping checks.
Ridge numerical reference sections are preserved; only constrained-ridge tests
are excluded. Full/standard/thorough tier names currently include the same
supported quick inventory.

The old symmetric display allocation test's 64-byte growth allowance also
fails against the exact frozen source under Julia 1.13.3. The reproduction log
in verification/baseline-display.log reports 208-byte growth for otherwise
constant-sized displays. The minimal test permits a constant 512-byte
formatting/allocator allowance and keeps the strict 65,536-byte total cap;
no storage or display implementation changed, and no failure is skipped.

Documentation now builds with `checkdocs=:exports`, executes the lasso
mathematical example and custom separable-function example, and contains only
the supported API/extension/storage/numerical contracts. The CI workflow uses
current Julia 1.x on Linux, preserves exact required job names `PR required
jobs` and `PR merge gate`, and requires bounds-checked Pkg.test, frozen-baseline
benchmark reference smoke and documentation/extension-example build.

Independent critic verification preserved under verification/: 10,080 scalar
and 3,360 array-step exceptional configurations give exact canonical output or
exception-kind agreement with the frozen source; 80 custom Scaled extension
assertions passed. Expanded independent mathematical references pass 12,832
baseline/candidate assertions, including weighted block KKT. The package's
current-function half (6,416 assertions) is in test/minimal_math.jl.

## Local validation of the packaged core

Julia 1.13.3, Linux, bounds checks enabled: all 18 declared files pass through
Pkg.test (2m44.7s runner time). Documentation builds with exported-API checking
and both executable examples; the frozen-baseline benchmark smoke passes all
102 probes. Passing logs are preserved under verification. Statistics remains
a test-only dependency because the unchanged sparse/view suite tests mean.
Source digest (ordered path:SHA256 list): `5e173f646116a9b70d0a0dbc22eb3b209a92b253241febecb9aa69acaa5f7c18`; source commit
`d11d70893697579aef34bddeed900c6144f38df8`. Current file inventory is in
current-source-inventory.tsv. Final performance reproduction, final-SHA remote
CI and independent final-source review remain integration gates.
