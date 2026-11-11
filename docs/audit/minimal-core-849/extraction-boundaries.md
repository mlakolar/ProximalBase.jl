# Phase A: extraction boundaries

Baseline: `eff21b6d7d33c622313122f46c6ee71693c101ee`, tree
`c76123a0109c0a0f683cb75085371696dffe1c8c`. This is a plan for review;
no package source has been removed or modified in Phase A.

`declarations.tsv` enumerates declarations parsed from all 52 frozen source
files, including constructors, callable methods, typed signatures, constant
bindings and literal `@eval` loop instances. `dependency-edges.tsv` records
lexical edges; `symbol-targets.tsv` resolves their names to possible declaration
IDs. This is a conservative static graph, not a claim of dispatch resolution:
an overloaded name points at all declarations bearing that name. Dynamic
interpolations inside generated bodies stay visible as templates. Runtime
loading, ambiguity checks and retained tests must close that remaining gap.

`source-segments.tsv` gives exact inclusive byte ranges of each top-level AST
form in the frozen blob. `segment_start`/`segment_end` include leading comments
and docstrings and may meet at a newline. They are review locators; extraction
must use complete declarations or the byte ranges, never cut a method at a
printed line boundary. Nested declarations share their enclosing source segment.

## Proposed retained surface

Keep `SparseIterate`, `BlockIterate` and `SymmetricSparseIterate`, their storage
views and products; `AbstractFunction`, the four basic traits,
`SeparableFunction`, `iszeropreserving`, callable values, gradients and combined
values/gradients, proximal operators and combined values/proximal operators,
scalar `prox_block`, `value_entry`, `lipschitz`, parameter validation hooks;
`ZeroFunction`, `LinearFunction`, `SqrNormL2`, `QuadraticFunction`,
`LeastSquaresLoss`, `NormL1`, `Scaled`; and qualified scalar `ScalarLoss`,
`Squared`, `deriv`, `deriv2`, `curvature`, `value_and_deriv`.

Retain `kkt_violation` and `restrict` only for retained scalar/separable
functions, since these are small coherent extension hooks already implemented
by those functions. Archive gradient-mapping solver helpers, all LMO machinery,
coordinate-Lipschitz catalogs, Hessian/Newton machinery, compatibility aliases,
all other function families and general composition. No numerical formulas are
to be rewritten to make this selection easier.

`prox_block!` remains a declared mathematical extension hook, with the generic
allocating `prox_block(f,u::AbstractVector,k,γ)` adapter from `block_prox.jl`
allocating a fresh vector and forwarding to a custom implementation. There is no
built-in compact vector/group implementation in milestone 0: scalar
`prox_block` already implements NormL1 and the retained separable functions.
Archive the remaining `block_prox.jl` implementation. Its group kernels require
`AbstractGroupPenalty`, `GroupNormL2`, `Regularize` and group decomposition;
pretending that NormL1 uses them would keep the deferred group family alive.
`BlockIterate` remains fully supported independently of group penalties.

## Exact family boundaries and required adaptations

| Frozen file | Keep/extract boundary | Archive / required narrowing |
|---|---|---|
| `sparse_iterate.jl` | All declarations, views, copy/alias/active-zero logic and products unchanged | Docstrings may remove links/examples to retired function families; no hot-path numerical edits |
| `block_iterate.jl` | All declarations unchanged, including raw-groups/integer-count constructors | Retire examples using built-in group penalties; preserve ordered/noncontiguous layouts and all view semantics. Typed penalty constructors live in deferred norms.jl and are archived |
| `functions/smooth.jl` | Start through `restrict(::QuadraticFunction, ...)` (before the `KronQuadratic` docstring); final `lipschitz` methods for ZeroFunction, LinearFunction, SqrNormL2, QuadraticFunction | Archive KronQuadratic and its final lipschitz method. Keep every ridge ordinary/scaled/wide-center helper, `_RidgeOrdinary`, `_RidgeScaled`, `_OrdinaryRidge`, quadratic value recovery and diagonal specialization within the retained segment |
| `functions/leastsquares.jl` | All declarations, including vector and multiresponse buffers, lazy SVD, arbitrary destinations, overflow/underflow recovery, deepcopy and lipschitz | `_xty_cache!` has only affine-constrained consumers; archive helper while preserving constructor/field layout (`XtY`) initially to minimize numerical churn |
| `functions/glm.jl` | Start through `value_and_deriv(::Squared, ...)` (line 81), exactly the scalar interface and Squared formulas | Archive Logistic onward, GLMLoss and weighted-predictor helper machinery. Rewrite scalar-loss docs to describe only retained extension contract |
| `functions/separable.jl` | Generic separable interface and kernels through `_restrict_entries`; NormL1 type/constructors/traits/entry/prox/KKT/restrict/dense-value methods through before NormL1Nonneg docs; `_l1_value(::NormL1,...)`, `_l1_value_sum`, narrowed `_value_dense(::NormL1{T,<:Array{T}},::Array{T})` | Remove NormL1Nonneg arm from shared weighted L1 value specialization. Generic exponent-parts helpers are retained only if reached by retained value recovery; no nonconvex payloads. `_HostArray` is a structural CPU-array alias used by Scaled `_parts_argument`, despite its current location beside NormL0; extract the alias, archive NormL0 methods |
| `calculus.jl` | Scaled section (start through `_infinite_value_exact(::Scaled,...)` before `_BlockParameter`), pruned by typed signatures; late hard-level helpers and `lipschitz(::Scaled)` below | Archive `_ShiftedArray` (Translate only), other wrapper methods and unions. Keep positive-scale exponent machinery `_ProjectionScale`, `_ScaledArray`, `_ExpScaledArray`, `_NormalScale`, `_ShiftWindow`, `_LevelProbe`; nested-chain scalar/array route, value parts/recovery, zero-scale validation, NormL1 folding and SqrNormL2 parts. Narrow `_exactly_zero_at` union to NormL1 and replace the two ElasticNet body branches with the NormL1 path (`λ=f.λ0`; no ridge test), retaining weighted/penalized loop. Expand `_fold_level` loop to its NormL1 member with unchanged body. Keep `_has_infinite_level(::AbstractFunction/NormL1/Scaled)`, `_hard_scale`, `_steps_hard`, `_hard_function(f)`, `_hard(_)`, `_hard(::ZeroFunction/NormL1/Scaled)` from lines 2267–2328. Do not retain `_PlainIndicator` or its typed method. Keep no Conjugate/spectral/late projection helpers merely because their signature mentions NormL1 or Scaled |
| `model_data.jl` | `_data_type(::Type)`, `_data_float`, LeastSquaresLoss and QuadraticFunction `_data_type` methods; narrowed `_data_type(::Type{<:Scaled{F}})`, `_data_of(f)`, `_data_of(::Nothing)`, `_data_of(::Scaled)` | Drop generated tuple/sum/operator data families and all eager references to GLMLoss/KronQuadratic/deferred wrappers |
| `metric.jl` | Array-step `prox` and ZeroFunction/SeparableFunction methods at start; Scaled array-step section from `prox!(y,s::Scaled,x,γ::AbstractArray)` through `_value_and_prox_chain!` before Tilt | Drop `_ReciprocalArray`, all group/precompose/regularize/conjugate/indicator/coordinate-Lipschitz machinery. Keep wide-factor band, `_array_fold`, `_shifted_fold`, `_infinite_alone`, overflow helpers and steps needed by Scaled |
| `symmetric_parameters.jl` | Retained-family parameter declarations; generic pair scans/domains, shape/pre-write validation, owner buffers and argument conversion; owner allocating methods; retained checked values; copy/assign/lower-coordinate helpers | Narrow entry-parameter unions to LinearFunction/NormL1/SqrNormL2 and `_inner_functions` to Scaled; `_stored_value` unions to LinearFunction and Scaled; `_OwnerSmooth` to LinearFunction/SqrNormL2; `_OwnerWrapperProx`/`_OwnerWrapperSmooth` to Scaled; final owner `prox` union to ZeroFunction; unchecked combined-prox union to ZeroFunction. Remove all deferred `@eval` members, second-order/coordinate-Lipschitz helpers, sum/precompose/constraint branches. Generic `_check_oracle_arrays` keeps the same order of checks after removing second-order-only branches |
| `utils.jl` | Transitive numerical/storage dependency closure of retained declarations, with exact methods recorded in TSV | Preserve product/rounding, Scratch/buffers/wrappers, checks, `_valtype`, generated tuple plumbing if referenced, exponent/split-dot/compensated sums, value-part/alias metadata, `_ExpSum` quadratic helpers, triangular coordinates, square residual recovery, norms and `opnorm2`/upper bounds, PrecisionError/Float16 validation. Archive simplex/select/projection thresholds, `_L1ProjectionScratch`, exponential-loss helpers and spectral input helpers unless an actual retained call requires a shared method |
| `compat.jl` | Sparse-dot type aliases and capability flags used by LinearFunction and sparse products; aliases used by unchanged iterate method signatures | Archive BLAS/Cholesky/orthogonal-factor-only aliases when unreferenced. Keep version-gated definitions before the signatures that mention them |
| `lapack.jl`, `partial.jl` | Extract only `_is_device_array(::AbstractArray)=false` from lapack.jl:206 into utils; no partial.jl declaration required | The fallback is called by unchanged sparse_iterate and retained LinearFunction/Scaled. LS and QuadraticFunction use LinearAlgebra `svd`/`eigen` rather than package partial spectral workspaces/foreign LAPACK wrappers |
| `interface.jl` | AbstractFunction/basic traits/prox/gradient/combined/allocating/default/error/shape/parameter hooks; scalar prox_block declarations; retained KKT/restrict/lipschitz | Archive LMO section, Hessian/Newton declarations, coordinate-Lipschitz declaration, gradient_mapping helpers and group KKT branch. Generic checked-step introspection and `_StepInput` remain dependencies of Scaled dispatch; do not simplify them without testing custom extension methods |
| `testing.jl`, `caches.jl`, `deprecated.jl` | None | Testing and cache dispatch catalogs assume deferred families; retire associated extension registration and public symbols rather than leaving partial stubs |
| Every other `src/*.jl` path | None | All declarations archived; full methods still recoverable at frozen commit and preservation refs |

## Include graph and cycles

Proposed order: stdlib imports and local `ldexp`; compat → utils → unchanged
sparse_iterate → unchanged block_iterate → interface → smooth → leastsquares →
scalar Squared → separable/NormL1 → Scaled → model_data → array metric →
symmetric validation → minimal precompile workload.

An unqualified name in a function body may be defined later, so the existing
utils/iterate/interface and Scaled/model_data/metric/symmetric-body cycles do
not require numerical code changes. Types in method signatures, field types,
constant unions and evaluated loop templates must already exist when included.
The proposed order defines Scratch before loss fields, iterates before owner
aliases, AbstractFunction before concrete functions, SeparableFunction before
NormL1, Scaled before typed model-data/metric/symmetric dispatches. Builtin
parameter checks are first called after the module is fully loaded. Removing
union arms and loop members removes eager references to deferred types.

## Hooks outside src

`external-references.tsv` indexes frozen test/doc/extension/benchmark/example/
script/CI line lists by declared symbols. It is a navigation index, including text
and examples; it does not assert that every lexical mention is an executable
dependency. `module-hooks.tsv` inventories includes, imports, exports, public
statements, conditional declarations and precompile macros.

* `Project.toml`: retain LinearAlgebra/SparseArrays/PrecompileTools. Remove
  Adapt/GPUArraysCore weakdeps and extension, remove Test weakdep and Testing
  extension registration. Test remains a test dependency. Remove GPU/JLArrays/
  LambertW extras once historical catalogs are removed. No version bump/release.
* Both `ext/*.jl` are archived. GPU extension has typed methods over deferred
  families; Testing extension imports the retired Testing module and has
  runtime Conjugate defaults and tuple/composite helpers. A small custom
  SeparableFunction example proves extension authors can still implement the
  ordinary mathematical hooks without either packaged extension.
  A custom compact-vector `prox_block!` fixture also verifies the pure block
  interface, fresh allocating result and absence of iterate-storage mutation.
* `precompile.jl`: replace workload payload with only retained functions,
  Float32/64, least-squares gradient/prox, scalar NormL1 prox_block, Scaled and
  three iterate constructors/views/products. Remove all group/spectral/GLM/
  Conjugate/SeparableSum/Kron calls. Keep PrecompileTools wrappers.
* Tests: preserve/migrate retained cases from sparse_iterate, block_iterate,
  symmetric owner/view/alias/parameter suites, smooth, leastsquares, separable,
  scaled chains, numerics and interface tests. Mixed historical test files
  cannot run wholesale. Archive old catalogs and tier entries and write an
  explicit minimal manifest whose runner still rejects undeclared test files.
  Independent dense references, Float32/64, deliberate active zeros, copy and
  alias behavior, arbitrary block layouts, symmetric multiplicities, finite and
  exceptional scales, LS multiresponse and lasso composition are required.
  Check all exports resolve and run meaningful ambiguity detection after
  narrowing shared union dispatches.
* Docs: replace broad API/function catalogs and GLM/group/solver examples with
  supported API, storage contracts, numerical contracts and one custom
  separable extension example. Remove stale Documenter refs from preserved
  storage docstrings by documentation-only edits. Audit history stays labelled
  historical. Require docs build and supported docstring inventory.
* Existing benchmark/test/oracle/conformance catalogs and scripts that import
  removed symbols must be archived or rewritten, not left as runnable broken
  examples. Retained performance comparison uses frozen post-848 baseline on
  the same machine/harness plus A/A controls; no historical full-suite claim.
* CI: one required Linux/current Julia 1.x test job; preserve protected check
  names `PR required jobs` and `PR merge gate`. Remove cross-platform/LTS/GPU
  gates, retired inventory/conformance references and obsolete workflows from
  active minimal validation. No ProximalSolvers edit/pin change.

## Numerical contracts carried across extraction

Preserve default/output-first/alias conventions, unsupported partial alias
policy, real-input/shape checks before writes, Float32/Float64 model precision
and data-vs-parameter distinction, Float16 rejection roles, NaN/Inf and zero
level/weight priorities, rounded-once coefficients and power-of-two recovery,
`value_and_gradient! == f(x)` and `value_and_prox! == f(stored output)` contracts,
source coefficient/data alias protections, owner exactly-symmetric parameter
validation and independent-coordinate multiplicity, and lazy loss caches.
Storage keeps deliberate active zeros, deep copies/alias-safe views and
noncontiguous ordered groups. The implementation must validate each retained
contract against frozen behavior and independent mathematical references.

## Preservation and stop gate

Root verified the three archive branches at the exact baseline and pushed the
annotated non-release tag `preservation/post-848-full-2026-11-10`. Archive branch
protection is not enabled; authenticated administrator action is still needed.
Do not call the branches immutable. Root owns remote updates. Phase B starts
only after the independent architecture critic accepts this inventory and its
boundaries; this document does not constitute that approval.
