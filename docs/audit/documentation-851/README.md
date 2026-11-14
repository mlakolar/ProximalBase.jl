# Documentation rebuild audit and execution plan (#851)

Baseline: post-#850 `master`, `2b373c54`. Editorial reference: post-#848
`eff21b6d7d33c622313122f46c6ee71693c101ee` (`legacy/full`). The latter is
not API authority. One implementation branch and one documentation PR; no
application implementation changes, downstream changes, release, or revived families.

The historical make script, all source pages, design records, and examples were
inventoried before writing. `historical-inventory.tsv` records every file, byte
length and SHA-256. Current source includes 13 top-level Julia files plus four
function files; exports, method dispatch, docstrings, tests, six short current
pages, two examples and Linux/current-Julia CI were inspected. The historical
site's page paths, concise category navigation, KaTeX and collapselevel=1 are
the presentation baseline. Its copied design/acceptance records contain obsolete
function families and pending historical requirements, so they cannot be republished
unchanged as current contracts.

## Page crosswalk

| Historical page(s) | Action | Live authority / executable evidence | Gap to fill |
|:--|:--|:--|:--|
| `index.md` | Modify | `src/ProximalBase.jl`; lasso example | Composite problem, loss vs penalty, PB/solver boundary |
| `getting_started.md` | Modify | `functions/leastsquares.jl`, `scaled.jl`, `separable.jl`; lasso, PG | Explicit 1/(2n) normalization; replace removed gradient-mapping/curvature calls |
| `function_objects.md` | Modify | `interface.jl`, concrete traits; function example | Actual capability table, traits vs available signatures |
| `operations.md`, `interface.md` | Modify | `interface.jl`, `array_steps.jl`; reference calculations | Complete calls, returns, scalar/array steps, KKT and restriction limitations |
| `numerical_usage.md` | Modify | `model_data.jl`, `utils.jl`, `ridge_precision.jl` | FP32/64, ownership, CPU scope, no GPU support claims |
| `extending.md` | Modify | `functions/separable.jl`, `block_prox.jl`, `core_functions.jl` | Complete separable and compact-vector examples, validation and metadata |
| `function_library.md`, `smooth.md` | Modify | `functions/smooth.jl`, `leastsquares.jl`, `scalar_loss.jl`; mathematical checks | Exhaustive retained concrete definitions and errors |
| `penalties.md` | Consolidate | `functions/separable.jl`; weighted soft threshold checks | NormL1, ridge cross-link; remove all deleted penalties |
| `sets.md` | Drop | No retained constraint/indicator library | No live constraints category |
| `calculus.md` | Consolidate | `scaled.jl`; scaled examples | Only Scaled retained; no generic composition or sum API |
| `optimization_algorithms.md`, `proximal_gradient.md` | Modify | gradient/prox/lipschitz; PG example | Educational iterations; explicit locally computed gradient mapping |
| `coordinate_descent.md` | Modify | scalar prox_block, KKT, restrict; coordinate example | Caller computes curvatures/residuals; no removed solver interface |
| `iterates.md`, `sparse_iterate.md` | Retain/adapt | `sparse_iterate.jl`; `iterate_queries`, `iterate_writes`, `active_handle` | Active zeros, capacity vs sparse work, state/value writes, product domains |
| `block_iterate.md` | Retain/adapt | `block_iterate.jl`; `block_copy`, `cache_and_blocks` | Ordered/noncontiguous partitions; no group-penalty constructors; no coordinates/activevalues method |
| `symmetric_sparse_iterate.md` | Retain/adapt | `sparse_iterate.jl`; `matrixview`, `symmetric_iterate_*` | Non-array owner, multiplicity, strict import vs proximal symmetrization |
| `iterate_oracles.md` | Modify | `interface.jl`, `block_prox.jl`; coordinate, block extension | Entry/group/independent-coordinate distinction and explicit storage write |
| `symmetric_parameters.md` | Modify | `symmetric_parameters.jl`, `core_functions.jl` | Retained metadata; exact pair checks; scalar-step owner restriction |
| `migration.md` | Consolidate | exports/method absence; historical snapshot | Short actionable migration, removed names only in historical context |
| `architecture.md` and old architecture figures | Rewrite text / drop obsolete figures | Current include graph in `ProximalBase.jl` | Accurate file ownership without retired hierarchies |
| `developer/index.md`, `oracle_dispatch.md` | Retain/adapt | `interface`, `separable`, `scaled`, `symmetric_parameters` | Stable extension boundaries vs private helpers |
| `developer/{sparse_storage,block_storage,symmetric_storage,views_aliasing}.md` | Retain/adapt | Storage implementation and retained regression suites | Current invariants, ownership, preflight and partial failures |
| `numerics.md` | Modify | `utils`, `ridge_precision`, `minimal_math`, sparse product tests | Scope numerical rules to retained families; distinguish proven tests and broad goals |
| `developer/change_process.md` | Modify | current `CI.yml`, `scripts`, `test/tiers.toml` | Reproducible strict docs build; actual test inventory |
| `developer/contracts/*` from `docs/design` | Consolidate | Retained storage and block hook methods | Updated current contracts; historical acceptance records linked at immutable SHA, not made normative |
| `api/{interface,functions,calculus,iterates}.md` | Retain/adapt | All exports plus essential qualified public hooks | Grouped explicit docstrings, coverage matrix, caveats for generic declarations |

## Historical executable-example crosswalk

| Historical script | Decision and retained material |
|:--|:--|
| 01 lasso three ways | Adapt lasso primitives and PG; no gradient_mapping API |
| 02 lasso path coordinate descent | Adapt scalar update and caller residual/curvature calculations |
| 03 nonconvex penalties | Drop (deleted families); NormL1 covered separately |
| 04 group lasso | Drop deleted penalties; document BlockIterate storage and a custom block oracle |
| 05 logistic regression | Drop deleted loss |
| 06 robust and count regression | Drop deleted losses |
| 07 constrained least squares | Drop deleted indicators |
| 08 graphical lasso | Drop deleted logdet loss |
| 09 PSD and correlation | Drop deleted projections; retain independent symmetric-storage tutorial |
| 10 matrix completion | Drop deleted spectral families |
| 11 sparse plus low rank | Drop deleted wrappers and nuclear norm |
| 12 multitask regression | Retain matrix-response least squares in function checks; drop group norm |
| 13 differential network | Drop deleted KronQuadratic; retain symmetric index/multiplicity concepts |
| 14 active-set lasso | Retain working sets, restrict and zero KKT screens |
| 15 calculus and custom functions | Retain Scaled and custom separable definition only |
| 16 constrained penalties | Drop deleted constraints |
| 17 sparse PCA Fantope | Drop deleted spectral/constraint families |
| 18 ordered penalties | Drop deleted families |
| 19 weighted isotonic regression | Drop deleted indicator |

Eight core examples will cover lasso primitives, PG, coordinate updates, sparse
working sets, ordered BlockIterate storage, symmetric storage, custom separable
functions and a pure compact-vector block extension. Each runs alone in a fresh
CPU Julia process; Literate generates its page and Documenter executes it again
in a page-specific module. Small deterministic fixtures include independent
formulas, stationarity checks, and explicit working-set state assertions.

## Verified interface boundaries

- Retained built-ins: LeastSquaresLoss, SqrNormL2, QuadraticFunction, NormL1,
  Scaled, ZeroFunction, LinearFunction; qualified Squared/ScalarLoss hooks.
- Only NormL1 is a retained built-in SeparableFunction. SqrNormL2,
  LinearFunction and ZeroFunction have separability traits and scalar prox_block,
  but no value_entry method. Traits do not synthesize missing methods.
- Compact-vector prox_block allocates and delegates to user prox_block!; there
  is no built-in compact group kernel or automatic full-array lifting.
- Array steps are implemented for SeparableFunction and ZeroFunction and
  forwarded by Scaled when the leaf supports them. They are not generic ridge,
  linear, least-squares or quadratic capabilities. Symmetric owner output rejects them.
- Symmetric owner smooth allocating methods cover SqrNormL2 and LinearFunction
  and supported Scaled leaves; ZeroFunction has owner value/prox but no owner gradient.
- check_parameters defaults to nothing; entry_parameters must be explicitly
  declared by custom separable functions for symmetric use. Pair validation
  recurses through Scaled. Raw scalar hooks have no owner to validate.
- Mathematical positivity assumptions on steps are not a universal runtime
  positivity check. Do not invent uniform errors or a universal NaN policy.
- Current minimal docs lack most of this material and use one undifferentiated
  autodocs block. Retained source docstrings include historical generic/group
  wording; clarify exact live scope in reference pages and make documentation-only
  corrections if needed, without changing executable source behavior.

## Execution and merge gates

1. Publish this audit and plan to #851 before large edits.
2. Restore historical paths/navigation, rewrite user/function/storage guides,
   current developer contracts, grouped reference and complete API matrix.
3. Restore Literate; strict exports check, doctests, unresolved references and
   invalid docstrings remain fatal. Validate every script in isolation on Linux
   current Julia 1.x. Keep the existing CI platform and merge-gate contract.
4. Record complete validation logs and exact source/commit/environment evidence.
   Check page reachability, removed-name scope and independent mathematical formulas.
5. Inspect actual rendered HTML in Chrome: collapsed sidebar sections, math,
   code blocks, internal links and previous/next navigation, at desktop/mobile widths.
6. Open one PR linked to #851. Obtain independent GPT-6.1 Sol high-effort review
   at the final gate, posted directly on GitHub; fix and re-review. Merge only
   after review passes, required checks pass and HTML inspection is recorded.

Implementation defects or unclear contracts discovered during documentation
validation are reported separately. Documentation will describe actual supported
methods rather than change the API to fit historical examples.

## Findings during validation

- #853 records the infinite-weight NormL1 / non-finite-trial contract ambiguity.
  The manual describes the retained NaN result for finite levels; no kernel was changed.
- A fresh allocating separable prox need not retain explicit active zeros from
  its input. Mutating destination value writes retain its existing active zeros.
- The exported prox_block! hook had no attached docstring; the grouped strict
  reference exposed this. A documentation-only source change supplies it and
  narrows historical group/capability language. Executable source text remains
  identical after stripping docstrings.
- Historical publication was removed with the docs reduction. CI now preserves
  HTML and logs separately and restores Documenter publication on master only,
  after the existing validation/merge gates, without rebuilding or releasing.
