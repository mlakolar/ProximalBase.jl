# Initial ProximalBase 0.5.4 release audit

**Historical edition:** this records the first audit through `5839b4e`.
The subsequent [necessity review](necessity-review.md) finds additional numerical defects, changes the domain contract and simplifies
the implementation. The execution results below apply
to the stated historical commits, not later revisions.

Audit date: 26 October 2026. The fetched `master` edition reviewed is
`f5c5797ded0800afd1c3c8f6289860906473b8e4`, which includes merged PRs
[#125](https://github.com/mlakolar/ProximalBase.jl/pull/125) and
[#126](https://github.com/mlakolar/ProximalBase.jl/pull/126). The first repair
commit is `ddedef20b1827e9109f37c60aa3a7197cc2a0bd8`. The second implementation
repair commit is `f68cd73fd5efcc1b7a59e4c8099f17e7060cafa3`. The final domain-guard
implementation is `c0e13053a5b4d19d113f2d2815cd1c15563268e3`.

**Reviewed master: blocked by the fixed findings below. Repaired candidate:
ready with stated limitations, subject to the final PR-head CI integration
gate.** Local validation is complete. [PR #128](https://github.com/mlakolar/ProximalBase.jl/pull/128)
contains the repairs; its final CI/head receipt accompanies the audit delivery.
Package metadata remains at 0.5.3. No merge, release, tag or publication is
performed by this audit. The review does not prove the absence of bugs.

## Scope and provenance

The audit used an isolated worktree, `pb-release-audit-20261026`, on branch
`audit/release-0.5.4-20261026`. Existing package and downstream checkouts were
preserved. Prior reviews, tests, timing reports and the two merged PRs were
background evidence. Reviewers read the implementation independently, derived
important formulas, traced callers through dispatch boundaries and added
regressions for confirmed defects.

The source ledgers cover all implementation and extension files:

- [Function kernels](function-kernels.md): all 18 `src/functions/*.jl` files
  and `src/lla.jl`, including values, scalar kernels, group operations,
  restrictions and matrix functions.
- [Composition and interface](composition.md): all twelve interface,
  composition, constrained, envelope, metric, second-order and cache files.
- [Storage and extensions](storage-extensions.md): sparse and atom iterates,
  numerical utilities, LAPACK and partial workspaces, testing, precompilation
  and both extensions.
- `src/ProximalBase.jl`: module imports, exports and include order were checked
  against the [public API inventory](public-api.md).
- `src/deprecated.jl`: every compatibility binding, forwarding method, field
  alias and matrix-product helper was checked against the new
  [migration guide](../../src/migration.md). The helpers remain available.

The [documentation ledger](documentation-ledger.md) inventories all baseline
Markdown guidance and distinguishes historical records. All 18 maintained
example scripts were read. Package metadata, test runners, instantiation
scripts, naming checks, CI, documentation CI and benchmark analysis controls
were also reviewed. The [type inventory](type-inventory.md) records declarations
from source and extensions. The ledgers retain local evidence, proposed
repairs, regressions and limitations; the final validation table below is the
authority for execution status.

## Findings ordered by severity

Severity P1 denotes materially wrong mathematical results, invalid state or an
unsafe dispatch boundary. P2 denotes a localized interface, documentation or
numerical boundary defect. The examples establish specific defects, rather
than their frequency in typical workloads. There is no confirmed P0 finding.

### P1 repairs

| Finding and evidence | Repair and regression coverage | Ledger |
| --- | --- | --- |
| Squaring representable large or tiny norms before taking a square root produced `Inf` or zero. For example, sphere projection of `[0,1e-200]` lost its direction. Group-L0 selection also ranked distinct groups as tied after square underflow or overflow. | Retain the fast ordinary reduction and use scaled norms at exceptional square ranges. Rank group norms or magnitudes instead of their squares. CPU, sparse, symmetric, atom and shared-device regressions include independent known projections and group selections in Float32/64. | FK1, FK2; ST04, ST05; [norm fallback](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/smooth.jl#L148), [device group norms](../../../ext/ProximalBaseGPUArraysCoreExt.jl#L359) |
| Entropy and KL proximal formulas overflowed intermediate products or quotients for finite inputs with finite proximal solutions. KL value overflowed `x/q` before taking its logarithm. | Retain logarithmic levels where the step product lies outside the working range, preserve the small-omega limit and use a log-ratio fallback. Tests use asymptotic limits and BigFloat values, including tiny positive KL reference values. | FK3; [entropy kernel](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/entropy.jl#L61) |
| Centered ridge and `Regularize` formed an overflowing step product and numerator, yielding `NaN` instead of a finite convex combination. | Stable reciprocal weights and bounded endpoint interpolation are shared across full, scalar and grouped paths. Tests cover large products, same-sign `floatmax` endpoints, opposite endpoints, mixed precision, aliasing and subnormal effective steps. | FK4; C07; [ridge weights/interpolation](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/smooth.jl#L201), [Regularize transform](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/composition.jl#L205) |
| The `q=2/3` nonconvex proximal formula could throw or return a nonfinite result at finite large inputs. A development repair also exposed sensitivity at subnormal levels. | Replace the fragile quartic calculation with a dimensionless root equation; retain accurate direct ratios for finite nonzero levels and use logarithmic levels only when necessary. Existing BigFloat comparisons and tolerances are retained; new tests check scaled stationarity and extreme levels. | FK5; [threshold and root](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/nonconvex.jl#L181) |
| `LogDetLoss` value and gradient accepted nonsymmetric matrices by factoring one triangle; a general-matrix proximal input likewise ignored the other triangle. | Enforce the finite symmetric positive-definite value domain and use the symmetric part for the Frobenius proximal problem. Tests check domain rejection, independent stationarity references, precisions and aliasing. | FK7; [domain check](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/logdet.jl#L53), [proximal matrix](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/logdet.jl#L90) |
| `BoxConstrained` could clamp a proximal point into a location outside the inner function's domain and report a finite combined value. Disjoint nested boxes supply an elementary counterexample. | Check original interval intersections and working-precision clamp results against built-in scalar domains, including open boundaries and wrapper/conjugate propagation. Test disjoint domains, entropy versus log boundaries, scalar and coordinate calls. A fresh cross-review also repaired mixed-precision touching endpoints: test original interval emptiness before working-precision comparisons, including cases where rounding makes the clamp unchanged. Arbitrary custom domains retain a documented proper-intersection precondition. | C04; [domain intervals](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/constrained.jl#L139), [clamp boundary](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/constrained.jl#L217) |
| A grouped `Regularize` coordinate call wrote storage under `@inbounds` before checking center length and group agreement. Generic atom-coordinate dispatch also suppressed the group-index boundary check. | Validate lengths, group identity and indices before writes. Regressions assert unchanged destination state after rejection. | C02; [group validation](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/composition.jl#L240), [atom boundary](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/interface.jl#L216) |
| Empty-observation least squares worked through its direct proximal shortcut but failed when an affine-constrained caller requested its Hessian decomposition. | Construct dimension-correct empty thin-SVD factors and response buffers; recognize valid empty caches. Tests exercise the full affine call path, vector/matrix responses, reuse and allocations. | FK13; C06; [empty SVD cache](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/leastsquares.jl#L76) |
| Partial spectral prose overstated what represented-pair residual checks establish. They cannot certify that an omitted leading eigenspace was found. | Mark the partial path as opt-in and approximate; explain the guard and the full-decomposition choice. The full path remains the default. A bounded adversarial warm-subspace search found agreement, which is not a certificate of the omitted space. | FK11; ST06 |

### P2 repairs

| Finding and evidence | Repair and regression coverage | Ledger |
| --- | --- | --- |
| `Tilt`, `Translate`, direct `LinearFunction` and centered `SqrNormL2` had inconsistent parameter length/shape handling across value, gradient and proximal calls. Singleton broadcasting could silently change the function. | Check parameters before mutation. Preserve `LinearFunction`'s same-length linear-index convention and ridge/translation's established size contract. Regressions cover singleton and differently shaped inputs, aliasing and unchanged outputs on failure. | C01; FK14; [Tilt](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/calculus.jl#L128), [linear methods](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/smooth.jl#L68), [ridge checks](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/smooth.jl#L165) |
| A proximable `Conjugate` without a standalone value formula failed the common combined value/prox operation. | Use Fenchel equality at the proximal output when a closed value formula is absent. Retain existing closed forms and exact indicator zeros. Test group norms, double conjugates, diagonal steps, aliasing and allocations. Standalone value support is unchanged. | C05; [Fenchel fallback](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/calculus.jl#L269) |
| Moreau proximal interpolation silently used an overflowing sum of finite steps; difference-based interpolation overflowed between finite endpoints. | Reject an unrepresentable combined step before writes and use stable convex interpolation when it is representable. Regressions cover both precisions, scalar/full calls and rejection without mutation. | C08; [step guard](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/envelope.jl#L107) |
| Moreau values squared the distance before dividing by the envelope parameter. `MoreauEnvelope(NormL1(1),1e200)([1e200])` returned `Inf` although the exact value is `5e199`; small energies could underflow. Overflowing endpoint differences also lost finite gradients, and the fused prox could multiply a squared-distance infinity by a tiny factor. | Normalize distance by `sqrt(2γ)` before squaring; scale endpoints when their difference overflows. Use the same helper for scalar `Smoothed`, preserve finite divided differences, and compute fused energy from the returned point and inner proximal point. Float32/64 tests use independently known Huber values and BigFloat opposite-endpoint references, including aliased combined calls. | C09; [energy and derivative helpers](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/envelope.jl#L56), [combined call](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/envelope.jl#L127) |
| Zero penalty levels or zero group weights could multiply an overflowing reduction by zero and return `NaN`. An identity coordinate path must also activate an atom written through its raw view. | Skip absent terms before reductions; preserve independent sparse-group L1 shrinkage and constrained domains. CPU and device regressions include huge finite entries, free groups, aliasing, allocation checks and inactive-to-active atom updates. | FK6, FK12; ST07 |
| Stored-matrix triple `dot` products omitted conjugation of the left vector. | Conjugate the left scalar and compare against independent dense standard-library dot products with real/complex storage and ComplexF32/F64 vectors. | ST01; [stored matrix dot](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/sparse_iterate.jl#L587), [symmetric dot](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/sparse_iterate.jl#L871) |
| `SymmetricSparseIterate` lower-triangle copying differed for views, transposes, triangular and CSC sources. Broadening the method exposed CHOLMOD intersections, including a version-dependent ambiguity. | Use one lower-triangle kernel and exact CSC/CHOLMOD dispatch intersections. Match the library's `CHOLMOD.VTypes` bound across Julia versions. Test mixed precision, dimensions and return identity; run Aqua. | ST02; [copy intersections](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/sparse_iterate.jl#L727) |
| Negative symmetric-iterate dimensions could yield a nonnegative triangular storage length. | Reject negative dimensions before storage calculation; test default and typed constructors and the valid zero case. | ST03; [dimension validation](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/sparse_iterate.jl#L639) |
| GLM coordinate bounds rejected valid columns when there were zero observations. Empty affine SVD coordinates were also misused as an uninitialized-cache sentinel. | Check the column axis and use the existing NaN step sentinel for affine cache initialization. Regressions cover Float32/64, invalid indices, repeated calls and cache reset. | C03, C06 |
| Maintained guidance misstated KL orientation, group partition completion, gradient bounds, curvature models and allocation/device scope. Mutation rules for retained bounds/normals and the compatibility policy were incomplete. | Correct formulas and assumptions; qualify curvature floors, approximate norm estimates and tested CPU allocation scope; document reconstruction and 0.4 migration. Documentation and example checks remain part of final validation. | FK8–FK10; documentation ledger |

These are compatibility-preserving repairs to the advertised operations. They
do not introduce a new optimization algorithm or broaden device support.
Final execution status is recorded below, rather than inferred from the
presence of regression assertions.

## Public API, naming and documentation

No exported operation, constructor, alias or deprecated call was removed or
renamed. Output-first mutating calls, argument order and return conventions
are preserved. Invalid parameter shapes now fail before mutation instead of
silently broadcasting. Mathematically invalid LogDet inputs and unsupported
box-domain intersections now receive the intended rejection.

The [public inventory](public-api.md) includes exported functions and types,
deprecated bindings hidden by Julia's default name listing, and documented
qualified APIs. The qualified `ScalarLoss` family, derivative/curvature
operations and `ProximalBase.Testing` helpers remain discoverable without
exporting internal implementation machinery. `AbstractGroupNorm` is the same
binding as `AbstractGroupPenalty`; it is not another parent type.

The new [migration guide](../../src/migration.md) documents the policy already
stated in [the compatibility source](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/deprecated.jl#L3): 0.4 names
deprecated in 0.5 and removed in 0.6. It covers output-first calls,
coordinate names, type bindings, coefficient normalization and matrix-product
replacements. The group alias has no separately announced removal version.
`IndBallLinf` and `Constrained` are constructor functions, while named GLM
losses are specializations; callers should not assume every constructor name
denotes a distinct concrete type.

Documentation changes align KL orientation and its factor, explicit atom
partitions, dense versus conservative sparse Lipschitz bounds, the distinction
between `opnorm2` estimates and certified bounds, GLM curvature flooring,
combined conjugate values, partial spectral limitations, parameter mutation
rules and warmed CPU allocation guarantees. Example descriptions state what
their checks establish. Historical CUDA, oracle and benchmark records retain
their original dates and editions and do not validate this repair head.

## Package structure and retained machinery

The abstractions remain proportionate to their mathematical roles. Capability
traits do not coincide with inheritance: a direct `AbstractFunction` subtype
may be separable. Group penalties and convex homogeneous group norms need
different dispatch because ball-constrained and nonconvex operations use
different arguments. Sparse, symmetric and atom iterates are array types,
rather than function subtypes.

Two superseded private GPU functors were removed after normalized group
reductions replaced their work. Other apparently overlapping paths were
retained for distinct purposes:

| Machinery retained | Purpose preserved |
| --- | --- |
| Built-in scalar domain and conjugate slope intervals | Reject empty constrained domains without treating numerical objective overflow as infeasibility; share checks across scalar and multiplier paths, while retaining the custom-domain precondition. |
| Full and partial eigen/SVD implementations | Exact default versus opt-in approximate cost tradeoff; partial acceptance does not certify omitted directions. |
| Full and top-eigenpair LAPACK workspaces | Different decomposition workloads, output sizes and cache ownership. |
| Scalar, full-array, symmetric-coordinate and atom-coordinate kernels | Different indexing and mutation contracts, including Frobenius factors and group activation. |
| Closed-form conjugate values and the combined Fenchel fallback | Standalone value availability, exact indicator values and general fused proximal reuse. |
| Linear-constrained live-entry compaction and shared multiplier search | Large-array performance and one underlying root equation. |
| Fixed group metadata shared by atom copies, mutable scratch owned by functions | Group alignment and allocation control; numerical workspaces still require separate thread ownership. |
| Deprecated wrappers and product helpers | Published compatibility through 0.5; removal is separate release work. |

No broad API redesign or architectural rewrite is included. Generic symbolic
domain validation, certified complement-space spectral bounds and pervasive
runtime tracking of mutable parameter arrays would be separate proposals.

## Architecture diagrams

The existing [overview Page](https://chatgpt.com/space/page_19d0374c8de481918344267237028f43)
and its five child Pages are the canonical location. Their native Markdown
and Mermaid format is preserved. Recovered original blocks are saved in
[original Page sources](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/original-pages.json); local diagram sources
and inspected renders are coordinated mirrors:

| View | Source | Render |
| --- | --- | --- |
| Type hierarchy, traits and compatibility aliases | [Mermaid](architecture/01-types.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/01-types.png) |
| Public operations and internal kernels | [Mermaid](architecture/02-oracles.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/02-oracles.png) |
| Wrapper composition and dispatch | [Mermaid](architecture/03-composition.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/03-composition.png) |
| Iterate and coordinate storage | [Mermaid](architecture/04-storage-device.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/04-storage-device.png) |
| Matrix decompositions and spectral calls | [Mermaid](architecture/05-spectral.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/05-spectral.png) |
| Cache/workspace ownership and extensions | [Mermaid](architecture/06-state-device.mmd) | [PNG](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/architecture/rendered/06-state-device.png) |

Thin arrows show inheritance, circle ends composition, dotted arrows dispatch
or selection, and thick arrows runtime calls. The alias edge denotes the same
binding. Grouped boxes summarize source-verified members; they are not extra
Julia types. [Architecture notes](architecture/README.md) give representative
source files, call paths and a rendering recipe. The Pages retain the same titles and hierarchy; their source footer identifies
the assembled audit head. Native Mermaid sources also have inspected standalone renders.

### Principal call paths checked

| Entry and dispatch boundary | Runtime path and state |
| --- | --- |
| `gradient!` / `value_and_gradient!` for `GLMLoss` | The [GLM implementation](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/glm.jl) validates parameter dimensions, forms its cached linear predictor, calls scalar `value_and_deriv`, then multiplies by the transpose design. The [device extension](../../../ext/ProximalBaseGPUArraysCoreExt.jl) substitutes broadcast/reduction and matrix products at this boundary. |
| `hessian_vec!` and `quadratic_model!` | [Second-order methods](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/second_order.jl) call scalar `deriv2`, apply observation weights and design products, or construct/update a squared-loss model. Curvature flooring belongs to model construction; it is not silently treated as the exact Hessian. |
| `prox!` / `value_and_prox!` for separable functions | [Separable array kernels](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/separable.jl) validate parameters and call `prox_entry`; [coordinate dispatch](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/interface.jl) maps symmetric stored coordinates to full entry indices or updates matching atom groups. Group activation is part of the coordinate mutation contract. |
| `Constrained` and wrappers | [Constraint construction and scalar kernels](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/constrained.jl) select the appropriate constrained implementation. The box path calls the inner proximal operation and clamps with a domain check. [Regularize](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/composition.jl) changes the point and step before delegating. |
| `Conjugate` and `MoreauEnvelope` | [Conjugate methods](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/calculus.jl) call an inner prox through Moreau decomposition and own scratch; the combined value can use Fenchel equality. [Envelope methods](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/envelope.jl) reuse an inner proximal buffer for value, gradient and proximal interpolation. |
| Affine-constrained quadratic or least-squares prox | [Affine methods](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/affineconstrained.jl) reuse the inner loss decomposition, build a step-dependent Schur factor and apply the projection correction. [Cache reset](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/caches.jl) invalidates both inner and outer state. Empty observations must remain dimension-correct through this entire path. |
| Matrix prox and projection | [Spectral kernels](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/spectral.jl), [Spectral](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/spectralfunction.jl) and [SingularValue](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/functions/singular.jl) obtain eigenvalues or singular values, delegate vector operations and reconstruct the matrix. [LAPACK workspaces](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/lapack.jl) serve the full path; [partial workspaces](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/src/partial.jl) serve the opt-in approximate path. |

## Validation tied to editions

All local Julia processes are run sequentially with one Julia and one BLAS
thread. The repository workflow uses `scripts/test_master.jl`, its isolated
test environment and bounds checking. A user-install smoke must use a fresh
package environment and an installed copy at a recorded commit, rather than
only the development path.

| Check | Exact edition | Result and evidence |
| --- | --- | --- |
| Full development suite, Julia 1.10.12 | f68cd73fd5efcc1b7a59e4c8099f17e7060cafa3 | Pass: 63,133 assertions, 39 modules; [log](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/full-1.10.log) |
| Full development suite, Julia 1.13.2 | f68cd73fd5efcc1b7a59e4c8099f17e7060cafa3 | Pass: 63,133 assertions, 39 modules; [log](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/full-1.13.log) |
| Final domain guard, constrained/nonconvex/metric/allocation/Aqua suites, both versions | c0e13053a5b4d19d113f2d2815cd1c15563268e3 | Pass: 13,283 assertions per version; [1.10](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/final-guard-1.10.log), [1.13](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/final-guard-1.13.log) |
| Installed-copy load and README examples, both versions | c0e1305 | Pass: exact Git revision installed in fresh environments; four Julia fences each. [1.10](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/installed-user-1.10-retry.log), [1.13](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/installed-user-1.13-retry.log). Harness failures retained separately. |
| Naming and benchmark analysis controls | f68cd73; affected declarations unchanged at c0e1305 | Pass: [naming](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/naming.log), [analysis](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/benchmark-analysis.log), [GPU analysis controls](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/gpu-analysis.log) |
| Documentation, doctests, 18 Literate examples and cross-references | c0e1305 source plus documented final prose edits | Pass with link checking enabled; [build log](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/docs-links.log). API page-size and local remote-HEAD warnings retained; no deployment. |
| Maintained guidance links | final prose tree | [Inventory](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/doc-links.json): all local targets exist and final external targets respond successfully. Prior migration 404 retained in [first check](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/doc-links-before.json) and repaired with a source link. |
| Warmed allocations and bounded timing diagnostics | c0e1305 | [24 CPU cases](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/hot-paths.log): zero bytes/call. [JLArrays groups](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/device-diagnostics.log): 2,184,336 and 2,440,032 host bytes/call. Exploratory one-run timings, no physical-device or speedup claim. |
| Independent Float64 Hypatia/JuMP primitive oracles | c0e1305 | Pass: 75 cases, 1,206 assertions; [log](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/oracle.log). Three explicit retries retained with rejected statuses. |
| Representative downstream usage | c0e1305; dirty sibling editions recorded separately | Pass: [smoke log](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/downstream.log). ProximalSolvers L1 KKT residual 8.33e-17; atom group residual 9.33e-8; HighDimensionalModels lasso residual 6.56e-15. This is representative compatibility, not full downstream replication. |
| Exact assembled-head CI and documentation | [PR #128 current head](https://github.com/mlakolar/ProximalBase.jl/pull/128/checks) | Required integration gate; final immutable run IDs, job conclusions and head SHA are supplied in the delivery receipt. Earlier-head CI is background evidence. |
| Source snapshot and checkout preservation | assembled audit head | Final hash/protocol verification and clean-checkout receipts accompany delivery; the baseline source bundle and all local execution records remain in the workspace validation folder. |

The only implementation change after the full local suites was the domain
check in `src/constrained.jl`. Its related suites were rerun on both versions;
CI runs the complete suite on the assembled head. [Receipt notes](https://github.com/mlakolar/ProximalBase.jl/blob/5839b4e973a115fa0d3c1a328903a0a58ded3aac/docs/audit/release-0.5.4/validation/README.md)
explain command provenance, accepted oracle statuses and harness failures.

### Retained development outcomes

Before the machine restart, the session reported a complete Julia 1.10.12
development suite at `ddedef20b1827e9109f37c60aa3a7197cc2a0bd8`: 62,813
assertions across all 39 modules, including examples, Aqua and JLArrays. A
focused Julia 1.13.2 run reported 22,842 passing assertions before the later
followups. The documentation build executed doctests and all 18 Literate
examples. These are interim outcomes, not validation of subsequent changes.
The temporary receipt directory did not survive the restart. The rerun logs
above are durable final evidence and do not depend on those earlier outcomes.

The adverse development outcomes remain relevant:

- An early run loaded the package before source edits and used an unsupported
  diagonal test case. This was not evidence about the repaired loaded code.
- The first new entropy/KL run exposed tiny-reference accuracy loss caused by
  a log round trip. Preserving the original reference repaired it; tolerances
  were not relaxed.
- A later existing half-threshold test exposed subnormal-level accuracy loss.
  The repair retained direct arithmetic for finite nonzero levels; existing
  tolerances were retained.
- A collected Julia 1.13 suite found the empty least-squares/affine cache
  defect and CHOLMOD ambiguities introduced by generalized matrix copying.
- The first LTS attempt found the narrower Julia 1.10 CHOLMOD scalar bound.
  Matching the library's own bound repaired that dispatch intersection.
- The link-enabled documentation build encountered HTTP 429 for one pinned
  historical GitHub report. The exact target was independently read through
  the GitHub connector. A rate limit does not establish a broken target, and
  the automated link check must remain marked incomplete unless rerun.
- A bounded warm-subspace adversarial partial-decomposition probe agreed
  with the full result to about `9.65e-15`; it supplied no counterexample and
  did not prove complement-space coverage.

Final validation retains failures as well as passes and identifies remaining
execution gaps. Passing assertions alone do not settle release
readiness.

## Unverified areas and residual risks

- No physical NVIDIA, AMD or Apple GPU execution, backend compilation or
  device performance measurement is established for the final repair head.
  JLArrays exercises shared device dispatch and scalar-indexing restrictions
  on CPU-backed arrays; it cannot establish physical-device compatibility.
- Local endpoint versions do not exhaust intermediate Julia versions, every
  weak-dependency version allowed by compat, 32-bit targets or every backend.
  Final CI expands platform coverage only for the jobs that actually pass.
- Numerical repairs target demonstrated finite-range failures. Weighted
  products, quadratic values, PAVA/Toeplitz means and all nonconvex candidate
  energies are not universally scaled. A true norm outside the result type's
  range can still overflow even when every coordinate is finite. The Moreau
  energy repairs do not create a package-wide arbitrary-range arithmetic
  guarantee. Parameters that convert outside the input/output working
  precision's range and nonfinite coordinates remain separate limits.
- Partial spectral acceptance remains local to represented and guard pairs.
  Use full decomposition when the missing-leading-subspace risk is material.
- Arbitrary custom scalar domains are not inferred. `ScalarLoss` convexity
  and curvature, and `SingularValue` symmetry/monotonicity requirements,
  remain caller contracts rather than symbolic runtime checks.
- Retained bounds, normals and fixed structural metadata must respect their
  documented mutation rules. Reconstruction is required for immutable cached
  properties; cache reset does not track arbitrary field mutations.
- Exact proximal input/output aliasing is tested in important paths. Derivative outputs require separate storage unless a method explicitly supports aliasing. Every layout of
  partially overlapping views, exotic axes and multithreaded workspace races
  has not been exhaustively checked. `gradient_mapping!` rejects identical
  input/output storage, but partial overlap remains caller-sensitive.
- Linear-constrained and intersection routines have finite numerical stopping
  rules. Their residual checks are not global feasibility or exact projection
  certificates for arbitrary extreme or nonfinite inputs.
- External oracles cover the 75 listed Float64 primitive problems. They do not
  cover all wrappers, nonconvex global minima, GPU methods or other precisions.
  Downstream smoke checks use preserved dirty sibling source, not clean released
  downstream editions. Optional StatsModels integration was not loaded.

## Release decision and remaining work

**The fetched master edition is blocked as-is.** Its confirmed mathematical,
domain and storage defects require incorporating the repairs in PR #128.
The repaired candidate is locally ready with the stated limitations. The
integration gate is successful CI on the exact assembled PR head; the audit
delivery records that head and its final jobs. No confirmed implementation
finding is intentionally left unfixed on the repair branch.

For each P1/P2 implementation blocker listed above, the concrete resolution is
the corresponding repair and regression already present in PR #128. Before
0.5.4, review and incorporate that diff, then validate the resulting master
release head, especially if master has advanced or conflict resolution changes
source. Full decomposition remains the default; physical GPU support must not
be described as revalidated by this audit. A physical-device validation claim
would require running the maintained CUDA suite on the claimed device and
recording its compiler, dependencies and source head.

Metadata, tagging and publication are separate release actions. They were not
performed. No broad API redesign is required by the confirmed findings.
