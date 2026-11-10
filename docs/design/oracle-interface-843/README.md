# Mathematical function interface research for issue 843

Preserve the scalar interface and add reusable group mathematics only where
the consumer evidence supports it. Julia execution confirms numerical failures
in ProximalSolvers' duplicated group kernels, a missed sparse-group KKT
violation, incomplete metric dispatch, and an inadequate public validation
boundary. Wrapper and storage capabilities need an explicit account before
solvers can safely infer them from the existing traits.

This is a research record for [PB #843](https://github.com/mlakolar/ProximalBase.jl/issues/843),
dated November 10, 2026. The contracts below are proposals for review. No package
implementation, public API, dependency declaration, or existing tests change.
The initial [static audit](https://github.com/mlakolar/ProximalBase.jl/issues/843#issuecomment-6757224452)
remains useful background; the runtime findings below extend and qualify it.

## Evidence and reproduction

The executed pairing is PB
[`8eaf8f812fd0720e180f35cfc75bc7b090e8ea97`](https://github.com/mlakolar/ProximalBase.jl/commit/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97)
and PS
[`f9b7a26663b499b6f881bcc26c56d1acced06749`](https://github.com/mlakolar/ProximalSolvers.jl/commit/f9b7a26663b499b6f881bcc26c56d1acced06749).
PS actually declares PB `e8a45a8a6ffe2a6f92d83c4c77d62a9b7631e291` in
its [project](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/Project.toml#L15).
The research environment deliberately develops the newer PB checkout; these
results do not certify PS's declared dependency pairing.

Execution uses Julia 1.10.10 on Linux x86_64, one Julia thread and one OpenBLAS
thread, with bounds checking enabled. The runtime executable and package depot
were installed under `/tmp`; neither source checkout's project or manifest was
modified. `results/runtime.toml` records the loaded source paths, source commits,
CPU, BLAS, and harness digest. The numerical assertions include expected
defects: a passing research harness means those defects were reproduced,
not that the packages passed their regression suites.

The final run records **2,500 operation observations, 30 checked reproductions,
and 1,636 passing research assertions**. Of the observations, 1,758 return and
742 raise an exception (168 dimension errors, 318 missing-method errors, and
256 argument errors). Those exceptions include deliberately unsupported or
malformed combinations; they are not a count of defects. The catalog contains
90 exported function types, including abstract and uninstantiated types.

| Artifact | Meaning |
|:---|:---|
| [audit.jl](audit.jl) | Actual PB/PS calls, independent analytic/KKT/high precision references, extension fixture, and measurements |
| [audit.log](results/audit.log) | Checked reproductions and Julia assertion summary |
| [runtime.toml](results/runtime.toml) | Provenance, checked results and performance observations |
| [capabilities.tsv](results/capabilities.tsv) | Every attempted operation, representation, return or exception; successful combined value comparisons are included |
| [family-traits.tsv](results/family-traits.tsv) | Public function types and type-level traits; unconcretized wrapper types are identified |
| [consumer-matrix.tsv](results/consumer-matrix.tsv) | Each of 45 oracle names to PB definitions and PS consumers/extensions, including explicit zero-use rows |
| [source-sites.tsv](results/source-sites.tsv) | Both directions of the source inventory, with individual pinned links |
| [inventory.jl](inventory.jl) | Repeatable Julia syntax inventory of 51 PB and 39 PS source files |

Read `MethodError` as an absent method for the actual signature,
`ArgumentError` as the operation's explicit rejection, and `DimensionMismatch`
as an incompatible tested shape. A family with both successful and rejected
signatures has partial support. These observations do not distinguish a
documented rejection from a defect by themselves. Type-trait queries on an
uninstantiated wrapper such as `Tilt` can return the generic `false`; that
does not describe `Tilt(SqrNormL2(),c)`. Concrete instance traits in
`capabilities.tsv` and operation-specific documentation take precedence when
choosing a solver.

The inventory parses Julia syntax, so docstrings and comments are excluded.
Counts are unique syntactic sites, not method-table sizes, executions, or
proof that a branch is reachable. Generated definitions, macro expansion,
higher-order function references, and calls through variables need manual
interpretation. Precompile workloads are counted separately. Ordinary `f(x)`
evaluation cannot be attributed to a named oracle by this inventory; objective
evaluations in PS are inspected separately in the solver initialization,
line-search, CD loss-value and certificate paths.

To reproduce, set `PB_ROOT` and `PS_ROOT` to clean checkouts of the commits above,
and set `AUDIT_ENV` and `AUDIT_DEPOT` to fresh writable directories. Julia and
the package dependencies must be available; instantiation may require network
access. Only the isolated environment receives a manifest.

```sh
JULIA_DEPOT_PATH="$AUDIT_DEPOT" JULIA_PKG_PRECOMPILE_AUTO=0 \
  julia --startup-file=no --project="$AUDIT_ENV" -e \
  'using Pkg; Pkg.develop([PackageSpec(path=ARGS[1]), PackageSpec(path=ARGS[2])]); Pkg.instantiate()' \
  "$PB_ROOT" "$PS_ROOT"

JULIA_DEPOT_PATH="$AUDIT_DEPOT" JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --startup-file=no --check-bounds=yes --project="$AUDIT_ENV" \
  "$PB_ROOT/docs/design/oracle-interface-843/audit.jl" /tmp/oracle-audit-results

julia --startup-file=no "$PB_ROOT/docs/design/oracle-interface-843/inventory.jl" \
  "$PB_ROOT" "$PS_ROOT" /tmp/oracle-audit-results
```

This is representative CPU research: Float32 and Float64, ordinary/noncontiguous
groups, sparse iterates, the symmetric owner, matrix functions and a custom
extension. GPU arrays, BigFloat public-oracle coverage, every built-in instance,
all nonfinite combinations, and whole-solver release regression suites remain
outside the executed coverage. BigFloat at 256 bits is used as an independent
reference for ordinary diagonal L-infinity projections.

## Complete operation surface and consumer boundary

The table covers the public mathematical operations rather than only group
lasso. Exact consumer links for each operation are in the generated matrix.

| Surface | Existing contract and consumer evidence | Interface conclusion |
|:---|:---|:---|
| `f(x)` | Objective value; extended-real domain behavior. Used throughout PS. Scalar loss `loss(eta,y)` has a separate two-argument domain. | Keep. Document domain, model precision and value overflow alongside each family. |
| `value_entry` | Function-author term hook; no named PS call found. Generates values and combined values for user separable functions. | Keep as an extension hook; lack of direct solver calls is expected. |
| `gradient`, `gradient!`, `value_and_gradient[!]` | Mutating/combined operations are used in PS; cache-aware CD model methods extend `gradient!`. | Keep; model residual/predictor caches remain in PS. Allocating forms are conveniences. |
| `deriv`, `deriv2`, `curvature`, `value_and_deriv` | PS GLM model kernels consume all four. | Keep the `ScalarLoss` author protocol separate from `AbstractFunction`. |
| `prox`, `prox!`, `value_and_prox!` | PS uses the mutating and combined operations, including ADMM, primal-dual, SLOPE and gradient methods. Combined output is the value at the written point, with documented indicator exceptions. | Keep. A combined indicator value is not a general feasibility certificate. |
| `prox_entry` | 13 named PS call sites in the syntax inventory, including positive-curvature and flat-coordinate logic. | Keep as the scalar mathematical kernel; it does not manage storage. |
| `prox_coordinate!` | Two named PS calls, both in generic grouped model steps. Returns the scalar or a group view while updating active storage. | Keep as a storage operation. It is not a suitable mandatory replacement for pure scalar kernels. |
| Array-step prox | PB separable steps are per entry; group steps are per group. PS's separate metric code implements heterogeneous within-group metrics. | Preserve existing meaning; add a separate metric operation only with an explicit capability. |
| `hessian_vec[!]`, `quadratic_model[!]`, `hashessian` | PS generic Newton uses the model and trait; no direct PS Hessian-vector calls found. | Keep. An unused public oracle can still serve other consumers. Verify model variable conventions per family. |
| `lipschitz`, `coordinate_lipschitz[!]`, `opnorm2` | PS uses the global bound and mutating per-entry bounds. No named calls to scalar coordinate bounds or `opnorm2`. | Keep. A Hessian diagonal is not a simultaneous-step majorizer. |
| `kkt_violation` | Public zero-coordinate score; one named PS consumer. Private CD screening duplicates mathematical predicates. | Reuse mathematical zero-subgradient tests where parity and cost are established. Do not equate this with a full nonzero-coordinate stationarity residual. |
| `restrict` | Used by PS active APG; preserves the objective up to allowed finite additive constants. | Keep; do not promise restriction for every prox-capable wrapper. |
| `gradient_mapping[!]` | Public proximal-gradient residual; no named PS use found. PS stopping tests also include metric and relative normalizations. | Keep. Metric residuals and certificate construction remain solver computations. |
| `lla_weight`, `lla_weights[!]` | PS consumes the mutating weights. Group output is one weight per group, not per array entry. | Keep. State the output index space explicitly. |
| `check_parameters`, `entry_parameters`, `check_symmetric_parameters` | Custom hook, symmetric metadata, and public symmetry check. PS still calls private `_checkparams` at two sites. | Public complete preflight is missing; a blind hook substitution is incorrect. |
| `issmooth`, `isproximable`, `isseparable`, `isconvex`, `hashessian`, `iszeropreserving` | Mathematical/type properties; support varies with output, metric, composition and layout. | Publish operation/representation capabilities. Avoid interpreting a Boolean as a promise for every representation. |
| `haslmo`, `lmo[!]`, `lmo_atom` | Compact-set linear minimization; instance-dependent finiteness and explicitly limited output layouts. No named PS use found. | Retain the existing optional interface. Dynamic simplex/ball checks support its distinct role. |
| `reset_cache!` | Required after documented in-place changes to retained model data; no named PS use found. | Document lifecycle; do not move caches between packages to create usage. |
| Conjugate/dual values | `Conjugate` provides proximal behavior and selected values for `AbstractFunction`; PS owns scalar-loss conjugates and dual certificates. | A small scalar-loss value hook is plausible. The existing wrapper is not a substitute for `loss*(u;y)`. |
| Group metadata and `samegroups` | Group identity/layout validation; three named PS uses of `samegroups`. | Group IDs, entry indices, symmetric coordinates and active slots must remain distinct. |

Primary contracts: [interface](https://github.com/mlakolar/ProximalBase.jl/blob/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97/src/interface.jl),
[separable extension](https://github.com/mlakolar/ProximalBase.jl/blob/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97/src/functions/separable.jl#L8),
[metric and curvature](https://github.com/mlakolar/ProximalBase.jl/blob/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97/src/metric.jl),
[second order](https://github.com/mlakolar/ProximalBase.jl/blob/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97/src/second_order.jl),
[LLA](https://github.com/mlakolar/ProximalBase.jl/blob/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97/src/lla.jl),
[linear minimization](https://github.com/mlakolar/ProximalBase.jl/blob/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97/src/lmo.jl).

### Author protocol and solver protocol

The documented minimum `SeparableFunction` fixture implements `value_entry`
and `prox_entry`, plus the public parameter hook for its weight vector. It gets
value, full prox, combined value/prox and coordinate updates. Declaring
`iszeropreserving` enables sparse traversal. Declaring `entry_parameters` makes
the symmetric-owner boundary checkable; the fixture succeeds with symmetric
weights and rejects asymmetric weights before output writes.

It does **not** get `kkt_violation`, `restrict`, gradients or Hessians simply
from those two entry methods. The runtime table records `MethodError` for its
KKT and restriction calls. This qualifies the trait-table language linking
separability to a KKT oracle: the extension documentation and solver requirements
need to distinguish optional hooks. Strengthening every existing author's
minimum protocol would be a compatibility change. A solver using active-set
screening must either require KKT/restriction hooks or choose a documented
algorithmic fallback.

### Representative capability matrix

`yes` means the listed ordinary runtime operation returned successfully;
`unsupported` means the tested combination raised an error. It does not mean
that all numerical inputs, layouts or wrapper chains were covered. `entry`
steps have four elements and `group` steps have two in the four-entry fixture.
Input/output representation is part of the operation, especially for owners.

| Family | Full scalar prox on Array/SparseIterate/AtomIterate | Atom coordinate | Array step layout | Restriction | Main qualification |
|:---|:---|:---|:---|:---|:---|
| L1, nonnegative L1, ElasticNet, MCP, SCAD, L0 | yes | yes | entry | supported examples | Mathematical and solver-screening semantics differ for one-sided penalties. |
| Entropy, negative log, boxes | yes | yes | entry | domain-dependent | Zero need not be in the domain; restriction is not universally legal. |
| Group L2/L-infinity, sparse-group, group MCP/SCAD | yes | yes | group | yes on the subset example | Scalar steps do not imply heterogeneous diagonal support. |
| `Scaled(group)` | yes | yes | group | yes | Scaling must be combined without premature coefficient overflow. |
| `Translate(group)` | yes | unsupported | group full prox succeeds | rejected in fixture | Nonseparable translation requires excluded translations to vanish for restriction. |
| `Tilt(group)` | yes | unsupported | tested group vector rejected | yes | The wrapper's array-step shift requires entry shape; group metadata alone does not close this operation. |
| `Regularize(group)` | yes | yes | group on Array; Atom output rejected | yes | Existing explicit output-layout restriction. |
| Group-constant array `Precompose(group,a)` | yes | unsupported | tested vectors rejected | unsupported | Group KKT succeeds; values/full scalar prox do not imply coordinate/restrict closure. |
| `Sum` of smooth squares | prox unsupported; gradient succeeds | unsupported | unsupported | not in this fixture | Generic sums are not automatically cheaply proximable. |
| Custom `UserSquare` | yes | yes | entry | missing hook | Minimum author protocol is smaller than the screening protocol. |

For symmetric matrix points, L1, SqrNormL2, nuclear norm, PSD, Spectral(L1),
SingularValue(L1), and the symmetric custom square have successful owner
value/prox cases. LogDetLoss has owner values, dense gradients and Hessian
products, while allocating prox and owner-output prox are absent. Its source
provides a **dense-output prox at an owner input**. This is a representation
distinction, not evidence that its mathematical prox is missing.
[LogDetLoss owner methods](https://github.com/mlakolar/ProximalBase.jl/blob/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97/src/functions/logdet.jl#L330).

The matrix functions cover symmetric/spectral behavior; the vector functions
cover smooth losses, separable and nonconvex penalties, indicators and wrappers.
The complete exported function-type inventory is supplied separately to avoid
mistaking this representative runtime selection for every built-in's semantic
certification.

## Confirmed runtime findings

### R1 Group shrinkage loses finite answers at both magnitude extremes

With `u=(1e155,0)`, `lambda=1e155`, and `L=2`, PB returns `(5e154,0)`.
PS's sum of squares overflows; its shrink factor is one, and its quadratic
model step returns `(1e155,0)`. With `u=(1e-200,0)` and the correspondingly
small level, PS's squared norm underflows and returns zero instead of
`(5e-201,0)`. Float32 examples use `1e20` and `1e-30` and reproduce both
failures. The actual LS, quadratic and multiresponse model group steps are
executed, not just a retyped formula.

The PB differential checks compare 25 ordinary groups per precision, of lengths
1, 2, 5, 32 and 256, through full group prox, `NormL2`, exact input/output alias,
and the five-argument Atom coordinate operation. They agree. These results
support sharing PB's mathematics without discarding PS residual caches and its
inactive-group/lazy-curvature optimization.

Classification: **missing storage-independent group primitive; numerical
duplication in PS**. Sources:
[PB stable group norm and shrink](https://github.com/mlakolar/ProximalBase.jl/blob/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97/src/functions/norms.jl#L552),
[PS factor](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/models.jl#L637),
[LS](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/leastsquares.jl#L597),
[quadratic](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/quadratic.jl#L173),
[multiresponse](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/multiresponse.jl#L117).

### R2 Diagonal group metric kernels need their own numerical contract

For group L2, `z=(1e200,0)`, `d=(1,1)`, `c=2e200` has exact result zero.
PS returns `(NaN,NaN)`: the overflowed norm also gives an infinite root-search
bracket. Float32 at `1e20` reproduces it. This includes a uniform metric, so
PB's ordinary group prox is an independent exact-zero cross-check.

For group L-infinity, let `v=floatmax(T)/4`, `z=(v,v)`, `d=(8,8)`, `c=v`.
The exact clipping level is `15v/16`, a finite representable answer. PS returns
the unchanged `(v,v)` after weighted sums overflow, for both precisions.

Ordinary heterogeneous L2 output satisfies its stationarity equation to about
machine precision and differs from radial scalar-step shrinkage. Twenty
ordinary L-infinity groups per precision agree with an independent 256-bit
bisection of the clipping-level equation. The problem is numerical range
handling, not the underlying ordinary optimality equations.

Classification: **missing metric mathematical primitive and numerical
contract**. Source:
[PS metric group kernels](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/solvers/diagonal_metric.jl#L112).

### R3 Broad metric dispatch writes before discovering missing support

The actual PS `_metric_prox!` accepts all PB group penalties via the
`AbstractGroupNorm` alias. SparseGroupNormL2, GroupMCP and GroupSCAD reach a
missing `_metric_group!` method and throw `MethodError`. The output, initialized
to 17, already equals the input afterward. L2, L-infinity and the tested scaled
L2 work. A broad full-prox trait is insufficient to authorize this call.

Classification: **capability and validation gap**. Reject unsupported metric
combinations before output writes; implementing more penalties is a separate
decision. This behavior belongs to PS's private routine and is not presented
as a violation of a PB atomicity promise.

### R4 Public preflight misses a built-in shape error

`check_parameters(NormL1(0.3,ones(3)), zeros(4))` returns `nothing`.
Private `_checkparams` and public `prox!` both throw `DimensionMismatch`; the
prox output remains unchanged. The custom fixture's public parameter hook
does reject the same mismatch. The hook is working for its author-extension
purpose but cannot replace solver preflight for built-ins.

Classification: **validation gap**, coordinated with
[PB #374](https://github.com/mlakolar/ProximalBase.jl/issues/374).
[Public default/private fallback](https://github.com/mlakolar/ProximalBase.jl/blob/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97/src/interface.jl#L195),
[PS private dependency](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/models.jl#L617).

### R5 Flat group rejection does not establish unboundedness

For `f(z)=0.1z1` and `g(z)=||z||1+||z||2`, zero is the global minimizer:
the linear term is bounded by `0.1||z||2`. PS's generic quadratic-model group
step rejects its zero curvature and nonzero slope. This reproduces the bounded
sparse-group limitation in [PS #154](https://github.com/mlakolar/ProximalSolvers.jl/issues/154).

The scalar probes confirm finite minimizers for an ElasticNet with positive
ridge, a negative-log barrier, and a box excluding zero. Nonzero-slope MCP and
SCAD affine problems are rejected; their bounded penalties cannot bound a
nonzero linear function. The existing scalar logic is useful evidence for an
optional oracle, not justification for a universal infinite-step fallback.

Classification: **missing affine minimization primitive; existing PS
limitation**. Sources:
[flat scalar minimizers](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/models.jl#L143),
[flat group rejection](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/groups.jl#L49).

### R6 KKT reuse needs numerical and directional parity

Ordinary KKT sign comparisons cover 560 cases per precision across L1,
nonnegative L1, ElasticNet, MCP, SCAD, group L2, scaled sparse-group and a
one-sided boxed L1. Each precision has 26 disagreements, all for
`NormL1Nonneg`: PS uses an absolute gradient statistic while PB correctly uses
the allowed direction. With gradient 2 and level 1, PB's score is -3 and PS
says violated. For `f(z)=z^2/2+2z` on `z>=0`, PS `lambda_max` returns 2 while
the smallest nonnegative level yielding zero is 0. This overestimates the path
start and retains unnecessary candidates; it is not evidence of an incorrect
final minimizer.

A separate sparse-group example is consequential in the other direction:
`grad=(1e-200,0)`, L1 level zero, L2 level `5e-201`, overall scale one.
PB's score is `5e-201`; PS returns false because its thresholded squared norm
underflows. Float32 at `1e-30` reproduces this false negative. This is a
reproduced predicate error; a whole screened path that terminates incorrectly
has not been demonstrated. Exact L1 ties and zero/infinite weights agree in the
additional probes.

Classification: **existing oracle missing from a consumer, plus numerical
duplication**. Sources:
[PB one-sided score](https://github.com/mlakolar/ProximalBase.jl/blob/8eaf8f812fd0720e180f35cfc75bc7b090e8ea97/src/functions/separable.jl#L447),
[PS generic screening](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/models.jl#L522),
[PS sparse-group squared norm](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/sparsegroup.jl#L77),
[PS path start](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/path.jl#L28).

### R7 Working conventions worth preserving

The multiresponse curvature fixture returns `L=[35 35;56 56]`, with entry 3
equal to 35. `coordinate_lipschitz!(L,f)` returns the supplied array and the
two-argument `coordinate_lipschitz(f,k)` returns a scalar. There is no
one-argument all-entry allocating method. The name is consistent with a scalar
query plus bulk fill; document it rather than rename it for superficial
allocating/mutating symmetry.

GLM Hessian products agree with finite differences and its updated quadratic
model agrees in gradient at the expansion point. Gradient mapping agrees with
an independently composed gradient/prox step. Built-in gradient and Hessian
exact aliases are rejected in the tested examples. Ordinary combined
value/prox and value/gradient observations that return successfully have no
reported value disagreement in the runtime inventory.

The group NaN probes distinguish dependence and endpoint behavior: a finite
positive step propagates NaN through the coupled group; step zero keeps it in
its input entry; infinite step or infinite group weight fixes the group at
zero. An inactive group that stays zero is not activated. A previously active
group can remain stored after becoming zero until `dropzeros!`; this is storage
policy, not a prox error.

Twelve scalar-loss checks for Squared, Logistic, Poisson and Huber satisfy
Fenchel equality at `u=deriv(loss,eta,y)`, with worst Float64 absolute error
`4.44e-16`. These confirm reusable mathematics in the PS conjugate routines;
they do not certify all dual-domain endpoints or an entire certificate.
[PS scalar conjugates](https://github.com/mlakolar/ProximalSolvers.jl/blob/f9b7a26663b499b6f881bcc26c56d1acced06749/src/cd/gap.jl#L38).

## Reverse inventory of solver mathematics

| Solver-owned computation | PB operation or missing primitive | Ownership and action |
|:---|:---|:---|
| Three group-L2 shrink factors and their norm accumulation | Full group prox exists; pure one-group hook missing | Share PB mathematical kernel, retain PS cache updates and inactive skips. R1. |
| Entrywise metric L2 root and L-infinity clip searches | No PB heterogeneous group metric oracle | Correct PS numerical behavior independently; consider a narrow PB primitive. R2/R3. |
| Flat scalar closed forms, exponent quotient, wrapper/domain handling | No explicit affine-entry minimizer | Optional PB mathematical operation after status design; do not move CD drivers. R5. |
| Flat group boundedness checks | PB group KKT can test some zero solutions, not return all affine minimizers | Fix PS #154 separately; add an affine-group operation only for proved classes. |
| Zero-coordinate KKT checks | PB `kkt_violation` exists | Establish parity and use the existing mathematical predicate. R6. |
| Level replacement, lambda-max root, sequential strong rule | Path parameterization is not an intrinsic function operation | Keep in PS. A mathematical zero-subgradient test may be reused. |
| MCP/SCAD concavity constants | No public weak-convexity bound | Candidate PB metadata. Define a sufficient nonnegative bound; PS's source comment says “largest” rho, but convexification holds for all sufficiently large rho. |
| Scalar-loss conjugate and saturated values | Missing loss-value hooks; `Conjugate` is a different protocol | Optional PB extraction; dual feasibility, certificate buffers, tolerance and radii stay in PS. |
| Cached gradients, group curvature eigenvalues, centered/weighted Gram products | PB gradients, scalar derivatives and quadratic models already used | Keep model caches and loss-specific incremental operations in PS. |
| Symmetric index conversion and multiplicity | `tril_cartesian`, `coordinates`, symmetry check already public | Coordinate with PS #230; no owner redesign. A descriptor is an optional documentation/convenience experiment. |
| SqrtLeastSquaresLoss, SqrtQuadraticFunction | PS extends PB's function types | Review mathematical type placement after 0.6; exact coupled CD updates remain in PS. |
| SLOPE cluster CD, ADMM splitting, primal-dual steps, Newton damping, line searches | PB prox/Conjugate/models already consumed | Solver algorithms. No new PB operation justified by formulas alone. |

Source coverage and individual sites are in `source-sites.tsv`. The decision
to retain a computation in PS is part of the audit, not an unimplemented PB
work item.

## Candidate APIs and explicit tradeoffs

No name in this section is implemented or approved. Each meaningful addition
has two possible designs; prefer the smaller contract until benchmarks or a
second consumer justify more structure.

| Need | Design A | Design B | Recommended experiment |
|:---|:---|:---|:---|
| Pure group prox | `prox_group!(out,g,u,k,gamma)` with compact vectors and a penalty group ID | `prox!(out,group_function(g,k),u,gamma)` using an immutable parameter view; for L2 initially compare an equivalent `NormL2` | Benchmark A against an accurately parameterized B. Do not round `lambda*weight` before applying gamma. |
| Heterogeneous group metric | `prox_group_metric!(out,g,u,k,gamma,d)` | `prox!(out,g,u,DiagonalMetric(d),gamma)` with a separate metric dispatch family | Start with A for L2/L-infinity. B is broader and needs representation-specific capability rules. Neither reinterprets an existing step vector. |
| Affine term minimization | `minimize_affine_entry(g,slope,i,scale)` returning a typed status and a scalar when attained | `minimize_affine!(out,g,slope,subset,scale,workspace)` for a prevalidated subset/group | Start with A. Add a compact-group variant only for classes with a proved boundedness/minimizer algorithm. |
| Complete preflight | New `validate_function(f,prototype)` composing full built-in checks and author hooks | Make `check_parameters` complete for built-ins and recursively composed wrappers | A avoids silently changing the extension hook's meaning; B avoids a second name. Audit recursion and public return semantics before choosing. |
| Operation capabilities | Small operation-specific predicates, e.g. `supports_group_metric(g,layout)` | Rely on narrowly dispatched operations with explicit preflight rejection and a documented capability table | Start with table plus honest rejection. Add predicates only when PS needs to choose algorithms before constructing buffers. Avoid a new global trait hierarchy. |
| Scalar conjugate/saturation values | `conjugate_value(loss,u,y)` and `saturated_value(loss,y)` | A `ConjugateLoss(loss)` callable plus a separate saturation hook | A fits the existing scalar-loss protocol. B is extensible but adds another mathematical object for one value query. |
| Weak convexity | `weak_convexity_bound(g)` returning a sufficient nonnegative rho | `curvature_lower_bound(g)` returning a nonpositive lower Hessian/subgradient curvature bound | Prefer an unambiguous bound and documented scaling law; do not equate it with a path level. |

### Shared numerical and layout requirements

The proposed group operation solves
`min_z g_k(z) + ||z-u||^2/(2gamma)`. Compact `u[j]` corresponds to
`g.groups[k][j]`; `k` is not an active slot or an entry index. Output has the
same compact shape and its `!` form returns that output. No active set or
model cache changes. Empty/malformed groups, output length, index and layout
validation occur before writing.

The metric version solves
`min_z g_k(z) + sum(d[j]*(z[j]-u[j])^2)/(2gamma)` for finite positive `gamma`
and finite positive per-entry `d`. This distinguishes metric weights from PB's
existing reciprocal entry steps. Uniform `d=a` must agree with ordinary group
prox at step `gamma/a`. A zero-group criterion uses the dual norm of `d.*u`
against `gamma` times the penalty coefficient without overflowing intermediate
products. Nonuniform L2 is generally not radial.

Precision follows the 0.6 model/output rules: preserve common Float32/Float64
fast paths, construct effective coefficients from level, weight and step
without losing finite representable results, and round at the documented
boundary. Wider parameter types do not justify prematurely narrowing a
coefficient. Use norm scaling and range-safe root brackets on exceptional
paths; use allocation-free ordinary paths. Certified high precision recovery
can be selective when needed and must not change global rounding/precision.

Support exact object alias `out===u`; partial/shifted overlap is unsupported
unless deliberately implemented. Output must not alias metric/parameter
arrays or mutable workspace. An owner, AtomIterate or SparseIterate is not a
compact-block output in the initial contract. Strided vectors and ordinary
views are the first layouts; other backends need explicit dispatch and tests.
Return value and active-storage bookkeeping remain those of the existing
`prox_coordinate!` when that method delegates to the pure kernel.

For ordinary group prox, match PB's existing zero/Inf step and weight policies,
NaN dependence, identity and hard-fix behavior; do not replace them with a
blanket finite-input rule. For the new metric operation, initially reject zero,
negative, NaN and infinite metric entries and degenerate gamma before writes.
Specify nonfinite **argument** behavior per supported penalty, distinct from
invalid metric parameters. A numeric root failure must throw a documented
error instead of returning a false finite minimizer; whether that failure
commits staged output must be decided explicitly. Predictable unsupported
types/layouts always fail before output writes.

Affine minimization must distinguish `attained`, `unbounded_below`,
`infimum_not_attained`, `unsupported` and `nonrepresentable`. For example,
`min_z NegLog(z)` has infimum minus infinity; an exponential/entropy-related
limit can instead be finite but unattained or unattainable in the output type.
The attained status carries a scalar/compact group; other statuses carry no
bogus minimizer. Define tie selection (prefer zero or a specified nearest
minimizer), domains, zero scale, and wrapper propagation. A scale of zero
requires an explicit decision about preserving an extended-real domain; it
must not silently invoke an infinite-step heuristic. Nonconvex zero-prox
fixed points do not prove affine boundedness. The scalar API has no aliasing;
a mutating group variant stages non-attained outcomes and preserves output.

Preflight validates parameter shapes, group layout, required metadata and the
documented symmetry policy using `size` for a non-array owner. It does not
certify later mutation of retained arrays, numerical convergence, or every
future argument. No public “validated” token should outlive mutable parameters
without ownership/version rules. Repeated coordinate calls retain their
existing guarantees until a separately measured prepared context is designed.
The contract must also say which parameter-domain constraints are rechecked
after permitted mutation of retained arrays, rather than assuming that
constructor validation still covers changed weights or bounds.

Scalar conjugate values must state the exact dual domain, endpoint limits,
NaN/Inf behavior, response constraints, model precision, zero-weight policy,
and ordinary-path allocation. A public loss oracle must not inherit PS's
certificate-specific tolerance for a point just outside the dual domain
without an explicit decision. Saturated value needs a clear infimum/domain
definition even when a finite saturating predictor does not exist. Weak
convexity bounds must state scaling, wrapper support, infinity for unknown
finite bounds, and unsupported nonconvex compositions; do not return zero
merely because a method is absent.

## Performance interpretation and implementation gate

The executable record measures existing full one-group prox, the `NormL2`
alternative, and Atom coordinate prox at group lengths 2, 32 and 256 for both
precisions. It also runs actual CD and diagonal PGD on a small group quadratic
and checks their independent gradient-mapping residuals. Measurements include
warmed minimum/median time and allocated bytes; no proposed API is installed.
The small CPU fixture cannot establish an application-level speedup or a
release regression threshold. Use `runtime.toml` for exact measurements.

For example, the following are warmed median nanoseconds on this runner;
all these kernels allocate zero bytes in the typed harness. The block
alternative has a unit group weight here, so the measurement does not settle
how to represent exceptional products of a general level and weight.

| Precision and group length | Full group prox | `NormL2` on compact block | Atom coordinate step |
|:---|---:|---:|---:|
| Float32, 2 | 10.2 | 6.3 | 13.6 |
| Float32, 32 | 40.1 | 31.9 | 62.9 |
| Float32, 256 | 312.4 | 301.5 | 483.5 |
| Float64, 2 | 10.2 | 6.1 | 14.6 |
| Float64, 32 | 42.4 | 39.1 | 72.6 |
| Float64, 256 | 312.0 | 301.3 | 500.9 |

The small whole-solve CD runs have maximum gradient-mapping residual below
`6e-7`; diagonal PGD is below `3e-6`. Their measured median times range from
1.3 to 1.4 microseconds for CD and 4.9 to 15.3 microseconds for diagonal PGD,
with fresh-iterate/setup allocations included. These are baseline observations
for a four-entry problem, not a ranking of algorithms. They motivate measuring
the real cache-preserving PS migration rather than accepting a kernel-only
speed claim.

Before implementing a shared kernel or migrating PS, require differential
tests across lengths 1, 2, 5, 32, 256, unequal/zero/infinite weights,
noncontiguous groups, inactive and retained-zero groups, wrapper coefficients,
Float32/64, exact alias and unsupported overlap. Include the R1/R2 extremes,
threshold ties and NaN/Inf endpoints. For metrics, check both zero-block dual
conditions and nonzero stationarity; use independent scalar root/projection
references. For affine problems, check analytic domains, nonattainment,
boundedness and nonconvex counterexamples. For KKT reuse, include one-sided
constraints and the R6 false negative.

Use a fixed PB/PS pair for repeated A/A and A/B whole-solver runs: LS,
quadratic and multiresponse group CD; L-infinity/sparse-group generic CD;
diagonal PGD/APG; ordinary lasso/ElasticNet; active/momentum paths; Newton;
ADMM and primal-dual; symmetric-owner migration. Vary dense/CSC data, ordinary
group sizes, active density, both precisions and inactive KKT skips. Compare
time, allocations, objective/stationarity/certificates and lazy curvature
behavior. Keep exceptional recovery measurements separate from normal data.
Agree on a regression budget before comparing implementations.

## Proposed follow-up packages and release sequence

These are issue-sized drafts, not tracker mutations or approved API decisions.

| Package | Scope and acceptance | Sequencing |
|:---|:---|:---|
| PS group numerical correction | Differential reproductions for all three R1 models; range-safe norm/threshold; no algorithm/cache changes; whole-solver comparisons | Small correctness work can precede a new PB API. |
| PS metric correctness and rejection | R2 high/low ranges and heterogeneous KKT; R3 unsupported dispatch rejected before writes; scaled versions | Correctness now; broader PB metric API can follow in 0.7. |
| PS zero-KKT parity | R6 nonnegative statistic/path start and sparse-group underflow; conservative ties; screened-path test | Reuse PB's existing operation where costs allow; do not wait for group API expansion. |
| PB standalone group kernel | Choose one of the two designs; specified compact layout, coefficient precision, aliasing and no active-set side effects; migrate coordinate implementation to it | Additive 0.6 only if narrowly agreed and benchmarked; otherwise 0.7. PS migration follows its published contract. |
| PB validation boundary | Complete public preflight preserving custom hooks, group shape and symmetry policy; replace the two PS private calls only after parity tests | Coordinate PB #374 and PS #120; suitable narrow 0.6 interface improvement. |
| PB capability/extension documentation | Author minimum vs optional KKT/restrict; owner input/output matrix; wrapper/step matrix; unsupported metric diagnostics | Documentation can accompany 0.6 without redesigning `AbstractFunction`. |
| PS bounded flat groups | Reproduce R5 and extend only proved penalty classes | Continue PS #154; avoid duplicate umbrella implementation work. |
| PB affine-entry design | Explicit status, ties, scale/domain rules, supported wrappers; analytic differential cases | Prefer 0.7; group extension is a separate decision. |
| PB optional loss/curvature metadata | Scalar conjugate/saturated values and weak-convexity bound only after domain/precision review | 0.7 research; path level and certificates remain in PS. |
| PS integration pin and migration | Pin the intended PB commit; run supported solver/owner tests, benchmarks and documentation examples | Update PB #527 and PS #230 after evidence. Do not infer success from the research environment's deliberate override. |

Respect the existing ownership of
[PB #225](https://github.com/mlakolar/ProximalBase.jl/issues/225) for group wrapper closure,
[PB #787](https://github.com/mlakolar/ProximalBase.jl/issues/787) for measured Atom update cost,
[PB #527](https://github.com/mlakolar/ProximalBase.jl/issues/527) for the migration ledger, and
[PS #230](https://github.com/mlakolar/ProximalSolvers.jl/issues/230) for the owner migration.
Do not rename `prox_entry`, `prox_coordinate!` or `coordinate_lipschitz`, remove
`value_entry`, relocate model caches, or redesign `AbstractFunction` to resolve
these narrower findings.

Documentation work should update the user operation/layout tables in
`docs/src/interface.md`, the author examples in the separable/ScalarLoss
docstrings, and developer/numerical contracts for validation, precision,
aliasing and stored-zero behavior. Give a custom-function example that states
which solver hooks it intentionally implements. Document new mathematical
operations before migrating a consumer; keep the research record outside
Documenter's public API pages until a contract is accepted.

The research is ready for interface review. Closing #843 still requires
maintainer decisions on the proposed contracts, accepted follow-up destinations,
and release sequencing. This audit does not mark those approvals complete.
