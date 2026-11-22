# Documentation migration open questions

**Closed.** These historical questions and decisions are frozen, with
no further edits expected. The owner approved the
[Phase 5 acceptance record](https://github.com/mlakolar/ProximalBase.jl/issues/1019#issuecomment-7075157741)
at `8a5fac060a5618cedac05563267212e2e3b15bd6`.
[Owner sign-off](https://github.com/mlakolar/ProximalBase.jl/issues/1019#issuecomment-7081858154)

## Numerical policy

No numerical-policy discrepancy was identified in the verbatim `NormL1` pilot
against source and tests at `7ae86e2093060eb124211e2f21f043c6379f97ae`.
The pilot preserves existing guarantee, characterization and limitation wording.
It assigns no new clause statuses and changes no source or test behavior.

If a later migration exposes an ambiguity, record the source block, both possible
readings, source/test evidence, and the owner's decision here before converting
that statement into a classified clause. Do not infer a decision from an example
that happens to pass.

## GroupMCP decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Gradient finiteness in the zero-group screen](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/docs/src/spec/groupmcp.md#L148-L153) could refer to stored or converted entries | The screen classifies entries after conversion to the working type. A finite stored entry that overflows in the conversion is treated as infinite. `characterized`. | [GroupMCP dispatch](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/src/functions/groupmcp.jl#L514-L533) · [Shared screen](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/src/numerical_kernels.jl#L1134-L1145) · [Shared mixed-width checks](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/test/kkt_block.jl#L1233-L1241) |

## Migration scope decisions

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Retained and removed scope](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/migration.md#L11-L25) could describe the #850 reduction or the current inventory | Date the #850 reduction and keep the historical list of removals. Record later restorations with the corresponding PRs and link to the function library for the current inventory. Implementation note. | [#850 merge](https://github.com/mlakolar/ProximalBase.jl/commit/2b373c546e247781d9514a7000ef3e010343ca4c) · [Exports at #850](https://github.com/mlakolar/ProximalBase.jl/blob/2b373c546e247781d9514a7000ef3e010343ca4c/src/ProximalBase.jl#L18-L25) · [Exports at review](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/ProximalBase.jl#L18-L28) |
| [IndBallL0 input precision](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/migration.md#L23-L25) could qualify every listed representation or only the symmetric sparse one | Place the homogeneous precision requirement before the list of representations. `precondition`. | [Source](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/indballl0.jl#L80-L88) · [Rejection tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/indballl0.jl#L324-L367) · [Supported inputs](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/indballl0.jl#L653-L680) |
| [KronQuadratic input precision](https://github.com/mlakolar/ProximalBase.jl/blob/d4edbeaef1538a0b2c2328b0f7a9b4c928857197/docs/src/migration.md#L26-L27) could qualify both representations or only the symmetric iterate | Put the matching `Float32` or `Float64` requirement before the dense and symmetric-iterate routes. `precondition`. | [Both input checks](https://github.com/mlakolar/ProximalBase.jl/blob/d4edbeaef1538a0b2c2328b0f7a9b4c928857197/src/functions/kronquadratic.jl#L149-L164) · [Dense and symmetric-iterate validation tests](https://github.com/mlakolar/ProximalBase.jl/blob/d4edbeaef1538a0b2c2328b0f7a9b4c928857197/test/kronquadratic.jl#L511-L549) |

## GroupNormLinf decisions

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Zero-group scores](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/group_penalty.md#L482-L489) could agree for every one-entry gradient or only when the margin is nonnegative | In exact arithmetic a zero margin gives zero distance. With one nonzero gradient entry a positive margin equals the distance. A negative margin gives zero distance. `characterized`. | [Margin and distance](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/groupnormlinf.jl#L422-L445) · [Signed margin](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/groupnormlinf.jl#L1117-L1131) · [Distance comparison](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/groupnormlinf.jl#L1344-L1363) |
| [Accuracy row](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/kkt_violation.md#L53-L54) could describe subtraction for ordinary sums or also describe overflow | A finite sum of magnitudes uses subtraction. If the sum overflows, the margin is `+Inf`, including when the coefficient is infinite. `guaranteed`. | [Before subtraction](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/groupnormlinf.jl#L436-L445) · [Overflow tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/groupnormlinf.jl#L1143-L1149) |
| [Ordered endpoints](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/groupnormlinf.md#L115-L127) could put invalid steps after identity cases or apply the ordered group rules after step validation | Direct array and compact calls validate the step before the per-group exits. `characterized` for the ordering. | [Array validation](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/groupnormlinf.jl#L347-L350) · [Compact validation](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/groupnormlinf.jl#L406-L415) · [Invalid-step sentinel tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/groupnormlinf.jl#L483-L500) |

## SqrNormL2 decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Infinite-level endpoint](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/sqrnorml2.md#L82-L88) could use “elsewhere” for every nonzero residual or only for nonzero residuals other than `NaN` | With finite `v` and `x` in the working type, a nonzero residual other than `NaN` gives an `Inf` entry score. A `NaN` center or a zero residual gives a `NaN` entry score, which propagates in the full-point maximum. `characterized`. | [Gradient arithmetic](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/sqrnorml2.jl#L203-L211) · [Point-aware score](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/sqrnorml2.jl#L382-L390) · [Separate endpoint tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/kkt_point.jl#L465-L474) |

## LeastSquaresLoss decisions

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Strided-vector range clause](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/leastsquares.md#L49-L52) could qualify the caller's coefficients or the operands of the kernel | The strided vectors are the kernel operands. Coefficient calls form the predictor in the loss's work array. Implementation note. | [Coefficient and kernel paths](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/leastsquares.jl#L87-L131) · [Guard tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/leastsquares_predictor.jl#L980-L1005) |
| [Matrix predictor range clause](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/leastsquares.md#L50-L52) could promise that both oracles retry an overflowing subtraction or describe a shared guard with different operands | The value-only oracle can retry from its supplied predictor and response. The fused oracle uses its stored residual, which can give an infinite value after subtraction overflow. `characterized`. | [Matrix oracle value routes](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/leastsquares_predictor.jl#L281-L313) · [Overflow test](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/leastsquares_predictor.jl#L1353-L1366) |

## SqrtLeastSquaresLoss decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Norm accuracy](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/sqrt_leastsquares.md#L56-L63) could compare with the loss at unrounded `Xβ` or with the mathematical norm at the stored or supplied kernel operands | Compare with the exact scaled norm for the stored or supplied predictor or supplied residual. The precision and layout hypotheses apply to the operands and the bound is stated for `Float32` and `Float64`. `characterized`. | [Coefficient evaluation](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/sqrt_leastsquares.jl#L78-L83) · [Proof](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/sqrt_leastsquares.jl#L114-L122) · [Tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/sqrt_leastsquares.jl#L310-L346) |

## LinearFunction decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Absolute error bound](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/linear.md#L72-L83) could count nonzero products or all positions | The error analysis counts the nonzero products entering the accumulator. Each zero product returns the accumulator unchanged. The bound applies in the stated homogeneous configurations and requires `n < 2^50` for `Float64`. `characterized`. | [Recovery analysis](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/linear.jl#L200-L218) · [Zero-product exit](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/numerical_kernels.jl#L319-L325) · [Sparse reference comparison](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/linear_sparse_value.jl#L614-L621) · [Other storage recovery tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/linear_value_recovery.jl#L5-L10) |

## NormL2 decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Symmetric value traversal](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/norml2.md#L43-L47) could count visits to a stored off-diagonal coordinate or its contributions to the squared norm | Each off-diagonal coordinate occupies one stored entry, with multiplicity two in the squared norm. Implementation note. | [Ordinary sum and recovery multiplicity](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/sqrnorml2.jl#L63-L94) · [Dense-reference comparisons](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/norml2.jl#L161-L173) |

## NormLinf decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Rounded-radius level](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/normlinf.md#L43-L55) could imply uniqueness at zero radius or only for positive radii | A positive rounded radius has one real root, which can be negative. At zero rounded radius the smallest root is the largest magnitude, giving an unchanged ideal clip at the represented input. `characterized`. | [Threshold analysis](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/normlinf.jl#L354-L368) · [Tiny and zero radius tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/normlinf.jl#L1505-L1515) |

## GroupNormL2 decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Infinite-step text](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/groupnorml2.md#L53-L68) could be a separate case or fall under the finite-step qualifier | Give the infinite step a separate row. With a positive finite level and weight, an infinite step fills the group with `+0`. `guaranteed`. | [Infinite-step exit](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/numerical_kernels.jl#L907-L923) · [Array result](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/groupnorml2.jl#L216-L217) · [Compact positive-zero result](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/groupnorml2.jl#L271-L276) |

## IndBallL2 decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [All-NaN projection](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/indballl2.md#L63-L67) could apply to every radius or only to positive finite ones | The all-NaN rule applies at a positive finite radius. A zero radius fills with positive zeros. An infinite radius copies the input. `guaranteed`. | [Radius branches](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/indballl2.jl#L171-L182) · [Radius endpoint tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/indballl2.jl#L239-L257) |

## LogDetLoss decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Positive-step input wording](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/logdet.md#L43-L50) could describe the finiteness check or promise success for every finite input | At a positive finite step the input check requires finite entries. `precondition`. Later working-range checks can reject a finite matrix before any writes. `characterized`. | [Input check and range failures](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/logdet.jl#L121-L169) · [Preserved destination test](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/logdet.jl#L195-L197) |

## SparseGroupNormL2 ownership decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [Copying groups and weights](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/sparsegroupnorml2.md#L18-L23) could promise a copy of each range object or only of the outer collection | The outer constructors copy the outer collection, supplied weights and ordinary index vectors. `guaranteed`. The concrete range element-type path keeps the individual range objects. Implementation note. | [Outer constructors](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/sparsegroupnorml2.jl#L47-L59) · [Concrete range path](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/block_iterate.jl#L119-L126) · [Ownership tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/sparsegroupnorml2.jl#L109-L136) |

## SparseGroupNormL2 decisions recorded in #1123

The owner approved three wordings and statuses in
[#1123](https://github.com/mlakolar/ProximalBase.jl/pull/1123):

| Source question | Merged clause | Status |
|:--|:--|:--|
| Symmetric iterate arguments | [Symmetric iterate arguments](../../src/spec/sparsegroupnorml2.md#sparse-group-norm-l2.oracles.symmetric-arguments) | `precondition` |
| Invalid proximal steps with no groups | [Invalid proximal steps](../../src/spec/sparsegroupnorml2.md#sparse-group-norm-l2.prox.invalid-step) | `guaranteed` |
| Classification after conversion to the working type | [Point-aware score classification](../../src/spec/sparsegroupnorml2.md#sparse-group-norm-l2.kkt.entry-classification) | `characterized` |

The following notes record the questions as written before that review.

- The SparseGroupNormL2 guide calls `SymmetricSparseIterate` unsupported
  for KKT. This could refer to the point argument alone or also to the
  supplied gradient. The point methods reject symmetric iterates, and the
  gradient checks accept a one-based real array of matching length. The
  tests exercise rejection for a symmetric point. The scope awaits the
  owner's decision. [Source](../../../src/functions/sparsegroupnorml2.jl)
  · [Tests](../../../test/sparsegroupnorml2.jl)

- The SparseGroupNormL2 endpoint table introduces its rules for each covered
  group. The negative or NaN step row could be read as conditional on there
  being a covered group or as a preflight for every call. The array oracle
  checks the step before processing groups. The tests cover the empty-group
  identity at a valid step and invalid-step rejection with groups. The
  scope awaits the owner's decision. [Source](../../../src/functions/sparsegroupnorml2.jl)
  · [Tests](../../../test/sparsegroupnorml2.jl)

- The SparseGroupNormL2 KKT endpoint paragraph says “For finite entries”.
  This could refer to stored inputs or to values after conversion to the
  working type. The score classifies converted gradient and point entries.
  The shared group-score tests include a finite integer point that becomes
  infinite in `Float16`. The scope awaits the owner's decision.
  [Source](../../../src/functions/sparsegroupnorml2.jl)
  · [Shared group-score tests](../../../test/kkt_block.jl)

## Scaling decisions

| Question | Approved wording and status | Evidence |
|:--|:--|:--|
| 3. IndNonnegative scaling and invalid scalar steps | Nonnegative scaling returns the indicator. The resulting scalar-step array call still validates the step. [Shared constructor rule](../../src/spec/scaled.md#scaled.constructor.indicator-identity), `guaranteed`. | [Original endpoint](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/indnonnegative.md#L46-L50), [constructor](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/scaled.jl#L107-L118) and [before-write tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/indnonnegative.jl#L581-L591) |
| 13. Zero scaling and finite loss data | Finite design and response let the two supported least-squares families simplify to `ZeroFunction()`. [Supported zero scaling](../../src/spec/scaled.md#scaled.constructor.zero-scale-losses), `guaranteed`. For losses without a supported zero-scale representation, the zero factor raises `ArgumentError`. [Default zero scaling](../../src/spec/scaled.md#scaled.constructor.unsupported-zero-scale), `characterized`. | [Original endpoint](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/scaled.md#L98-L103), [zero-scale methods](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/scaled.jl#L133-L159), [LeastSquaresLoss tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/numerical_recovery.jl#L1495-L1507) and [SqrtLeastSquaresLoss tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/sqrt_leastsquares.jl#L257-L261) |
| 15. Box rejection after scaling | Nonnegative scaling returns the original box, including `IndBallLinf`. Its scalar-step array oracles reject negative and `NaN` steps before writing. [Shared constructor rule](../../src/spec/scaled.md#scaled.constructor.indicator-identity), `guaranteed`. | [Original endpoint](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/docs/src/spec/indbox.md#L43-L47), [direct checks](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/src/functions/indbox.jl#L118-L123), [box tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/indbox.jl#L16-L64) and [thin-constructor tests](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/indballlinf.jl#L115-L131) |

## Owner review and next gates

The owner accepted the external Claude review prepared for PR #1021 at
`1e35bcf` against master at `2a55362`, dated 18 November 2026, satisfying the
Phase 2 step 3 review gate. The owner approved the fourteen-section layout and
shared-rule links for the remaining verbatim family moves. The review's scope,
its verification limits, the five requested corrections and the inspected
source revision are recorded in `review.md`.

Unrelated merged PR changes use the bounded `upstream` action with exact Git
provenance and independent review. The reconciliation through `8364532f`
has 66 receipts covering 132 transitions, retaining every earlier event and all
2,162 original block identities. Reconcile pending external changes to an
all-pass six-check report before starting a migration change.
The supplemental removed-lines report also protects later text during moves.
See `upstream.json`, `baseline.json` and `review.md`.

Heading compatibility retains explicit IDs and authored link/citation targets.
The uncited original API heading's two automatic fragments remain the only
approved retirements; the complete frozen snapshots remain unchanged. Later
valid retirements use the same data record and validations, without a Python
whitelist change. Phase 5 removes the temporary writing-guide instruction to
record these in the migration audit, retaining the durable forwarding rule.

[#1085](https://github.com/mlakolar/ProximalBase.jl/pull/1085) recorded
the sentence refinements for `smooth.md`, `group_penalty.md` and
`calculus.md` and the content identities for the three runtime-generated
admonitions. It also extended token and example accounting to ledgered
`reword` and `split` rows with exact reviewed allowances.
[#1122](https://github.com/mlakolar/ProximalBase.jl/pull/1122) added the
standalone preservation check for the published SparseGroupNormL2 passages.

Pointer-only sections identify their owning sections. Duplicate numerical
statements use a reviewed deduplication record when conversion consolidates
them. Phase 5 retires the temporary checks after final blocking acceptance
and retains the frozen audit and ordinary documentation gates.

New function families must place their numerical rules directly in
`docs/src/spec/<family>.md` using the fourteen-section schema and stable section
IDs. Algorithm steps and path-selection thresholds belong in a paragraph
labelled implementation notes, not guarantees. Thin constructors such as
`IndBallLinf` have a family-map row linking to their underlying owner, without a
separate contract page. The [family map](../../src/spec/index.md) has no entries in the
“Existing pages” state. [#1121](https://github.com/mlakolar/ProximalBase.jl/pull/1121)
completed the verbatim family moves with SparseGroupNormL2. Keep exact
receipts for the named inventory blocks and reconcile upstream edits
before the next preservation run.

The later `IndBallL1` and `SCAD` restores already have fourteen-section contract
pages and family-map/navigation entries. They add no existing-page migration
backlog. Existing explicit endpoint IDs remain unchanged when the other section
IDs are added.

Clause conversion, the remaining owner decisions and final acceptance are tracked
in [#1019](https://github.com/mlakolar/ProximalBase.jl/issues/1019).

## Test calibration decision

| Source question | Approved wording and status | Evidence |
|:--|:--|:--|
| [56-file sharding snapshot](https://github.com/mlakolar/ProximalBase.jl/blob/11fd9eff56e2bc15947f2eaccd746eb8cea6f0ed/test/TESTING.md#L309-L313) could assign the 40-second weight to the measured checkout or the merged PR | The measured checkout used 35 seconds. The Julia 1.13.4 run observed 38.971 seconds. #1063 landed with a provisional 40-second weight. Implementation note. | [Measured weight](https://github.com/mlakolar/ProximalBase.jl/blob/9ef0db982e264d00452c4877f1e58dd1d91778e3/test/tiers.toml#L103) · [Shard log](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44665776339/job/136054574400) · [Landed weight](https://github.com/mlakolar/ProximalBase.jl/blob/735de29173bd3e547c5d40039d3f27aece83d915/test/tiers.toml#L108) |
