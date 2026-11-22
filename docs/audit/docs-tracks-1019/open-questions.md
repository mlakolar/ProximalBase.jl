# Documentation migration open questions

## Numerical policy

No numerical-policy discrepancy was identified in the verbatim `NormL1` pilot
against source and tests at `7ae86e2093060eb124211e2f21f043c6379f97ae`.
The pilot preserves existing guarantee, characterization and limitation wording.
It assigns no new clause statuses and changes no source or test behavior.

If a later migration exposes an ambiguity, record the source block, both possible
readings, source/test evidence, and the owner's decision here before converting
that statement into a classified clause. Do not infer a decision from an example
that happens to pass.

- The `LeastSquaresLoss` range text says “For strided `Float32` and `Float64`
  vectors”. This could describe the caller's coefficients `β` or the operands
  of the squared-residual kernel, including the predictor or residual. The
  coefficient methods use an internal work vector, and the fast value kernel
  dispatches on strided operands of a common type. The scope of the phrase
  awaits the owner's decision. [Source](../../../src/functions/leastsquares.jl)
  · [Tests](../../../test/leastsquares_predictor.jl)

- The matrix predictor sentence added by #1115 says both oracles reuse “the same
  squared-residual recovery”. This could refer to both oracles retrying an
  overflowing subtraction or to their use of the same range kernel on different
  operands. The value-only oracle can retry with the original predictor and
  response. The fused oracle uses the stored residual, and the overflow test
  gives a finite value for the former and `Inf` for the latter. The scope of
  “same” awaits the owner's decision.
  [Source](../../../src/functions/leastsquares_predictor.jl)
  · [Tests](../../../test/leastsquares_predictor.jl)

- The `SqrtLeastSquaresLoss` range text uses “`v` the exact value”. This
  could mean the loss from unrounded `Xβ` or the exact norm at the stored or
  supplied predictor or residual. The same paragraph leaves open whether
  the precision and layout conditions apply to the caller's coefficients and
  design or to the operands of the sum. Its length-and-epsilon condition can
  also be read to cover precisions beyond `Float32` and `Float64`, which the
  kernel proof names. The scope of the bound awaits the owner's decision.
  [Source](../../../src/functions/sqrt_leastsquares.jl)
  · [Tests](../../../test/sqrt_leastsquares.jl)

- The `LinearFunction` range paragraph gives the bound
  `B = 4 n eps(T) (sum |cᵢxᵢ| + |d|)` without defining `n`. It could count
  nonzero products or all entries in the argument or its storage. The recovery
  tests count nonzero products. The intended scope awaits the owner's decision.
  [Source](../../../src/functions/linear.jl)
  · [Tests](../../../test/linear_value_recovery.jl)

- The `NormL2` range text says the sums visit “each off-diagonal symmetric
  coordinate twice”. This could count contributions to the squared norm or
  visits to the stored entry. The ordinary sum doubles a stored off-diagonal
  square, and recovery uses a multiplicity. The wording awaits the owner's
  decision. [Source](../../../src/functions/sqrnorml2.jl)
  · [Tests](../../../test/norml2.jl)

- The NormLinf range paragraph calls τ* the root of the level equation
  at the rounded radius. If the radius rounds to zero, every level at or above
  the largest magnitude solves the equation. The statement could refer to the
  smallest root or apply only when the rounded radius stays positive. The
  implementation uses the largest entry as the zero-radius reference in its
  exceptional-path derivation. The range interpretation awaits the owner's
  decision. [Source](../../../src/functions/normlinf.jl)
  · [Tests](../../../test/normlinf.jl)

- The GroupNormL2 endpoint row for positive finite parameters and step also
  says `γ = Inf` fixes the group at zero. It could describe a separate
  infinite-step branch or fall under the row's finite-step qualifier. The
  guide table and infinite-step tests exercise the branches separately.
  The qualifier awaits the owner's decision.
  [Guide](../../src/group_penalty.md#group-l2-endpoints)
  · [Tests](../../../test/groupnorml2.jl)

- The IndBallL2 endpoint row says multiple infinite entries or any NaN give
  an all-NaN projection. That sentence could inherit the positive finite
  radius qualifier or apply to all radii. The tests check zero-radius fill,
  infinite-radius copy, and positive finite radius separately. The qualifier
  awaits the owner's decision. [Tests](../../../test/indballl2.jl)

- The LogDetLoss endpoint paragraph says a positive finite step accepts
  finite inputs. It could describe the input check or imply that every
  finite matrix yields a result. The extreme-range test uses a finite
  matrix and checks a failure before writes. The scope of “accepts”
  awaits the owner's decision. [Source](../../../src/functions/logdet.jl)
  · [Tests](../../../test/logdet.jl)

- The SparseGroupNormL2 definition says the constructor “copies the
  supplied groups and weights”. For a collection with a concrete range
  element type, this could promise an independent copy of each supplied
  range or a copy of the outer collection. `_own_groups` copies the outer
  vector and retains the range objects. The ownership test mutates the
  supplied index vectors and weight vector. The copy requirement for
  range objects awaits the owner's decision.
  [Definition](../../src/spec/sparsegroupnorml2.md#spec-sparsegroupnorml2-definition)
  · [Ownership implementation](../../../src/block_iterate.jl#L119-L127)
  · [Tests](../../../test/sparsegroupnorml2.jl#L109-L136)

- The GroupMCP zero-group screen describes the margin “for finite `v`,
  `λ0` and `w_k`”. Finiteness of the gradient could refer to the stored
  entries or to the entries after conversion to the working type. The
  shared screen classifies the converted entries. Tests of the shared
  screen exercise finite integers that become infinite after conversion.
  The wording awaits the owner's decision.
  [GroupMCP failure behavior](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/docs/src/spec/groupmcp.md#L148-L153)
  · [GroupMCP dispatch](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/src/functions/groupmcp.jl#L514-L533)
  · [Shared screen](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/src/numerical_kernels.jl#L1134-L1145)
  · [Mixed-width tests](https://github.com/mlakolar/ProximalBase.jl/blob/a6b16d63d9c5102b1ac82d9995014145681d4343/test/kkt_block.jl#L1233-L1240)

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

The #1063 test inventory says the historical snapshot at `879e067` “added a
provisional 40-second weight”. This could refer to the measured checkout or
to the later weight based on its timing. The measured checkout used a weight
of 35 seconds and #1063 landed with 40 seconds. The wording awaits the owner's
decision. [Test inventory](../../../test/TESTING.md#ci-sharding)

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
