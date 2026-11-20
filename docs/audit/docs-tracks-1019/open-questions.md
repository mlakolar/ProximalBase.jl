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

- `docs/src/migration.md`, “Retained and removed scope”, can be read as a
  history of the #850 reduction or as an inventory of what is unavailable
  today. #1062 adds `IndBallL1` and `IndBallL0` to the indicator inventory.
  The current exports and [function library](../../src/function_library.md)
  include `NormL0`, `MCP` and `SCAD`. The scope of the statement about removals
  awaits the owner's decision.

- The `IndBallL0` sentence in the same section ends with “with `Float32` or
  `Float64`”. The phrase could qualify the whole list of input representations
  or `SymmetricSparseIterate` alone. The implementation uses those element types
  across the listed representations, and the tests exercise unsupported and
  mixed-type calls. The wording awaits the owner's decision.
  [Source](../../../src/functions/indballl0.jl) · [Tests](../../../test/indballl0.jl)

- The `IndNonnegative` endpoint row says “`Scaled` routes do not check and
  project”. This could refer to calls on a concrete `Scaled` wrapper or to
  calls on the result of `Scaled(f, α)`, including `α = 0`. The tests exercise
  projection at negative and `NaN` steps through `Scaled(f, 2)`, and confirm
  that `Scaled(f, 0)` returns the original indicator, whose public scalar-step
  oracles check the step. The moved row retains the original wording pending
  an owner decision. [Source](../../../src/functions/indnonnegative.jl)
  · [Scaling source](../../../src/functions/scaled.jl)
  · [Tests](../../../test/indnonnegative.jl)

- The `SqrNormL2` endpoint text says “`NaN` at a zero residual, `Inf`
  elsewhere” at infinite level with finite `v` and `x`. “Elsewhere” could
  refer to a nonzero residual that is not `NaN` or include a `NaN` center.
  The constructor accepts a `NaN` center and the score helpers propagate its
  `NaN`. The tests exercise `NaN` centers and infinite levels separately.
  The wording awaits the owner's decision.
  [Source](../../../src/functions/sqrnorml2.jl) · [Tests](../../../test/kkt_point.jl)

- The `LeastSquaresLoss` range text says “For strided `Float32` and `Float64`
  vectors”. This could describe the caller's coefficients `β` or the operands
  of the squared-residual kernel, including the predictor or residual. The
  coefficient methods use an internal work vector, and the fast value kernel
  dispatches on strided operands of a common type. The scope of the phrase
  awaits the owner's decision. [Source](../../../src/functions/leastsquares.jl)
  · [Tests](../../../test/leastsquares_predictor.jl)

## Owner review and next gates

The owner accepted the external Claude review prepared for PR #1021 at
`1e35bcf` against master at `2a55362`, dated 18 November 2026, satisfying the
Phase 2 step 3 review gate. The owner approved the fourteen-section layout and
shared-rule links for the remaining verbatim family moves. The review's scope,
its verification limits, the five requested corrections and the inspected
source revision are recorded in `review.md`.

Unrelated merged PR changes use the bounded `upstream` action with exact Git
provenance and independent review. The reconciliation through `5cf8b27f`
has 60 receipts covering 109 transitions, retaining every earlier event and all
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

Before partial-paragraph moves from `smooth.md`, `group_penalty.md` or
`calculus.md`, refine their units with original-block provenance. This does not
block the whole numerical subsections of `NormL1Nonneg` and `ElasticNet`.
Replacing the three uncited runtime admonition hashes with exact
page/category/title comparisons remains separate preparation. Keep original
explicit heading IDs and forwarding paragraphs for authored link/citation targets.
The next pages must label pointer-only
sections and use specific shared-rule anchors; verbatim duplicates stay until
their separately reviewed clause conversion.

Clause conversion is held until token and example checks also protect
`reword` and `split` rows, with exact recorded allowances for intentional
token changes and regression fixtures for missing numbers and assertions.
The migration ledger and six temporary checks are retired after the final
Phase 5 blocking acceptance; the frozen audit and ordinary documentation gates
remain. The preparation work is tracked in #1019.

New function families must place their numerical rules directly in
`docs/src/spec/<family>.md` using the fourteen-section schema and stable section
IDs. Algorithm steps and path-selection thresholds belong in a paragraph
labelled implementation notes, not guarantees. Thin constructors such as
`IndBallLinf` have a family-map row linking to their underlying owner, without a
separate contract page. The map now also includes the later `LogDetLoss`,
`LogisticLoss` and `GroupNormLinf` restores at their current guide owners.
It has 17 existing-page family/operation entries after the `NormL1Nonneg` and
`ElasticNet` moves. `IndNonnegative` is next in the approved order. Named
inventory blocks retain upstream receipts:
the API coverage check verifies presence, not their descriptive claims.
Batch several external changes when practical, before the next migration starts.

The later `IndBallL1` and `SCAD` restores already have fourteen-section contract
pages and family-map/navigation entries. They add no existing-page migration
backlog. Existing explicit endpoint IDs remain unchanged when the other section
IDs are added.

Clause conversion, the remaining families, numerical routine pages, lessons,
figures, the agent entry file and the final blocking-mode switch remain tracked
in [#1019](https://github.com/mlakolar/ProximalBase.jl/issues/1019).
