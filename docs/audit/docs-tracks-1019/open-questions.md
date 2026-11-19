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

## Owner review and next gates

The owner accepted the external Claude review prepared for PR #1021 at
`1e35bcf` against master at `2a55362`, dated 18 November 2026, satisfying the
Phase 2 step 3 review gate. The owner approved the fourteen-section layout and
shared-rule links for the remaining verbatim family moves. The review's scope,
its verification limits, the five requested corrections and the inspected
source revision are recorded in `review.md`.

Unrelated merged PR changes use the bounded `upstream` action with exact Git
provenance and independent review. Eight receipts now cover fourteen transitions
from PRs #1020, #1022, #1023, #1024, #1026 and #1025. These include the original
heading's renames, a sentence that became three, the later `IndBallLinf`
inventory changes, the least-squares test description and the scaled-indicator
screen rule. No baseline identity is replaced. Reconcile pending external
changes to an all-pass six-check report before starting a migration change.
The supplemental removed-lines report also protects later text during moves.
See `upstream.json`, `baseline.json` and `review.md`.

Heading compatibility retains explicit IDs and authored link/citation targets.
The uncited original API heading's two automatic fragments remain the only
approved retirements; the complete frozen snapshots remain unchanged. Later
valid retirements use the same data record and validations, without a Python
whitelist change. Phase 5 removes the temporary writing-guide instruction to
record these in the migration audit, retaining the durable forwarding rule.

Before the next family, refine the three remaining family guides to sentence
units with original-block provenance, and replace the three uncited runtime
admonition hashes with exact page/category/title comparisons. Keep original
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

New function families must now place their numerical rules directly in
`docs/src/spec/<family>.md` using the fourteen-section schema. The four restores
already merged into the numerical hub increase the existing-page backlog from
13 to 17 families/operations. The fourth, `IndBallLinf`, is now in the family
map with its existing hub location. Named inventory blocks retain upstream receipts:
the API coverage check verifies presence, not their descriptive claims.
Batch several external changes when practical, before the next migration starts.

Clause conversion, the remaining families, numerical routine pages, lessons,
figures, the agent entry file and the final blocking-mode switch remain tracked
in [#1019](https://github.com/mlakolar/ProximalBase.jl/issues/1019).
