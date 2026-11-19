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

The owner completed the pilot review required by Phase 2 step 3 on
18 November 2026, reviewing PR #1021 at `1e35bcf` against master at `2a55362`.
The fourteen-section layout and shared-rule links are approved for the
remaining verbatim family moves. The five requested pilot corrections and
the inspected source revision are recorded in `review.md`.

Unrelated merged PR changes now use the bounded `upstream` action with exact
Git provenance and independent review. Three receipts each contain two verified
transitions, for merged IndBox PR #1020 and IndBallL2 PR #1022. Two more account
for NormL0 PR #1023, including one sentence that became three. Both earlier API
heading URLs are preserved by forwarding stubs. See `upstream.json` and
`review.md`.

Before the next family, refine the three remaining family guides to sentence
units with original-block provenance, and replace the three uncited runtime
admonition hashes with exact page/category/title comparisons. Keep original
heading IDs and forwarding paragraphs. The next pages must label pointer-only
sections and use specific shared-rule anchors; verbatim duplicates stay until
their separately reviewed clause conversion.

Clause conversion is held until token and example checks also protect
`reword` and `split` rows, with exact recorded allowances for intentional
token changes and regression fixtures for missing numbers and assertions.
The migration ledger and six temporary checks are retired after the final
Phase 5 blocking acceptance; the frozen audit and ordinary documentation gates
remain. The preparation work is tracked in #1019.

Clause conversion, the remaining families, numerical routine pages, lessons,
figures, the agent entry file and the final blocking-mode switch remain tracked
in [#1019](https://github.com/mlakolar/ProximalBase.jl/issues/1019).
