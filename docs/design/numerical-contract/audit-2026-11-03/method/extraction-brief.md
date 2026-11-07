# #556 audit — Phase 3 extraction brief (read fully before starting)

You are one of 15 parallel extractors in a historical audit of ProximalBase.jl
(a Julia package of proximal operators; repo mlakolar/ProximalBase.jl).
The audit answers: which limitations reported in CLOSED issues and CLOSED PRs
still exist today? Your job is ONLY extraction (Phase 3). Another agent will
verify every candidate against current master later. Do not run Julia, do not
open the repository source, do not post anything to GitHub, do not modify any
file except your own output file.

## Inputs

- Your item list: the numbers given in your prompt. Each item is a cached,
  chronological bundle at `/tmp/pb556-audit/items/NNNN.md` (4-digit,
  zero-padded): header (title, URL, closed/merged, closing refs,
  cross-references), body, then every issue comment, submitted review body and
  inline review comment (with `(id N)` / `(discussion_rN)` and thread
  resolved/outdated flags), in time order.
- `/tmp/pb556-audit/open-items.txt`: the currently OPEN issues and PRs, with
  titles. Use it to name an existing open owner when the item's discussion
  defers something to it.
- Read EVERY item in your list completely, including long run reports and the
  last review comments before a merge (older PRs often hold substantial
  unresolved findings in their final review discussion).

## What counts as a candidate limitation

A statement in the item (body, comment, review, inline comment) that some
behavior of the PACKAGE was left incomplete, wrong, unsupported, untested or
inconsistent at the time. Discovery signals (not proof): "remaining
limitations", "known issue", "not fixed", "deferred", "out of scope",
"follow-up", "unsupported", "not tested", "still fails", "partial",
"numerical instability", "wrong on exceptional inputs (NaN/Inf/zero/subnormal/
overflow)", "writes output before throwing", "missing validation", "missing
oracle/method", "MethodError", "performance regression", "@test_broken /
skipped test", "docs contradict implementation", "pre-existing", "not covered",
"left for", "tracked in #N", "separate issue".

Rules:
- Do NOT discard a candidate because the PR merged or the issue closed.
- Do NOT treat a candidate as real just because a review called it a
  limitation; record what was claimed and whether the thread itself later
  resolved it (e.g. a follow-up commit in the same PR fixed it, or the
  reviewer withdrew it). Record that in `status_in_thread`.
- One candidate = one behavior. Split compound reports. Merge duplicates
  WITHIN your chunk into one candidate with several `sources`.
- If the item defers the behavior to another issue/PR (open or closed), still
  record the candidate and list that number in `followups`.
- Infrastructure-only limitations (CI workflows, benchmark harness tooling,
  docs build, release process) get `category: "infra"` and a one-line record;
  don't spend effort on them.
- A deliberate design decision that excludes a behavior ("we do not support
  X", "not planned") is still a candidate: category as appropriate and
  `status_in_thread: "excluded-by-decision"`, quoting the decision.
- Keep quotes short (<= 300 chars) and exact. Do not copy large discussions.

## Current numerical contract (docs/src/numerics.md), labels to use in `contract_rule`

- S1 scope: real Float32/Float64 on CPU; Float16 rejected (T6); BigFloat per T7;
  GPU per G-rules; complex rejected; AD numbers not covered. S4: one rule for
  the package; a function that does not meet a rule is a gap, not a documented
  limitation (only stated limitation: R1 wrappers whose transformed argument
  overflows, except norms in Translate/Precompose via homogeneity).
- T1 return/destination types; T2 data vs parameters; T3 model precision
  (widest of delivery, argument, data; parameters rounded once after
  combining); T4 wider destination computed in D; T5 constructors validate
  before converting, never narrow/turn finite positive into 0/Inf; T6 Float16
  rejected by role before writing; T7 BigFloat (caches keyed by precision,
  LAPACK rejects BigFloat); T8 Float64 bookkeeping.
- R1 normal values returned despite intermediate overflow/underflow; R2 same for
  gradient entries and closed-form prox entries; R3 conversions keep range;
  R4 final range, no saturation.
- A1 per-output guarantee; A2 decisions (exact entrywise, same on every
  path/backend); A3 search decisions certified, else PrecisionError before
  writing; A4 tolerances precision-aware (no fixed default); A5 LAPACK
  functions (non-finite never passed to LAPACK, scaling, element types
  checked); A6 reductions.
- E1 order: structure (shapes/types/steps) first, raising DimensionMismatch /
  ArgumentError before writing; E2 NaN propagates to dependent results only,
  never becomes zero/Inf through a failed comparison; E3 infinite entries take
  determined limits, else NaN (decision 48: staged ball/sphere/simplex limits);
  E4 zero/infinite parameters; NaN parameters and non-finite LinearFunction/Tilt
  coefficients rejected at construction; E5 exact Inf/NaN survive conversions;
  E6 out-of-domain value +Inf, gradient NaN.
- P1 negative/NaN steps -> ArgumentError before writing; zero/Inf steps only at
  named endpoints, else rejected before writing; P2 step types/routes agree;
  P3 diagonal metrics not representable in W (scale times ratios).
- O1 value_and_gradient! agrees with standalone oracles; O2 value_and_prox!
  returns the value at the stored output (no false zero/feasibility/class
  change); O3 named exceptions (IndIntersection approximate, LogDetLoss
  transitional); O4 Translate/Tilt staging bound; O5 restrict/conjugate
  identities with conditions.
- O6 failure semantics: O6.1 validation failures (shapes, types, steps,
  parameters) raise before ANY write, every caller-owned output unchanged
  incl. tuple blocks and iterate storage; O6.2 infeasibility found during a
  search -> ArgumentError before writing; O6.3 uncertifiable/unrepresentable
  result -> PrecisionError before writing; O6.4 external library failure
  propagates, documented. O6 block paragraph: in SeparableSum/SliceSum/FiberSum
  only kind-1 failures checked before writing are detected before the first
  block is written; a later block's kind 2/3/4 failure (e.g. Conjugate's
  finite-result check) leaves earlier blocks written.
- O7 aliasing: only y === x guaranteed for prox!/value_and_prox!; any other
  overlap unsupported, need not be detected; gradient/Hessian outputs must not
  alias inputs unless documented.
- M1 implicit zeros of sparse iterates are entries; M2 SymmetricSparseIterate
  off-diagonal multiplicity; M3 caches.
- G1-G6 devices: same meaning as CPU; G6 unsupported raises; JLArrays checks
  dispatch only. C1-C4 cost (C3: >5% slowdown investigated; >10% narrow-only
  needs approval).
- Use "none" if no numerical-contract rule applies (e.g. API feature gaps,
  iterate-family semantics outside numerics, docs), or "iterate-design" for
  0.6 iterate-family design rules (FD-x/FN-x in docs/design/iterate-family-0.6.md).

## Output

Write ONE JSON file: `/tmp/pb556-audit/extract/chunk-XX.json` (XX = your chunk
id), valid JSON (check it with `python3 -I -m json.tool <file> >/dev/null`),
with this shape:

```json
{
  "chunk": "XX",
  "items": [
    {
      "number": 216,
      "kind": "issue" | "PR",
      "what_it_did": "one line: what the issue reported / what the PR changed",
      "residual": "none" | "candidates" | "infra-only" | "decision-record",
      "item_hint": "implemented-no-residual | residual-extracted | superseded | duplicate-of-#N | not-planned | process-infra | design-record | unmerged-abandoned | unmerged-superseded-by-#N",
      "candidate_ids": ["C05-03"],
      "notes": "optional, <= 2 lines"
    }
  ],
  "candidates": [
    {
      "id": "CXX-NN",
      "sources": [
        {"item": 543, "ref": "inline review comment by X 2026-.. (discussion_r123) | comment id 456 | body | review by X 2026-..",
         "url": "https://github.com/mlakolar/ProximalBase.jl/pull/543#issuecomment-456 (or #discussion_r123, or the item URL for body/review)",
         "quote": "exact short quote"}
      ],
      "api": "exact functions/wrappers/oracles affected (e.g. prox! of SeparableSum with a Conjugate block)",
      "behavior": "what was reported to happen",
      "trigger": "conditions: values, steps, shapes, types, backend",
      "types_backend": "e.g. Float32 output, Float64 data; CPU | JLArray | CUDA | any",
      "expected_then": "what was expected at the time",
      "reproducer": "verbatim Julia snippet from the source if given, else a minimal proposed one marked '# proposed'",
      "category": "correctness | exceptional-values | precision-range | step-validation | failure-atomicity | aliasing | unsupported-type | sparse-iterate | wrapper | gpu | lapack | performance | tests-skipped | docs-mismatch | missing-feature | infra",
      "contract_rule": "e.g. O6.1, P1, E2, R1, none, iterate-design",
      "status_in_thread": "open-at-close | resolved-in-thread | deferred-to-#N | excluded-by-decision | withdrawn | unclear",
      "followups": [323, 239],
      "possible_fix_refs": ["PR #N or commit sha mentioned as fixing it, if any"],
      "confidence": "high | medium | low  (that the report describes a real behavior at the time)"
    }
  ]
}
```

Every item number in your list must appear exactly once in `items`. Candidate
ids are `C<chunk>-<2-digit seq>`. Return to the caller only a 5-line summary:
items processed, candidates extracted, and the 3 most significant candidates.
