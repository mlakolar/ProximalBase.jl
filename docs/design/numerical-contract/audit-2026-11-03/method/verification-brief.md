# #556 audit — Phase 4 verification brief (read fully before starting)

You are one of 7 parallel verifiers in a historical-limitations audit of
ProximalBase.jl (Julia package of proximal operators). Extractors have already
pulled 749 candidate limitations out of every closed issue and closed PR. Your
partition is a JSON list of those candidates. Your job: deduplicate them into
behaviors, verify each behavior against CURRENT master, and classify it with
the rubric below. Accuracy over volume: a small verified list beats a long
unsubstantiated one.

## Fixed facts

- Current master SHA: `920ebece7cea104eba61bb1e9c751c347444c802`
  (version 0.6.0-DEV). Checkout (READ-ONLY, never modify, never commit,
  never `git config`): `/Users/mladen.kolar/projects/juliaPkg/ProximalBase.jl-issue556`
  (call it WT below). `git -C WT log`, `git -C WT log -S'...'`,
  `git -C WT show`, `grep -rn ... WT/src WT/test WT/docs` are fine.
- Julia: `julia` = 1.13.3, `julia +lts` = 1.10.12 (minimum supported).
  Run scripts as `JULIA_PKG_OFFLINE=true julia --project=WT yourscript.jl`
  (and `julia +lts ...`). One process per batch of probes: startup costs
  ~10 s, so put many probes in one script.
- Probe helper: `include("/tmp/pb556-audit/verify/vlib.jl")` gives
  `probe(label, thunk, watched...)`, which prints the result or the exception
  and whether each watched array/tuple changed (deep-copied before the call),
  and prints the SHA and Julia version as a header line.
- GPU semantics: JLArrays (a CPU-backed GPUArrays fake; checks dispatch and
  layout, NOT real-device behavior) is available in the prepared environment
  `/tmp/pb556-audit/verify/jlenv` (`julia --project=/tmp/pb556-audit/verify/jlenv`,
  it has ProximalBase dev'd from WT plus JLArrays, Adapt, GPUArraysCore). Real
  CUDA/ROCm hardware is NOT available.
- Put your scripts and outputs only under `/tmp/pb556-audit/verify/<YOUR-ID>/`.
  Keep the raw outputs (`*.out.113.txt`, `*.out.110.txt`) there.
- Do NOT run the package test suite (`scripts/test.jl`, `Pkg.test`); the
  coordinator runs it. You may cite current tests by `test/<file>.jl:<line>`
  after reading them.
- Do NOT post to GitHub, open issues, or edit any repository file.
- Already verified by the coordinator (reuse, don't redo): the seed cases in
  `/tmp/pb556-audit/verify/s1_seed.jl` with outputs
  `s1_seed.out.113.txt` / `s1_seed.out.110.txt` (identical on both
  versions): #216/#543 mechanism F (Conjugate finite-result check fires at
  VALID finite positive steps, e.g. `Conjugate(SqrNormL2())` at `[1e300]`,
  γ=1e-10, and γ=1e-310; ArgumentError standalone, earlier block written in
  SeparableSum/SliceSum), #323 (BoxConstrained inner and IndBox(-1e308,1)
  cases still write the earlier block; NaN input / γ=Inf on an EMPTY set
  return NaN with no error), #311 (BernoulliLoss/CubeNormL2 negative step
  DomainError after earlier block written; NaN step gives NaN output;
  NormL1/SqrNormL2/IndBallGroupL0/IndHyperplane/IndHalfspace/IndHyperslab/
  IndIsotonic accept negative and NaN steps; IndSimplex rejects), #239/#543 J
  (non-proximable LogisticLoss block: MethodError after earlier block written;
  old #239 rows Conjugate(ElasticNet) value, nested box, NuclearNorm Float16
  now throw before writing), #371 (tuple Conjugate InexactError after writing
  block 1, both cases), #255 (Conjugate(SeparableSum(NormL1,SqrNormL2)) at
  γ=1e-310 throws; Conjugate(IndHalfspace/IndHyperplane) at 1e-310 throw;
  Conjugate(IndBallL1/IndSimplex) at 1e-310 now succeed;
  `prox_entry(Conjugate(2*IndBox(0.0,Inf)),3.0,1,1e-310)` returns -Inf, exact
  0; spectral conjugates of SqrNormL2 at `[floatmax 0;0 1]`, γ=0.5 throw;
  array steps `[1e-310,1e-310]` on Conjugate(IndBallL1) rejected as "not
  positive and finite").
- Open issues/PRs (possible owners): `/tmp/pb556-audit/open-items.txt`.
  The 12 contract group trackers are #484 G1 kernels, #485 G2 NaN/Inf/
  degenerate parameters, #486 G3 steps, #487 G4 data functions, #488 G5 linear
  sets/searches, #489 G6 spectral/LAPACK, #490 G7 wrappers/value recovery,
  #491 G8 second-order, #492 G9 element types/constructors/caches, #493 G10
  tolerances, #494 G11 GPU, #495 G12 failure safety/aliasing/storage/testing;
  #366 is the umbrella. Prefer the most specific open issue as owner; use a
  group tracker only when no specific issue exists. Bundles of closed items:
  `/tmp/pb556-audit/items/NNNN.md`; open seed issues: `/tmp/pb556-audit/open/*.md`.
  Fetch other open-issue text with
  `gh issue view N -R mlakolar/ProximalBase.jl --json body,comments` if needed
  (read only).
- Contract: `WT/docs/src/numerics.md` (normative) and
  `WT/docs/design/numerical-contract/decisions.md` (each entry's **Decided**
  line governs). The summary of rule labels is in
  `/tmp/pb556-audit/EXTRACTION-BRIEF.md` ("Current numerical contract").
  Note: the O6 paragraph on block sums (numerics.md ~line 763, "In a sum over
  blocks...leaves the earlier blocks written") was added by PR #543 on
  2026-11-04 WITHOUT a Decided line; decision 42 says block functions decide
  before writing any block. Treat that paragraph as a description of current
  behavior, NOT as an approval.

## Procedure

1. Read your partition `/tmp/pb556-audit/verify/<YOUR-ID>-candidates.json`.
2. Cluster duplicates: one cluster = one underlying behavior (same API +
   mechanism). Candidates that are only statements of intent ("PR adds X"),
   or that describe fixes made inside the same PR before merge, still get a
   cluster (usually FIXED or CONFORMS) — every candidate id must map to
   exactly one cluster.
3. For each cluster:
   A. locate the current implementation (file:line);
   B. search later history for fixes (`git -C WT log --oneline -S'<symbol>'`,
      PR numbers in commit subjects, the candidate's `followups` and
      `possible_fix_refs`, and whether the owning issue is open or closed);
   C. read current tests and docs for it;
   D. reproduce on current master, adapting the old reproducer only as needed
      for the current API — record every adaptation;
   E. for CONFIRMED clusters run the reproducer on BOTH 1.13 and 1.10; for
      others 1.13 is enough unless version matters;
   F. compare with the CURRENT contract (not the expectation at the time);
   G. classify.
4. Write `/tmp/pb556-audit/verify/<YOUR-ID>.json` (valid JSON; check with
   `python3 -I -m json.tool`).

## Classification rubric (use exactly these labels)

- `CONFIRMED` — reproduced on current master AND violates the current contract
  (or the current docstring/docs promise) AND no approved decision permits it.
- `FIXED` — the historical defect no longer reproduces; you ran the current
  successful behavior (probe output) and/or name the positive regression test
  `test/...:line` that pins it, plus the fixing PR/commit if found. A later PR
  titled "Fix" is not enough by itself.
- `INTENTIONALLY_UNSUPPORTED` — the CURRENT contract or current user docs
  explicitly exclude the behavior (cite the clause: e.g. S1 complex/AD
  numbers, T6 Float16 by role, T7 LAPACK rejects BigFloat, O7 partial overlap,
  G6 unsupported backend raises, S3 not promised, a docstring that states the
  call is unsupported and raises). An old 0.5-era exclusion that the 0.6
  contract no longer states does NOT qualify; under S4 a function that misses
  a rule is a gap.
- `APPROVED_LIMITATION` — a maintainer-approved exception permits it. Cite the
  exact decision: a decisions.md entry number whose Decided line permits it
  (e.g. 29 IndIntersection value, 32 aliasing, 39 LogDetLoss transitional, 45
  wrapper argument overflow not covered), a numerics.md named exception (O3,
  the R1 wrapper limitation), or an explicit maintainer sign-off comment on a
  PR (quote + URL) for an accepted cost. Documentation alone is not approval.
- `UNVERIFIED` — status cannot be established here. Say precisely why: needs
  real GPU hardware, timing needs controlled hardware, reproducer not
  reconstructible from the record, contract ambiguous (say which clause),
  etc. Separate "not tested" from "known incorrect" from "unsupported".
- `OBSOLETE` — the API/architecture changed so the old behavior no longer
  applies (e.g. function removed, rule reversed by a later decision such as
  #287's NaN rule or decision 20). Name what replaced it.
- `DUPLICATE` — the same behavior as another cluster (yours or, if obvious,
  another partition's; name it by API + mechanism).
- `CONFORMS` — (audit-index only) on review, current behavior satisfies the
  current contract and was never a limitation under it (e.g. a documented
  approximate iterative method within A1, a statement of intent that was
  implemented as described). Use sparingly; prefer FIXED when a defect was
  actually repaired.
- `OUT_OF_SCOPE` — (audit-index only) infrastructure/process (CI, benchmark
  harness tooling, docs build, release) with no package-behavior effect.

For performance candidates: CONFIRMED only if you measured it (state method;
min-of-N `@elapsed` with warm-up; no BenchmarkTools available) and it exceeds
a contract threshold (C3); otherwise UNVERIFIED ("timing not re-measured on
controlled hardware") with its open owner, or APPROVED_LIMITATION when a
maintainer explicitly accepted the cost, or FIXED when a later PR measured and
fixed it AND the code path changed (cite).

## Output schema

```json
{
  "verifier": "<YOUR-ID>",
  "sha": "920ebece7cea104eba61bb1e9c751c347444c802",
  "julia": ["1.13.3", "1.10.12"],
  "clusters": [
    {
      "id": "<YOUR-ID>-NN",
      "title": "precise one-sentence statement of the behavior",
      "candidate_ids": ["C05-03", "C08-11"],
      "source_items": [211, 304],
      "api": "functions / wrappers / oracles",
      "conditions": "types, shapes, steps, values, backend",
      "contract_rule": "e.g. O6.1, P1, E2, R2 (current labels)",
      "expected": "what the CURRENT contract requires",
      "actual": "what current master does (observed)",
      "classification": "CONFIRMED | FIXED | INTENTIONALLY_UNSUPPORTED | APPROVED_LIMITATION | UNVERIFIED | OBSOLETE | DUPLICATE | CONFORMS | OUT_OF_SCOPE",
      "classification_reason": "cite decision / clause / fix PR + test; for UNVERIFIED the precise reason",
      "duplicate_of": "cluster id or description, only for DUPLICATE",
      "evidence": {
        "reproducer": "Julia code exactly as run (short)",
        "output_113": "key output lines",
        "output_110": "key output lines or 'not run: <reason>'",
        "adaptations": "changes to the historical reproducer, or 'none'",
        "test_refs": ["test/foo.jl:123"],
        "fix_refs": ["PR #N", "commit abcdef12"],
        "code_refs": ["src/foo.jl:45"]
      },
      "owner_issue": 323,
      "related": [216, 543],
      "priority_suggestion": "P1 | P2 | P3 — one-line reason (P1 = silently wrong result or broken documented guarantee on ordinary supported inputs; P2 = contract gap on exceptional/edge inputs or wrong exception class/partial write; P3 = cosmetic, very narrow, or test/perf only)",
      "release_impact_suggestion": "blocks 0.6 | candidate for deferral | not yet decided — one-line reason",
      "resolution_criteria": "smallest objective test that would show it resolved",
      "notes": "optional"
    }
  ],
  "candidate_map": {"C05-03": "<YOUR-ID>-01"},
  "phase5_findings": [
    "docs vs code contradictions, wrapper-vs-leaf differences, partial fixes, skipped tests, or contract gaps without any open owner that you noticed in your area (short, with file:line and evidence)"
  ]
}
```

Fill `owner_issue` for CONFIRMED and UNVERIFIED clusters (null + note "no
open owner" if none exists — that is itself a finding). `candidate_map` must
contain every candidate id in your partition. Keep text concise; no long
quotes from discussions.

When done, reply with at most 8 lines: counts per classification, the
CONFIRMED cluster titles (one line each), and any blocker.
