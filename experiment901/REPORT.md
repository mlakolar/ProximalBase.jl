# #901 architecture experiment: comparative decision report

Status: **recommendation for maintainer approval. Nothing integrated into #901**; no merge, no PR change. The experiment
branches below are pushed to `origin` for review; no pull request was opened for them.
Date: 2026-11-13. Orchestrator: Opus; four Sonnet implementers (high effort) worked in parallel in separate worktrees.

## Recommendation

**Adopt design C, "optimized consolidation", in its Y1 form: commit `64cab3f1` (branch `exp/C-y1`, on top of
`582932fb`, branch `exp/901-C-optimized-consolidation`).**

- It keeps the #865/#901 architecture: one shared ℓ2 kernel layer, and one copy each of the scaled norm, the
  exponent sum and the threshold product.
- It has the smallest change from `0eda76a9`: `src` +133 / −42 lines in 3 files.
- It removes the measured `0eda76a9` regressions and keeps or extends its cold-path gains.
  - Comprehensive campaign: cold-path geomean **0.720** against base (`ref` 0.905), `l2_norms` 0.991.
  - Screen: geomean 0.767 against base, with 0 of 826 cases above 1.05× in all 6 processes.
- Accuracy is identical to base, all 30 quick-tier files pass, and nothing allocates more than base.

It needs approval for two things:

1. **The screen-alignment exception.**
   - What it is: the small-group overflow/underflow `kkt` fix (1.15–1.38× at `ref`, 0.9–1.0× in Y1) comes
     entirely from forming the group screen's two-term difference directly in the shared `_l2_screen_cold`,
     instead of through `_pow2_sum`.
   - Why it conflicts: this is the hand-aligned form the #901 D-MPFR ruling said not to restore. The ruling's
     motive was BigFloat; here the motive is speed for Float32/Float64.
   - Effect: with `_pow2_sum` back (variant V4, measured), groups of 1, 2 and 5 return to the `ref` level.
   - Every design that fixes this case does it the same way: B, C and D directly, A by restoring base's own
     screen. D-X is kept. D-MPFR would no longer apply to the group screen. LinearFunction and
     LeastSquaresLoss keep `_pow2_sum`.
   - Options: (a) grant the exception; (b) keep `_pow2_sum` and approve 1.15–1.38× on small-group
     overflow/underflow `kkt`.
2. **Two residual families above 5%**, repeatable in the campaign (both role orders):
   - **Zero groups with groups of 2–5:** `value` and fused at about 1.08–1.13×, in both the screen and the
     campaign. On ordinary data with 3 of 4 groups zero (`l2_norms` "sparse"), the geomean is 0.989 but single
     cases reach about 1.05.
   - **`prox_block` with parameter-decided exits:** Float32, groups of 5 (λ0 = 0/Inf, γ = 0, w = Inf), about
     1.08–1.14×. Y1's once-per-call exit decision covers `prox!`/`value_and_prox!`, not `prox_block!`.

   Options: approve them as they are, or run one more targeted round before integration. I recommend the
   targeted round, scoped to these two families with a 3-screen budget:
   - Candidate ideas: B's data-free loop for parameter-decided regimes, applied to `prox_block!`; A/D's
     base-style zero test in the value path.
   - Neither idea has been measured in C yet. C's own attempt (V1, restoring `_all_zero`) did not fix the
     zero-group case with groups of 1.

Fallbacks:

- **A (`2fde7b13`)** if a strict "no measurable slowdown anywhere" gate outranks the architecture. It runs at
  base speed (its campaign profile is the noise floor). It gives up `ref`'s gains, re-duplicates three recovery
  algorithms, and turns the D-X pins into `@test_broken`.
- **D (`2971e5fe`)** is clean on ordinary paths. It has 1.18–1.34× on Float64 groups of 5 with
  subnormal/underflow `prox`, and more code than C.
- **B (`02dba9e3`) is rejected.** Its screen geomean of 0.57 hides up to **2.0×** regressions that only the
  comprehensive campaign exposed (mixed-weight `value`, λ0 = 0 with infinite weights in `value_and_prox`,
  γ = 0). It also adds 342 lines.

## Trees

| Label | Commit | Branch / worktree (`/home/mkolar/projects/juliaPkg/pb901x/…`) | What it is |
|---|---|---|---|
| base | `f36a6d6a` | `exp/901-baseline`, `base` | performance baseline: `76964085` + master's `linear.jl` (as in the #901 campaign) |
| ref | `0eda76a9` | detached, `ref` | frozen #901 head at experiment start |
| active | `04821b7e` | detached, `active` | the active #901 agent's in-progress repair (comparator only, untouched) |
| A | `2fde7b13` | `exp/901-A-original-fastpaths`, `A` | original fast paths restored |
| B | `02dba9e3` (screened src `71c4b37b`) | `exp/901-B-shared-arith-separate-loops`, `B` | shared arithmetic, separate loops |
| C | `582932fb` (screened src `dd9cc043`) | `exp/901-C-optimized-consolidation`, `C` | optimized consolidation |
| **Y1** | **`64cab3f1`** | `exp/C-y1`, `C-y1` | C + per-call parameter exits, one out-of-line loop per case |
| D | `2971e5fe` | `exp/901-D-minimal-primitive`, `D` | minimal shared primitive |
| harness | `bfeddb37` + this report | `exp/901-harness`, `harness` | methodology, harnesses, scripts, results |

C's unadopted experiment branches: `exp/C-v1`, `-v2`, `-v3`, `-v4`, `-w1`, `-w2`, `-x1`, `-x3`, `-x5`. All experiment
branches are pushed to `origin`; none has a pull request.

## Methodology (summary; full text in METHODOLOGY.md)

- **Hardware and settings:** Discovery `oneweek`, `--constraint=xeon-2640v4` only, exclusive nodes, Julia
  1.13.3, one thread, `--check-bounds=no`, pinned cores.
- **Paired, in-process design:** each process compiles base twice (X and the A/A control) and the other
  trees; inputs are shared and role order rotates. Every comparison has forward and reversed role orders in
  fresh processes.
- **Calibration** (jobs 14217165 and 14217167):
  - Running 5, 10 or 20 concurrent processes per node lowers absolute speed by 7%, 15% or 28% (turbo), but
    moves paired ratios by only about 1%.
  - Single cases vary by 5–10% between processes, so screens use 3 or more processes per order.
- **Screen** (`screen901.jl`):
  - 826 cases: GroupNormL2 with groups of 1, 2, 5, 32 and 1024 × 11 regimes (ordinary, zero / mixed weights,
    zero groups, overflow, underflow, threshold over/under, Inf, NaN, infinite level) × `value`, `prox!`,
    fused, `prox_block!`, `kkt`; NormL2 with n from 1 to 65536; consumers (SqrNormL2, `norm_diff`, NormL1).
  - About 95–150 s per process.
- **Comprehensive campaign:**
  - `l2_cold_paths.jl`: 5,046 cases, with the bitwise check replaced by a coarse sanity check, in 40 shards.
  - `l2_norms.jl`: 368 cases, in 8 shards.
  - One forward and one reversed job per tree, about 34–35 min of wall time each.
  - With only 2 processes per case, placement noise produces false "slower in both" cases. A, which runs
    base's ℓ2 code, measures that floor: 28 cold-path cases above 1.05× and 2 above 1.10×.
- **Accuracy** (`accuracy901.jl`):
  - Independent BigFloat-512 references from the formulas.
  - Bounds: value `2(n+4)eps|e|`; prox entries `(m+8)eps|x_i|` with a required exact zero below
    `(1-4(m+4)eps)·threshold`; screen `(m+8)eps(‖g‖+λw)`.
  - Exact class checks for NaN/Inf, zero / infinite weights, infinite level and step; the alias identity.
  - No candidate may exceed max(2×, +0.25 bound) of base's error per cell.

## Results

### Correctness (orchestrator-run, independent of the implementers)

| Tree | Quick tier, Discovery, 1.13.3 | Accuracy (fresh seed 4242) | Allocation regressions |
|---|---|---|---|
| A | 30/30 (job 14218417) | 0 violations, 0 class failures, 0 regressions | 0 |
| B | 30/30 | 0 / 0 / 0 | 0 |
| C | 30/30 (job 14218415) | 0 / 0 / 0 | 0 |
| Y1 | 30/30 | 0 / 0 / 0 | 0 |
| D | 30/30 (job 14218416) | 0 / 0 / 0 | 0 |

- Max errors per family, op and type are identical to base for all designs. For example, Group prox is 0.125
  bound units and Group value 0.0139 (Float64).
- The only coarse-sanity difference in the campaigns is the documented BigFloat D-X case. A matches base
  there; the others keep the widened sum.
- Test changes:
  - C, Y1 and B changed no existing test.
  - A and D replaced `_IndexedValues` tests with tests of their own kernels.
  - A turned the two D-X pins into `@test_broken`.
- Julia 1.10 was not run for any design.

### Screens (paired ratio to base; median over processes)

Joint screens (jobs 14218311–3 and 14220936; all designs in the same processes) and the Y1 screen (job 14228110):

| Tree | geomean | cases >1.02 / >1.05 in all runs | worst |
|---|---|---|---|
| ref | 0.933 | 71 / 52 | 3.16 |
| A | 0.945 | 0 / 0 | 1.04 |
| B | 0.568 | 6 / 0 | 1.08 |
| C | 0.769 | 6 / 1 | 1.15 |
| **Y1** | **0.767** | **1 / 0** | **1.06** |
| D | 0.888 | 1 / 0 | 1.07 |

Target cases (ratio to base, Float64, range groups; from the joint screen, and the Y1 screen for Y1):

| Case | ref | A | B | C | Y1 | D |
|---|---|---|---|---|---|---|
| zero-weight `value`, groups of 1 / 2 / 5 / 32 | 2.0–3.1 | 0.96 | 0.16–0.29 | 0.60–0.67 | 0.63 / 0.61 / 0.60 / 0.37 | 0.91 |
| zero-weight fused, groups of 5 | 1.31 | 0.95 | — | 0.47 | 0.43 | — |
| overflow `kkt`, groups of 2 / 5 | 1.40 / 1.30 | 1.0 | 0.99 | 0.91 / 1.02 | 0.91 / 1.03 | ≈1.0 |
| infinite-level `prox`, groups of 1–5 | 0.93–0.96 | 1.0 | — | 1.04–1.16 | **0.82–0.97** | 0.93–0.98 |
| ordinary `prox`, groups of 1024 | 1.00 | 1.00 | 0.99–1.01 | 1.00 | 1.01 | 1.00 |
| groups of 1024, exceptional categories | 0.36–0.97 | ≈1.00 | 0.26–0.73 | 0.34–0.85 | 0.34–0.85 | 0.86–0.99 |
| NormL2 Inf / NaN categories | 0.93 / 0.35 | 0.55 / 0.56 | — | 0.37 / 0.38 | 0.37 / 0.38 | 0.83 / 0.31 |

### Comprehensive campaign (one forward and one reversed job per tree, 2 runs per case)

| Tree | cold-path geomean | >1.05 / >1.10 / >1.20 / >1.50 in both | worst | `l2_norms` geomean | >1.05 in both | worst |
|---|---|---|---|---|---|---|
| ref | 0.905 | 315 / 219 / 149 / 16 | 2.87 | 0.991 | 1 | 1.06 |
| A (noise floor) | 0.945 | 28 / 2 / 0 / 0 | 1.13 | 1.000 | 2 | 1.25 |
| B | 0.573 | 61 / 35 / 22 / **14** | **2.02** | 0.981 | 10 | 1.14 |
| C | 0.723 | 61 / 6 / 0 / 0 | 1.16 | 0.986 | 4 | 1.11 |
| **Y1** | **0.720** | **45 / 3 / 0 / 0** | **1.12** | 0.991 | 2 | 1.20* |
| D | 0.881 | 19 / 13 / 5 / 0 | 1.34 | 0.997 | 0 | 1.04 |

\*Y1's two worst `l2_norms` cases come from noisy processes (A/A 1.195 and 0.870 in the same runs).

Where each design's cases above 1.05× in both runs fall:

- **Y1:** 20 in `prox_block` with parameter-decided exits; 14 in zero / negative-zero group `value` and fused
  (groups of 2–5); 7 in `prox`/fused with parameter exits; 4 other.
- **D:** 14 in Float64 groups of 5 with subnormal / underflow / threshold-underflow `prox` and fused
  (1.10–1.34).
- **B:** mixed-weight `value` with groups of 5 and 32 up to 2.0×; λ0 = 0 with infinite weights in fused,
  groups of 32, 1.5–1.8×; γ = 0 fused, about 1.46×.

### Complexity

| Tree | `numerical_kernels.jl` | `norml2.jl` | `groupnorml2.jl` | total | `src` diff vs ref | recovery algorithms duplicated |
|---|---|---|---|---|---|---|
| base | 727 | 333 | 347 | 1407 | — | (pre-consolidation) |
| ref | 1112 | 88 | 310 | 1510 | — | 0 |
| A | 835 | 339 | 347 | 1521 | +312 / −301 | 3 (scaled norm, single-rounding product, screen difference) |
| B | 1021 | 194 | 547 | 1762 | +456 / −204 | 0 (+ a second form of the screen difference); family loops duplicated |
| C | 1153 | 88 | 331 | 1572 | +120 / −41 | 0 (screen difference written directly) |
| **Y1** | 1154 | 88 | 342 | 1584 | **+133 / −42** | 0 (same) |
| D | 928 | 188 | 457 | 1573 | +312 / −249 | 0 recovery, control flow per family (screen restored to base form) |

## Does the best approach combine implementers' ideas?

- **Within C:** yes. Y1 is C plus the per-call parameter decision that C proposed and that B used as
  data-free loops. It was measured, and it fixes C's gate failures without the subnormal-multiply trap that
  the in-loop variants hit.
- **Across designs:** two further grafts look promising for the residuals, but neither is measured, so
  neither is recommended yet:
  - B's data-free loop for parameter-decided regimes, applied to `prox_block!`;
  - A/D's base-style zero test in the group value path.
- **Shared by B, C and D:** all three converged on the direct screen difference. That is the decision item
  above, not a combination.
- **Ideas that did not transfer:** B's very low cold-path ratios come from duplicating the loops. The campaign
  shows that this structure is fragile outside the screened regimes, so its loop architecture should not be
  imported wholesale.

## Lessons for the methodology

- **Screen–campaign gap:** the 826-case screen was enough to rank designs on the targeted regressions, but it
  missed B's 2.0× cases (mixed weights, λ0 = 0 with infinite weights). Screen first, but never decide without
  the comprehensive campaign.
- **Noise floor:** with 2 processes per case, "slower in both" counts need a noise-floor control; A, which
  runs base's own ℓ2 code, served as one.

## Discovery resources and wall times

60 jobs, about 10.9 node-hours and about 201 core-hours (Xeon E5-2640 v4, 20 cores per node; tests on `main`):

| Kind | Jobs | Node-hours | Core-hours | Typical wall time |
|---|---|---|---|---|
| Comprehensive campaigns (A, B, C, D, Y1, ref, calibration) | 13 | 7.75 | 155.0 | 34–35 min per direction (calibration 52 min) |
| Screens and calibration | 39 | 2.09 | 41.7 | 2.5–8 min per job (95–150 s per process) |
| Quick-tier tests (orchestrator) | 8 | 1.09 | 4.4 | 11–15 min |

The per-job list is in `results/discovery-sacct.txt`. Key job IDs:

- calibration: 14217165, 14217167, 14217256;
- joint screens: 14218311–3, 14220936; Y1 screen: 14228110;
- campaigns:
  - A: 14218314/5; C: 14218317/8; D: 14218319/20;
  - ref: 14218336/7; B: 14220937/8; Y1: 14228169/70;
- quick-tier tests: A 14218417, C 14218415, D 14218416.

The implementers' own screening jobs are listed in their reports:

- A: 14217214 … 14217639;
- B: 14217217 … 14218281;
- C: 14217223 … 14217962, revision 14221135, 14223411, 14223938;
- D: 14217483 and earlier.

## Not done (by design or pending approval)

- No integration into #901, no merge, no ready-for-review, no new PR (the experiment branches are pushed only).
- No second campaign repetition for A, C and D: cancelled to free nodes once B and Y1 needed them.
- Julia 1.10 was not run, and nothing was run on the local Raptor Lake machine, so the #901 agent's local
  campaign was not disturbed.
- The targeted residual round for Y1 waits for approval.
