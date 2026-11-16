# Prior findings: what has already been measured and ruled

Compiled 2026-11-15 from GitHub threads (`gh`) and local worktrees, read-only. PB = ProximalBase.jl
(master `50db4b1e`, the #918 merge). PS = ProximalSolvers.jl (master `637f06a`, which pins PB `c123d56`).
"States" means the source says it; **[inferred]** marks my own reading. Every maintainer ruling is
collected, with a link, in the section "Maintainer rulings and constraints" near the end.

## 0. Read this first

- **The gate is "no reproducible slowdown above 5% on any existing ordinary hot path".** It is judged by
  paired in-process ratios against an A/A control and a byte-identical noise tree, in both role orders and
  in fresh processes, on Julia 1.10 **and** 1.13. A 13–18% slowdown that appears only on Julia 1.10 on one
  AMD EPYC model was **not** waived (#910).
- **#910 is open and blocked.** The C+A2 ridge-kernel prototype is 0.12–0.36× the base on ElasticNet array
  steps. But on Julia 1.10.12 on an EPYC 7513 it is +13–15% on the ElasticNet `prox_block` loop with a
  scalar step (8 of 8 fresh processes). The cause is that the per-call ridge weight no longer branches on `t ≤ 1`.
  The recommended next step, C1 (`b2099ff`), is unmeasured and awaits a maintainer decision.
- **Placement noise is large.** Identical code reads 1.69× (#910), 1.9× (#909), 1.8× with per-tree arrays
  (#865) and 7–40% in general. Single-cell differences of 10–20% on unchanged code are expected.
  Every past campaign needed a noise-floor tree N, shared arrays, reversed roles and fresh-process repeats.

## 1. PB #910: per-entry (vector) step evaluation in the shared `SqrNormL2` ridge kernel

**Summary.**
- Status: OPEN, at a mandatory STOP after the Phase 3 correction round, waiting for a maintainer decision.
- Designs tried: A (one-division weight), B (split loop with repair), C (prescan, then a specialized
  `prox_block`), C+A and C+A2.
- C+A2 fixed the centered Julia 1.10 regression. A second Julia 1.10 regression, on AMD EPYC 7513, blocks
  every branch-free weight form.
- Recommended next (unmeasured): C1, which is C's structure with the base weight code untouched.
- Nothing is merged and no PR exists. The experiment branches are local and unpushed (`pb910x/*`).

**Sources.**
- Thread: https://github.com/mlakolar/ProximalBase.jl/issues/910
- Run record: `/path/to/pb910x/harness/.codex-orchestrator/runs/run-20261115T0252-issue910-c5aa/`
  (`state.md`, `phase{1,2}-report.md`, `phase3-stop-report.md`, `evidence/`).
- Harness: branch `exp/910-harness` (`pb910x/harness`), files `benchmark/ridge910.jl` (76 paired cases),
  `check910.jl` and `check910_extra.jl`.

**Origin.** On PR #909, ElasticNet's array-step `prox!` measured 1.76–1.86× the archive `eff21b6d` on a
shared node, and 1.93–2.00× (+2.1 to 2.3 ns/entry) on an exclusive AMD EPYC 7542 node, Julia 1.13.3.
The maintainer waived these four cells on condition that #910 investigates
([D3 waiver](https://github.com/mlakolar/ProximalBase.jl/pull/909#issuecomment-6845205619)).
A variant that replaced only the in-band weight with one division `z/(1+t)` reached 1.29–1.36×.
So the cost is in the per-entry band decision and the weight formation, not in the division alone (states).

**Scope correction** ([Phase 1 start](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6888390276)).
`SqrNormL2` has no array-step `prox!`: it is an `AbstractFunction`, and the call is a `MethodError`.
Its only per-entry route is `prox_block` with a step per entry. The maintainer kept a `SqrNormL2`
array-step `prox!` out of scope (ruling 3, below).

**Baseline.** `6adfc790ab3323af588fb3d18fa3b06e911df32a`, the #909 head at the time. Its `sqrnorml2.jl` is
byte-identical to master. Since then #909 moved to `3fb64af`; per `state.md`, `elasticnet.jl` and
`sqrnorml2.jl` are unchanged, while `scaled.jl` and `scaled_array.jl` changed.

### Designs

| design | branch @ commit | change | ordinary bits |
|:--|:--|:--|:--|
| A | `exp/910-A-ordinary` @ `e3e0ac9` (final tip `66fa6ac`, after reverts) | `_ridge_weights_t`: `a = inv(1+t)`, `b = t/(1+t)`, no `t ≤ 1` branch; endpoint select before the division | change for t > 1 (5688 of 38348 probe entries); worst error unchanged at 1.841 ulp |
| B | `exp/910-B-split` @ `f325c7e` (tip `ae4d093`) | new `_ridge_prox_steps!`: blocks of 256, a branch-free `@simd` pass, non-ordinary entries repaired by sub-blocks of 16; dense `Array` only | identical |
| C (C2) | `exp/910-C-dispatch` @ `fea102f` (tip `2047783`) | a prescan of steps; if all are ordinary, `prox_block` specialized by a constant flag that drops the exceptional branches (vectorizes); otherwise the old loop. C2 = weights from selects of operands | identical |
| C1 | `exp/910-C-dispatch` @ `b2099ff` | C's prescan and specialization with the base `_ridge_weights_t` / `_ridge_ordinary` untouched | identical; **not measured on the cluster** |
| C+A | `exp/910-CA-prototype` @ `6882542` | C's `elasticnet.jl` + A's `sqrnorml2.jl` | as A |
| C+A2 | `exp/910-CA-prototype` @ `534dd1c` | base `sqrnorml2.jl` + `_ridge_ordinary_free` (A's arithmetic) at center-free call sites only; the centered `SqrNormL2` keeps the base code byte for byte | change for center-free t > 1 only (3689 on 1.13, 3207 on 1.10) |

**Rejected sub-variants** ([Phase 1 report](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6889017565)):
- A v2 (one product for the band test and the weights): LLVM already merges the products, and it broke
  `===` on NaN bit patterns.
- A v3 (`b = t·a`): it broke the scalar/array identity at ±floatmax with a center.
- B v3 (a prescan instead of repair): slower.
- C3 (scaled ridge path as one `@noinline` cold call): no gain.

### Measurements

All runs are paired in-process with roles base, candidates, N (a byte-identical copy of the base) and an
A/A recompile, alternating rounds, both role orders. **Rule used:** a regression is above 1.05 in both
orders and more than 0.03 above N in the same cell.

Phase 1 (Julia 1.13.3, shared Slurm node, 11 rounds; design/base, forward / reversed):

| case | A | B | C |
|:--|:--|:--|:--|
| ElasticNet `prox!` array, Float32 n=1024 / 65536 | 0.89/0.80, 0.89/0.80 | **0.09–0.11** | 0.15–0.17 |
| ElasticNet `prox!` array, Float64 n=1024 / 65536 | 0.75 | **0.25–0.28** | 0.36–0.40 |
| 1 step in 64 on the scaled path | 0.82–0.92 | 0.52–0.68 | 1.00–1.03 (falls back) |
| `SqrNormL2` `prox_block` loop, step per entry, no center | **0.56–0.59** | ≈1.00 | ≈1.00 |

- B had two unexplained cells on untouched `SqrNormL2` code: `Float64/ridge/value/n1024` 1.205/1.208 and
  `value_and_prox!/scalar/n1024` 1.077/1.065.
- In B's own runs that value cell read 1.000, while N and the A/A recompile read 1.21. **[inferred]** This
  is placement, but it was never cleared, and the maintainer refers to it as "B's unexplained regression cells".
- N itself read 1.693/1.668 on `Float64/ridge_c/prox_block_loop/steps/n1024`.

Phase 2, C+A ([report](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6889828453)):
- **Julia 1.13.3, exclusive AMD EPYC 7513, 21 rounds.** ElasticNet array 0.130 (Float32) and 0.235 (Float64);
  unweighted 0.15/0.26; scaled mix 0.77–0.86; `value_and_prox!` array 0.15/0.27. `SqrNormL2` per-entry
  loop: no center 0.50–0.62, centered 0.63–0.91. No cell met the rule.
- **Julia 1.10.12, exclusive EPYC 7513 (11 rounds) and EPYC 7542 (4 fresh processes, 21 rounds).** #910 cells
  0.06–0.21 (Float32) and 0.11–0.37 (Float64).
- **Regression:** `Float64/ridge_c/prox_block_loop/steps/n1024` (the centered `SqrNormL2` coordinate loop with
  a step per entry) read **1.175–1.190** in 7 of 8 processes. A alone shows it too. It is about +0.46 ns/entry
  (base 2.56 µs per 1024).
- **Diagnosis (states):** t ≤ 0.75 there, so A and the base do the same two divisions; the difference is
  control flow (branch vs `ifelse` with the select moved before the division). "Julia 1.10 code generation,
  not added arithmetic."
- **C alone was rejected on 1.10:** 1.3–2.0× slower in every `SqrNormL2` per-entry loop, because its
  branch-free select always computes `1/t`.
- **Ruling:** the regression was not waived; correct the centered path
  ([ruling](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6889937381)).

Phase 3, C+A2 ([report](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6890786463),
[addendum](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6890796211)):
- **Centered cell fixed.** On the EPYC 7513, Julia 1.10.12, 4 fresh processes: C+A 1.163–1.166 → C+A2
  **0.999–1.001** (N 0.991–0.993).
- **Xeon Silver 4116 shows nothing useful for the centered cell.** There C+A was faster, not slower, and the
  spikes were 0.475–1.674 with no consistent sign, the A/A recompile reaching 1.55.
- **ElasticNet gains are kept** (Julia 1.13.3, Xeon 4116, exclusive, 21 rounds): array 0.12–0.13 (Float32),
  0.24–0.36 (Float64); on Julia 1.10 the array cells are 0.12–0.35.

### The Julia 1.10 / EPYC finding (the blocker)

- **Case.** `{Float32,Float64}/enet/prox_block_loop/scalar/n1024` in `ridge910.jl`, a coordinate-descent pattern:
  ```julia
  s = 0; for i in eachindex(x); s += prox_block(ElasticNet(0.3, 0.5, w), x[i], i, 0.7); end
  ```
  with `x = 2randn`, `w ∈ [0.5, 1.5)`, n = 1024. So t = γλ2 = 0.35, always on the `t ≤ 1` side.
- **Magnitude.** On the EPYC 7513 (Zen 3, `znver3`), Julia 1.10.12, exclusive, 21 rounds:
  - A, C and C+A2 all read **1.126–1.153**, Float32 and Float64, both orders, in **8 of 8 fresh processes**
    for C+A2 (Slurm 14259166, 14259382). N reads 0.989–1.014.
  - Absolute: base ≈1.59 µs vs candidates ≈1.81 µs per 1024 calls, about **+0.22 ns per call**
    (`evidence/excl-attr/a10-fwd/raw.csv`).
- **Elsewhere it does not appear or does not reproduce.**
  - Xeon Silver 4116: 0.80.
  - EPYC 7542: 0.72–1.19, inconsistent.
  - Any CPU on Julia 1.13.3: 0.99–1.01.
- **Not the shared weight functions in general.** The center-free `SqrNormL2` `prox_block` scalar loop uses the
  same weight functions and reads 1.000–1.004.
- **Diagnosis (states).** What A, C and C+A2 share is that the per-call ridge weight no longer branches on
  `t ≤ 1`: A uses one division, C a select of operands, C+A2 `_ridge_ordinary_free`. In this loop the weight is
  formed on every call. "Under Julia 1.10 on the EPYC 7513, the base's predictable branch is faster than any of
  the three branch-free forms."
  - **[inferred]** No LLVM/native diff for this cell is posted. The maintainer required one only for the
    centered diagnosis, and the thread does not show it was produced for this cell either.
- **Why it cannot be patched narrowly (states).** Restoring the base formula for ElasticNet's scalar
  `prox_block` alone would break `prox!(γ) === prox!(fill(γ)) === prox_block` wherever the center-free weight
  differs (t > 1).

### Open alternatives (none authorized yet)
1. **C1** (`b2099ff`, recommended to measure): bit-identical, only `elasticnet.jl` changes. LLVM turns the
   branch into selects in the vectorized array loop; agent C saw C1 ≈ C2 in one process, with vectorization
   confirmed in the IR.
   - It gives up A's center-free `SqrNormL2` gains (0.50–0.62 on 1.13).
   - It gives up part of the Float64 array gain (C 0.30 vs C+A 0.23, preliminary).
2. **C+A2 with a waiver** of +13–15% on the EPYC 7513 under Julia 1.10.
3. **A branchy single-division weight:** keep the `t ≤ 1` branch, use `1/(1+t)` in the t > 1 arm. Speculative.

**Other #910 facts.**
- **Pre-existing 1-ulp difference (Julia 1.13).** Centered `SqrNormL2` `prox!` and `prox_block` differ by 1 ulp
  at 4 of 40 random entries, at the base too. Likely cause: `muladd` in `_ridge_blend` contracts differently in
  the broadcast. Ruled non-blocking.
- **Julia 1.10 benchmark environment.** It failed because the run helper sets `JULIA_PKG_OFFLINE=true` and
  BenchmarkTools was never installed in that depot. Fix: resolve `benchmark/Project.toml` online in a fresh
  per-version environment. The 1.13 Manifest cannot be used on 1.10 (StyledStrings).
- **Known 1.10 test failure.** `numerical_kernels.jl:735` (16 bytes) fails on 1.10 at the base too.

## 2. PB #908 / PR #909: ElasticNet restoration

**Summary.**
- **Status.** PR OPEN. The final maintainer approval to merge exact head `3fb64af1e6a3a9bfa4a7f201f6b2b05dd5942ff9`
  is posted ([ruling](https://github.com/mlakolar/ProximalBase.jl/pull/909#issuecomment-6904997589)); the merge
  was not yet done when read.
- **Kernel reuse.** ElasticNet reuses the shared kernels: the NormL1 threshold (`_l1_prox_threshold` →
  `_threshold_product`) and the SqrNormL2 ridge (`_ridge_needs_scaling`, `_ridge_ordinary`, `_ridge_scaled`,
  `_ridge_apply`). It has no ridge code of its own, by ruling.
- **Performance.** 56/60 cells pass against `eff21b6d`. The array-step `prox!` cells are waived at 1.93–2.00× (#910).

**Plan and kernels** ([plan](https://github.com/mlakolar/ProximalBase.jl/pull/909#issuecomment-6842821861)):
- **Scalar step.** A scalar step forms the ridge weight once per call (`_ElasticStep`), with the same bits as
  `prox_block`. Per-entry steps run the band decision and weight per entry.
- **Value.** The NormL1 value + the ridge value. On dense Float32/64 it is a **fused loop** with a proved
  (n+2)·eps acceptance test (kept only for n ≤ 1/(2eps) − 2). The composed value failed P at 1.5–15×, so the
  contingency fused loop was built.
- **Small values.** Below 2·floatmin, one compensated `_ExpSum` recomputation.
- **Steps (D1).** Negative and NaN **scalar** steps are rejected before writes. No array-step scan, citing
  #874: endpoint checks cost +5.9% (Float32) and +9.9% (Float64) on weighted array steps.
- **Unchanged files.** `norml1.jl`, `sqrnorml2.jl`, `separable.jl` and `numerical_kernels.jl` (criterion N).

**Criterion P (pre-registered, `benchmark/elasticnet.jl`, on the #909 branch, not on master).**
- **Setup.** Archive vs candidate vs an A/A copy, shared arrays, ≥21 alternating rounds, 60 cells.
- **Cells.** The `prox_block` coordinate loop n = 1024; dense scalar-step `prox!` weighted and unweighted,
  1024/65536, λ1 = 0 and λ2 = 0; array step; values in both dominance regimes; `value_and_prox!`; Sparse,
  Block and Symmetric iterates; `Scaled(ElasticNet)`.
- **Pass and regression.** Pass at median ≤ 1.05. Above that, rerun reversed in a fresh process; a cell is a
  regression only if it stays above 1.05 in both while A/A is within 1 ± 0.03.

**Results.**
- **Exclusive EPYC 7542, Julia 1.13.3, 21 rounds, both orders, at `3b4ad7c`.** 56/60 at ≤ 1.05. The 4
  array-step cells are at 1.93–2.00.
- **At merge head `4129db0`** ([addendum](https://github.com/mlakolar/ProximalBase.jl/pull/909#issuecomment-6889171170)):
  the other 56 cells are 0.235–0.990 vs archive. Old head vs new head: 0.960–1.020.
- **Later fused-value correction (`82b15da`/`3fb64af`).** `Scaled(ElasticNet)` `value_and_prox!` cells 0.992–1.014.
  - `Float64/block/value/n65536` is a **two-state placement artifact**: about 152 µs vs about 80 µs, landing on
    whichever role is in the slow state, the A/A control included (A/A 1.909 in one run).
- **No Julia 1.10 timings** in #909: its benchmark environment did not resolve. Allocations on 1.10 are
  covered by tests.

**Limitations kept.** #567 (Scaled fused-value gaps, overflow of αγ), #566 (array-step fold), #896 (Julia 1.10
`ldexp` subnormal).

## 3. PB #900 / PR #901: NormL2 and GroupNormL2

**Summary.**
- **Status.** MERGED 2026-11-14 as `f8531c9d` (head `df42e3ec`).
- **Final design: "Y1f".** Design C (optimized consolidation) + per-call level/step exits + two targeted fixes.
  It is one shared ℓ2 kernel layer in `src/numerical_kernels.jl`.
- **Baseline.** Measured against `76964085` (+ master `linear.jl` = `f36a6d6a`), **not** against `legacy/full`.
- **Rulings.** "Must not increase execution time"; the 5% rule is "a mandatory escalation threshold, not a
  5%-slower budget". 12 cold-path cells are waived for #901 only.
- **Do not touch** the shared two-term `_pow2_sum`.

**Sources.**
- PR: https://github.com/mlakolar/ProximalBase.jl/pull/901
- Reports: `git show exp/901-harness:experiment901/REPORT.md` (and `FINAL_REPORT.md`).
- Raw local archive (not opened): `/path/to/pb901-experiment-archive/`.

**Four-design experiment** (from #901 head `0eda76a9`; REPORT.md, #900 comment 6873214824).

| design | branch @ SHA | idea |
|:--|:--|:--|
| A | `exp/901-A-original-fastpaths` @ `2fde7b13` | restore the base's own fast paths; runs at base speed, so it served as the **noise-floor tree**; duplicates 3 recovery algorithms |
| B | `exp/901-B-shared-arith-separate-loops` @ `02dba9e3` | shared arithmetic, a separate loop per family (+456/−204) |
| C | `exp/901-C-optimized-consolidation` @ `582932fb` | shared kernels, screen difference written directly (+120/−41) |
| Y1 | `exp/C-y1` @ `64cab3f1` | C + level/step exits decided once per call, one `@noinline` loop per case |
| D | `exp/901-D-minimal-primitive` @ `2971e5fe` | minimal shared primitive, control flow per family |

**Setup.**
- Hardware: Discovery `oneweek`, `--constraint=xeon-2640v4` (Intel Xeon E5-2640 v4), exclusive, Julia 1.13.3,
  one pinned thread, `--check-bounds=no`.
- Design: base compiled twice (X and A/A), shared inputs, both orders, fresh processes.
- Calibration (jobs 14217165/7): 5/10/20 concurrent processes per node slow absolute speed by 7/15/28% but move
  paired ratios only about 1%. Single cases vary 5–10% between processes.
- Screen: 826 cases. Campaign: `l2_cold_paths.jl` (5046) + `l2_norms.jl` (368).
- Totals: about 94 jobs and 547 core-hours.

**Campaign results** (ratio to base, 2 runs per case):

| tree | cold geomean | cases >1.05 / >1.10 / >1.50 | worst |
|:--|:--|:--|:--|
| A (noise floor) | 0.945 | 28 / 2 / 0 | 1.13 |
| B | 0.573 | 61 / 35 / 14 | **2.02** |
| C | 0.723 | 61 / 6 / 0 | 1.16 |
| **Y1** | **0.720** | 45 / 3 / 0 | 1.12 |
| D | 0.881 | 19 / 13 / 0 | 1.34 |

- **Screen vs campaign gap (B).** B had the best screen: geomean 0.568, 0 cases >1.05. The full campaign found
  regimes absent from the screen at up to 2.0×:
  - mixed-weight `value` with groups of 5 and 32;
  - λ0 = 0 with infinite weights in the fused path, 1.5–1.8×;
  - γ = 0 fused, 1.46×.
- **Lesson (REPORT.md):** "Screen first, but never decide without the comprehensive campaign". Use a
  baseline-equal noise-floor tree.
- **Rejected.**
  - B: fragile duplicated loops, +342 lines.
  - D: Float64 groups of 5, subnormal `prox` at 1.10–1.34×.
  - A: a fallback only.
  - V3: inlining the exits into `_l2_shrink!` lost LLVM loop unswitching; threshold-underflow `prox` 1.3–2.8×.
  - A "known-norm term" fused variant: m = 2 was 4–16% slower.
- **Not fixable cheaply.** Zero-group `value` with m = 2–5 is bimodal (1.00 or 1.13 per process). "Every change
  to that zero test cost about 5% on ordinary Float64 values."

**Final kernels** (`f8531c9d`):
- **`_pow2_sum` / `_pow2_weighted`.** The optimized two-term method is kept for LinearFunction and LeastSquaresLoss.
  The group screen uses direct arithmetic, `_l2_screen_cold`: the difference is formed in the frame of the
  larger exponent (an approved exception).
- **Value path.**
  - `_group_value_checked`: the ordinary test is inline; `@noinline _group_value_unkept` holds the rest.
  - `_all_zero_early` scans in blocks of 512.
  - The kept-window `K*floatmin` test counts all groups.
- **Scaling.**
  - `_power_factors` builds 2^k from bits.
  - `_square_exponent` is one vectorized `_maximum_abs` pass with no NaN early exit (#917).
- **Prox.**
  - `_group_prox_all!` decides the exits once per call and runs `_group_prox_loop!` (`@noinline`) with either
    `_l2_shrink!` or `_l2_shrink_fixed!`.
  - `prox_block!` inlines the exits.

**Against `legacy/full` (`eff21b6d`).** There is **no per-group-size table at the merged head** in the threads.
Pre-consolidation heads were measured locally on one pinned i5-13600K (Raptor Lake) P-core, Julia 1.13.3, with
`l2_norms.jl` (sizes 2, 5, 10, 32, 128, 1024; contiguous and permuted):
- **`ebfe9e30` (11 rounds, 320 cases).**
  - GroupNormL2 `prox!` 0.83–1.02; `prox_block!` and sweeps ≤ 1.04; values ≤ 1.04.
  - NormL2 `prox!` 0.84–0.89, `value_and_prox!` 0.86–0.89, value 1.00–1.04.
  - Flags: Float32 m = 128 **contiguous** `value_and_prox!` 1.144/1.115, and Float64 m = 1024 dense 1.054. The
    permuted layouts read 0.99–1.01. The m = 128 flag did not reproduce at `9437d7a5`.
- **`c3f7a70e` (= `76964085` code; 9 rounds forward, 7 reversed).**
  - Worst forward median 1.063, only at Float32 m = 1024 (`prox` perm 1.052, `value_and_prox!` 1.058–1.063).
  - Accepted narrowly ([6841775511](https://github.com/mlakolar/ProximalBase.jl/pull/901#issuecomment-6841775511)),
    later superseded by the hold.
- **Earlier, `5d30ffde`.** `value_and_prox!` was 1.23–2.1× because accumulating squares in the write loop
  serialized it. Fixed by taking the stored group's norm as `t·‖x[G]‖`.
- **NormL2 #808 recovery on sparse/symmetric iterates:** 0.014–0.022× legacy, because legacy traversed the logical matrix.

**Merged code vs base `76964085`** (final campaign at `d1c75645`, Xeon E5-2640 v4, jobs 14239033/4, 4 runs per
case; [6884078940](https://github.com/mlakolar/ProximalBase.jl/pull/901#issuecomment-6884078940)):
- Cold paths: geomean 0.706, 12 cases >1.05 (worst about 1.10).
- `l2_norms`: 0.988, 0 cases.
- NormL1 harnesses: 0.999/1.001. LinearFunction: 0.999.
- Core `run.jl`: 0.997.
- **The 12 waived cells** ([6884588138](https://github.com/mlakolar/ProximalBase.jl/pull/901#issuecomment-6884588138)):
  - permuted all-zero groups of 32, Float64 (7 cases, 1.05–1.09);
  - Float32 zero-group `value` m5;
  - Float64 `prox` `ext_over` m5;
  - two threshold-restore cases at m32 perm (+28 ns on 320 ns, 1.08–1.10);
  - ordinary sparse-value rows at 1.05–1.06.
- **Cause named at `1d8bad5d`** for slow `kkt` with groups of 2/5: the shared two-term `_pow2_sum` vs the base's
  three-`ldexp` difference. That is why the screen got direct arithmetic.

**Open follow-ups.**
- **#917 (OPEN, optional).** A NaN early exit in the vectorized `_square_exponent`, and cheaper scaling in the
  small-group KKT screen (`ldexp` in `_frame_shift` and `ldexp(d,E)`).
  - Constraints: bit-identical outside D-X, 0 allocations, ≤ 5% vs base; the #901 exceptions do not carry over.
  - **[inferred]** This is the one open, explicitly sanctioned kernel-speed item in the L2 family.
- **#905 (OPEN, `candidate:0.7`).** Certified finite/Inf classification near overflow. Deferred: "No code work
  authorized without specific new practical evidence."
- **#904.** CLOSED as superseded by #905.

## 4. PB #915 / PR #918: least-squares oracles (design H implementation)

**Summary.**
- **Status.** MERGED 2026-11-16 as `50db4b1e` (final head `e6f875f7`), on base `f8531c9d`.
- **What it optimized:** the ordinary least-squares value.
  - (a) one `@simd` loop sum up to `_LS_ONE_LOOP_MAX = 2^17` entries, pairwise `sum(abs2)` / `mapreduce` above;
  - (b) an ordinary-path range test on the exponent bits;
  - (c) one `@noinline` cold call, `_ls_value_cold`.
- **Final ruling:** "Do not change the reduction threshold, trim precompilation, or initiate another
  optimization round." **Earlier:** "No more ordinary-path micro-optimization rounds for this issue."

**Sources.**
- PR: https://github.com/mlakolar/ProximalBase.jl/pull/918
- Issue: https://github.com/mlakolar/ProximalBase.jl/issues/915
- Code:
  - `src/functions/leastsquares.jl`: `_ls_value`, `_ls_ordinary_sum`, `_ls_product_fits`, `_ls_value_cold`
    (about lines 100–175);
  - `src/functions/leastsquares_predictor.jl`;
  - `src/numerical_kernels.jl:641-660`: `_LS_ONE_LOOP_MAX`, `_sumabs2_loop`, `_simd_sumabs2`.

**What was done (states).**
- **(a) One-loop sum.**
  - The generic pairwise `_sumabs2` was 1.2–2.1× slower than PS's loops at n = 4096–1e5.
  - `sum(abs2)` is 1.3–1.4× slower than the loop at n ≤ 1e5; the two are within 0–25% from 2^18 up.
  - The loop's error at 2^17 is ≤ 201 eps (Float32) and ≤ 462 eps (Float64). The reviewer found 3574 eps
    (Float32, n = 1e6) and 18931 eps (Float64, 1e7) without the switch, hence the 2^17 cut.
- **(b) Exponent-bit test.** The kept value is `α·s/2` when K ≤ s ≤ floatmax, K = 2^63·floatmin/eps.
  - It is one integer compare on the biased exponent and rejects 0, subnormal, Inf and NaN.
  - It must be stricter than the kernel's test (A′ ⇒ A).
  - Piece costs at n = 64: bit test about 0.4–0.7 ns; deferred scale check about 0.6–2.6 ns; avoiding a GC frame
    via the `_LSResponse` marker about 0.4–1.5 ns.
- **(c) Single cold call.** A separate long-vector call cost about 1 ns at n = 64 on Julia 1.13.
- **Also:** `@inline` on `_leastsquares_residual_gradient!` removed a 1.07–1.26 regression at (64, 6).

**Campaigns.**
- Final 1 at `6822772`: one exclusive AMD EPYC 7513.
- Final 2 at `e6f875f`: exclusive Intel Xeon E5-2640 v4, one node per version. The cluster is the "second cluster
  (Slurm)", **[inferred]** Discovery; no job ids are posted.
- Both on Julia 1.10.12 and 1.13.3. 12 fresh processes per configuration: 3 per role order, 21 rounds, with
  `--check-bounds=no` and with default bounds.
- A/A copy in-process, shared arrays. A/A spread about ±10% on shared nodes.
- **Flag rule:** median > 1.05 in both orders.

**Results (head / comparator; existing methods vs base, new oracles vs PS-style loops).**

| cells | Julia 1.10.12 | Julia 1.13.3 |
|:--|:--|:--|
| existing LS ordinary cells (`f(β)`, `gradient!`, `value_and_gradient!`) | 0.79–1.02 | 0.92–1.04 |
| combined oracle (64,6), (2000,500) | 0.76–1.00 | 0.89–1.00 |
| oracles n = 4096 / 1e5 / 1e7 | 0.97–1.09 / ≈1.00 / ≈1.01 | 0.99–1.06 / ≈1.00 / ≈1.01 |
| oracles n = 1e6 (**waived**) | 1.03–1.15 | 1.03–1.09 |
| n = 64 value-only (16 cells, **waived**; loops 6–12 ns) | 1.25–1.54 | 1.07–1.48 |

- **Accuracy.** At n = 1e6 and 1e7 the head errs by 0.02–0.64 eps; the PS loops by 745–758 eps.
- **Role-position artifact.** At n = 1e5, Float32 read 0.71 in one order and 1.33–1.43 in the other, with the
  A/A at 0.70–0.74.
- **Compile.** Precompile +1.3 s (about +9%), cache +0.64–0.87 MB. Accepted.

**Rejected or ruled out (states).**
- `llvmcall`, branch-weight hints, an unchecked public method, a second recovery kernel
  ([6889899617](https://github.com/mlakolar/ProximalBase.jl/pull/918#issuecomment-6889899617)).
- Pairwise recursion with `@simd` leaves: 1.2–1.4× slower around 1e5.
- Raising the pairwise threshold: it gives back hundreds of eps.
- Trimming the precompile workload.
- A style rewrite of the `_LSResponse` marker, the exponent-bit test or the deferred scale check.
- Implementer's floor analysis at n = 64: "To reach 1.05 against a 60-70-instruction loop, the oracle would need
  about 3 extra instructions or fewer. That is not possible while a length check, the acceptance test and a call
  to the recovery all stay." **[inferred]** The LS small-n value is at its accepted floor; reopening it needs a
  new maintainer ruling.

**Open after merge.** PS #307 must measure end-to-end whether the n = 64 overhead and the 14 µs at n = 1e6
matter; a repeatable regression "would reopen the implementation there". Related: #609 (`Scaled{LS}`) and #916.

**Background, PB #911** (research, CLOSED;
[memo](https://github.com/mlakolar/ProximalBase.jl/issues/911#issuecomment-6858247973),
[ruling](https://github.com/mlakolar/ProximalBase.jl/issues/911#issuecomment-6858843565)).

Design H was chosen over B1, B2, B3 and C, and over a generic `ScalarLoss` layer.

Setup: exclusive Intel Xeon E5-2640 v4, pinned, 60 paired rounds, Julia 1.13.3 and 1.10.12. The EPYC 7513
replicate was cancelled.

- **Oracle kernels vs K (unchanged PS).**
  - The value call costs **about 5–10 ns more per call** in every PB-owned design: the fast-path test and the
    call boundary.
  - `residual_value` via `sum(abs2, ·)` is 1.1–1.4× CD's own `@simd` loop.
  - Recovery inputs cost 1.1–2.1× K, where K is silently inaccurate.
- **End to end** (65 configurations, CD/PGD/APG): H is at 0.977–1.029 (1.13.3) and 0.994–1.011 (1.10.12).
  - Five Float32 PGD configurations exceed A/A by 1–3%; unresolved, below the 5% threshold.
- **PB's own LS methods after the refactor:** no reproducible change. The A/A spans 0.917–1.051.
- **Compile.** PB precompile is +0.3–0.7 s. The first raw-LS CD solve is about +0.5 s (raw LS is not in PS's
  precompile workload).

## 5. Benchmark methodology: PB #153, `benchmark/README.md`, CONTRIBUTING

**Summary.**
- **Formal stack (states).** BenchmarkTools + `PerfCase` + `compare.py`, with the paired-log-ratio,
  primary-case, one-sided 5% pass/investigate/inconclusive policy (#462/#499). `PB_CASES` selection already
  exists (#457/#738).
- **Stage 1 of #153 recommends keeping it.** PkgBenchmark, AirspeedVelocity and Chairmarks were rejected for
  formal use.
- **The kernel harnesses** (`l2_norms.jl` and similar) use X / Y / XC (A/A) in one process, alternating rounds,
  shared arrays, and a forward run plus a reversed run, each summarized on its own.

**Sources.**
- https://github.com/mlakolar/ProximalBase.jl/issues/153 (OPEN), especially the
  [scope ruling](https://github.com/mlakolar/ProximalBase.jl/issues/153#issuecomment-6710436685)
  and the [Stage 1 conclusion](https://github.com/mlakolar/ProximalBase.jl/issues/153#issuecomment-6717168958).

**Agreed practice, as written.**
- **The 5% rule.**
  - `benchmark/README.md`: "Judge reproducible slowdowns above five percent using the paired distribution,
    A/A controls, allocations, fresh-process repeats and reversed source roles."
  - `CONTRIBUTING.md`: compare "against the frozen post-#848 source [eff21b6d] on the same machine with the
    same harness, including an A/A control. Record any slowdown above five percent with its allocations,
    dispersion, controls and attribution."
- **Campaign recipe** (README "Timing campaign"): a fresh process per run, `--threads=1 --check-bounds=no`, idle
  and pinned (`taskset -c`); forward and reversed roles summarized separately; **also rerun without
  `--check-bounds=no`** (accesses not marked `@inbounds` cost only there); judge against `XC/X`; repeat
  flagged cells with `PB_FILTER` in a fresh process.
- **Robust statistic.** Use medians, not means: in #156 the means were inflated by a few slow samples.
- **The timer floor drifts within a process,** 24 → 40 ns (#331). Fix `evals` per case across both revisions.
- **The noise floor limits gates.** Ten fresh-process repetitions could not reach a "lower bound ≥ 0.97" bar in
  67/194 cases (#351).
- **Do not build** a two-worker scheduler, a drift monitor or new statistic defaults without approval
  ([ruling](https://github.com/mlakolar/ProximalBase.jl/issues/153#issuecomment-6710436685)).
- **Lessons from campaigns:** per-tree arrays caused 1.8× differences on identical code, so share the arrays
  (#865 `bench.jl`); placement moves tight recovery loops by ±6%; #910 added a byte-identical noise tree N and
  the rule "> 1.05 in both orders and > 0.03 above N"; #849 cleared 10–20% placement flags with
  same-namespace separate-process confirmations (31 rounds).
- **#849 audit facts** (`docs/audit/minimal-core-849/performance.md`): 102 cases, median B/A 1.000 vs `eff21b6d`.
  The inactive block writes at +5–6 ns (1.13–1.16×) were treated as flags despite their tiny absolute size:
  "absolute sizes ... do not exempt them from the requested relative 5% gate".

## 6. PS #307 and how PS consumes PB kernels

**Summary.**
- **Status.** PS #307 (migrate CD, PGD and APG to the design-H oracles) is OPEN. Draft PR #319 has no
  production change yet; arm A (PS `637f06a` + PB `c123d56`) was being measured.
- **What PS calls today.** PS uses only `LeastSquaresLoss` (or one `Scaled`) + unweighted `NormL1`.
  ElasticNet, SqrNormL2, NormL2 and GroupNormL2 are not used in PS `src/`.
- **Per-entry PB calls in PS hot loops:** only `NormL1` `prox_block` with a scalar step in CD.

**Sources.** https://github.com/mlakolar/ProximalSolvers.jl/issues/307;
https://github.com/mlakolar/ProximalSolvers.jl/pull/319 (plan rev 1:
[comment](https://github.com/mlakolar/ProximalSolvers.jl/pull/319#issuecomment-6904783434)).

**PB call sites (PS `637f06a`).**
- **CD** (`src/cd/leastsquares.jl`): per coordinate, `prox_block(g, c, k, γ)` with `γ = 1/(m.s*a)` (`:108`,
  `_scaled_prox`; PS's own `_exceptional_step` below floatmin). The inner products are PS `@simd` loops
  (`_column_dot` `:36-50`, `_residual_axpy!` `:52-58`). `SparseIterate` get/set per coordinate;
  `activecoordinates`/`dropzeros!` per pass (`src/cd/solver.jl:170-198`); `loss_value + g(work)` once per solve.
- **PGD/APG:** `value_and_prox!(x, g, xhat, 1/L)` per backtracking trial (PGD `proximal_gradient_descent.jl:152`,
  APG `accelerated_proximal_gradient_descent.jl:227`). The loss goes through PS's `_image_value*`
  (`src/solvers/image_models.jl:15-31`): BLAS `mul!` + `sum(abs2, d)`.
- **After #319 (planned):** `value_from_predictor` and `value_and_gradient_from_predictor!` (PGD/APG),
  `value_from_residual` (CD), data via `leastsquares_data(f)`.
- **[inferred]** The PS-relevant PB hot paths are (i) `NormL1` `prox_block` with a scalar step, per coordinate;
  (ii) `NormL1` `value_and_prox!` over p entries, per trial; (iii) after migration, the O(n) squared-residual sum
  plus α. PGD/APG time is dominated by `X*x` and `X'*d`.

**Gates for #307/#319 (states).**
- A reproducible ordinary hot-path slowdown above 5% goes to the maintainer.
- First raw-LS solve latency should stay at baseline (C ≤ A × 1.05, 10 fresh interleaved processes per arm).
- A load or precompile increase above 10% is escalated.
- Products per evaluation must be identical across arms; 0 warm bytes.
- Arms: A (old PS + old PB), B (old PS + PB `50db4b1`), C (migrated PS + PB `50db4b1`).
- "Do not add raw-LS precompilation unless measurements justify it"
  ([D1 answer](https://github.com/mlakolar/ProximalSolvers.jl/pull/319#issuecomment-6904765782)).
- **The PB bump `c123d56` → `50db4b1` is 166 commits.** It includes `NormL1` value/prox changes, so arm B
  isolates the PB-side effect on PS's `NormL1` calls.

**PS benchmark suite (`benchmark/`).**
- **Cases.** 40 current-core cases (groups `cd`, `gradient`): lasso 100×32, lasso-B 240×160 (ρ = 0.8) in the
  quick suite; lasso-C 800×120, lasso-D 180×600 in the extended suite.
- **Variants.** CD dense or sparse, PGD/APG dense, Float32/Float64, cold or warm start; ten Float64 `acc-`
  PGD/APG cases in the extended suite.
- **Certificate.** An independent relative duality gap (1e-8 for Float64, 1e-5 for Float32); a case that misses
  it is excluded from timing.
- **Invocation:** `julia benchmark/setup.jl "$PWD" benchmark/results/env pinned`, then
  `JULIA_NUM_THREADS=1 PS_EXPECTED_PROXIMALBASE_REV=<sha> julia --project=benchmark/results/env benchmark/run.jl`.
  - Env vars: `PS_SUITE`, `PS_GROUPS`, `PS_CASES`, `PS_REPETITION`, `PS_SAMPLES`, `PS_SECONDS`, `PS_OUTPUT`, ….
  - Revisions are compared with `compare.py`.
- **No timing results are tracked in the repo.** The evidence lives in GitHub comments (#318, PB #911).

## 7. PB #852: historical iterate audit

Measurement only, against the pre-agent original `bd0740d0` (2020-08-13), Julia 1.13.3; report in draft PR #855
(not merged). It covers iterates, not the NormL1, L2 or LS kernels, so its relevance here is low.
[Findings](https://github.com/mlakolar/ProximalBase.jl/issues/852#issuecomment-6808169332):
- scalar CD, four residual-maintaining sweeps: 128.1 → 129.5 µs (R = 1.011);
- block CD 7.8–15.8% slower; b64 diagonal sweeps 5.2× (Float64) and 8.8× (Float32) slower, with "reduced
  vectorization in a representative kernel"; block copying up to 51× slower in one case;
- sparse clearing 254 → 0.6 µs. Follow-ups are ranked but not authorized.

Harness: `benchmark/audit852/` (a fresh worker per role per round, bootstrap over rounds).

## 8. Maintainer rulings and constraints

Quotes are verbatim where marked with quotation marks.

| # | Ruling | Source |
|:--|:--|:--|
| R1 | "Enforce the 5% regression threshold for every benchmark cell. Escalate reproducible violations." (#909 plan) | [#909](https://github.com/mlakolar/ProximalBase.jl/pull/909#issuecomment-6842986840) |
| R2 | Retain the shared SqrNormL2 ridge kernels. "Do not introduce an ElasticNet-specific ridge implementation." The waiver covers only the 4 array-step `prox!` cells | [#909 D3](https://github.com/mlakolar/ProximalBase.jl/pull/909#issuecomment-6845205619) |
| R3 | Treat "*reproducible* >5% slowdowns on **any existing ordinary hot path** as disqualifying without another maintainer ruling. Prefer no measurable regression whatsoever. Escalate anomalous unchanged-code cells with fresh-process placement controls rather than ignoring them." | [#910 Phase 2 auth](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6889148758) |
| R4 | The +18% on Julia 1.10.12 "violates our no-regression policy and is **not approved as an exception**, even though the absolute cost is small." | [#910 ruling](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6889937381) |
| R5 | Distinguish centered and center-free cases "by existing Julia dispatch/specialization where possible, **not** by a Julia-version check"; "no parallel family-local ridge implementations, no new recovery framework"; "Do not presume the exact implementation before inspecting generated code"; "do not start a broad optimization search without asking." | same |
| R6 | "Favor **the simplest passing implementation** … the major win is the ElasticNet SIMD path, not making every ridge operation use A's formula. Do not add B's per-block repair machinery at this stage." If C+A brings little over C alone, prefer simplicity | same; [Phase 2 auth](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6889148758) |
| R7 | A's bit changes for t > 1 are accepted "in principle" (worst error unchanged); this "does **not** relax accuracy, scalar-vs-constant-array agreement, exceptional-input semantics, aliases, or stored-output guarantees." | [Phase 2 auth](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6889148758) |
| R8 | "**Keep `SqrNormL2` array-step `prox!` out of scope** … do not introduce a new public operation for benchmarks." | same |
| R9 | The final #910 suite must cover Julia 1.10 **and** 1.13, both orders, A/A, exclusive-node evidence, "No extra allocations or meaningful first-call compilation regression." 1.10: "do not reuse a 1.13-only manifest or silently omit 1.10 timings." | [#910 ruling](https://github.com/mlakolar/ProximalBase.jl/issues/910#issuecomment-6889937381) |
| R10 | The 1-ulp `muladd` discrepancy in centered SqrNormL2 is non-blocking; "Do not open another broad numerical-recovery project for a harmless last-bit difference." | same |
| R11 | "No additional checks or scans are authorized in the hot loop for #853 or #866" (endpoint checks cost +5.9% / +9.9% on array steps in #874; +19–31% on weighted scalar steps on a newer CPU) | `docs/src/numerics.md:255-263` |
| R12 | Negative/NaN steps: check scalar steps only (O(1)); "Do not add array-step scans." | [#909](https://github.com/mlakolar/ProximalBase.jl/pull/909#issuecomment-6842986840) |
| R13 | "No new numerical recovery work is authorized" (#909 integration); #567 stays deferred | [#909](https://github.com/mlakolar/ProximalBase.jl/pull/909#issuecomment-6887094887) |
| R14 | #911: no generic vector `ScalarLoss` layer; α kept as supplied; gradient α·fl(Xᵀd), "α LAST … Do not add an adaptive alternative or new recovery machinery"; "Bitwise equality … is not required … Do not credit a faster but less accurate solve." The 1–3% PGD effects are "below the 5% review threshold"; a second CPU is needed only if a repeatable >5% or an unexplained pattern appears | [#911 ruling](https://github.com/mlakolar/ProximalBase.jl/issues/911#issuecomment-6858843565) |
| R15 | #153: keep BenchmarkTools + `PerfCase` + `compare.py`; do not build the two-worker driver, drift monitor or new statistic defaults; "the existing statistical acceptance policy changes only with explicit maintainer approval." | [#153](https://github.com/mlakolar/ProximalBase.jl/issues/153#issuecomment-6710436685) |
| R16 | #852: "Investigate repeatable >5% slowdowns on representative hot paths; do **not** automatically optimize"; open optimization issues only after maintainer review | [#852 body](https://github.com/mlakolar/ProximalBase.jl/issues/852) |
| R17 | PS #307: ">5% ordinary-hot-path slowdown goes back to the maintainer"; first raw-LS latency at baseline; "Do not add raw-LS precompilation unless measurements justify it" | [PS #307](https://github.com/mlakolar/ProximalSolvers.jl/issues/307), [PS #319](https://github.com/mlakolar/ProximalSolvers.jl/pull/319#issuecomment-6904765782) |
| R18 | Leave the PS pins and the 0.6 release unchanged; do not delete `pb910x/*` worktrees or data | [#909 final](https://github.com/mlakolar/ProximalBase.jl/pull/909#issuecomment-6904997589) |

| R19 | #901: must "not increase execution time" vs base. "Treat any reproducible slowdown that exceeds measured A/A variation as a regression ... even when below 5%. The existing >5% rule is a mandatory escalation threshold, **not** a 5%-slower budget." A duplicate algorithm is allowed only with an A/A-controlled regression, two non-duplicating attempts, and approval | [#901](https://github.com/mlakolar/ProximalBase.jl/pull/901#issuecomment-6842282369) |
| R20 | "Do not restore the old screen arithmetic or modify the shared `_pow2_sum`." Later: "Keep the optimized generic `_pow2_sum` for LinearFunction and LeastSquaresLoss" | [#901 D-MPFR](https://github.com/mlakolar/ProximalBase.jl/pull/901#issuecomment-6858163417), [Y1f adoption](https://github.com/mlakolar/ProximalBase.jl/pull/901#issuecomment-6879207550) |
| R21 | "Do not chase sub-percent noise by adding more code"; keep the two-term `_pow2_weighted` fast path (constrain, do not generalize) | [#901](https://github.com/mlakolar/ProximalBase.jl/pull/901#issuecomment-6847254548) |
| R22 | "Do not optimize the 346 cases where the old head beats Y1f"; "Do not enter another open-ended performance optimization cycle"; do not restart architecture experiments | [#901](https://github.com/mlakolar/ProximalBase.jl/pull/901#issuecomment-6879207550) |
| R23 | The #901 exceptions apply "to this PR only. The policy stays as it is: anything more than 5% slower than base needs explicit approval." | [#901](https://github.com/mlakolar/ProximalBase.jl/pull/901#issuecomment-6884588138) |
| R24 | "Do not introduce additional complexity or specialized kernels solely to support extreme BigFloat inputs" | [#901](https://github.com/mlakolar/ProximalBase.jl/pull/901#issuecomment-6842884992) |
| R25 | #918: "Reuse ProximalSolvers' faster ordinary-path implementations wherever they demonstrably improve performance ... Do not accept regressions merely for the sake of code reuse." | [#918](https://github.com/mlakolar/ProximalBase.jl/pull/918#issuecomment-6889136754) |
| R26 | "No more ordinary-path micro-optimization rounds for this issue." Accept the small-n overhead as a narrow exception; "No `llvmcall`, branch-weight hints, unchecked public method, altered numerical contract, or second recovery kernel." | [#918](https://github.com/mlakolar/ProximalBase.jl/pull/918#issuecomment-6889899617) |
| R27 | "The n = 10⁶ slowdown is an acceptable cost of restoring pairwise-quality summation. Preserve that numerical accuracy." "Do not change the reduction threshold, trim precompilation, or initiate another optimization round." | [#918 final](https://github.com/mlakolar/ProximalBase.jl/pull/918#issuecomment-6904387419) |
| R28 | #915/#918: "Bit-for-bit outputs are diagnostics, not acceptance gates"; `SqrNormL2` unchanged; no extra BLAS products; zero warm allocation | [#918](https://github.com/mlakolar/ProximalBase.jl/pull/918#issuecomment-6887320573) |

**[inferred] What these rulings mean for a new kernel study:**
- Any change to a shared kernel (ridge weights, `_pow2_sum`, `_l1_prox_threshold`, the LS sum) must be measured
  on Julia 1.10 **and** 1.13, on more than one CPU kind (EPYC 7513 was decisive in #910), with N and A/A controls.
- A gain elsewhere does not buy a >5% reproducible loss on any existing ordinary path.
- Do not reopen the LS small-n value, the 2^17 threshold, or the #901 screen design without a new ruling.

## 9. Existing PB benchmark scripts (`benchmark/` on master)

All the kernel harnesses load source trees as modules: role X twice (X and the A/A control XC) and role Y.
They time alternating rounds and write `raw.csv` + `provenance.txt` for
`summarize_norml1_infinite_weight.py`.
- Run: `julia --threads=1 --check-bounds=no --project=benchmark benchmark/<script>`.
- Common env vars: `PB_FIRST` (X root), `PB_SECOND` (Y root), `PB_OUTPUT`, `PB_ROUNDS`, `PB_FILTER` (regex on
  case ids), `PB_CHECK_ONLY=1` (agreement and allocation check only, no timing).

| script | covers | invocation notes |
|:--|:--|:--|
| `run.jl` (+ `probes.jl`) | the 102-case #849 minimal-core suite vs the frozen `eff21b6d` (iterates, L1 value/prox, LS value/gradient); the CI frozen differential | `PB_BASELINE`, `PB_CANDIDATE`, `PB_OUTPUT`, `PB_ROUNDS` (default 11), `PB_FILTER`, `PB_CHECK_ONLY`; summarize with `summarize.py` |
| `norml1_infinite_weight.jl` | NormL1 prox loops: weighted/unweighted, scalar and per-entry steps, dense, sparse and block, n = 1024 and 65536, Float32/64 (#853, #866) | `PB_FIRST/SECOND`, `PB_ROUNDS` (15), `PB_OUTPUT`, `PB_FILTER`, `PB_CHECK_ONLY` |
| `norml1_value_underflow.jl` | NormL1 value: the dense weighted SIMD loop, zero weights, small values, views, sparse, block and symmetric, `value_and_prox!`, the sub-floatmin recomputation; n = 100, 1024, 65536, 100000 (#373) | plus `PB_SIZES` |
| `l2_norms.jl` | GroupNormL2 `prox_block!` per group (sizes 2, 5, 10, 32, 128, 1024), sweeps, array prox/vprox/value, contiguous vs permuted, about 4096 entries, BlockIterate block-CD sweep; NormL2 dense at 1024/65536, sparse and symmetric values (#900) | `PB_ROUNDS` (11), `PB_EXACT=1` (bitwise check with `PB_CHECK_ONLY`) |
| `l2_cold_paths.jl` | exceptional paths of NormL2 and GroupNormL2 and the shared kernels (overflow, underflow, extreme thresholds, NaN/Inf positions, zero/Inf levels); also SqrNormL2, `norm_diff`, LS and NormL1 values; BigFloat references; exact differential | `PB_EXPECT_DIFF` (auto/same/forward/reversed), `PB_STRIDE`, `PB_INVENTORY`, `PB_ROUNDS` (11) |
| `leastsquares_predictor.jl` | LS `value` (p = 6, n = 64/4096/100000, raw and scaled), `value_and_gradient`/`gradient` at (64, 6) and (2000, 500); SqrNormL2 value with a center; the three #915 oracles (raw, scaled, `scaled_alpha64`); exceptional `over/under_scaled`; long vectors n = 10⁶, 10⁷ | `PB_ROUNDS` (11), `PB_PRINT=1`; BLAS on one thread; oracle cells compare the oracle against script loops (forward: X = loops) |
| `linear_sparse_value.jl`, `linear_shape.jl` | LinearFunction value at matrix SparseIterates (#810) | same pattern |
| `differential.jl` | independent dense/BigFloat references (no timing) | — |
| `copy861/`, `copy872/`, `dropzeros834/`, `audit852/` | iterate copy, dropzeros and historical audits | own READMEs |

On branches, not on master:
- `benchmark/elasticnet.jl` (#909, 60–68 cells, archive vs candidate vs A/A);
- `benchmark/ridge910.jl`, `check910.jl`, `check910_extra.jl` (`exp/910-harness`; env `PB_BASE`,
  `PB_CANDIDATES=name=path,...`, `PB_REVERSE`, `PB_ROUNDS` (21), `PB_FILTER`, `PB_CHECK_ONLY`, `PB_OUTPUT`);
- `docs/audit/numerical-kernels-865/bench.jl` (on master: SqrNormL2, LS, L1 and Linear kernels, shared arrays).
