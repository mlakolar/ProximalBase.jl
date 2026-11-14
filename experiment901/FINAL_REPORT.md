# #901 architecture experiment: final decision report (Y1 targeted round)

The earlier comparative report, [REPORT.md](REPORT.md), selected design C in its Y1 form (`64cab3f1`). The maintainer then:

- approved Y1 as the preferred design;
- granted the narrow exception for direct group-screen arithmetic in the shared kernel layer;
- allowed at most three targeted screens on Y1's two residual families;
- ruled that MPFR exponent-boundary results need no special BigFloat code.

This report covers that round and the final comparison against the #901 head and the baseline.

**Nothing was integrated into #901.** Its branch, its agent and the PR are unchanged, no PR was opened, and nothing was merged.

## Decision

**Winner: Y1f = `0bd9974c`** on branch `exp/901-Y1-final`, which is Y1 `64cab3f1` plus two small commits.

**Integration form:** **`7452d25f`** on branch `exp/901-Y1f-on-head`. This is Y1f merged onto the #901 head `1d8bad5d`, with the two conflicts resolved and one head test adapted. It is an integration aid only; see [the changes to integrate](#changes-for-the-901-agent-to-integrate).

| SHA | What |
|---|---|
| `f36a6d6a` | base: `76964085` + master's `src/functions/linear.jl` (`exp/901-baseline`), the performance baseline |
| `0eda76a9` | frozen #901 reference, from which every design started |
| `1d8bad5d` | #901 head. Code = `0eda76a9` + `04821b7e` (small-group cold paths); the later commits change only comments and tests |
| `64cab3f1` | Y1 (`exp/C-y1`) |
| **`0bd9974c`** | **Y1f (`exp/901-Y1-final`)**: Y1 + `2ea57152` (W2) + `0bd9974c` (Z5) |
| `7452d25f` | Y1f merged onto `1d8bad5d` (`exp/901-Y1f-on-head`) |

**Why Y1f:**
- **Against base:** fewest repeatable slowdowns of any candidate, with 9 of 5,046 cold-path cases above 1.05 in all 4 runs (worst 1.07–1.09). Head has 35.
- **Speed:** cold-path geomean 0.705 of base. Head is at 0.887.
- **Complexity:**
  - no duplicated recovery algorithm and no BigFloat-specific code;
  - one shared ℓ2 kernel layer;
  - `src` diff of +151/−46 against `0eda76a9`;
  - the targeted round added +23/−9 to Y1.
- **Correctness:**
  - Float32/Float64 accuracy against independent BigFloat-512 references is identical to base;
  - all NaN/Inf/zero/infinite-weight class checks pass;
  - 0 allocation regressions;
  - Julia 1.13 quick tier 30/30;
  - Julia 1.10: only the pre-existing `norm(S)` allocation test fails, as it does on base and head.

**Where head is faster than Y1f:** in 346 cold-path cases, all of them cases where both are faster than base and head more so.
- **NaN inputs:** head's entry scan stops at the first NaN. Examples: NormL2 value with NaN, head 0.06 of base vs Y1f 0.10–0.21.
- **Groups of 2–5 with overflowing entries** (`kkt` `over_norm_val`/`over_entries`): head 0.43–0.56 of base vs Y1f 0.95–1.0.

No case in these families is a regression against base. They are listed under [follow-ups](#possible-follow-ups-not-done-outside-the-three-screen-budget).

## The targeted round (≤ 3 screens, Opus implementer, Discovery)

| Variant | SHA | Change | Verdict |
|---|---|---|---|
| V1 | `c0804a01` | zero test as a float sum of magnitudes over `G` | did not help zero `value`; ordinary value +2% |
| V1b | `334f1583` | zero test as a bit OR over `G` | zero `value` 1.04→1.02; ordinary Float64 value m1/m2 1.05× |
| V2 | `db9ee6db` | V1 + one cold call holding exits and values inline (base structure) | prox `wInf` 1.2–1.4×, `prox_block` exits 1.11 |
| V3 | `77ce98da` | V1 + call-site `@inline` of the exits in `_l2_shrink!` | fixes `prox_block` but loses LLVM unswitching: thr_under prox 1.3–2.8× |
| W2 | `2ea57152` | exits inlined in `prox_block!` and in the per-call exit loop | clean |
| X1, X2 | `7a7d3eb5`, `75c70524` | W2/W1 + V1b | ordinary Float64 value m1/m2 1.05× |
| **Z5** | **`0bd9974c`** | W2 + bit-OR zero test in the shrinkage kernel only | **selected** |

Jobs: screens 14235538 / 14236214 / 14236629; focus 14235539, 14235543, 14236218, 14236219, 14236630, 14236631; quick tier 14236634 (W2), 14236747 (Z5).

**Results on the two residual families** (focused run, 3 processes per order):

| Family | Y1 → Y1f | Status |
|---|---|---|
| Float32 `prox_block!` with parameter exits, m5 | 1.016 → 0.891 | fixed |
| Float64 `prox_block!` with parameter exits, m5 | 1.041 → 0.914 | fixed |
| Zero-group fused, range layouts | 1.009 → 0.962 | fixed |
| Zero-group prox, range layouts | 0.975 → 0.916 | fixed |
| Zero-group `value` with groups of 2–5, range layouts | ≈1.04–1.05 | **not fixed** |
| Float64 prox/fused with infinite weights, m5 | 1.03–1.09 | unchanged |

- **Zero-group `value`:** the per-process ratio is bimodal (1.00 or 1.13) in Y1 and in Y1f alike, which points to code placement. Every change to that loop's zero test cost about 5% on ordinary Float64 values, so it was left alone.
- **Float64 infinite-weight prox/fused:** inlining the exits in the ordinary loop (V3) triggers the measured unswitching trap.
- **W2 (`2ea57152`) is the smaller alternative** (+10/−7 to Y1): same screen and the same `prox_block!` fix, but zero fused stays at about 1.02. Z5 adds a second form of the trivial zero predicate, `_l2_zero_group`, not a recovery algorithm.

## Final comparison (Discovery, identical inputs)

**Method:**
- **Hardware:** `oneweek`, Xeon E5-2640 v4, exclusive nodes, Julia 1.13.3, one pinned thread, `--check-bounds=no`, fresh processes.
- **Pairing:** every process compiles the first tree twice, as its own independent A/A control.
- **Replication:** each pair ran two replicate jobs in both role orders, so 4 runs per case.
- **Harnesses:**
  - all 5,046 `l2_cold_paths` cases and all 368 `l2_norms` cases;
  - the shared-kernel consumers: `norml1_infinite_weight` (56), `norml1_value_underflow` (144), `linear_sparse_value` (114) and `run.jl` core probes (102 cases, plus 102 differential probes);
  - plus a joint 826-case screen with base, `0eda76a9`, Y1f, head and Y1 in the same 6 processes.

### Against base (ratio to base; "slower" = above the threshold in all 4 runs)

| Harness | head `1d8bad5d` | Y1f `0bd9974c` | merge `7452d25f` (2 runs) |
|---|---|---|---|
| cold paths (5,046) | geomean **0.887**; >1.02: 138, >1.05: **35**; max median 1.108 | geomean **0.705**; >1.02: 48, >1.05: **9**; max median 1.104 | geomean **0.705**; >1.05 in both runs: 18 (Y1f, same 2-run criterion: 18 and 25; head: 62 and 59); max median 1.181 |
| `l2_norms` (368) | 0.991; >1.05: 0 | 0.988; >1.05: 1 (`Float64/m10/range/sparse/value` 1.05–1.27) | 0.990; >1.05: 3 (Float32 m2/m5 range sparse value ≈1.08, as Y1f; one NormL2 sparse-matrix case with A/A 1.11) |
| `norml1_infinite_weight` (56) | 1.003; 0 | 1.002; 0 | 1.002; 0 |
| `norml1_value_underflow` (144) | 1.004; 0 | 0.999; 0 | 0.999; 0 |
| `linear_sparse_value` (114) | 0.997; 0 | 1.000; 0 | 1.002; 1 (`Float64/n1048576/diagonal/d100pct` 1.31, A/A 0.77: noise) |
| `run.jl` core (102) | 0.998; 0 | 0.997; 0 | 1.002; 2 (symmetric `copyto_iterate` / `pair_read`, code neither tree changes: placement) |
| allocation regressions | 0 | 0 | 0 |

**Cold-path geomean by family:**

| Family | head | Y1f |
|---|---|---|
| GroupNormL2 | 0.917 | 0.755 |
| NormL2 | 0.684 | 0.396 |
| SqrNormL2 | 0.834 | 0.627 |
| NormL1, LeastSquaresLoss, `norm_diff` | 0.997–1.002 | 0.995–1.003 |

The A/A medians are 0.4–0.6% per run (p90 2.6–4.0%).

### Repeatable slowdowns of Y1f against base (cold paths, >1.05 in all 4 runs)

| Case | Runs | A/A |
|---|---|---|
| `GroupNormL2/value/Float64/negzero/m32/perm` | 1.073/1.078/1.075/1.073 | ≤1.004 |
| `GroupNormL2/value_matrix/Float64/zero/m32/perm` | 1.076/1.067/1.069/1.068 | one run 1.081 |
| `GroupNormL2/value/Float64/zero/m32/perm` | 1.071/1.068/1.071/1.063 | one run 1.050 |
| `GroupNormL2/prox/Float64/reg_wInf_nan/m5/range` | 1.057/1.083/1.128/1.056 | ≤1.006 |
| `GroupNormL2/value/Float32/negzero/m5/range` | 1.088/1.061/1.056/1.090 | 0.94–1.06 |
| `GroupNormL2/prox_matrix/Float64/zero/m32/perm` | 1.063/1.072/1.088/1.055 | ≤1.009 |
| `GroupNormL2/value_matrix/Float32/zero/m2/range` | 1.061/1.088/1.054/1.054 | 0.95–1.05 |
| `GroupNormL2/prox/Float64/negzero/m32/perm` | 1.052/1.062/1.061/1.063 | ≤1.003 |
| `GroupNormL2/prox/Float64/zero/m32/perm` | 1.061/1.068/1.065/1.052 | ≤1.019 |

- **Classes:**
  - all-zero groups of 32 on permuted layouts (value/prox), which head shares (its `value/Float64/zero/m32/perm` is 1.074–1.083);
  - the zero-group `value` residual of the targeted round;
  - Float64 infinite-weight prox at m5.
- **For comparison:** head's 35 cases are mostly GroupNormL2 `kkt` with groups of 2 at overflow/underflow/threshold regimes (1.07–1.12), plus m32 zero-group values. In the design reports' noise-floor run, design A, whose ℓ2 code equals base, had 28 such cases in a 2-run campaign. The 4-run criterion here is stricter.

### Y1f against head (ratio to head)

| Harness | Geomean | Y1f faster (<0.95 in all runs) | Y1f slower (>1.05 in all runs) | Worst |
|---|---|---|---|---|
| cold paths | 0.795 | 3,028 | 346 | 3.78 |
| `l2_norms` | 0.996 | — | 2 | 1.11 |
| consumers (`l1inf`, `l1under`, `linsparse`, core) | 1.000–1.002 | — | 0 | 1.08–1.11 |

**The 346 cold-path cases where Y1f is slower than head** (Y1f/head; head/base; Y1f/base):

| Family | Y1f/head | head/base | Y1f/base |
|---|---|---|---|
| NaN entries, GroupNormL2 `value`/`kkt`/prox/fused, m5 and m32 (≈180 cases) | 1.15–2.0 | 0.2–0.68 | 0.40–0.80 |
| NaN entries, NormL2/SqrNormL2 `value`/prox/fused, n1024 and n65536 (≈70 cases) | 1.4–2.3 | 0.05–0.16 | 0.10–0.24 |
| overflowing entries, GroupNormL2 `kkt` m2 and m5 (`over_norm_val`, `over_entries`; ≈20 cases) | 1.8–2.3 | 0.43–0.56 | 0.95–1.0 |
| overflowing entries, GroupNormL2 `kkt` m32 and `value` m5 | 1.08–1.10 | 0.73–0.88 | 0.81–0.96 |

- **Against base:** Y1f is faster than base in almost all of these, and none is in Y1f's list of repeatable slowdowns. 52 have a median Y1f/base just above 1.0, all within noise.
- **Joint screen** (826 cases, 6 processes, job 14237277): Y1f/head geomean 0.827, 28 cases >1.05 in all runs, all NaN NormL2/GroupNormL2 paths.

### Joint screen (ratio to base, 6 processes, same processes for every tree)

| Tree | Geomean | >1.02 / >1.05 in all 6 | Max median |
|---|---|---|---|
| Y1f | 0.757 | 2 / 0 | 1.119 |
| Y1 | 0.769 | 1 / 0 | 1.065 |
| head | 0.916 | 36 / 24 | 1.260 |

Y1f/Y1 geomean is 0.986, with 0 cases >1.05.

## Correctness

| Check | base | head | Y1f | merge `7452d25f` |
|---|---|---|---|---|
| Quick tier, Julia 1.13.3 | — (not rerun) | — (not rerun) | **30/30** (job 14237284; implementer job 14236747) | 29/30 local before the test adaptation (only head's shrinkage-route test, which named `_l2_shrink_cold!`); after it `numerical_kernels` and `groupnorml2` pass |
| Quick tier, Julia 1.10.12 | 28/29, only `@allocated norm(S) == 0` (16 bytes) | 29/30, the same test | **29/30, the same test** (job 14237382) | — |
| `accuracy901.jl` (BigFloat-512 references, Float32/Float64, PB_TRIALS=6) | 0 violations | 0 violations, 0 regressions | **0 violations, 0 class failures, 0 regressions; max errors identical to base for every family/op/type** | 0 violations, 0 regressions |
| NaN/Inf/zero/infinite-weight/alias class checks | pass | pass | **pass**; 4,096-case exceptional grid bitwise equal to base (implementer) | pass |
| Allocation regressions (screen and campaigns) | — | 0 | **0** | 0 |

**The 1.10 failure** is the known pre-existing one: `test/numerical_kernels.jl` `@allocated norm(S) == 0` on Julia 1.10. CI runs current Julia only.

### BigFloat: ordinary inputs and documented directed-rounding differences (not blockers)

`dmpfr901.jl` covers 8,192 cases per precision and mode:
- the screen `kkt_violation(GroupNormL2(λ, [1:m], [w]), g, 0, 1)`;
- precisions 256 and 512;
- RoundNearest, RoundUp, RoundDown and RoundToZero;
- λ, w and g with exponents up to MPFR's limits.

**Results:**
- **λw inside MPFR's exponent range** (5,888 cases per mode): Y1f is bit-identical to base in all modes at both precisions. Head differs from base in 36 (RoundDown) and 60 (RoundToZero) of these cases.
- **Error against a reference at 8× the precision**, rounded once in the active mode:
  - every tree is within 0.085 bound units, where the bound is `(m+8) eps (‖g‖ + λw)`;
  - base, Y1 and Y1f have the same max error and the same count of results that are not correctly rounded (for example 1,367 of 5,888 at precision 256, RoundToZero);
  - the directed modes are faithful but not correctly rounded, as in base.
- **λw outside MPFR's range** (D-X class): every tree differs from base, as decided in #901 (Y1f 208–480 cases per mode). No special code was added, per the maintainer's ruling.
- **Codex's D-MPFR reproducer** (precision 256, RoundToZero, all operands `floatmin(BigFloat)`): base 0, head `floatmin`, Y1f 0.

Ordinary BigFloat inputs go through the same generic code as before: the IEEE fast paths dispatch on `Float32`/`Float64` only.

## Changes for the #901 agent to integrate

Apply `0eda76a9..0bd9974c` onto `1d8bad5d`, or take `7452d25f` (`exp/901-Y1f-on-head`), which is exactly that merge.

**The two conflicts**, both in `src/numerical_kernels.jl`:
1. **`_l2_term_cold`:** keep head's `_scaled_product_norm`. Its body is identical to Y1f's `_norm_product`; delete `_norm_product` and its testset "scaled norm product equals the three-factor kernel", which head's `_scaled_product_norm` tests cover.
2. **Comment above `_l2_expsum_add`:** either wording. Both trees made it `@inline`.

**Head's shrinkage-route test** ("shrinkage kernel at exceptional parameters and values", from `04821b7e`) calls `_l2_shrink_cold!`, which Y1f removes. In `7452d25f` the test's cold reference is
`(iszero(w) | isinf(w) | isinf(λ) | iszero(λ) | iszero(γ) | isinf(γ)) ? _l2_shrink_fixed!(…) : _l2_shrink_values!(…)`.
It also checks `_l2_shrink!(…, _l2_shrink_fixed!)`, the `prox_block!` form.

### Function by function (Y1f vs `0eda76a9`)

**`src/functions/groupnorml2.jl`:**
- **`_group_value_checked`:** the inline part keeps only the ordinary test, `_underflow_checked`, and `_sum_kept(s, length(groups))`. Everything below moves to a new `@noinline _group_value_unkept`, so the callers' group loops do not carry the scans. It runs:
  1. the zero-argument scan, now `_all_zero_early`: blocks of 512, returning at the first nonzero block;
  2. `_sum_kept(s, _penalized_count(f.w))`, where the new `_penalized_count` is a vectorized count of nonzero weights;
  3. `_group_value_recomputed`.
- **`_group_value_recomputed`:** skips zero-weight groups (`iszero(w) && continue`).
- **`_group_prox_all!`:** decides the level/step exits once per call. If `λ0` or `γ` is zero or infinite, it runs `_group_prox_loop!(_l2_shrink_fixed!, …)`, otherwise `_group_prox_loop!(_l2_shrink!, …)`. The new `@noinline _group_prox_loop!` gives one compiled loop per case, and LLVM still hoists the threshold out of the ordinary loop.
- **`prox_block!`:** calls `_l2_shrink!(…, γ, _l2_shrink_fixed!)`, so the parameter exits are inlined and no call is made.

**`src/numerical_kernels.jl`:**
- **`_power_factors`:** new Float32/Float64 method. For `k` in the normal range it builds `2^k` from its bit pattern (`_normal_power`) with second factor 1. The old body becomes `_power_factors_split`, which is still the generic method.
- **`_square_exponent(x::AbstractArray{<:Union{Float32,Float64}}, nothing, R)`:** new method from `_maximum_abs` on the bit patterns, one vectorized pass with the NaN > Inf > finite order. The scan is renamed `_square_exponent_scan` and stays the generic method; the tests compare the two. This method is the cause of the NaN-input gap to head ([follow-ups](#possible-follow-ups-not-done-outside-the-three-screen-budget)).
- **`_l2_zero_group`:** new. A bit-OR over `G` for Float32/Float64; other types use `_l2_all_zero`. Used only by `_l2_shrink!`'s ordinary zero test; the value term keeps `_l2_all_zero`.
- **`_l2_shrink!(y, x, G, λ, w, γ, fixed::F = _l2_shrink_fixed_call!)`:** the fallthrough
  1. copies an unpenalized group and returns `_L2_ZERO`, so the zero-term code now also covers `w = 0`; the comment on the codes is updated;
  2. runs `fixed(…)` for an infinite weight, an infinite or zero level, or a zero or infinite step;
  3. otherwise runs `_l2_shrink_values!`.
- **`_l2_shrink_cold!` is removed:**
  - `_l2_shrink_fixed!` is now `@inline` and holds the parameter exits in their old order;
  - `@noinline _l2_shrink_fixed_call!` is its out-of-line wrapper, the default for the ordinary group loop, whose code is unchanged;
  - `_l2_shrink_values!` is unchanged.
- **`_l2_value` / `_l2_value_cold(λ, x, T)`:**
  - the `s` argument is dropped;
  - the cold value uses `_square_exponent`, then `_square_scaled_sum`, then `_scaled_product`, directly instead of the `_scaled_norm` frame;
  - NaN → `T(NaN)`, all-zero → 0, and an infinite level or entry → `Inf`.
- **`_l2_term_cold`:** product without a third frexp (`_scaled_product_norm` in head).
- **`_l2_expsum_add`:** `@inline` (as in head).
- **`_l2_screen_cold`:** the approved exception. The two bounded terms are differenced in the frame of the larger exponent: `ek, ec = _exponent_sum(R, k), _exponent_sum(R, eλ, ew)`, `E = max(ek, ec)`, `d = ldexp(ns, _frame_shift(R, ek, E)) - ldexp(mλ*mw, _frame_shift(R, ec, E))`, `ldexp(d, E)`. This replaces `_pow2_sum`. Head's optimized `_pow2_sum`/`_pow2_weighted` stay for LinearFunction and LeastSquaresLoss.

**`src/utils.jl`:**
- `_zero_bits(vals, lo, hi, U)` is factored out of `_all_zero`;
- the new `_all_zero_early` scans in blocks of `_ALL_ZERO_BLOCK = 512` and returns at the first nonzero block.

**Tests** (`test/numerical_kernels.jl`, +69 lines):
- "group kernel fast paths": powers of two from bits vs split factors, blocked all-zero scan, `_square_exponent` vs `_square_exponent_scan` on random data with zeros, Inf and NaN;
- "group value of unpenalized groups";
- in `7452d25f`, also the adapted shrinkage-route test.

**Docs:**
- `docs/design/numerical-kernels-865.md`: the two table rows for `_l2_value_cold` and the screen difference, and the `_l2_shrink_cold!` split.
- Stale comments to fix at integration: the design table row "Exponent-ordered sum of parts" still names the group screen as a `_pow2_sum` user, and so does head's comment above the two-term `_pow2_weighted`.

## Possible follow-ups (not done; outside the three-screen budget)

1. **NaN inputs** (Y1f 1.4–2.3× head, still 0.1–0.8× base):
   - Y1f's vectorized `_square_exponent` reads every entry, while head's scan returns at the first NaN.
   - A blocked early exit, like `_all_zero_early`, would likely recover most of the gap without a second algorithm.
2. **Groups of 2–5 with overflowing entries in `kkt`** (Y1f ≈ base, head 0.43–0.56× base):
   - head's optimized `_pow2_sum` is faster there than Y1f's direct frame difference, probably the general `ldexp` in `_frame_shift`/`ldexp(d, E)`;
   - but head's form is 1.07–1.12× base on the other small-group screen regimes;
   - a cheaper scaling in the direct form would probably get both.
3. **Zero-group `value` with groups of 2–5 and all-zero groups of 32 on permuted layouts:** about 1.05–1.09, placement-sensitive. Head shares the m32 case.

## Discovery resources and wall times (this round)

- **Jobs:** 34, about 346 core-hours on exclusive 20-core xeon-2640v4 nodes, out of a 2-day limit. Accounting is in `results/final/discovery-sacct-round2.txt`.
- **Implementer:** 3 screens of 3 min each, 6 focus jobs of 45–83 min, 2 quick tiers.
- **Final campaigns:** 14 jobs of 40–41 min each, plus 4 re-runs of the `run.jl` core task of 2 min each. The first head-vs-base jobs lacked `Project.toml`, `Manifest.toml` and `differential.jl` beside `run.jl`, a provenance read; `submit_final.sh` is fixed.
- **Joint screen:** 3 min.
- **Quick tiers:** 4 jobs of 11–13 min. The 1.10 jobs end FAILED only because of the pre-existing test.
- **Local runs** (E-cores 12–19 only, no timing): accuracy, D-MPFR, merge tests.

**Total for the experiment:** about 94 jobs and 547 core-hours (first round: about 60 jobs, 201 core-hours).

## Artifacts (branch `exp/901-harness`)

- **Results:** `experiment901/results/final/`
  - `campaign/`: per-pair summaries and CSVs (`head-base`, `Y1f-base`, `Y1f-head`, `merge-base`);
  - `screen/`, `accuracy/`, `dmpfr/`, `tests/`, and the sacct accounting.
- **Scripts:** `discovery/submit_final.sh` (pairwise final campaign) and `discovery/finsum.sh` (summary).
- **Raw run directories:** `/scratch2/mkolar/pb-901x/runs/final-*` on Discovery.
