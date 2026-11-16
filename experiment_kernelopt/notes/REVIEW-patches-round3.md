# Third-round focused review

Base: `18b2714955c224300f8924aeb3a205b84c62be57`. Both prior reviews were read. All four diffs apply cleanly. No tracked file was edited. Findings below concern the new ElasticNet extension and B925; the inherited A3 implementation/design findings are not re-reviewed.

Julia **1.10.13 / 1.13.4**, one Julia thread, affinity **0 / 12** (performance / efficiency core). Trees were included in separate modules. Timings use warmed BenchmarkTools calls, alternating tree order, fresh processes, and three or five repetitions; the maintainer harness uses seven rotating rounds with an A/A control. Ratios mean candidate/base. Small-call results depend on compiler/caller layout. Raw evidence, source trees, scripts, LLVM and `timings.tsv` are in `.codex-probes/round3/`; its README identifies two overlapping initial timing screens excluded from the findings.

“Bit-identical” below uses the harness's `isequal` convention: exact finite values and signed zeros, NaN for NaN. A separate raw-NaN-bit screen found no differences, but the tests do not establish a payload/sign guarantee for every NaN.

## A3r — new ElasticNet extension: REJECT

**BLOCKER — Both newly changed sites introduce reproducible small-call regressions.** Locations: `review3/patch-A3r.diff:11,14`. These are supported inputs, independently of the inherited A3 small-call findings. The scalar-step zero-level case reaches the newly changed zero-λ1 line; the ordinary array-step cases reach the newly changed `ordinary` line.

| Public `prox!`, unweighted | Master → A3r | Five-repetition ratio range |
|---|---:|---:|
| 1.10/core 0, Float32, λ1=0, λ2=0.5, scalar γ=0.7, n=16 | 6.666 → 10.983 ns | 1.647–1.676 |
| 1.10/core 0, Float64, λ1=0, λ2=0.5, scalar γ=0.7, n=8 | 6.861 → 8.134 ns | 1.170–1.186 |
| 1.13/core 0, Float32, λ1=0.3, λ2=0.5, ordinary array γ, n=32 | 17.438 → 18.441 ns | 1.056–1.060 |
| 1.13/core 0, Float64, same ordinary parameters, n=9 | 14.560 → 15.474 ns | 1.062–1.066 |

Data are `2*randn(Xoshiro(910),T,n)` and ordinary steps `0.5+rand(...)`; see `bench.jl` for generation order. `A-*-p0-2.csv` supplies these measurements. An attribution control, **A0 = A3r with master's elasticnet.jl restored**, isolates these new edits: fresh-process `A-v-A0-lts-p0.csv` repeats the first two cases at **6.666 → 10.780 ns** and **6.860 → 8.235 ns**; `A-v-A0-release-p0.csv` repeats the last two at **17.423 → 18.469 ns** and **14.558 → 15.475 ns**. Thus these are not merely inherited NormL1 regressions. Core-12 screens also contain ordinary unweighted array losses on 1.13; the zero-level losses above do not occur universally across cores/versions.

Smallest safe fix: retain `shrink` at the losing new call sites, or select the new formula only for demonstrated winning caller/size configurations and remeasure the public calls. The gate results below do not resolve these small-call losses.

**NOTE — The maintainer's large gate mostly wins; no scalar-loop regression was found.** Exact unchanged `ridge910.jl`, core 0, seven rounds, weighted cases:

| Julia / T | Scalar `prox_block` loop, n=1024, ns | Scalar-step dense `prox!` ratio, 1024 / 65536 | Array-step dense `prox!` ratio, 1024 / 65536 |
|---|---:|---:|---:|
| 1.13 / Float32 | 850 → 596 | 0.663 / 0.722 | 0.936 / 0.936 |
| 1.13 / Float64 | 850 → 596 | 0.652 / 0.759 | 1.001 / 1.008 |
| 1.10 / Float32 | 1383 → 791 | 0.517 / 0.519 | 0.941 / 0.947 |
| 1.10 / Float64 | 1383 → 823 | 0.469 / 0.611 | 0.998 / 0.998 |

For **unweighted array steps**, 1024 / 65536 ratios are **1.036 / 1.034** (1.13 Float32), **1.003 / 1.010** (1.13 Float64), **0.945 / 0.950** (1.10 Float32), and **0.999 / 1.001** (1.10 Float64). Harness allocations are zero. Core-12 weighted scalar-loop ratios at n=1024 are **0.640/0.642** (1.13, Float32/64) and **0.402/0.399** (1.10). The separate screens cover every size 1–32, both weights/types/steps/versions/cores, and zero λ1; at zero λ1 and n=65536, core-0 scalar-step ratios are **0.697–0.812** (1.13) and **0.530–0.650** (1.10), while array-step ratios are **0.987–1.001** and **0.939–1.001**. The large zero-level runs are in `A-zero-large-*`. Positive-λ1 scalar-step gains also include the inherited `_l1_prox_threshold` change, so they are not attributed solely to the two new lines.

**NOTE — Threshold preconditions are sound for valid parameters.** Source dependencies checked: `_threshold_product` (`src/numerical_kernels.jl:82`), `_elastic_ordinary_step` (`src/functions/elasticnet.jl:114`), and `_ridge_needs_scaling` (`src/functions/sqrnorml2.jl:361`).

- At λ1=0, the threshold is exactly +Inf if `isinf(w)`, otherwise +0, regardless of step. This also handles mutated NaN weights. Even an invalid mutated −Inf weight takes +Inf on this line.
- At nonzero λ1, `ordinary=true` requires γλ1 in the positive normal range of the working type. With valid λ1, γ is positive and finite. The weight's **value** is not range-tested: −0 yields a −0 threshold; positive Inf yields +Inf; a mutated NaN yields NaN. All satisfy the helper's `!(c < 0)` domain. Conversion/underflow/overflow cannot create a negative threshold from these factors.
- λ2=Inf does not exclude the ordinary route: `_ridge_needs_scaling` returns false for that nonfinite factor, and `_elastic_fixed` supplies +0 for γ≥0. At λ1=0, negative/NaN array steps can also pass the selector in some ridge configurations; the zero-level threshold remains +0/+Inf. Nonzero λ1 rejects those steps from the ordinary route. Public negative/NaN **scalar** steps fail before writes; `prox_block` is unchecked. Their retained exceptional arithmetic agrees with master.
- −Inf weights are invalid at construction. Mutating a nonzero-λ1 weight to −Inf can produce a negative threshold despite `ordinary=true`; construction does not prevent later mutation. That is outside the parameter contract, not a supported-input counterexample, and does not justify a validation scan.

**NOTE — Numerical and test review passes.** Both versions: harness check-only **76/76 identical**, complete patched `elasticnet.jl` tests (including **983 new assertions per type**), and `elasticnet_extremes.jl` pass (the latter retains four broken tests). The new reference independently spells master's soft threshold while trusting the unchanged threshold/ridge kernels; it covers both direct sites, three selection blocks, signed zeros, infinite levels/weights, mutated NaNs, exceptional steps, matrices, exact aliases and fused values. Reviewer master comparisons additionally cover Float16, mixed data/step/weight types, integer/BigFloat levels, invalid scalar steps, and combined y===x===γ. No numerical repair is needed. The added integration tests are homogeneous Float32/64; retaining mixed/fallback fixtures would strengthen them.

## B925 — APPROVE WITH CHANGES

**Costs relative to #925: more at some tiny NormL1 calls, chiefly as a percentage of its cheaper baseline; no greater scan complexity or general absolute failed-pass tax.** The whole-array fallback trade-off itself is already present in #925 and is not a new correctness finding.

**MAJOR — Tiny ordinary calls can lose where equivalent #925 calls win, and tiny fallback percentages are larger.** Locations: `review3/patch-B925.diff:23–41`. Representative clean five-repetition core-0 results on 1.13, unweighted:

| Case | NormL1 master → B925 | ElasticNet pre925 → master |
|---|---:|---:|
| Float32, n=1, ordinary | 4.116 → 6.860 ns (1.667×) | 4.946 → 9.212 ns (1.863×) |
| Float32, n=8, ordinary | 8.744 → 9.212 ns (1.054×) | 19.641 → 11.370 ns (0.579×) |
| Float32, n=8, last step zero | 9.389 → 15.094 ns (1.608×) | 20.361 → 26.813 ns (1.317×) |
| Float64, n=8, last step zero | 9.552 → 15.297 ns (1.602×) | 20.810 → 27.511 ns (1.322×) |

The last two calls add **5.7 ns for L1 versus 6.5–6.7 ns for ElasticNet**, so describing L1 as introducing a larger absolute pass would be wrong. On 1.10/core 0, n=8 last-zero ratios are **1.486 / 1.240** (L1 / ElasticNet, Float32) and **1.462 / 1.274** (Float64); core 12 likewise shows larger L1 percentages. At n=1, already-merged #925 itself has roughly 1.7–2.3× ordinary-call ratios across the measured configurations. Tiny regressions alone therefore do not distinguish this algorithm from #925; the extra ordinary losing cells and L1 percentages do.

Smallest change: bypass the selection pass for losing tiny lengths, keeping the generic loop, and measure the actual public caller after that change; a fallback helper alone does not guarantee elimination of caller overhead. No request to redesign merged #925 follows from these measurements.

First Float64 unweighted array-step calls, core 12, three fresh processes with alternating order: B925 adds **32–59 ms** versus #925's **47–73 ms** on 1.13; on 1.10, **26–53 ms** versus **21–49 ms**. These elapsed/compilation costs are comparable and order-dependent (`compile-*.csv`), rather than an additional 0.1-second penalty beyond #925.

**NOTE — Large ordinary gains and fallback costs resemble #925.** Core-0 unweighted median ratios, **L1 B925/master ; ElasticNet master/pre925**, same inputs within each pair:

| Julia / T | Ordinary, n=1024 / 65536 | One zero at first entry, n=1024 | One zero at last entry, n=1024 / 65536 | All zero steps, n=1024 |
|---|---|---|---|---|
| 1.13 / Float32 | 0.167 / 0.178 ; 0.205 / 0.204 | 1.024 ; 1.023 | 1.071 / 1.072 ; 1.092 / 1.091 | 1.016 ; 1.019 |
| 1.13 / Float64 | 0.328 / 0.327 ; 0.529 / 0.528 | 1.029 ; 1.044 | 1.128 / 1.131 ; 1.171 / 1.166 | 1.013 ; 1.031 |
| 1.10 / Float32 | 0.154 / 0.036 ; 0.188 / 0.086 | 1.014 ; 0.924 | 1.053 / 1.062 ; 0.967 / 1.006 | 1.085 ; 1.144 |
| 1.10 / Float64 | 0.292 / 0.067 ; 0.565 / 0.230 | 1.019 ; 1.006 | 1.093 / 1.055 ; 1.097 / 1.054 | 1.112 ; 0.993 |

A first exceptional entry scans one 256-entry block; a last exceptional entry scans the entire array and then reruns the entire generic loop. The 1.10 all-zero cells are layout-sensitive: a fresh seven-repetition confirmation changes the Float64 L1 ratio from **1.112 to 1.044**. All-Inf Float64 steps at n=1024 are also layout-sensitive: 1.13/core 0 gives L1 **1.194** in one clean process and **0.935** in the fresh confirmation, versus ElasticNet **0.999 / 1.027**; 1.10 L1 gives **0.981 / 1.012**. These are not stable extra fallback blockers. Screens include **all lengths 1–16**, weighted/unweighted Float32/64, all-Inf steps as well as all-zero steps, both versions/cores, 256/257 and large arrays. On core 12 ordinary n=1024/65536 L1 ratios span about **0.20–0.56**; equivalent ElasticNet can range from a large gain to a small loss. Raw tables avoid turning a gate on one CPU core into a universal speed claim. Pre925 contains source from precisely `8d506d44`, including #923, so this comparison isolates #925.

**NOTE — Correctness, dispatch, validation and aliases agree with the generic loop.** The wrapper calls exactly the ordinary `_l1_prox_threshold` expression, including working-type promotion and conversion. A normal γλ implies finite positive λ and γ, making the exceptional infinite-level fixed-zero override unreachable. There is no need to test the final weighted threshold: zero/Inf/NaN/negative mutated weights still go through the identical `shrink` expression. Negative/NaN steps retain NormL1's generic arithmetic.

The deviations from ElasticNet are necessary: L1 has no ridge predicate/application or `_elastic_fixed`; it has **no zero-level exemption**, since λ0=0 is an exceptional product for its scalar kernel; its wrapper directly calls `shrink`; the invoke helper has an L1-specific name. The 256-entry bounds, concrete-type guard, whole-array decision and invoke signature otherwise match #925. No harmful deviation or ambiguity was found, including ElasticNet's same-shaped method.

SparseIterate→SparseIterate keeps its stored-entry loop and does not scan unused steps. Dense output from SparseIterate, and BlockIterate's dense route, take the new method; values and block activity agree with the generic loop. Symmetric destinations still reject array steps with ArgumentError; dense output from SymmetricSparseIterate with array steps remains unsupported by dispatch (MethodError). Public one-based/type checks precede `_checkwrite`, weight extent, and step extent, all before writes; invoke repeats structural checks as #925 does. Same-length differently shaped steps retain linear indexing. Exact y===x, y===γ, y===x===γ are safe: the scan finishes first and each loop entry reads its operands before storing. Partially overlapping views remain outside the documented guarantee (`docs/src/operations.md:378`).

**NOTE — The Bool reduction vectorizes on both versions; integer extrema have type-dependent costs.** Actual `_l1_ordinary_steps` and master `_elastic_ordinary_steps` LLVM contains vector `fcmp`, `<8 x i1>` Float32 / `<4 x i1>` Float64 accumulators and horizontal conjunction on **both 1.10 and 1.13**, without adding `@simd` to the submitted Bool loop. The old B2 comment about Julia 1.10 is not evidence about these methods.

Isolated passes, core 0, ordinary unweighted steps, **256-entry / 65536-entry time in ns**:

| Julia / T | L1 Bool | B2 integer min/max | ElasticNet Bool | Two-product integer min/max |
|---|---:|---:|---:|---:|
| 1.13 / Float32 | 17.6 / 4099 | 10.9 / 2769 | 53.1 / 12926 | 20.9 / 5327 |
| 1.13 / Float64 | 32.9 / 7988 | 46.4 / 11645 | 101.4 / 25514 | 86.5 / 21781 |
| 1.10 / Float32 | 17.4 / 4320 | 10.8 / 2769 | 51.9 / 13015 | 20.6 / 5207 |
| 1.10 / Float64 | 32.1 / 8077 | 46.1 / 11463 | 101.3 / 25460 | 86.5 / 21854 |

The L1 integer probe is the B2 formulation, applied in the same 256-entry blocks. The ElasticNet integer probe reduces extrema for **both γλ1 and γλ2**, testing the threshold and ridge bands; it is equivalent for the same-type positive finite levels used here, **not** a complete replacement for zero/Inf/mixed-level semantics. Integer reductions also vectorize. Core-12 results are retained in `pass-bits-*-e12.csv`. These are measurements, not a recommendation to change merged code.

**NOTE — Added tests pass and cover the broad signature.** The complete patched `infinite_weights.jl` passes on both versions; additions comprise **18,255 + 13,353** array assertions and **122 + 122** other-array assertions. References call unchanged `prox_block` or invoke the unchanged generic loop. Tests cover selection-block boundaries, exception positions, levels/weights/steps, mixed/BigFloat levels, views/reshapes/abstract element types, allocations, sparse/block routes and dispatch ambiguity. They compare fused values on the correct `_value_sep` route. Useful additions would directly assert selector true/false results and add combined exact aliases, resized-weight/output sentinels and offset rejection; reviewer probes cover those latter cases. No test defect or supported-input numerical mismatch was found.

## Vr

**Round-2 findings still stand:** applies cleanly with identical changed lines; #923's checks remain ahead of the changed reductions, and #909's ElasticNet values use `_value_unchecked`/`_value_sep`, not V's standalone NormL1 value route; no new interaction found.

## D3r

**Round-2 findings still stand:** applies cleanly with identical changed lines; #923's preflights remain ahead of the kernel, and #909 uses ridge kernels rather than `_l2_shrink!`; no new interaction found.

Independent reviewer differential checks total **28,082 passing assertions per Julia version**; offset sentinels pass for master and all four patched trees. This was a focused review, not a full package-suite run. Threshold/routing dependencies were source-audited; existing recovery mathematics outside the changes was trusted. Snapshot manifest and source bundle are preserved with the probes; final source verification is recorded in `snapshot-verification.log`.
