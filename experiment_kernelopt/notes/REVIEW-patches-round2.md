# Second-round independent review

Reviewed against master `50db4b1ef376cbe01831c3dd019b6be0027db0b6`. I read `REVIEW-patches.md` first. **A3, V, and D3 are REJECT as submitted under the stated performance rule; FINAL2 is consequently not production-ready.** No supported-input numerical discrepancy was found in the submitted patches. B2 is a genuine performance trade-off and is correctly excluded from the proposed production set.

References such as `review2/patch-V.diff:39` mean physical lines of the supplied diff. Scratch trees, scripts, raw logs, and `timing-summary.tsv` are under `.codex-probes/round2/`. No tracked file was edited.

Verification used Julia **1.13.4 and 1.10.13**, with affinity **0** (performance core) and **12** (efficiency core). Master and patched trees were included in separate modules in the same session. BenchmarkTools measured warmed calls, alternating master/patch order on successive repetitions. `bench.jl` uses three repetitions; `confirm.jl`, `stableprox.jl`, and `dscan.jl` use five. Three fresh release processes ran `confirm.jl`. Small-call results depend materially on caller and code layout: some individual cases change sign between processes. The rejection does not assume a universal slowdown percentage; it relies on repeated ordinary callers, including large Julia 1.10 regressions that also appear in FINAL2. Times below are medians unless a range is stated. Ratios are **patched/master**.

The combined `numerical_kernels`, `norml2`, and `groupnorml2` tests passed on 1.13, as did `core_functions`, `numerical_recovery`, `ridge_precision`, and `infinite_weights`. On 1.10, the combined norm/group/ridge tests and **all added numerical tests** passed separately. The complete numerical-kernel file has inherited failures: the allocation assertion at master line 735 reported in round 1, and, after skipping only that assertion in the scratch harness, the Float64 group-value assertions at lines 1018–1019. The latter two failures reproduce on unchanged master (`tests-M-lts.log`), so they are not attributed to these patches. B2's complete changed test files passed on 1.13; its public step tests and added primitive tests passed on 1.10. These are targeted checks, not a claim that the entire repository suite passed.

A representative reproduction command is:

```sh
JULIA_DEPOT_PATH="$PWD/.codex-probes/depot:$HOME/.julia:" \
  timeout 900 taskset -c 0 julia +lts --startup-file=no --project=. \
  .codex-probes/round2/bench.jl A
```

Other modes are `V`, `D`, `B`, and `F` (FINAL2). The scripts add the existing scratch BenchmarkTools environment to `LOAD_PATH`; repository dependencies/manifests were not changed.

## A3 — REJECT

### Round-1 findings

| Finding in A / A2 | Round-2 status |
|---|---|
| A: unconditional hardware-instruction and exhaustive-test claims | Partly resolved. The select-based `shrink` is gone; the new comment still overstates hardware mapping and its Float16 test coverage. |
| A: scalar IEEE equivalence | Preserved on the new helper's stated domain; independently verified on the actual function. |
| A: public integration/aliasing coverage missing | Improved: actual public routes and exact in-place output are now tested. Additional reviewer differential tests also pass. |
| A2: slowdown relative to A | That comparison is superseded. I compared A3 with **master** and found new disqualifying small-call regressions. |
| A2: duplicated soft-thresholding implementations | Unresolved: master `shrink` and the projection/subtraction formula coexist. |
| A2: tests did not call the helper | Resolved. The new tests call `PB._shrink_nonneg` directly. |
| A2: nonnegative-threshold precondition and mutated weights | Valid for supported parameters. Construction still does not prevent later negative-weight mutation; the revised wording says valid parameters, which is appropriate. |
| A2: operation count / NaN explanation | NaN explanation corrected; hardware/operation-count wording still needs qualification. |

### Findings

**BLOCKER — Ordinary small-array calls still lose to MASTER.** The changed call at `review2/patch-A3.diff:10` affects every ordinary `prox_block`, scalar-step dense loop, and generic array-step loop. Use `x=fill(T(1.1),n)`, level `T(0.3)`, scalar step `T(0.5)`, and, when weighted, `w=fill(T(0.75),n)`. On Julia 1.10/core 0:

| Public `prox!` case | Master | A3 | Evidence |
|---|---:|---:|---|
| Float32, weighted, n=8 | 4.13 ns | 6.42 ns | +49–57% over five repetitions, `confirm-lts-1.log` |
| Float32, weighted, n=9 | 5.11 ns | 6.68 ns | +30–34%, same log |
| Float64, unweighted, n=8 | 3.74 ns | 6.27 ns | +65–68%, same log |

The broader `A-lts-p0.log` independently finds weighted Float32 n=8 at +47%, and FINAL2 retains +47–48%. A conventional noinline wrapper around the public call (`stableprox.jl`) retains the losses: Float32 weighted n=8 **4.52 → 6.86 ns**, Float64 unweighted n=8 **3.93 → 6.87 ns**. These are substantially beyond plausible subnanosecond measurement noise. Julia 1.13 also has losing tiny callers: the wrapper measures Float64 unweighted n=4 **4.13 → 5.30 ns**, with every repetition above +22%; the direct-call screen has Float64 n=2 **3.56 → 4.24 ns**, +19%. The exact losing cases on release vary with code layout, which is why the 1.10 evidence is particularly useful.

Scalar coordinate calls and exact in-place loops were also measured for both types and both weight configurations. Standalone A3 scalar results mostly improve or are layout-sensitive; I do not claim an invariant scalar slowdown. **The actual production composition** does expose one: `coord(p,f,x,g) = p(f,x[1],1,g)` behind a noinline boundary, unweighted Float32, gives **2.384 → 2.576 ns** on core 0 (+7.7–8.2%) and **4.642 → 5.154 ns** on core 12 (+11–22%), three repetitions each (`F-release-p0.log`, `F-release-e12.log`). In-place calls are not exempt either: the initial release screen has unweighted Float64 n=2 at **4.00 → 4.24 ns**, about +6%. I checked sizes 1,2,3,4,5,7,8,9,15,16, plus 64 and 1024, on both Julia versions and both cores.

Smallest safe fix: retain master's kernel on losing supported paths until a replacement passes these measurements. Large-array gains and the supplied cluster measurements do not waive the small-call rule. Merely replacing `shrink` globally with this formula would not repair these timing findings.

**MAJOR — The no-duplicated-implementations requirement remains unresolved.** `review2/patch-A3.diff:19–39` retains one soft-thresholding formula and adds another, with call sites responsible for their domain distinction. This is the same design issue raised for A2, notwithstanding the now-correct tests. If domain-specific numerical implementations are to be allowed, that needs to be an explicit relaxation of the project rule; this review does not assume one.

There is also a concrete qualification to the proposed alternative of changing IEEE `shrink` itself. Its documented domain is indeed `c >= 0`, but **public NormL1 routes retain arithmetic at negative steps** (`docs/src/numerics.md:171–176`). For example:

```julia
prox(NormL1(1.0), [0.5], -1.0)  # master and A3: [1.0]
# An unconditional clamp-form IEEE shrink would give [1.5].
```

The exceptional branch of `_l1_prox_threshold` calls `shrink` at that negative threshold. Thus changing negative-threshold behavior is not confined to unsupported direct `shrink` calls. A canonical IEEE implementation is preferable in principle, but an unconditional replacement needs a public-contract change; a centralized negative-threshold compatibility branch would need fresh correctness/performance review. Under the present rules, keep master rather than introduce either an unapproved contract change or the submitted duplicate.

**NOTE — The helper is numerically equivalent on its claimed domain.** `differential.jl` compares the actual A3 helper with master's actual `shrink` for **all 2,214,592,512 Float16 pairs with `!(c < 0)`**, including both NaN signs and -0: zero mismatches. The added edge/random/near-threshold Float32/64 tests pass on both versions. For finite nonnegative c, the selected clipped value is v inside the interval and the relevant endpoint outside it; subtraction therefore reproduces the active term of master's formula. The +0 restores master's zero sign. NaN c reaches the subtraction, NaN v survives it, and infinite endpoints reproduce master's NaNs. Non-IEEE AbstractFloat types forward to `shrink`; mixed/integer arguments retain promotion. Reviewer public-route differential tests add **10,080 assertions per version**, including exceptional thresholds/weights, negative/NaN steps, scalar and array steps, exact in-place output, and fused outputs/values: all pass. No new recovery algorithm or ambiguity was found.

**MINOR — Helper coverage is strong; public integration coverage is useful but narrow.** The added tests at `review2/patch-A3.diff:116` onward now exercise the package helper, not a local imitation. Public tests at line 219 onward exercise prox, prox!, scalar `prox_block`, array steps, fused calls, signed zero, and mutated NaN weights. This resolves the former missing-helper test finding. Add one nontrivial ordinary weighted/unweighted exact in-place fixture and public overflowed/subnormal-threshold fixtures against an independent reference; most of those guarantees currently rely on inherited tests and reviewer probes. The two `copy(x)` arguments at diff line 239 are distinct arrays; that particular assertion is not an alias test, although the preceding zero fixture does test `z === x`.

**MINOR — Qualify the comment.** `review2/patch-A3.diff:23–31` omits threshold negation from its operation count, implies hardware maximum/minimum selection independently of compiler/type/target, and does not clearly say thresholds are sampled in the committed Float16 test. Say “all Float16 inputs against sampled nonnegative thresholds”; describe the arithmetic without promising an instruction count. The reviewer's exhaustive run does not make the committed test exhaustive.

## V — REJECT

### Round-1 findings

| Finding in C / C2 | Round-2 status |
|---|---|
| C: singleton slowdown | The new loop is bypassed, but fallback call overhead still produces >5% losses. Not resolved by `@noinline`. |
| C: long sequential accumulation changes error growth | Resolved in scope: the new window ends at 1024 and retains Base above it. |
| C: no stronger universal few-eps L1 contract established | Unchanged; the new finite window has a defensible conservative bound. |
| C: NormL1 views never reach the new loop | Now documented accurately. |
| C: missing recovery/fallback tests and unqualified speed claim | Recovery coverage improved substantially; explicit fallback-type coverage remains thin. Speed is still caller/length-dependent. |
| C2: SqrNormL2 singleton cost | Resolved by removing SqrNormL2 from the change. NormL2 fallback overhead remains. |
| C2: numerical compatibility / error discussion | Improved: there is no longer a 2^17-entry new reduction; see precise comparison below. |
| C2: no tests for newly selected norm routes | Resolved for NormL2: substantial new tests, including strided views, boundaries and recovery. |
| C2: stale SqrNormL2/shared-cutoff comments | Resolved by reverting that scope extension and using separate value-loop constants. |

### Findings

**BLOCKER — Retaining master's expression does not retain its small-call speed.** The comparisons at `review2/patch-V.diff:39,42`, and the changed call sites at lines 17 and 60, remain in fallback callers. The claim at line 43 that callers “keep the code that they had” is not true as a performance claim. With `@noinline callvalue(f,x)=f(x)`, `x=fill(T(1.1),n)`, and level `T(0.3)`, representative five-repetition results on core 0 are:

| Version / ordinary fallback | Master | V | Repetition range |
|---|---:|---:|---:|
| 1.13, NormL2 Float64, n=1 | 4.324 ns | 4.905 ns | +13–23%, `confirm-release-1.log` |
| 1.13, NormL2 Float64, n=1, another fresh process | 4.122 ns | 4.514 ns | +9.5%, `confirm-release-3.log` |
| 1.10, NormL1 Float32, n=1 | 4.717 ns | 5.492 ns | +16–17%, `confirm-lts-1.log` |
| 1.10, NormL1 Float64, n=8 | 5.784 ns | 6.277 ns | +8.5–8.9%, same log |

Some other fresh layouts eliminate a particular loss; for example, release confirmation process 2 does not lose the NormL2 singleton. This does not establish “no measurable slowdown,” and FINAL2 independently retains the release NormL2 Float64 singleton loss at **4.332 → 4.922 ns**, +13.6%. Empty inputs also sometimes regress. Above 1024, the sampled lengths 1025, 2048, and 8192 did not show a reproducible >5% loss.

Smallest safe fix: retain the master value route until the fallback caller itself meets the rule. Raising the lower cutoff only addresses losses inside the window; it cannot eliminate the length test and layout cost on the fallback route. `@noinline` on the loop is not sufficient evidence of preservation.

**BLOCKER — 16 is not a universally winning lower cutoff, and losses are not monotone in length.** On 1.10/core 0, NormL2 Float64 n=24 measured **8.07 → 9.06 ns** in one repetition and +12–32% over the three-repetition screen; n=100 also repeatedly loses by +17–18%. On 1.13/core 0, NormL1 Float64 n=31 loses **8.240 → 8.734 ns** (+6.0–8.4%) in confirmation process 1 and **7.458 → 8.245 ns** (+10–18%) in process 2. Float32 n=31 in confirmation process 2 loses about +33–38%. These are ordinary finite arrays within the proposed window (`review2/patch-V.diff:35–47`).

Here are the requested first window sizes from the three-repetition screen, median ratios on core 0. They are measurements of that caller/layout, not portable crossover constants:

| Julia | Function / type | 16 | 17 | 24 | 31 | 32 | 33 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.13 | L1 / Float32 | 1.02 | 0.95 | 0.92 | 1.08 | 0.78 | 0.74 |
| 1.13 | L2 / Float32 | 0.90 | 0.81 | 0.89 | 1.03 | 0.62 | 0.62 |
| 1.13 | L1 / Float64 | 0.71 | 0.65 | 0.79 | 0.89 | 0.70 | 0.65 |
| 1.13 | L2 / Float64 | 0.78 | 0.68 | 0.96 | 1.00 | 0.79 | 0.70 |
| 1.10 | L1 / Float32 | 0.78 | 0.91 | 0.92 | 0.93 | 0.46 | 0.45 |
| 1.10 | L2 / Float32 | 0.98 | 0.78 | 0.93 | 0.94 | 0.39 | 0.37 |
| 1.10 | L1 / Float64 | 0.56 | 0.53 | 0.97 | 1.04 | 0.59 | 0.56 |
| 1.10 | L2 / Float64 | 0.54 | 0.50 | 1.29 | 1.14 | 0.57 | 0.54 |

Both versions were also screened on core 12 (`V-*-e12.log`). Many window cases win there; they do not erase the core-0 failures. Smallest fix for this finding alone: bypass the new reduction at measured losing sizes, then remeasure other sizes and fallbacks. A single higher cutoff has not been established as sufficient.

**NOTE — Base's 1024-leaf claim is correct for the actual package reductions, with important qualifications.** I read the installed `base/reduce.jl` on both versions. In 1.13.4, `mapreduce_impl` is at lines 252–274, `pairwise_blocksize` at 311–314, `_mapreduce` at 428–453, and `sum(f,a)` at 535; corresponding 1.10.13 locations are 253–275, 312–315, 428–453, and 535. `sum(f,a)` uses **`Base.add_sum`**, whose block size is 1024 for both abs and abs2. The easily confused 4096 specialization is for `mapreduce(abs2, +, ...)`, which these calls do not use. `inspect-*.ll` records both runtime block-size queries and generated LLVM.

For an IndexLinear array of 16–1024 entries, Base explicitly combines `f(x[1]) + f(x[2])`, then runs an `@simd` reduction over entries 3:n. V initializes +0 and loops over 1:n. On the inspected performance-core target, **both versions and both formulas have four main vector accumulators**, each with **8 Float32 lanes or 4 Float64 lanes**: 32 or 16 simultaneous accumulation chains. Base seeds one lane with the first-two sum; V seeds with zero. Starting at a different offset changes which inputs share a lane and the scalar tail. Julia 1.13 also emits a narrower vector epilogue (4 Float32 or 2 Float64 lanes in the inspected code); 1.10's inspected code has a scalar remainder. Below the main vector trip count, these distinctions matter greatly. Lane counts/unrolling are compiler decisions, not properties guaranteed by `@simd`.

Thus the loops have the **same conservative error class**, not the same bits, error constant, or worst input. For ordinary nonnegative finite terms without overflow, a portable conservative sum bound is approximately `gamma_n = n*u/(1-n*u)`, `u=eps(T)/2`; Base's first-pair initialization can save an addition, whereas adding the initial zero is exact. At n=1024 this conservative bound is about **6.104e-5 for Float32** and **1.137e-13 for Float64**. Squares, final scaling, and the L2 square root add their usual rounding; contraction may also change actual rounding. SIMD chains generally give a smaller bound, but tail handling and the seeded first pair prevent claiming an identical bound from lane count alone. Above the cutoff, Base again bounds leaf length and adds pairwise-tree depth; error still grows logarithmically with the number of leaves, so diff line 34's “does not grow with its length” should say that long **sequential-chain** growth is avoided.

For IndexCartesian strided multidimensional views, Base uses `mapfoldl`, not this IndexLinear pairwise implementation. The new loop can change that order too; the conservative nonnegative-sum bound still applies. This qualification belongs in the comment if it discusses all strided arrays.

**NOTE — Outside-window arithmetic is unchanged, and dispatch is narrower for L1 than L2.** Source inspection and **4,845 reviewer assertions per version** verify bitwise equality outside the window and unchanged Int/Float16/BigFloat fallbacks, in addition to the supplied tests. The dispatch is:

| Route | Can reach `_value_loop`? |
|---|---|
| Unweighted NormL1 on `DenseArray{Float32/64}` | Yes, only lengths 16–1024 and after existing level checks. Ordinary vectors/matrices qualify. `DenseArray` is included in Base's StridedArray alias; this is not formally an Array-only signature. |
| NormL1 on ordinary SubArray views, strided reshaped wrappers, reinterpret wrappers, or weighted arrays | No: those public routes keep the generic separable value computation unless the object itself is a DenseArray. Ordinary `reshape(::Array,...)` returns an Array and does qualify. |
| NormL2 on `StridedArray{Float32/64}` | Yes: Arrays, qualifying strided views/reshapes/reinterpret arrays, including multidimensional and nonunit-stride views, in the window. |
| NormL2 on other wrappers/custom arrays, integer/Float16/BigFloat arrays, SparseIterate, SymmetricSparseIterate, BlockIterate, or sparse arrays | No new loop. `_value_sumabs2` delegates to existing `_sumabs2`, preserving stored-entry handling where previously implemented. |
| SqrNormL2, centered square values, least-squares reductions | Unchanged. |

**MINOR — The tests now address the numerical change well, but not its speed guarantee.** `review2/patch-V.diff:70–141,152–232` tests actual helpers/routes, independent BigFloat references, cutoff neighbors, remainder lengths, matrices/views, overflow/underflow recovery, endpoint priorities, unchanged SqrNormL2, and allocation. Add explicit Float16/BigFloat/integer/sparse fallback fixtures and mixed levels outside T's range; the reviewer probes cover several of these but are not package tests. Keep the first-window and outside-window performance probes: output tests cannot detect either blocker. Qualify the two-to-three-times claim at diff line 32 by measured size/platform.

## D3 — REJECT under the stated slowdown rule

### Round-1 findings

| Finding in D | Round-2 status |
|---|---|
| Guard squares tiny c and incurs severe underflow cost | The square and its severe penalty are gone. A smaller but still >5% failed-guard penalty remains. |
| Zero-decision inequality | Sound with the stronger condition, including rounded conversion of length; one exact statement in the comment needs correction. |
| Return codes, fused values, types and representations | Preserved; new and inherited tests pass as described above. |
| Strong correctness tests but no failed-guard timings | Correctness coverage remains strong. Requested zero/tiny/dense timings now expose the remaining cost. |

### Findings

**BLOCKER — The failed guard still taxes an existing ordinary zero-group path.** At `review2/patch-D3.diff:24`, every zero square sum now performs a length conversion, multiplication, comparison, and branch before master's zero scan. Use 1000 disjoint groups of size n, unit weights and step, all-zero input, and `GroupNormL2(floatmin(T), groups, ones(T,1000))`. These are positive **normal** thresholds. The new guard fails, and both versions return +0 with zero fused value.

| Configuration | Master | D3 | Evidence |
|---|---:|---:|---|
| 1.13/core 0, Float32, n=1 | 3.155 us | 3.592 us | about +14%, five-repetition `Dscan-release.log` |
| 1.13/core 0, Float32, n=5 | 4.315 us | 4.677 us | about +8%, same log |
| 1.13/core 12, Float64, n=1 | 6.442 us | 7.209 us | +11.4–11.9%, three repetitions, `D-release-e12.log` |
| 1.10/core 0, Float32, n=5 | 5.873 us | 6.576 us | +11.5–12.7%, `D-lts-p0.log` |

These are roughly 0.2–0.8 ns extra per group, not D's former underflow catastrophe. At n=1024 the failed-guard overhead is mostly amortized; at ordinary level 0.5 the zero-group shortcut often wins, e.g. Float32 n=5 around **0.82x**, n=1024 around **0.88x** on core 0. Dense nonzero groups generally stay near master, although the 1.13/core-12 screen also finds Float32 singleton dense groups at ordinary level about **+6%**. Those layout-sensitive cases deserve rechecking after any fix. The scripts cover n=1,5,1024, both types, ordinary/tiny thresholds, zero/dense input, both versions and both cores; the large case uses 32 groups to bound benchmark duration.

**Classification under the rule:** I count this as an existing ordinary hot path. A zero group is a routine active-set state; master handles it without recovery, with a finite normal threshold. The absolute size of a penalty is scale-dependent. In particular the Float32 cutoff starts near 4.44e-16 per entry, not 2^-485. The Float64 threshold is extreme, but the stated rule did not grant a scale exemption, and the patch should not define “ordinary” by whether its new guard succeeds. If the maintainer chooses to exclude this regime, that changes the acceptance criterion; numerically D3 is sound. I have not assumed that exclusion.

Smallest demonstrated repair for this particular cost: reverse the two short-circuit operands,

```julia
iszero(s) && (_l2_zero_group(x, G, T) || _l2_threshold_decides(c, length(G)))
```

This uses existing machinery. The scratch `Dscan` variant measures Float32 singleton tiny-threshold groups **3.153 → 3.156 us**, Float64 singletons **3.434 → 3.429 us**, and Float64 n=5 **4.563 → 4.548 us** on 1.13/core 0. It gives up scan elimination for exactly zero groups, but still avoids the cold route for a nonzero underflowing group at a sufficient threshold. It is a candidate repair for the identified zero-group tax, not a fully validated replacement: both cores/versions and dense callers must be rechecked. Division or exponent/bit tricks are not an established simpler/faster substitute for the current multiplication.

**NOTE — The stronger inequality is sufficient, including conversion of n.** Let delta be `nextfloat(zero(T)) = eps(T)*floatmin(T)`. With zero computed sum, each converted entry's squared magnitude is at most delta/2, hence the converted exact norm squared is at most `n*delta/2`. All terms are nonnegative; an addition cannot cancel a positive square to zero. The guard is tested only after the existing normal-threshold checks.

Write `n_hat = T(n)` and `r = 2^-485` (Float64) or `2^-51` (Float32). For a positive Int-sized length, `n_hat` is finite and positive, `n_hat*r` is normal and finite, and multiplication by this power of two is exact. Rounding n down gives `n_hat >= n*(1-u)`. At n=1 conversion is exact; for n>=2, `n_hat^2 >= n` despite that rounding. Therefore a passing guard gives

```text
c^2 >= n_hat^2*r^2 >= n*floatmin(T)/eps(T) >> n*delta/2.
```

The Float32 root is rounded upward by an additional factor sqrt(2), increasing the margin. Wider raw inputs can exceed their converted entries by at most the usual rounding interval, far too little to close the factor of order `1/eps(T)^2`. Thus the cold norm is below c too. This is a sufficient condition, not a tight necessary one. It does not form an underflowing square in execution. `guardproof.jl` checks the actual helper around nonrepresentable lengths and typemax(Int), in addition to the committed boundary tests.

**MINOR — The comment's exact n-squared bound ignores length rounding.** `review2/patch-D3.diff:26–27` asserts `c^2 >= n^2*floatmin/eps`. For Float64 take `n=2^53+1` and `c=Float64(n)*2^-485`: the guard passes, but that exact n-squared assertion is false because n rounds down. This is a proof-comment defect, **not** a false zero decision; the weaker sufficient inequality above holds by an enormous margin. Replace n-squared by the rounded-length expression or state only `c^2 >= n*floatmin/eps`. Add `2^24+1`, `2^53+1`, and typemax(Int) to helper tests; current large lengths in the diff are exactly representable.

**NOTE — Correctness/test coverage remains strong.** Direct tests compare stored outputs **and return codes** with the existing cold implementation, then check nonzero shrinkage independently against BigFloat. Public tests cover zero/nonzero underflowing groups, normal/tiny/overflowing-square thresholds, both sides of the new boundary, ranges/permutations, matrices, mixed precision, compact block routes, exact in-place operation, and fused values at stored output. Float16/BigFloat retain the scan. Empty-group behavior and exceptional parameter exits remain unchanged. No new recovery mathematics, allocation, or dispatch ambiguity was found.

## B2 — genuine trade-off; not acceptable under the strict production rule

### Round-1 findings

| Finding in B | Round-2 status |
|---|---|
| Full-array prepass taxes short and failed calls | Reduced to at most the first failing 256-entry block, with n<8 bypassing scans, but substantial short/failure losses remain. |
| Signature, working precision, and invoke correctness | Signature/precision retained correctly; invoke removed. |
| Validation and exact aliases, repeated fallback checks | Correct; fallback no longer invokes a second set of validations. Cross-block aliases checked independently. |
| Unconditional compiler claims and signed bit-order explanation | Sign explanation corrected. Qualify compiler/vectorization claims by target/configuration. |
| Good numerical tests but missing failure/performance cases | Improved substantially: block boundaries, BigFloat levels, before-write step errors, public aliases. Reviewer adds resized weights and malformed output shapes. Trade-off timings are confirmed. |

### Findings

**MAJOR if considered for production — Failure-path cost is real, not removed by blocking.** `review2/patch-B2.diff:26–37` adds the dispatch/body overhead even when scanning is bypassed, and a failed block performs work that the generic tail repeats. With unweighted level T(0.3), x filled with T(1.1), and steps T(0.5), core-0 1.13 measurements confirm large wins: n=64 gives **0.192x Float32 / 0.436x Float64**, n=1024 **0.143x / 0.390x**. But Float64 n=8 with one zero step at the end gives about **+56–58%**, and all-zero n=64 about **+23–24%** (`B-release-p0.log`). On core 12, eight ordinary Float64 steps give **16.96 → 22.22 ns**, +26–34%; sixteen ordinary steps also regress about +35–37%. n=7 still loses despite bypassing `_steps_normal`: Float32 about +10% and Float64 about +8–9% on core 0. This agrees with the maintainer's trade-off characterization.

There is no demonstrated formulation here that removes all failure cost. Smaller blocks/early exit can reduce wasted scanning but do not make it free, and extra branches can damage the winning vector loop. Moving the generic tail out of line or adding a short dispatcher may improve some layouts but requires measurement; neither removes the already-paid failed scan. A single per-entry implementation might avoid pretesting but has not been shown to retain the vectorization gains. Under the strict rule, leave B2 out. As an explicitly accepted trade-off, its benefit/cost presentation is fair; it should not be described as a universal speedup.

**NOTE — The blocked structure preserves results and exact aliases.** The range test (`review2/patch-B2.diff:59–68`) checks exactly the working-precision product gamma_i*lambda used by `_threshold_product`, not the final weighted threshold. That is correct: master's ordinary predicate is also on that intermediate. Zero/infinite weights and overflow/subnormal final thresholds still use the same expression. Float32 data with Float64 or integer levels correctly uses Float64 working precision; BigFloat levels skip block testing. The signed-integer min/max test rejects negative products, signed zeros, infinities and either NaN sign correctly.

Each block's step scan finishes before that block writes. The successful loop loads x[i], gamma[i], and w[i] before storing y[i]. With y===x, y===gamma, or all three identical, no store changes a later coordinate or a future block's step. A failed block leaves i0 at its beginning, so the tail neither skips nor repeats previously written entries. The generic tail is master's `convert(T, prox_block(f,x[i],i,gamma[i]))`; replacing `_entry(gamma,i)` by indexed access is equivalent for the specialized Array signature. The prepass does not consume or alter exceptional steps. Partial final blocks are handled correctly.

Reviewer differential tests add 332 assertions per version, including y===gamma and y===x===gamma at 255/256/257, 511/512/513, and 1025 entries, with exceptional steps on both sides of boundaries, weighted/unweighted. All agree with actual master. These exact aliases do not imply a guarantee for shifted overlapping views, which do not dispatch here anyway.

**NOTE — Validation order is unchanged and occurs before writes.** The specialization starts with `_checkwrite`, then `_checkparam(gamma,x)`, matching the generic method (`src/functions/separable.jl:143–151`). Public real-array checks still run before entering it. Output size and resized weight errors precede step-length errors exactly as before; same-length differently shaped step arrays retain linear indexing. Reviewer malformed-output/resized-weight/short-step probes preserve exception types and leave sentinel output untouched. Views, mixed element types, differently typed weights, Float16 and other arrays retain generic dispatch. No invoke/repeated validation, new recovery algorithm, allocation or ambiguity is introduced.

**MINOR — Test/comment refinements.** Added public tests exercise both vectorized and failed blocks, in-place output, output-as-steps, matrix shapes, mixed/integer/BigFloat levels, exceptional data, and boundary positions. Add the combined y===x===gamma case and malformed-weight/output before-write cases to the package tests. Qualify “keeps the generic loop scalar” / “vectorize” at diff lines 10 and 57: that is observed code generation, not a cross-target semantic guarantee.

## FINAL2 — composition correct; REJECT for production performance

### Round-1 cross-patch findings

| Earlier interaction | Round-2 status |
|---|---|
| A2 must be compared against A | Superseded: A3 is based on and compared with master. |
| A tests miss A2 helper | Resolved: helper tests are included. |
| B uses shrink while A2 changes another path | B2 is absent, as stated. If combined later, its fast block still calls shrink while its tail would reach A3's helper; benchmark against that actual composition. |
| C affects standalone L1, not fused separable value | Still true for V. |
| C2 value change interacts with D zero-code/fused result | SqrNormL2 change removed. V can affect NormL2 recomputed values; D3's zero code still gives exactly zero. |
| Shared source/test hunks require combined verification | Done. Source/test contents match sequential A3 + V + D3 application. |

**BLOCKER — Composition retains the losing ordinary paths.** `F-lts-p0.log` repeats weighted Float32 n=8 public prox at **4.154 → 6.122 ns** (+47–48%) and unweighted Float64 n=8 at **3.758 → 4.828 ns** (+28%). `F-release-p0.log` retains the scalar Float32 and fallback NormL2 singleton losses quoted above. `F-release-e12.log` has NormL1 Float64 n=31 at **14.598 → 16.950 ns** (+12–16%). Tiny-threshold zero-group costs also survive. Code layout changes the exact percentages, so accepting isolated components would not substitute for accepting this production artifact.

**NOTE — No composition defect was found.** All three diffs apply in order to master. After ignoring patch-generated `.orig` backups, sequential source and test files are byte-identical to FINAL2. The combined tests contain all additions, with module/precision scopes intact, and the norm/group ambiguity checks pass. No B2 specialization is present.

A3 changes L1 proximal arithmetic; V changes only unweighted dense standalone L1 values, so fused L1 values still use `_value_sep` at stored output. V changes standalone NormL2 and the `_L2_RECOMPUTE` value route (e.g. identity steps), while ordinary fused NormL2 values use the shrink kernel's returned norm. GroupNormL2's value reductions do not call V's helper. D3's zero exit uses the existing `_L2_ZERO` code, preserving zero fused values with or without V. SqrNormL2 and least-squares arithmetic are unchanged. The main interaction observed in timings is shared-file/caller code layout, not an arithmetic or dispatch conflict.
