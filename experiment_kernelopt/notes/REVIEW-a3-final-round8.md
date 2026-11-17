The final evidence overturns my provisional round-7 preference for universal A3: **Zen 3 has material regressions under ordinary allocations, and the maintainer's stated condition for approving it is not met. I recommend N (A3n), conditional on its own pending runs confirming the projection below.** I would not merge the 2 MiB variant: it leaves the important sub-cutoff regressions and introduces small-call costs. The 32 KiB variants now demonstrate a relevant advantage, but do not provide complete protection and have more machinery; all cutoff variants also retain an API edge-case defect. Several claims in the question are too strong: the matrix evidence covers two Zen nodes, the listed regressions do not occur in every process, some relevant setup backing is 33%, and tiny in-place calls are penalized. The regression itself is not in doubt.

**A. Independent matrix recomputation; conclusions 1, 2, 5 and 6.**

Commands, from `review-code`:

```bash
python3 round8/setup.py > round8/setup.txt
python3 round8/recompute.py > round8/recompute.txt
python3 round8/summary.py
```

The [source identities and hashes](round8/manifest.json) pin M4 `f8148203`, universal `0271e994`, A3m `87270d4e`, A3d `8d36871b`, A3dr `c5911eb7`, and A3n `684e1ee5f`. Checks use pinned source copies, not the later `pr-A3` working tree. The independent script requires DONE, no CHECKFAIL, unique metadata, matching raw/metadata keys, exactly rounds 1–21 without duplicates, finite positive timings, reference M4, status `ok`, zero `diff_ref` and zero active-set mismatches. **All 36 processes pass**: final matrix, 1008 cases/130368 rows per process; earlier matrix, 1000/150360. See [validation](round8/matrix-validation.txt).

Each process contributes the median of round-wise candidate/M4 ratios; a cell uses their median across six processes. For Zen I pool the two three-process jobs within each campaign, and also retain their separate results. “Reproducible slow” means ratio >1.05 in every process with every M4C/M4 ratio in [0.98,1.02], as in round 7. Validity and a clean control are different: geometric means include valid cells with unclean controls and are descriptive, not workload-weighted estimates.

**The complete requested per-CPU, per-family, per-operation, per-size-class means for BOTH campaigns are in [matrix-gm.md](round8/matrix-gm.md)** ([CSV](round8/matrix-gm.csv)). A3m was not measured in the earlier campaign; its entries correctly say “unmeasured.” The complete all-process >1.05 cells, including those rejected by the control gate and absolute times, are in [matrix-slow.md](round8/matrix-slow.md); [matrix-slow.csv](round8/matrix-slow.csv) and [matrix-cells.json](round8/matrix-cells.json) retain individual process ratios, controls and times.

For orientation, final-matrix means pooled over the four families are below. Entries are **short/medium/large**, candidate time/M4 time; short n≤64, medium n128–8192, large n>8192, both element types. Family means have 14/14/12 cells, respectively.

| CPU | operation | universal | 2 MiB |
|:--|:--|:--|:--|
| Broadwell | prox_out | .9464/.7033/.9286 | .9747/.7053/.9351 |
| Broadwell | prox_in | .8436/.6717/.6679 | .8505/.6719/.6679 |
| Broadwell | prox_incopy | .8591/.6915/.7218 | .8722/.6922/.7219 |
| Broadwell | valprox_out | .9644/.7859/.9463 | .9767/.7874/.9510 |
| Skylake-SP | prox_out | .9648/.8515/.9172 | 1.0023/.8536/.9167 |
| Skylake-SP | prox_in | .9090/.8014/.8051 | .9139/.7980/.8037 |
| Skylake-SP | prox_incopy | .9229/.8205/.8471 | .9392/.8171/.8451 |
| Skylake-SP | valprox_out | .9847/.9007/.9394 | .9996/.9024/.9403 |
| Zen 3 | prox_out | .9584/.7498/.9298 | .9851/.7525/.9365 |
| Zen 3 | prox_in | .8746/.7190/.7169 | .8816/.7196/.7172 |
| Zen 3 | prox_incopy | .8969/.7399/.7520 | .9128/.7405/.7524 |
| Zen 3 | valprox_out | .9664/.8340/.9499 | .9844/.8361/.9541 |

The final matrix's clean slow-cell counts across **all operations** are:

| CPU | universal | 2 MiB | worst reproducible cell, universal; 2 MiB |
|:--|--:|--:|:--|
| Broadwell | 3 | 4 | both: l1/prox_out/Float32/n1 |
| Skylake-SP | 0 | 0 | none meeting the rule |
| Zen 3 | 8 | 5 | enet/prox_out/Float32/n1048576; enet/prox_out/Float32/n262144 |

“Worst” maximizes the minimum process ratio. Broadwell universal is 1.124996–1.125733, **7.0795→7.9646 ns**; 2 MiB is 1.124995–1.208326, **7.0795→8.1121 ns**. Zen universal is 1.279302–1.347862, **111422→147211 ns**, controls .995894–1.007237; 2 MiB is 1.268336–1.345267, **27928→36278 ns**, controls .990996–1.007528. The earlier matrix has universal clean slow counts Broadwell 3, Skylake-SP 0, Zen 1 when all six processes are required; its Zen weighted ElasticNet four-entry block loop is 1.111119–1.138900, 9.8441→11.0747 ns. Separate Zen three-process groups have additional large slow cells; see [summary.txt](round8/summary.txt).

Conclusion 1 is broadly right about Intel's gains and the small absolute size of its reproducible universal losses, but “2 MiB equals universal at large sizes” is inaccurate. For example, Broadwell l1/prox_out/Float32/n1048576 is **.9092 for universal versus 1.0000 for A3m**: the cutoff discards a real gain. Some earlier small block-loop losses are also not one-entry cells.

**Conclusion 5 is false literally.** A3m preserves the medium/large in-place benefit: its direct ratios to universal are approximately one, with no comparable large-array penalty. But Broadwell l1/prox_incopy/Float32/n1 costs **13.5691 ns for universal versus 14.4536–14.7496 ns for A3m**, direct per-process ratios **1.0652–1.0871**, with M4 A/A .99806–1.00003. A3d and A3dr also have reproducible tiny in-place costs. The identity exemption avoids switching long `y === x` calls back to shrink; it does not make short dispatch free. `prox_in` remains repeated stationary-zero input; `prox_incopy` includes copying and supplies the nonzero-input companion.

Conclusion 6 is approximately right about aggregate cutoff costs, but **“universal and N add none” is wrong**: universal's Broadwell n1 losses remain in N's NormL1 projection. There is no added selection logic in U/N, which is a different statement. Final short out-of-place means, combining prox_out and valprox_out, are:

| CPU | universal | A3m | A3d | A3dr |
|:--|--:|--:|--:|--:|
| Broadwell | .95538 | .97572 | .97068 | 1.00187 |
| Skylake-SP | .97473 | 1.00094 | .99451 | 1.01492 |
| Zen 3 | .96241 | .98477 | .98331 | .99645 |

For prox_out alone, the direct short-call A3m/U means are **1.0287, 1.0407, 1.0296**, respectively; A3dr/U is **1.0658, 1.0572, 1.0404**. Thus “about .05” is reasonable as an absolute increase in the combined M4-relative mean, not a universal 5% bound. Sub-nanosecond costs and roughly 1–1.6 ns examples are material fractions of tiny calls. These measurements include dispatch, cutoff and code-placement effects; they do not isolate a pure branch or Ref latency.

**B. Reality, allocation pattern, pages and family difference.**

```bash
python3 round8/probes.py > round8/probe-summary.txt
```

[Process-level matrix evidence](round8/matrix-detail.txt), [probe recomputation](round8/probes.txt), [probe cells](round8/probe-cells.json), and [allocation trials](round8/alloc-cells.json) substantiate the following corrections and conclusions.

The regression is real. The final Float32 n262144 and n1048576 unweighted ElasticNet prox_out cells regress in **all six processes**, with clean controls; valprox_out is about **1.198–1.258**, also with clean controls. The Float64 statement is too strong: final process **epyc7513b-p3** has prox_out ratios **.8666 at n32768 and .8477 at n65536**, despite output offset +64; its controls are 1.0017 and .9975. The other five processes regress at these sizes. The earlier matrix likewise contains both wins and losses, not the same regression in every process. Offsets modulo 4096 alone do not determine the outcome.

The matrix provenance shows **b05-19 for all six earlier Zen processes and b04-05 for all six final processes**, two nodes rather than the claimed three. Independent universal ordinary-allocation probes on **a04-02** supply another node with regressions; that is additional probe evidence, not a third matrix node. Different fresh processes and campaigns support repeatability without turning jobs on one node into independent machines.

**The matrix uses an ordinary allocation pattern.** `l1_data` makes x with randn and y with similar(x); no chosen offsets or MADV_HUGEPAGE advice are applied there. It retains the suite's arrays and runs for many minutes, a plausible caller state. The setup backing reader does briefly split VMAs using DONTDUMP/DODUMP, so this is an instrumented pattern, not a claim that instrumentation cannot affect subsequent heap/page behavior. Nothing makes the observed allocation pattern dismissible as an artificial forced layout. The fresh short probe and long-lived matrix sample different memory states; neither supplies application frequencies.

The independent allocation analysis finds **254 ordinary unadvised universal function comparisons above 1.05 with positive measured huge-page coverage**, of which 251 have A/A within ±2% (28 NormL1, 223 ElasticNet). ElasticNet examples span all three probe processes. A clean example is Float64 n1048576, later/trial16/p1: **227574.75→289273.00 ns, ratio 1.271112, control .998514**, coverage 60% before and after. NormL1 Float64 n16384, later/trial9/p1 is **3412.48→4205.70 ns, ratio 1.232447, control .999440**, coverage 100% throughout the measured endpoints. These ordinary allocations also show that N's residual NormL1 exposure is not confined to isolated matrix processes. They do not establish power-of-two lengths as a necessary condition for a regression.

The family distinction is real but your bounds need qualification. NormL1 often stays faster while ElasticNet loses 20–35%; adding ridge arithmetic changes the compiled loop, so equal soft-threshold formulas need not have equal placement sensitivity. Some NormL1 matrix losses are unresolved: final Float32 n262144/b-p1 prox_out is 1.0769 with control 1.1641. Others are clean: final Float64 n32768/a-p3 prox_out is **1.1468, control .9932**; earlier b-p3 Float32 n1048576 is **1.2398, control .9954**. Thus the residual NormL1 risk is real, not all noise. If “with weights .99–1.06” refers to both families it is false: final weighted ElasticNet Float32 n1048576 prox_out is **1.0764–1.1735**, controls .9987–1.0069; earlier weighted NormL1 also reaches **1.1226**, control 1.0003. “Everything else .67–.86” excludes too little: weighted separate-output cells, unchanged array-step routes, tiny calls and allocating cases do not obey that envelope.

The page-shift reading is directionally correct. For the forced-huge sweep, **Float32 n1048576, ElasticNet, offset128**, the universal median ratios at shifts **0/1/5/64/301 pages** are **1.4172/.8327/.8314/1.4117/.8314**; all arrays report 100% coverage, with every control within ±2%. At +64 pages the individual ratios are 1.4077–1.4259, controls .99828–1.00226. Skylake-SP Float32 n16384/offset128 changes from **1.0526** at shift0 to **.8099/.8104/.8076/.8087**. This supports sensitivity to more than just the low 12 address bits and is consistent with an alias condition preserved by 256 KiB. **It does not prove an exact power-of-two alias period or a hardware mechanism** from five shifts. Some other NormL1 and A3m offset cells have very dirty controls; their medians cannot support equally precise conclusions.

Explanation 3 is **plausible, not verified**. The recorded kernel policy is THP `[always]`; allocations are retained; there is time for collapse. An unadvised universal trial actually changes y coverage **60→80%** during measurement, so backing can evolve without explicit advice. But final n1048576 Float32 matrix arrays already report **33/33% at setup in every process**, contradicting “0% for those arrays”; the listed smaller arrays report 0/0. There is no later matrix smaps sample or collapse event record. [Round profiles](round8/time-profiles.txt) show many regressions already stable within their timed case; this neither proves nor excludes earlier collapse. Also, an **advised** Float32 n1000000 allocation trial reports 0/0 before and after yet ElasticNet ratio **1.108985**, control 1.002653. Thus the ordinary unadvised small-page samples are favorable (3950 function comparisons, max .920560), but “small pages never regress” and “THP collapse explains the matrix” are not established.

**C. A3n source, correctness, API and code generation.**

The [NormL1 change](round8/trees/A3n/src/functions/norml1.jl:79) adds a statically specialized soft-function argument defaulting to the existing optimized helper; [ElasticNet](round8/trees/A3n/src/functions/elasticnet.jl:110) explicitly passes shrink. The exceptional branch remains shrink. The pre-existing `ordinary=true` ElasticNet branch already uses shrink. Public signatures/exports and array dispatch routes do not change; no runtime cutoff/CPU initialization is added. I found no correctness/API problem in N.

Commands (all Julia runs set `JULIA_DEPOT_PATH="$PWD/round8/depot:/path/to/.julia"` and `TMPDIR="$PWD/round8/tmp"`):

```bash
julia +release --startup-file=no --threads=1 --check-bounds=yes \
  --project=../harness/experiment_kernelopt/env-v1.13 round8/differential.jl
# Repeat with --check-bounds=no; outputs differential.log and differential-no.log.
julia +release --startup-file=no --threads=1 --check-bounds=yes \
  --project=../harness/experiment_kernelopt/env-v1.13 \
  ../harness/experiment_kernelopt/bench/equiv_trees.jl \
  round8/trees/M4 A3n=round8/trees/A3n A3d=round8/trees/A3d
julia +release --startup-file=no --threads=1 --check-bounds=yes \
  --project=../harness/experiment_kernelopt/env-v1.13 round8/boundary_checks.jl
julia +release --startup-file=no --threads=1 --check-bounds=no \
  --project=../harness/experiment_kernelopt/env-v1.13 round8/codecheck.jl
```

[Independent differential output](round8/differential.log), repeated [bounds-off](round8/differential-no.log): **32256 comparisons per candidate, zero differences; zero detected ambiguities for all six trees**. It preserves round 7's raw-bit/special-value, scalar/array step, view/reshape/non-host, public-entry-point and sparse-output coverage. [Supplied expanded comparison](round8/equiv-yes.log): **332640 results each for A3n and A3d, zero differences including NaN bits**. [Boundary checks](round8/boundary-checks.log): **480 additional comparisons**, both sides of 32 KiB and 2 MiB, weights and nonzero in-place input, zero differences; Real/AbstractFloat/BigFloat allocating and separate-output routes pass.

[LLVM checks](round8/codecheck.log) inspect both types/families/weight modes, public wrappers, scalar prox_block, prepared routes and actual scalar-step/array-step loop kernels. **55/56 normalized dumps agree**: N's NormL1 with A3, ElasticNet with M4. Normalization removes generated symbol IDs/module names and renames local block labels, not arithmetic instructions. The remaining weighted Float32 ElasticNet prepared wrapper differs only in GC root-slot assignment; its loop kernel agrees. There is no added dynamic soft-function call or selection branch. This is strong local support for the projection, not a cross-CPU timing identity proof.

The expected matrix is **A3M4's l1/l1w cells and M4's enet/enetw cells**, including operations beyond those tabulated in A. [A3n-projection.csv](round8/A3n-projection.csv) gives the requested four-operation means: retain the universal NormL1 rows in matrix-gm.md and set both ElasticNet families to 1. NormL1's small losses and placement risk remain; all ElasticNet gains also disappear, including its substantial in-place gains. “ElasticNet exactly master everywhere” is a source/kernel projection, not measured N performance. Its own runs must test compiler/code placement, allocations, controls and small-call behavior across these CPUs; the root-slot difference illustrates why copying old numbers is not a completed comparison.

Physical diff counts against M4 are N **22 added/3 removed (net19)**; universal **20/1 (net19)**; A3m/A3d **59/11 (net48)**; A3dr **73/11 (net62)**. N changes three source lines on top of A3, as stated; different notions of “22 lines” should not be mixed.

**D. Unexpected advantage of 32 KiB.**

**Yes.** It rescues sizeable ordinary-allocation Zen matrix regressions that both primary candidates retain, while preserving long identical-output gains. That is relevant under the maintainer's allocation focus, not just success at an artificial pointer arrangement. The final Zen four-operation matrix has no comparable large clean slow cells for A3d/A3dr; fixed 128 KiB Float64/offset128 ElasticNet is **1.3281–1.4091 for U versus .9804–1.0021 for A3d**, with clean controls for both.

But conclusion 4 overstates protection. The finest grid shows **20 KiB onset**, with clean universal ElasticNet controls: Float32 n5120/offset128 **1.2450–1.2796**, Float64 n2560/offset128 **1.2042–1.2787**. A3d still uses A3 below 32 KiB; its Float32 n6144/offset128 is **1.2993–1.3216**, controls .98206–1.00348. The ordinary A3d probe also has a clean NormL1 **1.125853** at Float32 n1048576/later/trial7/p1, controls 1.003412, mixed positive coverage; ElasticNet has a clean **1.055531** example. It does not follow that choosing shrink makes all measured placements identical to M4. The matrix also retains a roughly **1.1–1.4 ns weighted four-entry block-loop regression** in the 32 KiB variants; “only fractional-nanosecond cells remain” is inaccurate across the whole suite.

The advantage warrants considering C. It does not establish that the current 32 KiB rule is a uniformly safe winner, or that an unmeasured 16 KiB rule has already won. A universal cutoff sacrifices Intel gains above the cutoff; Zen-only preserves those while adding initialization/Ref overhead and leaving Intel forced-placement losses.

**E. Merge recommendation and maintainer decision.**

I would select **N for the merge candidate, and merge it after its own queued results validate the projection**, without requiring a new long campaign on this machine. I would not approve U under the maintainer's conditional statement: the realistic Zen evidence does not match the earlier favorable sample. U remains a defensible conscious choice if the maintainer values its broader gains over its observed ordinary-allocation losses, but that is acceptance of regressions, not fulfillment of the condition. I would not select A3m: it sacrifices some Intel gains and tiny-call performance without removing the central sub-2-MiB Zen problem.

I would not merge C as currently written, given its API defect below and incomplete 20–28 KiB protection. If preserving ElasticNet's good-placement/in-place gains and reducing NormL1's placement risk outweigh N's simplicity, a repaired C deserves consideration: universal for broader coverage, Zen-only for retaining Intel gains. That choice requires accepting its short-call/maintenance cost and testing the actual selected cutoff. I do not recommend pointer heuristics or a larger dispatch framework.

The maintainer must decide **how much worst-case robustness versus breadth of gains to buy**, especially whether N's residual NormL1 losses are acceptable and whether losing all ElasticNet gains is preferable to C's machinery. Neither cell counts nor equal-cell geometric means uniquely define “best overall” without workload weights. My preference for N applies the stated simplicity policy to real allocation evidence; it is not a claim that N has the lowest aggregate timing on every workload.

**F. Fixes, remaining defects and digests.**

The listed round-7 repairs are present and mostly effective. [fixtures.jl](round8/fixtures.jl), run with the Julia environment above, produces [fixtures.log](round8/fixtures.log): **block_one calls once and stores the result; the actual prox_in validation catches a defective timed no-op; hugepct gives 100/67/0/50 on controlled mappings, returns -1 for pre-existing DONTDUMP without clearing it, and returns -1 for missing coverage**. The new offset backing read and shifted-region bounds check are present. The equivalence script now includes payload/signaling NaNs, strided views, 4095/4096, and fails on changed NaN bits; [a payload-only fixture](round8/nan-exit.log) reports classification1 and **exit1**.

**Remaining implementation defect:** [A3d norml1.jl:100](round8/trees/A3d/src/functions/norml1.jl:100), also A3m/A3dr, divides by `sizeof(T)` after checking `isbitstype(T)`. A bitstype Real can have size zero. In codecheck.jl:

```julia
struct ZeroStorageReal <: Real end
Base.float(::Type{ZeroStorageReal}) = Float64
Base.float(::ZeroStorageReal) = 1.0
x = [ZeroStorageReal(), ZeroStorageReal()]
prox!(zeros(2), NormL1(.6), x, .8)
```

[Output](round8/codecheck.log): **M4/A3/A3n → [.52,.52]; A3m/A3d/A3dr → DivideError**; ElasticNet reproduces it too. The previous abstract-element error is fixed and its added tests pass, but the guard is still not fully API compatible. Treat zero storage as taking the existing route before division; no extensive dispatch is needed. This does not change Float32/Float64 performance numbers, but it prevents merging the cutoff trees unchanged.

```bash
python3 round8/digest_fixtures.py > round8/digest-fixtures.log
python3 ../harness/experiment_kernelopt/bench/matrix_digest.py \
  ../harness/experiment_kernelopt/results/m6/a3matrix3 > round8/supplied-matrix3.md
python3 ../harness/experiment_kernelopt/bench/a3probe_digest.py \
  ../harness/experiment_kernelopt/results/m6/a3probe3 > round8/supplied-probe3.md
```

The repaired matrix digest excludes incomplete rounds, missing metadata, DIFF and CHECKFAIL from means/classifications in [fixtures](round8/digest-fixtures.log). Its denominator drops cases having no valid implementation, so “1 of 2” in the fixture does not mean only two cases were attempted. The final raw files pass the stronger independent checks. Remaining generic limitations—no provenance consistency validation, no check of `zero_mismatch`, checking the number rather than the exact set of round IDs—do not invalidate these complete files. Both probe formats now run successfully; outputs for m4/m5/m6 are retained in round8. The supplied matrix digest separates epyc7513a/b, explaining its different counts from my six-process pooling.

The allocation digest's “huge” label also overstates what the helper knows for partially backed large arrays. At 4 MiB, 33% of the three intersecting 2 MiB windows means one such page is huge; aggregate smaps counts do not identify whether it is the interior page or an end page. I use raw percentages/positive coverage rather than asserting fully huge backing. This and the missing post-timing matrix read limit the causal explanation, not the observed ordinary-allocation regressions. No remaining postprocessing defect found here reverses the merge tradeoff. Final [hash verification](round8/verify.log): **1028 pinned source/evidence files unchanged**, including 590 external input files.

**What I did not check.** I did not contact a cluster, obtain N's pending results, run long local benchmarks, run full Pkg.test, repeat Julia 1.10 checks, inspect GPU implementations or all third-party extensions, or test new package images/CPU migration. The numerical checks compare against M4 rather than proving all its exceptional conventions. Local LLVM is for Julia 1.13.4 on this Intel host; it cannot certify native-code identity or timings on the cluster CPUs. The full round-7 CPU-selection image test was not repeated for A3dr's renamed byte cutoff. I inspected and hashed the supplied old probes and reran their digests, but independently recomputed the new probes and the two requested matrices. No timing-time matrix backing or application workload distribution can be reconstructed. All source copies, scripts, fixtures and outputs created for this review are in round8; the source trees and harness outside review-code were left unchanged.
