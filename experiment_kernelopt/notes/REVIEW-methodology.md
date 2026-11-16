# Benchmark methodology review

The study is useful for screening strategies, but it cannot currently support a production decision at ±5%. Correctness checks can accept NaNs, failed checks do not fail jobs, the order schedules are not balanced, and pooled round quantiles do not measure uncertainty across code placements. The new function-capturing closures substantially improve on the package harness; I found no hoisting of the actual sampled operations on Julia 1.13.4. That does not rescue the decision procedure. Fix the blockers and retain process-level evidence before accepting either an improvement or a regression.

Evidence below uses repository-relative `file:line` locations. Short local probes used Julia 1.13.4, one Julia thread, one BLAS thread, and `--check-bounds=no`. Their times diagnose harness overhead, not cluster performance. The check-only L1/L2/group run completed 230 Float64 cases / 1,321 implementations with zero reported differences. Julia 1.10 and candidate-tree behavior were not experimentally verified. No source files were changed; temporary probes were removed.

## Findings, ranked by severity

### BLOCKER — Nonfinite errors pass correctness checks, and failed checks still produce successful jobs

**Location:** `experiment_kernelopt/bench/harness.jl:53–67,129–136,158`; `bench/run.jl:79–82`; `slurm/screen.sbatch:35–38`; `slurm/formal.sbatch:47–50`; `bench/summarize.jl:13–17,32–44` (paths after the first are within `experiment_kernelopt/`).

`reldiff(NaN, 1.0)` and `reldiff([NaN], [1.0])` both return NaN. `NaN > rtol` is false, so these wrong results are marked `ok`. A finite scalar compared with an infinite reference also produces `Inf/Inf = NaN`. One NaN can poison an array's maximum error and conceal other errors. Tuple comparison silently truncates unequal-length tuples through `zip`: `reldiff((1.0,), (1.0, 2.0)) == 0`.

Even recognized differences only increment `nfail`. The driver prints that count and exits successfully; both Slurm scripts then create `DONE`. The summarizer ignores correctness status entirely. A deliberately wrong probe returned `nfail=1`; an added NaN probe was not counted. A fast, incorrect implementation can therefore reach the performance report through two independent failures.

**Smallest fix:** compare shape/tuple arity and nonfinite classes explicitly; return `Inf` on any unmatched nonfinite result. Require `d <= rtol`, with deliberate handling of absent references, rather than testing only `d > rtol`. Fail the driver and exclude invalid cases from summaries. Require exact A/A output agreement separately from candidate rounding tolerance.

### BLOCKER — Rotation followed by reversal does not balance timing or load order

**Location:** `experiment_kernelopt/bench/harness.jl:142–145`; `slurm/formal.sbatch:30–38`; `bench/run.jl:45–53,64–69`; `harness.jl:118–127`.

For two implementations, the harness order is `[1,2]` in **every** round: the even-round rotation and reversal cancel. For four, reference positions are `1,3,3,1`, repeating; it never occupies an even position. More generally, even-sized implementation sets preserve position parity. Thus predecessor effects, frequency ramping and position costs can remain correlated with implementation identity. Baseline-only solver cases have exactly two implementations, the reference and its control.

The formal process schedule has the same defect. With `KO_TREE_LIST="base candidate"`, all four default processes load `base candidate`. For three trees it generates `123,132,312,321`: tree 2 is never first. These schedules were reproduced directly from the indexing expressions.

Furthermore, source inclusion order is not native-code compilation order. Constructors and lazy kernel specializations are subsequently first exercised in the fixed `pbs` order, reference then control then candidates. Changing inclusion order can perturb placement, but does not establish balanced placement experiments.

**Why ±5% is unsafe:** the advertised counterbalancing is absent in important configurations; repeated processes can reproduce the same artifact.

**Smallest fix:** run complete rotation blocks and their reversed blocks separately, or use an explicitly balanced schedule; record and verify position and predecessor counts. Generate distinct process permutations, swap reference/candidate roles, and vary first-compilation order as well as inclusion order. Keep the ratio denominator explicit when roles change.

### BLOCKER — An A/A column and pooled round quantiles are not a ±5% decision rule

**Location:** `experiment_kernelopt/bench/run.jl:39–53`; `bench/summarize.jl:9–22,33–44`; `slurm/formal.sbatch:30`; `benchmark/README.md:27–30,47–50`.

Pairing correctly uses `(directory, round)`, but pooling treats all rounds as exchangeable observations. Twenty-one rounds in one process share its code layout and machine state; they are not 21 independent placement experiments. Processes with more rounds receive more weight. The 10th/90th percentiles describe observed round dispersion, **not** a confidence interval for a reproducible effect. A stable 1.12 A/A ratio can have extremely narrow quantiles.

The summarizer also retains only the first metadata row for each `(case, impl)` and never validates provenance. It will pool different Julia versions, CPUs, candidate commits, reference choices or harness settings under the same names. Changing a reference between directories is especially dangerous because the denominator is taken from the first directory's metadata. Missing pairs are silently dropped.

**Smallest fix / required decision rule:**

- Stratify by Julia version, CPU model, source/dependency/harness identity and experiment settings; report every process and load/compile order before aggregating. Require complete valid pairs.
- Use independent processes as the primary replication unit, preferably across fresh allocations. Estimate log ratios per process; use a hierarchical/block bootstrap or an appropriate process-level model, preserving within-process dependence. Four processes are a weak basis when placement variation reaches tens of percent; determine replication from the observed between-process spread.
- For a claimed slowdown above 5%, require a one-sided confidence bound above `1.05` and reproduction under swapped roles and distinct placements. For a greater-than-5% improvement require the upper bound below `0.95`. To clear the no-regression gate, require the upper bound below `1.05` for every affected ordinary path; full ±5% equivalence requires the entire interval inside `[0.95,1.05]`. Use simultaneous bounds or independent confirmation for cells selected from many comparisons.
- Predeclare an A/A acceptance band substantially narrower than the decision margin, e.g. ±2% with adequate precision. If the control deviates materially, label that cell **unresolved** and rerun with additional same-source controls and balanced placements. Do not call it a win, waive a candidate regression, subtract the A/A deviation, or divide by the control: different functions need not share its placement penalty. A control near one is necessary evidence, not proof that the candidate's placement is harmless.

### MAJOR — The existing package harness adds dynamic lookup, dispatch and allocation

**Location:** `benchmark/l2_norms.jl:59,71–82,92–109,114–121,184–191`; `benchmark/README.md:74–89`.

`PB` is a captured value of type `Module`; its identity is not represented by that type. Calls such as `PB.prox_block!` perform runtime module-field lookup and dispatch. Interpolating `$op` in BenchmarkTools specializes the outer closure, but does not turn that captured module into a constant module binding.

The user's mechanism and allocation claim are confirmed. In two short local runs, the simple module-capturing form cost about **31–60 ns and 64 bytes/call**, versus **5.3–10.6 ns and zero bytes** through a captured function value. The actual `makeprobes` `prox_block1` closure likewise allocated 64 bytes. This does not reproduce the exact 23 ns figure on this workstation; it verifies the defect. Common overhead compresses `(kernel_Y + overhead)/(kernel_X + overhead)` toward one and adds allocation/GC noise.

Materially affected: `prox_block1`, `prox_block_zero`, every call inside `block_sweep`, and the `blockindices`/`prox_block!`/`blockview` calls inside `block_update_sweep`. Group array `prox`/`value_and_prox` and dense NormL2 `prox`/`value_and_prox` also incur dynamic-call overhead; its fraction must be checked, especially for the shorter NormL2 inputs. For long arrays it may be negligible. Group/dense `value` calls through captured `f`, and sparse/symmetric value calls through captured `g`, do not have this module-lookup defect; their constructors run outside timing.

**Smallest fix:** bind each function value before constructing the closure, using a typed function barrier as the research harness already does. Recheck allocation counts and rerun affected tiny-kernel ratios.

### MAJOR — An older-harness probe changes its captured input after another probe runs

**Location:** `benchmark/l2_norms.jl:70–71,92–102,169–179,200–212`.

The `u,out` bindings captured by `prox_block1` are assigned again **inside** `block_update_sweep` at line 96. They are shared `Core.Box` fields. After that sweep runs, `prox_block1` uses the last update trial/output, not its selected nonzero `U[k1]` block. This also loses the intended shared-array placement across trees because update scratch is private to each tree.

A two-group probe produced `[1.7337439058, 2.6006158587]` before running the update sweep and `[-0.1605887450, -0.1605887450]` afterward, with unchanged `prox_block1` arguments at construction. Initial correctness comparison precedes this mutation; warmup and timing occur afterward. The benchmark can time a different branch/input from the one checked.

**Smallest fix:** use distinct local names inside the update closure, and bind the single-block probe's arguments through a separate factory. Verify its input identity and output after exercising all probes.

### MAJOR — Strategy comparisons are not uniformly equal-work comparisons

**Location:** `experiment_kernelopt/bench/suites.jl`, locations in the table; checks illustrated by `src/functions/norml2.jl:69–85`, `src/functions/groupnorml2.jl:144–166`, and `src/functions/leastsquares_predictor.jl:171–183`.

`level` distinguishes inner kernels from public APIs, but does not prevent their ratios from being interpreted as production speedups. It also does not describe differences between two level-2 implementations. Inventory:

| Family / location in `suites.jl` | Work difference and disclosure |
|---|---|
| L1 `prox`, `prox_w` (`37–65`) | Kernels receive precomputed `γλ`, omit public validation and exceptional recovery. PO weighted coefficients `λ .* w` are precomputed outside timing, whereas PB retains factors; multiplication association/rounding can differ. PO `prox!` also returns the value: the note discloses this, but not coefficient preprocessing. |
| L1 `prox_g`, `prox_gw` (`69–83`) | Same per-entry mathematical thresholds for PB/kernel; raw loops omit API checks/recovery, disclosed only broadly by level. There is no PO competitor here. |
| L1 values / value+prox (`88–127`) | Same intended value, including value **at the proximal output**. `thunkscaled`'s `λ * result` is outside the kernel function but **inside timing**, so no missing multiply. Weighted PO coefficients are again formed outside timing. Fused versus two-pass kernels intentionally differ in passes, not mathematical objective. |
| L2 (`158–189`) | PO prox returns a value; note discloses this. `k_direct` uses an unchecked unscaled sum of squares and a precomputed threshold. Norm/nrm2 variants use a more robust norm but still omit PB parameter/recovery rules. Value+prox's `λ * K.l2prox!` is timed and correctly uses the returned output norm. Zero-vector norm/broadcast variants are omitted, although at the actual positive threshold they return zeros correctly; this omits a valid performance branch. |
| Group (`235–280`) | Range/column representations use the same entries because all weights equal `sqrt(m)`; PO folds this into `poλ` outside timing. Matrix columns avoid index-group traversal/metadata and cannot represent permuted groups this way. The prox note discloses columns and returned value; value/value+prox entries have no such note. Raw block sweeps omit the public block checks. `k_cols_norm` also uses the column representation without a note. |
| LS residual/predictor/value/gradient (`321–373`) | Same mathematical values/gradients, but raw kernels omit checks and recovery. Predictor cases deliberately avoid `Xβ`; they must only be compared within that family. PO `gradient!` returns the value, so its wrapper does **not** omit the value computation. PB/PO/private scratch and standalone shared scratch need not have identical memory placement. |
| LS proximal (`375–418`) | Fixed-step comparisons exclude SVD, Gram construction and initial Cholesky, as the notes largely disclose. PO additionally evaluates the loss at the output, including another `Xβ` product: much more than a scalar return. The varying-step PO note omits that cost; the first-prox PO entry has no note. `k_chol` reuses factor storage, while PO allocates a new shifted matrix/factorization when the step changes. These are algorithm/API comparisons, not isolated factorization comparisons. |

An unused PO return may lose some arithmetic through compiler elimination; do not assume that the entire nominal extra value cost survives. Conversely, the installed PO 0.16.1 LS implementation writes its residual through another matrix product, which is real observable scratch work. Its `leastSquaresDirect.jl:67–80` confirms that cost and step-keyed refactorization.

**Why ±5% matters:** a cheaper unchecked contract, preprocessing, representation or omitted output can exceed the entire margin without improving the production path.

**Smallest fix:** add explicit contract/setup/representation notes and separate exploratory strategy ratios from production candidate-versus-base gates. Match outputs by timing PB prox plus value where comparing with PO's combined API; also retain the native API comparison. Test proposed production changes through the complete checked API.

### MAJOR — Output tolerances can admit substantial errors and wrong active sets

**Location:** `experiment_kernelopt/bench/harness.jl:54–65`; `bench/suites.jl:32,86–88,156,234,253,302–304,320,346,381–382`; `bench/solvers.jl:103`.

The array metric is a legitimate global infinity-norm relative error, but cannot protect every component or group. `reldiff(Float32[1e10,0], Float32[1e10,1]) == 1e-10`: deleting the second component entirely passes. Flattening all compact blocks deliberately extends this masking across groups. Large LS gradients and near-threshold shrinkage are vulnerable to the same issue.

The length-scaled tolerances are especially loose for Float32: `4*n*eps(T)` is **0.5 at n=1,048,576**, and about **0.477 at n=1,000,000**. Those large L1/LS value cases lack a BigFloat reference. L2 n=65,536 permits about 0.78% global error; LS prox permits 2%. These are not narrow rounding checks, even though pessimistic reduction error bounds may motivate some of them. One `rtol` also applies to both scalar and vector outputs in value+prox tuples.

**Smallest fix:** separate scalar-reduction and component tolerances; check exact-zero regions, active sets and nonfinite classes independently. Use mixed absolute/relative component or group bounds derived from input scale and conditioning near cancellation. Add adversarial cases and residual/optimality checks; do not merely replace global scaling with unqualified componentwise relative error at zero. Use tighter measured/analytical reduction bounds for these actual algorithms and independent checks for large cases.

### MAJOR — Numerical stand-ins include an incorrect scaled norm and incomplete exceptional behavior

**Location:** `experiment_kernelopt/kernels/kernels.jl:12–28,126–146,188–214,253–335`; `bench/numerics.jl:48–57,84–107`.

All eight `st_*` forms compute ordinary finite soft thresholding for nonnegative finite thresholds, modulo signed zero. Their exceptional differences are:

| Form | Assessment |
|---|---|
| `st_maxmin` | Faithful to PB's ordinary max/min shrink, including NaN propagation. Does not implement PB's higher-level infinite-parameter rules. |
| `st_selmaxmin` | Same ordinary mathematical strategy and NaN propagation; select tie/signed-zero semantics are not generally identical to `max`/`min`. It is not a promise about emitted x86 instructions. |
| `st_clamp` | Valid finite strategy; `st_clamp(1.0, NaN) == 0`, losing threshold NaN. |
| `st_clampsel` | Also loses threshold NaN, contrary to its line-20 comment: its comparisons are `>`/`<`, not the claimed negated comparison. |
| `st_copysign` | Valid magnitude strategy; returns `-0` for negative entries shrunk to zero. NaNs propagate. |
| `st_copysignsel` | Same qualification as copysign; selective maximum does not repair the signed-zero contract. |
| `st_ifelse` | Returns zero for NaN input, NaN threshold, and `(Inf,Inf)`; unsuitable as an unguarded PB replacement. |
| `st_po` | Faithful to the installed PO real scalar correction formula; loses a NaN threshold for finite input. It is a scalar-form stand-in, not PO's complete prox/value loop. |

`norm_scaled` is **not** a safe replacement for LinearAlgebra scaling: `inv(m)` overflows for sufficiently small subnormals. With `x=[nextfloat(0.0),0.0]` it returns NaN (`0*Inf`); with only nonzero tiny entries it can return Inf although the norm is tiny. Confirmed for Float32 and Float64. Use `x[i]/m` or safe power-of-two scaling. It also always scales and accumulates in `T`, unlike all details of the generic library implementation; its ordinary timings cannot be labeled the cost of Julia's fallback.

`norm_simd`, `norm_plain`, `norm_base`, and `norm_dot` intentionally square before taking the root, so can overflow/underflow while the norm is representable. `norm_la` and `norm_blas` call the actual libraries and are fair references. `l2prox!`, indexed group variants and `block_prox!` inherit the raw-square failures: an overflowing norm can leave a vector unshrunk; an underflowed norm can erase a nonzero vector; a NaN norm takes the zero branch. They are fair ordinary-path sketches, not substitutes for PB's checked, recovering strategy.

`l2prox_norm!`, `l2prox_nrm2!` and column shrinkage correctly handle zero input with a **positive** threshold (`c/0=Inf`, scale zero). The suite's `0/0` rationale for excluding zero cases is inaccurate for its inputs. Zero threshold plus zero input does give NaNs; these forms also lack the library's full exceptional parameter policy. The norm/broadcast form approximates PO's strategy, but its broadcast loop is not the exact PO scalar loop or combined value API.

`group_prox_early!`'s guarded comparison is conservative relative to the **computed** ordinary sum/sqrt; square underflow can lose the shortcut, and overflow falls back to the same unsafe direct strategy. Four eps is not a group-size-independent bound on the error of the accumulated sum relative to the exact sum. Its comment overstates that guarantee. I found no additional ordinary finite-domain mathematical error in this shortcut compared with `group_prox!`.

**Smallest fix:** repair the scaled norm; accurately label finite-domain strategy sketches, and require production exceptional/underflow/overflow tests before adoption. Add both omitted soft-threshold variants and NaN thresholds to the numerical grid. A faster, weaker numerical contract is not a production improvement.

### MAJOR — Cholesky failures are ignored and the numerical design misses important failure regimes

**Location:** `experiment_kernelopt/kernels/kernels.jl:426–461`; `bench/numerics.jl:141–161`; `bench/suites.jl:398–408`.

`LAPACK.potrf!` returns an `info` indicating a non-positive pivot; `chol_factor!` ignores it and marks the factor cached. `potrs!` then uses the incomplete factor. Locally, `X=fill(1e8,2,2)`, `Y=v=ones(2)`, `γ=1` produced a factor with diagonal `-4` and output approximately `[1.1863e-8,-1.8626e-9]`; the symmetric exact solution is approximately `[5.0000e-9,5.0000e-9]`. No exception was raised. The shift was lost in the rounded Gram matrix.

The numerical LS tests use only a tall design with largest singular value about one and `γ <= 1e6`. Regularization therefore caps the normal-equation condition number at about `1e6`, even when `cond(X)=1e14`; that heading alone does not establish stability at that condition number. No fat, large-scale Gram/cancellation example is included. The varying-step suite has no independent reference for either output.

**Smallest fix:** check `potrf!` status and fail explicitly or use an authorized stable fallback. Test tall and fat designs across absolute scales, rank deficiency and step sizes, and compare both varying-step outputs to independent solutions/residuals. Cholesky's numerical and setup tradeoffs must accompany its speed ratios.

### MAJOR — The cases cannot certify every affected ordinary hot path

**Location:** `experiment_kernelopt/bench/suites.jl:23–30,143,199–225,270–280`; `bench/solvers.jl:100–177`; `benchmark/l2_norms.jl:123–137`.

The research microcases primarily use dense vectors, disjoint equal-size groups and uniform group weights. Block kernels are only measured in sweeps; scalar L1 coordinate prox appears inside a much larger solver loop. Sparse/BlockIterate and symmetric-storage value paths, heterogeneous group weights/sizes, aliasing, views/strides, and zero-level/zero-step ordinary branches are not covered comprehensively. Sparse solver designs exercise sparse **X**, not sparse iterate storage. The older harness covers some omitted storage paths, but has the defects above.

Solver gradients can dominate and conceal a >5% slowdown in the prox component. Solver speedups are useful corroboration, not a waiver for a kernel regression.

**Smallest fix:** enumerate production methods touched by each proposed patch and add direct cases for their existing ordinary branches/storage types. Repair and retain the package harness where it provides that coverage.

### MINOR — Calibration and minimum sampling need sensitivity checks, not an assumed common bias cancellation

**Location:** `experiment_kernelopt/bench/harness.jl:32–49,123–127,141–151`.

Calibration separately chooses eval counts to target a similar batch duration. Different counts are not inherently biased: dividing elapsed batch time by count is valid, and clock overhead should be negligible for the intended 2 ms batches. They do change the mixture of loop overhead, cache/branch training, allocating-call GC and frequency behavior. A compilation/GC interruption during calibration can undershoot the target; there is no final duration validation.

Taking the minimum of three batches estimates a lower envelope, not typical call cost. Implementations with different variability or GC behavior can receive different downward selection effects. Most individual batch observations, and all GC diagnostics, are discarded. This matters particularly for allocating first-prox calls; it is not evidence that different eval counts alone create a 5% error.

**Smallest fix:** retain individual samples and realized batch durations, distinguish GC-free kernel latency from amortized allocating-call cost, and confirm borderline ratios at more than one batch duration and with a robust non-minimum estimator. For similarly sized competitors, test common eval counts as a sensitivity check.

### MINOR — Pinning is helpful but the scripts do not establish a controlled machine state

**Location:** `experiment_kernelopt/slurm/screen.sbatch:4,7–21,29–34`; `slurm/formal.sbatch:5,8–22,42–46`; `bench/run.jl:11–12`; `bench/harness.jl:70–94`.

Exclusivity is a submission example, not an enforced directive. The selected item in an affinity list is a logical CPU, not necessarily a distinct physical core with an idle SMT sibling. Frequency governor/turbo/thermal state, sibling activity, effective affinity, automatic NUMA migration, transparent huge pages and page-fault/page-size behavior are not recorded. Both scripts also allow user startup files, which can alter configuration.

Starting the entire process under `taskset`, before allocating data, and using single-threaded BLAS substantially addresses first-touch locality for shared arrays; it is not a demonstrated NUMA bug. Private library scratch/factors still have distinct allocations, and memory policy/THP can vary between processes. Exclusivity and order balancing reduce noise but cannot remove code-placement effects.

**Smallest fix:** require or verify the intended allocation, physical-core/sibling placement and effective affinity; add `--startup-file=no`; record frequency policy and observed frequency, NUMA policy/locality, THP settings and faults. Use a reproducible warmup before measurements and investigate these factors for unresolved cells. Memory binding is useful when topology warrants it, not an automatic requirement on every node.

### MINOR — The numerical tables overstate rounding checks and omit two variants

**Location:** `experiment_kernelopt/bench/numerics.jl:18–28,49,56,95`; `kernels/kernels.jl:149–185`; `bench/suites.jl:344–369`.

`ulps` computes relative error in units of `eps(T)`, not ULP distance. Its `u <= 0.5` test does **not** mean “correctly rounded,” despite the table caption. For `exact = big(1.5) + big(eps(Float64))*big(0.49)`, it labels `nextfloat(1.5)` as `ok`, although conversion of the reference rounds to `1.5`. The denominator floor further changes the meaning for subnormals. Compare with `T(exact)` explicitly for that label; retain the relative-error diagnostic under an accurate name.

The soft grid omits `st_clampsel` and `st_copysignsel`, and has no NaN threshold, so it cannot validate the former's incorrect NaN comment.

`norm_blue` is a reasonable three-accumulator strategy sketch, with ordinary sequential-summation error rather than guaranteed correct rounding. Its Float64 `ssml` is `2^538`, whereas the reference formula gives `2^537`: `exponent(floatmin(T))` differs by one from Fortran's `MINEXPONENT`. I found no demonstrated wrong norm caused by this more conservative factor, but it is not the exact named algorithm or the installed vendor's optimized BLAS implementation. Align the constants or disclose the variation. The reference formula is in [Netlib DNRM2, lines 103–110](https://netlib.org/lapack/explore-html/d6/de0/dnrm2_8f90_source.html).

For predictor value+gradient, the expected result uses the high-precision product `big(X)*big(β)`, while the supplied predictor is the already rounded `z = X*β`. The correct independent reference for that API is `big(z)-big(Y)`, then `big(X)'*residual`. Current loose tolerances hide this small reference mismatch.

### MINOR — Some named input regimes do not exercise the advertised branch

**Location:** `experiment_kernelopt/bench/suites.jl:205–213,227`; `bench/numerics.jl:130–132`.

Group `:allshrink` only suppresses deliberate downscaling; normal random groups can still lie below the threshold. On the actual Float64 range-layout input at size two, **459 of 2,048 groups** are set to zero; Float32 gives 441. These are valid mixed-work cases, but cannot isolate the all-active branch or explain its speed. Rename the pattern or construct and verify norms above threshold, including near-boundary cases.

The LS numerical case labeled “squares overflow, value finite?” uses two entries `sqrt(floatmax(T))/2`. Their squares and their sum are finite; it misses its stated regime. Add a residual such as `[1.1*sqrt(floatmax(T))]`, whose square overflows but exact half-square is finite, and a separate finite-square/overflowing-sum example. This matters when judging raw half-sum kernels and `abs2(nrm2(r))/2`, which can both overflow before the division even though the desired value is representable.

### NOTE — The new call barrier is effective for the sampled operations, but is not an optimizer black box

**Location:** `experiment_kernelopt/bench/harness.jl:12–16,32–39`; `bench/suites.jl:7–10,286–300`.

The factories capture concrete function values and argument tuples. `Impl.op::Any` causes dispatch into a specialized `runbatch` outside its timed interval; it does not insert a dynamic module lookup in each evaluation. `invoke_op` normally inlines the closure/kernel into a specialized noinline wrapper, whose call remains inside the batch loop. Ref writes and output-array writes are observable escaping mutations. LLVM inspection of norm-through-Ref and output-writing prox probes showed calls inside the loop, and the discarded-array-norm probe retained its reduction/domain check. No actual-operation hoisting was demonstrated.

Nevertheless, `@noinline` only prevents inlining. An empty operation's loop disappeared in a local probe, and unused pure arithmetic may disappear inside a wrapper. It is not a general optimization fence. `thunk` must only discard returns when the operation has the intended observable output; pure scalar cases need their Ref sink.

Per-call cost is not guaranteed identical: the wrapper's ABI/root arguments, captured state, surviving checks, return computation and Ref store vary. Local new-harness two-entry probes took roughly 1–3 ns for norm/Ref forms and 3–7 ns for a raw prox, depending on the run. That is material at this scale. `thunkscaled` adds a timed multiplication. Empty-loop timing cannot simply be subtracted after the optimizer removes it.

**Smallest fix:** preserve these factories; inspect representative generated code on both Julia versions and benchmark realistic typed batches/sweeps alongside single-call probes. Require a tiny-kernel ratio to survive a wrapper/batch sensitivity check before interpreting a 5% difference.

### NOTE — Solver loops are reproducible fixed-work experiments, not convergence or objective benchmarks

**Location:** `experiment_kernelopt/bench/solvers.jl:7–17,20–38,41–54,74–97,114–177`.

The loops reset iterates/residuals on entry and run the same fixed iteration counts across trees. That is suitable for kernel impact. PGD's returned scalar adds the smooth value at the old iterate to the penalty at the new iterate; APG overwrites the smooth value and returns only the penalty. Neither is the final objective. Solver cases have no independent reference and could agree on a shared algorithm error.

**Smallest fix:** label the returned diagnostic accurately and validate final iterates plus an independently recomputed objective/optimality diagnostic outside timing. Do not infer convergence speed from these fixed-count timings.

## Checks found sound

- `suites.jl:17,41–44,107–127`: BigFloat soft thresholding and weighted output-value formulas are independent of tested kernels and mathematically correct for the ordinary supplied parameters.
- `suites.jl:137–140,216–223`: BigFloat Euclidean norm, shrink factor and weighted group objective are correct; the generated groups partition the arrays, and range-to-column reshaping preserves them.
- `suites.jl:347–348,382`; `numerics.jl:153–154`: LS value/gradient and proximal normal equations have the correct sign and `1/γ` scaling. BigFloat arithmetic uses no tested factorization; 1,024-bit precision in the numerical tables is ample for the tested regularized systems. The suite normally uses Julia's default BigFloat precision; record it explicitly.
- `suites.jl:425–429`: `two_steps_thunk` applies both steps to the original `v`, with distinct outputs; first-prox factories correctly use PB's `(Y,X)` constructor and kernel/PO's `(X,Y)` constructor.
- `suites.jl:383–417`; `kernels.jl:432–445`: fixed-step factors are warmed outside timing; alternating steps leave Cholesky at `γ2`, ensuring both step changes occur on every subsequent batch evaluation. Each implementation owns its cache; the shared destination does not share factors.
- `harness.jl:118–120`: deep-copying outputs immediately after each implementation prevents later shared-buffer writes from changing saved comparison results.
- `suites.jl:35,148,229,343–345,378–379`; `kernels.jl:371–410`: the new microcases have distinct inputs and destinations; residual/gradient scratch is overwritten before use. I found no unintended destructive input/output aliasing. Matrix reshapes intentionally share the corresponding vector's storage.
- `solvers.jl:8,21,42,54,75`: solver entry resets and per-step scratch writes prevent recurrence across timed invocations; cache warmth is intentional steady-state behavior. A repeated-call/input-immutability check would strengthen this against future candidate changes.
- `suites.jl:23–27,145–151,199–213`: ordinary timing inputs do not intentionally contain subnormals. Repetition trains branches and warms caches rather than repeatedly shrinking the input toward subnormal values; evolving-input behavior is partially represented by solver cases.
- `run.jl:20–24,45–53`: separate module namespaces produce distinct tree types/functions and a same-source reference recompile. They share runtime/dependencies, not independent processes; that is appropriate for paired timing, with the placement limitations above.
- `harness.jl:70–94`: Julia/LLVM/CPU/BLAS/thread/bounds and tree identity provenance is useful. Preserve exact manifests and source/harness hashes too: a `+dirty` marker or unverified `TREE_SHA` is not a content identity.
