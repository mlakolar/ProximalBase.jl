# Fourth independent review: placement and benchmark conclusions

Edition: master `18b2714955c224300f8924aeb3a205b84c62be57`, supplied `review4/`, final A3 without the ElasticNet-site edit. No tracked files edited. Ratios below are candidate/reference unless stated otherwise.

| Conclusion | Verdict | Scope |
|---|---|---|
| (a) Placement causes slowdowns and can reverse A3's gain | **SUPPORTED WITH CORRECTIONS** | Strong controlled evidence for placement sensitivity; the page-size exclusion, universal band, and Intel exception are false. The precise hardware mechanism and attribution of earlier processes remain unproved. |
| (b) Consecutive equal-size power-of-two arrays are +64 modulo 4096 | **NOT SUPPORTED** as an allocator rule | A common allocation-history-dependent outcome, including in the supplied cluster data; neither guaranteed nor confined to the stated size intervals. |
| (c) Repetition benefits master's Julia 1.10 branches | **SUPPORTED WITH CORRECTIONS** | Code generation and timing support this explanation strongly. “Every changing input,” a universal K cutoff, and exact copy-subtracted prox times go beyond the evidence. |
| (d) Do not merge A3 under the reproducible >5% rule | **SUPPORTED WITH CORRECTIONS** | Retain this recommendation. D3/V remain conditional proposals with disclosed tiny regressions; B925 also needs placement coverage before proposing its revision. |

## MAJOR

**1. “Page size does not matter” is contradicted by matched-offset rows.** `REPORT.md:891–893` only describes the equal-offset cases correctly. In `results/probe1/epyc7513-a03-04-julia1.13.4-A3-p1.csv`, Float64 ElasticNet, n=2^20, +64:

- Row 103, huge: master 225,323.1 ns, A3 367,322.7 ns, ratio **1.630**.
- Row 123, small: master 222,467.8 ns, A3 184,739.5 ns, ratio **0.830**.

The rows record huge-page coverage 8192/10240 KiB versus 0/0. On Julia 1.10, a03-04 process 2, Float32 ElasticNet, corresponding rows 7/27 give **1.376 versus 0.686**. These discrepancies are much larger than dynamic-call overhead. Probe5 likewise sweeps both advice settings at +64/+128.

**Correction:** say that backing has little effect in the tested equal-offset cases. At positive offsets, backing **or associated physical/higher-address placement** changes the penalty substantially. These tests allocate separate regions for the two advice settings; they do not isolate page size itself from those other changes. `madvise` requests a policy, not successful huge-page allocation; offset_probe does not record actual backing.

**2. Regenerate the tables: 17 cells do not reproduce.** I independently parsed the CSVs, separately checked the supplied digests, and excluded `probe-invalid`. In the +64 table (`REPORT.md:925–941`), these are **all 15 discrepant cells**, shown as printed → recomputed. All other cells match at the printed precision.

| Report line; Zen 3 case | NormL1 +64 | NormL1 0 | ElasticNet +64 | ElasticNet 0 |
|---|---|---|---|---|
| 931; 1.13, both, 1024 | 0.67 → **0.67–0.68** | | | |
| 932; 1.10, Float32, 2^20 | 1.33–1.38 → **1.35–1.38** | 0.74 → **0.73–0.74** | 1.16–1.38 → **1.27–1.38** | |
| 933; 1.10, Float64, 2^20 | | | 0.89–0.92 → **0.87–0.92** | |
| 934; 1.10, Float32, 16384 | 0.58–1.37 → **0.60–1.38** | | 0.57–1.17 → **0.58–1.28** | |
| 935; 1.10, Float64, 16384 | 0.56–0.71 → **0.59–0.84** | 0.55–0.61 → **0.60–0.61** | 0.54–0.68 → **0.54–0.87** | 0.55–0.57 → **0.54–0.57** |
| 936; 1.10, both, 1024 | 0.54–0.58 → **0.51–0.53** | 0.59–0.60 → **0.58–0.59** | 0.52–0.55 → **0.48–0.53** | 0.54–0.55 → **0.53–0.54** |

The “largest slowdown against itself” table, interpreted as maxima **within 16…256**, has **two** errors, both Zen 3 / 1024 entries:

- Ridge **1.99 → 2.05**: probe5 `epyc7513-a04-18-julia1.13.4-A3-p2.csv:1631`, Float64, +80, 166.53/81.15 ns (baseline row 1551).
- Scale **2.24 → 2.27**: same file, row 1633, 163.07/71.97 ns (baseline row 1553).

The other 43 numbers match. If “largest” means the entire sweep, also change Zen 3 / 16384 copy **1.03 → 1.11**, and Skylake-SP / 1024 copy **1.21 → 1.32**; both maxima occur at +4000, outside the band. Label the table explicitly. Every extremum and its CSV row is in `.codex-probes/round4/{table-mismatches,largest-slowdowns,largest-in-band}.txt`. The supplied `tables/report-offset64.md` is stale too; rerunning `offset64.py --report` gives the corrected table.

**3. The band is a useful description, not a universal boundary; Intel is not uniformly exempt.** `REPORT.md:30–34,899–911,920,959–960` overstate the results.

- **Zen 3:** the main million-entry penalty is concentrated around +16…256, but is strongly serrated. At 1024, +64 often has essentially no penalty. At 16384, A3's NormL1 loop remains up to **1.12×** its own offset-zero time at +512.
- **Broadwell:** the main threshold-loop penalty is concentrated around +16…256, but million-entry ridge/scale peak at only **1.017×**. Copy at 16384 is approximately unchanged at several band offsets; “every loop is slower” is untenable.
- **Skylake-SP:** the tail extends past 256. In probe5 / Julia 1.13, NormL1 A3 at 16384 is **1.26–1.28×** its own time at +512; master is **1.20×** at +384. At 1024 A3 still loses **14–17%** at +512. Copy also varies outside the band.
- **Intel A3 losses exist:** Broadwell / Julia 1.10 / Float32 / n=1024 / ElasticNet / +16 gives A3/master **1.207 and 1.274** in probe2 d18-14 processes 1/2 (candidate row 526; controls 0.997/0.998). Probe5 d18-13 process 1 gives **1.258**, control 1.003 (rows 645–647); process 2 instead gives 0.513. Thus this is not a universally reproducible Intel regression, but “equal or faster in every placement” is false.

**Correction:** describe the measured CPU/type/size/backing-dependent profiles. Remove “No formula avoids cost (a)” (`1019–1022`): testing several formulas does not establish impossibility, or that throughput alone determines the penalty rather than instruction scheduling/unrolling.

**4. Placement is demonstrated; the full causal explanation is not.** The same compiled closure is reused across offsets within an offset sweep, so fixed per-process JIT placement cannot by itself explain the within-process profile. However, code placement can interact with it: identical-source A/A copies differ substantially even in fixed probe5. Examples: `epyc7513-a04-18-julia1.13.4-A3-p2.csv:84` gives NormL1 A/A **1.152** at +80; process 1, Float64, row 996 gives **0.861**. Original probe2 Broadwell controls reach **0.649/1.544** at n=1024/+80.

The probes fix compilation order, sweep offsets in ascending order, retain only minimum/median summaries, and do not enforce the main campaign's A/A acceptance rule. Rotating/reversing implementation order helps, but 15 rounds with 16 implementations do not cover every position, and the final round is unpaired. Frequency, cache state, physical placement, and code placement remain contributors the data cannot separate. Low-address memory disambiguation is plausible; no hardware counters establish that mechanism.

**Correction:** retain the empirical placement result and large regressions, qualify exact maxima and mechanism, and randomize/reverse offset order and compilation order while saving round/order/evals data. Earlier formal runs did not record pointers: “formal4 happened to be out of it” and definitive attribution of individual formal6 processes (`951–954`, `1013–1015`) are inferences. “Averaged over placements” (`1016`) also needs a specified placement distribution; this irregular sweep supplies none.

**5. The allocator claim is conditional, and the proposed avoidance advice is false.** I ran the supplied `alloc_offsets.jl` unchanged on both versions. Selected x→y / y→z offsets:

| Condition | Julia 1.10.13 | Julia 1.13.4 |
|---|---|---|
| Fresh Float64, n=65536 (512 KiB) | 0 / 0 | 0 / 0 |
| After-free Float32, n=16384 (64 KiB) | 3840 / 2688 | 2368 / 1792 |
| After-free Float64, n=2^20 (8 MiB) | 64 / 1408 | 64 / 3648 |

Evidence: `.codex-probes/round4/alloc-1.{10,13}.txt`. My extended retained/recycled allocation probe records absolute distances too: adjacent allocations with distance `bytes+64` occur frequently, but after-free runs also produce non-64 offsets at 256/512 KiB and 4 MiB, and offset 0 at 32 MiB. A fixed “freed a large array” condition does not guarantee adjacency. The supplied check measures only residues, so alone cannot establish `bytes+64` absolute spacing or separate mappings.

`suites.jl:23–28,320–325` really does create y with `similar(x)` after generating x; the placement tendency is relevant. It does **not** establish actual addresses in the formal campaign. Replaying the exact `l1_data` function locally gives, for example, Float32 million-entry offsets 1792/2816/3712/3776 after freeing large arrays on Julia 1.13; Float64 in that phase gives 64 in all four repetitions. Allocation history matters.

Also, a non-multiple-of-4096 byte length does **not** avoid the band (`REPORT.md:1066–1067`): Float32 n=1048592 has 4 MiB+64 bytes and repeatedly lands at **+128** in my harness replay. The “6% of placements/other lengths” is a uniform-residue heuristic, not an observed workload probability. Cluster counts of 62–78/96 at +64 reproduce, but each array pair is counted twice, once per operation: 31–39/48 distinct pairs.

**Correction:** document observed frequencies and allocator/history conditions, measure pointers in every campaign, and deliberately sample several placements. Do not prescribe non-power-of-two lengths as a guarantee.

**6. Branch prediction is strongly implicated, but the workload and net-time claims need limits.** Cross-compiling with `-C znver3` confirms the two sign-dependent `jns` instructions in master's ordinary scalar threshold on Julia 1.10; they disappear on 1.13 and in ordinary A3. The full prepared sparse ElasticNet loop confirms the same distinction. A3 still has other control flow and unchanged exceptional paths; it is not globally branch-free.

Local i5-13600K, CPU 2, one thread, five rounds of the current shared-array sparse probe, Float32 ElasticNet, copy-subtracted master times:

| Julia | K=1 | K=2 | K=8 | K=64 | K=512 |
|---|---:|---:|---:|---:|---:|
| 1.10.13, µs | 13.70 | 28.18 | 35.86 | 36.66 | 36.96 |
| 1.13.4, µs | 6.66 | 6.39 | 6.34 | 6.55 | 6.36 |

These are directional checks, not stable 5% decisions; some local A/A controls fail. A separate matched-footprint experiment keeps x/y fixed: at K=512, identical-valued arrays give master **7.69 µs**, different-valued arrays **35.48 µs**; A3 gives **7.18/7.23 µs**. Thus cycling-cache footprint alone cannot explain the local effect. Zen 3's supplied K dependence, its absence on 1.13, and the code generation strongly support predictor learning; the equivalent matched-footprint experiment or branch-miss counters have not been supplied for Zen 3.

`sparse_digest.py:18` subtracts a separately measured median copy-only time from a median copy+prox time. This is **not an exact isolated prox measurement**: copy/prox interact through caches, stores, and execution state; the original own-array version additionally subtracts the reference array's copy cost from all trees. Nevertheless, subtraction cannot explain the large Zen 3 effect: probe3 b09-17 / 1.10 / process 1 rows 2–16 have raw copy+prox master **11.762 → 50.224 µs**, A3 **14.116 → 14.759 µs**, copy only **0.442 → 0.513 µs**.

**Correction:** report raw copy+prox as primary and net times as approximate. Preserve the measured repeated-input sensitivity, but replace “a caller whose data change sees the other columns,” “every scalar loop,” and “eight or more removes the effect” (`985–991,1068–1072`) with claims about these random-value sets. Changing values can preserve predictable branch outcomes; repeated or correlated data are legitimate workloads. Broadwell's 43→57 µs K dependence also contradicts “does not learn … even at K=1.” The original probe3 used different arrays per tree; its precise A3/master ratios await shared-array Zen 3 confirmation.

**7. Apply the policy consistently; extend placement coverage to B925.** Rejecting A3 under the stated rule remains warranted even without proving the hardware mechanism. The formal6 aggregate reports two clean-control losses in all six processes, and controlled placements independently reproduce substantial losses. Reclassifying a repeated-input result as predictor sensitivity does not erase that supported workload.

The new-master tables do disclose costs of the other proposals: D3's 3-entry **1.06–1.08×** and 5-entry zero-group **1.10–1.11×**; V's 1-entry Float32 **1.08–1.13×** and 2-entry Float64 **1.06–1.08×**; B925's tiny ordinary calls up to **2.55×**, plus fallback costs. D3/V therefore require explicit acceptance under the same >5% policy, despite tiny absolute costs. Larger per-process maxima (e.g. D3 valprox 1.39, V value 1.20 in §8.6) are not additional demonstrated six-process regressions; they remain unresolved variation.

Source inspection distinguishes placement exposure:

- **V:** the new loops reduce one input to a scalar; no output-array store in the fast loop.
- **D3:** the changed zero-group route skips a read scan and writes zeros. Its unchanged nonzero scaling loop does read x and store y (`src/numerical_kernels.jl:915–920`), so absolute placement sensitivity is possible, but no new faster read/store loop was introduced there.
- **B925:** yes. Its ordinary step-array route reads x and steps/weights and stores y in the newly vectorized loop (`review3/patch-B925.diff:10–39`; `src/functions/separable.jl:143–148`). Sweep y relative to **each input**, not just x, before calling one short-array revision sufficient for proposal.

The mechanism is not A3-specific; the probes themselves show ridge and broadcast penalties. Differential regression for D3 or B925 cannot be inferred just from their having stores, and has not been tested here. Finally, `REPORT.md:850–852` measures a descriptive fraction of outside-reach regressions, not a calibrated false-positive probability: these cells share code and layout effects. That fraction does not justify dismissing D3's affected 3-entry regression.

## MINOR

**8. Layout probe still dispatches dynamically.** `layout_probe.jl:41–46` passes a Module and looks up `M.prox!` inside every timed call. My Float64 check in `check_impls.jl` shows `Body::Any` and **32 allocated bytes/call on both versions**, versus concrete return/no allocation for the offset closure. Capture the function value as in offset_probe. This contaminates small timings and permits GC interference; it does not explain the hundred-microsecond placement reversals.

**9. Digests and summaries need explicit provenance/completeness rules.** The offset/layout digests use **ratios of round medians**, not medians of paired round ratios; underlying rounds cannot be recovered from these CSVs. The CPU×Julia grouping regex correctly separates every supplied filename; it intentionally pools nodes/campaigns. It does not mix Julia versions. But `offset_summary.py:26` silently drops an entire size when its last offset is missing, while `offset64.py` includes available cells. Probe5 a04-18 / Julia 1.10 / process 2 stops after 592 data rows, partway through Float32/16384. Label partial runs and sample counts. There are two Broadwell nodes in the pooled +64 data, not one (`REPORT.md:922`).

**10. Some summary language omits already documented costs.** A3's §9 cost (c) says 4–16 entries, but the new-master tiny table includes Float32 ElasticNet block-loop n=100, **1.06–1.12×** on Zen 3 / 1.13 (`845`; `tiny3-slowcells.txt`). Its tiny NormL1 prox losses at n=1/4 are also outside that summary. Carry the actual affected cases into the recommendation instead of compressing them into three inaccurately bounded categories.

## NOTE: checks and limitations

The fixed `offset_probe.jl:35–50` has distinct concrete captured function/object values and no `Core.Box` fields in my checks on either Julia version; all 16 operations produced the expected finite outputs. Its pointers really have the specified residues: separate 2 MiB-aligned regions mean x at px+4096 and y at py+4096+yoff differ by yoff modulo 4096. Regions do not overlap. `unsafe_wrap` defaults to non-owning; keeping the malloc regions until timings finish and freeing their base pointers afterwards is sound. GC cannot move/free these C allocations; no `GC.@preserve` repair is needed for their ownership. Actual data alignment is sufficient. Setup/fill/calibration are outside offset timings; noinline side-effecting calls prevent hoisting. Calibration chooses approximate batch lengths, not an exact controlled duration.

No retained-closure/shared-variable timing bug was found in sparse_probe: each timing closure is consumed immediately, and `sweep(p::P,...)` specializes the inner calls. The shared-array version keeps separate input/output backing arrays and stable stored-coordinate metadata for this workload. Its copying is intentionally timed. Probe1's calls select the right implementations despite the dynamic dispatch above. None of these three probes independently asserts output equivalence; keep a correctness check before timings.

Local full offset sweeps on both Julia versions reproduce the direction of the positive-offset penalty on this Intel CPU; they cannot reproduce EPYC 7513 ratios. All scripts, source copy, generated assembly, environments, CSVs and recomputations are under `.codex-probes/round4/`. The supplied command needed a resolved environment containing **PrecompileTools**, imported by `src/precompile.jl:1`; it is not literally standard-library-only. I used the installed versioned Julia binaries because juliaup's lockfile is read-only here.

I did not run cluster jobs, inspect hardware counters, isolate huge-page policy at fixed physical addresses, or audit patch numerics again. Raw formal5/formal6/tiny3/native campaign CSVs are **absent from review4**; their six-process decisions and D3/V/B925 classifications were checked against supplied tables/report, not independently recomputed from those raw rounds. Original probe3 source bytes were not supplied separately; its own-array behavior is reproduced by the documented switch, not verified against a per-job source hash. Presentation/typesetting outside numerical claim precision was not assessed.

The inspected source/evidence snapshot and adjacent source bundle are preserved in `.codex-probes/round4/placement-round4-snapshot.json{,.sources}`; final verification is recorded beside them. Newly generated probe outputs are additional evidence, not part of that initial snapshot.
