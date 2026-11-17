Independent review of PR #969, round 7 — 2026-11-16

The numerical replacement A3 passes the checks below. **A3d is not API compatible as written:** valid separate-output calls on arrays with abstract element types now throw. The available data favor universal A3 for ordinary allocations, but do not yet establish the final choice. In particular, a Zen-only cutoff does not cover all observed placement regressions.

Findings that need resolution before a merge decision:

- **Implementation, major, confirmed:** A3d evaluates `sizeof(Real)` or `sizeof(AbstractFloat)` on supported arrays. Guard the byte calculation and retain the existing route for these element types. No Float32/Float64 numerical mismatch was found.
- **Measurement coverage, major, confirmed:** neither A3c nor A3cr implements the proposed 32 KiB rule. On Zen, A3cr still selects A3 for Float64 lengths 4096–8191, where the old probe already shows regressions; it also gives up A3 for long `y === x` calls. A3d is absent from the supplied Intel matrix. A final comparison must actually measure the byte-cutoff candidate on all three architectures.
- **Postprocessing, moderate, confirmed:** both probe digests crash on the new allocation schema; the offset digest silently overwrites different `ypages` cases. The matrix digest can classify incomplete rounds or missing metadata as valid, and includes explicitly invalid cells in geometric means. These are small repairs; they do not invalidate the complete Intel raw data supplied here.
- **Measurement qualification, moderate, confirmed:** `MADV_HUGEPAGE` is advice, not verification of backing. The offset CSV does not record backing actually obtained. Record it before the final runs. The new allocation `hugepct` works on the controlled layouts tested here, with a pre-existing `MADV_DONTDUMP` caveat below.

Other remarks: the three clean-control Intel matrix regressions of A3 are indeed tiny. However, the ordinary-allocation maximum on Skylake-SP is **1.033518**, not at most 1.03. The forced-placement probe also contains clean-control Skylake-SP regressions of about 8–9%, so “non-Zen always faster” is false. A3cr's package-image initialization passed, but its overhead is measurable on very short arrays. Claims (c) and (e) need qualifications about offsets and input state, respectively.

All retained scripts and outputs are in [round7](round7/). Source trees and the harness were left unchanged. I used the audit skill to preserve source snapshots and check direct implementation and measurement dependencies. This is a source/code/data audit, not a mathematical verification of the package's pre-existing numerical recovery machinery. A transient audit snapshot was initially created under `/tmp`; it was removed and replaced by snapshots inside `round7`. No cluster was contacted.

**1. Implementations and numerical/API checks**

The inspected commits are M4 `f8148203`, A3 `0271e994`, A3d `720203fb`, and A3cr `c3db3496`. Identities, source diff counts, and input-file SHA-256 hashes are in [input-identities.txt](round7/input-identities.txt). Snapshots and adjacent source bundles are `round7/{master4,pr-A3,A3d,A3cr,harness}-snapshot.json*`; final verification is in [snapshot-verify.log](round7/snapshot-verify.log). All five report `valid: true`; the harness snapshot reports an uninspected worktree-status change, with no change to any captured file. Raw-input hash rechecking is in [input-hash-verify.txt](round7/input-hash-verify.txt).

For the commands below, the working directory is `review-code`. I used these environment settings for Julia dependency/cache writes; the shared benchmark project was read-only:

```bash
export JULIA_DEPOT_PATH="$PWD/round7/depot:/path/to/.julia"
export TMPDIR="$PWD/round7/tmp"
# Release commands use --project=../harness/experiment_kernelopt/env-v1.13
```

I reran the supplied public-entry-point comparison with both bounds settings:

```bash
bash round7/run_equiv.sh
```

[Bounds-yes output](round7/equiv-yes.log), also reproduced with bounds disabled:

```text
A3:   274752 results compared with the reference, 0 different, 0 equal except for the bits of a NaN
A3d:  274752 results compared with the reference, 0 different, 0 equal except for the bits of a NaN
A3cr: 274752 results compared with the reference, 0 different, 0 equal except for the bits of a NaN
```

Thus (a) reproduces **for that test's coverage**. Its exit status alone is insufficient for strict bitwise equivalence: `equiv_trees.jl` counts changed NaN bits separately but does not fail on them. Here that count really is zero. It also lacks custom NaN payloads and views, and its lengths do not bracket the Float64 byte cutoff exactly.

My independent [differential.jl](round7/differential.jl) compares raw UInt bits, without treating different NaNs as equal. It covers Float32/Float64, positive and negative quiet payload NaNs and signaling NaNs, signed zeros, subnormals, extreme finite values and infinities; absent and zero/infinite mixed weights; scalar and array steps including ±0, ±Inf, negative and NaN; both functions; and `prox!`, allocating `prox`, `value_and_prox!`, every scalar `prox_block`, and `SparseIterate` output. Shapes include arrays, strided views, reshapes, and a custom non-host `DenseArray`. Lengths include 0, 1, 16, and the immediate neighbors of 4096, 8192 and 16384. Constructor levels/ridge exceptions are additionally covered by the supplied comparison.

```bash
julia +release --startup-file=no --threads=1 --check-bounds=yes \
  --project=../harness/experiment_kernelopt/env-v1.13 round7/differential.jl
KO_REVIEW_CUTOFF=8192 julia +release --startup-file=no --threads=1 --check-bounds=yes \
  --project=../harness/experiment_kernelopt/env-v1.13 round7/differential.jl
KO_REVIEW_CUTOFF=8192 julia +lts --startup-file=no --threads=1 --check-bounds=yes \
  --project=../harness/experiment_kernelopt/env-v1.10 round7/differential.jl
```

[Default release](round7/differential.log), [forced cutoff](round7/differential-cutoff8192.log), and [Julia 1.10.13](round7/differential-lts.log) each report:

```text
master4 ambiguities=0
pr-A3 ambiguities=0
A3d ambiguities=0
A3cr ambiguities=0
pr-A3: 32256 independent comparisons, differences=0
A3d:   32256 independent comparisons, differences=0
A3cr:  32256 independent comparisons, differences=0
```

The default release log predates a change to use `invokelatest` in my loader; its world-age warnings come from my test loader, not the candidates. The subsequent forced-cutoff and LTS runs use the corrected loader. The explicit 8192 setting exercises A3cr's long branch even on this Intel machine.

[extra_checks.jl](round7/extra_checks.jl), run with the same release project and bounds disabled, adds actual **view destinations**, strided/reshaped/reversed views, and `y === x`, across Float16/32/64. [Output](round7/extra_checks.log): `864 comparisons, differences=0`. A small BigFloat example also returns identical values and signs across all four trees; BigFloat keeps `shrink`.

A3d's API counterexample is separate from those concrete floating-point comparisons:

```julia
x = Real[1.0, -2.0]
prox!(similar(x), NormL1(0.6), x, 0.8)
# M4, A3, A3cr: [0.52, -1.52]
# A3d: ErrorException("Abstract type Real does not have a definite size.")
```

The logs reproduce the error for both NormL1 and ElasticNet, and for `AbstractFloat` too. The cause is [A3d norml1.jl:98](/path/to/pbopt/A3d/src/functions/norml1.jl:98). It also affects allocating `prox` through this route. `y === x` short-circuits the calculation and avoids it, which makes separate-output coverage important. A small repair is to apply the byte test only to the intended concrete Float32/Float64 element types, returning the existing A3 route otherwise. Compare lengths against a divided cutoff to avoid unnecessary byte multiplication.

There are **no detected new ambiguities**, but the cutoff trees certainly change which existing calls reach which internal method: host-array scalar-step calls now hit their `_prox_sep!` overloads. `which` output is retained in the differential logs. Non-host arrays retain the generic NormL1 route; ElasticNet's generic `AbstractArray` prepared route remains present. The custom `DenseArray` checks exercise these fallbacks. They are evidence against accidentally capturing device `DenseArray` subtypes, not a GPU test. Public signatures/exports are unchanged; A3d's added host specialization is nevertheless an observable API break. My verdicts are: A3 passes the tested correctness/API scope; A3d fails API compatibility; A3cr passes the tested correctness/API scope, with the performance-selection deficiencies discussed below.

The arithmetic replacement itself has a simple reason to agree for valid ordinary thresholds: outside `[-c,c]`, subtraction of the selected endpoint gives the same surviving `v-c` or `v+c`; inside, it gives zero, and addition of +0 repairs negative zero. Exceptional parameter branches still use `shrink`. I found no reason to add numerical recovery code.

**Package images.** I copied A3cr's `src`, project file and package-image check into `round7/A3cr-copy` and ran:

```bash
julia +release --startup-file=no round7/A3cr-copy/experiment_pkgimage_check.jl
julia +release --startup-file=no round7/A3cr-copy/pkgimage_runtime_name.jl
```

The [original check output](round7/pkgimage.log) reports precompilation with override 123, fresh loads from the same temporary `.ji`/`.so` with `--compiled-modules=existing --pkgimages=existing`, runtime values `typemax(Int)` and 456, and unchanged cache names/contents. My [extended check](round7/pkgimage-runtime-name.log) changes `Sys.CPU_NAME` **before loading** in fresh processes, using the same cache:

```text
raptorlake -> 9223372036854775807
znver3    -> 8192
znver4    -> 8192
override  -> 456
cache files unchanged (names and contents): true  # all four loads
```

Those Zen names are simulated, not a test on Zen hardware. They directly test that the cached initialization still reads runtime state. Julia 1.13.4's local `base/sysinfo.jl:146,188` declares `CPU_NAME` as a non-constant global and assigns it in `Sys.__init__`; `isconst(Sys, :CPU_NAME)` prints `false`. The candidate's `Ref` is mutated in package `__init__`, rather than embedding the CPU decision as a precompilation-time constant. **The selection mechanism is safe within these checks.** Portability of the native code image itself is still Julia's CPU-target/cache-loader responsibility.

**Short-array cost.** [short_timing.jl](round7/short_timing.jl) uses shared x/y, alternating order, 11 short batches, both functions, on this machine (Raptor Lake, Julia 1.13.4):

```bash
julia +release --startup-file=no --threads=1 --check-bounds=no \
  --project=../harness/experiment_kernelopt/env-v1.13 round7/short_timing.jl
```

Selected [output](round7/short_timing.log):

| Function | n | A3 ns | A3cr ns | A3cr/A3 |
|:--|--:|--:|--:|--:|
| NormL1 | 1 | 2.7456 | 2.9416 | 1.0719 |
| NormL1 | 16 | 3.5404 | 3.7348 | 1.0549 |
| ElasticNet | 1 | 6.1255 | 6.3202 | 1.0351 |
| ElasticNet | 16 | 7.1211 | 7.3179 | 1.0286 |

The [LLVM output](round7/short-llvm.txt) has one global Int load, length comparison and branch per call, outside the loop. The measured difference includes the whole selection implementation and code placement, so it is not a pure isolated Ref latency. About 0.2 ns here is small absolutely but 3–7% on the smallest calls. The supplied Intel matrix also has extra tiny regressions for A3cr (eight clean-control cells on Broadwell, two on Skylake-SP versus three/zero for A3). Do not claim zero or uniformly negligible overhead without defining an acceptable short-call cost.

**Source size and machinery.** Command: `git -C ../<tree> diff --numstat f8148203 -- src`.

| Candidate | Lines added | Removed | Net added |
|:--|--:|--:|--:|
| A3 | 20 | 1 | 19 |
| A3d | 57 | 11 | 46 |
| A3cr | 71 | 11 | 60 |

These are physical source lines, including comments/blanks, not test changes. A3 adds three helper methods and changes one call site. The cutoffs add two host-array routes, a NormL1 wrapper with forwarding checks, long-loop entry points, and an extra soft-function parameter in the existing ElasticNet prepared step; A3cr adds initialization/Ref/environment handling too. A3 plainly meets the simplicity policy. The cutoff variants are still localized machinery, not an extensive public dispatch framework, but need a demonstrated benefit to justify their extra surface. A3d cannot merge before its API repair.

**2. Matrix methodology and independent recomputation**

The main timing design is sensible: typed closures, an opaque non-inlined invocation to resist loop hoisting, warmup/calibration outside timing, shared input/output placement for preallocated comparisons, paired rounds, and rotating/reversing order. Six fresh processes vary tree compilation order and A/A-control position. Minimum-of-three calibrated batches estimates warmed throughput rather than latency under arbitrary application conditions. Twenty-one rounds do not perfectly balance every position for every implementation count; rotation/reversal is still a reasonable drift control. It is not a statistical confidence procedure.

The A/A compilation is useful for detecting code-placement/frequency effects, especially at nanosecond scales. A clean A/A control does not eliminate all code-layout effects; an unclean control makes a result unresolved, not proof that a candidate is equal or faster. A 5% effect in every process plus A/A within ±2% in every process is a conservative operational rule. Report its exclusions and absolute ns, and do not interpret geometric means as workload-weighted performance.

The operation audit of `bench/suites.jl:463–522` is:

| Operation | What is actually timed / checked |
|:--|:--|
| `prox_out` | Correct preallocated separate-output scalar-step call, on the same x/y for all trees. |
| `prox_in` | Repeated `prox!(z,f,z,γ)` without restoring z. Calibration quickly drives it to all zeros. It measures repeated in-place calls on that stationary state. Its output callback restores x **and reruns prox**, so it does not validate the timed operation's result. |
| `prox_incopy` | Correctly times `copyto!(z,x)` plus one in-place prox. The copy is included, not an isolated prox time. The `copy` implementation is a useful separate floor, but subtracting noisy timings is not a reliable isolated-prox estimate. |
| `prox_alloc` | Calls allocating `prox`, retains the latest result through a Ref, and includes allocation/GC behavior. Outputs are newly allocated, so placement cannot be identical across trees. It is a valid end-to-end measure with that limitation. |
| `block_loop` | Calls `prox_block` for each entry in an inbounds loop, writing y. The compiler may inline/vectorize this loop. It is not an isolated scalar-call latency benchmark. |
| `sparse_prox` | Genuine sparse stored-entry call. The tree-specific wrappers share value/index arrays; each wrapper has its own nnz field. The supplied case has 6553 stored entries among 65536 logical entries. The full nzval backing is compared, but index maps/nnz/logical output are not independently checked by this benchmark. The separate equivalence tests provide stronger numerical coverage. |

[Evidence script](round7/inplace_check.jl):

```text
$ julia +release ... round7/inplace_check.jl
 after 10 timed operations: [0.0, 0.0]
 output callback: [1.52, -1.52]
 Harness check failures with defective timed no-op: 0
```

This demonstrates the `prox_in` validation hole without alleging an incorrect timed closure in the supplied suite: its actual closure does call prox. `prox_incopy` supplies the important nonzero-input companion. Fix validation by resetting before the validation call and comparing the result of that call, then label the repeated stationary timing explicitly.

**Digest validity is incomplete.** `matrix_digest.py:50` checks the number of processes contributing a median and the absence of a recorded `DIFF`; it does not require metadata to exist, expected round sets, unique raw rows, finite positive times, matching provenance/settings, or the planned number of processes. Its geometric means at lines 60–63 bypass even this filter. A stale DONE can conceal a partial rerun. In contrast, the current raw data passed my stricter checks.

```bash
python3 round7/digest_fixture.py
```

The [fixture/output](round7/digest_fixture.log) has two DONE processes, only one round each, missing candidate/control metadata for n16, and an explicitly wrong n32 candidate. The digest says:

```text
cells with complete, valid data: 1 of 2
| l1 | prox_out | short | 2 | 2.00 [2.00–2.00] | 1.00 |
A3: 1 of 2 cells slower in every process
```

Thus the “complete, valid” claim is false in general, and the GM includes a known-invalid cell. Require the expected metadata and paired rounds, reject duplicates/nonfinite values and CHECKFAIL, validate consistent provenance, then use one explicit validity predicate for classifications **and** means. DONE alone is insufficient.

**Actual recomputation.** [recompute.py](round7/recompute.py) independently requires 21 unique rounds for each `(case,impl)`, positive finite timings, matching raw/meta keys, status `ok`, zero output/active-set differences, DONE and no CHECKFAIL. It computes the median of round-wise A3/M4 ratios within each process, then the median across processes per cell. All 12 Intel processes have 1000 cases and 150360 raw rows; they passed these checks.

```bash
python3 round7/recompute.py > round7/recompute.txt
python3 ../harness/experiment_kernelopt/bench/matrix_digest.py \
  ../harness/experiment_kernelopt/results/m4/a3matrix1 > round7/matrix_digest.txt
```

A3M4 has these **>1.05 in every process with clean control** cells:

| CPU | Cell | Ratio range | M4 → A3 median ns |
|:--|:--|:--|:--|
| Broadwell | `l1/prox_out/Float32/n1/normal` | 1.124548–1.125081 | 7.0794 → 7.9644 |
| Broadwell | `enetw/block_loop/Float64/n16/normal` | 1.073539–1.083414 | 17.9932 → 19.3872 |
| Broadwell | `enetw/block_loop/Float32/n4/normal` | 1.058821–1.091864 | 14.7492 → 15.9290 |
| Skylake-SP | none | — | — |

Without the control gate there are six Broadwell cells: those three, plus `l1/prox_in/{Float32,Float64}/n1/normal` (both about 1.125, 7.0795 → 7.9643 ns) and `l1/prox_out/Float64/n1/normal` (1.124981–1.173923, 6.7846 → 7.9643 ns). Skylake-SP has one: `l1w/prox_g_out/Float64/n1/normal`, 1.059669–1.200103, 8.3693 → 9.6655 ns, with control 1.098628–1.120098. They are unresolved by the declared rule. Full process-level ratios/control ranges are in `recompute.txt`.

These match the supplied digest: **3/1000 and 0/1000** clean-control slow cells; 480 and 227 clean-control fast cells. Claim (b)'s matrix description is fair when “slower” means the declared reproducibility rule, not literal absence of every >1 ratio.

The independently recomputed `prox_out` geometric means below combine Float32/Float64. Short is n≤64 (14 cells/family), medium n128–8192 (14), large n>8192 (12). Entries are **A3 time / M4 time**; speedup is their reciprocal.

| CPU | Family | Short | Medium | Large |
|:--|:--|--:|--:|--:|
| Broadwell | l1 | 0.955360 | 0.667350 | 0.883962 |
| Broadwell | l1w | 0.920791 | 0.717945 | 0.972236 |
| Broadwell | enet | 0.957454 | 0.685988 | 0.891608 |
| Broadwell | enetw | 0.952984 | 0.741250 | 0.968570 |
| Skylake-SP | l1 | 0.954098 | 0.772972 | 0.857757 |
| Skylake-SP | l1w | 0.966419 | 0.877113 | 0.959778 |
| Skylake-SP | enet | 0.992808 | 0.840260 | 0.893083 |
| Skylake-SP | enetw | 0.974712 | 0.917723 | 0.961937 |

All round to the corresponding entries in [matrix_digest.txt](round7/matrix_digest.txt). Across all four families, the ratios are Broadwell **0.946527 / 0.702557 / 0.928170**, Skylake-SP **0.971908 / 0.850327 / 0.917051**. Those correspond to speedups **1.057 / 1.423 / 1.077** and **1.029 / 1.176 / 1.090**, respectively. The report's size classes are in entries, whereas the proposed cutoff is in bytes; do not conflate them.

A3c's all-family large means are 1.000330 and 1.000585; A3cr's are 0.928307 and 0.917795. This confirms that the universal entry cutoff gives up meaningful Intel gains. It does **not** directly measure A3d, which cuts Float64 off earlier and exempts `y === x`.

**3. Probe methodology and raw-file results**

**Offsets and backing.** `offset_probe.jl:67–78` obtains separate 2 MiB-aligned allocations, advises their backing, and wraps them as actual Arrays. Input begins at `px+4096`; output begins at `py+4096+yo+4096*pg4k`. Thus `yo` is the requested relative offset modulo 4096. At zero page shift they also have nearly matching locations within their respective 2 MiB pages. These are separate arrays, not overlapping views or an in-place test.

Nonzero page shifts separate the two alignment hypotheses as intended: they keep the 4 KiB residue and change the within-2-MiB position by `4096*pg4k` modulo 2 MiB. Multiples of 512 pages do not change that position. Shifts run only for `PAGE_SIZES` and offsets 0,64,96,128,192; default `PAGES` is **only 0**, so the submission environment must actually set `KO_OFFSET_PAGES`. Large or negative unchecked shifts could escape the allocated region; modest shifts such as 1 and 17 used here are safe. Add a bounds assertion or size the region using the maximum shift.

I instrumented a copy of the real probe, retaining its allocation and timing code, with actual pointers/backing. The local run was intentionally small:

```bash
KO_OFFSET_SIZES=4096 KO_OFFSET_PAGES=0,1,17 KO_OFFSET_PAGE_SIZES=4096 \
KO_OFFSET_IMPLS='^(l1)/(ref|cand|refC)$' julia +release --startup-file=no \
  --threads=1 --check-bounds=no --project=../harness/experiment_kernelopt/env-v1.13 \
  round7/offset_instrumented.jl ../master4 ../pr-A3 round7/offset-local.csv 1
```

[Output](round7/offset-local.log) includes, for Float64 n4096, yo64:

```text
pages=0  actual-mod4K=64 within2MiB=(0x1000,0x1040)  hugepct=(100,100)
pages=1  actual-mod4K=64 within2MiB=(0x1000,0x2040)  hugepct=(100,100)
pages=17 actual-mod4K=64 within2MiB=(0x1000,0x12040) hugepct=(100,100)
```

This verifies the placement arithmetic and demonstrates successful THP backing locally. It does not prove the backing of the old cluster CSVs: a successful advice call alone cannot do that. `small` requests MADV_NOHUGEPAGE and currently runs only at n=1048576, at three offsets. No old sweep varies the argument's own offset (always 4096 within its 2 MiB page); the new page shifts vary the output's position, not every possible placement variable.

**The new huge-page share.** I extracted the actual `hugepct` definition from the audited allocation script and tested it on an aligned 12 MiB anonymous region. MADV_HUGEPAGE plus successful MADV_COLLAPSE established known THPs; a middle 4 MiB was then discarded/refaulted under MADV_NOHUGEPAGE.

```bash
julia +release --startup-file=no round7/hugepct_check.jl
```

[Output](round7/hugepct_check.log):

```text
before kb=12288
probe 1 percentage=100
probe 2 percentage=100
probe 3 percentage=100
after kb=12288
mixed before kb=8192
mixed full pct=67 expected=67
mixed inside-small pct=0 expected=0
mixed huge+small pct=50 expected=50
mixed after kb=8192
```

For ordinary dumpable anonymous mappings, splitting VMAs at 2 MiB boundaries this way worked and did not split the existing THPs or change their total backing. Ordinary `zeros` arrays of the supplied sizes were also exercised (all 0% on this local madvise-default machine). The denominator is the enclosing 2 MiB windows, **not** a byte-weighted percentage of the array's exact slice of its boundary windows. All-100% and all-0% are useful; intermediate numbers need that interpretation.

There is a genuine qualification: the code ignores advice return values and assumes the dump flag changes at the requested boundaries. I marked the entire huge region DONTDUMP before probing its interior:

```text
preexisting DONTDUMP rc=0
preexisting DONTDUMP inner pct=0 expected=100
```

The advice no longer splits that already-DONTDUMP VMA, the `inside` predicate excludes it, and the function's DODUMP cleanup changes the pre-existing flag. This is not evidence that the current Julia allocation trials have such mappings. It means “exact under any mapping state, changing nothing else” is too strong. A small defensive repair is to detect pre-existing `dd` flags / failed split coverage and emit **unknown**, and preserve original flags, rather than silently report zero. Record advice errors too. No complex recovery is needed. This probe also cannot certify that THP backing remains unchanged throughout later timings; a post-timing recheck would distinguish backing transitions if the new results are mixed.

**Allocation patterns.** `similar`, `gap`, `prox` and `later` are useful, recognizable caller allocation patterns, not a representative probability sample of Julia applications. `prox` selects its initial output using M4 but then times preallocated calls for all trees on the same x/y; it measures the placement obtained by allocating prox, not the cost of allocating prox itself. The matrix's `prox_alloc` measures that cost. The unrelated arrays in `later` are undef; dropping their references does not ensure collection before y is allocated. The header's “some freed” should be softened to “some made unreachable.” Even trials explicitly clear retained arrays and collect; an odd trial retains the preceding trial's arrays until the next clearing, not the entire history.

The current loop puts **all ordinary cases across both types and all lengths before either advised pattern**. Advice persists in allocator-reused memory, but the advised patterns therefore cannot contaminate earlier ordinary trials in that process. Fresh processes do not inherit VMAs. The two advised patterns may affect each other, appropriately treated as an explicit THP stress condition rather than unadvised ordinary allocation. The smaps reader itself allocates and affects later heap state; neither this probe nor any finite allocation sample establishes the application's placement distribution.

The trials establish an observed range of warmed timing ratios on the sampled address/backing/process states. They cannot prove “A3 is never slower than M4 by more than r under ordinary allocations” for all heaps, hosts, memory pressures, input types or applications. Rounds within a trial and trials on one node are not independent machines. Report observed maxima, process/node counts, controls and backing, and call the bound empirical. The old mapping-wide huge-page columns cannot determine whether an array itself was on THPs, as the question already recognizes.

**Raw-file recomputation.** My script keys every trial by process, type, length, pattern and trial number, checks that all six function/implementation timings are present, and uses `med_ns(cand)/med_ns(ref)`. Offset cells use the median of these per-process ratios; the supplied probe digests do likewise, rather than the matrix's median of round-wise paired ratios. Old CSVs do not preserve individual rounds, so those cannot be reconstructed.

```bash
python3 round7/recompute.py
python3 ../harness/experiment_kernelopt/bench/a3probe_digest.py \
  ../harness/experiment_kernelopt/results/m4/a3probe1
python3 ../harness/experiment_kernelopt/bench/alloc_digest.py \
  ../harness/experiment_kernelopt/results/m4/a3probe1
```

Full outputs: [recompute.txt](round7/recompute.txt), [a3probe_digest.txt](round7/a3probe_digest.txt), [alloc_digest.txt](round7/alloc_digest.txt). The old allocator source is preserved as [alloc_probe-874ac729.jl](round7/alloc_probe-874ac729.jl).

| CPU | Allocation trials | Function comparisons | Observed A3/M4 range | Worst M4 → A3 ns |
|:--|--:|--:|:--|:--|
| Zen 3 | 3456 | 6912 | 0.664710–0.931964 | 110353.12 → 102845.11 |
| Broadwell | 3456 | 6912 | 0.640179–0.995066 | 534748.50 → 532110.00 |
| Skylake-SP | 3456 | 6912 | 0.701276–1.033518 | 900209.00 → 930382.00 |

The Zen maximum is process 2, Float32, n1048576, `gap`, trial 3, ElasticNet. There are 1410 function comparisons with output residues 48–192, ranging 0.665971–0.931964, and none of all 6912 comparisons exceeds 1. Thus **(d) is correct as an empirical statement**. It does not resolve the backing question. Broadwell never exceeds 1 in these samples; Skylake-SP has 1331 ratios above 1, none above 1.05. Its maximum is process 1, Float64, n1000000, `later`, trial 2, NormL1. **(b)'s “never above 1.03” is literally wrong**; “none above 1.034 in these trials” is accurate. Control ranges over all allocation comparisons are Zen 0.964344–1.037574, Broadwell 0.975105–1.063958, Skylake-SP 0.934823–1.041605; do not attach unwarranted precision to isolated small differences.

**(c) is directionally right about onset and the severe Zen effect, but wrong as a uniform band/outside-band description.** Every tested size ≤16 KiB has median ratios below 1 at every tested offset. The first 32 KiB sizes show substantial regressions. The worst median-of-process ratio is Float32 ElasticNet n262144, offset 96: **1.430360**, with process ratios 1.414789, 1.430360, 1.433882 and M4→A3 times:

```text
28426.74 -> 40217.84 ns
28776.26 -> 41160.40 ns
28717.00 -> 41176.79 ns
```

Its control is 1.065940, 1.020452, 1.021048, so it fails the matrix's strict control rule. The largest minimum process ratio among **clean-control** Zen probe cells is Float32 ElasticNet n32768, offset 96: ratios **1.361939–1.418691**, control **0.983308–1.015710**, median absolute times **3554.48 → 4910.17 ns**. There are 140 huge-advised Zen cells >1.05 in every process, 51 with every control within ±2%.

Not every 48–192 offset/size is 1.07–1.43: for example Float64 NormL1 n4096 at offset 64 has median 1.005421, and Float64 n262144 at offset 80 has median 0.943748. Nor is “elsewhere 0.67–0.88” uniformly true: **offset 32**, Float64 ElasticNet n4096, has median **1.112697** (outside the stated band); offset 256 can also be slightly above 1. The low 4 KiB-size case even reaches 0.589987. Use the full offset curves or a precisely defined envelope, not a rectangular threshold claim.

The same sweep is not universally harmless on Intel. Broadwell has no cell >1.05 in every process. **Skylake-SP has 25 huge-advised cells >1.05 in every process, all with clean controls.** Its worst by minimum process ratio is Float64 NormL1 n4096, offset 128:

```text
A3/M4: 1.081963, 1.091678, 1.080984
control: 0.998394, 0.999146, 0.999479
M4 -> A3: 3118.96 -> 3374.60; 3126.60 -> 3413.24; 3127.27 -> 3380.53 ns
```

That is a reproducible placement regression even though the Intel matrix and ordinary-allocation samples look favorable. A3cr's source comment asserting improvement on every non-Zen processor/length needs correction.

**(e):** local LLVM dumps for both M4 and A3 show `vector.memcheck`, interval overlap comparisons and a branch to the scalar loop when x/y overlap. For identical nonempty storage that condition is true. This supports the mechanism in the claim for the inspected dense scalar-step loop; exact identity removes the relative-placement variable. It does not establish the quoted 0.67–0.80 range on Zen at every length. There are no Zen matrix results supplied yet, and `prox_in` measures zero-state repetition. The queued `prox_incopy` data are the useful companion for that claim. The existing Intel n1 in-place regressions already disprove extending the numeric range to every architecture/length.

**4. Concrete changes before the final Zen runs**

The timing scripts do not have a discovered universal failure that makes all newly produced timings unusable. The current Intel raw files are complete and recoverable. These are the small changes I would make, distinguishing collection from later analysis:

1. **Fix A3d's element-type guard**, then include it in the formal tree list on Zen 3, Broadwell and Skylake-SP. Retain both soft implementations. Add a small fixed-placement comparison of A3d at Float64 4096 and Float32 8192 (and neighboring lengths), plus long identical-output calls. Running just A3 versus M4 again cannot establish that the actual cutoff removes regressions without other costs. Current A3cr needs a byte-based threshold and identity exemption before it can claim the same protection.
2. **Record actual backing in the offset CSV**, using the validated VMA-splitting measurement with its guard/unknown case, and retain `ypages` in analysis keys. Set and record nonzero `KO_OFFSET_PAGES` explicitly (e.g. 0,1,17), because the batch script does not supply nonzero defaults. Assert the shifted array stays in its region. Without backing measurement, the forced-THP causal interpretation can remain unresolved after the run.
3. **Add a true single `prox_block` call** for each function/type (sink its result), rather than treating the scalar API called in an optimizable 4–1024-entry loop as isolated scalar performance. Existing array operations, step-array operations, sizes and end-to-end allocating prox are otherwise present. Correct `prox_in` validation and label its stationary-state input; retain `prox_incopy`.
4. **Fix the postprocessors:** accept `xhuge_pct/yhuge_pct` and retain the old schema only as “mapping-wide/unknown array backing”; key offsets by backing and `ypages`; require complete valid metadata/rounds for the matrix and filter means too. These fixes can follow collection, but must precede interpretation. My [current-schema fixture](round7/probe_schema_digest.log) first prints only the final page shift's ratio (1.50, losing the shift-zero 0.50), then crashes with `KeyError: 'xhuge_kb'`; [alloc_digest](round7/alloc_schema_digest.log) crashes identically.
5. **Make strict equivalence fail on changed NaN bits** and add a few signed/payload NaNs and cutoff-neighbor/view cases to the pre-run check. Current supplied runs have zero NaN-bit differences, so this is a coverage/gating repair, not an observed numerical failure.

Also record the allocation RNG seed (currently `getpid()`), the selected A3cr cutoff/override and page-shift settings in probe provenance. If testing processor selection, compare it on all three architectures and retain the very-short-array overhead cells. There is no need for long Julia 1.10 performance runs, pointer heuristics, or new numerical recovery.

**5. Which implementation should be proposed?**

**My provisional choice is universal A3**, retaining M4's soft threshold as the experimental comparator. It is the smallest change, passes the numerical checks, wins substantially in the Intel medium/large matrix, and beats M4 in every sampled Zen ordinary-allocation trial. This is a provisional performance recommendation, not permission to describe the known placement regressions as nonexistent. The queued data must decide how relevant those regressions are under realistic backing and placement.

| Option | What the final data must establish | Current judgment |
|:--|:--|:--|
| Universal A3 | Zen matrix covers all policy operations; ordinary/advised allocation probes distinguish backing and show whether sizeable regressions occur in plausible caller states. Quantify worst reproducible absolute/relative losses alongside its common gains. | Simplest and best supported for ordinary allocations so far. Forced-placement losses remain real evidence against a blanket win. |
| Universal byte cutoff, A3d | After API repair, shared-placement timings show it removes the important regressions at both type-dependent boundaries and keeps long in-place gains. Measure its Intel losses and Zen good-placement losses, not just its worst-case rescue. | A credible simple fallback if those losses buy a worthwhile worst-case guarantee. It also covers the known Skylake-SP 32 KiB case. Intel A3c data already warn that unconditional cutoffs sacrifice gains; A3d itself still needs measurement. |
| Processor-specific cutoff, A3cr | Demonstrate a meaningful overall advantage over the universal choices, acceptable short-call cost on each CPU, and adequate protection in all relevant states; retain the tested runtime/image-safe initialization. | **As written, not a complete solution:** it misses Zen Float64 32–64 KiB, discards long in-place gains on Zen, and leaves the known Skylake-SP placement regressions. A corrected runtime byte cutoff with identity exemption would be a separate measured variant. |

If ordinary unadvised Zen arrays remain safely faster across the newly identified backing states, prefer universal A3 and report the stress-case regression rather than adding selection just because a cutoff is possible. If the new data reproduce material losses in plausible ordinary states and the byte cutoff removes them cheaply overall, prefer the repaired universal byte rule. Evaluate the processor-specific version only if preserving Intel gains materially outweighs its overhead/maintenance cost **and** its residual Intel placement regressions are acceptable. The current Intel medium means show that preserving those gains could be meaningful; they do not by themselves prove the processor-specific option is best.

“No regression seen in a finite ordinary sample” is not a global guarantee; equally, a deliberately constructed worst placement does not specify a workload frequency. The policy does not supply such frequencies or a strict no-regression requirement. A final winner therefore needs the queued evidence and an explicit tradeoff, not a purported theorem from trial maxima. Nothing in the evidence calls for architecture-dependent pointer rules or extensive dispatch machinery.

**What I did not check**

I did not contact either cluster, inspect or alter queued job environments, run long local performance campaigns, or obtain new Broadwell/Skylake-SP/Zen hardware measurements. I did not run full `Pkg.test`, exhaustive bit-pattern enumeration, actual GPU packages, all third-party array extensions, symmetric/block iterates, concurrent calls, or real cross-CPU migration of a package image. The non-host check uses a custom DenseArray; runtime CPU-name tests simulate selection while executing on Raptor Lake. I checked Julia 1.10.13 correctness/ambiguities on the independent grid, not its performance or package-image behavior. Old huge-page backing and per-round probe timings cannot be recovered from the supplied files. The numerical equivalence checks compare to M4; they do not independently establish M4's mathematical conventions for every exceptional parameter. Direct source dependencies and the supplied raw results were inspected; unrelated A3i/Mi experiments were included only insofar as their presence affects compilation/order in the existing matrix.
