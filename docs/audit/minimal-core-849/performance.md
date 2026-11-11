# Independent numerical and performance evidence for #849

**Numerical checks pass; the performance evidence supports acceptance after attribution.** Two fixed-fixture inactive-write flags above 5% were investigated through source, dispatch, generated code, state validation and independently reconstructed public fixtures. The hot native kernel is identical, and the candidate/baseline median over 24 independent fixture sets is 1.000 for each layout; identical baseline/baseline controls vary more. No representative source-attributable slowdown above 5% or allocation regression remains established. No numerical formula or source optimization was changed. All flags and raw results are retained.

The immutable baseline is the full post-848 commit `eff21b6d7d33c622313122f46c6ee71693c101ee`, peeled from annotated tag object `5239e1043a370f999753710e958fb2742aa786ac`. Its source digest is `40b7a3a59db8820b6dcb00f31ae7f4dce09bead4e0992d21cee1743ef7465f94`. No ProximalSolvers code or remote references were changed by this investigation.

## Independent numerical checks

`differential.jl` passes **12,832 assertions**, 3,208 for each implementation and each of Float32/Float64, with bounds checks enabled. Both implementations are checked against independently constructed dense mathematics, rather than merely against each other. Coverage includes seeded scalar/coordinate and direct symmetric two-index writes, activation and retained-zero/dropzeros state, owner and dense copies, Float32 conversions, sparse/block/symmetric products, noncontiguous and reordered group views/writes, weighted NormL1 and Scaled block KKT, NormL1 scalar/full prox, dense/sparse/block and multiresponse least-squares values and gradients, and independent BigFloat references for centered SqrNormL2 ridge and Scaled L1 exceptional arithmetic. Tiny finite levels, huge finite levels and steps, overflowing γλ, large centers, underflowed squares and overflowed intermediate squares are exercised.

Two extraction closure omissions were found by this harness and repaired by the implementer before final timing: symmetric parameter tuple validation and the five-argument oracle preflight fallback. The final numerical run also includes the subsequently restored block KKT reducers. Regularize is deliberately absent from the minimal core; ridge references use the retained SqrNormL2 function.

## Controlled measurements

`run.jl` covers **102 operation cases** over Float32/Float64. All three iterate families have read, active-write, first-activation, full-sweep, copy, conversion, product and L1 value/prox cases. Additional cases cover direct symmetric two-index reads/writes and range, arbitrary and reversed block views and writes. Least-squares value/gradient cases cover sparse and block arguments.

The primary comparison uses separate baseline, candidate and independently compiled baseline-control modules in a single process. It uses Julia 1.13.3, one Julia thread, one BLAS thread, `--startup-file=no --check-bounds=no`, identical dependencies, warmed code, one evaluation per sample, up to 180 samples per measurement, restoration outside timing, alternating A/B/control order and alternating case order. Eleven repetitions yield **3,366 raw rows**. Each records median/minimum time and allocation bytes/count. Source, harness, dependency and Julia binary hashes and machine/compiler details are recorded in provenance. Bootstrap intervals describe the median of paired per-round B/A ratios; they are descriptive rather than a model for every source of machine variability.

The median primary B/A ratio across 102 cases is **1.000**. No case increases allocation bytes or count. Two primary median time increases exceed 5%, and both were investigated:

| Case | Primary 11-round B/A (95% interval) | Fresh normal-role 21 rounds | Source roles reversed, 21 rounds | Same-namespace separate-process confirmation, 31 rounds |
| --- | --- | --- | --- | --- |
| Float32 symmetric full sweep | 1.208 [1.205, 1.212] | 0.993 [0.989, 0.995] | old/candidate 1.000 [0.996, 1.005] | 1.031 [1.025, 1.034] |
| Float64 block L1 prox | 1.102 [0.997, 1.107] | 0.979 [0.924, 0.996] | old/candidate 0.994 [0.994, 0.998] | 0.990 [0.988, 0.993] |

The first sweep increase is stable within its original process, so it was not discarded as random sample noise. It does not survive fresh normal-role or reversed-role executions. Independently compiled copies of the same source also vary substantially in that case (the reversed-role candidate/candidate control has a median ratio 1.068 and reaches 1.193). A further confirmation loads each source into the same `Main.ProximalBase` and `Main.Suite` namespaces in three independent persistent processes, restores identical fixtures, and pairs 31 sequential repetitions with A/A controls. The sweep's candidate median is 860 ns versus 835 ns, an increase of 3.1%, with only 3 of 31 pairs exceeding 5%; block L1 prox is 863 ns versus 869 ns. Allocation counts and bytes agree throughout. These checks support compiler/module placement and process context as the source of the original large discrepancy, without establishing an intrinsic >5% extraction regression.

The confirmation is a source-loaded same-namespace package-module comparison; package startup, precompilation time, cold-cache behavior and solver-level performance are outside this warmed kernel evidence. The initial raw results remain intact so reviewers can see the original flag rather than only the successful repetitions.

## Exact final-source run, fixed-fixture flags and attribution

The final acceptance candidate is commit `e272df3b87dcacb982b64b0e2124ca75cc50a966`, tree `03f82e4837af3d8895d6f11636e15fa0270eeb28`. A fresh immutable `git archive` snapshot has source digest `5e173f646116a9b70d0a0dbc22eb3b209a92b253241febecb9aa69acaa5f7c18`, independently matching the root agent's digest. The same 102 cases × 11 repetitions × three roles produce another **3,366 raw rows** in `final-source-results/`; all earlier measurements remain unchanged. The independent numerical reference script rerun against this exact snapshot passes **12,832/12,832 checks** with bounds checks enabled (`final-source-differential.log`).

The exact-final full run has no allocation regressions and two bootstrap flags: Float64 reversed inactive block write (37→44 ns, B/A 1.189, interval [1.158,1.189]) and Float32 sparse L1 value (37→40 ns, B/A 1.081, interval [1.081,1.081]). Five other case medians exceed 5% with wider intervals or A/A noise. All seven cases were therefore rerun in three independent persistent processes using identical `Main.ProximalBase`/`Main.Suite` namespaces, identical compiler/bounds flags and dependency graph, and **one shared pinned CPU** to remove process/core scheduling differences. Thirty-one paired repetitions produce **651 raw rows** in `final-package-results/`.

| Exact-final case | Baseline → candidate median | Paired B/A median (95% interval) | Pairs >5% | A/A median | Allocation change |
| --- | --- | --- | --- | --- | --- |
| Float64 arbitrary-group inactive write | 37 → 43 ns | **1.162 [1.132,1.162]** | 26/31 | 1.027 | None |
| Float64 reversed-group inactive write | 37 → 42 ns | **1.135 [1.108,1.135]** | 27/31 | 1.027 | None |
| Float64 sparse dense import | 514 → 520 ns | 1.001 [0.998,1.016] | 9/31 | 0.996 | None |
| Float32 sparse L1 value | 37 → 37 ns | 1.000 [1.000,1.000] | 0/31 | 1.000 | None |
| Float32 block first activation | 112 → 110 ns | 0.965 [0.938,0.982] | 0/31 | 0.982 | None |
| Float32 sparse active writes | 524 → 505 ns | 0.963 [0.951,0.981] | 0/31 | 0.989 | None |
| Float64 range-group inactive write | 44 → 42 ns | 0.955 [0.933,0.955] | 0/31 | 0.932 | None |

The arbitrary and reversed inactive-write increases were treated as performance flags requiring attribution; their absolute sizes of 6 ns and 5 ns do not exempt them from the requested relative 5% gate. Source identity alone was not used to dismiss them.

Read-only diagnostics establish the following:

- The relevant scalar access/write methods, `blockview` and `activate!` are byte-identical (hashes in `final-source-attribution.txt`). Both sources dispatch to the same Julia runtime's `copyto!(AbstractArray,AbstractArray)` and generic `copyto_unaliased!`, with the corresponding unchanged owner `setindex!` and `activate!` methods.
- For each source/layout, 1,000 setup/operation checks verify that the target group is inactive, every storage value and position is zero before the timed operation, the source does not alias its destination, and the operation activates only group 1 and stores exactly the source values. Setup remains outside the timed region.
- The **hot `copyto_unaliased!` native kernel is identical** for both sources/layouts after normalizing only a cold-path absolute `movabs` relocation. All instructions, registers, branch offsets and stack offsets are preserved. The shared normalized SHA256 is `443351061c09d6080ae13fb71fd2da30ada1a95c52baa4114b2058b9bd581145`. The outer wrapper has relocation and temporary-stack-assignment differences, so whole-wrapper binary identity is not claimed. Raw LLVM/native dumps, diagnostics and the normalization script are included.
- The original fixed fixtures have different source/storage/index pointer alignments between processes. This is an observed context difference rather than an assumed explanation.

A further exclusive experiment therefore reconstructs **24 independent public fixture sets**, varies held allocation padding, retains all fixture sets to avoid simple allocator reuse, and uses the **original probe closures** from `probes.jl`. Each source runs in the same namespaces in an independent persistent process pinned to the same CPU. Seven paired rounds with baseline/baseline controls yield **1,008 raw rows**. All 24 fixture sets/layouts pass repeated inactive-state, value and non-aliasing assertions. Source, storage and index pointer offsets modulo 64 and 4096 are recorded, along with all timings and allocations.

| Layout | Candidate/baseline median over 24 independent fixtures | Fixture medians >5% | Fixture ratio range | Baseline/baseline median | Median baseline → candidate → baseline-control time |
| --- | --- | --- | --- | --- | --- |
| Arbitrary groups | **1.000** | 1/24 | 0.579–1.366 | **1.093** | 43 → 43 → 47 ns |
| Reversed groups | **1.000** | 0/24 | 0.977–1.023 | **1.093** | 43 → 43 → 47 ns |

The fixture-level bootstrap interval for each candidate/baseline median is [1.000,1.000] at this timer's nanosecond quantization; this does not imply exact equality below that resolution. The arbitrary-group identical-source control ranges from 0.627 to 2.122 across fixture sets. Storage offsets cover 12–19 distinct values modulo 4096, depending on process/layout; source offsets cover 19–21. Thus the candidate increase does not persist across independently reconstructed representative fixtures, while identical-source controls exhibit a larger systematic process-context difference. This supports clearing the source-regression gate on measured attribution evidence, rather than on the small absolute cost or a presumed source-code explanation. The experiment does not distinguish the exact contributions of heap/cache placement and instruction placement. Individual fixed contexts remain sensitive, and their raw flags are deliberately retained. No source repair or formula change was attempted by this investigator.

The original sweep and block-L1 flags did not survive their independent confirmations and are documented separately above; the exact-final inactive-write flags are resolved through their own generated-code and varied-fixture investigation above, not on the basis of those earlier outcomes.

## Source identity and artifacts

Source docstrings were being edited during the permitted documentation work; raw source digests therefore differ between timing stages even though no numerical kernel changed. The earlier separate-process confirmation uses an immutable local `candidate-snapshot` with source digest `a5f25867e676ea48047474fa845e069a7846dda7eacc25a135cbcd787c279eab`. The primary source digest is separately recorded as `7d9046c20813bce9d36815f44658e1176e01cce9917cfecc5c6635bcc94648a9`. The frozen confirmation source is included in the standalone evidence archive.

- `fixture-results/`: 24 independent public fixture sets × seven paired rounds, pointer alignments, per-fixture summaries and provenance.
- `fixture_server.jl`, `fixture_repeat.py`, `fixture_summarize.py`: varied-fixture reproduction and analysis.
- `attribution-baseline/`, `attribution-candidate/`, `attribution.jl`, `native-attribution.json`, `normalize_native.py`: dispatch/state/alias diagnostics, raw LLVM/native and normalized comparison.
- `final-source-identity.json`: exact final commit, tree and source digest.
- `final-source-results/`: exact-final full102case raw results, summary and provenance.
- `final-package-results/`: exact-final pinned-CPU seven-case confirmation.
- `final_package_server.jl`, `final_package_repeat.py`: pinned-CPU reproduction driver.
- `final-source-differential.log`: final mathematical reference run.
- `candidate-differential.log`: earlier mathematical reference run.
- `candidate-smoke.log`: 102-case baseline/candidate/control equality preflight.
- `final-results/`: complete primary raw CSV, summary CSV/JSON and provenance.
- `repeat-results/`: normal-role targeted fresh-process repetition.
- `swap-results/`: targeted repetition with candidate/baseline source roles reversed.
- `package-results/`: same-namespace independent-process repetition, including provenance and stderr logs.
- `run.jl`, `probes.jl`, `summarize.py`, `package_server.jl`, `package_repeat.py`: full reproduction/analysis harness.
- `Project.toml`, `Manifest.toml`: exact dependency versions for the harness.

`PB_BASELINE`, `PB_CANDIDATE`, `PB_OUTPUT`, `PB_ROUNDS` and `PB_FILTER` control the main driver. See README.md for the command. The independent-process confirmation runner has its source locations recorded explicitly in `package_repeat.py` and `package-results/provenance.json`; adjust those paths after unpacking. The immutable baseline can be restored with `git archive` from the recorded commit.

For dependency bootstrapping on a different machine, the stored Project/Manifest contains an unused historical ProximalBase path. The harness loads each compared implementation explicitly from its source path. Point that unused package entry to the restored baseline/current checkout before instantiation, preserving the other pinned versions, or create a BenchmarkTools/PrecompileTools/stdlib environment. Original manifests remain unchanged as measurement provenance.
