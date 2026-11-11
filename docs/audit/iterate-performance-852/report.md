# Historical iterate performance audit after #850 (#852)

Measurements, numerical verification, independent review and extracted-artifact
replay are complete. No production implementation or frozen historical baseline
is modified. The benchmark-only PR is submitted for maintainer review.

## Executive findings

**The result is mixed.** Current iterates improve sparse state management,
pruning and several product APIs. Fixed-set scalar coordinate work is close to
the original under matched preparation. Block value updates and some in-place
copies remain slower, and symmetric products depend on density and dispatch.
There is no meaningful overall speedup factor across these operations.

* **Sparse/active-set methods:** at n=1,000,000/s=1,000, a matched hot pruning and
  reconstruction sequence is about **24% faster**. Native state copying and
  clearing avoid full-capacity passes and can save hundreds of microseconds
  or more at low activity. Storage capacity still scales with ambient dimension.
* **Coordinate descent:** the complete residual-maintaining fixed/expanding
  examples are about **1% slower**, practically close in this experiment.
  Apparent large native-setup scalar gains largely disappear with matched cache
  preparation. This is not a universal statistical equivalence claim.
* **Block coordinate descent:** a confirmed Float64 residual workload adds about
  **13%** for n65536/b8/819 contiguous groups; an arbitrary-layout example adds
  about **16%**. Storage-focused contiguous sweeps expose larger gaps: **5.2×**
  in Float64 and **8.8×** in a Float32 example. Modern writes maintain group
  activity, an additional feature absent from original AtomIterate.
* **Copying:** a representative allocating block copy is **61% faster** and
  allocates half the bytes. In-place replacement can be much slower: a Float32
  singleton strided-descriptor case is **0.534→26.726 µs** (about **51×**).
  Shared layouts remove equality work but do not remove all per-group cost.
* **Products:** default block owner APIs show about **9×** gains in a representative
  case. Against the original's existing dense backing-array BLAS strategy,
  representative dot/matrix gains are **1.7×/5×**, and some dots favor dense BLAS
  by up to about **2.4×**. Current backing storage can also use BLAS. Symmetric
  native bilinear comparison is about **36% faster**, dense-active generic
  bilinear about **2× slower**, and several matrix products **10–20% slower**.
* **Memory:** Float64 sparse n1m remains **24,000,152 reachable bytes** in both
  sources. Representative contiguous block owners use **54% more** at b8 and
  **91% more** at b64, while shared-layout copies can allocate fewer bytes.

The primary original is the 2020 pre-agent source, not the already agent-era
0.5.x benchmark revision. Corrected 0.5.x block examples are close to current;
source history and the measurements place the large cached-view/owner-write
tradeoff before that baseline. Post848/post850 iterate executable ASTs match,
and the corrected extraction flags do not repeat in their 21-round follow-up.

The audit contains 17 campaigns, 1,218 fresh measurement processes, 150,535
passing timing observations and 11,518,964 individual timing samples. All 560
exploratory flagged contrasts have 21 follow-up rounds. These counts include
A/A and multiple preparation/strategy contexts, not independent workload votes.
The old-adapter historical05 block observations are explicitly superseded;
all final source conclusions use the corrected controls. Constructor process
sensitivity remains inconclusive despite matching scalar allocation counts.

These are warmed storage and synthetic solver sequences, not ProximalSolvers
convergence or full solve times. Complete case tables, variability, A/A,
allocation estimates and explicit exclusions are in [the result index](results/README.md).

## Sources and prior evidence

The original is **bd0740d0ebbebcc76fe208e4318fe6fc5451d34b**, committed
2020-08-13T19:44:24Z. It is the base of PR #4, which explicitly identifies Claude
Code generation. The first-parent chronology and the six-year hiatus corroborate
this boundary. This identifies the last original before the documented agent
sequence; it cannot rule out unrecorded earlier assistance.

| Source | Full commit | Commit date UTC | Interpretation |
|---|---|---|---|
| original | bd0740d0ebbebcc76fe208e4318fe6fc5451d34b | 2020-08-13 19:44:24 | pre-agent reference, package 0.3.0 |
| historical05 | def81ec9d021c39ed1727cfeb9598357291dd60d | 2026-11-01 12:04:08 | already agent-era; original #420 design baseline |
| post848 | eff21b6d7d33c622313122f46c6ee71693c101ee | 2026-11-11 04:42:52 | full implementation after #848 |
| post850 | 2b373c546e247781d9514a7000ef3e010343ca4c | 2026-11-11 06:57:16 | fixed audit target after extraction |

The [historical specification](history-and-spec.md) records selection evidence,
issues [#420](https://github.com/mlakolar/ProximalBase.jl/issues/420),
[#463](https://github.com/mlakolar/ProximalBase.jl/issues/463),
[#849](https://github.com/mlakolar/ProximalBase.jl/issues/849),
[PR #850](https://github.com/mlakolar/ProximalBase.jl/pull/850), and the existing
[minimal-core audit](../minimal-core-849/performance.md). It was committed before
full timing; subsequent supplemental questions were specified before their own
measurements. [Exact metadata](research/versions.json) and API snapshots retain
the supporting history.

Original source has no benchmark infrastructure. Post848 contains the old
run.jl/cases.jl/compare.py catalog and #463 provenance tools; post850's active
runner uses its 102 minimal-core probes. Frozen catalog files remain available
at their historical commit. Removed prox/penalty/GLM/spectral cases are obsolete
for this audit. We reuse the deterministic fixture/reference patterns, locked
BenchmarkTools environment, evals=1 preparation design, and independent-process
controls. The old comparator assumes its catalog and at most ten repetitions;
it cannot analyze this cross-version, original-API experiment unchanged.
New tools live separately in benchmark/audit852/.

The unchanged #463 reproducer regenerates 3,719 historical rows byte for byte;
94 are timer-limited. Its 45-member provenance archive has SHA256
`ae0235adad9f932ceb95ea3ad1709621d73fa0a506dd231b9c442fd005b1bf87`.
[The reproduction log](research/historical-reproduction.log) is retained.
Those old-machine times never enter the ratios reported here.

The full src/ trees contain 5 files / 33,829 bytes, 48 / 1,036,075,
52 / 1,649,968 and 17 / 333,106, respectively. These are source-size counts,
not performance measurements. Post848/post850 iterate executable ASTs match
when only documentation macros and line-number nodes are removed
([digests and script](research/iterate-ast.txt)). This supports attributing kernel
redesigns to earlier work; it does not imply that code placement, compiler
context or heap layout cannot change observed timing.


## Comparable work and correctness

The primary specification has **942 case IDs** across Float32/Float64.
Sparse vectors use n=10,000/1,000,000 and s=100/1,000/10,000. Symmetric sizes
p=144/1414 give 10,440/1,000,405 independent lower-triangle coordinates and the
same active counts. Block storage uses n=8,192, sizes 1/8/64, contiguous,
strided and arbitrarily ordered groups, with 1%/10% target activity, plus
n=65,536/b=8 scaling. Integer group counts use floor with a minimum of one;
case IDs give the actual count.

Construction, first activation, active writes, repeated updates, active sweeps,
random/sequential access, import, copying, clear/active zeros, compaction, dot,
preallocated products, allocation estimates and reachable heap size are covered.
Complete sequences include four fixed/expanding CD sweeps, prune/rebuild,
four block sweeps, eight symmetric two-index sweeps, four complete
zero/compact/reconstruct cycles, and four residual-maintaining lasso CD/BCD
sweeps. The latter use deterministic 16-row dense normal designs, lambda=8,
fixed step schedules, and coefficient/residual updates after each coordinate
or block. They are full sequences, not sums of separately measured kernels.

Original symmetric linear indexing represents independent coordinates despite
its AbstractMatrix shape. Benchmark-only adapters use original x[k] versus
new coordinates(x)[k]. Independent lower-triangle enumeration defines the
reference without package mapping helpers. The repeated two-index workload
uses the same mathematical matrix syntax across all versions and checks the
owner redesign without relying on the linear-coordinate adapter.

Original AtomIterate has dense backing storage and cached views, without active
group metadata. Comparable block operations use the same external schedule
and values; modern implementations additionally maintain activity. Original
convenience APIs support contiguous Float64 vector blocks and native matrix
row/column blocks. The primary vector comparison excludes other original layouts
and Float32. Its raw autogenerated constructor could represent additional view
layouts/precision; these exclusions are API-scope decisions, not claims of
representational impossibility. Native row blocks represent k:K:n in a
supplemental strided comparison; no timed vector conversion or fabricated
historical activity list is introduced. The separate raw-native campaign also covers arbitrary groups and Float32
using that existing field constructor and native vector views. Views/storage
are prepared outside timing; factory construction is excluded from that
variant because prebuilt fields would undercount historical construction.

The original symmetric clear adapter loops over all independent coordinates;
current clear uses fill!(coordinates(x),0). Both preserve active zeros. This
compares equivalent complete writes, not an original optimized fill! primitive
or the fastest conceivable historical custom loop. Historical05 native fill
removes activity, so it is explicitly N/A for retained-active-zero clear;
explicit zero-write/compact/rebuild sequences remain comparable. Full-active
sparse fixtures have no inactive coordinate to activate or expand into.

Bounds-checked references pass **516 original, 872 historical05 and 938 each
post848/post850** primary cases. Remaining rows are explicit incompatibilities,
not omitted failures. References verify complete independent values, exact
active sets/counts, reverse-map consistency, retained zeros, absent-zero writes,
copy independence, live view aliases, symmetry, block ordering and state after
repeated updates. The original block activity checks are explicitly absent
because the feature did not exist. Each timed case validates twice before
warming, including after restoration.

Additional self-copy, deliberate-active-zero copy, disjoint destination state
replacement and four complete reconstruction checks pass. The 18 supported
current package test files pass with bounds checks
([clean standard Pkg.test log](research/package-tests-clean.log)); this does not claim that obsolete historical
test environments were resolved and run unchanged. An earlier local test run
passed its assertions but emitted obsolete extension-metadata load errors from
a reused development environment; [that log](research/package-tests.log) remains
retained. The final standard Pkg.test run uses the current declared environment
and passes without those errors; its [command and manifest](research/package-test-validation.json)
are retained separately.

Harness development failures are retained separately: deprecated-alias version
detection, a Set reference bug, and Core.Box callback captures were corrected
before final timing. The stopped preliminary stream with boxed callbacks is
excluded from all claims. Its exact unversioned harness was not fully archived;
final valid timing files are pinned by
[runtime-hashes.json](research/runtime-hashes.json). The supplement initially
had 64 original strided validation failures because collect(matrixview) was
1×b while the reference was a vector. Untimed vec(collect(view)) fixed the
shape comparison before supplemental timing; no numerical tolerance or source
kernel was changed. These were harness failures, not repaired source behavior.


## Machine and measurement protocol

Linux only, Julia **1.13.3**, host **usc-desktop**, Pop!_OS 24.04, kernel
7.2.15-76070215-generic, Intel Core i5-13600K. Workers run sequentially, pinned
to logical CPU 0 (SMT sibling 1), one Julia and one BLAS thread. The recorded
governor is powersave; no frequency lock or exclusive host reservation is
claimed. Per-process A/A, calibration, alternating order and repeated rounds
assess effects that affinity alone cannot remove.

Runtime flags are --startup-file=no --threads=1 -O2 --check-bounds=no;
correctness uses --check-bounds=yes. Each source loads unchanged as
Main.ProximalBase in its own process under the same locked harness dependencies:
BenchmarkTools 1.8.0, PrecompileTools 1.4.0, and the committed Manifest.
Historical full development dependencies are not resolved. Every campaign
retains OS/CPU, exact binary path/hash, source/harness hashes, Julia/BLAS
configuration, compiler flags, fixture seeds/pointers/padding and provenance.

Seven broad rounds each start six fresh processes: four revisions, originalAA
and post850AA. Role order and fixture traversal reverse between rounds.
Fixture seed is 852000 + 101*round + scenario index, shared across roles.
Held seeded padding perturbs allocation placement; pointers are recorded rather
than assumed equal. Three warmups and a preliminary timed-wrapper trial precede
recording. Every mutation uses evals=1. Setup, validation, state restoration,
design generation and scratch allocation occur outside timing unless construction
itself is the measured operation.

Primary trials request up to 80 samples / 8 ms. This is a cap, not a minimum;
BenchmarkTools' wall budget includes untimed setup. Historical full-state copy
can therefore both evict active data and reduce sample count relative to modern
active-prefix copy. Large allocation trials sometimes retained only two or three
samples; their samples and GC counters are preserved. Conclusions do not treat
these short trials as equally precise to an 80-sample control.

Every initial >5% median original/current or post848/current difference is
selected for 21 additional fresh-process rounds without a significance filter.
Separate allocation-only controls use one selected operation per process,
80 samples / 1 s and A/A. Common-full controls copy the three full scalar
capacity arrays in the same order outside timing; common-active controls clear
the old active map, copy values/IDs including zeros, and rebuild positions.
They bracket full-array and hot-working-set preparation. These contexts are
reported separately. The control budget is 1 s / 80 samples to avoid truncation
from expensive setup.

Supplemental residual BCD initially used linear v[j] on all views. Original
row-block views are IndexCartesian matrices, so the corrected best-native
control uses v[1,j] for those matrices and v[j] for vectors. Shapes are checked
outside timing; no timed conversion occurs. It runs 28 fresh rounds under
common restoration. The first seven and subsequent 21 are reported separately.
Because restoration and indexing both change across contexts, their cross-context
difference is not attributed solely to indexing.

New >5% scalar flags exposed by common preparation receive 21 further rounds
with new IDs/seeds (201–221), separately from exploratory 101–107. Confirmation
estimates stay visible rather than being silently pooled. Data selection,
compatibility, samples and exclusion reasons are all retained.


## Statistical interpretation

R = post850_time / original_time. R<1 favors current, R>1 favors original.
Each process contributes its trial median; the point estimate is the median
of paired round ratios. Absolute times are marginal medians of per-round
medians, so their quotient need not equal the paired ratio. Allocation bytes
and counts are BenchmarkTools trial estimates; individual timing/GC samples
are separate. They describe Julia-managed heap allocation; native-library
allocation and process RSS are not measured. Reachable heap is Base.summarysize of the fixture, not RSS,
reserved capacity outside reachable Julia objects or incremental copy ownership.

95% intervals use a seeded 10,000-resample percentile bootstrap of whole
paired rounds, never within-trial samples as independent experimental units.
They describe this sampled fixture/process distribution, are not adjusted for
multiple comparisons, and do not establish universal equivalence. All
comparisons retain ratio range/dispersion, p90, A/A and allocation estimates.
Cases with either marginal time below 100 ns are marked timer-limited; a tight
interval around a few nanoseconds is insufficient evidence of practical cost.
Complete sequences and absolute saved/added time guide the ranking.

A/A is essential: in isolated Float32 n=10,000 construction the originalAA /
original median ratio is about 0.271, despite original/current about 0.997.
Identical source varied by roughly 3.7× across process roles. Large-construction
and import intervals also remain broad in isolation. Their matching source
allocation counts and near-one A/B estimates do not prove universal equivalence,
and the early large ratios do not establish a source-kernel regression. The
exact heap/placement mechanism remains unresolved.

Source-load and first-use/warmup durations are saved separately. They include
execution, compilation, validation/fixture and measurement-wrapper effects in
different fields; they are not pure compilation timings. A separate diagnostic
records @timed compile counters, outside the runtime campaigns. Its limited
contexts cannot support a general startup or compiler-performance claim.


## Matched scalar and complete solver sequences

All rows below are Float64 and allocate zero Julia heap bytes. The scalar
operation is identical within its three preparation contexts. Native old copy
touches the whole capacity while current native copy touches active state;
common full/active preparation removes that asymmetry outside timing.

| Case / preparation | Rounds | Original µs | Current µs | R [95% CI] |
|---|---:|---:|---:|---|
| Four diagonal CD sweeps, native copy setup | 7 | 6.548 | 3.833 | 0.584 [0.555,0.594] |
| Same CD operation, common full arrays | 7 | 6.629 | 6.761 | 0.994 [0.974,1.062] |
| Same CD operation, common active prefix | 7 | 3.869 | 3.842 | 0.990 [0.986,1.042] |
| Eight symmetric two-index sweeps, common full | 7 | 46.711 | 47.076 | 1.000 [0.978,1.057] |
| Same symmetric operation, common active prefix | 7 | 27.981 | 28.393 | 1.004 [0.967,1.049] |

Fixed-set scalar arithmetic and symmetric owner updates are practically close
in these matched examples. Native preparation alone would suggest a large
scalar kernel gain; that attribution does not survive the controls.

The following are complete four-sweep residual-maintaining lasso sequences,
not sums of microbenchmarks. These are the independent 21-round confirmation
partition of the 28-round common-full/native-index control.

| Case / preparation | Rounds | Original µs | Current µs | R [95% CI] |
|---|---:|---:|---:|---|
| Residual CD fixed, n1m/s1000 | 21 | 128.129 | 129.476 | 1.011 [1.006,1.025] |
| Residual CD expanding schedule, n1m/s1000 | 21 | 146.807 | 148.673 | 1.012 [1.008,1.020] |
| Residual BCD contiguous, n8192/b8/102 groups | 21 | 29.686 | 31.883 | 1.078 [1.070,1.098] |
| Residual BCD contiguous, n65536/b8/819 groups | 21 | 273.497 | 310.422 | 1.134 [1.119,1.147] |
| Residual BCD native row/strided, n65536/b8/819 groups | 21 | 350.351 | 397.240 | 1.132 [1.128,1.142] |

For fixed residual CD, original/current same-source A/A ratios are 1.004/0.994;
for n65536 contiguous residual BCD they are 0.990/1.005. Full A/A intervals are
in the corresponding summary CSV. The roughly 1% scalar shift is small beside
the implementation costs seen in block sequences; it is not a universal
equivalence proof. The native-row strided comparison uses v[1,j] in original
1×b views and v[j] in current vector views, with no timed conversion.


## Raw-native Float32 and arbitrary block results

The existing original raw constructor supplies native vector SubArray views
for every precision/layout. Both owners execute the same value sequences under
an external schedule; only current maintains group activity. Construction is
excluded from this variant. These are 21 independent confirmation rounds after
the exploratory seven, with common-full preparation.

| Case / preparation | Rounds | Original µs | Current µs | R [95% CI] |
|---|---:|---:|---:|---|
| Float32 contiguous diagonal BCD, n8192/b64/12 groups | 21 | 0.317 | 2.775 | 8.751 [7.149,8.888] |
| Float64 arbitrary diagonal BCD, same n/b/count | 21 | 2.036 | 3.792 | 1.862 [1.857,1.875] |
| Float32 arbitrary residual BCD, n8192/b8/102 groups | 21 | 33.141 | 37.800 | 1.130 [1.115,1.148] |
| Float64 arbitrary residual BCD, n65536/b8/819 groups | 21 | 457.245 | 533.352 | 1.158 [1.134,1.183] |
| Float32 singleton strided descriptors, in-place copy n8192/819 groups | 21 | 0.534 | 26.726 | 50.847 [47.651,58.607] |

The singleton strided descriptor is k:K:n with one element. It does not imply
that a size-one block has noncontiguous payload; descriptor type and the copy
path still matter. Its roughly 51× ratio adds about 26 µs per call, rather than
representing a few-nanosecond difference. Arbitrary-layout and Float32 results
are separate fixtures; cross-precision differences are not attributed solely
to arithmetic width because the seeded typed operands differ.


## Matched-preparation active-set operations

Complete active-set pruning benefits beyond the faster native state copy.
For Float64 n=1,000,000/s=1,000, these seven-round controls give:

| Operation | Common preparation | Original µs | Post850 µs | Paired R [95% CI] |
|---|---|---:|---:|---|
| dropzeros after zeroing half the active entries | full arrays | 2.085 | 1.597 | 0.760 [0.716,0.785] |
| same compaction | active prefix | 1.291 | 0.771 | 0.601 [0.594,0.603] |
| zero half, compact, add up to 256 inactive coordinates | full arrays | 4.623 | 4.161 | 0.888 [0.865,0.990] |
| same complete pruning/reconstruction sequence | active prefix | 1.865 | 1.422 | 0.757 [0.733,0.766] |

All four rows allocate zero in both versions. Preparation changes absolute
costs and the size of the benefit, but the direction persists in both controls.
This supports a practical improvement in these active-set sequences, alongside
approximately unchanged fixed-set scalar arithmetic. The four-cycle complete
clear/compact/rebuild supplement is retained in its native-setup context;
its ratio is not treated as an isolated kernel improvement without a matching
whole-sequence preparation control.

The sparse-specific compaction implementation now keeps the arrays and active
count in local variables and discards trailing stored zeros before moving the
last nonzero into a hole. Original SparseIterate moves the last entry even when
it is zero, then examines that hole again. Both are active-set proportional;
the newer loop avoids unnecessary moves and reverse-map writes in this fixture.
The generic symmetric compaction path is separate, so this mechanism is not
attributed to every iterate family.


## Completed native-setup confirmation results

The following are **21 independent confirmation rounds**, separate from the
initial seven. Times are microseconds per complete operation/sequence, and
all listed non-construction cases allocate zero in both sources. These are
public value-equivalent storage operations; for blocks, only current maintains
group activity.

| Operation (Float64) | Original µs | Post850 µs | Paired R [95% CI] | Original A/A |
|---|---:|---:|---|---:|
| Contiguous copyto, n=8192, b=1, 819 active groups | 1.124 | 12.850 | 11.217 [9.097, 11.570] | 0.999 |
| Contiguous copyto, n=8192, b=1, 81 active groups | 1.116 | 5.543 | 4.973 [4.951, 4.982] | 1.000 |
| Four diagonal BCD sweeps, n=8192, b=64, 12 groups | 0.529 | 2.756 | 5.229 [5.175, 5.267] | 0.998 |
| Four diagonal BCD sweeps, n=8192, b=8, 102 groups | 0.965 | 3.723 | 3.782 [3.641, 3.829] | 0.993 |
| Four diagonal BCD sweeps, n=65536, b=8, 819 groups | 10.952 | 49.192 | 4.388 [4.335, 4.519] | 1.003 |
| Contiguous construction, n=8192, b=8 | 14.175 | 19.149 | 1.554 [1.358, 2.514] | 0.994 |
| Symmetric generic bilinear, p=144, 10000 active coordinates | 16.437 | 32.196 | 1.958 [1.950, 1.968] | 1.005 |
| Symmetric × 8-column matrix, p=144, 10000 active coordinates | 95.005 | 110.348 | 1.160 [1.147, 1.318] | 1.001 |

The b=8 construction row allocates 106,640→164,264 bytes and 6→16 objects;
constructor time is more variable than the zero-allocation sequences. The
b=64 diagonal workload adds about 2.23 µs per four sweeps, whereas n=65536/b=8
adds about 38.24 µs. Those absolute costs and the residual-maintaining workloads
matter more than the largest percentage on a single tiny write.

Current block copy first checks layout identity or ordered group equality.
With independently owned equal layouts, the equality result is deliberately
not remembered by copyto!: work can remain O(K) for K groups, or O(n) for
explicit arbitrary ID arrays. It then clears destination active groups,
activates source groups and copies each group. Original copyto! copies the
dense backing array in one pass. This explains why sparse activity does not
automatically make modern state copy cheaper. A similar(src) destination
shares layout and isolates the equality component; supplemental results keep
that use context separate. Shared-layout copy can be much faster for a few
large active blocks and still slower for many size-one groups.

Elementwise writes through current block views pass through the owner writer,
which checks/maintains group activation on each nonzero value. Original cached
views write the dense backing array directly. This is additional behavior,
not an unfairly inserted historical conversion. Generated-code diagnostics and
the fuller BCD sequences distinguish its cost from a percentage inflated by a
very small original kernel. The separately dumped actual `block_update!` kernel
(not just its timing closure) uses packed `ymm` multiply/add/max and stores
in the original Float64 specialization. The current specialization uses scalar
`vmulsd`/`vaddsd` and group/position checks. This supports reduced vectorization
as part of the observed owner-write cost. It does not isolate how much each
check or vectorization change contributes; no optimization was tested.
[Commands and status](research/kernel-diagnostics.json), inferred code, LLVM
and assembly are retained in the validation archive.


## Allocating copy and in-place state replacement

These are different operations and have different outcomes. In the raw-native
seven-round control, Float64 n=8192/b=1/819 active groups has allocating copy at
32.271→12.575 µs, R=0.386 [0.358,0.407]. Original and current A/A ratios are
1.012 and 1.011. Allocated bytes fall from 393,360 to 196,888, while allocation
count rises from 6 to 10. Current copies share the validated layout and its
coordinate-to-group map; original copies rebuild their cached views.

That gain coexists with the in-place copyto! regressions: current state
replacement can pay repeated layout equality and per-active-group work, while
original copies the dense backing array. Fresh-object copy, preallocated state
replacement, shared-layout buffers and independently owned layouts are therefore
reported as separate cases. Reachable owner size includes shared objects;
allocated copy bytes measure newly allocated Julia heap instead.


## Optimized symmetric products and density

These rows use independent 21-round confirmations and zero Julia allocations.
The native bilinear comparison uses original A_mul_X_mul_B_rc with prebuilt
symmetric operand columns versus current dot on the same strided column views.
Original already had an active-set-proportional helper; the generic API gain
is not a new capability that was absent from the original.

| Case / preparation | Rounds | Original µs | Current µs | R [95% CI] |
|---|---:|---:|---:|---|
| Optimized bilinear helper, p1414/s1000, common full | 21 | 7.804 | 4.987 | 0.641 [0.628,0.653] |
| Generic bilinear, p144/s10000, common full | 21 | 16.405 | 32.219 | 1.955 [1.938,1.971] |
| Eight-column product, p144/s1000, common full | 21 | 9.300 | 11.050 | 1.186 [1.172,1.209] |
| Same product, common active prefix | 21 | 9.419 | 11.198 | 1.205 [1.164,1.342] |

The owner redesign therefore has different outcomes for updates and products.
The dense-active generic bilinear route is about twice as slow here, while
the native-helper comparison favors current by about 36%. Eight-column product
slowdowns persist under both common preparation controls. No single symmetric
performance factor describes all these operations.

## Existing native BLAS as a historical strategy

The original exposes a dense backing Vector. Passing it to existing dot/mul!
BLAS methods adds no historical kernel and requires no conversion. This control
compares that original strategy with current sparse-aware owner methods; current
storage can also use dense BLAS. Inputs are finite, and all values/operands match
within each paired round. These are available-strategy comparisons, separate
from the default owner-API measurements above.

| Case | Rounds | Original backing BLAS µs | Current owner µs | Paired R [95% CI] |
|---|---:|---:|---:|---|
| Float64 dot, n8192/b8/102 contiguous groups | 7 | 0.775 | 0.465 | 0.597 [0.590,0.602] |
| Float64 16-row product, same layout | 7 | 8.434 | 1.640 | 0.194 [0.189,0.197] |
| Float32 dot, n8192/b1/819 arbitrary groups | 21 confirmation | 0.479 | 1.143 | 2.380 [1.930,2.455] |
| Float64 dot, n65536/b8/819 arbitrary groups | 21 confirmation | 5.465 | 11.195 | 2.065 [2.017,2.122] |
| Float32 dot, n8192/b8/102 strided groups | 21 confirmation | 0.467 | 0.556 | 1.190 [1.181,1.199] |

All rows allocate zero Julia heap bytes. Dense BLAS narrows the representative
dot advantage to about 1.7× and matrix-product advantage to about 5×, while
several many-group or arbitrary-layout dots favor the original dense strategy.
No blanket product speedup is justified from the default generic-dispatch
comparison. This also does not establish an intrinsic inability of current
storage to attain the dense strategy: its own backing Vector supports it.

Eleven initial >5% strategy contrasts received 21 additional rounds; ten retain
>5% median ratios. Confirmation estimates are kept separate. For example,
Float32 contiguous singleton dot changes from exploratory R=1.947 to confirmation
R=1.559 [1.540,1.566], demonstrating fixture/process sensitivity even where the
direction repeats. A Float64 contiguous singleton confirmation interval includes
one, so its median alone is not called a confirmed source regression. The
regression ledger retains every flag, A/A result and outcome. Ratios from these
separate strategy/preparation contexts are not pooled or subtracted to assign
an exact share of the default-API gain to a single mechanism.


## Complexity and memory

Scalar and symmetric copyto! used to copy all three full-capacity arrays.
Current same-shape state replacement clears old destination active positions
and copies the source active prefix: O(s_destination+s_source), while copying
public values, coordinate order, activity and active zeros. Unused capacity
bytes are not part of the compared mathematical state. At n=1,000,000/s=100,
Float64 sparse native copy is 1,610.869 µs versus 0.097 µs in the broad seven
rounds. The paired ratio and intervals are preserved without rounding to zero
in the result CSV. The 97 ns endpoint is timer-limited, so this is evidence of
a large complexity improvement rather than a precise universal 16,000× factor.
Symmetric state replacement has the same capacity-versus-active distinction.

Scalar clear changes from an ambient pass to a stored-value-prefix write.
Under common-full preparation, Float64 n=1,000,000/s=1,000 clear is
254.130→0.603 µs (exploratory seven rounds). The improvement persists when both
versions start with the same full-array cache preparation, unlike the apparent
native-setup speedup of many active-coordinate reads/updates. The original
symmetric clear number includes the explicitly documented complete-coordinate
adapter. Both clear operations retain deliberate active zeros until compaction.

Backing capacity remains ambient-sized. Float64 n=1,000,000 sparse storage is
**24,000,152 reachable bytes in both sources**; p=1414 symmetric storage is
24,009,880 bytes in both. Faster active-prefix operations did not make those
owners dynamically sparse in memory.

| Float64 contiguous n=8192 | Original reachable bytes | Post850 reachable bytes | Ratio |
|---|---:|---:|---:|
| b=1 (8192 groups) | 393312 | 393472 | 1.0004 |
| b=8 (1024 groups) | 106592 | 164096 | 1.5395 |
| b=64 (128 groups) | 70752 | 135424 | 1.9141 |

The block redesign stores a coordinate-to-group map, group activity and position
metadata in addition to values and layouts; original cached views have their
own per-group footprint. Consequently the memory tradeoff changes with block
size. summarysize follows shared layouts as reachable objects, so these totals
must not be interpreted as incremental allocation for each shared-layout copy.
Allocated bytes/counts for construction and copy are separate CSV columns.


## Historical attribution

These broad seven-round marginal times show why historical05 is a misleading
substitute for the original. They all come from this machine and Julia binary.
They are descriptive attribution points; only the stated revisions are timed.

| Case (Float64, native setup) | Original µs | Historical05 µs | Post848 µs | Post850 µs |
|---|---:|---:|---:|---:|
| Block four diagonal sweeps, n8192/b64/12 active groups | 0.533 | excluded: adapter artifact | 2.750 | 2.761 |
| Block copyto, n8192/b1/819 active groups | 1.134 | superseded: copy setup | 12.864 | 12.858 |
| Symmetric generic bilinear, p144/10000 active | 16.335 | 26.268 | 32.104 | 32.166 |
| Sparse copyto, n1m/100 active | 1610.869 | 0.095 | 0.095 | 0.097 |

Historical05's first block sequence recorded 248.911 µs and 345,600 bytes /
18,480 objects. **That observation is excluded from source attribution.** Its
Main adapter selected atom versus blockview with a runtime isdefined check;
the missing-symbol branch prevented inference in that revision. A static
selection of the native API removes all 345,600 allocated bytes in the
[independent CI diagnostic](research/ci-adapter-diagnostic.log). Original,
post848 and post850 record zero bytes before and after the same adapter control.
The primary original adapter directly accesses cached views; its comparison to
current is separate and unaffected. The copy setup also differed: native
historical05 fill deletes activity, while current fill retains active zeros.
Corrected copy preparation zeroes existing active block views in both versions,
preserving their working sets before timing. All legacy historical05 block
contrasts are conservatively marked superseded in the exported CSVs. Corrected
static-adapter examples appear below; raw observations remain available.
An agent-era baseline cannot silently be substituted for the pre-agent source.
This defect belongs to this new cross-version adapter. Frozen historical
baseline files and their reproducer remain unchanged.

The active-group design appears in source history at
7cee9dd439479ba8d94dff375810beaabd396461 (2026-10-22). That is source-history
localization, not a timing of that intermediate commit. Post848/post850
executable iterate ASTs are identical. Their paired results and A/A are kept
for every comparable case; small placement/context differences are not ascribed
to a new iterate algorithm introduced by #850.

The recorded full-directory harness digest varies during the long primary
campaign because analysis/support scripts were added while the seven timed
files remained frozen. It is not presented as one unchanged directory hash.
Those timed file hashes were saved before the final campaign and checked again
afterward; generated supplemental/control harnesses are preserved separately
with their exact files and digests. Report/statistical tools can evolve without
silently changing the already-recorded timed callbacks.

Block products also benefit from active-group kernels. For contiguous Float64
n=8192/b=8 with 102 active groups (816 active values), the broad seven rounds
measure dot at 3.234→0.373 µs, R=0.115 [0.114,0.117], and a 16-row dense-matrix
product at 14.943→1.636 µs, R=0.109 [0.108,0.111]. Both allocate zero; original
A/A is 1.000 and 1.001. The original generic AtomIterate products traverse dense
storage, whereas current products visit active groups. These are native-API,
matched-value product comparisons; they are not predictions of BCD sequence
speed, which also pays for coefficient writes and residual maintenance.

CSV work counts describe logical active updates/product contributions or problem
size. In particular, a copy's n/time is logical coordinate-equivalents per
second, not physical memory bandwidth: the modern implementation intentionally
does not write all n entries. Whole-operation throughput is 1e9/time_ns; solver
active-update throughput is work*1e9/time_ns. No FLOP/s or bandwidth claim is
inferred from those logical counts.


## Corrected 0.5.x attribution

The static-API/equal-copy-state control replaces the affected legacy block
comparisons. Its seven paired rounds give these Float64 examples (common full
preparation, 80-sample cap / 1 s):

| Operation | Historical05 µs | Post850 µs | Paired R [95% CI] | Bytes, both |
|---|---:|---:|---|---:|
| Four diagonal BCD sweeps, n8192/b64/12 active groups | 2.795 | 2.793 | 0.999 [0.992,1.004] | 0 |
| Four residual BCD sweeps, n8192/b8/102 active groups | 31.815 | 31.737 | 1.000 [0.993,1.010] | 0 |
| In-place copy, n8192/b1/819 active groups | 12.927 | 12.424 | 0.965 [0.953,1.000] | 0 |

These are separate corrected measurements, not replacements fabricated from
the old 248.911 µs observation. The local allocation diagnostic reproduces
345,600→0 bytes for historical05 and 0→0 in original/post848/post850. The large
legacy historical05 view-path gap was a harness inference defect. Corrected
examples instead put historical05 and current close together, while the
independently measured pre-agent cached-view path remains faster for these
coefficient-update sequences. This is consistent with the active-group design
cost already being present before the requested 0.5.x baseline; the exact
intermediate design commit is source-history evidence, not a timed revision.


## Potential follow-up work, ranked for investigation

1. **Bulk value writes to already-active blocks.** The largest coefficient-update
   gaps are in the contiguous storage-focused sweeps, with a smaller but material
   cost in residual-maintaining BCD. Current per-entry writers preserve group
   activity; original cached views write backing values directly. Investigate a
   public operation that validates the group once and preserves the working-set
   contract, then measure complete solver sequences. This aligns with the active
   group value-write part of [#787](https://github.com/mlakolar/ProximalBase.jl/issues/787).
   Its old prox_coordinate! validation proposal predates #850 and is not a
   recommendation to restore that removed API.
2. **State copying with many small groups.** Separate repeated equality checks
   for independently owned layouts from per-group clearing/activation/copy cost.
   The shared-layout similar(src) control measures the latter without claiming
   that layout validation explains the whole gap. Any change must preserve
   activity, active zeros, layout-cache semantics and ownership guarantees.
   Existing shared-layout buffer construction is a measured usage distinction.
3. **Symmetric matrix products and dense-active bilinear calls.** The stored-entry
   route has low-density advantages but pays more in several dense-active
   cases. Inspect mapping, zero handling, loop structure and dispatch before
   proposing changes. A density-dependent route must retain current numerical
   semantics; these finite-input timings do not justify dropping zero/nonfinite
   behavior or predictable validation.
4. **Block product strategy selection.** Default owner APIs and direct native
   BLAS are different available strategies. Use the backing-array control to
   identify density/group-size crossovers before treating a generic-dispatch
   gain as an intrinsic storage improvement. Current backing arrays can also
   use BLAS; no new API or implementation is proposed here.
5. **Block layout memory and construction.** For large blocks the additional
   coordinate-to-group map and activity structures materially increase reachable
   memory. Investigate sharing/reuse and actual solver buffer lifetimes before
   redesigning storage. Allocation-only scalar construction is not a justified
   optimization target from these data: same-source controls show unresolved
   process/heap context effects despite matching allocation counts.

These are opportunities supported by this audit, not predicted speedups. No
production code, compatibility shim, optimization issue or optimization PR is
introduced by this work.


## Limits of the conclusions

All absolute times and ratios come from this one Linux machine and Julia
1.13.3 build. Reproduction on another machine should compare paired source
revisions there, rather than divide its times by these published observations.
The host is not frequency-locked or exclusively reserved. A/A, calibration,
round reversal, seeds and allocation padding expose important variability but
cannot remove every heap/layout or process-order effect.

Reported sequences use finite deterministic data, fixed external schedules and
small synthetic designs. They establish operation/state performance, not whole
ProximalSolvers solve times, iteration counts, convergence or the proportion of
time each operation occupies in a user's model. Active group bookkeeping is an
additional modern feature, absent from original AtomIterate. Where behavior or
representation differs, the case/context and its exclusions remain explicit.

Seven-round intervals have limited tail resolution; confirmation intervals
apply to their own process/fixture context. Extremely small operations remain
timer-limited even when a percentage looks large. Allocation bytes/counts are
Julia heap estimates, and reachable object size is not process RSS or marginal
ownership cost. Constructor context sensitivity remains unresolved rather than
being labeled either a confirmed source regression or universal equivalence.



## Figures and complete data

* [Scalar preparation controls](figures/scalar-preparation.svg) show the same
  coordinate/dot kernels under native, common-full and common-active setup.
* [Residual BCD scaling](figures/residual-bcd.svg) shows block-size and activity
  effects from the independent 21-round confirmation partition.
* [Native bilinear and active clear scaling](figures/active-kernels.svg) separates
  the historical optimized helper from complete sparse clearing.
* Broad native-setup scaling: [sparse](results/broad7/sparse-scaling.svg),
  [symmetric](results/broad7/symmetric-scaling.svg),
  [blocks](results/broad7/block-scaling.svg). Interpret scalar native-setup gains
  with the matched-preparation controls, rather than as isolated kernel changes.

Standalone PNG versions accompany the SVGs. Figure input hashes are in
[figures/inputs.json](figures/inputs.json).
[All 14,994 paired comparison summaries](results/all-comparisons.csv) include
A/A and context labels; [the campaign index](results/campaigns.csv) gives raw
names, exact source-role/round sets, analysis harnesses and observation hashes.
[Adapter supersession](results/adapter-exclusions.csv) marks 1,449 legacy
historical05 block contrasts conservatively; original rows remain preserved.
Each campaign's incompatibility CSV records the raw reason and a scope
annotation distinguishing missing features from convenience-API exclusions.

[The 560-entry regression ledger](results/regression-ledger.csv) maps every
exploratory >5% contrast to 21 follow-up rounds and its investigation/context.
405 confirmation intervals are above 1.05 in their stated contexts, 32 retain
>5% medians without that interval support, and 123 do not repeat the initial
>5% median. These are descriptive contrast counts, including tiny operations,
A/A/context questions and strategy comparisons; they are not 405 independent
practical implementation regressions or a workload-weighted score.


## Reproduction and retained artifacts

Follow [the executable Linux recipe](../../../benchmark/audit852/README.md)
from a full Git clone, using Julia 1.13.3, the committed Project/Manifest and a
fresh output prefix. The recipe runs bounds-checked correctness first, seven
paired broad rounds, independent confirmations and each preparation/strategy
control sequentially. Choose an allowed CPU with `--cpu`; the published host
used 0. Install the optional pinned plotting requirements only for rendering.
Do not overlap compilation diagnostics, plots, packing or other heavy work with
the timing campaign. The expected duration depends on the machine and setup
costs; an interrupted campaign must not be silently reused as a complete one.

The four [checksummed archives](artifacts/README.md) retain individual time/GC
samples, observations, complete per-process provenance and logs, generated
harnesses, source archives, correctness runs, generated code and compilation
diagnostics. The separate excluded-development archive retains failed/stopped
harness work and contributes no observations to the final analysis. Git blobs
and SHA256SUMS pin the published artifacts; no external mutable storage is
required. [The distribution manifest](research/julia-distribution.json) hashes
all 5,407 installed Julia files, including shared libraries and the system image,
in addition to the executable hash recorded by every worker. The Julia binary
itself is not redistributed. Dependency versions are pinned by the harness
Manifest; source revisions and machine/OS/BLAS/compiler information are also
retained in process provenance.

To replay published statistics without Julia or new measurements, verify and
extract the three valid-data archives into one new directory and run
`replay_results.py` with that directory as `--raw-parent` and the committed
results directory as `--results`. The archive README gives exact commands.
The replay recomputes every exported metric cell and every incompatibility row
from individual samples using each campaign's retained analysis harness. It
excludes only explicit attribution/scope annotation columns from numeric
identity checks. The 28-round supplemental partitions are derived from their
same retained raw campaign, not extra independent observations. The result
index and full comparison tables preserve all contexts, exclusions and A/A.

## Self-review, independent review and validation

Self-review checked equivalent intermediate states, deliberate zeros and active
sets, layout ordering, ownership/aliasing, full versus shared-layout copies,
old matrix versus new vector indexing, fixture placement, compilation outside
timing, evals=1 restoration and complete role/round/seed coverage. It exposed
or resolved the cache-preparation asymmetry, historical adapter inference
artifact, retained-zero copy preparation mismatch and raw-constructor/native
strategy coverage gaps described above. Every final case passes the applicable
independent reference; unsupported features are explicit incompatibilities.

[Post-timing validation](research/post-timing-validation.json) passes tightened
Float64 references for primary, supplemental, raw-native and backing-BLAS
campaigns, nonuniform every-coordinate block layout checks for both precisions
and all four sources, generated-code/compilation diagnostics, statistical
negative checks and full raw-to-summary replay. [Integrity verification](results/integrity.json)
independently checks all 1,218 process records, unchanged source/binary identity,
compiler/thread/Linux constraints, calibration endpoints, all expected
round/role/case observations, individual-sample median/minimum/p90 and generated
harness digests. [Replay verification](results/replay.json) matches every
exported metric cell in all 17 campaigns. [Extracted-archive verification](research/archive-validation.json) reproduces
all 17 campaigns from the published archives; all four archive hashes and local
Markdown links pass. The extracted source/provenance/sample integrity pass also
checks all 1,218 processes again.

The [independent GPT-6.1-Sol high-effort review](research/independent-review.md)
found no unresolved substantive blocker in harness, numerical results,
statistical analysis or report. It independently matched all 122 frozen source
blobs to Git, recomputed five raw-sample/bootstrap examples exactly, verified
all 560 ledger entries and all 1,449 superseded historical block contrasts,
and checked the report figures. Its remaining publication checks were archive
packaging, hashes and links. No production change is included and the PR is
left for maintainer review, without merging.


The repository CI and dedicated Julia 1.13.3 audit-correctness workflow pass on
commit 4082afeb10f45cce5e637209a37229b5897257cf ([CI](https://github.com/mlakolar/ProximalBase.jl/actions/runs/43093908907),
[audit correctness](https://github.com/mlakolar/ProximalBase.jl/actions/runs/43093908826)).
The final publication commit is also checked on GitHub; its status is visible
on [PR #855](https://github.com/mlakolar/ProximalBase.jl/pull/855). The dedicated
workflow checks all four sources, supplemental/native/raw layouts and independent
references, active-zero/copy/reconstruction invariants, historical evidence and
statistical negative cases. The final local standard package test run passes all
18 supported test files with bounds checks on Julia 1.13.3.
