# Historical iterate performance audit after #850 (#852)

Status: final controls and independent review in progress. The tables below
identify completed campaigns; the executive conclusion will be finalized after
those controls. This is a measurement-only change; no production implementation
or frozen historical baseline is modified.

## Executive findings

The changes have mixed effects. Sparse state copying and clearing avoid full
ambient-dimension passes and improve substantially at low activity. The new
block implementation adds activity tracking and more general layouts, with
material costs in several value-update and state-copy workloads. Fixed-active
scalar arithmetic is much closer to the original when state preparation is
matched than native-copy setup alone suggests. There is no justified single
speedup number across these uses.

These results concern warmed storage operations and complete synthetic solver
sequences. They do not measure ProximalSolvers convergence or complete solve
times. In particular, the diagonal-model BCD sequence emphasizes storage cost;
the residual-maintaining lasso sequence includes enough design/residual work
to test how much of that cost remains in a fuller workload.

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
historical activity list is introduced. Arbitrary original raw-field groups
remain unmeasured.

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
([log](research/package-tests.log)); this does not claim that obsolete historical
test environments were resolved and run unchanged.

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
are separate. Reachable heap is Base.summarysize of the fixture, not RSS,
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
