# Historical selection and benchmark specification (#852)

This document was written before full timing began. The primary contrast is
`post850 / original`, never a comparison against old-machine published timings.
Exact Git metadata is in `research/versions.json`.

## Historical boundary

The last pre-agent mainline commit is **bd0740d0ebbebcc76fe208e4318fe6fc5451d34b**,
authored/committed 2020-08-13 (package 0.3.0). The next mainline change is
5ba264a5206f6afc28e2e51d97f470d945eb0eea, 2026-10-20 PT, PR #4.
The [PR #4](https://github.com/mlakolar/ProximalBase.jl/pull/4) body explicitly
says “Generated with [Claude Code]”; its API base SHA is bd0740d. The API
snapshot is retained in research/pr-4.json. Git first-parent chronology and the
six-year hiatus independently corroborate the boundary. Author identity alone
would not establish agent involvement. We cannot prove no earlier unrecorded
assistance existed; this is the last original implementation before the
explicitly documented agent-driven development sequence.

The other pinned sources are:

| Label | Full SHA | Date (UTC) | Role |
|---|---|---|---|
| original | bd0740d0ebbebcc76fe208e4318fe6fc5451d34b | 2020-08-13T19:44:24Z | primary baseline |
| historical05 | def81ec9d021c39ed1727cfeb9598357291dd60d | 2026-11-01T12:04:08Z | 0.5.x design baseline, already agent-era |
| post848 | eff21b6d7d33c622313122f46c6ee71693c101ee | 2026-11-11T04:42:52Z | full implementation, renamed blocks and owner redesign |
| post850 | 2b373c546e247781d9514a7000ef3e010343ca4c | 2026-11-11T06:57:16Z | minimal-core merge, fixed target |

Original source uses src/sparseIterate.jl for all three types. Sparse and
symmetric iterates have three full-capacity arrays: values, active IDs and
coordinate-to-position map. The original symmetric array exposes independent
lower-triangle linear indexing despite its matrix shape. Historical05 has
extensive agent changes, active-group AtomIterate, ordinary matrix indexing
for symmetric arrays, checked writers and additive coordinate API. Post848
uses BlockIterate and a non-array symmetric owner with coordinates(x).
Post850 extracts a smaller mathematical library; the iterate implementations
are preserved (docstrings differ). Source archives and source digests are
recorded by the runner; none is patched to make it load.

## Prior infrastructure examined

* Original: no benchmark directory; runtime depends only on Julia stdlibs.
* historical05: benchmark/run.jl and cases.jl are broad mathematical-function
  catalogs; no completed #420 baseline yet at this source.
* post848: full cases.jl/compare.py, deterministic scenario_rng, fixed-count
  sparse generators, independent references, prepare/evals=1, input mutation
  guards, calibration, locked-environment setup and #463 reproducer.
* post850: run.jl/probes.jl/differential.jl/summarize.py replace that catalog
  with 102 minimal-core probes. The old cases.jl, compare.py and baselines/
  remain in Git at post848, not in the active tree.

[Issue #420](https://github.com/mlakolar/ProximalBase.jl/issues/420) specifies
n=10^4/10^6 and s=100/1000/10000, block sizes 1/8/64 and 1%/10% activity.
Its pinned source def81ec9 deliberately precedes 0.6 FD implementations,
not all agent development. [Issue #463](https://github.com/mlakolar/ProximalBase.jl/issues/463)
recovered the generator/verifier and evidence; its tools reproduce 3,719 rows
byte-for-byte but original individual timing samples are unavailable. The
45-member archive has SHA256 ae0235adad9f932ceb95ea3ad1709621d73fa0a506dd231b9c442fd005b1bf87.
We reuse those tools unchanged for historical integrity verification only.
The historical compare.py caps repetition counts at 10 and assumes its catalog
and metadata, so it cannot directly analyze this 7 + >=20-round experiment.
The deterministic fixture/reference patterns, BenchmarkTools setup and
allocation recording are reused; a separate adapter/spec/driver is necessary
for original APIs, solver sequences, independent processes and larger rounds.

[Issue #849](https://github.com/mlakolar/ProximalBase.jl/issues/849),
[PR #850](https://github.com/mlakolar/ProximalBase.jl/pull/850) and
[performance.md](../minimal-core-849/performance.md) document the extraction
comparison, including code-placement/fixture effects and A/A controls. We
reuse BenchmarkTools 1.8.0 / PrecompileTools 1.4.0 (locked Manifest), the
one-evaluation prepare pattern and independent-process control design.
Removed penalty/prox/GLM/spectral catalog cases are obsolete after #850 and
outside this storage-only audit. Existing 102 probes remain suitable for
post848/post850 correctness; they cannot load original APIs unchanged.
No historical baseline or existing harness file is edited.

## Equivalent operations and adapters

Shared specification uses deterministic full input vectors and explicit ID
schedules. Sparse shapes are n=10,000/1,000,000, s=100/1,000/10,000 (s<=n),
both Float32/64. Symmetric matrix sizes p=144/1414 give 10,440/1,000,405
independent coordinates with the same active counts. IDs use independent
column-major lower-triangle enumeration; no package mapping helper builds the
reference. Public 2-index operations are compared directly; coordinate
operations adapt original x[k], historical05/post848/post850 coordinates(x)[k]. This adapter difference is explicit and
2-index workloads provide a syntax-independent cross-check.

Blocks use p=8192/65536, sizes 1/8/64, contiguous, strided and deterministically
permuted groups, and 1%/10% activity (at least one group). Original documented convenience
constructors support only equal contiguous Float64 blocks (and matrix row/
column blocks), without an active list. We do NOT fabricate historical active
metadata or arbitrary-layout constructors. Original noncontiguous and Float32
block cases are reported N/A. The newer three versions cover the full matrix.
For original contiguous Float64 blocks, reads/writes, sweep, copy, clear and
products compare values under an externally supplied group schedule. They
perform different bookkeeping by design: only newer versions maintain an
active set. Native activation/isactive/dropzeros and state-copy activity have
no original counterpart and are labeled incompatible. No timed conversions.

Sparse and symmetric cases cover empty/populated construction, first nonzero
activation, active writes, repeated read-modify-write, active sweep, sequential
and random reads, deliberate active zeros, copy/copyto, clear, dropzeros,
dense import (sparse), dot and preallocated matrix products. Separate scalar
reads/writes complement batched solver paths. Full clear retains active zeros
before compaction. Original symmetric full clear is an explicit independent-
coordinate loop because generic matrix fill has an inconsistent length
contract; newer owners use fill!(coordinates(x),0). This is a semantic adapter,
not an original native complexity claim. Copy operations must preserve state;
unsupported generic original copies are surfaced as failures, never repaired.

Complete workload sequences use a diagonal quadratic coordinate model with
soft thresholding: fixed-set CD, expanding CD, pruning/compaction/rebuilding,
contiguous/noncontiguous block CD, and repeated symmetric 2-index updates.
The dense reference executes the same arithmetic and tracks activity as a
separate set; tests verify values, active sets/counts, retained zeros, copy
independence and live block views. Every case is checked before warmup/timing
in every worker; Float32/64 bounds-checked runs precede measurement.

## Measurement and analysis planned

Linux / Julia 1.13.3 only; one Julia and BLAS thread, same startup/compiler/
bounds flags, immutable archived source loaded in Main.ProximalBase in separate
processes, common locked dependencies, same pinned CPU. Seven fresh-process
paired rounds include all four sources plus original A/A (and current A/A
for original-incompatible cases); alternate case/version order. Setup/restores,
independent checks and compilation are outside warmed timing. All mutations
use evals=1. Save individual sample times, GC times, allocation bytes/counts,
trial estimates, object summarysize, compilation/warmup duration and complete
machine/binary/source/harness provenance. Object memory is reachable Julia
heap, not RSS/native allocations.

Per case: median paired post850/original ratio and deterministic percentile
bootstrap 95% interval across process rounds, absolute times, allocation
bytes/counts, A/A ratio and dispersion. Attribution uses all four revisions.
No multiplicity-adjusted significance claims. Any primary median above 1.05
is followed up for at least 20 additional paired rounds, with varied fixture
placement and process controls; nanosecond cases remain timer-limited unless
batch/solver evidence supports a practical regression. Stable source costs
are investigated without production modifications. Old-machine baseline
numbers never enter these ratios. Full conclusions await actual measurements.

Pre-timing compatibility checks found that historical05 fill!(x,0) deletes the active set in all three families, unlike original/current retained-zero writes. Native clear is therefore N/A for that attribution revision; explicit zero_active and prune_rebuild sequences remain comparable. Full-active fixtures have no first-activation/expansion operation and report N/A. Original detection uses absence of ncoordinates, since historical05 retains a deprecated numCoordinates alias. The initial failed adapter checks are retained as harness-development evidence; no measurements from them enter conclusions.

## Supplemental questions identified during measurement review

These controls are specified before their timing, in a separate generated
harness; the 942-case primary timing code remains frozen.

* The original row-block constructor AtomIterate((K,b),K,true) represents the
  strided IDs k:K:n. A supplemental comparison uses that native matrix shape
  against current vector groups for value reads/writes, sweeps, copies, clears
  and block CD. There is no timed shape conversion. Native dot/matvec and active
  metadata operations are excluded from that shape-adapted comparison.
* Four explicit zero-write/compact/reconstruct cycles exercise complete active
  set rebuilding. Using coordinate zero writes instead of native fill gives
  identical intermediate active-zero and empty-set transitions in every source.
* Block state-copy cost is measured for both independently constructed equal
  layouts (primary copyto) and a destination built by similar(src) (supplemental
  copyto_shared). Current copyto checks group equality without registering its
  result in the identity cache, so independently owned equal layouts can require
  O(number of groups) work at every call. This is a real use-context distinction,
  not an implementation optimization in the audit.
* A separately generated full-field restoration control copies values, IDs and
  reverse maps in the same order for every scalar source, entirely outside
  timing. It tests native restore cache conditioning with a common full-array
  preparation context, including for large apparent improvements. The separate
  active-prefix control below tests a common hot working set.
* Original generic dot(a,x,b) traverses its matrix interface, but original source
  also provides the active-set helper A_mul_X_mul_B_rc(A,x,B,r,c). A supplemental
  bilinear_native case compares that helper to current dot(view(A,:,1),x,
  view(B,:,1)) using preconstructed symmetric A/B whose first columns are the
  same independent operand vectors. No timed construction/conversion is added.
  The generic API improvement must not be presented as a new storage-kernel
  capability that was absent from the original optimized helper.

The common-restoration control also raises the per-trial wall budget from 8 ms
to 1 s while retaining the 80-sample cap. BenchmarkTools' wall budget includes
untimed setup: a full O(q) historical restore can otherwise yield fewer timing
samples than a current O(s) restore. Raw sample counts remain explicit. The
longer control budget permits matching sample counts and tests that precision/
cache-conditioning asymmetry without changing timed package operations.
* Complete residual-maintaining lasso workloads supplement the diagonal model:
  four coordinate sweeps (fixed and expanding schedule) and four block proximal
  sweeps (contiguous, strided and arbitrary layouts), with 16-row deterministic
  dense designs. Timed loops update coefficients and the residual after each
  coordinate/block. Independent dense-vector/BLAS references and a final
  residual = target - design*coefficients identity check validate the sequence.
  Scratch, steps, design and state restoration are outside timing. The original
  strided row-block matrix uses native block views and linear coefficient IDs;
  no vector conversion is inserted. These results, rather than sums of separate
  microbenchmarks, support conclusions about that fuller synthetic workload.

Supplemental CI initially found 64 validation failures for original Float64
strided row blocks: collect(blockview) had shape 1×b while the independent
reference was a vector. The values and timed operation were equivalent; the
untimed validator now uses vec(collect(view)). No numerical tolerance or
production implementation was changed, and no supplemental timing preceded
this correction. The independent Sol/high reviewer identified the exact count
and shape mismatch.

Original AtomIterate's autogenerated raw-field constructor could also hold
Float32 or arbitrary SubArray groups. Those raw-field constructions are not
used in the primary campaign; the later raw-native variant below fills that
coverage separately. Primary N/A rows describe exclusions from the documented
convenience-API vector comparison, not proof of representational impossibility.
Native original strided matrix blocks are measured in the supplement; no
cross-layout speed superiority is inferred from primary N/A rows.

Allocation-only isolation includes Float32 n=10,000 empty construction and
Float64 n=1,000,000 empty construction/import, one selected operation per fresh
process. The initial 8-ms large-allocation trials sometimes retained only two
or three samples; their individual times and zero GC counters are preserved.
The isolation uses a 1-s budget with the same 80-sample cap to distinguish
fixture history and truncated transient samples from sustained source costs.

A complementary hot scalar control uses the same active-prefix field restore
for both original and current owners, entirely outside timing. It clears only
previously active reverse-map entries, copies the source active values/IDs and
rebuilds their positions, preserving active zeros. This avoids the native
old-full/current-active setup asymmetry while retaining hot working sets. The
full-field and active-prefix controls bracket cache-conditioning effects; they
are reported separately, never pooled into one universal kernel ratio.

The initial residual BCD context uses linear v[j] indexing for every view.
For original strided row-block matrix views, that can invoke generic linear-to-
Cartesian conversion. A corrected best-native control uses inlined typed
accessors: v[1,j] for original 1×b matrix views and v[j] for vector views, with
row shape asserted outside timing. There is no timed conversion. This control
runs 28 fresh-process paired rounds (seven plus 21 confirmation rounds) under
common state restoration and a 1-s budget. Initial linear-index measurements
remain separate and are never pooled with this corrected context. Because
indexing and restoration both differ, cross-context changes are not attributed
solely to indexing; source comparisons within each context remain explicit.

Common-full restoration's exploratory seven rounds exposed additional
symmetric matrix-product differences. Those context-specific >5% flags, and
any flags from the seven active-prefix rounds, receive 21 further fresh-process
rounds numbered 201–221. They use new seeds and are analyzed separately from
the exploratory seven. Native-restore confirmation alone cannot resolve a
change observed under common restoration. For the supplemental 28-round
campaign, the first seven and subsequent 21 are also reported separately as
exploration and confirmation; its combined interval is descriptive only.

## Raw-constructor block coverage, specified before timing

The independent review identified a remaining scope gap: original
AtomIterate(storage, atoms) can hold Float32 and arbitrarily ordered native
SubArray views even though convenience constructors cannot create them. A
separate raw-native block campaign prepares dense storage and its native views
outside timing, invokes that existing autogenerated constructor, and measures
unchanged native access, writes, sweeps, copy/copyto, clear, products and
residual-maintaining BCD. No package method or compatibility shim is added.
It includes both precisions and every primary block layout/size/activity/scaling
fixture. Original activation/dropzeros remain genuinely incompatible.

Construction is excluded from this variant: timing a field constructor with
prebuilt views against a modern full factory would be unfair, and a timed
harness factory would add historical construction logic. Convenience-constructor
measurements remain the construction evidence. Raw original views retain caller
layout descriptors while current construction owns/validates its partition;
those setup/ownership differences are explicit, and layouts remain unchanged.
The raw vector/strided views are a separate representation context from original
native matrix row blocks; they are not pooled. The variant uses common full
state restoration, 80 samples/1 s, fresh original/current/A/A processes, seven
exploratory rounds and 21 further rounds for every >5% original/current flag.

A further adapter inference control resolves atom versus blockview at adapter
*definition* time with @static and no new inline hint. The runtime-selection
adapter may prevent inference when the newer symbol is absent in historical05.
Separate allocation/typed-code diagnostics for all four sources test that
hypothesis. If it changes historical05 allocations, its affected view-operation
observations are retained but excluded from source attribution, and a corrected
seven-round historical05/post848/post850 block campaign is measured. Any new
post848/post850 >5% flags receive 21 further rounds. Original/current adapter
allocation parity is checked explicitly; any effect there stops export for
investigation. This changes Main benchmark adapters only, never package methods.

The static control also equalizes block copy preparation: native historical05
fill!(dest,0) deletes its active groups, while current fill retains them. Both
copyto and copyto_shared instead zero existing active block views without
changing the working set before timing. This is untimed preparation, not a
replacement copy kernel. All legacy historical05 block contrasts are
conservatively marked superseded in the exported CSVs and replaced for source
attribution by this control; raw observations remain unchanged. The CI
allocation/typed-code evidence already confirms the runtime-selection artifact
(345600→0 bytes, Any→concrete SubArray), with 0→0 bytes in the other sources.
