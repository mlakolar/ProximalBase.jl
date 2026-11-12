# BlockIterate in-place replacement (#861)

Investigation and implementation: [issue #861](https://github.com/mlakolar/ProximalBase.jl/issues/861),
[PR #863](https://github.com/mlakolar/ProximalBase.jl/pull/863).
The [reproduction driver](../../../benchmark/copy861/README.md) reuses the
historical audit #852 infrastructure. Ordinary entry/block value updates belong
to #860 and are unchanged here.

## Execution paths and contracts

| Operation | Baseline behavior | Proposed behavior |
| --- | --- | --- |
| Self `copyto!` / `copy!` | Immediate no-op | Unchanged |
| Length mismatch | `DimensionMismatch` before layout or writes | Unchanged |
| Shared layout or validated cache hit | Skip equality | Unchanged |
| Equal independently owned layouts | Compare ordered groups once per call, without updating either cache | Unchanged |
| Different compatible layouts | Base array value-copy fallback; existing activity is retained and nonzero writes activate destination groups | Unchanged |
| Equal layout, nonthrowing element conversion | Clear every destination-active block, activate each source group, copy between storage views | Clear destination-only groups; copy source-active values by validated indices; install IDs and reverse positions in source order |
| Equal layout, potentially throwing conversion | Convert every source-active block before writes; then clear/activate/copy from buffers | Unchanged |
| Allocating `copy(src)` | Copy full value/active/position arrays; share validated immutable-by-contract groups, group map and remembered layout | Unchanged |
| `similar(src, T)` | Allocate independent zero value/active/position arrays; share validated layout | Unchanged |

Other incoming sources stay on their existing routes. An ordinary
`AbstractArray` passes `_checkreal_broadcast` then Base's array `copyto!`, writing
values through `setindex!` without clearing activity. Its abstract element types
retain entry-by-entry conversion and possible partial-write semantics. Base's
generic `copy!` performs its length/axes handling before reaching that route.
`SparseIterate` ↔ `BlockIterate` checks real types and equal lengths, then uses
the same generic value-write fallback; its `copy!` explicitly delegates to
`copyto!`. Specialized CHOLMOD array methods only resolve dispatch and retain
the real-type check. Broadcasts retain their existing real/result checks and
staging rules. None of those routes gains the same-layout state-replacement
semantics or the new helper.

`_samelayout(dest, src)` calls `_layouthit` before ordered group equality. Hits
include identical group arrays, the same remembered non-null layout, and one
owner remembering the other's layout. Each cache entry was previously validated;
transitivity permits those identity checks. Copies must preserve penalty caches,
so `_samelayout` intentionally records nothing. `samegroups` can explicitly
cache a successful comparison. There is no repeated equality check inside one
replacement. The repeated cost is across calls on independent uncached layouts.
No layout-sharing mechanism or mutable public storage interface is added.

The fast path visits destination-active metadata, only destination-only active
values, and source-active metadata/values. It does not scan ambient storage or
all group positions. Existing uncached equality still depends on the layout's
size. Because supported constructors, `copy` and `similar` own value/activity
storage, direct indexed copies need no per-block alias analysis. Shared storage
constructed by modifying internal fields is outside the supported ownership
contract. Supported aliases are self-copy and views of an owner; no ordinary
view-copy method changes here.

The conversion predicate and preflight stay before mutation. A fallible
conversion runs once into buffers; the existing buffered branch remains intact.
Mixed Float32/Float64 copies use the nonthrowing path. Active zeros, including
signed zeros, are copied and their groups stay active. Destination-only groups
are zeroed and deactivated. Source activity ordering becomes destination
ordering, and every reverse position is rebuilt accordingly. Unused active-array
capacity is internal; the active prefix and position map define the state.

Clearing constructs `zero(T)` once per destination-only group, as the baseline
does. A review refinement hoisted this construction out of the inner loop to
preserve allocation and zero-object reuse for `BigFloat` and other real types.
The broad Float32/64 campaign uses the frozen pre-hoist snapshot. All twelve
`_copy_active!` native specializations (both source/destination precisions and
three descriptor types) are byte-identical before/after the hoist. All twelve
public `copyto!` listings have identical instructions, registers and labels
after replacing only assembler-declared global/GOT addresses with their symbol
names. Both snapshots and all generated listings are preserved as evidence.
The final source receives a separate exact-#852 focal repeat; broad timings are
not described as direct measurements of the final source. Independent review
accepted this distinction and code-equivalence evidence.

## Causal investigation

Final component diagnostics are recorded separately from headline paired
measurements. They use a fixed first-10%-active fixture; headline cases randomize
the active set and add destination-only groups. Component times are not an
additive accounting identity, and the benchmark-only direct prototype excludes
layout checks and conversion preflight.

For Float32 n=8192, b=1, 819 active groups, strided descriptors, the diagnostic
master medians were:

| Component | Time (µs) | Allocations / bytes |
| --- | ---: | ---: |
| Independent layout equality | 5.800 | 0 / 0 |
| Shared layout identity hit | 0.042 | 0 / 0 |
| Clear active destination blocks | 4.161 | 0 / 0 |
| Activate known source group set | 0.604 | 0 / 0 |
| Copy blocks between storage views | 14.919 | 0 / 0 |
| Activate and copy after clearing outside timing | 15.604 | 0 / 0 |
| Complete shared-layout replacement | 19.310 | 0 / 0 |
| Complete independent-layout replacement | 25.316 | 0 / 0 |

Layout equality explains part of the cost; it cannot explain the expensive
shared-layout path. Profiling and generated code identify repeated SubArray
construction/indexing, range length/division, axes and alias checks, copy and
fill machinery. Specialization is concrete and the in-place operations allocate
zero heap bytes. A view object need not allocate to impose substantial overhead.
Singleton strided descriptors still copy one physical value each.

This evidence favors bypassing intermediate views and repeated activation for
a complete validated state replacement. It does not justify a new layout cache
or changing ordinary update APIs. The final campaign was rerun after coordinating
with #860: CPU0 was reserved for #861; #860 used physical core 1 / logical CPU2.
Preliminary exploratory diagnostics and failed harness-development checks are
excluded from headline measurements. Shared cache/memory effects remain possible;
paired order rotation and independent A/A measurements provide controls.

## Validation and review

The bounds-enabled Linux/Julia 1.13.3 package suite passed all 19 declared test
files. Independent-reference tests cover Float32/64, contiguous/strided/arbitrary
layouts, equal descriptors with different representations, shared/independent/
cached ownership, deliberate active zeros, destination-only activity, empty
sources/destinations, self-copy, `copy!`, allocating copy, `similar`, cache
preservation, position maps, storage independence, subsequent writes and pruning.
Integer conversion failure after a successfully staged group leaves full value,
active and position arrays, active count and cache identity unchanged. Length
errors likewise leave the destination unchanged. Different layouts are compared
against an independent scalar-write reference, preserving their distinct contract.

All 300 benchmark cases pass independent reference checks twice in each of
original/master/proposed/master-control processes with bounds checking enabled.
The original has no group-activity metadata; those comparisons cover values only.
Independent GPT-6.1-Sol/high review found no production correctness issue. It
caught boxed callbacks and process-order imbalance before final timing; both
were fixed. Final evidence review and CI receipts are recorded with the results.

## Results

The main campaign contains **300 cases, 112 fresh processes, 33,600 trial rows
and 2,604,026 individual timing samples**: seven broad rounds followed by 21
independent confirmation rounds for every case. Linux/Julia 1.13.3, i5-13600K,
CPU0, one Julia/BLAS thread, `-O2 --check-bounds=no`. Original is
`bd0740d0ebbebcc76fe208e4318fe6fc5451d34b`; master is
`eb74a914d779371a02e4a6637eaa724aeb66fdf5`; the timed candidate source is
`9b12babc69d9939b21d38f564db132c23429dbe9`. Final source is
`933d104d853ff6c00534b75789cb2d9359cf532a` (the zero hoist described above).

Main active destinations contain both source-active groups and the same number
of destination-only groups. The following are **21-round confirmation**
medians in µs. `R` is the median of paired candidate/master ratios; intervals
are seeded 10,000-resample 95% bootstrap intervals over rounds. Separate source
medians need not divide to the paired median ratio, especially for allocations.

| Case | Original | Master | Timed candidate | R [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| F32 n8192/b1 strided, shared | 0.522 | 24.957 | 3.630 | 0.146 [0.143, 0.146] |
| F32 n8192/b1 strided, independent equal | 0.521 | 31.034 | 9.404 | 0.304 [0.301, 0.307] |
| F64 n8192/b1 contiguous, shared | 0.922 | 9.243 | 3.867 | 0.412 [0.398, 0.422] |
| F64 n8192/b1 contiguous, independent equal | 0.931 | 14.176 | 8.681 | 0.614 [0.595, 0.616] |
| F32 n8192/b8 contiguous, shared | 0.459 | 0.954 | 0.418 | 0.443 [0.438, 0.454] |
| F64 n8192/b8 arbitrary, shared | 0.924 | 3.462 | 1.803 | 0.521 [0.516, 0.531] |
| F32 n8192/b64 contiguous, shared | 0.461 | 0.193 | 0.113 | 0.594 [0.584, 0.598] |
| F64 n8192/b64 arbitrary, shared | 0.922 | 2.051 | 1.298 | 0.632 [0.627, 0.640] |
| F64 n65536/b64 strided, shared | 7.554 | 44.845 | 27.121 | 0.605 [0.603, 0.610] |
| F32 b1 strided, four shared-layout restarts | 9.329 | 338.585 | 50.143 | 0.146 [0.146, 0.148] |
| F32 b1 strided, four independent-layout restarts | 9.364 | 433.978 | 142.248 | 0.329 [0.322, 0.331] |

Each restart sequence includes sixteen complete replacements and four active-set
update passes. Its independent final reference is baseline values plus one
increment, since each cycle restarts from the same baseline. These sequences
measure state-management overhead; whole-solver convergence and total solve
time remain unmeasured. Every in-place case, including mixed Float32/Float64 and
restarts, remains **0 bytes / 0 allocations** in every round.

**Performance gate:** the only initial seven-round >5% flag was the F64 n8192/b64
arbitrary different-layout fallback, R=1.116. Confirmation gives
**R=0.982 [0.954, 1.009]**, A/A=0.989; it does not repeat. Across all 300
confirmation cases the largest median ratio is **1.044**, with **zero cases
above 1.05**. The combined 28-round maximum is 1.027. Allocation bytes/counts
match master in every paired trial, including allocating copies.

The 30 allocating-copy confirmation ratios range 0.967–1.044. The representative
F64 n8192/b1 contiguous copy is 32.128 µs original, 12.295 µs master and 12.601 µs
candidate, paired R=0.999 [0.989, 1.053]. Against original its paired ratio is
0.386 [0.380, 0.417]. Bytes/counts are **393,360 / 6** original and
**196,888 / 10** for both master and candidate. The allocating-copy implementation
and byte reduction are preserved, with the broad benefit over original retained.
The additional final-source context controls below allow a small timing cost.

The 18 different-layout confirmation ratios range 0.977–1.033; self-copy ratios
are exactly 1.000 (timer-limited). A/A controls in the combined campaign have
median 1.001, with three of 300 cases outside ±5% (range 0.950–1.055). These
controls and independent confirmation support the large same-layout reductions;
they do not establish precise sub-percent improvements for unchanged operations.

There are still important gaps against the original raw full-vector copy. In
the combined campaign an independent arbitrary singleton layout reaches 38.4×
original, dominated by retained validation plus activity/index work. The median
shared-active candidate/original ratio across cases is 2.31. Original stores no
active-group state and performs no layout validation; its value-copy comparison
cannot validate modern activity metadata. The change substantially improves
current replacement and retains validation and sparse work. Original raw-copy
speed remains unrecovered in some cases.

The broad campaign had #860 work on CPU2 and two short code-generation checks
on CPU4; CPU0 stayed reserved for #861. The final-source focal repeat runs on
an entirely quiet host. Its exact #852 preparation and results are reported
separately below.

## Conversion preflight diagnostics

Single-process diagnostics additionally measure Float64→Int conversion buffers,
complete staged replacement and a conversion failure near the end of staging.
For b1 strided descriptors, 819 active groups, the buffer alone takes 11.536 µs
on master and 11.308 µs on final source, allocating **59,080 bytes / 1,641** in
both. Complete staged copies take 23.912/23.714 µs with the same allocation
figures. Failing copies allocate **59,160 bytes / 1,644** in both and leave full
destination storage, activity, positions, count and cache unchanged. All 27
diagnostic allocation pairs match. These component timings are diagnostic;
the throwing-conversion branch has not been optimized or bypassed.

## Final-source exact #852 repeat

The quiet-host seven-plus-21-round focal campaign contains **eight cases, 112
fresh processes, 896 trial rows and 71,602 samples**. It uses final source and
the unchanged #852 copy/copyto fixtures and original raw native-vector
constructor. These are independently owned equal layouts; `fill!(dest,0)`
retains the destination's source-active groups, with no additional groups.

| Exact in-place case | Original µs | Master µs | Final µs | Paired R [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| F32 strided, 81 groups | 0.452 | 7.848 | 6.071 | 0.777 [0.761, 0.781] |
| F32 strided, 819 groups | 0.456 | 26.675 | 8.152 | 0.307 [0.306, 0.309] |
| F64 contiguous, 81 groups | 1.114 | 5.537 | 5.071 | 0.916 [0.914, 0.918] |
| F64 contiguous, 819 groups | 1.111 | 12.773 | 7.133 | 0.562 [0.533, 0.577] |

All are 0 bytes / 0 allocations. The targeted F32 expensive replacement is
**3.26× faster than current master**, while retaining the ordered layout check
and active metadata. It remains slower than the original raw copy. This
reproduces the expensive current regime behind the historical ~51× comparison;
the present original-reference time differs from the earlier audit's 0.534 µs.

Allocating copies have identical bytes/counts to master in every focal trial.
Their absolute times differ substantially from the broad live-allocation
context: F64/819-groups original/master/final medians are 99.109/42.253/47.729 µs,
paired R=1.011 [0.995, 1.046], compared with the separately reported broad
32.128/12.295/12.601 µs. Contexts are retained separately.

One allocating result crosses the initial 5% screen: F32/81-groups copy has
seven-round R=1.021 [0.930, 1.064], confirmation R=1.056 [0.996, 1.075], and
combined R=1.029 [0.994, 1.070]. Its confirmation interval includes parity and
the flag does not repeat in the initial seven or combined 28 rounds; the
primary 300-case confirmation had no allocating-copy flag. Independent review
requested an additional **24-round copy-only control**, with baseline/candidate
and independent A/A processes for both, at 81 and 819 groups. This preserves
the flagged data and removes the preceding timed replacement from the copy
trial context.

The copy-only control completed **24 balanced rounds, 96 fresh processes and
192 trials**. At 81 groups its paired ratio is **1.037 [1.010, 1.067]**;
baseline A/A is 0.990 and candidate A/A is 1.001. At 819 groups the ratio is
**1.009 [0.986, 1.021]**, baseline A/A=0.993 and candidate A/A=1.007. Both cases
remain **164,120 bytes / 10 allocations** in every source/round. The >5% flag
does not repeat in this control. A small context-dependent timing difference
remains possible; these results do not claim exact allocating-copy parity or
exclude every slowdown above 5% within the intervals. The predeclared repeated
median threshold passes, and the unchanged allocating-copy source/allocations
retain the primary campaign's measured improvements over original. Across all
four cross-source pairings the 81-group medians range 1.011–1.037; pooled
per-round geometric ratios are 1.029 at 81 groups and 1.020 at 819 groups. The
evidence supports a small contextual cost of roughly 1–4%, within the declared
gate, and does not establish exact parity.

## Evidence, replay and integration

All 320 measurement processes, 34,688 trial rows and their individual samples
are preserved in `evidence.tar.gz`, with source trees, both measurement-harness
snapshots and the copy-control harness, manifests, complete Julia distribution
hashes, component diagnostics, generated code, validation logs and development
failures. `results/` contains the seven/21/combined summaries separately for the
main and exact campaigns, plus the 24-round copy-only control. No unfavorable
observations or context variants are removed. Source/harness hashes are checked
against every process's recorded provenance before archiving.

From this directory, verify and extract:

```sh
sha256sum -c SHA256SUMS
mkdir -p /tmp/copy861-replay
tar -xzf evidence.tar.gz -C /tmp/copy861-replay
```

From the repository root, replay the raw-to-summary checks:

```sh
python3 benchmark/copy861/analyze.py /tmp/copy861-replay/pb861-evidence/data/broad /tmp/copy861-replay/pb861-evidence/data/confirm --output /tmp/copy861-replay-main
python3 benchmark/copy861/analyze.py /tmp/copy861-replay/pb861-evidence/data/exact7 /tmp/copy861-replay/pb861-evidence/data/exact21 --output /tmp/copy861-replay-exact
python3 benchmark/copy861/analyze.py /tmp/copy861-replay/pb861-evidence/data/copy-control24 --output /tmp/copy861-replay-copy
```

The analyzer joins observations to individually verified fixture/sample rows,
checks seeds and complete configured round×role×case coverage, verifies rotation
order, source aliases, runtime/thread/bounds/optimization settings and stable
per-campaign harness digests, then recomputes paired ratios and intervals. The
source snapshots are independently matched to those recorded digests. The
updated analyzer strengthens verification without changing timed operations.

The refined source passed [Linux/current-Julia CI](https://github.com/mlakolar/ProximalBase.jl/actions/runs/43129678930),
including all 19 package files with bounds checks, frozen benchmark references,
strict documentation/doctests, isolated examples, HTML checks and both required
gates. Final-head checks and the independent GPT-6.1-Sol/high review receipt are
recorded on [PR #863](https://github.com/mlakolar/ProximalBase.jl/pull/863).
The PR is not merged. #860 uses an isolated branch/worktree; revalidation against
its source changes is required if it lands first, before merging this PR.
