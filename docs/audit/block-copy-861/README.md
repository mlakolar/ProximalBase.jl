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

The paired seven-round campaign and independent 21-round confirmation are in
progress. This document does not yet make a final performance claim.
