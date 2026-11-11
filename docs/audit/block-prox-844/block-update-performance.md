# PB-only block-update performance (#787, #844)

## Baseline and call paths

The starting PR #848 head is `a9f53e6ef3febffd58496c4a1c4fcb7609d0ca1c`,
with master base `147d025211b681e53e5c987116609a8351a4391e`.
The exact benchmark hashes, official Julia 1.10.13/1.13.3 archive hashes and
compiler/thread settings are in [baseline.json](measurements/block-updates-787/baseline.json).
No ProximalSolvers source, tests, documentation, CI, pin or PR is changed.

At that baseline, `blockview(x,k)` (`src/block_iterate.jl:220`) validates `k`
by indexing the owned group array and returns a `SubArray` whose parent is `x`.
For an ordinary vector source, `copyto!` selects Base's
`copyto!(::AbstractArray,::AbstractArray)` (Julia 1.13.3
`base/abstractarray.jl:1073`). Base checks aliasing once, snapshots an overlapping
source, checks its extent, and executes a scalar destination loop.
Each destination `setindex!` (`src/block_iterate.jl:196`) converts and writes the
value; each nonzero then reads `group_of[i]` and calls `activate!` (`:287`),
which reads `position[k]` even when the block is already active.
Bounds propagation can remove entry bounds checks in this loop. It does not
remove the data-dependent group/activation bookkeeping. The initial dispatch
and compiler output were inspected directly, rather than inferred from timing.

`SparseIterate` behaves differently (`src/sparse_iterate.jl:326–352`):
`_setstored!` converts once, reads its entry's slot, then writes an existing
slot directly or calls `_store!` for an absent nonzero. Its writable
`activevalues` accessor (`:316`) is a view of a plain value vector. The block
oracle's independent-output caller precondition already covers these views;
performance work does not change that ownership policy.

The mathematical entry point (`src/block_prox.jl:6–40`) checks the group ID,
one-based indexing, real element types, lengths, input/output overlap and
selected parameters, then calls the shared numerical kernel. Ordinary type
and indexing checks specialize away; lengths, group bounds and possible
aliasing remain runtime checks where not established by the caller's code.
The exact in-place case avoids the overlap test. `Scaled` and `Regularize`
(`:104–123`) re-enter the public oracle, repeating buffer preflight after the
outer call has already checked those same buffers and selected parameters.
This is a separate cost from the storage writer.

The pinned historical whole-solver quadratic path calls the oracle on an
in-place compact view, followed by individual coefficient/cache writes. It
does not call the bulk block-view copy. The PB-only `quadratic_like` probe
therefore sweeps 24 groups of 5 with that public math/individual-write boundary.
It diagnoses PB overhead; it is not a PS solve or proof that the historical
31.17→32.93 µs / 23.19→23.83 µs whole-solver costs are resolved.

## Check frequency and retained API

| Operation | Once per block call | Per entry |
|:--|:--|:--|
| Baseline ordinary block-view copy | source alias handling and extent | conversion/store; reverse lookup and activation check for each nonzero |
| Proposed owned active-block copy | exact owned-group recognition, extent, alias snapshot and activity | backing-value copy |
| Proposed owned inactive-block copy | same preflight; at most one activation after an infallible traversal | backing-value copy and zero reduction |
| Fallible or arbitrary-view fallback | Base alias and extent handling | established scalar conversion/activation behavior |
| Public mathematical call | group, shape, real type, indexing, overlap and selected parameters | shared numerical kernel only |
| Wrapper mathematical call | public preflight again at nested public boundaries | ridge transformation/shared kernel as applicable |

Julia can specialize away matching primitive element types and ordinary
one-based indexing. An `@inbounds` caller can also elide bounds checks; it does
not authorize skipping group activity bookkeeping or alias snapshots. Nested
wrapper preflight remains repeated in the accepted numerical policy: broad
annotation changes affected floating-point results or allocations, so they
were rejected rather than deleting checks.

Option A is used: the existing `SubArray`, parent, indices, lifetime and
norm/dot forwarding remain intact. Owned-index identity is a constant-time
test, not a whole-layout comparison. No custom view, BlockArrays dependency,
additional public writer or exported unchecked numerical API is introduced.

## Measurement boundaries

`benchmark/block_updates.jl` separates public preflight, the public numerical
call, its internal kernel, the public block writer, combined math/write,
individual sparse/grouped entry writes, arbitrary views, and public/raw
view norms and dots. The mathematical grid covers sizes 1, 5, 8, 32 and 256,
both floating types, all five group families, separate/in-place vectors and
compact views, scaling, ridge regularization and nested wrappers. Writer
probes distinguish already-active and initially inactive blocks, zero and
nonzero outputs, ranges, strided and reversed groups, ordinary buffers and
checked conversions.

Every timed sample has one evaluation. Input restoration and inactive-state
setup happen outside timing. Three operation warm-ups and the first complete
timing round are excluded. A function barrier gives every timed closure
concrete buffer types; an initial exploratory harness without this barrier
boxed vector/view unions and is excluded from acceptance evidence. In
particular, neither an artificial closure allocation nor a batched timing is
treated as a production allocation or as a reproduction of the historical
single-evaluation 48→69 ns comparison.

The original performance gate remains pending until the matched measurements
and final independent review establish the outcome. PB #380's other numerical
paths and PS #238 / PB #527 integration remain separately scoped.

## Candidate call paths and numerical boundaries

The writer retains `SubArray`. Its method requires a compatible owned group
index type and built-in same-type floating buffers, then verifies exact owned
group identity using one reverse-map lookup. Unrecognized views retain scalar
value writes. The source restriction is expressed in dispatch, so a custom dense
source's own copy method remains applicable. It resolves overlap once before
any mutation. An active block uses Base's unaliased index-style storage copy;
an inactive block uses one SIMD copy/nonzero reduction and then activates once
if needed. Reads and primitive same-type writes cannot throw in this restricted
path, and owned-group construction reserves activation capacity. No fallible
operation can intervene between a nonzero store and activation. Fallible
sources/conversions keep scalar failure state.

The mathematical candidate specializes only `GroupNormL2`'s validated public
boundary for buffers of at most 32 entries, with exactly the same buffer checks,
shared numerical kernel and return value. The generic public method, checker
and Scaled/Regularize forwarding retain their baseline bodies. Larger buffers
invoke the original public method to preserve Julia 1.10 code generation. No unchecked
API or competing formula is introduced.

Two broad annotation variants were rejected during measurement. Forcing every
public leaf inline changed 224 of 3,770 numerical fixtures by 1–3 ULPs in
regularized SparseGroupNormL2/MCP/SCAD calls. Keeping wrapper calls explicitly
noinline restored those bits, but ten paired repetitions then exposed allocations
and large slowdowns in nested wrappers. Neither variant is accepted. The
narrow candidate leaves those existing routes intact instead. See
[the discovery record](measurements/block-updates-787/wrapper-boundary.json).

The first whole-block writer used a separate nonzero scan followed by a copy.
Ten paired Julia 1.13.3 repetitions found repeatable inactive-zero slowdowns
(up to about 1.54× for size-256 views). That two-pass variant is rejected.
A zero-prefix variant removed the second traversal but still regressed on
large Julia 1.10 zero copies. The final candidate instead preserves Base's
active-copy loop and uses a SIMD copy/reduction for infallible inactive writes.


## Pending performance decision: bounded candidate

The immutable candidate `2fb4d9aab0e5db1f51217b9d03e5e158b4d90bb1`
passes 30,513 targeted assertions with bounds enabled on each requested Julia
version. Ten paired repetitions in two fresh process cohorts on Julia 1.10.13
confirm that the target and the zero-write paths have different outcomes:

| PB-only case | Baseline median | Candidate median | Paired geometric ratio | Conditional 95% interval | A/A ratio |
|:--|--:|--:|--:|:--|--:|
| Float32 GroupNormL2, size 32, combined math + public write | 69 ns | 48 ns | 0.700 | 0.689–0.711 | 1.005 |
| Float32 size 256, contiguous, inactive zero/vector | 127 ns | 219 ns | 1.727 | 1.716–1.737 | 0.991 |
| Float64 size 256, contiguous, inactive zero/vector | 126 ns | 219 ns | 1.741 | 1.733–1.749 | 0.996 |
| Float64 GroupNormL2, size 5, separate vectors, public math | 28 ns | 27 ns | 0.964 | 0.964–0.964 | 1.000 |
| Float64 GroupNormL2, size 5, exact in-place compact view, public math | 32 ns | 29 ns | 0.906 | 0.896–0.916 | 1.000 |
| Float64 fixed ten quadratic passes, 24 groups of 5 | 6.760 µs | 5.917 µs | 0.875 | 0.874–0.876 | 1.005 |

These intervals are conditional on two process cohorts; nanosecond quantization
can produce a degenerate interval. The raw sample quartiles and all case ratios
are retained. This is **not an accepted candidate**: eleven of the 52 focused
cases meet the repeatability criterion above 5%, including ordinary zero copies.
Optimization attempts stop here pending the maintainer's choice. Narrow
alternatives are an active-block-only fast path, retaining just the measured
small-block mathematical specialization, or keeping this measured proposal
for review. No regression is silently approved.

The full focused record (including warm-up, A/A and allocation data) is in
[pending/bounded-focused-110](measurements/block-updates-787/pending/bounded-focused-110/manifest.json).
The earlier zero-prefix candidate's full 3,774-case ten-pair grids on both
runtimes and its separate Julia 1.10 confirmation are archived under
`measurements/block-updates-787/rejected/zero-prefix-*`. Its Julia 1.13 run
had no repeatable flags, while 24 Julia 1.10 flags were confirmed separately.
Do not substitute those earlier grids for a final-candidate acceptance run.

The small-block diagnostic isolates a real public-call overhead: the baseline
size-5 Float64 internal kernel takes 26 ns for separate vectors and 27 ns for
an exact in-place compact view, versus 28 ns and 32 ns through the public call.
The candidate retains all validation and reduces the corresponding public
costs to 27 ns and 29 ns. The independent coefficient-write quadratic probe
improves without any bulk view copy. These measurements support the narrow
boundary optimization; they do not identify every cost in historical PS solves
or close PS #238 / PB #527.


The corresponding ten-pair Julia 1.13.3 focus run (55 cases, including separate
writer/math controls) gives a combined target of 53.5→35 ns, ratio 0.649
[0.639, 0.659], A/A 1.017. Its public writer alone is 46→25 ns, ratio 0.541
[0.512, 0.570], A/A 0.990; pure math at size 32 is 33→32 ns, ratio 0.989
[0.857, 1.140], A/A 1.005. This isolates a storage improvement without assuming
that independently compiled component timings add to the combined timing.
Four ordinary writer cases still flag above 5%, including Float32 size-256
inactive-zero strided views (127→144 ns, 1.134 [1.128, 1.140], A/A 1.007).
All 55 cases allocate zero bytes. Public size-4 norms remain within the
5% gate in both focused runs. The Julia 1.13 fixed-pass quadratic comparison
has an A/A ratio of 1.106; its apparent speedup is therefore inconclusive.
See [its raw record](measurements/block-updates-787/pending/bounded-focused-113/manifest.json).

## Historical 48→69 ns case

The historical PB-only adapter retains the original coefficients, two groups,
block sizes and warmed nonzero state, and compares immutable sources without
loading PS. Ten paired Julia 1.13.3 repetitions compare the exact original
`8eaf8f8` storage-aware operation with `8fb17445` pure-math/public-write code:
47→53 ns, paired ratio 1.116 [1.105, 1.128], A/A 1.006. Replacing the latter
by the starting PR head `a9f53e6e` gives 47→54.5 ns, 1.161 [1.138, 1.184],
A/A 1.008. On Julia 1.10.13, old versus the starting head is 55→54.5 ns,
0.983 [0.956, 1.011], A/A 0.995.

Thus the original direction of cost is reproducible on Julia 1.13, but the full
21 ns magnitude is not reproduced by this PB-only adapter. The original
four-repetition, three-reported-pair evidence stays intact. Differences in
compiler context and harness shape remain material: the current-head target
also measures 69 ns in the larger Julia 1.10 diagnostic. These are not grounds
to label the original result noise, attribute all 21 ns to activation, or waive
the historical gate. Direct writer measurements and the source audit establish
which redundant work the proposed writer removes; final performance acceptance
still fails on other ordinary paths.

## Remaining work and CI review

PB #787 now has an existing-API bulk-write proposal, storage/failure tests and
separate validated-math diagnostics. It remains open: the writer needs a
maintainer decision about narrowing or deferral, nested wrapper preflight is
retained, and formal downstream whole solves belong to PS #238 / PB #527.
PB #380's historical 1,024-entry Float64 residual and scalar-step investigation
are not closed by size-256 or size-4 controls. PB #703's zero-group-level and
infinite-level cleanup/step-policy questions remain separate; no global step,
KKT, symmetric-parameter or zero-curvature policy changes are made here.

The previous PR head's full package tests and conformance check passed in CI,
while its benchmark quick catalog failed with duplicate IDs. The storage rename
had collided with the preexisting `block_cases!` builder. The storage builder
is now `block_iterate_cases!`; the `blocks` and `block` families retain their
IDs and an explicit combined-family regression check. This fixes a catalog
bug without changing numerical or storage production code. Final-head CI,
documentation and independent review remain required; a known performance
failure is not a green acceptance result.
