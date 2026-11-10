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
group identity using one reverse-map lookup. Unrecognized views use Base's
copy. The source restriction is expressed in dispatch, so a custom dense
source's own copy method remains applicable. It resolves overlap once before
any mutation, reads activity once, and copies through `_block_storage` using
an in-bounds loop with no entry activation calls or second alias preflight.
For an inactive block it first observes an infallible nonzero and activates
once before copying. Fallible sources/conversions keep scalar failure state.

The mathematical candidate specializes only `GroupNormL2`'s validated public
boundary for the measured small loop, with exactly the same buffer checks,
shared numerical kernel and return value. The generic public method, checker
and Scaled/Regularize forwarding retain their baseline bodies. No unchecked
API or competing formula is introduced.

Two broad annotation variants were rejected during measurement. Forcing every
public leaf inline changed 224 of 3,770 numerical fixtures by 1–3 ULPs in
regularized SparseGroupNormL2/MCP/SCAD calls. Keeping wrapper calls explicitly
noinline restored those bits, but ten paired repetitions then exposed allocations
and large slowdowns in nested wrappers. Neither variant is accepted. The
narrow candidate leaves those existing routes intact instead. See
[the discovery record](measurements/block-updates-787/wrapper-boundary.json).
