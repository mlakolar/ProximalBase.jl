# PB-only block-update performance (#787 / #844)

## Sources and scope

The performance baseline is PR head `a9f53e6ef3febffd58496c4a1c4fcb7609d0ca1c`,
based on master `147d025211b681e53e5c987116609a8351a4391e`.
The rejected broad writer and its full history are preserved at
[the durable audit archive](https://github.com/mlakolar/ProximalBase.jl/tree/audit/pb848-block-updates-20261111).
[Checksums](measurements/archive.json) identify removed raw files; this archive
has no CI artifact expiry. Benchmark and reproduction scripts remain in PB.
No ProximalSolvers source, pin, tests, documentation, CI or PR is changed.

## Measurement correction

`fill!(x,0)` intentionally preserves active IDs. Earlier `inactive_nonzero`
probes therefore measured active blocks after warm-up. Their inactive-nonzero
timing claims are superseded, including earlier full grids and focused runs.
The earlier inactive-zero and active-zero regressions remain credible.

The corrected setup, outside each timed evaluation, clears values, calls
`dropzeros!`, activates group 2, then activates group 1 only for active cases.
Before timing, every writer scenario asserts initial values, activity and order
through three operation/reset rounds. A post-write check alone is insufficient.
The corrected driver is used unchanged for baseline, candidate and A/A workers.

## Before and after call paths

At the baseline, [`blockview` returns a SubArray](https://github.com/mlakolar/ProximalBase.jl/blob/a9f53e6ef3febffd58496c4a1c4fcb7609d0ca1c/src/block_iterate.jl#L220).
Base `copyto!` resolves aliasing and extent once, then writes through the
iterate's [scalar setindex!](https://github.com/mlakolar/ProximalBase.jl/blob/a9f53e6ef3febffd58496c4a1c4fcb7609d0ca1c/src/block_iterate.jl#L196). Every nonzero entry loads `group_of` and calls
[activate!](https://github.com/mlakolar/ProximalBase.jl/blob/a9f53e6ef3febffd58496c4a1c4fcb7609d0ca1c/src/block_iterate.jl#L287), which checks `position` even for an already-active group.
Compiler-elided entry bounds checks do not remove this activity bookkeeping;
optimized Julia 1.10 copy-loop output confirms the field loads remain.

The [narrowed proposal](https://github.com/mlakolar/ProximalBase.jl/blob/26b6ec6446d732771dabb8bede50e90cf49bf125/src/block_iterate.jl#L291) keeps SubArray and targets ordinary matching floating
vectors only. Small, partial and zero-leading sources return to Base before
a group lookup. Other sources, including Array-backed views and fallible
conversions, keep Base dispatch. Eligible views require constant-time exact
owned-index identity and an already-active group. They snapshot aliases once,
then use the existing `_block_storage` helper and Base's unaliased index-style
copy. The active list is untouched. Inactive groups always use Base value
writes. There is no custom inactive scan, activation or deferred raw write.

A first cutoff trial at 16 (`1c2205da`) retained inactive fallback overhead:
21 of 506 Julia 1.10 probes flagged above 5% over ten pairs, notably small
Array-backed sources and size-64 zero writes. The next trial uses vector
sources and a cutoff of 32; zero-leading inputs bypass the activity lookup.
This is a conservative optimization condition, not a changed value contract.
The final cutoff remains subject to the complete corrected gate.

The [mathematical entry](https://github.com/mlakolar/ProximalBase.jl/blob/26b6ec6446d732771dabb8bede50e90cf49bf125/src/block_prox.jl#L42) retains public group, indexing, real-type, shape, overlap and
parameter validation, then the existing shared numerical kernel. The small
GroupNormL2 specialization exposes that boundary for buffers up to 32 entries;
larger buffers invoke the generic method. A redundant outer `checkbounds`
is removed because both routes already check bounds in `_check_block_buffers`.
Generic and wrapper bodies, numerical formulas and output-ownership policy
remain unchanged. Nested wrappers still re-enter the validated public oracle.
Broad wrapper annotations were rejected for changed numerical bits or new
allocations; this work does not delete checks to obtain a timing result.

## Protocol and acceptance

Julia 1.10.13 and 1.13.3, Float32 and Float64, matched immutable PB sources
and dependencies, one Julia/BLAS thread, fixed coordinate order and CPU.
At least ten paired rounds use two fresh process cohorts, alternating order
and an independent baseline A/A worker. Warm-up and setup are excluded;
each timing sample has one evaluation. No competing local test/build work
runs during timing. Raw records retain medians, quartiles, allocation counts,
result hashes, exact driver hashes and process/source provenance.

The grid covers five penalty families; math sizes 1/5/8/32/256; plain, scaled,
regularized and nested wrappers; separate/in-place vectors and compact views;
writer sizes 1/5/8/16/32/64/128/256, activity/value states, ranges, strided and
reversed groups, ordinary/converted/checked sources; scalar writers, norm/dot
controls and PB-only quadratic-style loops. Full corrected acceptance is
pending. No repeatable ordinary-path regression above 5% is accepted.
Conditional log-ratio intervals describe the sampled cohorts; A/A and separate
focused confirmation distinguish repeatable costs from compiler/run variation.

## Historical evidence and remaining work

The original 48→69 ns figure is retained in the archive. Exact historical PB
sources in a PB-only adapter reproduced a smaller Julia 1.13 cost (47→53 ns);
this neither dismisses the original magnitude as noise nor attributes all
21 ns to activation. Component timings need not add to combined timings.

The historical PS whole-solver figures (31.17→32.93 µs on Julia 1.10 and
23.19→23.83 µs on Julia 1.13) remain separate evidence. Its quadratic path
uses compact in-place math and individual coefficient writes, not the public
bulk writer. The size-5 public/kernel probes and fixed-pass PB-only quadratic
loop diagnose that boundary; they cannot establish a PS whole-solve result.
The earlier 6.760→5.917 µs PB-only result is promising on Julia 1.10;
the Julia 1.13 A/A of 1.106 makes its apparent improvement inconclusive.
Formal downstream integration belongs to PS #238 / PB #527.

PB #380's historical large-group/scalar residuals and PB #703's cleanup and
step-policy questions remain separate. No KKT, zero-curvature, global-step,
symmetric-parameter or #426 private-layout protection changes are introduced.
Under PB #527, PB correctness/CI/review/performance govern PB merges;
downstream exact-pin integration governs the 0.6 release. This PR remains
draft and unmerged, with final-source performance and review gates pending.
