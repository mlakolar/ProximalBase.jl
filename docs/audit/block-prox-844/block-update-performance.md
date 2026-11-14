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

The active-only public writer (`26b6ec64`) improved the target but failed
independent confirmation on tiny/arbitrary and inactive writes. Moving the
shortcut into Base's copy helper (`7def2e65`) also caused ordinary fallback
regressions. Both are rejected. No special Base copy method is retained:
`src/block_iterate.jl` now matches the baseline byte for byte. Base handles
all block-view copies, including activation, aliases and checked failures.
The larger tests remain to guard those public invariants for future #787 work.

The remaining mathematical candidate is `424c5e9e`. It specializes direct
GroupNormL2 calls only for small builtin Float32/Float64 CPU vectors and compact
Array-backed views. Every call retains the existing buffer/parameter preflight
and shared numerical kernel. Larger calls invoke the original generic entry.
Builtin CPU wrapper leaves deliberately use that original checked entry;
custom functions and backend-owned buffers retain public dispatch. This avoids
slowing wrapped large views with the small-buffer specialization. No formula,
required validation or output-ownership policy is changed. No unchecked API
or custom view is introduced.

The generic checker, numerical kernels and all storage/constructor/view/norm/dot
bodies are unchanged. The source audit records the only changed production file
and verifies the original wrapper bodies after reversing the three forwarding
call names. Tests pass 48,801 assertions on each requested runtime, including
mixed zero-leading values, larger groups, aliasing and extension dispatch.

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
controls and PB-only quadratic-style loops. No repeatable ordinary-path
regression above 5% is accepted.
Conditional log-ratio intervals describe the sampled cohorts; A/A and separate
focused confirmation distinguish repeatable costs from compiler/run variation.

## Final mathematical candidate results

The full corrected matrix completed 4,080 cases and ten pairs on each runtime.
The Julia 1.10 matrix flagged five cases; an independent ten-pair confirmation
of nine selected cases flagged none. Julia 1.13 initially flagged eight cases;
its independent twelve-case confirmation flagged none. Initial flags include
unchanged storage/kernel controls and large wrapped calls: they remain in the
evidence rather than being discarded. These confirmations support the
incremental mathematical change, not a universal guarantee about other
compiler layouts or the original API-change performance gate.

| Float64 diagnostic | Julia | Baseline | Candidate | Paired geometric ratio | A/A ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Public math, five-element in-place compact view | 1.10.13 | 32 ns | 29 ns | 0.915 | 1.006 |
| Public math, five-element in-place compact view | 1.13.3 | 30 ns | 28 ns | 0.949 | 0.990 |
| PB-only quadratic-style loop, ten fixed passes | 1.10.13 | 6.768 µs | 5.931 µs | 0.877 | 1.003 |
| PB-only quadratic-style loop, ten fixed passes | 1.13.3 | 6.622 µs | 6.308 µs | 0.950 | 1.013 |

No candidate allocation increase appears in any recorded pair in either full
matrix or confirmation. Public block copying is unchanged; no bulk-write
speedup is shipped. [Compact results](measurements/math-only-results.csv)
include the selected diagnostics, every initial flag and every confirmation.
[The immutable evidence archive](https://github.com/mlakolar/ProximalBase.jl/tree/4d05b0d6338d869c769887a91771fca85fd76d6d/docs/audit/block-prox-844/measurements/block-updates-787/follow-up/math-only-evidence)
contains raw timings, summaries, source revisions, dependency manifests,
scripts and checksums. Its compressed measurements have SHA-256
`4cb5e7817ad98a50cc41350bcc478f6f4d8f655c6fdfc28b7b450f4d5cc44e0d`.

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
The final matched PB-only measurements above replace the earlier exploratory
6.760→5.917 µs result and its noisy Julia 1.13 comparison (A/A 1.106).
Formal downstream integration belongs to PS #238 / PB #527.

PB #380's historical large-group/scalar residuals and PB #703's cleanup and
step-policy questions remain separate. No KKT, zero-curvature, global-step,
symmetric-parameter or #426 private-layout protection changes are introduced.
Under PB #527, PB correctness/CI/review/performance govern PB merges;
downstream exact-pin integration governs the 0.6 release. This PR remains
draft and unmerged. The original bulk/whole-solver performance gate and
independent review remain pending; the mathematical improvement does not
resolve the bulk-write part of #787.
