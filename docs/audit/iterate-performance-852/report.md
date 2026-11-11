# Historical iterate performance audit after #850 (#852)

Draft during execution. No final timing conclusions yet.

## Scope and historical interpretation

This audit compares fixed post-#850 source with the original source before the
explicitly documented agent-development sequence. It measures storage and
synthetic solver sequences, not ProximalSolvers convergence or full solve times.
No production implementation is changed or optimized.

The primary original is bd0740d0ebbebcc76fe208e4318fe6fc5451d34b, the August
2020 parent of PR #4. PR #4 explicitly identifies Claude Code generation. The
0.5.x source def81ec9d021c39ed1727cfeb9598357291dd60d is already agent-era;
using it as the original would miss the substantial 2026 redesigns. All four
full hashes, dates, version-selection evidence, infrastructure reuse and the
pre-timing operation specification are in [history-and-spec.md](history-and-spec.md).

The full-source sizes are 5 files / 33,829 bytes (original), 48 / 1,036,075
(historical05), 52 / 1,649,968 (post848) and 17 / 333,106 (post850). These are
source-size counts, not runtime measurements. The two iterate source files
have identical executable ASTs between post848 and post850 after removing
only doc macros and line-number nodes; [AST evidence](research/iterate-ast.txt)
records matching digests. Source identity alone does not rule out compiler,
module-context or heap-layout timing differences.

## Compatibility and correctness

The shared matrix has 942 case IDs. Bounds-checked independent references
pass 516 original, 872 historical05 and 938 each post848/post850 cases. The
remaining 426/70/4/4 rows are explicit incompatibilities. The original grouped
storage supports only contiguous Float64 vector blocks through public
constructors and has no active-group list. Its comparable block results are
**value-equivalent operations under an external group schedule**; newer types
also maintain activity metadata. No synthetic historical group bookkeeping or
arbitrary-layout conversion is inserted into timing.

historical05 fill!(x,0) removes activity; original/current writes retain active
zeros. Native clear is therefore N/A for that attribution revision. Full-active
sparse fixtures cannot activate absent coordinates and report N/A for first
activation/expansion. Original symmetric coordinate indexing is adapted to the
new coordinates vector; repeated two-index update workloads use the same public
matrix syntax in every source. The original symmetric clear case is an
explicit q-coordinate loop versus current fill!(coordinates(x),0); it is an
adapted equivalent whole-state operation, not a claim about an original native
fill! implementation or an optimal historical clear strategy.

References check full independent-coordinate values, exact active sets/counts,
position-map consistency, retained zeros and absent-zero writes, symmetry,
block order, live view aliases and copy independence. Every timed case repeats
its reference checks before warming. The 18 supported package test files pass
with bounds checks ([test log](research/package-tests.log)). The unchanged #463
generator reproduces its historical Markdown/CSV byte for byte ([log](research/historical-reproduction.log));
those old-machine observations are never included in new performance ratios.

Development failures are retained separately: an API-detection bug involving a
deprecated alias, a Set validator bug, and callback Core.Box captures were
corrected before the final campaign. A stopped preliminary timing stream showed
identical harness allocations in both versions and is explicitly excluded.
The corrected callbacks are guarded against boxed captures, and representative
non-construction operations allocate zero in the diagnostic check
([log](research/callback-allocation-check.log)). No source correctness failure
was repaired or hidden by an adapter.

## Controlled measurement

Linux, Julia 1.13.3, Intel Core i5-13600K; one Julia thread and one BLAS thread;
all workers sequentially pinned to logical CPU 0. Governor and sibling topology,
OS/host, exact binary SHA256, BLAS configuration, source and harness hashes,
dependency Manifest and compiler flags are retained with the raw provenance.
The common locked environment uses BenchmarkTools 1.8.0 and PrecompileTools
1.4.0; historical source is loaded unchanged rather than resolving its obsolete
full development/test dependencies. All source versions load as Main.ProximalBase
in separate processes. Runtime flags are --startup-file=no --threads=1 -O2
--check-bounds=no. Independent correctness uses --check-bounds=yes.

Each of seven broad rounds starts six fresh processes: the four revisions,
originalAA and post850AA. Both source order and fixture order reverse between
rounds. A round shares deterministic IDs/operands across source roles and uses a
new deterministic fixture seed. Three warmups and a preliminary timed-wrapper
trial precede the recorded trial. Up to 80 individual samples / 8 ms are
requested, with evals=1 and native state restoration outside timing. Both
sample count and GC time are retained; expensive operations may have fewer
samples. Constructor allocation and populated construction are separate cases.
Object size is reachable Julia heap (summarysize), not RSS or native memory.

The ratio is R = post850/original. Point estimates are medians of the seven
paired per-round median ratios. A deterministic percentile bootstrap resamples
whole rounds 10,000 times for 95% intervals. They describe the sampled process/
fixture distribution and are not multiplicity-adjusted hypothesis tests. A/A,
absolute durations and allocation changes accompany every comparison. Timings
below 100 ns are conservatively marked timer-limited; a narrow nanosecond
interval does not prove a practical source regression.

The first-use/warmup duration and source-load duration are retained separately.
They include compilation, execution and timing-wrapper preparation, and are
not presented as pure compiler timings. These warmed results do not measure
package startup, precompilation amortization or cold-cache solver execution.

## Reproduction

The [runner instructions](../../../benchmark/audit852/README.md) specify fresh
Linux reproduction, locked dependencies, source archives, correctness, broad
and follow-up campaigns, sample verification and figure generation. Each raw
campaign has observations.csv, per-trial samples, per-process provenance,
fixture seeds/pointers/padding, calibration and error logs. The summary script
checks raw sample reductions and complete paired records. A checksummed
archive will preserve all raw data and development diagnostics.
