# Block value update investigation (#860)

**No evaluated optimization meets the acceptance gate. Production source is
unchanged.** The experiments produce real gains on some complete BCD workloads,
but also repeatable regressions on other ordinary write paths. This PR retains
the diagnosis, independent tests, generated code and reproducible measurements.

The target is block value writes. Whole-iterate state replacement, layout
comparison/cache behavior and removed proximal interfaces are out of scope.
The starting source is master `eb74a914d779371a02e4a6637eaa724aeb66fdf5`.
The final audit branch is rebased onto merged #861 at
`998b3b4c1ef2a77fddb5477bf823def9b2811041`; its entire `src/` tree is unchanged
from that base. The frozen experimental comparisons remain explicitly pinned
to the starting source. Twenty package test files pass after the rebase,
including live-view updates after whole-owner state replacement.
The investigation and implementation plan were posted [on #860 before source
changes](https://github.com/mlakolar/ProximalBase.jl/issues/860#issuecomment-6810647366).

## Diagnosis

The retained [#852 audit](../iterate-performance-852/report.md) identifies a
real cost of the activity-maintaining owner write interface relative to the
original cached backing views. New paired measurements reproduce that cost.
At n65536/b8, seven fresh-process pairs give current/original residual BCD
ratios 1.152 contiguous and 1.150 arbitrary (Float64). The corresponding
same-source A/A medians are 0.995 and 1.002. Contiguous b64 owner sweeps are
9.19 times original for Float32 and 4.47 times for Float64. These are synthetic
four-sweep sequences, not measurements of ProximalSolvers convergence.

Generated-code inspection distinguishes interacting mechanisms:

* `blockview` and `activate!` are already inlined in representative kernels.
  The view does not escape or allocate. Explicit `@inline` annotations alone
  do not remove the entry-to-group lookup or activity bookkeeping.
* Every nonzero owner write loads `group_of[i]`, explicitly bounds-checks
  `position[k]`, tests activity, and may mutate active IDs/count/positions.
  The explicit `checkbounds` survives `--check-bounds=no`. Zero writes skip
  that work, which makes zero-only cases essential controls.
* Contiguous storage-view sweeps have packed SIMD arithmetic; corresponding
  owner sweeps have scalar arithmetic and metadata-dependent control flow.
  Raw storage diagnostics construct views too, so the large gap cannot be
  assigned solely to the original's cached-view construction advantage.
* The residual kernel vectorizes other arithmetic even while coefficient
  writes remain scalar. Its matrix/residual work dilutes the storage-loop gap.
  A sweep ratio is not a prediction of full BCD speedup.
* Same-precision Float32/Float64 conversion specializes away. General source
  indexing, fallible conversions and arbitrary view semantics remain reasons
  to preserve the existing Base fallback rather than bypass validation.

These comparisons do not decompose elapsed time into additive instruction
costs. Mapping, activation branches, memory dependencies, bounds checks and
vectorization affect each other. #787's bulk-write motivation remains relevant;
its removed proximal API proposals are not restored.

## Method and scope

The [runner and recipe](../../../benchmark/audit860/README.md) extend the #852
layout/reference helpers and use its residual BCD kernel unchanged for the
`residual_bcd` cases. A separately labeled `residual_bulk` case commits the
numerical buffer with the existing documented `copyto!(blockview(x,k),buffer)`
idiom, then updates the residual. Both use identical gradients, residual
arithmetic and coordinate schedules, and independent dense block algebra
checks coefficients and residuals before timing. They are distinct caller
workloads: optimizing buffered writes does not accelerate an unchanged scalar
caller by assumption.

The matrix includes Float32/64, n8192/65536, b1/8/64, contiguous/strided/arbitrary/
reversed layouts and 10% of groups. Copy sweeps separately cover already active,
first activation, inactive all-zero writes, active groups written all zero and
pre-existing active zeros. Raw storage and once-per-block activity-check sweeps
are diagnostic controls, not proposed APIs or valid inactive-group replacements.

Linux, Julia 1.13.3, startup disabled, -O2, one Julia/BLAS thread, no bounds checks
for timing; correctness uses bounds checks. Each round reconstructs fixtures
from a new deterministic seed, paired across separate fresh source processes.
Source/fixture order rotates. Full storage/activity/residual restoration occurs
outside timing and does not call production state-copy routines. Three warmups
and a preliminary wrapper trial precede up to 80 evals=1 samples / 0.1 seconds.
Raw times/GC samples, allocation counts, source/harness hashes, Julia binary
hash and compiler/machine provenance are retained. The runner snapshots its
harness and validates source immutability; the analyzer rejects incomplete
role/round/case pairs or mismatched provenance and recomputes sample medians.

The historical diagnosis uses CPU0. Candidate campaigns use CPU2, a separate
physical core; only within-campaign paired ratios enter comparisons. The
concurrent #861 task used CPU0 during candidate timing, and unpinned #861 tests
also overlapped part of the historical diagnosis. The host is not exclusively
reserved or frequency locked. A/A controls and fresh confirmation rounds
quantify variability; they do not eliminate shared-cache/frequency effects.

## Development exclusions

Exploratory one-process runs lacked per-sample restored fixtures and are not
acceptance evidence. An early candidate snapshot accidentally included the
concurrent #861 patch from the shared checkout. That stopped campaign is
excluded; all subsequent source trees are built from frozen master plus only
the block-buffer specialization. No whole-iterate state-copy source change is
part of this branch. The original harness's first diagnosis attempt failed on
an original-only raw-view adapter error; the corrected seven-round campaign
was rerun from a new output directory.

## Whole-kernel code generation

The general bulk experiment recognizes a complete stored group, unaliases the
same-type buffer, fuses storage writes with a nonzero OR reduction, and activates
at most once after the nonthrowing loop. Its contiguous copy loop vectorizes,
including the boolean reduction. It preserves the tested value/activity contract,
but its copy-only gains are insufficient evidence for accepting it.

Static native listings show an important whole-kernel tradeoff. For the same
`residual_bulk!` signature with `StepRange{Int,Int}` groups (used by strided and
reversed layouts), the Float64 baseline listing has 654 lines and 91 spill/reload
annotations. The bulk experiment has 1,370 lines and 181 such annotations; the
singleton-only experiment has 757 lines and 107 annotations. The explicit
`sub rsp` reservation changes from 360 to 672 bytes for bulk, and to 376 bytes
for singleton (prologue alignment also differs). Float32 reservations are
360/608/376 bytes respectively. These are stack reservations, not heap allocations.

The counts include cold/fallback paths; they are not executed instruction or
spill counts. They establish larger generated kernels and increased register
pressure in the bulk variant. It is an inference that this contributes to
full-workload costs, not a uniquely isolated causal assignment. The measurements
must decide whether the tradeoff is acceptable. In the seven-round comparison,
one high-penalty Float64 reversed b8 workload is about 6.5% slower despite fast
isolated copies, with A/A close to one. That flag was carried into the fresh confirmation below.

The narrower singleton experiment uses the existing scalar owner-backed write
for a one-value same-precision buffer. It retains `Base.copyto!` for longer
buffers and custom index descriptors. Generated LLVM branches on source length;
the singleton branch reads one source value, writes through the owner and runs
its original activation logic. It avoids generic copy iteration and alias
analysis for that one write. Reading the sole source value before the write
preserves singleton overlap behavior. Custom alias-metadata callbacks retain
Base's path, including its exceptions.

Independent generated-code review adds two useful constraints on interpretation.
The baseline outlines `copyto!`; the bulk experiment puts recognition, alias
checks, address calculations and reduction machinery inside BCD. Both retain
vectorized residual-update arithmetic. At Float64 block size 8, disjoint copies
skip the 16-element main vector body and run a four-lane epilogue twice. That
path computes general stride-based addresses, performs scalar stores and reduces
packed nonzero flags. Static spill counts from the skipped main vector body
therefore cannot be assigned to the executed b8 path.

An additional controlled variant changes only the helper's executable inlining
annotation, from `@inline` to `@noinline`. This tests the combined effect of
outlining, including call overhead and optimization/code-layout boundaries,
without changing the copy or activity algorithm. It cannot independently
identify spill costs versus instruction-cache effects. The singleton-only
variant provides a separate minimal candidate if broader bulk writes fail the
complete-workload gate.

## Acceptance rule

The final decision uses the complete paired source matrix, not only favorable
rows. Every broad median above 1.05 receives fresh confirmation across the
entire matrix. A candidate must improve a representative complete BCD workload
without a repeatable >5% regression in another ordinary BCD or value-copy sweep.
The raw-storage and once-checked sweeps are diagnostics. Very short cases and
same-source control sensitivity require interpretation using absolute times,
round dispersion and A/A, rather than treating every percentage as a source
mechanism. An evaluated candidate that fails this gate is reported and excluded;
it is not shipped solely because its copy microbenchmark is faster.

## Confirmed results and decision

The final placement-controlled campaign contains **60,480 observations and
4,837,587 individual timing samples**: 576 cases, five source/control roles and
21 fresh processes per role. The complete matrix is Float32/64 × n8192/65536 ×
b1/8/64 × four layouts, with 12 operations per fixture. Three candidates were
compared to the same baseline; a separate baseline process provides A/A.
Every operation passed its independent reference before timing.

R is the median of paired process ratios. Times below are marginal medians,
so their quotient need not equal R. Intervals are descriptive, unadjusted 95%
round-bootstrap intervals (10,000 seeded resamples).

| Experiment and case | Baseline µs | Experiment µs | R [95% interval] |
|:--|--:|--:|:--|
| Inlined bulk, Float32 n65536/b64 contiguous, complete buffered BCD | 195.068 | 183.215 | 0.9424 [0.9364, 0.9529] |
| Inlined bulk, Float64 n65536/b8 reversed, complete zero-commit BCD | 291.552 | 310.691 | **1.0641 [1.0609, 1.0684]** |
| Outlined bulk, Float32 n65536/b8 contiguous, complete zero-commit BCD | 231.083 | 242.872 | 1.0559 [1.0486, 1.0701] |
| Outlined bulk, Float32 n8192/b8 strided, active groups copied to zero | 1.398 | 2.017 | **1.4379 [1.3903, 1.5030]** |
| Singleton path, Float64 n65536/b1 contiguous, complete buffered BCD | 952.606 | 829.297 | 0.8690 [0.8664, 0.8718] |
| Singleton path, Float64 n8192/b8 strided, active groups copied to zero | 1.809 | 2.0025 | **1.1066 [1.0957, 1.1276]** |
| Singleton path, same size/layout, inactive groups copied to zero | 1.823 | 2.000 | **1.0952 [1.0824, 1.1139]** |

These are real gains and real tradeoffs. The inlined bulk experiment has three
complete-BCD median flags above 1.05; the reversed Float64 case has its entire
interval above 1.05. Outlining reduces the caller's explicit stack reservation
to 376 bytes (versus 360 baseline and 672 inlined), but it has 37 copy-sweep
flags and one BCD median flag. The singleton specialization has no complete-BCD
flags, but six ordinary copy-sweep flags, including the two strong strided
zero-copy regressions above. Its one additional diagnostic flag is not a
bulk-write result.

The singleton regression is not an extra copy call: both versions make one
copy call on the ordinary b8 path. Its generated `copy_blocks!` caller now
loads/tests source length inside each group iteration before falling back to
Base, and retains the larger cold scalar/activation branch. The caller's stack
reservation is 168 versus 152 bytes. This supports added guard/code-shape costs;
it does not uniquely attribute elapsed time to one instruction. The active-zero
case adds about **0.194 µs across 102 groups**, which is small absolutely but a
repeatable 10.7% regression on a targeted value-write path.

All 576 case-level A/A medians lie between **0.9653 and 1.0384**, with no >5%
median flag. Every timed disjoint-buffer operation reports **0 Julia heap bytes
and 0 allocations** for baseline and all candidates. Stack reservations and
native-code size are separate from those heap observations. Aliased copies may
allocate a snapshot, as Base requires; aliases are tested independently rather
than hidden inside the disjoint-buffer allocation claims.

The [regression ledger](results/regression-ledger.csv) accounts for all **64**
contrasts flagged in discovery or confirmation. Forty-one repeat a >5% median;
37 confirmation intervals are entirely above 1.05. None is timer-limited by the
100 ns marginal-time screen. Each row retains both phases, absolute added time,
A/A controls, allocations and an investigation class. Initial flags that do not
repeat remain visible. Static codegen classes are evidence, not claims of a
uniquely isolated cache/spill cause.

**No production source optimization is included.** These three evaluated
candidates fail the declared gate; this does not establish that a safe internal
optimization is impossible. The public `blockview` implementation and all
working-set semantics remain unchanged. Whole-iterate state copying is left to
#861, which landed separately during this investigation.

## Reproduction, review and remaining options

The [recipe](../../../benchmark/audit860/README.md) reconstructs the experimental
source trees from the frozen baseline and the [rejected patches](patches/README.md),
not from the moving checkout. The published artifacts retain raw samples,
source/harness hashes, fixture pointers/padding, provenance and generated code.
Discovery and placement-controlled phases are not pooled: the latter prebuilds
both designs before numerical warmup and explicitly preserves allocation padding.
The final twenty-one-round comparison evaluates every strategy in that same
controlled context on CPU2 while other agent CPU-heavy work is stopped.

The [independent review record](review.md) documents GPT-6.1-Sol/high findings and
corrections. Its final independent verification reproduced all 105 confirmation
files, all 60,480 observations and individual sample medians, every exported
ratio/allocation metric, and every seeded bootstrap endpoint exactly.

Future work can evaluate payload/layout-specific kernels or compiler-boundary
choices that avoid both short-copy fixed costs and large-caller code expansion.
Any proposed public active-group/bulk writer requires a maintainer API ruling
before implementation. The removed `prox_coordinate!` and `prox_entry!`
interfaces are not candidates for restoration. The retained evidence supports
further investigation, not shipping one favorable microbenchmark in spite of
ordinary-workload regressions.
