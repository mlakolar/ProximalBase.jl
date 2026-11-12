# Block value update investigation (#860)

The target is block value writes. Whole-iterate state replacement, layout
comparison/cache behavior and removed proximal interfaces are out of scope.
The starting source is master `eb74a914d779371a02e4a6637eaa724aeb66fdf5`.
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
isolated copies, with A/A close to one. Fresh quiet-host confirmation is required.

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
