# BlockIterate value-update investigation (#860)

**Investigation complete: reject all three evaluated optimizations.** No
production code change is proposed, and the underlying performance regression
has not been fixed. This clean replacement supersedes PR #867's integration
work while the historical exposure remains separately tracked by #876.

## Scope and diagnosis

The measured baseline is `eb74a914d779371a02e4a6637eaa724aeb66fdf5`; the original
cached-view implementation is `bd0740d0ebbebcc76fe208e4318fe6fc5451d34b`.
The [#852 audit](../iterate-performance-852/report.md) found approximately
13.4% and 15.8% slower residual BCD in contiguous and arbitrary layouts.
Seven new paired diagnostic rounds measured current/original Float64 ratios
of 1.152 and 1.150 at n65,536/block size 8, with A/A medians 0.995 and 1.002.
Contiguous size-64 owner sweeps were 9.19× original for Float32 and 4.47× for
Float64. Sweep ratios do not predict complete-BCD speedups.

Representative Float32/64 LLVM/native inspection supports interacting costs:

- `blockview` and `activate!` already inline, and views do not heap-allocate.
  Adding an inlining annotation alone does not remove the regression.
- Nonzero owner writes load `group_of[i]`, check/test `position[k]`, and may
  update activity IDs, counts and reverse positions. The explicit activation
  bounds check survives disabled bounds checking; zero writes bypass activity.
- Backing-storage sweeps vectorize while owner writes have scalar arithmetic
  and metadata-dependent control flow. Matrix/residual work dilutes that gap
  in complete BCD, whose other arithmetic remains vectorized.
- Same-precision floating conversion specializes away. General indexing,
  conversions, aliases and live-owner view semantics prevent replacing the
  public view with an unchecked cached storage view.

These observations do not assign additive time costs to individual operations.
Mapping, validation, branches, dependencies and vectorization interact.
Historical #787 motivated bulk writes; its obsolete proximal APIs are excluded.
Whole-owner state-copy implementation work belongs to #861/#863 and is unchanged.

## Method and acceptance gate

Linux / Julia 1.13.3, -O2, startup disabled, one Julia/BLAS thread; correctness
uses bounds checks, timing disables them. Float32/64, n8,192/65,536, block sizes
1/8/64 and contiguous, strided, arbitrary and reversed layouts are covered.
Each fixture visits 10% of groups in a fixed schedule; this is synthetic
residual-maintaining BCD, not a convergence or end-to-end solver measurement.

The original interleaved scalar residual kernel is retained. A separately
labeled complete buffered BCD commits with `copyto!(blockview(x,k),buffer)`
before its residual update. Dense block algebra and the residual identity
independently validate both numerical paths and activity before timing.
Copy sweeps distinguish first activation, inactive zeros, active-to-zero writes,
pre-existing active zeros and nonzero active writes.

The final placement-controlled campaign contains 576 cases × five roles ×
21 fresh rounds: **60,480 observations and 4,837,587 timing samples**. Roles rotate;
seeds vary by round and match across roles. Both designs precede numerical
warmup, seeded allocation padding stays alive, and storage/activity/residuals
are restored outside timing without production owner-copy methods.

The analyzer validates source/compiler/binary provenance, complete pairs and
raw medians. R is the median paired process ratio; intervals are descriptive,
unadjusted 95% round-bootstrap intervals from 10,000 seeded resamples.
Exploratory and confirmation phases are not pooled. All exploratory/confirmation
median flags above 1.05 are retained in a 64-row ledger: 41 repeated median
flags, 37 confirmation intervals wholly above 1.05, none below the 100 ns screen.

A candidate must improve representative complete BCD without a repeatable >5%
regression in another ordinary BCD or value-copy path. Raw-storage and
once-per-block activation-check sweeps are diagnostic controls, not valid
inactive-group replacements or public API proposals.

## Rejected candidates

| Candidate and representative case | Baseline µs | Candidate µs | R [95% interval] |
|:--|--:|--:|:--|
| Inlined bulk: Float32 n65,536/b64 contiguous buffered BCD | 195.068 | 183.215 | 0.9424 [0.9364, 0.9529] |
| Inlined bulk: Float64 n65,536/b8 reversed zero-commit BCD | 291.552 | 310.691 | **1.0641 [1.0609, 1.0684]** |
| Outlined bulk: Float32 n8,192/b8 strided active-to-zero copies | 1.398 | 2.017 | **1.4379 [1.3903, 1.5030]** |
| Singleton: Float64 n65,536/b1 contiguous buffered BCD | 952.606 | 829.297 | 0.8690 [0.8664, 0.8718] |
| Singleton: Float64 n8,192/b8 strided active-to-zero copies | 1.809 | 2.0025 | **1.1066 [1.0957, 1.1276]** |
| Singleton: same size/layout, inactive zero copies | 1.823 | 2.000 | **1.0952 [1.0824, 1.1139]** |

Marginal time quotients need not equal paired R. Inlined bulk gains 5.8% on
one complete BCD case but loses 6.4% on another. Outlining reduces code expansion
but ordinary copies lose up to 43.8%; it has 37 copy-sweep flags and one BCD
median flag. Singleton gains 13.1% on singleton BCD but loses about 10% on
ordinary size-8 copies. **None passes the gate.**

Bulk recognizes a complete group, handles aliasing, writes/reduces nonzeros,
and activates once. Correctness is insufficient for acceptance: its general
inlined loop enlarges complete BCD. Representative Float64 StepRange listings
reserve 360 bytes of stack for baseline, 672 inlined and 376 outlined.
Static spill/listing counts include cold paths and do not prove executed spills.
Size-8 copies skip the main vector body and use a general-address epilogue.
The singleton fallback adds per-group length guards; both callers retain one
copy call. No unique instruction-cache, spill or instruction cause is isolated.

All 576 A/A median ratios are 0.9653–1.0384, with no >5% median flag. Timed
disjoint-buffer operations report zero Julia heap bytes/allocations in baseline
and candidates; this excludes stack, native/RSS observations and alias snapshots.
The host was not exclusively reserved or frequency locked. Earlier exploratory
rounds overlapped other work; the final confirmation held other heavy local
work. A/A and fresh fixtures quantify variability without eliminating it.

## Retained tests, evidence and future work

`test/block_value_updates.jl` retains independent dense references for active
and inactive zeros, activation/compaction ordering/counts/positions, mixed
precision and underflow, layouts, overlapping sources, prefix/partial writes,
conversion failures, custom indexing/alias callbacks and offset axes. Its
whole-owner interaction checks live block views after replacement; the generic
replacement matrices remain in the existing `block_replacement` suite.

The [evidence manifest](evidence.md) links the immutable approved sanitized
release and replay instructions. Frozen harnesses, raw samples, generated code,
rejected source variants and detailed review records remain archived there.
No additional permanent harness or rejected implementation patch is needed:
the existing #852 harness stays unchanged, and the release replays the exact
published statistics without the moving repository checkout.

Activity tracking and scalar owner writes introduce overhead, but these
experiments reject only the evaluated candidates and measured contexts.
Future work may test narrower layout/payload kernels or compiler boundaries;
a new public bulk-write contract requires a separate maintainer ruling.
Do not restore `prox_coordinate!` or `prox_entry!`. Closing #860 as investigated
would record this research outcome, not claim a performance fix.
See the separate [#876 remediation assessment](remediation-876.md) before
retiring the original contaminated branch or requesting historical cleanup.
