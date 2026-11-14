# Independent review of historical iterate audit #852

Reviewer: independent Codex GPT-6.1-Sol review, high effort. Date: 2026-11-11.

**Disposition:** no unresolved substantive blocker in the reviewed harness,
numerical results, statistical analysis, or completed report findings. Final
archive packaging and its checksum/link checks remain publication work.

## Scope and evidence

I reviewed the frozen primary harness, generated supplemental/control factories,
restoration and API adapters, residual CD/BCD arithmetic and independent
references, analysis and completeness checks, export/replay/integrity tools,
regression ledger, report, and three report figures. I did not change package
source or execute a measurement campaign. Only this review record was written.

Independent read-only checks established:

* All **122 source blobs** in the four frozen `src/` trees match their exact Git
  objects. The audit checkout has no `src/` difference from the pinned post850
  revision.
* Five checks starting from individual raw samples reproduce the exported paired
  median ratio and seeded 10,000-resample interval exactly: native Float64 block
  CD; common-active scalar CD; raw-native Float32 arbitrary-layout residual BCD;
  original backing-BLAS versus current owner dot; and corrected historical05
  block CD. These span seven- and 21-round campaigns.
* The campaign index contains **17 campaigns**, **150,535 passing timing
  observations**, and **11,518,964 individual samples**. Its totals agree with
  the report. The integrity record covers **1,218 fresh measurement processes**.
* All **560 ledger entries** have 21 follow-up rounds. The 16 corrected
  post848/post850 exploratory extraction contrasts have no median above 1.05
  in their independent confirmation.
* Every legacy historical05 block contrast is marked superseded in the exported
  comparisons: **1,449 of 14,994 comparison rows**. This annotation does not
  silently discard the original observations.

The exhaustive raw-to-summary replay records a pass for every campaign and every
exported metric cell. I inspected that implementation and its results; my
separate sample recomputation was the five-case check above, not a second claim
to have manually examined every sample. Completeness checks derive the expected
round/role/case product and exact fixture seeds; provenance checks also reject
missing process records. Median, minimum, p90, finite sample values, GC counters,
binary/source identity, runtime flags, thread counts, calibration, and retained
generated-harness hashes are checked separately.

## Fairness findings and their resolution

The final analysis addresses the material adapter and preparation issues found
during review:

* The stopped stream with boxed callbacks is excluded. Valid primary timing
  files were frozen, and callbacks are guarded against direct `Core.Box`
  captures. Preparation, validation, operand generation, and scratch allocation
  remain outside timed operations.
* Native original full-array restoration and current active-prefix restoration
  condition caches differently. Separate common-full and common-active controls
  avoid presenting that setup effect as a universal scalar-kernel speedup.
  Newly exposed control flags received their own independent confirmations.
* Persistent-process follow-up was replaced by fresh-process rounds. The
  supplemental 28-round campaign is also reported as exploratory seven and
  independent confirmation 21; changed contexts are not silently pooled.
* Original native row-block views have matrix shape. I identified the exact
  64-failure validation mismatch between a collected 1-by-b view and a vector
  reference. Untimed flattening corrected validation before supplemental
  timing. The later best-native residual control uses direct matrix indexing,
  with no timed conversion and no tolerance change.
* Historical05 runtime API selection caused inference loss and 345,600 allocated
  bytes in a representative block sequence. Static selection removes those
  allocations without modifying the package. Historical05 native zero-fill
  also removed destination activity before copy, unlike current retained-zero
  fill. The corrected control preserves equal active working sets. Legacy
  block attribution is superseded; the report does not credit the spurious
  historical slowdown to a source redesign.
* The original raw constructor genuinely represents Float32 and arbitrary
  ordered vector views. Separate raw-native campaigns cover those operations;
  constructor timing is correctly excluded from that variant. Original
  activation/compaction feature exclusions remain genuine.
* Original optimized symmetric bilinear and backing-array BLAS strategies are
  measured separately from generic owner dispatch. The report acknowledges
  those existing historical capabilities and that current backing storage can
  also use BLAS.

The residual BCD implementation computes a whole block's trial changes from the
same preblock residual, then applies the coefficient and residual updates. Its
dense reference implements the same mathematics. Nonuniform layout fingerprints
test every ordered block, preventing uniform fixtures from concealing an index
permutation error. In-place copy and allocating copy, independently owned and
shared layouts, retained zeros, and absent historical activity metadata are
distinguished appropriately.

## Interpretation and remaining limits

The report now presents mixed outcomes with absolute costs, allocations, memory,
precision/layout coverage, preparation contexts, and existing native strategies.
It does not infer whole-solver convergence or solve times from these sequences.
The generated-code observations can support a representative mechanism, but
cannot assign an exact share of a timing difference to vectorization or activity
checks without a separate intervention.

Bootstrap intervals resample paired rounds, not timing samples. They remain
descriptive, unadjusted for multiple comparisons, and conditional on this one
machine, Julia build, finite fixtures, and process/preparation distribution.
Seven-round improvements have limited tail resolution. Timer-limited cases and
the large same-source constructor discrepancy are disclosed rather than used
as precise kernel or equivalence claims. Reachable size is not RSS or marginal
ownership cost. The adapted original symmetric clear is not evidence about the
fastest possible custom historical clear implementation.

These limitations are substantive boundaries of the experiment, not unresolved
correctness defects. The balanced conclusions are supported within those
boundaries.
