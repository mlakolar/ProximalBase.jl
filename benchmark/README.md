# Benchmarks

This suite tracks the runtime and Julia heap allocations of ProximalBase
operations. Keep reusable cases and measurement tools here. Save generated
results with the pull request or release that uses them.

## Setup and selections

From the package root, with Julia 1.10 or later and Python 3.11 or later:

```sh
julia scripts/instantiate.jl benchmark
JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/run.jl
python3 benchmark/summarize.py benchmark/results benchmark/results/summary
```

The default is one quick run on small inputs, mostly Float64. It is an exploratory
check, with no between-run uncertainty estimate. Use a fresh output directory
for each measurement series; existing repetitions are never overwritten.

| Setting | Use |
|---|---|
| `PB_SUITE=quick` | Small representative cases; the default. |
| `PB_SUITE=extended` | Float32/Float64, larger inputs and broader cases. |
| `PB_GROUPS=vectors,smooth` | Select groups before constructing their inputs. |
| `PB_CASES=ordered/SLOPE` | Keep only the cases whose ID matches this regular expression, after the groups are built (inputs are unchanged by the selection). |
| `PB_REPETITION=1` | Seeded process/input draw, numbered 1–10. |
| `PB_OUTPUT=/absolute/path` | Output directory; defaults to `benchmark/results`. |
| `PB_LIST=true` | List selected case IDs without measuring them. |
| `PB_PROFILE_SECONDS=0.2` | Enable representative profiles; disabled by default. |

Groups are `vectors`, `smooth`, `matrices`, `constraints`, `tensors`, `products`, `blocks`, `owner`,
`sparse`, `atom` and `ridge`. They cover dense/sparse/atom calls, values and derivatives,
coordinate operations, ordered penalties, constraints and workspace states.
The blocks group covers flat and nested tuple oracles, wrappers and long block sums
on Float32/Float64 inputs, with the same cases in both suites.
The `ridge` group measures `SqrNormL2`, centered and uncentered, in homogeneous `Float32` and
`Float64` and in `Float32` storage with `Float64` parameters (`prox!`, `value_and_prox!`, a
`prox_entry` loop, `gradient!`, `value_and_gradient!`, the value), and, as separate
`layout=recovery` cases, calls whose parameters or centers leave the model's range. The
extended suite adds `Constrained(SqrNormL2, IndHyperplane)` with per-entry steps (repeated
and cold calls) and a scalar-step control (`layout=constrained`).
The matrix group includes partial/full decompositions and fallback inputs.
Construction plus first use is measured separately from workspace reuse.
The sparse operands of the `products` group hold exactly density × N nonzeros, rounded to the
nearest integer and at least 1 (N the number of entries), at random positions drawn from their
own scenario keys, so the `work` of a case is the same in every repetition.

The `owner` group measures `SymmetricSparseIterate` (reads, writes, first activation,
coordinate sweeps, `mul!` (also with half of the stored coordinates explicit zeros), the
three clears, the momentum value loops, `Matrix`, construction) beside a `SparseMatrixCSC`
and a dense `Symmetric` product. It uses only calls that exist in 0.5.x and 0.6, so the same harness
gives the before and after of the redesign in
[#391](https://github.com/mlakolar/ProximalBase.jl/issues/391). The pre-refactor
numbers are recorded in `baselines/symmetric-owner-0.5.md`.

The `sparse` and `atom` groups measure `SparseIterate` and `AtomIterate` on the hot paths of the
iterate-family design, the cases of the
[benchmark protocol](../docs/design/iterate-family-0.6-acceptance.md#6-benchmark-protocol-a-future-measurement-not-a-result)
(§6 of the acceptance guide): reads, writes, first activation, the handle and value sweeps,
`atom` and `atomindices`, products, scaling, clears, the block trial, the momentum loops,
`samegroups`, aliasing, copy and import, and compaction. They run the same calls on 0.5.x and
0.6, so each 0.6 change on a hot path is measured against the 0.5.x baseline in
`baselines/iterate-family-0.5.md`. The FD-7 handle sweeps that `owner` once held are the
`sweep_activecoordinates` cases of these groups. [Iterate-family groups](#iterate-family-groups)
lists the fixtures and the rows that need a note.

For example, select matrix operations across the extended input sizes:

```sh
PB_SUITE=extended PB_GROUPS=matrices PB_OUTPUT=benchmark/results/matrices \
  JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/run.jl
```

The optional [GPU solver suite](gpu_solver/README.md) measures CPU/CUDA time
to independently checked accuracy on physical GPU hardware.

### Reference checks without timing

`benchmark/check_only.jl` builds the same cases and runs every selected case's reference checks
and warm-up, without timing it. It does what `measure!` does before and after the timed trial:
`validate()` and the digests of the case's read-only inputs, three `prepare(); op()` rounds and the
allocation probe. A case with an automatic evaluation count then gets two samples of ten
evaluations. Every case then has its inputs compared with their digests, and a case with an
automatic evaluation count gets `validate()` again (see [Read-only inputs](#read-only-inputs)).
The cases run in the measurement order of the repetition. The script reads `PB_SUITE`, `PB_GROUPS`,
`PB_CASES`, `PB_REPETITION` and `PB_EXPECTED_PACKAGE_ROOT`. With `PB_OUTPUT` set, it writes
`check-rep<NN>-errors.tsv` there: every failure with its exception and full stack trace, written
and flushed as it occurs, so a timeout keeps what was recorded. It prints the ID of every failing
case and exits with status 1 if any case fails. `benchmark/check_only_test.jl` tests the driver on
synthetic cases and the read-only input checks on a few real cases, then runs
`benchmark/cases_test.jl`, which tests the sparse operands of the `products` cases and runs their
reference checks for repetitions 1–10.

```sh
JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/check_only.jl
PB_SUITE=extended PB_GROUPS=products,sparse,atom,owner JULIA_NUM_THREADS=1 \
  julia --project=benchmark benchmark/check_only.jl
```

CI runs the driver's tests and the quick suite (839 cases) on Julia lts and 1 whenever the package
tests run or the benchmark harness changes, and uploads the errors file of a failing run as the
artifact `benchmark-check-errors-<version>-<attempt>`. On recent pull-request runs, CI's step for
the driver's tests took 15–73 s and its quick-suite step between about 1.5 and 4 minutes. The extended suite
(5,540 cases) is a manual run; its `products,sparse,atom,owner` cases need about 16 GB of memory
in one process. `check_only.jl` is not part of the harness hash, which covers `run.jl` and
`cases.jl`.

## Iterate-family groups

A new case in `sparse`, `atom` or `owner` must keep these rules:

- It calls only what exists in 0.5.x and 0.6. A new accessor (`atomindices`, `samegroups`,
  `matrixview`) or private field is reached when the case is built, through `isdefined` or
  `hasfield`, and reaches the timed closure through a function barrier; nothing is tested inside
  the timed call. The set of case IDs never depends on the source. The fallback depends on the
  control. For `atomindices`, `samegroups` and `matrixview` it is the 0.5.x spelling
  (`x.groups[k]`, the private `_samelayout`, the owner itself). A control that reads a private
  field falls back the other way when the field is missing: the `private_fields` momentum loops
  (`nzval`; for `AtomIterate`, `storage`, `active`, `nactive` and `groups`) fall back to the
  accessors (`activevalues`, `activecoordinates`, `atomindices`), the FD-12 staging control
  (`storage`) to a write through `atom(x, k)`, and the layout-cache reset of `samegroups`
  (`layout_cache`) to no reset. The row keeps its ID and measures the other path.
- It is checked against an independent reference built from the harness's own data (a dense
  array, a CSC matrix, a formula), never against the kernel under test. A reference that depends on
  an array the operation reads is computed when the case is built. A product starts from a
  sentinel output, so that a call that does nothing cannot pass. Where 0.5.x and 0.6 differ (the
  working set after a clear, the inactive entries after `rmul!(x, Inf)`), only the values both
  agree on are checked.
- It declares its read-only inputs (see [Read-only inputs](#read-only-inputs)): the shared working
  iterate only where the case leaves it logically unchanged, and the snapshot that `prepare`
  restores it from where the case changes it.
- A case that changes state restores it in `prepare`, outside the timed region, and runs with
  `evals = 1`. Every first activation takes distinct absent coordinates or groups, and the
  restore asserts that none is active.
- `zero_alloc` is set only on the operations that FT23 of the acceptance guide lists as
  allocation-free, and is `false` explicitly on every other warm case (the default for a warm
  case is `true`).

**Fixtures.** `Float32` and `Float64`. `SparseIterate` vectors have `n ∈ {10⁴, 10⁶}` and
`(n, s) ∈ {(10⁴, 10²), (10⁶, 10²), (10⁶, 10³), (10⁶, 10⁴)}`, so `s ≤ n/100`. The matrices are
`1000 × 64` and `256 × 1000` with 0.5%, 3% and 20% of the entries stored, exactly `round(d·L)`
of them, so that a repetition never crosses the 1/32 switch between the two walks of the matrix
`mul!` (the stored entries at 0.5% and 3%, the columns at 20%). `AtomIterate` has `p = 8·10⁴`
(`8·10⁵` in the extended suite), groups of 1, 8 and 64 entries, contiguous or strided (the
multi-response layout), with 1% and 10% of the groups active. A dense operand is built only
where it fits: the vector `mul!` (strided and transposed `A`, 256 and 1024 rows) runs at
`n = 10⁴`, `s = 10²` only, so the growth with `s` is read from the matrix fixtures; the matrix
`mul!` takes an `A` of 256 and 1024 rows; the `AtomIterate` `mul!` takes a strided `A` of 256
and 1024 rows at `p = 8·10⁴` and nothing dense at `8·10⁵`; the `Symmetric` operands have
`p ∈ {256, 1024}` for `SparseIterate` (3% and 20% of the entries stored) and `p = 1024` for
`AtomIterate` (10% of the groups active). Every product runs with 0% and 50% of the stored
values exact zeros at random slots (layouts ending `z0` and `z50`). The quick suite builds a
small subset of `Float64` fixtures: for `sparse`, the vectors `(10⁴, 10²)` and `(10⁶, 10³)`, the
matrices at 3% (the conversion cases also use the 20% `1000 × 64`) and the smallest dense
operands; for `atom`, `p = 8·10⁴` with 10% of the groups active, `g = 1` and `g = 8` contiguous,
`g = 8` strided, and 256-row operands.

**Rows that need a note.**

- The sweeps (`sweep_full`, `sweep_active_update`, `sweep_via_coordinates`, `sweep_via_x`) write
  back `x[i] + zero`: the compiler removes `x[i] = x[i]`, and the sweep would measure nothing.
- `rmul!`: one call on a `SparseIterate` repeats a power-of-two scale (layout `x<r>`,
  `work = r·s`) and stops before overflow, so the reference is exact. `AtomIterate` has no
  `rmul!` method: the generic loop visits all `p` entries (repeat count 1), and `rmul!(x, Inf)`
  is checked on the whole vector. For a `SparseIterate` with `Inf`, only the active entries are
  checked.
- FD-12, `prox_coordinate!_5arg`: a call takes 1536 distinct coordinates of a `SparseIterate`
  (at most 768 of them active), or 256 groups of an `AtomIterate` (64 for `g = 64`), so that a
  call takes about 2 µs or more on 0.5.x.
- `broadcast_assign` equals `copyto!_state` on 0.5.x: `x .= y` from an iterate of the same type
  takes the Base identity fast path and calls `copyto!(x, y)`. `broadcast_self` (`x .= abs.(x)`)
  is the real broadcast.
- FD-6, `atomindices_loop` takes the exclusive or of the indices of every active group. A sum over a
  range has a closed form that the compiler uses, and the loop would measure nothing per index.
  The exclusive or is real work per index, and it vectorizes on long contiguous groups, so its time
  per index falls as the group size grows (on the pinned source, about 1.0 ns per index at `g = 1`,
  0.7 at `g = 8` contiguous, and about 0.15 at `g = 64`); compare a row only with the same row.
  `atomindices_view` reads a dense vector through the indices instead.
- The owner's clears: `fill!_coordinates_zero` writes a zero into each of the `q = p(p + 1)/2`
  coordinates and keeps the working set; `fill!_coordinates_zero_dropzeros` adds `dropzeros!`
  (the 0.6 clear); `fill!_zero` is `fill!(X, 0)`, the 0.5.x clear, O(s). Only the values are
  checked. The other new owner cases draw their own fixture (keys `owner_extra/...`); `owner_z50`
  has the coordinates of the `owner` row of the same shape, with half of the stored values set to
  exact zero at random slots.
- FD-11, `samegroups` (the private `_samelayout` on 0.5.x), has three cases per layout.
  `shared`: the iterate shares the penalty's group array, a pointer comparison.
  `distinct_first_seen`: `prepare` clears the layout cache, so the call does one full comparison
  of the `K` groups (for `K < 4096`, it compares `cld(4096, K)` iterates, so that the call is
  not below the timer). `alternating`: two distinct equal layouts in turn, one full comparison
  per call. The cache is set by the first call, so a case that repeated one call would measure
  the pointer comparison.
- The first activation of 256 coordinates or groups is fixed by the design; its companion has as
  many disjoint batches of 256 as fit, up to 16 (layout ending `_x16`, or `_x4` where only four
  fit: `AtomIterate`, `p = 8·10⁴`, `g = 64`).
- Some rows are timer-limited (the baseline lists them): the small clears and copies (`fill!_*`,
  `dropzeros!`, `copyto!_state`, `broadcast_assign` at small `s`; the owner's `fill!_zero`, which
  is O(s) and takes well under 2 µs there), the `group_layouts` coordinate rows and `samegroups`
  `shared`. A clear could be enlarged by clearing several independent iterates in one call and
  restoring them all in `prepare`; that is not done here, because it multiplies the memory of the
  large fixtures. The rows with a larger active set (about 5 µs or more) can gate, and the rows
  at small `s` cannot. The first activations are not timer-limited (about 0.6 to 3 µs per call at
  256 coordinates or groups), but the `first_activation` rows of `SparseIterate` for the layouts
  `absent_distinct_batch` and `absent_distinct_batch_x16` are diagnostic and never gate: each of
  the two missed the 20% cross-check with an independent loop on one of the two machines tried
  (see the baseline).
- Zero-allocation readings are checked on Julia 1.13. Julia 1.10 reads a few bytes for closures
  that return a value, in the older `owner` group too. The five-argument `prox_coordinate!` with
  a strided `u` allocates for some group penalties on both sources, so FT23's zero-byte promise
  does not hold today: `GroupNormL2` and `SparseGroupNormL2` allocate for contiguous and strided
  groups (16,384 B per call of 256 steps), `GroupNormLinf` only for strided groups, and `NormL1`
  never. Its rows keep `zero_alloc`, so that the baseline lists them.
- BenchmarkTools protects the timed closure from the optimizer only where `Base.blackbox` exists,
  and neither Julia 1.10 nor 1.13.3 has it. On Julia 1.13.3 the timed operations were checked by
  perturbing an input between evaluations: the time did not change, so nothing is hoisted. Another
  Julia version needs that check again, and a baseline and its comparison must use the same Julia
  version.

**Baseline and comparisons.** The metadata records `check_bounds` (`default`, `yes` or `no`,
from `--check-bounds`) and `os` (`Sys.KERNEL` and `Sys.MACHINE`). `compare.py` rejects two sides
whose recorded values differ for the same Julia version (see [Compare revisions](#compare-revisions)). `harness_hash` covers `run.jl` and `cases.jl`: any
edit to either changes it, and `compare.py` requires it, like the suite and the groups, to
match. After such an edit, measure the baseline source again with the same harness in the same
session. The owner baseline `baselines/symmetric-owner-0.5.md` was recorded with an earlier
harness, so its rows are a reference and not a comparison point for the current one.

Whether a case can gate a 5% slowdown is decided by an A/A control, not by the spread of the
repetitions: every repetition draws its own inputs (`PB_REPETITION` is a seed), so the spread mixes
noise with input effects, while `compare.py` pairs the same repetition numbers on both sides and the
inputs cancel in the pair. The baseline therefore carries a second measurement of the same source
with the same harness and the same repetition numbers (separate processes, alternating order),
compared with `compare.py` as a revision would be. A case can gate when it is not timer-limited and
the half-width of that A/A 95% interval for the ratio is at most 5%. The csv carries the A/A ratio,
the half-width and the verdict (`gate5`). Compare a revision with the same repetition numbers
as the baseline, in the same session as its own measurement of the pinned source.

The iterate-family baseline is the extended suite on the 0.5.x source pinned at `def81ec9`,
three repetitions, in two output directories: the groups, and the existing `group_layouts` cases
of `vectors` (FD-11). Validate and summarize each directory with `--expected-runs 3`.

```sh
julia benchmark/setup.jl /absolute/path/to/pinned benchmark/results/env-pinned
for r in 1 2 3; do
  PB_SUITE=extended PB_GROUPS=products,sparse,atom,owner PB_REPETITION=$r \
    PB_OUTPUT=benchmark/results/groups PB_EXPECTED_PACKAGE_ROOT=/absolute/path/to/pinned \
    JULIA_NUM_THREADS=1 julia --project=benchmark/results/env-pinned benchmark/run.jl
  PB_SUITE=extended PB_GROUPS=vectors PB_CASES='^group_layouts/' PB_REPETITION=$r \
    PB_OUTPUT=benchmark/results/layouts PB_EXPECTED_PACKAGE_ROOT=/absolute/path/to/pinned \
    JULIA_NUM_THREADS=1 julia --project=benchmark/results/env-pinned benchmark/run.jl
done
```

`benchmark/sphere_l1.jl` is a standalone script for the sphere proximal map of the ℓ1 norm
(#459); it uses only the public API, so it runs against any checkout. Its results for #459 are in
`baselines/sphere-l1-459.csv`, described in `baselines/sphere-l1-459.md`.

## Compare revisions

Keep one copy of this harness fixed while measuring two package checkouts.
Prepare fresh environments for the starting and proposed revisions with the
same Julia executable, then run that same harness with each environment:

```sh
julia benchmark/setup.jl /absolute/path/to/before benchmark/results/env-before
julia benchmark/setup.jl /absolute/path/to/after benchmark/results/env-after
PB_EXPECTED_PACKAGE_ROOT=/absolute/path/to/before \
  PB_OUTPUT=benchmark/results/before JULIA_NUM_THREADS=1 \
  julia --project=benchmark/results/env-before benchmark/run.jl
PB_EXPECTED_PACKAGE_ROOT=/absolute/path/to/after \
  PB_OUTPUT=benchmark/results/after JULIA_NUM_THREADS=1 \
  julia --project=benchmark/results/env-after benchmark/run.jl
python3 benchmark/compare.py benchmark/results/before benchmark/results/after \
  benchmark/results/comparison
```

The runner verifies `PB_EXPECTED_PACKAGE_ROOT` before building cases. Use the
same suite, groups, CPU, Julia/BLAS versions and thread settings on both sides. The runner uses one BLAS thread. For stronger evidence, repeat in fresh
processes with `PB_REPETITION=1,...,5` (or up to 10), alternating which revision
runs first. Pass `--expected-runs 5` to the summarizer, validator and comparator.
Repetitions belong in the same directory; different Julia versions need separate
environments. Versions are discovered from metadata, or explicitly required by
`validate.py` and `summarize.py` with `--versions`.

The comparison produces `REPORT.md`, `comparison.csv`, `coverage.json`, `settings.json` and `outcomes.csv`.
It reports timing variation, paired speedup intervals, mean and maximum bytes,
and unexpected warmed allocations. Single-run ratios are exploratory. Failed
checks, missing repetitions, unmatched cases or incompatible settings prevent
a comparison and retain diagnostics. An interval containing one is inconclusive.

To compare only some repetitions, name them with `--reps` after the positional
arguments, each as a number `k` or an inclusive range `a-b`:

```sh
python3 benchmark/compare.py benchmark/results/before benchmark/results/after \
  benchmark/results/comparison --expected-runs 10 --reps 6-10
```

`--reps 6-10`, `--reps 6 7 8 9 10` and `--reps 6-8 --reps 9 10` are the same
selection; `--reps 1-3 7` is allowed. A repetition outside `1..N` for
`--expected-runs N`, one named twice (also through overlapping ranges) or a
reversed range is an error. Both directories are still validated whole: every
repetition `1..N` must exist and be valid, also those not compared. Everything
except `settings.json` uses only the chosen repetitions: coverage, the
timer-limited rule, the unexpected-allocation flag, the intervals and the
timing outcomes, and `paired_runs` is the number chosen (one chosen repetition
is a single run). Below its title, `REPORT.md` names the compared
repetitions, for example `Repetitions compared: 6-10 (5 of the 10 per Julia
version in each directory).`, and `coverage.json` lists them under
`repetitions`; both are present without `--reps` as well.

`check_bounds` and `os` are compared separately for each Julia version present on both sides;
values of different versions are never compared. Two different known values in any version
reject the comparison, and `errors.json` names each version and key. A record older than these
keys does not have them. Such a record is still validated and compared, but the setting counts
as unknown for that version, never as matching. `settings.json` holds, for each version and key,
the status (`match`, `unknown_before`, `unknown_after` or `unknown_both`) and both values, with
`null` for a missing one:

```json
{"by_version": {"1.13.3": {"check_bounds": {"status": "match", "before": "default", "after": "default"},
                           "os": {"status": "unknown_before", "before": null, "after": "Linux x86_64-linux-gnu"}}}}
```

`REPORT.md` has one "Recorded settings, Julia <version>" line per version, naming every setting
of that version that could not be verified to match. `validate.py` lists the keys a directory
lacks under `unrecorded_settings`. Within one Julia version of a directory, all repetitions must
agree on both keys, including whether they record them.

**Three timing outcomes for a PR (#462).** `outcomes.csv` gives every case a slowdown
(after/before, the inverse of the speedup) with one-sided 95% bounds from the paired log ratios,
and a timing outcome:
**pass** when the upper bound is below 1.05; **investigate** when the slowdown is at least 1.05
and the lower bound is above 1 (evidence of a slowdown, to confirm with independent measurements);
**inconclusive** otherwise, and always for a single run or a timer-limited case, with the reason
(their observed slowdown is kept, without bounds). A case never passes because no slowdown was
significant. These are timing outcomes, not an overall approval: `outcomes.csv` also carries
`unexpected_allocation`, and `REPORT.md` flags every primary case with an unexpected allocation
whatever its timing outcome. For a performance-sensitive PR, name
its primary cases (chosen from the changed code paths, never for low variance) in a file with one
ID per line and pass `--primary FILE`; they get `role=primary` and a table in `REPORT.md`, and
every other case is diagnostic. A named ID missing from the comparison is an error. Use five
paired, interleaved repetitions (`--expected-runs 5`); an inconclusive primary case needs more
repetitions or the maintainer's decision. `comparison.csv` and its interval assessment are
unchanged.

## Measurement and evidence

Compilation, numerical checks and input/cache preparation are outside timing.
Cases without a prepare step use automatic evaluations per sample, tuned within
`PB_SECONDS`. Cases with a prepare step use one evaluation and reset their
starting state before each sample.
`PB_SAMPLES` and `PB_SECONDS` override the sample/time limits; keep them matched
between revisions. Intervals describe variation across process/input draws,
including machine noise; they do not correct for correlated noise or multiple
comparisons. Julia heap bytes exclude native library allocations and peak memory.
Profiles include preparation callbacks and are diagnostic, not latency estimates.

Each process retains a `-calibration.tsv` file with raw no-op timer readings at
`start` and `end`. Metadata records each phase's median, minimum and timer
step: the smallest positive reading after rounding to whole nanoseconds, which
is one tick on a coarse timer. It also records `timer_floor_ns`: the maximum
of the medians and usable steps. Fewer than 100 readings in either phase, or two sub-nanosecond medians
with no steps, gives an unusable calibration (NaN floor).

A measurement is timer-limited when `median_ns * evals < K * timer_floor_ns`
(strict). With a missing or NaN floor, only `evals == 1` is limited. Missing
`evals` is also timer-limited. Older records can be summarized but fail record
validation. Both analysis tools accept `--timer-limit-multiple K` (positive and finite; default 10). Any limited
repetition marks the case/version's `timer_limited` column; comparisons check
both sides over the compared repetitions only. A ratio gate must filter on
`timer_limited`. Comparisons retain ratios, intervals and allocation warnings,
with a `timer-limited` assessment.
Their CSV also records `timer_limited_by` (for example,
`before:rep01,rep03;after:rep02`), `timer_limit_multiple`, and distinct
`before_evals` / `after_evals` joined with `/`. Summary reports give the timer
floor range and unusable calibration count per Julia version.

Results retain all samples, checks, allocation observations, settings, package
revision, dirty status and source/harness content hashes. Preserve manifests and
any uncommitted source changes with evidence supporting a PR claim. Do not
select the fastest draw or discard adverse runs. The default suite is not a
substitute for the package tests or broad coverage of every parameter regime.

```sh
python3 benchmark/validate.py benchmark/results benchmark/results/validation.json
python3 benchmark/check_analysis.py
```

Generated results, environments and profiles are ignored by Git. Put the concise
finding and relevant limitations in the PR, and use durable attachments when
detailed records are needed. Historical experiments remain recoverable in the
[pre-cleanup snapshot](https://github.com/mlakolar/ProximalBase.jl/tree/3ec6f5513d3d8c91beec010f156c533f09cbf739/benchmark).
The repeated CPU baseline and its raw-data locations are documented with
[PR #102](https://github.com/mlakolar/ProximalBase.jl/pull/102) and its
[original report](https://github.com/mlakolar/ProximalBase.jl/blob/0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a/benchmark/campaign/baselines/discovery-2026-10-25/README.md).

### Read-only inputs

Every case declares its read-only inputs (`addcase!(…; inputs = (; x, …))`, a required keyword): the
arrays and iterates that the operation or `prepare` reads and must leave logically unchanged. These
include operands, steps, weights, directions, index and trial-point arrays, the snapshots that
`prepare` restores from, the arrays of the call site that a function object holds, and the shared
working iterate of a case that leaves it unchanged. Destinations, state that a case changes (even
when `prepare` restores it), workspaces, caches, scalars and function objects are not declared. A case
with nothing to declare passes `inputs = (;)` with a comment.

`measure!` and `check_only.jl` digest the inputs right after the pre-trial `validate()` (another case
may legitimately have changed shared state before) and compare the digests after the trial, for every
case, before the post-trial `validate()` of an automatic evaluation count (the sparse and atom
validations restore the shared iterate first). A change throws
``read-only input `x` changed during the trial of <id>``, and no metrics row is written. Profiling
runs after the last check; it is diagnostic and writes no metrics.

The digest is 64 bits over a logical record, computed by a loop over every stored element of type
`Float32`, `Float64` or an integer type (an input of another element type is rejected when the case
is built): a dense array by its size and elements (with `isequal` semantics: `-0.0` differs from `0.0`, NaNs are equal),
a `Transpose` by its parent, a `Symmetric` by its whole parent (only the referenced triangle is
meaningful), a view by its elements, a `SparseIterate` or `SymmetricSparseIterate` by its stored
(coordinate, value) pairs, stored zeros included, sorted by coordinate, an `AtomIterate` by its sorted
active groups, its entries and its group layout (through the build-time spelling of `atomindices`), and
a tuple by its leaves. A digest names the input, not the entry.

The builders (`addprox!`, `addcold!`, `addgrad!`, `addblockprox!`, `addblockgrad!`) compute their
references when the case is built, from the untouched input and a copy of the function object, and
start every validation by comparing their inputs with digests taken then
(``read-only input `x` changed after the case was built``). So do the CD trace and the copies whose
source is also the reference. Every other reference that depends on an array the operation reads is
also computed when the case is built. A few checks still evaluate the output at check time instead of
comparing it with a stored reference: the value checks `f(y)` of a proximal output, the
checks of a quadratic model (`q(x)`, `gradient(q, x)`, `hessian_vec(q, x, v)`) and the residual
`y - X*β` of the CD trace.
