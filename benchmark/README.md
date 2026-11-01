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
`sparse` and `atom`. They cover dense/sparse/atom calls, values and derivatives,
coordinate operations, ordered penalties, constraints and workspace states.
The blocks group covers flat and nested tuple oracles, wrappers and long block sums
on Float32/Float64 inputs, with the same cases in both suites.
The matrix group includes partial/full decompositions and fallback inputs.
Construction plus first use is measured separately from workspace reuse.

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

## Iterate-family groups

A new case in `sparse`, `atom` or `owner` must keep these rules:

- It calls only what exists in 0.5.x and 0.6. A new accessor (`atomindices`, `samegroups`, a
  private field) is reached when the case is built, through `isdefined` or `hasfield`, with the
  0.5.x spelling as the fallback, and reaches the timed closure through a function barrier;
  nothing is tested inside the timed call. The set of case IDs never depends on the source.
- It is checked against an independent reference built from the harness's own data (a dense
  array, a CSC matrix, a formula), never against the kernel under test. A product starts from a
  sentinel output, so that a call that does nothing cannot pass. Where 0.5.x and 0.6 differ (the
  working set after a clear, the inactive entries after `rmul!(x, Inf)`), only the values both
  agree on are checked.
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
- The owner's clears: `fill!_coordinates_zero` writes a zero into each of the `q = p(p + 1)/2`
  coordinates and keeps the working set; `fill!_coordinates_zero_dropzeros` adds `dropzeros!`
  (the 0.6 clear); `fill!_zero` is `fill!(X, 0)`, the 0.5.x clear, O(s). Only the values are
  checked. The new owner cases draw their own fixture (keys `owner_extra/...`), so `owner_z50`
  and the `owner` row of the same shape hold different coordinates.
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
- Some rows are inherently timer-limited: the 256-coordinate first activations (the companions
  can gate), the small clears and copies (`fill!_*`, `dropzeros!`, `copyto!_state`,
  `broadcast_assign` at small `s`; the owner's `fill!_zero`, which is O(s) and takes well under
  2 µs there), and `samegroups` `shared`. A clear cannot be enlarged without timing its restore.
- Zero-allocation readings are checked on Julia 1.13. Julia 1.10 reads a few bytes for closures
  that return a value, in the older `owner` group too. The five-argument `prox_coordinate!` with
  a strided `u` allocates for the group penalties on both sources, so FT23's zero-byte promise
  does not hold today; its rows keep `zero_alloc`, so that the baseline lists them.

**Baseline and comparisons.** The metadata records `check_bounds` (`default`, `yes` or `no`,
from `--check-bounds`) and `os` (`Sys.KERNEL` and `Sys.MACHINE`); `compare.py` does not compare
them, so use the same setting on both sides. `harness_hash` covers `run.jl` and `cases.jl`: any
edit to either changes it, and `compare.py` requires it, like the suite and the groups, to
match. After such an edit, measure the baseline source again with the same harness in the same
session. The owner baseline `baselines/symmetric-owner-0.5.md` was recorded with an earlier
harness, so its rows are a reference and not a comparison point for the current one.

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

The comparison produces `REPORT.md`, `comparison.csv` and `coverage.json`.
It reports timing variation, paired speedup intervals, mean and maximum bytes,
and unexpected warmed allocations. Single-run ratios are exploratory. Failed
checks, missing repetitions, unmatched cases or incompatible settings prevent
a comparison and retain diagnostics. An interval containing one is inconclusive.

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
both sides. A ratio gate must filter on `timer_limited`. Comparisons retain
ratios, intervals and allocation warnings, with a `timer-limited` assessment.
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
