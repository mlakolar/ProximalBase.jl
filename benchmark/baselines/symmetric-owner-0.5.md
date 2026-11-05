# SymmetricSparseIterate, 0.5.x baseline (issue #391, Phase 0)

The pre-refactor measurement required by the acceptance guide, section 4, and by the
Phase 3 gate of #391: *a coordinate-descent sweep must not be slower*. Re-run the same
harness on the new owner and compare row by row.

## Identity of the measurement

| | |
|:--|:--|
| Comparison point | the **final post-#392, pre-break 0.5.x** owner: `master` at `def81ec9`, whose package source hash is `a1583f070615bd3ea38705eacda4401ca1c6bafd` (`src/`, `ext/`, `Project.toml`) |
| Measured at | commit `4b15e1bf` of this branch (`master` plus benchmark files only), `source_dirty = false` |
| Harness | `benchmark/run.jl` and `benchmark/cases.jl`, hash `f79742e1558d4fb499cbd70fc7be266a24331a30` (recorded as `harness_hash` in each metadata file) |
| Julia | 1.13.3, `JULIA_NUM_THREADS=1`, BLAS 1 thread, default `--check-bounds` |
| CPU | Apple M3 Pro (`apple-m3`), macOS |
| Settings | extended suite, `PB_REPETITION=1,2,3`, 40 samples of at least 0.1 s |
| Limits | Float64 only, one machine, three repetitions |

An earlier measurement at `a73f99e5` (before #392) was discarded: the owner code is the same, but
it used a harness with the fixture errors listed in the review of #401.

```sh
for r in 1 2 3; do
  PB_SUITE=extended PB_GROUPS=owner PB_REPETITION=$r PB_OUTPUT=/absolute/path \
    PB_EXPECTED_PACKAGE_ROOT=$PWD JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/run.jl
done
```

## How to read the table

`median` is the median over the three repetitions of each repetition's median sample time;
`min` is the smallest sample over all repetitions. Times are in microseconds **per call**, and
`work` is the number of operations in one call. `bytes` is the largest `@allocated` per call
(see the memory section before using it for anything but "zero").
A case with `evals = 1` (first activation, `Matrix`, construction) has page-fault and GC noise:
read its `min`, not its `median`. Three repetitions on one machine are an alarm, not a
confidence interval.

What one call does:

- `read_X[i,j]`: 256 reads. `active` and `absent` name the coordinates read; `lower` reads `X[r,c]` with
  `r >= c` and `upper` reads the same coordinates through `X[c,r]`. Four cases, so each is interpretable
  alone. With `s = 100`, the 256 active reads cycle over the 100 active coordinates.
- `write_X[i,j]_active`: 1024 writes to active coordinates (the working set does not change).
- `first_activation`: 256 **distinct** coordinates, each inactive just before the call. The restore runs outside
  the timed region and asserts that none is active; the check also asserts that all 256 are active afterwards.
- `sweep_activecoordinates`: one pass over `activecoordinates(X)` summing the IDs.
- `sweep_prox_coordinate!`: one `prox_coordinate!(X, NormL1(0.1), k, 0.5)` per active coordinate.
- `mul!`: `y = A v` for the owner (through `matrixview`), a `SparseMatrixCSC` with the same entries, and a dense `Symmetric` (p ≤ 2000).
- `Matrix`: dense copy. `construct`: an empty owner.

```text
case                    layout                    shape               work    median       min       bytes
Matrix                  owner                     p1000_s100       1000000    182.31     69.12     8011856
Matrix                  owner                     p1000_s1000      1000000     78.04     74.00     8011856
Matrix                  owner                     p1000_s10000     1000000    329.71    100.83     8011856
Matrix                  owner                     p1000_s100000    1000000    379.98    350.50     8011856
Matrix                  owner                     p2000_s100       4000000   1780.29    258.21    32014416
Matrix                  owner                     p4000_s100      16000000   1304.77   1081.58   128008272
Matrix                  owner                     p500_s100         250000    105.58     17.71     2015312
construct               empty                     p1000             500500    667.31    104.92    24035568
construct               empty                     p2000            2001000    501.79    406.42    96043248
construct               empty                     p4000            8002000   2094.88   1630.38   192086256
construct               empty                     p500              125250    159.56     26.96     3047664
first_activation        absent_distinct_batch     p1000_s100           256      0.29      0.25           0
first_activation        absent_distinct_batch     p1000_s1000          256      0.29      0.25           0
first_activation        absent_distinct_batch     p1000_s10000         256      0.29      0.29           0
first_activation        absent_distinct_batch     p1000_s100000        256      0.29      0.29           0
first_activation        absent_distinct_batch     p2000_s100           256      0.29      0.25           0
first_activation        absent_distinct_batch     p4000_s100           256      0.29      0.25           0
first_activation        absent_distinct_batch     p500_s100            256      0.29      0.25           0
mul!                    csc_baseline              p1000_s100           100      0.68      0.67           0
mul!                    csc_baseline              p1000_s1000         1000      1.90      1.67           0
mul!                    csc_baseline              p1000_s10000       10000      9.31      9.18           0
mul!                    csc_baseline              p1000_s100000     100000     72.08     71.75           0
mul!                    csc_baseline              p2000_s100           100      1.77      1.71           0
mul!                    csc_baseline              p4000_s100           100      3.63      3.56           0
mul!                    csc_baseline              p500_s100            100      0.38      0.37           0
mul!                    dense_symmetric_baseline  p1000_s100       1000000     66.21     65.85           0
mul!                    dense_symmetric_baseline  p1000_s1000      1000000     68.74     65.78           0
mul!                    dense_symmetric_baseline  p1000_s10000     1000000     66.18     65.81           0
mul!                    dense_symmetric_baseline  p1000_s100000    1000000     66.25     65.75           0
mul!                    dense_symmetric_baseline  p2000_s100       4000000    288.63    274.32           0
mul!                    dense_symmetric_baseline  p500_s100         250000     16.25     16.18           0
mul!                    owner                     p1000_s100           100      0.27      0.26           0
mul!                    owner                     p1000_s1000         1000      1.97      1.95           0
mul!                    owner                     p1000_s10000       10000     19.71     19.39           0
mul!                    owner                     p1000_s100000     100000    199.13    193.85           0
mul!                    owner                     p2000_s100           100      0.42      0.39           0
mul!                    owner                     p4000_s100           100      0.86      0.77           0
mul!                    owner                     p500_s100            100      0.24      0.23           0
read_X[i,j]             absent_lower              p1000_s100           256      0.18      0.17           0
read_X[i,j]             absent_lower              p1000_s1000          256      0.18      0.18           0
read_X[i,j]             absent_lower              p1000_s10000         256      0.18      0.18           0
read_X[i,j]             absent_lower              p1000_s100000        256      0.18      0.17           0
read_X[i,j]             absent_lower              p2000_s100           256      0.18      0.18           0
read_X[i,j]             absent_lower              p4000_s100           256      0.18      0.18           0
read_X[i,j]             absent_lower              p500_s100            256      0.17      0.17           0
read_X[i,j]             absent_upper              p1000_s100           256      0.18      0.17           0
read_X[i,j]             absent_upper              p1000_s1000          256      0.18      0.18           0
read_X[i,j]             absent_upper              p1000_s10000         256      0.17      0.17           0
read_X[i,j]             absent_upper              p1000_s100000        256      0.18      0.18           0
read_X[i,j]             absent_upper              p2000_s100           256      0.18      0.18           0
read_X[i,j]             absent_upper              p4000_s100           256      0.18      0.18           0
read_X[i,j]             absent_upper              p500_s100            256      0.17      0.17           0
read_X[i,j]             active_lower              p1000_s100           256      0.21      0.20           0
read_X[i,j]             active_lower              p1000_s1000          256      0.22      0.21           0
read_X[i,j]             active_lower              p1000_s10000         256      0.23      0.21           0
read_X[i,j]             active_lower              p1000_s100000        256      0.23      0.21           0
read_X[i,j]             active_lower              p2000_s100           256      0.22      0.20           0
read_X[i,j]             active_lower              p4000_s100           256      0.22      0.20           0
read_X[i,j]             active_lower              p500_s100            256      0.22      0.19           0
read_X[i,j]             active_upper              p1000_s100           256      0.20      0.20           0
read_X[i,j]             active_upper              p1000_s1000          256      0.23      0.21           0
read_X[i,j]             active_upper              p1000_s10000         256      0.23      0.22           0
read_X[i,j]             active_upper              p1000_s100000        256      0.23      0.21           0
read_X[i,j]             active_upper              p2000_s100           256      0.22      0.20           0
read_X[i,j]             active_upper              p4000_s100           256      0.22      0.20           0
read_X[i,j]             active_upper              p500_s100            256      0.20      0.20           0
sweep_activecoordinates trivial_update            p1000_s100           100      0.01      0.01           0
sweep_activecoordinates trivial_update            p1000_s1000         1000      0.06      0.06           0
sweep_activecoordinates trivial_update            p1000_s10000       10000      0.66      0.65           0
sweep_activecoordinates trivial_update            p1000_s100000     100000      7.94      7.75           0
sweep_activecoordinates trivial_update            p2000_s100           100      0.01      0.01           0
sweep_activecoordinates trivial_update            p4000_s100           100      0.01      0.01           0
sweep_activecoordinates trivial_update            p500_s100            100      0.01      0.01           0
sweep_prox_coordinate!  NormL1                    p1000_s100           100      0.35      0.33           0
sweep_prox_coordinate!  NormL1                    p1000_s1000         1000      3.50      3.42           0
sweep_prox_coordinate!  NormL1                    p1000_s10000       10000     36.96     35.38           0
sweep_prox_coordinate!  NormL1                    p1000_s100000     100000    380.06    374.08           0
sweep_prox_coordinate!  NormL1                    p2000_s100           100      0.38      0.33           0
sweep_prox_coordinate!  NormL1                    p4000_s100           100      0.33      0.29           0
sweep_prox_coordinate!  NormL1                    p500_s100            100      0.33      0.33           0
write_X[i,j]_active     active                    p1000_s100          1024      0.84      0.81           0
write_X[i,j]_active     active                    p1000_s1000         1024      0.96      0.92           0
write_X[i,j]_active     active                    p1000_s10000        1024      1.00      0.97           0
write_X[i,j]_active     active                    p1000_s100000       1024      1.08      0.94           0
write_X[i,j]_active     active                    p2000_s100          1024      0.87      0.84           0
write_X[i,j]_active     active                    p4000_s100          1024      0.84      0.83           0
write_X[i,j]_active     active                    p500_s100           1024      0.81      0.81           0
```

## Memory: three separate claims

1. **Retained size is `24q` bytes plus a header.** An empty owner holds one `Float64` array and two `Int`
   arrays of length `q = p(p+1)/2`. `construct` asserts `Base.summarysize(X)` equals `24q` within 1 KiB
   for every `p`, which is `48.0 MB` at `p = 2000` and `192.0 MB` at `p = 4000`.
2. **Allocated bytes of one construction are the same `24q` plus about 0.1%** when measured by itself:
   `@allocated` gives 48,070,896 B at `p = 2000` and 192,086,256 B at `p = 4000`, also after a `GC.gc()`,
   after other cases have run (any one family of cases, or all cases but this one), and in a direct run of the four
   construction cases.
3. **In the full `owner` group the `bytes` column for `construct` is not reliable.** It showed exactly
   twice the allocation (96,043,248 B at `p = 2000`, 384,024,816 B at `p = 4000`) in a first run; in this
   run it shows 2x at `p = 1000` and `p = 2000` and 1x at `p = 4000`. The doubling moves between sizes from run to run and
   disappears in every smaller reproduction, so it is an artifact of `@allocated` for single allocations of
   16 MB or more in a process that has run all cases, not extra allocation by the constructor
   (the source allocates three arrays). I did not find the mechanism in the Julia accounting.
   Do **not** use the `bytes` column of `construct` or `Matrix` as a memory gate; use the retained-size assertion in
   claim 1. The `bytes` column is reliable for the zero-allocation claims below, where the only question is
   whether it is zero.

## What the numbers say about the contract targets

- Reads, writes and first activation are flat in `p` (500 to 4000) and in `s` (10² to 10⁵), and allocate nothing.
  A read costs about 0.7 ns (absent) and 0.85 ns (active) and does not depend on the orientation.
  First activation is about 1 ns per coordinate, which an independent loop outside the harness reproduces (0.98 ns).
- A sweep is linear in `s`: `prox_coordinate!` is about 3.7 ns per coordinate, and the bare iteration about 0.08 ns.
- `mul!` on the owner is linear in `p` at fixed `s` and linear in `s` at fixed `p`, with no allocation.
  It is about 2.7 times slower than the CSC product at `s = 10⁵` and faster at `s ≤ 100`.
- `Matrix` and construction are quadratic in `p` in their `min`, as the contract expects.

## Still needed before the Phase 3 gate

- A deterministic end-to-end ProximalSolvers coordinate-descent baseline (`KronModel`, preferably `KronUModel`,
  including cache updates). `prox_coordinate!` here is a primitive; it does not measure a solver sweep.
- A second machine (Discovery), and Float32 and `p = 8000`.
