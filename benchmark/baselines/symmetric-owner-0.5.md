# SymmetricSparseIterate, 0.5.x baseline (issue #391, Phase 0)

The pre-refactor measurement required by the acceptance guide, section 4, and by the
Phase 3 gate of #391: *a coordinate-descent sweep must not be slower*. Re-run the same
harness on the new owner and compare row by row.

| | |
|:--|:--|
| Package | ProximalBase at `a73f99e5` (the 0.5.x owner; code identical to `abaceda25`, docs only since) |
| Julia | 1.13.3, `JULIA_NUM_THREADS=1`, default `--check-bounds` |
| CPU | Apple M3 Pro, macOS |
| Harness | `benchmark/` group `owner`, extended suite, `PB_REPETITION=1,2,3`, 40 samples of at least 0.1 s |

```sh
for r in 1 2 3; do
  PB_SUITE=extended PB_GROUPS=owner PB_REPETITION=$r PB_OUTPUT=/absolute/path \
    JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/run.jl
done
```

`median` is the median over the three repetitions of each repetition's median sample time;
`min` is the smallest sample over all repetitions. Times are in microseconds **per call of the
case** (see the work column below); `bytes` is the largest allocation per call.
A case with `evals = 1` (first activation, `Matrix`, construction) has page-fault and GC noise:
read its `min`, not its `median`. Three repetitions on one machine are an alarm, not a
confidence interval.

What one call does:

- `read_X[i,j]`: 1024 reads, alternating active and absent positions, both orientations.
- `write_X[i,j]_active`: 1024 writes to active coordinates (no change of the working set).
- `first_activation`: 256 distinct absent coordinates activated; the state is restored outside the timed region.
- `sweep_activecoordinates`: one pass over `activecoordinates(X)` summing the IDs.
- `sweep_prox_coordinate!`: one `prox_coordinate!(X, NormL1(0.1), k, 0.5)` per active coordinate.
- `mul!`: `y = A v` for the owner, a `SparseMatrixCSC` with the same entries, and a dense `Symmetric` (p ≤ 2000).
- `Matrix`: dense copy. `construct`: an empty owner.

```text

Matrix                  owner                     p1000_s100        490.96     70.88     8011856
Matrix                  owner                     p1000_s1000       434.33     75.79     8011856
Matrix                  owner                     p1000_s10000      423.67    114.38     8011856
Matrix                  owner                     p1000_s100000     444.35    369.67     8011856
Matrix                  owner                     p2000_s100        453.60    276.21    32014416
Matrix                  owner                     p4000_s100       1300.23   1109.04   128008272
Matrix                  owner                     p500_s100         111.67     17.92     2015312
construct               empty                     p1000             593.19    106.17    12042480
construct               empty                     p2000             573.17    399.83    96043248
construct               empty                     p4000            1999.50   1686.62   384024816
construct               empty                     p500              167.25     27.46     3047664
first_activation        absent_batch              p1000_s100          0.29      0.25           0
first_activation        absent_batch              p1000_s1000         0.29      0.25           0
first_activation        absent_batch              p1000_s10000        0.29      0.25           0
first_activation        absent_batch              p1000_s100000       0.29      0.29           0
first_activation        absent_batch              p2000_s100          0.29      0.25           0
first_activation        absent_batch              p4000_s100          0.29      0.25           0
first_activation        absent_batch              p500_s100           0.29      0.25           0
mul!                    csc_baseline              p1000_s100          0.69      0.68           0
mul!                    csc_baseline              p1000_s1000         1.70      1.66           0
mul!                    csc_baseline              p1000_s10000        9.46      9.17           0
mul!                    csc_baseline              p1000_s100000      73.50     73.13           0
mul!                    csc_baseline              p2000_s100          1.80      1.66           0
mul!                    csc_baseline              p4000_s100          3.77      3.67           0
mul!                    csc_baseline              p500_s100           0.38      0.38           0
mul!                    dense_symmetric_baseline  p1000_s100         69.68     66.68           0
mul!                    dense_symmetric_baseline  p1000_s1000        67.72     66.85           0
mul!                    dense_symmetric_baseline  p1000_s10000       68.93     67.25           0
mul!                    dense_symmetric_baseline  p1000_s100000      67.79     67.10           0
mul!                    dense_symmetric_baseline  p2000_s100        297.22    283.11           0
mul!                    dense_symmetric_baseline  p500_s100          16.58     16.52           0
mul!                    owner                     p1000_s100          0.29      0.27           0
mul!                    owner                     p1000_s1000         2.04      2.00           0
mul!                    owner                     p1000_s10000       20.13     20.04           0
mul!                    owner                     p1000_s100000     198.89    197.77           0
mul!                    owner                     p2000_s100          0.39      0.37           0
mul!                    owner                     p4000_s100          0.88      0.78           0
mul!                    owner                     p500_s100           0.24      0.23           0
read_X[i,j]             active_and_absent         p1000_s100          1.75      1.71           0
read_X[i,j]             active_and_absent         p1000_s1000         1.95      1.89           0
read_X[i,j]             active_and_absent         p1000_s10000        1.94      1.89           0
read_X[i,j]             active_and_absent         p1000_s100000       1.93      1.85           0
read_X[i,j]             active_and_absent         p2000_s100          1.83      1.72           0
read_X[i,j]             active_and_absent         p4000_s100          1.91      1.79           0
read_X[i,j]             active_and_absent         p500_s100           1.72      1.68           0
sweep_activecoordinates trivial_update            p1000_s100          0.01      0.01           0
sweep_activecoordinates trivial_update            p1000_s1000         0.06      0.06           0
sweep_activecoordinates trivial_update            p1000_s10000        0.67      0.66           0
sweep_activecoordinates trivial_update            p1000_s100000       8.10      7.86           0
sweep_activecoordinates trivial_update            p2000_s100          0.01      0.01           0
sweep_activecoordinates trivial_update            p4000_s100          0.01      0.01           0
sweep_activecoordinates trivial_update            p500_s100           0.01      0.01           0
sweep_prox_coordinate!  NormL1                    p1000_s100          0.38      0.33           0
sweep_prox_coordinate!  NormL1                    p1000_s1000         3.79      3.62           0
sweep_prox_coordinate!  NormL1                    p1000_s10000       36.40     35.79           0
sweep_prox_coordinate!  NormL1                    p1000_s100000     401.52    365.33           0
sweep_prox_coordinate!  NormL1                    p2000_s100          0.38      0.33           0
sweep_prox_coordinate!  NormL1                    p4000_s100          0.38      0.33           0
sweep_prox_coordinate!  NormL1                    p500_s100           0.38      0.33           0
write_X[i,j]_active     active                    p1000_s100          0.86      0.83           0
write_X[i,j]_active     active                    p1000_s1000         1.00      0.93           0
write_X[i,j]_active     active                    p1000_s10000        1.03      0.96           0
write_X[i,j]_active     active                    p1000_s100000       0.99      0.96           0
write_X[i,j]_active     active                    p2000_s100          0.88      0.84           0
write_X[i,j]_active     active                    p4000_s100          0.85      0.83           0
write_X[i,j]_active     active                    p500_s100           0.82      0.82           0
```

## What the numbers say about the contract targets

- Reads, writes and first activation are flat in `p` (500 to 4000) and in `s` (10² to 10⁵), and allocate nothing.
  First activation is about 1 ns per coordinate, which an independent loop outside the harness reproduces (0.98 ns).
- A sweep is linear in `s`: `prox_coordinate!` is about 3.7 ns per coordinate, and the bare iteration about 0.08 ns.
- `mul!` on the owner is linear in `p` at fixed `s` and linear in `s` at fixed `p`, with no allocation.
  It is about 2.7 times slower than the CSC product at `s = 10⁵` (199 µs versus 73.5 µs) and faster at `s ≤ 100`.
- `Matrix` and construction are quadratic in `p`, as the contract expects.
- The harness does **not** time first activation at `p = 8000` or above, nor Float32.
