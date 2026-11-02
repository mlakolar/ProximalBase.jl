# The iterate family, 0.5.x baseline (issue #420)

The pre-change measurement that the performance gate of the iterate-family design requires (FT24,
numerics C3): every 0.6 change on a hot path reports its cases against this baseline, in the same
session, with `benchmark/compare.py`. The cases are the `sparse` and `atom` groups, the strong-zero,
clear and momentum cases of the `owner` group, the existing `products` group and the existing
`group_layouts` cases (FD-11). The source is the audited 0.5.x commit, so that no FD is in the measured code.

## Identity of the measurement

| | |
|:--|:--|
| Comparison point | ProximalBase 0.5.x at `def81ec9d021c39ed1727cfeb9598357291dd60d`, the audited commit (the owner baseline's source too), with package source hash `a1583f070615bd3ea38705eacda4401ca1c6bafd` (`src/`, `ext/`, `Project.toml`), `source_dirty = false` |
| Harness | `benchmark/run.jl` and `benchmark/cases.jl` at commit `fc8426229ab622dfc83cc75293de3c3cfd1706cd` of this branch, hash `0986ce4e175c9c3fcbd031569b69702f921d3e77` (`harness_hash` in each metadata file) |
| Julia | 1.13.3, `JULIA_NUM_THREADS=1`, 1 BLAS thread (LBTConfig([ILP64] libopenblas64_.so)), `--check-bounds=default`, BenchmarkTools 1.8.0 |
| CPU and OS | Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz (`skylake-avx512`), Linux x86_64-linux-gnu |
| Settings | extended suite, `PB_REPETITION=1,2,3` in fresh processes, 40 samples of at most 0.1 s; timer floor 24–25 ns |
| Cases | 3719: 3455 in the groups run (`atom,owner,products,sparse`), 264 in the layouts run (`vectors` with `PB_CASES='^group_layouts/'`) |
| Limits | one machine and one operating system (the rows above); the owner baseline `symmetric-owner-0.5.md` was measured on a different machine, so its rows are not comparable row by row; three repetitions; no end-to-end solver sweep; no `p = 8·10⁶` |

```sh
julia benchmark/setup.jl /absolute/path/to/pinned benchmark/results/env-pinned
for r in 1 2 3; do
  PB_SUITE=extended PB_GROUPS=atom,owner,products,sparse PB_REPETITION=$r \
    PB_OUTPUT=benchmark/results/groups PB_EXPECTED_PACKAGE_ROOT=/absolute/path/to/pinned \
    JULIA_NUM_THREADS=1 julia --project=benchmark/results/env-pinned benchmark/run.jl
  PB_SUITE=extended PB_GROUPS=vectors PB_CASES='^group_layouts/' PB_REPETITION=$r \
    PB_OUTPUT=benchmark/results/layouts PB_EXPECTED_PACKAGE_ROOT=/absolute/path/to/pinned \
    JULIA_NUM_THREADS=1 julia --project=benchmark/results/env-pinned benchmark/run.jl
done
python3 benchmark/validate.py benchmark/results/groups benchmark/results/groups/validation.json --expected-runs 3
python3 benchmark/validate.py benchmark/results/layouts benchmark/results/layouts/validation.json --expected-runs 3
```

Any edit of `benchmark/run.jl` or `benchmark/cases.jl` changes the harness hash: measure the pinned
source again with the new harness, in the same session as the revision you compare, before using
these numbers as a gate.

## How to read the tables

`median` is the median over the three repetitions of each repetition's median sample time; `min` is
the smallest sample over all repetitions. Times are in microseconds **per call**; `work` is the number of
operations in one call, and the per-work figure is the median divided by it. `spread` is the largest
repetition median over the smallest minus one. Each repetition draws its own inputs (`PB_REPETITION` is a seed), so the
spread mixes the machine's noise with the effect of the inputs: it bounds the noise from above. **A case whose spread
exceeds 5% is not a safe gate for a 5% slowdown on this machine**, and a comparison must pair the
same repetition numbers on both sides, as `compare.py` does. The noise with the inputs held fixed was measured on a single earlier repetition 1 of the same sources on the same machine type (the canary run, harness before the review fixes; `atomindices_loop` left out because its operation changed): of 3407 cases, 65% agreed within 5%, 83% within 10% and 94% within 20% (median deviation 2.8%). In 104 cases the input draw changes the number of
operations in a call between repetitions (all of them in the `products` group, where the draw decides how many entries of an operand are active): `work` is then the median over the
repetitions, the per-work figure the median of the per-repetition ratios, `work_varies` is `true` in the CSV, and the spread includes the change of work. A case with `evals = 1` carries page-fault and GC
noise, so read its `min`.
`bytes` is the largest `@allocated` of one warm call; it is reliable as "zero or not" and not as a size for
large allocations (see #401). A row is timer-limited when `median · evals < 10 · timer floor`: it cannot gate.
The full table, one line per case, is in `iterate-family-0.5.csv` (the same directory); the tables below
group the cases by family, case and layout, over both element types and every fixture.

Spellings that differ between 0.5.x and 0.6 are chosen when the case is built: on 0.5.x, `atomindices(x, k)` is
`x.groups[k]`, `samegroups` is `_samelayout`, and `coordinates(x)` is `x` itself for a `SparseIterate`, so the
`via_coordinates` and `via_x` rows coincide. See the README of this directory for the rows that need a note.

## Cases by family

| family | case | layout | cases | median µs (range) | per work unit ns (range) | max spread | max bytes | timer-limited |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| atom | `_nvalues` | `accessors` | 48 | 0.0143–492 | 0.915–7.64 | 47.3% | 0 | 0 |
| atom | `_nvalues` | `private_fields` | 48 | 0.0115–489 | 0.664–7.75 | 66.6% | 0 | 0 |
| atom | `_restartdot` | `accessors` | 48 | 1.08–2.04e+03 | 1.36–25.5 | 18.4% | 0 | 0 |
| atom | `_restartdot` | `private_fields` | 48 | 1.08–1.9e+03 | 1.37–23.7 | 17.6% | 0 | 0 |
| atom | `_setvalues!` | `accessors` | 48 | 1.5–5.55e+03 | 1.95–69.3 | 23.8% | 0 | 0 |
| atom | `_setvalues!` | `private_fields` | 48 | 0.0791–1.29e+03 | 0.103–48.9 | 67.7% | 0 | 0 |
| atom | `_values!` | `accessors` | 48 | 0.0971–1.34e+03 | 0.126–16.7 | 31.3% | 0 | 0 |
| atom | `_values!` | `private_fields` | 48 | 0.0826–1.26e+03 | 0.108–15.7 | 37.4% | 0 | 0 |
| atom | `atom_dot` | `atoms` | 48 | 0.402–7.21 | 0.213–14.2 | 65.7% | 0 | 0 |
| atom | `atom_norm` | `atoms` | 48 | 0.766–15.5 | 0.546–20.5 | 22.7% | 0 | 0 |
| atom | `atomindices_loop` | `active_groups` | 48 | 0.1–711 | 0.112–8.88 | 19.6% | 0 | 0 |
| atom | `atomindices_view` | `view_range_group` | 24 | 0.266–700 | 0.346–8.75 | 21.5% | 0 | 0 |
| atom | `atomindices_view` | `view_vector_group` | 24 | 1.16–2.59e+03 | 1.51–32.3 | 13.8% | 0 | 0 |
| atom | `broadcast_assign` | `atoms` | 48 | 0.356–1.83e+04 | 0.00445–22.8 | 28.8% | 0 | 0 |
| atom | `broadcast_self` | `abs` | 48 | 68.7–2.15e+03 | 0.858–2.69 | 39.3% | 0 | 0 |
| atom | `construct` | `empty` | 24 | 319–5.12e+04 | 3.98–65.4 | 143.7% | 83200424 | 0 |
| atom | `copy` | `atoms` | 48 | 54.6–3.76e+03 | 0.682–5.23 | 244.8% | 19200280 | 0 |
| atom | `copyto!_convert` | `to_Float32` | 24 | 0.403–1.38e+04 | 0.525–173 | 23.7% | 0 | 0 |
| atom | `copyto!_convert` | `to_Float64` | 24 | 0.397–1.56e+04 | 0.517–195 | 26.0% | 0 | 0 |
| atom | `copyto!_convert` | `to_Int64` | 24 | 1.67–1.57e+04 | 2.18–196 | 17.8% | 0 | 0 |
| atom | `copyto!_dense` | `dense_other_entries` | 48 | 71.1–2.82e+03 | 0.889–3.53 | 32.2% | 0 | 0 |
| atom | `copyto!_same_kind` | `same_entries` | 48 | 0.308–1.59e+04 | 0.201–99.1 | 32.9% | 0 | 0 |
| atom | `copyto!_state` | `atoms` | 48 | 0.361–1.83e+04 | 0.235–115 | 28.8% | 0 | 0 |
| atom | `copyto!_unrelated` | `dense_same_entries` | 48 | 70–2.05e+03 | 0.875–2.56 | 32.2% | 0 | 0 |
| atom | `dot_vx` | `z0` | 48 | 1.05–1.65e+03 | 1.34–20.6 | 18.1% | 0 | 0 |
| atom | `dot_vx` | `z50` | 48 | 1.06–1.63e+03 | 1.34–20.4 | 28.4% | 0 | 0 |
| atom | `dot_xy` | `atom_atom_z0` | 48 | 1.05–1.65e+03 | 1.34–20.6 | 24.8% | 0 | 0 |
| atom | `dot_xy` | `atom_atom_z50` | 48 | 1.05–1.66e+03 | 1.34–20.7 | 17.3% | 0 | 0 |
| atom | `dot_xy` | `atom_sparse_z0` | 48 | 1.08–1.87e+03 | 1.35–23.3 | 17.8% | 0 | 0 |
| atom | `dot_xy` | `atom_sparse_z50` | 48 | 1.08–1.79e+03 | 1.35–22.4 | 28.5% | 0 | 0 |
| atom | `dropzeros!` | `zeros10` | 48 | 0.097–4.44e+03 | 0.126–55.5 | 21.9% | 0 | 4 |
| atom | `dropzeros!` | `zeros50` | 48 | 0.346–5.85e+03 | 0.451–73.2 | 86.8% | 0 | 0 |
| atom | `fill!_zero` | `atoms` | 48 | 0.105–4.15e+03 | 0.137–63.8 | 76.9% | 0 | 2 |
| atom | `fill!_zero_dropzeros` | `atoms` | 48 | 0.108–4.11e+03 | 0.141–63.7 | 70.3% | 0 | 2 |
| atom | `first_activation` | `absent_distinct_batch` | 48 | 0.824–7.84 | 3.22–30.6 | 185.8% | 0 | 0 |
| atom | `first_activation` | `absent_distinct_batch_x16` | 40 | 18.7–91.8 | 4.56–22.4 | 60.7% | 0 | 0 |
| atom | `first_activation` | `absent_distinct_batch_x4` | 8 | 3.37–6.05 | 3.29–5.91 | 16.6% | 0 | 0 |
| atom | `first_activation` | `activate_distinct_batch` | 48 | 0.39–0.532 | 1.52–2.08 | 28.4% | 0 | 0 |
| atom | `first_activation` | `activate_distinct_batch_x16` | 40 | 7–8.24 | 1.71–2.01 | 10.7% | 0 | 0 |
| atom | `first_activation` | `activate_distinct_batch_x4` | 8 | 1.49–1.7 | 1.46–1.66 | 9.6% | 0 | 0 |
| atom | `isactive` | `half_active` | 48 | 0.187–0.232 | 0.729–0.906 | 21.4% | 0 | 0 |
| atom | `mul!` | `rows1024_z0` | 24 | 147–6.76e+03 | 0.187–0.825 | 15.7% | 0 | 0 |
| atom | `mul!` | `rows1024_z50` | 24 | 70.6–3.62e+03 | 0.0898–0.442 | 15.5% | 0 | 0 |
| atom | `mul!` | `rows256_z0` | 24 | 17.6–2.08e+03 | 0.0896–1.02 | 25.0% | 0 | 0 |
| atom | `mul!` | `rows256_z50` | 24 | 9.38–1.01e+03 | 0.0477–0.492 | 21.5% | 0 | 0 |
| atom | `mul!` | `symmetric_z0` | 6 | 174–274 | 1.67–2.32 | 1243.3% | 0 | 0 |
| atom | `mul!` | `symmetric_z50` | 6 | 87.3–137 | 0.81–1.23 | 930.3% | 0 | 0 |
| atom | `prox_coordinate!_5arg` | `GroupNormL2_strided_u` | 48 | 6.48–74.6 | 1.58–291 | 32.1% | 16384 | 0 |
| atom | `prox_coordinate!_5arg` | `GroupNormLinf_strided_u` | 48 | 15–78.2 | 7.41–305 | 30.9% | 16384 | 0 |
| atom | `prox_coordinate!_5arg` | `NormL1_strided_u` | 48 | 5.65–44.9 | 1.38–175 | 32.1% | 0 | 0 |
| atom | `prox_coordinate!_5arg` | `SparseGroupNormL2_strided_u` | 48 | 15.1–81.9 | 3.79–320 | 28.6% | 16384 | 0 |
| atom | `prox_coordinate!_staged` | `GroupNormL2_strided_u` | 48 | 4.22–62.6 | 1.03–235 | 58.6% | 0 | 0 |
| atom | `prox_coordinate!_staged` | `GroupNormLinf_strided_u` | 48 | 9.77–73.8 | 7–267 | 35.8% | 0 | 0 |
| atom | `prox_coordinate!_staged` | `NormL1_strided_u` | 48 | 5.29–46.4 | 1.29–181 | 45.6% | 0 | 0 |
| atom | `prox_coordinate!_staged` | `SparseGroupNormL2_strided_u` | 48 | 7.62–69.8 | 3.29–233 | 64.8% | 0 | 0 |
| atom | `read_x[i]` | `absent` | 48 | 0.317–0.332 | 1.24–1.3 | 11.0% | 0 | 0 |
| atom | `read_x[i]` | `active` | 48 | 0.313–0.332 | 1.22–1.3 | 9.2% | 0 | 0 |
| atom | `replace_cheap` | `atoms` | 48 | 71.8–7.22e+03 | 0.897–9.02 | 33.7% | 0 | 0 |
| atom | `replace_staged` | `atoms` | 48 | 113–1.97e+04 | 1.41–24.7 | 103.2% | 19200280 | 0 |
| atom | `rmul!_finite` | `x1` | 48 | 68.3–2.21e+03 | 0.854–2.76 | 19.9% | 0 | 0 |
| atom | `rmul!_inf` | `x1` | 48 | 161–3.11e+03 | 2.02–3.89 | 9.9% | 0 | 0 |
| atom | `samegroups` | `alternating` | 12 | 3.92–1.67e+04 | 1.55–63.9 | 9.1% | 0 | 0 |
| atom | `samegroups` | `distinct_first_seen` | 12 | 8.1–8.67e+03 | 1.59–54.3 | 24.0% | 0 | 0 |
| atom | `samegroups` | `shared` | 12 | 0.00453–0.00488 | 4.53–4.88 | 16.5% | 0 | 0 |
| atom | `sweep_active_update` | `trivial_update` | 48 | 1.36–7.11e+03 | 1.77–88.9 | 39.5% | 0 | 0 |
| atom | `sweep_activecoordinates` | `atoms` | 48 | 0.00834–7.9 | 0.0881–0.763 | 32.2% | 0 | 0 |
| atom | `write_x[i]_active` | `active` | 48 | 2.15–3.56 | 2.1–3.48 | 22.2% | 0 | 0 |
| group_layouts | `GroupNormL2/coordinate` | `indices_distinct` | 4 | 0.079–0.141 | 9.88–17.6 | 96.6% | 0 | 4 |
| group_layouts | `GroupNormL2/coordinate` | `indices_shared` | 4 | 0.08–0.083 | 10–10.4 | 6.2% | 0 | 4 |
| group_layouts | `GroupNormL2/coordinate` | `ranges_distinct` | 4 | 0.064–0.068 | 8–8.5 | 43.2% | 0 | 4 |
| group_layouts | `GroupNormL2/coordinate` | `ranges_shared` | 4 | 0.063–0.068 | 7.88–8.5 | 9.1% | 0 | 4 |
| group_layouts | `GroupNormL2/kkt` | `indices_distinct` | 4 | 0.0204–0.023 | 2.54–2.87 | 31.5% | 0 | 0 |
| group_layouts | `GroupNormL2/kkt` | `indices_shared` | 4 | 0.0202–0.023 | 2.53–2.88 | 6.3% | 0 | 0 |
| group_layouts | `GroupNormL2/kkt` | `ranges_distinct` | 4 | 0.0156–0.0163 | 1.95–2.04 | 31.9% | 0 | 0 |
| group_layouts | `GroupNormL2/kkt` | `ranges_shared` | 4 | 0.0153–0.016 | 1.91–2 | 30.5% | 0 | 0 |
| group_layouts | `GroupNormL2/prox!` | `indices_distinct` | 4 | 0.519–0.543 | 8.12–8.48 | 3.0% | 0 | 0 |
| group_layouts | `GroupNormL2/prox!` | `indices_shared` | 4 | 0.518–0.546 | 8.09–8.53 | 2.2% | 0 | 0 |
| group_layouts | `GroupNormL2/prox!` | `ranges_distinct` | 4 | 0.328–0.361 | 5.12–5.64 | 17.3% | 0 | 0 |
| group_layouts | `GroupNormL2/prox!` | `ranges_shared` | 4 | 0.326–0.357 | 5.09–5.58 | 17.4% | 0 | 0 |
| group_layouts | `GroupNormL2/value` | `indices_distinct` | 4 | 0.123–0.142 | 1.93–2.22 | 10.6% | 0 | 0 |
| group_layouts | `GroupNormL2/value` | `indices_shared` | 4 | 0.124–0.144 | 1.94–2.25 | 9.9% | 0 | 0 |
| group_layouts | `GroupNormL2/value` | `ranges_distinct` | 4 | 0.102–0.127 | 1.59–1.99 | 6.9% | 0 | 0 |
| group_layouts | `GroupNormL2/value` | `ranges_shared` | 4 | 0.102–0.123 | 1.6–1.93 | 4.2% | 0 | 0 |
| group_layouts | `GroupNormL2/value_alternating` | `indices_distinct` | 4 | 10.5–259 | 82.2–2.02e+03 | 3.1% | 0 | 0 |
| group_layouts | `GroupNormL2/value_alternating` | `ranges_distinct` | 4 | 3.39–31.1 | 26.5–243 | 1.6% | 0 | 0 |
| group_layouts | `GroupNormL2/value_and_prox!` | `indices_distinct` | 4 | 0.523–0.545 | 8.17–8.51 | 5.6% | 0 | 0 |
| group_layouts | `GroupNormL2/value_and_prox!` | `indices_shared` | 4 | 0.52–0.545 | 8.13–8.52 | 3.2% | 0 | 0 |
| group_layouts | `GroupNormL2/value_and_prox!` | `ranges_distinct` | 4 | 0.337–0.351 | 5.26–5.48 | 14.4% | 0 | 0 |
| group_layouts | `GroupNormL2/value_and_prox!` | `ranges_shared` | 4 | 0.329–0.344 | 5.14–5.38 | 14.0% | 0 | 0 |
| group_layouts | `GroupNormLinf/coordinate` | `indices_distinct` | 4 | 0.117–0.123 | 14.6–15.4 | 10.7% | 0 | 4 |
| group_layouts | `GroupNormLinf/coordinate` | `indices_shared` | 4 | 0.116–0.12 | 14.5–15 | 5.9% | 0 | 4 |
| group_layouts | `GroupNormLinf/coordinate` | `ranges_distinct` | 4 | 0.104–0.116 | 13–14.5 | 9.8% | 0 | 4 |
| group_layouts | `GroupNormLinf/coordinate` | `ranges_shared` | 4 | 0.104–0.117 | 13–14.6 | 12.9% | 0 | 4 |
| group_layouts | `GroupNormLinf/kkt` | `indices_distinct` | 4 | 0.0195–0.0201 | 2.44–2.51 | 7.3% | 0 | 0 |
| group_layouts | `GroupNormLinf/kkt` | `indices_shared` | 4 | 0.0189–0.0194 | 2.36–2.42 | 23.2% | 0 | 0 |
| group_layouts | `GroupNormLinf/kkt` | `ranges_distinct` | 4 | 0.0136–0.0146 | 1.69–1.82 | 20.6% | 0 | 0 |
| group_layouts | `GroupNormLinf/kkt` | `ranges_shared` | 4 | 0.0135–0.0146 | 1.69–1.82 | 29.6% | 0 | 0 |
| group_layouts | `GroupNormLinf/prox!` | `indices_distinct` | 4 | 0.838–0.876 | 13.1–13.7 | 7.3% | 0 | 0 |
| group_layouts | `GroupNormLinf/prox!` | `indices_shared` | 4 | 0.836–0.863 | 13.1–13.5 | 7.8% | 0 | 0 |
| group_layouts | `GroupNormLinf/prox!` | `ranges_distinct` | 4 | 0.673–0.697 | 10.5–10.9 | 7.0% | 0 | 0 |
| group_layouts | `GroupNormLinf/prox!` | `ranges_shared` | 4 | 0.677–0.692 | 10.6–10.8 | 7.4% | 0 | 0 |
| group_layouts | `GroupNormLinf/value` | `indices_distinct` | 4 | 0.194–0.224 | 3.03–3.51 | 5.5% | 0 | 0 |
| group_layouts | `GroupNormLinf/value` | `indices_shared` | 4 | 0.194–0.226 | 3.03–3.53 | 5.5% | 0 | 0 |
| group_layouts | `GroupNormLinf/value` | `ranges_distinct` | 4 | 0.112–0.133 | 1.74–2.08 | 4.4% | 0 | 0 |
| group_layouts | `GroupNormLinf/value` | `ranges_shared` | 4 | 0.112–0.134 | 1.75–2.09 | 3.2% | 0 | 0 |
| group_layouts | `GroupNormLinf/value_alternating` | `indices_distinct` | 4 | 10.6–252 | 83–1.97e+03 | 7.3% | 0 | 0 |
| group_layouts | `GroupNormLinf/value_alternating` | `ranges_distinct` | 4 | 3.41–31.6 | 26.7–247 | 1.7% | 0 | 0 |
| group_layouts | `GroupNormLinf/value_and_prox!` | `indices_distinct` | 4 | 0.844–0.866 | 13.2–13.5 | 4.8% | 0 | 0 |
| group_layouts | `GroupNormLinf/value_and_prox!` | `indices_shared` | 4 | 0.835–0.872 | 13–13.6 | 7.4% | 0 | 0 |
| group_layouts | `GroupNormLinf/value_and_prox!` | `ranges_distinct` | 4 | 0.673–0.696 | 10.5–10.9 | 10.0% | 0 | 0 |
| group_layouts | `GroupNormLinf/value_and_prox!` | `ranges_shared` | 4 | 0.675–0.698 | 10.5–10.9 | 10.6% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/coordinate` | `indices_distinct` | 4 | 0.0875–0.164 | 10.9–20.4 | 146.4% | 0 | 4 |
| group_layouts | `SparseGroupNormL2/coordinate` | `indices_shared` | 4 | 0.088–0.094 | 11–11.8 | 7.9% | 0 | 4 |
| group_layouts | `SparseGroupNormL2/coordinate` | `ranges_distinct` | 4 | 0.072–0.085 | 9–10.6 | 29.9% | 0 | 4 |
| group_layouts | `SparseGroupNormL2/coordinate` | `ranges_shared` | 4 | 0.072–0.079 | 9–9.88 | 5.6% | 0 | 4 |
| group_layouts | `SparseGroupNormL2/kkt` | `indices_distinct` | 4 | 0.0275–0.0322 | 3.43–4.03 | 19.7% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/kkt` | `indices_shared` | 4 | 0.0278–0.0304 | 3.48–3.8 | 18.0% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/kkt` | `ranges_distinct` | 4 | 0.0235–0.0259 | 2.94–3.24 | 7.8% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/kkt` | `ranges_shared` | 4 | 0.0232–0.0247 | 2.91–3.08 | 11.3% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/prox!` | `indices_distinct` | 4 | 0.681–0.703 | 10.6–11 | 5.4% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/prox!` | `indices_shared` | 4 | 0.675–0.697 | 10.5–10.9 | 3.5% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/prox!` | `ranges_distinct` | 4 | 0.526–0.537 | 8.21–8.39 | 5.3% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/prox!` | `ranges_shared` | 4 | 0.522–0.539 | 8.16–8.43 | 4.7% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value` | `indices_distinct` | 4 | 0.144–0.155 | 2.25–2.42 | 5.5% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value` | `indices_shared` | 4 | 0.148–0.156 | 2.32–2.43 | 7.3% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value` | `ranges_distinct` | 4 | 0.112–0.128 | 1.75–2 | 5.0% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value` | `ranges_shared` | 4 | 0.112–0.124 | 1.75–1.94 | 6.7% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value_alternating` | `indices_distinct` | 4 | 10.5–252 | 82.1–1.97e+03 | 4.4% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value_alternating` | `ranges_distinct` | 4 | 3.42–31.6 | 26.7–247 | 3.4% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `indices_distinct` | 4 | 0.682–0.704 | 10.6–11 | 5.8% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `indices_shared` | 4 | 0.678–0.697 | 10.6–10.9 | 3.3% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `ranges_distinct` | 4 | 0.522–0.536 | 8.15–8.38 | 2.9% | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `ranges_shared` | 4 | 0.522–0.539 | 8.15–8.42 | 4.2% | 0 | 0 |
| owner | `Matrix` | `owner` | 7 | 198–3.15e+04 | 0.792–2.64 | 172.0% | 128004112 | 0 |
| owner | `_restartdot` | `accessors` | 7 | 0.129–152 | 1.29–1.52 | 9.7% | 0 | 0 |
| owner | `_restartdot` | `private_fields` | 7 | 0.127–151 | 1.27–1.51 | 9.9% | 0 | 0 |
| owner | `_setvalues!` | `accessors` | 7 | 0.0257–70.1 | 0.123–0.701 | 21.2% | 0 | 0 |
| owner | `_setvalues!` | `private_fields` | 7 | 0.0154–35.5 | 0.0913–0.355 | 33.4% | 0 | 0 |
| owner | `_values!` | `accessors` | 7 | 0.0206–69.8 | 0.0989–0.698 | 13.5% | 0 | 0 |
| owner | `_values!` | `private_fields` | 7 | 0.0161–37 | 0.0903–0.37 | 32.7% | 0 | 0 |
| owner | `construct` | `empty` | 4 | 322–4.91e+04 | 2.43–6.13 | 177.8% | 192049200 | 0 |
| owner | `fill!_coordinates_zero` | `owner` | 7 | 51.4–7.25e+03 | 0.41–2.79 | 172.7% | 0 | 0 |
| owner | `fill!_coordinates_zero_dropzeros` | `owner` | 7 | 52.7–7.1e+03 | 0.421–4.15 | 15.6% | 0 | 0 |
| owner | `fill!_zero` | `owner` | 7 | 0.0655–378 | 0.655–3.78 | 15.4% | 0 | 4 |
| owner | `first_activation` | `absent_distinct_batch` | 7 | 0.926–1.02 | 3.62–3.99 | 100.3% | 0 | 0 |
| owner | `mul!` | `csc_baseline` | 7 | 0.971–192 | 1.92–66.3 | 10.8% | 0 | 0 |
| owner | `mul!` | `dense_symmetric_baseline` | 6 | 35.6–1.02e+03 | 0.142–0.254 | 9.0% | 0 | 0 |
| owner | `mul!` | `owner` | 7 | 0.901–822 | 8.22–14.7 | 34.4% | 0 | 0 |
| owner | `mul!` | `owner_z50` | 7 | 0.872–814 | 8.04–12.7 | 29.5% | 0 | 0 |
| owner | `read_X[i,j]` | `absent_lower` | 7 | 0.508–0.536 | 1.99–2.09 | 12.9% | 0 | 0 |
| owner | `read_X[i,j]` | `absent_upper` | 7 | 0.506–0.534 | 1.98–2.09 | 9.1% | 0 | 0 |
| owner | `read_X[i,j]` | `active_lower` | 7 | 0.537–0.608 | 2.1–2.38 | 8.5% | 0 | 0 |
| owner | `read_X[i,j]` | `active_upper` | 7 | 0.537–0.577 | 2.1–2.25 | 11.6% | 0 | 0 |
| owner | `sweep_activecoordinates` | `trivial_update` | 7 | 0.0142–9.43 | 0.0897–0.149 | 32.4% | 0 | 0 |
| owner | `sweep_prox_coordinate!` | `NormL1` | 7 | 1.03–1.52e+03 | 10.2–15.2 | 4.5% | 0 | 0 |
| owner | `write_X[i,j]_active` | `active` | 7 | 2.14–2.4 | 2.09–2.35 | 25.5% | 0 | 0 |
| products | `strided_atom/mul!` | `atom_0.01` | 4 | 0.124–0.956 | 0.102–0.201 | 264.9% | 0 | 0 |
| products | `strided_atom/mul!` | `atom_0.2` | 4 | 2.62–24.9 | 0.099–0.244 | 45.1% | 0 | 0 |
| products | `strided_atom/mul!_scaled` | `atom_0.01` | 4 | 0.152–1.27 | 0.115–0.311 | 228.8% | 0 | 1 |
| products | `strided_atom/mul!_scaled` | `atom_0.2` | 4 | 2.75–24.9 | 0.0947–0.259 | 57.1% | 0 | 0 |
| products | `strided_matrix/mul!` | `sparse_0.01` | 4 | 0.671–4.01 | 0.119–0.23 | 87.8% | 0 | 0 |
| products | `strided_matrix/mul!` | `sparse_0.2` | 4 | 10.5–163 | 0.103–0.374 | 21.2% | 0 | 0 |
| products | `strided_matrix/mul!_scaled` | `sparse_0.01` | 4 | 0.759–4.09 | 0.153–0.399 | 135.2% | 0 | 0 |
| products | `strided_matrix/mul!_scaled` | `sparse_0.2` | 4 | 10.7–165 | 0.104–0.382 | 22.0% | 0 | 0 |
| products | `strided_vector/mul!` | `sparse_0.01` | 4 | 0.104–0.757 | 0.0818–0.199 | 77.7% | 0 | 0 |
| products | `strided_vector/mul!` | `sparse_0.2` | 4 | 2.47–22.6 | 0.0858–0.23 | 32.7% | 0 | 0 |
| products | `strided_vector/mul!_scaled` | `sparse_0.01` | 4 | 0.136–0.797 | 0.0896–0.312 | 187.0% | 0 | 1 |
| products | `strided_vector/mul!_scaled` | `sparse_0.2` | 4 | 2.3–23.8 | 0.0793–0.265 | 61.1% | 0 | 0 |
| products | `switch_matrix/mul!` | `random_0.02` | 4 | 1.41–62.4 | 0.109–0.461 | 31.5% | 0 | 0 |
| products | `switch_matrix/mul!` | `random_0.05` | 4 | 5.69–166 | 0.187–0.562 | 30.6% | 0 | 0 |
| products | `switch_matrix/mul!` | `random_0.1` | 4 | 7.69–355 | 0.185–0.392 | 22.2% | 0 | 0 |
| products | `symmetric_L` | `atom_0.01` | 4 | 0.26–23.2 | 203–2.58e+03 | 2416.3% | 0 | 1 |
| products | `symmetric_L` | `atom_0.2` | 4 | 8.65–457 | 173–2.45e+03 | 35.4% | 0 | 0 |
| products | `symmetric_L` | `sparse_0.01` | 4 | 0.254–23.1 | 195–2.56e+03 | 2468.7% | 0 | 1 |
| products | `symmetric_L` | `sparse_0.2` | 4 | 8.12–453 | 161–2.43e+03 | 38.2% | 0 | 0 |
| products | `symmetric_U` | `atom_0.01` | 4 | 0.46–24.5 | 182–2.72e+03 | 990.1% | 0 | 0 |
| products | `symmetric_U` | `atom_0.2` | 4 | 7.48–484 | 174–2.52e+03 | 33.3% | 0 | 0 |
| products | `symmetric_U` | `sparse_0.01` | 4 | 0.427–24.3 | 179–2.7e+03 | 947.0% | 0 | 0 |
| products | `symmetric_U` | `sparse_0.2` | 4 | 7.13–478 | 166–2.45e+03 | 27.1% | 0 | 0 |
| products | `transposed_vector/mul!` | `sparse_0.01` | 4 | 0.879–14.7 | 0.88–2.87 | 226.4% | 0 | 0 |
| products | `transposed_vector/mul!` | `sparse_0.2` | 4 | 13.9–157 | 0.532–1.59 | 24.5% | 0 | 0 |
| products | `transposed_vector/mul!_scaled` | `sparse_0.01` | 4 | 1.19–14.7 | 1.07–2.88 | 230.5% | 0 | 0 |
| products | `transposed_vector/mul!_scaled` | `sparse_0.2` | 4 | 15.2–157 | 0.583–1.59 | 53.1% | 0 | 0 |
| sparse | `_restartdot` | `accessors` | 8 | 0.128–13.4 | 1.28–1.4 | 1.3% | 0 | 0 |
| sparse | `_restartdot` | `private_fields` | 8 | 0.127–13.4 | 1.27–1.4 | 7.2% | 0 | 0 |
| sparse | `_setvalues!` | `accessors` | 8 | 0.0226–2.1 | 0.0592–0.27 | 8.7% | 0 | 0 |
| sparse | `_setvalues!` | `private_fields` | 8 | 0.0126–3.07 | 0.0538–0.307 | 49.7% | 0 | 0 |
| sparse | `_values!` | `accessors` | 8 | 0.0166–2.09 | 0.0542–0.215 | 17.7% | 0 | 0 |
| sparse | `_values!` | `private_fields` | 8 | 0.0112–2.98 | 0.0492–0.298 | 22.3% | 0 | 0 |
| sparse | `broadcast_assign` | `vector` | 8 | 0.176–76.7 | 0.000196–0.0767 | 31.5% | 0 | 4 |
| sparse | `broadcast_self` | `abs` | 8 | 11.5–1.18e+03 | 1.02–1.2 | 6.6% | 0 | 0 |
| sparse | `construct` | `empty` | 4 | 32.7–2.97e+03 | 2.46–5.21 | 81.6% | 24000264 | 0 |
| sparse | `copy` | `vector` | 8 | 30.3–3.09e+03 | 2.42–4.87 | 303.3% | 24000264 | 0 |
| sparse | `copyto!_activevalues` | `slot_order` | 8 | 0.0969–10.5 | 0.747–1.25 | 52.6% | 0 | 0 |
| sparse | `copyto!_convert` | `to_Float32_dense_branch` | 2 | 125–531 | 1.95–2.07 | 0.9% | 0 | 0 |
| sparse | `copyto!_convert` | `to_Float32_sparse_branch` | 4 | 0.145–58.9 | 1.45–5.89 | 4.4% | 0 | 0 |
| sparse | `copyto!_convert` | `to_Float64_dense_branch` | 2 | 129–593 | 2.02–2.32 | 3.2% | 0 | 0 |
| sparse | `copyto!_convert` | `to_Float64_sparse_branch` | 4 | 0.145–61.6 | 1.45–6.16 | 5.5% | 0 | 0 |
| sparse | `copyto!_convert` | `to_Int64_dense_branch` | 2 | 210–876 | 3.28–3.42 | 1.7% | 0 | 0 |
| sparse | `copyto!_convert` | `to_Int64_sparse_branch` | 4 | 0.285–55.3 | 2.85–5.53 | 31.4% | 0 | 0 |
| sparse | `copyto!_dense` | `dense_other_nonzeros` | 8 | 11.2–1.31e+03 | 1.01–1.31 | 12.6% | 0 | 0 |
| sparse | `copyto!_same_kind` | `same_entries` | 8 | 0.145–59.7 | 0.724–2.99 | 37.2% | 0 | 0 |
| sparse | `copyto!_state` | `vector` | 8 | 0.172–76.1 | 0.86–3.81 | 44.9% | 0 | 4 |
| sparse | `copyto!_unrelated` | `dense_same_nonzeros` | 8 | 10.8–1.21e+03 | 1.03–1.21 | 8.2% | 0 | 0 |
| sparse | `dot_aXb` | `z0` | 12 | 1.84–308 | 5.5–6.17 | 17.3% | 0 | 0 |
| sparse | `dot_aXb` | `z50` | 12 | 1.84–308 | 5.5–6.28 | 12.9% | 0 | 0 |
| sparse | `dot_via_coordinates` | `rank1` | 8 | 0.0495–8.39 | 0.495–0.839 | 30.9% | 0 | 0 |
| sparse | `dot_vx` | `z0` | 8 | 0.0495–8.41 | 0.495–0.841 | 19.6% | 0 | 0 |
| sparse | `dot_vx` | `z50` | 8 | 0.0495–8.01 | 0.495–0.801 | 17.8% | 0 | 0 |
| sparse | `dot_x` | `rank1` | 8 | 0.0495–8.4 | 0.495–0.84 | 32.2% | 0 | 0 |
| sparse | `dot_xy` | `z0` | 8 | 0.134–16.6 | 1.34–1.66 | 7.6% | 0 | 0 |
| sparse | `dot_xy` | `z50` | 8 | 0.134–16.4 | 1.34–1.64 | 7.5% | 0 | 0 |
| sparse | `dropzeros!` | `zeros10` | 8 | 0.12–16.6 | 1.17–1.66 | 13.8% | 0 | 4 |
| sparse | `dropzeros!` | `zeros50` | 8 | 0.238–66.6 | 2.38–6.66 | 19.6% | 0 | 2 |
| sparse | `fill!_coordinates_zero` | `vector` | 8 | 0.064–20.5 | 0.64–2.05 | 54.0% | 0 | 4 |
| sparse | `fill!_zero` | `vector` | 8 | 0.065–20.4 | 0.65–2.04 | 80.0% | 0 | 4 |
| sparse | `fill!_zero_dropzeros` | `vector` | 8 | 0.069–20.4 | 0.69–2.04 | 60.9% | 0 | 4 |
| sparse | `first_activation` | `absent_distinct_batch` | 8 | 0.566–0.62 | 2.21–2.42 | 12.4% | 0 | 0 |
| sparse | `first_activation` | `absent_distinct_batch_x16` | 8 | 9.26–9.82 | 2.26–2.4 | 24.5% | 0 | 0 |
| sparse | `first_activation` | `activate_distinct_batch` | 8 | 0.505–0.572 | 1.97–2.23 | 25.5% | 0 | 0 |
| sparse | `first_activation` | `activate_distinct_batch_x16` | 8 | 8.74–9.39 | 2.13–2.29 | 17.3% | 0 | 0 |
| sparse | `isactive` | `half_active` | 8 | 0.199–0.207 | 0.778–0.81 | 15.2% | 0 | 0 |
| sparse | `mul!` | `strided_z0` | 4 | 2.46–25.4 | 0.0883–0.248 | 33.5% | 0 | 0 |
| sparse | `mul!` | `strided_z50` | 4 | 2.55–19.3 | 0.0878–0.189 | 26.9% | 0 | 0 |
| sparse | `mul!` | `symmetric_z0` | 8 | 1.17–496 | 0.572–2.45 | 80.5% | 0 | 0 |
| sparse | `mul!` | `symmetric_z50` | 8 | 1.18–496 | 0.574–2.44 | 104.9% | 0 | 0 |
| sparse | `mul!` | `transposed_z0` | 4 | 63.5–312 | 2.48–3.04 | 12.6% | 0 | 0 |
| sparse | `mul!` | `transposed_z50` | 4 | 63.6–310 | 2.48–3.03 | 12.3% | 0 | 0 |
| sparse | `mul!_matrix` | `rows1024_z0` | 12 | 65.9–1.64e+04 | 0.124–0.949 | 13.4% | 0 | 0 |
| sparse | `mul!_matrix` | `rows1024_z50` | 12 | 65.5–1.65e+04 | 0.124–0.95 | 13.4% | 0 | 0 |
| sparse | `mul!_matrix` | `rows256_z0` | 12 | 24.6–3.43e+03 | 0.151–0.712 | 11.0% | 0 | 0 |
| sparse | `mul!_matrix` | `rows256_z50` | 12 | 15–3.45e+03 | 0.148–0.689 | 61.5% | 0 | 0 |
| sparse | `prox_coordinate!_5arg` | `NormL1_scalar_u` | 8 | 3.96–4.54 | 2.58–2.95 | 18.7% | 0 | 0 |
| sparse | `read_x[i]` | `absent` | 8 | 0.332–0.439 | 1.3–1.71 | 32.7% | 0 | 0 |
| sparse | `read_x[i]` | `active` | 8 | 0.339–0.354 | 1.32–1.38 | 4.1% | 0 | 0 |
| sparse | `replace_cheap` | `vector` | 8 | 10–1.22e+03 | 1–1.22 | 17.9% | 0 | 0 |
| sparse | `replace_staged` | `vector` | 8 | 45.8–4.41e+03 | 3.78–6.05 | 75.4% | 24000264 | 0 |
| sparse | `rmul!_finite` | `x123` | 2 | 0.74–0.794 | 0.0602–0.0646 | 20.0% | 0 | 0 |
| sparse | `rmul!_finite` | `x164` | 2 | 1.57–1.68 | 0.0955–0.103 | 11.6% | 0 | 0 |
| sparse | `rmul!_finite` | `x17` | 2 | 0.804–1.55 | 0.0473–0.0912 | 7.2% | 0 | 0 |
| sparse | `rmul!_finite` | `x2` | 2 | 1.51–3.07 | 0.0755–0.153 | 1.7% | 0 | 0 |
| sparse | `rmul!_inf` | `x123` | 2 | 0.76–0.763 | 0.0618–0.062 | 0.6% | 0 | 0 |
| sparse | `rmul!_inf` | `x164` | 2 | 1.57 | 0.0955 | 7.6% | 0 | 0 |
| sparse | `rmul!_inf` | `x17` | 2 | 0.8–1.55 | 0.0471–0.0914 | 7.8% | 0 | 0 |
| sparse | `rmul!_inf` | `x2` | 2 | 1.52–3.08 | 0.0762–0.154 | 3.6% | 0 | 0 |
| sparse | `sweep_active_update` | `trivial_update` | 8 | 0.113–16.2 | 1.13–1.62 | 17.5% | 0 | 0 |
| sparse | `sweep_activecoordinates` | `vector` | 8 | 0.0139–0.898 | 0.0897–0.153 | 26.5% | 0 | 0 |
| sparse | `sweep_activevalues` | `vector` | 8 | 0.0813–13.4 | 0.813–1.34 | 3.4% | 0 | 0 |
| sparse | `sweep_full` | `vector` | 8 | 4.83–1.1e+03 | 0.483–1.1 | 128.3% | 0 | 0 |
| sparse | `sweep_via_coordinates` | `rank1` | 8 | 10.9–1.1e+03 | 1.01–1.1 | 116.1% | 0 | 0 |
| sparse | `sweep_via_coordinates` | `rank2` | 12 | 26.7–791 | 0.417–3.09 | 155.7% | 0 | 0 |
| sparse | `sweep_via_x` | `rank1` | 8 | 5.02–642 | 0.502–0.748 | 129.8% | 0 | 0 |
| sparse | `sweep_via_x` | `rank2` | 12 | 67–791 | 1.05–3.09 | 152.7% | 0 | 0 |
| sparse | `write_x[i]_active` | `active` | 8 | 0.889–1.31 | 0.868–1.28 | 22.9% | 0 | 0 |

## Scaling

The exponent is the slope of log(median per call) against log(size) between two fixtures, so that the work per call
changes only along the axis in question. The size axis is `n` (10⁴ to 10⁶ at `s = 100`), `p` at the same number of
active groups (8·10⁴ with 10% of the groups active against 8·10⁵ with 1%) or the owner's `p` (500 to 4000 at `s = 100`);
the count axis is `s` (10³ to 10⁴ at `n = 10⁶`), the share of active groups (1% to 10% at equal `p`) or the owner's `s`
(10³ to 10⁵ at `p = 1000`). The target is the acceptance guide's §6 table, read for the 0.5.x code, with room for the
cache hierarchy and fixed overheads: flat is |exponent| ≤ 0.4, linear 0.6 to 1.5, quadratic 1.6 to 2.6. A column
outside its target is a statement about what 0.5.x does today (most often a cache effect over a factor of 10 or 100),
not a defect. A curve that is flat over a factor of 100 is a regression alarm and not a proof of O(1). Exceptions that are
known: `AtomIterate` has no `rmul!` method (an O(p) loop), the owner's `fill!(coordinates(X), 0)` is an O(q) loop, and
`coordinates(x) === x` on a `SparseIterate`.

| group | case | size axis: exponent (range) | expected | count axis: exponent (range) | expected | outside the target |
|:--|:--|:--|:--|:--|:--|:--|
| sparse (n, s) | `_restartdot` | -0.00–0.00 | no stated target | 0.98 | – |  |
| sparse (n, s) | `_setvalues!` | -0.01–0.01 | no stated target | 1.24–1.44 | – |  |
| sparse (n, s) | `_values!` | -0.01–0.01 | no stated target | 1.29–1.50 | – |  |
| sparse (n, s) | `broadcast_assign` | 0.02–0.05 | flat | 1.18–1.20 | linear |  |
| sparse (n, s) | `broadcast_self` | 0.97 | linear | 0.05–0.06 | – |  |
| sparse (n, s) | `copy` | 0.89–0.95 | linear | 0.02–0.04 | – |  |
| sparse (n, s) | `copyto!_activevalues` | -0.00–0.01 | no stated target | 0.99–1.00 | – |  |
| sparse (n, s) | `copyto!_convert` | -0.01–0.00 | no stated target | 1.07–1.10 | – |  |
| sparse (n, s) | `copyto!_dense` | 0.97–1.00 | linear | 0.06 | – |  |
| sparse (n, s) | `copyto!_same_kind` | -0.00–0.05 | flat | 1.06–1.09 | linear |  |
| sparse (n, s) | `copyto!_state` | 0.00–0.01 | flat | 1.18–1.20 | linear |  |
| sparse (n, s) | `copyto!_unrelated` | 0.98–1.00 | no stated target | 0.02 | – |  |
| sparse (n, s) | `dot_via_coordinates` | -0.00–0.00 | flat | 1.08–1.09 | linear |  |
| sparse (n, s) | `dot_vx` | -0.01–0.00 | flat | 1.03–1.09 | linear |  |
| sparse (n, s) | `dot_x` | 0.01 | flat | 1.05–1.09 | linear |  |
| sparse (n, s) | `dot_xy` | -0.00–0.00 | flat | 1.01–1.06 | linear |  |
| sparse (n, s) | `dropzeros!` | 0.00–0.01 | flat | 1.14–1.41 | linear |  |
| sparse (n, s) | `fill!_coordinates_zero` | -0.01–0.00 | flat | 1.09–1.13 | linear |  |
| sparse (n, s) | `fill!_zero` | 0.00–0.02 | flat | 1.09–1.13 | linear |  |
| sparse (n, s) | `fill!_zero_dropzeros` | -0.00–0.00 | flat | 1.07–1.12 | linear |  |
| sparse (n, s) | `first_activation` | -0.01–0.02 | flat | -0.04–0.03 | flat |  |
| sparse (n, s) | `isactive` | 0.00 | flat | -0.01–0.00 | flat |  |
| sparse (n, s) | `prox_coordinate!_5arg` | -0.01–0.00 | no stated target | 0.02–0.03 | – |  |
| sparse (n, s) | `read_x[i]` | -0.00–0.03 | flat | -0.12–0.02 | flat |  |
| sparse (n, s) | `replace_cheap` | 0.98–1.02 | linear | 0.05–0.06 | – |  |
| sparse (n, s) | `replace_staged` | 0.92–0.96 | linear | 0.03 | – |  |
| sparse (n, s) | `rmul!_finite` | -0.02–0.02 | flat | – | linear |  |
| sparse (n, s) | `rmul!_inf` | -0.00 | flat | – | linear |  |
| sparse (n, s) | `sweep_active_update` | 0.00–0.03 | flat | 1.05 | linear |  |
| sparse (n, s) | `sweep_activecoordinates` | -0.02–0.02 | flat | 1.00 | linear |  |
| sparse (n, s) | `sweep_activevalues` | -0.00 | flat | 1.00 | linear |  |
| sparse (n, s) | `sweep_full` | 0.98–1.02 | linear | 0.03 | – |  |
| sparse (n, s) | `sweep_via_coordinates` | 0.98 | linear | 0.03–0.04 | – |  |
| sparse (n, s) | `sweep_via_x` | 0.94–1.01 | linear | 0.06 | – |  |
| sparse (n, s) | `write_x[i]_active` | -0.00–0.00 | flat | -0.00–0.03 | flat |  |
| atom (p, active groups) | `_nvalues` | -0.06–0.56 | flat | 0.83–1.50 | linear | size, count |
| atom (p, active groups) | `_restartdot` | -0.00–0.16 | flat | 0.98–1.39 | linear |  |
| atom (p, active groups) | `_setvalues!` | -0.04–0.72 | flat | 0.48–1.46 | linear | size, count |
| atom (p, active groups) | `_values!` | -0.04–0.23 | flat | 0.98–1.48 | linear |  |
| atom (p, active groups) | `atom_dot` | -0.05–0.20 | flat | -0.04–0.09 | flat |  |
| atom (p, active groups) | `atom_norm` | -0.03–0.02 | flat | -0.03–0.04 | flat |  |
| atom (p, active groups) | `atomindices_loop` | -0.04–0.28 | flat | 0.95–1.39 | linear |  |
| atom (p, active groups) | `atomindices_view` | -0.03–0.12 | flat | 1.00–1.38 | linear |  |
| atom (p, active groups) | `broadcast_assign` | -0.00–0.25 | flat | 1.04–1.40 | linear |  |
| atom (p, active groups) | `broadcast_self` | 0.75–1.00 | linear | 0.05–0.35 | – |  |
| atom (p, active groups) | `copy` | 0.75–1.09 | linear | -0.09–0.29 | – |  |
| atom (p, active groups) | `copyto!_convert` | -0.01–0.16 | no stated target | 1.01–1.35 | – |  |
| atom (p, active groups) | `copyto!_dense` | 0.68–1.04 | linear | 0.05–0.43 | – |  |
| atom (p, active groups) | `copyto!_same_kind` | 0.00–0.28 | flat | 1.03–1.37 | linear |  |
| atom (p, active groups) | `copyto!_state` | 0.00–0.25 | flat | 1.03–1.38 | linear |  |
| atom (p, active groups) | `copyto!_unrelated` | 0.75–1.01 | no stated target | 0.05–0.36 | – |  |
| atom (p, active groups) | `dot_vx` | -0.01–0.30 | flat | 0.99–1.45 | linear |  |
| atom (p, active groups) | `dot_xy` | -0.02–0.26 | no stated target | 0.99–1.44 | – |  |
| atom (p, active groups) | `dropzeros!` | -0.01–0.20 | flat | 0.98–1.53 | linear | count |
| atom (p, active groups) | `fill!_zero` | 0.00–0.44 | flat | 0.90–1.45 | linear | size |
| atom (p, active groups) | `fill!_zero_dropzeros` | -0.00–0.44 | flat | 0.90–1.41 | linear | size |
| atom (p, active groups) | `first_activation` | -0.04–0.39 | flat | -0.03–0.56 | flat | count |
| atom (p, active groups) | `isactive` | -0.06–0.06 | flat | -0.03–0.03 | flat |  |
| atom (p, active groups) | `mul!` | – | no stated target | 1.14–1.42 | – |  |
| atom (p, active groups) | `prox_coordinate!_5arg` | -0.02–0.18 | no stated target | -0.12–0.40 | – |  |
| atom (p, active groups) | `prox_coordinate!_staged` | -0.01–0.22 | no stated target | -0.12–0.47 | – |  |
| atom (p, active groups) | `read_x[i]` | -0.02–0.01 | flat | -0.02–0.02 | flat |  |
| atom (p, active groups) | `replace_cheap` | 0.63–0.99 | linear | 0.05–0.65 | – |  |
| atom (p, active groups) | `replace_staged` | 0.68–1.01 | linear | 0.04–0.54 | – |  |
| atom (p, active groups) | `rmul!_finite` | 0.73–1.03 | linear | 0.00–0.37 | – |  |
| atom (p, active groups) | `rmul!_inf` | 0.88–1.02 | linear | -0.01–0.17 | – |  |
| atom (p, active groups) | `sweep_active_update` | -0.00–0.38 | flat | 0.99–1.43 | linear |  |
| atom (p, active groups) | `sweep_activecoordinates` | -0.01–0.09 | flat | 0.25–1.04 | linear | count |
| atom (p, active groups) | `write_x[i]_active` | -0.05–0.01 | flat | -0.03–0.12 | flat |  |
| owner (p, s) | `Matrix` | 2.44 | quadratic | 0.25 | – |  |
| owner (p, s) | `_restartdot` | -0.00 | flat | 1.02 | linear |  |
| owner (p, s) | `_setvalues!` | 0.01–0.03 | flat | 1.29–1.38 | linear |  |
| owner (p, s) | `_values!` | -0.02–0.04 | flat | 1.31–1.42 | linear |  |
| owner (p, s) | `fill!_coordinates_zero` | 2.38 | quadratic | 0.35 | flat |  |
| owner (p, s) | `fill!_coordinates_zero_dropzeros` | 2.36 | quadratic | 0.44 | – |  |
| owner (p, s) | `fill!_zero` | -0.00 | flat | 1.20 | linear |  |
| owner (p, s) | `first_activation` | -0.05 | flat | -0.01 | flat |  |
| owner (p, s) | `mul!` | 0.18–0.92 | no stated target | 0.00–1.00 | – |  |
| owner (p, s) | `read_X[i,j]` | 0.00–0.02 | flat | -0.00–0.01 | flat |  |
| owner (p, s) | `sweep_activecoordinates` | 0.00 | flat | 1.00 | linear |  |
| owner (p, s) | `sweep_prox_coordinate!` | 0.00 | flat | 1.09 | linear |  |
| owner (p, s) | `write_X[i,j]_active` | 0.00 | flat | 0.02 | flat |  |

## Cases that cannot gate a 5% slowdown (spread above 5%)

| family | case | layout | cases above the limit | of | worst spread |
|:--|:--|:--|--:|--:|--:|
| products | `symmetric_L` | `sparse_0.01` | 4 | 4 | 2469% |
| products | `symmetric_L` | `atom_0.01` | 4 | 4 | 2416% |
| atom | `mul!` | `symmetric_z0` | 6 | 6 | 1243% |
| products | `symmetric_U` | `atom_0.01` | 4 | 4 | 990% |
| products | `symmetric_U` | `sparse_0.01` | 4 | 4 | 947% |
| atom | `mul!` | `symmetric_z50` | 6 | 6 | 930% |
| sparse | `copy` | `vector` | 6 | 8 | 303% |
| products | `strided_atom/mul!` | `atom_0.01` | 4 | 4 | 265% |
| atom | `copy` | `atoms` | 41 | 48 | 245% |
| products | `transposed_vector/mul!_scaled` | `sparse_0.01` | 4 | 4 | 230% |
| products | `strided_atom/mul!_scaled` | `atom_0.01` | 4 | 4 | 229% |
| products | `transposed_vector/mul!` | `sparse_0.01` | 4 | 4 | 226% |
| products | `strided_vector/mul!_scaled` | `sparse_0.01` | 4 | 4 | 187% |
| atom | `first_activation` | `absent_distinct_batch` | 39 | 48 | 186% |
| owner | `construct` | `empty` | 4 | 4 | 178% |
| owner | `fill!_coordinates_zero` | `owner` | 7 | 7 | 173% |
| owner | `Matrix` | `owner` | 4 | 7 | 172% |
| sparse | `sweep_via_coordinates` | `rank2` | 12 | 12 | 156% |
| sparse | `sweep_via_x` | `rank2` | 6 | 12 | 153% |
| group_layouts | `SparseGroupNormL2/coordinate` | `indices_distinct` | 3 | 4 | 146% |
| atom | `construct` | `empty` | 16 | 24 | 144% |
| products | `strided_matrix/mul!_scaled` | `sparse_0.01` | 4 | 4 | 135% |
| sparse | `sweep_via_x` | `rank1` | 8 | 8 | 130% |
| sparse | `sweep_full` | `vector` | 7 | 8 | 128% |
| sparse | `sweep_via_coordinates` | `rank1` | 4 | 8 | 116% |
| sparse | `mul!` | `symmetric_z50` | 6 | 8 | 105% |
| atom | `replace_staged` | `atoms` | 30 | 48 | 103% |
| owner | `first_activation` | `absent_distinct_batch` | 7 | 7 | 100% |
| group_layouts | `GroupNormL2/coordinate` | `indices_distinct` | 3 | 4 | 97% |
| products | `strided_matrix/mul!` | `sparse_0.01` | 4 | 4 | 88% |
| atom | `dropzeros!` | `zeros50` | 27 | 48 | 87% |
| sparse | `construct` | `empty` | 2 | 4 | 82% |
| sparse | `mul!` | `symmetric_z0` | 6 | 8 | 80% |
| sparse | `fill!_zero` | `vector` | 5 | 8 | 80% |
| products | `strided_vector/mul!` | `sparse_0.01` | 4 | 4 | 78% |
| atom | `fill!_zero` | `atoms` | 36 | 48 | 77% |
| sparse | `replace_staged` | `vector` | 6 | 8 | 75% |
| atom | `fill!_zero_dropzeros` | `atoms` | 35 | 48 | 70% |
| atom | `_setvalues!` | `private_fields` | 31 | 48 | 68% |
| atom | `_nvalues` | `private_fields` | 20 | 48 | 67% |
| atom | `atom_dot` | `atoms` | 32 | 48 | 66% |
| atom | `prox_coordinate!_staged` | `SparseGroupNormL2_strided_u` | 22 | 48 | 65% |
| sparse | `mul!_matrix` | `rows256_z50` | 8 | 12 | 62% |
| products | `strided_vector/mul!_scaled` | `sparse_0.2` | 4 | 4 | 61% |
| sparse | `fill!_zero_dropzeros` | `vector` | 2 | 8 | 61% |
| atom | `first_activation` | `absent_distinct_batch_x16` | 23 | 40 | 61% |
| atom | `prox_coordinate!_staged` | `GroupNormL2_strided_u` | 27 | 48 | 59% |
| products | `strided_atom/mul!_scaled` | `atom_0.2` | 3 | 4 | 57% |
| sparse | `fill!_coordinates_zero` | `vector` | 4 | 8 | 54% |
| products | `transposed_vector/mul!_scaled` | `sparse_0.2` | 2 | 4 | 53% |
| sparse | `copyto!_activevalues` | `slot_order` | 8 | 8 | 53% |
| sparse | `_setvalues!` | `private_fields` | 7 | 8 | 50% |
| atom | `_nvalues` | `accessors` | 24 | 48 | 47% |
| atom | `prox_coordinate!_staged` | `NormL1_strided_u` | 28 | 48 | 46% |
| products | `strided_atom/mul!` | `atom_0.2` | 3 | 4 | 45% |
| sparse | `copyto!_state` | `vector` | 6 | 8 | 45% |
| group_layouts | `GroupNormL2/coordinate` | `ranges_distinct` | 4 | 4 | 43% |
| atom | `sweep_active_update` | `trivial_update` | 26 | 48 | 39% |
| atom | `broadcast_self` | `abs` | 8 | 48 | 39% |
| products | `symmetric_L` | `sparse_0.2` | 4 | 4 | 38% |
| atom | `_values!` | `private_fields` | 32 | 48 | 37% |
| sparse | `copyto!_same_kind` | `same_entries` | 4 | 8 | 37% |
| atom | `prox_coordinate!_staged` | `GroupNormLinf_strided_u` | 31 | 48 | 36% |
| products | `symmetric_L` | `atom_0.2` | 4 | 4 | 35% |
| owner | `mul!` | `owner` | 5 | 7 | 34% |
| atom | `replace_cheap` | `atoms` | 35 | 48 | 34% |
| sparse | `mul!` | `strided_z0` | 4 | 4 | 33% |
| owner | `_setvalues!` | `private_fields` | 6 | 7 | 33% |
| products | `symmetric_U` | `atom_0.2` | 3 | 4 | 33% |
| atom | `copyto!_same_kind` | `same_entries` | 26 | 48 | 33% |
| owner | `_values!` | `private_fields` | 4 | 7 | 33% |
| products | `strided_vector/mul!` | `sparse_0.2` | 4 | 4 | 33% |
| sparse | `read_x[i]` | `absent` | 7 | 8 | 33% |
| owner | `sweep_activecoordinates` | `trivial_update` | 4 | 7 | 32% |
| atom | `copyto!_unrelated` | `dense_same_entries` | 32 | 48 | 32% |
| sparse | `dot_x` | `rank1` | 5 | 8 | 32% |
| atom | `copyto!_dense` | `dense_other_entries` | 32 | 48 | 32% |
| atom | `sweep_activecoordinates` | `atoms` | 17 | 48 | 32% |
| atom | `prox_coordinate!_5arg` | `NormL1_strided_u` | 19 | 48 | 32% |
| atom | `prox_coordinate!_5arg` | `GroupNormL2_strided_u` | 31 | 48 | 32% |
| group_layouts | `GroupNormL2/kkt` | `ranges_distinct` | 2 | 4 | 32% |
| products | `switch_matrix/mul!` | `random_0.02` | 4 | 4 | 32% |
| group_layouts | `GroupNormL2/kkt` | `indices_distinct` | 2 | 4 | 32% |
| sparse | `broadcast_assign` | `vector` | 3 | 8 | 32% |
| sparse | `copyto!_convert` | `to_Int64_sparse_branch` | 3 | 4 | 31% |
| atom | `_values!` | `accessors` | 27 | 48 | 31% |
| atom | `prox_coordinate!_5arg` | `GroupNormLinf_strided_u` | 29 | 48 | 31% |
| sparse | `dot_via_coordinates` | `rank1` | 4 | 8 | 31% |
| products | `switch_matrix/mul!` | `random_0.05` | 4 | 4 | 31% |
| group_layouts | `GroupNormL2/kkt` | `ranges_shared` | 4 | 4 | 30% |
| group_layouts | `SparseGroupNormL2/coordinate` | `ranges_distinct` | 3 | 4 | 30% |
| group_layouts | `GroupNormLinf/kkt` | `ranges_shared` | 2 | 4 | 30% |
| owner | `mul!` | `owner_z50` | 3 | 7 | 29% |
| atom | `broadcast_assign` | `atoms` | 24 | 48 | 29% |
| atom | `copyto!_state` | `atoms` | 22 | 48 | 29% |
| atom | `prox_coordinate!_5arg` | `SparseGroupNormL2_strided_u` | 25 | 48 | 29% |
| atom | `dot_xy` | `atom_sparse_z50` | 22 | 48 | 28% |
| atom | `dot_vx` | `z50` | 21 | 48 | 28% |
| atom | `first_activation` | `activate_distinct_batch` | 34 | 48 | 28% |
| products | `symmetric_U` | `sparse_0.2` | 3 | 4 | 27% |
| sparse | `mul!` | `strided_z50` | 4 | 4 | 27% |
| sparse | `sweep_activecoordinates` | `vector` | 5 | 8 | 26% |
| atom | `copyto!_convert` | `to_Float64` | 11 | 24 | 26% |
| sparse | `first_activation` | `activate_distinct_batch` | 8 | 8 | 26% |
| owner | `write_X[i,j]_active` | `active` | 4 | 7 | 25% |
| atom | `mul!` | `rows256_z0` | 20 | 24 | 25% |
| atom | `dot_xy` | `atom_atom_z0` | 19 | 48 | 25% |
| sparse | `first_activation` | `absent_distinct_batch_x16` | 5 | 8 | 25% |
| products | `transposed_vector/mul!` | `sparse_0.2` | 3 | 4 | 24% |
| atom | `samegroups` | `distinct_first_seen` | 5 | 12 | 24% |
| atom | `_setvalues!` | `accessors` | 34 | 48 | 24% |
| atom | `copyto!_convert` | `to_Float32` | 11 | 24 | 24% |
| group_layouts | `GroupNormLinf/kkt` | `indices_shared` | 1 | 4 | 23% |
| sparse | `write_x[i]_active` | `active` | 6 | 8 | 23% |
| atom | `atom_norm` | `atoms` | 16 | 48 | 23% |
| sparse | `_values!` | `private_fields` | 8 | 8 | 22% |
| atom | `write_x[i]_active` | `active` | 25 | 48 | 22% |
| products | `switch_matrix/mul!` | `random_0.1` | 4 | 4 | 22% |
| products | `strided_matrix/mul!_scaled` | `sparse_0.2` | 4 | 4 | 22% |
| atom | `dropzeros!` | `zeros10` | 31 | 48 | 22% |
| atom | `atomindices_view` | `view_range_group` | 8 | 24 | 22% |
| atom | `mul!` | `rows256_z50` | 22 | 24 | 21% |
| atom | `isactive` | `half_active` | 47 | 48 | 21% |
| products | `strided_matrix/mul!` | `sparse_0.2` | 4 | 4 | 21% |
| owner | `_setvalues!` | `accessors` | 6 | 7 | 21% |
| group_layouts | `GroupNormLinf/kkt` | `ranges_distinct` | 3 | 4 | 21% |
| sparse | `rmul!_finite` | `x123` | 2 | 2 | 20% |
| atom | `rmul!_finite` | `x1` | 35 | 48 | 20% |
| group_layouts | `SparseGroupNormL2/kkt` | `indices_distinct` | 3 | 4 | 20% |
| sparse | `dropzeros!` | `zeros50` | 5 | 8 | 20% |
| sparse | `dot_vx` | `z0` | 5 | 8 | 20% |
| atom | `atomindices_loop` | `active_groups` | 20 | 48 | 20% |
| sparse | `prox_coordinate!_5arg` | `NormL1_scalar_u` | 2 | 8 | 19% |
| atom | `_restartdot` | `accessors` | 20 | 48 | 18% |
| atom | `dot_vx` | `z0` | 20 | 48 | 18% |
| group_layouts | `SparseGroupNormL2/kkt` | `indices_shared` | 2 | 4 | 18% |
| sparse | `replace_cheap` | `vector` | 8 | 8 | 18% |
| sparse | `dot_vx` | `z50` | 4 | 8 | 18% |
| atom | `dot_xy` | `atom_sparse_z0` | 23 | 48 | 18% |
| atom | `copyto!_convert` | `to_Int64` | 9 | 24 | 18% |
| sparse | `_values!` | `accessors` | 7 | 8 | 18% |
| atom | `_restartdot` | `private_fields` | 24 | 48 | 18% |
| sparse | `sweep_active_update` | `trivial_update` | 6 | 8 | 17% |
| group_layouts | `GroupNormL2/prox!` | `ranges_shared` | 3 | 4 | 17% |
| sparse | `first_activation` | `activate_distinct_batch_x16` | 1 | 8 | 17% |
| atom | `dot_xy` | `atom_atom_z50` | 22 | 48 | 17% |
| group_layouts | `GroupNormL2/prox!` | `ranges_distinct` | 2 | 4 | 17% |
| sparse | `dot_aXb` | `z0` | 10 | 12 | 17% |
| atom | `first_activation` | `absent_distinct_batch_x4` | 7 | 8 | 17% |
| atom | `samegroups` | `shared` | 6 | 12 | 16% |
| atom | `mul!` | `rows1024_z0` | 11 | 24 | 16% |
| owner | `fill!_coordinates_zero_dropzeros` | `owner` | 3 | 7 | 16% |
| atom | `mul!` | `rows1024_z50` | 16 | 24 | 16% |
| owner | `fill!_zero` | `owner` | 5 | 7 | 15% |
| sparse | `isactive` | `half_active` | 4 | 8 | 15% |
| group_layouts | `GroupNormL2/value_and_prox!` | `ranges_distinct` | 2 | 4 | 14% |
| group_layouts | `GroupNormL2/value_and_prox!` | `ranges_shared` | 2 | 4 | 14% |
| atom | `atomindices_view` | `view_vector_group` | 8 | 24 | 14% |
| sparse | `dropzeros!` | `zeros10` | 7 | 8 | 14% |
| owner | `_values!` | `accessors` | 5 | 7 | 13% |
| sparse | `mul!_matrix` | `rows1024_z0` | 6 | 12 | 13% |
| sparse | `mul!_matrix` | `rows1024_z50` | 6 | 12 | 13% |
| sparse | `dot_aXb` | `z50` | 10 | 12 | 13% |
| group_layouts | `GroupNormLinf/coordinate` | `ranges_shared` | 2 | 4 | 13% |
| owner | `read_X[i,j]` | `absent_lower` | 6 | 7 | 13% |
| sparse | `mul!` | `transposed_z0` | 1 | 4 | 13% |
| sparse | `copyto!_dense` | `dense_other_nonzeros` | 4 | 8 | 13% |
| sparse | `first_activation` | `absent_distinct_batch` | 6 | 8 | 12% |
| sparse | `mul!` | `transposed_z50` | 1 | 4 | 12% |
| sparse | `rmul!_finite` | `x164` | 2 | 2 | 12% |
| owner | `read_X[i,j]` | `active_upper` | 6 | 7 | 12% |
| group_layouts | `SparseGroupNormL2/kkt` | `ranges_shared` | 2 | 4 | 11% |
| sparse | `mul!_matrix` | `rows256_z0` | 6 | 12 | 11% |
| atom | `read_x[i]` | `absent` | 4 | 48 | 11% |
| owner | `mul!` | `csc_baseline` | 2 | 7 | 11% |
| group_layouts | `GroupNormLinf/coordinate` | `indices_distinct` | 3 | 4 | 11% |
| atom | `first_activation` | `activate_distinct_batch_x16` | 8 | 40 | 11% |
| group_layouts | `GroupNormLinf/value_and_prox!` | `ranges_shared` | 3 | 4 | 11% |
| group_layouts | `GroupNormL2/value` | `indices_distinct` | 2 | 4 | 11% |
| group_layouts | `GroupNormLinf/value_and_prox!` | `ranges_distinct` | 3 | 4 | 10% |
| owner | `_restartdot` | `private_fields` | 2 | 7 | 10% |
| group_layouts | `GroupNormL2/value` | `indices_shared` | 2 | 4 | 10% |
| atom | `rmul!_inf` | `x1` | 10 | 48 | 10% |
| group_layouts | `GroupNormLinf/coordinate` | `ranges_distinct` | 3 | 4 | 10% |
| owner | `_restartdot` | `accessors` | 2 | 7 | 10% |
| atom | `first_activation` | `activate_distinct_batch_x4` | 5 | 8 | 10% |
| atom | `read_x[i]` | `active` | 15 | 48 | 9% |
| owner | `read_X[i,j]` | `absent_upper` | 7 | 7 | 9% |
| group_layouts | `GroupNormL2/coordinate` | `ranges_shared` | 2 | 4 | 9% |
| atom | `samegroups` | `alternating` | 6 | 12 | 9% |
| owner | `mul!` | `dense_symmetric_baseline` | 2 | 6 | 9% |
| sparse | `_setvalues!` | `accessors` | 5 | 8 | 9% |
| owner | `read_X[i,j]` | `active_lower` | 4 | 7 | 8% |
| sparse | `copyto!_unrelated` | `dense_same_nonzeros` | 6 | 8 | 8% |
| group_layouts | `SparseGroupNormL2/coordinate` | `indices_shared` | 2 | 4 | 8% |
| sparse | `rmul!_inf` | `x17` | 1 | 2 | 8% |
| group_layouts | `SparseGroupNormL2/kkt` | `ranges_distinct` | 1 | 4 | 8% |
| group_layouts | `GroupNormLinf/prox!` | `indices_shared` | 1 | 4 | 8% |
| sparse | `dot_xy` | `z0` | 3 | 8 | 8% |
| sparse | `rmul!_inf` | `x164` | 1 | 2 | 8% |
| sparse | `dot_xy` | `z50` | 1 | 8 | 7% |
| group_layouts | `GroupNormLinf/prox!` | `ranges_shared` | 1 | 4 | 7% |
| group_layouts | `GroupNormLinf/value_and_prox!` | `indices_shared` | 2 | 4 | 7% |
| group_layouts | `GroupNormLinf/value_alternating` | `indices_distinct` | 1 | 4 | 7% |
| group_layouts | `GroupNormLinf/prox!` | `indices_distinct` | 1 | 4 | 7% |
| group_layouts | `SparseGroupNormL2/value` | `indices_shared` | 2 | 4 | 7% |
| group_layouts | `GroupNormLinf/kkt` | `indices_distinct` | 2 | 4 | 7% |
| sparse | `rmul!_finite` | `x17` | 1 | 2 | 7% |
| sparse | `_restartdot` | `private_fields` | 1 | 8 | 7% |
| group_layouts | `GroupNormLinf/prox!` | `ranges_distinct` | 1 | 4 | 7% |
| group_layouts | `GroupNormL2/value` | `ranges_distinct` | 2 | 4 | 7% |
| group_layouts | `SparseGroupNormL2/value` | `ranges_shared` | 1 | 4 | 7% |
| sparse | `broadcast_self` | `abs` | 1 | 8 | 7% |
| group_layouts | `GroupNormL2/kkt` | `indices_shared` | 1 | 4 | 6% |
| group_layouts | `GroupNormL2/coordinate` | `indices_shared` | 3 | 4 | 6% |
| group_layouts | `GroupNormLinf/coordinate` | `indices_shared` | 2 | 4 | 6% |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `indices_distinct` | 1 | 4 | 6% |
| group_layouts | `GroupNormL2/value_and_prox!` | `indices_distinct` | 1 | 4 | 6% |
| group_layouts | `SparseGroupNormL2/coordinate` | `ranges_shared` | 1 | 4 | 6% |
| sparse | `copyto!_convert` | `to_Float64_sparse_branch` | 1 | 4 | 6% |
| group_layouts | `GroupNormLinf/value` | `indices_shared` | 1 | 4 | 6% |
| group_layouts | `SparseGroupNormL2/value` | `indices_distinct` | 3 | 4 | 6% |
| group_layouts | `GroupNormLinf/value` | `indices_distinct` | 1 | 4 | 6% |
| group_layouts | `SparseGroupNormL2/prox!` | `indices_distinct` | 1 | 4 | 5% |
| group_layouts | `SparseGroupNormL2/prox!` | `ranges_distinct` | 1 | 4 | 5% |
| group_layouts | `SparseGroupNormL2/value` | `ranges_distinct` | 1 | 4 | 5% |

## Timer-limited cases

| family | case | layout | timer-limited | of | median µs (range of the limited ones) |
|:--|:--|:--|--:|--:|--:|
| atom | `dropzeros!` | `zeros10` | 4 | 48 | 0.097–0.238 |
| atom | `fill!_zero` | `atoms` | 2 | 48 | 0.105–0.152 |
| atom | `fill!_zero_dropzeros` | `atoms` | 2 | 48 | 0.108–0.157 |
| group_layouts | `GroupNormL2/coordinate` | `indices_distinct` | 4 | 4 | 0.079–0.141 |
| group_layouts | `GroupNormL2/coordinate` | `indices_shared` | 4 | 4 | 0.08–0.083 |
| group_layouts | `GroupNormL2/coordinate` | `ranges_distinct` | 4 | 4 | 0.064–0.068 |
| group_layouts | `GroupNormL2/coordinate` | `ranges_shared` | 4 | 4 | 0.063–0.068 |
| group_layouts | `GroupNormLinf/coordinate` | `indices_distinct` | 4 | 4 | 0.117–0.123 |
| group_layouts | `GroupNormLinf/coordinate` | `indices_shared` | 4 | 4 | 0.116–0.12 |
| group_layouts | `GroupNormLinf/coordinate` | `ranges_distinct` | 4 | 4 | 0.104–0.116 |
| group_layouts | `GroupNormLinf/coordinate` | `ranges_shared` | 4 | 4 | 0.104–0.117 |
| group_layouts | `SparseGroupNormL2/coordinate` | `indices_distinct` | 4 | 4 | 0.0875–0.164 |
| group_layouts | `SparseGroupNormL2/coordinate` | `indices_shared` | 4 | 4 | 0.088–0.094 |
| group_layouts | `SparseGroupNormL2/coordinate` | `ranges_distinct` | 4 | 4 | 0.072–0.085 |
| group_layouts | `SparseGroupNormL2/coordinate` | `ranges_shared` | 4 | 4 | 0.072–0.079 |
| owner | `fill!_zero` | `owner` | 4 | 7 | 0.0655–0.071 |
| products | `strided_atom/mul!_scaled` | `atom_0.01` | 1 | 4 | 0.152 |
| products | `strided_vector/mul!_scaled` | `sparse_0.01` | 1 | 4 | 0.136 |
| products | `symmetric_L` | `atom_0.01` | 1 | 4 | 0.594 |
| products | `symmetric_L` | `sparse_0.01` | 1 | 4 | 0.565 |
| sparse | `broadcast_assign` | `vector` | 4 | 8 | 0.176–0.224 |
| sparse | `copyto!_state` | `vector` | 4 | 8 | 0.172–0.183 |
| sparse | `dropzeros!` | `zeros10` | 4 | 8 | 0.12–0.133 |
| sparse | `dropzeros!` | `zeros50` | 2 | 8 | 0.238–0.249 |
| sparse | `fill!_coordinates_zero` | `vector` | 4 | 8 | 0.064–0.072 |
| sparse | `fill!_zero` | `vector` | 4 | 8 | 0.065–0.072 |
| sparse | `fill!_zero_dropzeros` | `vector` | 4 | 8 | 0.069–0.072 |

## Allocation

Cases flagged `zero_alloc` (the operations FT23 lists) that allocate on 0.5.x:

| family | case | layout | cases | of | bytes per call (range) |
|:--|:--|:--|--:|--:|--:|
| atom | `prox_coordinate!_5arg` | `GroupNormL2_strided_u` | 48 | 48 | 4096–16384 |
| atom | `prox_coordinate!_5arg` | `GroupNormLinf_strided_u` | 24 | 48 | 4096–16384 |
| atom | `prox_coordinate!_5arg` | `SparseGroupNormL2_strided_u` | 48 | 48 | 4096–16384 |

Measured bytes per call of operations whose target is 0 B in the acceptance guide's §6 table but that FT23 does not list
(not flagged; `construct`, `copy` and the staged replace allocate by design):

| family | case | cases | max bytes per call |
|:--|:--|--:|--:|
| atom | `broadcast_assign` | 48 | 0 |
| atom | `broadcast_self` | 48 | 0 |
| atom | `copyto!_convert` | 72 | 0 |
| atom | `copyto!_dense` | 48 | 0 |
| atom | `copyto!_same_kind` | 48 | 0 |
| atom | `copyto!_state` | 48 | 0 |
| atom | `copyto!_unrelated` | 48 | 0 |
| atom | `fill!_zero` | 48 | 0 |
| atom | `fill!_zero_dropzeros` | 48 | 0 |
| atom | `rmul!_finite` | 48 | 0 |
| atom | `rmul!_inf` | 48 | 0 |
| owner | `fill!_zero` | 7 | 0 |
| sparse | `broadcast_assign` | 8 | 0 |
| sparse | `broadcast_self` | 8 | 0 |
| sparse | `copyto!_activevalues` | 8 | 0 |
| sparse | `copyto!_convert` | 18 | 0 |
| sparse | `copyto!_dense` | 8 | 0 |
| sparse | `copyto!_same_kind` | 8 | 0 |
| sparse | `copyto!_state` | 8 | 0 |
| sparse | `copyto!_unrelated` | 8 | 0 |
| sparse | `fill!_zero` | 8 | 0 |
| sparse | `fill!_zero_dropzeros` | 8 | 0 |
| sparse | `rmul!_finite` | 8 | 0 |
| sparse | `rmul!_inf` | 8 | 0 |

## Cross-check with independent loops

Four operations were timed again by a plain `time_ns` loop outside the harness (median of many calls, one thread,
same machine, once before the repetitions and once after them). The ratio is the harness median per operation over
the loop's median.

| operation | fixture | harness case / operations per call | harness median ns | loop before ns | ratio | loop after ns | ratio |
|:--|:--|:--|--:|--:|--:|--:|--:|
| first_activation_per_coordinate | Float64 n=1000000 s=1000 | `sparse/first_activation/Float64/n1000000_s1000/absent_distinct_batch/first_activation` / 256 | 2.26 | 2.92 | 0.77 | 3.08 | 0.74 |
| first_activation_per_coordinate_x16 | Float64 n=1000000 s=1000 | `sparse/first_activation/Float64/n1000000_s1000/absent_distinct_batch_x16/first_activation` / 4096 | 2.39 | 2.3 | 1.04 | 2.32 | 1.03 |
| sweep_activecoordinates_per_active | Float64 n=1000000 s=10000 | `sparse/sweep_activecoordinates/Float64/n1000000_s10000/vector/warm` / 10000 | 0.0897 | 0.0929 | 0.97 | 0.092 | 0.97 |
| dot_v_x_per_active | Float64 n=1000000 s=10000 | `sparse/dot_vx/Float64/n1000000_s10000/z0/warm` / 10000 | 0.841 | 0.844 | 1.00 | 0.819 | 1.03 |

Rows more than 20% apart (ratio before and after): `first_activation_per_coordinate` 0.77 and 0.74. The harness figure is the lower one. The same operation with 16 times as many first activations per call (`absent_distinct_batch_x16`, 4096 coordinates) agrees within 4% in both loops; the difference at 256 coordinates (168 to 208 ns per call) is not explained, and the first-activation figure for 256 coordinates should be read with that margin.
