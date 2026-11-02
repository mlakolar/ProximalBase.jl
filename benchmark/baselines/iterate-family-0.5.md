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
| Harness | `benchmark/run.jl` and `benchmark/cases.jl` at commit `5d43d165722bfb5a401abf735ba2a673d9ee0946` of this branch, hash `844fa5330b9065ab33fc7bb9ef7605dbe8560e16` (`harness_hash` in each metadata file) |
| Julia | 1.13.3, `JULIA_NUM_THREADS=1`, 1 BLAS thread (LBTConfig([ILP64] libopenblas64_.so)), `--check-bounds=default`, BenchmarkTools 1.8.0 |
| CPU and OS | AMD EPYC 7513 32-Core Processor (`znver3`), Linux x86_64-linux-gnu |
| Settings | extended suite, `PB_REPETITION=1,2,3` in fresh processes, 40 samples of at most 0.1 s; timer floor 20–21 ns |
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
spread mixes the machine's noise with the effect of the inputs: it describes the baseline and does not decide whether a case can gate, because
`compare.py` pairs the same repetition numbers on both sides and the inputs cancel in the pair. What decides is the A/A control: the same source was measured a second time with the same harness and the same repetition numbers (the set `aa`, in separate processes, in alternating order) and the two sets were compared with `compare.py` exactly as a revision would be. A case can gate a 5% slowdown when it is not timer-limited and the half-width of its A/A 95% interval for the ratio, √(high/low) − 1, is at most 5%, so that a true 5% slowdown would put the whole interval of a comparison below 1. 1092 of 3719 cases can gate a 5% slowdown (94 are timer-limited, 2533 have an A/A interval wider than ±5%); the A/A comparison flagged 104 of the 3625 cases that are not timer-limited as faster or slower (2.9%; about 5% is expected from a 95% interval). In 104 cases the input draw changes the number of
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

| family | case | layout | cases | median µs (range) | per work unit ns (range) | max spread | max bytes | timer-limited | can gate 5% |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| atom | `_nvalues` | `accessors` | 48 | 0.0109–161 | 0.618–2.69 | 88.9% | 0 | 0 | 18 |
| atom | `_nvalues` | `private_fields` | 48 | 0.00902–158 | 0.435–2.38 | 76.3% | 0 | 0 | 16 |
| atom | `_restartdot` | `accessors` | 48 | 0.632–766 | 0.823–12 | 55.3% | 0 | 0 | 14 |
| atom | `_restartdot` | `private_fields` | 48 | 0.633–730 | 0.824–11.3 | 54.0% | 0 | 0 | 16 |
| atom | `_setvalues!` | `accessors` | 48 | 0.965–2.87e+03 | 1.26–35.9 | 46.9% | 0 | 0 | 12 |
| atom | `_setvalues!` | `private_fields` | 48 | 0.0443–723 | 0.0576–13.2 | 73.5% | 0 | 0 | 11 |
| atom | `_values!` | `accessors` | 48 | 0.0557–412 | 0.0725–6.73 | 47.6% | 0 | 0 | 13 |
| atom | `_values!` | `private_fields` | 48 | 0.0507–362 | 0.066–5.58 | 40.6% | 0 | 0 | 10 |
| atom | `atom_dot` | `atoms` | 48 | 0.312–5.34 | 0.144–8.55 | 20.5% | 0 | 0 | 18 |
| atom | `atom_norm` | `atoms` | 48 | 0.484–7.91 | 0.289–11.5 | 19.1% | 0 | 0 | 5 |
| atom | `atomindices_loop` | `active_groups` | 48 | 0.067–232 | 0.0873–3.89 | 51.9% | 0 | 0 | 21 |
| atom | `atomindices_view` | `view_range_group` | 24 | 0.158–371 | 0.206–4.8 | 22.5% | 0 | 0 | 6 |
| atom | `atomindices_view` | `view_vector_group` | 24 | 0.707–880 | 0.92–13 | 54.1% | 0 | 0 | 3 |
| atom | `broadcast_assign` | `atoms` | 48 | 0.26–1.28e+04 | 0.00325–16 | 35.4% | 0 | 0 | 5 |
| atom | `broadcast_self` | `abs` | 48 | 44.7–1.57e+03 | 0.559–1.96 | 99.8% | 0 | 0 | 24 |
| atom | `construct` | `empty` | 24 | 183–2.6e+04 | 2.03–32.5 | 79.0% | 83200424 | 0 | 3 |
| atom | `copy` | `atoms` | 48 | 22.5–1.82e+03 | 0.256–2.27 | 478.9% | 19200280 | 0 | 1 |
| atom | `copyto!_convert` | `to_Float32` | 24 | 0.256–5.97e+03 | 0.334–74.6 | 23.6% | 0 | 0 | 4 |
| atom | `copyto!_convert` | `to_Float64` | 24 | 0.289–6.99e+03 | 0.376–87.3 | 38.0% | 0 | 0 | 6 |
| atom | `copyto!_convert` | `to_Int64` | 24 | 1.06–6.89e+03 | 1.38–86.1 | 15.5% | 0 | 0 | 4 |
| atom | `copyto!_dense` | `dense_other_entries` | 48 | 44.6–1.51e+03 | 0.558–1.89 | 42.9% | 0 | 0 | 40 |
| atom | `copyto!_same_kind` | `same_entries` | 48 | 0.229–7.34e+03 | 0.149–45.8 | 53.2% | 0 | 0 | 5 |
| atom | `copyto!_state` | `atoms` | 48 | 0.251–1.21e+04 | 0.163–75.4 | 30.3% | 0 | 0 | 3 |
| atom | `copyto!_unrelated` | `dense_same_entries` | 48 | 44.4–1.18e+03 | 0.554–1.48 | 58.1% | 0 | 0 | 36 |
| atom | `dot_vx` | `z0` | 48 | 0.609–465 | 0.793–6.92 | 46.6% | 0 | 0 | 16 |
| atom | `dot_vx` | `z50` | 48 | 0.609–472 | 0.793–6.63 | 48.1% | 0 | 0 | 13 |
| atom | `dot_xy` | `atom_atom_z0` | 48 | 0.61–461 | 0.794–6.95 | 39.1% | 0 | 0 | 16 |
| atom | `dot_xy` | `atom_atom_z50` | 48 | 0.61–453 | 0.794–6.68 | 31.2% | 0 | 0 | 18 |
| atom | `dot_xy` | `atom_sparse_z0` | 48 | 0.632–520 | 0.822–7.68 | 32.3% | 0 | 0 | 13 |
| atom | `dot_xy` | `atom_sparse_z50` | 48 | 0.626–530 | 0.816–7.24 | 42.1% | 0 | 0 | 12 |
| atom | `dropzeros!` | `zeros10` | 48 | 0.07–2.13e+03 | 0.0911–26.6 | 31.7% | 0 | 4 | 9 |
| atom | `dropzeros!` | `zeros50` | 48 | 0.29–2.77e+03 | 0.378–34.6 | 32.4% | 0 | 0 | 11 |
| atom | `fill!_zero` | `atoms` | 48 | 0.09–2.84e+03 | 0.111–35.5 | 114.3% | 0 | 2 | 7 |
| atom | `fill!_zero_dropzeros` | `atoms` | 48 | 0.08–2.85e+03 | 0.104–35.6 | 92.1% | 0 | 2 | 5 |
| atom | `first_activation` | `absent_distinct_batch` | 48 | 0.987–2.85 | 3.86–11.1 | 47.1% | 0 | 0 | 2 |
| atom | `first_activation` | `absent_distinct_batch_x16` | 40 | 19–42.6 | 4.64–10.4 | 37.6% | 0 | 0 | 9 |
| atom | `first_activation` | `absent_distinct_batch_x4` | 8 | 3.32–9.16 | 3.24–8.94 | 22.5% | 0 | 0 | 1 |
| atom | `first_activation` | `activate_distinct_batch` | 48 | 0.271–0.361 | 1.06–1.41 | 131.4% | 0 | 0 | 2 |
| atom | `first_activation` | `activate_distinct_batch_x16` | 40 | 4.9–5.84 | 1.2–1.43 | 12.8% | 0 | 0 | 4 |
| atom | `first_activation` | `activate_distinct_batch_x4` | 8 | 1.05–1.13 | 1.03–1.1 | 20.5% | 0 | 0 | 1 |
| atom | `isactive` | `half_active` | 48 | 0.147–0.224 | 0.575–0.874 | 48.0% | 0 | 0 | 18 |
| atom | `mul!` | `rows1024_z0` | 24 | 46.4–3.66e+03 | 0.059–0.447 | 32.9% | 0 | 0 | 7 |
| atom | `mul!` | `rows1024_z50` | 24 | 31.9–1.73e+03 | 0.0405–0.211 | 30.0% | 0 | 0 | 3 |
| atom | `mul!` | `rows256_z0` | 24 | 11.5–749 | 0.0584–0.366 | 38.2% | 0 | 0 | 0 |
| atom | `mul!` | `rows256_z50` | 24 | 6.4–413 | 0.0326–0.202 | 31.5% | 0 | 0 | 6 |
| atom | `mul!` | `symmetric_z0` | 6 | 292–402 | 2.72–3.25 | 1516.0% | 0 | 0 | 0 |
| atom | `mul!` | `symmetric_z50` | 6 | 147–188 | 1.36–1.66 | 1698.7% | 0 | 0 | 0 |
| atom | `prox_coordinate!_5arg` | `GroupNormL2_strided_u` | 48 | 3.76–42.7 | 0.917–167 | 30.9% | 16384 | 0 | 11 |
| atom | `prox_coordinate!_5arg` | `GroupNormLinf_strided_u` | 48 | 7.91–42.9 | 3.17–167 | 11.5% | 16384 | 0 | 18 |
| atom | `prox_coordinate!_5arg` | `NormL1_strided_u` | 48 | 2.2–19.5 | 0.538–76.3 | 62.3% | 0 | 0 | 10 |
| atom | `prox_coordinate!_5arg` | `SparseGroupNormL2_strided_u` | 48 | 8.98–42.9 | 2.22–168 | 27.5% | 16384 | 0 | 17 |
| atom | `prox_coordinate!_staged` | `GroupNormL2_strided_u` | 48 | 2.37–40.3 | 0.58–158 | 31.9% | 0 | 0 | 15 |
| atom | `prox_coordinate!_staged` | `GroupNormLinf_strided_u` | 48 | 7.11–45 | 3.04–176 | 13.9% | 0 | 0 | 15 |
| atom | `prox_coordinate!_staged` | `NormL1_strided_u` | 48 | 2.35–21.6 | 0.575–81.2 | 53.6% | 0 | 0 | 14 |
| atom | `prox_coordinate!_staged` | `SparseGroupNormL2_strided_u` | 48 | 5.95–40.7 | 1.97–159 | 25.5% | 0 | 0 | 10 |
| atom | `read_x[i]` | `absent` | 48 | 0.16–0.164 | 0.624–0.641 | 6.6% | 0 | 0 | 43 |
| atom | `read_x[i]` | `active` | 48 | 0.159–0.164 | 0.621–0.639 | 4.1% | 0 | 0 | 43 |
| atom | `replace_cheap` | `atoms` | 48 | 44.6–4.68e+03 | 0.557–5.85 | 32.5% | 0 | 0 | 35 |
| atom | `replace_staged` | `atoms` | 48 | 66.5–1.52e+04 | 0.798–19 | 87.0% | 19200280 | 0 | 8 |
| atom | `rmul!_finite` | `x1` | 48 | 44.2–1.3e+03 | 0.553–1.62 | 66.2% | 0 | 0 | 33 |
| atom | `rmul!_inf` | `x1` | 48 | 84.4–1.92e+03 | 1.05–2.4 | 26.2% | 0 | 0 | 27 |
| atom | `samegroups` | `alternating` | 12 | 2.36–7.44e+03 | 0.933–17.5 | 26.5% | 0 | 0 | 1 |
| atom | `samegroups` | `distinct_first_seen` | 12 | 5.05–3.88e+03 | 0.995–17.6 | 29.8% | 0 | 0 | 1 |
| atom | `samegroups` | `shared` | 12 | 0.00329–0.00356 | 3.29–3.56 | 7.9% | 0 | 0 | 6 |
| atom | `sweep_active_update` | `trivial_update` | 48 | 0.927–3.78e+03 | 1.21–47.2 | 31.4% | 0 | 0 | 15 |
| atom | `sweep_activecoordinates` | `atoms` | 48 | 0.00575–7.45 | 0.0697–0.479 | 52.3% | 0 | 0 | 26 |
| atom | `write_x[i]_active` | `active` | 48 | 1.29–2.47 | 1.26–2.41 | 18.1% | 0 | 0 | 9 |
| group_layouts | `GroupNormL2/coordinate` | `indices_distinct` | 4 | 0.0505–0.07 | 6.31–8.75 | 40.0% | 0 | 4 | 0 |
| group_layouts | `GroupNormL2/coordinate` | `indices_shared` | 4 | 0.05–0.06 | 6.25–7.5 | 20.0% | 0 | 4 | 0 |
| group_layouts | `GroupNormL2/coordinate` | `ranges_distinct` | 4 | 0.05 | 6.25 | 25.0% | 0 | 4 | 0 |
| group_layouts | `GroupNormL2/coordinate` | `ranges_shared` | 4 | 0.0455–0.05 | 5.69–6.25 | 22.0% | 0 | 4 | 0 |
| group_layouts | `GroupNormL2/kkt` | `indices_distinct` | 4 | 0.013–0.015 | 1.62–1.88 | 16.4% | 0 | 0 | 0 |
| group_layouts | `GroupNormL2/kkt` | `indices_shared` | 4 | 0.0129–0.0146 | 1.62–1.82 | 8.4% | 0 | 0 | 1 |
| group_layouts | `GroupNormL2/kkt` | `ranges_distinct` | 4 | 0.0104–0.0116 | 1.3–1.45 | 45.1% | 0 | 0 | 1 |
| group_layouts | `GroupNormL2/kkt` | `ranges_shared` | 4 | 0.0102–0.0119 | 1.28–1.49 | 33.2% | 0 | 0 | 1 |
| group_layouts | `GroupNormL2/prox!` | `indices_distinct` | 4 | 0.338–0.362 | 5.27–5.66 | 4.9% | 0 | 0 | 1 |
| group_layouts | `GroupNormL2/prox!` | `indices_shared` | 4 | 0.341–0.374 | 5.33–5.84 | 5.4% | 0 | 0 | 1 |
| group_layouts | `GroupNormL2/prox!` | `ranges_distinct` | 4 | 0.22–0.25 | 3.44–3.9 | 16.0% | 0 | 0 | 0 |
| group_layouts | `GroupNormL2/prox!` | `ranges_shared` | 4 | 0.219–0.252 | 3.42–3.94 | 18.4% | 0 | 0 | 0 |
| group_layouts | `GroupNormL2/value` | `indices_distinct` | 4 | 0.0839–0.0993 | 1.31–1.55 | 13.5% | 0 | 0 | 1 |
| group_layouts | `GroupNormL2/value` | `indices_shared` | 4 | 0.0834–0.0985 | 1.3–1.54 | 15.4% | 0 | 0 | 0 |
| group_layouts | `GroupNormL2/value` | `ranges_distinct` | 4 | 0.0799–0.0985 | 1.25–1.54 | 27.1% | 0 | 0 | 2 |
| group_layouts | `GroupNormL2/value` | `ranges_shared` | 4 | 0.0801–0.0888 | 1.25–1.39 | 14.5% | 0 | 0 | 0 |
| group_layouts | `GroupNormL2/value_alternating` | `indices_distinct` | 4 | 7.18–89.9 | 56.1–702 | 24.1% | 0 | 0 | 1 |
| group_layouts | `GroupNormL2/value_alternating` | `ranges_distinct` | 4 | 2.01–18.4 | 15.7–144 | 19.2% | 0 | 0 | 0 |
| group_layouts | `GroupNormL2/value_and_prox!` | `indices_distinct` | 4 | 0.344–0.365 | 5.38–5.7 | 4.5% | 0 | 0 | 1 |
| group_layouts | `GroupNormL2/value_and_prox!` | `indices_shared` | 4 | 0.34–0.364 | 5.32–5.68 | 6.9% | 0 | 0 | 0 |
| group_layouts | `GroupNormL2/value_and_prox!` | `ranges_distinct` | 4 | 0.219–0.259 | 3.43–4.05 | 12.8% | 0 | 0 | 0 |
| group_layouts | `GroupNormL2/value_and_prox!` | `ranges_shared` | 4 | 0.219–0.247 | 3.42–3.86 | 23.8% | 0 | 0 | 1 |
| group_layouts | `GroupNormLinf/coordinate` | `indices_distinct` | 4 | 0.07 | 8.75 | 16.7% | 0 | 4 | 0 |
| group_layouts | `GroupNormLinf/coordinate` | `indices_shared` | 4 | 0.07 | 8.75 | 16.7% | 0 | 4 | 0 |
| group_layouts | `GroupNormLinf/coordinate` | `ranges_distinct` | 4 | 0.06–0.07 | 7.5–8.75 | 33.3% | 0 | 4 | 0 |
| group_layouts | `GroupNormLinf/coordinate` | `ranges_shared` | 4 | 0.06–0.07 | 7.5–8.75 | 33.3% | 0 | 4 | 0 |
| group_layouts | `GroupNormLinf/kkt` | `indices_distinct` | 4 | 0.0115–0.0116 | 1.43–1.45 | 4.2% | 0 | 0 | 2 |
| group_layouts | `GroupNormLinf/kkt` | `indices_shared` | 4 | 0.0114–0.012 | 1.43–1.5 | 5.8% | 0 | 0 | 2 |
| group_layouts | `GroupNormLinf/kkt` | `ranges_distinct` | 4 | 0.0082–0.00875 | 1.03–1.09 | 15.9% | 0 | 0 | 0 |
| group_layouts | `GroupNormLinf/kkt` | `ranges_shared` | 4 | 0.0082–0.00874 | 1.03–1.09 | 13.3% | 0 | 0 | 0 |
| group_layouts | `GroupNormLinf/prox!` | `indices_distinct` | 4 | 0.515–0.544 | 8.04–8.51 | 5.1% | 0 | 0 | 2 |
| group_layouts | `GroupNormLinf/prox!` | `indices_shared` | 4 | 0.521–0.548 | 8.15–8.57 | 5.4% | 0 | 0 | 2 |
| group_layouts | `GroupNormLinf/prox!` | `ranges_distinct` | 4 | 0.413–0.428 | 6.46–6.68 | 9.5% | 0 | 0 | 1 |
| group_layouts | `GroupNormLinf/prox!` | `ranges_shared` | 4 | 0.41–0.42 | 6.4–6.56 | 8.9% | 0 | 0 | 2 |
| group_layouts | `GroupNormLinf/value` | `indices_distinct` | 4 | 0.0793–0.0814 | 1.24–1.27 | 9.9% | 0 | 0 | 0 |
| group_layouts | `GroupNormLinf/value` | `indices_shared` | 4 | 0.0792–0.083 | 1.24–1.3 | 9.6% | 0 | 0 | 1 |
| group_layouts | `GroupNormLinf/value` | `ranges_distinct` | 4 | 0.0507–0.0526 | 0.793–0.823 | 7.4% | 0 | 0 | 2 |
| group_layouts | `GroupNormLinf/value` | `ranges_shared` | 4 | 0.0497–0.0524 | 0.777–0.819 | 7.3% | 0 | 0 | 2 |
| group_layouts | `GroupNormLinf/value_alternating` | `indices_distinct` | 4 | 7.21–85.5 | 56.3–668 | 16.9% | 0 | 0 | 0 |
| group_layouts | `GroupNormLinf/value_alternating` | `ranges_distinct` | 4 | 1.96–18.4 | 15.3–144 | 9.4% | 0 | 0 | 1 |
| group_layouts | `GroupNormLinf/value_and_prox!` | `indices_distinct` | 4 | 0.517–0.542 | 8.07–8.46 | 5.6% | 0 | 0 | 2 |
| group_layouts | `GroupNormLinf/value_and_prox!` | `indices_shared` | 4 | 0.513–0.549 | 8.02–8.58 | 4.7% | 0 | 0 | 4 |
| group_layouts | `GroupNormLinf/value_and_prox!` | `ranges_distinct` | 4 | 0.414–0.428 | 6.47–6.69 | 10.5% | 0 | 0 | 1 |
| group_layouts | `GroupNormLinf/value_and_prox!` | `ranges_shared` | 4 | 0.415–0.423 | 6.48–6.61 | 9.7% | 0 | 0 | 0 |
| group_layouts | `SparseGroupNormL2/coordinate` | `indices_distinct` | 4 | 0.06–0.081 | 7.5–10.1 | 50.0% | 0 | 4 | 0 |
| group_layouts | `SparseGroupNormL2/coordinate` | `indices_shared` | 4 | 0.06 | 7.5 | 0.0% | 0 | 4 | 0 |
| group_layouts | `SparseGroupNormL2/coordinate` | `ranges_distinct` | 4 | 0.0505–0.06 | 6.31–7.5 | 20.0% | 0 | 4 | 0 |
| group_layouts | `SparseGroupNormL2/coordinate` | `ranges_shared` | 4 | 0.055–0.06 | 6.88–7.5 | 9.9% | 0 | 4 | 0 |
| group_layouts | `SparseGroupNormL2/kkt` | `indices_distinct` | 4 | 0.0169–0.0187 | 2.12–2.33 | 6.4% | 0 | 0 | 1 |
| group_layouts | `SparseGroupNormL2/kkt` | `indices_shared` | 4 | 0.0169–0.0183 | 2.12–2.29 | 10.7% | 0 | 0 | 2 |
| group_layouts | `SparseGroupNormL2/kkt` | `ranges_distinct` | 4 | 0.0152–0.0167 | 1.9–2.08 | 7.4% | 0 | 0 | 0 |
| group_layouts | `SparseGroupNormL2/kkt` | `ranges_shared` | 4 | 0.0149–0.0164 | 1.86–2.05 | 6.6% | 0 | 0 | 1 |
| group_layouts | `SparseGroupNormL2/prox!` | `indices_distinct` | 4 | 0.427–0.439 | 6.68–6.85 | 2.9% | 0 | 0 | 2 |
| group_layouts | `SparseGroupNormL2/prox!` | `indices_shared` | 4 | 0.421–0.44 | 6.58–6.88 | 4.9% | 0 | 0 | 2 |
| group_layouts | `SparseGroupNormL2/prox!` | `ranges_distinct` | 4 | 0.326–0.354 | 5.1–5.54 | 9.8% | 0 | 0 | 2 |
| group_layouts | `SparseGroupNormL2/prox!` | `ranges_shared` | 4 | 0.326–0.378 | 5.1–5.91 | 16.5% | 0 | 0 | 1 |
| group_layouts | `SparseGroupNormL2/value` | `indices_distinct` | 4 | 0.0976–0.109 | 1.53–1.71 | 3.6% | 0 | 0 | 4 |
| group_layouts | `SparseGroupNormL2/value` | `indices_shared` | 4 | 0.0978–0.107 | 1.53–1.67 | 3.5% | 0 | 0 | 2 |
| group_layouts | `SparseGroupNormL2/value` | `ranges_distinct` | 4 | 0.0761–0.0933 | 1.19–1.46 | 7.9% | 0 | 0 | 1 |
| group_layouts | `SparseGroupNormL2/value` | `ranges_shared` | 4 | 0.0759–0.0889 | 1.19–1.39 | 18.5% | 0 | 0 | 2 |
| group_layouts | `SparseGroupNormL2/value_alternating` | `indices_distinct` | 4 | 7.15–84.3 | 55.8–658 | 15.7% | 0 | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value_alternating` | `ranges_distinct` | 4 | 2–18.2 | 15.7–142 | 2.0% | 0 | 0 | 2 |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `indices_distinct` | 4 | 0.425–0.443 | 6.64–6.93 | 12.4% | 0 | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `indices_shared` | 4 | 0.418–0.45 | 6.53–7.03 | 12.9% | 0 | 0 | 0 |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `ranges_distinct` | 4 | 0.325–0.354 | 5.08–5.53 | 7.4% | 0 | 0 | 2 |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `ranges_shared` | 4 | 0.326–0.357 | 5.09–5.58 | 10.7% | 0 | 0 | 1 |
| owner | `Matrix` | `owner` | 7 | 105–1.07e+04 | 0.394–1.11 | 65.2% | 128004112 | 0 | 1 |
| owner | `_restartdot` | `accessors` | 7 | 0.0768–82.4 | 0.768–0.826 | 16.8% | 0 | 0 | 5 |
| owner | `_restartdot` | `private_fields` | 7 | 0.075–82.6 | 0.75–0.826 | 6.8% | 0 | 0 | 6 |
| owner | `_setvalues!` | `accessors` | 7 | 0.0159–23.4 | 0.0779–0.234 | 75.6% | 0 | 0 | 2 |
| owner | `_setvalues!` | `private_fields` | 7 | 0.0112–23.3 | 0.0717–0.233 | 175.5% | 0 | 0 | 2 |
| owner | `_values!` | `accessors` | 7 | 0.0132–23 | 0.0728–0.23 | 338.3% | 0 | 0 | 2 |
| owner | `_values!` | `private_fields` | 7 | 0.0114–23.5 | 0.0734–0.235 | 218.1% | 0 | 0 | 2 |
| owner | `construct` | `empty` | 4 | 163–1.71e+04 | 1.19–2.47 | 12.8% | 192049200 | 0 | 1 |
| owner | `fill!_coordinates_zero` | `owner` | 7 | 35.2–2.82e+03 | 0.275–1.57 | 99.4% | 0 | 0 | 0 |
| owner | `fill!_coordinates_zero_dropzeros` | `owner` | 7 | 35.8–2.81e+03 | 0.275–2.01 | 99.3% | 0 | 0 | 0 |
| owner | `fill!_zero` | `owner` | 7 | 0.05–98.8 | 0.5–1.14 | 40.0% | 0 | 4 | 0 |
| owner | `first_activation` | `absent_distinct_batch` | 7 | 0.636–0.761 | 2.48–2.97 | 53.2% | 0 | 0 | 1 |
| owner | `mul!` | `csc_baseline` | 7 | 0.471–107 | 1.07–33.2 | 12.4% | 0 | 0 | 5 |
| owner | `mul!` | `dense_symmetric_baseline` | 6 | 21.8–298 | 0.0746–0.087 | 24.0% | 0 | 0 | 0 |
| owner | `mul!` | `owner` | 7 | 1–999 | 9.93–14.4 | 12.5% | 0 | 0 | 0 |
| owner | `mul!` | `owner_z50` | 7 | 1.04–982 | 9.82–13.2 | 13.3% | 0 | 0 | 2 |
| owner | `read_X[i,j]` | `absent_lower` | 7 | 0.333–0.383 | 1.3–1.5 | 21.3% | 0 | 0 | 0 |
| owner | `read_X[i,j]` | `absent_upper` | 7 | 0.33–0.384 | 1.29–1.5 | 22.9% | 0 | 0 | 0 |
| owner | `read_X[i,j]` | `active_lower` | 7 | 0.378–0.428 | 1.48–1.67 | 12.1% | 0 | 0 | 3 |
| owner | `read_X[i,j]` | `active_upper` | 7 | 0.379–0.405 | 1.48–1.58 | 11.2% | 0 | 0 | 6 |
| owner | `sweep_activecoordinates` | `trivial_update` | 7 | 0.0101–10.7 | 0.07–0.11 | 36.9% | 0 | 0 | 1 |
| owner | `sweep_prox_coordinate!` | `NormL1` | 7 | 0.721–1.19e+03 | 7.15–11.9 | 12.3% | 0 | 0 | 2 |
| owner | `write_X[i,j]_active` | `active` | 7 | 2.68–4.08 | 2.62–3.99 | 39.0% | 0 | 0 | 0 |
| products | `strided_atom/mul!` | `atom_0.01` | 4 | 0.0797–0.493 | 0.0638–0.127 | 154.1% | 0 | 0 | 1 |
| products | `strided_atom/mul!` | `atom_0.2` | 4 | 1.67–16.7 | 0.0546–0.165 | 35.9% | 0 | 0 | 0 |
| products | `strided_atom/mul!_scaled` | `atom_0.01` | 4 | 0.1–0.521 | 0.0686–0.137 | 138.1% | 0 | 2 | 0 |
| products | `strided_atom/mul!_scaled` | `atom_0.2` | 4 | 1.68–18.8 | 0.0552–0.158 | 37.9% | 0 | 0 | 0 |
| products | `strided_matrix/mul!` | `sparse_0.01` | 4 | 0.442–2.17 | 0.0718–0.148 | 80.4% | 0 | 0 | 1 |
| products | `strided_matrix/mul!` | `sparse_0.2` | 4 | 5.56–70.2 | 0.0546–0.167 | 29.1% | 0 | 0 | 0 |
| products | `strided_matrix/mul!_scaled` | `sparse_0.01` | 4 | 0.471–2.25 | 0.0726–0.153 | 81.1% | 0 | 0 | 1 |
| products | `strided_matrix/mul!_scaled` | `sparse_0.2` | 4 | 5.57–69.8 | 0.0535–0.166 | 30.7% | 0 | 0 | 1 |
| products | `strided_vector/mul!` | `sparse_0.01` | 4 | 0.0701–0.417 | 0.0499–0.109 | 135.3% | 0 | 0 | 1 |
| products | `strided_vector/mul!` | `sparse_0.2` | 4 | 1.57–16.1 | 0.0536–0.16 | 63.8% | 0 | 0 | 0 |
| products | `strided_vector/mul!_scaled` | `sparse_0.01` | 4 | 0.09–0.431 | 0.0549–0.114 | 113.1% | 0 | 2 | 2 |
| products | `strided_vector/mul!_scaled` | `sparse_0.2` | 4 | 1.58–19.5 | 0.0508–0.183 | 44.4% | 0 | 0 | 2 |
| products | `switch_matrix/mul!` | `random_0.02` | 4 | 0.626–27.4 | 0.0782–0.26 | 35.7% | 0 | 0 | 1 |
| products | `switch_matrix/mul!` | `random_0.05` | 4 | 3.44–59.1 | 0.0766–0.33 | 44.0% | 0 | 0 | 1 |
| products | `switch_matrix/mul!` | `random_0.1` | 4 | 3.46–125 | 0.0759–0.182 | 50.2% | 0 | 0 | 1 |
| products | `symmetric_L` | `atom_0.01` | 4 | 0.423–27.9 | 215–2.97e+03 | 2636.6% | 0 | 1 | 1 |
| products | `symmetric_L` | `atom_0.2` | 4 | 7.58–564 | 165–2.82e+03 | 37.3% | 0 | 0 | 2 |
| products | `symmetric_L` | `sparse_0.01` | 4 | 0.421–27.1 | 212–2.94e+03 | 2580.0% | 0 | 1 | 0 |
| products | `symmetric_L` | `sparse_0.2` | 4 | 7.47–568 | 162–2.82e+03 | 47.2% | 0 | 0 | 1 |
| products | `symmetric_U` | `atom_0.01` | 4 | 0.32–32.9 | 160–2.94e+03 | 1646.6% | 0 | 0 | 0 |
| products | `symmetric_U` | `atom_0.2` | 4 | 8.59–624 | 170–2.88e+03 | 29.4% | 0 | 0 | 0 |
| products | `symmetric_U` | `sparse_0.01` | 4 | 0.308–33.8 | 154–2.93e+03 | 1844.3% | 0 | 0 | 1 |
| products | `symmetric_U` | `sparse_0.2` | 4 | 8.56–611 | 157–2.82e+03 | 20.0% | 0 | 0 | 1 |
| products | `transposed_vector/mul!` | `sparse_0.01` | 4 | 0.609–6.35 | 0.736–1.15 | 331.8% | 0 | 0 | 0 |
| products | `transposed_vector/mul!` | `sparse_0.2` | 4 | 8.63–62.4 | 0.326–0.586 | 32.4% | 0 | 0 | 1 |
| products | `transposed_vector/mul!_scaled` | `sparse_0.01` | 4 | 0.821–6.52 | 1.02–1.27 | 552.0% | 0 | 0 | 0 |
| products | `transposed_vector/mul!_scaled` | `sparse_0.2` | 4 | 8.94–50.1 | 0.339–0.638 | 27.5% | 0 | 0 | 0 |
| sparse | `_restartdot` | `accessors` | 8 | 0.0765–8.21 | 0.765–0.825 | 2.1% | 0 | 0 | 8 |
| sparse | `_restartdot` | `private_fields` | 8 | 0.0747–8.2 | 0.747–0.825 | 2.0% | 0 | 0 | 8 |
| sparse | `_setvalues!` | `accessors` | 8 | 0.0139–1.45 | 0.0417–0.161 | 36.5% | 0 | 0 | 6 |
| sparse | `_setvalues!` | `private_fields` | 8 | 0.00835–1.46 | 0.0379–0.146 | 99.1% | 0 | 0 | 4 |
| sparse | `_values!` | `accessors` | 8 | 0.0113–1.45 | 0.0387–0.145 | 102.1% | 0 | 0 | 4 |
| sparse | `_values!` | `private_fields` | 8 | 0.00741–1.45 | 0.0363–0.145 | 101.9% | 0 | 0 | 4 |
| sparse | `broadcast_assign` | `vector` | 8 | 0.14–23.7 | 0.00014–0.0237 | 18.8% | 0 | 4 | 1 |
| sparse | `broadcast_self` | `abs` | 8 | 9.02–912 | 0.823–0.912 | 2.3% | 0 | 0 | 8 |
| sparse | `construct` | `empty` | 4 | 14.5–2.29e+03 | 1.21–2.71 | 184.2% | 24000264 | 0 | 0 |
| sparse | `copy` | `vector` | 8 | 15.2–1.57e+03 | 0.994–1.57 | 163.3% | 24000264 | 0 | 1 |
| sparse | `copyto!_activevalues` | `slot_order` | 8 | 0.0705–6.8 | 0.568–0.741 | 18.4% | 0 | 0 | 3 |
| sparse | `copyto!_convert` | `to_Float32_dense_branch` | 2 | 41.3–163 | 0.635–0.646 | 27.0% | 0 | 0 | 0 |
| sparse | `copyto!_convert` | `to_Float32_sparse_branch` | 4 | 0.113–22.1 | 1.13–2.21 | 8.9% | 0 | 0 | 1 |
| sparse | `copyto!_convert` | `to_Float64_dense_branch` | 2 | 45.3–178 | 0.697–0.707 | 24.6% | 0 | 0 | 0 |
| sparse | `copyto!_convert` | `to_Float64_sparse_branch` | 4 | 0.113–22.4 | 1.13–2.24 | 10.2% | 0 | 0 | 1 |
| sparse | `copyto!_convert` | `to_Int64_dense_branch` | 2 | 82.5–331 | 1.29 | 7.9% | 0 | 0 | 0 |
| sparse | `copyto!_convert` | `to_Int64_sparse_branch` | 4 | 0.155–24.6 | 1.55–2.46 | 8.8% | 0 | 0 | 4 |
| sparse | `copyto!_dense` | `dense_other_nonzeros` | 8 | 6.73–918 | 0.551–0.918 | 49.6% | 0 | 0 | 0 |
| sparse | `copyto!_same_kind` | `same_entries` | 8 | 0.113–22.5 | 0.563–1.13 | 23.4% | 0 | 0 | 1 |
| sparse | `copyto!_state` | `vector` | 8 | 0.131–23.5 | 0.652–1.17 | 18.7% | 0 | 4 | 0 |
| sparse | `copyto!_unrelated` | `dense_same_nonzeros` | 8 | 6.22–880 | 0.551–0.88 | 49.7% | 0 | 0 | 1 |
| sparse | `dot_aXb` | `z0` | 12 | 0.733–122 | 2.28–2.38 | 2.9% | 0 | 0 | 6 |
| sparse | `dot_aXb` | `z50` | 12 | 0.731–122 | 2.28–2.38 | 2.7% | 0 | 0 | 6 |
| sparse | `dot_via_coordinates` | `rank1` | 8 | 0.0323–9.65 | 0.323–0.965 | 21.9% | 0 | 0 | 2 |
| sparse | `dot_vx` | `z0` | 8 | 0.0329–9.55 | 0.329–0.955 | 21.2% | 0 | 0 | 3 |
| sparse | `dot_vx` | `z50` | 8 | 0.0319–9.19 | 0.319–0.919 | 19.9% | 0 | 0 | 2 |
| sparse | `dot_x` | `rank1` | 8 | 0.0328–9.64 | 0.328–0.964 | 18.0% | 0 | 0 | 2 |
| sparse | `dot_xy` | `z0` | 8 | 0.0769–11.8 | 0.769–1.18 | 12.6% | 0 | 0 | 3 |
| sparse | `dot_xy` | `z50` | 8 | 0.0772–11.9 | 0.772–1.19 | 13.3% | 0 | 0 | 4 |
| sparse | `dropzeros!` | `zeros10` | 8 | 0.08–9.07 | 0.707–0.907 | 23.5% | 0 | 4 | 0 |
| sparse | `dropzeros!` | `zeros50` | 8 | 0.18–22.2 | 1.8–2.22 | 39.4% | 0 | 4 | 2 |
| sparse | `fill!_coordinates_zero` | `vector` | 8 | 0.05–11.5 | 0.5–1.15 | 25.0% | 0 | 4 | 2 |
| sparse | `fill!_zero` | `vector` | 8 | 0.05–11.7 | 0.5–1.17 | 28.9% | 0 | 4 | 1 |
| sparse | `fill!_zero_dropzeros` | `vector` | 8 | 0.05–11.7 | 0.5–1.17 | 40.0% | 0 | 4 | 0 |
| sparse | `first_activation` | `absent_distinct_batch` | 8 | 0.581–0.601 | 2.27–2.35 | 42.3% | 0 | 0 | 0 |
| sparse | `first_activation` | `absent_distinct_batch_x16` | 8 | 9.7–10.2 | 2.37–2.49 | 7.7% | 0 | 0 | 0 |
| sparse | `first_activation` | `activate_distinct_batch` | 8 | 0.451–0.461 | 1.76–1.8 | 70.0% | 0 | 0 | 1 |
| sparse | `first_activation` | `activate_distinct_batch_x16` | 8 | 7.54–8.11 | 1.84–1.98 | 7.2% | 0 | 0 | 5 |
| sparse | `isactive` | `half_active` | 8 | 0.148–0.164 | 0.578–0.641 | 40.7% | 0 | 0 | 1 |
| sparse | `mul!` | `strided_z0` | 4 | 1.66–15.7 | 0.059–0.153 | 35.4% | 0 | 0 | 0 |
| sparse | `mul!` | `strided_z50` | 4 | 1.43–15.6 | 0.0558–0.153 | 35.2% | 0 | 0 | 0 |
| sparse | `mul!` | `symmetric_z0` | 8 | 1.05–610 | 0.513–3.13 | 162.4% | 0 | 0 | 1 |
| sparse | `mul!` | `symmetric_z50` | 8 | 1.05–609 | 0.51–3.13 | 162.1% | 0 | 0 | 2 |
| sparse | `mul!` | `transposed_z0` | 4 | 24.4–98.8 | 0.929–0.965 | 36.1% | 0 | 0 | 0 |
| sparse | `mul!` | `transposed_z50` | 4 | 24.4–96.2 | 0.928–0.967 | 45.1% | 0 | 0 | 0 |
| sparse | `mul!_matrix` | `rows1024_z0` | 12 | 37.9–7.27e+03 | 0.0717–0.363 | 35.2% | 0 | 0 | 0 |
| sparse | `mul!_matrix` | `rows1024_z50` | 12 | 37.9–7.26e+03 | 0.0718–0.374 | 34.6% | 0 | 0 | 0 |
| sparse | `mul!_matrix` | `rows256_z0` | 12 | 9.32–2.06e+03 | 0.0879–0.37 | 12.8% | 0 | 0 | 7 |
| sparse | `mul!_matrix` | `rows256_z50` | 12 | 9.11–2.05e+03 | 0.0881–0.369 | 12.0% | 0 | 0 | 5 |
| sparse | `prox_coordinate!_5arg` | `NormL1_scalar_u` | 8 | 2.55–5.73 | 1.66–3.73 | 20.4% | 0 | 0 | 0 |
| sparse | `read_x[i]` | `absent` | 8 | 0.217 | 0.846–0.847 | 7.9% | 0 | 0 | 0 |
| sparse | `read_x[i]` | `active` | 8 | 0.205–0.214 | 0.802–0.837 | 8.4% | 0 | 0 | 6 |
| sparse | `replace_cheap` | `vector` | 8 | 6.4–656 | 0.551–0.656 | 1.4% | 0 | 0 | 8 |
| sparse | `replace_staged` | `vector` | 8 | 29.2–2.77e+03 | 1.8–3.61 | 116.5% | 24000264 | 0 | 0 |
| sparse | `rmul!_finite` | `x123` | 2 | 0.481 | 0.0391 | 2.1% | 0 | 0 | 1 |
| sparse | `rmul!_finite` | `x164` | 2 | 1.14 | 0.0696 | 78.1% | 0 | 0 | 0 |
| sparse | `rmul!_finite` | `x17` | 2 | 0.621–1.2 | 0.0365–0.0707 | 0.8% | 0 | 0 | 1 |
| sparse | `rmul!_finite` | `x2` | 2 | 0.912–1.8 | 0.0456–0.0902 | 2.2% | 0 | 0 | 2 |
| sparse | `rmul!_inf` | `x123` | 2 | 0.481 | 0.0391 | 1.2% | 0 | 0 | 1 |
| sparse | `rmul!_inf` | `x164` | 2 | 1.14 | 0.0696 | 62.3% | 0 | 0 | 0 |
| sparse | `rmul!_inf` | `x17` | 2 | 0.621–1.2 | 0.0365–0.0707 | 0.8% | 0 | 0 | 2 |
| sparse | `rmul!_inf` | `x2` | 2 | 0.922–1.82 | 0.0461–0.0912 | 4.4% | 0 | 0 | 1 |
| sparse | `sweep_active_update` | `trivial_update` | 8 | 0.0914–22.8 | 0.914–2.28 | 20.0% | 0 | 0 | 0 |
| sparse | `sweep_activecoordinates` | `vector` | 8 | 0.0103–0.7 | 0.0699–0.106 | 95.2% | 0 | 0 | 3 |
| sparse | `sweep_activevalues` | `vector` | 8 | 0.045–8.16 | 0.45–0.816 | 0.5% | 0 | 0 | 7 |
| sparse | `sweep_full` | `vector` | 8 | 3.38–373 | 0.275–0.373 | 100.1% | 0 | 0 | 0 |
| sparse | `sweep_via_coordinates` | `rank1` | 8 | 3.35–624 | 0.275–0.624 | 100.5% | 0 | 0 | 0 |
| sparse | `sweep_via_coordinates` | `rank2` | 12 | 19.7–407 | 0.307–1.59 | 88.6% | 0 | 0 | 0 |
| sparse | `sweep_via_x` | `rank1` | 8 | 3.35–366 | 0.275–0.366 | 100.3% | 0 | 0 | 4 |
| sparse | `sweep_via_x` | `rank2` | 12 | 19.7–442 | 0.308–1.73 | 88.3% | 0 | 0 | 0 |
| sparse | `write_x[i]_active` | `active` | 8 | 1.34–1.79 | 1.31–1.75 | 23.5% | 0 | 0 | 2 |

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
| sparse (n, s) | `_restartdot` | -0.00–0.00 | no stated target | 1.00 | – |  |
| sparse (n, s) | `_setvalues!` | -0.02–0.01 | no stated target | 1.24–1.31 | – |  |
| sparse (n, s) | `_values!` | -0.00–0.00 | no stated target | 1.27–1.30 | – |  |
| sparse (n, s) | `broadcast_assign` | 0.00 | flat | 1.08–1.10 | linear |  |
| sparse (n, s) | `broadcast_self` | 0.98 | linear | 0.04 | – |  |
| sparse (n, s) | `copy` | 0.91–0.99 | linear | -0.00–0.11 | – |  |
| sparse (n, s) | `copyto!_activevalues` | -0.01–-0.00 | no stated target | 1.01 | – |  |
| sparse (n, s) | `copyto!_convert` | -0.00–0.02 | no stated target | 1.07–1.09 | – |  |
| sparse (n, s) | `copyto!_dense` | 0.96–0.98 | linear | 0.04–0.08 | – |  |
| sparse (n, s) | `copyto!_same_kind` | -0.00–0.02 | flat | 1.08–1.09 | linear |  |
| sparse (n, s) | `copyto!_state` | -0.01–0.02 | flat | 1.08–1.09 | linear |  |
| sparse (n, s) | `copyto!_unrelated` | 0.97–1.06 | no stated target | 0.03–0.05 | – |  |
| sparse (n, s) | `dot_via_coordinates` | 0.00–0.01 | flat | 1.07–1.14 | linear |  |
| sparse (n, s) | `dot_vx` | -0.00–0.01 | flat | 1.07–1.13 | linear |  |
| sparse (n, s) | `dot_x` | 0.00 | flat | 1.07–1.14 | linear |  |
| sparse (n, s) | `dot_xy` | -0.00–0.00 | flat | 1.09–1.13 | linear |  |
| sparse (n, s) | `dropzeros!` | -0.01–0.00 | flat | 1.04–1.09 | linear |  |
| sparse (n, s) | `fill!_coordinates_zero` | 0.00 | flat | 1.09–1.16 | linear |  |
| sparse (n, s) | `fill!_zero` | 0.00 | flat | 1.16 | linear |  |
| sparse (n, s) | `fill!_zero_dropzeros` | 0.00 | flat | 1.12–1.17 | linear |  |
| sparse (n, s) | `first_activation` | -0.00–0.02 | flat | -0.01–0.01 | flat |  |
| sparse (n, s) | `isactive` | -0.00–0.00 | flat | -0.01–0.01 | flat |  |
| sparse (n, s) | `prox_coordinate!_5arg` | -0.02–0.00 | no stated target | 0.22–0.26 | – |  |
| sparse (n, s) | `read_x[i]` | 0.00 | flat | -0.00–0.01 | flat |  |
| sparse (n, s) | `replace_cheap` | 0.96–0.97 | linear | 0.07 | – |  |
| sparse (n, s) | `replace_staged` | 0.85–0.90 | linear | 0.00–0.02 | – |  |
| sparse (n, s) | `rmul!_finite` | 0.00 | flat | – | linear |  |
| sparse (n, s) | `rmul!_inf` | 0.00 | flat | – | linear |  |
| sparse (n, s) | `sweep_active_update` | -0.00–0.00 | flat | 1.23–1.24 | linear |  |
| sparse (n, s) | `sweep_activecoordinates` | 0.00 | flat | 1.00 | linear |  |
| sparse (n, s) | `sweep_activevalues` | 0.00 | flat | 1.03 | linear |  |
| sparse (n, s) | `sweep_full` | 0.95–0.96 | linear | 0.10–0.12 | – |  |
| sparse (n, s) | `sweep_via_coordinates` | 0.96–0.98 | linear | 0.05–0.10 | – |  |
| sparse (n, s) | `sweep_via_x` | 0.96 | linear | 0.10–0.11 | – |  |
| sparse (n, s) | `write_x[i]_active` | -0.00 | flat | -0.00–0.00 | flat |  |
| atom (p, active groups) | `_nvalues` | -0.07–0.17 | flat | 0.76–1.34 | linear |  |
| atom (p, active groups) | `_restartdot` | -0.02–0.15 | flat | 0.90–1.38 | linear |  |
| atom (p, active groups) | `_setvalues!` | -0.04–0.21 | flat | 0.83–1.32 | linear |  |
| atom (p, active groups) | `_values!` | -0.02–0.15 | flat | 0.88–1.33 | linear |  |
| atom (p, active groups) | `atom_dot` | -0.07–0.08 | flat | -0.04–0.10 | flat |  |
| atom (p, active groups) | `atom_norm` | -0.05–0.02 | flat | -0.03–0.04 | flat |  |
| atom (p, active groups) | `atomindices_loop` | -0.02–0.12 | flat | 0.87–1.27 | linear |  |
| atom (p, active groups) | `atomindices_view` | -0.01–0.12 | flat | 0.89–1.37 | linear |  |
| atom (p, active groups) | `broadcast_assign` | 0.01–0.14 | flat | 0.98–1.40 | linear |  |
| atom (p, active groups) | `broadcast_self` | 0.62–1.04 | linear | -0.01–0.40 | – |  |
| atom (p, active groups) | `copy` | 0.69–1.08 | linear | -0.23–0.25 | – |  |
| atom (p, active groups) | `copyto!_convert` | -0.01–0.09 | no stated target | 1.00–1.28 | – |  |
| atom (p, active groups) | `copyto!_dense` | 0.71–1.03 | linear | -0.02–0.33 | – |  |
| atom (p, active groups) | `copyto!_same_kind` | -0.00–0.12 | flat | 0.97–1.27 | linear |  |
| atom (p, active groups) | `copyto!_state` | 0.01–0.14 | flat | 0.98–1.37 | linear |  |
| atom (p, active groups) | `copyto!_unrelated` | 0.74–1.03 | no stated target | -0.00–0.28 | – |  |
| atom (p, active groups) | `dot_vx` | -0.04–0.11 | flat | 0.90–1.27 | linear |  |
| atom (p, active groups) | `dot_xy` | -0.03–0.12 | no stated target | 0.92–1.28 | – |  |
| atom (p, active groups) | `dropzeros!` | -0.02–0.14 | flat | 1.00–1.36 | linear |  |
| atom (p, active groups) | `fill!_zero` | 0.00–0.16 | flat | 0.80–1.27 | linear |  |
| atom (p, active groups) | `fill!_zero_dropzeros` | -0.00–0.16 | flat | 0.98–1.28 | linear |  |
| atom (p, active groups) | `first_activation` | -0.01–0.30 | flat | -0.08–0.30 | flat |  |
| atom (p, active groups) | `isactive` | -0.05–0.18 | flat | -0.18–0.12 | flat |  |
| atom (p, active groups) | `mul!` | – | no stated target | 0.99–1.44 | – |  |
| atom (p, active groups) | `prox_coordinate!_5arg` | -0.02–0.11 | no stated target | -0.07–0.49 | – |  |
| atom (p, active groups) | `prox_coordinate!_staged` | -0.02–0.13 | no stated target | -0.07–0.52 | – |  |
| atom (p, active groups) | `read_x[i]` | -0.01–0.01 | flat | -0.01–0.01 | flat |  |
| atom (p, active groups) | `replace_cheap` | 0.46–0.94 | linear | 0.06–0.75 | – | size |
| atom (p, active groups) | `replace_staged` | 0.53–0.92 | linear | 0.06–0.87 | – | size |
| atom (p, active groups) | `rmul!_finite` | 0.70–1.03 | linear | 0.00–0.34 | – |  |
| atom (p, active groups) | `rmul!_inf` | 0.82–1.00 | linear | -0.00–0.23 | – |  |
| atom (p, active groups) | `sweep_active_update` | 0.01–0.17 | flat | 0.95–1.32 | linear |  |
| atom (p, active groups) | `sweep_activecoordinates` | -0.00–0.01 | flat | 0.33–1.12 | linear | count |
| atom (p, active groups) | `write_x[i]_active` | -0.03–0.02 | flat | -0.03–0.14 | flat |  |
| owner (p, s) | `Matrix` | 2.22 | quadratic | 0.22 | – |  |
| owner (p, s) | `_restartdot` | -0.00–0.00 | flat | 1.00 | linear |  |
| owner (p, s) | `_setvalues!` | 0.00–0.11 | flat | 1.24–1.26 | linear |  |
| owner (p, s) | `_values!` | -0.02–0.06 | flat | 1.25 | linear |  |
| owner (p, s) | `fill!_coordinates_zero` | 2.11 | quadratic | 0.37 | flat |  |
| owner (p, s) | `fill!_coordinates_zero_dropzeros` | 2.10 | quadratic | 0.42 | – |  |
| owner (p, s) | `fill!_zero` | 0.00 | flat | 1.05 | linear |  |
| owner (p, s) | `first_activation` | -0.06 | flat | 0.02 | flat |  |
| owner (p, s) | `mul!` | 0.10–0.94 | no stated target | -0.00–1.00 | – |  |
| owner (p, s) | `read_X[i,j]` | -0.04–0.00 | flat | -0.02–0.02 | flat |  |
| owner (p, s) | `sweep_activecoordinates` | 0.02 | flat | 1.09 | linear |  |
| owner (p, s) | `sweep_prox_coordinate!` | -0.01 | flat | 1.11 | linear |  |
| owner (p, s) | `write_X[i,j]_active` | -0.02 | flat | -0.02 | flat |  |

## What can gate a 5% slowdown (the A/A control)

1092 of 3719 cases can gate a 5% slowdown (94 are timer-limited, 2533 have an A/A interval wider than ±5%); the A/A comparison flagged 104 of the 3625 cases that are not timer-limited as faster or slower (2.9%; about 5% is expected from a 95% interval). The interval is that of `compare.py` for three paired repetitions (a Student interval with two degrees of freedom), so a pass is evidence from one control, not a guarantee; the rows of the family table below carry the number of cases that can gate. Cases not listed here and not timer-limited can gate at every fixture. The sparse `first_activation` rows of the layouts `absent_distinct_batch` (256 coordinates per call) and `absent_distinct_batch_x16` (4096) are diagnostic and never gate (see the cross-check below).

| family | cases | timer-limited | can gate 5% | cannot gate 5% | A/A false alarms |
|:--|--:|--:|--:|--:|--:|
| atom | 2688 | 8 | 824 | 1856 | 76 |
| group_layouts | 264 | 48 | 60 | 156 | 7 |
| owner | 157 | 4 | 41 | 112 | 8 |
| products | 108 | 6 | 19 | 83 | 7 |
| sparse | 502 | 28 | 148 | 326 | 6 |
| all | 3719 | 94 | 1092 | 2533 | 104 |

Cases, by family, case and layout, that are neither timer-limited nor able to gate:

| family | case | layout | cannot gate | of | worst A/A half-width |
|:--|:--|:--|--:|--:|--:|
| owner | `_values!` | `accessors` | 5 | 7 | 698% |
| atom | `write_x[i]_active` | `active` | 39 | 48 | 579% |
| atom | `first_activation` | `absent_distinct_batch` | 46 | 48 | 427% |
| owner | `_values!` | `private_fields` | 5 | 7 | 389% |
| owner | `_setvalues!` | `private_fields` | 5 | 7 | 350% |
| sparse | `copy` | `vector` | 7 | 8 | 238% |
| sparse | `copyto!_unrelated` | `dense_same_nonzeros` | 7 | 8 | 219% |
| sparse | `copyto!_dense` | `dense_other_nonzeros` | 8 | 8 | 217% |
| atom | `first_activation` | `activate_distinct_batch` | 46 | 48 | 203% |
| products | `transposed_vector/mul!` | `sparse_0.01` | 4 | 4 | 191% |
| atom | `copy` | `atoms` | 47 | 48 | 187% |
| sparse | `_setvalues!` | `private_fields` | 4 | 8 | 182% |
| atom | `copyto!_state` | `atoms` | 45 | 48 | 178% |
| group_layouts | `GroupNormL2/value_and_prox!` | `indices_shared` | 4 | 4 | 176% |
| sparse | `sweep_via_coordinates` | `rank1` | 8 | 8 | 172% |
| atom | `fill!_zero` | `atoms` | 39 | 48 | 171% |
| atom | `isactive` | `half_active` | 30 | 48 | 170% |
| atom | `rmul!_finite` | `x1` | 15 | 48 | 170% |
| owner | `fill!_coordinates_zero` | `owner` | 7 | 7 | 169% |
| sparse | `sweep_via_x` | `rank1` | 4 | 8 | 169% |
| sparse | `sweep_full` | `vector` | 8 | 8 | 168% |
| sparse | `_values!` | `accessors` | 4 | 8 | 168% |
| owner | `fill!_coordinates_zero_dropzeros` | `owner` | 7 | 7 | 168% |
| sparse | `_values!` | `private_fields` | 4 | 8 | 165% |
| atom | `_setvalues!` | `accessors` | 36 | 48 | 165% |
| atom | `first_activation` | `activate_distinct_batch_x16` | 36 | 40 | 164% |
| sparse | `sweep_active_update` | `trivial_update` | 8 | 8 | 158% |
| sparse | `sweep_activecoordinates` | `vector` | 5 | 8 | 154% |
| sparse | `sweep_via_x` | `rank2` | 12 | 12 | 150% |
| sparse | `sweep_via_coordinates` | `rank2` | 12 | 12 | 148% |
| atom | `atomindices_view` | `view_vector_group` | 21 | 24 | 143% |
| atom | `prox_coordinate!_5arg` | `NormL1_strided_u` | 38 | 48 | 142% |
| atom | `_setvalues!` | `private_fields` | 37 | 48 | 128% |
| sparse | `construct` | `empty` | 4 | 4 | 127% |
| sparse | `rmul!_finite` | `x164` | 2 | 2 | 121% |
| sparse | `isactive` | `half_active` | 7 | 8 | 118% |
| owner | `_setvalues!` | `accessors` | 5 | 7 | 116% |
| atom | `fill!_zero_dropzeros` | `atoms` | 41 | 48 | 110% |
| sparse | `rmul!_inf` | `x164` | 2 | 2 | 108% |
| atom | `prox_coordinate!_staged` | `GroupNormL2_strided_u` | 33 | 48 | 108% |
| group_layouts | `GroupNormLinf/value_and_prox!` | `ranges_distinct` | 3 | 4 | 107% |
| sparse | `_setvalues!` | `accessors` | 2 | 8 | 105% |
| atom | `construct` | `empty` | 21 | 24 | 105% |
| atom | `_nvalues` | `accessors` | 30 | 48 | 100% |
| atom | `broadcast_assign` | `atoms` | 43 | 48 | 99% |
| atom | `dot_vx` | `z0` | 32 | 48 | 97% |
| atom | `_nvalues` | `private_fields` | 32 | 48 | 97% |
| owner | `Matrix` | `owner` | 6 | 7 | 96% |
| group_layouts | `GroupNormL2/kkt` | `ranges_distinct` | 3 | 4 | 95% |
| sparse | `first_activation` | `activate_distinct_batch` | 7 | 8 | 93% |
| atom | `first_activation` | `absent_distinct_batch_x16` | 31 | 40 | 92% |
| group_layouts | `GroupNormL2/kkt` | `indices_shared` | 3 | 4 | 92% |
| products | `switch_matrix/mul!` | `random_0.05` | 3 | 4 | 91% |
| atom | `prox_coordinate!_staged` | `GroupNormLinf_strided_u` | 33 | 48 | 90% |
| products | `transposed_vector/mul!_scaled` | `sparse_0.2` | 4 | 4 | 90% |
| products | `transposed_vector/mul!_scaled` | `sparse_0.01` | 4 | 4 | 89% |
| products | `switch_matrix/mul!` | `random_0.1` | 3 | 4 | 89% |
| atom | `read_x[i]` | `active` | 5 | 48 | 87% |
| atom | `prox_coordinate!_5arg` | `GroupNormL2_strided_u` | 37 | 48 | 85% |
| atom | `copyto!_same_kind` | `same_entries` | 43 | 48 | 83% |
| atom | `sweep_activecoordinates` | `atoms` | 22 | 48 | 78% |
| atom | `first_activation` | `absent_distinct_batch_x4` | 7 | 8 | 78% |
| owner | `mul!` | `owner` | 7 | 7 | 78% |
| sparse | `fill!_coordinates_zero` | `vector` | 2 | 8 | 77% |
| atom | `replace_staged` | `atoms` | 40 | 48 | 76% |
| atom | `prox_coordinate!_staged` | `SparseGroupNormL2_strided_u` | 38 | 48 | 74% |
| owner | `write_X[i,j]_active` | `active` | 7 | 7 | 73% |
| atom | `copyto!_dense` | `dense_other_entries` | 8 | 48 | 72% |
| atom | `atomindices_loop` | `active_groups` | 27 | 48 | 71% |
| atom | `copyto!_convert` | `to_Float32` | 20 | 24 | 69% |
| sparse | `mul!` | `transposed_z0` | 4 | 4 | 69% |
| atom | `dropzeros!` | `zeros50` | 37 | 48 | 69% |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `indices_shared` | 4 | 4 | 68% |
| sparse | `rmul!_inf` | `x123` | 1 | 2 | 68% |
| atom | `dot_vx` | `z50` | 35 | 48 | 68% |
| group_layouts | `GroupNormLinf/prox!` | `ranges_distinct` | 3 | 4 | 66% |
| atom | `_values!` | `accessors` | 35 | 48 | 66% |
| atom | `_values!` | `private_fields` | 38 | 48 | 65% |
| sparse | `mul!` | `transposed_z50` | 4 | 4 | 65% |
| atom | `prox_coordinate!_5arg` | `SparseGroupNormL2_strided_u` | 31 | 48 | 65% |
| atom | `prox_coordinate!_5arg` | `GroupNormLinf_strided_u` | 30 | 48 | 64% |
| sparse | `mul!_matrix` | `rows1024_z0` | 12 | 12 | 64% |
| atom | `copyto!_convert` | `to_Float64` | 18 | 24 | 63% |
| group_layouts | `GroupNormL2/value` | `indices_shared` | 4 | 4 | 63% |
| sparse | `write_x[i]_active` | `active` | 6 | 8 | 62% |
| atom | `atom_dot` | `atoms` | 30 | 48 | 62% |
| atom | `mul!` | `rows256_z0` | 24 | 24 | 61% |
| atom | `replace_cheap` | `atoms` | 13 | 48 | 61% |
| owner | `sweep_activecoordinates` | `trivial_update` | 6 | 7 | 60% |
| atom | `prox_coordinate!_staged` | `NormL1_strided_u` | 34 | 48 | 60% |
| group_layouts | `GroupNormL2/value_and_prox!` | `indices_distinct` | 3 | 4 | 60% |
| atom | `broadcast_self` | `abs` | 24 | 48 | 60% |
| group_layouts | `GroupNormL2/value_and_prox!` | `ranges_distinct` | 4 | 4 | 57% |
| group_layouts | `GroupNormL2/prox!` | `ranges_distinct` | 4 | 4 | 57% |
| atom | `sweep_active_update` | `trivial_update` | 33 | 48 | 57% |
| group_layouts | `GroupNormL2/kkt` | `ranges_shared` | 3 | 4 | 55% |
| atom | `_restartdot` | `accessors` | 34 | 48 | 55% |
| sparse | `mul!_matrix` | `rows1024_z50` | 12 | 12 | 54% |
| atom | `dropzeros!` | `zeros10` | 35 | 48 | 54% |
| sparse | `sweep_activevalues` | `vector` | 1 | 8 | 53% |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `indices_distinct` | 4 | 4 | 53% |
| owner | `construct` | `empty` | 3 | 4 | 53% |
| group_layouts | `GroupNormL2/value_alternating` | `ranges_distinct` | 4 | 4 | 52% |
| atom | `mul!` | `rows1024_z0` | 17 | 24 | 50% |
| owner | `first_activation` | `absent_distinct_batch` | 6 | 7 | 49% |
| sparse | `replace_staged` | `vector` | 8 | 8 | 49% |
| atom | `mul!` | `rows256_z50` | 18 | 24 | 48% |
| atom | `dot_xy` | `atom_atom_z50` | 30 | 48 | 48% |
| atom | `_restartdot` | `private_fields` | 32 | 48 | 47% |
| atom | `mul!` | `rows1024_z50` | 21 | 24 | 47% |
| sparse | `copyto!_activevalues` | `slot_order` | 5 | 8 | 45% |
| atom | `dot_xy` | `atom_sparse_z0` | 35 | 48 | 45% |
| atom | `samegroups` | `distinct_first_seen` | 11 | 12 | 44% |
| atom | `dot_xy` | `atom_atom_z0` | 32 | 48 | 44% |
| sparse | `mul!` | `strided_z50` | 4 | 4 | 44% |
| owner | `read_X[i,j]` | `absent_lower` | 7 | 7 | 44% |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `ranges_shared` | 3 | 4 | 43% |
| atom | `copyto!_convert` | `to_Int64` | 20 | 24 | 43% |
| sparse | `first_activation` | `activate_distinct_batch_x16` | 3 | 8 | 43% |
| atom | `atom_norm` | `atoms` | 43 | 48 | 42% |
| atom | `samegroups` | `alternating` | 11 | 12 | 42% |
| sparse | `mul!` | `strided_z0` | 4 | 4 | 42% |
| products | `transposed_vector/mul!` | `sparse_0.2` | 3 | 4 | 42% |
| group_layouts | `SparseGroupNormL2/prox!` | `indices_distinct` | 2 | 4 | 40% |
| sparse | `dot_via_coordinates` | `rank1` | 6 | 8 | 39% |
| sparse | `copyto!_convert` | `to_Float32_dense_branch` | 2 | 2 | 39% |
| atom | `dot_xy` | `atom_sparse_z50` | 36 | 48 | 39% |
| sparse | `mul!` | `symmetric_z0` | 7 | 8 | 39% |
| sparse | `mul!` | `symmetric_z50` | 6 | 8 | 38% |
| sparse | `copyto!_convert` | `to_Float64_sparse_branch` | 3 | 4 | 38% |
| atom | `rmul!_inf` | `x1` | 21 | 48 | 37% |
| sparse | `copyto!_convert` | `to_Float64_dense_branch` | 2 | 2 | 37% |
| atom | `copyto!_unrelated` | `dense_same_entries` | 12 | 48 | 36% |
| atom | `atomindices_view` | `view_range_group` | 18 | 24 | 36% |
| sparse | `copyto!_same_kind` | `same_entries` | 7 | 8 | 36% |
| owner | `mul!` | `dense_symmetric_baseline` | 6 | 6 | 35% |
| group_layouts | `GroupNormL2/value_alternating` | `indices_distinct` | 3 | 4 | 35% |
| sparse | `copyto!_state` | `vector` | 4 | 8 | 35% |
| group_layouts | `GroupNormL2/value` | `ranges_distinct` | 2 | 4 | 35% |
| sparse | `dot_vx` | `z0` | 5 | 8 | 34% |
| sparse | `prox_coordinate!_5arg` | `NormL1_scalar_u` | 8 | 8 | 34% |
| group_layouts | `GroupNormL2/prox!` | `indices_distinct` | 3 | 4 | 34% |
| sparse | `fill!_zero_dropzeros` | `vector` | 4 | 8 | 33% |
| group_layouts | `SparseGroupNormL2/value_and_prox!` | `ranges_distinct` | 2 | 4 | 32% |
| group_layouts | `GroupNormLinf/prox!` | `indices_distinct` | 2 | 4 | 32% |
| sparse | `dot_vx` | `z50` | 6 | 8 | 31% |
| group_layouts | `GroupNormL2/prox!` | `indices_shared` | 3 | 4 | 31% |
| atom | `mul!` | `symmetric_z0` | 6 | 6 | 31% |
| products | `strided_matrix/mul!` | `sparse_0.2` | 4 | 4 | 31% |
| group_layouts | `SparseGroupNormL2/prox!` | `indices_shared` | 2 | 4 | 30% |
| products | `strided_matrix/mul!_scaled` | `sparse_0.2` | 3 | 4 | 30% |
| sparse | `first_activation` | `absent_distinct_batch` | 8 | 8 | 30% |
| products | `strided_vector/mul!` | `sparse_0.01` | 3 | 4 | 30% |
| group_layouts | `GroupNormL2/kkt` | `indices_distinct` | 4 | 4 | 30% |
| group_layouts | `GroupNormL2/value_and_prox!` | `ranges_shared` | 3 | 4 | 29% |
| group_layouts | `SparseGroupNormL2/kkt` | `indices_shared` | 2 | 4 | 29% |
| products | `strided_atom/mul!` | `atom_0.2` | 4 | 4 | 28% |
| group_layouts | `SparseGroupNormL2/value_alternating` | `indices_distinct` | 4 | 4 | 28% |
| group_layouts | `GroupNormLinf/value_alternating` | `indices_distinct` | 4 | 4 | 28% |
| group_layouts | `GroupNormLinf/value_and_prox!` | `indices_distinct` | 2 | 4 | 27% |
| products | `strided_vector/mul!` | `sparse_0.2` | 4 | 4 | 27% |
| owner | `_restartdot` | `accessors` | 2 | 7 | 27% |
| owner | `read_X[i,j]` | `absent_upper` | 7 | 7 | 26% |
| atom | `mul!` | `symmetric_z50` | 6 | 6 | 26% |
| group_layouts | `GroupNormLinf/value_alternating` | `ranges_distinct` | 3 | 4 | 26% |
| group_layouts | `GroupNormLinf/kkt` | `indices_shared` | 2 | 4 | 26% |
| sparse | `dot_x` | `rank1` | 6 | 8 | 25% |
| group_layouts | `GroupNormL2/prox!` | `ranges_shared` | 4 | 4 | 25% |
| atom | `first_activation` | `activate_distinct_batch_x4` | 7 | 8 | 25% |
| products | `symmetric_U` | `atom_0.2` | 4 | 4 | 23% |
| group_layouts | `GroupNormL2/value` | `ranges_shared` | 4 | 4 | 23% |
| group_layouts | `SparseGroupNormL2/value_alternating` | `ranges_distinct` | 2 | 4 | 23% |
| group_layouts | `SparseGroupNormL2/value` | `ranges_shared` | 2 | 4 | 22% |
| sparse | `read_x[i]` | `absent` | 8 | 8 | 22% |
| products | `strided_atom/mul!_scaled` | `atom_0.2` | 4 | 4 | 22% |
| sparse | `mul!_matrix` | `rows256_z0` | 5 | 12 | 22% |
| group_layouts | `SparseGroupNormL2/prox!` | `ranges_shared` | 3 | 4 | 21% |
| owner | `mul!` | `owner_z50` | 5 | 7 | 21% |
| products | `symmetric_L` | `sparse_0.2` | 3 | 4 | 21% |
| products | `symmetric_U` | `sparse_0.2` | 3 | 4 | 21% |
| sparse | `fill!_zero` | `vector` | 3 | 8 | 20% |
| products | `symmetric_U` | `atom_0.01` | 4 | 4 | 19% |
| sparse | `mul!_matrix` | `rows256_z50` | 7 | 12 | 19% |
| atom | `read_x[i]` | `absent` | 5 | 48 | 19% |
| owner | `mul!` | `csc_baseline` | 2 | 7 | 19% |
| group_layouts | `GroupNormL2/value` | `indices_distinct` | 3 | 4 | 19% |
| products | `symmetric_L` | `atom_0.2` | 2 | 4 | 18% |
| sparse | `copyto!_convert` | `to_Float32_sparse_branch` | 3 | 4 | 18% |
| sparse | `dropzeros!` | `zeros10` | 4 | 8 | 18% |
| owner | `sweep_prox_coordinate!` | `NormL1` | 5 | 7 | 17% |
| owner | `read_X[i,j]` | `active_lower` | 4 | 7 | 17% |
| group_layouts | `GroupNormLinf/value` | `indices_shared` | 3 | 4 | 16% |
| group_layouts | `SparseGroupNormL2/prox!` | `ranges_distinct` | 2 | 4 | 16% |
| group_layouts | `GroupNormLinf/value` | `indices_distinct` | 4 | 4 | 16% |
| group_layouts | `SparseGroupNormL2/kkt` | `ranges_distinct` | 4 | 4 | 16% |
| owner | `read_X[i,j]` | `active_upper` | 1 | 7 | 16% |
| sparse | `broadcast_assign` | `vector` | 3 | 8 | 15% |
| atom | `samegroups` | `shared` | 6 | 12 | 15% |
| group_layouts | `GroupNormLinf/kkt` | `ranges_distinct` | 4 | 4 | 14% |
| products | `strided_matrix/mul!_scaled` | `sparse_0.01` | 3 | 4 | 14% |
| products | `strided_atom/mul!` | `atom_0.01` | 3 | 4 | 14% |
| owner | `fill!_zero` | `owner` | 3 | 7 | 14% |
| products | `strided_vector/mul!_scaled` | `sparse_0.2` | 2 | 4 | 14% |
| group_layouts | `GroupNormLinf/value` | `ranges_distinct` | 2 | 4 | 14% |
| group_layouts | `GroupNormLinf/kkt` | `ranges_shared` | 4 | 4 | 13% |
| sparse | `dot_xy` | `z0` | 5 | 8 | 13% |
| sparse | `dropzeros!` | `zeros50` | 2 | 8 | 13% |
| sparse | `first_activation` | `absent_distinct_batch_x16` | 8 | 8 | 12% |
| products | `strided_matrix/mul!` | `sparse_0.01` | 3 | 4 | 12% |
| sparse | `rmul!_inf` | `x2` | 1 | 2 | 12% |
| sparse | `dot_xy` | `z50` | 4 | 8 | 12% |
| products | `symmetric_L` | `sparse_0.01` | 3 | 4 | 11% |
| sparse | `copyto!_convert` | `to_Int64_dense_branch` | 2 | 2 | 11% |
| group_layouts | `GroupNormLinf/value_and_prox!` | `ranges_shared` | 4 | 4 | 10% |
| group_layouts | `SparseGroupNormL2/value` | `ranges_distinct` | 3 | 4 | 10% |
| sparse | `rmul!_finite` | `x17` | 1 | 2 | 10% |
| products | `switch_matrix/mul!` | `random_0.02` | 3 | 4 | 10% |
| products | `strided_atom/mul!_scaled` | `atom_0.01` | 2 | 4 | 9% |
| products | `symmetric_U` | `sparse_0.01` | 3 | 4 | 9% |
| group_layouts | `SparseGroupNormL2/kkt` | `indices_distinct` | 3 | 4 | 9% |
| group_layouts | `SparseGroupNormL2/kkt` | `ranges_shared` | 3 | 4 | 9% |
| sparse | `rmul!_finite` | `x123` | 1 | 2 | 8% |
| sparse | `dot_aXb` | `z50` | 6 | 12 | 8% |
| owner | `_restartdot` | `private_fields` | 1 | 7 | 8% |
| sparse | `read_x[i]` | `active` | 2 | 8 | 8% |
| group_layouts | `GroupNormLinf/prox!` | `ranges_shared` | 2 | 4 | 8% |
| products | `symmetric_L` | `atom_0.01` | 2 | 4 | 8% |
| sparse | `dot_aXb` | `z0` | 6 | 12 | 7% |
| group_layouts | `GroupNormLinf/prox!` | `indices_shared` | 2 | 4 | 6% |
| group_layouts | `SparseGroupNormL2/value` | `indices_shared` | 2 | 4 | 6% |
| group_layouts | `GroupNormLinf/value` | `ranges_shared` | 2 | 4 | 6% |
| group_layouts | `GroupNormLinf/kkt` | `indices_distinct` | 2 | 4 | 6% |

## Timer-limited cases

| family | case | layout | timer-limited | of | median µs (range of the limited ones) |
|:--|:--|:--|--:|--:|--:|
| atom | `dropzeros!` | `zeros10` | 4 | 48 | 0.07–0.17 |
| atom | `fill!_zero` | `atoms` | 2 | 48 | 0.09–0.14 |
| atom | `fill!_zero_dropzeros` | `atoms` | 2 | 48 | 0.08–0.1 |
| group_layouts | `GroupNormL2/coordinate` | `indices_distinct` | 4 | 4 | 0.0505–0.07 |
| group_layouts | `GroupNormL2/coordinate` | `indices_shared` | 4 | 4 | 0.05–0.06 |
| group_layouts | `GroupNormL2/coordinate` | `ranges_distinct` | 4 | 4 | 0.05 |
| group_layouts | `GroupNormL2/coordinate` | `ranges_shared` | 4 | 4 | 0.0455–0.05 |
| group_layouts | `GroupNormLinf/coordinate` | `indices_distinct` | 4 | 4 | 0.07 |
| group_layouts | `GroupNormLinf/coordinate` | `indices_shared` | 4 | 4 | 0.07 |
| group_layouts | `GroupNormLinf/coordinate` | `ranges_distinct` | 4 | 4 | 0.06–0.07 |
| group_layouts | `GroupNormLinf/coordinate` | `ranges_shared` | 4 | 4 | 0.06–0.07 |
| group_layouts | `SparseGroupNormL2/coordinate` | `indices_distinct` | 4 | 4 | 0.06–0.081 |
| group_layouts | `SparseGroupNormL2/coordinate` | `indices_shared` | 4 | 4 | 0.06 |
| group_layouts | `SparseGroupNormL2/coordinate` | `ranges_distinct` | 4 | 4 | 0.0505–0.06 |
| group_layouts | `SparseGroupNormL2/coordinate` | `ranges_shared` | 4 | 4 | 0.055–0.06 |
| owner | `fill!_zero` | `owner` | 4 | 7 | 0.05 |
| products | `strided_atom/mul!_scaled` | `atom_0.01` | 2 | 4 | 0.1–0.35 |
| products | `strided_vector/mul!_scaled` | `sparse_0.01` | 2 | 4 | 0.09–0.291 |
| products | `symmetric_L` | `atom_0.01` | 1 | 4 | 0.423 |
| products | `symmetric_L` | `sparse_0.01` | 1 | 4 | 0.421 |
| sparse | `broadcast_assign` | `vector` | 4 | 8 | 0.14 |
| sparse | `copyto!_state` | `vector` | 4 | 8 | 0.131–0.14 |
| sparse | `dropzeros!` | `zeros10` | 4 | 8 | 0.08–0.09 |
| sparse | `dropzeros!` | `zeros50` | 4 | 8 | 0.18–0.19 |
| sparse | `fill!_coordinates_zero` | `vector` | 4 | 8 | 0.05 |
| sparse | `fill!_zero` | `vector` | 4 | 8 | 0.05–0.0505 |
| sparse | `fill!_zero_dropzeros` | `vector` | 4 | 8 | 0.05–0.051 |

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
| first_activation_per_coordinate | Float64 n=1000000 s=1000 | `sparse/first_activation/Float64/n1000000_s1000/absent_distinct_batch/first_activation` / 256 | 2.35 | 2.04 | 1.15 | 2.04 | 1.15 |
| first_activation_per_coordinate_x16 | Float64 n=1000000 s=1000 | `sparse/first_activation/Float64/n1000000_s1000/absent_distinct_batch_x16/first_activation` / 4096 | 2.45 | 1.77 | 1.39 | 1.69 | 1.45 |
| sweep_activecoordinates_per_active | Float64 n=1000000 s=10000 | `sparse/sweep_activecoordinates/Float64/n1000000_s10000/vector/warm` / 10000 | 0.0699 | 0.0721 | 0.97 | 0.0712 | 0.98 |
| dot_v_x_per_active | Float64 n=1000000 s=10000 | `sparse/dot_vx/Float64/n1000000_s10000/z0/warm` / 10000 | 0.955 | 0.973 | 0.98 | 0.805 | 1.19 |

Rows more than 20% apart (ratio before and after): `first_activation_per_coordinate_x16` 1.39 and 1.45 (the harness figure is the higher one). The loop itself is not reproducible across processes for some operations: its own two runs (before and after the repetitions, separate processes) differ by up to 1.96× (`first_activation_per_coordinate_x16`, Float32 n=1000000 s=1000: 1.74 and 3.42 ns per operation). The restore probe runs the loop's own code for 4096 coordinates (restore by zeroing and `dropzeros!`) in a third process of the same job and gets 2.18 ns per coordinate, against 1.77 and 1.69 in the two runs of the loop; restoring with `copyto!` instead gives 2.3. The harness figure for that row, 2.45, is 12% and 7% away from the two probe values. On an Intel Xeon Silver 4116 node, measured earlier on this branch with the same timed operations, the first-activation ratios were 0.77 and 0.74 for 256 coordinates and 1.04 and 1.03 for 4096; the cause of the differences is not established. Because each of the two layouts missed the 20% criterion on one of the two machines, the `first_activation` rows of `absent_distinct_batch` and `absent_distinct_batch_x16` of the sparse family are **diagnostic**: they do not gate. The `sweep_activecoordinates` and `dot_vx` rows agree within 20% on this machine.
