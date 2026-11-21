# ProximalBase #1138 research evidence

This directory contains experimental Julia scripts, an error analysis, and reproducible data for a signed feasibility enclosure on stored dense `IndBallL1` coefficients. The archived package trees are inputs to the scripts. [Research issue](https://github.com/mlakolar/ProximalBase.jl/issues/1138)

## Revisions and case provenance

| Role | Commit | Source tree |
| --- | --- | --- |
| PB master audited and measured | `21f823f00a3234c810f95ffa49976bf1a464c61e` | `5db26d308035760de1311be28d39a7f8bbc1954f` |
| PB source loaded in PS #508 and replayed | `bc9e888d09e5fc3c3feb35905dd7a14e70657fa5` | `e3aa6977485dbe9b40b7fd6bec8ae4742cc3a8f3` |
| PS #508 source used for internal engine replay | `1a9f196bb8622180a52fc4e3d25279f1d65cb223` | `a1b1fb502a99942c9c90c7fffbede47c20b2cf20` |

The `historical_cases.jl` generator is byte-identical to the PB #1119 and PS #508 suites. The historical projection archive for job `14431540` has SHA-256 `6326f0b28078b32ef74bc8caca56f280c3312861e554b702fe6891c52a077af8`. The PS projection/solver research bundle used by jobs `14431772` and `14431773` has SHA-256 `b70fea8a217e1dfd7a7511dd283916912ebd9bea53a1d6cb1dc1216000e40e65`. `ps508_fixtures.jl` extracts the original fixture generators, cap calculation, fixed-work options, and legacy diagnostic routines. The historic logs saved margin and solver TSVs. The individual trial vectors in this study come from new instrumented runs. [PB #1119 evidence](https://github.com/mlakolar/ProximalBase.jl/pull/1120#issuecomment-7058566019) · [PS #508 evidence](https://github.com/mlakolar/ProximalSolvers.jl/issues/508#issuecomment-7059237814)

## Reproduce

Use Julia 1.10.13 and 1.13.4. Pin BLAS and Julia to one thread. Run validation both normally and with `--check-bounds=yes`. The PB-only scripts include the package source from the supplied directory and require no nonstandard Julia dependencies. Supply archive or checkout paths from the exact revisions above.

```bash
julia --startup-file=no --threads=1 --check-bounds=auto validate.jl "$PB_MASTER/src" validation.tsv
julia --startup-file=no --threads=1 --check-bounds=yes validate.jl "$PB_PIN/src" validation-checked.tsv
julia --startup-file=no --threads=1 --check-bounds=auto fusion_check.jl "$PB_MASTER/src" fusion.tsv
python3 -I profile_source.py "$PB_MASTER/src" "$PROFILE_COPY"
julia --startup-file=no --threads=1 scan_counts.jl "$PB_MASTER/src" "$PROFILE_COPY" scans.tsv
```

Run the timings on an exclusive CPU allocation with a bound core. Time one driver at a time. The steady-state benchmark uses a typed result sink, three timed batches per paired sample, eight paired samples per order and independent A/A controls. The raw tables preserve each pair, time, allocation count and measured bytes. The profile copy is used only for the separate scan-count run.

```bash
julia --startup-file=no --threads=1 --check-bounds=auto micro.jl "$PB_MASTER/src" "$PB_PIN/src" micro.tsv 16 full
julia --startup-file=no --threads=1 --pkgimages=no --check-bounds=auto --project="$PS_PINNED_PROJECT" solver_replay.jl replay 16 full
python3 -I summarize.py numerical data/validation/master-*.tsv.gz
python3 -I summarize.py micro data/timing/*-micro.tsv.gz
python3 -I summarize.py solve data/timing/*-solve-pairs.tsv.gz
python3 -I summarize.py replay data/replay/*-replay.tsv.gz
python3 -I compare_archived_margins.py data/archive-508 data/validation
```

Use a disposable checkout of the exact PS revision with the corresponding file from `manifests/` alongside its `Project.toml`. Instantiate that environment and verify the loaded PB tree in the table. The archived runs used the 1.10.13 manifest hash `2e74a05b70dddc126d5475a4498d551bb72f7eac859605bb4ef31e6adb942be8` and 1.13.4 manifest hash `c8df8ca89d685ee10661b3aa6473a12642df13363b2e904fc48cfc1211334f7f`. The solver script calls internal PGD and APG on research-only wrappers. All timed variants complete matched fixed work. The legacy column uses the archived sign-and-cap query. Cap decisions on these stored values are checked independently in the replay. The cap-and-stop timing adds one accurate enclosure at the returned coefficients. The untimed accuracy-recipe replay records actual proposed stop points and stationarity projections separately. It uses the archived starting points and budgets with the pinned engine's generic stationarity route.

`replay-vectors.hex.gz` files contain a lossless hex encoding of concatenated little-endian IEEE float bytes. Decode the hex before using `byte_offset`, `type` and `n` from the corresponding replay TSV. Each decoded slice carries its own SHA-256 in `vector_sha256`.

Run `python3 -I verify_replay.py data/replay/1.13.4-replay.tsv.gz data/replay/1.13.4-replay-vectors.hex.gz` to verify one version from its exact stored bytes. The independent verifier uses Python integers in units of the smallest binary64 subnormal.

See `METHODS.md` for the bounds and arithmetic assumptions. `data/jobs.tsv` records the Slurm outcomes. Data integrity is recorded in `SHA256SUMS`.
