# Tools and archived inputs of the iterate-family 0.5 baseline

This directory holds the tools that generated `benchmark/baselines/iterate-family-0.5.md` and
`benchmark/baselines/iterate-family-0.5.csv`, and an archive of the inputs they were generated from.
`reproduce.py` regenerates both files from the archive and compares them byte for byte with the
committed ones. [`iterate-family-0.5-provenance.md`](../iterate-family-0.5-provenance.md) records where each
input came from and what was verified; this page covers the files and the commands.

The five original tools are committed unchanged (same bytes as when the baseline was generated).
`pack_evidence.py` and `reproduce.py` are new.

| file | what it is |
|:--|:--|
| `make_baseline.py` | the generator of the md and csv from the result directories (original) |
| `verify_csv3.py` | recomputes every csv figure and the A/A columns from the raw metrics and metadata (original) |
| `identity.py` | lists the identity keys of the metadata files (original) |
| `crosscheck.jl` | independent `time_ns` loops, the source of the cross-check files (original, manual) |
| `restore_probe.jl` | the first-activation loop with two restore methods, the source of `restore-probe.txt` (original, manual) |
| `pack_evidence.py` | builds the archive deterministically from its 45 files |
| `reproduce.py` | extracts the archive, runs the generator, compares the outputs |
| `../data/iterate-family-0.5-evidence.tar.gz` | the archived inputs |

`crosscheck.jl` and `restore_probe.jl` are manual scripts for the pinned 0.5.x API of the baseline. They
are not run by CI, and the archive holds their recorded output, not a way to re-run them.

## Reproduce the baseline

From any directory, with Python 3.11 or later:

```sh
python3 benchmark/baselines/tools/reproduce.py
```

The script finds the repository from its own location. It checks the archive against the pinned SHA-256,
extracts the 45 members into a temporary directory, runs `make_baseline.py` there with `python -I -B`, and
compares the generated md and csv with the committed files. It writes nothing inside the repository.
Expected output:

```text
archive SHA-256 ae0235adad9f932ceb95ea3ad1709621d73fa0a506dd231b9c442fd005b1bf87 (pinned value)
extracted 45 members
iterate-family-0.5.md: identical, SHA-256 9055d230ecb74a654d5bb3beb315d25b1298260ae6373615b90d7edf99a968f2
iterate-family-0.5.csv: identical, SHA-256 cb8348bb8099e91fce8ac733454682a82f26d875db9f98ad92bc0a8479c71705
result: both files identical
```

Exit code 0: both files are identical. 1: a generated file differs from the committed one (the output names it
and gives both hashes). 2: an archive, usage or generator error (the generator's error output is shown).

Options:

- `--keep DIR` keeps the extraction (`DIR/iterate-family-0.5-evidence`) and the generated files
  (`DIR/generated`). `DIR` must not exist or must be empty.
- `--archive PATH` uses another archive and skips the hash check. It is meant for tests: a changed archive
  that the generator accepts gives exit code 1, a truncated archive or one with a missing or extra member
  gives exit code 2.

The generator, `verify_csv3.py` and `identity.py` can also be run by hand on a kept extraction. With `E` the
extraction directory (`DIR/iterate-family-0.5-evidence`), from the repository root:

```sh
python3 -I -B benchmark/baselines/tools/verify_csv3.py E benchmark/baselines/iterate-family-0.5.csv
python3 -I -B benchmark/baselines/tools/identity.py E
```

- `verify_csv3.py` prints three lines. The first is `rows 3719 | mismatches: none`, the second the worst relative
  error per column, the third `timer-limited (either side) 94; not timer-limited 3625; can gate 1092 (csv 1092);
  A/A conclusive among not timer-limited 104 (2.9%)`. It exits 0 whatever the mismatch count, so read the first
  line (provenance record, section 7). The second line shows 1.0e+00 for `spread` and `aa_half_width`; that is
  a purely relative error against a value near zero. The test is the tolerance (relative 6e-4, plus 5e-4 absolute
  for `spread` and the A/A fields), and no row exceeds it.
- `identity.py` prints the identity keys per set. For the `host` and `slurm_job_id` fields it prints
  `"redacted"` (see below). It exits 0 always.

The generator by hand, run from the directory that holds `iterate-family-0.5-evidence`, with relative names:

```sh
python3 -I -B REPO/benchmark/baselines/tools/make_baseline.py --reps 3 \
  --groups iterate-family-0.5-evidence/base-groups --layouts iterate-family-0.5-evidence/base-layouts \
  --benchmark-dir REPO/benchmark \
  --harness-commit 5d43d165722bfb5a401abf735ba2a673d9ee0946 \
  --source-commit def81ec9d021c39ed1727cfeb9598357291dd60d \
  --crosscheck iterate-family-0.5-evidence/crosscheck-pinned-start.tsv \
  --crosscheck-end iterate-family-0.5-evidence/crosscheck-pinned-end.tsv \
  --restore-probe iterate-family-0.5-evidence/restore-probe.txt \
  --aa-compare iterate-family-0.5-evidence/cmp-groups/comparison.csv iterate-family-0.5-evidence/cmp-layouts/comparison.csv \
  --out-md iterate-family-0.5.md --out-csv iterate-family-0.5.csv
```

`REPO/benchmark` must hold the `validate.py` whose `timer_limited` the generator imports. `-I` stops Python from
importing from the script's directory and `-B` stops it from writing bytecode.

## The archive

`data/iterate-family-0.5-evidence.tar.gz` is 1,502,753 bytes.

| | SHA-256 |
|:--|:--|
| the `.tar.gz` | `ae0235adad9f932ceb95ea3ad1709621d73fa0a506dd231b9c442fd005b1bf87` |
| the uncompressed tar | `f8f20aeeb0ce332a7c19b70a962b2165ae51f4737b9bb11407959dc7dd0c6f3e` |

`reproduce.py` pins the first value. The second does not depend on the deflate implementation: zlib-ng at
level 9 produces other compressed bytes for the same tar, so a rebuild with it differs in the `.tar.gz`
only. The archive holds 45 files under `iterate-family-0.5-evidence/`, sorted by name, as regular files with
mode 0644, mtime 0, owner 0 and no owner names, in GNU tar format, with gzip mtime 0 and no stored name.

To rebuild it from an extraction (`E` as above), with Python 3.11 or later and a zlib that is not zlib-ng:

```sh
python3 -I -B benchmark/baselines/tools/pack_evidence.py E rebuilt.tar.gz
```

The packer refuses any file set other than the 45 files below, any symbolic link, and an existing output file.
It prints both SHA-256 values.

### Members

Origin labels:

- retained: copied from the measurement directories of the baseline run;
- retained, redacted: retained, with the values of `host`, `source_path`, `expected_package_root` and
  `slurm_job_id` replaced by `"redacted"` (the key names stay);
- recovered from a record: the original file is lost; the contents come from a verbatim record (provenance
  record, section 3);
- header only, reconstructed: the original file is lost; the harness wrote only the header line when no case
  failed (provenance record, section 4).

| member | SHA-256 | origin |
|:--|:--|:--|
| `aa-groups/julia-1.13.3-rep01-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `aa-groups/julia-1.13.3-rep01-metadata.toml` | `50be5095f8f3d07bd39b9fa904260a57c7358576ff61eac0239d65cafc181389` | retained, redacted |
| `aa-groups/julia-1.13.3-rep01-metrics.tsv` | `8af3295b2fc4edc074a50e79e134d241a7a637f9ebaa872e94dd676f17b00cb6` | retained |
| `aa-groups/julia-1.13.3-rep02-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `aa-groups/julia-1.13.3-rep02-metadata.toml` | `ca82d455f988c733de62be814b9bae38e19f7ade0e537e50f26622d2bc247a01` | retained, redacted |
| `aa-groups/julia-1.13.3-rep02-metrics.tsv` | `cd5d92489ec300719fb0d72e75ff8004368aa06b49ee26195e9cf6cdc003d604` | retained |
| `aa-groups/julia-1.13.3-rep03-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `aa-groups/julia-1.13.3-rep03-metadata.toml` | `a101224dbb93620113da4693e97e7d0f2c4b704d0eeacab2013349b65fe82057` | retained, redacted |
| `aa-groups/julia-1.13.3-rep03-metrics.tsv` | `acccb2f14aa3c5a7f535a6dd311b90859f9e8d143c306d9f5dffc01520ee2499` | retained |
| `aa-layouts/julia-1.13.3-rep01-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `aa-layouts/julia-1.13.3-rep01-metadata.toml` | `fb61a8ffcd08b819f111f2c177a513130d981cd15cb4e7d2b156e1b270717ef5` | retained, redacted |
| `aa-layouts/julia-1.13.3-rep01-metrics.tsv` | `398c297a8203a0fa656b94412da465f9ff82a97bfff7ef2dce0f405526c97aa7` | retained |
| `aa-layouts/julia-1.13.3-rep02-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `aa-layouts/julia-1.13.3-rep02-metadata.toml` | `dc80cce7726c3fc5270cbbc72f09b45460e52280f5c537bb06f4972854d6d9be` | retained, redacted |
| `aa-layouts/julia-1.13.3-rep02-metrics.tsv` | `c8701942f7f7cfbf50ddb657d60406692f95cc8e8bdcdde69066f33942132dcd` | retained |
| `aa-layouts/julia-1.13.3-rep03-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `aa-layouts/julia-1.13.3-rep03-metadata.toml` | `ebd796d99ed8c1993955c53228b6dbbffa7a23f95924953fc2c5a4d51ccbc7d6` | retained, redacted |
| `aa-layouts/julia-1.13.3-rep03-metrics.tsv` | `5e38ca73a30f11b1aacacebfc9ab21b162cb009e5792a1698bac46e8a8bd7ec7` | retained |
| `base-groups/julia-1.13.3-rep01-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `base-groups/julia-1.13.3-rep01-metadata.toml` | `34c3bd5a6212ad3bde4668e2979c454d9f2d4face2d678cf400cbbafe9c627f7` | retained, redacted |
| `base-groups/julia-1.13.3-rep01-metrics.tsv` | `8191c248744b6130ac27614ac0036f0e32229c434a4989a92af9b1003c8398b7` | retained |
| `base-groups/julia-1.13.3-rep02-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `base-groups/julia-1.13.3-rep02-metadata.toml` | `b59f86a73bc3fdfb5b90610dc93a447ef41a8a821af2c748aa67e388b4235e6a` | retained, redacted |
| `base-groups/julia-1.13.3-rep02-metrics.tsv` | `ce1ab9ea169986ea633fe78985d964fa1bfbedd8b49708626a247bdde7e1d42b` | retained |
| `base-groups/julia-1.13.3-rep03-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `base-groups/julia-1.13.3-rep03-metadata.toml` | `cc4ae80b0faa728c4bdf865e454c0b19d65a1615d110c56bb4301db26a9d6063` | retained, redacted |
| `base-groups/julia-1.13.3-rep03-metrics.tsv` | `243e5c9f2f2f45bcbb1480c1ce7cb3536e134f483f31c5cf152a05e61f12fa78` | retained |
| `base-layouts/julia-1.13.3-rep01-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `base-layouts/julia-1.13.3-rep01-metadata.toml` | `c599464fc2b76d6cdc4360981b3f89ef67a983ee61b744f559d3f4700761bf45` | retained, redacted |
| `base-layouts/julia-1.13.3-rep01-metrics.tsv` | `03d049ad4ed9f177d36686d7b7c6f9cef72815bec32899828ecb6dcfab7ae82f` | retained |
| `base-layouts/julia-1.13.3-rep02-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `base-layouts/julia-1.13.3-rep02-metadata.toml` | `e9bec9d5d9e6d8ff15a0b5e4f9ed22ef0ac42633d248b4135c266bd9924fde1a` | retained, redacted |
| `base-layouts/julia-1.13.3-rep02-metrics.tsv` | `d62b005b3b3dc69039ba6ac5cf0d42b61504f642976b6294839293efbd02caf8` | retained |
| `base-layouts/julia-1.13.3-rep03-errors.tsv` | `d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770` | header only, reconstructed |
| `base-layouts/julia-1.13.3-rep03-metadata.toml` | `9e919df723e85d8f1f30acc98f14cc6bda018bd50e4ba8c26ab9db6ec37ec8f7` | retained, redacted |
| `base-layouts/julia-1.13.3-rep03-metrics.tsv` | `7de2defb93c4d243c4c6048ba96e4c4c1787aa310366b65730dedecd9dc21701` | retained |
| `cmp-groups/REPORT.md` | `492c49bf1433cb0caed3ef6764f4b7fe1a8e9d3693650d3e83890e8e7499aed8` | retained |
| `cmp-groups/comparison.csv` | `bc105706fe25a2119ec054cb4862a0ba1884ee03af2ec6bcab50807281d8716f` | retained |
| `cmp-groups/coverage.json` | `ae7aec7a3fc0aac5f71d64177b07867d9c8c2ea109fb29e4d4c6200bf68df9db` | retained |
| `cmp-layouts/REPORT.md` | `5b3e6b782c63a7968f59fd54a8fdef9883ca655b9b0cb7f0aee466aa1a607532` | retained |
| `cmp-layouts/comparison.csv` | `eb929a479c2aba1ef60bef975973c2c1571706e6d6b2f1608e9ffb41dc7d3b29` | retained |
| `cmp-layouts/coverage.json` | `b172d397ea1f1c319e49dd219a6a221d6628c3774c4f7da17fe8a6b34fff89c8` | retained |
| `crosscheck-pinned-end.tsv` | `09815345aae65feefbbe04c42bfc273f36f6e54b76c510f761ad078bdb24cd6a` | recovered from a record |
| `crosscheck-pinned-start.tsv` | `24b600ceabeb75b22eb4344c541f9006f8923f62ca7348e7cc8e7fcc73bd6368` | recovered from a record |
| `restore-probe.txt` | `f10da4fa60f634c066ce0f9941eb33f90f26bf025eacbc4c797131c21d559de5` | recovered from a record |

The redaction changes the SHA-256 of the twelve metadata files relative to the inventory of the provenance
record, and nothing else: against the unredacted files, a diff touches exactly those four keys in each of
the twelve files. The generator's sixteen identity keys do not include these fields, the generator's
outputs are byte-identical with and without the redaction, and `verify_csv3.py` reports no mismatches either
way. Before the redaction, all twelve metadata files recorded the same host and the same scheduler job id, so
a cross-file equality check on these fields loses nothing.

## What the archive does not contain

- The per-sample files, profiles and allocation stacks of the runs, and the calibration files. The raw
  `mean_ns`, `median_ns`, `evals` and `min_ns` values of every case are in the metrics files; the
  samples they were computed from are not retained.
- The historical `compare.py` cannot be re-run on this data, since it needs the calibration files. The two
  `comparison.csv` files are retained as its output.
- `check_only.jl` (now `benchmark/check_only.jl`) and `sabotage2.jl`, listed in the original notes below, are
  not included: they test the harness, not the baseline.

The tools have the limitations listed in section 7 of the provenance record. The generator checks its
identity keys only across the base metadata, and `verify_csv3.py` exits 0 whatever it finds.

## Python versions

The generator is byte-identical in its outputs on Python 3.11.9, 3.12.8, 3.12.15, 3.13.2 and 3.13.13
(provenance record, section 5). `reproduce.py` and `pack_evidence.py` were run on 3.11, 3.12 and 3.13 when they were added. The generator
writes the md in the platform's text mode, so the outputs are byte-identical on Linux and macOS (newline `\n`);
other platforms are not tested.

## Original notes

The notes below are the README that accompanied the tools before they were added to the repository.
They predate the move: the paths and the run they mention are of that time, `check_only.jl` is now
`benchmark/check_only.jl`, and `sabotage2.jl` is not included.

````text
# Tools of the #420 baseline run (kept outside the repository)

Written by the orchestrator during run `run-20261101T1915-issue420-19c1` (PR #429, merged as 9dba71ee).
They are not part of the repository; the committed baseline files were produced with them.

- `make_baseline.py`: generates `benchmark/baselines/iterate-family-0.5.{md,csv}` from the result directories of
  three repetitions (groups and layouts), the A/A comparison (`compare.py` output of set `base` against set `aa`),
  the two independent-loop files and the restore probe. Run with Python 3.13. The invocation of the committed baseline:

      python3.13 make_baseline.py --reps 3 --groups <base-groups dir> --layouts <base-layouts dir> \
        --benchmark-dir <worktree>/benchmark --harness-commit 5d43d165722bfb5a401abf735ba2a673d9ee0946 \
        --source-commit def81ec9d021c39ed1727cfeb9598357291dd60d --crosscheck <crosscheck start tsv> \
        --crosscheck-end <crosscheck end tsv> --restore-probe <restore-probe.txt> \
        --aa-compare <compare-groups/comparison.csv> <compare-layouts/comparison.csv> --out-md <md> --out-csv <csv>

  where the compare directories are `compare.py --expected-runs 3 base-* aa-*`. The cross-check text about the Intel
  node is hard-coded for this baseline.
- `crosscheck.jl`: plain `time_ns` loops for first activation (256 and 4096 coordinates), the active sweep and `dot(v, x)`.
- `restore_probe.jl`: the first-activation loop with two restore methods (`dropzeros!` against `copyto!`).
- `check_only.jl`: builds the harness cases and runs every reference check without timing.
- `sabotage2.jl`: the sabotage tests of review round 2 (each strengthened check must fail under its sabotage).
- `verify_csv3.py`: recomputes every csv figure and the A/A columns from the raw metrics and metadata.
- `identity.py`: lists the identity keys of the metadata files of the result directories.

Issues that need them: #461 (first-activation rows against an independent loop), #462 (gate coverage),
#440 (check-only mode).
````
