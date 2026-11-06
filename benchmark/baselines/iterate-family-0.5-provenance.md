# Provenance and reproduction record of the iterate-family 0.5 baseline

This page records where every input of [`iterate-family-0.5.md`](iterate-family-0.5.md) and
[`iterate-family-0.5.csv`](iterate-family-0.5.csv) came from, what was verified, and what rests on a
record alone. It was written for issue #721 (phase B0 of #463, the parent issue). It adds no measurement
and changes neither baseline file. The next phases are #722 (B1, archive the inputs and tools) and
#723 (B2, a regression check on the archived inputs).

The committed files are:

| file | SHA-256 |
|:--|:--|
| `iterate-family-0.5.md` | `9055d230ecb74a654d5bb3beb315d25b1298260ae6373615b90d7edf99a968f2` |
| `iterate-family-0.5.csv` | `cb8348bb8099e91fce8ac733454682a82f26d875db9f98ad92bc0a8479c71705` |

## 1. Identity of the measurement

The identity table of the baseline (source commit, harness commit, Julia, CPU, settings) is in
[Identity of the measurement](iterate-family-0.5.md#identity-of-the-measurement). It is not copied
here. Two of its values were recomputed independently from git, with the `fingerprint` rule of
`benchmark/run.jl` (SHA-1 over the sorted `relpath\0bytes\0` records):

| key | value | recomputed from |
|:--|:--|:--|
| `harness_hash` | `844fa5330b9065ab33fc7bb9ef7605dbe8560e16` | `benchmark/cases.jl` and `benchmark/run.jl` at the harness commit `5d43d165722bfb5a401abf735ba2a673d9ee0946` |
| `source_hash` | `a1583f070615bd3ea38705eacda4401ca1c6bafd` | `Project.toml`, `src/**/*.jl` and `ext/**/*.jl` (51 files) at the source commit `def81ec9d021c39ed1727cfeb9598357291dd60d` |

Both equal the values recorded in all twelve metadata files (`harness_hash` and `source_hash`).

## 2. Inputs

The baseline is generated from three repetitions of two result directories, measured on the pinned
source (`base-*`), and from a second measurement of the same source with the same harness (`aa-*`, the
A/A control). Each of the four sets (`base-groups`, `base-layouts`, `aa-groups`, `aa-layouts`) holds
repetitions 1 to 3.

| input | role | state |
|:--|:--|:--|
| `julia-1.13.3-rep0{1,2,3}-metrics.tsv` and `-metadata.toml` in `base-groups`, `base-layouts`, `aa-groups`, `aa-layouts` | measured metrics and run metadata | retained (24 files) |
| `cmp-groups/comparison.csv`, `cmp-layouts/comparison.csv` | A/A comparison of `base-*` against `aa-*`, output of `compare.py --expected-runs 3` | retained |
| `cmp-{groups,layouts}/coverage.json`, `REPORT.md` | by-products of the comparison | retained, read by none of the original tools |
| `crosscheck-pinned-start.tsv`, `crosscheck-pinned-end.tsv`, `restore-probe.txt` | independent-loop cross-check and the restore probe | recovered from a record (section 3) |
| twelve `julia-1.13.3-rep0{1,2,3}-errors.tsv` | failed cases of each run | not retained (section 4) |

The archive of these inputs is `data/iterate-family-0.5-evidence.tar.gz`; `tools/reproduce.py` regenerates both baseline files from it (see [`tools/README.md`](tools/README.md)).

### Custody of the retained metrics, metadata and comparison files

These files were copied on 2026-11-02, before the measurement campaign directory was removed. No hash
was taken at copy time. Samples, profiles, allocation stacks and calibration files were deliberately not
copied. The files are corroborated, not signed:

- the generator, run on them, reproduces both committed outputs byte for byte (section 5);
- the `comparison.csv` files were recomputed from the `mean_ns` values of the retained base and A/A
  metrics: `paired_geometric_speedup`, `ci95_low` and `ci95_high` are exactly equal (`==`) for 3,719
  of 3,719 cases (probe P6, section 6);
- the `harness_hash` and `source_hash` of all twelve metadata files equal the independent recomputation
  of section 1 (probe P5).

The full per-file inventory with SHA-256 is on
[PR #725](https://github.com/mlakolar/ProximalBase.jl/pull/725#issuecomment-6693712712), not here: the
B1 archive redacts the host, path and job fields of the metadata files, which changes those hashes. The generator and
`verify_csv3.py` do not read these fields (probe P7); `identity.py` displays the host and job fields
(section 7).

## 3. Recovery of the three probe files

The originals of `crosscheck-pinned-start.tsv`, `crosscheck-pinned-end.tsv` and `restore-probe.txt` are
no longer available. Their contents come from a verbatim record that was printed in full when the files
were inspected after transfer (2026-11-02). `restore-probe.txt` has a second, independent verbatim record,
taken before transfer. No hash of the originals exists, and nothing was re-measured.

The recovered files have these SHA-256 values. They identify the recovered files used for the
reproduction, not the lost originals.

| file | SHA-256 |
|:--|:--|
| `crosscheck-pinned-start.tsv` | `24b600ceabeb75b22eb4344c541f9006f8923f62ca7348e7cc8e7fcc73bd6368` |
| `crosscheck-pinned-end.tsv` | `09815345aae65feefbbe04c42bfc273f36f6e54b76c510f761ad078bdb24cd6a` |
| `restore-probe.txt` | `f10da4fa60f634c066ce0f9941eb33f90f26bf025eacbc4c797131c21d559de5` |

What byte equality pins, and what it does not:

- The csv does not depend on the three files at all.
- The md pins the rendered cross-check medians (the four rendered rows and the Float32 `x16` row, at
  three significant figures), the start/end pair with the largest drift, and the two `nb=16`
  restore-probe values.
- The `min_ns_per_op` column, the other rows, the full digits, the file framing and the `nb=1` lines
  rest on the verbatim record alone. The record is supported by its framing (`echo` and `---` lines show
  where each file ends), by the format of the code that wrote the files, and by tabs preserved
  throughout.

## 4. Error files

The twelve `*-errors.tsv` files were not retained. The harness at `5d43d165` (`benchmark/run.jl`) writes
one header line, `id`, `exception`, `stack` separated by tabs (`id\texception\tstack\n`, 19 bytes, SHA-256
`d0a9d07a6d313cbd138f327bc9e16d1ed43bf7af80ae9dc26a53b0dcb7954770`), and one row per failed case. The
harness exits with an error on any failure, and all twelve metadata files record `failures = 0`. So each
original file very probably held the header only. This is an inference from the code and from `failures`.
It is not load-bearing: the generator only checks that these files have no rows, and a 0-byte file gives
the same outputs.

## 5. Reproduction

The baseline was generated with `make_baseline.py`, written for the 0.5 baseline and kept outside the
repository (B1 archives it). Its README gives this invocation of the committed baseline:

```sh
python3.13 make_baseline.py --reps 3 --groups <base-groups dir> --layouts <base-layouts dir> \
  --benchmark-dir <worktree>/benchmark --harness-commit 5d43d165722bfb5a401abf735ba2a673d9ee0946 \
  --source-commit def81ec9d021c39ed1727cfeb9598357291dd60d --crosscheck <crosscheck start tsv> \
  --crosscheck-end <crosscheck end tsv> --restore-probe <restore-probe.txt> \
  --aa-compare <compare-groups/comparison.csv> <compare-layouts/comparison.csv> --out-md <md> --out-csv <csv>
```

The exact form of the reproduction, with relative names, a `--benchmark-dir` that holds the
`validate.py` under test, and isolated mode:

```sh
python3 -I make_baseline.py --reps 3 --groups base-groups --layouts base-layouts \
  --benchmark-dir <directory holding validate.py> \
  --harness-commit 5d43d165722bfb5a401abf735ba2a673d9ee0946 \
  --source-commit def81ec9d021c39ed1727cfeb9598357291dd60d \
  --crosscheck crosscheck-pinned-start.tsv --crosscheck-end crosscheck-pinned-end.tsv \
  --restore-probe restore-probe.txt \
  --aa-compare cmp-groups/comparison.csv cmp-layouts/comparison.csv \
  --out-md iterate-family-0.5.md --out-csv iterate-family-0.5.csv
```

Results:

| probe | what | result |
|:--|:--|:--|
| P1 | generator with `validate.py` at `5d43d165722bfb5a401abf735ba2a673d9ee0946` (harness commit) | md and csv byte-identical to the committed files |
| P1 | the same with `validate.py` at `233c11cc284284fbd29df487308b4ee962f875a1`, last changed by `d4c3f70d692dd53a263d132513f1428a61d3463f` (#471) | byte-identical. `timer_limited` and `check_timer_limit_multiple` are textually identical in the two versions; the differences are confined to `validate()` and the helpers it uses |
| P1 to P3 | Python 3.11.9, 3.12.8, 3.12.15, 3.13.2 and 3.13.13 | identical results and byte-identical outputs on every version |
| P2 | the original `verify_csv3.py` on the committed csv | 3,719 rows, no mismatches; 94 timer-limited (either side, by its own inline rule), 3,625 not; 1,092 can gate (csv: 1,092); 104 A/A-conclusive among the 3,625 |
| P3 | count with each `validate.timer_limited`, independent of the generator | 3,719 cases, 94 timer-limited, with both `validate.py` versions |
| P4 | `harness_hash` planted in one base metadata file (one line changed) | generator exits 1 with `metadata key harness_hash differs`; `verify_csv3.py` exits 0 with no mismatches; `identity.py` lists both values and exits 0 |
| P4 | `harness_hash` planted in one A/A metadata file (one line changed) | generator exits 0, outputs byte-identical; `verify_csv3.py` exits 0; `identity.py` lists both values and exits 0 |
| P5 | fingerprints recomputed from git | both equal the recorded values (section 1) |
| P6 | archived `comparison.csv` recomputed from the base and A/A `mean_ns` | with t = 4.303 (the table in `benchmark/summarize.py` at `5d43d165`): `paired_geometric_speedup`, `ci95_low` and `ci95_high` exact for 3,719 of 3,719. With t = 4.302652729911275: speedup exact, 0 of 3,719 bounds exact |
| P7 | host, path and job fields redacted in all twelve metadata files (48 lines changed) | generator exits 0, outputs byte-identical; `verify_csv3.py` reports no mismatches |

The acceptance test is the byte comparison of the generated md and csv with the committed files.

## 6. Counts in the committed outputs

Recomputed from the committed csv and from the archived comparison files.

- 3,719 unique ids: 3,455 in the groups run (`atom` 2,688, `owner` 157, `products` 108, `sparse` 502) and
  264 in the layouts run.
- 94 `timer_limited`: the same set on the base side, the A/A side and in `comparison.csv`.
- 1,092 `gate5`; 104 `work_varies`, all in `products`.
- 104 of the 3,625 non-timer-limited cases are flagged by the A/A interval: 55 `slower` and 49 `faster`
  in `timing_assessment`.

### A wording discrepancy in the committed md (recorded, not changed)

The md says "2533 have an A/A interval wider than ±5%". The number 2,533 is the count of
non-timer-limited cases that cannot gate (3,625 minus 1,092). Of these, 2,531 have an A/A half-width above
5%. The other two are the diagnostic `first_activation` rows of `sparse`, which are forced off
the gate with half-widths 0.0242 and 0.0434. The committed md is left as it is; the baseline files do not change in #463.

## 7. Limitations of the original tools

The original tools are preserved by B1 (#722) and their limitations are addressed by B2 (#723).

`verify_csv3.py`:

- exits 0 whatever the mismatch count (P2, P4); it fails only on a missing file or id (B2);
- compares at printed precision (relative 6e-4, plus 5e-4 absolute for the spread and the A/A fields),
  and uses t = 4.302652729911275, which differs from the historical comparator's 4.303 (P6). B2 uses
  4.303 when it checks the archived comparison (B2);
- loops over csv rows only, so it misses rows that are absent or duplicated (B2);
- never reads `cmp-*/comparison.csv`, the md, the error files, `failures`, `harness_hash`, or any
  metadata key except `timer_floor_ns` (B2).

The generator:

- checks sixteen identity and settings keys (among them `harness_hash`, `source_hash`, `source_sha`,
  `julia`, `cpu_model`, `os`, `check_bounds`) only across the base metadata, never across the A/A
  metadata (P4). It compares `source_sha` with `--source-commit`, but `--harness-commit` is only printed
  and is never compared with `harness_hash` (B2);
- ignores the `julia` and `paired_runs` columns of `comparison.csv`, and extra ids in it (B2).

Other tools:

- `identity.py` only displays the identity keys (exit code 0 always) and prints host and job fields,
  so its output must not be published as is (B1 ports the script).
- The historical `compare.py` cannot be re-run on the retained data, because it requires the calibration
  files, which were not retained. The comparison files can still be recomputed exactly from the retained
  metrics (P6) (B2).
- The NaN-floor branch of `timer_limited` is never exercised: all twelve recorded timer floors are 20 or 21 ns,
  and every row has `evals`.

## 8. What is verified and what is not

Verified by recomputation: the two fingerprints, the committed csv and md from the retained inputs, the
3,719 comparison rows, the counts of section 6, and the outputs on five Python versions.

Not verifiable: the bytes of the three recovered files beyond what section 3 pins; the contents of the
error files; the retained metrics against their originals, for which no hash exists; and the
per-repetition `min_ns` and `median_ns` values against the original timing samples, which were not
retained. Any later use of this record should state these four points again.
