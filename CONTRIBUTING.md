# Contributing

Use `UpperCamelCase` for modules and types, `snake_case` for descriptive
functions, fields and keyword arguments, and a trailing `!` for mutating
functions. Retain established Julia interface names, familiar numerical
abbreviations, and mathematical identifiers such as `X`, `Σ` and `λ`.
Deprecated names are compatibility exceptions, not examples for new code.

For standalone development, run from the package root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Julia installs the declared dependencies and test dependencies automatically.
No clones of ProximalSolvers or other related packages are required, and you
can work on any Git branch.

## Tests

The test script used by continuous integration (CI) calls `Pkg.test()` with array
bounds checking enabled; no separate setup command is needed. Test files run in
parallel worker processes, and every file belongs to one of four cumulative tiers
listed in [`test/tiers.toml`](test/tiers.toml):

| Tier | Runs | Target on CI |
|---|---|---|
| `quick` | representative cases of the whole package; every PR | 3–5 min |
| `standard` | quick, plus most function families, wrappers and constraints | 8–10 min |
| `thorough` | standard, plus numerics, conformance, symmetric and iterate catalogs | 15–20 min |
| `full` | every test file, including the exhaustive catalogs | about 30 min or more |

While working on a change, run the files that test it, then a tier:

```sh
julia scripts/test.jl sparse_iterate atom_iterate   # files whose names start with these
julia scripts/test.jl --tier=quick                  # early feedback
julia scripts/test.jl --tier=standard               # before an ordinary implementation PR
julia scripts/test.jl --tier=thorough               # numerical or structural changes
julia scripts/test.jl                               # every file (the full tier)
```

`--tier=quick --list` lists a tier's files, `--extra=gpu,metric` adds files to a
tier, and names after a tier filter it (`--tier=thorough symmetric`). `--jobs=N`
or `PTR_NUM_JOBS` sets the number of workers. The same arguments work through
`Pkg.test(; test_args=...)`. Set `JULIA_TEST_COVERAGE=true` to collect coverage, or
`JULIA_TEST_DEPWARN=error` to treat deprecation warnings as errors. `--verbose`
adds each file's start, initialization time and compilation share. The old
`--core` switch still runs every file except `precision_mixed`, `large_inputs`,
`spectral_overflow` and `conformance/*`, with a deprecation warning; use a tier
instead.

**A new test file** goes into `test/tiers.toml` under the lowest tier that should
run it: `quick` for a cheap test of a basic invariant or a serious regression,
otherwise the tier of the code it tests. An unlisted file stops every test run and
the CI plan, and a PR that changes `test/tiers.toml` runs the full tier. Files run alone in a sandbox module, and compilation takes most of
their time, so a cheap file reaches few type combinations. Helper files that are
only included go under `[not_tests]`. [`test/TESTING.md`](test/TESTING.md) records
why each file sits in its tier.

### Which tests CI runs

The plan job of the CI workflow picks the tier of each run with
`scripts/ci_plan.jl` and the rules in
[`.github/tier-rules.toml`](.github/tier-rules.toml); the run's summary page shows
its decision for every changed file. `julia scripts/ci_plan.jl` prints the same
decision for your working tree against `origin/master`.

- **Pull requests** run Julia lts and Julia 1 on Ubuntu at the minimum tier for
  the changed files, and beside it the changed test files and the focus files
  of the matched rules:
  - documentation only: no package tests (the API pages, whose entries
    `test/docstrings.jl` checks, get quick);
  - test files, examples or README only: quick;
  - a self-contained function family together with a test change: quick and that
    family's own test files;
  - one iterate implementation (`src/atom_iterate.jl`, `src/sparse_iterate.jl`,
    `src/symmetric_parameters.jl`): standard and that iterate's focus files, or
    quick and the same files with the label `test:iterate-narrow` (see
    "Iterate changes" below);
  - other package code (`src/`): standard; code shared by every function
    (`interface.jl`, `utils.jl`, `calculus.jl`, `metric.jl`, `partial.jl`,
    spectral helpers) or the conformance suite and its coverage map: thorough;
  - package code without any test change: at least standard;
  - `Project.toml`, `ext/`, `src/lapack.jl`, the test runner, the CI rules and
    workflow, and `test/tiers.toml`: full, which also runs Julia 1 on macOS and
    Windows and Julia pre. The tier lists decide what every tier runs, so a
    change to them gets full, unless it only adds the lines of test files that
    the PR adds: the plan compares the parsed lists with the base, and then the
    new files run beside the PR's tier. Moving, removing or renaming a listed
    file, or changing the helpers or environments, still gets full;
  - a path no rule knows: full.
- **Labels** `test:standard`, `test:thorough` and `test:full` raise a PR's tier;
  they never lower it. `test:iterate-narrow` declares a narrow change to one
  iterate implementation; it lowers only that implementation's rule, never
  another path. Adding or removing any label starts a new run. Add
  `test:thorough` when a change touches element-type promotion, precision,
  overflow-safe scaling or non-finite handling, which the paths cannot show, and
  `test:full` for a release or a platform-sensitive change. A label starts a
  new run on the same commit, and its gate replaces the earlier gate as a
  pending check as soon as GitHub creates the run (a few seconds after the
  label), so an earlier green gate cannot be used to merge.
- **Pushes to master and tags** run the full tier on all five jobs (none for a
  documentation-only push), with coverage and the conformance reports. A
  **nightly** full run catches changes in dependencies and Julia releases; it is
  skipped when a full run already passed on the same commit that day.
- **Manual runs** test any branch at any tier, full by default:
  `gh workflow run CI.yml --ref <branch> -f tier=thorough`. On a PR branch the
  result appears on the head commit beside the PR's own checks.

### Iterate changes

`SparseIterate`, `AtomIterate` and `SymmetricSparseIterate` change in many small
PRs during the 0.6 work. A change to one of them runs the iterate's own tests and
those of its main interactions (its focus files), not the unrelated families
that thorough adds. The focus files are in `.github/tier-rules.toml`, beside the
iterate files that quick always runs (`active_handle`, `atom_iterate`,
`iterate_queries`, `matrixview`, `sparse_iterate_basics`,
`symmetric_parameters`, `symmetric_products`, `wrapper_buffers`,
`allocations_basics`):

| Change | Before merge |
|---|---|
| `src/atom_iterate.jl`, narrow | quick + `real_iterates`, `block_step`, `norms`, `sparsegroup`, `infinite_group_weights` |
| `src/sparse_iterate.jl`, narrow | quick + `sparse_iterate`, `real_iterates`, `block_step`, `sparse_value_cost`, `conformance/failures_storage`, `symmetric_writes`, `symmetric_mul`, `symmetric_argument`, `symmetric_mutators`, `symmetric_mutator_errors` |
| `src/symmetric_parameters.jl`, narrow | quick + `symmetric_parameters_paths`, `_trait`, `_writes`, `_merge`, `_extensions` |
| one of them, not declared narrow | standard + the same focus files |
| two of them | standard + both groups (the label does not apply) |
| shared semantics across iterate types, or an architectural or breaking milestone | thorough: add `test:thorough` |
| release candidate | full: add `test:full` |
| merge to master | full, automatically |

`src/sparse_iterate.jl` holds `SparseIterate`, `SymmetricSparseIterate` and the
stored-iterate code they share, so a change to it runs the tests of both; the
file name cannot tell a narrow `SparseIterate` fix from a symmetric redesign.
Neither can any rule tell a narrow change from an architectural one: the author
states the classification in the PR description (the PR template asks for it),
and the reviewer checks it.

- **Narrow** (`test:iterate-narrow`): a local fix or addition inside one
  implementation that keeps its documented behavior, with a regression test.
  Example: an agent fixes the group indexing of `AtomIterate`, adds a test to
  `test/atom_iterate.jl` and sets the label; CI runs quick and the atom focus
  files, and the PR merges once both required checks pass.
- **Shared**: behavior that more than one iterate type uses, such as the stored
  iterates' activation, storage, copying or aliasing. Example: an agent changes
  how `dropzeros!` treats stored zeros for both stored iterates; no label, so
  CI runs standard and the focus files. Add `test:thorough` when the change
  affects working-set semantics across types or numerical behavior (zeros,
  infinities, NaN, precision).
- **Architectural** (always `test:thorough`, never narrow): a new supertype or
  public interface of an iterate; a new meaning of `fill!`, `copyto!`,
  broadcasting or coordinate access; working-set, aliasing or ownership
  guarantees across interfaces; `SymmetricSparseIterate` as a coordinate-first
  owner (no longer `<: AbstractMatrix`); removing much of the old symmetric
  array interface; changes visible to downstream packages.
- **After merging**, the push to master runs full. If it fails, fix or revert
  the change before stacking dependent PRs on master, and when the failure is a
  bug the focus files should have caught, add a regression test to them (or to
  quick).

The label never lowers a rule other than the iterate rules: a PR that also
changes `src/utils.jl` stays thorough, one that changes `Project.toml` or the
workflow stays full, and other package code without tests still needs standard.
To add regression tests without a full run, add a new test file and its line in
`test/tiers.toml` (an addition alone keeps the PR's tier), or extend an existing
test file.

Branch protection on master requires two checks, both from GitHub Actions:
**PR required jobs** and **PR merge gate**. PR required jobs runs after the
plan, the tests and the benchmark analysis controls, and passes when the plan
succeeded, the benchmark analysis controls passed and every test job of the
chosen tier passed; Julia pre is advisory. A failed, cancelled, skipped or
missing test job fails it, and so does a plan that found an unclassified test
file. The gate job starts with the run, so a push or a label replaces the
commit's earlier green gate with a pending one within seconds; it waits for
the run's PR required jobs and passes only if they passed. Pushes, the nightly
run and manual runs name the two checks **CI required jobs** and **CI
result**, so a manual run on a PR branch cannot stand in for the PR's checks.
Coverage is collected only on full runs, so Codecov reports on master and on
full PR runs.

To retry jobs of a PR run:

- **Re-run failed jobs** re-runs the failed jobs, PR required jobs and, if it
  failed, the gate; the PR stays blocked until the new results pass. If only
  the gate failed (for example on a GitHub API outage), it re-runs only the
  gate, which then takes the run's PR required jobs result.
- **Re-run all jobs** starts the whole run again, with a pending gate.
- **Re-run job** on one test job re-runs it and PR required jobs, but not the
  gate. Until the re-run creates PR required jobs again, the PR's checks lack
  it, so the PR is blocked while the job runs and stays blocked if the job
  fails. After a failed run, the gate keeps its failure even when the re-run
  passes; then use **Re-run failed jobs**, which re-runs only the gate.

Each test job's summary page lists the selected files, the workers, the time
before the test runner started (Julia, test environment, precompilation), the
runner's wall time, the worker time of every file with its compilation share,
and the slowest files. Thorough and full runs also write the conformance reports
(every job checks that they exist), check the committed coverage map (Julia
lts) and upload the reports. An upload counts only when GitHub returns an
artifact id for it; otherwise it is retried once. The Julia lts, Linux
artifact, from which the coverage map is regenerated, is required: without a
confirmed upload that job fails. On the other jobs a missing upload is a
warning in the job and its summary, and the job keeps the result of its tests.

**When a master or nightly run fails**, the workflow opens (or comments on) the
issue "CI failing on master". Fix the regression or revert the PR that caused it
the same day; reproduce locally with `julia scripts/test.jl <file>` and the Julia
version of the failing job. A PR that merged after quick or standard is not at
fault for passing its gate; the post-merge run exists to catch what the smaller
tiers leave out.

Documentation, benchmarks and optional conic tests each have their own
`Project.toml` with additional tools. The setup script installs those tools
and calls `Pkg.develop` to select this local ProximalBase checkout, including
on Julia 1.10 where `[sources]` is not supported. It preserves the environment's
`Project.toml` and writes a local manifest. It does not clone other repositories
or check Git branches.

For example, to build documentation for your current changes, run from the
package root:

```sh
julia scripts/instantiate.jl docs
julia --project=docs docs/make.jl
```

To prepare the benchmark environment:

```sh
julia scripts/instantiate.jl benchmark
```

For the optional conic tests, use `julia scripts/instantiate.jl test/oracles`;
see [the conic test instructions](test/oracles/README.md) for running them.
Calling `julia scripts/instantiate.jl` without an argument simply installs
the package's declared dependencies, like `Pkg.instantiate()`.

Use the [benchmark suite](benchmark/README.md) to measure runtime and allocations
before and after implementation changes. Keep generated measurements with the
PR that uses them; add reusable cases for workloads worth protecting.

Release tags must match the version in `Project.toml`. Labels under
`docs/audit` describe historical audit plans; for example, `release-0.5.4`
does not set the package version.
