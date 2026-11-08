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
| `full` | every test file, including the exhaustive catalogs and the device-array suites on GPUArrays' reference backend (`gpu`, `gpu_smoke`, `conformance/gpu_arrays`) | about 30 min or more |

While working on a change, run the files that test it, then a tier:

```sh
julia scripts/test.jl sparse_iterate atom_iterate   # files whose names start with these
julia scripts/test.jl --tier=quick                  # early feedback
julia scripts/test.jl --tier=standard               # before an ordinary implementation PR
julia scripts/test.jl --tier=thorough               # numerical or structural changes
julia scripts/test.jl                               # every file (the full tier)
```

`--tier=quick --list` lists a tier's files, `--extra=gpu,metric` adds files to a
tier (`--extra=gpu,gpu_smoke,conformance/gpu_arrays` runs the device-array suites beside a lower tier), and names after a tier filter it (`--tier=thorough symmetric`). `--jobs=N`
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
only included go under `[not_tests]`, and the test files that run a helper,
directly or through another helper, under `[not_tests.run_through]`, which the
error for a helper given as a test name uses. A new test file that includes a
helper is listed there too, and its PR then runs the full tier. Renaming or
removing a test listed there, or a helper that is a key of that table, means
updating the table, or every test run stops on the manifest problem.
[`test/TESTING.md`](test/TESTING.md) records why each file sits in its tier.

### Which tests CI runs

The plan job of the CI workflow decides what each run does with `scripts/ci_plan.jl`
and the rules in [`.github/tier-rules.toml`](.github/tier-rules.toml); for a pull
request, the run's summary page shows the decision for each changed file. `julia scripts/ci_plan.jl`
prints the same decision for your working tree against `origin/master`. The plan
makes five independent decisions: the **tier** (how many test files run), the
**platforms** (which Julia versions and systems run them), whether to collect
**coverage**, whether to run the **benchmark reference checks**
(`benchmark/check_only.jl` on the quick suite, Julia lts and 1), and, from the
tier, whether the test jobs write the **conformance reports** (thorough and full).

| Run | Tier | Platforms | Coverage | Benchmark reference checks |
|---|---|---|---|---|
| pull request | the minimum for the changed files, raised by labels | quick, standard, none: `stable`; thorough, full: `linux` | no | changed benchmark harness or `CI.yml`, or the label `test:benchmark` |
| pull request with `test:full` | full | `cross` | no | yes |
| push to master | thorough (none for documentation only) | `linux` | no | changed benchmark harness or `CI.yml` |
| tag push | full | `cross` | no | yes |
| nightly | full, also when an earlier run passed | `cross` | yes | yes |
| manual | the input `tier` (full) | the input `platforms` (auto) | the input `coverage` (off) | the input `benchmark`, and also with tier full |

The platform modes are `stable` (Julia 1 on Linux), `linux` (Julia 1 and lts on
Linux) and `cross` (`linux`, Julia 1 on macOS and Windows, and Julia pre on Linux,
which is advisory: reported, not counted). They are defined in
`.github/tier-rules.toml`; the planner validates its `[events.*]` and
`[platforms.*]` tables (an unknown key, mode or tier there is an error), and each
other table is validated by its own reader (`test/tiers.jl` or
`scripts/ci_plan.jl`). A push whose previous commit is unknown runs thorough.

- **Pull requests** run the tier for the changed files, and beside it the changed
  test files and the focus files of the matched rules:
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
  - the GPU test files and helpers (`test/gpu*.jl`): quick and `gpu`, `gpu_smoke`
    (the helpers) or the changed file; the conformance suite, its scripts and the
    coverage map also run `conformance/gpu_arrays`, so that the coverage check has
    its reports; the GPU extension (`ext/`) is full;
  - package code without any test change: at least standard;
  - `Project.toml`, `ext/`, `src/lapack.jl`, the test runner, the CI rules,
    scripts and workflow, and `test/tiers.toml`: full, on Julia 1 and lts on
    Linux (the other systems need the label `test:full`). The tier lists decide
    what every tier runs, so a change to them gets full, unless it only adds the
    lines of test files that the PR adds: the plan compares the parsed lists with
    the base, and then the new files run beside the PR's tier. Moving, removing or
    renaming a listed file, or changing the helpers, the environments or
    `[not_tests.run_through]`, still gets full;
  - a path no rule knows: full.
- **Labels** `test:standard`, `test:thorough` and `test:full` raise a PR's tier or
  leave it alone. `test:full` also selects the `cross` platforms (macOS, Windows
  and Julia pre besides Linux) and the benchmark reference checks; a path rule
  that needs the full tier runs it on `linux` only. `test:benchmark` adds the
  benchmark reference checks and does not change the tier or the platforms; it
  names no tier. `test:gpu` adds the device-array suites (`gpu_smoke`, `gpu`,
  `conformance/gpu_arrays`) to the run, whatever the paths; on a change that needs no
  package tests it runs quick with them. Add it to a change that can affect device
  arrays without touching the GPU files: those suites are otherwise in the full tier,
  which runs nightly and on tags and stays outside quick, standard and thorough (the CPU files that
  load JLArrays as a second array type, and the ambiguity guard for the GPU extension,
  `gpu_extension_ambiguities`, stay in the tiers below full). `test:iterate-narrow` declares a narrow change to one iterate
  implementation; it lowers that implementation's rule and no other path.
  Adding or removing a label starts a new run. Add `test:thorough` when a change touches
  element-type promotion, precision, overflow-safe scaling or non-finite
  handling, which the paths cannot show, `test:full` for a release or a
  platform-sensitive change, and `test:benchmark` for a change that can affect
  the benchmark suite without touching its files. A label starts a new run on the
  same commit, and its gate replaces the earlier gate as a pending check as soon
  as GitHub creates the run (a few seconds after the label), so the earlier green
  gate no longer stands.
- **Pushes to master** run thorough on Julia 1 and lts on Linux, with the
  conformance reports (none for a documentation-only push), and no coverage.
  **Tags** run the full tier on the `cross` platforms, with the benchmark
  reference checks. The **nightly** run is the scheduled run, and it collects
  coverage by default (a manual run can ask for it): the full tier on the `cross`
  platforms, with the benchmark reference checks, every night, also when a run
  passed on the same commit. It covers changes in dependencies and Julia
  releases and the platforms that pull requests and pushes do not run.
- **Manual runs** test any branch:
  `gh workflow run CI.yml --ref <branch> -f tier=thorough -f platforms=linux`.
  The inputs are `tier` (`full` by default), `platforms` (`auto`: `cross`
  for full, `linux` otherwise; `stable`, `linux` and `cross`
  choose a mode), `benchmark` (the reference checks; they also run with tier
  full) and `coverage` (collect coverage on Julia 1 on Linux and upload it to
  Codecov; the nightly run does so too). `tier=full -f platforms=cross -f
  coverage=true` runs what the nightly run runs. On a PR branch the result
  appears on the head commit beside the PR's own checks.
  One test file on Windows and macOS only: see "Focused runs on Windows and
  macOS" below.

#### Focused runs on Windows and macOS

The workflow `platform-focused.yml` runs one declared test file on a branch that
contains this workflow and its companion scripts, on native Windows and macOS
(Linux as a comparator on request), so that a platform bug can be reproduced and
its fix checked without a full cross-platform run. It is
advisory: nothing in `CI.yml`, its verdict or gate, or branch protection refers
to it, so a red run does not change a PR's required checks.

```sh
gh workflow run platform-focused.yml --ref <branch> -f os=both -f test_file=three_factor_product
gh run list --workflow platform-focused.yml --branch <branch>   # the run id
gh run watch <id>
gh run view <id> --log-failed
gh run download <id>
```

| Input | Meaning |
|---|---|
| `os` | `windows` (windows-latest, x64), `macos` (macos-latest, aarch64), `linux` (ubuntu-latest, x64), `both` (Windows and macOS, the default) or `all` (all three) |
| `test_file` | one declared test name, exactly as in `test/tiers.toml` (`three_factor_product`, `conformance/tolerances`); required |
| `julia_version` | `1` (the default), `lts`, `pre`, `1.N` or `1.N.P`; a `1.N` or `1.N.P` below the `julia` bound of `Project.toml` is refused. An old version may have no macOS aarch64 build |
| `expected_sha` | optional, 7 to 40 hex digits; the plan job fails unless they are the start of the dispatched commit |

- **One file.** `scripts/test.jl <name>` runs every test whose name starts with
  `<name>` (`blockchecks` selects five files). The workflow passes the name
  followed by `!<other>` for every other declared test that starts with it, and
  no `--tier` or `--extra`, so the runner's name filter keeps that file and no
  other declared test. A declared name that is also the start of other names
  (such as `blockchecks`) is accepted and runs only that file. When the check
  step runs (it does not after a cancellation), `scripts/platform_focus.jl
  check-log` reads the header that `test/runtests.jl` prints before any worker
  starts and fails the job unless it says `1 files` and lists exactly the file;
  it runs also when the tests failed.
- **Checks before any other system starts.** The plan job (Linux) refuses an
  unknown name, an undeclared prefix of test names (`three_factor`), a
  directory, a helper, an environment, a name that is not letters, digits, `_`
  and `/` (shell metacharacters, `..`, a leading `!` or `-`, a newline), a name
  whose file is missing, an `os` that is not in the table, a `julia_version`
  that is not in the table or is below the bound, an `expected_sha` that is not
  hex or is not the start of the dispatched commit, and a checkout that is not
  the dispatched commit. It reports the detected dispatch-input validation failures together rather
  than stopping at the first bad field, as `::error::` lines in which a rejected input is quoted and percent-encoded, so that an input cannot
  add a workflow command of its own. The jobs for the other systems need the
  plan job, so none is started when it fails. In the `run:` steps the inputs are
  read only from environment variables.
- **Results.** A job that reaches the test step writes the timing table, the
  failing lines and the last lines of the log, and the run metadata (commit,
  ref, run, system image, Julia `versioninfo`, selector, seconds) to its
  summary, and uploads the artifact
  `focus-<test name with / as ->-<system>-<julia version>-<attempt>` (14 days)
  with `test-output.log`, `metadata.txt` and a copy of the test file. This is
  best effort: the timing summary is `continue-on-error`, the copy of the test
  file ignores failure, a cancellation suppresses the upload, and an
  infrastructure failure can leave any of it missing. The plan job has a
  different summary (the inputs, or the refused inputs). A later dispatch for the
  same branch, file, `os` and Julia version cancels an earlier one that is still
  running. No cache is used; a first cold run for a single file took about 2 to
  3 minutes per system on GitHub-hosted runners.
- **The loop.** Push the branch, dispatch with `-f test_file=<file>` (add
  `-f expected_sha=<commit>` to be sure of the commit), wait with `gh run
  watch`, read `gh run view <id> --log-failed` and the summary, download the
  artifact if the stack trace is needed, reproduce on Linux (below), push the
  fix and dispatch again. A platform bug counts as fixed only with a green
  native run of the file at the fixing commit.
- **Reproducing on Linux.** Where a C `long` has 32 bits (Windows), some
  `BigFloat` and exponent code takes another path. The existing tests probe it
  from Linux in two ways. An explicit `Int32` argument exercises the methods that
  dispatch on an `Int32` exponent; `Clong` is `Int64` on Linux x64, so a
  `Clong`-typed value does not do that, and the tests use `Clong`-typed shifts as
  exact references. A constrained MPFR exponent range (`Base.MPFR.set_emax!` and
  `set_emin!`, with the previous range restored) reaches the overflow and
  underflow cases of the smaller range of Windows.
  - `test/int32_exponents.jl` passes `Int32` exponents to the helpers that take
    one, compares with the same call with an `Int` exponent, and runs the
    `BigFloat` paths that produce them.
  - `test/ldexp_bigfloat.jl` compares the package's `ldexp` and what depends on
    it with references built from `Clong`-typed shifts. Its cases that need the
    exponent range of Windows (`2^30 - 1`) run under it with `windows_range`
    (some of them also in the default range); its other cases run in the default
    range.
  - `test/exponent_sums.jl` builds its exponents relative to the platform's own
    range (`get_emax`, `get_emin`), so the same body runs where the range is
    smaller, and uses `Clong`-typed powers of two as exact operands.
  - `test/bigfloat_integer_operands.jl` compares results with integer operands
    (of several integer types, `Int32` among them) at two to five bits and at 256
    bits with a model, written in the file, that forms the integer exactly; it
    does not change the exponent range.

  A reproduction on Linux can show a cause and give the fix a test; the native
  run above is what shows the fix on the platform.
- **A new workflow file** cannot be dispatched until one run of that path has
  registered it (for example a run from a temporary `push` trigger on the
  branch); a dispatch of a file that exists only on a branch is refused with HTTP
  404 before that. Once the file is on master it is registered.

### Iterate changes

`SparseIterate`, `AtomIterate` and `SymmetricSparseIterate` change in many small
PRs during the 0.6 work. A change to one of them runs the iterate's own tests and
those of its main interactions (its focus files), not the unrelated families
that thorough adds. The focus files are in `.github/tier-rules.toml`, beside the
iterate files that quick always runs (`active_handle`, `atom_iterate`,
`iterate_invariants`, `iterate_queries`, `matrixview`, `sparse_iterate_basics`,
`symmetric_parameters`, `symmetric_products`, `wrapper_buffers`,
`allocations_basics`). The cases that five quick files leave out (Float32 repeats and
type grids: `atom_iterate_grids`, `conjugate_basics_grids`, `lmo_atoms_grids`,
`nested_scaled_grids`, `sparse_iterate_basics_grids`, listed in `test/TESTING.md`) are
in the full tier. They run in the nightly run, and as focus files for a change to
package code at the standard or thorough tier, so the rows below list them as "the
grids files"; a change to `src/symmetric_parameters.jl` does not run them. The quick,
standard and thorough tiers do not include them:

| Change | Before merge |
|---|---|
| `src/atom_iterate.jl`, narrow | quick + the grids files, `real_iterates`, `block_step`, `norms`, `sparsegroup`, `infinite_group_weights` |
| `src/sparse_iterate.jl`, narrow | quick + the grids files, `sparse_iterate`, `real_iterates`, `block_step`, `sparse_value_cost`, `conformance/failures_storage`, `symmetric_mul`, `symmetric_argument` |
| `src/symmetric_parameters.jl`, narrow | quick + `symmetric_parameters_paths`, `_trait`, `_writes`, `_merge`, `_extensions` |
| one of them, not declared narrow | standard + the same focus files |
| two of them | standard + both groups (the label does not apply) |
| shared semantics across iterate types, or an architectural or breaking milestone | thorough: add `test:thorough` |
| release candidate | full: add `test:full` |
| merge to master | thorough on Julia 1 and lts, automatically; the nightly run is full |

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
- **After merging**, the push to master runs thorough on Julia 1 and lts. If it
  fails, fix or revert the change before stacking dependent PRs on master, and
  when the failure is a bug the focus files should have caught, add a regression
  test to them (or to quick).

The label never lowers a rule other than the iterate rules: a PR that also
changes `src/utils.jl` stays thorough, one that changes `Project.toml` or the
workflow stays full, and other package code without tests still needs standard.
To add regression tests without a full run, add a new test file and its line in
`test/tiers.toml` (an addition alone keeps the PR's tier), or extend an existing
test file.

Branch protection on master requires two checks, both from GitHub Actions:
**PR required jobs** and **PR merge gate**. PR required jobs
(`scripts/ci_verdict.jl`, tested in `test/ci_verdict.jl`) runs after the plan,
the tests, the benchmark reference checks and the benchmark analysis controls,
and passes when the plan succeeded with a tier, the benchmark analysis
controls passed, the benchmark reference checks passed if the plan asked for them
(and were skipped if it did not), and each test job the plan lists as required
passed; Julia pre is advisory and printed, not counted. The plan outputs it
validates are the tier, `run_tests`, the benchmark flag, the lists of required
and advisory jobs, and, when tests run, the platforms mode; it does not validate the
others. It compares the plan's list of required jobs with a minimum that it
writes out itself from the kind of run and the plan's platforms mode: the floor
of the run (Julia 1 on Linux for a pull request, with lts from the thorough tier,
and for a manual run; Julia 1 and lts on Linux for a push to master; Julia 1 and
lts on Linux, Julia 1 on macOS and Julia 1 on Windows for a tag and the nightly
run) together with the jobs of the mode (`stable`, `linux` or `cross`). The
advisory list may hold Julia pre on Linux and nothing else, and tier none is
accepted for a pull request and a push to master only. When tests run, a plan
that omits a job of the minimum from the required list, names a platforms mode
other than those three, or lists another advisory job fails. The minimum does not
validate the planner's choice of mode or tier: a plan that lists the minimum and
more passes, including one whose mode is smaller than its labels asked for (the
`cross` jobs of a PR with `test:full`, if its mode said `linux`). The verdict
still checks each listed required job and its conclusion. A failed, cancelled,
skipped, timed out or missing required job fails it, and so does a Julia job that
is neither required nor advisory, a name that is both required and advisory, a
plan that gave no tier, and a job list that cannot be read. When no tests run,
each Julia job it lists must have been skipped. The gate job starts with the run, so a push or a label replaces the
commit's earlier green gate with a pending one within seconds; it waits for the
run's PR required jobs and passes only if they passed. It also compares the test
labels (`test:*`) of the pull request, read through the API, with those the run
was started with, and fails with "labels changed since this run; the newer run
decides" when they differ or cannot be read: a re-run of an older run, or of only its gate,
does not turn a stricter newer run's red gate green. Pushes, tags, the
nightly run and manual runs name the two checks **CI required jobs** and **CI
result**, so a manual run on a PR branch does not report under the PR's check names.
Coverage is collected only by the nightly run and by manual runs that ask for it,
so Codecov reports on master from the nightly run; its statuses are informational.

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
(each of those jobs checks that they exist), check the committed coverage map and upload
the reports. Thorough runs no `conformance/gpu_arrays` unless the change or the label
`test:gpu` adds it, so its check (`scripts/conformance_map.jl --check --tier thorough
--extra <focus files>`) compares a partial inventory: every other report file must be
there, and the rows of the other families, the case ids and the required rows must match
the committed map; a report file missing for another reason fails. Below full the
artifacts are named `conformance-reports-partial-...`. The map check and the required upload run on Julia lts on Linux when
the run has it, otherwise (a manual run on the `stable` platforms) on Julia 1 on
Linux. An upload counts only when GitHub returns an artifact id for it; otherwise
it is retried once. The artifact of that job is required: without a confirmed upload that job fails.
Only the artifact of a full run (not the `partial` one) serves to regenerate the coverage map. On the other
jobs a missing upload is a warning in the job and its summary, and the job keeps
the result of its tests.

**When a master or nightly run fails**, the workflow opens (or comments on) an
issue listing the failed jobs: "CI failing on master" for a failed push to master,
"Nightly CI failing" for the nightly run, which also says whether a Linux job
failed or only jobs on other systems (Windows, macOS). Manual runs, tags and pull
requests report nothing. Fix the regression or revert the PR that caused it the
same day; reproduce locally with `julia scripts/test.jl <file>` and the Julia
version of the failing job. A PR that merged after quick or standard is not at
fault for passing its gate; the post-merge run exists to catch what the smaller
tiers leave out, and the nightly run what the pull request and push runs do not
run on the other systems.

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
