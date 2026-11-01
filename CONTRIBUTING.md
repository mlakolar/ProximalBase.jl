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
the CI plan. Files run alone in a sandbox module, and compilation takes most of
their time, so a cheap file reaches few type combinations. Helper files that are
only included go under `[not_tests]`. [`test/TESTING.md`](test/TESTING.md) records
why each file sits in its tier.

### Which tests CI runs

The plan job of the CI workflow picks the tier of each run with
`scripts/ci_plan.jl` and the rules at the end of `test/tiers.toml`; the run's
summary page shows its decision for every changed file.

- **Pull requests** run Julia lts and Julia 1 on Ubuntu at the minimum tier for
  the changed files, and the changed test files beside it:
  - documentation only: no package tests;
  - test files, examples or README only: quick;
  - a self-contained function family together with a test change: quick and that
    family's own test files;
  - other package code (`src/`): standard; code shared by every function
    (iterates, `interface.jl`, `utils.jl`, `calculus.jl`, `metric.jl`, spectral
    helpers) or the conformance suite and its coverage map: thorough;
  - package code without any test change: at least standard;
  - `Project.toml`, `ext/`, `src/lapack.jl`, the test runner, the tier manifest
    and CI: full, which also runs Julia 1 on macOS and Windows and Julia pre;
  - a path no rule knows: full.
- **Labels** `test:standard`, `test:thorough` and `test:full` raise a PR's tier;
  they never lower it. Adding or removing any label starts a new run. Add
  `test:thorough` when a change touches element-type promotion, precision,
  overflow-safe scaling or non-finite handling, which the paths cannot show, and
  `test:full` for a release or a platform-sensitive change.
- **Pushes to master and tags** run the full tier on all five jobs (none for a
  documentation-only push), with coverage and the conformance reports. A
  **nightly** full run catches changes in dependencies and Julia releases; it is
  skipped when a full run already passed on the same commit that day.
- **Manual runs** test any branch at any tier, full by default:
  `gh workflow run CI.yml --ref <branch> -f tier=thorough`. On a PR branch the
  result appears on the head commit beside the PR's own checks.

Branch protection should require only the check **PR merge gate**. It passes
when the plan succeeded, the benchmark analysis controls passed and every test
job of the chosen tier passed; Julia pre is advisory. A failed, cancelled or
missing test job fails it, and so does a plan that found an unclassified test
file. Coverage is collected only on full runs, so Codecov reports on master and
on full PR runs.

Each test job's summary page lists the selected files, the workers, the time
before the test runner started (Julia, test environment, precompilation), the
runner's wall time, the worker time of every file with its compilation share,
and the slowest files. Thorough and full runs also write the conformance reports,
check the committed coverage map (Julia lts) and upload the reports.

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
