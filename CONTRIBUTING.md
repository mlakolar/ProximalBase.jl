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

To run the full suite through the test script used by continuous integration (CI):

```sh
julia scripts/test.jl
```

This calls `Pkg.test()` with array bounds checking enabled. No separate setup
command is needed. Set `JULIA_TEST_COVERAGE=true` to collect coverage, as CI
does, or `JULIA_TEST_DEPWARN=error` to treat deprecation warnings as errors.

Test files run in parallel worker processes. Extra arguments list tests, select a subset,
or set the workers: `julia scripts/test.jl --list`, `julia scripts/test.jl lapack caches`,
or `julia scripts/test.jl --jobs=2`. The same arguments work through `Pkg.test(; test_args=...)`.

`julia scripts/test.jl --core` skips the extended tier (`precision_mixed`,
`large_inputs`, `spectral_overflow`); plain `julia scripts/test.jl` runs everything.
`--list` shows the full catalog, including with `--core`. The core switch combines
with test names, for example `julia scripts/test.jl --core spectral`.

CI runs the following tests:

- Pull requests run Julia lts and Julia 1 on Ubuntu with the core tier (`--core`).
  Coverage from Julia 1 counts only the core tier, so Codecov may report lower
  coverage on a PR than on master. Codecov merges uploads for the same commit,
  so a full run on the same head adds to the report.
- Pushes to master, tags and manual runs use all jobs and the full suite:
  Julia 1 on macOS and Windows and Julia pre on Ubuntu in addition to the
  Ubuntu jobs. A newer pending master run can replace an older pending one.

For most pull requests the core tier is enough, and the master run after merge
checks the rest. Run the full matrix before merging when the PR changes:

- numerics the extended tier checks: element-type promotion or precision
  handling, overflow-safe scaling, non-finite handling, spectral functions, or
  a helper shared across the catalog (`utils.jl`, `interface.jl`,
  `calculus.jl`, traits);
- platform- or version-sensitive code: `src/lapack.jl`, BLAS calls,
  threading, `VERSION` checks, `ext/`, or Base internals;
- infrastructure: `Project.toml` dependencies or compat, `.github/workflows/`,
  `test/runtests.jl`, `scripts/test.jl` or `test/testutils.jl`;
- the version, for a release.

To get a full run, start one manual run on the final commit after review:
`gh workflow run CI.yml --ref <branch>`. It reports on the head commit, so the
PR shows it. Start another only if later commits also fall under the list
above. The PR description should say which tier the PR got and why. Labelling
a PR `full-tests` also gives full runs, but on every later push until the label
is removed. The label is read when a run starts, so re-running an old run does
not pick it up.

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
