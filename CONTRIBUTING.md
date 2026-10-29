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

To run the tests with the same options as continuous integration (CI):

```sh
julia scripts/test.jl
```

This calls `Pkg.test()` with array bounds checking enabled. No separate setup
command is needed. Set `JULIA_TEST_COVERAGE=true` to collect coverage, as CI
does, or `JULIA_TEST_DEPWARN=error` to treat deprecation warnings as errors.

Test files run in parallel worker processes. Extra arguments list tests, select a subset,
or set the workers: `julia scripts/test.jl --list`, `julia scripts/test.jl lapack caches`,
or `julia scripts/test.jl --jobs=2`. The same arguments work through `Pkg.test(; test_args=...)`.

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
