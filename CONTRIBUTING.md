# Contributing

Use `UpperCamelCase` for modules and types, `snake_case` for descriptive
functions, fields and keyword arguments, and a trailing `!` for mutating
functions. Retain established Julia interface names, familiar numerical
abbreviations, and mathematical identifiers such as `X`, `Σ` and `λ`.
Deprecated names are compatibility exceptions, not examples for new code.

Develop and test against local `master` checkouts of workspace packages.
Active documentation and benchmark environments specify those checkouts with
relative `[sources]` paths. Run `julia scripts/instantiate.jl` from the package
root to develop these dependencies explicitly on Julia 1.10 and newer. Use
`julia scripts/instantiate.jl docs` for the documentation environment.

Use the [benchmark suite](benchmark/README.md) to measure runtime and allocations
before and after implementation changes. Keep generated measurements with the
PR that uses them; add reusable cases for workloads worth protecting.

The documentation presents the package as a whole, with 0.5.4 as the planned
initial release baseline. Version metadata and release tags are updated
separately during the release process.

Run the complete tests with:

```sh
julia scripts/test_master.jl
```

The runner creates an isolated environment with the package, dependencies and
test extras, and runs the suite with bounds checking. It preserves extra-only
master dependencies on Julia 1.10 and leaves the package's `Project.toml`
unchanged. This is the development test command used by CI.
