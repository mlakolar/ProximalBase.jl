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

Julia installs the declared dependencies and test extras automatically.
ProximalBase currently has no dependencies on the other workspace packages,
so their sibling checkouts and `master` branches are not required.

The shared `scripts/instantiate.jl` helper selects known workspace packages
only when they occur in the chosen environment's dependency and test-extra
graph. For those dependencies, use local `master` checkouts; missing sibling
checkouts are cloned from `master`. The package being developed uses its own
checkout and may be on a feature branch. The catalog in
`scripts/master_dependencies.jl` is not a list of packages to install in every
environment.

The documentation and benchmark environments specify the local ProximalBase
checkout with a relative `[sources]` path. To build the documentation, including
on Julia 1.10 where the helper explicitly develops that checkout, run:

```sh
julia scripts/instantiate.jl docs
julia --project=docs docs/make.jl
```

Use the [benchmark suite](benchmark/README.md) to measure runtime and allocations
before and after implementation changes. Keep generated measurements with the
PR that uses them; add reusable cases for workloads worth protecting.

The documentation presents the package as a whole, with 0.5.4 as the planned
initial release baseline. Version metadata and release tags are updated
separately during the release process.

To run the complete tests using the same runner as CI:

```sh
julia scripts/test_master.jl
```

The runner performs its own dependency setup, creates an isolated environment
with the local package and test extras, and runs the suite with bounds checking.
It leaves the package's `Project.toml` unchanged and needs no preceding
`scripts/instantiate.jl` invocation. Its support for extra-only workspace
dependencies on Julia 1.10 does not add any such dependencies to ProximalBase.
