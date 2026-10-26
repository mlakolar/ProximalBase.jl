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
ProximalBase currently has no dependencies on other related packages, such as
ProximalSolvers, so local clones of those packages are not required.

The setup script, `scripts/instantiate.jl`, can use local clones of related
packages listed in `scripts/master_dependencies.jl`. It selects only packages
that are direct or indirect dependencies of the selected environment,
including dependencies used for testing. For those dependencies, it requires
the local clones to be on their `master` branches. Missing clones are created
in the parent directory from those branches. The package being developed
uses its own local clone and may be on a feature branch. The script does not
install every package in the list.

The documentation and benchmark environments specify the local ProximalBase
clone with a relative path in the `[sources]` section of `Project.toml`.
To build the documentation, including on Julia 1.10 where the setup script
calls `Pkg.develop` to use the local clone explicitly, run:

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

To run the complete tests using the same script as continuous integration (CI):

```sh
julia scripts/test_master.jl
```

The script installs dependencies, creates a separate Julia environment with
the local package and test dependencies, and runs the suite with array bounds
checking.
It leaves the package's `Project.toml` unchanged and needs no preceding
`scripts/instantiate.jl` invocation. On Julia 1.10 it also supports related
packages needed only for testing; ProximalBase currently has no such dependencies.
