# Contributing

Use UpperCamelCase for types/modules, snake_case for descriptive functions,
fields and keyword arguments, and `!` for mutating functions. Keep established
Julia interface names and mathematical identifiers.

Run the complete supported inventory with bounds checking:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test(; julia_args=["--check-bounds=yes"])'
```

`julia scripts/test.jl core_functions` selects filename prefixes.
`julia scripts/test.jl --tier=quick --list` lists files. The four cumulative
tier spellings remain accepted; the minimal inventory currently lives in quick,
so full/standard/thorough run that same supported inventory. Every Julia file
under test must be declared in test/tiers.toml as a test or helper. Undeclared,
missing and duplicate files fail before execution; unmatched selectors fail.
New tests go in that manifest and test/TESTING.md.

The unchanged storage suites test active zeros, copies, aliases, views, indexing,
products and allocation regressions. Independent numerical references and
retained ridge precision tests cover Float32/64; a custom function verifies the
extension hooks. This suite is not the historical full package suite, which is
preserved on legacy/full.

Build the strict docs, doctests and all eight isolated Literate examples:

```sh
julia scripts/instantiate.jl docs
julia --project=docs docs/make.jl
python3 docs/check_html.py
python3 -m http.server 8000 --directory docs/build
```

Open `http://localhost:8000/` to inspect the generated site. The build restores
the historical expandable navigation and checks every export/qualified public
hook against grouped API entries and the coverage matrix. Edit `examples/*.jl`
and `docs/design/*.md`; generated example/contract pages are ignored. See
[the developer workflow](docs/src/developer/change_process.md) for validation
and ownership contracts. CI retains HTML and full validation logs as artifacts.

CI gates pull requests with current Julia 1.x on Linux. The required checks
are named `PR required jobs` and `PR merge gate`; both require the test and docs
job to succeed. GPU, cross-platform and release automation are deferred for
this reduction. No release or ProximalSolvers pin/source change is implied.

Use the retained benchmark harness for changes affecting hot paths and compare
against the frozen post-#848 source on the same machine with the same harness,
including an A/A control. Record any slowdown above five percent with its
allocations, dispersion, controls and attribution. Historical def81ec9 results
are diagnostic rather than the primary comparison.
