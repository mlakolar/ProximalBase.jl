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
ridge precision tests cover `Float32` and `Float64`; a custom function verifies the
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

CI gates pull requests with current Julia 1.x on Linux. Three independent
test shards cover the complete retained inventory with bounds checks. The
frozen benchmark differential, repository hygiene checks and strict docs
build run concurrently on separate Linux runners. The required checks remain
`PR required jobs` and `PR merge gate` and require every validation lane to
pass. Job-specific Julia caches are reused on subsequent runs. Reproduce one
shard with `julia scripts/test.jl --shard=1/3` (or `2/3`, `3/3`).
GPU, cross-platform and release automation remain deferred. No release or
ProximalSolvers pin/source change is implied.

Use the retained benchmark harness for changes affecting hot paths and compare
against the frozen post-#848 source on the same machine with the same harness,
including an A/A control. Record any slowdown above five percent with its
allocations, dispersion, controls and attribution. Historical def81ec9 results
are diagnostic rather than the primary comparison.
