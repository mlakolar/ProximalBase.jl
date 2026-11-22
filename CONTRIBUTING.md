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

Build the strict docs, doctests and all nine isolated Literate examples:

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
Comments in `src/` and `test/` state numerical behavior in words, and refer to a
section of `docs/src/numerics.md` where that page states it, not to rule labels such
as `R1` or `D2`; `@testset` names state what the test checks in the same way.
`julia docs/check_source_labels.jl` checks for such labels in the comments of both
directories and in `@testset` names.

New function families put their numerical rules in
`docs/src/spec/<family>.md`, using the fourteen-section schema and links to
existing shared-rule owners. Follow the
[new-family writing rules](docs/src/developer/writing_documentation.md#New-function-families)
and update the contract family map and navigation.

CI gates pull requests and master pushes with Julia 1.13 on Linux. A
change-classification job runs the repository hygiene checks and classifies
the changed paths. Three independent test shards cover the complete retained
inventory with bounds checks, and the frozen benchmark differential and the
strict docs build are eligible to run in parallel on separate Linux runners. A non-empty change is
documentation-only when every changed path, including both sides of a rename
and each deletion, is an added, modified or deleted regular file without the
execute bit (an execute-bit change, a symlink or a submodule entry on either
side is not), and the path is a top-level `*.md` file, a `.md`, `.svg`, `.png`
or `.mmd` file under `docs/`, or a `.tsv`, `.json`, `.csv` or `.txt` file under
`docs/audit/docs-tracks-1019/`. Suffixes are matched case-sensitively. When
the classification job succeeds with `docs_only=true`, the test shards and the
benchmark are skipped; the strict docs build is scheduled independently and is
still required. When it succeeds with
`docs_only=false`, they run: this covers any other change, an empty change set,
a push without a usable previous commit and a manual run. The strict docs build
runs the executable Documenter blocks of the Markdown pages it builds, docstring
doctests and isolated Literate examples. The classifier is `scripts/ci_changes.py`, with tests in
`scripts/test_ci_changes.py`. The required checks remain `PR required jobs` and
`PR merge gate`; they fail unless the classification and docs lanes succeed and
the test and benchmark lanes were skipped (`docs_only=true`) or succeeded
(`docs_only=false`).
Job-specific Julia caches are reused on subsequent runs. Reproduce one
shard with `julia scripts/test.jl --shard=1/3` (or `2/3`, `3/3`).
GPU, cross-platform and release automation remain deferred. No release or
ProximalSolvers pin/source change is implied.

## Julia performance and correctness

Use Julia 1.13.4 for performance benchmarks and optimization decisions.
Use Julia 1.10.13 to verify correctness. Performance parity on Julia 1.10.13
is outside the performance gate.

Investigate and escalate repeatable slowdowns above 5% on ordinary Julia 1.13
paths. Compare equivalent workloads and consider the absolute cost and how
often the workload runs. Prioritize bottlenecks on Julia 1.13.4. Add
version-specific complexity only when it delivers a substantial improvement
on Julia 1.13.

Use the retained benchmark harness for changes affecting hot paths and compare
against the frozen post-#848 source on the same machine with the same harness,
including an A/A control. Record slowdowns under investigation with their
allocations, dispersion, controls and attribution. Historical def81ec9 results
are diagnostic rather than the primary comparison.
