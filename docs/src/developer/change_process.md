# Contract conformance and contributor workflow

## Reproducible documentation build

From the repository root on Linux with current Julia 1.x:

```sh
julia scripts/instantiate.jl docs
julia --project=docs docs/make.jl
python3 -m http.server 8000 --directory docs/build
```

Open `http://localhost:8000/`. The default local build uses `.html` links;
CI uses directory URLs. Set `CI=true` locally to inspect that form. To rebuild
without running each script in a separate process again, use
`PB_DOCS_SKIP_ISOLATED=true` only during editing; the final/CI build runs them.
The build always executes Literate examples in Documenter sandboxes and doctests.
External link availability checks are optional via `DOCUMENTER_LINKCHECK=true`;
internal cross-references and docstrings remain strict regardless.

The docs project contains Documenter, Literate and the local package. The
instantiate script develops this checkout explicitly and restores Project.toml
after Pkg setup. Generated `docs/src/examples`, `developer/contracts`, manifests
and `docs/build` are ignored. Edit examples/*.jl or docs/design/*.md, not generated
copies. The source listing in the extension tutorial is read from executable
examples, so there is one authored definition.

## Required checks

`docs/make.jl` preserves `checkdocs=:exports`, `doctest=true`, `warnonly=false`,
KaTeX and HTML collapselevel=1. It checks explicit public reference coverage,
regenerates Literate pages/contracts, runs every example in a fresh process,
then builds with Documenter. Missing pages, unresolved @ref links, doctest
failures and invalid docstrings fail the build. `docs/check_html.py` checks
rendered internal links/anchors, page coverage and sidebar/math/code markup.
Actual browser inspection complements that structural check.

```sh
julia --project=docs docs/run_examples.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test(; julia_args=["--check-bounds=yes"])'
python3 docs/check_html.py
```

CI is `.github/workflows/CI.yml`. A change-classification job checks
repository hygiene and classifies the changed paths with
`scripts/ci_changes.py`. A non-empty change is documentation-only when every
changed path is an added, modified or deleted regular, non-executable file that
is a top-level `*.md` file, a `.md`, `.svg`, `.png` or `.mmd` file under
`docs/`, or a `.tsv`, `.json`, `.csv` or `.txt` file under
`docs/audit/docs-tracks-1019/` (suffixes are case-sensitive). When the job
succeeds with `docs_only=true`, the three package-test shards and the frozen
benchmark differential are skipped. When it succeeds with `docs_only=false`
they are eligible to run in parallel with strict documentation validation, on
Linux with Julia 1.13. The strict documentation job is scheduled independently
of the classification; it runs the executable Documenter blocks of the Markdown
pages it builds, docstring doctests and isolated Literate examples. The
`PR required jobs` verdict and `PR merge gate` require the classification and
documentation lanes, and either both skipped lanes (documentation-only) or both
successful lanes (any other change). There is no LTS/cross-platform/GPU matrix
in the required checks. HTML and job-specific logs are retained as artifacts.
On master pushes, a deployment job becomes eligible after `gate` and `docs`
succeed. Its publication steps run only if the fetched `refs/heads/master` tip
equals its commit at the time of the check: the job then attempts to retrieve
the validated HTML artifact and publish development documentation with
`docs/deploy.jl`. After `gate` and `docs` succeed, a separate job for
same-repository non-draft PRs becomes eligible to publish their validated HTML
artifact to an isolated `gh-pages` preview through a separate job that never
executes PR code. Local `docs/make.jl` only validates/builds. Do not bypass
required gates because local docs passed.

## Tests and independent numerical checks

`test/tiers.toml` declares every retained test/helper. The current full inventory
runs in isolated Julia processes; historical conformance suites are archived.
`julia scripts/test.jl core_functions` selects a filename prefix; use `--list`
to inspect selection. Add tests to the declared inventory and test/TESTING.md.

Choose independent reference calculations: direct residual/gradient formulas,
linear solves for quadratic prox, BigFloat for range-sensitive ridge cases,
explicit lower-triangle enumerations for symmetric iterates, and expected active IDs for
storage. Comparing two PB oracles alone can reproduce the same bug. Test
FP32/FP64, dimensions, non-finite/zero parameters, aliasing, partial failures,
active zeros and arbitrary ordered blocks as appropriate to the change.

## Local job execution and process ownership

Run test and benchmark jobs in the foreground unless detached execution is
explicitly authorized. Track the PIDs or process group of jobs you start,
including any authorized detached jobs. Before cleanup, verify their identity
and terminate **only those owned processes**. Never use broad name-matching
`pkill` (such as `pkill julia`) to clean up sessions on a shared machine;
leave unknown or other agents' processes untouched. Include or link to these
rules in coding-agent and reviewer handoffs.

## Adding a function or storage method

1. Specify mathematics, supported signatures, domain/shapes, traits, parameter
   ownership, return values and errors. Distinguish scalar/block/global index spaces.
2. Implement public validation before predictable writes. Define parameter metadata
   and cache invalidation where needed. Keep private helpers private.
3. Add independent regression evidence and a small executable example. For new
   storage writers check reverse maps/activity after both success and failure.
4. Update docstrings, operations catalog, concrete guide, current contracts and
   the explicit API coverage matrix. Every public binding needs a grouped @docs entry.
5. Measure relevant hot paths with the retained benchmark harness before making
   performance claims. Frozen-baseline controls and A/A comparisons matter for
   numerical changes; historical timings alone are not a current guarantee.
6. Build strictly, inspect generated HTML/navigation and obtain an independent
   substantive review. Report implementation defects/ambiguous contracts separately
   during documentation work rather than changing code to fit an example.

For a new function family, create its numerical entry page under
`docs/src/spec/<family>.md` and follow the
[new-family documentation rules](writing_documentation.md#New-function-families).
Use the fourteen-section schema, link shared owners by anchor, and update the
family map and navigation alongside the API entries.

## Historical acceptance records

The archived [iterate-family acceptance](https://github.com/mlakolar/ProximalBase.jl/blob/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design/iterate-family-0.6-acceptance.md)
and [symmetric-iterate acceptance](https://github.com/mlakolar/ProximalBase.jl/blob/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design/symmetric-sparse-iterate-0.6-acceptance.md)
include old targets, amendments and past evidence. They are not current test
commands or new public requirements. The current contracts and actual retained
source/tests must agree before a behavior is described as supported.

For editorial changes, follow [Writing useful documentation](writing_documentation.md)
and preserve every moved contract condition at its rendered destination.
