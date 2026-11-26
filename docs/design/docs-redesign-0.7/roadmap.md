# Implementation roadmap

Ten stages after this one, implemented as fourteen PRs because the remaining function
entries (G7) are split by role page. The sequence follows the preliminary one in the
G0 brief with three changes, each motivated by the audit:

1. **The capability tables come before the function catalog**, so that the catalog
   and the function entries link to one capability reference instead of carrying
   their own copies.
2. **Function entries are split by role page.** The pilot (G6) writes four entries and
   creates the missing constraints page; the remaining entries are five independent
   PRs, one per role page.
3. **No separate pitfalls page for now.** Each entry's "Good to know" list carries the
   material; a pitfalls page is reconsidered after the usability review (G10).

## Rules that make each PR mergeable on its own

- **Link only to what exists.** A PR links only to pages and anchors that exist on
  `master` when it merges. The prototypes show the final state; until a destination
  exists, a PR links to the current page that serves the purpose (for example
  `function_library.md` before the catalog rewrite, the API entry of a function before
  its entry exists). The PR that creates a destination updates the links that should
  point to it.
- **A page joins the sidebar in the PR that creates it**, with its line in
  `docs/make.jl`. `docs/check_html.py` changes only when the number of groups or
  collapse controls changes, or a hidden page is added.
- **Forwarding is checked in every PR that moves or renames a section**: the PR lists
  each old page and fragment with its forwarding target, or the evidence that the
  fragment was not cited, and checks them in the built site.
- **Shared files.** `docs/make.jl` and the guarantee overview are touched by several
  PRs; PRs that run in parallel touch different lines, and the second to merge updates
  its branch from `master`.

## Overview

| PR | Title | Depends on | Docs-only CI | #1212 goal |
|:--|:--|:--|:--|:--|
| G0 | Audit and redesign proposal (this PR) | — | no (scripts in the design folder) | G0 |
| G1 | Home page and README | G0 approved | yes | G1 |
| G2 | Five-minute quick start | G1 | no (`make.jl`) | G2 |
| G3 | Navigation | G2 | no (`make.jl`, `check_html.py`) | G3 |
| G4 | Capability tables | G3 | no (`make.jl`) | G4 |
| G5 | Function catalog | G4 | no (forwarding page for `choosing.md`) | G4 |
| G6 | Function-entry pilot: four entries and the constraints page | G4 | no (`make.jl`) | G5 |
| G7a–e | Remaining function entries, one PR per role page | G5, G6 | G7a–d yes, unless docstrings change; G7e no (`make.jl`) | G5 |
| G8 | Examples and figures | G3 | no (`make.jl`, figure script, CI job) | G6 |
| G9 | User-facing guarantees, separated from contributor material | G3 | no (`make.jl`, `check_contract_schema.py`) | G7 |
| G10 | Independent usability review and final cleanup | all | yes, unless fixes need code | G8 |

```text
G0 ── G1 ── G2 ── G3 ──┬── G4 ──┬── G5 ──┐
                       │        └── G6 ──┴── G7a, G7b, G7c, G7d, G7e (parallel)
                       ├── G8 (the figure work may start right after G0)
                       └── G9
                                              all ── G10
```

After G3, the tracks G4–G7, G8 and G9 touch different pages and can proceed in
parallel. G5 and G6 can run in parallel once G4 is merged; the later of the two links
the catalog rows to the four pilot entries. The G7 PRs start when both have merged, so
that each can link its entries from the new catalog.

## Validation that applies to every PR

- Strict documentation build: `julia --project=docs docs/make.jl` (doctests,
  `@example` blocks and the isolated Literate examples).
- `python3 docs/check_html.py` for navigation, internal links and fragments;
  `python3 docs/check_markdown.py`; `python3 docs/check_contract_schema.py` when a
  `spec/` page changes.
- Rendered pages inspected at desktop and phone widths, with screenshots in the PR for
  any page whose layout changed.
- The forwarding list above, for any moved or renamed section.
- Every new statement about behavior checked against source and tests; no change to a
  guarantee. A discrepancy found on the way becomes an issue.
- Independent review of each PR, as for any change.

## The PRs

### G1 — Home page and README

- **Scope:** rewrite `docs/src/index.md` following the [prototype](prototypes/index.md),
  linking to the current pages where the prototype's destinations do not exist yet
  (`getting_started.md` for the quick start, `function_library.md` for the catalog,
  `function_objects.md` for capabilities, `spec/index.md` for guarantees). Rewrite the
  README's opening: one paragraph, install, a short example, links to the rendered
  manual instead of `docs/src` paths, and the ProximalSolvers relationship. Keep the
  README's testing section.
- **Acceptance:** the five one-minute questions of #1212 are answered on the first
  screen of Home at 1440 × 900 and within two screens on a phone; the install text and
  version table match the registry on the day of merge; every README link opens a
  rendered page; the Home example runs and the text describes its output; the anchors
  `Where-to-start` and `documentation-tracks` survive.
- **Areas:** `index.md`, `README.md`.

### G2 — Five-minute quick start

- **Scope:** add `quickstart.md` following the [prototype](prototypes/quickstart.md),
  at the top of the current Learn group; point Home's quick-start links to it; replace
  the install section of `getting_started.md` with a link to Home's version table.
- **Acceptance:** runs top to bottom in a fresh environment; uses only `Vector{Float64}`
  and `Matrix{Float64}`; introduces no term it does not explain (no working set, KKT,
  compact block, active zero); the reader evaluates a function, takes a proximal step,
  computes a gradient and uses an in-place call.
- **Areas:** `quickstart.md` (new), `index.md` links, `getting_started.md`, one line in
  `make.jl`.

### G3 — Navigation

- **Scope:** the six groups of [the proposed architecture](architecture.md), with the
  pages that exist at that point; sidebar labels and page headings aligned; the
  "Assumes · You will learn" banners kept on tutorials only; `check_html.py` constants
  updated.
- **Acceptance:** every existing URL still resolves (no page path changes); install,
  the quick start, the tutorials and the extension guide are reachable from Home within
  the choices listed in the [reader tasks](architecture.md#paths-for-the-readers-tasks),
  on desktop and on a phone (counting the menu tap); `check_html.py` passes. The tasks
  that need the catalog and the capability tables are accepted in G10, once those pages
  exist.
- **Areas:** `docs/make.jl`, `docs/check_html.py`, the headings of retitled pages,
  `AGENTS.md` and `CONTRIBUTING.md` where they name sidebar sections.
- **Note:** keep this PR free of content edits.

### G4 — Capability tables

- **Scope:** `capabilities.md` following the [prototype](prototypes/capabilities.md),
  for all 30 functions, with function names linking to their API entries and without the
  prototype's note on its verification record; the
  built-in trait table of `function_objects.md` replaced by a link; iterate pages link
  to the tables; the API coverage matrix gains the test behind each ✓ and ◐.
- **Acceptance:** every ✓ and ◐ is backed by a named test, or by a probe recorded in the
  PR when no test exists (never by a method, trait or fallback alone); every — has been
  seen to throw or is documented as unsupported; each condition is one sentence with a
  link to its owner; the rendered note labels match the cells; no table is wider than
  about twice the content column of a 390 px phone.
- **Areas:** `capabilities.md` (new), its line in `make.jl`, `function_objects.md`, iterate pages, `api/coverage.md`.

### G5 — Function catalog

- **Scope:** rewrite `function_library.md` as the [catalog](prototypes/catalog.md); move
  the "choose an operation" table of `choosing.md` into `operations.md`; keep
  `choosing.md` as a hidden forwarding page that keeps `@id choosing`.
- **Acceptance:** all 30 functions (29 types and the `IndBallLinf` constructor) appear,
  with 31 definition rows for the two forms of `NormL0`; the "If your model needs…" rows cover every row of the old
  selection guide; table cells stay short; every row links to the function's API entry,
  or to its entry where one exists.
- **Areas:** `function_library.md`, `choosing.md`, `operations.md`, `make.jl` (hidden
  page), `check_html.py` (hidden-page list).
- **Parallel:** with G6.

### G6 — Function-entry pilot

- **Scope:** the four entries of the prototypes, written in place: `NormL1` at the top
  of `penalties.md`, `GroupNormL2` in `group_penalty.md`, `LeastSquaresLoss` at the top of
  `smooth.md`, retitled "Losses", and `IndBallL1` in a new `constraints.md`. The capability
  tables (and the catalog, if G5 has merged) link to the new entries. The four docstrings
  may be aligned with the entry template.
- **Acceptance:** each entry follows the [template](editorial-principles.md#function-entries);
  nothing is removed unless the guarantee page or the same page keeps it; the forwarding
  list is complete. The maintainer approves the pilot before G7 starts.
- **Areas:** `penalties.md`, `group_penalty.md`, `smooth.md`, `constraints.md` (new), up to
  four docstrings, `make.jl`.

### G7a–e — Remaining function entries

- **Scope:** one PR per role page, each also linking its entries from the catalog and
  the capability tables:
  - **G7a**, losses (`smooth.md`): `LogisticLoss`, `PoissonLoss`, `SqrtLeastSquaresLoss`, `LogDetLoss`;
  - **G7b**, sparsity penalties: `ElasticNet`, `NormL1Nonneg`, `MCP`, `SCAD`, `NormL0`,
    `NormL0(λ; diagonal = false)`;
  - **G7c**, norm and group penalties: `NormL2`, `NormLinf`, `SparseGroupNormL2`,
    `GroupNormLinf`, `GroupMCP`, `GroupSCAD`;
  - **G7d**, constraints: `IndBox`, `IndBallLinf`, `IndNonnegative`, `IndBallL2`, `IndBallL0`;
  - **G7e**, quadratic, linear and scaled functions: creates `quadratic.md` from the
    quadratic sections of `smooth.md` (forwarding headings left in `smooth.md`), with
    `SqrNormL2`, `QuadraticFunction`, `KronQuadratic`, `LinearFunction`, `ZeroFunction`,
    and the `Scaled` entry in `calculus.md`.
  Long derivations and edge-case lists move to the guarantee pages or into collapsed
  `!!! details` blocks.
- **Acceptance:** as G6, for every entry on the page.
- **Parallel:** all five.

### G8 — Examples and figures

- **Scope:** the [prototype figures](prototypes/visualizations.md) moved into
  `docs/generate_learn_figures.py`, their coordinates computed by ProximalBase in the
  build, and the figure check added to the documentation job; example titles changed to
  tasks; one new Literate example that changes one argument at a time (lasso → MCP →
  `SparseIterate`), added to the example registry in `make.jl` and to the isolated
  example run.
- **Acceptance:** every featured example runs on `master`; each figure is checked
  against ProximalBase by the build; the example inventory grows from nine to ten with
  its title and order registered.
- **Areas:** `examples/`, `docs/make.jl`, `docs/run_examples.jl` if needed,
  `docs/generate_learn_figures.py`, `docs/src/assets/`, the CI documentation job.

### G9 — User-facing guarantees

- **Scope:** `spec/index.md` rewritten as the [guarantee overview](prototypes/guarantees.md),
  keeping its anchor and a forwarding heading for "Family-page schema"; its authoring
  rules moved, unchanged, to `developer/contract_pages.md`; the family map kept as one
  checked table with a role column instead of the migration column, and
  `check_contract_schema.py` changed to parse it with the same coverage and owner
  checks (compared before and after); process wording on family pages reworded without
  changing any rule; the ownership table of `numerical_usage.md` rewritten as short
  prose per function; closed-issue history moved to `migration.md` or `docs/design/`.
- **Acceptance:** a review of the diff confirms that every guaranteed, precondition,
  limitation, non-guarantee and characterized statement keeps its meaning and its
  status; the checker reports the same families and owners before and after; no user
  page mentions the migration process. The treatment of "Links only." sections follows
  the maintainer's [decision](README.md#decisions-for-the-maintainer).
- **Areas:** `spec/`, `numerical_usage.md`, `numerics.md`, `developer/` (new page and its
  line in `make.jl`), `docs/check_contract_schema.py`, `docs/src/developer/writing_documentation.md` (link to
  the moved schema).

### G10 — Independent usability review and final cleanup

- **Scope:** a reviewer who has not followed the work does the
  [reader tasks](architecture.md#paths-for-the-readers-tasks) from Home, without
  instructions on where to click, and reports paths, time and friction with screenshots;
  the task paths of the architecture are the acceptance targets;
  small fixes; the catalog reconciled with the 0.7 release candidate.
- **Acceptance:** the report says whether people or agents did the review and keeps
  their observations apart; each task is completed, or the remaining friction is
  recorded as an issue; if no newcomer is available, the maintainer records a deferral;
  all ordinary documentation checks pass; the `dev` and release manuals link correctly.

## What is deliberately not planned

- Per-function pages, a generated capability table, a new documentation framework or
  Documenter plugin, a migration ledger, and new preservation infrastructure.
- Runtime changes. The display of function objects, the `MethodError` raised by
  `gradient` on nonsmooth functions and the other discrepancies are recorded as
  [separate findings](audit.md#defects-and-gaps-recorded-separately).
- Backporting any of this to `release/0.6`.
