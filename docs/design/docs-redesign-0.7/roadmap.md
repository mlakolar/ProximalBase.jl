# Implementation roadmap

Ten PRs after this one. They follow the preliminary sequence in the G0 brief with
three changes, each motivated by the audit:

1. **The capability tables come before the function catalog.** The catalog's
   "main operations" column and every function entry's support line are copies
   of the tables. Writing the tables first gives them a single source.
2. **Function entries are split by role page, not into "pilot" and "everything
   else".** The pilot (G6) creates the two missing role pages (losses and
   constraints) with one entry each. The remaining entries (G7) are then four
   independent PRs, one per role page, that can run in parallel.
3. **The "Common pitfalls" page joins the examples PR (G8)**, which already
   collects the material it needs.

Navigation changes edit `docs/make.jl` and `docs/check_html.py`, so the PR that
makes them runs the full test lanes (see [constraints](audit.md#constraints-a-redesign-must-respect)).
They are concentrated in G3; later PRs that add one page add one `make.jl` line.

## Overview

| PR | Title | Depends on | Docs-only CI? | #1212 goal |
|:--|:--|:--|:--|:--|
| G0 | Audit and redesign proposal (this PR) | — | yes, apart from the figure scripts | G0 |
| G1 | Home page and README | G0 approved | yes | G1 |
| G2 | Five-minute quick start | G0 approved | no (one `make.jl` line) | G2 |
| G3 | Navigation | G1, G2 | no | G3 |
| G4 | Capability tables | G3 | no (new page) | G4 |
| G5 | Function catalog | G4 | no (forwarding page for `choosing.md`) | G4 |
| G6 | Function-entry pilot: four entries, two new role pages | G4 | no | G5 |
| G7a–d | Remaining function entries, one PR per role page | G6 | yes | G5 |
| G8 | Examples, figures and common pitfalls | G3 (figures can start after G0) | no | G6 |
| G9 | User-facing guarantees, separated from contributor material | G3 | no | G7 |
| G10 | Independent usability review and final cleanup | all | yes, unless fixes need code | G8 |

```text
G0 ──┬── G1 ──┐
     └── G2 ──┴── G3 ──┬── G4 ──┬── G5
                       │        └── G6 ── G7a, G7b, G7c, G7d (parallel)
                       ├── G8 (figures may start right after G0)
                       └── G9
                                         all ── G10
```

G1 and G2 can be written in parallel. After G3, the tracks G4–G7, G8 and G9 touch
different pages and can proceed in parallel; G5 and G6 can also run in parallel
once G4 is merged.

## Validation that applies to every PR

- Strict documentation build: `julia --project=docs docs/make.jl` (doctests,
  `@example` blocks and the nine isolated Literate examples).
- `python3 docs/check_html.py` for navigation, links and fragments;
  `python3 docs/check_markdown.py`; `python3 docs/check_contract_schema.py` when a
  `spec/` page changes.
- Rendered pages inspected at desktop and phone widths, with screenshots in the PR
  for any page whose layout changed.
- A list of every moved or retired heading, with its forwarding heading or the
  evidence that it was not cited (the existing rule in `writing_documentation.md`).
- Every new statement about behavior checked against source and tests; no change to
  a guarantee. A discrepancy found on the way becomes an issue.
- Independent review of each PR, as for any change.

## The PRs

### G1 — Home page and README

- **Scope:** rewrite `docs/src/index.md` from the [prototype](prototypes/index.md);
  rewrite the README's opening (one paragraph, install, a five-line example, links to
  the rendered manual instead of `docs/src` paths, the ProximalSolvers relationship);
  keep the README's testing section.
- **Acceptance:** the five one-minute questions of #1212 are answered on the first
  screen of Home at 1440 × 900 and within two screens on a phone; the install text
  matches the registry state on the day of merge; all README links open rendered pages;
  the Home example runs and its printed output is what the text says.
- **Areas:** `index.md`, `README.md`. No navigation change, so the `@id`s of the
  current Home sections (`Where-to-start`, `documentation-tracks`) stay as forwarding
  headings until G3.
- **Parallel:** with G2.

### G2 — Five-minute quick start

- **Scope:** add `quickstart.md` from the [prototype](prototypes/quickstart.md) at the
  top of the current Learn group; trim the install section of `getting_started.md` to
  a link.
- **Acceptance:** runs top to bottom in a fresh environment; uses only `Vector{Float64}`
  and `Matrix{Float64}`; introduces no term it does not explain (no working set,
  KKT, compact block, active zero); the reader evaluates a function, takes a proximal
  step, computes a gradient and uses an in-place call.
- **Areas:** `quickstart.md` (new), `getting_started.md`, one line in `make.jl`.
- **Parallel:** with G1.

### G3 — Navigation

- **Scope:** the sidebar of [the proposed architecture](architecture.md); sidebar labels
  aligned with page headings; explicit `@id`s on the pages the new Home links to; the
  "Assumes · You will learn" banners kept on tutorials only; `check_html.py` constants
  updated.
- **Acceptance:** every existing URL still resolves (page paths unchanged);
  `check_html.py` passes with the new constants; every page is reachable within two
  choices from Home; the rendered sidebar is checked on desktop and phone.
- **Areas:** `docs/make.jl`, `docs/check_html.py`, H1 lines of retitled pages,
  `AGENTS.md` and `CONTRIBUTING.md` where they name sidebar sections.
- **Parallel:** none; this is the hinge of the project. Keep it free of content edits.

### G4 — Capability tables

- **Scope:** `capabilities.md` from the [prototype](prototypes/capabilities.md)
  covering all 30 function types; the built-in trait table of `function_objects.md`
  replaced by a link; each iterate page gains a short "functions you can use with
  this storage" list taken from the tables; the API coverage matrix (developer guide)
  gains the probe or test that supports each ✓.
- **Acceptance:** every ✓ and ◐ is backed by a test or a probe recorded in the PR (not
  by a method, trait or fallback alone); every — has been confirmed to error or to be
  documented as unsupported; conditions are footnoted in at most one sentence; no
  table is wider than about twice the content column of a 390 px phone.
- **Areas:** `capabilities.md` (new), `function_objects.md`, iterate pages, `api/coverage.md`.

### G5 — Function catalog

- **Scope:** rewrite `function_library.md` as the [catalog](prototypes/catalog.md);
  move the "choose an operation" table of `choosing.md` into `operations.md`; leave
  `choosing.md` as a hidden forwarding page, its `@id` moved to the catalog.
- **Acceptance:** all 30 function types appear exactly once as a primary row; the
  "if your model needs" rows cover every row of the old selection guide; no table cell
  longer than about 25 words; every row links to its entry.
- **Areas:** `function_library.md`, `choosing.md`, `operations.md`, `make.jl`
  (hidden page), `check_html.py` (hidden-page list).
- **Parallel:** with G6.

### G6 — Function-entry pilot

- **Scope:** the four entries of the prototypes, written in place: `NormL1` at the top
  of `penalties.md`, `GroupNormL2` in `group_penalty.md`, `LeastSquaresLoss` in a new
  `losses.md` (its sections moved from `smooth.md` with their `@id`s and forwarding
  headings), and `IndBallL1` in a new `constraints.md`. The four docstrings are
  aligned with the entry template.
- **Acceptance:** each entry follows the [template](editorial-principles.md#function-entries)
  and fits about two laptop screens; nothing is removed that is not available on the
  guarantee page or elsewhere on the same page; every forwarding heading is listed.
  The maintainer approves the pilot before G7 starts.
- **Areas:** `penalties.md`, `group_penalty.md`, `smooth.md`, `losses.md` (new),
  `constraints.md` (new), four docstrings, `make.jl`.

### G7a–d — Remaining function entries

- **Scope:** one PR per role page: losses (4 entries), sparsity penalties (5),
  norm and group penalties (6), constraints (5); quadratic, linear and `Scaled`
  (6) can join whichever finishes first. Long derivations and edge-case lists move
  to the guarantee pages or into `!!! details` blocks.
- **Acceptance:** as G6, for every entry on the page.
- **Areas:** one role page and its docstrings each. Docs-only once G6 has created the
  pages, unless a docstring changes, in which case it touches `src/` (docstrings only).
- **Parallel:** all four.

### G8 — Examples, figures and common pitfalls

- **Scope:** the [prototype figures](prototypes/visualizations.md) moved into
  `docs/generate_learn_figures.py`, with the figure check added to the documentation
  job and the numerical check run by the build; example titles changed to tasks; one
  new example that changes one argument at a time (lasso → MCP → `SparseIterate`); a
  `pitfalls.md` page whose entries link to each rule's owner.
- **Acceptance:** every featured example runs on `master`; each pitfall has code,
  output and a fix, and states no rule that is not owned elsewhere; each figure is
  verified against ProximalBase by the build.
- **Areas:** `examples/`, `docs/generate_learn_figures.py`, `docs/src/assets/`,
  `pitfalls.md` (new), CI documentation job, `make.jl`.
- **Parallel:** with G4–G7 and G9.

### G9 — User-facing guarantees

- **Scope:** `spec/index.md` rewritten as the [guarantee overview](prototypes/guarantees.md);
  its authoring rules moved to `developer/contract_pages.md`; the "Migration state"
  column removed (with `check_contract_schema.py`); process wording on family pages
  ("This pilot gathers…", "verbatim", "Links only.") reworded without changing any rule;
  the ownership table of `numerical_usage.md` rewritten as short prose per function;
  closed-issue history moved to `migration.md` or `docs/design/`.
- **Acceptance:** a diff review confirms that every guaranteed, precondition,
  limitation and non-guarantee statement survives with its meaning; the open-limitations
  table lists each open limitation exactly once; no user page mentions the migration
  process. The treatment of empty schema sections follows the maintainer's
  [decision](README.md#decisions-for-the-maintainer).
- **Areas:** `spec/`, `numerical_usage.md`, `numerics.md`, `developer/`,
  `docs/check_contract_schema.py`.
- **Parallel:** with G4–G8.

### G10 — Independent usability review and final cleanup

- **Scope:** a reviewer who has not followed the work does the ten tasks of the
  [audit](audit.md#ten-tasks-from-the-home-page) from Home and reports paths, time and
  friction with screenshots; small fixes; the catalog reconciled with the 0.7 release
  candidate.
- **Acceptance:** the report states whether people or agents did the review; each task
  is completed within the step budget of [the architecture](architecture.md) or the
  remaining friction is recorded as an issue; all ordinary documentation checks pass;
  the `dev` and release manuals link correctly.

## What is deliberately not planned

- Per-function pages, a generated capability table, a new documentation framework or
  Documenter plugin, a migration ledger, and new preservation infrastructure.
- Runtime changes. The display of function objects and the `MethodError` raised by
  `gradient` on nonsmooth functions are recorded as [separate findings](audit.md#defects-and-gaps-recorded-separately).
- Backporting any of this to `release/0.6`.
