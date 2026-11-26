# Proposed information architecture

## The central idea: reorganize the sidebar, not the file tree

In Documenter a page's place in the sidebar is set in `docs/make.jl` and is
independent of its file path. Almost all of the redesign can therefore be done by
changing the navigation, rewriting a few entry pages and adding six new ones
(five for users, one for contributors),
while **every existing page keeps its path and therefore its public URL**. The only
content that changes page is a set of sections moving between existing pages,
and those leave forwarding headings behind, as the manual already does
(`penalties.md#l1-endpoints` is an example).

This avoids the expensive part of a reorganization: about 420 relative
`page.md#Heading` links inside the manual, about 50 anchors that `src/`
docstrings reference with `@ref`, and links from the README, `AGENTS.md`,
`CONTRIBUTING.md` and the test and benchmark comments (see the
[audit](audit.md#constraints-a-redesign-must-respect)).

## Proposed navigation

Six groups after Home instead of seven, at most three levels deep, in reading
order. **New** marks a new page; every other entry is an existing file at its
current path. *Italic* titles are the proposed sidebar labels where they differ
from today's.

```text
Home                                         index.md            rewrite
Getting started
  Five-minute quick start                    quickstart.md       New
  Tutorial: a lasso problem step by step     getting_started.md
  Tutorial: proximal gradient methods        proximal_gradient.md
  Tutorial: coordinate descent               coordinate_descent.md
  From operations to an algorithm            optimization_algorithms.md
  Exercises: proximal operations by hand     learn/proximal_exercises.md
Functions
  Function catalog                           function_library.md rewrite; absorbs choosing.md
  Capability tables                          capabilities.md     New
  Losses                                     losses.md           New (sections from smooth.md)
  Sparsity penalties                         penalties.md
  Norm and group penalties                   group_penalty.md
  Constraints                                constraints.md      New
  Quadratic and linear functions             smooth.md           (losses moved out)
  Scaling a function                         calculus.md
User guide
  Operations on functions                    operations.md
  Function properties                        function_objects.md
  Restricting a function                     restrict.md
  Iterates and working sets                  iterates.md
  SparseIterate                              sparse_iterate.md
  BlockIterate                               block_iterate.md
  SymmetricSparseIterate                     symmetric_sparse_iterate.md
  Coordinate and block operations            iterate_oracles.md
  Symmetric parameters                       symmetric_parameters.md
  Writing your own function                  extending.md
  Common pitfalls                            pitfalls.md         New
Examples                                     examples/*.md       retitled by task
Reference
  API reference ▸ (Overview, Interface, Functions, Scaling, Iterates)   api.md, api/*.md
  Numerical guarantees ▸
    Overview                                 spec/index.md       rewrite
    Arrays, precision and aliasing           numerical_usage.md
    Range, non-finite values, sparse products numerics.md
    Why floating-point range matters         learn/floating_point.md
    NormL1 … SparseGroupNormL2 (30 pages)    spec/*.md
  Glossary                                   glossary.md
  Changes from earlier versions              migration.md
Developer guide
  Overview and package structure             developer/index.md
  System architecture                        architecture.md
  Storage internals (3 pages), views and aliasing, dispatch
  Numerical routines ▸ (11 pages)            developer/routines/*
  How the guarantee pages are written        developer/contract_pages.md New (from spec/index.md)
  Design contracts (3 pages)                 developer/contracts/*
  API coverage matrix                        api/coverage.md
  Change process                             developer/change_process.md
  Writing documentation                      developer/writing_documentation.md
```

![The proposed sidebar in the preview build, with Getting started and Functions expanded. The preview lists each prototype role page next to the current page it would replace.](screenshots/preview-sidebar.png)

Counts: 6 groups (today 7), 14 entries in "Getting started" and "Functions"
together, 30 guarantee pages one level deeper than today but collapsed by default.
The family guarantee pages and the routine pages remain one click away for reader C,
and readers A and B never see them unless they follow a link.

### Why these groups

| Group | Diátaxis kind | Reader | Answers |
|:--|:--|:--|:--|
| Getting started | tutorials | A, B | "How do I use this at all?", "How is an algorithm built from these calls?" |
| Functions | reference, with one how-to (choosing) | A first, B | "Which function do I need?", "What does it support?" |
| User guide | how-to and explanation | B, then A | "How do operations, iterates and extensions work?" |
| Examples | tutorials and how-to | A, B | "Show me a complete small program." |
| Reference | reference | B, C | signatures, guarantees, terms, version changes |
| Developer guide | contributor material | C | internals, routines, process |

The old reader tracks (Learn, Developer guide, Numerical contracts) were
organized by *kind of content*. The new groups are organized by *what the reader
is trying to do*, and #1212's requirement that path (c) not be needed for path (a)
holds because nothing in the first three groups links to process material.

### What happens to "Learn"

The Learn group disappears as a label. Its tutorials join "Getting started", the
floating-point lesson moves next to the range rules it introduces, and the glossary
joins Reference. No page is removed.

## Function entries: grouped pages, not one page per function

Every public function gets a standardized entry (template in the
[editorial principles](editorial-principles.md#function-entries)), placed as a
section on one of six role pages:

| Page | Functions |
|:--|:--|
| Losses (new) | `LeastSquaresLoss`, `LogisticLoss`, `PoissonLoss`, `SqrtLeastSquaresLoss`, `LogDetLoss` |
| Sparsity penalties | `NormL1`, `ElasticNet`, `NormL1Nonneg`, `MCP`, `SCAD`, `NormL0` (both forms) |
| Norm and group penalties | `NormL2`, `NormLinf`, `GroupNormL2`, `SparseGroupNormL2`, `GroupNormLinf`, `GroupMCP`, `GroupSCAD` |
| Constraints (new) | `IndBox`, `IndBallLinf`, `IndNonnegative`, `IndBallL1`, `IndBallL2`, `IndBallL0` |
| Quadratic and linear functions | `SqrNormL2`, `QuadraticFunction`, `KronQuadratic`, `LinearFunction`, `ZeroFunction` |
| Scaling a function | `Scaled` |

Why grouped pages: #1212 asks to prefer existing pages over per-function pages;
related functions are compared on one page (lasso, elastic net and MCP); the
sidebar stays short; and the existing `@id` anchors of these pages stay where they
are. The cost is longer pages, which the entry template keeps manageable by moving
edge-case material to the guarantee pages. A per-function layout is the
alternative listed in the [decisions](README.md#decisions-for-the-maintainer).

Today the constraints have no user page at all: the catalog links four of the six
to the general numerical contract page, one to its API entry and one to its
guarantee page. Losses are mixed with quadratic utilities
in `smooth.md`, which makes "find all loss functions" a scan of a 6,300-word page.
These are the two new role pages.

## Mapping from existing pages

Actions: **retain** (content unchanged apart from links), **retitle** (sidebar or
H1 only), **rewrite** (same path and anchors, new content), **split** (part moves
to another page, forwarding headings left behind), **consolidate** (content moves
into another page, the old path stays as a hidden forwarding page), **relocate**
(new sidebar position only), **new**. No existing page is removed.

### User pages

| Current page (sidebar title) | Action | Destination | Notes |
|:--|:--|:--|:--|
| `index.md` Home | rewrite | Home | [Prototype](prototypes/index.md). The track tables and subdifferential paragraph move to tutorials and the glossary |
| `getting_started.md` A first lasso problem | retitle, trim | Getting started | Install section replaced by a link to the quick start and the version table |
| `proximal_gradient.md` | relocate | Getting started | — |
| `coordinate_descent.md` | relocate | Getting started | — |
| `optimization_algorithms.md` What a solver adds | retitle, relocate | Getting started | The ProximalSolvers comparison moves to Home |
| `learn/proximal_exercises.md` | relocate | Getting started | — |
| `learn/floating_point.md` | relocate | Reference ▸ Numerical guarantees | Next to `numerics.md`, which it introduces |
| `glossary.md` | relocate | Reference | — |
| `choosing.md` Choose a function or operation | consolidate | Function catalog (functions), Operations (operations table) | Old path kept as a hidden page; `@id choosing` moves to the catalog |
| `function_objects.md` | retain, trim | User guide ▸ Function properties | The built-in trait table moves to the capability tables (it omits 9 functions today) |
| `operations.md` | retain | User guide | Its "at a glance" table stays; per-function support moves to the capability tables |
| `restrict.md` | relocate | User guide | — |
| `calculus.md` Scaling a function | relocate | Functions | Gains a standardized `Scaled` entry |
| `extending.md` | relocate | User guide | — |
| `function_library.md` Browse available functions | rewrite | Function catalog | [Prototype](prototypes/catalog.md) |
| `smooth.md` Smooth array functions | split, retitle | Losses (new) and "Quadratic and linear functions" | The five loss sections move with their `@id`s; forwarding headings stay for auto-slug anchors that are cited |
| `penalties.md` The ℓ₁ norm and soft thresholding | rewrite in place | Sparsity penalties | [NormL1 prototype](prototypes/functions/norml1.md) |
| `group_penalty.md` The ℓ₂ and group ℓ₂ norms | rewrite in place, retitle | Norm and group penalties | [GroupNormL2 prototype](prototypes/functions/groupnorml2.md) |
| `iterates.md`, `sparse_iterate.md`, `block_iterate.md`, `symmetric_sparse_iterate.md`, `iterate_oracles.md`, `symmetric_parameters.md` | relocate | User guide | Each iterate page gains a short "functions you can use with this storage" table |
| `numerical_usage.md` | relocate, retitle | Reference ▸ Numerical guarantees | Its ownership table needs plain-language rewriting (G7) |
| `numerics.md` | relocate | Reference ▸ Numerical guarantees | Path and section titles fixed: 34 source and test comments cite them |
| `migration.md` Compatibility and migration | relocate, retitle | Reference ▸ Changes from earlier versions | — |
| `interface.md`, `storage.md` (hidden) | retain | hidden | Existing forwarding pages |
| — | new | `quickstart.md`, `capabilities.md`, `losses.md`, `constraints.md`, `pitfalls.md` | — |

### Examples

The nine Literate pages keep their paths. Titles change to say what each
example does:

| File | Current title | Proposed title |
|:--|:--|:--|
| `lasso` | Least-squares and lasso primitives | Evaluate a lasso objective and its proximal step |
| `proximal_gradient` | Proximal-gradient iteration | Write a proximal-gradient loop |
| `coordinate_operations` | Coordinate-level operations | Update one coordinate at a time |
| `sparse_working_sets` | Sparse working sets | Keep a working set with `SparseIterate` |
| `block_storage` | Ordered grouped storage | Store groups of coefficients with `BlockIterate` |
| `symmetric_storage` | Independent symmetric coordinates | Work with a symmetric matrix variable |
| `group_lasso_prox` | Group-lasso proximal building blocks | Take group-lasso proximal steps |
| `custom_function` | A custom separable function | Write a separable function |
| `block_prox_extension` | A compact-vector block extension | Add a block proximal operator to your function |

### Reference, guarantees and developer pages

| Current page | Action | Destination |
|:--|:--|:--|
| `api.md`, `api/interface.md`, `api/functions.md`, `api/calculus.md`, `api/iterates.md` | relocate | Reference ▸ API reference |
| `api/coverage.md` | relocate | Developer guide (it maps symbols to tests, a maintainer view) |
| `spec/index.md` How to read the contracts | split, rewrite | User half stays at this path as "Numerical guarantees: overview" ([prototype](prototypes/guarantees.md)); the authoring rules move to a new `developer/contract_pages.md` |
| 30 `spec/<family>.md` | relocate | Reference ▸ Numerical guarantees (collapsed). Process words in the user-visible text (the "Links only." markers, "verbatim migration", "this pilot") are reworded in G7 without changing any rule |
| `developer/contracts/*` (generated from `docs/design/`) | relocate | Developer guide |
| `developer/index.md`, `architecture.md`, `developer/*.md`, `developer/routines/*.md` | retain | Developer guide |
| `developer/change_process.md`, `developer/writing_documentation.md` | relocate | Developer guide (from "Project") |

## Preserving links and anchors

The rules are the existing ones in `developer/writing_documentation.md`; nothing
new is introduced.

1. **Page URLs.** No page path changes, so every `/dev/<page>/` URL keeps working.
   A consolidated page (`choosing.md`) stays as a hidden forwarding page, like
   `interface.md` and `storage.md`, and is added to the hidden-page list in
   `docs/check_html.py`.
2. **Explicit anchors** (`@id`) move with their section. `@ref` links from the
   manual and from `src/` docstrings follow automatically, and the strict build
   fails if one is lost.
3. **Automatic heading anchors** that are cited (from the manual, docstrings,
   README, `AGENTS.md`, `CONTRIBUTING.md`, examples or source comments) get a
   forwarding heading on the old page when their section moves. Uncited ones may
   retire with a checked successor, as #1198 did.
4. **External links.** Only the `/dev/` root is known to be linked from
   ProximalSolvers. `/stable/` is the frozen 0.5.4 manual and is untouched.
5. **Checks that must change with the navigation**, in the same PR:
   `docs/make.jl` (pages), `docs/check_html.py` (`EXPECTED_CATEGORIES`,
   `EXPECTED_COLLAPSE_CONTROLS`, hidden pages, and the home-page check that requires
   two math elements), and for G7, `docs/check_contract_schema.py` (which requires
   the "Migration state" column header). Editing these makes that PR non-docs-only,
   so it runs the full test lanes; this is the reason to make one navigation PR
   rather than spreading `make.jl` edits over many PRs.
