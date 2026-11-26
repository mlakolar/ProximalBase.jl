# Editorial principles

These principles are for user-facing pages: everything outside the Developer
guide and the family guarantee pages. They add to the existing
[writing guide](../../src/developer/writing_documentation.md), which keeps its
rules on anchors, executable examples, one owner per rule and docstrings. Once
approved, they would become a short section of that guide.

## Principles

1. **Write for the user first.** A page in Getting started, Functions or the User
   guide answers a user's question. Contributor procedure, review conventions and
   source-to-test maps belong in the Developer guide.

2. **Simple case first, contract second.** Show the ordinary call with ordinary
   `Float64` arrays before shapes, element types, aliasing, extreme values and
   `NaN`. The guarantee page is one link away; it is not the first thing on the page.

3. **One notation throughout.** ``x`` is the argument of a penalty or constraint and
   ``\beta`` the coefficients of a loss; ``\lambda`` is a level, ``w`` weights,
   ``\gamma>0`` the proximal step, ``G_k`` a group; ``\|\cdot\|`` is Euclidean (Frobenius
   for matrices) unless marked. Every formula states its normalization: write
   ``\tfrac12\|y-X\beta\|_2^2``, not "the least-squares loss".

4. **Separate what is promised from how it is computed.** A function entry
   states the mathematical result and the material limits. How it is computed
   (thresholds, recovery paths, kernels) belongs in the guarantee pages and the
   numerical routines. When a page describes current behavior that is not
   promised, it says so.

5. **No history in user pages.** Do not write "restored in #1160", "this pilot",
   "verbatim migration" or "before #373". The current state is what the reader
   needs; history belongs in [Changes from earlier versions](../../src/migration.md),
   commit messages and `docs/design/`. An issue link is fine where it points to an open limitation.

6. **Say it once, link to it everywhere else.** Each fact has one owner: formulas
   and examples in the function entry, support in the capability tables, edge cases
   in the guarantee page, signatures in the docstring. A safety-critical reminder
   (aliasing, an unchecked precondition) may be repeated in one sentence with a link.

7. **Keep important limits visible.** A limitation that changes whether a user's
   code is correct (an unchecked symmetry precondition, a nonconvex prox tie, a
   shared buffer that is unsafe across tasks) appears in the function entry's
   "Good to know" list, not only on the guarantee page.

8. **Every example runs.** Examples are `@example` blocks, doctests or Literate
   scripts, run by the strict build. Prefer showing the output and saying why it is
   useful ("the third entry became exactly zero") over a wall of `@assert`s. Put
   independent checks in tests, or keep one `@assert` that teaches something.

9. **Make support and non-support easy to find.** Every function entry has a
   support summary that links to the capability tables. "Not supported" is stated
   plainly, with the alternative where one exists.

10. **Preserve every guarantee.** An editorial change may move, shorten or
    reword a statement but must not change what is promised. If a page and the
    code disagree, open an issue instead of picking one.

11. **Prefer plain prose to structures.** A short paragraph beats an empty heading.
    Avoid section headings with no content (the guarantee pages contain "Links only."
    206 times), "Assumes · You will learn" banners on reference pages, and tables
    whose cells are paragraphs. A table cell holds a few words; longer text goes in prose.

## Style rules

- **Length.** A function entry fits on about two screens on a laptop. A table has at most
  eight columns, one-word headers and cells of about 25 characters or fewer.
- **Titles say what the page does**: "Store groups of coefficients with
  `BlockIterate`", not "Ordered grouped storage".
- **Admonitions have fixed meanings**: `!!! warning` for something that gives
  wrong results or errors; `!!! tip` for practical advice; `!!! note` for a
  precondition or scope statement; `!!! details` for an optional derivation.
- **Define a term where it first appears** or link to the glossary. Terms
  such as working set, active zero, compact block, KKT score and zero-preserving
  are not used on Home or in the quick start.
- **Use symbols with their legend**: ✓ implemented for ordinary inputs (conditions in the
  entry), ◐ with a lettered note stating the restriction, — not implemented. Never leave a
  capability cell blank.

## Function entries

Every public function type gets one entry, in this order. The four prototypes
([`NormL1`](prototypes/functions/norml1.md),
[`LeastSquaresLoss`](prototypes/functions/leastsquaresloss.md),
[`GroupNormL2`](prototypes/functions/groupnorml2.md),
[`IndBallL1`](prototypes/functions/indballl1.md)) follow it.

1. **Heading and one sentence**: mathematical name, Julia name, what it is used for.
2. **Definition**: the formula with its normalization, the domain, and the
   meaning of each parameter.
3. **Constructor**: signatures, defaults and what is checked at construction, as a
   short parameter table.
4. **Example**: four to eight lines with visible output.
5. **Operations**: the main operation (prox, projection or gradient) with its
   formula, then a short table of the operations it supports. Calls appear in the
   order a user would make them.
6. **Inputs and storage**: one line per storage kind, from the capability tables.
7. **Good to know**: three to six bullets of material limitations and
   preconditions, each linking to its owner.
8. **See also**: related functions, the guarantee page, the API entry.

The docstring stays the signature reference. It keeps its summary, formula,
example and edge cases and links to the entry; it does not copy the support
table.
