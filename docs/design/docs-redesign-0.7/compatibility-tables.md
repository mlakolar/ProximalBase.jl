# Designing the capability tables

The [prototype](prototypes/capabilities.md) shows the result. This page
explains the choices.

## Three questions, three places

| Question | Answered by |
|:--|:--|
| Which operations does a function support? | Function × operation tables, one per role |
| Which storage kinds does a function accept? | Function × storage tables, one per role |
| Which functions can I use with *this* storage, and with which operations? | A short list on each iterate page, taken from the same data |

A single table with all three dimensions (function × operation × storage) would have
about 30 × 8 × 6 cells and could not be read. Splitting by the question a reader asks
keeps each table small. The third question is a slice of the first two, so it goes on
the iterate page where readers of the iterate guides already are, not into a third table.

**Separate tables per storage kind** (dense, sparse, symmetric, block) were
considered and rejected for the main reference. Most functions behave identically on
dense arrays, `SparseIterate` and `BlockIterate`, so four tables would repeat the same
✓ three times and hide the few differences. One function × storage table puts those
differences side by side. The per-storage slices on the iterate pages serve the
reader who starts from the storage.

## Split by role, not by operation

Readers filter by what they are modeling (a loss, a penalty, a constraint), so
there is one pair of tables per role, as in Convex.jl's tables per cone. Each table has
at most eight columns with one-word headers and fits the 816 px content column of a
desktop screen. On a 390 px phone (358 px content column) the tables of the
[preview](README.md#previews) are 460–631 px wide and scroll sideways; today's catalog
is 806 px and the coverage matrix 1,537 px. Documenter offers no fixed first column, and
a card layout would need custom HTML, so moderate scrolling is the accepted cost.

## Columns

**Operations:** value, grad (gradient), prox, coord (`prox_block` on one entry), group
(`prox_block` on a compact group), score (`kkt_violation`, the optimality score, with
the forms available), and restrict (`restrict`). `lipschitz`, `value_entry` and the
supplied-predictor oracles are rarer and stay in the function entries.

**Storage:** vector, matrix (dense arrays and strided views), sparse (`SparseIterate`),
block (`BlockIterate`), symmetric (`SymmetricSparseIterate`). A `Symmetric` matrix
behaves the same for every function, so it is one sentence above the tables rather
than a column. Element types are not a column:
the supported precision is `Float64` or `Float32` for every function, used for data,
parameters, step and output alike, so it is one sentence above the tables. Other
element types are not guaranteed and are not tabulated, even where a probe shows that
they work.

## Symbols

| Symbol | Meaning |
|:--|:--|
| ✓ | supported for every input of the column's kind |
| ◐ | supported with a condition, stated in a numbered note under the table |
| — | not supported: the call throws, or is documented as unsupported |
| ? | not verified (to be resolved before the table is published) |

No cell is left blank, unlike PETSc's tables, where a blank can mean "no" or "not
marked". Symbols are text characters, not images, so search and screen readers can
read them. A ◐ never hides a condition that changes correctness: the condition is in
the note, one sentence, with a link to its owner.

## What counts as evidence

A cell is ✓ only when a test or a recorded probe exercises that operation on that kind
of input; a method that exists, a trait value (`isproximable`, `isseparable`) or a
generic fallback is not enough (#1212 G4). A — means the call was seen to throw or the
documentation says it is unsupported. The evidence for each cell is recorded in the API
coverage matrix of the developer guide, not in the user table.

Two distinctions from #1212 are kept visible:

- **A projection is not an arbitrary prox.** Constraint rows say "projection" in the
  prox column, and their prox ignores the step.
- **Coordinate support is not group support.** Separate columns, because `NormL1` has
  one and `GroupNormL2` has the other.

The tables describe what ProximalBase supports. Whether a ProximalSolvers algorithm
accepts a function is a property of that solver, and the tables link to the
ProximalSolvers documentation instead of claiming it.

## Maintenance

- The tables are hand-written Markdown in one page. A change to a family's operations
  updates its rows in the same PR, in the family-change checklist of
  `developer/writing_documentation.md`.
- The function entries and the catalog repeat only a one-line summary and link to the
  tables. The docstrings do not repeat support lists.
- No generation script, runtime capability API or test framework is added. The
  capability audit for this proposal used throwaway probes; G4 records the evidence for
  each cell in the coverage matrix.
