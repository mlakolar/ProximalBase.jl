# Designing the capability tables

The [prototype](prototypes/capabilities.md) shows the result. This page explains the
choices.

## One reference, two tables per role

| Question | Answered by |
|:--|:--|
| Which operations does a function implement? | the operations table of its role |
| Which kinds of input do value and prox accept? | the inputs table of its role |
| Does *this* operation work on *this* input? | the notes of both tables, then the function's entry |

A table of every function × operation × input would have about 30 × 8 × 6 cells and
could not be read. The two tables per role are its margins, and margins cannot answer
every combination: `NormL2` accepts a `SymmetricSparseIterate` for its value but not for
prox, `IndBallL0` on a matrix `SparseIterate` needs an output of the same kind, and
`LogDetLoss` needs a dense output for mutating calls. Such combinations are written as
notes on the cell they restrict, and the entry of each function gives its full input
rules. There is one capability reference: the iterate pages link to it rather than
keeping their own copies, and the catalog helps choose a function without listing support.

**Separate tables per storage kind** (dense, sparse, symmetric, block) were considered
and rejected. Most functions behave alike on dense arrays, `SparseIterate` and
`BlockIterate`, so four tables would repeat the same marks and hide the few
differences, which one inputs table puts side by side.

## Split by role

Readers filter by what they are modeling (a loss, a penalty, a constraint), so there
is one pair of tables per role, as Convex.jl has one table per cone. Each table has at
most eight columns with one-word headers and fits the 816 px content column of a desktop
screen. On a 390 px phone (358 px content column) the tables of the
[preview](README.md#previews) are 460–631 px wide and scroll sideways; today's catalog is
806 px and the coverage matrix 1,537 px. Documenter offers no fixed first column, and a
card layout would need custom HTML, so moderate scrolling is the accepted cost.

## Columns

**Operations:** value, grad (gradient), prox, coord (`prox_block` on one entry), group
(`prox_block` and `prox_block!` on a group's buffer), score (`kkt_violation`, with the
forms available) and restrict. `lipschitz` and the absence of a Hessian product are one
paragraph below the tables; `value_entry` and the held-predictor calls are in the
entries.

**Inputs:** vector, matrix (dense arrays and strided views), sparse (`SparseIterate`),
block (`BlockIterate`), symmetric (`SymmetricSparseIterate`). `Symmetric` wrappers differ
between functions (most return a dense matrix from `prox`, `ZeroFunction` writes into a
`Symmetric` output, the off-diagonal `NormL0` refuses them), so they are covered in the
entries rather than in a column. Element types are not a column: the supported
precision is `Float64` or `Float32` for every function, used for data, parameters, step
and output alike, so it is one sentence above the tables. Other element types are not
guaranteed and are not tabulated, even where a probe shows that they work.

## Symbols

| Symbol | Meaning |
|:--|:--|
| ✓ | implemented for the function's ordinary inputs; the entry gives the conditions (shapes, steps, preconditions) |
| ◐ *a* | implemented with the restriction in note *a* |
| — | not implemented: the call throws |
| ? | not verified; G4 uses it for any cell it cannot confirm with a test |

A ✓ is not a promise that every input works: it means the operation is implemented and
its conditions are on the entry. No cell is left blank, unlike PETSc's tables, where a
blank can mean "no" or "not marked". Symbols are text characters, not images, so search
and screen readers can read them. Notes use letters, written in the cell and at the
start of the note, because Documenter restarts the numbering of each list. A note is one
sentence and links to its owner when the condition affects correctness.

## What counts as evidence

A cell is ✓ or ◐ only when a test or a recorded probe exercises that operation on that
kind of input; a method that exists, a trait value (`isproximable`, `isseparable`) or a
generic fallback is not enough (#1212 G4). A — means the call was seen to throw or is
documented as unsupported. In the prototype, the cells of the four pilot functions rest
on the reproducible scripts of the [verification record](verification/README.md), and the
other cells on an exploratory probe run and the docstrings. For the published table, G4
records the test behind each cell in the API coverage matrix of the developer guide, not
in the user table.

Two distinctions from #1212 stay visible:

- **A projection is not an arbitrary prox.** The constraints section says that prox is
  the projection onto the set.
- **Coordinate support is not group support.** They are separate columns, because
  `NormL1` has one and `GroupNormL2` has the other.

The tables describe what ProximalBase implements. Whether a ProximalSolvers algorithm
accepts a function is a property of that solver, and the tables link to its
supported-problems page instead of claiming it.

## Maintenance

- The tables are hand-written Markdown on one page. A change to a family's operations
  updates its rows in the same PR, as a checklist item of the family-change process in
  `developer/writing_documentation.md`.
- The catalog and the entries link to the tables; docstrings do not repeat support lists.
- No generation script, runtime capability API or test framework is added.
