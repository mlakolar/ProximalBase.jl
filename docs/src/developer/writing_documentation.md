# Writing useful documentation

API help should let a reader make the next correct call. The manual should
explain the model, storage and numerical conditions behind that call. Keep
historical decisions and measurement reports in `docs/design/` or their issue
records. Preserve traceability when moving a historical issue reference out of
source comments; do not silently delete the only pointer to an active rule. The [editorial audit](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/documentation-style-921.md)
records the inventory, reviewed samples and preservation map for this structure.

## API help

Prefer the following order, adapting it to the operation:

1. Four-space-indented signatures.
2. One sentence describing the result, then its mathematical definition.
3. Parameters, required shapes and essential ownership rules.
4. Return value and mutation behavior.
5. A short deterministic example.
6. Critical restrictions and checked links to detailed rules.

About 250 words is a useful review threshold, not a hard limit. Keep a
restriction in the docstring if a reader needs it to avoid an unsafe alias,
choose the correct shape, or understand an unsupported operation. For example,
the return values of `value_and_gradient!` and `value_and_prox!` refer to
different points and must stay explicit.

Move a detailed rule only after it has a canonical rendered destination.
Use the [numerical contract](../numerics.md) for precision and endpoint rules,
[array and precision conventions](../numerical_usage.md) for arrays and ownership, and concrete
function or iterate guides for their method-specific behavior. Brief safety
reminders can appear in several help entries; a long rationale needs one owner.

## New function families

Put a new family's numerical rules in `docs/src/spec/<family>.md`, using all
fourteen headings in the [family-page schema](../spec/index.md#Family-page-schema).
That page owns its family-specific rules and links to shared rules by their
named anchors. Shared range and endpoint rules remain in
[the numerical hub](../numerics.md); shared array and ownership rules remain in
[array and precision conventions](../numerical_usage.md). A section containing
only links must say so. Add source and test evidence, executable examples where
useful, and entries in the contract family map and navigation. User guides and
API help link to the detailed contract.

State results and accuracy in the contract; put path-selection thresholds and
algorithm steps in a paragraph labelled **Implementation notes (not guarantees)**.
Give all fourteen section headings stable `spec-<family>-<section>` IDs, using
the suffixes in the family-page schema; preserve any existing explicit ID.
A thin constructor that only wraps another family gets a family-map row linking
to that family's owner, without a separate contract page. Until that owner moves,
link to its existing detailed rules.

This rule routes newly authored documentation. Converting existing numerical
prose into clauses remains held until the token and example checks in
[migration #1019](https://github.com/mlakolar/ProximalBase.jl/issues/1019) protect
`reword` and `split`, including exact reviewed allowances for intentional changes.

## Examples and cross-references

Use `jldoctest` for a transcript whose exact output matters. Prefer small
exact values, tuples and Boolean checks over random output or long array
displays. Source docstrings inherit `using ProximalBase, LinearAlgebra` from
`DocTestSetup`; a manual transcript must supply its imports or its own setup.
Retain `@example` when several blocks intentionally share variables. Add
`@assert` where the displayed calculation needs an invariant checked.

Use checked links to API symbols, such as [`prox!`](@ref), and named page anchors.
Preserve existing heading IDs when adding explicit anchors, or repair every
incoming fragment link. Method-specific reference entries, including the
three-factor products and `dropzeros!` overloads, need their full signatures.
The API coverage checker includes qualified public names as well as exports;
keep their explicit reference entries and coverage rows. Adding or removing a
function family means updating the first-use [chooser](../choosing.md), the
[function library](../function_library.md), the API coverage matrix, navigation
and the architectural source map. Do not leave obsolete references behind.

When a heading is renamed or its content moves, retain a forwarding stub for
an explicit `@id` or a heading with an authored incoming link or source citation.
Check the manual, docstrings, repository entry documents and examples before
retiring an automatic heading fragment. Generated self-permalinks and navigation
links do not require a stub. Record an uncited automatic heading's retirement
and its successor in the migration audit; preserve the frozen inventory as
evidence of the earlier page.

## Review and validation

For editorial changes, compare every removed guarantee with its destination.
Check the implemented interface and tests before simplifying a statement;
do not import APIs from the archived package or decide a new numerical policy.
Source comments should explain present invariants and arithmetic choices.
Prefer a description of a current invariant to an unexplained historical rule
label. Do not partially remove indexed labels or issue references from source
comments during an unrelated editorial change; audit them in a separate pass.
REPL quick help should use readable Unicode or code spans for formulas, while
the detailed manual may use rendered LaTeX. Check both displays.

Run the [strict documentation workflow](change_process.md), including doctests,
isolated examples, Markdown and rendered-link checks. Inspect actual rendered
help, mathematical formulas and navigation. A successful Markdown diff review
does not establish that signatures, links or examples render correctly.

An optional future source linter can begin in report-only mode after its
parser assumptions and fixtures are audited. Word counts alone are not a
blocking correctness check and do not justify dropping a precondition.

## Numerical routines and Learn exercises

Describe the calculation and its current implementation in a
[numerical routine](@ref numerical-routines). Pin the source and test revision
used to verify the description. Link supported arguments and numerical claims
to the section that owns them in the [family contracts](../spec/index.md).

Use ordinary finite inputs in Learn exercises. Give readers enough information
to predict the result, then use a Documenter example to check it against the
derivation or a reference calculation. The
[floating-point range lesson](@ref learn-floating-point) uses an overflowing
intermediate to explain working-type arithmetic.

Regenerate the Learn figures from the repository root.

```sh
python3 docs/generate_learn_figures.py
python3 docs/generate_learn_figures.py --check
```
