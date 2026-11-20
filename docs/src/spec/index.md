# [How to read the numerical contracts](@id numerical-contracts-guide)

The order of authority is **current source and tests, numerical contracts,
user and learning guides, then design history**. If two disagree, report the
disagreement as an issue. A documentation migration preserves the original
wording and does not decide a new numerical policy or change an implementation.

## Shared rules

Start with [Array and precision conventions](@ref numerical-usage) for arrays,
precision, output ownership and aliases, and the
[Numerical contract](@ref numerics) for range recovery, non-finite values,
sparse products and rules shared by several families.

Each family page is an entry point to its own rules and to these shared owners.
The API reference continues to own signatures and docstrings. Brief safety
reminders may repeat; detailed rules link to their owner.

## Family map

`NormL1` was the first verbatim migration pilot. The map distinguishes migrated
families, newly authored contracts and families still at their existing owners.
Thin constructors link to the family they wrap. The shared numerical rules above
apply alongside each entry.

| Family or operation | Current entry point | Migration state |
|:--|:--|:--|
| `NormL0` | [API entry](@ref NormL0) and [cardinality endpoint rules](../numerics.md#Cardinality-penalty-endpoints) | Existing pages |
| `NormL1` | [Numerical contract](norml1.md) | Verbatim pilot |
| `MCP` | [Numerical contract](mcp.md) | Authored contract |
| `SCAD` | [Numerical contract](scad.md) | Authored contract |
| `NormL1Nonneg` | [Numerical contract](norml1nonneg.md) | Verbatim migration |
| `ElasticNet` | [Numerical contract](elasticnet.md) | Verbatim migration |
| `IndNonnegative` | [Numerical contract](indnonnegative.md) | Verbatim migration |
| `IndBox` | [API entry](@ref IndBox) and [numerical endpoint rules](../numerics.md#NaN,-infinities-and-zeros) | Existing pages |
| `IndBallLinf` | [IndBox API entry](@ref IndBox) and its [numerical endpoint rules](../numerics.md#NaN,-infinities-and-zeros) | Thin constructor of `IndBox` |
| `IndBallL0` | [Numerical contract](indballl0.md) | Verbatim migration |
| `IndBallL1` | [Numerical contract](indballl1.md) | Authored contract |
| `IndBallL2` | [API entry](@ref IndBallL2), [range guarantees](../numerics.md#numerical-value-range) and [numerical endpoint rules](../numerics.md#NaN,-infinities-and-zeros) | Existing pages |
| `SqrNormL2` | [Numerical contract](sqrnorml2.md) | Verbatim migration |
| `LeastSquaresLoss` | [Least-squares loss](@ref Least-squares-loss), including predictor and residual oracles | Existing pages |
| `LogDetLoss` | [Log-determinant loss](@ref Log-determinant-loss) | Existing pages |
| `LogisticLoss` | [Logistic regression](@ref Logistic-regression-loss), including predictor oracles | Existing pages |
| `SqrtLeastSquaresLoss` | [Square-root least squares](../smooth.md#Square-root-least-squares-is-not-smooth) | Existing pages |
| `QuadraticFunction` | [Quadratic function](@ref Quadratic-function) | Existing pages |
| `LinearFunction` and `ZeroFunction` | [Linear function](@ref Linear-function) and [zero function](@ref Zero-function) | Existing pages |
| `NormL2` | [The ℓ₂ norm](@ref norm-l2-guide) | Existing pages |
| `NormLinf` | [The ℓ∞ norm](../group_penalty.md#The-NormLinf-norm) | Existing pages |
| `GroupNormL2` | [The group ℓ₂ norm](@ref group-l2-guide) | Existing pages |
| `SparseGroupNormL2` | [Sparse group lasso](@ref sparse-group-l2-guide) and [endpoint rules](@ref sparse-group-l2-endpoints) | Existing pages |
| `GroupNormLinf` | [The group ℓ∞ norm](@ref group-linf-guide) and [clip accuracy](@ref groupnormlinf-clip-accuracy) | Existing pages |
| `Scaled` | [Scaling a function](@ref scaling-guide) | Existing pages |
| `kkt_violation` | [Checking optimality with KKT conditions](../operations.md#Checking-optimality-with-KKT-conditions) and [numerical endpoint rules](../numerics.md#NaN,-infinities-and-zeros) | Existing pages |

## Status vocabulary

Clause status must follow an explicit statement in the owning text and its
evidence. An implementer must not promote a statement to a guarantee because
the current implementation happens to produce that result.

| Status | Meaning |
|:--|:--|
| `guaranteed` | A promised behavior that must hold and needs test evidence; changing that promise is a breaking change |
| `characterized` | Current behavior recorded so a change is noticed, without promising it |
| `limitation` | A known gap linked to an issue; do not fix it during a documentation change |
| `precondition` | A caller obligation, unchecked unless the statement explicitly says otherwise |
| `non-guarantee` | Behavior explicitly not promised |
| `unclassified` | The source text does not establish a status; an owner decision is required |

The pilot retains the existing prose and endpoint table. It does not silently
assign a status to every sentence. Clause conversion follows the review of
that mechanical move.

## Clause identifiers and ownership

After conversion, each clause uses a permanent identifier of the form
`family.operation.case`, with lowercase words and hyphens within each part.
An identifier is not reused or renumbered. A withdrawn clause keeps its
identifier and a withdrawn marker. Short letter-and-number rule labels are
not introduced into source comments.

Each clause has one owning section. Other pages link to it. Routine pages
explain computations and cite the owning clauses; they do not add guarantees.
The [writing rules](../developer/writing_documentation.md) and
[change process](../developer/change_process.md) govern preservation and review.

## Family-page schema

Every family page uses these headings in this order:

Use stable explicit IDs `spec-<family>-<section>` on all fourteen headings.
The suffixes, in order, are `identity`, `definition`, `traits`, `arguments`,
`preconditions`, `results`, `ordinary`, `range`, `endpoints`, `failures`,
`limitations`, `elsewhere`, `evidence` and `history`. Preserve an existing
explicit ID, such as `scad-endpoints`, when applying this convention to a
published page.

1. Identity.
2. Definition and domain.
3. Traits.
4. Supported arguments.
5. Preconditions.
6. Results.
7. Ordinary-path clauses.
8. Range clauses.
9. Endpoint clauses.
10. Failure behavior.
11. Limitations and non-guarantees.
12. Clauses owned elsewhere.
13. Evidence.
14. History.

During a verbatim migration, a section can link to an existing owner instead
of duplicating a statement. Converting text into clauses is a separate,
independently reviewed change. Ambiguous status or conflicting statements go
to the migration's open questions for an owner decision.
