# [Numerical guarantees](@id numerical-contracts-guide)

*Prototype for `docs/src/spec/index.md`, not for publication as written: the page keeps
its path and its explicit anchor, and the authoring rules it holds today move to a
developer page. The family table below replaces the current family map with one row per
family and the same links, without the "Migration state" column and with a role column;
G9 updates `docs/check_contract_schema.py` to parse it with the same coverage and owner
checks.*

The function entries describe what each function computes for ordinary finite
inputs. The pages in this section state what is promised and what is not, where it is
documented, including the rules and known gaps for very large or very small values,
infinities, `NaN`, zero or infinite parameters and steps, and kinds of storage. Most users need them only when an
algorithm meets extreme data, when tests depend on exact results, or when they
contribute a function.

## Rules that apply across functions

| Page | Read it when |
|:--|:--|
| [Arrays, precision and aliasing](@ref numerical-usage) | you pass views, mix `Float32` and `Float64`, reuse output buffers or call from several tasks |
| [Range, non-finite values and sparse products](@ref numerics) | values may overflow or underflow, inputs may contain `Inf` or `NaN`, or you multiply sparse iterates |
| [Why floating-point range matters](@ref learn-floating-point) | you want a worked example before reading the range rules |

The shared rules apply alongside each family's rules, and the family pages link to them.

## Guarantees for each function

| Family or operation | Role | Guarantee page |
|:--|:--|:--|
| `LeastSquaresLoss` | loss | [guarantees](@ref spec-leastsquares) |
| `LogisticLoss` | loss | [guarantees](@ref spec-logisticloss) |
| `PoissonLoss` | loss | [guarantees](@ref spec-poissonloss) |
| `SqrtLeastSquaresLoss` | loss | [guarantees](@ref spec-sqrt_leastsquares) |
| `LogDetLoss` | loss | [guarantees](@ref spec-logdet) |
| `NormL1` | sparsity penalty | [guarantees](@ref spec-norml1) |
| `ElasticNet` | sparsity penalty | [guarantees](@ref spec-elasticnet) |
| `NormL1Nonneg` | sparsity penalty | [guarantees](@ref spec-norml1nonneg) |
| `MCP` | sparsity penalty | [guarantees](@ref mcp-contract) |
| `SCAD` | sparsity penalty | [guarantees](@ref spec-scad) |
| `NormL0` | sparsity penalty | [guarantees](@ref spec-norml0); for `diagonal = false`, [its own page](@ref spec-norml0offdiag) |
| `NormL2` | norm penalty | [guarantees](@ref spec-norml2) |
| `NormLinf` | norm penalty | [guarantees](@ref spec-normlinf) |
| `GroupNormL2` | group penalty | [guarantees](@ref spec-groupnorml2) |
| `SparseGroupNormL2` | group penalty | [guarantees](@ref spec-sparsegroupnorml2) |
| `GroupNormLinf` | group penalty | [guarantees](@ref spec-groupnormlinf) |
| `GroupMCP` | group penalty | [guarantees](@ref spec-groupmcp) |
| `GroupSCAD` | group penalty | [guarantees](@ref spec-groupscad) |
| `IndBox` | constraint | [guarantees](@ref spec-indbox) |
| `IndBallLinf` | constraint | constructs an `IndBox`: [its endpoint rules](@ref spec-indbox-endpoints) |
| `IndNonnegative` | constraint | [guarantees](@ref spec-indnonnegative) |
| `IndBallL1` | constraint | [guarantees](@ref spec-indballl1) |
| `IndBallL2` | constraint | [guarantees](@ref spec-indballl2) |
| `IndBallL0` | constraint | [guarantees](@ref spec-indballl0) |
| `SqrNormL2` | quadratic | [guarantees](@ref spec-sqrnorml2) |
| `QuadraticFunction` | quadratic | [guarantees](@ref spec-quadratic) |
| `KronQuadratic` | quadratic | [guarantees](@ref spec-kronquadratic) |
| `LinearFunction` and `ZeroFunction` | linear | [guarantees](@ref spec-linear) |
| `Scaled` | wrapper | [guarantees](@ref spec-scaled) |
| `kkt_violation` | operation | [guarantees](@ref spec-kkt_violation) |

## How to read a guarantee page

Each page has the same fourteen sections, from the definition and supported arguments
to the behavior at extreme values, the evidence and the history, including a section
**Limitations and non-guarantees** (for example [`NormL1`](@ref spec-norml1-limitations)).
Some statements carry one of these words:

| Word on the page | Meaning |
|:--|:--|
| guaranteed | Promised. Changing it is a breaking change, and it needs test evidence. |
| precondition | Your obligation. It is not checked unless the statement says so. |
| limitation | A known gap, linked to an issue. |
| non-guarantee | Explicitly not promised. |
| characterized | Current behavior, recorded so that a change is noticed, but not promised. |
| unclassified | The text does not establish a status, and the maintainer has to decide one. Do not rely on it as a promise. |

Statements without one of these words are unclassified: assigning a status to every
statement is pending review. When sources disagree, the order of authority is: the
source code and its tests, then the guarantee pages, then the user and learning guides,
then the design history. Please [report the discrepancy](https://github.com/mlakolar/ProximalBase.jl/issues).

## For contributors

How these pages are structured, how statements get a status and how the pages are
checked is described in [How the guarantee pages are written](@ref contract-pages).

### Family-page schema

The fourteen-section schema of the family pages is described in
[How the guarantee pages are written](@ref contract-pages-schema).
