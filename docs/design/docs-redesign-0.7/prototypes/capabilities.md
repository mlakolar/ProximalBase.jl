# [Capability tables](@id capability-tables)

Find what a function implements. Each role has two tables: the **operations**, and
the kinds of **input** that value and prox accept.

| Symbol | Meaning |
|:--|:--|
| ✓ | implemented for the function's ordinary inputs; the function's entry gives the conditions (shapes, steps, preconditions) |
| ◐ *a* | implemented with the restriction in note *a* below the table |
| — | not implemented: the call throws |

*Prototype note: the cells come from the G0 capability probes (Julia 1.13.4, recorded
in [the verification record](../verification/README.md)), not from a review of every
test. G4 confirms each cell against the tests before publication.*

**Operation columns**

| Column | Calls |
|:--|:--|
| value | `f(x)` |
| grad | `gradient`, `gradient!`, `value_and_gradient`, `value_and_gradient!` |
| prox | `prox`, `prox!`, `value_and_prox!` |
| coord | `prox_block(f, xᵢ, i, γ)`: the proximal point of one entry |
| group | `prox_block(f, u, k, γ)` and `prox_block!(out, f, u, k, γ)`: the proximal point of the entries `u` of group `k`, in a separate buffer |
| score | `kkt_violation`: the optimality score of a coordinate, a group or the whole point |
| restrict | `restrict(f, S)` |

**Input columns**: dense vectors, dense matrices (strided views included), *sparse*
for [`SparseIterate`](@ref), *block* for [`BlockIterate`](@ref), *symmetric* for
[`SymmetricSparseIterate`](@ref). Unless a note says otherwise, an output of the same
kind as the input is accepted. `Symmetric` matrix wrappers, element types other than
`Float64` and `Float32`, and mixed precision are covered in each entry and in the
[precision rules](@ref numerical-usage), not in these tables.

## Losses

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [`LeastSquaresLoss`](@ref) | ✓ | ✓ | ✓ | — | — | ◐ *a* | ◐ *b* |
| [`LogisticLoss`](@ref) | ✓ | ✓ | — | — | — | — | — |
| [`PoissonLoss`](@ref) | ✓ | ✓ | — | — | — | — | — |
| [`SqrtLeastSquaresLoss`](@ref) | ✓ | — | — | — | — | — | — |
| [`LogDetLoss`](@ref) | ✓ | ✓ | ◐ *c* | — | — | — | — |

- *a* — whole point only.
- *b* — vector response only.
- *c* — a finite nonnegative scalar step.

The first four also evaluate from a linear predictor that you already hold (for
`LogisticLoss`, including its offset), and all but `SqrtLeastSquaresLoss` compute a
gradient from it; see each loss's entry.

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `LeastSquaresLoss` | ✓ | ◐ *d* | ◐ *e* | ✓ | — |
| `LogisticLoss` | ◐ *f* | — | — | — | — |
| `PoissonLoss` | ◐ *g* | — | — | — | — |
| `SqrtLeastSquaresLoss` | ✓ | — | ✓ | ✓ | — |
| `LogDetLoss` | — | ◐ *h* | ◐ *i* | — | ◐ *i* |

- *d* — with an ``n\times k`` response; the coefficients are ``p\times k``.
- *e* — a vector, or a matrix with an ``n\times k`` response.
- *f* — a dense vector or strided view with the loss's element type.
- *g* — a dense `Float64` vector or strided view; the loss itself takes `Float64` data.
- *h* — square, with the element type of ``S``; `gradient!` and `prox!` need a separate strided dense output.
- *i* — matrix input; value, gradient and `prox` with a dense output.

## Sparsity penalties

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [`NormL1`](@ref) | ✓ | — | ✓ | ✓ | — | ✓ | ✓ |
| [`ElasticNet`](@ref) | ✓ | — | ✓ | ✓ | — | ✓ | ✓ |
| [`NormL1Nonneg`](@ref) | ✓ | — | ✓ | ✓ | — | ✓ | ✓ |
| [`MCP`](@ref) | ✓ | — | ✓ | ✓ | — | — | ✓ |
| [`SCAD`](@ref) | ✓ | — | ✓ | ✓ | — | — | ✓ |
| [`NormL0`](@ref) | ✓ | — | ✓ | ✓ | — | — | ✓ |
| `NormL0(λ; diagonal = false)` | ◐ *a* | — | ◐ *a* | — | — | — | — |

- *a* — finite level; square strided `Float32` or `Float64` matrices with finite entries,
  symmetric by precondition and under the storage preconditions of its entry; prox takes a
  finite positive scalar step.

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `NormL1`, `ElasticNet`, `NormL1Nonneg` | ✓ | ✓ | ✓ | ✓ | ◐ *b* |
| `MCP`, `SCAD`, `NormL0` | ✓ | ✓ | ✓ | ✓ | ◐ *c* |
| `NormL0(λ; diagonal = false)` | — | ◐ *a* | — | — | — |

- *b* — scalar step; supplied weights must be exactly mirrored (``p^2`` entries in
  linear order); the score is per coordinate only.
- *c* — scalar step.

## Norm and group penalties

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [`NormL2`](@ref) | ✓ | — | ✓ | — | — | ◐ *a* | — |
| [`NormLinf`](@ref) | ✓ | — | ✓ | — | — | ◐ *a* | — |
| [`GroupNormL2`](@ref) | ✓ | — | ✓ | — | ✓ | ◐ *b* | — |
| [`SparseGroupNormL2`](@ref) | ✓ | — | ✓ | — | ✓ | ◐ *b* | — |
| [`GroupNormLinf`](@ref) | ✓ | — | ✓ | — | ✓ | ◐ *b* | — |
| [`GroupMCP`](@ref) | ✓ | — | ✓ | — | ✓ | ◐ *c* | — |
| [`GroupSCAD`](@ref) | ✓ | — | ✓ | — | ✓ | ◐ *c* | — |

- *a* — whole point only.
- *b* — per group and for the whole point; with a `BlockIterate`, its groups must be the
  function's groups, in the same order.
- *c* — zero groups and entries in no group only; no whole-point score.

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `NormL2`, `NormLinf` | ✓ | ✓ | ✓ | ✓ | ◐ *d* |
| the five group penalties | ✓ | ◐ *e* | ✓ | ✓ | — |

- *d* — value only.
- *e* — groups refer to linear (column-major) indices.

## Constraints

For a constraint, prox is the Euclidean projection onto the set.

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [`IndBox`](@ref), [`IndBallLinf`](@ref) | ✓ | — | ✓ | ✓ | — | — | ◐ *a* |
| [`IndNonnegative`](@ref) | ✓ | — | ✓ | ✓ | — | ◐ *b* | ✓ |
| [`IndBallL1`](@ref) | ✓ | — | ✓ | — | — | — | — |
| [`IndBallL2`](@ref) | ✓ | — | ✓ | — | — | — | — |
| [`IndBallL0`](@ref) | ✓ | — | ◐ *c* | — | — | — | — |

- *a* — throws unless every omitted interval contains zero.
- *b* — the screen of a coordinate assumed to be zero only.
- *c* — finite counted entries.

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `IndBox`, `IndBallLinf`, `IndNonnegative` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `IndBallL1`, `IndBallL2` | ✓ | ✓ | ✓ | ✓ | — |
| `IndBallL0` | ◐ *d* | ◐ *d* | ◐ *e* | — | ◐ *e* |

- *d* — `Float32` or `Float64` arrays, not views.
- *e* — the output must be of the same kind as the input.

## Quadratic and linear functions

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [`SqrNormL2`](@ref) | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| [`QuadraticFunction`](@ref) | ✓ | ✓ | ◐ *a* | — | — | ◐ *b* | ✓ |
| [`KronQuadratic`](@ref) | ✓ | ✓ | — | — | — | — | — |
| [`LinearFunction`](@ref) | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| [`ZeroFunction`](@ref) | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |

- *a* — for an indefinite ``A``, a step that fails the computed check on its
  [guarantee page](@ref spec-quadratic) throws `DomainError`; ``A`` must be symmetric
  (not checked).
- *b* — whole point only.

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `SqrNormL2` | ✓ | ✓ | ✓ | ✓ | ◐ *c* |
| `LinearFunction` | ◐ *d* | ◐ *d* | ◐ *d* | ◐ *d* | ◐ *c* |
| `ZeroFunction` | ✓ | ✓ | ✓ | ✓ | ◐ *e* |
| `QuadraticFunction` | ✓ | — | ◐ *f* | ✓ | — |
| `KronQuadratic` | — | ◐ *g* | ◐ *i* | — | ◐ *h* |

- *c* — no whole-point score.
- *d* — the coefficients must have the shape of the input.
- *e* — value, a coordinate score, and `prox` into a `SymmetricSparseIterate` output only.
- *f* — a vector `SparseIterate`.
- *g* — `gradient!` needs a writable strided output.
- *h* — square variable, matching element types, and ``C`` exactly symmetric when given.
- *i* — a matrix `SparseIterate`; value and gradient.

## Scaled functions

`Scaled(f, α)`, written `α * f`, with a finite ``\alpha>0`` provides the operations of
`f` above except group steps. Zero and infinite scales, scaled indicators (which return
the indicator itself) and the remaining exceptions follow the
[scaling rules](@ref scaling-guide). The G0 probes covered `NormL1`,
`LeastSquaresLoss` and `GroupNormL2` as inner functions only.

## Other operations

- `lipschitz(f)` returns a computed bound on the gradient's Lipschitz constant for
  `SqrNormL2`, `QuadraticFunction`, `KronQuadratic`, `LeastSquaresLoss`,
  `LinearFunction`, `ZeroFunction` and their positive multiples. It is `0` for
  `ZeroFunction` and `LinearFunction` and can be `0` for others, so check before dividing. `LogDetLoss` returns `Inf`. The other functions throw.
- There is no Hessian or Hessian-vector product in this version.

## What these tables do not say

- Whether a ProximalSolvers algorithm accepts a function: see its
  [supported problems](https://mlakolar.github.io/ProximalSolvers.jl/dev/support/).
- What happens at extreme or non-finite values: see each function's
  [guarantees](@ref numerical-contracts-guide).
- Preconditions that you must check yourself: see the function's entry.
