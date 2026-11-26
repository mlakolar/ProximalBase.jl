# [Capability tables](@id capability-tables)

Find what a function implements. Each role has two tables: the **operations**, and
the kinds of **input** that value and prox accept.

| Symbol | Meaning |
|:--|:--|
| ✓ | implemented for the function's ordinary inputs; the function's entry gives the conditions (shapes, steps, preconditions) |
| ◐ *a* | implemented with the main restriction in note *a* below the table; the entry gives the complete rules |
| — | not implemented: the call throws, or the docstring says it is not supported |
| ? | not yet verified |

On a phone, the tables scroll sideways.

*Prototype note, not for publication: only the cells listed in the
[verification record](../verification/README.md) rest on reproducible scripts. All other
cells are provisional, taken from an exploratory probe run and the docstrings, and G4
establishes each one before publication.*

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
kind as the input is accepted. The shared precision baseline is homogeneous `Float64`
or `Float32` ([precision rules](@ref numerical-usage)); some families require or accept
other types (for example, `PoissonLoss` takes `Float64` data). Family-specific types,
`Symmetric` matrix wrappers and mixed precision are not separate columns: some
restrictions appear in the cell notes, and the others, where they are documented, are
on the function's entry or guarantee page.

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
- *c* — a scalar step; see the entry for the step and input rules.

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
- *g* — the loss is built from a `Vector{Float64}` and a `Matrix{Float64}`; the coefficients are a one-based `StridedVector{Float64}`.
- *h* — the size and element type of ``S``; mutating calls need a writable strided output (see the entry).
- *i* — matrix input; value, gradient and `prox` with a dense output (see the entry).

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
- *b* — per group and for the whole point. With a `BlockIterate`, the per-group score
  needs as many groups as the function, with the selected group listing the same indices
  in the same order; the whole-point score needs every group to match.
- *c* — zero groups and entries in no group only, with the per-group layout rule of *b*;
  no whole-point score.

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `NormL2`, `NormLinf` | ✓ | ✓ | ✓ | ✓ | ◐ *d* |
| the five group penalties | ✓ | ◐ *e* | ✓ | ✓ | — |

- *d* — value only.
- *e* — groups refer to linear (column-major) indices.

## Constraints

For a constraint, prox at a supported step is, in exact arithmetic, the Euclidean projection onto the set (the entry gives the rules for structured inputs and non-finite values).

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [`IndBox`](@ref), [`IndBallLinf`](@ref) | ✓ | — | ✓ | ✓ | — | — | ◐ *a* |
| [`IndNonnegative`](@ref) | ✓ | — | ✓ | ✓ | — | ◐ *b* | ✓ |
| [`IndBallL1`](@ref) | ✓ | — | ✓ | — | — | — | — |
| [`IndBallL2`](@ref) | ✓ | — | ✓ | — | — | — | — |
| [`IndBallL0`](@ref) | ✓ | — | ◐ *c* | — | — | — | — |

- *a* — throws unless the omitted intervals contain zero (scalar bounds must contain zero).
- *b* — the screen of a coordinate assumed to be zero only.
- *c* — finite counted entries.

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `IndBox`, `IndBallLinf`, `IndNonnegative` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `IndBallL1`, `IndBallL2` | ✓ | ✓ | ✓ | ✓ | — |
| `IndBallL0` | ◐ *d* | ◐ *d* | ◐ *e* | — | ◐ *e* |

- *d* — `Float32` or `Float64` arrays, not views. A `Symmetric` input counts pairs of
  entries and gives a dense output ([`IndBallL0`](@ref)).
- *e* — the output must be of the same kind as the input; symmetric storage counts pairs.

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

- *c* — centers or coefficients exactly mirrored ([symmetric parameters](@ref symmetric-parameter-guide)); no whole-point score.
- *d* — the coefficients must have the shape of the input.
- *e* — value, a coordinate score, and `prox` into a `SymmetricSparseIterate` output only.
- *f* — a vector `SparseIterate`.
- *g* — `gradient!` needs a writable strided output, for every input kind.
- *h* — square variable, matching element types, and ``C`` exactly symmetric when given.
- *i* — a matrix `SparseIterate`; value and gradient.

## Scaled functions

For `Scaled(f, α)`, written `α * f`, with a finite ``\alpha>0``, the pilot scripts check
the value of `NormL1`, the value, gradient and `lipschitz` of `LeastSquaresLoss`, and the
value and prox of `GroupNormL2`, whose group steps are not available under `Scaled`. Other
functions, operations and scales, including scaled indicators (which return the
indicator itself), follow the [scaling rules](@ref scaling-guide). One known exception is
recorded in the audit: the allocating `prox` of a scaled `LogDetLoss` on a matrix
`SparseIterate` throws.

## Other operations

- `lipschitz(f)` evaluates in floating point, subject to rounding and range limits, a
  formula for the gradient's Lipschitz constant ``L`` or a bound on it, for `SqrNormL2`,
  `QuadraticFunction`, `KronQuadratic`, `LeastSquaresLoss`, `LinearFunction`,
  `ZeroFunction` and their positive multiples. It is `0` for `ZeroFunction` and
  `LinearFunction` and can be `0` for others: check for a positive finite ``L`` before
  considering a step ``1/L``, as in the [proximal-gradient tutorial](proximal_gradient.md).
  `LogDetLoss` returns `Inf`, and so can its positive multiples; the other unscaled
  functions throw.
- There is no Hessian or Hessian-vector product in this version.

## What these tables do not say

- Whether a ProximalSolvers algorithm accepts a function: see its
  [supported problems](https://mlakolar.github.io/ProximalSolvers.jl/dev/support/).
- What happens at extreme or non-finite values: see each function's
  [guarantees](@ref numerical-contracts-guide).
- Preconditions that you must check yourself: see the function's entry.
