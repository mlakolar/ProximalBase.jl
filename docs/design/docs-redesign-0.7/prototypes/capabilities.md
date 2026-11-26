# [Capability tables](@id capability-tables)

Find what a function supports. Each role has two tables: the **operations** a
function provides, and the **storage** it accepts.

| Symbol | Meaning |
|:--|:--|
| ✓ | supported |
| ◐ | supported with the condition in the numbered note below the table |
| — | not supported (the call throws an error) |

*Prototype note: the cells come from the G0 capability audit (probes on Julia 1.13.4,
compared with independent formulas or with the dense-array result). G4 confirms each
cell against the test suite before the table is published.*

**Operations**, as columns:

| Column | Call |
|:--|:--|
| value | `f(x)` |
| grad | `gradient`, `gradient!`, `value_and_gradient(!)` |
| prox | `prox`, `prox!`, `value_and_prox!` |
| coord | `prox_block(f, xᵢ, i, γ)`: one entry |
| group | `prox_block(f, u, k, γ)`: the entries `u` of group `k` |
| score | `kkt_violation`, the optimality score (a coordinate screen, a coordinate or group score, the whole point) |
| restrict | `restrict(f, S)` |

**Storage**, as columns: dense vectors and matrices (strided views included),
*sparse* for [`SparseIterate`](@ref), *block* for [`BlockIterate`](@ref) and
*symmetric* for [`SymmetricSparseIterate`](@ref).
Two rules hold for every function and are not repeated in the tables:

- A `Symmetric` matrix is accepted wherever a matrix is; `prox` returns a dense
  `Matrix`, and a `Symmetric` output is not accepted.
- The supported precision is `Float64` or `Float32`, used for data, parameters, step
  and output alike. Other element types and mixed precision may work but are not
  guaranteed ([precision rules](@ref numerical-usage)).

## Losses

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [`LeastSquaresLoss`](@ref entry-leastsquaresloss) | ✓ | ✓ | ✓ | — | — | ◐¹ | ◐² |
| `LogisticLoss` | ✓ | ✓ | — | — | — | — | — |
| `PoissonLoss` | ✓ | ✓ | — | — | — | — | — |
| `SqrtLeastSquaresLoss` | ✓ | — | — | — | — | — | — |
| `LogDetLoss` | ✓ | ✓ | ✓ | — | — | — | — |

1. Whole point only.
2. Vector response only.

`LeastSquaresLoss`, `LogisticLoss`, `PoissonLoss` and `SqrtLeastSquaresLoss` also
evaluate from a predictor ``X\beta`` that you already hold
([details](@ref Least-squares-at-a-predictor-or-residual-that-the-caller-holds)).

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `LeastSquaresLoss` | ✓ | ◐³ | ✓ | ✓ | — |
| `LogisticLoss` | ◐⁴ | — | — | — | — |
| `PoissonLoss` | ◐⁵ | — | — | — | — |
| `SqrtLeastSquaresLoss` | ✓ | — | ✓ | ✓ | — |
| `LogDetLoss` | — | ✓ | ◐⁶ | — | ◐⁶ |

3. With an ``n\times k`` response; the coefficients are ``p\times k``.
4. A dense vector, or a strided view, with the loss's element type.
5. A dense vector, or a strided view, of `Float64`.
6. Value, gradient and `prox` with a dense result.

## Sparsity penalties

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [`NormL1`](@ref entry-norml1) | ✓ | — | ✓ | ✓ | — | ✓ | ✓ |
| `ElasticNet` | ✓ | — | ✓ | ✓ | — | ✓ | ✓ |
| `NormL1Nonneg` | ✓ | — | ✓ | ✓ | — | ✓ | ✓ |
| `MCP` | ✓ | — | ✓ | ✓ | — | — | ✓ |
| `SCAD` | ✓ | — | ✓ | ✓ | — | — | ✓ |
| `NormL0` | ✓ | — | ✓ | ✓ | — | — | ✓ |
| `NormL0(λ; diagonal = false)` | ✓ | — | ✓ | — | — | — | — |

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `NormL1`, `ElasticNet`, `NormL1Nonneg` | ✓ | ✓ | ✓ | ✓ | ◐¹ |
| `MCP`, `SCAD`, `NormL0` | ✓ | ✓ | ✓ | ✓ | ◐² |
| `NormL0(λ; diagonal = false)` | — | ◐³ | — | — | — |

1. Scalar step; weights form a symmetric ``p\times p`` array; optimality per coordinate only.
2. Scalar step.
3. Square `Float32` or `Float64` matrices with finite entries.

## Norm and group penalties

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `NormL2` | ✓ | — | ✓ | — | — | ◐¹ | — |
| `NormLinf` | ✓ | — | ✓ | — | — | ◐¹ | — |
| [`GroupNormL2`](@ref entry-groupnorml2) | ✓ | — | ✓ | — | ✓ | ◐² | — |
| `SparseGroupNormL2` | ✓ | — | ✓ | — | ✓ | ◐² | — |
| `GroupNormLinf` | ✓ | — | ✓ | — | ✓ | ◐² | — |
| `GroupMCP` | ✓ | — | ✓ | — | ✓ | ◐³ | — |
| `GroupSCAD` | ✓ | — | ✓ | — | ✓ | ◐³ | — |

1. Whole point only.
2. Per group and whole point.
3. Zero groups only.

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `NormL2`, `NormLinf` | ✓ | ✓ | ✓ | ✓ | ◐⁴ |
| the five group penalties | ✓ | ◐⁵ | ✓ | ✓ | — |

4. Value only.
5. Groups refer to linear (column-major) indices.

## Constraints

For a constraint, prox is the Euclidean projection and does not depend on the step.

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `IndBox`, `IndBallLinf` | ✓ | — | ✓ | ✓ | — | — | ✓ |
| `IndNonnegative` | ✓ | — | ✓ | ✓ | — | ◐¹ | ✓ |
| [`IndBallL1`](@ref entry-indballl1) | ✓ | — | ✓ | — | — | — | — |
| `IndBallL2` | ✓ | — | ✓ | — | — | — | — |
| `IndBallL0` | ✓ | — | ✓ | — | — | — | — |

1. The screen of a coordinate assumed to be zero only.

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `IndBox`, `IndBallLinf`, `IndNonnegative` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `IndBallL1`, `IndBallL2` | ✓ | ✓ | ✓ | ✓ | — |
| `IndBallL0` | ◐² | ◐² | ◐³ | — | ◐³ |

2. `Float32` or `Float64` arrays, not views.
3. The output must be of the same kind as the input.

## Quadratic, linear and scaled functions

| Function | value | grad | prox | coord | group | score | restrict |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `SqrNormL2` | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| `QuadraticFunction` | ✓ | ✓ | ◐¹ | — | — | ◐² | ✓ |
| `KronQuadratic` | ✓ | ✓ | — | — | — | — | — |
| `LinearFunction` | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| `ZeroFunction` | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| `Scaled(f, α)`, `α * f` | as `f` | as `f` | as `f` | as `f` | — | as `f` | as `f` |

1. For an indefinite ``A``, only steps with ``1+\gamma\lambda_{\min}(A)>0``; otherwise a `DomainError`.
2. Whole point only.

| Function | vector | matrix | sparse | block | symmetric |
|:--|:-:|:-:|:-:|:-:|:-:|
| `SqrNormL2`, `LinearFunction` | ✓ | ✓ | ✓ | ✓ | ◐³ |
| `ZeroFunction` | ✓ | ✓ | ✓ | ✓ | ◐⁴ |
| `QuadraticFunction` | ✓ | — | ✓ | ✓ | — |
| `KronQuadratic` | — | ✓ | ◐⁵ | — | ✓ |
| `Scaled(f, α)` | as `f` | as `f` | as `f` | as `f` | as `f` |

3. No whole-point optimality score.
4. Value, a coordinate optimality score, and `prox` into a `SymmetricSparseIterate` output only.
5. Matrix `SparseIterate`.

## What these tables do not say

- Whether a ProximalSolvers algorithm accepts a function: see the
  [ProximalSolvers documentation](https://github.com/mlakolar/ProximalSolvers.jl).
- What happens at extreme or non-finite values: see each function's
  [guarantees](@ref guarantees).
- Which ✓ comes with a precondition you must check yourself (for example, symmetric
  weights on a `SymmetricSparseIterate`): see the function's entry.
