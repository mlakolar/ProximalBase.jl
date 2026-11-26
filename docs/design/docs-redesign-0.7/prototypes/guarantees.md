# [Numerical guarantees](@id guarantees)

The [function entries](@ref function-catalog) describe what each function computes
for ordinary finite inputs. The pages in this section state precisely what is
promised, including for very large or very small values, infinities, `NaN`, zero or
infinite parameters and steps, and each kind of storage. Most users never need them. Read them when an
algorithm meets extreme data, when you write tests that depend on exact results,
or when you contribute a function.

## Rules shared by every function

| Page | Read it when |
|:--|:--|
| [Arrays, precision and aliasing](@ref numerical-usage) | you pass views, mix `Float32` and `Float64`, reuse output buffers or call from several tasks |
| [Range, non-finite values and sparse products](@ref numerics) | values may overflow or underflow, inputs may contain `Inf` or `NaN`, or you multiply sparse iterates |
| [Why floating-point range matters](@ref learn-floating-point) | you want a worked example before reading the range rules |

## Guarantees for each function

**Losses:**
[`LeastSquaresLoss`](@ref spec-leastsquares) ·
[`LogisticLoss`](@ref spec-logisticloss) ·
[`PoissonLoss`](@ref spec-poissonloss) ·
[`SqrtLeastSquaresLoss`](@ref spec-sqrt_leastsquares) ·
[`LogDetLoss`](@ref spec-logdet)

**Sparsity penalties:**
[`NormL1`](@ref spec-norml1) ·
[`ElasticNet`](@ref spec-elasticnet) ·
[`NormL1Nonneg`](@ref spec-norml1nonneg) ·
[`MCP`](@ref mcp-contract) ·
[`SCAD`](@ref spec-scad) ·
[`NormL0`](@ref spec-norml0) ·
[`NormL0(λ; diagonal = false)`](@ref spec-norml0offdiag)

**Norm and group penalties:**
[`NormL2`](@ref spec-norml2) ·
[`NormLinf`](@ref spec-normlinf) ·
[`GroupNormL2`](@ref spec-groupnorml2) ·
[`SparseGroupNormL2`](@ref spec-sparsegroupnorml2) ·
[`GroupNormLinf`](@ref spec-groupnormlinf) ·
[`GroupMCP`](@ref spec-groupmcp) ·
[`GroupSCAD`](@ref spec-groupscad)

**Constraints:**
[`IndBox` and `IndBallLinf`](@ref spec-indbox) ·
[`IndNonnegative`](@ref spec-indnonnegative) ·
[`IndBallL1`](@ref spec-indballl1) ·
[`IndBallL2`](@ref spec-indballl2) ·
[`IndBallL0`](@ref spec-indballl0)

**Quadratic and linear functions:**
[`SqrNormL2`](@ref spec-sqrnorml2) ·
[`QuadraticFunction`](@ref spec-quadratic) ·
[`KronQuadratic`](@ref spec-kronquadratic) ·
[`LinearFunction` and `ZeroFunction`](@ref spec-linear)

**Wrappers and operations:**
[`Scaled`](@ref spec-scaled) ·
[optimality scores (`kkt_violation`)](@ref spec-kkt_violation)

## How to read a guarantee page

Each page has the same sections, from the definition and the supported inputs
to the behavior at extreme values. Statements are marked with one of these words:

| Word on the page | Meaning for you |
|:--|:--|
| guaranteed | Promised. Changing it would be a breaking change, and tests check it. |
| precondition | Your responsibility. It is usually not checked, and violating it can give wrong results without an error. |
| limitation | A known gap, linked to an open issue. |
| non-guarantee | Explicitly not promised. Do not rely on it. |
| characterized | What happens today, recorded so that a change is noticed, but not promised. |
| unclassified | Not yet given one of the meanings above. Treat it like "characterized". |

When a page and the code disagree, the code and its tests decide, and the
disagreement is a bug in the documentation: please
[open an issue](https://github.com/mlakolar/ProximalBase.jl/issues).

## Known limitations

These limitations are open issues. Each owning page states its exact scope.

| Function | Limitation | Owner |
|:--|:--|:--|
| `NormL1`, `NormL1Nonneg` | A threshold computed from finite factors can overflow to `Inf`; an infinite input can then give `NaN` ([#866](https://github.com/mlakolar/ProximalBase.jl/issues/866)) | [NormL1](@ref spec-norml1-limitations) |
| … | One row per open limitation, gathered from the owning pages | … |

How the guarantee pages are written and reviewed is described in the
[developer guide](@ref developer-guide).
