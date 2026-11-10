
# Function library: choose a mathematical object

ProximalBase includes losses, regularizers, constraint indicators and
functions composed from other functions. They are **mathematical objects**
with a common [`AbstractFunction`](@ref) interface. Choose one by the
objective you want to express; then consult
[Operations on functions](operations.md) to see what computations your
algorithm can request.

The detailed pages below retain their formulas, proximal mappings,
exceptions and algorithmic notes. This catalog is an entry point, **not**
a complete list of supported signatures or a guarantee of every oracle
for every function and argument type.

## Smooth functions and losses

| Family | Selected functions | Typical use |
|:--|:--|:--|
| Elementary functions | [`ZeroFunction`](@ref), [`LinearFunction`](@ref), [`SqrNormL2`](@ref) | Constants, linear objectives, squared norms |
| Quadratic models | [`QuadraticFunction`](@ref), [`KronQuadratic`](@ref) | Quadratic objectives on vectors or matrices |
| Statistical losses | [`LeastSquaresLoss`](@ref), [`LogisticLoss`](@ref), [`PoissonLoss`](@ref), [`HuberLoss`](@ref) | Regression, classification and robust estimation |
| Matrix and other losses | [`LogDetLoss`](@ref), [`BernoulliLoss`](@ref) | Likelihood and matrix estimation |

Read [Smooth functions](smooth.md) for mathematical definitions,
curvature and example uses. Some functions that provide a gradient also
have a proximal map; check the capability and operand domain rather
than relying solely on this classification.

## Penalties and regularizers

| Family | Selected functions | Typical use |
|:--|:--|:--|
| Entrywise sparse | [`NormL1`](@ref), [`ElasticNet`](@ref), [`NormL0`](@ref) | Sparsity and shrinkage |
| Group sparse | [`GroupNormL2`](@ref), [`GroupNormLinf`](@ref), [`SparseGroupNormL2`](@ref) | Group or sparse-group selection |
| Nonconvex sparse | [`MCP`](@ref), [`SCAD`](@ref), [`GroupMCP`](@ref), [`GroupSCAD`](@ref) | Bias reduction and nonconvex regularization |
| Structured vectors | [`TotalVariation1D`](@ref), [`FusedLasso`](@ref), [`SLOPE`](@ref), [`SumLargest`](@ref) | Ordered, fused and sorted penalties |
| Matrix and spectral | [`NuclearNorm`](@ref), [`TracePSD`](@ref), [`SingularValue`](@ref) | Low rank and eigenvalue/singular-value models |
| Other penalties and losses | [`NormL2`](@ref), [`NormLinf`](@ref), [`NegEntropy`](@ref), [`KLDivergence`](@ref), [`CheckLoss`](@ref) | Norms, entropy and robust objectives |

See [Penalties](penalties.md) for formulas and per-family proximal
operators, including which penalties are entrywise separable and which
have group or spectral structure.

## Constraints as indicator functions

An indicator function returns `0` on its feasible set and `Inf` outside.
For supported indicator functions, the proximal operation is a projection.

| Family | Selected functions |
|:--|:--|
| Coordinate and norm bounds | [`IndBox`](@ref), [`IndNonnegative`](@ref), [`IndBallL1`](@ref), [`IndBallL2`](@ref), [`IndSimplex`](@ref) |
| Linear constraints | [`IndHyperplane`](@ref), [`IndHalfspace`](@ref), [`IndHyperslab`](@ref), [`IndAffine`](@ref), [`IndPolyhedron`](@ref) |
| Cones and matrix constraints | [`IndSOC`](@ref), [`IndPSD`](@ref), [`IndBallRank`](@ref), [`IndStiefel`](@ref) |
| Structure, intersections and cardinality | [`IndPattern`](@ref), [`IndToeplitz`](@ref), [`IndIntersection`](@ref), [`IndBallL0`](@ref), [`IndBallGroupL0`](@ref) |

See [Sets](sets.md) for the full supported constraint catalog, formulas,
projection algorithms and numerical caveats. Some projections are computed
iteratively; a returned result or value of zero does not necessarily
certify exact feasibility.

## Transformations and compositions

| Construction | Purpose |
|:--|:--|
| [`Scaled`](@ref), [`Tilt`](@ref), [`Translate`](@ref), [`Regularize`](@ref) | Change scaling, location or curvature |
| [`Conjugate`](@ref), [`MoreauEnvelope`](@ref) | Convex conjugation and smoothing |
| [`Sum`](@ref), [`SeparableSum`](@ref), [`PointwiseMinimum`](@ref) | Combine functions on one variable or blocks |
| [`Constrained`](@ref), [`Precompose`](@ref), [`LinearImage`](@ref) | Restrict a domain or transform an argument |
| [`Spectral`](@ref), [`SqrDistL2`](@ref), [`DistL2`](@ref) | Spectral lifts and distance penalties |
| [`SliceSum`](@ref), [`FiberSum`](@ref) | Assemble functions across slices and fibers |

Read [Functions built from functions](calculus.md) for the exact
algebraic identities and which oracles propagate. Wrapper operations
are not automatic for every combination of functions and types.

## A small example

```@example function_library_intro
using ProximalBase

g = NormL1(0.5)
x = [2.0, -1.0]
@assert isproximable(g)
@assert prox(g, x, 1.0) ≈ [1.5, -0.5]
(family = typeof(g), value = g(x), proximal_point = prox(g, x, 1.0))
```

Next see [Function objects and capabilities](function_objects.md) for
the conceptual model, [Operations on functions](operations.md) for
the callable methods, and [Examples](examples/01_lasso_three_ways.md)
for full optimization workflows.
