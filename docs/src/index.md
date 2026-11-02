# ProximalBase.jl

ProximalBase provides building blocks for writing optimization algorithms in
Julia. We focus on problems in high-dimensional statistics, where a smooth
loss is combined with a penalty or a constraint:

```math
\min_x\; F(x) = f(x) + g(x).
```

Here ``f`` may be a least-squares loss or a generalized linear model, while
``g`` may impose sparsity, a group structure or a matrix constraint.
The package provides the mathematical operations used by an optimization
procedure, and sparse iterates for updating entries or groups of entries.

## What the package provides

Coordinate-descent procedures use scalar or group updates, curvature bounds
and checks of the optimality conditions. Proximal methods use gradients and
proximal operators on the full variable. Both use the same function objects.
In particular, a penalty such as [`NormL1`](@ref) provides soft thresholding
for either a single coordinate or the whole array.

| Part | Main operations | Use |
|:-----|:----------------|:----|
| Losses, penalties and constraints | `f(x)`, `gradient!`, `prox!` | Evaluate an objective and construct its updates |
| Coordinate operations | `prox_entry`, `prox_coordinate!`, `coordinate_lipschitz`, `kkt_violation`, `restrict` | Update entries or groups and manage a working set |
| Sparse iterates | `SparseIterate`, `SymmetricSparseIterate`, `AtomIterate` | Store and update active entries or groups |

We leave the optimization loop, stopping rule and choice of working set to the
calling procedure. For complete solvers built on these operations, see
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl).

## Where to start

The [Getting started](@ref) guide solves one lasso problem by coordinate
descent and proximal gradient. It introduces the interface through executable
examples, explains the loss normalization and checks the optimality conditions.
ProximalBase requires Julia 1.10 or later, and the guide includes installation
instructions.

Next, choose the material that matches your procedure:

| Goal | Read |
|:-----|:-----|
| Understand the type hierarchy, traits and who calls whom | [Package structure](@ref architecture) |
| Derive entry or block updates, or construct an active-set procedure | [Coordinate descent](@ref) |
| Understand coordinate indexing and sparse storage | [Sparse iterates](@ref) |
| Derive a proximal step or choose a stopping rule | [The interface](@ref) |
| Choose a loss, penalty or constraint | [Smooth functions](@ref), [Penalties](@ref), [Sets](@ref) |
| Combine functions or add one of your own | [Functions built from functions](@ref), [Writing your own function](@ref) |
| Look up a signature | [Interface reference](@ref), [Function reference](@ref), [Iterate and utility reference](@ref) |

The examples section contains complete programs for regression, graphical
models, matrix estimation and constrained optimization. Each program checks
its result when the documentation is built.

## Related packages

For statistical estimation and inference, see
[HighDimensionalModels.jl](https://github.com/mlakolar/HighDimensionalModels.jl) for
penalized regression,
[CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl)
for Gaussian graphical models, and
[DiffPrecTest.jl](https://github.com/mlakolar/DiffPrecTest.jl) for inference on
differences of precision matrices.

<!-- scenario H -->
