# ProximalBase.jl

**Losses, penalties and constraints for optimization in Julia, with their
gradients and proximal operators.**

ProximalBase is a library of mathematical functions that appear in statistical
estimation and machine learning: least-squares, logistic and Poisson losses; ℓ₁,
group, elastic-net, MCP and SCAD penalties; and constraints such as boxes and
norm balls. Each function is a Julia object that can be evaluated and, where the
mathematics allows, differentiated or passed to its proximal operator. Storage
types for sparse, grouped and symmetric variables let an algorithm update only
the coordinates it is working on.

ProximalBase does not fit models. It is the layer an optimization algorithm is
built on. To fit a lasso, group lasso or similar model directly, use
[ProximalSolvers.jl](@ref with-proximalsolvers), which is built on ProximalBase.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/mlakolar/ProximalBase.jl", rev="master")
```

This manual describes the development version. The registered release is older
and has a different interface; see [Which version should I install?](@ref install-versions).

## Thirty seconds of ProximalBase

The ℓ₁ penalty ``g(x)=0.5\,\|x\|_1`` is one line of Julia:

```@example home
using ProximalBase

g = NormL1(0.5)           # g(x) = 0.5 * sum(abs, x)
x = [2.0, -1.0, 0.25]

g(x)                      # value: 1.625
```

```@example home
prox(g, x, 1.0)           # proximal step of size 1: soft thresholding
```

Each entry moved toward zero by ``0.5``, and the small one became exactly zero.
The [five-minute quick start](@ref quickstart) continues from here with losses,
gradients and in-place calls.

## What is in the package

| | |
|:--|:--|
| **[Function catalog](@ref function-catalog)** | About 30 functions, grouped as losses, penalties, constraints and wrappers, each with its formula and Julia constructor. |
| **[Operations](@ref operation-guide)** | Values, gradients, proximal operators and projections, single-coordinate and group steps, and optimality checks. |
| **[Capability tables](@ref capability-tables)** | Which operation and which storage each function supports, with the restrictions. |
| **[Iterates](@ref Sparse-iterates)** | `SparseIterate`, `BlockIterate` and `SymmetricSparseIterate` for working-set algorithms. Ordinary arrays work everywhere. |
| **[Guarantees](@ref guarantees)** | What results are promised for extreme, infinite and `NaN` inputs, for each function. |

## Find your starting point

- **I want to use a loss, penalty or constraint in my estimation code.**
  [Quick start](@ref quickstart) → [function catalog](@ref function-catalog).
- **I am writing an optimization algorithm.**
  [Quick start](@ref quickstart) → [proximal gradient](@ref proximal-gradient-tutorial) →
  [coordinate descent](@ref coordinate-descent-tutorial) → [iterates](@ref Sparse-iterates).
- **I want to add a function or rely on exact numerical behavior.**
  [Writing your own function](@ref extending) → [guarantees](@ref guarantees) →
  [developer guide](@ref developer-guide).

## [ProximalBase and ProximalSolvers](@id with-proximalsolvers)

| If you want to… | Use |
|:--|:--|
| fit a lasso, elastic net, group lasso or similar model | [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl) |
| evaluate a loss, penalty or constraint, or its gradient or proximal operator | ProximalBase |
| implement or modify an optimization algorithm | ProximalBase |
| add a new penalty that ProximalSolvers can use | ProximalBase ([extension guide](@ref extending)) |

ProximalSolvers chooses step sizes, update order and stopping rules, and calls
ProximalBase for the mathematics. A function that ProximalBase supports is not
automatically supported by every ProximalSolvers algorithm; check the solver's
own documentation.

## [Which version should I install?](@id install-versions)

| Version | Status | Documentation |
|:--|:--|:--|
| development (`master`, 0.7 in progress) | the interface described here | this manual (`dev`) |
| 0.5.4, registered | older interface, different function set | [stable](https://mlakolar.github.io/ProximalBase.jl/stable/) |

[Changes from earlier versions](@ref migration) lists what was renamed, removed
and restored.
