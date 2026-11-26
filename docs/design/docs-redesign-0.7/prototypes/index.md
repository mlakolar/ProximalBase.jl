# ProximalBase.jl

**Losses, penalties and constraints for optimization in Julia, with their
gradients and proximal operators.**

ProximalBase is a library of mathematical functions that appear in statistical
estimation and machine learning: least-squares, logistic and Poisson losses; ℓ₁,
group, elastic-net, MCP and SCAD penalties; and constraints such as boxes and
norm balls. Each function is a Julia object that you can evaluate and, where
ProximalBase implements it, differentiate or pass to its proximal operator. Storage
types for sparse, grouped and symmetric variables let an algorithm update only the
coordinates it is working on.

ProximalBase does not fit models: it is the layer an optimization algorithm is built
on. To fit a lasso, group lasso or similar model, see
[ProximalSolvers.jl](@ref with-proximalsolvers).

**Start here:** [install](@ref install) · [quick start](@ref quickstart) ·
[function catalog](@ref function-catalog) · [ProximalSolvers](@ref with-proximalsolvers)

## [Install](@id install)

This manual describes the development version (Julia 1.10 or later):

```julia
using Pkg
Pkg.add(url="https://github.com/mlakolar/ProximalBase.jl", rev="master")
```

`Pkg.add("ProximalBase")` installs the registered release, which is older and has a
different interface; see [Which version should I install?](@ref install-versions).

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

Each entry moved toward zero by up to ``0.5``, and the small one became exactly zero.
The [five-minute quick start](@ref quickstart) continues from here with losses,
gradients and in-place calls.

## [What is in the package](@id documentation-tracks)

| | |
|:--|:--|
| **[Function catalog](@ref function-catalog)** | 30 functions, grouped as losses, penalties, constraints and building blocks, each with its formula and Julia constructor. |
| **[Operations](@ref operation-guide)** | Values, gradients, proximal operators and projections, single-coordinate and group steps, and optimality checks. |
| **[Capability tables](@ref capability-tables)** | Which operations each function implements and which storage it accepts, with the main conditions. |
| **[Iterates](@ref Sparse-iterates)** | `SparseIterate`, `BlockIterate` and `SymmetricSparseIterate` for algorithms that work on part of a variable. Most functions also accept ordinary arrays. |
| **[Guarantees](@ref numerical-contracts-guide)** | The documented promises and known gaps for each function, including rules for extreme, infinite and `NaN` inputs. |

## [Find your starting point](@id Where-to-start)

- **I want to use a loss, penalty or constraint in my estimation code.**
  [Quick start](@ref quickstart) → [function catalog](@ref function-catalog).
- **I am writing an optimization algorithm.**
  [Quick start](@ref quickstart) → [proximal gradient](proximal_gradient.md) →
  [coordinate descent](coordinate_descent.md) → [iterates](@ref Sparse-iterates).
- **I want to add a function or rely on exact numerical behavior.**
  [Writing your own function](extending.md) → [guarantees](@ref numerical-contracts-guide) →
  [developer guide](developer/index.md).

## [ProximalBase and ProximalSolvers](@id with-proximalsolvers)

| If you want to… | Use |
|:--|:--|
| fit a lasso, elastic net, group lasso or similar model | ProximalSolvers.jl: see its [getting-started guide](https://mlakolar.github.io/ProximalSolvers.jl/dev/getting-started/) and [supported problems](https://mlakolar.github.io/ProximalSolvers.jl/dev/support/) |
| evaluate a loss, penalty or constraint, or its gradient or proximal operator | ProximalBase |
| implement or modify an optimization algorithm | ProximalBase |
| add a function | ProximalBase ([extension guide](extending.md)); whether a solver accepts it is up to the solver |

ProximalSolvers chooses step sizes, update order and stopping rules, and calls
ProximalBase for the mathematics. A function that ProximalBase supports is not
automatically supported by every ProximalSolvers algorithm; its supported-problems page
says which are. The two packages are versioned separately: install ProximalSolvers by
its own instructions, in its own environment, rather than adding it to an environment
that holds this development version of ProximalBase.

## [Which version should I install?](@id install-versions)

| Version | Install | Interface | Manual |
|:--|:--|:--|:--|
| development (0.7 in progress) | the command [above](@ref install) | described here | `dev` (this one) |
| 0.5.4, registered | `Pkg.add("ProximalBase")` | older, different | [stable](https://mlakolar.github.io/ProximalBase.jl/stable/) |

The table is updated at each release. [Changes from earlier versions](migration.md)
lists what was renamed, removed and restored.
