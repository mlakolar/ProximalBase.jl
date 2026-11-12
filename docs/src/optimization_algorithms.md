# Building optimization algorithms

*Assumes: gradient descent and basic KKT conditions · You will learn: which ProximalBase operations a solver combines and what the solver must supply.*

ProximalBase supplies mathematical operations and storage. A caller chooses
an update schedule, step size, residual/cache lifecycle, working set and stopping
criterion. A [working set](glossary.md#Working-set) is a chosen subset
of coordinates or blocks to visit. [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl)
provides optimization algorithms; the loops here explain how such algorithms
use ProximalBase's mathematical operations.

| Method | ProximalBase operations | Caller responsibility |
|:--|:--|:--|
| Proximal gradient | `gradient!`, `prox!`, `lipschitz` | Step, iteration, optional acceleration and stationarity test |
| Scalar coordinate descent | `prox_block`, `kkt_violation`, `restrict` | Coordinate curvature, trial, write and residual update |
| Block coordinate method | Custom `prox_block!`; `BlockIterate` views | Block decomposition/layout, curvature and cache updates |
| Working sets | `activate!`, `activecoordinates`, `dropzeros!` | Screening policy and verification outside the working set |

For ``F=f+g``, smooth ``f`` and
[proper closed convex](glossary.md#Proper-closed-convex-function) ``g``, proximal gradient
uses a quadratic upper model of ``f`` and the exact prox of ``g``. Coordinate
methods use an upper model on one entry/block. Their guarantees depend on
valid bounds and update schedules, which the function objects do not verify.

The [proximal-gradient page](proximal_gradient.md) derives the full step.
The [coordinate page](coordinate_descent.md) derives a scalar lasso update
and KKT checks. The [executable examples](examples/proximal_gradient.md) use
small deterministic fixtures, with independently derived answers.
