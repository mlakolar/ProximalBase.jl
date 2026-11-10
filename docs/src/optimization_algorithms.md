
# Building optimization algorithms

ProximalBase is an **algorithm-building library**, not a package that
chooses and runs one general-purpose solver. Its mathematical function
objects and their oracles provide the local computations needed by many
optimization methods. You supply the iteration scheme, step-size policy,
stopping test, and—where appropriate—the working set. For implemented
solvers, see [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl).

This section connects the [function operations](operations.md) to
optimization algorithms. The equations describe what an algorithm can
assemble from ProximalBase; they do **not** imply that the package exports
a routine named after every algorithm below.

## Match an algorithm to its oracles

| Method / idea | Primary operations | Next page |
|:--|:--|:--|
| Proximal gradient, projected gradient, FISTA | `gradient!`, `prox!`, `lipschitz`, `gradient_mapping!` | [Proximal gradient methods](proximal_gradient.md) |
| Coordinate descent and active sets | `prox_block`, `prox_block!`, `coordinate_lipschitz`, `kkt_violation`, `restrict` | [Coordinate descent](coordinate_descent.md) |
| Group or block coordinate descent | `prox_block!` with compact buffers, then public `AtomIterate` writes | [Coordinate oracles](iterate_oracles.md) |
| Conditional gradient / Frank–Wolfe | `lmo`, `lmo!`, sometimes `lmo_atom` | [Operations on functions](operations.md) |
| Newton or second-order model methods | `hessian_vec!`, `quadratic_model`, `quadratic_model!` | [Operations on functions](operations.md) |
| Local linear approximation | `lla_weight`, `lla_weights!` | [Penalties](penalties.md) |

A method's guarantees depend on mathematical conditions (convexity,
smoothness, step-size choice, etc.) that the function objects do not
enforce for the solver. Traits describe available function capabilities;
the solver remains responsible for algorithmic validation and iteration.

## Function, iterate, and solver each have a job

- A function object defines the loss, penalty or feasible set and supplies
  its supported mathematical operations.
- An iterate represents the optimization variable, sometimes with an
  explicit sparse working set. Its coordinates need not coincide with
  scalar entries.
- A solver chooses which operations to request, in what order, and when
  it has enough evidence to stop.

For example, a proximal-gradient solver can use a dense `Vector` and
call the same `NormL1` proximal operator used by a coordinate-descent
solver's entrywise update. An active-set solver may instead use
`SparseIterate` or `AtomIterate` to avoid visiting every coordinate
on each sweep. Storage is a performance and index-space decision, not
a change in the mathematical penalty.

## Start with an executable example

[Getting started](getting_started.md) constructs both coordinate-descent
and proximal-gradient loops for the *same* lasso objective and verifies
their answers. Further complete programs are under [Examples](examples/01_lasso_three_ways.md).

For mathematical derivations, read
[Proximal gradient methods](proximal_gradient.md) and
[Coordinate descent](coordinate_descent.md).
For public signatures and their contract boundaries, use
[Operations on functions](operations.md).
