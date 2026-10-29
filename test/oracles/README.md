# Independent conic oracles

Run from the repository root with Julia 1.10 or later:

```sh
julia scripts/instantiate.jl test/oracles
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --check-bounds=yes --project=test/oracles test/oracles/runtests.jl
```

The setup script explicitly develops this checkout, including on Julia 1.10,
which does not use `[sources]`, and preserves `Project.toml`. JuMP and Hypatia are confined to this optional
environment. Neither ordinary CI nor `test/runtests.jl` loads it. Generated
manifests are ignored; record `Pkg.status()` when comparing environments.
On Windows, set the two environment variables before running the command.
The script also sets BLAS to one thread.

The suite uses Float64 matrices of orders 2, 3 and 5 and `Xoshiro(1042026)`.
It includes deterministic diagonal inputs, seeded rotations and covariance
matrices, and feasible inputs near the PSD, simplex and Fantope boundaries.
It varies proximal steps (0.3, 0.7, 1.0, 1.7) and log-det weights (0.5, 1, 2).
All problems have a strictly convex squared Frobenius term, so their proximal
points are unique even at repeated eigenvalues.

The independent model minimizes `f(X) + norm(X-A)^2/(2*gamma)` using a rotated
second-order cone for the squared norm. Both matrix triangles contribute.
For `LogDetLoss`, a scaled log-determinant cone represents `t <= logdet(X)`;
its upper-triangle packing multiplies off-diagonals by `sqrt(2)`. For spectral
primitives, semidefinite constraints represent eigenvalue bounds directly,
with scalar trace constraints where needed. These models do not call the
package's eigendecomposition, clipping or simplex projection routines.

Coverage:

- `LogDetLoss(S, w)`: positive definiteness and smooth first-order residual.
- `IndPSD()`: PSD projection.
- `Spectral(IndBox(-0.3, 0.8))`: two-sided eigenvalue bounds.
- `TracePSD(0.4)`: PSD trace penalty.
- `Spectral(IndSimplex(1))`: density matrices with trace one.
- `Spectral(IndSimplex(1; equality=false))`: PSD trace ball.
- `Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, mass)))`:
  Fantope with trace 1 or 1.5.
- `IndExpPrimal()` and `IndExpDual()`: exponential cone projections on triples.

Every case checks the point from `value_and_prox!` and its returned **function
value**, which excludes the quadratic term. It compares independently evaluated
objectives with both the conic epigraph objective and the package point, and
compares the unique solutions. Spectral cases check primal and dual feasibility,
stationarity and complementary slackness using Hypatia's dual matrices and
trace multipliers, at both the oracle and package points. Log-det stationarity
is evaluated directly by matrix inversion.

Hypatia uses `tol_rel_opt = tol_abs_opt = tol_feas = 1e-9` and at most 200
iterations. If the initial solve returns `ALMOST_OPTIMAL`, the suite logs the
rejected status and retries once with all three solver tolerances set to 1e-8.
Independent comparison thresholds remain unchanged. No values from the rejected
solve enter a comparison. Other failed statuses abort without a retry.
Only `OPTIMAL` with primal and dual `FEASIBLE_POINT` is accepted;
any other status aborts before values are read. Separate comparison tolerances:

| Check | Tolerance |
| --- | --- |
| Objective, absolute and relative | 2e-7 |
| Solution Frobenius distance | 5e-5 times max(1, oracle norm) |
| Spectral feasibility, absolute | 2e-7 |
| Normalized stationarity | 5e-5 |
| Complementarity | 5e-5 times the documented scale in the code |
| Package log-det stationarity | 1e-10 |
| Returned function value, absolute and relative | 1e-10 |

Interior-point point errors can be larger than objective errors near a
boundary. The comparison tolerances reflect this and are intentionally
separate from solver stopping tolerances. Per-case output reports statuses,
objective and solution differences, and first-order residuals. The header
records commit, working-tree status, seed and Julia/JuMP/Hypatia/MOI versions.

The exponential cone checks use `Xoshiro(1712026)` and 35 triples for each
cone, at scales 0.1, 1 and 10. They include interior and face points, both
sides of `s = 0`, `r = 0` and `t = 0`, large root parameters, and 20 random
points. A rotated second-order cone represents the squared distance. The
primal constraint uses `MOI.ExponentialCone()` in `(r,s,t)` order. The dual
constraint maps `(u,v,w)` to `(-v,-u,e*w)` in that cone, independently of the
package's projection algorithm. Objective, solution, feasibility, normalized
stationarity and complementarity use the separate thresholds in the table
above. The conic multiplier supplies the normal vector for stationarity;
primal and polar inequalities are also checked directly on both solutions.
These checks do not cover extreme magnitudes, underflow or Float32; the
ordinary `test/expcone.jl` covers those with manufactured projections and
independent Moreau checks. The historical results linked below predate these
additional cone checks until a new validation run is recorded.

These are independent numerical checks on small, moderately scaled triples and real
symmetric matrices. They do not establish analytic guarantees, accuracy for
ill-conditioned or large matrices, Float32 accuracy, or nonconvex rank
constraints and penalties. The Fantope is a convex relaxation, including at
integer trace. Graphical-lasso/QUIC benchmarks are outside this suite.

See [RESULTS.md](RESULTS.md) for the recorded validation run.
