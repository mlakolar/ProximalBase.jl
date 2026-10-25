# Recorded oracle validation

Run on macOS arm64 with one Julia thread and one BLAS thread. The recorded
run used pre-squash commit `c12c30a2d4535e5dcfcf96b8ce26d2880704968d` with a
clean working tree. The oracle source and environment are unchanged in the
squashed PR. Production code is unchanged from base commit `b6c949f`.

The [full output](results/julia-1.13.2.log) records all 75 problems and statuses.
The [README](README.md) gives the exact setup and run commands. The documented
setup command was also verified to preserve the tracked project file.

| Item | Observed |
| --- | --- |
| Julia | 1.13.2 |
| Hypatia | 0.11.0 |
| JuMP | 1.32.1 |
| MathOptInterface | 1.54.0 |
| ProximalBase | 0.5.3, checkout at the tested commit |
| RNG | Xoshiro(1042026) |
| Matrix orders | 2, 3, 5 |
| Problems / assertions | 75 / 1206, all passed |
| Reported testset time, excluding module loading | 22.2 seconds |
| Largest absolute objective difference | 1.66e-9 |
| Largest Frobenius solution difference | 4.18e-5 |
| Largest normalized log-det oracle stationarity residual | 1.36e-7 |
| Largest normalized log-det package stationarity residual | 1.77e-15 |
| Largest normalized spectral stationarity residual, either point | 4.90e-5 |

Every accepted solve returned `OPTIMAL`, with primal and dual
`FEASIBLE_POINT`. Three problems required the explicitly
logged retry from 1e-9 to 1e-8: the rotated order-two trace ball and Fantope,
and the rotated order-five eigenvalue box. Rejected initial statuses were
`ALMOST_OPTIMAL` with `NEARLY_FEASIBLE_POINT`; their values were not compared.
The retries used fresh optimizers and the same independent accuracy thresholds.
Hypatia also emitted two `cannot step in centering direction` warnings.
They remain in the raw log; the final accepted solves passed all checks.

During development, uniform stopping tolerances of 1e-10 and 1e-9 produced
nearly optimal statuses on some rotated cases. A uniform tolerance of 1e-8
produced optimal statuses but failed 29 of the tighter point/KKT assertions,
mostly near boundaries. Those runs were rejected. This motivated the explicit
retry policy, which retains the original point/KKT thresholds. Objective
agreement alone was insufficient to accept those runs.

This is numerical evidence for these small real Float64 problems, separate
from the analytic tests and guarantees. No Julia 1.10, GPU, Float32, large or
ill-conditioned conic run is claimed here. Numerical behavior can vary across
solver/library versions and platforms; such failures must be investigated,
not converted to skipped tests or accepted nearly optimal statuses.
