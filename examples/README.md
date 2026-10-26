# Examples

Each script is a short, complete program that solves one problem with
ProximalBase. Each writes its own optimization loop, so the examples show how
the pieces fit together. Run one from the repository root with

```sh
julia --project=. examples/01_lasso_three_ways.jl
```

or with ProximalBase installed in your environment. They need only ProximalBase
and the standard library.

Every script checks its result, using optimality conditions, feasibility,
agreement between procedures, or properties of the seeded example. These are
numerical checks for the displayed problem and tolerances. In particular, a
nonconvex procedure's checks do not establish global optimality. The test suite
runs all scripts.

| Script | Problem | What it shows |
|:-------|:--------|:--------------|
| [`01_lasso_three_ways.jl`](01_lasso_three_ways.jl) | the lasso | ISTA, FISTA and ADMM from the same two functions: `LeastSquaresLoss` is smooth and proximable; `lipschitz`, `value_and_prox!`, `gradient_mapping` |
| [`02_lasso_path_coordinate_descent.jl`](02_lasso_path_coordinate_descent.jl) | a lasso path | coordinate descent with `prox_entry` on a `SparseIterate`; warm starts, an active set, `kkt_violation` |
| [`03_nonconvex_penalties.jl`](03_nonconvex_penalties.jl) | sparse regression | the thresholding functions of `NormL1`, `MCP`, `SCAD` and `NormL0`; the bias of the lasso against MCP and SCAD |
| [`04_group_lasso.jl`](04_group_lasso.jl) | the group lasso | `GroupNormL2` with groups of unequal sizes, by FISTA and by block coordinate descent on an `AtomIterate` |
| [`05_logistic_regression.jl`](05_logistic_regression.jl) | sparse logistic regression | `LogisticLoss`, an unpenalized intercept through the weights of `NormL1`, FISTA with a backtracking line search |
| [`06_robust_and_count_regression.jl`](06_robust_and_count_regression.jl) | robust and count regression | `HuberLoss` against least squares with outliers; `PoissonLoss` with an exposure offset |
| [`07_constrained_least_squares.jl`](07_constrained_least_squares.jl) | constrained least squares | projected gradient onto `IndNonnegative`, `IndSimplex`, `IndBallL1`, and `IndBallL0` (iterative hard thresholding) |
| [`08_graphical_lasso.jl`](08_graphical_lasso.jl) | the graphical lasso | ADMM with `LogDetLoss` and a weighted `NormL1` that leaves the diagonal free; stopping rules |
| [`09_psd_and_correlation.jl`](09_psd_and_correlation.jl) | covariance matrices | `IndPSD`, the nearest correlation matrix by Dykstra's algorithm, `IndPSD(rank = r)`, `TracePSD`, hard thresholding into a `SymmetricSparseIterate` |
| [`10_matrix_completion.jl`](10_matrix_completion.jl) | matrix completion | soft-impute with `NuclearNorm` along a path of λ; hard-impute with `IndBallRank` |
| [`11_sparse_plus_low_rank.jl`](11_sparse_plus_low_rank.jl) | robust PCA | `SeparableSum(NormL1, NuclearNorm)` on a pair of matrices |
| [`12_multitask_regression.jl`](12_multitask_regression.jl) | multitask regression | `LeastSquaresLoss` with a matrix response; `GroupNormL2` on the rows of the coefficient matrix |
| [`13_differential_network.jl`](13_differential_network.jl) | a differential network | the D-trace loss as `KronQuadratic` on a `SymmetricSparseIterate`; its coordinates and `tril_cartesian` |
| [`14_active_set_lasso.jl`](14_active_set_lasso.jl) | the lasso with p ≫ n | a working-set solver with `restrict` and `kkt_violation` |
| [`15_calculus_and_custom_functions.jl`](15_calculus_and_custom_functions.jl) | building functions | `Scaled`, `Translate`, `Tilt`, `Conjugate` (Moreau's identity), `Sum`, `SeparableSum`, and a `SeparableFunction` of your own |
| [`16_constrained_penalties.jl`](16_constrained_penalties.jl) | constrained penalties | `Constrained`: the lasso on coefficients that sum to zero for compositional data, the lasso on a box by coordinate descent, the capped simplex |
| [`17_sparse_pca_fantope.jl`](17_sparse_pca_fantope.jl) | sparse principal components | `Spectral` of a constrained function: the Fantope; ADMM with `Tilt` for the linear term, against plain PCA |
| [`18_ordered_penalties.jl`](18_ordered_penalties.jl) | piecewise-constant and monotone fits, SLOPE | `TotalVariation1D` and `FusedLasso` on a signal with jumps, `IndIsotonic`, `SLOPE` against the lasso on correlated predictors |

For complete solvers built on these pieces, including coordinate descent with
penalty paths, see [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl).
