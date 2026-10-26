# Public API inventory

Generated from the loaded package. Julia's default `names` listing suppresses
deprecated bindings; those bindings are inventoried separately below. Legacy
operations without independent docstrings are covered by the maintained
migration guide. `nnz` and `dropzeros!` extend the SparseArrays interface.

| Export | Kind | Docstring |
| --- | --- | --- |
| `A_mul_B_group!` | function | compatibility or module binding |
| `A_mul_B_row` | function | compatibility or module binding |
| `A_mul_UUt_mul_B` | function | compatibility or module binding |
| `A_mul_UUt_mul_B_rc` | function | compatibility or module binding |
| `A_mul_X_mul_B` | function | compatibility or module binding |
| `A_mul_X_mul_B_rc` | function | compatibility or module binding |
| `AbstractFunction` | type or type alias | present |
| `At_mul_B_group!` | function | compatibility or module binding |
| `At_mul_B_row` | function | compatibility or module binding |
| `AtomIterate` | type or type alias | present |
| `Berhu` | type or type alias | present |
| `CappedL1` | type or type alias | present |
| `CheckLoss` | type or type alias | present |
| `Conjugate` | type or type alias | present |
| `Constrained` | function | present |
| `ElasticNet` | type or type alias | present |
| `EpsInsensitiveLoss` | type or type alias | present |
| `FiberSum` | type or type alias | present |
| `FusedLasso` | type or type alias | present |
| `GLMLoss` | type or type alias | present |
| `GroupMCP` | type or type alias | present |
| `GroupNormL2` | type or type alias | present |
| `GroupNormLinf` | type or type alias | present |
| `GroupSCAD` | type or type alias | present |
| `HingeLoss` | type or type alias | present |
| `HuberLoss` | type or type alias | present |
| `IndAffine` | type or type alias | present |
| `IndBallGroupL0` | type or type alias | present |
| `IndBallL0` | type or type alias | present |
| `IndBallL1` | type or type alias | present |
| `IndBallL2` | type or type alias | present |
| `IndBallLinf` | function | present |
| `IndBallRank` | type or type alias | present |
| `IndBanded` | type or type alias | present |
| `IndBox` | type or type alias | present |
| `IndHalfspace` | type or type alias | present |
| `IndHyperplane` | type or type alias | present |
| `IndIntersection` | type or type alias | present |
| `IndIsotonic` | type or type alias | present |
| `IndNonnegative` | type or type alias | present |
| `IndPSD` | type or type alias | present |
| `IndPattern` | type or type alias | present |
| `IndSimplex` | type or type alias | present |
| `IndSphereL2` | type or type alias | present |
| `IndToeplitz` | type or type alias | present |
| `KLDivergence` | type or type alias | present |
| `KronQuadratic` | type or type alias | present |
| `L2Loss` | function | compatibility or module binding |
| `LeastSquaresLoss` | type or type alias | present |
| `LinearFunction` | type or type alias | present |
| `LogDetLoss` | type or type alias | present |
| `LogPenalty` | type or type alias | present |
| `LogisticLoss` | type or type alias | present |
| `MCP` | type or type alias | present |
| `MoreauEnvelope` | type or type alias | present |
| `NegEntropy` | type or type alias | present |
| `NegLog` | type or type alias | present |
| `NormL0` | type or type alias | present |
| `NormL1` | type or type alias | present |
| `NormL1Fused` | type or type alias | present |
| `NormL1Nonneg` | type or type alias | present |
| `NormL2` | type or type alias | present |
| `NormLinf` | type or type alias | present |
| `NormLq` | type or type alias | present |
| `NuclearNorm` | type or type alias | present |
| `PoissonLoss` | type or type alias | present |
| `Precompose` | type or type alias | present |
| `ProxL2Sq` | function | compatibility or module binding |
| `ProximalBase` | Module | present |
| `QuadraticFunction` | type or type alias | present |
| `Regularize` | type or type alias | present |
| `SCAD` | type or type alias | present |
| `SLOPE` | type or type alias | present |
| `Scaled` | type or type alias | present |
| `SeparableFunction` | type or type alias | present |
| `SeparableSum` | type or type alias | present |
| `SingularValue` | type or type alias | present |
| `SliceSum` | type or type alias | present |
| `SparseGroupNormL2` | type or type alias | present |
| `SparseIterate` | type or type alias | present |
| `Spectral` | type or type alias | present |
| `SqrNormL2` | type or type alias | present |
| `Sum` | type or type alias | present |
| `SymmetricSparseIterate` | type or type alias | present |
| `Tilt` | type or type alias | present |
| `TotalVariation1D` | type or type alias | present |
| `TracePSD` | type or type alias | present |
| `Translate` | type or type alias | present |
| `ZeroFunction` | type or type alias | present |
| `activate!` | function | present |
| `activeCoordinate` | function | compatibility or module binding |
| `activecoordinates` | function | present |
| `atom` | function | present |
| `cdprox!` | function | compatibility or module binding |
| `coordinate_lipschitz` | function | present |
| `coordinate_lipschitz!` | function | present |
| `coordinates` | function | present |
| `dropzeros!` | function | SparseArrays interface |
| `gradient` | function | present |
| `gradient!` | function | present |
| `gradient_mapping` | function | present |
| `gradient_mapping!` | function | present |
| `hashessian` | function | present |
| `hessian_vec` | function | present |
| `hessian_vec!` | function | present |
| `ind2subLowerTriangular` | function | compatibility or module binding |
| `isActive` | function | compatibility or module binding |
| `isactive` | function | present |
| `isconvex` | function | present |
| `isproximable` | function | present |
| `isseparable` | function | present |
| `issmooth` | function | present |
| `iszeropreserving` | function | present |
| `kkt_violation` | function | present |
| `lipschitz` | function | present |
| `lla_weight` | function | present |
| `lla_weights` | function | present |
| `lla_weights!` | function | present |
| `nactive` | function | present |
| `ncoordinates` | function | present |
| `nnz` | function | SparseArrays interface |
| `norm_diff` | function | present |
| `numActive` | function | compatibility or module binding |
| `numCoordinates` | function | compatibility or module binding |
| `opnorm2` | function | present |
| `prox` | function | present |
| `prox!` | function | present |
| `proxL1Fused` | function | compatibility or module binding |
| `prox_coordinate!` | function | present |
| `prox_entry` | function | present |
| `quadratic_model` | function | present |
| `quadratic_model!` | function | present |
| `reset_cache!` | function | present |
| `restrict` | function | present |
| `shrink` | function | present |
| `shrinkL2!` | function | compatibility or module binding |
| `sub2indLowerTriangular` | function | compatibility or module binding |
| `tril_cartesian` | function | present |
| `tril_index` | function | present |
| `value` | function | compatibility or module binding |
| `value_and_gradient` | function | present |
| `value_and_gradient!` | function | present |
| `value_and_prox!` | function | present |
| `value_entry` | function | present |

# Deprecated type bindings

These 0.4 compatibility bindings remain available throughout 0.5, with removal
scheduled for 0.6. They do not introduce additional abstract or concrete types.

| Binding | Replacement |
| --- | --- |
| `DifferentiableFunction` | `AbstractFunction` plus `issmooth` |
| `ProximableFunction` | `AbstractFunction` plus `isproximable` |
| `ProxZero` | `ZeroFunction` |
| `ProxL1` | `NormL1` |
| `ProxL2` | `NormL2` |
| `ProxGroupL2` | `GroupNormL2` |
| `ProxNuclear` | `NuclearNorm` |
| `ProxL1Fused` | `NormL1Fused` |
| `ProxGaussLikelihood` | `LogDetLoss` |

# Documented qualified API

- `ProximalBase.ScalarLoss`
- `ProximalBase.Squared`
- `ProximalBase.Logistic`
- `ProximalBase.Poisson`
- `ProximalBase.Huber`
- `ProximalBase.SquaredHinge`
- `ProximalBase.SmoothCheck`
- `ProximalBase.ConvolutionCheck`
- `ProximalBase.Smoothed`
- `ProximalBase.deriv`
- `ProximalBase.deriv2`
- `ProximalBase.curvature`
- `ProximalBase.value_and_deriv`
- `ProximalBase.Testing`
- `ProximalBase.AbstractGroupPenalty`
- `ProximalBase.HomogeneousGroupNorm`
- `ProximalBase.AbstractGroupNorm`
- `ProximalBase.Testing.test_prox`
- `ProximalBase.Testing.test_gradient`
- `ProximalBase.Testing.test_separable`
- `ProximalBase.Testing.test_conjugate`
- `ProximalBase.Testing.test_kkt`
- `ProximalBase.Testing.test_traits`
- `ProximalBase.Testing.allocations`
- `ProximalBase.Testing.brute_force_prox`
