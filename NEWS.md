# ProximalBase.jl news

## 0.5.0

A breaking release: a Julian interface, functions that can be both smooth and
proximable, new functions for covariance selection and nonconvex problems, and
zero allocations in the hot calls. The 0.4 interface still works with
deprecation warnings (run tests with `--depwarn=yes` to see them), except for
the changes under [Silent changes](#silent-changes); it will be removed in 0.6.

### Migrating from 0.4

**Calls.** Functions are callable, and mutating functions take their output
first:

| 0.4 | 0.5 |
|:----|:----|
| `value(f, x)` | `f(x)` |
| `prox!(g, out, x, γ)` | `prox!(out, g, x, γ)` |
| `prox!(g, out, x); value(g, out)` | `value_and_prox!(out, g, x)` |
| `gradient!(f, grad, x)` | `gradient!(grad, f, x)` |
| `value_and_gradient!(f, grad, x)` | `value_and_gradient!(grad, f, x)` |
| `cdprox!(g, x, k, γ)` | `prox_coordinate!(x, g, k, γ)` |
| a private scalar shrink, such as CoordinateDescent's `_cdshrink` | `prox_entry(g, v, i, γ)` |

**Types.** One abstract type with traits replaces the two abstract types, and
functions are named for what they are:

| 0.4 | 0.5 |
|:----|:----|
| `DifferentiableFunction`, `ProximableFunction` | `AbstractFunction`, with `issmooth(f)`, `isproximable(f)`, `isseparable(f)`, `isconvex(f)` |
| `ProxZero()` | `ZeroFunction()` |
| `ProxL1(λ0)`, `ProxL1(λ0, λ)` | `NormL1(λ0)`, `NormL1(λ0, w)`; the weights are the field `w` (was `λ`) |
| `ProxL2(λ)` | `NormL2(λ)` |
| `ProxGroupL2(λ0, groups, λ)` | `GroupNormL2(λ0, groups, w)`; the weights are the field `w` (was `λ`) |
| `ProxL2Sq(λ)`, the penalty `λ‖x‖²` | `SqrNormL2(2λ)`: `SqrNormL2(μ)` is `(μ/2)‖x‖²` |
| `L2Loss(y)`, `L2Loss()` | `SqrNormL2(1, y)`, `SqrNormL2()` |
| `ProxNuclear(λ)` | `NuclearNorm(λ)` |
| `ProxL1Fused(λ1, λ2)` | `NormL1Fused(λ1, λ2)` |
| `ProxGaussLikelihood(S)` | `LogDetLoss(S)`, now also smooth |
| `QuadraticFunction(A, b, c)` | unchanged, now also proximable |
| `LeastSquaresLoss(Y, X)` | unchanged, now also proximable |

Code that dispatches on the type parameters of `QuadraticFunction`,
`LeastSquaresLoss`, `L2Loss` or `ProxL1` (such as `ProxL1{T, Nothing}`) needs
updating: `NormL1{T, Nothing}` keeps the meaning of the latter.

**Iterates and utilities.**

| 0.4 | 0.5 |
|:----|:----|
| `numCoordinates(x)` | `ncoordinates(x)` |
| `numActive(x)` | `nactive(x)` |
| `activeCoordinate(x, i)` | `activecoordinates(x)[i]` (a view) |
| `isActive(x, k)` | `isactive(x, k)` |
| `ind2subLowerTriangular(p, k)`, `sub2indLowerTriangular(p, r, c)` | `tril_cartesian(p, k)`, `tril_index(p, r, c)` |
| `A_mul_B_row(A, b, i)`, `At_mul_B_row(A, b, j)` | `dot(view(A, i, :), b)`, `dot(view(A, :, j), b)`, O(nnz) for sparse iterates |
| `At_mul_B_group!(out, A, b, G)` | `mul!(out, view(A, :, G)', b)` |
| `A_mul_B_group!(out, A, v, G, α, β)` | `mul!(out, view(A, :, G), v, α, β)` |
| `A_mul_X_mul_B(A, X, B)` | `A * X * B` |
| `A_mul_X_mul_B_rc(A, X, B, r, c)` | `dot(view(A, :, r), X, view(B, :, c))` |
| `A_mul_UUt_mul_B(A, U, B)` | `(A * U) * transpose(B * U)` |
| `proxL1Fused(x, y, λ1, λ2, γ)` | `prox(NormL1Fused(λ1, λ2), (x, y), γ)` |
| `shrinkL2!(out, x, c)` | `prox!(out, NormL2(c), x)` |

A `SparseIterate` vector that holds `vec` of a matrix in `A_mul_X_mul_B` should
become a `SparseIterate` matrix.

### Silent changes

These cannot warn, so check for them when migrating:

- **`SymmetricSparseIterate` is an ordinary `p × p` matrix.** Linear indexing,
  `length(x)` (now `p²`), iteration, `vec` and broadcasting follow the dense
  matrix. The `p(p+1)/2` coordinates, the entries of the lower triangle, go
  through `ncoordinates(x)` and `coordinates(x)`, a vector view in which
  `coordinates(x)[k]` is the old `x[k]`. Replace `length(x)` used as a number
  of coordinates by `ncoordinates(x)`, and `x[k]` with a coordinate `k` by
  `coordinates(x)[k]` or `x[r, c]`.
- **Per-entry weights on a `SymmetricSparseIterate`** are indexed like those of
  the dense matrix: a `p × p` matrix, not a vector over the lower triangle. The
  step `γ` of `prox_coordinate!` still applies to one coordinate, so
  off-diagonal coordinates take `2/a` where the diagonal takes `1/a`, as before.
- `ProxL2Sq`'s replacement `SqrNormL2` halves the convention (see above); the
  deprecated `ProxL2Sq(λ)` converts.

### New

- **Interface:** `value_and_prox!`, the allocating `gradient` and
  `value_and_gradient`, `prox_entry` for separable functions, and
  `prox_coordinate!` on every iterate for every separable function, including
  the groups of an `AtomIterate`. Steps accept any real number and computations
  follow the element type of `x`.
- **Smooth functions:** `LinearFunction`; `GLMLoss`, with `LogisticLoss`,
  `PoissonLoss` and `HuberLoss`, the losses of `η = Xβ + offset` with weights,
  whose scalar losses expose their derivatives and a curvature bound;
  proximal operators of `QuadraticFunction` and `LeastSquaresLoss` from a
  factorization computed once; `LogDetLoss` smooth, with value `Inf` off the
  positive definite cone.
- **Separable penalties:** `NormL1Nonneg`, `ElasticNet`, `IndBox`,
  `IndNonnegative`, `IndBallLinf`, `NormL0`, `MCP` and `SCAD`, with exact
  proximal operators also for the nonconvex steps of MCP and SCAD.
- **Other penalties and sets:** `NormLinf`, `GroupNormLinf`, `IndBallL0`,
  `IndBallL1`, `IndBallL2`, `IndSimplex`, `IndHyperplane`, `IndHalfspace`,
  `IndPSD` (with an optional rank), `TracePSD` and `IndBallRank`.
  `IndBallL0(s; diagonal = false)` on a `Symmetric` or a
  `SymmetricSparseIterate` is CovSel's `HardThreshold!(out, X, s, false)`:
  `prox!(out, IndBallL0(s; diagonal = false), Symmetric(X))`, except that ties
  keep exactly `s` entries.
- **Calculus:** `Scaled` (also `α * f`), `Tilt`, `Translate`, `Conjugate`,
  `SeparableSum` and `Sum`.
- **Iterates:** `coordinates(x)`; `activate!` for a `SymmetricSparseIterate`;
  `AtomIterate(x, numGroups)`; `mul!` with `SparseIterate` matrices; the
  three-argument `dot` and `*` with sparse iterates; `fill!` and `norm` on the
  stored entries.

### Fixes

- On a `SymmetricSparseIterate`, whose entries `(r, c)` and `(c, r)` share
  storage, `rmul!`, `lmul!`, `rdiv!`, `ldiv!`, `axpy!` and `axpby!` updated an
  off-diagonal entry twice, and `triu` and `tril` kept only the diagonal. Each
  entry is now updated once, scaling takes O(nnz), and `triu`, `tril` and
  `sparse` build a `SparseMatrixCSC` from the stored entries.

### Performance

The hot calls allocate nothing, which the tests check on Julia 1.10 and the
current release. The spectral functions and `LogDetLoss` keep LAPACK
workspaces (in 0.4, `ProxNuclear` and `ProxGaussLikelihood` allocated 70–220 KB
per call on 100 × 50 and 100 × 100 matrices), `value_and_prox!` takes the value
from the decomposition instead of a second one, projections onto the simplex
and the ℓ1 ball take O(n) expected time, and the separable functions touch only
the stored entries of a sparse iterate. Functions that own a workspace are not
safe to share between threads; give each thread a `deepcopy`.
