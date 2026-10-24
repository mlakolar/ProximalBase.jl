# ProximalBase.jl news

## Unreleased

- **`Constrained` on a hyperplane or a halfspace** starts its search for the
  multiplier from the multiplier of the last call, which it keeps (#77).
  - From there, the first step goes to the root of the line with the exact
    slope of the constraint, where the scalar proximal operators are piecewise
    linear. A first call starts from zero, as before.
  - The calls of a solver on slowly changing points take three to five passes
    over `x`, against 7 to 16 from zero, and are 1.9 to 3.5 times faster.
  - Do not share one such function between threads.
- **The projection onto the simplex** is 1.2 to 17 times faster, and with it
  `IndBallL1`, `NormLinf` and `GroupNormLinf` (#77). Condat's pass computes its
  level from a sum instead of a division at every step. It hands over to
  Michelot's algorithm when more than an eighth of the entries, and more than
  1024, become candidates, as when many stay positive.

- **Functions of stacked arrays**, for the joint estimation of K related
  matrices, such as the joint, fused and time-varying graphical lasso (#42):
  - `FiberSum(f; diagonal = f, symmetric = false)`: a vector function `f` of
    each fiber `x[i, j, :]` of an `m × n × K` array. `diagonal` gives the
    fibers `x[i, i, :]` their own function, and `nothing` leaves them
    unpenalized. The proximal operator acts fiber by fiber. With symmetric
    slices, `symmetric = true` reads the fibers with `i ≥ j` and mirrors them,
    which halves the work. `NormL1Fused` fibers take a loop without views: on
    two stacked 200 × 200 matrices the proximal operator takes 40 μs.
  - `SliceSum(f₁, …, f_K)`: the sum of the `fₖ` of the slices `x[:, :, k]`,
    with value, gradient and proximal operator slice by slice. With the
    log-det losses of K classes, it is the smooth part of the joint graphical
    lasso. ProximalSolvers' `ADMMSolver` takes the stacked array.

  Neither allocates beyond what `f` and the `fₖ` do.
- **`NormL1Fused`** acts on a pair of arrays of any shape, not only vectors. On a
  vector of length 2, it is the function of that one pair, the fiber of two
  stacked matrices. `FiberSum(NormL1Fused(λ1, λ2); diagonal =
  NormL1Fused(0, λ2))` is the penalty of the fused graphical lasso of two
  classes with the diagonal left unshrunk (#32).
- **`Constrained(f, C)`:** a convex separable `f` restricted to a box, a
  hyperplane, a halfspace or a simplex `C`.
  - On a box, the proximal operator clamps that of `f`, and the function stays
    separable, with `prox_entry` for coordinate descent.
  - On a hyperplane `⟨a, x⟩ = b`, the proximal operator is `prox_{γf}(x - γνa)`
    for the multiplier `ν` that meets the constraint, found by a search over
    the scalar `ν`. On the problems tried it took 6 to 17 passes over `x`, and
    it allocates nothing.
  - Examples: `Constrained(NormL1(λ), IndHyperplane(1, 0))`, the lasso penalty
    on coefficients that sum to zero, and
    `Constrained(IndBox(0, 1), IndHyperplane(1, k))`, the capped simplex.
- **`Spectral(f)`:** the function `f(λ(X))` of the eigenvalues of a symmetric
  matrix, for an `f` that does not depend on the order of its entries. Its
  proximal operator applies that of `f` to the eigenvalues, in a LAPACK
  workspace that it owns, and allocates nothing.
  - `Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, k)))` is the Fantope
    of sparse principal component analysis, `Spectral(IndBox(l, u))` bounds the
    eigenvalues, and `Spectral(IndSimplex(1))` gives the density matrices.
  - Weights or bounds that differ between entries are rejected.
- **`IndHyperplane` and `IndHalfspace`** take a scalar `a`, which stands for an
  array of `a`s of any size: `IndHyperplane(1, 0)` is the set of arrays whose
  entries sum to zero.
- **Example 16:** the lasso on compositional data, the lasso on a box by
  coordinate descent, and the capped simplex.
- **Example 17:** sparse principal component analysis by Fantope projection
  and selection.
- **`IndBox` with integer bounds**, such as `IndBox(0, 1)`: its proximal
  operator converts the bounds to the type of the result. A clamp between
  integer bounds compared each float with an integer exactly, which did not
  vectorize; the projection is about six times faster.

## 0.5.3

- **Separable functions of your own:** `SeparableFunction` is exported as the
  supertype to subtype. Implementing `value_entry(f, v, i)` and
  `prox_entry(f, v, i, γ)` gives the value, `prox!`, `value_and_prox!` and
  `prox_coordinate!` on every array and iterate. Declaring
  `iszeropreserving(f)` gives the O(nnz) paths on sparse iterates. These were
  the internal `_entry_value` and `_zeropreserving`.
- **`lipschitz(f)`:** a Lipschitz constant of the gradient of a smooth function,
  for step sizes.
  - It uses the closed form where there is one.
  - For a loss of a matrix, it computes the largest eigenvalue by the Lanczos
    process, without forming `X'X`: `‖X‖₂²` for `LeastSquaresLoss`, the
    curvature bound times `‖W^½ X‖₂²` for `GLMLoss`, and `‖A‖₂‖B‖₂` for
    `KronQuadratic`.
  - It is `Inf` for `LogDetLoss` and `PoissonLoss`.
- **`gradient_mapping(f, g, x, γ)`:** `(x - prox_{γg}(x - γ∇f(x))) / γ`. It is
  zero exactly at the minimizers of `f + g` when both are convex, so its norm is
  a stopping rule for any smooth plus proximable pair. `gradient_mapping!`
  allocates nothing.
- **`KronQuadratic`** accepts `A` and `B` that are symmetric up to rounding, such
  as `inv(Ω)`, and symmetrizes them.

## 0.5.2

- **Precompilation:** a PrecompileTools workload compiles the common calls into
  the package image, all in `Float64`:
  - the proximal operators of the separable, group and projection functions;
  - the values and gradients of the least-squares, logistic and quadratic losses;
  - `LogDetLoss`, `NuclearNorm`, `IndPSD` and `KronQuadratic`;
  - coordinate steps on the three iterates.

  Nine typical first calls take 0.02 s instead of 0.55 s on Julia 1.13, and
  0.01 s instead of 0.69 s on 1.10. The cost is 1.5–2 s more when the package
  is precompiled and about 30 ms more at `using ProximalBase`. PrecompileTools'
  preference `precompile_workload = false` skips the workload.
- `tril_cartesian` keeps its integer square root, used only for `p ≥ 2²⁴`, out
  of line, so that it inlines into the loops that call it (#26).

## 0.5.1

- `tril_cartesian` finds the column with the floating square root for
  `p < 2²⁴`, where it is exact, as 0.4's `ind2subLowerTriangular` did, and
  keeps the integer square root for larger `p` (#24). Loops that convert
  coordinates to entries are faster: in DiffPrecTest's `skron` at `p = 20`, an
  entry takes 9.8 ns instead of 14.3 ns on Julia 1.10, and 9.9 ns instead of
  11.1 ns on Julia 1.13.

## 0.5.0

A breaking release: a Julian interface, functions that can be both smooth and
proximable, new functions for covariance selection and nonconvex problems, and
zero allocations in the hot calls. The 0.4 interface still works with
deprecation warnings, except for the
[changes that cannot warn](#changes-that-cannot-warn); it will be removed in
0.6. `Pkg.test` shows the warnings, and with them on, every call of a
deprecated function builds a backtrace. Move the inner loops of solvers to the
new names first: CoordinateDescent's tests take 40 seconds with
`--depwarn=no`, and had not finished after 39 minutes with the warnings on.

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

### Changes that cannot warn

Check for these when migrating:

- **`SymmetricSparseIterate` is an ordinary `p × p` matrix** (silent: an index
  `k ≤ p(p+1)/2` still exists, but names another entry). Linear indexing,
  `length(x)` (now `p²`), iteration, `vec` and broadcasting follow the dense
  matrix. The `p(p+1)/2` coordinates, the entries of the lower triangle, go
  through `ncoordinates(x)` and `coordinates(x)`, a vector view in which
  `coordinates(x)[k]` is the old `x[k]`. Replace `length(x)` used as a number
  of coordinates by `ncoordinates(x)`, and `x[k]` with a coordinate `k` by
  `coordinates(x)[k]` or `x[r, c]`.
- **Per-entry weights on a `SymmetricSparseIterate`** are indexed like those of
  the dense matrix: a `p × p` matrix, with the weight of `(r, c)` at `[r, c]`
  and `[c, r]`. A vector over the lower triangle throws a `DimensionMismatch`.
  The step `γ` of `prox_coordinate!` still applies to one coordinate, so
  off-diagonal coordinates take `2/a` where the diagonal takes `1/a`, as before.
- **Constructors check their parameters**, such as `λ0 ≥ 0` and `w ≥ 0` for
  `NormL1(λ0, w)`, and throw an `ArgumentError` when the function is built
  rather than when it is used. Initialize a weight array that is filled after
  the function is built (`zeros`, not `undef`).
- **Functions defined outside ProximalBase** must implement the 0.5 names:
  callable, `prox!(y, f, x, γ)`, `prox_coordinate!(x, f, k, γ)`, and so on.
  Methods of the old names are found only by code that still calls the old
  names. While the deprecations exist, `hasmethod(ProximalBase.value, ...)`
  holds for every function; test `hasmethod(f, Tuple{typeof(x)})` instead.
- `ProxL2Sq`'s replacement `SqrNormL2` halves the convention (see above); the
  deprecated `ProxL2Sq(λ)` converts.

### New

- **Interface:** `value_and_prox!`, the allocating `gradient` and
  `value_and_gradient`, `prox_entry` for separable functions, and
  `prox_coordinate!` on every iterate for every separable function, including
  the groups of an `AtomIterate`. Steps accept any real number and computations
  follow the element type of `x`.
- **Smooth functions:** `LinearFunction`; `KronQuadratic`, the quadratic
  `⟨Θ, AΘB⟩ / 2 + ⟨C, Θ⟩` of a matrix, such as CovSel's D-trace loss, with the
  gradient over symmetric matrices on a `SymmetricSparseIterate`; `GLMLoss`,
  with `LogisticLoss`, `PoissonLoss` and `HuberLoss`, the losses of
  `η = Xβ + offset` with weights, whose scalar losses expose their derivatives
  and a curvature bound; proximal operators of `QuadraticFunction` and
  `LeastSquaresLoss` from a factorization computed once; `LogDetLoss` smooth,
  with value `Inf` off the positive definite cone.
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
