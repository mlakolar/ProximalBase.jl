# Compatibility and migration

For terms used in the current manual, see the [Glossary](glossary.md).

This manual describes the minimal core after merged PR #850, based on
`2b373c54`. The full post-#848 implementation and its documentation are
preserved at [`eff21b6d7d33c622313122f46c6ee71693c101ee`](https://github.com/mlakolar/ProximalBase.jl/tree/eff21b6d7d33c622313122f46c6ee71693c101ee)
and `legacy/full`. Those archives are historical references, not a second live
API. This development reduction is not a release announcement.

## Retained and removed scope

The [function library](function_library.md) is exhaustive. Norms and group penalties
other than `NormL1`, `NormL2`, `NormLinf` and `GroupNormL2`, nonconvex penalties, logistic/Poisson/robust-loss families, constraints and indicators,
spectral/nuclear families, generic composition/sum wrappers, LMO, second-order
frameworks and GPU extensions were removed. They have no replacement capability
implied by this manual. `BlockIterate` retains grouped **storage**; the group-lasso
penalty `GroupNormL2` and the norm `NormL2` were restored separately (#900), and
the norm `NormLinf` separately (#946), without the other group penalties and norms.

Keep two function objects for `f(x)+g(x)`; no generic function-object sum is
retained. Compute mathematical gradient mappings in caller code. Coordinate
curvatures and residual/working-set policies are algorithm responsibilities.
ProximalSolvers migration is separate; this documentation does not change it.

### Removed scalar-loss interface (0.6, breaking)

ProximalBase 0.6 removes the qualified-public `ProximalBase.ScalarLoss` and
`ProximalBase.Squared` types and their scalar derivative interface
(`ProximalBase.deriv`, `ProximalBase.deriv2`, `ProximalBase.curvature`,
`ProximalBase.value_and_deriv`)
([#916](https://github.com/mlakolar/ProximalBase.jl/issues/916)). No retained loss in
ProximalBase used them, and the names were not exported. A qualified reference such as
`ProximalBase.deriv`, a subtype declaration `<: ProximalBase.ScalarLoss`, and a
qualified method definition such as `ProximalBase.deriv(ℓ::MyLoss, η, y) = …` raise
`UndefVarError`. What `import` or `using` of these names reports, and what a later
unqualified method definition does, depends on the Julia version and the form used.
For least squares, use `LeastSquaresLoss` and its predictor/residual operations
(`ProximalBase.value_from_predictor`, `ProximalBase.value_and_gradient_from_predictor!`,
`ProximalBase.value_from_residual`, `ProximalBase.leastsquares_data`; see
[Least squares at a caller-held predictor or residual](smooth.md#Least-squares-at-a-predictor-or-residual-that-the-caller-holds)).
For squared-norm regularization, use `SqrNormL2`. Code that relied on this interface can
implement the operations it needs through the [`AbstractFunction` interface](extending.md),
or keep its own scalar-loss types and functions.

### The four-argument `kkt_violation` reads the point (0.6, breaking)

Before 0.6, for `NormL1`, `ElasticNet`, `ZeroFunction`, `LinearFunction` and `SqrNormL2`,
`kkt_violation(g,v,x,k)` returned the screen of coordinate `k` assumed zero, also when
`x[k]` was nonzero. In 0.6 it evaluates the actual coordinate or group:
at a zero coordinate it is the signed screen as before, and at a nonzero coordinate it is
a nonnegative stationarity mismatch. The three-argument form
`kkt_violation(g,v,i)` is unchanged. With `g = NormL1(0.5)`, `x = [0, 0, 2, -1]` and
`v = [0.8, 0.2, 0, 0.5]` (values to rounding):

| `i` | `xᵢ` | `kkt_violation(g,v,i)` | `kkt_violation(g,v,x,i)` before 0.6 | `kkt_violation(g,v,x,i)` in 0.6 |
|--:|--:|--:|--:|--:|
| 1 | 0 | 0.3 | 0.3 | 0.3 |
| 2 | 0 | -0.3 | -0.3 | -0.3 |
| 3 | 2 | -0.5 | -0.5 | **0.5** |
| 4 | -1 | 0 | 0 | 0 |

A new full-point form gives `kkt_violation(g,v,x) = 0.5` here, the nonnegative
``\max(0,\max_k s_k)``. A positive score at a nonzero coordinate is a stationarity
mismatch and not an instruction to set the coordinate to zero.

What did not change:

* Three-argument calls `kkt_violation(g,v,i)`.
* Four-argument calls at zero coordinates, when `x`, `v` and the function's parameters
  (levels, weights, centers and coefficients) are `Float32` or `Float64` arrays and values of one type, `v` is finite and
  the function is not a `SqrNormL2` with an infinite level: they return the same values as
  before. A working-set solver that ranks the inactive coordinates of a `SparseIterate` with
  the four-argument form in that case is unaffected. Outside that case the arithmetic is in
  the working type of the call, which can differ: for `f = SqrNormL2(0.1, Float32[1])`,
  `v = Float32[1]` and `x = Float32[0]` the value was the `Float64` `0.9` before 0.6 and is the
  `Float32` `0.9f0` in 0.6.
* The group screen of `GroupNormL2` at a zero group (on an array point, for an entry
  of a zero group), and its value `abs(v[i])` for an entry in no group.
* `Scaled` over `GroupNormL2`: the screen at a zero group, and the value for an entry in no
  group (`kkt_violation(Scaled(GroupNormL2(1.0, [[1]], [1.0]), 2.0), [0.0, 3.0], [0.0, 1.0], 2)`
  is `3.0`).

What changed besides the values of the table:

* `GroupNormL2` at a group with a nonzero entry, on a `BlockIterate` or on an array
  point, `Scaled` over `GroupNormL2` at a nonzero penalized group, and `Scaled` over
  `NormL1`, `ElasticNet`, `ZeroFunction`, `LinearFunction` or `SqrNormL2` at a nonzero
  coordinate, throw an `ArgumentError`; before 0.6 they returned a screen value or a
  surrogate such as `abs(v[i])`. A malformed parameter at the selected entry raises its own
  error first.
* A function defined outside the package with only the three-argument method gets its
  value at a zero coordinate and an `ArgumentError` at a nonzero one, on arrays, a
  `BlockIterate` and a `SymmetricSparseIterate`; see [Writing your own function](extending.md).
* On a `SymmetricSparseIterate` the four-argument form reads the stored value of the
  coordinate, and its score is in matrix-entry units. The full-point form throws an
  `ArgumentError` there.
* For arrays and `BlockIterate`, `length(v)` must equal `length(x)`
  (`DimensionMismatch`).

If code relied on the old value at a nonzero coordinate, call the three-argument form
`kkt_violation(g,v,i)`, which is the screen of the coordinate assumed zero.

## Restored functions

| Function | Restored from the archive | Differences from the archived version |
|:--|:--|:--|
| `ElasticNet(λ1, λ2, w)` ([#908](https://github.com/mlakolar/ProximalBase.jl/issues/908)) | Constructor, fields `λ1`, `λ2`, `w`, value, `prox_block`, `value_entry`, array proximal operators, KKT screen, `restrict`, `Scaled` folds and zero scaling | Built on the retained `NormL1` threshold and `SqrNormL2` ridge kernels, so ordinary results can differ in the last bits (multiplication by the rounded `1/(1 + γλ2)`). An infinite `λ1` follows the current `NormL1` rule (`+0` for positive weights, also at `γ = 0` and for `NaN`/`±Inf` inputs) where the archive returned `NaN`. An infinite `λ2` is the indicator of `{0}`. A negative or `NaN` scalar step throws `ArgumentError` before writes; the archive applied the arithmetic |

## Names, coordinates and writes

| Historical code | Retained interface |
|:--|:--|
| `AtomIterate`, `atom`, `atomindices` | `BlockIterate`, `blockview`, `blockindices`; no compatibility aliases |
| Scalar `prox_entry` | `prox_block(f,v,i,γ)` |
| `prox_coordinate!`, `prox_entry!`, `cdprox!` storage updates | Compute a pure result, then write through public storage; no retained aliases |
| Function-first mutating calls | Destination-first `prox!(out,f,x,γ)` and `gradient!(out,f,x)` |
| Iterate `nnz`, `nonzeros`, `findnz` assumptions | `nactive`, `activecoordinates`, `activevalues` (latter absent on `BlockIterate`) |
| Treat symmetric iterate as `AbstractMatrix` | `coordinates(x)` for variables, read-only `matrixview(x)` for matrix operations |
| Symmetric-iterate `x[k]` | `coordinates(x)[k]` or `matrixview(x)[k]`, depending on index space |
| Symmetric-iterate broadcast/fill | Write coordinates, or compute a dense symmetric matrix then `copyto!` |
| Symmetric array steps | Symmetric-iterate outputs accept scalar steps only; use dense output for entry-step calculations where supported |
| User separable parameter arrays without metadata | Declare `ProximalBase.entry_parameters`; call `ProximalBase.check_symmetric_parameters` once per model, because the oracles assume symmetric parameters and do not compare them |

The working set survives value writes. `fill!(x,0)` followed by `dropzeros!(x)`
clears array iterates; use `fill!(coordinates(x),0)` for a symmetric iterate.
Same-kind matching state copies replace active state; ordinary broadcasts copy
values and keep previous activity. Read the [storage guide](iterates.md) before
assuming conventional dense-array behavior.

## Changed numerical results

| Call | Before | Now |
|:--|:--|:--|
| `NormL1(Inf, w)` at step `γ = 0` ([#665](https://github.com/mlakolar/ProximalBase.jl/issues/665)) | Input returned unchanged (`shrink(x, 0⋅Inf)` read as a zero threshold) | Projection onto the closure of the domain: `+0` where `w[i] > 0`, including `NaN`/`±Inf` inputs; unchanged where `w[i] = 0`. Unweighted `NormL1(Inf)` gives all `+0` |
| `LinearFunction(c, d)(x)` at a matrix `SparseIterate` `x`, with `NaN` or `±Inf` in `c` at a position that `x` does not store or stores as `+0` or `-0` | `NaN`: the value was `dot(c, x) + d`, which formed the product of the coefficient and the zero at each position it visited (all `mn` positions for a dense `c`, the `p` diagonal positions for a `Diagonal`) | The value of the stored nonzero entries plus `d`, with `c` not read at a zero, in `O(nactive(x))` (`O(p)` for a `Diagonal` `c`, which scans the `p` diagonal positions). A nonzero entry of `x` keeps ordinary arithmetic, a `NaN` coefficient there gives `NaN`. The overflow recovery requires the coefficients at the nonzero entries to be finite, not all of `c`. `f(Matrix(x))` and `dot(c, x)` are unchanged and give `NaN` there |

Positive steps, finite levels at `γ = 0`, infinite coordinate weights at a
finite level (#853) and finite-factor overflow (#866) are unchanged. See
[NaN, infinities and zeros](numerics.md#NaN,-infinities-and-zeros).

## Changed argument checks

| Call | Before | Now |
|:--|:--|:--|
| `LinearFunction(c, d)` with `c` and `x` of the same length but different shapes, such as a vector `c` and a matrix `x`, or a `2 × 3` `c` and a `3 × 2` `x` | Accepted or refused depending on which `dot` method the array types reached: dense pairs were accepted (the prox paired entries in linear order), sparse pairs were refused by the value but accepted by `gradient!` and `prox!` | `DimensionMismatch` from every oracle, before any output is written. `c` must satisfy `size(c) == size(x)`; on a `SymmetricSparseIterate` it must be `p × p`, including for `kkt_violation` |

Pairs of the same size are unchanged, with the same values. To keep a model
written with a flattened coefficient vector, reshape it once to the shape of the
argument, as in `LinearFunction(reshape(c, size(x)), d)`.

## Changed traits and proximal domains

| Call | Before | Now |
|:--|:--|:--|
| `isconvex` of `QuadraticFunction`, its type or an instance, and of `Scaled` of it ([#919](https://github.com/mlakolar/ProximalBase.jl/issues/919)) | `true` | `false`, which means that convexity is not guaranteed for an arbitrary instance; `f` is convex when the symmetric `A` is positive semidefinite |
| `prox!`, `prox` and `value_and_prox!` of `QuadraticFunction`, directly or through `Scaled`, when the smallest eigenvalue `λmin` of the cached decomposition of `A` is negative and the rounded product `γT·(-λmin)` is not `< 1` (a NaN step included), where `γT` is the step after the factors of any `Scaled` are applied, converted to the element type of the function ([#919](https://github.com/mlakolar/ProximalBase.jl/issues/919)) | Unchecked spectral division by `1 + γλi`, which could return non-finite values or finite values that are not the proximal point | `DomainError` before the output is written |

A symmetric `A` need not be positive semidefinite. The check applies to a
singular positive semidefinite `A` whose computed `λmin` is slightly negative
from rounding, at steps with `γT·(-λmin) ≥ 1` (about `γ ≥ 1/|λmin|`); at those
steps the earlier kernel divided by a computed denominator `1 + γT·λmin ≤ 0`.
A matrix with `λmin ≥ 0`, and a negative step, are not checked and behave as
before. For a nonnegative finite `γT` and finite cached eigenvalues, the check
accepts exactly when every computed denominator `1 + γT·λi` is positive. In
exact arithmetic and for `γ > 0`, the proximal subproblem has a unique minimizer if and only if
`1 + γ λmin(A) > 0` for the true smallest eigenvalue, so near that boundary
the check can accept a step for which the true `I + γA` is not positive
definite, and reject one for which it is.

A caller that knows `A ⪰ 0` and relies on convexity asserts it at its own model
boundary, and does so again after changing `A`. A successful prox does not
establish convexity.

## Historical contracts

Current [developer contracts](developer/index.md) are reconciled against retained
methods and tests. Historical acceptance records remain available at the
immutable archive, including pending requirements and measured experiments.
They are not automatically current guarantees. Numerical endpoints, aliasing
and decomposition caches must follow the [retained numerical contract](numerics.md).
