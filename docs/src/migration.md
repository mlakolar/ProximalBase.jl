# Compatibility and migration

For terms used in the current manual, see the [Glossary](glossary.md).

This manual describes the minimal core after merged PR #850, based on
`2b373c54`. The full post-#848 implementation and its documentation are
preserved at [`eff21b6d7d33c622313122f46c6ee71693c101ee`](https://github.com/mlakolar/ProximalBase.jl/tree/eff21b6d7d33c622313122f46c6ee71693c101ee)
and `legacy/full`. Those archives are historical references, not a second live
API. This development reduction is not a release announcement.

## Retained and removed scope

The [function library](function_library.md) is exhaustive. Norms and group penalties
other than `NormL1`, `NormL2`, `NormLinf`, `GroupNormL2` and `GroupNormLinf`, nonconvex penalties,
Poisson and robust-loss families, constraints and indicators other than
`IndNonnegative`, `IndBox`, `IndBallLinf` and `IndBallL2`,
spectral/nuclear families, generic composition/sum wrappers, LMO, second-order
frameworks and GPU extensions were removed. They have no replacement capability
implied by this manual. `BlockIterate` retains grouped **storage**; the group-lasso
penalty `GroupNormL2` and the norm `NormL2` were restored separately (#900), and
the norm `NormLinf` (#946) and the group norm `GroupNormLinf` (#956). The restored indicators are
`IndNonnegative` (#948), `IndBox` (#949), `IndBallLinf` (#954) and
`IndBallL2` (#950).

Keep two function objects for `f(x)+g(x)`; no generic function-object sum is
retained. Compute mathematical gradient mappings in caller code. Coordinate
curvatures and residual/working-set policies are algorithm responsibilities.
ProximalSolvers migration is separate; this documentation does not change it.

### Logistic regression

`LogisticLoss(y, X; weights, offset)` is available as a standalone smooth loss.
It requires finite responses in `[0, 1]`, including on zero-weight rows. The
archived logistic constructor did not validate the response domain. The current
constructor uses homogeneous strided `Float32` or `Float64` vectors and
accepts a strided or CSC design. See [logistic regression](smooth.md#Logistic-regression-loss).

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

`GroupNormL2` follows the same rule for a group. With `g = GroupNormL2(1.0, [[1, 2], [3, 4]], [1.0, 1.0])`,
`x = [3, 4, 0, 0]` and `v = [-0.6, -0.8, 0.3, 0.4]`, the vector `v` is stationary on the first group.
Before 0.6 the call at entry 1 returned a surrogate that is not a score of the group
(the entry of `v`, or the group screen, or an `ArgumentError`, according to the version); in 0.6 it returns
the block mismatch ``\lVert v_G+c\,x_G/\lVert x_G\rVert_2\rVert_2``, which is zero up to rounding
(of the order of `1e-16` in `Float64`), and the zero group scores ``\lVert v_G\rVert_2-c=-0.5``:

| `k` | group of `x` | `kkt_violation(g,v,x,k)` in 0.6 |
|--:|:--|--:|
| 1, 2 | nonzero | about `0` (rounding level) |
| 3, 4 | zero | `-0.5` |

The full-point value is the maximum of `0` and of the positive block scores, here about `0`; it is a
maximum of block 2-norms, not an ``\ell_\infty`` distance. `NormL2` and `NormLinf` have the full-point form and no other.

What did not change, for the unscaled functions and for a single wrapper with a finite positive factor (nested wrappers
with a product of factors that is not a normal number, and infinite factors, have the changes listed below):

* Three-argument calls `kkt_violation(g,v,i)`, except the screen of a nested wrapper around `LinearFunction` and a
  centered `SqrNormL2` (corrected below) and of `NormL2` and `NormLinf` (an `ArgumentError` below).
* Four-argument calls at zero coordinates, when `x`, `v` and the function's parameters
  (levels, weights, centers and coefficients) are `Float32` or `Float64` arrays and values of one type, `v` is finite and
  the function is not a `SqrNormL2` with an infinite level: they return the same values as
  before. A working-set solver that ranks the inactive coordinates of a `SparseIterate` with
  the four-argument form in that case is unaffected in the values it receives, although each call of the four-argument
  form does more work than a call of the screen (see [scoring a coordinate](@ref kkt-screens)). Outside that case the arithmetic is in
  the working type of the call, which can differ: for `f = SqrNormL2(0.1, Float32[1])`,
  `v = Float32[1]` and `x = Float32[0]` the value was the `Float64` `0.9` before 0.6 and is the
  `Float32` `0.9f0` in 0.6. The same applies to `GroupNormL2` at a zero group, whose result type
  was that of `v` alone.
* The group screen of `GroupNormL2` at a zero group (on an array point, for an entry
  of a zero group), and its value `abs(v[i])` for an entry in no group, for `x`, `v` and the
  parameters of one floating type and a finite `v`. One value of a zero group differs from the screen
  of earlier versions: an infinite entry of `v` with an infinite coefficient is `+Inf`, where the screen
  was `NaN` (`Inf - Inf`).
* `Scaled` over `GroupNormL2`: the screen at a zero group, and the value for an entry in no
  group (`kkt_violation(Scaled(GroupNormL2(1.0, [[1]], [1.0]), 2.0), [0.0, 3.0], [0.0, 1.0], 2)`
  is `3.0`), for a single wrapper with a finite positive factor and under the same conditions.

What changed besides the values of the table:

* `GroupNormL2` at a group with a nonzero entry, on a `BlockIterate` or on an array
  point, has the block mismatch (see above), and so has `Scaled` over `GroupNormL2` and, at a nonzero
  coordinate, `Scaled` over `NormL1`, `ElasticNet`, `ZeroFunction`, `LinearFunction` or `SqrNormL2`
  (the score evaluated on `v/α`, times `α`). A malformed parameter at the selected entry raises its own
  error first.
* The result type of the point-aware forms of the unscaled functions is `W = float(promote_type(eltype(v), eltype(x)))`
  (with the element type of a parameter array for `LinearFunction` and a centered `SqrNormL2`). A single `Scaled`
  wrapper also promotes with the type of the reciprocal factor, and a smooth chain includes the type of its factors:
  `kkt_violation(2 * NormL1(0.5f0), Float32[1], Float32[1], 1)` is a `Float64`. A `Float32` gradient with a `Float64`
  iterate, such as the default `BlockIterate(p, f)`, gives a `Float64` score for `GroupNormL2`, where earlier versions
  gave the type of `v`. The branches of the scaled forms are in the [numerical contract](numerics.md).
* `NormL2` has the full-point form `kkt_violation(f,v,x)`. Its three-argument and four-argument
  forms, which were a `MethodError`, throw an `ArgumentError` that names the full-point form.
* `NormLinf` has the full-point form `kkt_violation(f,v,x)`: the Euclidean distance from `-v` to the
  subdifferential of `λ*maximum(abs, x)` at `x`, with the face of the largest entries decided by
  equality after conversion to the working type (see [the ℓ∞ norm](@ref norm-linf-kkt)). In 0.5.4 the three-argument and four-argument calls were a `MethodError`; a call on a positive `Scaled`
  was forwarded to the inner function, so that the error named `NormLinf`, and there was no full-point form of any function.
  Now the full-point form
  has the value, also under `Scaled` with the limits of the [scaling guide](@ref scaled-kkt), and the three-argument and
  four-argument forms throw an `ArgumentError` that names the full-point form; the full-point form at a
  `SymmetricSparseIterate` throws an `ArgumentError`.
* `LeastSquaresLoss` and `QuadraticFunction` now support
  `kkt_violation(f,v,x)`, the infinity norm of the computed stationarity
  residual. Coordinate, group and symmetric-iterate point forms throw
  `ArgumentError`. The new call uses function-owned scratch and can allocate a
  coefficient buffer on its first use with a loss. See
  [smooth stationarity scores](@ref smooth-kkt) for the ownership rules and
  the caller-supplied gradient recipe.
* Nested `Scaled` wrappers whose product of factors is not a normal number: around
  `LinearFunction` and a centered `SqrNormL2` the three-argument screen returned the screen of the function
  at the unscaled `v` and omitted the product; it now applies the product to the gradient entry of the
  function (`Scaled(Scaled(LinearFunction([1.0]), 1e-200), 1e-200)` at `v = [0.25]` was `1.25` and is `0.25`). Around
  `NormL1`, `ElasticNet`, `GroupNormL2`, `NormL2` and `NormLinf` whose level folds (to a nonzero finite level that its type accepts: a normal one, or a subnormal one for a `Float32` or `Float64` level), a
  `GroupNormL2`, `NormL2` or `NormLinf` chain now scores with the folded level (for `NormLinf` the full-point form, which is its one form) (a `GroupNormL2` chain at a zero group
  was computed from `v` divided by the product, which overflows for an ordinary `v`). Where the level does not fold, the
  three-argument screen of `NormL1` and `ElasticNet` and the four-argument form of `NormL1`, `ElasticNet` and `GroupNormL2` at a zero
  coordinate or group have the value of the definition at finite entries of `v` (`NaN` and infinite entries as the [scaling guide](@ref scaled-kkt) states), computed from the coefficient `a·λ·w`, with the exponents of the three applied in
  one last step for a positive finite `λ` and `w` (a zero weight gives the coefficient 0, an infinite weight or level gives `Inf`, and an
  `ElasticNet` with an infinite `λ2` gives `-Inf`) (`Scaled(Scaled(NormL1(1.0), 2.0^-600), 2.0^-600)` at `v = [0.25]` was `Inf` and is `0.25`). `NormLinf` has no screen: around a chain whose level does not fold it was a `MethodError` in the three-argument form and at a zero coordinate, and it is now an `ArgumentError` there and in the full-point form. A nonzero coordinate or group and the
  full-point form throw an `ArgumentError`, as does an infinite factor (see the [scaling guide](@ref scaled-kkt)). A stack that
  nests an infinite factor with another `Scaled` wrapper is built by a direct call of the parameterized constructor
  `Scaled{F,T}(f, α)` (`Scaled(f, α)` and `α * f` merge the factors): its three-argument and zero-coordinate values are unspecified and can differ from
  0.5, and the nonzero and full-point forms throw an `ArgumentError`.
* `GroupNormLinf` has the margin `‖v_G‖₁ - λ0 w` of its dual norm at a zero group, and the call throws an `ArgumentError` at a
  nonzero group, in the full-point form and in the three-argument form. Before 0.6 (the legacy implementation) it returned that
  screen for a group of a `BlockIterate`, also where the group was nonzero, and `abs(v[i])` at an entry of a nonzero group of an
  array point, neither of which is a score of the nonzero group; the Euclidean screen of `GroupNormL2` is not used for it ([#634](https://github.com/mlakolar/ProximalBase.jl/issues/634)).
* `Scaled` over a function of the package without a score (the losses and `GroupNormLinf` in the four-argument and full-point forms) throws a
  `MethodError` for the call on the `Scaled` function; the `MethodError` named the inner function
  and a private array type before.
* A function defined outside the package gets its three-argument method's value at a zero
  coordinate and an `ArgumentError` at a nonzero one, on arrays, a `BlockIterate` and a
  `SymmetricSparseIterate`, unless it defines the four-argument method; see
  [Writing your own function](extending.md). Under `Scaled`, a function that has a four-argument method of
  its own was scaled through that method at a nonzero coordinate too; it now gives its scaled three-argument
  screen at a zero coordinate and an `ArgumentError` at a nonzero one, unless the methods for
  `Scaled{<:TheirType}` are defined.
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
| `NormL1Nonneg(λ0, w)` ([#947](https://github.com/mlakolar/ProximalBase.jl/issues/947)) | Constructor, fields `λ0`, `w`, value, `prox_block`, `value_entry`, array proximal operators, KKT screen, `restrict`, `Scaled` zero scaling and level folds | The domain test is exact: `x ≥ 0` with `-0.0` feasible, where the archive accepted entries down to `-sqrt(eps)` and clamped them at zero (the value rules are in the [numerical contract](numerics.md)). An infinite level follows the current `NormL1` rule (`+0` for positive weights at every `γ ≥ 0`, `NaN` and `±Inf` inputs included). With a `Float32` level and an unweighted `Float32` entry, `value_entry` returns `Float32` rather than the archive's `Float64` ([#684](https://github.com/mlakolar/ProximalBase.jl/issues/684)). A negative or `NaN` scalar step throws `ArgumentError` before writes; the archive applied the arithmetic. New: the point-aware `kkt_violation(f,v,x,k)` and `kkt_violation(f,v,x)` |
| `IndNonnegative()` ([#948](https://github.com/mlakolar/ProximalBase.jl/issues/948)) | The type, `isconvex`, `iszeropreserving`, `value_entry`, `prox_block`, the screen `kkt_violation(f,v,i) = -v[i]`, `restrict`, and zero scaling (`Scaled(f, 0)` is `f`) | The value is exact: `0` at a point with no negative entry, `Inf` at a negative entry of any size (the archive accepted `v >= -_tol`), and `NaN` at a `NaN` entry (the archive gave `Inf`). `value_and_prox!` returns the value at the stored output, so `NaN` where an output entry is `NaN` (the archive returned `0`). A negative or `NaN` scalar step throws `ArgumentError` before writes in `prox`, `prox!` and `value_and_prox!`; the archive did not check the step. The point-aware `kkt_violation` forms, conjugates, `Constrained` and the envelope slopes are not restored |
| `IndBox(l, u)` ([#949](https://github.com/mlakolar/ProximalBase.jl/issues/949)) | Scalar and per-entry bounds, value, coordinatewise projection, and restriction | Array bounds are copied, feasibility uses exact interval comparisons, and a `NaN` entry gives value `NaN` |
| `IndBallLinf(r)` ([#954](https://github.com/mlakolar/ProximalBase.jl/issues/954)) | The required-radius constructor, returning the scalar `IndBox(-r, r)` with its value, projection, restriction and scaling | The restored box uses exact bounds and propagates `NaN` entries. Negative, `NaN` and positive unsigned radii throw `ArgumentError`. Packed symmetric projection takes scalar steps |
| `IndBallL2(r)` ([#950](https://github.com/mlakolar/ProximalBase.jl/issues/950)) | Constructor, convex indicator, value and scalar-step Euclidean projection | The domain uses the computed norm without the archived feasibility slack. NaNs propagate in the value, and the fused oracle evaluates the stored point. Packed symmetric iterates are outside the restored interface. |

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
| `maximum(prox(NormLinf(0.5f0), [fill(0.17933106f0, 100_000); 1f0], 1f0))`, the clip level of `NormLinf` ([#1003](https://github.com/mlakolar/ProximalBase.jl/issues/1003), [#987](https://github.com/mlakolar/ProximalBase.jl/issues/987)) | `0.17933007f0` where the exact level is `0.5` (Julia 1.10.13, default mode, AMD Zen 3; the result depends on the order of the vectorized sums). The level was wrong where a large group of equal or nearly equal entries lay within the rounding of a sum of the entries from the level | `0.5f0`. The level now changes by roundings where the old level was accurate, and by more on inputs of this kind; the bound `2.25 n eps(T) ‖x‖∞ + 2nδ` is in the [numerical contract](numerics.md) |

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
