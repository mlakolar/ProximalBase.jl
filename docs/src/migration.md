# Compatibility and migration

The 0.5 interface uses callable function objects, output-first mutating
operations, and lowercase coordinate operations. The 0.4 compatibility names
remain available with deprecation warnings throughout 0.5 and are scheduled
for removal in 0.6. Run existing code with `--depwarn=yes` to find uses.

## Operations and types

| Compatibility call or name | Current interface |
|:---|:---|
| `value(f, x)` | `f(x)` |
| `prox!(f, y, x, γ)` | `prox!(y, f, x, γ)` |
| `gradient!(f, g, x)` | `gradient!(g, f, x)` |
| `value_and_gradient!(f, g, x)` | `value_and_gradient!(g, f, x)` |
| `cdprox!(f, x, k, γ)` | `prox_coordinate!(x, f, k, γ)` |
| `DifferentiableFunction`, `ProximableFunction` | `AbstractFunction`, with explicit capability traits |
| `ProxZero`, `ProxL1`, `ProxL2`, `ProxGroupL2` | `ZeroFunction`, `NormL1`, `NormL2`, `GroupNormL2` |
| `ProxNuclear`, `ProxL1Fused`, `ProxGaussLikelihood` | `NuclearNorm`, `NormL1Fused`, `LogDetLoss` |
| `ProxL2Sq(λ)` | `SqrNormL2(2λ)`; the new coefficient multiplies half the squared norm |
| `L2Loss(y)`, `L2Loss()` | `SqrNormL2(1, y)`, `SqrNormL2()` |
| `f.λ` for `NormL1` or `GroupNormL2` | `f.w` for weights; `f.λ0` is the overall level |
| `numCoordinates`, `numActive`, `isActive` | `ncoordinates`, `nactive`, `isactive` |
| `activeCoordinate(x, i)` | `activecoordinates(x)[i]` |
| `ind2subLowerTriangular`, `sub2indLowerTriangular` | `tril_cartesian`, `tril_index` |
| `proxL1Fused(x, y, λ1, λ2, γ)` | `prox(NormL1Fused(λ1, λ2), (x, y), γ)` |
| `shrinkL2!(out, x, c)` | `prox!(out, NormL2(c), x)` |

The legacy abstract-type bindings alias `AbstractFunction`; they do not
identify smooth or proximable subsets. Query [`issmooth`](@ref) and
[`isproximable`](@ref) instead.

## Values in mixed precision

Every scalar value now has the argument's precision, as stated in
[Precision](@ref). Results with parameters, argument, output and step all in
one precision are unchanged. Arrays are unchanged, except that
`PointwiseMinimum` chooses its candidate by comparing the rounded values. The following return types
change.

- Values that widened with parameters of another precision now have the
  argument's precision. `Scaled(NormL1(1f0), 2.0)(Float32[1, 2])` and
  `SqrNormL2(1f0, Float64[0, 1])(Float32[1, 2])` return `Float32`, where they
  returned `Float64`. This covers centred `SqrNormL2` and `Regularize`,
  `LinearFunction`, `QuadraticFunction`, `KronQuadratic`, `Scaled`, `Tilt`,
  `Sum`, `SeparableSum`, `Precompose`, `LinearImage`, `IndGraph`,
  `NormL1Fused` on pairs and tuples, `PointwiseMinimum`, and the conjugates of
  `IndBox`, `Translate`, centred `SqrNormL2` and `HingeLoss(y)`.
- The data losses `LeastSquaresLoss`, `GLMLoss` and its specializations, and
  `LogDetLoss` return the argument's precision instead of their data's. Under
  the 0.6 contract, a loss built from `Float32` data computes a `Float64`
  argument's value in `Float64` from the stored data (see below).
- Wrappers of data losses, such as `Scaled` and `Sum`, follow the same rule.
- The spectral functions and sets (`Spectral`, `SingularValue`,
  `NuclearNorm`, `IndPSD`, `IndBallRank`, `TracePSD`) raise an
  `ArgumentError` for element types other than `Float32` and `Float64`, where
  they raised a `MethodError`.

## The numerical contract (0.6)

0.6 adopts the [Numerical contract](@ref numerics). The changes below land in
the pull requests tracked in
[#366](https://github.com/mlakolar/ProximalBase.jl/issues/366); until then,
the earlier behavior applies.

- **Mixed precision.**
  - Each result is computed in its model precision. Parameters are rounded
    once to it, after being combined. Data and the geometry of linear sets are
    used as stored.
  - The value of `value_and_prox!` uses the converted parameters: centered
    `SqrNormL2(1.0, [1 + 2^-28])` with a `Float32` output returns `0` where it
    returned `2^-57`.
  - A wider output forms combined parameters in its own precision.
  - A wider argument keeps its digits with a narrower output.
  - Data-holding functions with narrower data compute in the argument's
    precision.
- **NaN.** NaN propagates to every result that depends on it: a result
  depends on an entry if changing that entry between finite values can change
  it, with the step fixed (decision 48).
  - The values of indicators, `NegEntropy` and the functions that use LAPACK at
    a NaN entry are NaN, where they were `Inf`.
  - `value_and_prox!` of an indicator at a NaN entry returns NaN, where it
    returned `0`, unless the projection does not depend on that entry. The
    radius-zero balls `IndBallL1(0)` and `IndBallL2(0)` project to zero with
    value `0`, and a coordinate fixed by an infinite weight stays zero.
  - The proximal outputs of `NormL2` and `GroupNormL2` at a NaN entry are NaN,
    where they were zeros.
  - A zero-level function, such as `NuclearNorm(0)`, returns `0` at a NaN
    entry.
  - `NormL1Nonneg` at a NaN entry is NaN, where it was `Inf`, and
    `EpsInsensitiveLoss` with `ε = Inf` is `0` at NaN and `±Inf` entries,
    where it was NaN.
  - At an admitted infinite step, a proximal map that is constant on finite
    arguments ignores a NaN entry: `prox(NormL1(1.0), [NaN], Inf)` is `[0]`,
    where it was `[NaN]`.
  - An infinite level whose proximal map is constant ignores a NaN or infinite
    entry (decision 49): `prox(NormL1(Inf), [1.0, NaN], 1.0)` is `[0, 0]`,
    where it was `[0, NaN]`, and so for `SLOPE` with an infinite first weight
    and `NormL1Fused` with an infinite ℓ1 level at a positive finite step.
    At the zero vector, `NormL1(Inf)` (a dense array, no weights) and
    `NormL1Fused(Inf, λ2)` give the value `0`, where they gave NaN.
  - `NormLinf(Inf)` is the indicator of `{0}` (decision 49, #663): at a finite
    positive step its proximal map is zero at every entry,
    `prox(NormLinf(Inf), [1.0, NaN], 1.0)` is `[0, 0]`, where it was
    `[-Inf, NaN]`. Its value at the zero vector is `0`, where it was NaN, and
    so is the value of the conjugate of `IndBallL1(Inf)` there. A zero,
    infinite, negative or NaN step keeps its results.
  - `NormL1Fused` with an infinite fusion level `λ2 = Inf` and a finite `λ1` projects onto
    the vectors of equal entries and then shrinks (decision 49, #664): at a positive finite
    step the proximal point of a vector of K ≠ 2 entries is `shrink(mean, γλ1)` in every entry,
    `prox(NormL1Fused(1.0, Inf), [1.0, -2.0, 0.5], 1.0)` is `[0, 0, 0]`, where it was
    `[-Inf, Inf, NaN]`, and `prox(NormL1Fused(0.5, Inf), [3.0], 1.0)` is `[2.5]`, where it was
    `[NaN]`; a NaN entry gives NaN in every entry. The value at equal finite entries is
    `λ1 Σ|v|`, where it was NaN (`Inf⋅0`), for a vector, a pair, a tuple of arrays and
    `FiberSum`, and the value of `value_and_prox!` is that of the stored output; a vector of
    unequal entries has the value `Inf` as before. `value_and_prox!` of a tuple of arrays or of
    `FiberSum` of two slices into a separate output stores the output of `prox!` for a pair near
    `floatmax`, where it stored `Inf`. The pair, the tuple of arrays and the vector of two
    entries gave the common value already. `λ1 = Inf`, a finite `λ2`, and a zero, infinite,
    negative or NaN step keep their results.
  - A wrapper step that is not a normal number keeps the hard domain of an
    infinite level (decision 49, #666): `prox(Scaled(NormL1(Inf), 1e-300), [1.0, 2.0, 3.0], 1e-300)`
    is `[0, 0, 0]`, where it was the identity `[1, 2, 3]`, and `SLOPE` with an
    infinite first weight, `NormL1Fused` with an infinite ℓ1 level, `NormLinf(Inf)`,
    `ElasticNet` and `SparseGroupNormL2` with an infinite level that fixes every coordinate
    that their other levels touch give zeros, where they gave NaN or the identity. This
    holds for a product `αγ` that underflows, is subnormal or overflows, for chains of
    `Scaled`, blocks of sums, array steps with positive finite entries and
    `value_and_prox!`. The call returns what the same function returns at an ordinary positive
    step, the hard domain `{0}` at finite arguments included, up to the sign of a zero
    (`ElasticNet(1.0, Inf)` at `[-1.0]` gives `[0.0]` at step 1 and `[-0.0]` at step 0.75 and in the
    wrapper); a NaN or infinite entry follows the
    leaf's own rule at an ordinary step (decision 49 for `NormL1`, `SLOPE`, `NormL1Fused` and
    `NormLinf`; the others are tracked in #662), so `prox(Scaled(NormL1Nonneg(Inf), 1e-300), [Inf, 1.0], 1e-300)`
    is `[NaN, 0]`, as `NormL1Nonneg(Inf)` gives at step 1. Nothing finite is folded, multiplied or combined on this route, so a
    finite level, a finite factor and the working precision keep their results. Declined, with
    the earlier result: a finite level that acts on a coordinate that the infinite level leaves
    free (a zero weight), a finite sibling block next to an infinite one, a factor or step that
    is zero, negative, infinite or NaN (in the chain, in a block, under an indicator, or an array entry), an
    indicator that is not on the list of plain-data indicators (an intersection, a constrained or composed
    indicator, a user type; also under a `Scaled` or a `Translate`), and a
    function without a stored infinite level; an infinite level can still be lost there, and a
    follow-up issue tracks these configurations.
  - A box that misses the domain of an infinite-level `NormL1` raises before writing
    (decision 49, #666): `prox(Constrained(NormL1(Inf), IndBox(1.0, 2.0)), [1.0, 2.0], 1.0)`
    throws `ArgumentError("the IndBox misses the domain of NormL1")`, where it returned
    `[1.0, 1.0]`, the clamp of the hard zero into the box. The same holds for `prox!`,
    `value_and_prox!`, array steps and `Scaled(Constrained(...), α)` at a positive step, with
    the check on the entries of positive weight (a zero weight leaves its entry free), and
    nothing is written before the error. A box that contains zero keeps the hard zero of an
    infinite level. Finite levels, a zero step, other penalties in the box and the standalone
    value (`Inf` away from the domain) keep their earlier behaviour. Coordinate updates of a `SparseIterate` or an `AtomIterate`
    do not run the check yet.
  - `value_and_prox!` of `FiberSum(NormL1Fused(...))` on two slices, into a separate output, at a zero level,
    a zero step or a free diagonal (`diagonal = nothing`) beside a fused pair near `floatmax`, stores
    the output of `prox!` (finite there, where a midpoint overflow stored `Inf`) and returns the value of
    that output, where it returned `Inf` or NaN (#667). The same holds for finite arguments stored into
    a narrower destination (Float64 into Float32), where free fibers become infinite: the value is that
    of the stored output, which those fibers do not enter. An output with offset axes keeps its earlier result.
  - `SparseGroupNormL2` with an infinite ℓ1 level sets every grouped entry to
    zero at every step, also with a zero group level or weight:
    `prox(SparseGroupNormL2(Inf, 0.0, [1:2], [0.0]), [3.0, 4.0, 9.0], 0.0)` is
    `[0, 0, 9]`, where it was `[NaN, NaN, 9]`, and a NaN or infinite grouped
    entry gives `0` at a positive step, where it gave NaN.
- **Infinite entries.** A bounded projection at an argument with one infinite
  entry returns its limit, where it returned NaN or a partly NaN block
  ([#469](https://github.com/mlakolar/ProximalBase.jl/issues/469), decision 48).
  - `IndBallL1`, `IndBallL2` and `IndSphereL2` move to `±r` in that coordinate
    and zero elsewhere: `prox(IndBallL2(1.0), [Inf, 0.0])` is `[1, 0]`, where
    it was `[NaN, NaN]`, and `prox(IndBallL1(1.0), [Inf, 0.0])` is `[1, 0]`,
    where it was `[NaN, 0]`. On a `SymmetricSparseIterate`, one infinite
    off-diagonal gives `±r/2` (ℓ1) or `±r/sqrt(2)` (ℓ2) in each mirrored entry.
  - `IndSimplex` selects the vertex of a single `+Inf` entry, where it
    returned `[NaN, 0]`, and gives zero to `-Inf` entries.
  - Two or more infinite entries whose relative growth decides the result give
    NaN in the whole block.
  - `IndBallL2(0)` projects to zero at an infinite entry, where it returned
    NaN.
  - `Constrained(NormL1, IndBallL2/IndSphereL2)` and
    `Constrained(GroupNormL2, IndBallL2/IndSphereL2)` take the determined limits
    group by group: a hard-fixed coordinate is zero, one infinite governed
    coordinate gives `±r` there, and two or more, or a NaN governed coordinate,
    give NaN over the coupled block
    ([#469](https://github.com/mlakolar/ProximalBase.jl/issues/469), Stages 3A
    and 3B). The families `Constrained(IndBallL0, IndBallL2/IndSphereL2)` and
    `Constrained(ZeroFunction, IndBallL2)` follow in later pull requests.
- **Float16** arguments, outputs, directions and data are rejected with an
  `ArgumentError`.
- **Steps.** Negative and NaN steps raise an `ArgumentError`. Zero and infinite
  steps are accepted only where a function names the limit.
  - `Constrained(NormL1, IndBallL2)` and `Constrained(GroupNormL2, IndBallL2)`
    name `γ = 0` and `γ = Inf`. The penalized spheres name `γ = 0`, and
    `γ = Inf` only when the level is zero or some free coordinate or group has
    no penalty; otherwise `γ = Inf` raises before writing, as it does now.
  - A named endpoint with an infinite entry returns the joint limit over
    finite steps and arguments, or NaN where that limit depends on the path:
    `prox(NormL1(1.0), [Inf], Inf)` is NaN.
- **Failures.**
  - A search that cannot certify its result in the working precision, and a
    quadratic model whose coefficients cannot be represented, raise a
    dedicated precision exception, where some raised an `ArgumentError` or
    returned `Inf` and NaN.
  - Size mismatches raise `DimensionMismatch`.
  - Non-finite coefficients of `LinearFunction` and `Tilt` are rejected at
    construction.
  - A built-in `hessian_vec!` call whose output is `x` or `v` raises an
    `ArgumentError` before writing. `QuadraticFunction`, `Regularize` and `Sum`
    returned wrong values there without an error; some other built-in calls
    happened to return the correct product.
- **`LogDetLoss`.** `value_and_prox!` returns `Inf` when the stored output is
  singular, where it could return a finite value.
- **Reproducibility.** Results are promised within their stated accuracy, not
  bit for bit across versions.

## Matrix products

Use `dot(view(A, row, :), b)` in place of `A_mul_B_row`, and
`dot(view(A, :, row), b)` in place of `At_mul_B_row`. Replace group-product
helpers with `mul!` on `view(A, :, G)` or its transpose.

For `A_mul_X_mul_B`, use `A * X * B`, with `X` represented as a
`SparseIterate` matrix rather than a vector of its entries. A single entry is
`dot(view(A, :, r), X, view(B, :, c))` for symmetric real `A` and `B`.
That `dot` skips the zero coefficients of `X` (M1 of the
[Numerical contract](@ref numerics-storage)) and `A * X * B` is not one of M1's
products, so the two can differ when `A` or `B` has a non-finite entry.
For `A_mul_UUt_mul_B`, use `(A * U) * transpose(B * U)`; its entry helper
becomes `dot(U' * view(A, :, r), U' * view(B, :, c))` in the same real setting.

## Group compatibility alias

Existing code may use `ProximalBase.AbstractGroupNorm` to recognize grouped
penalties. This compatibility alias includes `GroupMCP` and `GroupSCAD`, so it
does not imply convexity or positive homogeneity.

`LogisticLoss`, `PoissonLoss`, and `HuberLoss` are specializations of
[`GLMLoss`](@ref). `IndBallLinf` and [`Constrained`](@ref) are constructor
functions that return suitable concrete functions. Avoid dispatch that
assumes every exported constructor name is a concrete type.

## `LogDetLoss` and `KronQuadratic` check their matrices (0.6)

`LogDetLoss(S)` and the matrices `A` and `B` of `KronQuadratic(A, B, C)` must now
be exactly symmetric: pairs `(i, j)` and `(j, i)` equal (exact numeric equality, with NaN matching NaN and
`-0.0` matching `0.0`), or a
`Symmetric`, `Hermitian` of reals or `Diagonal` matrix. Anything else throws an
`ArgumentError` that names the pair with the largest difference and the relative
asymmetry `max |A[i, j] - A[j, i]| / max |A[i, j]|`, and says when that is the
size of rounding error. Before 0.6, `LogDetLoss` kept the upper triangle and
`KronQuadratic` averaged the triangles of a matrix that agreed to within `√eps`
times its largest entry. `C` is unchanged. The messages of the checks in the
two sections below are the same.

| Before 0.6 | Now |
|:---|:---|
| `LogDetLoss(S)` with `S` not symmetric | `Symmetric(S, :U)` keeps the old upper-triangle values; `Symmetric(S, :L)` takes the lower |
| `KronQuadratic(inv(S), …)`, or `A` or `B` symmetric only up to rounding | `inv(Symmetric(S))` or `inv(cholesky(S))`, which are exactly symmetric, or `Symmetric(inv(S), :L)`; `(M + M')/2` for the symmetric part |

## Writes into a `SymmetricSparseIterate` (0.6)

A [`SymmetricSparseIterate`](@ref) now accepts only symmetric values. Before
0.6, the constructor, `copyto!`, broadcasting into it, `axpy!` and `axpby!` kept
the lower triangle of a matrix that was not symmetric, and `axpy!` and `axpby!`
reshaped a vector of `p²` entries. `map!`, and `copyto!` from a vector, wrote
entry by entry, so the upper position won and a shorter source filled a prefix. They now throw an `ArgumentError` for a value whose entries `(r, c)`
and `(c, r)` differ, and a `DimensionMismatch` for a source without the axes of
the destination, before writing anything. `copyto!` with offsets or index
ranges and the indexed `axpy!` throw an `ArgumentError`. `LinearFunction` and
`Tilt` check their parameter, and `SqrNormL2`, `Translate` and `Precompose`
with an array their center, shift or scale, the same way when they write into a
`SymmetricSparseIterate`.

Indexed array assignment, broadcasts and `fill!` into part of `x`, and the
position-wise mutators listed in the [`SymmetricSparseIterate`](@ref)
docstring, now throw an `ArgumentError` before writing. Whole, in-order
indexed assignments, view and reshape broadcasts, transposition and
permutation use the symmetry checks too. Before 0.6, these calls could store
the upper position, update an off-diagonal entry twice, or write outside a
view. Use scalar writes over the lower triangle, or compute into a dense
matrix and `copyto!` a symmetric result.

| Before 0.6 | Now |
|:---|:---|
| `SymmetricSparseIterate(A)`, `copyto!(x, A)` or `x .= A` with `A` not symmetric | `Symmetric(A, :L)` in place of `A` keeps the old lower-triangle values |
| a matrix symmetric only up to rounding, such as `X' * (w .* X)` or `inv(S)` | symmetrize it, or compute it exactly symmetric: `inv(cholesky(S))`, `inv(Symmetric(S))` |
| `copyto!(x, v)`, `map!(f, x, v)` or `axpy!(α, v, x)` with a vector `v` of `p²` entries | `reshape(v, p, p)`, which must be symmetric |
| `x[:, :] = A`, `view(x, :, :) .= A` or `transpose!(x, A)` with `A` not symmetric | `Symmetric(A, :L)` in place of `A` |
| `x[i, :] = v`, `x[I, J] = A` or `view(x, I, J) .= …` writing part of `x` | scalar writes over the lower triangle, or a whole symmetric matrix |
| `cumsum(x; dims)`, `reverse(x)`, `triu!(x)`, `lu!(x)`, `sum!(x, A)`, `lmul!(D, x)` and the other rejected operations listed in the docstring | apply them to `Matrix(x)`, or compute into a dense matrix and `copyto!` a symmetric result |
| `mul!(x, v, s)` with a vector `v` of `p²` entries | `reshape(v, p, p)`, which must be symmetric |
| `copyto!(x, itr)` from a tuple or a generator | `copyto!(x, reshape(collect(itr), p, p))`, which must be symmetric |

## Per-entry parameters on a `SymmetricSparseIterate` (0.6)

Arrays that assign a weight, bound or other parameter to each matrix entry
must be exactly symmetric. Checked operations now throw an `ArgumentError`
when they find different values at mirrored positions. Some calls check only
the pairs they use; see [Symmetric parameters](@ref) for examples, the
operation table and complete validation before a solver starts.
Dense arrays, `SparseIterate` and `AtomIterate` are unchanged.

The rows below list the earlier behavior and how to adapt.
The linked operation table explains which symmetric arguments trigger checks.

| Earlier behavior | New behavior and adaptation |
|:---|:---|
| In 0.5.4, weighted `NormL1` values counted weights from both triangles, while symmetric `prox!` used only the lower weight | Checked pairs must agree; choose one triangle with `Symmetric(W, :L)`, or revise the weights to match the intended model |
| `prox!` into a `SymmetricSparseIterate` from a dense argument that is not symmetric threw, or stored the lower triangle's proximal point | The output is the proximal point at the symmetric part `(x + x') / 2`, for every built-in function; `Symmetric(x, :L)` still selects the lower triangle |
| Spectral `prox!` read the upper triangle of a dense argument that is not symmetric | It decomposes the symmetric part, for dense outputs too |
| `gradient!` and `hessian_vec!` of `KronQuadratic` into a symmetric output threw at a dense point that is not symmetric | They write the symmetric part of the result |
| Values did not check parameter symmetry | Symmetric inputs trigger checks of stored pairs or all pairs, depending on the computation |
| Gradients and gradient mappings did not check parameter symmetry | Symmetric outputs trigger checks; gradient mappings check both functions |
| Hessian products and quadratic models did not check parameter symmetry | A symmetric point triggers full parameter validation, even with a dense Hessian-product output |
| Coordinate curvature calls did not check parameter symmetry | Symmetric `L` in `coordinate_lipschitz!` triggers full validation |
| Symmetric coordinate steps read the lower-triangle parameter without comparing its mirror | `prox_coordinate!` checks the pair at the chosen coordinate |
| Four-argument `kkt_violation` read the lower entry of `g` | It uses the symmetric part at the chosen pair and checks that parameter pair; `+Inf` and `-Inf` at that pair throw `ArgumentError` |
| Symmetric proximal calls read lower-triangle steps without comparing their mirrors | Proximal calls with symmetric outputs check the relevant step pairs too |
| Separable `lla_weights!` did not check parameter symmetry | A symmetric input triggers full validation, including with a dense output or allocating `lla_weights` |
| User `SeparableFunction` types needed no per-entry declaration | Define `ProximalBase.entry_parameters(f::MyFunction)` as a tuple such as `(:w => f.w,)`, or `()` when there are none |
| Calls did not require or validate a per-entry declaration | Checked symmetric calls reject undeclared functions and malformed declarations with `ArgumentError` |
| Wrappers and tuple blocks did not check declared arrays for symmetry | Checks also visit built-in wrappers and nested `SeparableSum` blocks |
| `Testing.test_separable(...; symmetric = true)` did not validate declarations | It now validates the declaration and every mirrored parameter pair; it does not generate asymmetric copies of a user function |

Averaging `(W + W') / 2` changes both weights and can change the model.
It is not a general repair for patterns, bounds or labels, and the sum can
overflow. For extension declarations and public oracle overrides, see
[Separable functions](@ref extending-separable). Validation does not protect against later
mutation; reconstruct functions when their parameter documentation requires it.

These parameter checks complement the checks on values written into an
iterate. See [Writes into a `SymmetricSparseIterate` (0.6)](@ref).

## Coordinate operations of a nonseparable `MoreauEnvelope` (0.6)

`restrict` of a [`MoreauEnvelope`](@ref) of a nonseparable function, including
through any wrapper that forwards it (`Scaled`, `Tilt`, `Translate`, `Regularize`,
`Precompose`), now throws an `ArgumentError`.
Previously it returned a function whose values and gradients were generally
wrong, except for group penalties. `prox_entry` and `kkt_violation` now throw
an `ArgumentError` instead of a `MethodError`. Restricting a group-penalty
envelope, including as the smooth term of an active-set solver, is no longer
supported. These coordinate operations require a separable underlying function.

## The groups of an `AtomIterate` (0.6)

The fields of an [`AtomIterate`](@ref), including `groups`, are private. Read the indices of
group `k` with [`atomindices`](@ref)`(x, k)`, a read-only vector, instead of `x.groups[k]`, and
compare layouts with `ProximalBase.samegroups(a, b)` instead of comparing `a.groups` and
`b.groups`. `similar(x)` makes an iterate with the same groups. Changing the array that
`parentindices(atom(x, k))` returns is not supported. See [AtomIterate](@ref).

## Writes through `atom(x, k)` (0.6)

Writes through [`atom`](@ref)`(x, k)` now activate the group, as `x[i] = v`
does. Before 0.6, they left it inactive, and values, `copyto!` and products
ignored the entries. The type of `atom(x, k)` changes to a view whose parent
is the iterate. [`prox_coordinate!`](@ref) on an [`AtomIterate`](@ref) returns
this view.

Code that wrote a trial point into a group before
`prox_coordinate!(x, f, k, γ)` should call `prox_coordinate!(x, f, k, u, γ)`
instead. Writing the trial point through `atom` now activates the group even
when the step ends at zero. The block form activates it only when the step
ends nonzero.

The `storage` field is internal. Writing into it can leave the iterate
inconsistent. Activation follows the stored value: a value that rounds to
zero in the element type does not activate a group or store a coordinate.

## Scaling a `SparseIterate` by a number (0.6)

`rmul!`, `lmul!`, `rdiv!` and `ldiv!` of a `SparseIterate` by a number now
equal the corresponding broadcasts, `x .*= s`, `x .= s .* x`, `x ./= s` and
`x .= s .\ x`, in values and in the working set. Before 0.6, they scaled
only the stored entries. With an `Inf` or NaN multiplier or a zero divisor,
unstored entries stayed zero.

For example, take `x = SparseIterate(4); x[1] = 1; activate!(x, 2)`.
`rmul!(x, Inf)` gave `[Inf, NaN, 0, 0]` with active coordinates `[1, 2]`.
It now gives `[Inf, NaN, NaN, NaN]` with active coordinates `[1, 2, 3, 4]`.

Finite multipliers and finite nonzero divisors still cost `O(nnz)` and give
the same results as before. `SymmetricSparseIterate` keeps the earlier
behavior for now.

## Complex values written into a `SparseIterate` or an `AtomIterate` (0.6)

Two-argument `copyto!`, `copy!` and `fill!` into a [`SparseIterate`](@ref) or
an [`AtomIterate`](@ref) now check the element type of the source or value, as
the constructors and broadcasts already did. A complex source or value throws
an `ArgumentError` before anything is written. Before 0.6, `copyto!` and
`copy!` from a complex array converted entry by entry: a nonzero imaginary
part threw an `InexactError` after the earlier entries had been written, and a
zero imaginary part was accepted. `fill!` threw an `InexactError` for a
nonzero imaginary part and accepted `fill!(x, 1 + 0im)` and `fill!(x, 0im)`.
`copyto!(x, λ * I)` and `copyto!(x, Q)` from an orthogonal factor into a
matrix `SparseIterate` are checked too. Before 0.6, a complex one with zero
imaginary parts, such as `(1 + 0im) * I`, was accepted, and one with a nonzero
imaginary part wrote `x` before it failed.

Writes through a view, through `vec` of a matrix or a `reshape` to another
shape, and through `atom(x, k)` keep Base's conversion: a complex value throws
an `InexactError`, possibly after a partial write.

| Before 0.6 | Now |
|:---|:---|
| `copyto!(x, c)` or `copy!(x, c)` with a complex `c` whose imaginary parts are zero | `copyto!(x, real(c))` |
| `fill!(x, 1 + 0im)` or `fill!(x, 0im)` | `fill!(x, real(v))` |
| `copyto!(x, (1 + 0im) * I)`, or a complex orthogonal factor with zero imaginary parts | `copyto!(x, real(λ) * I)`, or a real factor |

## Writes into a `SparseIterate` or `AtomIterate` keep the working set (0.6)

Every write through the array interface is now a value write: it activates the
coordinates (groups) it makes nonzero and removes none. Only `dropzeros!`
removes coordinates. Three spellings change.

- `fill!(x, 0)` and `x .= 0` no longer clear the working set. They write zeros
  into the active coordinates, which stay active. Base and LinearAlgebra
  mutators that start with a `fill!`, such as `sum!(x, A)`, keep the working
  set too. To clear, write `dropzeros!(fill!(x, 0))`.
- `x .= y` with `y` an iterate of the same kind (a `SparseIterate` of the same
  size, or an `AtomIterate` with equal groups) is a value write, like every
  other broadcast: `x` keeps its working set and gains the nonzeros of `y`.
  Before 0.6 it copied the working set of `y`. For that copy, write
  `copyto!(x, y)`, which costs `O(nnz)` where the broadcast costs `O(length(x))`.
- `copyto!` and `copy!` between two `SparseIterate`s of different shapes and
  the same length are value writes in linear order. Before 0.6, `copyto!`
  threw a `DimensionMismatch` and `copy!` an `ArgumentError`. Between two
  array iterates of different lengths, both now throw a `DimensionMismatch`
  before writing. Before 0.6, `copyto!` between a `SparseIterate` and an
  `AtomIterate` copied a shorter source into the first entries and threw a
  `BoundsError` for a longer one, and `copy!` threw a `MethodError` (an
  `ArgumentError` between a matrix and a vector).

For example, take `x = SparseIterate(4); x[4] = 7; activate!(x, 2)`.
`fill!(x, 0)` gave active coordinates `[]`. It now gives `[4, 2]`, and
`dropzeros!(fill!(x, 0))` gives `[]`.

To replace the working set by the support of an array `A`, first check
`length(A) == length(x)` (else throw a `DimensionMismatch`) and
`eltype(A) <: Real` (else throw an `ArgumentError`). Then write
`copyto!(dropzeros!(fill!(x, 0)), A)` when `A` is not an iterate of the same
kind as `x`, shares no memory with `x`, and its reads and conversions to
`eltype(x)` cannot fail, for example a real floating-point array into a
floating-point `x`. Otherwise stage the values first, which leaves `x`
unchanged if a read or conversion fails:

```julia
t = similar(x)
for (i, j) in zip(eachindex(t), eachindex(A))
    t[i] = A[j]
end
copyto!(x, t)
```

## `coordinates(x)` of a `SparseIterate` is a vector view (0.6)

[`coordinates`](@ref)`(x)` of a [`SparseIterate`](@ref) is now a vector view of
the entries of `x`, for a vector and for a matrix alike, as it already was for a
[`SymmetricSparseIterate`](@ref). Before 0.6 it was `x` itself, of the shape of
`x`. Code that relied on that changes in six ways.

- `coordinates(x) === x` no longer holds. For a matrix `x`, `size(coordinates(x))`
  is `(length(x),)` and not `size(x)`, so a loop over `axes(coordinates(x), 1)`
  now visits every coordinate. Indexing by coordinate, `coordinates(x)[k]`, is
  unchanged.
- A write through the view converts the value as a write into a plain vector
  does. A complex value throws an `InexactError`, possibly after part of `x` has
  been written, where `coordinates(x) .= c` threw an `ArgumentError` before
  writing for a vector `x`, and a `DimensionMismatch` for a matrix `x` (the view
  was the matrix, of another shape than `c`). `fill!(coordinates(x), 0im)` and `fill!(coordinates(x), 1 + 0im)` are
  accepted, and `fill!(coordinates(x), 2im)` throws before writing.
- Products and reductions of the view are those of an ordinary vector: IEEE
  arithmetic, and `O(length(x))`. `dot(v, coordinates(x))` was the sparse
  `O(nnz)` product that skips absent coordinates, so with a NaN in `v` at an
  absent coordinate it gave a number and now gives NaN. For the sparse product
  write `dot(v, x)`.
- `copyto!(coordinates(x), coordinates(y))` is a value write: `x` keeps its
  working set and gains the nonzeros of `y`. Before 0.6, for a vector it copied
  the working set of `y`. For that copy, write `copyto!(x, y)`.
- The view as a source is a plain vector too. `copyto!(x, coordinates(y))` and
  `copy!(x, coordinates(y))` were the state copy and are now a value write: `x`
  keeps its working set and gains the nonzeros of `y`; for the state copy write
  `copyto!(x, y)`. A view shorter than `x` copies a prefix, as any vector does,
  where `copyto!(x, y)` throws a `DimensionMismatch`. For a matrix `x`,
  `copy!(xm, coordinates(ym))` throws Base's `ArgumentError` for the axes of a
  vector and a matrix.
- `fill!(coordinates(x), 0)` writes zeros into the stored entries, which stay
  active, as `fill!(x, 0)` does; to clear, write `dropzeros!(fill!(x, 0))`.

## Products of an iterate: zero coefficients are strong (0.6)

In the products and inner products that ProximalBase defines for a
[`SparseIterate`](@ref) or an [`AtomIterate`](@ref), a zero coefficient of the
iterate, stored or not, now contributes nothing, even against `±Inf` or NaN in
the other operand. These are `dot` with a strided vector or another iterate,
`dot(a, X, b)` for a matrix `X`, and `mul!` and `*` with a strided, transposed,
adjoint or `Symmetric` matrix; M1 of the
[Numerical contract](@ref numerics-storage) lists them. Before 0.6, the
`SparseIterate` products skipped unstored zeros and multiplied stored ones, so
`0 · NaN` gave NaN, and the `AtomIterate` `dot` multiplied the zeros of its
active groups: a result could change with `dropzeros!` and with the order of
the arguments.

For example, take `A = [NaN 1 0; NaN 1 0]` and
`u = SparseIterate(3); u[2] = 1; activate!(u, 1)`, with an explicit zero at 1.
`A * u` gave `[NaN, NaN]`, and `[1.0, 1.0]` after `dropzeros!(u)`; it now gives
`[1.0, 1.0]` both times. With `s = [1, 0, 0]` and `t = [0, NaN, 0]` as
`SparseIterate`s, `dot(s, t)` gave `0.0` and `dot(t, s)` NaN; both now give
`0.0`. Values and gradients computed through these products change with them:
`LeastSquaresLoss([1, 1], [NaN 1; NaN 1])` at `z = SparseIterate([0.0, 1.0])`
gave `0.0`, and NaN once `activate!(z, 1)` stored an explicit zero at 1; it now
gives `0.0` both times. With no nonzero coefficient, the five-argument `mul!`
now leaves `β * out` (zero for `β = 0`) also for an infinite or NaN `α`, where
the transposed product gave NaN, and so did the others when every stored
coefficient was zero.

A non-finite entry that meets a nonzero coefficient still propagates, and a
zero of a dense operand is still an IEEE zero: with `w[1] = Inf`,
`dot([0, 1, 0], w)` is NaN. For IEEE propagation through a zero coefficient,
multiply the dense array: `A * Vector(x)`. Products that ProximalBase does not
define, such as `sparse(A) * x`, products of objects built from an iterate,
such as `view(x, :)`, `coordinates(x)` and `atom(x, k)`, and scaling by a number
follow IEEE arithmetic and are unchanged. The products of a
`SymmetricSparseIterate` keep the earlier behavior for now.
