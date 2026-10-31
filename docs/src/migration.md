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
  `LogDetLoss` return the argument's precision instead of their data's. A loss
  built from `Float32` data returns a `Float64` value, with `Float32`
  accuracy, for a `Float64` argument.
- Wrappers of data losses, such as `Scaled` and `Sum`, follow the same rule.
- The spectral functions and sets (`Spectral`, `SingularValue`,
  `NuclearNorm`, `IndPSD`, `IndBallRank`, `TracePSD`) raise an
  `ArgumentError` for element types other than `Float32` and `Float64`, where
  they raised a `MethodError`.

## Matrix products

Use `dot(view(A, row, :), b)` in place of `A_mul_B_row`, and
`dot(view(A, :, row), b)` in place of `At_mul_B_row`. Replace group-product
helpers with `mul!` on `view(A, :, G)` or its transpose.

For `A_mul_X_mul_B`, use `A * X * B`, with `X` represented as a
`SparseIterate` matrix rather than a vector of its entries. A single entry is
`dot(view(A, :, r), X, view(B, :, c))` for symmetric real `A` and `B`.
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

## Coordinate operations of a nonseparable `MoreauEnvelope` (0.6)

`restrict` of a [`MoreauEnvelope`](@ref) of a nonseparable function, including
through any wrapper that forwards it (`Scaled`, `Tilt`, `Translate`, `Regularize`,
`Precompose`), now throws an `ArgumentError`.
Previously it returned a function whose values and gradients were generally
wrong, except for group penalties. `prox_entry` and `kkt_violation` now throw
an `ArgumentError` instead of a `MethodError`. Restricting a group-penalty
envelope, including as the smooth term of an active-set solver, is no longer
supported. These coordinate operations require a separable underlying function.

## User separable functions on symmetric iterates

For 0.6, a user `SeparableFunction` used on a `SymmetricSparseIterate` must define
`ProximalBase.entry_parameters(f::MyFunction) = (:w => f.w,)`, listing every
per-entry array, or return `()` when it has none. Undeclared functions throw
`ArgumentError`. Dense arrays, `SparseIterate` and `AtomIterate` are unchanged.
Public oracle overrides should call `ProximalBase.check_symmetric_parameters(f, x)`
before writing. Solvers can use this check once before calling raw entry oracles;
mutating parameter arrays later requires checking them again.
