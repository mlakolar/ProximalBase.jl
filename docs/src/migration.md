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

| Before 0.6 | Now |
|:---|:---|
| `SymmetricSparseIterate(A)`, `copyto!(x, A)` or `x .= A` with `A` not symmetric | `Symmetric(A, :L)` in place of `A` keeps the old lower-triangle values |
| a matrix symmetric only up to rounding, such as `X' * (w .* X)` or `inv(S)` | symmetrize it, or compute it exactly symmetric: `inv(cholesky(S))`, `inv(Symmetric(S))` |
| `copyto!(x, v)`, `map!(f, x, v)` or `axpy!(α, v, x)` with a vector `v` of `p²` entries | `reshape(v, p, p)`, which must be symmetric |
