# [NormL0OffDiag numerical contract](@id spec-norml0offdiag)

## [Identity](@id spec-norml0offdiag-identity)

[`NormL0OffDiag`](@ref NormL0OffDiag) is the nonconvex cardinality penalty on
the off-diagonal entries of a square matrix, with its proximal map. The
implementation is in
[`src/functions/norml0offdiag.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/norml0offdiag.jl).
It is not a [`SeparableFunction`](@ref SeparableFunction); it takes the
scalar keep-or-drop decision from [`NormL0`](@ref NormL0).

## [Definition and domain](@id spec-norml0offdiag-definition)

`NormL0OffDiag(λ)` takes a finite real level ``\lambda\geq0``. For a square
matrix ``X`` it is

```math
g(X)=\lambda\,\#\{(i,j):i\neq j,\ X_{ij}\neq0\}.
```

A pair ``X_{ij}=X_{ji}\neq0`` with ``i<j`` contributes ``2\lambda`` and the
diagonal contributes nothing. Both signed zeros count as zero.

The oracles accept a one-based, strided ``p\times p`` matrix of `Float32` or
`Float64` (``p=0`` is allowed) with finite entries that is symmetric under
`==`, so a pair `(+0.0, -0.0)` is symmetric. The step ``\gamma`` is a finite
positive real number. For such ``V``, the proximal map solves

```math
\min_{Z=Z^\top}\ g(Z)+\frac{\|Z-V\|_F^2}{2\gamma}.
```

For a pair ``i<j`` the objective is
``2\lambda\mathbf 1\{z\neq0\}+2(z-v)^2/(2\gamma)``. Dividing by two gives the
scalar `NormL0` problem, so the threshold ``2\gamma\lambda`` is unchanged. The
diagonal term ``(z-v)^2/(2\gamma)`` is minimized by ``z=v``.

## [Traits](@id spec-norml0offdiag-traits)

`isproximable` is true. `isconvex`, `issmooth` and `isseparable` are false.

## [Supported arguments](@id spec-norml0offdiag-arguments)

`X` and the output `Y` are `StridedMatrix` values with the same element type,
`Float32` or `Float64`. Dense matrices, strided views and reshapes and
reinterpreted arrays that are strided are accepted. The step is a scalar
`Real`. The methods dispatch on `AbstractMatrix` arguments and a `Real` step,
so a vector, a tuple, a [`SymmetricSparseIterate`](@ref SymmetricSparseIterate),
a [`BlockIterate`](@ref BlockIterate) and an array step give `MethodError`.
`prox_block`, `restrict` and `kkt_violation` have no method for this function
and give `MethodError`.

## [Preconditions](@id spec-norml0offdiag-preconditions)

Exact input/output identity `Y === X` is supported. A different output `Y` for
which `Base.mightalias(Y, X)` is true is refused. Other ownership rules follow
the [array and precision conventions](@ref numerical-usage). The step is
checked before any entry is read.

`Scaled(NormL0OffDiag(λ), α)` takes the value ``\alpha g`` and the proximal map
of `NormL0OffDiag(λ)` at step `α * γ`, for positive finite `α` when the rounded
product `α * γ` is finite and positive. A product that rounds to zero or to
`Inf` is refused as a step.

## [Results](@id spec-norml0offdiag-results)

`prox` returns `similar(X, T)`, a `Matrix` for Base arrays, `prox!` returns the
destination, and `value_and_prox!` returns the value at the stored output. The
value has the element type `T` of the matrix. With ``k`` the number of nonzero
entries off the diagonal, it is `zero(T)` when ``k=0`` or ``\lambda=0``, and
otherwise `convert(T, R(λ) * R(k))` with `R = promote_type(T, float(typeof(λ)))`,
so a `Float64` level is not rounded to `Float32` before the product.

## [Ordinary-path clauses](@id spec-norml0offdiag-ordinary)

The diagonal is copied bit for bit, including `-0.0`. Each entry ``(i,j)`` with
``i\neq j`` is the ``(i,j)`` entry of `prox(NormL0(λ), V, γ)`, computed from
``V_{ij}`` alone and not mirrored from ``V_{ji}``. The entry is kept when
``|V_{ij}|^2>2\gamma\lambda`` and is positive zero otherwise, so a mathematical
tie gives zero. At ``\lambda=0`` the map is the identity, and a pair
`(+0.0, -0.0)` stays as it is. At ``\lambda>0`` the same pair becomes
`(+0.0, +0.0)`. The output is symmetric under `==` when the input is.

The comparison is exact when `λ` and `γ` convert exactly to a `Float32` or
`Float64` working type. Other levels and steps, for example a `Rational` or an
`Int` above ``2^{53}``, are rounded to that type first, and with a `BigFloat`
working type the product ``2\gamma\lambda`` is rounded too. The decision is then
made with the rounded values, as for `NormL0`.

## [Range clauses](@id spec-norml0offdiag-range)

The proximal entries are the entries of `V` or positive zero, so they do not
leave the range of the input. The comparison follows the
[`NormL0` rules](../numerics.md#Cardinality-penalty-endpoints) for thresholds
near the ends of the floating-point range. The value is the product of the
level and the entry count formed in `R` and then rounded to `T`, and can
overflow to `Inf` when the product rounds beyond `floatmax(T)`.

## [Endpoint clauses](@id spec-norml0offdiag-endpoints)

| Level, step or input | Behavior |
|:--|:--|
| `λ = 0` | The map is the identity and the value is zero |
| `λ < 0`, `λ = NaN`, `λ = ±Inf` | The constructor throws `ArgumentError` |
| `γ = 0`, `γ = Inf`, `γ < 0`, `γ = NaN` | `ArgumentError` before any write |
| A NaN or infinite entry of `X` | `ArgumentError`, also from the value oracle |

## [Failure behavior](@id spec-norml0offdiag-failures)

The checks run in this order and precede every write, so `X` and `Y` are
unchanged by a refused call.

1. One-based axes: `ArgumentError`.
2. A non-square `X`, or a `Y` of another size: `DimensionMismatch`.
3. A matrix that is not a `StridedMatrix` (`Symmetric`, `Hermitian`, sparse
   matrices, `Adjoint`, `Transpose`, [`matrixview`](@ref matrixview), a matrix
   [`SparseIterate`](@ref SparseIterate)): `ArgumentError`.
4. An element type other than `Float32` and `Float64`, or different types in
   `X` and `Y`: `ArgumentError`.
5. A different `Y` with `Base.mightalias(Y, X)` true: `ArgumentError`.
6. A step that is not finite and positive: `ArgumentError`.
7. A non-finite entry of `X`, then an asymmetric `X` (both triangles are
   compared and neither is selected): `ArgumentError`.

The value oracle runs the checks that concern `X`. `prox(f, Symmetric(A))`
reaches the strided check and throws `ArgumentError`.

## [Limitations and non-guarantees](@id spec-norml0offdiag-limitations)

The output is not promised to be positive definite, and no solver convergence
is promised. Weighted levels, groups, sparse and symmetric-iterate
representations, `restrict`, `prox_block`, `kkt_violation`, array steps, steps
`0` and `Inf`, and non-finite input are outside the contract. A zero-stride
view as `Y` is not detected as self-aliased, as in `Base.mightalias`.

**Implementation notes (not guarantees).** The scalar decision is
`prox_block(NormL0(λ), v, 1, γ)`. When `λ`, `γ` and the element type are the
same float type and the matrix has at least 64 entries, the implementation
uses the certified magnitude cutoff of the dense `NormL0` kernel instead, behind
a function barrier. The validation reads both triangles in one blocked pass
before the first write. The write loop uses `@simd` without `ivdep`, because on
Julia 1.10 the loop with `ivdep` gave wrong entries for a reinterpreted
`Float32` matrix over `Float64` or `ComplexF32` storage. `Y === X` has its own
single-array loop. These choices can change.

## [Clauses owned elsewhere](@id spec-norml0offdiag-elsewhere)

This section links to the shared owners.

- [Array and precision conventions](@ref numerical-usage)
- [Shared range guarantees](@ref numerical-value-range)
- [Scaling a function](@ref scaling-guide)
- [Cardinality penalty endpoints](../numerics.md#Cardinality-penalty-endpoints)
  for the scalar rule of `NormL0`

## [Evidence](@id spec-norml0offdiag-evidence)

[`test/norml0offdiag.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/norml0offdiag.jl)
compares the proximal map with an exact `Rational{BigInt}` minimization over the
keep and drop patterns of the pairs, and with `NormL0` off the diagonal for
mixed level and step types. It covers ties, signed zeros, the diagonal, every
refusal with an unchanged sentinel output, strided views, reshapes and
reinterpreted matrices as input, output and in place, `Scaled`, and zero
allocation on both routes. The
[penalty guide](@ref norml0offdiag-guide) contains an executable example, and
[`benchmark/norml0offdiag.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/benchmark/norml0offdiag.jl)
times the oracles against `NormL0`.

## [History](@id spec-norml0offdiag-history)

The [feature issue](https://github.com/mlakolar/ProximalBase.jl/issues/1058)
and its [pull request](https://github.com/mlakolar/ProximalBase.jl/pull/1063)
record the choice of a separate type over a diagonal option on `NormL0`, and
the probe of the all-entry prox with a positive-definiteness check, which
accepted smaller steps at levels of two and above and with different variable
scales.
