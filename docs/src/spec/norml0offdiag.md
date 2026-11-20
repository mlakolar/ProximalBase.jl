# [`NormL0(λ; diagonal = false)` numerical contract](@id spec-norml0offdiag)

## [Identity](@id spec-norml0offdiag-identity)

[`NormL0(λ; diagonal = false)`](@ref NormL0) is the nonconvex cardinality
penalty on the off-diagonal entries of a square matrix, with its proximal map.
It returns an internal type, not a `NormL0`. `NormL0(λ)` and
`NormL0(λ; diagonal = true)` are the all-entry penalty with its full set of
arguments and are not described on this page. The implementation is in
[`src/functions/norml0offdiag.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/norml0offdiag.jl).
It is not a [`SeparableFunction`](@ref SeparableFunction); it takes the
scalar keep-or-drop decision from [`NormL0`](@ref NormL0).

## [Definition and domain](@id spec-norml0offdiag-definition)

`NormL0(λ; diagonal = false)` takes a finite real level ``\lambda\geq0``. For a square
matrix ``X`` it is

```math
g(X)=\lambda\,\#\{(i,j):i\neq j,\ X_{ij}\neq0\}.
```

For a symmetric ``X``, a pair ``X_{ij}=X_{ji}\neq0`` with ``i<j`` contributes
``2\lambda``; the diagonal contributes nothing. Both signed zeros count as zero.

The oracles accept a one-based, strided ``p\times p`` matrix of `Float32` or
`Float64` (``p=0`` is allowed) with finite entries. The step ``\gamma`` is a
finite positive real number. Symmetry of the matrix under `==` is a precondition
of the statements about the symmetric problem on this page, and it is not
checked; a pair `(+0.0, -0.0)` is symmetric. For a finite ``V`` that is
symmetric under `==`, when the comparison of the
[ordinary-path clauses](@ref spec-norml0offdiag-ordinary) is exact, the proximal
map solves

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
reinterpreted arrays that are strided and that Base can both read and write
are accepted. A reinterpretation that Base can write but not read, or the
reverse, such as a `Float64` reinterpretation of a parent whose element type
has padding, is refused before any write. The step is a scalar
`Real`. Choosing `diagonal = false` changes the penalty and narrows the
arguments: the vectors, sparse matrices, [`SparseIterate`](@ref SparseIterate)
and [`SymmetricSparseIterate`](@ref SymmetricSparseIterate) values and array
steps that `NormL0(λ)` accepts are refused here. A vector, a tuple, a
[`SparseIterate`](@ref SparseIterate) vector, a
[`SymmetricSparseIterate`](@ref SymmetricSparseIterate), a
[`BlockIterate`](@ref BlockIterate) and an array step throw `ArgumentError`
from the value, `prox`, `prox!` and `value_and_prox!`. A step that is neither
`Real` nor an array throws `ArgumentError` from `prox!` and
`value_and_prox!`, and `MethodError` from `prox`. `prox` runs the storage,
element-type and shape checks on `X` before it allocates the output. Other
arguments outside these families, for example a number as `X`, can give
`MethodError`. Called on this function directly, `prox_block`,
`prox_block!`, `restrict` and `kkt_violation` throw `ArgumentError`.

## [Preconditions](@id spec-norml0offdiag-preconditions)

Exact input/output identity `Y === X` is supported. A different output `Y` for
which `Base.mightalias(Y, X)` is true is refused. Other ownership rules follow
the [array and precision conventions](@ref numerical-usage). The step is
checked before any entry is read. The caller establishes the symmetry of the
matrix; see the definition above.

`Scaled(NormL0(λ; diagonal = false), α)` returns `α` times the rounded value
of `NormL0(λ; diagonal = false)`, with the product converted to the element
type of the matrix. Because the value is rounded first, the result can be
`Inf` when ``\alpha g(X)`` is representable: for `X = Float32[1 2; 2 1]`,
`λ = floatmax(Float32)` and `α = 0.25f0` it is `Inf32`. The proximal map at
step `γ` is the map of `NormL0(λ; diagonal = false)` at the step that `Scaled`
computes from `α` and `γ` (see [Scaling a function](@ref scaling-guide)), and
a computed step that is not finite and positive is refused. That step can
differ from the rounded product `α * γ` near the ends of the floating-point
range: `α = nextfloat(1/6)` and `γ = 3 * nextfloat(0.0)` give a step of zero,
which is refused, although `α * γ` is `nextfloat(0.0)`.

Through `Scaled`, the preprocessing of `Scaled` can read an unsupported
argument before the refusal, for example an array step, or a matrix input with
a [`SymmetricSparseIterate`](@ref SymmetricSparseIterate) output. Through
`Scaled`, `prox_block` of a vector block, `prox_block!` and the four-argument
and full-point forms of `kkt_violation` throw `MethodError`.

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
``|V_{ij}|^2>2\gamma\lambda`` and is positive zero otherwise, so, when the
comparison is exact as described below, a mathematical tie gives zero. At ``\lambda=0`` the map is the identity, and a pair
`(+0.0, -0.0)` stays as it is. At ``\lambda>0`` the same pair becomes
`(+0.0, +0.0)`. The output is symmetric under `==` when the input is.

For an asymmetric input the same rule is applied to each entry, so neither
triangle is selected. The output is then the entrywise map, which need not be the
proximal point over symmetric matrices, and the value counts every nonzero
off-diagonal entry of the given matrix.

The comparison is exact when `λ` and `γ` convert exactly to a `Float32` or
`Float64` working type. `NormL0` converts the entry, `λ` and `γ` to the working
type and compares the square of the entry with ``2\gamma\lambda``. Other
levels and steps, for example a `Rational` or an `Int` above ``2^{53}``, are
rounded to that type first. With a `BigFloat` working type the converted
entry, its square and the product ``2\gamma\lambda`` can each be rounded at
the current precision: at a precision of two bits, `λ = BigFloat(0.5)` and
`γ = 1.0` give the exact threshold one, and a pair of `1.25` is dropped because
the entry converts to `1.0`. The decision is then made with the converted and
rounded values, as for `NormL0`.

## [Range clauses](@id spec-norml0offdiag-range)

Each proximal entry is the entry of `V` or positive zero, so its magnitude
does not increase. The comparison follows the
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

The direct oracles run these checks in this order and precede every write, so `X` and `Y` are
unchanged by a refused call.

1. One-based axes: `ArgumentError`.
2. A matrix that is not a `StridedMatrix` (`Symmetric`, `Hermitian`, sparse
   matrices, `Adjoint`, `Transpose`, [`matrixview`](@ref matrixview), a matrix
   [`SparseIterate`](@ref SparseIterate)): `ArgumentError`, whatever its shape.
3. An element type other than `Float32` and `Float64`, or different types in
   `X` and `Y`: `ArgumentError`.
4. A reinterpreted `X` or `Y` that Base cannot both read and write:
   `ArgumentError`.
5. A non-square `X`, or a `Y` of another size: `DimensionMismatch`.
6. A different `Y` with `Base.mightalias(Y, X)` true: `ArgumentError`.
7. A step that is not finite and positive: `ArgumentError`.
8. A non-finite entry of `X`: `ArgumentError`.

An asymmetric `X` is not refused. The value oracle runs the checks that concern `X`. `prox(f, Symmetric(A))`
reaches the strided check and throws `ArgumentError`.

## [Limitations and non-guarantees](@id spec-norml0offdiag-limitations)

The output is not promised to be positive definite, and no solver convergence
is promised. Weighted levels, groups, sparse and symmetric-iterate
representations, `restrict`, `prox_block` and `kkt_violation` are not provided.
The proximal map is not provided at steps `0` and `Inf`, at non-finite input
or for array steps, and these calls are refused as the
[supported-argument](@ref spec-norml0offdiag-arguments),
[endpoint](@ref spec-norml0offdiag-endpoints) and
[failure](@ref spec-norml0offdiag-failures) sections state. A zero-stride
view as `Y` is not detected as self-aliased, as in `Base.mightalias`.

### [Implementation notes](@id spec-norml0offdiag-implementation)

These notes describe the current code and are not guarantees; they can change.

- The scalar decision is `prox_block(NormL0(λ), v, 1, γ)`. When `λ`, `γ` and
  the element type are the same float type and the matrix has at least 64
  entries, the implementation uses the certified magnitude cutoff of the dense
  `NormL0` kernel instead when a cutoff can be certified, behind a function
  barrier.
- The finiteness check reads the entries before the first write. For a
  `Matrix` it sums `x - x` over the entries in one pass in linear index order,
  which gives `+0.0` when every entry is finite and `NaN` otherwise. For other
  strided matrices it counts the non-finite entries column by column, and
  each column sum is an integer of the entry width (`Int32` for `Float32`,
  `Int64` for `Float64`).
- The write loop uses `@simd` without `ivdep`, because on Julia 1.10 the loop
  with `ivdep` gave wrong entries for a reinterpreted `Float32` matrix over
  `Float64` or `ComplexF32` storage. `Y === X` has its own single-array loop.
- The [benchmark notes](https://github.com/mlakolar/ProximalBase.jl/blob/master/benchmark/README.md)
  record measured time ratios to the `NormL0` oracles. They describe one
  machine and Julia version.

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
mixed level and step types. It covers ties, signed zeros, the diagonal, the
refusals listed in the failure section, with an unchanged sentinel output where
the caller passes one, the entrywise result for an asymmetric input, the `ArgumentError` refusals of non-matrix arguments, strided views, reshapes and
reinterpreted matrices as input and output, and strided views and reinterpreted matrices in place, `Scaled`, and zero
allocation on both routes. It also checks that `NormL0(λ)` and
`NormL0(λ; diagonal = true)` are the same object, with bit-identical values and
proximal outputs. The
[penalty guide](@ref norml0offdiag-guide) contains an executable example, and
[`benchmark/norml0offdiag.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/benchmark/norml0offdiag.jl)
times the oracles against `NormL0`.

## [History](@id spec-norml0offdiag-history)

The [feature issue](https://github.com/mlakolar/ProximalBase.jl/issues/1058)
and its [pull request](https://github.com/mlakolar/ProximalBase.jl/pull/1063)
record the choice of a separate type, later replaced by the `diagonal`
keyword of `NormL0` with an internal type, and the probe of the all-entry prox with a positive-definiteness check, which
accepted smaller steps at levels of two and above and with different variable
scales.
