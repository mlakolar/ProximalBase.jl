"""
    NormL0OffDiag(λ = 1)

The nonconvex penalty `g(X) = λ ⋅ #{(i, j) : i ≠ j, Xᵢⱼ ≠ 0}` on a square
matrix: each nonzero off-diagonal entry costs `λ` and the diagonal costs
nothing. A pair `Xᵢⱼ = Xⱼᵢ ≠ 0` with `i < j` contributes `2λ`. The level `λ`
must be a finite real number that is not negative; the constructor throws
`ArgumentError` otherwise. Both signed zeros count as zero.

**Inputs.** `X` and the output `Y` are one-based strided `p × p` matrices
(`p = 0` is allowed) with the same element type, `Float32` or `Float64`. `X`
has finite entries and is symmetric under `==`, so a pair `(+0.0, -0.0)` is
accepted. The step `γ` is a finite real number that is positive.

**Value.** With `k` the number of nonzero entries off the diagonal, `g(X)` has
the element type `T` of `X`. It is `zero(T)` when `k = 0` or `λ = 0`.
Otherwise it is `convert(T, R(λ) * R(k))` with
`R = promote_type(T, float(typeof(λ)))`, so a `Float64` level is not rounded to
`Float32` before the product. The result is `Inf` when the product exceeds the
range of `T`.

**Proximal map.** For `V` that meets the input conditions, the output `Y` of
`prox!(Y, g, V, γ)` minimizes `g(Z) + ‖Z - V‖²_F / (2γ)` over symmetric `Z`:
- The diagonal is copied bit for bit, including `-0.0`.
- Each entry `(i, j)` with `i ≠ j` is the `(i, j)` entry of
  `prox(NormL0(λ), V, γ)`, computed from `Vᵢⱼ` alone and not mirrored from
  `Vⱼᵢ`. The entry is kept when `|Vᵢⱼ|² > 2γλ`, using the comparison of
  [`NormL0`](@ref), and is positive zero otherwise, so a tie gives zero. At
  `λ = 0` the map is the identity, and a `(+0.0, -0.0)` pair stays as it is.
- A pair `i < j` costs `2λ·1{x ≠ 0} + 2(x - v)²/(2γ)`. Dividing by two gives
  the scalar problem of `NormL0`, so the threshold `2γλ` is unchanged. The
  diagonal term `(x - v)²/(2γ)` has the minimizer `x = v`.
- `Y` is symmetric under `==` when `V` is.

`value_and_prox!` returns the value at the stored `Y`. `Y === X` is
supported. A different `Y` for which `Base.mightalias(Y, X)` is true throws
`ArgumentError` before any write. `prox` returns a new `Matrix`.

**Refused before any write.** `X` and `Y` are left unchanged. `ArgumentError`:
a matrix that is not a `StridedMatrix` (`Symmetric`, `Hermitian`, sparse
matrices, `Adjoint`, `Transpose`, [`matrixview`](@ref)), offset axes, an element
type other than `Float32` and `Float64` or differing types of `X` and `Y`, a
step that is not finite and positive, a non-finite entry of `X`, and an
asymmetric `X`. `DimensionMismatch`: a non-square `X`, or a `Y` of another
size. Arguments that are not matrices, such as vectors, tuples,
[`SymmetricSparseIterate`](@ref) and [`BlockIterate`](@ref), and array steps give
`MethodError`.

`Scaled(NormL0OffDiag(λ), α)` multiplies the value by `α`, and its proximal
map at step `γ` is this map at step `α * γ`, for positive finite `α`.

Not provided: positive definiteness of the output, `restrict`,
`prox_block` and `kkt_violation` (each throws `MethodError`), steps `0` and
`Inf`, and non-finite input. See the [numerical contract](@ref numerics).
"""
struct NormL0OffDiag{T<:Real} <: AbstractFunction
  λ::T

  function NormL0OffDiag{T}(λ::T) where {T<:Real}
    isfinite(λ) || throw(ArgumentError("NormL0OffDiag requires a finite λ, got $λ"))
    _checknonneg("λ", λ)
    new{T}(λ)
  end
end
NormL0OffDiag(λ::Real = 1) = NormL0OffDiag{typeof(λ)}(λ)

isproximable(::Type{<:NormL0OffDiag}) = true

@noinline _l0od_error(what) = throw(ArgumentError("NormL0OffDiag $what"))

# The checks on the matrix X shared by the value and the proximal oracles.
function _l0od_check_matrix(X)
  _check_one_based(X)
  checksquare(X)
  X isa StridedMatrix || _l0od_error("requires a strided matrix, got $(typeof(X))")
  _l0od_check_float(X)
end
function _l0od_check_float(X)
  eltype(X) === Float32 || eltype(X) === Float64 ||
    _l0od_error("requires Float32 or Float64 entries, got $(eltype(X))")
  nothing
end

# One pass over both triangles, in blocks as in _check_pairs, counting the entries that are
# non-finite or differ from their mirror image under ==. The diagnosis of a failure is separate.
function _l0od_scan(X::StridedMatrix)
  p = size(X, 1)
  bad = 0
  @inbounds for cb in 1:32:p
    ce = min(cb + 31, p)
    for c in cb:ce
      bad += !isfinite(X[c, c])
    end
    for rb in cb:32:p
      for c in cb:ce
        s = 0
        @simd for r in max(c + 1, rb):min(rb + 31, p)
          u = X[r, c]
          v = X[c, r]
          s += !(isfinite(u) & isfinite(v) & (u == v))
        end
        bad += s
      end
    end
  end
  bad == 0 || _l0od_diagnose(X)
  nothing
end

# A non-finite entry is reported before an asymmetric pair.
@noinline function _l0od_diagnose(X)
  all(isfinite, X) || _l0od_error("requires finite entries in X")
  p = size(X, 1)
  for c in 1:p, r in c + 1:p
    X[r, c] == X[c, r] || _l0od_error("requires X to be symmetric under ==")
  end
  _l0od_error("requires finite entries in X that are symmetric under ==")
end

# The number of nonzero entries off the diagonal. A column sum has the integer width of the entries.
_l0od_zero(::Type{Float32}) = Int32(0)
_l0od_zero(::Type{Float64}) = Int64(0)
function _l0od_count(X::StridedMatrix)
  p = size(X, 1)
  k = 0
  @inbounds for j in 1:p
    s = _l0od_zero(eltype(X))
    @simd for i in 1:p
      s += !iszero(X[i, j])
    end
    k += s - !iszero(X[j, j])
  end
  k
end

function _l0od_check_step(γ)
  (isfinite(γ) && γ > 0) || _l0od_error("requires a finite positive step, got $γ")
  nothing
end

function _l0od_check_pair(Y, X)
  _check_one_based(Y, X)
  checksquare(X)
  _checksize(Y, X)
  (X isa StridedMatrix && Y isa StridedMatrix) ||
    _l0od_error("requires strided matrices, got $(typeof(Y)) and $(typeof(X))")
  eltype(Y) === eltype(X) || _l0od_error("requires the same element type in X and Y")
  _l0od_check_float(X)
  Y !== X && Base.mightalias(Y, X) && _l0od_error("requires Y to be X or independent of it")
  nothing
end

@inline _l0od_value(::Type{T}, λ, k::Integer) where {T} =
  (k == 0 || iszero(λ)) ? zero(T) : (R = promote_type(T, float(typeof(λ))); convert(T, R(λ) * R(k)))

function (f::NormL0OffDiag)(X::AbstractMatrix)
  _l0od_check_matrix(X)
  _l0od_scan(X)
  _l0od_value(eltype(X), f.λ, _l0od_count(X))
end

# Both rules decide one entry from its own value. The cutoff of NormL0 is used
# in the same cases as in its dense kernel: equal element, level and step types
# and at least 64 entries. The union result is handed to a loop that does not
# depend on it.
@inline _l0od_rule(f::NormL0OffDiag, X, γ) = NormL0(f.λ)
@inline function _l0od_rule(f::NormL0OffDiag{T}, X::AbstractMatrix{T}, γ::T) where {T<:Union{Float32, Float64}}
  g = NormL0(f.λ)
  length(X) < 64 ? g : _l0_cutoff(g, γ)
end

# Writes the output. The diagonal entry is read before the column is written, so that Y === X is
# supported.
function _l0od_loop!(Y::StridedMatrix, g, X::StridedMatrix, γ)
  p = size(X, 1)
  @inbounds for j in 1:p
    d = X[j, j]
    @simd ivdep for i in 1:p
      Y[i, j] = prox_block(g, X[i, j], 1, γ)
    end
    Y[j, j] = d
  end
  Y
end

function _l0od_prox!(Y, f::NormL0OffDiag, X, γ)
  _l0od_check_pair(Y, X)
  _l0od_check_step(γ)
  _l0od_scan(X)
  _l0od_loop!(Y, _l0od_rule(f, X, γ), X, γ)
  nothing
end

function prox!(Y::AbstractMatrix, f::NormL0OffDiag, X::AbstractMatrix, γ::Real)
  _l0od_prox!(Y, f, X, γ)
  Y
end

function value_and_prox!(Y::AbstractMatrix, f::NormL0OffDiag, X::AbstractMatrix, γ::Real)
  _l0od_prox!(Y, f, X, γ)
  _l0od_value(eltype(Y), f.λ, _l0od_count(Y))
end
