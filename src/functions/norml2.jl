####################################
#  ℓ2 norm  λ ‖x‖₂
####################################

# NormL2 and GroupNormL2 share the small arithmetic of numerical_kernels.jl (sums of squares, the scaled norm
# ns 2^k, the writes of a group) and nothing else: this file has the checks, the type, the oracles and the
# shrinkage, the value and the recovery of NormL2, which has no weights; groupnorml2.jl has those of
# GroupNormL2.

@noinline _l2_step_error(f, γ) = throw(ArgumentError("the step of $(nameof(typeof(f))) must be nonnegative, got $γ"))
# a negative or NaN step fails the comparison
@inline _check_l2_step(f, γ::Real) = (γ >= zero(γ) || _l2_step_error(f, γ); nothing)

@noinline _symmetric_prox_error(f) = throw(ArgumentError(
  "$(nameof(typeof(f))) has no proximal operator for a SymmetricSparseIterate; use a dense matrix such as Matrix(x)"))

####################################
#  NormL2
####################################

"""
    NormL2(λ = 1)

The Euclidean norm `f(x) = λ ‖x‖₂` (the Frobenius norm for a matrix), with a
nonnegative level `λ` (an `ArgumentError` otherwise, `NaN` included). Its
proximal operator shrinks the whole argument towards zero:
`prox_{γf}(x) = max(1 - γλ/‖x‖₂, 0) x`. On a nonempty argument it is
`GroupNormL2(λ, [1:length(x)], [1])`, one group with weight one, and follows the
same rules: a sum of such norms over groups of entries is `GroupNormL2`.

The value takes arrays, the three iterates and a `SymmetricSparseIterate`, whose
norm is that of the symmetric matrix it represents: each stored off-diagonal
coordinate counts twice in the squared norm. On a `SparseIterate` or a
`SymmetricSparseIterate` the value reads the stored entries only, also when a sum
of squares that overflows or underflows is recomputed in power-of-two units (#808).
With `λ = 0` the value is zero; with `λ = Inf` it is zero at the zero argument
(an empty one included), `Inf` at any other argument without a `NaN` entry, and
`NaN` at a `NaN` entry.

`prox!`, `prox` and `value_and_prox!` take arrays of the same size (a `BlockIterate`
or a `SparseIterate` through indexed reads and writes of its entries, O(length) in
general, as for an array) and a scalar step. A negative or `NaN` step throws an `ArgumentError`
before anything is written, and so do sizes, real element types and one-based
indexing; `prox!(x, f, x, γ)` is supported. A `SymmetricSparseIterate` argument or
output throws an `ArgumentError` before anything is written: its proximal operator is
not provided. Exceptional cases are those of a single group of `GroupNormL2` with
weight one, in this order: `λ = Inf` gives `+0` at every nonnegative step, `γ = 0`
included (#665), for every input; otherwise `λ = 0` or `γ = 0` gives the identity
(so `γ = Inf` with `λ = 0` is the identity); `γ = Inf` with a positive finite `λ` gives `+0`; at
positive finite `γ` and `λ`, a `NaN` entry makes every output `NaN`, an infinite entry
without `NaN` leaves the argument unchanged, also when the finite threshold `γλ`
overflows in floating point, and the zero argument gives `+0`. For finite inputs,
a norm whose sum of squares overflows or underflows and a threshold out of range are
recomputed in power-of-two units. A value at or above `floatmax/2` is recomputed in
power-of-two units, accumulated in at least Float64; within a few eps of the overflow
threshold either `floatmax` or `Inf` may be returned, regardless of which of them the
exact value rounds to (#904). `value_and_prox!` returns the value at the stored output.
"""
struct NormL2{T<:Real} <: AbstractFunction
  λ::T

  NormL2{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ))
end
NormL2(λ::Real = 1) = NormL2{typeof(λ)}(λ)

isproximable(::Type{<:NormL2}) = true
isconvex(::Type{<:NormL2}) = true

# λ‖x‖₂ in O(stored entries): _sumabs2 and the squared-residual kernel visit the stored values of the
# sparse and symmetric iterates only (each off-diagonal symmetric coordinate twice). The ordinary value
# needs a level that fits T, a finite product and a normal sum of squares; the exactly zero argument
# returns zero at every level (the domain of an infinite level is {0}).
(f::NormL2)(x::_ArrayOrSymmetricIterate) = (_checkreal(f, x); _norm_value(f.λ, x))
function _norm_value(λ, x)
  T = _valtype(x)
  iszero(λ) && return zero(T)
  s = _sumabs2(x)
  if _level_fits(T, λ)
    v = convert(T, convert(T, λ) * sqrt(s))
    (_l2_ordinary(v, T) & _squares_normal(s, _square_term_count(x), T)) && return v
  end
  iszero(s) && _all_zero(_stored_values(x)) && return zero(T)
  _norm_value_cold(λ, x, s, T)
end

# The scaled norm (_scaled_norm: the sum of squares recomputed in units of the largest stored entry, in
# at least Float64) times the level, rounded once to T. A NaN or ±Inf entry keeps the ordinary IEEE result
# (NaN, or Inf). An infinite level is decided by the class of x alone: T(NaN) at a NaN entry (not a
# propagated NaN), +0 where every entry is zero, otherwise Inf.
@noinline function _norm_value_cold(λ, x, s, ::Type{T}) where {T}
  R = _recovery_type(T, float(typeof(λ)))
  if isinf(λ)
    k = _square_exponent(x, nothing, R)
    return k == _EXPONENT_NAN ? T(NaN) : k == _EXPONENT_ZERO ? zero(T) : T(Inf)
  end
  convert(T, _scaled_norm(x, nothing, convert(R, s), convert(R, λ)))
end

function _check_l2_arrays(f, y::AbstractArray, x::AbstractArray, γ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  Base.require_one_based_indexing(y, x)
  _checksize(y, x)
  _check_l2_step(f, γ)
  nothing
end

prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real) =
  (_check_l2_arrays(f, y, x, γ); _norm_shrink!(y, x, f.λ, γ); y)
function value_and_prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real)
  _check_l2_arrays(f, y, x, γ)
  s = _norm_shrink!(y, x, f.λ, γ)
  T = _valtype(y)
  s == _L2_ZERO && return zero(T)
  if s >= 0
    R = _working_type(T, f.λ, true)
    λR = convert(R, f.λ)
    v = convert(T, λR * convert(R, s))
    ((floatmin(R) <= λR <= floatmax(R)) & _l2_ordinary(v, T)) && return v
  end
  # a recomputed value reads y as the value does: its stored entries only for a sparse iterate
  _norm_value(f.λ, y)
end
# routes of wrappers such as Scaled, which preflight with _checkwrite first
_value_and_prox_unchecked!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)

# y = prox of γλ‖⋅‖₂ at x, possibly in place (y === x). Returns, in T, the norm t‖x‖ of the stored vector on the
# ordinary shrinking path (each stored entry is t x[i] rounded once, so this is the norm of the stored entries to
# within a few rounding errors, without a second pass), or one of the codes _L2_ZERO and _L2_RECOMPUTE. The
# ordinary path needs a normal threshold c = γλ and a normal sum of squares that no underflowing square can have
# changed by more than ordinary rounding; everything else, exceptional parameters and values included, is
# decided by the cold path, which does not run on that path.
@inline function _norm_shrink!(y, x, λ, γ::Real)
  T = float(eltype(y))
  G = Base.OneTo(length(x))
  c, normal = _threshold_product(T, γ, λ, true)
  if normal & (floatmin(T) <= c <= floatmax(T))
    s = _indexed_sumabs2(x, G, T)
    if isfinite(s) & _squares_normal(s, length(x), T)
      n = sqrt(s)
      if n > c
        t = one(T) - c / n
        _l2_scale!(y, x, G, t)
        return t * n
      end
      return T(_l2_zero!(y, G))
    elseif iszero(s) && _l2_all_zero(x, G, T)
      return T(_l2_zero!(y, G))
    end
  end
  T(_norm_shrink_cold!(y, x, λ, γ))
end

# The parameters decide first, in this order: a level fixed at infinity (zero at every nonnegative step, #665), the
# identity (a zero level or step), an infinite step. Otherwise the values decide, at positive finite γ and λ: a NaN
# entry makes every output NaN; an infinite entry (and no NaN) makes the norm infinite and the threshold finite, so
# 1 - c/‖x‖ = 1 and x is copied, also when c overflows in floating point; a zero argument gives +0. Otherwise
# r = c/‖x‖ = (γλ 2^-k) / ns from the scaled norm ns 2^k, with c formed by the bounded-product kernel, so that
# neither c nor the norm needs to be in range: r ≥ 1 gives zero, and t = 1 - r in (0, 1] scales x without overflow.
# A product γλ 2^-k that overflows means r > 1, and one that underflows means r < floatmin, where t rounds to 1.
@noinline function _norm_shrink_cold!(y, x, λ, γ::Real)
  G = Base.OneTo(length(x))
  isinf(λ) && return _l2_zero!(y, G)
  (iszero(λ) || iszero(γ)) && return _l2_copy!(y, x, G)
  isinf(γ) && return _l2_zero!(y, G)
  R = _recovery_type(_working_type(float(eltype(y)), γ, λ, true), float(eltype(x)))
  k, ns = _l2_classified_norm(x, G, R)
  k == _EXPONENT_NAN && return _l2_fill!(y, G, convert(eltype(y), NaN))
  k == _EXPONENT_INF && return _l2_copy!(y, x, G)
  k == _EXPONENT_ZERO && return _l2_zero!(y, G)
  r = _scaled_product(convert(R, γ), convert(R, λ), one(R), -k) / ns
  r >= one(R) && return _l2_zero!(y, G)
  _l2_scale!(y, x, G, one(R) - r)
  _L2_RECOMPUTE
end
