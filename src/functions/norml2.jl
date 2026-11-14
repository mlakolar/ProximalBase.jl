####################################
#  ℓ2 norm  λ ‖x‖₂
####################################

# The recovery building blocks that NormL2 and GroupNormL2 share (the classified norm, the cold proximal kernel, the
# exponent sum) are in numerical_kernels.jl; this file has the checks, the type, the oracles and the loops of NormL2,
# and groupnorml2.jl those of GroupNormL2.

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

(f::NormL2)(x::_ArrayOrSymmetricIterate) = (_checkreal(f, x); _l2_value(f.λ, x))

function _check_l2_arrays(f, y::AbstractArray, x::AbstractArray, γ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  Base.require_one_based_indexing(y, x)
  _checksize(y, x)
  _check_l2_step(f, γ)
  nothing
end

function prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real)
  _check_l2_arrays(f, y, x, γ)
  s = _norml2_shrink!(y, x, f.λ, γ)
  s == float(eltype(y))(_L2_COLD) && _l2_prox_cold!(y, x, Base.OneTo(length(x)), f.λ, true, γ)
  y
end
function value_and_prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real)
  _check_l2_arrays(f, y, x, γ)
  T = _valtype(y)
  s = _norml2_shrink!(y, x, f.λ, γ)
  s == T(_L2_COLD) ? _norml2_cold_fused!(y, x, f.λ, γ, T) : _norml2_fused_value(f.λ, y, s, T)
end
# routes of wrappers such as Scaled, which preflight with _checkwrite first
_value_and_prox_unchecked!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)

####################################
#  Kernels of NormL2
####################################

# The code of the ordinary path when it does not apply (nothing is written then): the cold kernel decides.
const _L2_COLD = -5

# y = prox of γλ‖⋅‖₂ at x, possibly in place (y === x), the one group of all the entries and the weight one, on the
# ordinary path. Returns, in T, the norm t‖x‖ of the stored output (each stored entry is t x[i] rounded once, so this is
# the norm of the stored entries to within a few rounding errors, without a second pass), the code _L2_ZERO of a zero
# output, or _L2_COLD with nothing written. The threshold is the product c = γλ, which the path needs in the normal range
# of the type of the product and of T, and a normal sum of squares that no underflowing square can have changed by more
# than ordinary rounding. Everything else, exceptional parameters and values included, is decided by the cold kernel,
# which does not run on that path.
@inline function _norml2_shrink!(y, x, λ, γ::Real)
  T = float(eltype(y))
  G = Base.OneTo(length(x))
  R = _working_type(T, γ, λ, true)
  ab = convert(R, γ) * convert(R, λ)
  c = convert(T, ab)
  if (floatmin(R) <= ab <= floatmax(R)) & (floatmin(T) <= c <= floatmax(T))
    s = _indexed_sumabs2(x, G, T)
    if isfinite(s) & _squares_normal(s, length(G), T)
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
  T(_L2_COLD)
end

# The cold kernel and the value at the stored output from its passes: the zero output is 0, a NaN or an infinite group
# gives its class, and the stored norm m 2^e of a scaled group is scaled by λ with one rounding; anything else (a copy)
# is the value of y.
@noinline function _norml2_cold_fused!(y, x, λ, γ, ::Type{T}) where {T}
  code, m, e = _l2_prox_cold_norm!(y, x, Base.OneTo(length(x)), λ, true, γ)
  code == _L2_ZERO && return zero(T)
  code == _L2_NAN && return T(NaN)
  code == _L2_INF && return T(Inf)
  iszero(m) || return convert(T, _scaled_product(convert(typeof(m), λ), m, e))
  _l2_value(λ, y)
end

# The value at the stored output y, given the code or the stored norm s of the ordinary path: the zero output is 0 and the
# norm t‖x‖ gives λ t‖x‖ where the level is in the normal range of its working type and the value is below floatmax(T)/2;
# anything else (a level or value out of the range) is the value of y.
@inline function _norml2_fused_value(λ, y, s::T, ::Type{T}) where {T}
  s == T(_L2_ZERO) && return zero(T)
  R = _working_type(T, λ, true)
  lr = convert(R, λ)
  v = convert(T, lr * convert(R, s))
  ((floatmin(R) <= lr <= floatmax(R)) & _l2_ordinary(v, T)) && return v
  _l2_value(λ, y)
end

# λ‖x‖₂ in O(stored entries): _sumabs2 and the squared-residual kernel visit the stored values of the
# sparse and symmetric iterates only (each off-diagonal symmetric coordinate twice). The ordinary value
# needs a level that fits T, a finite product and a normal sum of squares; the exactly zero argument
# returns zero at every level (the domain of an infinite level is {0}).
function _l2_value(λ, x)
  T = _valtype(x)
  iszero(λ) && return zero(T)
  s = _sumabs2(x)
  if _level_fits(T, λ)
    v = convert(T, convert(T, λ) * sqrt(s))
    (_l2_ordinary(v, T) & _squares_normal(s, _square_term_count(x), T)) && return v
  end
  iszero(s) && _all_zero(_stored_values(x)) && return zero(T)
  _l2_value_cold(λ, x, s, T)
end

# The class of a dense array of Float32 or Float64 and its scaled norm, from the classified norm of the kernels (one pass
# for the class and the largest exponent, one for the scaled sum): NaN at a NaN entry, Inf at an infinite entry or an
# infinite level away from zero, 0 at the zero array, and otherwise λ ns 2^k rounded once to T.
@noinline function _l2_value_cold(λ, x::DenseArray{<:Union{Float32, Float64}}, s, ::Type{T}) where {T}
  R = _recovery_type(T, float(typeof(λ)))
  k, ns = _l2_classified_norm(x, Base.OneTo(length(x)), R, !isinf(λ))
  k == _EXPONENT_NAN && return T(NaN)
  k == _EXPONENT_ZERO && return zero(T)
  (k == _EXPONENT_INF || isinf(λ)) && return T(Inf)
  convert(T, _scaled_product(convert(R, λ), ns, k))
end

# The scaled norm (_scaled_norm: the sum of squares recomputed in units of the largest stored entry, in
# at least Float64) times the level, rounded once to T. A NaN or ±Inf entry keeps the ordinary IEEE result
# (NaN, or Inf). An infinite level is decided by the class of x alone: T(NaN) at a NaN entry (not a
# propagated NaN), +0 where every entry is zero, otherwise Inf.
@noinline function _l2_value_cold(λ, x, s, ::Type{T}) where {T}
  R = _recovery_type(T, float(typeof(λ)))
  if isinf(λ)
    k = _square_exponent(x, nothing, R)
    return k == _EXPONENT_NAN ? T(NaN) : k == _EXPONENT_ZERO ? zero(T) : T(Inf)
  end
  convert(T, _scaled_norm(x, nothing, convert(R, s), convert(R, λ)))
end
