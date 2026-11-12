####################################
#  ℓ2 norm  λ ‖x‖₂, and the shared ℓ2 shrinkage kernels
####################################

# The kernels below compute, for an index set G of an array x (all of x for NormL2, one group for
# GroupNormL2), the shrinkage max(1 - γλw/‖x[G]‖₂, 0) x[G] and the term λw‖x[G]‖₂ of a value. They take
# the level λ and the weight w as scalars (w = true for NormL2) and allocate nothing.

@noinline _l2_step_error(f, γ) = throw(ArgumentError("the step of $(nameof(typeof(f))) must be nonnegative, got $γ"))
# a negative or NaN step fails the comparison
@inline _check_l2_step(f, γ::Real) = (γ >= zero(γ) || _l2_step_error(f, γ); nothing)

# Codes returned by the shrinkage kernel in place of the norm of the stored group: the stored group is
# exactly zero, or its value must be recomputed from the stored entries.
const _L2_ZERO = -2
const _L2_RECOMPUTE = -1

@inline function _l2_sumsq(x, G, ::Type{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in G
    s += abs2(convert(T, x[i]))
  end
  s
end

# A zero sum of squares is also the sum of an underflowing group; a vectorized sum of magnitudes
# tells an exactly zero group apart (it is NaN or nonzero otherwise).
@inline function _l2_allzero(x, G, ::Type{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in G
    s += abs(convert(T, x[i]))
  end
  iszero(s)
end

# One pass for the cold paths: whether the group holds a NaN or an infinity, and the largest exponent
# of its finite nonzero entries (typemin(Int) when there is none).
function _l2_scan(x, G, ::Type{R}) where {R}
  nan = inf = false
  e = typemin(Int)
  @inbounds for i in G
    v = convert(R, x[i])
    if isnan(v)
      nan = true
    elseif isinf(v)
      inf = true
    elseif !iszero(v)
      e = max(e, Int(exponent(v)))
    end
  end
  nan, inf, e
end

# ‖x[G]‖₂ = ns 2^e for finite entries with the largest exponent e: every entry is scaled by 2^-e before
# it is squared, so its largest magnitude is in [1, 2), and the squares are summed compensated. Then
# ns is in [1, 2√|G|) and no square overflows or underflows below the rounding of the sum.
function _l2_scaled_norm(x, G, ::Type{R}, e::Int) where {R}
  sc = _power_factors(R, -e)
  s = c = zero(R)
  @inbounds for i in G
    s, c = _compensated_add(s, c, abs2(_scale_value(convert(R, x[i]), sc)))
  end
  sqrt(s + c)
end

@inline function _l2_zero!(y, G)
  S = eltype(y)
  @inbounds for i in G
    y[i] = zero(S)
  end
  _L2_ZERO
end
@inline function _l2_copy!(y, x, G)
  if y !== x
    @inbounds for i in G
      y[i] = x[i]
    end
  end
  _L2_RECOMPUTE
end
@inline function _l2_fill!(y, G, v)
  @inbounds for i in G
    y[i] = v
  end
  _L2_RECOMPUTE
end
@inline function _l2_scale!(y, x, G, t::T) where {T}
  S = eltype(y)
  @inbounds @simd for i in G
    y[i] = convert(S, t * convert(T, x[i]))
  end
  nothing
end

# y[G] = prox of γλw‖⋅‖₂ at x[G], possibly in place (y === x), where G indexes both arrays. Returns, in
# T, the norm t‖x[G]‖ of the stored group on the ordinary shrinking path (each stored entry is t x[i]
# rounded once, so this is the norm of the stored entries to within a few rounding errors, without a
# second pass), or one of the codes above. The ordinary path needs a normal threshold c = γλw (formed
# as _l1_prox_threshold forms it) and a normal sum of squares that no underflowing square can have
# changed by more than ordinary rounding; everything else, exceptional parameters and values included,
# is decided by the cold path, which does not run on that path.
@inline function _l2_shrink!(y, x, G, λ, w, γ::Real)
  T = float(eltype(y))
  R = _working_type(T, γ, λ, w)
  ab = convert(R, γ) * convert(R, λ)
  c = convert(T, ab * convert(R, w))
  if (floatmin(R) <= ab <= floatmax(R)) & (floatmin(T) <= c <= floatmax(T))
    s = _l2_sumsq(x, G, T)
    if isfinite(s) & _squares_normal(s, length(G), T)
      n = sqrt(s)
      if n > c
        t = one(T) - c / n
        _l2_scale!(y, x, G, t)
        return t * n
      end
      return T(_l2_zero!(y, G))
    elseif iszero(s) && _l2_allzero(x, G, T)
      return T(_l2_zero!(y, G))
    end
  end
  T(_l2_shrink_cold!(y, x, G, λ, w, γ))
end

# The parameters decide first, in this order: an unpenalized group (w = 0), a group fixed at zero (an
# infinite weight, or an infinite level at every nonnegative step, #665), the identity (a zero level or
# step), a group fixed by an infinite step. Then the values, at positive finite γ, λ and w: a NaN entry
# makes every output NaN; an infinite entry (and no NaN) makes the norm infinite and the threshold finite,
# so 1 - c/‖x‖ = 1 and the group is copied, also when c overflows in floating point; a zero group gives +0.
# Otherwise r = c/‖x‖ = (γλw 2^-e) / ns from the scaled norm ns 2^e, with c formed by the bounded-product
# kernel, so that neither c nor the norm needs to be in range: r ≥ 1 gives zero, and t = 1 - r in (0, 1]
# scales the group without overflow. A product γλw 2^-e that overflows means r > 1, and one that
# underflows means r < floatmin, where t rounds to 1.
@noinline function _l2_shrink_cold!(y, x, G, λ, w, γ::Real)
  iszero(w) && return _l2_copy!(y, x, G)
  (isinf(w) || isinf(λ)) && return _l2_zero!(y, G)
  (iszero(λ) || iszero(γ)) && return _l2_copy!(y, x, G)
  isinf(γ) && return _l2_zero!(y, G)
  T = float(eltype(y))
  R = promote_type(_working_type(T, γ, λ, w), float(eltype(x)))
  nan, inf, e = _l2_scan(x, G, R)
  nan && return _l2_fill!(y, G, convert(eltype(y), NaN))
  inf && return _l2_copy!(y, x, G)
  e == typemin(Int) && return _l2_zero!(y, G)
  ns = _l2_scaled_norm(x, G, R, e)
  r = _scaled_product(convert(R, γ), convert(R, λ), convert(R, w), -e) / ns
  r >= one(R) && return _l2_zero!(y, G)
  t = one(R) - r
  S = eltype(y)
  @inbounds for i in G
    y[i] = convert(S, t * convert(R, x[i]))
  end
  _L2_RECOMPUTE
end

####################################
#  Values near the overflow threshold
####################################

# An ordinary value v carries a relative error of a few eps, which decides its class only away from the
# overflow threshold: below floatmax(T)/2 it is certainly finite, and the ordinary paths return it only
# there (a NaN fails the test too). Larger values are formed in power-of-two units as m 2^E, with m from
# compensated sums and a few rounded products (relative error below 8 eps(T)), and classified here: a
# value certainly below the threshold floatmax + ulp/2 at which rounding to T overflows is returned, one
# certainly above it is Inf, and one within 9 eps of it is recomputed exactly by `exact()` (_l2_exact,
# over the same entries), so that the class of a value is that of its correctly rounded result.
@inline _l2_ordinary(v, ::Type{T}) where {T} = v < floatmax(T) / 2

@noinline function _l2_overflow_class(::Type{T}, m, E::Integer, exact::F) where {T, F}
  fm, fe = _frexp(m)
  fe += E
  top = exponent(floatmax(T)) + 1                 # floatmax(T) = (1 - eps/2) 2^top
  (fe < top || (fe == top && fm <= 1 - 9eps(T))) && return convert(T, ldexp(fm, fe))
  (fe > top + 1 || (fe == top + 1 && fm >= (1 + 9eps(T)) / 2)) && return T(Inf)
  exact()
end

# The exact recomputation: value() evaluates a value in BigFloat with more than four times the precision
# of T, from the norms below, and the result is rounded once to T.
_l2_exact_precision(::Type{T}) where {T} = 4precision(T) + 64
@noinline _l2_exact(::Type{T}, value::F) where {T, F} = setprecision(() -> convert(T, value()), BigFloat, _l2_exact_precision(T))
# ‖x[G]‖₂, and the norm of the stored entries of x each with its multiplicity, in BigFloat
function _l2_big_norm(x, G)
  s = zero(BigFloat)
  for i in G
    v = BigFloat(float(x[i]))
    s += v * v
  end
  sqrt(s)
end
function _l2_big_norm_stored(x)
  s = zero(BigFloat)
  for (i, a) in enumerate(_stored_values(x))
    v = BigFloat(float(a))
    s += _square_multiplicity(x, i) * (v * v)
  end
  sqrt(s)
end

# The term λw‖x[G]‖₂ of a value in T, given the norm s of x[G] from the shrinkage kernel, or one of the
# codes above (s = _L2_RECOMPUTE reads x). The ordinary path needs a normal λw, a value below
# floatmax(T)/2 and, when it reads x, a normal sum of squares; a larger value given the norm s is
# recomputed from the stored entries x[G].
@inline function _l2_term(λ, w, x, G, s, ::Type{T}) where {T}
  iszero(w) && return zero(T)
  s == _L2_ZERO && return zero(T)
  R = _working_type(T, λ, w)
  λw = convert(R, λ) * convert(R, w)
  if floatmin(R) <= λw <= floatmax(R)
    if s == _L2_RECOMPUTE
      q = _l2_sumsq(x, G, T)
      if isfinite(q) & _squares_normal(q, length(G), T)
        v = convert(T, λw * sqrt(convert(R, q)))
        _l2_ordinary(v, T) && return v
      elseif iszero(q) && _l2_allzero(x, G, T)
        return zero(T)
      end
    else
      v = convert(T, λw * convert(R, s))
      _l2_ordinary(v, T) && return v
    end
  end
  _l2_term_cold(λ, w, x, G, T)
end

# Exceptional terms, in this order: a zero level with a finite weight gives 0 whatever x holds; a NaN
# entry gives NaN and an infinite one Inf; the zero group gives 0; an infinite weight or level gives Inf.
# A positive finite level and weight give λw ns 2^e from the scaled norm, classified near the overflow
# threshold (_l2_overflow_class).
@noinline function _l2_term_cold(λ, w, x, G, ::Type{T}) where {T}
  iszero(λ) && !isinf(w) && return zero(T)
  R = promote_type(_working_type(T, λ, w), float(eltype(x)))
  nan, inf, e = _l2_scan(x, G, R)
  nan && return T(NaN)
  inf && return T(Inf)
  e == typemin(Int) && return zero(T)
  (isinf(w) || isinf(λ)) && return T(Inf)
  mλ, eλ = _frexp(convert(R, λ))
  mw, ew = _frexp(convert(R, w))
  m = (mλ * mw) * _l2_scaled_norm(x, G, R, e)
  _l2_overflow_class(T, m, eλ + ew + e, () -> _l2_exact(T, () -> BigFloat(λ) * BigFloat(w) * _l2_big_norm(x, G)))
end

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
power-of-two units, and within 9 eps of the overflow threshold in BigFloat, so that it
is finite exactly when its correctly rounded value is. `value_and_prox!` returns the
value at the stored output.
"""
struct NormL2{T<:Real} <: AbstractFunction
  λ::T

  NormL2{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ))
end
NormL2(λ::Real = 1) = NormL2{typeof(λ)}(λ)

isproximable(::Type{<:NormL2}) = true
isconvex(::Type{<:NormL2}) = true

(f::NormL2)(x::_ArrayOrSymmetricIterate) = (_checkreal(f, x); _l2_value(f.λ, x))

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

# The sum of squares recomputed in units of the largest stored entry, m 2^(2k), and λ√m 2^k classified
# near the overflow threshold (_l2_overflow_class, whose exact recomputation also reads the stored
# entries only). A NaN or ±Inf entry keeps the ordinary IEEE result (NaN, or Inf), and an infinite level
# is Inf away from zero.
@noinline function _l2_value_cold(λ, x, s, ::Type{T}) where {T}
  R = promote_type(T, float(typeof(λ)))
  q = _residual_squares(x, nothing, R)
  q === nothing && return isinf(λ) ? (isnan(s) ? T(NaN) : T(Inf)) : convert(T, convert(R, λ) * sqrt(convert(R, s)))
  m, k = q
  iszero(m) && return zero(T)
  isinf(λ) && return T(Inf)
  mλ, eλ = _frexp(convert(R, λ))
  _l2_overflow_class(T, mλ * sqrt(m), eλ + k, () -> _l2_exact(T, () -> BigFloat(λ) * _l2_big_norm_stored(x)))
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
  (_check_l2_arrays(f, y, x, γ); _l2_shrink!(y, x, Base.OneTo(length(x)), f.λ, true, γ); y)
function value_and_prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real)
  _check_l2_arrays(f, y, x, γ)
  G = Base.OneTo(length(x))
  s = _l2_shrink!(y, x, G, f.λ, true, γ)
  # a recomputed value reads y as the value does: its stored entries only for a sparse iterate
  s == _L2_RECOMPUTE ? _l2_value(f.λ, y) : _l2_term(f.λ, true, y, G, s, _valtype(y))
end
# routes of wrappers such as Scaled, which preflight with _checkwrite first
_value_and_prox_unchecked!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)
