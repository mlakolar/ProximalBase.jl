####################################
#
#  Diagonal metrics: a step per entry, and per-coordinate curvatures
#
####################################

# In the metric of D = Diagonal(d), the proximal operator of a separable g is
#   argmin_z g(z) + Σᵢ dᵢ (zᵢ - xᵢ)² / 2,
# the scalar proximal operator with the step γᵢ = 1/dᵢ in every entry, and for a
# group norm a metric constant on each group keeps the closed form. A gradient
# method with the step 1/Lₖ in coordinate k, from the per-coordinate curvatures
# below, takes the scale of each column into account without standardizing X.

prox(f::AbstractFunction, x::AbstractArray, γ::AbstractArray) = prox!(similar(x, float(eltype(x))), f, x, γ)

prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  y === x ? y : copyto!(y, x)
end
value_and_prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_unchecked!(y, f, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (_checkreal(f, γ); _prox_unchecked!(y, f, x, γ); zero(_valtype(y)))

# separable functions: the step of each entry; on a SymmetricSparseIterate the
# steps are a p × p array read at the lower triangle, so give a symmetric one.
# Scalar and array steps share the loops of separable.jl; _entry specializes
# to the scalar or to γ[i] without a branch inside the loop.
prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_sep!(y, f, x, γ))
_prox_unchecked!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  _isbuiltin(f) ? _prox_sep!(y, f, x, γ) : prox!(y, f, x, γ)
value_and_prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_sep!(y, f, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  _isbuiltin(f) ? _value_and_prox_sep!(y, f, x, γ) : value_and_prox!(y, f, x, γ)

prox!(y, s::Scaled, x, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, s, y, x, γ); _prox_unchecked!(y, s, x, γ))
_prox_unchecked!(y, s::Scaled, x, γ::AbstractArray) =
  (_checkparams(s, x); _array_route(s, x, γ) ? _prox_chain!(y, s, x, γ) : _prox_unchecked!(y, s.f, x, _ScaledArray(γ, s.α)))
value_and_prox!(y, s::Scaled, x, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, s, y, x, γ); _value_and_prox_unchecked!(y, s, x, γ))
_value_and_prox_unchecked!(y, s::Scaled, x, γ::AbstractArray) =
  (_checkparams(s, x); _array_route(s, x, γ) ? _value_and_prox_chain!(y, s, x, γ) :
   _value_at(_scaled_value_overflow(s.f, s.α, y, _value_and_prox_unchecked!(y, s.f, x, _ScaledArray(γ, s.α))), y))
# α v at the output y of an array step with a factor inside the band: a product that overflows
# with α < 1 is recovered as for a scalar step (the levels with the factor folded in, or the
# parts of the value); an underflow is not (#359). An indicator ignores its scale and stays on this
# route for any factor: its value is exactly 0 or Inf, so an infinite factor at 0 gives 0, as
# `_scaled_nan_value` certifies on the other routes, and a NaN value stays NaN.
@inline function _scaled_value_overflow(f, α, y, v)
  r = α * v
  (!isfinite(r) && _scale_below_one(α)) ? _scaled_value_recover(f, α, y, v, r) : _indicator_product(f, r, v)
end
@inline _indicator_product(f, r, v) = _isindicator(f) ? _weighted_value(r, v) : r

# Whether an entry of an array step may leave the range in the product with the factors. A chain whose
# factors have a product that is not a normal number takes the route below. A single factor does
# when it is far from one, outside the band of a quarter of the exponent range (2^±255 for Float64,
# 2^±31 for Float32), whatever the entries are: reading them to decide would cost a pass over the step
# on every call. An entry of the step that is itself within a quarter of the range limit, with a factor
# inside the band, is rounded with the product as before. An indicator ignores its scale. An infinite
# factor over any other function takes the route below too, which hands it on as before and applies the
# rule of its value there.
@inline _factor_band(::Type{T}) where {T<:AbstractFloat} = exponent(floatmax(T)) ÷ 4
@inline _wide_factor(α::Real, ::Type{T}) where {T<:AbstractFloat} =
  !(ldexp(one(T), -_factor_band(T)) <= α <= ldexp(one(T), _factor_band(T)))
@inline _wide_factor_type(x::AbstractArray) = float(eltype(x))
@inline _wide_factor_type(x) = nothing
@inline _wide_factor(α::Real, ::Nothing) = false
@inline function _wide_scale(s::Scaled, x, γ::AbstractArray)
  α = s.α
  (α > 0 && !_isindicator(s.f)) || return false
  _wide_factor(α, float(promote_type(eltype(γ), typeof(α)))) || _wide_factor(α, _wide_factor_type(x))
end
@inline _array_route(s::Scaled, x, γ::AbstractArray) = _chained(s) || _wide_scale(s, x, γ)

# The function and the steps that a proximal operator gets for the factors a and the step γ, and how
# they relate to the chain:
# - `:folded`: the whole factor goes into the levels of the function and the steps are the caller's
#   array. The function forms the coefficient of an entry from the level, the weight and the step
#   without a product that leaves the range, and its value is the value of the chain.
# - `:shifted`: the levels do not hold the whole factor (its product with them is not a normal
#   number), so a power of two 2^k is moved from the steps into the levels, with k centring the
#   exponents of the entries of γ. This reads γ once, and only here, never on an ordinary call.
#   The value of the function is 2^k times too large for the chain.
# - `:rounded`: no level can be folded, or the shifted levels or entries still leave the range. A
#   function with no level to fold takes each step multiplied by the factors with the exponent
#   applied last, rounded into the step's type.
function _array_fold(f, a, γ)
  # a function that a stored infinite level fixes does not depend on the factors and the steps, when all
  # are positive finite numbers (`_hard_function`): its steps are the caller's array, and nothing is folded
  if _has_infinite_level(f) && _hard_scale(a) && _steps_hard(γ)
    h = _hard_function(f)
    h === nothing || return h, γ, :folded
  end
  g = _fold_level(f, _NormalScale(a))
  g === nothing || return g, γ, :folded
  shifted = _shifted_fold(f, a, γ)
  shifted === nothing ? (f, _ExpScaledArray(γ, a), :rounded) : (shifted..., :shifted)
end
function _shifted_fold(f, a::_ProjectionScale, γ::AbstractArray)
  isempty(γ) && return nothing    # no entry to shift: the rounded fallback has nothing to round
  S = promote_type(float(eltype(γ)), typeof(a.m))
  positive(g) = g > 0 && isfinite(g)
  hi = maximum(g -> positive(g) ? exponent(float(g)) : typemin(Int), γ)
  lo = minimum(g -> positive(g) ? exponent(float(g)) : typemax(Int), γ)
  lo > hi && return nothing       # no positive finite entry
  # the shifts that keep every positive entry in the range; the one that centres them is tried
  # first, then the extremes, for a level that needs to be raised or lowered by as much as they allow
  kmin, kmax = hi - exponent(floatmax(S)), lo - exponent(floatmin(S))
  kmin > kmax && return nothing
  for k in (clamp((lo + hi) ÷ 2, kmin, kmax), kmax, kmin)
    g = _fold_level(f, _NormalScale(_ProjectionScale(a.m, a.e + k)))
    g === nothing || return g, _ExpScaledArray(γ, _ProjectionScale(S(0.5), 1 - k))
  end
  nothing
end
# a single infinite factor: the steps multiplied by it, as for a factor inside the band
@inline _infinite_alone(s) = s isa Scaled && !_chained(s) && isinf(s.α)
function _prox_chain!(y, s, x, γ::AbstractArray)
  _infinite_alone(s) && return _prox_unchecked!(y, s.f, x, _ScaledArray(γ, s.α))
  f, a = _scaled_chain(s)
  g, steps, _ = _array_fold(f, a, γ)
  _prox_unchecked!(y, g, x, steps)
end
function _value_and_prox_chain!(y, s, x, γ::AbstractArray)
  if _infinite_alone(s)
    # the value at the stored output under the rule of an infinite factor (`_scaled_nan_value`)
    v = _value_and_prox_unchecked!(y, s.f, x, _ScaledArray(γ, s.α))
    return _value_at(_scaled_product_value(s.f, s.α, y, v), y)
  end
  f, a = _scaled_chain(s)
  g, steps, kind = _array_fold(f, a, γ)
  kind === :folded && return _value_at(_value_and_prox_unchecked!(y, g, x, steps), y)
  if kind === :shifted
    # the value of the chain at the point, not that of the shifted function
    _prox_unchecked!(y, g, x, steps)
    return _value_at(_scaled_value_by(f, a, y, _value_unchecked(f, y)), y)
  end
  v = _value_and_prox_unchecked!(y, g, x, steps)
  # a leaf value that left the range is recovered in the function's own units, as for a scalar step
  _value_at(_scaled_value_by(f, a, y, v), y)
end
