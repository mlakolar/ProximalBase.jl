"""
    SCAD(λ, a = 3.7)

The separable smoothly clipped absolute deviation penalty of Fan and Li (2001).
The level satisfies `λ ≥ 0` and the shape satisfies `a > 2`. For finite
positive `λ` and finite `a`, an entry of magnitude `t` contributes `λ*t`
up to `λ`, `λ*t - (t-λ)^2/(2(a-1))` from `λ` to `a*λ`, and
`(a+1)*λ^2/2` beyond that.

The scalar proximal operator selects a global minimizer. At an exact tie it
chooses the smaller magnitude, with positive zero at a zero minimizer.
A zero level gives the zero function and an identity prox. An infinite level
gives the indicator of zero, with a constant-zero prox even at a zero step.
An infinite shape gives the ℓ₁ limit. At a positive finite level, a zero
step gives the identity and an infinite step gives the constant-zero map.

Direct array oracles validate scalar and per-entry steps before writing. Use
a nonnegative, non-NaN step with `prox_block` and as the effective step
through `Scaled`. A symmetric sparse iterate output takes a scalar step.
See the [numerical contract](@ref scad-endpoints) for endpoint details.
"""
struct SCAD{T<:Real} <: SeparableFunction
  λ::T
  a::T

  function SCAD{T}(λ::T, a::T) where {T<:Real}
    _checknonneg("λ", λ)
    a > 2 || throw(ArgumentError("a must exceed 2, got $a"))
    new{T}(λ, a)
  end
end
SCAD(λ::Real, a::Real = 3.7) = SCAD(promote(λ, a)...)
SCAD(λ::T, a::T) where {T<:Real} = SCAD{T}(λ, a)

iszeropreserving(::SCAD) = true
restrict(f::SCAD, ::AbstractVector{<:Integer}) = f

@inline value_entry(f::SCAD, v::Real, ::Integer) = _scad_value_entry(_scad_value, f, v)
@inline _stored_value_entry(f::SCAD, v::Real, ::Integer) = _scad_value_entry(_scad_stored_value, f, v)

@inline function _scad_value_entry(kernel::K, f::SCAD, v::T) where {K, T<:Real}
  S = float(T)
  iszero(f.λ) && return zero(S)
  R = _working_type(S, f.λ, f.a)
  λ, a, t = R(f.λ), R(f.a), abs(R(v))
  S(kernel(λ, a, t))
end

@inline function _scad_value(λ, a, t)
  isinf(λ) && return iszero(t) ? zero(t) : λ * t
  linear = λ * t
  isinf(a) && return linear
  ifelse(t > λ, ifelse(t < a * λ, _scad_middle_value(λ, a, t), _scad_plateau(λ, a)), linear)
end

@inline function _scad_stored_value(λ, a, t)
  isinf(λ) && return iszero(t) ? zero(t) : λ * t
  (!(t > λ) || isinf(a)) && return λ * t
  t < a * λ ? _scad_middle_value(λ, a, t) : _scad_plateau(λ, a)
end

@inline function _scad_middle_value(λ, a, t)
  # Integrate the linear slope from λ. The mean slope in this piece
  # is between λ/2 and λ, and both terms are nonnegative.
  d = t - λ
  fma(d, λ - (d * inv(a - 1)) / 2, λ * λ)
end

@inline _scad_plateau(λ, a) = (((a + 1) / 2) * λ) * λ

function _value_sep(f::SCAD{T}, x::SymmetricSparseIterate{T}) where {T<:Union{Float32, Float64}}
  _checkparams(f, x)
  p = x.p
  s = zero(T)
  @inbounds for (i, v) in zip(activecoordinates(x), activevalues(x))
    r, c = tril_cartesian(p, i)
    q = _stored_value_entry(f, v, i)
    # SCAD is position independent. Keep both additions in storage order.
    s += q
    r == c || (s += q)
  end
  s
end

@inline prox_block(f::SCAD, v::Real, ::Integer, γ::Real) = _scad_prox_entry(f, v, γ)

@inline prox_block(f::SCAD{T}, x::T, ::Integer, γ::T) where {T<:Union{Float32, Float64}} =
  _scad_prox_ieee(f, x, γ)

# The public default is the integer 1. Keep its existing Float64 working
# precision, including for Float32 data and the promoted default shape.
@inline prox_block(f::SCAD{P}, x::T, ::Integer, γ::Int) where {P<:Union{Float32, Float64}, T<:Union{Float32, Float64}} =
  γ == 1 ? _scad_prox_ieee(f, x, 1.0) : _scad_prox_entry(f, x, γ)

@inline prox_block(f::SCAD{Float64}, x::Float32, ::Integer, γ::Union{Float32, Float64}) =
  _scad_prox_ieee(f, x, Float64(γ))

@inline function _scad_prox_ieee(f::SCAD, x::T, γ::W) where {T, W}
  λ, a = W(f.λ), W(f.a)
  iszero(λ) && return x
  curvature = (a - γ) - 1
  ratio = γ / curvature
  if (λ >= floatmin(W) / eps(W)) & (curvature > 0) & (ratio > 0) & isfinite(a * λ)
    # The middle root is below soft in the first piece and above t in the
    # last piece. Clamp it against soft, t and zero to combine the pieces.
    t = abs(W(x))
    soft = fma(-γ, λ, t)
    root = if ratio <= 1
      d = soft - λ
      fma(ratio, ifelse(d > 0, d, zero(W)), soft)
    else
      # Near the curvature transition, form the residual at the upper
      # join in one rounding to avoid amplifying rounding of soft.
      middle = fma(ratio, fma(-a, λ, t), t)
      ifelse(middle > soft, middle, soft)
    end
    bounded = ifelse(root < t, root, t)
    u = T(ifelse(bounded > 0, bounded, zero(W)))
    return ifelse(isfinite(x), copysign(u, x) + zero(T), x)
  end
  _scad_prox_entry(f, x, γ)
end

@noinline function _scad_prox_entry(f::SCAD, v::Real, γ::Real)
  x = float(v)
  λ = f.λ
  iszero(λ) && return x
  (isinf(λ) || isinf(γ)) && return zero(x)
  (iszero(γ) || !isfinite(x)) && return x
  R = _working_type(typeof(x), γ, λ, f.a)
  u = oftype(x, _scad_magnitude(abs(R(x)), R(λ), R(f.a), R(γ)))
  iszero(u) ? zero(x) : copysign(u, x)
end

Base.@propagate_inbounds function _prox_sep_entry(f::SCAD{T}, v::T, i, γ::Array{T}) where {T<:Union{Float32, Float64}}
  step = γ[i]
  # Short step buffers benefit from skipping the middle root on outer
  # entries. For longer buffers the branchless kernel avoids unpredictable
  # per-entry branches.
  length(γ) > 1024 && return _scad_prox_ieee(f, v, step)
  λ, a = f.λ, f.a
  curvature = (a - step) - 1
  if (λ >= floatmin(T) / eps(T)) & (step > 0) & (curvature > 0) & isfinite(a * λ)
    t = abs(v)
    upper = fma(-a, λ, t)
    (upper >= 0 || isnan(t)) && return v
    soft = fma(-step, λ, t)
    if soft <= λ
      soft <= 0 && return zero(T)
      return copysign(soft, v)
    end
    u = clamp(fma(step / curvature, upper, t), λ, t)
    return copysign(u, v)
  end
  _scad_prox_entry(f, v, step)
end

@inline _scad_magnitude(t, λ, a, γ) = _scad_prox(t, λ, a, γ)
@inline function _scad_magnitude(t::T, λ::T, a::T, γ::T) where {T<:Union{Float32, Float64}}
  t < floatmin(T) / eps(T) && λ < t && return _scad_small_magnitude(t, λ, a, γ)
  _scad_prox(t, λ, a, γ)
end

@noinline function _scad_small_magnitude(t::T, λ::T, a::T, γ::T) where {T}
  # A subnormal difference at the upper join can round to zero before
  # the comparison, even when the exact middle piece has nonzero slope.
  scale = ldexp(one(T), precision(T))
  _scad_prox(t * scale, λ * scale, a, γ) / scale
end

@inline function _scad_prox(t, λ, a, γ)
  soft = fma(-γ, λ, t)
  (t <= λ || isinf(a)) && return max(zero(t), soft)
  upper = fma(-a, λ, t)
  curvature = (a - γ) - 1
  if curvature > 0
    # The derivative is increasing and continuous across the two joins.
    upper >= 0 && return t
    soft <= λ && return max(zero(t), soft)
    ratio = γ / curvature
    # If aλ-t exceeds the range, use the same root anchored at the
    # soft-threshold point, which is already inside [λ,t].
    root = isfinite(upper) ? fma(ratio, upper, t) : fma(ratio, soft - λ, soft)
    return clamp(root, λ, t)
  end
  # The middle is affine or concave. Its endpoints are already covered by
  # the minima on the two closed outer pieces. Below aλ the middle is
  # increasing, and above (1+γ)λ it is decreasing.
  upper < 0 && return max(zero(t), soft)
  soft > λ && return t
  # Here the right minimum is t, whose complete cost is the plateau.
  # Subtract that cost from the left minimum's cost in magnitude units.
  if soft > 0
    # At a tie, γ-a lies in [-1,1). Form the cutoff relative to aλ
    # so that the comparison rounds both sides in the same units.
    return upper <= ((1 + (γ - a)) / 2) * λ ? soft : t
  end
  t / λ <= _scad_zero_threshold(a, γ) ? zero(t) : t
end

@inline function _scad_zero_threshold(a, γ)
  # For a zero left minimum, compare t/λ with sqrt(γ*(a+1)).
  product = fma(γ, a, γ)
  isinf(product) ? sqrt(γ) * sqrt(a + 1) : sqrt(product)
end

@inline function _scad_zero_threshold(a::T, γ::T) where {T<:Union{Float32, Float64}}
  product = fma(γ, a, γ)
  isfinite(product) && return sqrt(product)
  # Scale both factors by the same power of two when their product
  # overflows. Separate square roots can round an exact tie downward.
  scale = T === Float32 ? T(0x1p-64) : T(0x1p-512)
  g = γ * scale
  sqrt(fma(g, a * scale, g * scale)) / scale
end

prox!(y::_ArrayOrSymmetricIterate, f::SCAD, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_checknonneg("γ", γ); invoke(prox!, _SeparableScalarStep, y, f, x, γ))
value_and_prox!(y::_ArrayOrSymmetricIterate, f::SCAD, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_checknonneg("γ", γ); invoke(value_and_prox!, _SeparableScalarStep, y, f, x, γ))

_check_scad_steps(γ::AbstractArray) = _checknonneg("γ", γ)
function _check_scad_steps(γ::Array{T}) where {T<:Union{Float32, Float64}}
  # Check even inactive coordinates. A Boolean reduction vectorizes
  # the pass over a dense step buffer before a stored-entry traversal.
  mapreduce(>=(zero(T)), &, γ; init = true) ||
    throw(ArgumentError("γ must be nonnegative"))
end

function prox!(y::AbstractArray, f::SCAD, x::AbstractArray, γ::AbstractArray)
  _check_oracle_arrays(prox!, f, y, x, γ)
  _check_scad_steps(γ)
  _prox_sep!(y, f, x, γ)
end
function value_and_prox!(y::AbstractArray, f::SCAD, x::AbstractArray, γ::AbstractArray)
  _check_oracle_arrays(value_and_prox!, f, y, x, γ)
  _check_scad_steps(γ)
  _value_and_prox_sep!(y, f, x, γ)
end
