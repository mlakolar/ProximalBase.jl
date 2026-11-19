"""
    MCP(λ, a = 3)

The separable minimax concave penalty with level `λ ≥ 0` and shape `a > 0`.
For finite positive parameters, the entry penalty at magnitude `t` is
`λ*t - t^2/(2a)` up to `t = a*λ` and `a*λ^2/2` thereafter, including at
infinite entries. A NaN entry gives a NaN value at any positive level.
At `a = Inf` and finite positive level, the penalty is `λ*abs(x)` per entry.

At a positive finite step `γ < a` the proximal map uses firm thresholding
between `γ*λ` and `a*λ` (soft thresholding for infinite `a`). At a finite
step `γ ≥ a` it keeps an entry when `abs(x) > λ*sqrt(a*γ)` and returns
positive zero otherwise, including on an exact tie. Both rules preserve
NaN and signed infinities.

A zero level gives zero value even at nonfinite entries and the identity
proximal map at every valid step. An infinite level gives zero value at
zero and infinite value at nonzero numbers, and maps all inputs to positive
zero at every valid step. At a positive finite level, a zero step gives the
identity and an infinite step gives positive zero, including for nonfinite
inputs. Identity maps preserve signed zeros.

Direct array and iterate oracles reject negative and NaN steps before
writing. The scalar hook `prox_block` and steps forwarded by `Scaled`
require nonnegative, non-NaN steps. Symmetric sparse iterates use scalar
steps and count off-diagonal values twice.
"""
struct MCP{T<:Real} <: SeparableFunction
  λ::T
  a::T

  function MCP{T}(λ::T, a::T) where {T<:Real}
    _checknonneg("λ", λ)
    a > 0 || throw(ArgumentError("a must be positive, got $a"))
    new{T}(λ, a)
  end
end

function MCP(λ::Real, a::Real = 3)
  _checknonneg("λ", λ)
  a > 0 || throw(ArgumentError("a must be positive, got $a"))
  level, shape = promote(λ, a)
  MCP{typeof(level)}(level, shape)
end

iszeropreserving(::MCP) = true
entry_parameters(::MCP) = ()
restrict(f::MCP, ::AbstractVector{<:Integer}) = f

@inline function value_entry(f::MCP, v::T, ::Integer) where {T<:Real}
  S = float(T)
  (iszero(f.λ) || iszero(v)) && return zero(S)
  isnan(v) && return S(v)
  isinf(f.λ) && return S(Inf)
  λ, a, t = S(f.λ), S(f.a), abs(S(v))
  isinf(a) && return λ * t
  k = a * λ
  t < k ? t * (λ - (t / a) / 2) : _mcp_saturation(λ, a, k)
end

@inline function _mcp_saturation(λ::T, a::T, k::T) where {T<:AbstractFloat}
  half = k / 2
  floatmin(T) <= half < T(Inf) && return λ * half
  _mcp_saturation_scaled(λ, a)
end

@noinline function _mcp_saturation_scaled(λ::T, a::T) where {T<:AbstractFloat}
  ml, el = _frexp(λ)
  ma, ea = _frexp(a)
  _scale_value((ml * ml) * ma, _power_factors(T, 2el + ea - 1))
end

struct _MCPOrdinaryValue{T} <: SeparableFunction
  λ::T
  a::T
  kink::T
  plateau::T
end

iszeropreserving(::_MCPOrdinaryValue) = true
@inline function value_entry(f::_MCPOrdinaryValue{T}, v::T, ::Integer) where {T}
  t = abs(v)
  p = t * (f.λ - (t / f.a) / 2)
  ifelse(isnan(v), v, ifelse(t < f.kink, p, f.plateau))
end

@inline function _mcp_ordinary_value(f::MCP{T}) where {T}
  k = f.a * f.λ
  half = k / 2
  floatmin(T) <= half < T(Inf) || return nothing
  _MCPOrdinaryValue(f.λ, f.a, k, f.λ * half)
end

function _value_dense(f::MCP{T}, x::Array{T}) where {T<:Union{Float32,Float64}}
  g = _mcp_ordinary_value(f)
  g === nothing && return _sum_entries(value_entry, f, x, T)
  _sum_entries(value_entry, g, x, T)
end

function _value_sep(f::MCP{T}, x::BlockIterate{T}) where {T<:Union{Float32,Float64}}
  g = _mcp_ordinary_value(f)
  g === nothing && return invoke(_value_sep, Tuple{Any,BlockIterate}, f, x)
  _value_sep(g, x)
end

@inline function prox_block(f::MCP, v::T, ::Integer, γ::Real) where {T<:Real}
  S = float(T)
  x = S(v)
  iszero(f.λ) && return x
  (isinf(f.λ) || isinf(γ)) && return zero(S)
  (iszero(γ) || !isfinite(x)) && return x
  t = abs(x)
  iszero(t) && return zero(S)
  λ, a, s = S(f.λ), S(f.a), S(γ)
  isnan(s) && return s
  if s < a
    low = s * λ
    t < low && return zero(S)
    if isinf(a)
      r = fma(-s, λ, t)
      iszero(r) && return _mcp_firm_scaled(x, t, s, λ, one(S))
      return r > 0 ? copysign(r, x) : zero(S)
    end
    k = a * λ
    (t > k || (t == k && _mcp_at_or_above_kink(t, a, λ))) && return x
    d = (a - s) / a
    r = fma(-s, λ, t)
    abs(r) < floatmin(S) && return _mcp_firm_scaled(x, t, s, λ, d)
    return r > 0 ? copysign(min(t, r / d), x) : zero(S)
  end
  _mcp_keeps(t, λ, a, s) ? x : zero(S)
end

@inline function _mcp_at_or_above_kink(t, a, λ)
  r = fma(-a, λ, t)
  # Equality with the rounded product needs the sign of the unrounded
  # difference. Near underflow the native residual can round to zero.
  !iszero(r) && return r > 0
  m, _ = _mcp_scaled_residual(t, a, λ)
  m >= 0
end

@noinline function _mcp_firm_scaled(x::T, t::T, γ::T, λ::T, d::T) where {T<:AbstractFloat}
  m, e = _mcp_scaled_residual(t, γ, λ)
  m > 0 || return zero(T)
  u = _scale_value(m / d, _power_factors(T, e))
  copysign(min(t, u), x)
end

function _mcp_scaled_residual(t::T, γ::T, λ::T) where {T<:AbstractFloat}
  mt, et = _frexp(t)
  mg, eg = _frexp(γ)
  ml, el = _frexp(λ)
  shift = eg + el - et
  shift > 2 && return (-one(T), et)
  # A smaller product changes the normalized residual by less than half an
  # ulp. Its contribution cannot affect the sign in these units.
  shift < -precision(T) - 1 && return (mt, et)
  fma(-mg, ldexp(ml, shift), mt), et
end

# The affine/concave inner objective minimizes at an endpoint. The outer
# minimum is max(t, aλ). For t ≤ aλ its cost relative to zero is
# aλ * (γλ + aλ - 2t) / (2γ) ≥ 0. Otherwise t wins against aλ and
# its cost relative to zero is (aγλ² - t²) / (2γ).
@inline function _mcp_keeps(t::T, λ::T, a::T, γ::T) where {T<:AbstractFloat}
  p = t * t
  aa = a * γ
  ll = λ * λ
  q = aa * ll
  _mcp_keeps_rounded(p, q, aa, ll, t, λ, a, γ)
end

@inline _mcp_keeps_rounded(p, q, aa, ll, t, λ, a, γ) = p > q
@inline function _mcp_keeps_rounded(p::T, q::T, aa::T, ll::T, t::T, λ::T, a::T, γ::T) where {T<:Union{Float32,Float64}}
  # One rounded multiplication in p and three in q. With normal products,
  # 8eps times their larger magnitude bounds the combined rounding error.
  lo, hi = minmax(p, q)
  if lo >= floatmin(T) && isfinite(hi) && min(aa, ll) >= floatmin(T) && hi - lo > (8eps(T)) * hi
    return p > q
  end
  # A factor can become zero or infinite when converted from another type.
  (0 < λ < T(Inf) && 0 < a < T(Inf) && 0 < γ < T(Inf)) || return p > q
  _mcp_keeps_exact(t, λ, a, γ)
end

@noinline function _mcp_keeps_exact(t::T, λ::T, a::T, γ::T) where {T<:Union{Float32,Float64}}
  mt, et = _frexp(t)
  ml, el = _frexp(λ)
  ma, ea = _frexp(a)
  mg, eg = _frexp(γ)
  shift = 2et - 2el - ea - eg
  # The normalized squares are in [1/4,1), the four-factor product in [1/16,1).
  shift >= 2 && return true
  shift <= -4 && return false
  n = precision(T)
  st = unsafe_trunc(UInt64, ldexp(mt, n))
  sl = unsafe_trunc(UInt64, ldexp(ml, n))
  sa = unsafe_trunc(UInt64, ldexp(ma, n))
  sg = unsafe_trunc(UInt64, ldexp(mg, n))
  p = widemul(st, st)
  # Compare the at-most-212-bit product with p shifted by at most 107 bits.
  qhi, qlo = _mcp_widemul(widemul(sl, sl), widemul(sa, sg))
  nshift = 2n + shift
  (p >> (128 - nshift), p << nshift) > (qhi, qlo)
end

@inline function _mcp_widemul(x::UInt128, y::UInt128)
  xhi, xlo = UInt64(x >> 64), x % UInt64
  yhi, ylo = UInt64(y >> 64), y % UInt64
  lo = widemul(xlo, ylo)
  # Each high limb has at most 42 bits. The cross terms fit in 107 bits.
  cross = widemul(xhi, ylo) + widemul(xlo, yhi)
  qlo = lo + (cross << 64)
  qhi = widemul(xhi, yhi) + (cross >> 64) + (qlo < lo)
  qhi, qlo
end

# Prepare scalar thresholds once for a long ordinary array. The generic
# entry loop then has no range checks or fallback calls on each entry.
struct _MCPHardCutoff{T} <: SeparableFunction
  cutoff::T
end

@inline prox_block(f::_MCPHardCutoff{T}, v::T, ::Integer, ::Real) where {T} =
  ifelse(abs(v) <= f.cutoff, zero(v), v)

function _mcp_hard_cutoff(f::MCP{T}, γ::T) where {T}
  a, λ = f.a, f.λ
  product = a * γ
  floatmin(T) <= product <= floatmax(T) || return f
  t = λ * sqrt(product)
  floatmin(T) <= t <= floatmax(T) || return f
  # Certify both sides of the cutoff. The two rounded operations that form
  # t can land on either side of the exact threshold.
  for _ in 1:4
    if _mcp_keeps_exact(t, λ, a, γ)
      t = prevfloat(t)
    else
      next = nextfloat(t)
      (isinf(next) || _mcp_keeps_exact(next, λ, a, γ)) && return _MCPHardCutoff(t)
      t = next
    end
  end
  f
end

struct _MCPFirmThresholds{T} <: SeparableFunction
  low::T
  correction::T
  upper::T
  gain::T
end

@inline _mcp_firm_product_min(::Type{T}) where {T} = ldexp(floatmin(T), precision(T) + 1)

@inline function prox_block(f::_MCPFirmThresholds{T}, v::T, ::Integer, ::Real) where {T}
  t = abs(v)
  u = ((t - f.low) + f.correction) * f.gain
  ifelse((t >= f.upper) | isnan(v), v,
         ifelse(u > 0, copysign(min(t, u), v), zero(v)))
end

function _mcp_firm_thresholds(f::MCP{T}, γ::T) where {T}
  λ, a = f.λ, f.a
  low, k = γ * λ, a * λ
  # A product of two p-bit significands has at most 2p bits. At this
  # magnitude its low bit is representable even in a subnormal residual.
  _mcp_firm_product_min(T) <= low <= k <= floatmax(T) || return f
  correction = fma(-γ, λ, low)
  upper = fma(a, λ, -k) > 0 ? nextfloat(k) : k
  # Near low, t - low is exact by Sterbenz's lemma. Adding the product
  # residual before scaling retains the side of the true threshold.
  _MCPFirmThresholds(low, correction, upper, a / (a - γ))
end

function _prox_sep_dense!(y::AbstractArray, f::MCP{T}, x::AbstractArray{T}, γ::T) where {T<:Union{Float32,Float64}}
  if length(x) >= 64 && 0 < γ < T(Inf) && 0 < f.λ < T(Inf) && isfinite(f.a)
    g = γ < f.a ? _mcp_firm_thresholds(f, γ) : _mcp_hard_cutoff(f, γ)
    return _prox_entries!(y, g, x, γ)
  end
  _prox_entries!(y, f, x, γ)
end

# Certify and write one block at a time so the step and input data stay hot.
# The public route has already checked the entire step array before any write.
function _prox_sep_dense!(y::Array{T}, f::MCP{T}, x::Array{T}, γ::Array{T}) where {T<:Union{Float32,Float64}}
  k = f.a * f.λ
  length(x) >= 64 && _mcp_firm_product_min(T) <= k <= floatmax(T) ||
    return _prox_entries!(y, f, x, γ)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  _mcp_steps!(y, f, x, γ, k)
end

@noinline function _mcp_steps!(y, f::MCP{T}, x, γ, k) where {T}
  a, λ = f.a, f.λ
  upper = fma(a, λ, -k) > 0 ? nextfloat(k) : k
  ll = λ * λ
  n = length(x)
  for first in 1:256:n
    inds = first:(first + min(255, n - first))
    firm = @inbounds γ[first] < a
    if firm && _mcp_firm_steps(γ, λ, a, inds)
      _mcp_write_firm_steps!(y, x, γ, λ, a, upper, inds)
    elseif !firm && floatmin(T) <= ll < T(Inf) && _mcp_hard_steps(x, γ, a, ll, inds)
      _mcp_write_hard_steps!(y, x, γ, a, ll, inds)
    else
      @inbounds for i in inds
        y[i] = prox_block(f, x[i], i, γ[i])
      end
    end
  end
  y
end

@inline function _mcp_firm_steps(γ, λ::T, a, inds) where {T}
  ordinary = true
  @inbounds @simd for i in inds
    s = γ[i]
    ordinary &= (s < a) & (s * λ >= _mcp_firm_product_min(T))
  end
  ordinary
end

@inline function _mcp_write_firm_steps!(y, x, γ, λ, a, upper, inds)
  @inbounds @simd for i in inds
    v, s = x[i], γ[i]
    t = abs(v)
    # The product bound keeps the exact residual's low bit representable.
    r = fma(-s, λ, t)
    u = r * (a / (a - s))
    y[i] = ifelse((t >= upper) | isnan(v), v,
                  ifelse(u > 0, copysign(min(t, u), v), zero(v)))
  end
end

@inline function _mcp_hard_steps(x, γ, a::T, ll, inds) where {T}
  ordinary = true
  @inbounds @simd for i in inds
    s, v = γ[i], x[i]
    aa, p = a * s, v * v
    q = aa * ll
    # Same error margin as the scalar comparison with normal products.
    separated = (floatmin(T) <= p) & (p < T(Inf)) & (floatmin(T) <= q) &
                (q < T(Inf)) & (floatmin(T) <= aa) &
                (abs(p - q) > (8eps(T)) * max(p, q))
    ordinary &= (a <= s) & (s < T(Inf)) & (separated | iszero(v) | !isfinite(v))
  end
  ordinary
end

@inline function _mcp_write_hard_steps!(y, x, γ, a, ll, inds)
  @inbounds @simd for i in inds
    v = x[i]
    p, q = v * v, (a * γ[i]) * ll
    y[i] = ifelse(!isfinite(v) | (p > q), v, zero(v))
  end
end

prox!(y::_ArrayOrSymmetricIterate, f::MCP, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_checknonneg("γ", γ); invoke(prox!, _SeparableScalarStep, y, f, x, γ))
value_and_prox!(y::_ArrayOrSymmetricIterate, f::MCP, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_checknonneg("γ", γ); invoke(value_and_prox!, _SeparableScalarStep, y, f, x, γ))

function prox!(y::AbstractArray, f::MCP, x::AbstractArray, γ::AbstractArray)
  _check_oracle_arrays(prox!, f, y, x, γ)
  _checknonneg("γ", γ)
  _prox_sep!(y, f, x, γ)
end

function value_and_prox!(y::AbstractArray, f::MCP, x::AbstractArray, γ::AbstractArray)
  _check_oracle_arrays(value_and_prox!, f, y, x, γ)
  _checknonneg("γ", γ)
  _value_and_prox_sep!(y, f, x, γ)
end
