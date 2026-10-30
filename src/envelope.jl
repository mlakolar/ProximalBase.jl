####################################
#
#  Moreau envelopes
#
####################################

"""
    MoreauEnvelope(f, γ)

The Moreau envelope `e(x) = min_z f(z) + ‖z - x‖² / (2γ)` of a convex,
proximable `f`, for `γ > 0`: a smooth function below `f` that approaches it as
`γ → 0`. With `p = prox_{γf}(x)`, its value is `f(p) + ‖x - p‖² / (2γ)` and its
gradient `(x - p) / γ`, Lipschitz with constant `1/γ`, so a value and a gradient
cost one proximal operator of `f` (and no second evaluation of `f`, which comes
from [`value_and_prox!`](@ref)).

It is also proximable, by
`prox_{te}(x) = x + t/(γ + t) ⋅ (prox_{(γ + t)f}(x) - x)`, and separable when
`f` is, so coordinate descent applies to it. The envelope of `δ|⋅|` is the
Huber function, that of the indicator function of a set `C` the squared
distance to `C` divided by `2γ`; [`SqrDistL2`](@ref) also supports nonconvex
sets. The value, the gradient and the proximal
operator use a buffer that `e` owns; do not share `e` between threads, and pass
a gradient output that is not `x`. For a proximal step `t`, `γ` and `t` are
converted to the element type of the output. Their sum must be finite and
positive; otherwise `prox!` and `value_and_prox!` throw an `ArgumentError` and
leave the output unchanged.
"""
struct MoreauEnvelope{F<:AbstractFunction, T<:Real} <: AbstractFunction
  f::F
  γ::T
  scratch::Scratch

  function MoreauEnvelope{F, T}(f::F, γ::T) where {F<:AbstractFunction, T<:Real}
    isconvex(f) || throw(ArgumentError("the Moreau envelope requires a convex f"))
    isproximable(f) || throw(ArgumentError("the Moreau envelope needs a proximable f, but f is a $(typeof(f))"))
    γ > 0 || throw(ArgumentError("γ must be positive, got $γ"))
    new{F, T}(f, γ, Scratch())
  end
end
MoreauEnvelope(f::AbstractFunction, γ::Real) = MoreauEnvelope{typeof(f), typeof(γ)}(f, γ)

issmooth(::Type{<:MoreauEnvelope}) = true
isproximable(::Type{<:MoreauEnvelope}) = true
isseparable(::Type{<:MoreauEnvelope{F}}) where {F} = isseparable(F)
isconvex(::Type{<:MoreauEnvelope{F}}) where {F} = isconvex(F)

# the proximal point of f, in the buffer that e owns
_envelope_buffer(e::MoreauEnvelope, ::Type{T}, x::AbstractArray) where {T} = _array_buffer!(e.scratch, T, size(x))

function _checkwrite(op::O, e::MoreauEnvelope, y::AbstractArray, x::AbstractArray) where {O}
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  if op === gradient! || op === value_and_gradient!
    _checkparams(e, x)
    _checkwrite(op === gradient! ? prox! : value_and_prox!, e.f, y, x)
  else
    _checkparams(e, x)
    p = _checkbuffer(e.scratch, float(eltype(y)), size(x))
    _checkwrite(op, e.f, p, x)
  end
  true
end
_checkvalue(e::MoreauEnvelope, x::AbstractArray) =
  _checkwrite(value_and_prox!, e.f, _checkbuffer(e.scratch, float(eltype(x)), size(x)), x)

# Form the distance in units of sqrt(2γ) before squaring it: an unscaled
# squared distance can overflow or underflow although its energy is finite.
# If even the distance overflows, scale the entries before subtracting only
# where the subtraction itself overflows. Ordinary finite differences retain
# their accuracy, including cancellation between nearby endpoints.
function _envelope_energy(x::AbstractArray, p::AbstractArray, γ::Real)
  r = norm_diff(x, p)
  scale = sqrt(γ) * sqrt(oftype(γ, 2))
  isfinite(r) && return abs2(r / scale)
  q = zero(r)
  @inbounds for i in eachindex(x, p)
    d = x[i] - p[i]
    z = isfinite(d) ? d / scale : x[i] / scale - p[i] / scale
    q = hypot(q, abs(z))
  end
  abs2(q)
end

@inline function _envelope_energy(x::Real, p::Real, γ::Real)
  scale = sqrt(γ) * sqrt(oftype(γ, 2))
  d = x - p
  abs2(isfinite(d) ? d / scale : x / scale - p / scale)
end

# The difference can overflow before division even for a finite derivative.
@inline function _envelope_derivative(x, p, γ)
  d = x - p
  isfinite(d) ? d / γ : x / γ - p / γ
end

function (e::MoreauEnvelope)(x::AbstractArray)
  T = float(eltype(x))
  γ = convert(T, e.γ)
  p = _envelope_buffer(e, T, x)
  v = value_and_prox!(p, e.f, x, γ)
  _value_at(v + _envelope_energy(x, p, γ), x)
end

function value_and_gradient!(g::AbstractArray, e::MoreauEnvelope, x::AbstractArray)
  _checkwrite(value_and_gradient!, e, g, x)
  γ = convert(float(eltype(g)), e.γ)
  v = value_and_prox!(g, e.f, x, γ)
  energy = _envelope_energy(x, g, γ)
  g .= _envelope_derivative.(x, g, γ)
  eltype(g) === eltype(x) ? _value_at(v + energy, x) : e(x)
end

function gradient!(g::AbstractArray, e::MoreauEnvelope, x::AbstractArray)
  _checkwrite(gradient!, e, g, x)
  γ = convert(float(eltype(g)), e.γ)
  prox!(g, e.f, x, γ)
  g .= _envelope_derivative.(x, g, γ)
end

@inline function _envelope_step(γ::T, t::T) where {T}
  combined = γ + t
  isfinite(combined) && combined > zero(T) ||
    throw(ArgumentError("the envelope parameter plus proximal step must be finite and positive in working precision; rescale the objective and step"))
  combined
end

# With P = prox_{(γ+t)f}(x), the result is y = x + t/(γ + t) (P - x); since
# (y - P)/γ = (x - P)/(γ + t) ∈ ∂f(P), P is also prox_{γf}(y), so the value at y
# is f(P) + ‖y - P‖²/(2γ), with ‖y - P‖ = γ/(γ + t) ‖x - P‖
# The step is checked after the sizes and parameters and before the inner
# function, as before the pre-write checks.
function prox!(y::AbstractArray, e::MoreauEnvelope, x::AbstractArray, t::Real)
  _checksize(y, x)
  _checkparams(e, x)
  T = float(eltype(y))
  γ, t = convert(T, e.γ), convert(T, t)
  combined = _envelope_step(γ, t)
  _checkwrite(prox!, e, y, x)
  P = _envelope_buffer(e, T, x)
  prox!(P, e.f, x, combined)
  y .= _ridge_blend.(x, P, γ / combined, t / combined)
end

function value_and_prox!(y::AbstractArray, e::MoreauEnvelope, x::AbstractArray, t::Real)
  _checksize(y, x)
  _checkparams(e, x)
  T = float(eltype(y))
  γ, t = convert(T, e.γ), convert(T, t)
  combined = _envelope_step(γ, t)
  _checkwrite(value_and_prox!, e, y, x)
  P = _envelope_buffer(e, T, x)
  v = value_and_prox!(P, e.f, x, combined)
  a, b = γ / combined, t / combined
  y .= _ridge_blend.(x, P, a, b)
  _value_at(v + _envelope_energy(y, P, γ), y)
end

Base.@propagate_inbounds function prox_entry(e::MoreauEnvelope, v::Real, i::Integer, t::Real)
  T = typeof(float(v))
  γ, step = convert(T, e.γ), convert(T, t)
  combined = _envelope_step(γ, step)
  p = prox_entry(e.f, v, i, combined)
  _ridge_blend(v, p, γ / combined, step / combined)
end
_checkparams(e::MoreauEnvelope, x) = _checkparams(e.f, x)

# the envelope is smooth: a coordinate at zero is optimal when the gradient of
# the smooth part cancels the envelope's derivative there, -prox_{γfᵢ}(0)/γ
function kkt_violation(e::MoreauEnvelope, g::AbstractArray, i::Integer)
  γ = e.γ
  abs(g[i] - prox_entry(e.f, zero(float(eltype(g))), i, γ) / γ)
end
# the dropped terms are minᵧ fᵢ(y) + y²/(2γ), finite for a proper fᵢ
restrict(e::MoreauEnvelope, S::AbstractVector{<:Integer}) = MoreauEnvelope(_restrict_unchecked(e.f, S), e.γ)

lipschitz(e::MoreauEnvelope) = inv(float(e.γ))

####################################
#  the envelope of a scalar function, as a scalar loss
####################################

"""
    ProximalBase.Smoothed(φ, γ)

The Moreau envelope of a convex scalar function `φ` of the residual
`r = η - y`, as a [`ProximalBase.ScalarLoss`](@ref) for [`GLMLoss`](@ref):

    ℓ(η, y) = min_t φ(t) + (t - r)² / (2γ),   γ > 0.

`φ` is a [`SeparableFunction`](@ref) with scalar parameters, of which the term
of entry 1 is used. With `p = prox_{γφ}(r)`, the value is
`φ(p) + (r - p)²/(2γ)`, the derivative `(r - p)/γ` and the second derivative
`(1 - ∂p/∂r)/γ`, which is `1/γ` or `0` on the pieces of a piecewise linear
proximal operator; the curvature is at most `1/γ`. So a nonsmooth loss such as
the check loss of quantile regression or the hinge, smoothed, runs in
coordinate descent with the curvature bound `1/γ`, and in proximal Newton.
`Smoothed(NormL1(δ), 1)` is [`ProximalBase.Huber`](@ref)`(δ)`.
"""
struct Smoothed{F<:SeparableFunction, T<:Real} <: ScalarLoss
  φ::F
  γ::T

  function Smoothed{F, T}(φ::F, γ::T) where {F<:SeparableFunction, T<:Real}
    isconvex(φ) || throw(ArgumentError("Smoothed requires a convex scalar function"))
    γ > 0 || throw(ArgumentError("γ must be positive, got $γ"))
    new{F, T}(φ, γ)
  end
end
Smoothed(φ::SeparableFunction, γ::Real) = Smoothed{typeof(φ), typeof(γ)}(φ, γ)

@inline function value_and_deriv(ℓ::Smoothed, η::Real, y::Real)
  r = float(η - y)
  γ = convert(typeof(r), ℓ.γ)
  p = prox_entry(ℓ.φ, r, 1, γ)
  (convert(typeof(r), value_entry(ℓ.φ, p, 1)) + _envelope_energy(r, p, γ),
   _envelope_derivative(r, p, γ))
end
(ℓ::Smoothed)(η::Real, y::Real) = value_and_deriv(ℓ, η, y)[1]
deriv(ℓ::Smoothed, η::Real, y::Real) = value_and_deriv(ℓ, η, y)[2]
function deriv2(ℓ::Smoothed, η::Real, y::Real)
  r = float(η - y)
  γ = convert(typeof(r), ℓ.γ)
  (1 - _prox_slope(ℓ.φ, r, γ)) / γ
end
curvature(ℓ::Smoothed) = inv(float(ℓ.γ))

# ∂prox_{γφ₁}(r)/∂r ∈ [0, 1]: exact for the piecewise linear proxes, by a central
# difference otherwise
_prox_slope(φ::NormL1, r::T, γ::T) where {T} = abs(r) > _weight_coefficient(γ, φ.λ0, _entry(φ.w, 1)) ? one(T) : zero(T)
_prox_slope(φ::NormL1Nonneg, r::T, γ::T) where {T} = r > _weight_coefficient(γ, φ.λ0, _entry(φ.w, 1)) ? one(T) : zero(T)
_prox_slope(φ::ElasticNet, r::T, γ::T) where {T} =
  abs(r) > _weight_coefficient(γ, φ.λ1, _entry(φ.w, 1)) ? inv(1 + γ * convert(T, φ.λ2)) : zero(T)
_prox_slope(φ::IndBox, r::T, ::T) where {T} = _entry(φ.l, 1) < r < _entry(φ.u, 1) ? one(T) : zero(T)
_prox_slope(::IndNonnegative, r::T, ::T) where {T} = r > 0 ? one(T) : zero(T)
function _prox_slope(φ::SeparableFunction, r::T, γ::T) where {T}
  h = sqrt(eps(T)) * max(one(T), abs(r))
  s = (prox_entry(φ, r + h, 1, γ) - prox_entry(φ, r - h, 1, γ)) / (2h)
  clamp(convert(T, s), zero(T), one(T))
end

function _checkstep(op::O, e::MoreauEnvelope, y::AbstractArray, x::AbstractArray, γ::Real) where {O}
  T = float(eltype(y))
  combined = _envelope_step(convert(T, e.γ), convert(T, γ))
  p = _checkbuffer(e.scratch, T, size(x))
  _checkstep(op, e.f, p, x, combined)
end
