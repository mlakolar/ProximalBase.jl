####################################
#
#  Losses of residuals and margins: the check, hinge and ε-insensitive losses
#
####################################

"""
    CheckLoss(τ)
    CheckLoss(τ, w)

The check loss of quantile regression, `f(x) = Σᵢ wᵢ ρ_τ(xᵢ)` with
`ρ_τ(u) = max(τu, (τ - 1)u)` for `τ ∈ (0, 1)`, and all `wᵢ = 1` when `w` is
omitted. Its proximal operator moves an entry above `γwᵢτ` down by `γwᵢτ`, one
below `-γwᵢ(1 - τ)` up by `γwᵢ(1 - τ)`, and sets those in between to zero.

It is a function of residuals. Since `ρ_τ(-u) = ρ_{1-τ}(u)`, the loss
`Σᵢ ρ_τ(yᵢ - zᵢ)` of predictions `z` of responses `y` is
`Translate(CheckLoss(1 - τ), y)`, which a splitting method such as ADMM applies
to `z = Xβ`. Its conjugate is the indicator of the box `[wᵢ(τ - 1), wᵢτ]`, the
dual feasible set of quantile regression. For coordinate descent and Newton
steps, [`ProximalBase.SmoothCheck`](@ref) and
[`ProximalBase.ConvolutionCheck`](@ref) are smooth versions of the loss of `Xβ`
for [`GLMLoss`](@ref).
"""
struct CheckLoss{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  τ::T
  w::W

  function CheckLoss{T, W}(τ::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    0 < τ < 1 || throw(ArgumentError("τ must be in (0, 1), got $τ"))
    _checknonneg("the weights", w)
    new{T, W}(τ, w)
  end
end
CheckLoss(τ::Real, w::Union{Nothing, AbstractArray} = nothing) = CheckLoss{typeof(τ), typeof(w)}(τ, w)

isconvex(::Type{<:CheckLoss}) = true
iszeropreserving(::CheckLoss) = true
_checkparams(f::CheckLoss, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::CheckLoss, v::T, i::Integer) where {T<:Real}
  τ = convert(float(T), f.τ)
  _entry(f.w, i) * (v >= 0 ? τ * v : (τ - 1) * v)
end
Base.@propagate_inbounds function prox_entry(f::CheckLoss, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  s, τ, u = convert(S, γ * _entry(f.w, i)), convert(S, f.τ), S(v)
  u > s * τ ? u - s * τ : (u < -s * (1 - τ) ? u + s * (1 - τ) : zero(S))
end

# from zero, the objective decreases upward at rate -gᵢ - wᵢτ and downward at
# rate gᵢ - wᵢ(1 - τ)
function kkt_violation(f::CheckLoss, g::AbstractArray, i::Integer)
  w = _entry(f.w, i)
  max(-g[i] - w * f.τ, g[i] - w * (1 - f.τ))
end
restrict(f::CheckLoss, S::AbstractVector{<:Integer}) = CheckLoss(f.τ, _restrict_entries(f.w, S))

"""
    HingeLoss()
    HingeLoss(y)
    HingeLoss(y, w)

The hinge loss of support vector machines, `f(x) = Σᵢ wᵢ max(0, 1 - yᵢxᵢ)`, for
nonzero labels `yᵢ`, usually `±1`, and weights `wᵢ`, all 1 when omitted. Without
labels, `x` holds the margins. For the label `1`, its proximal operator moves an
entry below `1 - γwᵢ` up by `γwᵢ`, puts one in between at 1, and keeps one above
1; a label `yᵢ` acts on the margin `yᵢxᵢ`, with the step `γwᵢyᵢ²`. Its conjugate
is `Σᵢ sᵢ/yᵢ` on the box `-wᵢ ≤ sᵢ/yᵢ ≤ 0`, and `Inf` off it.
"""
struct HingeLoss{Y<:Union{Nothing, AbstractArray}, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  y::Y
  w::W

  function HingeLoss{Y, W}(y::Y, w::W) where {Y<:Union{Nothing, AbstractArray}, W<:Union{Nothing, AbstractArray}}
    y === nothing || all(!iszero, y) || throw(ArgumentError("the labels must be nonzero"))
    _checknonneg("the weights", w)
    new{Y, W}(y, w)
  end
end
HingeLoss(y::Union{Nothing, AbstractArray} = nothing, w::Union{Nothing, AbstractArray} = nothing) =
  HingeLoss{typeof(y), typeof(w)}(y, w)

isconvex(::Type{<:HingeLoss}) = true
_checkparams(f::HingeLoss, x) = (_checkparam(f.y, x); _checkparam(f.w, x); true)

Base.@propagate_inbounds value_entry(f::HingeLoss, v::T, i::Integer) where {T<:Real} =
  _entry(f.w, i) * max(1 - _entry(f.y, i) * v, zero(float(T)))
# the proximal operator of γwᵢ max(0, 1 - u) at the margin u = yᵢv, mapped back
Base.@propagate_inbounds function prox_entry(f::HingeLoss, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  y = convert(S, _entry(f.y, i))
  s = convert(S, γ * _entry(f.w, i)) * y^2
  u = y * S(v)
  (u >= 1 ? u : (u < 1 - s ? u + s : one(S))) / y
end

# f is differentiable at zero, with derivative -wᵢyᵢ
kkt_violation(f::HingeLoss, g::AbstractArray, i::Integer) = abs(g[i] - _entry(f.w, i) * _entry(f.y, i))
restrict(f::HingeLoss, S::AbstractVector{<:Integer}) = HingeLoss(_restrict_entries(f.y, S), _restrict_entries(f.w, S))

"""
    EpsInsensitiveLoss(ε)
    EpsInsensitiveLoss(ε, w)

The ε-insensitive loss of support vector regression,
`f(x) = Σᵢ wᵢ max(0, |xᵢ| - ε)` for `ε ≥ 0`, with all `wᵢ = 1` when `w` is
omitted: residuals within `ε` cost nothing. Its proximal operator keeps an entry
with `|v| ≤ ε`, puts one with `ε < |v| ≤ ε + γwᵢ` at `±ε`, and moves a larger one
towards zero by `γwᵢ`. As for [`CheckLoss`](@ref), the loss of predictions `z`
of `y` is `Translate(EpsInsensitiveLoss(ε), y)`. Its conjugate is `ε Σᵢ |sᵢ|` on
the box `|sᵢ| ≤ wᵢ`.
"""
struct EpsInsensitiveLoss{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  ε::T
  w::W

  function EpsInsensitiveLoss{T, W}(ε::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    _checknonneg("ε", ε)
    _checknonneg("the weights", w)
    new{T, W}(ε, w)
  end
end
EpsInsensitiveLoss(ε::Real, w::Union{Nothing, AbstractArray} = nothing) = EpsInsensitiveLoss{typeof(ε), typeof(w)}(ε, w)

isconvex(::Type{<:EpsInsensitiveLoss}) = true
iszeropreserving(::EpsInsensitiveLoss) = true
_checkparams(f::EpsInsensitiveLoss, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds value_entry(f::EpsInsensitiveLoss, v::T, i::Integer) where {T<:Real} =
  _entry(f.w, i) * max(abs(v) - convert(float(T), f.ε), zero(float(T)))
Base.@propagate_inbounds function prox_entry(f::EpsInsensitiveLoss, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  s, ε, u = convert(S, γ * _entry(f.w, i)), convert(S, f.ε), S(v)
  a = abs(u)
  a <= ε ? u : (a <= ε + s ? copysign(ε, u) : u - copysign(s, u))
end

# zero is inside the flat part when ε > 0; with ε = 0 the loss is wᵢ|xᵢ|
kkt_violation(f::EpsInsensitiveLoss, g::AbstractArray, i::Integer) =
  f.ε > 0 ? abs(g[i]) : abs(g[i]) - _entry(f.w, i)
restrict(f::EpsInsensitiveLoss, S::AbstractVector{<:Integer}) = EpsInsensitiveLoss(f.ε, _restrict_entries(f.w, S))

# the values of their conjugates, for Conjugate, within the tolerance of the
# indicator functions (_within is in calculus.jl)
function _conjugate_value(f::CheckLoss, s::AbstractArray)
  _checkparams(f, s)
  T = float(eltype(s))
  τ = convert(T, f.τ)
  @inbounds for i in eachindex(IndexLinear(), s)
    w = _entry(f.w, i)
    (_within(s[i], w * τ, T) && _within(-s[i], w * (1 - τ), T)) || return T(Inf)
  end
  zero(T)
end
function _conjugate_value(f::HingeLoss, s::AbstractArray)
  _checkparams(f, s)
  T = float(eltype(s))
  v = zero(T)
  @inbounds for i in eachindex(IndexLinear(), s)
    q = s[i] / _entry(f.y, i)
    (_within(q, zero(T), T) && _within(-q, convert(T, _entry(f.w, i)), T)) || return T(Inf)
    v += q
  end
  v
end
function _conjugate_value(f::EpsInsensitiveLoss, s::AbstractArray)
  _checkparams(f, s)
  T = float(eltype(s))
  v = zero(T)
  @inbounds for i in eachindex(IndexLinear(), s)
    a = abs(s[i])
    _within(a, convert(T, _entry(f.w, i)), T) || return T(Inf)
    v += a
  end
  convert(T, f.ε) * v
end

####################################
#
#  Smooth losses of a linear predictor: the squared hinge and the smoothed
#  check losses, with bounded curvature
#
####################################

"""
    ProximalBase.SquaredHinge()

The squared hinge loss `ℓ(η, y) = max(0, 1 - yη)²` for a label `y ∈ {-1, 1}`,
the loss of the L2-loss support vector machine. Smooth, with curvature 2, so
coordinate descent and gradient methods minimize a [`GLMLoss`](@ref) with it
directly.
"""
struct SquaredHinge <: ScalarLoss end
@inline function value_and_deriv(::SquaredHinge, η::Real, y::Real)
  m = max(1 - y * float(η), zero(float(η)))
  (abs2(m), -2 * y * m)
end
(ℓ::SquaredHinge)(η::Real, y::Real) = value_and_deriv(ℓ, η, y)[1]
deriv(ℓ::SquaredHinge, η::Real, y::Real) = value_and_deriv(ℓ, η, y)[2]
deriv2(::SquaredHinge, η::Real, y::Real) = (t = float(η); y * t < 1 ? 2 * abs2(y) * one(t) : zero(t))
curvature(::SquaredHinge) = 2.0

"""
    ProximalBase.SmoothCheck(τ, h)

The check loss of quantile regression smoothed at the scale `h > 0`: the Moreau
envelope `min_s ρ_τ(s) + (s - u)²/(2h)` of `ρ_τ(u) = max(τu, (τ - 1)u)` at the
residual `u = y - η`. It is `τu - hτ²/2` for `u ≥ hτ`,
`(τ - 1)u - h(1 - τ)²/2` for `u ≤ -h(1 - τ)`, and `u²/(2h)` in between: an
asymmetric Huber loss, within `h/2` of `ρ_τ`, with curvature `1/h`. With it,
coordinate descent and proximal Newton fit penalized quantile regression
through [`GLMLoss`](@ref), without a linear program.
"""
struct SmoothCheck{T<:Real} <: ScalarLoss
  τ::T
  h::T

  function SmoothCheck{T}(τ::T, h::T) where {T<:Real}
    0 < τ < 1 || throw(ArgumentError("τ must be in (0, 1), got $τ"))
    h > 0 || throw(ArgumentError("h must be positive, got $h"))
    new{T}(τ, h)
  end
end
SmoothCheck(τ::Real, h::Real) = (p = promote(τ, h); SmoothCheck{typeof(p[1])}(p...))
@inline function value_and_deriv(ℓ::SmoothCheck, η::Real, y::Real)
  u = float(y - η)
  τ, h = convert(typeof(u), ℓ.τ), convert(typeof(u), ℓ.h)
  u >= h * τ && return (τ * u - h * τ^2 / 2, -τ)
  u <= -h * (1 - τ) && return ((τ - 1) * u - h * (1 - τ)^2 / 2, 1 - τ)
  (u^2 / (2h), -u / h)
end
(ℓ::SmoothCheck)(η::Real, y::Real) = value_and_deriv(ℓ, η, y)[1]
deriv(ℓ::SmoothCheck, η::Real, y::Real) = value_and_deriv(ℓ, η, y)[2]
function deriv2(ℓ::SmoothCheck, η::Real, y::Real)
  u = float(y - η)
  τ, h = convert(typeof(u), ℓ.τ), convert(typeof(u), ℓ.h)
  -h * (1 - τ) < u < h * τ ? inv(h) : zero(u)
end
curvature(ℓ::SmoothCheck) = inv(float(ℓ.h))

"""
    ProximalBase.ConvolutionCheck(τ, h; kernel = :epanechnikov)

The check loss of quantile regression convolved with a kernel of bandwidth
`h > 0`, the smoothing of the conquer estimator (He, Pan, Tan and Zhou, 2023):
`∫ ρ_τ(u - t) K(t/h)/h dt` at the residual `u = y - η`. The kernels are
`:uniform`, `K = 1/2` on `[-1, 1]`, and `:epanechnikov`,
`K(t) = 3(1 - t²)/4` on `[-1, 1]`. The loss equals `ρ_τ(u)` for `|u| ≥ h`
and is a polynomial in between, with second derivative `K(u/h)/h`, so its
curvature is `1/(2h)` or `3/(4h)`. Unlike [`ProximalBase.SmoothCheck`](@ref),
it is centered: its derivative is `τ - 1/2` at `u = 0`.
"""
struct ConvolutionCheck{T<:Real, K} <: ScalarLoss
  τ::T
  h::T

  function ConvolutionCheck{T, K}(τ::T, h::T) where {T<:Real, K}
    K in (:uniform, :epanechnikov) || throw(ArgumentError("the kernel must be :uniform or :epanechnikov, got :$K"))
    0 < τ < 1 || throw(ArgumentError("τ must be in (0, 1), got $τ"))
    h > 0 || throw(ArgumentError("h must be positive, got $h"))
    new{T, K}(τ, h)
  end
end
ConvolutionCheck(τ::Real, h::Real; kernel::Symbol = :epanechnikov) =
  (p = promote(τ, h); ConvolutionCheck{typeof(p[1]), kernel}(p...))

# ρ_τ(u) = (τ - 1/2)u + |u|/2, and the convolution of |u| with the kernel at
# bandwidth h is h m(u/h) for |u| < h, with m(s) = ∫|s - t|K(t)dt: m, m' and
# m'' = 2K on [-1, 1]
@inline _kernel_abs(::ConvolutionCheck{<:Any, :uniform}, s) = ((1 + s^2) / 2, s, one(s))
@inline _kernel_abs(::ConvolutionCheck{<:Any, :epanechnikov}, s) =
  (3 * (1 + 2s^2) / 8 - s^4 / 8, 3s / 2 - s^3 / 2, 3 * (1 - s^2) / 2)

@inline function value_and_deriv(ℓ::ConvolutionCheck, η::Real, y::Real)
  u = float(y - η)
  τ, h = convert(typeof(u), ℓ.τ), convert(typeof(u), ℓ.h)
  u >= h && return (τ * u, -τ)
  u <= -h && return ((τ - 1) * u, 1 - τ)
  m, dm, _ = _kernel_abs(ℓ, u / h)
  ((τ - one(τ) / 2) * u + h * m / 2, -(τ - one(τ) / 2) - dm / 2)
end
(ℓ::ConvolutionCheck)(η::Real, y::Real) = value_and_deriv(ℓ, η, y)[1]
deriv(ℓ::ConvolutionCheck, η::Real, y::Real) = value_and_deriv(ℓ, η, y)[2]
function deriv2(ℓ::ConvolutionCheck, η::Real, y::Real)
  u = float(y - η)
  h = convert(typeof(u), ℓ.h)
  abs(u) < h ? _kernel_abs(ℓ, u / h)[3] / (2h) : zero(u)
end
curvature(ℓ::ConvolutionCheck{<:Any, :uniform}) = inv(2 * float(ℓ.h))
curvature(ℓ::ConvolutionCheck{<:Any, :epanechnikov}) = 3 / (4 * float(ℓ.h))
