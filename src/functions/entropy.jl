####################################
#
#  Entropy, KL divergence, log and Bernoulli losses, and the reverse Huber penalty
#
####################################

####################################
#  the Wright omega function
####################################

# ω(u) = W₀(eᵘ), the solution of ω + log ω = u, for real u, without forming eᵘ
# for large u, where it overflows and the entropy proxes need it (u = v/γ - 1 -
# log γ is large for a small step). The start comes from the series of W at 0,
# a Taylor expansion at u = 0 or the asymptotic series, by the range
# of u, as in Lawrence, Corless and Jeffrey (2012, ACM TOMS Algorithm 917); then
# Halley's method converges in two or three steps. For u < -2, ω is small and
# log ω ≈ u, so the residual ω + log ω - u would lose the digits of ω; the
# equivalent ω = eᵘe^(-ω) keeps them.
function _wright_omega(u::T) where {T<:AbstractFloat}
  isfinite(u) || return u > 0 ? u : (isnan(u) ? u : zero(T))
  u < log(eps(T)) && return exp(u)                 # ω = eᵘ(1 - eᵘ + ⋯) to the precision of T
  if u < -2
    x = exp(u)
    w = x * (1 - x * (1 - 3x / 2))                 # W(x) = x - x² + 3x³/2 - ⋯
    for _ in 1:10
      e = x * exp(-w)
      g = w - e
      Δ = 2g * (1 + e) / (2 * (1 + e)^2 + g * e)
      w -= Δ
      abs(Δ) <= 2eps(T) * w && break
    end
    return w
  elseif u < 2
    # ω(0) is the omega constant. Implicit differentiation gives
    # ω' = ω/(1+ω) and ω'' = ω/(1+ω)^3. A quadratic start avoids exp and two
    # log1p calls here; the same Halley iteration below supplies the accuracy.
    a = T(0.567143290409783873)
    b = a / (1 + a)
    c = a / (2 * (1 + a)^3)
    w = a + u * (b + u * c)
  else
    l = log(u)
    w = u - l + l / u
  end
  for _ in 1:10
    f = w + log(w) - u
    Δ = 2f * w * (w + 1) / (2 * (w + 1)^2 + f)
    w -= Δ
    abs(Δ) <= 4eps(T) * w && break
  end
  w
end
_wright_omega(u::Real) = _wright_omega(float(u))

# Solve z + μ(log(z) + c) = v for μ = γλ. Keeping log(μ)
# separately handles a step product outside the floating-point range. When
# omega is tiny, z = exp(v/μ-c) to working precision; do not underflow omega
# before multiplying it by a large μ.
@inline _entropy_prox(v::T, γ::T, λ::T, c::T) where {T<:AbstractFloat} =
  _entropy_prox(v, γ, λ, c, zero(T))
@inline function _entropy_prox(v::T, γ::T, λ::T, c::T, reference::T) where {T<:AbstractFloat}
  μ = γ * λ
  ordinary = isfinite(μ) && !iszero(μ)
  logμ = ordinary ? log(μ) : log(γ) + log(λ)
  ratio = ordinary ? v / μ : (v / γ) / λ
  isinf(ratio) && isfinite(v) && return ratio > 0 ? v : zero(T)
  u = ratio - c - logμ
  if u < log(eps(T))
    # For KL retain q directly when q*exp(v/μ) is in range. In particular,
    # a huge step and tiny q should not lose digits through exp(log(q)).
    if reference > 0
      factor = exp(ratio)
      z = reference * factor
      isfinite(factor) && isfinite(z) && !iszero(z) && return z
    end
    return exp(ratio - c)
  end
  w = _wright_omega(u)
  ordinary ? μ * w : exp(logμ + log(w))
end

####################################
#  negative entropy
####################################

"""
    NegEntropy(λ = 1)

The negative entropy `f(x) = λ Σᵢ xᵢ log xᵢ` on `x ≥ 0` (with `0 log 0 = 0`), and
`Inf` for a negative entry, for `λ > 0`. Separable and convex. With `μ = γλ`, its
proximal operator is

    μ ω(v/μ - 1 - log μ),

where `ω(u) = W₀(eᵘ)` is the Wright omega function, computed without forming
`e^(v/μ)`, so the proximal operator stays accurate for a small step. The result
is positive, so the function does not preserve zeros.
"""
struct NegEntropy{T<:Real} <: SeparableFunction
  λ::T

  NegEntropy{T}(λ::T) where {T<:Real} = (λ > 0 || throw(ArgumentError("λ must be positive, got $λ")); new{T}(λ))
end
NegEntropy(λ::Real = 1) = NegEntropy{typeof(λ)}(λ)

isconvex(::Type{<:NegEntropy}) = true

@inline function value_entry(f::NegEntropy, v::T, ::Integer) where {T<:Real}
  S = float(T)
  t = S(v)
  t > 0 ? S(f.λ) * t * log(t) : (iszero(t) ? zero(S) : S(Inf))
end

@inline function prox_entry(f::NegEntropy, v::T, ::Integer, γ::Real) where {T<:Real}
  S = float(T)
  _entropy_prox(S(v), S(γ), S(f.λ), one(S))
end

restrict(f::NegEntropy, ::AbstractVector{<:Integer}) = f

####################################
#  KL divergence
####################################

"""
    KLDivergence(q; λ = 1)

The Kullback–Leibler divergence `f(x) = λ Σᵢ (xᵢ log(xᵢ/qᵢ) - xᵢ + qᵢ)` from a
positive reference `q` (a scalar, or an array of the size of `x`) on `x ≥ 0`, and
`Inf` for a negative entry, for `λ > 0`. Separable and convex, and zero at
`x = q`. It is half the Poisson deviance of observations `x` with fitted mean `q`,
viewed as a function of `x`. With
`μ = γλ`, its proximal operator is

    μ ω(v/μ + log(qᵢ/μ)),

with the Wright omega function `ω`, as for [`NegEntropy`](@ref).
"""
struct KLDivergence{T<:Real, Q<:Union{Real, AbstractArray}} <: SeparableFunction
  q::Q
  λ::T

  function KLDivergence{T, Q}(q::Q, λ::T) where {T<:Real, Q<:Union{Real, AbstractArray}}
    all(>(0), q) || throw(ArgumentError("the reference q must be positive"))
    λ > 0 || throw(ArgumentError("λ must be positive, got $λ"))
    new{T, Q}(q, λ)
  end
end
KLDivergence(q::Union{Real, AbstractArray}; λ::Real = 1) = KLDivergence{typeof(λ), typeof(q)}(q, λ)

isconvex(::Type{<:KLDivergence}) = true
_checkparams(f::KLDivergence, x) = (_checkparam(f.q, x); true)

Base.@propagate_inbounds function value_entry(f::KLDivergence, v::T, i::Integer) where {T<:Real}
  S = float(T)
  t, q = S(v), S(_entry(f.q, i))
  if t > 0
    ratio = t / q
    logratio = isfinite(ratio) && !iszero(ratio) ? log(ratio) : log(t) - log(q)
    return S(f.λ) * (t * logratio - t + q)
  end
  iszero(t) ? S(f.λ) * q : S(Inf)
end

Base.@propagate_inbounds function prox_entry(f::KLDivergence, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  q = S(_entry(f.q, i))
  _entropy_prox(S(v), S(γ), S(f.λ), -log(q), q)
end

restrict(f::KLDivergence, S::AbstractVector{<:Integer}) = KLDivergence(_restrict_entries(f.q, S); λ = f.λ)

####################################
#  log barrier
####################################

"""
    NegLog(λ = 1)

The log barrier `f(x) = -λ Σᵢ log xᵢ` on `x > 0`, and `Inf` elsewhere, for
`λ > 0`. Separable and convex. With `μ = γλ`, its proximal operator is the
positive root `(v + √(v² + 4μ))/2` of `z² - vz - μ = 0`, computed without
cancellation for negative `v`, the root that the proximal operator of
[`LogDetLoss`](@ref) takes for each eigenvalue.
"""
struct NegLog{T<:Real} <: SeparableFunction
  λ::T

  NegLog{T}(λ::T) where {T<:Real} = (λ > 0 || throw(ArgumentError("λ must be positive, got $λ")); new{T}(λ))
end
NegLog(λ::Real = 1) = NegLog{typeof(λ)}(λ)

isconvex(::Type{<:NegLog}) = true

@inline function value_entry(f::NegLog, v::T, ::Integer) where {T<:Real}
  S = float(T)
  t = S(v)
  t > 0 ? -S(f.λ) * log(t) : S(Inf)
end

@inline function prox_entry(f::NegLog, v::T, ::Integer, γ::Real) where {T<:Real}
  S = float(T)
  root_step = sqrt(S(γ)) * sqrt(S(f.λ))
  _logbarrier_root_half(S(v) / 2, root_step)
end

# the dimension is unknown, so a dropped term, infinite at zero, cannot be ruled out
restrict(::NegLog, ::AbstractVector{<:Integer}) =
  throw(ArgumentError("restrict of NegLog is not supported: zero is outside its domain"))
_restrict_unchecked(f::NegLog, ::AbstractVector{<:Integer}) = f

####################################
#  Bernoulli loss on the probability scale
####################################

"""
    BernoulliLoss(b, w = 1)

The Bernoulli negative log-likelihood
`f(p) = -Σᵢ wᵢ [bᵢ log pᵢ + (1 - bᵢ) log(1 - pᵢ)]` on the probability scale,
for `bᵢ ∈ [0, 1]` and finite `wᵢ ≥ 0`. Each parameter is a scalar or an array
with one entry per entry of `p`. Fractional responses with trial counts as
weights give the binomial loss. A zero weight masks a coordinate: its value
and gradient are zero everywhere and its proximal operator is the identity.
NaN inputs propagate.

For a positive weight the domain is `0 < pᵢ < 1`, with `0 log 0 = 0`: `pᵢ = 0`
is allowed for `bᵢ = 0`, and `pᵢ = 1` for `bᵢ = 1`. The value is `Inf` outside
the domain, and the gradient is `NaN` there. Separable, convex and smooth on
the interior, with gradient `-wᵢbᵢ/pᵢ + wᵢ(1 - bᵢ)/(1 - pᵢ)`. The gradient
is not globally Lipschitz; [`lipschitz`](@ref) and
[`coordinate_lipschitz`](@ref) return `Inf` for positive weights, or zero
when all relevant weights are zero.

With `μ = γwᵢ` and `0 < bᵢ < 1`, its scalar proximal point is the root in `(0, 1)` of
`p³ - (1 + v)p² + (v - μ)p + μbᵢ = 0`, computed by safeguarded Newton
iteration. For `bᵢ = 1` it is the proximal point of [`NegLog`](@ref), clipped
at 1; for `bᵢ = 0` it is its reflection. For a fractional response, a result
that rounds to an excluded endpoint is replaced by the nearest interior
float in the argument's floating type. A narrower output type may still round
it to the endpoint. [`value_and_prox!`](@ref) evaluates the stored result.
"""
struct BernoulliLoss{B<:Union{Real, AbstractArray}, W<:Union{Real, AbstractArray}} <: SeparableFunction
  b::B
  w::W

  function BernoulliLoss{B, W}(b::B, w::W) where {B<:Union{Real, AbstractArray}, W<:Union{Real, AbstractArray}}
    all(t -> t isa Real && 0 <= t <= 1, b) || throw(ArgumentError("the responses b must be in [0, 1]"))
    all(t -> t isa Real && isfinite(t) && t >= 0, w) || throw(ArgumentError("the weights must be finite and nonnegative"))
    new{B, W}(b, w)
  end
end
BernoulliLoss(b::Union{Real, AbstractArray}, w::Union{Real, AbstractArray} = 1) =
  BernoulliLoss{typeof(b), typeof(w)}(b, w)

isconvex(::Type{<:BernoulliLoss}) = true
issmooth(::Type{<:BernoulliLoss}) = true
_checkparams(f::BernoulliLoss, x) = (_checkparam(f.b, x); _checkparam(f.w, x); true)

@inline _bernoulli_domain(p, b, w) = iszero(w) ||
  (0 <= p <= 1 && (!iszero(p) || iszero(b)) && (p != 1 || b == 1))

@inline function _bernoulli_value(p::T, b::T, w::T) where {T<:AbstractFloat}
  isnan(p) && return p
  iszero(w) && return zero(T)
  _bernoulli_domain(p, b, w) || return T(Inf)
  left = iszero(b) ? zero(T) : _positive_product(w, b, -log(p))
  right = b == 1 ? zero(T) : _positive_product(w, 1 - b, -log1p(-p))
  left + right
end

Base.@propagate_inbounds function value_entry(f::BernoulliLoss, v::T, i::Integer) where {T<:Real}
  b, w = _entry(f.b, i), _entry(f.w, i)
  p, b, w = promote(float(v), float(b), float(w))
  float(T)(_bernoulli_value(p, b, w))
end

# Multiplying before dividing can underflow, and 1/p can overflow at a
# subnormal probability even when the weighted derivative is finite.
@inline function _bernoulli_ratio(w::T, b::T, p::T) where {T<:AbstractFloat}
  r = inv(p)
  isfinite(r) && return _positive_product(w, b, r)
  mw, ew = frexp(w)
  mb, eb = frexp(b)
  mp, ep = frexp(p)
  ldexp((mw * mb) / mp, ew + eb - ep)
end

@inline function _bernoulli_gradient(p::T, b::T, w::T) where {T<:AbstractFloat}
  isnan(p) && return p
  iszero(w) && return zero(T)
  _bernoulli_domain(p, b, w) || return T(NaN)
  left = iszero(b) ? zero(T) : _bernoulli_ratio(w, b, p)
  right = b == 1 ? zero(T) : _bernoulli_ratio(w, 1 - b, 1 - p)
  right - left
end

function _checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::BernoulliLoss, g::AbstractArray, x::AbstractArray)
  _checkgradient_output(g, x)
  _checksize(g, x)
  _checkparams(f, x)
end

function value_and_gradient!(g::AbstractArray, f::BernoulliLoss, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  T = float(eltype(x))
  value = zero(T)
  valid = true
  @inbounds for i in eachindex(IndexLinear(), g, x)
    p, b, w = promote(float(x[i]), float(_entry(f.b, i)), float(_entry(f.w, i)))
    value += T(_bernoulli_value(p, b, w))
    gi = _bernoulli_gradient(p, b, w)
    valid &= !isnan(gi)
    g[i] = gi
  end
  valid || fill!(g, eltype(g)(NaN))
  value
end

lipschitz(f::BernoulliLoss) = all(iszero, f.w) ? zero(float(eltype(f.w))) : float(eltype(f.w))(Inf)

# The positive root of a*t² + δ*t - c², capped at 1/2. Keeping sqrt(C)
# avoids underflow of C when the root is still representable.
@inline function _bernoulli_quadratic(a::T, δ::T, c::T) where {T<:AbstractFloat}
  half = -δ / 2
  iszero(half) && return min(one(T) / 2, c / sqrt(a))
  radius = hypot(half, sqrt(a) * c)
  root = half > 0 ? (half + radius) / a : c * (c / (radius - half))
  min(one(T) / 2, root)
end

function _bernoulli_prox(v::T, b::T, γ::T, w::T) where {T<:AbstractFloat}
  if b == 1 || iszero(b)
    root_step = sqrt(γ) * sqrt(w)
    b == 1 && return min(one(T), _logbarrier_root_half(v / 2, root_step))
    difference = fma(-γ, w, v)
    difference <= 0 && return zero(T)
    p = difference / (one(T) / 2 + v / 2 + hypot((1 - v) / 2, root_step))
    # Subtract the small reflected root only when p is nearer 1.
    return p <= one(T) / 2 ? p : 1 - _logbarrier_root_half((1 - v) / 2, root_step)
  end

  # Scale by a power of two at least as large as μ = γw, without forming μ.
  mg0, eg = frexp(γ)
  mw, ew = frexp(w)
  power = eg + ew
  shift = max(0, power)
  unit = ldexp(one(T), -shift)
  mg = ldexp(mg0, power - shift)
  μ = mg * mw
  error = fma(mg, mw, -μ)
  vs = ldexp(v, -shift)
  reflected = unit / 2 - vs + 2μ * (1 - 2b) < 0
  c, d = reflected ? (1 - b, b) : (b, 1 - b)
  u = reflected ? 1 - v : v
  # An error-free sum keeps the unit term in 1-v even for a large v.
  remainder = reflected ? (abs(v) >= 1 ? 1 - (u + v) : -v - (u - 1)) : zero(T)
  us = ldexp(u, -shift)
  # Retain μc in μ(1-c)-u when 1-c rounds to 1, and use an FMA for
  # cancellation between the step and u near an endpoint threshold.
  δ = c <= one(T) / 2 ? fma(-μ, c, fma(mg, mw, -us)) - error * c :
    fma(μ, d, -us) + error * d
  δ -= ldexp(remainder, -shift)
  md = μ * d
  # Reconstruct sqrt(μc / 2^shift) from mantissas and an integer exponent.
  rc = sqrt(mg0) * sqrt(mw) * sqrt(c)
  exponent = power - shift
  rc = ldexp(isodd(exponent) ? rc * sqrt(T(2)) : rc, fld(exponent, 2))

  # On 0 < t ≤ 1/2 the equation is
  # (1 + μd/(1-t))*t² + (μd-u)*t - μc = 0. Replacing 1/(1-t)
  # by 2 and 1 gives bounds whose ratio is at most two.
  tiny = nextfloat(zero(T))
  lo = max(tiny, prevfloat(_bernoulli_quadratic(unit + 2md, δ, rc), 4))
  hi = min(one(T) / 2, nextfloat(_bernoulli_quadratic(unit + md, δ, rc), 4))
  t = lo + (hi - lo) / 2
  for _ in 1:100
    a = unit + md / (1 - t)
    q = _bernoulli_quadratic(a, δ, rc)
    # t-q has the sign of the stationarity residual and is increasing.
    t == q && return reflected ? 1 - t : t
    t < q ? (lo = t) : (hi = t)
    if hi - lo <= 4eps(t)
      t = clamp(q, lo, hi)
      return reflected ? 1 - t : t
    end
    radius = hypot(δ / 2, sqrt(a) * rc)
    # q'(t) = -md*q²/(2radius*(1-t)²); ratios retain tiny q.
    derivative = 1 + ((md / a) / (1 - t)^2) * (q / (radius / a) / 2) * q
    candidate = t - (t - q) / derivative
    width = hi - lo
    t = lo + width / 4 < candidate < hi - width / 4 ? candidate : lo + width / 2
  end
  t = lo + (hi - lo) / 2
  reflected ? 1 - t : t
end

Base.@propagate_inbounds function prox_entry(f::BernoulliLoss, v::T, i::Integer, γ::Real) where {T<:Real}
  R = float(T)
  b, w = _entry(f.b, i), _entry(f.w, i)
  (isnan(v) || isnan(γ)) && return R(NaN)
  iszero(w) && return R(v)
  isinf(v) && return v > 0 ? (b == 1 ? one(R) : prevfloat(one(R))) : (iszero(b) ? zero(R) : nextfloat(zero(R)))
  p, b, step, w = promote(float(v), float(b), float(γ), float(w))
  result = R(_bernoulli_prox(p, b, step, w))
  0 < b < 1 ? clamp(result, nextfloat(zero(R)), prevfloat(one(R))) : result
end

####################################
#  reverse Huber (Berhu)
####################################

"""
    Berhu(λ, M)
    Berhu(λ, M, w)

The reverse Huber penalty of Owen (2007): `fᵢ(t) = λwᵢ|t|` for `|t| ≤ M` and
`λwᵢ(t² + M²)/(2M)` beyond, for `M > 0`, with all `wᵢ = 1` when `w` is omitted.
It is the ℓ1 norm near zero, for sparsity, and a ridge penalty for large
entries, which groups correlated predictors; convex, with the slope `λwᵢ` at
`±M` on both sides. With `c = γλwᵢ`, its proximal operator is

    0                      for |v| ≤ c,
    v - c sign(v)          for c < |v| ≤ M + c,
    vM / (M + c)           for |v| > M + c.

At zero its subdifferential is that of the ℓ1 norm, so
[`kkt_violation`](@ref) is `|gᵢ| - λwᵢ`.
"""
struct Berhu{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ::T
  M::T
  w::W

  function Berhu{T, W}(λ::T, M::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    _checknonneg("λ", λ)
    M > 0 || throw(ArgumentError("M must be positive, got $M"))
    _checknonneg("the weights", w)
    new{T, W}(λ, M, w)
  end
end
function Berhu(λ::Real, M::Real, w::Union{Nothing, AbstractArray} = nothing)
  λ, M = promote(λ, M)
  Berhu{typeof(λ), typeof(w)}(λ, M, w)
end

isconvex(::Type{<:Berhu}) = true
iszeropreserving(::Berhu) = true
_checkparams(f::Berhu, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::Berhu, v::T, i::Integer) where {T<:Real}
  S = float(T)
  λ, w = S(f.λ), S(_entry(f.w, i))
  c, M, t = _weight_coefficient(λ, w), S(f.M), abs(S(v))
  _weighted_value(t <= M ? c * t : c * (t^2 + M^2) / (2M), t)
end
Base.@propagate_inbounds function _value_entry_ordinary(f::Berhu, v::T, i::Integer) where {T<:Real}
  S = float(T)
  c, M, t = S(f.λ) * S(_entry(f.w, i)), S(f.M), abs(S(v))
  t <= M ? c * t : c * (t^2 + M^2) / (2M)
end
_has_ordinary_value(::Type{<:Berhu}) = true

Base.@propagate_inbounds function prox_entry(f::Berhu, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  step, λ, w = S(γ), S(f.λ), S(_entry(f.w, i))
  # Avoid rounding or overflow in the upper branch of an identity prox.
  !isinf(w) && (iszero(step) || iszero(λ) || iszero(w)) && return S(v)
  c = _weight_coefficient(step, λ, w)
  M, t = S(f.M), abs(S(v))
  t <= M + c ? shrink(S(v), c) : S(v) * M / (M + c)
end

kkt_violation(f::Berhu, g::AbstractArray, i::Integer) = abs(g[i]) - _weight_coefficient(f.λ, _entry(f.w, i))
restrict(f::Berhu, S::AbstractVector{<:Integer}) = Berhu(f.λ, f.M, _restrict_entries(f.w, S))
