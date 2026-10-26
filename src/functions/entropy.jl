####################################
#
#  Entropy, KL divergence, the log barrier, and the reverse Huber penalty
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

restrict(f::NegLog, ::AbstractVector{<:Integer}) = f

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
  c, M, t = S(f.λ) * S(_entry(f.w, i)), S(f.M), abs(S(v))
  t <= M ? c * t : c * (t^2 + M^2) / (2M)
end

Base.@propagate_inbounds function prox_entry(f::Berhu, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  c, M, t = S(γ) * S(f.λ) * S(_entry(f.w, i)), S(f.M), abs(S(v))
  t <= M + c ? shrink(S(v), c) : S(v) * M / (M + c)
end

kkt_violation(f::Berhu, g::AbstractArray, i::Integer) = abs(g[i]) - f.λ * _entry(f.w, i)
restrict(f::Berhu, S::AbstractVector{<:Integer}) = Berhu(f.λ, f.M, _restrict_entries(f.w, S))
