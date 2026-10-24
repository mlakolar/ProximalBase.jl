####################################
#
#  Losses of a linear predictor  Σᵢ wᵢ ℓ(ηᵢ, yᵢ),  η = Xβ + offset
#
####################################

"""
    ProximalBase.ScalarLoss

Abstract supertype of the losses `ℓ(η, y)` of one observation `y` as a function
of its linear predictor `η`, the building blocks of [`GLMLoss`](@ref). A loss is
callable, `ℓ(η, y)`, and implements [`ProximalBase.deriv`](@ref),
[`ProximalBase.deriv2`](@ref) and [`ProximalBase.curvature`](@ref), and
[`ProximalBase.value_and_deriv`](@ref) when the value and the derivative share
work. The losses are [`ProximalBase.Squared`](@ref),
[`ProximalBase.Logistic`](@ref), [`ProximalBase.Poisson`](@ref),
[`ProximalBase.Huber`](@ref), [`ProximalBase.SquaredHinge`](@ref),
[`ProximalBase.SmoothCheck`](@ref) and [`ProximalBase.ConvolutionCheck`](@ref);
they are not exported, as names such as `Poisson` belong to distributions.
"""
abstract type ScalarLoss end

"""
    ProximalBase.deriv(ℓ, η, y)

The derivative `∂ℓ(η, y)/∂η` of a [`ProximalBase.ScalarLoss`](@ref).
"""
function deriv end

"""
    ProximalBase.deriv2(ℓ, η, y)

The second derivative `∂²ℓ(η, y)/∂η²` of a [`ProximalBase.ScalarLoss`](@ref): the
weight of the observation in a Newton step.
"""
function deriv2 end

"""
    ProximalBase.curvature(ℓ)

An upper bound on the second derivative of the [`ProximalBase.ScalarLoss`](@ref)
`ℓ` over all `η` and `y`, `Inf` when it is unbounded: with it, the gradient of
`Σᵢ wᵢ ℓ((Xβ)ᵢ, yᵢ)` is Lipschitz with constant `curvature(ℓ) ⋅ maxᵢ wᵢ ⋅ ‖X‖₂²`,
and a coordinate step of size `1 / (curvature(ℓ) ⋅ Σᵢ wᵢ Xᵢⱼ²)` decreases the
loss.
"""
function curvature end

"""
    ProximalBase.value_and_deriv(ℓ, η, y)

`(ℓ(η, y), deriv(ℓ, η, y))`, computed together.
"""
value_and_deriv(ℓ::ScalarLoss, η, y) = (ℓ(η, y), deriv(ℓ, η, y))

"""
    ProximalBase.Squared()

The squared loss `ℓ(η, y) = (η - y)² / 2`.
"""
struct Squared <: ScalarLoss end
(::Squared)(η::Real, y::Real) = abs2(η - y) / 2
deriv(::Squared, η::Real, y::Real) = η - y
deriv2(::Squared, η::T, ::Real) where {T<:Real} = one(float(T))
curvature(::Squared) = 1.0
value_and_deriv(::Squared, η::Real, y::Real) = (r = η - y; (abs2(r) / 2, r))

"""
    ProximalBase.Logistic()

The logistic loss `ℓ(η, y) = log(1 + exp(η)) - yη`, the negative log-likelihood
of logistic regression with `y ∈ {0, 1}` (or a proportion), computed without
overflow.
"""
struct Logistic <: ScalarLoss end
@inline function value_and_deriv(::Logistic, η::Real, y::Real)
  η, y = promote(float(η), y)
  # Keep the small tail explicitly: (η + log1p(exp(-η))) - η and
  # sigmoid(η) - 1 both round to zero for correctly classified large η.
  if η > 0
    e = exp(-float(η))
    ((1 - y) * η + log1p(e), (1 - y) - e / (1 + e))
  else
    e = exp(float(η))
    (log1p(e) - y * η, e / (1 + e) - y)
  end
end
(ℓ::Logistic)(η::Real, y::Real) = value_and_deriv(ℓ, η, y)[1]
function deriv(::Logistic, η::Real, y::Real)
  η, y = promote(float(η), y)
  e = exp(-abs(float(η)))
  η > 0 ? (1 - y) - e / (1 + e) : e / (1 + e) - y
end
function deriv2(::Logistic, η::Real, ::Real)
  e = exp(-abs(float(η)))
  e / (1 + e)^2
end
curvature(::Logistic) = 0.25

"""
    ProximalBase.Poisson()

The Poisson loss with the log link, `ℓ(η, y) = exp(η) - yη`, the negative
log-likelihood up to a term in `y`. Its curvature is unbounded.
"""
struct Poisson <: ScalarLoss end
@inline value_and_deriv(::Poisson, η::Real, y::Real) = (m = exp(float(η)); (m - y * η, m - y))
(::Poisson)(η::Real, y::Real) = exp(float(η)) - y * η
deriv(::Poisson, η::Real, y::Real) = exp(float(η)) - y
deriv2(::Poisson, η::Real, ::Real) = exp(float(η))
curvature(::Poisson) = Inf

"""
    ProximalBase.Huber(δ)

The Huber loss of the residual `r = η - y`: `r²/2` for `|r| ≤ δ` and
`δ(|r| - δ/2)` beyond, for robust regression.
"""
struct Huber{T<:Real} <: ScalarLoss
  δ::T

  Huber{T}(δ::T) where {T<:Real} = (δ > 0 || throw(ArgumentError("δ must be positive, got $δ")); new{T}(δ))
end
Huber(δ::Real) = Huber{typeof(δ)}(δ)
@inline function value_and_deriv(ℓ::Huber, η::Real, y::Real)
  r = float(η - y)
  δ = convert(typeof(r), ℓ.δ)
  abs(r) <= δ ? (abs2(r) / 2, r) : (δ * (abs(r) - δ / 2), δ * sign(r))
end
(ℓ::Huber)(η::Real, y::Real) = value_and_deriv(ℓ, η, y)[1]
deriv(ℓ::Huber, η::Real, y::Real) = value_and_deriv(ℓ, η, y)[2]
deriv2(ℓ::Huber, η::Real, y::Real) = (r = float(η - y); abs(r) <= ℓ.δ ? one(r) : zero(r))
curvature(::Huber) = 1.0

"""
    GLMLoss(ℓ, y, X; weights = nothing, offset = nothing)

The loss `f(β) = Σᵢ wᵢ ℓ(ηᵢ, yᵢ)` of the linear predictor `η = Xβ + offset`, for
a [`ProximalBase.ScalarLoss`](@ref) `ℓ`, with all `wᵢ = 1` and no offset when
omitted. Smooth, with gradient `X' (wᵢ ℓ'(ηᵢ, yᵢ))ᵢ`. [`LogisticLoss`](@ref),
[`PoissonLoss`](@ref) and [`HuberLoss`](@ref) construct the common ones.

`f` keeps `X` (field `X`) and a buffer for `η` (field `η`), so
[`value_and_gradient!`](@ref) costs one product with `X`, one pass over the
observations and one product with `X'`, and allocates nothing. `β` may be a
dense vector, a [`SparseIterate`](@ref) or an [`AtomIterate`](@ref), whose
products with `X` cost O(n ⋅ nnz). [`restrict`](@ref) keeps the columns `S` of
`X`.
"""
struct GLMLoss{L<:ScalarLoss, T<:AbstractFloat, Ty<:AbstractVector, TX<:AbstractMatrix,
               W<:Union{Nothing, AbstractVector}, O<:Union{Nothing, AbstractVector}, V<:AbstractVector{T}} <: AbstractFunction
  loss::L
  y::Ty
  X::TX
  weights::W
  offset::O
  η::V             # the buffers live next to y
  d::V             # wᵢ ℓ'(ηᵢ, yᵢ)
end

function GLMLoss(loss::ScalarLoss, y::AbstractVector, X::AbstractMatrix;
                 weights::Union{Nothing, AbstractVector} = nothing, offset::Union{Nothing, AbstractVector} = nothing)
  n = length(y)
  size(X, 1) == n || throw(DimensionMismatch("X has $(size(X, 1)) rows and y has length $n"))
  weights === nothing || length(weights) == n || throw(DimensionMismatch("weights must have length $n"))
  offset === nothing || length(offset) == n || throw(DimensionMismatch("offset must have length $n"))
  weights === nothing || _checknonneg("the weights", weights)
  T = float(promote_type(eltype(y), eltype(X)))
  η, d = _zeros_like(y, T, n), _zeros_like(y, T, n)
  GLMLoss{typeof(loss), T, typeof(y), typeof(X), typeof(weights), typeof(offset), typeof(η)}(
    loss, y, X, weights, offset, η, d)
end

"""
    LogisticLoss(y, X; weights = nothing, offset = nothing)

The logistic regression loss `Σᵢ wᵢ (log(1 + exp(ηᵢ)) - yᵢηᵢ)`, `η = Xβ + offset`,
for responses `yᵢ ∈ {0, 1}`: a [`GLMLoss`](@ref) with
[`ProximalBase.Logistic`](@ref).
"""
const LogisticLoss = GLMLoss{Logistic}
LogisticLoss(y::AbstractVector, X::AbstractMatrix; kwargs...) = GLMLoss(Logistic(), y, X; kwargs...)

"""
    PoissonLoss(y, X; weights = nothing, offset = nothing)

The Poisson regression loss with the log link `Σᵢ wᵢ (exp(ηᵢ) - yᵢηᵢ)`,
`η = Xβ + offset`: a [`GLMLoss`](@ref) with [`ProximalBase.Poisson`](@ref). Its
gradient is not globally Lipschitz; use a line search.
"""
const PoissonLoss = GLMLoss{Poisson}
PoissonLoss(y::AbstractVector, X::AbstractMatrix; kwargs...) = GLMLoss(Poisson(), y, X; kwargs...)

"""
    HuberLoss(y, X, δ; weights = nothing, offset = nothing)

The Huber regression loss `Σᵢ wᵢ h_δ(ηᵢ - yᵢ)`, `η = Xβ + offset`, with
`h_δ(r) = r²/2` for `|r| ≤ δ` and `δ(|r| - δ/2)` beyond: a [`GLMLoss`](@ref) with
[`ProximalBase.Huber`](@ref).
"""
const HuberLoss = GLMLoss{<:Huber}
HuberLoss(y::AbstractVector, X::AbstractMatrix, δ::Real; kwargs...) = GLMLoss(Huber(δ), y, X; kwargs...)

issmooth(::Type{<:GLMLoss}) = true
isconvex(::Type{<:GLMLoss}) = true

function _linear_predictor!(f::GLMLoss, β::AbstractVector)
  length(β) == size(f.X, 2) || throw(DimensionMismatch("β has length $(length(β)) and X has $(size(f.X, 2)) columns"))
  mul!(f.η, f.X, β)
  f.offset === nothing || (f.η .+= f.offset)
  f.η
end

function (f::GLMLoss{L, T})(β::AbstractVector) where {L, T}
  η = _linear_predictor!(f, β)
  s = zero(T)
  @inbounds for i in eachindex(η)
    s += convert(T, _entry(f.weights, i)) * convert(T, f.loss(η[i], f.y[i]))
  end
  s
end

function value_and_gradient!(g::AbstractVector, f::GLMLoss{L, T}, β::AbstractVector) where {L, T}
  _checksize(g, β)
  η = _linear_predictor!(f, β)
  d = f.d
  s = zero(T)
  @inbounds for i in eachindex(η)
    v, dv = value_and_deriv(f.loss, η[i], f.y[i])
    w = convert(T, _entry(f.weights, i))
    s += w * convert(T, v)
    d[i] = w * dv
  end
  mul!(g, transpose(f.X), d)
  s
end

function gradient!(g::AbstractVector, f::GLMLoss{L, T}, β::AbstractVector) where {L, T}
  _checksize(g, β)
  η = _linear_predictor!(f, β)
  @inbounds for i in eachindex(η)
    f.d[i] = convert(T, _entry(f.weights, i)) * deriv(f.loss, η[i], f.y[i])
  end
  mul!(g, transpose(f.X), f.d)
end

restrict(f::GLMLoss, S::AbstractVector{<:Integer}) =
  GLMLoss(f.loss, f.y, f.X[:, S]; weights = f.weights, offset = f.offset)

# the curvature bound of ℓ times the largest eigenvalue of X' Diagonal(w) X
function lipschitz(f::GLMLoss{L, T}) where {L, T}
  c = curvature(f.loss)
  isinf(c) ? T(Inf) : c * _opnorm2(f.X, f.weights)
end
