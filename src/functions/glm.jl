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
work. A custom loss must be convex and differentiable in `η` for the response
values it supports, and `curvature` must bound its second derivative (it may be
`Inf`). `GLMLoss` reports itself smooth and convex on that basis and does not
check these properties. The losses are [`ProximalBase.Squared`](@ref),
[`ProximalBase.Logistic`](@ref), [`ProximalBase.Poisson`](@ref),
[`ProximalBase.IdentityBernoulli`](@ref), [`ProximalBase.Huber`](@ref),
[`ProximalBase.SquaredHinge`](@ref), [`ProximalBase.SmoothCheck`](@ref) and
[`ProximalBase.ConvolutionCheck`](@ref);
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
    ProximalBase.IdentityBernoulli()

The Bernoulli negative log-likelihood with the identity link, `ℓ(η, y) = -y log(η) -
(1 - y) log(1 - η)`: the linear predictor `η` is itself the probability, as in a
risk or risk-difference regression, and the response is `y ∈ [0, 1]` (binary or a
proportion). It is the loss of [`BernoulliLoss`](@ref)`(y)` at `p = η`, and
[`BernoulliLoss`](@ref) is the choice when a proximal operator is needed.

The loss is finite only for `0 < η < 1`, and at the endpoints where the response
agrees: `η = 0` for `y = 0` and `η = 1` for `y = 1`, where `ℓ = 0`. Elsewhere,
at a predictor outside `[0, 1]` or infinite, `ℓ = Inf` and the derivatives are `NaN`;
at a `NaN` predictor all three are `NaN`. At the two valid endpoints the derivatives
are one-sided limits: `ℓ' = 1`, `ℓ'' = 1` at `(0, 0)` and `ℓ' = -1`, `ℓ'' = 1` at
`(1, 1)`. A term with a zero coefficient is omitted, never evaluated as `0 ⋅ log(0)`.
`ℓ''` is formed without an intermediate underflow of `η²` whenever the result is
representable, and is `Inf` when it is not. Its curvature is unbounded, so
`lipschitz` of a [`GLMLoss`](@ref) with it is `Inf`: a step needs a feasible point
and a line search, and `GLMLoss` neither projects nor clamps the predictor.

[`GLMLoss`](@ref) rejects a response that is not in `[0, 1]` (also `NaN` and `±Inf`),
whatever its weight.
"""
struct IdentityBernoulli <: ScalarLoss end
@inline function value_and_deriv(::IdentityBernoulli, η::Real, y::Real)
  p, b = promote(float(η), float(y))
  w = one(p)
  (_bernoulli_value(p, b, w), _bernoulli_gradient(p, b, w))
end
@inline function (::IdentityBernoulli)(η::Real, y::Real)
  p, b = promote(float(η), float(y))
  _bernoulli_value(p, b, one(p))
end
@inline function deriv(::IdentityBernoulli, η::Real, y::Real)
  p, b = promote(float(η), float(y))
  _bernoulli_gradient(p, b, one(p))
end
@inline function deriv2(::IdentityBernoulli, η::Real, y::Real)
  p, b = promote(float(η), float(y))
  T = typeof(p)
  isnan(p) && return p
  _bernoulli_domain(p, b, one(T)) || return T(NaN)
  left = iszero(b) ? zero(T) : _bernoulli_curvature(b, p)
  right = b == 1 ? zero(T) : _bernoulli_curvature(1 - b, 1 - p)
  left + right
end
curvature(::IdentityBernoulli) = Inf

# c / p² for 0 < p <= 1 and c >= 0 without forming p², which underflows at p below
# about sqrt(floatmin) while the quotient is representable; the quotient is Inf when it
# overflows. The exponents are held apart from the significands in that case.
@inline function _bernoulli_curvature(c::T, p::T) where {T<:AbstractFloat}
  p² = p * p
  p² >= floatmin(T) && return c / p²
  mc, ec = frexp(c)
  mp, ep = frexp(p)
  ldexp(mc / (mp * mp), ec - 2 * ep)
end

# The responses of an IdentityBernoulli loss are probabilities. `_check_responses` is a
# no-op for the other losses; the predicate is an isbits Function, like the checks of the
# weights, so that `all` can reduce a device array.
_check_responses(::ScalarLoss, y, ::Type) = nothing
struct _UnitResponse{T} <: Function end
(::_UnitResponse{T})(t) where {T} = t isa Real && (r = convert(T, t); 0 <= r <= 1)
_check_responses(::IdentityBernoulli, y, ::Type{T}) where {T} =
  all(_UnitResponse{T}(), y) || throw(ArgumentError("the responses of an IdentityBernoulli loss must be in [0, 1]"))

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
omitted. The weights must be finite and nonnegative, and the offset finite, in
the working precision of `y` and `X`; the constructor throws an `ArgumentError`
otherwise. Smooth, with gradient `X' (wᵢ ℓ'(ηᵢ, yᵢ))ᵢ`. [`LogisticLoss`](@ref),
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

  function GLMLoss{L, T, Ty, TX, W, O, V}(loss, y, X, weights, offset, η, d) where {L<:ScalarLoss,
      T<:AbstractFloat, Ty<:AbstractVector, TX<:AbstractMatrix, W<:Union{Nothing, AbstractVector},
      O<:Union{Nothing, AbstractVector}, V<:AbstractVector{T}}
    all(a -> a === nothing || eltype(a) <: Real,
        (y, X, weights, offset, η, d)) || _real_error(:GLMLoss)
    _check_responses(loss, y, T)
    # η and d are scratch that gradient!, hessian_vec! and quadratic_model! overwrite, the
    # last staging two values per observation in them: they must be two arrays, neither of
    # them y, weights or offset. Identity only: a view or a reshape is not detected.
    (η === d || η === y || η === weights || η === offset ||
     d === y || d === weights || d === offset) &&
      throw(ArgumentError("the buffers η and d of a GLMLoss must be separate arrays, and not y, weights or offset"))
    new{L, T, Ty, TX, W, O, V}(loss, y, X, weights, offset, η, d)
  end
end
GLMLoss(loss::L, y::Ty, X::TX, weights::W, offset::O, η::V, d::V) where {L<:ScalarLoss, T<:AbstractFloat,
    Ty<:AbstractVector, TX<:AbstractMatrix, W<:Union{Nothing, AbstractVector},
    O<:Union{Nothing, AbstractVector}, V<:AbstractVector{T}} =
  GLMLoss{L, T, Ty, TX, W, O, V}(loss, y, X, weights, offset, η, d)

function GLMLoss(loss::ScalarLoss, y::AbstractVector, X::AbstractMatrix;
                 weights::Union{Nothing, AbstractVector} = nothing, offset::Union{Nothing, AbstractVector} = nothing)
  y === nothing || eltype(y) <: Real || _real_error(:GLMLoss)
  eltype(X) <: Real || _real_error(:GLMLoss)
  weights === nothing || eltype(weights) <: Real || _real_error(:GLMLoss)
  offset === nothing || eltype(offset) <: Real || _real_error(:GLMLoss)
  n = length(y)
  size(X, 1) == n || throw(DimensionMismatch("X has $(size(X, 1)) rows and y has length $n"))
  weights === nothing || length(weights) == n || throw(DimensionMismatch("weights must have length $n"))
  offset === nothing || length(offset) == n || throw(DimensionMismatch("offset must have length $n"))
  T = float(promote_type(eltype(y), eltype(X)))
  # finite in the working precision, where the oracles use them
  weights === nothing || all(_FiniteWeight{T}(), weights) ||
    throw(ArgumentError("the weights must be finite and nonnegative"))
  offset === nothing || all(_FiniteOffset{T}(), offset) ||
    throw(ArgumentError("the offset must be finite"))
  η, d = _zeros_like(y, T, n), _zeros_like(y, T, n)
  GLMLoss{typeof(loss), T, typeof(y), typeof(X), typeof(weights), typeof(offset), typeof(η)}(
    loss, y, X, weights, offset, η, d)
end

# The constructor's checks in the working precision T. They are singleton
# structs, not closures over T: a closure that captures a type is not isbits,
# and a GPU reduction (`all` on a device array, as Adapt rebuilds the loss)
# cannot take it as a kernel argument. They are Functions, since GPUArrays
# sends only `all(f::Function, A)` to a device reduction.
struct _FiniteWeight{T} <: Function end
(::_FiniteWeight{T})(t) where {T} = t isa Real && t >= 0 && isfinite(convert(T, t))
struct _FiniteOffset{T} <: Function end
(::_FiniteOffset{T})(t) where {T} = t isa Real && isfinite(convert(T, t))

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
_argument_type(::Type{<:GLMLoss}) = AbstractVector
isconvex(::Type{<:GLMLoss}) = true

_checkpredictor(f::GLMLoss, β::AbstractVector) =
  (length(β) == size(f.X, 2) || throw(DimensionMismatch("β has length $(length(β)) and X has $(size(f.X, 2)) columns")))

_checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::GLMLoss{L, T}, g::AbstractVector, β::AbstractVector) where {L, T} =
  (_checkreal(f, β); _checkreal(f, g); _checkgradient_output(g, β); _checksize(g, β); _checkpredictor(f, β); true)

function _linear_predictor!(f::GLMLoss, β::AbstractVector)
  _checkpredictor(f, β)
  mul!(f.η, f.X, β)
  f.offset === nothing || (f.η .+= f.offset)
  f.η
end

function (f::GLMLoss{L, T})(β::AbstractVector) where {L, T}
  _checkreal(f, β)
  η = _linear_predictor!(f, β)
  s = zero(T)
  @inbounds for i in eachindex(η)
    w = convert(T, _entry(f.weights, i))
    iszero(w) && continue
    s += w * convert(T, f.loss(η[i], f.y[i]))
  end
  _value_at(s, β)
end

function value_and_gradient!(g::AbstractVector, f::GLMLoss{L, T}, β::AbstractVector) where {L, T}
  _checkwrite(value_and_gradient!, f, g, β)
  η = _linear_predictor!(f, β)
  d = f.d
  s = zero(T)
  @inbounds for i in eachindex(η)
    w = convert(T, _entry(f.weights, i))
    if iszero(w)
      d[i] = zero(T)
      continue
    end
    v, dv = value_and_deriv(f.loss, η[i], f.y[i])
    s += w * convert(T, v)
    d[i] = w * dv
  end
  mul!(g, transpose(f.X), d)
  _value_at(s, β)
end

function gradient!(g::AbstractVector, f::GLMLoss{L, T}, β::AbstractVector) where {L, T}
  _checkwrite(gradient!, f, g, β)
  η = _linear_predictor!(f, β)
  @inbounds for i in eachindex(η)
    w = convert(T, _entry(f.weights, i))
    f.d[i] = iszero(w) ? zero(T) : w * deriv(f.loss, η[i], f.y[i])
  end
  mul!(g, transpose(f.X), f.d)
end

restrict(f::GLMLoss, S::AbstractVector{<:Integer}) =
  GLMLoss(f.loss, f.y, f.X[:, S]; weights = f.weights, offset = f.offset)

# the curvature bound of ℓ times the largest eigenvalue of X' Diagonal(w) X
function lipschitz(f::GLMLoss{L, T}) where {L, T}
  c = curvature(f.loss)
  isinf(c) ? T(Inf) : c * _opnorm2_upper(f.X, f.weights)
end
