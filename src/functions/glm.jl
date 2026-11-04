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
[`ProximalBase.Huber`](@ref), [`ProximalBase.SquaredHinge`](@ref),
[`ProximalBase.SmoothCheck`](@ref) and [`ProximalBase.ConvolutionCheck`](@ref);
they are not exported, as names such as `Poisson` belong to distributions.

In a `GLMLoss`, a weighted term `wᵢ ℓ(ηᵢ, yᵢ)` (and `wᵢ ℓ'(ηᵢ, yᵢ)`) of a
built-in loss keeps its accuracy when `ℓ` alone overflows before a small weight,
or underflows before a large one, and when `ηᵢ` itself is beyond the range of the
buffer type (the predictor is then recomputed in power-of-two units). For a user's
loss, the term is `wᵢ ℓ(ηᵢ, yᵢ)` evaluated as written: these recomputations use
internal methods that only the built-in losses implement.
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

####################################
#
#  Weighted terms wℓ(η, y) and wℓ'(η, y) out of range (R1, #358)
#
####################################

# The ordinary term p = w ⋅ ℓ(η, y) (or w ⋅ ℓ'(η, y)) is kept when it is finite,
# at a finite η, and the weight cannot have lifted a lost value: below
# floatmin(T), a value that underflowed (to a subnormal or to zero) has lost
# relative accuracy, and a weight w > 1 can bring it back into the normal range.
# A normal ℓ from the formulas of the built-in losses carries only a few
# roundings. `_lost_value` and `_lost_deriv` say whether ℓ or ℓ' may have
# underflowed: exact zeros (an exact fit, a hinge observation on the right side)
# do not count, so that ordinary weighted data do not recompute. A loss without
# methods, a user's loss among them, keeps the ordinary term.
_lost_value(::ScalarLoss, v, η, y) = false
_lost_deriv(::ScalarLoss, v, η, y) = false
@inline _below_normal(v) = abs(v) < floatmin(float(typeof(v)))
# Whether a lost value matters at the weight w: when w > 1 can lift it. A loss
# whose own formula can return a lost value that is normal in exact arithmetic
# (SmoothCheck squares the residual before dividing by a small h) says always.
_value_lifts(::ScalarLoss, w) = w > one(w)
@inline _value_ok(ℓ, w, v, p, η, y) = isfinite(p) & isfinite(η) & !(_value_lifts(ℓ, w) & _lost_value(ℓ, v, η, y))
@inline _deriv_ok(ℓ, w, v, p, η, y) = isfinite(p) & isfinite(η) & !((w > one(w)) & _lost_deriv(ℓ, v, η, y))
# The same tests as an accumulator of the CPU loops, 0 or NaN, so that the loops
# stay free of branches and can vectorize: 0 ⋅ x is NaN for an infinite or NaN x,
# and a lifted underflow adds NaN. The finiteness of a value term is that of the
# sum of the terms, tested after the loop with the accumulator. A test that the
# sum or another test already implies is left out at compile time, by the traits
# below; the loops then decide exactly as with every test.
#
# `_value_traps(ℓ)`: ℓ(η, y) is not finite whenever η is not finite and y is
# finite, so that a weighted term w ℓ with w > 0 makes the sum non-finite and the
# test of η is implied (a non-finite y keeps the ordinary term in the recovery).
# Squared, Huber, SmoothCheck and ConvolutionCheck are linear or quadratic in
# η - y beyond a bounded zone, and NaN in it; Logistic and Poisson contain
# c ⋅ η with c = 1 - y, y or -y, so ±Inf gives ±Inf, or 0 ⋅ Inf = NaN when c = 0,
# against a bounded remainder (log1p(e^{-|η|}) ≥ 0, e^η ∈ [0, Inf]); NaN
# propagates through every branch. SquaredHinge is not: max(1 - yη, 0) = 0 for
# yη = Inf. Nor is a loss without a method.
_value_traps(::ScalarLoss) = false
_value_traps(::Union{Squared, Huber, Logistic, Poisson}) = true
# `_deriv_traps(ℓ)`: ℓ'(η, y) is not finite whenever η is not (Squared: η - y);
# the others tend to finite limits, so `gradient!` tests η for them.
_deriv_traps(::ScalarLoss) = false
_deriv_traps(::Squared) = true
# `_deriv_in_value(ℓ)`: at finite η and y, a non-finite ℓ'(η, y) implies a
# non-finite ℓ(η, y), so that in value_and_gradient! without weights (d = ℓ') the
# test of d is implied by the sum. Squared: |r| = Inf gives r²/2 = Inf. Huber,
# Logistic, SmoothCheck and ConvolutionCheck have a bounded ℓ' (δ, |1 - y| + 1,
# max(τ, 1 - τ), 1/2 + |τ - 1/2|). Poisson: e^η - y = Inf needs e^η = Inf, and
# then e^η - yη is Inf or NaN. Not SquaredHinge: for |y| near floatmax,
# 2|y| max(1 - yη, 0) overflows where the square does not. With weights,
# w ℓ' can overflow where w ℓ does not, so d is always tested.
_deriv_in_value(::ScalarLoss) = false
_deriv_in_value(::Union{Squared, Huber, Logistic, Poisson}) = true
@inline _value_mark(ℓ, w, v, η::T, y) where {T} =
  (_value_traps(ℓ) ? zero(T) : zero(T) * η) + ifelse(_value_lifts(ℓ, w) & _lost_value(ℓ, v, η, y), T(NaN), zero(T))
# the test of d = w ℓ' in value_and_gradient!, beside the value's
@inline _deriv_mark(ℓ, w, v, p, η::T, y, unweighted::Bool) where {T} =
  (unweighted & _deriv_in_value(ℓ) & _value_traps(ℓ) ? zero(T) : zero(T) * convert(T, p)) +
  ifelse((w > one(w)) & _lost_deriv(ℓ, v, η, y), T(NaN), zero(T))
# the test of d in gradient!, alone
@inline _gradient_mark(ℓ, w, v, p, η::T, y) where {T} =
  (_deriv_traps(ℓ) ? zero(T) : zero(T) * η) + zero(T) * convert(T, p) +
  ifelse((w > one(w)) & _lost_deriv(ℓ, v, η, y), T(NaN), zero(T))

# The term again for η = ηs ⋅ 2^k (k ≥ 0: η may be beyond the range of T), with
# the weight w > 0 applied last: the value as a significand and an exponent (so
# that terms beyond floatmax can still cancel in the sum), the derivative in T,
# rounded once. The default, for losses without a method, is the ordinary term.
_weighted_pair(ℓ::ScalarLoss, w::T, ηs::T, k::Int, y::T) where {T} = frexp(w * convert(T, ℓ(ldexp(ηs, k), y)))
_weighted_deriv(ℓ::ScalarLoss, w::T, ηs::T, k::Int, y::T) where {T} = w * convert(T, deriv(ℓ, ldexp(ηs, k), y))

# The term of one observation in the recovery: the ordinary term where it is
# kept, otherwise the loss's recomputation. A non-finite η with k = 0, or a
# non-finite y, keeps the ordinary term (E2).
@inline function _recovered_pair(ℓ, w::T, η::T, ηs::T, k::Int, y, ::Type{T}) where {T}
  if isfinite(η)
    v = ℓ(η, y)
    p = w * convert(T, v)
    _value_ok(ℓ, w, v, p, η, y) && return frexp(p)
    ηs, k = η, 0
  end
  yt = convert(T, y)
  isfinite(ηs) & isfinite(yt) || return frexp(w * convert(T, ℓ(η, y)))
  _weighted_pair(ℓ, w, ηs, k, yt)
end
@inline _recovered_value(ℓ, w::T, η::T, ηs::T, k::Int, y, ::Type{T}) where {T} = ldexp(_recovered_pair(ℓ, w, η, ηs, k, y, T)...)
@inline function _recovered_deriv(ℓ, w::T, η::T, ηs::T, k::Int, y, ::Type{T}) where {T}
  if isfinite(η)
    v = deriv(ℓ, η, y)
    p = w * v
    _deriv_ok(ℓ, w, v, p, η, y) && return convert(T, p)
    ηs, k = η, 0
  end
  yt = convert(T, y)
  isfinite(ηs) & isfinite(yt) || return convert(T, w * deriv(ℓ, η, y))
  _weighted_deriv(ℓ, w, ηs, k, yt)
end

# η - y for η = ηs ⋅ 2^k, as a significand and an exponent, without overflow
@inline function _residual_pow2(ηs::T, k::Int, y::T) where {T}
  if k == 0
    r = ηs - y
    isfinite(r) && return frexp(r)
    m, e = frexp(ηs / 2 - y / 2)
    return m, e + 1
  end
  m, e = frexp(ηs - ldexp(y, -k))
  m, e + k
end
# a ⋅ b as a significand and an exponent
@inline _product_pow2(a::T, b::T) where {T} = (iszero(a) || iszero(b)) ? (zero(T), 0) :
  (fa = frexp(a); fb = frexp(b); (fa[1] * fb[1], fa[2] + fb[2]))

# Squared: (η - y)²/2 and η - y. A subnormal difference is exact, so the
# derivative loses nothing to underflow.
_lost_value(::Squared, v, η, y) = _below_normal(v) & (η != y)
function _weighted_pair(::Squared, w::T, ηs::T, k::Int, y::T) where {T}
  m, e = _residual_pow2(ηs, k, y)
  _pow2_pair(w, m * m / 2, 2e)
end
_weighted_deriv(::Squared, w::T, ηs::T, k::Int, y::T) where {T} = _scaled_product(w, _residual_pow2(ηs, k, y)...)

# Huber: the quadratic zone as Squared; beyond, δ(|r| - δ/2) in units of 2^e
_lost_value(::Huber, v, η, y) = _below_normal(v) & (η != y)
function _weighted_pair(ℓ::Huber, w::T, ηs::T, k::Int, y::T) where {T}
  m, e = _residual_pow2(ηs, k, y)
  δ = convert(T, ℓ.δ)
  ldexp(abs(m), e) <= δ && return _pow2_pair(w, m * m / 2, 2e)
  _pow2_pair(w, δ, abs(m) - ldexp(δ / 2, -e), e)
end
function _weighted_deriv(ℓ::Huber, w::T, ηs::T, k::Int, y::T) where {T}
  m, e = _residual_pow2(ηs, k, y)
  δ = convert(T, ℓ.δ)
  ldexp(abs(m), e) <= δ ? _scaled_product(w, m, e) : copysign(w * δ, m)
end

# Logistic: c|η| + log(1 + e^{-|η|}) with c = 1 - y for η > 0 and c = y otherwise,
# and the derivative (1 - y) - σ or σ - y, σ = e^{-|η|}/(1 + e^{-|η|}). The tail
# t = e^{-|η|} is kept as m 2^j, and log1p(t) and σ equal t to the precision of T
# for t < 2^-(precision + 2). The exact loss is positive for y in [0, 1].
_lost_value(::Logistic, v, η, y) = _below_normal(v)
_lost_deriv(::Logistic, v, η, y) = _below_normal(v)
@inline function _logistic_tail(s::T, k::Int, sigmoid::Bool) where {T}
  (k > 0 || s > _exp_bound(T)) && return zero(T), 0
  m, j = _exp_split(-s)
  j < -(precision(T) + 2) && return m, j
  t = ldexp(m, j)
  frexp(sigmoid ? t / (1 + t) : log1p(t))
end
function _weighted_pair(::Logistic, w::T, ηs::T, k::Int, y::T) where {T}
  s = abs(ηs)
  mc, ec = _product_pow2(ηs > 0 ? one(T) - y : y, s)
  mt, et = _logistic_tail(s, k, false)
  m, e = _pow2_add2(mc, ec + k, mt, et)
  _pow2_pair(w, m, e)
end
function _weighted_deriv(::Logistic, w::T, ηs::T, k::Int, y::T) where {T}
  ms, es = _logistic_tail(abs(ηs), k, true)
  m, e = ηs > 0 ? _pow2_add2(frexp(one(T) - y)..., -ms, es) : _pow2_add2(ms, es, frexp(-y)...)
  _scaled_product(w, m, e)
end

# Poisson: e^η - yη and e^η - y, with e^η = m 2^j. Beyond the bound of
# `_exp_bound`, w e^η overflows for every weight, or is negligible beside 2^-emin.
_lost_value(::Poisson, v, η, y) = _below_normal(v)
_lost_deriv(::Poisson, v, η, y) = _below_normal(v)
@inline function _poisson_exp(ηs::T, k::Int) where {T}
  (k > 0 || ηs < -_exp_bound(T)) && return zero(T), 0
  _exp_split(ηs)
end
function _weighted_pair(::Poisson, w::T, ηs::T, k::Int, y::T) where {T}
  ηs > 0 && (k > 0 || ηs > _exp_bound(T)) && return T(Inf), 0
  me, je = _poisson_exp(ηs, k)
  my, ey = _product_pow2(y, ηs)
  m, e = _pow2_add2(me, je, -my, ey + k)
  _pow2_pair(w, m, e)
end
function _weighted_deriv(::Poisson, w::T, ηs::T, k::Int, y::T) where {T}
  ηs > 0 && (k > 0 || ηs > _exp_bound(T)) && return T(Inf)
  me, je = _poisson_exp(ηs, k)
  m, e = _pow2_add2(me, je, frexp(-y)...)
  _scaled_product(w, m, e)
end

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
  weights === nothing || all(t -> t isa Real && t >= 0 && isfinite(convert(T, t)), weights) ||
    throw(ArgumentError("the weights must be finite and nonnegative"))
  offset === nothing || all(t -> t isa Real && isfinite(convert(T, t)), offset) ||
    throw(ArgumentError("the offset must be finite"))
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
  s = c = zero(T)
  @inbounds for i in eachindex(η)
    w = convert(T, _entry(f.weights, i))
    iszero(w) && continue
    ηi, yi = η[i], f.y[i]
    v = f.loss(ηi, yi)
    s += w * convert(T, v)
    c += _value_mark(f.loss, w, v, ηi, yi)
  end
  isfinite(s + c) || (s = _glm_recover!(f, β, Val(true), Val(false)))
  _value_at(s, β)
end

function value_and_gradient!(g::AbstractVector, f::GLMLoss{L, T}, β::AbstractVector) where {L, T}
  _checkwrite(value_and_gradient!, f, g, β)
  η = _linear_predictor!(f, β)
  d = f.d
  s = c = zero(T)
  @inbounds for i in eachindex(η)
    w = convert(T, _entry(f.weights, i))
    if iszero(w)
      d[i] = zero(T)
      continue
    end
    ηi, yi = η[i], f.y[i]
    v, dv = value_and_deriv(f.loss, ηi, yi)
    s += w * convert(T, v)
    d[i] = pd = w * dv
    c += _value_mark(f.loss, w, v, ηi, yi) + _deriv_mark(f.loss, w, dv, pd, ηi, yi, f.weights === nothing)
  end
  isfinite(s + c) || (s = _glm_recover!(f, β, Val(true), Val(true)))
  mul!(g, transpose(f.X), d)
  _value_at(s, β)
end

function gradient!(g::AbstractVector, f::GLMLoss{L, T}, β::AbstractVector) where {L, T}
  _checkwrite(gradient!, f, g, β)
  η = _linear_predictor!(f, β)
  c = zero(T)
  d = f.d
  # without branches, and @simd: each d[i] is computed alone, and c is 0 or NaN
  # whatever the order of its additions
  @inbounds @simd for i in eachindex(η)
    w = convert(T, _entry(f.weights, i))
    ηi, yi = η[i], f.y[i]
    dv = deriv(f.loss, ηi, yi)
    pd = w * dv
    z = iszero(w)
    d[i] = ifelse(z, zero(T), pd)
    c += ifelse(z, zero(T), _gradient_mark(f.loss, w, dv, pd, ηi, yi))
  end
  isfinite(c) || _glm_recover!(f, β, Val(false), Val(true))
  mul!(g, transpose(f.X), f.d)
end

# The recovery after an ordinary pass that did not keep every term, on the CPU and
# on device arrays alike (broadcasts and reductions only). With a non-finite
# argument, design, response or offset, the ordinary terms stand (E2). A
# predictor that is not finite is recomputed with β ⋅ 2^-k (and the offset
# ⋅ 2^-k), so that no partial sum can overflow: η = ηs 2^k, with ηs in the
# buffer d. Every observation is then evaluated again: the ordinary term where
# it is kept, the loss's recomputation otherwise. The value returned is the sum;
# d is rewritten when asked. A recomputed predictor allocates the scaled copy of
# β; Julia 1.10 may allocate a few bytes more. The per-observation functors below
# serve these broadcasts; the GPU extension's `_ObsValue` and `_ObsDeriv` are the
# ordinary pass on the device, with the tests of `_value_ok` and `_deriv_ok`.
@noinline function _glm_recover!(f::GLMLoss{L, T}, β::AbstractVector, ::Val{V}, ::Val{D}) where {L, T, V, D}
  w = f.weights === nothing ? true : f.weights
  if !(_all_finite(β) && _all_finite(f.X) && _all_finite(f.y) && _all_finite(f.offset))
    D && (f.d .= _PlainDeriv{T}(f.loss).(f.η, f.y, w))
    return V ? _broadcast_sum(T, _PlainValue{T}(f.loss), f.η, f.y, w) : zero(T)
  end
  k = _all_finite(f.η) ? 0 : _recover_predictor!(f, β)
  s = V ? _broadcast_sum(T, _RecoveredValue{T}(f.loss, k), f.η, f.d, f.y, w) : zero(T)
  # terms beyond floatmax that cancel between observations, or partial sums that overflow:
  # the terms as significands and exponents, rounded once after the cancellation
  V && !isfinite(s) && (s = _pow2_broadcast_sum(T, _RecoveredPair{T}(f.loss, k), f.η, f.d, f.y, w))
  D && (f.d .= _RecoveredDeriv{T}(f.loss, k).(f.η, f.d, f.y, w))
  s
end

# η = Xβ + offset again in units of 2^k, chosen as in the least-squares recovery so
# that no partial sum of X(β 2^-k) + offset 2^-k exceeds floatmax(T)/4; ηs goes to
# d, and every non-finite η that is representable after all is replaced. β is
# scaled in the wider of its type and T: a Float64 β with Float32 data can need a
# k beyond the exponents of Float32.
function _recover_predictor!(f::GLMLoss{L, T}, β::AbstractVector) where {L, T}
  E = _max_exponent(T)
  k = max(0, _bound_exponent(_stored_maxabs(f.X)) + _bound_exponent(_stored_maxabs(β)) + _log2_ceil(size(f.X, 2)) + 3 - E,
          f.offset === nothing ? 0 : _bound_exponent(_stored_maxabs(f.offset)) + 3 - E)
  s = _power_factors(promote_type(T, float(eltype(β))), -k)
  mul!(f.d, f.X, β .* s[1] .* s[2])
  f.offset === nothing || (f.d .+= f.offset .* s[1] .* s[2])
  f.η .= ifelse.(isfinite.(f.η), f.η, ldexp.(f.d, k))
  k
end

# a reduction of f.(η, args...) without materializing it: a loop on the CPU (a
# reduction over a lazy broadcast allocates on Julia 1.10); the GPU extension
# reduces on the device
function _broadcast_reduce(op::O, init, f::F, η::AbstractVector, args::Vararg{Any, N}) where {O, F, N}
  bc = Broadcast.instantiate(Broadcast.broadcasted(f, η, args...))
  r = init
  @inbounds for i in eachindex(bc)
    r = op(r, bc[i])
  end
  r
end
_broadcast_sum(::Type{T}, f::F, η::AbstractVector, args::Vararg{Any, N}) where {T, F, N} =
  _broadcast_reduce(+, zero(T), f, η, args...)

# Σ mᵢ 2^eᵢ over the pairs (mᵢ, eᵢ) = f(η, args...)ᵢ, rounded at the end. The terms
# within precision + 2 + ⌈log₂ n⌉ binades of the largest are aligned exactly in its
# units, the smaller ones in the units of the largest of them; each class is summed
# with the rounding error of every addition carried along (TwoSum). The error terms
# themselves are added in ordinary arithmetic, so a class's sum is within
# γ₂ₙ² Σ|terms| of its exact value (Ogita, Rump and Oishi's Sum2, γₖ = ku/(1 - ku),
# u = eps/2, with 2n covering the sequential loop and a reduction tree); a class sum no
# larger than twice that is indistinguishable from zero and is taken as zero, so large
# terms that cancel exactly leave nothing, even when 2^E γ₂ₙ² is beyond floatmax. The
# rule applies while 2nu ≤ 1/8 (n up to about a million for Float32), where the error it
# can add, 3γ₂ₙ² Σ|terms|, is below the ordinary computation's n eps Σ|terms| and a sum
# that does not cancel is never dropped; beyond that the sums are kept. Two limits: a
# residue above that bound dominates the small class, and a term below the subnormal range of T in the
# small class's units (more than about 149 binades below its largest for Float32,
# 1074 for Float64) underflows there. Both errors are far below the accuracy of the
# ordinary computation of the same terms (eps times their scale). A non-finite term
# gives the sum of the non-finite terms (Inf, -Inf or NaN). Reductions only, for the
# device as well.
function _pow2_broadcast_sum(::Type{T}, f::F, η::AbstractVector, args::Vararg{Any, N}) where {T, F, N}
  nonfinite = _broadcast_reduce(+, zero(T), _PairNonfinite(f), η, args...)
  iszero(nonfinite) || return nonfinite
  E = _broadcast_reduce(max, typemin(Int), _PairExponent(f, typemax(Int)), η, args...)
  E == typemin(Int) && return zero(T)
  cut = E - (precision(T) + 2 + _log2_ceil(length(η)))
  E2 = _broadcast_reduce(max, typemin(Int), _PairExponent(f, cut), η, args...)
  z = zero(T)
  bs, bc, ba, ss, sc, sa = _broadcast_reduce(_add_compensated, (z, z, z, z, z, z), _PairAligned(f, E, E2, cut), η, args...)
  nu = length(η) * eps(T)       # 2n u
  g = nu <= one(T) / 8 ? nu / (one(T) - nu) : zero(T)   # n > 1/(16u): keep the sums
  big, small = _beyond_error(bs + bc, g, ba), _beyond_error(ss + sc, g, sa)
  E2 == typemin(Int) && return ldexp(big, E)
  mb, eb = frexp(big)
  ms, es = frexp(small)
  m, e = _pow2_add2(mb, eb + E, ms, es + E2)
  ldexp(m, e)
end
# (sum, error, Σ|terms|) of two classes, combined with TwoSum (`_two_sum`,
# src/ballconstrained.jl): the exact error of each addition is carried along
@inline function _add_compensated(a::NTuple{6}, b::NTuple{6})
  s1, e1 = _two_sum(a[1], b[1])
  s2, e2 = _two_sum(a[4], b[4])
  s1, a[2] + b[2] + e1, a[3] + b[3], s2, a[5] + b[5] + e2, a[6] + b[6]
end
# a compensated class sum, or zero when it is within twice its error bound g² Σ|terms|
@inline _beyond_error(s, g, a) = abs(s) <= 2 * g * g * a ? zero(s) : s
struct _PairNonfinite{F}
  f::F
end
@inline (o::_PairNonfinite)(args...) = (m = o.f(args...)[1]; isfinite(m) ? zero(m) : m)
# the exponent of a nonzero term below `below`, typemin(Int) otherwise
struct _PairExponent{F}
  f::F
  below::Int
end
@inline (o::_PairExponent)(args...) = ((m, e) = o.f(args...); !iszero(m) & (e < o.below) ? e : typemin(Int))
struct _PairAligned{F}
  f::F
  large_exp::Int
  small_exp::Int
  cut::Int
end
@inline function (o::_PairAligned)(args...)
  m, e = o.f(args...)
  z = zero(m)
  iszero(m) && return z, z, z, z, z, z
  if e >= o.cut
    x = ldexp(m, e - o.large_exp)
    x, z, abs(x), z, z, z
  else
    x = ldexp(m, e - o.small_exp)
    z, z, z, x, z, abs(x)
  end
end

# the terms of one observation, for broadcasts: ordinary, and recovered
struct _PlainValue{T, L}
  loss::L
end
_PlainValue{T}(loss::L) where {T, L} = _PlainValue{T, L}(loss)
(o::_PlainValue{T})(η, y, w) where {T} = (w = convert(T, w); iszero(w) ? zero(T) : w * convert(T, o.loss(η, y)))
struct _PlainDeriv{T, L}
  loss::L
end
_PlainDeriv{T}(loss::L) where {T, L} = _PlainDeriv{T, L}(loss)
(o::_PlainDeriv{T})(η, y, w) where {T} = (w = convert(T, w); iszero(w) ? zero(T) : convert(T, w * deriv(o.loss, η, y)))
struct _RecoveredValue{T, L}
  loss::L
  k::Int
end
_RecoveredValue{T}(loss::L, k) where {T, L} = _RecoveredValue{T, L}(loss, k)
(o::_RecoveredValue{T})(η, ηs, y, w) where {T} =
  (w = convert(T, w); iszero(w) ? zero(T) : _recovered_value(o.loss, w, η, ηs, o.k, y, T))
struct _RecoveredPair{T, L}
  loss::L
  k::Int
end
_RecoveredPair{T}(loss::L, k) where {T, L} = _RecoveredPair{T, L}(loss, k)
(o::_RecoveredPair{T})(η, ηs, y, w) where {T} =
  (w = convert(T, w); iszero(w) ? (zero(T), 0) : _recovered_pair(o.loss, w, η, ηs, o.k, y, T))
struct _RecoveredDeriv{T, L}
  loss::L
  k::Int
end
_RecoveredDeriv{T}(loss::L, k) where {T, L} = _RecoveredDeriv{T, L}(loss, k)
(o::_RecoveredDeriv{T})(η, ηs, y, w) where {T} =
  (w = convert(T, w); iszero(w) ? zero(T) : _recovered_deriv(o.loss, w, η, ηs, o.k, y, T))

restrict(f::GLMLoss, S::AbstractVector{<:Integer}) =
  GLMLoss(f.loss, f.y, f.X[:, S]; weights = f.weights, offset = f.offset)

# the curvature bound of ℓ times the largest eigenvalue of X' Diagonal(w) X
function lipschitz(f::GLMLoss{L, T}) where {L, T}
  c = curvature(f.loss)
  isinf(c) ? T(Inf) : c * _opnorm2_upper(f.X, f.weights)
end
