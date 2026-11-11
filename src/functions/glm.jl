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

In a `GLMLoss`, a weighted term `wᵢ ℓ(ηᵢ, yᵢ)` (and `wᵢ ℓ'(ηᵢ, yᵢ)`) of
`Squared`, `Logistic`, `Poisson`, `Huber`, `SquaredHinge`, `SmoothCheck` or
`ConvolutionCheck` keeps its accuracy when `ℓ` alone overflows before a small
weight, or underflows before a large one, and when `ηᵢ` itself is beyond the range
of the buffer type. These recomputations use internal methods that only those
losses implement. For `IdentityBernoulli`, `ProximalBase.Smoothed` and a
user's loss, the term is `wᵢ ℓ(ηᵢ, yᵢ)` evaluated as written. For every loss, a
predictor whose partial sums leave the range of the buffer type is recomputed in
power-of-two units, and `ℓ` is evaluated at the recomputed `ηᵢ` where that value
is representable.
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
