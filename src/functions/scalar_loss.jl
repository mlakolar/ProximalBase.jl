####################################
#
#  Losses of a linear predictor  Σᵢ wᵢ ℓ(ηᵢ, yᵢ),  η = Xβ + offset
#
####################################

"""
    ProximalBase.ScalarLoss

Interface for a scalar observation loss `ℓ(η,y)`. Implement the callable value,
`deriv`, `deriv2`, and a global second-derivative bound `curvature`. The default
`value_and_deriv` calls value and derivative separately. Squared is the retained
built-in scalar loss. These names are qualified rather than exported.
"""
abstract type ScalarLoss end

"""
    ProximalBase.deriv(ℓ, η, y)

The derivative `∂ℓ(η, y)/∂η` of a `ProximalBase.ScalarLoss`.
"""
function deriv end

"""
    ProximalBase.deriv2(ℓ, η, y)

The second derivative `∂²ℓ(η, y)/∂η²` of a `ProximalBase.ScalarLoss`: the
weight of the observation in a Newton step.
"""
function deriv2 end

"""
    ProximalBase.curvature(ℓ)

An upper bound on the second derivative of the `ProximalBase.ScalarLoss`
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
