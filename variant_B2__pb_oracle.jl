# PB #911 prototype B2 — a minimal vector-level `ScalarLoss` layer plus the LS adapter, written as
# it would live in PB (src/functions/scalar_loss.jl and leastsquares.jl). Disposable.
#
# Ownership: the observation loss ℓ(η, y), deriv and value_and_deriv stay the existing
# `ScalarLoss` methods (`Squared`). The vector layer sums ℓ over externally supplied
# predictions, optionally writing the image derivative in the same pass. Whether the ordinary
# sum can be trusted, and how it is recomputed otherwise, is a per-loss hook: the generic default
# keeps the ordinary sum (no recovery is known for an arbitrary ℓ); `Squared` delegates to PB's
# existing squared-residual kernel (`_scaled_square_value`), so no second recovery is written.
# The LS adapter owns the design (X'd), the scale (PB's Scaled rule) and the result type.
module PBOracle
using LinearAlgebra
import ProximalBase
const PB = ProximalBase
using ProximalBase: LeastSquaresLoss, Scaled, ScalarLoss, Squared

# ---- generic vector-level layer (PB, scalar_loss.jl) ------------------------------------
# observations that are all zero, for a residual: ℓ(rᵢ, 0)
struct _ZeroObs{T} <: AbstractVector{T}
  n::Int
end
Base.size(z::_ZeroObs) = (z.n,)
Base.IndexStyle(::Type{<:_ZeroObs}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(::_ZeroObs{T}, ::Int) where {T} = zero(T)
_obs(y) = y
_obs(::_ZeroObs) = nothing

# Σᵢ ℓ(ηᵢ, yᵢ) in the value type T of the predictions
function _loss_sum(ℓ::ScalarLoss, η::AbstractVector{T}, y::AbstractVector) where {T}
  s = zero(float(T))
  @inbounds @simd for i in eachindex(η, y)
    s += ℓ(η[i], y[i])
  end
  s
end
# the same sum, with dᵢ = ∂ℓ(ηᵢ, yᵢ)/∂η written in the same pass
function _loss_sum_deriv!(d::AbstractVector, ℓ::ScalarLoss, η::AbstractVector{T}, y::AbstractVector) where {T}
  s = zero(float(T))
  @inbounds @simd for i in eachindex(d, η, y)
    v, di = PB.value_and_deriv(ℓ, η[i], y[i])
    d[i] = di
    s += v
  end
  s
end
# per-loss hooks: is the ordinary sum s of n terms trustworthy, and its recomputation otherwise
_loss_sum_ok(::ScalarLoss, s, n, ::Type{T}) where {T} = isfinite(s)
_loss_sum_recover(::ScalarLoss, η, y, s) = s
# Squared: ℓ = r²/2, so 2s is the sum of squares that PB's fast test examines; the recovery is
# PB's squared-residual kernel (exact zeros, finite residuals in power-of-two units; a
# non-finite entry keeps s).
_loss_sum_ok(::Squared, s, n, ::Type{T}) where {T} = isfinite(s) & PB._squares_normal(2s, n, T)
_loss_sum_recover(::Squared, η::AbstractVector{T}, y, s) where {T} = PB._scaled_square_value(one(T), η, _obs(y), s)

"""
    loss_value(ℓ, η, y) -> T

Σᵢ ℓ(ηᵢ, yᵢ) over externally supplied predictions η.
"""
function loss_value(ℓ::ScalarLoss, η::AbstractVector{T}, y::AbstractVector) where {T}
  s = _loss_sum(ℓ, η, y)
  _loss_sum_ok(ℓ, s, length(η), T) ? s : _loss_sum_recover(ℓ, η, y, s)
end
"""
    loss_value_and_deriv!(d, ℓ, η, y) -> T

Σᵢ ℓ(ηᵢ, yᵢ), writing dᵢ = ∂ℓ/∂η(ηᵢ, yᵢ). d must not overlap η or y.
"""
function loss_value_and_deriv!(d::AbstractVector, ℓ::ScalarLoss, η::AbstractVector{T}, y::AbstractVector) where {T}
  s = _loss_sum_deriv!(d, ℓ, η, y)
  _loss_sum_ok(ℓ, s, length(η), T) ? s : _loss_sum_recover(ℓ, η, y, s)
end

# ---- LS adapter (PB, leastsquares.jl) -----------------------------------------------------
const _LSV{T} = LeastSquaresLoss{T, <:AbstractVector}
_lsparts(f::_LSV) = (f, true)
_lsparts(s::Scaled{<:_LSV}) = (s.f, s.α)
@inline _apply_scale(::Bool, base, z, v) = v
@inline _apply_scale(α, base, z, v) = PB._value_at(PB._scaled_value_by(base, α, z, v), z)
@noinline _len_error(n, m) = throw(DimensionMismatch("the image has length $m, Y has length $n"))
@noinline _alias_error() = throw(ArgumentError("g and d must not overlap each other, X, Y or z"))

function value_at_image(f, z::AbstractVector{T}) where {T}
  base, α = _lsparts(f)
  length(z) == length(base.Y) || _len_error(length(base.Y), length(z))
  _apply_scale(α, base, z, loss_value(Squared(), z, base.Y))
end
function value_and_gradient_at_image!(g::AbstractVector{T}, d::AbstractVector{T}, f, z::AbstractVector{T}) where {T}
  base, α = _lsparts(f)
  X, Y = base.X, base.Y
  length(z) == length(Y) || _len_error(length(Y), length(z))
  length(d) == length(Y) && length(g) == size(X, 2) || throw(DimensionMismatch("d must have length n and g length p"))
  (Base.mightalias(d, z) | Base.mightalias(d, Y) | Base.mightalias(d, X) | Base.mightalias(g, d) |
   Base.mightalias(g, X) | Base.mightalias(g, Y) | Base.mightalias(g, z)) && _alias_error()
  v = loss_value_and_deriv!(d, Squared(), z, Y)
  mul!(g, transpose(X), d)
  α === true || PB._scale!(g, α)
  _apply_scale(α, base, z, v)
end
# residual-only value: specific to the squared loss (ℓ(r, 0)), not a ScalarLoss operation
function value_at_residual(f, r::AbstractVector{T}) where {T}
  base, α = _lsparts(f)
  length(r) == length(base.Y) || _len_error(length(base.Y), length(r))
  _apply_scale(α, base, r, loss_value(Squared(), r, _ZeroObs{T}(length(r))))
end
end
