# PB #911 prototype B1 — LS-specific cache-aware oracle, written as it would live in PB
# (src/functions/leastsquares.jl). Disposable; reaches PB internals by qualified name.
#
# Ownership: the sum of squares, its fast-path test and the finite-residual recovery are PB's
# existing `_square_value` kernel (the same loop and order as LS `f(β)` value-only path, and,
# with y = nothing, `sum(abs2, ·)` as LS `value_and_gradient!`). The scale is applied by PB's
# own `Scaled` rule (`_scaled_value_by`, which for LS — no value parts, no levels — is α v);
# the coefficient gradient g = α X'(z - Y) is formed here, α applied last as `Scaled` does.
# A real B1 would make LS `f(β)`/`value_and_gradient!` call the same helpers and keep only the
# β-dependent image-overflow branch of `_leastsquares_recover` LS-specific.
module PBOracle
using LinearAlgebra
import ProximalBase
const PB = ProximalBase
using ProximalBase: LeastSquaresLoss, Scaled

const _LSV{T} = LeastSquaresLoss{T, <:AbstractVector}
_lsparts(f::_LSV) = (f, true)
_lsparts(s::Scaled{<:_LSV}) = (s.f, s.α)

# the value of the loss with the scale α at the unscaled value v, by PB's Scaled rule
@inline _apply_scale(::Bool, base, z, v) = v
@inline _apply_scale(α, base, z, v) = PB._value_at(PB._scaled_value_by(base, α, z, v), z)

@noinline _len_error(n, m) = throw(DimensionMismatch("the image has length $m, Y has length $n"))
@noinline _alias_error() = throw(ArgumentError("g and d must not overlap each other, X, Y or z"))

"""
    value_at_image(f, z) -> T

α‖Y - z‖²/2 for a caller-held image z ≈ Xβ (length n), f a vector-response
`LeastSquaresLoss{T}` or a `Scaled` one. z and Y are read only; f.tmp is not touched.
"""
function value_at_image(f, z::AbstractVector{T}) where {T}
  base, α = _lsparts(f)
  length(z) == length(base.Y) || _len_error(length(base.Y), length(z))
  _apply_scale(α, base, z, PB._square_value(one(T), z, base.Y))
end

"""
    value_and_gradient_at_image!(g, d, f, z) -> T

d .= z .- Y (scratch, length n), g .= α X'd (length p); returns α‖d‖²/2.
"""
function value_and_gradient_at_image!(g::AbstractVector{T}, d::AbstractVector{T}, f, z::AbstractVector{T}) where {T}
  base, α = _lsparts(f)
  X, Y = base.X, base.Y
  length(z) == length(Y) || _len_error(length(Y), length(z))
  length(d) == length(Y) && length(g) == size(X, 2) || throw(DimensionMismatch("d must have length n and g length p"))
  (Base.mightalias(d, z) | Base.mightalias(d, Y) | Base.mightalias(d, X) | Base.mightalias(g, d) |
   Base.mightalias(g, X) | Base.mightalias(g, Y) | Base.mightalias(g, z)) && _alias_error()
  d .= z .- Y
  mul!(g, transpose(X), d)
  v = PB._square_value(one(T), d, nothing)
  α === true || PB._scale!(g, α)
  _apply_scale(α, base, z, v)
end

"""
    value_at_residual(f, r) -> T

α‖r‖²/2 for a caller-held residual r = ±(Y - Xβ) (length n).
"""
function value_at_residual(f, r::AbstractVector{T}) where {T}
  base, α = _lsparts(f)
  length(r) == length(base.Y) || _len_error(length(base.Y), length(r))
  _apply_scale(α, base, r, PB._square_value(one(T), r, nothing))
end
end
