# PB #911 prototype B3 — reuse SqrNormL2: α‖z - Y‖²/2 *is* SqrNormL2(α, Y) at z, and its gradient
# α(z - Y) is the image derivative. Written as it would live in PB (leastsquares.jl): the only new
# code is the LS adapter that names that function and forms g = X'd. Disposable.
#
# Ownership: value, image derivative, scale (as SqrNormL2's level), exceptional recovery
# (`_square_value`/`_exceptional_square_value`/`_scaled_square_value`, value parts) and the
# overflow repair of the derivative entries (`_ridge_gradient_dense!`) are all existing
# SqrNormL2 code. The adapter adds only X'd.
#
# Two call depths are prototyped in one file: the public checked SqrNormL2 calls (`B3`, the
# default) and PB's unchecked internals (selected by `B3_UNCHECKED`), to measure the checks.
module PBOracle
using LinearAlgebra
import ProximalBase
const PB = ProximalBase
using ProximalBase: LeastSquaresLoss, Scaled, SqrNormL2, value_and_gradient!

const UNCHECKED = false     # rewritten to true by the B3u copy

const _LSV{T} = LeastSquaresLoss{T, <:AbstractVector}
_lsparts(f::_LSV{T}) where {T} = (f, one(T))
_lsparts(s::Scaled{<:_LSV}) = (s.f, s.α)
# the function of the image whose value is the loss: SqrNormL2(α, Y)
@inline image_function(f) = ((base, α) = _lsparts(f); SqrNormL2(α, base.Y))
# the function of the residual: SqrNormL2(α)
@inline residual_function(f) = ((base, α) = _lsparts(f); SqrNormL2(α))

@noinline _len_error(n, m) = throw(DimensionMismatch("the image has length $m, Y has length $n"))
@noinline _alias_error() = throw(ArgumentError("g and d must not overlap each other, X, Y or z"))

@inline _val(h, z) = UNCHECKED ? PB._value_unchecked(h, z) : h(z)
@inline _vag!(d, h, z) = UNCHECKED ? PB._value_and_gradient_unchecked!(d, h, z) : value_and_gradient!(d, h, z)

function value_at_image(f, z::AbstractVector{T}) where {T}
  base, _ = _lsparts(f)
  length(z) == length(base.Y) || _len_error(length(base.Y), length(z))
  _val(image_function(f), z)
end
function value_and_gradient_at_image!(g::AbstractVector{T}, d::AbstractVector{T}, f, z::AbstractVector{T}) where {T}
  base, _ = _lsparts(f)
  X, Y = base.X, base.Y
  length(z) == length(Y) || _len_error(length(Y), length(z))
  length(d) == length(Y) && length(g) == size(X, 2) || throw(DimensionMismatch("d must have length n and g length p"))
  (Base.mightalias(d, z) | Base.mightalias(d, Y) | Base.mightalias(d, X) | Base.mightalias(g, d) |
   Base.mightalias(g, X) | Base.mightalias(g, Y) | Base.mightalias(g, z)) && _alias_error()
  v = _vag!(d, image_function(f), z)       # d = α(z - Y), v = α‖z - Y‖²/2
  mul!(g, transpose(X), d)
  v
end
function value_at_residual(f, r::AbstractVector{T}) where {T}
  base, _ = _lsparts(f)
  length(r) == length(base.Y) || _len_error(length(base.Y), length(r))
  _val(residual_function(f), r)
end
end
