# (revision 1 of H, superseded after review r1; kept for the record)
# PB #911 prototype H — hybrid of B3 and B1, written as it would live in PB (leastsquares.jl).
# Value: the SqrNormL2 squared-residual value with α as its level (`_square_value(α, ·, ·)` and its
# exceptional/scaled recovery), so an extreme α is folded into the recomputation instead of
# multiplying an inner value that has already overflowed or underflowed.
# Gradient: g = α Xᵀ(z - Y), α applied last by Scaled's rule (`_scale!`), so a tiny α does not
# underflow the n entries of d before the product (B3's weakness).
# value-only → `_square_value(α, z, Y)` (the LS f(β) loop and order); value+gradient → d = z - Y,
# g = Xᵀd, value `_square_value(α, d, nothing)` (the LS value_and_gradient! order, sum(abs2, d)).
# Disposable.
module PBOracle
using LinearAlgebra
import ProximalBase
const PB = ProximalBase
using ProximalBase: LeastSquaresLoss, Scaled, SqrNormL2

const _LSV{T} = LeastSquaresLoss{T, <:AbstractVector}
_lsparts(f::_LSV{T}) where {T} = (f, one(T))
_lsparts(s::Scaled{<:_LSV}) = (s.f, s.α)
_gscale!(g, ::_LSV) = g
_gscale!(g, s::Scaled{<:_LSV}) = PB._scale!(g, s.α)

@noinline _len_error(n, m) = throw(DimensionMismatch("the image has length $m, Y has length $n"))
@noinline _alias_error() = throw(ArgumentError("g and d must not overlap each other, X, Y or z"))

function value_at_image(f, z::AbstractVector{T}) where {T}
  base, α = _lsparts(f)
  length(z) == length(base.Y) || _len_error(length(base.Y), length(z))
  SqrNormL2(α, base.Y)(z)
end
function value_and_gradient_at_image!(g::AbstractVector{T}, d::AbstractVector{T}, f, z::AbstractVector{T}) where {T}
  base, α = _lsparts(f)
  X, Y = base.X, base.Y
  length(z) == length(Y) || _len_error(length(Y), length(z))
  length(d) == length(Y) && length(g) == size(X, 2) || throw(DimensionMismatch("d must have length n and g length p"))
  (Base.mightalias(d, z) | Base.mightalias(d, Y) | Base.mightalias(d, X) | Base.mightalias(g, d) |
   Base.mightalias(g, X) | Base.mightalias(g, Y) | Base.mightalias(g, z)) && _alias_error()
  d .= z .- Y
  mul!(g, transpose(X), d)
  _gscale!(g, f)
  SqrNormL2(α)(d)
end
function value_at_residual(f, r::AbstractVector{T}) where {T}
  base, α = _lsparts(f)
  length(r) == length(base.Y) || _len_error(length(base.Y), length(r))
  SqrNormL2(α)(r)
end
end
