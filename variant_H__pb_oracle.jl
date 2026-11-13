# PB #911 prototype H, revision 2 (after review r1): the same code as pkgs/pb-H/src (a disposable PB
# copy in which LS `f(β)`, `value_and_gradient!`, `gradient!` and these oracles share one kernel).
# This in-process copy cannot modify PB, so the two shared helpers of the refactor are copied below
# verbatim (`_square_value_level`/`_square_value_wide`, `_leastsquares_residual_gradient!`); in PB they
# replace code, they are not added beside it. Disposable.
#
# Value: the SqrNormL2 squared-residual kernel with the scale as its level, kept as supplied and the
# product rounded once at the result (D3); exceptional ranges by `_exceptional_square_value`.
# Gradient: g = Xᵀd with the scale applied last by Scaled's `_scale!`.
module PBOracle
using LinearAlgebra
import ProximalBase
const PB = ProximalBase
using ProximalBase: LeastSquaresLoss, Scaled

# ---- copied from pkgs/pb-H/src/numerical_kernels.jl (refactor of `_square_value`) ----
@inline _square_value_wide(λ, x, y) = _square_value_level(λ, λ, x, y, true)
@inline function _square_value_level(l, λ, x, y, fits::Bool)
  T = PB._valtype(x)
  s = y === nothing ? PB._sumabs2(x) : PB._sumabs2_diff(x, y)
  v = l * s / 2
  n = y === nothing ? PB._square_term_count(x) : length(x)
  if (isfinite(v) & PB._squares_normal(real(s), n, T)) && fits
    return PB._value_at(v, x)
  end
  PB._exceptional_square_value(λ, x, y, real(s), v)
end
# ---- copied from pkgs/pb-H/src/functions/leastsquares.jl ----
@inline function _leastsquares_residual_gradient!(g, d, X, z, Y)
  d .= z .- Y
  mul!(g, transpose(X), d)
end
@inline function _leastsquares_residual_gradient!(g, d, X, z, Y, α)
  _leastsquares_residual_gradient!(g, d, X, z, Y)
  _square_value_wide(α, d, nothing)
end
# ---- pkgs/pb-H/src/functions/leastsquares_image.jl ----
const _LSImageBase{T} = LeastSquaresLoss{T, <:StridedVector{T}, <:StridedMatrix{T}}
const _LSImageLoss{T} = Union{_LSImageBase{T}, Scaled{<:_LSImageBase{T}}}
_image_parts(f::_LSImageBase{T}) where {T} = (f, one(T))
_image_parts(s::Scaled{<:_LSImageBase}) = (s.f, s.α)
_image_scale!(g, ::_LSImageBase) = g
_image_scale!(g, s::Scaled{<:_LSImageBase}) = PB._scale!(g, s.α)
@noinline _image_length_error(name, m, n) = throw(DimensionMismatch("$name has length $m, expected $n"))
@noinline _image_scale_error(α) = throw(ArgumentError("the scale must be positive and finite, got $α"))
@noinline _image_alias_error() = throw(ArgumentError("g and d must not overlap each other, X, Y, z or the loss's internal storage"))
@inline function _image_checks(base, α, n, z, name)
  length(z) == n || _image_length_error(name, length(z), n)
  isfinite(α) & (α > 0) || _image_scale_error(α)
  nothing
end
function value_at_image(f::_LSImageLoss{T}, z::StridedVector{T}) where {T<:Union{Float32, Float64}}
  base, α = _image_parts(f)
  _image_checks(base, α, length(base.Y), z, "z")
  _square_value_wide(α, z, base.Y)
end
function value_and_gradient_at_image!(g::StridedVector{T}, d::StridedVector{T}, f::_LSImageLoss{T},
                                      z::StridedVector{T}) where {T<:Union{Float32, Float64}}
  base, α = _image_parts(f)
  X, Y = base.X, base.Y
  n = length(Y)
  _image_checks(base, α, n, z, "z")
  length(d) == n || _image_length_error("d", length(d), n)
  length(g) == size(X, 2) || _image_length_error("g", length(g), size(X, 2))
  (Base.mightalias(d, z) | Base.mightalias(d, Y) | Base.mightalias(d, X) | Base.mightalias(g, d) |
   Base.mightalias(g, X) | Base.mightalias(g, Y) | Base.mightalias(g, z) |
   Base.mightalias(d, base.tmp) | Base.mightalias(g, base.tmp)) && _image_alias_error()
  v = _leastsquares_residual_gradient!(g, d, X, z, Y, α)
  _image_scale!(g, f)
  v
end
function value_at_residual(f::_LSImageLoss{T}, r::StridedVector{T}) where {T<:Union{Float32, Float64}}
  base, α = _image_parts(f)
  _image_checks(base, α, length(base.Y), r, "r")
  _square_value_wide(α, r, nothing)
end
end
