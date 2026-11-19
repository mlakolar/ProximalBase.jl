"""
    NormL0(λ = 1)

The nonconvex cardinality penalty `f(x) = λ ⋅ #{i : xᵢ ≠ 0}`, with
nonnegative level `λ`. For positive finite `λ` and `γ`, the proximal
operator keeps an input `v` when `|v|² > 2γλ` and returns positive zero
on an exact tie or below the threshold. The comparison is exact for
finite homogeneous `Float32` and `Float64` inputs.

At a zero level the value is zero and the proximal map is the identity
for any valid step. At an infinite level the proximal map returns positive
zero, including at a zero step. A positive finite level gives the identity
at a zero step and the constant zero map at an infinite step. These maps
apply to nonfinite entries as well. With a positive finite level and step,
NaN and signed infinities pass through. A NaN entry gives a NaN value at
any positive level, and each infinity contributes one nonzero entry.

Direct array oracles reject negative and NaN steps before writing, including
entries of an array of steps. For `prox_block` and steps passed through
`Scaled`, use nonnegative, non-NaN steps. A symmetric sparse iterate output
requires a scalar step. Its off-diagonal entries count twice in the value.
See the [numerical contract](@ref numerics) for storage and endpoint details.
"""
struct NormL0{T<:Real} <: SeparableFunction
  λ::T

  NormL0{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ))
end
NormL0(λ::Real = 1) = NormL0{typeof(λ)}(λ)

iszeropreserving(::NormL0) = true
restrict(f::NormL0, ::AbstractVector{<:Integer}) = f

@inline function value_entry(f::NormL0, v::T, ::Integer) where {T<:Real}
  S = float(T)
  (iszero(f.λ) || iszero(v)) && return zero(S)
  isnan(v) ? S(NaN) : convert(S, f.λ)
end

@inline function prox_block(f::NormL0, v::Real, ::Integer, γ::Real)
  x = float(v)
  λ = f.λ
  iszero(λ) && return x
  (isinf(λ) || isinf(γ)) && return zero(x)
  (iszero(γ) || !isfinite(x)) && return x
  R = _working_type(typeof(x), γ, λ)
  _l0_keeps(convert(R, x), convert(R, γ), convert(R, λ)) ? x : zero(x)
end

@inline _l0_keeps(x::R, γ::R, λ::R) where {R<:AbstractFloat} = abs2(x) > (2γ) * λ

@inline function _l0_keeps(x::R, γ::R, λ::R) where {R<:Base.IEEEFloat}
  p = abs2(x)
  q = (2γ) * λ
  # Doubling γ is exact when finite. Unequal rounded products preserve the
  # order of the exact products when the doubled factor remains finite.
  (floatmin(R) <= q <= floatmax(R) && p != q) && return p > q
  # The exact comparison requires finite positive factors. Scaled may form
  # a NaN step, for example from an infinite scale at a zero step.
  (0 < γ <= floatmax(R) && 0 < λ <= floatmax(R)) || return p > q
  _l0_keeps_exact(x, γ, λ)
end

@noinline function _l0_keeps_exact(x::R, γ::R, λ::R) where {R<:Base.IEEEFloat}
  iszero(x) && return false
  mx, ex = _frexp(abs(x))
  mg, eg = _frexp(γ)
  ml, el = _frexp(λ)
  shift = 2ex - (eg + el + 1)
  # The products of the normalized significands lie in [1/4, 1).
  shift > 2 && return true
  shift < -2 && return false
  n = precision(R)
  sx = unsafe_trunc(UInt64, ldexp(mx, n))
  sg = unsafe_trunc(UInt64, ldexp(mg, n))
  sl = unsafe_trunc(UInt64, ldexp(ml, n))
  p = widemul(sx, sx)
  q = widemul(sg, sl)
  # At most 106 product bits and two shift bits are needed for Float64.
  shift >= 0 ? (p << shift) > q : p > (q << -shift)
end

prox!(y::_ArrayOrSymmetricIterate, f::NormL0, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_checknonneg("γ", γ); invoke(prox!, _SeparableScalarStep, y, f, x, γ))
value_and_prox!(y::_ArrayOrSymmetricIterate, f::NormL0, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_checknonneg("γ", γ); invoke(value_and_prox!, _SeparableScalarStep, y, f, x, γ))

function prox!(y::AbstractArray, f::NormL0, x::AbstractArray, γ::AbstractArray)
  _check_oracle_arrays(prox!, f, y, x, γ)
  _checknonneg("γ", γ)
  _prox_sep!(y, f, x, γ)
end
function value_and_prox!(y::AbstractArray, f::NormL0, x::AbstractArray, γ::AbstractArray)
  _check_oracle_arrays(value_and_prox!, f, y, x, γ)
  _checknonneg("γ", γ)
  _value_and_prox_sep!(y, f, x, γ)
end
