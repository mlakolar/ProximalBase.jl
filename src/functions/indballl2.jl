"""
    IndBallL2(r = 1)

Indicator of the Euclidean ball of radius `r ≥ 0`. For a matrix, the norm is
the Frobenius norm. The value is `0` inside the ball and `Inf` outside. A
`NaN` entry gives value `NaN`.

The proximal operator keeps feasible points and scales other finite points
radially to the boundary. It projects at any nonnegative scalar step,
including `0` and `Inf`. A negative or `NaN` step throws before writing.
Exact input/output identity is supported.

Zero radius projects to positive zeros, and infinite radius copies the
input. At a positive finite radius a single infinite entry, with no
`NaN`, projects to the signed radius at that entry and zeros elsewhere.
Several infinite entries or a `NaN` entry produce an all-`NaN` projection.

The value reads stored entries of a `SparseIterate`. With sparse input and
output buffers, the projection visits stored entries except when it fills
the logical array with `NaN`. A `BlockIterate` uses its full vector.
`SymmetricSparseIterate` inputs and outputs are unsupported.

See [the numerical contract](@ref numerics) for boundary rounding and
the value returned by [`value_and_prox!`](@ref).
"""
struct IndBallL2{T<:Real} <: AbstractFunction
  r::T

  IndBallL2{T}(r::T) where {T<:Real} = (_checknonneg("r", r); new{T}(r))
end
IndBallL2(r::Real = 1) = IndBallL2{typeof(r)}(r)

isconvex(::Type{<:IndBallL2}) = true
isproximable(::Type{<:IndBallL2}) = true
_isindicator(::Type{<:IndBallL2}) = true

# Return n and k with ‖x‖₂ ≈ n*2^k, keeping the scaled norm in range when
# the sum of squares leaves the ordinary range.
@inline function _ball_l2_norm(x)
  T = _valtype(x)
  isempty(x) && return (zero(T), 0)
  length(x) == 1 && return (abs(T(x[1])), 0)
  s = _value_sumabs2(x)
  if isnan(s) || (isfinite(s) && _squares_normal(s, _square_term_count(x), T))
    return (sqrt(s), 0)
  end
  _ball_l2_scaled_norm(x, _recovery_type(T))
end

@noinline function _ball_l2_scaled_norm(x, ::Type{R}) where {R}
  k = _square_exponent(x, nothing, R)
  k == _EXPONENT_ZERO && return (zero(R), 0)
  k == _EXPONENT_INF && return (R(Inf), 0)
  k == _EXPONENT_NAN && return (R(NaN), 0)
  (sqrt(_square_scaled_sum(x, nothing, k, R)), k)
end

@inline function _ball_l2_inside(r, n, k)
  k == 0 && return n <= r
  iszero(n) || n <= ldexp(convert(promote_type(typeof(n), typeof(float(r))), r), -k)
end

function (f::IndBallL2)(x::AbstractArray)
  _check_one_based(x)
  _checkreal(f, x)
  T = _valtype(x)
  r = f.r
  if iszero(r) || isinf(r)
    m = _maximum_abs(x)
    return isnan(m) ? T(NaN) : (isinf(r) || iszero(m) ? zero(T) : T(Inf))
  end
  n, k = _ball_l2_norm(x)
  isnan(n) ? T(NaN) : (_ball_l2_inside(r, n, k) ? zero(T) : T(Inf))
end

@noinline _ball_l2_symmetric_error() = throw(ArgumentError(
  "IndBallL2 does not support SymmetricSparseIterate; use a dense matrix such as Matrix(x)"))
(f::IndBallL2)(::SymmetricSparseIterate) = _ball_l2_symmetric_error()
prox(f::IndBallL2, ::SymmetricSparseIterate, ::Real = 1) = _ball_l2_symmetric_error()

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndBallL2,
                     y::_ArrayOrSymmetricIterate, x::_ArrayOrSymmetricIterate)
  (y isa SymmetricSparseIterate || x isa SymmetricSparseIterate) && _ball_l2_symmetric_error()
  _check_l2_arrays(f, y, x, 0)
  true
end

function prox!(y::_ArrayOrSymmetricIterate, f::IndBallL2, x::_ArrayOrSymmetricIterate, γ::Real)
  (y isa SymmetricSparseIterate || x isa SymmetricSparseIterate) && _ball_l2_symmetric_error()
  _check_l2_arrays(f, y, x, γ)
  _ball_l2_project!(y, f.r, x)
  y
end

function value_and_prox!(y::_ArrayOrSymmetricIterate, f::IndBallL2, x::_ArrayOrSymmetricIterate, γ::Real)
  prox!(y, f, x, γ)
  f(y)
end
_value_and_prox_unchecked!(y::_ArrayOrSymmetricIterate, f::IndBallL2,
                           x::_ArrayOrSymmetricIterate, γ::Real) = value_and_prox!(y, f, x, γ)

# An indicator ignores the scale. Keep the original step for validation,
# including a negative step whose product with a small scale rounds to -0.
_scaled_step(::Scaled{<:IndBallL2}, γ::Real) = γ
@inline _ball_l2_leaf(::Type) = false
@inline _ball_l2_leaf(::Type{<:IndBallL2}) = true
@inline _ball_l2_leaf(::Type{<:Scaled{F}}) where {F} = _ball_l2_leaf(F)
@inline _scaled_step(s::Scaled{<:Scaled}, γ::Real) =
  _ball_l2_leaf(typeof(s)) ? γ : invoke(_scaled_step, Tuple{Scaled, Real}, s, γ)

@inline function _ball_l2_project!(y, r, x)
  iszero(r) && return _ball_l2_zero!(y)
  isinf(r) && return _ball_l2_copy!(y, x)
  n, k = _ball_l2_norm(x)
  isfinite(n) || return _ball_l2_nonfinite!(y, r, x, n)
  _ball_l2_inside(r, n, k) && return _ball_l2_copy!(y, x)
  if length(x) == 1
    @inbounds y[1] = copysign(float(r), float(x[1]))
    return y
  end
  # An indexed view can refer to the same slot more than once. Read from a
  # snapshot for in-place destinations whose entries may share storage.
  if y === x && !(x isa Union{StridedArray, SparseIterate, BlockIterate})
    x = Array(x)
  end
  if k == 0
    a = r / n
    a >= floatmin(typeof(a)) && return _ball_l2_scale!(y, x, a)
  end
  _ball_l2_scale_recovered!(y, x, r, n, k)
end

_ball_l2_copy!(y, x) = (y === x ? y : copyto!(y, x))
_ball_l2_copy!(y::SparseIterate, x::AbstractArray) = _ball_l2_map!(identity, y, x)
_ball_l2_copy!(y::SparseIterate, x::SparseIterate) = (y === x ? y : copyto!(y, x))
# Inactive groups can contain signed zeros, which the block state copy skips.
_ball_l2_copy!(y::BlockIterate, x::BlockIterate) = (y === x ? y : _ball_l2_map!(identity, y, x))
_ball_l2_zero!(y) = fill!(y, zero(eltype(y)))
_ball_l2_zero!(y::BlockIterate) = (fill!(y.storage, zero(eltype(y))); y)

@inline function _ball_l2_scale!(y, x, a)
  _l2_scale!(y, x, Base.OneTo(length(x)), a)
  y
end

# Use the radius significand before dividing by n. Dividing the radius
# itself could erase a subnormal radius before it is multiplied by xᵢ.
@noinline function _ball_l2_scale_recovered!(y, x, r, n, k)
  R = _recovery_type(typeof(n), typeof(float(r)))
  m, e = _frexp(R(r))
  a = m / R(n)
  _ball_l2_map!(v -> _ball_l2_scaled_entry(a, R(v), e - k), y, x)
end

@inline _ball_l2_scaled_entry(a, v, k) = _scaled_product(a, v, k)

@inline function _ball_l2_scaled_entry(a::Float64, v::Float64, k)
  z = _scaled_product(a, v, k)
  if iszero(z) && !iszero(v)
    # Julia 1.10's ldexp can return zero in the binade that straddles the
    # rounding threshold. Lifting by 53 bits puts that binade in the normal
    # range. The final multiplication then rounds it into Float64.
    return _scaled_product(a, v, k + 53) * 0x1p-53
  end
  z
end

function _ball_l2_map!(op, y, x)
  @inbounds for i in 1:length(x)
    _ball_l2_write!(y, op(x[i]), i)
  end
  y
end

function _ball_l2_map!(op, y::SparseIterate, x::SparseIterate)
  if y === x
    vals = _stored_values(x)
    @inbounds for j in eachindex(vals)
      vals[j] = op(vals[j])
    end
  else
    fill!(y, zero(eltype(y)))
    @inbounds for j in 1:x.nnz
      _ball_l2_write!(y, op(x.nzval[j]), x.nzval2ind[j])
    end
  end
  y
end

Base.@propagate_inbounds _ball_l2_write!(y, v, i) = setindex!(y, v, i)

Base.@propagate_inbounds function _ball_l2_write!(y::SparseIterate, v, i)
  w = convert(eltype(y), v)
  # An implicit sparse zero would lose the sign of this rounded result.
  if iszero(w) && signbit(w) && iszero(y.ind2nzval[i])
    _store!(y, i, w)
  else
    y[i] = w
  end
end

_ball_l2_scale!(y::SparseIterate, x, a::T) where {T} =
  _ball_l2_map!(v -> a * T(v), y, x)

# Logical coordinates for ordinary arrays, and the stored coordinates for
# sparse input. A single infinity determines both a coordinate and its sign
# before anything is written, including when x === y.
_ball_l2_infinite_entries(x) = (1:length(x), x)
_ball_l2_infinite_entries(x::SparseIterate) = (view(x.nzval2ind, 1:x.nnz), _stored_values(x))

@noinline function _ball_l2_nonfinite!(y, r, x, n)
  T = eltype(y)
  isnan(n) && return fill!(y, T(NaN))
  indices, vals = _ball_l2_infinite_entries(x)
  at = 0
  vertex = zero(T)
  @inbounds for (i, v) in zip(indices, vals)
    if isinf(v)
      iszero(at) || return fill!(y, T(NaN))
      at = i
      vertex = T(copysign(float(r), float(v)))
    end
  end
  _ball_l2_zero!(y)
  y[at] = vertex
  y
end
