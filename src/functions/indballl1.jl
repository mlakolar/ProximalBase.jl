"""
    IndBallL1(r = 1)

Indicator of the ℓ1 ball `Σᵢ|xᵢ| ≤ r` of radius `r ≥ 0`. For a matrix, the sum
runs over all entries. The value is `0` inside the ball and `Inf` outside. A
`NaN` entry gives value `NaN`.

The proximal operator is the Euclidean projection onto the ball. It projects
at any nonnegative scalar step, including `0` and `Inf`. A negative or `NaN`
step throws before writing. Exact input/output identity is supported. A
feasible point is copied, signed zeros included. A finite point outside the
ball is soft thresholded at the level `τ > 0` with `Σᵢ max(|xᵢ| - τ, 0) = r`,
and entries at or below the level become `+0`.

The ball compares the computed sum of magnitudes with the radius, without a
feasibility tolerance, so rounding can decide a point at the boundary:
`IndBallL1(1.0)([1.0, 2.0^-54])` is `0.0`, and the projection copies the point.

Zero radius projects to positive zeros, and infinite radius copies the
input. At a positive finite radius a single infinite entry, with no
`NaN`, projects to the signed radius at that entry and zeros elsewhere.
Several infinite entries or a `NaN` entry produce an all-`NaN` projection.

The value reads stored entries of a `SparseIterate`. With sparse input and
output buffers, the projection visits stored entries except when it fills
the logical array with `NaN`. A `BlockIterate` uses its full vector.
`SymmetricSparseIterate` inputs and outputs are unsupported.

The level search needs one buffer entry per magnitude it reads. It can use a
distinct dense `Array` output when the input is a nonoverlapping array or
view of an array. Other level searches and the tiny-radius fallback use a
buffer in the object. Use a separate `IndBallL1(f.r)` in each task whose calls
may need the object's buffer.

# Examples
```jldoctest
julia> prox(IndBallL1(3.0), [3.0, -1.0])
2-element Vector{Float64}:
  2.5
 -0.5
```

See [the numerical contract](@ref numerics) for large sums, tiny radii and
the value returned by [`value_and_prox!`](@ref).
"""
struct IndBallL1{T<:Real} <: AbstractFunction
  r::T
  scratch::Scratch

  IndBallL1{T}(r::T) where {T<:Real} = (_checknonneg("r", r); new{T}(r, Scratch()))
end
IndBallL1(r::Real = 1) = IndBallL1{typeof(r)}(r)

isconvex(::Type{<:IndBallL1}) = true
isproximable(::Type{<:IndBallL1}) = true
_isindicator(::Type{<:IndBallL1}) = true

# The buffer is not part of the function.
Base.:(==)(f::IndBallL1, g::IndBallL1) = f.r == g.r
Base.hash(f::IndBallL1, h::UInt) = hash(iszero(f.r) ? zero(f.r) : f.r, hash(:IndBallL1, h))
Base.show(io::IO, f::IndBallL1) = print(io, "IndBallL1(", f.r, ")")

# Return (s, k, m) with ‖x‖₁ ≈ s*2^k. The ordinary sum has k = 0 and m = s.
# NaN and stored infinity also have k = 0. For finite entries whose sum exceeds
# floatmax/2, s is recomputed in units of 2^k and m is the largest magnitude.
@inline function _ball_l1_norm(v, ::Type{T}) where {T}
  s = _linf_sum(v, _AbsRead{T}(), T)
  (isnan(s) || s <= floatmax(T) / 2) && return (s, 0, s)
  _ball_l1_scaled_norm(v, T)
end

@noinline function _ball_l1_scaled_norm(v, ::Type{T}) where {T}
  m = _linf_classify(v, _AbsRead{T}(), T)[2]
  isinf(m) && return (m, 0, m)
  k = _linf_exponent(length(v))
  while true
    s = _linf_sum(v, _ScaledAbsRead{T}(k), T)
    s <= floatmax(T) / 2 && return (s, k, m)
    k += 1
  end
end

@inline _ball_l1_inside(a, s, k) = s <= (k == 0 ? a : ldexp(a, -k))

function (f::IndBallL1)(x::AbstractArray)
  _check_one_based(x)
  _checkreal(f, x)
  T = _valtype(x)
  r = f.r
  if iszero(r) || isinf(r)
    m = _maximum_abs(x)
    return isnan(m) ? T(NaN) : (isinf(r) || iszero(m) ? zero(T) : T(Inf))
  end
  s, k, _ = _ball_l1_norm(_stored_values(x), T)
  isnan(s) && return T(NaN)
  isfinite(s) && _ball_l1_inside(convert(T, r), s, k) ? zero(T) : T(Inf)
end

@noinline _ball_l1_symmetric_error() = throw(ArgumentError(
  "IndBallL1 does not support SymmetricSparseIterate; use a dense matrix such as Matrix(x)"))
(f::IndBallL1)(::SymmetricSparseIterate) = _ball_l1_symmetric_error()
prox(f::IndBallL1, ::SymmetricSparseIterate, ::Real = 1) = _ball_l1_symmetric_error()

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndBallL1,
                     y::_ArrayOrSymmetricIterate, x::_ArrayOrSymmetricIterate)
  (y isa SymmetricSparseIterate || x isa SymmetricSparseIterate) && _ball_l1_symmetric_error()
  _check_l2_arrays(f, y, x, 0)
  true
end

function prox!(y::_ArrayOrSymmetricIterate, f::IndBallL1, x::_ArrayOrSymmetricIterate, γ::Real)
  (y isa SymmetricSparseIterate || x isa SymmetricSparseIterate) && _ball_l1_symmetric_error()
  _check_l2_arrays(f, y, x, γ)
  _ball_l1_project!(y, f, x)
  y
end

# The value of the stored output, which rounding can put outside the ball.
function value_and_prox!(y::_ArrayOrSymmetricIterate, f::IndBallL1, x::_ArrayOrSymmetricIterate, γ::Real)
  prox!(y, f, x, γ)
  f(y)
end
_value_and_prox_unchecked!(y::_ArrayOrSymmetricIterate, f::IndBallL1,
                           x::_ArrayOrSymmetricIterate, γ::Real) = value_and_prox!(y, f, x, γ)

# For an exact normal level τ, let τ̂ be its rounding with unit roundoff u.
# If k entries exceed the smaller level, rounding changes the projected mass
# by at most k*u*τ. Their total magnitude is at least k*(1-u)*τ, so this error
# is at most u/(1-u) times the exact sum of magnitudes. For a > 64*eps(T)*s
# this contribution is below (exact sum / s)/(128*(1-u)) of a. This screen
# uses the caller's sum in the units of the level search. The selector's own
# absolute error and subnormal rounding are separate.
@inline _ball_l1_may_cancel(a::T, s::T) where {T} = a <= 64 * eps(T) * s

# The endpoints and nonfinite inputs reuse the IndBallL2 routes. The magnitudes
# are read from the stored values: for a SparseIterate the implicit zeros add
# nothing to the sum and are never above a positive level.
function _ball_l1_project!(y, f::IndBallL1, x)
  isempty(x) && return y
  r = f.r
  iszero(r) && return _ball_l2_zero!(y)
  isinf(r) && return _ball_l2_copy!(y, x)
  T = _valtype(x)
  a = convert(T, r)
  s, k, m = _ball_l1_norm(_stored_values(x), T)
  isfinite(s) || return _ball_l2_nonfinite!(y, r, x, s)
  _ball_l1_inside(a, s, k) && return _ball_l2_copy!(y, x)
  if length(x) == 1
    @inbounds y[1] = copysign(a, convert(T, x[1]))
    return y
  end
  v = _stored_values(x)
  k == 0 || return _ball_l1_scaled_project!(y, f, x, v, a, s, k, m)
  rd = _AbsRead{T}()
  if a < floatmin(T) || _ball_l1_may_cancel(a, s)
    return _ball_l1_check_cancellation!(y, f, x, v, a, a, s, 0, T(_maximum_abs(v)), rd)
  end
  _ball_l1_shrink!(y, f, x, _ball_l1_threshold(y, f, v, a, s, rd))
end

# A finite sum above floatmax/2: the level is selected for the entries and the
# radius scaled by 2^-k, where s is the scaled sum, and multiplied back by 2^k.
# The projection is positively homogeneous, so this is the same level.
@noinline function _ball_l1_scaled_project!(y, f::IndBallL1, x, v, a::T, s::T, k::Int, m::T) where {T}
  a′ = ldexp(a, -k)
  rd = _ScaledAbsRead{T}(k)
  if _ball_l1_may_cancel(a′, s)
    return _ball_l1_check_cancellation!(y, f, x, v, a, a′, s, k, m, rd)
  end
  τ = _ball_l1_threshold(y, f, v, a′, s, rd)
  # Scaling the largest read magnitude back can round above m.
  _ball_l1_shrink!(y, f, x, min(ldexp(τ, k), m))
end

# A small radius against the sum can still be large against the active entries.
# In that case, inspect the mass before sorting or overwriting the input.
@noinline function _ball_l1_check_cancellation!(y, f, x, v, a::T, a′::T, s::T, k, m::T, rd) where {T}
  a <= 8 * eps(T) * m && return _ball_l1_water_fill!(y, f, x, v, a)
  τ = _ball_l1_threshold(y, f, v, a′, s, rd)
  τ = k == 0 ? τ : min(ldexp(τ, k), m)
  mass = _linf_sum(v, w -> max(abs(T(w)) - τ, zero(T)), T)
  tolerance = a < floatmin(T) ? zero(T) : 8 * eps(T) * a
  abs(mass - a) > tolerance && return _ball_l1_water_fill!(y, f, x, v, a)
  _ball_l1_shrink!(y, f, x, τ)
end

# The level from the shared ℓ1-ball threshold of NormLinf, clamped at zero.
# Under `_linf_buffer_is_output`, the output serves as the buffer before the
# shrink overwrites it. Otherwise the object's buffer is used.
@inline function _ball_l1_threshold(y, f::IndBallL1, v, a::T, s::T, rd) where {T}
  τ = if _linf_buffer_is_output(y, v, T)
    _l1ball_threshold!(y, v, a, rd, s)
  else
    _l1ball_threshold!(_array_buffer!(f.scratch, T, (length(v),)), v, a, rd, s)
  end
  max(τ, zero(T))
end

# An indexed view can refer to a slot more than once. Once the level has
# been chosen, reuse the magnitude buffer for the original input entries.
@inline function _ball_l1_write_input(y, f, x, ::Type{T}) where {T}
  if !(x isa Union{StridedArray, SparseIterate, BlockIterate}) && y === x
    return _ball_l1_snapshot!(f, x, T)
  end
  x
end

@noinline function _ball_l1_snapshot!(f, x, ::Type{T}) where {T}
  buf = _array_buffer!(f.scratch, T, (length(x),))
  @inbounds for i in 1:length(x)
    buf[i] = x[i]
  end
  buf
end

# y[i] = sign(x[i]) max(|x[i]| - τ, 0) for τ ≥ 0, with +0 at and below the level
@inline _ball_l1_shrink!(y, f, x, τ::T) where {T} =
  _ball_l1_shrink_loop!(y, _ball_l1_write_input(y, f, x, T), τ)

@inline function _ball_l1_shrink_loop!(y::_DenseOutput, x, τ::T) where {T}
  S = eltype(y)
  @inbounds @simd for i in 1:length(x)
    y[i] = convert(S, _shrink_nonneg(convert(T, x[i]), τ))
  end
  y
end
_ball_l1_shrink_loop!(y, x, τ::T) where {T} = _ball_l2_map!(v -> _shrink_nonneg(convert(T, v), τ), y, x)

# For x = [M, -M] with M = floatmax(Float64) and r = 1, the exact level is
# M - 1/2. Storing that level in Float64 gives M and erases the projection.
#
# The magnitudes are sorted in decreasing order b₁ ≥ … ≥ bₙ, with bₙ₊₁ = 0,
# and the radius is spread over the gaps. Lowering the level from bⱼ to bⱼ₊₁
# uses j (bⱼ - bⱼ₊₁) of the radius. At the first j whose gap holds the
# remaining radius ρ, the top j entries become (|xᵢ| - bⱼ) + ρ/j, which never
# forms the level bⱼ - ρ/j. Ties give empty gaps. A gap whose product
# j (bⱼ - bⱼ₊₁) overflows to Inf holds any finite ρ. If rounding leaves radius
# after the last gap, the entries are kept. This path sorts a copy of the
# magnitudes in the object's buffer, in O(n log n).
@noinline function _ball_l1_water_fill!(y, f::IndBallL1, x, v, a::T) where {T}
  n = length(v)
  b = _array_buffer!(f.scratch, T, (n,))
  @inbounds for i in 1:n
    b[i] = abs(convert(T, v[i]))
  end
  sort!(b; rev = true)
  ρ = a
  @inbounds for j in 1:n
    level = b[j]
    gap = j * (level - (j < n ? b[j + 1] : zero(T)))
    if ρ <= gap
      source = _ball_l1_write_input(y, f, x, T)
      # At a subnormal radius the active gaps and the remaining radius are
      # small. Lift them before dividing so the share can be added to a gap
      # before rounding back to the subnormal spacing.
      lift = a < floatmin(T) ? ldexp(one(T), precision(T)) : one(T)
      share = (ρ * lift) / j
      unit = inv(lift)
      return _ball_l2_map!(w -> _ball_l1_fill_entry(T(w), level, share, lift, unit), y, source)
    end
    ρ -= gap
  end
  _ball_l2_copy!(y, x)
end

@inline function _ball_l1_fill_entry(v::T, level::T, share::T, lift::T, unit::T) where {T}
  d = abs(v) - level
  d < zero(T) && return zero(T)
  z = (d * lift + share) * unit
  iszero(z) ? zero(T) : copysign(z, v)
end
