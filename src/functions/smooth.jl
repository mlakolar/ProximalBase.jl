####################################
#
#  Smooth functions that are also proximable
#
####################################

####################################
#  zero
####################################

"""
    ZeroFunction()

The zero function `f(x) = 0`: smooth, with gradient zero, and proximable, with
the identity as proximal operator. The penalty of an unpenalized problem.
"""
struct ZeroFunction <: AbstractFunction end

issmooth(::Type{ZeroFunction}) = true
isproximable(::Type{ZeroFunction}) = true
isseparable(::Type{ZeroFunction}) = true
isconvex(::Type{ZeroFunction}) = true
iszeropreserving(::ZeroFunction) = true
_checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)},
            f::ZeroFunction, y::AbstractArray, x::AbstractArray) =
  (_checkreal(f, x); _checkreal(f, y); (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x); _checksize(y, x); true)

(f::ZeroFunction)(x::AbstractArray) = (_checkreal(f, x); zero(_valtype(x)))

function prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  y === x || copyto!(y, x)
  y
end
value_and_prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::Real) =
  (_prox_unchecked!(y, f, x, γ); zero(_valtype(y)))

function value_and_gradient!(g::AbstractArray, f::ZeroFunction, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  fill!(g, zero(eltype(g)))
  zero(_valtype(x))
end
gradient!(g::AbstractArray, f::ZeroFunction, x::AbstractArray) = (_value_and_gradient_unchecked!(g, f, x); g)

prox_entry(::ZeroFunction, v::Real, ::Integer, ::Real) = v
kkt_violation(::ZeroFunction, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]))
restrict(f::ZeroFunction, ::AbstractVector{<:Integer}) = f

####################################
#  linear
####################################

"""
    LinearFunction(c, d = 0)

The linear function `f(x) = ⟨c, x⟩ + d`, with gradient `c`. Its proximal operator
is the shift `x - γc`.
"""
struct LinearFunction{C<:AbstractArray, D<:Real} <: AbstractFunction
  c::C
  d::D
  function LinearFunction{C, D}(c, d) where {C<:AbstractArray, D<:Real}
    eltype(c) <: Real || _real_error(:LinearFunction)
    new{C, D}(c, d)
  end
end
LinearFunction(c::AbstractArray, d::Real) = LinearFunction{typeof(c), typeof(d)}(c, d)

LinearFunction(c::AbstractArray) = (eltype(c) <: Real || _real_error(:LinearFunction); LinearFunction(c, zero(float(eltype(c)))))

issmooth(::Type{<:LinearFunction}) = true
isproximable(::Type{<:LinearFunction}) = true
isseparable(::Type{<:LinearFunction}) = true
isconvex(::Type{<:LinearFunction}) = true

(f::LinearFunction)(x::AbstractArray) =
  (_check_value_arrays(f, x); _value_checked(f, x))
_value_unchecked(f::LinearFunction, x::AbstractArray) = (_checkvalue(f, x); _value_at(dot(f.c, x) + f.d, x))

# Generic dot only needs equal lengths; specialized matrix methods check shapes.
_dot_checks_size(::Type{C}, ::Type{X}) where {C, X} =
  (C <: Diagonal && X <: AbstractMatrix) ||
  (C <: AbstractMatrix && X <: Diagonal) ||
  (C <: StridedMatrix && X <: SymmetricSparseIterate) ||
  (C <: SymmetricSparseIterate && X <: StridedMatrix) ||
  (C <: _SparseDotCSC && X <: Union{_SparseDotCSC, _SparseDotMatrix}) ||
  (C <: _SparseDotMatrix && X <: _SparseDotCSC) ||
  (_SparseWrappedColumnDotChecksSize &&
   ((C <: _SparseDotWrappedCSC && X <: _SparseDotCSC) ||
    (C <: _SparseDotCSC && X <: _SparseDotWrappedCSC)))
_dot_checks_size(::Type{<:Adjoint{<:Real, C}}, ::Type{<:Adjoint{<:Real, X}}) where {C, X} = _dot_checks_size(C, X)
_dot_checks_size(::Type{<:Transpose{<:Any, C}}, ::Type{<:Transpose{<:Any, X}}) where {C, X} = _dot_checks_size(C, X)
_dot_checks_size(::Type{<:Adjoint{<:Any, C}}, ::Type{<:Transpose{<:Any, X}}) where {C, X} =
  _SparseMixedDotChecksSize && C <: _SparseDotCSC && X <: _SparseDotCSC
_dot_checks_size(::Type{<:Transpose{<:Any, C}}, ::Type{<:Adjoint{<:Any, X}}) where {C, X} =
  _SparseMixedDotChecksSize && C <: _SparseDotCSC && X <: _SparseDotCSC
_checkdot(c::C, x::X) where {C, X} = (_dot_checks_size(C, X) ? _checksize(c, x) : _checkparam(c, x))
_checkvalue(f::LinearFunction, x) = (_checkreal(f, x); _checkdot(f.c, x))

function _checkwrite(op::O, f::LinearFunction, y::AbstractArray, x::AbstractArray) where {O}
  _checkreal(f, x)
  _checkreal(f, y)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(f, x)
  op === value_and_prox! && _checkdot(f.c, y)
  op === value_and_gradient! && _checkdot(f.c, x)
  # c is written into y: copied by the gradient, subtracted by the prox
  _check_parameter_axes(y, f.c)
  true
end

prox!(y::AbstractArray, f::LinearFunction, x::AbstractArray, γ::Real) =
  (_check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::LinearFunction, x::AbstractArray, γ::Real)
  _checkwrite(prox!, f, y, x)
  γ = convert(float(eltype(y)), γ)
  if size(f.c) == size(x)
    y .= x .- γ .* _checked_parameter(y, f.c)
  else
    # dot accepts equal-length arrays even when their shapes differ.
    @inbounds for i in eachindex(IndexLinear(), y, x)
      y[i] = x[i] - γ * f.c[i]
    end
  end
  y
end

value_and_gradient!(g::AbstractArray, f::LinearFunction, x::AbstractArray) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
function _value_and_gradient_unchecked!(g::AbstractArray, f::LinearFunction, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  v = _value_unchecked(f, x)
  _copy_parameter!(g, f.c)
  v
end
gradient!(g::AbstractArray, f::LinearFunction, x::AbstractArray) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
_gradient_unchecked!(g::AbstractArray, f::LinearFunction, x::AbstractArray) =
  (_checkwrite(gradient!, f, g, x); _copy_parameter!(g, f.c))

Base.@propagate_inbounds prox_entry(f::LinearFunction, v::Real, i::Integer, γ::Real) = v - γ * f.c[i]
_checkparams(f::LinearFunction, x) = _checkparam(f.c, x)
kkt_violation(f::LinearFunction, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i] + f.c[i]))
restrict(f::LinearFunction, S::AbstractVector{<:Integer}) = LinearFunction(f.c[S], f.d)

####################################
#  squared Euclidean norm
####################################

"""
    SqrNormL2(λ = 1)
    SqrNormL2(λ, y)

The squared Euclidean norm `f(x) = (λ/2) ‖x - y‖²`, with `y = 0` when omitted:
the ridge penalty, and for `λ = 1` the loss `‖x - y‖² / 2`. Smooth, with
gradient `λ(x - y)`, separable, and proximable, with proximal operator
`(x + γλy) / (1 + γλ)`. For a matrix `x`, `‖⋅‖` is the Frobenius norm.
Values follow the overflow rule in [Overflow and underflow of values](@ref).

# Precision

The level `λ`, the center `y` and the step are parameters; they are rounded once
to the *model precision* of each result ([the numerical contract](@ref numerics),
T3) and never widen it. That precision is the wider of the argument's and the
destination's for a gradient or a proximal point, and the argument's for the value
(so the value written by `value_and_prox!` is the value at the stored point, and
`value_and_gradient!` returns the standalone value and writes the standalone
gradient). The proximal weight `γλ` is formed from the unrounded factors before it
is rounded.

A parameter, a product or a coefficient that would leave the range of the model
precision does not change a representable result (R2, R3): the gradient of
`SqrNormL2(2^-1022, [-2^1023])` at `[2^1023]` is `[4.0]` although `x - y`
overflows, and `prox!` of `SqrNormL2(2^800)` with step `2^800` at `[2^900]`
gives `2^-700` in `Float64`, where `1/(1 + γλ)` is not a number. The ordinary
computation runs first, and the recovery is chosen once per call from the
parameters, except for a `Float64` center entry outside the range of a `Float32`
model, which is handled for that entry.
"""
struct SqrNormL2{T<:Real, Y<:Union{Nothing, AbstractArray}} <: AbstractFunction
  λ::T
  y::Y

  function SqrNormL2{T, Y}(λ::T, y::Y) where {T<:Real, Y<:Union{Nothing, AbstractArray}}
    y === nothing || eltype(y) <: Real || _real_error(:SqrNormL2)
    _checknonneg("λ", λ)
    new{T, Y}(λ, y)
  end
end
SqrNormL2(λ::Real = 1, y::Union{Nothing, AbstractArray} = nothing) = SqrNormL2{typeof(λ), typeof(y)}(λ, y)

issmooth(::Type{<:SqrNormL2}) = true
isproximable(::Type{<:SqrNormL2}) = true
isseparable(::Type{<:SqrNormL2}) = true
isconvex(::Type{<:SqrNormL2}) = true
iszeropreserving(f::SqrNormL2) = f.y === nothing && isfinite(f.λ)

# Σ xᵢ² over the entries of x, in O(nnz) for the sparse iterates
_sumabs2(x::AbstractArray) = eltype(x) <: Integer ? sum(v -> abs2(float(v)), x) : sum(abs2, x)
function _sumabs2(x::SparseIterate)
  s = zero(float(eltype(x)))
  @inbounds @simd for k in 1:x.nnz
    s += abs2(float(x.nzval[k]))
  end
  s
end
function _sumabs2(x::SymmetricSparseIterate{T}) where {T}
  s = zero(float(T))
  @inbounds for k = 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    s += (r == c ? 1 : 2) * abs2(float(x.nzval[k]))
  end
  s
end
_sumabs2(x::AtomIterate) = _sumabs2(x.storage)

_square_term_count(x::Union{SparseIterate, SymmetricSparseIterate}) = x.nnz
_square_term_count(x::AtomIterate) = _square_term_count(x.storage)
function _square_multiplicity(x::SymmetricSparseIterate, i)
  r, c = tril_cartesian(x.p, x.nzval2ind[i])
  r == c ? 1 : 2
end

# Keep the vectorized sum-of-squares path for ordinary magnitudes; use a
# scaled norm when squaring overflows or loses the subnormal entries.
function _norm2(x::AbstractArray)
  eltype(x) <: AbstractFloat || return _norm2_scaled(x)
  s = _sumabs2(x)
  isfinite(s) && s >= floatmin(typeof(s)) && return sqrt(s)
  iszero(s) && x isa DenseArray && iszero(sum(abs, x)) && return s
  _norm2_scaled(x)
end
_norm2_scaled(x::AbstractArray) = norm(x)
_norm2_scaled(x::AtomIterate) = norm(x.storage)
function _norm2_scaled(x::SymmetricSparseIterate{T}) where {T}
  n = zero(float(T))
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    isnan(x.nzval[k]) && return oftype(n, NaN)
    n = hypot(n, x.nzval[k])
    r == c || (n = hypot(n, x.nzval[k]))
  end
  n
end

(f::SqrNormL2)(x::AbstractArray) =
  (_check_value_arrays(f, x); _value_checked(f, x))
function _value_unchecked(f::SqrNormL2, x::AbstractArray)
  _checkvalue(f, x)
  T = _valtype(x)
  iszero(f.λ) && return zero(T)
  y = f.y
  # A center is a parameter: it is rounded to the model precision of the value, the
  # argument's (T3), so a Float64 center does not widen a Float32 value.
  T === Float32 && y !== nothing && eltype(y) === Float64 && return _ridge_value_wide_center(f.λ, x, y)
  _square_value(f.λ, x, y)
end

_checkvalue(f::SqrNormL2, x) = (_checkreal(f, x); _checkparams(f, x); f.y === nothing || _checksize(f.y, x); true)

# Does a center entry keep its range when rounded to the model precision W (R3)?
# Branchless, so that a loop over the entries still vectorizes. Free for a center
# already stored in W; non-finite and zero entries convert without loss.
@inline _ridge_fits(::Type{W}, c::W) where {W} = true
@inline _ridge_fits(::Type{W}, c) where {W} = _level_fits(W, c)
@inline function _ridge_fits(::Type{Float32}, c::Float64)
  a = abs(c)
  ((a >= Float64(floatmin(Float32))) & (a <= Float64(floatmax(Float32)))) | iszero(a) | !isfinite(a)
end
# Did rounding c to cw underflow (to zero or a subnormal)? An overflow gives an infinite
# cw, which reaches the result and is caught there.
@inline _ridge_underflows(::Type{W}, c, cw) where {W} = false
@inline _ridge_underflows(::Type{Float32}, c::Float64, cw::Float32) = (abs(cw) < floatmin(Float32)) & (c != 0)
# Can rounding the centers of eltype C to W leave W's range?
@inline _ridge_center_wider(::Type{W}, ::Type{C}) where {W, C} = promote_type(W, float(C)) !== W
# Every center entry keeps its range on rounding to W. One pass over c only, so an
# output that aliases the argument is still unwritten when a fallback is chosen.
function _ridge_all_fit(::Type{W}, c) where {W}
  ok = true
  for ci in c
    ok &= _ridge_fits(W, ci)
  end
  ok
end
# Float64 centers, Float32 model: the largest magnitude and the smallest nonzero
# magnitude decide (conservatively), compared as bit patterns, which vectorize.
# b - 1 wraps for b = 0, so zeros do not enter the minimum. NaN and infinities
# fail the test and take the checked path, which converts them without loss.
const _F32_MAX_BITS = reinterpret(UInt64, Float64(floatmax(Float32)))
const _F32_MIN_BITS = reinterpret(UInt64, Float64(floatmin(Float32)))
function _ridge_all_fit(::Type{Float32}, c::AbstractArray{Float64})
  hi = zero(UInt64)
  lo = typemax(UInt64)
  @inbounds @simd for i in eachindex(c)
    b = reinterpret(UInt64, c[i]) & 0x7fffffffffffffff
    hi = max(hi, b)
    lo = min(lo, b - one(UInt64))
  end
  hi <= _F32_MAX_BITS && lo >= _F32_MIN_BITS - one(UInt64)
end

# Σ (xᵢ - round(cᵢ))² for a Float32 argument and a Float64 center. The ordinary
# computation runs in Float32 when every center entry keeps its range on rounding;
# otherwise the centers that fit are rounded and the scaled recovery of
# _square_value sees them in Float64 (R3).
function _ridge_value_wide_center(λ, x, y)
  T = Float32
  s = zero(T)
  u = false
  @inbounds @simd for i in eachindex(x, y)
    cw = T(y[i])
    s += abs2(T(x[i]) - cw)
    u |= _ridge_underflows(T, y[i], cw)
  end
  v = convert(T, λ) * s / 2
  if !u & isfinite(v) & (s >= length(x) * (floatmin(T) / eps(T))) && _level_fits(T, λ)
    return _value_at(v, x)
  end
  # a rounded center left the range, or the ordinary value is out of range
  _exceptional_square_value(λ, x, map(c -> _ridge_fits(T, c) ? Float64(T(c)) : c, y), s, v)
end

value_and_gradient!(g::AbstractArray, f::SqrNormL2, x::AbstractArray) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
function _value_and_gradient_unchecked!(g::AbstractArray, f::SqrNormL2, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  _ridge_gradient!(g, f.λ, f.y, x)
  _value_unchecked(f, x)
end
gradient!(g::AbstractArray, f::SqrNormL2, x::AbstractArray) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
function _gradient_unchecked!(g::AbstractArray, f::SqrNormL2, x::AbstractArray)
  _checkwrite(gradient!, f, g, x)
  _ridge_gradient!(g, f.λ, f.y, x)
end

# gᵢ = λ(xᵢ - cᵢ) in the model precision W of the gradient: the wider of the
# destination and the argument. λ and c are rounded to W (T3), except where the
# rounding leaves W's range (R3); an overflowing xᵢ - cᵢ does not hide a finite
# gradient entry (R2).
function _ridge_gradient!(g, λ, c, x)
  W = _model_precision(eltype(g), eltype(x))
  if _level_fits(W, λ)
    l = convert(W, λ)
    c === nothing && return g .= l .* x
    cp = _checked_parameter(g, c)
    if g isa DenseArray && x isa DenseArray && cp isa DenseArray && size(g) == size(x) == size(cp)
      _ridge_gradient_dense!(g, l, x, cp)
    else
      g .= _ridge_gradient_entry.(l, x, cp)
    end
  else
    R = promote_type(W, float(typeof(λ)))
    l = convert(R, λ)
    g .= _ridge_gradient_wide.(W, l, x, c === nothing ? zero(W) : _checked_parameter(g, c))
  end
  g
end
# The ordinary loop, which vectorizes; when a sum of the entries is not finite (an
# overflowing difference or product, or a center that overflowed on rounding) or a
# center underflowed, every entry is recomputed one by one. The gradient never aliases x.
function _ridge_gradient_dense!(g, l::W, x, c) where {W}
  t = zero(W)       # a non-finite entry makes the sum non-finite
  u = false
  @inbounds @simd for i in eachindex(g)
    cw = W(c[i])
    r = l * (W(x[i]) - cw)
    g[i] = r
    t += r
    u |= _ridge_underflows(W, c[i], cw)
  end
  (u | !isfinite(t)) && _ridge_gradient_repair!(g, l, x, c)
  g
end
@noinline function _ridge_gradient_repair!(g, l::W, x, c) where {W}
  @inbounds for i in eachindex(g)
    g[i] = _ridge_gradient_entry(l, x[i], c[i])
  end
end
@inline function _ridge_gradient_entry(l::W, x, c) where {W}
  _ridge_fits(W, c) || return _ridge_gradient_wide(W, l, x, c)
  r = l * (W(x) - W(c))
  isfinite(r) ? r : _ridge_gradient_overflow(l, W(x), W(c), r)
end
@noinline _ridge_gradient_overflow(l, x, c, r) =
  isfinite(x) && isfinite(c) ? _scaled_product(l, x / 2 - c / 2, 1) : r
# l is a level wider than W, or c is a center wider than W: the entry is computed in R
function _ridge_gradient_wide(::Type{W}, l, x, c) where {W}
  R = promote_type(W, typeof(l), float(typeof(c)))
  cw = _ridge_fits(W, c) ? R(W(c)) : R(c)
  r = R(l) * (R(x) - cw)
  isfinite(r) ? r : _ridge_gradient_overflow(R(l), R(x), cw, r)
end

prox!(z::AbstractArray, f::SqrNormL2, x::AbstractArray, γ::Real) =
  (_check_oracle_arrays(prox!, f, z, x, γ); _prox_unchecked!(z, f, x, γ))
function _prox_unchecked!(z::AbstractArray, f::SqrNormL2, x::AbstractArray, γ::Real)
  _checkwrite(prox!, f, z, x)
  W = _model_precision(eltype(z), eltype(x))
  c = f.y === nothing ? nothing : _checked_parameter(z, f.y)
  if _ridge_needs_scaling(W, γ, f.λ)
    _ridge_prox!(z, _ridge_scaled(W, γ, f.λ), x, c)
  else
    _ridge_prox!(z, _ridge_ordinary(W, γ, f.λ), x, c)
  end
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}, f::SqrNormL2, y::AbstractArray, x::AbstractArray)
  _checkreal(f, x)
  _checkreal(f, y)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(f, x)
  f.y === nothing || (_checksize(f.y, x); _check_parameter_axes(y, f.y))
  true
end

# The convex weights 1/(1+γλ), γλ/(1+γλ), without overflowing γλ or
# the numerator x + γλy of a centered proximal operator. Shared with Regularize,
# the distance functions, the envelope and the metric prox; the ridge itself uses
# _ridge_ordinary and _ridge_scaled below.
@inline function _ridge_weights(γ::T, λ::T) where {T}
  t = γ * λ
  if t <= one(T)
    d = one(T) + t
    return inv(d), t / d
  end
  ρ = inv(γ) / λ
  d = one(T) + ρ
  ρ / d, inv(d)
end
# Interpolate same-sign endpoints from the endpoint with larger weight,
# so rounding in two weighted products cannot overflow near floatmax.
@inline function _ridge_blend(x, y, a::T, b::T) where {T}
  x, y = T(x), T(y)
  if signbit(x) == signbit(y)
    return b <= a ? muladd(b, y - x, x) : muladd(a, x - y, y)
  end
  a * x + b * y
end

# The ridge proximal point (x + γλc)/(1 + γλ) = a x + b c with the convex weights
# a = 1/(1+t), b = t/(1+t), t = γλ. t is formed once from the unrounded factors in
# P = promote_type(W, factors) and rounded to the model precision W (T3).
#
# Ordinary: t, 1/t and both weights are normal numbers in W (or a factor is zero or
# not finite, which keeps its earlier meaning). Scaled (R3): t = τ·2^k from the
# significands; the smaller weight is carried as m·2^e and applied last, so a
# coefficient that is not representable still acts on the coordinate.
struct _RidgeOrdinary{W<:Real}
  a::W
  b::W
  t::W
end
struct _RidgeScaled{W<:Real, P<:Real}
  L::W              # the larger weight, in [1/2, 1]
  m::W              # significand of the smaller weight, in (0, 1]
  e::Int
  small_x::Bool     # the smaller weight multiplies x (t > 1); else the center
  γ::P
  λ::P
end

@inline function _ridge_weights_t(t::W) where {W}
  if t <= one(W)
    d = one(W) + t
    return inv(d), t / d
  end
  ρ = inv(t)
  d = one(W) + ρ
  ρ / d, inv(d)
end

# The decision is made once per call, from the factors, not from t: the product of
# finite nonzero factors can underflow or overflow in P.
@inline function _ridge_needs_scaling(::Type{W}, γ, λ) where {W}
  W <: Union{Float32, Float64} && isfinite(γ) && isfinite(λ) && !iszero(γ) && !iszero(λ) || return false
  P = promote_type(W, float(typeof(γ)), float(typeof(λ)))
  t = P(γ) * P(λ)
  !(P(_ridge_band_low(W)) <= t <= P(_ridge_band_high(W)))
end
# t, 1/t and both convex weights are normal in W for t in [2^-126, 2^125] (Float32)
# and [2^-1022, 2^1021] (Float64)
@inline _ridge_band_low(::Type{Float32}) = reinterpret(Float32, 0x00800000)          # 2^-126
@inline _ridge_band_high(::Type{Float32}) = reinterpret(Float32, 0x7e000000)         # 2^125
@inline _ridge_band_low(::Type{Float64}) = reinterpret(Float64, 0x0010000000000000)   # 2^-1022
@inline _ridge_band_high(::Type{Float64}) = reinterpret(Float64, 0x7fc0000000000000)  # 2^1021
@inline function _ridge_ordinary(::Type{W}, γ, λ) where {W}
  P = promote_type(W, float(typeof(γ)), float(typeof(λ)))
  tw = convert(W, P(γ) * P(λ))
  a, b = _ridge_weights_t(tw)
  _RidgeOrdinary{W}(a, b, tw)
end
@inline _ridge_scaled(::Type{W}, γ, λ) where {W} =
  (P = promote_type(W, float(typeof(γ)), float(typeof(λ))); _ridge_scaled(W, P(γ), P(λ)))

Base.@assume_effects :effect_free :nothrow :terminates_globally @noinline function _ridge_scaled(::Type{W}, γ::P, λ::P) where {W, P}
  mγ, eγ = frexp(γ)
  mλ, eλ = frexp(λ)
  τ = convert(W, mγ * mλ)                    # in [1/4, 1]
  k = eγ + eλ
  if k > 0
    r = inv(τ)
    d = one(W) + r * ldexp(one(W), -k)
    _RidgeScaled{W, P}(inv(d), r / d / 4, 2 - k, true, γ, λ)
  else
    d = one(W) + τ * ldexp(one(W), k)
    _RidgeScaled{W, P}(inv(d), τ / d, k, false, γ, λ)
  end
end

@inline function _ridge_scaled_blend(x::W, c::W, s::_RidgeScaled{W}) where {W}
  big, small = s.small_x ? (c, x) : (x, c)
  signbit(big) == signbit(small) ? big + ldexp(s.m * (small - big), s.e) :
                                   s.L * big + ldexp(s.m * small, s.e)
end

@inline _ridge_apply(o::_RidgeOrdinary, x) = o.a * x
@inline _ridge_apply(s::_RidgeScaled{W}, x) where {W} =
  s.small_x ? ldexp(s.m * W(x), s.e) : s.L * W(x)
@inline function _ridge_apply(o::_RidgeOrdinary{W}, x, c) where {W}
  _ridge_fits(W, c) ? _ridge_blend(x, c, o.a, o.b) : _ridge_apply_wide(o, x, c)
end
@inline function _ridge_apply(s::_RidgeScaled{W}, x, c) where {W}
  _ridge_fits(W, c) ? _ridge_scaled_blend(W(x), W(c), s) : _ridge_apply_wide(s, x, c)
end
# the same when every center entry is known to fit: no per-entry test
@inline _ridge_apply_fit(o::_RidgeOrdinary, x, c) = _ridge_blend(x, c, o.a, o.b)
@inline _ridge_apply_fit(s::_RidgeScaled{W}, x, c) where {W} = _ridge_scaled_blend(W(x), W(c), s)
# A center outside W's range on rounding: the entry is computed in the wider type
@noinline function _ridge_apply_wide(o::_RidgeOrdinary{W}, x, c) where {W}
  R = promote_type(W, float(typeof(c)))
  a, b = _ridge_weights_t(convert(R, o.t))
  W(_ridge_blend(R(x), R(c), a, b))
end
@noinline function _ridge_apply_wide(s::_RidgeScaled{W}, x, c) where {W}
  R = promote_type(W, float(typeof(c)))
  W(_ridge_scaled_blend(R(x), R(c), _ridge_scaled(R, promote_type(R, typeof(s.γ))(s.γ), promote_type(R, typeof(s.λ))(s.λ))))
end

_ridge_prox!(z, o::_RidgeOrdinary, x, ::Nothing) = (z .= o.a .* x)
_ridge_prox!(z, s::_RidgeScaled, x, ::Nothing) = (z .= (xi -> _ridge_apply(s, xi)).(x))
function _ridge_prox!(z, co::Union{_RidgeOrdinary{W}, _RidgeScaled{W}}, x, c) where {W}
  # a center wider than W is tested once, before the loop: the pass reads c only,
  # so an aliased output (z === x) is still unwritten when a fallback is chosen
  if _ridge_center_wider(W, eltype(c)) && !_ridge_all_fit(W, c)
    z .= ((xi, ci) -> _ridge_apply(co, xi, ci)).(x, c)
  else
    z .= ((xi, ci) -> _ridge_apply_fit(co, xi, ci)).(x, c)
  end
end

function prox_entry(f::SqrNormL2{<:Real, Nothing}, v::Real, ::Integer, γ::Real)
  W = typeof(float(v))
  _ridge_needs_scaling(W, γ, f.λ) ? _ridge_apply(_ridge_scaled(W, γ, f.λ), W(v)) :
                                    _ridge_apply(_ridge_ordinary(W, γ, f.λ), W(v))
end
Base.@propagate_inbounds function prox_entry(f::SqrNormL2, v::Real, i::Integer, γ::Real)
  W = typeof(float(v))
  _ridge_needs_scaling(W, γ, f.λ) ? _ridge_apply(_ridge_scaled(W, γ, f.λ), W(v), f.y[i]) :
                                    _ridge_apply(_ridge_ordinary(W, γ, f.λ), W(v), f.y[i])
end
_checkparams(f::SqrNormL2, x) = _checkparam(f.y, x)

kkt_violation(::SqrNormL2{<:Real, Nothing}, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]))
kkt_violation(f::SqrNormL2, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i] - f.λ * f.y[i]))

restrict(f::SqrNormL2{<:Real, Nothing}, ::AbstractVector{<:Integer}) = f
restrict(f::SqrNormL2, S::AbstractVector{<:Integer}) = SqrNormL2(f.λ, f.y[S])

####################################
#  quadratic
####################################

"""
    QuadraticFunction(A, b = nothing, c = 0)

The quadratic `f(x) = x'Ax / 2 + b'x + c` for a symmetric positive semidefinite
`A`, with `b = 0` when omitted. Smooth, with gradient `Ax + b`, and proximable:
the first proximal operator computes the eigendecomposition `A = QΛQ'` once,
after which `prox_{γf}(x) = Q (I + γΛ)⁻¹ Q'(x - γb)` costs two products with
`Q` for any `γ`.
"""
mutable struct QuadraticFunction{T<:AbstractFloat, M<:AbstractMatrix, V<:Union{Nothing, AbstractVector}} <: AbstractFunction
  A::M
  b::V
  c::T
  tmp::Vector{T}        # A x, and scratch for the proximal operator
  Λ::Vector{T}          # A = Q Diagonal(Λ) Q', computed by the first prox
  Q::Matrix{T}
  w::Vector{T}

  function QuadraticFunction{T, M, V}(A, b, c, tmp, Λ, Q, w) where {T<:AbstractFloat, M<:AbstractMatrix,
      V<:Union{Nothing, AbstractVector}}
    all(a -> a === nothing || eltype(a) <: Real,
        (A, b, tmp, Λ, Q, w)) || _real_error(:QuadraticFunction)
    new{T, M, V}(A, b, c, tmp, Λ, Q, w)
  end
end
QuadraticFunction(A::M, b::V, c::T, tmp::Vector{T}, Λ::Vector{T}, Q::Matrix{T}, w::Vector{T}) where {T<:AbstractFloat,
    M<:AbstractMatrix, V<:Union{Nothing, AbstractVector}} =
  QuadraticFunction{T, M, V}(A, b, c, tmp, Λ, Q, w)

function QuadraticFunction(A::AbstractMatrix, b::Union{Nothing, AbstractVector} = nothing, c::Real = 0)
  eltype(A) <: Real || _real_error(:QuadraticFunction)
  b === nothing || eltype(b) <: Real || _real_error(:QuadraticFunction)
  n = checksquare(A)
  b === nothing || length(b) == n || throw(DimensionMismatch("A is $n × $n and b has length $(length(b))"))
  T = float(promote_type(eltype(A), b === nothing ? eltype(A) : eltype(b), typeof(c)))
  QuadraticFunction{T, typeof(A), typeof(b)}(A, b, T(c), zeros(T, n), T[], Matrix{T}(undef, 0, 0), zeros(T, n))
end

issmooth(::Type{<:QuadraticFunction}) = true
_argument_type(::Type{<:QuadraticFunction}) = AbstractVector
isproximable(::Type{<:QuadraticFunction}) = true
isconvex(::Type{<:QuadraticFunction}) = true

function (f::QuadraticFunction)(x::AbstractVector)
  _checkreal(f, x)
  mul!(f.tmp, f.A, x)
  v = dot(x, f.tmp) / 2 + f.c
  _value_at(f.b === nothing ? v : v + dot(f.b, x), x)
end

function value_and_gradient!(g::AbstractVector, f::QuadraticFunction, x::AbstractVector)
  _checkwrite(value_and_gradient!, f, g, x)
  if eltype(g) !== eltype(x)
    _gradient_unchecked!(g, f, x)
    return _value_unchecked(f, x)
  end
  mul!(g, f.A, x)
  v = dot(g, x) / 2 + f.c
  if f.b !== nothing
    g .+= f.b
    v += dot(f.b, x)
  end
  _value_at(v, x)
end

function gradient!(g::AbstractVector, f::QuadraticFunction, x::AbstractVector)
  _checkwrite(gradient!, f, g, x)
  mul!(g, f.A, x)
  f.b === nothing || (g .+= f.b)
  g
end

function _eigen!(f::QuadraticFunction{T}) where {T}
  if isempty(f.Λ)
    E = eigen(Symmetric(Matrix{T}(f.A)))
    f.Λ, f.Q = E.values, E.vectors
  end
  f
end

function prox!(y::AbstractVector, f::QuadraticFunction{T}, x::AbstractVector, γ::Real) where {T}
  _checkwrite(prox!, f, y, x)
  _eigen!(f)
  γ = convert(T, γ)
  t = f.tmp
  f.b === nothing ? copyto!(t, x) : (t .= x .- γ .* f.b)
  mul!(f.w, transpose(f.Q), t)
  f.w ./= 1 .+ γ .* f.Λ
  mul!(t, f.Q, f.w)
  copyto!(y, t)
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!)}, f::QuadraticFunction{T}, y::AbstractVector, x::AbstractVector) where {T}
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  length(x) == length(f.tmp) || throw(DimensionMismatch())
  true
end

function _checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::QuadraticFunction, g::AbstractVector, x::AbstractVector)
  _checkreal(f, x)
  _checkreal(f, g)
  _checkgradient_output(g, x)
  _checksize(g, x)
  # Preserve the dimension error of the product, which checks before writing.
  length(x) == size(f.A, 2) || mul!(g, f.A, x)
end

restrict(f::QuadraticFunction, S::AbstractVector{<:Integer}) =
  QuadraticFunction(f.A[S, S], f.b === nothing ? nothing : f.b[S], f.c)

####################################
#  Kronecker quadratic
####################################

"""
    KronQuadratic(A, B, C = nothing)

The quadratic `f(Θ) = ⟨Θ, AΘB⟩ / 2 + ⟨C, Θ⟩` of a `p × q` matrix `Θ`, for
symmetric positive semidefinite `A` (`p × p`) and `B` (`q × q`), with `C = 0`
when omitted: `vec(Θ)'(B ⊗ A) vec(Θ) / 2 + vec(C)'vec(Θ)`, without forming the
`pq × pq` matrix `B ⊗ A`. Smooth, with gradient `AΘB + C`; `size(f)` is
`(p, q)`. `A` and `B` are stored as dense matrices; one that is symmetric only up
to rounding, such as an inverse computed by `inv`, is symmetrized.

`Θ` may be a `p × q` matrix, the vector `vec(Θ)`, a [`SparseIterate`](@ref) of
either shape or, when `p = q`, a [`SymmetricSparseIterate`](@ref). On a
`SymmetricSparseIterate` the gradient is the one over symmetric matrices,
`(AΘB + BΘA + C + C')/2`, so that a solver that steps on one triangle moves
correctly. `C` must be exactly symmetric for a value or second-order oracle at
a symmetric iterate, and for a gradient written into one. The value and the
gradient use buffers that `f` owns and allocate nothing; do not share `f`
between threads.

Examples:
- the direct difference (D-trace) loss `tr(ΣxΔΣyΔ)/2 - tr(Δ(Σy - Σx))` of a
  symmetric `Δ` (CovSel) is `KronQuadratic(Σx, Σy, Σx - Σy)`;
- the loss `tr(ΣxΘΣyΘ')/2 - Θ[a, b]` (DiffPrecTest), whose minimizer is the
  column `(b - 1)p + a` of `(Σy ⊗ Σx)⁻¹`, is `KronQuadratic(Σx, Σy, -E)`, with
  `E` the indicator of the entry `(a, b)`; DiffPrecTest's reduced loss on
  `svec(Θ)` is this function on a symmetric `Θ`, in coordinates that scale the
  off-diagonal entries by `√2`. For a symmetric iterate, use `-(E + E')/2`
  as the linear term.

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
"""
struct KronQuadratic{T<:AbstractFloat, MC<:Union{Nothing, Matrix{T}}} <: AbstractFunction
  A::Matrix{T}
  B::Matrix{T}
  C::MC
  D::Matrix{T}          # Θ as a dense p × q matrix
  AD::Matrix{T}         # A Θ
  W::Matrix{T}          # A Θ B, and the gradient

  function KronQuadratic{T, MC}(A, B, C, D, AD, W) where {T<:AbstractFloat, MC<:Union{Nothing, Matrix{T}}}
    all(a -> a === nothing || eltype(a) <: Real,
        (A, B, C, D, AD, W)) || _real_error(:KronQuadratic)
    new{T, MC}(A, B, C, D, AD, W)
  end
end
KronQuadratic(A::Matrix{T}, B::Matrix{T}, C::MC, D::Matrix{T}, AD::Matrix{T}, W::Matrix{T}) where {T<:AbstractFloat,
    MC<:Union{Nothing, Matrix{T}}} =
  KronQuadratic{T, MC}(A, B, C, D, AD, W)

function KronQuadratic(A::AbstractMatrix, B::AbstractMatrix, C::Union{Nothing, AbstractMatrix} = nothing)
  eltype(A) <: Real || _real_error(:KronQuadratic)
  eltype(B) <: Real || _real_error(:KronQuadratic)
  C === nothing || eltype(C) <: Real || _real_error(:KronQuadratic)
  p, q = checksquare(A), checksquare(B)
  C === nothing || size(C) == (p, q) ||
    throw(DimensionMismatch("A is $p × $p and B is $q × $q, so C must be $p × $q, not $(size(C, 1)) × $(size(C, 2))"))
  T = float(promote_type(eltype(A), eltype(B), C === nothing ? Bool : eltype(C)))
  # dense copies, so that the products never index a Symmetric or sparse matrix
  Ad, Bd = Matrix{T}(A), Matrix{T}(B)
  Cd = C === nothing ? nothing : Matrix{T}(C)
  all(isfinite, Ad) && all(isfinite, Bd) && (Cd === nothing || all(isfinite, Cd)) ||
    throw(ArgumentError("A, B and C must be finite"))
  KronQuadratic{T, typeof(Cd)}(_symmetrized!(Ad, "A"), _symmetrized!(Bd, "B"), Cd, zeros(T, p, q), zeros(T, p,
      q), zeros(T, p, q))
end

# M, which must be symmetric; one that is symmetric up to rounding, such as an
# inverse computed by inv, becomes (M + M')/2
function _symmetrized!(M::Matrix{T}, name) where {T}
  issymmetric(M) && return M
  δ = maximum(abs(M[i, j] - M[j, i]) for j in axes(M, 2) for i in j+1:size(M, 1))
  δ <= sqrt(eps(T)) * maximum(abs, M) ||
    throw(ArgumentError("$name must be symmetric (one that is symmetric up to rounding is accepted); wrap it in Symmetric to use one of its triangles"))
  for j in axes(M, 2), i in j+1:size(M, 1)
    M[i, j] = M[j, i] = (M[i, j] + M[j, i]) / 2
  end
  M
end

issmooth(::Type{<:KronQuadratic}) = true
isconvex(::Type{<:KronQuadratic}) = true

Base.size(f::KronQuadratic) = size(f.D)

# Θ is vec(Θ) or a p × q matrix
_checkshape(f::KronQuadratic, Θ::AbstractVector) = length(Θ) == length(f.D) ||
  throw(DimensionMismatch("Θ has length $(length(Θ)) and f takes $(size(f.D, 1)) × $(size(f.D, 2)) matrices"))
_checkshape(f::KronQuadratic, Θ::AbstractArray) = size(Θ) == size(f.D) ||
  throw(DimensionMismatch("Θ has size $(size(Θ)) and f takes $(size(f.D, 1)) × $(size(f.D, 2)) matrices"))

# Θ as a dense matrix in f.D
function _densify!(f::KronQuadratic, Θ::AbstractArray)
  _checkshape(f, Θ)
  copyto!(f.D, Θ)
end
function _densify!(f::KronQuadratic, Θ::SparseIterate)
  _checkshape(f, Θ)
  D = fill!(f.D, 0)
  @inbounds for k = 1:nnz(Θ)
    D[Θ.nzval2ind[k]] = Θ.nzval[k]
  end
  D
end
function _densify!(f::KronQuadratic, Θ::SymmetricSparseIterate)
  _checkshape(f, Θ)
  D = fill!(f.D, 0)
  @inbounds for k = 1:nnz(Θ)
    r, c = tril_cartesian(Θ.p, Θ.nzval2ind[k])
    D[r, c] = D[c, r] = Θ.nzval[k]
  end
  D
end

_checkvalue(f::KronQuadratic, Θ) = (_checkreal(f, Θ); _checkshape(f, Θ))

(f::KronQuadratic)(Θ::AbstractArray) =
  (_check_value_arrays(f, Θ); _value_checked(f, Θ))
function _value_unchecked(f::KronQuadratic, Θ::AbstractArray)
  _checkreal(f, Θ)
  D = _densify!(f, Θ)
  mul!(f.AD, f.A, D)
  mul!(f.W, f.AD, f.B)
  v = dot(D, f.W) / 2
  _value_at(f.C === nothing ? v : v + dot(f.C, D), Θ)
end

value_and_gradient!(G::AbstractArray, f::KronQuadratic, Θ::AbstractArray) =
  (_check_oracle_arrays(value_and_gradient!, f, G, Θ); _value_and_gradient_unchecked!(G, f, Θ))
function _value_and_gradient_unchecked!(G::AbstractArray, f::KronQuadratic, Θ::AbstractArray)
  _checkwrite(value_and_gradient!, f, G, Θ)
  D = _densify!(f, Θ)
  W = f.W
  mul!(f.AD, f.A, D)
  mul!(W, f.AD, f.B)
  v = dot(D, W) / 2
  if f.C !== nothing
    v += dot(f.C, D)
    W .+= f.C
  end
  Θ isa SymmetricSparseIterate && _symmetrize!(W)
  copyto!(G, W)
  _value_at(v, Θ)
end

gradient!(G::AbstractArray, f::KronQuadratic, Θ::AbstractArray) =
  (_check_oracle_arrays(gradient!, f, G, Θ); _gradient_unchecked!(G, f, Θ))
_gradient_unchecked!(G::AbstractArray, f::KronQuadratic, Θ::AbstractArray) =
  (_checkwrite(gradient!, f, G, Θ); _value_and_gradient_unchecked!(G, f, Θ); G)

_checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::KronQuadratic, G::AbstractArray, Θ::AbstractArray) =
  (_checkreal(f, Θ); _checkreal(f, G); _checkgradient_output(G, Θ); _checkshape(f, G); _checkshape(f, Θ))

# (W + W')/2 in place
function _symmetrize!(W::Matrix)
  p = checksquare(W)
  @inbounds for c = 1:p, r = c+1:p
    W[r, c] = W[c, r] = (W[r, c] + W[c, r]) / 2
  end
  W
end

####################################
#  Lipschitz constants of the gradients
####################################

lipschitz(::ZeroFunction) = 0.0
lipschitz(f::LinearFunction) = zero(float(eltype(f.c)))
lipschitz(f::SqrNormL2) = float(f.λ)
lipschitz(f::QuadraticFunction) = _opnorm_upper(f.A)
lipschitz(f::KronQuadratic) = _opnorm_upper(f.A) * _opnorm_upper(f.B)       # also over symmetric Θ
