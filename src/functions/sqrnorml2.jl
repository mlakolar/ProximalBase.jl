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
Values follow the overflow rule in Overflow and underflow of values.

# Precision

The level `λ`, the center `y` and the step are parameters; they are rounded once
to the *model precision* of each result and never widen it. That precision is the wider of the argument's and the
destination's for a gradient or a proximal point, and the argument's for the value
(so the value written by `value_and_prox!` is the value at the stored point, and
`value_and_gradient!` returns the standalone value and writes the standalone
gradient). The proximal weight `γλ` is formed from the unrounded factors before it
is rounded.

A parameter, a product or a coefficient that would leave the range of the model
precision does not change a representable result: the gradient of
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
_sumabs2(x::BlockIterate) = _sumabs2(x.storage)

_square_term_count(x::Union{SparseIterate, SymmetricSparseIterate}) = x.nnz
_square_term_count(x::BlockIterate) = _square_term_count(x.storage)
# inlined into the loop of the squared-residual kernel, whose @inbounds then reaches the index
Base.@propagate_inbounds function _square_multiplicity(x::SymmetricSparseIterate, i)
  r, c = tril_cartesian(x.p, x.nzval2ind[i])
  r == c ? 1 : 2
end

# The Frobenius norm of the matrix that an owner represents, an off-diagonal coordinate counted
# twice, in O(nnz): the ordinary sum of squares where it is a finite normal number, and otherwise
# the norm from the squared residuals in power-of-two units (_scaled_norm), as for norm_diff. Where
# every square is below floatmin (the largest stored magnitude tells, in one vectorized pass), the
# recomputation runs at once: the ordinary sum would form only subnormal squares, which are slow.
function _norm2_scaled(x::SymmetricSparseIterate)
  m = _stored_maxabs(x)
  isfinite(m) && m * m < floatmin(typeof(m)) && return _scaled_norm(x, nothing, zero(m))
  s = _sumabs2(x)
  isfinite(s) && s >= floatmin(typeof(s)) && return sqrt(s)
  _scaled_norm(x, nothing, s)
end

(f::SqrNormL2)(x::_ArrayOrOwner) =
  (_check_value_arrays(f, x); _value_checked(f, x))
function _value_unchecked(f::SqrNormL2, x::_ArrayOrOwner)
  # a center is read at every entry of the matrix that the owner represents
  x isa SymmetricSparseIterate && f.y !== nothing && return _value_unchecked(f, matrixview(x))
  _checkvalue(f, x)
  T = _valtype(x)
  iszero(f.λ) && return zero(T)
  y = f.y
  # A center is a parameter: it is rounded to the model precision of the value, the
  # argument's (T3), so a Float64 center does not widen a Float32 value.
  T === Float32 && y !== nothing && eltype(y) === Float64 && return _ridge_value_wide_center(f.λ, x, y)
  _square_value(f.λ, x, _ridge_value_center(T, y))
end

_checkvalue(f::SqrNormL2, x) = (_checkreal(f, x); _checkparams(f, x); f.y === nothing || _checksize(f.y, x); true)

# The scaled-value mechanism: the recomputation of _square_value in power-of-two units, with the
# centers rounded as the value rounds them.
# Exact-zero protection: the ordinary value is λ/2 Σ (xᵢ - yᵢ)², with the sum in the type T of the
# value. A difference is exact where it is subnormal and zero only for equal entries. Where a square
# can underflow (the sum below n floatmin/eps), or the level does not fit T, the value is recomputed
# in power-of-two units and rounded once (_square_value), so a zero value there is below δ. Otherwise
# the sum is normal, with relative error below n eps, and a zero value is the product with λ/2
# rounded to zero: below δ/2 (1 + n eps). The centers are rounded as the value rounds them, both
# here and in the parts. So |f(x)| < n δ in every case, whatever λ is (λ = 0 is exact).
_has_value_parts(::Type{<:SqrNormL2}) = true
_value_parts(f::SqrNormL2, x::AbstractArray, v) = _square_parts(f.λ, x, _ridge_value_center(_valtype(x), f.y))
@inline _zero_value_exponent(f::SqrNormL2, x::AbstractArray) =
  iszero(f.λ) ? _EXACT_ZERO : _underflow_exponent(_valtype(x), length(x))
_parameter_alias(f::SqrNormL2, out) = _mightalias_parameter(out, f.y)

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
@inline function _ridge_underflows(::Type{W}, c, cw::W) where {W<:Union{Float32, Float64}}
  c isa W && return false
  (abs(cw) < floatmin(W)) & !iszero(c)
end
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

# A center entry rounded to the model precision T where that keeps its range, as given otherwise (R3).
@inline _ridge_round(::Type{T}, c) where {T} = _ridge_fits(T, c) ? oftype(c, T(c)) : c
# The centers as the value at an argument of type T reads them (T3, R3), for the recomputation and
# the value parts: a Float64 center at a Float32 argument, and a BigFloat center at a Float32 or
# Float64 argument, are rounded entry by entry (_ridge_round); other centers are read as stored.
function _ridge_value_center(::Type{T}, y) where {T}
  y === nothing && return y
  ((T === Float32 && eltype(y) === Float64) || (T <: Union{Float32, Float64} && float(eltype(y)) <: BigFloat)) || return y
  map(c -> _ridge_round(T, c), y)
end

# Σ (xᵢ - round(cᵢ))² for a Float32 argument and a Float64 center. The ordinary
# computation runs in Float32 when every center entry keeps its range on rounding;
# otherwise the centers that fit are rounded and the scaled recovery of
# _square_value sees them in Float64 (R3).
function _ridge_value_wide_center(λ, x, y)
  T = Float32
  s = 0.0           # the squares are exact in Float64; the sum accumulates there
  u = false
  @inbounds @simd for i in eachindex(x, y)
    cw = T(y[i])
    s += abs2(Float64(T(x[i]) - cw))
    u |= _ridge_underflows(T, y[i], cw)
  end
  sT = T(s)
  v = convert(T, λ) * sT / 2
  if !u & isfinite(v) & _squares_normal(sT, length(x), T) && _level_fits(T, λ)
    return _value_at(v, x)
  end
  # a rounded center left the range, or the ordinary value is out of range
  _exceptional_square_value(λ, x, _ridge_value_center(T, y), sT, v)
end

value_and_gradient!(g::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
function _value_and_gradient_unchecked!(g::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner)
  _checkwrite(value_and_gradient!, f, g, x)
  v = _value_unchecked(f, x)      # before the gradient, which may be written over the center
  _ridge_gradient!(g, f.λ, f.y, x)
  v
end
gradient!(g::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
function _gradient_unchecked!(g::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner)
  _checkwrite(gradient!, f, g, x)
  _ridge_gradient!(g, f.λ, f.y, x)
end

# an owner point with a dense output: the entries of the matrix that the owner represents
_ridge_gradient!(g::AbstractArray, λ, c, x::SymmetricSparseIterate) = _ridge_gradient!(g, λ, c, matrixview(x))
# an owner output: the symmetric part of the entrywise gradient, once per coordinate
function _ridge_gradient!(g::SymmetricSparseIterate, λ, c, x)
  W = _model_precision(eltype(g), eltype(x))
  if _level_fits(W, λ)
    _ridge_gradient_coordinates!(g, convert(W, λ), c, x)
  else
    _ridge_gradient_wide_coordinates!(g, W, convert(promote_type(W, float(typeof(λ))), λ), c, x)
  end
  g
end
_ridge_gradient_coordinates!(g, l, ::Nothing, x) = _symmetric_coordinates!(g, (v, j) -> l * v, x)
_ridge_gradient_coordinates!(g, l, c, x) =
  _symmetric_coordinates!(g, (v, j) -> _ridge_gradient_entry(l, v, @inbounds c[j]), x)
_ridge_gradient_wide_coordinates!(g, ::Type{W}, l, c, x) where {W} =
  _symmetric_coordinates!(g, (v, j) -> _ridge_gradient_wide(W, l, v, c === nothing ? zero(W) : @inbounds c[j]), x)
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
    if g isa DenseArray && x isa DenseArray && cp isa DenseArray && size(g) == size(x) == size(cp) &&
       !Base.mightalias(g, cp)       # the repair pass rereads c
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
# a nonfinite entry of finite operands: λ(x - c) from the residual in its own units (_halved_difference)
@noinline function _ridge_gradient_overflow(l, x, c, r)
  (isfinite(x) && isfinite(c)) || return r
  d, h = _halved_difference(x, c, typeof(l))
  _scaled_product(l, d, h)
end
# l is a level wider than W, or c is a center wider than W: the entry is computed in R
function _ridge_gradient_wide(::Type{W}, l, x, c) where {W}
  R = promote_type(W, typeof(l), float(typeof(c)))
  cw = _ridge_fits(W, c) ? R(W(c)) : R(c)
  r = R(l) * (R(x) - cw)
  isfinite(r) ? r : _ridge_gradient_overflow(R(l), R(x), cw, r)
end

prox!(z::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(z, x); _check_oracle_arrays(prox!, f, z, x, γ); _prox_unchecked!(z, f, x, γ))
_prox_unchecked!(z::AbstractArray, f::SqrNormL2, x::SymmetricSparseIterate, γ::Real) =
  _prox_unchecked!(z, f, matrixview(x), γ)
function _prox_unchecked!(z::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner, γ::Real)
  _checkwrite(prox!, f, z, x)
  W = _model_precision(eltype(z), eltype(x))
  c = f.y === nothing ? nothing : _checked_parameter(z, f.y)
  if _ridge_needs_scaling(W, γ, f.λ)
    _ridge_prox!(z, _ridge_scaled(W, γ, f.λ), x, c)
  else
    _ridge_prox!(z, _ridge_ordinary(W, γ, f.λ), x, c)
  end
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}, f::SqrNormL2, y::_ArrayOrOwner, x::_ArrayOrOwner)
  _checkreal(f, x)
  _checkreal(f, y)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(f, x)
  f.y === nothing || (_checksize(f.y, x); _check_parameter_axes(y, f.y))
  true
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
  W <: Union{Float32, Float64} || return false
  P = promote_type(W, float(typeof(γ)), float(typeof(λ)))
  t = P(γ) * P(λ)
  # t in the band implies finite nonzero factors, so they are examined only outside it
  (P(_ridge_band_low(W)) <= t <= P(_ridge_band_high(W))) && return false
  isfinite(γ) & isfinite(λ) & !iszero(γ) & !iszero(λ)
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
  # A level that is zero as given and the step Inf as given (not as converted, which turns a finite step
  # beyond the range into Inf) make the product γλ Inf⋅0. The weights are then those of the step 1,
  # which are also those of every positive finite step: (1, ±0) with the sign of the level. Any other
  # step, NaN and negative ones included, keeps the arithmetic of the product. The selects follow the
  # weights, so that they wait behind the division, and they are no branch to a call: the loops over the
  # entries of a ridge keep their vectorization.
  endpoint = iszero(λ) & (γ == Inf)
  t1 = convert(W, one(P) * P(λ))
  _RidgeOrdinary{W}(ifelse(endpoint, one(W), a), ifelse(endpoint, t1, b), ifelse(endpoint, t1, tw))
end
@inline _ridge_scaled(::Type{W}, γ, λ) where {W} =
  (P = promote_type(W, float(typeof(γ)), float(typeof(λ))); _ridge_scaled(W, P(γ), P(λ)))

Base.@assume_effects :effect_free :nothrow :terminates_globally @noinline function _ridge_scaled(::Type{W}, γ::P, λ::P) where {W, P}
  mγ, eγ = _frexp(γ)
  mλ, eλ = _frexp(λ)
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
# A center outside W's range on rounding: the entry is computed in the wider type R.
# For a hardware-float center, R holds x and c with room to spare (it is wider than W,
# which holds x), so the plain convex combination cannot overflow. prox_block inlines
# it, so that a loop over the entries has no call and vectorizes (its weights do not
# depend on the entry); the array methods call it, as before.
@inline function _ridge_wide_entry(o::_RidgeOrdinary{W}, x, c::Base.IEEEFloat) where {W}
  R = promote_type(W, typeof(c))
  a, b = _ridge_weights_t(convert(R, o.t))
  W(a * R(x) + b * c)
end
@noinline _ridge_apply_wide(o::_RidgeOrdinary, x, c::Base.IEEEFloat) = _ridge_wide_entry(o, x, c)
@noinline function _ridge_apply_wide(o::_RidgeOrdinary{W}, x, c) where {W}
  R = promote_type(W, float(typeof(c)))
  a, b = _ridge_weights_t(convert(R, o.t))
  W(_ridge_blend(R(x), R(c), a, b))
end
@noinline function _ridge_apply_wide(s::_RidgeScaled{W}, x, c) where {W}
  R = promote_type(W, float(typeof(c)))
  W(_ridge_scaled_blend(R(x), R(c), _ridge_scaled(R, promote_type(R, typeof(s.γ))(s.γ), promote_type(R, typeof(s.λ))(s.λ))))
end

# an owner output: each coordinate once, from the lower entry of the symmetric argument
_ridge_prox!(z::SymmetricSparseIterate, o::_RidgeOrdinary, x, ::Nothing) =
  _lower_coordinates!(z, (v, j) -> _ridge_apply(o, v), x)
_ridge_prox!(z::SymmetricSparseIterate, s::_RidgeScaled, x, ::Nothing) =
  _lower_coordinates!(z, (v, j) -> _ridge_apply(s, v), x)
function _ridge_prox!(z::SymmetricSparseIterate, co::Union{_RidgeOrdinary{W}, _RidgeScaled{W}}, x, c) where {W}
  if _ridge_center_wider(W, eltype(c)) && !_ridge_all_fit(W, c)
    _lower_coordinates!(z, (v, j) -> _ridge_apply(co, v, @inbounds c[j]), x)
  else
    _lower_coordinates!(z, (v, j) -> _ridge_apply_fit(co, v, @inbounds c[j]), x)
  end
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

@inline function prox_block(f::SqrNormL2{<:Real, Nothing}, v::Real, ::Integer, γ::Real)
  W = typeof(float(v))
  _ridge_needs_scaling(W, γ, f.λ) ? _ridge_apply(_ridge_scaled(W, γ, f.λ), W(v)) :
                                    _ridge_apply(_ridge_ordinary(W, γ, f.λ), W(v))
end
Base.@propagate_inbounds function prox_block(f::SqrNormL2, v::Real, i::Integer, γ::Real)
  W = typeof(float(v))
  _ridge_needs_scaling(W, γ, f.λ) ? _ridge_apply(_ridge_scaled(W, γ, f.λ), W(v), f.y[i]) :
                                    _ridge_apply_entry(_ridge_ordinary(W, γ, f.λ), W(v), f.y[i])
end
@inline _ridge_apply_entry(o::_RidgeOrdinary, x, c) = _ridge_apply(o, x, c)
@inline _ridge_apply_entry(o::_RidgeOrdinary, x, c::Base.IEEEFloat) =
  _ridge_fits(typeof(o.a), c) ? _ridge_blend(x, c, o.a, o.b) : _ridge_wide_entry(o, x, c)

_checkparams(f::SqrNormL2, x) = _checkparam(f.y, x)

kkt_violation(::SqrNormL2{<:Real, Nothing}, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]))
kkt_violation(f::SqrNormL2, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i] - f.λ * f.y[i]))

restrict(f::SqrNormL2{<:Real, Nothing}, ::AbstractVector{<:Integer}) = f
restrict(f::SqrNormL2, S::AbstractVector{<:Integer}) = SqrNormL2(f.λ, f.y[S])


lipschitz(f::SqrNormL2) = float(f.λ)
