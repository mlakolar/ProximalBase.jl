####################################
#  squared Euclidean norm
####################################

"""
    SqrNormL2(λ = 1)
    SqrNormL2(λ, y)

Squared-distance function `f(x) = (λ/2)‖x-y‖²` with nonnegative level `λ`
and center zero when `y` is omitted. For a matrix, the norm is Frobenius.
This is the ridge penalty when uncentered, or half the squared error when `λ = 1`.

The function is smooth, convex and separable, with gradient `λ*(x-y)` and
proximal point `(x + γ*λ*y)/(1 + γ*λ)`. A supplied center is retained by reference
and must have the same shape as `x`. Use real, one-based arrays; `λ` is
checked nonnegative and not `NaN` at construction. The retained center's
array type and axes are checked at construction; later edits are not revalidated.

# Examples
```jldoctest
julia> f = SqrNormL2(2.0, [1.0, 0.0]);

julia> f([3.0, -1.0])
5.0

julia> prox(f, [3.0, -1.0], 0.5)
2-element Vector{Float64}:
  2.0
 -0.5
```

# Edge cases
- For a symmetric iterate, the center must be exactly symmetric; validate with
  [`ProximalBase.check_symmetric_parameters`](@ref).
- Exact in-place prox is supported; gradient output must be independent of its input.
- An infinite level follows IEEE arithmetic, not a general projection rule:
  a zero residual can yield a `NaN` gradient from `Inf*0`.
  A zero level does not suppress every `0*Inf` or `0*NaN` in gradients or prox.

See [squared-norm details](@ref Squared-Euclidean-norm) and the
[numerical contract](@ref numerics) for supported precision, range recovery,
restriction constants and non-finite values.
"""
struct SqrNormL2{T<:Real, Y<:Union{Nothing, AbstractArray}} <: AbstractFunction
  λ::T
  y::Y

  function SqrNormL2{T, Y}(λ::T, y::Y) where {T<:Real, Y<:Union{Nothing, AbstractArray}}
    _check_one_based(y)
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
_sumabs2(x::SparseIterate) = _indexed_sumabs2(x.nzval, Base.OneTo(x.nnz), float(eltype(x)))
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

(f::SqrNormL2)(x::_ArrayOrSymmetricIterate) =
  (_check_value_arrays(f, x); _value_checked(f, x))
function _value_unchecked(f::SqrNormL2, x::_ArrayOrSymmetricIterate)
  # a center is read at every entry of the matrix that the symmetric iterate represents
  x isa SymmetricSparseIterate && f.y !== nothing && return _value_unchecked(f, matrixview(x))
  _checkvalue(f, x)
  T = _valtype(x)
  iszero(f.λ) && return zero(T)
  y = f.y
  T === Float32 && y !== nothing && eltype(y) === Float64 && return _ridge_value32(f.λ, x, y)
  _square_value(f.λ, x, _ridge_value_center(T, y))
end

_checkvalue(f::SqrNormL2, x) = (_checkreal(f, x); _checkparams(f, x); f.y === nothing || _checksize(f.y, x); true)

# Scaled uses the same center conversion as the value when recomputing in power-of-two units.
# For homogeneous inputs, _underflow_exponent bounds the exact value when the ordinary or
# recomputed value rounded to zero. An exactly zero level is handled before any arithmetic.
_has_value_parts(::Type{<:SqrNormL2}) = true
_value_parts(f::SqrNormL2, x::AbstractArray, v) = _square_parts(f.λ, x, _ridge_value_center(_valtype(x), f.y))
@inline _zero_value_exponent(f::SqrNormL2, x::AbstractArray) =
  iszero(f.λ) ? _EXACT_ZERO : _underflow_exponent(_valtype(x), length(x))
_parameter_alias(f::SqrNormL2, out) = _mightalias_parameter(out, f.y)

# Round the same centers for the value and its parts.
function _ridge_value_center(::Type{T}, y) where {T}
  y === nothing && return y
  T <: Union{Float32, Float64} || return y
  R = float(eltype(y))
  ((T === Float32 && eltype(y) === Float64) || R <: BigFloat) || return y
  map(c -> R(T(c)), y)
end

# Float32 residuals with a Float64 center. Accumulate their squares in Float64 as
# before. The exceptional path uses the same rounded center as the value parts.
function _ridge_value32(λ, x, y)
  s = 0.0
  @inbounds @simd for i in eachindex(x, y)
    s += abs2(Float64(Float32(x[i]) - Float32(y[i])))
  end
  s32 = Float32(s)
  v = convert(Float32, λ) * s32 / 2
  (isfinite(v) & _squares_normal(s32, length(x), Float32)) && _level_fits(Float32, λ) && return v
  _exceptional_square_value(λ, x, _ridge_value_center(Float32, y), s32, v)
end

value_and_gradient!(g::_ArrayOrSymmetricIterate, f::SqrNormL2, x::_ArrayOrSymmetricIterate) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
function _value_and_gradient_unchecked!(g::_ArrayOrSymmetricIterate, f::SqrNormL2, x::_ArrayOrSymmetricIterate)
  _checkwrite(value_and_gradient!, f, g, x)
  v = _value_unchecked(f, x)      # before the gradient, which may be written over the center
  _ridge_gradient!(g, f.λ, f.y, x)
  v
end
gradient!(g::_ArrayOrSymmetricIterate, f::SqrNormL2, x::_ArrayOrSymmetricIterate) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
function _gradient_unchecked!(g::_ArrayOrSymmetricIterate, f::SqrNormL2, x::_ArrayOrSymmetricIterate)
  _checkwrite(gradient!, f, g, x)
  _ridge_gradient!(g, f.λ, f.y, x)
end

# a symmetric iterate point with a dense output: the entries of the matrix that the symmetric iterate represents
_ridge_gradient!(g::AbstractArray, λ, c, x::SymmetricSparseIterate) = _ridge_gradient!(g, λ, c, matrixview(x))
# a symmetric iterate output: the symmetric part of the entrywise gradient, once per coordinate
function _ridge_gradient!(g::SymmetricSparseIterate, λ, c, x)
  W = _model_precision(eltype(g), eltype(x))
  _ridge_gradient_coordinates!(g, convert(W, λ), c, x)
  g
end
_ridge_gradient_coordinates!(g, l, ::Nothing, x) = _symmetric_coordinates!(g, (v, j) -> l * v, x)
_ridge_gradient_coordinates!(g, l, c, x) =
  _symmetric_coordinates!(g, (v, j) -> _ridge_gradient_entry(l, v, @inbounds c[j]), x)

# gᵢ = λ(xᵢ - cᵢ) in the precision W of the gradient (_model_precision), the wider of the
# destination and the argument. An overflowing xᵢ - cᵢ can still give a finite gradient entry.
function _ridge_gradient!(g, λ, c, x)
  W = _model_precision(eltype(g), eltype(x))
  l = convert(W, λ)
  c === nothing && return g .= l .* x
  cp = _assumed_symmetric(g, c)
  if g isa DenseArray && x isa DenseArray && cp isa DenseArray && size(g) == size(x) == size(cp) &&
     !Base.mightalias(g, cp)       # the repair pass rereads c
    _ridge_gradient_dense!(g, l, x, cp)
  else
    g .= _ridge_gradient_entry.(l, x, cp)
  end
  g
end
# The vectorized loop. When the sum is nonfinite, recompute by entry to recover gradients
# whose residuals overflowed. The gradient never aliases x.
function _ridge_gradient_dense!(g, l::W, x, c) where {W}
  t = zero(W)       # a non-finite entry makes the sum non-finite
  @inbounds @simd for i in eachindex(g)
    cw = W(c[i])
    r = l * (W(x[i]) - cw)
    g[i] = r
    t += r
  end
  isfinite(t) || _ridge_gradient_repair!(g, l, x, c)
  g
end
@noinline function _ridge_gradient_repair!(g, l::W, x, c) where {W}
  @inbounds for i in eachindex(g)
    g[i] = _ridge_gradient_entry(l, x[i], c[i])
  end
end
@inline function _ridge_gradient_entry(l::W, x, c) where {W}
  r = l * (W(x) - W(c))
  isfinite(r) ? r : _ridge_gradient_overflow(l, W(x), W(c), r)
end
# a nonfinite entry of finite operands: λ(x - c) from the residual in its own units (_halved_difference)
@noinline function _ridge_gradient_overflow(l, x, c, r)
  (isfinite(x) && isfinite(c)) || return r
  d, h = _halved_difference(x, c, typeof(l))
  _scaled_product(l, d, h)
end

prox!(z::_ArrayOrSymmetricIterate, f::SqrNormL2, x::_ArrayOrSymmetricIterate, γ::Real) =
  (x = _symmetric_argument(z, x); _check_oracle_arrays(prox!, f, z, x, γ); _prox_unchecked!(z, f, x, γ))
_prox_unchecked!(z::AbstractArray, f::SqrNormL2, x::SymmetricSparseIterate, γ::Real) =
  _prox_unchecked!(z, f, matrixview(x), γ)
function _prox_unchecked!(z::_ArrayOrSymmetricIterate, f::SqrNormL2, x::_ArrayOrSymmetricIterate, γ::Real)
  _checkwrite(prox!, f, z, x)
  W = _model_precision(eltype(z), eltype(x))
  c = f.y === nothing ? nothing : _assumed_symmetric(z, f.y)
  if _ridge_needs_scaling(W, γ, f.λ)
    _ridge_prox!(z, _ridge_scaled(W, γ, f.λ), x, c)
  else
    _ridge_prox!(z, _ridge_ordinary(W, γ, f.λ), x, c)
  end
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}, f::SqrNormL2, y::_ArrayOrSymmetricIterate, x::_ArrayOrSymmetricIterate)
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
# P = promote_type(W, factors) and rounded to W.
#
# Ordinary: t, 1/t and both weights are normal numbers in W (or a factor is zero or
# not finite, which keeps its earlier meaning). Scaled: t = τ·2^k from the
# significands; the smaller weight is carried as m·2^e and applied last, so a
# coefficient that is not representable still acts on the coordinate.
struct _RidgeOrdinary{W<:Real}
  a::W
  b::W
end
struct _RidgeScaled{W<:Real}
  L::W              # the larger weight, in [1/2, 1]
  m::W              # significand of the smaller weight, in (0, 1]
  e::Int
  small_x::Bool     # the smaller weight multiplies x (t > 1); else the center
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
  _RidgeOrdinary{W}(ifelse(endpoint, one(W), a), ifelse(endpoint, t1, b))
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
    _RidgeScaled{W}(inv(d), r / d / 4, 2 - k, true)
  else
    d = one(W) + τ * ldexp(one(W), k)
    _RidgeScaled{W}(inv(d), τ / d, k, false)
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
@inline _ridge_apply(o::_RidgeOrdinary, x, c) = _ridge_blend(x, c, o.a, o.b)
@inline _ridge_apply(s::_RidgeScaled{W}, x, c) where {W} = _ridge_scaled_blend(W(x), W(c), s)

# a symmetric iterate output: each coordinate once, from the lower entry of the symmetric argument
_ridge_prox!(z::SymmetricSparseIterate, o::_RidgeOrdinary, x, ::Nothing) =
  _lower_coordinates!(z, (v, j) -> _ridge_apply(o, v), x)
_ridge_prox!(z::SymmetricSparseIterate, s::_RidgeScaled, x, ::Nothing) =
  _lower_coordinates!(z, (v, j) -> _ridge_apply(s, v), x)
_ridge_prox!(z::SymmetricSparseIterate, co::Union{_RidgeOrdinary, _RidgeScaled}, x, c) =
  _lower_coordinates!(z, (v, j) -> _ridge_apply(co, v, @inbounds c[j]), x)
_ridge_prox!(z, o::_RidgeOrdinary, x, ::Nothing) = (z .= o.a .* x)
_ridge_prox!(z, s::_RidgeScaled, x, ::Nothing) = (z .= (xi -> _ridge_apply(s, xi)).(x))
_ridge_prox!(z, co::Union{_RidgeOrdinary, _RidgeScaled}, x, c) =
  (z .= ((xi, ci) -> _ridge_apply(co, xi, ci)).(x, c))

@inline function prox_block(f::SqrNormL2{<:Real, Nothing}, v::Real, ::Integer, γ::Real)
  W = typeof(float(v))
  _ridge_needs_scaling(W, γ, f.λ) ? _ridge_apply(_ridge_scaled(W, γ, f.λ), W(v)) :
                                    _ridge_apply(_ridge_ordinary(W, γ, f.λ), W(v))
end
Base.@propagate_inbounds function prox_block(f::SqrNormL2, v::Real, i::Integer, γ::Real)
  W = typeof(float(v))
  _ridge_needs_scaling(W, γ, f.λ) ? _ridge_apply(_ridge_scaled(W, γ, f.λ), W(v), f.y[i]) :
                                    _ridge_apply(_ridge_ordinary(W, γ, f.λ), W(v), f.y[i])
end

_checkparams(f::SqrNormL2, x) = _checkparam(f.y, x)

kkt_violation(::SqrNormL2{<:Real, Nothing}, g::AbstractArray, i::Integer) =
  (_check_one_based(g); eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]))
kkt_violation(f::SqrNormL2, g::AbstractArray, i::Integer) =
  (_check_one_based(g); eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i] - f.λ * f.y[i]))


# The point-aware score (see kkt_violation): |vᵢ + dᵢ| with dᵢ the entry of the gradient at x as the ridge gradient
# forms it, with the level and center converted to W and with recovery of an overflowing residual.
# The type W includes the element type of a center.
@inline _kkt_type(f::SqrNormL2, ::Type{V}, ::Type{X}) where {V, X} = _kkt_center_type(f.y, V, X)
@inline _kkt_center_type(::Nothing, ::Type{V}, ::Type{X}) where {V, X} = float(promote_type(V, X))
@inline _kkt_center_type(y, ::Type{V}, ::Type{X}) where {V, X} = float(promote_type(V, X, eltype(y)))
@inline _kkt_state(f::SqrNormL2, ::Type{W}) where {W} = convert(W, f.λ)
# the loop of dense arrays computes the gradient entry without the recovery, which a non-finite score repeats by entry
@inline _kkt_plain(::Nothing, l, x, i) = l * x
Base.@propagate_inbounds _kkt_plain(y::AbstractArray, l::W, x::W, i) where {W} = l * (x - convert(W, y[i]))
Base.@propagate_inbounds _kkt_natural(f::SqrNormL2, s, ::Type{W}, v::W, x::W, i) where {W} =
  abs(v + _kkt_plain(f.y, s, x, i))
@inline _kkt_flag(::SqrNormL2, x, r) = r - r
@inline _kkt_ridge(::Nothing, l, ::Type{W}, x, i) where {W} = l * x
Base.@propagate_inbounds _kkt_ridge(y, l, ::Type{W}, x, i) where {W} =
  convert(W, _ridge_gradient_entry(l, x, y[i]))
Base.@propagate_inbounds function _kkt_entry(f::SqrNormL2, s, ::Type{W}, v, x, i) where {W}
  vw = convert(W, v)
  xw = convert(W, x)
  _kkt_override(abs(vw + _kkt_ridge(f.y, s, W, xw, i)), vw, xw)
end
@_kkt_entrywise SqrNormL2

restrict(f::SqrNormL2{<:Real, Nothing}, ::AbstractVector{<:Integer}) = f
restrict(f::SqrNormL2, S::AbstractVector{<:Integer}) = SqrNormL2(f.λ, f.y[S])


lipschitz(f::SqrNormL2) = float(f.λ)
