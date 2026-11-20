####################################
#
#  Functions built from other functions
#
####################################

# Retain exceptional wrapper products until the leaf coefficient is formed.
struct _ProjectionScale{T}
  m::T
  e::Int
end
Base.broadcastable(a::_ProjectionScale) = Ref(a)
Base.isfinite(a::_ProjectionScale) = isfinite(a.m)
Base.isinf(a::_ProjectionScale) = isinf(a.m)
Base.iszero(a::_ProjectionScale) = iszero(a.m)
Base.isone(a::_ProjectionScale) = a.m == 0.5 && a.e == 1
_projection_scale_parts(a::Real) = _frexp(float(a))
_projection_scale_parts(a::_ProjectionScale) = (a.m, a.e)
function _projection_accumulate(a::Real, b::Real)
  isone(a) && return b
  (iszero(a) || iszero(b)) && return zero(float(a) + float(b))
  p = float(a) * float(b)
  (!isfinite(a) || !isfinite(b) || iszero(a) || iszero(b)) && return p
  floatmin(typeof(float(p))) <= abs(p) <= floatmax(typeof(float(p))) && return p
  ma, ea = _frexp(float(a))
  mb, eb = _frexp(float(b))
  m, e = _frexp(ma * mb)
  _ProjectionScale(m, Int(ea + eb + e))
end
function _projection_accumulate(a::_ProjectionScale, b::Real)
  (!isfinite(b) || iszero(b)) && return b
  mb, eb = _frexp(float(b))
  m, e = _frexp(a.m * mb)
  _ProjectionScale(m, Int(a.e + eb + e))
end

# lazy views of a gradient for the optimality tests: c g and g + c
struct _ScaledArray{T, N, A<:AbstractArray} <: AbstractArray{T, N}
  g::A
  c::T
end
_ScaledArray(g::AbstractArray{T, N}, c) where {T, N} = (S = promote_type(T, typeof(c)); _ScaledArray{S, N, typeof(g)}(g, S(c)))
_ScaledArray{T}(g::AbstractArray{<:Any, N}, c) where {T, N} = _ScaledArray{T, N, typeof(g)}(g, convert(T, c))
Base.size(a::_ScaledArray) = size(a.g)
Base.IndexStyle(::Type{<:_ScaledArray}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::_ScaledArray{T}, i::Int) where {T} = a.c * convert(T, a.g[i])
_smooth_kkt_overlap(buf, a::_ScaledArray) = _smooth_kkt_overlap(buf, a.g)

# the entries of g times a factor that is not a normal number, with the exponent applied last
struct _ExpScaledArray{T, N, A<:AbstractArray, S<:_ProjectionScale} <: AbstractArray{T, N}
  g::A
  a::S
end
_ExpScaledArray(g::AbstractArray{T, N}, a::_ProjectionScale{S}) where {T, N, S} =
  _ExpScaledArray{promote_type(float(T), S), N, typeof(g), typeof(a)}(g, a)
_ExpScaledArray(g::AbstractArray, a::Real) = _ScaledArray(g, a)
Base.size(a::_ExpScaledArray) = size(a.g)
Base.IndexStyle(::Type{<:_ExpScaledArray}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::_ExpScaledArray{T}, i::Int) where {T} = convert(T, _apply_scale(a.a, a.g[i]))

####################################
#  α f
####################################

"""
    Scaled(f, α)

Nonnegative multiple `h(x) = α f(x)`. The expressions `α*f` and `f*α` construct
it too. Negative or `NaN` scales are invalid. For positive finite `α`, a proximal
step for `h` uses the inner function's step `α*γ`.

The wrapper retains `f` by reference, including its parameters, scratch and caches.
Traits and supported oracles forward to `f`; scaling does not add missing methods.
Call [`reset_cache!`](@ref) on the outermost function after changing cached data.

# Examples
```jldoctest
julia> f = Scaled(SqrNormL2(2.0), 0.5);

julia> f([2.0, -1.0])
2.5

julia> prox(f, [2.0, -1.0], 1.0)
2-element Vector{Float64}:
  1.0
 -0.5
```

# Edge cases
- A zero scale may return a simpler function, or throw if no domain-preserving
  representation is supported; the result need not be a `Scaled`.
- Nested positive finite factors retain their exponents when intermediate
  products exceed floating-point range; whether a result is recovered depends
  on the inner function. Infinite scales have function-specific endpoint rules.
- Exact input/output identity follows the supported inner prox; other overlaps
  remain subject to the [ownership rules](@ref numerical-usage).

See [scaling details](@ref scaling-guide) for zero scales, numerical limits and
which operations are available.
"""
struct Scaled{F<:AbstractFunction, T<:Real} <: AbstractFunction
  f::F
  α::T

  function Scaled{F, T}(f::F, α::T) where {F<:AbstractFunction, T<:Real}
    _checknonneg("α", α)
    iszero(α) && return _zero_scaled(f)
    new{F, T}(f, α)
  end
end
Scaled(f::AbstractFunction, α::Real) = Scaled{typeof(f), typeof(α)}(f, α)
# Two scales merge when their product is an ordinary number. A product that is not a normal
# number (it underflows, is subnormal or overflows) no longer says what the two factors said,
# so the wrappers stay nested and every oracle forms the coefficient it needs from all of the
# factors (`_scaled_chain`). Zero and infinite scales keep their meaning and merge as before.
@inline function _nested_scales(a, b)
  (isfinite(a) & isfinite(b) & (a > 0) & (b > 0)) || return false
  p = a * b
  p isa AbstractFloat && !(floatmin(typeof(p)) <= p <= floatmax(typeof(p)))
end
function Scaled(f::Scaled, α::Real)
  _checknonneg("α", α)
  _nested_scales(f.α, α) ? Scaled{typeof(f), typeof(α)}(f, α) : Scaled(f.f, f.α * α)
end

@noinline _zero_scale_error(f) = throw(ArgumentError("zero scaling of $(typeof(f)) has no supported domain-preserving representation"))
_zero_scaled(f::AbstractFunction) = _isindicator(f) ? f : _zero_scale_error(f)
_zero_scaled(f::ZeroFunction) = f
_finite_parameter(::Nothing) = true
_finite_parameter(x::Real) = isfinite(x)
_finite_parameter(x::AbstractArray) = all(isfinite, x)
_finite_zero(f, parameters...) = all(_finite_parameter, parameters) ? ZeroFunction() : _zero_scale_error(f)
function _zero_scaled(f::NormL1)
  isfinite(f.λ0) || _zero_scale_error(f)
  _finite_parameter(f.w) ? ZeroFunction() : NormL1(zero(f.λ0), f.w)
end
# a finite level gives the level zero, which keeps the domain xᵢ >= 0 and the fixed zeros of an infinite weight
function _zero_scaled(f::NormL1Nonneg)
  isfinite(f.λ0) || _zero_scale_error(f)
  NormL1Nonneg(zero(f.λ0), f.w)
end
# finite levels give the zero function, or, with an infinite weight, zero levels that keep its fixed zeros
function _zero_scaled(f::ElasticNet)
  isfinite(f.λ1) && isfinite(f.λ2) || _zero_scale_error(f)
  _finite_parameter(f.w) ? ZeroFunction() : ElasticNet(zero(f.λ1), zero(f.λ2), f.w)
end
_zero_scaled(f::SqrNormL2) = _finite_zero(f, f.λ, f.y)
_zero_scaled(f::QuadraticFunction) = _finite_zero(f, f.A, f.b, f.c)
_zero_scaled(f::LeastSquaresLoss) = _finite_zero(f, f.X, f.Y)
_zero_scaled(f::SqrtLeastSquaresLoss) = _finite_zero(f, f.X, f.Y)
_zero_scaled(f::LinearFunction) = _finite_zero(f, f.c, f.d)
_zero_scaled(f::Scaled) = (_finite_parameter(f.α) || _zero_scale_error(f); _zero_scaled(f.f))

Base.:*(α::Real, f::AbstractFunction) = Scaled(f, α)
Base.:*(f::AbstractFunction, α::Real) = Scaled(f, α)

for trait in (:issmooth, :isproximable, :isseparable, :isconvex)
  @eval $trait(::Type{<:Scaled{F}}) where {F} = $trait(F)
end

# The function under a chain of nested scales, and the product of the chain: an ordinary
# number, or a `_ProjectionScale` that keeps the factors' combined exponent when the product
# is not a normal number. Only nested wrappers, which exist for such products, take this path.
# The product is always a `_ProjectionScale` here, so that its type does not depend on the values.
@inline _scale_start(a::Real) = ((m, e) = frexp(float(a)); _ProjectionScale(m, Int(e)))
@inline _scaled_chain(f, a) = (f, a)
@inline _scaled_chain(f::Scaled, a) = _scaled_chain(f.f, _projection_accumulate(a, f.α))
@inline _scaled_chain(s::Scaled) = _scaled_chain(s.f, _scale_start(s.α))
# The direct Scaled{F,T} constructor can nest wrappers with ordinary products.
# Such wrappers keep their recursive oracles.
@inline _chained(::Scaled) = false
@inline _chained(s::Scaled{<:Scaled}) = _chain_exceptional(s.f, s.α)
@inline _chain_exceptional(f, a) = false
@inline _chain_exceptional(f::Scaled, a) = _nested_scales(a, f.α) || _chain_exceptional(f.f, a * f.α)

# a⋅v with the factor's exponent applied last, so that a product that is representable is
# not lost to an intermediate; an ordinary factor is the plain product
@inline _apply_scale(a::Real, v) = a * v
@inline function _apply_scale(a::_ProjectionScale, v)
  w = float(v)
  (iszero(w) || !isfinite(w)) && return a.m * w
  # the significand of v, so that neither a subnormal v nor the product of significands loses bits
  R = promote_type(typeof(a.m), typeof(w))
  mv, ev = _frexp(convert(R, w))
  ldexp(a.m * mv, a.e + ev)
end
# a < 1 and a > 1 of the product (the significand is in [1/2, 1))
@inline _scale_below_one(a::Real) = a < 1
@inline _scale_below_one(a::_ProjectionScale) = a.e <= 0
@inline _scale_above_one(a::Real) = a > 1
@inline _scale_above_one(a::_ProjectionScale) = a.e > 1 || (a.e == 1 && a.m > 0.5)
# the reciprocal of a factor, as a factor
_scale_inverse(a::Real) = inv(a)
function _scale_inverse(a::_ProjectionScale)
  m, e = _frexp(inv(a.m))
  _ProjectionScale(m, Int(e - a.e))
end

# Fold a scale into a normal level, retaining exact zero levels.
function _folded_level(λ, α)
  iszero(λ) && return λ
  c = _apply_scale(α, λ)
  isfinite(c) && !iszero(c) && _level_fits(typeof(float(c)), c) ? c : nothing
end
# A scale that only folds into levels that are normal numbers. A subnormal level has fewer bits and
# is no longer the coefficient that an array step multiplies, entry by entry.
struct _NormalScale{T}
  s::_ProjectionScale{T}
end
_apply_scale(a::_NormalScale, v) = _apply_scale(a.s, v)
# An explicit zero or infinite factor is not a scale to fold: the accumulation returns it as a plain
# number, as for `_ProjectionScale`, and no level folds it (`_folded_level` refuses the product).
function _projection_accumulate(a::_NormalScale, b::Real)
  p = _projection_accumulate(a.s, b)
  p isa _ProjectionScale ? _NormalScale(p) : p
end
function _folded_level(λ, α::_NormalScale)
  c = _folded_level(λ, α.s)
  c === nothing || iszero(c) || abs(c) >= floatmin(typeof(float(c))) ? c : nothing
end
_fold_level(::AbstractFunction, α) = nothing
# NormL1 member of the frozen literal folding loop.
function _fold_level(f::NormL1, α)
  c = _folded_level(f.λ0, α)
  c === nothing ? nothing : NormL1(c, f.w)
end
# A normal folded level can recover a rounded leaf value of NormL2.
function _fold_level(f::NormL2, α::_NormalScale)
  c = _folded_level(f.λ, α)
  c === nothing ? nothing : NormL2(c)
end
function _fold_level(f::NormL1Nonneg, α)
  c = _folded_level(f.λ0, α)
  c === nothing ? nothing : NormL1Nonneg(c, f.w)
end
# Both levels fold, or neither. An infinite level is kept as it is: it is the same level after any positive factor,
# and the finite level still folds (the ridge of an entry with a zero weight under an infinite λ1, say). Probes of
# the partial fold see the finite level only.
@inline _elastic_folded(λ, α) = isinf(λ) ? λ : _folded_level(λ, α)
function _fold_level(f::ElasticNet, α)
  c1, c2 = _elastic_folded(f.λ1, α), _elastic_folded(f.λ2, α)
  c1 === nothing || c2 === nothing ? nothing : ElasticNet(c1, c2, f.w)
end
_fold_level(f::Scaled, α) = _fold_level(f.f, _projection_accumulate(α, f.α))
# The levels as the proximal operator folds them together with the exponent of the step, for a step
# reduced to its significand in [1/2, 1) (`_nested_step`): those of `_fold_level`, except that an
# ElasticNet ridge level that the fold takes below floatmin (to zero included) is set to zero instead
# of discarding the fold. Times that step it is below floatmin, so the ridge weight 1/(1 + γλ2) of the
# kernel is 1 for it as for zero. The folded function then lacks that ridge term, and its value is not
# the value of the chain.
_fold_level_step(f, α) = _fold_level(f, α)
function _fold_level_step(f::Union{NormL1, NormL1Nonneg}, α)
  c = _folded_level(f.λ0, α)
  c === nothing ? nothing : _with_folded_level(f, c)
end
function _fold_level_step(f::ElasticNet, α)
  c1 = _elastic_folded(f.λ1, α)
  c = iszero(f.λ2) || isinf(f.λ2) ? f.λ2 : _apply_scale(α, f.λ2)
  c2 = isfinite(c) && abs(c) < floatmin(typeof(float(c))) ? zero(c) : _elastic_folded(f.λ2, α)
  c1 === nothing || c2 === nothing ? nothing : ElasticNet(c1, c2, f.w)
end

# The scale of the fold that recovers a value: only a fold into normal levels is used. A subnormal
# level has fewer bits than the factors give, and the partial fold below recovers those values at
# normal levels. A factor that is not a finite positive IEEE number (an infinite one, an Int or a
# BigFloat) folds as it is.
_value_fold_scale(α::_ProjectionScale) = _NormalScale(α)
_value_fold_scale(α::Base.IEEEFloat) = isfinite(α) && α > 0 ? _NormalScale(_scale_start(α)) : α
_value_fold_scale(α) = α

# The partial fold. When no normal level holds the whole factor, the value of a function with
# levels is still the value at levels that do: it is positively homogeneous in them, so the factor
# α splits as 2^k β, with β small or large enough that β times every level is a normal number, the
# function folded with β is evaluated at x, and 2^k is applied last, to the significand of that value.
#
# `_fold_level(f, probe)` records the scale of a retained NormL1 level
# instead of multiplying. For a level λ times the factors inside, in [2^q, 2^(q+1)), a shift s makes
# β = a.m 2^s times it a normal number of the type T of the fold when q + s - 1 ≥ exponent(floatmin(T))
# and q + s + 1 ≤ exponent(floatmax(T)). The product of the two significands is in [1/4, 1), so these
# bounds are conservative by up to a binade at each end (and by one more at the top, where floatmax
# is below 2^(exponent + 1)): every shift in the window gives levels that `_NormalScale` accepts,
# and the window may miss a shift at its very edge. The bounds are also those of the type of the
# value, so that the levels are normal there too. The window of several levels is the
# intersection. It is empty when their exponents are too far apart, and for an infinite level,
# which is refused; the partial fold then gives no value.
mutable struct _ShiftWindow
  lo::Int
  hi::Int
end
struct _LevelProbe{S}
  w::_ShiftWindow
  a::_ProjectionScale{S}   # the factors of the `Scaled` terms inside, as a product
  emin::Int                # the exponent range of the type of the value
  emax::Int
end
function _apply_scale(p::_LevelProbe, λ)
  w = float(λ)
  T = promote_type(typeof(p.a.m), typeof(w))
  if !isfinite(w) || !(T <: Base.IEEEFloat)
    p.w.lo = typemax(Int)
    return w
  end
  mλ, eλ = frexp(w)
  _, e = frexp(convert(T, p.a.m) * convert(T, mλ))
  q = p.a.e + Int(eλ) + Int(e) - 1
  p.w.lo = max(p.w.lo, max(exponent(floatmin(T)), p.emin) + 1 - q)
  p.w.hi = min(p.w.hi, min(exponent(floatmax(T)), p.emax) - 1 - q)
  w
end
function _projection_accumulate(p::_LevelProbe, b::Real)
  c = _projection_accumulate(p.a, b)
  c isa _ProjectionScale ? _LevelProbe(p.w, c, p.emin, p.emax) : c
end
# The shifts (lo, hi) at which every level of f folds into a normal number, or nothing.
function _level_window(f, ::Type{S}, ::Type{V}) where {S, V}
  w = _ShiftWindow(typemin(Int), typemax(Int))
  g = _fold_level(f, _LevelProbe(w, _ProjectionScale(S(0.5), 1), exponent(floatmin(V)), exponent(floatmax(V))))
  g === nothing || w.lo == typemin(Int) || w.hi == typemax(Int) || w.lo > w.hi ? nothing : (w.lo, w.hi)
end
# The value of f folded with the shift s, when it is a finite normal number.
function _shifted_fold_value(f, a::_ProjectionScale, s::Int, x)
  g = _fold_level(f, _NormalScale(_ProjectionScale(a.m, s)))
  g === nothing && return nothing
  u = _value_unchecked(g, x)
  u isa Base.IEEEFloat && isfinite(u) && abs(u) >= floatmin(typeof(u)) ? u : nothing
end
_partial_scale(α::_ProjectionScale{<:Base.IEEEFloat}) = isfinite(α.m) && !iszero(α.m) ? α : nothing
_partial_scale(α::Base.IEEEFloat) = isfinite(α) && α > 0 ? _scale_start(α) : nothing
_partial_scale(α) = nothing
# The factor as a product and the direction of the partial fold: -1 where the leaf value overflowed under
# a factor below one, so that the levels are taken as low as they go; 1 where it is zero or subnormal
# under a factor above one, so as high as they go; 0 where neither holds (a NaN or a normal value, whose
# product is already the correctly rounded value, or a factor of one) or the factor is not a finite
# positive IEEE number.
@inline function _partial_plan(α, v)
  v isa Base.IEEEFloat || return nothing, 0
  a = _partial_scale(α)
  a === nothing && return nothing, 0
  isinf(v) && return a, _scale_below_one(a) ? -1 : 0
  a, _scaled_small(v) && _scale_above_one(a) ? 1 : 0
end
# The value α f(x) by the partial fold, or nothing where it gives no finite normal value at any level of
# the window: the caller then keeps what it had. The first evaluation is at the end of the window that
# the direction asks for; a value within a few precisions of either end of the range of its type is
# evaluated once more with the shift that brings it to the middle (the sum of its terms is then not at
# the edge of the range of the terms), and that value is used if it is a finite normal number. The power
# of two goes last, through `_apply_parts`: a normal result is the folded significand rounded once, and a
# result outside the range is 0 or ±Inf.
@noinline function _scaled_value_partial(f, a::_ProjectionScale, d::Int, x, v, r)
  d > 0 && iszero(v) && _exactly_zero_at(f, x) && return nothing
  win = _level_window(f, typeof(a.m), typeof(v))
  win === nothing && return nothing
  lo, hi = win
  s = d < 0 ? lo : hi
  u = _shifted_fold_value(f, a, s, x)
  u === nothing && return nothing
  T = typeof(u)
  e = exponent(u)
  if e < exponent(floatmin(T)) + precision(T) + 16 || e > exponent(floatmax(T)) - precision(T) - 16
    s1 = clamp(s - e, lo, hi)
    u1 = s1 == s ? nothing : _shifted_fold_value(f, a, s1, x)
    if u1 !== nothing
      s = s1
      u = u1
    end
  end
  m, k = _frexp(u)
  convert(typeof(r), _apply_parts(_ProjectionScale(oftype(a.m, 0.5), a.e - s + 1), m, k))
end

# Retry exceptional products in the function's own units and domain: with the scale folded into
# the levels, or else (where the levels have no normal fold) with the factor split in two, the
# fold into a subnormal level where that finds no value either, or else from the value of f as a
# significand and a power of two (`_value_parts`), with the exponents applied last. `out` is a
# gradient already written, which the parts must not read in place of a parameter. A NaN product at
# a value that is not NaN, which the fold does not recover, is Inf⋅0 at an infinite factor: 0 where
# f is certainly 0 (`_scaled_nan_value`), and otherwise what the parts give.
# The stored-entry sum for a small NormL1 value applies both scales before rounding the result.
# A symmetric iterate contributes each stored off-diagonal at both of its matrix positions.
@inline _stored_l1_small(f, α, x, v) = false
@inline _stored_l1_small(::NormL1{T,<:Union{Nothing, AbstractArray{T}}}, α::T,
                       ::_StoredIterate{T}, v::T) where {T<:Union{Float32, Float64}} =
  _scaled_small(v) && α > 1 && isfinite(α)
function _scaled_stored_l1_value(f, α, x, r::T) where {T}
  λ = f.λ0
  (λ > 0 && isfinite(λ)) || return r
  iszero(r) && _all_zero(x) && return r
  acc = _l1_weighted_abs(_expsum_start(Float64), f.w, x)
  m = acc.s + acc.c
  iszero(m) && return zero(T)
  scale = _projection_accumulate(_scale_start(Float64(α)), Float64(λ))
  convert(T, _apply_parts(scale, m, acc.ex))
end

_scaled_folded_value(f, x) = _value_unchecked(f, x)

# A folded level can leave the array's range even when the scaled sum fits.
@inline function _scaled_folded_value(f::NormL1{<:Real, Nothing}, x::DenseArray{T}) where {T<:Union{Float32, Float64}}
  _level_fits(T, f.λ0) && return _value_unchecked(f, x)
  all(isfinite, x) ? _l1_underflow_value(f, x, zero(T)) : _value_dense(f, x)
end

@noinline function _scaled_value_recover(f, α, x, v, r, out = nothing)
  _stored_l1_small(f, α, x, v) && return _scaled_stored_l1_value(f, α, x, r)
  x isa SymmetricSparseIterate && return _scaled_value_recover(f, α, matrixview(x), v, r, out)
  g = _fold_level(f, _value_fold_scale(α))
  g === nothing || return convert(typeof(r), _scaled_folded_value(g, x))
  a, d = _partial_plan(α, v)
  if d != 0
    z = _scaled_value_partial(f, a, d, x, v, r)
    z === nothing || return z
    # no level that is a normal number gives a finite normal leaf value (levels of very different
    # size, say): the fold into a subnormal level, which has fewer bits but may still hold the value
    g = _fold_level(f, α)
    g === nothing || return convert(typeof(r), _scaled_folded_value(g, x))
  end
  if isnan(r) && !isnan(v)
    z = _scaled_nan_value(f, α, x, r)
    iszero(z) && return z
  end
  _parts_wanted(f, α, x, v, out) || return r
  _recover_from_parts(f, α, x, v, r)
end
@noinline function _recover_from_parts(f, α, x, v, r)
  p = _value_parts_scaled(f, x, v, α)
  p === nothing ? r : convert(typeof(r), _apply_parts(α, p[1], p[2]))
end
_scaled_small(v) = iszero(v) || (v isa AbstractFloat && abs(v) < floatmin(v))
# The fused value of a finite factor α > 1 at a nonzero subnormal leaf value v (`value_and_prox!` with a scalar step
# whose product with the factor is normal, or with an array step and a factor inside the band), for an ElasticNet with
# positive finite levels: v carries fewer bits than its product with α needs, so the value at the stored output y is
# recomputed. The terms of both parts at the entries that y stores are added in one exponent-sum accumulator
# (`_elastic_expsum`, under the precondition with which the ElasticNet value recomputes a sum below 2 floatmin: both
# levels positive and finite, so that a NaN or ±Inf entry makes v non-finite and never reaches it), and α multiplies
# the sum before its single rounding (`_expsum_value`). This reads the stored entries of y once.
_fused_small_case(f, α, v) = false
@inline _fused_small_case(f::ElasticNet, α, v) =
  α isa Real && v isa AbstractFloat && !iszero(v) && abs(v) < floatmin(v) && α > 1 && isfinite(α) &&
  f.λ1 > 0 && f.λ2 > 0 && isfinite(f.λ1) && isfinite(f.λ2)
@noinline function _fused_small_value(f::ElasticNet, α, y, r::T) where {T}
  R = promote_type(T, Float64, _factor_type(f.λ1), _factor_type(f.λ2), _factor_type(α), _l1_factor_type(f.w),
                   _l1_factor_type(_stored_values(y)))
  m1, e1 = _frexp(convert(R, f.λ1))
  m2, e2 = _frexp(convert(R, f.λ2))
  _expsum_value(T, _elastic_expsum(_expsum_start(R), f.w, m1, e1, m2, e2, y), α)
end
# The value α v of the function f with the scale α at the unscaled value v.
@inline function _scaled_value_by(f, α, x, v, recover_inner_small = true, out = nothing)
  r = _apply_scale(α, v)
  if (!isfinite(r) && _scale_below_one(α)) || (_scaled_small(r) && !iszero(v)) ||
     (recover_inner_small && _scaled_small(v) && _scale_above_one(α))
    return _scaled_value_recover(f, α, x, v, r, out)
  end
  r
end
# Inf⋅0 at an infinite factor: 0 only where f is certainly 0 at x (`_exactly_zero_at`); otherwise the
# product r is returned unchanged. A computed 0 may be a positive value that underflowed, whose scaled
# value is ±Inf: the recovery then tries the parts of the value, which give ±Inf for such a value on
# host arrays, and the NaN stays where they do not apply.
@noinline _scaled_nan_value(f, α, x, r) = _infinite_factor(α) && _exactly_zero_at(f, x) ? zero(r) : r
_infinite_factor(α::Real) = isinf(α)
_infinite_factor(α) = false
# α v with an infinite factor α, for the array-step route, which does not recover a product
_scaled_product_value(f, α, y, v) = (r = α * v; isnan(r) && !isnan(v) ? _scaled_nan_value(f, α, y, r) : r)

# Whether f(x) is exactly 0, decided from the argument and the stored parameters, never from a
# computed value. Called only where the computed value of f is 0 (not NaN). An indicator: its value
# is exactly 0 or Inf. The families below: their value is 0 exactly when every entry they penalize
# is 0. The entries they leave free (a zero level or weight, an index in no group) do not enter the
# value, whatever they hold, as in the function's own value. Any other function: false.
_exactly_zero_at(f, x) = _isindicator(f)
# an entry with the weight w under the level λ is penalized (an infinite weight at any level)
@inline _penalized(λ, w) = w > 0 && (λ > 0 || isinf(w))
function _exactly_zero_at(f::NormL1, x::AbstractArray)
  w = f.w
  w === nothing || length(w) == length(x) || return false
  λ = f.λ0
  for (j, i) in enumerate(eachindex(x))
    wi = w === nothing ? one(λ) : w[j]
    _penalized(λ, wi) && !iszero(x[i]) && return false
  end
  true
end
# Exact zero also requires nonnegative entries.
_exactly_zero_at(f::NormL1Nonneg, x::AbstractArray) = all(>=(0), x) && _exactly_zero_at(f.l1, x)
# a positive ridge level penalizes every entry (an infinite one is the indicator of {0}, #908); a zero
# one leaves the entries that the ℓ1 part penalizes, as for NormL1
_exactly_zero_at(f::ElasticNet, x::AbstractArray) =
  f.λ2 > 0 ? _all_zero(x) : _exactly_zero_at(f.l1, x)

# The recovery of α f(x) for an ElasticNet f: the recovery of every function first (the fold of the factor into both
# levels, the partial fold and the fold into a subnormal level, the rule of an infinite factor at a point where f is 0,
# the product itself otherwise). Where that gives no finite value, or 0 at an x where f is not exactly 0, and the factor
# is a finite positive number, the sum of the two parts: α λ1 Σ wᵢ|xᵢ| from Scaled(NormL1(λ1, w), α) and α λ2/2 Σ xᵢ²
# from Scaled(SqrNormL2(λ2), α) (the indicator value for an infinite λ2, which such a factor leaves unchanged), each
# recovered as its family recovers it, rounded on its own, and added once; the sum replaces the first result only if
# it is finite after conversion to the type of that result.
# The two levels need not fold by one factor: ElasticNet(2^-1000, 2^1000) under the factor 2^-600 at [2^300] has the
# ℓ1 part 2^-1300, the ridge part overflows before the factor, and the value is 2^999. A first result of 0 can be a
# value that the parts recover: ElasticNet(0, 2.242946011063277e131, [2.5]) under 1.6778225587956818e288 at
# [4.4835202439803e-310] is 3.7824493718933436e-200, as for Scaled(SqrNormL2). For parts that are not NaN the sum is
# nonnegative, and it is 0 where both parts are 0.
@noinline function _scaled_value_recover(f::ElasticNet, α, x::AbstractArray, v, r, out = nothing)
  z = invoke(_scaled_value_recover, Tuple{Any, Any, Any, Any, Any, Any}, f, α, x, v, r, out)
  _elastic_finite_scale(α) || return z
  isfinite(z) && !iszero(z) && return z
  iszero(z) && _exactly_zero_at(f, x) && return z
  l = _scaled_value_by(f.l1, α, x, _value_unchecked(f.l1, x))
  λ = f.λ2
  q = iszero(λ) ? zero(l) : isinf(λ) ? _elastic_ridge_value(λ, x) :
      (h = SqrNormL2(λ); _scaled_value_by(h, α, x, _value_unchecked(h, x)))
  p = convert(typeof(z), l + q)
  isfinite(p) ? p : z
end
_elastic_finite_scale(α::Real) = isfinite(α) && α > 0
_elastic_finite_scale(α::_ProjectionScale) = isfinite(α.m) && !iszero(α.m)
_elastic_finite_scale(α) = false

# The scaled-value mechanism (numerics.md, "Overflow and underflow of values"). A function with
# `_has_value_parts` returns its value at x as (m, e), the value m 2^e with a finite significand m, or
# `nothing` to keep the ordinary result (`_value_parts`, defaults in utils.jl). Only host arrays take it;
# a NaN value stays NaN, and a normal value is its own significand and exponent, so the function is asked
# only for a value that is infinite, zero or subnormal. An exactly zero value whose true value,
# times α, cannot be a normal number (`_zero_value_exponent`) is not recomputed.
@inline _parts_argument(x) = false
@inline _parts_argument(::Union{Array, _HostArray, SparseIterate, SymmetricSparseIterate, SymmetricMatrixView, BlockIterate,
                                SparseArrays.AbstractSparseArray, AbstractRange}) = true
@inline function _parts_wanted(f, α, x, v, out)
  _has_value_parts(typeof(f)) || return false
  _parts_argument(x) || return false
  v isa AbstractFloat || return false
  if iszero(v)
    E = _zero_value_exponent(f, x)
    E === _EXACT_ZERO && return false
    # |α f(x)| < 2^(E + e) for α < 2^e: below the normal range of the value when E + e is at most
    # the exponent of its floatmin. The value is returned in the type of x (`_value_at`), whose
    # normal range is the narrower when α is wider than x (a Float64 factor at a Float32 argument):
    # the larger of the two exponents decides.
    E === nothing || E + _scale_upper_exponent(α) >
      max(exponent(floatmin(_valtype(x))), exponent(floatmin(typeof(float(_apply_scale(α, v)))))) || return false
  elseif !isinf(v) && (isnan(v) || abs(v) >= floatmin(v))
    return false
  end
  out === nothing || !_parameter_alias(f, out)
end
_scaled_parts(f, α, x, v, out) = _parts_wanted(f, α, x, v, out) ? _value_parts_scaled(f, x, v, α) : nothing
# e with α < 2^e
@inline _scale_upper_exponent(α::Real) = _upper_exponent(α)
@inline _scale_upper_exponent(a::_ProjectionScale) = a.e
# the parts of f, for wrappers that pass the scale on to the function inside them
_value_parts_scaled(f, x, v, α) = _value_parts(f, x, v)
# α m 2^e, rounded once: the product of the significands, with the exponents applied last
function _apply_parts(α, m, e::Integer)
  ma, ea = _projection_scale_parts(α)
  P = promote_type(typeof(float(ma)), typeof(m))
  # ea is a retained exponent that may have wrapped: Int sums (#767)
  _scaled_product_retained(convert(P, ma), convert(P, m), Int(ea) + e)
end
@inline _scaled_value(s, x, v, recover_inner_small = true) = _scaled_value_by(s.f, s.α, x, v, recover_inner_small)
(s::Scaled)(x) =
  (_check_value_arrays(s, x); _value_checked(s, x))
_value_unchecked(s::Scaled, x) = _chained(s) ? _value_chain(s, x) : _value_at(_scaled_value(s, x, _value_unchecked(s.f, x)), x)
# a chain is evaluated at the function inside it with the product of all its factors
function _value_chain(s, x)
  f, a = _scaled_chain(s)
  _value_at(_scaled_value_by(f, a, x, _value_unchecked(f, x)), x)
end
_checkvalue(s::Scaled, x) = _checkvalue(s.f, x)
_scale!(g::AbstractArray, α) = (g .*= α)
_scale!(g::AbstractArray, α::_ProjectionScale) = (g .= _apply_scale.(α, g))
# Finite scales preserve the active set, including explicitly stored zeros.
# Non-finite scales can turn an implicit zero into NaN and activate it.
function _scale!(g::SymmetricSparseIterate, α)
  if isfinite(α)
    v = activevalues(g)
    v .*= α
  else
    _scale!(coordinates(g), α)
  end
  g
end
function _scale!(g::SymmetricSparseIterate, α::_ProjectionScale)
  if isfinite(α)
    v = activevalues(g)
    v .= _apply_scale.(α, v)
  else
    _scale!(coordinates(g), α)
  end
  g
end
@generated _scale!(g::Tuple, α) = _blockwise(k -> :(_scale!(g[$k], α)), fieldcount(g), :g)
value_and_gradient!(g, s::Scaled, x) =
  (_check_oracle_arrays(value_and_gradient!, s, g, x); _value_and_gradient_unchecked!(g, s, x))
_value_and_gradient_unchecked!(g, s::Scaled, x) = _chained(s) ? _value_and_gradient_chain!(g, s, x) :
  (_checkwrite(value_and_gradient!, s, g, x); v = _value_and_gradient_unchecked!(g, s.f, x); _scale!(g, s.α); _value_at(_scaled_value_by(s.f, s.α, x, v, true, g), x))
function _value_and_gradient_chain!(g, s, x)
  _checkwrite(value_and_gradient!, s, g, x)
  f, a = _scaled_chain(s)
  v = _value_and_gradient_unchecked!(g, f, x)
  _scale!(g, a)
  _value_at(_scaled_value_by(f, a, x, v, true, g), x)
end
gradient!(g, s::Scaled, x) =
  (_check_oracle_arrays(gradient!, s, g, x); _gradient_unchecked!(g, s, x))
_gradient_unchecked!(g, s::Scaled, x) = _chained(s) ? _gradient_chain!(g, s, x) :
  (_checkwrite(gradient!, s, g, x); _gradient_unchecked!(g, s.f, x); _scale!(g, s.α))
function _gradient_chain!(g, s, x)
  _checkwrite(gradient!, s, g, x)
  f, a = _scaled_chain(s)
  _gradient_unchecked!(g, f, x)
  _scale!(g, a)
end
prox!(y, s::Scaled, x, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, s, y, x, γ); _prox_unchecked!(y, s, x, γ))
# The step that Scaled passes to f. The scale of an indicator does not matter, so a zero step stays
# zero for α = Inf instead of becoming the NaN of Inf * 0; that NaN stays for any other f.
@inline _scaled_step(s::Scaled, γ::Real) = _isindicator(s.f) && iszero(γ) ? γ : s.α * γ

# A single factor takes the route of a chain (`_nested_step`) for a step that it takes out of the
# range, so that the product α γ is not rounded before the function forms its coefficient. A zero,
# infinite, negative or NaN step or factor keeps its meaning, and an indicator ignores its scale.
@inline function _step_exceptional(s::Scaled, γ::Real)
  p = s.α * γ
  p isa AbstractFloat || return false
  # one test for the ordinary product: a loop over `prox_block` still vectorizes with a single
  # branch to a function that is not inlined, and does not with a chain of tests
  ((p >= floatmin(typeof(p))) & (p <= floatmax(typeof(p)))) && return false
  _step_positive_finite(s, γ)
end
@noinline _step_positive_finite(s::Scaled, γ::Real) =
  isfinite(s.α) && isfinite(γ) && s.α > 0 && γ > 0 && !_isindicator(s.f)
@inline _nested_route(s::Scaled, γ::Real) = _chained(s) || _step_exceptional(s, γ)

# What the function inside a nested chain receives for the step γ: the function, its step, and the
# factor still to apply to its value.
# - An indicator ignores its scale, so it keeps γ.
# - A step that is not a finite positive number, which the function rejects or reads as a limit,
#   is only multiplied.
# - The product of the factors and γ goes into the step when it is a normal number.
# - Otherwise, for a NormL1 or NormL1Nonneg whose level is a positive finite IEEE number after `float`,
#   and when the level that the fold below would store (the product of the factors, the step and the
#   level, divided by the significand of the step) is not a normal number (it is zero, it is subnormal,
#   or it overflows, which can happen although the product of the factors, the step and the level is
#   at most floatmax), the exponent is split between the step and the level (`_split_level_step`) if
#   the step and the folded level are both normal numbers at the split, though their product need not
#   be one. The kernel of the leaf then forms the threshold γλ0wᵢ from them: from the three
#   significands with one exponent on its exceptional branch, as (γλ0)wᵢ on its ordinary one. The
#   returned factor is `nothing`.
# - Otherwise the exponent of the factors goes into the levels, with the step reduced to its
#   significand, for the functions that fold a scale into their levels (`_fold_level`): the
#   products of level and step are those of the plain chain, formed in range. For a function that the
#   split does not cover (ElasticNet, for one), the fold can store a subnormal level.
# - A function without levels to fold, or a fold that is refused, gets the step as rounded.
# The factor is the one that takes the value of the returned function to the value of the chain.
# Marked `@inline`: the result is a union of tuples (folded or plain function, the step, and the
# factor or `nothing`), which a call that is not inlined must box. Without the mark, Julia 1.10.13 did
# not inline the call for some leaf types (NormL1Nonneg, for one) into `_prox_chain!`,
# `_value_and_prox_chain!` and `_prox_block_nested`, and the chain routes allocated for the boxed
# result. The callers that sit in a loop over `prox_block` call `_prox_block_nested`, which is not
# inlined itself.
@inline function _nested_step(s::Scaled, γ::Real)
  f, a = _scaled_chain(s)
  _isindicator(f) && return (f, γ, a)
  step = _apply_scale(a, γ)
  T = typeof(step)
  (T <: AbstractFloat && isfinite(γ) && γ > 0 && !(floatmin(T) <= step <= floatmax(T))) || return (f, step, a)
  mγ, eγ = _frexp(float(γ))
  # A function with a stored infinite level, all of whose blocks are fixed by it, does not depend on the
  # factor and the step (`_hard_function`): nothing finite is folded. Otherwise the earlier route.
  if _has_infinite_level(f) && _hard_scale(a)
    h = _hard_function(f)
    h === nothing || return (h, convert(typeof(step), mγ), nothing)
  end
  ma, ea = _projection_scale_parts(a)
  e = Int(ea + eγ)
  h = _split_level_step(f, ma, e, convert(T, mγ))
  h === nothing || return (h[1], h[2], nothing)
  g = _fold_level_step(f, _ProjectionScale(ma, e))
  # g carries the levels without the ridge that may have been dropped: it is not the chain, and
  # its value is not the value of the chain (`nothing` marks this)
  g === nothing ? (f, step, a) : (g, convert(typeof(step), mγ), nothing)
end
# The function and the step of the split of the exponent e (the exponent of the factors and of γ, whose
# significand is mγ) between the step and the level, for a NormL1 or NormL1Nonneg whose level is a
# positive finite IEEE number after `float` and that the fold of the whole exponent into the level
# (`_nested_step`) would not leave a normal number: the step takes mγ 2^k and the level the exponent
# e - k, with k near half the exponent of the product of the level and 2^e, so that each is a normal
# number although their product need not be. `nothing` for the other functions and levels, when the
# fold stores a normal level, and when the step or the folded level is not a normal number at that
# split (for example when the product is beyond what a normal step and a normal level hold).
@inline _split_level_step(f, ma, e::Int, mγ) = nothing
@inline function _split_level_step(f::Union{NormL1, NormL1Nonneg}, ma, e::Int, mγ::T) where {T<:Base.IEEEFloat}
  λ = float(f.λ0)
  (λ isa Base.IEEEFloat && isfinite(λ) && λ > 0) || return nothing
  _folded_level(f.λ0, _NormalScale(_ProjectionScale(ma, e))) === nothing || return nothing
  k = (e + _frexp(λ)[2]) >> 1
  step = ldexp(mγ, k)
  floatmin(T) <= step <= floatmax(T) || return nothing
  g = _fold_level_step(f, _NormalScale(_ProjectionScale(ma, e - k)))
  g === nothing ? nothing : (g, step)
end

_prox_unchecked!(y, s::Scaled, x, γ::Real) = _nested_route(s, γ) ? _prox_chain!(y, s, x, γ) :
  (_checkwrite(prox!, s, y, x); _prox_unchecked!(y, s.f, x, _scaled_step(s, γ)))
@noinline function _prox_chain!(y, s, x, γ::Real)
  _checkwrite(prox!, s, y, x)
  g, step, _ = _nested_step(s, γ)
  _prox_unchecked!(y, g, x, step)
end
# Along with the leaf value, return whether its kernel certified an exact zero.
@inline _scaled_leaf_value_and_prox!(y, f, x, γ) = (_value_and_prox_unchecked!(y, f, x, γ), false)
@inline _scaled_leaf_value_and_prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real) =
  _norml2_value_and_prox!(y, f, x, γ)
@inline function _scaled_leaf_value_and_prox!(y::_HostArray, f::NormL1{T,<:Union{Nothing, AbstractArray{T}}},
                                              x::_HostArray, γ::Union{Real, AbstractArray}) where {T<:Union{Float32, Float64}}
  if eltype(y) === T && eltype(x) === T
    _prox_sep!(y, f, x, γ)
    return _l1_dense_result(f, y)
  end
  _value_and_prox_unchecked!(y, f, x, γ), false
end

value_and_prox!(y, s::Scaled, x, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, s, y, x, γ); _value_and_prox_unchecked!(y, s, x, γ))
function _value_and_prox_unchecked!(y, s::Scaled, x, γ::Real)
  _nested_route(s, γ) && return _value_and_prox_chain!(y, s, x, γ)
  _checkwrite(value_and_prox!, s, y, x)
  v, exact_zero = _scaled_leaf_value_and_prox!(y, s.f, x, _scaled_step(s, γ))
  exact_zero && isfinite(s.α) && return _value_at(v, y)
  r = _fused_small_case(s.f, s.α, v) ? _fused_small_value(s.f, s.α, y, s.α * v) :
      _scaled_value(s, y, v, s.f isa Union{NormL1, NormL2} || isinf(s.α))
  _value_at(r, y)
end
@noinline function _value_and_prox_chain!(y, s, x, γ::Real)
  _checkwrite(value_and_prox!, s, y, x)
  g, step, a = _nested_step(s, γ)
  if a === nothing
    # the value of the chain at the output, not that of the folded function
    _prox_unchecked!(y, g, x, step)
    f, b = _scaled_chain(s)
    return _value_at(_scaled_value_by(f, b, y, _value_unchecked(f, y)), y)
  end
  v = _value_and_prox_unchecked!(y, g, x, step)
  _value_at(_scaled_value_by(g, a, y, v), y)
end
Base.@propagate_inbounds function prox_block(s::Scaled, v::Real, i::Integer, γ::Real)
  _nested_route(s, γ) && return _prox_block_nested(s, v, i, γ)
  prox_block(s.f, v, i, s.α * γ)
end
@noinline function _prox_block_nested(s::Scaled, v::Real, i::Integer, γ::Real)
  g, step, _ = _nested_step(s, γ)
  prox_block(g, v, i, step)
end
_checkparams(s::Scaled, x) = _checkparams(s.f, x)
_checkwrite(op::O, s::Scaled, y, x) where {O} = (_checkreal(s, y); _checkreal(s, x); (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x); _checkwrite(op, s.f, y, x))
iszeropreserving(s::Scaled) = isfinite(s.α) && iszeropreserving(s.f)
_isindicator(::Type{<:Scaled{F}}) where {F} = _isindicator(F)
kkt_violation(s::Scaled, g::AbstractArray, i::Integer) =
  (_check_one_based(g); eltype(g) <: Real || _real_error(:kkt_violation); _chained(s) ? _screen_chain(s, g, i) :
   _isindicator(_kkt_innermost(s.f)) ? kkt_violation(s.f, g, i) :
   iszero(s.α) ? abs(g[i]) : s.α * kkt_violation(s.f, _ScaledArray(g, inv(s.α)), i))

# ---- the point-aware scores of a scaled function ----
# Which leaves have the four-argument and full-point scores: _KktScore for the functions of this package that have them
# (NormL2 and NormLinf have the full-point form and throw an ArgumentError in the others), _KktNone for the other
# functions of this package, _KktCustom for a function defined outside it. The class of a Scaled is that of the function
# under all its wrappers. The class is a singleton type chosen by dispatch, or by the module of the type, which inference
# folds.
struct _KktScore end
struct _KktNone end
struct _KktCustom end
Base.@assume_effects :foldable _kkt_class(::Type{F}) where {F} = parentmodule(F) === (@__MODULE__) ? _KktNone() : _KktCustom()
_kkt_class(::Type{<:Union{NormL1, ElasticNet, ZeroFunction, LinearFunction, SqrNormL2, GroupNormL2, GroupNormLinf, NormL2, NormLinf}}) = _KktScore()
_kkt_class(::Type{<:NormL1Nonneg}) = _KktScore()
_kkt_class(::Type{<:_SmoothKktFunction}) = _KktScore()
_kkt_class(::Type{<:Scaled{F}}) where {F} = _kkt_class(F)
_kkt_innermost(f::AbstractFunction) = f
_kkt_innermost(s::Scaled) = _kkt_innermost(s.f)

# Whether the coordinate or group that a call selects holds a nonzero entry of x: coordinate k of an array or of a
# symmetric iterate, group k of a BlockIterate, and, for GroupNormL2 and GroupNormLinf, the group of entry k of an array (an
# entry in no group has none).
_kkt_nonzero_at(f, x::AbstractArray, k) = !iszero(x[k])
_kkt_nonzero_at(f, x::BlockIterate, k) = any(j -> !iszero(x[j]), x.groups[k])
_kkt_nonzero_at(f, x::SymmetricSparseIterate, k) = !iszero(_getstored(x, k))
function _kkt_nonzero_at(f::_GroupNorm, x::AbstractArray, i)
  k = i <= length(f.group_of) ? f.group_of[i] : 0
  k != 0 && any(j -> !iszero(x[j]), f.groups[k])
end
_kkt_nonzero_at(f::_GroupNorm, x::BlockIterate, k) = any(j -> !iszero(x[j]), x.groups[k])
# the same for the refusals of an infinite factor and of an unfoldable chain, which also count a nonzero entry in no group of a
# GroupNormL2 or GroupNormLinf as a nonzero coordinate
_kkt_nonzero_coord(f, x, k) = _kkt_nonzero_at(f, x, k)
_kkt_nonzero_coord(f::_GroupNorm, x::AbstractArray, i) = (i <= length(f.group_of) && f.group_of[i] != 0) ? _kkt_nonzero_at(f, x, i) : !iszero(x[i])
_kkt_nonzero_coord(f::_GroupNorm, x::BlockIterate, k) = _kkt_nonzero_at(f, x, k)

@noinline _kkt_infinite_error(s) = throw(ArgumentError(
  "kkt_violation of a Scaled function with an infinite factor is not defined at a nonzero coordinate or group, or in the full-point form"))
@noinline _kkt_chain_error(s, f) = throw(ArgumentError(
  "kkt_violation of a chain of $(_kkt_depth(s)) Scaled wrappers around $(nameof(typeof(f))) is not defined at a nonzero " *
  "coordinate or group, or in the full-point form: the product of the factors is not a normal number and does not fold " *
  "into a normal level of the function"))
_kkt_depth(::AbstractFunction) = 0
_kkt_depth(s::Scaled) = 1 + _kkt_depth(s.f)

# What a chain of wrappers around the leaf f, with the product a of its factors, is scored as, or nothing: the product
# is applied to a parameter of the leaf, and v is not divided by it.
#   ZeroFunction                 itself (|vᵢ|);
#   LinearFunction, SqrNormL2    |vᵢ + a dᵢ| with dᵢ the entry of the leaf's gradient (_KktChain);
#   NormL1, NormL1Nonneg, ElasticNet
#                                the function with the folded levels (_fold_level);
#   GroupNormL2, GroupNormLinf, NormL2
#                                the block kernel (for GroupNormLinf the margin or the distance of a group) with the level
#                                folded by _folded_level (_KktLevel);
#   NormLinf                     the full-point kernel with the level folded by _folded_level (_KktLevel).
# Levels that do not fold to a normal number, and a function defined outside the package, have none.
_kkt_resolve(f::ZeroFunction, a) = f
_kkt_resolve(f::Union{LinearFunction, SqrNormL2}, a) = _KktChain(f, a)
_kkt_resolve(f::Union{NormL1, NormL1Nonneg, ElasticNet}, a) = _fold_level(f, a)
_kkt_resolve(f::_GroupNorm, a) = (λ = _folded_level(f.λ0, a); λ === nothing ? nothing : _KktLevel(f, λ))
_kkt_resolve(f::NormL2, a) = (λ = _folded_level(f.λ, a); λ === nothing ? nothing : _KktLevel(f, λ))
_kkt_resolve(f::NormLinf, a) = (λ = _folded_level(f.λ, a); λ === nothing ? nothing : _KktLevel(f, λ))
_kkt_resolve(f, a) = nothing

# |vᵢ + a dᵢ| with dᵢ the entry of the gradient of f at x, multiplied by a as the gradient of the chain multiplies it
# (_apply_scale), in a working type W that includes the type of the factors. It is not an AbstractFunction: it has the
# entry kernels of interface.jl, which are what the local, block, symmetric and full-point forms of an entrywise function
# call. An uncentered ridge uses the zero-gradient screen |vᵢ| at a zero coordinate.
struct _KktChain{F, A}
  f::F
  a::A
end
@inline _kkt_type(c::_KktChain, ::Type{V}, ::Type{X}) where {V, X} = _kkt_type(c.f, promote_type(V, _kkt_factor_type(c.a)), X)
@inline _kkt_factor_type(a::_ProjectionScale) = typeof(float(a.m))
@inline _kkt_factor_type(a::Real) = typeof(float(a))
@inline _kkt_state(c::_KktChain, ::Type{W}) where {W} = _kkt_state(c.f, W)
@inline _kkt_fast_ok(::_KktChain, s) = false
_checkparams(c::_KktChain, x) = _checkparams(c.f, x)
Base.@propagate_inbounds function _kkt_entry(c::_KktChain{<:LinearFunction}, s, ::Type{W}, v, x, i) where {W}
  vw = convert(W, v)
  _kkt_override(abs(vw + convert(W, _apply_scale(c.a, c.f.c[i]))), vw)
end
Base.@propagate_inbounds function _kkt_entry(c::_KktChain{<:SqrNormL2}, s, ::Type{W}, v, x, i) where {W}
  vw = convert(W, v)
  xw = convert(W, x)
  iszero(xw) && c.f.y === nothing && return _kkt_override(abs(vw), vw, xw)
  _kkt_override(abs(vw + convert(W, _apply_scale(c.a, _kkt_ridge(c.f.y, s, W, xw, i)))), vw, xw)
end
# The coefficient a λ w of an entry, or of a group, of a chain whose leaf does not resolve (_kkt_resolve returns nothing; for an
# ElasticNet one level that does not fold is enough, so the λ passed here can be a level that folds on its own), rounded
# to the floating type W (which includes the type of the factors of a). For a positive finite λ and w, the significands of a, λ
# and w are multiplied and the exponents of the three are applied by one final ldexp (_scaled_product_retained), so that neither
# a λ nor λ w is rounded first: either could underflow or overflow where a λ w is representable. _kkt_coefficient(W, λ, w)
# takes a plain level, so composing it with a would apply a to λ (or to λ w) first, which is that rounding. For a zero
# or infinite λ or w the coefficient is that of _kkt_coefficient (a zero weight gives 0, also for an infinite λ; an infinite
# weight or level gives Inf), which a positive finite a does not change.
@inline function _chain_coefficient(::Type{W}, λ, w, a::_ProjectionScale) where {W}
  if isfinite(λ) & isfinite(w) & (λ > 0) & (w > 0)
    R = _working_type(W, λ, w)
    return convert(W, _scaled_product_retained(convert(R, λ), convert(R, w), convert(R, a.m), a.e))
  end
  _kkt_coefficient(W, λ, w)
end
# The screen |vᵢ| - c of the three-argument form, with c = _chain_coefficient (a λ wᵢ for a positive finite λ and wᵢ), which is
# the value of the entry at a zero coordinate (a nonzero coordinate is refused by the caller of the four-argument form after
# this value is formed). The endpoint cases are those of the
# three-argument screen of NormL1 and ElasticNet: a NaN entry gives NaN, and |Inf| - Inf is NaN. An infinite λ2 of an
# ElasticNet gives -Inf for any entry, as its screen does.
Base.@propagate_inbounds function _kkt_entry(c::_KktChain{<:NormL1, <:_ProjectionScale}, s, ::Type{W}, v, x, i) where {W}
  abs(convert(W, v)) - _chain_coefficient(W, c.f.λ0, _entry(c.f.w, i), c.a)
end
# NormL1Nonneg: the screen -vᵢ - c of a zero coordinate, with the same coefficient c; a nonzero coordinate (a negative
# one included) is refused by the caller after this value is formed
Base.@propagate_inbounds function _kkt_entry(c::_KktChain{<:NormL1Nonneg, <:_ProjectionScale}, s, ::Type{W}, v, x, i) where {W}
  -convert(W, v) - _chain_coefficient(W, c.f.λ0, _entry(c.f.w, i), c.a)
end
Base.@propagate_inbounds function _kkt_entry(c::_KktChain{<:ElasticNet, <:_ProjectionScale}, s, ::Type{W}, v, x, i) where {W}
  isinf(c.f.λ2) && return W(-Inf)
  abs(convert(W, v)) - _chain_coefficient(W, c.f.λ1, _entry(c.f.w, i), c.a)
end
# The level of a GroupNormL2 or GroupNormLinf under a chain whose level does not fold: the level λ of the function and the product
# a of the factors, not combined. Its _l2_score and _linf_group_score are the zero-group score with the coefficient c of
# _chain_coefficient for the group (a λ w for a positive finite λ and w), the margin ‖v[G]‖₁ - c for GroupNormLinf.
struct _KktChainLevel{T, A<:_ProjectionScale}
  λ::T
  a::A
end
# W includes the type of the factors. The value at a nonzero group is not used by the caller, which refuses it. The endpoint
# cases are those of the zero-block score (_l2_zero_score): an infinite entry of v[G] gives Inf, also where the coefficient is
# Inf, and a NaN entry gives NaN.
@inline function _l2_score(λ::_KktChainLevel, w, v, x, G)
  W = promote_type(float(promote_type(eltype(v), eltype(x))), _kkt_factor_type(λ.a))
  _l2_zero_score(_chain_coefficient(W, λ.λ, w, λ.a), true, v, G, W)
end
# The same for a group of GroupNormLinf: the margin of the zero group, with the coefficient c passed as the level and the weight
# true, which _kkt_coefficient returns as it is. The entries of x[G] are not read (x enters through its element type only); the
# caller refuses a nonzero group after this value is formed (_kkt_chain), so the distance of a nonzero group is not formed here.
@inline function _linf_group_score(f::GroupNormLinf, λ::_KktChainLevel, k::Integer, v, x)
  W = promote_type(float(promote_type(eltype(v), eltype(x))), _kkt_factor_type(λ.a))
  _linf_zero_score(_chain_coefficient(W, λ.λ, @inbounds(f.w[k]), λ.a), true, v, @inbounds(f.groups[k]), W)
end
# a GroupNormL2, GroupNormLinf, NormL2 or NormLinf with the level λ that the chain folds into it
struct _KktLevel{F, T}
  f::F
  λ::T
end

# the four-argument and full-point scores of what _kkt_resolve returns
_kkt_apply(h, g, x::AbstractArray, k) = _kkt_local(h, g, x, k)
_kkt_apply(h, g, x::BlockIterate, k) = _kkt_block(h, g, x, k)
_kkt_apply(h, g, x::SymmetricSparseIterate, k) = _kkt_symmetric(h, g, x, k)
_kkt_apply(h::_KktLevel, g, x::AbstractArray, k) = _level_score(h.f, h.λ, g, x, k)
_kkt_apply(h::_KktLevel, g, x::BlockIterate, k) = _level_score(h.f, h.λ, g, x, k)
_kkt_apply(h::_KktLevel, g, x::SymmetricSparseIterate, k) = _level_score(h.f, h.λ, g, x, k)
_kkt_full(h, v, x::AbstractArray) = _kkt_point(h, v, x)
_kkt_full(h::_KktLevel, v, x::AbstractArray) = _level_point(h.f, h.λ, v, x)

# The screen of a chain (the three-argument form). Around LinearFunction and a centered SqrNormL2 it is |gᵢ + a dᵢ| with
# dᵢ the gradient entry at zero. For NormL1 and ElasticNet it is that of the function with the folded levels and, where the
# levels do not fold, |gᵢ| - c with the coefficient c ≥ 0 of _chain_coefficient (a λ wᵢ for a positive finite λ and wᵢ), formed
# without dividing g by a; an ElasticNet with an infinite λ2 returns -Inf there before c is formed. For the
# other functions it is that of the function with the folded levels, and where the levels do not fold, that of the function
# at g divided by the product (kept as an exponent), multiplied by it, which does not recover a g that is out of range after
# the division. Other smooth functions and indicators forward their existing three-argument screen, where one is defined.
# A chain with an infinite factor takes the last two routes (_chain_screen_plain); its three-argument value is outside the
# contract (see the scaling guide) and is computed by them, without a guarantee.
_screen_chain(s::Scaled, g, i) = ((f, a) = _scaled_chain(s); _chain_finite(a) ? _chain_screen(f, a, g, i) : _chain_screen_plain(f, a, g, i))
_chain_screen(f, a, g, i) = _chain_screen_plain(f, a, g, i)
function _chain_screen_plain(f, a, g, i)
  (issmooth(typeof(f)) || _isindicator(f)) && return kkt_violation(f, g, i)
  h = _fold_level(f, a)
  h === nothing ? _apply_scale(a, kkt_violation(f, _ExpScaledArray(g, _scale_inverse(a)), i)) : kkt_violation(h, g, i)
end
# NormL1 and ElasticNet: the fold, where the levels fold, and otherwise the screen |gᵢ| - c with the coefficient c ≥ 0 of
# _chain_coefficient (a λ wᵢ for a positive finite λ and wᵢ), which is formed without dividing g by a; an ElasticNet with an
# infinite λ2 returns -Inf before c is formed. NormL1Nonneg: the same with the screen -gᵢ - c.
function _chain_screen(f::Union{NormL1, ElasticNet}, a::_ProjectionScale, g, i)
  h = _fold_level(f, a)
  h === nothing || return kkt_violation(h, g, i)
  _chain_screen_entry(_KktChain(f, a), g, i)
end
function _chain_screen(f::NormL1Nonneg, a::_ProjectionScale, g, i)
  h = _fold_level(f, a)
  h === nothing || return kkt_violation(h, g, i)
  _chain_screen_entry(_KktChain(f, a), g, i)
end
_chain_screen(f::LinearFunction, a, g, i) = _chain_screen_entry(_KktChain(f, a), g, i)
_chain_screen(f::SqrNormL2, a, g, i) = f.y === nothing ? kkt_violation(f, g, i) : _chain_screen_entry(_KktChain(f, a), g, i)
@inline function _chain_screen_entry(c, g, i)
  W = _kkt_type(c, eltype(g), eltype(g))
  _kkt_entry(c, _kkt_state(c, W), W, g[i], zero(W), i)
end

# The four-argument form. A wrapper with a finite positive factor α evaluates the existing score of its function on the lazy
# gradient v/α, with the rounding and range limits of that evaluation (underflow, overflow and NaN are not recovered), and
# multiplies it by α. For one wrapper with a finite positive factor, with finite entries of v/α, at a zero coordinate and for
# arrays and parameters of one floating type, this is the scaled three-argument screen, bit for bit, except for the endpoint
# cases of the numerical contract. A function of this
# package with a score uses its private point-aware score; a function defined outside the package uses its screen at a zero
# coordinate or group and is refused at a nonzero one, also when it defines a four-argument method, unless the method for
# Scaled{<:T} is defined. A function of this package without a score gives a MethodError for the call on the Scaled. An
# infinite factor is refused at a nonzero coordinate or group, also in a retained chain that holds one; at a zero one, one
# wrapper with an infinite factor takes the formula's value for a finite v. A stack that nests an infinite factor with another
# Scaled is built by a direct call of the parameterized constructor Scaled{F, T} (Scaled(f, α) merges the factors): it is
# refused in the same cases, and its values at a zero coordinate or group are outside the contract and are computed by
# the existing wrapper routes, without a guarantee.
# The rare branches are in functions that are not inlined, so that the route is small enough to be inlined into the call
# that reaches it. Inlining permits the compiler to hoist loop-invariant state for the unweighted cases; weighted
# coefficients remain entry-dependent.
@inline function _kkt_scaled(s::Scaled, g, x, k)
  cls = _kkt_class(typeof(s))
  cls isa _KktNone && _kkt_none_error(s, g, x, k)
  _chained(s) && return _kkt_chain(s, g, x, k)
  r = s.α * _kkt_inner(s.f, _ScaledArray(g, inv(s.α)), x, k)
  isinf(s.α) && _kkt_infinite_guard(cls, s, x, k)
  r
end
@noinline _kkt_none_error(s, g, x, k) = throw(MethodError(kkt_violation, (s, g, x, k)))
@noinline function _kkt_infinite_guard(cls, s, x, k)
  cls isa _KktScore && _kkt_nonzero_coord(_kkt_innermost(s), x, k) && _kkt_infinite_error(s)
  nothing
end
# the function under one wrapper: a Scaled through its own route, a function of this package by its own method, and a
# function defined outside the package by its screen at a zero coordinate or group (_kkt_custom_zero)
@inline _kkt_inner(f::Scaled, g, x, k) = _kkt_scaled(f, g, x, k)
@inline _kkt_inner(f, g, x, k) = _kkt_leaf(_kkt_class(typeof(f)), f, g, x, k)
@inline _kkt_leaf(::_KktCustom, f, g, x, k) = _kkt_custom_zero(f, g, x, k)
@inline _kkt_leaf(::Union{_KktScore, _KktNone}, f, g, x, k) = _kkt_violation(f, g, x, k)
# A function defined outside the package has its three-argument screen at a zero coordinate or group, also when it defines
# a four-argument method of its own: that method is not a score that homogeneity carries through the wrapper. At a nonzero
# one the call is refused, and the method for the scaled function is the one to define.
_kkt_custom_zero(f, g, x::AbstractArray, i) = iszero(x[i]) ? kkt_violation(f, g, i) : _kkt_scaled_custom_error(f, g, i)
function _kkt_custom_zero(f, g, x::BlockIterate, k)
  G = x.groups[k]
  all(j -> iszero(x[j]), G) ? maximum(i -> kkt_violation(f, g, i), G) : _kkt_scaled_custom_error(f, g, first(G))
end
function _kkt_custom_zero(f, g, x::SymmetricSparseIterate, k)
  r, c = tril_cartesian(x.p, k)
  i = (c - 1) * x.p + r
  iszero(_getstored(x, k)) ? kkt_violation(f, g, i) : _kkt_scaled_custom_error(f, g, i)
end
@noinline function _kkt_scaled_custom_error(f, g, i)
  hasmethod(kkt_violation, Tuple{typeof(f), typeof(g), typeof(i)}) || throw(MethodError(kkt_violation, (f, g, i)))
  name = nameof(typeof(f))
  throw(ArgumentError("kkt_violation(f, g, x, k) of a scaled $name at a nonzero coordinate or group is not defined: its " *
    "three-argument method is the screen of a coordinate assumed zero. Define " *
    "kkt_violation(f::Scaled{<:$name}, g::AbstractArray, x::AbstractArray, k::Integer), and a method for BlockIterate " *
    "and SymmetricSparseIterate where they are used"))
end
# A chain whose product of factors is not a normal number (_chained), scored through _kkt_resolve. Where there is nothing
# to resolve to, a zero coordinate or group is scored by _kkt_unresolved, and a nonzero one is refused; a function defined
# outside the package keeps its own semantics.
function _kkt_chain(s::Scaled, g, x, k)
  f, a = _scaled_chain(s)
  _chain_finite(a) || return _kkt_chain_infinite(s, f, a, g, x, k)
  h = _kkt_resolve(f, a)
  h === nothing || return _kkt_apply(h, g, x, k)
  r = _kkt_unresolved(f, a, g, x, k)
  _kkt_class(typeof(f)) isa _KktScore && _kkt_nonzero_coord(f, x, k) && _kkt_chain_error(s, f)
  r
end
# The value at a zero coordinate or group of a chain that has nothing to resolve to (a nonzero one is refused by the caller
# after this value is formed). NormL1, NormL1Nonneg and ElasticNet score the entry with the coefficient c of _chain_coefficient (_KktChain),
# GroupNormL2 and GroupNormLinf the group with the coefficient of the group (_KktChainLevel), in the type of the arguments and the factors,
# with g read in that type. Neither divides g by a. The other functions that this package scores, and a function defined outside it, score the screen
# of the function at g divided by the product (kept as an exponent), multiplied by it, which does not recover a g that is
# out of range after the division.
function _kkt_unresolved(f, a, g, x, k)
  cls = _kkt_class(typeof(f))
  ge = _ExpScaledArray(g, _scale_inverse(a))
  _apply_scale(a, cls isa _KktCustom ? _kkt_custom_zero(f, ge, x, k) : _kkt_unfolded(f, ge, x, k))
end
_kkt_unresolved(f::Union{NormL1, NormL1Nonneg, ElasticNet}, a::_ProjectionScale, g, x, k) = _kkt_apply(_KktChain(f, a), g, x, k)
function _kkt_unresolved(f::_GroupNorm, a::_ProjectionScale, g, x, k)
  W = promote_type(float(promote_type(eltype(g), eltype(x))), _kkt_factor_type(a))
  # the validation and the entry in no group read g through a lazy copy whose entries are converted to W
  _level_score(f, _KktChainLevel(f.λ0, a), _ScaledArray{W}(g, one(W)), x, k)
end
# The product of the factors of a chain is finite (it is infinite or NaN when a retained chain holds an infinite factor, which
# a direct call of the parameterized constructor Scaled{F, T} builds; Scaled(f, α) merges an infinite factor).
_chain_finite(a::_ProjectionScale) = isfinite(a.m)
_chain_finite(a::Real) = isfinite(a)
# A chain with an infinite factor is refused at a nonzero coordinate or group, as an infinite factor of one wrapper is. Its
# value at a zero coordinate or group is outside the contract (see the scaling guide) and is computed here without a guarantee:
# by the screen of the function at g for a smooth function or an indicator, of the function with its folded levels, or of the
# function at g divided by the product, multiplied by it.
@noinline function _kkt_chain_infinite(s::Scaled, f, a, g, x, k)
  cls = _kkt_class(typeof(f))
  ge = _ExpScaledArray(g, _scale_inverse(a))
  r = if cls isa _KktCustom
    _apply_scale(a, _kkt_custom_zero(f, ge, x, k))
  elseif issmooth(typeof(f)) || _isindicator(f)
    _kkt_unfolded(f, g, x, k)
  else
    h = _fold_level(f, a)
    h === nothing ? _apply_scale(a, _kkt_unfolded(f, ge, x, k)) : _kkt_unfolded(h, g, x, k)
  end
  cls isa _KktScore && _kkt_nonzero_coord(f, x, k) && _kkt_infinite_error(s)
  r
end
# The value at a zero coordinate or group of the function f inside a chain, scored at its argument g as it stands: the screen
# of the three-argument form of the scalar functions (at the coordinate, at each coordinate of a group of a BlockIterate,
# at the lower position of a symmetric coordinate); for the other functions their own method. NormL1, NormL1Nonneg, ElasticNet,
# GroupNormL2 and GroupNormLinf under a finite chain whose product is a _ProjectionScale do not use it (_kkt_unresolved); under a
# chain with an infinite factor they do.
const _KktScalar = Union{NormL1, NormL1Nonneg, ElasticNet, ZeroFunction, LinearFunction, SqrNormL2}
_kkt_unfolded(f, g, x, k) = _kkt_violation(f, g, x, k)
_kkt_unfolded(f::_KktScalar, g, x::AbstractArray, i) = kkt_violation(f, g, i)
_kkt_unfolded(f::_KktScalar, g, x::BlockIterate, k) = maximum(i -> kkt_violation(f, g, i), x.groups[k])
function _kkt_unfolded(f::_KktScalar, g, x::SymmetricSparseIterate, k)
  r, c = tril_cartesian(x.p, k)
  kkt_violation(f, g, (c - 1) * x.p + r)
end
for X in (:AbstractArray, :BlockIterate)
  @eval @inline _kkt_violation(s::Scaled, g::AbstractArray, x::$X, k::Integer) = _kkt_scaled(s, g, x, k)
end
# The public four-argument methods of a Scaled: the checks of the methods for any function (interface.jl) and then the route
# above, marked for inlining. Inlining permits the compiler to hoist loop-invariant state for the unweighted cases; weighted
# coefficients remain entry-dependent. A user overload with a narrower Scaled leaf type and the same owner signature takes
# precedence; provide the owner overloads listed in the extension guide.
@inline kkt_violation(s::Scaled, g::AbstractArray, x::AbstractArray, i::Integer) =
  (_check_one_based(g, x); eltype(g) <: Real || _real_error(:kkt_violation); eltype(x) <: Real || _real_error(:kkt_violation);
   _check_kkt_coordinate(g, x, i); length(g) == length(x) || _gradient_length_error(g, x); _kkt_scaled(s, g, x, i))
@inline kkt_violation(s::Scaled, g::AbstractArray, x::BlockIterate, k::Integer) =
  (_check_one_based(g); eltype(g) <: Real || _real_error(:kkt_violation); length(g) == length(x) || _gradient_length_error(g, x);
   _kkt_scaled(s, g, x, k))

# The full-point form. v is validated before it is wrapped, since the lazy array reports one-based axes for any parent.
function kkt_violation(s::Scaled, v::AbstractArray, x::AbstractArray)
  _kkt_class(typeof(s)) isa _KktScore || throw(MethodError(kkt_violation, (s, v, x)))
  _check_one_based(v, x)
  eltype(v) <: Real || _real_error(:kkt_violation)
  eltype(x) <: Real || _real_error(:kkt_violation)
  length(v) == length(x) || _gradient_length_error(v, x)
  _chained(s) && return _kkt_chain_point(s, v, x)
  isinf(s.α) && _kkt_infinite_error(s)
  s.α * kkt_violation(s.f, _ScaledArray(v, inv(s.α)), x)
end
function _kkt_chain_point(s::Scaled, v, x)
  f, a = _scaled_chain(s)
  _chain_finite(a) || _kkt_infinite_error(s)
  h = _kkt_resolve(f, a)
  h === nothing && _kkt_chain_error(s, f)
  _kkt_full(h, v, x)
end
kkt_violation(s::Scaled, v::AbstractArray, x::SymmetricSparseIterate) =
  _kkt_class(typeof(s)) isa _KktScore ? _kkt_symmetric_scaled_error(s, _kkt_innermost(s)) : throw(MethodError(kkt_violation, (s, v, x)))
# the refusal names the function under the wrappers when the function has no coordinate form to advise (NormLinf and
# GroupNormLinf)
_kkt_symmetric_scaled_error(s, f) = _kkt_symmetric_point_error(s)
_kkt_symmetric_scaled_error(s, f::NormLinf) = _normlinf_symmetric_error(f)
_kkt_symmetric_scaled_error(s, f::GroupNormLinf) = _linf_symmetric_error(f)
_kkt_symmetric_scaled_error(s, f::_SmoothKktFunction) = _smooth_kkt_symmetric_error(f)
restrict(s::Scaled, S::AbstractVector{<:Integer}) = Scaled(restrict(s.f, S), s.α)

# The parts of a scaled function inside another wrapper: those of the function under the chain,
# times the product of its factors, which the scale still to apply multiplies further.
_has_value_parts(::Type{<:Scaled{F}}) where {F} = _has_value_parts(F)
_value_parts(s::Scaled, x, v) = _value_parts_scaled(s, x, v, true)
function _value_parts_scaled(s::Scaled, x, v, α)
  f, a = _scaled_chain(s)
  isfinite(a.m) && !iszero(a.m) || return nothing
  vf = _value_unchecked(f, x)
  vf isa AbstractFloat || return nothing
  p = isfinite(vf) && abs(vf) >= floatmin(vf) ? _frexp(vf) :
    _scaled_parts(f, _scale_times(α, a), x, vf, nothing)
  p === nothing && return nothing
  P = promote_type(typeof(a.m), typeof(p[1]))
  m, e = _frexp(convert(P, a.m) * convert(P, p[1]))
  (m, e + a.e + p[2])
end
# the product of a factor and a chain's product, as a factor
function _scale_times(α, a::_ProjectionScale)
  ma, ea = _projection_scale_parts(α)
  P = promote_type(typeof(float(ma)), typeof(a.m))
  m, e = _frexp(convert(P, ma) * convert(P, a.m))
  _ProjectionScale(m, Int(ea) + a.e + e)
end
# A zero value of s is a zero value of s.f, times the factor (|α f| < 2^(E + e)), or the product of
# the factor and a value of s.f that is nonzero: Scaled's own trigger then fires (the product is
# small while the value is not zero) and the result is α v rounded once, or the parts of a subnormal
# v rounded once, so that the zero is below δ. E is the larger of the two.
function _zero_value_exponent(s::Scaled, x)
  E = _zero_value_exponent(s.f, x)
  E === nothing && return nothing
  U = _underflow_exponent(_valtype(x), 1)
  E === _EXACT_ZERO ? U : max(E + _upper_exponent(s.α), U)
end
_parameter_alias(s::Scaled, out) = _parameter_alias(s.f, out)
# The functions that a stored infinite level fixes, for a positive finite factor and step. The infinite level
# makes the map the constant map of its domain, whatever positive finite number the factors and the step
# are, so the call needs no factor and no step folded into a finite level: `_hard_function(f)` is f without the
# factors of its blocks and with the finite levels that the infinite level makes irrelevant removed, or
# `nothing` when some block is not of this kind, and the call keeps the route that it had.
_hard_scale(a::Real) = isfinite(a) && a > 0
_hard_scale(a::_ProjectionScale) = isfinite(a.m) && a.m > 0
_steps_hard(γ) = all(g -> g isa Real && isfinite(g) && g > 0, γ)
# a stored infinite level in the function tree, read from the parameters as stored
_has_infinite_level(::AbstractFunction) = false
_has_infinite_level(f::NormL1) = isinf(f.λ0)
_has_infinite_level(f::NormL1Nonneg) = isinf(f.λ0)
_has_infinite_level(f::ElasticNet) = isinf(f.λ1) || isinf(f.λ2)
_has_infinite_level(f::Scaled) = _has_infinite_level(f.f)
_hard_function(f) = _hard(f)
_hard(_) = nothing
_hard(f::ZeroFunction) = f
_hard(f::NormL1) = isinf(f.λ0) ? f : nothing
_hard(f::NormL1Nonneg) = isinf(f.λ0) ? f : nothing
# Two levels. An infinite ridge level fixes every entry (#908), and an infinite ℓ1 level every entry with a
# positive weight. The finite level is then removed (set to zero) when the infinite one fixes every entry, or
# when it is zero already; otherwise (an infinite ℓ1 level, a positive finite ridge level and a zero weight)
# the ridge acts with the step on the entries with a zero weight, and the call keeps its route.
function _hard(f::ElasticNet)
  (isinf(f.λ1) || isinf(f.λ2)) || return nothing
  zeroed = (isinf(f.λ1) || iszero(f.λ1)) && (isinf(f.λ2) || iszero(f.λ2))
  (zeroed || isinf(f.λ2) || f.w === nothing || all(>(0), f.w)) || return nothing
  ElasticNet(isinf(f.λ1) ? f.λ1 : zero(f.λ1), isinf(f.λ2) ? f.λ2 : zero(f.λ2), f.w)
end
# a factor is irrelevant when it is a positive finite number; any other factor keeps the earlier route
function _hard(f::Scaled)
  c = f.α
  (c isa Real && isfinite(c) && c > 0) ? _hard_function(f.f) : nothing
end

####################################
#  Lipschitz constants of the gradients
####################################

lipschitz(s::Scaled) = _chained(s) ? _apply_scale(_scaled_chain(s)[2], lipschitz(_scaled_chain(s)[1])) : abs(s.α) * lipschitz(s.f)
# A step is needed only for conditional values of nested box constraints.
_checkwrite(op::O, f, y, x, γ) where {O} = (_checkreal(f, γ); _checkwrite(op, f, y, x))
function _checkwrite(op::O, s::Scaled, y, x, γ::Real) where {O}
  _checkreal(s, y); _checkreal(s, x)
  if _nested_route(s, γ)
    g, step, _ = _nested_step(s, γ)
    return _checkwrite(op, g, y, x, step)
  end
  _checkwrite(op, s.f, y, x, s.α * γ)
end
_checkwrite(op::O, s::Scaled, y, x, γ::AbstractArray) where {O} = (_checkreal(s, y); _checkreal(s, x); _checkreal(s, γ); _checkwrite(op, s.f, y, x, _ScaledArray(γ, s.α)))
