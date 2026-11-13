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

The nonnegative multiple `α*f`. Multiplication `α*f` and `f*α` construct it.
Traits and mathematical oracles forward to the wrapped function. Zero scales
simplify only when the original data and parameters permit it; invalid or
non-finite parameters keep the frozen function's validation policy.

Nested positive finite factors retain their exponents when an intermediate
product would overflow or underflow. NormL1 can fold a scale into its level;
SqrNormL2 supplies exponent-separated values. Ordinary calls preserve their
original multiply order. Combined oracles return the value at the stored
result using its model precision. Exact input/output aliasing is supported.
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
# finite levels give the zero function, or, with an infinite weight, zero levels that keep its fixed zeros
function _zero_scaled(f::ElasticNet)
  isfinite(f.λ1) && isfinite(f.λ2) || _zero_scale_error(f)
  _finite_parameter(f.w) ? ZeroFunction() : ElasticNet(zero(f.λ1), zero(f.λ2), f.w)
end
_zero_scaled(f::SqrNormL2) = _finite_zero(f, f.λ, f.y)
_zero_scaled(f::QuadraticFunction) = _finite_zero(f, f.A, f.b, f.c)
_zero_scaled(f::LeastSquaresLoss) = _finite_zero(f, f.X, f.Y)
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
# Nested wrappers also arise with ordinary products (a block of a conjugate keeps its own scale in
# a wrapper of its own), and those keep their recursive oracles. Only a chain whose product is
# not a normal number takes the methods that combine the factors.
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
# The recoveries read the entries of the argument: a symmetric iterate is read as the matrix that it represents, which
# counts an off-diagonal coordinate at both of its positions
@noinline _scaled_value_recover(f, α, x::SymmetricSparseIterate, v, r, out = nothing) =
  _scaled_value_recover(f, α, matrixview(x), v, r, out)
@noinline function _scaled_value_recover(f, α, x, v, r, out = nothing)
  g = _fold_level(f, _value_fold_scale(α))
  g === nothing || return convert(typeof(r), _value_unchecked(g, x))
  a, d = _partial_plan(α, v)
  if d != 0
    z = _scaled_value_partial(f, a, d, x, v, r)
    z === nothing || return z
    # no level that is a normal number gives a finite normal leaf value (levels of very different
    # size, say): the fold into a subnormal level, which has fewer bits but may still hold the value
    g = _fold_level(f, α)
    g === nothing || return convert(typeof(r), _value_unchecked(g, x))
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
# The value α v of the function f with the scale α at the unscaled value v.
@inline function _scaled_value_by(f, α, x, v, recover_inner_small = true, out = nothing)
  r = _apply_scale(α, v)
  if (!isfinite(r) && _scale_below_one(α)) || (_scaled_small(r) && !iszero(v)) ||
     (recover_inner_small && !_zero_value_exact(f) && _scaled_small(v) && _scale_above_one(α))
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
_host_entries(x::AbstractArray) = _is_device_array(x) ? Array(x) : x
_host_entries(::Nothing) = nothing
# an entry with the weight w under the level λ is penalized (an infinite weight at any level)
@inline _penalized(λ, w) = w > 0 && (λ > 0 || isinf(w))
function _exactly_zero_at(f::NormL1, x::AbstractArray)
  h, w = _host_entries(x), _host_entries(f.w)
  w === nothing || length(w) == length(h) || return false
  λ = f.λ0
  for (j, i) in enumerate(eachindex(h))
    wi = w === nothing ? one(λ) : w[j]
    _penalized(λ, wi) && !iszero(h[i]) && return false
  end
  true
end
# a positive ridge level penalizes every entry (an infinite one is the indicator of {0}, #908); a zero
# one leaves the entries that the ℓ1 part penalizes, as for NormL1
_exactly_zero_at(f::ElasticNet, x::AbstractArray) =
  f.λ2 > 0 ? _all_zero(_host_entries(x)) : _exactly_zero_at(f.l1, x)

# The recovery of α f(x) for an ElasticNet f. First the fold of the factor into both levels and the rule of an infinite
# factor at a point where f is 0, as for every function with levels. Then, for a finite positive factor, the sum of
# the two parts, α λ1 Σ wᵢ|xᵢ| from Scaled(NormL1(λ1, w), α) and α λ2/2 Σ xᵢ² from Scaled(SqrNormL2(λ2), α) (the
# indicator value for an infinite λ2, which a positive finite factor leaves unchanged), each recovered as its family
# recovers it (folds, partial folds, value parts), and added once: the two levels need not fold by one factor
# (ElasticNet(2^-1000, 2^1000) under the factor 2^-600 at [2^300] has the ℓ1 part 2^-1300 and the value 2^999).
# Any other factor keeps the product r.
@noinline function _scaled_value_recover(f::ElasticNet, α, x::AbstractArray, v, r, out = nothing)
  g = _fold_level(f, _value_fold_scale(α))
  g === nothing || return convert(typeof(r), _value_unchecked(g, x))
  if isnan(r) && !isnan(v)
    z = _scaled_nan_value(f, α, x, r)
    iszero(z) && return z
  end
  _elastic_finite_scale(α) || return r
  l = _scaled_value_by(f.l1, α, x, _value_unchecked(f.l1, x))
  λ = f.λ2
  q = iszero(λ) ? zero(l) : isinf(λ) ? _elastic_ridge_value(λ, x) :
      (h = SqrNormL2(λ); _scaled_value_by(h, α, x, _value_unchecked(h, x)))
  convert(typeof(r), l + q)
end
_elastic_finite_scale(α::Real) = isfinite(α) && α > 0
_elastic_finite_scale(α::_ProjectionScale) = isfinite(α.m) && !iszero(α.m)
_elastic_finite_scale(α) = false

# The scaled-value mechanism (R1, "Wrappers"). A function with `_has_value_parts` returns its
# value at x as (m, e), the value m 2^e with a finite significand m, or `nothing` to keep the
# ordinary result (`_value_parts`, defaults in utils.jl). Only host arrays take it; a NaN value
# stays NaN, and a normal value is its own significand and exponent, so the function is asked
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
  elseif isinf(v)
    _infinite_value_exact(f, x) && return false
  elseif isnan(v) || abs(v) >= floatmin(v)
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
# a symmetric iterate gradient: every coordinate that is stored, which are the coordinates of the symmetric gradient
# that are not zero
_scale!(g::SymmetricSparseIterate, α) = (v = activevalues(g); v .*= α; g)
_scale!(g::SymmetricSparseIterate, α::_ProjectionScale) = (v = activevalues(g); v .= _apply_scale.(α, v); g)
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
# - Otherwise the exponent of the factors goes into the levels, with the step reduced to its
#   significand, for the functions that fold a scale into their levels (`_fold_level`): the
#   products of level and step are those of the plain chain, formed in range. The leaf then
#   never sees a step or a level that is not representable while their product is.
# - A function without levels to fold gets the step as rounded.
# The factor is the one that takes the value of the returned function to the value of the chain.
# Not marked `@noinline`: the result is a union of tuples (folded or plain function, and the factor or
# `nothing`), which a call that is not inlined must box (144 bytes per call). The callers that sit
# in a loop over `prox_block` are the ones that are not inlined (`_prox_block_nested`).
function _nested_step(s::Scaled, γ::Real)
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
  g = _fold_level_step(f, _ProjectionScale(ma, Int(ea + eγ)))
  # g carries the levels without the ridge that may have been dropped: it is not the chain, and
  # its value is not the value of the chain (`nothing` marks this)
  g === nothing ? (f, step, a) : (g, convert(typeof(step), mγ), nothing)
end

_prox_unchecked!(y, s::Scaled, x, γ::Real) = _nested_route(s, γ) ? _prox_chain!(y, s, x, γ) :
  (_checkwrite(prox!, s, y, x); _prox_unchecked!(y, s.f, x, _scaled_step(s, γ)))
@noinline function _prox_chain!(y, s, x, γ::Real)
  _checkwrite(prox!, s, y, x)
  g, step, _ = _nested_step(s, γ)
  _prox_unchecked!(y, g, x, step)
end
value_and_prox!(y, s::Scaled, x, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, s, y, x, γ); _value_and_prox_unchecked!(y, s, x, γ))
_value_and_prox_unchecked!(y, s::Scaled, x, γ::Real) = _nested_route(s, γ) ? _value_and_prox_chain!(y, s, x, γ) :
  (_checkwrite(value_and_prox!, s, y, x); v = _value_and_prox_unchecked!(y, s.f, x, _scaled_step(s, γ)); _value_at(_scaled_value(s, y, v, isinf(s.α)), y))
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
  (eltype(g) <: Real || _real_error(:kkt_violation); _chained(s) ? _nested_kkt((f, h) -> kkt_violation(f, h, i), s, g) :
   iszero(s.α) ? abs(g[i]) : s.α * kkt_violation(s.f, _ScaledArray(g, inv(s.α)), i))
for X in (:AbstractArray, :BlockIterate)
  @eval _kkt_violation(s::Scaled, g::AbstractArray, x::$X, k::Integer) =
    _chained(s) ? _nested_kkt((f, h) -> _kkt_violation(f, h, x, k), s, g) :
    iszero(s.α) ? _kkt_violation(ZeroFunction(), g, x, k) : s.α * _kkt_violation(s.f, _ScaledArray(g, inv(s.α)), x, k)
end
# The violation of a nested chain: that of the function with the factors folded into its levels, or
# of the function at the gradient divided by the product of the factors, and multiplied by it.
function _nested_kkt(kkt, s::Scaled, g)
  f, a = _scaled_chain(s)
  # a smooth function has the violation |g|, whatever the scale, and an indicator is its own scaling
  (issmooth(typeof(f)) || _isindicator(f)) && return kkt(f, g)
  h = _fold_level(f, a)
  h === nothing ? _apply_scale(a, kkt(f, _ExpScaledArray(g, _scale_inverse(a)))) : kkt(h, g)
end
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
_infinite_value_exact(s::Scaled, x) = _infinite_value_exact(s.f, x)
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
_has_infinite_level(f::ElasticNet) = isinf(f.λ1) || isinf(f.λ2)
_has_infinite_level(f::Scaled) = _has_infinite_level(f.f)
_hard_function(f) = _hard(f)
_hard(_) = nothing
_hard(f::ZeroFunction) = f
_hard(f::NormL1) = isinf(f.λ0) ? f : nothing
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
