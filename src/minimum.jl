####################################
#
#  Pointwise minima
#
####################################

"""
    PointwiseMinimum(f1, f2, ...)

The function `g(x) = minᵢ fᵢ(x)` on arrays, for one or more proximable
functions. With `zᵢ = prox_{γfᵢ}(x)`, its proximal operator returns the candidate
with the smallest `fᵢ(zᵢ) + ‖zᵢ - x‖²/(2γ)`. Ties go to the first function;
a `NaN` score never beats a number. If all scores are `NaN`, the first
candidate is returned and [`value_and_prox!`](@ref) returns `NaN`.
Steps are positive scalars; arrays of steps and tuple inputs are unsupported.
The minimum is generally nonconvex, and supplies neither smooth nor separable
oracles. A single convex component retains its convexity.

For indicator functions this projects onto the union of their sets, comparing
distances without squaring. For example,
`PointwiseMinimum(IndIsotonic(), IndIsotonic(decreasing = true))` fits a
monotone vector with unknown direction. The package's indicators return zero
from `value_and_prox!` without checking membership. The same convention holds
for this union; it is not a feasibility certificate. An inexact projection,
such as [`IndIntersection`](@ref), can win a comparison it should lose.

Evaluating `g(x)` requires every component's value oracle; for example,
`Conjugate(ElasticNet(0.5, 0.2))` has no standalone value oracle. The proximal operator
uses each component's `value_and_prox!`, and is exact when the component
proximal operators and reported values are exact. Approximate projections,
partial decompositions and cancellation in [`Conjugate`](@ref) values can
change the selected branch. The returned scalar is promoted across all
candidate values.

Each call checks all component parameters before writing the output. Two
owned buffers follow the output's element type and shape, and allow the
output to alias the input. Warmed calls on fixed-shape CPU arrays allocate
nothing when the components allocate nothing. The cost is one proximal
operator per component; do not share the function between threads.
"""
struct PointwiseMinimum{F<:Tuple} <: AbstractFunction
  fs::F
  best::Scratch
  current::Scratch

  function PointwiseMinimum{F}(fs::F) where {F<:Tuple}
    isempty(fs) && throw(ArgumentError("PointwiseMinimum requires at least one function"))
    all(isproximable, fs) || throw(ArgumentError("PointwiseMinimum requires proximable functions"))
    new{F}(fs, Scratch(), Scratch())
  end
end
PointwiseMinimum(fs::AbstractFunction...) = PointwiseMinimum{typeof(fs)}(fs)

isproximable(::Type{<:PointwiseMinimum}) = true
isconvex(::Type{<:PointwiseMinimum{F}}) where {F} = length(fieldtypes(F)) == 1 && isconvex(first(fieldtypes(F)))
_isindicator(::Type{<:PointwiseMinimum{F}}) where {F} = all(_isindicator, fieldtypes(F))
# The proximal path asks the instance: Julia 1.10 does not fold `all` over
# `fieldtypes` and allocates, while recursion over the tuple folds.
_isindicator(g::PointwiseMinimum) = _all_indicators(g.fs)
_all_indicators(::Tuple{}) = true
_all_indicators(fs::Tuple) = _isindicator(first(fs)) && _all_indicators(Base.tail(fs))
_checkparams(g::PointwiseMinimum, x) = (map(f -> _checkparams(f, x), g.fs); true)

(g::PointwiseMinimum)(x::AbstractArray) =
  (_check_value_arrays(g, x); _value_checked(g, x))
function _value_unchecked(g::PointwiseMinimum, x::AbstractArray)
  _checkparams(g, x)
  _value_at(minimum(map(f -> _value_unchecked(f, x), g.fs)), x)
end

function _checkwrite(op::O, g::PointwiseMinimum, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(g, y); _checkreal(g, x)
  _checksize(y, x)
  _checkparams(g, x)
  b = _array_buffer!(g.current, eltype(y), size(y))
  map(g.fs) do f
    _checkwrite(op, f, b, x)
    op === prox! && _checkwrite(value_and_prox!, f, b, x)
  end
  true
end

# Represent distances by a mantissa and exponent, even when subtraction or
# the norm itself exceeds the floating-point range.
function _minimum_distance(x::AbstractArray, p::AbstractArray)
  r = norm_diff(x, p)
  isfinite(r) && return frexp(r)
  isnan(r) && return (r, 0)
  scale = zero(r)
  @inbounds for i in eachindex(x, p)
    scale = max(scale, abs(x[i]), abs(p[i]))
  end
  isfinite(scale) || return (r, 0)
  _, e = frexp(scale)
  a = ldexp(one(scale), -e)
  m, k = frexp(norm_diff(_ScaledArray(x, a), _ScaledArray(p, a)))
  (m, k + e)
end

# Keep the value and quadratic term separate, so a common offset cannot erase
# a smaller distance difference. Exponents avoid infinite and zero ties.
function _minimum_score(v::Real, x, p, γ::Real, indicator::Bool)
  r, re = _minimum_distance(x, p)
  (isnan(v) || isnan(r)) && return (oftype(float(v), NaN), zero(r), 0)
  indicator && return (zero(v), r, re)
  if !isfinite(γ)
    energy = _envelope_energy(x, p, γ)
    !isfinite(energy) && return (v + energy, zero(r), 0)
    m, e = frexp(energy)
    return (v, m, e)
  end
  (!isfinite(v) || !isfinite(r)) && return (v + r, zero(r), 0)
  iszero(r) && return (v, r, 0)
  gm, ge = frexp(float(γ))
  m, e = frexp(abs2(r) / (2gm))
  (v, m, e + 2re - ge)
end

@inline function _minimum_pair_less(a, b)
  am, ae = a
  bm, be = b
  isnan(am) && return false
  isnan(bm) && return true
  if !isfinite(am) || !isfinite(bm) || iszero(am) || iszero(bm) || signbit(am) != signbit(bm)
    return am < bm
  end
  ae == be ? am < bm : (am > 0 ? ae < be : ae > be)
end

# Sum two scaled numbers with a rounding residual. If their exponents are
# far apart, retaining the smaller term directly also avoids its underflow.
@inline function _minimum_add(a, b)
  am, ae = a
  bm, be = b
  z = (zero(am), 0)
  iszero(am) && return (b, z)
  iszero(bm) && return (a, z)
  if ae < be
    am, bm, ae, be = bm, am, be, ae
  end
  ae - be > precision(am) + 2 && return ((am, ae), (bm, be))
  bm = ldexp(bm, be - ae)
  s = am + bm
  t = s - am
  residual = (am - (s - t)) + (bm - t)
  sm, se = frexp(s)
  rm, re = frexp(residual)
  ((sm, se + ae), (rm, re + ae))
end

@inline function _minimum_less(a, b)
  av, am, ae = a
  bv, bm, be = b
  av, bv, am, bm = promote(float(av), float(bv), am, bm)
  isnan(av) && return false
  isnan(bv) && return true
  (!isfinite(av) || !isfinite(bv)) && return av < bv
  ah, al = _minimum_add(frexp(av), (am, ae))
  bh, bl = _minimum_add(frexp(bv), (bm, be))
  _minimum_pair_less(ah, bh) || (!_minimum_pair_less(bh, ah) && _minimum_pair_less(al, bl))
end

_minimum_candidates!(best, current, x, γ, indicator, v, score, ::Tuple{}) = (v, score)
function _minimum_candidates!(best, current, x, γ, indicator, v, score, fs::Tuple)
  w = _value_and_prox_unchecked!(current, first(fs), x, γ)
  next = _minimum_score(w, x, current, γ, indicator)
  v, w = promote(v, w)
  if _minimum_less(next, score)
    copyto!(best, current)
    v, score = w, next
  end
  _minimum_candidates!(best, current, x, γ, indicator, v, score, Base.tail(fs))
end

value_and_prox!(y::AbstractArray, g::PointwiseMinimum, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, g, y, x, γ); _value_and_prox_unchecked!(y, g, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, g::PointwiseMinimum, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, g, y, x)
  best = _array_buffer!(g.best, eltype(y), size(y))
  current = _array_buffer!(g.current, eltype(y), size(y))
  indicator = _isindicator(g)
  v = _value_and_prox_unchecked!(best, first(g.fs), x, γ)
  score = _minimum_score(v, x, best, γ, indicator)
  v, score = _minimum_candidates!(best, current, x, γ, indicator, v, score, Base.tail(g.fs))
  copyto!(y, best)
  _value_at(isnan(first(score)) ? oftype(float(v), NaN) : (indicator ? zero(v) : v), y)
end

prox!(y::AbstractArray, g::PointwiseMinimum, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, g, y, x, γ); _prox_unchecked!(y, g, x, γ))
function _prox_unchecked!(y::AbstractArray, g::PointwiseMinimum, x::AbstractArray, γ::Real)
  _checkwrite(prox!, g, y, x)
  _value_and_prox_unchecked!(y, g, x, γ)
  y
end

for op in (:prox!, :value_and_prox!)
  @eval $op(y::AbstractArray, g::PointwiseMinimum, x::AbstractArray, γ::AbstractArray) =
    (_checkreal(g, y); _checkreal(g, x); _checkreal(g, γ); throw(ArgumentError("PointwiseMinimum requires a scalar step")))
end

prox(g::PointwiseMinimum, x::AbstractArray, γ::AbstractArray) =
  (_checkreal(g, x); _checkreal(g, γ); throw(ArgumentError("PointwiseMinimum requires a scalar step")))

function _checkstep(op::O, g::PointwiseMinimum, y::AbstractArray, x::AbstractArray, γ::Real) where {O}
  b = _checkbuffer(g.current, eltype(y), size(y))
  _checkstep_candidates(g.fs, b, x, γ)
end
_checkstep_candidates(::Tuple{}, y, x, γ) = true
_checkstep_candidates(fs::Tuple, y, x, γ) =
  (_checkstep(value_and_prox!, first(fs), y, x, γ); _checkstep_candidates(Base.tail(fs), y, x, γ))
