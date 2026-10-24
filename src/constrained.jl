####################################
#
#  Separable functions with constraints
#
####################################

"""
    Constrained(f, C)

The function `f(x)` on the set `C`, and `Inf` outside it, for a convex separable
`f` and a set `C` that is

- a box: [`IndBox`](@ref), [`IndNonnegative`](@ref) or [`IndBallLinf`](@ref);
- a hyperplane or a halfspace: [`IndHyperplane`](@ref) or [`IndHalfspace`](@ref);
- a simplex, [`IndSimplex`](@ref): the box `x ≥ 0` with the hyperplane
  `Σᵢ xᵢ = r`, or with the halfspace `Σᵢ xᵢ ≤ r`;
- for a norm, an ℓ2 ball or sphere, [`IndBallL2`](@ref) or
  [`IndSphereL2`](@ref), described below.

On a box, the proximal operator clamps that of `f` entry by entry, so
`Constrained(f, box)` is separable, with [`prox_entry`](@ref) for coordinate
descent. On a hyperplane `⟨a, x⟩ = b`, the proximal operator is
`prox_{γf}(x - γνa)` for the multiplier `ν` at which it satisfies the
constraint, found by a search over the scalar `ν`; `f` may itself be
constrained to a box. Each step of the search is a pass over `x`. It starts
from the multiplier of the previous call, which the function keeps, with a
first step from the exact slope of the constraint where the scalar proximal
operators of `f` are piecewise linear, as for every function below. The calls
of a solver on slowly changing points then take three to five passes, against
7 to 16 from zero. Do not share one function object between threads. For
example:

- `Constrained(NormL1(λ), IndHyperplane(1, 0))` is the lasso penalty on
  coefficients that sum to zero, as in regression on compositional data;
- `Constrained(NormL1(λ), IndBox(0, u))` is the nonnegative lasso penalty with
  upper bounds;
- `Constrained(IndBox(0, 1), IndHyperplane(1, k))` is the indicator of the capped
  simplex `{x : 0 ≤ x ≤ 1, Σᵢ xᵢ = k}`;
- `Constrained(SqrNormL2(λ), IndSimplex(1))` is the ridge penalty on the
  probability simplex.

`f` must be convex: for a nonconvex penalty, clamping its proximal operator does
not give the constrained one. The exceptions are [`MCP`](@ref), [`SCAD`](@ref)
and [`NormL0`](@ref) on a box, whose scalar problems are solved exactly by
comparing the minimizers of their convex pieces with the ends of the interval;
the result is separable but not convex. When the set misses the domain of `f`,
the proximal operator throws an `ArgumentError`.
"""
function Constrained end

# f on a box: separable, with the clamped scalar proximal operators
struct BoxConstrained{F<:AbstractFunction, B<:IndBox} <: AbstractFunction
  f::F
  box::B

  BoxConstrained{F, B}(f::F, box::B) where {F<:AbstractFunction, B<:IndBox} =
    (_check_box_constrainable(f); new{F, B}(f, box))
end
BoxConstrained(f::AbstractFunction, box::IndBox) = BoxConstrained{typeof(f), typeof(box)}(f, box)

# f on a hyperplane or a halfspace; the search for the multiplier starts where
# the last one ended
struct LinearConstrained{F<:AbstractFunction, H<:Union{IndHyperplane, IndHalfspace}} <: AbstractFunction
  f::F
  set::H
  last::Base.RefValue{Float64}      # the multiplier of the last call
  slope::Base.RefValue{Float64}     # the slope of φ near it, from the last secant

  LinearConstrained{F, H}(f::F, set::H) where {F<:AbstractFunction, H<:Union{IndHyperplane, IndHalfspace}} =
    (_check_constrainable(f); new{F, H}(f, set, Ref(NaN), Ref(NaN)))
end
LinearConstrained(f::AbstractFunction, set::Union{IndHyperplane, IndHalfspace}) =
  LinearConstrained{typeof(f), typeof(set)}(f, set)

@noinline function _check_constrainable(f)
  isseparable(f) || throw(ArgumentError("Constrained needs a separable function, and $(nameof(typeof(f))) is not"))
  isconvex(f) || throw(ArgumentError("Constrained needs a convex function, and $(nameof(typeof(f))) is not"))
  nothing
end

# on a box, also the nonconvex penalties whose constrained scalar problems
# constrained_nonconvex.jl solves exactly; the multiplier search on a hyperplane
# needs the monotone scalar proximal operators of a convex function
@noinline function _check_box_constrainable(f)
  isseparable(f) || throw(ArgumentError("Constrained needs a separable function, and $(nameof(typeof(f))) is not"))
  isconvex(f) || _has_box_prox(f) ||
    throw(ArgumentError("Constrained needs a convex function, or MCP, SCAD or NormL0 on a box, and $(nameof(typeof(f))) is neither"))
  nothing
end
_has_box_prox(::AbstractFunction) = false

Constrained(f::AbstractFunction, C::IndBox) = BoxConstrained(f, C)
Constrained(f::AbstractFunction, ::IndNonnegative) = BoxConstrained(f, IndBox(0.0, Inf))
Constrained(f::AbstractFunction, C::Union{IndHyperplane, IndHalfspace}) = LinearConstrained(f, C)
Constrained(f::AbstractFunction, C::IndSimplex) =
  LinearConstrained(Constrained(f, IndNonnegative()), C.equality ? IndHyperplane(1, C.r) : IndHalfspace(1, C.r))
# a box goes under the linear constraint, where the function stays separable
Constrained(c::LinearConstrained, C::IndBox) = LinearConstrained(Constrained(c.f, C), c.set)
Constrained(c::LinearConstrained, C::IndNonnegative) = LinearConstrained(Constrained(c.f, C), c.set)
Constrained(::AbstractFunction, C::AbstractFunction) =
  throw(ArgumentError("Constrained takes a box, a hyperplane, a halfspace, a simplex, an ℓ2 ball or an ℓ2 sphere, not $(nameof(typeof(C)))"))

isproximable(::Type{<:Union{BoxConstrained, LinearConstrained}}) = true
isconvex(::Type{<:LinearConstrained}) = true
isconvex(::Type{<:BoxConstrained{F}}) where {F} = isconvex(F)
isseparable(::Type{<:BoxConstrained}) = true
_isindicator(::Type{<:BoxConstrained{F}}) where {F} = _isindicator(F)
_isindicator(::Type{<:LinearConstrained{F}}) where {F} = _isindicator(F)

####################################
#  on a box
####################################

(c::BoxConstrained)(x::AbstractArray) = (T = float(eltype(x)); iszero(c.box(x)) ? convert(T, c.f(x)) : T(Inf))

# the scalar problem is convex, so its minimizer on an interval is the nearest
# point to its unconstrained minimizer
Base.@propagate_inbounds prox_entry(c::BoxConstrained, v::Real, i::Integer, γ::Real) =
  prox_entry(c.box, prox_entry(c.f, v, i, γ), i, γ)

prox!(y::AbstractArray, c::BoxConstrained, x::AbstractArray, γ::Real) = _prox_sep!(y, c, x, γ)
function value_and_prox!(y::AbstractArray, c::BoxConstrained, x::AbstractArray, γ::Real)
  _prox_sep!(y, c, x, γ)
  T = float(eltype(y))
  _isindicator(c) ? zero(T) : convert(T, c.f(y))
end

_checkparams(c::BoxConstrained, x) = (_checkparams(c.f, x); _checkparams(c.box, x); true)
iszeropreserving(c::BoxConstrained) = _iszeropreserving(c.f) && iszeropreserving(c.box)

# whether a separable function is known to vanish at 0 and map 0 to 0
_iszeropreserving(f::Union{SeparableFunction, BoxConstrained}) = iszeropreserving(f)
_iszeropreserving(::Union{ZeroFunction, SqrNormL2{<:Real, Nothing}}) = true
_iszeropreserving(s::Scaled) = _iszeropreserving(s.f)
_iszeropreserving(::AbstractFunction) = false

# For a convex f with 0 in its domain, the proximal step of f from -gᵢ moves in
# the direction that improves on zero, if one does; the box must allow it.
function kkt_violation(c::BoxConstrained, g::AbstractArray, i::Integer)
  v = kkt_violation(c.f, g, i)
  v > 0 || return v
  d = prox_entry(c.f, -g[i], i, one(float(eltype(g))))
  (d > 0 ? _entry(c.box.u, i) > 0 : _entry(c.box.l, i) < 0) ? v : zero(v)
end

restrict(c::BoxConstrained, S::AbstractVector{<:Integer}) = BoxConstrained(restrict(c.f, S), restrict(c.box, S))

####################################
#  on a hyperplane or a halfspace
####################################

(c::LinearConstrained)(x::AbstractArray) = (T = float(eltype(x)); iszero(c.set(x)) ? convert(T, c.f(x)) : T(Inf))

_checkparams(c::LinearConstrained, x) = (_checkparams(c.f, x); _checkparams(c.set, x); true)

function prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::Real)
  _checksize(y, x)
  _checkparams(c, x)
  T = float(eltype(y))
  s = convert(T, γ)
  ν = _multiplier(c, x, s)
  f, a = c.f, c.set.a
  @inbounds for i in eachindex(IndexLinear(), y, x)
    y[i] = convert(eltype(y), prox_entry(f, x[i] - s * ν * convert(T, _entry(a, i)), i, s))
  end
  y
end

function value_and_prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::Real)
  prox!(y, c, x, γ)
  T = float(eltype(y))
  _isindicator(c) ? zero(T) : convert(T, c.f(y))
end

# φ(ν) = ⟨a, z⟩ - b at z = prox_{γf}(x - γνa), and Σᵢ |aᵢzᵢ| + |b|, the scale of
# its rounding error
function _constraint_gap(c::LinearConstrained, x::AbstractArray, ν::T, γ::T) where {T}
  f, a = c.f, c.set.a
  s, m = zero(T), zero(T)
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    aᵢ = convert(T, _entry(a, i))
    t = aᵢ * convert(T, prox_entry(f, x[i] - γ * ν * aᵢ, i, γ))
    s += t
    m += abs(t)
  end
  b = convert(T, c.set.b)
  s - b, m + abs(b)
end

# The same with the slope of φ, -γ Σᵢ aᵢ² Pᵢ', where Pᵢ' is the slope of the
# scalar proximal operator at its argument
function _constraint_gap_slope(c::LinearConstrained, x::AbstractArray, ν::T, γ::T) where {T}
  f, a = c.f, c.set.a
  s, m, d = zero(T), zero(T), zero(T)
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    aᵢ = convert(T, _entry(a, i))
    z, dz = _prox_entry_slope(f, x[i] - γ * ν * aᵢ, i, γ)
    t = aᵢ * convert(T, z)
    s += t
    m += abs(t)
    d += aᵢ * aᵢ * convert(T, dz)
  end
  b = convert(T, c.set.b)
  s - b, m + abs(b), -γ * d
end

const _MAX_BRACKET_STEPS = 64
const _MAX_SECANTS = 100

@noinline _infeasible(c::LinearConstrained) =
  throw(ArgumentError("the $(nameof(typeof(c.set))) misses the domain of $(nameof(typeof(c.f)))"))

# φ and the scale of its rounding error at ν, and its slope when the scalar
# proximal operators of f have known slopes, NaN otherwise
_gap_and_slope(c::LinearConstrained{F}, x::AbstractArray, ν::T, γ::T) where {F, T} =
  _gap_and_slope(c, x, ν, γ, Val(_has_prox_slope(F)))
_gap_and_slope(c::LinearConstrained, x::AbstractArray, ν::T, γ::T, ::Val{true}) where {T} =
  _constraint_gap_slope(c, x, ν, γ)
_gap_and_slope(c::LinearConstrained, x::AbstractArray, ν::T, γ::T, ::Val{false}) where {T} =
  ((φ, m) = _constraint_gap(c, x, ν, γ); (φ, m, T(NaN)))

# The multiplier of the constraint, a root of φ, which is nonincreasing because
# each prox_entry is nondecreasing in its argument; for a halfspace, 0 when the
# proximal point of f satisfies the constraint. The search starts from the
# multiplier of the last call, and from 0 at the first call or when that was
# an inactive halfspace.
function _multiplier(c::LinearConstrained, x::AbstractArray, γ::T) where {T}
  ν0 = convert(T, c.last[])
  warm = isfinite(ν0) && !(c.set isa IndHalfspace && !(ν0 > 0))
  ν = _search(c, x, γ, warm ? ν0 : zero(T), warm)
  c.last[] = ν
  ν
end

# The search from start. The first step is at least |φ(start)|/(γ‖a‖²), a lower
# bound on the distance to the root, as the slope of φ is at least -γ‖a‖². From
# the multiplier of the last call, it goes to the root of the line through
# (start, φ(start)) with the slope of φ there, exact where the scalar proximal
# operators of f have known slopes and otherwise that of the last call's
# secant; from a start on the linear piece of the root, this step ends the
# search. From zero it is the lower bound: a pass that computes the slope costs
# half a pass more, and saves passes only when the root lies near zero. Each
# further step is twice the step to the root of the secant through the last two
# points, which passes the root when that secant predicts it well, or twice the
# last step where φ is flat. Regula falsi with the Illinois rule then shrinks
# the bracket; it is exact once both ends lie on one linear piece of φ.
function _search(c::LinearConstrained, x::AbstractArray, γ::T, start::T, warm::Bool) where {T}
  φs, ms, dφ = warm ? _gap_and_slope(c, x, start, γ) : (_constraint_gap(c, x, start, γ)..., T(NaN))
  abs(φs) <= 2eps(T) * ms && return start
  if c.set isa IndHalfspace && φs < 0
    iszero(start) && return zero(T)                           # the constraint is inactive
    # the root lies below the start, where the constraint may be inactive
    φ0, m0 = _constraint_gap(c, x, zero(T), γ)
    (φ0 <= 0 || abs(φ0) <= 2eps(T) * m0) && return zero(T)
    return _refine(c, x, γ, zero(T), φ0, start, φs)
  end
  step = abs(φs) / (γ * convert(T, _normal_sqnorm(c.set, x)))
  σ = dφ < 0 ? dφ : (warm ? convert(T, c.slope[]) : T(NaN))
  σ < 0 && (step = max(step, abs(φs) / -σ))
  _bracket_and_refine(c, x, γ, start, φs, step)
end

function _bracket_and_refine(c::LinearConstrained, x::AbstractArray, γ::T, ν0::T, φ0::T, step::T) where {T}
  dir = φ0 > 0 ? one(T) : -one(T)
  lo, φlo = ν0, φ0                          # φlo has the sign of φ(ν₀), φhi the other
  hi, φhi = lo, φlo
  for k in 1:_MAX_BRACKET_STEPS
    hi = lo + dir * step
    φhi, mhi = _constraint_gap(c, x, hi, γ)
    if abs(φhi) <= 2eps(T) * mhi
      c.slope[] = (φhi - φlo) / (hi - lo)
      return hi
    end
    signbit(φhi) == signbit(φ0) || break
    k == _MAX_BRACKET_STEPS && _infeasible(c)
    s = step * φhi / (φlo - φhi)
    step = isfinite(s) && s > 0 ? min(2s, 64step) : 2step
    lo, φlo = hi, φhi
  end
  _refine(c, x, γ, lo, φlo, hi, φhi)
end

# regula falsi with the Illinois rule on the bracket [lo, hi], where φlo and φhi
# have opposite signs; keeps the slope of the last secant for the next call
function _refine(c::LinearConstrained, x::AbstractArray, γ::T, lo::T, φlo::T, hi::T, φhi::T) where {T}
  best, φbest = abs(φlo) < abs(φhi) ? (lo, φlo) : (hi, φhi)
  νp, φp = hi, φhi                          # the last point evaluated, for the slope
  side = 0                                  # which end the last step kept
  for _ in 1:_MAX_SECANTS
    ν = (lo * φhi - hi * φlo) / (φhi - φlo)
    φν, mν = _constraint_gap(c, x, ν, γ)
    ν != νp && (c.slope[] = (φν - φp) / (ν - νp))
    νp, φp = ν, φν
    abs(φν) < abs(φbest) && ((best, φbest) = (ν, φν))
    abs(φν) <= 2eps(T) * mν && return ν
    if signbit(φν) == signbit(φhi)
      hi, φhi = ν, φν
      side == 1 && (φlo /= 2)
      side = 1
    else
      lo, φlo = ν, φν
      side == -1 && (φhi /= 2)
      side = -1
    end
    abs(hi - lo) <= 2eps(T) * max(abs(lo), abs(hi)) && break
  end
  best
end

####################################
#  the slopes of scalar proximal operators
####################################

# Whether the scalar proximal operators of a separable function are piecewise
# linear, with _prox_entry_slope(f, v, i, γ) giving prox_entry(f, v, i, γ) and
# its slope at v, on either side at a kink. The value must be that of
# prox_entry, computed the same way.
_has_prox_slope(::Type{<:AbstractFunction}) = false

for F in (:ZeroFunction, :LinearFunction, :SqrNormL2, :NormL1, :NormL1Nonneg, :ElasticNet, :IndBox, :IndNonnegative)
  @eval _has_prox_slope(::Type{<:$F}) = true
end
_has_prox_slope(::Type{<:Scaled{F}}) where {F} = _has_prox_slope(F)
_has_prox_slope(::Type{<:Tilt{F}}) where {F} = _has_prox_slope(F)
_has_prox_slope(::Type{<:Translate{F}}) where {F} = _has_prox_slope(F)
_has_prox_slope(::Type{<:BoxConstrained{F}}) where {F} = _has_prox_slope(F)

@inline _slope(c::Bool, s::T) where {T} = ifelse(c, s, zero(T))

@inline _prox_entry_slope(f::ZeroFunction, v::T, i::Integer, γ::Real) where {T<:Real} =
  (prox_entry(f, v, i, γ), one(float(T)))
Base.@propagate_inbounds _prox_entry_slope(f::LinearFunction, v::T, i::Integer, γ::Real) where {T<:Real} =
  (prox_entry(f, v, i, γ), one(float(T)))
Base.@propagate_inbounds function _prox_entry_slope(f::SqrNormL2, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  prox_entry(f, v, i, γ), inv(one(S) + convert(S, γ * f.λ))
end
Base.@propagate_inbounds function _prox_entry_slope(f::NormL1, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  t = convert(S, γ * f.λ0 * _entry(f.w, i))
  shrink(v, t), _slope(abs(v) > t, one(S))
end
Base.@propagate_inbounds function _prox_entry_slope(f::NormL1Nonneg, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  t = convert(S, γ * f.λ0 * _entry(f.w, i))
  max(v - t, zero(S)), _slope(v > t, one(S))
end
Base.@propagate_inbounds function _prox_entry_slope(f::ElasticNet, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  t = convert(S, γ * f.λ1 * _entry(f.w, i))
  d = 1 + convert(S, γ * f.λ2)
  shrink(v, t) / d, _slope(abs(v) > t, inv(d))
end
Base.@propagate_inbounds function _prox_entry_slope(f::IndBox, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  l, u = convert(S, _entry(f.l, i)), convert(S, _entry(f.u, i))
  clamp(convert(S, v), l, u), _slope((v > l) & (v < u), one(S))
end
@inline _prox_entry_slope(::IndNonnegative, v::T, ::Integer, ::Real) where {T<:Real} =
  (max(float(v), zero(float(T))), _slope(v > 0, one(float(T))))
Base.@propagate_inbounds _prox_entry_slope(s::Scaled, v::Real, i::Integer, γ::Real) =
  _prox_entry_slope(s.f, v, i, s.α * γ)
Base.@propagate_inbounds _prox_entry_slope(t::Tilt, v::Real, i::Integer, γ::Real) =
  _prox_entry_slope(t.f, v - γ * t.c[i], i, γ)
Base.@propagate_inbounds function _prox_entry_slope(t::Translate, v::Real, i::Integer, γ::Real)
  z, dz = _prox_entry_slope(t.f, v - t.b[i], i, γ)
  t.b[i] + z, dz
end
Base.@propagate_inbounds function _prox_entry_slope(c::BoxConstrained, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  z, dz = _prox_entry_slope(c.f, v, i, γ)
  l, u = convert(S, _entry(c.box.l, i)), convert(S, _entry(c.box.u, i))
  clamp(convert(S, z), l, u), _slope((z > l) & (z < u), convert(S, dz))
end
