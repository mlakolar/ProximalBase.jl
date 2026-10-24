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
constraint, found by a search over the scalar `ν` that takes a few passes over
`x`; `f` may itself be constrained to a box. For example:

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

# f on a hyperplane or a halfspace
struct LinearConstrained{F<:AbstractFunction, H<:Union{IndHyperplane, IndHalfspace}} <: AbstractFunction
  f::F
  set::H

  LinearConstrained{F, H}(f::F, set::H) where {F<:AbstractFunction, H<:Union{IndHyperplane, IndHalfspace}} =
    (_check_constrainable(f); new{F, H}(f, set))
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

const _MAX_BRACKET_STEPS = 64
const _MAX_SECANTS = 100

@noinline _infeasible(c::LinearConstrained) =
  throw(ArgumentError("the $(nameof(typeof(c.set))) misses the domain of $(nameof(typeof(c.f)))"))

# The multiplier of the constraint, the root of φ, which is nonincreasing because
# each prox_entry is nondecreasing in its argument; 0 for a halfspace that the
# proximal point of f satisfies. The slope of φ is at least -γ‖a‖², so the first
# step, |φ(0)|/(γ‖a‖²), does not pass the root. Each further step is twice the
# step to the root of the secant through the last two points, which passes the
# root when that secant predicts it well, or twice the last step where φ is flat.
# Regula falsi with the Illinois rule then shrinks the bracket; it is exact once
# both ends lie on one linear piece of φ, as for the ℓ1 norm and boxes.
function _multiplier(c::LinearConstrained, x::AbstractArray, γ::T) where {T}
  φ0, m0 = _constraint_gap(c, x, zero(T), γ)
  (abs(φ0) <= 2eps(T) * m0 || (c.set isa IndHalfspace && φ0 < 0)) && return zero(T)
  step = abs(φ0) / (γ * convert(T, _normal_sqnorm(c.set, x)))
  dir = φ0 > 0 ? one(T) : -one(T)
  lo, φlo = zero(T), φ0                     # φlo has the sign of φ(0), φhi the other
  hi, φhi = lo, φlo
  for k in 1:_MAX_BRACKET_STEPS
    hi = lo + dir * step
    φhi, mhi = _constraint_gap(c, x, hi, γ)
    abs(φhi) <= 2eps(T) * mhi && return hi
    signbit(φhi) == signbit(φ0) || break
    k == _MAX_BRACKET_STEPS && _infeasible(c)
    s = step * φhi / (φlo - φhi)
    step = isfinite(s) && s > 0 ? min(2s, 64step) : 2step
    lo, φlo = hi, φhi
  end
  best, φbest = hi, φhi
  side = 0                                  # which end the last step kept
  for _ in 1:_MAX_SECANTS
    ν = (lo * φhi - hi * φlo) / (φhi - φlo)
    φν, mν = _constraint_gap(c, x, ν, γ)
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
