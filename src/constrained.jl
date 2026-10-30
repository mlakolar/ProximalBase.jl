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
- a hyperplane, halfspace or hyperslab: [`IndHyperplane`](@ref),
  [`IndHalfspace`](@ref) or [`IndHyperslab`](@ref);
- a simplex, [`IndSimplex`](@ref): the box `x ≥ 0` with the hyperplane
  `Σᵢ xᵢ = r`, or with the halfspace `Σᵢ xᵢ ≤ r`;
- for a norm, an ℓ2 ball or sphere, [`IndBallL2`](@ref) or
  [`IndSphereL2`](@ref), described below;
- for a quadratic, an affine set [`IndAffine`](@ref), described below.

On a box, the proximal operator clamps that of `f` entry by entry, so
`Constrained(f, box)` is separable, with [`prox_entry`](@ref) for coordinate
descent. On a hyperplane `⟨a, x⟩ = b`, the proximal operator is
`prox_{γf}(x - γνa)` for the multiplier `ν` at which it satisfies the
constraint, found by a search over the scalar `ν`; `f` may itself be
constrained to a box. Each step of the search is a pass over `x`. It starts
from the previous multiplier when the maximum step is unchanged, with a
first step from the exact slope of the constraint where the scalar proximal
operators of `f` are piecewise linear, as for every function below. The calls
of a solver on slowly changing points then take three to five passes, against
7 to 16 from zero. From zero, on 4096 entries or more, where the scalar
proximal operators of `f` stop at a bound, as on a box, the nonnegative orthant
or a simplex, the search fixes the entries that have reached one once few
others stay, and its later passes skip them: when the multiplier lies far out
in the tail of the entries, as for a simplex of radius 1, such a call takes
about six passes and is 1.6 to 2.9 times faster. Do not share one function
object between threads.

On a hyperslab `l ≤ ⟨a, x⟩ ≤ u`, the multiplier is zero when the unconstrained
proximal point lies between the bounds. Otherwise the same search meets the
violated bound, with a negative multiplier at the lower bound and a positive
one at the upper bound. A previous multiplier from the other side is discarded.

On a box, a hyperplane, a halfspace, a hyperslab or a simplex, `γ` may also be
an array of steps, one per entry: the proximal operator in the metric of `Diagonal(1 ./ γ)`,
by the same search with `prox_{γᵢfᵢ}(xᵢ - γᵢνaᵢ)` in entry `i`. For example:

- `Constrained(NormL1(λ), IndHyperplane(1, 0))` is the lasso penalty on
  coefficients that sum to zero, as in regression on compositional data;
- `Constrained(NormL1(λ), IndBox(0, u))` is the nonnegative lasso penalty with
  upper bounds;
- `Constrained(IndBox(0, 1), IndHyperplane(1, k))` is the indicator of the capped
  simplex `{x : 0 ≤ x ≤ 1, Σᵢ xᵢ = k}`;
- `Constrained(SqrNormL2(λ), IndSimplex(1))` is the ridge penalty on the
  probability simplex.

The conjugate of the indicator of a box with scalar bounds intersected with a
hyperplane or a halfspace with a scalar `a`, such as the capped simplex, has a
value: the support function of the set. For the capped simplex it is the sum of
the `k` largest entries, and [`SingularValue`](@ref) and [`Spectral`](@ref) turn
it into the Ky Fan `k`-norm and the sum of the `k` largest eigenvalues.

`f` must be convex: for a nonconvex penalty, clamping its proximal operator does
not give the constrained one. The exceptions are [`MCP`](@ref), [`SCAD`](@ref)
and [`NormL0`](@ref) on a box, whose scalar problems are solved exactly by
comparing the minimizers of their convex pieces with the ends of the interval;
the result is separable but not convex.

Repeated box constraints on these three penalties are intersected entrywise in
linear order, including boxes made by `IndBallLinf`. All array bounds must have
equal length; the resulting array bounds have the shape of the first array
bound. Endpoints and emptiness are compared before conversion. Same-type bounds
keep their type. Mixed bounds use their promoted type, widened to the promoted
floating-point type only if a selected finite endpoint would become infinite.
Mixed integer and floating-point bounds can still round integers beyond the
floating-point type's precision. Convex nested box constraints retain their
composition and the empty-domain behaviour described below.

For a linear constraint (including a simplex), steps must be positive and
finite. Array steps are normalized by their maximum; a ratio below the
smallest normal number in working precision is rejected with `ArgumentError`,
before writing.
The search uses the scaled multiplier `μ = max(γ)ν`. For finite inputs, it
returns a point only when it is finite and satisfies the constraint to the search tolerance.
Otherwise it throws `ArgumentError`: the multiplier could not be bracketed or
resolved in working precision. Huge steps for non-indicators can lose the
information needed to resolve a multiplier, even when the problem is feasible.
Nonfinite inputs follow the package's IEEE policy. A NaN search residual
returns a NaN multiplier without replacing the stored search history.

The caller must ensure that the set meets the domain of `f`. For a convex
function on a box, the method clamps the unconstrained proximal point even
when the intersection is empty. Neither that point nor the returned value
certifies feasibility; `value_and_prox!` returns zero for an indicator even in
this improper case. On a hyperplane, halfspace or hyperslab, the search throws
an `ArgumentError` if it cannot bracket a multiplier. This is not a general test
for an empty intersection: rounding may put a point on the closure of an open
domain, and a finite multiplier may not exist when the set meets only the
boundary of a closed domain. Empty arguments that fail the search's own
acceptance test are rejected before writing. Block oracles also reject clearly
infeasible scalar boxes with scalar linear normals at finite positive scalar
steps, when finite inputs and conservative rounding bounds certify that the
search throws. Overflowing ranges and transformed inputs remain unchecked.
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

# f on a linear constraint; the search for the multiplier starts where
# the last one ended
struct LinearConstrained{F<:AbstractFunction, H<:Union{IndHyperplane, IndHalfspace, IndHyperslab}} <: AbstractFunction
  f::F
  set::H
  last::Base.RefValue{Float64}      # μ = max(γ)ν for the normalized constraint
  scale::Base.RefValue{Float64}     # the step scale of the stored multiplier
  slope::Base.RefValue{Float64}     # its residual's slope, from the last secant
  verify::Base.RefValue{Bool}       # certify fixed sums or unresolved refinement in output
  scratch::Scratch                  # for the value of the conjugate
  output::Scratch                   # candidate point, committed only after verification
  live::NTuple{6, Scratch}          # the entries the search still evaluates: x, a, γ, index; bits; state

  LinearConstrained{F, H}(f::F, set::H) where {F<:AbstractFunction, H<:Union{IndHyperplane, IndHalfspace, IndHyperslab}} =
    (_check_constrainable(f); new{F, H}(f, set, Ref(NaN), Ref(NaN), Ref(NaN), Ref(false), Scratch(), Scratch(), ntuple(_ -> Scratch(), 6)))
end
LinearConstrained(f::AbstractFunction, set::Union{IndHyperplane, IndHalfspace, IndHyperslab}) =
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
Constrained(f::AbstractFunction, C::Union{IndHyperplane, IndHalfspace, IndHyperslab}) = LinearConstrained(f, C)
Constrained(f::AbstractFunction, C::IndSimplex) =
  LinearConstrained(Constrained(f, IndNonnegative()), C.equality ? IndHyperplane(1, C.r) : IndHalfspace(1, C.r))
# a box goes under the linear constraint, where the function stays separable
Constrained(c::LinearConstrained, C::IndBox) = LinearConstrained(Constrained(c.f, C), c.set)
Constrained(c::LinearConstrained, C::IndNonnegative) = LinearConstrained(Constrained(c.f, C), c.set)
Constrained(::AbstractFunction, C::AbstractFunction) =
  throw(ArgumentError("Constrained takes a box, a hyperplane, a halfspace, a hyperslab, a simplex, an ℓ2 ball, an ℓ2 sphere or an affine set, not $(nameof(typeof(C)))"))

isproximable(::Type{<:Union{BoxConstrained, LinearConstrained}}) = true
isconvex(::Type{<:LinearConstrained}) = true
isconvex(::Type{<:BoxConstrained{F}}) where {F} = isconvex(F)
isseparable(::Type{<:BoxConstrained}) = true
_isindicator(::Type{<:BoxConstrained{F}}) where {F} = _isindicator(F)
_isindicator(::Type{<:LinearConstrained{F}}) where {F} = _isindicator(F)

####################################
#  on a box
####################################

(c::BoxConstrained)(x::AbstractArray) =
  (_checkparams(c, x); T = float(eltype(x)); iszero(c.box(x)) ? convert(T, c.f(x)) : T(Inf))

# the scalar problem is convex, so its minimizer on an interval is the nearest
# point to its unconstrained minimizer
Base.@propagate_inbounds prox_entry(c::BoxConstrained, v::Real, i::Integer, γ::Real) =
  prox_entry(c.box, prox_entry(c.f, v, i, γ), i, γ)

# γ is a scalar step or an array of steps, one per entry
prox!(y::AbstractArray, c::BoxConstrained, x::AbstractArray, γ::Real) = (_checkwrite(prox!, c, y, x); _prox_sep!(y, c, x, γ))
prox!(y::AbstractArray, c::BoxConstrained, x::AbstractArray, γ::AbstractArray) = (_checkwrite(prox!, c, y, x); _prox_sep!(y, c, x, γ))
value_and_prox!(y::AbstractArray, c::BoxConstrained, x::AbstractArray, γ::Real) = _value_and_prox_box!(y, c, x, γ)
value_and_prox!(y::AbstractArray, c::BoxConstrained, x::AbstractArray, γ::AbstractArray) = _value_and_prox_box!(y, c, x, γ)
function _value_and_prox_box!(y, c, x, γ)
  _checkwrite(value_and_prox!, c, y, x, γ)
  _prox_sep!(y, c, x, γ)
  T = float(eltype(y))
  _isindicator(c) ? zero(T) : convert(T, c.f(y))
end

_checkparams(c::BoxConstrained, x) = (_checkparams(c.f, x); _checkparams(c.box, x); true)
function _checkwrite(op::O, c::BoxConstrained, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checksize(y, x)
  _checkparams(c, x)
  op === value_and_prox! && !_isindicator(c) && _checkvalue(c.f, y)
  true
end
# A step is needed only for conditional values of nested box constraints.
_checkwrite(op::O, f, y, x, γ) where {O} = _checkwrite(op, f, y, x)
_checkwrite(op::O, s::Scaled, y, x, γ::Real) where {O} = _checkwrite(op, s.f, y, x, s.α * γ)
_checkwrite(op::O, s::Scaled, y, x, γ::AbstractArray) where {O} = _checkwrite(op, s.f, y, x, _ScaledArray(γ, s.α))
function _checkwrite(op::O, t::Tilt, y::AbstractArray, x::AbstractArray, γ) where {O}
  _checkwrite(op, t, y, x)
  if op === value_and_prox! && !_box_value_available(t.f, y)
    z = _array_buffer!(t.scratch, float(eltype(y)), size(y))
    step = γ isa Real ? convert(float(eltype(y)), γ) : γ
    z .= x .- step .* t.c
    _checkwrite(op, t.f, y, z, γ)
  end
  true
end
function _checkwrite(op::O, t::Translate, y::AbstractArray, x::AbstractArray, γ) where {O}
  _checkwrite(op, t, y, x)
  if op === value_and_prox! && !_box_value_available(t.f, y)
    z = _array_buffer!(t.proxscratch, float(eltype(y)), size(y))
    z .= x .- t.b
    _checkwrite(op, t.f, y, z, γ)
  end
  true
end
function _checkwrite(op::O, s::SeparableSum, y::Tuple, x::Tuple, γ) where {O}
  _checksize(y, x)
  _checkblocks(s, x)
  _checkwrite_blocks(op, s.fs, y, x, γ)
end
_checkwrite_blocks(op, ::Tuple{}, ::Tuple{}, ::Tuple{}, γ) = true
_checkwrite_blocks(op::O, fs::Tuple, y::Tuple, x::Tuple, γ) where {O} =
  (_checkwrite(op, first(fs), first(y), first(x), γ); _checkwrite_blocks(op, Base.tail(fs), Base.tail(y), Base.tail(x), γ))

# Read the prospective output only when an inner value check fails.
struct _BoxProxArray{T, N, C, X, G} <: AbstractArray{T, N}
  c::C
  x::X
  γ::G
end
@inline _BoxProxArray(y::AbstractArray{T, N}, c, x, γ) where {T, N} =
  _BoxProxArray{T, N, typeof(c), typeof(x), typeof(γ)}(c, x, γ)
Base.size(p::_BoxProxArray) = size(p.x)
Base.IndexStyle(::Type{<:_BoxProxArray}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(p::_BoxProxArray{T}, i::Int) where {T} =
  convert(T, prox_entry(p.c, p.x[i], i, _entry(p.γ, i)))

struct _BoxShiftArray{T, N, P, B} <: AbstractArray{T, N}
  p::P
  b::B
end
_BoxShiftArray(p::AbstractArray{T, N}, b) where {T, N} =
  _BoxShiftArray{T, N, typeof(p), typeof(b)}(p, b)
Base.size(p::_BoxShiftArray) = size(p.p)
Base.IndexStyle(::Type{<:_BoxShiftArray}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(p::_BoxShiftArray{T}, i::Int) where {T} =
  convert(T, p.p[i] - p.b[i])

@inline _check_box_value(f, y, p) = _checkvalue(f, y)
@inline _check_box_value(s::Scaled, y, p) = _check_box_value(s.f, y, p)
@inline _check_box_value(t::Tilt, y, p) = (_checkdot(t.c, y); _check_box_value(t.f, y, p))
@inline _check_box_value(t::Translate, y, p) =
  _check_box_value(t.f, _checkshifted(t, y), _BoxShiftArray(p, t.b))
function _box_value_available(f, y)
  try
    _checkvalue(f, y)
    true
  catch
    false
  end
end
# Avoid constructing an exception when the outer box excludes the inner value.
_box_value_available(f::SqrNormL2, y) = f.y === nothing || size(f.y) == size(y)
_box_value_available(c::Conjugate, y) =
  _has_conjugate_value(c.f) && invoke(_box_value_available, Tuple{Any, Any}, c, y)
_box_value_available(s::Scaled, y) = _box_value_available(s.f, y)
_box_value_available(t::Tilt, y) =
  _box_value_available(t.f, y) && invoke(_box_value_available, Tuple{Any, Any}, t, y)
_box_value_available(t::Translate, y) = _box_value_available(t.f, _checkshifted(t, y))
_box_value_available(c::BoxConstrained, y) = _box_value_available(c.f, y)
@inline function _check_box_value(c::BoxConstrained, y, p)
  _checkparams(c, y)
  _box_value_available(c.f, y) && return true
  iszero(c.box(p)) && _check_box_value(c.f, y, p)
  true
end
@inline function _checkwrite(op::O, c::BoxConstrained, y::AbstractArray, x::AbstractArray, γ) where {O}
  _checkwrite(op, c, y, x)
  if op === value_and_prox! && !_isindicator(c)
    _checkparam(γ, x)
    _check_box_value(c.f, y, _BoxProxArray(y, c, x, γ))
  end
  true
end

# Membership can return Inf without evaluating the inner value.
_checkvalue(c::BoxConstrained, x::AbstractArray) = _checkparams(c, x)
_checkvalue(c::LinearConstrained, x::AbstractArray) = _checkvalue(c.set, x)
iszeropreserving(c::BoxConstrained) = iszeropreserving(c.f) && iszeropreserving(c.box)

# For a convex f with 0 in its domain, the proximal step of f from -gᵢ moves in
# the direction that improves on zero, if one does; the box must allow it.
function kkt_violation(c::BoxConstrained, g::AbstractArray, i::Integer)
  v = kkt_violation(c.f, g, i)
  v > 0 || return v
  d = prox_entry(c.f, -g[i], i, one(float(eltype(g))))
  (d > 0 ? _entry(c.box.u, i) > 0 : _entry(c.box.l, i) < 0) ? v : zero(v)
end

restrict(c::BoxConstrained, S::AbstractVector{<:Integer}) = BoxConstrained(restrict(c.f, S), restrict(c.box, S))
_restrict_unchecked(c::BoxConstrained, S::AbstractVector{<:Integer}) =
  BoxConstrained(_restrict_unchecked(c.f, S), _restrict_unchecked(c.box, S))

####################################
#  on a hyperplane, halfspace or hyperslab
####################################

(c::LinearConstrained)(x::AbstractArray) = (_checkparams(c, x); T = float(eltype(x)); iszero(c.set(x)) ? convert(T, c.f(x)) : T(Inf))

_checkparams(c::LinearConstrained, x) = (_checkparams(c.f, x); _checkparams(c.set, x); true)
function _checkwrite(op::O, c::LinearConstrained, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checksize(y, x)
  _checkparams(c, x)
  c.set isa IndHyperslab && _checkwrite(op, c.set, y, x)
  op === value_and_prox! && !_isindicator(c) && _checkvalue(c.f, y)
  true
end

# Keep the true steps for prox_entry and normalize only the multiplier shift.
struct _SearchSteps{G, T, M}
  steps::G
  scale::T
  factor::T
end
Base.@propagate_inbounds _entry(g::_SearchSteps, i) = _entry(g.steps, i)
@inline _ratio(g::_SearchSteps{<:Real}, i, ::Type{T}) where {T} = one(T)
# Select the division fallback once, when the reciprocal would overflow.
Base.@propagate_inbounds _ratio(g::_SearchSteps{<:AbstractArray, S, true}, i, ::Type{T}) where {S, T} =
  convert(T, _entry(g, i)) * g.factor
Base.@propagate_inbounds _ratio(g::_SearchSteps{<:AbstractArray, S, false}, i, ::Type{T}) where {S, T} =
  convert(T, _entry(g, i)) / g.scale
_with_steps(g::_SearchSteps{G, T, M}, steps) where {G, T, M} =
  _SearchSteps{typeof(steps), T, M}(steps, g.scale, g.factor)

@noinline _invalid_linear_step() =
  throw(ArgumentError("the steps of LinearConstrained must be positive and finite"))
@noinline _invalid_linear_ratio() =
  throw(ArgumentError("Constrained needs normal step ratios to their maximum in working precision"))
@inline _search_steps(γ, ::Type{T}) where {T} = _search_steps(identity, γ, T)
@inline function _search_steps(done::F, γ::Real, ::Type{T}) where {F, T}
  g = convert(T, γ)
  isfinite(g) && g > 0 || _invalid_linear_step()
  done(_SearchSteps{typeof(g), T, true}(g, g, one(T)))
end
@inline function _search_steps(done::F, γ::AbstractArray, ::Type{T}) where {F, T}
  least, scale = T(Inf), zero(T)
  valid = true
  @inbounds @simd for i in eachindex(IndexLinear(), γ)
    t = convert(T, γ[i])
    valid &= isfinite(t) & (t > 0)
    least = min(least, t)
    scale = max(scale, t)
  end
  isempty(γ) && (scale = one(T))
  valid || _invalid_linear_step()
  factor = inv(scale)
  if isfinite(factor)
    least * factor >= floatmin(T) || _invalid_linear_ratio()
    return done(_SearchSteps{typeof(γ), T, true}(γ, scale, factor))
  end
  least / scale >= floatmin(T) || _invalid_linear_ratio()
  done(_SearchSteps{typeof(γ), T, false}(γ, scale, factor))
end

# γ is a scalar step, or an array of steps, one per entry: the proximal operator
# in the metric of Diagonal(1 ./ γ), where z(ν) has the entries
# prox_{γᵢfᵢ}(xᵢ - γᵢνaᵢ)
prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::Real) =
  _linear_constrained_prox!(y, c, x, convert(float(eltype(y)), γ))
prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::AbstractArray) =
  _linear_constrained_prox!(y, c, x, γ)

_linear_constrained_prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ) =
  _linear_constrained_prox!(prox!, y, c, x, γ)
function _linear_constrained_prox!(op::O, y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ) where {O}
  _checkwrite(op, c, y, x, γ)
  _checkparam(γ, x)
  T = float(eltype(y))
  # Keep the ratio mode concrete so the step wrapper need not be boxed.
  _search_steps(γ, T) do steps
    _linear_constrained_search!(y, c, x, steps)
  end
end

@noinline function _linear_constrained_search!(y, c, x, γ::_SearchSteps{G, T}) where {G, T}
  last, slope, scale = c.last[], c.slope[], c.scale[]
  if !isequal(c.scale[], Float64(γ.scale))
    c.last[] = NaN
    c.slope[] = NaN
  end
  c.verify[] = false
  ν = _multiplier(c, x, γ, T)
  if isnan(ν)
    c.last[], c.slope[], c.scale[] = last, slope, scale
  else
    c.scale[] = γ.scale
  end
  set = c.set
  set isa IndHyperslab && iszero(ν) && !c.verify[] && return _linear_constrained_inactive_output!(y, c.f, x, γ)
  if _isscaled(set)
    return _linear_constrained_finish!(y, c, _normalized_normal(set.a, set.s), x, γ, ν)
  end
  _linear_constrained_finish!(y, c, set.a, x, γ, ν)
end

function _linear_constrained_finish!(y, c, a, x, γ, ν)
  c.verify[] && return _linear_constrained_verified_finish!(y, c, a, x, γ, ν)
  _linear_constrained_output!(y, c.f, a, x, γ, ν, Val(false))
  y
end

# Keep buffer preparation out of the ordinary full-pass hit.
@noinline function _linear_constrained_verified_finish!(y, c, a, x, γ, ν)
  candidate = _linear_constrained_output_buffer!(c.output, y)
  s, m = _linear_constrained_output!(candidate, c.f, a, x, γ, ν, Val(true))
  _verify_output(c, s, m, ν)
  _linear_constrained_commit!(y, candidate)
  y
end

# Compile the copy once per output layout, independently of f and the set.
@noinline _linear_constrained_commit!(y, candidate) = copyto!(y, candidate)

@noinline _linear_constrained_output_buffer!(s::Scratch, y::AbstractArray) =
  _array_buffer!(s, eltype(y), size(y))
@noinline function _linear_constrained_output_buffer!(s::Scratch, y::SymmetricSparseIterate{T}) where {T}
  b = s.buf
  if b isa SymmetricSparseIterate{T} && size(b) == size(y)
    return b
  end
  b = SymmetricSparseIterate(T, y.p)
  s.buf = b
  b
end

# Full passes already certify ordinary hits. For fixed sums or a refinement
# fallback, certify the buffered point before committing it to the output.
function _verify_output(c, s, m, ν)
  isfinite(s) && isfinite(m) || _infeasible(c)
  set = c.set
  if set isa IndHyperslab
    ν < 0 && return _verify_output_bound(c, s, m, _scaled_offset(set, set.l), 0)
    ν > 0 && return _verify_output_bound(c, s, m, _scaled_offset(set, set.u), 0)
    isfinite(set.l) && _verify_output_bound(c, s, m, _scaled_offset(set, set.l), -1)
    isfinite(set.u) && _verify_output_bound(c, s, m, _scaled_offset(set, set.u), 1)
  else
    _verify_output_bound(c, s, m, _scaled_offset(set), set isa IndHyperplane || !iszero(ν) ? 0 : 1)
  end
  nothing
end
function _verify_output_bound(c, s::T, m, b, side) where {T}
  φ, scale = s - b, m + abs(b)
  isfinite(φ) && isfinite(scale) &&
    (side == 0 ? abs(φ) : side * φ) <= 8eps(T) * scale || _infeasible(c)
  nothing
end

# Match the promoted argument and converted steps of the inactive-point check.
function _linear_constrained_inactive_output!(y, f, x, γ::_SearchSteps{G, T}) where {G, T}
  @inbounds for i in eachindex(IndexLinear(), y, x)
    γᵢ = convert(T, _entry(γ, i))
    y[i] = convert(eltype(y), prox_entry(f, x[i] - zero(T), i, γᵢ))
  end
  y
end

@noinline function _linear_constrained_output!(y, f, a, x, γ, ν::T, ::Val{C} = Val(false)) where {T, C}
  s, m = zero(T), zero(T)
  @inbounds @simd for i in eachindex(IndexLinear(), y, x)
    γᵢ = convert(T, _entry(γ, i))
    aᵢ = convert(T, _entry(a, i))
    z = convert(eltype(y), prox_entry(f, x[i] - _ratio(γ, i, T) * ν * aᵢ, i, γᵢ))
    y[i] = z
    if C
      t = aᵢ * convert(T, z)
      s += t
      m += abs(t)
    end
  end
  C ? (s, m) : nothing
end

# The two matrix positions of a symmetric coordinate share storage.
function _linear_constrained_inactive_output!(y::SymmetricSparseIterate, f, x::AbstractMatrix, γ::_SearchSteps{G, T}) where {G, T}
  p = y.p
  @inbounds for c in 1:p, r in c:p
    i = (c - 1) * p + r
    γᵢ = convert(T, _entry(γ, i))
    z = prox_entry(f, x[r, c] - zero(T), i, γᵢ)
    _setstored!(y, convert(eltype(y), z), tril_index(p, r, c))
  end
  y
end

@noinline function _linear_constrained_output!(y::SymmetricSparseIterate, f, a, x::AbstractMatrix, γ, ν::T, ::Val{C} = Val(false)) where {T, C}
  p = y.p
  s, m = zero(T), zero(T)
  @inbounds for c in 1:p, r in c:p
    i = (c - 1) * p + r
    γᵢ = convert(T, _entry(γ, i))
    aᵢ = convert(T, _entry(a, i))
    v = x[r, c] - _ratio(γ, i, T) * ν * aᵢ
    z = convert(eltype(y), prox_entry(f, v, i, γᵢ))
    _setstored!(y, z, tril_index(p, r, c))
    if C
      t = aᵢ * convert(T, z)
      s += t
      m += abs(t)
      if r != c
        t = convert(T, _entry(a, (r - 1) * p + c)) * convert(T, z)
        s += t
        m += abs(t)
      end
    end
  end
  C ? (s, m) : nothing
end

value_and_prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::Real) = _value_and_prox_linear!(y, c, x, γ)
value_and_prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::AbstractArray) = _value_and_prox_linear!(y, c, x, γ)
function _value_and_prox_linear!(y, c, x, γ)
  _linear_constrained_prox!(value_and_prox!, y, c, x, γ)
  T = float(eltype(y))
  _isindicator(c) ? zero(T) : _linear_constrained_value(c.f, y, T)
end

# Share value computation across constraint and step types.
@noinline _linear_constrained_value(f, y, ::Type{T}) where {T} = convert(T, f(y))

# φ(ν) = ⟨a, z⟩ - b at z = z(ν), and Σᵢ |aᵢzᵢ| + |b|, the scale of its rounding
# error; with Val(true), also the slope of φ, -Σᵢ rᵢaᵢ² Pᵢ', where Pᵢ' is the
# slope of the scalar proximal operator at its argument. c is the function, or
# a named tuple with its f and set, as for the live entries of variable fixing.
function _constraint_gap(c::LinearConstrained, x::AbstractArray, ν::T, γ, b::Real, v::Val = Val(false)) where {T}
  set = c.set
  if _isscaled(set)
    return _constraint_gap(c.f, _normalized_normal(set.a, set.s), b, x, ν, γ, v)
  end
  _constraint_gap(c.f, set.a, b, x, ν, γ, v)
end
_constraint_gap(c, x::AbstractArray, ν::T, γ, v::Val = Val(false)) where {T} =
  _constraint_gap(c.f, c.set.a, c.set.b, x, ν, γ, v)

function _constraint_gap(f, a, b, x::AbstractArray, ν::T, γ, ::Val{S}) where {T, S}
  # Tail constraints subtract nearly equal fixed and live sums. Widen Float32
  # reductions to reduce sensitivity to contraction and normal scaling.
  A = T === Float32 ? Float64 : T
  s, m, d = zero(A), zero(A), zero(T)
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    aᵢ = convert(T, _entry(a, i))
    γᵢ = convert(T, _entry(γ, i))
    v = x[i] - _ratio(γ, i, T) * ν * aᵢ
    z, dz = S ? _prox_entry_slope(f, v, i, γᵢ) : (prox_entry(f, v, i, γᵢ), zero(T))
    t = convert(A, aᵢ) * convert(A, convert(T, z))
    s += t
    m += abs(t)
    S && (d += _ratio(γ, i, T) * aᵢ * aᵢ * convert(T, dz))
  end
  b = convert(A, convert(T, b))
  S ? (T(s - b), T(m + abs(b)), -d) : (T(s - b), T(m + abs(b)))
end

# Σᵢ γᵢaᵢ², the steepest slope of φ
_metric_sqnorm(set::Union{IndHyperplane, IndHalfspace, IndHyperslab}, x::AbstractArray, γ::Real) = γ * _normal_sqnorm(set, x)
function _metric_sqnorm(set::Union{IndHyperplane, IndHalfspace, IndHyperslab}, x::AbstractArray, γ::AbstractArray)
  T = float(promote_type(eltype(γ), typeof(set.q)))
  set.a isa Real && return convert(T, set.q) * convert(T, sum(γ))
  _isscaled(set) && return _metric_sqnorm(_normalized_normal(set.a, set.s), γ, T)
  _metric_sqnorm(set.a, γ, T)
end

# Convert integer steps before summing, as for the products in the array pass.
function _metric_sqnorm(set::IndHyperslab, x::AbstractArray, γ::AbstractArray{<:Integer})
  T = float(promote_type(eltype(γ), typeof(set.q)))
  set.a isa Real && return convert(T, set.q) * sum(T, γ)
  _metric_sqnorm(_scaled_normal(set), γ, T)
end

function _metric_sqnorm(a::AbstractArray, γ::AbstractArray, ::Type{T}) where {T}
  s = zero(T)
  @inbounds for i in eachindex(IndexLinear(), γ)
    s += convert(T, γ[i]) * convert(T, abs2(float(a[i])))
  end
  return s
end

# Σᵢ rᵢaᵢ², the steepest slope with respect to μ.
_metric_sqnorm(set, x, γ::_SearchSteps{<:Real}) = _normal_sqnorm(set, x)
function _metric_sqnorm(set, x, γ::_SearchSteps)
  T = typeof(γ.scale)
  a = _scaled_normal(set)
  s = zero(T)
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    s += _ratio(γ, i, T) * abs2(convert(T, _entry(a, i)))
  end
  s
end

@inline _resolved(φ::T, m, ulps = 2) where {T} = isfinite(φ) && isfinite(m) && abs(φ) <= ulps * eps(T) * m

const _MAX_BRACKET_STEPS = 64
const _MAX_SECANTS = 100

@noinline _infeasible(c::LinearConstrained) =
  throw(ArgumentError("the multiplier for $(nameof(typeof(c.set))) could not be bracketed or resolved in working precision: the set may miss the domain of $(nameof(typeof(c.f))), or the step is outside the range the search resolves"))

# Only nonfinite caller inputs take the package's IEEE path. Overflow or NaN
# generated by the search at finite inputs remains a precision failure.
@noinline function _nonfinite_linear_input(c, x)
  any(v -> !isfinite(v), x) || _infeasible(c)
  nothing
end

# φ and the scale of its rounding error at ν, and its slope when the scalar
# proximal operators of f have known slopes, NaN otherwise
_gap_and_slope(c::LinearConstrained{F}, x::AbstractArray, ν::T, γ, b) where {F, T} =
  _gap_and_slope(c, x, ν, γ, b, Val(_has_prox_slope(F)))
_gap_and_slope(c::LinearConstrained, x::AbstractArray, ν::T, γ, b, ::Val{true}) where {T} =
  _constraint_gap(c, x, ν, γ, b, Val(true))
_gap_and_slope(c::LinearConstrained, x::AbstractArray, ν::T, γ, b, ::Val{false}) where {T} =
  ((φ, m) = _constraint_gap(c, x, ν, γ, b); (φ, m, T(NaN)))

# The multiplier of the constraint, a root of φ, which is nonincreasing because
# each prox_entry is nondecreasing in its argument; for a halfspace, 0 when the
# proximal point of f satisfies the constraint. The search starts from the
# multiplier of the last call, and from 0 at the first call or when that was
# an inactive halfspace.
function _multiplier(c::LinearConstrained, x::AbstractArray, γ, ::Type{T}) where {T}
  c.set isa IndHyperslab && return _hyperslab_multiplier(c, x, γ, T)
  ν0 = convert(T, c.last[])
  warm = isfinite(ν0) && !(c.set isa IndHalfspace && !(ν0 > 0))
  slope = c.slope[]
  ν = _search(c, x, γ, warm ? ν0 : zero(T), warm, _scaled_offset(c.set), c.set isa IndHalfspace ? 1 : 0)
  isnan(ν) ? (c.slope[] = slope) : (c.last[] = ν)
  ν
end

# Keep the selected finite bound through every pass, including variable fixing.
function _hyperslab_multiplier(c::LinearConstrained, x::AbstractArray, γ, ::Type{T}) where {T}
  set = c.set
  if isfinite(set.l)
    b = _scaled_offset(set, set.l)
    φ, m = _constraint_gap(c, x, zero(T), γ, b)
    if !(isfinite(φ) && isfinite(m))
      _nonfinite_linear_input(c, x)
      isnan(φ) && return T(NaN)
    end
    φ < 0 && return _slab_multiplier(c, x, γ, T, b, -1)
  end
  if isfinite(set.u)
    b = _scaled_offset(set, set.u)
    φ, m = _constraint_gap(c, x, zero(T), γ, b)
    if !(isfinite(φ) && isfinite(m))
      _nonfinite_linear_input(c, x)
      isnan(φ) && return T(NaN)
    end
    φ > 0 && return _slab_multiplier(c, x, γ, T, b, 1)
  end
  if !isfinite(set.l) && !isfinite(set.u)
    # The unused normal need not be representable in the output's precision.
    finite = true
    @inbounds @simd for i in eachindex(IndexLinear(), x)
      z = prox_entry(c.f, x[i] - zero(T), i, convert(T, _entry(γ, i)))
      finite &= isfinite(convert(T, z))
    end
    finite || _nonfinite_linear_input(c, x)
  end
  c.last[] = 0
  c.slope[] = NaN
  zero(T)
end

function _slab_multiplier(c, x, γ, ::Type{T}, b, side) where {T}
  ν0 = convert(T, c.last[])
  warm = isfinite(ν0) && side * ν0 > 0
  slope = c.slope[]
  warm || (c.slope[] = NaN)
  ν = _search(c, x, γ, warm ? ν0 : zero(T), warm, b, side)
  isnan(ν) ? (c.slope[] = slope) : (c.last[] = ν)
  ν
end

# The search from start. The first step is at least |φ(start)|/Σᵢ rᵢaᵢ², a lower
# bound on the distance to the root, as the slope of φ is at least -Σᵢ rᵢaᵢ². From
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
_search(c::LinearConstrained, x::AbstractArray, γ, start::T, warm::Bool, b, side) where {T} =
  warm || length(x) < _FIX_NMIN ? _search(c, x, γ, start, warm, b, side, nothing) : _search(c, x, γ, start, false, b, side, _live(c, x, T))

function _search(c::LinearConstrained, x::AbstractArray, γ, start::T, warm::Bool, b, side, L) where {T}
  φs, ms, dφ = warm ? _gap_and_slope(c, x, start, γ, b) : (_constraint_gap(c, x, start, γ, b)..., T(NaN))
  if !(isfinite(φs) && isfinite(ms))
    _nonfinite_linear_input(c, x)
    return isnan(φs) ? T(NaN) : start
  end
  _resolved(φs, ms) && return start
  if side != 0 && side * φs < 0
    iszero(start) && return zero(T)                           # the constraint is inactive
    # Check zero when the root lies between it and the previous multiplier.
    φ0, m0 = _constraint_gap(c, x, zero(T), γ, b)
    if isnan(φ0)
      _nonfinite_linear_input(c, x)
      return T(NaN)
    end
    (isfinite(φ0) && isfinite(m0) && (side * φ0 <= 0 || _resolved(φ0, m0))) && return zero(T)
    return _refine(c, x, γ, b, zero(T), φ0, start, φs, nothing)
  end
  a2 = convert(T, _metric_sqnorm(c.set, x, γ))
  step = abs(φs) / a2
  L = _seen(L, start, φs, a2)
  σ = dφ < 0 ? dφ : (warm ? convert(T, c.slope[]) : T(NaN))
  σ < 0 && (step = max(step, abs(φs) / -σ))
  _bracket_and_refine(c, x, γ, b, start, φs, step, L)
end

_bracket_and_refine(c::LinearConstrained, x::AbstractArray, γ::Real, b, ν0::T, φ0::T, step::T, L) where {T} =
  _bracket_and_refine(c, x, _search_steps(γ, T), b, ν0, φ0, step, L)
_refine(c::LinearConstrained, x::AbstractArray, γ::Real, b, lo::T, φlo::T, hi::T, φhi::T, L) where {T} =
  _refine(c, x, _search_steps(γ, T), b, lo, φlo, hi, φhi, L)

function _bracket_and_refine(c::LinearConstrained, x::AbstractArray, γ, b, ν0::T, φ0::T, step::T, L) where {T}
  dir = φ0 > 0 ? one(T) : -one(T)
  lo, φlo = ν0, φ0                          # φlo has the sign of φ(ν₀), φhi the other
  hi, φhi = lo, φlo
  for k in 1:_MAX_BRACKET_STEPS
    hi = lo + dir * step
    isfinite(hi) || _infeasible(c)
    φhi, mhi, L = _gap!(c, x, hi, γ, b, L)
    if isnan(φhi)
      _nonfinite_linear_input(c, x)
      return T(NaN)
    end
    isfinite(φhi) && isfinite(mhi) || _infeasible(c)
    if _resolved(φhi, mhi)
      c.slope[] = (φhi - φlo) / (hi - lo)
      return _search_result(c, hi, L)
    end
    signbit(φhi) == signbit(φ0) || break
    k == _MAX_BRACKET_STEPS && _infeasible(c)
    s = step * (φhi / (φlo - φhi))
    step = isfinite(s) && s > 0 ? min(2s, 64step) : 2step
    lo, φlo = hi, φhi
  end
  # regula falsi passes over the live entries once some are fixed, and searches
  # as before otherwise
  L isa Base.RefValue && L[].compact && return _refine(c, x, γ, b, lo, φlo, hi, φhi, L)
  _refine(c, x, γ, b, lo, φlo, hi, φhi, nothing)
end

# Fixed sums can round differently from the output's full residual.
@inline _search_result(c, μ, ::Nothing) = μ
@inline function _search_result(c, μ, L::Base.RefValue)
  c.verify[] = L[].compact
  μ
end

# regula falsi with the Illinois rule on the bracket [lo, hi], where φlo and φhi
# have opposite signs; keeps the slope of the last secant for the next call
function _refine(c::LinearConstrained, x::AbstractArray, γ, b, lo::T, φlo::T, hi::T, φhi::T, L) where {T}
  best, φbest = abs(φlo) < abs(φhi) ? (lo, φlo) : (hi, φhi)
  νp, φp = hi, φhi                          # the last point evaluated, for the slope
  side = 0                                  # which end the last step kept
  for _ in 1:_MAX_SECANTS
    # Preserve ordinary interpolation rounding; scale only when it breaks down.
    ν = (lo * φhi - hi * φlo) / (φhi - φlo)
    if !isfinite(ν) || !(min(lo, hi) < ν < max(lo, hi))
      w = abs(φlo) / max(abs(φlo), abs(φhi))
      v = abs(φhi) / max(abs(φlo), abs(φhi))
      t = w / (w + v)
      ν = (one(T) - t) * lo + t * hi
      if !isfinite(ν) || !(min(lo, hi) < ν < max(lo, hi))
        ν = lo / 2 + hi / 2
      end
    end
    φν, mν, L = _gap!(c, x, ν, γ, b, L, Val(false))
    if isnan(φν)
      _nonfinite_linear_input(c, x)
      return T(NaN)
    end
    isfinite(φν) && isfinite(mν) || _infeasible(c)
    ν != νp && (c.slope[] = (φν - φp) / (ν - νp))
    νp, φp = ν, φν
    abs(φν) < abs(φbest) && ((best, φbest) = (ν, φν))
    _resolved(φν, mν) && return _search_result(c, ν, L)
    if signbit(φν) == signbit(φhi)
      hi, φhi = ν, φν
      side == 1 && (φlo /= 2)
      side = 1
    else
      lo, φlo = ν, φν
      side == -1 && (φhi /= 2)
      side = -1
    end
    # Resolve the remaining floats before declaring the residual unresolvable.
    nextfloat(min(lo, hi)) >= max(lo, hi) && break
  end
  # The output pass certifies a fallback, even when neither endpoint resolved.
  c.verify[] = true
  best
end

####################################
#  variable fixing
####################################

# Each aᵢzᵢ(ν) = aᵢPᵢ(xᵢ - γᵢνaᵢ) is nonincreasing in ν. Where the scalar proximal
# operator Pᵢ has a range [Lᵢ, Uᵢ] with a finite end, an entry whose aᵢzᵢ sits at
# the bottom of its range at a point with φ > 0 stays there over the rest of the
# search, which looks above that point only, and an entry at the top at a point
# with φ < 0 stays there below it. Such entries are fixed: aᵢzᵢ and |aᵢzᵢ| join
# two sums, and the passes that follow skip them (Kiwiel, 2008). As computed, zᵢ
# is monotone in ν and equals the end of its range exactly once there, so the
# sums hold the terms that the pass would add. The live entries are copied to
# buffers, x and the arrays among a and γ with the indices, so that a pass over
# them reads contiguous memory and vectorizes as a pass over x.
#
# A pass costs about a seventh of a nanosecond per entry on an AMD EPYC 7513,
# and the copy two to three passes, so a call from zero fixes entries only where
# few stay live: at a point of the bracketing, when the secant through the last
# two points is flat against the steepest slope of the live entries, as in the
# tail of the entries, and an evenly spaced sample of them shows that at most a
# quarter stays. After a sample that shows more, the next waits for a secant
# half as steep. Regula falsi then passes over the live entries. A call from the
# last multiplier ends within a few passes, which the copy would not repay, and
# a call on fewer than 4096 entries would lose about a percent where the root is
# not in the tail; both search as before.
# The sums, slopes and multipliers use the same normalized constraint.
struct _Live{T}
  n::Int          # the live entries: the first n in the buffers, or all of x
  compact::Bool   # whether they are in the buffers
  s::T            # Σᵢ aᵢzᵢ over the fixed entries, with the normalized normal
  m::T            # Σᵢ |aᵢzᵢ| over them
  νp::T           # the last point evaluated, and φ there
  φp::T
  a2::T           # Σᵢ γᵢaᵢ² over all entries, the steepest slope of φ
  ρmax::T         # the flattest secant that calls for a sample, against the
                  # steepest slope of the live entries
end

const _FIX_NMIN = 4096          # the least n with fixing
const _FIX_MIN = 256            # the least live entries worth a copy
const _FIX_RHO = 0.2            # the flattest secant that calls for a sample
const _FIX_KEEP = 0.25          # the largest share of the sample that stays live for a copy
const _FIX_SAMPLE = 32          # the size of the sample

# The state lives in the function object, and the search passes a reference to
# it, so that it does not crowd the loop of the pass, inlined into the search,
# out of the registers.
function _live(c::LinearConstrained{F}, x::AbstractArray, ::Type{T}) where {F, T}
  _has_prox_range(F) || return nothing
  holder = c.live[6]
  R = holder.buf
  if !(R isa Base.RefValue{_Live{T}})
    R = Ref{_Live{T}}()
    holder.buf = R
  end
  n = length(x)
  R[] = _Live{T}(n, false, zero(T), zero(T), T(NaN), T(NaN), T(NaN), T(_FIX_RHO))
  return R
end

# the first point and the steepest slope, once known
_seen(::Nothing, ν, φ, a2) = nothing
function _seen(R::Base.RefValue{_Live{T}}, ν, φ, a2) where {T}
  L = R[]
  R[] = _Live{T}(L.n, L.compact, L.s, L.m, ν, φ, a2, L.ρmax)
  return R
end

# φ, the scale of its rounding error, and the state; with Val(true), the point
# may fix entries, as while the search brackets the root: regula falsi then
# passes over the entries left live
@inline _gap!(c::LinearConstrained, x::AbstractArray, ν::T, γ, b, ::Nothing, ::Val = Val(true)) where {T} =
  (_constraint_gap(c, x, ν, γ, b)..., nothing)
@inline _gap!(c::LinearConstrained, x::AbstractArray, ν::T, γ, b, R::Base.RefValue{_Live{T}}, fix::Val = Val(true)) where {T} =
  (_fixing_gap!(c, x, ν, γ, b, R, fix)..., R)

# A call, so that the loop of the pass keeps the registers that the state of
# the search would take
@noinline function _fixing_gap!(c::LinearConstrained, x::AbstractArray, ν::T, γ, b, R::Base.RefValue{_Live{T}}, ::Val{F}) where {T, F}
  φ, M = R[].compact ? _live_gap(c, x, ν, γ, b, R[]) : _constraint_gap(c, x, ν, γ, b)
  F && _note!(c, x, ν, γ, R, φ, M)
  φ, M
end

# keeps the point, and fixes entries when the secant through the last two
# points is flat against the steepest slope of the live entries, unless
# rounding dominates it
@noinline function _note!(c::LinearConstrained, x::AbstractArray, ν::T, γ, R::Base.RefValue{_Live{T}}, φ::T, M::T) where {T}
  L = R[]
  Δ = abs(L.φp - φ)
  if abs(φ) > 2eps(T) * M && L.n >= _FIX_MIN && Δ > 8eps(T) * M &&
     Δ * length(x) <= L.ρmax * L.a2 * abs(ν - L.νp) * L.n
    L = _try_fix(c, x, ν, γ, L, φ > 0, Δ * length(x) / (L.a2 * abs(ν - L.νp) * L.n))
  end
  R[] = _Live{T}(L.n, L.compact, L.s, L.m, ν, φ, L.a2, L.ρmax)
  return nothing
end

# φ and the scale of its rounding error, from the live entries in the buffers
@noinline function _live_gap(c::LinearConstrained, x::AbstractArray, ν::T, γ, b, L::_Live{T}) where {T}
  X, A, G, I = _live_buffers(c, x, γ, T)
  φ, M = _live_pass(c.f, b, X, A, G, I, L.n, ν)
  φ + L.s, M + L.m
end

# the pass of _constraint_gap over the first n entries of the buffers, with the
# indices I in f
_live_pass(f, b, X, A, G, I, n, ν) =
  _constraint_gap((f = _Reindexed(f, I), set = (a = _head(A, n), b = b)), view(X, Base.OneTo(n)), ν, _head(G, n))
_head(g::_SearchSteps, n::Integer) = _with_steps(g, _head(g.steps, n))
_head(a::Real, ::Integer) = a
_head(a::AbstractVector, n::Integer) = view(a, Base.OneTo(n))

# f with the entry j at the index I[j]
struct _Reindexed{F, V}
  f::F
  I::V
end
Base.@propagate_inbounds prox_entry(g::_Reindexed, v::Real, j::Integer, γ::Real) = prox_entry(g.f, v, g.I[j], γ)

# fixes the entries at the bottom (low) or the top of their ranges at ν when a
# sample shows that few stay live; if it does not, the next sample waits for a
# secant half as steep as this one, ρ
@noinline function _try_fix(c::LinearConstrained, x::AbstractArray, ν::T, γ, L::_Live{T}, low::Bool, ρ::T) where {T}
  set = c.set
  if _isscaled(set)
    return _try_fix(c, _normalized_normal(set.a, set.s), x, ν, γ, L, low, ρ)
  end
  _try_fix(c, set.a, x, ν, γ, L, low, ρ)
end

function _try_fix(c::LinearConstrained, a, x::AbstractArray, ν::T, γ, L::_Live{T}, low::Bool, ρ::T) where {T}
  few = L.compact ? _few_stay(c, _live_buffers(c, x, γ, T)..., Base.OneTo(L.n), ν, low) :
                    _few_stay(c, x, a, γ, nothing, eachindex(IndexLinear(), x), ν, low)
  few || return _Live{T}(L.n, L.compact, L.s, L.m, L.νp, L.φp, L.a2, ρ / 2)
  X, A, G, I = _live_buffers(c, x, γ, T)
  w, s, m = L.compact ? _fix!(c, a, X, A, G, I, X, A, G, I, Base.OneTo(L.n), ν, low) :
                        _fix!(c, a, X, A, G, I, x, a, γ, nothing, eachindex(IndexLinear(), x), ν, low)
  _Live{T}(w, true, L.s + s, L.m + m, L.νp, L.φp, L.a2, T(_FIX_RHO))
end

function _live_buffers(c::LinearConstrained, x::AbstractArray, γ, ::Type{T}) where {T}
  bx, ba, bg, bi = c.live
  n = length(x)
  a = c.set.a
  X = _buffer!(bx, promote_type(eltype(x), T), n)
  A = a isa Real ? _scaled_normal(c.set) : _buffer!(ba, T, n)
  G = _with_steps(γ, γ.steps isa Real ? γ.steps : _buffer!(bg, T, n))
  X, A, G, _buffer!(bi, Int, n)
end

# whether zᵢ, the value at the index i with the normal aᵢ, sits at the bottom
# (low) or the top of the range of aᵢzᵢ
Base.@propagate_inbounds function _at_end(f, z, aᵢ, i, low::Bool)
  l, u = _prox_range(f, i, typeof(z))
  z == ifelse((aᵢ > 0) == low, l, u)
end

# whether entry j of (Xs, As, Gs, Is) sits at an end at ν
Base.@propagate_inbounds function _at_end(f, Xs, As, Gs, Is, j, ν::T, low::Bool) where {T}
  i = Is === nothing ? j : Is[j]
  aᵢ = convert(T, _entry(As, j))
  γᵢ = convert(T, _entry(Gs, j))
  _at_end(f, prox_entry(f, Xs[j] - _ratio(Gs, j, T) * ν * aᵢ, i, γᵢ), aᵢ, i, low)
end

# the scalar proximal operator of f where it sits at an end, and 0 elsewhere:
# through the pass, the sums of aᵢzᵢ and |aᵢzᵢ| over the entries that it fixes
struct _AtEnd{F, A}
  f::F
  a::A          # the normal, by index
  low::Bool
end
Base.@propagate_inbounds function prox_entry(g::_AtEnd, v::Real, i::Integer, γ::Real)
  z = prox_entry(g.f, v, i, γ)
  ifelse(_at_end(g.f, z, _entry(g.a, i), i, g.low), z, zero(z))
end

# whether at most a share _FIX_KEEP of an evenly spaced sample of the live
# entries r stays live
function _few_stay(c::LinearConstrained, Xs, As, Gs, Is, r, ν::T, low::Bool) where {T}
  n = length(r)
  step = max(1, n ÷ _FIX_SAMPLE)
  stay = total = 0
  @inbounds for p in 1:step:n
    stay += !_at_end(c.f, Xs, As, Gs, Is, r[p], ν, low)
    total += 1
  end
  stay <= _FIX_KEEP * total
end

# copies the entries r of (Xs, As, Gs, Is) that stay live at ν to the front of
# (X, A, G, I), in order, in place when they are the same; their number, and
# the sums of aᵢzᵢ and |aᵢzᵢ| over the others, which sit at the bottom (low) or
# the top of their ranges
function _fix!(c::LinearConstrained{F}, a, X, A, G, I, Xs, As, Gs, Is, r, ν::T, low::Bool) where {F, T}
  f = c.f
  B = _buffer!(c.live[5], UInt64, cld(length(r), 64))
  _keep_bits!(B, f, Xs, As, Gs, Is, r, ν, low)
  if a isa Real && _has_uniform_range(F)     # every fixed entry has the same aᵢzᵢ
    w = _copy_live!(X, A, G, I, B, Xs, As, Gs, Is, r, T)
    l, u = _prox_range(f, 1, T)
    aT = convert(T, a)
    t = aT * ifelse((aT > 0) == low, l, u)
    return w, (length(r) - w) * t, (length(r) - w) * abs(t)
  end
  g = _AtEnd(f, a, low)
  s, m = Is === nothing ? _constraint_gap((f = g, set = (a = As, b = zero(T))), Xs, ν, Gs) :
                          _live_pass(g, zero(T), Xs, As, Gs, Is, length(r), ν)
  _copy_live!(X, A, G, I, B, Xs, As, Gs, Is, r, T), s, m
end

# the entries r as bits, 64 to a word, set where an entry stays live (the
# inner loop on one line, which coverage counts as run)
function _keep_bits!(B::Vector{UInt64}, f, Xs, As, Gs, Is, r, ν::T, low::Bool) where {T}
  n, j0 = length(r), first(r) - 1
  @inbounds for k in 1:cld(n, 64)
    o = j0 + 64(k - 1)
    w = zero(UInt64)
    @simd for q in 1:min(64, n - o + j0); w |= _live_bit(f, Xs, As, Gs, Is, o, q, ν, low); end
    B[k] = w
  end
  return B
end
Base.@propagate_inbounds _live_bit(f, Xs, As, Gs, Is, o, q, ν, low) =
  UInt64(!_at_end(f, Xs, As, Gs, Is, o + q, ν, low)) << (q - 1)

# copies the entries r of (Xs, As, Gs, Is) whose bit is set to the front of
# (X, A, G, I), in order, in place when they are the same; their number
function _copy_live!(X, A, G, I, B::Vector{UInt64}, Xs, As, Gs, Is, r, ::Type{T}) where {T}
  w = 0
  j0 = first(r) - 1
  @inbounds for k in 1:cld(length(r), 64)
    keep = B[k]
    while keep != 0
      j = j0 + 64(k - 1) + trailing_zeros(keep) + 1
      w += 1
      X[w] = Xs[j]
      I[w] = Is === nothing ? j : Is[j]
      A isa AbstractArray && (A[w] = convert(T, As[j]))
      G.steps isa AbstractArray && (G.steps[w] = convert(T, _entry(Gs, j)))
      keep &= keep - 1
    end
  end
  return w
end

# The range [L, U] of the scalar proximal operator, in the type S of its value,
# with ±Inf for an open end; _has_prox_range is true where an end may be finite.
# prox_entry must return the end itself where it reaches it, and be
# nondecreasing in v as computed.
_has_prox_range(::Type{<:AbstractFunction}) = false
for F in (:IndBox, :IndNonnegative, :NormL1Nonneg, :BoxConstrained)
  @eval _has_prox_range(::Type{<:$F}) = true
end
_has_prox_range(::Type{<:Scaled{F}}) where {F} = _has_prox_range(F)
_has_prox_range(::Type{<:Tilt{F}}) where {F} = _has_prox_range(F)
_has_prox_range(::Type{<:Translate{F}}) where {F} = _has_prox_range(F)

# whether the range is the same for every entry
_has_uniform_range(::Type{<:AbstractFunction}) = false
for F in (:ZeroFunction, :LinearFunction, :SqrNormL2, :NormL1, :ElasticNet, :IndNonnegative, :NormL1Nonneg)
  @eval _has_uniform_range(::Type{<:$F}) = true
end
_has_uniform_range(::Type{<:IndBox{<:Real, <:Real}}) = true
_has_uniform_range(::Type{<:Scaled{F}}) where {F} = _has_uniform_range(F)
_has_uniform_range(::Type{<:Tilt{F}}) where {F} = _has_uniform_range(F)
_has_uniform_range(::Type{<:BoxConstrained{F, B}}) where {F, B} = _has_uniform_range(F) && _has_uniform_range(B)

_prox_range(::AbstractFunction, ::Integer, ::Type{S}) where {S} = (S(-Inf), S(Inf))
_prox_range(::Union{IndNonnegative, NormL1Nonneg}, ::Integer, ::Type{S}) where {S} = (zero(S), S(Inf))
Base.@propagate_inbounds _prox_range(f::IndBox, i::Integer, ::Type{S}) where {S} =
  (convert(S, _entry(f.l, i)), convert(S, _entry(f.u, i)))
Base.@propagate_inbounds function _prox_range(c::BoxConstrained, i::Integer, ::Type{S}) where {S}
  l, u = _prox_range(c.f, i, S)
  bl, bu = _prox_range(c.box, i, S)
  clamp(l, bl, bu), clamp(u, bl, bu)
end
Base.@propagate_inbounds _prox_range(s::Scaled, i::Integer, ::Type{S}) where {S} = _prox_range(s.f, i, S)
Base.@propagate_inbounds _prox_range(t::Tilt, i::Integer, ::Type{S}) where {S} = _prox_range(t.f, i, S)
Base.@propagate_inbounds function _prox_range(t::Translate, i::Integer, ::Type{S}) where {S}
  l, u = _prox_range(t.f, i, S)
  t.b[i] + l, t.b[i] + u
end

####################################
#  the support function of a box with a linear constraint
####################################

# The value of the conjugate of the indicator of {x : l ≤ x ≤ u, a Σᵢ xᵢ = b}, or
# a Σᵢ xᵢ ≤ b, for scalar l, u and a: the support function of the set. A
# halfspace with a < 0 bounds the sum from below, and σ_C(y) = σ_{-C}(-y) turns it
# into one that bounds it from above. With x = l + (u - l)w, the support function
# is l Σᵢ yᵢ plus u - l times the largest ⟨y, w⟩ over 0 ≤ w ≤ 1 with Σᵢ wᵢ = κ:
# the sum of the ⌊κ⌋ largest yᵢ and the fraction κ - ⌊κ⌋ of the next one. Under
# Σᵢ wᵢ ≤ κ, it is the same sum of the positive parts. For the capped simplex it
# is the sum of the k largest entries, and with the halfspace on the singular
# values the Ky Fan k-norm.
_constant_normal(a::Real) = a
function _constant_normal(a::AbstractArray)
  isempty(a) && return nothing
  v = first(a)
  all(==(v), a) ? v : nothing
end

function _conjugate_value(c::LinearConstrained{<:IndBox{<:Real, <:Real}, <:Union{IndHyperplane, IndHalfspace}},
                          y::AbstractArray)
  _checkconjugate(c, y)
  T = float(eltype(y))
  box, set = c.f, c.set
  a = _constant_normal(set.a)
  n = length(y)
  upper = set isa IndHyperplane || a > 0
  s = upper ? one(T) : -one(T)
  l, u = upper ? (convert(T, box.l), convert(T, box.u)) : (-convert(T, box.u), -convert(T, box.l))
  β = _isscaled(set) ? s * convert(T, set.b / float(a)) : s * convert(T, set.b) / convert(T, a)
  v = _buffer!(c.scratch, T, n)
  total = zero(T)
  @inbounds for (i, yi) in enumerate(y)
    z = s * convert(T, yi)
    total += z
    v[i] = set isa IndHyperplane ? z : max(z, zero(T))
  end
  m = β - n * l                                   # the sum of w, times u - l
  iszero(u - l) && return l * total
  κ = clamp(m / (u - l), zero(T), convert(T, n))
  return l * total + (u - l) * _top_sum!(view(v, 1:n), κ)
end

# the checks of the conjugate value above, before it computes
function _checkconjugate(c::LinearConstrained{<:IndBox{<:Real, <:Real}, <:Union{IndHyperplane, IndHalfspace}}, y::AbstractArray)
  _checkparams(c, y)
  T = float(eltype(y))
  box, set = c.f, c.set
  a = _constant_normal(set.a)
  a === nothing && throw(ArgumentError("the value of the conjugate of a box with a linear constraint needs a constant normal"))
  n = length(y)
  upper = set isa IndHyperplane || a > 0
  s = upper ? one(T) : -one(T)
  l, u = upper ? (convert(T, box.l), convert(T, box.u)) : (-convert(T, box.u), -convert(T, box.l))
  isfinite(l) && isfinite(u) ||
    throw(ArgumentError("the value of the conjugate of Constrained(IndBox(l, u), $(nameof(typeof(set)))(a, b)) needs finite bounds"))
  β = _isscaled(set) ? s * convert(T, set.b / float(a)) : s * convert(T, set.b) / convert(T, a)
  slack = _tol(T) * max(one(T), abs(β), n * abs(l))
  m = β - n * l
  (m < -slack || (set isa IndHyperplane && m > n * (u - l) + slack)) && _infeasible(c)
  true
end

_has_conjugate_value(c::LinearConstrained{<:IndBox{<:Real, <:Real}, <:Union{IndHyperplane, IndHalfspace}}) =
  isfinite(c.f.l) && isfinite(c.f.u) && _constant_normal(c.set.a) !== nothing

# the sum of the ⌊κ⌋ largest entries of v and the fraction κ - ⌊κ⌋ of the next
# one, for 0 ≤ κ ≤ length(v); reorders v
function _top_sum!(v::AbstractVector{T}, κ::T) where {T}
  n = length(v)
  k = min(floor(Int, κ), n)
  f = κ - k
  j = f > 0 && k < n ? k + 1 : k
  j == 0 && return zero(T)
  j < n && _kth_largest!(v, j)
  t = zero(T)
  @inbounds for i in 1:k
    t += v[i]
  end
  return j > k ? t + f * v[j] : t
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
  prox_entry(f, v, i, γ), first(_ridge_weights(S(γ), S(f.λ)))
end
Base.@propagate_inbounds function _prox_entry_slope(f::NormL1, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  t = convert(S, _weight_product(γ, f.λ0, _entry(f.w, i)))
  shrink(v, t), _slope(abs(v) > t, one(S))
end
Base.@propagate_inbounds function _prox_entry_slope(f::NormL1Nonneg, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  t = convert(S, _weight_product(γ, f.λ0, _entry(f.w, i)))
  max(v - t, zero(S)), _slope(v > t, one(S))
end
Base.@propagate_inbounds function _prox_entry_slope(f::ElasticNet, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  t = convert(S, _weight_coefficient(γ, f.λ1, _entry(f.w, i)))
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

function _checkstep(op::O, c::LinearConstrained, y::AbstractArray, x::AbstractArray, γ::Real) where {O}
  _search_steps(γ, float(eltype(y)))
  f, set = c.f, c.set
  if isempty(x)
    if !(set isa IndHyperslab)
      T = float(eltype(y))
      φ = -convert(T, _scaled_offset(set))
      if set isa IndHyperplane || φ > 0
        _resolved(φ, abs(φ)) || _infeasible(c)
      end
    end
    return true
  end
  f isa IndBox && f.l isa Real && f.u isa Real && set.a isa Real || return true
  T = float(eltype(y))
  T <: Union{Float32, Float64} && eltype(x) <: Union{Float32, Float64} || return true
  step = convert(T, γ)
  isfinite(step) && step > 0 || return true
  a = convert(T, _scaled_normal(set))
  l, u = convert(T, f.l), convert(T, f.u)
  isfinite(a) && !iszero(a) && isfinite(l) && isfinite(u) || return true
  n = convert(T, length(x))
  lo, hi = n * min(a * l, a * u), n * max(a * l, a * u)
  if set isa IndHyperslab
    lower, upper = convert(T, _scaled_offset(set, set.l)), convert(T, _scaled_offset(set, set.u))
    b = hi < lower ? lower : lo > upper ? upper : zero(T)
    gap = max(lower - hi, lo - upper)
  else
    b = convert(T, _scaled_offset(set))
    gap = set isa IndHalfspace ? lo - b : max(lo - b, b - hi)
  end
  M = n * abs(a) * max(abs(l), abs(u)) + abs(b)
  isfinite(M) && M < floatmax(T) / 16 || return true
  # Cover accumulation, endpoint rounding and the search's acceptance threshold.
  r = 32eps(T) * max(n, one(T))
  r < T(0.25) || return true
  gap > r * max(M, floatmin(T)) || return true
  _step_values(x) || return true
  for v in x
    isfinite(v) || return true
  end
  _infeasible(c)
end
