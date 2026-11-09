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
from the multiplier of the previous call, which the function keeps, with a
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

The caller must ensure that the set meets the domain of `f`. For a convex
function on a box, the method clamps the unconstrained proximal point even
when the intersection is empty. Neither that point nor the returned value
shows that the intersection is nonempty; `value_and_prox!` returns zero for an indicator
even in this improper case. The exception is a [`NormL1`](@ref) with a stored infinite level
`λ0 = Inf`: at a positive step it fixes every entry with a positive weight at zero, so a
box that excludes zero at such an entry raises an `ArgumentError` ("the IndBox misses the
domain of NormL1") before anything is written. A zero weight leaves its entry free, a
zero or non-positive step is not decided here, and other functions and wrappers are not
checked. The standalone value of an infeasible function at an argument stays `Inf`.

**Precision of the multiplier search.** On a hyperplane, halfspace, hyperslab or simplex, a
call has a *model precision* `W`: the widest of the destination, the argument and the data, the
data being those of `f` and the normal and offsets of the set. Levels, weights, steps and the bounds of a box are
parameters, rounded once to `W`, after being combined (numerics T2 and T3). A normal that
is a fixed-width integer scalar, such as the `1` of `IndHyperplane(1, b)` or of
`IndSimplex`, does not raise `W` when the precision given by the destination, the
argument, the data of `f` and the offsets of the set holds it exactly; the offsets are
never exempt, so `IndHyperplane(1, 0)` is a `Float64` model. An integer that this
precision cannot hold, an array of integers and a `BigInt` follow the general rule.

In `Float64` the search stops where the residual `⟨a, z⟩ - b` is at most
`2 eps(Float64)(Σᵢ|aᵢzᵢ| + |b|)`, as in earlier releases, and accepts that point. That is one of its exits: where
the bracket collapses or the iteration limit is reached it returns its best point without that condition. It raises an
`ArgumentError` if it cannot bracket a multiplier, which can be a failure of the bracketing and not a proof that the
constraint is infeasible. Certifying `Float64` searches is not provided.

In `Float32` the same stop can leave the multiplier, and so the output, far from the
projection where the constraint function is flat (about 500 `eps(Float32)` in issue 151).
The search therefore *checks* its result before writing. From the residual `φ̂` at the point
found, the slope `s` of `φ` there and an assumed noise floor of the evaluation,
`2 eps(Float32)·m` with `m = Σᵢ|aᵢzᵢ| + |b|`, the check estimates the distance of the output
from the exact point, in the norm of `Diagonal(1 ./ γ)`, as `max(|φ̂|, noise)/√s`. The slope is the
derivative of `φ` at the point, where the pass computed it, or the slope of the secant through the last two points
of the search; a search that ends at its first point without a derivative there has no slope, and the check does
not accept it. The check accepts the result when the estimate is at most
`16 eps(Float32)·(m/(2√Σᵢγᵢaᵢ²) + min(|ν̂|√s, ‖x‖))`, where `ν̂` is the multiplier found and `‖x‖` the norm of the
argument in the same norm. The bracket is a scale estimate of the size of the answer, not a bound on the norm of the
computed point: `m/(2√Σᵢγᵢaᵢ²)` bounds from below the norm of a point that satisfies the constraint, and the point found
satisfies it only up to the residual. The estimate needs the sums `Σᵢγᵢ` and `Σᵢγᵢaᵢ²` of the steps rounded to
`Float32`: for a scalar step no pass, for an array of steps two passes over the steps, the second also over an array
normal. The cap `‖x‖` is a further read of the argument, in blocks and only as far as the comparison needs, where the
first term alone does not accept and the uncapped check does.

The sums of the evaluation of `φ` are accumulated in blocks of 1024 terms. For a scalar step the terms of a block
are summed in `Float32` lanes, whatever the function. For an array of steps they are summed in `Float64` lanes,
except for [`IndBox`](@ref), [`IndNonnegative`](@ref), a [`Scaled`](@ref) or a box on one of them (the simplex
routes) and the ordinary route of [`SqrNormL2`](@ref). The `Float32` lanes of those were measured for this change, with
steps stored as `Float32` arrays, to stay within `1.6 eps(Float32)·m` of the exact sum, and the other kernels measured were
off by up to `77 eps(Float32)·m` in `Float32` lanes. The rule selects by the kind of function and the form of the steps;
it does not look at the code that a function compiles to, and a function of your own sums in `Float64` lanes with array steps.

A result that fails the check is searched again from zero if the search started from the previous multiplier, and then
in `Float64` on the `Float32` problem: the entry maps are evaluated in `Float64` on the exactly widened argument with
their parameters rounded to `Float32` as the `Float32` kernel rounds them, and the output is formed in `Float64` and
rounded once, so for these functions the problem is the same one. That recovery exists for
[`NormL1`](@ref), [`NormL1Nonneg`](@ref), [`ElasticNet`](@ref), [`IndBox`](@ref), [`IndNonnegative`](@ref),
[`SqrNormL2`](@ref) (with no center, or a center of `Float16`, `Float32` or `Float64` entries that keep their range on
rounding to `Float32`, which the kernel rounds to `Float32`), a box on one of them, and a [`Scaled`](@ref) of one of
them by a `Float32`, `Float16` or fixed-width integer factor. For any other function (a [`Tilt`](@ref), a
[`Translate`](@ref), a [`LinearFunction`](@ref), a [`Conjugate`](@ref), a `Scaled` by a wider factor or a `BigInt`, a
ridge with another center, or a function of your own) it is refused, and a result that fails the check raises a
[`PrecisionError`](@ref) before the destination is written.

This is a check. It is not a certificate, a conservative error bound or a guaranteed
forward error. The noise floor is a constant chosen from measurements and the slope is
that of a secant or a derivative at the point, which can cross a kink of the constraint function, so an accepted
result can exceed `16 eps(Float32)`; on the random problems up to `n = 100 000` measured
for this change, by a factor of about 1.2 at most. An accepted result can in principle be
inaccurate. `16 eps(Float32)`, in the 2-norm against a `BigFloat` reference of the `Float32`
problem, is the target of the package's regression tests for specified fixtures, not a
guarantee for every call. The policy is the exception to A3 in the numerical contract
(`docs/src/numerics.md`).

**Failures.** A call that raises from the search for the multiplier (a `PrecisionError`, or an `ArgumentError` that the set
misses the domain) writes nothing to its destination and leaves the function's multiplier and slope as they were. That is a promise about the single call: in a sum over blocks (`SeparableSum`) the blocks before the one
that raises may have been written, as for any failure found by computing (numerics O6; the general case is open,
[#323](https://github.com/mlakolar/ProximalBase.jl/issues/323)). A `PrecisionError` means that the result could not be
accepted in the model precision, as above. The `Float32` search raises an `ArgumentError` that the set "misses the domain" of `f`
only where that is established independently of the search: the range of `⟨a, z⟩` over the ranges of the
entries of `f`, with a margin that covers the rounding of its sums, lies strictly on the wrong side of the offset, or
lies entirely outside the hyperslab, each finite end of the range being bounded with the scale of its own terms. A
failure to bracket in `Float32`, or in its recovery, does not prove it. A `Float64` search raises it when it cannot
bracket a multiplier, which is not a proof that none exists.
The error is not a general test for an empty intersection either: rounding may put a
point on the closure of an open domain, and a finite multiplier may not exist when the
set meets only the boundary of a closed domain. Empty arguments that fail the search's own
acceptance test are rejected before writing. Block oracles also reject clearly infeasible
scalar boxes with scalar linear normals at finite positive scalar steps, when finite
inputs and conservative rounding bounds show that the search would find no multiplier.
Overflowing ranges and transformed inputs remain unchecked.
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
  last::Base.RefValue{Float64}      # the multiplier for the normalized constraint
  slope::Base.RefValue{Float64}     # the slope of φ at that multiplier, where it had one: a seed for the next search
  scratch::Scratch                  # for the value of the conjugate
  live::NTuple{6, Scratch}          # the entries the search still evaluates: x, a, γ, index; bits; state

  LinearConstrained{F, H}(f::F, set::H) where {F<:AbstractFunction, H<:Union{IndHyperplane, IndHalfspace, IndHyperslab}} =
    (_check_constrainable(f); new{F, H}(f, set, Ref(NaN), Ref(NaN), Scratch(), ntuple(_ -> Scratch(), 6)))
  # the same set and state under another function, for the recovery: no new state, no allocation
  LinearConstrained{F, H}(f::F, c::LinearConstrained) where {F<:AbstractFunction, H<:Union{IndHyperplane, IndHalfspace, IndHyperslab}} =
    new{F, H}(f, c.set, c.last, c.slope, c.scratch, c.live)
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
_counts_nonzeros(f::Union{BoxConstrained, LinearConstrained}) = _counts_nonzeros(f.f)
_scale_invariant(f::BoxConstrained) = _scale_invariant(f.f) && _cone_bound(f.box.l) && _cone_bound(f.box.u)

####################################
#  on a box
####################################

(c::BoxConstrained)(x::_ArrayOrOwner) =
  (_check_value_arrays(c, x); _value_checked(c, x))
_value_unchecked(c::BoxConstrained, x::_ArrayOrOwner) =
  (_checkparams(c, x); T = float(eltype(x)); iszero(_value_unchecked(c.box, x)) ? convert(T, _value_unchecked(c.f, x)) : T(Inf))
# MCP, SCAD and NormL0 on a box: a NaN entry gives NaN where the value depends
# on it (constrained_nonconvex.jl).
_value_unchecked(c::BoxConstrained{<:Union{MCP, SCAD, NormL0}}, x::_ArrayOrOwner) =
  (_checkparams(c, x); T = float(eltype(x)); iszero(_value_unchecked(c.box, x)) ? convert(T, _value_unchecked(c.f, x)) : _box_nan_value(c, x, T))

# the scalar problem is convex, so its minimizer on an interval is the nearest
# point to its unconstrained minimizer
Base.@propagate_inbounds prox_entry(c::BoxConstrained, v::Real, i::Integer, γ::Real) =
  prox_entry(c.box, prox_entry(c.f, v, i, γ), i, γ)

# γ is a scalar step or an array of steps, one per entry
prox!(y::_ArrayOrOwner, c::BoxConstrained, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
_prox_unchecked!(y::_ArrayOrOwner, c::BoxConstrained, x::_ArrayOrOwner, γ::Real) = (_checkwrite(prox!, c, y, x, γ); _prox_sep!(y, c, x, γ))
prox!(y::AbstractArray, c::BoxConstrained, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
_prox_unchecked!(y::AbstractArray, c::BoxConstrained, x::AbstractArray, γ::AbstractArray) = (_checkwrite(prox!, c, y, x, γ); _prox_sep!(y, c, x, γ))
value_and_prox!(y::_ArrayOrOwner, c::BoxConstrained, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
_value_and_prox_unchecked!(y::_ArrayOrOwner, c::BoxConstrained, x::_ArrayOrOwner, γ::Real) = _value_and_prox_box!(y, c, x, γ)
value_and_prox!(y::AbstractArray, c::BoxConstrained, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, c::BoxConstrained, x::AbstractArray, γ::AbstractArray) = _value_and_prox_box!(y, c, x, γ)
function _value_and_prox_box!(y, c, x, γ)
  _checkwrite(value_and_prox!, c, y, x, γ)
  _prox_sep!(y, c, x, γ)
  T = float(eltype(y))
  _isindicator(c) ? zero(T) : convert(T, _value_unchecked(c.f, y))
end

_checkparams(c::BoxConstrained, x) = (_checkparams(c.f, x); _checkparams(c.box, x); true)
function _checkwrite(op::O, c::BoxConstrained, y::_ArrayOrOwner, x::_ArrayOrOwner) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(c, y); _checkreal(c, x)
  _checksize(y, x)
  _checkparams(c, x)
  op === value_and_prox! && !_isindicator(c) && _checkvalue(c.f, y)
  true
end
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
function _checkwrite(op::O, t::Tilt, y::AbstractArray, x::AbstractArray, γ) where {O}
  _checkreal(t, γ)
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
  _checkreal(t, γ)
  _checkwrite(op, t, y, x)
  if op === value_and_prox! && !_box_value_available(t.f, y)
    z = _array_buffer!(t.proxscratch, float(eltype(y)), size(y))
    z .= x .- t.b
    _checkwrite(op, t.f, y, z, γ)
  end
  true
end
function _checkwrite(op::O, s::SeparableSum, y::Tuple, x::Tuple, γ) where {O}
  _checkreal(s, y); _checkreal(s, x); _checkreal(s, γ)
  _checksize(y, x)
  _checkblocks(s, x)
  _checkwrite_blocks(op, s.fs, y, x, γ)
end
@generated _checkwrite_blocks(op, fs::Tuple, y::Tuple, x::Tuple, γ) =
  _blockwise(k -> :(_checkwrite(op, fs[$k], y[$k], x[$k], γ)), fieldcount(fs), true)

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
  iszero(_value_unchecked(c.box, p)) && _check_box_value(c.f, y, p)
  true
end
@inline function _checkwrite(op::O, c::BoxConstrained, y::AbstractArray, x::AbstractArray, γ) where {O}
  _checkreal(c, γ)
  _checkwrite(op, c, y, x)
  _check_box_domain(c, x, γ)
  if op === value_and_prox! && !_isindicator(c)
    _checkparam(γ, x)
    _check_box_value(c.f, y, _BoxProxArray(y, c, x, γ))
  end
  true
end

# The conjugate of an elastic net has no standalone value, but on a box the value at the stored output y is known
# in closed form: for f = ElasticNet(λ1, λ2, w) and λ2 > 0, f*(y) = Σᵢ max(|yᵢ| - λ1 wᵢ, 0)² / (2 λ2), the
# SqrNormL2(λ2) conjugate at the soft-thresholded y; for λ2 = 0, f* is the indicator of |yᵢ| ≤ λ1 wᵢ, the NormL1
# conjugate. The value is taken at the clamped proximal point y that is stored (`_prox_sep!` is not repeated), and
# only for the combinations below, decided from types before anything is written or indexed; every other
# combination keeps the generic method, which refuses before writing (`_checkvalue` of the conjugate has no value
# to offer). The route reads and writes with linear indices and recovers an overflowing or underflowing value
# through `Conjugate(SqrNormL2)`, which scales through arrays that know their size only, so an array is accepted
# only as itself, never through a view, a reshape or a wrapper, whose storage it would have to inspect:
#   * the output: an Array of Float32 or Float64, or a SparseIterate (both one-based);
#   * the input: an Array or a SparseIterate of a machine-number element type (Bool, an integer or an IEEE float);
#     the owner of a symmetric matrix (SymmetricSparseIterate) is not an array and is refused, as an output and as an input;
#   * the function: an ElasticNet declared with a machine-number type for its levels and with Nothing or a concrete
#     Array type for its weights, so that no abstract or union type reaches the working type; finite levels of
#     that type; weights absent, or an Array of a machine-number type, finite, nonnegative and not sharing memory
#     with the output (the value reads them after the output is written);
#   * bounds and steps: machine-number scalars, or Arrays of a machine-number type;
#   * the working type float(promote(output, level, weights, ridge)): Float32 or Float64.
_elastic_number(::Type{T}) where {T} = isconcretetype(T) && T <: Union{Base.IEEEFloat, Base.BitInteger, Bool}
_elastic_array(a) = a isa Array || a isa SparseIterate
_elastic_output(y) = _elastic_array(y) && (eltype(y) === Float32 || eltype(y) === Float64)
_elastic_operand(a) = _elastic_array(a) && _elastic_number(eltype(a))
_elastic_parameter(x::Real) = _elastic_number(typeof(x))
_elastic_parameter(a::Array) = _elastic_number(eltype(a))
_elastic_parameter(::Any) = false
_elastic_weights(::Nothing) = true
_elastic_weights(w::Array) = _elastic_number(eltype(w))
_elastic_weights(::Any) = false
# only the type parameters are read, so no `float` of a union or of an abstract type is ever taken
_elastic_declared(::ElasticNet{S, W}) where {S, W} = _elastic_number(S) && (W === Nothing || (W <: Array && isconcretetype(W)))
_elastic_working(::Type{T}, f) where {T} = (S = float(promote_type(_shrink_type(T, f), typeof(f.λ2))); S === Float32 || S === Float64)
_elastic_alias(y, ::Nothing) = false
_elastic_alias(y, w) = Base.mightalias(y, w)
_finite_nonneg(::Nothing) = true
_finite_nonneg(w::Array) = all(v -> isfinite(v) && v >= 0, w)
function _elastic_value_route(c::BoxConstrained{<:Conjugate{<:ElasticNet}}, y, x)
  f = c.f.f
  _elastic_output(y) && _elastic_operand(x) && _elastic_number(typeof(f.λ1)) && _elastic_number(typeof(f.λ2)) &&
    _elastic_declared(f) && _elastic_weights(f.w) && _elastic_parameter(c.box.l) && _elastic_parameter(c.box.u) &&
    _elastic_working(eltype(y), f) &&
    isfinite(f.λ1) && isfinite(f.λ2) && _finite_nonneg(f.w) && !_elastic_alias(y, f.w)
end
_elastic_value_route(c, y, x, γ) = _elastic_value_route(c, y, x) && _elastic_parameter(γ)
# the checks that the generic methods make, except that of the value
function _elastic_checks(c, y, x, γ)
  _checkreal(c, γ)
  _checkreal(c, y); _checkreal(c, x)
  _checksize(y, x)
  _checkparams(c, x)
  _check_box_domain(c, x, γ)
  _checkparam(γ, x)
  true
end

function _checkwrite(op::O, c::BoxConstrained{<:Conjugate{<:ElasticNet}}, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  if op === value_and_prox! && _elastic_value_route(c, y, x)
    _checkreal(c, y); _checkreal(c, x)
    _checksize(y, x)
    _checkparams(c, x)
    return true
  end
  invoke(_checkwrite, Tuple{O, BoxConstrained, AbstractArray, AbstractArray}, op, c, y, x)
end
@inline function _checkwrite(op::O, c::BoxConstrained{<:Conjugate{<:ElasticNet}}, y::AbstractArray, x::AbstractArray, γ) where {O}
  op === value_and_prox! && _elastic_value_route(c, y, x, γ) && return _elastic_checks(c, y, x, γ)
  # a refusal does what the generic method does for a value that is not available: the step is checked
  # as a number, then the arrays; the generic 5-argument method would reach the check above
  op === value_and_prox! && return (_checkreal(c, γ); invoke(_checkwrite, Tuple{O, BoxConstrained, AbstractArray, AbstractArray}, op, c, y, x))
  invoke(_checkwrite, Tuple{O, BoxConstrained, AbstractArray, AbstractArray, Any}, op, c, y, x, γ)
end
function _value_and_prox_box!(y, c::BoxConstrained{<:Conjugate{<:ElasticNet}}, x, γ)
  _elastic_value_route(c, y, x, γ) || return invoke(_value_and_prox_box!, Tuple{Any, BoxConstrained, Any, Any}, y, c, x, γ)
  _elastic_checks(c, y, x, γ)
  _prox_sep!(y, c, x, γ)
  _elastic_conjugate_value(c.f.f, y)
end

# For λ2 = 0 the conjugate is the indicator of |yᵢ| ≤ λ1 wᵢ, with the comparison of the NormL1 conjugate (`_within`: the
# relative tolerance of the output type, and a NaN entry is infeasible). The threshold is formed in the working type of the
# output, the level and the weights after both factors are converted, so that no product wraps or rounds in a narrower
# type; a threshold that overflows exceeds every finite entry, and an infinite entry is never feasible.
function _elastic_indicator_value(f::ElasticNet, y::AbstractArray)
  T = float(eltype(y))
  S = _shrink_type(T, f)
  λ = convert(S, f.λ1)
  @inbounds for i in eachindex(IndexLinear(), y)
    v = y[i]
    thr = λ * convert(S, _entry(f.w, i))
    (isinf(v) || !_within(abs(v), thr, T)) && return T(Inf)
  end
  zero(T)
end

function _elastic_conjugate_value(f::ElasticNet, y::AbstractArray)
  iszero(f.λ2) && return _elastic_indicator_value(f, y)
  z = _ElasticShrunkArray(f, y)
  T = float(eltype(y))
  # Everything up to the value is formed in the promoted type of y, λ1, w and λ2, and only the value
  # is rounded to the type of y: the residual or its square may not be representable in that type.
  S = float(promote_type(eltype(z), typeof(f.λ2)))
  # The sum of squares over 2λ2, with one rounding of the quotient, when no square or the quotient
  # leaves the normal range; the conjugate of SqrNormL2 recovers the other cases (and NaN, Inf).
  s = zero(S)
  @inbounds for i in eachindex(IndexLinear(), z)
    s += abs2(convert(S, z[i]))
  end
  q = s / (2 * convert(S, f.λ2))
  isfinite(s) && s >= sqrt(floatmin(S)) && floatmin(S) <= q <= floatmax(S) && return convert(T, q)
  convert(T, _conjugate_value(SqrNormL2(f.λ2), z))
end

# The entries max(|yᵢ| - λ1 wᵢ, 0), read lazily, in the wider of the types of y,
# λ1 and w (the threshold is zero for λ1 = 0 or wᵢ = 0). NaN and infinite entries are kept.
struct _ElasticShrunkArray{R, N, F<:ElasticNet, Y<:AbstractArray} <: AbstractArray{R, N}
  f::F
  y::Y
end
_shrink_type(::Type{T}, f::ElasticNet{S, Nothing}) where {T, S} = promote_type(T, float(S))
_shrink_type(::Type{T}, f::ElasticNet{S, <:AbstractArray}) where {T, S} = promote_type(T, float(S), float(eltype(f.w)))
function _ElasticShrunkArray(f::ElasticNet, y::AbstractArray{<:Real, N}) where {N}
  R = _shrink_type(float(eltype(y)), f)
  _ElasticShrunkArray{R, N, typeof(f), typeof(y)}(f, y)
end
Base.size(a::_ElasticShrunkArray) = size(a.y)
Base.IndexStyle(::Type{<:_ElasticShrunkArray}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(a::_ElasticShrunkArray{R}, i::Int) where {R}
  v = abs(convert(R, a.y[i]))
  t = _weight_coefficient(convert(R, a.f.λ1), convert(R, _entry(a.f.w, i)))
  (isnan(v) | isinf(v)) && return v
  v > t ? v - t : zero(R)
end

# An infinite level at a positive step fixes every penalized entry (w > 0) of NormL1 at zero, and the
# clamp of that zero into a box that excludes it is not feasible. Decided from the stored level, the
# weights, the step and the bounds as given, before anything is written; nothing is evaluated for
# another function, for a finite level, or for a box that contains zero. Other families are not decided.
_check_box_domain(::BoxConstrained, x, γ) = nothing
@inline function _check_box_domain(c::BoxConstrained{<:NormL1}, x, γ)
  isinf(c.f.λ0) && !c.box.zp && _check_box_fixed(c, x, γ)
  nothing
end
function _check_box_fixed(c::BoxConstrained{<:NormL1}, x, γ)
  f, box = c.f, c.box
  for i in eachindex(IndexLinear(), x)
    _l1_fixed(_entry(γ, i), f.λ0, _entry(f.w, i)) &&
      !((_entry(box.l, i) <= 0) & (_entry(box.u, i) >= 0)) && _box_misses_domain(f)
  end
  nothing
end
@noinline _box_misses_domain(f) = throw(ArgumentError("the IndBox misses the domain of $(nameof(typeof(f)))"))

# Membership can return Inf without evaluating the inner value.
_checkvalue(c::BoxConstrained, x::AbstractArray) = (_checkreal(c, x); _checkparams(c, x))
_checkvalue(c::LinearConstrained, x::AbstractArray) = (_checkreal(c, x); _checkvalue(c.set, x))
iszeropreserving(c::BoxConstrained) = iszeropreserving(c.f) && iszeropreserving(c.box)

# For a convex f with 0 in its domain, the proximal step of f from -gᵢ moves in
# the direction that improves on zero, if one does; the box must allow it.
function kkt_violation(c::BoxConstrained, g::AbstractArray, i::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
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

(c::LinearConstrained)(x::_ArrayOrOwner) =
  (_check_value_arrays(c, x); _value_checked(c, x))
_value_unchecked(c::LinearConstrained, x::_ArrayOrOwner) = (_checkparams(c, x); T = float(eltype(x)); iszero(_value_unchecked(c.set, x)) ? convert(T, _value_unchecked(c.f, x)) : T(Inf))

_checkparams(c::LinearConstrained, x) = (_checkparams(c.f, x); _checkparams(c.set, x); true)
function _checkwrite(op::O, c::LinearConstrained, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(c, y); _checkreal(c, x)
  _checksize(y, x)
  _checkparams(c, x)
  c.set isa IndHyperslab && _checkwrite(op, c.set, y, x)
  op === value_and_prox! && !_isindicator(c) && _checkvalue(c.f, y)
  true
end

# γ is a scalar step, or an array of steps, one per entry: the proximal operator
# in the metric of Diagonal(1 ./ γ), where z(ν) has the entries
# prox_{γᵢfᵢ}(xᵢ - γᵢνaᵢ)
# At an owner the parameters are checked before anything is written, which is all that is defined: the
# multiplier search runs over arrays
prox!(y::_ArrayOrOwner, c::LinearConstrained, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
_prox_unchecked!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::Real) =
  _linear_constrained_prox!(y, c, x, γ)
prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
_prox_unchecked!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::AbstractArray) =
  (_checkparam(γ, x); _check_metric_steps(c, γ); _linear_constrained_prox!(y, c, x, γ))
# the ridge checks its steps and decides its route in one pass (see _ridge_checked_steps),
# after the shapes: the pass reads the center without bounds checks
function _prox_unchecked!(y::AbstractArray, c::LinearConstrained{<:SqrNormL2}, x::AbstractArray, γ::AbstractArray)
  _checkparam(γ, x)
  _checkwrite(prox!, c, y, x)
  W = _linear_model_type(c, x, y)
  valid, ordinary = _ridge_checked_steps(c.f, γ, x, W)
  valid || _check_metric_steps(c, γ)           # raises its error
  ordinary ? _linear_constrained_prox!(y, c, x, _RidgeSteps(γ), W) : _linear_constrained_prox!(y, c, x, γ, W)
end

# The search, its check and the output run in the model precision W of the call. A scalar step is a
# parameter, rounded once to W; the steps of an array are converted per access.
function _linear_constrained_prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ)
  _checkwrite(prox!, c, y, x)
  W = _linear_model_type(c, x, y)
  s = γ isa Real ? convert(W, γ) : γ
  _ridge_ordinary_route(c.f, s, x, W) ? _linear_constrained_prox!(y, c, x, _RidgeSteps(s), W) :
                                        _linear_constrained_prox!(y, c, x, s, W)
end
# The destination is not written before the multiplier is accepted: by the check of the search for W = Float32, by the search otherwise.
function _linear_constrained_prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ, ::Type{W}) where {W}
  ν = _multiplier(c, x, γ, W)
  set = c.set
  # a hyperslab that holds the proximal point of f needs no correction, and none is formed: at an infinite step the
  # product γ ν a is Inf⋅0
  set isa IndHyperslab && ν isa Number && iszero(ν) && return _prox_sep!(y, c.f, x, _steps(γ))
  if _isscaled(set)
    return _linear_constrained_output!(y, c.f, _normalized_normal(set.a, set.s), x, γ, ν)
  end
  _linear_constrained_output!(y, c.f, set.a, x, γ, ν)
end

# The model precision W of a call of a LinearConstrained function (numerics T2, T3): the widest of the
# destination, the argument and the data, where the normal and the offsets or bounds of the set are data
# and the levels, bounds of boxes and steps are parameters. The one exception: a normal that is a
# fixed-width integer scalar does not raise W when it is exactly representable in the precision W0 that the
# destination, the argument, the data of the function and the offsets of the set give, as for `IndHyperplane(1, 100f0)`
# or the normal of an `IndSimplex` with a Float32 radius. The offsets are never exempt: `IndHyperplane(1, 0)`
# has a Float64 offset and is a Float64 model. An integer scalar that W0 cannot hold (2^24 + 1 in Float32), an
# array of integers and a BigInt follow the general rule. The check runs at every call, as W0 depends on the
# destination and the argument of the call.
function _linear_model_type(c::LinearConstrained, x::AbstractArray, y::AbstractArray)
  W = float(promote_type(eltype(y), eltype(x), _data_type(typeof(c))))
  a = c.set.a
  a isa Base.BitInteger || return W
  W0 = float(promote_type(eltype(y), eltype(x), _data_type(typeof(c.f)), _offset_type(c.set)))
  convert(W0, a) == a ? W0 : W
end
_offset_type(set::Union{IndHyperplane, IndHalfspace}) = typeof(set.b)
_offset_type(set::IndHyperslab) = typeof(set.l)

function _linear_constrained_output!(y, f, a, x, γ, ν::T) where {T}
  @inbounds for i in eachindex(IndexLinear(), y, x)
    γᵢ = convert(T, _entry(γ, i))
    y[i] = convert(eltype(y), prox_entry(f, x[i] - γᵢ * ν * convert(T, _entry(a, i)), i, γᵢ))
  end
  y
end

# A ridge whose entries are decided once per call. Where every γᵢλ, formed as
# prox_entry forms it for the arguments of the search, lies in the band of the ordinary
# coefficient (so none needs the scaled one of R3), and, for centers wider than the
# model precision W, every center entry keeps its range on rounding to W, the steps are
# wrapped, and every pass of the search and the output take the ordinary entry without
# a test, so that their loops have no call and vectorize. The decision is made at every
# call, as the steps may change between calls; anything else (a step outside the band,
# zero or not finite, a center entry that leaves W, another function or precision)
# keeps the tested entries. A scalar step is wrapped only for a wider center: otherwise
# its coefficient is already decided outside the loops. (A third route, for a wider
# center that leaves W somewhere, cost a homogeneous call on 8 entries about 40 ns.)
# The kernels index the steps and the centers alike, by the linear indices of x.
struct _RidgeSteps{G}
  γ::G
end
_steps(γ) = γ
_steps(s::_RidgeSteps) = s.γ

_ridge_ordinary_route(f, γ, x, ::Type) = false
_ridge_ordinary_route(f::SqrNormL2, γ::AbstractArray, x::AbstractArray, ::Type{T}) where {T} =
  last(_ridge_checked_steps(f, γ, x, T))
function _ridge_ordinary_route(f::SqrNormL2, γ::Real, x::AbstractArray, ::Type{T}) where {T}
  W, P, usable = _ridge_route_types(f, x, T)
  usable && _ridge_wide_center(f, W) && !_ridge_needs_scaling(W, convert(T, γ), f.λ) && _ridge_all_fit(W, f.y)
end
@inline _ridge_wide_center(f::SqrNormL2, ::Type{W}) where {W} = f.y !== nothing && _ridge_center_wider(W, eltype(f.y))
# W, the type of the arguments xᵢ - γᵢνaᵢ, P, the type of γᵢλ as _ridge_needs_scaling(W,
# γᵢ, λ) forms it, and whether the route applies to them
@inline function _ridge_route_types(f::SqrNormL2, x::AbstractArray, ::Type{T}) where {T}
  W = float(promote_type(eltype(x), T))
  P = promote_type(W, T, float(typeof(f.λ)))
  usable = W <: Union{Float32, Float64} && P <: Union{Float32, Float64} &&
           !(_ridge_wide_center(f, W) && !(eltype(f.y) <: Base.IEEEFloat))
  W, P, usable
end

# Per-entry steps: whether they are valid (as _check_metric_steps checks them) and take
# the route, in one pass, as every short loop before the search costs a call on few
# entries (a separate check and route cost one on 8 entries about 40 ns).
function _ridge_checked_steps(f::SqrNormL2, γ::AbstractArray, x::AbstractArray, ::Type{T}) where {T}
  W, P, usable = _ridge_route_types(f, x, T)
  usable || return _valid_steps(γ), false
  _ridge_wide_center(f, W) ? _ridge_scan(W, P, f.λ, γ, T, f.y) : _ridge_scan(W, P, f.λ, γ, T, nothing)
end
_valid_steps(γ) = (v = true; for g in γ; v &= (g > zero(g)) & isfinite(g); end; v)
@inline _ridge_center_fits(::Type, ::Nothing, i) = true
Base.@propagate_inbounds _ridge_center_fits(::Type{W}, c, i) where {W} = _ridge_fits(W, c[i])
function _ridge_scan(::Type{W}, ::Type{P}, λ, γ::AbstractArray, ::Type{T}, c) where {W, P, T}
  λ, lo, hi = P(λ), P(_ridge_band_low(W)), P(_ridge_band_high(W))
  valid, ok = true, true
  @inbounds for i in eachindex(IndexLinear(), γ)
    g = γ[i]
    valid &= (g > zero(g)) & isfinite(g)
    t = P(convert(T, g)) * λ
    ok &= (lo <= t) & (t <= hi) & _ridge_center_fits(W, c, i)
  end
  valid, valid & ok
end
_constraint_gap(f::SqrNormL2, a, b, x::AbstractArray, ν::T, s::_RidgeSteps, v::Val) where {T} =
  _constraint_gap(_OrdinaryRidge(f), a, b, x, ν, s.γ, v)
_linear_constrained_output!(y, f::SqrNormL2, a, x, s::_RidgeSteps, ν::T) where {T} =
  _linear_constrained_output!(y, _OrdinaryRidge(f), a, x, s.γ, ν)
_metric_sqnorm(set::Union{IndHyperplane, IndHalfspace, IndHyperslab}, x::AbstractArray, s::_RidgeSteps) =
  _metric_sqnorm(set, x, s.γ)

value_and_prox!(y::_ArrayOrOwner, c::LinearConstrained, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::Real) = _value_and_prox_linear!(y, c, x, γ)
value_and_prox!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, c::LinearConstrained, x::AbstractArray, γ::AbstractArray) = _value_and_prox_linear!(y, c, x, γ)
function _value_and_prox_linear!(y, c, x, γ)
  _checkwrite(value_and_prox!, c, y, x, γ)
  _prox_unchecked!(y, c, x, γ)
  T = float(eltype(y))
  _isindicator(c) ? zero(T) : convert(T, _value_unchecked(c.f, y))
end

# φ(ν) = ⟨a, z⟩ - b at z = z(ν), and Σᵢ |aᵢzᵢ| + |b|, the scale of its rounding
# error; with Val(true), also the slope of φ, -Σᵢ γᵢaᵢ² Pᵢ', where Pᵢ' is the
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

# The pass is summed in blocks of _GAP_BLOCK entries for a Float32 model: the terms of a block are formed in Float32 and
# summed in the lanes of the loop, and the sums of the blocks in Float64, so that the rounding of the sum is that of a block
# and not of n additions. The other precisions form one sum over the entries. φ and its scale are rounded once
# to T.
# The lanes of a block sum in Float32 for a scalar step, whatever the function. With an array of steps they sum in Float64
# unless the entry map is one of those that were measured to vectorize with Float32 lanes whose block sums stayed inside the
# noise floor of the check (_vector_lanes: IndBox, IndNonnegative and the ordinary ridge, and a Scaled or a BoxConstrained of one
# of them; off by at most 1.6 eps m on the measured problems, with steps stored as Float32 arrays). The other measured kernels
# did not vectorize with array steps: a block was one sequential sum of 1024 terms, off by up to 77 eps m (20 to 40 eps m for a
# box on an ℓ1 map); their terms, still formed in Float32 as the kernel forms them, are accumulated in Float64 (_gap_sums_wide),
# which took about 1.1 times the time of a pass of those maps in the timing of this change. The rule selects by the kind of
# function and the form of the steps; it does not examine the code that an entry map compiles to, so a function of your own,
# which is not in the list, sums in Float64 with array steps.
const _GAP_BLOCK = 1024

_vector_lanes(f) = false
_vector_lanes(::Union{IndBox, IndNonnegative, _OrdinaryRidge}) = true
_vector_lanes(f::Scaled) = _vector_lanes(f.f)
_vector_lanes(f::BoxConstrained) = _vector_lanes(f.f)
_wide_lanes(f, γ) = false
_wide_lanes(f, γ::AbstractArray) = !_vector_lanes(f)

function _gap_sums(f, a, x::AbstractArray, ν::T, γ, r, ::Val{S}) where {T, S}
  s, m, d = zero(T), zero(T), zero(T)
  @inbounds @simd for i in r
    aᵢ = convert(T, _entry(a, i))
    γᵢ = convert(T, _entry(γ, i))
    v = x[i] - γᵢ * ν * aᵢ
    z, dz = S ? _prox_entry_slope(f, v, i, γᵢ) : (prox_entry(f, v, i, γᵢ), zero(T))
    t = aᵢ * convert(T, z)
    s += t
    m += abs(t)
    S && (d += γᵢ * aᵢ * aᵢ * convert(T, dz))
  end
  s, m, d
end

# the same terms, summed in Float64 lanes
function _gap_sums_wide(f, a, x::AbstractArray, ν::T, γ, r, ::Val{S}) where {T, S}
  s, m, d = 0.0, 0.0, zero(T)
  @inbounds @simd for i in r
    aᵢ = convert(T, _entry(a, i))
    γᵢ = convert(T, _entry(γ, i))
    v = x[i] - γᵢ * ν * aᵢ
    z, dz = S ? _prox_entry_slope(f, v, i, γᵢ) : (prox_entry(f, v, i, γᵢ), zero(T))
    t = aᵢ * convert(T, z)
    s += Float64(t)
    m += Float64(abs(t))
    S && (d += γᵢ * aᵢ * aᵢ * convert(T, dz))
  end
  s, m, d
end

function _gap_sums(f, a, x::AbstractArray, ν::Float32, γ, ::Val{S}) where {S}
  idx = eachindex(IndexLinear(), x)
  wide = _wide_lanes(f, γ)
  s, m, d = 0.0, 0.0, 0.0f0
  @inbounds for lo in first(idx):_GAP_BLOCK:last(idx)
    r = lo:min(lo + (_GAP_BLOCK - 1), last(idx))
    sb, mb, db = wide ? _gap_sums_wide(f, a, x, ν, γ, r, Val(S)) : _gap_sums(f, a, x, ν, γ, r, Val(S))
    s += sb
    m += mb
    d += db
  end
  s, m, d
end
_gap_sums(f, a, x::AbstractArray, ν::T, γ, v::Val) where {T} = _gap_sums(f, a, x, ν, γ, eachindex(IndexLinear(), x), v)

function _constraint_gap(f, a, b, x::AbstractArray, ν::T, γ, ::Val{S}) where {T, S}
  s, m, d = _gap_sums(f, a, x, ν, γ, Val(S))
  A = T === Float32 ? Float64 : T
  b = convert(A, b)
  φ, M = convert(T, s - b), convert(T, m + abs(b))
  S ? (φ, M, -d) : (φ, M)
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

const _MAX_BRACKET_STEPS = 64
const _MAX_SECANTS = 100

@noinline _infeasible(c::LinearConstrained) =
  throw(ArgumentError("the $(nameof(typeof(c.set))) misses the domain of $(nameof(typeof(c.f)))"))

# φ and the scale of its rounding error at ν, and its slope when the scalar
# proximal operators of f have known slopes, NaN otherwise
_gap_and_slope(c::LinearConstrained{F}, x::AbstractArray, ν::T, γ, b) where {F, T} =
  _gap_and_slope(c, x, ν, γ, b, Val(_has_prox_slope(F)))
_gap_and_slope(c::LinearConstrained, x::AbstractArray, ν::T, γ, b, ::Val{true}) where {T} =
  _constraint_gap(c, x, ν, γ, b, Val(true))
_gap_and_slope(c::LinearConstrained, x::AbstractArray, ν::T, γ, b, ::Val{false}) where {T} =
  ((φ, m) = _constraint_gap(c, x, ν, γ, b); (φ, m, T(NaN)))

# What the search proposes: a multiplier ν, the slope of φ at it (a derivative there or a secant, NaN where it has none),
# the residual there and, where no bracket was found, the direction that was tried. The search sets neither the
# multiplier nor the slope of the function and writes nothing to the output (its scratch buffers are used before the
# proposal is accepted); the multiplier and the slope are committed by _multiplier, after the check for W = Float32 and
# right after the search otherwise.
struct _Proposal{T}
  ν::T                # NaN where the residual was NaN
  slope::Float64      # of φ at ν: a derivative or the last secant; NaN where there is none (Float32), or the seed (the other precisions)
  res::Float64        # |φ| at ν, NaN where unknown
  dir::Int8           # ±1: no bracket in that direction, and ν is the last point tried; 0 otherwise
  warm::Bool          # whether the search started from the multiplier of the last call
  m::Float64          # Σᵢ |aᵢzᵢ| + |b| at ν, the scale of the rounding of φ there, NaN where unknown
end
_Proposal(ν::T, slope, res, m = NaN) where {T} =
  _Proposal{T}(ν, convert(Float64, slope), convert(Float64, res), Int8(0), false, convert(Float64, m))
_nan_proposal(::Type{T}, slope) where {T} = _Proposal{T}(T(NaN), convert(Float64, slope), NaN, Int8(0), false, NaN)
_warmed(p::_Proposal{T}, warm::Bool) where {T} = _Proposal{T}(p.ν, p.slope, p.res, p.dir, warm, p.m)

# The multiplier of the constraint, a root of φ, which is nonincreasing because
# each prox_entry is nondecreasing in its argument; for a halfspace, 0 when the
# proximal point of f satisfies the constraint. The search starts from the
# multiplier of the last call, and from 0 at the first call or when that was
# an inactive halfspace. For W = Float32 the search proposes a multiplier, which is accepted when
# the check of the noise of its own evaluation leaves the output within the tolerance (_accepted), and otherwise,
# after a retry from zero where it started from the last multiplier, refined in Float64 (_refined); the function
# keeps the multiplier and the slope of the accepted one only. The refined multiplier is a Float64 of the Float32 problem and its output is formed in Float64,
# rounded once. Another precision has no refinement: a search that finds no bracket raises the ArgumentError there, which is
# a failure to bracket and not a proof that the constraint is infeasible.
function _multiplier(c::LinearConstrained, x::AbstractArray, γ, ::Type{T}) where {T}
  p = c.set isa IndHyperslab ? _hyperslab_proposal(c, x, γ, T, false) : _proposal(c, x, γ, T, false)
  if T === Float32
    # a NaN of the search on a finite argument is an overflow of the evaluation, not a NaN result
    if isnan(p.ν) ? all(isfinite, x) : !_accepted(c, x, γ, p)
      # a start from the last multiplier that does not hold is retried from zero before the recovery
      if p.warm
        p0 = c.set isa IndHyperslab ? _hyperslab_proposal(c, x, γ, T, true) : _proposal(c, x, γ, T, true)
        if !isnan(p0.ν) && _accepted(c, x, γ, p0)
          c.last[] = p0.ν
          c.slope[] = p0.slope
          return p0.ν
        end
      end
      ν, slope = _refined(c, x, γ, p)
      c.last[] = ν.ν
      c.slope[] = slope
      return ν
    end
  else
    p.dir != 0 && _infeasible(c)
  end
  isnan(p.ν) && return p.ν
  c.last[] = p.ν
  c.slope[] = p.slope
  p.ν
end

function _proposal(c::LinearConstrained, x::AbstractArray, γ, ::Type{T}, cold::Bool, seed = c.last[], slope = c.slope[]) where {T}
  ν0 = convert(T, seed)
  warm = !cold && isfinite(ν0) && !(c.set isa IndHalfspace && !(ν0 > 0))
  _search(c, x, γ, warm ? ν0 : zero(T), warm, _scaled_offset(c.set), c.set isa IndHalfspace ? 1 : 0, cold ? NaN : slope)
end

# Keep the selected finite bound through every pass, including variable fixing.
function _hyperslab_proposal(c::LinearConstrained, x::AbstractArray, γ, ::Type{T}, cold::Bool, seed = c.last[], slope = c.slope[]) where {T}
  set = c.set
  res, mm = Inf, 0.0                        # the distance of ⟨a, z(0)⟩ from the nearest bound, as computed, and its scale
  if isfinite(set.l)
    b = _scaled_offset(set, set.l)
    φ, m = _constraint_gap(c, x, zero(T), γ, b)
    isnan(φ) && return _nan_proposal(T, NaN)
    φ < 0 && return _slab_proposal(c, x, γ, T, b, -1, cold, seed, slope)
    res, mm = min(res, abs(φ)), max(mm, m)
  end
  if isfinite(set.u)
    b = _scaled_offset(set, set.u)
    φ, m = _constraint_gap(c, x, zero(T), γ, b)
    isnan(φ) && return _nan_proposal(T, NaN)
    φ > 0 && return _slab_proposal(c, x, γ, T, b, 1, cold, seed, slope)
    res, mm = min(res, abs(φ)), max(mm, m)
  end
  isfinite(set.l) || isfinite(set.u) || return _Proposal(zero(T), NaN, 0.0, 0.0)       # the whole space: z(0) is the answer
  _Proposal(zero(T), NaN, res, mm)
end

function _slab_proposal(c, x, γ, ::Type{T}, b, side, cold::Bool, seed, slope) where {T}
  ν0 = convert(T, seed)
  warm = !cold && isfinite(ν0) && side * ν0 > 0
  _search(c, x, γ, warm ? ν0 : zero(T), warm, b, side, warm ? slope : NaN)
end

# The search from start. The first step is at least |φ(start)|/Σᵢ γᵢaᵢ², a lower
# bound on the distance to the root, as the slope of φ is at least -Σᵢ γᵢaᵢ². From
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
_search(c::LinearConstrained, x::AbstractArray, γ, start::T, warm::Bool, b, side, slope0) where {T} =
  warm || length(x) < _FIX_NMIN ? _search(c, x, γ, start, warm, b, side, nothing, slope0) :
                                  _search(c, x, γ, start, false, b, side, _live(c, x, T), slope0)

_search(c::LinearConstrained, x::AbstractArray, γ, start::T, warm::Bool, b, side, L, slope0) where {T} =
  _warmed(_search_from(c, x, γ, start, warm, b, side, L, slope0), warm)

# The slope that a proposal carries to the acceptance check of a Float32 call is one computed at this evaluation: the derivative
# of φ at the point, where the pass computed it, or the secant through the last two points. A search that ends at its first point
# without a derivative there has none (NaN), and the check does not accept it; for a Float32 call the seed slope0 sets the size
# of a step and nothing else. The other precisions keep the seed.
_current_slope(::Type{Float32}, dφ, slope0) = dφ < 0 ? dφ : NaN
_current_slope(::Type, dφ, slope0) = slope0
_no_slope(::Type{Float32}, slope0) = NaN
_no_slope(::Type, slope0) = slope0

function _search_from(c::LinearConstrained, x::AbstractArray, γ, start::T, warm::Bool, b, side, L, slope0) where {T}
  φs, ms, dφ = warm ? _gap_and_slope(c, x, start, γ, b) : (_constraint_gap(c, x, start, γ, b)..., T(NaN))
  isnan(φs) && return _nan_proposal(T, slope0)
  abs(φs) <= 2eps(T) * ms && return _Proposal(start, _current_slope(T, dφ, slope0), abs(φs), ms)
  if side != 0 && side * φs < 0
    iszero(start) && return _Proposal(zero(T), _no_slope(T, slope0), abs(φs), ms)     # the constraint is inactive
    # Check zero when the root lies between it and the previous multiplier.
    φ0, m0 = _constraint_gap(c, x, zero(T), γ, b)
    isnan(φ0) && return _nan_proposal(T, slope0)
    (side * φ0 <= 0 || abs(φ0) <= 2eps(T) * m0) && return _Proposal(zero(T), _no_slope(T, slope0), abs(φ0), m0)
    return _refine(c, x, γ, b, zero(T), φ0, start, φs, nothing, slope0)
  end
  a2 = convert(T, _metric_sqnorm(c.set, x, γ))
  step = abs(φs) / a2
  L = _seen(L, start, φs, a2)
  σ = dφ < 0 ? dφ : (warm ? convert(T, slope0) : T(NaN))
  σ < 0 && (step = max(step, abs(φs) / -σ))
  _bracket_and_refine(c, x, γ, b, start, φs, step, L, slope0)
end

function _bracket_and_refine(c::LinearConstrained, x::AbstractArray, γ, b, ν0::T, φ0::T, step::T, L, slope0) where {T}
  dir = φ0 > 0 ? one(T) : -one(T)
  lo, φlo = ν0, φ0                          # φlo has the sign of φ(ν₀), φhi the other
  hi, φhi = lo, φlo
  for k in 1:_MAX_BRACKET_STEPS
    hi = lo + dir * step
    φhi, mhi, L = _gap!(c, x, hi, γ, b, L)
    isnan(φhi) && return _nan_proposal(T, slope0)
    if abs(φhi) <= 2eps(T) * mhi
      return _Proposal(hi, (φhi - φlo) / (hi - lo), abs(φhi), mhi)
    end
    signbit(φhi) == signbit(φ0) || break
    k == _MAX_BRACKET_STEPS && return _unbracketed(c, T, hi, dir, slope0)
    s = step * φhi / (φlo - φhi)
    step = isfinite(s) && s > 0 ? min(2s, 64step) : 2step
    lo, φlo = hi, φhi
  end
  # regula falsi passes over the live entries once some are fixed, and searches
  # as before otherwise
  L isa Base.RefValue && L[].compact && return _refine(c, x, γ, b, lo, φlo, hi, φhi, L, slope0)
  _refine(c, x, γ, b, lo, φlo, hi, φhi, nothing, slope0)
end

# the bracketing and the refinement without a slope seed, for a caller that holds none: the multiplier alone
_bracket_and_refine(c::LinearConstrained, x::AbstractArray, γ, b, ν0::T, φ0::T, step::T, L) where {T} =
  _bracketed(c, _bracket_and_refine(c, x, γ, b, ν0, φ0, step, L, NaN))
_refine(c::LinearConstrained, x::AbstractArray, γ, b, lo::T, φlo::T, hi::T, φhi::T, L) where {T} =
  _bracketed(c, _refine(c, x, γ, b, lo, φlo, hi, φhi, L, NaN))
_bracketed(c::LinearConstrained, p::_Proposal) = (p.dir != 0 && _infeasible(c); p.ν)

# 64 steps found no multiplier on the other side of the root: the proposal carries the direction and the last point
# tried. That is not a proof of infeasibility (a stale multiplier far from a root whose scale is small is out of
# reach of 64 doublings); _multiplier raises it for the precisions that have no refinement, and the refinement of
# a Float32 call decides it by the range of the constraint (_proven_infeasible).
_unbracketed(c, ::Type{T}, hi, dir, slope0) where {T} = _Proposal{T}(hi, slope0, NaN, Int8(dir), false, NaN)

# regula falsi with the Illinois rule on the bracket [lo, hi], where φlo and φhi
# have opposite signs; returns the slope of the last secant for the next call
function _refine(c::LinearConstrained, x::AbstractArray, γ, b, lo::T, φlo::T, hi::T, φhi::T, L, slope0) where {T}
  best, φbest = abs(φlo) < abs(φhi) ? (lo, φlo) : (hi, φhi)
  mbest = T(NaN)                            # the scale of φ at the best point, once evaluated
  νp, φp = hi, φhi                          # the last point evaluated, for the slope
  side = 0                                  # which end the last step kept
  slope = _no_slope(T, convert(Float64, slope0))     # the slope of the last secant, once there is one
  for _ in 1:_MAX_SECANTS
    ν = (lo * φhi - hi * φlo) / (φhi - φlo)
    φν, mν, L = _gap!(c, x, ν, γ, b, L, Val(false))
    isnan(φν) && return _nan_proposal(T, slope0)
    ν != νp && (slope = convert(Float64, (φν - φp) / (ν - νp)))
    νp, φp = ν, φν
    abs(φν) < abs(φbest) && ((best, φbest, mbest) = (ν, φν, mν))
    abs(φν) <= 2eps(T) * mν && return _Proposal(ν, slope, abs(φν), mν)
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
  # a best point that is an end of a bracket that closed on it has no scale yet: the check of a Float32 multiplier needs it
  T === Float32 && isnan(mbest) && isfinite(lo) && isfinite(hi) && abs(hi - lo) <= 2eps(T) * max(abs(lo), abs(hi)) &&
    ((_, mbest, L) = _gap!(c, x, best, γ, b, L, Val(false)))
  _Proposal(best, slope, abs(φbest), mbest)
end

####################################
#  the acceptance of a Float32 multiplier, and its refinement in Float64
####################################

# The Float32 search stops where |φ| is at most 2 eps m, m = Σᵢ |aᵢzᵢ| + |b|. Where φ is flat that leaves the
# multiplier, and so the output, far off: in exact arithmetic, by the firm nonexpansiveness of the proximal map in the metric
# of Diagonal(1 ./ γ), ‖z(ν̂) - z(ν*)‖² ≤ |ν̂ - ν*| |φ(ν̂)| ≤ φ(ν̂)² / s, s the slope of φ between them. The check estimates
# that distance from what the search has: the evaluation of φ has a noise floor, taken as _NOISE_C eps m (the rounding
# of the entries and of the sums of a block; sampled against exact arithmetic over problems of n = 64 to 100 000 to be
# 0.02 to 0.12 eps m at the median and 3.8 eps m at the largest, the largest being a single block; the constant leaves
# out the rare largest values, as refining the call each time they could occur would cost a tenth of the ordinary calls), and
# the output is accepted when
#     max(|φ̂|, noise) / √s ≤ _ACCEPT_K eps(Float32) (m / (2 √Σᵢγᵢaᵢ²) + min(|ν̂| √s, ‖x‖)),   noise = _NOISE_C eps m
# (norms of Diagonal(1 ./ γ)). The slope s is the derivative of φ at ν̂ where the pass computed it, or that of the secant
# through the last two points of the search (a Proposal carries no other: _current_slope, _no_slope). The bracket is a
# scale estimate of the size of the answer, not a bound on the norm of the computed point. Its first term is a lower bound of
# the norm of a point that satisfies the constraint (Σᵢ|aᵢzᵢ| ≥ |b| there, and the Cauchy-Schwarz inequality); at ν̂ the
# constraint holds only up to the residual, and m is a rounded sum. Its second term is the size of what the output pass
# rounds: the entries of the argument x - γνa that are not clamped carry the rounding of |ν| ‖γa‖ over the live entries,
# which is |ν| √s; it is capped by ‖x‖, the norm of the argument, because where the search ends on a flat piece of φ beyond
# the root (no entry free there, the slope from the secant of the bracket) |ν̂| √s is many times the size of the answer and
# would accept any point of the piece.
# The check reads the steps (and the normal) and the argument, besides the pass: _step_sums forms Σᵢγᵢaᵢ² and Σᵢγᵢ of the
# steps rounded to Float32, which is no pass for a scalar step and two passes over the steps for an array of steps (the second
# also over an array normal). ‖x‖ is a separate, conditional read: it is compared with what the check still needs of the
# second term, summing the entries in blocks until that is reached (_norm_below), and only where the first term alone does
# not accept and the uncapped check does; and only for the Float32 search: the check of the recovery, in Float64, keeps |ν̂| √s.
# The terms of the evaluation are summed in blocks (_GAP_BLOCK), in Float32 lanes for a scalar step and, with array steps, for
# IndBox, IndNonnegative, a Scaled or a BoxConstrained of one of them and the ordinary ridge; in Float64 lanes for the other
# entry maps with array steps (_wide_lanes), which takes the summation error of a block out of the noise: the noise floor is
# then that of the entries.
# A constraint that is inactive with a margin of the noise needs no slope.
# This is a check and not a bound: the slope is that of a secant or of a derivative at the point, and a secant can cross a
# kink of φ and overstate the slope next to the root; the noise floor is a constant chosen from measurements and not proved
# (the Float32 lane sums of a scalar step were found 3 to 4 eps m off in two calls); the rounding of the argument is
# counted through the size alone. An accepted result can exceed 16 eps(Float32), by a factor of about 1.2 at most on the random
# problems up to n = 100 000 that it was measured on, and can in principle be inaccurate.
# An evaluation that may have overflowed (γ|ν| or γ|ν||a| near the range of Float32, bounded through Σᵢγᵢ and Σᵢγᵢaᵢ²)
# is not accepted: the computed argument of an entry is then infinite where the exact one is moderate, and the entry
# looks like an exact clamp.
const _ACCEPT_K = 16
const _NOISE_C = 2.0

# Σᵢ γᵢaᵢ² and Σᵢ γᵢ (a scalar step is not summed), for the steps rounded to Float32 as the passes round them
_step_sums(set, x::AbstractArray, γ::Real) = (g = convert(Float32, γ); (Float64(_metric_sqnorm(set, x, g)), Float64(g)))
function _step_sums(set, x::AbstractArray, γ::AbstractArray)
  G = 0.0
  @inbounds @simd for i in eachindex(IndexLinear(), γ)
    G += Float64(convert(Float32, γ[i]))
  end
  set.a isa AbstractArray || return Float64(_metric_sqnorm(set, x, γ)), G
  a, A2 = _scaled_normal(set), 0.0
  @inbounds @simd for i in eachindex(IndexLinear(), γ)
    A2 += Float64(convert(Float32, γ[i])) * abs2(Float64(a[i]))
  end
  A2, G
end

# Whether ‖x‖ in the norm of Diagonal(1 ./ γ), for the steps rounded to Float32 as _step_sums rounds them, is below `need`
# (false where it is not a number or need is not positive); the sum stops as soon as the entries seen so far reach need², in blocks of _NORM_BLOCK.
const _NORM_BLOCK = 256
function _norm_below(x::AbstractArray, γ::Real, need)
  need > 0 || return false
  need2 = need * need * Float64(convert(Float32, γ))
  acc = 0.0
  idx = eachindex(IndexLinear(), x)
  @inbounds for lo in first(idx):_NORM_BLOCK:last(idx)
    b = 0.0
    @simd for i in lo:min(lo + (_NORM_BLOCK - 1), last(idx))
      b += abs2(Float64(x[i]))
    end
    acc += b
    acc >= need2 && return false
  end
  acc < need2
end
function _norm_below(x::AbstractArray, γ::AbstractArray, need)
  need > 0 || return false
  need2 = need * need
  acc = 0.0
  idx = eachindex(IndexLinear(), x)
  @inbounds for lo in first(idx):_NORM_BLOCK:last(idx)
    b = 0.0
    @simd for i in lo:min(lo + (_NORM_BLOCK - 1), last(idx))
      b += abs2(Float64(x[i])) / Float64(convert(Float32, γ[i]))
    end
    acc += b
    acc >= need2 && return false
  end
  acc < need2
end

function _criterion(c::LinearConstrained, x::AbstractArray, γ, p::_Proposal, ::Type{E}, slope_free::Bool) where {E}
  ν, r, m = Float64(p.ν), p.res, p.m
  (isfinite(ν) && isfinite(r) && isfinite(m)) || return false
  εn = Float64(eps(E))
  noise = _NOISE_C * εn * m
  noise == 0 && r == 0 && return true                      # m is zero: the terms aᵢzᵢ that contribute and the offset are zero
  iszero(ν) && r > noise && return true                    # inactive with a margin of the noise
  s = p.slope < 0 && isfinite(p.slope) ? -p.slope : NaN
  isfinite(s) || return slope_free && r <= 2 * εn * m
  A2, G = _step_sums(c.set, x, _steps(γ))
  (isfinite(A2) && A2 > 0 && isfinite(G)) || return false
  abs(ν) * max(G, sqrt(G * A2)) <= Float64(floatmax(E)) / 8 || return false
  lhs, tol = max(r, noise) / sqrt(s), _ACCEPT_K * Float64(eps(Float32))
  t = m / (2 * sqrt(A2))
  lhs <= tol * t && return true
  lhs <= tol * (t + abs(ν) * sqrt(s)) || return false
  E === Float32 && _norm_below(x, _steps(γ), lhs / tol - t) && return false      # min(|ν̂|√s, ‖x‖) is too small
  true
end

_accepted(c::LinearConstrained, x::AbstractArray, γ, p::_Proposal{Float32}) =
  p.dir == 0 && isfinite(p.ν) && _criterion(c, x, γ, p, Float32, false)

####################################
#  the recovery in Float64
####################################

# Where the check of the Float32 multiplier fails, the multiplier is found again in Float64, on the problem the Float32
# kernel solves: x, the normal and the offsets are data of the Float32 model and widen exactly; a step is rounded to
# Float32 once and widened; each of these entry maps takes its parameters (combined thresholds γλw, ridge weights γλ, the center of
# a ridge, bounds of a box, the step of a Scaled) rounded to Float32 as the kernel rounds them and does the rest in Float64
# (_prox_entry_wide). That is a fixed set of functions (_has_wide, _wide_ok): NormL1, NormL1Nonneg, ElasticNet, IndBox,
# IndNonnegative, a ridge with its weight in the ordinary band or the scaled one, with no center or one of Float16,
# Float32 or Float64 entries that keep their range on rounding to Float32, a box on one of them, and Scaled by a Float32,
# Float16 or fixed-width integer factor. For any other function (and for a Scaled with a wider factor or a BigInt, whose
# steps are combined in the wider type, and a ridge with another center), the recovery is refused: the entry map in
# Float64 would form those parameters unrounded, which is another problem, and the failed check raises PrecisionError
# before the destination is written. The search is the one of the Float32 call (from the candidate, then from zero) over the
# wide entry maps with no slopes of their own; its result is accepted by the same check at the noise of Float64. The
# output is formed in Float64 at that multiplier and rounded once to the destination.

# the entry maps of f in Float64 with the parameters of the Float32 problem, as a function of the search
struct _Wide{F} <: AbstractFunction
  f::F
end
isseparable(::Type{<:_Wide}) = true
isconvex(::Type{<:_Wide}) = true
Base.@propagate_inbounds prox_entry(g::_Wide, v::Real, i::Integer, γ::Real) = _prox_entry_wide(g.f, Float64(v), i, Float32(γ))

# the multiplier of the recovery, which the output pass forms with the wide entry maps
struct _WideMultiplier
  ν::Float64
end

# which functions the recovery has entry maps for: by type here, and by the values of the parameters in _wide_ok. A ridge has
# its center rounded to Float32 as the entry of the Float32 kernel rounds it, which the wide entry does for centers of the
# hardware floats that keep their range on rounding; any other center is refused. A Scaled has its step combined as the
# kernel combines it, in Float32, for a Float32, Float16 or fixed-width integer factor.
_has_wide(::Type{<:AbstractFunction}) = false
for F in (:NormL1, :NormL1Nonneg, :ElasticNet, :IndBox, :IndNonnegative)
  @eval _has_wide(::Type{<:$F}) = true
end
_has_wide(::Type{<:SqrNormL2{<:Real, Nothing}}) = true
_has_wide(::Type{<:SqrNormL2{<:Real, Y}}) where {Y<:AbstractArray} = eltype(Y) <: Union{Float16, Float32, Float64}
_has_wide(::Type{<:BoxConstrained{F}}) where {F} = _has_wide(F)
_has_wide(::Type{<:Scaled{F, T}}) where {F, T} = _has_wide(F) && T <: Union{Float32, Float16, Base.BitInteger}
_wide_ok(f) = true
_wide_ok(f::SqrNormL2) = f.y === nothing || _ridge_all_fit(Float32, f.y)
_wide_ok(f::BoxConstrained) = _wide_ok(f.f)
_wide_ok(f::Scaled) = _wide_ok(f.f)

# _prox_entry_wide(f, v, i, γ) is the entry map at the Float64 argument v with the Float32 step γ; NaN where the
# parameters leave the range that the map is written for (the recovery then finds no root)
Base.@propagate_inbounds _prox_entry_wide(f::NormL1, v::Float64, i::Integer, γ::Float32) =
  first(_l1_prox_threshold(v, γ, f.λ0, _entry(f.w, i), Float32))
Base.@propagate_inbounds _prox_entry_wide(f::NormL1Nonneg, v::Float64, i::Integer, γ::Float32) =
  max(v - Float64(_weight_product(Float32, γ, f.λ0, _entry(f.w, i))), 0.0)
Base.@propagate_inbounds function _prox_entry_wide(f::ElasticNet, v::Float64, i::Integer, γ::Float32)
  t = _typed_product(Float32, γ, f.λ2)
  isfinite(t) || return NaN
  shrink(v, _weight_coefficient(Float32, γ, f.λ1, _entry(f.w, i))) / (1.0 + Float64(t))
end
Base.@propagate_inbounds _prox_entry_wide(f::IndBox, v::Float64, i::Integer, ::Float32) =
  clamp(v, Float64(convert(Float32, _entry(f.l, i))), Float64(convert(Float32, _entry(f.u, i))))
@inline _prox_entry_wide(::IndNonnegative, v::Float64, ::Integer, ::Float32) = max(v, 0.0)
# the ridge weight of the Float32 kernel: t = γλ rounded to Float32 in the band of the ordinary coefficient, and outside
# the band (the scaled coefficient) the product of the significands rounded to Float32, times the power of two of the
# exponents; NaN where that leaves the range of a Float64 or the factors are not finite and positive
@inline function _wide_ridge_weight(γ::Float32, λ)
  if _ridge_needs_scaling(Float32, γ, λ)
    P = promote_type(Float32, float(typeof(λ)))
    mγ, eγ = _frexp(P(γ))
    mλ, eλ = _frexp(P(λ))
    t = ldexp(Float64(convert(Float32, mγ * mλ)), eγ + eλ)
    return (2.0^-1000 < t < 2.0^1000) ? t : NaN
  end
  t = _typed_product(Float32, γ, λ)
  ok = iszero(λ) | ((_ridge_band_low(Float32) <= t) & (t <= _ridge_band_high(Float32)))
  ok ? Float64(t) : NaN
end
@inline _prox_entry_wide(f::SqrNormL2{<:Real, Nothing}, v::Float64, ::Integer, γ::Float32) = v / (1.0 + _wide_ridge_weight(γ, f.λ))
# the center as the Float32 entry reads it: rounded to Float32 (the centers that reach this entry keep their range on rounding)
@inline _wide_center(c::Float64) = Float64(Float32(c))
@inline _wide_center(c::Union{Float32, Float16}) = Float64(c)
Base.@propagate_inbounds function _prox_entry_wide(f::SqrNormL2, v::Float64, i::Integer, γ::Float32)
  t = _wide_ridge_weight(γ, f.λ)
  (v + t * _wide_center(f.y[i])) / (1.0 + t)
end
Base.@propagate_inbounds function _prox_entry_wide(s::Scaled, v::Float64, i::Integer, γ::Float32)
  _nested_route(s, γ) && return NaN
  _prox_entry_wide(s.f, v, i, s.α * γ)
end
Base.@propagate_inbounds function _prox_entry_wide(c::BoxConstrained, v::Float64, i::Integer, γ::Float32)
  clamp(_prox_entry_wide(c.f, v, i, γ), Float64(convert(Float32, _entry(c.box.l, i))), Float64(convert(Float32, _entry(c.box.u, i))))
end

# the steps as the Float32 kernel reads them; steps stored wider are read through a view that rounds each entry, not copied
struct _RoundedSteps{N, G<:AbstractArray{<:Real, N}} <: AbstractArray{Float32, N}
  γ::G
end
Base.size(r::_RoundedSteps) = size(r.γ)
Base.axes(r::_RoundedSteps) = axes(r.γ)
Base.IndexStyle(::Type{<:_RoundedSteps}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(r::_RoundedSteps, i::Int) = convert(Float32, r.γ[i])
_round_steps(γ::Real) = convert(Float32, γ)
_round_steps(γ::AbstractArray{Float32}) = γ
_round_steps(γ::AbstractArray) = _RoundedSteps(γ)

# the output pass at the multiplier of the recovery
_linear_constrained_output!(y, f, a, x, γ, ν::_WideMultiplier) = _wide_output!(y, f, a, x, γ, ν.ν)
_linear_constrained_output!(y, f::SqrNormL2, a, x, s::_RidgeSteps, ν::_WideMultiplier) = _wide_output!(y, f, a, x, s, ν.ν)
function _wide_output!(y, f, a, x, γ, ν::Float64)
  w, s = _Wide(f), _steps(γ)
  @inbounds for i in eachindex(IndexLinear(), y, x)
    γᵢ = Float64(convert(Float32, _entry(s, i)))
    y[i] = convert(eltype(y), prox_entry(w, Float64(x[i]) - γᵢ * ν * Float64(_entry(a, i)), i, γᵢ))
  end
  y
end

# Whether the constraint is infeasible: the range of ⟨a, z⟩ over the ranges of the entry maps (the bounds of the Float32
# kernel) lies strictly on the wrong side of an offset, by more than the rounding of the Float64 sums. A search that
# finds no bracket does not decide it: its steps are bounded (64 steps), and a start far from a root of small scale can be out of
# reach of them.
function _proven_infeasible(c::LinearConstrained, x::AbstractArray)
  set, f = c.set, c.f
  a = _scaled_normal(set)
  lo = hi = Mlo = Mhi = 0.0              # the ends of the range, and the sums of the magnitudes of their terms
  @inbounds for i in eachindex(IndexLinear(), x)
    aᵢ = Float64(_entry(a, i))
    iszero(aᵢ) && continue
    l, u = _prox_range(f, i, Float32)
    l, u = Float64(l), Float64(u)
    p, q = aᵢ > 0 ? (aᵢ * l, aᵢ * u) : (aᵢ * u, aᵢ * l)
    lo += p
    hi += q
    Mlo += abs(p)
    Mhi += abs(q)
  end
  # an end that is infinite proves nothing, and does not take the proof at the other end with it
  below(b) = b < lo - 4 * length(x) * eps(Float64) * (Mlo + abs(b))         # b is under the range
  above(b) = b > hi + 4 * length(x) * eps(Float64) * (Mhi + abs(b))         # b is over the range
  if set isa IndHyperplane
    b = Float64(_scaled_offset(set))
    below(b) || above(b)
  elseif set isa IndHalfspace
    below(Float64(_scaled_offset(set)))
  else
    l = isfinite(set.l) ? Float64(_scaled_offset(set, set.l)) : -Inf
    u = isfinite(set.u) ? Float64(_scaled_offset(set, set.u)) : Inf
    (isfinite(u) && below(u)) || (isfinite(l) && above(l))
  end
end

# The recovery: the multiplier of the wide search and its slope, or an error before the destination is written: ArgumentError
# where the constraint is proven infeasible, PrecisionError where the recovery is refused or finds no multiplier that the
# check accepts.
function _refined(c::LinearConstrained, x::AbstractArray, γ, p::_Proposal{Float32})
  wide = _has_wide(typeof(c.f)) && _wide_ok(c.f)
  if wide
    w = LinearConstrained{_Wide{typeof(c.f)}, typeof(c.set)}(_Wide(c.f), c)
    s = _round_steps(_steps(γ))
    seed = isfinite(p.ν) ? Float64(p.ν) : 0.0
    for cold in (iszero(seed) ? (true,) : (false, true))
      q = c.set isa IndHyperslab ? _hyperslab_proposal(w, x, s, Float64, cold, seed, p.slope) :
                                   _proposal(w, x, s, Float64, cold, seed, p.slope)
      _criterion(c, x, s, q, Float64, true) && return _WideMultiplier(Float64(q.ν)), q.slope
    end
  end
  _proven_infeasible(c, x) && _infeasible(c)
  throw(PrecisionError("the multiplier of the $(nameof(typeof(c.set))) on a $(nameof(typeof(c.f))) is not resolved by the " *
                       "Float32 check and " * (wide ? "the Float64 recovery does not find one it accepts" :
                       "the function has no Float64 recovery of the Float32 problem") * "; no result is written"))
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
_head(a::Real, ::Integer) = a
_head(a::AbstractVector, n::Integer) = view(a, Base.OneTo(n))

# f with the entry j at the index I[j]
struct _Reindexed{F, V}
  f::F
  I::V
end
_vector_lanes(f::_Reindexed) = _vector_lanes(f.f)
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
  G = γ isa Real ? γ : _buffer!(bg, T, n)
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
  _at_end(f, prox_entry(f, Xs[j] - γᵢ * ν * aᵢ, i, γᵢ), aᵢ, i, low)
end

# the scalar proximal operator of f where it sits at an end, and 0 elsewhere:
# through the pass, the sums of aᵢzᵢ and |aᵢzᵢ| over the entries that it fixes
struct _AtEnd{F, A}
  f::F
  a::A          # the normal, by index
  low::Bool
end
_vector_lanes(f::_AtEnd) = _vector_lanes(f.f)
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
      G isa AbstractArray && (G[w] = convert(T, Gs[j]))
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
  _checkreal(c, y)
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
#  the linear minimization oracle of the capped simplex
####################################

# {s : 0 ≤ sᵢ ≤ 1, Σᵢ sᵢ = k} for an integer k, as Constrained(IndBox(0, 1), IndHyperplane(a, k)) with
# a = 1 or an array of ones. Nothing else is recognized: other bounds, normals, halfspaces and
# hyperslabs have no oracle. A scalar normal does not know the dimension, so 0 ≤ k ≤ n is
# checked by lmo!, as for the empty simplex.
const _CappedSimplex = LinearConstrained{<:IndBox{<:Real, <:Real}, <:IndHyperplane}

_support_fallback(::_CappedSimplex) = false   # its support function is the sum of the k largest entries

function haslmo(c::_CappedSimplex)
  c.f.l == 0 && c.f.u == 1 && c.set.b >= 0 && isinteger(c.set.b) && isfinite(c.set.b) &&
    _constant_normal(c.set.a) == 1
end

function lmo!(s::AbstractArray, c::_CappedSimplex, g::AbstractArray)
  _checkreal(c, g)
  _checkreal(c, s)
  s isa Union{Array, SubArray{<:Any, <:Any, <:Array}} && eltype(s) <: AbstractFloat ||
    throw(ArgumentError("the output of lmo! must be an Array, or a view of one, of floating-point numbers; got $(typeof(s))"))
  _checksize(s, g)
  _checkparams(c, g)
  haslmo(c) || _lmo_unsupported(c)
  n = length(g)
  c.set.b <= n || throw(ArgumentError("the capped simplex with sum $(c.set.b) has no point of size $(size(g))"))
  _capped_lmo!(s, c, g, Int(c.set.b))
  s
end

function _capped_lmo!(s, c, g, k::Int)
  T = eltype(s)
  n = length(g)
  # the minimizer does not depend on g when k is 0 or n (E2)
  k == 0 && return fill!(s, zero(T))
  k == n && return fill!(s, one(T))
  v = _buffer!(c.scratch, eltype(g), n)
  nan = false
  @inbounds for i in 1:n
    x = g[i]
    nan |= isnan(x)
    v[i] = x
  end
  nan && return fill!(s, T(NaN))
  # the k-th smallest entry, then the entries below it, then the first ties
  t = _kth_largest!(view(v, 1:n), n - k + 1)
  below = 0
  @inbounds for i in 1:n
    below += v[i] < t
  end
  ties = k - below
  # a single pass in linear order, reading g[i] before writing s[i]: s may be g
  @inbounds for i in 1:n
    x = g[i]
    if x < t
      s[i] = one(T)
    elseif x == t && ties > 0
      ties -= 1
      s[i] = one(T)
    else
      s[i] = zero(T)
    end
  end
  s
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
# the slope is the weight prox_entry applies to the argument, also where γλ leaves the
# range of the model type; _OrdinaryRidge takes the ordinary weight, as the tested route does
Base.@propagate_inbounds _prox_entry_slope(f::SqrNormL2, v::Real, i::Integer, γ::Real) = _ridge_prox_slope(f, v, i, γ)
Base.@propagate_inbounds _prox_entry_slope(r::_OrdinaryRidge, v::Real, i::Integer, γ::Real) = _ridge_prox_slope(r, v, i, γ)
Base.@propagate_inbounds function _prox_entry_slope(f::NormL1, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  # a fixed entry has an infinite threshold, so its slope is zero
  z, t = _l1_prox_threshold(v, γ, f.λ0, _entry(f.w, i), S)
  z, _slope(abs(v) > t, one(S))
end
Base.@propagate_inbounds function _prox_entry_slope(f::NormL1Nonneg, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  t = _weight_product(S, γ, f.λ0, _entry(f.w, i))
  max(v - t, zero(S)), _slope(v > t, one(S))
end
Base.@propagate_inbounds function _prox_entry_slope(f::ElasticNet, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  w = _entry(f.w, i)
  r = _typed_product(S, γ, f.λ2)
  isfinite(r) || return _elastic_prox_slope_wide(S, f, v, w, γ, r)
  t = _weight_coefficient(S, γ, f.λ1, w)
  shrink(v, t) / (1 + r), _slope(abs(v) > t, inv(1 + r))
end
@noinline function _elastic_prox_slope_wide(::Type{S}, f::ElasticNet, v, w, γ, r) where {S}
  if γ == Inf
    z, dz = _elastic_infinite_slope(v, f.λ1 > zero(f.λ1), f.λ2 > zero(f.λ2), w)
    return convert(S, z), convert(S, dz)
  end
  t = _weight_coefficient(S, γ, f.λ1, w)
  _elastic_ridge_wide(S, shrink(v, t), γ, f.λ2, r), _slope(abs(v) > t, _elastic_slope_wide(S, γ, f.λ2, r))
end
Base.@propagate_inbounds function _prox_entry_slope(f::IndBox, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  l, u = convert(S, _entry(f.l, i)), convert(S, _entry(f.u, i))
  clamp(convert(S, v), l, u), _slope((v > l) & (v < u), one(S))
end
@inline _prox_entry_slope(::IndNonnegative, v::T, ::Integer, ::Real) where {T<:Real} =
  (max(float(v), zero(float(T))), _slope(v > 0, one(float(T))))
Base.@propagate_inbounds function _prox_entry_slope(s::Scaled, v::Real, i::Integer, γ::Real)
  if _nested_route(s, γ)
    g, step, _ = _nested_step(s, γ)
    return _prox_entry_slope(g, v, i, step)
  end
  _prox_entry_slope(s.f, v, i, s.α * γ)
end
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

# the range of the ordinary ridge, which has no end
_prox_range(::_OrdinaryRidge, ::Integer, ::Type{S}) where {S} = (S(-Inf), S(Inf))

function _checkstep(op::O, c::LinearConstrained, y::AbstractArray, x::AbstractArray, γ::Real) where {O}
  f, set = c.f, c.set
  W = _linear_model_type(c, x, y)
  if isempty(x)
    if !(set isa IndHyperslab)
      T = W
      φ = -convert(T, _scaled_offset(set))
      if set isa IndHyperplane || φ > 0
        abs(φ) <= 2eps(T) * abs(φ) || _infeasible(c)
      end
    end
    return true
  end
  f isa IndBox && f.l isa Real && f.u isa Real && set.a isa Real || return true
  T = W
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

_decided_on_spectrum(::BoxConstrained) = true
function _spectrum_value(c::BoxConstrained, v::AbstractVector{T}, k::Int) where {T}
  _checkparams(c, v)
  # A finite ordinary value needs neither relaxed membership nor a new sum.
  if k == 0 && !_isindicator(c)
    ordinary = c(v)
    isfinite(ordinary) && return convert(T, ordinary)
  end
  iszero(_spectrum_value(c.box, v, k)) || return T(Inf)
  convert(T, _spectrum_value(c.f, v, k))
end
_decided_on_spectrum(::LinearConstrained) = true
function _spectrum_value(c::LinearConstrained, v::AbstractVector{T}, k::Int) where {T}
  _checkparams(c, v)
  # A finite ordinary value needs neither relaxed membership nor a new sum.
  if k == 0 && !_isindicator(c)
    ordinary = c(v)
    isfinite(ordinary) && return convert(T, ordinary)
  end
  iszero(_spectrum_value(c.set, v, k)) || return T(Inf)
  convert(T, _spectrum_value(c.f, v, k))
end
