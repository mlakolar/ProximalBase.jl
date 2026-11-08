####################################
#
#  Functions built from other functions
#
####################################

# Retain exceptional wrapper products until the leaf coefficient is formed.
#
# The exponent is exact. Its type follows the significand: Int for every type but BigFloat, whose
# MPFR exponents reach ±(2^62 - 1), so that a product of three or more extreme factors has an
# exponent beyond Int64; it is an Int128 there, which is the width of `_exponent_sum(BigFloat, …)`.
# One composition adds at most about 2^63 to the magnitude of the exponent, so an Int128 is exact
# for chains of up to 2^64 factors, and a chain of that length does not fit in memory. A retained
# exponent is exact on the host through its last cancellation, and the final shift of `ldexp` saturates. Two
# bounded handoffs are clamped on purpose: the exponent of a ball bound that goes to the spectral ball test
# (`_ball_exponent`), and a scale converted for a device (`_device_exponent`, device types only).
_scale_exponent_type(::Type) = Int
_scale_exponent_type(::Type{BigFloat}) = Int128
struct _ProjectionScale{T, E<:Integer}
  m::T
  e::E
  _ProjectionScale{T, E}(m, e) where {T, E<:Integer} = new{T, E}(m, e)
end
_ProjectionScale(m::T, e::Integer) where {T} =
  _ProjectionScale{T, _scale_exponent_type(T)}(m, convert(_scale_exponent_type(T), e))
Base.broadcastable(a::_ProjectionScale) = Ref(a)
Base.isfinite(a::_ProjectionScale) = isfinite(a.m)
Base.isinf(a::_ProjectionScale) = isinf(a.m)
Base.iszero(a::_ProjectionScale) = iszero(a.m)
Base.isone(a::_ProjectionScale) = a.m == 0.5 && a.e == 1
_projection_real(a::Real) = a
_projection_real(a::_ProjectionScale) = ldexp(a.m, a.e)
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
  _ProjectionScale(m, _exponent_sum(typeof(m), ea, eb, e))
end
function _projection_accumulate(a::_ProjectionScale, b::Real)
  (!isfinite(b) || iszero(b)) && return b
  mb, eb = _frexp(float(b))
  m, e = _frexp(a.m * mb)
  _ProjectionScale(m, _exponent_sum(typeof(m), a.e, eb, e))
end
_projection_product(a, b, w) = _positive_product(a, b, w)
_projection_product(a::_ProjectionScale, b, w) = _scaled_product(a.m, b, w, a.e)
# f scaled by an accumulated factor, as nested wrappers when the factor is not a normal number:
# pieces of at most half the exponent range, so that each is stored exactly
_projection_scaled(f, α::Real) = Scaled(f, α)
# Two wrappers hold any factor within twice that range, and have a type that inference follows;
# the general case is the loop below.
_projection_two_pieces(α::_ProjectionScale{T}) where {T} = abs(α.e) <= 2 * (exponent(floatmax(T)) ÷ 2)
function _projection_scaled2(f, α::_ProjectionScale{T}) where {T}
  h = α.e ÷ 2
  Scaled(Scaled(f, ldexp(one(T), h)), ldexp(α.m, α.e - h))
end
function _projection_scaled(f, α::_ProjectionScale{T}) where {T}
  step = exponent(floatmax(T)) ÷ 2
  g, e = f, α.e
  while abs(e) > step
    k = copysign(step, e)
    g = Scaled(g, ldexp(one(T), k))
    e -= k
  end
  Scaled(g, ldexp(α.m, e))
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

struct _ShiftedArray{T, N, A<:AbstractArray, C<:AbstractArray} <: AbstractArray{T, N}
  g::A
  c::C
end
_ShiftedArray(g::AbstractArray{T, N}, c::AbstractArray) where {T, N} =
  _ShiftedArray{promote_type(T, eltype(c)), N, typeof(g), typeof(c)}(g, c)
Base.size(a::_ShiftedArray) = size(a.g)
Base.IndexStyle(::Type{<:_ShiftedArray}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::_ShiftedArray, i::Int) = a.g[i] + a.c[i]

####################################
#  α f
####################################

"""
    Scaled(f, α)
    α * f

The function `α f(x)` for `α ≥ 0`, with the roles of `f`: its gradient is
`α ∇f`, and its proximal operator `prox_{γαf}`, so that `0.5 * f` halves a loss
and `(1/n) * f` averages one. An infinite factor gives the value `0` where `f` is certainly `0`:
an indicator inside its set, and `NormL1`, `NormL1Nonneg`, `NormL2`, `NormLinf`, `NuclearNorm`,
`ElasticNet`, `GroupNormL2`, `GroupNormLinf` and `SparseGroupNormL2` where `f` is `0` at an argument
that is zero on every entry they penalize (an entry under a zero level or weight, or in no group, may
hold anything); elsewhere a computed zero of `f` gives NaN, except where the value of `f` is recovered
from its significand and exponent (the functions without a level listed below, on host arrays), which
gives `±Inf` at a value that underflowed.

When `α f(x)` overflows or underflows although it is representable, the value
is computed with the scale folded into the level for `NormL1`, `NormL1Nonneg`,
`NormL2`, `NormLinf`, `ElasticNet`, the homogeneous group norms,
`SparseGroupNormL2`, `NuclearNorm`, `TracePSD`, and the support functions of
the ℓ1 and ℓ2 balls. For functions without such a level, it is computed from
the value of `f` as a significand and a power of two, with `α` applied last, for
`SqrNormL2`, `CubeNormL2`, `SqrDistL2`, `DistL2`, `NormL1Fused`,
`TotalVariation1D`, `FusedLasso`, `SLOPE`, `SumLargest`, `NegEntropy`,
`KLDivergence`, `NegLog`, `Berhu`, `HingeLoss`, `SqrHingeLoss`,
`EpsInsensitiveLoss`, `CheckLoss`, `NormL0`, `CappedL1`, `LogPenalty`, `NormLq`,
`MCP`, `SCAD`, `GroupMCP` and `GroupSCAD`, and for `Translate` and nested
`Scaled` of these. This holds for the value, the value of
`value_and_gradient!` and, when `α f` overflows, the value of `value_and_prox!`,
on host arrays (an `Array`, a view or a reshape of one, the iterates, sparse
arrays and ranges); other array types and tuple arguments keep the ordinary
result. A `value_and_gradient!` whose gradient output may share memory with a
parameter of `f` returns the ordinary value. Not covered: the data functions
(`LinearFunction`, `QuadraticFunction`, the least-squares and GLM losses,
`BernoulliLoss`; #609), the sums and other wrappers (`Sum`, `SeparableSum`,
`Tilt`, `MoreauEnvelope`, `Conjugate`; #359), wrappers that transform the
argument (#357), spectral functions (#384), and `Regularize`, `Constrained`,
`PointwiseMinimum` and the gradients of these functions (#610).

`Scaled(Scaled(f, a), b)` is `Scaled(f, a * b)` when the product is an ordinary
number. When it underflows, is subnormal or overflows, although the factors are
finite and positive, the wrappers stay nested, so that the function is never replaced
by `ZeroFunction` or an infinite scale because of an intermediate product. The value,
the gradient, the proximal operator, `prox_entry`, the optimality violation, the
Lipschitz constants and the conjugate then use the product of all the factors where
the coefficient is formed: it is folded into the levels of the functions listed above,
or applied to the result with its exponent last, so a chain such as `Scaled(Scaled(NormL1(1e300), 1e-200), 1e-200)`
is the penalty `1e-100 * norm(x, 1)`. The levels are folded through `Translate`, `Sum` and
`SeparableSum` of such functions (indicators and the zero function pass through). The conjugate
value of a function without levels to fold, under a product that is not a number in range,
throws. The value of a function with levels, whose leaf value overflows or underflows when the whole
product does not fold into normal levels (`Scaled(Scaled(Scaled(NormL1(1e300), 1e-300), 1e-300), 1e-300)`
at `[1e300]` is `1e-300`), is computed at levels that are normal numbers, with the power of two
that this takes out of the product applied last. Not recovered: levels whose exponents are too far
apart for one normal fold (`Sum(NormL1(1e300), NormL2(1e-300))` under such a chain at `[1e300]`),
weights or entries for which no normal level gives a leaf value that is a finite normal number, and
results that are subnormal.

A step is not rounded together with the factors. For a scalar step whose product with the
factor is not a normal number, the exponent of the factors and of the step goes into the levels
(for a single factor as for a chain). An array step takes the whole factor into the levels and
passes the caller's array on unchanged; when the product of the levels and the factors is not a
normal number, a power of two that centres the exponents of the entries moves from the step into the
levels (the step is read once, only then). A single factor takes the array route when it lies outside
the band `2^-E` to `2^E`, with `E` a quarter of the exponent range of the element type (255 for
`Float64`, 31 for `Float32`), so that an ordinary call reads nothing more than before; for a factor
inside the band, an entry of the step that is itself within `E` of a limit of the range is rounded
together with the factor. Array-step entries whose exponents spread too far for the levels to hold
them are rounded, and so are the steps of a function without levels to fold.
A level that is stored as infinite is a hard constraint, which no product of a factor and a step can
change: `Scaled(NormL1(Inf), 1e-300)` at the step `1e-300` has the proximal map of the domain `{0}`, as at
the step `1`, and not the identity of a step rounded to zero. The contract of this route is exact: the call
returns what the same function returns at a step whose product is an ordinary (normal) positive number, the
hard domain `{0}` at finite arguments included, up to the sign of a zero (`ElasticNet(1.0, Inf)` at `[-1.0]` gives
`[0.0]` at the step `1` and `[-0.0]` at the step `0.75` and in the wrapper). How a leaf treats a NaN or infinite entry of the argument at an
ordinary step is that leaf's own rule (decision 49 settles `NormL1`, `SLOPE`, `NormL1Fused` and `NormLinf`; the
others are tracked in #662), and the call follows it. A call takes this route only when its function holds a
stored infinite level (through `Scaled`, `Translate` and sums), every factor of the whole tree, indicators and
translated or nested wrappers included, and the step are positive finite numbers (for an array step, every
entry), and every block is a function with an infinite level whose other levels the infinite level makes
irrelevant, or an indicator of the library's list of plain-data indicators (`IndBox`, `IndNonnegative`, `IndBinary`,
`IndBallL0`, `IndBallL1`, `IndBallL2`, `IndSphereL2`, `IndSimplex`, `IndHyperplane`, `IndHalfspace`, `IndHyperslab`, `IndSOC`,
`IndRotatedSOC`, `IndExpPrimal`, `IndExpDual`, `IndPattern`, `IndBanded`, `IndToeplitz`, `IndIsotonic`, `IndBallGroupL0`), also
under a positive finite `Scaled` or a `Translate`. These are `NormL1`, `NormL1Nonneg`, `NormL2`, `NormLinf`,
the group norms, `NuclearNorm` and `TracePSD` with an infinite level; `SLOPE` and `NormL1Fused` with an infinite
first level; `ElasticNet` with an infinite ridge level, or an infinite ℓ1 level and every weight positive, or with
the finite levels zero; `SparseGroupNormL2` with an infinite ℓ1 level, or an infinite group level and every group
weight positive, or a zero ℓ1 level. No finite level or factor is folded, multiplied or combined on this route:
the leaf gets the step of any positive finite size, and the levels that the infinite level makes irrelevant are
removed. Every other call, a function with a finite level that acts, a finite sibling block, a factor or step that
is zero, negative, infinite or NaN anywhere in the tree, an indicator that is not on that list (an intersection, a
constrained or composed indicator, a spectral or factorized one, a user type; also under a `Scaled` or a `Translate`), or a
function without a stored infinite level, takes the route that it always had.
A function without levels to fold (a `SqrNormL2`, a loss, a `Tilt`) forms its gradient and Hessian
product at its own scale before the factors are applied, so an intermediate that leaves the range is
lost, as under a single `Scaled`: `Scaled(Scaled(SqrNormL2(1e300), 1e-200), 1e-200)` at `[1e10]`
has the value `5e-81`, recovered as above, but the gradient and Hessian product `Inf`, not `1e-90`
(#610).

At zero scale, supported finite-parameter penalties and losses simplify to
[`ZeroFunction`](@ref), while an indicator retains its domain and is returned
unchanged. Zero scaling removes finite penalty terms and keeps domain constraints.
For `NormL1`, `NormL1Nonneg`, `ElasticNet`, `CappedL1`, `LogPenalty`, `NormLq`,
`Berhu`, `GroupNormL2`, `GroupNormLinf`, `SparseGroupNormL2`, `GroupMCP` and
`GroupSCAD`, infinite weights retain the same family with all levels zero.
With finite weights, `NormL1Nonneg` becomes `IndNonnegative()` and the others
become `ZeroFunction()`. Non-finite levels or shape parameters are rejected.
The retained zero-level family is not an indicator type, so wrappers that
require one still reject it. Zero scaling of other functions is rejected when
the package has no domain-preserving proximal representation.
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
for (F, shape) in ((CappedL1, :a), (LogPenalty, :ε), (NormLq, :q), (Berhu, :M))
  @eval function _zero_scaled(f::$F)
    isfinite(f.λ) && isfinite(f.$shape) || _zero_scale_error(f)
    _finite_parameter(f.w) ? ZeroFunction() : $F(zero(f.λ), f.$shape, f.w)
  end
end
function _zero_scaled(f::Union{GroupNormL2, GroupNormLinf})
  isfinite(f.λ0) || _zero_scale_error(f)
  _finite_parameter(f.w) ? ZeroFunction() : _rebuild(f, zero(f.λ0), f.groups, f.w)
end
function _zero_scaled(f::SparseGroupNormL2)
  isfinite(f.λ0) && isfinite(f.λ1) || _zero_scale_error(f)
  _finite_parameter(f.w) ? ZeroFunction() : SparseGroupNormL2(zero(f.λ1), zero(f.λ0), f.groups, f.w)
end
_zero_scaled(f::Union{NormL2, CubeNormL2, NormLinf, NormL0}) = _finite_zero(f, f.λ)
_zero_scaled(::SumLargest) = ZeroFunction()
_zero_scaled(f::Union{MCP, SCAD}) = _finite_zero(f, f.λ, f.a)
function _zero_scaled(f::Union{GroupMCP, GroupSCAD})
  isfinite(f.λ0) && isfinite(f.a) || _zero_scale_error(f)
  _finite_parameter(f.w) ? ZeroFunction() : _rebuild(f, zero(f.λ0), f.groups, f.w)
end
function _zero_scaled(f::ElasticNet)
  isfinite(f.λ1) && isfinite(f.λ2) || _zero_scale_error(f)
  _finite_parameter(f.w) ? ZeroFunction() : ElasticNet(zero(f.λ1), zero(f.λ2), f.w)
end
_zero_scaled(f::SqrNormL2) = _finite_zero(f, f.λ, f.y)
_zero_scaled(f::QuadraticFunction) = _finite_zero(f, f.A, f.b, f.c)
_zero_scaled(f::LeastSquaresLoss) = _finite_zero(f, f.X, f.Y)
_zero_scaled(f::LinearFunction) = _finite_zero(f, f.c, f.d)
_zero_scaled(f::Scaled) = (_finite_parameter(f.α) || _zero_scale_error(f); _zero_scaled(f.f))
function _zero_scaled(f::NormL1Nonneg)
  isfinite(f.λ0) || _zero_scale_error(f)
  _finite_parameter(f.w) ? IndNonnegative() : NormL1Nonneg(zero(f.λ0), f.w)
end

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
@inline _chained_scaled(f) = false
@inline _chained_scaled(f::Scaled) = _chained(f) || _chained_scaled(f.f)
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
  _ProjectionScale(m, _exponent_sum(typeof(m), e, -a.e))
end

# Fold a scale into a normal level, retaining exact zero levels.
function _folded_level(λ, α)
  iszero(λ) && return λ
  c = _apply_scale(α, λ)
  isfinite(c) && !iszero(c) && _level_fits(typeof(float(c)), c) ? c : nothing
end
# A scale that only folds into levels that are normal numbers. A subnormal level has fewer bits and
# is no longer the coefficient that an array step multiplies, entry by entry.
struct _NormalScale{P<:_ProjectionScale}
  s::P
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
for (F, level) in ((:NormL1, :λ0), (:NormL1Nonneg, :λ0), (:NormL2, :λ), (:NormLinf, :λ))
  args = F in (:NormL1, :NormL1Nonneg) ? :(c, f.w) : :(c,)
  @eval function _fold_level(f::$F, α)
    c = _folded_level(f.$level, α)
    c === nothing ? nothing : $F($(args.args...))
  end
end
# A scale with a BigFloat significand that no level takes (its product with the level is outside the range) is
# absorbed into the weights instead, which bring the products of scale, level and weight back into range. For the
# scale mα 2^E, the level λ = mλ 2^eλ and a nonzero finite weight w = mw 2^ew, the coefficient is
# c = mα mλ mw 2^t with t = E + eλ + ew (Int128, exact), and it lies in [2^(t-3), 2^t). One power of two 2^s becomes
# the level and each weight becomes c 2^-s, rounded once (the product of the three significands is exact at three
# times the precision). The shift s is the centre of the interval in which the level 2^s and every weight c 2^-s are
# normal numbers with a margin of three binades (room for the significand of the step and for the product);
# the fold is refused (`nothing`, as without it) when no such s exists, that is, when the weights are spread wider
# than the range allows. A zero or infinite weight stays as it is, and with no nonzero finite weight, no weights, a
# zero or non-finite level or a zero or non-finite scale there is no fold. The weights are any array, read by their
# linear index as the leaf reads them. Only BigFloat scales take this fold; every other type folds as before.
# The array-step route (`_shifted_fold`) has no fold of its own.
function _fold_weights(f::Union{NormL1, NormL1Nonneg}, α::_ProjectionScale{BigFloat})
  w, λ = f.w, f.λ0
  (w isa AbstractArray && isfinite(λ) && !iszero(λ) && isfinite(α.m) && !iszero(α.m)) || return nothing
  mλ, eλ = _frexp(float(λ))
  base = α.e + eλ
  tmin, tmax = typemax(Int128), typemin(Int128)
  for wi in w
    (iszero(wi) || !isfinite(wi)) && continue
    t = base + _frexp(convert(BigFloat, wi))[2]
    tmin, tmax = min(tmin, t), max(tmax, t)
  end
  tmax < tmin && return nothing
  emx, emn = Int128(Base.MPFR.get_emax()), Int128(Base.MPFR.get_emin())
  lo, hi = max(emn + 3, tmax - (emx - 3)), min(emx - 3, tmin - (emn + 3))
  lo > hi && return nothing
  s = fld(lo + hi, 2)
  out = Array{BigFloat}(undef, size(w))
  for (k, i) in enumerate(eachindex(w))
    wi = w[i]
    out[k] = iszero(wi) || !isfinite(wi) ? convert(BigFloat, wi) : _scaled_weight(α.m, mλ, convert(BigFloat, wi), base - s)
  end
  level = ldexp(one(BigFloat), s)
  f isa NormL1 ? NormL1(level, out) : NormL1Nonneg(level, out)
end
# ma ml w 2^k rounded once, for the significands ma and ml and a nonzero finite weight w
function _scaled_weight(ma::BigFloat, ml, w::BigFloat, k::Integer)
  mw, ew = _frexp(w)
  p = precision(BigFloat)
  q = 3 * max(p, precision(ma), precision(mw))
  c = setprecision(BigFloat, q) do
    (ma * convert(BigFloat, ml)) * mw
  end
  r = setprecision(BigFloat, p) do
    c * one(BigFloat)
  end
  ldexp(r, k + ew)
end
for F in (:NormL1, :NormL1Nonneg)
  @eval function _fold_level(f::$F, α::_NormalScale{<:_ProjectionScale{BigFloat}})
    c = _folded_level(f.λ0, α)
    c === nothing ? _fold_weights(f, α.s) : $F(c, f.w)
  end
end
function _fold_level(f::ElasticNet, α)
  c1, c2 = _folded_level(f.λ1, α), _folded_level(f.λ2, α)
  c1 === nothing || c2 === nothing ? nothing : ElasticNet(c1, c2, f.w)
end
function _fold_level(f::Union{GroupNormL2, GroupNormLinf}, α)
  c = _folded_level(f.λ0, α)
  c === nothing ? nothing : _rebuild(f, c, f.groups, f.w)
end
function _fold_level(f::SparseGroupNormL2, α)
  c1, c0 = _folded_level(f.λ1, α), _folded_level(f.λ0, α)
  c1 === nothing || c0 === nothing ? nothing : SparseGroupNormL2(c1, c0, f.groups, f.w)
end
function _fold_level(f::NuclearNorm, α)
  c = _folded_level(f.λ, α)
  c === nothing && return nothing
  g = NuclearNorm(c; partial = f.partial)
  g.ws, g.pws = f.ws, f.pws
  g
end
function _fold_level(f::TracePSD, α)
  c = _folded_level(f.λ, α)
  c === nothing && return nothing
  g = TracePSD(c; partial = f.partial)
  g.eig, g.pws, g.out, g.diag = f.eig, f.pws, f.out, f.diag
  g
end
_fold_level(f::Scaled, α) = _fold_level(f.f, _projection_accumulate(α, f.α))
# The levels as the proximal operator folds them together with the exponent of the step: a ridge
# level that falls below the range multiplies the step by less than the range can resolve, so it
# is dropped instead of discarding the fold.
_fold_level_step(f, α) = _fold_level(f, α)
for F in (:NormL1, :NormL1Nonneg)
  @eval function _fold_level_step(f::$F, α::_ProjectionScale{BigFloat})
    g = _fold_level(f, α)
    g === nothing ? _fold_weights(f, α) : g
  end
end
_fold_block_step(f, α) = _isindicator(f) ? f : _fold_level_step(f, α)
_fold_block_step(f::ZeroFunction, α) = f
function _fold_level_step(f::ElasticNet, α)
  c1 = _folded_level(f.λ1, α)
  c = iszero(f.λ2) ? f.λ2 : _apply_scale(α, f.λ2)
  c2 = isfinite(c) && abs(c) < floatmin(typeof(float(c))) ? zero(c) : _folded_level(f.λ2, α)
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
# `_fold_level(f, probe)` runs the fold of every function that has one (the levels of `ElasticNet`
# and `SparseGroupNormL2`, a `Translate`, `Sum` or `SeparableSum` of such functions with the
# factors of their `Scaled` terms, the radius of the conjugate of a ball) with a scale that records
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
struct _LevelProbe{P<:_ProjectionScale}
  w::_ShiftWindow
  a::P                     # the factors of the `Scaled` terms inside, as a product
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
# The recoveries read the entries of the argument: an owner is read as the matrix that it represents, which
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
_exactly_zero_at(f::Union{NormL2, NormLinf, NuclearNorm}, x::AbstractArray) = iszero(f.λ) || all(iszero, _host_entries(x))
function _exactly_zero_at(f::Union{NormL1, NormL1Nonneg, ElasticNet}, x::AbstractArray)
  h, w = _host_entries(x), _host_entries(f.w)
  w === nothing || length(w) == length(h) || return false
  λ = f isa ElasticNet ? f.λ1 : f.λ0
  quadratic = f isa ElasticNet && f.λ2 > 0
  for (j, i) in enumerate(eachindex(h))
    wi = w === nothing ? one(λ) : w[j]
    (quadratic || _penalized(λ, wi)) && !iszero(h[i]) && return false
  end
  true
end
function _exactly_zero_at(f::Union{GroupNormL2, GroupNormLinf, SparseGroupNormL2}, x::AbstractArray)
  h = _host_entries(x)
  length(f.group_of) <= length(h) || return false
  l1 = f isa SparseGroupNormL2 && f.λ1 > 0
  for (i, xi) in enumerate(h)
    k = i <= length(f.group_of) ? f.group_of[i] : 0
    k != 0 && (l1 || _penalized(f.λ0, f.w[k])) && !iszero(xi) && return false
  end
  true
end

# The scaled-value mechanism (R1, "Wrappers"). A function with `_has_value_parts` returns its
# value at x as (m, e), the value m 2^e with a finite significand m, or `nothing` to keep the
# ordinary result (`_value_parts`, defaults in utils.jl). Only host arrays take it; a NaN value
# stays NaN, and a normal value is its own significand and exponent, so the function is asked
# only for a value that is infinite, zero or subnormal. An exactly zero value whose true value,
# times α, cannot be a normal number (`_zero_value_exponent`) is not recomputed.
@inline _parts_argument(x) = false
@inline _parts_argument(::Union{Array, _HostArray, SparseIterate, SymmetricSparseIterate, SymmetricMatrixView, AtomIterate,
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
  _scaled_product(convert(P, ma), convert(P, m), ea + e)
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
# an owner gradient: every coordinate that is stored, which are the coordinates of the symmetric gradient
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
  # one test for the ordinary product: a loop over `prox_entry` still vectorizes with a single
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
# in a loop over `prox_entry` are the ones that are not inlined (`_prox_entry_nested`).
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
  g = _fold_level_step(f, _ProjectionScale(ma, _exponent_sum(typeof(ma), ea, eγ)))
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
Base.@propagate_inbounds function prox_entry(s::Scaled, v::Real, i::Integer, γ::Real)
  _nested_route(s, γ) && return _prox_entry_nested(s, v, i, γ)
  prox_entry(s.f, v, i, s.α * γ)
end
@noinline function _prox_entry_nested(s::Scaled, v::Real, i::Integer, γ::Real)
  g, step, _ = _nested_step(s, γ)
  prox_entry(g, v, i, step)
end
_checkparams(s::Scaled, x) = _checkparams(s.f, x)
_checkwrite(op::O, s::Scaled, y, x) where {O} = (_checkreal(s, y); _checkreal(s, x); (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x); _checkwrite(op, s.f, y, x))
iszeropreserving(s::Scaled) = isfinite(s.α) && iszeropreserving(s.f)
_isindicator(::Type{<:Scaled{F}}) where {F} = _isindicator(F)
_counts_nonzeros(f::Scaled) = _counts_nonzeros(f.f)
_scale_invariant(f::Scaled) = _scale_invariant(f.f)
# a scaled group penalty updates a group of an AtomIterate with a scaled step
function prox_coordinate!(x::AtomIterate, s::Scaled, k::Integer, γ::Real)
  if _nested_route(s, γ)
    g, step, _ = _nested_step(s, γ)
    return prox_coordinate!(x, g, k, step)
  end
  prox_coordinate!(x, s.f, k, s.α * γ)
end
@inline _check_coordinate(x::AtomIterate, s::Scaled, k) = _check_coordinate(x, s.f, k)
@inline function _block_step!(x::AtomIterate, s::Scaled, k, u, γ, ctx)
  if _nested_route(s, γ)
    g, step, _ = _nested_step(s, γ)
    return _block_step!(x, g, k, u, step, ctx)
  end
  _block_step!(x, s.f, k, u, s.α * γ, ctx)
end
kkt_violation(s::Scaled, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); _chained(s) ? _nested_kkt((f, h) -> kkt_violation(f, h, i), s, g) :
   iszero(s.α) ? abs(g[i]) : s.α * kkt_violation(s.f, _ScaledArray(g, inv(s.α)), i))
for X in (:AbstractArray, :AtomIterate)
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
_restrict_unchecked(s::Scaled, S::AbstractVector{<:Integer}) = Scaled(_restrict_unchecked(s.f, S), s.α)

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
  _ProjectionScale(m, _exponent_sum(typeof(m), ea, a.e, e))
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

####################################
#  f + ⟨c, ⋅⟩ + d
####################################

# Tuple parameters of Tilt and Translate pair with the argument block by block, also for
# nested tuples. Every leaf of the parameter is an array of the real entries, and its size is
# exactly the size of the block of the argument: equal lengths do not suffice, and nothing
# broadcasts. The structure is known only when the argument is, so it is checked in the oracles.
const _BlockParameter = Union{AbstractArray, Tuple}
_check_parameter_leaves(name, c::AbstractArray) = eltype(c) <: Real || _real_error(name)
function _check_parameter_leaves(name, c::Tuple)
  isempty(c) && _throw_empty_parameter(name)
  _check_parameter_leaf_blocks(name, c)
end
@generated _check_parameter_leaf_blocks(name, c::Tuple) =
  _blockwise(k -> :(_check_parameter_leaves(name, c[$k])), fieldcount(c), true)
_check_parameter_leaves(name, c) = _throw_parameter_leaf(name, c)
@noinline _throw_empty_parameter(name) = throw(ArgumentError("a tuple parameter of $name needs at least one block"))
@noinline _throw_parameter_leaf(name, c) = throw(ArgumentError(
  "the blocks of a tuple parameter of $name must be arrays or tuples of arrays, got $(typeof(c))"))

# SparseIterate, SymmetricSparseIterate and AtomIterate keep their stored entries and, for the
# symmetric one, the per-entry parameter checks, which the staged tuple buffers do not carry over:
# they are not accepted as blocks of a tuple variable. Other array types are not restricted.
_check_iterate_block(x) = true
_check_iterate_block(x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate}) = throw(ArgumentError(
  "Tilt and Translate on tuples do not support a $(nameof(typeof(x))) as a block"))
_check_tuple_parameter(c::AbstractArray, x::AbstractArray) =
  (_check_iterate_block(x); size(c) == size(x) || _parameter_size_error(size(c), size(x)))
# an owner is refused as a block, whatever the parameter
_check_tuple_parameter(c, x::SymmetricSparseIterate) = _check_iterate_block(x)
function _check_tuple_parameter(c::Tuple, x::Tuple)
  length(c) == length(x) || _parameter_arity_error(length(c), length(x))
  _check_tuple_parameter_blocks(c, x)
end
_check_tuple_parameter(c, x) = _parameter_structure_error()
@generated _check_tuple_parameter_blocks(c::Tuple, x::Tuple) =
  _blockwise(k -> :(_check_tuple_parameter(c[$k], x[$k])), min(fieldcount(c), fieldcount(x)), true)
@noinline _parameter_size_error(sc, sx) =
  throw(DimensionMismatch("a block of the parameter has size $sc and the block of the argument $sx"))
@noinline _parameter_arity_error(n, m) =
  throw(DimensionMismatch("the parameter has $n blocks and the argument $m"))
@noinline _parameter_structure_error() =
  throw(DimensionMismatch("the parameter and the argument do not have the same block structure"))
# Tilt and Translate write whole blocks, which these wrappers refuse outside their structure; the
# type decides, so the check reads no entry and runs before the first block is written
const _StructuredOutput = Union{Symmetric, Hermitian, Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal,
  LinearAlgebra.AbstractTriangular, UpperHessenberg}
_check_block_output(y) = _check_iterate_block(y)
_check_block_output(::_StructuredOutput) = throw(ArgumentError(
  "Tilt and Translate need an output that accepts every entry; use a dense matrix, not a Symmetric, Hermitian, Diagonal, triangular or banded one"))
@generated _check_block_output(y::Tuple) = _blockwise(k -> :(_check_block_output(y[$k])), fieldcount(y), true)
_checkparam(c::Tuple, x) = _check_tuple_parameter(c, x)
_checkparam(c::AbstractArray, x::Tuple) = _check_tuple_parameter(c, x)
_checkdot(c::Tuple, x::Tuple) = _check_tuple_parameter(c, x)

# z = x - γc, in the precision of z, and the translation z = x - b, block by block
_shift_tilt!(z::AbstractArray, x::AbstractArray, γ, c::AbstractArray) =
  (z .= x .- convert(float(eltype(z)), γ) .* _checked_parameter(z, c))
@generated _shift_tilt!(z::Tuple, x::Tuple, γ, c::Tuple) =
  _blockwise(k -> :(_shift_tilt!(z[$k], x[$k], γ, c[$k])), min(fieldcount(z), fieldcount(x), fieldcount(c)), :z)
_shift_translate!(z::AbstractArray, x::AbstractArray, b::AbstractArray) = (z .= x .- _checked_parameter(z, b))
@generated _shift_translate!(z::Tuple, x::Tuple, b::Tuple) =
  _blockwise(k -> :(_shift_translate!(z[$k], x[$k], b[$k])), min(fieldcount(z), fieldcount(x), fieldcount(b)), :z)

"""
    Tilt(f, c, d = 0)

The function `f(x) + ⟨c, x⟩ + d`, with the roles of `f`. Its proximal operator
is that of `f` at `x - γc`, and its gradient `∇f + c`. The linear terms of
augmented Lagrangians and of duality take this form.
The proximal operator passes `x - γc` to `f` in a buffer that the function owns,
so do not share one object between threads. A rejected inner call leaves the output
unchanged. Sparse iterate buffers preserve stored entries and their order. `Symmetric` and `Hermitian` outputs are unsupported.

For a variable that is a tuple of arrays, possibly nested, as for a
[`SeparableSum`](@ref), `c` is a tuple of the same structure whose leaves are
real arrays of the sizes of the blocks of `x`; `⟨c, x⟩` is the sum over the blocks and
every block is shifted on its own. A [`SparseIterate`](@ref), [`SymmetricSparseIterate`](@ref)
or [`AtomIterate`](@ref) is refused as a block of a tuple, and so is an output block that is
`Symmetric`, `Hermitian`, `Diagonal`, triangular or banded, which cannot take every entry.
The oracles check the structure, every size and these types
before writing, and the proximal operator stages the result in the buffer and writes
it to the output last, so a predictable failure leaves the output unchanged.
The tuple methods cover the value, the gradient, the proximal operator, the
Hessian-vector product and the quadratic model; the coordinate-level interfaces
(`prox_entry`, `kkt_violation`, `restrict`) take array coefficients only.
"""
struct Tilt{F<:AbstractFunction, C<:_BlockParameter, D<:Real} <: AbstractFunction
  f::F
  c::C
  d::D
  scratch::Scratch

  function Tilt{F, C, D}(f::F, c::C, d::D, scratch::Scratch) where {F<:AbstractFunction, C<:_BlockParameter, D<:Real}
    _check_parameter_leaves(:Tilt, c)
    new{F, C, D}(f, c, d, scratch)
  end
end
Tilt(f::F, c::C, d::D, scratch::Scratch) where {F<:AbstractFunction, C<:_BlockParameter, D<:Real} = Tilt{F, C, D}(f, c, d, scratch)
Tilt{F, C, D}(f::F, c::C, d::D) where {F<:AbstractFunction, C<:_BlockParameter, D<:Real} = Tilt{F, C, D}(f, c, d, Scratch())
Tilt(f::AbstractFunction, c::_BlockParameter, d::Real) = Tilt(f, c, d, Scratch())
Tilt(f::AbstractFunction, c::_BlockParameter) = (_check_parameter_leaves(:Tilt, c); Tilt(f, c, zero(_valtype(c))))

for trait in (:issmooth, :isproximable, :isseparable, :isconvex)
  @eval $trait(::Type{<:Tilt{F}}) where {F} = $trait(F)
end

(t::Tilt)(x) =
  (_check_value_arrays(t, x); _value_checked(t, x))
_value_unchecked(t::Tilt, x) = (_checkreal(t, x); _value_at(_value_unchecked(t.f, x) + _tuple_dot(t.c, x) + t.d, x))
_checkvalue(t::Tilt, x) = (_checkvalue(t.f, x); _checkdot(t.c, x); true)
value_and_gradient!(g, t::Tilt, x) =
  (_check_oracle_arrays(value_and_gradient!, t, g, x); _value_and_gradient_unchecked!(g, t, x))
_value_and_gradient_unchecked!(g, t::Tilt, x) = (_checkwrite(value_and_gradient!, t, g, x); v = _value_and_gradient_unchecked!(g, t.f, x); _add_tilt!(g, t.c); _value_at(v + _tuple_dot(t.c, x) + t.d, x))
gradient!(g, t::Tilt, x) =
  (_check_oracle_arrays(gradient!, t, g, x); _gradient_unchecked!(g, t, x))
_gradient_unchecked!(g, t::Tilt, x) = (_checkwrite(gradient!, t, g, x); _gradient_unchecked!(g, t.f, x); _add_tilt!(g, t.c))
@inline _add_tilt!(g, c) = (g .+= _checked_parameter(g, c))
_add_tilt!(g::SymmetricSparseIterate, c::AbstractMatrix) = _accumulate!(g, c)
_add_tilt!(g::Tuple, c::Tuple) = _accumulate!(g, c)

function _checkwrite(op::O, t::Tilt, g, x) where {O<:Union{typeof(gradient!), typeof(value_and_gradient!)}}
  _checkreal(t, g); _checkreal(t, x)
  _checkgradient_output(g, x)
  _checkparams(t, x)
  _check_tilt_axes(g, x, t.c)
  _checkwrite(op, t.f, g, x)
end
@inline _check_tilt_axes(g, x, c) = (Base.Broadcast.check_broadcast_axes(_baxes(g), c); _check_parameter_axes(g, c))
# exact sizes, so that no block of c broadcasts into a block of g
_check_tilt_axes(g::Tuple, x::Tuple, c::Tuple) = (_check_block_output(g); _checksize(g, x))

function _checkwrite(op::O, t::Tilt, y::_ArrayOrOwner, x::_ArrayOrOwner) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(t, y); _checkreal(t, x)
  _check_wrapper_output(y)
  _checksize(y, x)
  _checkparams(t, x)
  Base.Broadcast.check_broadcast_axes(_baxes(y), _vw(x), t.c)
  _check_parameter_axes(y, t.c)
  _checkwrite(op, t.f, y, y)
end

# x - γc in a buffer owned by t, so that y is unchanged when f throws
prox!(y::_ArrayOrOwner, t::Tilt, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, t, y, x, γ); _prox_unchecked!(y, t, x, γ))
function _prox_unchecked!(y::_ArrayOrOwner, t::Tilt, x::_ArrayOrOwner, γ::Real)
  _checkwrite(prox!, t, y, x)
  z = _output_buffer!(t.scratch, y)
  zz = _ow(z)
  zz .= _pt(zz, x) .- convert(float(eltype(y)), γ) .* _checked_parameter(z, t.c)
  w = _wrapper_output(y, z)
  _prox_unchecked!(w, t.f, z, γ)
  _wrapper_commit!(y, w)
end
value_and_prox!(y::_ArrayOrOwner, t::Tilt, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, t, y, x, γ); _value_and_prox_unchecked!(y, t, x, γ))
function _value_and_prox_unchecked!(y::_ArrayOrOwner, t::Tilt, x::_ArrayOrOwner, γ::Real)
  _checkwrite(value_and_prox!, t, y, x)
  z = _output_buffer!(t.scratch, y)
  zz = _ow(z)
  zz .= _pt(zz, x) .- convert(float(eltype(y)), γ) .* _checked_parameter(z, t.c)
  w = _wrapper_output(y, z)
  v = _value_and_prox_unchecked!(w, t.f, z, γ)
  v = _value_at(v + _tuple_dot(t.c, w) + t.d, w)
  _wrapper_commit!(y, w)
  v
end
# The proximal operator of a tuple stages the result: x - γc is formed in a buffer of the
# output's precision, f acts on it in place, and the output is written last.
function _checkwrite(op::O, t::Tilt, y::Tuple, x::Tuple) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(t, y); _checkreal(t, x)
  _check_block_output(y)
  _checksize(y, x)
  _checkparams(t, x)
  _checkwrite(op, t.f, y, y)
end
prox!(y::Tuple, t::Tilt, x::Tuple, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, t, y, x, γ); _prox_unchecked!(y, t, x, γ))
function _prox_unchecked!(y::Tuple, t::Tilt, x::Tuple, γ::Real)
  _checkwrite(prox!, t, y, x)
  z = _tuple_buffer!(t.scratch, y, x)
  _shift_tilt!(z, x, γ, t.c)
  _prox_unchecked!(z, t.f, z, γ)
  _projection_commit!(y, z)
end
value_and_prox!(y::Tuple, t::Tilt, x::Tuple, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, t, y, x, γ); _value_and_prox_unchecked!(y, t, x, γ))
function _value_and_prox_unchecked!(y::Tuple, t::Tilt, x::Tuple, γ::Real)
  _checkwrite(value_and_prox!, t, y, x)
  z = _tuple_buffer!(t.scratch, y, x)
  _shift_tilt!(z, x, γ, t.c)
  v = _value_and_prox_unchecked!(z, t.f, z, γ)
  # z has the entries that y will have (the buffer has the output's element types), so the value
  # is formed, and can fail, before the output is written
  result = _value_at(v + _tuple_dot(t.c, z) + t.d, y)
  _projection_commit!(y, z)
  result
end
Base.@propagate_inbounds prox_entry(t::Tilt, v::Real, i::Integer, γ::Real) = prox_entry(t.f, v - γ * t.c[i], i, γ)
_checkparams(t::Tilt, x) = (_checkparam(t.c, x); _checkparams(t.f, x))
kkt_violation(t::Tilt, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); kkt_violation(t.f, _ShiftedArray(g, t.c), i))
restrict(t::Tilt, S::AbstractVector{<:Integer}) = Tilt(restrict(t.f, S), t.c[S], t.d)
_restrict_unchecked(t::Tilt, S::AbstractVector{<:Integer}) = Tilt(_restrict_unchecked(t.f, S), t.c[S], t.d)

####################################
#  f(⋅ - b)
####################################

"""
    Translate(f, b)

The function `f(x - b)`, with the roles of `f`: its gradient is `∇f(x - b)` and
its proximal operator `b + prox_{γf}(x - b)`. For example, `Translate(NormL1(λ), β₀)`
is the penalty `λ ‖β - β₀‖₁` that shrinks towards `β₀`.
Values and gradients use a shifted buffer in the argument's precision, and the
proximal operator one in the output's, so do not share one object between threads.
A rejected inner call leaves the output unchanged. Sparse iterate buffers preserve
stored entries and their order.
`Symmetric` and `Hermitian` outputs are unsupported.

For a variable that is a tuple of arrays, possibly nested, as for a
[`SeparableSum`](@ref), `b` is a tuple of the same structure whose leaves are real
arrays of the sizes of the blocks of `x`, and every block is shifted on its own.
A [`SparseIterate`](@ref), [`SymmetricSparseIterate`](@ref) or [`AtomIterate`](@ref) is refused
as a block of a tuple, and so is an output block that is `Symmetric`, `Hermitian`, `Diagonal`,
triangular or banded, which cannot take every entry.
The oracles check the structure, every size and these types before writing, and the proximal
operator stages the result in the buffer and writes it to the output last, so a
predictable failure leaves the output unchanged. The tuple methods cover the value,
the gradient, the proximal operator, the Hessian-vector product and the quadratic
model; the coordinate-level interfaces (`prox_entry`, `restrict`) and
[`Conjugate`](@ref) take array translations only.
"""
struct Translate{F<:AbstractFunction, B<:_BlockParameter} <: AbstractFunction
  f::F
  b::B
  scratch::Scratch
  proxscratch::Scratch

  function Translate{F, B}(f::F, b::B, scratch::Scratch, proxscratch::Scratch) where {F<:AbstractFunction, B<:_BlockParameter}
    _check_parameter_leaves(:Translate, b)
    new{F, B}(f, b, scratch, proxscratch)
  end
end
Translate(f::F, b::B, scratch::Scratch, proxscratch::Scratch) where {F<:AbstractFunction, B<:_BlockParameter} = Translate{F, B}(f, b, scratch, proxscratch)
Translate{F, B}(f::F, b::B) where {F<:AbstractFunction, B<:_BlockParameter} = Translate{F, B}(f, b, Scratch(), Scratch())
Translate(f::AbstractFunction, b::_BlockParameter) = Translate(f, b, Scratch(), Scratch())

for trait in (:issmooth, :isproximable, :isseparable, :isconvex)
  @eval $trait(::Type{<:Translate{F}}) where {F} = $trait(F)
end

function _shifted_buffer(t::Translate, x::_ArrayOrOwner)
  size(x) == size(t.b) || throw(DimensionMismatch("x has size $(size(x)) and b $(size(t.b))"))
  _checkparams(t, x)
  _array_buffer!(t.scratch, float(eltype(x)), size(x))
end

# x - b in a buffer owned by t
function _shifted(t::Translate, x::_ArrayOrOwner)
  z = _shifted_buffer(t, x)
  z .= _vw(x) .- _checked_parameter(z, t.b)
end

# the shifted argument of a tuple, in a buffer of the argument's precision
function _shifted_buffer(t::Translate, x::Tuple)
  _checkparams(t, x)
  _tuple_buffer!(t.scratch, x, x)
end
_shifted(t::Translate, x::Tuple) = _shift_translate!(_shifted_buffer(t, x), x, t.b)

(t::Translate)(x::Union{_ArrayOrOwner, Tuple}) =
  (_check_value_arrays(t, x); _value_checked(t, x))
_value_unchecked(t::Translate, x::Union{_ArrayOrOwner, Tuple}) = _value_at(_value_unchecked(t.f, _shifted(t, x)), x)
value_and_gradient!(g, t::Translate, x::Union{_ArrayOrOwner, Tuple}) =
  (_check_oracle_arrays(value_and_gradient!, t, g, x); _value_and_gradient_unchecked!(g, t, x))
_value_and_gradient_unchecked!(g, t::Translate, x::Union{_ArrayOrOwner, Tuple}) =
  (_checkwrite(value_and_gradient!, t, g, x); _value_at(_value_and_gradient_unchecked!(g, t.f, _shifted(t, x)), x))
gradient!(g, t::Translate, x::Union{_ArrayOrOwner, Tuple}) =
  (_check_oracle_arrays(gradient!, t, g, x); _gradient_unchecked!(g, t, x))
_gradient_unchecked!(g, t::Translate, x::Union{_ArrayOrOwner, Tuple}) =
  (_checkwrite(gradient!, t, g, x); _gradient_unchecked!(g, t.f, _shifted(t, x)))

_checkwrite(op::O, t::Translate, g, x::_ArrayOrOwner) where {O<:Union{typeof(gradient!), typeof(value_and_gradient!)}} =
  (_checkgradient_output(g, x); _check_parameter_axes(g, t.b); _checkwrite(op, t.f, g, _checkshifted(t, x)))

function _checkshifted(t::Translate, x::_ArrayOrOwner)
  size(x) == size(t.b) || throw(DimensionMismatch("x has size $(size(x)) and b $(size(t.b))"))
  _checkparams(t, x)
  _checkbuffer(t.scratch, float(eltype(x)), size(x))
end
function _checkshifted(t::Translate, x::Tuple)
  _checkparams(t, x)
  _conjugate_checkbuffer(t.scratch, x, x)
end
_checkwrite(op::O, t::Translate, g::Tuple, x::Tuple) where {O<:Union{typeof(gradient!), typeof(value_and_gradient!)}} =
  (_checkreal(t, g); _checkreal(t, x); _checkgradient_output(g, x); _check_block_output(g); _checksize(g, x); _checkwrite(op, t.f, g, _checkshifted(t, x)))
_checkvalue(t::Translate, x::Tuple) = (_checkreal(t, x); _checkvalue(t.f, _checkshifted(t, x)))
_checkvalue(t::Translate, x::_ArrayOrOwner) = (_checkreal(t, x); _checkvalue(t.f, _checkshifted(t, x)))
# The value of f at x - b as a significand and an exponent, while x - b is in range: an argument
# that overflowed on translation is not recovered (#357).
_has_value_parts(::Type{<:Translate{F}}) where {F} = _has_value_parts(F)
_value_parts(t::Translate, x, v) = _value_parts_scaled(t, x, v, true)
function _value_parts_scaled(t::Translate, x::AbstractArray, v, α)
  z = _shifted(t, x)
  _all_finite(z) || return nothing
  _scaled_parts(t.f, α, z, v, nothing)
end
_value_parts_scaled(t::Translate, x, v, α) = nothing
# the bound of the inner function, for its value at the shifted argument as computed (the argument
# the ordinary value and the parts both use), which has the length of x
_zero_value_exponent(t::Translate, x) = _zero_value_exponent(t.f, x)
_infinite_value_exact(t::Translate, x) = _infinite_value_exact(t.f, x)
_parameter_alias(t::Translate, out) = _mightalias_parameter(out, t.b) || _parameter_alias(t.f, out)

function _checkwrite(op::O, t::Translate, y::_ArrayOrOwner, x::_ArrayOrOwner) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(t, y); _checkreal(t, x)
  _check_wrapper_output(y)
  _checksize(y, x)
  _checksize(t.b, x)
  _checkparams(t, x)
  _check_parameter_axes(y, t.b)
  _checkreal(t.f, x)
  _checkwrite(op, t.f, y, y)
end

# x - b in a buffer owned by t, so that y is unchanged when f throws
prox!(y::_ArrayOrOwner, t::Translate, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, t, y, x, γ); _prox_unchecked!(y, t, x, γ))
function _prox_unchecked!(y::_ArrayOrOwner, t::Translate, x::_ArrayOrOwner, γ::Real)
  _checkwrite(prox!, t, y, x)
  z = _output_buffer!(t.proxscratch, y)
  zz = _ow(z)
  zz .= _pt(zz, x) .- _checked_parameter(z, t.b)
  w = _wrapper_output(y, z)
  _prox_unchecked!(w, t.f, z, γ)
  ww = _ow(w)
  ww .+= _checked_parameter(w, t.b)
  _wrapper_commit!(y, w)
end
value_and_prox!(y::_ArrayOrOwner, t::Translate, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, t, y, x, γ); _value_and_prox_unchecked!(y, t, x, γ))
function _value_and_prox_unchecked!(y::_ArrayOrOwner, t::Translate, x::_ArrayOrOwner, γ::Real)
  _checkwrite(value_and_prox!, t, y, x)
  z = _output_buffer!(t.proxscratch, y)
  zz = _ow(z)
  zz .= _pt(zz, x) .- _checked_parameter(z, t.b)
  w = _wrapper_output(y, z)
  v = _value_and_prox_unchecked!(w, t.f, z, γ)
  ww = _ow(w)
  ww .+= _checked_parameter(w, t.b)
  _wrapper_commit!(y, w)
  _value_at(v, y)
end
# As for Tilt, the proximal operator of a tuple works in place on the staged x - b
function _checkwrite(op::O, t::Translate, y::Tuple, x::Tuple) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(t, y); _checkreal(t, x)
  _check_block_output(y)
  _checksize(y, x)
  _checkparams(t, x)
  _checkwrite(op, t.f, y, y)
end
prox!(y::Tuple, t::Translate, x::Tuple, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, t, y, x, γ); _prox_unchecked!(y, t, x, γ))
function _prox_unchecked!(y::Tuple, t::Translate, x::Tuple, γ::Real)
  _checkwrite(prox!, t, y, x)
  z = _tuple_buffer!(t.proxscratch, y, x)
  _shift_translate!(z, x, t.b)
  _prox_unchecked!(z, t.f, z, γ)
  _accumulate!(z, t.b)
  _projection_commit!(y, z)
end
value_and_prox!(y::Tuple, t::Translate, x::Tuple, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, t, y, x, γ); _value_and_prox_unchecked!(y, t, x, γ))
function _value_and_prox_unchecked!(y::Tuple, t::Translate, x::Tuple, γ::Real)
  _checkwrite(value_and_prox!, t, y, x)
  z = _tuple_buffer!(t.proxscratch, y, x)
  _shift_translate!(z, x, t.b)
  v = _value_and_prox_unchecked!(z, t.f, z, γ)
  _accumulate!(z, t.b)
  result = _value_at(v, y)
  _projection_commit!(y, z)
  result
end
Base.@propagate_inbounds prox_entry(t::Translate, v::Real, i::Integer, γ::Real) = t.b[i] + prox_entry(t.f, v - t.b[i], i, γ)
_checkparams(t::Translate, x) = (_checkparam(t.b, x); _checkparams(t.f, x))
_isindicator(::Type{<:Translate{F}}) where {F} = _isindicator(F)
# a nonseparable f requires zero translation outside S, which is about more
# than a dropped constant
_check_translation_outside(t::Translate, excluded) =
  any(i -> !iszero(t.b[i]), excluded) &&
    throw(ArgumentError("restrict of a nonseparable Translate requires zero translation outside S"))
# the dropped terms of a separable f are evaluated at -b, not at zero, so f is
# restricted unchecked once they are known to be finite
function restrict(t::Translate, S::AbstractVector{<:Integer})
  excluded = _excluded_coordinates(length(t.b), S)
  if !isseparable(t.f)
    _check_translation_outside(t, excluded)
    return Translate(restrict(t.f, S), t.b[S])
  end
  if !isempty(excluded)
    # Dropping finite terms is harmless for minimization, but dropping an
    # infinite term would turn an infeasible supported problem into a feasible one.
    v = _restrict_unchecked(t.f, excluded)(-t.b[excluded])
    isfinite(v) || throw(ArgumentError("excluded translated coordinates are outside the function's domain"))
  end
  Translate(_restrict_unchecked(t.f, S), t.b[S])
end
function _restrict_unchecked(t::Translate, S::AbstractVector{<:Integer})
  isseparable(t.f) || _check_translation_outside(t, _excluded_coordinates(length(t.b), S))
  Translate(_restrict_unchecked(t.f, S), t.b[S])
end

####################################
#  convex conjugate
####################################

"""
    Conjugate(f)

The convex conjugate `f*(y) = supₓ ⟨x, y⟩ - f(x)` of a convex `f`. Its proximal
operator follows from that of `f` by Moreau's identity,
`prox_{γf*}(x) = x - γ prox_{f/γ}(x/γ)`, so every proximable function gives one;
for a separable `f` so does [`prox_entry`](@ref). Its value is known for the
norms (the indicators of the dual balls), the balls (their support functions),
the box, [`IndSimplex`](@ref), [`SumLargest`](@ref), the nonnegative orthant,
[`SqrNormL2`](@ref), [`LinearFunction`](@ref),
[`ZeroFunction`](@ref), and the losses [`CheckLoss`](@ref), [`HingeLoss`](@ref),
[`SqrHingeLoss`](@ref) and [`EpsInsensitiveLoss`](@ref), and for translations and strictly positive
rescalings of these functions; for other functions, evaluating the conjugate
throws an `ArgumentError`. [`value_and_prox!`](@ref) returns the value at the
proximal point for every proximable `f`: where no closed form is known, it uses
the Fenchel equality `f*(y) = ⟨p, y⟩ - f(p)`, with `p` the corresponding
proximal point of `f`. Cancellation in this subtraction and errors in the inner
proximal point affect the reported value.

The conjugate is an indicator, and [`value_and_prox!`](@ref) returns `0` after
the proximal step, when `f` is positively homogeneous (the norms,
[`NuclearNorm`](@ref), the homogeneous group norms, [`SLOPE`](@ref),
[`NormL1Nonneg`](@ref), [`TracePSD`](@ref), [`TotalVariation1D`](@ref),
[`FusedLasso`](@ref), [`NormL1Fused`](@ref), [`SumLargest`](@ref),
[`CheckLoss`](@ref), [`ZeroFunction`](@ref)) or the indicator of a convex cone
([`IndNonnegative`](@ref), `IndPSD()`, [`IndIsotonic`](@ref),
[`IndPattern`](@ref), [`IndBanded`](@ref), [`IndToeplitz`](@ref), the
second-order and exponential cones). So is the conjugate of a positive
[`Scaled`](@ref), [`SingularValue`](@ref), [`Spectral`](@ref) or
[`SeparableSum`](@ref) of these, and `Conjugate(Conjugate(g))`, which is `g`,
for an indicator `g`. Not
[`LinearFunction`](@ref), whose conjugate is `-d` on its set. Another conjugate
that is an indicator, such as that of a precomposed norm, takes its value from
the Fenchel equality and can return a small nonzero value instead of zero;
this is not a feasibility certificate.
At an input with a non-finite entry, a direct `Conjugate(SingularValue(f))`
returns `Inf` if the proximal step returns.

The support functions of the ℓ1 and ℓ2 balls follow the overflow handling of
`NormLinf` and `NormL2`, including a radius not representable in the argument's precision.

The conjugates of `NormL1`, `NormL1Nonneg`, `CheckLoss`, `ZeroFunction`,
`IndNonnegative`, `NormL2`, `GroupNormL2`, `NormLinf` and `GroupNormLinf` use
direct box, ball or polar projections, including positive scaling, translations
and separable sums.
Clipping is accurate to the rounding of each coefficient. Euclidean projections
use scaled significands at exceptional magnitudes, preserving small coordinates
when the norm overflows or the projection factor underflows. Ungrouped dual
coordinates of `GroupNormL2` and `GroupNormLinf` are zero for every input,
including non-finite entries. Infinite weights
leave the corresponding dual coordinates or groups free, including at zero
level. Array steps and `prox_entry` use the same clipping formulas for separable
families. Translations require `γb` to be finite in working precision.

On known CPU storage with one-based input and output axes, the ℓ1-ball
projections use a certified native threshold path for
ordinary radii, and maximum-relative offsets with compensated scaled sums
otherwise. For a group of size n and radius r, the coordinate error is at
most 4n*eps(T)*r, with an additional half-subnormal rounding allowance.
Coefficient factors are compensated through wrapper depth; membership uses a
conservative lower radius bound. With working type R (at least Float64), floating
factors satisfy the stated bound for d*eps(R)<=1/128, where d counts wrapper
scales, the norm level and the group weight. Outside this regime, add twice
the explicitly accumulated coefficient error, in radius units, to the accuracy
and feasibility bounds. CPU inputs must have a concrete floating or
machine-integer element type; other real arrays require explicit conversion.
Coordinates outside the exact support satisfy the same coordinate error bound.
They are exactly zero where support is certified by the careful path or the
small-group shortcuts. Certified inside-ball inputs
and infinite radii return the input unchanged. Feasibility allows
4n*eps(T)*r plus n*nextfloat(zero(T)). Scalar steps cancel for untranslated
projections; translations return the linear value ⟨b,y⟩. Array steps retain
their existing behaviour for these nonseparable families. Other storage and
axes, including device ℓ∞ conjugates, retain Moreau's identity.

Other families use Moreau's identity, including `NuclearNorm`, singular-value
and spectral wrappers, `SLOPE`, TV and fused penalties, and cones other than
`IndNonnegative`. Cancellation can lose small results at large inputs. `SumLargest` retains its dedicated method. Where a
step-free Moreau computation is available it avoids division by the step;
otherwise finite inputs require a finite computed result, checked before writing.

Supported zero rescalings simplify in [`Scaled`](@ref) while preserving the
underlying domain. Each conjugate owns a working buffer; do not share it
between threads. Tuples, including nested tuples, use one buffer per array
block and support an output that is the input tuple.
"""
struct Conjugate{F<:AbstractFunction} <: AbstractFunction
  f::F
  scratch::Scratch

  function Conjugate{F}(f::F) where {F<:AbstractFunction}
    isconvex(f) || throw(ArgumentError("Conjugate requires a convex function"))
    new{F}(f, Scratch())
  end
end
Conjugate(f::AbstractFunction) = Conjugate{typeof(f)}(f)

function _fold_level(c::Conjugate{F}, α) where {F<:Union{IndBallL1, IndBallL2}}
  r = _folded_level(c.f.r, α)
  r === nothing ? nothing : Conjugate(c.f isa IndBallL1 ? IndBallL1(r) : IndBallL2(r))
end

# The conjugate of a positively homogeneous convex function, a support function
# such as a norm, is the indicator of a set; that of the indicator of a convex
# cone is the indicator of its polar cone. Not LinearFunction: its conjugate is
# -d on its set.
_conjugate_isindicator(::Type) = false
_conjugate_isindicator(::Type{<:Union{NormL1, NormL2, NormLinf, NuclearNorm, HomogeneousGroupNorm, SLOPE,
                                      NormL1Nonneg, TracePSD, TotalVariation1D, FusedLasso, NormL1Fused,
                                      CheckLoss, ZeroFunction, SumLargest,
                                      IndNonnegative, IndPSD{Nothing}, IndIsotonic, IndPattern, IndBanded,
                                      IndToeplitz, IndSOC, IndRotatedSOC, IndExpPrimal, IndExpDual}}) = true
# positive rescalings and the singular values keep it; f** = f
_conjugate_isindicator(::Type{<:Union{Scaled{F}, SingularValue{F}}}) where {F} = _conjugate_isindicator(F)
_conjugate_isindicator(::Type{<:Conjugate{F}}) where {F} = _isindicator(F)
_isindicator(::Type{<:Conjugate{F}}) where {F} = _conjugate_isindicator(F)

# the value of an indicator at its projection
_zero_value(y) = zero(_valtype(y))
function _zero_value(y::Tuple)
  T = _valtype(y)
  T === Union{} ? 0 : zero(T)
end

isproximable(::Type{<:Conjugate{F}}) where {F} = isproximable(F)
isseparable(::Type{<:Conjugate{F}}) where {F} = isseparable(F)
isconvex(::Type{<:Conjugate}) = true
iszeropreserving(c::Conjugate) = iszeropreserving(c.f)

# With Pγ(f, x) = γ prox_{f/γ}(x/γ), prox_{γf*}(x) = x - Pγ(f, x). For a
# positively homogeneous f, or the indicator of a cone, Pγ(f, x) = prox_f(x) at
# every step; for the indicator of a set C, Pγ is the projection onto γC, which
# for the ℓ2 ball and the box needs no division by γ; Pγ(f(⋅ - b), x) =
# γb + Pγ(f, x - γb), and Pγ(αg) = Pγ(g) for an indicator g and α > 0.
_moreau_scaled(::Type) = false
_moreau_scaled(::Type{<:Union{IndBallL2, IndBox}}) = true
_moreau_scaled(::Type{<:Translate{F}}) where {F} = _moreau_free(F)
_moreau_scaled(::Type{<:Scaled{F}}) where {F} = _isindicator(F) && _moreau_free(F)
_moreau_free(::Type{F}) where {F} = _conjugate_isindicator(F) || _moreau_scaled(F)
# a translation forms γb, which can overflow, so its result is checked
_moreau_checked(::Type) = false
_moreau_checked(::Type{<:Translate}) = true
_moreau_checked(::Type{<:Scaled{F}}) where {F} = _moreau_checked(F)

# Pγ(f, x) into z, for _moreau_free(typeof(f)), with γ in the precision of z;
# z may be x
_moreau_point!(z, f, x, γ) = _conjugate_isindicator(typeof(f)) ? _prox_unchecked!(z, f, x, one(γ)) : _moreau_set!(z, f, x, γ)
# γ times a bound in the precision S of γ, keeping an infinite or zero bound, so
# that a step that underflows to 0 or overflows to Inf never forms 0 * Inf. A
# finite bound that overflows in S, or a nonzero one that underflows, is
# dilated in the wider of the two types, so that it can still give a
# representable bound; only then, so that a bound of a wide type such as
# BigFloat does not allocate
@inline function _dilate(γ, b)
  S = float(typeof(γ))
  c = convert(S, b)
  if (isinf(c) & !isinf(b)) | (iszero(c) & !iszero(b))
    P = float(promote_type(typeof(γ), typeof(b)))
    convert(S, convert(P, γ) * convert(P, b))
  else
    ifelse(isinf(c) | iszero(c), c, convert(S, γ) * c)
  end
end
_moreau_set!(z, f::IndBallL2, x, γ) = _prox_unchecked!(z, IndBallL2(_dilate(γ, f.r)), x, one(γ))
# the dilated bounds: a scalar step and bound once, outside the broadcast, since
# converting a wide bound such as a BigFloat can allocate; otherwise lazily,
# entry by entry, inside it
_dilated(γ::Real, b::Real) = _dilate(γ, b)
_dilated(γ, b) = Base.Broadcast.broadcasted(_dilate, γ, b)
function _moreau_set!(z, f::IndBox, x, γ)
  step = _reshapeparam(γ, x)
  z .= clamp.(x, _dilated(step, _reshapeparam(f.l, x)), _dilated(step, _reshapeparam(f.u, x)))
end
_moreau_set!(z, s::Scaled, x, γ) = _moreau_point!(z, s.f, x, γ)
function _moreau_set!(z, t::Translate, x, γ)
  z .= x .- _dilate.(γ, _checked_parameter(z, t.b))
  _moreau_point!(z, t.f, z, γ)
  z .+= _dilate.(γ, _checked_parameter(z, t.b))
end

# NaN exactly when x is finite and x - γz in the precision T of the output is
# not (Inf * 0 and NaN * 0 are NaN, a finite value times 0 is 0), so that the
# check is a sum without branches, which vectorizes
struct _NonfiniteMoreau{T} end
@inline (::_NonfiniteMoreau{T})(x, γ, z) where {T} = ifelse(isfinite(x), convert(T, x - γ * z) * zero(T), zero(T))
_moreau_step(γ::Real, ::Type{T}) where {T} = convert(T, γ)
_moreau_step(γ::AbstractArray, ::Type) = γ
# The reciprocal 1/γ of a step in Moreau's identity is a combined expression (numerics T3, R3): it is
# formed from the stored γ in P, the promotion of the model precision W and the float type of γ, and
# passed on in P. Rounding 1/γ to the type of γ first, as inv(γ) does, puts the error of a Float32
# step into a Float64 model. W is the promotion over the destination and the argument, over the
# blocks of a tuple: a block-level call uses its own block's types.
@inline _moreau_reciprocal(::Type{W}, γ::Real) where {W} = inv(convert(promote_type(W, float(typeof(γ))), γ))
# the model precision of a Conjugate call: the destination, the argument and the data of f (model_data.jl)
@inline _moreau_model(f, y, x) = promote_type(_valtype(y), _valtype(x), _data_of(f))
@inline _moreau_reciprocal(f, y, x, γ::Real) = _moreau_reciprocal(_moreau_model(f, y, x), γ)
# The step of a tuple call, widened (exactly, or as it is when already wider) to the promotion P of the
# model precision over all blocks and its own type. Every block then forms its reciprocal in the same P
# as the preflight of the whole tuple, whatever the precision of the block itself.
@inline _moreau_tuple_step(f, y, x, γ::Real) = convert(promote_type(_moreau_model(f, y, x), float(typeof(γ))), γ)
# a/γ for a coefficient a, formed in the promotion of W and the types of a and γ, rounded once
@inline function _moreau_ratio(::Type{W}, a::Real, γ::Real) where {W}
  P = promote_type(W, float(typeof(a)), float(typeof(γ)))
  convert(P, a) / convert(P, γ)
end
# whether y = x - γz, block by block, is not finite at a finite input; with
# always = false, only an input wider than the output is checked
function _moreau_nonfinite(y::AbstractArray, x::AbstractArray, z::AbstractArray, γ, always::Bool)
  T = float(eltype(y))
  always || promote_type(eltype(x), T) != T || return false
  term = Base.Broadcast.broadcasted(_NonfiniteMoreau{T}(), x, _moreau_step(γ, T), z)
  bc = Base.Broadcast.instantiate(term)
  # an empty array has nothing to check; a sum without an initial value is
  # pairwise and vectorizes, where one with an initial value is a sequential fold
  !isempty(x) && isnan(sum(bc))
end
@generated _moreau_nonfinite(y::Tuple, x::Tuple, z::Tuple, γ, always::Bool) =
  _blockany(k -> :(_moreau_nonfinite(y[$k], x[$k], z[$k], γ, always)), fieldcount(y))
@noinline _moreau_error(γ) =
  throw(ArgumentError("the proximal point of the conjugate is not finite at a finite input " *
                      "for the step γ = $(γ isa AbstractArray ? summary(γ) : γ) in working " *
                      "precision; rescale the step or the input"))
_checkmoreau(y, x, z, step, γ, always::Bool) =
  (_moreau_nonfinite(y, x, z, step, always) && _moreau_error(γ); true)

prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
function _prox_unchecked!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::Real)
  _checkwrite(prox!, c, y, x)
  if _projection_direct_output(c.f, y, x)
    result = _conjugate_projection!(y, c.f, x, γ)
    result isa _L1ProjectionFallback && _projection_l1_moreau_blocks!(y, c.f, x, γ, 1, result, Val(false), prox!)
    return y
  end
  T = float(eltype(y))
  z = _array_buffer!(c.scratch, T, x)
  result = _projection_fallback(c.f, y, x) ? _conjugate_moreau!(z, c.f, x, γ) : _conjugate_projection!(z, c.f, x, γ)
  result isa _L1ProjectionFallback && _projection_l1_moreau_blocks!(z, c.f, x, γ, 1, result, Val(false), prox!)
  _check_projection_result(z, x, γ)
  copyto!(y, z)
end
# f** = f, with the checks of f's own oracles
_checkwrite(op::O, c::Conjugate{<:Conjugate}, y::Union{AbstractArray, Tuple}, x::Union{AbstractArray, Tuple}) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}} =
  _checkwrite(op, c.f.f, y, x)
prox!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
function _prox_unchecked!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::Real)
  _checkwrite(prox!, c, y, x)
  _prox_unchecked!(y, c.f.f, x, γ)
end
value_and_prox!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, c, y, x)
  _value_and_prox_unchecked!(y, c.f.f, x, γ)
end
Base.@propagate_inbounds prox_entry(c::Conjugate{<:Conjugate}, v::Real, i::Integer, γ::Real) =
  prox_entry(c.f.f, v, i, γ)
# At a proximal point y = x - γp, Fenchel equality gives f*(y) = <p,y> - f(p).
# Keep the direct value for known closed forms, especially exact indicator zeros.
# A convex indicator with a linear minimization oracle has the value of its support function
# when nothing more specific is known (see _support_value)
_has_conjugate_value(f::AbstractFunction) = _support_fallback(f)

# Whether the value of the conjugate is the generic support function through the oracle: a
# convex indicator with an oracle, and no direct formula. The sets of this package that have an
# oracle and a formula opt out below, and the wrappers whose value is that of their argument
# follow it. Only such a value can raise an error that no check of the sizes and parameters
# foresees, from the oracle of a set of the user, so value_and_prox! stages its result (see
# _staged_prox_value!).
_support_fallback(f::AbstractFunction) = _isindicator(f) && isconvex(f) && haslmo(f)
_support_fallback(::Union{IndSimplex, IndBallL1, IndBallL2, IndBox}) = false
_support_fallback(f::Union{Scaled, Translate, SingularValue}) = _support_fallback(f.f)
_has_conjugate_value(::Union{NormL1, NormL2, NormLinf, IndBallL1, IndBallL2, IndSOC, IndRotatedSOC,
                            IndExpPrimal, IndExpDual, ZeroFunction, IndNonnegative, IndBox, IndSimplex,
                            SumLargest, SqrNormL2,
                            LinearFunction, CheckLoss, HingeLoss, SqrHingeLoss, EpsInsensitiveLoss}) = true
_has_conjugate_value(f::Union{Scaled, Translate}) = _has_conjugate_value(f.f)
_has_conjugate_value(f::SingularValue) = _has_conjugate_value(f.f)

value_and_prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, c, y, x)
  _isindicator(c) && (_prox_unchecked!(y, c, x, γ); return _zero_value(y))
  if _any_direct_projection(c.f, x) && !_projection_fallback(c.f, y, x)
    z = _array_buffer!(c.scratch, float(eltype(y)), x)
    v = _conjugate_projection_value!(z, c.f, x, γ)
    _check_projection_result(z, x, γ)
    _projection_commit!(y, z)
    return _value_at(v, y)
  end
  _has_conjugate_value(c.f) && return _conjugate_prox_value!(y, c, x, γ)
  _checksize(y, x)
  T = float(eltype(y))
  step = convert(T, γ)
  p = _array_buffer!(c.scratch, T, x)
  p .= x ./ step
  fp = _value_and_prox_unchecked!(p, c.f, p, _moreau_reciprocal(c.f, y, x, γ))
  _checkmoreau(y, x, p, step, γ, true)
  y .= x .- step .* p
  _value_at(dot(p, y) - fp, y)
end
value_and_prox!(y::AbstractArray, c::Conjugate{<:SingularValue}, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::Conjugate{<:SingularValue}, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, c, y, x)
  finite = all(isfinite, x)
  value = invoke(_value_and_prox_unchecked!, Tuple{AbstractArray, Conjugate, AbstractArray, Real}, y, c, x, γ)
  finite ? value : typeof(value)(Inf)
end

# entrywise Pγ without dividing by γ where the arrays do; never throws, since
# entrywise wrappers write entry by entry
Base.@propagate_inbounds prox_entry(c::Conjugate, v::Real, i::Integer, γ::Real) = _conjugate_entry(c.f, v, i, γ)
Base.@propagate_inbounds function _conjugate_entry(f, v, i, γ)
  if isseparable(typeof(f)) && _conjugate_isindicator(typeof(f))
    S = float(promote_type(typeof(v), typeof(γ)))
    w = convert(S, v)
    w - prox_entry(f, w, i, one(S))
  else
    v - γ * prox_entry(f, v / γ, i, _moreau_reciprocal(promote_type(float(typeof(v)), _data_of(f)), γ))
  end
end
Base.@propagate_inbounds function _conjugate_entry(f::IndBox, v, i, γ)
  S = float(promote_type(typeof(v), typeof(γ)))
  w, g = convert(S, v), convert(S, γ)
  w - clamp(w, _dilate(g, _entry(f.l, i)), _dilate(g, _entry(f.u, i)))
end
_checkparams(c::Conjugate, x) = _checkparams(c.f, x)
function _checkvalue(c::Conjugate, x)
  _checkreal(c, x)
  _checkparams(c, x)
  _has_conjugate_value(c.f) || invoke(_conjugate_value, Tuple{AbstractFunction, Any}, c.f, x)
  _checkconjugate(c.f, x)
end

function _checkwrite(op::O, c::Conjugate, y::Union{AbstractArray, Tuple}, x::Union{AbstractArray, Tuple}) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(c, y); _checkreal(c, x)
  op === value_and_prox! && y isa AbstractArray && _checkparams(c, x)
  _checksize(y, x)
  _checkparams(c, x)
  _projection_l1_check_input(c.f, y, x)
  if _projection_direct_output(c.f, y, x)
    _checkwrite(prox!, c.f, y, x)
    return true
  end
  p = _conjugate_checkbuffer(c.scratch, y, x)
  # an indicator returns zero after the projection, without the inner value
  withvalue = op === value_and_prox! && !_isindicator(c)
  _checkwrite(withvalue && !_has_conjugate_value(c.f) ? value_and_prox! : prox!, c.f, p, p)
  # Direct L1 values need no scaled standalone-value check; prox checks suffice.
  withvalue && _has_conjugate_value(c.f) &&
    !(_projection_l1_leaf(typeof(c.f)) && _projection_l1_host(x)) && _checkconjugate(c.f, y)
  true
end
_conjugate_checkbuffer(s::Scratch, y::AbstractArray, x::AbstractArray) = _checkbuffer(s, float(eltype(y)), x)

(c::Conjugate)(y::Union{AbstractArray, Tuple}) =
  (_check_value_arrays(c, y); _value_checked(c, y))
_value_unchecked(c::Conjugate, y::Union{AbstractArray, Tuple}) = (_checkparams(c, y); _value_at(_conjugate_value(c.f, y), y))
# the value at the matrix that an owner represents
(c::Conjugate)(y::SymmetricSparseIterate) = (_check_value_arrays(c, y); _value_checked(c, y))
_value_unchecked(c::Conjugate, y::SymmetricSparseIterate) =
  (_checkparams(c, y); _value_at(_conjugate_value(c.f, matrixview(y)), y))

# The proximal point of the conjugate with an owner as the output: it is formed in a staged matrix by the
# kernel for matrices, which reads the owner point as its matrix, and written to the owner last, one
# coordinate at a time, after the commit check of a staged result (symmetric, in the axes of the owner, and
# converted without failure). A rejected call leaves the owner as it was. The staged matrix is a frame, not
# the scratch of c, which the kernel uses. An owner point with a dense output reads the matrix of the
# owner, whose parameters are checked first.
prox!(y::SymmetricSparseIterate, c::Conjugate, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
function _prox_unchecked!(y::SymmetricSparseIterate, c::Conjugate, x::_ArrayOrOwner, γ::Real)
  frame = _acquire_frame!(_STAGE_FRAMES)
  try
    z = _stage!(frame, y, x)
    _prox_unchecked!(z, c, _vw(x), γ)
    _stage_commit!(y, z, c.f)
  finally
    _release_frame!(_STAGE_FRAMES)
  end
  y
end
value_and_prox!(y::SymmetricSparseIterate, c::Conjugate, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::SymmetricSparseIterate, c::Conjugate, x::_ArrayOrOwner, γ::Real)
  frame = _acquire_frame!(_STAGE_FRAMES)
  try
    z = _stage!(frame, y, x)
    v = _value_and_prox_unchecked!(z, c, _vw(x), γ)
    _stage_commit!(y, z, c.f)
    return _value_at(v, y)
  finally
    _release_frame!(_STAGE_FRAMES)
  end
end
_stage!(s::Scratch, y::SymmetricSparseIterate, x) = _frame_array!(s, float(eltype(y)), _vw(x))
# op(y, f, x, γ) for an owner y by the kernel for matrices on a staged matrix, written to the owner last after
# the commit check of a staged result: a rejected call leaves the owner as it was
function _staged_owner!(op::O, y::SymmetricSparseIterate, f, x0, γ) where {O}
  x = _symmetric_argument(y, x0)
  frame = _acquire_frame!(_STAGE_FRAMES)
  try
    z = _stage!(frame, y, x)
    r = op(z, f, _vw(x), γ)
    _stage_commit!(y, z)
    return op === prox! ? y : _value_at(r, y)
  finally
    _release_frame!(_STAGE_FRAMES)
  end
end
_stage_commit!(y::SymmetricSparseIterate, z, f = nothing) =
  (_projection_check_commit(y, z, _rounding_symmetric_prox(f)); _store_lower!(y, z); y)
# f** = f: the oracles of f, in the domain of f (stored coordinates, for the entry kernels of an owner)
prox!(y::SymmetricSparseIterate, c::Conjugate{<:Conjugate}, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
_prox_unchecked!(y::SymmetricSparseIterate, c::Conjugate{<:Conjugate}, x::_ArrayOrOwner, γ::Real) =
  _prox_unchecked!(y, c.f.f, x, γ)
value_and_prox!(y::SymmetricSparseIterate, c::Conjugate{<:Conjugate}, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
_value_and_prox_unchecked!(y::SymmetricSparseIterate, c::Conjugate{<:Conjugate}, x::_ArrayOrOwner, γ::Real) =
  _value_and_prox_unchecked!(y, c.f.f, x, γ)
prox!(y::AbstractArray, c::Conjugate, x::SymmetricSparseIterate, γ::Real) =
  (_check_entry_arrays(c, x); prox!(y, c, matrixview(x), γ))
value_and_prox!(y::AbstractArray, c::Conjugate, x::SymmetricSparseIterate, γ::Real) =
  (_check_entry_arrays(c, x); value_and_prox!(y, c, matrixview(x), γ))

_checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!)}, c::Conjugate{<:SumLargest}, y::AbstractArray, x::AbstractArray) =
  _checkwrite(op, c.f, y, x)
prox!(y::AbstractArray, c::Conjugate{<:SumLargest}, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
_prox_unchecked!(y::AbstractArray, c::Conjugate{<:SumLargest}, x::AbstractArray, ::Real) =
  _sumlargest_prox!(y, c.f, x, 1, Val(true))

_tuple_buffer_type(::Type{<:AbstractArray{T, N}}) where {T, N} = Array{float(T), N}
# the staged block of an owner is a dense matrix, which the commit check turns into the lower triangle
_tuple_buffer_type(::Type{<:SymmetricSparseIterate{T}}) where {T} = Matrix{float(T)}
@generated _tuple_buffer_type(::Type{X}) where {X<:Tuple} =
  _inline(Expr(:curly, :Tuple, (:(_tuple_buffer_type(fieldtype(X, $k))) for k in 1:fieldcount(X))...))
_same_buffer_sizes(b::AbstractArray, x::AbstractArray) = size(b) == size(x)
_same_buffer_sizes(b::AbstractArray, x::SymmetricSparseIterate) = size(b) == size(x)
@generated _same_buffer_sizes(b::Tuple, x::Tuple) =
  _blockall(k -> :(_same_buffer_sizes(b[$k], x[$k])), fieldcount(b))

function _tuple_buffer!(s::Scratch, y::Y, x::X) where {Y<:Tuple, X<:Tuple}
  B = _tuple_buffer_type(Y)
  cached = s.buf
  if cached isa Tuple{Val{X}, Val{Y}, B}
    b = cached[3]
    _same_buffer_sizes(b, y) && return b
  end
  b = _gradient_buffer(y)
  s.buf = (Val{X}(), Val{Y}(), b)
  b::B
end

function _conjugate_checkbuffer(s::Scratch, y::Y, x::X) where {Y<:Tuple, X<:Tuple}
  B = _tuple_buffer_type(Y)
  cached = s.buf
  if cached isa Tuple{Val{X}, Val{Y}, B}
    b = cached[3]
    _same_buffer_sizes(b, y) && return b
  end
  _gradient_buffer(y)::B
end

_divide!(z::AbstractArray, x::AbstractArray, γ) = (z .= x ./ convert(float(eltype(z)), γ))
@generated _divide!(z::Tuple, x::Tuple, γ) = _blockwise(k -> :(_divide!(z[$k], x[$k], γ)), min(fieldcount(z), fieldcount(x)), :z)
_moreau_output!(y::AbstractArray, x::AbstractArray, p::AbstractArray, γ) =
  (y .= x .- convert(float(eltype(y)), γ) .* p)
@generated _moreau_output!(y::Tuple, x::Tuple, p::Tuple, γ) =
  _blockwise(k -> :(_moreau_output!(y[$k], x[$k], p[$k], γ)), min(fieldcount(y), fieldcount(x), fieldcount(p)), :y)
_tuple_dot(x::AbstractArray, y::AbstractArray) = dot(x, y)
# a parameter and an owner: the off-diagonal coordinates count at both of their positions
_tuple_dot(c::AbstractArray, x::SymmetricSparseIterate) = _owner_dot(c, x)
@generated _tuple_dot(x::Tuple, y::Tuple) = _blocksum(k -> :(_tuple_dot(x[$k], y[$k])), min(fieldcount(x), fieldcount(y)))

# as for arrays, with every block checked before any block is written
prox!(y::Tuple, c::Conjugate, x::Tuple, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
function _prox_unchecked!(y::Tuple, c::Conjugate, x::Tuple, γ0::Real)
  _checkwrite(prox!, c, y, x)
  γ = _moreau_tuple_step(c.f, y, x, γ0)
  p = _tuple_buffer!(c.scratch, y, x)
  _projection_fallback(c.f, y, x) ? _conjugate_projection_target!(p, c.f, y, x, γ) : _conjugate_projection!(p, c.f, x, γ)
  _check_projection_result(p, x, γ)
  _projection_commit!(y, p)
end
prox!(y::Tuple, c::Conjugate{<:Conjugate}, x::Tuple, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
_prox_unchecked!(y::Tuple, c::Conjugate{<:Conjugate}, x::Tuple, γ::Real) = (_checkwrite(prox!, c, y, x); _prox_unchecked!(y, c.f.f, x, γ))
value_and_prox!(y::Tuple, c::Conjugate{<:Conjugate}, x::Tuple, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
_value_and_prox_unchecked!(y::Tuple, c::Conjugate{<:Conjugate}, x::Tuple, γ::Real) =
  (_checkwrite(value_and_prox!, c, y, x); _value_and_prox_unchecked!(y, c.f.f, x, γ))

value_and_prox!(y::Tuple, c::Conjugate, x::Tuple, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::Tuple, c::Conjugate, x::Tuple, γ0::Real)
  _checkwrite(value_and_prox!, c, y, x)
  γ = _moreau_tuple_step(c.f, y, x, γ0)
  _isindicator(c) && (_prox_unchecked!(y, c, x, γ); return _zero_value(y))
  if _any_direct_projection(c.f, x)
    z = _tuple_buffer!(c.scratch, y, x)
    v = _projection_fallback(c.f, y, x) ? _conjugate_projection_target_value!(z, c.f, y, x, γ) :
      _conjugate_projection_value!(z, c.f, x, γ)
    _check_projection_result(z, x, γ)
    _projection_commit!(y, z)
    return _value_at(v, y)
  end
  _has_conjugate_value(c.f) && return _conjugate_prox_value!(y, c, x, γ)
  p = _tuple_buffer!(c.scratch, y, x)
  _divide!(p, x, γ)
  fp = _value_and_prox_unchecked!(p, c.f, p, _moreau_reciprocal(c.f, y, x, γ))
  _checkmoreau(y, x, p, γ, γ, true)
  _moreau_output!(y, x, p, γ)
  _value_at(_tuple_dot(p, y) - fp, y)
end

# (αf)*(y) = α f*(y/α), and (f(⋅-b))*(y) = f*(y) + ⟨b,y⟩.
# Zero scaling of an extended-valued function depends on its domain, so the
# positive-scaling formula deliberately does not cover it.
function _conjugate_value(f::Scaled, y::AbstractArray)
  _checkconjugate(f, y)
  # a chain keeps the exponent of the product of its factors until the leaf's level is formed
  _chained(f) && return _scaled_conjugate_value(f.f, y, f.α)
  T = float(eltype(y))
  α = convert(T, f.α)
  α * _conjugate_value(f.f, _ScaledArray(y, inv(α)))
end
function _conjugate_value(f::Scaled, y::Tuple)
  _checkconjugate(f, y)
  _scaled_conjugate_value(f.f, y, f.α)
end
function _conjugate_value(f::Translate, y::AbstractArray)
  _checkconjugate(f, y)
  _conjugate_value(f.f, y) + dot(f.b, y)
end

function _checkconjugate(f::Scaled, y::AbstractArray)
  _checkreal(f, y)
  f.α > 0 || throw(ArgumentError("conjugate values of Scaled require a positive scale"))
  _checkconjugate(f.f, _ScaledArray(y, inv(convert(float(eltype(y)), f.α))))
end
function _checkconjugate(f::Scaled, y::Tuple)
  _checkreal(f, y)
  f.α > 0 || throw(ArgumentError("conjugate values of Scaled require a positive scale"))
  _checkscaledconjugate(f.f, y, f.α)
end
_checkconjugate(f::Translate, y::AbstractArray) = (_checkreal(f, y); _checksize(f.b, y); _checkconjugate(f.f, y))
_checkconjugate(f::Union{NormL1, IndBox, IndSOC, IndRotatedSOC, IndExpPrimal, IndExpDual, SumLargest,
                          CheckLoss, HingeLoss, SqrHingeLoss, EpsInsensitiveLoss}, y::AbstractArray) = _checkparams(f, y)
_checkconjugate(f::LinearFunction, y::AbstractArray) = (_checkreal(f, y); size(y) == size(f.c) || throw(DimensionMismatch()))
_checkconjugate(f::SqrNormL2, y::AbstractArray) = (_checkreal(f, y); iszero(f.λ) || f.y === nothing || _checkdot(f.y, y); true)

@noinline _conjugate_value(f::AbstractFunction, y) =
  throw(ArgumentError("the value of the conjugate of $(typeof(f)) is not implemented; its proximal operator is"))

# The support function σ_C(y) = ⟨y, lmo(C, -y)⟩ of a convex indicator with an oracle. Every
# specialized method is more specific and takes precedence; a set without an oracle keeps the
# error above, as does the value of the conjugate of an indicator of an unbounded set.
function _conjugate_value(f::AbstractFunction, y::AbstractArray)
  _has_conjugate_value(f) || return invoke(_conjugate_value, Tuple{AbstractFunction, Any}, f, y)
  _checkconjugate(f, y)
  _support_value(f, y)
end
_checkconjugate(f::AbstractFunction, y::AbstractArray) = (_checkreal(f, y); _has_conjugate_value(f) && _checkparams(f, y); true)

# Scratch for the minimizers and the staged results: per task, so that tasks do not share it, and a
# stack of frames, so that an oracle that evaluates another conjugate while it holds its own
# destination does not overwrite it. A frame is a Scratch, reused by the calls at its depth.
mutable struct _Frames
  stack::Vector{Scratch}
  depth::Int
end
_Frames() = _Frames(Scratch[], 0)
# one stack for the minimizers and one for the staged results, which cache different buffers
const _VALUE_FRAMES = Val(:ProximalBase_value_frames)
const _STAGE_FRAMES = Val(:ProximalBase_stage_frames)
_frames(::Val{K}) where {K} = get!(_Frames, task_local_storage(), K)::_Frames
function _acquire_frame!(kind::Val)
  F = _frames(kind)
  F.depth += 1
  F.depth > length(F.stack) && push!(F.stack, Scratch())
  @inbounds F.stack[F.depth]
end
_release_frame!(kind::Val) = (F = _frames(kind); F.depth -= 1; nothing)

# An array of the shape of x in a frame. A vector is a view of a buffer that only grows, so that the
# blocks of a separable sum, or conjugates of different dimensions, can use one frame in turn without
# reallocating; an array of another shape is cached at its exact size, and reallocated when the
# shape changes at the same depth.
_frame_array!(s::Scratch, ::Type{T}, x::AbstractVector) where {T} =
  (n = length(x); view(_buffer!(s, T, n), 1:n))
_frame_array!(s::Scratch, ::Type{T}, x::AbstractArray) where {T} = _array_buffer!(s, T, x)

# σ_C(y) = ⟨y, s⟩ at the minimizer s of ⟨-y, ·⟩, written in the shape of y to a frame, the argument
# read as -y lazily
function _support_value(f, y::AbstractArray)
  T = float(eltype(y))
  frame = _acquire_frame!(_VALUE_FRAMES)
  try
    s = _frame_array!(frame, T, y)
    lmo!(s, f, _ScaledArray{T}(y, -1))
    return _support_dot(y, s, T)
  finally
    _release_frame!(_VALUE_FRAMES)
  end
end

# Σᵢ yᵢ sᵢ over the entries with sᵢ ≠ 0: an entry at zero adds nothing, also when yᵢ is infinite. A NaN in
# y is an undefined direction and gives NaN. The sum is in Float64 for the narrower types, which holds
# every product and sum of Float32 numbers. The ordinary case is one compensated pass. When a product
# is not finite, exceeds floatmax/2n or underflows, a second pass sorts the products into three
# groups: the large ones scaled down, the underflowing ones scaled up, both by a power of two, and
# the rest as they are; the three compensated sums are combined at the end.
function _support_dot(y, s, ::Type{T}) where {T}
  R = promote_type(T, Float64)
  R <: Union{Float32, Float64} || return _support_dot_plain(y, s, T)
  n = length(s)
  t = floatmax(R) / (2 * max(n, 1))
  lo = floatmin(R)
  # one compensated pass; its flags, set without branching, say that a product is not finite or
  # exceeds t, or underflowed although both factors are nonzero. The sum is then discarded.
  v = e = zero(R)
  bad = false
  @inbounds for i in 1:n
    yi = convert(R, y[i])
    si = convert(R, s[i])
    p = yi * si
    ap = abs(p)
    bad |= !(ap <= t) | ((ap < lo) & (yi != 0) & (si != 0))
    v, e = _compensated_add(v, e, p)
  end
  bad || return convert(T, v + e)
  _support_dot_scaled(y, s, T, R, t, lo)
end

function _support_dot_scaled(y, s, ::Type{T}, ::Type{R}, t::R, lo::R) where {T, R}
  k = exponent(floatmax(R)) ÷ 2 + 16
  up, down = ldexp(one(R), k), ldexp(one(R), -k)
  mid = em = big = eb = tiny = et = nonfinite = zero(R)
  anynonfinite = false
  @inbounds for i in 1:length(s)
    yi = convert(R, y[i])
    isnan(yi) && return convert(T, NaN)
    si = convert(R, s[i])
    iszero(si) && continue
    if !(isfinite(yi) && isfinite(si))
      nonfinite += yi * si
      anynonfinite = true
      continue
    end
    iszero(yi) && continue
    p = yi * si
    ap = abs(p)
    if ap > t
      big, eb = _compensated_add(big, eb, (yi * down) * (si * down))
    elseif ap < lo
      tiny, et = _compensated_add(tiny, et, (yi * up) * (si * up))
    else
      mid, em = _compensated_add(mid, em, p)
    end
  end
  anynonfinite && return convert(T, nonfinite)
  r = mid + em
  b = big + eb
  if !iszero(b)
    rb = ldexp(b, 2k)
    r = isfinite(rb) ? r + rb : ldexp(b + ldexp(r, -2k), 2k)
  end
  tn = tiny + et
  iszero(tn) || (r += ldexp(tn, -2k))
  convert(T, r)
end

# element types without a fixed range, such as BigFloat
function _support_dot_plain(y, s, ::Type{T}) where {T}
  v = e = naive = zero(T)
  @inbounds for i in 1:length(s)
    yi = convert(T, y[i])
    isnan(yi) && return convert(T, NaN)
    si = s[i]
    iszero(si) && continue
    p = yi * si
    naive += p
    v, e = _compensated_add(v, e, p)
  end
  r = v + e
  isfinite(r) ? r : naive
end

# value_and_prox! of the conjugate of a set with an oracle but no formula. The oracle of the set is the
# user's, and can fail on its arguments after the proximal point exists, so the point and its value
# are computed in a frame, and only a success writes the output (y may be x). The frame is not
# the scratch of c, which the proximal operator of c uses; the cost is one copy of the result.
function _staged_prox_value!(y, c::Conjugate, x, γ)
  frame = _acquire_frame!(_STAGE_FRAMES)
  try
    z = _stage!(frame, y, x)
    _prox_unchecked!(z, c, x, γ)
    v = _conjugate_value(c.f, z)
    _stage_commit!(y, z)
    return _value_at(v, y)
  finally
    _release_frame!(_STAGE_FRAMES)
  end
end
_stage!(s::Scratch, y::AbstractArray, x::AbstractArray) = _frame_array!(s, float(eltype(y)), x)
_stage!(s::Scratch, y::Tuple, x::Tuple) = _tuple_buffer!(s, y, x)
_stage_commit!(y::AbstractArray, z) = copyto!(y, z)
_stage_commit!(y::Tuple, z) = _projection_commit!(y, z)

# the proximal point of the conjugate and its value, at the output
function _conjugate_prox_value!(y, c::Conjugate, x, γ)
  # a chain may refuse the value after the point exists: stage, and commit only on success
  (_support_fallback(c.f) || _chained_scaled(c.f)) && return _staged_prox_value!(y, c, x, γ)
  _prox_unchecked!(y, c, x, γ)
  _value_at(_conjugate_value(c.f, y), y)
end

# indicators of the dual balls and support functions of the balls, within the
# tolerance of the indicator functions; loops rather than closures, which would
# capture the element type as a DataType and dispatch dynamically
_within(a, b, ::Type{T}) where {T} = a <= b || a - b <= _tol(T) * max(one(T), abs(b))

function _conjugate_value(f::NormL1, y::AbstractArray)
  _checkconjugate(f, y)
  T = float(eltype(y))
  @inbounds for i in eachindex(IndexLinear(), y)
    _within(abs(y[i]), _weight_coefficient(f.λ0, _entry(f.w, i)), T) || return T(Inf)
  end
  zero(T)
end
_conjugate_value(f::NormL2, y::AbstractArray) = (_checkconjugate(f, y); T = float(eltype(y)); _ind(_within(_norm2(y), f.λ, T), T))
_conjugate_value(f::NormLinf, y::AbstractArray) = (_checkconjugate(f, y); T = float(eltype(y)); _ind(_within(norm(y, 1), f.λ, T), T))
_conjugate_value(f::IndBallL1, y::AbstractArray) = (_checkconjugate(f, y); _weighted_normlinf(f.r, y))
_conjugate_value(f::IndBallL2, y::AbstractArray) =
  (_checkconjugate(f, y); _weighted_norm2(f.r, y))
_conjugate_value(f::IndSOC, y::AbstractArray) =
  (_checkconjugate(f, y); _soc_value(f, y, true))
_conjugate_value(f::IndRotatedSOC, y::AbstractArray) =
  (_checkconjugate(f, y); _rotatedsoc_value(f, y, true))
_conjugate_value(f::IndExpPrimal, y::AbstractArray) =
  (_checkconjugate(f, y); _exp_value(f, y, true))
_conjugate_value(f::IndExpDual, y::AbstractArray) =
  (_checkconjugate(f, y); _exp_value(f, y, true))
function _conjugate_value(f::IndSimplex, y::AbstractArray)
  _checkconjugate(f, y)
  T = float(eltype(y))
  isempty(y) && return f.equality && f.r > 0 ? T(-Inf) : zero(T)
  iszero(f.r) && return zero(T)
  if isinf(f.r)
    f.equality && return T(-Inf)
    return _conjugate_value(IndNonnegative(), y)
  end
  m = maximum(y)
  convert(T, float(f.r) * (f.equality ? m : max(m, zero(T))))
end

function _conjugate_value(f::SumLargest, y::AbstractArray)
  _checkconjugate(f, y)
  T = float(eltype(y))
  S = float(promote_type(T, typeof(f.λ)))
  λ = S(f.λ)
  t = _tol(T)
  m = abs(λ)
  @inbounds for yi in y
    isfinite(yi) || return T(Inf)
    yi >= -t * max(one(S), λ) && _within(yi, λ, T) || return T(Inf)
    m = max(m, abs(yi))
  end
  e = _comparison_exponent(_magnitude_exponent(m))
  s = _scaled_sum(y, _power_factors(S, -e))
  mass = f.k * ldexp(λ, -e)
  _ind(abs(s - mass) <= t * max(ldexp(one(S), -e), abs(s), mass), T)
end

function _conjugate_value(f::ZeroFunction, y::AbstractArray)
  _checkconjugate(f, y)
  T = float(eltype(y))
  for v in y
    _within(abs(v), zero(T), T) || return T(Inf)
  end
  zero(T)
end
function _conjugate_value(f::IndNonnegative, y::AbstractArray)
  _checkconjugate(f, y)
  T = float(eltype(y))
  t = _tol(T)
  for v in y
    v <= t || return T(Inf)
  end
  zero(T)
end
function _conjugate_value(f::IndBox, y::AbstractArray)
  _checkconjugate(f, y)
  T = float(eltype(y))
  s = zero(T)
  @inbounds for i in eachindex(IndexLinear(), y)
    v = y[i]
    s += convert(T, v >= 0 ? (iszero(v) ? zero(T) : v * _entry(f.u, i)) : v * _entry(f.l, i))
  end
  s
end
function _conjugate_value(f::SqrNormL2, y::AbstractArray)
  _checkconjugate(f, y)
  iszero(f.λ) && return _conjugate_value(ZeroFunction(), y)
  T = float(eltype(y))
  S = float(promote_type(T, typeof(f.λ)))
  scale = inv(sqrt(convert(S, f.λ)) * sqrt(S(2)))
  r = _norm2(y)
  # Preserve sparse O(nnz) norms at ordinary scales. If the norm itself is
  # outside its working range, scale the entries before computing it.
  v = convert(T, isfinite(r) ? abs2(r * scale) : abs2(_norm2(_ScaledArray(y, scale))))
  f.y === nothing ? v : v + dot(f.y, y)
end
function _conjugate_value(f::LinearFunction, y::AbstractArray)
  _checkconjugate(f, y)
  T = float(eltype(y))
  @inbounds for i in eachindex(IndexLinear(), y)
    _within(abs(y[i] - f.c[i]), zero(T), T) || return T(Inf)
  end
  -convert(T, f.d)
end

####################################
#  blocks
####################################

"""
    SeparableSum(f1, f2, ...)

The function `Σₖ fₖ(xₖ)` of a tuple `x = (x1, x2, ...)` of blocks, with the
roles that all the `fₖ` share: the proximal operator, gradient and value act
block by block on tuples. Sparse plus low-rank models, for example, take
`SeparableSum(NormL1(λ), TracePSD(μ))` on the pair `(S, L)`.
Blocks may themselves be tuples for nested sums. Allocating oracles and
[`Conjugate`](@ref) preserve this block structure.
Before writing any output, the oracles check all blocks, including functions
inside wrappers, against the checks their own oracles run before writing:
shapes, parameters, and the scalar step, also through the supported wrappers.
The check covers the conditions known without computing, the step validation
that the supported functions and wrappers implement; it is not meant to reject
a step that the function itself accepts. It does not cover every failure. A
failure found only while computing can leave earlier blocks written: a step that
a function rejects according to the values it sees, `Conjugate`'s check that a
finite input gave a finite result, a search that finds a constraint empty, a
failure inside an external library, and the oracles of user-defined functions
beyond `check_parameters`. Block sums take scalar steps only.
"""
struct SeparableSum{F<:Tuple} <: AbstractFunction
  fs::F
end
SeparableSum(fs::AbstractFunction...) = SeparableSum(fs)
@generated _conjugate_isindicator(::Type{SeparableSum{F}}) where {F} = _blockall(k -> :(_conjugate_isindicator(fieldtype(F, $k))), fieldcount(F))

_argument_type(::Type{<:SeparableSum}) = Tuple

for trait in (:issmooth, :isproximable, :isconvex)
  @eval $trait(::Type{SeparableSum{F}}) where {F} = all($trait, fieldtypes(F))
end

function _checkblocks(s::SeparableSum, x::Tuple)
  length(x) == length(s.fs) || throw(DimensionMismatch("$(length(s.fs)) functions for $(length(x)) blocks"))
end

_checkparams(s::SeparableSum, x::Tuple) = (_checkblocks(s, x); _checkparams_blocks(s.fs, x))
@generated _checkparams_blocks(fs::Tuple, x::Tuple) = _blockwise(k -> :(_checkparams(fs[$k], x[$k])), min(fieldcount(fs), fieldcount(x)), true)
function _checkwrite(op::O, s::SeparableSum, y::Tuple, x::Tuple) where {O<:Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}}
  _checkreal(s, y); _checkreal(s, x)
  _checksize(y, x)
  _checkblocks(s, x)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checkwrite_blocks(op, s.fs, y, x)
end
# Inline the temporary wrapper so parameter arrays do not escape on Julia 1.10.
@inline _checkscaledconjugate_block(f, y, α) =
  _checkconjugate((@inline Scaled{typeof(f), typeof(α)}(f, α)), y)
@inline _checkscaledconjugate_block(f::Scaled, y, α) =
  _checkscaledconjugate_block(f.f, y, f.α * α)
@generated _checkwrite_blocks(op, fs::Tuple, y::Tuple, x::Tuple) =
  _blockwise(k -> :(_checkwrite(op, fs[$k], y[$k], x[$k])), fieldcount(fs), true)
@generated _has_conjugate_value(s::SeparableSum{F}) where {F} = _blockall(k -> :(_has_conjugate_value(s.fs[$k])), fieldcount(F))
@generated _support_fallback(s::SeparableSum{F}) where {F} = _blockany(k -> :(_support_fallback(s.fs[$k])), fieldcount(F))
_checkconjugate(s::SeparableSum, y::Tuple) = (_checkreal(s, y); _checkparams(s, y); _checkconjugate_blocks(s.fs, y))
@generated _checkconjugate_blocks(fs::Tuple, y::Tuple) = _blockwise(k -> :(_checkconjugate(fs[$k], y[$k])), min(fieldcount(fs), fieldcount(y)), true)
_checkscaledconjugate(s::SeparableSum, y::Tuple, α) = (_checkparams(s, y); _checkscaled_blocks(s.fs, y, α))
# the checks validate parameters only, so the rounded product of the factors serves
_checkscaledconjugate(f::Scaled, y::Tuple, α) = _checkscaledconjugate(f.f, y, f.α * α)
@generated _checkscaled_blocks(fs::Tuple, y::Tuple, α) =
  _blockwise(k -> :(_checkscaledconjugate_block(fs[$k], y[$k], α)), min(fieldcount(fs), fieldcount(y)), true)
_conjugate_value(s::SeparableSum, y::Tuple) = (_checkconjugate(s, y); _sum_conjugate(s.fs, y))
@generated _sum_conjugate(fs::Tuple, y::Tuple) = _blocksum(k -> :(_conjugate_value(fs[$k], y[$k])), min(fieldcount(fs), fieldcount(y)))
_scaled_conjugate_value(s::SeparableSum, y::Tuple, α) = (_checkscaledconjugate(s, y, α); _sum_scaled(s.fs, y, α))
@generated _sum_scaled(fs::Tuple, y::Tuple, α) = _blocksum(k -> :(_conjugate_value(Scaled(fs[$k], α), y[$k])), min(fieldcount(fs), fieldcount(y)))
_checkvalue(s::SeparableSum, x::Tuple) = (_checkreal(s, x); _checkparams(s, x); _checkvalue_blocks(s.fs, x))
@generated _checkvalue_blocks(fs::Tuple, x::Tuple) = _blockwise(k -> :(_checkvalue(fs[$k], x[$k])), min(fieldcount(fs), fieldcount(x)), true)

@generated _sum_values(fs::Tuple, x::Tuple) = _blocksum(k -> :(_value_unchecked(fs[$k], x[$k])), min(fieldcount(fs), fieldcount(x)))
@generated _prox_blocks!(fs::Tuple, y::Tuple, x::Tuple, γ) = _blockwise(k -> :(_prox_unchecked!(y[$k], fs[$k], x[$k], γ)), min(fieldcount(fs), fieldcount(y), fieldcount(x)), nothing)
@generated _value_and_prox_blocks!(fs::Tuple, y::Tuple, x::Tuple, γ) =
  _blocksum(k -> :(_value_and_prox_unchecked!(y[$k], fs[$k], x[$k], γ)), min(fieldcount(fs), fieldcount(y), fieldcount(x)))
@generated _gradient_blocks!(fs::Tuple, g::Tuple, x::Tuple) = _blockwise(k -> :(_gradient_unchecked!(g[$k], fs[$k], x[$k])), min(fieldcount(fs), fieldcount(g), fieldcount(x)), nothing)
@generated _value_and_gradient_blocks!(fs::Tuple, g::Tuple, x::Tuple) =
  _blocksum(k -> :(_value_and_gradient_unchecked!(g[$k], fs[$k], x[$k])), min(fieldcount(fs), fieldcount(g), fieldcount(x)))

(s::SeparableSum)(x::Tuple) =
  (_check_value_arrays(s, x); _value_checked(s, x))
_value_unchecked(s::SeparableSum, x::Tuple) = (_checkparams(s, x); _value_at(_sum_values(s.fs, x), x))
prox!(y::Tuple, s::SeparableSum, x::Tuple, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, s, y, x, γ); _prox_unchecked!(y, s, x, γ))
_prox_unchecked!(y::Tuple, s::SeparableSum, x::Tuple, γ::Real) =
  (_checkwrite(prox!, s, y, x, γ); _checkstep(prox!, s, y, x, γ); _owner_role_preflight(prox!, s, y, x, γ); _prox_blocks!(s.fs, y, x, γ); y)
value_and_prox!(y::Tuple, s::SeparableSum, x::Tuple, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, s, y, x, γ); _value_and_prox_unchecked!(y, s, x, γ))
_value_and_prox_unchecked!(y::Tuple, s::SeparableSum, x::Tuple, γ::Real) =
  (_checkwrite(value_and_prox!, s, y, x, γ); _checkstep(value_and_prox!, s, y, x, γ); _owner_role_preflight(value_and_prox!, s, y, x, γ); _value_at(_value_and_prox_blocks!(s.fs, y, x, γ), y))
value_and_gradient!(g::Tuple, s::SeparableSum, x::Tuple) =
  (_check_oracle_arrays(value_and_gradient!, s, g, x); _value_and_gradient_unchecked!(g, s, x))
_value_and_gradient_unchecked!(g::Tuple, s::SeparableSum, x::Tuple) =
  (_checkwrite(value_and_gradient!, s, g, x); _value_at(_value_and_gradient_blocks!(s.fs, g, x), x))
gradient!(g::Tuple, s::SeparableSum, x::Tuple) =
  (_check_oracle_arrays(gradient!, s, g, x); _gradient_unchecked!(g, s, x))
_gradient_unchecked!(g::Tuple, s::SeparableSum, x::Tuple) =
  (_checkwrite(gradient!, s, g, x); _gradient_blocks!(s.fs, g, x); g)

####################################
#  sums of smooth functions
####################################

"""
    Sum(f1, f2, ...)

The function `Σₖ fₖ(x)`, smooth when all the `fₖ` are, with the sum of their
gradients: a loss plus a smooth penalty, for example. A sum has no proximal
operator in general.

Every `fₖ` acts on the whole of `x`, unlike the blocks of a
[`SeparableSum`](@ref). When `x` is a tuple of arrays, possibly nested, the
gradient and the Hessian-vector product are the sums of those of the `fₖ`
block by block, with the structure of `x` kept; the `fₖ` must then accept a
tuple, as a `SeparableSum` does. The oracles check every `fₖ` against the
checks their own oracles run before writing, so a predictable failure in a
later function leaves the output unchanged.
"""
struct Sum{F<:Tuple} <: AbstractFunction
  fs::F
  scratch::Scratch
end
Sum(fs::AbstractFunction...) = Sum(fs, Scratch())

# α f(⋅ - b) is (α f)(⋅ - b), and the scale of a sum is the scale of every term. An indicator and
# the zero function do not change under a positive scale.
function _fold_level(t::Translate, α)
  g = _fold_level(t.f, α)
  g === nothing ? nothing : Translate(g, t.b, t.scratch, t.proxscratch)
end
_fold_block(f, α) = _isindicator(f) ? f : _fold_level(f, α)
_fold_block(f::ZeroFunction, α) = f
function _fold_blocks(fs::Tuple, α)
  gs = map(f -> _fold_block(f, α), fs)
  any(isnothing, gs) ? nothing : gs
end
function _fold_level(s::SeparableSum, α)
  gs = _fold_blocks(s.fs, α)
  gs === nothing ? nothing : SeparableSum(gs)
end
function _fold_level(s::Sum, α)
  gs = _fold_blocks(s.fs, α)
  gs === nothing ? nothing : Sum(gs, s.scratch)
end
function _fold_level_step(t::Translate, α)
  g = _fold_level_step(t.f, α)
  g === nothing ? nothing : Translate(g, t.b, t.scratch, t.proxscratch)
end
function _fold_level_step(s::SeparableSum, α)
  gs = map(f -> _fold_block_step(f, α), s.fs)
  any(isnothing, gs) ? nothing : SeparableSum(gs)
end
function _fold_level_step(s::Sum, α)
  gs = map(f -> _fold_block_step(f, α), s.fs)
  any(isnothing, gs) ? nothing : Sum(gs, s.scratch)
end
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
for (F, fields) in ((:NormL1, (:λ0,)), (:NormL1Nonneg, (:λ0,)), (:NormL2, (:λ,)), (:NormLinf, (:λ,)),
                    (:ElasticNet, (:λ1, :λ2)), (:GroupNormL2, (:λ0,)), (:GroupNormLinf, (:λ0,)),
                    (:SparseGroupNormL2, (:λ1, :λ0)), (:NormL1Fused, (:λ1, :λ2)), (:NuclearNorm, (:λ,)),
                    (:TracePSD, (:λ,)))
  @eval _has_infinite_level(f::$F) = $(foldl((a, b) -> :($a || isinf(f.$b)), fields[2:end]; init = :(isinf(f.$(fields[1])))))
end
_has_infinite_level(f::SLOPE) = any(isinf, f.λ)
_has_infinite_level(f::Scaled) = _has_infinite_level(f.f)
_has_infinite_level(f::Translate) = _has_infinite_level(f.f)
_has_infinite_level(f::Union{SeparableSum, Sum}) = any(_has_infinite_level, f.fs)
# the weights are tested by value, with no arithmetic on them
_weights_positive(::Nothing) = true
_weights_positive(w) = all(x -> x > 0, w)
# The wrappers are classified first, so that every factor of the tree is validated (`_hard(::Scaled)`) before
# anything is accepted: a scaled or translated indicator is a wrapper and is never taken as a leaf. An
# indicator is accepted as a leaf only when it is one of the library's plain-data indicators, listed here by
# name (their fields are numbers, arrays of numbers, flags, index lists and internal buffers, none of which
# can hold a function); every other indicator (constrained, composed or intersected ones, spectral and
# factorized ones, user types, any other subtype) is declined as a whole. The classification is by dispatch
# on this list, with no reflection on the fields of a type, so it is total.
const _PlainIndicator = Union{IndBox, IndNonnegative, IndBinary, IndBallL0, IndBallL1, IndBallL2, IndSphereL2,
                              IndSimplex, IndHyperplane, IndHalfspace, IndHyperslab, IndSOC, IndRotatedSOC,
                              IndExpPrimal, IndExpDual, IndPattern, IndBanded, IndToeplitz, IndIsotonic,
                              IndBallGroupL0}
_hard_function(f::Union{Scaled, Translate, SeparableSum, Sum}) = _hard(f)
_hard_function(f::_PlainIndicator) = f
_hard_function(f) = _hard(f)
_hard(_) = nothing
_hard(f::ZeroFunction) = f
# one level, infinite
for (F, level) in ((:NormL1, :λ0), (:NormL1Nonneg, :λ0), (:NormL2, :λ), (:NormLinf, :λ), (:GroupNormL2, :λ0),
                   (:GroupNormLinf, :λ0), (:NuclearNorm, :λ), (:TracePSD, :λ))
  @eval _hard(f::$F) = isinf(f.$level) ? f : nothing
end
# an infinite first level fixes every entry (the domain is {0}); the other levels are zeros, as a block, which
# keeps the levels of a SLOPE nonincreasing
_hard(f::SLOPE) = any(isinf, f.λ) ? SLOPE(map(λ -> isinf(λ) ? λ : zero(float(λ)), f.λ)) : nothing
_hard(f::NormL1Fused) = isinf(f.λ1) ? NormL1Fused(f.λ1, isinf(f.λ2) ? f.λ2 : zero(f.λ2)) : nothing
# two levels: the finite level is irrelevant when the infinite level fixes every coordinate that it
# touches (a weight of zero leaves its coordinate free), or when it is zero
function _hard(f::ElasticNet)
  (isinf(f.λ1) || isinf(f.λ2)) || return nothing
  zeroed = (isinf(f.λ1) || iszero(f.λ1)) && (isinf(f.λ2) || iszero(f.λ2))
  (zeroed || isinf(f.λ2) || _weights_positive(f.w)) || return nothing
  ElasticNet(isinf(f.λ1) ? f.λ1 : zero(f.λ1), isinf(f.λ2) ? f.λ2 : zero(f.λ2), f.w)
end
# an infinite ℓ1 level fixes every grouped entry at every step, whatever the group level and weights are
function _hard(f::SparseGroupNormL2)
  isinf(f.λ1) && return f
  isinf(f.λ0) && (iszero(f.λ1) || _weights_positive(f.w)) || return nothing
  SparseGroupNormL2(zero(f.λ1), f.λ0, f.groups, f.w)
end
# a factor is irrelevant when it is a positive finite number; any other factor keeps the earlier route
function _hard(f::Scaled)
  c = f.α
  (c isa Real && isfinite(c) && c > 0) ? _hard_function(f.f) : nothing
end
function _hard(t::Translate)
  g = _hard_function(t.f)
  g === nothing ? nothing : Translate(g, t.b, t.scratch, t.proxscratch)
end
function _hard(s::SeparableSum)
  gs = map(_hard_function, s.fs)
  any(isnothing, gs) ? nothing : SeparableSum(gs)
end
function _hard(s::Sum)
  gs = map(_hard_function, s.fs)
  any(isnothing, gs) ? nothing : Sum(gs, s.scratch)
end
# a chain anywhere among the blocks may refuse the value of the conjugate
_chained_scaled(t::Union{Translate, SingularValue}) = _chained_scaled(t.f)
_chained_scaled(s::Union{SeparableSum, Sum}) = any(_chained_scaled, s.fs)

issmooth(::Type{<:Sum{F}}) where {F} = all(issmooth, fieldtypes(F))
isconvex(::Type{<:Sum{F}}) where {F} = all(isconvex, fieldtypes(F))
_checkparams(s::Sum, x) = (map(f -> _checkparams(f, x), s.fs); true)

(s::Sum)(x) =
  (_check_value_arrays(s, x); _value_checked(s, x))
_value_unchecked(s::Sum, x) = _value_at(sum(f -> _value_unchecked(f, x), s.fs), x)
_checkvalue(s::Sum, x) = (map(f -> _checkvalue(f, x), s.fs); true)

# The buffer through which the remaining functions' gradients and Hessian
# products are added into the output: an array of the output's type and size, or
# the tuple of such arrays, kept while the shapes and types agree. The check
# versions leave the cache alone.
_sum_buffer!(s::Scratch, g::_ArrayOrOwner, x) = _array_buffer!(s, eltype(g), size(g))
_sum_buffer!(s::Scratch, g::Tuple, x::Tuple) = _tuple_buffer!(s, g, x)
_sum_checkbuffer(s::Scratch, g::_ArrayOrOwner, x) = _checkbuffer(s, eltype(g), size(g))
_sum_checkbuffer(s::Scratch, g::Tuple, x::Tuple) = _conjugate_checkbuffer(s, g, x)

# Adding the functions' results writes every entry of the output, which Symmetric and
# Hermitian wrappers refuse; found before the first function writes.
_check_sum_output(g) = true
_check_sum_output(g::Union{Symmetric, Hermitian}) = throw(ArgumentError(
  "a sum of functions needs an output that accepts off-diagonal writes; use a dense matrix"))
@generated _check_sum_output(g::Tuple) = _blockwise(k -> :(_check_sum_output(g[$k])), fieldcount(g), true)

# g += b, block by block for tuples
_accumulate!(g, b) = (g .+= b)
@generated _accumulate!(g::Tuple, b::Tuple) =
  _blockwise(k -> :(_accumulate!(g[$k], b[$k])), min(fieldcount(g), fieldcount(b)), :g)

function _checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, s::Sum, g::Union{_ArrayOrOwner, Tuple}, x::Union{_ArrayOrOwner, Tuple})
  _checkreal(s, g); _checkreal(s, x)
  _checkgradient_output(g, x)
  _checksize(g, x)
  length(s.fs) > 1 && _check_sum_output(g)
  _checkparams(s, x)
  _checkwrite(value_and_gradient!, first(s.fs), g, x)
  if length(s.fs) > 1
    b = _sum_checkbuffer(s.scratch, g, x)
    map(f -> _checkwrite(value_and_gradient!, f, b, x), Base.tail(s.fs))
  end
  true
end

# the remaining functions' gradients added into g through the buffer b
_add_gradients!(g, b, x, v, ::Tuple{}) = v
function _add_gradients!(g, b, x, v, fs::Tuple)
  w = _value_and_gradient_unchecked!(b, first(fs), x)
  _accumulate!(g, b)
  _add_gradients!(g, b, x, v + w, Base.tail(fs))
end

value_and_gradient!(g::Union{_ArrayOrOwner, Tuple}, s::Sum, x::Union{_ArrayOrOwner, Tuple}) =
  (_check_oracle_arrays(value_and_gradient!, s, g, x); _value_and_gradient_unchecked!(g, s, x))
function _value_and_gradient_unchecked!(g::Union{_ArrayOrOwner, Tuple}, s::Sum, x::Union{_ArrayOrOwner, Tuple})
  _checkwrite(value_and_gradient!, s, g, x)
  v = _value_and_gradient_unchecked!(g, first(s.fs), x)
  length(s.fs) == 1 && return _value_at(v, x)
  b = _sum_buffer!(s.scratch, g, x)
  _value_at(_add_gradients!(g, b, x, v, Base.tail(s.fs)), x)
end

####################################
#  Availability of standalone values
####################################

_has_value(::AbstractFunction) = true
_has_value(c::Conjugate) = _has_conjugate_value(c.f)
_has_value(f::Union{Scaled, Tilt, Translate, SingularValue}) = _has_value(f.f)
_has_value(s::Union{SeparableSum, Sum, SliceSum}) = all(_has_value, s.fs)
_has_value(fb::FiberSum) = _has_value(fb.f) && (fb.diagonal === nothing || _has_value(fb.diagonal))

####################################
#  Lipschitz constants of the gradients
####################################

lipschitz(s::Scaled) = _chained(s) ? _apply_scale(_scaled_chain(s)[2], lipschitz(_scaled_chain(s)[1])) : abs(s.α) * lipschitz(s.f)
lipschitz(t::Tilt) = lipschitz(t.f)
lipschitz(t::Translate) = lipschitz(t.f)
lipschitz(s::Sum) = sum(lipschitz, s.fs)
lipschitz(s::SeparableSum) = maximum(lipschitz, s.fs)

function _checkstep(op::O, s::Scaled, y, x, γ::Real) where {O}
  if _nested_route(s, γ)
    g, step, _ = _nested_step(s, γ)
    return _checkstep(op, g, y, x, step)
  end
  _checkstep(op, s.f, y, x, _scaled_step(s, γ))
end
_checkstep(op::O, t::Union{Tilt, Translate}, y::AbstractArray, x::AbstractArray, γ::Real) where {O} =
  _checkstep(op, t.f, y, _StepInput(y), γ)
_checkstep(op::O, t::Union{Tilt, Translate}, y::Tuple, x::Tuple, γ::Real) where {O} =
  _checkstep(op, t.f, y, _step_input(y), γ)

_checkstep(op::O, s::SeparableSum, y::Tuple, x::Tuple, γ::Real) where {O} =
  _checkstep_blocks(op, s.fs, y, x, γ)
@generated _checkstep_blocks(op, fs::Tuple, y::Tuple, x::Tuple, γ) =
  _blockwise(k -> :(_checkstep(op, fs[$k], y[$k], x[$k], γ)), fieldcount(fs), true)

_checkstep(op::O, c::Conjugate{<:Conjugate}, y::AbstractArray, x::AbstractArray, γ::Real) where {O} =
  _checkstep(op, c.f.f, y, x, γ)
_checkstep(op::O, c::Conjugate{<:Conjugate}, y::Tuple, x::Tuple, γ::Real) where {O} =
  _checkstep(op, c.f.f, y, x, γ)
function _checkstep(op::O, c::Conjugate, y::AbstractArray, x::AbstractArray, γ::Real) where {O}
  _direct_projection(c.f, x) && !_projection_fallback(c.f, y, x) && return true
  T = float(eltype(y))
  step = convert(T, γ)
  p = _conjugate_checkbuffer(c.scratch, y, x)
  if op === value_and_prox! && !_isindicator(c) && !_has_conjugate_value(c.f)
    return _checkstep(value_and_prox!, c.f, p, _StepInput(p), _moreau_reciprocal(c.f, p, x, γ))
  end
  _moreau_free(typeof(c.f)) && return _checkstep_moreau(c.f, p, x, step)
  _checkstep(prox!, c.f, p, _StepInput(p), _moreau_reciprocal(c.f, p, x, γ))
end
_checkstep(op::O, c::Conjugate{<:SumLargest}, y::AbstractArray, x::AbstractArray, γ::Real) where {O} =
  _checkstep(prox!, c.f, y, x, 1)
function _checkstep_moreau(f, y, x, γ)
  _conjugate_isindicator(typeof(f)) && return _checkstep(prox!, f, y, x, one(γ))
  _checkstep_moreau_set(f, y, x, γ)
end
_checkstep_moreau_set(f::IndBox, y, x, γ) = true
# The set of the Moreau point is the ball of radius γ r, which a negative or NaN radius rejects.
_checkstep_moreau_set(f::IndBallL2, y, x, γ) = (IndBallL2(_dilate(γ, f.r)); true)
_checkstep_moreau_set(s::Scaled, y, x, γ) = _checkstep_moreau(s.f, y, x, γ)
_checkstep_moreau_set(t::Translate, y, x, γ) = _checkstep_moreau(t.f, y, _StepInput(y), γ)

_step_input(x::AbstractArray) = _StepInput(x)
@generated _step_input(x::Tuple) = _blocktuple(k -> :(_step_input(x[$k])), fieldcount(x))

# The step pass needs only the shape and element type of the dense buffer.
struct _StepBuffer{T, N} <: AbstractArray{T, N}
  dims::NTuple{N, Int}
end
Base.size(x::_StepBuffer) = x.dims
_step_values(::_StepBuffer) = false
_step_buffer(x::AbstractArray{T, N}) where {T, N} = _StepBuffer{float(T), N}(size(x))
@generated _step_buffer(x::Tuple) = _blocktuple(k -> :(_step_buffer(x[$k])), fieldcount(x))
function _checkstep(op::O, c::Conjugate, y::Tuple, x::Tuple, γ::Real) where {O}
  p = _step_buffer(y)
  if op === value_and_prox! && !_isindicator(c) && !_has_conjugate_value(c.f)
    return _checkstep(value_and_prox!, c.f, p, _step_input(p), _moreau_reciprocal(c.f, p, x, γ))
  end
  _conjugate_isindicator(typeof(c.f)) && return _checkstep(prox!, c.f, p, x, one(float(typeof(γ))))
  _checkstep(prox!, c.f, p, _step_input(p), _moreau_reciprocal(c.f, p, x, γ))
end

# Keep L1 coefficients separate from the legacy product used by other blocks.
struct _L1ProjectionFactors{F}
  factors::F
end
Base.iszero(a::_L1ProjectionFactors) = any(iszero, a.factors)
Base.isinf(a::_L1ProjectionFactors) = any(isinf, a.factors)
_projection_l1_leaf(::Type) = false
_projection_l1_leaf(::Type{<:Union{NormLinf,GroupNormLinf}}) = true
_projection_l1_leaf(::Type{<:Union{Scaled{F},Translate{F}}}) where {F} = _projection_l1_leaf(F)
_projection_l1_family(::Type) = false
_projection_l1_family(::Type{<:Union{NormLinf,GroupNormLinf}}) = true
_projection_l1_family(::Type{<:Union{Scaled{F},Translate{F}}}) where {F} = _projection_l1_family(F)
@generated _projection_l1_family(::Type{SeparableSum{F}}) where {F} = _blockany(k -> :(_projection_l1_family(fieldtype(F, $k))), fieldcount(F))
_projection_l1_host(x) = _projection_host(x)
@generated _projection_l1_host(x::Tuple) = _blockany_eager(k -> :(_projection_l1_host(x[$k])), fieldcount(x))
_projection_l1_factors(a) = (a,)
_projection_l1_factors(a::_L1ProjectionFactors) = a.factors
_projection_l1_plain(a) = a
_projection_l1_plain(a::_L1ProjectionFactors) = foldl(_projection_accumulate, a.factors; init = 1)
_projection_l1_chain(f) = ()
_projection_l1_chain(f::Scaled) = (f.α, _projection_l1_chain(f.f)...)
_projection_l1_chain(f::Translate) = _projection_l1_chain(f.f)
@inline function _projection_wrapper_scale(f, a, b, x)
  if _projection_l1_family(typeof(f)) && _projection_l1_host(x)
    return _L1ProjectionFactors((a, b, _projection_l1_chain(f)...))
  end
  _projection_accumulate(a, b)
end
# Collect once per linear wrapper chain; keep its type fixed during recursion.
_projection_wrapper_scale(f, a::_L1ProjectionFactors, b, x) = a
@inline function _projection_block_scale(f, a, x)
  if _projection_l1_family(typeof(f)) && _projection_l1_host(x)
    return _L1ProjectionFactors((_projection_l1_factors(a)..., _projection_l1_chain(f)...))
  end
  _projection_l1_plain(a)
end

_projection_l1_check_input(f, x) = true
function _projection_l1_check_input(::Union{NormLinf,GroupNormLinf}, x::AbstractArray)
  if _projection_host(x) &&
      !(isconcretetype(eltype(x)) && eltype(x) <: Union{AbstractFloat,Base.BitInteger})
    throw(ArgumentError("L1 conjugate projections require floating or machine-integer input arrays; convert other real inputs explicitly"))
  end
  true
end
_projection_l1_check_input(f::Union{Scaled,Translate}, x) = _projection_l1_check_input(f.f, x)
_projection_l1_check_input(f::SeparableSum, x::Tuple) = _projection_l1_check_blocks(f.fs, x)
@generated _projection_l1_check_blocks(fs::Tuple, x::Tuple) =
  _blockwise(k -> :(_projection_l1_check_input(fs[$k], x[$k])), min(fieldcount(fs), fieldcount(x)), true)
_projection_l1_check_input(f, y, x) = _projection_fallback(f, y, x) || _projection_l1_check_input(f, x)
_projection_l1_check_input(f::Union{Scaled,Translate}, y, x) = _projection_l1_check_input(f.f, y, x)
_projection_l1_check_input(f::SeparableSum, y::Tuple, x::Tuple) = _projection_l1_check_blocks(f.fs, y, x)
@generated _projection_l1_check_blocks(fs::Tuple, y::Tuple, x::Tuple) =
  _blockwise(k -> :(_projection_l1_check_input(fs[$k], y[$k], x[$k])), min(fieldcount(fs), fieldcount(y), fieldcount(x)), true)

# The shift may be wider than the output even when the projection is bounded.
struct _FiniteProjectionShift{T} end
(::_FiniteProjectionShift{T})(v) where {T} = isfinite(convert(T, v))
_projection_step_type(γ::Real) = float(typeof(γ))
_projection_step_type(γ::AbstractArray) = float(eltype(γ))
_projection_shift(γ, b) = _dilate(promote(float(γ), float(b))...)
function _projection_shifted!(f::Translate, z, x, γ)
  T = promote_type(float(eltype(x)), float(eltype(z)), float(eltype(f.b)), _projection_step_type(γ))
  p = _array_buffer!(f.scratch, T, x)
  p .= _projection_shift.(_reshapeparam(γ, x), _reshapeparam(f.b, x))
  mapreduce(identity, &, Base.Broadcast.instantiate(Base.Broadcast.broadcasted(_FiniteProjectionShift{float(eltype(z))}(), p)); init = true) || _moreau_error(γ)
  p .= x .- p
  bad = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(_projection_nonfinite, x, p))
  mapreduce(identity, |, bad; init = false) && _moreau_error(γ)
  p
end

# Pure projections on equal-precision dense arrays cannot fail a finite-result
# check. Keep scratch for translations, mixed precision and aliased parameters.
_projection_writes_safe(::Type) = false
_projection_writes_safe(::Type{<:Union{NormL1, NormL1Nonneg, CheckLoss, NormL2, GroupNormL2, NormLinf, GroupNormLinf,
                                      ZeroFunction, IndNonnegative}}) = true
_projection_writes_safe(::Type{<:Scaled{F}}) where {F} = _projection_writes_safe(F)
_projection_parameter_alias(y, ::Union{Nothing, Real}) = false
_projection_parameter_alias(y, p::AbstractArray) = Base.mightalias(y, p)
_projection_parameters_alias(f, y) = false
_projection_parameters_alias(f::Union{NormL1, NormL1Nonneg, CheckLoss, GroupNormL2, GroupNormLinf}, y) =
  _projection_parameter_alias(y, f.w)
_projection_parameters_alias(f::IndBox, y) =
  _projection_parameter_alias(y, f.l) || _projection_parameter_alias(y, f.u)
_projection_parameters_alias(f::Scaled, y) = _projection_parameters_alias(f.f, y)
_projection_direct_output(f, y, x) = false
_projection_direct_output(f, y::Array{T}, x::Array{T}) where {T<:AbstractFloat} =
  _projection_writes_safe(typeof(f)) && (y === x || !Base.mightalias(y, x)) &&
    !_projection_parameters_alias(f, y)

# The proximal point itself, separate from the point subtracted in Moreau's identity.
_direct_conjugate(::Type) = false
_direct_conjugate(::Type{<:Union{NormL1, NormL1Nonneg, CheckLoss, ZeroFunction,
                                IndNonnegative, NormL2, GroupNormL2, NormLinf, GroupNormLinf}}) = true
_direct_conjugate(::Type{<:Union{Scaled{F}, Translate{F}}}) where {F} = _direct_conjugate(F)

# Unknown storage and unsupported axes retain Moreau, including on devices.
_projection_host(::Array) = true
_projection_host(x::Union{BitArray, UnitRange, StepRange, StepRangeLen, LinRange, Base.OneTo,
                          SparseVector, SparseMatrixCSC, SparseIterate, SymmetricSparseIterate, AtomIterate}) =
  !Base.has_offset_axes(x)
@static if isdefined(Base, :Memory)
  _projection_host(::Base.Memory) = true
end
@inline _projection_host(x::AbstractArray) = !Base.has_offset_axes(x) &&
  !_is_device_array(x) && parent(x) !== x && _projection_host(parent(x))
_projection_host(x::Diagonal) = !Base.has_offset_axes(x) && _projection_host(x.diag)
_projection_host(x::Union{Bidiagonal, SymTridiagonal}) = !Base.has_offset_axes(x) && _projection_host(x.dv) && _projection_host(x.ev)
_projection_host(x::Tridiagonal) = !Base.has_offset_axes(x) && _projection_host(x.dl) && _projection_host(x.d) && _projection_host(x.du)
_projection_fallback(f, y, x) = false
_projection_fallback(::Union{NormLinf, GroupNormLinf}, y::AbstractArray, x::AbstractArray) =
  !(_projection_host(y) && _projection_host(x))
_projection_fallback(f::Union{Scaled, Translate}, y, x) = _projection_fallback(f.f, y, x)
_projection_fallback(f::SeparableSum, y::Tuple, x::Tuple) = _projection_fallback_blocks(f.fs, y, x)
@generated _projection_fallback_blocks(fs::Tuple, y::Tuple, x::Tuple) =
  _blockany_eager(k -> :(_projection_fallback(fs[$k], y[$k], x[$k])), min(fieldcount(fs), fieldcount(y), fieldcount(x)))
_direct_projection(f, x) = _direct_conjugate(typeof(f))
_direct_projection(::Union{NormLinf, GroupNormLinf}, x) = _projection_host(x)
_direct_projection(f::Union{Scaled, Translate}, x) = _direct_projection(f.f, x)
_direct_projection(s::SeparableSum, x::Tuple) = _direct_projection_blocks(s.fs, x)
@generated _direct_projection_blocks(fs::Tuple, x::Tuple) =
  _blockall_eager(k -> :(_direct_projection(fs[$k], x[$k])), min(fieldcount(fs), fieldcount(x)))
_any_direct_projection(f, x) = _direct_projection(f, x)
_any_direct_projection(f::Scaled, x) = _any_direct_projection(f.f, x)
_any_direct_projection(s::SeparableSum, x::Tuple) = _any_direct_projection_blocks(s.fs, x)
@generated _any_direct_projection_blocks(fs::Tuple, x::Tuple) =
  _blockany_eager(k -> :(_any_direct_projection(fs[$k], x[$k])), min(fieldcount(fs), fieldcount(x)))

_conjugate_projection!(z, f, x, γ, α = 1) = _conjugate_moreau!(z, f, x, γ, α)
# Keep the original function concrete; merging it with Scaled boxes parameters.
_conjugate_moreau!(z, f, x, γ, α) = isone(α) ?
  _conjugate_moreau!(z, f, x, γ) : _conjugate_moreau_by!(z, f, x, γ, α)
# The wrappers of an accumulated factor that is not a normal number have no fixed type: the
# function barrier keeps the result of the call concrete.
_conjugate_moreau_by!(z, f, x, γ, α::Real) = _conjugate_moreau!(z, Scaled(f, α), x, γ)
function _conjugate_moreau_by!(z, f, x, γ, α::_ProjectionScale)
  if _projection_two_pieces(α)
    _conjugate_moreau!(z, _projection_scaled2(f, α), x, γ)
  else
    _conjugate_moreau!(z, _projection_scaled(f, α), x, γ)
  end
  z
end
function _conjugate_moreau!(z::AbstractArray, f, x::AbstractArray, γ)
  T = float(eltype(z))
  step = convert(T, γ)
  if _moreau_free(typeof(f))
    _moreau_point!(z, f, x, step)
    _checkmoreau(z, x, z, one(T), γ, _moreau_checked(typeof(f)))
    z .= x .- z
  else
    z .= x ./ step
    _prox_unchecked!(z, f, z, _moreau_reciprocal(f, z, x, γ))
    _checkmoreau(z, x, z, step, γ, true)
    z .= x .- step .* z
  end
  z
end
_conjugate_moreau!(z::Tuple, f, x::Tuple, γ) = _conjugate_moreau_block!(z, f, x, γ)
# Keep a block's own scale inside the sum's scaled reciprocal step.
@inline _conjugate_block_scaled(f, α) = @inline Scaled{typeof(f), typeof(α)}(f, α)
_conjugate_moreau_block!(z, f, x, γ, α) = isone(α) ?
  _conjugate_moreau_block!(z, f, x, γ) : _conjugate_moreau_block_by!(z, f, x, γ, α)
_conjugate_moreau_block_by!(z, f, x, γ, α::Real) = _conjugate_moreau_block!(z, _conjugate_block_scaled(f, α), x, γ)
function _conjugate_moreau_block_by!(z, f, x, γ, α::_ProjectionScale)
  if _projection_two_pieces(α)
    _conjugate_moreau_block!(z, _projection_scaled2(f, α), x, γ)
  else
    _conjugate_moreau_block!(z, _projection_scaled(f, α), x, γ)
  end
  z
end
# Tuple blocks narrow their arithmetic, but retain the original reciprocal step.
function _conjugate_moreau_block!(z::Tuple, f, x::Tuple, γ)
  if _conjugate_isindicator(typeof(f))
    unit = one(float(typeof(γ)))
    _prox_unchecked!(z, f, x, unit)
    _checkmoreau(z, x, z, unit, γ, false)
    _moreau_output!(z, x, z, unit)
  else
    _divide!(z, x, γ)
    _prox_unchecked!(z, f, z, _moreau_reciprocal(f, z, x, γ))
    _checkmoreau(z, x, z, γ, γ, true)
    _moreau_output!(z, x, z, γ)
  end
  z
end

@inline function _conjugate_moreau_block!(z::AbstractArray, f, x::AbstractArray, γ)
  step = convert(float(eltype(z)), γ)
  if _conjugate_isindicator(typeof(f))
    unit = one(float(typeof(γ)))
    _prox_unchecked!(z, f, x, unit)
    _checkmoreau(z, x, z, unit, γ, false)
    z .= x .- z
  else
    z .= x ./ step
    _prox_unchecked!(z, f, z, _moreau_reciprocal(f, z, x, γ))
    _checkmoreau(z, x, z, step, γ, true)
    z .= x .- step .* z
  end
  z
end

# Keep all factors until the coefficient is formed; a free coordinate stays free
# even at zero level. This differs from the group zero-level convention.
_projection_coefficient(α, λ, w) = isinf(w) ? oftype(float(w), Inf) : _projection_product(α, λ, w)
_projection_clip(v, α, λ, w) = (c = _projection_coefficient(α, λ, w); clamp(v, -c, c))
_projection_upper(v, α, λ, w) = min(v, _projection_coefficient(α, λ, w))
_projection_check(v, α, τ, w) = clamp(v, -_projection_coefficient(α, 1 - τ, w), _projection_coefficient(α, τ, w))

@inline _projection_bound(v, c, ::Val{false}) = clamp(v, -c, c)
@inline _projection_bound(v, c, ::Val{true}) = min(v, c)
function _projection_clipping!(z, x, c::Real, w, upper)
  T = promote_type(float(eltype(x)), float(eltype(z)))
  if w === nothing
    bound = convert(T, c)
    @inbounds @simd for i in eachindex(IndexLinear(), z, x)
      z[i] = _projection_bound(x[i], bound, upper)
    end
  elseif !isconcretetype(eltype(w))
    @inbounds for i in eachindex(IndexLinear(), z, x)
      bound = convert(T, _projection_coefficient(c, 1, w[i]))
      z[i] = _projection_bound(x[i], bound, upper)
    end
  else
    P = promote_type(float(typeof(c)), float(eltype(w)))
    level = convert(P, c)
    if level > 0 && isfinite(level)
      # Promote parameters before multiplying, then narrow the completed bound.
      @inbounds @simd for i in eachindex(IndexLinear(), z, x)
        z[i] = _projection_bound(x[i], convert(T, level * convert(P, w[i])), upper)
      end
    else
      @inbounds @simd for i in eachindex(IndexLinear(), z, x)
        z[i] = _projection_bound(x[i], convert(T, _weight_coefficient(level, convert(P, w[i]))), upper)
      end
    end
  end
  z
end
function _projection_clipping!(z, x, c, w, upper)
  T = promote_type(float(eltype(x)), float(eltype(z)))
  if w === nothing
    bound = convert(T, _projection_coefficient(c, 1, 1))
    @inbounds @simd for i in eachindex(IndexLinear(), z, x)
      z[i] = _projection_bound(x[i], bound, upper)
    end
  else
    @inbounds @simd for i in eachindex(IndexLinear(), z, x)
      bound = convert(T, _projection_coefficient(c, 1, w[i]))
      z[i] = _projection_bound(x[i], bound, upper)
    end
  end
  z
end

for (F, op) in ((:NormL1, :_projection_clip), (:NormL1Nonneg, :_projection_upper), (:CheckLoss, :_projection_check))
  level = F === :CheckLoss ? :τ : :λ0
  @eval begin
    function _conjugate_projection!(z, f::$F, x, γ, α = 1)
      if f isa Union{NormL1, NormL1Nonneg}
        c = _projection_accumulate(α, f.$level)
        return _projection_clipping!(z, x, c, f.w, Val(f isa NormL1Nonneg))
      end
      @inbounds @simd for i in eachindex(IndexLinear(), z, x)
        z[i] = $op(x[i], α, f.$level, _entry(f.w, i))
      end
      z
    end
    function _conjugate_entry(f::$F, v, i, γ)
      T = float(promote_type(typeof(v), typeof(γ)))
      convert(T, $op(v, 1, f.$level, _entry(f.w, i)))
    end
  end
end
_conjugate_projection!(z, ::ZeroFunction, x, γ, α = 1) = fill!(z, zero(eltype(z)))
_conjugate_projection!(z, ::IndNonnegative, x, γ, α = 1) = (z .= min.(x, zero(eltype(z))))
_conjugate_entry(::ZeroFunction, v, i, γ) = zero(float(v))
_conjugate_entry(::IndNonnegative, v, i, γ) = min(v, zero(v))

_conjugate_projection!(z, f::Scaled, x, γ, α = 1) =
  _direct_projection(f.f, x) || f.f isa SeparableSum ? _conjugate_projection!(z, f.f, x, γ, _projection_wrapper_scale(f.f, α, f.α, x)) : _conjugate_moreau!(z, f, x, γ, α)
function _conjugate_projection!(z, f::Translate, x, γ, α = 1)
  _direct_projection(f.f, x) || return _conjugate_moreau!(z, f, x, γ, α)
  p = _projection_shifted!(f, z, x, γ)
  _conjugate_projection!(z, f.f, p, γ, α)
end
# the conjugate of a translation is available for array translations only
@noinline _throw_tuple_translation() = throw(ArgumentError(
  "Conjugate of a Translate with a tuple translation is not supported; its translation must be an array"))
_conjugate_projection!(z, f::Translate{<:AbstractFunction, <:Tuple}, x, γ, α = 1) = _throw_tuple_translation()
_conjugate_projection_value!(z, f::Translate{<:AbstractFunction, <:Tuple}, x, γ, α = 1) = _throw_tuple_translation()
_moreau_set!(z, t::Translate{<:AbstractFunction, <:Tuple}, x, γ) = _throw_tuple_translation()
_checkscaledconjugate(f::Translate{<:AbstractFunction, <:Tuple}, y, α) = _throw_tuple_translation()
_conjugate_entry(f::Scaled, v, i, γ) =
  convert(float(promote_type(typeof(v), typeof(γ))), _scaled_conjugate_entry(f.f, v, i, γ, f.α))
_scaled_conjugate_entry(f::NormL1, v, i, γ, α) = _projection_clip(v, α, f.λ0, _entry(f.w, i))
_scaled_conjugate_entry(f::NormL1Nonneg, v, i, γ, α) = _projection_upper(v, α, f.λ0, _entry(f.w, i))
_scaled_conjugate_entry(f::CheckLoss, v, i, γ, α) = _projection_check(v, α, f.τ, _entry(f.w, i))
_scaled_conjugate_entry(f, v, i, γ, α) =
  v - γ * prox_entry(f, v / γ, i, _moreau_ratio(promote_type(float(typeof(v)), _data_of(f)), _projection_real(α), γ))
_conjugate_entry(f::Translate, v, i, γ) = _direct_conjugate(typeof(f.f)) ?
  _conjugate_entry(f.f, v - γ * f.b[i], i, γ) : v - γ * prox_entry(f, v / γ, i, _moreau_reciprocal(promote_type(float(typeof(v)), _data_of(f)), γ))

_projection_commit!(y::AbstractArray, z::AbstractArray) = copyto!(y, z)
_projection_check_commit(y, z) = true
function _projection_check_conversion(y::AbstractArray{T}, z::AbstractArray{S}) where {T,S}
  if T !== S && _projection_host(y) && _projection_host(z)
    @inbounds for v in z
      convert(T, v)
    end
  end
  true
end
# an owner: the entries of the staged matrix are converted to its type
function _projection_check_conversion(y::SymmetricSparseIterate{T}, z::AbstractArray{S}) where {T, S}
  if T !== S && _projection_host(z)
    @inbounds for v in z
      convert(T, v)
    end
  end
  true
end
_projection_check_commit(y::AbstractArray, z::AbstractArray) = _projection_check_conversion(y, z)
# An owner stores the lower triangle of a staged matrix, so the matrix must be exactly symmetric (every pair
# (i, j), i > j, `_symequal`: equal NaNs, infinities and signed zeros included): a pair that differs is refused
# before anything is written. `rounded` is the one exemption, for the wrappers named at `_rounding_symmetric_prox`.
_projection_check_commit(y::SymmetricSparseIterate, z::AbstractArray, rounded::Bool = false) =
  (_check_axes(y, z) && (rounded || _check_staged_symmetric(z)); _projection_check_conversion(y, z))
_check_staged_symmetric(z::AbstractMatrix) =
  _check_symmetric(z, "the staged result written into a SymmetricSparseIterate", "z")

# The functions whose proximal point at a symmetric point is symmetric by mathematics but is computed, in a
# dense matrix, with the two triangles differing by rounding: the nuclear norm is convex, so its proximal point
# is unique, and it is invariant under transposition. `Conjugate(f)`, `MoreauEnvelope(f)` and the scalar
# `Precompose(f, μ)` of such an f store the lower triangle of the dense result at an owner without the pair
# test. No other function has it, and nothing is derived through other wrappers: `Scaled(NuclearNorm)` or
# `Conjugate(NuclearNorm)` as the inner function, `IndBallRank`, `IndPSD` and the rest take the exact check.
_rounding_symmetric_prox(f) = false
_rounding_symmetric_prox(::NuclearNorm) = true

# a dense result of the inner function f of a wrapper, checked before it is blended into an owner
_check_staged_result(y, z, f = nothing) = true
_check_staged_result(y::Union{SymmetricSparseIterate, _OwnerMatrix}, z::AbstractArray, f = nothing) =
  _rounding_symmetric_prox(f) || _check_staged_symmetric(z)
@generated _projection_check_commit(y::Tuple, z::Tuple) =
  _blockwise(k -> :(_projection_check_commit(y[$k], z[$k])), min(fieldcount(y), fieldcount(z)), true)
_projection_copy!(y::AbstractArray, z::AbstractArray) = copyto!(y, z)
@generated _projection_copy!(y::Tuple, z::Tuple) =
  _blockwise(k -> :(_projection_copy!(y[$k], z[$k])), min(fieldcount(y), fieldcount(z)), :y)
_projection_commit!(y::Tuple, z::Tuple) = (_projection_check_commit(y, z); _projection_copy!(y, z))
@generated _direct_conjugate(::Type{SeparableSum{F}}) where {F} = _blockall(k -> :(_direct_conjugate(fieldtype(F, $k))), fieldcount(F))
function _conjugate_projection!(z::Tuple, s::SeparableSum, x::Tuple, γ, α = 1, output = z)
  _conjugate_projection_blocks!(z, s.fs, x, γ, α, output)
  z
end
function _conjugate_projection_block!(z, f, x, γ, α, output = z)
  a = _projection_block_scale(f, α, x)
  # Tuple targets unwrap their scales; array targets retain their wrappers.
  _projection_fallback(f, output, x) && return _conjugate_projection_target!(z, f, output, x, γ, x isa Tuple ? a : α)
  result = _any_direct_projection(f, x) || f isa SeparableSum ?
    _conjugate_projection!(z, f, x, γ, a) : _conjugate_moreau_block!(z, f, x, γ, a)
  if result isa _L1ProjectionFallback
    _projection_l1_moreau_blocks!(z, f, x, γ, α, result, Val(true), prox!)
  end
  z
end
@generated _conjugate_projection_blocks!(z::Tuple, fs::Tuple, x::Tuple, γ, α, output) =
  _blockwise(k -> :(_conjugate_projection_block!(z[$k], fs[$k], x[$k], γ, α, output[$k])), fieldcount(fs), nothing)

# Carry original targets only for unsupported tuple blocks; other blocks retain
# their direct projections and coefficient chains.
_conjugate_projection_target!(z, f, y, x, γ, α = 1) =
  _conjugate_moreau_block!(z, f, x, γ, _projection_l1_plain(α))
_conjugate_projection_target!(z::Tuple, f::SeparableSum, y::Tuple, x::Tuple, γ, α = 1) =
  _conjugate_projection!(z, f, x, γ, α, y)
_conjugate_projection_target!(z::Tuple, f::Scaled, y::Tuple, x::Tuple, γ, α = 1) =
  _conjugate_projection_target!(z, f.f, y, x, γ, _projection_wrapper_scale(f.f, α, f.α, x))

_conjugate_projection!(z, f::NormL2, x, γ, α = 1) = _project_l2!(z, x, f.λ, α)
function _conjugate_projection!(z, f::GroupNormL2, x, γ, α = 1)
  @inbounds for i in f.free
    z[i] = zero(eltype(z))
  end
  @inbounds @simd for i in (length(f.group_of) + 1):length(z)
    z[i] = zero(eltype(z))
  end
  @inbounds for k in eachindex(f.groups)
    G = f.groups[k]
    isempty(G) && continue
    _project_l2_group!(z, x, G, f.λ0, α, f.w[k])
  end
  z
end

# Only exceptional groups allocate a recovery request.
struct _L1ProjectionFallback
  groups::Vector{Int}
end
function _conjugate_projection!(z, f::NormLinf, x, γ, α = 1)
  (_projection_host(x) && _projection_host(z)) || return _conjugate_moreau!(z, f, x, γ, α)
  _project_l1!(z, x, eachindex(IndexLinear(), x), f.scratch, f.λ, α) !== nothing && return z
  _L1ProjectionFallback([1])
end
function _conjugate_projection!(z, f::GroupNormLinf, x, γ, α = 1)
  (_projection_host(x) && _projection_host(z)) || return _conjugate_moreau!(z, f, x, γ, α)
  @inbounds for i in f.free
    z[i] = zero(eltype(z))
  end
  @inbounds @simd for i in (length(f.group_of) + 1):length(z)
    z[i] = zero(eltype(z))
  end
  result = _projection_l1_unit_groups!(z, x, f, α, γ)
  result === true && return z
  result isa _L1ProjectionFallback && return result
  fallback = nothing
  @inbounds for k in eachindex(f.groups)
    G = f.groups[k]
    if _project_l1!(z, x, G, f.scratch, f.λ0, α, f.w[k]) === nothing
      fallback === nothing && (fallback = Int[])
      push!(fallback, k)
    end
  end
  fallback === nothing ? z : _L1ProjectionFallback(fallback)
end

_scaled_conjugate_entry(::ZeroFunction, v, i, γ, α) = zero(float(v))
_scaled_conjugate_entry(::IndNonnegative, v, i, γ, α) = min(v, zero(v))
_scaled_conjugate_entry(f::Translate, v, i, γ, α) = _scaled_conjugate_entry(f.f, v - γ * f.b[i], i, γ, α)
_scaled_conjugate_entry(f::Scaled, v, i, γ, α) = _scaled_conjugate_entry(f.f, v, i, γ, _projection_accumulate(α, f.α))

# A translated indicator has a linear value even when its family's standalone
# conjugate value is unavailable. Mixed sums must keep this blockwise as well.
_any_direct_conjugate(::Type{F}) where {F} = _direct_conjugate(F)
@generated _any_direct_conjugate(::Type{SeparableSum{F}}) where {F} = _blockany(k -> :(_any_direct_conjugate(fieldtype(F, $k))), fieldcount(F))
_any_direct_conjugate(::Type{<:Scaled{F}}) where {F} = _any_direct_conjugate(F)

function _conjugate_projection_value!(z, f, x, γ, α = 1)
  if _conjugate_isindicator(typeof(f))
    _conjugate_projection!(z, f, x, γ, α)
    return _zero_value(z)
  end
  _conjugate_moreau_value!(z, f, x, γ, α, Val(false))
end
# Separate scaled variants before computing a value, just as for the point.
_conjugate_moreau_value!(z, f, x, γ, α, context) = isone(α) ?
  _conjugate_moreau_value!(z, f, x, γ, context) : _moreau_value_scaled!(z, f, x, γ, α, context)
_conjugate_moreau_value!(z::Tuple, f, x::Tuple, γ, α, context::Val{true}) = isone(α) ?
  _conjugate_moreau_value!(z, f, x, γ, context) : _moreau_value_block!(z, f, x, γ, α, context)
# as for the point, a factor that is not a normal number gives wrappers without a fixed type
_moreau_value_scaled!(z, f, x, γ, α::Real, context) = _conjugate_moreau_value!(z, Scaled(f, α), x, γ, context)
function _moreau_value_scaled!(z, f, x, γ, α::_ProjectionScale, context)::_valtype(z)
  _projection_two_pieces(α) ? _conjugate_moreau_value!(z, _projection_scaled2(f, α), x, γ, context) :
    _conjugate_moreau_value!(z, _projection_scaled(f, α), x, γ, context)
end
_moreau_value_block!(z, f, x, γ, α::Real, context) = _conjugate_moreau_value!(z, _conjugate_block_scaled(f, α), x, γ, context)
function _moreau_value_block!(z, f, x, γ, α::_ProjectionScale, context)::_valtype(z)
  _projection_two_pieces(α) ? _conjugate_moreau_value!(z, _projection_scaled2(f, α), x, γ, context) :
    _conjugate_moreau_value!(z, _projection_scaled(f, α), x, γ, context)
end
# Known block values can keep the scale in the step without a temporary wrapper.
function _conjugate_moreau_value!(z::AbstractArray, f, x::AbstractArray, γ, α, context::Val{true})
  isone(α) && return _conjugate_moreau_value!(z, f, x, γ, context)
  α isa _ProjectionScale && return _moreau_value_scaled!(z, f, x, γ, α, context)
  a = _projection_real(α)
  if !iszero(a) && _has_conjugate_value(f)
    _divide!(z, x, γ)
    _prox_unchecked!(z, f, z, a * _moreau_reciprocal(f, z, x, γ))
    _checkmoreau(z, x, z, γ, γ, true)
    _moreau_output!(z, x, z, γ)
    return _scaled_conjugate_value(f, z, a)
  end
  _conjugate_moreau_value!(z, _conjugate_block_scaled(f, a), x, γ, context)
end
# Combine nested scales before dividing coordinates, retaining exceptional products.
function _scaled_conjugate_value(f::Scaled, y::AbstractArray, α)::_valtype(y)
  a = _projection_accumulate(α, f.α)
  # Split before recursion so Julia 1.10 does not box the scale or value.
  if a isa _ProjectionScale
    return _scaled_conjugate_value(f.f, y, a)
  end
  _scaled_conjugate_value(f.f, y, a)
end
_scaled_conjugate_value(f::Scaled, y::AbstractArray, α::_ProjectionScale) =
  _scaled_conjugate_value(f.f, y, _projection_accumulate(α, f.α))
_scaled_conjugate_value(f::Scaled, y::Tuple, α) =
  _scaled_conjugate_value(f.f, y, _projection_accumulate(α, f.α))
# Keep exponent parts across a sum until each block's coefficient is complete.
function _scaled_conjugate_value(s::SeparableSum, y::Tuple, α::_ProjectionScale)::_valtype(y)
  _checkparams(s, y)
  _sum_projection_scaled(s.fs, y, α)
end
@generated _sum_projection_scaled(fs::Tuple, y::Tuple, α) =
  _blocksum(k -> :(_scaled_conjugate_value(fs[$k], y[$k], α)), min(fieldcount(fs), fieldcount(y)))
# (αf)* is the conjugate of f with α folded into its levels, which keeps a factor that is not a
# normal number in range. A function without levels to fold takes the rounded factor.
function _scaled_conjugate_value(f, y::AbstractArray, α::_ProjectionScale)::_valtype(y)
  # an indicator and the zero function do not change under a positive scale
  g = _fold_block(f, α)
  g === nothing || return _conjugate_value(g, y)
  a = _projection_real(α)
  (isfinite(a) && !iszero(a)) || _nested_scale_error(f, a)
  _scaled_conjugate_value_rounded(f, y, α)
end
@noinline _nested_scale_error(f, a) = throw(ArgumentError(
  "the conjugate value of $(typeof(f)) under nested scales whose product $(iszero(a) ? "underflows" : "overflows") is not supported: $(typeof(f)) does not fold a scale into its level"))
_scaled_conjugate_value(f, y::AbstractArray, α) = _scaled_conjugate_value_rounded(f, y, α)
function _scaled_conjugate_value_rounded(f, y::AbstractArray, α)::_valtype(y)
  a = convert(float(eltype(y)), _projection_real(α))
  scaled = _ScaledArray(y, inv(a))
  _checkconjugate(f, scaled)
  a * _conjugate_value(f, scaled)
end
@inline _conjugate_moreau_context!(z, f, x, γ, ::Val{false}) = (_conjugate_moreau!(z, f, x, γ); z)
@inline _conjugate_moreau_context!(z, f, x, γ, ::Val{true}) = (_conjugate_moreau_block!(z, f, x, γ); z)
_conjugate_moreau_step(z, γ, ::Val{false}) = convert(float(eltype(z)), γ)
_conjugate_moreau_step(z, γ, ::Val{true}) = γ
_conjugate_moreau_value(f, y) = _conjugate_value(f, y)
_conjugate_moreau_value(f::Scaled, y::AbstractArray) = _scaled_conjugate_value(f.f, y, f.α)
_conjugate_moreau_value(f::Scaled, y::Tuple) = _scaled_conjugate_value(f.f, y, f.α)
function _conjugate_moreau_value!(z, f, x, γ, context)
  if _has_conjugate_value(f)
    _conjugate_moreau_context!(z, f, x, γ, context)
    return _conjugate_moreau_value(f, z)
  end
  step = _conjugate_moreau_step(z, γ, context)
  _divide!(z, x, step)
  fp = _value_and_prox_unchecked!(z, f, z, _moreau_reciprocal(f, z, x, γ))
  _checkmoreau(z, x, z, step, γ, true)
  v = _projection_fenchel_value(z, x, step) - fp
  _moreau_output!(z, x, z, step)
  v
end
_projection_fenchel(p, x, γ) = p * (x - γ * p)
function _projection_fenchel_value(z::AbstractArray, x::AbstractArray, γ)
  step = convert(float(eltype(z)), γ)
  mapreduce(identity, +, Base.Broadcast.instantiate(Base.Broadcast.broadcasted(_projection_fenchel, z, x, step)); init = _zero_value(z))
end
@generated _projection_fenchel_value(z::Tuple, x::Tuple, γ) =
  _blocksum(k -> :(_projection_fenchel_value(z[$k], x[$k], γ)), min(fieldcount(z), fieldcount(x)))
_projection_l1_groups(f::NormLinf, x) = (eachindex(IndexLinear(), x),)
_projection_l1_groups(f::GroupNormLinf, x) = f.groups
_projection_l1_groups(f::Union{Scaled,Translate}, x) = _projection_l1_groups(f.f, x)
_projection_l1_block(f::NormLinf, k, G) = f
_projection_l1_block(f::GroupNormLinf, k, G) = GroupNormLinf(f.λ0, [1:length(G)], [f.w[k]])
_projection_l1_block(f::Scaled, k, G) = _conjugate_block_scaled(_projection_l1_block(f.f, k, G), f.α)
_projection_l1_block(f::Translate, k, G) = Translate(_projection_l1_block(f.f, k, G), view(f.b, G))
# Keep the original wrappers and Moreau oracle. Only requested groups may
# be written; infinite-weight and zero-radius shortcuts remain untouched.
@noinline function _projection_l1_moreau_blocks!(z, f, x, γ, α, request, context, op)::_valtype(z)
  v = _zero_value(z)
  # Requests arrive in ascending group order.
  next = firstindex(request.groups)
  for (k, G) in enumerate(_projection_l1_groups(f, x))
    isempty(G) && continue
    block = _projection_l1_block(f, k, G)
    output = view(z, G)
    if next <= lastindex(request.groups) && k == request.groups[next]
      next += 1
      input = Base.mightalias(z, x) ? copy(view(x, G)) : view(x, G)
      if op === prox!
        if context isa Val{true}
          _conjugate_moreau_block!(output, block, input, γ, _projection_l1_plain(α))
        else
          _conjugate_moreau!(output, block, input, γ, _projection_l1_plain(α))
        end
      else
        v += _conjugate_moreau_value!(output, block, input, γ, _projection_l1_plain(α), context)
      end
    elseif op === value_and_prox!
      v += _projection_l1_value(block, output)
    end
  end
  v
end
_projection_l1_value(f::Union{NormLinf,GroupNormLinf}, z) = _zero_value(z)
_projection_l1_value(f::Scaled, z) = _projection_l1_value(f.f, z)
_projection_l1_value(f::Translate, z) = _projection_l1_value(f.f, z) + dot(f.b, z)
@inline function _conjugate_l1_value!(z, f, x, γ, α, context)
  result = _conjugate_projection!(z, f, x, γ, _projection_block_scale(f, α, x))
  result isa _L1ProjectionFallback && return _projection_l1_moreau_blocks!(z, f, x, γ, α, result, context, value_and_prox!)
  _projection_l1_value(f, z)
end
function _conjugate_projection_value!(z, f::Translate, x, γ, α = 1)
  _projection_l1_leaf(typeof(f)) && return _conjugate_l1_value!(z, f, x, γ, α, Val(false))
  p = _projection_shifted!(f, z, x, γ)
  v = _conjugate_projection_value!(z, f.f, p, γ, α)
  v + dot(f.b, z)
end
_conjugate_projection_value!(z, f::Scaled, x, γ, α = 1) = _projection_l1_leaf(typeof(f)) ?
  _conjugate_l1_value!(z, f, x, γ, α, Val(false)) :
  _conjugate_projection_value!(z, f.f, x, γ, _projection_wrapper_scale(f.f, α, f.α, x))
_conjugate_projection_value!(z::Tuple, f::SeparableSum, x::Tuple, γ, α = 1) =
  _conjugate_projection_values!(z, f.fs, x, γ, α)
function _conjugate_projection_value_block!(z, f, x, γ, α, output = z)
  a = _projection_block_scale(f, α, x)
  if _projection_fallback(f, output, x)
    scale = x isa Tuple ? a : α
    _conjugate_isindicator(typeof(f)) || return _conjugate_projection_target_value!(z, f, output, x, γ, scale)
    _conjugate_projection_target!(z, f, output, x, γ, scale)
    return _zero_value(z)
  end
  if _conjugate_isindicator(typeof(f))
    _conjugate_projection_block!(z, f, x, γ, α)
    return _zero_value(z)
  end
  if _projection_l1_leaf(typeof(f))
    return _conjugate_l1_value!(z, f, x, γ, α, Val(true))
  end
  _any_direct_projection(f, x) || f isa SeparableSum ?
    _conjugate_projection_value!(z, f, x, γ, a) : _conjugate_moreau_value!(z, f, x, γ, a, Val(true))
end
_conjugate_projection_values!(z::Tuple, fs::Tuple, x::Tuple, γ, α) = _conjugate_projection_values!(z, fs, x, γ, α, z)
# Preserve the recursive projection sum's right grouping and terminal zero.
@generated _conjugate_projection_values!(z::Tuple, fs::Tuple, x::Tuple, γ, α, output) =
  _blockreduce(k -> :(_conjugate_projection_value_block!(z[$k], fs[$k], x[$k], γ, α, output[$k])), fieldcount(fs),
               vs -> foldr((v, rest) -> :($v + $rest), vs; init = 0))

_conjugate_projection_target_value!(z, f, y, x, γ, α = 1) =
  _conjugate_moreau_value!(z, f, x, γ, _projection_l1_plain(α), Val(true))
_conjugate_projection_target_value!(z::Tuple, f::SeparableSum, y::Tuple, x::Tuple, γ, α = 1) =
  _conjugate_projection_values!(z, f.fs, x, γ, α, y)
_conjugate_projection_target_value!(z::Tuple, f::Scaled, y::Tuple, x::Tuple, γ, α = 1) =
  _conjugate_projection_target_value!(z, f.f, y, x, γ, _projection_wrapper_scale(f.f, α, f.α, x))

_projection_nonfinite(x, z) = isfinite(x) & !isfinite(z)
function _check_projection_result(z::AbstractArray, x::AbstractArray, γ)
  promote_type(eltype(x), eltype(z)) == eltype(z) && return true
  bad = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(_projection_nonfinite, x, z))
  mapreduce(identity, |, bad; init = false) && _moreau_error(γ)
  true
end
@generated _check_projection_result(z::Tuple, x::Tuple, γ) =
  _blockwise(k -> :(_check_projection_result(z[$k], x[$k], γ)), fieldcount(z), true)

# Spectrum decisions are private to the matrix wrappers.
_decided_on_spectrum(f::Scaled) = _decided_on_spectrum(f.f)
function _spectrum_value(f::Scaled, v::AbstractVector{T}, k::Int) where {T}
  # Keep ordinary value arithmetic when no domain relaxation is needed.
  if k == 0 && !_isindicator(f)
    ordinary = f(v)
    isfinite(ordinary) && return convert(T, ordinary)
  end
  # An overflowing spectrum: the scales reach the leaf, after the domain decisions.
  if k > 0 && !_isindicator(f)
    s = _spectrum_scaled(f, v, k, _unit_scale(T))
    s === nothing || return convert(T, s)
  end
  a = _spectrum_value(f.f, v, k)
  r = f.α * a
  isnan(r) && !isnan(a) && return convert(T, _scaled_nan_value(f.f, f.α, v, r))   # Inf⋅0, as for the value
  if (!isfinite(r) && f.α < 1) || (_scaled_small(r) && !iszero(a)) ||
     (_scaled_small(a) && f.α > 1)
    g = _fold_level(f.f, f.α)
    g === nothing || return convert(T, _spectrum_value(g, v, k))
  end
  convert(T, r)
end
_decided_on_spectrum(f::Conjugate) = _conjugate_spectrum_indicator(f.f)
_conjugate_spectrum_indicator(::AbstractFunction) = false
_conjugate_spectrum_indicator(::Union{IndNonnegative, ZeroFunction, NormL2, NormLinf, SumLargest}) = true
_conjugate_spectrum_indicator(::NormL1{<:Real, Nothing}) = true
function _spectrum_value(f::Conjugate, v::AbstractVector{T}, k::Int) where {T}
  _checkconjugate(f.f, v)
  _conjugate_spectrum_value(f.f, v, k)
end
_conjugate_spectrum_value(f::NormL1, v, k) = _spectrum_ball(v, f.λ0, k, Inf)
_conjugate_spectrum_value(f::NormL2, v, k) = _spectrum_ball(v, f.λ, k, 2)
_conjugate_spectrum_value(f::NormLinf, v, k) = _spectrum_ball(v, f.λ, k, 1)
function _conjugate_spectrum_value(::IndNonnegative, v::AbstractVector{T}, k) where {T}
  t = _tol(T) * max(ldexp(one(T), -k), _spectrum_scale(v))
  _ind(isempty(v) || maximum(v) <= t, T)
end
_conjugate_spectrum_value(::ZeroFunction, v, k) = _spectrum_ball(v, zero(eltype(v)), k, Inf)
function _conjugate_spectrum_value(f::SumLargest, v::AbstractVector{T}, k) where {T}
  iszero(_spectrum_value(IndBox(0, f.λ), v, k)) || return T(Inf)
  # Scale the level before multiplying by k to avoid an overflowing mass.
  S = float(promote_type(T, typeof(f.λ)))
  e = _comparison_exponent(_magnitude_exponent(S(_spectrum_scale(v))),
    _magnitude_exponent(S(f.λ)) - k)
  w = _power_factors(S, -e)
  s = _scaled_sum(v, w)
  mass = f.k * ldexp(S(f.λ), -k - e)
  _ind(abs(s - mass) <= _tol(T) * max(ldexp(one(S), -k - e), abs(s), mass), T)
end
