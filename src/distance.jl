####################################
#
#  Distances to sets
#
####################################

"""
    SqrDistL2(C, λ = 1)

The squared Euclidean distance to a nonempty closed set `C`, scaled as
`f(x) = (λ/2) dist(x, C)²`, for finite `λ ≥ 0`. `C` is a proximable indicator
function whose proximal operator is an exact projection in its argument space.
With the projection `p ∈ P_C(x)` returned by `C`, the value is
`(λ/2)‖x - p‖²`, and the proximal operator is
`prox_{γf}(x) = (x + γλp)/(1 + γλ)`, including for a nonconvex `C`.

Convex and smooth when `C` is convex, with a `λ`-Lipschitz gradient. For
`λ > 0` and convex `C`, this is [`MoreauEnvelope`](@ref)`(C, 1/λ)`.
For a nonconvex `C`, [`issmooth`](@ref) is false and [`lipschitz`](@ref)
throws. The gradient oracles are still available: they return `λ(x - p)`,
the gradient wherever the projection is unique, almost everywhere. At ties
this is the residual of the projection that `C` chooses, not a gradient.
These residuals are used by proximal distance algorithms.

Separable when `C` is, with scalar proximal operators, optimality tests and
coordinate curvatures for coordinate descent. Restriction is supported only
for separable `C`. At `λ = 0`, the value and gradient are zero and the proximal
operator is the identity, after structural checks, without a projection;
the Lipschitz constant is zero. Traits still follow `C`.

The oracles use one projection into a buffer that `f` owns; do not share `f`
between threads. The proximal output may be `x`; the gradient output must not
be `x`. The buffer follows the argument's precision for values and gradients,
and the output's for proximal operators. Only scalar proximal steps are
supported. `Spectral(SqrDistL2(...))` is not supported.

Results are approximate when the projection is, as with [`IndIntersection`](@ref),
partial spectral decompositions or a truncated [`IndAffine`](@ref) system.
Nonemptiness of a [`Constrained`](@ref) set is the caller's responsibility.
[`Spectral`](@ref) sets read one triangle of a symmetric matrix; their distance
here is within symmetric matrices, so supply a symmetric argument.
"""
struct SqrDistL2{F<:AbstractFunction, T<:Real} <: AbstractFunction
  f::F
  λ::T
  scratch::Scratch

  function SqrDistL2{F, T}(C::F, λ::T) where {F<:AbstractFunction, T<:Real}
    _isindicator(C) || throw(ArgumentError("SqrDistL2 requires an indicator function of a set"))
    isproximable(C) || throw(ArgumentError("SqrDistL2 requires a proximable set whose proximal operator is a projection"))
    _checknonneg("λ", λ)
    isfinite(λ) || throw(ArgumentError("λ must be finite, got $λ"))
    new{F, T}(C, λ, Scratch())
  end
end
SqrDistL2(C::AbstractFunction, λ::Real = 1) = SqrDistL2{typeof(C), typeof(λ)}(C, λ)

isproximable(::Type{<:SqrDistL2}) = true
issmooth(::Type{<:SqrDistL2{F}}) where {F} = isconvex(F)
isconvex(::Type{<:SqrDistL2{F}}) where {F} = isconvex(F)
isseparable(::Type{<:SqrDistL2{F}}) where {F} = isseparable(F)
iszeropreserving(f::SqrDistL2) = isseparable(f) && (iszero(f.λ) || iszeropreserving(f.f))
_zero_scaled(::SqrDistL2) = ZeroFunction()
_checkparams(f::SqrDistL2, x) = _checkparams(f.f, x)

function _checkwrite(op::O, f::SqrDistL2, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}}
  _checkreal(f, x)
  _checkreal(f, y)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(f, x)
  # the set's check on what its projection receives, the distance buffer; it
  # runs at a zero weight too, as the structural checks do
  T = float(eltype(op === prox! || op === value_and_prox! ? y : x))
  _checkwrite(prox!, f.f, _checkbuffer(f.scratch, T, size(x)), x)
  f.f isa IndSphereL2 && isempty(x) &&
    throw(ArgumentError("the sphere has no point of size $(size(x))"))
  true
end

_distance_buffer(f::SqrDistL2, ::Type{T}, x::AbstractArray) where {T} = _array_buffer!(f.scratch, T, size(x))

# Scale the distance before squaring, without taking the reciprocal of λ.
# Scale endpoints separately only when their subtraction itself overflows.
function _distance_energy(x::AbstractArray, p::AbstractArray, λ::Real)
  r = norm_diff(x, p)
  scale = sqrt(λ) / sqrt(oftype(λ, 2))
  isfinite(r) && return abs2(r * scale)
  q = zero(r)
  @inbounds for i in eachindex(x, p)
    d = x[i] - p[i]
    z = isfinite(d) ? d * scale : x[i] * scale - p[i] * scale
    q = hypot(q, abs(z))
  end
  abs2(q)
end

@inline function _distance_derivative(x, p, λ)
  d = x - p
  isfinite(d) ? λ * d : λ * x - λ * p
end

function (f::SqrDistL2)(x::AbstractArray)
  _checkwrite(prox!, f, x, x)
  T = float(eltype(x))
  λ = convert(T, f.λ)
  iszero(λ) && return zero(T)
  p = _distance_buffer(f, T, x)
  prox!(p, f.f, x, one(T))
  _distance_energy(x, p, λ)
end

function value_and_gradient!(g::AbstractArray, f::SqrDistL2, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  T = float(eltype(x))
  λ = convert(T, f.λ)
  iszero(λ) && (fill!(g, zero(eltype(g))); return zero(T))
  p = _distance_buffer(f, T, x)
  prox!(p, f.f, x, one(T))
  v = _distance_energy(x, p, λ)
  g .= _distance_derivative.(x, p, λ)
  v
end

function gradient!(g::AbstractArray, f::SqrDistL2, x::AbstractArray)
  _checkwrite(gradient!, f, g, x)
  T = float(eltype(x))
  λ = convert(T, f.λ)
  iszero(λ) && return fill!(g, zero(eltype(g)))
  p = _distance_buffer(f, T, x)
  prox!(p, f.f, x, one(T))
  g .= _distance_derivative.(x, p, λ)
end

# A rounded subnormal weight loses precision before multiplying an endpoint.
# If γλ overflows, both finite factors exceed one, so successive divisions
# preserve any representable correction, even when its weight rounds to zero.
@inline function _distance_blend(x, p, γ::T, λ::T) where {T}
  t = γ * λ
  if floatmin(T) <= t <= floatmax(T)
    a, b = _ridge_weights(γ, λ)
    return _ridge_blend(x, p, a, b)
  end
  x, p = T(x), T(p)
  if isfinite(t)
    d = p - x
    correction = if isfinite(d)
      copysign(_positive_product(γ, λ, abs(d)), d)
    else
      copysign(_positive_product(γ, λ, abs(p)), p) -
        copysign(_positive_product(γ, λ, abs(x)), x)
    end
    return x + correction / (one(T) + t)
  end
  d = x - p
  correction = isfinite(d) ? (d / γ) / λ : (x / γ) / λ - (p / γ) / λ
  p + correction
end

function prox!(y::AbstractArray, f::SqrDistL2, x::AbstractArray, γ::Real)
  _checkwrite(prox!, f, y, x)
  T = float(eltype(y))
  λ, step = convert(T, f.λ), convert(T, γ)
  (iszero(λ) || iszero(step)) && return copyto!(y, x)
  p = _distance_buffer(f, T, x)
  prox!(p, f.f, x, one(T))
  y .= _distance_blend.(x, p, step, λ)
end

function value_and_prox!(y::AbstractArray, f::SqrDistL2, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, f, y, x)
  T = float(eltype(y))
  λ, step = convert(T, f.λ), convert(T, γ)
  iszero(λ) && (copyto!(y, x); return zero(T))
  p = _distance_buffer(f, T, x)
  prox!(p, f.f, x, one(T))
  y .= _distance_blend.(x, p, step, λ)
  _distance_energy(y, p, λ)
end

Base.@propagate_inbounds function prox_entry(f::SqrDistL2, v::Real, i::Integer, γ::Real)
  isseparable(f) || throw(ArgumentError("prox_entry of SqrDistL2 requires a separable set"))
  T = typeof(float(v))
  λ, step = convert(T, f.λ), convert(T, γ)
  (iszero(λ) || iszero(step)) && return T(v)
  p = prox_entry(f.f, v, i, one(T))
  _distance_blend(v, p, step, λ)
end

function kkt_violation(f::SqrDistL2, g::AbstractArray, i::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
  isseparable(f) || throw(ArgumentError("kkt_violation of SqrDistL2 requires a separable set"))
  iszero(f.λ) && return abs(g[i])
  T = float(eltype(g))
  abs(g[i] - f.λ * prox_entry(f.f, zero(T), i, one(T)))
end

function restrict(f::SqrDistL2, S::AbstractVector{<:Integer})
  isseparable(f) || throw(ArgumentError("restrict of SqrDistL2 requires a separable set"))
  # the dropped terms are λ dist²(0, Cᵢ)/2, finite for a nonempty Cᵢ
  SqrDistL2(_restrict_unchecked(f.f, S), f.λ)
end

function lipschitz(f::SqrDistL2)
  (iszero(f.λ) || isconvex(f)) && return float(f.λ)
  invoke(lipschitz, Tuple{AbstractFunction}, f)
end

"""
    DistL2(C, λ = 1)

The Euclidean distance to a nonempty closed set `C`, scaled as
`f(x) = λ dist(x, C)`, for finite `λ ≥ 0`. `C` is a proximable indicator
function whose proximal operator is an exact projection in its argument space.
With `p ∈ P_C(x)` and `d = ‖x - p‖₂`, the proximal operator is `p` when
`d ≤ γλ`, and `x + (γλ/d)(p - x)` otherwise (Beck, 2017, Chapter 6).
The formula also holds for nonconvex sets, using the projection that `C`
chooses at ties. `DistL2(IndBallL2(0), λ)` is [`NormL2`](@ref)`(λ)`.

Convex when `C` is convex, not separable, and not smooth in general. For a
convex `C` and `λ > 0`, the distance is nonsmooth on its boundary and zero,
hence smooth, in its interior. No gradient oracles are provided, and
[`lipschitz`](@ref), the gradient's constant, throws. For an `L`-Lipschitz
loss with a constrained minimizer on `C`, `λ > L` makes every minimizer of
`loss + DistL2(C, λ)` feasible; equality need not suffice.

The oracles use one projection into a buffer that `f` owns; do not share `f`
between threads. The proximal output may be `x`. The buffer follows the
argument's precision for values, and the output's for proximal operators.
`value_and_prox!` measures the distance from the written output to the retained
projection, without a second projection. At `λ = 0`, the value is zero and
the proximal operator is the identity, after structural checks, without a
projection. Only scalar steps are supported. `Spectral(DistL2(...))` and
`Constrained(DistL2(...), box)` are not supported.

Nonemptiness, closedness and exactness of the projection are the caller's
responsibility. Results are approximate for an inexact [`IndIntersection`](@ref),
partial spectral decomposition or truncated [`IndAffine`](@ref) system.
Nonemptiness of a [`Constrained`](@ref) set is not checked. [`Spectral`](@ref)
sets read one triangle of a symmetric matrix; supply a symmetric argument.
Landing in `C` holds to the projection's accuracy.
"""
struct DistL2{F<:AbstractFunction, T<:Real} <: AbstractFunction
  f::F
  λ::T
  scratch::Scratch

  function DistL2{F, T}(C::F, λ::T) where {F<:AbstractFunction, T<:Real}
    _isindicator(C) || throw(ArgumentError("DistL2 requires an indicator function of a set"))
    isproximable(C) || throw(ArgumentError("DistL2 requires a proximable set whose proximal operator is a projection"))
    _checknonneg("λ", λ)
    isfinite(λ) || throw(ArgumentError("λ must be finite, got $λ"))
    new{F, T}(C, λ, Scratch())
  end
end
DistL2(C::AbstractFunction, λ::Real = 1) = DistL2{typeof(C), typeof(λ)}(C, λ)

isproximable(::Type{<:DistL2}) = true
isconvex(::Type{<:DistL2{F}}) where {F} = isconvex(F)
_zero_scaled(::DistL2) = ZeroFunction()
_checkparams(f::DistL2, x) = _checkparams(f.f, x)

function _checkwrite(op::O, f::DistL2, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkparams(f, x)
  _checkwrite(prox!, f.f, y, x)
  f.f isa IndSphereL2 && isempty(x) &&
    throw(ArgumentError("the sphere has no point of size $(size(x))"))
  true
end

_distance_buffer(f::DistL2, ::Type{T}, x::AbstractArray) where {T} = _array_buffer!(f.scratch, T, size(x))

@inline _distl2_ldexp(x::Real, e::Int) = ldexp(x, e)

# Keep distances beyond floatmax in units of a power of two. Scaling the
# endpoints before subtraction also covers an overflowing single residual.
function _distl2_parts(x::AbstractArray, p::AbstractArray)
  r = norm_diff(x, p)
  isfinite(r) && return frexp(r)
  S = float(promote_type(eltype(x), eltype(p)))
  m = zero(S)
  @inbounds for i in eachindex(x, p)
    m = max(m, abs(S(x[i])), abs(S(p[i])))
  end
  isfinite(m) || return (r, 0)
  k = _magnitude_exponent(m)
  r = zero(r)
  @inbounds for i in eachindex(x, p)
    d = _distl2_ldexp(S(x[i]), -k) - _distl2_ldexp(S(p[i]), -k)
    r = hypot(r, abs(d))
  end
  m, e = frexp(r)
  m, e + k
end

# Delay narrowing λ until its exponent has been combined with the distance.
@inline function _distl2_value(::Type{T}, λ::Real, m::Real, e::Int) where {T}
  a, k = frexp(float(λ))
  T(ldexp(a * m, k + e))
end

function (f::DistL2)(x::AbstractArray)
  _checkwrite(prox!, f, x, x)
  T = float(eltype(x))
  iszero(f.λ) && return zero(T)
  p = _distance_buffer(f, T, x)
  prox!(p, f.f, x, one(T))
  m, e = _distl2_parts(x, p)
  _distl2_value(T, f.λ, m, e)
end

# A subnormal interpolation weight can still move a large endpoint. Apply
# its exponent only after multiplication by the residual's significand.
@inline function _distl2_blend(x::Real, p::Real, a::T, b::T, m::T, e::Int) where {T}
  x, p = T(x), T(p)
  b >= floatmin(T) && return _ridge_blend(x, p, a, b)
  d = p - x
  k = 0
  if !isfinite(d) && isfinite(x) && isfinite(p)
    d = p / 2 - x / 2
    k = 1
  end
  r, j = frexp(d)
  x + ldexp(m * r, e + j + k)
end

# Near the threshold, subtract squared distances with their rounding errors
# retained. Subtracting a rounded norm would lose the small remaining distance.
@inline function _distl2_square_parts(x::Real, p::Real, e::Int, ::Type{S}) where {S}
  x, p = S(x), S(p)
  d = x - p
  k = -e
  if !isfinite(d)
    x, p = x / 2, p / 2
    d = x - p
    k += 1
  end
  b = d - x
  err = (x - (d - b)) - (p + b)
  v, r = ldexp(d, k), ldexp(err, k)
  q = v * v
  q, fma(v, v, -q) + 2v * r + r * r
end

function _distl2_margin(x, p, gm::S, lm::S, k::Int, e::Int) where {S}
  s, err = zero(S), zero(S)
  @inbounds for i in eachindex(x, p)
    q, r = _distl2_square_parts(x[i], p[i], e, S)
    v = s + q
    z = v - s
    err += (s - (v - z)) + (q - z) + r
    s = v
  end
  u = gm * lm
  t = ldexp(u, k - e)
  r = ldexp(fma(gm, lm, -u), k - e)
  q = t * t
  (s - q) + (err - (fma(t, t, -q) + 2t * r + r * r)), t
end

function _distl2_prox!(y::AbstractArray, x::AbstractArray, p::AbstractArray, γ::Real, λ::Real)
  iszero(γ) && return copyto!(y, x)
  S = promote_type(float(eltype(x)), float(eltype(y)), typeof(float(γ)), typeof(float(λ)))
  dm, de = _distl2_parts(x, p)
  gm, ge = frexp(S(γ))
  lm, le = frexp(S(λ))
  tm, te = frexp(gm * lm)
  te += ge + le
  # Non-finite arguments must propagate even when the step is infinite.
  snap = isfinite(dm) && !isnan(tm) && (isinf(tm) ||
    (iszero(dm) && tm >= 0) || (!iszero(tm) && (de < te || (de == te && dm <= tm))))
  refined = false
  a = zero(S)
  if isfinite(dm) && dm > 0 && isfinite(tm) && tm > 0 && abs(te - de) <= 1
    t = ldexp(tm, te - de)
    if S(0.75) * dm < t <= 2S(dm)
      margin, t = _distl2_margin(x, p, gm, lm, ge + le, de)
      snap = margin <= 0
      a = margin / (S(dm) * (S(dm) + t))
      refined = true
    end
  end
  snap && return copyto!(y, p)
  if !isfinite(dm) || !isfinite(tm)
    y .= x .+ (tm / dm) .* (p .- x)
    return y
  end
  m, e = frexp(tm / S(dm))
  e += te - de
  b = refined ? one(S) - a : ldexp(m, e)
  refined || (a = (S(dm) - ldexp(tm, te - de)) / S(dm))
  y .= _distl2_blend.(x, p, a, b, m, e)
end

function prox!(y::AbstractArray, f::DistL2, x::AbstractArray, γ::Real)
  _checkwrite(prox!, f, y, x)
  (iszero(f.λ) || iszero(γ)) && return copyto!(y, x)
  T = float(eltype(y))
  p = _distance_buffer(f, T, x)
  prox!(p, f.f, x, one(T))
  _distl2_prox!(y, x, p, γ, f.λ)
end

function value_and_prox!(y::AbstractArray, f::DistL2, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, f, y, x)
  T = float(eltype(y))
  iszero(f.λ) && (copyto!(y, x); return zero(T))
  p = _distance_buffer(f, T, x)
  prox!(p, f.f, x, one(T))
  _distl2_prox!(y, x, p, γ, f.λ)
  m, e = _distl2_parts(y, p)
  _distl2_value(T, f.λ, m, e)
end
