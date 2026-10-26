####################################
#
#  A norm on an ℓ2 ball or sphere
#
####################################

"""
    Constrained(f, IndBallL2(r))
    Constrained(f, IndSphereL2(r))

The function `f(x)` on the Euclidean ball `‖x‖₂ ≤ r` or on the sphere `‖x‖₂ = r`
(the Frobenius ball and sphere for matrices), and `Inf` outside it: the problems
of sparse principal component and canonical correlation analysis, whose
alternating steps put a shrunken vector on the ball or the sphere.

On the ball, `f` is positively homogeneous: a norm (`NormL1`, `NormL1Nonneg`,
`NormL2`, `NormLinf`, `GroupNormL2`, `GroupNormLinf`, `SparseGroupNormL2`,
`NuclearNorm`, `TracePSD`, or a `Scaled` one) or the indicator of a cone (`IndNonnegative`, `IndPSD`, and
the nonconvex `IndBallL0` and `IndBallRank`). The proximal operator is the
proximal point of `f` scaled onto the ball, and the value scales with it:
`f(tz) = t f(z)` for `t > 0`.

On the sphere, `f` is `NormL1`, `GroupNormL2` or `IndBallL0`, and the proximal
operator is the global minimizer of the nonconvex problem: the proximal point
of `f` scaled onto the sphere. When that point is zero, so that every entry of
`x` is within its threshold, the minimizer puts all the mass on one entry (one
group) where `|xᵢ|` exceeds the threshold the least, along the sign of `xᵢ` (the
direction of `x` on the group).

```julia
u = prox(Constrained(NormL1(λ), IndBallL2(1)), X * v)        # the step of the penalized matrix decomposition
u = prox(Constrained(IndBallL0(k), IndSphereL2(1)), S * u)    # the truncated power method
```
"""
Constrained(f::AbstractFunction, C::IndBallL2) = (_scales_onto_ball(f) || _ball_error(f); BallConstrained(f, C))

# the functions whose proximal point, scaled onto an ℓ2 ball, is the proximal
# point on the ball. For a positively homogeneous convex f, ∂f(tz) = ∂f(z) for
# t > 0, so the optimality condition of the proximal point carries over, with
# the multiplier of the ball. For the indicator of a cone C, nonconvex too, the
# best point on the ball supported on a face of C is the projection there scaled
# onto the ball, at distance² ‖x‖² - ‖P(x)‖² + (‖P(x)‖ - r)₊², which decreases in
# ‖P(x)‖: the projection onto C already maximizes it.
_scales_onto_ball(::AbstractFunction) = false
_scales_onto_ball(::Union{ZeroFunction, NormL1, NormL1Nonneg, NormL2, NormLinf, HomogeneousGroupNorm,
                          NuclearNorm, TracePSD, IndNonnegative, IndPSD, IndBallL0, IndBallRank}) = true
_scales_onto_ball(s::Scaled) = _scales_onto_ball(s.f)

@noinline _ball_error(f) = throw(ArgumentError(
  "Constrained on an ℓ2 ball takes a positively homogeneous function, a norm or the indicator of a cone, not $(nameof(typeof(f)))"))

Constrained(f::Union{NormL1, GroupNormL2, IndBallL0}, C::IndSphereL2) = SphereConstrained(f, C)
Constrained(f::AbstractFunction, ::IndSphereL2) =
  throw(ArgumentError("Constrained on an ℓ2 sphere takes NormL1, GroupNormL2 or IndBallL0, not $(nameof(typeof(f)))"))

# f on the ball {‖x‖₂ ≤ r}
struct BallConstrained{F<:AbstractFunction, T<:Real} <: AbstractFunction
  f::F
  ball::IndBallL2{T}
end

# f on the sphere {‖x‖₂ = r}
struct SphereConstrained{F<:AbstractFunction, T<:Real} <: AbstractFunction
  f::F
  sphere::IndSphereL2{T}
end

isproximable(::Type{<:Union{BallConstrained, SphereConstrained}}) = true
isconvex(::Type{<:BallConstrained{F}}) where {F} = isconvex(F)
_isindicator(::Type{<:BallConstrained{F}}) where {F} = _isindicator(F)
_isindicator(::Type{<:SphereConstrained{F}}) where {F} = _isindicator(F)
_checkparams(c::Union{BallConstrained, SphereConstrained}, x) = _checkparams(c.f, x)

(c::BallConstrained)(x::AbstractArray) = (T = float(eltype(x)); iszero(c.ball(x)) ? convert(T, c.f(x)) : T(Inf))
(c::SphereConstrained)(x::AbstractArray) = (T = float(eltype(x)); iszero(c.sphere(x)) ? convert(T, c.f(x)) : T(Inf))

prox!(y::AbstractArray, c::Union{BallConstrained, SphereConstrained}, x::AbstractArray, γ::Real) =
  (value_and_prox!(y, c, x, γ); y)

# the proximal point of f in y and its value, without the value for an indicator
function _value_and_inner_prox!(y, f, x, γ, ::Type{T}) where {T}
  _isindicator(f) && return (prox!(y, f, x, γ); zero(T))
  convert(T, value_and_prox!(y, f, x, γ))
end

function value_and_prox!(y::AbstractArray, c::BallConstrained, x::AbstractArray, γ::Real)
  T = float(eltype(y))
  v = _value_and_inner_prox!(y, c.f, x, γ, T)
  ny = sqrt(convert(T, _sumabs2(y)))
  r = convert(T, c.ball.r)
  if ny > r
    t = r / ny
    y .*= t
    v *= t
  end
  v
end

####################################
#  on the sphere
####################################

# y = r z/‖z‖ for the proximal point z ≠ 0 of f in y, and the value v of f at z
# scaled with it
function _onto_sphere!(y, v::T, r::T) where {T}
  t = r / sqrt(convert(T, _sumabs2(y)))
  y .*= t
  v * t
end

# On the sphere ‖z‖² is fixed, so the proximal operator maximizes
# ⟨z, x⟩ - γf(z). For the ℓ1 norm that is Σᵢ |zᵢ|(|xᵢ| - γλwᵢ) for z with the
# signs of x, maximized by z ∝ the soft thresholding of x, the proximal point,
# when an entry exceeds its threshold. Otherwise every coefficient is ≤ 0, and
# since Σᵢ|zᵢ| ≥ ‖z‖₂ the maximum r ⋅ maxᵢ(|xᵢ| - γλwᵢ) is at one entry. The
# thresholds are those of prox_entry, so that the two cases agree with it.
function value_and_prox!(y::AbstractArray, c::SphereConstrained{<:NormL1}, x::AbstractArray, γ::Real)
  _checksize(y, x)
  _checkparams(c, x)
  T, S = float(eltype(y)), float(eltype(x))
  f, r = c.f, convert(T, c.sphere.r)
  k, best = 0, S(-Inf)
  @inbounds for i in eachindex(IndexLinear(), x)
    m = abs(convert(S, x[i])) - convert(S, γ * f.λ0 * _entry(f.w, i))
    m > best && ((k, best) = (i, m))
  end
  k == 0 && throw(ArgumentError("the sphere misses the domain of NormL1: every weight is infinite"))
  best > 0 && return _onto_sphere!(y, convert(T, value_and_prox!(y, f, x, γ)), r)
  negative = signbit(x[k])
  fill!(y, zero(eltype(y)))
  y[k] = negative ? -r : r
  convert(T, f.λ0) * convert(T, _entry(f.w, k)) * r
end

# The same with groups: ⟨z_G, x_G⟩ - γλ0w_G‖z_G‖ ≤ ‖z_G‖(‖x_G‖ - γλ0w_G), with
# equality along x_G, so the coefficient of a group is its margin over the
# threshold, and an index in no group has the coefficient |xᵢ|. The norms and
# thresholds are those of the group shrinkage.
function value_and_prox!(y::AbstractArray, c::SphereConstrained{<:GroupNormL2}, x::AbstractArray, γ::Real)
  _checksize(y, x)
  _checkgroups(c.f, x)
  T = float(eltype(y))
  U = promote_type(float(eltype(x)), T)       # the type in which the shrinkage compares
  f, r = c.f, convert(T, c.sphere.r)
  kbest, ibest, best = 0, 0, U(-Inf)
  for k in eachindex(f.groups)
    m = convert(U, sqrt(_sumsq(x, f.groups[k]))) - convert(U, convert(T, γ * f.λ0 * f.w[k]))
    m > best && ((kbest, ibest, best) = (k, 0, m))
  end
  @inbounds for i in eachindex(IndexLinear(), x)
    _group(f, i) == 0 || continue
    m = convert(U, abs(x[i]))
    m > best && ((kbest, ibest, best) = (0, i, m))
  end
  best > 0 && return _onto_sphere!(y, convert(T, value_and_prox!(y, f, x, γ)), r)
  if ibest != 0                         # an index in no group, where x is zero
    fill!(y, zero(eltype(y)))
    y[ibest] = r
    return zero(T)
  end
  G = f.groups[kbest]
  nG = convert(T, sqrt(_sumsq(x, G)))
  @inbounds if iszero(nG)
    fill!(y, zero(eltype(y)))
    y[first(G)] = r
  else
    t = r / nG
    for i in G
      y[i] = t * x[i]
    end
    for i in eachindex(IndexLinear(), y)
      _group(f, i) == kbest || (y[i] = zero(eltype(y)))
    end
  end
  convert(T, f.λ0) * convert(T, f.w[kbest]) * r
end

# On the sphere a point with at most s nonzero entries has ⟨z, x⟩ ≤ r‖x_S‖ on its
# support S, so the best support carries the most of x: the s largest entries
# (and the free diagonal), the proximal point of the ℓ0 ball.
function value_and_prox!(y::AbstractArray, c::SphereConstrained{IndBallL0}, x::AbstractArray, γ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  c.f.s == 0 && _allcount(c.f, x) &&
    throw(ArgumentError("the sphere misses the ℓ0 ball of no nonzero entries"))
  prox!(y, c.f, x, γ)
  r = convert(T, c.sphere.r)
  if iszero(_sumabs2(y))
    # x is zero on the entries that count and on the free ones, so it is zero
    fill!(y, zero(eltype(y)))
    y[firstindex(y)] = r
  else
    _onto_sphere!(y, zero(T), r)
  end
  zero(T)
end
