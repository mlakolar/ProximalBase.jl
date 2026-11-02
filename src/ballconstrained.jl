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
direction of `x` on the group). If all thresholds exceed the finite range of
the working type, the method throws an `ArgumentError`; rescale the penalty
or step.

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

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, c::BallConstrained, y::AbstractArray, x::AbstractArray)
  _checkreal(c, y); _checkreal(c, x)
  _checkwrite(_isindicator(c.f) ? prox! : value_and_prox!, c.f, y, x)
  _isindicator(c.f) || _checkvalue(c.f, y)
  true
end
function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, c::SphereConstrained{<:NormL1}, y::AbstractArray, x::AbstractArray)
  _checkreal(c, y); _checkreal(c, x)
  _checksize(y, x)
  _checkparams(c, x)
  isempty(x) && throw(ArgumentError("the sphere has no point of size $(size(x))"))
  _checkwrite(value_and_prox!, c.f, y, x)
end
function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, c::SphereConstrained{<:GroupNormL2}, y::AbstractArray, x::AbstractArray)
  _checkreal(c, y); _checkreal(c, x)
  _checksize(y, x)
  _checkgroups(c.f, x)
  isempty(x) && isempty(c.f.groups) && throw(ArgumentError("the sphere has no point of size $(size(x))"))
  _checkwrite(value_and_prox!, c.f, y, x)
end
function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, c::SphereConstrained{IndBallL0}, y::AbstractArray, x::AbstractArray)
  _checkreal(c, y); _checkreal(c, x)
  _checksize(y, x)
  c.f.s == 0 && _allcount(c.f, x) &&
    throw(ArgumentError("the sphere misses the ℓ0 ball of no nonzero entries"))
  _checkwrite(prox!, c.f, y, x)
end
_checkvalue(c::BallConstrained, x::AbstractArray) = (_checkreal(c, x); _checkvalue(c.ball, x))
_checkvalue(c::SphereConstrained, x::AbstractArray) = (_checkreal(c, x); _checkvalue(c.sphere, x))

(c::BallConstrained)(x::AbstractArray) =
  (_check_value_arrays(c, x); _value_checked(c, x))
_value_unchecked(c::BallConstrained, x::AbstractArray) = (T = float(eltype(x)); iszero(_value_unchecked(c.ball, x)) ? convert(T, _value_unchecked(c.f, x)) : T(Inf))
(c::SphereConstrained)(x::AbstractArray) =
  (_check_value_arrays(c, x); _value_checked(c, x))
_value_unchecked(c::SphereConstrained, x::AbstractArray) = (T = float(eltype(x)); iszero(_value_unchecked(c.sphere, x)) ? convert(T, _value_unchecked(c.f, x)) : T(Inf))

prox!(y::AbstractArray, c::Union{BallConstrained, SphereConstrained}, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
_prox_unchecked!(y::AbstractArray, c::Union{BallConstrained, SphereConstrained}, x::AbstractArray, γ::Real) =
  (_value_and_prox_unchecked!(y, c, x, γ); y)

# the proximal point of f in y and its value, without the value for an indicator
function _value_and_inner_prox!(y, f, x, γ, ::Type{T}) where {T}
  _isindicator(f) && return (_prox_unchecked!(y, f, x, γ); zero(T))
  convert(T, _value_and_prox_unchecked!(y, f, x, γ))
end

value_and_prox!(y::AbstractArray, c::BallConstrained, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::BallConstrained, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, c, y, x)
  T = float(eltype(y))
  v = _value_and_inner_prox!(y, c.f, x, γ, T)
  ny = convert(T, _norm2(y))
  r = convert(T, c.ball.r)
  if ny > r
    t = _scale_l2!(y, y, r, ny)
    v = isfinite(v) && ny >= floatmin(T) && isfinite(t) && t >= floatmin(T) ? v * t :
        (_isindicator(c.f) ? zero(T) : convert(T, _value_unchecked(c.f, y)))
  end
  v
end

####################################
#  on the sphere
####################################

# y = r z/‖z‖ for the proximal point z ≠ 0 of f in y, and the value v of f at z
# scaled with it
function _onto_sphere!(y, f, v::T, r::T) where {T}
  n = convert(T, _norm2(y))
  t = _scale_l2!(y, y, r, n)
  isfinite(v) && n >= floatmin(T) && isfinite(t) && t >= floatmin(T) ? v * t :
    (_isindicator(f) ? zero(T) : convert(T, _value_unchecked(f, y)))
end

# On the sphere ‖z‖² is fixed, so the proximal operator maximizes
# ⟨z, x⟩ - γf(z). For the ℓ1 norm that is Σᵢ |zᵢ|(|xᵢ| - γλwᵢ) for z with the
# signs of x, maximized by z ∝ the soft thresholding of x, the proximal point,
# when an entry exceeds its threshold. Otherwise every coefficient is ≤ 0, and
# since Σᵢ|zᵢ| ≥ ‖z‖₂ the maximum r ⋅ maxᵢ(|xᵢ| - γλwᵢ) is at one entry. The
# thresholds are those of prox_entry, so that the two cases agree with it.
value_and_prox!(y::AbstractArray, c::SphereConstrained{<:NormL1}, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
# Margins n - t compared without rounding errors that decide the winner:
#   n - t > bn - bt  is  n + bt > bn + t, compared as normalized sums (hi, lo) by
#   error-free transformation; where a sum overflows, the same with every term halved
#   (exact for normal numbers, and a term that small cannot decide a comparison whose
#   other terms are that large).
@inline function _two_sum(a, b)
  s = a + b
  bb = s - a
  s, (a - (s - bb)) + (b - bb)
end
# The threshold γλ0w rounded once to U, with the product formed in the promotion of U and the
# factors' types, so that a product that overflows the factors' type is still finite in U.
@inline function _sphere_threshold(::Type{U}, γ, λ0, w) where {U<:AbstractFloat}
  R = promote_type(U, float(typeof(γ)), float(typeof(λ0)), float(typeof(w)))
  convert(U, _weight_product(convert(R, γ), convert(R, λ0), convert(R, w)))
end
@inline function _margin_greater(n, t, bn, bt)
  s1, e1 = _two_sum(n, bt)
  s2, e2 = _two_sum(bn, t)
  if !(isfinite(s1) && isfinite(s2))
    h = oftype(n, 0.5)
    s1, e1 = _two_sum(h * n, h * bt)
    s2, e2 = _two_sum(h * bn, h * t)
  end
  s1 != s2 ? s1 > s2 : e1 > e2
end
# √2 a > b, for the margin a of a pair and b of a diagonal entry, by a ratio so that
# nothing underflows or overflows: the sign cases first, then b/a against √2.
@inline function _pair_greater(a, b)
  a > 0 && return b <= 0 || b / a < sqrt(oftype(a, 2))
  a == 0 && return b < 0
  b < 0 && b / a > sqrt(oftype(a, 2))
end
# b > √2 a, for the margin b of a diagonal entry and a of a pair
@inline function _diagonal_greater(b, a)
  b > 0 && return a <= 0 || b / a > sqrt(oftype(a, 2))
  b == 0 && return a < 0
  a < 0 && b / a < sqrt(oftype(a, 2))
end

# ldexp(x, e) for 1/8 < x < 2, rounded once also where the result is subnormal (before
# Julia 1.11, ldexp truncates there): by a multiplication with a power of two
@inline function _ldexp_once(x::U, e::Integer) where {U<:AbstractFloat}
  e >= exponent(floatmin(U)) + 3 && return ldexp(x, e)
  k = precision(U) + 3
  ldexp(x, e + k) * ldexp(one(U), -k)
end

# The soft-thresholded point scaled onto the sphere of radius r, for the indices inds of x
# (a linear index also addresses the weight), with multiplicity mult(k) in the norm, written
# by put!(k, z). x is read at k before k is written, so y may be x.
function _sphere_l1_shrink!(y, x, f, γ, r, ::Type{U}, ::Type{T}, inds, mult, put!) where {U, T}
  # Soft thresholding and the scaling onto the sphere in the type of the comparison, so
  # that the entries that survive are those whose margin was positive and nothing is
  # narrowed to T before the norm is taken. The norm is scaled by the largest margin.
  # x is read before y is written, so y may be x.
  margin(k) = @inbounds max(abs(convert(U, x[k])) - _sphere_threshold(U, γ, f.λ0, _entry(f.w, k)), zero(U))
  top = zero(U)
  for k in inds
    top = max(top, margin(k))
  end
  ss = zero(U)
  for k in inds
    ss += mult(k) * (margin(k) / top)^2
  end
  ru, root = convert(U, r), sqrt(ss)       # 1 ≤ ss ≤ 2p², since the largest margin counts once or twice
  # y = margin ⋅ r / (top ⋅ root). When the factor r / (root ⋅ top) is a normal number one
  # product is accurate. Otherwise the three terms are split into mantissas and exponents,
  # so that nothing underflows or overflows before the one rounding of the result.
  q = ru / root
  scale = q / top
  direct = isfinite(scale) && scale >= floatmin(U)
  aq, eq = frexp(q)
  at, et = frexp(top)
  @inbounds for k in inds
    xi = x[k]
    m = margin(k)
    z = if iszero(m)
      zero(U)                           # not m * scale, which is 0 ⋅ Inf where scale overflows
    elseif direct
      m * scale
    else
      am, em = frexp(m)
      _ldexp_once((am / at) * aq, em + eq - et)
    end
    put!(k, copysign(convert(T, z), xi))
  end
  convert(T, _value_unchecked(f, y))
end

function _value_and_prox_unchecked!(y::SymmetricSparseIterate, c::SphereConstrained{<:NormL1}, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, c, y, x)
  T = float(eltype(y))
  U = promote_type(float(eltype(x)), T)       # the type in which the margins compare, as in the group sphere
  f, r = c.f, convert(T, c.sphere.r)
  p = size(y, 1)
  k, kr, kc, bestn, bestc = 0, 0, 0, zero(U), zero(U)
  @inbounds for col in 1:p, row in col:p
    i = row + (col - 1) * p
    n = abs(convert(U, x[row, col]))
    threshold = _sphere_threshold(U, γ, f.λ0, _entry(f.w, i))   # not rounded to T first: selection and shrinkage agree
    better = if k == 0 || isinf(bestc)
      true
    elseif isinf(threshold)
      false
    elseif (row == col) == (kr == kc)         # the same multiplicity: compare the margins
      _margin_greater(n, threshold, bestn, bestc)
    elseif row == col                         # a diagonal entry against a pair, whose margin counts √2 times
      _diagonal_greater(n - threshold, bestn - bestc)
    else
      _pair_greater(n - threshold, bestn - bestc)
    end
    better && ((k, kr, kc, bestn, bestc) = (i, row, col, n, threshold))
  end
  isfinite(bestc) || throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step, and give some entry a finite weight"))
  if bestn > bestc
    p = size(y, 1)
    return _sphere_l1_shrink!(y, x, f, γ, r, U, T, (row + (col - 1) * p for col in 1:p for row in col:p),
      k -> mod1(k, p) == cld(k, p) ? one(U) : U(2), (k, z) -> (y[mod1(k, p), cld(k, p)] = z))
  end
  negative = signbit(x[kr, kc])
  fill!(y, zero(eltype(y)))
  kr == kc && (y[kr, kc] = negative ? -r : r; return convert(T, _positive_product(f.λ0, _entry(f.w, k), r)))
  # an off-diagonal pair reaches the sphere at ±r/√2 in each of its two entries
  ry = r / sqrt(T(2))
  y[kr, kc] = negative ? -ry : ry
  convert(T, 2 * _positive_product(f.λ0, _entry(f.w, k), ry))
end
function _value_and_prox_unchecked!(y::AbstractArray, c::SphereConstrained{<:NormL1}, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, c, y, x)
  T = float(eltype(y))
  U = promote_type(float(eltype(x)), T)       # the type in which the margins compare, as for a symmetric output
  f, r = c.f, convert(T, c.sphere.r)
  k, bestn, bestc = 0, zero(U), zero(U)
  @inbounds for i in eachindex(IndexLinear(), x)
    n = abs(convert(U, x[i]))
    threshold = _sphere_threshold(U, γ, f.λ0, _entry(f.w, i))
    # margins n - t compared without rounding (see _margin_greater)
    (k == 0 || isinf(bestc) || (!isinf(threshold) && _margin_greater(n, threshold, bestn, bestc))) && ((k, bestn, bestc) = (i, n, threshold))
  end
  isfinite(bestc) || throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step, and give some entry a finite weight"))
  bestn > bestc && return _sphere_l1_shrink!(y, x, f, γ, r, U, T, eachindex(IndexLinear(), x), _ -> one(U), (i, z) -> (y[i] = z))
  negative = signbit(x[k])
  fill!(y, zero(eltype(y)))
  y[k] = negative ? -r : r
  convert(T, _positive_product(f.λ0, _entry(f.w, k), r))
end

# On a SymmetricSparseIterate an off-diagonal index is two entries, so a point of
# norm r takes r/√2 in each of them; a diagonal index and any other output take r.
_write_unit!(y, i, r) = (y[i] = r; y)
function _write_unit!(y::SymmetricSparseIterate, i::Integer, r)
  p = size(y, 1)
  row, col = mod1(i, p), cld(i, p)
  y[row, col] = row == col ? r : r / sqrt(oftype(r, 2))
  y
end
# the first index of a group that is a diagonal entry of a SymmetricSparseIterate, else its first
_diagonal_first(y, G) = first(G)
function _diagonal_first(y::SymmetricSparseIterate, G)
  p = size(y, 1)
  for i in G
    mod1(i, p) == cld(i, p) && return i
  end
  first(G)
end

# The same with groups: ⟨z_G, x_G⟩ - γλ0w_G‖z_G‖ ≤ ‖z_G‖(‖x_G‖ - γλ0w_G), with
# equality along x_G, so the coefficient of a group is its margin over the
# threshold, and an index in no group has the coefficient |xᵢ|. The norms and
# thresholds are those of the group shrinkage.
value_and_prox!(y::AbstractArray, c::SphereConstrained{<:GroupNormL2}, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::SphereConstrained{<:GroupNormL2}, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, c, y, x)
  T = float(eltype(y))
  U = promote_type(float(eltype(x)), T)       # the type in which the shrinkage compares
  f, r = c.f, convert(T, c.sphere.r)
  kbest, ibest, bestn, bestc = 0, 0, zero(U), zero(U)
  for k in eachindex(f.groups)
    n = convert(U, _group_norm2(x, f.groups[k]))
    threshold = convert(U, convert(T, _weight_product(γ, f.λ0, f.w[k])))
    (kbest == 0 || isinf(bestc) || (!isinf(threshold) && n - bestn > threshold - bestc)) && ((kbest, bestn, bestc) = (k, n, threshold))
  end
  @inbounds for i in eachindex(IndexLinear(), x)
    _group(f, i) == 0 || continue
    n = convert(U, abs(x[i]))
    ((kbest == 0 && ibest == 0) || isinf(bestc) || n - bestn > -bestc) && ((kbest, ibest, bestn, bestc) = (0, i, n, zero(U)))
  end
  isfinite(bestc) || throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step"))
  bestn > bestc && return _onto_sphere!(y, f, convert(T, _value_and_prox_unchecked!(y, f, x, γ)), r)
  if ibest != 0                         # an index in no group, where x is zero
    fill!(y, zero(eltype(y)))
    _write_unit!(y, ibest, r)
    return zero(T)
  end
  G = f.groups[kbest]
  nG = convert(T, _group_norm2(x, G))
  @inbounds if iszero(nG)
    fill!(y, zero(eltype(y)))
    _write_unit!(y, _diagonal_first(y, G), r)
  else
    _scale_l2!(view(y, G), view(x, G), r, nG)
    for i in eachindex(IndexLinear(), y)
      _group(f, i) == kbest || (y[i] = zero(eltype(y)))
    end
  end
  convert(T, _group_value(f, f.λ0, f.w[kbest], r, y, G, T))
end

# On the sphere a point with at most s nonzero entries has ⟨z, x⟩ ≤ r‖x_S‖ on its
# support S, so the best support carries the most of x: the s largest entries
# (and the free diagonal), the proximal point of the ℓ0 ball.
value_and_prox!(y::AbstractArray, c::SphereConstrained{IndBallL0}, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::SphereConstrained{IndBallL0}, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, c, y, x)
  T = float(eltype(y))
  _prox_unchecked!(y, c.f, x, γ)
  r = convert(T, c.sphere.r)
  if iszero(_norm2(y))
    # x is zero on the entries that count and on the free ones, so it is zero
    fill!(y, zero(eltype(y)))
    y[firstindex(y)] = r
  else
    _onto_sphere!(y, c.f, zero(T), r)
  end
  zero(T)
end
