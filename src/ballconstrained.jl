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

The step `γ` is a scalar in `[0, Inf]`; a negative or NaN step throws an
`ArgumentError` before anything is written. On the ball, `γ = Inf` projects onto
the minimizers of `f`. On the sphere, `γ = Inf` is a step when the level is zero
or some free entry (group) has weight zero, and projects onto the sphere within
the zero-penalty entries; when every free entry (group) is penalized it throws.

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
  _checkstep_nonneg(γ)
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
# The same comparison at the cost of one subtraction where that decides it. Rounding to nearest
# is monotone, so rounded margins that differ are ordered as the exact ones; only equal rounded
# margins can hide the order, and _margin_greater decides those (exact ties included, which
# stay false). A NaN margin compares false either way, and an infinite n or bn orders the
# rounded margins as _margin_greater does, so the result is _margin_greater's for every input.
@inline function _margin_better(n, t, bn, bt)
  m, bm = n - t, bn - bt
  m > bm || (m == bm && _margin_greater(n, t, bn, bt))
end
# The sign of D - √2 P for the exact margins D = nd - td of a diagonal entry and P = np - tp of
# an off-diagonal pair, whose objective counts √2 times (A2); the tie, sign 0, is only D = P = 0,
# since √2 is irrational. A rounded difference of two nonnegative numbers has the sign of the
# exact one, so the signs decide unless both margins have one sign. Then the rounded ratio
# |d|/|p| is within 3 roundings of |D|/|P| (an overflow or underflow is far from √2), so it
# decides outside 8 ulps of √2, and within them D² and 2P² are compared exactly. Nothing
# allocates for machine floats. With a non-finite entry the result does not matter, since the
# block is NaN (E2); nothing throws.
@inline function _diagonal_pair_sign(nd, td, np, tp)
  d, p = nd - td, np - tp
  s = d > 0 && p > 0 ? 1 : d < 0 && p < 0 ? -1 : 0
  s == 0 && return d > 0 ? 1 : d < 0 ? -1 : p > 0 ? -1 : p < 0 ? 1 : 0   # also 0 at a NaN
  ρ, s2, δ = d / p, sqrt(oftype(d, 2)), 8 * eps(oftype(d, 1))   # s2 ± δ: 8 ulps, exactly
  ρ > s2 + δ && return s
  ρ < s2 - δ && return -s
  s * _diagonal_pair_exact(nd, td, np, tp, s)
end

# The type of the exact comparison: every machine float is exact in Float64, whose exponent range
# holds the scaled products below; a BigFloat (or another float) computes in the current BigFloat
# precision (T7), whose exponent range is unbounded for this purpose.
_pair_exact_type(::Type{<:Union{Float16, Float32, Float64}}) = Float64
_pair_exact_type(::Type) = BigFloat
@inline _two_prod(a, b) = (h = a * b; (h, fma(a, b, -h)))

# sign(A² - 2B²) for A = |D|, B = |P| (sign s of D and P), within the band, so 1 < A/B < 2.
# A and B are exact pairs ah + al and bh + bl (a difference of two numbers is exact as a sum by
# _two_sum). Let 2^E ≤ ah < 2^(E+1) and q the precision. F(al, bl) = (ah + al)² - 2(bh + bl)² is a
# multiple of u², for u the smallest ulp of its operands, and is not 0 (√2 is irrational), so
# |F| ≥ u²; setting a tail x to 0 changes F by less than 2^(E+4)|x|. With both tails dropped,
# u ≥ 2^(E-q), so they go if both are below 2^(E-2q-5). Otherwise the larger one stays, so
# u ≥ 2^(E-3q-4), and the smaller goes if below 2^(E-6q-12). Scaled by 2^-E, the remaining terms
# lie in [2^-(6q+12), 2), or are at least the smallest subnormal scaled by 2^-E, so their products
# and the products' rounding errors are normal numbers in Float64: F is the exact sum of 12
# numbers, whose sign comes from an exact expansion.
@noinline function _diagonal_pair_exact(nd, td, np, tp, s)
  isfinite(nd) && isfinite(np) || return 0     # an infinite entry: the block is NaN (E2)
  W = _pair_exact_type(typeof(nd))
  nd, td, np, tp = W(nd), W(td), W(np), W(tp)
  ah, al = s > 0 ? _two_sum(nd, -td) : _two_sum(td, -nd)
  bh, bl = s > 0 ? _two_sum(np, -tp) : _two_sum(tp, -np)
  E, q = exponent(ah), precision(W)
  if max(abs(al), abs(bl)) < ldexp(one(W), E - 2q - 5)
    al, bl = zero(W), zero(W)
  else
    lim = ldexp(one(W), E - 6q - 12)
    abs(al) < lim && (al = zero(W))
    abs(bl) < lim && (bl = zero(W))
  end
  A, a, B, b = ldexp(ah, -E), ldexp(al, -E), ldexp(bh, -E), ldexp(bl, -E)
  iszero(a) && iszero(b) && return _sum_sign((_two_prod(A, A)..., _two_prod(-2B, B)...))   # exact margins
  _sum_sign((_two_prod(a, a)..., _two_prod(-2b, b)..., _two_prod(2A, a)..., _two_prod(-4B, b)...,
             _two_prod(A, A)..., _two_prod(-2B, B)...))
end

# The sign of the exact sum of the terms. A pass of error-free sums (VecSum) keeps the sum and
# leaves its rounded value last; when that dominates twice the sum of the other magnitudes (whose
# rounding error is far smaller), it has the sign of the sum. Otherwise, after a few passes, an
# exact expansion decides.
function _sum_sign(e::Tuple{T, Vararg{T}}) where {T}
  N = length(e)
  for _ in 1:3
    e = _vecsum(e)
    rest = sum(abs, Base.front(e))
    abs(e[N]) > 2rest && return Int(sign(e[N]))
    iszero(rest) && return Int(sign(e[N]))
  end
  _expansion_sign(e)
end
@inline _vecsum(e::NTuple{1}) = e
@inline function _vecsum(e::NTuple{N}) where {N}
  s, l = _two_sum(e[2], e[1])
  (l, _vecsum((s, Base.tail(Base.tail(e))...))...)
end

# The sign of the exact sum of the terms: Shewchuk's Grow-Expansion adds them one by one into a
# nonoverlapping expansion of increasing magnitude, whose largest nonzero component has the sign
# of the sum. The sums stay far from overflow here.
function _expansion_sign(e::Tuple{T, Vararg{T}}) where {T}
  N = length(e)
  for j in 2:N
    h = e[j]
    for i in 1:(j - 1)
      h, l = _two_sum(h, e[i])
      e = Base.setindex(e, l, i)
    end
    e = Base.setindex(e, h, j)
  end
  for i in N:-1:1
    iszero(e[i]) || return Int(sign(e[i]))
  end
  0
end

# ldexp(x, e) for 1/8 < x < 2, rounded once also where the result is subnormal (before
# Julia 1.11, ldexp truncates there): by a multiplication with a power of two
@inline function _ldexp_once(x::U, e::Integer) where {U<:AbstractFloat}
  e >= exponent(floatmin(U)) + 3 && return ldexp(x, e)
  k = precision(U) + 3
  ldexp(x, e + k) * ldexp(one(U), -k)
end

# The output for an argument with a NaN or infinite entry. The sphere couples every entry, so a
# NaN entry is NaN in the whole block (E2); a comparison with NaN selects nothing, so the
# candidate selection cannot be relied on to reach it. An infinite entry gives the same block,
# as in every ball and sphere projection of the package: one infinite coordinate has a limit on
# the sphere (±r there), several have none, and no coupled projection documents its limits yet
# (E3, #469). Called after the thresholds are validated (E1); nothing is read from x, so y may be x.
function _sphere_nonfinite!(y, ::Type{T}) where {T}
  fill!(y, T(NaN))
  T(NaN)
end

# The soft-thresholded point scaled onto the sphere of radius r, over the indices of the layout
# L (a linear index also addresses the weight), whose entries count with a multiplicity in the
# norm. A layout is a type, not closures, so that nothing allocates on any Julia version. x is
# read at k before k is written, so y may be x.
struct _DenseLayout end                      # every entry once
struct _LowerLayout                           # the lower triangle of a symmetric p × p point
  p::Int
end
_layout_indices(::_DenseLayout, x) = eachindex(IndexLinear(), x)
_layout_indices(L::_LowerLayout, x) = (row + (col - 1) * L.p for col in 1:L.p for row in col:L.p)
_layout_mult(::_DenseLayout, ::Type{U}, k) where {U} = one(U)
_layout_mult(L::_LowerLayout, ::Type{U}, k) where {U} = mod1(k, L.p) == cld(k, L.p) ? one(U) : U(2)
@inline _layout_put!(::_DenseLayout, y, k, z) = (y[k] = z)
@inline _layout_put!(L::_LowerLayout, y, k, z) = (y[mod1(k, L.p), cld(k, L.p)] = z)
@inline _l1_margin(x, f, γ, ::Type{U}, k) where {U} =
  @inbounds max(abs(convert(U, x[k])) - _weight_product(U, γ, f.λ0, _entry(f.w, k)), zero(U))
# Σ multiplicity ⋅ (margin / top)². Over every entry the sum may be reassociated by SIMD (A6),
# as the norms of the package are; it only scales the result, and decides nothing.
function _sphere_sumsq(::_DenseLayout, x, f, γ, ::Type{U}, top) where {U}
  ss = zero(U)
  @simd for k in eachindex(IndexLinear(), x)
    ss += (_l1_margin(x, f, γ, U, k) / top)^2
  end
  ss
end
function _sphere_sumsq(L::_LowerLayout, x, f, γ, ::Type{U}, top) where {U}
  ss = zero(U)
  for k in _layout_indices(L, x)
    ss += _layout_mult(L, U, k) * (_l1_margin(x, f, γ, U, k) / top)^2
  end
  ss
end
function _sphere_l1_shrink!(y, x, f, γ, r, ::Type{U}, ::Type{T}, L) where {U, T}
  inds = _layout_indices(L, x)
  # Soft thresholding and the scaling onto the sphere in the type of the comparison, so
  # that the entries that survive are those whose margin was positive and nothing is
  # narrowed to T before the norm is taken. The norm is scaled by the largest margin.
  # x is read before y is written, so y may be x.
  top = zero(U)
  for k in inds
    top = max(top, _l1_margin(x, f, γ, U, k))
  end
  ss = _sphere_sumsq(L, x, f, γ, U, top)
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
    m = _l1_margin(x, f, γ, U, k)
    z = if iszero(m)
      zero(U)                           # not m * scale, which is 0 ⋅ Inf where scale overflows
    elseif direct
      m * scale
    else
      am, em = frexp(m)
      _ldexp_once((am / at) * aq, em + eq - et)
    end
    _layout_put!(L, y, k, copysign(convert(T, z), xi))
  end
  convert(T, _value_unchecked(f, y))
end

function _value_and_prox_unchecked!(y::SymmetricSparseIterate, c::SphereConstrained{<:NormL1}, x::AbstractArray, γ::Real)
  _checkstep_nonneg(γ)
  _checkwrite(value_and_prox!, c, y, x)
  T = float(eltype(y))
  U = promote_type(float(eltype(x)), T)       # the type in which the margins compare, as in the group sphere
  f, r = c.f, convert(T, c.sphere.r)
  p = size(y, 1)
  k, kr, kc, bestn, bestc = 0, 0, 0, zero(U), zero(U)
  nonfinite = false
  @inbounds for col in 1:p, row in col:p
    i = row + (col - 1) * p
    n = abs(convert(U, x[row, col]))
    nonfinite |= !isfinite(n)
    threshold = _weight_product(U, γ, f.λ0, _entry(f.w, i))   # not rounded to T first: selection and shrinkage agree
    better = if k == 0 || isinf(bestc)
      true
    elseif isinf(threshold) || bestn > bestc  # a positive margin: the shrink follows, and no winner is needed
      false
    elseif (row == col) == (kr == kc)         # the same multiplicity: compare the margins
      _margin_better(n, threshold, bestn, bestc)
    elseif row == col                         # a diagonal entry against a pair, whose margin counts √2 times
      _diagonal_pair_sign(n, threshold, bestn, bestc) > 0
    else
      _diagonal_pair_sign(bestn, bestc, n, threshold) < 0
    end
    better && ((k, kr, kc, bestn, bestc) = (i, row, col, n, threshold))
  end
  isfinite(bestc) || throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step, and give some entry a finite weight"))
  nonfinite && return _sphere_nonfinite!(y, T)
  if bestn > bestc
    return _sphere_l1_shrink!(y, x, f, γ, r, U, T, _LowerLayout(size(y, 1)))
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
  _checkstep_nonneg(γ)
  _checkwrite(value_and_prox!, c, y, x)
  T = float(eltype(y))
  U = promote_type(float(eltype(x)), T)       # the type in which the margins compare, as for a symmetric output
  f, r = c.f, convert(T, c.sphere.r)
  k, bestn, bestc = 0, zero(U), zero(U)
  nonfinite = false
  @inbounds for i in eachindex(IndexLinear(), x)
    n = abs(convert(U, x[i]))
    nonfinite |= !isfinite(n)
    threshold = _weight_product(U, γ, f.λ0, _entry(f.w, i))
    # margins n - t compared without rounding (see _margin_greater and _margin_better)
    (k == 0 || isinf(bestc) || (!isinf(threshold) && _margin_better(n, threshold, bestn, bestc))) && ((k, bestn, bestc) = (i, n, threshold))
  end
  isfinite(bestc) || throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step, and give some entry a finite weight"))
  nonfinite && return _sphere_nonfinite!(y, T)
  bestn > bestc && return _sphere_l1_shrink!(y, x, f, γ, r, U, T, _DenseLayout())
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
  _checkstep_nonneg(γ)
  _checkwrite(value_and_prox!, c, y, x)
  T = float(eltype(y))
  U = promote_type(float(eltype(x)), T)       # the type in which the shrinkage compares
  f, r = c.f, convert(T, c.sphere.r)
  kbest, ibest, bestn, bestc = 0, 0, zero(U), zero(U)
  for k in eachindex(f.groups)
    n = convert(U, _group_norm2(x, f.groups[k]))
    threshold = _group_sphere_threshold(f, γ, U, k)
    (kbest == 0 || isinf(bestc) || (!isinf(threshold) && n - bestn > threshold - bestc)) && ((kbest, bestn, bestc) = (k, n, threshold))
  end
  @inbounds for i in eachindex(IndexLinear(), x)
    _group(f, i) == 0 || continue
    n = convert(U, abs(x[i]))
    ((kbest == 0 && ibest == 0) || isinf(bestc) || n - bestn > -bestc) && ((kbest, ibest, bestn, bestc) = (0, i, n, zero(U)))
  end
  isfinite(bestc) || throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step"))
  if bestn > bestc
    v = convert(T, _value_and_prox_unchecked!(y, f, x, γ))
    iszero(maximum(abs, y)) || return _onto_sphere!(y, f, v, r)
    # A positive margin whose shrinkage rounds to zero in the destination has no stored direction.
    # Where U is wider, the shrinkage is formed again in U (every positive group keeps its share),
    # normalized there and stored once. Otherwise it takes the selection below, along x.
    _wide_sphere_direction!(y, c, x, γ, U, r) && return convert(T, _value_unchecked(f, y))
  end
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

# y = r z/‖z‖ for the shrinkage z of the groups in the type U, which is wider than the type of y;
# false, with y unchanged in meaning, where z is zero in U too. Any array y takes it: z is a
# buffer of the shape of y, written only on this exceptional path. U is wider than the type of y
# only for an argument of a wider type, which cannot alias y, so x is intact here. A norm of z that
# is not a normal number in the type of y takes the rescaled branch of _scale_l2!.
_wide_sphere_direction!(y, c, x, γ, ::Type{U}, r) where {U} = false
function _wide_sphere_direction!(y::AbstractArray{T}, c, x, γ, ::Type{U}, r) where {T, U}
  U === T && return false
  z = similar(y, U)
  _value_and_prox_unchecked!(z, c.f, x, γ)
  iszero(maximum(abs, z)) && return false
  _scale_l2!(y, z, r, convert(T, _norm2(z)))
  true
end

# On the sphere a point with at most s nonzero entries has ⟨z, x⟩ ≤ r‖x_S‖ on its
# support S, so the best support carries the most of x: the s largest entries
# (and the free diagonal), the proximal point of the ℓ0 ball.
value_and_prox!(y::AbstractArray, c::SphereConstrained{IndBallL0}, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::SphereConstrained{IndBallL0}, x::AbstractArray, γ::Real)
  _checkstep_nonneg(γ)
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

####################################
#  steps, checked before any block of a sum is written
####################################

# The ball takes the step of f; every step in [0, Inf] that f accepts is a step of the ball.
_checkstep(op::O, c::BallConstrained, y::AbstractArray, x::AbstractArray, γ::Real) where {O} =
  (_checkstep_nonneg(γ); _checkstep(op, c.f, y, x, γ))
_checkstep(op::O, c::SphereConstrained{IndBallL0}, y::AbstractArray, x::AbstractArray, γ::Real) where {O} =
  _checkstep_nonneg(γ)

# On the sphere the kernels reject a step for which every threshold is infinite: a zero level
# or one zero-penalty free entry (group) keeps γ = Inf a step of the projection onto the
# zero-penalty sphere. The thresholds are those of the kernels and read no values of x.
function _checkstep(op::O, c::SphereConstrained{<:NormL1}, y::AbstractArray, x::AbstractArray, γ::Real) where {O}
  _checkstep_nonneg(γ)
  isempty(x) && return true
  U = promote_type(float(eltype(x)), float(eltype(y)))
  _sphere_finite_threshold(c.f, γ, U, x) ||
    throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step, and give some entry a finite weight"))
  true
end
function _checkstep(op::O, c::SphereConstrained{<:GroupNormL2}, y::AbstractArray, x::AbstractArray, γ::Real) where {O}
  _checkstep_nonneg(γ)
  isempty(x) && return true
  U = promote_type(float(eltype(x)), float(eltype(y)))
  _sphere_finite_threshold(c.f, γ, U, x) ||
    throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step"))
  true
end
# The threshold of group k: γλ0w formed in the promotion of U and the factors' types and rounded
# once to U, the model precision of the call (numerics T3), as the group prox forms it. The kernel
# and its preflight both take it from here, so they agree.
@inline _group_sphere_threshold(f, γ, ::Type{U}, k) where {U} =
  _weight_product(U, γ, f.λ0, f.w[k])
# Plain loops that stop at the first finite threshold: an ordinary step ends at the first entry (group).
function _sphere_finite_threshold(f::NormL1, γ, ::Type{U}, x) where {U}
  for k in eachindex(IndexLinear(), x)
    isfinite(_weight_product(U, γ, f.λ0, _entry(f.w, k))) && return true
  end
  false
end
function _sphere_finite_threshold(f::GroupNormL2, γ, ::Type{U}, x) where {U}
  for k in eachindex(f.groups)
    isfinite(_group_sphere_threshold(f, γ, U, k)) && return true
  end
  for i in eachindex(IndexLinear(), x)      # an entry in no group is not penalized
    _group(f, i) == 0 && return true
  end
  false
end
