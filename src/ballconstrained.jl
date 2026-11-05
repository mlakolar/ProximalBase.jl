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

For a `NormL1` `f`, an argument with a NaN or infinite entry has these limits
(decision 48). An entry with an infinite weight is hard-fixed: its output is zero
whatever the argument. A NaN free entry, or two infinite free entries, give NaN in
every free output, since their limit depends on how the argument diverges. One
infinite free entry gives ±r there and zero elsewhere, the limit along finite
arguments; in a `SymmetricSparseIterate` a mirrored off-diagonal pair is one
entry, with limit ±r/√2 in each of its two entries. At `γ = Inf` the penalized
entries are zero and project nothing, so only the free entries are governed. A NaN
penalized entry at `γ = Inf` is a fixed endpoint, and leaves the free outputs as
they are. An infinite penalized entry at `γ = Inf` is a joint limit in the argument
and the step: with `x = (M, 0)`, `γ = M/2` gives the limit `(1, 0)` and `γ = 2M`
gives `(0, 0)`, so the limit has no unique value. The output is then NaN in every entry
that is not hard-fixed, the penalized ones included (and the value is NaN). A sphere with no entry of finite threshold throws an
`ArgumentError` before anything is written.

For a `GroupNormL2` `f`, the same limits hold group by group. An entry of a group
with an infinite weight is hard-fixed: its output is zero whatever the argument
holds, at every level and step, and a zero level does not remove it. A free entry
(in no group, in a group of zero weight, or with a zero level or step) and a
penalized entry at a finite step are governed: they determine the output. One
infinite governed entry gives `±r` there and zero elsewhere, and in a
`SymmetricSparseIterate` a mirrored off-diagonal pair is one entry, with `±r/√2`
in each of its two stored entries. Two or more infinite governed entries, in one
group or in several, and a NaN governed entry, give NaN in every governed output,
since their direction depends on how they diverge. At `γ = Inf` the penalized
entries are zero and a NaN there is ignored; an infinite penalized entry is a
joint limit, so the output is NaN in every entry that is not hard-fixed. The
value is that of the stored output, and NaN when the output is NaN.

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
  _ball_projection!(y, c, x, γ)
end
# NormL1 on the ball. An argument with a NaN or infinite entry takes the exceptional path. When y may be x,
# the argument is classified before anything is written. Otherwise x is intact, and the projection
# stores a nonfinite norm exactly when some entry of x is NaN or infinite in a coordinate that is not
# hard-fixed (a hard-fixed one may store NaN too, which only sends it to the exact path): the projection
# then takes the exceptional path, which reads x. A finite argument takes the ordinary path, unchanged.
function _value_and_prox_unchecked!(y::AbstractArray, c::BallConstrained{<:NormL1}, x::AbstractArray, γ::Real)
  _checkstep_nonneg(γ)
  _checkwrite(value_and_prox!, c, y, x)
  _l1_ball_host(c, y, x) || return _ball_projection!(y, c, x, γ)
  if Base.mightalias(y, x)
    _has_nonfinite(x) && return _l1_nonfinite!(y, c, x, γ)
    return _ball_projection!(y, c, x, γ)
  end
  v, ny = _ball_core!(y, c, x, γ)
  isfinite(ny) ? v : _l1_nonfinite!(y, c, x, γ)
end
# GroupNormL2 on the ball (#469, Stage 3B). A finite argument takes the group kernel, unchanged. An argument with
# a NaN or infinite entry is classified before anything is written. The test is a reduction over the argument (its
# stored entries for a sparse iterate): the group kernel gives NaN for a NaN in a penalized group, and the NaN limits
# are decided here before anything is written; an infinite penalized entry at the infinite step gives a zero output.
function _value_and_prox_unchecked!(y::AbstractArray, c::BallConstrained{<:GroupNormL2}, x::AbstractArray, γ::Real)
  _checkstep_nonneg(γ)
  _checkwrite(value_and_prox!, c, y, x)
  _group_host(c, y, x) || return _ball_projection!(y, c, x, γ)
  if Base.mightalias(y, x)
    _has_nonfinite(x) && return _group_nonfinite!(y, c, x, γ)
    return _ball_projection!(y, c, x, γ)
  end
  # A reduction over the grouped span sees a NaN in a penalized group before any output is written.
  # An infinite or NaN entry outside the span is copied to the output and reaches its norm, which is checked below.
  _group_span_nonfinite(x, c.f) && return _group_nonfinite!(y, c, x, γ)
  # A finite argument stored in a narrower output can overflow there before the scaling (1e100 in Float32): a
  # nonfinite stored norm then takes the wide projection of the surrogate, which is the finite one. The norm of
  # the stored point before scaling is nonfinite also for a NaN or infinite entry outside the group span.
  v, ny = _ball_core!(y, c, x, γ)
  isfinite(ny) ? v : _group_nonfinite!(y, c, x, γ)
end
# the projection in y, its value and the norm of the stored point before scaling (nonfinite exactly when
# the argument has a nonfinite entry in a coordinate that is not hard-fixed, for a NormL1: see above)
function _ball_core!(y::AbstractArray, c::BallConstrained, x::AbstractArray, γ::Real)
  T = float(eltype(y))
  v = _value_and_inner_prox!(y, c.f, x, γ, T)
  ny = convert(T, _norm2(y))
  r = convert(T, c.ball.r)
  if ny > r
    t = _scale_l2!(y, y, r, ny)
    v = isfinite(v) && ny >= floatmin(T) && isfinite(t) && t >= floatmin(T) ? v * t :
        (_isindicator(c.f) ? zero(T) : convert(T, _value_unchecked(c.f, y)))
  end
  v, ny
end
_ball_projection!(y::AbstractArray, c::BallConstrained, x::AbstractArray, γ::Real) = first(_ball_core!(y, c, x, γ))

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
  nonfinite && return _l1_sphere_nonfinite(y, c, x, γ)
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
  nonfinite && return _l1_sphere_nonfinite(y, c, x, γ)
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
  _sphere_group!(y, c, x, γ, true)
end
# The sphere's kernel. With classify, a nonfinite norm (of an entry, or of a group, which can overflow from finite
# entries too) sends the argument to the exact classification, which then runs this kernel with classify off on a
# finite surrogate: an overflowing finite norm is then handled by the scaling of this kernel, not classified again.
function _sphere_group!(y::AbstractArray, c::SphereConstrained{<:GroupNormL2}, x::AbstractArray, γ::Real, classify::Bool)
  T = float(eltype(y))
  U = promote_type(float(eltype(x)), T)       # the type in which the shrinkage compares
  f, r = c.f, convert(T, c.sphere.r)
  kbest, ibest, bestn, bestc = 0, 0, zero(U), zero(U)
  # the loops read every entry, so the check costs nothing extra
  nonfinite = false
  for k in eachindex(f.groups)
    n = convert(U, _group_norm2(x, f.groups[k]))
    nonfinite |= !isfinite(n)
    threshold = _group_sphere_threshold(f, γ, U, k)
    (kbest == 0 || isinf(bestc) || (!isinf(threshold) && n - bestn > threshold - bestc)) && ((kbest, bestn, bestc) = (k, n, threshold))
  end
  @inbounds for i in eachindex(IndexLinear(), x)
    _group(f, i) == 0 || continue
    n = convert(U, abs(x[i]))
    ((kbest == 0 && ibest == 0) || isinf(bestc) || n - bestn > -bestc) && ((kbest, ibest, bestn, bestc) = (0, i, n, zero(U)))
  end
  # the entries in no group are not classified by the loops above, and a check inside the loop costs more than a
  # vectorized reduction over the argument (the sum is nonfinite exactly for a NaN or infinite entry, and a finite
  # sum that overflows only sends the argument to the exact classification, which resolves it)
  nonfinite |= (!isempty(f.free) || length(x) > length(f.group_of)) && _has_nonfinite(x)
  isfinite(bestc) || throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step"))
  classify && nonfinite && _group_host(c, y, x) && return _group_nonfinite!(y, c, x, γ)
  if bestn > bestc
    v = convert(T, _value_and_prox_unchecked!(y, f, x, γ))
    m = maximum(abs, y)
    # A shrinkage stored as zero has no direction, and one stored in the subnormal range has too
    # few bits to give one. Where U is wider, the shrinkage is formed again in U (every positive
    # group keeps its share), normalized there and stored once. Otherwise a zero takes the
    # selection below, along x.
    (U === T ? !iszero(m) : !(m < floatmin(T))) && return _onto_sphere!(y, f, v, r)
    U !== T && _wide_sphere_direction!(y, c, x, γ, U, r) && return convert(T, _value_unchecked(f, y))
    iszero(m) || return _onto_sphere!(y, f, v, r)
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
    _group_direction!(y, x, G, r, nG)
    for i in eachindex(IndexLinear(), y)
      _group(f, i) == kbest || (y[i] = zero(eltype(y)))
    end
  end
  convert(T, _group_value(f, f.λ0, f.w[kbest], r, y, G, T))
end

# y = r z/‖z‖ for the shrinkage z of the groups in the type U, which is wider than the type of y;
# false where z is zero in U too. Any array y takes it: z is a buffer of the shape of y, written
# only on this exceptional path. U is wider than the type of y only for an argument of a wider
# type, which cannot alias y, so x is intact here. A norm of z that is not a normal number in the
# type of y takes the rescaled branch of _scale_l2!.
function _wide_sphere_direction!(y::AbstractArray{T}, c, x, γ, ::Type{U}, r) where {T, U}
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
#  NormL1 on a ball or sphere at NaN and infinite arguments (#469, Stage 3A)
####################################

# The coordinates of the output that the projection of a NormL1 governs. A coordinate with an infinite
# weight is hard-fixed: outside the domain, its output is zero whatever its argument holds (E2). Any
# other coordinate has the threshold γλ0w: zero for a free coordinate (a zero weight, a zero level or a
# zero step), positive for a penalized one, and infinite for a penalized one at γ = Inf. At γ = Inf the
# penalized coordinates are zero at every finite argument, so they do not govern their outputs, and an
# infinite one among them is a joint limit in x and γ, which has no unique value (E3).
const _HARD_L1 = 0
const _FREE_L1 = 1
const _PENALIZED_L1 = 2

@inline function _l1_class(f::NormL1, γ, ::Type{U}, k) where {U}
  isinf(_entry(f.w, k)) && return _HARD_L1
  iszero(_weight_product(U, γ, f.λ0, _entry(f.w, k))) ? _FREE_L1 : _PENALIZED_L1
end
@inline _governed_l1(cls, γ) = cls == _FREE_L1 || (cls == _PENALIZED_L1 && !isinf(γ))

@inline _layout_get(::_DenseLayout, x, k) = x[k]
@inline _layout_get(L::_LowerLayout, x, k) = x[mod1(k, L.p), cld(k, L.p)]
_layout_zero!(::_DenseLayout, xs, k) = (xs[k] = zero(eltype(xs)); xs)
function _layout_zero!(L::_LowerLayout, xs, k)
  row, col = mod1(k, L.p), cld(k, L.p)
  xs[row, col] = xs[col, row] = zero(eltype(xs))
  xs
end

# What x holds over the governed coordinates: NaN entries, infinite entries (the first one, with its sign)
# and whether an infinite penalized coordinate at γ = Inf makes the joint limit ambiguous. `finite` says
# that some coordinate that is not hard-fixed has a finite threshold, so that the sphere is feasible.
struct _L1Reading
  nan::Int
  inf::Int
  first::Int
  sign::Int
  joint::Bool
  finite::Bool
end

function _l1_reading(L, x, f::NormL1, γ, ::Type{U}) where {U}
  nan = inf = first = sgn = 0
  joint = finite = false
  for k in _layout_indices(L, x)
    cls = _l1_class(f, γ, U, k)
    cls == _HARD_L1 && continue
    finite |= isfinite(_weight_product(U, γ, f.λ0, _entry(f.w, k)))
    v = _layout_get(L, x, k)
    if _governed_l1(cls, γ)
      if isnan(v)
        nan += 1
      elseif isinf(v)
        inf += 1
        inf == 1 && ((first, sgn) = (k, Int(sign(v))))
      end
    else
      joint |= isinf(v)
    end
  end
  _L1Reading(nan, inf, first, sgn, joint, finite)
end

# A copy of x, symmetric as x is, in which every coordinate that the projection does not govern is zero:
# the finite kernel then projects the governed coordinates alone (their thresholds and norms do not
# involve the zeroed ones, whose outputs are zero).
function _l1_surrogate(L, x, f::NormL1, γ, ::Type{U}) where {U}
  xs = Array{float(eltype(x))}(undef, size(x))
  for k in eachindex(IndexLinear(), x)
    xs[k] = x[k]
  end
  for k in _layout_indices(L, x)
    _governed_l1(_l1_class(f, γ, U, k), γ) && continue
    _layout_zero!(L, xs, k)
  end
  xs
end

# The output that takes NaN on the governed coordinates (every coordinate that is not hard-fixed when
# joint, the joint limit) and zero elsewhere.
function _l1_write_nan!(y, L, f::NormL1, γ, ::Type{U}, joint::Bool) where {U}
  fill!(y, zero(eltype(y)))
  nanv = convert(eltype(y), NaN)
  for k in _layout_indices(L, y)
    cls = _l1_class(f, γ, U, k)
    cls == _HARD_L1 && continue
    (joint || _governed_l1(cls, γ)) && _layout_put!(L, y, k, nanv)
  end
  y
end

# The output at a single infinite governed coordinate: ±r there, zero elsewhere. A mirrored pair of a
# SymmetricSparseIterate is one coordinate counted twice in the norm (M2), so each entry of the pair is
# ±r/√2.
function _l1_write_unit!(y, L::_LowerLayout, rd::_L1Reading, r)
  fill!(y, zero(eltype(y)))
  k = rd.first
  row, col = mod1(k, L.p), cld(k, L.p)
  scale = row == col ? one(r) : sqrt(oftype(r, 2))
  _layout_put!(L, y, k, convert(eltype(y), rd.sign * r / scale))
  y
end
function _l1_write_unit!(y, L::_DenseLayout, rd::_L1Reading, r)
  fill!(y, zero(eltype(y)))
  _layout_put!(L, y, rd.first, convert(eltype(y), rd.sign * r))
  y
end

# The projection at an argument with a NaN or infinite entry (E2, E3, M2). The argument is read before
# anything is written, so y may be x. Nothing is classified from a narrowed copy: the entries are those of
# the argument, after the symmetrization that the caller applied.
#  - a joint infinite penalized coordinate at γ = Inf: NaN over every coordinate that is not hard-fixed;
#  - a NaN governed coordinate, or two or more infinite ones: NaN over the governed coordinates;
#  - one infinite governed coordinate: ±r there (±r/√2 in a mirrored pair), zero elsewhere;
#  - otherwise the finite projection of the governed coordinates, which is the finite kernel's, run on a copy.
# A sphere with no coordinate of finite threshold is infeasible and throws before anything is written.
function _l1_nonfinite!(y::AbstractArray, c, x::AbstractArray, γ::Real)
  T = float(eltype(y))
  U = promote_type(float(eltype(x)), T)
  f = c.f
  sphere = c isa SphereConstrained
  r = convert(T, sphere ? c.sphere.r : c.ball.r)
  L = y isa SymmetricSparseIterate ? _LowerLayout(size(y, 1)) : _DenseLayout()
  if !sphere && iszero(r)                      # the ball of radius zero is the singleton {0}
    fill!(y, zero(eltype(y)))
    return zero(T)
  end
  rd = _l1_reading(L, x, f, γ, U)
  sphere && !rd.finite &&
    throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step, and give some entry a finite weight"))
  if isinf(γ) && rd.joint
    _l1_write_nan!(y, L, f, γ, U, true)
  elseif rd.nan > 0 || rd.inf > 1
    _l1_write_nan!(y, L, f, γ, U, false)
  elseif rd.inf == 1
    _l1_write_unit!(y, L, rd, r)
  else
    return _finite_l1!(y, c, _l1_surrogate(L, x, f, γ, U), γ)
  end
  # O2: the value at the stored point, which is NaN wherever the stored point is: a zero weight
  # would otherwise make the NaN entry a zero contribution to the value
  any(isnan, y) ? convert(T, NaN) : convert(T, _value_unchecked(f, y))
end

# The exceptional path is taken only for a host argument with a finite radius; a device array keeps the
# finite kernels (its exceptional coordinates are not classified yet: #366).
# A sum of a NaN or infinite entry is not finite, and a sum of finite entries that overflows only sends the
# argument to the exact classifier, which then runs the finite kernel as before. The sum is vectorized,
# where a scan with a Boolean accumulator is not.
_has_nonfinite(x::AbstractArray) = !isfinite(sum(x))
# a sparse iterate has zeros outside its stored values, so only those are read (O(nnz), not O(length))
_has_nonfinite(x::Union{SparseIterate, SymmetricSparseIterate}) = !isfinite(sum(view(x.nzval, 1:x.nnz)))
_l1_ball_host(c::BallConstrained, y, x) = isfinite(c.ball.r) && !_is_device_array(x) && !_is_device_array(y)
# the finite projection of a surrogate, which is finite by construction: the kernel itself, not its dispatch.
# A ball whose output is narrower than the argument can overflow in the output type before the scaling
# (a finite 1e100 stored in Float32): then the same kernel runs in the wider type and the result is narrowed.
_finite_l1!(y, c::BallConstrained, x, γ) = _finite_ball!(y, c, x, γ, promote_type(float(eltype(x)), float(eltype(y))))
_finite_l1!(y, c::SphereConstrained, x, γ) = _value_and_prox_unchecked!(y, c, x, γ)
function _finite_ball!(y, c, x, γ, ::Type{U}) where {U}
  T = float(eltype(y))
  U === T && return _ball_projection!(y, c, x, γ)
  b = similar(y, U)
  _ball_core!(b, c, x, γ)
  _narrow_into!(y, b)
  # O2: the value at the stored output, not at the wider intermediate point, which can have a
  # larger value (a coordinate that underflows to zero in y) or be infinite in T
  any(isnan, y) ? convert(T, NaN) : convert(T, _value_unchecked(c.f, y))
end
_narrow_into!(y::SymmetricSparseIterate, b) = (p = size(y, 1); for c in 1:p, r in c:p
    y[r, c] = convert(eltype(y), b[r, c])
  end; y)
_narrow_into!(y::AbstractArray, b) = (for k in eachindex(IndexLinear(), y)
    y[k] = convert(eltype(y), b[k])
  end; y)
_l1_sphere_nonfinite(y, c, x, γ) =
  _is_device_array(x) || _is_device_array(y) ? _sphere_nonfinite!(y, float(eltype(y))) : _l1_nonfinite!(y, c, x, γ)

####################################
#  GroupNormL2 on a ball or sphere at NaN and infinite arguments (#469, Stage 3B)
####################################

# The exceptional path is taken only for a host array (a device array keeps the finite kernels, #366), and only
# for a ball of finite radius.
_group_host(c::BallConstrained, y, x) = isfinite(c.ball.r) && !_is_device_array(x) && !_is_device_array(y)
_group_host(c::SphereConstrained, y, x) = !_is_device_array(x) && !_is_device_array(y)

# Whether an entry of the span of the groups (the first length(group_of) positions, which contain every grouped
# index) is NaN or infinite: a sum of the span is nonfinite exactly then (an infinity cannot cancel to a finite sum),
# and a finite sum that overflows is a false alarm that the classification resolves. A sparse iterate is read in
# its stored values, and any other array is read in full.
_group_span_nonfinite(x::Union{SparseIterate, SymmetricSparseIterate}, f) = _has_nonfinite(x)
function _group_span_nonfinite(x::DenseArray, f::GroupNormL2)
  m = min(length(f.group_of), length(x))
  s = zero(float(eltype(x)))
  @inbounds @simd for i in 1:m
    s += x[i]
  end
  !isfinite(s)
end
_group_span_nonfinite(x::AbstractArray, f) = _has_nonfinite(x)

# The output classes of the independent coordinates of a GroupNormL2 f at the step γ (E2, E3, E4). A coordinate
# of a hard-fixed group, an infinite weight at every level and step, is zero whatever its argument holds. A free
# one (in no group, in a group of zero weight, or with a zero level or step) and a penalized one at a finite step
# are governed: their outputs depend on the whole argument. A penalized one at the infinite step is zero at every
# finite argument, so it governs nothing, and an infinite one among them is a joint limit in x and γ (E3).
const _HARD_G = 0
const _FREE_G = 1
const _PENALIZED_G = 2

@inline function _group_class(f::GroupNormL2, γ, k::Integer)
  g = _group(f, k)
  g == 0 && return _FREE_G
  w = f.w[g]
  isinf(w) && return _HARD_G
  (iszero(f.λ0) || iszero(w) || iszero(γ)) ? _FREE_G : _PENALIZED_G
end
@inline _governed_group(cls, γ) = cls == _FREE_G || (cls == _PENALIZED_G && !isinf(γ))

# What x holds over the governed coordinates: the number of NaN entries, of infinite entries, the first infinite
# one (its index and sign), and whether an infinite penalized coordinate at the infinite step makes the joint limit
# undetermined. Each independent coordinate counts once: a mirrored pair is one (M2), so one pair is one infinity,
# while two infinite coordinates in one group, or in two groups, give no determined direction (E3).
struct _GroupReading
  nan::Int
  inf::Int
  k1::Int
  sign1::Int
  joint::Bool
end

function _group_reading(L, x, f::GroupNormL2, γ)
  nan = inf = k1 = sgn = 0
  joint = false
  for k in _layout_indices(L, x)
    cls = _group_class(f, γ, k)
    cls == _HARD_G && continue
    v = _layout_get(L, x, k)
    if _governed_group(cls, γ)
      if isnan(v)
        nan += 1
      elseif isinf(v)
        inf += 1
        inf == 1 && ((k1, sgn) = (k, Int(sign(v))))
      end
    else
      joint |= isinf(v)
    end
  end
  _GroupReading(nan, inf, k1, sgn, joint)
end

# A copy of x, symmetric as x is, in which every coordinate that the projection does not govern is zero: the
# finite kernel then projects the governed coordinates alone. A hard-fixed or non-governed output is zero in any
# case, and its argument does not enter the governed projection.
function _group_surrogate(L, x, f::GroupNormL2, γ)
  xs = Array{float(eltype(x))}(undef, size(x))
  for k in eachindex(IndexLinear(), x)
    xs[k] = x[k]
  end
  for k in _layout_indices(L, x)
    _governed_group(_group_class(f, γ, k), γ) && continue
    _layout_zero!(L, xs, k)
  end
  xs
end

# NaN on the governed coordinates (every coordinate that is not hard-fixed at the joint limit), zero elsewhere
function _group_write_nan!(y, L, f::GroupNormL2, γ, joint::Bool)
  fill!(y, zero(eltype(y)))
  nanv = convert(eltype(y), NaN)
  for k in _layout_indices(L, y)
    cls = _group_class(f, γ, k)
    cls == _HARD_G && continue
    (joint || _governed_group(cls, γ)) && _layout_put!(L, y, k, nanv)
  end
  y
end

# ±r at the one infinite governed coordinate, zero elsewhere. A mirrored pair of a SymmetricSparseIterate is
# one coordinate counted twice in the norm (M2), so each of its two entries is ±r/√2.
function _group_write_unit!(y, L, rd::_GroupReading, r)
  fill!(y, zero(eltype(y)))
  k = rd.k1
  _layout_put!(L, y, k, convert(eltype(y), rd.sign1 * r / sqrt(_layout_mult(L, eltype(y), k))))
  y
end

# NaN in the output, read without a scan of a sparse iterate's logical length (only its stored values)
_has_nan_output(y::AbstractArray) = any(isnan, y)
_has_nan_output(y::Union{SparseIterate, SymmetricSparseIterate}) = any(isnan, view(y.nzval, 1:y.nnz))

# The projection at an argument with a NaN or infinite entry (E1, E2, E3, M2). The argument is classified before
# anything is written, so y may be x. Nothing is read from the stored output except to write it.
#  - a joint limit (an infinite penalized coordinate at γ = Inf): NaN over every coordinate that is not hard-fixed;
#  - a governed NaN, or two or more infinite governed coordinates: NaN over the governed coordinates, where the
#    penalized outputs at the infinite step are zero (they are fixed by the endpoint, whatever x holds);
#  - one infinite governed coordinate: ±r there (±r/√2 in a mirrored pair), zero elsewhere;
#  - otherwise the finite projection of the governed coordinates, which is the finite kernel's, run on a copy.
# A sphere with no coordinate of finite threshold is infeasible and throws before anything is written.
function _group_nonfinite!(y::AbstractArray, c, x::AbstractArray, γ::Real)
  T = float(eltype(y))
  U = promote_type(float(eltype(x)), T)
  f = c.f
  sphere = c isa SphereConstrained
  r = convert(T, sphere ? c.sphere.r : c.ball.r)
  L = y isa SymmetricSparseIterate ? _LowerLayout(size(y, 1)) : _DenseLayout()
  # the ball of radius zero is the singleton {0} (E2); the test is on the radius itself, not on its rounding to T
  if !sphere && iszero(c.ball.r)
    fill!(y, zero(eltype(y)))
    return zero(T)
  end
  rd = _group_reading(L, x, f, γ)
  sphere && !_sphere_finite_threshold(f, γ, U, x) &&
    throw(ArgumentError("sphere thresholds must be finite; rescale the penalty or step"))
  if isinf(γ) && rd.joint
    _group_write_nan!(y, L, f, γ, true)
  elseif rd.nan > 0 || rd.inf > 1
    _group_write_nan!(y, L, f, γ, false)
  elseif rd.inf == 1
    _group_write_unit!(y, L, rd, r)
  else
    return _finite_group!(y, c, _group_surrogate(L, x, f, γ), γ)
  end
  # O2: the value at the stored point, which is NaN wherever the stored point is (a zero weight would otherwise
  # make a NaN entry a zero contribution to the value)
  _has_nan_output(y) ? convert(T, NaN) : convert(T, _value_unchecked(f, y))
end

# the finite projection of a surrogate, finite by construction: the ball's group kernel at the model precision,
# and the sphere's kernel, which reads the surrogate and writes y
_finite_group!(y, c::BallConstrained, x, γ) = _finite_ball!(y, c, x, γ, promote_type(float(eltype(x)), float(eltype(y))))
_finite_group!(y, c::SphereConstrained, x, γ) = _sphere_group!(y, c, x, γ, false)

# y[G] ← r x[G]/‖x_G‖ for a group G of a sphere's shrinkage, scaled as _scale_l2!. A SymmetricSparseIterate stores each
# mirrored pair once, so its group is written through the lower positions only (the pair is one stored value).
_group_direction!(y::AbstractArray, x, G, r, nG) = _scale_l2!(view(y, G), view(x, G), r, nG)
function _group_direction!(y::SymmetricSparseIterate, x, G, r, nG)
  W = _group_writes(y, G)
  t = r / nG
  if isfinite(nG) && nG >= floatmin(typeof(nG)) && isfinite(t) && t >= floatmin(typeof(t))
    for i in W
      y[i] = t * x[i]
    end
  else
    m = _maxabs(x, G)
    n = sqrt(sum(abs2, x[i] / m for i in G))
    for i in W
      y[i] = (x[i] / m) * (r / n)
    end
  end
  y
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
