# Reusable numerical computation and recovery: soft thresholding, products with exponent
# scaling, compensated and power-of-two sums, values as significand/exponent parts, underflow
# recovery of signed sums, squared residuals and norms, and operator-norm estimates. General
# infrastructure (checks, buffers, stored-value scans, scaled-value protocol hooks, indexing) is
# in utils.jl, which is included first; function-specific triggers and formulas stay with the
# function that uses them.

####################################
#
#  Scalar operations
#
####################################

# without branches, so that broadcasting it vectorizes
"""
    shrink(v, c)

Soft thresholding: `sign(v) ⋅ max(|v| - c, 0)` for a threshold `c ≥ 0`, the
proximal operator of `c|⋅|`. A `NaN` in `v` propagates.
"""
shrink(v::T, c::T) where {T<:AbstractFloat} = max(v - c, zero(T)) + min(v + c, zero(T))
shrink(v::Real, c::Real) = shrink(promote(float(v), float(c))...)

# Nonnegative coefficients: keep the usual multiply order when it is well
# scaled. At exceptional scales, multiply bounded significands and apply the
# combined exponent once, so no intermediate subnormal loses precision. The
# exceptional path is a separate call. Only the intermediate needs to be
# normal: a subnormal or overflowing final multiply has no later factor to
# restore its range. With a normal intermediate, a zero third factor is safe.
# For scalar-step L1 proximal loops, ab is invariant: LLVM can unswitch this
# check and vectorize the ordinary loop without a per-weight recovery branch.
@inline function _positive_product(a::T, b::T, c::T) where {T<:AbstractFloat}
  ab = a * b
  p = ab * c
  floatmin(T) <= ab <= floatmax(T) && return p
  _exceptional_product(a, b, c)
end
# the cold path: abc in scaled units (_scaled_product), a zero factor giving +0
@noinline _exceptional_product(a::T, b::T, c::T) where {T<:AbstractFloat} = _scaled_product(a, b, c, 0)
@inline _positive_product(a::Real, b::Real, c::Real) = _positive_product(promote(float(a), float(b), float(c))...)

# Weights of separable functions: an infinite weight fixes the entry at zero,
# at every level and step, and a zero weight leaves it unpenalized at every
# level. These helpers keep the ordinary products of positive finite scales,
# so that loops over entries are unchanged.

# the coefficient s⋅w of an entry with the weight w ≥ 0 at the scale s ≥ 0 (a
# level, or a step times a level); the test of s does not depend on the entry.
# Otherwise s⋅w is NaN only as 0⋅Inf, where the weight decides: Inf for an
# infinite weight, 0 for a zero one.
@inline function _weight_coefficient(s, w)
  p = s * w
  (s > zero(s)) & isfinite(s) && return p
  ifelse(isnan(p) & !isnan(s), oftype(p, w), p)
end

# γλw as _positive_product, where an infinite weight gives Inf also when γλ is
# zero; only the exceptional path tests the weight
@inline function _weight_product(a::T, b::T, w::T) where {T<:AbstractFloat}
  ab = a * b
  p = ab * w
  floatmin(T) <= ab <= floatmax(T) && return p
  isinf(w) ? T(Inf) : _exceptional_product(a, b, w)
end
@inline _weight_product(a::Real, b::Real, w::Real) = _weight_product(promote(float(a), float(b), float(w))...)

# The same products rounded once to the destination type U, formed in the promotion of U and
# the factors' types (numerics T3), so that a product that overflows or underflows the factors'
# type is still finite in U. For R == U the promotion and the conversion are no-ops.
# A Bool factor (the weight `true` of an unweighted entry) does not widen the type.
@inline _factor_type(a) = float(typeof(a))
@inline _factor_type(::Bool) = Float16
@inline _working_type(::Type{U}, a, b) where {U<:AbstractFloat} = promote_type(U, _factor_type(a), _factor_type(b))
@inline _working_type(::Type{U}, a, b, c) where {U<:AbstractFloat} = promote_type(U, _factor_type(a), _factor_type(b), _factor_type(c))
@inline function _weight_product(::Type{U}, a, b, w) where {U<:AbstractFloat}
  R = _working_type(U, a, b, w)
  convert(U, _weight_product(convert(R, a), convert(R, b), convert(R, w)))
end
# The threshold γλw rounded once to S, and whether γλ, formed in the working type R, is in the normal range
# (the range test of _positive_product). The soft-thresholding loop of NormL1 and the ℓ2 shrinkage form their
# ordinary threshold with it.
@inline function _threshold_product(::Type{S}, γ, λ, w) where {S}
  R = _working_type(S, γ, λ, w)
  ab = convert(R, γ) * convert(R, λ)
  convert(S, ab * convert(R, w)), floatmin(R) <= ab <= floatmax(R)
end
# The coefficient and product of a KKT diagnostic in the gradient's element type T, rounded once. A
# gradient that is not floating point (integers) keeps the untyped value and type of the baseline.
@inline _kkt_coefficient(::Type{T}, λ, w) where {T<:AbstractFloat} = _weight_coefficient(T, λ, w)
@inline _kkt_coefficient(::Type, λ, w) = _weight_coefficient(λ, w)
@inline function _weight_coefficient(::Type{U}, λ, w) where {U<:AbstractFloat}
  R = _working_type(U, λ, w)
  convert(U, _weight_coefficient(convert(R, λ), convert(R, w)))
end

# the value p = c⋅φ of a term with the coefficient c and the unweighted value
# φ ≥ 0: zero where c⋅φ is 0⋅Inf (an infinite weight at a zero entry, a zero
# coefficient at an infinite entry); NaN arguments propagate
@inline _weighted_value(p, φ) = ifelse(isnan(p) & !isnan(φ), zero(p), p)

# w v 2^k for a weight w >= 0, with the power kept outside the floating-point range until the final
# rounding (the three-factor kernel below); a zero factor gives the signed zero w v.
@inline function _scaled_product(w::T, v::T, k::Integer) where {T<:AbstractFloat}
  (iszero(w) || iszero(v)) && return w * v
  _scaled_product(w, v, one(T), k)
end

# A converted parameter must retain normal precision on the ordinary path.
@inline _level_fits(::Type, ::Real) = true
# A level already stored in this precision needs no conversion check.
@inline _level_fits(::Type{T}, λ::T) where {T<:Union{Float32, Float64}} = true
@inline function _level_fits(::Type{T}, λ::Real)::Bool where {T<:AbstractFloat}
  iszero(λ) || !isfinite(λ) || floatmin(T) <= abs(convert(T, λ)) <= floatmax(T)
end

# The bounded-product kernel: a b c 2^k from three significands in [1/2, 1), whose product is a
# normal number, with all exponents and the power applied by one final ldexp, so that no
# intermediate overflows or underflows. A zero factor gives +0; a factor that is not finite gives
# the ordinary product. Every other exponent-scaled product above is a call of this one.
@inline function _scaled_product(a::Real, b::Real, c::Real, k::Integer)
  a, b, c = promote(float(a), float(b), float(c))
  (iszero(a) || iszero(b) || iszero(c)) && return zero(a)
  (isfinite(a) && isfinite(b) && isfinite(c)) || return (a * b) * c
  k = _product_power(typeof(a), k)
  ma, ea = _frexp(a)
  mb, eb = _frexp(b)
  mc, ec = _frexp(c)
  ldexp((ma * mb) * mc, _exponent_sum(typeof(a), ea, eb, ec, k))
end

# a b c 2^k for finite positive a and b and a scaled norm c (_l2_classified_norm), which is finite with
# 1 <= c < 2 sqrt(n): the significands of a and b are multiplied by c itself, so that c needs no frexp, and the
# product is in range. The result is that of _scaled_product, whose significand of c is c 2^-ec, scaled back.
@inline function _norm_product(a::R, b::R, c::R, k::Integer) where {R<:AbstractFloat}
  ma, ea = _frexp(a)
  mb, eb = _frexp(b)
  ldexp((ma * mb) * c, _exponent_sum(R, ea, eb, _product_power(R, k)))
end

# _scaled_product with the exponents summed in Int, for a power k that is the retained exponent of a
# _ProjectionScale (issue #767). That exponent is an Int that may already have wrapped; the Int sum is
# taken modulo 2^64 too, so it returns the right exponent whenever the final one fits in Int. Widening
# the wrapped k as if it were true would lose that.
@inline function _scaled_product_retained(w::T, v::T, k::Integer) where {T<:AbstractFloat}
  (iszero(w) || iszero(v)) && return w * v
  _scaled_product_retained(w, v, one(T), k)
end
@inline function _scaled_product_retained(a::Real, b::Real, c::Real, k::Integer)
  a, b, c = promote(float(a), float(b), float(c))
  (iszero(a) || iszero(b) || iszero(c)) && return zero(a)
  (isfinite(a) && isfinite(b) && isfinite(c)) || return (a * b) * c
  k = Int(k)
  ma, ea = _frexp(a)
  mb, eb = _frexp(b)
  mc, ec = _frexp(c)
  ldexp((ma * mb) * mc, ea + eb + ec + k)
end

@inline function _compensated_add(s, e, v)
  t = s + v
  e += abs(s) >= abs(v) ? (s - t) + v : (v - t) + s
  t, e
end

####################################
#
#  Power-of-two scaling and exponents
#
####################################

# Powers of two keep the units of a comparison without rounding the scale.
# Split a power that is outside the range of a single floating-point factor.
# 2^k for a k in the normal range of a Float32 or Float64, from its bit pattern
@inline _normal_power(::Type{Float64}, k::Int) = reinterpret(Float64, ((k + 1023) % UInt64) << 52)
@inline _normal_power(::Type{Float32}, k::Int) = reinterpret(Float32, ((k + 127) % UInt32) << 23)
@inline function _power_factors(::Type{T}, k::Integer) where {T<:Union{Float32, Float64}}
  k = Int(k)
  # k in the normal range: the single factor is exact, and the second one is one
  -exponent(floatmax(T)) + 1 <= k <= exponent(floatmax(T)) && return (_normal_power(T, k), one(T))
  _power_factors_split(T, k)
end
@inline function _power_factors(::Type{T}, k::Integer) where {T<:AbstractFloat}
  _power_factors_split(T, k)
end
@inline function _power_factors_split(::Type{T}, k::Integer) where {T<:AbstractFloat}
  k = _product_power(T, k)
  k1 = clamp(k, exponent(nextfloat(zero(T))), exponent(floatmax(T)))
  (ldexp(one(T), k1), ldexp(one(T), k - k1))
end
@inline _scale_value(v, s) = (float(v) * s[1]) * s[2]
@inline _magnitude_exponent(v) = iszero(v) ? 0 : Int(exponent(v))
# frexp with an Int exponent: frexp(::BigFloat) returns an Int32 exponent where a C long is 32 bits (Windows)
@inline function _frexp(x)
  m, e = frexp(x)
  (m, Int(e))
end
# MPFR exponents reach ±(2^62 - 1), so a sum of three of them leaves Int64. A BigFloat's exponent terms are widened to
# Int128 before the first addition; other types keep their Int sums.
@inline _exponent_sum(::Type{BigFloat}, es::Integer...) = foldl(+, map(Int128, es))
@inline _exponent_sum(::Type, es::Integer...) = +(es...)
# The power of a scaled product is narrowed to an Int, except for BigFloat, whose power may be an Int128 sum.
@inline _product_power(::Type, k::Integer) = Int(k)
@inline _product_power(::Type{BigFloat}, k::Integer) = k

# Exceptional values need the small terms even after a large one was added.
function _scaled_value_sum(x::AbstractArray, s, f::F = identity) where {F}
  v = e = zero(s[1])
  @inbounds for u in x
    v, e = _compensated_add(v, e, f(_scale_value(u, s)))
  end
  v + e
end

# Floating-point products also cover integer entries and integer coefficients.
function _scaled_dot(a, x::AbstractArray, s)
  _checkparam(a, x)
  v = zero(s[1])
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    v += float(_entry(a, i)) * _scale_value(x[i], s)
  end
  v
end

####################################
#
#  Values recomputed in power-of-two units (R1)
#
####################################

# 2^b bounds the magnitudes below m; a zero (or non-finite) m gives 0
@inline _bound_exponent(m) = (iszero(m) || !isfinite(m)) ? 0 : exponent(m) + 1
# ⌈log₂ n⌉, the bits that a sum of n terms adds
@inline _log2_ceil(n::Integer) = n <= 1 ? 0 : 8sizeof(Int) - leading_zeros(Int(n) - 1)
@inline _max_exponent(::Type{R}) where {R} = exponent(floatmax(R))

# w Σ mᵢ 2^eᵢ over finite mᵢ, for the terms of a value recomputed in different
# power-of-two units (R1). The terms are added from the largest exponent down
# to an accumulator kept as a significand in [1/2, 1) and its own exponent,
# renormalized after each addition, so terms that cancel do so before a
# smaller term is aligned to what remains; the weight is applied last, with
# one rounding. Non-finite terms give the ordinary sum. The weight is w ≥ 0
# (_scaled_product takes the sign from the sum).
function _pow2_weighted(::Type{R}, w, terms::Vararg{Tuple{Any, Int}, N}) where {R, N}
  ms = map(t -> convert(R, t[1]), terms)
  es = map(t -> t[2], terms)
  all(isfinite, ms) || return convert(R, w) * sum(ms; init = zero(R))
  # The frame exponents are Int128 for BigFloat, where the sum of an MPFR exponent and an Int exponent can leave Int64,
  # and Int for the other types.
  s, E = zero(R), _exponent_sum(R, 0)
  used = ntuple(_ -> false, Val(N))
  for _ in 1:N
    j, ej = 0, _exponent_sum(R, typemin(Int))
    for i in 1:N                                # the largest term not yet added
      (used[i] || iszero(ms[i])) && continue
      ei = _exponent_sum(R, exponent(ms[i]), es[i])
      ei > ej && ((j, ej) = (i, ei))
    end
    j == 0 && break
    used = Base.setindex(used, true, j)
    mj, fj = _frexp(ms[j])
    D = _exponent_sum(R, fj, es[j])
    if iszero(s)
      s, E = mj, D
    else              # the larger of the two sets the units (a remainder of a cancellation may be the smaller)
      s = D <= E ? s + ldexp(mj, D - E) : ldexp(s, E - D) + mj
      E = max(E, D)
      iszero(s) || ((s, f) = _frexp(s); E = _exponent_sum(R, E, f))
    end
  end
  iszero(s) && return convert(R, w) * zero(R)
  _scaled_product(convert(R, w), s, E)
end
# Two terms, the case of the dot with a shift d (LinearFunction), of the two sums of LeastSquaresLoss and of the
# group screen: the operations of the method above for N = 2 without the loops over the terms (the larger term
# first, the first one if their exponents are equal; the aligned addition and the renormalization; a zero term
# skipped; the final product), and, for the IEEE types only, the skip of a shift that cannot change the sum (below).
# The method above remains the definition. The two agree where each nonzero term has an exponent sum exponent(m) + e
# above typemin(Int) (taken in Int128 for BigFloat), the value from which the selection of the method above starts; the
# tests compare them there, also under directed rounding. At that sum the method above selects no term, and this method
# takes the single nonzero term, so the two differ there: for BigFloat (floatmin(BigFloat), -2^62) with a zero term, and
# for the IEEE types (T(0.75), typemin(Int) + 1) with a zero term and the weight T(Inf) (Inf from this method, NaN from
# the method above).
function _pow2_weighted(::Type{R}, w, t1::Tuple{Any, Int}, t2::Tuple{Any, Int}) where {R}
  m1, m2 = convert(R, t1[1]), convert(R, t2[1])
  (isfinite(m1) & isfinite(m2)) || return convert(R, w) * ((zero(R) + m1) + m2)
  e1, e2 = t1[2], t2[2]
  if iszero(m1) | iszero(m2)
    (iszero(m1) & iszero(m2)) && return convert(R, w) * zero(R)
    m, e = iszero(m1) ? (m2, e2) : (m1, e1)
    s, f = _frexp(m)
    return _scaled_product(convert(R, w), s, _exponent_sum(R, f, e))
  end
  # the frame exponents are Int128 for BigFloat and Int for the other types, as above
  E1 = _exponent_sum(R, exponent(m1), e1)
  E2 = _exponent_sum(R, exponent(m2), e2)
  if E2 > E1
    m1, m2, e1, e2 = m2, m1, e2, e1
  end
  s, f1 = _frexp(m1)
  E = _exponent_sum(R, f1, e1)
  mj, f2 = _frexp(m2)
  D = _exponent_sum(R, f2, e2)
  # For the IEEE types frexp is exact, so D <= E: the larger term is first. Such a term more than 2 precision(R) binades
  # below the sum is below half an ulp of s, which is in [1/2, 1) in magnitude, and the IEEE types round to nearest only,
  # so s + ldexp(mj, D - E) is s. The shift, which can round to a subnormal or to zero (slow), is skipped. Every other
  # type does the sum of the generic method: a BigFloat is rounded in the mode that is set, which a term below half an ulp
  # can still change, and the frexp of one above the current precision rounds, so D > E can occur there.
  if !(R <: Base.IEEEFloat && D - E < -2 * precision(R))
    s = D <= E ? s + ldexp(mj, D - E) : ldexp(s, E - D) + mj
    E = max(E, D)
  end
  iszero(s) && return convert(R, w) * zero(R)
  s, f = _frexp(s)
  _scaled_product(convert(R, w), s, _exponent_sum(R, E, f))
end
_pow2_sum(::Type{R}, terms::Vararg{Tuple{Any, Int}, N}) where {R, N} = _pow2_weighted(R, one(R), terms...)

####################################
#
#  Values as a significand and an exponent (R1, "Wrappers")
#
####################################

# The bound of `_zero_value_exponent` (utils.jl) for a zero value that comes from underflow:
# n losses of at most δ/2, each multiplied by less than 2^amp, with a margin of two binades for the
# relative errors of the ordinary sums: n δ 2^amp / 2 < 2^E
@inline _underflow_exponent(::Type{T}, n::Integer, amp::Integer = 0) where {T} =
  exponent(nextfloat(zero(T))) + _log2_ceil(n) + amp + 2

# e with |a| < 2^e, from the bits of a Float64 or Float32 (a subnormal gives the exponent of
# floatmin, an Inf or a NaN a large e): integer operations only, so that the bounds of
# `_zero_value_exponent` cost a few independent instructions. It is an upper bound only:
# 2^(e-1) <= |a| holds for a normal a. The
# bounds that use it from below (λ times the smallest weight) check λ with `_level_fits`, which a
# subnormal λ stored in T passes; that is harmless: its e is then exponent(floatmin) + 1, so the test
# e - 1 + exponent(min w) > exponent(floatmin) needs w >= 4 and gives c = λ w >= 4 δ, and the
# remaining terms of those bounds count the losses of c as for a normal c.
@inline _upper_exponent(a::Float64) = Int((reinterpret(UInt64, a) >> 52) & 0x7ff) - 1022
@inline _upper_exponent(a::Float32) = Int((reinterpret(UInt32, a) >> 23) & 0xff) - 126
@inline _upper_exponent(a::Real) = (b = abs(float(a)); iszero(b) ? -1000000 : isfinite(b) ? exponent(b) + 1 : 1000000)

# m 2^e with the significand normalized
@inline function _parts(m, e::Integer)
  fm, fe = frexp(m)
  (fm, Int(fe) + e)
end
# the product of two values given as parts, normalized
@inline function _parts_product(a::Tuple, b::Tuple)
  P = promote_type(typeof(a[1]), typeof(b[1]))
  _parts(convert(P, a[1]) * convert(P, b[1]), Int(a[2]) + Int(b[2]))
end

####################################
#
#  Values recomputed when an intermediate underflows (R1, #520)
#
####################################

# Only these types have an exponent range that the recovery reasons about.
_underflow_checked(::Type) = false
_underflow_checked(::Type{<:Union{Float32, Float64}}) = true

# A sum of terms m 2^e, each a significand m (|m| < 1) and an integer exponent, kept as the
# compensated sum (s + c) 2^ex of the terms in the frame ex of the largest exponent seen so far. A
# term in a larger frame rescales the sum (the part below 2^-1074 of the new frame is dropped,
# which is below eps times the scale of the sum); a term in a smaller frame is added after
# scaling it. Compensated additions keep a small term after larger ones cancel.
struct _ExpSum{R<:AbstractFloat}
  s::R
  c::R
  ex::Int
end
_expsum_start(::Type{R}) where {R<:AbstractFloat} = _ExpSum{R}(zero(R), zero(R), 0)
@inline function _expsum_add(acc::_ExpSum{R}, m::R, e::Integer) where {R}
  iszero(m) && return acc
  if iszero(acc.s) && iszero(acc.c)
    return _ExpSum{R}(m, zero(R), e)
  elseif e > acc.ex
    d = _frame_shift(R, acc.ex, e)
    s, c = _compensated_add(ldexp(acc.s, d), ldexp(acc.c, d), m)
    return _ExpSum{R}(s, c, e)
  end
  s, c = _compensated_add(acc.s, acc.c, ldexp(m, _frame_shift(R, e, acc.ex)))
  _ExpSum{R}(s, c, acc.ex)
end
# The shift a - b from one frame to another. The exponents of BigFloat terms reach ±(2^62 - 1), so two of them can
# differ by more than typemax(Int) (a wrapped shift would scale a negligible part up to Inf): the difference is taken
# in Int128 and clamped to Int, which loses nothing, as a shift beyond 2^63 binades already scales any significand to
# 0 (or Inf). Other types keep their Int difference.
@inline _frame_shift(::Type, a::Integer, b::Integer) = a - b
@inline _frame_shift(::Type{BigFloat}, a::Integer, b::Integer) =
  Int(clamp(Int128(a) - Int128(b), Int128(typemin(Int)), Int128(typemax(Int))))
# the term a b 2^k, with the product of the significands rounded once per factor
@inline function _expsum_product(acc::_ExpSum{R}, a, b, k::Int = 0) where {R}
  (iszero(a) || iszero(b)) && return acc
  ma, ea = _frexp(convert(R, a))
  mb, eb = _frexp(convert(R, b))
  _expsum_add(acc, ma * mb, ea + eb + k)
end
@inline function _expsum_term(acc::_ExpSum{R}, a, xi, mj::R, ej::Integer, k::Int) where {R}
  (iszero(a) || iszero(xi)) && return acc
  ma, ea = _frexp(convert(R, a))
  mi, ei = _frexp(convert(R, xi))
  _expsum_add(acc, (ma * mi) * mj, ea + ei + ej + k)
end

# Σ aᵢⱼ xᵢ xⱼ 2^k over the entries of A, with no product rounded into a narrower type: exact zeros are
# skipped, the stored entries only of a sparse or diagonal matrix are visited
function _expsum_quadratic(acc::_ExpSum{R}, A::AbstractMatrix, x::AbstractVector, k::Int) where {R}
  @inbounds for j in 1:size(A, 2)
    xj = x[j]
    iszero(xj) && continue
    mj, ej = _frexp(convert(R, xj))
    for i in 1:size(A, 1)
      acc = _expsum_term(acc, A[i, j], x[i], mj, ej, k)
    end
  end
  acc
end
function _expsum_quadratic(acc::_ExpSum{R}, A::SparseArrays.SparseMatrixCSC, x::AbstractVector, k::Int) where {R}
  rv, nz = rowvals(A), nonzeros(A)
  @inbounds for j in 1:size(A, 2)
    xj = x[j]
    iszero(xj) && continue
    mj, ej = _frexp(convert(R, xj))
    for p in nzrange(A, j)
      acc = _expsum_term(acc, nz[p], x[rv[p]], mj, ej, k)
    end
  end
  acc
end
function _expsum_quadratic(acc::_ExpSum{R}, A::Diagonal, x::AbstractVector, k::Int) where {R}
  d = A.diag
  @inbounds for i in eachindex(d)
    xi = x[i]
    iszero(xi) && continue
    mi, ei = _frexp(convert(R, xi))
    acc = _expsum_term(acc, d[i], xi, mi, ei, k)
  end
  acc
end
# Σ aᵢ xᵢ 2^k
function _expsum_dot(acc::_ExpSum, a::AbstractArray, x::AbstractArray, k::Int)
  _checksize(a, x)
  @inbounds for i in eachindex(IndexLinear(), a, x)
    acc = _expsum_product(acc, a[i], x[i], k)
  end
  acc
end
# The accumulated sum times a scale w >= 0 in R: w multiplies the significand and the exponent is applied
# with it, so that a sum outside the range of R (or of A) that w brings back is not rounded to 0 or Inf
# first. The product is formed by the three-factor kernel in A and converted to R; with A wider than R
# (Float64 for Float32 data) that conversion is the only rounding that matters. An exactly zero sum is +0
# at every w.
function _expsum_value(::Type{R}, acc::_ExpSum{A}, w = one(A)) where {R, A}
  m = acc.s + acc.c
  iszero(m) && return zero(R)
  convert(R, _scaled_product(convert(A, w), m, acc.ex))
end

# The smallest nonzero |x_i| as an exponent (the exponent of floatmax if there is none): a product of two
# entries with exponents e1 and e2 is at least 2^(e1+e2), so e1 + e2 ≥ exponent(floatmin) rules out its
# rounding into the subnormal range.
function _min_nonzero_exponent(x::AbstractArray{T}) where {T<:AbstractFloat}
  m = floatmax(T)
  @inbounds for v in _stored_values(x)
    a = abs(v)
    (iszero(a) || a >= m) || (m = a)
  end
  exponent(m)
end

# On the bit patterns, where the integer minimum vectorizes: with the sign cleared, b - 1 is the bit pattern
# of the magnitude minus one step, and for a zero it wraps to the largest integer, so that zeros are skipped
# without a select. An Inf or a NaN, whose patterns exceed that of floatmax, never replaces it.
function _min_nonzero_exponent(x::AbstractArray{T}) where {T<:Union{Float32, Float64}}
  U = _int_type(T)
  vals = _stored_values(x)
  m = typemax(U)
  @inbounds @simd for i in eachindex(vals)
    m = min(m, ((reinterpret(U, vals[i]) & typemax(U)) - one(U)) & typemax(U))
  end
  fm = reinterpret(U, floatmax(T))
  exponent(reinterpret(T, m == typemax(U) ? fm : min(m + one(U), fm)))
end
_min_nonzero_exponent(x::Diagonal) = _min_nonzero_exponent(x.diag)
_stored_minexp(x) = _min_nonzero_exponent(x)
_stored_minexp(x::_WrappedSparse) = _min_nonzero_exponent(parent(x))

# An absolute error that the rounding of one product into P cannot exceed is denorm_min/2 =
# eps floatmin/2. A quadratic form with n products per entry of A x then carries at most
# n² eps floatmin max|x| / 4 of it, the N products of the final dots at most N eps floatmin / 2 more,
# and it matters when it exceeds eps times |v|: that is, when |v| < floatmin (n² max|x| / 4 + N). The bound is sound for any finite data, needs only
# max|x| from the data, and does not fire at an exact zero of x. The threshold is kept in
# Float64 and is Inf when it overflows (the recovery then runs, which is safe).
@inline function _underflow_threshold(::Type{P}, κ, m, N = 0) where {P}
  # κ floatmin first: floatmin max|x| alone underflows for a small max|x| (a huge κ then needs it)
  (Float64(κ) * Float64(floatmin(P))) * Float64(m) + N * Float64(floatmin(P))
end

####################################
#
#  Norms
#
####################################

@inline _norm_difference(a, b) = a - b
@inline _norm_difference(a::Integer, b::Integer) = _wide_signed(a) - _wide_signed(b)
@inline _wide_signed(a::Integer) = signed(widen(a))
@inline _wide_signed(a::Bool) = Int(a)          # widen has no method for Bool

"""
    norm_diff(A, B, p = 2)

`‖A - B‖` without allocating `A - B`: the Euclidean (Frobenius) norm for
`p = 2`, the largest absolute entry for `p = Inf`. Both arrays must have
real entries.
"""
function norm_diff(A::AbstractArray, B::AbstractArray, p::Real = 2)
  eltype(A) <: Real && eltype(B) <: Real || _real_error(:norm_diff)
  size(A) == size(B) || throw(DimensionMismatch())
  T = float(promote_type(eltype(A), eltype(B)))
  v = zero(T)
  if p == 2
    return _norm_diff2(A, B)
  elseif p == Inf
    @inbounds for i in eachindex(A, B)
      v = max(v, abs(T(_norm_difference(A[i], B[i]))))
    end
    return v
  else
    throw(ArgumentError("p should be 2 or Inf"))
  end
end

# Preserve the inexpensive sum-of-squares path at ordinary scales. Where the sum
# overflows or falls below floatmin, the norm is recomputed from the squared
# residuals in power-of-two units (_scaled_norm), without a temporary array.
# Equal arrays also have a zero sum of squares; a vectorized sum of absolute
# differences confirms them without the recomputation.
function _norm_diff2(A::AbstractArray, B::AbstractArray)
  s = _sumabs2_diff(A, B)
  T = typeof(s)
  S = float(promote_type(eltype(A), eltype(B)))
  isfinite(s) && s >= floatmin(T) && return sqrt(s)
  if iszero(s)
    a = zero(T)
    @inbounds @simd for i in eachindex(A, B)
      d = S(_norm_difference(A[i], B[i]))
      a += abs(d)
    end
    iszero(a) && return s
  end
  _scaled_norm(A, B, s)
end

# scale ⋅ √(Σ μᵢ (xᵢ - yᵢ)²) for a scale ≥ 0 (one by default) and an ordinary sum of squares s in R that
# overflowed or fell below floatmin(R): the squared-residual kernel gives the sum as m 2^(2k), and the
# norm is scale √m 2^k, rounded once (twice where it is subnormal). An entry that is not finite keeps
# the ordinary result scale √s: Inf, or NaN where a NaN or an Inf - Inf residual occurs.
@noinline function _scaled_norm(x, y, s::R, scale::R = one(R)) where {R}
  q = _residual_squares(x, y, R)
  q === nothing && return scale * sqrt(s)
  m, k = q
  _scaled_product(scale, sqrt(m), k)
end

# Σᵢ (xᵢ - yᵢ)², without allocating x - y
function _sumabs2_diff(x::AbstractArray, y::AbstractArray)
  size(x) == size(y) || _difference_size_mismatch(size(x), size(y))
  T = float(promote_type(eltype(x), eltype(y)))
  s = zero(T)
  @inbounds @simd for i in eachindex(x, y)
    s += abs2(T(_norm_difference(x[i], y[i])))
  end
  s
end

@noinline _difference_size_mismatch(sx, sy) =
  throw(DimensionMismatch("the arrays have sizes $sx and $sy"))

# Count the terms actually reduced, rather than the ambient sparse dimension.
_square_term_count(x) = length(x)
_square_term_count(x::SparseArrays.AbstractSparseArray) = length(nonzeros(x))
_square_multiplicity(x, i) = 1

# Can an ordinary sum s of n squares in T be kept (R1)? Below n floatmin/eps a square may have
# underflowed and lost more than eps of the sum, and the scaled recomputation runs instead.
@inline _squares_normal(s, n, ::Type{T}) where {T} = s >= n * (floatmin(T) / eps(T))

@inline function _square_value(λ, x, y)
  T = _valtype(x)
  s = y === nothing ? _sumabs2(x) : _sumabs2_diff(x, y)
  v = convert(T, λ) * s / 2
  n = y === nothing ? _square_term_count(x) : length(x)
  if (isfinite(v) & _squares_normal(real(s), n, T)) && _level_fits(T, λ)
    return _value_at(v, x)
  end
  _exceptional_square_value(λ, x, y, real(s), v)
end

@noinline function _exceptional_square_value(λ, x, y, s, v)
  T = _valtype(x)
  !isfinite(λ) && return _value_at(v, x)
  iszero(λ) && isfinite(s) && return _value_at(v, x)
  _level_fits(T, λ) && iszero(s) && y === nothing && x isa DenseArray && iszero(sum(abs, x)) && return _value_at(v, x)
  _scaled_square_value(λ, x, y, v)
end

@inline _square_difference(a, b, ::Type{R}) where {R} = R(a) - R(b)
@inline _square_difference(a::Integer, b::Integer, ::Type{R}) where {R} = R(_norm_difference(a, b))

# The residual a - b in R and the binade shift h of its units: formed before any scaling, so that
# cancellation keeps its units (h = 0); where the difference of finite a and b overflows it is
# formed as a/2 - b/2 instead (h = 1), which is finite and loses at most the last bit of a
# subnormal operand. Shared by the squared residuals below and the ridge gradient.
@inline function _halved_difference(a, b, ::Type{R}) where {R}
  d = _square_difference(a, b, R)
  isfinite(d) ? (d, 0) : (_halved(a, b, R), 1)
end
@inline _halved(a, b, ::Type{R}) where {R} = R(a) / 2 - R(b) / 2

# The exponent of the largest residual, or a code that no exponent equals: _EXPONENT_ZERO for an exactly
# zero residual, and for an entry that is not finite _EXPONENT_NONFINITE, which does not tell NaN from
# Inf (the callers with a second operand only need to know that the residual is not finite). The method
# for y = 0 below tells them apart: _EXPONENT_NAN (the same value as _EXPONENT_NONFINITE, the largest, so
# that a NaN wins the maximum) for a NaN entry and _EXPONENT_INF for an infinite one.
const _EXPONENT_ZERO = typemin(Int)
const _EXPONENT_NONFINITE = typemax(Int)
const _EXPONENT_NAN = typemax(Int)
const _EXPONENT_INF = typemax(Int) - 1
@inline function _residual_exponent(a, b, ::Type{R}) where {R}
  (isfinite(a) && isfinite(b)) || return _EXPONENT_NONFINITE
  d, h = _halved_difference(a, b, R)
  iszero(d) ? _EXPONENT_ZERO : exponent(abs(d)) + h
end

# the residual scaled by 2^-k (s) or, where it was halved, by 2^(1-k) (h). The loop over the entries
# branches on the difference itself: the pair (d, h) of _halved_difference would become a select,
# which computes the rare halved case and both scalings for every entry (about 1.5x slower).
@inline function _scaled_residual(a, b, s, h, ::Type{R}) where {R}
  d = _square_difference(a, b, R)
  isfinite(d) ? _scale_value(d, s) : _scale_value(_halved(a, b, R), h)
end

function _square_exponent(x, y, ::Type{R}) where {R}
  k = _EXPONENT_ZERO
  @inbounds for i in eachindex(x, y)
    k = max(k, _residual_exponent(x[i], y[i], R))
  end
  k
end
# y = 0, with R at least as wide as the entries: the exponent is that of the entry itself, converted to R, and no
# difference is formed. The first NaN decides (the NaN code is the largest, so the maximum would give it anyway),
# an infinite entry gives the Inf code unless a NaN follows, and zero entries are skipped. The largest exponent and
# the zero code are those of the method above for entries whose precision is at most that of the arithmetic of R
# (the IEEE types, and a BigFloat at its current precision). A BigFloat entry above the current precision gives the
# exponent of the entry (as the _l2_scan of the code before the consolidation did), which can differ by one from that of
# a - 0 rounded to the current precision: it is one less where that rounding carries into the next binade (v = 1 - 2^-258
# stored at 768 bits, used at 256 bits).
@inline function _square_exponent(x, ::Nothing, ::Type{R}) where {R}
  k = _EXPONENT_ZERO
  inf = false
  for v in _stored_values(x)
    a = convert(R, v)
    if isfinite(a)
      iszero(a) || (k = max(k, Int(exponent(a))))
    else
      isnan(a) && return _EXPONENT_NAN
      inf = true
    end
  end
  inf ? _EXPONENT_INF : k
end

# For Float32 and Float64 entries the same result from the largest magnitude, taken on the bit patterns where it
# vectorizes (_maximum_abs): a NaN has the largest pattern, then an Inf, and the exponent of the largest finite
# entry is the largest exponent. No entry is classified, converted or its exponent taken, but one.
@inline function _square_exponent(x::AbstractArray{<:Union{Float32, Float64}}, ::Nothing, ::Type{R}) where {R}
  m = _maximum_abs(x)
  isnan(m) && return _EXPONENT_NAN
  isinf(m) && return _EXPONENT_INF
  iszero(m) && return _EXPONENT_ZERO
  Int(exponent(convert(R, m)))
end

function _square_scaled_sum(x, y, k, ::Type{R}) where {R}
  s, h = _power_factors(R, -k), _power_factors(R, 1-k)
  v = e = zero(R)
  @inbounds for i in eachindex(x, y)
    v, e = _compensated_add(v, e, abs2(_scaled_residual(x[i], y[i], s, h, R)))
  end
  v + e
end
# y = 0 with a finite exponent k, R at least as wide as the entries: the residual a - 0 of a finite a is
# a in R and is not halved, so the entry is scaled by the one power factor 2^-k
@inline function _square_scaled_sum(x, ::Nothing, k, ::Type{R}) where {R}
  s = _power_factors(R, -k)
  v = e = zero(R)
  @inbounds for (i, a) in enumerate(_stored_values(x))
    q = _square_multiplicity(x, i) * abs2(_scale_value(convert(R, a), s))
    v, e = _compensated_add(v, e, q)
  end
  v + e
end

# The squared-residual kernel (R1): Σ μᵢ (xᵢ - yᵢ)² = s 2^(2k) in R, so that the half sum is
# s 2^(2k-1). y === nothing stands for y = 0 and visits the stored entries of x only, each with its
# logical multiplicity μᵢ (2 for an off-diagonal entry of a symmetric iterate). Every residual is
# formed before it is scaled by 2^-k, k the exponent of the largest one, and the squares are summed
# compensated, so that no square overflows and a small residual after a cancellation is kept. It
# gives (zero(R), 0) when every residual is exactly zero and `nothing` when a residual is not finite
# (the ordinary result then stands: a genuine Inf or NaN is not an intermediate failure).
function _residual_squares(x, y, ::Type{R}) where {R}
  k = _square_exponent(x, y, R)
  k >= _EXPONENT_INF && return nothing
  k == _EXPONENT_ZERO && return (zero(R), 0)
  (_square_scaled_sum(x, y, k, R), k)
end
# the type of the recomputation of λ/2 Σ (xᵢ - yᵢ)²
_square_type(λ, x, y) = promote_type(_valtype(x), y === nothing ? _valtype(x) : _valtype(y), float(typeof(λ)))

@noinline function _scaled_square_value(λ, x, y, ordinary)
  R = _square_type(λ, x, y)
  q = _residual_squares(x, y, R)
  q === nothing && return _value_at(ordinary, x)
  s, k = q
  _value_at(_scaled_product(convert(R, λ), s, _exponent_sum(R, k, k, -1)), x)
end

# The k >= 0 for which n products of entries below 2^bound(ma) and 2^bound(mb), scaled by 2^-k, sum
# below 2^(emax - margin) in R: the units of the recomputation of an overflowing dot (R1).
@inline _product_sum_shift(::Type{R}, ma, mb, n, margin) where {R} =
  max(0, _bound_exponent(ma) + _bound_exponent(mb) + _log2_ceil(n) + margin - _max_exponent(R))

# Σ (xᵢ - yᵢ)² λ/2 as a significand and an exponent (the scaled-value mechanism), from the
# recomputation above; a non-finite entry or level gives `nothing`
@noinline function _square_parts(λ, x, y)
  (iszero(λ) || !isfinite(λ)) && return nothing
  R = _square_type(λ, x, y)
  q = _residual_squares(x, y, R)
  q === nothing && return nothing
  s, k = q
  iszero(s) && return (zero(R), 0)
  _parts_product(_frexp(convert(R, λ)), _parts(s, 2k - 1))
end

####################################
#
#  The ℓ2 norm of an index set (NormL2, GroupNormL2)
#
####################################

# The kernels below compute, for an index set G of an array x (all of x for NormL2, one group for
# GroupNormL2), the shrinkage max(1 - γλw/‖x[G]‖₂, 0) x[G] and the term λw‖x[G]‖₂ of a value. They take
# the level λ and the weight w as scalars (w = true for NormL2). They allocate nothing when the data, the destination
# and the level, weight and step are all Float32 or Float64 (a BigFloat, for example, allocates in its own arithmetic).

# The vector of the entries x[G[j]], j = 1:length(G), read through x and G without a copy, in the element
# type of x or, if one is given, rounded to it. A view would reshape a matrix x, which allocates; the callers have
# checked that G is within x, so the reads are not checked again (the index j is, unless @inbounds).
# _stored_values gives all of its entries and _square_multiplicity gives 1, as for any vector.
struct _IndexedValues{T, X, I} <: AbstractVector{T}
  x::X
  G::I
end
_IndexedValues(x, G) = _IndexedValues(x, G, eltype(x))
# with an element type T, the entries are convert(T, x[G[j]]): the entries as the loops in T read them
_IndexedValues(x, G, ::Type{T}) where {T} = _IndexedValues{T, typeof(x), typeof(G)}(x, G)
Base.size(v::_IndexedValues) = (length(v.G),)
Base.length(v::_IndexedValues) = length(v.G)
Base.IndexStyle(::Type{<:_IndexedValues}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(v::_IndexedValues{T}, j::Int) where {T}
  @boundscheck checkbounds(v, j)
  @inbounds convert(T, v.x[v.G[j]])
end
@inline function Base.iterate(v::_IndexedValues{T}, j::Int = 1) where {T}
  j > length(v.G) && return nothing
  @inbounds (convert(T, v.x[v.G[j]]), j + 1)
end

# Whether every entry of x[G], rounded to T, is a zero of either sign: an exactly zero group has the norm 0 also where its
# sum of squares underflows. A group that is zero only after the rounding (x wider than T) counts as zero, as in the
# sums of squares in T.
@inline _l2_all_zero(x, G, ::Type{T}) where {T} = _all_zero(_IndexedValues(x, G, T))

# Σ x[i]² over i ∈ G, each entry converted to T before it is squared, in one vectorizable loop
@inline function _indexed_sumabs2(x, G, ::Type{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in G
    s += abs2(convert(T, x[i]))
  end
  s
end

# the type of the cold recomputations: at least Float64, so that a compensated sum of Float32 squares
# does not stall as a Float32 one does over more than 2^24 terms
_recovery_type(types...) = promote_type(Float64, types...)

# Can the ordinary sum s of n nonnegative terms be kept? NormL1 (n entries) and GroupNormL2 (n groups)
# test their ordinary values with it; the argument below is stated for NormL1.
# Products that round below floatmin (#373). The value is a sum of n terms λ0 wᵢ |xᵢ| ≥ 0, each rounded
# once or twice (the coefficient, then the product with |xᵢ|); a coefficient out of the normal range is
# formed by value_entry with one rounding, so the ordinary flag and these exact products leave only the
# rounding of a term into the subnormal range (or to zero). That rounding loses at most δ/2 =
# eps(T) floatmin(T)/2 per term, n eps floatmin / 2 in all, which is at most eps/2 of a sum s ≥ n floatmin.
# A sum below that bound (a zero sum included) may have lost more than ordinary rounding, even a normal one:
# 512 terms 2^-135 + 2^-150 in Float32 each round to 2^-135 and the sum loses 512 2^-150 = 2^-141, 256 units of 2^-149
# (the spacing of the numbers below floatmin(Float32)). Sums of
# subnormal partial sums are exact, and the relative errors of normal terms and of the summation are those
# of every ordinary value. The test is false for NaN, so an ordinary NaN (or Inf) is returned as it was.
@inline _sum_kept(s, n) = !(s < n * floatmin(typeof(s)))

# Codes returned by the shrinkage kernel in place of the norm of the stored group: the term of the group is zero
# (the stored group is exactly zero, or the group is unpenalized), or its value must be recomputed from the stored entries.
const _L2_ZERO = -2
const _L2_RECOMPUTE = -1

@inline function _l2_zero!(y, G)
  S = eltype(y)
  @inbounds for i in G
    y[i] = zero(S)
  end
  _L2_ZERO
end
@inline function _l2_copy!(y, x, G)
  if y !== x
    @inbounds for i in G
      y[i] = x[i]
    end
  end
  _L2_RECOMPUTE
end
@inline function _l2_fill!(y, G, v)
  @inbounds for i in G
    y[i] = v
  end
  _L2_RECOMPUTE
end
@inline function _l2_scale!(y, x, G, t::T) where {T}
  S = eltype(y)
  @inbounds @simd for i in G
    y[i] = convert(S, t * convert(T, x[i]))
  end
  nothing
end

# ‖x[G]‖₂ = ns 2^k from the classified exponent pass over the entries (the method of _square_exponent for
# y = 0), for the cold paths. k is the largest exponent of the entries, or the code of the group: _EXPONENT_NAN if
# an entry is NaN, _EXPONENT_INF if one is infinite and none is NaN, _EXPONENT_ZERO if all are zero. For a finite
# nonzero group and need_norm, ns is the norm of the entries scaled by 2^-k, so that its largest entry is in
# [1, 2) and no square overflows or underflows below the rounding of the sum: ns is in [1, 2√|G|). Otherwise
# ns is zero. A caller that does not need the norm passes need_norm = false, and then there is no second pass.
@noinline function _l2_classified_norm(x, G, ::Type{R}, need_norm::Bool = true) where {R}
  v = _IndexedValues(x, G)
  k = _square_exponent(v, nothing, R)
  (need_norm & (_EXPONENT_ZERO < k < _EXPONENT_INF)) || return (k, zero(R))
  (k, sqrt(_square_scaled_sum(v, nothing, k, R)))
end

# y[G] = prox of γλw‖⋅‖₂ at x[G], possibly in place (y === x), where G indexes both arrays. Returns, in
# T, the norm t‖x[G]‖ of the stored group on the ordinary shrinking path (each stored entry is t x[i]
# rounded once, so this is the norm of the stored entries to within a few rounding errors, without a
# second pass), or one of the codes above. The ordinary path needs a normal threshold c = γλw (formed
# by _threshold_product) and a normal sum of squares that no underflowing square can have
# changed by more than ordinary rounding; everything else, exceptional parameters and values included,
# is decided by the cold path, which does not run on that path.
@inline function _l2_shrink!(y, x, G, λ, w, γ::Real)
  T = float(eltype(y))
  c, normal = _threshold_product(T, γ, λ, w)
  if normal & (floatmin(T) <= c <= floatmax(T))
    s = _indexed_sumabs2(x, G, T)
    if isfinite(s) & _squares_normal(s, length(G), T)
      n = sqrt(s)
      if n > c
        t = one(T) - c / n
        _l2_scale!(y, x, G, t)
        return t * n
      end
      return T(_l2_zero!(y, G))
    elseif iszero(s) && _l2_all_zero(x, G, T)
      return T(_l2_zero!(y, G))
    end
  end
  # an unpenalized group (its threshold is 0, below the normal range) is copied without the call, and its term is zero
  iszero(w) && (_l2_copy!(y, x, G); return T(_L2_ZERO))
  T(_l2_shrink_cold!(y, x, G, λ, w, γ))
end

# The parameters decide first, in this order: an unpenalized group (w = 0, copied by _l2_shrink! itself), a group fixed at zero (an
# infinite weight, or an infinite level at every nonnegative step, #665), the identity (a zero level or
# step), a group fixed by an infinite step. Otherwise the values decide (below). The values are in a
# function of their own, so that the exits decided by the parameters do not carry its frame.
@noinline function _l2_shrink_cold!(y, x, G, λ, w, γ::Real)
  (isinf(w) || isinf(λ)) && return _l2_zero!(y, G)
  (iszero(λ) || iszero(γ)) && return _l2_copy!(y, x, G)
  isinf(γ) && return _l2_zero!(y, G)
  _l2_shrink_values!(y, x, G, λ, w, γ)
end
# The values, at positive finite γ, λ and w: a NaN entry makes every output NaN; an infinite entry (and
# no NaN) makes the norm infinite and the threshold finite, so 1 - c/‖x‖ = 1 and the group is copied, also
# when c overflows in floating point; a zero group gives +0. Otherwise r = c/‖x‖ = (γλw 2^-k) / ns from
# the scaled norm ns 2^k, with c formed by the bounded-product kernel, so that neither c nor the norm needs
# to be in range: r ≥ 1 gives zero, and t = 1 - r in (0, 1] scales the group without overflow. A product
# γλw 2^-k that overflows means r > 1, and one that underflows means r < floatmin, where t rounds to 1.
@noinline function _l2_shrink_values!(y, x, G, λ, w, γ::Real)
  T = float(eltype(y))
  R = _recovery_type(_working_type(T, γ, λ, w), float(eltype(x)))
  k, ns = _l2_classified_norm(x, G, R)
  k == _EXPONENT_NAN && return _l2_fill!(y, G, convert(eltype(y), NaN))
  k == _EXPONENT_INF && return _l2_copy!(y, x, G)
  k == _EXPONENT_ZERO && return _l2_zero!(y, G)
  r = _scaled_product(convert(R, γ), convert(R, λ), convert(R, w), -k) / ns
  r >= one(R) && return _l2_zero!(y, G)
  t = one(R) - r
  S = eltype(y)
  @inbounds for i in G
    y[i] = convert(S, t * convert(R, x[i]))
  end
  _L2_RECOMPUTE
end

# The ordinary paths return a value only below floatmax(T)/2 (a NaN fails the test too). A larger value
# is formed again in power-of-two units, as m 2^E from compensated sums accumulated in at least Float64
# and a few rounded products, and rounded once to T. Within a few eps of the overflow threshold
# floatmax + ulp/2 this does not decide on which side the exact value lies: the result may then be
# floatmax or Inf (#904).
@inline _l2_ordinary(v, ::Type{T}) where {T} = v < floatmax(T) / 2

# The term λw‖x[G]‖₂ of a value in T, given the norm s of x[G] from the shrinkage kernel, or one of the
# codes above (s = _L2_RECOMPUTE reads x). The ordinary path needs a normal λw, a value below
# floatmax(T)/2 and, when it reads x, a normal sum of squares; a larger value is recomputed from x[G].
@inline function _l2_term(λ, w, x, G, s, ::Type{T}) where {T}
  iszero(w) && return zero(T)
  s == _L2_ZERO && return zero(T)
  R = _working_type(T, λ, w)
  λw = convert(R, λ) * convert(R, w)
  if floatmin(R) <= λw <= floatmax(R)
    if s == _L2_RECOMPUTE
      q = _indexed_sumabs2(x, G, T)
      if isfinite(q) & _squares_normal(q, length(G), T)
        v = convert(T, λw * sqrt(convert(R, q)))
        _l2_ordinary(v, T) && return v
      elseif iszero(q) && _l2_all_zero(x, G, T)
        return zero(T)
      end
    else
      v = convert(T, λw * convert(R, s))
      _l2_ordinary(v, T) && return v
    end
  end
  _l2_term_cold(λ, w, x, G, T)
end

# The term of the fused value_and_prox!, after the shrinkage kernel: the stored group's norm s ≥ 0 and the
# zero group inline, as _l2_term computes them; the codes of the cold shrinkage and out-of-range values
# call _l2_term out of line, which keeps its sums out of the loop over the groups.
@inline function _l2_fused_term(λ, w, x, G, s, ::Type{T}) where {T}
  s == _L2_ZERO && return zero(T)
  if s >= 0
    R = _working_type(T, λ, w)
    λw = convert(R, λ) * convert(R, w)
    v = convert(T, λw * convert(R, s))
    ((floatmin(R) <= λw <= floatmax(R)) & _l2_ordinary(v, T)) && return v
  end
  _l2_term_outofline(λ, w, x, G, s, T)
end
@noinline _l2_term_outofline(λ, w, x, G, s, ::Type{T}) where {T} = _l2_term(λ, w, x, G, s, T)

# Exceptional terms, in this order: a zero level with a finite weight gives 0 whatever x holds; a NaN
# entry gives NaN and an infinite one Inf; the zero group gives 0; an infinite weight or level gives Inf
# (decided by the class of x, without the norm). A positive finite level and weight give λw ns 2^k from
# the scaled norm, in one rounding to T (_scaled_product).
@noinline function _l2_term_cold(λ, w, x, G, ::Type{T}) where {T}
  iszero(λ) && !isinf(w) && return zero(T)
  R = _recovery_type(T, float(typeof(λ)), float(typeof(w)), float(eltype(x)))
  unbounded = isinf(w) || isinf(λ)
  k, ns = _l2_classified_norm(x, G, R, !unbounded)
  k == _EXPONENT_NAN && return T(NaN)
  k == _EXPONENT_INF && return T(Inf)
  k == _EXPONENT_ZERO && return zero(T)
  unbounded && return T(Inf)
  convert(T, _norm_product(convert(R, λ), convert(R, w), ns, k))
end

# λ‖x‖₂ in O(stored entries): _sumabs2 and the squared-residual kernel visit the stored values of the
# sparse and symmetric iterates only (each off-diagonal symmetric coordinate twice). The ordinary value
# needs a level that fits T, a finite product and a normal sum of squares; the exactly zero argument
# returns zero at every level (the domain of an infinite level is {0}).
function _l2_value(λ, x)
  T = _valtype(x)
  iszero(λ) && return zero(T)
  s = _sumabs2(x)
  if _level_fits(T, λ)
    v = convert(T, convert(T, λ) * sqrt(s))
    (_l2_ordinary(v, T) & _squares_normal(s, _square_term_count(x), T)) && return v
  end
  iszero(s) && _all_zero(_stored_values(x)) && return zero(T)
  _l2_value_cold(λ, x, T)
end

# The norm from the classified exponent pass and the scaled sum (the squared-residual kernel for y = 0), times the
# level, rounded once to T. A NaN entry gives T(NaN) (not a propagated NaN), and an infinite one Inf; an infinite level
# gives +0 where every entry is zero, otherwise Inf. A level of zero was decided by the caller.
@noinline function _l2_value_cold(λ, x, ::Type{T}) where {T}
  R = _recovery_type(T, float(typeof(λ)))
  k = _square_exponent(x, nothing, R)
  k == _EXPONENT_NAN && return T(NaN)
  k == _EXPONENT_ZERO && return zero(T)
  (isinf(λ) || k == _EXPONENT_INF) && return T(Inf)
  convert(T, _scaled_product(convert(R, λ), sqrt(_square_scaled_sum(x, nothing, k, R)), k))
end

# The term w‖x[G]‖₂ of a penalized group, added to the accumulator of a value recomputed with the terms'
# exponents (_ExpSum). A zero weight and a zero group add nothing, and a NaN or ±Inf entry, or an infinite
# weight at a nonzero group, make the ordinary sum infinite or NaN: nothing is returned for them, and the
# caller keeps that sum. Inlined into the loop over the groups, of which it is the whole body.
@inline function _l2_expsum_add(acc::_ExpSum{R}, x, G, w) where {R}
  iszero(w) && return acc
  k, ns = _l2_classified_norm(x, G, R, !isinf(w))
  k >= _EXPONENT_INF && return nothing
  k == _EXPONENT_ZERO && return acc
  isinf(w) && return nothing
  _expsum_product(acc, w, ns, k)
end

# ‖g[G]‖₂ - λw, with the coefficient of the NormL1 screen (an infinite weight or level gives Inf, a zero
# weight 0) in the floating type of g
@inline function _l2_screen(λ, w, g, G)
  T = float(eltype(g))
  c = _kkt_coefficient(T, λ, w)
  s = _indexed_sumabs2(g, G, T)
  (isfinite(s) & _squares_normal(s, length(G), T) & isfinite(c)) && return sqrt(s) - c
  _l2_screen_cold(λ, w, g, G, T, c)
end

# A norm or a coefficient out of range: the difference ns 2^k - mλ mw 2^(eλ+ew) of the scaled norm and the
# coefficient is formed in the recovery type, in the units of the larger exponent, and
# converted to T. Coefficients that are zero or infinite by the parameters keep c.
@noinline function _l2_screen_cold(λ, w, g, G, ::Type{T}, c) where {T}
  R = _recovery_type(T, float(typeof(λ)), float(eltype(g)))
  k, ns = _l2_classified_norm(g, G, R)
  k == _EXPONENT_NAN && return T(NaN)
  k == _EXPONENT_INF && return convert(T, T(Inf) - c)
  k == _EXPONENT_ZERO && return -c
  if iszero(λ) || iszero(w) || isinf(λ) || isinf(w)
    isinf(c) && return T(-Inf)
    return convert(T, _scaled_product(ns, one(R), k)) - c
  end
  # Both terms are bounded (ns < 2 sqrt(n), mλ mw < 1) and nonzero, so the difference needs only the common frame of the larger
  # exponent: the smaller term is scaled down into it (below its precision, or exactly), the difference is exact
  # where the terms are close, and the final scaling rounds once.
  mλ, eλ = _frexp(convert(R, λ))
  mw, ew = _frexp(convert(R, w))
  ek, ec = _exponent_sum(R, k), _exponent_sum(R, eλ, ew)
  E = max(ek, ec)
  d = ldexp(ns, _frame_shift(R, ek, E)) - ldexp(mλ * mw, _frame_shift(R, ec, E))
  convert(T, ldexp(d, E))
end

####################################
#
#  The largest eigenvalue of an operator
#
####################################

# The largest eigenvalue of a symmetric positive semidefinite n × n operator,
# given as op!(w, v) for w = Av, by the Lanczos process. The largest Ritz value
# increases to it; the process stops when the residual of the largest Ritz pair
# is below rtol times the value, and then the value is an eigenvalue to that
# accuracy. Without reorthogonalization the Ritz values stay within the
# spectrum (Paige, 1980), so the estimate does not exceed the largest
# eigenvalue. The start is a fixed vector with no simple structure, which is
# orthogonal to the leading eigenvector only by accident.
function _largest_eigenvalue(op!, n::Integer, ::Type{T}; rtol::Real = 1e-10, maxiter::Integer = 500) where {T<:AbstractFloat}
  n == 0 && return zero(T)
  v = T[sin(1.0 * i) + cos(sqrt(2.0) * i) for i in 1:n]
  v ./= norm(v)
  vold, w = zeros(T, n), zeros(T, n)
  α, β = T[], T[]
  θ = zero(T)
  for j in 1:min(n, maxiter)
    op!(w, v)
    push!(α, dot(v, w))
    w .-= α[j] .* v
    j > 1 && (w .-= β[j-1] .* vold)
    E = eigen(SymTridiagonal(α, β), j:j)                 # the largest Ritz pair
    θ = E.values[1]
    b = norm(w)
    b * abs(E.vectors[j, 1]) <= rtol * abs(θ) && return θ
    iszero(b) && return θ
    push!(β, b)
    vold, v = v, vold
    v .= w ./ b
  end
  θ
end

# ‖X‖₂², the largest eigenvalue of X' Diagonal(w) X, through products with X
# and X' only, on the smaller of the two Gram matrices
"""
    opnorm2(X)

Estimate the squared spectral norm `‖X‖₂²` by Lanczos iteration, using only
products with `X` and its adjoint, without forming a Gram matrix or densifying
a sparse matrix. The result has floating element type matching `X`.

This is a numerical estimate, not a certified upper bound: Lanczos can miss
an eigenspace orthogonal to its deterministic starting vector. Algorithms
requiring a guaranteed bound should use an independently known bound (for
example `opnorm(X, 1) * opnorm(X, Inf)`) or a suitable line search.
"""
opnorm2(X::AbstractMatrix{<:Real}) = _opnorm2(X)

function _opnorm2(X::AbstractMatrix, w = nothing)
  T = float(eltype(X))
  m, n = size(X)
  if w === nothing && m < n
    t = zeros(T, n)
    return _largest_eigenvalue(m, T) do out, v
      mul!(t, X', v)
      mul!(out, X, t)
    end
  end
  t = zeros(T, m)
  _largest_eigenvalue(n, T) do out, v
    mul!(t, X, v)
    w === nothing || (t .*= w)
    mul!(out, X', t)
  end
end

# Bounds used by fixed-step methods must account for every eigenspace. Dense
# matrices use a full spectral calculation; sparse and other indexed matrices
# use ‖A‖₂ ≤ √(‖A‖₁ ‖A‖∞), without forming a Gram matrix or densifying A.
function _opnorm_upper(A::AbstractMatrix)
  T = float(eltype(A))
  isempty(A) && return zero(T)
  sqrt(convert(T, opnorm(A, 1))) * sqrt(convert(T, opnorm(A, Inf)))
end
function _opnorm_upper(A::Union{StridedMatrix, Symmetric{<:Real, <:StridedMatrix}, Hermitian{<:Real, <:StridedMatrix}})
  isempty(A) ? zero(float(eltype(A))) : opnorm(A)
end
_opnorm_upper(A::Union{Transpose, Adjoint}) = _opnorm_upper(parent(A))
_opnorm2_upper(A::AbstractMatrix) = abs2(_opnorm_upper(A))
function _opnorm2_upper(A::AbstractMatrix, w)
  w === nothing && return _opnorm2_upper(A)
  convert(float(eltype(A)), _opnorm2_upper(Diagonal(sqrt.(w)) * A))
end

