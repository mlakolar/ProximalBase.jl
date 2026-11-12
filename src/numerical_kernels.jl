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
@inline _weight_coefficient(γ, λ, w) = _weight_coefficient(γ * λ, w)

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
# The coefficient and product of a KKT diagnostic in the gradient's element type T, rounded once. A
# gradient that is not floating point (integers) keeps the untyped value and type of the baseline.
@inline _kkt_coefficient(::Type{T}, λ, w) where {T<:AbstractFloat} = _weight_coefficient(T, λ, w)
@inline _kkt_coefficient(::Type, λ, w) = _weight_coefficient(λ, w)
@inline function _weight_coefficient(::Type{U}, λ, w) where {U<:AbstractFloat}
  R = _working_type(U, λ, w)
  convert(U, _weight_coefficient(convert(R, λ), convert(R, w)))
end
@inline function _weight_coefficient(::Type{U}, γ, λ, w) where {U<:AbstractFloat}
  R = _working_type(U, γ, λ, w)
  convert(U, _weight_coefficient3(convert(R, γ), convert(R, λ), convert(R, w)))
end
# γλw in R. While γλ is a normal number it is the ordinary coefficient. Otherwise, for finite
# nonzero factors, the product is formed in scaled units, so that a weight restores a γλ that
# overflows or underflows in R (R3); zero, infinite and NaN factors keep the meaning of
# _weight_coefficient (a zero level is 0·Inf only where the weight is infinite).
@inline function _weight_coefficient3(γ::R, λ::R, w::R) where {R<:AbstractFloat}
  s = γ * λ
  floatmin(R) <= s <= floatmax(R) && return _weight_coefficient(s, w)
  _recoverable_factors(γ, λ, w) ? _scaled_coefficient(γ, λ, w, s * w) : _weight_coefficient(s, w)
end
# The product of finite nonzero factors, correctly rounded once (to nearest, ties to even) in R, also
# where it is subnormal and whatever the Julia version (ldexp truncates into the subnormal range up
# to Julia 1.10). The significands are integers of `precision(R)` bits, so their product is formed
# exactly in 128-bit limbs and rounded at the last bit that R keeps.
@noinline function _scaled_coefficient(a::R, b::R, c::R, p::R) where {R<:Base.IEEEFloat}
  n = precision(R)
  ma, ea = frexp(abs(a))
  mb, eb = frexp(abs(b))
  mc, ec = frexp(abs(c))
  Ma, Mb, Mc = (unsafe_trunc(UInt64, ldexp(m, n)) for m in (ma, mb, mc))   # in [2^(n-1), 2^n)
  P = widemul(Ma, Mb)                                                      # < 2^(2n), exact
  mask = (UInt128(1) << n) - 1
  A, B = (P >> n) * Mc, (P & mask) * Mc                                    # the product is A·2^n + B
  H, L = A + (B >> n), B & mask                                            # = H·2^n + L, L < 2^n
  nb = 128 - leading_zeros(H)
  E0 = ea + eb + ec - 2n                                                   # the product ≈ H·2^E0
  top = nb - 1 + E0                                                        # exponent of its leading bit
  g = exponent(floatmin(R)) - (n - 1)                                      # the subnormals' spacing 2^g
  keep = min(n, top - g + 1)                                               # bits that R keeps
  keep < 0 && return copysign(zero(R), p)
  u = nb - keep
  q, rem, half = H >> u, H & ((UInt128(1) << u) - 1), UInt128(1) << (u - 1)
  q += (rem > half) | ((rem == half) & ((L != 0) | isodd(q)))
  r = ldexp(R(q), E0 + u)                                                  # exact: q fits R's grid
  (signbit(a) ⊻ signbit(b) ⊻ signbit(c)) ? -r : r
end
@noinline _scaled_coefficient(a::R, b::R, c::R, p::R) where {R<:AbstractFloat} = _exceptional_product(a, b, c)
@inline _recoverable_factors(a, b, c) =
  isfinite(a) & isfinite(b) & isfinite(c) & !iszero(a) & !iszero(b) & !iszero(c)

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
@inline function _power_factors(::Type{T}, k::Integer) where {T<:AbstractFloat}
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
    d = acc.ex - e
    s, c = _compensated_add(ldexp(acc.s, d), ldexp(acc.c, d), m)
    return _ExpSum{R}(s, c, e)
  end
  s, c = _compensated_add(acc.s, acc.c, ldexp(m, e - acc.ex))
  _ExpSum{R}(s, c, acc.ex)
end
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
function _expsum_value(::Type{R}, acc::_ExpSum{A}) where {R, A}
  m = acc.s + acc.c
  iszero(m) && return zero(R)
  convert(R, _scaled_product(one(A), m, acc.ex))
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

# √(Σ μᵢ (xᵢ - yᵢ)²) for an ordinary sum of squares s in R that overflowed or fell below
# floatmin(R): the squared-residual kernel gives the sum as m 2^(2k), and the norm is √m 2^k,
# rounded once (twice where it is subnormal). An entry that is not finite keeps the ordinary
# result √s: Inf, or NaN where a NaN or an Inf - Inf residual occurs.
@noinline function _scaled_norm(x, y, s::R) where {R}
  q = _residual_squares(x, y, R)
  q === nothing && return sqrt(s)
  m, k = q
  _scaled_product(sqrt(m), one(R), k)
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

# the exponent of the largest residual: typemax(Int) for an entry that is not finite, typemin(Int)
# for an exactly zero residual
@inline function _residual_exponent(a, b, ::Type{R}) where {R}
  (isfinite(a) && isfinite(b)) || return typemax(Int)
  d, h = _halved_difference(a, b, R)
  iszero(d) ? typemin(Int) : exponent(abs(d)) + h
end

# the residual scaled by 2^-k (s) or, where it was halved, by 2^(1-k) (h). The loop over the entries
# branches on the difference itself: the pair (d, h) of _halved_difference would become a select,
# which computes the rare halved case and both scalings for every entry (about 1.5x slower).
@inline function _scaled_residual(a, b, s, h, ::Type{R}) where {R}
  d = _square_difference(a, b, R)
  isfinite(d) ? _scale_value(d, s) : _scale_value(_halved(a, b, R), h)
end

function _square_exponent(x, y, ::Type{R}) where {R}
  k = typemin(Int)
  if y === nothing
    for a in _stored_values(x)
      k = max(k, _residual_exponent(a, zero(R), R))
    end
  else
    @inbounds for i in eachindex(x, y)
      k = max(k, _residual_exponent(x[i], y[i], R))
    end
  end
  k
end

function _square_scaled_sum(x, y, k, ::Type{R}) where {R}
  s, h = _power_factors(R, -k), _power_factors(R, 1-k)
  v = e = zero(R)
  if y === nothing
    @inbounds for (i, a) in enumerate(_stored_values(x))
      q = _square_multiplicity(x, i) * abs2(_scaled_residual(a, zero(R), s, h, R))
      v, e = _compensated_add(v, e, q)
    end
  else
    @inbounds for i in eachindex(x, y)
      v, e = _compensated_add(v, e, abs2(_scaled_residual(x[i], y[i], s, h, R)))
    end
  end
  v + e
end

# The squared-residual kernel (R1): Σ μᵢ (xᵢ - yᵢ)² = s 2^(2k) in R, so that the half sum is
# s 2^(2k-1). y === nothing stands for y = 0 and visits the stored entries of x only, each with its
# logical multiplicity μᵢ (2 for an off-diagonal entry of a symmetric iterate). Every residual is
# formed before it is scaled by 2^-k, k the exponent of the largest one, and the squares are summed
# compensated, so that no square overflows and a small residual after a cancellation is kept. It
# gives (zero(R), 0) when every residual is exactly zero and `nothing` when an entry is not finite
# (the ordinary result then stands: a genuine Inf or NaN is not an intermediate failure).
function _residual_squares(x, y, ::Type{R}) where {R}
  k = _square_exponent(x, y, R)
  k == typemax(Int) && return nothing
  k == typemin(Int) && return (zero(R), 0)
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

