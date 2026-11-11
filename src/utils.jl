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
  _exceptional_product(a, b, c, p)
end
@noinline function _exceptional_product(a::T, b::T, c::T, p::T) where {T<:AbstractFloat}
  (iszero(a) || iszero(b) || iszero(c)) && return zero(T)
  (isfinite(a) && isfinite(b) && isfinite(c)) || return p
  ma, ea = _frexp(a)
  mb, eb = _frexp(b)
  mc, ec = _frexp(c)
  ldexp((ma * mb) * mc, _exponent_sum(T, ea, eb, ec))
end
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
  isinf(w) ? T(Inf) : _exceptional_product(a, b, w, p)
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
@inline function _positive_product(::Type{U}, a, b, c) where {U<:AbstractFloat}
  R = _working_type(U, a, b, c)
  convert(U, _positive_product(convert(R, a), convert(R, b), convert(R, c)))
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
@noinline _scaled_coefficient(a::R, b::R, c::R, p::R) where {R<:AbstractFloat} = _exceptional_product(a, b, c, p)
@inline _recoverable_factors(a, b, c) =
  isfinite(a) & isfinite(b) & isfinite(c) & !iszero(a) & !iszero(b) & !iszero(c)

# the value p = c⋅φ of a term with the coefficient c and the unweighted value
# φ ≥ 0: zero where c⋅φ is 0⋅Inf (an infinite weight at a zero entry, a zero
# coefficient at an infinite entry); NaN arguments propagate
@inline _weighted_value(p, φ) = ifelse(isnan(p) & !isnan(φ), zero(p), p)

# Keep the power outside the floating-point range until the final rounding.
@inline function _scaled_product(w::T, v::T, k::Integer) where {T<:AbstractFloat}
  (iszero(w) || iszero(v)) && return w * v
  (isfinite(w) && isfinite(v)) || return w * v
  k = _product_power(T, k)
  mw, ew = _frexp(w)
  mv, ev = _frexp(abs(v))
  copysign(ldexp(_positive_product(mw, mv, one(T)), _exponent_sum(T, ew, ev, k)), v)
end

# A converted parameter must retain normal precision on the ordinary path.
@inline _level_fits(::Type, ::Real) = true
# A level already stored in this precision needs no conversion check.
@inline _level_fits(::Type{T}, λ::T) where {T<:Union{Float32, Float64}} = true
@inline function _level_fits(::Type{T}, λ::Real)::Bool where {T<:AbstractFloat}
  iszero(λ) || !isfinite(λ) || floatmin(T) <= abs(convert(T, λ)) <= floatmax(T)
end

# Three bounded significands, with the power applied after all factors.
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
  (isfinite(w) && isfinite(v)) || return w * v
  k = Int(k)
  mw, ew = _frexp(w)
  mv, ev = _frexp(abs(v))
  copysign(ldexp(_positive_product(mw, mv, one(T)), ew + ev + k), v)
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
#  Buffers
#
####################################

# A buffer that a function object caches and sizes on demand, so that its hot
# calls allocate only when the size or the element type of the input changes.
# The field is untyped; _buffer! returns a concretely typed vector.
mutable struct Scratch
  buf::Any
end
Scratch() = Scratch(nothing)

# an array of eltype T and size dims, reused while they do not change
function _array_buffer!(s::Scratch, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
  b = s.buf
  if b isa Array{T, N}
    size(b) == dims && return b
  end
  b = Array{T, N}(undef, dims)
  s.buf = b
  b
end

# CPU prototypes keep the existing dense buffer and cache checks.
_array_buffer!(s::Scratch, ::Type{T}, x::AbstractArray) where {T} = _array_buffer!(s, T, size(x))

# A buffer for the transformed input, reused while its type and size agree.
# Adapted functions have no cache and allocate a device buffer on each call.
_output_buffer!(s::Scratch, y::Union{Array, SubArray{<:Any, <:Any, <:Array}}) = _array_buffer!(s, eltype(y), size(y))
_output_buffer!(::Nothing, y::AbstractArray) = similar(y)

# A view of an Array whose indices shift to start at 1 keeping their types gets
# a view of the same type into an owned Array, so that f sees the same layout;
# a view with other indices, such as a vector of indices, gets an Array.
const _ShiftableIndex = Union{Int, UnitRange{Int}, StepRange{Int, Int}, Base.Slice{Base.OneTo{Int}}, Base.OneTo{Int}}
function _output_buffer!(s::Scratch, y::SubArray{T, N, Array{T, M}, I}) where {T, N, M, I<:NTuple{M, _ShiftableIndex}}
  shifted = map(_shifted_index, y.indices)
  P = _array_buffer!(s, T, map(last, shifted))
  view(P, map(first, shifted)...)
end

# an index shifted to start at 1, and the extent of the parent dimension that it spans
_shifted_index(i::Int) = (1, 1)
_shifted_index(r::UnitRange{Int}) = (1:length(r), length(r))
_shifted_index(r::Union{Base.Slice{Base.OneTo{Int}}, Base.OneTo{Int}}) = (r, length(r))
function _shifted_index(r::StepRange{Int, Int})
  s = step(r)
  isempty(r) && return (StepRange(1, s, s > 0 ? 0 : 2), 0)
  lo = min(first(r), last(r))
  (StepRange(first(r) - lo + 1, s, last(r) - lo + 1), abs(last(r) - first(r)) + 1)
end
function _output_buffer!(s::Scratch, y::A) where {A<:DenseArray}
  b = s.buf
  b isa A && size(b) == size(y) && return b
  b = similar(y)
  s.buf = b
  b
end
_output_buffer!(::Scratch, y::AbstractArray) = copy(y)

# Most kernels read scratch and write y. Stored iterates must run in place to
# preserve their stored zeros and summation order, then publish the full state.
_wrapper_output(y, z) = y

# Scratch is an implementation detail: copies of a function object share no buffer
Base.copy(s::Scratch) = Scratch()

####################################
#
#  Checks
#
####################################

_checksize(y, x) = size(y) == size(x) || _size_mismatch(size(y), size(x))
@noinline _size_mismatch(sy, sx) =
  throw(DimensionMismatch("the output has size $sy and the input $sx"))

_checknonneg(name, v::Real) = v >= 0 || throw(ArgumentError("$name must be nonnegative, got $v"))
_checknonneg(name, v::AbstractArray) = all(>=(0), v) || throw(ArgumentError("$name must be nonnegative"))
_checknonneg(name, ::Nothing) = true

# A weight or a bound: a scalar, or the entry i of an array. The bounds check
# propagates: a direct call checks it, and the loops that first check the
# lengths of the parameter arrays (_checkparam) elide it with @inbounds.
@inline _entry(w::Real, ::Integer) = w
Base.@propagate_inbounds _entry(w::AbstractArray, i::Integer) = w[i]
@inline _entry(::Nothing, ::Integer) = true

# a parameter array has an entry for every entry of x
_checkparam(::Union{Nothing, Real}, x) = true
_checkparam(w::AbstractArray, x) = length(w) == length(x) || _parameter_length_mismatch(length(w), length(x))
@noinline _parameter_length_mismatch(nw, nx) =
  throw(DimensionMismatch("a parameter array has length $nw and the argument has length $nx"))

# The type of a value at x: the real floating type of its entries, promoted
# over the blocks of a tuple. A value computed in another type is rounded to it.
_valtype(x) = float(eltype(x))
# Bodies of generated functions that visit tuple blocks in order. Recursion with Base.tail
# is not inferred beyond 32 blocks, and Base's map stops unrolling there.
_inline(ex) = Expr(:block, Expr(:meta, :inline), ex)
_blockwise(ex, n, result) = Expr(:block, Expr(:meta, :inline), (ex(k) for k in 1:n)..., result)
_blockany(ex, n) = _inline(foldl((a, k) -> :($a || $(ex(k))), 1:n; init = false))
_blocktuple(ex, n) = _inline(Expr(:tuple, (ex(k) for k in 1:n)...))
@generated function _valtype(x::Tuple)
  n = fieldcount(x)
  n == 0 && return _inline(:(Union{}))
  n == 1 && return _inline(:(promote_type(_valtype(x[1]))))
  types = [:(_valtype(x[$k])) for k in 1:n]
  combine(a, b) = :(promote_type($a, $b))
  _inline(VERSION < v"1.11" ? foldr(combine, types) : foldl(combine, types))
end
_value_at(v, x) = convert(_valtype(x), v)
# Arguments, outputs and parameters are real; complex entries are rejected.
_checkreal(f, x) = eltype(x) <: Real || _real_error(nameof(typeof(f)))
@generated _checkreal(f, x::Tuple) = _blockwise(k -> :(_checkreal(f, x[$k])), fieldcount(x), true)
@noinline _real_error(name) = throw(ArgumentError("$name requires real entries"))

# A buffer of zeros of type T and size dims, next to the array a: an Array,
# and a device array of the type of a GPU array a (see the GPU extension).
_zeros_like(a, ::Type{T}, dims) where {T} = zeros(T, dims)

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

# The stored values suffice for maxima; symmetric multiplicities do not matter.
_stored_values(x::AbstractArray) = x
_stored_values(x::SparseArrays.AbstractSparseArray) = nonzeros(x)
function _maximum_abs(x::AbstractArray)
  m = zero(float(eltype(x)))
  @inbounds for v in _stored_values(x)
    m = max(m, abs(float(v)))
  end
  m
end

# The signed integer type of the same width, by dispatch (a conditional on the type is not folded by Julia 1.10)
_int_type(::Type{Float32}) = Int32
_int_type(::Type{Float64}) = Int64

# For Float32 and Float64 the maximum is taken on the bit patterns: with the sign cleared they are ordered like the
# magnitudes (as signed integers, whose maximum vectorizes on more targets than the unsigned one), an integer maximum vectorizes
# where the NaN-aware float maximum is a scalar chain (the trigger of #520 reads this on every value call),
# and a NaN or an Inf gives a NaN or an Inf as before.
function _maximum_abs(x::AbstractArray{T}) where {T<:Union{Float32, Float64}}
  U = _int_type(T)
  vals = _stored_values(x)
  m = zero(U)
  @inbounds @simd for i in eachindex(vals)
    m = max(m, reinterpret(U, vals[i]) & typemax(U))
  end
  reinterpret(T, m)
end
_maximum_abs(x::Diagonal{T}) where {T<:Union{Float32, Float64}} = _maximum_abs(x.diag)

# Scale each entry before summing, including before taking its square.
function _scaled_sum(x::AbstractArray, s, f::F = identity) where {F}
  v = zero(s[1])
  @inbounds for u in _stored_values(x)
    v += f(_scale_value(u, s))
  end
  v
end

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

# Σᵢ aᵢxᵢ over n products as three compensated sums in the type of the scales:
# the large products, those that overflow or exceed floatmax/2n, in units of
# the two scales; the products below floatmin·2^(p+1), which underflow or lose
# bits, in units of 2^-U (_tiny_unit), formed from the significands of the
# factors, so that a weight applied later still finds them; and the others in
# their own units, whose sum cannot overflow. Exact zeros are skipped.
function _split_dot(a::AbstractArray, x::AbstractArray, sa, sx)
  _checksize(a, x)
  R = typeof(sx[1])
  acc = _split_start(R, length(a))
  @inbounds for i in eachindex(IndexLinear(), a, x)
    acc = _split_add(acc, a[i], x[i], sa, sx)
  end
  _split_finish(acc)
end
struct _SplitAcc{R}
  big::R
  eb::R
  small::R
  es::R
  tiny::R
  et::R
  t::R              # the largest product summed in its own units
end
_split_start(::Type{R}, n::Integer) where {R} =
  (z = zero(R); _SplitAcc{R}(z, z, z, z, z, z, floatmax(R) / (2 * max(n, 1))))
_split_finish(acc::_SplitAcc) = (acc.big + acc.eb, acc.small + acc.es, acc.tiny + acc.et)
# Products below floatmin·2^(p+1) are summed in units of 2^-U. With
# U = emax + p + 4 they stay normal there down to about 2^-(emax - emin), below
# which no weight up to floatmax lifts them into the normal range.
_tiny_unit(::Type{R}) where {R} = exponent(floatmax(R)) + precision(R) + 4
_tiny_limit(::Type{R}) where {R} = ldexp(floatmin(R), precision(R) + 1)
@inline function _split_add(acc::_SplitAcc{R}, a, x, sa, sx) where {R}
  ar, xr = convert(R, a), convert(R, x)
  p = ar * xr
  m = abs(p)
  if m > acc.t
    u = ((ar * sa[1]) * sa[2]) * ((xr * sx[1]) * sx[2])
    big, eb = _compensated_add(acc.big, acc.eb, u)
    return _SplitAcc{R}(big, eb, acc.small, acc.es, acc.tiny, acc.et, acc.t)
  elseif m >= _tiny_limit(R) || iszero(ar) || iszero(xr)
    small, es = _compensated_add(acc.small, acc.es, p)
    return _SplitAcc{R}(acc.big, acc.eb, small, es, acc.tiny, acc.et, acc.t)
  end
  ma, ea = _frexp(ar)
  mx, ex = _frexp(xr)
  tiny, et = _compensated_add(acc.tiny, acc.et, ldexp(ma * mx, ea + ex + _tiny_unit(R)))
  _SplitAcc{R}(acc.big, acc.eb, acc.small, acc.es, tiny, et, acc.t)
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

# The largest magnitude, and finiteness, over the stored entries only.
const _WrappedSparse = Union{Adjoint{<:Any, <:SparseArrays.AbstractSparseArray},
                             Transpose{<:Any, <:SparseArrays.AbstractSparseArray}}
_stored_maxabs(x) = _maximum_abs(x)
_stored_maxabs(x::_WrappedSparse) = _maximum_abs(parent(x))
_all_finite(x) = all(isfinite, _stored_values(x))
_all_finite(x::_WrappedSparse) = _all_finite(parent(x))
_all_finite(::Nothing) = true

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

# The scaled-value mechanism that `Scaled` uses when its value leaves the range (calculus.jl,
# `_scaled_parts`). A function that implements it declares `_has_value_parts` and returns from
# `_value_parts(f, x, v)`, given its ordinary value v (infinite, zero or subnormal), either
# `nothing`, to keep the ordinary result, or (m, e): its value m 2^e, with m a finite float,
# zero or in [1/2, 1) in magnitude, to the accuracy of its ordinary computation.
_has_value_parts(::Type) = false
_value_parts(f, x, v) = nothing
# Exact-zero protection. When the ordinary value is exactly zero, `_zero_value_exponent(f, x)` gives
# an exponent E with |f(x)| < 2^E for the exact value, from the parameters and the size of x (no pass
# over x), `_EXACT_ZERO` when that zero is exact, or `nothing` when no such bound is known (the value
# is then recomputed). A zero value that is not exact comes from underflow: a rounding into the
# subnormal range loses less than δ/2, δ the smallest subnormal of the computation type, and a later
# product with a bounded factor (a level, a weight, a logarithm) multiplies that loss. A zero that
# comes from cancellation is within the accuracy of the ordinary computation (R1 measures it against
# the sum of the magnitudes of the terms), which the recomputation does not improve on either.
_zero_value_exponent(f, x) = nothing
# n losses of at most δ/2, each multiplied by less than 2^amp, with a margin of two binades for the
# relative errors of the ordinary sums: n δ 2^amp / 2 < 2^E
@inline _underflow_exponent(::Type{T}, n::Integer, amp::Integer = 0) where {T} =
  exponent(nextfloat(zero(T))) + _log2_ceil(n) + amp + 2
# Is a zero or subnormal value of f exact for every argument, by its type? Then a factor above one
# has nothing to recover from it, and `Scaled` does not try (the test folds away at compile time).
_zero_value_exact(f) = false
# the marker of an exact zero, outside every exponent range (no Int exponent can equal it)
struct _ExactZero end
const _EXACT_ZERO = _ExactZero()
# e with |a| < 2^e, from the bits of a Float64 or Float32 (a subnormal gives the exponent of
# floatmin, an Inf or a NaN a large e): integer operations only, so that the bound above costs a
# few independent instructions. It is an upper bound only: 2^(e-1) <= |a| holds for a normal a. The
# bounds that use it from below (λ times the smallest weight) check λ with `_level_fits`, which a
# subnormal λ stored in T passes; that is harmless: its e is then exponent(floatmin) + 1, so the test
# e - 1 + exponent(min w) > exponent(floatmin) needs w >= 4 and gives c = λ w >= 4 δ, and the
# remaining terms of those bounds count the losses of c as for a normal c.
@inline _upper_exponent(a::Float64) = Int((reinterpret(UInt64, a) >> 52) & 0x7ff) - 1022
@inline _upper_exponent(a::Float32) = Int((reinterpret(UInt32, a) >> 23) & 0xff) - 126
@inline _upper_exponent(a::Real) = (b = abs(float(a)); iszero(b) ? -1000000 : isfinite(b) ? exponent(b) + 1 : 1000000)
# Is an infinite ordinary value known to be the value, from the parameters and the size of x only?
_infinite_value_exact(f, x) = false
# Can the gradient output `out`, already written, share memory with a parameter that the
# parts read? Decided from the identities of the arrays (O(1)).
_parameter_alias(f, out) = false
_mightalias_parameter(out, p) = false
_mightalias_parameter(out::AbstractArray, p::AbstractArray) = Base.mightalias(out, p)
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
#  Lower-triangular indexing
#
####################################

"""
    tril_index(p, r, c)

The linear index of entry `(r, c)`, `r ≥ c`, of the lower triangle of a `p × p`
matrix, numbered column by column: the coordinate of a
`SymmetricSparseIterate` that holds the entry. Inverse of
`tril_cartesian`.
"""
tril_index(p::Integer, r::Integer, c::Integer) = p * (c - 1) - div(c * (c - 1), 2) + r

"""
    tril_cartesian(p, k)

The entry `(r, c)`, `r ≥ c`, of the lower triangle of a `p × p` matrix with
linear index `k`, numbered column by column. Inverse of `tril_index`.
"""
@inline function tril_cartesian(p::Integer, k::Integer)
  rv = div(p * (p + 1), 2) - k
  # j = ⌊(√(1 + 8rv) - 1)/2⌋. For p < 2²⁴, 1 + 8rv < 2⁵⁰ and the floating square
  # root is exact, and three times faster than isqrt; the larger p, beyond any
  # p × p matrix in memory, take integer arithmetic, exact for any p, out of
  # line so that this function stays small enough to inline in inner loops
  j = p < 16_777_216 ? trunc(typeof(rv), (sqrt(1 + 8rv) - 1) / 2) : _tril_column_isqrt(rv)
  i = rv - div(j * (j + 1), 2)
  (p - i, p - j)
end

@noinline _tril_column_isqrt(rv::Integer) = div(isqrt(1 + 8rv) - 1, 2)

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

# Preserve the inexpensive sum-of-squares path at ordinary scales. Outside
# its representable range, hypot accumulates the norm without squaring the
# unscaled differences and without constructing a temporary array. Equal
# arrays also have a zero sum of squares; a vectorized sum of absolute
# differences confirms them without hypot.
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
  v = zero(T)
  @inbounds for i in eachindex(A, B)
    d = S(_norm_difference(A[i], B[i]))
    isnan(d) && return T(NaN)
    v = hypot(v, abs(d))
  end
  v
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

@inline function _square_value(λ, x, y)
  T = _valtype(x)
  s = y === nothing ? _sumabs2(x) : _sumabs2_diff(x, y)
  v = convert(T, λ) * s / 2
  n = y === nothing ? _square_term_count(x) : length(x)
  if (isfinite(v) & (real(s) >= n * (floatmin(T) / eps(T)))) && _level_fits(T, λ)
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

# A finite difference is formed before scaling, so cancellation retains its units.
@inline function _residual_exponent(a, b, ::Type{R}) where {R}
  (isfinite(a) && isfinite(b)) || return typemax(Int)
  d = _square_difference(a, b, R)
  iszero(d) && return typemin(Int)
  isfinite(d) && return exponent(abs(d))
  exponent(abs(R(a) / 2 - R(b) / 2)) + 1
end

@inline function _scaled_residual(a, b, s, h, ::Type{R}) where {R}
  d = _square_difference(a, b, R)
  isfinite(d) ? _scale_value(d, s) : _scale_value(R(a) / 2 - R(b) / 2, h)
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

@noinline function _scaled_square_value(λ, x, y, ordinary)
  R = promote_type(_valtype(x), y === nothing ? _valtype(x) : _valtype(y), float(typeof(λ)))
  k = _square_exponent(x, y, R)
  k == typemax(Int) && return _value_at(ordinary, x)
  k == typemin(Int) && return _value_at(convert(R, λ) * zero(R), x)
  s = _square_scaled_sum(x, y, k, R)
  _value_at(_scaled_product(convert(R, λ), s, _exponent_sum(R, k, k, -1)), x)
end

# Σ (xᵢ - yᵢ)² λ/2 as a significand and an exponent (the scaled-value mechanism), from the
# recomputation above; a non-finite entry or level gives `nothing`
@noinline function _square_parts(λ, x, y)
  R = promote_type(_valtype(x), y === nothing ? _valtype(x) : _valtype(y), float(typeof(λ)))
  (iszero(λ) || !isfinite(λ)) && return nothing
  k = _square_exponent(x, y, R)
  k == typemax(Int) && return nothing
  k == typemin(Int) && return (zero(R), 0)
  s = _square_scaled_sum(x, y, k, R)
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

"""
    _model_precision(D, types...)

Promote real floating entry types with the delivered type `D`. Fixed-width
integers and `Bool` give `Float64`; `BigInt` gives `BigFloat`.

For `value_and_gradient!`, the value passes `R`, the argument and data types;
the gradient passes `D`, the argument and data types. The value of
`value_and_prox!` passes the output and data types, not the argument type.
Callers promote tuple blocks. This helper is not yet used by the oracles.
"""
function _model_precision(::Type{D}, types::Type...) where {D}
  all(T -> T <: Real, (D, types...)) || _real_error(:_model_precision)
  promote_type(map(T -> real(float(T)), (D, types...))...)
end

# CPU fallback retained from frozen lapack.jl.
_is_device_array(::AbstractArray) = false
