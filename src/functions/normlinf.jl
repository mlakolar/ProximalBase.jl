####################################
#  ℓ∞ norm  λ max|xᵢ|
####################################

# The type, the checks and the oracles are here, with the threshold of the ℓ1 ball that the clip needs (`_l1ball_threshold!`,
# Condat's algorithm with a fallback to Michelot's). The array methods of `prox!` and `value_and_prox!`, and the hook of
# the wrappers, check the arguments (`_check_l2_arrays`) before any write, as NormL2 does.

"""
    NormLinf(λ = 1)

The ℓ∞ norm `f(x) = λ max|xᵢ|` (the largest magnitude of the entries, of a matrix included), with a
nonnegative level `λ` (an `ArgumentError` otherwise, `NaN` included). It is not separable. Its proximal operator
clips the entries at a level `τ ≥ 0`: `prox_{γf}(x)ᵢ = clamp(xᵢ, -τ, τ)`, where `τ` solves
`Σᵢ (|xᵢ| - τ)₊ = γλ`, and `τ = 0` (the output `+0`) when `γλ ≥ ‖x‖₁`, for finite entries. The comparison is made in
floating point, and in units of a power of two when the sum or the radius leaves the range: with `M = floatmax(Float64)`,
`prox(NormLinf(1.0), [M, M], M)` gives `[M/2, M/2]` and `prox(NormLinf(1.5), [M, M], M)`, whose radius `1.5M` is beyond the
range, gives about `[M/4, M/4]`. This is `x` minus its projection onto the ℓ1 ball of radius `γλ`, formed by the clip
itself: for a positive clip level, the entries within the clip interval are copied, and a zero level fills `+0`. The
level `τ` comes from Condat's algorithm, in O(n) expected time, with Michelot's algorithm when many entries become
candidates.

The value takes arrays, the three iterates and a `SymmetricSparseIterate` (its largest stored magnitude: a
repeated off-diagonal coordinate does not change a maximum), and is `λ` times the largest magnitude rounded once, in
the floating type of the entries. With `λ = 0` the value is zero, a `NaN` or infinite entry included; with `λ = Inf` it
is zero at the zero argument (an empty one included), `Inf` at any other argument without a `NaN` entry, and `NaN` at
a `NaN` entry. A level that is not of the type of the entries is applied in the promoted type and rounded once.

`prox!`, `prox` and `value_and_prox!` take arrays of the same size (a `BlockIterate` or a `SparseIterate` through
indexed reads and writes of its entries, O(length)) and a scalar step; `prox!(x, f, x, γ)` is supported. A negative
or `NaN` step, complex entries and indexing that is not one-based throw an `ArgumentError`, and a size mismatch a
`DimensionMismatch`, before anything is written, also through `Scaled`. A `SymmetricSparseIterate` argument or output
throws an `ArgumentError` before anything is written, also through `Scaled`: its proximal operator is not provided.
After those checks the rules apply in this order. `λ = Inf` gives `+0` at every nonnegative step, `γ = 0` and `γ = Inf`
included (#665), for every input. Otherwise `λ = 0` or `γ = 0` gives the identity (so `γ = Inf` with `λ = 0` is the
identity). Otherwise `γ = Inf` gives `+0`. At positive finite `γ` and `λ`, a `NaN` entry makes every output `NaN`, and an
infinite entry without `NaN` leaves the argument unchanged, also when the finite product `γλ` overflows in floating
point; otherwise the clip above, with the output `+0` when `γλ` is at least `‖x‖₁` or overflows while `‖x‖₁` is finite.
For finite entries a sum of magnitudes above `floatmax/2` is handled in power-of-two units. `value_and_prox!` returns the
value of the stored output. Through `Scaled`, the wrapper's recovery of a radius `γλα` beyond the range has a limit: for
`Scaled(NormLinf(2.0 * 2.0^-1000), 2.0^1000)` at the step `0.75 floatmax(Float64)` on two entries equal to
`floatmax(Float64)`, even the folded level at half the scale is out of range, the wrapper passes an infinite step, and the
output is `+0` where the exact result is about `floatmax/4`.

The accuracy of `τ` is absolute, of order `n eps(T) max|xᵢ| + δ` in the floating type `T` of the output for `n`
entries, with `δ = nextfloat(zero(T))`: unlike the relative bounds of `NormL2`, an entry far below `max|xᵢ|` has no
relative guarantee, and the storage of a subnormal radius or output adds the term `δ` (`x = [δ]`, `λ = δ`, `γ = 0.5`
gives `[δ]` where the exact point is `δ/2`). The entries are read as `abs(convert(T, xᵢ))`; Float32 and Float64 with
arguments of the same type are the supported precisions.

The threshold needs a buffer of length `length(x)`. The object's own buffer is left untouched when the output `y` is an
`Array` with element type `T`, the argument is an `Array` or a view of an `Array`, and `y` does not overlap the
argument (`Base.mightalias` is false): `y` then serves as the buffer, before the clip overwrites it. Otherwise (for
instance `y === x`, an overlapping output, a view or an iterate as output, another output type, or an argument of
another type) a threshold call may use the object's own buffer, which is allocated by the first threshold call that
needs it and again when the size or the element type changes. The early exits before the threshold is selected (a zero or
infinite level, a zero or infinite step, a `NaN` or infinite entry, the zero fill when `γλ ≥ ‖x‖₁`) do not touch it. The same object must not be used from several
tasks at once when its buffer may be in use: construct a fresh `NormLinf(f.λ)` for each task, or synchronize the calls.
"""
struct NormLinf{T<:Real} <: AbstractFunction
  λ::T
  scratch::Scratch

  NormLinf{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ, Scratch()))
end
NormLinf(λ::Real = 1) = NormLinf{typeof(λ)}(λ)

isproximable(::Type{<:NormLinf}) = true
isconvex(::Type{<:NormLinf}) = true

# The buffer is not part of the function. Equal levels give equal hashes, `-0.0` and `0.0` included.
Base.:(==)(f::NormLinf, g::NormLinf) = f.λ == g.λ
Base.hash(f::NormLinf, h::UInt) = hash(iszero(f.λ) ? zero(f.λ) : f.λ, hash(:NormLinf, h))
Base.show(io::IO, f::NormLinf) = print(io, "NormLinf(", f.λ, ")")

####################################
#  Values
####################################

# λ m in the floating type V of the entries, from the largest magnitude m: m = 0 gives zero at every level (an infinite
# one included), a zero level gives zero for any other m (NaN and infinite included), an infinite level gives the product
# with m, and otherwise the product of the level and m is rounded once, in the promoted type when the level is not a V.
@inline function _linf_scale(λ, m, ::Type{V}) where {V}
  iszero(m) && return zero(V)           # zero for every level: 0 λ with λ = Inf is defined as zero
  iszero(λ) && return zero(V)
  if isinf(λ)
    return convert(V, λ) * convert(V, m)
  end
  R = promote_type(V, float(typeof(λ)))
  convert(V, convert(R, λ) * convert(R, m))
end

(f::NormLinf)(x::_ArrayOrSymmetricIterate) =
  (_check_one_based(x); _checkreal(f, x); _linf_scale(f.λ, _maximum_abs(x), _valtype(x)))

####################################
#  Proximal operator
####################################

prox!(y::AbstractArray, f::NormLinf, x::AbstractArray, γ::Real) =
  (_check_l2_arrays(f, y, x, γ); _linf_prox!(y, f, x, γ); y)
value_and_prox!(y::AbstractArray, f::NormLinf, x::AbstractArray, γ::Real) =
  (_check_l2_arrays(f, y, x, γ); _linf_value_and_prox!(y, f, x, γ))
# Routes of wrappers such as Scaled. Their `_checkwrite` does not check sizes or the step, so the hook calls the checked
# method, as the NormL2 hook does: the checks of `_check_l2_arrays` precede any write.
_value_and_prox_unchecked!(y::AbstractArray, f::NormLinf, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)

# The value at the stored output. `_linf_prox!` returns the largest stored magnitude when it knows it without a pass (the
# clip level, or the zero and NaN fills), and a negative number otherwise (the identity branches); the value is then
# that of the argument y, which `f(y)` computes by the same function `_linf_scale` of the same largest magnitude.
@inline function _linf_value_and_prox!(y, f::NormLinf, x, γ::Real)
  T = float(eltype(y))
  m = _linf_prox!(y, f, x, γ)
  m < zero(T) && return f(y)
  eltype(y) === T || return f(y)
  _linf_scale(f.λ, m, T)
end

# no proximal operator for a symmetric iterate (argument or output); nothing is written
prox!(::SymmetricSparseIterate, f::NormLinf, x, ::Real) = _symmetric_prox_error(f)
prox!(::AbstractArray, f::NormLinf, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
value_and_prox!(::SymmetricSparseIterate, f::NormLinf, x, ::Real) = _symmetric_prox_error(f)
value_and_prox!(::AbstractArray, f::NormLinf, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
prox(f::NormLinf, ::SymmetricSparseIterate, ::Real = 1) = _symmetric_prox_error(f)

# The output when the arguments have been checked. It returns the largest magnitude of the stored output, in T =
# float(eltype(y)), or -1 when that is not known without a pass (the identity branches). The branches are those of the
# docstring, in its order. The ordinary path is the pass over the magnitudes (their sum), the threshold and the clip; a
# NaN or infinite sum of magnitudes, or one above floatmax/2, goes to `_linf_exceptional!`, which is not inlined. This
# function is marked `@inline`, so that the compiler can resolve the returned magnitude of the zero output in its callers
# (the measurements at n = 1 favored it).
@inline function _linf_prox!(y, f::NormLinf, x, γ::Real)
  T = float(eltype(y))
  λ = f.λ
  n = length(x)
  isinf(λ) && return _linf_fill!(y, n, zero(T))
  (iszero(λ) | iszero(γ)) && return _linf_copy!(y, x, n, T)
  isinf(γ) && return _linf_fill!(y, n, zero(T))
  s = _linf_sum(x, _AbsRead{T}(), T)
  if s <= floatmax(T) / 2                 # false for a NaN or an infinite sum
    r = _linf_radius(T, γ, λ)
    r >= s && return _linf_fill!(y, n, zero(T))
    τ = _linf_threshold(y, f, x, r, _AbsRead{T}(), T)
    return _linf_clip!(y, x, τ, n)
  end
  _linf_exceptional!(y, f, x, γ, s)
end

# The branches at a sum of magnitudes that is NaN, infinite or above floatmax/2: a NaN entry gives NaN everywhere;
# an infinite entry without NaN gives the identity, the limit of the clip as that entry grows (for a finite radius γλ the
# level τ ≥ max|xᵢ| - γλ grows with it, so the clip changes no other entry), also when the rounded γλ overflows. Finite
# entries whose sum overflows or exceeds floatmax/2 are read scaled by 2^-k, and the clip level found in those units is
# multiplied by 2^k: the prox is positively homogeneous, prox_c(x) = 2^k prox_{c 2^-k}(2^-k x).
#
# The scaled computation, for finite entries and a finite positive step and level. Hypotheses are stated where used.
# - Range. Each scaled entry is at most floatmax 2^-k. With the initial k = ceil(log2 n) + 2 their sum is at most
#   floatmax/4 before rounding. Under the hypothesis n eps(T) ≤ 1/2 (as at `_l1ball_threshold!`) the computed sum s′ is at
#   most floatmax/2. The loop below raises k when it is not, and ends since the scaled entries shrink as k grows. The
#   selector then sees the data under the conditions stated there.
# - Subnormal losses. Hypothesis: the scaling of an entry, or of the radius, changes it by at most one subnormal spacing
#   δ = nextfloat(zero(T)). Rounding to nearest gives at most δ/2; the allowance δ is meant to cover Julia 1.10, whose
#   `ldexp` can lose a whole spacing. Then the scaled sum changes by at most n δ and the scaled radius by at most δ. For a
#   positive radius, the left side of the level equation Σᵢ (yᵢ - τ′)₊ = r′ decreases with slope at most -1 where it is
#   positive, so those changes move the level τ′ by at most (n + 1) δ. If the computed scaled radius is zero, the exact
#   scaled radius is at most δ, and the exact level is within that δ of the largest exact scaled entry (the level for
#   radius zero), whose computed value differs by at most δ: the level moves by at most 2 δ ≤ (n + 1) δ. In the units of
#   x the level 2^k τ′ moves by at most 2^k (n + 1) δ < 16 n² δ (2^k < 8n for the initial k).
#   The nominal scale of the error, n eps(T) max|xᵢ|, is at least eps(T) floatmax/3 here, as the exact sum is above
#   floatmax/3 and max|xᵢ| ≥ floatmax/(3n). Take θ ≤ 0.29 (hypothesis n eps(T) ≤ 1/2), so that a finite computed sum of
#   some of the entries is within a factor 1 + θ of the exact sum of its terms. If the computed sum is finite and exceeds
#   floatmax/2, the exact sum is at least that divided by 1 + θ. If it is infinite, the first overflowing addition adds two
#   finite computed partial sums (or one and an entry) whose sum is at least floatmax, so the exact sum of their terms,
#   and the exact total, is at least floatmax/(1 + θ). The movement is below that scale
#   when 64 n² δ ≤ eps(T) floatmax(T). The ratio eps(T) floatmax(T)/δ is about 2^254 in Float32 and 2^2046 in Float64,
#   against 64 n² ≤ 2^50 (n ≤ 2^22, Float32) and 64 n² ≤ 2^108 (n ≤ 2^51, Float64), the sizes of the hypothesis. The
#   statement of the docstring then applies to the scaled path.
# - No lower bound on τ is needed: x = [floatmax, 2^970] with γλ = floatmax in Float64 has the level 2^969.
# - r′ = 2^-k γλ is formed from the significands and exponents of γ and λ, so that γλ does not overflow prematurely when
#   r′ is representable. r′ can itself be Inf (`_linf_scaled_radius(T, floatmax(T), floatmax(T), 2)` is), and then
#   r′ ≥ s′ gives the output +0, as the exact radius exceeds the sum. An r′ below the range rounds to a subnormal or zero,
#   which the radius allowance δ above covers.
@noinline function _linf_exceptional!(y, f::NormLinf, x, γ, s::T) where {T}
  n = length(x)
  isnan(s) && return _linf_fill!(y, n, T(NaN))
  m = _linf_classify(x, _AbsRead{T}(), T)[2]
  isinf(m) && return (_linf_copy!(y, x, n, T); m)
  k = _linf_exponent(n)
  local s′
  while true
    s′ = _linf_sum(x, _ScaledAbsRead{T}(k), T)
    s′ <= floatmax(T) / 2 && break
    k += 1
  end
  r′ = _linf_scaled_radius(T, γ, f.λ, k)
  r′ >= s′ && return _linf_fill!(y, n, zero(T))
  τ′ = _linf_threshold(y, f, x, r′, _ScaledAbsRead{T}(k), T)
  # τ′ is at most the largest scaled entry, 2^-k m rounded; 2^k τ′ is capped at m, which the rounding of the scaling can exceed
  _linf_clip!(y, x, min(ldexp(τ′, k), m), n)
end

# ceil(log2(n)) + 2 for n ≥ 1, from the bit length of n - 1
@inline _linf_exponent(n::Int) = 64 - leading_zeros(n - 1) + 2

# The radius γλ, rounded once to T from the product in the promoted type; Inf when that product rounds to an overflow.
@inline function _linf_radius(::Type{T}, γ, λ) where {T}
  R = promote_type(T, float(typeof(γ)), float(typeof(λ)))
  convert(T, convert(R, γ) * convert(R, λ))
end

# 2^-k γλ for a finite positive γ and λ, from their significands and exponents
@inline function _linf_scaled_radius(::Type{T}, γ, λ, k::Int) where {T}
  R = promote_type(T, float(typeof(γ)), float(typeof(λ)))
  g, a = frexp(convert(R, γ))
  l, b = frexp(convert(R, λ))
  convert(T, ldexp(g * l, Int(a) + Int(b) - k))
end

# The clip level from the threshold of the ℓ1 ball, as a nonnegative number: a level of zero or below gives the output
# +0. `_l1ball_threshold!` returns a level that is at most one of the magnitudes (see there). The buffer is y itself
# when that is safe (see `_linf_buffer_is_output`), and the scratch of f otherwise.
#
# The level makes λτ equal f(y) when y is stored in T. Let m be the largest magnitude of the entries, finite, and
# τ = max(level, 0), so that τ ≤ m. If τ > 0, the entry of magnitude m is stored as ±τ and no stored entry exceeds τ in
# magnitude, so the largest stored magnitude is τ, and `f(y)` applies `_linf_scale` to that same τ, as `value_and_prox!`
# does. The max needs a level that is not NaN: under the conditions at `_l1ball_threshold!` the level is (s - r)/c with
# s finite and c ≥ 1, or the initial y₁ - r, or an entry, so it is a number.
@inline function _linf_threshold(y, f::NormLinf, x, r::T, rd, ::Type{T}) where {T}
  τ = if _linf_buffer_is_output(y, x, T)
    _l1ball_threshold!(y, x, r, rd)
  else
    _l1ball_threshold!(_array_buffer!(f.scratch, T, (length(x),)), x, r, rd)
  end
  max(τ, zero(T))
end

# The output holds Condat's buffer when it is a dense Array of the threshold type that does not overlap the argument.
# The argument is restricted to an Array or a view of one, whose overlap with y `mightalias` detects (a wrapper whose
# parent is not reported by `dataids` is not).
@inline _linf_buffer_is_output(y, x, ::Type{T}) where {T} = false
@inline _linf_buffer_is_output(y::Array{T}, x::Union{Array, SubArray{<:Any, <:Any, <:Array}}, ::Type{T}) where {T} =
  !Base.mightalias(y, x)

####################################
#  Passes over the entries
####################################

# Entry maps: the magnitude of an entry in T, read as abs(convert(T, v)) (abs of an integer entry first would overflow
# at typemin), and the same scaled by 2^-k.
struct _AbsRead{T} end
@inline (::_AbsRead{T})(v) where {T} = abs(convert(T, v))
struct _ScaledAbsRead{T}
  k::Int
end
@inline (g::_ScaledAbsRead{T})(v) where {T} = ldexp(abs(convert(T, v)), -g.k)

# The sum of the magnitudes rd(x[i]) in one vectorized pass (a NaN entry gives a NaN sum, an infinite one an infinite
# sum, and the sum can overflow for finite entries). This is the pass of the ordinary path: a sum that is finite and at
# most floatmax/2 certifies finite magnitudes and a sum that the selector can use, before the selector runs.
function _linf_sum(x, rd, ::Type{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in 1:length(x)
    s += rd(x[i])
  end
  s
end

# The sum and the largest of the magnitudes rd(x[i]), in one pass, for the exceptional path, whose data may hold NaN or
# infinite entries (a NaN entry gives a NaN sum). The maximum is that of the bit patterns for Float32 and Float64 (as
# in `_maximum_abs`): a NaN or an infinite entry gives a NaN or an infinite maximum.
function _linf_classify(x, rd, ::Type{T}) where {T}
  s = zero(T)
  m = zero(T)
  @inbounds for i in 1:length(x)
    a = rd(x[i])
    s += a
    m = ifelse(a > m, a, m)
  end
  s, m
end
function _linf_classify(x, rd, ::Type{T}) where {T<:Union{Float32, Float64}}
  U = _int_type(T)
  s = zero(T)
  m = zero(U)
  @inbounds @simd for i in 1:length(x)
    a = rd(x[i])
    s += a
    m = max(m, reinterpret(U, a))
  end
  s, reinterpret(T, m)
end

# y[i] = x[i] clipped to [-τ, τ], for τ ≥ 0; τ = 0 gives +0. Returns τ, the largest stored magnitude. A zero entry
# keeps its sign below τ. The entries are read in T and stored in the type of y.
@inline function _linf_clip!(y, x, τ::T, n::Int) where {T}
  iszero(τ) && return _linf_fill!(y, n, zero(T))
  _linf_clip_loop!(y, x, τ, n)
  τ
end
# a dense output is written by a vectorized loop; any other (an iterate, whose write keeps its own records) by a plain one
const _DenseOutput = Union{Array, SubArray{<:Any, <:Any, <:Array}}
@inline function _linf_clip_loop!(y::_DenseOutput, x, τ::T, n::Int) where {T}
  S = eltype(y)
  nτ = -τ
  @inbounds @simd for i in 1:n
    v = convert(T, x[i])
    y[i] = convert(S, ifelse(v > τ, τ, ifelse(v < nτ, nτ, v)))
  end
end
@inline function _linf_clip_loop!(y, x, τ::T, n::Int) where {T}
  S = eltype(y)
  nτ = -τ
  @inbounds for i in 1:n
    v = convert(T, x[i])
    y[i] = convert(S, ifelse(v > τ, τ, ifelse(v < nτ, nτ, v)))
  end
end

# y[i] = v, for the zero and NaN outputs; returns the largest stored magnitude, |v| in the type of y
@inline function _linf_fill!(y, n::Int, v)
  S = eltype(y)
  w = convert(S, v)
  @inbounds for i in 1:n
    y[i] = w
  end
  float(abs(v))
end

# y = x, the identity; returns -1, as the largest stored magnitude is not known without a pass
@noinline function _linf_copy!(y, x, n::Int, ::Type{T}) where {T}
  if y !== x
    @inbounds for i in 1:n
      y[i] = x[i]
    end
  end
  -one(T)
end

####################################
#  The threshold of the ℓ1 ball
####################################

# The level τ with Σᵢ (yᵢ - τ)₊ = r for the magnitudes yᵢ = rd(x[i]), i = 1:length(x), by Condat's algorithm (Fast
# projection onto the simplex and the l1 ball, Math. Program. 158, 2016; expected O(n)), which hands over to Michelot's
# algorithm (J. Optim. Theory Appl. 50, 1986) when many entries become candidates. `rd` maps an entry to its magnitude
# in T = eltype(buf): `_AbsRead{T}()`, or the scaled read of the overflow path. `buf` is a scratch array whose length
# is at least length(x), indexed linearly. The level is a number of type T with a rounding error of order
# n eps(T) max yᵢ + δ, δ = nextfloat(zero(T)) (absolute), and it is at most one of the yᵢ; it can be negative or zero,
# and the caller clamps it below at zero. The projection onto the ℓ1 ball can reuse it as its threshold.
#
# Conditions on the data, which the caller (`_linf_prox!`) establishes with its pass over the magnitudes: the yᵢ are
# finite and nonnegative (a sum that is neither NaN nor infinite certifies finite mapped entries), r is a number with
# 0 ≤ r < s, where s is the sum of that pass and is at most floatmax(T)/2. The proof below also uses the hypothesis
# n eps(T) ≤ 1/2 on the number of entries, which the caller does not check: the Float32 inputs with more than 2^22 entries
# do not satisfy it. The comparisons are plain: a NaN entry would be skipped (`NaN > ρ` is false), which was #677 in the
# legacy kernel, and a NaN entry is routed to its own branch before this function runs.
#
# Why the selector sees finite data and partial sums in range, under that hypothesis. A partial sum in the order of
# Condat's or Michelot's passes, a sum of nonnegative terms, differs from the exact sum of its terms by a relative error
# of at most θ = (1 + eps(T)/2)^n - 1 ≤ 0.29 (n eps(T) ≤ 1/2), and the sum of the pass over the magnitudes is at most
# floatmax/2; the exact total is then at most floatmax/2 / (1 - θ) and a computed partial sum at most
# floatmax/2 (1 + θ)/(1 - θ) < floatmax. The sums s of the passes are therefore finite (the subtractions `s -= y` in the
# last loop of Condat's algorithm do not increase s).
#
# Why an overflowing (y - r) nv still decides correctly. The test of Condat's pass is s > (y - r) nv, the exact test
# (s + y - r)/(nv + 1) > y - r. With y ≤ floatmax/2 (a computed sum of nonnegative terms is at least each term, and the
# sum of the pass is at most floatmax/2) the factor y - r is finite, and the product with nv ≥ 1 overflows
# to +Inf, which needs (y - r) nv > floatmax; since s is a sum of some of the yᵢ it is below floatmax (see above), so
# the exact test is false, as the comparison s > Inf is. A product that overflows to -Inf has y - r < 0 and the exact
# test (s ≥ 0 > -∞) is true, as s > -Inf is. The level (s - r)/nv is finite as s is and nv ≥ 1.
#
# Why the level is at most one of the yᵢ. Condat's algorithm returns a level ρ below every remaining candidate (the last
# loop ends when no candidate is at or below ρ), and the candidates are entries, so ρ is below an entry; the one
# exit with ρ at or above its single candidate y (reached through rounding) returns y. Michelot's level (s - r)/c is a
# mean of entries minus a nonnegative number in exact arithmetic, but its rounded sum can exceed c times the largest
# entry, so that path returns the minimum of its level and the largest entry, which its first pass computes.
function _l1ball_threshold!(buf::AbstractArray{T}, x, a::T, rd::F) where {T, F}
  n = length(x)
  n == 0 && return zero(T)
  @inbounds y1 = rd(x[1])
  @inbounds buf[1] = y1
  # buf[1:nt] holds the discarded entries, buf[nt+1:nt+nv] the candidates, of sum s and level ρ = (s - a)/nv. The level
  # is recomputed from the sum when a candidate joins, rather than updated by a division in a chain of dependent steps.
  # Michelot's algorithm takes over when more than cap = max(1024, n ÷ 8) entries have become candidates. This is a
  # heuristic for the cost: a large candidate set often means that the level starts far below the entries and rises
  # slowly, and then Condat's pass costs most (it need not hold, for instance for many tied entries and a tiny radius).
  nt, nv = 0, 1
  s = y1
  ρ = y1 - a
  cap = max(1024, n ÷ 8)
  many = false
  # Once the level has risen, most entries fall below it. One test for four entries reduces the sensitivity of the loop
  # to where its code lands in memory, as observed on one processor.
  j = 2
  @inbounds while j + 3 <= n
    y₁, y₂, y₃, y₄ = rd(x[j]), rd(x[j + 1]), rd(x[j + 2]), rd(x[j + 3])
    if (y₁ > ρ) | (y₂ > ρ) | (y₃ > ρ) | (y₄ > ρ)
      nt, nv, s, ρ, many = _condat_step!(buf, y₁, nt, nv, s, ρ, a, cap)
      many || ((nt, nv, s, ρ, many) = _condat_step!(buf, y₂, nt, nv, s, ρ, a, cap))
      many || ((nt, nv, s, ρ, many) = _condat_step!(buf, y₃, nt, nv, s, ρ, a, cap))
      many || ((nt, nv, s, ρ, many) = _condat_step!(buf, y₄, nt, nv, s, ρ, a, cap))
      many && break
    end
    j += 4
  end
  @inbounds while !many && j <= n
    nt, nv, s, ρ, many = _condat_step!(buf, rd(x[j]), nt, nv, s, ρ, a, cap)
    j += 1
  end
  many && return _l1ball_michelot(x, a, rd, T)
  _condat_finish!(buf, nt, nv, s, a)
end

# One entry y of Condat's pass: below the level ρ it is discarded; otherwise it joins the candidates when their level
# with y, (s + y - a)/(nv + 1), exceeds y - a, and starts them anew if not. The last value is true when more than cap
# entries have become candidates.
@inline function _condat_step!(buf::AbstractArray{T}, y::T, nt::Int, nv::Int, s::T, ρ::T, a::T, cap::Int) where {T}
  y > ρ || return nt, nv, s, ρ, false
  if s > (y - a) * nv
    nv += 1
    nv > cap && return nt, nv, s, ρ, true
    s += y
    @inbounds buf[nt + nv] = y
    return nt, nv, s, (s - a) / nv, false
  end
  nt += nv
  @inbounds buf[nt + 1] = y
  nt, 1, y, y - a, false
end

# The end of Condat's algorithm, from the discarded entries buf[1:nt] and the candidates buf[nt+1:nt+nv] of sum s, whose
# level is ρ = (s - a)/nv, recomputed from the sum after each change of the candidates. The level returned is below every
# remaining candidate, or is the single remaining candidate when rounding leaves it at or below ρ.
function _condat_finish!(buf::AbstractArray{T}, nt::Int, nv::Int, s::T, a::T) where {T}
  ρ = (s - a) / nv
  # the discarded entries above ρ rejoin the candidates, collected at the front
  w = 0
  @inbounds for i in 1:nt
    y = buf[i]
    if y > ρ
      w += 1
      buf[w] = y
      s += y
      ρ = (s - a) / (w + nv)
    end
  end
  @inbounds for i in 1:nv
    buf[w + i] = buf[nt + i]
  end
  len = w + nv
  # drop the candidates at or below ρ until none is left
  changed = true
  @inbounds while changed
    changed = false
    i = 1
    while i <= len
      y = buf[i]
      if y <= ρ
        len == 1 && return y            # reached through rounding: the single candidate is returned
        buf[i] = buf[len]
        len -= 1
        s -= y
        ρ = (s - a) / len
        changed = true
      else
        i += 1
      end
    end
  end
  ρ
end

# Michelot's algorithm: the level of all the entries, then of those above it, until that set stops changing (Newton's
# method on the convex τ ↦ Σᵢ max(yᵢ - τ, 0) - a, from below, each step a vectorized pass). Each pass gives the sum s and
# the count c of the entries above the current level τ, and the next level is (s - a)/c.
#
# Stopping rule. In exact arithmetic the sets above the levels decrease and the process ends when two consecutive sets
# have the same count. With rounding, a level can round onto the entries: x = ones(2001), a = 8e-14 in Float64 has the
# level (2001 - a)/2001 = 1.0, no entry is above it, and the next level is -a/0 = -Inf, from which the sets alternate
# between all the entries and none (the loop of the legacy kernel ended at -Inf for n = 2001). Here the
# loop stops when the count c of the entries above τ is 0 or is not below the count of the pass before, and returns τ,
# the last level computed from a nonempty set. The first pass has τ = -Inf, c = n ≥ 1, and a count that decreases at each
# later pass that continues, so the loop runs at most n + 1 passes. When c is 0 or has not decreased, the level τ of the
# last nonempty set is returned, capped at the largest entry: 1.0 in the example above, the value of the entries.
function _l1ball_michelot(x, a::T, rd::F, ::Type{T}) where {T, F}
  n = length(x)
  # the first pass has τ = -Inf, so that every entry is above it: the sum and the count n, and with them the largest
  # entry, which bounds the level returned (see `_l1ball_threshold!`)
  s, mx = _linf_classify(x, rd, T)
  τ = (s - a) / n
  cnt = n
  for _ in 1:n
    s, c = zero(T), 0
    @inbounds @simd for i in 1:n
      y = rd(x[i])
      above = y > τ
      s += ifelse(above, y, zero(T))
      c += above
    end
    (c == 0 || c >= cnt) && return min(τ, mx)
    cnt = c
    τ = (s - a) / c
  end
  min(τ, mx)
end

####################################
#  Optimality
####################################

# kkt_violation(f, v, x) of f = λ‖·‖∞ is the Euclidean distance from -v to the subdifferential of f at x. The norm is not
# separable, so it has the full-point form and no form for an entry or a group. With W = float(promote_type(eltype(v),
# eltype(x))), the entries read as convert(W, ·) and c the level λ rounded once to W (_kkt_coefficient, as _l2_score forms it),
# the value is
#   x = 0:  the distance from v to the ℓ1 ball of radius c: 0 when Σ|vᵢ| ≤ c, and otherwise sqrt(Σ min(|vᵢ|, τ)²) with
#           Σ (|vᵢ| - τ)₊ = c, that is the norm of prox_{c‖·‖∞}(v);
#   x ≠ 0:  with m = max|xᵢ|, the face J = {i : |xᵢ| == m} (equality after conversion to W, no tolerance) and uᵢ = -sign(xᵢ) vᵢ
#           on J, sqrt(Σ_{i∉J} vᵢ² + Σ_{i∈J} min(uᵢ, τ)²) with Σ_{i∈J} (uᵢ - τ)₊ = c. Here τ can be negative.
# Rules, in this order. v and x are classified in one pass each (_linf_classify, on abs(convert(W, ·))): sums sv, sx and
# largest magnitudes mv, m.
#   1. A NaN sum (isnan(sx) | isnan(sv)) gives NaN: for a type other than Float32 and Float64 the maximum of _linf_classify
#      does not carry a NaN, its sum does. Otherwise an infinite mv or m gives Inf. A sum that is infinite while the
#      largest magnitudes are finite is an overflow of finite entries, and goes to the scaled path (_linf_scaled).
#   2. An infinite c gives 0 at x = 0 and Inf at x ≠ 0. c = 0 gives ‖v‖₂.
#   3. Otherwise τ and the sum of squares of the residual, as below.
#
# The ordinary path forms τ and the sum of squares in W. It uses no buffer, and allocates nothing for Array arguments of
# Float32 or Float64.
#   Zero point: sv ≤ c gives 0. Otherwise τ = max(level, 0), the level from `_l1ball_michelot` on v. Its preconditions
#      (see there) are met by the values |vᵢ|, which are finite and nonnegative since sv is finite, by 0 ≤ c < sv ≤
#      floatmax(W)/2 (the tests of this function), and by n eps(W) ≤ 1/2, which is not checked (as for the prox).
#   Face: one pass for S = Σ_J uᵢ, cnt = |J| and μ = min_J uᵢ, and τ0 = (S - c)/cnt. If τ0 ≤ μ the entries of J are all active
#      and τ = τ0, which is negative when S < c: τ is not clamped on a face. Otherwise the entry of μ is not active, and
#      τ = τ0 + max(level, 0) with the level from `_l1ball_michelot` on Lᵢ = max(uᵢ - τ0, 0) for i ∈ J and 0 off J (the lazy
#      _LinfShifted). τ0 is a lower bound of the exact level (it is the first level of Michelot's iteration on J), and for
#      any σ ≤ τ, Σ_J (uᵢ - τ)₊ = Σ_J (max(uᵢ - σ, 0) - (τ - σ))₊, so the level of the Lᵢ for the radius c is τ - τ0.
#      The shift is by τ0 and not by μ: with the shift by μ each Lᵢ has the size of |μ| when one face entry is far below the
#      level, and the level inherits an error of order a u |μ|. Preconditions: the Lᵢ are nonnegative and finite, and since
#      τ0 > μ each is at most uᵢ - μ, so their sum is at most Σ_J (uᵢ - μ) ≤ (cnt + 1) Σ|vᵢ| (as Σ_J uᵢ ≤ Σ|vᵢ| and |μ| ≤ Σ|vᵢ|),
#      with the exact Σ|vᵢ| at most 1.41 sv (a computed sum of nonnegative terms has a relative error of at most θ = 0.29 for
#      n eps(W) ≤ 1/2). The ordinary path on a face requires (n + 1) sv ≤ floatmax(W)/4 (_linf_in_range), so the computed sum of
#      the Lᵢ is at most 1.29 · 1.41/4 floatmax(W) < floatmax(W)/2, as `_l1ball_michelot` needs. S is then finite, and τ0 is
#      finite unless S - c overflows (c close to floatmax(W) and S negative), which the test isfinite(τ0) sends to the scaled
#      path. Σ_J Lᵢ ≥ Σ_J (uᵢ - τ0) = c in exact arithmetic, which is the condition 0 ≤ c < sum of the routine up to the
#      boundary below. In the other branch no routine runs.
#   Where rounding makes the sum of the routine not exceed c (the computed τ0 is above the exact level, or its sum is formed in
#   another order than the test), it returns a level that is at or below zero, or within rounding of zero. The max then gives
#   τ = τ0 on a face, and 0 at the zero point. That is the limit of the exact level: when Σ_J (uᵢ - τ0) = c the level is τ0 (the
#   closed form, within its rounding), and sv = c puts v on the boundary of the ball, at distance 0.
#   Squares: one loop for Q = Σ rᵢ², with rᵢ = min(|vᵢ|, τ) at the zero point (τ = Inf for c = 0, where rᵢ = |vᵢ|), and on a face
#   uᵢ clipped at τ on J and vᵢ off J. sqrt(Q) is returned when Q is finite and not below n floatmin/eps (_squares_normal).
#   Otherwise _linf_cold reads the residual, as a lazy vector, once in `_residual_squares`, which scales by the exponent of
#   its largest entry (as _l2_score_cold). It gives zero when all the residual entries are zero, and Inf for an entry that is
#   not finite.
#   Range: the scaled path is used when sv > floatmax(W)/2 at the zero point, when (n + 1) sv > floatmax(W)/4 on a face,
#   and when τ0 is not finite. That path forms τ in units of a power of two.
#
# Error of the ordinary path. To first order in n u, with u = eps(W)/2, n = length(x), the entries and c normal numbers,
# Λ = ‖v‖₂ + λ and d the exact distance for the radius c. A sum of k terms in any order has an error of at most
# (k - 1) u times the sum of the magnitudes.
#   (1) The level. p ↦ (c′/c) p maps the ball (the simplex) of radius c onto that of radius c′ and moves p by at most
#       |c - c′| ‖p‖₂/c ≤ |c - c′|, since ‖p‖₂ ≤ ‖p‖₁ = c. The distance changes by at most |c - c′|, so rounding λ to c costs at most
#       u λ.
#   (2) The level equation. Let τ̂ be the computed level, A = {i : uᵢ > τ̂} the active entries (uᵢ = |vᵢ| at the zero point),
#       a = |A| ≥ 1 and η = a τ̂ - (Σ_A uᵢ - c). Then Σ_A (uᵢ - τ̂) = c - η, so the residual formed from τ̂ is the exact
#       residual for the radius c - η. In d² = Σ_{i∉A} rᵢ² + a τ², with τ solving Σ_A (uᵢ - τ) = c, dτ/dc = -1/a and
#       d(d²)/dc = -2τ: the distance changes by |τ||η|/d to first order, and d ≥ √a |τ| gives |τ|/d ≤ 1/√a.
#   (3) Zero point, and a face with τ0 ≤ μ (then A = J, a = cnt). For the zero point A is the set of the last pass of the
#       routine, taken to be the set above the level it returns (the iteration ends when the count does not decrease). The sum of the a
#       terms uᵢ has an error of at most (a - 1) u ‖u_A‖₁; the subtraction and the division add u |Σ_A uᵢ - c| = u a |τ| each:
#       |η| ≤ (a - 1) u ‖u_A‖₁ + 2 u a |τ|. With ‖u_A‖₁ ≤ √a ‖u_A‖₂ ≤ √a ‖v‖₂ the effect on d is at most
#       (a - 1) u ‖v‖₂ + 2 u d.
#   (4) A face with τ0 > μ. Let σ = τ0 as computed (a number used consistently), t = τ - σ ≥ 0, A the active set, a = |A| and
#       B = J \ A, b = |B| = cnt - a, whose residuals are the uᵢ themselves, so ‖u_B‖₂ ≤ d. Σ_A Lᵢ = c + a t. The values Lᵢ are
#       rounded (u each), the sum of the a terms has an error of at most (a - 1) u Σ_A Lᵢ, the subtraction and the division add
#       u |Σ_A Lᵢ - c| = u a t each, and τ = σ + level is rounded: η = a τ̂ - (Σ_A uᵢ - c) satisfies
#       |η| ≤ a u (c + a t) + 2 u a t + a u |τ|. Since Σ_A uᵢ = c + a τ and S = Σ_A uᵢ + Σ_B uᵢ, τ0 = (a τ + Σ_B uᵢ)/cnt and
#       t = Σ_B (τ - uᵢ)/cnt. Then a t ≤ (a/cnt) (b |τ| + √b d), and with |τ| ≤ d/√a (as d ≥ √a |τ|)
#       a t |τ|/d ≤ (b/cnt + √(ab)/cnt) d ≤ 3d/2. The effect on d, |τ||η|/d, is at most
#       √a u c + (a + 2) (3/2) u d + u d = (3a/2 + 4) u d + √a u c.
#   (5) The residual is formed without rounding (a min, a sign change, or vᵢ). The sum of squares and the root add a
#       relative (n/2 + 1) u to the norm of the residual.
#   (6) d ≤ ‖v‖₂ + c, since -v is at distance d from a set of vectors of 2-norm at most c.
#   Together, with d ≤ ‖v‖₂ + c and a ≤ n - 1 on this branch: at most (3n/2 + 4) u Λ for the zero point and for τ0 ≤ μ, and
#   (3a/2 + n/2 + 5) u d + (√a + 1) u c ≤ (2n + √n + 5) u Λ for τ0 > μ. That is the bound of the function,
#   (2n + √n + 5) u (‖v‖₂ + λ). The √n term is the one with c: the shifted values of the active entries add up to c + a t, so
#   the summation error of the level is proportional to c. On a long sum of nearly equal values the error grows about
#   linearly in n in practice (measured about 0.01 n to 0.15 n units of u Λ on one entry far below the level and a bulk of equal
#   entries), inside the bound. The exit sv ≤ c costs at most (n - 1) u ‖v‖₂: v scaled by c/Σ|vᵢ| is in the ball. The bound has
#   no term for entries or terms near the subnormal range, where each adds up to the spacing of W.

# Entry reads with the sign: convert(W, v), and the same scaled by 2^-k (the scaled path).
struct _ValRead{W} end
@inline (::_ValRead{W})(v) where {W} = convert(W, v)
struct _ScaledValRead{W}
  k::Int
end
@inline (g::_ScaledValRead{W})(v) where {W} = ldexp(convert(W, v), -g.k)

# The values Lᵢ = max(uᵢ - σ, 0) for the face J = {i : |xᵢ| == m} and 0 off J, where uᵢ = -sign(xᵢ) rd(vᵢ), as a vector of
# length(x) that `_l1ball_michelot` reads without a copy. rd is _ValRead{W}() or the scaled read, and the shift σ (the
# first level τ0 of the face) is in the units of rd.
struct _LinfShifted{W, V, X, R} <: AbstractVector{W}
  v::V
  x::X
  rd::R
  m::W
  σ::W
end
Base.size(a::_LinfShifted) = (length(a.x),)
Base.IndexStyle(::Type{<:_LinfShifted}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(a::_LinfShifted{W}, i::Int) where {W}
  @boundscheck checkbounds(a, i)
  @inbounds begin
    xi = convert(W, a.x[i])
    vi = a.rd(a.v[i])
  end
  ifelse(abs(xi) == a.m, max(ifelse(xi > 0, -vi, vi) - a.σ, zero(W)), zero(W))
end

# The residual of the distance, in the units of v: min(|vᵢ|, τ) when m = 0 (the zero point, and c = 0 with τ = Inf), and on
# a face with the largest magnitude m: min(uᵢ, τ) on J and vᵢ off J.
struct _LinfResidual{W, V, X} <: AbstractVector{W}
  v::V
  x::X
  m::W
  τ::W
end
Base.size(a::_LinfResidual) = (length(a.v),)
Base.IndexStyle(::Type{<:_LinfResidual}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(a::_LinfResidual{W}, i::Int) where {W}
  @boundscheck checkbounds(a, i)
  @inbounds vi = convert(W, a.v[i])
  iszero(a.m) && return min(abs(vi), a.τ)
  @inbounds xi = convert(W, a.x[i])
  abs(xi) == a.m ? min(ifelse(xi > 0, -vi, vi), a.τ) : vi
end

# S = Σ_J uᵢ, cnt = |J| and μ = min_J uᵢ in one pass, for the face J = {i : |xᵢ| == m} and uᵢ = -sign(xᵢ) rd(vᵢ)
@inline function _linf_face_sums(v, x, m::W, rd, ::Type{W}) where {W}
  S = zero(W)
  cnt = 0
  μ = W(Inf)
  @inbounds @simd for i in 1:length(x)
    xi = convert(W, x[i])
    vi = rd(v[i])
    on = abs(xi) == m
    u = ifelse(xi > 0, -vi, vi)
    S += ifelse(on, u, zero(W))
    cnt += on
    μ = ifelse(on, min(μ, u), μ)
  end
  S, cnt, μ
end

# the sums of squares of the three residuals, for the ordinary path
@inline function _linf_squares_zero(v, τ::W, ::Type{W}) where {W}
  Q = zero(W)
  @inbounds @simd for i in 1:length(v)
    Q += abs2(min(abs(convert(W, v[i])), τ))
  end
  Q
end
@inline function _linf_squares_face(v, x, m::W, τ::W, ::Type{W}) where {W}
  Q = zero(W)
  @inbounds @simd for i in 1:length(x)
    xi = convert(W, x[i])
    vi = convert(W, v[i])
    Q += abs2(ifelse(abs(xi) == m, min(ifelse(xi > 0, -vi, vi), τ), vi))
  end
  Q
end

# sqrt(Q) when Q is finite and normal in the sense of _squares_normal, and the recomputation otherwise
@inline function _linf_finish(Q::W, v, x, m::W, τ::W) where {W}
  (isfinite(Q) & _squares_normal(Q, length(v), W)) && return sqrt(Q)
  _linf_cold(v, x, m, τ, W)
end

# The norm of the residual, by the squared-residual kernel (units of the exponent of its largest entry, as _l2_score_cold). A
# residual entry that is not finite gives Inf.
@noinline function _linf_cold(v, x, m::W, τ::W, ::Type{W}) where {W}
  R = _recovery_type(W)
  q = _residual_squares(_LinfResidual(v, x, m, τ), nothing, R)
  q === nothing && return W(Inf)
  s, k = q
  convert(W, _scaled_product(sqrt(s), one(R), k))
end

# Whether the sum sv of the magnitudes of v is within the range of the ordinary path (see above)
@inline _linf_in_range(sv::W, n::Int, zp::Bool, ::Type{W}) where {W} =
  zp ? sv <= floatmax(W) / 2 : convert(W, n + 1) * sv <= floatmax(W) / 4

# The value for the arguments of the same length n ≥ 1, the level c (a number of type W), and W.
function _linf_point(c::W, v, x, ::Type{W}) where {W}
  n = length(x)
  sx, m = _linf_classify(x, _AbsRead{W}(), W)
  sv, mv = _linf_classify(v, _AbsRead{W}(), W)
  (isnan(sx) | isnan(sv)) && return W(NaN)
  (isinf(m) | isinf(mv)) && return W(Inf)
  isinf(c) && return iszero(m) ? zero(W) : W(Inf)
  iszero(c) && return _linf_finish(_linf_squares_zero(v, W(Inf), W), v, x, zero(W), W(Inf))
  zp = iszero(m)
  zp && sv <= c && return zero(W)
  _linf_in_range(sv, n, zp, W) || return _linf_scaled(c, v, x, m, mv, W)
  if zp
    τ = max(_l1ball_michelot(v, c, _AbsRead{W}(), W), zero(W))
    return _linf_finish(_linf_squares_zero(v, τ, W), v, x, m, τ)
  end
  S, cnt, μ = _linf_face_sums(v, x, m, _ValRead{W}(), W)
  τ0 = (S - c) / cnt
  isfinite(τ0) || return _linf_scaled(c, v, x, m, mv, W)
  τ = τ0
  if τ0 > μ
    τ = τ0 + max(_l1ball_michelot(_LinfShifted(v, x, _ValRead{W}(), m, τ0), c, _AbsRead{W}(), W), zero(W))
  end
  _linf_finish(_linf_squares_face(v, x, m, τ, W), v, x, m, τ)
end

# The same out of the ordinary range, for finite entries and a finite positive c, in units of 2^k as `_linf_exceptional!` does
# for the prox: v is read as 2^-k vᵢ (the scaled reads), the level is 2^-k c (`_linf_scaled_radius`, formed from the significand
# and the exponent), and τ = 2^k τ′. k starts at ceil(log2 n) + 2, where the scaled magnitudes sum to at most floatmax/4 before
# rounding, and is raised until the scaled sum is in the range of the ordinary path (the scaled entries shrink as k grows, so
# this ends). Then the ordinary path's formulas hold in these units, with the same preconditions: the scaled sum is within the
# limits of `_linf_in_range`, c′ ≤ c/4 is finite (k ≥ 2, as _linf_exponent(n) ≥ 2), S′ and the shifted values are finite, and
# τ0′ = (S′ - c′)/cnt is finite as |S′| ≤ 1.29 Σ|vᵢ|′ ≤ 0.17 floatmax and c′ ≤ floatmax/4. The scaling of an entry or of c can lose up to a subnormal spacing of W, as in the prox.
# τ is capped at the largest |vᵢ|, mv, which τ does not exceed in exact arithmetic (it is below the largest uᵢ), so that the
# rounding of 2^k τ′ does not overflow; a τ below -floatmax is -Inf, and the cold path then gives Inf, as the distance
# is then above floatmax. A scaled level at least the scaled sum gives 0 at the zero point. The squares of the
# residual are out of the ordinary range, so the value goes to _linf_cold.
@noinline function _linf_scaled(c::W, v, x, m::W, mv::W, ::Type{W}) where {W}
  n = length(x)
  zp = iszero(m)
  k = _linf_exponent(n)
  local sv′
  while true
    sv′ = _linf_sum(v, _ScaledAbsRead{W}(k), W)
    _linf_in_range(sv′, n, zp, W) && break
    k += 1
  end
  c′ = _linf_scaled_radius(W, one(W), c, k)
  if zp
    c′ >= sv′ && return zero(W)
    τ′ = max(_l1ball_michelot(v, c′, _ScaledAbsRead{W}(k), W), zero(W))
    τ = min(ldexp(τ′, k), mv)
    return _linf_cold(v, x, m, τ, W)
  end
  rd = _ScaledValRead{W}(k)
  S′, cnt, μ′ = _linf_face_sums(v, x, m, rd, W)
  τ0′ = (S′ - c′) / cnt
  τ′ = τ0′
  if τ0′ > μ′
    τ′ = τ0′ + max(_l1ball_michelot(_LinfShifted(v, x, rd, m, τ0′), c′, _AbsRead{W}(), W), zero(W))
  end
  _linf_cold(v, x, m, min(ldexp(τ′, k), mv), W)
end

# The forms that select an entry or a group throw an ArgumentError, as private methods, so that they do not meet the generic
# methods for a BlockIterate (as for NormL2). The public four-argument methods are those of interface.jl, which reach them.
@noinline _normlinf_local_error(f) = throw(ArgumentError(
  "kkt_violation of $(nameof(typeof(f))) has the full-point form kkt_violation(f, v, x), and no form for an entry or a group: the norm is not separable"))
_level_score(f::NormLinf, λ, g, x, k) = _normlinf_local_error(f)
_kkt_violation(f::NormLinf, g::AbstractArray, x::AbstractArray, i::Integer) = _level_score(f, f.λ, g, x, i)
_kkt_violation(f::NormLinf, g::AbstractArray, x::BlockIterate, k::Integer) = _level_score(f, f.λ, g, x, k)
_kkt_violation(f::NormLinf, g::AbstractArray, x::SymmetricSparseIterate, k::Integer) = _level_score(f, f.λ, g, x, k)

kkt_violation(f::NormLinf, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); _normlinf_local_error(f))
# the level λ is passed in, not read from f: a chain of Scaled wrappers passes the level folded with its factors
function _level_point(f::NormLinf, λ, v::AbstractArray, x::AbstractArray)
  _check_one_based(v, x)
  eltype(v) <: Real || _real_error(:kkt_violation)
  eltype(x) <: Real || _real_error(:kkt_violation)
  length(v) == length(x) || _gradient_length_error(v, x)
  W = float(promote_type(eltype(v), eltype(x)))
  isempty(x) && return zero(W)
  _linf_point(_kkt_coefficient(W, λ, true), v, x, W)
end
kkt_violation(f::NormLinf, v::AbstractArray, x::AbstractArray) = _level_point(f, f.λ, v, x)
kkt_violation(f::NormLinf, ::AbstractArray, ::SymmetricSparseIterate) = _kkt_symmetric_point_error(f)

####################################
#  Scaled
####################################

# The member of the folding of a factor into the level (`_fold_level`) that NormL1 has. The nested route of `Scaled`
# (`_nested_step`) folds the exponents of the factors and of the step into the level, so that the function gets a level
# and a step whose product is the radius. The fold accepts what `_folded_level` accepts: an exact zero level, or a finite
# nonzero level representable in the type (a subnormal one included), and gives nothing otherwise.
function _fold_level(f::NormLinf, α)
  c = _folded_level(f.λ, α)
  c === nothing ? nothing : NormLinf(c)
end
# The fold of `_nested_step`, which passes the step reduced to its significand in [1/2, 1). The fold with the whole
# exponent can fail where the fold with half the scale succeeds: the level then overflows with the whole exponent. The
# result is the function with the half-scale level inside a factor 2: the leaf receives the step 2 mγ in [1, 2), a normal
# number, and the radius is the level times that step, the folded product. The half-scale fold is tried after the whole
# fold has failed. Its level is half the whole-fold level, so it does not repair an underflow of the whole fold to zero.
# When the half-scale fold fails too, the result is nothing and the wrapper keeps its rounded step (see the limit in the
# docstring). The fold with half the scale needs the scale as a `_ProjectionScale`, which `_nested_step` passes. The
# value of the chain is computed from the original chain by the caller (the factor returned with this fold is nothing),
# not from this function.
function _fold_level_step(f::NormLinf, α)
  g = _fold_level(f, α)
  g === nothing || return g
  α isa _ProjectionScale || return nothing
  h = _fold_level(f, _ProjectionScale(α.m, α.e - 1))
  h === nothing ? nothing : Scaled(h, convert(typeof(α.m), 2))
end
# a zero factor: the zero function at a finite level; an infinite level is an indicator, which has no zero multiple here
_zero_scaled(f::NormLinf) = (isfinite(f.λ) || _zero_scale_error(f); ZeroFunction())
# f(x) is zero when the level is zero or the entries are all zero (a NaN or infinite entry is not zero)
_exactly_zero_at(f::NormLinf, x::AbstractArray) = iszero(f.λ) || _all_zero(_host_entries(x))
