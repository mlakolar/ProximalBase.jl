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
    τ = _linf_threshold(y, f, x, r, s, _AbsRead{T}(), T)
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
  τ′ = _linf_threshold(y, f, x, r′, s′, _ScaledAbsRead{T}(k), T)
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
# +0. `_l1ball_threshold!` returns a level that is at most the largest magnitude, within the bound stated there of the exact
# level (it can be negative or zero). The buffer is y itself when that is safe (see `_linf_buffer_is_output`), and the scratch
# of f otherwise.
#
# The level makes λτ equal f(y) when y is stored in T. Let m be the largest magnitude of the entries, finite, and
# τ = max(level, 0), so that τ ≤ m. If τ > 0, the entry of magnitude m is stored as ±τ and no stored entry exceeds τ in
# magnitude, so the largest stored magnitude is τ, and `f(y)` applies `_linf_scale` to that same τ, as `value_and_prox!`
# does. The max needs a level that is not NaN: under the conditions at `_l1ball_threshold!` the level is a level of the
# iteration there (a finite number, computed from finite sums) or the largest entry, so it is a number. s is the sum of the
# magnitudes of the pass of the caller (the sum that is compared with the radius): Condat's pass uses it in the test of the level
# for an early handover and in the start of the handover.
@inline function _linf_threshold(y, f::NormLinf, x, r::T, s::T, rd, ::Type{T}) where {T}
  τ = if _linf_buffer_is_output(y, x, T)
    _l1ball_threshold!(y, x, r, rd, s)
  else
    _l1ball_threshold!(_array_buffer!(f.scratch, T, (length(x),)), x, r, rd, s)
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

# The level τ with Σᵢ (yᵢ - τ)₊ = a for the magnitudes yᵢ = rd(x[i]), i = 1:length(x): the clip level of the ℓ1 ball of radius a
# (the projection onto the ball is the soft threshold at max(τ, 0), τ the root). Condat's algorithm (Fast projection onto the
# simplex and the l1 ball, Math. Program. 158, 2016; expected O(n)) selects a small set of stored entries, or hands over to
# Michelot's algorithm (J. Optim. Theory Appl. 50, 1986) on all the entries when many entries become candidates. Both then run
# the same iteration, `_l1ball_iterate`: Newton's method on the convex function g(τ) - a, g(τ) = Σᵢ (yᵢ - τ)₊, from a level
# below the root or within ε₀ above it (claim 2), in the residual form. A pass at a level τ sums the excesses yᵢ - τ of the
# entries above τ, and the next level is τ + (excess - a)/(number of entries above τ). `rd` maps an entry to its magnitude in T
# = eltype(buf): `_AbsRead{T}()`, or the scaled read of the overflow path. `buf` is a scratch array whose length is at least
# length(x), indexed linearly.
#
# Contract. The caller (`_linf_prox!`) establishes with its pass over the magnitudes that the yᵢ are finite and nonnegative (a
# sum that is neither NaN nor infinite certifies finite mapped entries), that n ≥ 1, and that 0 ≤ a < s, where s is the sum of
# that pass and is at most floatmax(T)/2. The derivation below also uses the hypothesis n u ≤ 1/4, u = eps(T)/2 (n eps(T) ≤
# 1/2), which the caller does not check: the Float32 inputs with more than 2^22 entries do not satisfy it. Under these
# conditions the level returned is a number of type T, at most the largest entry M, within 4.5 n u M + 2 n δ of the exact level
# τ* (the root of g(τ) = a, and M for a = 0), with δ = nextfloat(zero(T)). The root is negative when a exceeds the sum of the
# entries, which the rounding of s allows; the level returned is then at least -ζ (claim 2; about 2 n u M), and the caller
# clamps it below at zero, as the exact prox clips at max(τ*, 0) (the clamp does not increase the distance between two levels).
# The accuracy statements are for n ≥ 1; the function returns zero for n = 0. The statements for each of the four returns of the
# iteration are in the claims below; they do not assume that the set of entries that the iteration last used is the set of
# entries above the exact level. Outside the conditions (finite yᵢ, n ≥ 1, any a and any start of the iteration) the iteration
# ends (claim 1) and returns a value that is not NaN and at most the largest entry that it reads (M; on Condat's path, the
# largest stored entry): each return is that largest entry, or a level that has an entry above it, or a step t < hi. A start of
# -Inf returns -Inf: with a = +Inf the first pass returns its level (r = Inf ≤ a) before a step is computed, and with any other a
# the first step is NaN and the last guard returns the start. With a = +Inf and finite sums the starts of `_l1ball_threshold!`
# are -Inf (ρ and the plain levels are -Inf, and the test of the level for an early handover, -Inf > -Inf, is false), so it
# returns -Inf. A start that is NaN or +Inf returns the largest entry; the starts of the callers are finite when s is. A pass
# sum that overflows gives the step +Inf, and then no entry is above the level. The bounds are not claimed there.
#
# Notation. The sums defining g are exact here. k(τ) = #{i : yᵢ > τ}, γ_k = k u/(1 - k u). The function g is convex, piecewise
# linear and nonincreasing, with the slope -k(τ) to the right of τ, and strictly decreasing where positive, so that τ* < M is
# unique for a > 0. Facts used:
#  (G1) g(τ + d) ≥ g(τ) - k(τ) d for all real d (the slope to the right of τ is a subgradient).
#  (G2) If τ′ ≥ τ and no entry lies in (τ, τ′], then g(τ′) = g(τ) - k(τ)(τ′ - τ).
#  (G3) For a nonempty subset S of the entries, g(τ) ≥ Σ_S (yᵢ - τ), so the level L_S = (Σ_S yᵢ - a)/|S| has g(L_S) ≥ a and L_S
#       ≤ τ*.
#  (G4) With K = #{i : yᵢ ≥ τ*} ≥ 1: for τ ≤ τ*, K (τ* - τ) ≤ g(τ) - a ≤ k(τ)(τ* - τ); for τ ≥ τ*, a - g(τ) ≥ k(τ)(τ - τ*); and
#       a = g(τ*) ≤ K (M - τ*).
# Rounding. A sum or difference is fl(z) = z(1 + ε) with |ε| ≤ u (exact when the result is subnormal), and |fl(z) - z| ≤ u
# |fl(z)|. A product or quotient is fl(z) = z(1 + ε) + η with |ε| ≤ u, |η| ≤ δ/2 and ε η = 0 (η = 0 unless the result is
# subnormal). fl is monotone, and r - a is exact when a/2 ≤ r ≤ 2a. A pass at a level τ (`_l1ball_excess`) returns the count k =
# k(τ) of the entries above τ, the smallest of them, hi, and the sum r of the floats fl(yᵢ - τ) over them, in any order. Each
# term goes through at most k - 1 additions, so
#  (P) r = Σ_{yᵢ>τ} (yᵢ - τ)(1 + θᵢ) with 1 - k u ≤ 1 + θᵢ ≤ 1/(1 - k u), hence (1 - k u) g(τ) ≤ r ≤ g(τ)/(1 - k u) and |r -
#      g(τ)| ≤ γ_k g(τ).
# The entries that are not above τ are at most τ, and those above τ are at least hi > τ. The subtraction of a term is exact when
# its result is subnormal, so the subnormal spacing δ enters a step through its quotient and its sum. The vectorized pass (Float32
# and Float64) takes hi as the minimum over the bit patterns; the plain pass (the other types, and Condat's path with at most four
# stored entries, read through `_plain_read`) sums in the order of its loop, which (P) covers, and takes hi by comparisons, which
# for entries that are nonnegative and not NaN, as these are, is the same number as the minimum over the bit patterns.
#
# The iteration, from a level τ (`_l1ball_iterate`): a pass at τ gives (r, k, hi); then
#   (a) k = 0: return the largest entry                      (no entry above the level)
#   (b) r ≤ a: return τ                                      (the level of the pass)
#   with t = fl(τ + fl(fl(r - a)/k)) ≥ τ, the Newton step τ + (r - a)/k:
#   (c) r - a ≤ a/2 and t < hi: return t                     (the certified step: no entry lies in (τ, t], and the pass measured the
#                                                              excess within a/2 of the radius)
#   (d) !(t > τ): return τ                                   (the step vanished: under the contract t ≥ τ, so this is t = τ, the level does
#                                                              not move; t < hi holds, so (c) did not return, and r - a > a/2)
#   Otherwise τ ← t and the pass is repeated (the step is taken: it is larger than certified, or it crosses an entry). These are
#   the four returns; the list is closed.
#
# Claim 1 (the levels increase; the loop ends). A level that is replaced is replaced by a larger one (the guard of the last
# return is !(t > τ): under the contract t ≥ τ, so it means t = τ; outside the conditions it also returns when t is unordered,
# as NaN after a start of -Inf with a ≠ +Inf), and the level of a pass is below M (k ≥ 1). A strictly increasing sequence of
# floats ends, so the loop ends for finite entries, for any n, a and start. A step that crosses an entry (t ≥ hi) leaves at most
# k - 1 entries above the next level. A step with t < hi crosses nothing, so by (G2) the next excess is g(t) = g(τ) - k(t - τ),
# and with the roundings of claim 3 it is within γ_k g(τ) + (2u + u²)|r - a| + k u|t| + kδ of a; such a step multiplies g - a by
# about γ_k + 2u when g(τ) is large against a, and a second one in a row needs the noise of the pass to be at least a/2. A bound
# on the number of passes is not proved; the accuracy sweep reports the largest count.
#
# Claim 2 (overshoot). Let ζ = 2(1 + u)² γ_{n-1} M + δ. The level of each pass is at least -ζ (claim 4 for Condat's start; the
# start of the handover, max(ρ, fl(fl(s - a)/n)), is at least its plain level fl(fl(s - a)/n), whose sum s is the sum of the
# caller when the caller gives it, and then a < s makes that level nonnegative; `_l1ball_michelot`, and the handover without a
# caller's sum, form their own sum, which is at least (1 - γ_{n-1}) Σ yᵢ, and with a < s_caller ≤ (1 + γ_{n-1}) Σ yᵢ the plain
# level fl(fl(s - a)/n) is at least -2(1 + u)² γ_{n-1} M - δ/2, and nonnegative when its sum is not below a). The return (a) is
# at most the level of its pass, and (b) is that level, so both are bounded by the overshoot of the level of a pass over τ*,
# which is at most ε = max(ε₀, ε₁), with
#   the start:  τ₀ - τ* ≤ ε₀ = M[(2u + u²) + (1 + u)² γ_{n-1}] + δ/2,
#   a step:     t - τ* ≤ ε₁ = (Γ (M + ζ) + u M + δ/2)/(1 - u),  Γ = max(γ_n + (2u + u²)/(1 - n u), (2 + 2u + u²) γ_n).
# Proof for the plain start τ₀ = fl(fl(s - a)/n), s = Σ yᵢ (1 + θ_s), |θ_s| ≤ γ_{n-1} (the sum of the caller, or the sum that
# `_l1ball_michelot` or the handover forms): s - a ≥ 0 (otherwise τ₀ ≤ 0, with no overshoot beyond -τ* ≤ γ_{n-1} M), and τ* ≥
# (Σ yᵢ - a)/n by (G3) with S = all the entries; s - a ≤ (Σ yᵢ - a) + γ_{n-1} Σ yᵢ, so τ₀ ≤ (1 + u)² [(Σ yᵢ - a) + γ_{n-1} Σ
# yᵢ]/n + δ/2, and Σ yᵢ - a ≤ Σ yᵢ ≤ n M. For Condat's start, max(ρ, plain level of the final candidates V), the plain level of
# V is bounded in the same way (S = V, |V| ≤ n, a negative level has no overshoot beyond -τ* ≤ γ_{n-1} M), and ρ ≤ τ* + γ_{n-1}
# M + δ/2 ≤ τ* + ε₀ by claim 4. The start of the handover, max(ρ, plain level of all the entries), is bounded by the same two
# arguments: its plain level is a plain start (the sum of the caller or of the handover), and ρ, the level of Condat's pass at
# the handover, is at most τ* + ε₀ by claim 4 (ii).
# Proof for a step from a level τ′ with the pass (r′, k′), r′ > a, g′ = g(τ′), t = fl(τ′ + d), d = fl(fl(r′ - a)/k′) ≥ 0: d ≤
# (r′ - a)(1 + u)²/k′ + δ/2, t ≤ τ′ + d + u|t|, and g′/k′ ≤ M - τ′ ≤ M + ζ (the k′ terms of g′ are at most M - τ′). If g′ ≥ a,
# the Newton point N = τ′ + (g′ - a)/k′ has g(N) ≥ a by (G1), so N ≤ τ* (for a = 0, N = τ′ + g′/k′ is the mean of the k′ entries
# above τ′, so N ≤ M = τ* directly); with r′ - a ≤ (g′ - a) + γ_{k′} g′ and r′ ≤ g′/(1 - k′u), t - τ* ≤ t - N ≤ [γ_{k′} g′ + (2u
# + u²) g′/(1 - k′u)]/k′ + δ/2 + u|t|. If g′ < a < r′, then a - g′ < r′ - g′ ≤ γ_{k′} g′ and, by (G4), τ′ - τ* ≤ (a - g′)/k′ ≤
# γ_{k′} g′/k′; with r′ - a < γ_{k′} g′, t - τ* ≤ (τ′ - τ*) + d + u|t| ≤ (2 + 2u + u²) γ_{k′} g′/k′ + δ/2 + u|t|. Both cases are
# at most Γ (M + ζ) + δ/2 + u|t|; for t > 0 solve for t (τ* < M), and for t ≤ 0 the overshoot is at most -τ* ≤ γ_{n-1} M.
#
# Claim 3 (the four returns). Let k be the count of the pass at the return and E = g(τ̂) - a the excess of the returned level
# τ̂.
#  (a) No entry above the level: τ̂ = M, the largest entry (Condat's buffer holds it, claim 4), g(M) = 0, E = -a. Since M ≤ τ,
#      τ̂ - τ* ≤ ε by claim 2, and τ̂ ≥ τ* because g(τ*) = a > 0 = g(M) (for a = 0, τ̂ = M = τ*).
#  (b) The level of the pass, r ≤ a: by (P), g(τ̂) ≤ a/(1 - k u), so E ≤ γ_k a; τ̂ - τ* ≤ ε by claim 2; and if τ̂ < τ*, then by
#      (G4) τ* - τ̂ ≤ E/K ≤ γ_k a/K ≤ γ_k (M - τ*).
#  (c) The certified step: the test fl(r - a) ≤ fl(a/2) gives r ≤ 2a (otherwise fl(r - a) ≥ a > fl(a/2)), so r - a is exact
#      (Sterbenz) and r - a ≤ fl(a/2) ≤ a/2 + δ/2; with r > a, a < r ≤ 1.5 a + δ/2. No entry lies in (τ, t], so the k entries
#      above t are those above τ, and (G2) gives g(t) - a = (g(τ) - r) + (r - a) - k (t - τ), where k (t - τ) = (r - a)(1 + ε₂)
#      + k η + e with |ε₂| ≤ u, |η| ≤ δ/2, |e| ≤ k u|t|, |g(τ) - r| ≤ γ_k g(τ) and g(τ) ≤ r/(1 - k u) ≤ (1.5 a + δ/2)/(1 - k u)
#      (the δ/2 is the rounding of a/2). Hence
#         |g(t) - a| ≤ E_c = (1.5 γ_k/(1 - k u) + u/2) a + k u|t| + k δ,
#      for any level of the pass relative to τ* (the proof does not use that the set of entries is the final one). For the
#      level, by (G4): if t > τ*, t - τ* ≤ E_c/k; if t ≤ τ*, τ* - t ≤ E_c/K ≤ [1.5 γ_k/(1 - k u) + u/2](M - τ*) + (k/K)(u|t| +
#      δ).
#  (d) The step vanished, t = τ < hi, so (c) failed on fl(r - a) > fl(a/2); rounding is monotone, so r - a > a/2. The step is at
#      most the half-spacing at τ, d ≤ u|τ|, and d ≥ (r - a)(1 - u)²/k - δ/2, so (r - a)(1 - u)² ≤ k (u|τ| + δ/2); with g(τ) ≤
#      r/(1 - k u) and a < 2(r - a),
#         g(τ̂) - a ≤ E_v = (1 + 2 k u)/(1 - k u) · k (u|τ̂| + δ/2)/(1 - u)²,
#      Also g(τ̂) ≥ (1 - k u) r > (3/4)(3/2) a > a for a > 0 when k u ≤ 1/4 (for a = 0, g(τ̂) > 0 = a because k ≥ 1 and each
#      term is positive), and then τ* - τ̂ ≤ E_v/K by (G4).
# So at (c) and (d) the level equation holds within E_c or E_v for the set of entries above τ̂ (the radius form), without the
# assumption that this is the set above τ*; at (a) and (b) the level is above τ* by at most ε, with g(τ̂) ≤ a (1 + γ_k) (the
# level form).
#
# Claim 4 (Condat's pass). The pass keeps a level ρ: ρ = prevfloat(y₁ - a) at the start; at a restart (the entry y is above ρ
# and fails the join test s > (y - a) nv, so that the candidates start anew at y), ρ ← max(ρ, prevfloat(y - a)); when an entry
# joins the nv candidates of running sum s, ρ ← max(ρ, (s (1 - (nv + 4) u) - a)/nv). An entry that is not above ρ is not stored.
# (i) ρ does not decrease, so an entry discarded at any time is at most the final ρ. (ii) ρ ≤ τ* + γ_{n-1} M + δ/2 ≤ τ* + ε₀,
# ρ ≤ τ* + δ/2 when ρ > 0, and ρ is at most the largest stored entry, after each step of the pass (so also when the pass hands
# over, see `_l1ball_threshold!`). ρ is the largest of the values below, so it suffices to bound each of them. Since
# prevfloat(fl(z)) < z for a real z, the first two values are below y - a ≤ τ* ((G3) with S = {y}) and below the entry y just
# stored. For the join value, f = 1 - (nv + 4) u is exact (nv u ≤ 1/4) and at
# most 1, so P = fl(s f) ≤ s. If P - a < 0, the value is at most 0, and 0 ≤ τ* + γ_{n-1} M because τ* ≥ -γ_{n-1} M (if τ* < 0,
# then τ* = (Σ yᵢ - a)/n and a < s ≤ (1 + γ_{n-1}) Σ yᵢ). Otherwise, if s is subnormal, the sum is exact, P ≤ s = Σ_V yᵢ, the
# difference (at most P) is exact, and the quotient has an absolute error of at most δ/2: the value is at most L_V + δ/2. If s
# is normal, s ≤ (1 + γ_{nv-1}) Σ_V yᵢ, so that P ≤ (1 + u)(1 + γ_{nv-1}) f Σ_V yᵢ + δ/2 ≤ (1 - 4u) Σ_V yᵢ + δ/2 ≤ (1 - 2u) Σ_V
# yᵢ (the term δ/2 occurs when s f is subnormal, and then 2u Σ_V yᵢ ≥ δ/2), and the difference and the quotient each have a
# relative error of at most u (or the quotient an absolute error of at most δ/2): the value is at most (1 + u)² ((1 - 2u) Σ_V yᵢ
# - a)/nv + δ/2 ≤ L_V + δ/2. With P - a ≥ 0 the value is therefore at most L_V + δ/2 ≤ τ* + δ/2 ((G3) with S = V), and at most
# the largest entry of V (L_V ≤ the mean of V, and no float lies in (max V, max V + δ/2]). A positive value is not a join value
# with a negative numerator, so it is at most τ* + δ/2. The value can exceed L_V: in Float32, x = [δ, …, δ, 1] with 25 entries
# δ and a = 1/2 gives -0.02f0 at 25 candidates, above L_V + δ/2 by about 4.5e-10 (a negative numerator, with s subnormal and a
# normal), and x = [4δ, 4δ, 4δ], a = δ gives the value 4δ at the second entry, above L_V = 3.5δ (within L_V + δ/2). An entry
# that is not stored when it arrives is at most ρ, hence at most the largest stored entry, so
# the largest stored entry is M. (ρ = M occurs in the subnormal range, as for x = [4δ, 4δ, 4δ], a = δ; the result is then M by
# (a).) (iii) The iteration starts at max(ρ, plain level of the final candidates) ≥ ρ and its levels increase, so each unstored
# entry is at most the level of each pass, and the stored entries (buf[1:nt+nv]) give the same g, k and hi as all the entries at
# each level that the iteration visits, and an excess r that satisfies (P) for the same set of terms (the order of the sum
# differs): claims 2 and 3 are statements about all the entries. (iv) A start below zero is at least ρ ≥ -δ after a restart (for
# y - a < 0 the product (y - a) nv is negative and the join test holds, so a restart has fl(y - a) ≥ 0). Without a restart no
# entry was discarded (a negative ρ discards no nonnegative entry), so the candidates are all the entries and s is their
# sequential sum; a < s_plain ≤ (1 + γ_{n-1}) Σ yᵢ gives s - a ≥ -2 γ_{n-1} Σ yᵢ, and the start is at least -ζ.
#
# Claim 5 (consequences). (i) The level. For n u ≤ 1/4 each bound of claims 2 and 3 on |τ̂ - τ*| is at most 4.5 n u M + 2 n δ.
# Since a < s ≤ (1 + γ_{n-1}) Σ yᵢ, a ≤ (1 + γ_{n-1}) n M (and a = g(τ*) ≤ n M when τ* ≥ 0). The bounds are: the overshoot ε;
# below τ*, γ_n (M - τ*) at (b), [1.5 γ_n/(1 - n u) + u/2](M - τ*) + (k/K)(u|t| + δ) at (c), with k/K ≤ n, and E_v at (d); above
# τ* at (c), E_c/k, which with a ≤ (1 + γ_{n-1}) n M is at most (n u [1.5/(1 - n u)² + 1/2] + u)(1 + γ_{n-1}) M + δ. For τ* < 0
# (a > Σ yᵢ) the lower-side bounds use that τ̂ ≤ τ* < 0 gives k = K = n and |τ̂| ≤ ζ, and M - τ* ≤ (1 + γ_{n-1}) M: (b) is at
# most γ_n (1 + γ_{n-1}) M, and (c) at most [1.5 γ_n/(1 - n u) + u/2](1 + γ_{n-1}) M + u ζ + δ. The largest coefficient, below
# 4.45, is that of the overshoot at n u = 1/4; at (c) above τ* it is below 4.23, and at (c) below τ* below 3.67 (3.56 for τ* <
# 0); for small n u the coefficients are about 2 + 1/n above τ* (4 at n = 1) and at most 3 below. The additive terms are at most
# 2 n δ. (ii) The distance. For τ ≥ 0 put d(τ) = (Σ min(yᵢ, τ)²)^(1/2), the norm of the clip at τ. The exact prox clips at τ₊ =
# max(τ*, 0), with the norm D(a) = d(τ₊), which is 0 for a ≥ Σ yᵢ. For 0 < a < Σ yᵢ, dD/da = -τ*/D with D ≥ τ*, and D(a) → 0 as
# a → Σ yᵢ, so D is 1-Lipschitz on [0, ∞) (it is 0 beyond Σ yᵢ), and d(τ) = D(g(τ)) for 0 ≤ τ < M. On the path of all the
# entries (`_l1ball_michelot`, and the handover of Condat's pass) with a nonnegative start (all levels are then ≥ 0, so the prox
# clips at τ̂ itself; the start of the handover is nonnegative when the caller gives its sum, and without that sum it can be
# negative, a case not covered here): at (c) and (d), |d(τ̂) - D(a)| ≤ |g(τ̂) - a| ≤ E_c or E_v. At (b) with τ̂ ≤ τ* (so τ* ≥ τ̂
# ≥ 0) it is at most g(τ̂) - a ≤ γ_k a. Otherwise ((a), and (b) above τ*), τ̂ ≥ τ₊ and |d(τ̂) - D(a)| ≤ √K′ (τ - τ₊) ≤ √K′ (τ -
# τ*), with τ the level of the last pass and K′ = #{i : yᵢ > τ₊} ≤ n. The K′ entries above τ₊ ≥ 0 give √K′ τ₊ ≤ ‖y‖₂, and the k
# entries above a level ℓ ≥ 0 give k ℓ ≤ √k ‖y‖₂. This gives, for the plain start:
# τ₀ - τ* ≤ c₀ Σ yᵢ/n + δ/2 with c₀ = 2u + u² + (1 + u)² γ_{n-1}, and Σ yᵢ ≤ √n ‖y‖₂, so √K′ (τ₀ - τ*) ≤ c₀ ‖y‖₂ + √n δ/2. The
# start of the handover is that plain level, or ρ when ρ is above it; with the caller's sum the plain level is nonnegative, so
# that such a ρ is positive. A nonnegative start ρ is 0, or positive and then at most τ* + δ/2 by claim 4 (ii) (a positive ρ is
# not a join value with a negative numerator, and the other values are at most τ* + δ/2); in both cases √K′ (ρ - τ₊) ≤ √n δ/2,
# within the term of the plain start. The bound ρ ≤ τ* + γ_{n-1} M + δ/2 of claim 4 (ii) alone would not give this form. For a
# step from τ′ with g′ ≥ a (then K′ ≤ k′, as τ′ ≤ τ* ≤ τ₊, and g′ ≤ √k′ ‖y‖₂): √K′ (t - τ*) ≤ ([γ_n + (2u + u²)/(1 - n u) + u]
# ‖y‖₂ + √n δ/2)/(1 - u); for a step from τ′ with g′ < a < r′ (then γ_{k′} g′/k′ < u a/(1 - k′u)): √K′ (t - τ*) ≤ ((2 + 2u + u²)
# √n u a/(1 - n u) + u ‖y‖₂ + √n δ/2)/(1 - u). At (c), E_c ≤ (1.5 n u/(1 - n u)² + u/2) a + √n u ‖y‖₂ + n δ, and at (d), E_v ≤
# (1 + 2 n u)(√n u ‖y‖₂ + n δ/2)/((1 - n u)(1 - u)²). For n u ≤ 1/4 all of these are at most [(n + 3) ‖y‖₂ + (1.5 n + 0.5)(1 +
# u)² a] u/(1 - n u)² + 2 n δ, to first order (n + 3) u ‖y‖₂ + (1.5 n + 0.5) u a (the first-order form drops 1/(1 - n u) from
# γ_n, 1/(1 - n u)² at (c), and 1/(1 - u) and (1 + u)²; at n u = 1/4 the factor 1/(1 - n u)² is 16/9: about (4/9) ‖y‖₂ + (2/3) a
# for large n). On Condat's path a start at the plain level of V keeps a factor √(K′/nv), because the entries above τ₊ need not
# be candidates; the distance form is not proved there.
#
# Range. Put S = Σ yᵢ, F = floatmax(T) and θ = (1 + u)^n - 1 ≤ 0.29 (n u ≤ 1/4). The sums are of nonnegative terms, so a partial
# sum of a pass is at most (1 + θ) times the exact sum of its terms, and the sum s of the full pass is at least (1 - θ) S: S ≤
# s/(1 - θ) ≤ F/[2(1 - θ)] < 0.705 F, and M ≤ s ≤ F/2. At a level τ ≥ 0 the exact pass sum g(τ) is at most S, so a computed pass
# sum is at most (1 + θ) S < 0.91 F, and so are the sequential sums of Condat's pass. At a negative level the exact pass sum is
# g(τ) = S - n τ, which does not increase as the levels increase, so it is at most its value at the start. On Condat's path, if a
# restart occurred and the start is negative, it is at least -δ, so g ≤ S + n δ. Otherwise the start is at least a plain level τ
# = fl(fl(s_V - a)/n) whose sum s_V is at least (1 - θ) S: on Condat's path without a restart no entry was discarded, and s_V is
# the sum of the candidates; for `_l1ball_michelot` and the handover s_V is the sum that they use (the start of the handover is
# max(ρ, τ) ≥ τ, and τ is nonnegative when s_V is the caller's sum). A negative τ has |τ| ≤ (1 + u)² (a - s_V)/n + δ/2, so g at
# the start is at most g(τ) ≤ S + (1 + u)² (a - s_V) + n δ/2 ≤ (1 + u)² (a + θ S) + n δ/2 ≤ (1 + u)² F/[2(1 - θ)] + n δ/2 (a <
# s ≤ F/2). In both cases a computed pass sum is at most (1 + θ) g < 0.91 F for u ≤ 2^-11, the terms n δ being far below the
# margin. The updates are finite: with C = (1 + u)²/(1 - n u) ≤ 1.335, t = τ + d, d = fl(fl(r - a)/k) ≤ (1 + u)² (r - a)/k + δ/2
# and r ≤ g/(1 - k u), so t ≤ τ + C g/k + δ/2 ≤ C M - (C - 1) τ + δ/2 ≤ C M + (C - 1) ζ + δ/2 < 0.79 F before the last rounding
# (M ≤ F/2, ζ ≤ 0.67 M + δ), and the rounding of a number below 0.79 F is finite; the levels are at least -ζ.
#
# Why an overflowing (y - a) nv still decides correctly. The test of Condat's pass is s > (y - a) nv, the exact test (s + y -
# a)/(nv + 1) > y - a. With y ≤ floatmax/2 (a computed sum of nonnegative terms is at least each term, and the sum of the pass
# is at most floatmax/2) the factor y - a is finite, and the product with nv ≥ 1 overflows to +Inf, which needs (y - a) nv >
# floatmax; since s is a sum of some of the yᵢ it is below floatmax (see above), so the exact test is false, as the comparison s
# > Inf is. A product that overflows to -Inf has y - a < 0 and the exact test (s ≥ 0 > -∞) is true, as s > -Inf is.

# Condat's pass. The stored entries are buf[1:nt] (the candidates that restarts have displaced) and buf[nt+1:nt+nv] (the nv
# candidates, of running sum s); the level ρ and the rules that update it are in claim 4. The entries after the first are read in
# groups of four, with one comparison with ρ for a group, and the last one to three entries (when n - 1 is not a multiple of four)
# one by one. The count nv is tested after each group in which an entry is above ρ, and not in the loop of the last entries: an
# input of at most 4 entries is not tested, and one of at most 8 entries is not handed over (after its one group of four the count
# is at most 5). A test compares the count with the count in force (`_condat_many`), cap = max(8, n ÷ 8) at first. A test that
# finds the count above cap2 = max(1024, n ÷ 8) while cap2 is in force hands over to Michelot's algorithm, on all the entries
# (the count can then be up to seven above cap2, see below). A test that finds it above cap while cap < cap2 hands over when ρ is
# at most the plain level (s_all - a)/n of all the entries, s_all being the sum of the caller's pass, and otherwise puts cap2 in
# force; without that sum (`nothing`) it puts cap2 in force. For n ≥ 8192 the two counts are equal. The level is tested at most
# once, at that test (not for n ≥ 8192, not without the caller's sum, and not when the count does not exceed cap): ρ does not
# decrease (claim 4 (i)), so a level that has passed the plain level stays above it, and a test of the level at the later tests
# would give the same answer. The test is ρ n > s_all - a in floating point (an overflow of ρ n included); either outcome is
# covered by the claims, so it decides cost only. A restart resets nv, so restarts are not counted against cap or cap2 (that
# costs time, not accuracy).
#
# A step stores its entry before the count is tested, so at a test the count can exceed the count in force by up to four. When
# cap is 1021 to 1023 (within four of cap2 = 1024) the first test can leave the count up to three above cap2, and the test at
# cap2 can then find it up to seven above. The stored entries are distinct entries of the argument, so a buffer of the length of
# the argument suffices, as the callers give.
#
# The counts trade one cost for another: an entry that joins the candidates costs a division, which the comparison of the next
# entries waits for, and a pass of the iteration over all the entries is (for Float32 and Float64) a vectorized loop with a few
# operations per entry. A count above cap while ρ is at most the plain level of all the entries, where the handover would start,
# is taken as a sign that the pass would join many more entries. Once ρ has passed that level, a handover would start at ρ
# itself, and the pass continues: the entries that ρ discards cost one comparison per group of four. This is a heuristic, and on
# some inputs the handover costs more than the rest of the pass would.
#
# Without a handover the iteration runs on the stored entries from max(ρ, (s - a)/nv), the plain level of the final candidates.
# With at most four stored entries it reads them through `_plain_read`, which selects the plain loop of `_l1ball_excess` in
# place of the vectorized loop, whose fixed cost per pass is not repaid on so few entries. On the handover it runs on all the
# entries from max(ρ, (s_all - a)/n) (`_l1ball_handover`), with the sum s_all of the caller's pass when the caller gives it
# (the callers do: one pass over the argument less) and with the sum of the entries formed there when it does not.
function _l1ball_threshold!(buf::AbstractArray{T}, x, a::T, rd::F, s_all::S = nothing) where {T, F, S<:Union{Nothing, T}}
  n = length(x)
  n == 0 && return zero(T)
  u = eps(T) / 2
  @inbounds y1 = rd(x[1])
  @inbounds buf[1] = y1
  nt, nv = 0, 1
  s = y1
  ρ = prevfloat(y1 - a)
  cap = max(8, n ÷ 8)
  cap2 = max(1024, n ÷ 8)
  many = false
  j = 2
  @inbounds while !many && j + 3 <= n
    y₁, y₂, y₃, y₄ = rd(x[j]), rd(x[j + 1]), rd(x[j + 2]), rd(x[j + 3])
    if (y₁ > ρ) | (y₂ > ρ) | (y₃ > ρ) | (y₄ > ρ)
      nt, nv, s, ρ = _condat_step!(buf, y₁, nt, nv, s, ρ, a, u)
      nt, nv, s, ρ = _condat_step!(buf, y₂, nt, nv, s, ρ, a, u)
      nt, nv, s, ρ = _condat_step!(buf, y₃, nt, nv, s, ρ, a, u)
      nt, nv, s, ρ = _condat_step!(buf, y₄, nt, nv, s, ρ, a, u)
      if nv > cap
        cap, many = _condat_many(ρ, n, a, s_all, cap, cap2)
      end
    end
    j += 4
  end
  @inbounds while !many && j <= n
    nt, nv, s, ρ = _condat_step!(buf, rd(x[j]), nt, nv, s, ρ, a, u)
    j += 1
  end
  many && return _l1ball_handover(x, n, a, rd, s_all, ρ)
  lv = nv == 1 ? s - a : (s - a) / nv                    # the plain level of the candidates (the division by 1 is skipped)
  nt + nv <= 4 && return _l1ball_iterate(buf, nt + nv, a, _plain_read, ifelse(lv > ρ, lv, ρ))
  _l1ball_iterate(buf, nt + nv, a, identity, ifelse(lv > ρ, lv, ρ))
end

# The test when the count of candidates exceeds cap: it returns the count to test from then on and whether to hand over. At
# cap == cap2 (n ≥ 8192, or after an earlier test has returned cap2) it hands over. Otherwise it hands over when ρ has not passed
# the plain level of all the entries, and else it returns cap2 and the pass continues. It is not inlined, which keeps the loop
# of Condat's pass to the steps and one call.
@noinline function _condat_many(ρ::T, n::Int, a::T, s_all, cap::Int, cap2::Int) where {T}
  (cap == cap2 || !_level_passed(ρ, n, a, s_all)) && return cap, true
  cap2, false
end
# Whether ρ has passed the plain level (s_all - a)/n of all the entries, by the test ρ n > s_all - a. Without the caller's sum
# it is true, so that only cap2 applies then. The bound T <: Real keeps the two methods apart (with T = Nothing both would apply).
@inline _level_passed(ρ::T, n::Int, a::T, s_all::T) where {T<:Real} = ρ * n > s_all - a
@inline _level_passed(ρ, n, a, ::Nothing) = true

# The handover: the iteration on all the entries from max(ρ, (s - a)/n), the larger of the level ρ of Condat's pass and the
# plain level of all the entries, with s the sum of the caller's pass when it is given and the sum of the entries formed here
# when it is not (claim 2 bounds this start). The handover at cap is taken when ρ n ≤ s - a, so that it starts at the plain
# level up to a rounding of that test; at cap2 ρ can be the larger (after it has passed the plain level; for n ≥ 8192, where the
# level is not tested; and without the caller's sum, where the level is not tested either and the plain level of the sum formed
# here can be negative). It is not inlined, which keeps `_l1ball_threshold!` to Condat's pass and one call for the handover.
@noinline function _l1ball_handover(x, n::Int, a::T, rd::F, s::T, ρ::T) where {T, F}
  lv = (s - a) / n
  _l1ball_iterate(x, n, a, rd, ifelse(ρ > lv, ρ, lv))
end
@noinline function _l1ball_handover(x, n::Int, a::T, rd::F, ::Nothing, ρ::T) where {T, F}
  _l1ball_handover(x, n, a, rd, _linf_sum(x, rd, T), ρ)
end

# One entry y of Condat's pass, with the level ρ (claim 4): an entry not above ρ is not stored. Otherwise it joins the nv candidates when
# their plain level with y, (s + y - a)/(nv + 1), exceeds y - a, which is the test s > (y - a) nv, and then ρ becomes
# max(ρ, (s (1 - (nv + 4) u) - a)/nv) for the new candidates, u = eps(T)/2: the factor covers the roundings of the sum, the product, the
# difference and the quotient: with the exact level L_V of the candidates, the value is at most L_V + δ/2 when its numerator is nonnegative and
# at most 0 otherwise (claim 4 gives the consequences for the exact level of all the entries). If the test fails, the candidates
# stay in the buffer in front of y (they are counted in nt) and y starts them anew, with ρ ← max(ρ, prevfloat(y - a)). The maximum is taken
# with a comparison (the values are not NaN). It returns (nt, nv, s, ρ); the count nv is tested by the caller, after a group of four.
@inline function _condat_step!(buf::AbstractArray{T}, y::T, nt::Int, nv::Int, s::T, ρ::T, a::T, u::T) where {T}
  y > ρ || return nt, nv, s, ρ
  if s > (y - a) * nv
    nv += 1
    s += y
    @inbounds buf[nt + nv] = y
    c = (s * (1 - (nv + 4) * u) - a) / nv
    return nt, nv, s, ifelse(c > ρ, c, ρ)
  end
  nt += nv
  @inbounds buf[nt + 1] = y
  c = prevfloat(y - a)
  nt, 1, y, ifelse(c > ρ, c, ρ)
end

# One pass at the level τ over the first n entries rd(x[i]) (nonnegative): the sum r of the excesses y - τ of the entries above τ, their
# count k, and the smallest of them, hi (meaningless when k = 0, and not to be used then). For Float32 and Float64 it is one vectorized
# loop, in which the smallest entry is taken over the bit patterns, as `_linf_classify` takes its maximum (the patterns of nonnegative floats
# are ordered as the floats, and typemax of the integer type, the pattern of a NaN, is above them); other types, and the reader
# `_plain_read` (Condat's path with at most four stored entries), take a plain loop (`_excess_plain`).
# The loop counts in the integer type of the width of the entries (Int32 for Float32), so that its lanes have one width, when n fits in
# it (`_excess_narrow`), and in Int otherwise: for 2^31 entries or more a Float32 count in Int32 would wrap.
_excess_narrow(::Type{T}, n::Int) where {T<:Union{Float32, Float64}} = n <= typemax(_int_type(T))
@inline function _excess_loop(x, n::Int, τ::T, rd::F, ::Type{K}) where {T, F, K}
  U = _int_type(T)
  r = zero(T)
  k = zero(K)
  h = typemax(U)
  @inbounds @simd for i in 1:n
    y = rd(x[i])
    above = y > τ
    r += ifelse(above, y - τ, zero(T))
    k += above
    h = min(h, ifelse(above, reinterpret(U, y), typemax(U)))
  end
  r, Int(k), reinterpret(T, h)
end
@noinline _excess_wide(x, n::Int, τ::T, rd::F) where {T, F} = _excess_loop(x, n, τ, rd, Int)
@inline function _l1ball_excess(x, n::Int, τ::T, rd::F) where {T<:Union{Float32, Float64}, F}
  _excess_narrow(T, n) ? _excess_loop(x, n, τ, rd, _int_type(T)) : _excess_wide(x, n, τ, rd)
end
# The reader of the stored entries on Condat's path with at most four of them: the identity, whose type selects the plain loop
_plain_read(y) = y
_l1ball_excess(x, n::Int, τ::T, rd::typeof(_plain_read)) where {T<:Union{Float32, Float64}} = _excess_plain(x, n, τ, rd)
_l1ball_excess(x, n::Int, τ::T, rd::F) where {T, F} = _excess_plain(x, n, τ, rd)
function _excess_plain(x, n::Int, τ::T, rd::F) where {T, F}
  r = zero(T)
  k = 0
  hi = T(Inf)
  @inbounds for i in 1:n
    y = rd(x[i])
    if y > τ
      r += y - τ
      k += 1
      hi = min(hi, y)
    end
  end
  r, k, hi
end

# The largest of the first n entries, the return of the iteration when no entry is above the level; a separate function that is
# not inlined, as the return is rare.
@noinline function _l1ball_top(x, n::Int, rd::F, ::Type{T}) where {T, F}
  m = zero(T)
  @inbounds for i in 1:n
    y = rd(x[i])
    m = ifelse(y > m, y, m)
  end
  m
end

# The iteration of the header from the level τ, on the first n entries rd(x[i]): the four returns (a) to (d) in that order, the step
# τ ← t otherwise. It does not allocate for Float32 and Float64 entries.
function _l1ball_iterate(x, n::Int, a::T, rd::F, τ::T) where {T, F}
  while true
    r, k, hi = _l1ball_excess(x, n, τ, rd)
    k == 0 && return _l1ball_top(x, n, rd, T)                       # (a) no entry above the level
    r <= a && return τ                                              # (b) the level of the pass
    t = τ + (r - a) / k
    (r - a <= a / 2 && t < hi) && return t                          # (c) the certified step
    t > τ || return τ                                               # (d) the step vanished
    τ = t
  end
end

# Michelot's algorithm: the iteration of the header on all the entries, from the plain level (s - a)/n of all of them, with s their sum
# from `_linf_sum` (the vectorized pass of the caller). The handover of Condat's pass runs the same iteration from the larger of ρ and
# that plain level, with the sum of the caller's pass when the caller gives it (`_l1ball_handover`).
function _l1ball_michelot(x, a::T, rd::F, ::Type{T}) where {T, F}
  n = length(x)
  s = _linf_sum(x, rd, T)
  _l1ball_iterate(x, n, a, rd, (s - a) / n)
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
# A length n for which n + 1 is not a finite number of W (Float16 from n = 65519) is refused with an ArgumentError before any pass:
# the counts of the kernel (cnt, n + 1) are formed in W.
# Rules, in this order. v and x are classified in one pass each (_linf_classify, on abs(convert(W, ·))): sums sv, sx and
# largest magnitudes mv, m.
#   1. A NaN sum (isnan(sx) | isnan(sv)) gives NaN: for a type other than Float32 and Float64 the maximum of _linf_classify
#      does not carry a NaN, its sum does. Otherwise an infinite mv or m gives Inf.
#   2. An infinite c gives 0 at x = 0 and Inf at x ≠ 0. c = 0 gives ‖v‖₂.
#   3. Otherwise τ and the sum of squares of the residual, as below. An overflow of sv with finite entries goes to the scaled
#      path (_linf_scaled) unless an exit of rules 1 and 2 applies. An overflow of sx with a finite m does not: the point is a
#      zero point or a face by m, and the range is decided by sv.
#
# The ordinary path forms τ and the sum of squares in W. It uses no buffer, and allocates nothing for Array arguments of
# Float32 or Float64. The level comes from `_l1ball_iterate`, the iteration of the threshold of the ℓ1 ball (its four returns
# and claims 1 to 5 are in the header of that section), which reads nonnegative entries through a read.
#   Zero point: sv ≤ c gives 0. Otherwise τ = max(level, 0), with the level from `_l1ball_iterate` on the magnitudes |vᵢ| from the
#      plain first level (sv - c)/n of the sum sv of the classification pass (the start of `_l1ball_michelot`; the handover of the
#      prox starts at the larger of the plain level of its sum and radius and the level ρ of Condat's pass).
#      The conditions of that routine hold: the |vᵢ| are finite and nonnegative since sv is finite, 0 ≤ c < sv ≤ floatmax(W)/2
#      (the tests of this function), so the start is nonnegative and all the levels are, and n eps(W) ≤ 1/2, which is not
#      checked (as for the prox).
#   Face: one pass for S = Σ_J uᵢ, cnt = |J| and μ = min_J uᵢ, and the first level τ0 = (S - c)/cnt, in floating point. The
#      closed form uses τ = τ0 when the rounded comparison τ0 + b ≤ μ succeeds, with b = 3u|τ0| + 2u cnt mv, u = eps(W)/2,
#      and mv = max|vᵢ|. For the ordinary Float32 and Float64 path this certifies that the exact level is the exact first level
#      (see the rounded comparison below). In the subnormal range this comparison can accept without certifying support.
#      The face level can be negative. Otherwise the shifted values Lᵢ = max(uᵢ - τ0, 0) on J and 0 off J (the lazy _LinfShifted)
#      go to `_l1ball_iterate` from the level 0, and τ = τ0 + level. Level 0 of the Lᵢ is the first level τ0, so the
#      iteration continues Michelot's from there, with one pass less than a start from the sum of the Lᵢ, and no sum of them
#      is formed. The levels are nonnegative (claim 1), so the entries off J, which are 0, are not above any level that is
#      visited. For any σ ≤ τ, Σ_J (uᵢ - τ)₊ = Σ_J (max(uᵢ - σ, 0) - (τ - σ))₊, so the level of the Lᵢ for the radius c is τ - σ.
#      The shift is by τ0 and not by μ: with the shift by μ each Lᵢ has the size of |μ| when one face entry is far below the
#      level, and the level inherits an error of order a u |μ|. The computed τ0 can be above or below the exact first level
#      and the exact level, the case that the certificate does not exclude: the derivation does not assume that the entries
#      above the returned level are the entries above the exact level. The routine's condition 0 ≤ a < s is not established here
#      (no sum of the Lᵢ is formed, and Σ_J Lᵢ ≤ c is possible when τ0 is above the exact level); the derivation uses the
#      statements of the kernel that do not need it (claims 1 and 3, and the proofs of claims 2 and 3 for a step).
#      Range: when the certificate fails, fl(τ0 + b) > μ ≥ -mv, so τ0 + b > -mv (fl is monotone and fl(-mv) = -mv); for τ0 < 0
#      this gives |τ0| < (1 + 2.001 cnt u) mv/(1 - 3.001u) ≤ 1.6 mv (n u ≤ 1/4, the factor 1.001 covers the rounding of b).
#      Hence each Lᵢ ≤ mv + max(-τ0, 0) ≤ 2.6 mv, their exact sum is at most 2.6 cnt mv ≤ 2.6 n sv (mv ≤ sv, a computed sum
#      of nonnegative terms), and (n + 1) sv ≤ floatmax(W)/4 on the ordinary path (_linf_in_range) bounds it by
#      0.65 floatmax(W). A pass sum at a level ≥ 0 is at most (1 + θ) times the exact sum of the computed Lᵢ (θ = (1 + u)^n - 1
#      ≤ 0.29, and Lᵢ ≤ (1 + u) times its exact value), below 0.85 floatmax(W), and the step t ≤ C M with C = (1 + u)²/(1 -
#      n u) ≤ 1.335 and M = max Lᵢ ≤ 2.6 mv ≤ 0.33 floatmax(W) (mv ≤ sv ≤ floatmax(W)/8) is finite, as in the Range
#      paragraph of the kernel. The sum S is finite, and τ0 is finite unless S - c overflows (c close to floatmax(W) and S
#      negative), which the test isfinite(τ0) sends to the scaled path.
#   Squares: one loop for Q = Σ rᵢ², with rᵢ = min(|vᵢ|, τ) at the zero point (τ = Inf for c = 0, where rᵢ = |vᵢ|), and on a face
#   uᵢ clipped at τ on J and vᵢ off J. sqrt(Q) is returned when Q is finite and not below n floatmin/eps (_squares_normal).
#   Otherwise _linf_cold passes the lazy residual to `_residual_squares`, which scans it for its largest exponent and then sums
#   the squares scaled by it (as _l2_score_cold). It gives zero when all the residual entries are zero, and Inf for an entry
#   that is not finite.
#   Range: the scaled path is used when sv > floatmax(W)/2 at the zero point, when (n + 1) sv > floatmax(W)/4 on a face,
#   and when τ0 is not finite. That path forms τ in units of a power of two.
#
# Ordinary-path error. In the round-to-nearest relative-error model for Float32 and Float64, the absolute score error
# is
# at most (2n + √n + 5)uΛ to first order, with u=eps(W)/2, n u ≤ 1/4, and Λ=‖v‖₂+λ. The remainder is O_n(u²Λ) for
# fixed n.
# This estimate excludes additive subnormal-spacing effects and the scaled or cold branches. Both x and v are read in
# W,
# and the penalty λ is rounded once. Let δ=nextfloat(zero(W)).
#
# Write V = ‖v‖₂, c for the working penalty, m for the number of tied coordinates, and y for their sign-adjusted
# gradients. Off-face gradients are identical throughout. For signed data w define H_w(s) = Σ(wᵢ−s)₊, R_w(s) as the
# norm of the clipped residual, and D_w(c) as the exact distance. For c>0 there is a unique root p_w of H_w(p_w)=c.
# For an empty-radius ball take the ordinary norm separately. The radius form |R_w(s)−D_w(c)| ≤ |H_w(s)−c| follows
# from the Hausdorff distance between scaled simplexes. Also D_w is 1-Lipschitz in w, as distance to a fixed closed
# set. For any s and w,w′ the clipped-coordinate map is 1-Lipschitz, hence |R_w(s)−R_w′(s)|≤‖w−w′‖₂. The strict count
# K=#{wᵢ>p_w} yields |R_w(s)−D_w(c)|≤√K (s−p_w) for s≥p_w, including exact ties at p_w. All four statements also
# permit ordinary off-face coordinates in the norm.
#
#
# A backward perturbation of the shifted reads
#
# Let σ be the actual first level from the signed sum, e=(Σ yᵢ−c)/m its exact counterpart, aᵢ=(yᵢ−σ)₊, Lᵢ=fl(aᵢ),
# zᵢ=yᵢ+Lᵢ−aᵢ mathematically (these are real numbers). At every nonnegative t,
#
#     H_z(σ+t) = Σ(Lᵢ−t)₊.
#
# Off-face zeros used by the iteration make zero contribution for these t. The signed data z and their root exist also
# when p_z<σ. No assertion about the root of a padded nonnegative vector at a negative t is needed.
#
# Put A=‖a‖₂ and A₀=V+c/√m. The possibly signed first-level sum has error ≤γ_{m−1}‖y‖₁ with γ_j=ju/(1−ju). At most m−1
# additions of nonzero terms affect any summand. Division by m is by an exactly represented integer. The first-level
# bound below gives √m |σ−e|=O(mu V+u c/√m), ignoring δ. Exact centering at ȳ yields
#
#     A ≤ ‖y−σ1‖₂ ≤ (‖y−ȳ1‖₂²+c²/m)^(1/2)+√m |σ−e|
#       ≤ A₀+O(mu V+u c/√m).
#
# ‖L‖₂≤(1+u)A and ‖z−y‖₂≤uA. Together the two Lipschitz comparisons transfer a clipped distance for z to y at cost
# ≤2uA. This avoids transferring an active count between problems.
#
# The implementation adds σ to the returned t, rounded once. Let ℓ=σ+t and ℓ′=fl(ℓ). When |ℓ′−ℓ|≤u|ℓ|, the clipped
# values differ by at most u times the old clipped magnitude.
# For positive levels this follows by clipping the positive entries from above, and for negative levels by clipping
# their residual
# magnitudes from below. This contributes uD_y(c) to first order, since the comparisons below bound the exact residual
# error by
# O_n(uΛ). The sum of squares and its rounded root contribute (n/2+1)uD_y(c)+O_n(u²Λ), from one product and at most
# n−1 additions
# per term. Terms underflowing on squaring can contribute at most O(nδ) to the sum and hence O(u²) relative error to
# an accepted Q≥n floatmin/eps. Rounding λ to c costs at most uλ by the radius inequality, when the coefficient is
# rounded once. The total allowance beyond a kernel error for z, for a non-initial return, is therefore
#
#     2uA₀+(n/2+2)uD_y(c)+uλ,   D_y(c)≤V+c.
#
# In the budgets below replace c by λ in first-order terms, and ignore second-order terms as declared above.
#
#
# Certified and vanished steps (c) and (d)
#
# The local (P) pass bound and the local return inequalities for E_c and E_v of the shared kernel, apply to the
# rounded L. They depend on finite sums, n u≤1/4, the return comparisons, and positive c. They hold regardless of
# which side of the root the pass visits and require no precomputed total. At (c) the trial is strictly below the
# smallest active entry; at (d) the trial equals the current pass level, also below that entry. The k active entries
# all satisfy Lᵢ>t. Thus k t ≤√k ‖L‖₂ ≤√m ‖L‖₂. The local residual |H_z(σ+t)−c| is at most
#
#     (1.5k+0.5)u c+k u t  at (c), and  k u t  at (d)
#
# to first order. Apply the radius inequality and add all three rounding allowances just given. A common coefficient
# on V is n/2+√m+4, and on c it is 1.5m+n/2+4.5+2/√m. For n≥2 both are at most 2n+√n+5. Indeed 1.5m+2/√m increases for
# m≥1, and 2/√n≤√n when n≥2.
#
#
# Returns (a) and (b) after a step
#
# Let η=p_z−σ, possibly negative, and K=#{zᵢ>p_z}≥1. For η≥0 each of the K coordinates has Lᵢ≥η. Therefore √K
# max(η,0)≤‖L‖₂ in both sign cases. At a previous pass t′≥0 with k′>0 write g′=H_z(σ+t′) and r′ for its finite
# computed sum. A step requires r′>c. Its exact-Newton counterpart N=t′+(g′−c)/k′ satisfies N≤η, since σ+N=(Σ_{i
# active}zᵢ−c)/k′ is the subset level and H_z(σ+N)≥c. This holds for either sign of g′−c and for any computed active
# subset of the read data. The pass estimate is |r′−g′|≤γ_{k′}g′. Comparing the rounded trial t with this SAME N gives
#
#     t−η ≤ [γ_{k′}g′+(2u+u²)(r′−c)]/k′ + u|t|,
#
# up to an additive division-underflow term. Relative error in both subtraction and division gives the factor 2u+u².
# Addition error measured at its output gives u|t|. If the final pass returns its maximum (a), its returned t-hat is
# ≤t, and σ+t-hat is the maximum signed entry and thus ≥p_z. If it returns its pass level (b), t-hat=t. For t-hat≤η at
# (b), (P) directly yields 0≤H_z(σ+t-hat)−c≤γ_k c. Use the radius form in that case.
#
# In the remaining case t≥η, an overshoot of size d_t=√K(t−η) costs at most d_t in distance. Bound √K t ≤ ‖L‖₂+d_t. If
# g′≥c, monotonicity gives t′≤η and K≤k′. Cauchy–Schwarz gives g′≤√k′ ‖L‖₂, since each positive excess is ≤ its own
# Lᵢ. The first additive term of the last display multiplied by √K contributes at most k′u‖L‖₂ to first order. The
# second contributes at most 2u‖L‖₂, using r′≤(1+γ_{k′})g′ and √K/k′ ≤1/√k′. Absorb u d_t on the left. The overshoot
# costs ≤(k′+3)u‖L‖₂ to first order. If g′<c<r′, then r′−c≤γ_{k′}g′≤γ_{k′}c, so its factor 2u+u² contributes only
# second order. The first term contributes ≤√m u c, and the last ≤u‖L‖₂ to first order. The constants in the
# second-order terms are controlled by denominators 1−nu and 1−u. In particular no potentially discontinuous indicator
# is replaced by its limit.
#
# Adding transfer, final rounding and norm error yields, for the overshoot cases, a common coefficient on V of 1.5n+7,
# and on c of n/2+3+√m+5/√m. For g′<c this envelope follows directly from (2+1)/√m+√m≤5/√m+√m. For the (b) undershoot
# bound γ_k c add the same allowances, giving on c at most m+n/2+3+2/√m, which also lies below the target
# independently, and on V at most n/2+4.
#
# For n≥2, 1.5n+7≤2n+√n+5. Moreover √m+5/√m attains its maximum on [1,n] at an endpoint. At m=1 the full coefficient
# is n/2+9, and at m=n it is n/2+3+√n+5/√n. Each is at most 2n+√n+5 for n≥2. These inequalities use only m≤n.
#
#
# Initial returns, zero point, and singleton
#
# For first-pass (a) on the signed problem all aᵢ vanish, so σ≥max yᵢ>p_y. For first-pass (b) with σ≥p_y the same
# level bound applies: the subset-level property e≤p_y yields √K_y(σ−p_y)≤√K_y|σ−e|. With B₀=q|σ|+γ_{m−1}‖y‖₁/m and
# q=(2+u)u this is at most [qD_y(c)+γ_{m−1}V]/(1−q), since √K_y |σ|≤D_y(c)+√K_y |σ−e|. When first-pass (b) has σ<p_y,
# (P) and Lᵢ≥(1−u)aᵢ give H_y(σ)≤c/[(1−k u)(1−u)]. The resulting distance error is ≤(k+1)u c to first order. Neither
# return requires a positive shifted root. The certified closed form has the same B₀ distance bound, with K replaced
# by m. The certificate's validity in the ordinary range is shown below.
#
# At the zero point outside the computed ball, all hypotheses of #1004 claim 5(ii) hold, including the plain
# nonnegative starting level. Its nonnegative-radius distance bound is (n+3)u V+(1.5n+0.5)u c to first order. The
# ordinary norm costs at most (n/2+1)u V since the residual is clipped coordinatewise at a nonnegative level. The
# rounded early membership exit obeys D≤γ_{n−1}V by radial scaling onto the ball. Zero penalty uses just the norm.
# These bounds all fit the original coefficient, including rounding of λ.
#
# For n=m=1, write y for the signed input scalar and d=|y−c|. Its first level is σ=fl(y−c), giving |σ−(y−c)|≤u d. When
# the shift returns zero immediately, clipping at σ changes the exact residual by at most this amount. Otherwise a
# step requires L>c (only one summand), y>σ, and L=c+O(uΛ), since y−σ=c+O(u d). At any pass with r>c and t′<L,
# r=fl(L−t′). The trial is fl(t′+fl(r−c)), division by one being exact. Relative errors give |t−(L−c)|≤3uL+O(u²Λ),
# uniformly over the passes: |r−(L−t′)|≤uL, |fl(r−c)−(r−c)|≤u(1+u)L, and before the final rounding the trial is at
# most L+O(uL). At (c) and (d) the returned value is this trial. At (a) capping a trial that reached L can only reduce
# its overshoot. At (b) the pass level was a preceding trial. These estimates apply to every final return after a
# step. Replacing L by y−σ costs u c, the final addition costs u d, and the norm and penalty cost 1.5u d+uλ. The total
# is ≤7.5uΛ to first order, below the original coefficient of 8.
#
# The rounded closed-form comparison
#
# With M=max |vᵢ| and e=(Σyᵢ−c)/m, the signed summation, subtraction and division give
# |σ−e| ≤ B₀ = (2+u)u|σ| + γ_{m−1}‖y‖₁/m + (1+u)δ/2.
# The additive term accounts for division into the subnormal range. Sums and differences there are exact multiples of
# δ.
# The rounded margin b̂ is at least 2.99u|σ|+1.99umM−δ for binary32 and binary64. Each of the two products with u can
# lose
# δ/2, and mM is exact if subnormal. If fl(σ+b̂)≤μ then σ+b̂≤μ+uM by monotonicity, since |μ|≤M.
# Using γ_{m−1}≤(4/3)(m−1)u leaves b̂−B₀−uM ≥ 0.98u(|σ|+M)−(1.5+u/2)δ.
# Thus u(|σ|+M)≥2δ suffices for e≤μ, so precisely the all-active formula gives the exact level. The ordinary squared
# norm
# cannot pass its range test below this scale, where all residual squares round to zero. For tiny subnormals the
# comparison can
# accept without certifying exact support. The cold residual norm recovers squares from the computed level.

# Entry reads with the sign: convert(W, v), and the same scaled by 2^-k (the scaled path).
struct _ValRead{W} end
@inline (::_ValRead{W})(v) where {W} = convert(W, v)
struct _ScaledValRead{W}
  k::Int
end
@inline (g::_ScaledValRead{W})(v) where {W} = ldexp(convert(W, v), -g.k)

# The values Lᵢ = max(uᵢ - σ, 0) for the face J = {i : |xᵢ| == m} and 0 off J, where uᵢ = -sign(xᵢ) rd(vᵢ), as a vector of
# length(x) that `_l1ball_iterate` reads without a copy. rd is _ValRead{W}() or the scaled read, and the shift σ (the
# computed first level τ0 of the face) is in the units of rd.
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

# Rounded test for the closed-form level. The ordinary-range certificate and its subnormal qualification are given above.
# The range test gives cnt mv ≤ floatmax/4 for an ordinary face.
@inline function _linf_certified(τ0::W, cnt::Int, μ::W, mv::W) where {W}
  u = eps(W) / 2
  τ0 + (3u * abs(τ0) + 2u * (cnt * mv)) <= μ
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
    τ = max(_l1ball_iterate(v, n, c, _AbsRead{W}(), (sv - c) / n), zero(W))
    return _linf_finish(_linf_squares_zero(v, τ, W), v, x, m, τ)
  end
  S, cnt, μ = _linf_face_sums(v, x, m, _ValRead{W}(), W)
  τ0 = (S - c) / cnt
  isfinite(τ0) || return _linf_scaled(c, v, x, m, mv, W)
  τ = τ0
  if !_linf_certified(τ0, cnt, μ, mv)
    τ = τ0 + _l1ball_iterate(_LinfShifted(v, x, _ValRead{W}(), m, τ0), n, c, _AbsRead{W}(), zero(W))
  end
  _linf_finish(_linf_squares_face(v, x, m, τ, W), v, x, m, τ)
end

# The same out of the ordinary range, for finite entries and a finite positive c, in units of 2^k as `_linf_exceptional!` does
# for the prox: v is read as 2^-k vᵢ (the scaled reads), the level is 2^-k c (`_linf_scaled_radius`, formed from the significand
# and the exponent), and τ = 2^k τ′. k starts at ceil(log2 n) + 2, where the scaled magnitudes sum to at most floatmax/4 before
# rounding, and is raised until the scaled sum is in the range of the ordinary path (`_linf_in_range`); the pass that forms the
# scaled sum also gives the largest scaled magnitude mv′, which the certificate of the face needs. The loop ends: the scaled
# entries shrink as k grows, the scaled sum decreases to zero, and (n + 1)·0 = 0 satisfies the predicate because
# `_level_point` has refused the lengths for which n + 1 is not finite in W. Then the ordinary path's formulas hold in these
# units, with the same preconditions: the scaled sum is within the limits of `_linf_in_range`, c′ ≤ c/4 is finite (k ≥ 2, as
# _linf_exponent(n) ≥ 2), S′ is finite, the zero point has 0 ≤ c′ < sv′ when it does not return, and τ0′ = (S′ - c′)/cnt is finite as
# |S′| ≤ 1.29 · 1.41 Σ|vᵢ|′ ≤ 0.23 floatmax (n + 1 ≥ 2) and c′ ≤ floatmax/4; the range of the shifted values is as above, with mv′
# in place of mv. The scaling of an entry or of c can lose up to a subnormal spacing of W,
# as in the prox. τ is capped at the largest |vᵢ|, mv, which the exact τ does not exceed (it is below the largest uᵢ), so that
# the rounding of 2^k τ′ does not overflow; a τ below -floatmax is -Inf, and the cold path then gives Inf, as the distance
# is then above floatmax. A scaled level at least the scaled sum gives 0 at the zero point. This path sends the residual
# directly to _linf_cold, without testing an ordinary sum of squares.
@noinline function _linf_scaled(c::W, v, x, m::W, mv::W, ::Type{W}) where {W}
  n = length(x)
  zp = iszero(m)
  k = _linf_exponent(n)
  local sv′, mv′
  while true
    sv′, mv′ = _linf_classify(v, _ScaledAbsRead{W}(k), W)
    _linf_in_range(sv′, n, zp, W) && break
    k += 1
  end
  c′ = _linf_scaled_radius(W, one(W), c, k)
  if zp
    c′ >= sv′ && return zero(W)
    τ′ = max(_l1ball_iterate(v, n, c′, _ScaledAbsRead{W}(k), (sv′ - c′) / n), zero(W))
    τ = min(ldexp(τ′, k), mv)
    return _linf_cold(v, x, m, τ, W)
  end
  rd = _ScaledValRead{W}(k)
  S′, cnt, μ′ = _linf_face_sums(v, x, m, rd, W)
  τ0′ = (S′ - c′) / cnt
  τ′ = τ0′
  if !_linf_certified(τ0′, cnt, μ′, mv′)
    τ′ = τ0′ + _l1ball_iterate(_LinfShifted(v, x, rd, m, τ0′), n, c′, _AbsRead{W}(), zero(W))
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
@noinline _linf_length_error(f, n, ::Type{W}) where {W} = throw(ArgumentError(
  "kkt_violation of $(nameof(typeof(f))): the length $n is not representable in the working type $W (n + 1 overflows it)"))
@noinline _normlinf_symmetric_error(f) = throw(ArgumentError(
  "kkt_violation of $(nameof(typeof(f))) has no form for a SymmetricSparseIterate point; use the full-point form on a dense matrix such as Matrix(x)"))
# the level λ is passed in, not read from f: a chain of Scaled wrappers passes the level folded with its factors
function _level_point(f::NormLinf, λ, v::AbstractArray, x::AbstractArray)
  _check_one_based(v, x)
  eltype(v) <: Real || _real_error(:kkt_violation)
  eltype(x) <: Real || _real_error(:kkt_violation)
  length(v) == length(x) || _gradient_length_error(v, x)
  W = float(promote_type(eltype(v), eltype(x)))
  isempty(x) && return zero(W)
  isfinite(convert(W, length(x) + 1)) || _linf_length_error(f, length(x), W)
  _linf_point(_kkt_coefficient(W, λ, true), v, x, W)
end
kkt_violation(f::NormLinf, v::AbstractArray, x::AbstractArray) = _level_point(f, f.λ, v, x)
kkt_violation(f::NormLinf, ::AbstractArray, ::SymmetricSparseIterate) = _normlinf_symmetric_error(f)

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
_exactly_zero_at(f::NormLinf, x::AbstractArray) = iszero(f.λ) || _all_zero(x)
