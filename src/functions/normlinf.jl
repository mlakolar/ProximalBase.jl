####################################
#  ℓ∞ norm  λ max|xᵢ|
####################################

# The type, the checks and the oracles are here, with the threshold of the ℓ1 ball that the clip needs (`_l1ball_threshold!`,
# Condat's algorithm with a fallback to Michelot's). The steps check the arguments before any write, as NormL2 does.

"""
    NormLinf(λ = 1)

The ℓ∞ norm `f(x) = λ max|xᵢ|` (the largest magnitude of the entries, of a matrix included), with a
nonnegative level `λ` (an `ArgumentError` otherwise, `NaN` included). It is not separable. Its proximal operator
clips the entries at a level `τ ≥ 0`: `prox_{γf}(x)ᵢ = clamp(xᵢ, -τ, τ)`, where `τ` solves
`Σᵢ (|xᵢ| - τ)₊ = γλ`, and `τ = 0` (the output `+0`) when `γλ ≥ ‖x‖₁`. This is `x` minus its projection onto the
ℓ1 ball of radius `γλ`, formed by the clip itself so that no cancellation enters the entries below `τ`. The level
`τ` comes from Condat's algorithm, in O(n) expected time, with Michelot's algorithm when many entries stay candidates.

The value takes arrays, the three iterates and a `SymmetricSparseIterate` (its largest stored magnitude: a
repeated off-diagonal coordinate does not change a maximum), and is `λ` times the largest magnitude rounded once, in
the floating type of the entries. With `λ = 0` the value is zero, a `NaN` or infinite entry included; with `λ = Inf` it
is zero at the zero argument (an empty one included), `Inf` at any other argument without a `NaN` entry, and `NaN` at
a `NaN` entry. A level that is not of the type of the entries is applied in the promoted type and rounded once.

`prox!`, `prox` and `value_and_prox!` take arrays of the same size (a `BlockIterate` or a `SparseIterate` through
indexed reads and writes of its entries, O(length)) and a scalar step; `prox!(x, f, x, γ)` is supported. A negative
or `NaN` step throws an `ArgumentError` before anything is written, and so do sizes, real element types and one-based
indexing. A `SymmetricSparseIterate` argument or output throws an `ArgumentError` before anything is written, also
through `Scaled`: its proximal operator is not provided. The rules apply in this order: `λ = Inf` gives `+0` at
every nonnegative step, `γ = 0` included (#665), for every input; otherwise `λ = 0` or `γ = 0` gives the identity (so
`γ = Inf` with `λ = 0` is the identity); `γ = Inf` with a positive finite `λ` gives `+0`. At positive finite `γ` and
`λ`, a `NaN` entry makes every output `NaN`, and an infinite entry without `NaN` leaves the argument unchanged, also
when the finite product `γλ` overflows in floating point; otherwise the clip above, with the output `+0` when `γλ`
is at least `‖x‖₁` or overflows while `‖x‖₁` is finite. A sum of magnitudes above `floatmax/2` is handled in
power-of-two units. `value_and_prox!` returns the value of the stored output.

The accuracy of `τ` is absolute, of order `n eps(T) max|xᵢ|` in the floating type `T` of the output for `n`
entries: unlike the relative bounds of `NormL2`, an entry far below `max|xᵢ|` has no relative guarantee. The
entries are read as `abs(convert(T, xᵢ))`; Float32 and Float64 with arguments of the same type are the supported
precisions.

The threshold needs a buffer of length `length(x)`. A dense `Array` output of type `T` that does not overlap an
`Array` or a view of an `Array` argument serves as that buffer, before the clip overwrites it. Otherwise (`y === x`,
a view, an iterate or another output type) the object keeps a buffer of its own, allocated on the first call and
again when the size or the element type changes: the same object must not be used from several tasks at once
when that buffer is in use.
"""
struct NormLinf{T<:Real} <: AbstractFunction
  λ::T
  scratch::Scratch

  NormLinf{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ, Scratch()))
end
NormLinf(λ::Real = 1) = NormLinf{typeof(λ)}(λ)

isproximable(::Type{<:NormLinf}) = true
isconvex(::Type{<:NormLinf}) = true

# the buffer is not part of the function
Base.:(==)(f::NormLinf, g::NormLinf) = f.λ == g.λ
Base.hash(f::NormLinf, h::UInt) = hash(f.λ, hash(:NormLinf, h))
Base.show(io::IO, f::NormLinf) = print(io, "NormLinf(", f.λ, ")")

####################################
#  Values
####################################

# λ m in the floating type V of the entries, from the largest magnitude m: a zero level gives zero (m of any kind), an
# infinite level gives zero at m = 0, and otherwise the product of the level and m is rounded once, in the promoted type
# when the level is not a V.
@inline function _linf_scale(λ, m, ::Type{V}) where {V}
  iszero(λ) && return zero(V)
  if isinf(λ)
    iszero(m) && return zero(V)
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
# routes of wrappers such as Scaled, which preflight with _checkwrite first
_value_and_prox_unchecked!(y::AbstractArray, f::NormLinf, x::AbstractArray, γ::Real) = _linf_value_and_prox!(y, f, x, γ)

# The value at the stored output. _linf_prox! returns the largest stored magnitude when it knows it without a pass (the
# threshold of the clip, or the zero and NaN fills), and nothing otherwise; the value is then that of the argument y,
# which `f(y)` computes by the same function of the same largest magnitude.
@inline function _linf_value_and_prox!(y, f::NormLinf, x, γ::Real)
  T = float(eltype(y))
  m = _linf_prox!(y, f, x, γ)
  m === nothing && return f(y)
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
# float(eltype(y)), or nothing when that is not known without a pass (the identity branches). The branches are those of
# the table of the docstring, in its order; the clip is out of line of the exceptional ones.
function _linf_prox!(y, f::NormLinf, x, γ::Real)
  T = float(eltype(y))
  λ = f.λ
  n = length(x)
  isinf(λ) && return _linf_fill!(y, n, zero(T))
  (iszero(λ) | iszero(γ)) && return _linf_copy!(y, x, n)
  isinf(γ) && return _linf_fill!(y, n, zero(T))
  s, m = _linf_classify(x, _AbsRead{T}(), T)
  if s <= floatmax(T) / 2                 # false for a NaN or an infinite sum
    r = _linf_radius(T, γ, λ)
    r >= s && return _linf_fill!(y, n, zero(T))
    τ = _linf_threshold(y, f, x, r, _AbsRead{T}(), T, m)
    return _linf_clip!(y, x, τ, n)
  end
  _linf_exceptional!(y, f, x, γ, s, m)
end

# The branches at a sum of magnitudes that is NaN, infinite or above floatmax/2: a NaN entry gives NaN everywhere;
# an infinite entry without NaN is a limit of the clip (τ ≥ max|xᵢ| - γλ → ∞) and gives the identity, also when γλ
# overflows. Finite entries whose sum overflows or exceeds floatmax/2 are read scaled by 2^-k, and the clip level
# found in those units is multiplied by 2^k: the prox is positively homogeneous, prox_c(x) = 2^k prox_{c 2^-k}(2^-k x).
#
# Conditions of the scaled computation: finite entries, a finite positive step and level, and n eps(T) ≤ 1/2.
# - Each scaled entry is at most floatmax 2^-k. With k = ceil(log2 n) + 2 the sum of the scaled entries is at most
#   n floatmax 2^-k ≤ floatmax/4 before rounding, so the computed sum s′ is at most floatmax/2 (the loop below raises k
#   when it is not). The selector then sees the data under the conditions stated at `_l1ball_threshold!`.
# - An entry that becomes subnormal when scaled changes by less than δ, the smallest subnormal of T (less than δ/2 when
#   the scaling rounds to nearest). The sum of the scaled entries then changes by less than n δ, and so does the
#   level τ′: Σᵢ (yᵢ - τ′)₊ = r′ has a left side whose slope in τ′ is minus the number of entries above τ′, at least
#   one at the solution, so a change of the left side by L moves the root by at most L. In the units of x the level
#   2^k τ′ moves by less than n 2^k δ < 8 n² δ (2^k < 8n for the initial k). The ordinary path has an error of order
#   n eps(T) max|xᵢ| in τ, and max|xᵢ| ≥ floatmax/(4n) here (the sum is at least floatmax/2 up to rounding), so its
#   error is at least eps(T) floatmax/4. The loss is below it when 32 n² δ ≤ eps(T) floatmax(T), which holds for
#   n ≤ 2^60 in Float32 (the ratio eps floatmax/(32 δ) is 2^249) and in Float64 (2^2041). The loss of the scaled path is
#   then below the rounding error of the ordinary path, and the accuracy statement of the docstring holds.
# - No lower bound on τ is needed: x = [floatmax, 2^970] with γλ = floatmax in Float64 has the level 2^969.
# - r′ = 2^-k γλ is formed from the significands and exponents of γ and λ, so that it does not overflow where γλ
#   does: r′ ≥ s′ gives the output +0, and an r′ below the range rounds to a subnormal or zero, a change of at most
#   δ/2 in the scaled units, covered by the bound above.
@noinline function _linf_exceptional!(y, f::NormLinf, x, γ, s::T, m::T) where {T}
  n = length(x)
  isnan(s) && return _linf_fill!(y, n, T(NaN))
  isinf(m) && return (_linf_copy!(y, x, n); m)
  k = _linf_exponent(n)
  local s′, m′
  while true
    rd = _ScaledAbsRead{T}(k)
    s′, m′ = _linf_classify(x, rd, T)
    s′ <= floatmax(T) / 2 && break
    k += 1
  end
  r′ = _linf_scaled_radius(T, γ, f.λ, k)
  r′ >= s′ && return _linf_fill!(y, n, zero(T))
  τ′ = _linf_threshold(y, f, x, r′, _ScaledAbsRead{T}(k), T, m′)
  # τ′ ≤ m′ = 2^-k m rounded; the product 2^k τ′ is capped at m (the rounding of m′ may exceed it by δ)
  _linf_clip!(y, x, min(ldexp(τ′, k), m), n)
end

# ceil(log2(n)) + 2 for n ≥ 1, from the bit length of n - 1
@inline _linf_exponent(n::Int) = 64 - leading_zeros(n - 1) + 2

# The radius γλ, rounded once to T from the product in the promoted type; infinite where it exceeds floatmax(T).
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

# The clip level τ from the threshold of the ℓ1 ball, clamped to [0, m]: the largest magnitude m bounds the exact level
# and a rounded one can exceed it; a level of zero or below gives the output +0. The buffer is y itself when that is
# safe (see `_linf_buffer_is_output`), and the scratch of f otherwise.
#
# The clamp makes λτ equal f(y) when y is stored in T. With τ = min(max(τ, 0), m), finite entries and τ > 0, the entry
# xⱼ of magnitude m ≥ τ is stored as ±τ and no stored entry exceeds τ in magnitude, so the largest stored magnitude is
# τ, and `f(y)` applies `_linf_scale` to that same τ, as `value_and_prox!` does. The clamp needs τ not NaN: under the
# conditions at `_l1ball_threshold!` the level is (s - r)/c with s finite and c ≥ 1, or the initial y₁ - r, so it is a
# number.
@inline function _linf_threshold(y, f::NormLinf, x, r::T, rd, ::Type{T}, m::T) where {T}
  τ = if _linf_buffer_is_output(y, x, T)
    _l1ball_threshold!(y, x, r, rd)
  else
    _l1ball_threshold!(_array_buffer!(f.scratch, T, (length(x),)), x, r, rd)
  end
  min(max(τ, zero(T)), m)
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

# The sum and the largest of the magnitudes rd(x[i]), in one pass. A NaN entry gives a NaN sum. The maximum is that of
# the bit patterns for Float32 and Float64, where it vectorizes (a NaN or an infinite entry gives a NaN or an infinite
# maximum, as in `_maximum_abs`); it is used for finite data only, which a sum that is not NaN or infinite certifies.
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

# y = x, the identity; the largest stored magnitude is not known without a pass
@inline function _linf_copy!(y, x, n::Int)
  if y !== x
    @inbounds for i in 1:n
      y[i] = x[i]
    end
  end
  nothing
end

####################################
#  The threshold of the ℓ1 ball
####################################

# The level τ with Σᵢ (yᵢ - τ)₊ = r for the magnitudes yᵢ = rd(x[i]), i = 1:length(x), by Condat's algorithm (Fast
# projection onto the simplex and the l1 ball, Math. Program. 158, 2016; expected O(n)), which hands over to Michelot's
# algorithm (J. Optim. Theory Appl. 50, 1986) when many entries become candidates. `rd` maps an entry to its magnitude
# in T = eltype(buf): `_AbsRead{T}()`, or the scaled read of the overflow path. `buf` is a scratch array whose length
# is at least length(x), indexed linearly. The level is a Float of type T with a rounding error of order n eps(T) max yᵢ
# (absolute), not clamped: the caller clamps it to [0, max yᵢ]. The proximal operator of the ℓ∞ norm, the projection
# onto the ℓ1 ball and the group variants of the restored functions use it as the threshold.
#
# Conditions on the data, which the caller (`_linf_prox!`) establishes with its classification pass: every yᵢ is finite
# and nonnegative, r is a number with 0 ≤ r < Σ yᵢ, the sum computed in the order of the classification pass is at
# most floatmax(T)/2, and n eps(T) ≤ 1/2. Plain comparisons are used, so that a NaN would be skipped (`NaN > ρ` is false):
# #677 was that omission in the legacy kernel, and a NaN entry is routed to its own branch before this function runs.
#
# Why the selector sees only finite data whose partial sums stay in range. A partial sum in the order of Condat's or
# Michelot's passes, a sum of nonnegative terms, differs from the exact sum of its terms by a relative error of at most
# θ = (1 + eps(T)/2)^n - 1 ≤ 1/3 (for n eps(T) ≤ 1/2), and the sum of the classification pass is at most floatmax/2; the
# exact total is then at most floatmax/2 / (1 - θ) and a computed partial sum at most floatmax/2 (1 + θ)/(1 - θ) ≤
# floatmax. The sums s of the passes are therefore finite.
#
# Why an overflowing (y - r) nv still decides correctly. The test of Condat's pass is s > (y - r) nv, the exact test
# (s + y - r)/(nv + 1) > y - r. With y ≤ floatmax/2 the factor y - r is finite, and the product with nv ≥ 1 overflows
# to +Inf only if (y - r) nv > floatmax; since s is a sum of some of the yᵢ it is below floatmax (see above), so the
# exact test is false, as the comparison s > Inf is. A product that overflows to -Inf has y - r < 0 and the exact
# test (s ≥ 0 > -∞) is true, as s > -Inf is. The level (s - r)/nv is finite as s is and nv ≥ 1.
function _l1ball_threshold!(buf::AbstractArray{T}, x, a::T, rd::F) where {T, F}
  n = length(x)
  n == 0 && return zero(T)
  @inbounds y1 = rd(x[1])
  @inbounds buf[1] = y1
  # buf[1:nt] holds the discarded entries, buf[nt+1:nt+nv] the candidates, of sum s and level ρ = (s - a)/nv. The
  # level is recomputed from the sum when a candidate joins, rather than updated by a division in a chain of dependent
  # steps, and Michelot's algorithm takes over when more than an eighth of the entries become candidates: then the
  # level starts far below the entries and rises slowly, and Condat's pass costs most.
  nt, nv = 0, 1
  s = y1
  ρ = y1 - a
  cap = max(1024, n ÷ 8)
  many = false
  # Once the level has risen, most entries fall below it. One test for four entries keeps the loop from depending on
  # where its code lands in memory.
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
# level is ρ = (s - a)/nv, recomputed from the sum whenever the candidates change.
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
        len == 1 && return ρ            # from rounding only: the largest entry stays
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
# between all the entries and none (the loop of the legacy kernel returned -Inf or 1.0 by the parity of n). Here the
# loop stops when the count c of the entries above τ is 0 or is not below the count of the pass before, and returns τ,
# the last level computed from a nonempty set. The first pass has τ = -Inf, c = n ≥ 1, and a count that decreases at each
# later pass that continues, so the loop runs at most n + 1 passes. When c is 0 or has not decreased, the set that gave τ
# is the set that the rounded level would give again, or is empty: τ is then the level of the last nonempty set, which
# is within rounding of the solution for finite data.
function _l1ball_michelot(x, a::T, rd::F, ::Type{T}) where {T, F}
  τ = T(-Inf)
  cnt = length(x) + 1
  for _ in 0:length(x)
    s, c = zero(T), 0
    @inbounds @simd for i in 1:length(x)
      y = rd(x[i])
      above = y > τ
      s += ifelse(above, y, zero(T))
      c += above
    end
    (c == 0 || c >= cnt) && return τ
    cnt = c
    τ = (s - a) / c
  end
  τ
end

####################################
#  Scaled
####################################

# NormL1's member of the folding of a factor into the level (`_fold_level`), so that the nested route of `Scaled`
# gives the function a step and a level that are each in range when their product is.
function _fold_level(f::NormLinf, α)
  c = _folded_level(f.λ, α)
  c === nothing ? nothing : NormLinf(c)
end
# a zero factor: the zero function at a finite level; an infinite level is an indicator, which has no zero multiple here
_zero_scaled(f::NormLinf) = (isfinite(f.λ) || _zero_scale_error(f); ZeroFunction())
# f(x) is zero when the level is zero or the entries are all zero (a NaN or infinite entry is not zero)
_exactly_zero_at(f::NormLinf, x::AbstractArray) = iszero(f.λ) || _all_zero(_host_entries(x))
