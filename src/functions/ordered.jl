####################################
#
#  Functions of the order of the entries
#
####################################

# The total variation, monotonicity and the sorted ℓ1 norm depend on the order
# of the entries of a vector, and their proximal operators pool adjacent
# entries into blocks of equal values.

####################################
#  total variation in one dimension
####################################

"""
    TotalVariation1D(λ = 1)

The total variation `f(x) = λ Σᵢ |xᵢ₊₁ - xᵢ|` of a vector. Its proximal operator,
total-variation denoising, is computed exactly and in place by Condat's direct
algorithm (2013), in O(n) time on most inputs and O(n²) in the worst case.
"""
struct TotalVariation1D{T<:Real} <: AbstractFunction
  λ::T

  TotalVariation1D{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ))
end
TotalVariation1D(λ::Real = 1) = TotalVariation1D{typeof(λ)}(λ)

isproximable(::Type{<:TotalVariation1D}) = true
_argument_type(::Type{<:TotalVariation1D}) = AbstractVector
isconvex(::Type{<:TotalVariation1D}) = true

function _tv(x::AbstractVector)
  T = _valtype(x)
  s = zero(T)
  @inbounds @simd for i in firstindex(x):lastindex(x)-1
    s += abs(x[i + 1] - x[i])
  end
  s
end

(f::TotalVariation1D)(x::AbstractVector) =
  (T = _valtype(x); iszero(f.λ) ? zero(T) : convert(T, f.λ) * _tv(x))

prox!(y::AbstractVector, f::TotalVariation1D, x::AbstractVector, γ::Real) =
  (_checkwrite(prox!, f, y, x); _tv_denoise!(y, x, convert(float(eltype(y)), γ * f.λ)))
value_and_prox!(y::AbstractVector, f::TotalVariation1D, x::AbstractVector, γ::Real) =
  f(prox!(y, f, x, γ))

# argmin_z λ Σₖ |zₖ₊₁ - zₖ| + ‖z - x‖²/2 by Condat's algorithm, in place (z may be
# x). It scans x once and keeps, for the current segment k0:k, the bounds vmin
# and vmax of its value and the residuals umin and umax of the last entries
# under each bound; a jump is necessary when the next entry leaves the tube
# around the segment, and the segment up to kmin (or kplus) is then final. The
# entries are read only beyond those written, so z may be x.
function _tv_denoise!(z::AbstractVector, x::AbstractVector, λ::T) where {T}
  _checksize(z, x)
  Base.require_one_based_indexing(z, x)
  N = length(x)
  N == 0 && return z
  if !(λ > 0)
    z === x || copyto!(z, x)
    return z
  end
  k = k0 = kmin = kplus = 1
  vmin, vmax = convert(T, x[1]) - λ, convert(T, x[1]) + λ
  umin, umax = λ, -λ
  @inbounds while true
    while k == N
      if umin < 0                                   # a negative jump before the end
        for i in k0:kmin
          z[i] = vmin
        end
        kmin += 1
        k = k0 = kmin
        vmin = convert(T, x[k])
        umin = λ
        umax = convert(T, x[k]) + λ - vmax
      elseif umax > 0                               # a positive jump before the end
        for i in k0:kplus
          z[i] = vmax
        end
        kplus += 1
        k = k0 = kplus
        vmax = convert(T, x[k])
        umax = -λ
        umin = convert(T, x[k]) - λ - vmin
      else                                          # the last segment
        v = vmin + umin / (k - k0 + 1)
        for i in k0:N
          z[i] = v
        end
        return z
      end
      if k == N                                     # the last entry alone
        z[N] = vmin + umin
        return z
      end
    end
    xk = convert(T, x[k + 1])
    if xk + umin < vmin - λ                         # a negative jump
      for i in k0:kmin
        z[i] = vmin
      end
      kmin += 1
      k = k0 = kplus = kmin
      vmin = convert(T, x[k])
      vmax = vmin + 2λ
      umin, umax = λ, -λ
    elseif xk + umax > vmax + λ                     # a positive jump
      for i in k0:kplus
        z[i] = vmax
      end
      kplus += 1
      k = k0 = kmin = kplus
      vmax = convert(T, x[k])
      vmin = vmax - 2λ
      umin, umax = λ, -λ
    else                                            # no jump
      k += 1
      umin += xk - vmin
      umax += xk - vmax
      if umin >= λ
        vmin += (umin - λ) / (k - k0 + 1)
        umin = λ
        kmin = k
      end
      if umax <= -λ
        vmax += (umax + λ) / (k - k0 + 1)
        umax = -λ
        kplus = k
      end
    end
  end
end

####################################
#  the fused lasso
####################################

"""
    FusedLasso(λ1, λ2)

The penalty `f(x) = λ1 Σᵢ |xᵢ| + λ2 Σᵢ |xᵢ₊₁ - xᵢ|` of the fused lasso signal
approximator, on a vector. Its proximal operator soft-thresholds the
total-variation denoising of `x` by `γλ1` (Friedman, Hastie, Höfling and
Tibshirani, 2007).
"""
struct FusedLasso{T<:Real} <: AbstractFunction
  λ1::T
  λ2::T

  function FusedLasso{T}(λ1::T, λ2::T) where {T<:Real}
    _checknonneg("λ1", λ1)
    _checknonneg("λ2", λ2)
    new{T}(λ1, λ2)
  end
end
FusedLasso(λ1::Real, λ2::Real) = (p = promote(λ1, λ2); FusedLasso{typeof(p[1])}(p...))

isproximable(::Type{<:FusedLasso}) = true
_argument_type(::Type{<:FusedLasso}) = AbstractVector
isconvex(::Type{<:FusedLasso}) = true

function (f::FusedLasso)(x::AbstractVector)
  T = _valtype(x)
  (iszero(f.λ1) ? zero(T) : convert(T, f.λ1) * convert(T, sum(abs, x; init = zero(T)))) +
  (iszero(f.λ2) ? zero(T) : convert(T, f.λ2) * _tv(x))
end

function prox!(y::AbstractVector, f::FusedLasso, x::AbstractVector, γ::Real)
  _checkwrite(prox!, f, y, x)
  T = float(eltype(y))
  _tv_denoise!(y, x, convert(T, γ * f.λ2))
  c = convert(T, γ * f.λ1)
  y .= shrink.(y, c)
end
value_and_prox!(y::AbstractVector, f::FusedLasso, x::AbstractVector, γ::Real) = f(prox!(y, f, x, γ))

####################################
#  pooling adjacent violators
####################################

# The isotonic regression of v into y: the nondecreasing sequence nearest to v,
# or with σ = -1 the nonincreasing one. Blocks of pooled entries sit on a stack,
# their sums in s and their sizes in c; a new entry is pooled with the block
# before it while their means violate the order. O(n); y may be v.
function _pava!(y::AbstractVector, v::AbstractVector, s::Vector{T}, c::Vector{Int}, σ::Int) where {T}
  n = length(v)
  nb = 0
  @inbounds for i in 1:n
    nb += 1
    s[nb] = convert(T, v[i])
    c[nb] = 1
    # the means of the last two blocks out of order: s₁/c₁ > s₂/c₂, times σ
    while nb > 1 && σ * (s[nb - 1] * c[nb] - s[nb] * c[nb - 1]) > 0
      s[nb - 1] += s[nb]
      c[nb - 1] += c[nb]
      nb -= 1
    end
  end
  i = 0
  @inbounds for b in 1:nb
    m = s[b] / c[b]
    for _ in 1:c[b]
      i += 1
      y[i] = m
    end
  end
  y
end

# Weighted isotonic regression of x into y, with weights γmin/γᵢ and direction
# σ. Blocks sit on a stack, their means in m, weights in W and sizes in c;
# adjacent blocks are pooled while their means violate the order. O(n); y may be x.
function _wpava!(y, x, γ, γmin, m::Vector{S}, W::Vector{S}, c::Vector{Int}, σ::Int) where {S}
  n = length(x)
  nb = 0
  @inbounds for i in 1:n
    nb += 1
    m[nb] = convert(S, x[i])
    W[nb] = convert(S, γmin) / convert(S, γ[i])
    c[nb] = 1
    while nb > 1 && σ * (m[nb - 1] - m[nb]) > 0
      m[nb - 1] += (m[nb] - m[nb - 1]) * (W[nb] / (W[nb - 1] + W[nb]))
      W[nb - 1] += W[nb]
      c[nb - 1] += c[nb]
      nb -= 1
    end
  end
  i = 0
  @inbounds for b in 1:nb
    for _ in 1:c[b]
      i += 1
      y[i] = convert(eltype(y), m[b])
    end
  end
  y
end

####################################
#  isotonic regression
####################################

"""
    IndIsotonic(; decreasing = false)

The indicator of the nondecreasing vectors `{x : x₁ ≤ x₂ ≤ … ≤ xₙ}`, or with
`decreasing = true` of the nonincreasing ones. Its proximal operator, the
projection, is isotonic regression, computed by pooling adjacent violators in
O(n) time, in buffers that the function owns.

An array of steps `γ` gives weighted isotonic regression, minimizing
`Σᵢ wᵢ(θᵢ − xᵢ)²` with `wᵢ = 1/γᵢ`. Only the ratios of the steps matter.
The steps must be positive and finite with a representable ratio; nonfinite
data are outside the guarantee. See [Weighted isotonic regression](@ref) for an example.
"""
struct IndIsotonic <: AbstractFunction
  decreasing::Bool
  sums::Scratch
  sizes::Scratch
  means::Scratch
  weights::Scratch
end
IndIsotonic(; decreasing::Bool = false) = IndIsotonic(decreasing, Scratch(), Scratch(), Scratch(), Scratch())

isproximable(::Type{IndIsotonic}) = true
_argument_type(::Type{<:IndIsotonic}) = AbstractVector
isconvex(::Type{IndIsotonic}) = true
_isindicator(::Type{IndIsotonic}) = true

function (f::IndIsotonic)(x::AbstractVector)
  T = float(eltype(x))
  t = _tol(T)
  σ = f.decreasing ? -1 : 1
  @inbounds for i in firstindex(x):lastindex(x)-1
    σ * (x[i] - x[i + 1]) <= t * max(one(T), abs(x[i]), abs(x[i + 1])) || return T(Inf)
  end
  zero(T)
end

function prox!(y::AbstractVector, f::IndIsotonic, x::AbstractVector, ::Real)
  _checkwrite(prox!, f, y, x)
  T = float(eltype(y))
  n = length(x)
  _pava!(y, x, _buffer!(f.sums, T, n), _buffer!(f.sizes, Int, n), f.decreasing ? -1 : 1)
end

####################################
#  the sorted ℓ1 norm
####################################

"""
    SLOPE(λ)

The sorted ℓ1 norm `f(x) = Σᵢ λᵢ |x|₍ᵢ₎`, where `|x|₍₁₎ ≥ |x|₍₂₎ ≥ …` are the
magnitudes of the entries in decreasing order and `λ₁ ≥ λ₂ ≥ … ≥ 0`: the penalty
of SLOPE (Bogdan, van den Berg, Sabatti, Su and Candès, 2015), which controls
the false discovery rate of the selected variables with the
Benjamini–Hochberg sequence of `λ`. With all `λᵢ` equal it is the ℓ1 norm, and
with `λ = (λ₁, 0, …, 0)` the ℓ∞ norm. Its proximal operator sorts the
magnitudes, subtracts `γλ`, pools adjacent violators into a nonincreasing
sequence, clips it at zero, and restores the order and the signs: O(n log n),
in buffers that the function owns.
"""
struct SLOPE{V<:AbstractVector{<:Real}} <: AbstractFunction
  λ::V
  perm::Scratch       # the order of the magnitudes
  vals::Scratch       # the result, in that order
  sums::Scratch       # the magnitudes, then the sums of the blocks of the pooling
  sizes::Scratch      # the scratch of the sort, then the sizes of the blocks

  function SLOPE{V}(λ::V) where {V<:AbstractVector{<:Real}}
    _checknonneg("λ", λ)
    issorted(λ; rev = true) || throw(ArgumentError("λ must be nonincreasing"))
    new{V}(λ, Scratch(), Scratch(), Scratch(), Scratch())
  end
end
SLOPE(λ::AbstractVector{<:Real}) = SLOPE{typeof(λ)}(λ)

isproximable(::Type{<:SLOPE}) = true
_argument_type(::Type{<:SLOPE}) = AbstractVector
isconvex(::Type{<:SLOPE}) = true
_checkparams(f::SLOPE, x) = (_checkparam(f.λ, x); true)

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)},
                     f::Union{TotalVariation1D, FusedLasso, SLOPE}, y::AbstractVector, x::AbstractVector)
  _checksize(y, x)
  _checkparams(f, x)
  Base.require_one_based_indexing(y, x)
  true
end

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)},
                     f::IndIsotonic, y::AbstractArray, x::AbstractArray)
  _checksize(y, x)
  _checkargument(f, y)
  _checkargument(f, x)
  _checkparams(f, x)
  Base.require_one_based_indexing(y, x)
  true
end

# the magnitudes a of x and the indices p that sort them in decreasing order,
# in the buffers of f, allocating nothing: an in-place quicksort for short
# vectors, and Base's sort with a scratch buffer, about twice as fast, once the
# indirect comparisons of the quicksort leave the cache
function _sortperm_abs!(f::SLOPE, x::AbstractVector, ::Type{T}) where {T}
  n = length(x)
  a = _buffer!(f.sums, T, n)
  p = _buffer!(f.perm, Int, n)
  @inbounds for i in 1:n
    a[i] = abs(convert(T, x[i]))
    p[i] = i
  end
  if n <= 4096
    sort!(p; alg = QuickSort, lt = (i, j) -> a[i] > a[j])
  else
    sortperm!(p, a; rev = true, initialized = true, scratch = _buffer!(f.sizes, Int, n))
  end
  p, a
end

function (f::SLOPE)(x::AbstractVector)
  _checkparams(f, x)
  Base.require_one_based_indexing(x)
  T = float(eltype(x))
  p, a = _sortperm_abs!(f, x, T)
  s = zero(T)
  @inbounds for k in eachindex(p)
    s += convert(T, f.λ[k]) * a[p[k]]
  end
  s
end

function value_and_prox!(y::AbstractVector, f::SLOPE, x::AbstractVector, γ::Real)
  _checkwrite(value_and_prox!, f, y, x)
  T = float(eltype(y))
  n = length(x)
  p, a = _sortperm_abs!(f, x, T)
  w = _buffer!(f.vals, T, n)
  g = convert(T, γ)
  @inbounds for k in 1:n
    w[k] = a[p[k]] - g * convert(T, f.λ[k])
  end
  # the nearest nonincreasing sequence, clipped at zero, is the projection onto
  # the nonincreasing nonnegative sequences
  _pava!(w, w, _buffer!(f.sums, T, n), _buffer!(f.sizes, Int, n), -1)
  v = zero(T)
  @inbounds for k in 1:n
    wk = max(w[k], zero(T))
    v += convert(T, f.λ[k]) * wk
    j = p[k]
    y[j] = signbit(x[j]) ? -wk : wk         # each entry once, so y may be x
  end
  v
end
prox!(y::AbstractVector, f::SLOPE, x::AbstractVector, γ::Real) = (value_and_prox!(y, f, x, γ); y)

####################################
#  the sum of the largest entries
####################################

"""
    SumLargest(k, λ = 1)

The convex, positively homogeneous function `f(x) = λ Σᵢ₌₁ᵏ x₍ᵢ₎`, where
`x₍₁₎ ≥ x₍₂₎ ≥ …` are the signed entries of an array in decreasing order.
`k` is an integer with `1 ≤ k ≤ length(x)` and `λ` is finite and nonnegative.
For `k = 1` this is `λ maximum(x)`, and for `k = length(x)` it is `λ sum(x)`.

Its proximal operator is `x - P_C(x)` for the capped simplex
`C = {w : 0 ≤ wᵢ ≤ γλ, Σᵢ wᵢ = kγλ}`. Selection among its breakpoints finds
the projection in O(n) expected time. Proximal inputs must be finite. Values
and proximal operators use buffers that the function owns; give each thread
its own `deepcopy`. Values have the argument's floating-point type, and
[`value_and_prox!`](@ref) returns a value in the output's precision.
"""
struct SumLargest{T<:Real} <: AbstractFunction
  k::Int
  λ::T
  vals::Scratch
  work::Scratch
  breaks::Scratch
  sums::Scratch

  function SumLargest{T}(k::Integer, λ::T) where {T<:Real}
    1 <= k <= typemax(Int) || throw(ArgumentError("k must be a positive integer representable as Int"))
    _checknonneg("λ", λ)
    isfinite(λ) || throw(ArgumentError("λ must be finite"))
    new{T}(Int(k), λ, Scratch(), Scratch(), Scratch(), Scratch())
  end
end
SumLargest(k::Integer, λ::Real = 1) = SumLargest{typeof(λ)}(k, λ)

isproximable(::Type{<:SumLargest}) = true
isconvex(::Type{<:SumLargest}) = true
_checkparams(f::SumLargest, x) = f.k <= length(x) ||
  throw(DimensionMismatch("k = $(f.k) exceeds the argument's length $(length(x))"))
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::SumLargest, y::AbstractArray, x::AbstractArray) =
  (_checksize(y, x); _checkparams(f, x); Base.require_one_based_indexing(y, x); true)

# Compensation preserves cancellation when the selected entries have both signs.
@inline function _sumlargest_add(s, e, v)
  t = s + v
  e += abs(s) >= abs(v) ? (s - t) + v : (v - t) + s
  t, e
end

function _sumlargest_sum(v::AbstractVector{T}, k::Int, λ) where {T}
  m = zero(T)
  @inbounds for i in 1:k
    m = max(m, abs(v[i]))
  end
  iszero(m) && return zero(T)
  if !isfinite(m)
    s = zero(T)
    @inbounds for i in 1:k
      s += v[i]
    end
    return convert(T, λ * s)
  end
  # Cancel opposite signs first. Scaling every entry immediately could erase
  # a tiny remainder, as in [floatmax, -floatmax, nextfloat(0)].
  lo, hi = 1, k
  @inbounds while lo <= hi
    if v[lo] >= 0
      lo += 1
    else
      v[lo], v[hi] = v[hi], v[lo]
      hi -= 1
    end
  end
  p, q = 1, lo
  s = e = zero(T)
  @inbounds while p < lo || q <= k
    if q <= k && (s >= 0 || p == lo)
      z = v[q]
      q += 1
    else
      z = v[p]
      p += 1
    end
    s, e = _sumlargest_add(s, e, z)
    isfinite(s) || break
  end
  S = float(promote_type(T, typeof(λ)))
  isfinite(s + e) && return convert(T, S(λ) * S(s + e))
  # Apply the scale after summing: neither the sum nor λ times it need fit
  # separately, even when their product does.
  a = exponent(m)
  s = e = zero(T)
  @inbounds for i in 1:k
    s, e = _sumlargest_add(s, e, ldexp(v[i], -a))
  end
  u, b = frexp(S(λ))
  w, c = frexp(S(s + e))
  convert(T, ldexp(u * w, a + b + c))
end

function (f::SumLargest)(x::AbstractArray{<:Real})
  _checkparams(f, x)
  T = float(eltype(x))
  iszero(f.λ) && return zero(T)
  n = length(x)
  v = _buffer!(f.sums, T, n)
  @inbounds for (i, xi) in enumerate(x)
    v[i] = xi
    isnan(v[i]) && return T(NaN)
  end
  f.k < n && _kth_largest!(view(v, 1:n), f.k)
  _sumlargest_sum(v, f.k, f.λ)
end

# In units of the cap c, center at the k-th largest entry a. The threshold
# lies in [a-c, a], so only breakpoints in (-1, 0) can matter. This narrower
# bracket also keeps a small step visible at a large common offset.
@inline function _sumlargest_difference(x::T, a::T, c::T, m::T, e::Int) where {T}
  d = x - a
  if floatmin(T) <= c <= floatmax(T)
    return clamp(d / c, -one(T), one(T))
  end
  z = isfinite(d) ? ldexp(d, -e) : ldexp(x, -e) - ldexp(a, -e)
  clamp(z / m, -one(T), one(T))
end

# The remaining breakpoints represent Σ sⱼ(bⱼ-t)₊ plus an affine part r-qt.
# A median discards at least half of them; discarded active terms join the
# affine part. Once none remain, solve the last linear piece exactly.
function _sumlargest_level!(v::AbstractVector{T}, n::Int, k::Int, points) where {T}
  full = q = nb = 0
  s = err = zero(T)
  @inbounds for i in 1:n
    d = v[i]
    if d >= one(T)
      full += 1
    elseif d >= zero(T)
      s, err = _sumlargest_add(s, err, d)
      q += 1
      if d > zero(T)
        nb += 1
        points[nb] = (d - one(T), -1)
      end
    elseif d > -one(T)
      nb += 1
      points[nb] = (d, 1)
    end
  end
  r, re = _sumlargest_add(s, err, T(full - k))
  lo, hi = -one(T), zero(T)
  @inbounds while nb > 0
    t = _kth_largest!(view(points, 1:nb), (nb + 1) ÷ 2)[1]
    g, ge = _sumlargest_add(r, re, -q * t)
    for j in 1:nb
      b, sign = points[j]
      if b > t
        g, ge = _sumlargest_add(g, ge, sign * (b - t))
      end
    end
    gap = g + ge
    iszero(gap) && return t
    live = 0
    if gap > 0
      lo = t
      for j in 1:nb
        if points[j][1] > t
          live += 1
          points[live] = points[j]
        end
      end
    else
      hi = t
      for j in 1:nb
        b, sign = points[j]
        if b >= t
          r, re = _sumlargest_add(r, re, sign * b)
          q += sign
        else
          live += 1
          points[live] = points[j]
        end
      end
    end
    nb = live
  end
  iszero(q) ? (lo + hi) / 2 : clamp((r + re) / q, lo, hi)
end

@inline function _sumlargest_output(x, g, λ, w, m, e, ::Val{false})
  p = _positive_product(g, λ, w)
  isfinite(p) ? x - p : ldexp(ldexp(x, -e) - m * w, e)
end
@inline _sumlargest_output(x, g, λ, w, m, e, ::Val{true}) = _positive_product(g, λ, w)

# Also used by the conjugate's projection, without a Moreau subtraction.
function _sumlargest_prox!(y, f::SumLargest, x, γ, projection::Val)
  _checkwrite(prox!, f, y, x)
  T = float(eltype(y))
  S = float(promote_type(T, eltype(x), typeof(γ), typeof(f.λ)))
  g, λ = S(γ), S(f.λ)
  isfinite(g) && g > 0 || throw(ArgumentError("γ must be finite and positive"))
  n = length(x)
  v = _buffer!(f.vals, S, n)
  @inbounds for (i, xi) in enumerate(x)
    isfinite(xi) || throw(ArgumentError("SumLargest proximal inputs must be finite"))
    v[i] = xi
  end
  if iszero(λ)
    @inbounds for (i, j) in enumerate(eachindex(IndexLinear(), y))
      y[j] = projection isa Val{true} ? zero(T) : v[i]
    end
    return y
  end
  mg, eg = frexp(g)
  ml, el = frexp(λ)
  m, e = mg * ml, eg + el
  c = _positive_product(g, λ, one(S))
  t = zero(S)
  w = _buffer!(f.work, S, n)
  if f.k < n
    copyto!(w, 1, v, 1, n)
    a = _kth_largest!(view(w, 1:n), f.k)
    @inbounds for i in 1:n
      w[i] = _sumlargest_difference(v[i], a, c, m, e)
    end
    t = _sumlargest_level!(w, n, f.k, _buffer!(f.breaks, Tuple{S, Int}, n))
  end
  @inbounds for (i, j) in enumerate(eachindex(IndexLinear(), y))
    z = f.k == n ? one(S) : clamp(w[i] - t, zero(S), one(S))
    y[j] = _sumlargest_output(v[i], g, λ, z, m, e, projection)
  end
  y
end

prox!(y::AbstractArray{<:Real}, f::SumLargest, x::AbstractArray{<:Real}, γ::Real) =
  _sumlargest_prox!(y, f, x, γ, Val(false))
value_and_prox!(y::AbstractArray{<:Real}, f::SumLargest, x::AbstractArray{<:Real}, γ::Real) = f(prox!(y, f, x, γ))

function _checkstep(op::O, f::SumLargest, y::AbstractArray{<:Real}, x::AbstractArray{<:Real}, γ::Real) where {O}
  T = float(eltype(y))
  S = float(promote_type(T, eltype(x), typeof(γ), typeof(f.λ)))
  g = S(γ)
  isfinite(g) && g > 0 || throw(ArgumentError("γ must be finite and positive"))
  true
end
