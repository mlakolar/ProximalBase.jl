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
  T = float(eltype(x))
  s = zero(T)
  @inbounds @simd for i in firstindex(x):lastindex(x)-1
    s += abs(x[i + 1] - x[i])
  end
  s
end

(f::TotalVariation1D)(x::AbstractVector) =
  (T = float(eltype(x)); iszero(f.λ) ? zero(T) : convert(T, f.λ) * _tv(x))

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
  T = float(eltype(x))
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

####################################
#  isotonic regression
####################################

"""
    IndIsotonic(; decreasing = false)

The indicator of the nondecreasing vectors `{x : x₁ ≤ x₂ ≤ … ≤ xₙ}`, or with
`decreasing = true` of the nonincreasing ones. Its proximal operator, the
projection, is isotonic regression, computed by pooling adjacent violators in
O(n) time, in buffers that the function owns.
"""
struct IndIsotonic <: AbstractFunction
  decreasing::Bool
  sums::Scratch
  sizes::Scratch
end
IndIsotonic(; decreasing::Bool = false) = IndIsotonic(decreasing, Scratch(), Scratch())

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
                     f::Union{TotalVariation1D, FusedLasso, IndIsotonic, SLOPE}, y::AbstractVector, x::AbstractVector)
  _checksize(y, x)
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
