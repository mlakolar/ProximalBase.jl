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

function _buffer!(s::Scratch, ::Type{T}, n::Integer) where {T}
  b = s.buf
  if b isa Vector{T}
    length(b) < n && resize!(b, n)
    return b
  end
  b = Vector{T}(undef, n)
  s.buf = b
  b
end

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

# Scratch is an implementation detail: copies of a function object share no buffer
Base.copy(s::Scratch) = Scratch()
Base.deepcopy_internal(::Scratch, ::IdDict) = Scratch()

####################################
#
#  Checks
#
####################################

_checksize(y, x) = size(y) == size(x) ||
  throw(DimensionMismatch("the output has size $(size(y)) and the input $(size(x))"))

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
_checkparam(w::AbstractArray, x) = length(w) == length(x) ||
  throw(DimensionMismatch("a parameter array has length $(length(w)) and the argument has length $(length(x))"))

# the value of an indicator function
_ind(ok::Bool, ::Type{T}) where {T} = ok ? zero(T) : T(Inf)

# The relative tolerance with which an indicator function accepts a point
# rounded onto the boundary of its set; points that a projection returns are
# feasible within it.
_tol(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))
_tol(::Type) = sqrt(eps(Float64))

####################################
#
#  Selection and projection onto the simplex
#
####################################

# The k-th largest entry of v, 1 ≤ k ≤ length(v), by the selection algorithm
# of Floyd and Rivest (Algorithm 489, Comm. ACM 18, 1975): reorders v so that
# v[k] is the k-th largest, with larger entries before it and smaller after.
# About n + min(k, n - k) comparisons, much fewer than a quickselect when k is
# small; Base's partialsort! allocates a scratch buffer, this does not.
function _kth_largest!(v::AbstractVector, k::Integer)
  lo, hi = firstindex(v), lastindex(v)
  kk = lo + k - 1
  lo <= kk <= hi || throw(ArgumentError("k = $k is not in 1:$(length(v))"))
  _select_desc!(v, lo, hi, kk)
end

function _select_desc!(v::AbstractVector, lo::Int, hi::Int, k::Int)
  @inbounds while hi > lo
    if hi - lo > 600
      # recurse on a sample whose k-th largest entry brackets that of v[lo:hi]
      n = hi - lo + 1
      i = k - lo + 1
      z = log(n)
      s = exp(2z / 3) / 2
      sd = sqrt(z * s * (n - s) / n) / 2 * sign(i - n / 2)
      _select_desc!(v, max(lo, floor(Int, k - i * s / n + sd)), min(hi, floor(Int, k + (n - i) * s / n + sd)), k)
    end
    t = v[k]
    i, j = lo, hi
    v[lo], v[k] = v[k], v[lo]
    if v[hi] < t
      v[hi], v[lo] = v[lo], v[hi]
    end
    while i < j
      v[i], v[j] = v[j], v[i]
      i += 1
      j -= 1
      while v[i] > t
        i += 1
      end
      while v[j] < t
        j -= 1
      end
    end
    if v[lo] == t
      v[lo], v[j] = v[j], v[lo]
    else
      j += 1
      v[j], v[hi] = v[hi], v[j]
    end
    j <= k && (lo = j + 1)
    k <= j && (hi = j - 1)
  end
  v[k]
end

# The threshold τ of the projection max.(y .- τ, 0) of the vector
# y = f.(x[idx]) onto the simplex {z ≥ 0 : Σ z = a}, a > 0, by Condat's
# algorithm (Fast projection onto the simplex and the l1 ball, Math. Program.
# 158, 2016), in O(n) expected time. buf is scratch of length at least
# length(idx). idx is eachindex(IndexLinear(), x), or the indices of a group,
# so that a group needs no view; f = abs gives the projection onto the l1 ball.
# The index set is indexed by position, with the first entry peeled: a plain
# loop over a range is three times faster than one through iterate.
function _simplex_threshold!(buf::AbstractVector{T}, x::AbstractArray, idx::AbstractVector{<:Integer}, a::T,
                             f::F = identity) where {T, F}
  n = length(idx)
  n == 0 && return zero(T)
  j0 = firstindex(idx)
  @inbounds y1 = convert(T, f(x[idx[j0]]))
  buf[1] = y1
  # buf[1:nt] holds the discarded entries ṽ, buf[nt+1:nt+nv] the candidates v
  nt, nv = 0, 1
  ρ = y1 - a
  @inbounds for j in j0+1:j0+n-1
    y = convert(T, f(x[idx[j]]))
    if y > ρ
      ρ += (y - ρ) / (nv + 1)
      if ρ > y - a
        nv += 1
        buf[nt + nv] = y
      else
        nt += nv
        nv = 1
        buf[nt + 1] = y
        ρ = y - a
      end
    end
  end
  _simplex_finish!(buf, nt, nv, ρ)
end

# the end of Condat's algorithm, from the discarded entries buf[1:nt], the
# candidates buf[nt+1:nt+nv] and the level ρ of the first pass
function _simplex_finish!(buf::AbstractVector{T}, nt::Int, nv::Int, ρ::T) where {T}
  # the discarded entries above ρ rejoin the candidates, collected at the front
  w = 0
  @inbounds for i in 1:nt
    y = buf[i]
    if y > ρ
      w += 1
      buf[w] = y
      ρ += (y - ρ) / (w + nv)
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
        buf[i] = buf[len]
        len -= 1
        len == 0 && return ρ        # only from rounding: the largest entry stays
        ρ += (ρ - y) / len
        changed = true
      else
        i += 1
      end
    end
  end
  ρ
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
[`SymmetricSparseIterate`](@ref) that holds the entry. Inverse of
[`tril_cartesian`](@ref).
"""
tril_index(p::Integer, r::Integer, c::Integer) = p * (c - 1) - div(c * (c - 1), 2) + r

"""
    tril_cartesian(p, k)

The entry `(r, c)`, `r ≥ c`, of the lower triangle of a `p × p` matrix with
linear index `k`, numbered column by column. Inverse of [`tril_index`](@ref).
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

"""
    norm_diff(A, B, p = 2)

`‖A - B‖` without allocating `A - B`: the Euclidean (Frobenius) norm for
`p = 2`, the largest absolute entry for `p = Inf`.
"""
function norm_diff(A::AbstractArray, B::AbstractArray, p::Real = 2)
  size(A) == size(B) || throw(DimensionMismatch())
  T = float(promote_type(eltype(A), eltype(B)))
  v = zero(T)
  if p == 2
    return sqrt(_sumabs2_diff(A, B))
  elseif p == Inf
    @inbounds for i in eachindex(A, B)
      v = max(v, abs(A[i] - B[i]))
    end
    return v
  else
    throw(ArgumentError("p should be 2 or Inf"))
  end
end

# Σᵢ (xᵢ - yᵢ)², without allocating x - y
function _sumabs2_diff(x::AbstractArray, y::AbstractArray)
  size(x) == size(y) || throw(DimensionMismatch("the arrays have sizes $(size(x)) and $(size(y))"))
  s = zero(float(promote_type(eltype(x), eltype(y))))
  @inbounds @simd for i in eachindex(x, y)
    s += abs2(x[i] - y[i])
  end
  s
end
