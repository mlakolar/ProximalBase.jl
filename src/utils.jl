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

# The k-th largest entry of v, 1 ≤ k ≤ length(v), by quickselect with a
# median-of-three pivot; reorders v. Base's partialsort! allocates a scratch
# buffer, this does not.
function _kth_largest!(v::AbstractVector, k::Integer)
  lo, hi = firstindex(v), lastindex(v)
  kk = lo + k - 1
  lo <= kk <= hi || throw(ArgumentError("k = $k is not in 1:$(length(v))"))
  @inbounds while lo < hi
    mid = (lo + hi) >>> 1
    a, b, c = v[lo], v[mid], v[hi]
    pivot = a < b ? (b < c ? b : (a < c ? c : a)) : (a < c ? a : (b < c ? c : b))
    i, j = lo, hi
    while i <= j
      while v[i] > pivot
        i += 1
      end
      while v[j] < pivot
        j -= 1
      end
      if i <= j
        v[i], v[j] = v[j], v[i]
        i += 1
        j -= 1
      end
    end
    # v[lo:j] ≥ pivot ≥ v[i:hi], and the entries in between equal the pivot
    if kk <= j
      hi = j
    elseif kk >= i
      lo = i
    else
      return v[kk]
    end
  end
  v[kk]
end

# The threshold τ of the projection max.(f.(x) .- τ, 0) of the vector f.(x)
# onto the simplex {z ≥ 0 : Σ z = a}, a > 0, by Condat's algorithm (Fast
# projection onto the simplex and the l1 ball, Math. Program. 158, 2016), in
# O(n) expected time. buf is scratch of length at least length(x); f is applied
# on the fly (abs gives the projection onto the l1 ball).
function _simplex_threshold!(buf::AbstractVector{T}, x::AbstractArray, a::T, f::F = identity) where {T, F}
  n = length(x)
  n == 0 && return zero(T)
  # buf[1:nt] holds the discarded entries ṽ, buf[nt+1:nt+nv] the candidates v
  it = iterate(x)
  y1 = T(f(it[1]))
  buf[1] = y1
  nt, nv = 0, 1
  ρ = y1 - a
  it = iterate(x, it[2])
  @inbounds while it !== nothing
    y = T(f(it[1]))
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
    it = iterate(x, it[2])
  end
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
function tril_cartesian(p::Integer, k::Integer)
  rv = div(p * (p + 1), 2) - k
  j = div(isqrt(1 + 8rv) - 1, 2)      # integer arithmetic, exact for any p
  i = rv - div(j * (j + 1), 2)
  (p - i, p - j)
end

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
    @inbounds @simd for i in eachindex(A, B)
      v += abs2(A[i] - B[i])
    end
    return sqrt(v)
  elseif p == Inf
    @inbounds for i in eachindex(A, B)
      v = max(v, abs(A[i] - B[i]))
    end
    return v
  else
    throw(ArgumentError("p should be 2 or Inf"))
  end
end
