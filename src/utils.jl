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

# Grouped nonnegative coefficients: keep the usual multiply order when it is
# well scaled. Retry exceptional intermediates with the smallest and largest
# factors paired first, so a compensating weight is applied before it is lost.
@inline function _positive_product(a::T, b::T, c::T) where {T<:AbstractFloat}
  (iszero(a) || iszero(b) || iszero(c)) && return zero(T)
  ab = a * b
  p = ab * c
  isfinite(ab) && ab >= floatmin(T) && isfinite(p) && p >= floatmin(T) && return p
  a > b && ((a, b) = (b, a))
  b > c && ((b, c) = (c, b))
  a > b && ((a, b) = (b, a))
  (a * c) * b
end
@inline _positive_product(a::Real, b::Real, c::Real) = _positive_product(promote(float(a), float(b), float(c))...)

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

# A buffer of zeros of type T and size dims, next to the array a: an Array,
# and a device array of the type of a GPU array a (see the GPU extension).
_zeros_like(a, ::Type{T}, dims) where {T} = zeros(T, dims)

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
  # buf[1:nt] holds the discarded entries ṽ, buf[nt+1:nt+nv] the candidates v,
  # of sum s and level ρ = (s - a)/nv. The level is recomputed from the sum when
  # a candidate joins, rather than updated by a division in a chain of dependent
  # steps, and Michelot's algorithm takes over when more than an eighth of the
  # entries become candidates: then the level starts far below the entries and
  # rises slowly, and Condat's pass costs most.
  nt, nv = 0, 1
  s = y1
  ρ = y1 - a
  cap = max(1024, n ÷ 8)
  many = false
  # Once the level has risen, most entries fall below it. One test for four
  # entries keeps the loop from depending on where its code lands in memory,
  # which changed its speed twofold on an AMD EPYC.
  j, jend = j0 + 1, j0 + n - 1
  @inbounds while j + 3 <= jend
    y₁, y₂ = convert(T, f(x[idx[j]])), convert(T, f(x[idx[j + 1]]))
    y₃, y₄ = convert(T, f(x[idx[j + 2]])), convert(T, f(x[idx[j + 3]]))
    if (y₁ > ρ) | (y₂ > ρ) | (y₃ > ρ) | (y₄ > ρ)
      nt, nv, s, ρ, many = _condat_step!(buf, y₁, nt, nv, s, ρ, a, cap)
      many || ((nt, nv, s, ρ, many) = _condat_step!(buf, y₂, nt, nv, s, ρ, a, cap))
      many || ((nt, nv, s, ρ, many) = _condat_step!(buf, y₃, nt, nv, s, ρ, a, cap))
      many || ((nt, nv, s, ρ, many) = _condat_step!(buf, y₄, nt, nv, s, ρ, a, cap))
      many && break
    end
    j += 4
  end
  @inbounds while !many && j <= jend
    nt, nv, s, ρ, many = _condat_step!(buf, convert(T, f(x[idx[j]])), nt, nv, s, ρ, a, cap)
    j += 1
  end
  many && return _simplex_threshold_michelot(x, idx, a, f)
  _simplex_finish!(buf, nt, nv, s, a)
end

# One entry y of Condat's pass: below the level ρ it is discarded; otherwise it
# joins the candidates when their level with y, (s + y - a)/(nv + 1), exceeds
# y - a, and starts them anew if not. The last value is true when more than cap
# entries have become candidates.
@inline function _condat_step!(buf::AbstractVector{T}, y::T, nt::Int, nv::Int, s::T, ρ::T, a::T, cap::Int) where {T}
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

# the end of Condat's algorithm, from the discarded entries buf[1:nt] and the
# candidates buf[nt+1:nt+nv] of sum s, whose level is ρ = (s - a)/nv, recomputed
# from the sum whenever the candidates change
function _simplex_finish!(buf::AbstractVector{T}, nt::Int, nv::Int, s::T, a::T) where {T}
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
        len == 1 && return ρ            # only from rounding: the largest entry stays
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

# The threshold by Michelot's algorithm (J. Optim. Theory Appl. 50, 1986): the
# level of all entries, then of those above it, until that set stops changing.
# It is Newton's method on the convex Σᵢ max(yᵢ - τ, 0) - a from below, each step
# a vectorized pass. Condat's pass hands over to it when many entries become
# candidates, as for a radius a that keeps a large share of them positive: its
# level then starts far below the entries and rises slowly.
function _simplex_threshold_michelot(x::AbstractArray, idx::AbstractVector{<:Integer}, a::T, f::F) where {T, F}
  τ, cnt = T(-Inf), -1                    # the first pass takes the level of all the entries
  for _ in 0:length(idx)
    s, c = zero(T), 0
    @inbounds @simd for j in idx
      y = convert(T, f(x[j]))
      above = y > τ
      s += ifelse(above, y, zero(T))
      c += above
    end
    τ = (s - a) / c
    c == cnt && break
    cnt = c
  end
  τ
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

@inline _norm_difference(a, b) = a - b
@inline _norm_difference(a::Integer, b::Integer) = signed(widen(a)) - signed(widen(b))
@inline _norm_difference(a::Complex, b::Complex) = complex(_norm_difference(real(a), real(b)), _norm_difference(imag(a), imag(b)))
@inline _norm_difference(a::Complex, b::Real) = complex(_norm_difference(real(a), b), imag(a))
@inline _norm_difference(a::Real, b::Complex) = complex(_norm_difference(a, real(b)), _norm_difference(0, imag(b)))

"""
    norm_diff(A, B, p = 2)

`‖A - B‖` without allocating `A - B`: the Euclidean (Frobenius) norm for
`p = 2`, the largest absolute entry for `p = Inf`. The result is real, also
for complex arrays.
"""
function norm_diff(A::AbstractArray, B::AbstractArray, p::Real = 2)
  size(A) == size(B) || throw(DimensionMismatch())
  T = float(promote_type(eltype(A), eltype(B)))
  v = real(zero(T))
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
# arrays also have a zero sum of squares; a vectorized sum of |real| + |imag|
# of the differences confirms them without hypot.
function _norm_diff2(A::AbstractArray, B::AbstractArray)
  s = _sumabs2_diff(A, B)
  T = typeof(s)
  S = float(promote_type(eltype(A), eltype(B)))
  isfinite(s) && s >= floatmin(T) && return sqrt(s)
  if iszero(s)
    a = zero(T)
    @inbounds @simd for i in eachindex(A, B)
      d = S(_norm_difference(A[i], B[i]))
      a += abs(real(d)) + abs(imag(d))
    end
    iszero(a) && return s
  end
  v = zero(T)
  @inbounds for i in eachindex(A, B)
    v = hypot(v, abs(S(_norm_difference(A[i], B[i]))))
  end
  v
end

# Σᵢ (xᵢ - yᵢ)², without allocating x - y
function _sumabs2_diff(x::AbstractArray, y::AbstractArray)
  size(x) == size(y) || throw(DimensionMismatch("the arrays have sizes $(size(x)) and $(size(y))"))
  T = float(promote_type(eltype(x), eltype(y)))
  s = real(zero(T))
  @inbounds @simd for i in eachindex(x, y)
    s += abs2(T(_norm_difference(x[i], y[i])))
  end
  s
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
