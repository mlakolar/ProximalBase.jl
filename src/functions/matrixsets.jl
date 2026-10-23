####################################
#
#  Sets of structured matrices
#
####################################

####################################
#  a zero pattern
####################################

"""
    IndPattern(P)

The indicator of the arrays supported on a pattern: those whose entries outside
the pattern are zero. `P` is an array of `Bool`s, whose `true` entries form the
pattern, or any other array, whose nonzero entries do, such as a
`SparseMatrixCSC`, a [`SymmetricSparseIterate`](@ref) or an estimate whose
support a refit keeps. The pattern is stored as a `BitArray`; a `BitArray`
argument is used as it is, not copied.

Its proximal operator zeroes the entries outside the pattern. It is separable
and preserves zeros, so [`prox_coordinate!`](@ref) updates any coordinate, and
on a sparse iterate the value and the proximal operator touch only the stored
entries. With [`LogDetLoss`](@ref) in ADMM it gives the Gaussian maximum
likelihood estimate on a known graph. On a `SymmetricSparseIterate` the pattern
must be symmetric.
"""
struct IndPattern{N} <: SeparableFunction
  P::BitArray{N}
end
IndPattern(P::Array{Bool}) = IndPattern(BitArray(P))

# the nonzero entries of X, which for an array of Bools are its true entries
function IndPattern(X::AbstractArray)
  P = falses(size(X))
  for I in CartesianIndices(X)
    P[I] = !iszero(X[I])
  end
  IndPattern(P)
end
function IndPattern(X::SparseMatrixCSC)
  P = falses(size(X))
  rows, vals = rowvals(X), nonzeros(X)
  for c in axes(X, 2), k in nzrange(X, c)
    iszero(vals[k]) || (P[rows[k], c] = true)
  end
  IndPattern(P)
end
function IndPattern(X::SymmetricSparseIterate)
  P = falses(size(X))
  for k in 1:X.nnz
    iszero(X.nzval[k]) && continue
    r, c = tril_cartesian(X.p, X.nzval2ind[k])
    P[r, c] = P[c, r] = true
  end
  IndPattern(P)
end
function IndPattern(X::SparseIterate)
  P = falses(size(X))
  for k in 1:X.nnz
    iszero(X.nzval[k]) || (P[X.nzval2ind[k]] = true)
  end
  IndPattern(P)
end

isconvex(::Type{<:IndPattern}) = true
iszeropreserving(::IndPattern) = true
_isindicator(::Type{<:IndPattern}) = true
_checkparams(f::IndPattern, x) = (_checkparam(f.P, x); true)

function _value_dense(f::IndPattern, x::AbstractArray)
  T = float(eltype(x))
  t = _tol(T)
  ok = true
  @inbounds for i in eachindex(IndexLinear(), x)
    ok &= f.P[i] | (abs(x[i]) <= t)
  end
  ok ? zero(T) : T(Inf)
end

Base.@propagate_inbounds value_entry(f::IndPattern, v::T, i::Integer) where {T<:Real} =
  (f.P[i] | (abs(v) <= _tol(float(T)))) ? zero(float(T)) : float(T)(Inf)
Base.@propagate_inbounds prox_entry(f::IndPattern, v::T, i::Integer, ::Real) where {T<:Real} =
  ifelse(f.P[i], float(v), zero(float(T)))

# an entry on the pattern is free, and one off it is fixed at zero
kkt_violation(f::IndPattern, g::AbstractArray, i::Integer) =
  (T = float(eltype(g)); f.P[i] ? T(abs(g[i])) : T(-Inf))
restrict(f::IndPattern, S::AbstractVector{<:Integer}) = IndPattern(f.P[S])

####################################
#  banded matrices
####################################

"""
    IndBanded(k, p)
    IndBanded(k, (m, n))
    IndBanded((kl, ku), (m, n))

The indicator of the banded `m × n` matrices, `p × p` in the first form: those
whose entries more than `kl` below the diagonal or more than `ku` above it are
zero, with `kl = ku = k` for a single bandwidth. `k = 0` gives the diagonal
matrices. Its proximal operator zeroes the entries outside the band, which for
a sample covariance is banding (Bickel and Levina, 2008).

The size is part of the function, so that the entry with linear index `i` has a
known row and column: like [`IndPattern`](@ref), it is separable and preserves
zeros, without storing a pattern, and it works on dense and `Symmetric`
matrices of that size and on the sparse iterates. On a
[`SymmetricSparseIterate`](@ref) the band must be symmetric, `kl = ku`.
"""
struct IndBanded <: SeparableFunction
  kl::Int
  ku::Int
  m::Int
  n::Int

  function IndBanded(kl::Integer, ku::Integer, m::Integer, n::Integer)
    kl >= 0 && ku >= 0 || throw(ArgumentError("the bandwidths must be nonnegative, got $kl and $ku"))
    m >= 0 && n >= 0 || throw(ArgumentError("the dimensions must be nonnegative, got $m × $n"))
    new(kl, ku, m, n)
  end
end
IndBanded(k::Integer, p::Integer) = IndBanded(k, k, p, p)
IndBanded(k::Integer, dims::Tuple{Integer, Integer}) = IndBanded(k, k, dims...)
IndBanded(k::Tuple{Integer, Integer}, dims::Tuple{Integer, Integer}) = IndBanded(k..., dims...)

isconvex(::Type{IndBanded}) = true
iszeropreserving(::IndBanded) = true
_isindicator(::Type{IndBanded}) = true

_checkparams(f::IndBanded, x) = size(x) == (f.m, f.n) ||
  throw(DimensionMismatch("the band is for $(f.m) × $(f.n) matrices, and the argument has size $(size(x))"))
function _checkparams(f::IndBanded, x::SymmetricSparseIterate)
  f.kl == f.ku || throw(ArgumentError("a SymmetricSparseIterate needs a symmetric band, got kl = $(f.kl) and ku = $(f.ku)"))
  invoke(_checkparams, Tuple{IndBanded, Any}, f, x)
end

# whether the entry with linear index i, at row r and column c, is in the band
@inline function _inband(f::IndBanded, i::Integer)
  c = div(i - 1, f.m) + 1
  r = i - (c - 1) * f.m
  (r - c <= f.kl) & (c - r <= f.ku)
end

Base.@propagate_inbounds value_entry(f::IndBanded, v::T, i::Integer) where {T<:Real} =
  (_inband(f, i) | (abs(v) <= _tol(float(T)))) ? zero(float(T)) : float(T)(Inf)
Base.@propagate_inbounds prox_entry(f::IndBanded, v::T, i::Integer, ::Real) where {T<:Real} =
  ifelse(_inband(f, i), float(v), zero(float(T)))

kkt_violation(f::IndBanded, g::AbstractArray, i::Integer) =
  (T = float(eltype(g)); _inband(f, i) ? T(abs(g[i])) : T(-Inf))
# a function of x[S], a vector: the pattern of the band on S
restrict(f::IndBanded, S::AbstractVector{<:Integer}) = IndPattern(BitVector(_inband(f, i) for i in S))

# on a dense matrix, by columns: the rows c - ku to c + kl of column c are in the band
function _value_dense(f::IndBanded, x::AbstractMatrix)
  T = float(eltype(x))
  t = _tol(T)
  m, n = size(x)
  ok = true
  @inbounds for c in 1:n
    lo, hi = max(1, c - f.ku), min(m, c + f.kl)
    for r in 1:min(lo - 1, m)
      ok &= abs(x[r, c]) <= t
    end
    for r in hi+1:m
      ok &= abs(x[r, c]) <= t
    end
  end
  ok ? zero(T) : T(Inf)
end

function _prox_sep!(y::StridedMatrix, f::IndBanded, x::AbstractMatrix, ::Real)
  _checksize(y, x)
  _checkparams(f, x)
  T = eltype(y)
  m, n = size(y)
  @inbounds for c in 1:n
    lo, hi = max(1, c - f.ku), min(m, c + f.kl)
    for r in 1:min(lo - 1, m)
      y[r, c] = zero(T)
    end
    for r in lo:hi
      y[r, c] = convert(T, x[r, c])
    end
    for r in hi+1:m
      y[r, c] = zero(T)
    end
  end
  y
end

####################################
#  Toeplitz matrices
####################################

"""
    IndToeplitz(; symmetric = false)

The indicator of the Toeplitz matrices, which are constant along each diagonal:
`Xᵢⱼ = c_{j-i}`. Its proximal operator replaces each diagonal by its mean. With
`symmetric = true` the set is that of the symmetric Toeplitz matrices, which are
square, and the projection averages the diagonals `d` and `-d` together. These
are the covariance matrices of a stationary series. The function is not
separable. The projection takes two passes over each diagonal, works in place,
and allocates nothing. Its output may be a [`SymmetricSparseIterate`](@ref)
when the result is symmetric: with `symmetric = true`, or from a symmetric
input.
"""
struct IndToeplitz <: AbstractFunction
  symmetric::Bool
end
IndToeplitz(; symmetric::Bool = false) = IndToeplitz(symmetric)

isproximable(::Type{IndToeplitz}) = true
_isindicator(::Type{IndToeplitz}) = true
isconvex(::Type{IndToeplitz}) = true

# the rows of the diagonal c - r = d of an m × n matrix
@inline _diagonal_rows(m::Int, n::Int, d::Int) = max(1, 1 - d):min(m, n - d)

# whether the projection is symmetric: a symmetric input averages d and -d alike
_symmetric_toeplitz(f::IndToeplitz, x::AbstractMatrix) = f.symmetric || x isa Union{Symmetric, SymmetricSparseIterate}

function (f::IndToeplitz)(x::AbstractMatrix)
  T = float(eltype(x))
  t = _tol(T)
  m, n = size(x)
  f.symmetric && m != n && return T(Inf)
  ok = true
  if f.symmetric
    @inbounds for d in 0:n-1
      μ = _symmetric_diagonal_mean(x, n, d)
      s = t * max(one(T), abs(μ))
      for r in 1:n-d
        ok &= (abs(x[r, r + d] - μ) <= s) & (abs(x[r + d, r] - μ) <= s)
      end
    end
  else
    @inbounds for d in -(m - 1):(n - 1)
      rows = _diagonal_rows(m, n, d)
      μ = _diagonal_mean(x, rows, d)
      s = t * max(one(T), abs(μ))
      for r in rows
        ok &= abs(x[r, r + d] - μ) <= s
      end
    end
  end
  ok ? zero(T) : T(Inf)
end

Base.@propagate_inbounds function _diagonal_mean(x::AbstractMatrix, rows::UnitRange{Int}, d::Int)
  s = zero(float(eltype(x)))
  for r in rows
    s += x[r, r + d]
  end
  s / length(rows)
end

# the mean of the diagonals d and -d of a square n × n matrix, d ≥ 0
Base.@propagate_inbounds function _symmetric_diagonal_mean(x::AbstractMatrix, n::Int, d::Int)
  s = zero(float(eltype(x)))
  for r in 1:n-d
    s += x[r, r + d] + x[r + d, r]
  end
  s / (2 * (n - d))
end

function prox!(y::AbstractMatrix, f::IndToeplitz, x::AbstractMatrix, ::Real)
  _checksize(y, x)
  m, n = size(x)
  if _symmetric_toeplitz(f, x)
    m == n || throw(DimensionMismatch("a symmetric Toeplitz matrix is square, and the argument is $m × $n"))
    @inbounds for d in 0:n-1
      μ = _symmetric_diagonal_mean(x, n, d)
      for r in 1:n-d
        _set_symmetric!(y, μ, r + d, r)
      end
    end
  else
    y isa SymmetricSparseIterate &&
      throw(ArgumentError("the projection of a matrix that is not symmetric onto the Toeplitz matrices is not symmetric; use a dense output"))
    @inbounds for d in -(m - 1):(n - 1)
      rows = _diagonal_rows(m, n, d)
      μ = _diagonal_mean(x, rows, d)
      for r in rows
        y[r, r + d] = μ
      end
    end
  end
  y
end
