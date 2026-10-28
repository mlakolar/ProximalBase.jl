####################################
#
#  Spectral functions
#
####################################

# Functions of the singular values or eigenvalues of a matrix. Each caches the
# LAPACK workspace for the size and element type of its input (Float32 or
# Float64), so that its hot calls allocate nothing; a function object is
# therefore not safe to share between threads. The value of a proximal
# operator comes from the spectrum it computes.

const _SPECTRAL_NOTE = """
The function caches a LAPACK workspace for the size and element type
(`Float32` or `Float64`) of its input, so that repeated calls allocate nothing;
do not share one function object between threads."""

const _PARTIAL_NOTE = """
With `partial = true` the proximal operator computes only the leading singular
triplets or eigenpairs, by randomized subspace iteration started from those of
the previous call, in O(p²k) instead of O(p³) for k of them. It accepts the
result when the residuals of the pairs kept are below
`max(eps^(3/4), 10eps*sqrt(p))` times the largest absolute computed value. The
next three computed pairs must lie below the threshold within this tolerance,
or below the smallest retained value for a fixed rank, accounting for their
residuals. Otherwise, or when the pairs kept
would be more than about a quarter of the dimension, it falls back to the full
decomposition. These checks cannot detect a leading singular value or
eigenvalue that the iteration missed, so the result is an approximation. It
pays when the result has low rank and the calls use nearby matrices; keep the
default full decomposition when an exact proximal step is needed. See
[Partial decompositions](@ref)."""

# Y = U[:, 1:k] * Vt[1:k, :], through the workspace's matrix when Y is not strided
function _mul_lowrank!(Y::AbstractMatrix, U::Matrix, Vt::Matrix, k::Integer, ws::SVDWorkspace)
  if k == 0
    fill!(Y, zero(eltype(Y)))
  elseif Y isa StridedMatrix
    mul!(Y, view(U, :, 1:k), view(Vt, 1:k, :))
  else
    mul!(ws.A, view(U, :, 1:k), view(Vt, 1:k, :))     # ws.A is free after the SVD
    copyto!(Y, ws.A)
  end
  Y
end

# whether X is symmetric, within the tolerance of the indicator functions
function _issymmetric_tol(X::AbstractMatrix)
  n = checksquare(X)
  t = _tol(float(eltype(X)))
  @inbounds for j in 1:n, i in 1:j-1
    a, b = X[i, j], X[j, i]
    abs(a - b) <= t * max(1, abs(a), abs(b)) || return false
  end
  true
end
_issymmetric_tol(::SymmetricSparseIterate) = true

# the symmetric Y = Σⱼ f(λⱼ) vⱼvⱼ' from the eigendecomposition V, λ, through the
# n × n matrix buffer when Y is not strided (a SymmetricSparseIterate, say)
function _rebuild_sym_out!(Y::AbstractMatrix, V::StridedMatrix, λ, f::F, buffer::StridedMatrix;
                           logs::Bool = false) where {F}
  Y isa StridedMatrix && return _rebuild_sym!(Y, V, λ, f; logs)
  s = _rebuild_sym!(buffer, V, λ, f; logs)
  copyto!(Y, buffer)
  s
end

####################################
#  nuclear norm
####################################

"""
    NuclearNorm(λ = 1; partial = false)

The nuclear norm `f(X) = λ ‖X‖_*`, the sum of the singular values of `X`. Its
proximal operator soft-thresholds the singular values, and
[`value_and_prox!`](@ref) returns the value from the thresholded singular values
without a second decomposition.

$_PARTIAL_NOTE The value `f(X)` alone needs every singular value and always
takes the full decomposition.

$_SPECTRAL_NOTE
"""
mutable struct NuclearNorm{T<:Real} <: AbstractFunction
  λ::T
  partial::Bool
  ws::Any
  pws::Any        # the workspace of the partial SVD

  NuclearNorm{T}(λ::T; partial::Bool = false) where {T<:Real} =
    (_checknonneg("λ", λ); new{T}(λ, partial, nothing, nothing))
end
NuclearNorm(λ::Real = 1; partial::Bool = false) = NuclearNorm{typeof(λ)}(λ; partial)

isproximable(::Type{<:NuclearNorm}) = true
isconvex(::Type{<:NuclearNorm}) = true

function (f::NuclearNorm)(X::AbstractMatrix)
  T = float(eltype(X))
  iszero(f.λ) && return zero(T)
  ws = _svd_workspace(f.ws, T, size(X)...)
  f.ws = ws
  copyto!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  convert(T, f.λ) * sum(s)
end

function value_and_prox!(Y::AbstractMatrix, f::NuclearNorm, X::AbstractMatrix, γ::Real)
  _checksize(Y, X)
  T = float(eltype(Y))
  if iszero(f.λ)
    Y === X || copyto!(Y, X)
    return zero(T)
  end
  c = convert(T, γ) * convert(T, f.λ)
  if f.partial
    pv = _partial_nuclear!(Y, f, X, c)
    pv === nothing || return convert(T, f.λ) * pv
  end
  ws = _svd_workspace(f.ws, T, size(X)...)
  f.ws = ws
  copyto!(ws.A, X)
  s, U, Vt = _svd!(ws)
  k = 0
  v = zero(T)
  @inbounds for j in eachindex(s)
    t = s[j] - c
    t > 0 || break                   # the singular values are in descending order
    k = j
    v += t
    @simd for i in axes(U, 1)
      U[i, j] *= t
    end
  end
  f.partial && _warm_start_svd!(f, U, k)
  _mul_lowrank!(Y, U, Vt, k, ws)
  convert(T, f.λ) * v
end
prox!(Y::AbstractMatrix, f::NuclearNorm, X::AbstractMatrix, γ::Real) = (value_and_prox!(Y, f, X, γ); Y)

####################################
#  rank constraint
####################################

"""
    IndBallRank(r; partial = false)

The indicator of the matrices of rank at most `r`, nonconvex. Its proximal
operator is the truncated singular value decomposition, which keeps the `r`
largest singular triplets.

$_PARTIAL_NOTE

$_SPECTRAL_NOTE
"""
mutable struct IndBallRank <: AbstractFunction
  r::Int
  partial::Bool
  ws::Any
  pws::Any        # the workspace of the partial SVD

  function IndBallRank(r::Integer; partial::Bool = false)
    r >= 1 || throw(ArgumentError("r must be positive, got $r"))
    new(Int(r), partial, nothing, nothing)
  end
end

isproximable(::Type{IndBallRank}) = true
_isindicator(::Type{IndBallRank}) = true

function (f::IndBallRank)(X::AbstractMatrix)
  T = float(eltype(X))
  minimum(size(X)) <= f.r && return zero(T)
  ws = _svd_workspace(f.ws, T, size(X)...)
  f.ws = ws
  copyto!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  s[f.r + 1] <= _tol(T) * max(one(T), s[1]) ? zero(T) : T(Inf)
end

function prox!(Y::AbstractMatrix, f::IndBallRank, X::AbstractMatrix, ::Real)
  _checksize(Y, X)
  T = float(eltype(Y))
  if minimum(size(X)) <= f.r
    Y === X || copyto!(Y, X)
    return Y
  end
  f.partial && _partial_rank!(Y, f, X, T) !== nothing && return Y
  ws = _svd_workspace(f.ws, T, size(X)...)
  f.ws = ws
  copyto!(ws.A, X)
  s, U, Vt = _svd!(ws)
  @inbounds for j in 1:f.r
    t = s[j]
    @simd for i in axes(U, 1)
      U[i, j] *= t
    end
  end
  f.partial && _warm_start_svd!(f, U, f.r)
  _mul_lowrank!(Y, U, Vt, f.r, ws)
end

####################################
#  positive semidefinite cone
####################################

"""
    IndPSD(; partial = false)
    IndPSD(; rank = r, partial = false)

The indicator of the positive semidefinite matrices, and with `rank` of those of
rank at most `r`, which is nonconvex. Its proximal operator is the projection of
a symmetric matrix (its upper triangle is read) onto the set: the
eigendecomposition with the negative eigenvalues set to zero, and for a rank
`r` only the `r` largest eigenpairs, which LAPACK computes without the others.
The result is rebuilt from the eigenvectors that remain, in O(n²k) for k of
them.

$_PARTIAL_NOTE Without a rank, the partial projection pays when few
eigenvalues are positive.

$_SPECTRAL_NOTE
"""
mutable struct IndPSD{R<:Union{Nothing, Int}} <: AbstractFunction
  rank::R
  partial::Bool
  eig::Any        # all eigenvalues: the value, and the projection onto the cone
  top::Any        # the largest r eigenpairs: the projection with a rank
  pws::Any        # the workspace of the partial eigendecomposition
  out::Scratch    # the result, for an output that is not strided
end
function IndPSD(; rank::Union{Nothing, Integer} = nothing, partial::Bool = false)
  rank === nothing && return IndPSD{Nothing}(nothing, partial, nothing, nothing, nothing, Scratch())
  rank >= 1 || throw(ArgumentError("rank must be positive, got $rank"))
  IndPSD{Int}(Int(rank), partial, nothing, nothing, nothing, Scratch())
end

isproximable(::Type{<:IndPSD}) = true
_isindicator(::Type{<:IndPSD}) = true
isconvex(::Type{IndPSD{Nothing}}) = true

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::Union{IndPSD{Nothing}, IndPSD{Int}}, Y::AbstractMatrix, X::AbstractMatrix) =
  (_checksize(Y, X); checksquare(X); true)

# whether X is symmetric positive semidefinite, and of rank at most r unless r is
# nothing, within the tolerance of the indicator functions; f caches the workspace
function _ispsd!(f, X::AbstractMatrix, r::Union{Nothing, Int})
  T = float(eltype(X))
  n = checksquare(X)
  _issymmetric_tol(X) || return false
  ws = _eig_workspace(f.eig, T, n)
  f.eig = ws
  copyto!(ws.A, X)
  λ = _eigvals!(ws)
  n == 0 && return true
  tol = _tol(T) * max(one(T), abs(λ[1]), abs(λ[n]))
  λ[1] >= -tol || return false
  r === nothing || count(>(tol), λ) <= r
end

(f::IndPSD)(X::AbstractMatrix) = _ispsd!(f, X, f.rank) ? zero(float(eltype(X))) : float(eltype(X))(Inf)

# the projection onto the cone through all eigenpairs, in the workspace f.eig
function _project_psd!(Y::AbstractMatrix, f::IndPSD, X::AbstractMatrix)
  _checksize(Y, X)
  T = float(eltype(Y))
  ws = _eig_workspace(f.eig, T, checksquare(X))
  f.eig = ws
  copyto!(ws.A, X)
  λ, V = _eigen!(ws)
  n = size(V, 1)
  _rebuild_sym_out!(Y, V, λ, t -> max(t, zero(t)), Y isa StridedMatrix ? Y : _array_buffer!(f.out, T, (n, n)))
  f.partial && _warm_start_eigen!(f, V, count(>(zero(T)), λ))
  Y
end

function prox!(Y::AbstractMatrix, f::IndPSD{Nothing}, X::AbstractMatrix, ::Real)
  _checkwrite(prox!, f, Y, X)
  if f.partial
    _partial_psd!(Y, f, X, zero(float(eltype(Y))), 0) === nothing || return Y
  end
  _project_psd!(Y, f, X)
end

function prox!(Y::AbstractMatrix, f::IndPSD{Int}, X::AbstractMatrix, γ::Real)
  _checkwrite(prox!, f, Y, X)
  n = checksquare(X)
  f.rank >= n && return _project_psd!(Y, f, X)
  T = float(eltype(Y))
  f.partial && _partial_psd!(Y, f, X, zero(T), f.rank) !== nothing && return Y
  ws = _eigtop_workspace(f.top, T, n, f.rank)
  f.top = ws
  copyto!(ws.A, X)
  λ, Z = _eigen_top!(ws)
  _rebuild_sym_out!(Y, Z, λ, t -> max(t, zero(t)), Y isa StridedMatrix ? Y : _array_buffer!(f.out, T, (n, n)))
  f.partial && _warm_start_eigen!(f, Z, count(>(zero(T)), λ))
  Y
end


####################################
#  trace on the positive semidefinite cone
####################################

"""
    TracePSD(λ = 1; partial = false)

The function `f(X) = λ tr(X)` on the positive semidefinite matrices, `Inf`
elsewhere: the nuclear norm restricted to them, as in the low-rank part of
latent-variable graphical models. Its proximal operator shifts the eigenvalues
of a symmetric matrix down by `γλ` and sets the negative ones to zero.

$_PARTIAL_NOTE

$_SPECTRAL_NOTE
"""
mutable struct TracePSD{T<:Real} <: AbstractFunction
  λ::T
  partial::Bool
  eig::Any
  pws::Any        # the workspace of the partial eigendecomposition
  out::Scratch

  TracePSD{T}(λ::T; partial::Bool = false) where {T<:Real} =
    (_checknonneg("λ", λ); new{T}(λ, partial, nothing, nothing, Scratch()))
end
TracePSD(λ::Real = 1; partial::Bool = false) = TracePSD{typeof(λ)}(λ; partial)

isproximable(::Type{<:TracePSD}) = true
isconvex(::Type{<:TracePSD}) = true

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::TracePSD, Y::AbstractMatrix, X::AbstractMatrix) =
  (_checksize(Y, X); checksquare(X); true)

(f::TracePSD)(X::AbstractMatrix) =
  _ispsd!(f, X, nothing) ? convert(float(eltype(X)), f.λ) * tr(X) : float(eltype(X))(Inf)

function value_and_prox!(Y::AbstractMatrix, f::TracePSD, X::AbstractMatrix, γ::Real)
  _checkwrite(value_and_prox!, f, Y, X)
  T = float(eltype(Y))
  c = convert(T, γ) * convert(T, f.λ)
  if f.partial
    pv = _partial_psd!(Y, f, X, c, 0)
    pv === nothing || return convert(T, f.λ) * pv
  end
  ws = _eig_workspace(f.eig, T, checksquare(X))
  f.eig = ws
  copyto!(ws.A, X)
  λ, V = _eigen!(ws)
  s = zero(T)
  @inbounds for t in λ
    s += max(t - c, zero(T))
  end
  n = size(V, 1)
  _rebuild_sym_out!(Y, V, λ, t -> max(t - c, zero(t)), Y isa StridedMatrix ? Y : _array_buffer!(f.out, T, (n, n)))
  f.partial && _warm_start_eigen!(f, V, count(>(c), λ))
  convert(T, f.λ) * s
end
prox!(Y::AbstractMatrix, f::TracePSD, X::AbstractMatrix, γ::Real) =
  (_checkwrite(prox!, f, Y, X); value_and_prox!(Y, f, X, γ); Y)
