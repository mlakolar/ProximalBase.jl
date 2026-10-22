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

# the symmetric Y = Σⱼ f(λⱼ) vⱼvⱼ' from the eigendecomposition in ws, through a
# temporary matrix when Y is not strided
function _rebuild_sym_out!(Y::AbstractMatrix, V::StridedMatrix, λ, f::F; logs::Bool = false) where {F}
  Y isa StridedMatrix && return _rebuild_sym!(Y, V, λ, f; logs)
  Z = similar(V, size(V, 1), size(V, 1))
  s = _rebuild_sym!(Z, V, λ, f; logs)
  copyto!(Y, Z)
  s
end

####################################
#  nuclear norm
####################################

"""
    NuclearNorm(λ = 1)

The nuclear norm `f(X) = λ ‖X‖_*`, the sum of the singular values of `X`. Its
proximal operator soft-thresholds the singular values, and
[`value_and_prox!`](@ref) returns the value from the thresholded singular values
without a second decomposition.

$_SPECTRAL_NOTE
"""
mutable struct NuclearNorm{T<:Real} <: AbstractFunction
  λ::T
  ws::Any

  NuclearNorm{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ, nothing))
end
NuclearNorm(λ::Real = 1) = NuclearNorm{typeof(λ)}(λ)

isproximable(::Type{<:NuclearNorm}) = true
isconvex(::Type{<:NuclearNorm}) = true

function (f::NuclearNorm)(X::AbstractMatrix)
  T = float(eltype(X))
  ws = _svd_workspace(f.ws, T, size(X)...)
  f.ws = ws
  copyto!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  convert(T, f.λ) * sum(s)
end

function value_and_prox!(Y::AbstractMatrix, f::NuclearNorm, X::AbstractMatrix, γ::Real)
  _checksize(Y, X)
  T = float(eltype(Y))
  ws = _svd_workspace(f.ws, T, size(X)...)
  f.ws = ws
  copyto!(ws.A, X)
  s, U, Vt = _svd!(ws)
  c = convert(T, γ) * convert(T, f.λ)
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
  _mul_lowrank!(Y, U, Vt, k, ws)
  convert(T, f.λ) * v
end
prox!(Y::AbstractMatrix, f::NuclearNorm, X::AbstractMatrix, γ::Real) = (value_and_prox!(Y, f, X, γ); Y)

####################################
#  rank constraint
####################################

"""
    IndBallRank(r)

The indicator of the matrices of rank at most `r`, nonconvex. Its proximal
operator is the truncated singular value decomposition, which keeps the `r`
largest singular triplets.

$_SPECTRAL_NOTE
"""
mutable struct IndBallRank <: AbstractFunction
  r::Int
  ws::Any

  IndBallRank(r::Integer) = (r >= 1 || throw(ArgumentError("r must be positive, got $r")); new(Int(r), nothing))
end

isproximable(::Type{IndBallRank}) = true

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
  _mul_lowrank!(Y, U, Vt, f.r, ws)
end
value_and_prox!(Y::AbstractMatrix, f::IndBallRank, X::AbstractMatrix, γ::Real) = (prox!(Y, f, X, γ); zero(float(eltype(Y))))

####################################
#  positive semidefinite cone
####################################

"""
    IndPSD()
    IndPSD(; rank = r)

The indicator of the positive semidefinite matrices, and with `rank` of those of
rank at most `r`, which is nonconvex. Its proximal operator is the projection of
a symmetric matrix (its upper triangle is read) onto the set: the
eigendecomposition with the negative eigenvalues set to zero, and for a rank
`r` only the `r` largest eigenpairs, which LAPACK computes without the others.
The result is rebuilt from the eigenvectors that remain, in O(n²k) for k of
them.

$_SPECTRAL_NOTE
"""
mutable struct IndPSD{R<:Union{Nothing, Int}} <: AbstractFunction
  rank::R
  eig::Any        # all eigenvalues: the value, and the projection onto the cone
  top::Any        # the largest r eigenpairs: the projection with a rank
end
function IndPSD(; rank::Union{Nothing, Integer} = nothing)
  rank === nothing && return IndPSD{Nothing}(nothing, nothing, nothing)
  rank >= 1 || throw(ArgumentError("rank must be positive, got $rank"))
  IndPSD{Int}(Int(rank), nothing, nothing)
end

isproximable(::Type{<:IndPSD}) = true
isconvex(::Type{IndPSD{Nothing}}) = true

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

function prox!(Y::AbstractMatrix, f::IndPSD{Nothing}, X::AbstractMatrix, ::Real)
  _checksize(Y, X)
  T = float(eltype(Y))
  ws = _eig_workspace(f.eig, T, checksquare(X))
  f.eig = ws
  copyto!(ws.A, X)
  λ, V = _eigen!(ws)
  _rebuild_sym_out!(Y, V, λ, t -> max(t, zero(t)))
  Y
end

function prox!(Y::AbstractMatrix, f::IndPSD{Int}, X::AbstractMatrix, γ::Real)
  _checksize(Y, X)
  n = checksquare(X)
  f.rank >= n && return prox!(Y, IndPSD(), X, γ)
  T = float(eltype(Y))
  ws = _eigtop_workspace(f.top, T, n, f.rank)
  f.top = ws
  copyto!(ws.A, X)
  λ, Z = _eigen_top!(ws)
  _rebuild_sym_out!(Y, Z, λ, t -> max(t, zero(t)))
  Y
end

value_and_prox!(Y::AbstractMatrix, f::IndPSD, X::AbstractMatrix, γ::Real) = (prox!(Y, f, X, γ); zero(float(eltype(Y))))

####################################
#  trace on the positive semidefinite cone
####################################

"""
    TracePSD(λ = 1)

The function `f(X) = λ tr(X)` on the positive semidefinite matrices, `Inf`
elsewhere: the nuclear norm restricted to them, as in the low-rank part of
latent-variable graphical models. Its proximal operator shifts the eigenvalues
of a symmetric matrix down by `γλ` and sets the negative ones to zero.

$_SPECTRAL_NOTE
"""
mutable struct TracePSD{T<:Real} <: AbstractFunction
  λ::T
  eig::Any

  TracePSD{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ, nothing))
end
TracePSD(λ::Real = 1) = TracePSD{typeof(λ)}(λ)

isproximable(::Type{<:TracePSD}) = true
isconvex(::Type{<:TracePSD}) = true

(f::TracePSD)(X::AbstractMatrix) =
  _ispsd!(f, X, nothing) ? convert(float(eltype(X)), f.λ) * tr(X) : float(eltype(X))(Inf)

function value_and_prox!(Y::AbstractMatrix, f::TracePSD, X::AbstractMatrix, γ::Real)
  _checksize(Y, X)
  T = float(eltype(Y))
  ws = _eig_workspace(f.eig, T, checksquare(X))
  f.eig = ws
  copyto!(ws.A, X)
  λ, V = _eigen!(ws)
  c = convert(T, γ) * convert(T, f.λ)
  s = zero(T)
  @inbounds for t in λ
    s += max(t - c, zero(T))
  end
  _rebuild_sym_out!(Y, V, λ, t -> max(t - c, zero(t)))
  convert(T, f.λ) * s
end
prox!(Y::AbstractMatrix, f::TracePSD, X::AbstractMatrix, γ::Real) = (value_and_prox!(Y, f, X, γ); Y)
