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
    _copyto_rounded!(Y, ws.A)
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
# n × n matrix buffer when BLAS cannot write Y or Y has a different element type
function _rebuild_sym_out!(Y::AbstractMatrix, V::StridedMatrix, λ, f::F, buffer::StridedMatrix;
                           logs::Bool = false) where {F}
  Y isa StridedMatrix{eltype(V)} && _blas_ready(Y) && return _rebuild_sym!(Y, V, λ, f; logs)
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
When an intermediate sum overflows although the weighted value is representable,
the value is computed in scaled units. The domain is the finite matrices: the
value is `Inf` at a matrix with a non-finite entry, also for `λ = 0`.

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
_argument_type(::Type{<:NuclearNorm}) = AbstractMatrix
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, ::NuclearNorm, Y::AbstractMatrix, X::AbstractMatrix) =
  (_checksize(Y, X); true)
isconvex(::Type{<:NuclearNorm}) = true

function (f::NuclearNorm)(X::AbstractMatrix)
  T = float(eltype(X))
  all(isfinite, X) || return T(Inf)              # the domain is the finite matrices
  iszero(f.λ) && return zero(T)
  ws = _svd_workspace(f.ws, T, size(X)...)
  f.ws = ws
  copyto!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  v = convert(T, f.λ) * sum(s)
  isfinite(v) || !isfinite(f.λ) || !all(isfinite, X) ? v : _scaled_nuclear_value!(ws, X, convert(T, f.λ))
end

# the exponent from the workspace's copy, which also serves inputs on devices
@noinline function _scaled_nuclear_value!(ws::SVDWorkspace{T}, X::AbstractMatrix, λ::T) where {T}
  copyto!(ws.A, X)
  k = _magnitude_exponent(_maximum_abs(ws.A))
  _scaled_spectral_input!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  _scaled_product(λ, sum(s), k)
end

@inline function _spectral_prox_value(v, λ::T, s::AbstractVector{T}, c::T) where {T}
  isfinite(v) || !isfinite(λ) || !isfinite(c) || !all(isfinite, s) ? v : _scaled_spectral_prox_value(λ, s, c)
end

@noinline function _scaled_spectral_prox_value(λ::T, s::AbstractVector{T}, c::T) where {T}
  k = _magnitude_exponent(_maximum_abs(s))
  scale = _power_factors(T, -k)
  threshold = _scale_value(c, scale)
  v = _scaled_value_sum(s, scale, t -> max(t - threshold, zero(T)))
  _scaled_product(λ, v, k)
end

function value_and_prox!(Y::AbstractMatrix, f::NuclearNorm, X::AbstractMatrix, γ::Real)
  _checkwrite(value_and_prox!, f, Y, X)
  T = float(eltype(Y))
  if iszero(f.λ)
    Y === X || copyto!(Y, X)
    return zero(T)
  end
  c = convert(T, γ) * convert(T, f.λ)
  if f.partial
    pv = _partial_nuclear!(Y, f, X, c)
    pv === nothing || return pv
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
  _spectral_prox_value(convert(T, f.λ) * v, convert(T, f.λ), s, c)
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

Membership is reliable for every finite input within the floating-point
tolerance; non-finite entries are not members.

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
_argument_type(::Type{<:IndBallRank}) = AbstractMatrix
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, ::IndBallRank, Y::AbstractMatrix, X::AbstractMatrix) =
  (_checksize(Y, X); true)
_isindicator(::Type{IndBallRank}) = true

function (f::IndBallRank)(X::AbstractMatrix)
  T = float(eltype(X))
  all(isfinite, X) || return T(Inf)
  minimum(size(X)) <= f.r && return zero(T)
  ws = _svd_workspace(f.ws, T, size(X)...)
  f.ws = ws
  copyto!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  _finite_value(s[1]) || return _rank_scaled_value(ws, X, f.r)
  s[f.r + 1] <= _tol(T) * max(one(T), s[1]) ? zero(T) : T(Inf)
end

@noinline function _rank_scaled_value(ws::SVDWorkspace{T}, X::AbstractMatrix, r::Int) where {T}
  α = _scaled_spectral_input!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  s[r + 1] <= _tol(T) * max(α, s[1]) ? zero(T) : T(Inf)
end

function prox!(Y::AbstractMatrix, f::IndBallRank, X::AbstractMatrix, ::Real)
  _checkwrite(prox!, f, Y, X)
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
#  orthonormal columns
####################################

"""
    IndStiefel()

The indicator of `St(p, k) = {X ∈ ℝ^{p×k} : XᵀX = I_k}`, for `k ≤ p`,
nonconvex. Its proximal operator is the Frobenius projection `UVᵀ` from the
thin SVD `X = UΣVᵀ`, independent of the step. For `k = p` this is the
orthogonal group, including reflections. Use [`IndSphereL2`](@ref)`(1)` for
vectors. Matrices with no columns belong to the set.

The projection is unique at full column rank, but becomes sensitive as the
smallest singular value approaches zero. At rank-deficient inputs it uses
the orthonormal completion returned by LAPACK's SVD. This choice is
deterministic for a given input, LAPACK backend and configuration, but need
not be continuous across rank-deficient inputs.

The value is `0` when `‖XᵀX - I‖_F ≤ sqrt(eps(T)) √k`, with
`T = float(eltype(X))`, and `Inf` otherwise. This tolerance is not a
feasibility certificate. Nonfinite entries give value `Inf`; projection
throws an `ArgumentError` before writing the output. Arguments must be real
matrices, and projection outputs must have element type `Float32` or
`Float64`. Integer arguments are converted. Inputs are scaled before
conversion to the output's precision, since positive scaling leaves the
projection unchanged.

The function owns its SVD and Gram buffers. Warmed calls to the value,
[`prox!`](@ref) and [`value_and_prox!`](@ref) allocate nothing for dense
`Float32` or `Float64` CPU matrices of fixed types and sizes, including
proximal calls whose output is the input. A new size replaces the buffers;
do not share one function object between threads.
"""
mutable struct IndStiefel <: AbstractFunction
  ws::Any
  input::Scratch
  gram::Scratch
end
IndStiefel() = IndStiefel(nothing, Scratch(), Scratch())

isproximable(::Type{IndStiefel}) = true
_isindicator(::Type{IndStiefel}) = true

function _checkparams(::IndStiefel, X)
  X isa AbstractMatrix || throw(ArgumentError("IndStiefel requires a matrix"))
  eltype(X) <: Real || throw(ArgumentError("IndStiefel requires real entries"))
  p, k = size(X)
  k <= p || throw(DimensionMismatch("IndStiefel requires k ≤ p, got $p × $k"))
  true
end

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndStiefel, Y, X)
  _checkparams(f, X)
  Y isa AbstractMatrix || throw(DimensionMismatch("the output must be a matrix"))
  _checksize(Y, X)
  T = eltype(Y)
  T === Float32 || T === Float64 || throw(ArgumentError("IndStiefel requires a Float32 or Float64 output"))
  true
end

function (f::IndStiefel)(X)
  _checkparams(f, X)
  T = float(eltype(X))
  isfinite(_maximum_abs(X)) || return T(Inf)
  k = size(X, 2)
  k == 0 && return zero(T)
  A = _array_buffer!(f.input, T, size(X))
  copyto!(A, X)                       # convert integers before their products
  G = _array_buffer!(f.gram, T, (k, k))
  mul!(G, transpose(A), A)
  @inbounds for j in 1:k
    G[j, j] -= one(T)
  end
  _ind(_norm2(G) <= _tol(T) * sqrt(T(k)), T)
end

function prox!(Y, f::IndStiefel, X, ::Real)
  _checkwrite(prox!, f, Y, X)
  m = _maximum_abs(X)
  isfinite(m) || throw(ArgumentError("IndStiefel requires finite entries for projection"))
  k = size(X, 2)
  k == 0 && return Y
  ws = _svd_workspace(f.ws, eltype(Y), size(X)...)
  f.ws = ws
  if iszero(m)
    fill!(ws.A, zero(eltype(Y)))
  else
    e = _magnitude_exponent(m)
    # Scale before conversion so a narrower output keeps the input's direction.
    @inbounds for i in eachindex(ws.A, X)
      ws.A[i] = ldexp(float(X[i]), -e)
    end
  end
  _, U, Vt = _svd!(ws)
  _mul_lowrank!(Y, U, Vt, k, ws)
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

Membership is reliable for every finite input within the floating-point
tolerance; non-finite entries are not members.

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
  out::Scratch    # the result, for an output BLAS cannot write
end
function IndPSD(; rank::Union{Nothing, Integer} = nothing, partial::Bool = false)
  rank === nothing && return IndPSD{Nothing}(nothing, partial, nothing, nothing, nothing, Scratch())
  rank >= 1 || throw(ArgumentError("rank must be positive, got $rank"))
  IndPSD{Int}(Int(rank), partial, nothing, nothing, nothing, Scratch())
end

isproximable(::Type{<:IndPSD}) = true
_argument_type(::Type{<:IndPSD}) = AbstractMatrix
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
  all(isfinite, ws.A) || return false
  λ = _eigvals!(ws)
  n == 0 && return true
  _finite_value(λ[1]) && _finite_value(λ[n]) || return _scaled_ispsd!(ws, X, r)
  tol = _tol(T) * max(one(T), abs(λ[1]), abs(λ[n]))
  λ[1] >= -tol || return false
  r === nothing || count(>(tol), λ) <= r
end

@noinline function _scaled_ispsd!(ws::SymEigenWorkspace{T}, X::AbstractMatrix, r::Union{Nothing, Int}) where {T}
  α = _scaled_spectral_input!(ws.A, X)
  λ = _eigvals!(ws)
  tol = _tol(T) * max(α, abs(λ[1]), abs(λ[end]))
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
  _rebuild_sym_out!(Y, V, λ, t -> max(t, zero(t)), _blas_ready(Y) ? Y : _array_buffer!(f.out, T, (n, n)))
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
  _rebuild_sym_out!(Y, Z, λ, t -> max(t, zero(t)), _blas_ready(Y) ? Y : _array_buffer!(f.out, T, (n, n)))
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
When an intermediate sum overflows although the weighted value is representable,
the value is computed in scaled units.

Membership in the positive semidefinite cone is reliable for every finite input
within the floating-point tolerance; non-finite entries are not members.

$_PARTIAL_NOTE

$_SPECTRAL_NOTE
"""
mutable struct TracePSD{T<:Real} <: AbstractFunction
  λ::T
  partial::Bool
  eig::Any
  pws::Any        # the workspace of the partial eigendecomposition
  out::Scratch
  diag::Scratch

  TracePSD{T}(λ::T; partial::Bool = false) where {T<:Real} =
    (_checknonneg("λ", λ); new{T}(λ, partial, nothing, nothing, Scratch(), Scratch()))
end
TracePSD(λ::Real = 1; partial::Bool = false) = TracePSD{typeof(λ)}(λ; partial)

isproximable(::Type{<:TracePSD}) = true
_argument_type(::Type{<:TracePSD}) = AbstractMatrix
isconvex(::Type{<:TracePSD}) = true

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::TracePSD, Y::AbstractMatrix, X::AbstractMatrix) =
  (_checksize(Y, X); checksquare(X); true)

_trace!(::TracePSD, X::AbstractMatrix) = tr(X)
function _trace!(f::TracePSD, X::Union{SparseIterate{<:Any, 2}, SymmetricSparseIterate})
  n = checksquare(X)
  d = _buffer!(f.diag, eltype(X), n)
  length(d) == n || resize!(d, n)
  @inbounds for i in 1:n
    d[i] = X[i, i]
  end
  sum(d)
end

function (f::TracePSD)(X::AbstractMatrix)
  T = float(eltype(X))
  _ispsd!(f, X, nothing) || return T(Inf)
  v = convert(T, f.λ) * _trace!(f, X)
  isfinite(v) || !isfinite(f.λ) ? v : _scaled_trace_value(f, X)
end

@noinline function _scaled_trace_value(f::TracePSD, X::AbstractMatrix)
  T = float(eltype(X))
  k = _magnitude_exponent(_maximum_abs(X))
  scale = _power_factors(T, -k)
  v = e = zero(T)
  @inbounds for i in axes(X, 1)
    v, e = _compensated_add(v, e, _scale_value(X[i, i], scale))
  end
  _scaled_product(convert(T, f.λ), v + e, k)
end

function value_and_prox!(Y::AbstractMatrix, f::TracePSD, X::AbstractMatrix, γ::Real)
  _checkwrite(value_and_prox!, f, Y, X)
  T = float(eltype(Y))
  c = convert(T, γ) * convert(T, f.λ)
  if f.partial
    pv = _partial_psd!(Y, f, X, c, 0, convert(T, f.λ))
    pv === nothing || return pv
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
  _rebuild_sym_out!(Y, V, λ, t -> max(t - c, zero(t)), _blas_ready(Y) ? Y : _array_buffer!(f.out, T, (n, n)))
  f.partial && _warm_start_eigen!(f, V, count(>(c), λ))
  _spectral_prox_value(convert(T, f.λ) * s, convert(T, f.λ), λ, c)
end
prox!(Y::AbstractMatrix, f::TracePSD, X::AbstractMatrix, γ::Real) =
  (_checkwrite(prox!, f, Y, X); value_and_prox!(Y, f, X, γ); Y)
