####################################
#
#  Gaussian negative log-likelihood  w (tr(SX) - log det X)
#
####################################

"""
    LogDetLoss(S, w = 1)

The Gaussian negative log-likelihood `f(X) = w (tr(SX) - log det X)` for finite
`w > 0`, of a
precision matrix `X`, for a symmetric `S` such as a sample covariance, and
`Inf` when `X` is not positive definite. The loss of the graphical lasso.

It is smooth on the positive definite matrices, with gradient `w(S - X⁻¹)`,
which [`value_and_gradient!`](@ref) computes from one Cholesky factorization;
the value never throws, so that a line search can reject a step that leaves the
cone. It is also proximable: with `μ = γw` and the eigendecomposition
`X - μS = U Diagonal(t) U'`, `prox_{γf}(X) = U Diagonal(z) U'`, where
`zᵢ = (tᵢ + √(tᵢ² + 4μ))/2`, evaluated without overflow or cancellation.
[`value_and_prox!`](@ref) returns the value from the `zᵢ`.
Both use workspaces that `f` owns, sized by `S`, and
allocate nothing; do not share `f` between threads. The element type of `X`
must be that of `S`, `Float32` or `Float64`.
"""
struct LogDetLoss{T<:_BlasReal} <: AbstractFunction
  S::Matrix{T}
  w::T
  eig::SymEigenWorkspace{T}     # the proximal operator
  C::Matrix{T}                  # Cholesky factors and inverses

  function LogDetLoss{T}(S::Matrix{T}, w::T) where {T<:_BlasReal}
    isfinite(w) && w > 0 || throw(ArgumentError("w must be finite and positive, got $w"))
    p = checksquare(S)
    new{T}(S, w, SymEigenWorkspace(T, p), zeros(T, p, p))
  end
end

function LogDetLoss(S::AbstractMatrix, w::Real = 1)
  T = float(promote_type(eltype(S), typeof(w)))
  T <: _BlasReal || throw(ArgumentError("LogDetLoss needs Float32 or Float64 entries, got $T"))
  LogDetLoss{T}(Matrix{T}(Symmetric(S)), convert(T, w))
end

issmooth(::Type{<:LogDetLoss}) = true
isproximable(::Type{<:LogDetLoss}) = true
isconvex(::Type{<:LogDetLoss}) = true

# log det X from the Cholesky factor of X in f.C, or nothing if X is not
# positive definite
function _logdet_chol!(f::LogDetLoss{T}, X::AbstractMatrix) where {T}
  size(X) == size(f.S) || throw(DimensionMismatch("X has size $(size(X)) and S $(size(f.S))"))
  C = f.C
  copyto!(C, X)
  _, info = LAPACK.potrf!('U', C)
  info == 0 || return nothing
  ld = zero(T)
  @inbounds for i in axes(C, 1)
    ld += log(C[i, i])
  end
  2ld
end

function (f::LogDetLoss{T})(X::AbstractMatrix) where {T}
  ld = _logdet_chol!(f, X)
  ld === nothing ? T(Inf) : f.w * (dot(f.S, X) - ld)
end

function value_and_gradient!(G::AbstractMatrix, f::LogDetLoss{T}, X::AbstractMatrix) where {T}
  _checksize(G, X)
  ld = _logdet_chol!(f, X)
  if ld === nothing
    fill!(G, T(NaN))
    return T(Inf)
  end
  v = f.w * (dot(f.S, X) - ld)
  C, S, w = f.C, f.S, f.w
  LAPACK.potri!('U', C)             # the upper triangle of X⁻¹
  @inbounds for j in axes(C, 2), i in 1:j
    g = w * (S[i, j] - C[i, j])
    G[i, j] = g
    G[j, i] = g
  end
  v
end

function value_and_prox!(Y::AbstractMatrix, f::LogDetLoss{T}, X::AbstractMatrix, γ::Real) where {T}
  _checksize(Y, X)
  size(X) == size(f.S) || throw(DimensionMismatch("X has size $(size(X)) and S $(size(f.S))"))
  root_step = sqrt(convert(T, γ)) * sqrt(f.w)
  A = f.eig.A
  logdet_Y = if root_step > one(T)
    # Half of X/(γw) - S; divide twice to retain the contribution of X even
    # when 1/(γw) underflows. No large step multiplies the covariance.
    A .= ((X ./ root_step) ./ root_step) ./ 2 .- f.S ./ 2
    t, U = _eigen!(f.eig)
    _rebuild_sym_out!(Y, U, t, s -> _logdet_scaled_root_half(s, root_step); logs = true, buffer = f.C)
  else
    # Half of X - γwS, without forming a possibly underflowed γw.
    A .= X ./ 2 .- ((f.S ./ 2 .* root_step) .* root_step)
    t, U = _eigen!(f.eig)
    _rebuild_sym_out!(Y, U, t, s -> _logbarrier_root_half(s, root_step); logs = true, buffer = f.C)
  end
  f.w * (dot(f.S, Y) - logdet_Y)
end

# The positive root of z² - 2half*z - root_step² = 0. Keeping the square root
# of the step avoids overflow and underflow of the step product itself.
@inline function _logbarrier_root_half(half::T, root_step::T) where {T}
  r = hypot(half, root_step)
  half >= 0 && return half + r
  denominator = r - half
  isinf(denominator) ? (root_step / 2) * (root_step / (r / 2 - half / 2)) :
                       root_step * (root_step / denominator)
end

# The same root when half = (X/(γw) - S)/2 and root_step = √(γw) > 1.
@inline function _logdet_scaled_root_half(half::T, root_step::T) where {T}
  r = hypot(half, inv(root_step))
  half >= 0 && return ((half + r) * root_step) * root_step
  denominator = r - half
  isinf(denominator) ? (one(T) / 2) / (r / 2 - half / 2) : inv(denominator)
end
prox!(Y::AbstractMatrix, f::LogDetLoss, X::AbstractMatrix, γ::Real) = (value_and_prox!(Y, f, X, γ); Y)

# the gradient S - X⁻¹ is not Lipschitz on the positive definite matrices
lipschitz(::LogDetLoss{T}) where {T} = T(Inf)
