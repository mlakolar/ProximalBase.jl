####################################
#
#  Least squares  ‖Y - Xβ‖² / 2
#
####################################

"""
    LeastSquaresLoss(Y, X)

The least-squares loss `f(β) = ‖Y - Xβ‖² / 2`, smooth with gradient
`X'(Xβ - Y)`. `Y` may be a vector or an `n × k` matrix with one column per
response; then `β` and the gradient are `p × k` matrices and `‖⋅‖` is the
Frobenius norm, as in multi-response regression with [`NuclearNorm`](@ref) or
the row groups of [`GroupNormL2`](@ref). `β` may be a dense array, a
[`SparseIterate`](@ref) or an [`AtomIterate`](@ref), whose products cost
O(n ⋅ nnz).

It is also proximable: the first proximal operator computes the thin SVD
`X = U Diagonal(s) V'` and `Diagonal(s) U'Y` once, after which
`prox_{γf}(v) = (X'X + I/γ)⁻¹ (X'Y + v/γ)` costs two products with `V` for any
`γ`, also when `p > n`. ADMM changes `γ` freely at no cost.
"""
mutable struct LeastSquaresLoss{T<:AbstractFloat, Ty<:AbstractVecOrMat, Tx<:AbstractMatrix, N, A<:AbstractArray{T, N}} <: AbstractFunction
  Y::Ty
  X::Tx
  tmp::A                  # Xβ - Y, next to Y; calls to f(β) do not allocate
  # the proximal operator: X = U Diagonal(√s2) V', computed on first use
  V::Matrix{T}
  s2::Vector{T}
  XtY::Array{T, N}        # only needed by an affine-constrained loss, built lazily
  sUtY::Array{T, N}       # Diagonal(s) U'Y, the right-hand side in the V basis
  w::Array{T, N}
  output::Scratch         # dense output for a non-strided destination
  svd_ready::Bool
end

function LeastSquaresLoss(Y::AbstractVecOrMat, X::AbstractMatrix)
  size(X, 1) == size(Y, 1) || throw(DimensionMismatch("X has $(size(X, 1)) rows and Y has $(size(Y, 1))"))
  T = float(promote_type(eltype(Y), eltype(X)))
  N = ndims(Y)
  tmp = _zeros_like(Y, T, size(Y))
  LeastSquaresLoss{T, typeof(Y), typeof(X), N, typeof(tmp)}(Y, X, tmp, Matrix{T}(undef, 0, 0), T[],
                                                            Array{T, N}(undef, ntuple(_ -> 0, N)),
                                                            Array{T, N}(undef, ntuple(_ -> 0, N)),
                                                            Array{T, N}(undef, ntuple(_ -> 0, N)), Scratch(), false)
end

issmooth(::Type{<:LeastSquaresLoss}) = true
isproximable(::Type{<:LeastSquaresLoss}) = true
isconvex(::Type{<:LeastSquaresLoss}) = true

function (f::LeastSquaresLoss{T})(β::AbstractVecOrMat) where {T}
  mul!(f.tmp, f.X, β)
  v = zero(T)
  @inbounds @simd for i in eachindex(f.tmp, f.Y)
    v += abs2(f.Y[i] - f.tmp[i])
  end
  v / 2
end

function value_and_gradient!(g::AbstractVecOrMat, f::LeastSquaresLoss, β::AbstractVecOrMat)
  _checksize(g, β)
  mul!(f.tmp, f.X, β)
  f.tmp .-= f.Y
  mul!(g, transpose(f.X), f.tmp)
  sum(abs2, f.tmp) / 2
end

function gradient!(g::AbstractVecOrMat, f::LeastSquaresLoss, β::AbstractVecOrMat)
  _checksize(g, β)
  mul!(f.tmp, f.X, β)
  f.tmp .-= f.Y
  mul!(g, transpose(f.X), f.tmp)
end

function _svd_cache!(f::LeastSquaresLoss{T, Ty, Tx, N}) where {T, Ty, Tx, N}
  if !f.svd_ready
    if isempty(f.X)
      # LAPACK's empty SVD may expose a square V even for a 0×p design.
      # Hessian consumers need the thin rank-zero factorization p×0.
      f.V = zeros(T, size(f.X, 2), 0)
      f.s2 = T[]
      dims = N == 1 ? (0,) : (0, size(f.Y, 2))
      f.sUtY = Array{T, N}(undef, dims)
      f.w = Array{T, N}(undef, dims)
      f.svd_ready = true
      return f
    end
    F = svd(Matrix{T}(f.X))
    f.V = F.V
    f.s2 = F.S .^ 2
    f.sUtY = Array{T, N}(transpose(F.U) * f.Y)
    f.sUtY .*= F.S
    f.w = N == 1 ? Array{T, N}(undef, length(f.s2)) : Array{T, N}(undef, length(f.s2), size(f.Y, 2))
    f.svd_ready = true
  end
  f
end

function _xty_cache!(f::LeastSquaresLoss{T, Ty, Tx, N}) where {T, Ty, Tx, N}
  dims = N == 1 ? (size(f.X, 2),) : (size(f.X, 2), size(f.Y, 2))
  if any(iszero, dims)
    # An empty result is also a ready cache; isempty cannot identify its
    # uninitialized state on later calls.
    size(f.XtY) == dims || (f.XtY = Array{T, N}(undef, dims))
  elseif isempty(f.XtY)
    f.XtY = Array{T, N}(transpose(f.X) * f.Y)
  end
  f.XtY
end

# In the V basis the solution is (s U'Y + ρ V'v)/(s² + ρ). When V is
# rectangular, keep v's null-space component with v + V(s U'Y - s² V'v)/(s² + ρ).
# Neither form subtracts terms of size γ X'Y at a large step γ.
function prox!(β::AbstractVecOrMat, f::LeastSquaresLoss{T}, v::AbstractVecOrMat, γ::Real) where {T}
  _checksize(β, v)
  size(β, 1) == size(f.X, 2) || throw(DimensionMismatch("β must have $(size(f.X, 2)) rows"))
  ndims(β) == ndims(f.Y) && size(β, 2) == size(f.Y, 2) ||
    throw(DimensionMismatch("β must have the same number of responses as Y"))
  # With no observations or coefficients the loss is constant; avoid an empty SVD.
  (isempty(f.X) || isempty(β)) && return β === v ? β : copyto!(β, v)
  β isa StridedVecOrMat || return copyto!(β, prox!(_array_buffer!(f.output, T, size(β)), f, v, γ))
  _svd_cache!(f)
  ρ = inv(convert(T, γ))
  mul!(f.w, transpose(f.V), v)
  if size(f.V, 1) == size(f.V, 2)
    if convert(T, γ) <= one(T)
      # At small steps, avoid overflow in ρ or ρ V'v by scaling both sides by γ.
      s = convert(T, γ)
      @inbounds for j in axes(f.w, 2), i in axes(f.w, 1)
        f.w[i, j] = (s * f.sUtY[i, j] + f.w[i, j]) / (s * f.s2[i] + one(T))
      end
    else
      @inbounds for j in axes(f.w, 2), i in axes(f.w, 1)
        f.w[i, j] = (f.sUtY[i, j] + ρ * f.w[i, j]) / (f.s2[i] + ρ)
      end
    end
    mul!(β, f.V, f.w)
  else
    @inbounds for j in axes(f.w, 2), i in axes(f.w, 1)
      f.w[i, j] = (f.sUtY[i, j] - f.s2[i] * f.w[i, j]) / (f.s2[i] + ρ)
    end
    β === v || copyto!(β, v)
    mul!(β, f.V, f.w, one(T), one(T))
  end
end

restrict(f::LeastSquaresLoss{T, <:AbstractVector}, S::AbstractVector{<:Integer}) where {T} =
  LeastSquaresLoss(f.Y, f.X[:, S])

lipschitz(f::LeastSquaresLoss) = _opnorm2_upper(f.X)
