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
`X = U Diagonal(s) V'` and `X'Y` once, after which
`prox_{γf}(v) = (X'X + I/γ)⁻¹ (X'Y + v/γ)` costs two products with `V` for any
`γ`, also when `p > n`. ADMM changes `γ` freely at no cost.
"""
mutable struct LeastSquaresLoss{T<:AbstractFloat, Ty<:AbstractVecOrMat, Tx<:AbstractMatrix, N, A<:AbstractArray{T, N}} <: AbstractFunction
  Y::Ty
  X::Tx
  tmp::A                  # Xβ - Y, next to Y; calls to f(β) do not allocate
  # the proximal operator: X = U Diagonal(√s2) V' and X'Y, computed on first use
  V::Matrix{T}
  s2::Vector{T}
  XtY::Array{T, N}
  w::Array{T, N}
end

function LeastSquaresLoss(Y::AbstractVecOrMat, X::AbstractMatrix)
  size(X, 1) == size(Y, 1) || throw(DimensionMismatch("X has $(size(X, 1)) rows and Y has $(size(Y, 1))"))
  T = float(promote_type(eltype(Y), eltype(X)))
  N = ndims(Y)
  tmp = _zeros_like(Y, T, size(Y))
  LeastSquaresLoss{T, typeof(Y), typeof(X), N, typeof(tmp)}(Y, X, tmp, Matrix{T}(undef, 0, 0), T[],
                                                            Array{T, N}(undef, ntuple(_ -> 0, N)),
                                                            Array{T, N}(undef, ntuple(_ -> 0, N)))
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
  if isempty(f.s2)
    F = svd(Matrix{T}(f.X))
    f.V = F.V
    f.s2 = F.S .^ 2
    f.XtY = Array{T, N}(transpose(f.X) * f.Y)
    f.w = N == 1 ? Array{T, N}(undef, length(f.s2)) : Array{T, N}(undef, length(f.s2), size(f.Y, 2))
  end
  f
end

# (X'X + ρI)⁻¹ z = z/ρ + V Diagonal(1/(s² + ρ) - 1/ρ) V'z, with z = X'Y + ρv
function prox!(β::AbstractVecOrMat, f::LeastSquaresLoss{T}, v::AbstractVecOrMat, γ::Real) where {T}
  _checksize(β, v)
  size(β, 1) == size(f.X, 2) || throw(DimensionMismatch("β must have $(size(f.X, 2)) rows"))
  β isa StridedVecOrMat || return copyto!(β, prox!(Array{T}(undef, size(β)), f, v, γ))
  _svd_cache!(f)
  ρ = inv(convert(T, γ))
  β .= f.XtY .+ ρ .* v
  mul!(f.w, transpose(f.V), β)
  @inbounds for j in axes(f.w, 2), i in axes(f.w, 1)
    s2 = f.s2[i]
    f.w[i, j] *= -s2 / (ρ * (s2 + ρ))
  end
  β ./= ρ
  mul!(β, f.V, f.w, one(T), one(T))
end

restrict(f::LeastSquaresLoss{T, <:AbstractVector}, S::AbstractVector{<:Integer}) where {T} =
  LeastSquaresLoss(f.Y, f.X[:, S])

lipschitz(f::LeastSquaresLoss) = _opnorm2(f.X)
