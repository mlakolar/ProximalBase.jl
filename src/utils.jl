
"""
  shrink(v, c)

Soft-threshold operator. Returns sign(v)⋅max(0, |v|-c)
"""
shrink{T<:AbstractFloat}(v::T, c::T) = v > c ? v - c : (v < -c ? v + c : zero(T))

function shrinkL2!(out::M, x::M, c::T) where {M <: AbstractVecOrMat{T}} where T
  tmp = max(one(T) - c / vecnorm(x), zero(T))
  tmp > zero(T) ? scale!(copy!(out, x), tmp) : fill!(out, zero(T))
end


"""
Computes a proximal operation for the penalty
\[ \lambda_1\cdot(|u|+|v|) + \lambda_2\cdot|u-v| . \]

Returns (u, v) = arg min (1/2γ) * ((u - x)^2 + (v-y)^2) + λ1 * (|u|+|v|) + λ2 * |u-v|
"""
function proxL1Fused{T <: AbstractFloat}(x::T, y::T, λ1::T, λ2::T, γ::T=one(T))

  if λ2 > zero(T)
    t = λ2 * γ
    if x > y + 2. * t
      u = x - t
      v = y + t
    elseif y > x + 2. * t
      u = x + t
      v = y - t
    else
      u = v = (x + y) / 2.
    end
  else
    u = x
    v = y
  end

  t = λ1 * γ
  (shrink(u, t), shrink(v, t))
end



####################################
#
# helper functions
#
####################################

function A_mul_B_row{T<:AbstractFloat}(A::AbstractMatrix{T}, b::AbstractVector{T}, row::Int64)
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  for i=1:p
    @inbounds v += A[row, i] * b[i]
  end
  v
end

function A_mul_B_row{T<:AbstractFloat}(A::AbstractMatrix{T}, b::SparseVector{T}, row::Int64)
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())


  nzval = SparseArrays.nonzeros(b)
  rowval = SparseArrays.nonzeroinds(b)
  v = zero(T)
  for i=1:length(nzval)
    @inbounds v += A[row, rowval[i]] * nzval[i]
  end
  v
end

function A_mul_B_row{T<:AbstractFloat}(A::AbstractMatrix{T}, b::SparseIterate{T}, row::Int64)
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds for icoef = 1:nnz(b)
      v += A[row, b.nzval2full[icoef]] * b.nzval[icoef]
  end
  v
end


function At_mul_B_row{T<:AbstractFloat}(A::AbstractMatrix{T}, b::StridedVector{T}, row::Int64)
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= p)) || throw(DimensionMismatch())

  v = zero(T)
  for i=1:n
    @inbounds v += A[i, row] * b[i]
  end
  v
end

function At_mul_B_row{T<:AbstractFloat}(A::AbstractMatrix{T}, b::SparseVector{T}, row::Int64)
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= p)) || throw(DimensionMismatch())

  nzval = SparseArrays.nonzeros(b)
  rowval = SparseArrays.nonzeroinds(b)
  v = zero(T)
  for i=1:length(nzval)
    @inbounds v += A[rowval[i], row] * nzval[i]
  end
  v
end

function At_mul_B_row{T<:AbstractFloat}(A::AbstractMatrix{T}, b::SparseIterate{T}, row::Int64)
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds for icoef = 1:b.nnz
      v += A[b.nzval2full[icoef], row] * b.nzval[icoef]
  end
  v
end
