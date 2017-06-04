
"""
  shrink(v, c)

Soft-threshold operator. Returns sign(v)⋅max(0, |v|-c)
"""
shrink{T<:AbstractFloat}(v::T, c::T) = v > c ? v - c : (v < -c ? v + c : zero(T))


"""
  out = max(1-c/|x|, 0) ⋅ x
"""
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

function A_mul_B_row{T<:AbstractFloat}(A::AbstractMatrix{T}, b::SparseIterate{T, 1}, row::Int64)
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds for i = 1:nnz(b)
      v += A[row, b.nzval2ind[i]] * b.nzval[i]
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

function At_mul_B_row{T<:AbstractFloat}(A::AbstractMatrix{T}, b::SparseIterate{T, 1}, row::Int64)
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds for i = 1:b.nnz
      v += A[b.nzval2ind[i], row] * b.nzval[i]
  end
  v
end


# function A_mul_X_mul_B_rc{T<:AbstractFloat}(
#   A::AbstractMatrix{T},
#   X::SparseVector{T},
#   B::AbstractMatrix{T},
#   r::Int,
#   c::Int
#   )
#
#   p = size(A, 1)
#
#   nzval = SparseArrays.nonzeros(X)
#   rowval = SparseArrays.nonzeroinds(X)
#
#   v = zero(T)
#   for j=1:length(nzval)
#     ri, ci = ind2subLowerTriangular(p, rowval[j])
#     if ri == ci
#       @inbounds v += A[ri, ar] * Σy[ci, ac] * nzval[j]
#     else
#       @inbounds v += (A[ri, ar] * Σy[ci, ac] + A[ci, ar] * Σy[ri, ac]) * nzval[j]
#     end
#   end
#   v
# end

function A_mul_X_mul_B_rc(
  A::Symmetric{T},
  X::SymmetricSparseIterate{T},
  B::Symmetric{T},
  r::Int,
  c::Int
  ) where {T<:AbstractFloat}
  # check input dimensions

  data = X.data
  p = size(A, 1)
  v = zero(T)
  for j=1:nnz(data)
    ri, ci = ind2sub(data, data.nzval2ind[j])
    if ri == ci
      @inbounds v += A[ri, r] * B[ci, c] * data.nzval[j]
    else
      @inbounds v += (A[ri, r] * B[ci, c] + A[ci, r] * B[ri, c]) * data.nzval[j]
    end
  end
  v
end

function A_mul_X_mul_B{T<:AbstractFloat}(
  A::Symmetric{T},
  X::SymmetricSparseIterate{T},
  B::Symmetric{T}
  )
  # check input dimensions

  p = size(A, 1)
  out = zeros(T, p, p)

  for c=1:p, r=1:p
    @inbounds out[r,c] = A_mul_X_mul_B_rc(A, X, B, r, c)
  end
  out
end


function A_mul_UUt_mul_B_rc{T<:AbstractFloat}(
  A::Symmetric{T},
  U::StridedMatrix{T},
  B::Symmetric{T},
  r::Int,
  c::Int
  )
  # check input dimensions

  nr, nc = size(U)
  v = zero(T)
  for j=1:nc
    v1 = zero(T)   # stores A[r, :]*U[:,j]
    v2 = zero(T)   # stores B[c, :]*U[:,j]
    for k=1:nr
      @inbounds v1 += A[k, r] * U[k, j]
      @inbounds v2 += B[k, c] * U[k, j]
    end
    v += v1 * v2
  end
  v
end

function A_mul_UUt_mul_B(A::Symmetric{T},
  U::StridedMatrix{T},
  B::Symmetric{T}
  ) where {T<:AbstractFloat}

  # check input dimensions

  p = size(A, 1)
  out = zeros(T, p, p)

  for c=1:p, r=1:p
    @inbounds out[r,c] = A_mul_UUt_mul_B_rc(A, U, B, r, c)
  end
  out
end
