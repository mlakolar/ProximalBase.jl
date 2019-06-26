
"""
  shrink(v, c)

Soft-threshold operator. Returns sign(v)⋅max(0, |v|-c)
"""
shrink(v::T, c::T) where {T<:AbstractFloat} = v > c ? v - c : (v < -c ? v + c : zero(T))


"""
  out = max(1-c/|x|, 0) ⋅ x
"""
function shrinkL2!(out::M, x::M, c::T) where {M <: AbstractVecOrMat{T}} where T
  tmp = max(one(T) - c / norm(x), zero(T))
  tmp > zero(T) ? rmul!(copyto!(out, x), tmp) : fill!(out, zero(T))
end


# """
# Computes a proximal operation for the penalty
# \[ \lambda_1\cdot(|u|+|v|) + \lambda_2\cdot|u-v| . \]
#
# Returns (u, v) = arg min (1/2γ) * ((u - x)^2 + (v-y)^2) + λ1 * (|u|+|v|) + λ2 * |u-v|
# """
function proxL1Fused(x::T, y::T, λ1::T, λ2::T, γ::T=one(T)) where {T<:AbstractFloat}

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

function A_mul_B_row(A::AbstractMatrix{T}, b::AbstractVector{T}, row::Int64) where {T<:AbstractFloat}
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  for i=1:p
    @inbounds v += A[row, i] * b[i]
  end
  v
end

function A_mul_B_row(A::AbstractMatrix{T}, b::SparseVector{T}, row::Int64) where {T<:AbstractFloat}
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

function A_mul_B_row(A::AbstractMatrix{T}, b::SparseIterate{T, 1}, row::Int64) where {T<:AbstractFloat}
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds for i = 1:nnz(b)
      v += A[row, b.nzval2ind[i]] * b.nzval[i]
  end
  v
end


function At_mul_B_row(A::AbstractMatrix{T}, b::StridedVector{T}, row::Int64) where {T<:AbstractFloat}
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= p)) || throw(DimensionMismatch())

  v = zero(T)
  for i=1:n
    @inbounds v += A[i, row] * b[i]
  end
  v
end

function At_mul_B_row(A::AbstractMatrix{T}, b::SparseVector{T}, row::Int64) where {T<:AbstractFloat}
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

function At_mul_B_row(A::AbstractMatrix{T}, b::SparseIterate{T, 1}, row::Int64) where {T<:AbstractFloat}
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds for i = 1:b.nnz
      v += A[b.nzval2ind[i], row] * b.nzval[i]
  end
  v
end

function A_mul_X_mul_B_rc(
  A::Symmetric{T},
  X::SymmetricSparseIterate{T},
  B::Symmetric{T},
  r::Int,
  c::Int
  ) where {T<:AbstractFloat}

  p = X.p
  v = zero(T)
  for j=1:nnz(X)
    ind = X.nzval2ind[j]
    ri, ci = ind2subLowerTriangular(p, ind)
    if ri == ci
      @inbounds v += A[ri, r] * B[ci, c] * X.nzval[j]
    else
      @inbounds v += (A[ri, r] * B[ci, c] + A[ci, r] * B[ri, c]) * X.nzval[j]
    end
  end
  v
end

function A_mul_X_mul_B_rc(
  A::Symmetric{T},
  X::SparseIterate{T, 1},
  B::Symmetric{T},
  r::Int,
  c::Int
  ) where {T<:AbstractFloat}

  v = zero(T)
  I = CartesianIndices((size(A, 2), size(B, 1)))
  for j=1:nnz(X)
    ind = X.nzval2ind[j]
    @inbounds v += A[I[ind][1], r] * B[I[ind][2], c] * X.nzval[j]
  end
  v
end

function A_mul_X_mul_B(
  A::Symmetric{T},
  X::Union{SymmetricSparseIterate{T},SparseIterate{T}},
  B::Symmetric{T}
  ) where {T<:AbstractFloat}

  p = size(A, 1)
  out = zeros(T, p, p)

  for c=1:p, r=1:p
    @inbounds out[r,c] = A_mul_X_mul_B_rc(A, X, B, r, c)
  end
  out
end




function A_mul_UVt_mul_B_rc(
  A::Symmetric{T},
  U::StridedMatrix{T},
  V::StridedMatrix{T},
  B::Symmetric{T},
  r::Int,
  c::Int
  ) where {T<:AbstractFloat}
  # check input dimensions

  nr, nc = size(U)
  v = zero(T)
  for j=1:nc
    v1 = zero(T)   # stores A[r, :]*U[:,j]
    v2 = zero(T)   # stores B[c, :]*U[:,j]
    for k=1:nr
      @inbounds v1 += A[k, r] * U[k, j]
      @inbounds v2 += B[k, c] * V[k, j]
    end
    v += v1 * v2
  end
  v
end

function A_mul_UVt_mul_B(
    A::Symmetric{T},
    U::StridedMatrix{T},
    V::StridedMatrix{T},
    B::Symmetric{T}
  ) where {T<:AbstractFloat}

  # check input dimensions

  p = size(A, 1)
  out = zeros(T, p, p)

  for c=1:p, r=1:p
    @inbounds out[r,c] = A_mul_UVt_mul_B_rc(A, U, V, B, r, c)
  end
  out
end


function A_mul_UUt_mul_B_rc(
  A::Symmetric{T},
  U::StridedMatrix{T},
  B::Symmetric{T},
  r::Int,
  c::Int
  ) where {T<:AbstractFloat}
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


#####
#
# functions to operate with sparse lower triangular
#

function ind2subLowerTriangular(p::T, ind::T) where {T<:Integer}
  rvLinear = div(p*(p+1), 2) - ind
  k = trunc(T, (sqrt(1+8*rvLinear)-1.)/2. )
  j = rvLinear - div(k*(k+1), 2)
  (p-j, p-k)
end

sub2indLowerTriangular(p::T, r::T, c::T) where {T<:Integer} = p*(c-1)-div(c*(c-1),2)+r

function vec2tril(x::SparseVector, p::Int64)
  nx = nnz(x)
  nzval = SparseArrays.nonzeros(x)
  nzind = SparseArrays.nonzeroinds(x)

  I = zeros(Int64, nx)
  J = zeros(Int64, nx)
  for i=1:nx
    I[i], J[i] = ind2subLowerTriangular(p, nzind[i])
  end

  sparse(I,J,nzval, p, p)
end

function vec2tril(x::SparseIterate, p::Int64)
  nx = nnz(x)

  I = zeros(Int64, nx)
  J = zeros(Int64, nx)
  for i=1:nx
    I[i], J[i] = ind2subLowerTriangular(p, x.nzval2full[i])
  end

  sparse(I,J,x.nzval[1:nx], p, p)
end


function tril2symmetric(Δ::SparseMatrixCSC)
  lDelta = tril(Δ, -1)
  dDelta = spdiagm(diag(Δ))
  (lDelta + lDelta') + dDelta
end


#######

function norm_diff(A::AbstractArray{T}, B::AbstractArray{T}, p::Real=2.) where {T<:AbstractFloat}
  size(A) == size(B) || throw(DimensionMismatch())

  v = zero(T)
  if p == 2.
    @inbounds @simd for i in eachindex(A)
      t =
      v += abs2( A[i] - B[i] )
    end
    return sqrt(v)
 elseif p == Inf
   @inbounds @simd for i in eachindex(A)
     t = abs( A[i] - B[i] )
     if t > v
       v = t
     end
   end
   return v
 else
   throw(ArgumentError("p should be 2 or Inf"))
 end
end
