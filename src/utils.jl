
"""
    shrink(v, c)

Soft-thresholding: return `sign(v) ⋅ max(|v| - c, 0)`.
"""
shrink(v::T, c::T) where {T<:AbstractFloat} = v > c ? v - c : (v < -c ? v + c : zero(T))


"""
    shrinkL2!(out, x, c)

Group soft-thresholding: store `max(1 - c / ‖x‖₂, 0) ⋅ x` in `out` and return it.
"""
function shrinkL2!(out::M, x::M, c::T) where {M <: AbstractVecOrMat{T}} where T
  tmp = max(one(T) - c / norm(x), zero(T))
  tmp > zero(T) ? rmul!(copyto!(out, x), tmp) : fill!(out, zero(T))
end


"""
    proxL1Fused(x, y, λ1, λ2, γ = 1)

Proximal operator of `λ1 (|u| + |v|) + λ2 |u - v|` at the scalar pair `(x, y)`:
return `(u, v) = argmin (1/2γ) ((u - x)² + (v - y)²) + λ1 (|u| + |v|) + λ2 |u - v|`.
"""
function proxL1Fused(x::T, y::T, λ1::T, λ2::T, γ::T=one(T)) where {T<:AbstractFloat}

  if λ2 > zero(T)
    t = λ2 * γ
    if x > y + 2 * t
      u = x - t
      v = y + t
    elseif y > x + 2 * t
      u = x + t
      v = y - t
    else
      u = v = (x + y) / 2
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

"""
    A_mul_B_row(A, b, row)

Compute `(A * b)[row]` without forming `A * b`. `b` may be dense, a
`SparseVector`, or a [`SparseIterate`](@ref).
"""
function A_mul_B_row(A::AbstractMatrix{T}, b::AbstractVector{T}, row::Int64) where {T<:AbstractFloat}
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds @simd for i=1:p
    v += A[row, i] * b[i]
  end
  v
end

function A_mul_B_row(A::AbstractMatrix{T}, b::SparseVector{T}, row::Int64) where {T<:AbstractFloat}
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())


  nzval = SparseArrays.nonzeros(b)
  rowval = SparseArrays.nonzeroinds(b)
  v = zero(T)
  @inbounds @simd for i=1:length(nzval)
    v += A[row, rowval[i]] * nzval[i]
  end
  v
end

function A_mul_B_row(A::AbstractMatrix{T}, b::SparseIterate{T, 1}, row::Int64) where {T<:AbstractFloat}
  n, p = size(A)
  ((p == length(b)) && (1 <= row <= n)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds @simd for i = 1:nnz(b)
      v += A[row, b.nzval2ind[i]] * b.nzval[i]
  end
  v
end


"""
    At_mul_B_row(A, b, row)

Compute `(A' * b)[row]` without forming `A' * b`. `b` may be dense, a
`SparseVector`, or a [`SparseIterate`](@ref).
"""
function At_mul_B_row(A::AbstractMatrix{T}, b::StridedVector{T}, row::Int64) where {T<:AbstractFloat}
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= p)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds @simd for i=1:n
    v += A[i, row] * b[i]
  end
  v
end

function At_mul_B_row(A::AbstractMatrix{T}, b::SparseVector{T}, row::Int64) where {T<:AbstractFloat}
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= p)) || throw(DimensionMismatch())

  nzval = SparseArrays.nonzeros(b)
  rowval = SparseArrays.nonzeroinds(b)
  v = zero(T)
  @inbounds @simd for i=1:length(nzval)
    v += A[rowval[i], row] * nzval[i]
  end
  v
end

function At_mul_B_row(A::AbstractMatrix{T}, b::SparseIterate{T, 1}, row::Int64) where {T<:AbstractFloat}
  n, p = size(A)
  ((n == length(b)) && (1 <= row <= p)) || throw(DimensionMismatch())

  v = zero(T)
  @inbounds @simd for i = 1:b.nnz
      v += A[b.nzval2ind[i], row] * b.nzval[i]
  end
  v
end

"""
    A_mul_X_mul_B_rc(A, X, B, r, c)

Compute `(A * X * B)[r, c]` for symmetric `A`, `B` and a sparse `X`, in O(nnz(X)).
`X` is a [`SymmetricSparseIterate`](@ref), or a [`SparseIterate`](@ref) vector
holding `vec` of a `size(A, 2) × size(B, 1)` matrix.
"""
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

"""
    A_mul_X_mul_B(A, X, B)

Compute `A * X * B` for symmetric `A`, `B` and a sparse `X`; see
[`A_mul_X_mul_B_rc`](@ref).
"""
function A_mul_X_mul_B(A::Symmetric{T}, X::SymmetricSparseIterate{T}, B::Symmetric{T}) where {T<:AbstractFloat}
  # dense BLAS products are much faster than filling in the result entry by entry
  # with A_mul_X_mul_B_rc, which costs O(p² nnz(X))
  A * (Matrix(X) * B)
end

A_mul_X_mul_B(A::Symmetric{T}, X::SparseIterate{T, 1}, B::Symmetric{T}) where {T<:AbstractFloat} =
  A * (reshape(Vector(X), size(A, 2), size(B, 1)) * B)




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

A_mul_UVt_mul_B(A::Symmetric{T}, U::StridedMatrix{T}, V::StridedMatrix{T}, B::Symmetric{T}) where {T<:AbstractFloat} =
  (A * U) * transpose(B * V)


"""
    A_mul_UUt_mul_B_rc(A, U, B, r, c)

Compute `(A * U * U' * B)[r, c]` for symmetric `A`, `B` without forming `U * U'`.
"""
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

"""
    A_mul_UUt_mul_B(A, U, B)

Compute `A * U * U' * B` for symmetric `A`, `B` without forming `U * U'`.
"""
A_mul_UUt_mul_B(A::Symmetric{T}, U::StridedMatrix{T}, B::Symmetric{T}) where {T<:AbstractFloat} =
  (A * U) * transpose(B * U)


#####
#
# functions to operate with sparse lower triangular
#

"""
    ind2subLowerTriangular(p, ind)

Convert a linear index into the lower triangle of a `p × p` matrix (numbered
column by column) to `(row, col)` with `row ≥ col`. Inverse of
[`sub2indLowerTriangular`](@ref).
"""
function ind2subLowerTriangular(p::T, ind::T) where {T<:Integer}
  rvLinear = div(p*(p+1), 2) - ind
  k = trunc(T, (sqrt(1+8*rvLinear)-1.)/2. )
  j = rvLinear - div(k*(k+1), 2)
  (p-j, p-k)
end

"""
    sub2indLowerTriangular(p, r, c)

Linear index of entry `(r, c)`, `r ≥ c`, in the lower triangle of a `p × p`
matrix, numbered column by column. Inverse of [`ind2subLowerTriangular`](@ref).
"""
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
    I[i], J[i] = ind2subLowerTriangular(p, x.nzval2ind[i])
  end

  sparse(I,J,x.nzval[1:nx], p, p)
end


function tril2symmetric(Δ::SparseMatrixCSC)
  lDelta = tril(Δ, -1)
  dDelta = spdiagm(diag(Δ))
  (lDelta + lDelta') + dDelta
end


#######

"""
    norm_diff(A, B, p = 2)

Compute `‖A - B‖` without allocating `A - B`: the Euclidean (Frobenius) norm for
`p = 2`, the largest absolute entry for `p = Inf`.
"""
function norm_diff(A::AbstractArray{T}, B::AbstractArray{T}, p::Real=2.) where {T<:AbstractFloat}
  size(A) == size(B) || throw(DimensionMismatch())

  v = zero(T)
  if p == 2.
    @inbounds @simd for i in eachindex(A)
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
