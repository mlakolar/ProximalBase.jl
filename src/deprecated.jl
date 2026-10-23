####################################
#
#  The interface of ProximalBase 0.4, deprecated in 0.5
#
####################################

# Old code runs with a deprecation warning (shown with --depwarn=yes, as in the
# tests of a package); NEWS.md lists what to change. Removed in 0.6.

# functions are callable, and mutating functions take their output first
@deprecate value(f::AbstractFunction, x) f(x)
@deprecate value(f::NormL1Fused, x1::Real, x2::Real) f((x1, x2))
@deprecate prox!(f::AbstractFunction, y::AbstractArray, x::AbstractArray, γ::Real) prox!(y, f, x, γ)
@deprecate prox!(f::AbstractFunction, y::AbstractArray, x::AbstractArray) prox!(y, f, x)
@deprecate prox!(f::AbstractFunction, y::Tuple, x::Tuple, γ::Real) prox!(y, f, x, γ)
@deprecate cdprox!(f::AbstractFunction, x::AbstractArray, k::Integer, γ::Real) prox_coordinate!(x, f, k, γ)
@deprecate gradient!(f::AbstractFunction, g::AbstractArray, x::AbstractArray) gradient!(g, f, x)
@deprecate value_and_gradient!(f::AbstractFunction, g::AbstractArray, x::AbstractArray) value_and_gradient!(g, f, x)

# types: one abstract type with traits, and functions named for what they are
Base.@deprecate_binding DifferentiableFunction AbstractFunction
Base.@deprecate_binding ProximableFunction AbstractFunction
Base.@deprecate_binding ProxZero ZeroFunction
Base.@deprecate_binding ProxL1 NormL1
Base.@deprecate_binding ProxL2 NormL2
Base.@deprecate_binding ProxGroupL2 GroupNormL2
Base.@deprecate_binding ProxNuclear NuclearNorm
Base.@deprecate_binding ProxL1Fused NormL1Fused
Base.@deprecate_binding ProxGaussLikelihood LogDetLoss

function ProxL2Sq(λ::Real)
  Base.depwarn("`ProxL2Sq(λ)`, the penalty λ‖x‖², is deprecated: use `SqrNormL2(2λ)`, as SqrNormL2(μ) is (μ/2)‖x‖².", :ProxL2Sq)
  SqrNormL2(2λ)
end
function L2Loss(y::AbstractArray)
  Base.depwarn("`L2Loss(y)` is deprecated, use `SqrNormL2(1, y)`.", :L2Loss)
  SqrNormL2(1, y)
end
function L2Loss()
  Base.depwarn("`L2Loss()` is deprecated, use `SqrNormL2()`.", :L2Loss)
  SqrNormL2()
end

# the weights of NormL1 and GroupNormL2 were the field λ
function Base.getproperty(f::Union{NormL1, GroupNormL2}, s::Symbol)
  if s === :λ
    Base.depwarn("the weights of $(nameof(typeof(f))) are the field `w`; `λ` is deprecated.", :getproperty)
    return getfield(f, :w)
  end
  getfield(f, s)
end

# the coordinate API of the iterates, in Julia's lowercase
@deprecate numCoordinates(x) ncoordinates(x)
@deprecate numActive(x) nactive(x)
@deprecate activeCoordinate(x, i) activecoordinates(x)[i]
@deprecate isActive(x, k) isactive(x, k)

# utilities
@deprecate ind2subLowerTriangular(p, ind) tril_cartesian(p, ind)
@deprecate sub2indLowerTriangular(p, r, c) tril_index(p, r, c)
@deprecate proxL1Fused(x::Real, y::Real, λ1::Real, λ2::Real, γ::Real) prox(NormL1Fused(λ1, λ2), (x, y), γ)
@deprecate proxL1Fused(x::Real, y::Real, λ1::Real, λ2::Real) prox(NormL1Fused(λ1, λ2), (x, y))
@deprecate shrinkL2!(out, x, c) prox!(out, NormL2(c), x)

# products: dot and mul! on views, and the three-argument * and dot
@deprecate A_mul_B_row(A, b, row) dot(view(A, row, :), b)
@deprecate At_mul_B_row(A, b, row) dot(view(A, :, row), b)
@deprecate At_mul_B_group!(out, A, b, G) mul!(out, transpose(view(A, :, G)), b)
@deprecate A_mul_B_group!(out, A, v, G) mul!(out, view(A, :, G), v)
@deprecate A_mul_B_group!(out, A, v, G, α, β) mul!(out, view(A, :, G), v, α, β)
@deprecate A_mul_X_mul_B(A::Symmetric, X::SymmetricSparseIterate, B::Symmetric) A * X * B
@deprecate A_mul_X_mul_B_rc(A::Symmetric, X::SymmetricSparseIterate, B::Symmetric, r::Integer, c::Integer) dot(view(A, :, r), X, view(B, :, c))

# a SparseIterate vector holding vec(X): use a SparseIterate matrix
function _vec_to_matrix(X::SparseIterate{T, 1}, m::Integer, n::Integer) where {T}
  length(X) == m * n || throw(DimensionMismatch())
  M = SparseIterate(T, m, n)
  for k in 1:nnz(X)
    M[X.nzval2ind[k]] = X.nzval[k]
  end
  M
end
function A_mul_X_mul_B(A::Symmetric, X::SparseIterate{T, 1}, B::Symmetric) where {T}
  Base.depwarn("`A_mul_X_mul_B` is deprecated: use `A * X * B` with `X` a SparseIterate matrix.", :A_mul_X_mul_B)
  A * _vec_to_matrix(X, size(A, 2), size(B, 1)) * B
end
function A_mul_X_mul_B_rc(A::Symmetric, X::SparseIterate{T, 1}, B::Symmetric, r::Integer, c::Integer) where {T}
  Base.depwarn("`A_mul_X_mul_B_rc` is deprecated: use `dot(view(A, :, r), X, view(B, :, c))` with `X` a SparseIterate matrix.", :A_mul_X_mul_B_rc)
  m = size(A, 2)
  v = zero(promote_type(eltype(A), T, eltype(B)))
  for k in 1:nnz(X)
    i = X.nzval2ind[k]
    v += A[mod1(i, m), r] * B[cld(i, m), c] * X.nzval[k]
  end
  v
end
function A_mul_UUt_mul_B(A::Symmetric, U::StridedMatrix, B::Symmetric)
  Base.depwarn("`A_mul_UUt_mul_B(A, U, B)` is deprecated, use `(A * U) * transpose(B * U)`.", :A_mul_UUt_mul_B)
  (A * U) * transpose(B * U)
end
function A_mul_UUt_mul_B_rc(A::Symmetric, U::StridedMatrix, B::Symmetric, r::Integer, c::Integer)
  Base.depwarn("`A_mul_UUt_mul_B_rc(A, U, B, r, c)` is deprecated, use `dot(U' * view(A, :, r), U' * view(B, :, c))`.", :A_mul_UUt_mul_B_rc)
  v = zero(promote_type(eltype(A), eltype(U), eltype(B)))
  for j in axes(U, 2)
    v1 = zero(v)
    v2 = zero(v)
    for k in axes(U, 1)
      v1 += A[k, r] * U[k, j]
      v2 += B[k, c] * U[k, j]
    end
    v += v1 * v2
  end
  v
end

export ProxL2Sq, L2Loss, A_mul_X_mul_B, A_mul_X_mul_B_rc, A_mul_UUt_mul_B, A_mul_UUt_mul_B_rc
