####################################
#
#  Affine sets {x : Ax = b}
#
####################################

"""
    IndAffine(A, b)
    IndAffine(A, b; μ)

The indicator of the affine set `{x : Ax = b}` for an `m × n` matrix `A`
(dense, sparse, or anything with `mul!`) and a vector `b` of length `m`. Its
proximal operator is the projection

    x - Aᵀ(AAᵀ)⁻¹(Ax - b),

from a factorization cached at construction:

- **full row rank:** a Cholesky factorization of the `m × m` matrix `AAᵀ`;
- **otherwise:** a truncated SVD `A ≈ Uᵣ Diagonal(sᵣ) Vᵣᵀ`, retaining the `r`
  singular values strictly above `max(m, n) * eps(T) * s₁`, with `T` the
  floating-point data type and `s₁` the largest singular value. The projection
  is `x - Vᵣ(Vᵣᵀx - Diagonal(sᵣ)⁻¹Uᵣᵀb)`. The constructor throws unless
  `‖b - UᵣUᵣᵀb‖ ≤ √eps(T) * max(1, ‖b‖)`;
- **with `μ`:** `A` is a tight frame, `AAᵀ = μI`, and nothing is factorized.

The SVD path is taken when Cholesky fails or a diagonal entry of its factor is
below `eps(T)^(1/4)` times the largest: rounding leaves a dependent row a pivot of
about `√eps` times the largest, and the factor of `AAᵀ` squares the condition
number of `A`. Smaller singular directions are dropped, so a nearly
rank-deficient `A` is treated as its truncation, with `b` projected onto its
retained range. The projection can then be far from `{x : Ax = b}` while the
value is `0`. A call costs two products with `A` (or with `Vᵣ`)
and allocates nothing. The value is `0` when
`‖Ax - b‖ ≤ √eps(S) * max(1, ‖b‖, ‖A‖_F * ‖x‖)`, with `S` the promoted
floating-point type of the data and `x`, and `Inf` otherwise.
"""
struct IndAffine{T<:AbstractFloat, M, F} <: AbstractFunction
  A::M
  b::Vector{T}
  fact::F               # a Cholesky factorization of AAᵀ, (V, c) of a thin SVD, or μ
  r::Vector{T}          # Ax - b for the projection, or Vᵀx - c on the rank-deficient path
  rb::Vector{T}         # Ax - b for the value
  scale::T              # ‖A‖_F, for the tolerance of the value
end

function IndAffine(A::AbstractMatrix, b::AbstractVector; μ::Union{Nothing, Real} = nothing)
  m, n = size(A)
  length(b) == m || throw(DimensionMismatch("A has $m rows and b has length $(length(b))"))
  T = float(promote_type(eltype(A), eltype(b)))
  bt = Vector{T}(b)
  scale = convert(T, norm(A))
  if μ !== nothing
    μ > 0 || throw(ArgumentError("μ must be positive, got $μ"))
    return IndAffine{T, typeof(A), T}(A, bt, convert(T, μ), zeros(T, m), zeros(T, m), scale)
  end
  if m == 0
    # A vacuous system has rank zero and projects every vector to itself.
    return IndAffine{T, typeof(A), Tuple{Matrix{T}, Vector{T}}}(
      A, bt, (zeros(T, n, 0), T[]), T[], T[], scale)
  end
  G = Matrix{T}(A * A')
  C = cholesky(Symmetric(G); check = false)
  if issuccess(C)
    d = diag(C.U)
    if minimum(d) >= sqrt(sqrt(eps(T))) * maximum(d)
      return IndAffine{T, typeof(A), typeof(C)}(A, bt, C, zeros(T, m), zeros(T, m), scale)
    end
  end
  # rank deficient: the thin SVD of the rank of A
  F = svd(Matrix{T}(A))
  tol = max(m, n) * eps(T) * (isempty(F.S) ? zero(T) : F.S[1])
  k = count(>(tol), F.S)
  V = F.V[:, 1:k]
  c = (F.U[:, 1:k]' * bt) ./ F.S[1:k]
  # b must be in the range of A: its part outside the range, b - UUᵀb, vanishes
  res = bt - F.U[:, 1:k] * (F.U[:, 1:k]' * bt)
  norm(res) <= sqrt(eps(T)) * max(one(T), norm(bt)) ||
    throw(ArgumentError("b is not in the range of A, so {x : Ax = b} is empty"))
  IndAffine{T, typeof(A), Tuple{Matrix{T}, Vector{T}}}(A, bt, (Matrix{T}(V), c), zeros(T, k), zeros(T, m), scale)
end

isproximable(::Type{<:IndAffine}) = true
_isindicator(::Type{<:IndAffine}) = true
isconvex(::Type{<:IndAffine}) = true

_checkaffine(f::IndAffine, x::AbstractArray) = length(x) == size(f.A, 2) ||
  throw(DimensionMismatch("x has length $(length(x)) and A has $(size(f.A, 2)) columns"))

function (f::IndAffine{T})(x::AbstractVector) where {T}
  _checkaffine(f, x)
  S = float(promote_type(T, eltype(x)))
  r = mul!(f.rb, f.A, x)
  r .-= f.b
  _ind(norm(r) <= _tol(S) * max(one(S), norm(f.b), f.scale * norm(x)), S)
end

function prox!(y::AbstractVector, f::IndAffine{T, M, <:Cholesky}, x::AbstractVector, ::Real) where {T<:AbstractFloat, M}
  _checksize(y, x)
  _checkaffine(f, x)
  r = mul!(f.r, f.A, x)
  r .-= f.b
  ldiv!(f.fact, r)
  y === x || copyto!(y, x)
  mul!(y, f.A', r, -one(T), one(T))
end

function prox!(y::AbstractVector, f::IndAffine{T, M, T}, x::AbstractVector, ::Real) where {T<:AbstractFloat, M}
  _checksize(y, x)
  _checkaffine(f, x)
  r = mul!(f.r, f.A, x)
  r .-= f.b
  y === x || copyto!(y, x)
  mul!(y, f.A', r, -inv(f.fact), one(T))
end

function prox!(y::AbstractVector, f::IndAffine{T, M, Tuple{Matrix{T}, Vector{T}}}, x::AbstractVector, ::Real) where {T<:AbstractFloat, M}
  _checksize(y, x)
  _checkaffine(f, x)
  V, c = f.fact
  t = mul!(f.r, V', x)
  t .-= c
  y === x || copyto!(y, x)
  mul!(y, V, t, -one(T), one(T))
end
