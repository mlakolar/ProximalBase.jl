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

  function IndAffine{T, M, F}(A, b, fact, r, rb, scale) where {T<:AbstractFloat, M, F}
    all(a -> a === nothing || eltype(a) <: Real,
        (A, b, r, rb)) || _real_error(:IndAffine)
    (fact === nothing || (fact isa Tuple ? all(a -> eltype(a) <: Real, fact) : eltype(fact) <: Real)) ||
      _real_error(:IndAffine)
    new{T, M, F}(A, b, fact, r, rb, scale)
  end
end
IndAffine(A::M, b::Vector{T}, fact::F, r::Vector{T}, rb::Vector{T}, scale::T) where {T<:AbstractFloat, M, F} =
  IndAffine{T, M, F}(A, b, fact, r, rb, scale)

function IndAffine(A::AbstractMatrix, b::AbstractVector; μ::Union{Nothing, Real} = nothing)
  eltype(A) <: Real || _real_error(:IndAffine)
  eltype(b) <: Real || _real_error(:IndAffine)
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
_argument_type(::Type{<:IndAffine}) = AbstractVector
_isindicator(::Type{<:IndAffine}) = true
isconvex(::Type{<:IndAffine}) = true

_checkaffine(f::IndAffine, x::AbstractArray) = (length(x) == size(f.A, 2) ||
  throw(DimensionMismatch("x has length $(length(x)) and A has $(size(f.A, 2)) columns")))

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndAffine{T, M, <:Cholesky}, y::AbstractVector, x::AbstractVector) where {T<:AbstractFloat, M} =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); _checkaffine(f, x); true)
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndAffine{T, M, T}, y::AbstractVector, x::AbstractVector) where {T<:AbstractFloat, M} =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); _checkaffine(f, x); true)
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndAffine{T, M, Tuple{Matrix{T}, Vector{T}}}, y::AbstractVector, x::AbstractVector) where {T<:AbstractFloat, M} =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); _checkaffine(f, x); true)

function (f::IndAffine{T})(x::AbstractVector) where {T}
  _checkreal(f, x)
  _checkaffine(f, x)
  S = float(promote_type(T, eltype(x)))
  R = float(eltype(x))
  eltype(x) <: Integer && return convert(R, _affine_scaled_value(f, x, S))
  r = mul!(f.rb, f.A, x)
  r .-= f.b
  nr = norm(r)
  slack = _tol(S) * max(one(S), norm(f.b), f.scale * norm(x))
  isfinite(nr) && isfinite(slack) && return _ind(nr <= slack, R)
  convert(R, _affine_scaled_value(f, x, S))
end

# Normalize the matrix separately, then put its product and b in common units.
function _affine_scaled_value(f, x, ::Type{S}) where {S}
  ma, mx, mb = _maximum_abs(f.A), _maximum_abs(x), _maximum_abs(f.b)
  if !(isfinite(ma) && isfinite(mx) && isfinite(mb))
    fill!(f.rb, zero(eltype(f.rb)))
    return S(Inf)
  end
  ea = _magnitude_exponent(ma)
  k = _comparison_exponent(ea + _magnitude_exponent(mx), _magnitude_exponent(mb))
  sa, sx = _power_factors(S, -ea), _power_factors(S, ea - k)
  _scaled_affine_residual!(f.rb, f.A, x, f.b, sa, sx, k)
  nb = sqrt(_scaled_sumabs2(f.b, _power_factors(S, -k)))
  na = sqrt(_scaled_sumabs2(f.A, sa))
  nx = sqrt(_scaled_sumabs2(x, sx))
  slack = _tol(S) * max(ldexp(one(S), -k), nb, na * nx)
  _ind(_norm2(f.rb) <= slack, S)
end

# Accumulate scaled products, and subtract b before rounding into the buffer.
function _scaled_affine_residual!(r, A::AbstractMatrix, x, b, sa, sx, k)
  @inbounds for i in axes(A, 1)
    v = zero(sa[1])
    for j in axes(A, 2)
      v += _scale_value(A[i, j], sa) * _scale_value(x[j], sx)
    end
    r[i] = v - ldexp(typeof(v)(b[i]), -k)
  end
  r
end

# Sparse columns update only their stored rows; every buffer entry is reset.
function _scaled_affine_residual!(r, A::SparseMatrixCSC, x, b, sa::NTuple{2, S}, sx, k) where {S}
  S === eltype(r) || return invoke(_scaled_affine_residual!,
    Tuple{Any, AbstractMatrix, Any, Any, Any, Any, Any}, r, A, x, b, sa, sx, k)
  @inbounds for i in eachindex(r, b)
    r[i] = -ldexp(S(b[i]), -k)
  end
  rows, vals = rowvals(A), nonzeros(A)
  @inbounds for j in axes(A, 2)
    v = _scale_value(x[j], sx)
    for h in nzrange(A, j)
      r[rows[h]] += _scale_value(vals[h], sa) * v
    end
  end
  r
end

function prox!(y::AbstractVector, f::IndAffine{T, M, <:Cholesky}, x::AbstractVector, ::Real) where {T<:AbstractFloat, M}
  _checkwrite(prox!, f, y, x)
  r = mul!(f.r, f.A, x)
  r .-= f.b
  ldiv!(f.fact, r)
  y === x || copyto!(y, x)
  mul!(y, f.A', r, -one(T), one(T))
end

function prox!(y::AbstractVector, f::IndAffine{T, M, T}, x::AbstractVector, ::Real) where {T<:AbstractFloat, M}
  _checkwrite(prox!, f, y, x)
  r = mul!(f.r, f.A, x)
  r .-= f.b
  y === x || copyto!(y, x)
  mul!(y, f.A', r, -inv(f.fact), one(T))
end

function prox!(y::AbstractVector, f::IndAffine{T, M, Tuple{Matrix{T}, Vector{T}}}, x::AbstractVector, ::Real) where {T<:AbstractFloat, M}
  _checkwrite(prox!, f, y, x)
  V, c = f.fact
  t = mul!(f.r, V', x)
  t .-= c
  y === x || copyto!(y, x)
  mul!(y, V, t, -one(T), one(T))
end
