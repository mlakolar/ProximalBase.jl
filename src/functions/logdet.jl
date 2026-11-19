"""
    LogDetLoss(S, w = 1)

The loss `w * (tr(S * X) - logdet(X))` on symmetric positive-definite matrices.
The constructor copies a finite, nonempty square matrix `S` with exact symmetry.
Use `Symmetric(S, :L)` or `Symmetric(S, :U)` to select a triangle explicitly.
The weight must be finite and strictly positive. `S` may be indefinite.

Use one-based matrices and a common `Float32` or `Float64` type for data,
weight, scalar step, and output buffers. Integer weights and steps are also
accepted. Treat the stored coefficients as read-only. Scratch is local to
each call.

A NaN entry in `X` gives a NaN value. Other points outside the domain give
`Inf`. The gradient is `w * (S - inv(X))` on the domain and all NaN outside it.
The proximal map at a positive finite step accepts finite nonsymmetric and
indefinite inputs. A zero step copies the input. Negative, infinite and NaN
steps throw `ArgumentError`.

Allocating oracles return full dense matrices. Mutating oracles accept
writable strided matrices, including row-strided views. A
[`SymmetricSparseIterate`](@ref) can supply the logical input matrix. Gradients
use the Frobenius inner product on full matrices.
"""
struct LogDetLoss{T<:Union{Float32, Float64}} <: AbstractFunction
  S::Matrix{T}
  w::T

  function LogDetLoss(S::AbstractMatrix{T}, w::Real = 1) where {T}
    _check_one_based(S)
    T <: Union{Float32, Float64} || throw(ArgumentError("LogDetLoss requires Float32 or Float64 data"))
    checksquare(S) > 0 || throw(ArgumentError("LogDetLoss requires a nonempty matrix S"))
    weight = _logdet_scalar(T, w)
    isfinite(weight) && weight > 0 || throw(ArgumentError("LogDetLoss weight must be finite and positive"))
    coefficients = Matrix(S)
    all(isfinite, coefficients) || throw(ArgumentError("LogDetLoss requires finite entries in S"))
    _check_symmetric(coefficients, "the LogDetLoss coefficients", "S")
    new{T}(coefficients, weight)
  end
end

issmooth(::Type{<:LogDetLoss}) = true
isconvex(::Type{<:LogDetLoss}) = true
isproximable(::Type{<:LogDetLoss}) = true
lipschitz(::LogDetLoss{T}) where {T} = T(Inf)

function _logdet_scalar(::Type{T}, x::Real) where {T}
  x isa Union{T, Integer} || throw(ArgumentError("LogDetLoss scalar and matrix types must match"))
  T(x)
end

_argument_type(::Type{<:LogDetLoss}) = Union{AbstractMatrix, SymmetricSparseIterate}
check_parameters(f::LogDetLoss, X::Union{AbstractMatrix, SymmetricSparseIterate}) = (_checksize(f.S, X); nothing)

function _check_logdet_input(f::LogDetLoss{T}, X) where {T}
  _check_one_based(X)
  _checksize(f.S, X)
  eltype(X) === T || throw(ArgumentError("LogDetLoss input and data types must match"))
end

function _check_logdet_output(Y, f, X)
  _check_one_based(Y)
  _checksize(Y, X)
  eltype(Y) === eltype(X) || throw(ArgumentError("LogDetLoss output and input types must match"))
  Y isa StridedMatrix || throw(ArgumentError("LogDetLoss requires a writable strided matrix output"))
  Base.mightalias(Y, f.S) && throw(ArgumentError("LogDetLoss output must be independent of S"))
end

function _logdet_oracle!(G, f::LogDetLoss{T}, X) where {T}
  if all(isfinite, X)
    if _symmetric_by_type(X) || _issymmetric_exact(X, _symequal)
      F = cholesky!(Symmetric(Matrix(X)); check=false)
      if issuccess(F)
        ld = logdet(F)
        trace = dot(f.S, X)
        v = f.w * (trace - ld)
        if !isfinite(v) || (iszero(ld) && abs(trace) < floatmin(T))
          acc = _expsum_dot(_expsum_start(Float64), f.S, X, 0)
          v = _expsum_value(T, _expsum_product(acc, -ld, one(T)), f.w)
        end
        if G !== nothing
          inverse = LinearAlgebra.inv!(F)
          G .= f.w .* (f.S .- inverse)
        end
        return v
      end
    end
    v = T(Inf)
  else
    v = any(isnan, X) ? T(NaN) : T(Inf)
  end
  G === nothing || fill!(G, T(NaN))
  v
end

function (f::LogDetLoss)(X::AbstractMatrix)
  _check_logdet_input(f, X)
  _logdet_oracle!(nothing, f, X)
end

function value_and_gradient!(G::AbstractMatrix, f::LogDetLoss, X::AbstractMatrix)
  _check_logdet_input(f, X)
  _check_logdet_output(G, f, X)
  _checkgradient_output(G, X)
  _logdet_oracle!(G, f, X)
end

gradient!(G::AbstractMatrix, f::LogDetLoss, X::AbstractMatrix) = (value_and_gradient!(G, f, X); G)

# The positive root of z^2 - dz - r^2. The reciprocal form avoids cancellation
# for negative d and does not require forming r^2.
@inline function _logdet_root(d, r)
  h = d / 2
  q = hypot(h, r)
  d >= 0 && return h + q
  denominator = q - h
  isinf(denominator) && return r * ((r / 2) / (hypot(h / 2, r / 2) - h / 2))
  r * (r / denominator)
end

function prox!(Y::AbstractMatrix, f::LogDetLoss{T}, X::AbstractMatrix, γ::Real) where {T}
  _check_logdet_input(f, X)
  _check_logdet_output(Y, f, X)
  step = _logdet_scalar(T, γ)
  isfinite(step) && step >= 0 || throw(ArgumentError("LogDetLoss step must be finite and nonnegative"))
  iszero(step) && return copyto!(Y, X)
  all(isfinite, X) || throw(ArgumentError("LogDetLoss requires finite input at positive steps"))

  μ = step * f.w
  ordinary = isfinite(μ) && μ >= floatmin(T)
  r = ordinary ? sqrt(μ) : sqrt(step) * sqrt(f.w)
  B = Matrix{T}(undef, size(X))
  @inbounds for j in axes(B, 2), i in 1:j
    x = i == j ? X[i, j] : _symmetric_mean(X[i, j], X[j, i])
    s = f.S[i, j]
    B[i, j] = ordinary ? muladd(-μ, s, x) : x - r * (r * s)
    isfinite(B[i, j]) || _logdet_prox_range_error()
  end
  values, U = LAPACK.syevd!('V', 'U', B)
  @inbounds for j in eachindex(values)
    d = values[j]
    isfinite(d) || _logdet_prox_range_error()
    z = _logdet_root(d, r)
    isfinite(z) || _logdet_prox_range_error()
    for i in axes(U, 1)
      U[i, j] *= sqrt(z)
    end
  end
  Z = Symmetric(BLAS.syrk('U', 'N', U))
  all(isfinite, Z) || _logdet_prox_range_error()
  copyto!(Y, Z)
end

@noinline _logdet_prox_range_error() = throw(ArgumentError("LogDetLoss proximal computation exceeded the finite working range"))

value_and_prox!(Y::AbstractMatrix, f::LogDetLoss, X::AbstractMatrix, γ::Real = 1) =
  (prox!(Y, f, X, γ); f(Y))

# Always allocate a full matrix, including for structured and sparse inputs.
const _LogDetMatrixInput = Union{AbstractMatrix, SymmetricSparseIterate}
prox(f::LogDetLoss, X::_LogDetMatrixInput, γ::Real = 1) =
  (_check_one_based(X); prox!(_gradient_buffer(X), f, X, γ))
prox(f::LogDetLoss, X::Union{Symmetric, Hermitian}, γ::Real = 1) =
  (_check_one_based(X); prox!(_gradient_buffer(X), f, X, γ))

(f::LogDetLoss)(X::SymmetricSparseIterate) = f(matrixview(X))
gradient(f::LogDetLoss, X::SymmetricSparseIterate) = gradient(f, matrixview(X))
value_and_gradient(f::LogDetLoss, X::SymmetricSparseIterate) = value_and_gradient(f, matrixview(X))
gradient!(G::AbstractMatrix, f::LogDetLoss, X::SymmetricSparseIterate) = gradient!(G, f, matrixview(X))
value_and_gradient!(G::AbstractMatrix, f::LogDetLoss, X::SymmetricSparseIterate) =
  value_and_gradient!(G, f, matrixview(X))
prox!(Y::AbstractMatrix, f::LogDetLoss, X::SymmetricSparseIterate, γ::Real) = prox!(Y, f, matrixview(X), γ)
value_and_prox!(Y::AbstractMatrix, f::LogDetLoss, X::SymmetricSparseIterate, γ::Real = 1) =
  value_and_prox!(Y, f, matrixview(X), γ)
