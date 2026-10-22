####################################
#
#  Smooth functions that are also proximable
#
####################################

####################################
#  zero
####################################

"""
    ZeroFunction()

The zero function `f(x) = 0`: smooth, with gradient zero, and proximable, with
the identity as proximal operator. The penalty of an unpenalized problem.
"""
struct ZeroFunction <: AbstractFunction end

issmooth(::Type{ZeroFunction}) = true
isproximable(::Type{ZeroFunction}) = true
isseparable(::Type{ZeroFunction}) = true
isconvex(::Type{ZeroFunction}) = true

(::ZeroFunction)(x::AbstractArray) = zero(float(eltype(x)))

function prox!(y::AbstractArray, ::ZeroFunction, x::AbstractArray, ::Real)
  _checksize(y, x)
  y === x || copyto!(y, x)
  y
end
value_and_prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::Real) =
  (prox!(y, f, x, γ); zero(float(eltype(y))))

function value_and_gradient!(g::AbstractArray, ::ZeroFunction, x::AbstractArray)
  _checksize(g, x)
  fill!(g, zero(eltype(g)))
  zero(float(eltype(x)))
end
gradient!(g::AbstractArray, f::ZeroFunction, x::AbstractArray) = (value_and_gradient!(g, f, x); g)

cdprox(::ZeroFunction, v::Real, ::Integer, ::Real) = v
kkt_violation(::ZeroFunction, g::AbstractArray, i::Integer) = abs(g[i])
restrict(f::ZeroFunction, ::AbstractVector{<:Integer}) = f

####################################
#  linear
####################################

"""
    LinearFunction(c, d = 0)

The linear function `f(x) = ⟨c, x⟩ + d`, with gradient `c`. Its proximal operator
is the shift `x - γc`.
"""
struct LinearFunction{C<:AbstractArray, D<:Real} <: AbstractFunction
  c::C
  d::D
end
LinearFunction(c::AbstractArray) = LinearFunction(c, zero(float(eltype(c))))

issmooth(::Type{<:LinearFunction}) = true
isproximable(::Type{<:LinearFunction}) = true
isseparable(::Type{<:LinearFunction}) = true
isconvex(::Type{<:LinearFunction}) = true

(f::LinearFunction)(x::AbstractArray) = dot(f.c, x) + f.d

function prox!(y::AbstractArray, f::LinearFunction, x::AbstractArray, γ::Real)
  _checksize(y, x)
  γ = convert(float(eltype(y)), γ)
  y .= x .- γ .* f.c
end

function value_and_gradient!(g::AbstractArray, f::LinearFunction, x::AbstractArray)
  _checksize(g, x)
  copyto!(g, f.c)
  f(x)
end
gradient!(g::AbstractArray, f::LinearFunction, x::AbstractArray) = (_checksize(g, x); copyto!(g, f.c))

Base.@propagate_inbounds cdprox(f::LinearFunction, v::Real, i::Integer, γ::Real) = v - γ * f.c[i]
_checkparams(f::LinearFunction, x) = _checkparam(f.c, x)
kkt_violation(f::LinearFunction, g::AbstractArray, i::Integer) = abs(g[i] + f.c[i])
restrict(f::LinearFunction, S::AbstractVector{<:Integer}) = LinearFunction(f.c[S], f.d)

####################################
#  squared Euclidean norm
####################################

"""
    SqrNormL2(λ = 1)
    SqrNormL2(λ, y)

The squared Euclidean norm `f(x) = (λ/2) ‖x - y‖²`, with `y = 0` when omitted:
the ridge penalty, and for `λ = 1` the loss `‖x - y‖² / 2`. Smooth, with
gradient `λ(x - y)`, separable, and proximable, with proximal operator
`(x + γλy) / (1 + γλ)`. For a matrix `x`, `‖⋅‖` is the Frobenius norm.
"""
struct SqrNormL2{T<:Real, Y<:Union{Nothing, AbstractArray}} <: AbstractFunction
  λ::T
  y::Y

  function SqrNormL2{T, Y}(λ::T, y::Y) where {T<:Real, Y<:Union{Nothing, AbstractArray}}
    _checknonneg("λ", λ)
    new{T, Y}(λ, y)
  end
end
SqrNormL2(λ::Real = 1, y::Union{Nothing, AbstractArray} = nothing) = SqrNormL2{typeof(λ), typeof(y)}(λ, y)

issmooth(::Type{<:SqrNormL2}) = true
isproximable(::Type{<:SqrNormL2}) = true
isseparable(::Type{<:SqrNormL2}) = true
isconvex(::Type{<:SqrNormL2}) = true

# Σ xᵢ² over the entries of x, in O(nnz) for the sparse iterates
_sumabs2(x::AbstractArray) = sum(abs2, x)
function _sumabs2(x::SparseIterate)
  s = zero(float(eltype(x)))
  @inbounds @simd for k in 1:x.nnz
    s += abs2(x.nzval[k])
  end
  s
end
function _sumabs2(x::SymmetricSparseIterate{T}) where {T}
  s = zero(T)
  @inbounds for k = 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    s += (r == c ? 1 : 2) * abs2(x.nzval[k])
  end
  s
end
_sumabs2(x::AtomIterate) = sum(abs2, x.storage)

function (f::SqrNormL2)(x::AbstractArray)
  T = float(eltype(x))
  s = f.y === nothing ? _sumabs2(x) : _sumabs2_diff(x, f.y)
  convert(T, f.λ) * s / 2
end

function value_and_gradient!(g::AbstractArray, f::SqrNormL2, x::AbstractArray)
  _checksize(g, x)
  λ = convert(float(eltype(g)), f.λ)
  f.y === nothing ? (g .= λ .* x) : (g .= λ .* (x .- f.y))
  f(x)
end
function gradient!(g::AbstractArray, f::SqrNormL2, x::AbstractArray)
  _checksize(g, x)
  λ = convert(float(eltype(g)), f.λ)
  f.y === nothing ? (g .= λ .* x) : (g .= λ .* (x .- f.y))
end

function prox!(z::AbstractArray, f::SqrNormL2, x::AbstractArray, γ::Real)
  _checksize(z, x)
  T = float(eltype(z))
  t = convert(T, γ) * convert(T, f.λ)
  f.y === nothing ? (z .= x ./ (1 + t)) : (z .= (x .+ t .* f.y) ./ (1 + t))
end

cdprox(f::SqrNormL2{<:Real, Nothing}, v::Real, ::Integer, γ::Real) = v / (1 + γ * f.λ)
Base.@propagate_inbounds cdprox(f::SqrNormL2, v::Real, i::Integer, γ::Real) = (v + γ * f.λ * f.y[i]) / (1 + γ * f.λ)
_checkparams(f::SqrNormL2, x) = _checkparam(f.y, x)

kkt_violation(::SqrNormL2{<:Real, Nothing}, g::AbstractArray, i::Integer) = abs(g[i])
kkt_violation(f::SqrNormL2, g::AbstractArray, i::Integer) = abs(g[i] - f.λ * f.y[i])

restrict(f::SqrNormL2{<:Real, Nothing}, ::AbstractVector{<:Integer}) = f
restrict(f::SqrNormL2, S::AbstractVector{<:Integer}) = SqrNormL2(f.λ, f.y[S])

####################################
#  quadratic
####################################

"""
    QuadraticFunction(A, b = nothing, c = 0)

The quadratic `f(x) = x'Ax / 2 + b'x + c` for a symmetric positive semidefinite
`A`, with `b = 0` when omitted. Smooth, with gradient `Ax + b`, and proximable:
the first proximal operator computes the eigendecomposition `A = QΛQ'` once,
after which `prox_{γf}(x) = Q (I + γΛ)⁻¹ Q'(x - γb)` costs two products with
`Q` for any `γ`.
"""
mutable struct QuadraticFunction{T<:AbstractFloat, M<:AbstractMatrix, V<:Union{Nothing, AbstractVector}} <: AbstractFunction
  A::M
  b::V
  c::T
  tmp::Vector{T}        # A x, and scratch for the proximal operator
  Λ::Vector{T}          # A = Q Diagonal(Λ) Q', computed by the first prox
  Q::Matrix{T}
  w::Vector{T}
end

function QuadraticFunction(A::AbstractMatrix, b::Union{Nothing, AbstractVector} = nothing, c::Real = 0)
  n = checksquare(A)
  b === nothing || length(b) == n || throw(DimensionMismatch("A is $n × $n and b has length $(length(b))"))
  T = float(promote_type(eltype(A), b === nothing ? eltype(A) : eltype(b), typeof(c)))
  QuadraticFunction{T, typeof(A), typeof(b)}(A, b, T(c), zeros(T, n), T[], Matrix{T}(undef, 0, 0), zeros(T, n))
end

issmooth(::Type{<:QuadraticFunction}) = true
isproximable(::Type{<:QuadraticFunction}) = true
isconvex(::Type{<:QuadraticFunction}) = true

function (f::QuadraticFunction)(x::AbstractVector)
  mul!(f.tmp, f.A, x)
  v = dot(x, f.tmp) / 2 + f.c
  f.b === nothing ? v : v + dot(f.b, x)
end

function value_and_gradient!(g::AbstractVector, f::QuadraticFunction, x::AbstractVector)
  _checksize(g, x)
  mul!(g, f.A, x)
  v = dot(g, x) / 2 + f.c
  if f.b !== nothing
    g .+= f.b
    v += dot(f.b, x)
  end
  v
end

function gradient!(g::AbstractVector, f::QuadraticFunction, x::AbstractVector)
  _checksize(g, x)
  mul!(g, f.A, x)
  f.b === nothing || (g .+= f.b)
  g
end

function _eigen!(f::QuadraticFunction{T}) where {T}
  if isempty(f.Λ)
    E = eigen(Symmetric(Matrix{T}(f.A)))
    f.Λ, f.Q = E.values, E.vectors
  end
  f
end

function prox!(y::AbstractVector, f::QuadraticFunction{T}, x::AbstractVector, γ::Real) where {T}
  _checksize(y, x)
  length(x) == length(f.tmp) || throw(DimensionMismatch())
  _eigen!(f)
  γ = convert(T, γ)
  t = f.tmp
  f.b === nothing ? copyto!(t, x) : (t .= x .- γ .* f.b)
  mul!(f.w, transpose(f.Q), t)
  f.w ./= 1 .+ γ .* f.Λ
  mul!(t, f.Q, f.w)
  copyto!(y, t)
end

restrict(f::QuadraticFunction, S::AbstractVector{<:Integer}) =
  QuadraticFunction(f.A[S, S], f.b === nothing ? nothing : f.b[S], f.c)
