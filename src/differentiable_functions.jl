
##########################################################
#
#  Create DifferentiableFunction
#
##########################################################

"""
    DifferentiableFunction

Abstract supertype for smooth functions `f`. Subtypes implement [`value`](@ref)
and [`value_and_gradient!`](@ref), and optionally [`gradient!`](@ref).
"""
abstract type DifferentiableFunction end

"""
    value(f, x)

Evaluate `f(x)`. Defined for both [`DifferentiableFunction`](@ref)s and
[`ProximableFunction`](@ref)s.
"""
function value end

"""
    gradient!(f, grad, x)

Write the gradient of `f` at `x` into `grad` and return `grad`.
"""
function gradient! end

"""
    value_and_gradient!(f, grad, x)

Write the gradient of `f` at `x` into `grad` and return `f(x)`. Usually cheaper
than calling [`value`](@ref) and [`gradient!`](@ref) separately.
"""
function value_and_gradient! end

"""
    restrict(f, S)
    restrict(g, S)

The function of `x[S]` that `f` (or `g`) becomes on the vectors `x` supported on
the coordinates `S`, as an ordinary function of the same kind. An active-set
solver minimizes the restriction over the active coordinates `S`.

For such `x`, at `x[S]` the restriction `restrict(g, S)` has value `g(x)`, and
`restrict(f, S)` has gradient `∇f(x)[S]` and value `f(x)`, except for
`L2Loss(y)`, whose restriction `L2Loss(y[S])` drops the constant
`Σⱼ∉S yⱼ² / 2`. Defined for [`QuadraticFunction`](@ref),
[`L2Loss`](@ref), [`LeastSquaresLoss`](@ref) with a vector response,
[`ProxZero`](@ref), [`ProxL1`](@ref) and [`ProxGroupL2`](@ref), whose
restriction keeps the part of each group in `S`, as positions in `S`, with the
group's weight.
"""
function restrict end


################ creates L2Loss =>  f(x) = ||x - y||^2 / 2

# N = 1 --> y
# N = 0 --> y = 0
"""
    L2Loss(y)
    L2Loss()

The loss `f(x) = ‖x - y‖² / 2`. Without `y`, `f(x) = ‖x‖² / 2`.
"""
struct L2Loss{T<:AbstractFloat, N, M<:AbstractArray} <: DifferentiableFunction
  y::M
end

L2Loss(y::AbstractArray{T}) where {T<:AbstractFloat} = L2Loss{T, 1, typeof(y)}(y)
L2Loss() = L2Loss{Float64, 0, Vector{Float64}}(Float64[])

function value(f::L2Loss{T, N}, x::AbstractArray{T}) where {T<:AbstractFloat, N}
  s = zero(T)
  if N == 1
    y = f.y
    @assert size(y) == size(x)
    @inbounds @simd for i in eachindex(x)
      s += abs2(x[i]-y[i])
    end
  else
    s = sum(abs2, x)
  end
  s / 2
end

function value_and_gradient!(f::L2Loss{T, N}, hat_x::AbstractArray{T}, x::AbstractArray{T}) where {T<:AbstractFloat, N}
  if N == 1
    y = f.y
    axes(y) == axes(x) == axes(hat_x) || throw(DimensionMismatch())
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
      d = x[i] - y[i]
      hat_x[i] = d
      s += abs2(d)
    end
  else
    s = sum(abs2, copyto!(hat_x, x))
  end
  s / 2
end

restrict(f::L2Loss{T, 1}, S::AbstractVector{<:Integer}) where {T} = L2Loss(f.y[S])
restrict(f::L2Loss{T, 0}, ::AbstractVector{<:Integer}) where {T} = f

################ creates a function x'Ax/2 + b'x + c

"""
    QuadraticFunction(A)
    QuadraticFunction(A, b)
    QuadraticFunction(A, b, c)

The quadratic `f(x) = x'Ax / 2 + b'x + c`, with `b = 0` and `c = 0` when omitted.
The gradient is computed as `Ax + b`, so `A` should be symmetric.
"""
struct QuadraticFunction{T<:AbstractFloat, N, M<:AbstractMatrix, V} <: DifferentiableFunction
  A::M
  b::V
  c::T
  tmp::Vector{T}    ## call to value does not allocate
end

# a SparseIterate or an AtomIterate x takes the O(p ⋅ active entries) product A * x
const _VectorOrIterate{T} = Union{StridedVector{T}, SparseIterate{T, 1}, AtomIterate{T}}

QuadraticFunction(A::AbstractMatrix{T}) where {T<:AbstractFloat} = QuadraticFunction{T, 1, typeof(A), Vector{T}}(A, T[], zero(T), Array{T}(undef, size(A, 1)))
QuadraticFunction(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:AbstractFloat} = QuadraticFunction{T, 2, typeof(A), typeof(b)}(A, b, zero(T), Array{T}(undef, size(A, 1)))
QuadraticFunction(A::AbstractMatrix{T}, b::AbstractVector{T}, c::T) where {T<:AbstractFloat} = QuadraticFunction{T, 3, typeof(A), typeof(b)}(A, b, c, Array{T}(undef, size(A, 1)))

function value(f::QuadraticFunction{T, 1}, x::_VectorOrIterate{T}) where {T<:AbstractFloat}
  mul!(f.tmp, f.A, x)
  dot(x, f.tmp) / 2
end
function value(f::QuadraticFunction{T, 2}, x::_VectorOrIterate{T}) where {T<:AbstractFloat}
  mul!(f.tmp, f.A, x)
  dot(x, f.tmp) / 2 + dot(x, f.b)
end
function value(f::QuadraticFunction{T, 3}, x::_VectorOrIterate{T}) where {T<:AbstractFloat}
  mul!(f.tmp, f.A, x)
  dot(x, f.tmp) / 2 + dot(x, f.b) + f.c
end

function gradient!(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::_VectorOrIterate{T}) where {T<:AbstractFloat, N}
  mul!(hat_x, f.A, x)
  if N > 1
    @. hat_x += f.b
  end
  hat_x
end

function value_and_gradient!(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::_VectorOrIterate{T}) where {T<:AbstractFloat, N}
  b = f.b
  mul!(hat_x, f.A, x)
  r = dot(hat_x, x) / 2
  if N > 1
    @. hat_x += b
    r += dot(x, b)
  end
  r + f.c
end

restrict(f::QuadraticFunction{T, 1}, S::AbstractVector{<:Integer}) where {T} = QuadraticFunction(f.A[S, S])
restrict(f::QuadraticFunction{T, 2}, S::AbstractVector{<:Integer}) where {T} = QuadraticFunction(f.A[S, S], f.b[S])
restrict(f::QuadraticFunction{T, 3}, S::AbstractVector{<:Integer}) where {T} = QuadraticFunction(f.A[S, S], f.b[S], f.c)



################ creates a function |Y - X⋅β|_2^2 / 2.

"""
    LeastSquaresLoss(Y, X)

The least-squares loss `f(β) = ‖Y - Xβ‖² / 2`, with gradient `X'(Xβ - Y)`.
`Y` may be a vector or an `n × k` matrix with one column per response; then `β`
and the gradient are `p × k` matrices and `‖⋅‖` is the Frobenius norm, as in
multi-response regression with [`ProxNuclear`](@ref) for a low-rank `β`.
"""
struct LeastSquaresLoss{T, Ty<:AbstractVecOrMat, Tx<:AbstractMatrix} <: DifferentiableFunction
  Y::Ty
  X::Tx
  tmp::VecOrMat{T}    ## call to value does not allocate
end

LeastSquaresLoss(Y::AbstractVecOrMat{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat} =
    LeastSquaresLoss{T, typeof(Y), typeof(X)}(Y, X, zeros(T, size(Y)))


function value(f::LeastSquaresLoss{T}, x) where {T<:AbstractFloat}
  mul!(f.tmp, f.X, x)
  v = zero(T)
  @inbounds @simd for i in eachindex(f.tmp)
    v += abs2(f.Y[i] - f.tmp[i])
  end
  v / 2
end

function value_and_gradient!(
  f::LeastSquaresLoss{T},
  grad_out::StridedVecOrMat{T},
  x) where {T<:AbstractFloat}

  mul!(f.tmp, f.X, x)
  @. f.tmp -= f.Y
  mul!(grad_out, transpose(f.X), f.tmp)
  sum(abs2, f.tmp) / 2
end

restrict(f::LeastSquaresLoss{T, <:AbstractVector}, S::AbstractVector{<:Integer}) where {T} =
  LeastSquaresLoss(f.Y, f.X[:, S])
