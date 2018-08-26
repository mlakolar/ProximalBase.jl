
##########################################################
#
#  Create DifferentiableFunction
#
##########################################################

abstract type DifferentiableFunction end


################ creates L2Loss =>  f(x) = ||x - y||^2 / 2

# N = 1 --> y
# N = 0 --> y = 0
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
      s += (x[i]-y[i])^2.
    end
  else
    s = sum(abs2, x)
  end
  s / 2.
end

function value_and_gradient!(f::L2Loss{T, N}, hat_x::AbstractArray{T}, x::AbstractArray{T}) where {T<:AbstractFloat, N}
  if N == 1
    @. hat_x = x - f.y
  else
    copyto!(hat_x, x)
  end
  sum(abs2, hat_x) / 2.
end

################ creates a function x'Ax/2 + b'x + c

struct QuadraticFunction{T<:AbstractFloat, N, M<:AbstractMatrix, V} <: DifferentiableFunction
  A::M
  b::V
  c::T
  tmp::Vector{T}    ## call to value does not allocate
end

QuadraticFunction(A::AbstractMatrix{T}) where {T<:AbstractFloat} = QuadraticFunction{T, 1, typeof(A), Vector{T}}(A, T[], zero(T), Array{T}(undef, size(A, 1)))
QuadraticFunction(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:AbstractFloat} = QuadraticFunction{T, 2, typeof(A), typeof(b)}(A, b, zero(T), Array{T}(undef, size(A, 1)))
QuadraticFunction(A::AbstractMatrix{T}, b::AbstractVector{T}, c::T) where {T<:AbstractFloat} = QuadraticFunction{T, 3, typeof(A), typeof(b)}(A, b, c, Array{T}(undef, size(A, 1)))

function value(f::QuadraticFunction{T, 1}, x::StridedVector{T}) where {T<:AbstractFloat}
  mul!(f.tmp, f.A, x)
  dot(x, f.tmp) / 2.
end
function value(f::QuadraticFunction{T, 2}, x::StridedVector{T}) where {T<:AbstractFloat}
  mul!(f.tmp, f.A, x)
  dot(x, f.tmp) / 2. + dot(x, f.b)
end
function value(f::QuadraticFunction{T, 3}, x::StridedVector{T}) where {T<:AbstractFloat}
  mul!(f.tmp, f.A, x)
  dot(x, f.tmp) / 2. + dot(x, f.b) + f.c
end

function gradient!(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::StridedVector{T}) where {T<:AbstractFloat, N}
  mul!(hat_x, f.A, x)
  if N > 1
    @. hat_x += f.b
  end
  hat_x
end

function value_and_gradient!(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::StridedVector{T}) where {T<:AbstractFloat, N}
  b = f.b
  mul!(hat_x, f.A, x)
  r = dot(hat_x, x) / 2.
  if N > 1
    @. hat_x += b
    r += dot(x, b)
  end
  r + f.c
end



################ creates a function |Y - X⋅β|_2^2 / 2.

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
  @inbounds for i in eachindex(f.tmp)
    v += (f.Y[i] - f.tmp[i])^2.
  end
  v / 2.
end

function value_and_gradient!(
  f::LeastSquaresLoss{T},
  grad_out::StridedVector{T},
  x) where {T<:AbstractFloat}

  mul!(f.tmp, f.X, x)
  @. f.tmp -= f.Y
  mul!(grad_out, transpose(f.X), f.tmp)
  sum(abs2, f.tmp) / 2.
end
