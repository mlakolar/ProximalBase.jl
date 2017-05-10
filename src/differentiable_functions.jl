
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

L2Loss{T<:AbstractFloat}(y::AbstractArray{T}) = L2Loss{T, 1, typeof(y)}(y)
L2Loss() = L2Loss{Float64, 0, Vector{Float64}}(Float64[])

function value{T<:AbstractFloat, N}(f::L2Loss{T, N}, x::AbstractArray{T})
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


# function value{T<:AbstractFloat, N}(f::L2Loss{T, N}, x::AbstractArray{T}, activeset::ActiveSet)
#   s = zero(T)
#   if N == 1
#     y = f.y
#     @assert size(y) == size(x)
#   end
#   indexes = activeset.indexes
#   @inbounds for i=1:activeset.numActive
#     ind = indexes[i]
#     if N == 1
#       s += (x[ind]-y[ind])^2.
#     else
#       s += x[ind]^2.
#     end
#   end
#   s / 2.
# end

function value_and_gradient!{T<:AbstractFloat, N}(f::L2Loss{T, N}, hat_x::AbstractArray{T}, x::AbstractArray{T})
  if N == 1
    @. hat_x = x - f.y
  else
    copy!(hat_x, x)
  end
  sum(abs2, hat_x) / 2.
end

# function value_and_gradient!{T<:AbstractFloat, N}(f::L2Loss{T, N}, hat_x::AbstractArray{T}, x::AbstractArray{T}, activeset::ActiveSet)
#   s = zero(T)
#   if N == 1
#     y = f.y
#     @assert size(y) == size(x)
#   end
#   indexes = activeset.indexes
#   @inbounds for rr=1:activeset.numActive
#     t = indexes[rr]
#     s += N == 1 ? (x[t]-y[t])^2. : x[t]^2.
#     hat_x[t] = N == 1 ? x[t] - y[t] : x[t]
#   end
#   s / 2.
# end

################

################ creates a function x'Ax/2 + b'x + c

immutable QuadraticFunction{T<:AbstractFloat, N, M<:AbstractMatrix, V} <: DifferentiableFunction
  A::M
  b::V
  c::T
  tmp::Vector{T}    ## call to value does not allocate
end

QuadraticFunction{T<:AbstractFloat}(A::AbstractMatrix{T}) = QuadraticFunction{T, 1, typeof(A), Vector{T}}(A, T[], zero(T), Array{T}(size(A, 1)))
QuadraticFunction{T<:AbstractFloat}(A::AbstractMatrix{T}, b::AbstractVector{T}) = QuadraticFunction{T, 2, typeof(A), typeof(b)}(A, b, zero(T), Array{T}(size(A, 1)))
QuadraticFunction{T<:AbstractFloat}(A::AbstractMatrix{T}, b::AbstractVector{T}, c::T) = QuadraticFunction{T, 3, typeof(A), typeof(b)}(A, b, c, Array{T}(size(A, 1)))

function value{T<:AbstractFloat}(f::QuadraticFunction{T, 1}, x::StridedVector{T})
  A_mul_B!(f.tmp, f.A, x)
  dot(x, f.tmp) / 2.
end
function value{T<:AbstractFloat}(f::QuadraticFunction{T, 2}, x::StridedVector{T})
  A_mul_B!(f.tmp, f.A, x)
  dot(x, f.tmp) / 2. + dot(x, f.b)
end
function value{T<:AbstractFloat}(f::QuadraticFunction{T, 3}, x::StridedVector{T})
  A_mul_B!(f.tmp, f.A, x)
  dot(x, f.tmp) / 2. + dot(x, f.b) + f.c
end

# function value{T<:AbstractFloat, N}(f::QuadraticFunction{T, N}, x::StridedVector{T}, activeset::ActiveSet)
#   A = f.A
#   if N > 1
#     b = f.b
#   end
#   indexes = activeset.indexes
#
#   s = zero(T)
#   # dot(x, f.A*x) / 2.
#   @inbounds for cc=1:activeset.numActive
#     ci = indexes[cc]
#     @inbounds for rr=1:activeset.numActive
#       ri = indexes[rr]
#       s += A[ri, ci] * x[ri] * x[ci]
#     end
#   end
#   s /= 2.
#
#   if N > 1
#     # dot(x, f.b)
#     @inbounds for rr=1:activeset.numActive
#       t = indexes[rr]
#       s += x[t] * b[t]
#     end
#   end
#   if N > 2
#     s += f.c
#   end
#   s
# end

function value_and_gradient!{T<:AbstractFloat, N}(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::StridedVector{T})
  b = f.b
  A_mul_B!(hat_x, f.A, x)
  r = dot(hat_x, x) / 2.
  if N > 1
    @. hat_x += b
    r += dot(x, b)
  end
  r + f.c
end


# function value_and_gradient!{T<:AbstractFloat, N}(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::StridedVector{T}, activeset::ActiveSet)
#   A = f.A
#   if N > 1
#     b = f.b
#   end
#
#   indexes = activeset.indexes
#
#   s = zero(T)
#   # dot(x, f.A*x) / 2.
#   @inbounds for rr=1:activeset.numActive
#     ri = indexes[rr]
#     hat_x[ri] = zero(T)
#     @inbounds for cc=1:activeset.numActive
#       ci = indexes[cc]
#       hat_x[ri] += A[ri, ci] * x[ci]
#     end
#     s += hat_x[ri] * x[ri] / 2.
#     if N > 1
#       hat_x[ri] += b[ri]
#       s += b[ri] * x[ri]
#     end
#   end
#   if N > 2
#     s += f.c
#   end
#   s
# end




####################
