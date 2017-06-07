##########################################################
#
#  Proximal Operators
#
##########################################################

abstract type ProximableFunction end

prox!(g::ProximableFunction, hat_x::AbstractArray, x::AbstractArray) = prox!(g, hat_x, x, one(eltype(x)))
prox(g::ProximableFunction,                        x::AbstractArray, γ) = prox!(g, similar(x), x, γ)
prox(g::ProximableFunction,                        x::AbstractArray) = prox!(g, similar(x), x, one(eltype(x)))



##########################################################
######  zero
##########################################################

struct ProxZero <: ProximableFunction end

value(g::ProxZero, x::AbstractVecOrMat{T}) where {T} = zero(T)

function prox!(::ProxZero, out_x::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, γ::T) where {T}
  size(out_x) == size(x) || throw(ArgumentError("Sizes of the input and ouput need to be the same."))
  copy!(out_x, x)
end

##########################################################
######  L1 norm  g(x) = λ0 ⋅ \sum_j λ_j |x_j|
##########################################################

struct ProxL1{T<:AbstractFloat, S} <: ProximableFunction
  λ0::T
  λ::S #Array{T, N}

  ProxL1{T, S}(λ0::T, λ::Union{Void, AbstractArray{T}}) where {T <: AbstractFloat, S} = new(λ0, λ)
end

ProxL1(λ0::T        ) where {T <: AbstractFloat} = ProxL1{T, Void}(λ0, nothing)
ProxL1(λ0::T, ::Void) where {T <: AbstractFloat} = ProxL1{T, Void}(λ0, nothing)
ProxL1(λ0::T, λ::AbstractArray{T}) where {T <: AbstractFloat} = ProxL1{T, typeof(λ)}(λ0, λ)

function value(g::ProxL1{T, S}, x::AbstractArray{T}) where {T <: AbstractFloat} where S <: AbstractArray
  size(g.λ) == size(x) || throw(DimensionMismatch("Sizes of g.λ and x need to be the same"))
  v = zero(T)
  @inbounds @simd for i in eachindex(x)
    v += abs(x[i]) * g.λ[i]
  end
  v * g.λ0
end
prox!(g::ProxL1{T, S}, out_x::AbstractArray{T}, x::AbstractArray{T}, γ::T) where {T <: AbstractFloat} where S <: AbstractArray =
    out_x .= shrink.(x, (g.λ0 * γ) * g.λ)
@inline function cdprox!(g::ProxL1{T, S}, x::SparseIterate{T}, k::Int, γ::T) where {T <: AbstractFloat} where S <: AbstractArray
  size(g.λ) == size(x) || throw(DimensionMismatch())
  @boundscheck checkbounds(x, k)
  x[k] = shrink(x[k], g.λ[k] * γ * g.λ0)
end

value(g::ProxL1{T, Void}, x::AbstractArray{T}) where {T<:AbstractFloat} = g.λ0 * sum(abs, x)
prox!(g::ProxL1{T, Void}, out_x::AbstractArray{T}, x::AbstractArray{T}, γ::T) where {T<:AbstractFloat} =
  out_x .= shrink.(x, γ * g.λ0)
cdprox!(g::ProxL1{T, Void}, x::SparseIterate{T}, k::Int, γ::T) where {T} =
  x[k] = shrink(x[k], g.λ0 * γ)

##########################################################
###### L1 + Fused  g(x1,x2) = λ1*(|x1|+|x2|) + λ2*|x1-x2|
##########################################################

struct ProxL1Fused{T<:AbstractFloat} <: ProximableFunction
  λ1::T
  λ2::T
end

value{T<:AbstractFloat}(g::ProxL1Fused{T}, x1::T, x2::T) = g.λ1*(abs(x1)+abs(x2)) + g.λ2*abs(x1-x2)
function prox!{T<:AbstractFloat}(g::ProxL1Fused{T}, out_x::Tuple{StridedVector{T},StridedVector{T}}, x::Tuple{StridedVector{T},StridedVector{T}}, γ::T)
  @assert size(out_x[1]) == size(x[1])
  @assert size(out_x[2]) == size(x[2])
  λ1 = g.λ1
  λ2 = g.λ2
  @inbounds @simd for i in eachindex(x[1])
    out_x[1][i], out_x[2][i] = proxL1Fused(x[1][i], x[2][i], λ1, λ2, γ)
  end
  out_x
end


##########################################################
###### L2 norm   g(x) = λ * ||x||_2
##########################################################

struct ProxL2{T<:AbstractFloat} <: ProximableFunction
  λ::T
end

value{T<:AbstractFloat}(g::ProxL2{T}, x::StridedVector{T}) = g.λ * vecnorm(x)
prox!{T<:AbstractFloat}(g::ProxL2{T}, out_x::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, γ::T) =
  shrinkL2!(out_x, x, g.λ*γ)

##########################################################
###### L2 norm squared g(x) = λ * ||x||_2^2
##########################################################

struct ProxL2Sq{T<:AbstractFloat} <: ProximableFunction
  λ::T
end
value(g::ProxL2Sq, x::StridedArray) = g.λ * sum(abs2, x)
function prox!{T<:AbstractFloat}(g::ProxL2Sq{T}, out_x::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, γ::T)
  size(out_x) == size(x) || throw(ArgumentError("Sizes of the input and ouput need to be the same."))
  c = g.λ * γ
  c = 1. / (1. + 2. * c)
  copy!(out_x, x)
  scale!(out_x, c)
end

##########################################################
####### nuclear norm  g(x) = λ * ||x||_*
###########################################################

struct ProxNuclear{T<:AbstractFloat} <: ProximableFunction
  λ::T
end
value(g::ProxNuclear, x::StridedMatrix) = g.λ * sum(svdvals(x))
function prox!{T<:AbstractFloat}(g::ProxNuclear{T}, out_x::StridedMatrix{T}, x::StridedMatrix{T}, γ::T)
  size(out_x) == size(x) || throw(ArgumentError("Sizes of the input and ouput need to be the same."))
  U, S, V = svd(x)
  c = g.λ * γ
  S .= shrink.(S, c)
  scale!(U, S)
  A_mul_Bt!(out_x, U, V)
end

##################################
#
#  sum_k g(x_k)
#
##################################

# λ0⋅sum_k λ_k g(x_k)
struct AProxL2{T<:AbstractFloat} <: ProximableFunction
  λ0::T
  λ::Vector{T}
end
function value{T<:AbstractFloat}(g::AProxL2{T}, x::AtomIterate{T})
  v = zero(T)
  @inbounds for i=1:length(x.atoms)
    v += vecnorm(x.atoms[i]) * g.λ[i]
  end
  v * g.λ0
end
function prox!(g::AProxL2{T}, out::AtomIterate{T}, x::AtomIterate{T}, γ::T) where {T<:AbstractFloat}
  for i=1:length(x.atoms)
    shrinkL2!(out.atoms[i], x.atoms[i], γ*g.λ[i]*g.λ0)
  end
  out
end
cdprox!(g::AProxL2{T}, x::AtomIterate{T}, k::Int, γ::T) where {T<:AbstractFloat} =
  shrinkL2!(x.atoms[k], x.atoms[k], g.λ[k] * γ * g.λ0)



##########################################################
# Gaussian likelihood prox
# f(X) = tr(SX) - log deg(X)
##########################################################

struct ProxGaussLikelihood{T} <: ProximableFunction
  S::Symmetric{T}
  tmp::Matrix{T}
end
ProxGaussLikelihood(S::Symmetric) = ProxGaussLikelihood{eltype(S)}(S, zeros(eltype(S),size(S)))

value(g::ProxGaussLikelihood{T}, X::StridedMatrix{T}) where {T<:AbstractFloat} = trace(g.S*X) - logdet(X)
function prox!(g::ProxGaussLikelihood{T}, hX::StridedMatrix{T}, X::StridedMatrix{T}, γ::T) where {T<:AbstractFloat}
  S = g.S
  tmp = g.tmp

  ρ = one(T) / γ
  @. tmp = X * ρ - S
  ef = eigfact!(Symmetric(tmp))
  d = getindex(ef, :values)::Vector{T}
  U = getindex(ef, :vectors)::Matrix{T}
  @inbounds @simd for i in eachindex(d)
    t = d[i]
    d[i] = sqrt( (t + sqrt(t^2. + 4.*ρ)) / (2.*ρ) )
  end
  scale!(U, d)
  A_mul_Bt!(hX, U, U)
end



















###
