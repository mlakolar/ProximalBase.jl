##########################################################
#
#  Proximal Operators
#
##########################################################

abstract type ProximableFunction end

prox!(g::ProximableFunction, hat_x::AbstractVecOrMat, x::AbstractVecOrMat) = prox!(g, hat_x, x, 1.0)
prox(g::ProximableFunction, x::AbstractVecOrMat, γ::Real) = prox!(g, similar(x), x, γ)
prox(g::ProximableFunction, x::AbstractVecOrMat) = prox!(g, similar(x), x, 1.0)



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
######  L1 norm  g(x) = λ * ||x||_1
##########################################################

struct AProxL1{T<:AbstractFloat, N} <: ProximableFunction
  λ::Array{T, N}
end

function value(g::AProxL1{T}, x::AbstractVecOrMat{T}) where {T}
  size(g.λ) == size(x) || throw(ArgumentError("Sizes of g.λ and x need to be the same"))
  λ = g.λ
  v = zero(T)
  @inbounds @simd for i in eachindex(x)
    v += abs(x[i]) * λ[i]
  end
  v
end
prox!{T<:AbstractFloat}(g::AProxL1{T}, out_x::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, γ::T) =
    out_x .= shrink.(x, γ * g.λ)
function cdprox!(g::AProxL1{T}, x::SparseIterate{T}, k::Int, γ::T) where {T}
  size(g.λ) == size(x) || throw(DimensionMismatch())
  x[k] = shrink(x[k], g.λ[k] * γ)
end


struct ProxL1{T<:AbstractFloat} <: ProximableFunction
  λ::T
end

value{T<:AbstractFloat}(g::ProxL1{T}, x::StridedArray{T}) = g.λ * sum(abs, x)
prox!{T<:AbstractFloat}(g::ProxL1{T}, out_x::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, γ::T) =
  out_x .= shrink.(x, γ * g.λ)
cdprox!(g::ProxL1{T}, x::SparseIterate{T}, k::Int, γ::T) where {T} =
  x[k] = shrink(x[k], g.λ * γ)

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

value{T<:AbstractFloat}(g::ProxL2{T}, x::StridedVector{T}) = g.λ * norm(x)
function prox!{T<:AbstractFloat}(g::ProxL2{T}, out_x::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, γ::T)
  size(out_x) == size(x) || throw(ArgumentError("Sizes of the input and ouput need to be the same."))
  tmp = max(one(T) - g.λ * γ / vecnorm(x), zero(T))
  if tmp > zero(T)
    out_x .= tmp .* x
  else
    out_x .= zero(T)
  end
  out_x
end

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

# ###### sum_k g(x_k)
#
# immutable ProxSumProx{P<:ProximableFunction, I} <: ProximableFunction
#   intern_prox::P
#   groups::Vector{I}
# end
#
# ProxL1L2{T<:AbstractFloat, I}(λ::T, groups::Vector{I}) = ProxSumProx{ProxL2{T}, I}(ProxL2{T}(λ), groups)
# ProxL1Nuclear{T<:AbstractFloat, I}(λ::T, groups::Vector{I}) = ProxSumProx{ProxNuclear{T}, I}(ProxNuclear{T}(λ), groups)
#
# function value{T<:AbstractFloat}(g::ProxSumProx, x::StridedArray{T})
#   intern_prox = g.intern_prox
#   groups = g.groups
#   v = zero(T)
#   for i in eachindex(groups)
#     v += value(intern_prox, sub(x, groups[i]))
#   end
#   v
# end
#
# function prox!{T<:AbstractFloat}(g::ProxSumProx, out_x::StridedArray{T}, x::StridedArray{T}, γ::T)
#   @assert size(out_x) == size(x)
#   intern_prox = g.intern_prox
#   groups = g.groups
#   for i in eachindex(groups)
#     prox!(intern_prox, sub(out_x, groups[i]), sub(x, groups[i]), γ)
#   end
#   out_x
# end
#
# immutable AProxSumProx{P<:ProximableFunction, I} <: ProximableFunction
#   intern_prox::Vector{P}
#   groups::Vector{I}
# end
#
# function ProxL1L2{T<:AbstractFloat, I}(
#     λ::Vector{T},
#     groups::Vector{I}
#     )
#   numGroups = length(groups)
#   @assert length(λ) == numGroups
#   proxV = Array(ProxL2{T}, numGroups)
#   @inbounds for i=1:numGroups
#     proxV[i] = ProxL2{T}(λ[i])
#   end
#   AProxSumProx{ProxL2{T}, I}(proxV, groups)
# end
#
# function value{T<:AbstractFloat}(g::AProxSumProx, x::StridedArray{T})
#   intern_prox = g.intern_prox
#   groups = g.groups
#   v = zero(T)
#   @inbounds for i in eachindex(groups)
#     v += value(intern_prox[i], sub(x, groups[i]))
#   end
#   v
# end
#
# function prox!{T<:AbstractFloat}(g::AProxSumProx, out_x::StridedArray{T}, x::StridedArray{T}, γ::T)
#   @assert size(out_x) == size(x)
#   intern_prox = g.intern_prox
#   groups = g.groups
#   @inbounds for i in eachindex(groups)
#     prox!(intern_prox[i], sub(out_x, groups[i]), sub(x, groups[i]), γ)
#   end
#   out_x
# end
#
# #
# function active_set{T<:AbstractFloat, I}(g::ProxSumProx{ProxNuclear{T}, I}, x::StridedArray{T}; zero_thr::T=1e-4)
#   groups = g.groups
#   numElem = length(groups)
#   activeset = [1:numElem;]
#   numActive = 0
#   for j = 1:numElem
#     if vecnorm(sub(x, groups[j])) > zero_thr
#       numActive += 1
#       activeset[numActive], activeset[j] = activeset[j], activeset[numActive]
#     end
#   end
#   GroupActiveSet(activeset, numActive, groups)
# end
# function value{T<:AbstractFloat, I}(g::ProxSumProx{ProxNuclear{T}, I}, x::StridedArray{T}, activeset::GroupActiveSet)
#   v = zero(T)
#   intern_prox = g.intern_prox
#   groups = g.groups
#   activeGroups = activeset.groups
#   @inbounds for i=1:activeset.numActive
#     ind = activeGroups[i]
#     v += value(intern_prox, sub(x, groups[ind]))
#   end
#   v
# end
# function prox!{T<:AbstractFloat, I}(g::ProxSumProx{ProxNuclear{T}, I}, out_x::StridedArray{T}, x::StridedArray{T}, γ::T, activeset::GroupActiveSet)
#   @assert size(out_x) == size(x)
#   intern_prox = g.intern_prox
#   groups = g.groups
#   activeGroups = activeset.groups
#   @inbounds for i=1:activeset.numActive
#     ind = activeGroups[i]
#     prox!(intern_prox, sub(out_x, groups[ind]), sub(x, groups[ind]), γ)
#   end
#   out_x
# end
# function add_violator!{T<:AbstractFloat, II}(
#     activeset::GroupActiveSet, x::StridedArray{T},
#     g::ProxSumProx{ProxNuclear{T}, II}, f::DifferentiableFunction, tmp::StridedArray{T}; zero_thr::T=1e-4, grad_tol=1e-6
#     )
#   λ = g.intern_prox.λ
#   groups = g.groups
#   numElem = length(groups)
#   changed = false
#
#   numActive = activeset.numActive
#   activeGroups = activeset.groups
#   # check for things to be removed from the active set
#   i = 0
#   while i < numActive
#     i = i + 1
#     ind = activeGroups[i]
#     xt = sub(x, groups[ind])
#     if vecnorm(xt) < zero_thr
#       fill!(xt, zero(T))
#       changed = true
#       activeGroups[numActive], activeGroups[i] = activeGroups[i], activeGroups[numActive]
#       numActive -= 1
#       i = i - 1
#     end
#   end
#
#   gradient!(f, tmp, x)
#   I = 0
#   V = zero(T)
#   for i=numActive+1:numElem
#     ind = activeGroups[i]
#     gxt = sub(tmp, groups[ind])
#     nV = sqrt(eigmax(gxt'*gxt))
#     if V < nV
#       I = i
#       V = nV
#     end
#   end
#   if I > 0 && V + grad_tol > λ
#     changed = true
#     numActive += 1
#     activeGroups[numActive], activeGroups[I] = activeGroups[I], activeGroups[numActive]
#   end
#   activeset.numActive = numActive
#   changed
# end


##########################################################
# Gaussian likelihood prox
# f(X) = tr(SX) - log deg(X)
##########################################################

struct ProxGaussLikelihood{T,M<:StridedMatrix} <: ProximableFunction
  S::M
  tmp::Matrix{T}
end
ProxGaussLikelihood(S::StridedMatrix) = ProxGaussLikelihood{eltype(S), typeof(S)}(S, zeros(eltype(S),size(S)))

value{T<:AbstractFloat}(g::ProxGaussLikelihood{T}, X::StridedMatrix{T}) = trace(g.S*X) - logdet(X)
function prox!{T<:AbstractFloat}(g::ProxGaussLikelihood{T}, hX::StridedMatrix{T}, X::StridedMatrix{T}, γ::T)
  S = g.S
  tmp = g.tmp

  ρ = one(T) / γ
  @. tmp = X * ρ -  S
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
