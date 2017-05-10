##########################################################
#
#  Proximal Operators
#
##########################################################

abstract type ProximableFunction end

prox!(g::ProximableFunction, hat_x::StridedArray, x::StridedArray) = prox!(g, hat_x, x, 1.0)
prox{T<:AbstractFloat}(g::ProximableFunction, x::StridedArray, γ::T) = prox!(g, similar(x), x, γ)
prox(g::ProximableFunction, x::StridedArray) = prox!(g, similar(x), x, 1.0)



##########################################################
######  zero
##########################################################

struct ProxZero <: ProximableFunction end

value{T<:AbstractFloat}(g::ProxZero, x::StridedArray{T}) = zero(T)

function prox!{T<:AbstractFloat}(::ProxZero, out_x::StridedArray{T}, x::StridedArray{T}, γ::T)
  @assert size(out_x) == size(x)
  copy!(out_x, x)
  out_x
end

##########################################################
######  L1 norm  g(x) = λ * ||x||_1
##########################################################

struct AProxL1{T<:AbstractFloat, N} <: ProximableFunction
  λ::Array{T, N}
end
VProxL1{T<:AbstractFloat} = AProxL1{T, 1}
MProxL1{T<:AbstractFloat} = AProxL1{T, 2}

function value{T<:AbstractFloat, N}(g::AProxL1{T, N}, x::StridedArray{T, N})
  λ = g.λ
  @assert size(x) == size(λ)
  v = zero(T)
  @inbounds @simd for i in eachindex(x)
    v += abs(x[i]) * λ[i]
  end
  v
end

prox!{T<:AbstractFloat}(g::AProxL1{T}, out_x::StridedArray{T}, x::StridedArray{T}, γ::T) =
    out_x .= shrink.(x, γ * g.λ)

struct ProxL1{T<:AbstractFloat} <: ProximableFunction
  λ::T
end

value{T<:AbstractFloat}(g::ProxL1{T}, x::StridedArray{T}) = g.λ * sum(abs, x)
function prox!{T<:AbstractFloat}(g::ProxL1{T}, out_x::StridedArray{T}, x::StridedArray{T}, γ::T)
  @assert size(out_x) == size(x)
  c = g.λ * γ
  @inbounds @simd for i in eachindex(x)
    out_x[i] = shrink(x[i], c)
  end
  out_x
end

# function value{T<:AbstractFloat, N}(g::AProxL1{T, N}, x::StridedArray{T, N}, activeset::ActiveSet)
#   λ = g.λ
#   @assert size(x) == size(λ)
#   v = zero(T)
#   indexes = activeset.indexes
#   @inbounds for i=1:activeset.numActive
#     t = indexes[i]
#     v += λ[t]*abs(x[t])
#   end
#   v
# end
# function prox!{T<:AbstractFloat}(
#     g::AProxL1{T},
#     out_x::StridedArray{T},
#     x::StridedArray{T},
#     γ::T,
#     activeset::ActiveSet)
#   λ = g.λ
#   @assert size(out_x) == size(x) == size(λ)
#   indexes = activeset.indexes
#   @inbounds for i=1:activeset.numActive
#     t = indexes[i]
#     c = λ[t]*γ
#     out_x[t] = shrink(x[t], c)
#   end
#   out_x
# end




# function value{T<:AbstractFloat}(g::ProxL1{T}, x::StridedArray{T}, activeset::ActiveSet)
#   r = zero(T)
#   indexes = activeset.indexes
#   @inbounds for i=1:activeset.numActive
#     t = indexes[i]
#     r += abs(x[t])
#   end
#   g.λ * r
# end
# function prox!{T<:AbstractFloat}(g::ProxL1{T}, out_x::StridedArray{T}, x::StridedArray{T}, γ::T, activeset::ActiveSet)
#   @assert size(out_x) == size(x)
#   c = g.λ * γ
#   indexes = activeset.indexes
#   @inbounds for i=1:activeset.numActive
#     t = indexes[i]
#     out_x[t] = shrink(x[t], c)
#   end
#   out_x
# end
# function active_set{T<:AbstractFloat}(
#     ::Union{ProxL1{T}, AProxL1{T}},
#     x::StridedArray{T};
#     zero_thr::T=1e-4
#     )
#   numElem = length(x)
#   activeset = [1:numElem;]
#   numActive = 0
#   for j = 1:numElem
#     if abs(x[j]) >= zero_thr
#       numActive += 1
#       activeset[numActive], activeset[j] = activeset[j], activeset[numActive]
#     end
#   end
#   ActiveSet(activeset, numActive)
# end
#
#
# # active set of coordinates
# # prox function
# # differentiable function
# # current iterate
# # tmp array
# function add_violator!{T<:AbstractFloat}(
#     activeset::ActiveSet,
#     x::StridedArray{T},
#     g::ProxL1{T},
#     f::DifferentiableFunction,
#     tmp::StridedArray{T};
#     zero_thr::T=1e-4,
#     grad_tol::T=1e-6
#     )
#   λ = g.λ
#   changed = false
#
#   numElem = length(x)
#   numActive = activeset.numActive
#   indexes = activeset.indexes
#   # check for things to be removed from the active set
#   i = 0
#   @inbounds while i < numActive
#     i = i + 1
#     t = indexes[i]
#     if abs(x[t]) < zero_thr
#       x[t] = zero(T)
#       changed = true
#       indexes[numActive], indexes[i] = indexes[i], indexes[numActive]
#       numActive -= 1
#       i = i - 1
#     end
#   end
#
#   value_and_gradient!(f, tmp, x)
#   I = 0
#   V = zero(T)
#   @inbounds for i=numActive+1:numElem
#     t = indexes[i]
#     nV = abs(tmp[t])
#     if V < nV
#       I = i
#       V = nV
#     end
#   end
#   if I > 0 && V + grad_tol > λ
#     changed = true
#     numActive += 1
#     indexes[numActive], indexes[I] = indexes[I], indexes[numActive]
#   end
#   activeset.numActive = numActive
#   changed
# end
#
# function add_violator!{T<:AbstractFloat}(
#     activeset::ActiveSet,
#     x::StridedArray{T},
#     g::AProxL1{T},
#     f::DifferentiableFunction,
#     tmp::StridedArray{T};
#     zero_thr::T=1e-4,
#     grad_tol::T=1e-4
#     )
#   λ = g.λ
#   @assert size(λ) == size(x)
#   changed = false
#
#   numElem = length(x)
#   numActive = activeset.numActive
#   indexes = activeset.indexes
#   # check for things to be removed from the active set
#   i = 0
#   @inbounds while i < numActive
#     i = i + 1
#     t = indexes[i]
#     if abs(x[t]) < zero_thr
#       x[t] = zero(T)
#       changed = true
#       indexes[numActive], indexes[i] = indexes[i], indexes[numActive]
#       numActive -= 1
#       i = i - 1
#     end
#   end
#
#   value_and_gradient!(f, tmp, x)
#   I = 0
#   V = zero(T)
#   @inbounds for i=numActive+1:numElem
#     t = indexes[i]
#     nV = abs(tmp[t]) - λ[i]
#     if V < nV
#       I = i
#       V = nV
#     end
#   end
#   if I > 0 && V + grad_tol > zero(T)
#     changed = true
#     numActive += 1
#     indexes[numActive], indexes[I] = indexes[I], indexes[numActive]
#   end
#   activeset.numActive = numActive
#   changed
# end

##########################################################
###### L2 norm   g(x) = λ * ||x||_2
##########################################################

struct ProxL2{T<:AbstractFloat} <: ProximableFunction
  λ::T
end

value{T<:AbstractFloat}(g::ProxL2{T}, x::StridedVector{T}) = g.λ * norm(x)
function prox!{T<:AbstractFloat}(g::ProxL2{T}, out_x::StridedVector{T}, x::StridedVector{T}, γ::T)
  @assert size(out_x) == size(x)
  tmp = max(one(T) - g.λ * γ / norm(x), zero(T))
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
value{T<:AbstractFloat}(g::ProxL2Sq{T}, x::StridedArray{T}) = g.λ * sum(abs2, x)
function prox!{T<:AbstractFloat}(g::ProxL2Sq{T}, out_x::StridedArray{T}, x::StridedArray{T}, γ::T)
  @assert size(out_x) == size(x)
  c = g.λ * γ
  c = 1. / (1. + 2. * c)
  @inbounds @simd for i in eachindex(x)
    out_x[i] = c * x[i]
  end
  out_x
end

##########################################################
####### nuclear norm  g(x) = λ * ||x||_*
###########################################################

struct ProxNuclear{T<:AbstractFloat} <: ProximableFunction
  λ::T
end
value{T<:AbstractFloat}(g::ProxNuclear{T}, x::StridedMatrix{T}) = g.λ * sum(svdvals(x))

function prox!{T<:AbstractFloat}(g::ProxNuclear{T}, out_x::StridedMatrix{T}, x::StridedMatrix{T}, γ::T)
  @assert size(out_x) == size(x)
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

struct ProxGaussLikelihood{T,M<:AbstractMatrix} <: ProximableFunction
  S::M
  tmpStorage::Matrix{T}

  ProxGaussLikelihood{T,M}(S::AbstractMatrix{T}, tmpStorage::Matrix{T}) where {T,M} = new(S, tmpStorage)
end

ProxGaussLikelihood(S::AbstractMatrix{T}) where {T} = ProxGaussLikelihood{T, Array{T, 2}}(S, zeros(T, size(S)))


value{T<:AbstractFloat}(g::ProxGaussLikelihood{T}, x::StridedMatrix{T}) = trace(g.S*x) - logdet(x)
function prox!{T<:AbstractFloat}(g::ProxGaussLikelihood{T}, out_x::StridedMatrix{T}, x::StridedMatrix{T}, γ::T)
  S = g.S
  @assert size(out_x) == size(x) == size(S)

  tmpStorage = g.tmpStorage
  ρ = one(T) / γ
  @. tmpStorage = ρ * x - S

  ef = eigfact!(tmpStorage)::Base.LinAlg.Eigen{T, T}
  d = ef[:values]::Vector{T}
  U = ef[:vectors]::Matrix{T}
  @inbounds @simd for i in eachindex(d)
    t = d[i]
    d[i] = sqrt( (t + sqrt(t^2. + 4.*ρ)) / (2.*ρ) )::T
  end
  scale!(U, d)
  A_mul_Bt!(out_x, U, U)::StridedMatrix{T}
end



















###
