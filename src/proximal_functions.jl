##########################################################
#
#  Proximal Operators
#
##########################################################

"""
    ProximableFunction

Abstract supertype for (possibly nonsmooth) functions `g` with an easy proximal
operator. Subtypes implement [`value`](@ref) and [`prox!`](@ref), and
[`cdprox!`](@ref) if they support coordinate-wise updates.
"""
abstract type ProximableFunction end

"""
    prox!(g, out, x, γ = 1)

Compute the proximal operator `argmin_z g(z) + ‖z - x‖² / (2γ)`, store it in
`out`, and return `out`.
"""
function prox! end

"""
    prox(g, x, γ = 1)

Allocating version of [`prox!`](@ref).
"""
function prox end

"""
    cdprox!(g, x, k, γ)

Apply the proximal operator of `g` with step `γ` to coordinate `k` of `x` in
place, as in one step of coordinate descent. For [`AtomIterate`](@ref)s, `k`
indexes a group rather than a single entry. Coordinates are numbered
`1:numCoordinates(x)`.
"""
function cdprox! end

"""
    kkt_violation(g, grad, i)
    kkt_violation(g, grad, x, i)

How much coordinate `i` of a point `x` with `x[i] = 0` violates the optimality
condition `0 ∈ grad[i] + ∂ᵢg(x)` of `f + g`, where `grad = ∇f(x)`. A positive
value means that moving `x[i]` away from zero decreases the objective; a value
`≤ 0` means that `x[i] = 0` is optimal. For [`ProxL1`](@ref) it is
`|grad[i]| - λ0 λᵢ`, and for [`ProxZero`](@ref) it is `|grad[i]|`. For these
penalties the condition does not involve the rest of `x`, and the second form
returns the first.

For [`ProxGroupL2`](@ref), only the second form is defined, as the condition
depends on the rest of the group `Gₖ` of `i`. If `x[Gₖ] = 0`, it is the
condition `‖grad[Gₖ]‖₂ ≤ λ0 λₖ` of the whole group, and the value is
`‖grad[Gₖ]‖₂ - λ0 λₖ` for every `i` in `Gₖ`; a positive value means that moving
`x[Gₖ]` away from zero decreases the objective. Otherwise the condition is
`grad[i] = 0`, and the value is `|grad[i]|`, as for an index in no group.
"""
function kkt_violation end

prox!(g::ProximableFunction, hat_x::AbstractArray, x::AbstractArray) = prox!(g, hat_x, x, one(eltype(x)))
prox(g::ProximableFunction,                        x::AbstractArray, γ) = prox!(g, similar(x), x, γ)
prox(g::ProximableFunction,                        x::AbstractArray) = prox!(g, similar(x), x, one(eltype(x)))

# for a separable penalty, the optimality of x[i] = 0 does not depend on the rest of x
kkt_violation(g::ProximableFunction, grad::AbstractArray, ::AbstractArray, i::Int) = kkt_violation(g, grad, i)



##########################################################
######  zero
##########################################################

"""
    ProxZero()

The zero function `g(x) = 0`. Its proximal operator is the identity.
"""
struct ProxZero <: ProximableFunction end

value(g::ProxZero, x::AbstractVecOrMat{T}) where {T} = zero(T)

function prox!(::ProxZero, out_x::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, γ::T) where {T}
  size(out_x) == size(x) || throw(ArgumentError("Sizes of the input and ouput need to be the same."))
  copyto!(out_x, x)
end

cdprox!(g::ProxZero, x::Union{SparseIterate{T},SymmetricSparseIterate{T}}, k::Int, γ::T) where {T} = x[k]

restrict(g::ProxZero, ::AbstractVector{<:Integer}) = g
kkt_violation(::ProxZero, grad::AbstractArray, i::Int) = abs(grad[i])

##########################################################
######  L1 norm  g(x) = λ0 ⋅ \sum_j λ_j |x_j|
##########################################################

"""
    ProxL1(λ0)
    ProxL1(λ0, λ)

The weighted ℓ1 norm `g(x) = λ0 ⋅ Σⱼ λⱼ |xⱼ|`, with all `λⱼ = 1` when `λ` is
omitted. Its proximal operator is coordinate-wise soft-thresholding
([`shrink`](@ref)).
"""
struct ProxL1{T<:AbstractFloat, S} <: ProximableFunction
  λ0::T
  λ::S

  ProxL1{T, S}(λ0::T, λ::Union{Nothing, AbstractArray{T}}) where {T <: AbstractFloat, S} = new(λ0, λ)
end

ProxL1(λ0::T        ) where {T <: AbstractFloat} = ProxL1{T, Nothing}(λ0, nothing)
ProxL1(λ0::T, ::Nothing) where {T <: AbstractFloat} = ProxL1{T, Nothing}(λ0, nothing)
ProxL1(λ0::T, λ::AbstractArray{T}) where {T <: AbstractFloat} = ProxL1{T, typeof(λ)}(λ0, λ)

function value(g::ProxL1{T, S}, x::AbstractArray{T}) where S <: AbstractArray{T} where {T <: AbstractFloat}
  size(g.λ) == size(x) || throw(DimensionMismatch("Sizes of g.λ and x need to be the same"))
  v = zero(T)
  @inbounds @simd for i in eachindex(x)
    v += abs(x[i]) * g.λ[i]
  end
  v * g.λ0
end
prox!(g::ProxL1{T, S}, out_x::AbstractArray{T}, x::AbstractArray{T}, γ::T) where S <: AbstractArray{T} where {T <: AbstractFloat} =
    out_x .= shrink.(x, (g.λ0 * γ) .* g.λ)
@inline function cdprox!(g::ProxL1{T, S}, x::Union{SparseIterate{T},SymmetricSparseIterate{T}}, k::Int, γ::T) where S <: AbstractArray{T} where {T <: AbstractFloat}
  length(g.λ) == numCoordinates(x) || throw(DimensionMismatch())
  @boundscheck checkbounds(x, k)
  x[k] = shrink(x[k], g.λ[k] * γ * g.λ0)
end

value(g::ProxL1{T, Nothing}, x::AbstractArray{T}) where {T<:AbstractFloat} = g.λ0 * sum(abs, x)
prox!(g::ProxL1{T, Nothing}, out_x::AbstractArray{T}, x::AbstractArray{T}, γ::T) where {T<:AbstractFloat} =
  out_x .= shrink.(x, γ * g.λ0)
cdprox!(g::ProxL1{T, Nothing}, x::Union{SparseIterate{T},SymmetricSparseIterate{T}}, k::Int, γ::T) where {T} =
  x[k] = shrink(x[k], g.λ0 * γ)

restrict(g::ProxL1{T, Nothing}, ::AbstractVector{<:Integer}) where {T} = g
restrict(g::ProxL1{T, S}, I::AbstractVector{<:Integer}) where S <: AbstractArray{T} where {T} = ProxL1(g.λ0, g.λ[I])

kkt_violation(g::ProxL1{T, Nothing}, grad::AbstractArray, i::Int) where {T} = abs(grad[i]) - g.λ0
kkt_violation(g::ProxL1{T, S}, grad::AbstractArray, i::Int) where S <: AbstractArray{T} where {T} = abs(grad[i]) - g.λ0 * g.λ[i]

##########################################################
###### L1 + Fused  g(x1,x2) = λ1*(|x1|+|x2|) + λ2*|x1-x2|
##########################################################

"""
    ProxL1Fused(λ1, λ2)

The penalty `g(x1, x2) = λ1 (|x1| + |x2|) + λ2 |x1 - x2|`, applied entry-wise to
a pair of vectors `(x1, x2)`. See [`proxL1Fused`](@ref).
"""
struct ProxL1Fused{T<:AbstractFloat} <: ProximableFunction
  λ1::T
  λ2::T
end

value(g::ProxL1Fused{T}, x1::T, x2::T) where {T<:AbstractFloat} = g.λ1*(abs(x1)+abs(x2)) + g.λ2*abs(x1-x2)
function prox!(g::ProxL1Fused{T}, out_x::Tuple{StridedVector{T},StridedVector{T}}, x::Tuple{StridedVector{T},StridedVector{T}}, γ::T) where {T<:AbstractFloat}
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

"""
    ProxL2(λ0)

The Euclidean norm `g(x) = λ0 ‖x‖₂`. Its proximal operator shrinks `x` towards
zero ([`shrinkL2!`](@ref)). For the group lasso, a sum of such norms over groups
of coordinates, use [`ProxGroupL2`](@ref).
"""
struct ProxL2{T<:AbstractFloat} <: ProximableFunction
  λ0::T
end

value(g::ProxL2{T}, x::AbstractArray{T}) where {T<:AbstractFloat} = g.λ0 * norm(x)
prox!(g::ProxL2{T}, out_x::AbstractArray{T}, x::AbstractArray{T}, γ::T) where {T<:AbstractFloat} =
  shrinkL2!(out_x, x, g.λ0*γ)


##########################################################
###### group L2 norm  g(x) = λ0 ⋅ Σₖ λₖ ‖x[Gₖ]‖₂
##########################################################

"""
    ProxGroupL2(λ0, groups)
    ProxGroupL2(λ0, groups, λ)

The group-lasso penalty `g(x) = λ0 ⋅ Σₖ λₖ ‖x[Gₖ]‖₂` over disjoint groups
`Gₖ = groups[k]` of (linear) indices of `x`, of any sizes, such as `1:5` or
`[2, 7, 9]`. The weights default to `λₖ = √|Gₖ|`, so that groups of one index
give `ProxL1(λ0)`. Indices in no group are not penalized. The proximal operator
shrinks each group towards zero in Euclidean norm, as [`shrinkL2!`](@ref) does.

It has no [`cdprox!`](@ref) on ordinary arrays, as its groups do not split into
coordinates; active-set solvers can use [`restrict`](@ref) and
[`kkt_violation`](@ref).

# Example
```julia
g = ProxGroupL2(0.5, [1:3, 4:5, [6, 8]])    # x[7] is not penalized
prox(g, randn(8))
```
"""
struct ProxGroupL2{T<:AbstractFloat, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: ProximableFunction
  λ0::T
  groups::G
  λ::Vector{T}
  group_of::Vector{Int}   # group_of[i] = k for i ∈ Gₖ, 0 for an index in no group

  function ProxGroupL2{T, G}(λ0::T, groups::G, λ::Vector{T}) where {T<:AbstractFloat, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    length(λ) == length(groups) || throw(DimensionMismatch("There must be one weight per group."))
    λ0 >= 0 && all(>=(0), λ) || throw(ArgumentError("λ0 and the weights must be nonnegative."))
    new(λ0, groups, λ, _group_of(groups))
  end
end

ProxGroupL2(λ0::T, groups::AbstractVector{<:AbstractVector{<:Integer}}) where {T<:AbstractFloat} =
  ProxGroupL2(λ0, groups, T[sqrt(T(length(G))) for G in groups])
function ProxGroupL2(λ0::T, groups::AbstractVector{<:AbstractVector{<:Integer}}, λ::AbstractVector{T}) where {T<:AbstractFloat}
  # groups of mixed types, such as [1:3, [4, 7]], are stored as Vector{Int}s, so
  # that value and prox! do not dispatch on each group
  G = isconcretetype(eltype(groups)) ? groups : [Vector{Int}(Gk) for Gk in groups]
  ProxGroupL2{T, typeof(G)}(λ0, G, convert(Vector{T}, λ))
end

# group_of[i] = k for i ∈ groups[k], after checking that the groups are nonempty,
# disjoint sets of positive indices
function _group_of(groups)
  n = 0
  for G in groups
    isempty(G) && throw(ArgumentError("Groups must not be empty."))
    minimum(G) >= 1 || throw(ArgumentError("Group indices must be positive."))
    n = max(n, maximum(G))
  end
  group_of = zeros(Int, n)
  for k in eachindex(groups), i in groups[k]
    group_of[i] == 0 || throw(ArgumentError("Groups must be disjoint, but index $i is in groups $(group_of[i]) and $k."))
    group_of[i] = k
  end
  group_of
end

# the group of index i, 0 if i is in no group
_group(g::ProxGroupL2, i::Integer) = i <= length(g.group_of) ? g.group_of[i] : 0

function value(g::ProxGroupL2{T}, x::AbstractArray{T}) where {T<:AbstractFloat}
  v = zero(T)
  for k in eachindex(g.groups)
    v += g.λ[k] * norm(view(x, g.groups[k]))
  end
  v * g.λ0
end

function prox!(g::ProxGroupL2{T}, out_x::AbstractArray{T}, x::AbstractArray{T}, γ::T) where {T<:AbstractFloat}
  size(out_x) == size(x) || throw(DimensionMismatch("Sizes of the input and output need to be the same."))
  out_x === x || copyto!(out_x, x)      # indices in no group are not penalized
  for k in eachindex(g.groups)
    _shrink_group!(out_x, g.groups[k], γ * g.λ0 * g.λ[k])
  end
  out_x
end

# x[G] ← max(1 - c / ‖x[G]‖₂, 0) ⋅ x[G] in place; shrinkL2! on a view would
# allocate, as copyto! guards against the aliased source
function _shrink_group!(x::AbstractArray{T}, G, c::T) where {T<:AbstractFloat}
  nx = norm(view(x, G))
  if nx > c
    t = one(T) - c / nx
    for i in G
      x[i] *= t
    end
  else
    for i in G
      x[i] = zero(T)
    end
  end
  x
end

# the parts of the groups in S, as positions in S, with their weights
function restrict(g::ProxGroupL2{T}, S::AbstractVector{<:Integer}) where {T}
  members = Dict{Int, Vector{Int}}()
  for (j, i) in enumerate(S)
    k = _group(g, i)
    k == 0 || push!(get!(Vector{Int}, members, k), j)
  end
  ks = sort!(collect(keys(members)))
  ProxGroupL2(g.λ0, [members[k] for k in ks], g.λ[ks])
end

function kkt_violation(g::ProxGroupL2, grad::AbstractArray, x::AbstractArray, i::Int)
  k = _group(g, i)
  k == 0 && return abs(grad[i])
  G = g.groups[k]
  all(iszero, view(x, G)) ? norm(view(grad, G)) - g.λ0 * g.λ[k] : abs(grad[i])
end


##########################################################
###### L2 norm squared g(x) = λ * ||x||_2^2
##########################################################

"""
    ProxL2Sq(λ)

The squared Euclidean norm `g(x) = λ ‖x‖₂²` (ridge penalty).
"""
struct ProxL2Sq{T<:AbstractFloat} <: ProximableFunction
  λ::T
end
value(g::ProxL2Sq, x::StridedArray) = g.λ * sum(abs2, x)
function prox!(g::ProxL2Sq{T}, out_x::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, γ::T) where {T<:AbstractFloat}
  size(out_x) == size(x) || throw(ArgumentError("Sizes of the input and ouput need to be the same."))
  c = g.λ * γ
  c = 1. / (1. + 2. * c)
  out_x .= c .* x
end

##########################################################
####### nuclear norm  g(x) = λ * ||x||_*
###########################################################

"""
    ProxNuclear(λ)

The nuclear norm `g(X) = λ ‖X‖_*`, the sum of the singular values of `X`. Its
proximal operator soft-thresholds the singular values.
"""
struct ProxNuclear{T<:AbstractFloat} <: ProximableFunction
  λ::T
end
value(g::ProxNuclear, x::StridedMatrix) = g.λ * sum(svdvals(x))
function prox!(g::ProxNuclear{T}, out_x::StridedMatrix{T}, x::StridedMatrix{T}, γ::T) where {T<:AbstractFloat}
  size(out_x) == size(x) || throw(ArgumentError("Sizes of the input and ouput need to be the same."))
  U, S, V = svd(x)
  c = g.λ * γ
  S .= shrink.(S, c)
  rmul!(U, Diagonal(S))
  mul!(out_x, U, transpose(V))
end

##################################
#
#  sum_k g(x_k)
#
##################################



##########################################################
# Gaussian likelihood prox
# f(X) = tr(SX) - log det(X)
##########################################################

"""
    ProxGaussLikelihood(S)

The Gaussian negative log-likelihood `g(X) = tr(SX) - log det X` for a
precision matrix `X`, where `S` is the (symmetric) sample covariance. The
proximal operator has a closed form via an eigendecomposition.
"""
struct ProxGaussLikelihood{T} <: ProximableFunction
  S::Symmetric{T}
  tmp::Matrix{T}
end
ProxGaussLikelihood(S::Symmetric) = ProxGaussLikelihood{eltype(S)}(S, zeros(eltype(S),size(S)))

# tr(SX) = dot(S, X) for symmetric S, without forming the product SX
value(g::ProxGaussLikelihood{T}, X::StridedMatrix{T}) where {T<:AbstractFloat} = dot(g.S, X) - logdet(X)
function prox!(g::ProxGaussLikelihood{T}, hX::StridedMatrix{T}, X::StridedMatrix{T}, γ::T) where {T<:AbstractFloat}
  S = g.S
  tmp = g.tmp

  ρ = one(T) / γ
  @. tmp = X * ρ - S
  d, U = _eigen_sym!(tmp)::Tuple{Vector{T}, Matrix{T}}
  @inbounds @simd for i in eachindex(d)
    t = d[i]
    d[i] = sqrt( (t + sqrt(t^2. + 4. * ρ)) / (2. * ρ) )
  end
  rmul!(U, Diagonal(d))
  mul!(hX, U, transpose(U))
end

# Eigenvalues and eigenvectors of the symmetric matrix stored in the upper triangle
# of A, overwriting A. LAPACK's divide-and-conquer driver (syevd) is faster than the
# one eigen! uses (syevr) for all but small matrices.
_eigen_sym!(A::StridedMatrix{<:LinearAlgebra.BlasReal}) = LAPACK.syevd!('V', 'U', A)
_eigen_sym!(A::AbstractMatrix) = (ef = eigen!(Symmetric(A)); (ef.values, ef.vectors))



















###
