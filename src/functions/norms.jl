####################################
#
#  Norms that are not separable
#
####################################

####################################
#  ℓ2 norm
####################################

"""
    NormL2(λ = 1)

The Euclidean norm `f(x) = λ ‖x‖₂` (the Frobenius norm for a matrix). Its
proximal operator shrinks `x` towards zero: `max(1 - γλ/‖x‖, 0) x`. For the
group lasso, a sum of such norms over groups of entries, use
[`GroupNormL2`](@ref).
"""
struct NormL2{T<:Real} <: AbstractFunction
  λ::T

  NormL2{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ))
end
NormL2(λ::Real = 1) = NormL2{typeof(λ)}(λ)

isproximable(::Type{<:NormL2}) = true
isconvex(::Type{<:NormL2}) = true

(f::NormL2)(x::AbstractArray) = convert(float(eltype(x)), f.λ) * sqrt(_sumabs2(x))

function value_and_prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  c = convert(T, γ) * convert(T, f.λ)
  nx = sqrt(convert(T, _sumabs2(x)))
  if nx > c
    t = 1 - c / nx
    y .= t .* x
    return convert(T, f.λ) * (nx - c)
  else
    fill!(y, zero(eltype(y)))
    return zero(T)
  end
end
prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real) = (value_and_prox!(y, f, x, γ); y)

####################################
#  ℓ∞ norm
####################################

"""
    NormLinf(λ = 1)

The ℓ∞ norm `f(x) = λ maxᵢ |xᵢ|`. Its proximal operator clips the entries at a
level `τ`, `sign(xᵢ) min(|xᵢ|, τ)`: by Moreau's identity it is `x` minus the
projection onto the ℓ1 ball of radius `γλ`, which takes O(n) expected time
(Condat's algorithm).
"""
struct NormLinf{T<:Real} <: AbstractFunction
  λ::T
  scratch::Scratch

  NormLinf{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ, Scratch()))
end
NormLinf(λ::Real = 1) = NormLinf{typeof(λ)}(λ)

isproximable(::Type{<:NormLinf}) = true
isconvex(::Type{<:NormLinf}) = true

(f::NormLinf)(x::AbstractArray) = convert(float(eltype(x)), f.λ) * norm(x, Inf)

function value_and_prox!(y::AbstractArray, f::NormLinf, x::AbstractArray, γ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  c = convert(T, γ) * convert(T, f.λ)
  if norm(x, 1) <= c
    fill!(y, zero(eltype(y)))
    return zero(T)
  end
  τ = _simplex_threshold!(_buffer!(f.scratch, T, length(x)), x, eachindex(IndexLinear(), x), c, abs)
  y .= sign.(x) .* min.(abs.(x), τ)
  convert(T, f.λ) * τ
end
prox!(y::AbstractArray, f::NormLinf, x::AbstractArray, γ::Real) = (value_and_prox!(y, f, x, γ); y)

####################################
#  group norms  λ0 ⋅ Σₖ wₖ ‖x[Gₖ]‖
####################################

# GroupNormL2 and GroupNormLinf share their groups, weights and the map from an
# index to its group, and differ in the norm of a group, its dual norm and the
# shrinkage of a group.
abstract type AbstractGroupNorm <: AbstractFunction end

isproximable(::Type{<:AbstractGroupNorm}) = true
isconvex(::Type{<:AbstractGroupNorm}) = true

"""
    GroupNormL2(λ0, groups)
    GroupNormL2(λ0, groups, w)

The group-lasso penalty `f(x) = λ0 Σₖ wₖ ‖x[Gₖ]‖₂` over disjoint groups
`Gₖ = groups[k]` of (linear) indices of `x`, of any sizes, such as `1:5` or
`[2, 7, 9]`. The weights default to `wₖ = √|Gₖ|`, so that groups of one index
give `NormL1(λ0)`. Indices in no group are not penalized. The proximal operator
shrinks each group towards zero in Euclidean norm.

The rows of a `p × k` matrix are the groups `[i:p:p*k for i in 1:p]`, which
gives the multi-task group lasso with a matrix response in
[`LeastSquaresLoss`](@ref).

On an [`AtomIterate`](@ref), whose coordinates are groups,
[`prox_coordinate!`](@ref) updates a group and [`kkt_violation`](@ref) measures
one. `AtomIterate(x, f)` builds an iterate that shares the groups of `f`, which
must then partition `1:length(x)` (an index in no group goes in a group of its
own with weight 0); with shared groups, the value and the proximal operator
touch only the active groups. On ordinary arrays, active-set solvers can use
[`restrict`](@ref) and [`kkt_violation`](@ref).

# Example
```julia
f = GroupNormL2(0.5, [1:3, 4:5, [6, 8]])    # x[7] is not penalized
prox(f, randn(8))
```
"""
struct GroupNormL2{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractGroupNorm
  λ0::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}   # group_of[i] = k for i ∈ Gₖ, 0 for an index in no group
  free::Vector{Int}       # the indices up to the largest in a group that are in none

  function GroupNormL2{T, G}(λ0::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    length(w) == length(groups) || throw(DimensionMismatch("There must be one weight per group."))
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    group_of = _group_of(groups)
    new{T, G}(λ0, groups, w, group_of, findall(iszero, group_of))
  end
end

"""
    GroupNormLinf(λ0, groups)
    GroupNormLinf(λ0, groups, w)

The ℓ1/ℓ∞ penalty `f(x) = λ0 Σₖ wₖ maxᵢ∈Gₖ |xᵢ|` over disjoint groups, as
[`GroupNormL2`](@ref) but with the ℓ∞ norm of each group, for simultaneous
selection in multi-task problems. The weights default to `wₖ = 1`. The proximal
operator clips each group, by Moreau's identity with the projection onto an ℓ1
ball, and its optimality test ([`kkt_violation`](@ref)) uses the dual norm,
`‖g[Gₖ]‖₁ - λ0 wₖ`.
"""
struct GroupNormLinf{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractGroupNorm
  λ0::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}
  free::Vector{Int}
  scratch::Scratch

  function GroupNormLinf{T, G}(λ0::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    length(w) == length(groups) || throw(DimensionMismatch("There must be one weight per group."))
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    group_of = _group_of(groups)
    new{T, G}(λ0, groups, w, group_of, findall(iszero, group_of), Scratch())
  end
end

for F in (:GroupNormL2, :GroupNormLinf)
  @eval function $F(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector{<:Real})
    T = float(promote_type(typeof(λ0), eltype(w)))
    G = _own_groups(groups)
    $F{T, typeof(G)}(convert(T, λ0), G, convert(Vector{T}, w))
  end
end
GroupNormL2(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}) =
  GroupNormL2(λ0, groups, [sqrt(float(typeof(λ0))(length(G))) for G in groups])
GroupNormLinf(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}) =
  GroupNormLinf(λ0, groups, ones(float(typeof(λ0)), length(groups)))

_rebuild(f::GroupNormL2, λ0, groups, w) = GroupNormL2(λ0, groups, w)
_rebuild(f::GroupNormLinf, λ0, groups, w) = GroupNormLinf(λ0, groups, w)

# the norm of the group G of x and its dual norm, by loops over the indices: a
# view of a matrix at linear indices reshapes it, which allocates
@inline function _sumsq(x::AbstractArray, G)
  s = zero(float(eltype(x)))
  @inbounds @simd for i in G
    s += abs2(x[i])
  end
  s
end
@inline function _sumabs(x::AbstractArray, G)
  s = zero(float(eltype(x)))
  @inbounds @simd for i in G
    s += abs(x[i])
  end
  s
end
function _maxabs(x::AbstractArray, G)
  s = zero(float(eltype(x)))
  @inbounds for i in G
    s = max(s, abs(x[i]))
  end
  s
end
function _allzero(x::AbstractArray, G)
  @inbounds for i in G
    iszero(x[i]) || return false
  end
  true
end
_group_norm(::GroupNormL2, x, G) = sqrt(_sumsq(x, G))
_group_norm(::GroupNormLinf, x, G) = _maxabs(x, G)
_dual_norm(::GroupNormL2, g, G) = sqrt(_sumsq(g, G))
_dual_norm(::GroupNormLinf, g, G) = _sumabs(g, G)

# the group of index i, 0 if i is in no group
_group(f::AbstractGroupNorm, i::Integer) = i <= length(f.group_of) ? f.group_of[i] : 0

# the groups fit in x: the loops over a group skip the bounds checks
@inline _checkgroups(f::AbstractGroupNorm, x::AbstractArray) =
  length(x) >= length(f.group_of) || throw(BoundsError(x, length(f.group_of)))

# y[G] ← prox_{c‖⋅‖₂}(x[G]), which may be in place; returns the new norm of the group
@inline function _shrink_group!(::GroupNormL2, y::AbstractArray{T}, x::AbstractArray, G, c) where {T}
  nx = sqrt(_sumsq(x, G))
  if nx > c
    t = one(nx) - c / nx
    @inbounds for i in G
      y[i] = t * x[i]
    end
    return nx - c
  else
    @inbounds for i in G
      y[i] = zero(T)
    end
    return zero(nx)
  end
end

# y[G] ← prox_{c‖⋅‖∞}(x[G]), which may be in place: the entries clipped at τ;
# returns τ, the new norm of the group
function _shrink_group!(f::GroupNormLinf, y::AbstractArray{T}, x::AbstractArray, G, c) where {T}
  S = float(T)
  if _sumabs(x, G) <= c
    @inbounds for i in G
      y[i] = zero(T)
    end
    return zero(S)
  end
  τ = _simplex_threshold!(_buffer!(f.scratch, S, length(G)), x, G, convert(S, c), abs)
  @inbounds for i in G
    v = x[i]
    y[i] = sign(v) * min(abs(v), τ)
  end
  τ
end

function (f::AbstractGroupNorm)(x::AbstractArray)
  _checkgroups(f, x)
  T = float(eltype(x))
  v = zero(T)
  for k in eachindex(f.groups)
    v += convert(T, f.w[k]) * _group_norm(f, x, f.groups[k])
  end
  convert(T, f.λ0) * v
end

value_and_prox!(y::AbstractArray, f::AbstractGroupNorm, x::AbstractArray, γ::Real) = _value_and_prox_groups!(y, f, x, γ)
prox!(y::AbstractArray, f::AbstractGroupNorm, x::AbstractArray, γ::Real) = (value_and_prox!(y, f, x, γ); y)

# the step γ is a scalar or a vector with a step per group (see metric.jl)
function _value_and_prox_groups!(y::AbstractArray, f::AbstractGroupNorm, x::AbstractArray, γ::Union{Real, AbstractVector})
  _checksize(y, x)
  _checkgroups(f, x)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  if y !== x                    # indices in no group are not penalized
    @inbounds for i in f.free
      y[i] = x[i]
    end
    @inbounds for i in length(f.group_of)+1:length(x)
      y[i] = x[i]
    end
  end
  v = zero(T)
  for k in eachindex(f.groups)
    v += convert(T, f.w[k]) * _shrink_group!(f, y, x, f.groups[k], convert(T, _entry(γ, k) * f.λ0 * f.w[k]))
  end
  convert(T, f.λ0) * v
end

# a step per group (see metric.jl): one entry for every group
_checkgroupsteps(::AbstractGroupNorm, ::Real) = true
_checkgroupsteps(f::AbstractGroupNorm, γ::AbstractVector) = length(γ) == length(f.groups) ||
  throw(DimensionMismatch("a step per group: $(length(f.groups)) groups and $(length(γ)) steps"))

# the parts of the groups in S, as positions in S, with their weights
function restrict(f::AbstractGroupNorm, S::AbstractVector{<:Integer})
  members = Dict{Int, Vector{Int}}()
  for (j, i) in enumerate(S)
    k = _group(f, i)
    k == 0 || push!(get!(Vector{Int}, members, k), j)
  end
  ks = sort!(collect(keys(members)))
  _rebuild(f, f.λ0, [members[k] for k in ks], f.w[ks])
end

function kkt_violation(f::AbstractGroupNorm, g::AbstractArray, x::AbstractArray, i::Integer)
  _checkgroups(f, g)
  _checkgroups(f, x)
  k = _group(f, i)
  k == 0 && return abs(g[i])
  G = f.groups[k]
  _allzero(x, G) ? _dual_norm(f, g, G) - f.λ0 * f.w[k] : abs(g[i])
end

# on an AtomIterate, k is a group
kkt_violation(f::AbstractGroupNorm, g::AbstractArray, x::AtomIterate, k::Integer) =
  (_checkgroups(f, g); _checksamegroups(f, x); _dual_norm(f, g, x.groups[k]) - f.λ0 * f.w[k])

# the coordinates of x are the groups of f; O(1) when they share them
_checksamegroups(f::AbstractGroupNorm, x::AtomIterate) = x.groups === f.groups || x.groups == f.groups ||
  throw(DimensionMismatch("f and x must have the same groups."))
kkt_violation(f::AbstractGroupNorm, ::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  throw(ArgumentError("$(typeof(f)) groups linear indices; it does not apply to the coordinates of a SymmetricSparseIterate"))

# an AtomIterate that shares the groups of f, which must partition 1:p
AtomIterate(p::Integer, f::AbstractGroupNorm) = AtomIterate(Float64, p, f)
function AtomIterate(::Type{T}, p::Integer, f::AbstractGroupNorm) where {T}
  _check_partition(f.group_of, p)
  _AtomIterate(zeros(T, p), f.groups, f.group_of)
end
function AtomIterate(x::AbstractVector{T}, f::AbstractGroupNorm) where {T}
  _check_partition(f.group_of, length(x))
  _AtomIterate(Vector{T}(x), f.groups, f.group_of)
end

# with the groups of f, only the active groups count
function (f::AbstractGroupNorm)(x::AtomIterate)
  x.groups === f.groups || return invoke(f, Tuple{AbstractArray}, x)
  T = float(eltype(x))
  v = zero(T)
  for a in 1:x.nactive
    k = x.active[a]
    v += convert(T, f.w[k]) * _group_norm(f, x.storage, f.groups[k])
  end
  convert(T, f.λ0) * v
end

value_and_prox!(y::AtomIterate, f::AbstractGroupNorm, x::AtomIterate, γ::Real) = _value_and_prox_groups!(y, f, x, γ)

function _value_and_prox_groups!(y::AtomIterate, f::AbstractGroupNorm, x::AtomIterate, γ::Union{Real, AbstractVector})
  y.groups === x.groups === f.groups ||
    return invoke(_value_and_prox_groups!, Tuple{AbstractArray, AbstractGroupNorm, AbstractArray, Union{Real, AbstractVector}}, y, f, x, γ)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  copyto!(y, x)
  v = zero(T)
  for a in 1:y.nactive
    k = y.active[a]
    v += convert(T, f.w[k]) * _shrink_group!(f, y.storage, y.storage, f.groups[k], convert(T, _entry(γ, k) * f.λ0 * f.w[k]))
  end
  convert(T, f.λ0) * v
end

# group k of x shrunk in place; the group becomes active if it is nonzero, as after
# writing to it through atom(x, k)
function prox_coordinate!(x::AtomIterate, f::AbstractGroupNorm, k::Integer, γ::Real)
  _checksamegroups(f, x)
  T = float(eltype(x))
  nrm = _shrink_group!(f, x.storage, x.storage, x.groups[k], convert(T, γ * f.λ0 * f.w[k]))
  iszero(nrm) || activate!(x, k)
  atom(x, k)
end

####################################
#  ℓ1 plus fusion of a pair
####################################

"""
    NormL1Fused(λ1, λ2)

The penalty `f(x1, x2) = Σᵢ λ1 (|x1ᵢ| + |x2ᵢ|) + λ2 |x1ᵢ - x2ᵢ|` of a pair of
arrays of the same size, given as a tuple `(x1, x2)`, as in the joint
estimation of two sparse models that should be close. Its proximal operator acts
entry by entry on the pairs `(x1ᵢ, x2ᵢ)`; `prox(f, (u, v), γ)` gives it for two
scalars. On a vector of length 2 it is the function of that one pair, which
makes it the penalty on the fibers of two stacked matrices in
[`FiberSum`](@ref): `FiberSum(NormL1Fused(λ1, λ2))` on an `n × n × 2` array is the
penalty of the fused graphical lasso of two classes.
"""
struct NormL1Fused{T<:Real} <: AbstractFunction
  λ1::T
  λ2::T

  function NormL1Fused{T}(λ1::T, λ2::T) where {T<:Real}
    _checknonneg("λ1", λ1)
    _checknonneg("λ2", λ2)
    new{T}(λ1, λ2)
  end
end
NormL1Fused(λ1::Real, λ2::Real) = (p = promote(λ1, λ2); NormL1Fused{typeof(p[1])}(p...))

isproximable(::Type{<:NormL1Fused}) = true
isconvex(::Type{<:NormL1Fused}) = true

(f::NormL1Fused)(xy::Tuple{Real, Real}) = f.λ1 * (abs(xy[1]) + abs(xy[2])) + f.λ2 * abs(xy[1] - xy[2])

function (f::NormL1Fused)(x::Tuple{AbstractArray, AbstractArray})
  x1, x2 = x
  size(x1) == size(x2) || throw(DimensionMismatch("the arrays of the pair have sizes $(size(x1)) and $(size(x2))"))
  T = float(promote_type(eltype(x1), eltype(x2)))
  s = zero(T)
  @inbounds for i in eachindex(x1, x2)
    s += f((x1[i], x2[i]))
  end
  s
end

# argmin (1/2γ) ((u - x)² + (v - y)²) + λ1 (|u| + |v|) + λ2 |u - v|: fuse, then shrink
function _prox_l1fused(x::T, y::T, λ1::T, λ2::T, γ::T) where {T<:AbstractFloat}
  if λ2 > zero(T)
    t = λ2 * γ
    if x > y + 2 * t
      u = x - t
      v = y + t
    elseif y > x + 2 * t
      u = x + t
      v = y - t
    else
      u = v = (x + y) / 2
    end
  else
    u = x
    v = y
  end
  t = λ1 * γ
  (shrink(u, t), shrink(v, t))
end

function prox(f::NormL1Fused, xy::Tuple{Real, Real}, γ::Real = 1)
  T = float(promote_type(typeof(xy[1]), typeof(xy[2])))
  _prox_l1fused(T(xy[1]), T(xy[2]), T(f.λ1), T(f.λ2), T(γ))
end

function prox!(y::Tuple{AbstractArray, AbstractArray}, f::NormL1Fused, x::Tuple{AbstractArray, AbstractArray}, γ::Real)
  size(y[1]) == size(x[1]) == size(y[2]) == size(x[2]) || throw(DimensionMismatch("the arrays of the pairs must have one size"))
  T = float(promote_type(eltype(y[1]), eltype(y[2])))
  λ1, λ2, γ = T(f.λ1), T(f.λ2), T(γ)
  @inbounds for i in eachindex(x[1])
    y[1][i], y[2][i] = _prox_l1fused(T(x[1][i]), T(x[2][i]), λ1, λ2, γ)
  end
  y
end
prox(f::NormL1Fused, x::Tuple{AbstractArray, AbstractArray}, γ::Real = 1) =
  prox!((similar(x[1], float(eltype(x[1]))), similar(x[2], float(eltype(x[2])))), f, x, γ)

# one pair, as a vector of length 2: a fiber of two stacked matrices
_checkpair(x::AbstractVector) =
  length(x) == 2 || throw(DimensionMismatch("NormL1Fused on a vector needs the pair (x1, x2), a vector of length 2; got length $(length(x))"))
(f::NormL1Fused)(x::AbstractVector) = (_checkpair(x); i = firstindex(x); @inbounds f((x[i], x[i+1])))
function value_and_prox!(y::AbstractVector, f::NormL1Fused, x::AbstractVector, γ::Real)
  prox!(y, f, x, γ)
  i = firstindex(y)
  @inbounds f((y[i], y[i+1]))
end
function prox!(y::AbstractVector, f::NormL1Fused, x::AbstractVector, γ::Real)
  _checkpair(x)
  _checksize(y, x)
  T = float(eltype(y))
  i, j = firstindex(x), firstindex(y)
  @inbounds y[j], y[j+1] = _prox_l1fused(T(x[i]), T(x[i+1]), T(f.λ1), T(f.λ2), T(γ))
  y
end
