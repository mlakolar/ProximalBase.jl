####################################
#
#  The sparse group lasso  λ1 Σᵢ |xᵢ| + λ2 Σₖ wₖ ‖x[Gₖ]‖₂
#
####################################

"""
    SparseGroupNormL2(λ1, λ2, groups)
    SparseGroupNormL2(λ1, λ2, groups, w)

The sparse-group-lasso penalty

    f(x) = λ1 Σᵢ |xᵢ| + λ2 Σₖ wₖ ‖x[Gₖ]‖₂

over disjoint groups `Gₖ = groups[k]` of (linear) indices of `x`, with the
weights `wₖ = √|Gₖ|` by default, as in [`GroupNormL2`](@ref). The group term
selects groups and the ℓ1 term zeroes entries inside a selected group (Simon,
Friedman, Hastie and Tibshirani, 2013); `λ1 = 0` gives the group lasso. Indices
in no group are not penalized.

Its proximal operator is exact: soft thresholding at `γλ1`, then the shrinkage of
each group at `γλ2wₖ`, in one pass over a group, which also gives the value.
With one group covering `x`, `SparseGroupNormL2(λ1, λ2, [1:n], [1.0])` is the
sparse group lasso of a single vector. As the fiber function of a
[`FiberSum`](@ref), `FiberSum(SparseGroupNormL2(λ1, λ2, [1:K], [1.0]))` is the
penalty `λ1 Σᵢⱼₖ |xᵢⱼₖ| + λ2 Σᵢⱼ ‖x[i, j, :]‖₂` of the group graphical lasso of
K precision matrices (Danaher, Wang and Witten, 2014).

As for `GroupNormL2`, on an [`AtomIterate`](@ref) that shares its groups
([`AtomIterate`](@ref)`(x, f)`), [`prox_coordinate!`](@ref) takes the block step
on a group and [`kkt_violation`](@ref) measures a group at zero, which is
optimal when `‖soft(g[Gₖ], λ1)‖₂ ≤ λ2wₖ`. Inside a nonzero group, an entry at
zero is optimal when `|gᵢ| ≤ λ1`.
"""
struct SparseGroupNormL2{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: HomogeneousGroupNorm
  λ1::T                   # the level of the ℓ1 term
  λ0::T                   # λ2, the level of the group term, as the shared code reads it
  groups::G
  w::Vector{T}
  group_of::Vector{Int}   # group_of[i] = k for i ∈ Gₖ, 0 for an index in no group
  free::Vector{Int}       # the indices up to the largest in a group that are in none

  function SparseGroupNormL2{T, G}(λ1::T, λ2::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    length(w) == length(groups) || throw(DimensionMismatch("There must be one weight per group."))
    _checknonneg("λ1", λ1)
    _checknonneg("λ2", λ2)
    _checknonneg("the weights", w)
    group_of = _group_of(groups)
    new{T, G}(λ1, λ2, groups, w, group_of, findall(iszero, group_of))
  end
end

function SparseGroupNormL2(λ1::Real, λ2::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector{<:Real})
  T = float(promote_type(typeof(λ1), typeof(λ2), eltype(w)))
  G = _own_groups(groups)
  SparseGroupNormL2{T, typeof(G)}(convert(T, λ1), convert(T, λ2), G, convert(Vector{T}, w))
end
function SparseGroupNormL2(λ1::Real, λ2::Real, groups::AbstractVector{<:AbstractVector{<:Integer}})
  T = float(promote_type(typeof(λ1), typeof(λ2)))
  SparseGroupNormL2(λ1, λ2, groups, [sqrt(T(length(G))) for G in groups])
end

_rebuild(f::SparseGroupNormL2, λ0, groups, w) = SparseGroupNormL2(f.λ1, λ0, groups, w)

# a group at zero is optimal when the soft threshold of the gradient at λ1 lies in
# the ball of radius λ2 wₖ: this is the norm it is measured in
function _dual_norm(f::SparseGroupNormL2, g, G)
  T = float(eltype(g))
  c = convert(T, f.λ1)
  _soft_group_norm(g, G, c)
end

# Retain the vectorized sum for ordinary magnitudes; the fallback keeps
# soft-thresholded norms whose squares overflow or underflow representable.
@inline function _soft_group_norm(x::AbstractArray, G, c::S) where {S}
  s = zero(S)
  @inbounds for i in G
    s += abs2(shrink(convert(S, x[i]), c))
  end
  isfinite(s) && s >= floatmin(S) && return sqrt(s)
  n = zero(S)
  @inbounds for i in G
    n = hypot(n, shrink(convert(S, x[i]), c))
  end
  n
end

# y[G] ← the soft threshold of x[G] at c1, shrunk by c2 in Euclidean norm, which
# may be in place; returns the ℓ2 and ℓ1 norms of the result
@inline function _sparse_shrink_group!(y::AbstractArray{T}, x::AbstractArray, G, c1::S, c2::S) where {T, S}
  ns = _soft_group_norm(x, G, c1)
  if ns > c2
    t = one(S) - c2 / ns
    l1 = zero(S)
    @inbounds for i in G
      v = t * shrink(convert(S, x[i]), c1)
      y[i] = v
      l1 += abs(v)
    end
    return ns - c2, l1
  else
    @inbounds for i in G
      y[i] = zero(T)
    end
    return zero(S), zero(S)
  end
end

# Independent soft thresholding when the group term vanishes. Do not
# compute a Euclidean norm only to multiply it by a zero group weight.
@inline function _soft_group_only!(y::AbstractArray, x::AbstractArray, G, c::S) where {S}
  l1 = zero(S)
  @inbounds for i in G
    z = shrink(convert(S, x[i]), c)
    y[i] = z
    l1 += abs(z)
  end
  l1
end

function (f::SparseGroupNormL2)(x::AbstractArray)
  _checkgroups(f, x)
  T = float(eltype(x))
  v2, v1 = zero(T), zero(T)
  for k in eachindex(f.groups)
    G = f.groups[k]
    w = convert(T, f.w[k])
    (iszero(f.λ0) || iszero(w)) || (v2 += _positive_product(convert(T, f.λ0), w, _group_norm2(x, G)))
    iszero(f.λ1) || (v1 += _sumabs(x, G))
  end
  v2 +
  (iszero(f.λ1) ? zero(T) : convert(T, f.λ1) * v1)
end

value_and_prox!(y::AbstractArray, f::SparseGroupNormL2, x::AbstractArray, γ::Real) = _value_and_prox_groups!(y, f, x, γ)

# the step γ is a scalar or a vector with a step per group (see metric.jl)
function _value_and_prox_groups!(y::AbstractArray, f::SparseGroupNormL2, x::AbstractArray, γ::Union{Real, AbstractVector})
  _checksize(y, x)
  _checkgroups(f, x)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  _copy_ungrouped!(y, f, x)
  v2, v1 = zero(T), zero(T)
  for k in eachindex(f.groups)
    γk = _entry(γ, k)
    w = convert(T, f.w[k])
    if iszero(f.λ0) || iszero(w)
      n1 = _soft_group_only!(y, x, f.groups[k], convert(T, γk * f.λ1))
    else
      n2, n1 = _sparse_shrink_group!(y, x, f.groups[k], convert(T, γk * f.λ1), convert(T, _positive_product(γk, f.λ0, f.w[k])))
      v2 += _positive_product(convert(T, f.λ0), w, n2)
    end
    iszero(f.λ1) || (v1 += n1)
  end
  v2 +
  (iszero(f.λ1) ? zero(T) : convert(T, f.λ1) * v1)
end

# with the groups of f, only the active groups count
function (f::SparseGroupNormL2)(x::AtomIterate)
  x.groups === f.groups || return invoke(f, Tuple{AbstractArray}, x)
  T = float(eltype(x))
  v2, v1 = zero(T), zero(T)
  for a in 1:x.nactive
    k = x.active[a]
    G = f.groups[k]
    w = convert(T, f.w[k])
    (iszero(f.λ0) || iszero(w)) || (v2 += _positive_product(convert(T, f.λ0), w, _group_norm2(x.storage, G)))
    iszero(f.λ1) || (v1 += _sumabs(x.storage, G))
  end
  v2 +
  (iszero(f.λ1) ? zero(T) : convert(T, f.λ1) * v1)
end

value_and_prox!(y::AtomIterate, f::SparseGroupNormL2, x::AtomIterate, γ::Real) = _value_and_prox_groups!(y, f, x, γ)

function _value_and_prox_groups!(y::AtomIterate, f::SparseGroupNormL2, x::AtomIterate, γ::Union{Real, AbstractVector})
  y.groups === x.groups === f.groups ||
    return invoke(_value_and_prox_groups!, Tuple{AbstractArray, SparseGroupNormL2, AbstractArray, Union{Real, AbstractVector}}, y, f, x, γ)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  copyto!(y, x)
  v2, v1 = zero(T), zero(T)
  for a in 1:y.nactive
    k = y.active[a]
    γk = _entry(γ, k)
    w = convert(T, f.w[k])
    if iszero(f.λ0) || iszero(w)
      n1 = _soft_group_only!(y.storage, y.storage, f.groups[k], convert(T, γk * f.λ1))
    else
      n2, n1 = _sparse_shrink_group!(y.storage, y.storage, f.groups[k], convert(T, γk * f.λ1), convert(T, _positive_product(γk, f.λ0, f.w[k])))
      v2 += _positive_product(convert(T, f.λ0), w, n2)
    end
    iszero(f.λ1) || (v1 += n1)
  end
  v2 +
  (iszero(f.λ1) ? zero(T) : convert(T, f.λ1) * v1)
end

# the block step on group k, in place; the group becomes active if it is nonzero
function prox_coordinate!(x::AtomIterate, f::SparseGroupNormL2, k::Integer, γ::Real)
  _checksamegroups(f, x)
  T = float(eltype(x))
  if iszero(f.λ0) || iszero(convert(T, f.w[k]))
    _soft_group_only!(x.storage, x.storage, x.groups[k], convert(T, γ * f.λ1))
    _allzero(x.storage, x.groups[k]) || activate!(x, k)
    return atom(x, k)
  end
  n2, _ = _sparse_shrink_group!(x.storage, x.storage, x.groups[k], convert(T, γ * f.λ1), convert(T, _positive_product(γ, f.λ0, f.w[k])))
  iszero(n2) || activate!(x, k)
  atom(x, k)
end

# an index of a group at zero violates optimality as much as its group; an
# index at zero inside a nonzero group by |gᵢ| - λ1
function kkt_violation(f::SparseGroupNormL2, g::AbstractArray, x::AbstractArray, i::Integer)
  _checkgroups(f, g)
  _checkgroups(f, x)
  k = _group(f, i)
  k == 0 && return abs(g[i])
  G = f.groups[k]
  _allzero(x, G) ? _dual_norm(f, g, G) - f.λ0 * f.w[k] : abs(g[i]) - f.λ1
end
kkt_violation(f::SparseGroupNormL2, g::AbstractArray, x::AtomIterate, k::Integer) =
  (_checkgroups(f, g); _checksamegroups(f, x); _dual_norm(f, g, x.groups[k]) - f.λ0 * f.w[k])
kkt_violation(f::SparseGroupNormL2, ::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  throw(ArgumentError("$(typeof(f)) groups linear indices; it does not apply to the coordinates of a SymmetricSparseIterate"))
