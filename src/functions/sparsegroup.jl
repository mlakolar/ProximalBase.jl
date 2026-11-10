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
in no group are not penalized. An infinite `λ1` fixes every grouped entry at
zero: the value is then 0 when those entries are zero, `Inf` when one is nonzero
and NaN when one is NaN, and the proximal output is zero at every grouped entry
at every step, `γ = 0` and `γ = Inf` included, whatever the entry holds and
whatever `λ2` and the weights are (zero included), so it does not depend on the
step. Indices in no group stay free.

Its proximal operator is exact: soft thresholding at `γλ1`, then the shrinkage of
each group at `γλ2wₖ`, in one pass over a group, which also gives the value.
With one group covering `x`, `SparseGroupNormL2(λ1, λ2, [1:n], [1.0])` is the
sparse group lasso of a single vector. As the fiber function of a
[`FiberSum`](@ref), `FiberSum(SparseGroupNormL2(λ1, λ2, [1:K], [1.0]))` is the
penalty `λ1 Σᵢⱼₖ |xᵢⱼₖ| + λ2 Σᵢⱼ ‖x[i, j, :]‖₂` of the group graphical lasso of
K precision matrices (Danaher, Wang and Witten, 2014).

As for `GroupNormL2`, on an [`AtomIterate`](@ref) that shares its groups
([`AtomIterate`](@ref)`(x, f)`), [`prox_block!`](@ref) computes the block result
on a group and [`kkt_violation`](@ref) measures a group at zero, which is
optimal when `‖soft(g[Gₖ], λ1)‖₂ ≤ λ2wₖ`. Inside a nonzero group, an entry at
zero is optimal when `|gᵢ| ≤ λ1`.
Values follow the overflow rule in [Overflow and underflow of values](@ref).
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

function SparseGroupNormL2(λ1::Real, λ2::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector)
  eltype(w) <: Real || _real_error(:SparseGroupNormL2)
  T = float(promote_type(typeof(λ1), typeof(λ2), eltype(w)))
  G = _own_groups(groups)
  SparseGroupNormL2{T, typeof(G)}(convert(T, λ1), convert(T, λ2), G, convert(Vector{T}, w))
end
function SparseGroupNormL2(λ1::Real, λ2::Real, groups::AbstractVector{<:AbstractVector{<:Integer}})
  T = float(promote_type(typeof(λ1), typeof(λ2)))
  SparseGroupNormL2(λ1, λ2, groups, [sqrt(T(length(G))) for G in groups])
end

_group_norm(::SparseGroupNormL2, x, G) = _group_norm2(x, G)

# The L1 sum covers penalized groups only, including for an active iterate.
@inline function _sparse_l1_value(f, x, v, ::Type{T}) where {T}
  iszero(f.λ1) && return zero(T)
  # an infinite level fixes every grouped entry at zero: v, a sum of absolute values, is zero exactly
  # when each entry is, so the term is 0 there, Inf where v > 0 and NaN where v is NaN
  isinf(f.λ1) && return convert(T, _weighted_value(f.λ1 * v, v))
  _level_fits(T, f.λ1) && isfinite(v) && return convert(T, f.λ1) * v
  _scaled_sparse_l1(f, x, T)
end

@noinline function _scaled_sparse_l1(f, x, ::Type{T}) where {T}
  R = promote_type(T, float(typeof(f.λ1)))
  m = zero(R)
  for G in f.groups
    m = max(m, convert(R, _maxabs(x, G)))
  end
  isfinite(m) || return convert(T, convert(R, f.λ1) * m)
  e = _magnitude_exponent(m)
  scales = _power_factors(R, -e)
  s = err = zero(R)
  for G in f.groups
    @inbounds for i in G
      s, err = _compensated_add(s, err, abs(_scale_value(convert(R, x[i]), scales)))
    end
  end
  convert(T, _scaled_product(convert(R, f.λ1), s + err, e))
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
  if iszero(s)                   # most groups threshold to zero; confirm without hypot
    a = zero(S)
    @inbounds for i in G
      a += abs(shrink(convert(S, x[i]), c))
    end
    iszero(a) && return s
  end
  n = zero(S)
  @inbounds for i in G
    z = shrink(convert(S, x[i]), c)
    isnan(z) && return oftype(n, NaN)
    n = hypot(n, z)
  end
  n
end

# y[G] ← the soft threshold of x[G] at c1, shrunk by c2 in Euclidean norm, which
# may be in place; returns the ℓ2 and ℓ1 norms of the result
@inline function _sparse_shrink_group!(y::AbstractArray{T}, x::AbstractArray, G, c1::S, c2::S, W = nothing)::Tuple{S, S} where {T, S}
  ns = _soft_group_norm(x, G, c1)
  if ns > c2
    if isinf(ns) && isfinite(c2)
      scale = convert(S, _maxabs(x, G))
      if isfinite(scale) && scale > 1
        n2, n1 = _sparse_shrink_group!(y, _ScaledArray(x, inv(scale)), G, c1 / scale, c2 / scale, W)
        @inbounds for i in _writes(W, G)
          y[i] *= scale
        end
        return n2 * scale, n1 * scale
      end
    end
    t = one(S) - c2 / ns
    l1 = zero(S)
    # sum before aliased writes, retaining both positions of a symmetric pair
    if W !== nothing && W !== G
      @inbounds for i in G
        v = t * shrink(convert(S, x[i]), c1)
        l1 += abs(v)
      end
    end
    @inbounds for i in _writes(W, G)
      v = t * shrink(convert(S, x[i]), c1)
      y[i] = v
      (W === nothing || W === G) && (l1 += abs(v))
    end
    return ns - c2, l1
  elseif ns <= c2
    @inbounds for i in _writes(W, G)
      y[i] = zero(T)
    end
    return zero(S), zero(S)
  else
    # A NaN norm (E2) fails both comparisons above. The group is left as it is, with NaN norms, and
    # settled by the caller after its loop (_nan_steps_sparse!), which knows the original step,
    # levels and weight; the zero fill above is the one the kernel always had.
    return ns, ns
  end
end

# An infinite weight or level (only in the general path, O = false) fixes the group at zero whatever it
# holds (E4), as _fixed_group in norms.jl: the norms of the zero group, decided before the kernel.
@inline _fixed_sparse_group(f::SparseGroupNormL2, ::Val{O}, γk, k) where {O} = !O && _hard_fixed_sparse(f, γk, k)
@inline _hard_fixed_sparse(f, γk, k) = _hard_fixed(f, γk, k) | isinf(f.λ1)
@inline function _fixed_sparse_group!(y, G, W, ::Type{T}) where {T}
  @inbounds for i in _writes(W, G)
    y[i] = zero(eltype(y))
  end
  zero(T), zero(T)
end

# After the loops, when the value is NaN: the groups the kernel left as they were because their norm is
# NaN, as for _nan_steps!. The original parameters decide (E4 first): an infinite weight, step, ℓ1 level or
# (at a positive step) group level fixes the group at zero; at the step γ = 0 the map is the identity (each
# entry as it is); otherwise, also when a product underflowed or overflowed, the NaN block. Returns the
# value at the stored output.
function _nan_steps_sparse!(f::SparseGroupNormL2, y, X, ks, γ, ::Val{O}, ::Type{W}, v, yobj) where {O, W}
  changed = false
  T = eltype(y)
  for k in ks
    iszero(f.w[k]) && continue
    (O && iszero(f.λ0)) && continue
    (!O && iszero(f.λ0) && !isinf(f.w[k])) && continue
    G = f.groups[k]
    γk = _entry(γ, k)
    c1 = _l1_threshold(W, γk, f.λ1)
    isnan(_soft_group_norm(X, G, c1)) || continue
    wr = _group_writes(y, G)
    if _hard_fixed_sparse(f, γk, k)
      @inbounds for i in _writes(wr, G)
        y[i] = zero(T)
      end
    elseif iszero(γk)
      @inbounds for i in _writes(wr, G)
        y[i] = convert(T, shrink(convert(W, X[i]), c1))
      end
    else
      @inbounds for i in _writes(wr, G)
        y[i] = T(NaN)
      end
    end
    changed = true
  end
  changed ? f(yobj) : v
end

# γλ1, where a zero level gives zero at every step, including γ = Inf (E4 before the step).
@inline _l1_threshold(::Type{U}, γ, λ1) where {U<:AbstractFloat} = iszero(λ1) ? zero(U) : _typed_product(U, γ, λ1)

# Independent soft thresholding when the group term vanishes. Do not
# compute a Euclidean norm only to multiply it by a zero group weight.
@inline function _soft_group_only!(y::AbstractArray, x::AbstractArray, G, c::S, W = nothing) where {S}
  l1 = zero(S)
  # sum before aliased writes, retaining both positions of a symmetric pair
  if W !== nothing && W !== G
    @inbounds for i in G
      z = shrink(convert(S, x[i]), c)
      l1 += abs(z)
    end
  end
  @inbounds for i in _writes(W, G)
    z = shrink(convert(S, x[i]), c)
    y[i] = z
    (W === nothing || W === G) && (l1 += abs(z))
  end
  l1
end

function (f::SparseGroupNormL2)(x::AbstractArray)
  _checkreal(f, x)
  isfinite(f.λ0) && !iszero(f.λ0) && return _sparse_group_value(f, x, Val(true))
  isfinite(f.λ1) && _ordinary_group_weights(f, float(eltype(x))) ?
    _sparse_group_value(f, x, Val(true)) : _sparse_group_value(f, x, Val(false))
end

function _sparse_group_value(f::SparseGroupNormL2, x::AbstractArray, ordinary::Val{O}) where {O}
  _checkgroups(f, x)
  T = float(eltype(x))
  v2, v1 = zero(T), zero(T)
  for k in eachindex(f.groups)
    G = f.groups[k]
    ((O && iszero(f.λ0)) || iszero(f.w[k]) || (!O && iszero(f.λ0) && !isinf(f.w[k]))) || (v2 += _group_value(f, f.λ0, f.w[k], _group_norm2(x, G), x, G, T, ordinary))
    iszero(f.λ1) || (v1 += _sumabs(x, G))
  end
  v2 +
  _sparse_l1_value(f, x, v1, T)
end

value_and_prox!(y::AbstractArray, f::SparseGroupNormL2, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _checkwrite(value_and_prox!, f, y, x); _value_and_prox_groups!(y, f, x, γ))

# the step γ is a scalar or a vector with a step per group (see metric.jl)
function _value_and_prox_groups!(y::AbstractArray, f::SparseGroupNormL2, x::AbstractArray, γ::Union{Real, AbstractVector})
  isinf(f.λ1) && return _infinite_l1_prox!(y, f, x, γ)
  if _ordinary_group_weights(f, _model_type(y, x))
    _sparse_group_prox!(y, f, x, γ, Val(true))
  else
    _sparse_group_prox!(y, f, x, γ, Val(false))
  end
end

# An infinite ℓ1 level, as stored (a finite level never counts, R3), fixes every grouped entry at zero
# at every step, γ = 0 and γ = Inf included, whatever the entry holds and whatever the group level
# and weight are: the projection onto the domain, decided before any threshold γ⋅λ1 is formed. The
# checks the kernel runs come first; indices in no group are copied. The value at the output is 0:
# the fixed groups and the ℓ1 term of zero entries give 0, and indices in no group are not penalized.
function _infinite_l1_prox!(y::AbstractArray, f::SparseGroupNormL2, x::AbstractArray, γ::Union{Real, AbstractVector})
  _checksize(y, x)
  _checkgroups(f, x)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  _copy_ungrouped!(y, f, x)
  for G in f.groups
    _fixed_sparse_group!(y, G, _group_writes(y, G), T)
  end
  zero(T)
end

function _sparse_group_prox!(y::AbstractArray, f::SparseGroupNormL2, x::AbstractArray, γ::Union{Real, AbstractVector}, ordinary::Val{O}) where {O}
  _checksize(y, x)
  _checkgroups(f, x)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  W = _model_type(y, x)
  _copy_ungrouped!(y, f, x)
  v2, v1 = zero(T), zero(T)
  for k in eachindex(f.groups)
    γk = _entry(γ, k)
    if (O && iszero(f.λ0)) || iszero(f.w[k]) || (!O && iszero(f.λ0) && !isinf(f.w[k]))
      n1 = _soft_group_only!(y, x, f.groups[k], _l1_threshold(W, γk, f.λ1), _group_writes(y, f.groups[k]))
      W === T || (n1 = _sumabs(y, f.groups[k]))        # O2: the values are those of the stored output
    else
      n2, n1 = if _fixed_sparse_group(f, ordinary, γk, k)
        _fixed_sparse_group!(y, f.groups[k], _group_writes(y, f.groups[k]), W)
      else
        _sparse_shrink_group!(y, x, f.groups[k], _l1_threshold(W, γk, f.λ1), (O ? _positive_product(W, γk, f.λ0, f.w[k]) : _weight_product(W, γk, f.λ0, f.w[k])), _group_writes(y, f.groups[k]))
      end
      W === T || isnan(n2) || ((n2, n1) = (_group_norm2(y, f.groups[k]), _sumabs(y, f.groups[k])))
      v2 += _group_value(f, f.λ0, f.w[k], n2, y, f.groups[k], T, ordinary)
    end
    iszero(f.λ1) || (v1 += convert(T, n1))
  end
  v = v2 + _sparse_l1_value(f, y, v1, T)
  isnan(v) ? _nan_steps_sparse!(f, y, x, eachindex(f.groups), γ, ordinary, W, v, y) : v
end

# with the groups of f, only the active groups count
function (f::SparseGroupNormL2)(x::AtomIterate)
  _checkreal(f, x)
  _samelayout(f, x) || return invoke(f, Tuple{AbstractArray}, x)
  isfinite(f.λ0) && !iszero(f.λ0) && return _sparse_group_value(f, x, Val(true))
  isfinite(f.λ1) && _ordinary_group_weights(f, float(eltype(x)), x.active, x.nactive) ?
    _sparse_group_value(f, x, Val(true)) : _sparse_group_value(f, x, Val(false))
end

function _sparse_group_value(f::SparseGroupNormL2, x::AtomIterate, ordinary::Val{O}) where {O}
  _samelayout(f, x) || return invoke(_sparse_group_value, Tuple{SparseGroupNormL2, AbstractArray, Val{O}}, f, x, ordinary)
  T = float(eltype(x))
  v2, v1 = zero(T), zero(T)
  for a in 1:x.nactive
    k = x.active[a]
    G = f.groups[k]
    ((O && iszero(f.λ0)) || iszero(f.w[k]) || (!O && iszero(f.λ0) && !isinf(f.w[k]))) || (v2 += _group_value(f, f.λ0, f.w[k], _group_norm2(x.storage, G), x.storage, G, T, ordinary))
    iszero(f.λ1) || (v1 += _sumabs(x.storage, G))
  end
  v2 +
  _sparse_l1_value(f, x, v1, T)
end

value_and_prox!(y::AtomIterate, f::SparseGroupNormL2, x::AtomIterate, γ::Real) =
  (_checkwrite(value_and_prox!, f, y, x); _value_and_prox_groups!(y, f, x, γ))

function _value_and_prox_groups!(y::AtomIterate, f::SparseGroupNormL2, x::AtomIterate, γ::Union{Real, AbstractVector})
  _samelayout(f, x) && _samelayout(f, y) ||
    return invoke(_value_and_prox_groups!, Tuple{AbstractArray, SparseGroupNormL2, AbstractArray, Union{Real, AbstractVector}}, y, f, x, γ)
  if isinf(f.λ1)
    # as _infinite_l1_prox!: the active groups are zeroed and stay active, the others are zero
    _checkgroupsteps(f, γ)
    T = float(eltype(y))
    copyto!(y, x)
    for a in 1:y.nactive
      _fixed_sparse_group!(y.storage, f.groups[y.active[a]], nothing, T)
    end
    return zero(T)
  end
  if _ordinary_group_weights(f, float(eltype(y)), x.active, x.nactive)
    _sparse_group_prox!(y, f, x, γ, Val(true))
  else
    _sparse_group_prox!(y, f, x, γ, Val(false))
  end
end

function _sparse_group_prox!(y::AtomIterate, f::SparseGroupNormL2, x::AtomIterate, γ::Union{Real, AbstractVector}, ordinary::Val{O}) where {O}
  _samelayout(f, x) && _samelayout(f, y) ||
    return invoke(_sparse_group_prox!, Tuple{AbstractArray, SparseGroupNormL2, AbstractArray, Union{Real, AbstractVector}, Val{O}}, y, f, x, γ, ordinary)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  copyto!(y, x)
  v2, v1 = zero(T), zero(T)
  for a in 1:y.nactive
    k = y.active[a]
    γk = _entry(γ, k)
    if (O && iszero(f.λ0)) || iszero(f.w[k]) || (!O && iszero(f.λ0) && !isinf(f.w[k]))
      n1 = _soft_group_only!(y.storage, y.storage, f.groups[k], _l1_threshold(T, γk, f.λ1))
    else
      n2, n1 = if _fixed_sparse_group(f, ordinary, γk, k)
        _fixed_sparse_group!(y.storage, f.groups[k], nothing, T)
      else
        _sparse_shrink_group!(y.storage, y.storage, f.groups[k], _l1_threshold(T, γk, f.λ1), (O ? _positive_product(T, γk, f.λ0, f.w[k]) : _weight_product(T, γk, f.λ0, f.w[k])))
      end
      v2 += _group_value(f, f.λ0, f.w[k], n2, y.storage, f.groups[k], T, ordinary)
    end
    iszero(f.λ1) || (v1 += convert(T, n1))
  end
  v = v2 + _sparse_l1_value(f, y, v1, T)
  isnan(v) ? _nan_steps_sparse!(f, y.storage, y.storage, view(y.active, 1:y.nactive), γ, ordinary, T, v, y) : v
end

# an index of a group at zero violates optimality as much as its group; an
# index at zero inside a nonzero group by |gᵢ| - λ1
function _kkt_violation(f::SparseGroupNormL2, g::AbstractArray, x::AbstractArray, i::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
  eltype(x) <: Real || _real_error(:kkt_violation)
  _checkgroups(f, g)
  _checkgroups(f, x)
  k = _group(f, i)
  k == 0 && return abs(g[i])
  G = f.groups[k]
  _allzero(x, G) ? _dual_norm(f, g, G) - _kkt_coefficient(eltype(g), f.λ0, f.w[k]) : abs(g[i]) - f.λ1
end
_kkt_violation(f::SparseGroupNormL2, g::AbstractArray, x::AtomIterate, k::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); _checkgroups(f, g); _checksamegroups(f, x); _dual_norm(f, g, x.groups[k]) - _kkt_coefficient(eltype(g), f.λ0, f.w[k]))
_kkt_violation(f::SparseGroupNormL2, g::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); throw(ArgumentError("$(typeof(f)) groups linear indices; it does not apply to the coordinates of a SymmetricSparseIterate")))
