####################################
#  group ℓ2 norm  λ0 Σₖ wₖ ‖x[Gₖ]‖₂
####################################

"""
    GroupNormL2(λ0, groups)
    GroupNormL2(λ0, groups, w)

Group-lasso penalty `f(x) = λ₀ Σₖ wₖ‖x[Gₖ]‖₂`, with default weights
`wₖ = √|Gₖ|`. Groups are nonempty, disjoint lists or ranges of positive
linear indices. Their order is preserved; entries in no group are unpenalized.
The nonnegative level and weights cannot be `NaN`. Supply one weight per group;
the constructor copies `groups` and `w`.

[`prox`](@ref) shrinks each group toward zero using a nonnegative **scalar** step.
[`prox_block`](@ref) takes a compact vector in `groups[k]` order; it does not
update an iterate. To commit its result, use `copyto!(blockview(x, k), out)`.
See [group operations](@ref group-l2-guide) for the formula and KKT screen.

# Examples
```jldoctest
julia> f = GroupNormL2(1.0, [1:2], [1.0]);

julia> f([3.0, 4.0])
5.0

julia> prox_block(f, [3.0, 4.0], 1, 2.5)
2-element Vector{Float64}:
 1.5
 2.0

julia> h = GroupNormL2(1.0, [[3, 1]], [1.0]);

julia> Tuple(prox(h, [3.0, 9.0, 4.0], 2.5))
(1.5, 9.0, 2.0)
```

# Edge cases
- Real, one-based arrays and valid shapes/groups are required. Negative or `NaN`
  steps fail before writes.
- Exact in-place prox is supported. For compact blocks, `out === u` is supported;
  other overlaps detected by `Base.mightalias` are rejected, and `out` must
  not alias iterate storage.
- Zero-weight groups stay unchanged. Infinite weights or an infinite level with
  positive weight fix a group at `+0`, including at zero step.
- No `SymmetricSparseIterate`, `restrict`, per-group steps or compact block method
  under [`Scaled`](@ref).

See [group endpoint rules](@ref group-l2-endpoints) and the
[numerical contract](@ref numerics) for non-finite inputs and range limits.
"""
struct GroupNormL2{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractFunction
  λ0::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}   # group_of[i] = k for i ∈ Gₖ, 0 for an index in no group
  free::Vector{Int}       # the indices up to the largest in a group that are in none

  function GroupNormL2{T, G}(λ0::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    _check_one_based_groups(groups)
    length(w) == length(groups) ||
      throw(DimensionMismatch("GroupNormL2 needs one weight per group: $(length(groups)) groups and $(length(w)) weights"))
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    group_of = _group_of(groups)
    new{T, G}(λ0, groups, w, group_of, findall(iszero, group_of))
  end
end
function GroupNormL2(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector)
  _check_one_based_groups(groups)
  _check_one_based(w)
  eltype(w) <: Real || _real_error(:GroupNormL2)
  T = float(promote_type(typeof(λ0), eltype(w)))
  G = _own_groups(groups)
  GroupNormL2{T, typeof(G)}(convert(T, λ0), G, Vector{T}(w))
end
function GroupNormL2(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}})
  _check_one_based_groups(groups)
  GroupNormL2(λ0, groups, [sqrt(float(typeof(λ0))(length(G))) for G in groups])
end

isproximable(::Type{<:GroupNormL2}) = true
isconvex(::Type{<:GroupNormL2}) = true

# A BlockIterate with the groups of f, which must partition 1:p (the public constructor checks it).
BlockIterate(p::Integer, f::GroupNormL2) = BlockIterate(Float64, p, f)
BlockIterate(::Type{T}, p::Integer, f::GroupNormL2) where {T} = BlockIterate(T, p, f.groups)
BlockIterate(x::AbstractVector, f::GroupNormL2) = BlockIterate(x, f.groups)

####################################
#  Checks
####################################

# GroupNormL2 and GroupNormLinf (groupnormlinf.jl, included before this file) have the same group fields; the checks
# and the sums of the checked value below serve both, and the messages name the type of f.
const _GroupNorm = Union{GroupNormL2, GroupNormLinf}

@noinline _group_extent_error(f, x) = throw(DimensionMismatch(
  "the groups of $(nameof(typeof(f))) reach index $(length(f.group_of)), but the argument has $(length(x)) entries"))
# every group index is a linear index of x: the loops over a group skip the bounds checks
@inline _check_group_extent(f::_GroupNorm, x) = (length(x) >= length(f.group_of) || _group_extent_error(f, x); nothing)
_checkparams(f::_GroupNorm, x) = (_check_group_extent(f, x); true)
# a symmetric iterate is rejected by the oracles themselves, with their own message
_checkparams(::_GroupNorm, ::SymmetricSparseIterate) = true

function _check_group_arrays(f, y::AbstractArray, x::AbstractArray, γ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  Base.require_one_based_indexing(y, x)
  _checksize(y, x)
  _check_group_extent(f, x)
  _check_l2_step(f, γ)
  nothing
end

@noinline _block_length_error(f, k, m, nu, nout) = throw(DimensionMismatch(
  "group $k of $(nameof(typeof(f))) has $m entries, but the input has $nu and the output $nout"))
@noinline _block_alias_error() = throw(ArgumentError("prox_block! supports exact input/output aliasing (out === u) only"))

@inline function _check_group_block(out::AbstractVector, f, u::AbstractVector, k::Integer, γ::Real)
  (!(k isa Bool) && checkbounds(Bool, f.groups, k)) || throw(BoundsError(Base.OneTo(length(f.groups)), k))
  Base.require_one_based_indexing(out, u)
  (eltype(out) <: Real && eltype(u) <: Real) || _real_error(:prox_block!)
  m = @inbounds length(f.groups[k])
  (length(u) == m && length(out) == m) || _block_length_error(f, k, m, length(u), length(out))
  (out === u || !Base.mightalias(out, u)) || _block_alias_error()
  _check_l2_step(f, γ)
  m
end

####################################
#  Group values
####################################

# The kernels are the shared ℓ2 kernels of numerical_kernels.jl, with the level λ0 and the weight wₖ of each group.

# The ordinary sum s of the terms is returned unless it may have lost its accuracy or its class: a sum
# below K floatmin(T), for K groups, whose terms may have rounded into the subnormal range (#373), and a
# sum at or above floatmax(T)/2 (Inf included), whose class the terms' rounding may decide. Those are
# recomputed from the terms w ns 2^k (w m 2^e for GroupNormLinf, with m its largest magnitude), accumulated with their
# exponents (_ExpSum, in at least Float64),
# with λ0 applied once (_expsum_value). Exceptional cases keep s: a NaN s, a zero level (every term with
# a finite weight is 0, and an infinite weight at a nonzero group without NaN makes s Inf), and an infinite level or
# weight at a nonzero group or a NaN or ±Inf entry in a penalized group, which make s infinite or NaN
# already. So does an exactly zero argument.
# K counts every group, penalized or not. The scan for a zero argument runs only below K floatmin, and for Float32
# and Float64 it returns at its first block of 512 stored values that holds a nonzero one.
@inline function _group_value_checked(f::_GroupNorm, x, s::T) where {T}
  if _l2_ordinary(s, T)
    _underflow_checked(T) || return s
    _sum_kept(s, length(f.groups)) && return s
  end
  _group_value_unkept(f, x, s)
end
# out of line, so that the loop over the groups of the callers does not carry the scans
@noinline function _group_value_unkept(f::_GroupNorm, x, s::T)::T where {T}
  if _l2_ordinary(s, T)
    iszero(s) && _all_zero_early(_stored_values(x)) && return s
  end
  _group_value_recomputed(f, x, s)
end
@noinline function _group_value_recomputed(f::_GroupNorm, x, s::T)::T where {T}
  isnan(s) && return s
  λ = f.λ0
  (λ > 0 && isfinite(λ)) || return s
  R = _recovery_type(T, float(typeof(λ)), float(eltype(x)))
  acc = _expsum_start(R)
  for k in eachindex(f.groups)
    w = @inbounds f.w[k]
    iszero(w) && continue          # an unpenalized group adds nothing, whatever it holds
    next = _group_expsum_add(f, acc, x, @inbounds(f.groups[k]), w)
    next === nothing && return s
    acc = next
  end
  _expsum_value(T, acc, λ)
end
# the term of one group in the recomputed sum: w ‖x[G]‖₂ or w max|x[G]|
@inline _group_expsum_add(::GroupNormL2, acc, x, G, w) = _l2_expsum_add(acc, x, G, w)
@inline _group_expsum_add(::GroupNormLinf, acc, x, G, w) = _linf_expsum_add(acc, x, G, w)

####################################
#  Oracles
####################################

function (f::GroupNormL2)(x::AbstractArray)
  _checkreal(f, x)
  Base.require_one_based_indexing(x)
  _check_group_extent(f, x)
  T = _valtype(x)
  v = zero(T)
  for k in eachindex(f.groups)
    v += _l2_term(f.λ0, @inbounds(f.w[k]), x, @inbounds(f.groups[k]), _L2_RECOMPUTE, T)
  end
  _group_value_checked(f, x, v)
end

prox!(y::AbstractArray, f::GroupNormL2, x::AbstractArray, γ::Real) =
  (_check_group_arrays(f, y, x, γ); _group_prox_all!(y, f, x, γ, Val(false)); y)
value_and_prox!(y::AbstractArray, f::GroupNormL2, x::AbstractArray, γ::Real) =
  (_check_group_arrays(f, y, x, γ); _group_prox_all!(y, f, x, γ, Val(true)))
# routes of wrappers such as Scaled, which preflight with _checkwrite first
_value_and_prox_unchecked!(y::AbstractArray, f::GroupNormL2, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)

function _group_prox_all!(y, f::GroupNormL2, x, γ::Real, ::Val{V}) where {V}
  if y !== x       # the entries in no group are unpenalized
    @inbounds for i in f.free
      y[i] = x[i]
    end
    @inbounds for i in length(f.group_of)+1:length(x)
      y[i] = x[i]
    end
  end
  # A zero or infinite level, or a zero or infinite step, makes the product γλ0 zero, infinite or NaN, so that the range
  # test of the ordinary path (_threshold_product) fails for every group and the parameters decide it: the loop calls the
  # parameter exits (_l2_shrink_fixed!, inlined) directly. _group_prox_loop! is compiled once for each shrinkage
  # function, so the ordinary loop has the code of a single loop, where LLVM can hoist the loop-invariant threshold.
  λ = f.λ0
  if isinf(λ) | iszero(λ) | iszero(γ) | isinf(γ)
    _group_prox_loop!(_l2_shrink_fixed!, y, f, x, γ, Val(V))
  else
    _group_prox_loop!(_l2_shrink!, y, f, x, γ, Val(V))
  end
end
@noinline function _group_prox_loop!(shrink::F, y, f::GroupNormL2, x, γ::Real, ::Val{V}) where {F, V}
  T = _valtype(y)
  v = zero(T)
  for k in eachindex(f.groups)
    G = @inbounds f.groups[k]
    w = @inbounds f.w[k]
    s = shrink(y, x, G, f.λ0, w, γ)
    V && (v += _l2_fused_term(f.λ0, w, y, G, s, T))
  end
  V ? _group_value_checked(f, y, v) : nothing
end

function prox_block!(out::AbstractVector, f::GroupNormL2, u::AbstractVector, k::Integer, γ::Real)
  m = _check_group_block(out, f, u, k, γ)
  _l2_shrink!(out, u, Base.OneTo(m), f.λ0, @inbounds(f.w[k]), γ, _l2_shrink_fixed!)
  out
end

####################################
#  Optimality
####################################

@noinline _not_separable_error(f) = throw(ArgumentError(
  "$(nameof(typeof(f))) is not separable: use kkt_violation(f, g, x, i) with an array point or a BlockIterate"))
kkt_violation(f::GroupNormL2, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); _not_separable_error(f))

@noinline _gradient_length_error(g, x) = throw(DimensionMismatch(
  "the gradient has length $(length(g)), but the argument has $(length(x)) entries"))

@noinline _block_layout_error(f, k) = throw(DimensionMismatch("group $k of the BlockIterate differs from group $k of $(nameof(typeof(f)))"))
@noinline _block_count_error(x, f) = throw(DimensionMismatch(
  "the BlockIterate has $(ncoordinates(x)) groups and $(nameof(typeof(f))) has $(length(f.groups))"))

# The score of a group (_l2_score) with the level λ of the call, which is f.λ0 or, for the function under a chain of
# Scaled wrappers, the level that the chain folds into it. Group k of f is selected by the entry i of an array point (an
# entry in no group has the score |gᵢ|, as the penalty does not depend on it) or by the group number k of a
# BlockIterate, whose layout is compared with that of f at group k alone. A SymmetricSparseIterate has no groups.
function _level_score(f::GroupNormL2, λ, g::AbstractArray, x::AbstractArray, i::Integer)
  length(g) == length(x) || _gradient_length_error(g, x)
  _check_group_extent(f, x)
  checkbounds(x, i)
  k = i <= length(f.group_of) ? @inbounds(f.group_of[i]) : 0
  k == 0 && return abs(convert(float(promote_type(eltype(g), eltype(x))), g[i]))
  _l2_score(λ, @inbounds(f.w[k]), g, x, @inbounds(f.groups[k]))
end
function _level_score(f::GroupNormL2, λ, g::AbstractArray, x::BlockIterate, k::Integer)
  ncoordinates(x) == length(f.groups) || _block_count_error(x, f)
  B = blockindices(x, k)          # BoundsError for an invalid group
  G = @inbounds f.groups[k]
  B == G || _block_layout_error(f, k)
  length(g) == length(x) || _gradient_length_error(g, x)
  _l2_score(λ, @inbounds(f.w[k]), g, x, G)
end
_level_score(::GroupNormL2, λ, g::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  throw(ArgumentError("GroupNormL2 groups linear indices; it does not apply to the coordinates of a SymmetricSparseIterate"))
_kkt_violation(f::GroupNormL2, g::AbstractArray, x::AbstractArray, i::Integer) = _level_score(f, f.λ0, g, x, i)
_kkt_violation(f::GroupNormL2, g::AbstractArray, x::BlockIterate, k::Integer) = _level_score(f, f.λ0, g, x, k)
_kkt_violation(f::GroupNormL2, g::AbstractArray, x::SymmetricSparseIterate, k::Integer) = _level_score(f, f.λ0, g, x, k)

# The group layout of a BlockIterate point of the full-point form: its group count, and each of its groups, are those of f
_check_point_layout(f::_GroupNorm, x::AbstractArray) = nothing
function _check_point_layout(f::_GroupNorm, x::BlockIterate)
  ncoordinates(x) == length(f.groups) || _block_count_error(x, f)
  for k in eachindex(f.groups)
    blockindices(x, k) == @inbounds(f.groups[k]) || _block_layout_error(f, k)
  end
  nothing
end

# max(0, max_k score_k), the maximum that keeps a NaN: the scores of the groups and |vᵢ| of the entries in no group
# (the entries that no group holds up to the largest index of a group, and those past it)
function _level_point(f::GroupNormL2, λ, v::AbstractArray, x::AbstractArray)
  _check_one_based(v, x)
  eltype(v) <: Real || _real_error(:kkt_violation)
  eltype(x) <: Real || _real_error(:kkt_violation)
  length(v) == length(x) || _gradient_length_error(v, x)
  _check_group_extent(f, x)
  _check_point_layout(f, x)
  W = float(promote_type(eltype(v), eltype(x)))
  m = zero(W)
  for k in eachindex(f.groups)
    m = max(m, _l2_score(λ, @inbounds(f.w[k]), v, x, @inbounds(f.groups[k])))
  end
  @inbounds for i in f.free
    m = max(m, abs(convert(W, v[i])))
  end
  @inbounds for i in length(f.group_of)+1:length(x)
    m = max(m, abs(convert(W, v[i])))
  end
  m
end
kkt_violation(f::GroupNormL2, v::AbstractArray, x::AbstractArray) = _level_point(f, f.λ0, v, x)
kkt_violation(f::GroupNormL2, ::AbstractArray, ::SymmetricSparseIterate) = _kkt_symmetric_point_error(f)

# no proximal operator for a symmetric iterate (argument or output); nothing is written
for F in (:NormL2, :GroupNormL2, :GroupNormLinf)
  @eval begin
    prox!(::SymmetricSparseIterate, f::$F, x, ::Real) = _symmetric_prox_error(f)
    prox!(::AbstractArray, f::$F, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
    value_and_prox!(::SymmetricSparseIterate, f::$F, x, ::Real) = _symmetric_prox_error(f)
    value_and_prox!(::AbstractArray, f::$F, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
    prox(f::$F, ::SymmetricSparseIterate, ::Real = 1) = _symmetric_prox_error(f)
  end
end
