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

@noinline _group_extent_error(f, x) = throw(DimensionMismatch(
  "the groups of GroupNormL2 reach index $(length(f.group_of)), but the argument has $(length(x)) entries"))
# every group index is a linear index of x: the loops over a group skip the bounds checks
@inline _check_group_extent(f::GroupNormL2, x) = (length(x) >= length(f.group_of) || _group_extent_error(f, x); nothing)
_checkparams(f::GroupNormL2, x) = (_check_group_extent(f, x); true)
# a symmetric iterate is rejected by the oracles themselves, with their own message
_checkparams(::GroupNormL2, ::SymmetricSparseIterate) = true

function _check_group_arrays(f::GroupNormL2, y::AbstractArray, x::AbstractArray, γ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  Base.require_one_based_indexing(y, x)
  _checksize(y, x)
  _check_group_extent(f, x)
  _check_l2_step(f, γ)
  nothing
end

@noinline _block_length_error(k, m, nu, nout) = throw(DimensionMismatch(
  "group $k of GroupNormL2 has $m entries, but the input has $nu and the output $nout"))
@noinline _block_alias_error() = throw(ArgumentError("prox_block! supports exact input/output aliasing (out === u) only"))

@inline function _check_group_block(out::AbstractVector, f::GroupNormL2, u::AbstractVector, k::Integer, γ::Real)
  (!(k isa Bool) && checkbounds(Bool, f.groups, k)) || throw(BoundsError(Base.OneTo(length(f.groups)), k))
  Base.require_one_based_indexing(out, u)
  (eltype(out) <: Real && eltype(u) <: Real) || _real_error(:prox_block!)
  m = @inbounds length(f.groups[k])
  (length(u) == m && length(out) == m) || _block_length_error(k, m, length(u), length(out))
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
# recomputed from the terms w ns 2^k, accumulated with their exponents (_ExpSum, in at least Float64),
# with λ0 applied once (_expsum_value). Exceptional cases keep s: a NaN s, a zero level (every term with
# a finite weight is 0, and an infinite weight at a nonzero group without NaN makes s Inf), and an infinite level or
# weight at a nonzero group or a NaN or ±Inf entry in a penalized group, which make s infinite or NaN
# already. So does an exactly zero argument.
# K counts every group, penalized or not. The scan for a zero argument runs only below K floatmin, and for Float32
# and Float64 it returns at its first block of 512 stored values that holds a nonzero one.
@inline function _group_value_checked(f::GroupNormL2, x, s::T) where {T}
  if _l2_ordinary(s, T)
    _underflow_checked(T) || return s
    _sum_kept(s, length(f.groups)) && return s
  end
  _group_value_unkept(f, x, s)
end
# out of line, so that the loop over the groups of the callers does not carry the scans
@noinline function _group_value_unkept(f::GroupNormL2, x, s::T)::T where {T}
  if _l2_ordinary(s, T)
    iszero(s) && _all_zero_early(_stored_values(x)) && return s
  end
  _group_value_recomputed(f, x, s)
end
@noinline function _group_value_recomputed(f::GroupNormL2, x, s::T)::T where {T}
  isnan(s) && return s
  λ = f.λ0
  (λ > 0 && isfinite(λ)) || return s
  R = _recovery_type(T, float(typeof(λ)), float(eltype(x)))
  acc = _expsum_start(R)
  for k in eachindex(f.groups)
    w = @inbounds f.w[k]
    iszero(w) && continue          # an unpenalized group adds nothing, whatever it holds
    next = _l2_expsum_add(acc, x, @inbounds(f.groups[k]), w)
    next === nothing && return s
    acc = next
  end
  _expsum_value(T, acc, λ)
end

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

_kkt_violation(f::GroupNormL2, g::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  throw(ArgumentError("GroupNormL2 groups linear indices; it does not apply to the coordinates of a SymmetricSparseIterate"))

@noinline _gradient_length_error(g, x) = throw(DimensionMismatch(
  "the gradient has length $(length(g)), but the argument has $(length(x)) entries"))

@noinline _group_point_error(f) = throw(ArgumentError(
  "kkt_violation of $(nameof(typeof(f))) is not supported at a group with a nonzero entry"))

# entry i of an array point: the group screen when the group of i is zero in x, |g[i]| when i is in no
# group (the penalty does not depend on xᵢ, and the partial derivative is 0); an ArgumentError when the group
# of i has a nonzero entry
function _kkt_violation(f::GroupNormL2, g::AbstractArray, x::AbstractArray, i::Integer)
  length(g) == length(x) || _gradient_length_error(g, x)
  _check_group_extent(f, x)
  checkbounds(x, i)
  k = i <= length(f.group_of) ? @inbounds(f.group_of[i]) : 0
  k == 0 && return abs(g[i])
  G = @inbounds f.groups[k]
  @inbounds for j in G
    iszero(x[j]) || _group_point_error(f)
  end
  _group_screen(f, k, g, G)
end

@noinline _block_layout_error(k) = throw(DimensionMismatch("group $k of the BlockIterate differs from group $k of GroupNormL2"))
@noinline _block_count_error(x, f) = throw(DimensionMismatch(
  "the BlockIterate has $(ncoordinates(x)) groups and GroupNormL2 has $(length(f.groups))"))

# group k of a BlockIterate: the group screen when its entries are zero, an ArgumentError when one is not;
# the two layouts are compared at group k alone
function _kkt_violation(f::GroupNormL2, g::AbstractArray, x::BlockIterate, k::Integer)
  ncoordinates(x) == length(f.groups) || _block_count_error(x, f)
  B = blockindices(x, k)          # BoundsError for an invalid group
  G = @inbounds f.groups[k]
  B == G || _block_layout_error(k)
  length(g) == length(x) || _gradient_length_error(g, x)
  @inbounds for j in G
    iszero(x[j]) || _group_point_error(f)
  end
  _group_screen(f, k, g, G)
end

# ‖g[G]‖₂ - λ0 w of group k, with the level and the weight of the group
@inline _group_screen(f::GroupNormL2, k::Integer, g, G) = _l2_screen(f.λ0, @inbounds(f.w[k]), g, G)

# no proximal operator for a symmetric iterate (argument or output); nothing is written
for F in (:NormL2, :GroupNormL2)
  @eval begin
    prox!(::SymmetricSparseIterate, f::$F, x, ::Real) = _symmetric_prox_error(f)
    prox!(::AbstractArray, f::$F, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
    value_and_prox!(::SymmetricSparseIterate, f::$F, x, ::Real) = _symmetric_prox_error(f)
    value_and_prox!(::AbstractArray, f::$F, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
    prox(f::$F, ::SymmetricSparseIterate, ::Real = 1) = _symmetric_prox_error(f)
  end
end
