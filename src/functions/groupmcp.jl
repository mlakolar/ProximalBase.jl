####################################
#  group MCP  Σₖ p(‖x[Gₖ]‖₂; λ0 wₖ, a)
####################################

"""
    GroupMCP(λ0, groups; a = 3)
    GroupMCP(λ0, groups, w; a = 3)

Group minimax concave penalty `f(x) = Σₖ p(‖x[Gₖ]‖₂; λ₀wₖ, a)`, where, for
finite positive `λ₀wₖ` and `a`, `p(t; λ, a)` is the [`MCP`](@ref) penalty
`λt - t²/(2a)` for `t ≤ aλ` and `aλ²/2` above; the endpoint table of the
numerical contract gives the other cases. The weight multiplies the level, and
`a` is not rescaled. Default weights are `wₖ = √|Gₖ|`. Groups are nonempty,
disjoint lists or ranges of positive linear indices; entries in no group are
unpenalized. The level and weights are nonnegative and not `NaN`, and `a > 0`
(`a = Inf` gives the penalty of [`GroupNormL2`](@ref)). The constructors copy
`groups` and `w`. The type trait `isconvex` is false: with finite positive
parameters the penalty is not convex.

[`prox`](@ref) takes a nonnegative **scalar** step `γ`. For finite positive
`λ₀wₖ` and `γ` and a finite nonzero group norm `r = ‖u[Gₖ]‖₂`, it replaces `r`
by the scalar MCP proximal point of `r`: the group is kept, scaled toward zero
(`γ < a`), or set to `+0`. [`prox_block`](@ref) takes a compact vector in
`groups[k]` order and does not update an iterate.

# Examples
```jldoctest
julia> f = GroupMCP(1.0, [1:2], [1.0]);

julia> f([3.0, 4.0])  # above the kink aλ = 3: aλ²/2
1.5

julia> prox_block(f, [3.0, 4.0], 1, 1.0)
2-element Vector{Float64}:
 3.0
 4.0

julia> prox_block(f, [1.2, 1.6], 1, 1.0) ≈ [0.9, 1.2]
true

julia> prox_block(f, [0.6, 0.8], 1, 1.0)  # r = γλ gives +0
2-element Vector{Float64}:
 0.0
 0.0
```

# Edge cases
- Real, one-based arrays and floating-point destinations are required.
  Negative or `NaN` steps fail before writes.
- Exact in-place use (`y === x`, `out === u`) is supported; other overlaps
  detected by `Base.mightalias` are rejected.
- Zero-weight groups stay unchanged. Infinite weights, or an infinite level
  with positive weight, fix a group at `+0`, including at zero step.
- [`kkt_violation`](@ref) has the zero-group screen of `GroupNormL2`
  (`‖v[Gₖ]‖₂ - λ₀wₖ` for finite data) and `abs(v[i])` for entries in no group;
  at a real array point it throws an `ArgumentError` at a nonzero group and in
  the full-point form. A zero score does not imply a global minimizer.
- No `SymmetricSparseIterate`, `restrict` or per-group steps.

See the [GroupMCP numerical contract](@ref spec-groupmcp) for endpoint and
range rules.
"""
struct _GroupMCPOwned end

struct GroupMCP{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractFunction
  λ0::T
  a::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}   # group_of[i] = k for i ∈ Gₖ, 0 for an index in no group
  free::Vector{Int}       # the indices up to the largest in a group that are in none

  # The route of the outer constructors, which pass groups and weights that they own already.
  function GroupMCP{T, G}(::_GroupMCPOwned, λ0::T, a::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    _check_one_based_groups(groups)
    length(w) == length(groups) ||
      throw(DimensionMismatch("GroupMCP needs one weight per group: $(length(groups)) groups and $(length(w)) weights"))
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    a > 0 || throw(ArgumentError("a must be positive, got $a"))
    group_of = _group_of(groups)
    new{T, G}(λ0, a, groups, w, group_of, findall(iszero, group_of))
  end
  # The typed constructor copies each group, the list of groups and the weights, and checks the copies; the copied list
  # is converted to G.
  function GroupMCP{T, G}(λ0::T, a::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    _check_one_based_groups(groups)
    GroupMCP{T, G}(_GroupMCPOwned(), λ0, a, convert(G, map(copy, groups)), copy(w))
  end
end

function GroupMCP(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector; a::Real = 3)
  _check_one_based_groups(groups)
  _check_one_based(w)
  eltype(w) <: Real || _real_error(:GroupMCP)
  # checked before conversion, where a negative value can round to -0
  _checknonneg("λ0", λ0)
  _checknonneg("the weights", w)
  a > 0 || throw(ArgumentError("a must be positive, got $a"))
  T = float(promote_type(typeof(λ0), eltype(w), typeof(a)))
  G = _own_groups(groups)
  GroupMCP{T, typeof(G)}(_GroupMCPOwned(), convert(T, λ0), convert(T, a), G, Vector{T}(w))
end

function GroupMCP(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}; a::Real = 3)
  _check_one_based_groups(groups)
  T = float(promote_type(typeof(λ0), typeof(a)))
  # the root of the group size in at least Float64, converted once: a narrow T (Float16) can overflow at the size
  R = promote_type(T, Float64)
  GroupMCP(λ0, groups, T[convert(T, sqrt(R(length(G)))) for G in groups]; a)
end

isproximable(::Type{<:GroupMCP}) = true
isconvex(::Type{<:GroupMCP}) = false

# A BlockIterate with the groups of f, which must partition 1:p (the public constructor checks it).
BlockIterate(p::Integer, f::GroupMCP) = BlockIterate(Float64, p, f)
BlockIterate(::Type{T}, p::Integer, f::GroupMCP) where {T} = BlockIterate(T, p, f.groups)
BlockIterate(x::AbstractVector, f::GroupMCP) = BlockIterate(x, f.groups)

####################################
#  Checks
####################################

@inline _check_group_extent(f::GroupMCP, x) = (length(x) >= length(f.group_of) || _group_extent_error(f, x); nothing)
_checkparams(f::GroupMCP, x) = (_check_group_extent(f, x); true)
# a symmetric iterate is rejected by the oracles themselves, with their own message
_checkparams(::GroupMCP, ::SymmetricSparseIterate) = true

@noinline _gmcp_float_error(op, S) = throw(ArgumentError(
  "$(nameof(op)) with GroupMCP needs a floating-point destination, got element type $S"))
@noinline _gmcp_alias_error() = throw(ArgumentError("GroupMCP supports exact input/output aliasing (y === x) only"))

# The checks of GroupNormL2's array oracles, a floating-point destination (a scaled group would otherwise fail its
# conversion part way through the writes), and the alias check of the compact blocks
@inline function _gmcp_check_arrays(op, f, y::AbstractArray, x::AbstractArray, γ::Real)
  _check_group_arrays(f, y, x, γ)
  eltype(y) <: AbstractFloat || _gmcp_float_error(op, eltype(y))
  (y === x || !Base.mightalias(y, x)) || _gmcp_alias_error()
  nothing
end

####################################
#  Scalar parts
####################################

# λ0 w as a normalized significand in [1/2, 1) and an exponent: the product of the significands of λ0 and w rounded
# once in R, with their exponents summed apart from it (positive finite factors)
@inline function _gmcp_level_parts(λ0, w, ::Type{R}) where {R}
  m0, e0 = _frexp(convert(R, λ0))
  mw, ew = _frexp(convert(R, w))
  _parts(m0 * mw, e0 + ew)
end

# aλ²/2 from the parts of λ and a
@inline _gmcp_plateau_parts(ml, el, ma, ea) = (((ma * ml) * ml) / 2, ea + 2el)
@inline function _gmcp_plateau_parts(λ0, w, a, ::Type{R}) where {R}
  ml, el = _gmcp_level_parts(λ0, w, R)
  ma, ea = _frexp(convert(R, a))
  _gmcp_plateau_parts(ml, el, ma, ea)
end

# p(r; λ0 w, a) as parts m 2^e for r = ns 2^k > 0 and positive finite λ0, w and a. r ≥ aλ (compared on the parts,
# with the level as formed) gives the plateau. Below it the term is r (λ - r/(2a)), whose second factor is in [λ/2, λ]
# and is formed in units of 2^el without cancellation.
@inline function _gmcp_value_parts(λ0, w, a, ns::R, k::Integer) where {R}
  ml, el = _gmcp_level_parts(λ0, w, R)
  ma, ea = _frexp(convert(R, a))
  mt, et = _parts(ns, k)
  m, _ = _mcp_scaled_residual(mt, et, ma, ea, ml, el)
  m >= 0 && return _gmcp_plateau_parts(ml, el, ma, ea)
  c = ml - ldexp(mt / (2ma), et - ea - el)
  (mt * c, et + el)
end

# The plateau term of the ordinary paths, and the term at a norm r; the value and the fused value form them with these
# expressions.
@inline _gmcp_plateau(λ, kink) = λ * (kink / 2)
# Below the kink, r (λ - r/(2a)), with r/(2a) formed as r (inv(a)/2) when that factor is a normal number (it does not
# depend on the group, so a loop over the groups can form it once) and as (r/a)/2 otherwise (a shape near floatmin or
# floatmax); r/a < λ there, so the second factor stays in [λ/2, λ] up to rounding.
@inline function _gmcp_ordinary_term(r, λ, a, kink)
  r < kink || return _gmcp_plateau(λ, kink)
  h = inv(a) / 2
  c = (floatmin(h) <= h <= floatmax(h)) ? r * h : (r / a) / 2
  r * (λ - c)
end

# y[G] = q x[G] stored, possibly in place, and the sum of the squares of the stored entries
@inline function _gmcp_scale!(y, x, G, q::T) where {T}
  S = eltype(y)
  s = zero(T)
  @inbounds @simd for i in G
    v = convert(S, q * convert(T, x[i]))
    y[i] = v
    s += abs2(convert(T, v))
  end
  s
end

####################################
#  Group values (finite a)
####################################

# The term of one group in T and whether it counts for the underflow test of the sum. A term rounded from positive
# finite parameters at a nonzero group counts, also when it rounds to Inf; zero groups, unpenalized groups, a zero level,
# hard groups (an infinite weight or level) and NaN groups do not.
# Ordinary path: homogeneous Float32 or Float64 parameters, a normal level λ = λ0 w and kink aλ, and a sum of squares
# that _squares_normal accepts. The plateau applies when the computed norm is at least the rounded kink.
@inline function _gmcp_term(λ0, w, a, x, G, ::Type{T}) where {T}
  iszero(w) && return (zero(T), false)
  if T === typeof(λ0) && T <: Union{Float32, Float64}
    λ = λ0 * w
    kink = a * λ
    if (floatmin(T) <= λ <= floatmax(T)) & (floatmin(T) <= kink <= floatmax(T))
      s = _indexed_sumabs2(x, G, T)
      if isfinite(s) & _squares_normal(s, length(G), T)
        t = _gmcp_ordinary_term(sqrt(s), λ, a, kink)
        t >= 0 && return (t, true)      # a negative or NaN term would be a defect of the ordinary formula: recomputed
      elseif iszero(s) && _l2_all_zero(x, G, T)
        return (zero(T), false)
      end
    end
  end
  _gmcp_term_cold(λ0, w, a, x, G, T)
end

# Exceptional terms, in this order: an infinite weight, or an infinite level with a positive weight, gives NaN at a NaN
# entry, 0 at a zero group and Inf otherwise; a zero level gives 0; at positive finite parameters a NaN entry gives NaN,
# an infinite one the plateau, a zero group 0, and other groups the parts of the term from the scaled norm, rounded
# once to T.
@noinline function _gmcp_term_cold(λ0, w, a, x, G, ::Type{T}) where {T}
  hard = isinf(w) | isinf(λ0)
  !hard && iszero(λ0) && return (zero(T), false)
  R = _recovery_type(T, float(typeof(λ0)), float(eltype(x)))
  k, ns = _l2_classified_norm(x, G, R, !hard)
  k == _EXPONENT_NAN && return (T(NaN), false)
  k == _EXPONENT_ZERO && return (zero(T), false)
  hard && return (T(Inf), false)
  m, e = k == _EXPONENT_INF ? _gmcp_plateau_parts(λ0, w, a, R) : _gmcp_value_parts(λ0, w, a, ns, k)
  (convert(T, _scaled_product(m, one(R), e)), true)
end

# out of line: the term of a group that the fused loop recomputes, which keeps its sums out of the loop over the groups
@noinline _gmcp_term_outofline(λ0, w, a, x, G, ::Type{T}) where {T} = _gmcp_term(λ0, w, a, x, G, T)

# For a = Inf the terms of GroupNormL2 at the level λ0 wₖ; they count where 0 < λ0, wₖ < Inf, zero groups included.
@inline _gmcp_counted(λ0, w) = (zero(λ0) < λ0 < Inf) & (zero(w) < w < Inf)
@inline function _gmcp_term(f::GroupMCP, x, k::Integer, ::Type{T}) where {T}
  G = @inbounds f.groups[k]
  w = @inbounds f.w[k]
  isinf(f.a) && return (_l2_term(f.λ0, w, x, G, _L2_RECOMPUTE, T), _gmcp_counted(f.λ0, w))
  _gmcp_term(f.λ0, w, f.a, x, G, T)
end

# The ordinary sum v of the terms is returned when it is below floatmax(T)/2 and, for Float32 and Float64, at least n
# floatmin(T), with n the number of counted terms (the test of GroupNormL2's values, _sum_kept). Otherwise the terms are
# recomputed as parts and summed with their exponents (_ExpSum, in at least Float64); terms that rounded to Inf from
# positive finite parameters, and the plateau of a finite-shape group with an infinite entry, are recomputed too. The
# ordinary sum is kept for a NaN sum, a NaN group, a nonzero hard group (an infinite weight or level) and, for a = Inf,
# a group with an infinite entry.
@inline function _gmcp_value_checked(f::GroupMCP, x, v::T, n::Int) where {T}
  (_l2_ordinary(v, T) && (!_underflow_checked(T) || _sum_kept(v, n))) && return v
  _gmcp_value_recomputed(f, x, v)
end

@noinline function _gmcp_value_recomputed(f::GroupMCP, x, v::T)::T where {T}
  isnan(v) && return v
  λ0, a = f.λ0, f.a
  R = _recovery_type(T, float(typeof(λ0)), float(eltype(x)))
  acc = _expsum_start(R)
  for k in eachindex(f.groups)
    w = @inbounds f.w[k]
    iszero(w) && continue
    hard = isinf(w) | isinf(λ0)
    !hard && iszero(λ0) && continue
    e, ns = _l2_classified_norm(x, @inbounds(f.groups[k]), R, !hard)
    e == _EXPONENT_ZERO && continue
    (hard || e == _EXPONENT_NAN) && return v
    if isinf(a)
      e == _EXPONENT_INF && return v
      acc = _expsum_term(acc, λ0, w, ns, e, 0)
    else
      m, s = e == _EXPONENT_INF ? _gmcp_plateau_parts(λ0, w, a, R) : _gmcp_value_parts(λ0, w, a, ns, e)
      acc = _expsum_add(acc, m, s)
    end
  end
  _expsum_value(T, acc)
end

function (f::GroupMCP)(x::AbstractArray)
  _checkreal(f, x)
  Base.require_one_based_indexing(x)
  _check_group_extent(f, x)
  T = _valtype(x)
  v, n = zero(T), 0
  for k in eachindex(f.groups)
    t, c = _gmcp_term(f, x, k, T)
    v += t
    n += c
  end
  _gmcp_value_checked(f, x, v, n)
end

####################################
#  Group proximal map (finite a)
####################################

# y[G] = the proximal point of γ p(‖⋅‖₂; λ0 w, a) at x[G], possibly in place (y === x), for a finite a. Returns, in T,
# _L2_ZERO when the term of the stored group is zero (a zero group, or an unpenalized one), _L2_RECOMPUTE when it is to
# be recomputed from the stored group, and otherwise the term of a group kept or scaled on the ordinary path: the
# plateau of a kept group, and, with V (the fused value), for a scaled group the term at the norm of its stored entries,
# whose squares are summed as they are written.
# Ordinary path (O = true: Float32 or Float64 throughout, the step converted exactly): a normal level λ = λ0 w (rounded
# once) and kink aλ, a firm threshold γλ of at least _mcp_firm_product_min for γ < a or a finite step for γ ≥ a, and a
# sum of squares that _squares_normal accepts. The decisions are the scalar MCP comparisons at the computed norm r and
# the rounded level λ: zero when r ≤ γλ (the sign of the fused residual r - γλ), kept when r ≥ aλ, both exact; for
# γ ≥ a kept when r² > aγλ² (_mcp_keeps). A firm group is scaled by min(1, (r - γλ) (a/(a - γ))/r).
@inline function _gmcp_shrink!(y, x, G, λ0, w, a, γ::Real, ::Val{O}, ::Val{V} = Val(false)) where {O, V}
  T = float(eltype(y))
  if O
    λ = λ0 * w
    kink = a * λ
    if (floatmin(T) <= λ <= floatmax(T)) & (floatmin(T) <= kink <= floatmax(T))
      firm = γ < a
      if ifelse(firm, γ * λ >= _mcp_firm_product_min(T), γ < T(Inf))
        s = _indexed_sumabs2(x, G, T)
        if isfinite(s) & _squares_normal(s, length(G), T)
          r = sqrt(s)
          if firm
            res = fma(-γ, λ, r)
            res > 0 || return T(_l2_zero!(y, G))
            if r < kink || (r == kink && !_mcp_at_or_above_kink(r, a, λ))
              q = min(one(T), (res * (a / (a - γ))) / r)
              V || (_l2_scale!(y, x, G, q); return T(_L2_RECOMPUTE))
              sy = _gmcp_scale!(y, x, G, q)
              (isfinite(sy) & _squares_normal(sy, length(G), T)) || return T(_L2_RECOMPUTE)
              t = _gmcp_ordinary_term(sqrt(sy), λ, a, kink)
              return t >= 0 ? t : T(_L2_RECOMPUTE)
            end
          else
            _mcp_keeps(r, λ, a, γ) || return T(_l2_zero!(y, G))
          end
          _l2_copy!(y, x, G)
          return _gmcp_plateau(λ, kink)
        elseif iszero(s) && _l2_zero_group(x, G, T)
          return T(_l2_zero!(y, G))
        end
      end
    end
  end
  (iszero(w) | isinf(w) | isinf(λ0) | iszero(λ0) | iszero(γ) | isinf(γ)) && return T(_gmcp_fixed!(y, x, G, λ0, w, γ))
  T(_gmcp_shrink_values!(y, x, G, λ0, w, a, γ))
end

# The exits that the stored parameters decide, in this order: an unpenalized group (w = 0, copied, zero term), a group
# fixed at +0 (an infinite weight, or an infinite level with a positive weight), the identity of a zero level (zero term)
# and of a zero step, and +0 at an infinite step.
@noinline function _gmcp_fixed!(y, x, G, λ0, w, γ::Real)
  iszero(w) && (_l2_copy!(y, x, G); return _L2_ZERO)
  (isinf(w) || isinf(λ0)) && return _l2_zero!(y, G)
  iszero(λ0) && (_l2_copy!(y, x, G); return _L2_ZERO)
  iszero(γ) && return _l2_copy!(y, x, G)
  _l2_zero!(y, G)                      # isinf(γ)
end

# The values, at positive finite λ0, w and γ and a finite a. A NaN entry makes each output entry of the group NaN; an infinite entry (and
# no NaN) keeps the group; a zero group gives +0. Otherwise r = ns 2^k from the scaled norm in R, and the decisions are
# those of the ordinary path on the parts of r, λ0 w (its significands' product rounded once in R), a and γ, so that
# neither the level nor its products with a and γ are formed as one number: the signs of r - γλ and r - aλ in units of r
# (_mcp_scaled_residual), and for γ ≥ a the comparison of r² with aγλ² (_mcp_keeps_parts). A firm group is scaled by
# min(1, m (a/(a - γ))/mt), with m the residual r - γλ in the units 2^et of r = mt 2^et.
@noinline function _gmcp_shrink_values!(y, x, G, λ0, w, a, γ::Real)
  T = float(eltype(y))
  R = _recovery_type(_working_type(T, γ, λ0, w), float(eltype(x)))
  k, ns = _l2_classified_norm(x, G, R)
  k == _EXPONENT_NAN && return _l2_fill!(y, G, convert(eltype(y), NaN))
  k == _EXPONENT_INF && return _l2_copy!(y, x, G)
  k == _EXPONENT_ZERO && return _l2_zero!(y, G)
  mt, et = _parts(ns, k)
  ml, el = _gmcp_level_parts(λ0, w, R)
  ra, rg = convert(R, a), convert(R, γ)
  ma, ea = _frexp(ra)
  mg, eg = _frexp(rg)
  if rg < ra
    m, _ = _mcp_scaled_residual(mt, et, mg, eg, ml, el)
    m > 0 || return _l2_zero!(y, G)
    u, _ = _mcp_scaled_residual(mt, et, ma, ea, ml, el)
    if u < 0
      q = min(one(R), (m * (ra / (ra - rg))) / mt)
      S = eltype(y)
      @inbounds for i in G
        y[i] = convert(S, q * convert(R, x[i]))
      end
      return _L2_RECOMPUTE
    end
  elseif !_mcp_keeps_parts(mt, et, ml, el, ma, ea, mg, eg)
    return _l2_zero!(y, G)
  end
  _l2_copy!(y, x, G)
end

@inline _gmcp_homogeneous(::Type{T}, y, x) where {T} =
  T <: Union{Float32, Float64} && eltype(y) === T && eltype(x) === T

function _gmcp_prox_all!(y, f::GroupMCP, x, γ::Real, ::Val{V}) where {V}
  if y !== x       # the entries in no group are unpenalized
    @inbounds for i in f.free
      y[i] = x[i]
    end
    @inbounds for i in length(f.group_of)+1:length(x)
      y[i] = x[i]
    end
  end
  λ = f.λ0
  if isinf(f.a)
    # GroupNormL2 at the level λ0 wₖ, with its split between the parameter exits and the shrinkage
    if isinf(λ) | iszero(λ) | iszero(γ) | isinf(γ)
      return _gmcp_l2_loop!(_l2_shrink_fixed!, y, f, x, γ, Val(V))
    end
    return _gmcp_l2_loop!(_l2_shrink!, y, f, x, γ, Val(V))
  end
  T = typeof(λ)
  _gmcp_homogeneous(T, y, x) && T(γ) == γ && return _gmcp_prox_loop!(y, f, x, T(γ), Val(true), Val(V))
  _gmcp_prox_loop!(y, f, x, γ, Val(false), Val(V))
end

# The fused value takes the term of each group as the value of the stored output does: zero for the _L2_ZERO code, the
# term returned by the kernel for a group kept or scaled on the ordinary path (for a scaled group the sum of squares of
# the stored entries can associate differently from that of the value), and otherwise the term of the stored group.
@noinline function _gmcp_prox_loop!(y, f::GroupMCP, x, γ::Real, ::Val{O}, ::Val{V}) where {O, V}
  T = _valtype(y)
  v, n = zero(T), 0
  for k in eachindex(f.groups)
    G = @inbounds f.groups[k]
    w = @inbounds f.w[k]
    s = _gmcp_shrink!(y, x, G, f.λ0, w, f.a, γ, Val(O), Val(V))
    if V
      # the kernel returns the two codes or a term that is not negative (a term that would not be is recomputed);
      # any other value is added, so that it is not dropped
      if s == _L2_RECOMPUTE
        t, c = _gmcp_term_outofline(f.λ0, w, f.a, y, G, T)
        v += t
        n += c
      elseif s != _L2_ZERO
        v += s
        n += 1
      end
    end
  end
  V ? _gmcp_value_checked(f, y, v, n) : nothing
end

@noinline function _gmcp_l2_loop!(shrink::F, y, f::GroupMCP, x, γ::Real, ::Val{V}) where {F, V}
  T = _valtype(y)
  v, n = zero(T), 0
  for k in eachindex(f.groups)
    G = @inbounds f.groups[k]
    w = @inbounds f.w[k]
    s = shrink(y, x, G, f.λ0, w, γ)
    if V
      v += _l2_term(f.λ0, w, y, G, s == _L2_ZERO ? s : T(_L2_RECOMPUTE), T)
      n += _gmcp_counted(f.λ0, w)
    end
  end
  V ? _gmcp_value_checked(f, y, v, n) : nothing
end

prox!(y::AbstractArray, f::GroupMCP, x::AbstractArray, γ::Real) =
  (_gmcp_check_arrays(prox!, f, y, x, γ); _gmcp_prox_all!(y, f, x, γ, Val(false)); y)
value_and_prox!(y::AbstractArray, f::GroupMCP, x::AbstractArray, γ::Real) =
  (_gmcp_check_arrays(value_and_prox!, f, y, x, γ); _gmcp_prox_all!(y, f, x, γ, Val(true)))
# routes of wrappers such as Scaled, which preflight with _checkwrite first
_value_and_prox_unchecked!(y::AbstractArray, f::GroupMCP, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)

function prox_block!(out::AbstractVector, f::GroupMCP, u::AbstractVector, k::Integer, γ::Real)
  m = _check_group_block(out, f, u, k, γ)
  eltype(out) <: AbstractFloat || _gmcp_float_error(prox_block!, eltype(out))
  G = Base.OneTo(m)
  w = @inbounds f.w[k]
  if isinf(f.a)
    _l2_shrink!(out, u, G, f.λ0, w, γ, _l2_shrink_fixed!)
  else
    T = typeof(f.λ0)
    if _gmcp_homogeneous(T, out, u) && T(γ) == γ
      _gmcp_shrink!(out, u, G, f.λ0, w, f.a, T(γ), Val(true))
    else
      _gmcp_shrink!(out, u, G, f.λ0, w, f.a, γ, Val(false))
    end
  end
  out
end

# no proximal operator for a symmetric iterate (argument or output); nothing is written
prox!(::SymmetricSparseIterate, f::GroupMCP, x, ::Real) = _symmetric_prox_error(f)
prox!(::AbstractArray, f::GroupMCP, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
value_and_prox!(::SymmetricSparseIterate, f::GroupMCP, x, ::Real) = _symmetric_prox_error(f)
value_and_prox!(::AbstractArray, f::GroupMCP, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
prox(f::GroupMCP, ::SymmetricSparseIterate, ::Real = 1) = _symmetric_prox_error(f)

####################################
#  Optimality
####################################

kkt_violation(f::GroupMCP, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); _not_separable_error(f))

@noinline _gmcp_kkt_nonzero_error(k) = throw(ArgumentError(
  "kkt_violation for GroupMCP is not implemented at a nonzero group (group $k); it scores zero groups and entries in no group"))
@noinline _gmcp_kkt_point_error() = throw(ArgumentError(
  "kkt_violation(f, v, x) is not implemented for GroupMCP; use kkt_violation(f, v, x, k) at a zero group or an entry in no group"))

# The screen of a zero group, that of GroupNormL2 with the level and the weight passed as factors (‖v[G]‖₂ - λ0 wₖ for
# finite data and coefficient, with GroupNormL2's rules for NaN and infinite entries and coefficients); a nonzero group
# throws an ArgumentError.
@inline function _gmcp_zero_score(f::GroupMCP, g, x, k, ::Type{W}) where {W}
  G = @inbounds f.groups[k]
  _l2_zero_group(x, G, W) || _gmcp_kkt_nonzero_error(k)
  _l2_zero_score(f.λ0, @inbounds(f.w[k]), g, G, W)
end
# Group k of f is selected by the entry i of an array point (an entry in no group scores |gᵢ|) or by the group number k
# of a BlockIterate, whose layout is compared with that of f at group k, as for GroupNormL2.
function _kkt_violation(f::GroupMCP, g::AbstractArray, x::AbstractArray, i::Integer)
  _check_group_extent(f, x)
  checkbounds(x, i)
  W = float(promote_type(eltype(g), eltype(x)))
  k = i <= length(f.group_of) ? @inbounds(f.group_of[i]) : 0
  k == 0 && return abs(convert(W, g[i]))
  _gmcp_zero_score(f, g, x, k, W)
end
function _kkt_violation(f::GroupMCP, g::AbstractArray, x::BlockIterate, k::Integer)
  ncoordinates(x) == length(f.groups) || _block_count_error(x, f)
  B = blockindices(x, k)          # BoundsError for an invalid group
  B == @inbounds(f.groups[k]) || _block_layout_error(f, k)
  _gmcp_zero_score(f, g, x, k, float(promote_type(eltype(g), eltype(x))))
end

function kkt_violation(f::GroupMCP, v::AbstractArray, x::AbstractArray)
  eltype(v) <: Real || _real_error(:kkt_violation)
  eltype(x) <: Real || _real_error(:kkt_violation)
  _gmcp_kkt_point_error()
end
