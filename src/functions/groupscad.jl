####################################
#  group SCAD  Σₖ p(‖x[Gₖ]‖₂; λ0 wₖ, a)
####################################

"""
    GroupSCAD(λ0, groups; a = 3.7)
    GroupSCAD(λ0, groups, w; a = 3.7)

Group smoothly clipped absolute deviation penalty `f(x) = Σₖ p(‖x[Gₖ]‖₂; λₖ, a)`,
where `λₖ` is the effective coefficient of group `k` defined by the endpoint table
of the [numerical contract](@ref spec-groupscad-endpoints) (`λ₀wₖ` for finite
positive `λ₀` and `wₖ`) and, for finite positive `λₖ` and `a`, `p(t; λ, a)` is the
[`SCAD`](@ref) penalty `λt` for `t ≤ λ`, `λt - (t-λ)²/(2(a-1))` for `λ < t ≤ aλ`
and `(a+1)λ²/2` above. The shape `a` is not rescaled by the weights. Default
weights are `wₖ = √|Gₖ|`. Groups are nonempty, disjoint lists or ranges of
positive linear indices; entries in no group are unpenalized. The level and
weights are nonnegative and not `NaN`, and `a > 2` (`a = Inf` is admitted). The
constructors above copy `w`, the list of groups and each group that is not a
range; a range group can be kept as given (when every group is a range of one
concrete type) and must not be mutated afterwards. The type trait `isconvex` is
false.

The type of the function is `float(promote_type(typeof(λ0), eltype(w)))`, or
`float(typeof(λ0))` without weights, and `a` is converted to it: the default
`a = 3.7` does not promote a `Float32` function. A converted shape that is
not above 2 is refused, and so is a finite supplied shape that converts to `Inf`.

[`prox`](@ref) takes a nonnegative **scalar** step `γ`. For finite positive
`λₖ`, `a` and `γ` and a finite nonzero group norm `r = ‖u[Gₖ]‖₂`, it replaces
`r` by a global minimizer of the scalar SCAD proximal problem at `r` (the smaller
magnitude at a tie, `+0` at a zero minimizer): the group is kept, scaled toward
zero, or set to `+0`. At `a = Inf` each group follows the rules of
[`GroupNormL2`](@ref). [`prox_block`](@ref) takes a compact vector in `groups[k]`
order and does not update an iterate.

# Examples
```jldoctest
julia> f = GroupSCAD(1.0, [1:2], [1.0]);

julia> f([3.0, 4.0]) ≈ 2.35  # above the kink aλ = 3.7: (a+1)λ²/2
true

julia> prox_block(f, [3.0, 4.0], 1, 1.0)
2-element Vector{Float64}:
 3.0
 4.0

julia> prox_block(f, [0.9, 1.2], 1, 1.0) ≈ [0.3, 0.4]
true

julia> prox_block(f, [1.8, 2.4], 1, 1.0) ≈ [1.552941176470588, 2.070588235294118]
true

julia> prox_block(f, [0.6, 0.8], 1, 1.0)  # r = γλ gives +0
2-element Vector{Float64}:
 0.0
 0.0
```

# Edge cases
- Real, one-based arrays and destinations of a concrete floating-point element type are required.
  Negative or `NaN` steps fail before writes.
- Exact in-place use (`y === x`, `out === u`) is supported; other overlaps
  detected by `Base.mightalias` are rejected.
- A group with zero weight is unpenalized (unchanged, value 0) at every level.
  An infinite weight, or an infinite level with positive weight, fixes a group
  at `+0`, including at zero step.
- [`kkt_violation`](@ref) has the zero-group screen of `GroupNormL2` and
  `abs(v[i])` for entries in no group; at a real array point it throws an
  `ArgumentError` at a nonzero group and in the full-point form. A zero score
  is a first-order condition and does not imply a global minimizer.
- The typed constructor `GroupSCAD{T, G}(λ0, a, groups, w)` stores `groups` and
  `w` without copying them: do not mutate them afterwards.
- No `SymmetricSparseIterate`, `restrict` or per-group steps.

See the [GroupSCAD numerical contract](@ref spec-groupscad) for endpoint and
range rules.
"""
struct GroupSCAD{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractFunction
  λ0::T
  a::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}   # group_of[i] = k for i ∈ Gₖ, 0 for an index in no group
  free::Vector{Int}       # the indices up to the largest in a group that are in none

  # The typed constructor stores groups and w as given (the outer constructors pass copied weights, a copied list and
  # copies of the groups that are not ranges); the caller must not mutate them afterwards.
  function GroupSCAD{T, G}(λ0::T, a::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    _check_one_based_groups(groups)
    length(w) == length(groups) ||
      throw(DimensionMismatch("GroupSCAD needs one weight per group: $(length(groups)) groups and $(length(w)) weights"))
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    a > 2 || throw(ArgumentError("a must exceed 2, got $a"))
    group_of = _group_of(groups)
    new{T, G}(λ0, a, groups, w, group_of, findall(iszero, group_of))
  end
end

function GroupSCAD(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector; a::Real = 3.7)
  _check_one_based_groups(groups)
  _check_one_based(w)
  eltype(w) <: Real || _real_error(:GroupSCAD)
  # checked before conversion, where a negative value can round to -0
  _checknonneg("λ0", λ0)
  _checknonneg("the weights", w)
  a > 2 || throw(ArgumentError("a must exceed 2, got $a"))
  T = float(promote_type(typeof(λ0), eltype(w)))
  A = convert(T, a)
  A > 2 || throw(ArgumentError("a must exceed 2 in $T, got $a, which converts to $A"))
  (isinf(A) && !isinf(a)) && throw(ArgumentError("the finite a = $a overflows $T"))
  G = _own_groups(groups)
  GroupSCAD{T, typeof(G)}(convert(T, λ0), A, G, Vector{T}(w))
end

function GroupSCAD(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}; a::Real = 3.7)
  _check_one_based_groups(groups)
  T = float(typeof(λ0))
  # the root of the group size in at least Float64, converted once: a narrow T (Float16) can overflow at the size
  R = promote_type(T, Float64)
  GroupSCAD(λ0, groups, T[convert(T, sqrt(R(length(G)))) for G in groups]; a)
end

isproximable(::Type{<:GroupSCAD}) = true
isconvex(::Type{<:GroupSCAD}) = false

# A BlockIterate with the groups of f, which must partition 1:p (the public constructor checks it).
BlockIterate(p::Integer, f::GroupSCAD) = BlockIterate(Float64, p, f)
BlockIterate(::Type{T}, p::Integer, f::GroupSCAD) where {T} = BlockIterate(T, p, f.groups)
BlockIterate(x::AbstractVector, f::GroupSCAD) = BlockIterate(x, f.groups)

####################################
#  Checks
####################################

@inline _check_group_extent(f::GroupSCAD, x) = (length(x) >= length(f.group_of) || _group_extent_error(f, x); nothing)
_checkparams(f::GroupSCAD, x) = (_check_group_extent(f, x); true)
# a symmetric iterate is rejected by the scalar-step proximal oracles themselves, with their own message
_checkparams(::GroupSCAD, ::SymmetricSparseIterate) = true

@noinline _gscad_float_error(op, S) = throw(ArgumentError(
  "$(nameof(op)) with GroupSCAD needs a destination with a concrete floating-point element type, got $S"))
@inline _gscad_concrete_float(::Type{S}) where {S} = isconcretetype(S) && S <: AbstractFloat
@noinline _gscad_alias_error() = throw(ArgumentError("GroupSCAD supports exact input/output aliasing (y === x) only"))

# The checks of GroupNormL2's array oracles, a destination of a concrete floating-point element type (an integer or
# abstract element type could fail the conversion of a scaled group part way through the writes), and the alias check
# of the compact blocks
@inline function _gscad_check_arrays(op, f, y::AbstractArray, x::AbstractArray, γ::Real)
  _check_group_arrays(f, y, x, γ)
  _gscad_concrete_float(eltype(y)) || _gscad_float_error(op, eltype(y))
  (y === x || !Base.mightalias(y, x)) || _gscad_alias_error()
  nothing
end

####################################
#  Scalar parts
####################################

# The smallest level, in units of its type, that the ordinary paths accept (the guard of the scalar kernel)
@inline _gscad_level_min(::Type{T}) where {T} = floatmin(T) / eps(T)

# (a+1)λ²/2 from the parts of λ, as parts, for finite a
@inline function _gscad_plateau_parts(λ0, w, a, ::Type{R}) where {R}
  ml, el = _gmcp_level_parts(λ0, w, R)
  _parts(((convert(R, a) + 1) / 2 * ml) * ml, 2el)
end

# Whether r = mt 2^et is at most λ = ml 2^el, compared on the parts
@inline _gscad_at_most_level(mt, et, ml, el) = et < el || (et == el && mt <= ml)

# p(r; λ0 w, a) as parts m 2^e for r = ns 2^k > 0, positive finite λ0 and w and finite a. With r = mt 2^et and
# λ = ml 2^el:
#  * r ≤ λ (decided on the parts): the product of the significands, mt ml 2^(et+el);
#  * the exponent et - el above exponent(floatmax(R)) + 1: r/λ exceeds 2^(exponent(floatmax(R)) + 1), hence a, and the
#    term is the plateau (a+1) ml² 2^(2el)/2;
#  * otherwise p(r′; ml, a) 2^(2el) with r′ = r/2^el = ldexp(mt, et - el), a normal number in [1/2, floatmax(R)].
@inline function _gscad_value_parts(λ0, w, a, ns::R, k::Integer) where {R}
  ml, el = _gmcp_level_parts(λ0, w, R)
  mt, et = _parts(ns, k)
  _gscad_at_most_level(mt, et, ml, el) && return _parts(mt * ml, et + el)
  d = et - el
  d > _max_exponent(R) + 1 && return _gscad_plateau_parts(λ0, w, a, R)
  _parts(_scad_value(ml, convert(R, a), ldexp(mt, d)), 2el)
end

####################################
#  Group values (finite a)
####################################

# The term of one group in T and whether it counts for the underflow test of the sum. A term rounded from positive
# finite parameters at a nonzero group counts, also when it rounds to Inf; zero groups, unpenalized groups, a zero level,
# hard groups (an infinite weight or level) and NaN groups do not.
# Ordinary path: homogeneous Float32 or Float64 parameters, a normal level λ = λ0 w with a finite kink aλ, and a sum of
# squares that is finite and that _squares_normal accepts. The term is that of the scalar SCAD value at the computed norm.
@inline function _gscad_term(λ0, w, a, x, G, ::Type{T}) where {T}
  iszero(w) && return (zero(T), false)
  if T === typeof(λ0) && T <: Union{Float32, Float64}
    λ = λ0 * w
    if (floatmin(T) <= λ <= floatmax(T)) & isfinite(a * λ)
      s = _indexed_sumabs2(x, G, T)
      if isfinite(s) & _squares_normal(s, length(G), T)
        return (_scad_value(λ, a, sqrt(s)), true)
      elseif iszero(s) && _l2_all_zero(x, G, T)
        return (zero(T), false)
      end
    end
  end
  _gscad_term_cold(λ0, w, a, x, G, T)
end

# Exceptional terms, in this order: an infinite weight, or an infinite level with a positive weight, gives NaN at a NaN
# entry, 0 at a zero group and Inf otherwise; a zero level gives 0; at positive finite parameters a NaN entry gives NaN,
# an infinite one the plateau, a zero group 0, and other groups the parts of the term from the scaled norm, rounded
# once to T.
@noinline function _gscad_term_cold(λ0, w, a, x, G, ::Type{T}) where {T}
  hard = isinf(w) | isinf(λ0)
  !hard && iszero(λ0) && return (zero(T), false)
  R = _recovery_type(T, float(typeof(λ0)), float(eltype(x)))
  k, ns = _l2_classified_norm(x, G, R, !hard)
  k == _EXPONENT_NAN && return (T(NaN), false)
  k == _EXPONENT_ZERO && return (zero(T), false)
  hard && return (T(Inf), false)
  m, e = k == _EXPONENT_INF ? _gscad_plateau_parts(λ0, w, a, R) : _gscad_value_parts(λ0, w, a, ns, k)
  (convert(T, _scaled_product(m, one(R), e)), true)
end

# out of line: the term of a group that the fused loop recomputes, which keeps its sums out of the loop over the groups
@noinline _gscad_term_outofline(λ0, w, a, x, G, ::Type{T}) where {T} = _gscad_term(λ0, w, a, x, G, T)

# For a = Inf the terms of GroupNormL2 at the level λ0 wₖ; they count where 0 < λ0, wₖ < Inf, zero groups included.
@inline function _gscad_term(f::GroupSCAD, x, k::Integer, ::Type{T}) where {T}
  G = @inbounds f.groups[k]
  w = @inbounds f.w[k]
  isinf(f.a) && return (_l2_term(f.λ0, w, x, G, _L2_RECOMPUTE, T), _gmcp_counted(f.λ0, w))
  _gscad_term(f.λ0, w, f.a, x, G, T)
end

# The ordinary sum v of the terms is returned when it is below floatmax(T)/2 and, for Float32 and Float64, at least n
# floatmin(T), with n the number of counted terms (the test of GroupNormL2's values, _sum_kept). Otherwise the terms are
# recomputed as parts and summed with their exponents (_ExpSum, in at least Float64). The recomputation skips groups with
# a zero weight, and with a zero level and a finite weight, before it reads their entries. Of the remaining groups, a
# NaN entry, a nonzero hard group (an infinite weight, or an infinite level with a positive weight) and, for a = Inf, an
# infinite entry keep the ordinary sum, as does a NaN sum.
@inline function _gscad_value_checked(f::GroupSCAD, x, v::T, n::Int) where {T}
  (_l2_ordinary(v, T) && (!_underflow_checked(T) || _sum_kept(v, n))) && return v
  _gscad_value_recomputed(f, x, v)
end

@noinline function _gscad_value_recomputed(f::GroupSCAD, x, v::T)::T where {T}
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
      m, s = e == _EXPONENT_INF ? _gscad_plateau_parts(λ0, w, a, R) : _gscad_value_parts(λ0, w, a, ns, e)
      acc = _expsum_add(acc, m, s)
    end
  end
  _expsum_value(T, acc)
end

function (f::GroupSCAD)(x::AbstractArray)
  _checkreal(f, x)
  Base.require_one_based_indexing(x)
  _check_group_extent(f, x)
  T = _valtype(x)
  v, n = zero(T), 0
  for k in eachindex(f.groups)
    t, c = _gscad_term(f, x, k, T)
    v += t
    n += c
  end
  _gscad_value_checked(f, x, v, n)
end

####################################
#  Group proximal map (finite a)
####################################

# y[G] = the proximal point of γ p(‖⋅‖₂; λ0 w, a) at x[G], possibly in place (y === x), for a finite a. Returns, in T,
# _L2_ZERO when it certifies that the term of the stored group is zero (a group set to zero, or an unpenalized one),
# _L2_RECOMPUTE when the term is to be recomputed from the stored group, and otherwise, with V (the fused value), the term
# of the group on the ordinary path: for a copied group at the norm r that was computed (the sum of squares is that of
# the stored group), and for a scaled group at the norm of its stored entries, whose squares are summed as they are
# written.
# Ordinary path (O = true: Float32 or Float64 throughout, the step equal to its conversion to T): a step with 0 < γ < Inf, a level
# λ = λ0 w (rounded once) of at least floatmin/eps with a finite kink aλ, and a finite sum of squares that _squares_normal
# accepts. t is the scalar kernel at the computed norm r and the rounded level λ; the group is copied for t == r, set to
# +0 for t == 0, and otherwise scaled by t / r.
@inline function _gscad_shrink!(y, x, G, λ0, w, a, γ::Real, ::Val{O}, ::Val{V} = Val(false)) where {O, V}
  T = float(eltype(y))
  if O
    if zero(T) < γ < T(Inf)
      λ = λ0 * w
      if (λ >= _gscad_level_min(T)) & isfinite(a * λ)
        s = _indexed_sumabs2(x, G, T)
        if isfinite(s) & _squares_normal(s, length(G), T)
          r = sqrt(s)
          t = _scad_magnitude(r, λ, a, γ)
          if t == r
            _l2_copy!(y, x, G)
            return V ? _scad_value(λ, a, r) : T(_L2_RECOMPUTE)
          elseif iszero(t)
            return T(_l2_zero!(y, G))
          end
          q = t / r
          V || (_l2_scale!(y, x, G, q); return T(_L2_RECOMPUTE))
          sy = _gmcp_scale!(y, x, G, q)
          (isfinite(sy) & _squares_normal(sy, length(G), T)) || return T(_L2_RECOMPUTE)
          return _scad_value(λ, a, sqrt(sy))
        elseif iszero(s) && _l2_zero_group(x, G, T)
          return T(_l2_zero!(y, G))
        end
      end
    end
  end
  (iszero(w) | isinf(w) | isinf(λ0) | iszero(λ0) | iszero(γ) | isinf(γ)) && return T(_gscad_fixed!(y, x, G, λ0, w, γ))
  T(_gscad_shrink_values!(y, x, G, λ0, w, a, γ))
end

# The exits that the stored parameters decide, in this order: an unpenalized group (w = 0, copied, zero term), a group
# fixed at +0 (an infinite weight, or an infinite level with a positive weight), the identity of a zero level (zero term)
# and of a zero step, and +0 at an infinite step.
@noinline function _gscad_fixed!(y, x, G, λ0, w, γ::Real)
  iszero(w) && (_l2_copy!(y, x, G); return _L2_ZERO)
  (isinf(w) || isinf(λ0)) && return _l2_zero!(y, G)
  iszero(λ0) && (_l2_copy!(y, x, G); return _L2_ZERO)
  iszero(γ) && return _l2_copy!(y, x, G)
  _l2_zero!(y, G)                      # isinf(γ)
end

# y[G] = q x[G] with q in the recovery type R
@inline function _gscad_scale_values!(y, x, G, q::R) where {R}
  S = eltype(y)
  @inbounds for i in G
    y[i] = convert(S, q * convert(R, x[i]))
  end
  _L2_RECOMPUTE
end

# The values, at positive finite λ0, w and γ and a finite a. A NaN entry makes each output entry of the group NaN; an
# infinite entry (and no NaN) keeps the group; a zero group gives +0. Otherwise r = ns 2^k from the scaled norm in R, with
# the parts r = mt 2^et and λ = ml 2^el (the significands of λ0 and w multiplied and rounded once in R), so that the
# level is not formed as one number:
#  * r ≤ λ (decided on the parts): the first piece, soft thresholding. The residual r - γλ in units of 2^et comes from
#    _mcp_scaled_residual; the group is set to +0 when it is not positive and otherwise scaled by m/mt;
#  * et - el above exponent(floatmax(R)) + 1: r/λ exceeds 2^(exponent(floatmax(R)) + 1), which is above a and 1 + γ for the
#    finite a and γ of R, and the group is copied;
#  * otherwise r′ = ldexp(mt, et - el) is a normal number in [1/2, floatmax(R)] and t′ is the scalar kernel at (r′, ml, a, γ),
#    the problem scaled by 2^-el; the group is copied for t′ == r′, set to +0 for t′ == 0 and otherwise scaled by t′ / r′.
# The decisions are those of the scalar kernel on the computed norm and the level as formed. R is at least Float64. For
# R = Float64, r′ ≥ 1/2 and ml ≥ 1/2 are above floatmin(R)/eps(R), where the small-magnitude branch of _scad_magnitude
# starts; a wider R has only the generic _scad_magnitude, which is _scad_prox. _scad_prox is called directly.
@noinline function _gscad_shrink_values!(y, x, G, λ0, w, a, γ::Real)
  T = float(eltype(y))
  R = _recovery_type(_working_type(T, γ, λ0, w), float(eltype(x)))
  k, ns = _l2_classified_norm(x, G, R)
  k == _EXPONENT_NAN && return _l2_fill!(y, G, convert(eltype(y), NaN))
  k == _EXPONENT_INF && return _l2_copy!(y, x, G)
  k == _EXPONENT_ZERO && return _l2_zero!(y, G)
  mt, et = _parts(ns, k)
  ml, el = _gmcp_level_parts(λ0, w, R)
  rg = convert(R, γ)
  if _gscad_at_most_level(mt, et, ml, el)
    mg, eg = _frexp(rg)
    m, _ = _mcp_scaled_residual(mt, et, mg, eg, ml, el)
    m > 0 || return _l2_zero!(y, G)
    q = m / mt
    q < one(R) || return _l2_copy!(y, x, G)
    return _gscad_scale_values!(y, x, G, q)
  end
  d = et - el
  d > _max_exponent(R) + 1 && return _l2_copy!(y, x, G)
  r′ = ldexp(mt, d)
  t′ = _scad_prox(r′, ml, convert(R, a), rg)
  t′ == r′ && return _l2_copy!(y, x, G)
  iszero(t′) && return _l2_zero!(y, G)
  _gscad_scale_values!(y, x, G, t′ / r′)
end

@inline _gscad_homogeneous(::Type{T}, y, x) where {T} =
  T <: Union{Float32, Float64} && eltype(y) === T && eltype(x) === T

function _gscad_prox_all!(y, f::GroupSCAD, x, γ::Real, ::Val{V}) where {V}
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
      return _gscad_l2_loop!(_l2_shrink_fixed!, y, f, x, γ, Val(V))
    end
    return _gscad_l2_loop!(_l2_shrink!, y, f, x, γ, Val(V))
  end
  T = typeof(λ)
  _gscad_homogeneous(T, y, x) && T(γ) == γ && return _gscad_prox_loop!(y, f, x, T(γ), Val(true), Val(V))
  _gscad_prox_loop!(y, f, x, γ, Val(false), Val(V))
end

# The fused value takes the term of each group as the value of the stored output does: zero for the _L2_ZERO code, the
# term returned by the kernel for a group copied or scaled on the ordinary path (for a scaled group the sum of squares of
# the stored entries can associate differently from that of the value), and otherwise the term of the stored group.
@noinline function _gscad_prox_loop!(y, f::GroupSCAD, x, γ::Real, ::Val{O}, ::Val{V}) where {O, V}
  T = _valtype(y)
  v, n = zero(T), 0
  for k in eachindex(f.groups)
    G = @inbounds f.groups[k]
    w = @inbounds f.w[k]
    s = _gscad_shrink!(y, x, G, f.λ0, w, f.a, γ, Val(O), Val(V))
    if V
      # the kernel returns the two codes or a term that is not negative (_scad_value at a finite positive level); any
      # other value is added, so that it is not dropped
      if s == _L2_RECOMPUTE
        t, c = _gscad_term_outofline(f.λ0, w, f.a, y, G, T)
        v += t
        n += c
      elseif s != _L2_ZERO
        v += s
        n += 1
      end
    end
  end
  V ? _gscad_value_checked(f, y, v, n) : nothing
end

@noinline function _gscad_l2_loop!(shrink::F, y, f::GroupSCAD, x, γ::Real, ::Val{V}) where {F, V}
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
  V ? _gscad_value_checked(f, y, v, n) : nothing
end

prox!(y::AbstractArray, f::GroupSCAD, x::AbstractArray, γ::Real) =
  (_gscad_check_arrays(prox!, f, y, x, γ); _gscad_prox_all!(y, f, x, γ, Val(false)); y)
value_and_prox!(y::AbstractArray, f::GroupSCAD, x::AbstractArray, γ::Real) =
  (_gscad_check_arrays(value_and_prox!, f, y, x, γ); _gscad_prox_all!(y, f, x, γ, Val(true)))
# routes of wrappers such as Scaled, which preflight with _checkwrite first
_value_and_prox_unchecked!(y::AbstractArray, f::GroupSCAD, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)

function prox_block!(out::AbstractVector, f::GroupSCAD, u::AbstractVector, k::Integer, γ::Real)
  m = _check_group_block(out, f, u, k, γ)
  _gscad_concrete_float(eltype(out)) || _gscad_float_error(prox_block!, eltype(out))
  G = Base.OneTo(m)
  w = @inbounds f.w[k]
  if isinf(f.a)
    _l2_shrink!(out, u, G, f.λ0, w, γ, _l2_shrink_fixed!)
  else
    T = typeof(f.λ0)
    if _gscad_homogeneous(T, out, u) && T(γ) == γ
      _gscad_shrink!(out, u, G, f.λ0, w, f.a, T(γ), Val(true))
    else
      _gscad_shrink!(out, u, G, f.λ0, w, f.a, γ, Val(false))
    end
  end
  out
end

# no proximal operator for a symmetric iterate (argument or output); nothing is written
prox!(::SymmetricSparseIterate, f::GroupSCAD, x, ::Real) = _symmetric_prox_error(f)
prox!(::AbstractArray, f::GroupSCAD, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
value_and_prox!(::SymmetricSparseIterate, f::GroupSCAD, x, ::Real) = _symmetric_prox_error(f)
value_and_prox!(::AbstractArray, f::GroupSCAD, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
prox(f::GroupSCAD, ::SymmetricSparseIterate, ::Real = 1) = _symmetric_prox_error(f)

####################################
#  Optimality
####################################

kkt_violation(f::GroupSCAD, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); _not_separable_error(f))

@noinline _gscad_kkt_nonzero_error(k) = throw(ArgumentError(
  "kkt_violation for GroupSCAD is not implemented at a nonzero group (group $k); it scores zero groups and entries in no group"))
@noinline _gscad_kkt_point_error() = throw(ArgumentError(
  "kkt_violation(f, v, x) is not implemented for GroupSCAD; use kkt_violation(f, v, x, k) at a zero group or an entry in no group"))

# The screen of a zero group, that of GroupNormL2 with the level and the weight passed as factors (‖v[G]‖₂ - λ0 wₖ for
# finite v, λ0 and wₖ; ‖v[G]‖₂ for a zero weight; GroupNormL2's rules for NaN and infinite values otherwise); a nonzero
# group throws an ArgumentError.
@inline function _gscad_zero_score(f::GroupSCAD, g, x, k, ::Type{W}) where {W}
  G = @inbounds f.groups[k]
  _l2_zero_group(x, G, W) || _gscad_kkt_nonzero_error(k)
  _l2_zero_score(f.λ0, @inbounds(f.w[k]), g, G, W)
end
# Group k of f is selected by the entry i of an array point (an entry in no group scores |gᵢ|) or by the group number k
# of a BlockIterate, whose layout is compared with that of f at group k, as for GroupNormL2.
function _kkt_violation(f::GroupSCAD, g::AbstractArray, x::AbstractArray, i::Integer)
  _check_group_extent(f, x)
  checkbounds(x, i)
  W = float(promote_type(eltype(g), eltype(x)))
  k = i <= length(f.group_of) ? @inbounds(f.group_of[i]) : 0
  k == 0 && return abs(convert(W, g[i]))
  _gscad_zero_score(f, g, x, k, W)
end
function _kkt_violation(f::GroupSCAD, g::AbstractArray, x::BlockIterate, k::Integer)
  ncoordinates(x) == length(f.groups) || _block_count_error(x, f)
  B = blockindices(x, k)          # BoundsError for an invalid group
  B == @inbounds(f.groups[k]) || _block_layout_error(f, k)
  _gscad_zero_score(f, g, x, k, float(promote_type(eltype(g), eltype(x))))
end

function kkt_violation(f::GroupSCAD, v::AbstractArray, x::AbstractArray)
  eltype(v) <: Real || _real_error(:kkt_violation)
  eltype(x) <: Real || _real_error(:kkt_violation)
  _gscad_kkt_point_error()
end
