####################################
#  group ℓ2 norm  λ0 Σₖ wₖ ‖x[Gₖ]‖₂
####################################

"""
    GroupNormL2(λ0, groups)
    GroupNormL2(λ0, groups, w)

The group-lasso penalty `f(x) = λ0 Σₖ wₖ ‖x[Gₖ]‖₂` over disjoint groups
`Gₖ = groups[k]` of linear indices of `x`, of any sizes and in any order, such
as `1:5` or `[7, 2, 9]`. The weights default to `wₖ = √|Gₖ|`. A group of one
index with weight `w` is the term `λ0 w |xᵢ|` of `NormL1(λ0, w)`. Indices in no
group are not penalized: `prox!` copies them and `f(x)` ignores them.

`λ0` and the weights must be nonnegative (an `ArgumentError` otherwise, `NaN`
included), with one weight per group (`DimensionMismatch`). Groups must be
nonempty and disjoint, with positive indices (`ArgumentError`). The constructor
stores its own copies of `groups` (integer indices: when the element type of
`groups` is a concrete range type the ranges are kept, otherwise every group
becomes a `Vector{Int}`) and of `w`; the level and the weights are converted to the
promoted floating type of `λ0` and `w`. Changing the arrays passed in does not
change `f`.

The proximal operator shrinks each group towards zero in Euclidean norm:
`prox_{γf}(x)[Gₖ] = max(1 - γλ0wₖ/‖x[Gₖ]‖₂, 0) x[Gₖ]`. On arrays, `prox!`,
`prox` and `value_and_prox!` take a scalar step. For one group,
`prox_block!(out, f, u, k, γ)` writes the result for the compact vector
`u = x[Gₖ]` (in the order of `groups[k]`) into `out` and returns `out`.
`prox_block(f, u, k, γ)` allocates `out`. Neither writes to or activates an
iterate (`u` may be a view that reads one; `out` must not alias iterate
storage, as the block contract requires): to update a `BlockIterate` with the
same groups, write `out` with
`copyto!(blockview(x, k), out)`. `BlockIterate(x, f)` and `BlockIterate(p, f)`
build an iterate with the groups of `f`. The groups must then partition
`1:length(x)`.

`kkt_violation(f, g, x, k)` on a `BlockIterate` is `‖g[Gₖ]‖₂ - λ0 wₖ`, the
violation of the optimality of the zero group `k` for the smooth-part gradient
`g`. Group `k` of `x` must equal `groups[k]` (`DimensionMismatch` otherwise). On
an array point, `kkt_violation(f, g, x, i)` takes an entry `i`: the same
group screen when the group of `i` is zero in `x`, and `|g[i]|` when `i` is in
no group or its group has a nonzero entry. The three-argument form throws an
`ArgumentError`: the penalty is not separable. In the group screen the
coefficient `λ0 wₖ` follows the `NormL1` screen: `0` for a zero weight, also
with an infinite level, and `Inf` for an infinite weight or for an infinite
level with a positive weight. A norm or coefficient outside the range is
recomputed for finite gradients. When the screened group holds a `±Inf` entry
and no `NaN`, the screen is the IEEE `Inf - λ0 wₖ`, which is `NaN` when that
coefficient is infinite, also when it overflows from finite factors; a `NaN`
entry gives `NaN`.

Steps must be nonnegative: a negative or `NaN` step throws an `ArgumentError`
before anything is written, in `prox!`, `value_and_prox!` and `prox_block!`.
Shapes, the group extent (every group index at most `length(x)`), real element
types, one-based indexing, and for `prox_block!` the group index, the buffer
lengths and the aliasing of `out` and `u`, are checked before writes as well.
`out === u` is supported; any other overlap of `out` and `u` that
`Base.mightalias` reports throws an `ArgumentError`. On arrays, `prox!(x, f, x, γ)`
is supported and other sharing of memory is not, as for every function.

Exceptional parameters and values, for each group `k`:

- `wₖ = 0`: unpenalized, the group is returned unchanged (`NaN` and `±Inf`
  included), and its value is zero.
- `wₖ = Inf`, or `λ0 = Inf` with `wₖ > 0`: the group is fixed at `+0` at every
  nonnegative step, `γ = 0` included, and for every input, `NaN` and `±Inf`
  included. At `γ = 0` this is the limit `γ → 0+` with the function fixed, the
  projection onto the closure of its domain, as for `NormL1(Inf)` (#665).
  The value is zero at the zero group and `Inf` (or `NaN` at a `NaN` entry)
  elsewhere.
- `λ0 = 0` or `γ = 0` (finite `wₖ` and `λ0`): the identity.
- `γ = Inf` (positive finite `λ0` and `wₖ`): the group is fixed at `+0`.
- Positive finite `γ`, `λ0` and `wₖ`: a group with a `NaN` entry becomes all
  `NaN`; a group with an infinite entry and no `NaN` is returned unchanged,
  since its norm is infinite and the threshold `γλ0wₖ` is finite, also when
  that product overflows in floating point; a zero group becomes `+0`.

For finite inputs, a norm whose sum of squares overflows or underflows, and a
threshold `γλ0wₖ` outside the floating-point range, are recomputed in
power-of-two units, so that a representable proximal point is returned
accurately. Values follow the overflow rule in Overflow and underflow of
values: for Float32 and Float64 values, a sum of group terms below `K⋅floatmin`,
for `K` groups, is recomputed with each term's exponent, as for `NormL1` (#373), and
a value at or above `floatmax/2` is recomputed in power-of-two units, accumulated in
at least Float64; within a few eps of the overflow threshold the result may be
`floatmax` or `Inf`, regardless of which of them the exact value rounds to (#904). `value_and_prox!` returns
the value at the stored output.

The penalty does not apply to a `SymmetricSparseIterate`, and it has no
`restrict`, no per-group step and no compact block method under `Scaled`.

# Example
```julia
f = GroupNormL2(0.5, [1:3, [6, 4]])     # x[5] is not penalized
prox(f, randn(6), 1.0)
out = prox_block(f, [3.0, 4.0], 2, 1.0)  # (1 - 0.5√2/5) .* [3, 4]
```
"""
struct GroupNormL2{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractFunction
  λ0::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}   # group_of[i] = k for i ∈ Gₖ, 0 for an index in no group
  free::Vector{Int}       # the indices up to the largest in a group that are in none

  function GroupNormL2{T, G}(λ0::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    length(w) == length(groups) ||
      throw(DimensionMismatch("GroupNormL2 needs one weight per group: $(length(groups)) groups and $(length(w)) weights"))
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    group_of = _group_of(groups)
    new{T, G}(λ0, groups, w, group_of, findall(iszero, group_of))
  end
end
function GroupNormL2(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector)
  eltype(w) <: Real || _real_error(:GroupNormL2)
  T = float(promote_type(typeof(λ0), eltype(w)))
  G = _own_groups(groups)
  GroupNormL2{T, typeof(G)}(convert(T, λ0), G, Vector{T}(w))
end
GroupNormL2(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}) =
  GroupNormL2(λ0, groups, [sqrt(float(typeof(λ0))(length(G))) for G in groups])

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

# The arithmetic is that of numerical_kernels.jl (sums of squares, the scaled norm ns 2^k, the writes of a
# group), with the level λ0 and the weight wₖ of each group; the shrinkage, the terms of a value, the screen
# and their recovery are below. NormL2 has its own (norml2.jl).

# The ordinary sum s of the terms is returned unless it may have lost its accuracy or its class: a sum
# below K floatmin(T), for K groups, whose terms may have rounded into the subnormal range (#373), and a
# sum at or above floatmax(T)/2 (Inf included), whose class the terms' rounding may decide. Those are
# recomputed from the terms w ns 2^k, accumulated with their exponents (_ExpSum, in at least Float64),
# with λ0 applied once (_expsum_value). Exceptional cases keep s: a NaN s, a zero level (every term with
# a finite weight is 0, and an infinite weight at a nonzero group without NaN makes s Inf), and an infinite level or
# weight at a nonzero group or a NaN or ±Inf entry in a penalized group, which make s infinite or NaN
# already. So does an exactly zero argument.
@inline function _group_value_checked(f::GroupNormL2, x, s::T) where {T}
  if _l2_ordinary(s, T)
    _underflow_checked(T) || return s
    _sum_kept(s, length(f.groups)) && return s
    iszero(s) && _all_zero(_stored_values(x)) && return s
  end
  _group_value_recomputed(f, x, s)
end
# A zero weight adds nothing and is skipped in the loop itself: a sum of zero-weight groups is zero and lands here
@noinline function _group_value_recomputed(f::GroupNormL2, x, s::T)::T where {T}
  isnan(s) && return s
  λ = f.λ0
  (λ > 0 && isfinite(λ)) || return s
  R = _recovery_type(T, float(typeof(λ)), float(eltype(x)))
  acc = _expsum_start(R)
  for k in eachindex(f.groups)
    w = @inbounds f.w[k]
    iszero(w) && continue
    # a NaN or ±Inf entry, or an infinite weight at a nonzero group, make the ordinary sum infinite or NaN
    e, ns = _l2_classified_norm(x, @inbounds(f.groups[k]), R, !isinf(w))
    e >= _EXPONENT_INF && return s
    e == _EXPONENT_ZERO && continue
    isinf(w) && return s
    acc = _expsum_product(acc, w, ns, e)
  end
  _expsum_value(T, acc, λ)
end

# The term λw‖x[G]‖₂ of a value in T, from x[G]. The ordinary path needs a normal λw, a normal sum of squares and a
# value below floatmax(T)/2; the exactly zero group gives 0 at every level; everything else is recomputed.
@inline function _group_term(λ, w, x, G, ::Type{T}) where {T}
  iszero(w) && return zero(T)
  R = _working_type(T, λ, w)
  λw = convert(R, λ) * convert(R, w)
  if floatmin(R) <= λw <= floatmax(R)
    q = _indexed_sumabs2(x, G, T)
    if isfinite(q) & _squares_normal(q, length(G), T)
      v = convert(T, λw * sqrt(convert(R, q)))
      _l2_ordinary(v, T) && return v
    elseif iszero(q) && _l2_all_zero(x, G, T)
      return zero(T)
    end
  end
  _group_term_cold(λ, w, x, G, T)
end

# The term of the fused value_and_prox!, after the shrinkage of the group: its stored norm s ≥ 0, the zero group and
# the unpenalized group inline; the codes of the cold shrinkage and out-of-range values call _group_term out of
# line, which keeps its sums out of the loop over the groups.
@inline function _group_fused_term(λ, w, y, G, s, ::Type{T}) where {T}
  iszero(w) && return zero(T)
  s == _L2_ZERO && return zero(T)
  if s >= 0
    R = _working_type(T, λ, w)
    λw = convert(R, λ) * convert(R, w)
    v = convert(T, λw * convert(R, s))
    ((floatmin(R) <= λw <= floatmax(R)) & _l2_ordinary(v, T)) && return v
  end
  _group_term_outofline(λ, w, y, G, T)
end
@noinline _group_term_outofline(λ, w, y, G, ::Type{T}) where {T} = _group_term(λ, w, y, G, T)

# Exceptional terms, in this order: a zero level with a finite weight gives 0 whatever x holds; a NaN
# entry gives NaN and an infinite one Inf; the zero group gives 0; an infinite weight or level gives Inf
# (decided by the class of x, without the norm). A positive finite level and weight give λw ns 2^k from
# the scaled norm, in one rounding to T (_scaled_product).
@noinline function _group_term_cold(λ, w, x, G, ::Type{T}) where {T}
  iszero(λ) && !isinf(w) && return zero(T)
  R = _recovery_type(T, float(typeof(λ)), float(typeof(w)), float(eltype(x)))
  unbounded = isinf(w) || isinf(λ)
  k, ns = _l2_classified_norm(x, G, R, !unbounded)
  k == _EXPONENT_NAN && return T(NaN)
  k == _EXPONENT_INF && return T(Inf)
  k == _EXPONENT_ZERO && return zero(T)
  unbounded && return T(Inf)
  convert(T, _scaled_product(convert(R, λ), convert(R, w), ns, k))
end

####################################
#  Shrinkage
####################################

# y[G] = prox of γλw‖⋅‖₂ at x[G], possibly in place (y === x), where G indexes both arrays. Returns, in
# T, the norm t‖x[G]‖ of the stored group on the ordinary shrinking path (each stored entry is t x[i]
# rounded once, so this is the norm of the stored entries to within a few rounding errors, without a
# second pass), or one of the codes _L2_ZERO and _L2_RECOMPUTE. The ordinary path needs a normal threshold
# c = γλw (formed by _threshold_product) and a normal sum of squares that no underflowing square can have
# changed by more than ordinary rounding; everything else, exceptional parameters and values included,
# is decided by the cold path, which does not run on that path.
@inline function _group_shrink!(y, x, G, λ, w, γ::Real)
  T = float(eltype(y))
  c, normal = _threshold_product(T, γ, λ, w)
  if normal & (floatmin(T) <= c <= floatmax(T))
    s = _indexed_sumabs2(x, G, T)
    if isfinite(s) & _squares_normal(s, length(G), T)
      n = sqrt(s)
      if n > c
        t = one(T) - c / n
        _l2_scale!(y, x, G, t)
        return t * n
      end
      return T(_l2_zero!(y, G))
    elseif iszero(s) && _l2_all_zero(x, G, T)
      return T(_l2_zero!(y, G))
    end
  end
  T(_group_shrink_cold!(y, x, G, λ, w, γ))
end

# The parameters decide first, in this order: an unpenalized group (w = 0), a group fixed at zero (an
# infinite weight, or an infinite level at every nonnegative step, #665), the identity (a zero level or
# step), a group fixed by an infinite step. Otherwise the values decide (below). The values are in a
# function of their own, so that the exits decided by the parameters do not carry its frame.
@noinline function _group_shrink_cold!(y, x, G, λ, w, γ::Real)
  iszero(w) && return _l2_copy!(y, x, G)
  (isinf(w) || isinf(λ)) && return _l2_zero!(y, G)
  (iszero(λ) || iszero(γ)) && return _l2_copy!(y, x, G)
  isinf(γ) && return _l2_zero!(y, G)
  _group_shrink_values!(y, x, G, λ, w, γ)
end
# The values, at positive finite γ, λ and w: a NaN entry makes every output NaN; an infinite entry (and
# no NaN) makes the norm infinite and the threshold finite, so 1 - c/‖x‖ = 1 and the group is copied, also
# when c overflows in floating point; a zero group gives +0. Otherwise r = c/‖x‖ = (γλw 2^-k) / ns from
# the scaled norm ns 2^k, with c formed by the bounded-product kernel, so that neither c nor the norm needs
# to be in range: r ≥ 1 gives zero, and t = 1 - r in (0, 1] scales the group without overflow. A product
# γλw 2^-k that overflows means r > 1, and one that underflows means r < floatmin, where t rounds to 1.
@noinline function _group_shrink_values!(y, x, G, λ, w, γ::Real)
  R = _recovery_type(_working_type(float(eltype(y)), γ, λ, w), float(eltype(x)))
  k, ns = _l2_classified_norm(x, G, R)
  k == _EXPONENT_NAN && return _l2_fill!(y, G, convert(eltype(y), NaN))
  k == _EXPONENT_INF && return _l2_copy!(y, x, G)
  k == _EXPONENT_ZERO && return _l2_zero!(y, G)
  r = _scaled_product(convert(R, γ), convert(R, λ), convert(R, w), -k) / ns
  r >= one(R) && return _l2_zero!(y, G)
  _l2_scale!(y, x, G, one(R) - r)
  _L2_RECOMPUTE
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
    v += _group_term(f.λ0, @inbounds(f.w[k]), x, @inbounds(f.groups[k]), T)
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
  T = _valtype(y)
  v = zero(T)
  for k in eachindex(f.groups)
    G = @inbounds f.groups[k]
    w = @inbounds f.w[k]
    s = _group_shrink!(y, x, G, f.λ0, w, γ)
    V && (v += _group_fused_term(f.λ0, w, y, G, s, T))
  end
  V ? _group_value_checked(f, y, v) : nothing
end

function prox_block!(out::AbstractVector, f::GroupNormL2, u::AbstractVector, k::Integer, γ::Real)
  m = _check_group_block(out, f, u, k, γ)
  _group_shrink!(out, u, Base.OneTo(m), f.λ0, @inbounds(f.w[k]), γ)
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

# entry i of an array point: the group screen when the group of i is zero in x, |g[i]| when i is in no
# group or its group is nonzero (the norm is then differentiable, and its partial derivative at xᵢ = 0 is 0)
function _kkt_violation(f::GroupNormL2, g::AbstractArray, x::AbstractArray, i::Integer)
  length(g) == length(x) || _gradient_length_error(g, x)
  _check_group_extent(f, x)
  checkbounds(x, i)
  k = i <= length(f.group_of) ? @inbounds(f.group_of[i]) : 0
  k == 0 && return abs(g[i])
  G = @inbounds f.groups[k]
  @inbounds for j in G
    iszero(x[j]) || return abs(g[i])
  end
  _group_screen(f, k, g, G)
end

@noinline _block_layout_error(k) = throw(DimensionMismatch("group $k of the BlockIterate differs from group $k of GroupNormL2"))
@noinline _block_count_error(x, f) = throw(DimensionMismatch(
  "the BlockIterate has $(ncoordinates(x)) groups and GroupNormL2 has $(length(f.groups))"))

# group k of a BlockIterate assumed zero; only that group of the two layouts is compared
function _kkt_violation(f::GroupNormL2, g::AbstractArray, x::BlockIterate, k::Integer)
  ncoordinates(x) == length(f.groups) || _block_count_error(x, f)
  B = blockindices(x, k)          # BoundsError for an invalid group
  G = @inbounds f.groups[k]
  B == G || _block_layout_error(k)
  length(g) == length(x) || _gradient_length_error(g, x)
  _group_screen(f, k, g, G)
end

# ‖g[G]‖₂ - λ0 w of group k, with the coefficient of the NormL1 screen (an infinite weight or level gives Inf, a
# zero weight 0) in the floating type of g
@inline function _group_screen(f::GroupNormL2, k::Integer, g, G)
  λ, w = f.λ0, @inbounds(f.w[k])
  T = float(eltype(g))
  c = _kkt_coefficient(T, λ, w)
  s = _indexed_sumabs2(g, G, T)
  (isfinite(s) & _squares_normal(s, length(G), T) & isfinite(c)) && return sqrt(s) - c
  _group_screen_cold(λ, w, g, G, T, c)
end

# A norm or a coefficient out of range: the difference ns 2^k - mλ mw 2^(eλ+ew) of the scaled norm and the
# coefficient is formed in the recovery type, in the units of the larger exponent (_pow2_sum), and
# converted to T. Coefficients that are zero or infinite by the parameters keep c.
@noinline function _group_screen_cold(λ, w, g, G, ::Type{T}, c) where {T}
  R = _recovery_type(T, float(typeof(λ)), float(eltype(g)))
  k, ns = _l2_classified_norm(g, G, R)
  k == _EXPONENT_NAN && return T(NaN)
  k == _EXPONENT_INF && return convert(T, T(Inf) - c)
  k == _EXPONENT_ZERO && return -c
  if iszero(λ) || iszero(w) || isinf(λ) || isinf(w)
    isinf(c) && return T(-Inf)
    return convert(T, _scaled_product(ns, one(R), k)) - c
  end
  mλ, eλ = _frexp(convert(R, λ))
  mw, ew = _frexp(convert(R, w))
  convert(T, _pow2_sum(R, (ns, k), (-(mλ * mw), eλ + ew)))
end

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
