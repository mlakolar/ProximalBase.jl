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

# The loops over the groups below are this file's own. They use the building blocks of numerical_kernels.jl: the sums
# of squares of a group, the classified norm and the exponent sum (_ExpSum) of the recoveries, and the cold prox kernel.

# λ0 Σₖ wₖ ‖x[Gₖ]‖₂. A group of weight zero adds nothing and its entries are not read. The ordinary loop adds the term
# λ0 wₖ √qₖ of each penalized group, from its sum of squares qₖ in T, and leaves for the recovery (below), once and for
# the whole value, at the first group that it cannot certify: a sum of squares that is not finite or may have lost its
# accuracy to underflow (below m floatmin/eps), unless the group is exactly zero (it adds 0 whatever the weight), or a
# coefficient λ0 wₖ out of the normal range, an infinite weight included. After the loop, the sum is kept if it is
# below floatmax(T)/2 (a NaN or Inf is not, and their class is decided by the recovery) and, for Float32 and Float64, if
# it is not below nt floatmin(T) for the nt groups with a nonzero term (#373): a term rounds into the subnormal range
# with an absolute loss of at most eps floatmin/2, so that a sum from nt floatmin on keeps its relative accuracy. A zero
# level and an infinite level are decided by the recovery at once.
function _group_value(f::GroupNormL2, x)
  T = _valtype(x)
  λ = f.λ0
  W = f.w
  isempty(W) && return zero(T)
  (iszero(λ) | isinf(λ)) && return _group_value_recovered(f, x, T)
  Q = _working_type(T, λ, oneunit(eltype(W)))
  lc = convert(Q, λ)
  v = zero(T)
  nt = 0
  inf = false
  if _all_singletons(f)
    # one entry per group: the term is λ0 wₖ |x[i]|, with no sum of squares to underflow or overflow
    @inbounds for k in eachindex(f.groups)
      w = W[k]
      iszero(w) && continue
      lw = lc * convert(Q, w)
      a = abs(convert(T, x[f.groups[k][1]]))
      if !isfinite(a)
        isnan(a) && return T(NaN)       # the term of a NaN entry at a positive finite level is NaN, and so is the value
        inf = true                      # an infinite entry: Inf, and a later NaN decides
        continue
      end
      ((floatmin(Q) <= lw) & (lw <= floatmax(Q))) || return _group_value_recovered(f, x, T)
      v += convert(T, lw * convert(Q, a))
      nt += !iszero(a)
    end
    inf && return T(Inf)
    _l2_ordinary(v, T) || return _group_value_recovered(f, x, T)
    _underflow_checked(T) && !_sum_kept(v, nt) && return _group_value_recovered(f, x, T)
    return v
  end
  @inbounds for k in eachindex(f.groups)
    w = W[k]
    iszero(w) && continue
    G = f.groups[k]
    lw = lc * convert(Q, w)
    q = _indexed_sumabs2(x, G, T)
    if isfinite(q) & _squares_normal(q, length(G), T) & (floatmin(Q) <= lw) & (lw <= floatmax(Q))
      v += convert(T, lw * sqrt(convert(Q, q)))
      nt += 1
    elseif isnan(q)
      return T(NaN)       # a NaN entry of a penalized group at a positive finite level: the value is NaN
    elseif iszero(q) && _l2_all_zero(x, G, T)
      continue
    elseif q == T(Inf) && _square_exponent(_IndexedValues(x, G), nothing, T) == _EXPONENT_INF
      inf = true          # an infinite entry (and no NaN in the group): the term is Inf, and a later NaN decides
    else
      return _group_value_recovered(f, x, T)
    end
  end
  inf && return T(Inf)
  _l2_ordinary(v, T) || return _group_value_recovered(f, x, T)
  _underflow_checked(T) && !_sum_kept(v, nt) && return _group_value_recovered(f, x, T)
  v
end

# The value from the terms w ns 2^k of the groups, accumulated with their exponents (_ExpSum, in at least Float64)
# with λ0 applied once (_expsum_value), one rounding to T. The class of each term, in this order: a zero weight, and a
# zero level with a finite weight, give 0 whatever x holds; a NaN entry gives NaN (and so the value) and an infinite one
# Inf; the zero group gives 0; an infinite weight or level gives Inf (decided by the class of x, without the norm). The
# value is the sum of the classes: NaN, else Inf, else the accumulated sum.
@noinline function _group_value_recovered(f::GroupNormL2, x, ::Type{T}) where {T}
  λ = f.λ0
  R = _recovery_type(T, float(typeof(λ)), float(eltype(x)))
  acc = _expsum_start(R)
  inf = false
  for k in eachindex(f.groups)
    w = @inbounds f.w[k]
    (iszero(w) || (iszero(λ) && !isinf(w))) && continue
    unbounded = isinf(w) || isinf(λ)
    k_, ns = _l2_classified_norm(x, @inbounds(f.groups[k]), R, !unbounded)
    k_ == _EXPONENT_NAN && return T(NaN)
    k_ == _EXPONENT_ZERO && continue
    if k_ == _EXPONENT_INF || unbounded
      inf = true
    else
      mw, ew = _frexp(convert(R, w))
      acc = _expsum_add(acc, mw * ns, ew + k_)
    end
  end
  inf && return T(Inf)
  _expsum_value(T, acc, λ)
end

####################################
#  Oracles
####################################

function (f::GroupNormL2)(x::AbstractArray)
  _checkreal(f, x)
  Base.require_one_based_indexing(x)
  _check_group_extent(f, x)
  _group_value(f, x)
end

prox!(y::AbstractArray, f::GroupNormL2, x::AbstractArray, γ::Real) =
  (_check_group_arrays(f, y, x, γ); _group_prox_all!(y, f, x, γ, Val(false)); y)
value_and_prox!(y::AbstractArray, f::GroupNormL2, x::AbstractArray, γ::Real) =
  (_check_group_arrays(f, y, x, γ); _group_prox_all!(y, f, x, γ, Val(true)))
# routes of wrappers such as Scaled, which preflight with _checkwrite first
_value_and_prox_unchecked!(y::AbstractArray, f::GroupNormL2, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)

# Whether every group has exactly one entry: the entries in groups are those up to the largest in a group that are in
# none of the free ones, and a group has at least one.
@inline _all_singletons(f::GroupNormL2) = length(f.group_of) - length(f.free) == length(f.groups)

# The proximal point of the groups, and with V the value at the stored output. The parameters of the call decide the
# loop, once: γλ0 in the normal range gives the ordinary loop, where each group tests its own threshold γλ0wₖ; a zero or
# infinite γ or λ0 decides every group by its weight alone (no data are read); otherwise γλ0 is out of the range and
# each group goes to the cold kernel.
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
  isempty(f.groups) && return V ? zero(T) : nothing
  λ = f.λ0
  R = _working_type(T, γ, λ, oneunit(eltype(f.w)))
  ab = convert(R, γ) * convert(R, λ)
  if floatmin(R) <= ab <= floatmax(R)
    return _all_singletons(f) ? _group_prox_ordinary!(y, f, x, γ, ab, Val(V), Val(true)) :
                                _group_prox_ordinary!(y, f, x, γ, ab, Val(V), Val(false))
  elseif iszero(γ) | iszero(λ) | isinf(γ) | isinf(λ)
    _group_prox_decided!(y, f, x, γ)
    V || return nothing
    # a group is zero or unchanged by its weight: at an infinite level, or an infinite step at a positive level, every
    # penalized group is +0 and the value is 0; at a zero step or level the value is that of the unchanged groups
    return isinf(λ) | (isinf(γ) & !iszero(λ)) ? zero(T) : _group_value(f, y)
  end
  return _group_prox_exceptional!(y, f, x, γ, Val(V))
end

# The ordinary loop, γλ0 = ab normal. With V, the term λ0 wₖ (t ‖xₖ‖) of the stored group (the norm that the kernel
# returns) is added to the value. The value at the stored output is formed again, once, after the loop, if the output of
# a group was recomputed without its norm (a group that the kernel left to the cold path and did not zero), if a
# coefficient λ0 wₖ is out of the normal range, or if the sum cannot be kept (as in _group_value).
function _group_prox_ordinary!(y, f::GroupNormL2, x, γ, ab, ::Val{V}, ::Val{single}) where {V, single}
  T = float(eltype(y))
  λ = f.λ0
  W = f.w
  Q = _working_type(T, λ, oneunit(eltype(W)))
  lc = convert(Q, λ)
  v = zero(T)
  nt = 0
  ok = true
  nan = inf = false
  @inbounds for k in eachindex(f.groups)
    w = W[k]
    s = single ? _single_shrink!(y, x, f.groups[k], λ, w, γ, ab) : _group_shrink!(y, x, f.groups[k], λ, w, γ, ab)
    V || continue
    if s >= 0
      lw = lc * convert(Q, w)
      ok &= (floatmin(Q) <= lw) & (lw <= floatmax(Q))
      v += convert(T, lw * convert(Q, s))
      nt += 1
    elseif s == T(_L2_NAN)
      nan = true
    elseif s == T(_L2_INF)
      inf = true
    elseif s != T(_L2_ZERO) && !iszero(w)
      ok = false
    end
  end
  V || return nothing
  nan && return T(NaN)
  inf && return T(Inf)
  ok && _l2_ordinary(v, T) && (!_underflow_checked(T) || _sum_kept(v, nt)) && return v
  _group_value(f, y)
end

# The group of weight w with the threshold c = γλ0w in the normal range of T, at γλ0 = ab in the normal range of its
# type: returns the norm t‖x‖ of the stored group on the ordinary shrinking path (each stored entry is t x[i] rounded
# once, so this is the norm of the stored entries to within a few rounding errors, without a second pass), or one of
# the codes of the kernels. The ordinary path needs a normal sum of squares that no underflowing square can have
# changed by more than ordinary rounding. A threshold out of the range (an unpenalized group, w = 0, included) and
# everything else, exceptional parameters and values, is decided by the cold kernel, which does not run on this path.
@inline function _group_shrink!(y, x, G, λ, w, γ::Real, ab::R) where {R}
  T = float(eltype(y))
  c = convert(T, ab * convert(R, w))
  if (floatmin(T) <= c) & (c <= floatmax(T))
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
  elseif iszero(w)
    return T(_l2_copy!(y, x, G))
  end
  T(_l2_prox_cold!(y, x, G, λ, w, γ))
end

# The same for a group of one entry, whose norm |x[i]| has no sum of squares to underflow or overflow: a NaN entry
# (which makes the group NaN) and an infinite one (which leaves it unchanged) follow from t = 1 - c/|x[i]|. A loop over
# groups that are all of one entry uses it (a call with a mixture of sizes uses _group_shrink!).
@inline function _single_shrink!(y, x, G, λ, w, γ::Real, ab::R) where {R}
  T = float(eltype(y))
  c = convert(T, ab * convert(R, w))
  if (floatmin(T) <= c) & (c <= floatmax(T))
    i = @inbounds G[1]
    a = convert(T, @inbounds x[i])
    n = abs(a)
    if n <= c
      @inbounds y[i] = zero(eltype(y))
      return T(_L2_ZERO)
    end
    t = one(T) - c / n
    @inbounds y[i] = convert(eltype(y), t * a)
    return t * n
  elseif iszero(w)
    return T(_l2_copy!(y, x, G))
  end
  T(_l2_prox_cold!(y, x, G, λ, w, γ))
end

# Groups decided by the parameters alone, in the order of the cold kernel: weight zero (unchanged), infinite weight or
# level (+0), zero level or step (unchanged), an infinite step at a positive finite level (+0).
function _group_prox_decided!(y, f::GroupNormL2, x, γ)
  λ = f.λ0
  identity = iszero(λ) | iszero(γ)
  infinite_level = isinf(λ)
  @inbounds for k in eachindex(f.groups)
    G = f.groups[k]
    w = f.w[k]
    if iszero(w) | (!isinf(w) & !infinite_level & identity)
      _l2_copy!(y, x, G)
    else
      _l2_zero!(y, G)
    end
  end
end

# γλ0 outside the range with positive finite γ and λ0: every group is decided by its threshold and its values, by the
# cold kernel. With V, the value at the stored output is formed from the same passes: the cold kernel gives the stored
# norm m 2^e of each scaled group, accumulated with its exponent (_ExpSum) as the recovery of a value does, and the
# class of a group that is NaN (the value is NaN) or infinite (Inf, if none is NaN). A group of weight zero adds nothing,
# an infinite weight gives +0. The recovery type of the cold kernel and that of the value must be one, or the value is
# formed from the output.
function _group_prox_exceptional!(y, f::GroupNormL2, x, γ, ::Val{V}) where {V}
  T = float(eltype(y))
  λ = f.λ0
  W = f.w
  R = _recovery_type(_working_type(T, γ, λ, oneunit(eltype(W))), float(eltype(x)))
  same = V && R === _recovery_type(T, float(typeof(λ)), float(eltype(y)))
  acc = _expsum_start(R)
  nan = inf = false
  @inbounds for k in eachindex(f.groups)
    w = W[k]
    code, m, e = _l2_prox_cold_norm!(y, x, f.groups[k], λ, w, γ)
    V || continue
    if code == _L2_NAN
      nan = true
    elseif code == _L2_INF
      inf = true
    elseif code == _L2_RECOMPUTE && !iszero(w)
      if same
        mw, ew = _frexp(convert(R, w))
        acc = _expsum_add(acc, mw * m, ew + e)
      else
        return _group_value(f, y)
      end
    end
  end
  V || return nothing
  nan && return T(NaN)
  inf && return T(Inf)
  _expsum_value(T, acc, λ)
end

function prox_block!(out::AbstractVector, f::GroupNormL2, u::AbstractVector, k::Integer, γ::Real)
  m = _check_group_block(out, f, u, k, γ)
  G = Base.OneTo(m)
  λ, w = f.λ0, @inbounds(f.w[k])
  T = float(eltype(out))
  R = _working_type(T, γ, λ, w)
  ab = convert(R, γ) * convert(R, λ)
  if floatmin(R) <= ab <= floatmax(R)
    m == 1 ? _single_shrink!(out, u, G, λ, w, γ, ab) : _group_shrink!(out, u, G, λ, w, γ, ab)
  else
    _l2_prox_cold!(out, u, G, λ, w, γ)
  end
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

# ‖g[G]‖₂ - λ0 w of group k, with the level and the weight of the group
@inline function _group_screen(f::GroupNormL2, k::Integer, g, G)
  T = float(eltype(g))
  λ, w = f.λ0, @inbounds(f.w[k])
  c = _kkt_coefficient(T, λ, w)
  if length(G) == 1 && isfinite(c)     # the norm of one entry is its magnitude: no square to underflow
    return abs(convert(T, @inbounds g[@inbounds G[1]])) - c
  end
  s = _indexed_sumabs2(g, G, T)
  (isfinite(s) & _squares_normal(s, length(G), T) & isfinite(c)) && return sqrt(s) - c
  _l2_screen_cold(λ, w, g, G, T, c)
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
