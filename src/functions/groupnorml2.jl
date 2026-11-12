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
stores its own copies of `groups` (integer indices: ranges are kept, other groups
become `Vector{Int}`) and of `w`; the level and the weights are converted to the
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
`ArgumentError`: the penalty is not separable. The coefficient `λ0 wₖ` follows
the `NormL1` screen (`Inf` for an infinite weight or level, `0` for a zero
weight); a norm or coefficient outside the range is recomputed for finite
gradients, but a `±Inf` gradient entry gives the IEEE `Inf - λ0 wₖ`, which is
`NaN` when that coefficient is infinite, also when it overflows from finite
factors.

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
for `K` groups, is recomputed with each term's exponent, as for `NormL1` (#373). `value_and_prox!` returns
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

@noinline _group_step_error(γ) = throw(ArgumentError("the step of GroupNormL2 must be nonnegative, got $γ"))
# a negative or NaN step fails the comparison
@inline _check_group_step(γ::Real) = (γ >= zero(γ) || _group_step_error(γ); nothing)

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
  _check_group_step(γ)
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
  _check_group_step(γ)
  m
end

####################################
#  Group kernels
####################################

# Codes returned by the proximal kernel in place of the sum of squares of the stored group: the
# stored group is exactly zero, or its value must be recomputed from the stored entries.
const _GROUP_ZERO = -2
const _GROUP_RECOMPUTE = -1

@inline function _group_sumsq(x, G, ::Type{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in G
    s += abs2(convert(T, x[i]))
  end
  s
end

# A zero sum of squares is also the sum of an underflowing group; a vectorized sum of magnitudes
# tells an exactly zero group apart (it is NaN or nonzero otherwise).
@inline function _group_allzero(x, G, ::Type{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in G
    s += abs(convert(T, x[i]))
  end
  iszero(s)
end

# One pass for the cold paths: whether the group holds a NaN or an infinity, and the largest exponent
# of its finite nonzero entries (typemin(Int) when there is none).
function _group_scan(x, G, ::Type{R}) where {R}
  nan = inf = false
  e = typemin(Int)
  @inbounds for i in G
    v = convert(R, x[i])
    if isnan(v)
      nan = true
    elseif isinf(v)
      inf = true
    elseif !iszero(v)
      e = max(e, Int(exponent(v)))
    end
  end
  nan, inf, e
end

# ‖x[G]‖₂ = ns 2^e for finite entries with the largest exponent e: every entry is scaled by 2^-e before
# it is squared, so its largest magnitude is in [1, 2), and the squares are summed compensated. Then
# ns is in [1, 2√|G|) and no square overflows or underflows below the rounding of the sum.
function _group_scaled_norm(x, G, ::Type{R}, e::Int) where {R}
  sc = _power_factors(R, -e)
  s = c = zero(R)
  @inbounds for i in G
    s, c = _compensated_add(s, c, abs2(_scale_value(convert(R, x[i]), sc)))
  end
  sqrt(s + c)
end

@inline function _group_zero!(y, G)
  S = eltype(y)
  @inbounds for i in G
    y[i] = zero(S)
  end
  _GROUP_ZERO
end
@inline function _group_copy!(y, x, G)
  if y !== x
    @inbounds for i in G
      y[i] = x[i]
    end
  end
  _GROUP_RECOMPUTE
end
@inline function _group_fill!(y, G, v)
  @inbounds for i in G
    y[i] = v
  end
  _GROUP_RECOMPUTE
end
# y[G] = t x[G], and the sum of squares of the stored entries when V
@inline function _group_scale!(y, x, G, t::T, ::Val{V}) where {T, V}
  S = eltype(y)
  s = zero(T)
  @inbounds @simd for i in G
    v = convert(S, t * convert(T, x[i]))
    y[i] = v
    V && (s += abs2(convert(T, v)))
  end
  s
end

# y[G] = prox of γλ0w‖⋅‖₂ at x[G], possibly in place (y === x), where G indexes both arrays. Returns,
# in T, the sum of squares of the stored group when V, or one of the codes above. The ordinary path needs a
# normal threshold c = γλ0w (formed as _l1_prox_threshold forms it) and a normal sum of squares that
# no underflowing square can have changed by more than ordinary rounding; everything else, exceptional
# parameters and values included, is decided by the cold path, which does not run on that path.
@inline function _group_prox!(y, x, G, f::GroupNormL2, k::Integer, γ::Real, ::Val{V}) where {V}
  T = float(eltype(y))
  w = @inbounds f.w[k]
  R = _working_type(T, γ, f.λ0, w)
  ab = convert(R, γ) * convert(R, f.λ0)
  c = convert(T, ab * convert(R, w))
  if (floatmin(R) <= ab <= floatmax(R)) & (floatmin(T) <= c <= floatmax(T))
    s = _group_sumsq(x, G, T)
    if isfinite(s) & _squares_normal(s, length(G), T)
      n = sqrt(s)
      n > c && return _group_scale!(y, x, G, one(T) - c / n, Val(V))
      return T(_group_zero!(y, G))
    elseif iszero(s) && _group_allzero(x, G, T)
      return T(_group_zero!(y, G))
    end
  end
  T(_group_prox_cold!(y, x, G, f, k, γ))
end

# The parameters decide first, in this order: an unpenalized group (w = 0), a group fixed at zero (an
# infinite weight, or an infinite level at every nonnegative step, #665), the identity (a zero level or
# step), a group fixed by an infinite step. Then the values, at positive finite γ, λ0 and w: a NaN entry
# makes every output NaN; an infinite entry (and no NaN) makes the norm infinite and the threshold finite,
# so 1 - c/‖x‖ = 1 and the group is copied, also when c overflows in floating point; a zero group gives +0.
# Otherwise r = c/‖x‖ = (γλ0w 2^-e) / ns from the scaled norm ns 2^e, with c formed by the bounded-product
# kernel, so that neither c nor the norm needs to be in range: r ≥ 1 gives zero, and t = 1 - r in (0, 1]
# scales the group without overflow. A product γλ0w 2^-e that overflows means r > 1, and one that
# underflows means r < floatmin, where t rounds to 1.
@noinline function _group_prox_cold!(y, x, G, f::GroupNormL2, k::Integer, γ::Real)
  λ, w = f.λ0, @inbounds(f.w[k])
  iszero(w) && return _group_copy!(y, x, G)
  (isinf(w) || isinf(λ)) && return _group_zero!(y, G)
  (iszero(λ) || iszero(γ)) && return _group_copy!(y, x, G)
  isinf(γ) && return _group_zero!(y, G)
  T = float(eltype(y))
  R = promote_type(_working_type(T, γ, λ, w), float(eltype(x)))
  nan, inf, e = _group_scan(x, G, R)
  nan && return _group_fill!(y, G, convert(eltype(y), NaN))
  inf && return _group_copy!(y, x, G)
  e == typemin(Int) && return _group_zero!(y, G)
  ns = _group_scaled_norm(x, G, R, e)
  r = _scaled_product(convert(R, γ), convert(R, λ), convert(R, w), -e) / ns
  r >= one(R) && return _group_zero!(y, G)
  t = one(R) - r
  S = eltype(y)
  @inbounds for i in G
    y[i] = convert(S, t * convert(R, x[i]))
  end
  _GROUP_RECOMPUTE
end

# The term λ0 w ‖x[G]‖₂ of the value in T, given the sum of squares s of x[G] in T, or one of the codes
# above (s = _GROUP_RECOMPUTE reads x). The ordinary path needs a normal λ0 w and a normal sum of squares.
@inline function _group_term(f::GroupNormL2, k::Integer, x, G, s, ::Type{T}) where {T}
  w = @inbounds f.w[k]
  iszero(w) && return zero(T)
  s == _GROUP_ZERO && return zero(T)
  R = _working_type(T, f.λ0, w)
  λw = convert(R, f.λ0) * convert(R, w)
  if floatmin(R) <= λw <= floatmax(R)
    s = s == _GROUP_RECOMPUTE ? _group_sumsq(x, G, T) : convert(T, s)
    (isfinite(s) & _squares_normal(s, length(G), T)) && return convert(T, λw * sqrt(convert(R, s)))
    iszero(s) && _group_allzero(x, G, T) && return zero(T)
  end
  _group_term_cold(f, k, x, G, T)
end

# Exceptional terms: an infinite weight or level is Inf away from the zero group (NaN at a NaN entry)
# and 0 at it, a zero level gives 0, and a positive finite level and weight give λ0 w ns 2^e with the
# exponent applied once (the significand products round as usual), or the ordinary Inf or NaN of a
# non-finite entry.
@noinline function _group_term_cold(f::GroupNormL2, k::Integer, x, G, ::Type{T}) where {T}
  λ, w = f.λ0, @inbounds(f.w[k])
  R = promote_type(_working_type(T, λ, w), float(eltype(x)))
  nan, inf, e = _group_scan(x, G, R)
  isinf(w) || iszero(λ) || isinf(λ) || return _finite_group_term(λ, w, x, G, nan, inf, e, T, R)
  !isinf(w) && iszero(λ) && return zero(T)
  nan && return T(NaN)
  (inf || e != typemin(Int)) ? T(Inf) : zero(T)
end
@inline function _finite_group_term(λ, w, x, G, nan, inf, e, ::Type{T}, ::Type{R}) where {T, R}
  nan && return T(NaN)
  inf && return T(Inf)
  e == typemin(Int) && return zero(T)
  convert(T, _scaled_product(convert(R, λ), convert(R, w), _group_scaled_norm(x, G, R, e), e))
end

# Recompute a value s of K terms below K floatmin(T), where the roundings of terms into the subnormal
# range may have lost more than ordinary rounding (#373): each finite term w ns 2^e is accumulated with
# its exponent and λ0 is applied once at the end. A zero, infinite or NaN level, an infinite or NaN s and
# an exactly zero argument keep s; infinite or NaN terms make s infinite or NaN, so they do not reach it.
@inline function _group_value_checked(f::GroupNormL2, x, s::T) where {T}
  _underflow_checked(T) || return s
  K = length(f.groups)
  !(s < K * floatmin(T)) && return s
  iszero(s) && _all_zero(_stored_values(x)) && return s
  _group_value_recomputed(f, x, s)
end
@noinline function _group_value_recomputed(f::GroupNormL2, x, s::T)::T where {T}
  λ = f.λ0
  (λ > 0 && isfinite(λ)) || return s
  R = promote_type(T, Float64, float(typeof(λ)), float(eltype(x)))
  acc = _expsum_start(R)
  for k in eachindex(f.groups)
    w = @inbounds f.w[k]
    iszero(w) && continue
    G = @inbounds f.groups[k]
    nan, inf, e = _group_scan(x, G, R)
    (nan || inf) && return s
    e == typemin(Int) && continue
    isinf(w) && return s
    acc = _expsum_product(acc, w, _group_scaled_norm(x, G, R, e), e)
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
    v += _group_term(f, k, x, @inbounds(f.groups[k]), _GROUP_RECOMPUTE, T)
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
    s = _group_prox!(y, x, G, f, k, γ, Val(V))
    V && (v += _group_term(f, k, y, G, s, T))
  end
  V ? _group_value_checked(f, y, v) : nothing
end

function prox_block!(out::AbstractVector, f::GroupNormL2, u::AbstractVector, k::Integer, γ::Real)
  m = _check_group_block(out, f, u, k, γ)
  _group_prox!(out, u, Base.OneTo(m), f, k, γ, Val(false))
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

# ‖g[G]‖₂ - λ0 w, with the coefficient of the NormL1 screen (an infinite weight or level gives Inf, a zero
# weight 0) in the floating type of g
@inline function _group_screen(f::GroupNormL2, k::Integer, g, G)
  T = float(eltype(g))
  c = _kkt_coefficient(T, f.λ0, @inbounds(f.w[k]))
  s = _group_sumsq(g, G, T)
  (isfinite(s) & _squares_normal(s, length(G), T) & isfinite(c)) && return sqrt(s) - c
  _group_screen_cold(f, k, g, G, T, c)
end

# A norm or a coefficient out of range: the difference ns 2^e - mc 2^ec of the scaled norm and the
# coefficient's significands is formed in the units of the larger exponent, where both are below 2√|G|,
# and scaled back once. Coefficients that are zero or infinite by the parameters keep c.
@noinline function _group_screen_cold(f::GroupNormL2, k::Integer, g, G, ::Type{T}, c) where {T}
  λ, w = f.λ0, @inbounds(f.w[k])
  R = promote_type(T, float(typeof(λ)), float(eltype(g)))
  nan, inf, e = _group_scan(g, G, R)
  nan && return T(NaN)
  inf && return convert(T, T(Inf) - c)
  e == typemin(Int) && return -c
  ns = _group_scaled_norm(g, G, R, e)
  if iszero(λ) || iszero(w) || isinf(λ) || isinf(w)
    isinf(c) && return T(-Inf)
    return convert(T, _scaled_product(ns, one(R), e)) - c
  end
  mλ, eλ = _frexp(convert(R, λ))
  mw, ew = _frexp(convert(R, w))
  E = max(e, eλ + ew)
  d = ldexp(ns, e - E) - ldexp(mλ * mw, eλ + ew - E)
  convert(T, ldexp(d, E))
end
