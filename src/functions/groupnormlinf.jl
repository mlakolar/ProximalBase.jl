####################################
#  group ℓ∞ norm  λ0 Σₖ wₖ max_{i∈Gₖ} |xᵢ|
####################################

# The type, the group step and the oracles. A group is read through `_IndexedValues` (no copy, no view) and its level
# comes from the helpers that NormLinf uses: the sum of magnitudes, the threshold of the ℓ1 ball, and the reads scaled
# by a power of two. The checks and the value sums that GroupNormL2 shares with this function are in groupnorml2.jl,
# which is included after this file, so that its shared methods can dispatch on both types.

"""
    GroupNormLinf(λ0, groups)
    GroupNormLinf(λ0, groups, w)

Group ℓ∞ penalty `f(x) = λ₀ Σₖ wₖ max_{i∈Gₖ} |xᵢ|`, with default weights `wₖ = 1`. Groups are
nonempty, disjoint lists or ranges of positive linear indices. Their order is preserved; entries in no group are
unpenalized. The nonnegative level and weights cannot be `NaN`. Supply one weight per group; the constructor copies
`groups` and `w`.

[`prox`](@ref) clips each group at a level `τₖ ≥ 0`: `prox_{γf}(x)ᵢ = clamp(xᵢ, -τₖ, τₖ)` for `i ∈ Gₖ`, where `τₖ` solves
`Σ_{i∈Gₖ} (|xᵢ| - τₖ)₊ = γλ₀wₖ`, and the group becomes `+0` when `γλ₀wₖ ≥ ‖x[Gₖ]‖₁` for finite entries. The radius is formed
from the three factors in power-of-two units when its product leaves the range. [`prox_block`](@ref) takes a compact
vector in `groups[k]` order; it does not update an iterate. To commit its result, use `copyto!(blockview(x, k), out)`.
See [the group ℓ∞ norm](@ref group-linf-guide) for the formula, the group step and the optimality screen at a zero
group.

# Examples
```jldoctest
julia> f = GroupNormLinf(1.0, [1:2, 3:3]);

julia> f([3.0, -1.0, 2.0])
5.0

julia> Tuple(prox(f, [3.0, -1.0, 2.0], 1.0))
(2.0, -1.0, 1.0)

julia> prox_block(f, [3.0, -1.0], 1, 4.0)
2-element Vector{Float64}:
 0.0
 0.0

julia> h = GroupNormLinf(1.0, [[3, 1]], [2.0]);

julia> Tuple(prox(h, [3.0, 9.0, 1.0], 1.0))
(1.0, 9.0, 1.0)

julia> kkt_violation(GroupNormLinf(1.0, [1:2]), [5/8, 5/8], zeros(2), 1)
0.25
```

# Edge cases
- Real, one-based arrays and valid shapes/groups are required. Negative or `NaN` steps fail before writes.
- Exact in-place prox is supported. For compact blocks, `out === u` is supported; other overlaps detected by
  `Base.mightalias` are rejected, and `out` must not alias iterate storage.
- The rules apply to each group in this order. `wₖ = 0` leaves the group unchanged, also with `λ₀ = Inf` or `γ = Inf`.
  Otherwise `wₖ = Inf`, or `λ₀ = Inf`, gives `+0` at every nonnegative step, `γ = 0` and `γ = Inf` included. Otherwise
  `λ₀ = 0` or `γ = 0` gives the identity, also when `λ₀wₖ` overflows, and `γ = Inf` gives `+0`. At positive finite
  parameters a `NaN` entry makes every entry of its group `NaN` and leaves the other groups unaffected, and an infinite
  entry without `NaN` leaves its group unchanged, also when the finite product `γλ₀wₖ` overflows. For finite entries a
  sum of magnitudes above `floatmax/2` is handled in power-of-two units, and a radius that overflows while the sum is
  finite gives `+0`.
- The value of a group is `λ₀wₖ max|x[Gₖ]|`, rounded once in the promoted type: zero when `wₖ = 0` (whatever the group
  holds), when the group is zero, and when `λ₀ = 0` with a finite weight; `NaN` at a `NaN` entry; `Inf` at a nonzero
  group, without `NaN`, when `λ₀` or `wₖ` is infinite. The sum of the terms is recomputed in power-of-two units where it
  may have lost accuracy (below `K floatmin` for `K` groups) or class (at or above `floatmax/2`). `value_and_prox!`
  returns the value of the stored output.
- The accuracy of each group's level is absolute, of order `n eps(T) max|x[Gₖ]| + δ` in the floating type `T` of the output
  for a group of `n` entries and `δ = nextfloat(zero(T))`, as for [`NormLinf`](@ref): an entry far below the largest has
  no relative guarantee. The entries are read as `abs(convert(T, xᵢ))`, and the level assumes `n eps(T) ≤ 1/2` for each
  group (for `Float32` a group of at most `2^22` entries). Float32 and Float64 with arguments of the same type are the
  supported precisions.
- `kkt_violation(f, v, x, k)` is provided at a zero group (and at an entry in no group): the signed margin
  `‖v[Gₖ]‖₁ - λ₀wₖ`. A nonzero group, the full-point form `kkt_violation(f, v, x)` and the three-argument form `kkt_violation(f, v, i)`
  throw an `ArgumentError`.
- The threshold needs a buffer of the length of the largest group. The object keeps one, allocated by the first group step
  that needs it and again when the element type changes. [`prox!`](@ref) and [`value_and_prox!`](@ref) use it for every
  group that reaches the threshold; [`prox_block!`](@ref) leaves it untouched when `out` is an `Array` of the output type
  that does not overlap `u`, and uses `out` as the buffer then. The parameter exits, a `NaN` or infinite group and the
  zero fill do not touch it. The same object must not be used from several tasks at once while its buffer may be in
  use: a [`Scaled`](@ref) wrapper shares the buffer of its function, and a threaded loop of `prox_block!(u, f, u, k, γ)`
  over groups does too. Construct a fresh `GroupNormLinf` for each task, or synchronize the calls.
- No `SymmetricSparseIterate`, `restrict`, per-group steps or compact block method under [`Scaled`](@ref).
"""
struct GroupNormLinf{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractFunction
  λ0::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}   # group_of[i] = k for i ∈ Gₖ, 0 for an index in no group
  free::Vector{Int}       # the indices up to the largest in a group that are in none
  maxlen::Int             # the length of the longest group, 0 without groups
  scratch::Scratch

  function GroupNormLinf{T, G}(λ0::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    _check_one_based_groups(groups)
    length(w) == length(groups) ||
      throw(DimensionMismatch("GroupNormLinf needs one weight per group: $(length(groups)) groups and $(length(w)) weights"))
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    group_of = _group_of(groups)
    new{T, G}(λ0, groups, w, group_of, findall(iszero, group_of), maximum(length, groups; init = 0), Scratch())
  end
end
function GroupNormLinf(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector)
  _check_one_based_groups(groups)
  _check_one_based(w)
  eltype(w) <: Real || _real_error(:GroupNormLinf)
  T = float(promote_type(typeof(λ0), eltype(w)))
  G = _own_groups(groups)
  GroupNormLinf{T, typeof(G)}(convert(T, λ0), G, Vector{T}(w))
end
function GroupNormLinf(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}})
  _check_one_based_groups(groups)
  GroupNormLinf(λ0, groups, ones(float(typeof(λ0)), length(groups)))
end

isproximable(::Type{<:GroupNormLinf}) = true
isconvex(::Type{<:GroupNormLinf}) = true

# A BlockIterate with the groups of f, which must partition 1:p (the public constructor checks it).
BlockIterate(p::Integer, f::GroupNormLinf) = BlockIterate(Float64, p, f)
BlockIterate(::Type{T}, p::Integer, f::GroupNormLinf) where {T} = BlockIterate(T, p, f.groups)
BlockIterate(x::AbstractVector, f::GroupNormLinf) = BlockIterate(x, f.groups)

####################################
#  Values
####################################

# The maximum of |x[i]|, i ∈ G, read in T: a NaN gives NaN and an infinite entry Inf. For Float32 and Float64 it is the
# maximum of the bit patterns with the sign cleared (as `_maximum_abs`), which orders the magnitudes and puts a NaN
# above Inf.
@inline function _linf_group_max(x, G, ::Type{T}) where {T}
  m = zero(T)
  @inbounds for i in G
    m = max(m, abs(convert(T, x[i])))
  end
  m
end
@inline function _linf_group_max(x, G, ::Type{T}) where {T<:Union{Float32, Float64}}
  U = _int_type(T)
  m = zero(U)
  @inbounds @simd for i in G
    m = max(m, reinterpret(U, convert(T, x[i])) & typemax(U))
  end
  reinterpret(T, m)
end

# The term λ0 w m of a group with the largest magnitude m (in T), in this order: a zero weight gives zero, whatever the
# group holds; a zero level with a finite weight gives zero, also at a NaN or infinite m; a NaN m gives NaN; a zero m
# gives zero at every level and weight (an infinite one included); an infinite level, weight or m gives Inf. Otherwise
# the product is formed in the working type R, as (λ0 w) m when λ0 w is a normal number and by the three-factor kernel
# when it is not (zero, subnormal or overflowing), and converted to T.
@inline function _linf_term(λ0, w, m, ::Type{T}) where {T}
  iszero(w) && return zero(T)
  (iszero(λ0) && !isinf(w)) && return zero(T)
  isnan(m) && return T(NaN)
  iszero(m) && return zero(T)
  (isinf(w) || isinf(λ0) || isinf(m)) && return T(Inf)
  R = _working_type(T, λ0, w)
  λw = convert(R, λ0) * convert(R, w)
  floatmin(R) <= λw <= floatmax(R) && return convert(T, λw * convert(R, m))
  convert(T, _scaled_product(convert(R, λ0), convert(R, w), convert(R, m), 0))
end

# the term of the group G of x, which is read only when it can matter
@inline function _linf_group_term(λ0, w, x, G, ::Type{T}) where {T}
  iszero(w) && return zero(T)
  (iszero(λ0) && !isinf(w)) && return zero(T)
  _linf_term(λ0, w, _linf_group_max(x, G, T), T)
end

function (f::GroupNormLinf)(x::AbstractArray)
  _checkreal(f, x)
  Base.require_one_based_indexing(x)
  _check_group_extent(f, x)
  T = _valtype(x)
  v = zero(T)
  for k in eachindex(f.groups)
    v += _linf_group_term(f.λ0, @inbounds(f.w[k]), x, @inbounds(f.groups[k]), T)
  end
  _group_value_checked(f, x, v)
end

# The term w m 2^e of a penalized group, added to the accumulator of a value recomputed with the terms' exponents
# (_ExpSum), as `_l2_expsum_add` does for the ℓ2 norm. A zero weight and a zero group add nothing; a NaN or infinite entry,
# or an infinite weight at a nonzero group, make the ordinary sum NaN or infinite: nothing is returned for them, and the
# caller keeps that sum. The entries are read in the floating type of x, as the ordinary value reads them.
@inline function _linf_expsum_add(acc::_ExpSum{R}, x, G, w) where {R}
  iszero(w) && return acc
  m = _linf_group_max(x, G, float(eltype(x)))
  (isnan(m) || isinf(m)) && return nothing
  iszero(m) && return acc
  isinf(w) && return nothing
  _expsum_product(acc, w, m)
end

####################################
#  The group step
####################################

# y[G] = prox of γ λ0 w ‖·‖∞ at x[G], where G indexes both arrays (y === x is supported). Returns, in T = float(eltype(y)),
# the largest magnitude of the stored group when the step knows it without a pass (the clip level, 0 for the zero fill, NaN
# for the NaN fill, Inf for the copy of an infinite group), or one of the codes of the ℓ2 kernels: _L2_ZERO for a group whose
# term is zero, _L2_RECOMPUTE for a group whose stored entries must be read.
#
# The parameter exits are those of _l2_shrink_fixed! (w = 0, then an infinite w or λ0, then a zero λ0 or γ, then an infinite
# γ). They are taken before a radius is formed, so that a product such as λ0 w with λ0 = Inf and w = 0 is not formed here.
# Past them λ0, w and γ are positive and finite. The entries are then read as abs(convert(T, x[i])).
#
# The sum s of the magnitudes decides the branch. A sum that is at most floatmax/2 is neither NaN nor infinite, so all the
# entries are finite; it is also the sum s of the hypotheses of _l1ball_threshold!. Any other sum goes to
# _linf_group_exceptional!. `src` is the Scratch of the function or an array that serves as the buffer (see
# _linf_buffer), and `maxlen` is the length of the longest group.
function _linf_group_prox!(y, x, G, λ0, w, γ::Real, src, maxlen::Int)
  T = float(eltype(y))
  if iszero(w) | isinf(w) | isinf(λ0) | iszero(λ0) | iszero(γ) | isinf(γ)
    return T(_l2_shrink_fixed_call!(y, x, G, λ0, w, γ))
  end
  xv = _IndexedValues(x, G, T)
  s = _linf_sum(xv, _AbsRead{T}(), T)
  s <= floatmax(T) / 2 || return _linf_group_exceptional!(y, x, G, xv, λ0, w, γ, s, src, maxlen)
  r = _linf_group_radius(T, γ, λ0, w)
  r >= s && return _linf_group_fill!(y, G, zero(T))
  # _l1ball_threshold! (in _linf_group_level) needs finite nonnegative magnitudes, which s <= floatmax/2 certifies, and a
  # radius with 0 <= r < s: r is a product of positive finite factors, rounded to T (a number, not NaN), and r >= s failed.
  # Its proof also assumes n eps(T) <= 1/2 for the n entries of the group, which is not checked here.
  τ = _linf_group_level(src, T, maxlen, xv, r, _AbsRead{T}())
  _linf_group_clip!(y, x, G, τ)
end

# The radius γ λ0 w rounded to T, for positive finite factors: the product in the working type R when γ λ0 is a normal
# number and the result lies in the normal range of T; otherwise the three-factor kernel with the power 2^-k, which forms
# the product from the significands and exponents, so that a factor product such as γ λ0 that underflows (or λ0 w that
# overflows) does not decide the radius. The result can be zero, subnormal or Inf after the conversion to T.
@inline function _linf_group_radius(::Type{T}, γ, λ0, w, k::Int = 0) where {T}
  if k == 0
    c, normal = _threshold_product(T, γ, λ0, w)
    (normal & (floatmin(T) <= c <= floatmax(T))) && return c
  end
  R = _working_type(T, γ, λ0, w)
  convert(T, _scaled_product(convert(R, γ), convert(R, λ0), convert(R, w), -k))
end

# The branches at a sum of magnitudes that is NaN, infinite or above floatmax/2, as _linf_exceptional! has them for the
# whole argument: a NaN entry gives NaN in the group; an infinite entry without NaN gives the identity (the limit of the
# clip as that entry grows, for a finite radius), also when the radius overflows; finite entries are read scaled by 2^-k and
# the level found in those units is multiplied by 2^k (the prox is positively homogeneous). The hypotheses on the scaled
# sum, the radius and the rounding are those stated at _linf_exceptional!, with n the number of entries of the group.
@noinline function _linf_group_exceptional!(y, x, G, xv, λ0, w, γ, s::T, src, maxlen) where {T}
  isnan(s) && return _linf_group_fill!(y, G, T(NaN))
  m = _linf_classify(xv, _AbsRead{T}(), T)[2]
  if isinf(m)
    _l2_copy!(y, x, G)
    return m
  end
  k = _linf_exponent(length(G))
  local s′
  while true
    s′ = _linf_sum(xv, _ScaledAbsRead{T}(k), T)
    s′ <= floatmax(T) / 2 && break
    k += 1
  end
  r′ = _linf_group_radius(T, γ, λ0, w, k)
  r′ >= s′ && return _linf_group_fill!(y, G, zero(T))
  # the conditions of _l1ball_threshold! hold for the scaled entries: they are finite and nonnegative because s′ is at most
  # floatmax/2, and 0 <= r′ < s′ as above
  τ′ = _linf_group_level(src, T, maxlen, xv, r′, _ScaledAbsRead{T}(k))
  # τ′ is at most the largest scaled entry, 2^-k m rounded; 2^k τ′ is capped at m, which the rounding of the scaling can exceed
  _linf_group_clip!(y, x, G, min(ldexp(τ′, k), m))
end

# The clip level max(τ, 0) from the threshold of the ℓ1 ball of the magnitudes rd(xv[j]) at the radius r, as
# _linf_threshold does for the whole argument. Under the conditions at _l1ball_threshold! the level is a number, not NaN.
@inline function _linf_group_level(src, ::Type{T}, maxlen::Int, xv, r::T, rd) where {T}
  max(_l1ball_threshold!(_linf_buffer(src, T, maxlen), xv, r, rd), zero(T))
end
# The buffer of the threshold: the scratch of the function at the length of its longest group (a buffer of the right size
# is returned as it is, so the allocation happens at the first group that needs it), or an array that the caller gives.
@inline _linf_buffer(sc::Scratch, ::Type{T}, maxlen::Int) where {T} = _array_buffer!(sc, T, (maxlen,))
@inline _linf_buffer(buf::AbstractArray, ::Type{T}, maxlen::Int) where {T} = buf

# y[i] = x[i] clipped to [-τ, τ] for i ∈ G, with τ ≥ 0; τ = 0 gives +0. Returns τ, the largest stored magnitude when τ is at
# most the largest magnitude of the group, as the callers have it. A zero entry keeps its sign below τ. The entries are
# read in T and stored in the type of y.
@inline function _linf_group_clip!(y, x, G, τ::T) where {T}
  iszero(τ) && return _linf_group_fill!(y, G, zero(T))
  _linf_group_clip_loop!(y, x, G, τ)
  τ
end
@inline function _linf_group_clip_loop!(y::_DenseOutput, x, G, τ::T) where {T}
  S = eltype(y)
  nτ = -τ
  @inbounds @simd for i in G
    v = convert(T, x[i])
    y[i] = convert(S, ifelse(v > τ, τ, ifelse(v < nτ, nτ, v)))
  end
end
# an iterate as output is written by a plain loop: its write keeps its own records
@inline function _linf_group_clip_loop!(y, x, G, τ::T) where {T}
  S = eltype(y)
  nτ = -τ
  @inbounds for i in G
    v = convert(T, x[i])
    y[i] = convert(S, ifelse(v > τ, τ, ifelse(v < nτ, nτ, v)))
  end
end

# y[i] = v for i ∈ G, for the zero and NaN outputs; returns the largest stored magnitude, |v| in T
@inline function _linf_group_fill!(y, G, v::T) where {T}
  s = convert(eltype(y), v)
  @inbounds for i in G
    y[i] = s
  end
  abs(v)
end

####################################
#  Proximal operators
####################################

prox!(y::AbstractArray, f::GroupNormLinf, x::AbstractArray, γ::Real) =
  (_check_group_arrays(f, y, x, γ); _linf_prox_all!(y, f, x, γ, Val(false)); y)
value_and_prox!(y::AbstractArray, f::GroupNormLinf, x::AbstractArray, γ::Real) =
  (_check_group_arrays(f, y, x, γ); _linf_value_and_prox!(y, f, x, γ))
# routes of wrappers such as Scaled, which preflight with _checkwrite first
_value_and_prox_unchecked!(y::AbstractArray, f::GroupNormLinf, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)

# The fused value needs the stored output to be in the floating type of its entries; an output of another element type (an
# integer one) stores rounded entries, and its value is read from it.
@inline function _linf_value_and_prox!(y, f::GroupNormLinf, x, γ::Real)
  if eltype(y) === float(eltype(y))
    _linf_prox_all!(y, f, x, γ, Val(true))
  else
    _linf_prox_all!(y, f, x, γ, Val(false))
    f(y)
  end
end

# The entries in no group are copied, then each group by its step. The fused value adds the term of each group from the
# largest stored magnitude that the step returns (_linf_fused_term), and the sum is checked as the value is
# (_group_value_checked).
function _linf_prox_all!(y, f::GroupNormLinf, x, γ::Real, ::Val{V}) where {V}
  if y !== x       # the entries in no group are unpenalized
    @inbounds for i in f.free
      y[i] = x[i]
    end
    @inbounds for i in length(f.group_of)+1:length(x)
      y[i] = x[i]
    end
  end
  T = float(eltype(y))
  v = zero(T)
  λ0, sc, maxlen = f.λ0, f.scratch, f.maxlen
  for k in eachindex(f.groups)
    G = @inbounds f.groups[k]
    w = @inbounds f.w[k]
    m = _linf_group_prox!(y, x, G, λ0, w, γ, sc, maxlen)
    V && (v += _linf_fused_term(λ0, w, y, G, m, T))
  end
  V ? _group_value_checked(f, y, v) : nothing
end

# The term of the fused value, from the largest stored magnitude m that the step returned: the zero code gives zero, a
# magnitude (zero and Inf included) gives its term, and the recompute code and a NaN go out of line.
@inline function _linf_fused_term(λ0, w, y, G, m, ::Type{T}) where {T}
  m == T(_L2_ZERO) && return zero(T)
  m >= zero(T) && return _linf_term(λ0, w, m, T)
  _linf_term_outofline(λ0, w, y, G, m, T)
end
@noinline function _linf_term_outofline(λ0, w, y, G, m, ::Type{T}) where {T}
  m == T(_L2_RECOMPUTE) ? _linf_group_term(λ0, w, y, G, T) : _linf_term(λ0, w, m, T)
end

# The compact block: u and out have the length of group k, whose entries they index as 1:m. out is the buffer of the threshold
# when it is an Array of the output type that does not overlap u (as for NormLinf, so not when out === u), and the scratch
# of the function otherwise.
function prox_block!(out::AbstractVector, f::GroupNormLinf, u::AbstractVector, k::Integer, γ::Real)
  m = _check_group_block(out, f, u, k, γ)
  T = float(eltype(out))
  w = @inbounds f.w[k]
  if _linf_buffer_is_output(out, u, T)
    _linf_group_prox!(out, u, Base.OneTo(m), f.λ0, w, γ, out, f.maxlen)
  else
    _linf_group_prox!(out, u, Base.OneTo(m), f.λ0, w, γ, f.scratch, f.maxlen)
  end
  out
end

####################################
#  Optimality
####################################

# BlockIterate(x, f), the three-argument form and the symmetric iterate follow GroupNormL2. At a zero group the score is
# the margin ‖v[G]‖₁ - c of the dual (ℓ1) norm, with c = λ0 w in the working type W; a nonzero group has no score here, and
# the full-point form is not provided: a maximum of the margins would carry no meaning that a solver could use as a
# stationarity distance.
kkt_violation(f::GroupNormLinf, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); _not_separable_error(f))

@noinline _linf_nonzero_error(f) = throw(ArgumentError(
  "kkt_violation(f, v, x, k) of $(nameof(typeof(f))) is provided at a zero group only: the score of a nonzero group is not provided"))
@noinline _linf_point_error(f) = throw(ArgumentError(
  "kkt_violation(f, v, x) of $(nameof(typeof(f))) is not provided: use kkt_violation(f, v, x, k) at a zero group"))

# the margin of the zero group G for the gradient entries v read in W: NaN for a NaN entry of v[G], Inf for an infinite
# entry (or a sum of magnitudes that overflows), also when c = Inf, and -Inf for c = Inf with finite entries. A zero weight
# gives c = 0 and the margin ‖v[G]‖₁.
@inline function _linf_zero_score(λ0, w, v, G, ::Type{W}) where {W}
  c = _kkt_coefficient(W, λ0, w)
  s = _linf_sum(_IndexedValues(v, G, W), _AbsRead{W}(), W)
  isnan(s) && return W(NaN)
  isinf(s) && return W(Inf)
  isinf(c) && return W(-Inf)
  s - c
end
@inline function _linf_group_score(f::GroupNormLinf, k::Integer, v, x)
  G = @inbounds f.groups[k]
  W = float(promote_type(eltype(v), eltype(x)))
  _l2_zero_group(x, G, W) || _linf_nonzero_error(f)
  _linf_zero_score(f.λ0, @inbounds(f.w[k]), v, G, W)
end

function _linf_score(f::GroupNormLinf, g::AbstractArray, x::AbstractArray, i::Integer)
  length(g) == length(x) || _gradient_length_error(g, x)
  _check_group_extent(f, x)
  checkbounds(x, i)
  k = i <= length(f.group_of) ? @inbounds(f.group_of[i]) : 0
  k == 0 && return abs(convert(float(promote_type(eltype(g), eltype(x))), g[i]))
  _linf_group_score(f, k, g, x)
end
function _linf_score(f::GroupNormLinf, g::AbstractArray, x::BlockIterate, k::Integer)
  ncoordinates(x) == length(f.groups) || _block_count_error(x, f)
  B = blockindices(x, k)          # BoundsError for an invalid group
  B == @inbounds(f.groups[k]) || _block_layout_error(f, k)
  length(g) == length(x) || _gradient_length_error(g, x)
  _linf_group_score(f, k, g, x)
end
_kkt_violation(f::GroupNormLinf, g::AbstractArray, x::AbstractArray, i::Integer) = _linf_score(f, g, x, i)
_kkt_violation(f::GroupNormLinf, g::AbstractArray, x::BlockIterate, k::Integer) = _linf_score(f, g, x, k)
_kkt_violation(f::GroupNormLinf, ::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  throw(ArgumentError("GroupNormLinf groups linear indices; it does not apply to the coordinates of a SymmetricSparseIterate"))

kkt_violation(f::GroupNormLinf, ::AbstractArray, ::AbstractArray) = _linf_point_error(f)
kkt_violation(f::GroupNormLinf, ::AbstractArray, ::SymmetricSparseIterate) = _kkt_symmetric_point_error(f)
