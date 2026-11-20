"""
    SparseGroupNormL2(λ1, λ2, groups[, w])

Sparse-group penalty λ1 Σᵢ∈U |xᵢ| + λ2 Σₖ wₖ‖x[Gₖ]‖₂, where U
is the union of the groups. Entries outside U are free. Groups are
nonempty, disjoint lists or ranges of positive linear indices. Levels and
weights are nonnegative and cannot be NaN. The constructor copies groups
and weights, preserves their order, and defaults to wₖ = √|Gₖ|.

Array and compact-block proximal operators apply soft thresholding followed
by Euclidean group shrinkage. Compact buffers follow the order of groups[k].
Use an independent output buffer for [`prox_block!`](@ref). Both proximal
interfaces accept a scalar nonnegative step and support exact input/output
identity. Full-array copies into a floating-point SparseIterate explicitly
store negative zero where needed. Infinite levels or weights can impose
hard-zero constraints even at zero step.

The point-aware [`kkt_violation`](@ref) reports signed slack at a zero group
and Euclidean subgradient distance at an active group. Array points select
a group by entry index and BlockIterate points select it by group number.
The full-point score takes the maximum with zero across group scores and
free-entry gradient magnitudes.

See [sparse-group operations](@ref sparse-group-l2-guide) and
[endpoint rules](@ref sparse-group-l2-endpoints).
"""
struct SparseGroupNormL2{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractFunction
  λ1::T
  λ2::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}
  free::Vector{Int}

  function SparseGroupNormL2{T, G}(λ1::T, λ2::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    _check_one_based_groups(groups)
    length(w) == length(groups) ||
      throw(DimensionMismatch("SparseGroupNormL2 needs one weight per group: $(length(groups)) groups and $(length(w)) weights"))
    _checknonneg("λ1", λ1)
    _checknonneg("λ2", λ2)
    _checknonneg("the weights", w)
    group_of = _group_of(groups)
    new{T, G}(λ1, λ2, groups, w, group_of, findall(iszero, group_of))
  end
end

function SparseGroupNormL2(λ1::Real, λ2::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector)
  _check_one_based(w)
  eltype(w) <: Real || _real_error(:SparseGroupNormL2)
  T = float(promote_type(typeof(λ1), typeof(λ2), eltype(w)))
  G = _own_groups(groups)
  SparseGroupNormL2{T, typeof(G)}(T(λ1), T(λ2), G, Vector{T}(w))
end

function SparseGroupNormL2(λ1::Real, λ2::Real, groups::AbstractVector{<:AbstractVector{<:Integer}})
  T = float(promote_type(typeof(λ1), typeof(λ2)))
  G = _own_groups(groups)
  w = T[sqrt(T(length(g))) for g in G]
  SparseGroupNormL2{T, typeof(G)}(T(λ1), T(λ2), G, w)
end

isproximable(::Type{<:SparseGroupNormL2}) = true
isconvex(::Type{<:SparseGroupNormL2}) = true

BlockIterate(p::Integer, f::SparseGroupNormL2) = BlockIterate(Float64, p, f)
BlockIterate(::Type{T}, p::Integer, f::SparseGroupNormL2) where {T} = BlockIterate(T, p, f.groups)
BlockIterate(x::AbstractVector, f::SparseGroupNormL2) = BlockIterate(x, f.groups)

@inline _check_group_extent(f::SparseGroupNormL2, x) =
  (length(x) >= length(f.group_of) || _group_extent_error(f, x); nothing)
_checkparams(f::SparseGroupNormL2, x) = (_check_group_extent(f, x); true)
_checkparams(::SparseGroupNormL2, ::SymmetricSparseIterate) = true

function (f::SparseGroupNormL2)(x::AbstractArray)
  _checkreal(f, x)
  Base.require_one_based_indexing(x)
  _check_group_extent(f, x)
  _sparse_group_value(f, x)
end

# Accumulate ℓ₁ magnitudes before applying their common level. Free entries
# do not enter either sum, even where they are NaN or infinite.
function _sparse_group_value(f, x)
  T = _valtype(x)
  λ1, λ2 = f.λ1, f.λ2
  R = _working_type(T, λ1, λ2)
  a, b = zero(R), zero(T)
  for k in eachindex(f.groups)
    G = @inbounds f.groups[k]
    w = @inbounds f.w[k]
    if !iszero(λ1)
      group_sum = zero(R)
      @inbounds @simd for i in G
        group_sum += abs(convert(R, x[i]))
      end
      a += group_sum
    end
    if !iszero(λ2) || isinf(w)
      b += _l2_term(λ2, w, x, G, _L2_RECOMPUTE, T)
    end
  end
  v = convert(T, _weighted_value(convert(R, λ1) * a, a)) + b
  if _l2_ordinary(v, T) && (!_underflow_checked(T) || _sum_kept(v, length(f.groups) + 1))
    return v
  end
  _sparse_group_value_cold(f, x, v)
end

# Keep the proven class for genuine nonfinite values. For finite inputs,
# recover products and sums that crossed a range limit in the ordinary path.
@noinline function _sparse_group_value_cold(f, x, v::T)::T where {T}
  isnan(v) && return v
  λ1, λ2 = f.λ1, f.λ2
  R = _recovery_type(T, typeof(λ1), float(eltype(x)))
  acc = _expsum_start(R)
  for k in eachindex(f.groups)
    G = @inbounds f.groups[k]
    w = @inbounds f.w[k]
    if !iszero(λ1)
      @inbounds for i in G
        a = abs(convert(R, x[i]))
        iszero(a) && continue
        (isfinite(a) && isfinite(λ1)) || return v
        acc = _expsum_product(acc, λ1, a)
      end
    end
    if !iszero(w) && (!iszero(λ2) || isinf(w))
      finite = isfinite(w) && isfinite(λ2)
      e, n = _l2_classified_norm(x, G, R, finite)
      e == _EXPONENT_ZERO && continue
      (finite && e < _EXPONENT_INF) || return v
      acc = _expsum_term(acc, λ2, w, n, 0, e)
    end
  end
  _expsum_value(T, acc)
end

function prox!(y::AbstractArray, f::SparseGroupNormL2, x::AbstractArray, γ::Real)
  _check_group_arrays(f, y, x, γ)
  _sparse_group_prox!(y, f, x, γ)
end

function value_and_prox!(y::AbstractArray, f::SparseGroupNormL2, x::AbstractArray, γ::Real)
  prox!(y, f, x, γ)
  _sparse_group_value(f, y)
end
_value_and_prox_unchecked!(y::AbstractArray, f::SparseGroupNormL2, x::AbstractArray, γ::Real) =
  value_and_prox!(y, f, x, γ)

for op in (:prox!, :value_and_prox!)
  @eval begin
    $op(::SymmetricSparseIterate, f::SparseGroupNormL2, x, ::Real) = _symmetric_prox_error(f)
    $op(::AbstractArray, f::SparseGroupNormL2, ::SymmetricSparseIterate, ::Real) = _symmetric_prox_error(f)
  end
end
prox(f::SparseGroupNormL2, ::SymmetricSparseIterate, ::Real = 1) = _symmetric_prox_error(f)

@inline function _sparse_group_prox!(y, f, x, γ)
  if y !== x
    _sparse_group_copy!(y, x, f.free)
    _sparse_group_copy!(y, x, (length(f.group_of) + 1):length(x))
  end
  c1 = _sparse_group_threshold(y, f, x, γ)
  for k in eachindex(f.groups)
    @inbounds _sparse_group_shrink!(y, x, f.groups[k], f.λ1, f.λ2, f.w[k], γ, c1, _sparse_group_copy!)
  end
  y
end

@inline _sparse_group_copy!(y, x, G) = _l2_copy!(y, x, G)

# An absent sparse coordinate reads positive zero. Keep a copy of
# negative zero in an explicit slot, leaving other writes to the iterate.
@inline function _sparse_group_copy!(y::SparseIterate{T}, x, G) where {T<:AbstractFloat}
  if y !== x
    @inbounds for i in G
      v = convert(T, x[i])
      iszero(v) && signbit(v) && activate!(y, i)
      y[i] = v
    end
  end
  _L2_RECOMPUTE
end

@inline function _sparse_group_threshold(y, f, x, γ)
  T = float(promote_type(eltype(y), eltype(x)))
  R = _working_type(T, γ, f.λ1)
  (iszero(γ) || isinf(γ) || iszero(f.λ1) || isinf(f.λ1)) && return zero(R)
  c, normal = _threshold_product(R, γ, f.λ1, true)
  normal ? c : _weight_product(R, γ, f.λ1, true)
end

@inline _sparse_group_hard(λ1, λ2, w) = isinf(λ1) || isinf(w) || (isinf(λ2) && !iszero(w))

@inline function _sparse_group_shrink!(y, x, G, λ1, λ2, w, γ, c1, copy_group!::F = _l2_copy!) where {F}
  _sparse_group_hard(λ1, λ2, w) && return _l2_zero!(y, G)
  coupled = !iszero(λ2) && !iszero(w)
  (iszero(γ) || (iszero(λ1) && !coupled)) && return copy_group!(y, x, G)
  isinf(γ) && return _l2_zero!(y, G)
  iszero(λ1) && return _l2_shrink!(y, x, G, λ2, w, γ)

  _sparse_group_soft!(y, x, G, c1, coupled)
  coupled && _l2_shrink!(y, y, G, λ2, w, γ)
  nothing
end

@inline function _sparse_group_soft!(y, x, G, c::T, coupled) where {T}
  if isfinite(c) || !coupled
    @inbounds @simd for i in G
      y[i] = _shrink_nonneg(convert(T, x[i]), c)
    end
  else
    # The factors are finite here. A threshold beyond the range zeros
    # finite inputs but does not remove a genuine infinite entry.
    @inbounds for i in G
      v = convert(T, x[i])
      y[i] = isfinite(v) ? zero(T) : v
    end
  end
  nothing
end

function prox_block!(out::AbstractVector, f::SparseGroupNormL2, u::AbstractVector, k::Integer, γ::Real)
  m = _check_group_block(out, f, u, k, γ)
  c1 = _sparse_group_threshold(out, f, u, γ)
  _sparse_group_shrink!(out, u, Base.OneTo(m), f.λ1, f.λ2, @inbounds(f.w[k]), γ, c1)
  out
end

kkt_violation(f::SparseGroupNormL2, v::AbstractArray, i::Integer) =
  (eltype(v) <: Real || _real_error(:kkt_violation); _not_separable_error(f))

@noinline _sparse_group_symmetric_kkt(f) =
  throw(ArgumentError("kkt_violation with $(nameof(typeof(f))) is not defined for SymmetricSparseIterate points"))
kkt_violation(f::SparseGroupNormL2, ::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  _sparse_group_symmetric_kkt(f)
kkt_violation(f::SparseGroupNormL2, ::AbstractArray, ::SymmetricSparseIterate) =
  _sparse_group_symmetric_kkt(f)

@inline function _check_sparse_group_kkt(f, v, x)
  _check_one_based(v, x)
  eltype(v) <: Real || _real_error(:kkt_violation)
  eltype(x) <: Real || _real_error(:kkt_violation)
  length(v) == length(x) || _gradient_length_error(v, x)
  _check_group_extent(f, x)
  nothing
end

function kkt_violation(f::SparseGroupNormL2, v::AbstractArray, x::AbstractArray, i::Integer)
  _check_sparse_group_kkt(f, v, x)
  checkbounds(x, i)
  W = float(promote_type(eltype(v), eltype(x)))
  k = i <= length(f.group_of) ? @inbounds(f.group_of[i]) : 0
  k == 0 ? abs(convert(W, v[i])) : _sparse_group_score(f, v, x, k, W)
end

function kkt_violation(f::SparseGroupNormL2, v::AbstractArray, x::BlockIterate, k::Integer)
  _check_sparse_group_kkt(f, v, x)
  ncoordinates(x) == length(f.groups) || _block_count_error(x, f)
  _check_sparse_group_layout(f, x, k)
  _sparse_group_score(f, v, x, k, float(promote_type(eltype(v), eltype(x))))
end

@inline function _check_sparse_group_layout(f, x, k)
  B = blockindices(x, k)
  B == @inbounds(f.groups[k]) || _block_layout_error(f, k)
  nothing
end

_check_sparse_group_layout(f, x) = nothing
function _check_sparse_group_layout(f, x::BlockIterate)
  ncoordinates(x) == length(f.groups) || _block_count_error(x, f)
  for k in eachindex(f.groups)
    _check_sparse_group_layout(f, x, k)
  end
  nothing
end

function kkt_violation(f::SparseGroupNormL2, v::AbstractArray, x::AbstractArray)
  _check_sparse_group_kkt(f, v, x)
  _check_sparse_group_layout(f, x)
  W = float(promote_type(eltype(v), eltype(x)))
  score = zero(W)
  for k in eachindex(f.groups)
    score = max(score, _sparse_group_score(f, v, x, k, W))
  end
  @inbounds for i in f.free
    score = max(score, abs(convert(W, v[i])))
  end
  @inbounds for i in (length(f.group_of) + 1):length(x)
    score = max(score, abs(convert(W, v[i])))
  end
  score
end

# A zero group has a signed soft-screen. The soft read is lazy so the
# existing screen can classify and rescale the norm and coefficient.
struct _SparseGroupSoft{T, V} <: AbstractVector{T}
  v::V
  c::T
end
Base.size(s::_SparseGroupSoft) = size(s.v)
Base.IndexStyle(::Type{<:_SparseGroupSoft}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(s::_SparseGroupSoft, i::Int) = _shrink_nonneg(s.v[i], s.c)

@inline function _sparse_group_score(f, v, x, k, ::Type{W}) where {W}
  G = @inbounds f.groups[k]
  w = @inbounds f.w[k]
  iszero(f.λ1) && return _l2_score(f.λ2, w, v, x, G)
  c1 = convert(W, f.λ1)
  c2 = _kkt_coefficient(W, f.λ2, w)
  if _l2_zero_group(x, G, W)
    vw = _IndexedValues(v, G, W)
    n = Base.OneTo(length(G))
    if isinf(c1)
      e, _ = _l2_classified_norm(vw, n, W, false)
      e == _EXPONENT_NAN && return W(NaN)
      return e == _EXPONENT_INF ? W(Inf) : W(-Inf)
    end
    return _l2_zero_score(f.λ2, w, _SparseGroupSoft(vw, c1), n, W)
  end

  sx = _indexed_sumabs2(x, G, W)
  if isfinite(sx) & _squares_normal(sx, length(G), W) & isfinite(c1) & isfinite(c2)
    t = c2 / sqrt(sx)
    if isfinite(t)
      s = zero(W)
      @inbounds @simd for i in G
        xi, vi = convert(W, x[i]), convert(W, v[i])
        s += abs2(_sparse_group_residual(vi, xi, t * xi, c1))
      end
      (isfinite(s) & _squares_normal(s, length(G), W)) && return sqrt(s)
    end
  end
  _sparse_group_score_cold(v, x, G, c1, c2)
end

@inline _sparse_group_residual(v, x, d, c) =
  ifelse(iszero(x), _shrink_nonneg(v, c), (v + d) + copysign(c, x))

struct _SparseGroupResiduals{T, V, X, D} <: AbstractVector{T}
  v::V
  x::X
  direction::D
  c1::T
  c2::T
end
Base.size(r::_SparseGroupResiduals) = size(r.x)
Base.IndexStyle(::Type{<:_SparseGroupResiduals}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(r::_SparseGroupResiduals, i::Int)
  x = r.x[i]
  # Each directional coordinate is bounded by c₂, up to rounding in the norm.
  d = clamp(_apply_scale(r.direction, x), -r.c2, r.c2)
  _sparse_group_residual(r.v[i], x, d, r.c1)
end

@noinline function _sparse_group_score_cold(v, x, G, c1::W, c2::W) where {W}
  R = _recovery_type(W)
  n = Base.OneTo(length(G))
  vw, xw = _IndexedValues(v, G, W), _IndexedValues(x, G, W)
  ev, _ = _l2_classified_norm(vw, n, R, false)
  ex, nx = _l2_classified_norm(xw, n, R)
  (ev == _EXPONENT_NAN || ex == _EXPONENT_NAN) && return W(NaN)
  (ev == _EXPONENT_INF || ex == _EXPONENT_INF || isinf(c1) || isinf(c2)) && return W(Inf)

  # Keep the norm's exponent separate from the directional coefficient
  # to avoid overflow or underflow before multiplying by each coordinate.
  m, e = _frexp(convert(R, c2))
  direction = _ProjectionScale(m / nx, Int(e - ex))
  vs, xs = _IndexedValues(vw, n, R), _IndexedValues(xw, n, R)
  residuals = _SparseGroupResiduals(vs, xs, direction, convert(R, c1), convert(R, c2))
  er, nr = _l2_classified_norm(residuals, n, R)
  er == _EXPONENT_ZERO && return zero(W)
  er == _EXPONENT_INF && return W(Inf)
  convert(W, _scaled_product(nr, one(R), er))
end
