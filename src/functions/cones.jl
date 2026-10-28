####################################
#
#  Second-order cone
#
####################################

"""
    IndSOC()
    IndSOC(groups)

The indicator of the second-order cone `{x : ‖x[2:end]‖₂ ≤ x[1]}`. The
argument is a real array, read by one-based linear index, with its first entry
as the head. An empty argument has no head and throws an `ArgumentError`.
Its proximal operator is the Euclidean projection: with `t = x[1]` and
`s = ‖x[2:end]‖₂`, keep `x` when `s ≤ t`, return zero when `s ≤ -t`, and
otherwise return `(α, (α/s) x[2:end])`, where `α = (t + s)/2`.

With `groups`, project onto a product of cones over disjoint groups of linear
indices, as for [`GroupNormL2`](@ref). The first listed index of each group is
its head, and the remaining indices its tail. Groups are copied and must be
nonempty; a group of one index is the half-line `t ≥ 0`. Entries in no group
are free and remain unchanged. An empty collection of groups leaves every
entry free. Only scalar steps are supported.

The value is zero when each block satisfies
`s ≤ t + tol max(1, |t|)`, with `tol = √eps(float(eltype(x)))`, and `Inf`
otherwise. Non-finite entries give value `Inf`; a block containing one
projects to `NaN` entries. [`Conjugate`](@ref) gives the indicator of the
polar cone: each grouped block is in the negative cone, and free entries
must be zero, within the same tolerance.
"""
struct IndSOC{G<:Union{Nothing, AbstractVector{<:AbstractVector{<:Integer}}}} <: AbstractFunction
  groups::G
  group_of::Vector{Int}

  function IndSOC{G}(groups::G) where {G}
    new{G}(groups, groups === nothing ? Int[] : _group_of(groups))
  end
end
IndSOC() = IndSOC{Nothing}(nothing)
function IndSOC(groups::AbstractVector{<:AbstractVector{<:Integer}})
  G = _own_groups(groups)
  IndSOC{typeof(G)}(G)
end

isproximable(::Type{<:IndSOC}) = true
isconvex(::Type{<:IndSOC}) = true
_isindicator(::Type{<:IndSOC}) = true

function _checkparams(f::IndSOC, x::AbstractArray)
  Base.require_one_based_indexing(x)
  eltype(x) <: Real || throw(ArgumentError("IndSOC requires real entries"))
  if f.groups === nothing
    isempty(x) && throw(ArgumentError("the second-order cone needs a head entry"))
  else
    length(x) >= length(f.group_of) || throw(BoundsError(x, length(f.group_of)))
  end
  true
end

function _checkwrite(op::O, f::IndSOC, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checksize(y, x)
  _checkparams(f, x)
  Base.require_one_based_indexing(y)
  eltype(y) <: Real || throw(ArgumentError("IndSOC requires real entries"))
  true
end

# Keep the usual group norm at ordinary scales. Exact powers of two also
# handle an overflowing norm and subnormal heads without rounding the halves.
function _soc_frame(x::AbstractArray{T}, G) where {T}
  t = x[first(G)]
  tail = view(G, 2:length(G))
  s = _group_norm2(x, tail)
  m = max(abs(t), s)
  isfinite(m) || (m = max(abs(t), _maxabs(x, tail)))
  if isfinite(m) && !iszero(m) &&
     (!isfinite(s) || m <= 2floatmin(T) || m > floatmax(T) / 2)
    k = exponent(m)
    a, b = _power_factors(T, -k)
    scaled = _ScaledArray(_ScaledArray(x, a), b)
    return ldexp(t, -k), _group_norm2(scaled, tail), k
  end
  t, s, 0
end

function _soc_contains(x::AbstractArray{T}, G, polar) where {T}
  t, s, k = _soc_frame(x, G)
  polar && (t = -t)
  slack = max(ldexp(_tol(T), -k), _tol(T) * abs(t))
  s <= t + slack
end

function _cone_value(f, x::AbstractArray, polar::Bool, contains::F) where {F}
  T = float(eltype(x))
  isfinite(_maximum_abs(x)) || return T(Inf)
  # Convert integer entries before taking a magnitude or negating the heads.
  z = _ScaledArray(x, one(T))
  if f.groups === nothing
    return _ind(contains(z, 1:length(x), polar), T)
  end
  for G in f.groups
    contains(z, G, polar) || return T(Inf)
  end
  if polar
    @inbounds for i in eachindex(IndexLinear(), x)
      if i > length(f.group_of) || iszero(f.group_of[i])
        abs(z[i]) <= _tol(T) || return T(Inf)
      end
    end
  end
  zero(T)
end

_soc_value(f::IndSOC, x::AbstractArray, polar::Bool) =
  _cone_value(f, x, polar, _soc_contains)

(f::IndSOC)(x::AbstractArray) = (_checkparams(f, x); _soc_value(f, x, false))

function _soc_project!(y, x::AbstractArray{T}, G) where {T}
  t, s, k = _soc_frame(x, G)
  if !isfinite(t) || !isfinite(s)
    @inbounds for i in G
      y[i] = T(NaN)
    end
  elseif s <= t
    @inbounds for i in G
      y[i] = x[i]
    end
  elseif s <= -t
    @inbounds for i in G
      y[i] = zero(T)
    end
  else
    α = t / 2 + s / 2
    c = α / s
    @inbounds y[first(G)] = ldexp(α, k)
    # Apply the contraction to the original entries: scaling the tail back
    # would lose tiny entries alongside very large ones.
    @inbounds for i in Iterators.drop(G, 1)
      y[i] = c * x[i]
    end
  end
  y
end

function prox!(y::AbstractArray, f::IndSOC, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  _cone_project!(y, f, x, _soc_project!)
end

function _cone_project!(y, f, x, project!::F) where {F}
  T = float(promote_type(eltype(y), eltype(x)))
  z = _ScaledArray(x, one(T))
  if f.groups === nothing
    return project!(y, z, 1:length(x))
  end
  y === x || copyto!(y, x)
  for G in f.groups
    project!(y, z, G)
  end
  y
end

####################################
#
#  Rotated second-order cone
#
####################################

"""
    IndRotatedSOC()
    IndRotatedSOC(groups)

The indicator of the rotated second-order cone
`{(u, v, z) : ‖z‖₂² ≤ 2uv, u ≥ 0, v ≥ 0}`. The argument is a real array,
read by one-based linear index, with its first two entries as `u` and `v`.
It needs at least two entries. Its proximal operator is the Euclidean
projection, obtained from [`IndSOC`](@ref) by the orthogonal map
`(u, v, z) ↦ ((u + v)/√2, (u - v)/√2, z)` without forming the map.

With `groups`, project onto a product of cones over disjoint groups of linear
indices, as for [`IndSOC`](@ref). Groups are copied and need at least two
indices; the first two listed indices are the heads. A group of two indices
is the nonnegative quadrant. Entries in no group are free and remain
unchanged. An empty collection of groups leaves every entry free. Only
scalar steps are supported.

The value is zero within tolerance `tol = √eps(float(eltype(x)))`, and `Inf`
otherwise. The heads satisfy `u, v ≥ -tol max(1, |u|, |v|)`. In a frame
scaled by a power `2^k`, put `a = u/2^k`, `b = v/2^k`, `s = ‖z/2^k‖₂`.
The quadratic inequality is checked as
`hypot(a - b, √2 s) ≤ a + b + tol max(2^(-k), |a|, |b|)`.
Non-finite entries give value `Inf`; a block containing one projects to
`NaN` entries. [`Conjugate`](@ref) gives the polar indicator: the negative
cone on each group and zero on every free entry, within the same tolerance.
The single cone is self-dual.
"""
struct IndRotatedSOC{G<:Union{Nothing, AbstractVector{<:AbstractVector{<:Integer}}}} <: AbstractFunction
  groups::G
  group_of::Vector{Int}

  function IndRotatedSOC{G}(groups::G) where {G}
    if groups !== nothing
      all(G -> length(G) >= 2, groups) ||
        throw(ArgumentError("rotated second-order cone groups need two head entries"))
    end
    new{G}(groups, groups === nothing ? Int[] : _group_of(groups))
  end
end
IndRotatedSOC() = IndRotatedSOC{Nothing}(nothing)
function IndRotatedSOC(groups::AbstractVector{<:AbstractVector{<:Integer}})
  G = _own_groups(groups)
  IndRotatedSOC{typeof(G)}(G)
end

isproximable(::Type{<:IndRotatedSOC}) = true
isconvex(::Type{<:IndRotatedSOC}) = true
_isindicator(::Type{<:IndRotatedSOC}) = true

function _checkparams(f::IndRotatedSOC, x::AbstractArray)
  Base.require_one_based_indexing(x)
  eltype(x) <: Real || throw(ArgumentError("IndRotatedSOC requires real entries"))
  if f.groups === nothing
    length(x) >= 2 || throw(ArgumentError("the rotated second-order cone needs two head entries"))
  else
    length(x) >= length(f.group_of) || throw(BoundsError(x, length(f.group_of)))
  end
  true
end

function _checkwrite(op::O, f::IndRotatedSOC, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checksize(y, x)
  _checkparams(f, x)
  Base.require_one_based_indexing(y)
  eltype(y) <: Real || throw(ArgumentError("IndRotatedSOC requires real entries"))
  true
end

# Scale before adding the heads, including when the tail norm would overflow.
function _rotatedsoc_frame(x::AbstractArray{T}, G) where {T}
  u, v = x[G[1]], x[G[2]]
  m = _maxabs(x, G)
  isfinite(m) || return u, v, m, 0
  k = _magnitude_exponent(m)
  a, b = _power_factors(T, -k)
  scaled = _ScaledArray(_ScaledArray(x, a), b)
  ldexp(u, -k), ldexp(v, -k), _group_norm2(scaled, view(G, 3:length(G))), k
end

function _rotatedsoc_contains(x::AbstractArray{T}, G, polar) where {T}
  u, v, s, k = _rotatedsoc_frame(x, G)
  if polar
    u, v = -u, -v
  end
  slack = max(ldexp(_tol(T), -k), _tol(T) * max(abs(u), abs(v)))
  u >= -slack && v >= -slack && hypot(u - v, sqrt(T(2)) * s) <= u + v + slack
end

_rotatedsoc_value(f::IndRotatedSOC, x::AbstractArray, polar::Bool) =
  _cone_value(f, x, polar, _rotatedsoc_contains)
(f::IndRotatedSOC)(x::AbstractArray) = (_checkparams(f, x); _rotatedsoc_value(f, x, false))

# Compare with sqrt(2uv) in the norm's frame, retaining small original heads.
function _rotatedsoc_inside(u::T, v::T, s::T, k) where {T}
  u >= 0 && v >= 0 || return false
  bound = _positive_product(T(2), ldexp(u, -k), ldexp(v, -k))
  bound >= floatmin(T) && return s * s <= bound
  j = fld(k, 2)
  s <= _positive_product(sqrt(T(2)), ldexp(sqrt(u), -j), ldexp(sqrt(v), j - k))
end

function _rotatedsoc_project!(y, x::AbstractArray{T}, G) where {T}
  u, v = x[G[1]], x[G[2]]
  a, b, s, k = _rotatedsoc_frame(x, G)
  tail = view(G, 3:length(G))
  if !isfinite(a) || !isfinite(b) || !isfinite(s)
    @inbounds for i in G
      y[i] = T(NaN)
    end
  elseif _allzero(x, tail)
    @inbounds y[G[1]] = max(u, zero(T))
    @inbounds y[G[2]] = max(v, zero(T))
    @inbounds for i in tail
      y[i] = x[i]
    end
  elseif _rotatedsoc_inside(u, v, s, k)
    @inbounds for i in G
      y[i] = x[i]
    end
  elseif _rotatedsoc_inside(-u, -v, s, k)
    @inbounds for i in G
      y[i] = zero(T)
    end
  else
    d = hypot(a - b, sqrt(T(2)) * s)
    r = s / (d + abs(a - b))
    h = (d / 2 + abs(a - b) / 2) / d
    scale = ldexp(one(T), k)
    # Rationalizing the eigenvalue correction retains a small positive head
    # beside a much larger negative head.
    λ = max(max(u, v) + _positive_product(s, r, scale), zero(T))
    if isfinite(λ)
      hi = λ * h
      lo = _positive_product(λ, r, s / d)
      c = ldexp(λ, -k) / d
    else
      λs = max(a, b) + s * r
      hi = ldexp(λs * h, k)
      lo = _positive_product(scale, r, λs * (s / d))
      c = λs / d
    end
    # This SOC contraction is stable here and exact at zero head sum.
    if a + b >= 0
      c = one(T) / 2 + (a / 2 + b / 2) / d
    end
    # Equal heads stay equal, including rounded subnormal outputs.
    u == v && (lo = hi)
    @inbounds y[G[1]] = u >= v ? hi : lo
    @inbounds y[G[2]] = u >= v ? lo : hi
    # Contract original entries, so a small tail survives beside large heads.
    @inbounds for i in tail
      if c < floatmin(T) && isfinite(λ)
        y[i] = copysign(_positive_product(λ, abs(ldexp(x[i], -k)), inv(d)), x[i])
      else
        y[i] = c * x[i]
      end
    end
  end
  y
end

function prox!(y::AbstractArray, f::IndRotatedSOC, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  _cone_project!(y, f, x, _rotatedsoc_project!)
end
