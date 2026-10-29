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

function _soc_value(f::IndSOC, x::AbstractArray, polar::Bool)
  T = float(eltype(x))
  isfinite(_maximum_abs(x)) || return T(Inf)
  # Convert integer entries before taking a magnitude or negating the head.
  z = _ScaledArray(x, one(T))
  if f.groups === nothing
    return _ind(_soc_contains(z, 1:length(x), polar), T)
  end
  for G in f.groups
    _soc_contains(z, G, polar) || return T(Inf)
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
  T = float(promote_type(eltype(y), eltype(x)))
  z = _ScaledArray(x, one(T))
  if f.groups === nothing
    return _soc_project!(y, z, 1:length(x))
  end
  y === x || copyto!(y, x)
  for G in f.groups
    _soc_project!(y, z, G)
  end
  y
end
