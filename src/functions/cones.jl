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

####################################
#
#  Exponential cones
#
####################################

"""
    IndExpPrimal()
    IndExpPrimal(groups)

The indicator of the exponential cone
`K = cl{(r, s, t) : s > 0, s exp(r/s) ≤ t}`. Its face at `s = 0` is
`{(r, 0, t) : r ≤ 0, t ≥ 0}`. The argument is a real array with exactly
three entries, read by one-based linear index in the order `(r, s, t)`.
The proximal operator is the Euclidean projection, using Friberg's (2023)
univariate root problem with a finite bracket and safeguarded Newton steps.

With `groups`, project disjoint triples of indices in their listed order.
Groups are copied and must each contain exactly three positive indices.
Entries in no group are free and remain unchanged; an empty collection of
groups leaves every entry free. Only scalar steps are supported.

For finite entries, write `δ = √eps(float(eltype(x))) max(1, |r|, |s|, |t|)`.
The value is zero when either `|s| ≤ δ, r ≤ δ, t ≥ -δ`, or
`s > 0, t ≥ -δ, r ≤ s(log(max(t, 0) + δ) - log(s)) + δ`;
otherwise it is `Inf`. The slack in `t` includes rounded underflow to zero.
Projection decisions use exact inequalities rather than this value tolerance.
Non-finite entries give value `Inf` and project to `NaN` within their triple.
Free entries are copied, including non-finite ones.

[`Conjugate`](@ref) gives the indicator of the polar cone `-K*`, with zero
on free entries within the same tolerance. Thus the grouped constructors
[`IndExpDual`](@ref) and `IndExpPrimal` are not duals when free entries exist.
Exponentials and distances are evaluated with scaling. Finite inputs have
finite projections when the result is representable in the output type;
an unrepresentable output component can overflow.
"""
struct IndExpPrimal{G<:Union{Nothing, AbstractVector{<:AbstractVector{<:Integer}}}} <: AbstractFunction
  groups::G
  group_of::Vector{Int}

  function IndExpPrimal{G}(groups::G) where {G}
    _check_exp_groups(groups)
    new{G}(groups, groups === nothing ? Int[] : _group_of(groups))
  end
end
IndExpPrimal() = IndExpPrimal{Nothing}(nothing)
function IndExpPrimal(groups::AbstractVector{<:AbstractVector{<:Integer}})
  G = _own_groups(groups)
  IndExpPrimal{typeof(G)}(G)
end

"""
    IndExpDual()
    IndExpDual(groups)

The indicator of the dual exponential cone
`K* = cl{(u, v, w) : u < 0, -u exp(v/u) ≤ exp(1) w}`. Its face at `u = 0`
is `{(0, v, w) : v ≥ 0, w ≥ 0}`. The argument has exactly three real
entries in the order `(u, v, w)`, read by one-based linear index.

The projection uses `P_{K*}(x) = x + P_K(-x)`, reconstructing the polar
component directly to avoid cancellation. Groups, scalar steps, copying,
non-finite inputs and representability follow [`IndExpPrimal`](@ref).
With `δ = √eps(float(eltype(x))) max(1, |u|, |v|, |w|)`, the value is zero
when either `|u| ≤ δ, v ≥ -δ, w ≥ -δ`, or
`u < 0, w ≥ -δ, -v ≤ (-u)(1 + log(max(w, 0) + δ) - log(-u)) + δ`.
Otherwise it is `Inf`. Projection decisions use exact inequalities.

[`Conjugate`](@ref) gives the indicator of `-K`, requiring zero on free
entries within the value tolerance. The grouped constructors are not each
other's duals when free entries exist.
"""
struct IndExpDual{G<:Union{Nothing, AbstractVector{<:AbstractVector{<:Integer}}}} <: AbstractFunction
  groups::G
  group_of::Vector{Int}

  function IndExpDual{G}(groups::G) where {G}
    _check_exp_groups(groups)
    new{G}(groups, groups === nothing ? Int[] : _group_of(groups))
  end
end
IndExpDual() = IndExpDual{Nothing}(nothing)
function IndExpDual(groups::AbstractVector{<:AbstractVector{<:Integer}})
  G = _own_groups(groups)
  IndExpDual{typeof(G)}(G)
end

_check_exp_groups(::Nothing) = true
function _check_exp_groups(groups)
  all(G -> length(G) == 3, groups) ||
    throw(ArgumentError("exponential cone groups must contain exactly three indices"))
  true
end

const _ExpCone = Union{IndExpPrimal, IndExpDual}
isproximable(::Type{<:_ExpCone}) = true
isconvex(::Type{<:_ExpCone}) = true
_isindicator(::Type{<:_ExpCone}) = true

function _checkparams(f::_ExpCone, x::AbstractArray)
  Base.require_one_based_indexing(x)
  eltype(x) <: Real || throw(ArgumentError("exponential cones require real entries"))
  if f.groups === nothing
    length(x) == 3 || throw(ArgumentError("an exponential cone needs exactly three entries"))
  else
    length(x) >= length(f.group_of) || throw(BoundsError(x, length(f.group_of)))
  end
  true
end

function _checkwrite(op::O, f::_ExpCone, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checksize(y, x)
  _checkparams(f, x)
  Base.require_one_based_indexing(y)
  eltype(y) <: Real || throw(ArgumentError("exponential cones require real entries"))
  true
end

# Logarithms of the individual coordinates avoid overflowing t/s. With no
# tolerance, use the original signs and a frame for just the linear terms.
function _exp_contains(r::T, s::T, t::T, polar::Bool, tol::T) where {T}
  if iszero(tol)
    if polar
      iszero(r) && s <= 0 && t <= 0 && return true
      r > 0 && t < 0 || return false
      m = max(abs(r), abs(s))
      return s / m <= (r / m) * (one(T) + log(-t) - log(r))
    end
    iszero(s) && r <= 0 && t >= 0 && return true
    s > 0 && t > 0 || return false
    m = max(abs(r), s)
    return r / m <= (s / m) * (log(t) - log(s))
  end
  m = max(one(T), abs(r), abs(s), abs(t))
  a, b, c = r / m, s / m, t / m
  if polar
    abs(a) <= tol && b <= tol && c <= tol && return true
    r > 0 && c <= tol || return false
    logw = log(m) + log(max(-c, zero(T)) + tol)
    return b <= a * (one(T) + logw - log(r)) + tol
  end
  abs(b) <= tol && a <= tol && c >= -tol && return true
  s > 0 && c >= -tol || return false
  logw = log(m) + log(max(c, zero(T)) + tol)
  a <= b * (logw - log(s)) + tol
end

function _exp_value(f::_ExpCone, x::AbstractArray, conjugate::Bool)
  T = float(eltype(x))
  isfinite(_maximum_abs(x)) || return T(Inf)
  dual = f isa IndExpDual
  polar = xor(dual, conjugate)
  negate = dual
  groups = f.groups === nothing ? (1:3,) : f.groups
  for G in groups
    @inbounds r, s, t = T(x[G[1]]), T(x[G[2]]), T(x[G[3]])
    negate && ((r, s, t) = (-r, -s, -t))
    _exp_contains(r, s, t, polar, _tol(T)) || return T(Inf)
  end
  if conjugate && f.groups !== nothing
    @inbounds for i in eachindex(IndexLinear(), x)
      if i > length(f.group_of) || iszero(f.group_of[i])
        abs(T(x[i])) <= _tol(T) || return T(Inf)
      end
    end
  end
  zero(T)
end

(f::_ExpCone)(x::AbstractArray) = (_checkparams(f, x); _exp_value(f, x, false))

# Evaluate a exp(b), even when exp(b) alone is outside the numerical range.
function _exp_product(a::T, b::T) where {T}
  iszero(a) && return a
  log(floatmin(T)) <= b < log(floatmax(T)) && return a * exp(b)
  m, k = frexp(a)
  logtwo = log(T(2))
  b > (exponent(floatmax(T)) - k + 2) * logtwo && return copysign(T(Inf), a)
  b < (exponent(nextfloat(zero(T))) - k - 2) * logtwo && return copysign(zero(T), a)
  j = round(Int, b / logtwo)
  ldexp(m * exp(muladd(-T(j), logtwo, b)), k + j)
end

function _exp_sum(a::T, b::T, c::T) where {T}
  v = (a + b) + c
  isfinite(v) ? v : (a / 2 + b / 2 + c / 2) * 2
end

_exp_distance(p, r, s, t, k) =
  hypot(ldexp(p[1], -k) - r, ldexp(p[2], -k) - s, ldexp(p[3], -k) - t)

# Compare squared distances by their difference. Subtracting the two norms
# loses the improvement when a small projection lies close to the origin.
function _exp_closer(p, q, r, s, t, k)
  a, b, c = ldexp(p[1], -k), ldexp(p[2], -k), ldexp(p[3], -k)
  all(isfinite, (a, b, c)) || return false
  u, v, w = ldexp(q[1], -k), ldexp(q[2], -k), ldexp(q[3], -k)
  muladd(a - u, (a + u) / 2 - r,
    muladd(b - v, (b + v) / 2 - s, (c - w) * ((c + w) / 2 - t))) <= 0
end

# Friberg, Section 5 and Algorithm 1 (paper order (t,s,r), here (r,s,t)).
# Scale r,s separately: scaling them by a much larger t can erase both.
# f and df share a positive factor, so f/df is the paper's Newton step.
function _exp_h(r, s, logt, signt, rho)
  shift = max(abs(rho), logt)
  ep, em, et = exp(rho - shift), exp(-rho - shift), signt * exp(logt - shift)
  d = muladd(rho, rho - 1, one(rho))
  f = muladd(muladd(rho - 1, r, s), ep,
             -muladd(muladd(-rho, s, r), em, d * et))
  df = muladd(muladd(rho, r, s), ep,
              muladd(muladd(1 - rho, s, r), em, -(2rho - 1) * et))
  f, df
end

function _exp_bracket(r, s, t, a, b, logt, pdist, ddist, limit)
  T = typeof(r)
  baselo = r > 0 ? one(T) - s / r : -T(Inf)
  basehi = s > 0 ? r / s : T(Inf)
  lo, hi = baselo, basehi
  # Corollaries 5.3--5.4: maxima of the linear-over-quadratic terms.
  z = sqrt(muladd(a, a - b, b * b))
  cp = 2b - a >= 0 ? (2b - a + 2z) / 3 : a * (a / (2z - 2b + a))
  cd = 2a - b >= 0 ? (2a - b + 2z) / 3 : b * (b / (2z - 2a + b))
  t > 0 && (lo = max(lo, logt - log(cp)))
  t < 0 && (hi = min(hi, log(cd) - logt))
  # Lemma 5.2 and Corollaries 5.5--5.6. Clamp negative radicands from
  # rounding; the sign checks below repair a bracket rounded inward.
  dp = sqrt(max(zero(T), (pdist - min(s, zero(T))) * (pdist + min(s, zero(T)))))
  dd = sqrt(max(zero(T), (ddist - min(r, zero(T))) * (ddist + min(r, zero(T)))))
  # Combine numerators before dividing to avoid Inf - Inf in the bounds.
  if r > 0 && isfinite(lo) && abs(lo) <= limit
    logomega = lo - log(muladd(lo, lo - 1, one(T)))
    lo < 2 && (logomega = min(logomega, T(2) - log(T(3))))
    cap = max(eps(T), min(dd, dp + t))
    hi = min(hi, max(lo, (r - s + exp(log(cap) - logomega)) / r))
  end
  if s > 0 && isfinite(hi) && abs(hi) <= limit
    logomega = -hi - log(muladd(hi, hi - 1, one(T)))
    hi > -1 && (logomega = min(logomega, one(T) - log(T(3))))
    cap = max(eps(T), min(dp, dd - t))
    lo = max(lo, min(hi, (r - exp(log(cap) - logomega)) / s))
  end
  # Weak bounds survive underflow in the strengthened bounds. The finite
  # window includes every root with representable off-face components.
  lower, upper = max(-limit, baselo), min(limit, basehi)
  lo, hi = min(lo, hi), max(lo, hi)
  pad = 8eps(T) * max(one(T), abs(clamp(lo, -limit, limit)), abs(clamp(hi, -limit, limit)))
  clamp(lo - pad, lower, upper), clamp(hi + pad, lower, upper), lower, upper
end

function _exp_root(a, b, logt, signt, lo, hi, lower, upper)
  T = typeof(a)
  f, _ = _exp_h(a, b, logt, signt, lo)
  f > 0 && (lo = lower)
  f, _ = _exp_h(a, b, logt, signt, hi)
  f < 0 && (hi = upper)
  rho = lo / 2 + hi / 2
  # Twenty damped Newton steps, then at most eighty bisections. Return
  # the last bracket midpoint if rounding prevents a smaller interval.
  for iter in 1:100
    f, df = _exp_h(a, b, logt, signt, rho)
    iszero(f) && return rho
    f < 0 ? (lo = rho) : (hi = rho)
    mid = lo / 2 + hi / 2
    (hi - lo <= 2eps(T) * max(one(T), abs(rho)) || mid == lo || mid == hi) && return mid
    if iter <= 20 && isfinite(df) && df > 0
      next = rho - f / df
      abs(next - rho) <= eps(T) * max(one(T), abs(rho)) && return clamp(next, lo, hi)
      rho = next <= lo ? T(0.05) * rho + T(0.95) * lo :
            next >= hi ? T(0.05) * rho + T(0.95) * hi : next
    else
      rho = mid
    end
  end
  lo / 2 + hi / 2
end

# Project onto each boundary ray at the root. These least-squares formulas
# equal Lemma 4.1 there, and avoid subtracting almost equal Moreau components.
# Reconstruct the largest component first, so a rounded subnormal is never
# multiplied back up. Exponential products retain the original coordinates.
function _exp_boundary(r::T, s::T, t::T, rho::T) where {T}
  if rho >= 0
    e = exp(-rho)
    d = one(T) + abs2(rho * e) + e * e
    c = max(zero(T), _exp_sum(_exp_product(r, -rho + log(rho)) / d,
                             _exp_product(s, -rho) / d, t / d))
    p = (_exp_product(c, -rho + log(rho)), _exp_product(c, -rho), c)
    if rho >= 2
      h = rho - one(T)
      d = one(T) + abs2(inv(h)) + abs2(e / h)
      c = max(zero(T), _exp_sum((r / h) / d, -s / d,
                               -(_exp_product(t, -rho) / h) / d))
      q = (c / h, -c, -_exp_product(c, -rho - log(h)))
    else
      d = one(T) + abs2(one(T) - rho) + e * e
      c = max(zero(T), _exp_sum(r / d, (one(T) - rho) * (s / d),
                               -_exp_product(t, -rho) / d))
      q = (c, (one(T) - rho) * c, -_exp_product(c, -rho))
    end
  else
    e = exp(rho)
    if rho < -1
      h = -rho
      d = one(T) + abs2(inv(h)) + abs2(e / h)
      c = max(zero(T), _exp_sum(-r / d, (s / h) / d, (_exp_product(t, rho) / h) / d))
      p = (-c, c / h, _exp_product(c, rho - log(h)))
    else
      d = one(T) + rho * rho + e * e
      c = max(zero(T), _exp_sum(rho * (r / d), s / d, _exp_product(t, rho) / d))
      p = (rho * c, c, _exp_product(c, rho))
    end
    d = one(T) + e * e + abs2((one(T) - rho) * e)
    c = max(zero(T), _exp_sum(_exp_product(r, rho) / d,
                             _exp_product(s, rho + log1p(-rho)) / d, -t / d))
    q = (_exp_product(c, rho), _exp_product(c, rho + log1p(-rho)), -c)
  end
  p, q
end

function _exp_project(r::T, s::T, t::T) where {T}
  z = (zero(T), zero(T), zero(T))
  if !(isfinite(r) && isfinite(s) && isfinite(t))
    n = (T(NaN), T(NaN), T(NaN))
    return n, n
  end
  _exp_contains(r, s, t, false, zero(T)) && return (r, s, t), z
  _exp_contains(r, s, t, true, zero(T)) && return z, (r, s, t)
  p, q = (min(r, zero(T)), zero(T), max(t, zero(T))), (zero(T), min(s, zero(T)), min(t, zero(T)))
  r <= 0 && s <= 0 && return p, q
  k = exponent(max(abs(r), abs(s), abs(t)))
  a, b, c = ldexp(r, -k), ldexp(s, -k), ldexp(t, -k)
  pdist, ddist = _exp_distance(p, a, b, c, k), _exp_distance(q, a, b, c, k)
  if s > 0
    candidate = (r, s, max(t, _exp_product(s, r / s)))
    if _exp_closer(candidate, p, a, b, c, k)
      p = candidate
      pdist = _exp_distance(p, a, b, c, k)
    end
  end
  if r > 0
    candidate = (r, s, min(t, -_exp_product(r, s / r - one(T))))
    if _exp_closer(candidate, q, a, b, c, k)
      q = candidate
      ddist = _exp_distance(q, a, b, c, k)
    end
  end
  # Outside this window, the missing components underflow even after
  # rescaling a largest finite coordinate. Keep the heuristic pair.
  limit = T(exponent(floatmax(T)) - exponent(nextfloat(zero(T))) + 32) * log(T(2))
  (r > 0 && one(T) - s / r > limit || s > 0 && r / s < -limit) && return p, q
  j = exponent(max(abs(r), abs(s)))
  u, v = ldexp(r, -j), ldexp(s, -j)
  logt = iszero(t) ? -T(Inf) : log(abs(t)) - T(j) * log(T(2))
  lo, hi, lower, upper = _exp_bracket(a, b, c, u, v, logt, pdist, ddist, limit)
  rho = _exp_root(u, v, logt, sign(t), lo, hi, lower, upper)
  pp, qq = _exp_boundary(r, s, t, rho)
  _exp_closer(pp, p, a, b, c, k) && (p = pp)
  _exp_closer(qq, q, a, b, c, k) && (q = qq)
  p, q
end

function _exp_project!(y, x::AbstractArray{T}, G, dual::Bool) where {T}
  @inbounds r, s, t = x[G[1]], x[G[2]], x[G[3]]
  # Conversion to T precedes negation, including for typemin(Int).
  p, q = dual ? _exp_project(-r, -s, -t) : _exp_project(r, s, t)
  result = dual ? (-q[1], -q[2], -q[3]) : p
  @inbounds y[G[1]], y[G[2]], y[G[3]] = result
  y
end

function prox!(y::AbstractArray, f::_ExpCone, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  T = float(promote_type(eltype(y), eltype(x)))
  z = _ScaledArray(x, one(T))
  dual = f isa IndExpDual
  f.groups === nothing && return _exp_project!(y, z, 1:3, dual)
  y === x || copyto!(y, x)
  for G in f.groups
    _exp_project!(y, z, G, dual)
  end
  y
end
