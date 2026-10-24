####################################
#
#  More nonconvex penalties: capped ℓ1, log, ℓq, group MCP and SCAD, and the
#  group ℓ0 ball
#
####################################

# Each proximal operator is the global minimizer of its nonconvex problem: on
# every piece where the scalar objective is convex the minimizer is a clamp or
# a root, and the candidates are compared.

####################################
#  capped ℓ1
####################################

"""
    CappedL1(λ, a)
    CappedL1(λ, a, w)

The capped ℓ1 penalty of Zhang (2010), nonconvex: `fᵢ(t) = λ wᵢ min(|t|, a)` for
`a > 0`, with all `wᵢ = 1` when `w` is omitted. It is the ℓ1 penalty up to `a`
and constant beyond, so that entries larger than `a` are not shrunk. Its
proximal operator is the global minimizer, the better of soft thresholding
clipped at `a` and the identity clipped below at `a`. At zero its
subdifferential is that of the ℓ1 norm, so [`kkt_violation`](@ref) is
`|gᵢ| - λwᵢ`.
"""
struct CappedL1{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ::T
  a::T
  w::W

  function CappedL1{T, W}(λ::T, a::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    _checknonneg("λ", λ)
    a > 0 || throw(ArgumentError("a must be positive, got $a"))
    _checknonneg("the weights", w)
    new{T, W}(λ, a, w)
  end
end
function CappedL1(λ::Real, a::Real, w::Union{Nothing, AbstractArray} = nothing)
  λ, a = promote(λ, a)
  CappedL1{typeof(λ), typeof(w)}(λ, a, w)
end

iszeropreserving(::CappedL1) = true
_checkparams(f::CappedL1, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::CappedL1, v::T, i::Integer) where {T<:Real}
  S = float(T)
  S(f.λ) * S(_entry(f.w, i)) * min(abs(S(v)), S(f.a))
end

Base.@propagate_inbounds function prox_entry(f::CappedL1, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  c, a, t = S(γ) * S(f.λ) * S(_entry(f.w, i)), S(f.a), abs(S(v))
  z1 = min(max(t - c, zero(S)), a)          # the ℓ1 piece |z| ≤ a: soft thresholding, clipped at a
  z2 = max(t, a)                             # the constant piece |z| ≥ a
  # γ times the objective at each; a tie goes to the smaller entry
  h1 = c * z1 + (z1 - t)^2 / 2
  h2 = c * a + (z2 - t)^2 / 2
  sign(S(v)) * ifelse(h1 <= h2, z1, z2)
end

kkt_violation(f::CappedL1, g::AbstractArray, i::Integer) = abs(g[i]) - f.λ * _entry(f.w, i)
restrict(f::CappedL1, S::AbstractVector{<:Integer}) = CappedL1(f.λ, f.a, _restrict_entries(f.w, S))

####################################
#  log penalty
####################################

"""
    LogPenalty(λ, ε)
    LogPenalty(λ, ε, w)

The log penalty `fᵢ(t) = λ wᵢ log(1 + |t|/ε)` for `ε > 0`, nonconvex, with all
`wᵢ = 1` when `w` is omitted: the penalty whose linearization gives the
reweighted ℓ1 minimization of Candès, Wakin and Boyd (2008). It grows only
logarithmically, so it shrinks large entries little, and it approaches the ℓ0
penalty as `ε → 0`. Its proximal operator is the global minimizer: zero or the
larger root of a quadratic, whichever is better. Its slope at zero is `λwᵢ/ε`,
so [`kkt_violation`](@ref) is `|gᵢ| - λwᵢ/ε`.
"""
struct LogPenalty{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ::T
  ε::T
  w::W

  function LogPenalty{T, W}(λ::T, ε::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    _checknonneg("λ", λ)
    ε > 0 || throw(ArgumentError("ε must be positive, got $ε"))
    _checknonneg("the weights", w)
    new{T, W}(λ, ε, w)
  end
end
function LogPenalty(λ::Real, ε::Real, w::Union{Nothing, AbstractArray} = nothing)
  λ, ε = promote(λ, ε)
  LogPenalty{typeof(λ), typeof(w)}(λ, ε, w)
end

iszeropreserving(::LogPenalty) = true
_checkparams(f::LogPenalty, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::LogPenalty, v::T, i::Integer) where {T<:Real}
  S = float(T)
  S(f.λ) * S(_entry(f.w, i)) * log1p(abs(S(v)) / S(f.ε))
end

Base.@propagate_inbounds function prox_entry(f::LogPenalty, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  c, ε, t = S(γ) * S(f.λ) * S(_entry(f.w, i)), S(f.ε), abs(S(v))
  # the stationary points of c log(1 + z/ε) + (z - t)²/2 on z > 0 are the roots
  # of z² + (ε - t)z + c - εt; the larger is a local minimum. Without real roots
  # the objective increases on [0, ∞)
  d = (t + ε)^2 - 4c
  d > 0 || return zero(S)
  r = sqrt(d)
  # the larger root, without cancellation: from the product c - εt of the roots
  # when the smaller one is computed accurately
  z = t >= ε ? (t - ε + r) / 2 : 2 * (c - ε * t) / (t - ε - r)
  z > 0 || return zero(S)
  c * log1p(z / ε) + (z - t)^2 / 2 < t^2 / 2 ? sign(S(v)) * z : zero(S)
end

kkt_violation(f::LogPenalty, g::AbstractArray, i::Integer) = abs(g[i]) - f.λ * _entry(f.w, i) / f.ε
restrict(f::LogPenalty, S::AbstractVector{<:Integer}) = LogPenalty(f.λ, f.ε, _restrict_entries(f.w, S))

####################################
#  ℓq penalty, 0 < q < 1
####################################

"""
    NormLq(λ, q)
    NormLq(λ, q, w)

The ℓq penalty `fᵢ(t) = λ wᵢ |t|^q` for `0 < q < 1`, nonconvex, with all `wᵢ = 1`
when `w` is omitted. It lies between the ℓ0 penalty ([`NormL0`](@ref), the limit
`q → 0`) and the ℓ1 norm (`q = 1`), and shrinks large entries less than the ℓ1
norm. Its proximal operator is the global minimizer. With `μ = γλwᵢ` it is zero
for `|v|` up to the threshold

    τ = (2 - q)/(2 - 2q) ⋅ (2μ(1 - q))^(1/(2 - q)),

where zero and the nonzero candidate tie, and above it the root `z > 0` of
`z + μq z^(q-1) = |v|` that exceeds `(2μ(1 - q))^(1/(2 - q))`, with the sign of
`v`. The root has a closed form for `q = 1/2` (half thresholding: Xu, Chang, Xu
and Zhang, 2012) and `q = 2/3` (Cao, Sun and Xu, 2013), and comes from Newton's
method otherwise.

Its slope at zero is infinite, so zero is a local minimizer of every problem
`h + f`, and [`kkt_violation`](@ref) is `-Inf`: an active-set method that grows
the support from zero by that test never adds a coordinate. Start such a method
from a nonzero point, or use the proximal operator of the whole vector, as
proximal gradient methods do.
"""
struct NormLq{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ::T
  q::T
  w::W

  function NormLq{T, W}(λ::T, q::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    _checknonneg("λ", λ)
    0 < q < 1 || throw(ArgumentError("q must be in (0, 1), got $q; use NormL0 for q = 0 and NormL1 for q = 1"))
    _checknonneg("the weights", w)
    new{T, W}(λ, q, w)
  end
end
function NormLq(λ::Real, q::Real, w::Union{Nothing, AbstractArray} = nothing)
  λ, q = promote(float(λ), q)
  NormLq{typeof(λ), typeof(w)}(λ, q, w)
end

iszeropreserving(::NormLq) = true
_checkparams(f::NormLq, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::NormLq, v::T, i::Integer) where {T<:Real}
  S = float(T)
  S(f.λ) * S(_entry(f.w, i)) * abs(S(v))^S(f.q)
end

Base.@propagate_inbounds function prox_entry(f::NormLq, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  μ, q, t = S(γ) * S(f.λ) * S(_entry(f.w, i)), S(f.q), abs(S(v))
  iszero(μ) && return S(v)
  t <= (2 - q) / (2 - 2q) * (2μ * (1 - q))^(1 / (2 - q)) && return zero(S)
  sign(S(v)) * _lq_root(t, μ, q)
end

# For t above the threshold, the root z > (2μ(1 - q))^(1/(2-q)) of
# g(z) = z + μq z^(q-1) - t, the minimizer of μ z^q + (z - t)²/2. g is convex
# and increasing there, so Newton's method from z = t, right of the root,
# decreases to it monotonically.
function _lq_root(t::S, μ::S, q::S) where {S<:AbstractFloat}
  if q == S(1) / 2
    # the cubic in √z by the trigonometric method (Xu et al., with λ = 2μ)
    φ = acos(μ / 4 * (t / 3)^(-S(3) / 2))
    return 2t / 3 * (1 + cos(2 * S(π) / 3 - 2φ / 3))
  elseif q == S(2) / 3
    # the quartic in z^(1/3) by its resolvent cubic (Cao et al., with λ = 2μ)
    λ2 = 2μ
    φ = acosh(27 * t^2 / (16 * λ2^(S(3) / 2)))
    A = 2 / sqrt(S(3)) * λ2^(S(1) / 4) * sqrt(cosh(φ / 3))
    return ((A + sqrt(2t / A - A^2)) / 2)^3
  end
  z = t
  for _ in 1:100
    Δ = (z + μ * q * z^(q - 1) - t) / (1 + μ * q * (q - 1) * z^(q - 2))
    z -= Δ
    abs(Δ) <= 4eps(z) && break
  end
  z
end

kkt_violation(f::NormLq, g::AbstractArray, ::Integer) = float(eltype(g))(-Inf)
restrict(f::NormLq, S::AbstractVector{<:Integer}) = NormLq(f.λ, f.q, _restrict_entries(f.w, S))

####################################
#  group MCP and group SCAD
####################################

"""
    GroupMCP(λ, groups; a = 3)
    GroupMCP(λ, groups, w; a = 3)

The group minimax concave penalty of Huang, Breheny and Ma (2012), nonconvex:
`f(x) = Σₖ pₖ(‖x[Gₖ]‖₂)`, where `pₖ` is the [`MCP`](@ref) with level `λwₖ`, over
disjoint groups `Gₖ = groups[k]` of (linear) indices. Groups are selected as a
whole, as by [`GroupNormL2`](@ref), but a group of large norm is not shrunk.
The weights default to `wₖ = √|Gₖ|`, and indices in no group are not penalized.

Its proximal operator is radial: the scalar proximal operator of `pₖ` at
`‖v[Gₖ]‖` rescales `v[Gₖ]`, which gives the global minimizer for every step.
At zero its subdifferential is that of the group lasso, so
[`kkt_violation`](@ref) is `‖g[Gₖ]‖₂ - λwₖ`. On an [`AtomIterate`](@ref) with
its groups, [`prox_coordinate!`](@ref) updates a group, the block step of
coordinate descent.
"""
struct GroupMCP{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractGroupNorm
  λ0::T                   # the level λ, under the field name of the other group penalties
  a::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}
  free::Vector{Int}

  function GroupMCP{T, G}(λ0::T, a::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    a > 0 || throw(ArgumentError("a must be positive, got $a"))
    group_of = _group_penalty_checks(λ0, groups, w)
    new{T, G}(λ0, a, groups, w, group_of, findall(iszero, group_of))
  end
end

"""
    GroupSCAD(λ, groups; a = 3.7)
    GroupSCAD(λ, groups, w; a = 3.7)

The group SCAD penalty, nonconvex: `f(x) = Σₖ pₖ(‖x[Gₖ]‖₂)`, where `pₖ` is the
[`SCAD`](@ref) penalty with level `λwₖ`, over disjoint groups, as
[`GroupMCP`](@ref) is for the MCP; `a > 2`. Its proximal operator is radial and
exact for every step.
"""
struct GroupSCAD{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractGroupNorm
  λ0::T
  a::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}
  free::Vector{Int}

  function GroupSCAD{T, G}(λ0::T, a::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    a > 2 || throw(ArgumentError("a must exceed 2, got $a"))
    group_of = _group_penalty_checks(λ0, groups, w)
    new{T, G}(λ0, a, groups, w, group_of, findall(iszero, group_of))
  end
end

function _group_penalty_checks(λ0, groups, w)
  length(w) == length(groups) || throw(DimensionMismatch("There must be one weight per group."))
  _checknonneg("λ", λ0)
  _checknonneg("the weights", w)
  _group_of(groups)
end

for (F, a0) in ((:GroupMCP, 3), (:GroupSCAD, 3.7))
  @eval function $F(λ::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector{<:Real}; a::Real = $a0)
    T = float(promote_type(typeof(λ), eltype(w)))     # a takes the type of λ, as its default must not promote
    G = _own_groups(groups)
    $F{T, typeof(G)}(convert(T, λ), convert(T, a), G, convert(Vector{T}, w))
  end
  @eval $F(λ::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}; a::Real = $a0) =
    $F(λ, groups, [sqrt(float(typeof(λ))(length(G))) for G in groups]; a)
end

const _RadialGroupPenalty = Union{GroupMCP, GroupSCAD}

isconvex(::Type{<:GroupMCP}) = false
isconvex(::Type{<:GroupSCAD}) = false

_rebuild(f::GroupMCP, λ0, groups, w) = GroupMCP(λ0, groups, w; a = f.a)
_rebuild(f::GroupSCAD, λ0, groups, w) = GroupSCAD(λ0, groups, w; a = f.a)

# the scalar penalty of group k, in the element type S
_group_scalar(f::GroupMCP, ::Type{S}, k::Integer) where {S} = MCP{S}(convert(S, f.λ0 * f.w[k]), convert(S, f.a))
_group_scalar(f::GroupSCAD, ::Type{S}, k::Integer) where {S} = SCAD{S}(convert(S, f.λ0 * f.w[k]), convert(S, f.a))

_group_norm(::_RadialGroupPenalty, x, G) = sqrt(_sumsq(x, G))
_dual_norm(::_RadialGroupPenalty, g, G) = sqrt(_sumsq(g, G))

# y[G] ← the proximal operator of γp(‖⋅‖₂) at x[G], which may be in place: the
# scalar proximal operator of p at the norm, along x[G]; returns the new norm
@inline function _radial_shrink!(p, y::AbstractArray{T}, x::AbstractArray, G, γ) where {T}
  nx = sqrt(_sumsq(x, G))
  r = iszero(nx) ? nx : prox_entry(p, nx, 1, γ)
  t = iszero(nx) ? nx : r / nx
  @inbounds for i in G
    y[i] = t * x[i]
  end
  r
end

function (f::_RadialGroupPenalty)(x::AbstractArray)
  _checkgroups(f, x)
  T = float(eltype(x))
  v = zero(T)
  for k in eachindex(f.groups)
    v += value_entry(_group_scalar(f, T, k), _group_norm(f, x, f.groups[k]), 1)
  end
  v
end

# with the groups of f, only the active groups count
function (f::_RadialGroupPenalty)(x::AtomIterate)
  x.groups === f.groups || return invoke(f, Tuple{AbstractArray}, x)
  T = float(eltype(x))
  v = zero(T)
  for a in 1:x.nactive
    k = x.active[a]
    v += value_entry(_group_scalar(f, T, k), _group_norm(f, x.storage, f.groups[k]), 1)
  end
  v
end

function value_and_prox!(y::AbstractArray, f::_RadialGroupPenalty, x::AbstractArray, γ::Real)
  _checksize(y, x)
  _checkgroups(f, x)
  T = float(eltype(y))
  if y !== x                    # indices in no group are not penalized
    @inbounds for i in f.free
      y[i] = x[i]
    end
    @inbounds for i in length(f.group_of)+1:length(x)
      y[i] = x[i]
    end
  end
  v = zero(T)
  for k in eachindex(f.groups)
    p = _group_scalar(f, T, k)
    v += value_entry(p, _radial_shrink!(p, y, x, f.groups[k], convert(T, γ)), 1)
  end
  v
end

function value_and_prox!(y::AtomIterate, f::_RadialGroupPenalty, x::AtomIterate, γ::Real)
  y.groups === x.groups === f.groups ||
    return invoke(value_and_prox!, Tuple{AbstractArray, _RadialGroupPenalty, AbstractArray, Real}, y, f, x, γ)
  T = float(eltype(y))
  copyto!(y, x)
  v = zero(T)
  for a in 1:y.nactive
    k = y.active[a]
    p = _group_scalar(f, T, k)
    v += value_entry(p, _radial_shrink!(p, y.storage, y.storage, f.groups[k], convert(T, γ)), 1)
  end
  v
end

# group k of x through the proximal operator in place; the group becomes active
# if it is nonzero, as after writing to it through atom(x, k)
function prox_coordinate!(x::AtomIterate, f::_RadialGroupPenalty, k::Integer, γ::Real)
  _checksamegroups(f, x)
  T = float(eltype(x))
  r = _radial_shrink!(_group_scalar(f, T, k), x.storage, x.storage, x.groups[k], convert(T, γ))
  iszero(r) || activate!(x, k)
  atom(x, k)
end

####################################
#  group ℓ0 ball
####################################

"""
    IndBallGroupL0(s, groups)

The indicator of the vectors with at most `s` nonzero groups, over disjoint
groups `Gₖ = groups[k]` of (linear) indices, nonconvex; indices in no group are
free. Its proximal operator keeps the `s` groups of largest Euclidean norm
(exactly `s` when there are ties, the earlier groups first) and zeroes the
others, by a selection in O(n) expected time, without allocating. Projected
gradient descent with it is iterative hard thresholding of groups, as in the
group IHT of MendelIHT (Chu et al., 2020). On an [`AtomIterate`](@ref) with its
groups, only the active groups are visited, and the dropped ones are
deactivated.
"""
struct IndBallGroupL0{G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractFunction
  s::Int
  groups::G
  group_of::Vector{Int}
  scratch::Scratch

  function IndBallGroupL0{G}(s::Integer, groups::G) where {G<:AbstractVector{<:AbstractVector{<:Integer}}}
    s >= 0 || throw(ArgumentError("s must be nonnegative, got $s"))
    new{G}(Int(s), groups, _group_of(groups), Scratch())
  end
end
IndBallGroupL0(s::Integer, groups::AbstractVector{<:AbstractVector{<:Integer}}) =
  (G = _own_groups(groups); IndBallGroupL0{typeof(G)}(s, G))

isproximable(::Type{<:IndBallGroupL0}) = true
_isindicator(::Type{<:IndBallGroupL0}) = true

_checkgroups(f::IndBallGroupL0, x::AbstractArray) =
  length(x) >= length(f.group_of) || throw(BoundsError(x, length(f.group_of)))

# the groups of an iterate are those of f, compared in O(number of groups)
_samegroups(f::IndBallGroupL0, x::AtomIterate) = x.groups === f.groups || x.groups == f.groups

function (f::IndBallGroupL0)(x::AbstractArray)
  _checkgroups(f, x)
  k = 0
  for G in f.groups
    k += !_allzero(x, G)
  end
  _ind(k <= f.s, float(eltype(x)))
end

function (f::IndBallGroupL0)(x::AtomIterate)
  _samegroups(f, x) || return invoke(f, Tuple{AbstractArray}, x)
  k = 0
  for a in 1:x.nactive
    k += !_allzero(x.storage, f.groups[x.active[a]])
  end
  _ind(k <= f.s, float(eltype(x)))
end

function prox!(y::AbstractArray, f::IndBallGroupL0, x::AbstractArray, ::Real)
  _checksize(y, x)
  _checkgroups(f, x)
  y === x || copyto!(y, x)
  _keep_largest_groups!(y, f, eachindex(f.groups))
end

function prox!(y::AtomIterate, f::IndBallGroupL0, x::AtomIterate, γ::Real)
  _samegroups(f, y) && _samegroups(f, x) ||
    return invoke(prox!, Tuple{AbstractArray, IndBallGroupL0, AbstractArray, Real}, y, f, x, γ)
  copyto!(y, x)
  _keep_largest_groups!(y.storage, f, view(y.active, 1:y.nactive))
  dropzeros!(y)
end

# zero all but the s groups of largest norm among the groups ks of y
function _keep_largest_groups!(y::AbstractArray, f::IndBallGroupL0, ks)
  T = float(eltype(y))
  n = length(ks)
  buf = _buffer!(f.scratch, T, 2n)
  nz = 0
  @inbounds for (j, k) in enumerate(ks)
    s2 = _sumsq(y, f.groups[k])
    buf[j] = s2
    nz += !iszero(s2) || !_allzero(y, f.groups[k])      # a sum of squares may underflow
  end
  nz <= f.s && return y
  # the s-th largest squared norm, from a copy that the selection reorders
  @inbounds for j in 1:n
    buf[n + j] = buf[j]
  end
  τ = f.s == 0 ? T(Inf) : _kth_largest!(view(buf, n+1:2n), f.s)
  above = 0
  @inbounds for j in 1:n
    above += buf[j] > τ
  end
  ties = f.s - above            # the groups at τ to keep, the earliest first
  @inbounds for (j, k) in enumerate(ks)
    s2 = buf[j]
    if s2 > τ
      continue
    elseif s2 == τ && ties > 0
      ties -= 1
    else
      for i in f.groups[k]
        y[i] = zero(eltype(y))
      end
    end
  end
  y
end
