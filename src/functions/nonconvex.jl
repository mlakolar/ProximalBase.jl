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
    w === nothing || eltype(w) <: Real || _real_error(:CappedL1)
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
  λ, w, m = S(f.λ), S(_entry(f.w, i)), min(abs(S(v)), S(f.a))
  _weighted_value(_weight_coefficient(λ, w) * m, m)
end
Base.@propagate_inbounds function _value_entry_ordinary(f::CappedL1, v::T, i::Integer) where {T<:Real}
  S = float(T)
  S(f.λ) * S(_entry(f.w, i)) * min(abs(S(v)), S(f.a))
end
_has_ordinary_value(::Type{<:CappedL1}) = true

Base.@propagate_inbounds function prox_entry(f::CappedL1, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  c = _weight_coefficient(S, γ, f.λ, _entry(f.w, i))
  a, t = S(f.a), abs(S(v))
  z1 = min(max(t - c, zero(S)), a)          # the ℓ1 piece |z| ≤ a: soft thresholding, clipped at a
  z2 = max(t, a)                             # the constant piece |z| ≥ a
  # γ times the objective at each; a tie goes to the smaller entry, and an
  # infinite coefficient (z1 = 0, where h1 is Inf⋅0) to z1
  h1 = c * z1 + (z1 - t)^2 / 2
  h2 = c * a + (z2 - t)^2 / 2
  sign(S(v)) * ifelse(isinf(c) | (h1 <= h2), z1, z2)
end

kkt_violation(f::CappedL1, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]) - _weight_coefficient(f.λ, _entry(f.w, i)))
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
    w === nothing || eltype(w) <: Real || _real_error(:LogPenalty)
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
  w0 = _entry(f.w, i)
  λ, w, φ = S(f.λ), S(w0), log1p(abs(S(v)) / S(f.ε))
  # The ratio can round to zero away from the fixed point.
  isinf(w0) && return isnan(v) ? S(v) : iszero(v) ? zero(S) : S(Inf)
  # A zero level or weight, or an infinite ε (the zero function on finite arguments), leaves the
  # entry free, NaN included; the parameters decide, not their conversions.
  (iszero(f.λ) | iszero(w0) | isinf(f.ε)) && return zero(S)
  _weighted_value(_weight_coefficient(λ, w) * φ, φ)
end
Base.@propagate_inbounds function _value_entry_ordinary(f::LogPenalty, v::T, i::Integer) where {T<:Real}
  S = float(T)
  S(f.λ) * S(_entry(f.w, i)) * log1p(abs(S(v)) / S(f.ε))
end
_has_ordinary_value(::Type{<:LogPenalty}) = true

# The result at a NaN z, decided by the parameters and not by the rounded coefficient c: an infinite
# ε makes f the zero function on finite arguments (the identity); a NaN coefficient is the zero
# level at γ = Inf (0⋅Inf), which is free; an infinite level has the domain {0}, so the map is
# zero; a NaN entry stays NaN, unless the step is Inf, where the map is constant. (A finite γλw
# that overflows to Inf is no endpoint.)
@noinline function _log_penalty_nan(v::S, c::S, t::S, f, γ) where {S}
  isinf(f.ε) && return v
  isnan(c) && return iszero(f.λ) ? v : zero(S)
  isinf(f.λ) && return zero(S)               # an infinite level: the domain is {0}, the map is 0
  isnan(t) && !(γ == Inf) ? v : zero(S)
end

Base.@propagate_inbounds function prox_entry(f::LogPenalty, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  w = _entry(f.w, i)
  c = _weight_coefficient(S, γ, f.λ, w)
  ε, t = S(f.ε), abs(S(v))
  # a zero level or weight leaves the entry free, an infinite weight fixes it at
  # zero: both come before any NaN
  iszero(c) && return S(v)
  isinf(w) && return zero(S)
  # the stationary points of c log(1 + z/ε) + (z - t)²/2 on z > 0 are the roots
  # of z² + (ε - t)z + c - εt; the larger is a local minimum. Without real roots
  # the objective increases on [0, ∞)
  d = (t + ε)^2 - 4c
  d <= 0 && return zero(S)
  r = sqrt(d)
  # the larger root, without cancellation: from the product c - εt of the roots
  # when the smaller one is computed accurately
  z = t >= ε ? (t - ε + r) / 2 : 2 * (c - ε * t) / (t - ε - r)
  # A NaN d (a NaN entry, a NaN coefficient: the zero level at γ = Inf, 0⋅Inf, or an infinite ε)
  # is not <= 0 and reaches here as a NaN z, which the helper classifies.
  z > 0 || return isnan(z) ? _log_penalty_nan(S(v), c, t, f, γ) : zero(S)
  c * log1p(z / ε) + (z - t)^2 / 2 < t^2 / 2 ? sign(S(v)) * z : zero(S)
end

kkt_violation(f::LogPenalty, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]) - _weight_coefficient(f.λ, _entry(f.w, i)) / f.ε)
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
`v`. The root uses a closed form for `q = 1/2` (half thresholding: Xu, Chang, Xu
and Zhang, 2012), and a dimensionless Newton iteration otherwise, avoiding
overflow and underflow in the quartic closed form at `q = 2/3`.

At a penalized entry its slope at zero is infinite, so zero is a local
minimizer of `h + f`, and [`kkt_violation`](@ref) is `-Inf`: an active-set method
that grows the support from zero by that test never adds a penalized coordinate.
A zero weight, or a zero level with finite weight, gives `abs(g[i])` instead.
Start such a method from a nonzero point, or use the proximal operator of the
whole vector, as proximal gradient methods do.
"""
struct NormLq{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ::T
  q::T
  w::W

  function NormLq{T, W}(λ::T, q::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    w === nothing || eltype(w) <: Real || _real_error(:NormLq)
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
  λ, w, φ = S(f.λ), S(_entry(f.w, i)), abs(S(v))^S(f.q)
  _weighted_value(_weight_coefficient(λ, w) * φ, φ)
end
Base.@propagate_inbounds function _value_entry_ordinary(f::NormLq, v::T, i::Integer) where {T<:Real}
  S = float(T)
  S(f.λ) * S(_entry(f.w, i)) * abs(S(v))^S(f.q)
end
_has_ordinary_value(::Type{<:NormLq}) = true

Base.@propagate_inbounds function prox_entry(f::NormLq, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  w0 = _entry(f.w, i)
  q, t = S(f.q), abs(S(v))
  isinf(w0) && return isnan(v) ? S(v) : zero(S)
  (iszero(γ) || iszero(f.λ) || iszero(w0)) && return S(v)
  # the factors in the promotion of S and their types, rounded once (T3)
  R = _working_type(S, γ, f.λ, w0)
  step, λ, w = convert(R, γ), convert(R, f.λ), convert(R, w0)
  μ = _positive_product(S, γ, f.λ, _entry(f.w, i))
  ordinary = isfinite(μ) && μ >= floatmin(S)
  # When μ cannot retain its precision, take the bounded powers of its factors
  # before multiplying. The threshold can fit even when μ does not.
  τ = if q == S(1)/2
    S(3)/2 * (ordinary ? cbrt(μ)^2 : convert(S, _positive_product(cbrt(step)^2, cbrt(λ)^2, cbrt(w)^2)))
  elseif ordinary
    power = inv(2 - q)
    # Separate powers retain subnormal μ without rounding μ*2(1-q)
    # first. The coefficient is bounded as q tends to either endpoint.
    coefficient = ((2 - q) / (2 - 2q)) * (2 * (1 - q))^power
    coefficient * μ^power
  else
    power = inv(2 - q)
    coefficient = ((2 - q) / (2 - 2q)) * (2 * (1 - q))^power
    coefficient * convert(S, _positive_product(step^power, λ^power, w^power))
  end
  t <= τ && return zero(S)
  sign(S(v)) * _lq_root(t, μ, q, τ)
end

# For t above the threshold, the root z > (2μ(1 - q))^(1/(2-q)) of
# g(z) = z + μq z^(q-1) - t, the minimizer of μ z^q + (z - t)²/2. g is convex
# and increasing there, so Newton's method from z = t, right of the root,
# decreases to it monotonically.
function _lq_root(t::S, μ::S, q::S, τ::S) where {S<:AbstractFloat}
  # μ/t^(2-q) is bounded above the threshold. Recover it from τ = c μ^(1/(2-q))
  # when μ is outside its normal range, without logarithmic cancellation.
  relative_level = if isfinite(μ) && μ >= floatmin(S)
    q == S(1) / 2 ? (μ / t) / sqrt(t) : (μ / t) / t^(1 - q)
  elseif q == S(1) / 2
    ratio = (τ / t) / (S(3)/2)
    ratio * sqrt(ratio)
  else
    coefficient = ((2 - q) / (2 - 2q)) * (2 * (1 - q))^inv(2 - q)
    ((τ / t) / coefficient)^(2 - q)
  end
  if q == S(1) / 2
    # the cubic in √z by the trigonometric method (Xu et al., with λ = 2μ)
    # μ/t^(3/2) is bounded above the threshold. Divide in this order so
    # neither t^(3/2) nor its reciprocal can overflow before multiplying μ.
    φ = acos(relative_level * (3sqrt(S(3))/4))
    return t * (S(2)/3 * (1 + cos(2 * S(π) / 3 - 2φ / 3)))
  end
  # The dimensionless equation r + a*r^(q-1) = 1 avoids t², μ^(3/2)
  # and negative powers of a tiny z. Above the threshold its larger root
  # is bounded away from zero for a fixed q, and Newton from r=1 decreases.
  a = relative_level * q
  r = one(S)
  # The nonzero branch starts at this ratio at the threshold. Rounding near
  # q=1 must not send a Newton step below it and into a negative power base.
  lower = 2 * (1 - q) / (2 - q)
  for _ in 1:100
    power = q == S(2) / 3 ? inv(cbrt(r)) : r^(q - 1)
    Δ = (r + a * power - 1) / (1 + a * (q - 1) * power / r)
    next = max(lower, r - Δ)
    next == r && break
    r = next
    abs(Δ) <= 4eps(r) && break
  end
  t * r
end

function kkt_violation(f::NormLq, g::AbstractArray, i::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
  w = _entry(f.w, i)
  !isinf(w) && (iszero(f.λ) || iszero(w)) ? abs(g[i]) : float(eltype(g))(-Inf)
end
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
struct GroupMCP{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractGroupPenalty
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
struct GroupSCAD{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractGroupPenalty
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
  @eval function $F(λ::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector; a::Real = $a0)
    eltype(w) <: Real || _real_error(nameof($F))
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
_group_scalar(f::GroupMCP, ::Type{S}, k::Integer) where {S} = MCP{S}(_weight_coefficient(S, f.λ0, f.w[k]), convert(S, f.a))
_group_scalar(f::GroupSCAD, ::Type{S}, k::Integer) where {S} = SCAD{S}(_weight_coefficient(S, f.λ0, f.w[k]), convert(S, f.a))

# The ordinary kernels retain the scalar product and loops used for finite weights.
_ordinary_radial_groups(f, ::Type{T}) where {T} = isfinite(f.λ0) &&
  !mapreduce(w -> isinf(w) | isinf(_typed_product(T, f.λ0, w)), |, f.w; init = false)
# Matching atom layouts classify only coefficients the kernels will visit.
function _ordinary_radial_groups(f, ::Type{T}, groups, n::Int) where {T}
  isfinite(f.λ0) || return false
  ordinary = true
  @inbounds @simd for a in 1:n
    w = f.w[groups[a]]
    ordinary &= !isinf(w) & !isinf(_typed_product(T, f.λ0, w))
  end
  ordinary
end
_infinite_radial_coefficient(f, ::Type{T}, k) where {T} =
  isinf(_weight_coefficient(T, f.λ0, f.w[k]))
_group_scalar(f::GroupMCP, ::Type{S}, k::Integer, ::Val{true}) where {S} = MCP{S}(_typed_product(S, f.λ0, f.w[k]), convert(S, f.a))
_group_scalar(f::GroupSCAD, ::Type{S}, k::Integer, ::Val{true}) where {S} = SCAD{S}(_typed_product(S, f.λ0, f.w[k]), convert(S, f.a))
_group_scalar(f::_RadialGroupPenalty, ::Type{S}, k::Integer, ::Val{false}) where {S} = _group_scalar(f, S, k)

@inline function _check_group_coordinate(f::_RadialGroupPenalty, x::AtomIterate, k::Integer)
  _checksamegroups(f, x)
  checkbounds(x.groups, k)
  _infinite_radial_coefficient(f, float(eltype(x)), k) && return nothing
  _group_scalar(f, float(eltype(x)), k)
end

# Construct every scalar penalty that the kernel will use before it writes.
_check_group_scalars(f::_RadialGroupPenalty, ::Type{T}, groups, n::Int = length(groups)) where {T} =
  _ordinary_radial_groups(f, T) ? _check_group_scalars(f, T, groups, n, Val(true)) :
    _check_group_scalars(f, T, groups, n, Val(false))
function _check_group_scalars(f::_RadialGroupPenalty, ::Type{T}, groups, n::Int, ordinary::Val{O}) where {T, O}
  for a in 1:n
    k = groups[a]
    !O && _infinite_radial_coefficient(f, T, k) && continue
    _group_scalar(f, T, k, ordinary)
  end
  true
end
# The shape is shared by every scalar. Check it once, then reduce the same
# converted coefficient checks; an invalid coefficient takes the ordered errors.
function _check_group_scalars(f::_RadialGroupPenalty, ::Type{T}, groups, n::Int, ordinary::Val{true}) where {T}
  iszero(n) && return true
  _group_scalar(f, T, groups[1], ordinary)
  valid = true
  @inbounds @simd for a in 1:n
    valid &= _typed_product(T, f.λ0, f.w[groups[a]]) >= zero(T)
  end
  if !valid
    for a in 1:n
      _group_scalar(f, T, groups[a], ordinary)
    end
  end
  true
end
_check_prox_group_scalars(f::_RadialGroupPenalty, ::Type{T}, y, x) where {T} =
  _check_group_scalars(f, T, eachindex(f.groups))
function _check_prox_group_scalars(f::_RadialGroupPenalty, ::Type{T}, y::AtomIterate, x::AtomIterate) where {T}
  if _samelayout(f, x) && _samelayout(f, y)
    return _ordinary_radial_groups(f, T, x.active, x.nactive) ?
      _check_group_scalars(f, T, x.active, x.nactive, Val(true)) :
      _check_group_scalars(f, T, x.active, x.nactive, Val(false))
  end
  _check_group_scalars(f, T, eachindex(f.groups))
end
function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::_RadialGroupPenalty, y::AbstractArray, x::AbstractArray)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkgroups(f, x)
  _check_symmetric_groups(f, y)
  _check_prox_group_scalars(f, _model_type(y, x), y, x)
end

for Step in (Real, AbstractVector)
  @eval begin
    value_and_prox!(y::AbstractArray, f::_RadialGroupPenalty, x::AbstractArray, γ::$Step) = (x = _symmetric_argument(y, x); _checkreal(f, x); _checkreal(f, y); _checkreal(f, γ); _value_and_prox_groups!(y, f, x, γ))
    value_and_prox!(y::AtomIterate, f::_RadialGroupPenalty, x::AtomIterate, γ::$Step) = (_checkreal(f, x); _checkreal(f, y); _checkreal(f, γ); _value_and_prox_groups!(y, f, x, γ))
    prox!(y::AbstractArray, f::_RadialGroupPenalty, x::AbstractArray, γ::$Step) = (x = _symmetric_argument(y, x); _checkreal(f, x); _checkreal(f, y); _checkreal(f, γ); _value_and_prox_groups!(y, f, x, γ); y)
  end
end

_group_norm(::_RadialGroupPenalty, x, G) = _group_norm2(x, G)
_dual_norm(::_RadialGroupPenalty, g, G) = _group_norm2(g, G)

# y[G] ← the proximal operator of γp(‖⋅‖₂) at x[G], which may be in place: the
# scalar proximal operator of p at the norm, along x[G]; returns the new norm
@inline function _radial_shrink!(p, y::AbstractArray{T}, x::AbstractArray, G, γ, W = nothing) where {T}
  nx = convert(typeof(p.λ), _group_norm2(x, G))   # the scalar proximal operator works in the type of p
  r = iszero(nx) ? nx : prox_entry(p, nx, 1, γ)
  t = iszero(nx) ? nx : r / nx
  @inbounds for i in _writes(W, G)
    y[i] = t * x[i]
  end
  r
end

function (f::_RadialGroupPenalty)(x::AbstractArray)
  _checkreal(f, x)
  iszero(f.λ0) && return _zero_level_radial_value(f, x)
  _ordinary_radial_groups(f, float(eltype(x))) ?
    _radial_group_value(f, x, Val(true)) : _radial_group_value(f, x, Val(false))
end

# At zero level, classify weights in the scalar loop instead of a separate scan.
function _zero_level_radial_value(f::_RadialGroupPenalty, x::AbstractArray)
  _checkgroups(f, x)
  T = float(eltype(x))
  v = zero(T)
  for k in eachindex(f.groups)
    if isinf(f.w[k])
      _allzero(x, f.groups[k]) || return T(Inf)
      continue
    end
    p = _group_scalar(f, T, k, Val(true))
    iszero(p.λ) && continue
    v += value_entry(p, _group_norm(f, x, f.groups[k]), 1)
  end
  v
end

function _radial_group_value(f::_RadialGroupPenalty, x::AbstractArray, ordinary::Val{O}) where {O}
  _checkgroups(f, x)
  T = float(eltype(x))
  v = zero(T)
  for k in eachindex(f.groups)
    if !O && _infinite_radial_coefficient(f, T, k)
      _allzero(x, f.groups[k]) || return T(Inf)
      continue
    end
    p = _group_scalar(f, T, k, ordinary)
    iszero(p.λ) && continue
    v += value_entry(p, _group_norm(f, x, f.groups[k]), 1)
  end
  v
end

# with the groups of f, only the active groups count
function (f::_RadialGroupPenalty)(x::AtomIterate)
  _checkreal(f, x)
  _samelayout(f, x) || return invoke(f, Tuple{AbstractArray}, x)
  _ordinary_radial_groups(f, float(eltype(x)), x.active, x.nactive) ?
    _radial_group_value(f, x, Val(true)) : _radial_group_value(f, x, Val(false))
end

function _radial_group_value(f::_RadialGroupPenalty, x::AtomIterate, ordinary::Val{O}) where {O}
  _samelayout(f, x) || return invoke(_radial_group_value, Tuple{_RadialGroupPenalty, AbstractArray, Val{O}}, f, x, Val(O))
  T = float(eltype(x))
  v = zero(T)
  for a in 1:x.nactive
    k = x.active[a]
    if !O && _infinite_radial_coefficient(f, T, k)
      _allzero(x.storage, f.groups[k]) || return T(Inf)
      continue
    end
    p = _group_scalar(f, T, k, ordinary)
    iszero(p.λ) && continue
    v += value_entry(p, _group_norm(f, x.storage, f.groups[k]), 1)
  end
  v
end

function _value_and_prox_groups!(y::AbstractArray, f::_RadialGroupPenalty, x::AbstractArray, γ::Union{Real, AbstractVector})
  _ordinary_radial_groups(f, _model_type(y, x)) ?
    _radial_group_prox!(y, f, x, γ, Val(true)) : _radial_group_prox!(y, f, x, γ, Val(false))
end

function _radial_group_prox!(y::AbstractArray, f::_RadialGroupPenalty, x::AbstractArray, γ::Union{Real, AbstractVector}, ordinary::Val{O}) where {O}
  _checksize(y, x)
  _checkgroups(f, x)
  _check_symmetric_groups(f, y)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  W = _model_type(y, x)
  _check_group_scalars(f, W, eachindex(f.groups), length(f.groups), ordinary)
  _copy_ungrouped!(y, f, x)
  v = zero(T)
  for k in eachindex(f.groups)
    if !O && _infinite_radial_coefficient(f, W, k)
      n = _shrink_group!(NormL2(one(W)), y, x, f.groups[k], _weight_product(W, _entry(γ, k), f.λ0, f.w[k]), _group_writes(y, f.groups[k]))
      W === T || (n = _group_norm2(y, f.groups[k]))   # O2: the value is that of the stored output
      iszero(n) || (v = T(Inf))
      continue
    end
    p = _group_scalar(f, W, k, ordinary)
    if iszero(p.λ)
      @inbounds for i in f.groups[k]
        y[i] = x[i]
      end
      continue
    end
    r = _radial_shrink!(p, y, x, f.groups[k], convert(W, _entry(γ, k)), _group_writes(y, f.groups[k]))
    W === T || (r = convert(W, _group_norm2(y, f.groups[k])))
    v += convert(T, value_entry(p, r, 1))
  end
  v
end

function _value_and_prox_groups!(y::AtomIterate, f::_RadialGroupPenalty, x::AtomIterate, γ::Union{Real, AbstractVector})
  _samelayout(f, x) && _samelayout(f, y) ||
    return invoke(_value_and_prox_groups!, Tuple{AbstractArray, _RadialGroupPenalty, AbstractArray, Union{Real, AbstractVector}}, y, f, x, γ)
  _ordinary_radial_groups(f, float(eltype(y)), x.active, x.nactive) ?
    _radial_group_prox!(y, f, x, γ, Val(true)) : _radial_group_prox!(y, f, x, γ, Val(false))
end

function _radial_group_prox!(y::AtomIterate, f::_RadialGroupPenalty, x::AtomIterate, γ::Union{Real, AbstractVector}, ordinary::Val{O}) where {O}
  _samelayout(f, x) && _samelayout(f, y) ||
    return invoke(_radial_group_prox!, Tuple{AbstractArray, _RadialGroupPenalty, AbstractArray, Union{Real, AbstractVector}, Val{O}}, y, f, x, γ, Val(O))
  _checksize(y, x)
  _checkgroups(f, x)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  _check_group_scalars(f, T, x.active, x.nactive, ordinary)
  copyto!(y, x)
  v = zero(T)
  for a in 1:y.nactive
    k = y.active[a]
    if !O && _infinite_radial_coefficient(f, T, k)
      n = _shrink_group!(NormL2(one(T)), y.storage, y.storage, f.groups[k], _weight_product(T, _entry(γ, k), f.λ0, f.w[k]))
      iszero(n) || (v = T(Inf))
      continue
    end
    p = _group_scalar(f, T, k, ordinary)
    iszero(p.λ) && continue
    v += convert(T, value_entry(p, _radial_shrink!(p, y.storage, y.storage, f.groups[k], convert(T, _entry(γ, k))), 1))
  end
  v
end

@inline _check_coordinate(x::AtomIterate, f::_RadialGroupPenalty, k) = _check_group_coordinate(f, x, k)
@inline _block_step!(x::AtomIterate, f::_RadialGroupPenalty, k, u, γ, ctx) = _stage_and_step!(x, f, k, u, γ, ctx)

# group k of x through the proximal operator in place; the group becomes active
# if it ends nonzero
function prox_coordinate!(x::AtomIterate, f::_RadialGroupPenalty, k::Integer, γ::Real)
  ctx = _check_coordinate(x, f, k)
  _unchecked_coordinate!(x, f, k, γ, ctx)
end

@inline function _unchecked_coordinate!(x::AtomIterate, f::_RadialGroupPenalty, k, γ, p)
  T = float(eltype(x))
  if _infinite_radial_coefficient(f, T, k)
    n = _shrink_group!(NormL2(one(T)), x.storage, x.storage, x.groups[k], _weight_product(T, γ, f.λ0, f.w[k]))
    iszero(n) || activate!(x, k)
    return atom(x, k)
  end
  if iszero(p.λ)
    _allzero(x.storage, x.groups[k]) || activate!(x, k)
    return atom(x, k)
  end
  r = _radial_shrink!(p, x.storage, x.storage, x.groups[k], convert(T, γ))
  iszero(r) || activate!(x, k)
  atom(x, k)
end

####################################
#  group ℓ0 ball
####################################

"""
    IndBallGroupL0(s, groups; k = nothing)

The indicator of the vectors with at most `s` nonzero groups, over disjoint
groups `Gₖ = groups[k]` of (linear) indices, nonconvex; indices in no group are
free. Its proximal operator keeps the `s` groups of largest Euclidean norm
(exactly `s` when there are ties, the earlier groups first) and zeroes the
others, by a selection in O(n) expected time, without allocating. Projected
gradient descent with it is iterative hard thresholding of groups, as in the
group IHT of MendelIHT (Chu et al., 2020). On an [`AtomIterate`](@ref) with its
groups, only the active groups are visited, and the dropped ones are
deactivated.

With `k`, each group also has at most `k` nonzero entries: the doubly sparse set
of group IHT with sparsity within the groups. The projection ranks the groups by
the norm of their `k` largest entries, keeps the `s` best, and within each the
`k` largest entries (exactly `k` when there are ties, the earlier entries
first).
"""
struct IndBallGroupL0{G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractFunction
  s::Int
  groups::G
  k::Int              # the nonzero entries allowed in a group, typemax(Int) for any number
  group_of::Vector{Int}
  scratch::Scratch    # the norms of the groups, and a copy for the selection
  entries::Scratch    # the magnitudes of entries of a group, for the selection

  function IndBallGroupL0{G}(s::Integer, groups::G, k::Integer) where {G<:AbstractVector{<:AbstractVector{<:Integer}}}
    s >= 0 || throw(ArgumentError("s must be nonnegative, got $s"))
    k >= 0 || throw(ArgumentError("k must be nonnegative, got $k"))
    new{G}(Int(s), groups, Int(k), _group_of(groups), Scratch(), Scratch())
  end
end
IndBallGroupL0(s::Integer, groups::AbstractVector{<:AbstractVector{<:Integer}}; k::Union{Integer, Nothing} = nothing) =
  (G = _own_groups(groups); IndBallGroupL0{typeof(G)}(s, G, k === nothing ? typemax(Int) : k))

isproximable(::Type{<:IndBallGroupL0}) = true
_isindicator(::Type{<:IndBallGroupL0}) = true

_checkgroups(f::IndBallGroupL0, x::AbstractArray) =
  length(x) >= length(f.group_of) || throw(BoundsError(x, length(f.group_of)))

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndBallGroupL0, y::AbstractArray, x::AbstractArray) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); _checkgroups(f, x); true)

# the functions whose groups an AtomIterate can be compared with
const _GroupFunction = Union{AbstractGroupPenalty, IndBallGroupL0}
samegroups(x::AtomIterate, f::_GroupFunction) = _samelayout(f, x)
samegroups(f::_GroupFunction, x::AtomIterate) = _samelayout(f, x)

# whether the entries of x in G are all zero, and, if not, whether there are at
# most k of them
function _group_status(x::AbstractArray, G, k::Int)
  _allzero(x, G) && return (true, true)
  length(G) <= k && return (false, true)
  c = 0
  @inbounds for i in G
    c += !iszero(x[i])
  end
  return (false, c <= k)
end

function (f::IndBallGroupL0)(x::AbstractArray)
  _checkreal(f, x)
  _checkgroups(f, x)
  nz, ok = 0, true
  for G in f.groups
    z, fits = _group_status(x, G, f.k)
    nz += !z
    ok &= fits
  end
  _ind(ok && nz <= f.s, _valtype(x))
end

function (f::IndBallGroupL0)(x::AtomIterate)
  _checkreal(f, x)
  _samelayout(f, x) || return invoke(f, Tuple{AbstractArray}, x)
  nz, ok = 0, true
  for a in 1:x.nactive
    z, fits = _group_status(x.storage, f.groups[x.active[a]], f.k)
    nz += !z
    ok &= fits
  end
  _ind(ok && nz <= f.s, _valtype(x))
end

function prox!(y::AbstractArray, f::IndBallGroupL0, x::AbstractArray, ::Real)
  x = _symmetric_argument(y, x)
  _checkwrite(prox!, f, y, x)
  y === x || copyto!(y, x)
  _keep_largest_groups!(y, f, eachindex(f.groups))
end

function prox!(y::AtomIterate, f::IndBallGroupL0, x::AtomIterate, γ::Real)
  _checkwrite(prox!, f, y, x)
  _samelayout(f, y) && _samelayout(f, x) ||
    return invoke(prox!, Tuple{AbstractArray, IndBallGroupL0, AbstractArray, Real}, y, f, x, γ)
  copyto!(y, x)
  _keep_largest_groups!(y.storage, f, view(y.active, 1:y.nactive))
  dropzeros!(y)
end

# the magnitudes of entries of y in G, in w[1:length(G)]
function _group_magnitudes!(w::AbstractVector{T}, y::AbstractArray, G) where {T}
  @inbounds for (t, i) in enumerate(G)
    w[t] = abs(convert(T, y[i]))
  end
  view(w, 1:length(G))
end

# the norm of the k largest magnitudes of entries of y in G
function _top_norm(f::IndBallGroupL0, y::AbstractArray, G, ::Type{T}) where {T}
  length(G) <= f.k && return convert(T, _group_norm2(y, G))
  f.k == 0 && return zero(T)
  v = _group_magnitudes!(_buffer!(f.entries, T, length(G)), y, G)
  _kth_largest!(v, f.k)
  return norm(view(v, 1:f.k))
end

# zero all but the k entries of largest magnitude of y in G, the earlier first
# among ties
function _keep_largest_entries!(y::AbstractArray, f::IndBallGroupL0, G, ::Type{T}) where {T}
  length(G) <= f.k && return y
  v = _group_magnitudes!(_buffer!(f.entries, T, length(G)), y, G)
  τ = f.k == 0 ? T(Inf) : _kth_largest!(v, f.k)
  above = 0
  @inbounds for i in G
    above += abs(convert(T, y[i])) > τ
  end
  ties = f.k - above
  @inbounds for i in G
    a = abs(convert(T, y[i]))
    if a > τ
      continue
    elseif a == τ && ties > 0
      ties -= 1
    else
      y[i] = zero(eltype(y))
    end
  end
  return y
end

# zero all but the s groups of largest norm among the groups ks of y, and with
# a bound k on the entries, all but the k largest entries of each group kept;
# the groups are ranked by the norm of those entries
function _keep_largest_groups!(y::AbstractArray, f::IndBallGroupL0, ks)
  T = float(eltype(y))
  n = length(ks)
  buf = _buffer!(f.scratch, T, 2n)
  nz = 0
  @inbounds for (j, k) in enumerate(ks)
    nrm = _top_norm(f, y, f.groups[k], T)
    buf[j] = nrm
    nz += !iszero(nrm) || !_allzero(y, f.groups[k])
  end
  if nz <= f.s
    _keep_entries_of_groups!(y, f, ks, T)
    return y
  end
  # the s-th largest norm, from a copy that the selection reorders
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
    nrm = buf[j]
    G = f.groups[k]
    if nrm > τ
      _keep_largest_entries!(y, f, G, T)
    elseif nrm == τ && ties > 0
      ties -= 1
      _keep_largest_entries!(y, f, G, T)
    else
      for i in G
        y[i] = zero(eltype(y))
      end
    end
  end
  y
end

function _keep_entries_of_groups!(y::AbstractArray, f::IndBallGroupL0, ks, ::Type{T}) where {T}
  f.k == typemax(Int) && return y
  for k in ks
    _keep_largest_entries!(y, f, f.groups[k], T)
  end
  return y
end

####################################
#  values as a significand and an exponent
####################################

# The scaled-value mechanism (`_sep_value_parts`, separable.jl): the terms λ w φ in parts, with
# φ = min(|v|, a), log(1 + |v|/ε) (in parts where |v|/ε leaves the range) and |v|^q. An infinite
# weight at a nonzero entry is an infinite term, and the value is kept.
_has_value_parts(::Type{<:Union{CappedL1, LogPenalty, NormLq}}) = true
function _value_parts(f::Union{CappedL1, LogPenalty, NormLq}, x::AbstractArray, v)
  (iszero(f.λ) || !isfinite(f.λ)) && return nothing
  shape = f isa CappedL1 ? f.a : f isa LogPenalty ? f.ε : f.q
  _sep_value_parts(f, x, _sep_parts_type(x, f.λ, shape, f.w), f.λ)
end
_parameter_alias(f::Union{CappedL1, LogPenalty, NormLq}, out) = _mightalias_parameter(out, f.w)
# Exact-zero protection: a proof per operation where a term can round into the subnormal range of the
# value type T, each such rounding losing less than δ/2. The levels are converted to T, where they
# must keep their range (normal or zero). The ordinary term is (λ w) φ with the weight converted to T
# first, so weights of a wider type than T have no bound (the parts form λ w in the wider type).
# - CappedL1: λ w loses less than δ/2 where it underflows, and φ = min(|v|, a) < 2^e(a) multiplies
#   that loss; the product with φ loses less than δ/2. Per term less than (2^e(a) + 1) δ/2.
# - LogPenalty: q = |v|/ε loses less than δ/2 where it underflows, and φ = log1p(q) passes that on,
#   which λ w < 2^(e(λ) + e(w)) multiplies; λ w loses less than δ/2 where it underflows, which
#   φ ≤ log(floatmax/ε) + 1 < log(floatmax) + |log floatmin| + 1 < 2^(L + 1) multiplies (ε is normal
#   in T; L = _log_exponent(T)); the product loses less than δ/2. When λ times the smallest nonzero
#   weight is a normal number, λ w never underflows, and the per-term loss is below
#   (2^(e(λ) + e(w)) + 1) δ/2; otherwise below (2^(L+1) + 2^(e(λ) + e(w)) + 1) δ/2.
# - NormLq: φ = |v|^q >= |v| is never zero for v ≠ 0, and where φ is subnormal its rounding loses
#   less than δ/2 <= φ, so the true φ is below twice the rounded one; a coefficient λ w that can
#   underflow (λ times the smallest nonzero weight below floatmin) is multiplied by an unbounded φ:
#   no bound. Otherwise a zero term is the product rounded to zero, below δ/2, or 2 δ/2 for a
#   subnormal φ.
@inline _weights_fit(::Type{T}, w) where {T} = w === nothing || promote_type(T, float(eltype(w))) === T
function _zero_value_exponent(f::CappedL1, x::AbstractArray)
  T = _valtype(x)
  iszero(f.λ) && return _EXACT_ZERO
  (_level_fits(T, f.λ) && _level_fits(T, f.a) && _weights_fit(T, f.w)) || return nothing
  _underflow_exponent(T, length(x), f.w === nothing ? 0 : 1 + max(0, _upper_exponent(convert(T, f.a))))
end
function _zero_value_exponent(f::LogPenalty, x::AbstractArray)
  T = _valtype(x)
  iszero(f.λ) && return _EXACT_ZERO
  (_level_fits(T, f.λ) && _level_fits(T, f.ε) && _weights_fit(T, f.w)) || return nothing
  eλ = _upper_exponent(convert(T, f.λ))
  f.w === nothing && return _underflow_exponent(T, length(x), max(0, eλ))
  we = _weight_exponents(f.w)
  we === nothing && return nothing
  ec = eλ + we[2]
  # λ is normal in T, so λ >= 2^(eλ - 1), and the smallest nonzero weight is at least 2^we[1]
  eλ - 1 + we[1] > exponent(floatmin(T)) && return _underflow_exponent(T, length(x), max(0, ec))
  _underflow_exponent(T, length(x), 1 + max(_log_exponent(T) + 1, ec))
end
function _zero_value_exponent(f::NormLq, x::AbstractArray)
  T = _valtype(x)
  iszero(f.λ) && return _EXACT_ZERO
  (_level_fits(T, f.λ) && _weights_fit(T, f.w)) || return nothing
  f.w === nothing && return _underflow_exponent(T, length(x), 1)
  we = _weight_exponents(f.w)
  (we !== nothing && _upper_exponent(convert(T, f.λ)) - 1 + we[1] > exponent(floatmin(T))) || return nothing
  _underflow_exponent(T, length(x), 1)
end
Base.@propagate_inbounds function _entry_terms(acc::_ExpSum{R}, f::CappedL1, v, i) where {R}
  u = convert(R, v)
  isnan(u) && return nothing
  _weighted_term_add(acc, _entry(f.w, i), frexp(min(abs(u), convert(R, f.a))))
end
Base.@propagate_inbounds function _entry_terms(acc::_ExpSum{R}, f::LogPenalty, v, i) where {R}
  # a zero weight or an infinite ε leaves the entry free, NaN included, as in value_entry
  (iszero(_entry(f.w, i)) || isinf(f.ε)) && return acc
  t, ε = abs(convert(R, v)), convert(R, f.ε)
  isfinite(t) || return nothing
  iszero(t) && return acc
  q = t / ε
  φ = if isinf(q)
    frexp(log(t) - log(ε))
  elseif q < floatmin(R)
    mt, et = frexp(t)
    mε, eε = frexp(ε)
    _parts(mt / mε, et - eε)
  else
    frexp(log1p(q))
  end
  _weighted_term_add(acc, _entry(f.w, i), φ)
end
Base.@propagate_inbounds function _entry_terms(acc::_ExpSum{R}, f::NormLq, v, i) where {R}
  t = abs(convert(R, v))
  isfinite(t) || return nothing
  _weighted_term_add(acc, _entry(f.w, i), frexp(t^convert(R, f.q)))
end

# The group penalties: Σₖ pₖ(‖x[Gₖ]‖₂) with the norm of a group and the level λwₖ as parts, and the
# pieces of the MCP and SCAD terms (separable.jl). A group with an infinite coefficient and a
# nonzero entry is an infinite term, and the value is kept; a zero level is exact.
_has_value_parts(::Type{<:_RadialGroupPenalty}) = true
# Exact-zero protection: the group norms are accurate (no square is rounded into the subnormal range
# before the root) and zero only for a zero group, and the MCP or SCAD terms of the groups lose as for
# the scalar penalties (separable.jl); a term is at most λ²wₖ²(a + 1)/2, so a coefficient λwₖ that
# underflows leaves only a negligible term. That cap needs a finite a: with a = Inf the term is
# λwₖ t with no cap, and a coefficient λwₖ that underflows is multiplied by an unbounded group norm
# t, so an infinite a has no bound (the scalar penalties have a single level λ, which fits T).
_zero_value_exponent(f::_RadialGroupPenalty, x::AbstractArray) = iszero(f.λ0) ? _EXACT_ZERO :
  _level_fits(_valtype(x), f.a) && isfinite(f.a) && abs(convert(_valtype(x), f.a)) >= floatmin(_valtype(x)) ?
    _underflow_exponent(_valtype(x), length(f.groups) + 1, _mcp_scad_amp(convert(_valtype(x), f.a))) : nothing
function _value_parts(f::_RadialGroupPenalty, x::AbstractArray, v)
  (iszero(f.λ0) || !isfinite(f.λ0) || isnan(f.a)) && return nothing
  R = promote_type(_valtype(x), float(typeof(f.λ0)))
  acc = _expsum_start(R)
  for k in eachindex(f.groups)
    G, w = f.groups[k], f.w[k]
    iszero(w) && continue                       # a zero weight leaves the group free, NaN included
    t = _group_norm_parts(R, x, G)
    t === nothing && return nothing
    iszero(t[1]) && continue
    isfinite(w) || return nothing
    λ = _parts_product(_rparts(R, f.λ0), _rparts(R, w))
    acc = f isa GroupMCP ? _mcp_terms(acc, λ, f.a, t) : _scad_terms(acc, λ, f.a, t)
    acc === nothing && return nothing
  end
  _expsum_pair(R, acc)
end
# ‖x[G]‖₂ as parts, with the entries scaled by the largest before squaring; `nothing` for a NaN
# entry
function _group_norm_parts(::Type{R}, x::AbstractArray, G) where {R}
  m = zero(R)
  inf = false
  @inbounds for i in G
    a = abs(convert(R, x[i]))
    isnan(a) && return nothing
    isinf(a) ? (inf = true) : (m = max(m, a))
  end
  # an infinite entry and no NaN: an infinite norm, as `_magnitude_parts` marks it, so that the MCP
  # and SCAD terms take their cap (E3) and a = Inf gives an infinite term
  inf && return (one(R) / 2, typemax(Int) ÷ 4)
  iszero(m) && return (m, 0)
  e = exponent(m)
  s = zero(R)
  @inbounds for i in G
    s += abs2(ldexp(convert(R, x[i]), -e))
  end
  _parts(sqrt(s), e)
end
