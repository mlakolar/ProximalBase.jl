####################################
#
#  Norms that are not separable
#
####################################

####################################
#  ℓ2 norm
####################################

"""
    NormL2(λ = 1)

The Euclidean norm `f(x) = λ ‖x‖₂` (the Frobenius norm for a matrix). Its
proximal operator shrinks `x` towards zero: `max(1 - γλ/‖x‖, 0) x`. For the
group lasso, a sum of such norms over groups of entries, use
[`GroupNormL2`](@ref).
Values follow the overflow rule in [Overflow and underflow of values](@ref).
"""
struct NormL2{T<:Real} <: AbstractFunction
  λ::T

  NormL2{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ))
end
NormL2(λ::Real = 1) = NormL2{typeof(λ)}(λ)

isproximable(::Type{<:NormL2}) = true
isconvex(::Type{<:NormL2}) = true
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::NormL2, y::AbstractArray, x::AbstractArray) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); true)

(f::NormL2)(x::AbstractArray) = (_checkreal(f, x); _weighted_norm2(f.λ, x))

function _weighted_norm2(λ::Real, x::AbstractArray)
  T = _valtype(x)
  iszero(λ) && return zero(T)
  _level_fits(T, λ) || return _converted_norm2(λ, x)
  _norm2_value(convert(T, λ) * _norm2(x), convert(T, λ), x)
end

# Scale in the parameter's precision, without narrowing its level first.
@noinline function _converted_norm2(λ, x)
  T = _valtype(x)
  R = promote_type(T, float(typeof(λ)))
  all(isfinite, x) || return convert(T, convert(R, λ) * _norm2(x))
  convert(T, _scaled_norm2_value(convert(R, λ), x))
end

@inline function _norm2_value(v, λ::T, x::AbstractArray) where {T}
  isfinite(v) || !isfinite(λ) || !all(isfinite, x) ? v : _scaled_norm2_value(λ, x)
end

@noinline function _scaled_norm2_value(λ::T, x::AbstractArray) where {T}
  k = _magnitude_exponent(_maximum_abs(x))
  n = sqrt(_scaled_value_sum(x, _power_factors(T, -k), abs2))
  _scaled_product(λ, n, k)
end

function _norm2_prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real, ::Val{V}) where {V}
  _checkwrite(V ? value_and_prox! : prox!, f, y, x)
  T = float(eltype(y))
  if iszero(f.λ)
    y === x || copyto!(y, x)
    return zero(_valtype(y))
  end
  c = _level_product(T, γ, f.λ)
  nx = convert(T, _norm2(x))
  if nx > c
    if isinf(nx) && isfinite(c)
      n = _norm2_shrink!(f, y, x, c)
      return V ? (_level_fits(T, f.λ) ? _norm2_value(convert(T, f.λ) * n, convert(T, f.λ), y) : _weighted_norm2(f.λ, y)) : zero(T)
    end
    t = 1 - c / nx
    y .= t .* x
    return V ? (_level_fits(T, f.λ) ? _norm2_value(convert(T, f.λ) * (nx - c), convert(T, f.λ), y) : _weighted_norm2(f.λ, y)) : zero(T)
  else
    fill!(y, zero(eltype(y)))
    return zero(T)
  end
end
value_and_prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real) = _norm2_prox!(y, f, x, γ, Val(true))
prox!(y::AbstractArray, f::NormL2, x::AbstractArray, γ::Real) = (_norm2_prox!(y, f, x, γ, Val(false)); y)

@inline _norm2_shrink!(f::NormL2, y::AbstractArray, x::AbstractArray, c) =
  _shrink_group!(f, y, x, eachindex(IndexLinear(), x), c)

# write each stored symmetric entry once, while reductions read the whole group
struct _LowerGroupPosition
  p::Int
end
@inline (lower::_LowerGroupPosition)(i) = mod1(i, lower.p) >= (i - 1) ÷ lower.p + 1
# ordinary outputs carry no second group argument through recursive rescaling
@inline _writes(::Nothing, G) = G
@inline _writes(W, G) = W
_group_writes(y::AbstractArray, G) = nothing
_group_writes(y::SymmetricSparseIterate, G) = Iterators.filter(_LowerGroupPosition(size(y, 1)), G)

@inline function _norm2_shrink!(f::NormL2, y::SymmetricSparseIterate, x::AbstractArray, c)
  G = eachindex(IndexLinear(), x)
  _shrink_group!(f, y, x, G, c, _group_writes(y, G))
end

# A finite input scaled by its maximum cannot need another rescaling. Keeping
# that branch out of this call also keeps the scaled array from escaping.
function _rescaled_group_prox!(::NormL2, y, x, G, c::S, scale, W = nothing)::S where {S}
  z = _ScaledArray(x, inv(scale))
  nx = _group_norm2(z, G)
  threshold = c / scale
  n = zero(nx)
  if nx > threshold
    t = one(nx) - threshold / nx
    @inbounds for i in _writes(W, G)
      y[i] = t * z[i]
    end
    n = nx - threshold
  else
    @inbounds for i in _writes(W, G)
      y[i] = zero(eltype(y))
    end
  end
  @inbounds for i in _writes(W, G)
    y[i] *= scale
  end
  n * scale
end

####################################
#  cubed Euclidean norm
####################################

"""
    CubeNormL2(λ = 1)

The cubed Euclidean norm `f(x) = (λ/3) ‖x‖₂³` for finite `λ ≥ 0` (the
Frobenius norm for a matrix). Convex and smooth, with gradient `λ ‖x‖₂ x`,
and proximable, with proximal operator `2x / (1 + √(1 + 4γλ‖x‖₂))`.
The gradient has no global Lipschitz constant unless `λ = 0`.

The argument may be an array of any shape. The gradient and proximal operator
may overwrite it: `gradient!(x, f, x)` and `prox!(x, f, x, γ)` are supported.
"""
struct CubeNormL2{T<:Real} <: AbstractFunction
  λ::T

  function CubeNormL2{T}(λ::T) where {T<:Real}
    _checknonneg("λ", λ)
    isfinite(λ) || throw(ArgumentError("λ must be finite, got $λ"))
    new{T}(λ)
  end
end
CubeNormL2(λ::Real = 1) = CubeNormL2{typeof(λ)}(λ)

issmooth(::Type{<:CubeNormL2}) = true
isproximable(::Type{<:CubeNormL2}) = true
isconvex(::Type{<:CubeNormL2}) = true

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)},
            f::CubeNormL2, y::AbstractArray, x::AbstractArray) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x))

# A finite array can have an unrepresentable norm. Only the norm calculation
# uses scaled entries; the final factor is applied to the original entries.
# A norm whose square is subnormal can also lose precision in norm's fallback.
_cube_norm_ordinary(n) = sqrt(floatmin(typeof(n))) <= n <= sqrt(floatmax(typeof(n)))

function _cube_norm_parts(x, n, ::Type{T}) where {T}
  _cube_norm_ordinary(n) && return frexp(convert(T, n))
  m = _maximum_abs(x)
  (iszero(m) || !isfinite(m)) && return (convert(T, m), 0)
  _, e = frexp(m)
  s = _power_factors(T, -e)
  a, k = frexp(sqrt(_scaled_sumabs2(x, s)))
  a, e + k
end

# Keep all four factors in range until the final rounding, as _positive_product
# does for three. In particular, neither λ/3 nor n³ needs to be representable.
function _cube_value_parts(λ::T, m::T, e::Int) where {T}
  (iszero(m) || !isfinite(m)) && return λ * m
  a, k = frexp(λ)
  ldexp(((a * m) * m) * (m / 3), k + 3e)
end

function (f::CubeNormL2)(x::AbstractArray)
  _checkreal(f, x)
  T = float(eltype(x))
  iszero(f.λ) && return zero(T)
  S = promote_type(T, float(typeof(f.λ)))
  λ = convert(S, f.λ)
  n = _norm2(x)
  if _cube_norm_ordinary(n)
    a = _positive_product(λ, convert(S, n), convert(S, n))
    b = convert(S, n) / 3
    if floatmin(S) <= a <= floatmax(S) && b >= floatmin(S)
      return convert(T, a * b)
    end
  end
  m, e = _cube_norm_parts(x, n, S)
  convert(T, _cube_value_parts(λ, m, e))
end

# A scale may not fit even when its product with an entry does. Round only
# after combining the exponents, preserving zeros beside overflowing entries.
@inline function _cube_scale(v::Real, m::T, e::Int) where {T}
  a = convert(T, v)
  iszero(a) && return a
  b, k = frexp(a)
  ldexp(b * m, k + e)
end

function gradient!(g::AbstractArray, f::CubeNormL2, x::AbstractArray)
  _checkwrite(gradient!, f, g, x)
  iszero(f.λ) && return fill!(g, zero(eltype(g)))
  T = promote_type(float(eltype(g)), float(eltype(x)), float(typeof(f.λ)))
  λ = convert(T, f.λ)
  n = _norm2(x)
  a = λ * convert(T, n)
  if _cube_norm_ordinary(n) && floatmin(T) <= a <= floatmax(T)
    return g .= a .* x
  end
  m, e = _cube_norm_parts(x, n, T)
  (iszero(m) || !isfinite(m)) && return g .= (λ * m) .* x
  b, k = frexp(λ)
  g .= _cube_scale.(x, b * m, k + e)
end

function value_and_gradient!(g::AbstractArray, f::CubeNormL2, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  v = f(x)                                 # x may be the gradient's output
  gradient!(g, f, x)
  v
end

# With z = γλn, write 1/(1/2 + √(z + 1/4)) as a bounded factor times a
# power of two. Neither z nor the shrinkage factor has to fit in T.
function _cube_prox_parts(γ::T, λ::T, m::T, e::Int) where {T}
  if !isfinite(m) || !isfinite(γ) || iszero(γ)
    return (2 / (1 + sqrt(1 + 4 * γ * λ * m)), 0)
  end
  a, k = frexp(γ)
  b, j = frexp(λ)
  c, i = frexp((a * b) * m)
  h = k + j + e + i
  if h <= 0
    return (2 / (1 + sqrt(1 + 4 * ldexp(c, h))), 0)
  end
  q = fld(h, 2)
  s = sqrt(ldexp(c, h - 2q))
  t = ldexp(inv(s), -q)
  (inv(s) / (sqrt(1 + (t / 2)^2) + t / 2), -q)
end

function prox!(y::AbstractArray, f::CubeNormL2, x::AbstractArray, γ::Real)
  _checkwrite(prox!, f, y, x)
  if iszero(f.λ)
    y === x || copyto!(y, x)
    return y
  end
  T = promote_type(float(eltype(y)), float(eltype(x)), float(typeof(f.λ)), float(typeof(γ)))
  λ, step = convert(T, f.λ), convert(T, γ)
  n = _norm2(x)
  iszero(n) && return fill!(y, zero(eltype(y)))
  z = _positive_product(step, λ, convert(T, n))
  if _cube_norm_ordinary(n) && isfinite(z) && z <= floatmax(T) / 4
    t = 2 / (1 + sqrt(1 + 4z))
    return y .= t .* x
  end
  m, e = _cube_norm_parts(x, n, T)
  a, k = _cube_prox_parts(step, λ, m, e)
  isfinite(m) || return y .= a .* x
  y .= _cube_scale.(x, a, k)
end

value_and_prox!(y::AbstractArray, f::CubeNormL2, x::AbstractArray, γ::Real) =
  (prox!(y, f, x, γ); f(y))

lipschitz(f::CubeNormL2) = _ind(iszero(f.λ), float(typeof(f.λ)))

####################################
#  ℓ∞ norm
####################################

"""
    NormLinf(λ = 1)

The ℓ∞ norm `f(x) = λ maxᵢ |xᵢ|`. Its proximal operator clips the entries at a
level `τ`, `sign(xᵢ) min(|xᵢ|, τ)`: by Moreau's identity it is `x` minus the
projection onto the ℓ1 ball of radius `γλ`, which takes O(n) expected time
(Condat's algorithm).
Values follow the overflow rule in [Overflow and underflow of values](@ref).
"""
struct NormLinf{T<:Real} <: AbstractFunction
  λ::T
  scratch::Scratch

  NormLinf{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ, Scratch()))
end
NormLinf(λ::Real = 1) = NormLinf{typeof(λ)}(λ)

isproximable(::Type{<:NormLinf}) = true
isconvex(::Type{<:NormLinf}) = true
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::NormLinf, y::AbstractArray, x::AbstractArray) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); true)

(f::NormLinf)(x::AbstractArray) = (_checkreal(f, x); _weighted_normlinf(f.λ, x))

function _weighted_normlinf(λ::Real, x::AbstractArray)
  T = _valtype(x)
  _level_fits(T, λ) && return convert(T, λ) * norm(x, Inf)
  _converted_normlinf(λ, x)
end

@noinline function _converted_normlinf(λ, x)
  T = _valtype(x)
  R = promote_type(T, float(typeof(λ)))
  convert(T, convert(R, λ) * convert(R, norm(x, Inf)))
end

function value_and_prox!(y::AbstractArray, f::NormLinf, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, f, y, x)
  T = float(eltype(y))
  c = _level_product(T, γ, f.λ)
  if norm(x, 1) <= c
    fill!(y, zero(eltype(y)))
    return zero(T)
  end
  τ = _simplex_threshold!(_buffer!(f.scratch, T, length(x)), x, eachindex(IndexLinear(), x), c, abs)
  y .= sign.(x) .* min.(abs.(x), τ)
  _level_fits(T, f.λ) ? convert(T, f.λ) * τ : _weighted_normlinf(f.λ, y)
end
prox!(y::AbstractArray, f::NormLinf, x::AbstractArray, γ::Real) = (value_and_prox!(y, f, x, γ); y)

####################################
#  group norms  λ0 ⋅ Σₖ wₖ ‖x[Gₖ]‖
####################################

# Group penalties share their groups, weights and the map from an index to its
# group. Only the group norms are positively homogeneous and convex; MCP and
# SCAD use the same grouping machinery but have different proximal kernels.
"""
    ProximalBase.AbstractGroupPenalty

Abstract supertype of penalties on disjoint groups of entries, including
[`GroupNormL2`](@ref), [`GroupNormLinf`](@ref), [`GroupMCP`](@ref) and
[`GroupSCAD`](@ref). They share groups, weights and a map from each entry to
its group. The group norms are convex and positively homogeneous; the group
MCP and SCAD penalties are nonconvex. An [`AtomIterate`](@ref) can use the
groups as its coordinates when they partition its entries.
"""
abstract type AbstractGroupPenalty <: AbstractFunction end
abstract type HomogeneousGroupNorm <: AbstractGroupPenalty end

# Keep the qualified name used by solvers to select grouped iterates. Its old
# meaning included MCP and SCAD; norm-only algorithms use HomogeneousGroupNorm.
const AbstractGroupNorm = AbstractGroupPenalty

isproximable(::Type{<:AbstractGroupPenalty}) = true
isconvex(::Type{<:HomogeneousGroupNorm}) = true

"""
    GroupNormL2(λ0, groups)
    GroupNormL2(λ0, groups, w)

The group-lasso penalty `f(x) = λ0 Σₖ wₖ ‖x[Gₖ]‖₂` over disjoint groups
`Gₖ = groups[k]` of (linear) indices of `x`, of any sizes, such as `1:5` or
`[2, 7, 9]`. The weights default to `wₖ = √|Gₖ|`, so that groups of one index
give `NormL1(λ0)`. Indices in no group are not penalized. The proximal operator
shrinks each group towards zero in Euclidean norm.

The rows of a `p × k` matrix are the groups `[i:p:p*k for i in 1:p]`, which
gives the multi-task group lasso with a matrix response in
[`LeastSquaresLoss`](@ref).

On an [`AtomIterate`](@ref), whose coordinates are groups,
[`prox_coordinate!`](@ref) updates a group and [`kkt_violation`](@ref) measures
one. `AtomIterate(x, f)` builds an iterate that shares the groups of `f`, which
must then partition `1:length(x)`; include otherwise unpenalized entries in
additional groups with weight 0. With equal groups, the value and the proximal
operator touch only the active groups. On ordinary arrays, active-set solvers can use
[`restrict`](@ref) and [`kkt_violation`](@ref).

# Example
```julia
f = GroupNormL2(0.5, [1:3, 4:5, [6, 8]])    # x[7] is not penalized
prox(f, randn(8))
```
Values follow the overflow rule in [Overflow and underflow of values](@ref).
"""
struct GroupNormL2{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: HomogeneousGroupNorm
  λ0::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}   # group_of[i] = k for i ∈ Gₖ, 0 for an index in no group
  free::Vector{Int}       # the indices up to the largest in a group that are in none

  function GroupNormL2{T, G}(λ0::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    length(w) == length(groups) || throw(DimensionMismatch("There must be one weight per group."))
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    group_of = _group_of(groups)
    new{T, G}(λ0, groups, w, group_of, findall(iszero, group_of))
  end
end

"""
    GroupNormLinf(λ0, groups)
    GroupNormLinf(λ0, groups, w)

The ℓ1/ℓ∞ penalty `f(x) = λ0 Σₖ wₖ maxᵢ∈Gₖ |xᵢ|` over disjoint groups, as
[`GroupNormL2`](@ref) but with the ℓ∞ norm of each group, for simultaneous
selection in multi-task problems. The weights default to `wₖ = 1`. The proximal
operator clips each group, by Moreau's identity with the projection onto an ℓ1
ball, and its optimality test ([`kkt_violation`](@ref)) uses the dual norm,
`‖g[Gₖ]‖₁ - λ0 wₖ`.
Values follow the overflow rule in [Overflow and underflow of values](@ref).
"""
struct GroupNormLinf{T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: HomogeneousGroupNorm
  λ0::T
  groups::G
  w::Vector{T}
  group_of::Vector{Int}
  free::Vector{Int}
  scratch::Scratch

  function GroupNormLinf{T, G}(λ0::T, groups::G, w::Vector{T}) where {T<:Real, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    length(w) == length(groups) || throw(DimensionMismatch("There must be one weight per group."))
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    group_of = _group_of(groups)
    new{T, G}(λ0, groups, w, group_of, findall(iszero, group_of), Scratch())
  end
end

for F in (:GroupNormL2, :GroupNormLinf)
  @eval function $F(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector)
    eltype(w) <: Real || _real_error(nameof($F))
    T = float(promote_type(typeof(λ0), eltype(w)))
    G = _own_groups(groups)
    $F{T, typeof(G)}(convert(T, λ0), G, convert(Vector{T}, w))
  end
end
GroupNormL2(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}) =
  GroupNormL2(λ0, groups, [sqrt(float(typeof(λ0))(length(G))) for G in groups])
GroupNormLinf(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}) =
  GroupNormLinf(λ0, groups, ones(float(typeof(λ0)), length(groups)))

_rebuild(f::GroupNormL2, λ0, groups, w) = GroupNormL2(λ0, groups, w)
_rebuild(f::GroupNormLinf, λ0, groups, w) = GroupNormLinf(λ0, groups, w)

# the norm of the group G of x and its dual norm, by loops over the indices: a
# view of a matrix at linear indices reshapes it, which allocates
@inline function _sumsq(x::AbstractArray, G)
  T = float(eltype(x))
  s = zero(T)
  @inbounds @simd for i in G
    s += abs2(convert(T, x[i]))
  end
  s
end
# As for whole arrays, only exceptional magnitudes need the scaled path. A
# zero group, the common case in a sparse iterate, also has a zero sum of
# squares; a vectorized sum of magnitudes confirms it without hypot.
@inline function _group_norm2(x::AbstractArray, G)
  s = _sumsq(x, G)
  isfinite(s) && s >= floatmin(typeof(s)) && return sqrt(s)
  iszero(s) && iszero(_sumabs(x, G)) && return s
  n = zero(s)
  @inbounds for i in G
    isnan(x[i]) && return oftype(n, NaN)
    n = hypot(n, x[i])
  end
  n
end

@inline function _sumabs(x::AbstractArray, G)
  T = float(eltype(x))
  s = zero(T)
  @inbounds @simd for i in G
    s += abs(convert(T, x[i]))
  end
  s
end
function _maxabs(x::AbstractArray, G)
  s = zero(float(eltype(x)))
  @inbounds for i in G
    s = max(s, abs(x[i]))
  end
  s
end
function _allzero(x::AbstractArray, G)
  @inbounds for i in G
    iszero(x[i]) || return false
  end
  true
end
_group_norm(::GroupNormL2, x, G) = _group_norm2(x, G)
_group_norm(::GroupNormLinf, x, G) = _maxabs(x, G)
_dual_norm(::GroupNormL2, g, G) = _group_norm2(g, G)
_dual_norm(::GroupNormLinf, g, G) = _sumabs(g, G)

# A full reduction lets finite-weight scans vectorize.
_has_infinite_group_weight(w) = mapreduce(isinf, |, w; init = false)

# Classify once; matching atom layouts inspect only their active groups.
function _ordinary_group_weights(f, ::Type{T}) where {T}
  isfinite(f.λ0) || return false
  # Nonnegative weights make the largest product bound every coefficient.
  w = maximum(f.w; init = zero(eltype(f.w)))
  isfinite(w) && isfinite(convert(T, f.λ0 * w))
end
function _ordinary_group_weights(f, ::Type{T}, ks, n) where {T}
  isfinite(f.λ0) || return false
  ordinary = true
  @inbounds @simd for a in 1:n
    w = f.w[ks[a]]
    ordinary &= isfinite(w) & isfinite(convert(T, f.λ0 * w))
  end
  ordinary
end

# Keep the ordinary group product; recover a finite group in scaled units.
@inline function _group_value(f, λ, w, n, x, G, ::Type{T}) where {T}
  isinf(w) && return convert(T, _weighted_value(w * n, n))
  (iszero(λ) || iszero(w) || iszero(n)) && return zero(T)
  if _level_fits(T, λ) && _level_fits(T, w)
    v = _positive_product(convert(T, λ), convert(T, w), n)
    isfinite(v) && return convert(T, v)
  end
  convert(T, _scaled_group_value(f, λ, w, x, G, n))
end

@inline function _group_value(f, λ, w, n, x, G, ::Type{T}, ::Val{true}) where {T}
  if _level_fits(T, λ) && _level_fits(T, w)
    v = _positive_product(convert(T, λ), convert(T, w), n)
    isfinite(v) && return convert(T, v)
  end
  convert(T, _scaled_group_value(f, λ, w, x, G, n))
end

@inline _group_value(f, λ, w, n, x, G, ::Type{T}, ::Val{false}) where {T} =
  _group_value(f, λ, w, n, x, G, T)

@noinline function _scaled_group_value(f, λ, w, x, G, ordinary_norm)
  T = float(eltype(x))
  R = promote_type(T, float(typeof(λ)), float(typeof(w)))
  m = convert(R, _maxabs(x, G))
  isfinite(m) || return _positive_product(convert(R, λ), convert(R, w), convert(R, ordinary_norm))
  e = _magnitude_exponent(m)
  scales = _power_factors(R, -e)
  s = err = zero(R)
  @inbounds for i in G
    z = abs(_scale_value(convert(R, x[i]), scales))
    if f isa GroupNormLinf
      s = max(s, z)
    else
      s, err = _compensated_add(s, err, abs2(z))
    end
  end
  n = f isa GroupNormLinf ? s : sqrt(s + err)
  _scaled_product(λ, w, n, e)
end

# the group of index i, 0 if i is in no group
_group(f::AbstractGroupPenalty, i::Integer) = i <= length(f.group_of) ? f.group_of[i] : 0

# the groups fit in x: the loops over a group skip the bounds checks
@inline _checkgroups(f::AbstractGroupPenalty, x::AbstractArray) =
  length(x) >= length(f.group_of) || throw(BoundsError(x, length(f.group_of)))

# y[G] ← prox_{c‖⋅‖₂}(x[G]), which may be in place; returns the new norm of the group
@inline function _shrink_group!(f::Union{NormL2, GroupNormL2}, y::AbstractArray{T}, x::AbstractArray, G, c, W = nothing) where {T}
  nx = _group_norm2(x, G)
  if nx > c
    if isinf(nx) && isfinite(c)
      scale = _maxabs(x, G)
      isfinite(scale) && scale > 1 && return _rescaled_group_prox!(f, y, x, G, c, scale, W)
    end
    t = one(nx) - c / nx
    @inbounds for i in _writes(W, G)
      y[i] = t * x[i]
    end
    return nx - c
  else
    @inbounds for i in _writes(W, G)
      y[i] = zero(T)
    end
    return zero(nx)
  end
end

# y[G] ← prox_{c‖⋅‖∞}(x[G]), which may be in place: the entries clipped at τ;
# returns τ, the new norm of the group
function _shrink_group!(f::GroupNormLinf, y::AbstractArray{T}, x::AbstractArray, G, c, W = nothing) where {T}
  S = float(T)
  s = _sumabs(x, G)
  if s <= c
    @inbounds for i in _writes(W, G)
      y[i] = zero(T)
    end
    return zero(S)
  end
  if isinf(s) && isfinite(c)
    scale = _maxabs(x, G)
    isfinite(scale) && scale > 1 && return _rescaled_group_prox!(f, y, x, G, c, scale, W)
  end
  τ = _simplex_threshold!(_buffer!(f.scratch, S, length(G)), x, G, convert(S, c), abs)
  @inbounds for i in _writes(W, G)
    v = x[i]
    y[i] = sign(v) * min(abs(v), τ)
  end
  τ
end

# A finite input can have a norm (or simplex sum) outside the working range.
# Homogeneity lets the same kernel operate in smaller units, without losing
# the shrinkage to c/Inf. This path is only needed for large finite entries.
function _rescaled_group_prox!(f, y, x, G, c::S, scale, W = nothing)::S where {S}
  n = _shrink_group!(f, y, _ScaledArray(x, inv(scale)), G, c / scale, W)
  @inbounds for i in _writes(W, G)
    y[i] *= scale
  end
  n * scale
end

function (f::HomogeneousGroupNorm)(x::AbstractArray)
  _checkreal(f, x)
  isfinite(f.λ0) && !iszero(f.λ0) && return _homogeneous_group_value(f, x, Val(true))
  _ordinary_group_weights(f, float(eltype(x))) ?
    _homogeneous_group_value(f, x, Val(true)) : _homogeneous_group_value(f, x, Val(false))
end

function _homogeneous_group_value(f::HomogeneousGroupNorm, x::AbstractArray, ordinary::Val{O}) where {O}
  _checkgroups(f, x)
  T = float(eltype(x))
  O && iszero(f.λ0) && return zero(T)
  v = zero(T)
  for k in eachindex(f.groups)
    iszero(f.w[k]) && continue
    v += _group_value(f, f.λ0, f.w[k], _group_norm(f, x, f.groups[k]), x, f.groups[k], T, ordinary)
  end
  v
end

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::AbstractGroupPenalty, y::AbstractArray, x::AbstractArray) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); _checkgroups(f, x); _check_symmetric_groups(f, y))

_check_symmetric_groups(f, y::AbstractArray) = true

function _check_symmetric_groups(f, y::SymmetricSparseIterate)
  p = size(y, 1)
  @inbounds for i in eachindex(f.group_of)
    k = f.group_of[i]
    iszero(k) && continue
    r, c = mod1(i, p), (i - 1) ÷ p + 1
    mirror = _group(f, c + (r - 1) * p)
    if k != mirror
      other = iszero(mirror) ? "in no group" : "in group $mirror"
      throw(ArgumentError("the groups of $(typeof(f)) must be symmetric for a SymmetricSparseIterate output, but position ($r, $c) is in group $k and position ($c, $r) is $other"))
    end
  end
  true
end

value_and_prox!(y::AbstractArray, f::AbstractGroupPenalty, x::AbstractArray, γ::Real) =
  (_checkwrite(value_and_prox!, f, y, x); _value_and_prox_groups!(y, f, x, γ))
prox!(y::AbstractArray, f::AbstractGroupPenalty, x::AbstractArray, γ::Real) =
  (_checkwrite(prox!, f, y, x); value_and_prox!(y, f, x, γ); y)

# Copy indices outside the penalized groups, including those beyond the last
# group. Call after checking sizes and groups; all proximal kernels leave these
# entries unchanged, and an in-place call already has the required values.
@inline function _copy_ungrouped!(y, f::AbstractGroupPenalty, x)
  if y !== x
    @inbounds for i in f.free
      y[i] = x[i]
    end
    @inbounds for i in length(f.group_of)+1:length(x)
      y[i] = x[i]
    end
  end
  y
end

# the step γ is a scalar or a vector with a step per group (see metric.jl)
function _value_and_prox_groups!(y::AbstractArray, f::HomogeneousGroupNorm, x::AbstractArray, γ::Union{Real, AbstractVector})
  _ordinary_group_weights(f, float(eltype(y))) ?
    _homogeneous_group_prox!(y, f, x, γ, Val(true)) : _homogeneous_group_prox!(y, f, x, γ, Val(false))
end

function _homogeneous_group_prox!(y::AbstractArray, f::HomogeneousGroupNorm, x::AbstractArray, γ::Union{Real, AbstractVector}, ordinary::Val{O}) where {O}
  _checksize(y, x)
  _checkgroups(f, x)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  if O && iszero(f.λ0)
    y === x || copyto!(y, x)
    return zero(T)
  end
  _copy_ungrouped!(y, f, x)
  v = zero(T)
  for k in eachindex(f.groups)
    if iszero(f.w[k])
      @inbounds for i in f.groups[k]
        y[i] = x[i]
      end
      continue
    end
    nrm = _shrink_group!(f, y, x, f.groups[k], convert(T, O ? _positive_product(_entry(γ, k), f.λ0, f.w[k]) : _weight_product(_entry(γ, k), f.λ0, f.w[k])), _group_writes(y, f.groups[k]))
    v += _group_value(f, f.λ0, f.w[k], nrm, y, f.groups[k], T, ordinary)
  end
  v
end

# a step per group (see metric.jl): one entry for every group
_checkgroupsteps(::AbstractGroupPenalty, ::Real) = true
_checkgroupsteps(f::AbstractGroupPenalty, γ::AbstractVector) = length(γ) == length(f.groups) ||
  throw(DimensionMismatch("a step per group: $(length(f.groups)) groups and $(length(γ)) steps"))

# the parts of the groups in S, as positions in S, with their weights
function restrict(f::AbstractGroupPenalty, S::AbstractVector{<:Integer})
  members = Dict{Int, Vector{Int}}()
  for (j, i) in enumerate(S)
    k = _group(f, i)
    k == 0 || push!(get!(Vector{Int}, members, k), j)
  end
  ks = sort!(collect(keys(members)))
  _rebuild(f, f.λ0, [members[k] for k in ks], f.w[ks])
end

function _kkt_violation(f::AbstractGroupPenalty, g::AbstractArray, x::AbstractArray, i::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
  eltype(x) <: Real || _real_error(:kkt_violation)
  _checkgroups(f, g)
  _checkgroups(f, x)
  k = _group(f, i)
  k == 0 && return abs(g[i])
  G = f.groups[k]
  _allzero(x, G) ? _dual_norm(f, g, G) - _weight_coefficient(f.λ0, f.w[k]) : abs(g[i])
end

# on an AtomIterate, k is a group
_kkt_violation(f::AbstractGroupPenalty, g::AbstractArray, x::AtomIterate, k::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); _checkgroups(f, g); _checksamegroups(f, x); _dual_norm(f, g, x.groups[k]) - _weight_coefficient(f.λ0, f.w[k]))

# the coordinates of x are the groups of f
_checksamegroups(f::AbstractGroupPenalty, x::AtomIterate) = _samelayout(f, x) ||
  throw(DimensionMismatch("f and x must have the same groups."))
_kkt_violation(f::AbstractGroupPenalty, g::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); throw(ArgumentError("$(typeof(f)) groups linear indices; it does not apply to the coordinates of a SymmetricSparseIterate")))

# an AtomIterate that shares the groups of f, which must partition 1:p
AtomIterate(p::Integer, f::AbstractGroupPenalty) = AtomIterate(Float64, p, f)
function AtomIterate(::Type{T}, p::Integer, f::AbstractGroupPenalty) where {T}
  _check_partition(f.group_of, p)
  _AtomIterate(zeros(T, p), f.groups, f.group_of)
end
function AtomIterate(x::AbstractVector{T}, f::AbstractGroupPenalty) where {T}
  _check_partition(f.group_of, length(x))
  _AtomIterate(Vector{T}(x), f.groups, f.group_of)
end

# with the groups of f, only the active groups count
function (f::HomogeneousGroupNorm)(x::AtomIterate)
  _checkreal(f, x)
  _samelayout(f, x) || return invoke(f, Tuple{AbstractArray}, x)
  isfinite(f.λ0) && !iszero(f.λ0) && return _homogeneous_group_value(f, x, Val(true))
  _ordinary_group_weights(f, float(eltype(x)), x.active, x.nactive) ?
    _homogeneous_group_value(f, x, Val(true)) : _homogeneous_group_value(f, x, Val(false))
end

function _homogeneous_group_value(f::HomogeneousGroupNorm, x::AtomIterate, ordinary::Val{O}) where {O}
  _samelayout(f, x) || return invoke(_homogeneous_group_value, Tuple{HomogeneousGroupNorm, AbstractArray, Val{O}}, f, x, ordinary)
  T = float(eltype(x))
  O && iszero(f.λ0) && return zero(T)
  v = zero(T)
  for a in 1:x.nactive
    k = x.active[a]
    (iszero(f.w[k]) || (!O && iszero(f.λ0) && !isinf(f.w[k]))) && continue
    v += _group_value(f, f.λ0, f.w[k], _group_norm(f, x.storage, f.groups[k]), x.storage, f.groups[k], T, ordinary)
  end
  v
end

value_and_prox!(y::AtomIterate, f::AbstractGroupPenalty, x::AtomIterate, γ::Real) =
  (_checkwrite(value_and_prox!, f, y, x); _value_and_prox_groups!(y, f, x, γ))

function _value_and_prox_groups!(y::AtomIterate, f::HomogeneousGroupNorm, x::AtomIterate, γ::Union{Real, AbstractVector})
  _samelayout(f, x) && _samelayout(f, y) ||
    return invoke(_value_and_prox_groups!, Tuple{AbstractArray, HomogeneousGroupNorm, AbstractArray, Union{Real, AbstractVector}}, y, f, x, γ)
  _ordinary_group_weights(f, float(eltype(y)), x.active, x.nactive) ?
    _homogeneous_group_prox!(y, f, x, γ, Val(true)) : _homogeneous_group_prox!(y, f, x, γ, Val(false))
end

function _homogeneous_group_prox!(y::AtomIterate, f::HomogeneousGroupNorm, x::AtomIterate, γ::Union{Real, AbstractVector}, ordinary::Val{O}) where {O}
  _samelayout(f, x) && _samelayout(f, y) ||
    return invoke(_homogeneous_group_prox!, Tuple{AbstractArray, HomogeneousGroupNorm, AbstractArray, Union{Real, AbstractVector}, Val{O}}, y, f, x, γ, ordinary)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  copyto!(y, x)
  O && iszero(f.λ0) && return zero(T)
  v = zero(T)
  for a in 1:y.nactive
    k = y.active[a]
    (iszero(f.w[k]) || (!O && iszero(f.λ0) && !isinf(f.w[k]))) && continue
    nrm = _shrink_group!(f, y.storage, y.storage, f.groups[k], convert(T, O ? _positive_product(_entry(γ, k), f.λ0, f.w[k]) : _weight_product(_entry(γ, k), f.λ0, f.w[k])))
    v += _group_value(f, f.λ0, f.w[k], nrm, y.storage, f.groups[k], T, ordinary)
  end
  v
end

# Validate the layout and group before a wrapper changes any entries.
@inline function _check_group_coordinate(f::AbstractGroupPenalty, x::AtomIterate, k::Integer)
  _checksamegroups(f, x)
  checkbounds(x.groups, k)
  nothing
end

@inline _unchecked_coordinate!(x::AtomIterate, f::AbstractGroupPenalty, k, γ, ctx) = prox_coordinate!(x, f, k, γ)

@inline _check_coordinate(x::AtomIterate, f::HomogeneousGroupNorm, k) = _check_group_coordinate(f, x, k)
@inline _block_step!(x::AtomIterate, f::HomogeneousGroupNorm, k, u, γ, ctx) = _stage_and_step!(x, f, k, u, γ, ctx)

# group k of x shrunk in place; the group becomes active if it ends nonzero
function prox_coordinate!(x::AtomIterate, f::HomogeneousGroupNorm, k::Integer, γ::Real)
  ctx = _check_coordinate(x, f, k)
  _unchecked_coordinate!(x, f, k, γ, ctx)
end

@inline function _unchecked_coordinate!(x::AtomIterate, f::HomogeneousGroupNorm, k, γ, ctx)
  T = float(eltype(x))
  if (iszero(f.λ0) && !isinf(f.w[k])) || iszero(f.w[k])
    _allzero(x.storage, x.groups[k]) || activate!(x, k)
    return atom(x, k)
  end
  nrm = _shrink_group!(f, x.storage, x.storage, x.groups[k], convert(T, isinf(f.w[k]) ? _weight_product(γ, f.λ0, f.w[k]) : _positive_product(γ, f.λ0, f.w[k])))
  iszero(nrm) || activate!(x, k)
  atom(x, k)
end

####################################
#  ℓ1 plus fusion of a pair
####################################

"""
    NormL1Fused(λ1, λ2)

The penalty `f(x1, x2) = Σᵢ λ1 (|x1ᵢ| + |x2ᵢ|) + λ2 |x1ᵢ - x2ᵢ|` of a pair of
arrays of the same size, given as a tuple `(x1, x2)`, as in the joint
estimation of two sparse models that should be close. Its proximal operator acts
entry by entry on the pairs `(x1ᵢ, x2ᵢ)`; `prox(f, (u, v), γ)` gives it for two
scalars.

On a vector `v` of length K it is `λ1 Σₖ |vₖ| + λ2 Σₖ<ₗ |vₖ - vₗ|`, which fuses
all pairs of entries; for K = 2 this is the function of one pair. As the fiber
function of [`FiberSum`](@ref), `FiberSum(NormL1Fused(λ1, λ2))` on an
`n × n × K` array is the penalty of the fused graphical lasso of K classes
(Danaher, Wang and Witten, 2014). With the entries sorted, `v₍₁₎ ≤ … ≤ v₍K₎`,
the fusion term is `Σᵢ (2i - K - 1) v₍ᵢ₎`, linear on the vectors sorted the
same way, and its proximal operator keeps the order of `v`. So the proximal
operator sorts `v`, subtracts `γλ2 (2i - K - 1)` from the `i`-th smallest entry,
pools adjacent violators into a nondecreasing sequence, restores the order, and
soft-thresholds by `γλ1`: exact, in O(K log K), in buffers that the function
owns.
Values follow the overflow rule in [Overflow and underflow of values](@ref).
"""
struct NormL1Fused{T<:Real} <: AbstractFunction
  λ1::T
  λ2::T
  perm::Scratch       # the order of the entries of a vector
  vals::Scratch       # the sorted entries, shifted, then pooled
  sums::Scratch       # the entries, then the sums of the blocks of the pooling
  sizes::Scratch      # the scratch of the sort, then the sizes of the blocks

  function NormL1Fused{T}(λ1::T, λ2::T) where {T<:Real}
    _checknonneg("λ1", λ1)
    _checknonneg("λ2", λ2)
    new{T}(λ1, λ2, Scratch(), Scratch(), Scratch(), Scratch())
  end
end
NormL1Fused(λ1::Real, λ2::Real) = (p = promote(λ1, λ2); NormL1Fused{typeof(p[1])}(p...))

isproximable(::Type{<:NormL1Fused}) = true
_argument_type(::Type{<:NormL1Fused}) = Union{AbstractVector, Tuple{AbstractArray, AbstractArray}}
isconvex(::Type{<:NormL1Fused}) = true

# A level already stored in the value type loses no precision on conversion.
@inline _fused_level_fits(::Type{T}, λ::T) where {T<:AbstractFloat} = true
@inline _fused_level_fits(::Type{T}, λ::Real) where {T} = _level_fits(T, λ)
@inline _fused_levels_fit(::Type{T}, f::NormL1Fused) where {T} =
  _fused_level_fits(T, f.λ1) && _fused_level_fits(T, f.λ2)

# Normalize weighted terms together, so subnormal contributions round once.
@noinline function _fused_scaled_value(terms, k, ::Type{T}) where {T}
  R = typeof(first(terms)[2])
  e = typemin(Int)
  for term in terms
    λ, v = term[1], term[2]
    units = length(term) == 2 ? k : term[3]
    if !iszero(λ) && !iszero(v)
      e = max(e, exponent(R(λ)) + exponent(v) + units)
    end
  end
  e == typemin(Int) && return zero(T)
  s = c = zero(R)
  for term in terms
    λ, v = term[1], term[2]
    units = length(term) == 2 ? k : term[3]
    s, c = _compensated_add(s, c, _scaled_product(R(λ), v, units - e))
  end
  # Generator inference can lose the term type on Julia 1.10.
  convert(T, _scaled_product(one(R), s + c, e))::T
end

@noinline function _recover_fused_pair(f, xy, ::Type{T}) where {T}
  R = promote_type(T, float(typeof(f.λ1)), float(typeof(f.λ2)))
  k = _magnitude_exponent(max(abs(R(xy[1])), abs(R(xy[2]))))
  scale = _power_factors(R, -k)
  u, v = _scale_value(R(xy[1]), scale), _scale_value(R(xy[2]), scale)
  _fused_scaled_value(((f.λ1, abs(u) + abs(v)), (f.λ2, abs(u - v))), k, T)
end

# The unchecked pair expression also serves the aggregate array reduction.
@inline function _fused_pair_value(f, xy)
  T = _valtype(xy)
  u, w = xy[1] isa Integer ? T(xy[1]) : xy[1], xy[2] isa Integer ? T(xy[2]) : xy[2]
  _value_at((iszero(f.λ1) ? zero(float(xy[1])) : f.λ1 * (abs(u) + abs(w))) +
    (iszero(f.λ2) ? zero(float(xy[1])) : f.λ2 * abs(u - w)), xy)
end

# A nonnegative finite result precedes infinity in the IEEE bit ordering.
@inline function _fused_finite(v::T) where {T<:Union{Float16, Float32, Float64}}
  U = T === Float64 ? UInt64 : T === Float32 ? UInt32 : UInt16
  reinterpret(U, v) < reinterpret(U, T(Inf))
end
@inline _fused_finite(v) = isfinite(v) && !signbit(v)

@inline function (f::NormL1Fused)(xy::Tuple{Real, Real})
  T = _valtype(xy)
  v = _fused_pair_value(f, xy)
  _fused_finite(v) && _fused_levels_fit(T, f) && return v
  _checked_fused_pair_value(f, xy, T)
end

@noinline function _checked_fused_pair_value(f, xy, ::Type{T}) where {T}
  ordinary = _fused_pair_value(f, xy)
  all(isfinite, xy) && isfinite(f.λ1) && isfinite(f.λ2) ? _recover_fused_pair(f, xy, T) : ordinary
end

@noinline function _recover_fused_arrays(f, x, ::Type{T}, ordinary) where {T}
  R = promote_type(T, float(typeof(f.λ1)), float(typeof(f.λ2)))
  isfinite(f.λ1) && isfinite(f.λ2) || return ordinary
  m = zero(R)
  for i in eachindex(x[1], x[2])
    m = max(m, abs(R(x[1][i])), abs(R(x[2][i])))
  end
  isfinite(m) || return ordinary
  iszero(m) && return zero(T)
  # Independent pairs may have very different scales and zero fusion gaps.
  terms = (begin
    u, w = R(x[1][i]), R(x[2][i])
    k = _magnitude_exponent(max(abs(u), abs(w)))
    scale = _power_factors(R, -k)
    u, w = _scale_value(u, scale), _scale_value(w, scale)
    h == 1 ? (f.λ1, abs(u) + abs(w), k) : (f.λ2, abs(u - w), k)
  end for i in eachindex(x[1], x[2]), h in 1:2)
  _fused_scaled_value(terms, 0, T)
end

function (f::NormL1Fused)(x::Tuple{AbstractArray, AbstractArray})
  _checkreal(f, x)
  x1, x2 = x
  size(x1) == size(x2) || throw(DimensionMismatch("the arrays of the pair have sizes $(size(x1)) and $(size(x2))"))
  T = float(promote_type(eltype(x1), eltype(x2)))
  s = zero(T)
  @inbounds for i in eachindex(x1, x2)
    s += _fused_pair_value(f, (x1[i], x2[i]))
  end
  isfinite(s) && _fused_levels_fit(T, f) && return s
  _recover_fused_arrays(f, x, T, s)
end

# argmin (1/2γ) ((u - x)² + (v - y)²) + λ1 (|u| + |v|) + λ2 |u - v|: fuse, then shrink
@inline function _prox_l1fused(x::T, y::T, λ1::T, λ2::T, γ::T, ::Val{C} = Val(true)) where {T<:AbstractFloat, C}
  if λ2 > zero(T)
    t = λ2 * γ
    if x > y + 2 * t
      u = x - t
      v = y + t
    elseif y > x + 2 * t
      u = x + t
      v = y - t
    else
      z = x + y
      u = v = C ? ifelse(isinf(z), x / 2 + y / 2, z / 2) : z / 2
    end
  else
    u = x
    v = y
  end
  t = λ1 * γ
  (shrink(u, t), shrink(v, t))
end

function prox(f::NormL1Fused, xy::Tuple{Real, Real}, γ::Real = 1)
  T = float(promote_type(typeof(xy[1]), typeof(xy[2])))
  _prox_l1fused(T(xy[1]), T(xy[2]), T(f.λ1), T(f.λ2), T(γ))
end

function prox!(y::Tuple{AbstractArray, AbstractArray}, f::NormL1Fused, x::Tuple{AbstractArray, AbstractArray}, γ::Real)
  _checkwrite(prox!, f, y, x)
  _prox_fused_arrays!(y, f, x, γ, Val(true))
end

@noinline function _prox_fused_arrays!(y, f, x, γ, ::Val{C}) where {C}
  T = float(promote_type(eltype(y[1]), eltype(y[2])))
  λ1, λ2, γ = T(f.λ1), T(f.λ2), T(γ)
  @inbounds for i in eachindex(x[1])
    y[1][i], y[2][i] = _prox_l1fused(T(x[1][i]), T(x[2][i]), λ1, λ2, γ, Val(C))
  end
  y
end

function value_and_prox!(y::Tuple{AbstractArray, AbstractArray}, f::NormL1Fused,
                         x::Tuple{AbstractArray, AbstractArray}, γ::Real)
  _checkwrite(value_and_prox!, f, y, x)
  alias = Base.mightalias(y[1], x[1]) || Base.mightalias(y[1], x[2]) ||
    Base.mightalias(y[2], x[1]) || Base.mightalias(y[2], x[2])
  alias && return f(prox!(y, f, x, γ))
  _prox_fused_arrays!(y, f, x, γ, Val(false))
  v = f(y)
  isfinite(v) ? v : _recover_fused_arrayprox!(y, f, x, γ, v)
end

@noinline function _recover_fused_arrayprox!(y, f, x, γ, ordinary)
  isfinite(f.λ1) && isfinite(f.λ2) && all(isfinite, x[1]) && all(isfinite, x[2]) || return ordinary
  f(prox!(y, f, x, γ))
end

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::NormL1Fused,
            y::Tuple{AbstractArray, AbstractArray}, x::Tuple{AbstractArray, AbstractArray}) =
  (_checkreal(f, x); _checkreal(f, y); size(y[1]) == size(x[1]) == size(y[2]) == size(x[2]) || throw(DimensionMismatch("the arrays of the pairs must have one size")))
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::NormL1Fused, y::AbstractVector, x::AbstractVector) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); Base.require_one_based_indexing(y, x); true)
prox(f::NormL1Fused, x::Tuple{AbstractArray, AbstractArray}, γ::Real = 1) =
  prox!((similar(x[1], float(eltype(x[1]))), similar(x[2], float(eltype(x[2])))), f, x, γ)

# a vector of K entries, all pairs fused: a fiber of K stacked matrices. Two
# entries take the closed form of a pair.

# the entries of x, and the indices p that sort them in increasing order, in the
# buffers of f, allocating nothing: an in-place quicksort for short vectors, and
# Base's sort with a scratch buffer for long ones, as for SLOPE
function _sortperm_fused!(f::NormL1Fused, x::AbstractVector, ::Type{T}) where {T}
  n = length(x)
  a = _buffer!(f.sums, T, n)
  p = _buffer!(f.perm, Int, n)
  @inbounds for i in 1:n
    a[i] = convert(T, x[i])
    p[i] = i
  end
  if n <= 4096
    sort!(view(p, 1:n); alg = QuickSort, lt = (i, j) -> a[i] < a[j])
  else
    sortperm!(view(p, 1:n), view(a, 1:n); initialized = true, scratch = _buffer!(f.sizes, Int, n))
  end
  p, a
end

@noinline function _recover_fused_vector(f, x, ::Type{T}, ordinary) where {T}
  all(isfinite, x) && isfinite(f.λ1) && isfinite(f.λ2) || return ordinary
  R = promote_type(T, float(typeof(f.λ1)), float(typeof(f.λ2)))
  p, a = _sortperm_fused!(f, x, R)
  n = length(x)
  k = _magnitude_exponent(_maximum_abs(x))
  scale = _power_factors(R, -k)
  s1 = s2 = c1 = c2 = zero(R)
  @inbounds for i in 1:n
    u = _scale_value(a[p[i]], scale)
    s1, c1 = _compensated_add(s1, c1, abs(u))
    if i < n
      gap = _scale_value(a[p[i + 1]], scale) - u
      s2, c2 = _compensated_add(s2, c2, (R(i) * R(n - i)) * gap)
    end
  end
  _fused_scaled_value(((f.λ1, s1 + c1), (f.λ2, s2 + c2)), k, T)
end

@inline function _fused_vector_value(f, x, v1, v2, ::Type{T}) where {T}
  w1 = iszero(f.λ1) ? zero(T) : convert(T, f.λ1) * v1
  w2 = iszero(f.λ2) ? zero(T) : convert(T, f.λ2) * v2
  v = w1 + w2
  (_fused_finite(v) & (iszero(f.λ2) | (v2 >= 0)) & _fused_levels_fit(T, f)) && return v
  _recover_fused_vector(f, x, T, v)
end

@inline function (f::NormL1Fused)(x::AbstractVector)
  _checkreal(f, x)
  Base.require_one_based_indexing(x)
  T = float(eltype(x))
  iszero(f.λ1) && iszero(f.λ2) && return zero(T)
  n = length(x)
  n == 2 && return @inbounds f((x[1], x[2]))
  p, a = _sortperm_fused!(f, x, T)
  v1, v2 = zero(T), zero(T)
  @inbounds for k in 1:n
    v1 += abs(a[p[k]])
    v2 += (2k - n - 1) * a[p[k]]
  end
  _fused_vector_value(f, x, v1, v2, T)
end

value_and_prox!(y::AbstractVector, f::NormL1Fused, x::AbstractVector, γ::Real) = _prox_fused!(y, f, x, γ, Val(true))
prox!(y::AbstractVector, f::NormL1Fused, x::AbstractVector, γ::Real) = (_prox_fused!(y, f, x, γ, Val(false)); y)

# the proximal operator of all pairs fused, and the value at the result if V
function _prox_fused!(y::AbstractVector, f::NormL1Fused, x::AbstractVector, γ::Real, ::Val{V}) where {V}
  _checkwrite(V ? value_and_prox! : prox!, f, y, x)
  T = float(eltype(y))
  λ1, λ2 = convert(T, f.λ1), convert(T, f.λ2)
  n = length(x)
  if iszero(λ1) && iszero(λ2)
    y === x || copyto!(y, x)
    return V ? f(y) : zero(T)
  end
  if n == 2
    @inbounds y[1], y[2] = _prox_l1fused(convert(T, x[1]), convert(T, x[2]), λ1, λ2, convert(T, γ))
    return V ? @inbounds(f((y[1], y[2]))) : zero(T)
  end
  p, a = _sortperm_fused!(f, x, T)          # x is read here, so y may be x
  w = _buffer!(f.vals, T, n)
  t = convert(T, γ) * λ2
  @inbounds for k in 1:n
    w[k] = a[p[k]] - t * (2k - n - 1)
  end
  # the buffers may be longer than x, which changes length from call to call
  wn = view(w, 1:n)
  _pava!(wn, wn, _buffer!(f.sums, T, n), _buffer!(f.sizes, Int, n), 1)
  # soft thresholding keeps the order, so the result is sorted as w
  c = convert(T, γ) * λ1
  v1, v2 = zero(T), zero(T)
  @inbounds for k in 1:n
    wk = shrink(w[k], c)
    y[p[k]] = wk
    v1 += abs(wk)
    v2 += (2k - n - 1) * wk
  end
  V ? _fused_vector_value(f, y, v1, v2, T) : zero(T)
end

# Project onto an L2 ball without subtracting two nearly equal coordinates.
# Only exceptional norms or factors need exponent arithmetic.
function _project_l2!(y, x, λ, α = 1, w = 1)
  isinf(w) && return copyto!(y, x)
  (iszero(λ) || iszero(w)) && return fill!(y, zero(eltype(y)))
  c = _projection_product(α, λ, w)
  n = _norm2(x)
  R = promote_type(float(eltype(x)), float(eltype(y)))
  bound = convert(R, c)
  t = min(one(n), bound / n)
  if isfinite(n) && (iszero(n) || (isfinite(bound) && t >= floatmin(typeof(t))))
    iszero(n) && return copyto!(y, x)
    y .= t .* x
    return y
  end
  _project_l2_scaled!(y, x, λ, α, w, c, n)
end

function _project_l2_scaled!(y, x, λ, α, w, c, n)
  m = _maximum_abs(x)
  (!isfinite(m) || !isfinite(λ) || !isfinite(α)) && return (y .= min(one(n), c / n) .* x)
  iszero(m) && return copyto!(y, x)
  # Scaling only the reduction avoids losing tiny original coordinates.
  a, e = frexp(m)
  q = sqrt(mapreduce(identity, +, Base.Broadcast.instantiate(Base.Broadcast.broadcasted(_projection_square, x, m)); init = zero(m)))
  r1, e1 = _projection_scale_parts(α)
  r2, e2 = frexp(float(λ))
  r3, e3 = frexp(float(w))
  r = ((r1 * r2) * r3) / (a * q)
  k = e1 + e2 + e3 - e
  fr, er = frexp(r)
  k += er
  (k > 1 || (k == 1 && fr >= 0.5)) && return copyto!(y, x)
  y .= _projection_scale.(x, fr, k)
end

_projection_square(v, m) = abs2(v / m)
_projection_scale(v::Real, r, k) = _scaled_product(promote(float(r), float(v))..., k)


# Linear group indices address the original matrix without reshape headers.
function _project_l2_group!(y, x, G, λ, α = 1, w = 1)
  isinf(w) && return _project_l2_group_scale!(y, x, G, 1)
  if iszero(λ) || iszero(w)
    @inbounds @simd for i in G
      y[i] = zero(eltype(y))
    end
    return y
  end
  c = _projection_product(α, λ, w)
  n = _projection_group_norm2(x, G)
  R = promote_type(float(eltype(x)), float(eltype(y)))
  bound = convert(R, c)
  t = min(one(n), bound / n)
  if isfinite(n) && (iszero(n) || (isfinite(bound) && t >= floatmin(typeof(t))))
    return _project_l2_group_scale!(y, x, G, iszero(n) ? one(n) : t)
  end
  _project_l2_group_scaled!(y, x, G, λ, α, w, c, n)
end

function _project_l2_group_scale!(y, x, G, t)
  @inbounds @simd for i in G
    y[i] = t * x[i]
  end
  y
end

function _project_l2_group_scaled!(y, x, G, λ, α, w, c, n)
  m = _projection_group_maxabs(x, G)
  (!isfinite(m) || !isfinite(λ) || !isfinite(α)) &&
    return _project_l2_group_scale!(y, x, G, min(one(n), c / n))
  iszero(m) && return _project_l2_group_scale!(y, x, G, 1)
  s = err = zero(m)
  @inbounds for i in G
    s, err = _compensated_add(s, err, _projection_square(x[i], m))
  end
  a, e = frexp(m)
  r1, e1 = _projection_scale_parts(α)
  r2, e2 = frexp(float(λ))
  r3, e3 = frexp(float(w))
  r = ((r1 * r2) * r3) / (a * sqrt(s + err))
  fr, er = frexp(r)
  k = e1 + e2 + e3 - e + er
  (k > 1 || (k == 1 && fr >= 0.5)) && return _project_l2_group_scale!(y, x, G, 1)
  @inbounds for i in G
    y[i] = _projection_scale(x[i], fr, k)
  end
  y
end


# Preserve the existing real-float reductions and their ordinary arithmetic.
@inline _projection_group_norm2(x::AbstractArray{<:AbstractFloat}, G) = _group_norm2(x, G)
@inline _projection_group_maxabs(x::AbstractArray{<:AbstractFloat}, G) = _maxabs(x, G)

function _projection_group_norm2(x::AbstractArray, G)
  T = float(eltype(x))
  s = zero(T)
  @inbounds @simd for i in G
    s += abs2(float(x[i]))
  end
  isfinite(s) && s >= floatmin(T) && return sqrt(s)
  if iszero(s)
    a = zero(T)
    @inbounds @simd for i in G
      a += abs(float(x[i]))
    end
    iszero(a) && return s
  end
  n = zero(T)
  @inbounds for i in G
    isnan(x[i]) && return T(NaN)
    n = hypot(n, abs(float(x[i])))
  end
  n
end

function _projection_group_maxabs(x::AbstractArray, G)
  m = zero(float(eltype(x)))
  @inbounds for i in G
    m = max(m, abs(float(x[i])))
  end
  m
end

@inline _projection_l1_scale(v, e) = ldexp(v, e)
function _projection_l1_scale(v::BigFloat, e)
  # Windows Int exponents otherwise select ldexp's rounded exp2 fallback.
  # MPFR's full exponent span fits Clong; wider shifts necessarily saturate.
  ldexp(v, Clong(clamp(e, typemin(Clong), typemax(Clong))))
end

# Julia 1.10's ldexp truncates some values below the smallest subnormal.
# A normal power-of-two scaling followed by multiplication rounds only once.
@inline function _projection_l1_ldexp(v::R, e) where {R}
  e < exponent(floatmin(R)) ? ldexp(v, e + precision(R)) * ldexp(one(R), -precision(R)) : ldexp(v, e)
end

_projection_l1_ldexp(v::BigFloat, e) = _projection_l1_scale(v, e)

# Products of finite BigFloat factors can exceed a machine exponent.
struct _L1ProjectionBigScale
  m::BigFloat
  e::BigInt
end
_projection_scale_parts(a::_L1ProjectionBigScale) = (a.m, a.e)

# Offsets at or beyond the radius cannot enter the support. Apply the exponent
# after subtraction: huge tied maxima must still have exactly zero offset.
@inline function _projection_l1_offset(v, m, r, scale, e, ::Type{R}) where {R}
  d = m - abs(convert(R, v))
  min(r, isfinite(scale) && !iszero(scale) ? d * scale : _projection_l1_scale(d, -e))
end

# Preserve integer offsets before rounding, including at typemin.
@inline _projection_l1_magnitude(v::Base.BitInteger) = v < 0 ? -unsigned(v) : unsigned(v)
@inline function _projection_l1_offset(v::Base.BitInteger, m::Unsigned, r, scale, e, ::Type{R}) where {R}
  d = convert(R, m - _projection_l1_magnitude(v))
  min(r, isfinite(scale) && !iszero(scale) ? d * scale : _projection_l1_scale(d, -e))
end
function _projection_l1_max(x, G, ::Type{R}) where {R}
  m = zero(R)
  @inbounds @simd for i in G
    m = max(m, abs(convert(R, x[i])))
  end
  m
end
function _projection_l1_max(x::AbstractArray{T}, G, ::Type{R}) where {T<:Base.BitInteger,R}
  m = zero(unsigned(T))
  @inbounds @simd for i in G
    m = max(m, _projection_l1_magnitude(x[i]))
  end
  m
end

_projection_l1_type(a) = typeof(first(_projection_scale_parts(a)))
_projection_l1_product(::Tuple{}, h, l, e, δ, ::Type{R}) where {R} = (h, l, e, δ)
function _projection_l1_product(fs::Tuple, h, l, e, δ, ::Type{R}) where {R}
  b = first(fs)
  if !isone(b)
    mb, eb = b isa Real ? frexp(convert(R, b)) : _projection_scale_parts(b)
    v = convert(R, mb)
    p = h * v
    residual = fma(h, v, -p)
    q = residual + l * v
    t = p + q
    lo = (p - t) + q
    hi, shift = frexp(t)
    if !iszero(residual) || !iszero(l)
      δ = nextfloat(δ + (1 + δ) * 16eps(R)^2)
    end
    # Non-floating coefficients may also round during their conversion.
    if b isa Real && !(b isa AbstractFloat) && convert(R, b) != b
      δ = nextfloat(δ + (1 + δ) * eps(R))
    end
    h, l, e = hi, _projection_l1_scale(lo, -shift), e + eb + shift
  end
  _projection_l1_product(Base.tail(fs), h, l, e, δ, R)
end

function _projection_l1_coefficient(α, λ, w, x, y)
  fs = (_projection_l1_factors(α)..., λ, w)
  R = promote_type(Float64, float(eltype(x)), float(eltype(y)), map(_projection_l1_type, fs)...)
  h, l, e, δ = _projection_l1_product(fs, R(0.5), zero(R), R === BigFloat ? BigInt(1) : 1, zero(R), R)
  error = iszero(δ) ? abs(l) : nextfloat(abs(l) + nextfloat(δ * h))
  lower = iszero(error) ? h : max(zero(R), prevfloat(h - error))
  a = R === BigFloat ? _L1ProjectionBigScale(h, e) : _ProjectionScale(h, e)
  a, error, lower
end

function _projection_l1_fast_buffer!(s, ::Type{Float32}, n)
  b = s.buf
  if !(b isa _L1ProjectionScratch)
    native = b isa Vector{Float32} ? b : Float32[]
    wide = b isa Vector{Float64} ? b : Float64[]
    b = _L1ProjectionScratch(native, wide)
    s.buf = b
  end
  length(b.native) < n && resize!(b.native, n)
  b.native
end
_projection_l1_fast_buffer!(s, ::Type{Float64}, n) = _projection_l1_careful_buffer!(s, Float64, n)
function _projection_l1_careful_buffer!(s, ::Type{R}, n) where {R}
  b = s.buf
  if R === Float64 && b isa _L1ProjectionScratch
    length(b.wide) < n && resize!(b.wide, n)
    return b.wide
  end
  _buffer!(s, R, n)
end

@inline function _projection_l1_stats(x, G)
  T = eltype(x)
  s = m = zero(T)
  @inbounds @simd for i in G
    v = abs(x[i])
    m = max(m, v)
    s += v
  end
  s, m
end
function _projection_l1_extrema(x, G, m)
  second = third = zero(m)
  count = 0
  @inbounds for i in G
    v = abs(x[i])
    if v == m
      count += 1
    else
      third = max(third, min(second, v))
      second = max(second, v)
    end
  end
  count == 1 ? (second, third, count) : count == 2 ? (m, second, count) : (m, m, count)
end
_projection_l1_fast!(y, x, G, scratch, λ, α, w) = false
function _projection_l1_fast!(y::AbstractArray{T}, x::AbstractArray{T}, G, scratch, r::T, α::Real, w) where {T<:Union{Float32,Float64}}
  (isone(α) && isone(w) && 32floatmin(T) <= r < floatmax(T) / 16) || return false
  _projection_l1_native!(y, x, G, scratch, r)
end
@inline function _projection_l1_native!(y::AbstractArray{T}, x::AbstractArray{T}, G, scratch, r::T) where {T<:Union{Float32,Float64}}
  n = length(G)
  s, m = _projection_l1_stats(x, G)
  (isfinite(s) && isfinite(m) && s < floatmax(T) / 16) || return false
  δ = (n + 8) * (eps(T) / 2)
  δ < T(0.125) || return false
  if s <= r * (1 - 2δ)
    @inbounds @simd for i in G
      y[i] = x[i]
    end
    return true
  end
  (s * (1 - δ) > r && m >= 32floatmin(T)) || return false
  if n <= 32 && m > r
    second, third, count = _projection_l1_extrema(x, G, m)
    d = m - second
    if count == 1 && prevfloat(d) >= r
      @inbounds @simd for i in G
        v = x[i]
        y[i] = copysign(abs(v) == m ? r : zero(T), v)
      end
      return true
    end
    # Sterbenz makes these offsets exact. Outward comparisons certify the
    # support; sums of at most three radius-sized terms cost <=4eps*r.
    if count <= 2 && second >= m / 2 && d < r
      t = (r + d) / 2
      if prevfloat(m - third) >= nextfloat(t)
        @inbounds @simd for i in G
          v = x[i]
          y[i] = copysign(abs(v) >= second ? t - (m - abs(v)) : zero(T), v)
        end
        return true
      end
    end
    if count <= 3 && third >= m / 2
      d₂, d₃ = m - second, m - third
      t = ((r + d₂) + d₃) / 3
      E = 4eps(T) * r
      if d₃ < t - E
        fourth = zero(T)
        above = 0
        @inbounds @simd for i in G
          v = abs(x[i])
          fourth = max(fourth, ifelse(v < third, v, zero(T)))
          above += v >= third
        end
        if above == 3 && prevfloat(m - fourth) >= t + E
          @inbounds @simd for i in G
            v = x[i]
            y[i] = copysign(abs(v) >= third ? t - (m - abs(v)) : zero(T), v)
          end
          return true
        end
      end
    end
  end
  θ, k, B = n <= 32 ? _projection_l1_michelot(x, G, r, m, s) :
    _projection_l1_threshold!(_projection_l1_fast_buffer!(scratch, T, n), x, G, r, m, s)
  budget = (n * eps(T)) * r
  # For Condat, B=4eps*m plus the bound on its accumulated sum residuals.
  # Ignoring outward inflation, m<=K*r, K=min(3n/8,3n/(4k)), suffices
  # with zero residual; equality can fall back after inflation. Michelot uses
  # gamma_k on its final sum. B encloses the selected equation's level error;
  # Condat separately checks discarded residuals against its largest tested
  # level. Michelot's last two equal counts give the same thresholded set,
  # hence the same sum and level; its discarded entries are <=that level.
  # The selected equation's mass residual is <=k*B.
  # If the level is too high, lowering it retains all k selected entries,
  # so its error is <=B; suppressing a small candidate costs another B.
  # If too low, wrongly retained entries can disappear: slope may fall to 1,
  # giving k*B, not B. At θ+max(k*B,2B) discarded entries vanish
  # and selected mass is <=r, so the true level is no higher. Thus coordinate
  # error is <=max(k*B,2B) plus subtraction rounding. B>=2u*m (u=eps/2),
  # so that rounding is <=B/2<=0.75budget. Both alternatives are <=3budget:
  # C=4, without a support
  # guarantee. Suppression removes discarded entries and only reduces mass;
  # the selected mass excess is <=k*B, plus subtraction rounding.
  (isfinite(θ) && θ >= 32floatmin(T) && B <= T(1.5) * budget && k * B <= 3budget) || return false
  @inbounds @simd for i in G
    v = x[i]
    q = abs(v) - θ
    y[i] = copysign(q <= B ? zero(T) : q, v)
  end
  true
end

_projection_l1_unit_groups!(y, x, f, α, γ) = false
function _projection_l1_unit_groups!(y::Array{T}, x::Array{T}, f::GroupNormLinf{T}, α::Real, γ) where {T<:Union{Float32,Float64}}
  r = f.λ0
  (isone(α) && 32floatmin(T) <= r < floatmax(T) / 16 && all(isone, f.w)) || return false
  a, error, lower = _projection_l1_coefficient(1, r, 1, x, y)
  fallback = nothing
  @inbounds for (k, G) in enumerate(f.groups)
    isempty(G) && continue
    if !_projection_l1_native!(y, x, G, f.scratch, r)
      if _project_l1_radius!(y, x, G, f.scratch, a, error, lower) === nothing
        fallback === nothing && (fallback = Int[])
        push!(fallback, k)
      end
    end
  end
  fallback === nothing ? true : _L1ProjectionFallback(fallback)
end

function _project_l1!(y, x, G, scratch, λ, α = 1, w = 1)
  isempty(G) && return y
  if isinf(w)
    @inbounds @simd for i in G
      y[i] = x[i]
    end
    return y
  end
  if iszero(λ) || iszero(w) || iszero(α)
    @inbounds @simd for i in G
      y[i] = zero(eltype(y))
    end
    return y
  end
  if isinf(λ) || isinf(α)
    @inbounds @simd for i in G
      y[i] = x[i]
    end
    return y
  end
  _projection_l1_fast!(y, x, G, scratch, λ, α, w) && return y
  a, error, lower = _projection_l1_coefficient(α, λ, w, x, y)
  _project_l1_radius!(y, x, G, scratch, a, error, lower)
end

function _projection_l1_inside(x, G, r::R, scale, e) where {R}
  s = zero(R)
  @inbounds @simd for i in G
    v = abs(convert(R, x[i]))
    s += isfinite(scale) && !iszero(scale) ? v * scale : _projection_l1_scale(v, -e)
  end
  δ = length(G) * eps(R)
  if δ < R(0.125)
    # A reduction's gamma_n bound, including conversion and subnormals.
    s * (1 + 2δ) + length(G) * nextfloat(zero(R)) <= r && return true
    s * (1 - 2δ) > r && return false
  end
  # An upper bound certifies the identity case, including exact sums on the
  # boundary. Subnormal scaled terms are rounded upward for this test only.
  s = err = zero(R)
  @inbounds for i in G
    v = abs(convert(R, x[i]))
    # Enclose integer conversion before the exact power-of-two scaling.
    if x[i] isa Base.BitInteger && v < _projection_l1_magnitude(x[i])
      v = nextfloat(v)
    end
    v = isfinite(scale) && !iszero(scale) ? v * scale : _projection_l1_scale(v, -e)
    if !isfinite(v)
      s = R(Inf)
      break
    end
    if v < floatmin(R) && !iszero(x[i])
      v = nextfloat(v)
    end
    t = s + v
    q = s >= v ? (s - t) + v : (v - t) + s
    q != 0 && (err = nextfloat(err + q))
    s = t
    s > r && break
  end
  s <= r && err <= r - s
end

function _project_l1_radius!(y, x, G, scratch, a, coefficient_error, lower)
  r0, e = _projection_scale_parts(a)
  R = promote_type(Float64, typeof(r0), float(eltype(y)))
  r = convert(R, r0)
  scale = _projection_l1_scale(one(R), -e)
  outscale = _projection_l1_scale(one(R), e)
  m = _projection_l1_max(x, G, R)
  # Let the caller retain Moreau's non-finite behaviour for this block.
  !isfinite(m) && return nothing
  if _projection_l1_inside(x, G, lower, scale, e)
    @inbounds @simd for i in G
      y[i] = x[i]
    end
    return y
  end
  # Condat on -d supplies a seed; compensated Newton/Michelot refines it.
  # The sums contain d<t<=r, so scaling by the radius bounds them by n.
  t = r
  if length(G) > 32
    f = let m = m
      v -> -_projection_l1_offset(v, m, r, scale, e, typeof(r))
    end
    t = min(r, -_simplex_threshold!(_projection_l1_careful_buffer!(scratch, R, length(G)), x, G, r, f))
  end
  count = -1
  for _ in 0:length(G)
    s = err = zero(R)
    k = 0
    last = zero(R)
    @inbounds for i in G
      d = _projection_l1_offset(x[i], m, r, scale, e, R)
      if d < t
        s, err = _compensated_add(s, err, d)
        k += 1
        last = max(last, d)
      end
    end
    k == 0 && break
    previous = t
    t = min(r, (r + (s + err)) / k)
    (k == count || (last <= t && t <= previous)) && break
    count = k
  end
  # Relevant offsets are <=r, with error <=u*r (Sterbenz is exact below
  # m/2). Neumaier's sum error and sum(d)<=k*t<=k*r give radius-relative
  # threshold error. Each inexact coefficient product retains its FMA residual:
  # after d factors its error is <=|low|+32d*eps(R)^2*r/(1-32d*eps(R)^2),
  # enclosing the outward-rounded recurrence above. Conversions add their own
  # bound. coefficient_error also gives the lower membership bound. For
  # floating factors and d*eps(R)<=1/128 it is <eps(R)*r, already budgeted
  # in 2n*eps(R)*r. The explicit guard also covers deeper chains/conversions.
  # Guarding spurious support and final rounding give C=4 plus half a T
  # subnormal in that regime; otherwise add 2*coefficient_error in original
  # units. Float32 uses Float64; n*eps(R)>1/4 permits the zero result.
  guard = 2length(G) * eps(R) * r + max(zero(R), coefficient_error - eps(R) * r)
  if length(G) * eps(R) > R(0.25)
    guard = r
  end
  @inbounds @simd for i in G
    v = convert(R, x[i])
    q = t - _projection_l1_offset(x[i], m, r, scale, e, R)
    p = isfinite(outscale) && !iszero(outscale) ? q * outscale : _projection_l1_ldexp(q, e)
    y[i] = copysign(q <= guard ? zero(R) : min(abs(v), p), v)
  end
  y
end
