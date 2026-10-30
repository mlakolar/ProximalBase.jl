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
When an intermediate overflows or underflows, or a weight is not representable
in the argument's precision although the weighted value is representable,
the value is computed in scaled units.
"""
struct NormL2{T<:Real} <: AbstractFunction
  λ::T

  NormL2{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ))
end
NormL2(λ::Real = 1) = NormL2{typeof(λ)}(λ)

isproximable(::Type{<:NormL2}) = true
isconvex(::Type{<:NormL2}) = true
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, ::NormL2, y::AbstractArray, x::AbstractArray) =
  (_checksize(y, x); true)

(f::NormL2)(x::AbstractArray) = _weighted_norm2(f.λ, x)

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
  isfinite(v) || !isfinite(λ) || !(eltype(x) <: Real) || !all(isfinite, x) ? v : _scaled_norm2_value(λ, x)
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
            ::CubeNormL2, y::AbstractArray, x::AbstractArray) = _checksize(y, x)

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
  T = real(float(eltype(x)))
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
@inline _cube_scale(v::Complex, m, e) = complex(_cube_scale(real(v), m, e), _cube_scale(imag(v), m, e))

function gradient!(g::AbstractArray, f::CubeNormL2, x::AbstractArray)
  _checkwrite(gradient!, f, g, x)
  iszero(f.λ) && return fill!(g, zero(eltype(g)))
  T = promote_type(real(float(eltype(g))), real(float(eltype(x))), float(typeof(f.λ)))
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
  T = promote_type(real(float(eltype(y))), real(float(eltype(x))), float(typeof(f.λ)), float(typeof(γ)))
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
When an intermediate overflows or underflows, or a weight is not representable
in the argument's precision although the weighted value is representable,
the value is computed in scaled units.
"""
struct NormLinf{T<:Real} <: AbstractFunction
  λ::T
  scratch::Scratch

  NormLinf{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ, Scratch()))
end
NormLinf(λ::Real = 1) = NormLinf{typeof(λ)}(λ)

isproximable(::Type{<:NormLinf}) = true
isconvex(::Type{<:NormLinf}) = true
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, ::NormLinf, y::AbstractArray, x::AbstractArray) =
  (_checksize(y, x); true)

(f::NormLinf)(x::AbstractArray) = _weighted_normlinf(f.λ, x)

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
When an intermediate overflows or underflows, or a weight is not representable
in the argument's precision although the weighted value is representable,
the value is computed in scaled units.
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
When an intermediate overflows or underflows, or a weight is not representable
in the argument's precision although the weighted value is representable,
the value is computed in scaled units.
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
  @eval function $F(λ0::Real, groups::AbstractVector{<:AbstractVector{<:Integer}}, w::AbstractVector{<:Real})
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

# Keep the ordinary group product; recover a finite group in scaled units.
@inline function _group_value(f, λ, w, n, x, G, ::Type{T}) where {T}
  if _level_fits(T, λ) && _level_fits(T, w)
    v = _positive_product(convert(T, λ), convert(T, w), n)
    isfinite(v) && return convert(T, v)
  end
  convert(T, _scaled_group_value(f, λ, w, x, G, n))
end

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
  _checkgroups(f, x)
  T = float(eltype(x))
  iszero(f.λ0) && return zero(T)
  v = zero(T)
  for k in eachindex(f.groups)
    iszero(f.w[k]) && continue
    v += _group_value(f, f.λ0, f.w[k], _group_norm(f, x, f.groups[k]), x, f.groups[k], T)
  end
  v
end

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::AbstractGroupPenalty, y::AbstractArray, x::AbstractArray) =
  (_checksize(y, x); _checkgroups(f, x); _check_symmetric_groups(f, y))

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
  _checksize(y, x)
  _checkgroups(f, x)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  if iszero(f.λ0)
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
    nrm = _shrink_group!(f, y, x, f.groups[k], convert(T, _positive_product(_entry(γ, k), f.λ0, f.w[k])), _group_writes(y, f.groups[k]))
    v += _group_value(f, f.λ0, f.w[k], nrm, y, f.groups[k], T)
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
  _checkgroups(f, g)
  _checkgroups(f, x)
  k = _group(f, i)
  k == 0 && return abs(g[i])
  G = f.groups[k]
  _allzero(x, G) ? _dual_norm(f, g, G) - f.λ0 * f.w[k] : abs(g[i])
end

# on an AtomIterate, k is a group
_kkt_violation(f::AbstractGroupPenalty, g::AbstractArray, x::AtomIterate, k::Integer) =
  (_checkgroups(f, g); _checksamegroups(f, x); _dual_norm(f, g, x.groups[k]) - f.λ0 * f.w[k])

# the coordinates of x are the groups of f
_checksamegroups(f::AbstractGroupPenalty, x::AtomIterate) = _samelayout(f, x) ||
  throw(DimensionMismatch("f and x must have the same groups."))
_kkt_violation(f::AbstractGroupPenalty, ::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  throw(ArgumentError("$(typeof(f)) groups linear indices; it does not apply to the coordinates of a SymmetricSparseIterate"))

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
  _samelayout(f, x) || return invoke(f, Tuple{AbstractArray}, x)
  T = float(eltype(x))
  iszero(f.λ0) && return zero(T)
  v = zero(T)
  for a in 1:x.nactive
    k = x.active[a]
    iszero(f.w[k]) && continue
    v += _group_value(f, f.λ0, f.w[k], _group_norm(f, x.storage, f.groups[k]), x.storage, f.groups[k], T)
  end
  v
end

value_and_prox!(y::AtomIterate, f::AbstractGroupPenalty, x::AtomIterate, γ::Real) =
  (_checkwrite(value_and_prox!, f, y, x); _value_and_prox_groups!(y, f, x, γ))

function _value_and_prox_groups!(y::AtomIterate, f::HomogeneousGroupNorm, x::AtomIterate, γ::Union{Real, AbstractVector})
  _samelayout(f, x) && _samelayout(f, y) ||
    return invoke(_value_and_prox_groups!, Tuple{AbstractArray, HomogeneousGroupNorm, AbstractArray, Union{Real, AbstractVector}}, y, f, x, γ)
  _checkgroupsteps(f, γ)
  T = float(eltype(y))
  copyto!(y, x)
  iszero(f.λ0) && return zero(T)
  v = zero(T)
  for a in 1:y.nactive
    k = y.active[a]
    iszero(f.w[k]) && continue
    nrm = _shrink_group!(f, y.storage, y.storage, f.groups[k], convert(T, _positive_product(_entry(γ, k), f.λ0, f.w[k])))
    v += _group_value(f, f.λ0, f.w[k], nrm, y.storage, f.groups[k], T)
  end
  v
end

# Validate the layout and group before a wrapper changes any entries.
function _check_group_coordinate(f::AbstractGroupPenalty, x::AtomIterate, k::Integer)
  _checksamegroups(f, x)
  checkbounds(x.groups, k)
  nothing
end

# group k of x shrunk in place; the group becomes active if it is nonzero, as after
# writing to it through atom(x, k)
function prox_coordinate!(x::AtomIterate, f::HomogeneousGroupNorm, k::Integer, γ::Real)
  _check_group_coordinate(f, x, k)
  T = float(eltype(x))
  if iszero(f.λ0) || iszero(f.w[k])
    _allzero(x.storage, x.groups[k]) || activate!(x, k)
    return atom(x, k)
  end
  nrm = _shrink_group!(f, x.storage, x.storage, x.groups[k], convert(T, _positive_product(γ, f.λ0, f.w[k])))
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

(f::NormL1Fused)(xy::Tuple{Real, Real}) = _value_at(
  (iszero(f.λ1) ? zero(float(xy[1])) : f.λ1 * (abs(xy[1]) + abs(xy[2]))) +
  (iszero(f.λ2) ? zero(float(xy[1])) : f.λ2 * abs(xy[1] - xy[2])), xy)

function (f::NormL1Fused)(x::Tuple{AbstractArray, AbstractArray})
  x1, x2 = x
  size(x1) == size(x2) || throw(DimensionMismatch("the arrays of the pair have sizes $(size(x1)) and $(size(x2))"))
  T = float(promote_type(eltype(x1), eltype(x2)))
  s = zero(T)
  @inbounds for i in eachindex(x1, x2)
    s += f((x1[i], x2[i]))
  end
  s
end

# argmin (1/2γ) ((u - x)² + (v - y)²) + λ1 (|u| + |v|) + λ2 |u - v|: fuse, then shrink
function _prox_l1fused(x::T, y::T, λ1::T, λ2::T, γ::T) where {T<:AbstractFloat}
  if λ2 > zero(T)
    t = λ2 * γ
    if x > y + 2 * t
      u = x - t
      v = y + t
    elseif y > x + 2 * t
      u = x + t
      v = y - t
    else
      u = v = (x + y) / 2
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
  T = float(promote_type(eltype(y[1]), eltype(y[2])))
  λ1, λ2, γ = T(f.λ1), T(f.λ2), T(γ)
  @inbounds for i in eachindex(x[1])
    y[1][i], y[2][i] = _prox_l1fused(T(x[1][i]), T(x[2][i]), λ1, λ2, γ)
  end
  y
end
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, ::NormL1Fused,
            y::Tuple{AbstractArray, AbstractArray}, x::Tuple{AbstractArray, AbstractArray}) =
  size(y[1]) == size(x[1]) == size(y[2]) == size(x[2]) || throw(DimensionMismatch("the arrays of the pairs must have one size"))
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, ::NormL1Fused, y::AbstractVector, x::AbstractVector) =
  (_checksize(y, x); Base.require_one_based_indexing(y, x); true)
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

function (f::NormL1Fused)(x::AbstractVector)
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
  (iszero(f.λ1) ? zero(T) : convert(T, f.λ1) * v1) +
  (iszero(f.λ2) ? zero(T) : convert(T, f.λ2) * v2)
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
    return V ? zero(f(y)) : zero(T)
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
  V ? (iszero(λ1) ? zero(T) : λ1 * v1) + (iszero(λ2) ? zero(T) : λ2 * v2) : zero(T)
end
