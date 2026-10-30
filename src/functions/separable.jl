####################################
#
#  Separable functions f(x) = Σᵢ fᵢ(xᵢ)
#
####################################

"""
    SeparableFunction

The supertype of the separable functions `f(x) = Σᵢ fᵢ(xᵢ)`, and the way to add
one of your own. A subtype implements, for the entry with linear index `i`,

- [`prox_entry`](@ref)`(f, v, i, γ)`: the proximal operator of `γfᵢ` at the
  scalar `v`;
- [`value_entry`](@ref)`(f, v, i)`: the value `fᵢ(v)`.

From these it gets the value `f(x)`, [`prox!`](@ref) and
[`value_and_prox!`](@ref) on any array, and [`prox_coordinate!`](@ref) on the
three iterates. Declare [`iszeropreserving`](@ref) when `fᵢ(0) = 0` and the
proximal operator maps 0 to 0; then the value and the proximal operator on a
sparse iterate touch only its stored entries. `isseparable` and `isproximable`
are true; declare `isconvex` when it holds.

# Example

```julia
struct WeightedSquare <: SeparableFunction            # Σᵢ wᵢ xᵢ² / 2
    w::Vector{Float64}
end
ProximalBase.value_entry(f::WeightedSquare, v, i) = f.w[i] * v^2 / 2
ProximalBase.prox_entry(f::WeightedSquare, v, i, γ) = v / (1 + γ * f.w[i])
ProximalBase.iszeropreserving(::WeightedSquare) = true
ProximalBase.isconvex(::Type{WeightedSquare}) = true
function ProximalBase.check_parameters(f::WeightedSquare, x)
  length(f.w) == length(x) || throw(DimensionMismatch("a parameter array has length \$(length(f.w)) and the argument has length \$(length(x))"))
  nothing
end
```

The generic kernels call `check_parameters` before writing any output or
updating a coordinate.
"""
abstract type SeparableFunction <: AbstractFunction end

isseparable(::Type{<:SeparableFunction}) = true
isproximable(::Type{<:SeparableFunction}) = true

# Broadcast parameters in the argument's shape to preserve linear pairing.
_reshapeparam(p, x) = p
_reshapeparam(p::AbstractArray, x) = reshape(p, size(x))
# The wrapper avoids allocating an Array header on CPU broadcasts.
_reshapeparam(p::Array, x) = Base.ReshapedArray(p, size(x), ())

"""
    value_entry(f, v, i)

For a [`SeparableFunction`](@ref) `f(x) = Σᵢ fᵢ(xᵢ)`, the value `fᵢ(v)` of the
term of entry `i` at the scalar `v`. `i` is a linear index of the array, as in
[`prox_entry`](@ref).
"""
function value_entry end

(f::SeparableFunction)(x::AbstractArray) = _value_sep(f, x)
prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::Real) = _prox_sep!(y, f, x, γ)
value_and_prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::Real) = _value_and_prox_sep!(y, f, x, γ)

# The value, entry by entry over the dense array. A weighted family sums its
# ordinary products, which vectorize; a NaN sum (a NaN entry, or a 0⋅Inf
# product of a zero or infinite weight) is summed again with value_entry.
_value_entry_ordinary(f, v, i) = value_entry(f, v, i)
_has_ordinary_value(::Type) = false

@inline function _sum_entries(g, f, x::AbstractArray, ::Type{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    s += convert(T, g(f, x[i], i))
  end
  s
end

function _value_dense(f, x::AbstractArray)
  T = _valtype(x)
  _has_ordinary_value(typeof(f)) || return _sum_entries(value_entry, f, x, T)
  s = _sum_entries(_value_entry_ordinary, f, x, T)
  isnan(s) ? _sum_entries(value_entry, f, x, T) : s
end

_value_sep(f, x::AbstractArray) = (_checkparams(f, x); _value_dense(f, x))

function _value_sep(f, x::SparseIterate)
  _checkparams(f, x)
  iszeropreserving(f) || return _value_dense(f, x)
  T = _valtype(x)
  s = zero(T)
  @inbounds for k in 1:x.nnz
    s += convert(T, value_entry(f, x.nzval[k], x.nzval2ind[k]))
  end
  s
end

# an off-diagonal coordinate is the entries (r, c) and (c, r)
function _value_sep(f, x::SymmetricSparseIterate)
  _checkparams(f, x)
  iszeropreserving(f) || return _value_dense(f, x)
  T = _valtype(x)
  p = x.p
  s = zero(T)
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(p, x.nzval2ind[k])
    v = x.nzval[k]
    s += convert(T, value_entry(f, v, (c - 1) * p + r))
    r == c || (s += convert(T, value_entry(f, v, (r - 1) * p + c)))
  end
  s
end

function _value_sep(f, x::AtomIterate)
  _checkparams(f, x)
  iszeropreserving(f) || return _value_dense(f, x)
  T = _valtype(x)
  s = zero(T)
  @inbounds for a in 1:x.nactive, i in x.groups[x.active[a]]
    s += convert(T, value_entry(f, x.storage[i], i))
  end
  s
end

# the proximal operator, entry by entry; y may be x
_prox_sep!(y::AbstractArray, f, x::AbstractArray, γ::Union{Real, AbstractArray}) = _prox_sep_dense!(y, f, x, γ)

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::SeparableFunction, y::AbstractArray, x::AbstractArray) =
  (_checksize(y, x); _checkparams(f, x); true)

function _prox_sep_dense!(y::AbstractArray, f, x::AbstractArray, γ)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  T = eltype(y)
  @inbounds for i in eachindex(IndexLinear(), y, x)
    y[i] = convert(T, prox_entry(f, x[i], i, _entry(γ, i)))
  end
  y
end

# Only same-precision CPU arrays opt in; lazy steps keep their existing loop.
_certified_weight_steps(y, f, x, γ) = false

function _prox_weight_steps!(y, f, x, γ)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  _certified_weight_steps(y, f, x, γ) ||
    return invoke(_prox_sep_dense!, Tuple{AbstractArray,Any,AbstractArray,Any}, y, f, x, γ)
  @inbounds @simd for i in eachindex(IndexLinear(), y, x)
    y[i] = _prox_entry_ordinary(f, x[i], i, γ[i])
  end
  y
end

# on sparse iterates, only the stored entries when f preserves zeros; as for a
# dense array, an entry stored in y stays stored, possibly as a zero
function _prox_sep!(y::SparseIterate, f, x::SparseIterate, γ::Union{Real, AbstractArray})
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  iszeropreserving(f) || return _prox_sep_dense!(y, f, x, γ)
  T = eltype(y)
  if y !== x
    @inbounds for k in 1:y.nnz
      y.nzval[k] = zero(T)
    end
  end
  @inbounds for k in 1:x.nnz
    i = x.nzval2ind[k]
    _setstored!(y, convert(T, prox_entry(f, x.nzval[k], i, _entry(γ, i))), i)
  end
  y
end

# each coordinate (an entry of the lower triangle) once, and from a
# SymmetricSparseIterate only its stored coordinates when f preserves zeros
function _prox_sep!(y::SymmetricSparseIterate, f, x::AbstractMatrix, γ::Union{Real, AbstractArray})
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  T = eltype(y)
  p = y.p
  if x isa SymmetricSparseIterate && iszeropreserving(f)
    if y !== x                       # as for a SparseIterate, y keeps its stored entries
      @inbounds for k in 1:y.nnz
        y.nzval[k] = zero(T)
      end
    end
    @inbounds for k in 1:x.nnz
      i = x.nzval2ind[k]
      r, c = tril_cartesian(p, i)
      j = (c - 1) * p + r
      _setstored!(y, convert(T, prox_entry(f, x.nzval[k], j, _entry(γ, j))), i)
    end
    return y
  end
  @inbounds for c in 1:p, r in c:p
    j = (c - 1) * p + r
    _setstored!(y, convert(T, prox_entry(f, x[r, c], j, _entry(γ, j))), tril_index(p, r, c))
  end
  y
end

# the proximal operator, then the value at the result: two vectorized passes are
# several times faster than one that stores and sums, which does not vectorize;
# an indicator function is 0 at its projection
_value_and_prox_sep!(y::AbstractArray, f, x::AbstractArray, γ::Union{Real, AbstractArray}) =
  _isindicator(f) ? (_prox_sep!(y, f, x, γ); zero(_valtype(y))) : _value_sep(f, _prox_sep!(y, f, x, γ))

_restrict_entries(w::Union{Nothing, Real}, ::AbstractVector{<:Integer}) = w
_restrict_entries(w::AbstractArray, S::AbstractVector{<:Integer}) = w[S]

####################################
#  ℓ1 norm
####################################

"""
    NormL1(λ0 = 1)
    NormL1(λ0, w)

The weighted ℓ1 norm `f(x) = λ0 Σᵢ wᵢ |xᵢ|`, with all `wᵢ = 1` when `w` is
omitted. Its proximal operator is entry-wise soft thresholding
([`shrink`](@ref)). Separable, so [`prox_coordinate!`](@ref) updates any
coordinate, and on a sparse iterate the value and the proximal operator touch
only the stored entries. On a [`SymmetricSparseIterate`](@ref) the weights are a
`p × p` matrix (or anything indexed by the linear indices of one), as for a
dense matrix.
When an intermediate sum overflows, or a weight is not representable in the
argument's precision although the weighted value is representable, the value
is computed in scaled units. An infinite weight fixes the entry at
zero at every level (the value is `Inf` away from zero), and a zero weight
leaves it unpenalized at every level.
"""
struct NormL1{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ0::T
  w::W

  function NormL1{T, W}(λ0::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    new{T, W}(λ0, w)
  end
end
NormL1(λ0::Real = 1, w::Union{Nothing, AbstractArray} = nothing) = NormL1{typeof(λ0), typeof(w)}(λ0, w)

isconvex(::Type{<:NormL1}) = true
iszeropreserving(::NormL1) = true
_checkparams(f::NormL1, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::NormL1, v::Real, i::Integer)
  a = abs(float(v))
  w = _entry(f.w, i)
  p = _positive_product(f.λ0, w, a)
  ifelse(isinf(w) & !iszero(a), oftype(p, Inf) * a, p)
end
Base.@propagate_inbounds prox_entry(f::NormL1, v::T, i::Integer, γ::Real) where {T<:Real} =
  shrink(v, convert(float(T), _weight_product(γ, f.λ0, _entry(f.w, i))))

kkt_violation(f::NormL1, g::AbstractArray, i::Integer) = abs(g[i]) - _weight_coefficient(f.λ0, _entry(f.w, i))
restrict(f::NormL1, S::AbstractVector{<:Integer}) = NormL1(f.λ0, _restrict_entries(f.w, S))

# without weights, Base's vectorized sum (a sum with init takes a slow sequential path)
function (f::NormL1{<:Real, Nothing})(x::DenseArray{<:AbstractFloat})
  T = float(eltype(x))
  iszero(f.λ0) && return zero(T)
  _level_fits(T, f.λ0) || return _converted_l1_value(f.λ0, x)
  v = convert(T, f.λ0) * sum(abs, x)
  isfinite(v) || !isfinite(f.λ0) || !all(isfinite, x) ? v : _value_dense(f, x)
end

# A level outside the argument's precision is applied before final conversion.
@noinline function _converted_l1_value(λ, x)
  T = float(eltype(x))
  R = promote_type(T, float(typeof(λ)))
  m = _maximum_abs(x)
  isfinite(m) || return convert(T, convert(R, λ) * sum(abs, x))
  e = _magnitude_exponent(m)
  s = _scaled_value_sum(x, _power_factors(R, -e), abs)
  convert(T, _scaled_product(convert(R, λ), s, e))
end

####################################
#  ℓ1 norm on the nonnegative orthant
####################################

"""
    NormL1Nonneg(λ0 = 1)
    NormL1Nonneg(λ0, w)

The weighted ℓ1 norm restricted to the nonnegative orthant:
`f(x) = λ0 Σᵢ wᵢ xᵢ` for `x ≥ 0`, and `Inf` otherwise, the penalty of the
nonnegative lasso. Its proximal operator is `max(x - γλ0w, 0)`.
"""
struct NormL1Nonneg{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ0::T
  w::W

  function NormL1Nonneg{T, W}(λ0::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    new{T, W}(λ0, w)
  end
end
NormL1Nonneg(λ0::Real = 1, w::Union{Nothing, AbstractArray} = nothing) = NormL1Nonneg{typeof(λ0), typeof(w)}(λ0, w)

isconvex(::Type{<:NormL1Nonneg}) = true
iszeropreserving(::NormL1Nonneg) = true
_checkparams(f::NormL1Nonneg, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::NormL1Nonneg, v::T, i::Integer) where {T<:Real}
  S = float(T)
  if v >= -_tol(S)
    a = max(S(v), zero(S))
    w = _entry(f.w, i)
    p = _positive_product(f.λ0, w, a)
    ifelse(isinf(w) & !iszero(a), oftype(p, Inf) * a, p)
  else
    S(Inf)
  end
end
Base.@propagate_inbounds prox_entry(f::NormL1Nonneg, v::T, i::Integer, γ::Real) where {T<:Real} =
  max(v - convert(float(T), _weight_product(γ, f.λ0, _entry(f.w, i))), zero(float(T)))

# moving up from zero decreases the objective when -gᵢ exceeds λ0 wᵢ
kkt_violation(f::NormL1Nonneg, g::AbstractArray, i::Integer) = -g[i] - _weight_coefficient(f.λ0, _entry(f.w, i))
restrict(f::NormL1Nonneg, S::AbstractVector{<:Integer}) = NormL1Nonneg(f.λ0, _restrict_entries(f.w, S))

@inline _l1_value(::NormL1, v, a) = a * abs(v)
@inline _l1_value(::NormL1Nonneg, v::T, a) where {T} =
  ifelse(v >= -_tol(T), a * max(v, zero(T)), T(Inf))

@inline function _l1_value_sum(f, x::Array{T}, allow_zero::Bool) where {T<:Union{Float32,Float64}}
  s = zero(T)
  ordinary = true
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    w = f.w[i]
    a = f.λ0 * w
    # Test the factors: a rounded zero may conceal recoverable underflow.
    z = allow_zero & (iszero(f.λ0) | iszero(w))
    s += _l1_value(f, x[i], a)
    ordinary &= z | (floatmin(T) <= a <= floatmax(T))
  end
  s, ordinary
end

# Preserve the original ordinary loop; retry zeros with SIMD before resorting
# to compensation. Every pass checks intermediates, not just the final sum.
function _value_dense(
    f::Union{NormL1{T,<:Array{T}}, NormL1Nonneg{T,<:Array{T}}},
    x::Array{T}) where {T<:Union{Float32,Float64}}
  s, ordinary = _l1_value_sum(f, x, false)
  ordinary && return s
  s, ordinary = _l1_value_sum(f, x, true)
  # 0*Inf/NaN needs the entrywise zero-factor and domain priorities.
  ordinary && !isnan(s) ? s : invoke(_value_dense, Tuple{Any,AbstractArray}, f, x)
end

####################################
#  elastic net
####################################

"""
    ElasticNet(λ1, λ2)
    ElasticNet(λ1, λ2, w)

The elastic-net penalty `f(x) = λ1 Σᵢ wᵢ |xᵢ| + (λ2/2) ‖x‖²`, with all `wᵢ = 1`
when `w` is omitted. Its proximal operator is
`shrink(x, γλ1w) / (1 + γλ2)`.
When an intermediate overflows or underflows, or a weight is not representable
in the argument's precision although the weighted value is representable,
the value is computed in scaled units.
"""
struct ElasticNet{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ1::T
  λ2::T
  w::W

  function ElasticNet{T, W}(λ1::T, λ2::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    _checknonneg("λ1", λ1)
    _checknonneg("λ2", λ2)
    _checknonneg("the weights", w)
    new{T, W}(λ1, λ2, w)
  end
end
function ElasticNet(λ1::Real, λ2::Real, w::Union{Nothing, AbstractArray} = nothing)
  λ1, λ2 = promote(λ1, λ2)
  ElasticNet{typeof(λ1), typeof(w)}(λ1, λ2, w)
end

isconvex(::Type{<:ElasticNet}) = true
iszeropreserving(::ElasticNet) = true
_checkparams(f::ElasticNet, x) = (_checkparam(f.w, x); true)

# Check every intermediate; exact zero factors do not conceal underflow.
@inline _normal_or_zero(v, z) = (z & iszero(v)) | ((floatmin(typeof(float(v))) <= v) & (v <= floatmax(typeof(float(v)))))
Base.@propagate_inbounds function _elastic_ordinary(f::ElasticNet, v::Real, i::Integer)
  v isa Integer && abs(v) < 0 && return false
  a = abs(float(v))
  w = _entry(f.w, i)
  c = f.λ1 * w
  q = abs2(float(v))
  _normal_or_zero(c, iszero(f.λ1) | iszero(w)) &
  _normal_or_zero(c * a, iszero(f.λ1) | iszero(w) | iszero(a)) &
  _normal_or_zero(q, iszero(a)) &
  _normal_or_zero(f.λ2 * q, iszero(f.λ2) | iszero(a))
end

Base.@propagate_inbounds function value_entry(f::ElasticNet, v::Real, i::Integer)
  if _elastic_ordinary(f, v, i)
    return _elastic_base_value_entry(f, v, i)
  end
  _elastic_exceptional(f, v, i)
end

# The original entry expression also preserves the retry's SIMD association.
Base.@propagate_inbounds function _elastic_base_value_entry(f::ElasticNet, v::Real, i::Integer)
  a = abs(v)
  _weighted_value(_weight_coefficient(f.λ1, _entry(f.w, i)) * a, a) + _weighted_value(f.λ2 * abs2(float(v)) / 2, abs2(float(v)))
end

# Recover the linear coefficient and the square before either loses range.
@noinline function _elastic_exceptional(f::ElasticNet, v::Real, i::Integer)
  a = abs(float(v))
  w = _entry(f.w, i)
  R = promote_type(typeof(a), typeof(f.λ1), typeof(f.λ2), typeof(w))
  isnan(a) && return convert(R, a)
  p = _positive_product(f.λ1, w, a)
  linear = ifelse(isinf(w) & !iszero(a), oftype(p, Inf) * a, p)
  quadratic = iszero(f.λ2) || iszero(a) ? zero(R) : _scaled_product(convert(R, f.λ2), convert(R, a), convert(R, a), -1)
  convert(R, linear) + quadratic
end

# Scalar levels and a bound on hidden underflow suffice after a finite sum.
@inline function _elastic_level_safe(::Type{T}, λ) where {T}
  (_normal_or_zero(float(λ), iszero(λ)) || !isfinite(λ)) && _level_fits(T, λ)
end

function _value_dense(f::ElasticNet, x::AbstractArray)
  T = float(eltype(x))
  eltype(x) <: Integer && return _sum_entries(value_entry, f, x, T)
  s = _sum_entries(_value_entry_ordinary, f, x, T)
  if isfinite(s) && _elastic_level_safe(T, f.λ1) && _elastic_level_safe(T, f.λ2)
    bound = _positive_product(float(length(x)), max(float(f.λ2), one(T)), floatmin(T) / eps(T))
    s >= bound && return s
  end
  _elastic_exact_sum(f, x, T)
end

# Recovery can change vectorization even when each entry is unchanged. In
# that case, retain the original entry loop and its reduction order.
@noinline function _elastic_exact_sum(f, x, ::Type{T}) where {T}
  s = zero(T)
  unchanged = true
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    v = value_entry(f, x[i], i)
    s += convert(T, v)
    unchanged &= isequal(v, _elastic_base_value_entry(f, x[i], i))
  end
  unchanged ? _sum_entries(_elastic_base_value_entry, f, x, T) : s
end
Base.@propagate_inbounds _value_entry_ordinary(f::ElasticNet, v::Real, i::Integer) = f.λ1 * _entry(f.w, i) * abs(v) + f.λ2 * abs2(v) / 2
_has_ordinary_value(::Type{<:ElasticNet}) = true
@inline _elastic_prox_entry(v, c, q) = shrink(v, c) / (1 + q)

Base.@propagate_inbounds function prox_entry(f::ElasticNet, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  _elastic_prox_entry(v, convert(S, _weight_coefficient(γ, f.λ1, _entry(f.w, i))), convert(S, γ * f.λ2))
end

_certified_weight_steps(y::Array{T}, f::ElasticNet{T, W}, x::Array{T}, γ::Array{T}) where {T<:Union{Float32,Float64}, W<:Union{Nothing,Array{T}}} =
  _certified_weight_scales(γ, f.λ1)

_prox_sep_dense!(y::Array{T}, f::ElasticNet{T, W}, x::Array{T}, γ::Array{T}) where {T<:Union{Float32,Float64}, W<:Union{Nothing,Array{T}}} =
  _prox_weight_steps!(y, f, x, γ)

Base.@propagate_inbounds function _prox_entry_ordinary(f::ElasticNet, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  _elastic_prox_entry(v, convert(S, (γ * f.λ1) * _entry(f.w, i)), convert(S, γ * f.λ2))
end

kkt_violation(f::ElasticNet, g::AbstractArray, i::Integer) = abs(g[i]) - _weight_coefficient(f.λ1, _entry(f.w, i))
restrict(f::ElasticNet, S::AbstractVector{<:Integer}) = ElasticNet(f.λ1, f.λ2, _restrict_entries(f.w, S))

####################################
#  box
####################################

"""
    IndBox(l, u)

The indicator of the box `{x : l ≤ x ≤ u}`: `0` inside and `Inf` outside. The
bounds are scalars or arrays read in linear order, and may be infinite. Array
bounds must have the length of `x`, and two array bounds must have equal length.
They satisfy `l ≤ u`, `l < Inf` and `u > -Inf` coordinatewise: equal finite bounds
are allowed, but equal infinite bounds are rejected because the coordinate is
empty. Its proximal operator clamps each coordinate to its corresponding bounds.
Array bounds are stored without copying; construct a new `IndBox` after changing them in
place, since whether the box contains zero is computed at construction.
[`restrict`](@ref) of the box itself requires `0 ∈ [lᵢ, uᵢ]` for every
coordinate outside `S`, and throws for scalar bounds that exclude zero.
"""
struct IndBox{L<:Union{Real, AbstractArray}, U<:Union{Real, AbstractArray}} <: SeparableFunction
  l::L
  u::U
  zp::Bool          # l ≤ 0 ≤ u, so that zero is feasible and its own projection

  function IndBox{L, U}(l::L, u::U) where {L<:Union{Real, AbstractArray}, U<:Union{Real, AbstractArray}}
    l isa AbstractArray && u isa AbstractArray && _checkparam(l, u)
    lv = l isa AbstractArray ? vec(l) : l
    uv = u isa AbstractArray ? vec(u) : u
    all(lv .<= uv) || throw(ArgumentError("the lower bounds must not exceed the upper bounds"))
    any((lv .== Inf) .| (uv .== -Inf)) && throw(ArgumentError("the lower bounds must be below Inf and the upper bounds above -Inf"))
    new{L, U}(l, u, all(l .<= 0) && all(u .>= 0))
  end
end
IndBox(l::Union{Real, AbstractArray}, u::Union{Real, AbstractArray}) = IndBox{typeof(l), typeof(u)}(l, u)

"""
    IndBallLinf(r)

The indicator of the ℓ∞ ball `{x : maxᵢ |xᵢ| ≤ r}`, the box [`IndBox`](@ref)`(-r, r)`.
"""
IndBallLinf(r::Real) = (_checknonneg("r", r); IndBox(-r, r))

isconvex(::Type{<:IndBox}) = true
iszeropreserving(f::IndBox) = f.zp
_checkparams(f::IndBox, x) = (_checkparam(f.l, x); _checkparam(f.u, x); true)

_isindicator(::Type{<:IndBox}) = true

# the limits of l ≤ v ≤ u within the tolerance t·max(floor, |bound|). A finite
# bound whose inflated limit overflows is clamped to the finite range instead:
# every finite entry lies within its tolerance, and an infinite one does not.
# Branch free, so that the loops below vectorize.
@inline function _box_limits(l, u, t, floor = one(t))
  lo, hi = l - t * max(floor, abs(l)), u + t * max(floor, abs(u))
  ifelse(isinf(l), lo, max(lo, -floatmax(lo))), ifelse(isinf(u), hi, min(hi, floatmax(hi)))
end
@inline function _inbox(v, l, u, t, floor = one(t))
  lo, hi = _box_limits(l, u, t, floor)
  (v >= lo) & (v <= hi)
end

# feasibility with a branch-free loop that vectorizes, then the value; scalar
# limits are computed once
function _value_dense(f::IndBox, x::AbstractArray)
  T = float(eltype(x))
  t = _tol(T)
  ok = true
  if f.l isa AbstractArray || f.u isa AbstractArray
    @inbounds @simd for i in eachindex(IndexLinear(), x)
      ok &= _inbox(x[i], _entry(f.l, i), _entry(f.u, i), t)
    end
  else
    lo, hi = _box_limits(f.l, f.u, t)
    @inbounds @simd for i in eachindex(IndexLinear(), x)
      v = x[i]
      ok &= (v >= lo) & (v <= hi)
    end
  end
  ok ? zero(T) : T(Inf)
end

Base.@propagate_inbounds function value_entry(f::IndBox, v::T, i::Integer) where {T<:Real}
  S = float(T)
  _inbox(v, _entry(f.l, i), _entry(f.u, i), _tol(S)) ? zero(S) : S(Inf)
end
# the bounds in the type of the result: a clamp between integer bounds compares
# a float with an integer exactly, which does not vectorize
Base.@propagate_inbounds function prox_entry(f::IndBox, v::T, i::Integer, ::Real) where {T<:Real}
  S = float(T)
  clamp(convert(S, v), convert(S, _entry(f.l, i)), convert(S, _entry(f.u, i)))
end

# from zero, the objective decreases upward at rate -gᵢ if u > 0, and downward
# at rate gᵢ if l < 0; zero must be feasible
function kkt_violation(f::IndBox, g::AbstractArray, i::Integer)
  T = float(eltype(g))
  up = _entry(f.u, i) > 0 ? -g[i] : T(-Inf)
  down = _entry(f.l, i) < 0 ? g[i] : T(-Inf)
  max(up, down)
end
# an excluded coordinate whose bounds exclude zero is infinite at zero
function restrict(f::IndBox, S::AbstractVector{<:Integer})
  if !f.zp
    f.l isa AbstractArray || f.u isa AbstractArray ||
      throw(ArgumentError("restrict of IndBox with scalar bounds requires 0 ∈ [l, u]"))
    n = length(f.l isa AbstractArray ? f.l : f.u)
    for i in _excluded_coordinates(n, S)
      _entry(f.l, i) <= 0 <= _entry(f.u, i) ||
        throw(ArgumentError("restrict of IndBox requires 0 ∈ [lᵢ, uᵢ] for every coordinate outside S"))
    end
  end
  _restrict_unchecked(f, S)
end
_restrict_unchecked(f::IndBox, S::AbstractVector{<:Integer}) = IndBox(_restrict_entries(f.l, S), _restrict_entries(f.u, S))

"""
    IndNonnegative()

The indicator of the nonnegative orthant `{x : x ≥ 0}`. Its proximal operator is
`max.(x, 0)`.
"""
struct IndNonnegative <: SeparableFunction end

isconvex(::Type{IndNonnegative}) = true
iszeropreserving(::IndNonnegative) = true

_isindicator(::Type{IndNonnegative}) = true
_value_dense(::IndNonnegative, x::AbstractArray) =
  (T = float(eltype(x)); isempty(x) || minimum(x) >= -_tol(T) ? zero(T) : T(Inf))

@inline value_entry(::IndNonnegative, v::T, ::Integer) where {T<:Real} =
  v >= -_tol(float(T)) ? zero(float(T)) : float(T)(Inf)
@inline prox_entry(::IndNonnegative, v::T, ::Integer, ::Real) where {T<:Real} = max(float(v), zero(float(T)))
kkt_violation(::IndNonnegative, g::AbstractArray, i::Integer) = -g[i]
restrict(f::IndNonnegative, ::AbstractVector{<:Integer}) = f

####################################
#  binary values
####################################

"""
    IndBinary(l, u)

The indicator of the nonconvex set `{x : xᵢ ∈ {lᵢ, uᵢ}}`: `0` on the set
and `Inf` outside, with the membership tolerance of [`IndBox`](@ref). The
bounds are finite scalars or arrays with the length of `x`, indexed linearly,
and must satisfy `lᵢ < uᵢ`. Array bounds are stored without copying; construct
a new `IndBinary` after changing them, since whether every pair contains zero
is computed at construction.

Its proximal operator selects the nearer bound, with ties going to `lᵢ`,
independently of the step. The comparison uses the promoted floating type of
the input and bounds; writing the result converts it to the output type.
`NaN` propagates, and `-Inf` and `Inf` project to the lower and upper bounds.
Nonfinite entries are never members. For `IndBinary(-1, 1)`, zero projects to
`-1`, unlike `sign(0) = 0`. For `IndBinary(0, 1)`, zero is preserved, so sparse
iterates visit only stored entries. [`restrict`](@ref) requires zero-preserving
bounds. No gradient-only [`kkt_violation`](@ref) is provided: a proximal
gradient update from zero projects `-γg`, so whether it stays zero depends on
the step.
"""
struct IndBinary{L<:Union{Real, AbstractArray}, U<:Union{Real, AbstractArray}} <: SeparableFunction
  l::L
  u::U
  zp::Bool

  function IndBinary{L, U}(l::L, u::U) where {L<:Union{Real, AbstractArray}, U<:Union{Real, AbstractArray}}
    l isa AbstractArray && u isa AbstractArray && _checkparam(l, u)
    all(isfinite, l) && all(isfinite, u) || throw(ArgumentError("the bounds must be finite"))
    n = l isa AbstractArray ? length(l) : u isa AbstractArray ? length(u) : 1
    all(i -> _entry(l, i) < _entry(u, i), 1:n) ||
      throw(ArgumentError("the lower bounds must be strictly less than the upper bounds"))
    zp = all(i -> iszero(_entry(l, i)) || iszero(_entry(u, i)), 1:n)
    new{L, U}(l, u, zp)
  end
end
IndBinary(l::Union{Real, AbstractArray}, u::Union{Real, AbstractArray}) = IndBinary{typeof(l), typeof(u)}(l, u)

iszeropreserving(f::IndBinary) = f.zp
_isindicator(::Type{<:IndBinary}) = true
_checkparams(f::IndBinary, x) = (_checkparam(f.l, x); _checkparam(f.u, x); true)

Base.@propagate_inbounds function value_entry(f::IndBinary, v::T, i::Integer) where {T<:Real}
  S = float(T)
  isfinite(v) || return S(Inf)
  w, l, u = promote(float(v), float(_entry(f.l, i)), float(_entry(f.u, i)))
  t = _tol(S)
  _ind(abs(w - l) <= t * max(one(l), abs(l)) || abs(w - u) <= t * max(one(u), abs(u)), S)
end

# The residual of a finite difference distinguishes distances that round to
# the same float, without rounding a midpoint or halving subnormal bounds.
@inline function _binary_distance_error(a, b, d)
  z = d - a
  (a - (d - z)) - (b + z)
end

Base.@propagate_inbounds function prox_entry(f::IndBinary, v::Real, i::Integer, ::Real)
  w, l, u = promote(float(v), float(_entry(f.l, i)), float(_entry(f.u, i)))
  isnan(w) && return w
  w <= l && return l
  w >= u && return u
  dl, du = w - l, u - w
  dl < du && return l
  dl > du && return u
  # At most one distance can overflow for finite bounds and an interior w.
  _binary_distance_error(w, l, dl) <= _binary_distance_error(u, w, du) ? l : u
end

function restrict(f::IndBinary, S::AbstractVector{<:Integer})
  iszeropreserving(f) || throw(ArgumentError("restrict requires zero-preserving binary bounds"))
  IndBinary(_restrict_entries(f.l, S), _restrict_entries(f.u, S))
end

####################################
#  ℓ0 penalty
####################################

"""
    NormL0(λ = 1)

The ℓ0 penalty `f(x) = λ ⋅ #{i : xᵢ ≠ 0}`, nonconvex. Its proximal operator is
hard thresholding: `xᵢ` is kept when `|xᵢ| > √(2γλ)` and set to zero otherwise.
"""
struct NormL0{T<:Real} <: SeparableFunction
  λ::T

  NormL0{T}(λ::T) where {T<:Real} = (_checknonneg("λ", λ); new{T}(λ))
end
NormL0(λ::Real = 1) = NormL0{typeof(λ)}(λ)

iszeropreserving(::NormL0) = true

Base.@propagate_inbounds value_entry(f::NormL0, v::T, ::Integer) where {T<:Real} = iszero(v) ? zero(float(T)) : convert(float(T), f.λ)
Base.@propagate_inbounds prox_entry(f::NormL0, v::T, ::Integer, γ::Real) where {T<:Real} =
  abs2(v) > 2 * γ * f.λ ? float(v) : zero(float(T))
restrict(f::NormL0, ::AbstractVector{<:Integer}) = f

####################################
#  MCP and SCAD
####################################

"""
    MCP(λ, a = 3)

The minimax concave penalty of Zhang (2010), nonconvex:
`fᵢ(t) = λ|t| - t²/(2a)` for `|t| ≤ aλ` and `aλ²/2` beyond, for `a > 0`. It
matches the ℓ1 penalty at zero and stops penalizing large entries. For a step
`γ < a` its proximal operator is firm thresholding; for larger steps it is the
global minimizer of the nonconvex problem.
"""
struct MCP{T<:Real} <: SeparableFunction
  λ::T
  a::T

  function MCP{T}(λ::T, a::T) where {T<:Real}
    _checknonneg("λ", λ)
    a > 0 || throw(ArgumentError("a must be positive, got $a"))
    new{T}(λ, a)
  end
end
MCP(λ::Real, a::Real = 3) = (p = promote(λ, a); MCP{typeof(p[1])}(p...))

iszeropreserving(::MCP) = true

Base.@propagate_inbounds function value_entry(f::MCP, v::T, ::Integer) where {T<:Real}
  S = float(T)
  λ, a, t = S(f.λ), S(f.a), abs(S(v))
  t <= a * λ ? λ * t - t^2 / (2a) : a * λ^2 / 2
end

Base.@propagate_inbounds function prox_entry(f::MCP, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  λ, a, s, t = S(f.λ), S(f.a), S(γ), abs(S(v))
  if s < a
    t <= s * λ && return zero(S)
    t > a * λ && return S(v)
    return sign(S(v)) * (t - s * λ) / (1 - s / a)
  end
  # concave between 0 and aλ: the minimizer is 0 or the largest candidate
  u = max(t, a * λ)
  h(z) = (z - t)^2 / (2s) + value_entry(f, z, i)
  h(u) < h(zero(S)) ? sign(S(v)) * u : zero(S)
end

kkt_violation(f::MCP, g::AbstractArray, i::Integer) = abs(g[i]) - f.λ
restrict(f::MCP, ::AbstractVector{<:Integer}) = f

"""
    SCAD(λ, a = 3.7)

The smoothly clipped absolute deviation penalty of Fan and Li (2001),
nonconvex: `fᵢ(t) = λ|t|` for `|t| ≤ λ`,
`(2aλ|t| - t² - λ²) / (2(a - 1))` for `λ < |t| ≤ aλ`, and `λ²(a + 1)/2` beyond,
for `a > 2`. For a step `γ < a - 1` its proximal operator is the SCAD
thresholding rule; for larger steps it is the global minimizer of the nonconvex
problem.
"""
struct SCAD{T<:Real} <: SeparableFunction
  λ::T
  a::T

  function SCAD{T}(λ::T, a::T) where {T<:Real}
    _checknonneg("λ", λ)
    a > 2 || throw(ArgumentError("a must exceed 2, got $a"))
    new{T}(λ, a)
  end
end
SCAD(λ::Real, a::Real = 3.7) = (p = promote(λ, a); SCAD{typeof(p[1])}(p...))

iszeropreserving(::SCAD) = true

Base.@propagate_inbounds function value_entry(f::SCAD, v::T, ::Integer) where {T<:Real}
  S = float(T)
  λ, a, t = S(f.λ), S(f.a), abs(S(v))
  t <= λ && return λ * t
  t <= a * λ && return (2a * λ * t - t^2 - λ^2) / (2 * (a - 1))
  λ^2 * (a + 1) / 2
end

Base.@propagate_inbounds function prox_entry(f::SCAD, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  λ, a, s, t = S(f.λ), S(f.a), S(γ), abs(S(v))
  if s < a - 1
    t <= (1 + s) * λ && return shrink(S(v), s * λ)
    t > a * λ && return S(v)
    return sign(S(v)) * ((a - 1) * t - s * a * λ) / (a - 1 - s)
  end
  # concave on [λ, aλ]: compare the minimizers of the three pieces
  h(z) = (z - t)^2 / (2s) + value_entry(f, z, i)
  best, hb = zero(S), h(zero(S))
  for z in (min(max(t - s * λ, zero(S)), λ), a * λ, max(t, a * λ))
    hz = h(z)
    hz < hb && ((best, hb) = (z, hz))
  end
  sign(S(v)) * best
end

kkt_violation(f::SCAD, g::AbstractArray, i::Integer) = abs(g[i]) - f.λ
restrict(f::SCAD, ::AbstractVector{<:Integer}) = f
