"""
    IndBox(l, u)

Indicator of the coordinatewise intervals `lᵢ ≤ xᵢ ≤ uᵢ`. Each bound is
a real scalar or a real one-based array with one entry per coordinate, paired
by linear index. Array bounds are copied at construction and the stored
bounds must be treated as read-only.

The value is `0` when every entry is in its interval and `Inf` otherwise.
A `NaN` entry gives value `NaN`, even when another entry is outside its
interval. Each interval can be unbounded below (`lᵢ = -Inf`), unbounded
above (`uᵢ = Inf`), or fixed at a finite point (`lᵢ = uᵢ`). Bounds that
give an empty interval or contain `NaN` throw `ArgumentError`.

[`prox`](@ref) clamps each entry to its interval and preserves `NaN` entries,
at every nonnegative step, including `0` and `Inf`. Public scalar-step
array oracles reject negative and `NaN` steps before writing. Array-step
entries and `prox_block` ignore the step.
[`value_and_prox!`](@ref) evaluates the value at the stored output.

```jldoctest
julia> prox(IndBox(0.0, [1.0, 2.0]), [-3.0, 4.0], 0.0)
2-element Vector{Float64}:
 0.0
 2.0
```

The box preserves structural zeros if every interval contains zero. Then
value evaluation on sparse, symmetric sparse and block iterates and
same-kind sparse or symmetric sparse projection visit stored entries only.
With a [`SymmetricSparseIterate`](@ref) input or output, each bound array
describes a full symmetric matrix in linear order. Validate paired entries
with [`ProximalBase.check_symmetric_parameters`](@ref). Proximal calls on
these iterates take scalar steps.

`Scaled(f, α)`, `α * f` and `f * α` return `f` itself for a nonnegative `α`,
including `0` and `Inf`. A negative or `NaN` factor throws `ArgumentError`.

[`restrict`](@ref) fixes omitted coordinates at zero and throws `ArgumentError`
if any omitted interval excludes zero. A scalar box that excludes zero
also throws, since its original dimension is unspecified.
"""
struct IndBox{L,U} <: SeparableFunction
  l::L
  u::U
  zp::Bool

  function IndBox(l, u)
    l, u = _box_bound(l), _box_bound(u)
    l isa Real && !(l < Inf) && _box_bounds_error()
    u isa Real && !(u > -Inf) && _box_bounds_error()
    l isa AbstractArray && u isa AbstractArray && _checkparam(l, u)
    n = l isa AbstractArray ? length(l) : u isa AbstractArray ? length(u) : 1
    zp = true
    for i in 1:n
      lo, hi = _entry(l, i), _entry(u, i)
      (lo <= hi && lo < Inf && hi > -Inf) || _box_bounds_error()
      zp &= lo <= 0 <= hi
    end
    new{typeof(l),typeof(u)}(l, u, zp)
  end
end

@noinline _box_bounds_error() = throw(ArgumentError("IndBox bounds must define nonempty intervals with no NaN endpoints"))
_box_bound(b::Real) = b
_box_bound(b) = _real_error(:IndBox)
function _box_bound(b::AbstractArray{T}) where {T<:Real}
  _check_one_based(b)
  T[b[i] for i in 1:length(b)]
end

isconvex(::Type{<:IndBox}) = true
iszeropreserving(f::IndBox) = f.zp
_isindicator(::Type{<:IndBox}) = true
check_parameters(f::IndBox, x) = (_checkparam(f.l, x); _checkparam(f.u, x); nothing)

Base.@propagate_inbounds function value_entry(f::IndBox, v::T, i::Integer) where {T<:Real}
  S = float(T)
  lo, hi = _entry(f.l, i), _entry(f.u, i)
  ifelse((lo <= v) & (v <= hi), zero(S), ifelse(isnan(v), S(NaN), S(Inf)))
end

# An ordered comparison rejects NaN along with points outside the box.
# When a chunk fails, scan from its start for NaN, which takes precedence.
function _value_dense(f::IndBox{<:Real,<:Real}, x::_HostArray)
  n = length(x)
  T = _valtype(x)
  chunk = 128
  tail = n - rem(n, chunk) + 1
  for start in 1:chunk:(tail - 1)
    _box_feasible(f, x, start, chunk) || return _box_infeasible_value(x, start:n, T)
  end
  _box_feasible(f, x, tail, n - tail + 1) ? zero(T) : _box_infeasible_value(x, tail:n, T)
end

@inline function _box_feasible(f, x, start, count)
  feasible = true
  @inbounds @simd for j in 0:(count - 1)
    v = x[start + j]
    feasible &= (f.l <= v) & (v <= f.u)
  end
  feasible
end

@noinline function _box_infeasible_value(x, indices, ::Type{T}) where {T}
  hasnan = false
  @inbounds @simd for i in indices
    hasnan |= isnan(x[i])
  end
  hasnan ? T(NaN) : T(Inf)
end

Base.@propagate_inbounds function prox_block(f::IndBox, v::T, i::Integer, ::Real) where {T<:Real}
  S = float(T)
  clamp(S(v), S(_entry(f.l, i)), S(_entry(f.u, i)))
end

@noinline _indbox_step_error(γ) = throw(ArgumentError("IndBox requires a nonnegative step γ, got $γ"))
@inline _check_indbox_step(γ::Real) = (γ >= 0 || _indbox_step_error(γ); nothing)
prox!(y::_ArrayOrSymmetricIterate, f::IndBox, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_check_indbox_step(γ); invoke(prox!, _SeparableScalarStep, y, f, x, γ))
value_and_prox!(y::_ArrayOrSymmetricIterate, f::IndBox, x::_ArrayOrSymmetricIterate, γ::Real) =
  (_check_indbox_step(γ); invoke(value_and_prox!, _SeparableScalarStep, y, f, x, γ))

"""
    IndBallLinf(r)

Construct the [`IndBox`](@ref) with scalar bounds `-r` and `r`, the
indicator of `maxᵢ |xᵢ| ≤ r` over the entries of a vector or matrix.
Negative, `NaN` and positive unsigned radii throw `ArgumentError`.
Zero gives the singleton box and `Inf` admits both infinite endpoints.
Value evaluation, projection, and supported steps and iterates follow
the `IndBox` contract.

```jldoctest
julia> prox(IndBallLinf(1.0), [-3.0, 0.5], 0.0)
2-element Vector{Float64}:
 -1.0
  0.5
```
"""
function IndBallLinf(r::Real)
  _checknonneg("r", r)
  r isa Unsigned && !iszero(r) &&
    throw(ArgumentError("IndBallLinf requires a radius with a representable negation, got $r"))
  IndBox(-r, r)
end

function restrict(f::IndBox, S::AbstractVector{<:Integer})
  if !iszeropreserving(f)
    bounds = f.l isa AbstractArray ? f.l : f.u
    bounds isa AbstractArray || throw(ArgumentError("restrict of IndBox with scalar bounds requires 0 ∈ [l, u]"))
    selected = falses(length(bounds))
    selected[S] .= true
    for i in eachindex(selected)
      selected[i] || _entry(f.l, i) <= 0 <= _entry(f.u, i) ||
        throw(ArgumentError("restrict of IndBox requires 0 to lie in every omitted interval"))
    end
  end
  f.l isa Real && f.u isa Real && return f
  IndBox(_restrict_entries(f.l, S), _restrict_entries(f.u, S))
end
