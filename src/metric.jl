####################################
#
#  Diagonal metrics: a step per entry, and per-coordinate curvatures
#
####################################

# In the metric of D = Diagonal(d), the proximal operator of a separable g is
#   argmin_z g(z) + Σᵢ dᵢ (zᵢ - xᵢ)² / 2,
# the scalar proximal operator with the step γᵢ = 1/dᵢ in every entry, and for a
# group norm a metric constant on each group keeps the closed form. A gradient
# method with the step 1/Lₖ in coordinate k, from the per-coordinate curvatures
# below, takes the scale of each column into account without standardizing X.

prox(f::AbstractFunction, x::AbstractArray, γ::AbstractArray) = prox!(similar(x, float(eltype(x))), f, x, γ)

prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  y === x ? y : copyto!(y, x)
end
value_and_prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_unchecked!(y, f, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (_checkreal(f, γ); _prox_unchecked!(y, f, x, γ); zero(_valtype(y)))

# separable functions: the step of each entry; on a SymmetricSparseIterate the
# steps are a p × p array read at the lower triangle, so give a symmetric one.
# Scalar and array steps share the loops of separable.jl; _entry specializes
# to the scalar or to γ[i] without a branch inside the loop.
prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_sep!(y, f, x, γ))
_prox_unchecked!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  _isbuiltin(f) ? _prox_sep!(y, f, x, γ) : prox!(y, f, x, γ)
value_and_prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_sep!(y, f, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  _isbuiltin(f) ? _value_and_prox_sep!(y, f, x, γ) : value_and_prox!(y, f, x, γ)

prox!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, h, y, x, γ); _prox_unchecked!(y, h, x, γ))
function _prox_unchecked!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::AbstractArray)
  isseparable(h) || throw(ArgumentError("array steps for an array scale require a separable function"))
  _checkwrite(prox!, h, y, x)
  _checkparam(γ, x)
  _value_and_prox_array!(y, h, x, γ, false)
  y
end
value_and_prox!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, h, y, x, γ); _value_and_prox_unchecked!(y, h, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::AbstractArray)
  isseparable(h) || throw(ArgumentError("array steps for an array scale require a separable function"))
  _checkwrite(value_and_prox!, h, y, x)
  _checkparam(γ, x)
  _value_at(_value_and_prox_array!(y, h, x, γ, true), y)
end

# group penalties: a step per group, constant on the group, keeps the closed form
value_and_prox!(y::AbstractArray, f::AbstractGroupPenalty, x::AbstractArray, γ::AbstractVector) = (x = _symmetric_argument(y, x); _checkreal(f, γ); _checkwrite(value_and_prox!, f, y, x); _value_and_prox_groups!(y, f, x, γ))
prox!(y::AbstractArray, f::AbstractGroupPenalty, x::AbstractArray, γ::AbstractVector) = (x = _symmetric_argument(y, x); _value_and_prox_unchecked!(y, f, x, γ); y)
value_and_prox!(y::AtomIterate, f::AbstractGroupPenalty, x::AtomIterate, γ::AbstractVector) = (_checkreal(f, γ); _checkwrite(value_and_prox!, f, y, x); _value_and_prox_groups!(y, f, x, γ))

# the entrywise reciprocal of an array of steps, without allocating
struct _ReciprocalArray{T, N, A<:AbstractArray} <: AbstractArray{T, N}
  γ::A
end
_ReciprocalArray(γ::AbstractArray{T, N}) where {T, N} = _ReciprocalArray{float(T), N, typeof(γ)}(γ)
Base.size(a::_ReciprocalArray) = size(a.γ)
Base.IndexStyle(::Type{<:_ReciprocalArray}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::_ReciprocalArray{T}, i::Int) where {T} = inv(convert(T, a.γ[i]))

# functions built from functions pass the steps on, entry by entry
prox!(y::AbstractArray, h::_ScalarPrecompose, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, h, y, x, γ); _prox_unchecked!(y, h, x, γ))
function _prox_unchecked!(y::AbstractArray, h::_ScalarPrecompose, x::AbstractArray, γ::AbstractArray)
  isseparable(h) || throw(ArgumentError("array steps for a scalar scale require a separable function"))
  _checkwrite(prox!, h, y, x)
  _checkparam(γ, x)
  _value_and_prox_precompose!(y, h, x, γ, false)
  y
end
value_and_prox!(y::AbstractArray, h::_ScalarPrecompose, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, h, y, x, γ); _value_and_prox_unchecked!(y, h, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, h::_ScalarPrecompose, x::AbstractArray, γ::AbstractArray)
  isseparable(h) || throw(ArgumentError("array steps for a scalar scale require a separable function"))
  _checkwrite(value_and_prox!, h, y, x)
  _checkparam(γ, x)
  _value_at(_value_and_prox_precompose!(y, h, x, γ, true), y)
end

function _value_and_prox_precompose!(y, h::_ScalarPrecompose, x, γ::AbstractArray, withvalue::Bool)
  z = _image!(h, x)
  _checkparams(h.f, z)
  T = eltype(z)
  steps = _ScaledArray{T}(γ, h.μ)
  v = withvalue ? _value_and_prox_unchecked!(z, h.f, z, steps) : (_prox_unchecked!(z, h.f, z, steps); zero(T))
  y .= z ./ h.A
  withvalue && eltype(y) !== eltype(x) && _has_value(h.f) ? _precompose_prox_value(h, y) : v
end

# the ridge-adjusted steps, without a second buffer
struct _RegularizeSteps{T, N, A<:AbstractArray} <: AbstractArray{T, N}
  γ::A
  ρ::T
  zero_level::Bool      # the level is zero as given, not as converted to T
end
_RegularizeSteps(γ::AbstractArray{<:Any, N}, ρ::T, zero_level::Bool) where {T, N} =
  _RegularizeSteps{T, N, typeof(γ)}(γ, ρ, zero_level)
Base.size(a::_RegularizeSteps) = size(a.γ)
Base.IndexStyle(::Type{<:_RegularizeSteps}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::_RegularizeSteps{T}, i::Int) where {T} =
  _regularize_step(convert(T, a.γ[i]), a.ρ, a.zero_level)

@inline function _regularize_shrink(v, γ, ρ::T, zero_level::Bool) where {T}
  a, b = _regularize_weights(convert(T, γ), ρ, zero_level)
  a * v
end
@inline function _regularize_shrink(v, c, γ, ρ::T, zero_level::Bool) where {T}
  a, b = _regularize_weights(convert(T, γ), ρ, zero_level)
  _ridge_blend(v, c, a, b)
end

function _shrunk!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::AbstractArray)
  T = float(eltype(y))
  ρ, zero_level = convert(T, r.ρ), iszero(r.ρ)
  r.c === nothing ? (y .= _regularize_shrink.(x, _checked_parameter(y, γ), ρ, zero_level)) :
                    (y .= _regularize_shrink.(x, _checked_parameter(y, r.c), _checked_parameter(y, γ), ρ, zero_level))
  _RegularizeSteps(γ, ρ, zero_level)
end

function _check_regularize_group_steps(op, r, y, x, γ)
  # Check shapes before the group layout, deferring the inner output checks.
  _check_wrapper_output(y)
  _checksize(y, x)
  _checkparams(r, x)
  r.c === nothing || Base.Broadcast.check_broadcast_axes(axes(y), x, r.c)
  if op === value_and_prox!
    r.c === nothing || _checksize(x, r.c)
  end
  _checkgroups(r.f, x)
  _checkgroupsteps(r.f, γ)
  isempty(r.f.free) && length(r.f.group_of) == length(x) ||
    throw(ArgumentError("per-group steps for Regularize require groups that cover every entry"))
  if y isa AtomIterate || y isa SymmetricSparseIterate
    name = y isa AtomIterate ? "AtomIterate" : "SymmetricSparseIterate"
    throw(ArgumentError("per-group steps for Regularize do not support $name outputs"))
  end
  true
end

_checkwrite(op, r::Regularize{<:AbstractGroupPenalty}, y::AbstractArray, x::AbstractArray, γ::AbstractVector) =
  (_check_regularize_group_steps(op, r, y, x, γ); _checkwrite(op, r, y, x))

function _value_and_prox_regularize_groups!(y, r::Regularize{<:AbstractGroupPenalty}, x, γ::AbstractVector, op::O) where {O}
  _check_regularize_group_steps(op, r, y, x, γ)
  _checkwrite(op, r, y, x)
  z = _output_buffer!(r.scratch, y)
  T = float(eltype(z))
  ρ, zero_level = convert(T, r.ρ), iszero(r.ρ)
  for k in eachindex(r.f.groups)
    step = convert(T, γ[k])
    @inbounds for i in r.f.groups[k]
      z[i] = r.c === nothing ? _regularize_shrink(x[i], step, ρ, zero_level) :
                              _regularize_shrink(x[i], r.c[i], step, ρ, zero_level)
    end
  end
  # Run in the buffer even for dense outputs, committing only after success.
  w = _wrapper_output(z, z)
  steps = _RegularizeSteps(γ, ρ, zero_level)
  if op === prox!
    _prox_unchecked!(w, r.f, z, steps)
    return _wrapper_commit!(y, w)
  end
  v = _value_and_prox_unchecked!(w, r.f, z, steps)
  v = _value_at(v + _ridge(r, w), w)
  _wrapper_commit!(y, w)
  v
end

prox!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, r, y, x, γ); _prox_unchecked!(y, r, x, γ))
function _prox_unchecked!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::AbstractArray)
  if γ isa AbstractVector && r.f isa AbstractGroupPenalty
    return _value_and_prox_regularize_groups!(y, r, x, γ, prox!)
  end
  isseparable(r) || throw(ArgumentError("array steps for Regularize require a separable function or a group penalty"))
  _checkwrite(prox!, r, y, x)
  _checkparam(γ, x)
  Base.Broadcast.check_broadcast_axes(axes(y), γ)
  _check_parameter_axes(y, γ)
  z = _output_buffer!(r.scratch, y)
  _prox_unchecked!(y, r.f, z, _shrunk!(z, r, x, γ))
end
value_and_prox!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, r, y, x, γ); _value_and_prox_unchecked!(y, r, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::AbstractArray)
  if γ isa AbstractVector && r.f isa AbstractGroupPenalty
    return _value_and_prox_regularize_groups!(y, r, x, γ, value_and_prox!)
  end
  isseparable(r) || throw(ArgumentError("array steps for Regularize require a separable function or a group penalty"))
  _checkwrite(value_and_prox!, r, y, x)
  _checkparam(γ, x)
  Base.Broadcast.check_broadcast_axes(axes(y), γ)
  _check_parameter_axes(y, γ)
  z = _output_buffer!(r.scratch, y)
  v = _value_and_prox_unchecked!(y, r.f, z, _shrunk!(z, r, x, γ))
  _value_at(v + _ridge(r, y), y)
end

prox!(y, s::Scaled, x, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, s, y, x, γ); _prox_unchecked!(y, s, x, γ))
_prox_unchecked!(y, s::Scaled, x, γ::AbstractArray) =
  (_checkparams(s, x); _array_route(s, x, γ) ? _prox_chain!(y, s, x, γ) : _prox_unchecked!(y, s.f, x, _ScaledArray(γ, s.α)))
value_and_prox!(y, s::Scaled, x, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, s, y, x, γ); _value_and_prox_unchecked!(y, s, x, γ))
_value_and_prox_unchecked!(y, s::Scaled, x, γ::AbstractArray) =
  (_checkparams(s, x); _array_route(s, x, γ) ? _value_and_prox_chain!(y, s, x, γ) :
   _value_at(_scaled_value_overflow(s.f, s.α, y, _value_and_prox_unchecked!(y, s.f, x, _ScaledArray(γ, s.α))), y))
# α v at the output y of an array step with a factor inside the band: a product that overflows
# with α < 1 is recovered as for a scalar step (the levels with the factor folded in, or the
# parts of the value); an underflow is not (#359)
@inline function _scaled_value_overflow(f, α, y, v)
  r = α * v
  (!isfinite(r) && _scale_below_one(α)) ? _scaled_value_recover(f, α, y, v, r) : r
end

# Whether an entry of an array step may leave the range in the product with the factors. A chain whose
# factors have a product that is not a normal number takes the route below. A single factor does
# when it is far from one, outside the band of a quarter of the exponent range (2^±255 for Float64,
# 2^±31 for Float32), whatever the entries are: reading them to decide would cost a pass over the step
# on every call. An entry of the step that is itself within a quarter of the range limit, with a factor
# inside the band, is rounded with the product as before. An indicator ignores its scale.
@inline _factor_band(::Type{T}) where {T<:AbstractFloat} = exponent(floatmax(T)) ÷ 4
@inline _wide_factor(α::Real, ::Type{T}) where {T<:AbstractFloat} =
  !(ldexp(one(T), -_factor_band(T)) <= α <= ldexp(one(T), _factor_band(T)))
@inline _wide_factor_type(x::AbstractArray) = float(eltype(x))
@inline _wide_factor_type(x) = nothing
@inline _wide_factor(α::Real, ::Nothing) = false
@inline function _wide_scale(s::Scaled, x, γ::AbstractArray)
  α = s.α
  (isfinite(α) && α > 0 && !_isindicator(s.f)) || return false
  _wide_factor(α, float(promote_type(eltype(γ), typeof(α)))) || _wide_factor(α, _wide_factor_type(x))
end
@inline _array_route(s::Scaled, x, γ::AbstractArray) = _chained(s) || _wide_scale(s, x, γ)

# The function and the steps that a proximal operator gets for the factors a and the step γ, and how
# they relate to the chain:
# - `:folded`: the whole factor goes into the levels of the function and the steps are the caller's
#   array. The function forms the coefficient of an entry from the level, the weight and the step
#   without a product that leaves the range, and its value is the value of the chain.
# - `:shifted`: the levels do not hold the whole factor (its product with them is not a normal
#   number), so a power of two 2^k is moved from the steps into the levels, with k centring the
#   exponents of the entries of γ. This reads γ once, and only here, never on an ordinary call.
#   The value of the function is 2^k times too large for the chain.
# - `:rounded`: no level can be folded, or the shifted levels or entries still leave the range. A
#   function with no level to fold takes each step multiplied by the factors with the exponent
#   applied last, rounded into the step's type.
function _array_fold(f, a, γ)
  g = _fold_level(f, _NormalScale(a))
  g === nothing || return g, γ, :folded
  shifted = _shifted_fold(f, a, γ)
  shifted === nothing ? (f, _ExpScaledArray(γ, a), :rounded) : (shifted..., :shifted)
end
function _shifted_fold(f, a::_ProjectionScale, γ::AbstractArray)
  isempty(γ) && return nothing    # no entry to shift: the rounded fallback has nothing to round
  S = promote_type(float(eltype(γ)), typeof(a.m))
  positive(g) = g > 0 && isfinite(g)
  hi = maximum(g -> positive(g) ? exponent(float(g)) : typemin(Int), γ)
  lo = minimum(g -> positive(g) ? exponent(float(g)) : typemax(Int), γ)
  lo > hi && return nothing       # no positive finite entry
  # the shifts that keep every positive entry in the range; the one that centres them is tried
  # first, then the extremes, for a level that needs to be raised or lowered by as much as they allow
  kmin, kmax = hi - exponent(floatmax(S)), lo - exponent(floatmin(S))
  kmin > kmax && return nothing
  for k in (clamp((lo + hi) ÷ 2, kmin, kmax), kmax, kmin)
    g = _fold_level(f, _NormalScale(_ProjectionScale(a.m, a.e + k)))
    g === nothing || return g, _ExpScaledArray(γ, _ProjectionScale(S(0.5), 1 - k))
  end
  nothing
end
function _prox_chain!(y, s, x, γ::AbstractArray)
  f, a = _scaled_chain(s)
  g, steps, _ = _array_fold(f, a, γ)
  _prox_unchecked!(y, g, x, steps)
end
function _value_and_prox_chain!(y, s, x, γ::AbstractArray)
  f, a = _scaled_chain(s)
  g, steps, kind = _array_fold(f, a, γ)
  kind === :folded && return _value_at(_value_and_prox_unchecked!(y, g, x, steps), y)
  if kind === :shifted
    # the value of the chain at the point, not that of the shifted function
    _prox_unchecked!(y, g, x, steps)
    return _value_at(_scaled_value_by(f, a, y, _value_unchecked(f, y)), y)
  end
  v = _value_and_prox_unchecked!(y, g, x, steps)
  # a leaf value that left the range is recovered in the function's own units, as for a scalar step
  _value_at(_scaled_value_by(f, a, y, v), y)
end

prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, t, y, x, γ); _prox_unchecked!(y, t, x, γ))
function _prox_unchecked!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, t, y, x)
  _checkparam(γ, x)
  Base.Broadcast.check_broadcast_axes(axes(y), γ)
  _check_parameter_axes(y, γ)
  z = _output_buffer!(t.scratch, y)
  z .= x .- _checked_parameter(z, γ) .* _checked_parameter(z, t.c)
  w = _wrapper_output(y, z)
  _prox_unchecked!(w, t.f, z, γ)
  _wrapper_commit!(y, w)
end
value_and_prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, t, y, x, γ); _value_and_prox_unchecked!(y, t, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::AbstractArray)
  _checkwrite(value_and_prox!, t, y, x)
  _checkparam(γ, x)
  Base.Broadcast.check_broadcast_axes(axes(y), γ)
  _check_parameter_axes(y, γ)
  z = _output_buffer!(t.scratch, y)
  z .= x .- _checked_parameter(z, γ) .* _checked_parameter(z, t.c)
  w = _wrapper_output(y, z)
  v = _value_and_prox_unchecked!(w, t.f, z, γ)
  v = _value_at(v + dot(t.c, w) + t.d, w)
  _wrapper_commit!(y, w)
  v
end

prox!(y::AbstractArray, t::Translate, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, t, y, x, γ); _prox_unchecked!(y, t, x, γ))
function _prox_unchecked!(y::AbstractArray, t::Translate, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, t, y, x)
  z = _output_buffer!(t.proxscratch, y)
  z .= x .- _checked_parameter(z, t.b)
  w = _wrapper_output(y, z)
  _prox_unchecked!(w, t.f, z, γ)
  w .+= _checked_parameter(w, t.b)
  _wrapper_commit!(y, w)
end
value_and_prox!(y::AbstractArray, t::Translate, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, t, y, x, γ); _value_and_prox_unchecked!(y, t, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, t::Translate, x::AbstractArray, γ::AbstractArray)
  _checkwrite(value_and_prox!, t, y, x)
  z = _output_buffer!(t.proxscratch, y)
  z .= x .- _checked_parameter(z, t.b)
  w = _wrapper_output(y, z)
  v = _value_and_prox_unchecked!(w, t.f, z, γ)
  w .+= _checked_parameter(w, t.b)
  _wrapper_commit!(y, w)
  _value_at(v, y)
end

# Moreau's identity in each entry: prox_{γᵢfᵢ*}(xᵢ) = xᵢ - γᵢ prox_{fᵢ/γᵢ}(xᵢ/γᵢ)
prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
function _prox_unchecked!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparams(c, x)
  _checkparam(γ, x)
  _checkwrite(prox!, c, y, x)
  if isseparable(c.f) && _projection_direct_output(c.f, y, x)
    return _conjugate_projection!(y, c.f, x, γ)
  end
  T = float(eltype(y))
  z = _array_buffer!(c.scratch, T, x)
  if isseparable(c.f) && _direct_conjugate(typeof(c.f))
    _conjugate_projection!(z, c.f, x, γ)
    _check_projection_result(z, x, γ)
    return copyto!(y, z)
  end
  F = typeof(c.f)
  # without dividing by γ: a separable positively homogeneous fᵢ or cone gives
  # prox_{fᵢ}(xᵢ), and the box the projection onto [γᵢlᵢ, γᵢuᵢ]
  if (isseparable(F) && _conjugate_isindicator(F)) || c.f isa IndBox
    c.f isa IndBox ? _moreau_set!(z, c.f, x, γ) : _prox_unchecked!(z, c.f, x, one(T))
    _checkmoreau(y, x, z, one(T), γ, false)
    y .= x .- z
  else
    z .= x ./ γ
    _prox_unchecked!(z, c.f, z, _ReciprocalArray(γ))
    _checkmoreau(y, x, z, γ, γ, true)
    y .= x .- _checked_parameter(y, γ) .* z
  end
end

# f** = f, with the steps and checks of f's own oracles
prox!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
function _prox_unchecked!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, c, y, x)
  _checkparam(γ, x)
  _prox_unchecked!(y, c.f.f, x, γ)
end
value_and_prox!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::AbstractArray)
  _checkwrite(value_and_prox!, c, y, x)
  _checkparam(γ, x)
  _value_and_prox_unchecked!(y, c.f.f, x, γ)
end

value_and_prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::AbstractArray)
  _checkwrite(value_and_prox!, c, y, x)
  _isindicator(c) && (_prox_unchecked!(y, c, x, γ); return _zero_value(y))
  if isseparable(c.f) && _any_direct_conjugate(typeof(c.f))
    _checkparam(γ, x)
    z = _array_buffer!(c.scratch, float(eltype(y)), x)
    v = _conjugate_projection_value!(z, c.f, x, γ)
    _check_projection_result(z, x, γ)
    copyto!(y, z)
    return _value_at(v, y)
  end
  _has_conjugate_value(c.f) && return _conjugate_prox_value!(y, c, x, γ)
  _checksize(y, x)
  _checkparam(γ, x)
  T = float(eltype(y))
  p = _array_buffer!(c.scratch, T, x)
  p .= x ./ γ
  fp = _value_and_prox_unchecked!(p, c.f, p, _ReciprocalArray(γ))
  _checkmoreau(y, x, p, γ, γ, true)
  y .= x .- _checked_parameter(y, γ) .* p
  _value_at(dot(p, y) - fp, y)
end

####################################
#  sets in a diagonal metric
####################################

function _check_metric_steps(f, γ::AbstractArray)
  for g in γ
    g > zero(g) && isfinite(g) ||
      throw(ArgumentError("the steps of $(nameof(typeof(f))) must be positive and finite"))
  end
  nothing
end

# The projection onto monotone vectors in the metric of Diagonal(1 ./ γ) is
# weighted isotonic regression with weights γmin/γᵢ; constant steps take the
# unweighted pooling.
prox!(y::AbstractArray, f::IndIsotonic, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
_prox_unchecked!(y::AbstractArray, f::IndIsotonic, x::AbstractArray, γ::AbstractArray) =
  (_isotonic_steps!(y, f, x, γ); y)
# The projection; the pooling status (see _isotonic_stored_value).
function _isotonic_steps!(y::AbstractArray, f::IndIsotonic, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  _check_metric_steps(f, γ)
  Base.require_one_based_indexing(γ)
  n = length(x)
  n == 0 && return 0
  T = float(eltype(y))
  S = promote_type(T, float(eltype(γ)))
  γmin = γmax = γ[1]
  equal = true
  @inbounds for i in 1:n
    g = γ[i]
    equal &= g == γmin
    γmin = min(γmin, g)
    γmax = max(γmax, g)
  end
  iszero(convert(S, γmin) / convert(S, γmax)) &&
    throw(ArgumentError("the ratio of the largest to the smallest step of IndIsotonic is too large to represent"))
  σ = f.decreasing ? -1 : 1
  status = if equal
    _pava!(y, x, _buffer!(f.sums, T, n), _buffer!(f.sizes, Int, n), σ, Val(true))
  else
    _wpava!(y, x, γ, γmin, _buffer!(f.means, S, n), _buffer!(f.weights, S, n), _buffer!(f.sizes, Int, n), σ, Val(true))
  end
  _isotonic_status!(y, x, status)       # a NaN couples every entry, as for a scalar step
end

# The projection onto a hyperplane in the metric of Diagonal(1 ./ γ),
#   x - γ .* a (⟨a, x⟩ - b) / Σᵢ γᵢaᵢ²,
# and onto a halfspace the same step when x violates the inequality
prox!(y::AbstractArray, f::Union{IndHyperplane, IndHalfspace}, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::Union{IndHyperplane, IndHalfspace}, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparams(f, x)
  _checkparam(γ, x)
  mγ = _check_linear_steps(f, γ)
  # Branch before passing the normal: the scaled and unscaled normals have different types.
  _isscaled(f) ? _metric_set_prox!(y, f, x, γ, _normalized_normal(f.a, f.s), _scaled_offset(f), mγ) :
                 _metric_set_prox!(y, f, x, γ, f.a, f.b, mγ)
end

# _check_metric_steps, which also returns the largest step: the loop reads every step anyway. For
# Float32 and Float64 it is branch-free, so that it vectorizes, and works on the bit patterns, which
# are ordered like the values among the positive finite numbers (as _maximum_abs uses): the steps
# are all positive and finite exactly when the smallest pattern is positive as an integer (a zero,
# a negative number and -0.0 are not) and the largest is below that of Inf (Inf and NaN are not).
# It throws for an invalid step before the largest is returned.
function _check_linear_steps(f, γ::AbstractArray{T}) where {T<:Union{Float32, Float64}}
  U = _int_type(T)
  lo, hi = typemax(U), zero(U)
  @inbounds @simd for i in eachindex(γ)
    b = reinterpret(U, γ[i])
    lo = min(lo, b)
    hi = max(hi, b)
  end
  (lo > zero(U)) & (hi < reinterpret(U, T(Inf))) || _throw_linear_steps(f)
  reinterpret(T, hi)
end
function _check_linear_steps(f, γ::AbstractArray)
  m = zero(eltype(γ))
  for g in γ
    g > zero(g) && isfinite(g) || _throw_linear_steps(f)
    m = ifelse(g > m, g, m)
  end
  m
end
@noinline _throw_linear_steps(f) =
  throw(ArgumentError("the steps of $(nameof(typeof(f))) must be positive and finite"))

# The working normal a and offset b: those of f, or the normalized ones; mγ = max γᵢ.
function _metric_set_prox!(y, f, x, γ, a, b, mγ)
  g = _dot_normal(a, x) - b                  # the residual; a halfspace clamps it only below
  d = _metric_sqnorm(f, x, γ)
  t = _model_coefficient(y, f isa IndHalfspace ? max(g, zero(g)) : g, d)
  if !_metric_step_safe(f, a, g, d, t, mγ) && _extreme_projection!(y, f, x, a, d, γ, typeof(t))
    return y
  end
  if _isscaled(f)
    # γᵢâᵢ in the coefficient's type, as for IndHyperslab.
    return _hyperslab_metric_output!(y, x, a, γ, t)
  end
  y .= x .- t .* _checked_parameter(y, γ) .* _checked_parameter(y, a)
end

# The diagonal-step counterpart of _scalar_step_safe (see src/functions/sets.jl for the facts F1
# and F2 and for B = floatmax(S) / 2, S the type of t). The order of evaluation is the dot
# product, g, then d = Σᵢ γᵢαᵢ² (formed by _metric_sqnorm, as before), then t = g / d, and then
# the correction cᵢ = (t γᵢ) αᵢ for a normal that is not scaled, or t (γᵢ âᵢ) for a scaled one,
# where the scaled normal's products are formed in the order of _hyperslab_metric_output!.
# - By (F1), a finite g covers the dot product and the offset (for a hyperplane the test of t
#   below covers it: g is infinite or NaN exactly when t is, d being finite and positive).
# - The terms of d are nonnegative, so a finite d covers every γᵢαᵢ² and the sum (αᵢ² itself
#   cannot overflow, since |αᵢ| ≤ 2^(emax / 4), or < 2 when scaled). Without it, d = Inf and a
#   finite g would give t = 0 and y = x without a warning.
# - An infinite or NaN t fails the two products below, since mγ > 0.
# - Let mγ = max γᵢ, R = |α| for a scalar normal and R = √q ≥ max |αᵢ| (1 - 2^-p) for an array
#   normal, and P = mγ R ≥ γᵢ|αᵢ| (up to the same rounding). By (F2), |t| mγ ≤ B covers every
#   partial product t γᵢ of a normal that is not scaled, P ≤ B every partial product γᵢ âᵢ of a
#   scaled one, and |t| P ≤ B every final product. A P that overflows in S is Inf and fails.
# It fires on an argument whose formula was finite only within a factor 2 of an overflow of one
# of these, and for an array normal and steps of very different sizes up to 2 √n times that.
@inline function _metric_step_safe(f, a, g, d, t::S, mγ) where {S}
  B, m, γm = _half_floatmax(S), abs(t), convert(S, mγ)
  p = γm * _normal_bound(f, a, S)
  _raw_finite(f, g) && _finite_value(d) &&
    (_isscaled(f) ? p <= B && m * p <= B : m * γm <= B && m * p <= B)
end

prox!(y::AbstractArray, f::IndHyperslab, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::IndHyperslab, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, f, y, x)
  _checkparams(f, x)
  _checkparam(γ, x)
  _check_metric_steps(f, γ)
  Base.Broadcast.check_broadcast_axes(axes(y), γ)
  isempty(x) && return y
  g = _hyperslab_residual(f, x)
  iszero(g) && return (y === x ? y : copyto!(y, x))
  t = g / _metric_sqnorm(f, x, γ)
  if _isscaled(f)
    return _hyperslab_metric_output!(y, x, _normalized_normal(f.a, f.s), γ, t)
  end
  _hyperslab_metric_output!(y, x, f.a, γ, t)
end

# Inline so that passing the scaled normal does not allocate a view.
@inline function _hyperslab_metric_output!(y, x, a, γ, t::T) where {T}
  y .= x .- t .* (T.(_checked_parameter(y, γ)) .* _checked_parameter(y, a))
end

# The simplex and the ℓ1 ball: max(xᵢ - γᵢν, 0) and shrink(xᵢ, γᵢν) for the
# level ν that meets the constraint
prox!(y::AbstractArray, f::IndSimplex, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::IndSimplex, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparam(γ, x)
  _check_metric_steps(f, γ)
  T = float(eltype(y))
  r = convert(T, f.r)
  if !f.equality
    s = zero(T)
    for v in x
      s += max(v, zero(v))
    end
    s <= r && return (y .= max.(x, zero(T)))
  end
  iszero(r) && return fill!(y, zero(eltype(y)))
  ν = _metric_simplex_level(x, γ, r, identity)
  y .= max.(x .- _checked_parameter(y, γ) .* ν, zero(T))
end

prox!(y::AbstractArray, f::IndBallL1, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::IndBallL1, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparam(γ, x)
  _check_metric_steps(f, γ)
  T = float(eltype(y))
  r = convert(T, f.r)
  norm(x, 1) <= r && return (y === x ? y : copyto!(y, x))
  iszero(r) && return fill!(y, zero(eltype(y)))
  ν = _metric_simplex_level(x, γ, r, abs)
  y .= shrink.(x, _checked_parameter(y, γ) .* ν)
end

value_and_prox!(y::AbstractArray, f::Union{IndHyperplane, IndHalfspace, IndHyperslab, IndSimplex, IndBallL1, IndIsotonic}, x::AbstractArray, γ::AbstractArray) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_unchecked!(y, f, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, f::Union{IndHyperplane, IndHalfspace, IndHyperslab, IndSimplex, IndBallL1}, x::AbstractArray, γ::AbstractArray) =
  (_prox_unchecked!(y, f, x, γ); zero(_valtype(y)))
_value_and_prox_unchecked!(y::AbstractArray, f::IndIsotonic, x::AbstractArray, γ::AbstractArray) =
  _isotonic_stored_value(y, _isotonic_steps!(y, f, x, γ))

# The level ν with Σᵢ max(uᵢ - γᵢν, 0) = r for uᵢ = h(xᵢ), by Newton's method
# from the left on this convex, decreasing, piecewise linear function. From
# ν = -∞, where every entry is active, each step solves for ν on the active set,
# which then only shrinks, so ν increases to the root; the steps end when the
# active set stays the same, at the exact root, as in Michelot's algorithm for
# the simplex. Rounding can only stop it a step early.
function _metric_simplex_level(x::AbstractArray, γ::AbstractArray, r::T, h::H) where {T, H}
  ν = T(-Inf)
  last = typemax(Int)
  while true
    s, w, k = zero(T), zero(T), 0
    @inbounds for i in eachindex(IndexLinear(), x)
      u, g = convert(T, h(x[i])), convert(T, γ[i])
      if u > g * ν
        s += u
        w += g
        k += 1
      end
    end
    (k >= last || k == 0) && return ν
    last = k
    ν = (s - r) / w
  end
end

####################################
#  per-coordinate curvatures
####################################

"""
    coordinate_lipschitz(f, k)

A Lipschitz constant `Lₖ` of the partial derivative `∂f/∂xₖ` of the smooth
function `f` along coordinate `k`, a linear index of its argument: the curvature
of `f` along the coordinate, for which a coordinate step `1/Lₖ` decreases `f`.
It is the diagonal entry of the Hessian for the quadratic losses, a bound on it
for [`GLMLoss`](@ref), and `Inf` where none exists:

| `f` | `Lₖ` |
|:----|:-----|
| [`LeastSquaresLoss`](@ref) | `‖X[:, j]‖²` for the row `j` of entry `k` |
| [`GLMLoss`](@ref) | `curvature(ℓ) ⋅ Σᵢ wᵢ X[i, k]²`; `Inf` for `PoissonLoss` |
| [`QuadraticFunction`](@ref) | `A[k, k]` |
| [`KronQuadratic`](@ref) | `A[r, r] B[c, c]` for entry `(r, c)` of `Θ` |
| [`SqrNormL2`](@ref)`(λ)` | `λ` |
| [`SqrHingeLoss`](@ref)`(y, w)` | `2wₖyₖ²` |
| [`CubeNormL2`](@ref)`(λ)` | `Inf` for `λ > 0`, `0` for `λ = 0` |
| [`LogDetLoss`](@ref) | `Inf` |

`Scaled` scales the constants, `Tilt` and `Translate` keep them, and `Sum` adds
them. [`coordinate_lipschitz!`](@ref) computes all of them.

These constants are what coordinate descent needs. A gradient method that moves
all coordinates at once can use them as a diagonal metric `D = Diagonal(L)`,
with the steps `γ = t ./ L` per entry in [`prox!`](@ref), but the diagonal of a
Hessian does not majorize it: the scalar `t` must come from a line search or be
`1/λmax(D^(-1/2) ∇²f D^(-1/2))`, which standardizing the columns of `X` amounts
to.
"""
coordinate_lipschitz(f::AbstractFunction, ::Integer) =
  throw(ArgumentError(issmooth(f) ? "no per-coordinate Lipschitz constant is known for $(nameof(typeof(f)))" :
                                    "$(nameof(typeof(f))) is not smooth"))

"""
    coordinate_lipschitz!(L, f)

Fill `L`, an array of the size of the argument of `f`, with the per-coordinate
Lipschitz constants [`coordinate_lipschitz`](@ref)`(f, k)`, and return `L`. It
allocates nothing.
"""
coordinate_lipschitz!(L::AbstractArray, f::AbstractFunction) =
  (_check_lipschitz_arrays(f, L); _coordinate_lipschitz_body!(L, f))
_coordinate_lipschitz_unchecked!(L::AbstractArray, f::AbstractFunction) =
  _isbuiltin(f) ? _coordinate_lipschitz_body!(L, f) : coordinate_lipschitz!(L, f)
function _coordinate_lipschitz_body!(L::AbstractArray, f::AbstractFunction)
  @inbounds for k in eachindex(IndexLinear(), L)
    L[k] = coordinate_lipschitz(f, k)
  end
  L
end

# Σᵢ wᵢ X[i, j]², over the stored entries of a sparse X
function _column_sumsq(X::AbstractMatrix, j::Integer, w = nothing)
  T = float(eltype(X))
  s = zero(T)
  @inbounds for i in axes(X, 1)
    s += convert(T, _entry(w, i)) * abs2(X[i, j])
  end
  s
end
function _column_sumsq(X::SparseMatrixCSC, j::Integer, w = nothing)
  T = float(eltype(X))
  s = zero(T)
  rows, vals = rowvals(X), nonzeros(X)
  @inbounds for t in nzrange(X, j)
    s += convert(T, _entry(w, rows[t])) * abs2(vals[t])
  end
  s
end

coordinate_lipschitz(::ZeroFunction, ::Integer) = 0.0
coordinate_lipschitz(f::LinearFunction, ::Integer) = zero(float(eltype(f.c)))
coordinate_lipschitz(f::SqrNormL2, ::Integer) = float(f.λ)
function coordinate_lipschitz(f::SqrDistL2, k::Integer)
  isseparable(f) && (iszero(f.λ) || isconvex(f)) && return float(f.λ)
  invoke(coordinate_lipschitz, Tuple{AbstractFunction, Integer}, f, k)
end

function coordinate_lipschitz(f::SqrHingeLoss, k::Integer)
  y, w = promote(float(_entry(f.y, k)), float(_entry(f.w, k)))
  _sqrhinge_slope(y, w, abs(y), 0)
end

coordinate_lipschitz!(L::AbstractArray, f::SqrHingeLoss) =
  (_check_lipschitz_arrays(f, L); _coordinate_lipschitz_unchecked!(L, f))
function _coordinate_lipschitz_unchecked!(L::AbstractArray, f::SqrHingeLoss)
  _checkparams(f, L)
  @inbounds for k in eachindex(IndexLinear(), L)
    L[k] = coordinate_lipschitz(f, k)
  end
  L
end
coordinate_lipschitz(f::CubeNormL2, ::Integer) = lipschitz(f)

function coordinate_lipschitz(f::QuadraticFunction{T}, k::Integer) where {T}
  checkbounds(f.A, k, k)
  convert(T, f.A[k, k])
end

function coordinate_lipschitz(f::KronQuadratic, k::Integer)
  p, q = size(f)
  checkbounds(1:p*q, k)
  r, c = mod1(k, p), cld(k, p)
  f.A[r, r] * f.B[c, c]
end

function coordinate_lipschitz(f::LeastSquaresLoss, k::Integer)
  p = size(f.X, 2)
  checkbounds(1:p*size(f.Y, 2), k)
  _column_sumsq(f.X, mod1(k, p))
end

# the squared norm of each column of X once, for every column of β
function coordinate_lipschitz!(L::AbstractArray, f::LeastSquaresLoss)
  eltype(L) <: Real || _real_error(:coordinate_lipschitz!)
  p, m = size(f.X, 2), size(f.Y, 2)
  length(L) == p * m || throw(DimensionMismatch("L has $(length(L)) entries and β has $(p * m)"))
  @inbounds for j in 1:p
    s = _column_sumsq(f.X, j)
    for c in 1:m
      L[(c - 1) * p + j] = s
    end
  end
  L
end

function coordinate_lipschitz(f::GLMLoss{L, T}, k::Integer) where {L, T}
  checkbounds(axes(f.X, 2), k)
  c = curvature(f.loss)
  isinf(c) ? T(Inf) : convert(T, c) * _column_sumsq(f.X, k, f.weights)
end

coordinate_lipschitz(::LogDetLoss{T}, ::Integer) where {T} = T(Inf)
Base.@propagate_inbounds function coordinate_lipschitz(f::BernoulliLoss, k::Integer)
  w = float(_entry(f.w, k))
  iszero(w) ? zero(w) : oftype(w, Inf)
end

coordinate_lipschitz(s::Scaled, k::Integer) = _chained(s) ?
  _apply_scale(_scaled_chain(s)[2], coordinate_lipschitz(_scaled_chain(s)[1], k)) : s.α * coordinate_lipschitz(s.f, k)
coordinate_lipschitz!(L::AbstractArray, s::Scaled) =
  (_check_lipschitz_arrays(s, L); _coordinate_lipschitz_unchecked!(L, s))
_coordinate_lipschitz_unchecked!(L::AbstractArray, s::Scaled) = _chained(s) ?
  (_coordinate_lipschitz_unchecked!(L, _scaled_chain(s)[1]); _scale!(L, _scaled_chain(s)[2])) :
  (_coordinate_lipschitz_unchecked!(L, s.f); L .*= s.α)
coordinate_lipschitz(t::Union{Tilt, Translate}, k::Integer) = coordinate_lipschitz(t.f, k)
coordinate_lipschitz!(L::AbstractArray, t::Union{Tilt, Translate}) =
  (_check_lipschitz_arrays(t, L); _coordinate_lipschitz_unchecked!(L, t))
_coordinate_lipschitz_unchecked!(L::AbstractArray, t::Union{Tilt, Translate}) = _coordinate_lipschitz_unchecked!(L, t.f)
coordinate_lipschitz(s::Sum, k::Integer) = sum(f -> coordinate_lipschitz(f, k), s.fs)

function coordinate_lipschitz(r::Regularize, k::Integer)
  r.c === nothing || checkbounds(r.c, k)
  coordinate_lipschitz(r.f, k) + r.ρ
end
coordinate_lipschitz!(L::AbstractArray, r::Regularize) =
  (_check_lipschitz_arrays(r, L); _coordinate_lipschitz_unchecked!(L, r))
function _coordinate_lipschitz_unchecked!(L::AbstractArray, r::Regularize)
  _checkparam(r.c, L)
  _coordinate_lipschitz_unchecked!(L, r.f)
  L .+= r.ρ
end
coordinate_lipschitz(h::_ScalarPrecompose, k::Integer) = h.μ * coordinate_lipschitz(h.f, k)
coordinate_lipschitz!(L::AbstractArray, h::_ScalarPrecompose) =
  (_check_lipschitz_arrays(h, L); _coordinate_lipschitz_unchecked!(L, h))
_coordinate_lipschitz_unchecked!(L::AbstractArray, h::_ScalarPrecompose) =
  (_coordinate_lipschitz_unchecked!(L, h.f); L .*= h.μ)

Base.@propagate_inbounds function coordinate_lipschitz(h::_ArrayPrecompose, k::Integer)
  a = abs(float(h.a[k]))
  _positive_product(coordinate_lipschitz(h.f, k), a, a)
end
coordinate_lipschitz!(L::AbstractArray, h::_ArrayPrecompose) =
  (_check_lipschitz_arrays(h, L); _coordinate_lipschitz_unchecked!(L, h))
function _coordinate_lipschitz_unchecked!(L::AbstractArray, h::_ArrayPrecompose)
  _checkparams(h, L)
  @inbounds for k in eachindex(IndexLinear(), L)
    L[k] = coordinate_lipschitz(h, k)
  end
  L
end
