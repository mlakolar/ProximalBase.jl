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
  (_check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  y === x ? y : copyto!(y, x)
end
value_and_prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_unchecked!(y, f, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::AbstractArray) =
  (_checkreal(f, γ); _prox_unchecked!(y, f, x, γ); zero(_valtype(y)))

# separable functions: the step of each entry; on a SymmetricSparseIterate the
# steps are a p × p array read at the lower triangle, so give a symmetric one.
# Scalar and array steps share the loops of separable.jl; _entry specializes
# to the scalar or to γ[i] without a branch inside the loop.
prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(prox!, f, y, x, γ); _prox_sep!(y, f, x, γ))
_prox_unchecked!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  _isbuiltin(f) ? _prox_sep!(y, f, x, γ) : prox!(y, f, x, γ)
value_and_prox!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_sep!(y, f, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, f::SeparableFunction, x::AbstractArray, γ::AbstractArray) =
  _isbuiltin(f) ? _value_and_prox_sep!(y, f, x, γ) : value_and_prox!(y, f, x, γ)

prox!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(prox!, h, y, x, γ); _prox_unchecked!(y, h, x, γ))
function _prox_unchecked!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::AbstractArray)
  isseparable(h) || throw(ArgumentError("array steps for an array scale require a separable function"))
  _checkwrite(prox!, h, y, x)
  _checkparam(γ, x)
  _value_and_prox_array!(y, h, x, γ, false)
  y
end
value_and_prox!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(value_and_prox!, h, y, x, γ); _value_and_prox_unchecked!(y, h, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::AbstractArray)
  isseparable(h) || throw(ArgumentError("array steps for an array scale require a separable function"))
  _checkwrite(value_and_prox!, h, y, x)
  _checkparam(γ, x)
  _value_at(_value_and_prox_array!(y, h, x, γ, true), y)
end

# group penalties: a step per group, constant on the group, keeps the closed form
value_and_prox!(y::AbstractArray, f::AbstractGroupPenalty, x::AbstractArray, γ::AbstractVector) = (_checkreal(f, γ); _checkwrite(value_and_prox!, f, y, x); _value_and_prox_groups!(y, f, x, γ))
prox!(y::AbstractArray, f::AbstractGroupPenalty, x::AbstractArray, γ::AbstractVector) = (_value_and_prox_unchecked!(y, f, x, γ); y)
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
  (_check_oracle_arrays(prox!, h, y, x, γ); _prox_unchecked!(y, h, x, γ))
function _prox_unchecked!(y::AbstractArray, h::_ScalarPrecompose, x::AbstractArray, γ::AbstractArray)
  isseparable(h) || throw(ArgumentError("array steps for a scalar scale require a separable function"))
  _checkwrite(prox!, h, y, x)
  _checkparam(γ, x)
  _value_and_prox_precompose!(y, h, x, γ, false)
  y
end
value_and_prox!(y::AbstractArray, h::_ScalarPrecompose, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(value_and_prox!, h, y, x, γ); _value_and_prox_unchecked!(y, h, x, γ))
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
end
_RegularizeSteps(γ::AbstractArray{<:Any, N}, ρ::T) where {T, N} =
  _RegularizeSteps{T, N, typeof(γ)}(γ, ρ)
Base.size(a::_RegularizeSteps) = size(a.γ)
Base.IndexStyle(::Type{<:_RegularizeSteps}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::_RegularizeSteps{T}, i::Int) where {T} =
  _regularize_step(convert(T, a.γ[i]), a.ρ)

@inline function _regularize_shrink(v, γ, ρ::T) where {T}
  a, b = _ridge_weights(convert(T, γ), ρ)
  a * v
end
@inline function _regularize_shrink(v, c, γ, ρ::T) where {T}
  a, b = _ridge_weights(convert(T, γ), ρ)
  _ridge_blend(v, c, a, b)
end

function _shrunk!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::AbstractArray)
  T = float(eltype(y))
  ρ = convert(T, r.ρ)
  r.c === nothing ? (y .= _regularize_shrink.(x, _checked_parameter(y, γ), ρ)) :
                    (y .= _regularize_shrink.(x, _checked_parameter(y, r.c), _checked_parameter(y, γ), ρ))
  _RegularizeSteps(γ, ρ)
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
  ρ = convert(T, r.ρ)
  for k in eachindex(r.f.groups)
    step = convert(T, γ[k])
    @inbounds for i in r.f.groups[k]
      z[i] = r.c === nothing ? _regularize_shrink(x[i], step, ρ) :
                              _regularize_shrink(x[i], r.c[i], step, ρ)
    end
  end
  # Run in the buffer even for dense outputs, committing only after success.
  w = _wrapper_output(z, z)
  steps = _RegularizeSteps(γ, ρ)
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
  (_check_oracle_arrays(prox!, r, y, x, γ); _prox_unchecked!(y, r, x, γ))
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
  (_check_oracle_arrays(value_and_prox!, r, y, x, γ); _value_and_prox_unchecked!(y, r, x, γ))
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
  (_check_oracle_arrays(prox!, s, y, x, γ); _prox_unchecked!(y, s, x, γ))
_prox_unchecked!(y, s::Scaled, x, γ::AbstractArray) = (_checkparams(s, x); _prox_unchecked!(y, s.f, x, _ScaledArray(γ, s.α)))
value_and_prox!(y, s::Scaled, x, γ::AbstractArray) =
  (_check_oracle_arrays(value_and_prox!, s, y, x, γ); _value_and_prox_unchecked!(y, s, x, γ))
_value_and_prox_unchecked!(y, s::Scaled, x, γ::AbstractArray) =
  (_checkparams(s, x); _value_at(s.α * _value_and_prox_unchecked!(y, s.f, x, _ScaledArray(γ, s.α)), y))

prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(prox!, t, y, x, γ); _prox_unchecked!(y, t, x, γ))
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
  (_check_oracle_arrays(value_and_prox!, t, y, x, γ); _value_and_prox_unchecked!(y, t, x, γ))
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
  (_check_oracle_arrays(prox!, t, y, x, γ); _prox_unchecked!(y, t, x, γ))
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
  (_check_oracle_arrays(value_and_prox!, t, y, x, γ); _value_and_prox_unchecked!(y, t, x, γ))
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
  (_check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
function _prox_unchecked!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparams(c, x)
  _checkparam(γ, x)
  _checkwrite(prox!, c, y, x)
  T = float(eltype(y))
  z = _array_buffer!(c.scratch, T, x)
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
  (_check_oracle_arrays(prox!, c, y, x, γ); _prox_unchecked!(y, c, x, γ))
function _prox_unchecked!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, c, y, x)
  _checkparam(γ, x)
  _prox_unchecked!(y, c.f.f, x, γ)
end
value_and_prox!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::Conjugate{<:Conjugate}, x::AbstractArray, γ::AbstractArray)
  _checkwrite(value_and_prox!, c, y, x)
  _checkparam(γ, x)
  _value_and_prox_unchecked!(y, c.f.f, x, γ)
end

value_and_prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(value_and_prox!, c, y, x, γ); _value_and_prox_unchecked!(y, c, x, γ))
function _value_and_prox_unchecked!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::AbstractArray)
  _checkwrite(value_and_prox!, c, y, x)
  _isindicator(c) && (_prox_unchecked!(y, c, x, γ); return _zero_value(y))
  if _has_conjugate_value(c.f)
    _prox_unchecked!(y, c, x, γ)
    return _value_at(_conjugate_value(c.f, y), y)
  end
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
  (_check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::IndIsotonic, x::AbstractArray, γ::AbstractArray)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  _check_metric_steps(f, γ)
  Base.require_one_based_indexing(γ)
  n = length(x)
  n == 0 && return y
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
  if equal
    _pava!(y, x, _buffer!(f.sums, T, n), _buffer!(f.sizes, Int, n), σ)
  else
    _wpava!(y, x, γ, γmin, _buffer!(f.means, S, n), _buffer!(f.weights, S, n), _buffer!(f.sizes, Int, n), σ)
  end
  y
end

# The projection onto a hyperplane in the metric of Diagonal(1 ./ γ),
#   x - γ .* a (⟨a, x⟩ - b) / Σᵢ γᵢaᵢ²,
# and onto a halfspace the same step when x violates the inequality
prox!(y::AbstractArray, f::Union{IndHyperplane, IndHalfspace}, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::Union{IndHyperplane, IndHalfspace}, x::AbstractArray, γ::AbstractArray)
  _checksize(y, x)
  _checkparams(f, x)
  _checkparam(γ, x)
  _check_metric_steps(f, γ)
  T = float(eltype(y))
  if _isscaled(f)
    a = _normalized_normal(f.a, f.s)
    g = _dot_normal(a, x) - _scaled_offset(f)
    f isa IndHalfspace && (g = max(g, zero(g)))
    t = convert(T, g / _metric_sqnorm(f, x, γ))
    return y .= x .- t .* (_checked_parameter(y, γ) .* _checked_parameter(y, a))
  end
  g = _dot_normal(f.a, x) - f.b
  f isa IndHalfspace && (g = max(g, zero(g)))
  t = convert(T, g / _metric_sqnorm(f, x, γ))
  y .= x .- t .* _checked_parameter(y, γ) .* _checked_parameter(y, f.a)
end

prox!(y::AbstractArray, f::IndHyperslab, x::AbstractArray, γ::AbstractArray) =
  (_check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
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
  (_check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
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
  (_check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
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
  (_check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_unchecked!(y, f, x, γ))
_value_and_prox_unchecked!(y::AbstractArray, f::Union{IndHyperplane, IndHalfspace, IndHyperslab, IndSimplex, IndBallL1, IndIsotonic}, x::AbstractArray, γ::AbstractArray) =
  (_prox_unchecked!(y, f, x, γ); zero(_valtype(y)))

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

coordinate_lipschitz(s::Scaled, k::Integer) = s.α * coordinate_lipschitz(s.f, k)
coordinate_lipschitz!(L::AbstractArray, s::Scaled) =
  (_check_lipschitz_arrays(s, L); _coordinate_lipschitz_unchecked!(L, s))
_coordinate_lipschitz_unchecked!(L::AbstractArray, s::Scaled) = (_coordinate_lipschitz_unchecked!(L, s.f); L .*= s.α)
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
