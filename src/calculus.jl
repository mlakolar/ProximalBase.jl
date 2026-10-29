####################################
#
#  Functions built from other functions
#
####################################

# lazy views of a gradient for the optimality tests: c g and g + c
struct _ScaledArray{T, N, A<:AbstractArray} <: AbstractArray{T, N}
  g::A
  c::T
end
_ScaledArray(g::AbstractArray{T, N}, c) where {T, N} = (S = promote_type(T, typeof(c)); _ScaledArray{S, N, typeof(g)}(g, S(c)))
Base.size(a::_ScaledArray) = size(a.g)
Base.IndexStyle(::Type{<:_ScaledArray}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::_ScaledArray, i::Int) = a.c * a.g[i]

struct _ShiftedArray{T, N, A<:AbstractArray, C<:AbstractArray} <: AbstractArray{T, N}
  g::A
  c::C
end
_ShiftedArray(g::AbstractArray{T, N}, c::AbstractArray) where {T, N} =
  _ShiftedArray{promote_type(T, eltype(c)), N, typeof(g), typeof(c)}(g, c)
Base.size(a::_ShiftedArray) = size(a.g)
Base.IndexStyle(::Type{<:_ShiftedArray}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::_ShiftedArray, i::Int) = a.g[i] + a.c[i]

####################################
#  α f
####################################

"""
    Scaled(f, α)
    α * f

The function `α f(x)` for `α ≥ 0`, with the roles of `f`: its gradient is
`α ∇f`, and its proximal operator `prox_{γαf}`, so that `0.5 * f` halves a loss
and `(1/n) * f` averages one.

At zero scale, supported finite-parameter penalties and losses simplify to
[`ZeroFunction`](@ref), while an indicator retains its domain and is returned
unchanged. Zero scaling of other functions is rejected when
the package has no domain-preserving proximal representation.
"""
struct Scaled{F<:AbstractFunction, T<:Real} <: AbstractFunction
  f::F
  α::T

  function Scaled{F, T}(f::F, α::T) where {F<:AbstractFunction, T<:Real}
    _checknonneg("α", α)
    iszero(α) && return _zero_scaled(f)
    new{F, T}(f, α)
  end
end
Scaled(f::AbstractFunction, α::Real) = Scaled{typeof(f), typeof(α)}(f, α)
Scaled(f::Scaled, α::Real) = Scaled(f.f, f.α * α)

@noinline _zero_scale_error(f) = throw(ArgumentError("zero scaling of $(typeof(f)) has no supported domain-preserving representation"))
_zero_scaled(f::AbstractFunction) = _isindicator(f) ? f : _zero_scale_error(f)
_zero_scaled(f::ZeroFunction) = f
_finite_parameter(::Nothing) = true
_finite_parameter(x::Real) = isfinite(x)
_finite_parameter(x::AbstractArray) = all(isfinite, x)
_finite_zero(f, parameters...) = all(_finite_parameter, parameters) ? ZeroFunction() : _zero_scale_error(f)
_zero_scaled(f::Union{NormL1, GroupNormL2, GroupNormLinf}) = _finite_zero(f, f.λ0, f.w)
_zero_scaled(f::SparseGroupNormL2) = _finite_zero(f, f.λ0, f.λ1, f.w)
_zero_scaled(f::Union{NormL2, CubeNormL2, NormLinf, NormL0}) = _finite_zero(f, f.λ)
_zero_scaled(f::Union{MCP, SCAD}) = _finite_zero(f, f.λ, f.a)
_zero_scaled(f::Union{GroupMCP, GroupSCAD}) = _finite_zero(f, f.λ0, f.w, f.a)
_zero_scaled(f::ElasticNet) = _finite_zero(f, f.λ1, f.λ2, f.w)
_zero_scaled(f::SqrNormL2) = _finite_zero(f, f.λ, f.y)
_zero_scaled(f::QuadraticFunction) = _finite_zero(f, f.A, f.b, f.c)
_zero_scaled(f::LeastSquaresLoss) = _finite_zero(f, f.X, f.Y)
_zero_scaled(f::LinearFunction) = _finite_zero(f, f.c, f.d)
_zero_scaled(f::Scaled) = (_finite_parameter(f.α) || _zero_scale_error(f); _zero_scaled(f.f))
function _zero_scaled(f::NormL1Nonneg)
  _finite_zero(f, f.λ0, f.w)
  IndNonnegative()
end

Base.:*(α::Real, f::AbstractFunction) = Scaled(f, α)
Base.:*(f::AbstractFunction, α::Real) = Scaled(f, α)

for trait in (:issmooth, :isproximable, :isseparable, :isconvex)
  @eval $trait(::Type{<:Scaled{F}}) where {F} = $trait(F)
end

(s::Scaled)(x) = s.α * s.f(x)
_scale!(g::AbstractArray, α) = (g .*= α)
_scale!(g::Tuple, α) = (map(gk -> _scale!(gk, α), g); g)
value_and_gradient!(g, s::Scaled, x) = (_checkwrite(value_and_gradient!, s, g, x); v = value_and_gradient!(g, s.f, x); _scale!(g, s.α); s.α * v)
gradient!(g, s::Scaled, x) = (_checkwrite(gradient!, s, g, x); gradient!(g, s.f, x); _scale!(g, s.α))
prox!(y, s::Scaled, x, γ::Real) = (_checkwrite(prox!, s, y, x); prox!(y, s.f, x, s.α * γ))
value_and_prox!(y, s::Scaled, x, γ::Real) = (_checkwrite(value_and_prox!, s, y, x); s.α * value_and_prox!(y, s.f, x, s.α * γ))
Base.@propagate_inbounds prox_entry(s::Scaled, v::Real, i::Integer, γ::Real) = prox_entry(s.f, v, i, s.α * γ)
_checkparams(s::Scaled, x) = _checkparams(s.f, x)
_checkwrite(op::O, s::Scaled, y, x) where {O} = _checkwrite(op, s.f, y, x)
iszeropreserving(s::Scaled) = isfinite(s.α) && iszeropreserving(s.f)
_isindicator(::Type{<:Scaled{F}}) where {F} = _isindicator(F)
# a scaled group penalty updates a group of an AtomIterate with a scaled step
prox_coordinate!(x::AtomIterate, s::Scaled, k::Integer, γ::Real) = prox_coordinate!(x, s.f, k, s.α * γ)
kkt_violation(s::Scaled, g::AbstractArray, i::Integer) =
  iszero(s.α) ? abs(g[i]) : s.α * kkt_violation(s.f, _ScaledArray(g, inv(s.α)), i)
for X in (:AbstractArray, :AtomIterate, :SymmetricSparseIterate)
  @eval kkt_violation(s::Scaled, g::AbstractArray, x::$X, k::Integer) =
    iszero(s.α) ? kkt_violation(ZeroFunction(), g, x, k) : s.α * kkt_violation(s.f, _ScaledArray(g, inv(s.α)), x, k)
end
restrict(s::Scaled, S::AbstractVector{<:Integer}) = Scaled(restrict(s.f, S), s.α)

####################################
#  f + ⟨c, ⋅⟩ + d
####################################

"""
    Tilt(f, c, d = 0)

The function `f(x) + ⟨c, x⟩ + d`, with the roles of `f`. Its proximal operator
is that of `f` at `x - γc`, and its gradient `∇f + c`. The linear terms of
augmented Lagrangians and of duality take this form.
"""
struct Tilt{F<:AbstractFunction, C<:AbstractArray, D<:Real} <: AbstractFunction
  f::F
  c::C
  d::D
end
Tilt(f::AbstractFunction, c::AbstractArray) = Tilt(f, c, zero(float(eltype(c))))

for trait in (:issmooth, :isproximable, :isseparable, :isconvex)
  @eval $trait(::Type{<:Tilt{F}}) where {F} = $trait(F)
end

(t::Tilt)(x) = t.f(x) + dot(t.c, x) + t.d
value_and_gradient!(g, t::Tilt, x) = (_checkwrite(value_and_gradient!, t, g, x); v = value_and_gradient!(g, t.f, x); g .+= t.c; v + dot(t.c, x) + t.d)
gradient!(g, t::Tilt, x) = (_checkwrite(gradient!, t, g, x); gradient!(g, t.f, x); g .+= t.c)

function _checkwrite(op::O, t::Tilt, g, x) where {O<:Union{typeof(gradient!), typeof(value_and_gradient!)}}
  _checkparams(t, x)
  Base.Broadcast.check_broadcast_axes(axes(g), t.c)
  _checkwrite(op, t.f, g, x)
end

function _checkwrite(op::O, t::Tilt, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checksize(y, x)
  _checkparams(t, x)
  Base.Broadcast.check_broadcast_axes(axes(y), x, t.c)
  _checkwrite(op, t.f, y, y)
end

function prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::Real)
  _checkwrite(prox!, t, y, x)
  y .= x .- convert(float(eltype(y)), γ) .* t.c
  prox!(y, t.f, y, γ)
end
function value_and_prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, t, y, x)
  y .= x .- convert(float(eltype(y)), γ) .* t.c
  value_and_prox!(y, t.f, y, γ) + dot(t.c, y) + t.d
end
Base.@propagate_inbounds prox_entry(t::Tilt, v::Real, i::Integer, γ::Real) = prox_entry(t.f, v - γ * t.c[i], i, γ)
_checkparams(t::Tilt, x) = (_checkparam(t.c, x); _checkparams(t.f, x))
kkt_violation(t::Tilt, g::AbstractArray, i::Integer) = kkt_violation(t.f, _ShiftedArray(g, t.c), i)
restrict(t::Tilt, S::AbstractVector{<:Integer}) = Tilt(restrict(t.f, S), t.c[S], t.d)

####################################
#  f(⋅ - b)
####################################

"""
    Translate(f, b)

The function `f(x - b)`, with the roles of `f`: its gradient is `∇f(x - b)` and
its proximal operator `b + prox_{γf}(x - b)`. For example, `Translate(NormL1(λ), β₀)`
is the penalty `λ ‖β - β₀‖₁` that shrinks towards `β₀`.
Values and gradients use a shifted buffer in the argument's precision.
"""
struct Translate{F<:AbstractFunction, B<:AbstractArray} <: AbstractFunction
  f::F
  b::B
  scratch::Scratch
end
Translate(f::AbstractFunction, b::AbstractArray) = Translate(f, b, Scratch())

for trait in (:issmooth, :isproximable, :isseparable, :isconvex)
  @eval $trait(::Type{<:Translate{F}}) where {F} = $trait(F)
end

function _shifted_buffer(t::Translate, x::AbstractArray)
  size(x) == size(t.b) || throw(DimensionMismatch("x has size $(size(x)) and b $(size(t.b))"))
  _checkparams(t, x)
  _array_buffer!(t.scratch, float(eltype(x)), size(x))
end

# x - b in a buffer owned by t
function _shifted(t::Translate, x::AbstractArray)
  z = _shifted_buffer(t, x)
  z .= x .- t.b
end

(t::Translate)(x::AbstractArray) = t.f(_shifted(t, x))
value_and_gradient!(g, t::Translate, x::AbstractArray) =
  (_checkwrite(value_and_gradient!, t, g, x); value_and_gradient!(g, t.f, _shifted(t, x)))
gradient!(g, t::Translate, x::AbstractArray) =
  (_checkwrite(gradient!, t, g, x); gradient!(g, t.f, _shifted(t, x)))

_checkwrite(op::O, t::Translate, g, x::AbstractArray) where {O<:Union{typeof(gradient!), typeof(value_and_gradient!)}} =
  _checkwrite(op, t.f, g, _shifted_buffer(t, x))

function _checkwrite(op::O, t::Translate, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checksize(y, x)
  _checksize(t.b, x)
  _checkparams(t, x)
  _checkwrite(op, t.f, y, y)
end

function prox!(y::AbstractArray, t::Translate, x::AbstractArray, γ::Real)
  _checkwrite(prox!, t, y, x)
  y .= x .- t.b
  prox!(y, t.f, y, γ)
  y .+= t.b
end
function value_and_prox!(y::AbstractArray, t::Translate, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, t, y, x)
  y .= x .- t.b
  v = value_and_prox!(y, t.f, y, γ)
  y .+= t.b
  v
end
Base.@propagate_inbounds prox_entry(t::Translate, v::Real, i::Integer, γ::Real) = t.b[i] + prox_entry(t.f, v - t.b[i], i, γ)
_checkparams(t::Translate, x) = (_checkparam(t.b, x); _checkparams(t.f, x))
_isindicator(::Type{<:Translate{F}}) where {F} = _isindicator(F)
function restrict(t::Translate, S::AbstractVector{<:Integer})
  selected = falses(length(t.b))
  selected[S] .= true
  excluded = findall(!, selected)
  if !isempty(excluded)
    if isseparable(t.f)
      # Dropping finite terms is harmless for minimization, but dropping an
      # infinite term would turn an infeasible supported problem into a feasible one.
      v = restrict(t.f, excluded)(-t.b[excluded])
      isfinite(v) || throw(ArgumentError("excluded translated coordinates are outside the function's domain"))
    elseif any(i -> !iszero(t.b[i]), excluded)
      throw(ArgumentError("restrict of a nonseparable Translate requires zero translation outside S"))
    end
  end
  Translate(restrict(t.f, S), t.b[S])
end

####################################
#  convex conjugate
####################################

"""
    Conjugate(f)

The convex conjugate `f*(y) = supₓ ⟨x, y⟩ - f(x)` of a convex `f`. Its proximal
operator follows from that of `f` by Moreau's identity,
`prox_{γf*}(x) = x - γ prox_{f/γ}(x/γ)`, so every proximable function gives one;
for a separable `f` so does [`prox_entry`](@ref). Its value is known for the
norms (the indicators of the dual balls), the balls (their support functions),
the box, the nonnegative orthant, [`SqrNormL2`](@ref), [`LinearFunction`](@ref),
[`ZeroFunction`](@ref), and the losses [`CheckLoss`](@ref), [`HingeLoss`](@ref),
[`SqrHingeLoss`](@ref) and [`EpsInsensitiveLoss`](@ref), and for translations and strictly positive
rescalings of these functions; for other functions, evaluating the conjugate
throws an `ArgumentError`. [`value_and_prox!`](@ref) returns the value at the
proximal point for every proximable `f`: where no closed form is known, it uses
the Fenchel equality `f*(y) = ⟨p, y⟩ - f(p)`, with `p` the corresponding
proximal point of `f`. Cancellation in this subtraction and errors in the inner
proximal point affect the reported value. A conjugate that is an indicator
can return a small nonzero value instead of zero; this is not a feasibility
certificate.
Supported zero rescalings simplify in [`Scaled`](@ref) while preserving the
underlying domain. Each conjugate owns a working buffer; do not share it
between threads. Tuples, including nested tuples, use one buffer per array
block and support an output that is the input tuple.
"""
struct Conjugate{F<:AbstractFunction} <: AbstractFunction
  f::F
  scratch::Scratch

  function Conjugate{F}(f::F) where {F<:AbstractFunction}
    isconvex(f) || throw(ArgumentError("Conjugate requires a convex function"))
    new{F}(f, Scratch())
  end
end
Conjugate(f::AbstractFunction) = Conjugate{typeof(f)}(f)

isproximable(::Type{<:Conjugate{F}}) where {F} = isproximable(F)
isseparable(::Type{<:Conjugate{F}}) where {F} = isseparable(F)
isconvex(::Type{<:Conjugate}) = true
iszeropreserving(c::Conjugate) = iszeropreserving(c.f)

function prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::Real)
  _checksize(y, x)
  _checkparams(c, x)
  T = float(eltype(y))
  γ = convert(T, γ)
  z = _array_buffer!(c.scratch, T, size(x))
  z .= x ./ γ
  prox!(z, c.f, z, inv(γ))
  y .= x .- γ .* z
end
# At a proximal point y = x - γp, Fenchel equality gives f*(y) = <p,y> - f(p).
# Keep the direct value for known closed forms, especially exact indicator zeros.
_has_conjugate_value(::AbstractFunction) = false
_has_conjugate_value(::Union{NormL1, NormL2, NormLinf, IndBallL1, IndBallL2, IndSOC,
                            ZeroFunction, IndNonnegative, IndBox, SqrNormL2,
                            LinearFunction, CheckLoss, HingeLoss, SqrHingeLoss, EpsInsensitiveLoss}) = true
_has_conjugate_value(f::Union{Scaled, Translate}) = _has_conjugate_value(f.f)
_has_conjugate_value(f::SingularValue) = _has_conjugate_value(f.f)

function value_and_prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::Real)
  _checkparams(c, x)
  if _has_conjugate_value(c.f)
    prox!(y, c, x, γ)
    return _conjugate_value(c.f, y)
  end
  _checksize(y, x)
  T = float(eltype(y))
  step = convert(T, γ)
  p = _array_buffer!(c.scratch, T, size(x))
  p .= x ./ step
  fp = value_and_prox!(p, c.f, p, inv(step))
  y .= x .- step .* p
  dot(p, y) - fp
end
Base.@propagate_inbounds prox_entry(c::Conjugate, v::Real, i::Integer, γ::Real) = v - γ * prox_entry(c.f, v / γ, i, inv(γ))
_checkparams(c::Conjugate, x) = _checkparams(c.f, x)

(c::Conjugate)(y::Union{AbstractArray, Tuple}) = (_checkparams(c, y); _conjugate_value(c.f, y))

_tuple_buffer_type(::Type{<:AbstractArray{T, N}}) where {T, N} = Array{float(T), N}
_tuple_buffer_type(::Type{X}) where {X<:Tuple} = Tuple{map(_tuple_buffer_type, fieldtypes(X))...}
_same_buffer_sizes(b::AbstractArray, x::AbstractArray) = size(b) == size(x)
_same_buffer_sizes(b::Tuple, x::Tuple) = all(map(_same_buffer_sizes, b, x))

function _tuple_buffer!(s::Scratch, y::Y, x::X) where {Y<:Tuple, X<:Tuple}
  B = _tuple_buffer_type(Y)
  cached = s.buf
  if cached isa Tuple{Val{X}, Val{Y}, B}
    b = cached[3]
    _same_buffer_sizes(b, y) && return b
  end
  b = _gradient_buffer(y)
  s.buf = (Val{X}(), Val{Y}(), b)
  b::B
end

_divide!(z::AbstractArray, x::AbstractArray, γ) = (z .= x ./ convert(float(eltype(z)), γ))
_divide!(z::Tuple, x::Tuple, γ) = (map((zk, xk) -> _divide!(zk, xk, γ), z, x); z)
_moreau_output!(y::AbstractArray, x::AbstractArray, p::AbstractArray, γ) =
  (y .= x .- convert(float(eltype(y)), γ) .* p)
_moreau_output!(y::Tuple, x::Tuple, p::Tuple, γ) =
  (map((yk, xk, pk) -> _moreau_output!(yk, xk, pk, γ), y, x, p); y)
_tuple_dot(x::AbstractArray, y::AbstractArray) = dot(x, y)
_tuple_dot(x::Tuple, y::Tuple) = sum(map(_tuple_dot, x, y))

function prox!(y::Tuple, c::Conjugate, x::Tuple, γ::Real)
  _checksize(y, x)
  _checkparams(c, x)
  p = _tuple_buffer!(c.scratch, y, x)
  _divide!(p, x, γ)
  prox!(p, c.f, p, inv(γ))
  _moreau_output!(y, x, p, γ)
end

function value_and_prox!(y::Tuple, c::Conjugate, x::Tuple, γ::Real)
  _checksize(y, x)
  _checkparams(c, x)
  if _has_conjugate_value(c.f)
    prox!(y, c, x, γ)
    return _conjugate_value(c.f, y)
  end
  p = _tuple_buffer!(c.scratch, y, x)
  _divide!(p, x, γ)
  fp = value_and_prox!(p, c.f, p, inv(γ))
  _moreau_output!(y, x, p, γ)
  _tuple_dot(p, y) - fp
end

# (αf)*(y) = α f*(y/α), and (f(⋅-b))*(y) = f*(y) + ⟨b,y⟩.
# Zero scaling of an extended-valued function depends on its domain, so the
# positive-scaling formula deliberately does not cover it.
function _conjugate_value(f::Scaled, y::AbstractArray)
  f.α > 0 || throw(ArgumentError("conjugate values of Scaled require a positive scale"))
  T = float(eltype(y))
  α = convert(T, f.α)
  α * _conjugate_value(f.f, _ScaledArray(y, inv(α)))
end
function _conjugate_value(f::Scaled, y::Tuple)
  f.α > 0 || throw(ArgumentError("conjugate values of Scaled require a positive scale"))
  _scaled_conjugate_value(f.f, y, f.α)
end
function _conjugate_value(f::Translate, y::AbstractArray)
  _checksize(f.b, y)
  _conjugate_value(f.f, y) + dot(f.b, y)
end

@noinline _conjugate_value(f::AbstractFunction, y) =
  throw(ArgumentError("the value of the conjugate of $(typeof(f)) is not implemented; its proximal operator is"))

# indicators of the dual balls and support functions of the balls, within the
# tolerance of the indicator functions; loops rather than closures, which would
# capture the element type as a DataType and dispatch dynamically
_within(a, b, ::Type{T}) where {T} = a <= b || a - b <= _tol(T) * max(one(T), abs(b))

function _conjugate_value(f::NormL1, y::AbstractArray)
  _checkparams(f, y)
  T = float(eltype(y))
  @inbounds for i in eachindex(IndexLinear(), y)
    _within(abs(y[i]), f.λ0 * _entry(f.w, i), T) || return T(Inf)
  end
  zero(T)
end
_conjugate_value(f::NormL2, y::AbstractArray) = (T = float(eltype(y)); _ind(_within(_norm2(y), f.λ, T), T))
_conjugate_value(f::NormLinf, y::AbstractArray) = (T = float(eltype(y)); _ind(_within(norm(y, 1), f.λ, T), T))
_conjugate_value(f::IndBallL1, y::AbstractArray) = convert(float(eltype(y)), f.r) * norm(y, Inf)
_conjugate_value(f::IndBallL2, y::AbstractArray) =
  (T = float(eltype(y)); iszero(f.r) ? zero(T) : convert(T, f.r) * _norm2(y))
_conjugate_value(f::IndSOC, y::AbstractArray) =
  (_checkparams(f, y); _soc_value(f, y, true))
function _conjugate_value(::ZeroFunction, y::AbstractArray)
  T = float(eltype(y))
  for v in y
    _within(abs(v), zero(T), T) || return T(Inf)
  end
  zero(T)
end
function _conjugate_value(::IndNonnegative, y::AbstractArray)
  T = float(eltype(y))
  t = _tol(T)
  for v in y
    v <= t || return T(Inf)
  end
  zero(T)
end
function _conjugate_value(f::IndBox, y::AbstractArray)
  _checkparams(f, y)
  T = float(eltype(y))
  s = zero(T)
  @inbounds for i in eachindex(IndexLinear(), y)
    v = y[i]
    s += v >= 0 ? (iszero(v) ? zero(T) : v * _entry(f.u, i)) : v * _entry(f.l, i)
  end
  s
end
function _conjugate_value(f::SqrNormL2, y::AbstractArray)
  iszero(f.λ) && return _conjugate_value(ZeroFunction(), y)
  T = float(eltype(y))
  S = float(promote_type(typeof(real(zero(T))), typeof(f.λ)))
  scale = inv(sqrt(convert(S, f.λ)) * sqrt(S(2)))
  r = _norm2(y)
  # Preserve sparse O(nnz) norms at ordinary scales. If the norm itself is
  # outside its working range, scale the entries before computing it.
  v = convert(T, isfinite(r) ? abs2(r * scale) : abs2(_norm2(_ScaledArray(y, scale))))
  f.y === nothing ? v : v + dot(f.y, y)
end
function _conjugate_value(f::LinearFunction, y::AbstractArray)
  T = float(eltype(y))
  size(y) == size(f.c) || throw(DimensionMismatch())
  @inbounds for i in eachindex(IndexLinear(), y)
    _within(abs(y[i] - f.c[i]), zero(T), T) || return T(Inf)
  end
  -convert(T, f.d)
end

####################################
#  blocks
####################################

"""
    SeparableSum(f1, f2, ...)

The function `Σₖ fₖ(xₖ)` of a tuple `x = (x1, x2, ...)` of blocks, with the
roles that all the `fₖ` share: the proximal operator, gradient and value act
block by block on tuples. Sparse plus low-rank models, for example, take
`SeparableSum(NormL1(λ), TracePSD(μ))` on the pair `(S, L)`.
Blocks may themselves be tuples for nested sums. Allocating oracles and
[`Conjugate`](@ref) preserve this block structure.
Before writing any output, the oracles check the sizes of all blocks and the
parameters, data dimensions and squareness that the block functions' own
oracles check before writing. A function that transforms its argument, such as
`Precompose` or `Constrained`, or fails during computation can still leave
earlier blocks written.
"""
struct SeparableSum{F<:Tuple} <: AbstractFunction
  fs::F
end
SeparableSum(fs::AbstractFunction...) = SeparableSum(fs)

for trait in (:issmooth, :isproximable, :isconvex)
  @eval $trait(::Type{SeparableSum{F}}) where {F} = all($trait, fieldtypes(F))
end

function _checkblocks(s::SeparableSum, x::Tuple)
  length(x) == length(s.fs) || throw(DimensionMismatch("$(length(s.fs)) functions for $(length(x)) blocks"))
end

_checkparams(s::SeparableSum, x::Tuple) = (_checkblocks(s, x); map(_checkparams, s.fs, x); true)
function _checkwrite(op::O, s::SeparableSum, y::Tuple, x::Tuple) where {O<:Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}}
  _checksize(y, x)
  _checkblocks(s, x)
  map((f, yk, xk) -> _checkwrite(op, f, yk, xk), s.fs, y, x)
  true
end
_has_conjugate_value(s::SeparableSum) = all(_has_conjugate_value, s.fs)
_conjugate_value(s::SeparableSum, y::Tuple) =
  (_checkparams(s, y); sum(map(_conjugate_value, s.fs, y)))
_scaled_conjugate_value(s::SeparableSum, y::Tuple, α) =
  (_checkparams(s, y); sum(map((f, yk) -> _conjugate_value(Scaled(f, α), yk), s.fs, y)))

(s::SeparableSum)(x::Tuple) = (_checkparams(s, x); sum(map((f, xk) -> f(xk), s.fs, x)))
prox!(y::Tuple, s::SeparableSum, x::Tuple, γ::Real) =
  (_checkwrite(prox!, s, y, x); map((f, yk, xk) -> prox!(yk, f, xk, γ), s.fs, y, x); y)
value_and_prox!(y::Tuple, s::SeparableSum, x::Tuple, γ::Real) =
  (_checkwrite(value_and_prox!, s, y, x); sum(map((f, yk, xk) -> value_and_prox!(yk, f, xk, γ), s.fs, y, x)))
value_and_gradient!(g::Tuple, s::SeparableSum, x::Tuple) =
  (_checkwrite(value_and_gradient!, s, g, x); sum(map((f, gk, xk) -> value_and_gradient!(gk, f, xk), s.fs, g, x)))
gradient!(g::Tuple, s::SeparableSum, x::Tuple) =
  (_checkwrite(gradient!, s, g, x); map((f, gk, xk) -> gradient!(gk, f, xk), s.fs, g, x); g)

####################################
#  sums of smooth functions
####################################

"""
    Sum(f1, f2, ...)

The function `Σₖ fₖ(x)`, smooth when all the `fₖ` are, with the sum of their
gradients: a loss plus a smooth penalty, for example. A sum has no proximal
operator in general.
"""
struct Sum{F<:Tuple} <: AbstractFunction
  fs::F
  scratch::Scratch
end
Sum(fs::AbstractFunction...) = Sum(fs, Scratch())

issmooth(::Type{<:Sum{F}}) where {F} = all(issmooth, fieldtypes(F))
isconvex(::Type{<:Sum{F}}) where {F} = all(isconvex, fieldtypes(F))
_checkparams(s::Sum, x) = (map(f -> _checkparams(f, x), s.fs); true)

(s::Sum)(x) = sum(f -> f(x), s.fs)

# the remaining functions' gradients added into g through the buffer b
_add_gradients!(g, b, x, v, ::Tuple{}) = v
function _add_gradients!(g, b, x, v, fs::Tuple)
  w = value_and_gradient!(b, first(fs), x)
  g .+= b
  _add_gradients!(g, b, x, v + w, Base.tail(fs))
end

function value_and_gradient!(g::AbstractArray, s::Sum, x::AbstractArray)
  _checksize(g, x)
  _checkparams(s, x)
  v = value_and_gradient!(g, first(s.fs), x)
  length(s.fs) == 1 && return v
  b = _array_buffer!(s.scratch, eltype(g), size(g))
  _add_gradients!(g, b, x, v, Base.tail(s.fs))
end

####################################
#  Availability of standalone values
####################################

_has_value(::AbstractFunction) = true
_has_value(c::Conjugate) = _has_conjugate_value(c.f)
_has_value(f::Union{Scaled, Tilt, Translate, SingularValue}) = _has_value(f.f)
_has_value(s::Union{SeparableSum, Sum, SliceSum}) = all(_has_value, s.fs)
_has_value(fb::FiberSum) = _has_value(fb.f) && (fb.diagonal === nothing || _has_value(fb.diagonal))

####################################
#  Lipschitz constants of the gradients
####################################

lipschitz(s::Scaled) = abs(s.α) * lipschitz(s.f)
lipschitz(t::Tilt) = lipschitz(t.f)
lipschitz(t::Translate) = lipschitz(t.f)
lipschitz(s::Sum) = sum(lipschitz, s.fs)
lipschitz(s::SeparableSum) = maximum(lipschitz, s.fs)
