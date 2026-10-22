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
"""
struct Scaled{F<:AbstractFunction, T<:Real} <: AbstractFunction
  f::F
  α::T

  function Scaled{F, T}(f::F, α::T) where {F<:AbstractFunction, T<:Real}
    _checknonneg("α", α)
    new{F, T}(f, α)
  end
end
Scaled(f::AbstractFunction, α::Real) = Scaled{typeof(f), typeof(α)}(f, α)
Scaled(f::Scaled, α::Real) = Scaled(f.f, f.α * α)

Base.:*(α::Real, f::AbstractFunction) = Scaled(f, α)
Base.:*(f::AbstractFunction, α::Real) = Scaled(f, α)

for trait in (:issmooth, :isproximable, :isseparable, :isconvex)
  @eval $trait(::Type{<:Scaled{F}}) where {F} = $trait(F)
end

(s::Scaled)(x) = s.α * s.f(x)
value_and_gradient!(g, s::Scaled, x) = (v = value_and_gradient!(g, s.f, x); g .*= s.α; s.α * v)
gradient!(g, s::Scaled, x) = (gradient!(g, s.f, x); g .*= s.α)
prox!(y, s::Scaled, x, γ::Real) = prox!(y, s.f, x, s.α * γ)
value_and_prox!(y, s::Scaled, x, γ::Real) = s.α * value_and_prox!(y, s.f, x, s.α * γ)
cdprox(s::Scaled, v::Real, i::Integer, γ::Real) = cdprox(s.f, v, i, s.α * γ)
kkt_violation(s::Scaled, g::AbstractArray, i::Integer) =
  iszero(s.α) ? abs(g[i]) : s.α * kkt_violation(s.f, _ScaledArray(g, inv(s.α)), i)
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
value_and_gradient!(g, t::Tilt, x) = (v = value_and_gradient!(g, t.f, x); g .+= t.c; v + dot(t.c, x) + t.d)
gradient!(g, t::Tilt, x) = (gradient!(g, t.f, x); g .+= t.c)

function prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::Real)
  _checksize(y, x)
  y .= x .- convert(float(eltype(y)), γ) .* t.c
  prox!(y, t.f, y, γ)
end
function value_and_prox!(y::AbstractArray, t::Tilt, x::AbstractArray, γ::Real)
  _checksize(y, x)
  y .= x .- convert(float(eltype(y)), γ) .* t.c
  value_and_prox!(y, t.f, y, γ) + dot(t.c, y) + t.d
end
cdprox(t::Tilt, v::Real, i::Integer, γ::Real) = cdprox(t.f, v - γ * t.c[i], i, γ)
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

# x - b in a buffer owned by t
function _shifted(t::Translate, x::AbstractArray)
  size(x) == size(t.b) || throw(DimensionMismatch("x has size $(size(x)) and b $(size(t.b))"))
  z = _array_buffer!(t.scratch, float(promote_type(eltype(x), eltype(t.b))), size(x))
  z .= x .- t.b
end

(t::Translate)(x::AbstractArray) = t.f(_shifted(t, x))
value_and_gradient!(g, t::Translate, x::AbstractArray) = value_and_gradient!(g, t.f, _shifted(t, x))
gradient!(g, t::Translate, x::AbstractArray) = gradient!(g, t.f, _shifted(t, x))

function prox!(y::AbstractArray, t::Translate, x::AbstractArray, γ::Real)
  _checksize(y, x)
  y .= x .- t.b
  prox!(y, t.f, y, γ)
  y .+= t.b
end
function value_and_prox!(y::AbstractArray, t::Translate, x::AbstractArray, γ::Real)
  _checksize(y, x)
  y .= x .- t.b
  v = value_and_prox!(y, t.f, y, γ)
  y .+= t.b
  v
end
cdprox(t::Translate, v::Real, i::Integer, γ::Real) = t.b[i] + cdprox(t.f, v - t.b[i], i, γ)
restrict(t::Translate, S::AbstractVector{<:Integer}) = Translate(restrict(t.f, S), t.b[S])

####################################
#  convex conjugate
####################################

"""
    Conjugate(f)

The convex conjugate `f*(y) = supₓ ⟨x, y⟩ - f(x)` of a convex `f`. Its proximal
operator follows from that of `f` by Moreau's identity,
`prox_{γf*}(x) = x - γ prox_{f/γ}(x/γ)`, so every proximable function gives one;
for a separable `f` so does [`cdprox`](@ref). Its value is known for the norms
(the indicators of the dual balls), the balls (their support functions), the
box, the nonnegative orthant, [`SqrNormL2`](@ref), [`LinearFunction`](@ref) and
[`ZeroFunction`](@ref).
"""
struct Conjugate{F<:AbstractFunction} <: AbstractFunction
  f::F
  scratch::Scratch
end
Conjugate(f::AbstractFunction) = Conjugate(f, Scratch())

isproximable(::Type{<:Conjugate{F}}) where {F} = isproximable(F)
isseparable(::Type{<:Conjugate{F}}) where {F} = isseparable(F)
isconvex(::Type{<:Conjugate}) = true

function prox!(y::AbstractArray, c::Conjugate, x::AbstractArray, γ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  γ = convert(T, γ)
  z = _array_buffer!(c.scratch, T, size(x))
  z .= x ./ γ
  prox!(z, c.f, z, inv(γ))
  y .= x .- γ .* z
end
cdprox(c::Conjugate, v::Real, i::Integer, γ::Real) = v - γ * cdprox(c.f, v / γ, i, inv(γ))

(c::Conjugate)(y::AbstractArray) = _conjugate_value(c.f, y)

@noinline _conjugate_value(f::AbstractFunction, y) =
  throw(ArgumentError("the value of the conjugate of $(typeof(f)) is not implemented; its proximal operator is"))

# indicators of the dual balls and support functions of the balls, within the
# tolerance of the indicator functions; loops rather than closures, which would
# capture the element type as a DataType and dispatch dynamically
_ind(ok::Bool, ::Type{T}) where {T} = ok ? zero(T) : T(Inf)
_within(a, b, ::Type{T}) where {T} = a <= b + _tol(T) * max(one(T), abs(b))

function _conjugate_value(f::NormL1, y::AbstractArray)
  T = float(eltype(y))
  @inbounds for i in eachindex(IndexLinear(), y)
    _within(abs(y[i]), f.λ0 * _entry(f.w, i), T) || return T(Inf)
  end
  zero(T)
end
_conjugate_value(f::NormL2, y::AbstractArray) = (T = float(eltype(y)); _ind(_within(sqrt(_sumabs2(y)), f.λ, T), T))
_conjugate_value(f::NormLinf, y::AbstractArray) = (T = float(eltype(y)); _ind(_within(norm(y, 1), f.λ, T), T))
_conjugate_value(f::IndBallL1, y::AbstractArray) = convert(float(eltype(y)), f.r) * norm(y, Inf)
_conjugate_value(f::IndBallL2, y::AbstractArray) = convert(float(eltype(y)), f.r) * sqrt(_sumabs2(y))
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
  T = float(eltype(y))
  s = zero(T)
  @inbounds for i in eachindex(IndexLinear(), y)
    v = y[i]
    s += v >= 0 ? (iszero(v) ? zero(T) : v * _entry(f.u, i)) : v * _entry(f.l, i)
  end
  s
end
function _conjugate_value(f::SqrNormL2, y::AbstractArray)
  T = float(eltype(y))
  v = convert(T, _sumabs2(y)) / (2 * convert(T, f.λ))
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

(s::SeparableSum)(x::Tuple) = (_checkblocks(s, x); sum(map((f, xk) -> f(xk), s.fs, x)))
prox!(y::Tuple, s::SeparableSum, x::Tuple, γ::Real) =
  (_checkblocks(s, x); map((f, yk, xk) -> prox!(yk, f, xk, γ), s.fs, y, x); y)
value_and_prox!(y::Tuple, s::SeparableSum, x::Tuple, γ::Real) =
  (_checkblocks(s, x); sum(map((f, yk, xk) -> value_and_prox!(yk, f, xk, γ), s.fs, y, x)))
prox(s::SeparableSum, x::Tuple, γ::Real = 1) = prox!(map(xk -> similar(xk, float(eltype(xk))), x), s, x, γ)
value_and_gradient!(g::Tuple, s::SeparableSum, x::Tuple) =
  (_checkblocks(s, x); sum(map((f, gk, xk) -> value_and_gradient!(gk, f, xk), s.fs, g, x)))
gradient!(g::Tuple, s::SeparableSum, x::Tuple) =
  (_checkblocks(s, x); map((f, gk, xk) -> gradient!(gk, f, xk), s.fs, g, x); g)

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

(s::Sum)(x) = sum(f -> f(x), s.fs)

# the remaining functions' gradients added into g through the buffer b
_add_gradients!(g, b, x, v, ::Tuple{}) = v
function _add_gradients!(g, b, x, v, fs::Tuple)
  w = value_and_gradient!(b, first(fs), x)
  g .+= b
  _add_gradients!(g, b, x, v + w, Base.tail(fs))
end

function value_and_gradient!(g::AbstractArray, s::Sum, x::AbstractArray)
  v = value_and_gradient!(g, first(s.fs), x)
  length(s.fs) == 1 && return v
  b = _array_buffer!(s.scratch, eltype(g), size(g))
  _add_gradients!(g, b, x, v, Base.tail(s.fs))
end
