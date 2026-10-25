####################################
#
#  Composition with a linear map, and a ridge term
#
####################################

####################################
#  f(Ax) with AA' = μI
####################################

"""
    Precompose(f, A, μ; check = true)
    Precompose(f, a)

The function `h(x) = f(Ax)` of a vector `x`, for a linear map `A` with
`AA' = μI` and `μ > 0`. Its proximal operator follows from that of `f`:

    prox_{γh}(x) = x + A'(prox_{μγf}(Ax) - Ax) / μ,

and since `A ⋅ prox_{γh}(x) = prox_{μγf}(Ax)`, the value at the result comes
from that of `f`. `A` is anything with `mul!(y, A, x)` and the five-argument
`mul!(y, A', z, α, β)`: a matrix, a sparse matrix or an operator object. For a
matrix, the constructor checks `AA' = μI` unless `check = false`. Rows of the
identity (a selection of entries), permutations, orthogonal transforms and
tight frames qualify. Smooth when `f` is, with gradient `A'∇f(Ax)` and
Lipschitz constant `μ L_f`; proximable and convex when `f` is. When `f` has
second-order oracles, so does `h`, with Hessian product `A'∇²f(Ax)Av`.

With a scalar `a ≠ 0`, `Precompose(f, a)` is `f(ax)` on arrays of any shape
(`μ = a²`), separable when `f` is, with coordinate curvatures scaled by `a²`;
with [`Translate`](@ref) it gives TFOCS's
`f(ax + b)`. The value, the gradient and the proximal operator use buffers that
`h` owns; do not share `h` between threads.
"""
struct Precompose{F<:AbstractFunction, M, T<:Real} <: AbstractFunction
  f::F
  A::M
  μ::T
  image::Scratch        # Ax
  inner::Scratch        # the proximal point or the gradient of f at Ax
  direction::Scratch    # Av for a Hessian-vector product

  function Precompose{F, M, T}(f::F, A::M, μ::T) where {F<:AbstractFunction, M, T<:Real}
    μ > 0 || throw(ArgumentError("μ must be positive, got $μ"))
    new{F, M, T}(f, A, μ, Scratch(), Scratch(), Scratch())
  end
end

function Precompose(f::AbstractFunction, A, μ::Real; check::Bool = true)
  A isa Real && throw(ArgumentError("for a scalar, use Precompose(f, a)"))
  if check && A isa AbstractMatrix
    m = size(A, 1)
    AAt = A * A'
    norm(AAt - μ * I) <= sqrt(eps(float(typeof(μ)))) * μ * max(1, m) ||
      throw(ArgumentError("A A' must be μI; pass check = false to skip this test"))
  end
  Precompose{typeof(f), typeof(A), typeof(μ)}(f, A, μ)
end
function Precompose(f::AbstractFunction, a::Real)
  iszero(a) && throw(ArgumentError("the scalar must be nonzero"))
  Precompose{typeof(f), typeof(a), typeof(abs2(a))}(f, a, abs2(a))
end

const _ScalarPrecompose = Precompose{<:AbstractFunction, <:Real}

for trait in (:issmooth, :isproximable, :isconvex)
  @eval $trait(::Type{<:Precompose{F}}) where {F} = $trait(F)
end
isseparable(::Type{<:Precompose{F, M}}) where {F, M} = M <: Real && isseparable(F)
_isindicator(::Type{<:Precompose{F}}) where {F} = _isindicator(F)

# the image Ax in a buffer that h owns
_image!(h::Precompose, x::AbstractArray) = _linear_image!(h.image, h.A, x)
function _linear_image!(scratch::Scratch, A, x)
  T = float(promote_type(eltype(x), eltype(A)))
  z = _array_buffer!(scratch, T, (size(A, 1),))
  mul!(z, A, x)
end
function _linear_image!(scratch::Scratch, a::Real, x)
  T = float(promote_type(eltype(x), typeof(a)))
  z = _array_buffer!(scratch, T, size(x))
  z .= a .* x
end
_inner_buffer(h::Precompose, z::AbstractArray) = _array_buffer!(h.inner, eltype(z), size(z))

(h::Precompose)(x::AbstractArray) = h.f(_image!(h, x))

function value_and_gradient!(g::AbstractArray, h::Precompose, x::AbstractArray)
  _checksize(g, x)
  z = _image!(h, x)
  b = _inner_buffer(h, z)
  v = value_and_gradient!(b, h.f, z)
  _adjoint_apply!(g, h, b)
  v
end
function gradient!(g::AbstractArray, h::Precompose, x::AbstractArray)
  _checksize(g, x)
  z = _image!(h, x)
  b = _inner_buffer(h, z)
  gradient!(b, h.f, z)
  _adjoint_apply!(g, h, b)
end

# g = A'b
_adjoint_apply!(g, h::Precompose, b) = mul!(g, h.A', b)
_adjoint_apply!(g, h::_ScalarPrecompose, b) = (g .= h.A .* b)

# y = x + A'(P - Ax)/μ, with P - Ax in the buffer; for a scalar a, y = P/a
function prox!(y::AbstractArray, h::Precompose, x::AbstractArray, γ::Real)
  _checksize(y, x)
  _value_and_prox_precompose!(y, h, x, γ, false)
  y
end
value_and_prox!(y::AbstractArray, h::Precompose, x::AbstractArray, γ::Real) =
  (_checksize(y, x); _value_and_prox_precompose!(y, h, x, γ, true))

function _value_and_prox_precompose!(y, h::Precompose, x, γ, withvalue::Bool)
  z = _image!(h, x)
  P = _inner_buffer(h, z)
  T = eltype(z)
  μγ = convert(T, h.μ) * convert(T, γ)
  v = withvalue ? value_and_prox!(P, h.f, z, μγ) : (prox!(P, h.f, z, μγ); zero(T))
  P .-= z
  y === x || copyto!(y, x)
  mul!(y, h.A', P, inv(convert(T, h.μ)), one(T))
  v
end
function _value_and_prox_precompose!(y, h::_ScalarPrecompose, x, γ, withvalue::Bool)
  z = _image!(h, x)
  T = eltype(z)
  μγ = convert(T, h.μ) * convert(T, γ)
  v = withvalue ? value_and_prox!(z, h.f, z, μγ) : (prox!(z, h.f, z, μγ); zero(T))
  y .= z ./ h.A
  v
end

Base.@propagate_inbounds prox_entry(h::_ScalarPrecompose, v::Real, i::Integer, γ::Real) =
  prox_entry(h.f, h.A * v, i, h.μ * γ) / h.A
_checkparams(h::_ScalarPrecompose, x) = _checkparams(h.f, x)
# 0 ∈ gᵢ + a∂fᵢ(0) when 0 ∈ gᵢ/a + ∂fᵢ(0)
kkt_violation(h::_ScalarPrecompose, g::AbstractArray, i::Integer) =
  abs(h.A) * kkt_violation(h.f, _ScaledArray(g, inv(h.A)), i)
restrict(h::_ScalarPrecompose, S::AbstractVector{<:Integer}) = Precompose(restrict(h.f, S), h.A)

lipschitz(h::Precompose) = h.μ * lipschitz(h.f)

####################################
#  f + ρ‖⋅ - c‖²/2
####################################

"""
    Regularize(f, ρ, c = nothing)

The function `f(x) + (ρ/2)‖x - c‖²` for `ρ ≥ 0`, with `c = 0` when omitted, with
the roles of `f`. Its proximal operator is that of `f` with a shorter step, at
a shrunk point,

    prox_{γ(f + ρ‖⋅ - c‖²/2)}(v) = prox_{γ/(1 + γρ) f}((v + γρc) / (1 + γρ)),

and the value at the result comes from that of `f`. Smooth when `f` is, with
gradient `∇f + ρ(x - c)` and Lipschitz constant `L_f + ρ`; separable when `f`
is, so coordinate descent applies. Hessian products and quadratic models are
available when `f` provides them, and coordinate curvatures increase by `ρ`.
On an [`AtomIterate`](@ref) a regularized
group norm updates a group. `Regularize(NormL1(λ), ρ)` is `ElasticNet(λ, ρ)`,
`Regularize(GroupNormL2(λ, groups), ρ)` the group elastic net,
`Regularize(NuclearNorm(λ), ρ)` the matrix elastic net, and with a center `c`
it is the strongly convex subproblem of proximal-point methods.
"""
struct Regularize{F<:AbstractFunction, T<:Real, C<:Union{Nothing, AbstractArray}} <: AbstractFunction
  f::F
  ρ::T
  c::C

  function Regularize{F, T, C}(f::F, ρ::T, c::C) where {F<:AbstractFunction, T<:Real, C<:Union{Nothing, AbstractArray}}
    _checknonneg("ρ", ρ)
    new{F, T, C}(f, ρ, c)
  end
end
Regularize(f::AbstractFunction, ρ::Real, c::Union{Nothing, AbstractArray} = nothing) =
  Regularize{typeof(f), typeof(ρ), typeof(c)}(f, ρ, c)

for trait in (:issmooth, :isproximable, :isseparable, :isconvex)
  @eval $trait(::Type{<:Regularize{F}}) where {F} = $trait(F)
end

_ridge(r::Regularize, x::AbstractArray) =
  convert(float(eltype(x)), r.ρ) * (r.c === nothing ? _sumabs2(x) : _sumabs2_diff(x, r.c)) / 2

(r::Regularize)(x::AbstractArray) = r.f(x) + _ridge(r, x)

function value_and_gradient!(g::AbstractArray, r::Regularize, x::AbstractArray)
  v = value_and_gradient!(g, r.f, x)
  _add_ridge_gradient!(g, r, x)
  v + _ridge(r, x)
end
gradient!(g::AbstractArray, r::Regularize, x::AbstractArray) = (gradient!(g, r.f, x); _add_ridge_gradient!(g, r, x))

function _add_ridge_gradient!(g, r::Regularize, x)
  ρ = convert(eltype(g), r.ρ)
  r.c === nothing ? (g .+= ρ .* x) : (g .+= ρ .* (x .- r.c))
end

# the point (v + γρc)/(1 + γρ) in y, and the step γ/(1 + γρ) of f
function _shrunk!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::Real)
  T = float(eltype(y))
  γρ = convert(T, γ) * convert(T, r.ρ)
  r.c === nothing ? (y .= x ./ (1 + γρ)) : (y .= (x .+ γρ .* r.c) ./ (1 + γρ))
  convert(T, γ) / (1 + γρ)
end

function prox!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::Real)
  _checksize(y, x)
  _checkparam(r.c, x)
  prox!(y, r.f, y, _shrunk!(y, r, x, γ))
end
function value_and_prox!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::Real)
  _checksize(y, x)
  _checkparam(r.c, x)
  v = value_and_prox!(y, r.f, y, _shrunk!(y, r, x, γ))
  v + _ridge(r, y)
end

Base.@propagate_inbounds function prox_entry(r::Regularize, v::Real, i::Integer, γ::Real)
  γρ = γ * r.ρ
  w = r.c === nothing ? v : v + γρ * r.c[i]
  prox_entry(r.f, w / (1 + γρ), i, γ / (1 + γρ))
end
_checkparams(r::Regularize, x) = (_checkparam(r.c, x); _checkparams(r.f, x))

# a regularized group norm on an AtomIterate: shrink the group towards c, then
# take the step of the group norm
function prox_coordinate!(x::AtomIterate, r::Regularize{<:AbstractGroupNorm}, k::Integer, γ::Real)
  T = float(eltype(x))
  γρ = convert(T, γ) * convert(T, r.ρ)
  @inbounds for i in x.groups[k]
    x.storage[i] = r.c === nothing ? x.storage[i] / (1 + γρ) : (x.storage[i] + γρ * r.c[i]) / (1 + γρ)
  end
  prox_coordinate!(x, r.f, k, convert(T, γ) / (1 + γρ))
end

# the ridge term adds ρ(xᵢ - cᵢ) to the gradient, -ρcᵢ at a coordinate at zero
_ridge_shifted(g::AbstractArray, r::Regularize) =
  r.c === nothing ? g : _ShiftedArray(g, _ScaledArray(r.c, -r.ρ))
kkt_violation(r::Regularize, g::AbstractArray, i::Integer) = kkt_violation(r.f, _ridge_shifted(g, r), i)
for X in (:AbstractArray, :AtomIterate, :SymmetricSparseIterate)
  @eval kkt_violation(r::Regularize, g::AbstractArray, x::$X, k::Integer) = kkt_violation(r.f, _ridge_shifted(g, r), x, k)
end
restrict(r::Regularize, S::AbstractVector{<:Integer}) = Regularize(restrict(r.f, S), r.ρ, r.c === nothing ? nothing : r.c[S])

lipschitz(r::Regularize) = lipschitz(r.f) + r.ρ

####################################
#  second-order oracles
####################################

hashessian(::Type{<:Precompose{F}}) where {F} = hashessian(F)
hashessian(::Type{<:Regularize{F}}) where {F} = hashessian(F)

function hessian_vec!(Hv::AbstractArray, h::Precompose, x::AbstractArray, v::AbstractArray)
  _checksize(Hv, x)
  _checkdirection(v, x)
  z = _image!(h, x)
  d = _linear_image!(h.direction, h.A, v)
  b = _inner_buffer(h, z)
  hessian_vec!(b, h.f, z, d)
  _adjoint_apply!(Hv, h, b)
end

function hessian_vec!(Hv::AbstractArray, r::Regularize, x::AbstractArray, v::AbstractArray)
  _checksize(Hv, x)
  _checkdirection(v, x)
  _checkparam(r.c, x)
  hessian_vec!(Hv, r.f, x, v)
  Hv .+= convert(eltype(Hv), r.ρ) .* v
end

function quadratic_model(h::Precompose, x::AbstractArray)
  f = quadratic_model(h.f, _image!(h, x))
  Precompose{typeof(f), typeof(h.A), typeof(h.μ)}(f, h.A, h.μ)
end
quadratic_model(r::Regularize, x) =
  (_checkparam(r.c, x); Regularize(quadratic_model(r.f, x), r.ρ, r.c))

function quadratic_model!(q::Precompose, h::Precompose, x::AbstractArray)
  q.A === h.A && q.μ == h.μ || throw(ArgumentError("the model must come from the same Precompose"))
  quadratic_model!(q.f, h.f, _image!(h, x))
  q
end
function quadratic_model!(q::Regularize, r::Regularize, x)
  q.c === r.c && q.ρ == r.ρ || throw(ArgumentError("the model must come from the same Regularize"))
  _checkparam(r.c, x)
  quadratic_model!(q.f, r.f, x)
  q
end
