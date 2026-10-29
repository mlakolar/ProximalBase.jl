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
`f(ax + b)`.

With a real array `a`, `Precompose(f, a)` is `f(a ⊙ x)`, scaling entry by
entry, including when `a` and `x` are matrices. Use `Precompose(f, A, μ)` for
a linear map. Every entry of `a` must be finite and nonzero, and `size(a)`
must equal `size(x)`. The scale is stored without copying; construct a new
function after changing it. For a separable `f`,

    prox_{γh}(x)ᵢ = prox_entry(f, aᵢxᵢ, i, γaᵢ²) / aᵢ.

Array steps are supported, with `γᵢaᵢ²` in each entry. Coordinate curvatures
are scaled by `aᵢ²`, and the gradient is `a ⊙ ∇f(a ⊙ x)`. Entrywise values
are available when `f` is a [`SeparableFunction`](@ref). Sparse proximal
calls use the separable kernels; symmetric iterates need symmetric scales
and symmetric inner parameters. A group penalty is also supported when `a`
is constant on each group, for values, full-array proximal operators and the
four-argument `kkt_violation`; coordinate updates and restriction of a group
penalty are not supported. Other nonseparable functions are rejected.

The array image uses the promoted floating type of `x` and `a`, and values
have `f`'s type at that image. Effective steps are formed without first
rounding `aᵢ²`. The scaled input, effective step and result must be
representable in the working precision, and the inner oracle must support
their range. The array form provides no Hessian products.

The value, the gradient and the proximal operator use buffers that
`h` owns; do not share `h` between threads.
"""
struct Precompose{F<:AbstractFunction, M, T<:Real} <: AbstractFunction
  f::F
  A::M
  μ::T
  image::Scratch        # Ax
  inner::Scratch        # the proximal point or the gradient of f at Ax
  direction::Scratch    # Av, or Ay for a mixed-precision proximal value

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
_argument_type(::Type{<:Precompose{<:AbstractFunction, <:AbstractMatrix}}) = AbstractVector

function _checkimage(h::Precompose, x)
  _checkargument(h, x)
  h.A isa AbstractMatrix && size(h.A, 2) != length(x) &&
    throw(DimensionMismatch("A has $(size(h.A, 2)) columns and the argument has length $(length(x))"))
  T = float(promote_type(eltype(x), eltype(h.A)))
  _checkbuffer(h.image, T, (size(h.A, 1),))
end
_checkimage(h::_ScalarPrecompose, x) = _checkbuffer(h.image, float(promote_type(eltype(x), typeof(h.A))), size(x))

function _checkwrite(op::O, h::Precompose, y::AbstractArray, x::AbstractArray) where {O}
  _checksize(y, x)
  z = _checkimage(h, x)
  _checkparams(h.f, z)
  b = h isa _ScalarPrecompose && (op === prox! || op === value_and_prox!) ? z :
      _checkbuffer(h.inner, eltype(z), size(z))
  _checkwrite(op, h.f, b, z)
end
_checkvalue(h::Precompose, x::AbstractArray) = _checkvalue(h.f, _checkimage(h, x))

for trait in (:issmooth, :isproximable, :isconvex)
  @eval $trait(::Type{<:Precompose{F}}) where {F} = $trait(F)
end
isseparable(::Type{<:Precompose{F, M}}) where {F, M} = M <: Real && isseparable(F)
iszeropreserving(h::_ScalarPrecompose) = isfinite(h.A) && isfinite(h.μ) && iszeropreserving(h.f)
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
  _checkwrite(value_and_gradient!, h, g, x)
  z = _image!(h, x)
  _checkparams(h.f, z)
  b = _inner_buffer(h, z)
  v = value_and_gradient!(b, h.f, z)
  _adjoint_apply!(g, h, b)
  v
end
function gradient!(g::AbstractArray, h::Precompose, x::AbstractArray)
  _checkwrite(gradient!, h, g, x)
  z = _image!(h, x)
  _checkparams(h.f, z)
  b = _inner_buffer(h, z)
  gradient!(b, h.f, z)
  _adjoint_apply!(g, h, b)
end

# g = A'b
_adjoint_apply!(g, h::Precompose, b) = mul!(g, h.A', b)
_adjoint_apply!(g, h::_ScalarPrecompose, b) = (g .= h.A .* b)

# y = x + A'(P - Ax)/μ, with P - Ax in the buffer; for a scalar a, y = P/a
function prox!(y::AbstractArray, h::Precompose, x::AbstractArray, γ::Real)
  _checkwrite(prox!, h, y, x)
  _value_and_prox_precompose!(y, h, x, γ, false)
  y
end
value_and_prox!(y::AbstractArray, h::Precompose, x::AbstractArray, γ::Real) =
  (_checkwrite(value_and_prox!, h, y, x); _value_and_prox_precompose!(y, h, x, γ, true))

function _value_and_prox_precompose!(y, h::Precompose, x, γ, withvalue::Bool)
  z = _image!(h, x)
  _checkparams(h.f, z)
  P = _inner_buffer(h, z)
  T = eltype(z)
  μγ = convert(T, h.μ) * convert(T, γ)
  v = withvalue ? value_and_prox!(P, h.f, z, μγ) : (prox!(P, h.f, z, μγ); zero(T))
  P .-= z
  y === x || copyto!(y, x)
  mul!(y, h.A', P, inv(convert(T, h.μ)), one(T))
  withvalue && eltype(y) !== eltype(x) && _has_value(h.f) ? _precompose_prox_value(h, y) : v
end
function _value_and_prox_precompose!(y, h::_ScalarPrecompose, x, γ, withvalue::Bool)
  z = _image!(h, x)
  _checkparams(h.f, z)
  T = eltype(z)
  μγ = convert(T, h.μ) * convert(T, γ)
  v = withvalue ? value_and_prox!(z, h.f, z, μγ) : (prox!(z, h.f, z, μγ); zero(T))
  y .= z ./ h.A
  withvalue && eltype(y) !== eltype(x) && _has_value(h.f) ? _precompose_prox_value(h, y) : v
end

function _precompose_prox_value(h::Precompose, y)
  v = h.f(_linear_image!(h.direction, h.A, y))
  _isindicator(h.f) ? zero(v) : v
end

Base.@propagate_inbounds prox_entry(h::_ScalarPrecompose, v::Real, i::Integer, γ::Real) =
  prox_entry(h.f, h.A * v, i, h.μ * γ) / h.A
_checkparams(h::_ScalarPrecompose, x) = _checkparams(h.f, x)
function _checkparams(h::Precompose{<:AbstractFunction, <:AbstractMatrix}, x::AbstractArray)
  size(h.A, 2) == length(x) ||
    throw(DimensionMismatch("A has $(size(h.A, 2)) columns and the argument has length $(length(x))"))
  T = float(promote_type(eltype(x), eltype(h.A)))
  z = _checkbuffer(h.image, T, (size(h.A, 1),))
  _checkparams(h.f, z)
end
# 0 ∈ gᵢ + a∂fᵢ(0) when 0 ∈ gᵢ/a + ∂fᵢ(0)
kkt_violation(h::_ScalarPrecompose, g::AbstractArray, i::Integer) =
  abs(h.A) * kkt_violation(h.f, _ScaledArray(g, inv(h.A)), i)
restrict(h::_ScalarPrecompose, S::AbstractVector{<:Integer}) = Precompose(restrict(h.f, S), h.A)
_restrict_unchecked(h::_ScalarPrecompose, S::AbstractVector{<:Integer}) = Precompose(_restrict_unchecked(h.f, S), h.A)

lipschitz(h::Precompose) = h.μ * lipschitz(h.f)

####################################
#  f(a ⊙ x)
####################################

struct _ArrayPrecompose{F<:AbstractFunction, A<:AbstractArray{<:Real}} <: AbstractFunction
  f::F
  a::A
  image::Scratch
  inner::Scratch
  steps::Scratch
end

function Precompose(f::AbstractFunction, a::AbstractArray{<:Real})
  isseparable(f) || f isa AbstractGroupPenalty ||
    throw(ArgumentError("an array scale requires a separable function or a group penalty with a scale constant on each group"))
  all(v -> isfinite(v) && !iszero(v), a) ||
    throw(ArgumentError("every entry of the scale must be finite and nonzero"))
  if f isa AbstractGroupPenalty
    _checkgroups(f, a)
    for G in f.groups
      c = a[first(G)]
      all(i -> a[i] == c, G) ||
        throw(ArgumentError("the scale must be constant on each group"))
    end
  end
  _ArrayPrecompose(f, a, Scratch(), Scratch(), Scratch())
end
Precompose(::AbstractFunction, ::AbstractArray) =
  throw(ArgumentError("an array scale must have real entries"))

for trait in (:issmooth, :isproximable, :isconvex, :isseparable)
  @eval $trait(::Type{<:_ArrayPrecompose{F}}) where {F} = $trait(F)
end
iszeropreserving(h::_ArrayPrecompose) = iszeropreserving(h.f)
_isindicator(::Type{<:_ArrayPrecompose{F}}) where {F} = _isindicator(F)
hashessian(::Type{<:_ArrayPrecompose}) = false

function check_parameters(h::_ArrayPrecompose, x)
  _checksize(h.a, x)
  _checkparams(h.f, x)
  h.f isa AbstractGroupPenalty && _checkgroups(h.f, x)
  nothing
end

# Direct division avoids an overflowing reciprocal for subnormal scales.
struct _PrecomposeArray{T, N, X<:AbstractArray, A<:AbstractArray, D} <: AbstractArray{T, N}
  x::X
  a::A
end
function _PrecomposeArray(x::AbstractArray{T, N}, a::AbstractArray, ::Val{D}) where {T, N, D}
  S = float(promote_type(T, eltype(a)))
  _PrecomposeArray{S, N, typeof(x), typeof(a), D}(x, a)
end
Base.size(z::_PrecomposeArray) = size(z.x)
Base.IndexStyle(::Type{<:_PrecomposeArray}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(z::_PrecomposeArray{T, N, X, A, D}, i::Int) where {T, N, X, A, D}
  x, a = convert(T, z.x[i]), convert(T, z.a[i])
  D ? x / a : a * x
end

function _image!(h::_ArrayPrecompose, x::AbstractArray)
  T = float(promote_type(eltype(x), eltype(h.a)))
  z = _array_buffer!(h.image, T, size(x))
  copyto!(z, _PrecomposeArray(x, h.a, Val(false)))
end
_inner_buffer(h::_ArrayPrecompose, z::AbstractArray) = _array_buffer!(h.inner, eltype(z), size(z))
(h::_ArrayPrecompose)(x::AbstractArray) = (_checkparams(h, x); h.f(_image!(h, x)))

_sparse_precompose(y, x) = (y isa SparseIterate && x isa SparseIterate) ||
  (y isa SymmetricSparseIterate && x isa AbstractMatrix)

function _checkwrite(op::O, h::_ArrayPrecompose, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}}
  _checksize(y, x)
  _checkparams(h, x)
  if (op === prox! || op === value_and_prox!) && isseparable(h) && _sparse_precompose(y, x)
    return _checkwrite(op, h.f, y, x)
  end
  T = float(promote_type(eltype(x), eltype(h.a)))
  z = _array_buffer!(h.image, T, size(x))
  _checkwrite(op, h.f, _inner_buffer(h, z), z)
end

@inline _precompose_step(γ, a, ::Type{T}) where {T} =
  convert(T, _positive_product(γ, abs(float(a)), abs(float(a))))

function _precompose_steps!(h::_ArrayPrecompose, z, γ)
  T = eltype(z)
  steps = _array_buffer!(h.steps, T, size(z))
  @inbounds for i in eachindex(IndexLinear(), steps)
    steps[i] = _precompose_step(_entry(γ, i), h.a[i], T)
  end
  steps
end
function _precompose_steps!(h::_ArrayPrecompose{<:AbstractGroupPenalty}, z, γ::Real)
  T = eltype(z)
  steps = _array_buffer!(h.steps, T, (length(h.f.groups),))
  @inbounds for k in eachindex(steps)
    steps[k] = _precompose_step(γ, h.a[first(h.f.groups[k])], T)
  end
  steps
end

function _value_and_prox_array!(y, h::_ArrayPrecompose, x, γ, withvalue::Bool)
  if isseparable(h) && _sparse_precompose(y, x)
    _prox_sep!(y, h, x, γ)
    T = float(promote_type(eltype(y), eltype(h.a)))
    return withvalue ? (_isindicator(h) ? zero(T) : h(y)) : zero(T)
  end
  z = _image!(h, x)
  b = _inner_buffer(h, z)
  steps = _precompose_steps!(h, z, γ)
  if h.f isa AbstractGroupPenalty
    v = value_and_prox!(b, h.f, z, steps)
  else
    _prox_sep!(b, h.f, z, steps)
    v = withvalue ? (_isindicator(h) ? zero(eltype(z)) : h.f(b)) : zero(eltype(z))
  end
  # Unchanged image entries include free indices and zero-weight groups.
  @inbounds for i in eachindex(IndexLinear(), y, x)
    y[i] = b[i] == z[i] ? x[i] : b[i] / h.a[i]
  end
  withvalue && eltype(y) !== eltype(x) && _has_value(h.f) ? _array_precompose_value(h, y) : v
end

# In mixed precision, the value at the output, through the lazy image of y
function _array_precompose_value(h::_ArrayPrecompose, y)
  v = h.f(_PrecomposeArray(y, h.a, Val(false)))
  _isindicator(h.f) ? zero(v) : v
end

function prox!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::Real)
  _checkwrite(prox!, h, y, x)
  _value_and_prox_array!(y, h, x, γ, false)
  y
end
function value_and_prox!(y::AbstractArray, h::_ArrayPrecompose, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, h, y, x)
  _value_and_prox_array!(y, h, x, γ, true)
end

Base.@propagate_inbounds function prox_entry(h::_ArrayPrecompose, v::Real, i::Integer, γ::Real)
  isseparable(h) || _array_group_operations()
  T = float(promote_type(typeof(v), eltype(h.a)))
  a = convert(T, h.a[i])
  z = a * convert(T, v)
  p = convert(T, prox_entry(h.f, z, i, _precompose_step(γ, a, T)))
  p == z ? convert(T, v) : p / a
end
Base.@propagate_inbounds function value_entry(h::_ArrayPrecompose{<:SeparableFunction}, v::Real, i::Integer)
  T = float(promote_type(typeof(v), eltype(h.a)))
  value_entry(h.f, convert(T, h.a[i]) * convert(T, v), i)
end

function gradient!(g::AbstractArray, h::_ArrayPrecompose, x::AbstractArray)
  _checkwrite(gradient!, h, g, x)
  z = _image!(h, x)
  b = _inner_buffer(h, z)
  gradient!(b, h.f, z)
  g .= h.a .* b
end
function value_and_gradient!(g::AbstractArray, h::_ArrayPrecompose, x::AbstractArray)
  _checkwrite(value_and_gradient!, h, g, x)
  z = _image!(h, x)
  b = _inner_buffer(h, z)
  v = value_and_gradient!(b, h.f, z)
  g .= h.a .* b
  v
end

function kkt_violation(h::_ArrayPrecompose, g::AbstractArray, i::Integer)
  isseparable(h) || _array_group_operations()
  _checkparams(h, g)
  abs(float(h.a[i])) * kkt_violation(h.f, _PrecomposeArray(g, h.a, Val(true)), i)
end
function _array_group_kkt(h, g, x, i)
  _checkparams(h, g)
  _checkparams(h, x)
  checkbounds(x, i)
  _group(h.f, i) == 0 && return abs(g[i])
  c = abs(float(h.a[i]))
  c * kkt_violation(h.f, _PrecomposeArray(g, h.a, Val(true)), _PrecomposeArray(x, h.a, Val(false)), i)
end
kkt_violation(h::_ArrayPrecompose{<:AbstractGroupPenalty}, g::AbstractArray, x::AbstractArray, i::Integer) =
  _array_group_kkt(h, g, x, i)
kkt_violation(::_ArrayPrecompose{<:AbstractGroupPenalty}, ::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  _array_group_operations()
function kkt_violation(h::_ArrayPrecompose{<:AbstractGroupPenalty}, g::AbstractArray, x::AtomIterate, k::Integer)
  _checkparams(h, g)
  _checkparams(h, x)
  _checksamegroups(h.f, x)
  checkbounds(x.groups, k)
  c = abs(float(h.a[first(x.groups[k])]))
  c * kkt_violation(h.f, _PrecomposeArray(g, h.a, Val(true)), x, k)
end

@noinline _array_group_operations() = throw(ArgumentError("an array-scaled group penalty supports values, full-array proximal operators and four-argument kkt_violation on arrays; coordinate updates and restrict are not supported"))
prox_coordinate!(::AtomIterate, ::_ArrayPrecompose{<:AbstractGroupPenalty}, ::Integer, ::Real) =
  _array_group_operations()
function restrict(h::_ArrayPrecompose, S::AbstractVector{<:Integer})
  isseparable(h) || _array_group_operations()
  Precompose(restrict(h.f, S), h.a[S])
end
function _restrict_unchecked(h::_ArrayPrecompose, S::AbstractVector{<:Integer})
  isseparable(h) || _array_group_operations()
  Precompose(_restrict_unchecked(h.f, S), h.a[S])
end

function lipschitz(h::_ArrayPrecompose)
  L = lipschitz(h.f)
  isempty(h.a) && return zero(promote_type(typeof(float(L)), float(eltype(h.a))))
  a = _maximum_abs(h.a)
  _positive_product(L, a, a)
end

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
iszeropreserving(r::Regularize) = r.c === nothing && isfinite(r.ρ) && iszeropreserving(r.f)

_ridge(r::Regularize, x::AbstractArray) =
  convert(float(eltype(x)), r.ρ) * (r.c === nothing ? _sumabs2(x) : _sumabs2_diff(x, r.c)) / 2

(r::Regularize)(x::AbstractArray) = (_checkparams(r, x); r.f(x) + _ridge(r, x))
function _checkvalue(r::Regularize, x::AbstractArray)
  _checkparams(r, x)
  _checkvalue(r.f, x)
  r.c === nothing || _checksize(x, r.c)
  true
end

function value_and_gradient!(g::AbstractArray, r::Regularize, x::AbstractArray)
  _checkwrite(value_and_gradient!, r, g, x)
  v = value_and_gradient!(g, r.f, x)
  _add_ridge_gradient!(g, r, x)
  v + _ridge(r, x)
end
gradient!(g::AbstractArray, r::Regularize, x::AbstractArray) =
  (_checkwrite(gradient!, r, g, x); gradient!(g, r.f, x); _add_ridge_gradient!(g, r, x))

function _checkwrite(op::O, r::Regularize, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}}
  _checksize(y, x)
  _checkparams(r, x)
  r.c === nothing || Base.Broadcast.check_broadcast_axes(axes(y), x, r.c)
  if op === value_and_prox! || op === value_and_gradient!
    r.c === nothing || size(x) == size(r.c) ||
      throw(DimensionMismatch("the arrays have sizes $(size(x)) and $(size(r.c))"))
  end
  _checkwrite(op, r.f, y, op === prox! || op === value_and_prox! ? y : x)
end

function _add_ridge_gradient!(g, r::Regularize, x)
  ρ = convert(eltype(g), r.ρ)
  r.c === nothing ? (g .+= ρ .* x) : (g .+= ρ .* (x .- r.c))
end

# the point (v + γρc)/(1 + γρ) in y, and the step γ/(1 + γρ) of f
@inline function _regularize_step(γ::T, ρ::T) where {T}
  product = γ * ρ
  isfinite(product) ? γ / (one(T) + product) : inv(ρ) / (one(T) + inv(γ) / ρ)
end
function _shrunk!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::Real)
  T = float(eltype(y))
  step, ρ = convert(T, γ), convert(T, r.ρ)
  a, b = _ridge_weights(step, ρ)
  r.c === nothing ? (y .= a .* x) : (y .= _ridge_blend.(x, r.c, a, b))
  _regularize_step(step, ρ)
end

function prox!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::Real)
  _checkwrite(prox!, r, y, x)
  prox!(y, r.f, y, _shrunk!(y, r, x, γ))
end
function value_and_prox!(y::AbstractArray, r::Regularize, x::AbstractArray, γ::Real)
  _checkwrite(value_and_prox!, r, y, x)
  v = value_and_prox!(y, r.f, y, _shrunk!(y, r, x, γ))
  v + _ridge(r, y)
end

Base.@propagate_inbounds function prox_entry(r::Regularize, v::Real, i::Integer, γ::Real)
  T = float(typeof(v))
  step, ρ = convert(T, γ), convert(T, r.ρ)
  a, b = _ridge_weights(step, ρ)
  w = r.c === nothing ? a * v : _ridge_blend(v, r.c[i], a, b)
  prox_entry(r.f, w, i, _regularize_step(step, ρ))
end
_checkparams(r::Regularize, x) = (_checkparam(r.c, x); _checkparams(r.f, x))

# a regularized group penalty on an AtomIterate: shrink the group towards c,
# then take the step of the group penalty
function prox_coordinate!(x::AtomIterate, r::Regularize{<:AbstractGroupPenalty}, k::Integer, γ::Real)
  _checkparams(r, x)
  _checksamegroups(r.f, x)
  checkbounds(x.groups, k)
  T = float(eltype(x))
  step, ρ = convert(T, γ), convert(T, r.ρ)
  a, b = _ridge_weights(step, ρ)
  @inbounds for i in x.groups[k]
    x.storage[i] = r.c === nothing ? a * x.storage[i] : _ridge_blend(x.storage[i], r.c[i], a, b)
  end
  prox_coordinate!(x, r.f, k, _regularize_step(step, ρ))
end

# the ridge term adds ρ(xᵢ - cᵢ) to the gradient, -ρcᵢ at a coordinate at zero
_ridge_shifted(g::AbstractArray, r::Regularize) =
  r.c === nothing ? g : _ShiftedArray(g, _ScaledArray(r.c, -r.ρ))
kkt_violation(r::Regularize, g::AbstractArray, i::Integer) = kkt_violation(r.f, _ridge_shifted(g, r), i)
for X in (:AbstractArray, :AtomIterate, :SymmetricSparseIterate)
  @eval kkt_violation(r::Regularize, g::AbstractArray, x::$X, k::Integer) = kkt_violation(r.f, _ridge_shifted(g, r), x, k)
end
restrict(r::Regularize, S::AbstractVector{<:Integer}) = Regularize(restrict(r.f, S), r.ρ, r.c === nothing ? nothing : r.c[S])
_restrict_unchecked(r::Regularize, S::AbstractVector{<:Integer}) =
  Regularize(_restrict_unchecked(r.f, S), r.ρ, r.c === nothing ? nothing : r.c[S])

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
  _checkparams(h.f, z)
  d = _linear_image!(h.direction, h.A, v)
  b = _inner_buffer(h, z)
  hessian_vec!(b, h.f, z, d)
  _adjoint_apply!(Hv, h, b)
end

function hessian_vec!(Hv::AbstractArray, r::Regularize, x::AbstractArray, v::AbstractArray)
  _checksize(Hv, x)
  _checkdirection(v, x)
  _checkparams(r, x)
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

# These wrappers, defined after calculus.jl, also evaluate the inner value.
_has_value(f::Union{Precompose, _ArrayPrecompose, Regularize, Spectral, BoxConstrained, LinearConstrained,
                   BallConstrained, SphereConstrained, AffineConstrained}) = _has_value(f.f)
