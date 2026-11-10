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

    prox_{γh}(x)ᵢ = prox_block(f, aᵢxᵢ, i, γaᵢ²) / aᵢ.

Array steps are supported, with `γᵢaᵢ²` in each entry. Coordinate curvatures
are scaled by `aᵢ²`, and the gradient is `a ⊙ ∇f(a ⊙ x)`. Entrywise values
are available when `f` is a [`SeparableFunction`](@ref). Sparse proximal
calls use the separable kernels; symmetric iterates need symmetric scales
and symmetric inner parameters. A group penalty is also supported when `a`
is constant on each group, for values, full-array proximal operators and the
four-argument `kkt_violation`; coordinate updates and restriction of a group
penalty are not supported. Other nonseparable functions are rejected.

The array image uses the promoted floating type of `x` and `a`; values are
rounded to the argument's type, as everywhere in the package (see the
Precision section of the manual). Effective steps are formed without first
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
    eltype(A) <: Real || _real_error(:Precompose)
    μ > 0 || throw(ArgumentError("μ must be positive, got $μ"))
    new{F, M, T}(f, A, μ, Scratch(), Scratch(), Scratch())
  end
end

function Precompose(f::AbstractFunction, A, μ::Real; check::Bool = true)
  eltype(A) <: Real || _real_error(:Precompose)
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

function _checkwrite(op::O, h::Precompose, y::_ArrayOrOwner, x::_ArrayOrOwner) where {O}
  _checkreal(h, y); _checkreal(h, x)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  z = _checkimage(h, x)
  _checkparams(h.f, z)
  b = h isa _ScalarPrecompose && (op === prox! || op === value_and_prox!) ? z :
      _checkbuffer(h.inner, eltype(z), size(z))
  _checkwrite(op, h.f, b, z)
end
_checkvalue(h::Precompose, x::_ArrayOrOwner) = (_checkreal(h, x); _checkvalue(h.f, _checkimage(h, x)))

for trait in (:issmooth, :isproximable, :isconvex)
  @eval $trait(::Type{<:Precompose{F}}) where {F} = $trait(F)
end
isseparable(::Type{<:Precompose{F, M}}) where {F, M} = M <: Real && isseparable(F)
iszeropreserving(h::_ScalarPrecompose) = isfinite(h.A) && isfinite(h.μ) && iszeropreserving(h.f)
_isindicator(::Type{<:Precompose{F}}) where {F} = _isindicator(F)

# the image Ax in a buffer that h owns
_image!(h::Precompose, x::AbstractArray) = _linear_image!(h.image, h.A, x)
# the scalar form of an owner: the image is the dense matrix a x, which is symmetric; a matrix operator
# does not take an owner (its argument is a vector)
_image!(h::_ScalarPrecompose, x::SymmetricSparseIterate) = _linear_image!(h.image, h.A, matrixview(x))
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
_linear_image!(scratch::Scratch, a::Real, x::SymmetricSparseIterate) = _linear_image!(scratch, a, matrixview(x))
_inner_buffer(h::Precompose, z::AbstractArray) = _array_buffer!(h.inner, eltype(z), size(z))

(h::Precompose)(x::_ArrayOrOwner) =
  (_check_value_arrays(h, x); _value_checked(h, x))
_value_unchecked(h::Precompose, x::_ArrayOrOwner) = _value_at(_value_unchecked(h.f, _image!(h, x)), x)

value_and_gradient!(g::_ArrayOrOwner, h::Precompose, x::_ArrayOrOwner) =
  (_check_oracle_arrays(value_and_gradient!, h, g, x); _value_and_gradient_unchecked!(g, h, x))
function _value_and_gradient_unchecked!(g::_ArrayOrOwner, h::Precompose, x::_ArrayOrOwner)
  _checkwrite(value_and_gradient!, h, g, x)
  z = _image!(h, x)
  _checkparams(h.f, z)
  b = _inner_buffer(h, z)
  v = _value_and_gradient_unchecked!(b, h.f, z)
  _adjoint_apply!(g, h, b)
  _value_at(v, x)
end
gradient!(g::_ArrayOrOwner, h::Precompose, x::_ArrayOrOwner) =
  (_check_oracle_arrays(gradient!, h, g, x); _gradient_unchecked!(g, h, x))
function _gradient_unchecked!(g::_ArrayOrOwner, h::Precompose, x::_ArrayOrOwner)
  _checkwrite(gradient!, h, g, x)
  z = _image!(h, x)
  _checkparams(h.f, z)
  b = _inner_buffer(h, z)
  _gradient_unchecked!(b, h.f, z)
  _adjoint_apply!(g, h, b)
end

# g = A'b
_adjoint_apply!(g, h::Precompose, b) = mul!(g, h.A', b)
_adjoint_apply!(g, h::_ScalarPrecompose, b) = (g .= h.A .* b)
# an owner receives the symmetric part of a b, one coordinate at a time
_adjoint_apply!(g::SymmetricSparseIterate, h::_ScalarPrecompose, b) = _scale_coordinates!(g, h.A, b)
function _scale_coordinates!(g::SymmetricSparseIterate, a, b)
  p = g.p
  @inbounds for c in 1:p, r in c:p
    v = a * b[r, c]
    r == c || (v = _symmetric_pair(v, a * b[c, r], r, c))
    _setstored!(g, v, tril_index(p, r, c))
  end
  g
end

# y = x + A'(P - Ax)/μ, with P - Ax in the buffer; for a scalar a, y = P/a
prox!(y::_ArrayOrOwner, h::Precompose, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, h, y, x, γ); _prox_unchecked!(y, h, x, γ))
function _prox_unchecked!(y::_ArrayOrOwner, h::Precompose, x::_ArrayOrOwner, γ::Real)
  _checkwrite(prox!, h, y, x)
  _value_and_prox_precompose!(y, h, x, γ, false)
  y
end
value_and_prox!(y::_ArrayOrOwner, h::Precompose, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, h, y, x, γ); _value_and_prox_unchecked!(y, h, x, γ))
_value_and_prox_unchecked!(y::_ArrayOrOwner, h::Precompose, x::_ArrayOrOwner, γ::Real) =
  (_checkwrite(value_and_prox!, h, y, x); _value_at(_value_and_prox_precompose!(y, h, x, γ, true), y))

function _value_and_prox_precompose!(y, h::Precompose, x, γ, withvalue::Bool)
  z = _image!(h, x)
  _checkparams(h.f, z)
  P = _inner_buffer(h, z)
  T = eltype(z)
  μγ = convert(T, h.μ) * convert(T, γ)
  v = withvalue ? _value_and_prox_unchecked!(P, h.f, z, μγ) : (_prox_unchecked!(P, h.f, z, μγ); zero(T))
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
  v = withvalue ? _value_and_prox_unchecked!(z, h.f, z, μγ) : (_prox_unchecked!(z, h.f, z, μγ); zero(T))
  _check_staged_result(y, z, h.f)
  yy = _ow(y)
  yy .= z ./ h.A
  withvalue && eltype(y) !== eltype(x) && _has_value(h.f) ? _precompose_prox_value(h, y) : v
end

function _precompose_prox_value(h::Precompose, y)
  v = _value_unchecked(h.f, _linear_image!(h.direction, h.A, y))
  _isindicator(h.f) ? zero(v) : v
end

Base.@propagate_inbounds prox_block(h::_ScalarPrecompose, v::Real, i::Integer, γ::Real) =
  prox_block(h.f, h.A * v, i, h.μ * γ) / h.A
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
  (eltype(g) <: Real || _real_error(:kkt_violation); abs(h.A) * kkt_violation(h.f, _ScaledArray(g, inv(h.A)), i))
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
  _real_error(:Precompose)

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
_image!(h::_ArrayPrecompose, x::SymmetricSparseIterate) = _image!(h, matrixview(x))
_inner_buffer(h::_ArrayPrecompose, z::AbstractArray) = _array_buffer!(h.inner, eltype(z), size(z))
(h::_ArrayPrecompose)(x::_ArrayOrOwner) =
  (_check_value_arrays(h, x); _value_checked(h, x))
_value_unchecked(h::_ArrayPrecompose, x::_ArrayOrOwner) = (_checkparams(h, x); _value_at(_value_unchecked(h.f, _image!(h, x)), x))

_sparse_precompose(y, x) = (y isa SparseIterate && x isa SparseIterate) ||
  (y isa SymmetricSparseIterate && (x isa AbstractMatrix || x isa SymmetricSparseIterate))

function _checkwrite(op::O, h::_ArrayPrecompose, y::_ArrayOrOwner, x::_ArrayOrOwner) where {O<:Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}}
  _checkreal(h, y); _checkreal(h, x)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(h, x)
  _check_parameter_axes(y, h.a)
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

# a dense output reads the matrix of an owner point, whose parameters have been checked
_dense_point(y::AbstractArray, x::SymmetricSparseIterate) = matrixview(x)
_dense_point(y, x) = x
function _value_and_prox_array!(y, h::_ArrayPrecompose, x0, γ, withvalue::Bool)
  x = _dense_point(y, x0)
  # an inner function without a standalone value gives it with its proximal point
  joint = withvalue && !_isindicator(h) && !_has_value(h.f)
  if isseparable(h) && _sparse_precompose(y, x)
    T = float(promote_type(eltype(y), eltype(h.a)))
    # from the dense image of what the sparse loop reads, before y is written
    v = joint ? _array_precompose_joint_value(h, y isa SymmetricSparseIterate ? Symmetric(_vw(x), :L) : x, γ) : zero(T)
    _prox_sep!(y, h, x, γ)
    return withvalue && !joint ? (_isindicator(h) ? zero(T) : _value_unchecked(h, y)) : v
  end
  z = _image!(h, x)
  b = _inner_buffer(h, z)
  steps = _precompose_steps!(h, z, γ)
  if h.f isa AbstractGroupPenalty || joint
    v = _value_and_prox_unchecked!(b, h.f, z, steps)
  else
    _prox_sep!(b, h.f, z, steps)
    v = withvalue ? (_isindicator(h) ? zero(eltype(z)) : _value_unchecked(h.f, b)) : zero(eltype(z))
  end
  # Unchanged image entries include free indices and zero-weight groups.
  @inbounds for i in eachindex(IndexLinear(), y, x)
    y[i] = b[i] == z[i] ? x[i] : b[i] / h.a[i]
  end
  withvalue && eltype(y) !== eltype(x) && _has_value(h.f) ? _array_precompose_value(h, y) : v
end

function _array_precompose_joint_value(h::_ArrayPrecompose, x, γ)
  z = _image!(h, x)
  _value_and_prox_unchecked!(_inner_buffer(h, z), h.f, z, _precompose_steps!(h, z, γ))
end

# In mixed precision, the value at the output, through the lazy image of y
function _array_precompose_value(h::_ArrayPrecompose, y)
  v = _value_unchecked(h.f, _PrecomposeArray(y, h.a, Val(false)))
  _isindicator(h.f) ? zero(v) : v
end

prox!(y::_ArrayOrOwner, h::_ArrayPrecompose, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, h, y, x, γ); _prox_unchecked!(y, h, x, γ))
function _prox_unchecked!(y::_ArrayOrOwner, h::_ArrayPrecompose, x::_ArrayOrOwner, γ::Real)
  _checkwrite(prox!, h, y, x)
  _value_and_prox_array!(y, h, x, γ, false)
  y
end
value_and_prox!(y::_ArrayOrOwner, h::_ArrayPrecompose, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, h, y, x, γ); _value_and_prox_unchecked!(y, h, x, γ))
function _value_and_prox_unchecked!(y::_ArrayOrOwner, h::_ArrayPrecompose, x::_ArrayOrOwner, γ::Real)
  _checkwrite(value_and_prox!, h, y, x)
  _value_at(_value_and_prox_array!(y, h, x, γ, true), y)
end

Base.@propagate_inbounds function prox_block(h::_ArrayPrecompose, v::Real, i::Integer, γ::Real)
  isseparable(h) || _array_group_operations()
  T = float(promote_type(typeof(v), eltype(h.a)))
  a = convert(T, h.a[i])
  z = a * convert(T, v)
  p = convert(T, prox_block(h.f, z, i, _precompose_step(γ, a, T)))
  p == z ? convert(T, v) : p / a
end
Base.@propagate_inbounds function value_entry(h::_ArrayPrecompose{<:SeparableFunction}, v::Real, i::Integer)
  T = float(promote_type(typeof(v), eltype(h.a)))
  value_entry(h.f, convert(T, h.a[i]) * convert(T, v), i)
end

gradient!(g::_ArrayOrOwner, h::_ArrayPrecompose, x::_ArrayOrOwner) =
  (_check_oracle_arrays(gradient!, h, g, x); _gradient_unchecked!(g, h, x))
function _gradient_unchecked!(g::_ArrayOrOwner, h::_ArrayPrecompose, x::_ArrayOrOwner)
  _checkwrite(gradient!, h, g, x)
  z = _image!(h, x)
  b = _inner_buffer(h, z)
  _gradient_unchecked!(b, h.f, z)
  _scale_gradient!(g, h.a, b)
end
value_and_gradient!(g::_ArrayOrOwner, h::_ArrayPrecompose, x::_ArrayOrOwner) =
  (_check_oracle_arrays(value_and_gradient!, h, g, x); _value_and_gradient_unchecked!(g, h, x))
function _value_and_gradient_unchecked!(g::_ArrayOrOwner, h::_ArrayPrecompose, x::_ArrayOrOwner)
  _checkwrite(value_and_gradient!, h, g, x)
  z = _image!(h, x)
  b = _inner_buffer(h, z)
  v = _value_and_gradient_unchecked!(b, h.f, z)
  _scale_gradient!(g, h.a, b)
  _value_at(v, x)
end
# g = a ⊙ b; an owner takes the symmetric part, from the lower entries of a symmetric a
_scale_gradient!(g::AbstractArray, a, b) = (g .= a .* b)
function _scale_gradient!(g::SymmetricSparseIterate, a, b)
  p = g.p
  @inbounds for c in 1:p, r in c:p
    v = a[r, c] * b[r, c]
    r == c || (v = _symmetric_pair(v, a[c, r] * b[c, r], r, c))
    _setstored!(g, v, tril_index(p, r, c))
  end
  g
end

function kkt_violation(h::_ArrayPrecompose, g::AbstractArray, i::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
  isseparable(h) || _array_group_operations()
  _checkparams(h, g)
  abs(float(h.a[i])) * kkt_violation(h.f, _PrecomposeArray(g, h.a, Val(true)), i)
end
function _array_group_kkt(h, g, x, i)
  eltype(g) <: Real && eltype(x) <: Real || _real_error(:kkt_violation)
  _checkparams(h, g)
  _checkparams(h, x)
  checkbounds(x, i)
  _group(h.f, i) == 0 && return abs(g[i])
  c = abs(float(h.a[i]))
  c * kkt_violation(h.f, _PrecomposeArray(g, h.a, Val(true)), _PrecomposeArray(x, h.a, Val(false)), i)
end
_kkt_violation(h::_ArrayPrecompose{<:AbstractGroupPenalty}, g::AbstractArray, x::AbstractArray, i::Integer) =
  _array_group_kkt(h, g, x, i)
_kkt_violation(::_ArrayPrecompose{<:AbstractGroupPenalty}, ::AbstractArray, ::SymmetricSparseIterate, ::Integer) =
  _array_group_operations()
function _kkt_violation(h::_ArrayPrecompose{<:AbstractGroupPenalty}, g::AbstractArray, x::AtomIterate, k::Integer)
  _checkparams(h, g)
  _checkparams(h, x)
  _checksamegroups(h.f, x)
  checkbounds(x.groups, k)
  c = abs(float(h.a[first(x.groups[k])]))
  c * kkt_violation(h.f, _PrecomposeArray(g, h.a, Val(true)), x, k)
end

@noinline _array_group_operations() = throw(ArgumentError("an array-scaled group penalty supports values, full-array proximal operators and four-argument kkt_violation on arrays; coordinate updates and restrict are not supported"))
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

and the value at the result comes from that of `f`. Array steps are per entry
for separable `f`, or per group for group penalties whose groups cover every
entry. Per-group steps do not support `AtomIterate` or `SymmetricSparseIterate`
outputs. Smooth when `f` is, with
gradient `∇f + ρ(x - c)` and Lipschitz constant `L_f + ρ`; separable when `f`
is, so coordinate descent applies. Hessian products and quadratic models are
available when `f` provides them, and coordinate curvatures increase by `ρ`.
On an [`AtomIterate`](@ref) a regularized
group norm updates a group. `Regularize(NormL1(λ), ρ)` is `ElasticNet(λ, ρ)`,
`Regularize(GroupNormL2(λ, groups), ρ)` the group elastic net,
`Regularize(NuclearNorm(λ), ρ)` the matrix elastic net, and with a center `c`
it is the strongly convex subproblem of proximal-point methods. The proximal
operator passes the shrunk point to `f` in a buffer that the function owns, so do
not share one object between threads; when `f` throws, the output is unchanged,
including sparse iterate outputs with their stored entries and order preserved.
`Symmetric` and `Hermitian` outputs are unsupported. A copy made for the device
by `Adapt.adapt` allocates a device buffer on each proximal call.
Values follow the overflow rule in [Overflow and underflow of values](@ref).

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
"""
struct Regularize{F<:AbstractFunction, T<:Real, C<:Union{Nothing, AbstractArray}, S<:Union{Nothing, Scratch}} <: AbstractFunction
  f::F
  ρ::T
  c::C
  scratch::S      # nothing on the device, where f must be a bits type

  function Regularize{F, T, C, S}(f::F, ρ::T, c::C, scratch::S) where {F<:AbstractFunction, T<:Real, C<:Union{Nothing, AbstractArray}, S<:Union{Nothing, Scratch}}
    c === nothing || eltype(c) <: Real || _real_error(:Regularize)
    _checknonneg("ρ", ρ)
    new{F, T, C, S}(f, ρ, c, scratch)
  end
end
Regularize{F, T, C}(f::F, ρ::T, c::C) where {F<:AbstractFunction, T<:Real, C<:Union{Nothing, AbstractArray}} =
  Regularize{F, T, C, Scratch}(f, ρ, c, Scratch())
Regularize(f::AbstractFunction, ρ::Real, c::Union{Nothing, AbstractArray} = nothing) =
  Regularize{typeof(f), typeof(ρ), typeof(c)}(f, ρ, c)

for trait in (:issmooth, :isproximable, :isseparable, :isconvex)
  @eval $trait(::Type{<:Regularize{F}}) where {F} = $trait(F)
end
iszeropreserving(r::Regularize) = r.c === nothing && isfinite(r.ρ) && iszeropreserving(r.f)

_ridge(r::Regularize, x::AbstractArray) = _square_value(r.ρ, x, r.c)
# a center is read at every entry of the matrix that the owner represents
_ridge(r::Regularize, x::SymmetricSparseIterate) = _square_value(r.ρ, r.c === nothing ? x : matrixview(x), r.c)

(r::Regularize)(x::_ArrayOrOwner) =
  (_check_value_arrays(r, x); _value_checked(r, x))
_value_unchecked(r::Regularize, x::_ArrayOrOwner) = (_checkparams(r, x); _value_at(_value_unchecked(r.f, x) + _ridge(r, x), x))
function _checkvalue(r::Regularize, x::_ArrayOrOwner)
  _checkreal(r, x)
  _checkparams(r, x)
  _checkvalue(r.f, x)
  r.c === nothing || _checksize(x, r.c)
  true
end

value_and_gradient!(g::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner) =
  (_check_oracle_arrays(value_and_gradient!, r, g, x); _value_and_gradient_unchecked!(g, r, x))
function _value_and_gradient_unchecked!(g::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner)
  _checkwrite(value_and_gradient!, r, g, x)
  v = _value_and_gradient_unchecked!(g, r.f, x)
  _add_ridge_gradient!(g, r, x)
  _value_at(v + _ridge(r, x), x)
end
gradient!(g::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner) =
  (_check_oracle_arrays(gradient!, r, g, x); _gradient_unchecked!(g, r, x))
_gradient_unchecked!(g::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner) =
  (_checkwrite(gradient!, r, g, x); _gradient_unchecked!(g, r.f, x); _add_ridge_gradient!(g, r, x))

function _checkwrite(op::O, r::Regularize, y::_ArrayOrOwner, x::_ArrayOrOwner) where {O<:Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}}
  _checkreal(r, y); _checkreal(r, x)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  (op === prox! || op === value_and_prox!) && _check_wrapper_output(y)
  _checksize(y, x)
  _checkparams(r, x)
  r.c === nothing || Base.Broadcast.check_broadcast_axes(_baxes(y), _vw(x), r.c)
  if op === value_and_prox! || op === value_and_gradient!
    r.c === nothing || size(x) == size(r.c) ||
      throw(DimensionMismatch("the arrays have sizes $(size(x)) and $(size(r.c))"))
  end
  _checkwrite(op, r.f, y, op === prox! || op === value_and_prox! ? y : x)
end

function _add_ridge_gradient!(g, r::Regularize, x)
  ρ = convert(eltype(g), r.ρ)
  r.c === nothing ? (g .+= ρ .* _vw(x)) : (g .+= ρ .* (_vw(x) .- _checked_parameter(g, r.c)))
end
# A gradient and a point of which one is an owner: ρ(x - c) is formed in the model precision, the widest of the
# output, the point and the center, where a level that the precision cannot hold is multiplied in a wider
# type and an entry that overflows in x - c is repaired, as for SqrNormL2; each entry is rounded to the output
# once, when it is added there. The model is a type parameter of the methods that form the entries, so that
# the entries are written by a loop that does not allocate.
_add_ridge_gradient!(g::AbstractArray, r::Regularize, x::SymmetricSparseIterate) =
  _add_ridge_gradient!(g, r.ρ, _ridge_model(eltype(g), eltype(x), r.c), r.c, matrixview(x))
# an owner gradient receives the symmetric part of ρ(x - c), from the lower entries when x is symmetric
_add_ridge_gradient!(g::SymmetricSparseIterate, r::Regularize, x) =
  _add_ridge_gradient!(g, r.ρ, _ridge_model(eltype(g), eltype(x), r.c), r.c, x)
_ridge_model(::Type{G}, ::Type{X}, ::Nothing) where {G, X} = _scaling_model(G, X, X)
_ridge_model(::Type{G}, ::Type{X}, c::AbstractArray) where {G, X} = _scaling_model(G, X, eltype(c))
function _add_ridge_gradient!(g, ρ, ::Type{W}, c, x) where {W}
  if _level_fits(W, ρ)
    _add_ridge_entries!(g, W, convert(W, ρ), c, x)
  else
    _add_ridge_entries!(g, W, convert(promote_type(W, float(typeof(ρ))), ρ), c, x)
  end
  g
end
# the entry ρ(x - c) of a level l, which is of type W unless W cannot hold the level
@inline _ridge_entry(::Type{W}, l, v, ::Nothing, i, j) where {W} = l * v
@inline _ridge_entry(::Type{W}, l, v, c, i, j) where {W} = _ridge_entry(W, l, v, @inbounds c[i, j])
@inline _ridge_entry(::Type{W}, l::W, v, cij) where {W} = _ridge_gradient_entry(l, v, cij)
@inline _ridge_entry(::Type{W}, l, v, cij) where {W} = _ridge_gradient_wide(W, l, v, cij)
function _add_ridge_entries!(g::SymmetricSparseIterate, ::Type{W}, l, c, x) where {W}
  p = g.p
  @inbounds for j in 1:p, i in j:p
    u = _ridge_entry(W, l, x[i, j], c, i, j)
    if i != j && !(x isa SymmetricSparseIterate)
      u = _symmetric_pair(u, _ridge_entry(W, l, x[j, i], c, j, i), i, j)
    end
    k = tril_index(p, i, j)
    _setstored!(g, _getstored(g, k) + u, k)
  end
  g
end
# a dense output, whose center may broadcast
_add_ridge_entries!(g::AbstractArray, ::Type{W}, l, ::Nothing, x) where {W} = (g .+= l .* x)
_add_ridge_entries!(g::AbstractArray, ::Type{W}, l, c, x) where {W} = (g .+= _ridge_entry.(W, l, x, c))

# the point (v + γρc)/(1 + γρ) in y, and the step γ/(1 + γρ) of f
@inline function _regularize_step(γ::T, ρ::T) where {T}
  product = γ * ρ
  isfinite(product) ? γ / (one(T) + product) : inv(ρ) / (one(T) + inv(γ) / ρ)
end
# The same at the endpoint: a level that is zero as given (from r.ρ, not from its conversion to the
# working type, which can round a tiny level to zero) and the step Inf as given (not as converted, which
# turns a finite step beyond the range of the working type into Inf), where the product γρ is Inf⋅0.
# Every caller decides `endpoint` from the raw level and the raw step, and passes the converted step.
# At the endpoint the weights are those of the step 1, which are also those of every positive finite
# step: (1, 0) with the sign of the level. The step of f is γ, as it is for every finite step
# (γ/(1 ± 0)). Any other step, NaN, negative and overflowed ones included, keeps the arithmetic of the
# formulas. The first form is a branch, for a step taken once per call; the second two are selects,
# for the steps of a loop over entries, which they leave free of a branch.
@inline _regularize_endpoint(r::Regularize, γ) = iszero(r.ρ) & (γ == Inf)
@inline function _regularize_parameters(γ::T, ρ::T, endpoint::Bool) where {T}
  if endpoint
    a, b = _ridge_weights(one(T), ρ)
    return a, b, γ
  end
  a, b = _ridge_weights(γ, ρ)
  a, b, _regularize_step(γ, ρ)
end
@inline _regularize_weights(γ::T, ρ::T, endpoint::Bool) where {T} =
  _ridge_weights(ifelse(endpoint, one(T), γ), ρ)
@inline _regularize_step(γ::T, ρ::T, endpoint::Bool) where {T} =
  ifelse(endpoint, γ, _regularize_step(γ, ρ))
function _shrunk!(y::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner, γ::Real)
  T = float(eltype(y))
  a, b, step = _regularize_parameters(convert(T, γ), convert(T, r.ρ), _regularize_endpoint(r, γ))
  yy = _ow(y)
  r.c === nothing ? (yy .= a .* _pt(yy, x)) : (yy .= _ridge_blend.(_pt(yy, x), _checked_parameter(y, r.c), a, b))
  step
end

# the shrunk point in a buffer owned by r, so that y is unchanged when f throws
prox!(y::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, r, y, x, γ); _prox_unchecked!(y, r, x, γ))
function _prox_unchecked!(y::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner, γ::Real)
  _checkwrite(prox!, r, y, x)
  z = _output_buffer!(r.scratch, y)
  w = r.scratch === nothing ? z : _wrapper_output(y, z)
  _prox_unchecked!(w, r.f, z, _shrunk!(z, r, x, γ))
  _wrapper_commit!(y, w)
end
value_and_prox!(y::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, r, y, x, γ); _value_and_prox_unchecked!(y, r, x, γ))
function _value_and_prox_unchecked!(y::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner, γ::Real)
  _checkwrite(value_and_prox!, r, y, x)
  z = _output_buffer!(r.scratch, y)
  w = r.scratch === nothing ? z : _wrapper_output(y, z)
  v = _value_and_prox_unchecked!(w, r.f, z, _shrunk!(z, r, x, γ))
  v = _value_at(v + _ridge(r, w), w)
  _wrapper_commit!(y, w)
  v
end

Base.@propagate_inbounds function prox_block(r::Regularize, v::Real, i::Integer, γ::Real)
  T = float(typeof(v))
  step, ρ, endpoint = convert(T, γ), convert(T, r.ρ), _regularize_endpoint(r, γ)
  a, b = _regularize_weights(step, ρ, endpoint)
  w = r.c === nothing ? a * v : _ridge_blend(v, r.c[i], a, b)
  prox_block(r.f, w, i, _regularize_step(step, ρ, endpoint))
end
_checkparams(r::Regularize, x) = (_checkparam(r.c, x); _checkparams(r.f, x))

# the ridge term adds ρ(xᵢ - cᵢ) to the gradient, -ρcᵢ at a coordinate at zero
_ridge_shifted(g::AbstractArray, r::Regularize) =
  r.c === nothing ? g : _ShiftedArray(g, _ScaledArray(r.c, -r.ρ))
kkt_violation(r::Regularize, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); kkt_violation(r.f, _ridge_shifted(g, r), i))
for X in (:AbstractArray, :AtomIterate)
  @eval _kkt_violation(r::Regularize, g::AbstractArray, x::$X, k::Integer) = _kkt_violation(r.f, _ridge_shifted(g, r), x, k)
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

hessian_vec!(Hv::_ArrayOrOwner, h::Precompose, x::_ArrayOrOwner, v::_ArrayOrOwner) =
  (_check_second_order_arrays(h, x, Hv, v); _hessian_vec_unchecked!(Hv, h, x, v))
function _hessian_vec_unchecked!(Hv::_ArrayOrOwner, h::Precompose, x::_ArrayOrOwner, v::_ArrayOrOwner)
  _checksize(Hv, x)
  _checkdirection(v, x)
  z = _image!(h, x)
  _checkparams(h.f, z)
  d = _linear_image!(h.direction, h.A, v)
  b = _inner_buffer(h, z)
  _hessian_vec_unchecked!(b, h.f, z, d)
  _adjoint_apply!(Hv, h, b)
end

hessian_vec!(Hv::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner, v::_ArrayOrOwner) =
  (_check_second_order_arrays(r, x, Hv, v); _hessian_vec_unchecked!(Hv, r, x, v))
function _hessian_vec_unchecked!(Hv::_ArrayOrOwner, r::Regularize, x::_ArrayOrOwner, v::_ArrayOrOwner)
  _checksize(Hv, x)
  _checkdirection(v, x)
  _checkparams(r, x)
  _hessian_vec_unchecked!(Hv, r.f, x, v)
  _add_ridge_direction!(Hv, r.ρ, x, v)
end
_add_ridge_direction!(Hv::AbstractArray, ρ, x, v) = (Hv .+= convert(eltype(Hv), ρ) .* _vw(v))
# a dense output with an owner as the point or the direction: ρ v in the model precision, as for an owner output
for (X, V) in ((:SymmetricSparseIterate, :Any), (:Any, :SymmetricSparseIterate),
               (:SymmetricSparseIterate, :SymmetricSparseIterate))
  @eval _add_ridge_direction!(Hv::AbstractArray, ρ, x::$X, v::$V) =
    _add_ridge_dense_direction!(Hv, ρ, _scaling_model(eltype(Hv), eltype(x), eltype(v)), v)
end
function _add_ridge_dense_direction!(Hv::AbstractArray, ρ, ::Type{W}, v) where {W}
  if _level_fits(W, ρ)
    _add_dense_entries!(Hv, convert(W, ρ), W, v)
  else
    R = promote_type(W, float(typeof(ρ)))
    _add_dense_entries!(Hv, convert(R, ρ), R, v)
  end
end
function _add_dense_entries!(Hv::AbstractArray, l, ::Type{R}, v) where {R}
  w = _vw(v)
  @inbounds for I in eachindex(Hv, w)
    Hv[I] += l * convert(R, w[I])
  end
  Hv
end
# ρ v added to an owner: the level and the entries in the model precision, the wider of the output, the point and
# the direction, and each entry rounded to the output once, when it is stored; a level that this precision
# cannot hold is multiplied in a wider type
_add_ridge_direction!(Hv::SymmetricSparseIterate, ρ, x, v) =
  _add_ridge_direction!(Hv, ρ, _scaling_model(eltype(Hv), eltype(x), eltype(v)), v)
function _add_ridge_direction!(Hv::SymmetricSparseIterate, ρ, ::Type{W}, v) where {W}
  if _level_fits(W, ρ)
    _add_entries!(Hv, convert(W, ρ), W, v)
  else
    R = promote_type(W, float(typeof(ρ)))
    _add_entries!(Hv, convert(R, ρ), R, v)
  end
end
function _add_entries!(Hv::SymmetricSparseIterate, l, ::Type{R}, v) where {R}
  p = Hv.p
  @inbounds for c in 1:p, r in c:p
    u = l * convert(R, v[r, c])
    if r != c && !(v isa SymmetricSparseIterate)
      u = _symmetric_pair(u, l * convert(R, v[c, r]), r, c)
    end
    k = tril_index(p, r, c)
    _setstored!(Hv, _getstored(Hv, k) + u, k)
  end
  Hv
end

quadratic_model(h::Precompose, x::_ArrayOrOwner) =
  (_check_second_order_arrays(h, x); _quadratic_model_unchecked(h, x))
function _quadratic_model_unchecked(h::Precompose, x::_ArrayOrOwner)
  f = _quadratic_model_unchecked(h.f, _image!(h, x))
  Precompose{typeof(f), typeof(h.A), typeof(h.μ)}(f, h.A, h.μ)
end
quadratic_model(r::Regularize, x) =
  (_check_second_order_arrays(r, x); _quadratic_model_unchecked(r, x))
_quadratic_model_unchecked(r::Regularize, x) =
  (_checkparam(r.c, x); Regularize(_quadratic_model_unchecked(r.f, x), r.ρ, r.c))

quadratic_model!(q::Precompose, h::Precompose, x::_ArrayOrOwner) =
  (_check_second_order_arrays(h, x); _quadratic_model_unchecked!(q, h, x))
function _quadratic_model_unchecked!(q::Precompose, h::Precompose, x::_ArrayOrOwner)
  q.A === h.A && q.μ == h.μ || throw(ArgumentError("the model must come from the same Precompose"))
  _quadratic_model_unchecked!(q.f, h.f, _image!(h, x))
  q
end
quadratic_model!(q::Regularize, r::Regularize, x) =
  (_check_second_order_arrays(r, x); _quadratic_model_unchecked!(q, r, x))
function _quadratic_model_unchecked!(q::Regularize, r::Regularize, x)
  q.c === r.c && q.ρ == r.ρ || throw(ArgumentError("the model must come from the same Regularize"))
  _checkparam(r.c, x)
  _quadratic_model_unchecked!(q.f, r.f, x)
  q
end

# These wrappers, defined after calculus.jl, also evaluate the inner value.
_has_value(f::Union{Precompose, _ArrayPrecompose, Regularize, Spectral, BoxConstrained, LinearConstrained,
                   BallConstrained, SphereConstrained, AffineConstrained}) = _has_value(f.f)

function _checkstep(op::O, r::Regularize, y::AbstractArray, x::AbstractArray, γ::Real) where {O}
  T = float(eltype(y))
  _checkstep(op, r.f, y, _StepInput(y), last(_regularize_parameters(convert(T, γ), convert(T, r.ρ), _regularize_endpoint(r, γ))))
end
function _checkstep(op::O, h::Precompose, y::AbstractArray, x::AbstractArray, γ::Real) where {O}
  z = _checkimage(h, x)
  b = h isa _ScalarPrecompose ? z : _checkbuffer(h.inner, eltype(z), size(z))
  T = eltype(z)
  _checkstep(op, h.f, b, _StepInput(z), convert(T, h.μ) * convert(T, γ))
end
