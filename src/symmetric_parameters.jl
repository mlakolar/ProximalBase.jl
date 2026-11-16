@inline _isbuiltin(f) = parentmodule(typeof(f)) === (@__MODULE__)

# Caller-supplied arrays in the iterate's entry domain. Design data and arrays
# indexed by groups, ranks or image coordinates are deliberately absent.
entry_parameters(::AbstractFunction) = ()
entry_parameters(::SeparableFunction) = nothing
_inner_functions(::AbstractFunction) = ()
entry_parameters(f::NormL1) = (:w => f.w,)
entry_parameters(f::ElasticNet) = (:w => f.w,)
entry_parameters(f::LinearFunction) = (:c => f.c,)
entry_parameters(f::SqrNormL2) = (:y => f.y,)
_inner_functions(f::Scaled) = (f.f,)

check_symmetric_parameters(f::AbstractFunction, x) = (_check_entry_arrays(f, x); true)

# Shape checks remain separate: a parameter vector can have matrix length,
# while a write of that parameter can still require the destination's axes.
_check_parameter_axes(y, a) = true
_check_parameter_axes(y::SymmetricSparseIterate, a::AbstractArray) = _check_axes(y, a)

# The explicit validation of a model: each declared array that is not symmetric by type is checked for extent and
# compared at every mirrored pair, including pairs that the iterate does not store; an array symmetric by type
# (such as Symmetric(W, :L)) is accepted without either. Only check_symmetric_parameters calls it; the
# numerical oracles take the symmetry of these arrays as a precondition and never compare their entries.
_check_entry_arrays(f, x) = true
function _check_entry_arrays(f::AbstractFunction, x::SymmetricSparseIterate)
  _check_own_arrays(f, entry_parameters(f), x)
  _check_inner_arrays(_inner_functions(f), x)
end
_check_own_arrays(f, ::Tuple{}, x) = true
_check_own_arrays(f, ::Nothing, x) = _throw_undeclared_parameters(f)
_check_own_arrays(f, arrays, x) = _throw_parameter_declaration(f, :entry_parameters)
@noinline _throw_undeclared_parameters(f) = throw(ArgumentError(
  "$(nameof(typeof(f))) does not declare its per-entry parameters; define ProximalBase.entry_parameters(f::$(nameof(typeof(f)))) to return them, or () when it has none, to use it on a SymmetricSparseIterate"))
@noinline _throw_parameter_declaration(f, name) = throw(ArgumentError(
  "$(nameof(typeof(f))) parameter $name must be nothing, a scalar or an array in a tuple of name => value pairs"))
function _check_own_arrays(f, arrays::Tuple, x)
  pair = first(arrays)
  pair isa Pair{Symbol} || _throw_parameter_declaration(f, :entry_parameters)
  name, a = pair
  _check_entry_array(f, name, a, x)
  _check_own_arrays(f, Base.tail(arrays), x)
end
_check_inner_arrays(::Tuple{}, x) = true
function _check_inner_arrays(fs::Tuple, x)
  _check_entry_arrays(first(fs), x)
  _check_inner_arrays(Base.tail(fs), x)
end
function _check_entry_arrays(f::AbstractFunction, x::Tuple)
  _check_inner_arrays(_inner_functions(f), x)
end
_check_entry_array(f, name, a, x) = _throw_parameter_declaration(f, name)
_check_entry_array(f, name, a::Union{Nothing, Number}, x) = true
function _check_entry_array(f, name, a::AbstractArray, x)
  _symmetric_by_type(a) && return true
  _checkparam(a, x)
  _check_pairs(f, name, a, x.p)
end
function _check_pairs(f, name, a, p)
  @inbounds for cb in 1:32:p, rb in cb:32:p
    for c in cb:min(cb + 31, p), r in max(c + 1, rb):min(rb + 31, p)
      u, v = a[(c - 1) * p + r], a[(r - 1) * p + c]
      _symequal(u, v) || _throw_asymmetric_parameter(f, name, a, p, 0, 0)
    end
  end
  true
end

# The value oracle's structural errors (a parameter of the wrong extent, an unsupported element type), raised
# before any entry is read.
_check_value_shape(f, x) = _checkvalue(f, x)
_check_value_shape(f::Scaled, x) = _check_value_shape(f.f, x)
# The extent of a function's parameters against an iterate: O(1) for the built-in functions, and
# check_parameters for a custom one.
_check_parameter_extent(f, x) = _checkparams(f, x)
_check_parameter_extent(f::Scaled, x) = _check_parameter_extent(f.f, x)
_check_value_arrays(f, x) = _checkreal(f, x)
function _check_value_arrays(f, x::SymmetricSparseIterate)
  _checkreal(f, x)
  _check_value_shape(f, x)
end

# A SymmetricSparseIterate output takes a scalar step, the Frobenius proximal operator of the
# symmetric matrix. An array or per-group step is refused before anything is written: the
# active coordinates, the values and the metadata of the output are untouched. Scaled has array-step
# methods of its own that do not constrain the output, so it needs methods here to stay unambiguous.
@noinline _throw_symmetric_iterate_step(γ) = throw(ArgumentError(
  "a step γ of type $(typeof(γ)) is not supported with a SymmetricSparseIterate as the output; " *
  "use a scalar step, or Matrix(x) or matrixview(x) with a dense output"))
prox!(y::SymmetricSparseIterate, f::AbstractFunction, x, γ::AbstractArray) = _throw_symmetric_iterate_step(γ)
prox!(y::SymmetricSparseIterate, f::Scaled, x, γ::AbstractArray) = _throw_symmetric_iterate_step(γ)
value_and_prox!(y::SymmetricSparseIterate, f::AbstractFunction, x, γ::AbstractArray) = _throw_symmetric_iterate_step(γ)
value_and_prox!(y::SymmetricSparseIterate, f::Scaled, x, γ::AbstractArray) = _throw_symmetric_iterate_step(γ)
prox(f::AbstractFunction, x::SymmetricSparseIterate, γ::AbstractArray) = _throw_symmetric_iterate_step(γ)
# The allocating oracles at a SymmetricSparseIterate, for the families that accept it. The
# proximal operator is an independent SymmetricSparseIterate, and the gradient a dense exactly symmetric matrix.
const _SymmetricProxFamily = Union{SeparableFunction, LinearFunction, SqrNormL2}
const _SymmetricGradientFamily = Union{LinearFunction, SqrNormL2}
prox(f::_SymmetricProxFamily, x::SymmetricSparseIterate, γ::Real = 1) = prox!(similar(x, float(eltype(x))), f, x, γ)
gradient(f::_SymmetricGradientFamily, x::SymmetricSparseIterate) = gradient!(_gradient_buffer(x), f, x)
function value_and_gradient(f::_SymmetricGradientFamily, x::SymmetricSparseIterate)
  g = _gradient_buffer(x)
  v = value_and_gradient!(g, f, x)
  v, g
end

# a symmetric iterate output that shares memory with a parameter array
_mightalias_parameter(out::SymmetricSparseIterate, p::AbstractArray) = _aliases(out, p)

_has_symmetric(x) = false
_has_symmetric(::SymmetricSparseIterate) = true
@generated _has_symmetric(x::Tuple) = _blockany(k -> :(_has_symmetric(x[$k])), fieldcount(x))

# Only public boundaries run these preflights; inner kernels keep the shape checks of their own. The
# arrays of a symmetric model are not compared here (see _check_entry_arrays), and the checks that remain
# are structural: element types, and for a symmetric output the sizes, aliases and supported operation of
# _checkwrite. A symmetric point with another output reaches the same _checkwrite in the kernel body.
function _check_oracle_arrays(op::O, f, y, x, γ = nothing) where {O}
  _checkreal(f, y)
  _checkreal(f, x)
  γ === nothing || _checkreal(f, γ)
  if _has_symmetric(y)
    γ === nothing ? _checkwrite(op, f, y, x) : _checkwrite(op, f, y, x, γ)
  end
  true
end
_value_unchecked(f, x) = f(x)

# Symmetrize computed gradients before any wrapper scales or shifts them.
struct _SymmetricGradient{T, N, G<:AbstractArray{T, N}} <: AbstractArray{T, N}
  g::G
  p::Int
end
Base.size(g::_SymmetricGradient) = size(g.g)
Base.IndexStyle(::Type{<:_SymmetricGradient}) = IndexLinear()
function Base.getindex(g::_SymmetricGradient, i::Int)
  r, c = mod1(i, g.p), (i - 1) ÷ g.p + 1
  _symmetric_pair(g.g[i], g.g[(r - 1) * g.p + c], r, c)
end
# the symmetric part of the pair of entries (r, c) and (c, r), which is the entry itself when they
# are equal, and the mean otherwise; opposite infinities have none
@inline function _symmetric_pair(a, b, r, c)
  _symequal(a, b) && return a
  isinf(a) && isinf(b) && a == -b && _throw_gradient_pair(r, c, a, b)
  _symmetric_mean(promote(a, b)...)
end
@noinline _throw_gradient_pair(r, c, a, b) = throw(ArgumentError(
  "the gradient g has no symmetric part: entry ($r, $c) is $a and entry ($c, $r) is $b"))

# The symmetric iterate g written one coordinate at a time. Coordinate (r, c) takes entry(v, j) at the lower
# position of a point x whose matrix is symmetric (a symmetric iterate, or an argument that has been symmetrized),
# j being the linear index (c - 1)p + r of the matrix where the parameters are read.
@inline function _lower_coordinates!(g::SymmetricSparseIterate, entry::F, x) where {F}
  p = g.p
  @inbounds for c in 1:p, r in c:p
    _setstored!(g, entry(x[r, c], (c - 1) * p + r), tril_index(p, r, c))
  end
  g
end
# The same for an entrywise gradient at any point: the symmetric part of the computed gradient, from
# the two positions of a pair when x is not a symmetric iterate, whose matrix is symmetric.
@inline function _symmetric_coordinates!(g::SymmetricSparseIterate, entry::F, x) where {F}
  p = g.p
  @inbounds for c in 1:p, r in c:p
    v = entry(x[r, c], (c - 1) * p + r)
    if r != c && !(x isa SymmetricSparseIterate)
      v = _symmetric_pair(v, entry(x[c, r], (r - 1) * p + c), r, c)
    end
    _setstored!(g, v, tril_index(p, r, c))
  end
  g
end

# ⟨a, x⟩ over the matrix that the symmetric iterate x represents, for a read at the linear indices of a p × p matrix:
# an off-diagonal coordinate counts at both of its positions, each with its own entry of a
function _symmetric_matrix_dot(a, x::SymmetricSparseIterate)
  s = zero(float(promote_type(eltype(a), eltype(x))))
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    v = a[(c - 1) * x.p + r]
    r == c || (v += a[(r - 1) * x.p + c])
    s += v * x.nzval[k]
  end
  s
end

_kkt_violation(s::Scaled, g::AbstractArray, x::SymmetricSparseIterate, k::Integer) =
  _chained(s) ? _nested_kkt((f, h) -> _kkt_violation(f, h, x, k), s, g) :
  iszero(s.α) ? _kkt_violation(ZeroFunction(), g, x, k) :
    s.α * _kkt_violation(s.f, _ScaledArray(g, inv(s.α)), x, k)

# Fallbacks for functions outside the entry-array kernels.
_prox_unchecked!(y, f::AbstractFunction, x, γ) = prox!(y, f, x, γ)
_value_and_prox_unchecked!(y, f::AbstractFunction, x, γ) = value_and_prox!(y, f, x, γ)
_value_and_gradient_unchecked!(y, f::AbstractFunction, x) = value_and_gradient!(y, f, x)

# A parameter matrix of a function that writes into a symmetric iterate. Its entries are read at their lower
# positions on the assumption that the caller's model is symmetric: nothing here compares them, and
# check_symmetric_parameters validates the model on request. The wrapper forwards the size, axes, aliasing ids,
# index style and entries of the parameter unchanged, and is symmetric by type for _symmetric_by_type.
struct _AssumedSymmetric{T, A<:AbstractMatrix{T}} <: AbstractMatrix{T}
  array::A
end
Base.size(a::_AssumedSymmetric) = size(a.array)
Base.axes(a::_AssumedSymmetric) = axes(a.array)
Base.dataids(a::_AssumedSymmetric) = Base.dataids(a.array)
Base.IndexStyle(::Type{<:_AssumedSymmetric{T, A}}) where {T, A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(a::_AssumedSymmetric, I::Vararg{Int, N}) where {N} = a.array[I...]
_symmetric_by_type(::_AssumedSymmetric) = true
@inline _assumed_symmetric(y, a) = a
@inline _assumed_symmetric(y::SymmetricSparseIterate, a::AbstractMatrix) =
  _symmetric_by_type(a) ? a : _AssumedSymmetric(a)
_copy_parameter!(y, a) = copyto!(y, a)
# a symmetric iterate output takes the values of a parameter matrix that is symmetric by the model's precondition, one
# coordinate at a time from its lower entry: a coordinate that is active stays active
_copy_parameter!(y::SymmetricSparseIterate, a::AbstractMatrix) = _store_lower!(y, a)
function _store_lower!(y::SymmetricSparseIterate, a::AbstractMatrix)
  p = y.p
  @inbounds for c in 1:p, r in c:p
    _setstored!(y, a[r, c], tril_index(p, r, c))
  end
  y
end
_store_lower!(y::AbstractArray, a::AbstractMatrix) = copyto!(y, a)
_value_and_prox_unchecked!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::Real) = value_and_prox!(y, f, x, γ)
_value_unchecked(f::NormL1{<:Real, Nothing}, x::DenseArray{<:AbstractFloat}) = f(x)

_gradient_unchecked!(g::AbstractVector, f::QuadraticFunction, x::AbstractVector) = gradient!(g, f, x)

# Under the precondition that mirrored entries of a parameter are equal, the dot of a parameter with a symmetric
# iterate reads the lower entry of each stored coordinate and counts it at both positions of an off-diagonal one.
_value_checked(f, x) = _value_unchecked(f, x)
function _checked_value_dot(a, x::SymmetricSparseIterate)
  s = zero(promote_type(eltype(a), eltype(x)))
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    v = a[(c - 1) * x.p + r]
    s += (r == c ? v : v + v) * x.nzval[k]
  end
  s
end
_value_checked(f::LinearFunction, x::SymmetricSparseIterate) =
  (_checkvalue(f, x); _value_at(_linear_value(f, x, _checked_value_dot(f.c, x) + f.d), x))
function _value_checked(s::Scaled, x::Union{SymmetricSparseIterate, Tuple})
  if _chained(s)
    f, a = _scaled_chain(s)
    return _value_at(_scaled_value_by(f, a, x, _value_checked(f, x)), x)
  end
  _value_at(_scaled_value(s, x, _value_checked(s.f, x)), x)
end

@inline _vw(x::SymmetricSparseIterate) = matrixview(x)
@inline _vw(x) = x

# y = x for a proximal step that is the identity: an array takes the copy, a symmetric iterate the lower entries of x,
# one coordinate at a time (a coordinate that is active stays active), and nothing when x is y
_assign!(y::AbstractArray, x) = copyto!(y, _vw(x))
_assign!(y::SymmetricSparseIterate, x) = (x === y ? y : _store_lower!(y, _vw(x)))
const _SymmetricProxWrapper = Scaled
prox(f::_SymmetricProxWrapper, x::SymmetricSparseIterate, γ::Real = 1) = prox!(similar(x, float(eltype(x))), f, x, γ)
const _SymmetricGradientWrapper = Scaled
gradient(f::_SymmetricGradientWrapper, x::SymmetricSparseIterate) = gradient!(_gradient_buffer(x), f, x)
function value_and_gradient(f::_SymmetricGradientWrapper, x::SymmetricSparseIterate)
  g = _gradient_buffer(x)
  v = value_and_gradient!(g, f, x)
  v, g
end
prox(f::ZeroFunction, x::SymmetricSparseIterate, γ::Real = 1) = prox!(similar(x, float(eltype(x))), f, x, γ)
