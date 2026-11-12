@inline _isbuiltin(f) = parentmodule(typeof(f)) === (@__MODULE__)

# Caller-supplied arrays in the iterate's entry domain. Design data and arrays
# indexed by groups, ranks or image coordinates are deliberately absent.
entry_parameters(::AbstractFunction) = ()
entry_parameters(::SeparableFunction) = nothing
_inner_functions(::AbstractFunction) = ()
entry_parameters(f::NormL1) = (:w => f.w,)
entry_parameters(f::LinearFunction) = (:c => f.c,)
entry_parameters(f::SqrNormL2) = (:y => f.y,)
_inner_functions(f::Scaled) = (f.f,)

check_symmetric_parameters(f::AbstractFunction, x) = (_check_entry_arrays(f, x); true)

# Shape checks remain separate: a parameter vector can have matrix length,
# while a write of that parameter can still require the destination's axes.
_check_parameter_axes(y, a) = true
_check_parameter_axes(y::SymmetricSparseIterate, a::AbstractArray) = _check_axes(y, a)

struct _AllPairs end
struct _StoredPairs{X, Y}
  x::X
  y::Y
end
struct _CoordinatePair
  k::Int
end

_check_entry_arrays(f, x, domain = _AllPairs()) = true
function _check_entry_arrays(f::AbstractFunction, x::SymmetricSparseIterate, domain = _AllPairs())
  _check_own_arrays(f, entry_parameters(f), x, domain)
  _check_inner_arrays(_inner_functions(f), x, domain)
end
_check_own_arrays(f, ::Tuple{}, x, domain) = true
_check_own_arrays(f, ::Nothing, x, domain) = _throw_undeclared_parameters(f)
_check_own_arrays(f, arrays, x, domain) = _throw_parameter_declaration(f, :entry_parameters)
@noinline _throw_undeclared_parameters(f) = throw(ArgumentError(
  "$(nameof(typeof(f))) does not declare its per-entry parameters; define ProximalBase.entry_parameters(f::$(nameof(typeof(f)))) to return them, or () when it has none, to use it on a SymmetricSparseIterate"))
@noinline _throw_parameter_declaration(f, name) = throw(ArgumentError(
  "$(nameof(typeof(f))) parameter $name must be nothing, a scalar or an array in a tuple of name => value pairs"))
function _check_own_arrays(f, arrays::Tuple, x, domain)
  pair = first(arrays)
  pair isa Pair{Symbol} || _throw_parameter_declaration(f, :entry_parameters)
  name, a = pair
  _check_entry_array(f, name, a, x, domain)
  _check_own_arrays(f, Base.tail(arrays), x, domain)
end
_check_inner_arrays(::Tuple{}, x, domain) = true
function _check_inner_arrays(fs::Tuple, x, domain)
  _check_entry_arrays(first(fs), x, domain)
  _check_inner_arrays(Base.tail(fs), x, domain)
end
function _check_entry_arrays(f::AbstractFunction, x::Tuple, domain = _AllPairs())
  _check_inner_arrays(_inner_functions(f), x, domain)
end
_check_entry_array(f, name, a, x, domain) = _throw_parameter_declaration(f, name)
_check_entry_array(f, name, a::Union{Nothing, Number}, x, domain) = true
function _check_entry_array(f, name, a::AbstractArray, x, domain)
  _symmetric_by_type(a) && return true
  _checkparam(a, x)
  _check_pairs(f, name, a, x.p, domain)
end
function _check_pair(f, name, a, p, k)
  r, c = tril_cartesian(p, k)
  _check_pair(f, name, a, p, r, c)
end
function _check_pair(f, name, a, p, r, c)
  r == c && return true
  u, v = a[(c - 1) * p + r], a[(r - 1) * p + c]
  _symequal(u, v) || _throw_asymmetric_parameter(f, name, a, p, r, c)
  true
end
function _check_pairs(f, name, a, p, ::_AllPairs)
  @inbounds for cb in 1:32:p, rb in cb:32:p
    for c in cb:min(cb + 31, p), r in max(c + 1, rb):min(rb + 31, p)
      u, v = a[(c - 1) * p + r], a[(r - 1) * p + c]
      _symequal(u, v) || _throw_asymmetric_parameter(f, name, a, p, 0, 0)
    end
  end
  true
end
_check_pairs(f, name, a, p, d::_CoordinatePair) = _check_pair(f, name, a, p, d.k)
function _check_stored_pairs(f, name, a, p, x)
  c, lo, hi = 0, 1, 0
  @inbounds for i in 1:x.nnz
    k = x.nzval2ind[i]
    if lo <= k <= hi
      r = c + k - lo
    else
      r, c = tril_cartesian(p, k)
      lo = k - r + c
      hi = lo + p - c
    end
    _check_pair(f, name, a, p, r, c)
  end
  true
end
function _check_pairs(f, name, a, p, d::_StoredPairs)
  d.x.nnz == ncoordinates(d.x) && return _check_pairs(f, name, a, p, _AllPairs())
  _check_stored_pairs(f, name, a, p, d.x)
  if d.y !== d.x
    @inbounds for i in 1:d.y.nnz
      k = d.y.nzval2ind[i]
      iszero(d.x.ind2nzval[k]) && _check_pair(f, name, a, p, k)
    end
  end
  true
end

# Only kernels that actually stay on stored coordinates use a sparse domain.
_stored_value(f::AbstractFunction) = isempty(entry_parameters(f)) && isempty(_inner_functions(f))
_stored_value(f::SeparableFunction) = _isbuiltin(f) && iszeropreserving(f)
_stored_value(::LinearFunction) = true
_stored_value(f::SqrNormL2) = f.y === nothing
_stored_value(f::Scaled) = _stored_value(f.f)
_stored_write(op, ::AbstractFunction) = false
_stored_write(op, f::SeparableFunction) = _isbuiltin(f)
_stored_write(op::O, f::Scaled) where {O} = _stored_write(op, f.f)
_stored_domain(op::O, f) where {O} =
  (op === prox! || op === value_and_prox!) && isseparable(f) && iszeropreserving(f) && _stored_write(op, f)
# Keep the value oracle's shape errors ahead of the pair scan.
_check_value_shape(f, x) = _checkvalue(f, x)
_check_value_shape(f::Scaled, x) = _check_value_shape(f.f, x)
_check_value_arrays(f, x) = (_checkreal(f, x); _check_entry_arrays(f, x))
function _check_value_arrays(f, x::SymmetricSparseIterate)
  _checkreal(f, x)
  _check_value_shape(f, x)
  if _stored_value(f)
    _check_entry_arrays(f, x, _StoredPairs(x, x))
  else
    _check_entry_arrays(f, x)
  end
end
_check_write_arrays(op::O, f, y, x) where {O} = _check_entry_arrays(f, y)
function _check_write_arrays(op::O, f, y::SymmetricSparseIterate, x::SymmetricSparseIterate) where {O}
  if _stored_domain(op, f)
    _check_entry_arrays(f, y, _StoredPairs(x, y))
  else
    _check_entry_arrays(f, y)
  end
end

# A SymmetricSparseIterate output takes a scalar step, the Frobenius proximal operator of the
# symmetric matrix. An array or per-group step is refused before anything is written: the
# active coordinates, the values and the metadata of the output are untouched. Scaled has array-step
# methods of its own that do not constrain the output, so it needs methods here to stay unambiguous.
@noinline _throw_owner_step(γ) = throw(ArgumentError(
  "a step γ of type $(typeof(γ)) is not supported with a SymmetricSparseIterate as the output; " *
  "use a scalar step, or Matrix(x) or matrixview(x) with a dense output"))
prox!(y::SymmetricSparseIterate, f::AbstractFunction, x, γ::AbstractArray) = _throw_owner_step(γ)
prox!(y::SymmetricSparseIterate, f::Scaled, x, γ::AbstractArray) = _throw_owner_step(γ)
value_and_prox!(y::SymmetricSparseIterate, f::AbstractFunction, x, γ::AbstractArray) = _throw_owner_step(γ)
value_and_prox!(y::SymmetricSparseIterate, f::Scaled, x, γ::AbstractArray) = _throw_owner_step(γ)
prox(f::AbstractFunction, x::SymmetricSparseIterate, γ::AbstractArray) = _throw_owner_step(γ)
# The allocating oracles at the owner of a symmetric matrix, for the families that accept it. The
# proximal operator is an independent owner, and the gradient a dense exactly symmetric matrix.
const _OwnerSeparable = Union{SeparableFunction, LinearFunction, SqrNormL2}
const _OwnerSmooth = Union{LinearFunction, SqrNormL2}
prox(f::_OwnerSeparable, x::SymmetricSparseIterate, γ::Real = 1) = prox!(similar(x, float(eltype(x))), f, x, γ)
gradient(f::_OwnerSmooth, x::SymmetricSparseIterate) = gradient!(_gradient_buffer(x), f, x)
function value_and_gradient(f::_OwnerSmooth, x::SymmetricSparseIterate)
  g = _gradient_buffer(x)
  v = value_and_gradient!(g, f, x)
  v, g
end

# an owner output that shares memory with a parameter array
_mightalias_parameter(out::SymmetricSparseIterate, p::AbstractArray) = _aliases(out, p)

_has_symmetric(x) = false
_has_symmetric(::SymmetricSparseIterate) = true
@generated _has_symmetric(x::Tuple) = _blockany(k -> :(_has_symmetric(x[$k])), fieldcount(x))

# Only public boundaries check arrays. Inner kernels retain the existing shape
# preflights, but never scan a caller's arrays a second time.
function _check_oracle_arrays(op::O, f, y, x, γ = nothing) where {O}
  _checkreal(f, y)
  _checkreal(f, x)
  γ === nothing || _checkreal(f, γ)
  if _has_symmetric(y)
    γ === nothing ? _checkwrite(op, f, y, x) : _checkwrite(op, f, y, x, γ)
    _check_write_arrays(op, f, y, x)
  elseif x isa SymmetricSparseIterate
    # an owner point with another output: the parameters are read at every entry of the matrix
    _check_entry_arrays(f, x)
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

# The owner g written one coordinate at a time. Coordinate (r, c) takes entry(v, j) at the lower
# position of a point x whose matrix is symmetric (an owner, or an argument that has been symmetrized),
# j being the linear index (c - 1)p + r of the matrix where the parameters are read.
@inline function _lower_coordinates!(g::SymmetricSparseIterate, entry::F, x) where {F}
  p = g.p
  @inbounds for c in 1:p, r in c:p
    _setstored!(g, entry(x[r, c], (c - 1) * p + r), tril_index(p, r, c))
  end
  g
end
# The same for an entrywise gradient at any point: the symmetric part of the computed gradient, from
# the two positions of a pair when x is not an owner, whose matrix is symmetric.
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

# ⟨a, x⟩ over the matrix that the owner x represents, for a read at the linear indices of a p × p matrix:
# an off-diagonal coordinate counts at both of its positions, each with its own entry of a
function _owner_dot(a, x::SymmetricSparseIterate)
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

# A local proof that a parameter was checked, without changing its axes.
struct _CheckedParameter{T, A<:AbstractMatrix{T}} <: AbstractMatrix{T}
  array::A
end
Base.size(a::_CheckedParameter) = size(a.array)
Base.axes(a::_CheckedParameter) = axes(a.array)
Base.dataids(a::_CheckedParameter) = Base.dataids(a.array)
Base.IndexStyle(::Type{<:_CheckedParameter{T, A}}) where {T, A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(a::_CheckedParameter, I::Vararg{Int, N}) where {N} = a.array[I...]
_symmetric_by_type(::_CheckedParameter) = true
@inline _checked_parameter(y, a) = a
@inline _checked_parameter(y::SymmetricSparseIterate, a::AbstractMatrix) =
  _symmetric_by_type(a) ? a : _CheckedParameter(a)
_copy_parameter!(y, a) = copyto!(y, a)
# an owner output takes the values of a parameter matrix that has been checked to be symmetric, one
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

# Public value preflights prove the pairs used by these reductions equal.
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

# y = x for a proximal step that is the identity: an array takes the copy, an owner the lower entries of x,
# one coordinate at a time (a coordinate that is active stays active), and nothing when x is y
_assign!(y::AbstractArray, x) = copyto!(y, _vw(x))
_assign!(y::SymmetricSparseIterate, x) = (x === y ? y : _store_lower!(y, _vw(x)))
const _OwnerWrapperProx = Scaled
prox(f::_OwnerWrapperProx, x::SymmetricSparseIterate, γ::Real = 1) = prox!(similar(x, float(eltype(x))), f, x, γ)
const _OwnerWrapperSmooth = Scaled
gradient(f::_OwnerWrapperSmooth, x::SymmetricSparseIterate) = gradient!(_gradient_buffer(x), f, x)
function value_and_gradient(f::_OwnerWrapperSmooth, x::SymmetricSparseIterate)
  g = _gradient_buffer(x)
  v = value_and_gradient!(g, f, x)
  v, g
end
prox(f::ZeroFunction, x::SymmetricSparseIterate, γ::Real = 1) = prox!(similar(x, float(eltype(x))), f, x, γ)
