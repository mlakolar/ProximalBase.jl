@inline _isbuiltin(f) = parentmodule(typeof(f)) === (@__MODULE__)

# Caller-supplied arrays in the iterate's entry domain. Design data and arrays
# indexed by groups, ranks or image coordinates are deliberately absent.
_entry_arrays(::AbstractFunction) = ()
_inner_functions(::AbstractFunction) = ()
for F in (:NormL1, :NormL1Nonneg, :ElasticNet, :CappedL1, :LogPenalty, :NormLq,
          :Berhu, :CheckLoss, :EpsInsensitiveLoss)
  @eval _entry_arrays(f::$F) = ((:w, f.w),)
end
_entry_arrays(f::Union{HingeLoss, SqrHingeLoss}) = ((:w, f.w), (:y, f.y))
_entry_arrays(f::BernoulliLoss) = ((:w, f.w), (:b, f.b))
_entry_arrays(f::Union{Tilt, LinearFunction, Regularize}) = ((:c, f.c),)
_entry_arrays(f::KronQuadratic) = ((:C, f.C),)
_entry_arrays(f::Union{IndBox, IndBinary}) = ((:l, f.l), (:u, f.u))
_entry_arrays(f::IndPattern) = ((:P, f.P),)
_entry_arrays(f::SqrNormL2) = ((:y, f.y),)
_entry_arrays(f::KLDivergence) = ((:q, f.q),)
_entry_arrays(f::Translate) = ((:b, f.b),)
_entry_arrays(f::Union{_ArrayPrecompose, IndHyperplane, IndHalfspace, IndHyperslab}) = ((:a, f.a),)
_inner_functions(f::Union{Scaled, Tilt, Translate, Regularize, _ArrayPrecompose,
                         MoreauEnvelope, Conjugate, SqrDistL2, DistL2,
                         BallConstrained, SphereConstrained}) = (f.f,)
_inner_functions(f::_ScalarPrecompose) = (f.f,)
_inner_functions(f::BoxConstrained) = (f.f, f.box)
_inner_functions(f::LinearConstrained) = (f.f, f.set)
_inner_functions(f::Union{Sum, SeparableSum, PointwiseMinimum}) = f.fs
_inner_functions(f::IndIntersection) = f.sets

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
  _check_own_arrays(f, _entry_arrays(f), x, domain)
  _check_inner_arrays(_inner_functions(f), x, domain)
end
_check_own_arrays(f, ::Tuple{}, x, domain) = true
function _check_own_arrays(f, arrays::Tuple, x, domain)
  name, a = first(arrays)
  _check_entry_array(f, name, a, x, domain)
  _check_own_arrays(f, Base.tail(arrays), x, domain)
end
_check_inner_arrays(::Tuple{}, x, domain) = true
function _check_inner_arrays(fs::Tuple, x, domain)
  _check_entry_arrays(first(fs), x, domain)
  _check_inner_arrays(Base.tail(fs), x, domain)
end
function _check_entry_arrays(f::SeparableSum, x::Tuple, domain = _AllPairs())
  _checkblocks(f, x)
  map((fk, xk) -> _check_entry_arrays(fk, xk), f.fs, x)
  true
end
function _check_entry_arrays(f::AbstractFunction, x::Tuple, domain = _AllPairs())
  _check_inner_arrays(_inner_functions(f), x, domain)
end
_check_entry_array(f, name, a, x, domain) = true
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
  _symequal(u, v) || _throw_asymmetric_parameter(f, name, r, c, u, v)
  true
end
function _check_pairs(f, name, a, p, ::_AllPairs)
  @inbounds for cb in 1:32:p, rb in cb:32:p
    for c in cb:min(cb + 31, p), r in max(c + 1, rb):min(rb + 31, p)
      u, v = a[(c - 1) * p + r], a[(r - 1) * p + c]
      _symequal(u, v) || _throw_asymmetric_parameter(f, name, r, c, u, v)
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
_stored_value(::AbstractFunction) = false
_stored_value(::SeparableFunction) = true
_stored_value(f::Scaled) = _stored_value(f.f)
_stored_value(f::BoxConstrained) = _stored_value(f.f) && _stored_value(f.box)
_stored_value(f::Regularize) = f.c === nothing && _stored_value(f.f)
_stored_write(op, ::AbstractFunction) = false
_stored_write(op, ::SeparableFunction) = true
_stored_write(op::O, f::Scaled) where {O} = _stored_write(op, f.f)
_stored_write(op::O, f::Conjugate{<:Conjugate}) where {O} = _stored_write(op, f.f.f)
_stored_write(op::O, f::BoxConstrained) where {O} = op === prox! || _stored_value(f.f)
_stored_write(op::O, f::_ArrayPrecompose) where {O} = op === prox!
_stored_domain(op::O, f) where {O} =
  (op === prox! || op === value_and_prox!) && isseparable(f) && iszeropreserving(f) && _stored_write(op, f)
# Keep the value oracle's shape errors ahead of the pair scan.
_check_value_shape(f, x) = _checkvalue(f, x)
_check_value_shape(f::Regularize, x) = _checkparams(f, x)
_check_value_shape(f::Scaled, x) = _check_value_shape(f.f, x)
_check_value_shape(f::_ScalarPrecompose, x) = _check_value_shape(f.f, _checkimage(f, x))
_check_value_shape(f::Translate, x) = _check_value_shape(f.f, _checkshifted(f, x))
_check_value_shape(f::Tilt, x) = (_checkreal(f, x); _check_value_shape(f.f, x); _checkdot(f.c, x))
_check_value_shape(f::Union{Sum, PointwiseMinimum}, x) = _check_value_shapes(f.fs, x)
_check_value_shapes(::Tuple{}, x) = true
_check_value_shapes(fs::Tuple, x) = (_check_value_shape(first(fs), x); _check_value_shapes(Base.tail(fs), x))
function _check_value_shape(f::SeparableSum, x::Tuple)
  _checkparams(f, x)
  map(_check_value_shape, f.fs, x)
  true
end
_check_value_arrays(f, x::Tuple) = (_check_value_shape(f, x); _check_entry_arrays(f, x))
_check_value_arrays(f, x) = _check_entry_arrays(f, x)
function _check_value_arrays(f, x::SymmetricSparseIterate)
  _check_value_shape(f, x)
  if isseparable(f) && iszeropreserving(f) && _stored_value(f)
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
function _check_write_arrays(op::O, f::SeparableSum, y::Tuple, x::Tuple) where {O}
  _checkblocks(f, x)
  _checksize(y, x)
  map((fk, yk) -> _check_entry_arrays(fk, yk), f.fs, y)
  true
end
_entry_steps(::AbstractFunction) = true
_entry_steps(::AbstractGroupPenalty) = false
_entry_steps(f::Union{Scaled, Regularize, Tilt, Translate, Conjugate, _ScalarPrecompose, _ArrayPrecompose}) = _entry_steps(f.f)
_check_steps(op, f, y, x, γ) = true
@inline function _check_steps(op::O, f, y::SymmetricSparseIterate, x, γ::AbstractArray) where {O}
  _entry_steps(f) || return true
  if x isa SymmetricSparseIterate && _stored_domain(op, f)
    _check_entry_array(nothing, :γ, γ, y, _StoredPairs(x, y))
  else
    _check_entry_array(nothing, :γ, γ, y, _AllPairs())
  end
end
_has_symmetric(x) = false
_has_symmetric(::SymmetricSparseIterate) = true
_has_symmetric(x::Tuple) = any(_has_symmetric, x)

# Only public boundaries check arrays. Inner kernels retain the existing shape
# preflights, but never scan a caller's arrays a second time.
function _check_oracle_arrays(op::O, f, y, x, γ = nothing) where {O}
  if _has_symmetric(y)
    _checkwrite(op, f, y, x)
    _check_write_arrays(op, f, y, x)
    _check_steps(op, f, y, x, γ)
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
  a, b = g.g[i], g.g[(r - 1) * g.p + c]
  _symequal(a, b) && return a
  isinf(a) && isinf(b) && a == -b && _throw_gradient_pair(r, c, a, b)
  a / 2 + b / 2
end
@noinline _throw_gradient_pair(r, c, a, b) = throw(ArgumentError(
  "the gradient g has no symmetric part: entry ($r, $c) is $a and entry ($c, $r) is $b"))

_kkt_violation(s::Scaled, g::AbstractArray, x::SymmetricSparseIterate, k::Integer) =
  iszero(s.α) ? _kkt_violation(ZeroFunction(), g, x, k) :
    s.α * _kkt_violation(s.f, _ScaledArray(g, inv(s.α)), x, k)
_kkt_violation(r::Regularize, g::AbstractArray, x::SymmetricSparseIterate, k::Integer) =
  _kkt_violation(r.f, _ridge_shifted(g, r), x, k)

# Fallbacks for functions outside the entry-array kernels.
_prox_unchecked!(y, f::AbstractFunction, x, γ) = prox!(y, f, x, γ)
_value_and_gradient_unchecked!(y, f::AbstractFunction, x) = value_and_gradient!(y, f, x)
_hessian_vec_unchecked!(y, f, x, v) = hessian_vec!(y, f, x, v)
_quadratic_model_unchecked!(q, f, x) = quadratic_model!(q, f, x)

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
_copy_parameter!(y::SymmetricSparseIterate, a::AbstractMatrix) = copyto!(y, _checked_parameter(y, a))

_hessian_vec_unchecked!(y::AbstractArray, f::Union{LeastSquaresLoss, GLMLoss, LogDetLoss}, x::AbstractArray, v::AbstractArray) = hessian_vec!(y, f, x, v)
_quadratic_model_unchecked(f::GLMLoss, x::AbstractVector) = quadratic_model(f, x)
_quadratic_model_unchecked(f::LogDetLoss, x::AbstractMatrix) = quadratic_model(f, x)
_coordinate_lipschitz_unchecked!(L::AbstractArray, f::LeastSquaresLoss) = coordinate_lipschitz!(L, f)

# Preserve the specialized joint oracles outside the entry-array domain.
_value_and_prox_unchecked!(y::AbstractArray, f::Union{ZeroFunction, NormL2, CubeNormL2, NormLinf,
    AbstractGroupPenalty, SliceSum, FiberSum, LinearImage, IndPolyhedron}, x::AbstractArray, γ::Real) =
  value_and_prox!(y, f, x, γ)
_value_and_prox_unchecked!(y::AbstractMatrix, f::Union{Spectral, SingularValue, NuclearNorm, TracePSD, LogDetLoss}, x::AbstractMatrix, γ::Real) =
  value_and_prox!(y, f, x, γ)
_value_and_prox_unchecked!(y::AbstractVector, f::Union{NormL1Fused, TotalVariation1D, FusedLasso, SLOPE, AffineConstrained}, x::AbstractVector, γ::Real) =
  value_and_prox!(y, f, x, γ)
_value_and_prox_unchecked!(y::AbstractArray{<:Real}, f::SumLargest, x::AbstractArray{<:Real}, γ::Real) =
  value_and_prox!(y, f, x, γ)
_value_and_prox_unchecked!(y, f::IndGraph, x, γ::Real) = value_and_prox!(y, f, x, γ)
_quadratic_model_unchecked!(q::GLMLoss{Squared, T, Vector{T}, <:Any, Vector{T}}, f::GLMLoss{L, T}, x::AbstractVector) where {L, T} =
  quadratic_model!(q, f, x)
_value_and_prox_unchecked!(y::AbstractArray, f::AbstractGroupPenalty, x::AbstractArray, γ::AbstractVector) =
  value_and_prox!(y, f, x, γ)
_value_unchecked(f::NormL1{<:Real, Nothing}, x::DenseArray{<:AbstractFloat}) = f(x)
