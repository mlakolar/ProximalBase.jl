@inline _isbuiltin(f) = parentmodule(typeof(f)) === (@__MODULE__)

# Caller-supplied arrays in the iterate's entry domain. Design data and arrays
# indexed by groups, ranks or image coordinates are deliberately absent.
entry_parameters(::AbstractFunction) = ()
entry_parameters(::SeparableFunction) = nothing
for F in (:IndNonnegative, :NormL0, :MCP, :SCAD, :NegEntropy, :NegLog, :IndBanded)
  @eval entry_parameters(::$F) = ()
end
_entry_arrays(f) = entry_parameters(f)
_inner_functions(::AbstractFunction) = ()
for F in (:NormL1, :NormL1Nonneg, :ElasticNet, :CappedL1, :LogPenalty, :NormLq,
          :Berhu, :CheckLoss, :EpsInsensitiveLoss)
  @eval entry_parameters(f::$F) = (:w => f.w,)
end
entry_parameters(f::Union{HingeLoss, SqrHingeLoss}) = (:w => f.w, :y => f.y)
entry_parameters(f::BernoulliLoss) = (:w => f.w, :b => f.b)
entry_parameters(f::Union{Tilt, LinearFunction, Regularize}) = (:c => f.c,)
entry_parameters(f::KronQuadratic) = (:C => f.C,)
entry_parameters(f::Union{IndBox, IndBinary}) = (:l => f.l, :u => f.u)
entry_parameters(f::IndPattern) = (:P => f.P,)
entry_parameters(f::SqrNormL2) = (:y => f.y,)
entry_parameters(f::KLDivergence) = (:q => f.q,)
entry_parameters(f::Translate) = (:b => f.b,)
entry_parameters(f::Union{_ArrayPrecompose, IndHyperplane, IndHalfspace, IndHyperslab}) = (:a => f.a,)
_inner_functions(f::Union{Scaled, Tilt, Translate, Regularize, _ArrayPrecompose,
                         MoreauEnvelope, Conjugate, SqrDistL2, DistL2,
                         BallConstrained, SphereConstrained}) = (f.f,)
_inner_functions(f::_ScalarPrecompose) = (f.f,)
_inner_functions(f::BoxConstrained) = (f.f, f.box)
_inner_functions(f::LinearConstrained) = (f.f, f.set)
_inner_functions(f::Union{Sum, SeparableSum, PointwiseMinimum}) = f.fs
_inner_functions(f::IndIntersection) = f.sets

check_symmetric_parameters(f::AbstractFunction, x) = (_check_entry_arrays(f, x); true)

# Shape checks remain separate: a parameter vector can have matrix length,
# while a write of that parameter can still require the destination's axes.
_check_parameter_axes(y, a) = true
_check_parameter_axes(y::SymmetricSparseIterate, a::AbstractArray) = _check_axes(y, a)

struct _AllPairs end
struct _StoredValuePairs end
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
function _check_entry_arrays(f::SeparableSum, x::Tuple, domain = _AllPairs())
  _checkblocks(f, x)
  _check_entry_array_blocks(f.fs, x, domain)
end
@generated _check_entry_array_blocks(fs::Tuple, x::Tuple, domain) =
  _blockwise(k -> :(_check_entry_arrays(fs[$k], x[$k],
    domain isa _StoredValuePairs ? _stored_value_pairs(x[$k]) : _AllPairs())), min(fieldcount(fs), fieldcount(x)), true)
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
_stored_value(f::AbstractFunction) = isempty(entry_parameters(f)) && isempty(_inner_functions(f))
_stored_value(f::SeparableFunction) = _isbuiltin(f) && iszeropreserving(f)
_stored_value(::Union{LinearFunction, IndHyperplane, IndHalfspace, IndHyperslab}) = true
_stored_value(f::SqrNormL2) = f.y === nothing
_stored_value(f::Union{Scaled, Tilt, BallConstrained, SphereConstrained}) = _stored_value(f.f)
_stored_value(f::BoxConstrained) = _stored_value(f.f) && _stored_value(f.box)
_stored_value(f::LinearConstrained) = _stored_value(f.f) && _stored_value(f.set)
_stored_value(f::Regularize) = f.c === nothing && _stored_value(f.f)
_stored_value(f::Union{Sum, PointwiseMinimum}) = all(_stored_value, f.fs)
@generated _stored_value(f::SeparableSum{F}) where {F} =
  _blockall(k -> :(_stored_value(f.fs[$k])), fieldcount(F))
_stored_value(f::IndIntersection) = all(_stored_value, f.sets)
_stored_value_pairs(x::SymmetricSparseIterate) = _StoredPairs(x, x)
_stored_value_pairs(x::Tuple) = _StoredValuePairs()
_stored_value_pairs(x) = _AllPairs()
_stored_write(op, ::AbstractFunction) = false
_stored_write(op, f::SeparableFunction) = _isbuiltin(f)
_stored_write(op::O, f::Scaled) where {O} = _stored_write(op, f.f)
_stored_write(op::O, f::Conjugate{<:Conjugate}) where {O} = _stored_write(op, f.f.f)
_stored_box_value(::AbstractFunction) = false
_stored_box_value(f::SeparableFunction) = _isbuiltin(f)
_stored_box_value(f::Scaled) = _stored_box_value(f.f)
_stored_box_value(f::BoxConstrained) = _stored_box_value(f.f) && _stored_box_value(f.box)
_stored_box_value(f::Regularize) = f.c === nothing && _stored_box_value(f.f)
_builtin_kernels(f) = _isbuiltin(f) && all(_builtin_kernels, _inner_functions(f))
_stored_write(op::O, f::BoxConstrained) where {O} =
  _builtin_kernels(f.f) && (op === prox! || _stored_box_value(f.f))
_stored_write(op::O, f::_ArrayPrecompose) where {O} = _builtin_kernels(f.f) && op === prox!
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
  _check_value_shape_blocks(f.fs, x)
end
@generated _check_value_shape_blocks(fs::Tuple, x::Tuple) =
  _blockwise(k -> :(_check_value_shape(fs[$k], x[$k])), min(fieldcount(fs), fieldcount(x)), true)
_check_value_arrays(f, x::Tuple) =
  (_checkreal(f, x); _check_value_shape(f, x);
   _check_entry_arrays(f, x, _stored_value(f) ? _StoredValuePairs() : _AllPairs()))
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
function _check_write_arrays(op::O, f::SeparableSum, y::Tuple, x::Tuple) where {O}
  _checkblocks(f, x)
  _checksize(y, x)
  _check_write_array_blocks(f.fs, y)
end
@generated _check_write_array_blocks(fs::Tuple, y::Tuple) =
  _blockwise(k -> :(_check_entry_arrays(fs[$k], y[$k])), min(fieldcount(fs), fieldcount(y)), true)
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
_value_and_prox_unchecked!(y, f::AbstractFunction, x, γ) = value_and_prox!(y, f, x, γ)
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

_check_second_order_arrays(f, x) = (_checkreal(f, x); _check_entry_arrays(f, x))
function _check_second_order_arrays(f, x, Hv, v)
  _checkreal(f, Hv)
  _checkreal(f, x)
  _checkreal(f, v)
  _check_hessian_shape(f, x, Hv, v)
  _check_entry_arrays(f, x)
end
_check_lipschitz_arrays(f, L) =
  (eltype(L) <: Real || _real_error(:coordinate_lipschitz!); _check_entry_arrays(f, L))
_value_and_prox_unchecked!(y::AbstractMatrix, f::Union{IndPSD, IndBallRank}, x::AbstractMatrix, γ::Real) =
  value_and_prox!(y, f, x, γ)
_value_and_prox_unchecked!(y, f::IndStiefel, x, γ::Real) = value_and_prox!(y, f, x, γ)

_hessian_vec_unchecked!(Hv::AbstractVector, f::QuadraticFunction, x::AbstractVector, v::AbstractVector) =
  hessian_vec!(Hv, f, x, v)
_check_hessian_shape(f, x, Hv, v) = true
_check_hessian_shape(f::Union{ZeroFunction, LinearFunction, SqrNormL2}, x, Hv, v) =
  (_checksize(Hv, x); _checkdirection(v, x); _checkparams(f, x))
_check_hessian_shape(f::KronQuadratic, x, Hv, v) = (_checkshape(f, Hv); _checkshape(f, x))
_check_hessian_shape(f::Union{Scaled, Tilt}, x, Hv, v) =
  (_checkparams(f, x); _check_hessian_shape(f.f, x, Hv, v))
_check_hessian_shape(f::Translate, x, Hv, v) = _check_hessian_shape(f.f, _checkshifted(f, x), Hv, v)
_check_hessian_shape(f::Precompose, x, Hv, v) = (_checksize(Hv, x); _checkdirection(v, x))
_check_hessian_shape(f::Regularize, x, Hv, v) =
  (_checksize(Hv, x); _checkdirection(v, x); _checkparams(f, x); _check_hessian_shape(f.f, x, Hv, v))
_check_hessian_shape(f::Sum, x, Hv, v) = (_checksize(Hv, x); _checkdirection(v, x); _checkparams(f, x))
_check_hessian_shape(f::SeparableSum, x, Hv, v) = (_checksize(Hv, x); _checkdirection(v, x); _checkparams(f, x))

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
function _checked_scaled_dot(a, x::SymmetricSparseIterate, factors)
  v = zero(factors[1])
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    u = _scale_value(x.nzval[k], factors)
    ar = float(_entry(a, (c - 1) * x.p + r))
    ac = r == c ? zero(ar) : ar
    v += (ar + ac) * u
  end
  v
end
_value_dot_normal(a, x, ::Val{false}) = _dot_normal(a, x)
function _value_dot_normal(a, x::SymmetricSparseIterate, ::Val{true})
  T = float(promote_type(eltype(a), eltype(x)))
  _checked_scaled_dot(a, x, (one(T), one(T)))
end
_value_scaled_dot(a, x, s, ::Val{false}) = _scaled_dot(a, x, s)
_value_scaled_dot(a, x::SymmetricSparseIterate, s, ::Val{true}) = _checked_scaled_dot(a, x, s)
_value_checked(f::LinearFunction, x::SymmetricSparseIterate) =
  (_checkvalue(f, x); _value_at(_checked_value_dot(f.c, x) + f.d, x))
_value_checked(f::Union{IndHyperplane, IndHalfspace, IndHyperslab}, x::SymmetricSparseIterate) =
  _normal_value(f, x, Val(true))
_value_checked(f::Tilt, x::SymmetricSparseIterate) =
  _value_at(_value_checked(f.f, x) + _checked_value_dot(f.c, x) + f.d, x)
_value_checked(f::Scaled, x::Union{SymmetricSparseIterate, Tuple}) =
  _value_at(_scaled_value(f, x, _value_checked(f.f, x)), x)
_value_checked(f::Regularize, x::SymmetricSparseIterate) =
  (_checkparams(f, x); _value_at(_value_checked(f.f, x) + _ridge(f, x), x))
_value_checked(f::Sum, x::Union{SymmetricSparseIterate, Tuple}) =
  _value_at(sum(g -> _value_checked(g, x), f.fs), x)
_value_checked(f::SeparableSum, x::Tuple) =
  (_checkparams(f, x); _value_at(_sum_checked_values(f.fs, x), x))
@generated _sum_checked_values(fs::Tuple, x::Tuple) =
  _blocksum(k -> :(_value_checked(fs[$k], x[$k])), min(fieldcount(fs), fieldcount(x)))
_value_checked(f::PointwiseMinimum, x::SymmetricSparseIterate) =
  (_checkparams(f, x); _value_at(minimum(map(g -> _value_checked(g, x), f.fs)), x))
_value_checked(f::BoxConstrained, x::SymmetricSparseIterate) =
  (_checkparams(f, x); T = float(eltype(x));
   iszero(_value_checked(f.box, x)) ? convert(T, _value_checked(f.f, x)) : T(Inf))
_value_checked(f::LinearConstrained, x::SymmetricSparseIterate) =
  (_checkparams(f, x); T = float(eltype(x));
   iszero(_value_checked(f.set, x)) ? convert(T, _value_checked(f.f, x)) : T(Inf))
_value_checked(f::BallConstrained, x::SymmetricSparseIterate) =
  (T = float(eltype(x)); iszero(_value_checked(f.ball, x)) ? convert(T, _value_checked(f.f, x)) : T(Inf))
_value_checked(f::SphereConstrained, x::SymmetricSparseIterate) =
  (T = float(eltype(x)); iszero(_value_checked(f.sphere, x)) ? convert(T, _value_checked(f.f, x)) : T(Inf))
_value_checked(f::IndIntersection, x::SymmetricSparseIterate) =
  _ind(_in_checked_sets(f.sets, x), _valtype(x))
_in_checked_sets(::Tuple{}, x) = true
_in_checked_sets(sets::Tuple, x) = iszero(_value_checked(first(sets), x)) && _in_checked_sets(Base.tail(sets), x)
