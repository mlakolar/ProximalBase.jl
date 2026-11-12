####################################
#  linear
####################################

"""
    LinearFunction(c, d = 0)

The linear function `f(x) = ⟨c, x⟩ + d`, with gradient `c`. Its proximal operator
is the shift `x - γc`.
"""
struct LinearFunction{C<:AbstractArray, D<:Real} <: AbstractFunction
  c::C
  d::D
  function LinearFunction{C, D}(c, d) where {C<:AbstractArray, D<:Real}
    eltype(c) <: Real || _real_error(:LinearFunction)
    new{C, D}(c, d)
  end
end
LinearFunction(c::AbstractArray, d::Real) = LinearFunction{typeof(c), typeof(d)}(c, d)

LinearFunction(c::AbstractArray) = (eltype(c) <: Real || _real_error(:LinearFunction); LinearFunction(c, zero(float(eltype(c)))))

issmooth(::Type{<:LinearFunction}) = true
isproximable(::Type{<:LinearFunction}) = true
isseparable(::Type{<:LinearFunction}) = true
isconvex(::Type{<:LinearFunction}) = true

(f::LinearFunction)(x::_ArrayOrSymmetricIterate) =
  (_check_value_arrays(f, x); _value_checked(f, x))
_value_unchecked(f::LinearFunction, x::AbstractArray) = (_checkvalue(f, x); _value_at(_linear_value(f, x, dot(f.c, x) + f.d), x))
# the matrix that the symmetric iterate represents, whose off-diagonal entries count at both positions
_value_unchecked(f::LinearFunction, x::SymmetricSparseIterate) =
  (_checkvalue(f, x); _value_at(_linear_value(f, x, _symmetric_matrix_dot(f.c, x) + f.d), x))

# A dot that overflows, alone or with d, at a representable value is summed
# again with x in units of 2^k, chosen so that no partial sum can overflow; d
# keeps its own units, so that a small d survives a dot that cancels (R1).
@inline _linear_value(f::LinearFunction, x, v) = isfinite(v) ? v : _linear_value_recover(f.c, f.d, x, v)
@noinline function _linear_value_recover(c, d, x, v)
  (_all_finite(x) && _all_finite(c) && isfinite(d)) || return v
  R = typeof(v)
  k = _product_sum_shift(R, _stored_maxabs(c), _stored_maxabs(x), _nentries(x), 2)
  p = convert(R, _linear_scaled_dot(c, x, _power_factors(R, -k)))
  _pow2_sum(R, (p, k), (convert(R, d), 0))
end
# Σ cᵢ (xᵢ 2^-k), over the stored entries of x where it has them; other array
# types (and device arrays) take the scaled copy of x with their own dot
_linear_scaled_dot(c, x::Union{DenseArray, SparseIterate}, s) =
  _is_device_array(x) || _is_device_array(c) ? _linear_scaled_copy_dot(c, x, s) : _scaled_dot(c, x, s)
# a symmetric iterate counts an off-diagonal entry for both of its positions:
# each position is a separate scaled product, so the doubled coefficient is
# never formed
function _linear_scaled_dot(c, x::SymmetricSparseIterate, s)
  v = zero(s[1])
  @inbounds for k in 1:x.nnz
    r, j = tril_cartesian(x.p, x.nzval2ind[k])
    q = float(_entry(c, (j - 1) * x.p + r)) * _scale_value(x.nzval[k], s)
    v += q
    r == j || (v += q)
  end
  v
end
_linear_scaled_dot(c, x, s) = _linear_scaled_copy_dot(c, x, s)
_linear_scaled_copy_dot(c, x, s) = dot(c, x .* s[1] .* s[2])

# Generic dot only needs equal lengths; specialized matrix methods check shapes.
_dot_checks_size(::Type{C}, ::Type{X}) where {C, X} =
  (C <: Diagonal && X <: AbstractMatrix) ||
  (C <: AbstractMatrix && X <: Diagonal) ||
  (C <: StridedMatrix && X <: SymmetricSparseIterate) ||
  (C <: _SparseDotCSC && X <: Union{_SparseDotCSC, _SparseDotMatrix}) ||
  (C <: _SparseDotMatrix && X <: _SparseDotCSC) ||
  (_SparseWrappedColumnDotChecksSize &&
   ((C <: _SparseDotWrappedCSC && X <: _SparseDotCSC) ||
    (C <: _SparseDotCSC && X <: _SparseDotWrappedCSC)))
_dot_checks_size(::Type{<:Adjoint{<:Real, C}}, ::Type{<:Adjoint{<:Real, X}}) where {C, X} = _dot_checks_size(C, X)
_dot_checks_size(::Type{<:Transpose{<:Any, C}}, ::Type{<:Transpose{<:Any, X}}) where {C, X} = _dot_checks_size(C, X)
_dot_checks_size(::Type{<:Adjoint{<:Any, C}}, ::Type{<:Transpose{<:Any, X}}) where {C, X} =
  _SparseMixedDotChecksSize && C <: _SparseDotCSC && X <: _SparseDotCSC
_dot_checks_size(::Type{<:Transpose{<:Any, C}}, ::Type{<:Adjoint{<:Any, X}}) where {C, X} =
  _SparseMixedDotChecksSize && C <: _SparseDotCSC && X <: _SparseDotCSC
_checkdot(c::C, x::X) where {C, X} = (_dot_checks_size(C, X) ? _checksize(c, x) : _checkparam(c, x))
_checkvalue(f::LinearFunction, x) = (_checkreal(f, x); _checkdot(f.c, x))

function _checkwrite(op::O, f::LinearFunction, y::_ArrayOrSymmetricIterate, x::_ArrayOrSymmetricIterate) where {O}
  _checkreal(f, x)
  _checkreal(f, y)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(f, x)
  op === value_and_prox! && _checkdot(f.c, y)
  op === value_and_gradient! && _checkdot(f.c, x)
  # c is written into y: copied by the gradient, subtracted by the prox
  _check_parameter_axes(y, f.c)
  true
end

prox!(y::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::LinearFunction, x::AbstractArray, γ::Real)
  _checkwrite(prox!, f, y, x)
  γ = convert(float(eltype(y)), γ)
  if size(f.c) == size(x)
    y .= x .- γ .* _checked_parameter(y, f.c)
  else
    # dot accepts equal-length arrays even when their shapes differ.
    @inbounds for i in eachindex(IndexLinear(), y, x)
      y[i] = x[i] - γ * f.c[i]
    end
  end
  y
end
# a symmetric iterate point and a dense output: the entries of the matrix that the symmetric iterate represents
_prox_unchecked!(y::AbstractArray, f::LinearFunction, x::SymmetricSparseIterate, γ::Real) =
  _prox_unchecked!(y, f, matrixview(x), γ)
# a symmetric iterate output: each coordinate once, from the lower entry of the symmetric argument
function _prox_unchecked!(y::SymmetricSparseIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate, γ::Real)
  _checkwrite(prox!, f, y, x)
  γ = convert(float(eltype(y)), γ)
  p = y.p
  @inbounds for c in 1:p, r in c:p
    _setstored!(y, x[r, c] - γ * f.c[(c - 1) * p + r], tril_index(p, r, c))
  end
  y
end

value_and_gradient!(g::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
function _value_and_gradient_unchecked!(g::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate)
  _checkwrite(value_and_gradient!, f, g, x)
  v = _value_unchecked(f, x)
  _copy_parameter!(g, f.c)
  v
end
gradient!(g::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
_gradient_unchecked!(g::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate) =
  (_checkwrite(gradient!, f, g, x); _copy_parameter!(g, f.c))

Base.@propagate_inbounds prox_block(f::LinearFunction, v::Real, i::Integer, γ::Real) = v - γ * f.c[i]
_checkparams(f::LinearFunction, x) = _checkparam(f.c, x)
kkt_violation(f::LinearFunction, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i] + f.c[i]))
restrict(f::LinearFunction, S::AbstractVector{<:Integer}) = LinearFunction(f.c[S], f.d)


lipschitz(f::LinearFunction) = zero(float(eltype(f.c)))
