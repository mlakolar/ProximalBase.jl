####################################
#  linear
####################################

"""
    LinearFunction(c, d = 0)

The linear function `f(x) = ⟨c, x⟩ + d`, with gradient `c`. Its proximal operator
is the shift `x - γc`.

The coefficients and the argument have the same dimensions, `size(c) == size(x)`;
on a `SymmetricSparseIterate`, `c` is `p × p`. Every oracle throws a
`DimensionMismatch` for any other pair, even one with as many entries, before it writes.

At a matrix `SparseIterate` the value is the sum over its stored entries, in the storage
order, and reads `c` at the stored entries whose value is nonzero (a `Diagonal` `c` is read
at the nonzero stored diagonal entries, and the other entries of `x` are ignored, as in
`dot`). An entry that is not stored, or is stored as `+0` or `-0`, adds nothing, even if
the coefficient there is `NaN`, `Inf` or `-Inf`; the constant `d` is added after the sum.
A nonzero entry follows ordinary arithmetic, also when it or its coefficient is not finite.
When `d`, the nonzero entries and the coefficients at them are finite, the value is finite if
the exact value lies inside the range of the type by the accuracy of the sum, also if the
products or the partial sums overflow.

```julia
x = SparseIterate(2, 2); x[1, 1] = 2.0
f = LinearFunction([1.0 NaN; 1.0 1.0], 0.25)    # NaN at an entry of x that is not stored
f(x)                                            # 2.25
f(Matrix(x))                                    # NaN, as dot(f.c, Matrix(x)) + f.d
```

This rule is for the value at a matrix `SparseIterate`. `dot(c, x)` itself is the generic
`dot`, and a vector `SparseIterate` and a `SymmetricSparseIterate` keep their own rules.
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

# A matrix SparseIterate (D1): an entry of x that is not stored, or is stored as a zero, adds
# nothing to the value, also if the coefficient there is NaN or Inf. The sum runs over the stored
# entries, tests the value of each against zero before it reads c, and reads c at the
# nonzero ones, with the linear index of the entry (size(c) == size(x) is checked first).
# The value of x is read from the storage, not through getindex, which would form a zero.
function _value_unchecked(f::LinearFunction, x::SparseIterate{<:Any, 2})
  _checkvalue(f, x)
  _value_at(_linear_value(f, x, _linear_stored_dot(f.c, x) + f.d), x)
end
function _linear_stored_dot(c, x::SparseIterate{<:Any, 2})
  s = zero(promote_type(eltype(c), eltype(x)))
  @inbounds for k in 1:x.nnz
    v = x.nzval[k]
    iszero(v) && continue
    s += _entry(c, x.nzval2ind[k]) * v
  end
  s
end

# The recomputation for a matrix SparseIterate, when the ordinary value is not finite. The
# participating entries are the stored entries of x that are nonzero. It runs when d and
# the participating entries of x and of c are finite, and reads c only at the participating
# entries. Otherwise the ordinary value v stands: an Inf or a NaN in the data, not in a
# product, is a genuine result. A coefficient at a zero of x is not read, so it does not stop
# the recomputation.
@noinline function _linear_value_recover(c, d, x::SparseIterate{<:Any, 2}, v)
  isfinite(d) || return v
  R = typeof(v)
  mc = mx = zero(R)
  @inbounds for k in 1:x.nnz
    xk = x.nzval[k]
    iszero(xk) && continue
    a, b = abs(float(_entry(c, x.nzval2ind[k]))), abs(float(xk))
    (isfinite(a) && isfinite(b)) || return v
    mc, mx = max(mc, a), max(mx, b)
  end
  _linear_recovered_value(R, mc, mx, d, x.nnz, _linear_scaled_dot, c, x)
end
# Σ cᵢ (xᵢ 2^-k) over the participating entries of a matrix SparseIterate
function _linear_scaled_dot(c, x::SparseIterate{<:Any, 2}, s)
  v = zero(s[1])
  @inbounds for k in 1:x.nnz
    xk = x.nzval[k]
    iszero(xk) && continue
    v += float(_entry(c, x.nzval2ind[k])) * _scale_value(xk, s)
  end
  v
end
# the shift k of the units, the sum in them and d in its own units
function _linear_recovered_value(::Type{R}, mc, mx, d, n, scaled_dot, c, x) where {R}
  k = _product_sum_shift(R, mc, mx, n, 2)
  p = convert(R, scaled_dot(c, x, _power_factors(R, -k)))
  _pow2_sum(R, (p, k), (convert(R, d), 0))
end

# A Diagonal coefficient at a matrix SparseIterate: dot(c, x) for a Diagonal c reads the
# diagonal of x only, in O(p), and ignores the other entries of x, as f(Matrix(x)) does. The
# value keeps that, and reads c.diag[j] only where x has a nonzero stored diagonal entry.
function _value_unchecked(f::LinearFunction{<:Diagonal}, x::SparseIterate{<:Any, 2})
  _checkvalue(f, x)
  _value_at(_linear_value(f, x, _linear_diagonal_dot(f.c.diag, x) + f.d), x)
end
function _linear_diagonal_dot(dg, x::SparseIterate{<:Any, 2})
  s = zero(promote_type(eltype(dg), eltype(x)))
  @inbounds for j in 1:size(x, 1)
    pos = x.ind2nzval[j, j]
    pos == 0 && continue
    v = x.nzval[pos]
    iszero(v) && continue
    s += dg[j] * v
  end
  s
end
# The recomputation for a Diagonal coefficient: the participating entries are the nonzero stored
# diagonal entries of x. The other entries of x are not read, so a NaN there does not stop it.
@noinline function _linear_value_recover(c::Diagonal, d, x::SparseIterate{<:Any, 2}, v)
  isfinite(d) || return v
  R = typeof(v)
  mc = mx = zero(R)
  n = 0
  @inbounds for j in 1:size(x, 1)
    pos = x.ind2nzval[j, j]
    pos == 0 && continue
    xj = x.nzval[pos]
    iszero(xj) && continue
    a, b = abs(float(c.diag[j])), abs(float(xj))
    (isfinite(a) && isfinite(b)) || return v
    mc, mx = max(mc, a), max(mx, b)
    n += 1
  end
  _linear_recovered_value(R, mc, mx, d, n, _linear_diagonal_scaled_dot, c, x)
end
# Σ cⱼⱼ (xⱼⱼ 2^-k) over the nonzero stored diagonal entries of x
function _linear_diagonal_scaled_dot(c, x::SparseIterate{<:Any, 2}, s)
  v = zero(s[1])
  @inbounds for j in 1:size(x, 1)
    pos = x.ind2nzval[j, j]
    pos == 0 && continue
    xj = x.nzval[pos]
    iszero(xj) && continue
    v += float(c.diag[j]) * _scale_value(xj, s)
  end
  v
end

# c and x have the same dimensions, size(c) == size(x), the size of a symmetric
# iterate being that of the p × p matrix it represents. Every oracle applies this
# rule before it reads or writes, whatever dot would accept for the array types.
_checkparams(f::LinearFunction, x) = size(f.c) == size(x) || _coefficient_size_mismatch(size(f.c), size(x))
@noinline _coefficient_size_mismatch(sc, sx) =
  throw(DimensionMismatch("the coefficients c have size $sc and the argument has size $sx"))

function _checkwrite(op::O, f::LinearFunction, y::_ArrayOrSymmetricIterate, x::_ArrayOrSymmetricIterate) where {O}
  _checkreal(f, x)
  _checkreal(f, y)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(f, x)
  # c is written into y: copied by the gradient, subtracted by the prox
  _check_parameter_axes(y, f.c)
  true
end

prox!(y::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::LinearFunction, x::AbstractArray, γ::Real)
  _checkwrite(prox!, f, y, x)
  γ = convert(float(eltype(y)), γ)
  y .= x .- γ .* _assumed_symmetric(y, f.c)
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
kkt_violation(f::LinearFunction, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i] + f.c[i]))
restrict(f::LinearFunction, S::AbstractVector{<:Integer}) = LinearFunction(f.c[S], f.d)


lipschitz(f::LinearFunction) = zero(float(eltype(f.c)))
