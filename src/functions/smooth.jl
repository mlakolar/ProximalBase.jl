####################################
#
#  Smooth functions that are also proximable
#
####################################

####################################
#  zero
####################################

"""
    ZeroFunction()

The zero function `f(x) = 0`: smooth, with gradient zero, and proximable, with
the identity as proximal operator. The penalty of an unpenalized problem.
Array proximal copies use copyto!, so matching SparseIterate/BlockIterate
input and output replace active state when distinct. Owner proximal writes
preserve destination activity. Owner gradients are not implemented.
"""
struct ZeroFunction <: AbstractFunction end

issmooth(::Type{ZeroFunction}) = true
isproximable(::Type{ZeroFunction}) = true
isseparable(::Type{ZeroFunction}) = true
isconvex(::Type{ZeroFunction}) = true
iszeropreserving(::ZeroFunction) = true
_checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)},
            f::ZeroFunction, y::_ArrayOrOwner, x::_ArrayOrOwner) =
  (_checkreal(f, x); _checkreal(f, y); (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x); _checksize(y, x); true)

(f::ZeroFunction)(x::AbstractArray) = (_checkreal(f, x); zero(_valtype(x)))
# the value at an owner, which a wrapper of ZeroFunction needs (its other oracles at an owner are the proximal operator)
(f::ZeroFunction)(x::SymmetricSparseIterate) = (_checkreal(f, x); zero(_valtype(x)))

function prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, ::Real)
  x = _symmetric_argument(y, x)
  _checkwrite(prox!, f, y, x)
  y === x || copyto!(y, x)
  y
end
value_and_prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _prox_unchecked!(y, f, x, γ); zero(_valtype(y)))
# an owner output takes the symmetric part of the argument, one coordinate at a time: a coordinate that is
# active stays active (the gradient of ZeroFunction is defined for arrays only)
function prox!(y::SymmetricSparseIterate, f::ZeroFunction, x::_ArrayOrOwner, ::Real)
  x = _symmetric_argument(y, x)
  _checkwrite(prox!, f, y, x)
  _assign!(y, x)
end
value_and_prox!(y::SymmetricSparseIterate, f::ZeroFunction, x::_ArrayOrOwner, γ::Real) =
  (prox!(y, f, x, γ); zero(_valtype(y)))

function value_and_gradient!(g::AbstractArray, f::ZeroFunction, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  fill!(g, zero(eltype(g)))
  zero(_valtype(x))
end
gradient!(g::AbstractArray, f::ZeroFunction, x::AbstractArray) = (_value_and_gradient_unchecked!(g, f, x); g)

prox_block(::ZeroFunction, v::Real, ::Integer, ::Real) = v
kkt_violation(::ZeroFunction, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]))
restrict(f::ZeroFunction, ::AbstractVector{<:Integer}) = f

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

(f::LinearFunction)(x::_ArrayOrOwner) =
  (_check_value_arrays(f, x); _value_checked(f, x))
_value_unchecked(f::LinearFunction, x::AbstractArray) = (_checkvalue(f, x); _value_at(_linear_value(f, x, dot(f.c, x) + f.d), x))
# the matrix that the owner represents, whose off-diagonal entries count at both positions
_value_unchecked(f::LinearFunction, x::SymmetricSparseIterate) =
  (_checkvalue(f, x); _value_at(_linear_value(f, x, _owner_dot(f.c, x) + f.d), x))

# A dot that overflows, alone or with d, at a representable value is summed
# again with x in units of 2^k, chosen so that no partial sum can overflow; d
# keeps its own units, so that a small d survives a dot that cancels (R1).
@inline _linear_value(f::LinearFunction, x, v) = isfinite(v) ? v : _linear_value_recover(f.c, f.d, x, v)
@noinline function _linear_value_recover(c, d, x, v)
  (_all_finite(x) && _all_finite(c) && isfinite(d)) || return v
  R = typeof(v)
  E = _max_exponent(R)
  k = max(0, _bound_exponent(_stored_maxabs(c)) + _bound_exponent(_stored_maxabs(x)) + _log2_ceil(_nentries(x)) + 2 - E)
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

function _checkwrite(op::O, f::LinearFunction, y::_ArrayOrOwner, x::_ArrayOrOwner) where {O}
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

prox!(y::_ArrayOrOwner, f::LinearFunction, x::_ArrayOrOwner, γ::Real) =
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
# an owner point and a dense output: the entries of the matrix that the owner represents
_prox_unchecked!(y::AbstractArray, f::LinearFunction, x::SymmetricSparseIterate, γ::Real) =
  _prox_unchecked!(y, f, matrixview(x), γ)
# an owner output: each coordinate once, from the lower entry of the symmetric argument
function _prox_unchecked!(y::SymmetricSparseIterate, f::LinearFunction, x::_ArrayOrOwner, γ::Real)
  _checkwrite(prox!, f, y, x)
  γ = convert(float(eltype(y)), γ)
  p = y.p
  @inbounds for c in 1:p, r in c:p
    _setstored!(y, x[r, c] - γ * f.c[(c - 1) * p + r], tril_index(p, r, c))
  end
  y
end

value_and_gradient!(g::_ArrayOrOwner, f::LinearFunction, x::_ArrayOrOwner) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
function _value_and_gradient_unchecked!(g::_ArrayOrOwner, f::LinearFunction, x::_ArrayOrOwner)
  _checkwrite(value_and_gradient!, f, g, x)
  v = _value_unchecked(f, x)
  _copy_parameter!(g, f.c)
  v
end
gradient!(g::_ArrayOrOwner, f::LinearFunction, x::_ArrayOrOwner) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
_gradient_unchecked!(g::_ArrayOrOwner, f::LinearFunction, x::_ArrayOrOwner) =
  (_checkwrite(gradient!, f, g, x); _copy_parameter!(g, f.c))

Base.@propagate_inbounds prox_block(f::LinearFunction, v::Real, i::Integer, γ::Real) = v - γ * f.c[i]
_checkparams(f::LinearFunction, x) = _checkparam(f.c, x)
kkt_violation(f::LinearFunction, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i] + f.c[i]))
restrict(f::LinearFunction, S::AbstractVector{<:Integer}) = LinearFunction(f.c[S], f.d)

####################################
#  squared Euclidean norm
####################################

"""
    SqrNormL2(λ = 1)
    SqrNormL2(λ, y)

The squared Euclidean norm `f(x) = (λ/2) ‖x - y‖²`, with `y = 0` when omitted:
the ridge penalty, and for `λ = 1` the loss `‖x - y‖² / 2`. Smooth, with
gradient `λ(x - y)`, separable, and proximable, with proximal operator
`(x + γλy) / (1 + γλ)`. For a matrix `x`, `‖⋅‖` is the Frobenius norm.
Values follow the overflow rule in Overflow and underflow of values.

# Precision

The level `λ`, the center `y` and the step are parameters; they are rounded once
to the *model precision* of each result and never widen it. That precision is the wider of the argument's and the
destination's for a gradient or a proximal point, and the argument's for the value
(so the value written by `value_and_prox!` is the value at the stored point, and
`value_and_gradient!` returns the standalone value and writes the standalone
gradient). The proximal weight `γλ` is formed from the unrounded factors before it
is rounded.

A parameter, a product or a coefficient that would leave the range of the model
precision does not change a representable result: the gradient of
`SqrNormL2(2^-1022, [-2^1023])` at `[2^1023]` is `[4.0]` although `x - y`
overflows, and `prox!` of `SqrNormL2(2^800)` with step `2^800` at `[2^900]`
gives `2^-700` in `Float64`, where `1/(1 + γλ)` is not a number. The ordinary
computation runs first, and the recovery is chosen once per call from the
parameters, except for a `Float64` center entry outside the range of a `Float32`
model, which is handled for that entry.

"""
struct SqrNormL2{T<:Real, Y<:Union{Nothing, AbstractArray}} <: AbstractFunction
  λ::T
  y::Y

  function SqrNormL2{T, Y}(λ::T, y::Y) where {T<:Real, Y<:Union{Nothing, AbstractArray}}
    y === nothing || eltype(y) <: Real || _real_error(:SqrNormL2)
    _checknonneg("λ", λ)
    new{T, Y}(λ, y)
  end
end
SqrNormL2(λ::Real = 1, y::Union{Nothing, AbstractArray} = nothing) = SqrNormL2{typeof(λ), typeof(y)}(λ, y)

issmooth(::Type{<:SqrNormL2}) = true
isproximable(::Type{<:SqrNormL2}) = true
isseparable(::Type{<:SqrNormL2}) = true
isconvex(::Type{<:SqrNormL2}) = true
iszeropreserving(f::SqrNormL2) = f.y === nothing && isfinite(f.λ)

# Σ xᵢ² over the entries of x, in O(nnz) for the sparse iterates
_sumabs2(x::AbstractArray) = eltype(x) <: Integer ? sum(v -> abs2(float(v)), x) : sum(abs2, x)
function _sumabs2(x::SparseIterate)
  s = zero(float(eltype(x)))
  @inbounds @simd for k in 1:x.nnz
    s += abs2(float(x.nzval[k]))
  end
  s
end
function _sumabs2(x::SymmetricSparseIterate{T}) where {T}
  s = zero(float(T))
  @inbounds for k = 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    s += (r == c ? 1 : 2) * abs2(float(x.nzval[k]))
  end
  s
end
_sumabs2(x::BlockIterate) = _sumabs2(x.storage)

_square_term_count(x::Union{SparseIterate, SymmetricSparseIterate}) = x.nnz
_square_term_count(x::BlockIterate) = _square_term_count(x.storage)
function _square_multiplicity(x::SymmetricSparseIterate, i)
  r, c = tril_cartesian(x.p, x.nzval2ind[i])
  r == c ? 1 : 2
end

# Keep the vectorized sum-of-squares path for ordinary magnitudes; use a
# scaled norm when squaring overflows or loses the subnormal entries.
function _norm2(x::_ArrayOrOwner)
  eltype(x) <: AbstractFloat || return _norm2_scaled(x)
  s = _sumabs2(x)
  isfinite(s) && s >= floatmin(typeof(s)) && return sqrt(s)
  iszero(s) && x isa DenseArray && iszero(sum(abs, x)) && return s
  _norm2_scaled(x)
end
_norm2_scaled(x::AbstractArray) = norm(x)
_norm2_scaled(x::BlockIterate) = norm(x.storage)
function _norm2_scaled(x::SymmetricSparseIterate{T}) where {T}
  n = zero(float(T))
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    isnan(x.nzval[k]) && return oftype(n, NaN)
    n = hypot(n, x.nzval[k])
    r == c || (n = hypot(n, x.nzval[k]))
  end
  n
end

(f::SqrNormL2)(x::_ArrayOrOwner) =
  (_check_value_arrays(f, x); _value_checked(f, x))
function _value_unchecked(f::SqrNormL2, x::_ArrayOrOwner)
  # a center is read at every entry of the matrix that the owner represents
  x isa SymmetricSparseIterate && f.y !== nothing && return _value_unchecked(f, matrixview(x))
  _checkvalue(f, x)
  T = _valtype(x)
  iszero(f.λ) && return zero(T)
  y = f.y
  # A center is a parameter: it is rounded to the model precision of the value, the
  # argument's (T3), so a Float64 center does not widen a Float32 value.
  T === Float32 && y !== nothing && eltype(y) === Float64 && return _ridge_value_wide_center(f.λ, x, y)
  # a BigFloat center is rounded entry by entry, except where that leaves T's range (R3)
  y !== nothing && T <: Union{Float32, Float64} && float(eltype(y)) <: BigFloat &&
    return _square_value(f.λ, x, map(c -> _ridge_round_big(T, c), y))
  _square_value(f.λ, x, y)
end

_checkvalue(f::SqrNormL2, x) = (_checkreal(f, x); _checkparams(f, x); f.y === nothing || _checksize(f.y, x); true)

# The scaled-value mechanism: the recomputation of _square_value in power-of-two units, with the
# centers rounded as the value rounds them.
# Exact-zero protection: the ordinary value is λ/2 Σ (xᵢ - yᵢ)², with the sum in the type T of the
# value. A difference is exact where it is subnormal and zero only for equal entries. Where a square
# can underflow (the sum below n floatmin/eps), or the level does not fit T, the value is recomputed
# in power-of-two units and rounded once (_square_value), so a zero value there is below δ. Otherwise
# the sum is normal, with relative error below n eps, and a zero value is the product with λ/2
# rounded to zero: below δ/2 (1 + n eps). The centers are rounded as the value rounds them, both
# here and in the parts. So |f(x)| < n δ in every case, whatever λ is (λ = 0 is exact).
_has_value_parts(::Type{<:SqrNormL2}) = true
function _value_parts(f::SqrNormL2, x::AbstractArray, v)
  T = _valtype(x)
  y = f.y
  T === Float32 && y !== nothing && eltype(y) === Float64 && return _square_parts(f.λ, x, map(_ridge_round_center32, y))
  y !== nothing && T <: Union{Float32, Float64} && float(eltype(y)) <: BigFloat &&
    return _square_parts(f.λ, x, map(c -> _ridge_round_big(T, c), y))
  _square_parts(f.λ, x, y)
end
@inline _zero_value_exponent(f::SqrNormL2, x::AbstractArray) =
  iszero(f.λ) ? _EXACT_ZERO : _underflow_exponent(_valtype(x), length(x))
_parameter_alias(f::SqrNormL2, out) = _mightalias_parameter(out, f.y)

# Does a center entry keep its range when rounded to the model precision W (R3)?
# Branchless, so that a loop over the entries still vectorizes. Free for a center
# already stored in W; non-finite and zero entries convert without loss.
@inline _ridge_fits(::Type{W}, c::W) where {W} = true
@inline _ridge_fits(::Type{W}, c) where {W} = _level_fits(W, c)
@inline function _ridge_fits(::Type{Float32}, c::Float64)
  a = abs(c)
  ((a >= Float64(floatmin(Float32))) & (a <= Float64(floatmax(Float32)))) | iszero(a) | !isfinite(a)
end
# Did rounding c to cw underflow (to zero or a subnormal)? An overflow gives an infinite
# cw, which reaches the result and is caught there.
@inline _ridge_underflows(::Type{W}, c, cw) where {W} = false
@inline function _ridge_underflows(::Type{W}, c, cw::W) where {W<:Union{Float32, Float64}}
  c isa W && return false
  (abs(cw) < floatmin(W)) & !iszero(c)
end
# Can rounding the centers of eltype C to W leave W's range?
@inline _ridge_center_wider(::Type{W}, ::Type{C}) where {W, C} = promote_type(W, float(C)) !== W
# Every center entry keeps its range on rounding to W. One pass over c only, so an
# output that aliases the argument is still unwritten when a fallback is chosen.
function _ridge_all_fit(::Type{W}, c) where {W}
  ok = true
  for ci in c
    ok &= _ridge_fits(W, ci)
  end
  ok
end
# Float64 centers, Float32 model: the largest magnitude and the smallest nonzero
# magnitude decide (conservatively), compared as bit patterns, which vectorize.
# b - 1 wraps for b = 0, so zeros do not enter the minimum. NaN and infinities
# fail the test and take the checked path, which converts them without loss.
const _F32_MAX_BITS = reinterpret(UInt64, Float64(floatmax(Float32)))
const _F32_MIN_BITS = reinterpret(UInt64, Float64(floatmin(Float32)))
function _ridge_all_fit(::Type{Float32}, c::AbstractArray{Float64})
  hi = zero(UInt64)
  lo = typemax(UInt64)
  @inbounds @simd for i in eachindex(c)
    b = reinterpret(UInt64, c[i]) & 0x7fffffffffffffff
    hi = max(hi, b)
    lo = min(lo, b - one(UInt64))
  end
  hi <= _F32_MAX_BITS && lo >= _F32_MIN_BITS - one(UInt64)
end

# Σ (xᵢ - round(cᵢ))² for a Float32 argument and a Float64 center. The ordinary
# computation runs in Float32 when every center entry keeps its range on rounding;
# otherwise the centers that fit are rounded and the scaled recovery of
# _square_value sees them in Float64 (R3).
@inline _ridge_round_big(::Type{T}, c) where {T} = _level_fits(T, c) ? oftype(c, T(c)) : c
# a Float64 center entry rounded to Float32 where that keeps its range, as given otherwise
@inline _ridge_round_center32(c::Float64) = _ridge_fits(Float32, c) ? Float64(Float32(c)) : c
function _ridge_value_wide_center(λ, x, y)
  T = Float32
  s = 0.0           # the squares are exact in Float64; the sum accumulates there
  u = false
  @inbounds @simd for i in eachindex(x, y)
    cw = T(y[i])
    s += abs2(Float64(T(x[i]) - cw))
    u |= _ridge_underflows(T, y[i], cw)
  end
  sT = T(s)
  v = convert(T, λ) * sT / 2
  if !u & isfinite(v) & (sT >= length(x) * (floatmin(T) / eps(T))) && _level_fits(T, λ)
    return _value_at(v, x)
  end
  # a rounded center left the range, or the ordinary value is out of range
  _exceptional_square_value(λ, x, map(_ridge_round_center32, y), sT, v)
end

value_and_gradient!(g::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
function _value_and_gradient_unchecked!(g::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner)
  _checkwrite(value_and_gradient!, f, g, x)
  v = _value_unchecked(f, x)      # before the gradient, which may be written over the center
  _ridge_gradient!(g, f.λ, f.y, x)
  v
end
gradient!(g::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
function _gradient_unchecked!(g::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner)
  _checkwrite(gradient!, f, g, x)
  _ridge_gradient!(g, f.λ, f.y, x)
end

# an owner point with a dense output: the entries of the matrix that the owner represents
_ridge_gradient!(g::AbstractArray, λ, c, x::SymmetricSparseIterate) = _ridge_gradient!(g, λ, c, matrixview(x))
# an owner output: the symmetric part of the entrywise gradient, once per coordinate
function _ridge_gradient!(g::SymmetricSparseIterate, λ, c, x)
  W = _model_precision(eltype(g), eltype(x))
  if _level_fits(W, λ)
    _ridge_gradient_coordinates!(g, convert(W, λ), c, x)
  else
    _ridge_gradient_wide_coordinates!(g, W, convert(promote_type(W, float(typeof(λ))), λ), c, x)
  end
  g
end
_ridge_gradient_coordinates!(g, l, ::Nothing, x) = _symmetric_coordinates!(g, (v, j) -> l * v, x)
_ridge_gradient_coordinates!(g, l, c, x) =
  _symmetric_coordinates!(g, (v, j) -> _ridge_gradient_entry(l, v, @inbounds c[j]), x)
_ridge_gradient_wide_coordinates!(g, ::Type{W}, l, c, x) where {W} =
  _symmetric_coordinates!(g, (v, j) -> _ridge_gradient_wide(W, l, v, c === nothing ? zero(W) : @inbounds c[j]), x)
# gᵢ = λ(xᵢ - cᵢ) in the model precision W of the gradient: the wider of the
# destination and the argument. λ and c are rounded to W (T3), except where the
# rounding leaves W's range (R3); an overflowing xᵢ - cᵢ does not hide a finite
# gradient entry (R2).
function _ridge_gradient!(g, λ, c, x)
  W = _model_precision(eltype(g), eltype(x))
  if _level_fits(W, λ)
    l = convert(W, λ)
    c === nothing && return g .= l .* x
    cp = _checked_parameter(g, c)
    if g isa DenseArray && x isa DenseArray && cp isa DenseArray && size(g) == size(x) == size(cp) &&
       !Base.mightalias(g, cp)       # the repair pass rereads c
      _ridge_gradient_dense!(g, l, x, cp)
    else
      g .= _ridge_gradient_entry.(l, x, cp)
    end
  else
    R = promote_type(W, float(typeof(λ)))
    l = convert(R, λ)
    g .= _ridge_gradient_wide.(W, l, x, c === nothing ? zero(W) : _checked_parameter(g, c))
  end
  g
end
# The ordinary loop, which vectorizes; when a sum of the entries is not finite (an
# overflowing difference or product, or a center that overflowed on rounding) or a
# center underflowed, every entry is recomputed one by one. The gradient never aliases x.
function _ridge_gradient_dense!(g, l::W, x, c) where {W}
  t = zero(W)       # a non-finite entry makes the sum non-finite
  u = false
  @inbounds @simd for i in eachindex(g)
    cw = W(c[i])
    r = l * (W(x[i]) - cw)
    g[i] = r
    t += r
    u |= _ridge_underflows(W, c[i], cw)
  end
  (u | !isfinite(t)) && _ridge_gradient_repair!(g, l, x, c)
  g
end
@noinline function _ridge_gradient_repair!(g, l::W, x, c) where {W}
  @inbounds for i in eachindex(g)
    g[i] = _ridge_gradient_entry(l, x[i], c[i])
  end
end
@inline function _ridge_gradient_entry(l::W, x, c) where {W}
  _ridge_fits(W, c) || return _ridge_gradient_wide(W, l, x, c)
  r = l * (W(x) - W(c))
  isfinite(r) ? r : _ridge_gradient_overflow(l, W(x), W(c), r)
end
@noinline _ridge_gradient_overflow(l, x, c, r) =
  isfinite(x) && isfinite(c) ? _scaled_product(l, x / 2 - c / 2, 1) : r
# l is a level wider than W, or c is a center wider than W: the entry is computed in R
function _ridge_gradient_wide(::Type{W}, l, x, c) where {W}
  R = promote_type(W, typeof(l), float(typeof(c)))
  cw = _ridge_fits(W, c) ? R(W(c)) : R(c)
  r = R(l) * (R(x) - cw)
  isfinite(r) ? r : _ridge_gradient_overflow(R(l), R(x), cw, r)
end

prox!(z::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(z, x); _check_oracle_arrays(prox!, f, z, x, γ); _prox_unchecked!(z, f, x, γ))
_prox_unchecked!(z::AbstractArray, f::SqrNormL2, x::SymmetricSparseIterate, γ::Real) =
  _prox_unchecked!(z, f, matrixview(x), γ)
function _prox_unchecked!(z::_ArrayOrOwner, f::SqrNormL2, x::_ArrayOrOwner, γ::Real)
  _checkwrite(prox!, f, z, x)
  W = _model_precision(eltype(z), eltype(x))
  c = f.y === nothing ? nothing : _checked_parameter(z, f.y)
  if _ridge_needs_scaling(W, γ, f.λ)
    _ridge_prox!(z, _ridge_scaled(W, γ, f.λ), x, c)
  else
    _ridge_prox!(z, _ridge_ordinary(W, γ, f.λ), x, c)
  end
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}, f::SqrNormL2, y::_ArrayOrOwner, x::_ArrayOrOwner)
  _checkreal(f, x)
  _checkreal(f, y)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(f, x)
  f.y === nothing || (_checksize(f.y, x); _check_parameter_axes(y, f.y))
  true
end

# Interpolate same-sign endpoints from the endpoint with larger weight,
# so rounding in two weighted products cannot overflow near floatmax.
@inline function _ridge_blend(x, y, a::T, b::T) where {T}
  x, y = T(x), T(y)
  if signbit(x) == signbit(y)
    return b <= a ? muladd(b, y - x, x) : muladd(a, x - y, y)
  end
  a * x + b * y
end

# The ridge proximal point (x + γλc)/(1 + γλ) = a x + b c with the convex weights
# a = 1/(1+t), b = t/(1+t), t = γλ. t is formed once from the unrounded factors in
# P = promote_type(W, factors) and rounded to the model precision W (T3).
#
# Ordinary: t, 1/t and both weights are normal numbers in W (or a factor is zero or
# not finite, which keeps its earlier meaning). Scaled (R3): t = τ·2^k from the
# significands; the smaller weight is carried as m·2^e and applied last, so a
# coefficient that is not representable still acts on the coordinate.
struct _RidgeOrdinary{W<:Real}
  a::W
  b::W
  t::W
end
struct _RidgeScaled{W<:Real, P<:Real}
  L::W              # the larger weight, in [1/2, 1]
  m::W              # significand of the smaller weight, in (0, 1]
  e::Int
  small_x::Bool     # the smaller weight multiplies x (t > 1); else the center
  γ::P
  λ::P
end

@inline function _ridge_weights_t(t::W) where {W}
  if t <= one(W)
    d = one(W) + t
    return inv(d), t / d
  end
  ρ = inv(t)
  d = one(W) + ρ
  ρ / d, inv(d)
end

# The decision is made once per call, from the factors, not from t: the product of
# finite nonzero factors can underflow or overflow in P.
@inline function _ridge_needs_scaling(::Type{W}, γ, λ) where {W}
  W <: Union{Float32, Float64} || return false
  P = promote_type(W, float(typeof(γ)), float(typeof(λ)))
  t = P(γ) * P(λ)
  # t in the band implies finite nonzero factors, so they are examined only outside it
  (P(_ridge_band_low(W)) <= t <= P(_ridge_band_high(W))) && return false
  isfinite(γ) & isfinite(λ) & !iszero(γ) & !iszero(λ)
end
# t, 1/t and both convex weights are normal in W for t in [2^-126, 2^125] (Float32)
# and [2^-1022, 2^1021] (Float64)
@inline _ridge_band_low(::Type{Float32}) = reinterpret(Float32, 0x00800000)          # 2^-126
@inline _ridge_band_high(::Type{Float32}) = reinterpret(Float32, 0x7e000000)         # 2^125
@inline _ridge_band_low(::Type{Float64}) = reinterpret(Float64, 0x0010000000000000)   # 2^-1022
@inline _ridge_band_high(::Type{Float64}) = reinterpret(Float64, 0x7fc0000000000000)  # 2^1021
@inline function _ridge_ordinary(::Type{W}, γ, λ) where {W}
  P = promote_type(W, float(typeof(γ)), float(typeof(λ)))
  tw = convert(W, P(γ) * P(λ))
  a, b = _ridge_weights_t(tw)
  # A level that is zero as given and the step Inf as given (not as converted, which turns a finite step
  # beyond the range into Inf) make the product γλ Inf⋅0. The weights are then those of the step 1,
  # which are also those of every positive finite step: (1, ±0) with the sign of the level. Any other
  # step, NaN and negative ones included, keeps the arithmetic of the product. The selects follow the
  # weights, so that they wait behind the division, and they are no branch to a call: the loops over the
  # entries of a ridge keep their vectorization.
  endpoint = iszero(λ) & (γ == Inf)
  t1 = convert(W, one(P) * P(λ))
  _RidgeOrdinary{W}(ifelse(endpoint, one(W), a), ifelse(endpoint, t1, b), ifelse(endpoint, t1, tw))
end
@inline _ridge_scaled(::Type{W}, γ, λ) where {W} =
  (P = promote_type(W, float(typeof(γ)), float(typeof(λ))); _ridge_scaled(W, P(γ), P(λ)))

Base.@assume_effects :effect_free :nothrow :terminates_globally @noinline function _ridge_scaled(::Type{W}, γ::P, λ::P) where {W, P}
  mγ, eγ = _frexp(γ)
  mλ, eλ = _frexp(λ)
  τ = convert(W, mγ * mλ)                    # in [1/4, 1]
  k = eγ + eλ
  if k > 0
    r = inv(τ)
    d = one(W) + r * ldexp(one(W), -k)
    _RidgeScaled{W, P}(inv(d), r / d / 4, 2 - k, true, γ, λ)
  else
    d = one(W) + τ * ldexp(one(W), k)
    _RidgeScaled{W, P}(inv(d), τ / d, k, false, γ, λ)
  end
end

@inline function _ridge_scaled_blend(x::W, c::W, s::_RidgeScaled{W}) where {W}
  big, small = s.small_x ? (c, x) : (x, c)
  signbit(big) == signbit(small) ? big + ldexp(s.m * (small - big), s.e) :
                                   s.L * big + ldexp(s.m * small, s.e)
end

@inline _ridge_apply(o::_RidgeOrdinary, x) = o.a * x
@inline _ridge_apply(s::_RidgeScaled{W}, x) where {W} =
  s.small_x ? ldexp(s.m * W(x), s.e) : s.L * W(x)
@inline function _ridge_apply(o::_RidgeOrdinary{W}, x, c) where {W}
  _ridge_fits(W, c) ? _ridge_blend(x, c, o.a, o.b) : _ridge_apply_wide(o, x, c)
end
@inline function _ridge_apply(s::_RidgeScaled{W}, x, c) where {W}
  _ridge_fits(W, c) ? _ridge_scaled_blend(W(x), W(c), s) : _ridge_apply_wide(s, x, c)
end
# the same when every center entry is known to fit: no per-entry test
@inline _ridge_apply_fit(o::_RidgeOrdinary, x, c) = _ridge_blend(x, c, o.a, o.b)
@inline _ridge_apply_fit(s::_RidgeScaled{W}, x, c) where {W} = _ridge_scaled_blend(W(x), W(c), s)
# A center outside W's range on rounding: the entry is computed in the wider type R.
# For a hardware-float center, R holds x and c with room to spare (it is wider than W,
# which holds x), so the plain convex combination cannot overflow. prox_block inlines
# it, so that a loop over the entries has no call and vectorizes (its weights do not
# depend on the entry); the array methods call it, as before.
@inline function _ridge_wide_entry(o::_RidgeOrdinary{W}, x, c::Base.IEEEFloat) where {W}
  R = promote_type(W, typeof(c))
  a, b = _ridge_weights_t(convert(R, o.t))
  W(a * R(x) + b * c)
end
@noinline _ridge_apply_wide(o::_RidgeOrdinary, x, c::Base.IEEEFloat) = _ridge_wide_entry(o, x, c)
@noinline function _ridge_apply_wide(o::_RidgeOrdinary{W}, x, c) where {W}
  R = promote_type(W, float(typeof(c)))
  a, b = _ridge_weights_t(convert(R, o.t))
  W(_ridge_blend(R(x), R(c), a, b))
end
@noinline function _ridge_apply_wide(s::_RidgeScaled{W}, x, c) where {W}
  R = promote_type(W, float(typeof(c)))
  W(_ridge_scaled_blend(R(x), R(c), _ridge_scaled(R, promote_type(R, typeof(s.γ))(s.γ), promote_type(R, typeof(s.λ))(s.λ))))
end

# an owner output: each coordinate once, from the lower entry of the symmetric argument
_ridge_prox!(z::SymmetricSparseIterate, o::_RidgeOrdinary, x, ::Nothing) =
  _lower_coordinates!(z, (v, j) -> _ridge_apply(o, v), x)
_ridge_prox!(z::SymmetricSparseIterate, s::_RidgeScaled, x, ::Nothing) =
  _lower_coordinates!(z, (v, j) -> _ridge_apply(s, v), x)
function _ridge_prox!(z::SymmetricSparseIterate, co::Union{_RidgeOrdinary{W}, _RidgeScaled{W}}, x, c) where {W}
  if _ridge_center_wider(W, eltype(c)) && !_ridge_all_fit(W, c)
    _lower_coordinates!(z, (v, j) -> _ridge_apply(co, v, @inbounds c[j]), x)
  else
    _lower_coordinates!(z, (v, j) -> _ridge_apply_fit(co, v, @inbounds c[j]), x)
  end
end
_ridge_prox!(z, o::_RidgeOrdinary, x, ::Nothing) = (z .= o.a .* x)
_ridge_prox!(z, s::_RidgeScaled, x, ::Nothing) = (z .= (xi -> _ridge_apply(s, xi)).(x))
function _ridge_prox!(z, co::Union{_RidgeOrdinary{W}, _RidgeScaled{W}}, x, c) where {W}
  # a center wider than W is tested once, before the loop: the pass reads c only,
  # so an aliased output (z === x) is still unwritten when a fallback is chosen
  if _ridge_center_wider(W, eltype(c)) && !_ridge_all_fit(W, c)
    z .= ((xi, ci) -> _ridge_apply(co, xi, ci)).(x, c)
  else
    z .= ((xi, ci) -> _ridge_apply_fit(co, xi, ci)).(x, c)
  end
end

@inline function prox_block(f::SqrNormL2{<:Real, Nothing}, v::Real, ::Integer, γ::Real)
  W = typeof(float(v))
  _ridge_needs_scaling(W, γ, f.λ) ? _ridge_apply(_ridge_scaled(W, γ, f.λ), W(v)) :
                                    _ridge_apply(_ridge_ordinary(W, γ, f.λ), W(v))
end
Base.@propagate_inbounds function prox_block(f::SqrNormL2, v::Real, i::Integer, γ::Real)
  W = typeof(float(v))
  _ridge_needs_scaling(W, γ, f.λ) ? _ridge_apply(_ridge_scaled(W, γ, f.λ), W(v), f.y[i]) :
                                    _ridge_apply_entry(_ridge_ordinary(W, γ, f.λ), W(v), f.y[i])
end
@inline _ridge_apply_entry(o::_RidgeOrdinary, x, c) = _ridge_apply(o, x, c)
@inline _ridge_apply_entry(o::_RidgeOrdinary, x, c::Base.IEEEFloat) =
  _ridge_fits(typeof(o.a), c) ? _ridge_blend(x, c, o.a, o.b) : _ridge_wide_entry(o, x, c)
# The ridge whose entries are known to take the ordinary coefficient, with every center
# entry keeping its range in W: prox_block without the tests, where
# _ridge_needs_scaling(W, γ, f.λ) is false and the centers fit. Constrained uses it for
# steps and centers it has checked, so that its loops have no call.
struct _OrdinaryRidge{F<:SqrNormL2}
  f::F
end
@inline prox_block(r::_OrdinaryRidge{<:SqrNormL2{<:Real, Nothing}}, v::Real, ::Integer, γ::Real) =
  (W = typeof(float(v)); _ridge_apply(_ridge_ordinary(W, γ, r.f.λ), W(v)))
Base.@propagate_inbounds prox_block(r::_OrdinaryRidge, v::Real, i::Integer, γ::Real) =
  (W = typeof(float(v)); _ridge_apply_fit(_ridge_ordinary(W, γ, r.f.λ), W(v), r.f.y[i]))
# The slope of the ridge proximal point in its argument is the weight that multiplies it:
# the larger weight, or the smaller one carried as m·2^e (which rounds to 0 only where
# the weight itself underflows). The center does not enter it.
@inline _ridge_slope(o::_RidgeOrdinary) = o.a
@inline _ridge_slope(s::_RidgeScaled) = s.small_x ? ldexp(s.m, s.e) : s.L

# prox_block and the slope of the ridge, from one coefficient: the value is computed by
# the code of prox_block, so that the two cannot disagree.
@inline function _ridge_prox_slope(f::SqrNormL2{<:Real, Nothing}, v::Real, ::Integer, γ::Real)
  W = typeof(float(v))
  if _ridge_needs_scaling(W, γ, f.λ)
    s = _ridge_scaled(W, γ, f.λ)
    _ridge_apply(s, W(v)), _ridge_slope(s)
  else
    o = _ridge_ordinary(W, γ, f.λ)
    _ridge_apply(o, W(v)), _ridge_slope(o)
  end
end
Base.@propagate_inbounds function _ridge_prox_slope(f::SqrNormL2, v::Real, i::Integer, γ::Real)
  W = typeof(float(v))
  if _ridge_needs_scaling(W, γ, f.λ)
    s = _ridge_scaled(W, γ, f.λ)
    _ridge_apply(s, W(v), f.y[i]), _ridge_slope(s)
  else
    o = _ridge_ordinary(W, γ, f.λ)
    _ridge_apply_entry(o, W(v), f.y[i]), _ridge_slope(o)
  end
end
@inline _ridge_prox_slope(r::_OrdinaryRidge{<:SqrNormL2{<:Real, Nothing}}, v::Real, ::Integer, γ::Real) =
  (W = typeof(float(v)); o = _ridge_ordinary(W, γ, r.f.λ); (_ridge_apply(o, W(v)), _ridge_slope(o)))
Base.@propagate_inbounds _ridge_prox_slope(r::_OrdinaryRidge, v::Real, i::Integer, γ::Real) =
  (W = typeof(float(v)); o = _ridge_ordinary(W, γ, r.f.λ); (_ridge_apply_fit(o, W(v), r.f.y[i]), _ridge_slope(o)))
_checkparams(f::SqrNormL2, x) = _checkparam(f.y, x)

kkt_violation(::SqrNormL2{<:Real, Nothing}, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]))
kkt_violation(f::SqrNormL2, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i] - f.λ * f.y[i]))

restrict(f::SqrNormL2{<:Real, Nothing}, ::AbstractVector{<:Integer}) = f
restrict(f::SqrNormL2, S::AbstractVector{<:Integer}) = SqrNormL2(f.λ, f.y[S])

####################################
#  quadratic
####################################

"""
    QuadraticFunction(A, b = nothing, c = 0)

The quadratic `f(x) = x'Ax / 2 + b'x + c` for a symmetric positive semidefinite
`A`, with `b = 0` when omitted. Smooth, with gradient `Ax + b`, and proximable:
the first proximal operator computes the eigendecomposition `A = QΛQ'` once,
after which `prox_{γf}(x) = Q (I + γΛ)⁻¹ Q'(x - γb)` costs two products with
`Q` for any positive finite `γ`.

The constructor checks real types, squareness and the length of b, but does
not verify symmetry or positive semidefiniteness. Those are caller assumptions.
Values/gradients use A directly; the cached eigenbasis uses Symmetric(Matrix(A))
with its default upper triangle. Use vectors of compatible length. Data are
retained by reference; after same-shape A changes call reset_cache!.
"""
mutable struct QuadraticFunction{T<:AbstractFloat, M<:AbstractMatrix, V<:Union{Nothing, AbstractVector}} <: AbstractFunction
  A::M
  b::V
  c::T
  tmp::Vector{T}        # A x, and scratch for the proximal operator
  Λ::Vector{T}          # A = Q Diagonal(Λ) Q', computed by the first prox
  Q::Matrix{T}
  w::Vector{T}

  function QuadraticFunction{T, M, V}(A, b, c, tmp, Λ, Q, w) where {T<:AbstractFloat, M<:AbstractMatrix,
      V<:Union{Nothing, AbstractVector}}
    all(a -> a === nothing || eltype(a) <: Real,
        (A, b, tmp, Λ, Q, w)) || _real_error(:QuadraticFunction)
    new{T, M, V}(A, b, c, tmp, Λ, Q, w)
  end
end
QuadraticFunction(A::M, b::V, c::T, tmp::Vector{T}, Λ::Vector{T}, Q::Matrix{T}, w::Vector{T}) where {T<:AbstractFloat,
    M<:AbstractMatrix, V<:Union{Nothing, AbstractVector}} =
  QuadraticFunction{T, M, V}(A, b, c, tmp, Λ, Q, w)

function QuadraticFunction(A::AbstractMatrix, b::Union{Nothing, AbstractVector} = nothing, c::Real = 0)
  eltype(A) <: Real || _real_error(:QuadraticFunction)
  b === nothing || eltype(b) <: Real || _real_error(:QuadraticFunction)
  n = checksquare(A)
  b === nothing || length(b) == n || throw(DimensionMismatch("A is $n × $n and b has length $(length(b))"))
  T = float(promote_type(eltype(A), b === nothing ? eltype(A) : eltype(b), typeof(c)))
  QuadraticFunction{T, typeof(A), typeof(b)}(A, b, T(c), zeros(T, n), T[], Matrix{T}(undef, 0, 0), zeros(T, n))
end

issmooth(::Type{<:QuadraticFunction}) = true
_argument_type(::Type{<:QuadraticFunction}) = AbstractVector
isproximable(::Type{<:QuadraticFunction}) = true
isconvex(::Type{<:QuadraticFunction}) = true

function (f::QuadraticFunction)(x::AbstractVector)
  _checkreal(f, x)
  m = _mul_maxabs!(f.tmp, f.A, x)
  v = dot(x, f.tmp) / 2 + f.c
  v = f.b === nothing ? v : v + dot(f.b, x)
  _value_at(_quadratic_checked(f, x, v, eltype(f.tmp), m), x)
end

function value_and_gradient!(g::AbstractVector, f::QuadraticFunction, x::AbstractVector)
  _checkwrite(value_and_gradient!, f, g, x)
  if eltype(g) !== eltype(x)
    _gradient_unchecked!(g, f, x)
    return _value_unchecked(f, x)
  end
  m = _mul_maxabs!(g, f.A, x)
  v = dot(g, x) / 2 + f.c
  if f.b !== nothing
    g .+= f.b
    v += dot(f.b, x)
  end
  _value_at(_quadratic_checked(f, x, v, eltype(g), m), x)
end

# y = A x, and max|x| for the trigger below. For a diagonal A the two are one pass: y[i] = d[i] x[i] is the
# product that mul! rounds into y, so y is the same, and the maximum is read where x[i] is already loaded.
@inline function _mul_maxabs!(y, A, x)
  mul!(y, A, x)
  _stored_maxabs(x)
end
@inline function _mul_maxabs!(y::StridedVector{P}, A::Diagonal{P}, x::StridedVector{P}) where {P<:Union{Float32, Float64}}
  U = _int_type(P)
  d = A.diag
  (length(y) == length(d) == length(x)) || return invoke(_mul_maxabs!, Tuple{Any, Any, Any}, y, A, x)
  m = zero(U)
  @inbounds @simd for i in eachindex(y)
    xi = x[i]
    y[i] = d[i] * xi
    m = max(m, reinterpret(U, xi) & typemax(U))
  end
  reinterpret(P, m)
end

# The ordinary value, or its recomputation from the exact products: after an overflow (a non-finite
# value of finite data), or when the products A x rounded into the buffer type P, and the final dots,
# could have lost more than eps |v| (R1, #520).
@inline _quadratic_checked(f::QuadraticFunction, x, v, ::Type{P}) where {P} =
  _quadratic_checked(f, x, v, P, _stored_maxabs(x))
@inline function _quadratic_checked(f::QuadraticFunction, x, v, ::Type{P}, m) where {P}
  if !isfinite(v) || (_quadratic_underflow_risk(f, x, v, P, m) && !_quadratic_products_normal(f, x, v, P))
    _quadratic_value_recover(f, x, v)
  else
    v
  end
end

# The threshold of _quadratic_underflow_risk is a worst case: it assumes that products may be subnormal.
# When none can be, the ordinary value carries only relative errors, and the value is as accurate as the
# ordinary computation (R1), however small it is, an exact zero of a null-space argument included.
# Every nonzero product of two operands is normal if the sum of the exponents of their smallest nonzero
# entries is at least exponent(floatmin): |a b| >= 2^(ea + eb) (Julia's exponent has a significand in
# [1, 2)); a product with a zero is exact; and a sum with a subnormal result is exact. The products are
# those of A x, of x and A x in the dot, and of b and x. The test is made on the operands and on
# A x, which is recomputed into f.tmp (the buffer may hold A x + b, and an ordinary call has not
# modified it; the cost is that of one product, on a path that is rare). It needs the whole computation
# in the type P (a wider argument or buffer is rounded elsewhere). Data need not be tested for being
# finite: a finite value has none that is not (an Inf makes a product Inf or NaN).
@noinline function _quadratic_products_normal(f::QuadraticFunction, x, v, ::Type{P}) where {P}
  (_underflow_checked(P) && eltype(x) === P && eltype(f.tmp) === P && typeof(v) === P &&
   eltype(f.A) <: Union{Float32, Float64} && (f.b === nothing || eltype(f.b) <: Union{Float32, Float64})) ||
    return false
  emin = exponent(floatmin(P))
  ex = _stored_minexp(x)
  _stored_minexp(f.A) + ex >= emin || return false
  f.b === nothing || _stored_minexp(f.b) + ex >= emin || return false
  mul!(f.tmp, f.A, x)
  _min_nonzero_exponent(f.tmp) + ex >= emin
end

# Rounding a product into P changes an entry of A x by at most denorm_min(P)/2 per term, and that
# error is multiplied by an entry of x; the products x[i] (A x)[i] and b[i] x[i] of the dots lose up to
# denorm_min(P)/2 each. |v| < floatmin(P) (n² max|x| / 4 + n) (see _underflow_threshold) is the condition
# under which that can exceed eps |v|. A well-scaled value passes with
# one pass over x. An exact zero of x, or any entry of A, does not fire it.
@inline _quadratic_underflow_risk(f::QuadraticFunction, x, v, ::Type{P}) where {P} =
  _quadratic_underflow_risk(f, x, v, P, _stored_maxabs(x))
@inline function _quadratic_underflow_risk(f::QuadraticFunction, x, v, ::Type{P}, m) where {P}
  (_underflow_checked(P) && _underflow_checked(typeof(v))) || return false
  n = length(x)
  iszero(m) && return false
  abs(v) < _underflow_threshold(P, n * n / 4, m, n)
end

# ½ x'Ax + b'x + c summed from the exact products, each a significand and an exponent (_ExpSum), so that
# no A[i,j] x[j] is rounded into a type that can lose it. The cost is that of A x with a few
# bit operations per entry (exact zeros are skipped), and nothing is allocated.
@noinline function _quadratic_value_recover(f::QuadraticFunction, x::AbstractVector, v)
  R = typeof(v)
  _all_finite(x) || return v
  (_all_finite(f.A) && _all_finite(f.b) && isfinite(f.c)) || return v
  acc = _expsum_start(promote_type(R, Float64))
  acc = _expsum_quadratic(acc, f.A, x, -1)
  f.b === nothing || (acc = _expsum_dot(acc, f.b, x, 0))
  acc = _expsum_product(acc, f.c, one(R))
  _expsum_value(R, acc)
end

function gradient!(g::AbstractVector, f::QuadraticFunction, x::AbstractVector)
  _checkwrite(gradient!, f, g, x)
  mul!(g, f.A, x)
  f.b === nothing || (g .+= f.b)
  g
end

function _eigen!(f::QuadraticFunction{T}) where {T}
  if isempty(f.Λ)
    E = eigen(Symmetric(Matrix{T}(f.A)))
    f.Λ, f.Q = E.values, E.vectors
  end
  f
end

function prox!(y::AbstractVector, f::QuadraticFunction{T}, x::AbstractVector, γ::Real) where {T}
  _checkwrite(prox!, f, y, x)
  _eigen!(f)
  γ = convert(T, γ)
  t = f.tmp
  f.b === nothing ? copyto!(t, x) : (t .= x .- γ .* f.b)
  mul!(f.w, transpose(f.Q), t)
  f.w ./= 1 .+ γ .* f.Λ
  mul!(t, f.Q, f.w)
  copyto!(y, t)
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!)}, f::QuadraticFunction{T}, y::AbstractVector, x::AbstractVector) where {T}
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  length(x) == length(f.tmp) || throw(DimensionMismatch())
  true
end

function _checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::QuadraticFunction, g::AbstractVector, x::AbstractVector)
  _checkreal(f, x)
  _checkreal(f, g)
  _checkgradient_output(g, x)
  _checksize(g, x)
  # Preserve the dimension error of the product, which checks before writing.
  length(x) == size(f.A, 2) || mul!(g, f.A, x)
end

restrict(f::QuadraticFunction, S::AbstractVector{<:Integer}) =
  QuadraticFunction(f.A[S, S], f.b === nothing ? nothing : f.b[S], f.c)

####################################
#  Lipschitz constants of the gradients
####################################

lipschitz(::ZeroFunction) = 0.0
lipschitz(f::LinearFunction) = zero(float(eltype(f.c)))
lipschitz(f::SqrNormL2) = float(f.λ)
lipschitz(f::QuadraticFunction) = _opnorm_upper(f.A)
