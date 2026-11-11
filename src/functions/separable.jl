####################################
#
#  Separable functions f(x) = Σᵢ fᵢ(xᵢ)
#
####################################

"""
    SeparableFunction

The supertype of the separable functions `f(x) = Σᵢ fᵢ(xᵢ)`, and the way to add
one of your own. A subtype implements, for the entry with linear index `i`,

- [`prox_block`](@ref)`(f, v, i, γ)`: the proximal operator of `γfᵢ` at the
  scalar `v`;
- [`value_entry`](@ref)`(f, v, i)`: the value `fᵢ(v)`.

From these it gets the value `f(x)`, [`prox!`](@ref) and
[`value_and_prox!`](@ref) on any array, including the
three iterates. Declare [`iszeropreserving`](@ref) when `fᵢ(0) = 0` and the
proximal operator maps 0 to 0; then the value and the proximal operator on a
sparse iterate touch only its stored entries. `isseparable` and `isproximable`
are true; declare `isconvex` when it holds.

# Example

```julia
struct WeightedSquare <: SeparableFunction            # Σᵢ wᵢ xᵢ² / 2
    w::Vector{Float64}
end
ProximalBase.value_entry(f::WeightedSquare, v, i) = f.w[i] * v^2 / 2
ProximalBase.prox_block(f::WeightedSquare, v, i, γ) = v / (1 + γ * f.w[i])
ProximalBase.iszeropreserving(::WeightedSquare) = true
ProximalBase.isconvex(::Type{WeightedSquare}) = true
ProximalBase.entry_parameters(f::WeightedSquare) = (:w => f.w,)
function ProximalBase.check_parameters(f::WeightedSquare, x)
  length(f.w) == prod(size(x)) || throw(DimensionMismatch("a parameter array has length \$(length(f.w)) and the argument has \$(prod(size(x))) entries"))
  nothing
end
```

The generic kernels call `check_parameters` before writing any output or
updating a coordinate; a [`SymmetricSparseIterate`](@ref) has a `size` and no
`length`. On a `SymmetricSparseIterate`, also declare
[`ProximalBase.entry_parameters`](@ref); public oracle overrides should call
[`ProximalBase.check_symmetric_parameters`](@ref) before writing.

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
"""
abstract type SeparableFunction <: AbstractFunction end

isseparable(::Type{<:SeparableFunction}) = true
isproximable(::Type{<:SeparableFunction}) = true

# Broadcast parameters in the argument's shape to preserve linear pairing.
_reshapeparam(p, x) = p
_reshapeparam(p::AbstractArray, x) = reshape(p, size(x))
# The wrapper avoids allocating an Array header on CPU broadcasts.
_reshapeparam(p::Array, x) = Base.ReshapedArray(p, size(x), ())

"""
    value_entry(f, v, i)

For a [`SeparableFunction`](@ref) `f(x) = Σᵢ fᵢ(xᵢ)`, the value `fᵢ(v)` of the
term of entry `i` at the scalar `v`. `i` is a linear index of the array, as in
[`prox_block`](@ref).
"""
function value_entry end

(f::SeparableFunction)(x::_ArrayOrOwner) =
  (_check_value_arrays(f, x); _value_sep(f, x))
_value_unchecked(f::SeparableFunction, x::_ArrayOrOwner) = _isbuiltin(f) ? _value_sep(f, x) : f(x)
prox!(y::_ArrayOrOwner, f::SeparableFunction, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_sep!(y, f, x, γ))
_prox_unchecked!(y::_ArrayOrOwner, f::SeparableFunction, x::_ArrayOrOwner, γ::Real) =
  _isbuiltin(f) ? _prox_sep!(y, f, x, γ) : prox!(y, f, x, γ)
value_and_prox!(y::_ArrayOrOwner, f::SeparableFunction, x::_ArrayOrOwner, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(value_and_prox!, f, y, x, γ); _value_and_prox_sep!(y, f, x, γ))
_value_and_prox_unchecked!(y::_ArrayOrOwner, f::SeparableFunction, x::_ArrayOrOwner, γ::Real) =
  _isbuiltin(f) ? _value_and_prox_sep!(y, f, x, γ) : value_and_prox!(y, f, x, γ)

# The value, entry by entry over the dense array. A weighted family sums its
# ordinary products, which vectorize; a NaN sum (a NaN entry, or a 0⋅Inf
# product of a zero or infinite weight) is summed again with value_entry.
_value_entry_ordinary(f, v, i) = value_entry(f, v, i)
_has_ordinary_value(::Type) = false

@inline function _sum_entries(g, f, x::AbstractArray, ::Type{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    s += convert(T, g(f, x[i], i))
  end
  s
end

function _value_dense(f, x::AbstractArray)
  T = _valtype(x)
  _has_ordinary_value(typeof(f)) || return _sum_entries(value_entry, f, x, T)
  s = _sum_entries(_value_entry_ordinary, f, x, T)
  isnan(s) ? _sum_entries(value_entry, f, x, T) : s
end

_value_sep(f, x::AbstractArray) = (_checkparams(f, x); _value_dense(f, x))

function _value_sep(f, x::SparseIterate)
  _checkparams(f, x)
  iszeropreserving(f) || return _value_dense(f, x)
  T = _valtype(x)
  s = zero(T)
  @inbounds for k in 1:x.nnz
    s += convert(T, value_entry(f, x.nzval[k], x.nzval2ind[k]))
  end
  s
end

# an off-diagonal coordinate is the entries (r, c) and (c, r)
function _value_sep(f, x::SymmetricSparseIterate)
  _checkparams(f, x)
  iszeropreserving(f) || return _value_dense(f, matrixview(x))
  T = _valtype(x)
  p = x.p
  s = zero(T)
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(p, x.nzval2ind[k])
    v = x.nzval[k]
    s += convert(T, value_entry(f, v, (c - 1) * p + r))
    r == c || (s += convert(T, value_entry(f, v, (r - 1) * p + c)))
  end
  s
end

function _value_sep(f, x::BlockIterate)
  _checkparams(f, x)
  iszeropreserving(f) || return _value_dense(f, x)
  T = _valtype(x)
  s = zero(T)
  @inbounds for a in 1:x.nactive, i in x.groups[x.active[a]]
    s += convert(T, value_entry(f, x.storage[i], i))
  end
  s
end

# the proximal operator, entry by entry; y may be x
_prox_sep!(y::AbstractArray, f, x::AbstractArray, γ::Union{Real, AbstractArray}) = _prox_sep_dense!(y, f, x, γ)

_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::SeparableFunction, y::_ArrayOrOwner, x::_ArrayOrOwner) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); _checkparams(f, x); true)

function _prox_sep_dense!(y::AbstractArray, f, x::AbstractArray, γ)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  T = eltype(y)
  @inbounds for i in eachindex(IndexLinear(), y, x)
    y[i] = convert(T, prox_block(f, x[i], i, _entry(γ, i)))
  end
  y
end

# on sparse iterates, only the stored entries when f preserves zeros; as for a
# dense array, an entry stored in y stays stored, possibly as a zero
function _prox_sep!(y::SparseIterate, f, x::SparseIterate, γ::Union{Real, AbstractArray})
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  iszeropreserving(f) || return _prox_sep_dense!(y, f, x, γ)
  T = eltype(y)
  if y !== x
    @inbounds for k in 1:y.nnz
      y.nzval[k] = zero(T)
    end
  end
  @inbounds for k in 1:x.nnz
    i = x.nzval2ind[k]
    _setstored!(y, convert(T, prox_block(f, x.nzval[k], i, _entry(γ, i))), i)
  end
  y
end

# an owner point and a dense destination: the entries of the matrix that it represents
_prox_sep!(y::AbstractArray, f, x::SymmetricSparseIterate, γ::Real) = _prox_sep!(y, f, matrixview(x), γ)

# each coordinate (an entry of the lower triangle) once, and from a
# SymmetricSparseIterate only its stored coordinates when f preserves zeros
function _prox_sep!(y::SymmetricSparseIterate, f, x::Union{AbstractMatrix, SymmetricSparseIterate}, γ::Real)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  T = eltype(y)
  p = y.p
  if x isa SymmetricSparseIterate && iszeropreserving(f)
    if y !== x                       # as for a SparseIterate, y keeps its stored entries
      @inbounds for k in 1:y.nnz
        y.nzval[k] = zero(T)
      end
    end
    @inbounds for k in 1:x.nnz
      i = x.nzval2ind[k]
      r, c = tril_cartesian(p, i)
      j = (c - 1) * p + r
      _setstored!(y, convert(T, prox_block(f, x.nzval[k], j, _entry(γ, j))), i)
    end
    return y
  end
  @inbounds for c in 1:p, r in c:p
    j = (c - 1) * p + r
    _setstored!(y, convert(T, prox_block(f, x[r, c], j, _entry(γ, j))), tril_index(p, r, c))
  end
  y
end

# The matrix of an owner as the destination: the coordinate kernel above, which reads and writes each
# coordinate of a mirrored pair once. A traversal of both positions of a pair would read the second one after
# the first has been written when the argument is the destination.
_prox_sep!(y::_OwnerMatrix, f, x::AbstractArray, γ::Real) = (_prox_sep!(y.x, f, _coordinate_source(x), γ); y)
_coordinate_source(x::AbstractArray) = x
_coordinate_source(x::Union{_OwnerMatrix, SymmetricMatrixView}) = x.x

# the proximal operator, then the value at the result: two vectorized passes are
# several times faster than one that stores and sums, which does not vectorize;
# an indicator function is 0 at its projection
_value_and_prox_sep!(y::_ArrayOrOwner, f, x::_ArrayOrOwner, γ::Union{Real, AbstractArray}) =
  _isindicator(f) ? (_prox_sep!(y, f, x, γ); zero(_valtype(y))) : _value_sep(f, _prox_sep!(y, f, x, γ))

_restrict_entries(w::Union{Nothing, Real}, ::AbstractVector{<:Integer}) = w
_restrict_entries(w::AbstractArray, S::AbstractVector{<:Integer}) = w[S]

####################################
#  ℓ1 norm
####################################

"""
    NormL1(λ0 = 1)
    NormL1(λ0, w)

The weighted ℓ1 norm `f(x) = λ0 Σᵢ wᵢ |xᵢ|`, with all `wᵢ = 1` when `w` is
omitted. Its proximal operator is entry-wise soft thresholding
([`shrink`](@ref)). Separable, so [`prox_block`](@ref) computes any scalar
coordinate, and on a sparse iterate the value and the proximal operator touch
only the stored entries. On a [`SymmetricSparseIterate`](@ref) the weights are a
`p × p` matrix (or anything indexed by the linear indices of one), as for a
dense matrix. When symmetry is required, mirrored weights must agree at each
coordinate the operation reads. Wrap a matrix in `Symmetric(W, :L)` to use its
lower triangle.
Values follow the overflow rule in [Overflow and underflow of values](@ref).
An infinite weight fixes the entry at
zero at every level (the value is `Inf` away from zero), and a zero weight
leaves it unpenalized at every level.
An infinite level `λ0` at a positive step fixes every penalized entry (`wᵢ > 0`)
at zero, so its output is `0` also where the argument is NaN or `±Inf`
(decision 49). The value is `0` at the zero vector and stays NaN at a NaN entry.

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
"""
struct NormL1{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ0::T
  w::W

  function NormL1{T, W}(λ0::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    w === nothing || eltype(w) <: Real || _real_error(:NormL1)
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    new{T, W}(λ0, w)
  end
end
NormL1(λ0::Real = 1, w::Union{Nothing, AbstractArray} = nothing) = NormL1{typeof(λ0), typeof(w)}(λ0, w)

isconvex(::Type{<:NormL1}) = true
iszeropreserving(::NormL1) = true
_checkparams(f::NormL1, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::NormL1, v::Real, i::Integer)
  a = abs(float(v))
  w = _entry(f.w, i)
  p = _positive_product(f.λ0, w, a)
  ifelse(isinf(w) & !iszero(a), oftype(p, Inf) * a, p)
end
Base.@propagate_inbounds prox_block(f::NormL1, v::T, i::Integer, γ::Real) where {T<:Real} =
  first(_l1_prox_threshold(v, γ, f.λ0, _entry(f.w, i), float(T)))

# Whether the level and the step, as given, fix a penalized entry (w > 0) at zero whatever it holds,
# NaN and ±Inf included (E4 before E2): an infinite level at a positive step, finite or Inf, where
# the domain of the entry is {0}. Never decided from the threshold γλ0w, which overflows for finite
# parameters (R3) and is 0⋅Inf at a zero step.
@inline _l1_fixing(γ, λ) = (γ > 0) & isinf(λ)
@inline _l1_fixed(γ, λ, w) = (w > 0) & _l1_fixing(γ, λ)

# The scalar proximal operator and its threshold γλ0w, as _weight_product forms it. In the ordinary
# range of γλ0, where the step and the level are finite and positive, no entry is fixed, and the loop
# keeps the bare soft thresholding.
Base.@propagate_inbounds function _l1_prox_threshold(v, γ, λ, w, ::Type{S}) where {S}
  R = _working_type(S, γ, λ, w)
  ab = convert(R, γ) * convert(R, λ)
  if floatmin(R) <= ab <= floatmax(R)
    t = convert(S, ab * convert(R, w))
    return shrink(v, t), t
  end
  t = _weight_product(S, γ, λ, w)
  r = shrink(v, t)
  ifelse(_l1_fixed(γ, λ, w), zero(r), r), t
end

kkt_violation(f::NormL1, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]) - _kkt_coefficient(eltype(g), f.λ0, _entry(f.w, i)))
restrict(f::NormL1, S::AbstractVector{<:Integer}) = NormL1(f.λ0, _restrict_entries(f.w, S))

# without weights, Base's vectorized sum (a sum with init takes a slow sequential path)
function (f::NormL1{<:Real, Nothing})(x::DenseArray{<:AbstractFloat})
  T = float(eltype(x))
  iszero(f.λ0) && return zero(T)
  _level_fits(T, f.λ0) || return _converted_l1_value(f.λ0, x)
  s = sum(abs, x)
  # an infinite level fixes every entry at zero (E4): the value at the zero vector is 0, not Inf⋅0
  isinf(f.λ0) && iszero(s) && return zero(T)
  v = convert(T, f.λ0) * s
  isfinite(v) || !isfinite(f.λ0) || !all(isfinite, x) ? v : _value_dense(f, x)
end

# A level outside the argument's precision is applied before final conversion.
@noinline function _converted_l1_value(λ, x)
  T = float(eltype(x))
  R = promote_type(T, float(typeof(λ)))
  m = _maximum_abs(x)
  isfinite(m) || return convert(T, convert(R, λ) * sum(abs, x))
  e = _magnitude_exponent(m)
  s = _scaled_value_sum(x, _power_factors(R, -e), abs)
  convert(T, _scaled_product(convert(R, λ), s, e))
end

@inline _l1_value(::NormL1, v, a) = a * abs(v)

@inline function _l1_value_sum(f, x::Array{T}, allow_zero::Bool) where {T<:Union{Float32,Float64}}
  s = zero(T)
  ordinary = true
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    w = f.w[i]
    a = f.λ0 * w
    # Test the factors: a rounded zero may conceal recoverable underflow.
    z = allow_zero & (iszero(f.λ0) | iszero(w))
    s += _l1_value(f, x[i], a)
    ordinary &= z | (floatmin(T) <= a <= floatmax(T))
  end
  s, ordinary
end

# Preserve the original ordinary loop; retry zeros with SIMD before resorting
# to compensation. Every pass checks intermediates, not just the final sum.
function _value_dense(
    f::NormL1{T,<:Array{T}},
    x::Array{T}) where {T<:Union{Float32,Float64}}
  s, ordinary = _l1_value_sum(f, x, false)
  ordinary && return s
  s, ordinary = _l1_value_sum(f, x, true)
  # 0*Inf/NaN needs the entrywise zero-factor and domain priorities.
  ordinary && !isnan(s) ? s : invoke(_value_dense, Tuple{Any,AbstractArray}, f, x)
end

# The arrays in host memory: an Array, or a view or a reshape of one. A device array is none of these, whatever
# its supertypes are (a GPU array is a DenseArray), and keeps its own routes.
const _HostArray = Union{Array,
  SubArray{<:Any, <:Any, <:Union{Array, Base.ReshapedArray{<:Any, <:Any, <:Array}}},
  Base.ReshapedArray{<:Any, <:Any, <:Array}}
