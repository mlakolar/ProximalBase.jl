####################################
#
#  Separable functions f(x) = Σᵢ fᵢ(xᵢ)
#
####################################

"""
    SeparableFunction

The supertype of the separable functions `f(x) = Σᵢ fᵢ(xᵢ)`, and the way to add
one of your own. A subtype implements, for the entry with linear index `i`,

- `prox_block(f, v, i, γ)`: the proximal operator of `γfᵢ` at the
  scalar `v`;
- `value_entry(f, v, i)`: the value `fᵢ(v)`.

From these it gets the value `f(x)`, `prox!` and
`value_and_prox!` on any array, including the
three iterates. Declare `iszeropreserving` when `fᵢ(0) = 0` and the
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
updating a coordinate; a `SymmetricSparseIterate` has a `size` and no
`length`. On a `SymmetricSparseIterate`, also declare
`ProximalBase.entry_parameters`; public oracle overrides should call
`ProximalBase.check_symmetric_parameters` before writing.

See Symmetric parameters for which arrays must be symmetric and which calls check them.
"""
abstract type SeparableFunction <: AbstractFunction end

isseparable(::Type{<:SeparableFunction}) = true
isproximable(::Type{<:SeparableFunction}) = true

"""
    value_entry(f, v, i)

For a `SeparableFunction` `f(x) = Σᵢ fᵢ(xᵢ)`, the value `fᵢ(v)` of the
term of entry `i` at the scalar `v`. `i` is a linear index of the array, as in
`prox_block`.
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

# the proximal operator, then the value at the result: two vectorized passes are
# several times faster than one that stores and sums, which does not vectorize;
# an indicator function is 0 at its projection
_value_and_prox_sep!(y::_ArrayOrOwner, f, x::_ArrayOrOwner, γ::Union{Real, AbstractArray}) =
  _isindicator(f) ? (_prox_sep!(y, f, x, γ); zero(_valtype(y))) : _value_sep(f, _prox_sep!(y, f, x, γ))

_restrict_entries(w::Union{Nothing, Real}, ::AbstractVector{<:Integer}) = w
_restrict_entries(w::AbstractArray, S::AbstractVector{<:Integer}) = w[S]

# The arrays in host memory: an Array, or a view or a reshape of one. A device array is none of these, whatever
# its supertypes are (a GPU array is a DenseArray), and keeps its own routes.
const _HostArray = Union{Array,
  SubArray{<:Any, <:Any, <:Union{Array, Base.ReshapedArray{<:Any, <:Any, <:Array}}},
  Base.ReshapedArray{<:Any, <:Any, <:Array}}
