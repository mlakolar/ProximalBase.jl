####################################
#
#  Buffers
#
####################################

# A buffer that a function object caches and sizes on demand, so that its hot
# calls allocate only when the size or the element type of the input changes.
# The field is untyped; _buffer! returns a concretely typed vector.
mutable struct Scratch
  buf::Any
end
Scratch() = Scratch(nothing)

# an array of eltype T and size dims, reused while they do not change
function _array_buffer!(s::Scratch, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
  b = s.buf
  if b isa Array{T, N}
    size(b) == dims && return b
  end
  b = Array{T, N}(undef, dims)
  s.buf = b
  b
end

# CPU prototypes keep the existing dense buffer and cache checks.
_array_buffer!(s::Scratch, ::Type{T}, x::AbstractArray) where {T} = _array_buffer!(s, T, size(x))

# Scratch is an implementation detail: copies of a function object share no buffer
Base.copy(s::Scratch) = Scratch()

####################################
#
#  Checks
#
####################################

_checksize(y, x) = size(y) == size(x) || _size_mismatch(size(y), size(x))
@noinline _size_mismatch(sy, sx) =
  throw(DimensionMismatch("the output has size $sy and the input $sx"))

_checknonneg(name, v::Real) = v >= 0 || throw(ArgumentError("$name must be nonnegative, got $v"))
_checknonneg(name, v::AbstractArray) = all(>=(0), v) || throw(ArgumentError("$name must be nonnegative"))
_checknonneg(name, ::Nothing) = true

# A weight or a bound: a scalar, or the entry i of an array. The bounds check
# propagates: a direct call checks it, and the loops that first check the
# lengths of the parameter arrays (_checkparam) elide it with @inbounds.
@inline _entry(w::Real, ::Integer) = w
Base.@propagate_inbounds _entry(w::AbstractArray, i::Integer) = w[i]
@inline _entry(::Nothing, ::Integer) = true

# a parameter array has an entry for every entry of x
_checkparam(::Union{Nothing, Real}, x) = true
_checkparam(w::AbstractArray, x) = length(w) == length(x) || _parameter_length_mismatch(length(w), length(x))
@noinline _parameter_length_mismatch(nw, nx) =
  throw(DimensionMismatch("a parameter array has length $nw and the argument has length $nx"))

# The type of a value at x: the real floating type of its entries, promoted
# over the blocks of a tuple. A value computed in another type is rounded to it.
_valtype(x) = float(eltype(x))
# Bodies of generated functions that visit tuple blocks in order. Recursion with Base.tail
# is not inferred beyond 32 blocks, and Base's map stops unrolling there.
_inline(ex) = Expr(:block, Expr(:meta, :inline), ex)
_blockwise(ex, n, result) = Expr(:block, Expr(:meta, :inline), (ex(k) for k in 1:n)..., result)
_blockany(ex, n) = _inline(foldl((a, k) -> :($a || $(ex(k))), 1:n; init = false))
_blocktuple(ex, n) = _inline(Expr(:tuple, (ex(k) for k in 1:n)...))
@generated function _valtype(x::Tuple)
  n = fieldcount(x)
  n == 0 && return _inline(:(Union{}))
  n == 1 && return _inline(:(promote_type(_valtype(x[1]))))
  types = [:(_valtype(x[$k])) for k in 1:n]
  combine(a, b) = :(promote_type($a, $b))
  _inline(VERSION < v"1.11" ? foldr(combine, types) : foldl(combine, types))
end
_value_at(v, x) = convert(_valtype(x), v)
# Arguments, outputs and parameters are real; complex entries are rejected.
_checkreal(f, x) = eltype(x) <: Real || _real_error(nameof(typeof(f)))
@generated _checkreal(f, x::Tuple) = _blockwise(k -> :(_checkreal(f, x[$k])), fieldcount(x), true)
@noinline _real_error(name) = throw(ArgumentError("$name requires real entries"))

# A buffer of zeros of type T and size dims, next to the array a: an Array,
# and a device array of the type of a GPU array a (see the GPU extension).
_zeros_like(a, ::Type{T}, dims) where {T} = zeros(T, dims)

# The stored values suffice for maxima; symmetric multiplicities do not matter.
_stored_values(x::AbstractArray) = x
_stored_values(x::SparseArrays.AbstractSparseArray) = nonzeros(x)
function _maximum_abs(x::AbstractArray)
  m = zero(float(eltype(x)))
  @inbounds for v in _stored_values(x)
    m = max(m, abs(float(v)))
  end
  m
end

# The signed integer type of the same width, by dispatch (a conditional on the type is not folded by Julia 1.10)
_int_type(::Type{Float32}) = Int32
_int_type(::Type{Float64}) = Int64

# For Float32 and Float64 the maximum is taken on the bit patterns: with the sign cleared they are ordered like the
# magnitudes (as signed integers, whose maximum vectorizes on more targets than the unsigned one), an integer maximum vectorizes
# where the NaN-aware float maximum is a scalar chain (the trigger of #520 reads this on every value call),
# and a NaN or an Inf gives a NaN or an Inf as before.
function _maximum_abs(x::AbstractArray{T}) where {T<:Union{Float32, Float64}}
  U = _int_type(T)
  vals = _stored_values(x)
  m = zero(U)
  @inbounds @simd for i in eachindex(vals)
    m = max(m, reinterpret(U, vals[i]) & typemax(U))
  end
  reinterpret(T, m)
end
_maximum_abs(x::Diagonal{T}) where {T<:Union{Float32, Float64}} = _maximum_abs(x.diag)

# Whether every stored value is a zero of either sign. For Float32 and Float64, by an OR of the bit patterns with the sign
# cleared at the end: an OR vectorizes on every target (an integer maximum of 64-bit lanes is a scalar chain without
# AVX-512), and a NaN or an Inf is not zero.
_all_zero(x) = all(iszero, _stored_values(x))
# whether vals[lo:hi] are all zeros of either sign, by that OR (U, the integer type of the width of the entries)
@inline function _zero_bits(vals, lo::Int, hi::Int, ::Type{U}) where {U}
  m = zero(U)
  @inbounds @simd for i in lo:hi
    m |= reinterpret(U, vals[i])
  end
  iszero(m & typemax(U))
end
function _all_zero(x::AbstractArray{T}) where {T<:Union{Float32, Float64}}
  vals = _stored_values(x)
  _zero_bits(vals, firstindex(vals), lastindex(vals), _int_type(T))
end
# The same, in blocks of _ALL_ZERO_BLOCK stored values, returning false at the first block that holds a nonzero value
# (NaN and ±Inf included): an array with a nonzero value in its first block costs one block of reads.
const _ALL_ZERO_BLOCK = 512
_all_zero_early(x) = _all_zero(x)
function _all_zero_early(x::AbstractArray{T}) where {T<:Union{Float32, Float64}}
  vals = _stored_values(x)
  n = lastindex(vals)
  lo = firstindex(vals)
  while lo <= n
    hi = min(lo + (_ALL_ZERO_BLOCK - 1), n)
    _zero_bits(vals, lo, hi, _int_type(T)) || return false
    lo = hi + 1
  end
  true
end

# The largest magnitude, and finiteness, over the stored entries only.
const _WrappedSparse = Union{Adjoint{<:Any, <:SparseArrays.AbstractSparseArray},
                             Transpose{<:Any, <:SparseArrays.AbstractSparseArray}}
_stored_maxabs(x) = _maximum_abs(x)
_stored_maxabs(x::_WrappedSparse) = _maximum_abs(parent(x))
_all_finite(x) = all(isfinite, _stored_values(x))
_all_finite(x::_WrappedSparse) = _all_finite(parent(x))
_all_finite(::Nothing) = true

####################################
#
#  The scaled-value protocol (R1, "Wrappers")
#
####################################

# The scaled-value mechanism that `Scaled` uses when its value leaves the range (calculus.jl,
# `_scaled_parts`). A function that implements it declares `_has_value_parts` and returns from
# `_value_parts(f, x, v)`, given its ordinary value v (infinite, zero or subnormal), either
# `nothing`, to keep the ordinary result, or (m, e): its value m 2^e, with m a finite float,
# zero or in [1/2, 1) in magnitude, to the accuracy of its ordinary computation.
_has_value_parts(::Type) = false
_value_parts(f, x, v) = nothing
# Exact-zero protection. When the ordinary value is exactly zero, `_zero_value_exponent(f, x)` gives
# an exponent E with |f(x)| < 2^E for the exact value, from the parameters and the size of x (no pass
# over x), `_EXACT_ZERO` when that zero is exact, or `nothing` when no such bound is known (the value
# is then recomputed). A zero value that is not exact comes from underflow: a rounding into the
# subnormal range loses less than δ/2, δ the smallest subnormal of the computation type, and a later
# product with a bounded factor (a level, a weight, a logarithm) multiplies that loss. A zero that
# comes from cancellation is within the accuracy of the ordinary computation (R1 measures it against
# the sum of the magnitudes of the terms), which the recomputation does not improve on either.
_zero_value_exponent(f, x) = nothing
# Is a zero or subnormal value of f exact for every argument, by its type? Then a factor above one
# has nothing to recover from it, and `Scaled` does not try (the test folds away at compile time).
_zero_value_exact(f) = false
# the marker of an exact zero, outside every exponent range (no Int exponent can equal it)
struct _ExactZero end
const _EXACT_ZERO = _ExactZero()
# Is an infinite ordinary value known to be the value, from the parameters and the size of x only?
_infinite_value_exact(f, x) = false
# Can the gradient output `out`, already written, share memory with a parameter that the
# parts read? Decided from the identities of the arrays (O(1)).
_parameter_alias(f, out) = false
_mightalias_parameter(out, p) = false
_mightalias_parameter(out::AbstractArray, p::AbstractArray) = Base.mightalias(out, p)

####################################
#
#  Lower-triangular indexing
#
####################################

"""
    tril_index(p, r, c)

The linear index of entry `(r, c)`, `r ≥ c`, of the lower triangle of a `p × p`
matrix, numbered column by column: the coordinate of a
`SymmetricSparseIterate` that holds the entry. Inverse of
`tril_cartesian`.
"""
tril_index(p::Integer, r::Integer, c::Integer) = p * (c - 1) - div(c * (c - 1), 2) + r

"""
    tril_cartesian(p, k)

The entry `(r, c)`, `r ≥ c`, of the lower triangle of a `p × p` matrix with
linear index `k`, numbered column by column. Inverse of `tril_index`.
"""
@inline function tril_cartesian(p::Integer, k::Integer)
  rv = div(p * (p + 1), 2) - k
  # j = ⌊(√(1 + 8rv) - 1)/2⌋. For p < 2²⁴, 1 + 8rv < 2⁵⁰ and the floating square
  # root is exact, and three times faster than isqrt; the larger p, beyond any
  # p × p matrix in memory, take integer arithmetic, exact for any p, out of
  # line so that this function stays small enough to inline in inner loops
  j = p < 16_777_216 ? trunc(typeof(rv), (sqrt(1 + 8rv) - 1) / 2) : _tril_column_isqrt(rv)
  i = rv - div(j * (j + 1), 2)
  (p - i, p - j)
end

@noinline _tril_column_isqrt(rv::Integer) = div(isqrt(1 + 8rv) - 1, 2)

####################################
#
#  Types and devices
#
####################################

"""
    _model_precision(D, types...)

Promote real floating entry types with the delivered type `D`. Fixed-width
integers and `Bool` give `Float64`; `BigInt` gives `BigFloat`.

For `value_and_gradient!`, the value passes `R`, the argument and data types;
the gradient passes `D`, the argument and data types. The value of
`value_and_prox!` passes the output and data types, not the argument type.
Callers promote tuple blocks. The SqrNormL2 gradient and proximal operator use it.
"""
function _model_precision(::Type{D}, types::Type...) where {D}
  all(T -> T <: Real, (D, types...)) || _real_error(:_model_precision)
  promote_type(map(T -> real(float(T)), (D, types...))...)
end

# CPU fallback retained from frozen lapack.jl.
_is_device_array(::AbstractArray) = false
