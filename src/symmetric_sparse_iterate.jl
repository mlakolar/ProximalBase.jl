####################################
#
#  Symmetric Sparse Iterate
#
####################################

SymmetricSparseIterate(n::Int) = SymmetricSparseIterate(Float64, n)
function SymmetricSparseIterate(::Type{T}, n::Int) where {T}
  n >= 0 || throw(ArgumentError("the dimension must be nonnegative, got $n"))
  _t = div(n*(n+1), 2)
  SymmetricSparseIterate{T}(zeros(T, _t), zeros(Int, _t), zeros(Int, _t), 0, n)
end

Base.eltype(::Type{SymmetricSparseIterate{T}}) where {T} = T
Base.eltype(::SymmetricSparseIterate{T}) where {T} = T
Base.size(x::SymmetricSparseIterate) = (x.p, x.p)
# the shape of a matrix: later dimensions have size one, as for any matrix
Base.size(x::SymmetricSparseIterate, d::Integer) = d <= 2 ? size(x)[d] : 1

# the coordinate of entry (r, c)
@inline _coordinate(x::SymmetricSparseIterate, r::Integer, c::Integer) =
  r >= c ? tril_index(x.p, r, c) : tril_index(x.p, c, r)

Base.@propagate_inbounds function Base.getindex(x::SymmetricSparseIterate, r::Int, c::Int)
  @boundscheck _checkpair(x, r, c)
  @inbounds _getstored(x, _coordinate(x, r, c))
end
Base.@propagate_inbounds function Base.setindex!(x::SymmetricSparseIterate, v, r::Int, c::Int)
  @boundscheck _checkpair(x, r, c)
  @inbounds _setstored!(x, v, _coordinate(x, r, c))
end
# one unsigned comparison per index, combined without a branch between them
@inline _checkpair(x::SymmetricSparseIterate, r::Int, c::Int) =
  (((r - 1) % UInt < x.p % UInt) & ((c - 1) % UInt < x.p % UInt)) || _throw_pair(x, r, c)
@noinline _throw_pair(x::SymmetricSparseIterate, r, c) = throw(BoundsError(x, (r, c)))

# Other integer types: the bounds are checked in the type of the index, before it is narrowed to
# Int, so that an out-of-range index of any integer type is a BoundsError.
@inline function _int_pair(x::SymmetricSparseIterate, r::Integer, c::Integer)
  (1 <= r <= x.p && 1 <= c <= x.p) || _throw_pair(x, r, c)
  (Int(r), Int(c))
end
Base.@propagate_inbounds function Base.getindex(x::SymmetricSparseIterate, r::Integer, c::Integer)
  i, j = _int_pair(x, r, c)
  @inbounds x[i, j]
end
Base.@propagate_inbounds function Base.setindex!(x::SymmetricSparseIterate, v, r::Integer, c::Integer)
  i, j = _int_pair(x, r, c)
  @inbounds setindex!(x, v, i, j)
end

# x[k] has two natural readings, the coordinate and the matrix linear index
@noinline _throw_linear_index(x::SymmetricSparseIterate, k) = throw(ArgumentError(
  "a SymmetricSparseIterate has no one-index access: x[$k] could mean a coordinate or a matrix " *
  "linear index. Use x[i, j] for a matrix entry, coordinates(x)[$k] for variable $k of the lower " *
  "triangle, or matrixview(x)[$k] for entry $k of the matrix in column-major order"))
Base.getindex(x::SymmetricSparseIterate, k::Integer) = _throw_linear_index(x, k)
Base.setindex!(x::SymmetricSparseIterate, v, k::Integer) = _throw_linear_index(x, k)

# Base derives axes(x) from size(x) for any object that has a size. The symmetric iterate has no axes,
# so that eachindex, broadcasting and checks of position lists are refused at once; its
# one-index access, length and iteration have no methods.
Base.axes(x::SymmetricSparseIterate) = throw(ArgumentError(
  "a SymmetricSparseIterate has no axes: use size(x), coordinates(x) for its variables, " *
  "or axes(matrixview(x)) for its matrix"))

# the coordinates of x, the entries of its lower triangle, as a vector
struct SymmetricCoordinates{T} <: AbstractVector{T}
  x::SymmetricSparseIterate{T}
end
# aliasing hooks from Base (non-public functions); the symmetric iterate reports its storage
# arrays, so that activevalues(x) and the views of x are all seen to overlap x
Base.dataids(x::SymmetricSparseIterate) =
  (Base.dataids(x.nzval)..., Base.dataids(x.nzval2ind)..., Base.dataids(x.ind2nzval)...)
Base.dataids(c::SymmetricCoordinates) = Base.dataids(c.x)
Base.unaliascopy(c::SymmetricCoordinates) = coordinates(copy(c.x))

Base.size(c::SymmetricCoordinates) = (length(c.x.ind2nzval),)
Base.IndexStyle(::Type{<:SymmetricCoordinates}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(c::SymmetricCoordinates, k::Int) = _getstored(c.x, k)
Base.@propagate_inbounds Base.setindex!(c::SymmetricCoordinates, v, k::Int) = (_setstored!(c.x, v, k); c)

coordinates(x::SymmetricSparseIterate) = SymmetricCoordinates(x)

# fill! of the vector of coordinates is a value write like any other (CRD-3): the value is
# converted first, and a zero reaches only the active coordinates, in O(nactive), so that
# fill!(coordinates(x), 0) followed by dropzeros!(x) empties x without a pass over all q
function Base.fill!(c::SymmetricCoordinates{T}, v) where {T}
  x = c.x
  s = convert(T, v)
  if iszero(s)
    @inbounds for k in 1:x.nnz
      x.nzval[k] = s
    end
  else
    for k in 1:length(x.ind2nzval)
      _setstored!(x, s, k)
    end
  end
  c
end

# a read-only p × p view of x: the matrix that x represents, both triangles
struct SymmetricMatrixView{T} <: AbstractMatrix{T}
  x::SymmetricSparseIterate{T}
end

"""
    matrixview(x::SymmetricSparseIterate)

A live, read-only `p × p` matrix view of `x`: ordinary matrix indexing, both
triangles, no copy of the data. Later writes to `x` show through it. Nothing can
be written through the view or through a wrapper derived from it (`view`,
`reshape`, `transpose`, `Symmetric`): the write is refused and `x` is
unchanged. A generic allocating operation on the view (`A .+ 1`, `similar(A)`,
`collect(A)`) returns an ordinary writable dense array. `mul!` of the view and a
dense vector costs `O(p + nactive(x))`; other generic matrix operations on the
view read all `p²` entries. Use `Matrix(x)` for an independent dense copy.

!!! note "Non-finite vectors"
    A coefficient of `x` that is zero contributes nothing to the product, whether
    it is absent or an explicitly stored zero: a non-finite entry of `v` that meets
    only zero coefficients does not reach the result, and `mul!(y, matrixview(x), v)`
    is the same before and after `dropzeros!(x)`, up to the order of the finite
    summation. A NaN coefficient is not zero and gives NaN.

Entries `(r, c)` and `(c, r)` are one variable of `x`; a coordinate of `x` is
reached through `coordinates`, not through this view, so
`matrixview(x)[4]` is a matrix linear index.
"""
matrixview(x::SymmetricSparseIterate) = SymmetricMatrixView(x)

Base.size(A::SymmetricMatrixView) = size(A.x)
Base.IndexStyle(::Type{<:SymmetricMatrixView}) = IndexCartesian()
Base.@propagate_inbounds Base.getindex(A::SymmetricMatrixView, r::Int, c::Int) = A.x[r, c]

# aliasing hooks from Base (non-public functions)
Base.dataids(A::SymmetricMatrixView) = Base.dataids(A.x)
Base.unaliascopy(A::SymmetricMatrixView) = SymmetricMatrixView(copy(A.x))
# A Symmetric or Hermitian wrapper of the view is copied through its parent, as a wrapper of the same
# type and the same triangle: copy of the wrapper has a dense parent, which Base.unalias refuses on
# Julia 1.10, where the copy must have the type of the original
Base.unaliascopy(A::Symmetric{<:Any, <:SymmetricMatrixView}) = Symmetric(Base.unaliascopy(A.data), Symbol(A.uplo))
Base.unaliascopy(A::Hermitian{<:Any, <:SymmetricMatrixView}) = Hermitian(Base.unaliascopy(A.data), Symbol(A.uplo))
# Base.unalias(x, A) of the symmetric iterate x copies a source that shares memory with it, keeping the type of the
# source. Base does that through unaliascopy of each wrapper, and on Julia 1.10 it has no such copy for a
# Symmetric or Hermitian wrapper (a copy of the wrapper has a dense parent), so a wrapper of a view or a
# reshape of x, or a view of such a wrapper, was refused. The copy here goes through every derived parent,
# the wrapper types of Base and LinearAlgebra that the views of a symmetric iterate can be in, and keeps the type, the
# triangle and the independence; any other source takes the copy of Base. The method is for the symmetric iterate as the
# destination, so that it adds nothing to a function of Base for Base's types.
function Base.unalias(dest::SymmetricSparseIterate, A::AbstractArray)
  Base.mightalias(dest, A) ? _unaliascopy(A) : A
end
Base.unalias(::SymmetricSparseIterate, A::AbstractRange) = A
_unaliascopy(A) = Base.unaliascopy(A)
_unaliascopy(A::Symmetric) = Symmetric(_unaliascopy(A.data), Symbol(A.uplo))
_unaliascopy(A::Hermitian) = Hermitian(_unaliascopy(A.data), Symbol(A.uplo))
_unaliascopy(A::SubArray) =
  typeof(A)(_unaliascopy(A.parent), map(Base.unaliascopy, A.indices), A.offset1, A.stride1)
_unaliascopy(A::Base.ReshapedArray) = typeof(A)(_unaliascopy(A.parent), A.dims, A.mi)
_unaliascopy(A::Transpose) = Transpose(_unaliascopy(A.parent))
_unaliascopy(A::Adjoint) = Adjoint(_unaliascopy(A.parent))
Base.transpose(A::SymmetricMatrixView) = A
Base.adjoint(A::SymmetricMatrixView{<:Real}) = A
Base.Matrix(A::SymmetricMatrixView) = Matrix(A.x)
Base.Array(A::SymmetricMatrixView) = Matrix(A.x)

# the stored-entry product, O(p + nactive), not the generic p² scan
LinearAlgebra.mul!(C::AbstractVector, A::SymmetricMatrixView, B::AbstractVector) = mul!(C, A.x, B)
LinearAlgebra.mul!(C::AbstractVector, A::SymmetricMatrixView, B::AbstractVector, α::Number, β::Number) =
  mul!(C, A.x, B, α, β)

function Base.show(io::IO, ::MIME"text/plain", A::SymmetricMatrixView)
  print(io, "matrixview of a ")
  show(io, MIME("text/plain"), A.x)
end

# A matrix written into x must have the axes of x and be exactly symmetric; the
# checks run before anything is written. Symmetric(A, :L) keeps one triangle of
# a matrix that is not symmetric.

# NaN matches NaN and 0.0 matches -0.0
_symequal(a, b) = isequal(a, b) || (a == b) === true

# sources whose entries (r, c) and (c, r) are equal by construction
_symmetric_by_type(A) = false
_symmetric_by_type(::Union{SymmetricSparseIterate, SymmetricMatrixView, Symmetric{<:Number},
                           Hermitian{<:Real}, Diagonal{<:Number}, SymTridiagonal{<:Number}}) = true
_symmetric_by_type(A::Union{Adjoint, Transpose}) = _symmetric_by_type(parent(A))

# The symmetric part (A + A') / 2, read lazily. The proximal point over symmetric
# matrices at a matrix V that is not symmetric is the proximal point at its symmetric
# part, for every function, so a SymmetricSparseIterate output reads its argument
# through this wrapper. An entry pair that is not equal is averaged without overflow or
# premature underflow, and exactly symmetrically; an opposite-infinity pair gives NaN.
struct _SymmetricPart{T, A<:AbstractMatrix} <: AbstractMatrix{T}
  A::A
end
_SymmetricPart(A::AbstractMatrix) = _SymmetricPart{typeof(zero(eltype(A)) / 2), typeof(A)}(A)
Base.size(S::_SymmetricPart) = size(S.A)
Base.axes(S::_SymmetricPart) = axes(S.A)
Base.IndexStyle(::Type{<:_SymmetricPart}) = IndexCartesian()
Base.dataids(S::_SymmetricPart) = Base.dataids(S.A)
Base.@propagate_inbounds function Base.getindex(S::_SymmetricPart{T}, r::Int, c::Int) where {T}
  a, b = convert(T, S.A[r, c]), convert(T, S.A[c, r])
  _symequal(a, b) ? a : _symmetric_mean(a, b)
end
# The correctly rounded (a + b) / 2 when the sum is finite, which keeps subnormal digits;
# the halves are taken first only when the sum overflows or a value is not finite, where
# they cannot underflow. Addition commutes, so the result is symmetric in (a, b).
@inline function _symmetric_mean(a::T, b::T) where {T}
  s = a + b
  isfinite(s) ? s / 2 : a / 2 + b / 2
end
_symmetric_by_type(::_SymmetricPart) = true
Base.adjoint(S::_SymmetricPart{<:Real}) = S
Base.transpose(S::_SymmetricPart) = S
# the products of the decompositions keep the speed of BLAS: ((A + A') / 2) B = (A B + A' B) / 2
# (strided output and factor, so that these methods meet neither those of a SymmetricSparseIterate
# output nor the structured-matrix methods of LinearAlgebra, which differ between Julia versions)
for (CT, BT) in ((:StridedMatrix, :StridedVecOrMat), (:StridedVector, :StridedVector))
  @eval begin
    function LinearAlgebra.mul!(C::$CT, S::_SymmetricPart{<:Real, <:StridedMatrix}, B::$BT, α::Number, β::Number)
      h = α / 2
      # halving α would lose the digits that the result keeps: scale the product instead
      !iszero(α) && abs(h) < floatmin(typeof(h)) && return _scaled_symmetric_mul!(C, S, B, α, β)
      mul!(C, S.A, B, h, β)
      mul!(C, S.A', B, h, true)
    end
    LinearAlgebra.mul!(C::$CT, S::_SymmetricPart{<:Real, <:StridedMatrix}, B::$BT) = mul!(C, S, B, true, false)
  end
end

# C = α (A + A')/2 B + β C for a tiny α, term by term: each product α s b is formed from
# significands and one exponent, so that it is rounded once whatever the range of the
# three factors (a global scaling of B cannot serve entries of very different sizes),
# and the finite terms are then added. Slow, and used only when halving α would lose digits.
function _scaled_symmetric_mul!(C, S::_SymmetricPart, B, α, β)
  A = S.A
  T = float(promote_type(eltype(A), eltype(B)))
  a = T(α)
  Cm = C isa AbstractVector ? reshape(C, :, 1) : C
  Bm = B isa AbstractVector ? reshape(B, :, 1) : B
  size(A, 1) == size(Cm, 1) && size(A, 2) == size(Bm, 1) && size(Cm, 2) == size(Bm, 2) ||
    throw(DimensionMismatch("the product has shapes $(size(A)) * $(size(Bm)) into $(size(Cm))"))
  @inbounds for k in axes(Bm, 2), i in axes(A, 1)
    acc = zero(T)
    for j in axes(A, 2)
      acc += _scaled_product(a, T(S[i, j]), T(Bm[j, k]), 0)
    end
    Cm[i, k] = iszero(β) ? acc : acc + β * Cm[i, k]
  end
  C
end

# The argument that a proximal map into y reads: x itself when y is not a
# SymmetricSparseIterate, x is symmetric by type, or x is exactly symmetric (found by
# a scan that stops at the first pair that differs, so an exactly symmetric matrix
# keeps its fast paths and its results), otherwise the symmetric part of x. A size
# that does not match y is left to the shape checks, which run before any entry is read.
_symmetric_argument(y, x) = x
function _symmetric_argument(y::SymmetricSparseIterate, x::AbstractMatrix)
  _symmetric_by_type(x) && return x
  # a device matrix is read on the host, as the decompositions do, so the rule holds for it too
  _is_device_array(x) && return _symmetric_argument(y, Array(x))
  _exactly_symmetric(x) ? x : _SymmetricPart(x)
end
_exactly_symmetric(x::AbstractMatrix) = size(x, 1) == size(x, 2) && _issymmetric_exact(x, _symequal)
# A tuple with no symmetric block, at any depth, is returned as it is; otherwise the
# blocks are unrolled by type, nested tuples included.
@inline _symmetric_argument(y::Tuple, x::Tuple) = _symmetric_blocks(y, x)
_has_symmetric_type(::Type{<:SymmetricSparseIterate}) = true
_has_symmetric_type(::Type{T}) where {T<:Tuple} = any(_has_symmetric_type, fieldtypes(T))
_has_symmetric_type(::Type) = false
@generated function _symmetric_blocks(y::Tuple, x::Tuple)
  (fieldcount(y) == fieldcount(x) && _has_symmetric_type(y)) || return :x
  Expr(:tuple, (:(_symmetric_argument(y[$k], x[$k])) for k in 1:fieldcount(y))...)
end

# the first pair (r, c), r > c, in column-major order, at which the square,
# one-based A is not symmetric
function _asymmetric_pair(A::AbstractMatrix, eq::E) where {E}
  p = size(A, 1)
  @inbounds for c in 1:p, r in c+1:p
    eq(A[r, c], A[c, r]) || return (r, c)
  end
  nothing
end
_issymmetric_exact(A::AbstractMatrix, eq::E) where {E} = _asymmetric_pair(A, eq) === nothing
# in O(nnz): each stored entry against its transpose
function _issymmetric_exact(A::SparseMatrixCSC, eq::E) where {E}
  rows, vals = rowvals(A), nonzeros(A)
  @inbounds for c in axes(A, 2), k in nzrange(A, c)
    r = rows[k]
    r == c || eq(vals[k], A[c, r]) || return false
  end
  true
end

_check_symmetric(A::AbstractMatrix, subject = "the value written into a SymmetricSparseIterate", name = "A") =
  _symmetric_by_type(A) || _issymmetric_exact(A, _symequal) || _throw_asymmetric(A, subject, name)

# The one message for a matrix that must be exactly symmetric and is not: it names
# the pair with the largest difference and the relative asymmetry
# max |A[r, c] - A[c, r]| / max |A[i, j]|. The tolerance √eps only chooses the
# wording; it never decides whether A is accepted. Differences and scales are
# computed on halves, so that neither a wrapped integer difference nor a
# difference of two large floats overflows. Entries without arithmetic (labels)
# are reported without a relative asymmetry.
# integers and rationals stay exact until the ratio is formed: adjacent integers
# above 2^53 have the same Float64
_exact(a) = a isa Union{Integer, Rational} ? big(a) : a
_half(a) = Float64(abs(_exact(a) / 2))
_halfdiff(a, b) = (d = Float64(abs(_exact(a) / 2 - _exact(b) / 2)); isnan(d) ? Inf : d)
_rounding_type(::Type{T}) where {T} =
  T <: Number && float(real(T)) <: AbstractFloat ? float(real(T)) : Float64

# the pairs (i, j), i > j, that can differ: all of them, or for a sparse matrix the
# stored entries, whose mirrors suffice
function _foreach_pair(f::F, A::AbstractMatrix) where {F}
  for j in axes(A, 2), i in j+1:size(A, 1)
    f(i, j)
  end
end
function _foreach_pair(f::F, A::SparseMatrixCSC) where {F}
  rows = rowvals(A)
  for c in axes(A, 2), k in nzrange(A, c)
    r = rows[k]
    r == c || f(max(r, c), min(r, c))
  end
end
function _max_half(A)
  s = 0.0
  for v in A
    v isa Number && isfinite(v) && (s = max(s, _half(v)))
  end
  s
end
_max_half(A::SparseMatrixCSC) = _max_half(nonzeros(A))

@noinline function _throw_asymmetric(A::AbstractMatrix, subject, name, where = "")
  best, r, c = -1.0, 0, 0
  labels = false
  _foreach_pair(A) do i, j
    x, y = A[i, j], A[j, i]
    _symequal(x, y) && return
    d = if x isa Number && y isa Number
      _halfdiff(x, y)
    else
      labels = true
      Inf
    end
    d > best && ((best, r, c) = (d, i, j))
  end
  r == 0 && throw(ArgumentError("$subject must be symmetric$where"))
  relative = labels ? nothing : best / _max_half(A)
  throw(ArgumentError(_asymmetry_message(subject, name, where, r, c, A[r, c], A[c, r], relative,
                                         _rounding_type(eltype(A)))))
end

# A check of one pair, such as a coordinate update, reads only that
# pair: its relative asymmetry is relative to the larger of the two entries.
@noinline function _throw_asymmetric_pair(subject, name, where, r, c, x, y)
  relative = if x isa Number && y isa Number
    scale = max(isfinite(x) ? _half(x) : 0.0, isfinite(y) ? _half(y) : 0.0)
    _halfdiff(x, y) / scale
  end
  throw(ArgumentError(_asymmetry_message(subject, name, where, r, c, x, y, relative,
                                         _rounding_type(typeof(x)))))
end

function _asymmetry_message(subject, name, where, r, c, x, y, relative, T)
  head = "$subject must be symmetric$where, but entry ($r, $c) is $x and entry ($c, $r) is $y"
  relative === nothing && return head * "; pass Symmetric($name, :L) to keep the lower triangle of $name"
  shown = isfinite(relative) ? round(relative; sigdigits = 2) : relative
  relative < sqrt(eps(T)) ?
    "$head (relative asymmetry $shown, the size of rounding error); if $name is symmetric in exact arithmetic, pass Symmetric($name, :L), or ($name + $name')/2" :
    "$head (relative asymmetry $shown); pass Symmetric($name, :L) to keep the lower triangle of $name, or ($name + $name')/2 for its symmetric part"
end

# a parameter array a of f, a matrix or the vector of its p² entries, read as the
# p × p matrix that the pair checks index; r = 0 checks every pair
@noinline function _throw_asymmetric_parameter(f, name, a, p, r, c)
  subject = f === nothing ? "the steps γ" : "the array $name of $(nameof(typeof(f)))"
  label = f === nothing ? "γ" : string(name)
  a isa AbstractMatrix && size(a) == (p, p) || (label = "reshape($label, $p, $p)")
  where = " on a SymmetricSparseIterate"
  r == 0 && _throw_asymmetric(reshape(a, p, p), subject, label, where)
  _throw_asymmetric_pair(subject, label, where, r, c, a[(c - 1) * p + r], a[(r - 1) * p + c])
end

# the axes of the matrix that x represents
_symmetric_matrix_axes(x::SymmetricSparseIterate) = (Base.OneTo(x.p), Base.OneTo(x.p))
_check_axes(x::SymmetricSparseIterate, A::AbstractArray) = axes(A) == _symmetric_matrix_axes(x) || _throw_axes(x, A)
@noinline _throw_axes(x, A) =
  throw(DimensionMismatch("source axes $(axes(A)) do not match destination axes $(_symmetric_matrix_axes(x))"))

Base.convert(::Type{SymmetricSparseIterate{T}}, ::Union{AbstractArray, SymmetricSparseIterate}) where {T<:Complex} =
  _real_error(:SymmetricSparseIterate)

# X must be symmetric
function SymmetricSparseIterate(X::SparseMatrixCSC{T}) where {T}
  T <: Real || _real_error(:SymmetricSparseIterate)
  n, m = size(X)
  n == m || throw(ArgumentError("X needs to be square matrix"))
  _check_symmetric(X)
  out = SymmetricSparseIterate(T, n)

  I, J, V = findnz(X)
  @inbounds for i=1:length(I)
    if I[i] >= J[i]
      out[I[i], J[i]] = V[i]
    end
  end

  out
end

# a symmetric matrix, or one triangle of a matrix as Symmetric(X, :L)
function SymmetricSparseIterate(X::AbstractMatrix{T}) where {T}
  T <: Real || _real_error(:SymmetricSparseIterate)
  p = checksquare(X)
  out = SymmetricSparseIterate(T, p)
  _check_axes(out, X)
  _check_symmetric(X)
  @inbounds for c = 1:p, r = c:p
    v = X[r, c]
    iszero(v) || _store!(out, tril_index(p, r, c), v)
  end
  out
end

function Base.Matrix(S::SymmetricSparseIterate{Tv}) where Tv
  p = S.p
  A = zeros(Tv, p, p)
  for k=1:S.nnz
    i = S.nzval2ind[k]
    v = S.nzval[k]
    r, c = tril_cartesian(p, i)
    A[r, c] = v
    if r != c
      A[c, r] = v
    end
  end
  return A
end
Base.Array(S::SymmetricSparseIterate) = Matrix(S)

# x is symmetric, so x' * b and b' * x take the sparse product
Base.transpose(x::SymmetricSparseIterate) = x
Base.adjoint(x::SymmetricSparseIterate{<:Real}) = x

# a parameter array has an entry for every position of the matrix that x represents
_checkparam(w::AbstractArray, x::SymmetricSparseIterate) =
  length(w) == x.p^2 || _parameter_length_mismatch(length(w), x.p^2)

# copies of the state of x: values, active zeros and the order of activation
Base.copy(x::SymmetricSparseIterate) = copyto!(similar(x), x)
# empty symmetric iterates of the same dimension
Base.similar(x::SymmetricSparseIterate{T}) where {T} = SymmetricSparseIterate(T, x.p)
Base.similar(x::SymmetricSparseIterate, ::Type{S}) where {S} = SymmetricSparseIterate(S, x.p)

# Between symmetric iterates, a state copy: the working set of y replaces that of x, active zeros and
# order included, and only the active values are converted, before anything is written.
function Base.copyto!(x::SymmetricSparseIterate, y::SymmetricSparseIterate)
  x === y && return x
  x.p == y.p || throw(DimensionMismatch(
    "destination is $(x.p)×$(x.p), source is $(y.p)×$(y.p)"))
  _copyto_stored!(x, y)
end

# Equality of two symmetric iterates is that of their matrices, so it does not depend on which coordinates are
# active or on their order: isequal(x, y) is isequal(Matrix(x), Matrix(y)), and x == y is
# Matrix(x) == Matrix(y). An absent coordinate is zero(T), so an active -0.0 differs from it
# under isequal and not under ==, and paired NaNs are isequal and not ==. Both comparisons read
# the active coordinates of the two symmetric iterates and nothing else.
function _symmetric_iterates_agree(eq::F, x::SymmetricSparseIterate{S}, y::SymmetricSparseIterate{T}) where {F, S, T}
  @inbounds for k in 1:x.nnz
    eq(x.nzval[k], _getstored(y, x.nzval2ind[k])) || return false
  end
  @inbounds for k in 1:y.nnz
    x.ind2nzval[y.nzval2ind[k]] == 0 && !eq(zero(S), y.nzval[k]) && return false
  end
  true
end
function Base.isequal(x::SymmetricSparseIterate, y::SymmetricSparseIterate)
  x === y && return true
  x.p == y.p && _symmetric_iterates_agree(isequal, x, y)
end
Base.:(==)(x::SymmetricSparseIterate, y::SymmetricSparseIterate) = x.p == y.p && _symmetric_iterates_agree(==, x, y)
# The sum over the coordinates that are not the zero of an absent coordinate, of the hash of
# the value and the coordinate: independent of the order of the active set. hash of a real
# number does not depend on its type, so symmetric iterates of different element types that are isequal
# hash alike, and an active -0.0 contributes where an absent coordinate does not.
function Base.hash(x::SymmetricSparseIterate, h::UInt)
  acc = zero(UInt)
  @inbounds for k in 1:x.nnz
    v = x.nzval[k]
    isequal(v, zero(v)) || (acc += hash(v, hash(x.nzval2ind[k])))
  end
  hash(acc, hash(x.p, hash(:SymmetricSparseIterate, h)))
end

# From a matrix, a checked import that replaces the state of x: the matrix is real, of the
# shape of x, does not share memory with x and is exactly symmetric, all before anything is
# written. The working set is then the coordinates of nonzero converted value, in increasing
# order of coordinate.
Base.copyto!(x::SymmetricSparseIterate, y::AbstractMatrix) = _import_matrix!(x, y)
function _import_matrix!(x::SymmetricSparseIterate{T}, y::AbstractMatrix) where {T}
  _checkreal(x, y)
  _check_axes(x, y)
  _aliases(x, y) && throw(ArgumentError(
    "the source of copyto!(x, A) shares memory with x, as matrixview(x) does: copy it first, " *
    "for example with Matrix(A)"))
  _check_symmetric(y)
  _clear!(x)
  p = x.p
  @inbounds for c = 1:p, r = c:p
    v = convert(T, y[r, c])
    iszero(v) || _store!(x, tril_index(p, r, c), v)
  end
  x
end
# Whether y shares memory with the storage of x. An empty array shares nothing, as for
# Base.mightalias: all empty arrays of a type can share one empty buffer. The wrappers of
# LinearAlgebra do not all report the data of their parent, so the parents are followed.
_aliases(x::SymmetricSparseIterate, y) = x.p > 0 && !isempty(y) && _shares_data(Base.dataids(x), y)
function _shares_data(ids, y)
  any(id -> id in ids, Base.dataids(y)) && return true
  q = parent(y)
  q === y ? false : _shares_data(ids, q)
end
# aliasing hooks from Base: a symmetric iterate is not an array, so Base.unalias copies a view of it
# (through the unaliascopy methods of the views) when it is the destination
Base.mightalias(x::SymmetricSparseIterate, a::AbstractArray) = _aliases(x, a)
Base.mightalias(a::AbstractArray, x::SymmetricSparseIterate) = _aliases(x, a)
# A source with another number of dimensions cannot have the axes of x.
Base.copyto!(x::SymmetricSparseIterate, y::AbstractArray) = _throw_axes(x, y)

# A broadcast that is written into an iterate is real, for the symmetric iterate's sources and for BlockIterate.
_checkreal_broadcast(x, bc::Broadcast.Broadcasted) = _checkreal_args(x, bc.args)
_checkreal_args(x, args::Tuple) = (_checkreal_broadcast(x, first(args)); _checkreal_args(x, Base.tail(args)))
_checkreal_args(x, ::Tuple{}) = true
# Broadcast leaves can be control arguments, such as functions or rounding modes.
_checkreal_broadcast(x, a) = true
_checkreal_broadcast(x, a::Union{AbstractArray, Ref, Number}) =
  !(eltype(a) <: Complex) || _real_error(nameof(typeof(x)))

# The result of a broadcast into an iterate is real as well. Its element type R is read from
# the types. _staged_result is false when R is Union{} or a subtype of Real (which includes
# Bool, Integer, Rational and BigFloat results): the broadcast then writes directly. The test
# of R is resolved by inference when the function and the argument types are known to the
# compiler, the usual case. Any other R is staged. A complex R is rejected before
# anything is evaluated, and a Union with a complex member, Any or Number is evaluated into a
# temporary first, so that nothing is written when an element is complex or f throws.
@inline function _staged_result(bc::Broadcast.Broadcasted)
  R = Broadcast.combine_eltypes(bc.f, bc.args)
  !(R === Union{} || R <: Real)
end
@noinline function _staged_broadcast!(x::AbstractArray, bc::Broadcast.Broadcasted{Nothing})
  R = Broadcast.combine_eltypes(bc.f, bc.args)
  axes(x) == axes(bc) ||
    throw(DimensionMismatch("destination axes $(axes(x)) do not match the broadcast axes $(axes(bc))"))
  R <: Complex && _real_error(nameof(typeof(x)))
  # the broadcast that bc represents, with the axes it carries (singleton expansion, an empty destination)
  staged = copy(Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{ndims(x)}}(bc.f, bc.args, bc.axes))
  any(v -> v isa Complex, staged) && _real_error(nameof(typeof(x)))
  @inbounds for I in eachindex(staged)
    x[I] = staged[I]
  end
  x
end

# The stored entries as a SparseMatrixCSC, in O(nactive): each nonzero (r, c) of the
# lower triangle gives the entries (r, c) and (c, r) for which keep holds. The generic
# sparse of a matrix visits all p² entries, and the generic triu and tril of the view
# return a dense matrix.
function _sparse(x::SymmetricSparseIterate{T}, keep::F) where {T, F}
  I, J, V = Int[], Int[], T[]
  for k = 1:x.nnz
    v = x.nzval[k]
    iszero(v) && continue
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    keep(r, c) && (push!(I, r); push!(J, c); push!(V, v))
    r != c && keep(c, r) && (push!(I, c); push!(J, r); push!(V, v))
  end
  sparse(I, J, V, x.p, x.p)
end
# the view and its triangles as sparse matrices; a stored zero is not an entry
SparseArrays.sparse(A::SymmetricMatrixView) = _sparse(A.x, (i, j) -> true)
LinearAlgebra.triu(A::SymmetricMatrixView, k::Integer = 0) = _sparse(A.x, (i, j) -> j - i >= k)
LinearAlgebra.tril(A::SymmetricMatrixView, k::Integer = 0) = _sparse(A.x, (i, j) -> j - i <= k)

# one line: the shape, the type and the number of active coordinates, not the storage arrays
function Base.show(io::IO, S::SymmetricSparseIterate)
  n = nactive(S)
  print(io, S.p, "×", S.p, " ", typeof(S), " with ", n, " stored ", n == 1 ? "entry" : "entries")
end

function Base.show(io::IO, ::MIME"text/plain", S::SymmetricSparseIterate)
  xnnz = nactive(S)
  print(io, S.p, "×", S.p, " ", typeof(S), " with ", xnnz, " stored ",
        xnnz == 1 ? "entry" : "entries", " in the lower triangle")
  xnnz == 0 && return
  print(io, ":")
  _show_entries(io, xnnz, pos -> tril_cartesian(S.p, S.nzval2ind[pos]), pos -> S.nzval[pos], S.p)
end

LinearAlgebra.mul!(C::AbstractMatrix{T}, A::SymmetricSparseIterate{T}, B::AbstractMatrix{T}) where T = _mul_symsparse!(C, A, B)

function _mul_symsparse!(C::AbstractMatrix{T}, A::SymmetricSparseIterate{T}, B::AbstractMatrix{T}) where T
  _checkreal(A, C); _checkreal(A, B)
  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())
  size(B, 2) == size(C, 2) || throw(DimensionMismatch())
  _checkalias(C, A, B)

  fill!(C, zero(T))
  i = 1
  @inbounds while i <= A.nnz
    v = A.nzval[i]
    ind = A.nzval2ind[i]
    r, c = tril_cartesian(size(A, 1), ind)
    if iszero(v)
      # a zero coefficient contributes nothing, even against a non-finite entry of B
    elseif r == c
      for k=1:size(C, 2)
        @inbounds C[r, k] += B[c, k] * v
      end
    else
      for k=1:size(C, 2)
        @inbounds C[r, k] += B[c, k] * v
        @inbounds C[c, k] += B[r, k] * v
      end
    end
    i += 1
  end
  C
end


# checked before C is modified: C may be activevalues(A), or coordinates(A) when p == 1
function _checkalias(C, A::SymmetricSparseIterate, B)
  Base.mightalias(C, B) && throw(ArgumentError("the output of mul! must not be aliased with its input"))
  _aliases(A, C) && throw(ArgumentError("the output of mul! must not be aliased with the matrix, e.g. coordinates(X) or activevalues(X)"))
  nothing
end

# any element types: the generic fallback would loop over all p² entries
function LinearAlgebra.mul!(C::AbstractVector, A::SymmetricSparseIterate, B::AbstractVector)
  _checkreal(A, C); _checkreal(A, B)
  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())
  _checkalias(C, A, B)

  fill!(C, zero(eltype(C)))
  i = 1
  @inbounds while i <= A.nnz
    v = A.nzval[i]
    ind = A.nzval2ind[i]
    r, c = tril_cartesian(size(A, 1), ind)
    if iszero(v)
      # a zero coefficient contributes nothing, even against a non-finite entry of B
    elseif r == c
        @inbounds C[r] += B[c] * v
    else
        @inbounds C[r] += B[c] * v
        @inbounds C[c] += B[r] * v
    end
    i += 1
  end
  C
end

# a' X b over the stored entries of X: the entry (A X B)[r, c] of symmetric A, B
# is dot(view(A, :, r), X, view(B, :, c))
function LinearAlgebra.dot(a::AbstractVector, X::SymmetricSparseIterate, b::AbstractVector)
  _checkreal(X, a); _checkreal(X, b)
  Base.require_one_based_indexing(a, b)
  length(a) == X.p == length(b) || throw(DimensionMismatch())
  s = zero(promote_type(eltype(a), eltype(X), eltype(b)))
  @inbounds for k = 1:X.nnz
    v = X.nzval[k]
    iszero(v) && continue     # a zero coefficient contributes nothing, even against a non-finite entry
    r, c = tril_cartesian(X.p, X.nzval2ind[k])
    s += r == c ? a[r] * v * b[c] : (a[r] * b[c] + a[c] * b[r]) * v
  end
  s
end

# the trace inner product tr(A'X) over the stored entries of X
function LinearAlgebra.dot(A::StridedMatrix, X::SymmetricSparseIterate)
  _checkreal(X, A)
  size(A) == size(X) || throw(DimensionMismatch())
  s = zero(promote_type(eltype(A), eltype(X)))
  @inbounds for k = 1:X.nnz
    v = X.nzval[k]
    iszero(v) && continue     # a zero coefficient contributes nothing, even against a non-finite entry
    r, c = tril_cartesian(X.p, X.nzval2ind[k])
    s += r == c ? dot(A[r, c], v) : dot(A[r, c] + A[c, r], v)
  end
  s
end
LinearAlgebra.dot(X::SymmetricSparseIterate, A::StridedMatrix) = dot(A, X)

# the 1-norm counts off-diagonal entries twice; the infinity norm needs them once
function LinearAlgebra.norm(x::SymmetricSparseIterate, p::Real = 2)
  p == Inf && return norm(view(x.nzval, 1:x.nnz), Inf)
  if p == 1
    s = zero(real(float(eltype(x))))
    @inbounds for k in 1:x.nnz
      r, c = tril_cartesian(x.p, x.nzval2ind[k])
      v = norm(x.nzval[k])
      s += r == c ? v : 2v
    end
    return s
  end
  p == 2 && return _norm2_scaled(x)
  norm(matrixview(x), p)
end

function *(A::Symmetric{T}, X::SymmetricSparseIterate{T}, B::Symmetric{T}) where {T}
  _checkreal(X, A); _checkreal(X, B)
  # Dense BLAS products cost O(p³) however sparse X is. Restricted to the rows K where
  # X has nonzeros, A[:, K] * X[K, K] * B[K, :] costs O(p² |K|). Once X has nonzeros
  # in more than half of the rows, the dense products are as fast.
  p = X.p
  size(A, 2) == p == size(B, 1) || throw(DimensionMismatch())
  pos = zeros(Int, p)
  nK = 0
  for k = 1:X.nnz
    iszero(X.nzval[k]) && continue
    for i in tril_cartesian(p, X.nzval2ind[k])
      pos[i] == 0 && (pos[i] = 1; nK += 1)
    end
    2 * nK > p && return A * (Matrix(X) * B)
  end

  K = findall(!iszero, pos)
  pos[K] = 1:nK                 # position of each row within K
  XK = zeros(T, nK, nK)
  for k = 1:X.nnz
    iszero(X.nzval[k]) && continue
    r, c = tril_cartesian(p, X.nzval2ind[k])
    XK[pos[r], pos[c]] = XK[pos[c], pos[r]] = X.nzval[k]
  end
  A[:, K] * (XK * B[K, :])
end


# the other element types: the stored-entry kernel of the five-argument product
function LinearAlgebra.mul!(C::AbstractMatrix, A::SymmetricSparseIterate, B::AbstractMatrix)
  _checkreal(A, C); _checkreal(A, B)
  _mul_stored!(C, A, B, true, false)
end
function LinearAlgebra.mul!(C::AbstractVector, A::SymmetricSparseIterate, B::AbstractVector, α::Number, β::Number)
  _checkreal(A, C); _checkreal(A, B); _checkreal(A, α); _checkreal(A, β)
  _mul_stored!(C, A, B, α, β)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::SymmetricSparseIterate, B::AbstractMatrix, α::Number, β::Number)
  _checkreal(A, C); _checkreal(A, B); _checkreal(A, α); _checkreal(A, β)
  _mul_stored!(C, A, B, α, β)
end

# C = α A B + β C over the stored entries of A, O(p + nactive(A)) and no allocation.
# As in BLAS, C is not read when β is zero and B is not read when α is zero.
# A coefficient that is zero, stored or not, contributes nothing, even against a
# non-finite entry of B or a non-finite α.
function _mul_stored!(C::AbstractVector, A::SymmetricSparseIterate, B::AbstractVector, α::Number, β::Number)
  p = A.p
  length(B) == p || throw(DimensionMismatch("A has $p columns but B has length $(length(B))"))
  length(C) == p || throw(DimensionMismatch("A has $p rows but C has length $(length(C))"))
  _checkalias(C, A, B)
  iszero(β) ? fill!(C, zero(eltype(C))) : isone(β) || rmul!(C, β)
  iszero(α) && return C
  @inbounds for k in 1:A.nnz
    v0 = A.nzval[k]
    iszero(v0) && continue
    r, c = tril_cartesian(p, A.nzval2ind[k])
    v = α * v0
    C[r] += B[c] * v
    r == c || (C[c] += B[r] * v)
  end
  C
end
function _mul_stored!(C::AbstractMatrix, A::SymmetricSparseIterate, B::AbstractMatrix, α::Number, β::Number)
  p = A.p
  Base.require_one_based_indexing(C, B)
  size(B, 1) == p || throw(DimensionMismatch("A has $p columns but B has $(size(B, 1)) rows"))
  size(C, 1) == p || throw(DimensionMismatch("A has $p rows but C has $(size(C, 1))"))
  size(C, 2) == size(B, 2) || throw(DimensionMismatch("B has $(size(B, 2)) columns but C has $(size(C, 2))"))
  _checkalias(C, A, B)
  iszero(β) ? fill!(C, zero(eltype(C))) : isone(β) || rmul!(C, β)
  iszero(α) && return C
  n = size(C, 2)
  @inbounds for k in 1:A.nnz
    v0 = A.nzval[k]
    iszero(v0) && continue
    r, c = tril_cartesian(p, A.nzval2ind[k])
    v = α * v0
    for j in 1:n
      C[r, j] += B[c, j] * v
      r == c || (C[c, j] += B[r, j] * v)
    end
  end
  C
end

# the allocating products: the type of the sum of the products, as for a matrix
_product_eltype(::Type{Ta}, ::Type{Tb}) where {Ta, Tb} =
  (T = Base.promote_op(*, Ta, Tb); Base.promote_op(+, T, T))
function Base.:*(A::SymmetricSparseIterate, x::AbstractVector)
  _checkreal(A, x)
  mul!(Vector{_product_eltype(eltype(A), eltype(x))}(undef, A.p), A, x)
end
function Base.:*(A::SymmetricSparseIterate, B::AbstractMatrix)
  _checkreal(A, B)
  mul!(Matrix{_product_eltype(eltype(A), eltype(B))}(undef, A.p, size(B, 2)), A, B)
end

# norm_diff(A, B, p) of the matrices that the arguments represent: both triangles of a symmetric iterate
norm_diff(A::SymmetricSparseIterate, B::SymmetricSparseIterate, p::Real = 2) =
  norm_diff(matrixview(A), matrixview(B), p)
norm_diff(A::SymmetricSparseIterate, B::AbstractArray, p::Real = 2) = norm_diff(matrixview(A), B, p)
norm_diff(A::AbstractArray, B::SymmetricSparseIterate, p::Real = 2) = norm_diff(A, matrixview(B), p)

# Mixed real operands keep LinearAlgebra's multiplication order.
function *(A::Symmetric, X::SparseIterate{<:Any, 2}, B::Symmetric)
  _checkreal(X, A); _checkreal(X, B)
  invoke(*, Tuple{AbstractMatrix, AbstractMatrix, AbstractMatrix}, A, X, B)
end
# a symmetric iterate of another element type than the factors: the dense product of the matrix it represents
function *(A::Symmetric, X::SymmetricSparseIterate, B::Symmetric)
  _checkreal(X, A); _checkreal(X, B)
  size(A, 2) == X.p == size(B, 1) || throw(DimensionMismatch())
  (A * Matrix(X)) * B
end

"""
    *(A::Symmetric{T}, X::SymmetricSparseIterate{T}, B::Symmetric{T}) where {T<:Union{Float32, Float64}}

The product `C = A * X * B` for a symmetric sparse iterate `X`, with the same contract as the
product for a matrix `SparseIterate`: `C[i, j]` is the sum of `A[i, k] * X[k, l] * B[l, j]` over the
entries `(k, l)` of the `p × p` matrix `X` that are mathematically nonzero, in `O(X.nnz * p^2)`
operations and `O(p^2)` memory. A stored off-diagonal coefficient stands for the two entries
`(r, c)` and `(c, r)`, each contributing one term, and a diagonal coefficient for one. A stored
zero and an unstored entry are the same: no term is formed, and an `X` without nonzeros gives the
zero matrix without reading `A` or `B`. A nonzero coefficient, including `NaN` and `±Inf`, always
contributes; the zeros of `A` and `B` are ordinary zeros, so `0 * Inf` is `NaN`. The result need
not equal `A * Matrix(X) * B`.

A term with three finite nonzero factors has the accuracy of an ordinary product of three
numbers: it differs from the correctly rounded exact product by at most one neighbouring
floating-point number for a normal result (a relative error below about `2 eps(T)`), and by at
most one spacing for a subnormal one. A term within that error of the overflow threshold or of
half the smallest subnormal can fall on either side of it. Correct rounding is not promised.
Beyond that margin the class of a term is decided by its final value and not by the first partial
product. Terms are added in the storage order of `X` in the precision `T`, the entry `(r, c)` of a
coefficient before `(c, r)`, so tiny terms next to large ones are rounded away, as in any
floating-point sum. An entry whose ordinary sum is non-finite is recomputed from its terms as a
scaled sum when every factor of every nonzero entry of `X` at that entry is finite, so a sum that
overflows before a later cancellation gives its finite value; if any such factor is `NaN` or
infinite the ordinary IEEE value stands. A sum that is exactly zero is `+0.0`.

The guarantee covers `Float32` and `Float64` with `Symmetric` wrappers of strided matrices of the
same element type as `X`. Other element types, and mixed element types, are outside the guarantee
and take other methods of `*`. The method reads the stored fields `nzval`, `nzval2ind`, `nnz` and
`p`. See Numerical contract.
"""
function *(A::Symmetric{T, <:StridedMatrix{T}}, X::SymmetricSparseIterate{T},
           B::Symmetric{T, <:StridedMatrix{T}}) where {T<:Union{Float32, Float64}}
  _checkreal(X, A); _checkreal(X, B)
  p = X.p
  size(A, 2) == p == size(B, 1) || throw(DimensionMismatch())
  C = zeros(T, p, p)
  a, ax, b = Vector{T}(undef, p), Vector{T}(undef, p), Vector{T}(undef, p)
  for k = 1:X.nnz
    x = X.nzval[k]
    iszero(x) && continue
    r, c = tril_cartesian(p, X.nzval2ind[k])
    _add_coefficient!(C, A, B, r, c, x, a, ax, b)
    r == c || _add_coefficient!(C, A, B, c, r, x, a, ax, b)
  end
  _rescue_nonfinite!(C, A, X, B)
end
