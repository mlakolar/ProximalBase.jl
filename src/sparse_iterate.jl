####################################
#
#  Sparse Iterate
#
####################################

SparseIterate(n::Integer) = SparseIterate(Float64, n)
SparseIterate(::Type{T}, n::Integer) where {T} = _sparse_iterate(T, Int(n))
SparseIterate(n::Integer, m::Integer) = SparseIterate(Float64, n, m)
SparseIterate(::Type{T}, n::Integer, m::Integer) where {T} = _sparse_iterate(T, Int(n), Int(m))

_sparse_iterate(::Type{T}, n::Int) where {T} = SparseIterate{T, 1}(zeros(T, n), zeros(Int, n), zeros(Int, n), 0)
function _sparse_iterate(::Type{T}, n::Int, m::Int) where {T}
  n >= 0 && m >= 0 || throw(ArgumentError("SparseIterate dimensions must be nonnegative, got ($n, $m)"))
  q = Base.widemul(n, m)
  q <= typemax(Int) || throw(ArgumentError(
    "SparseIterate dimensions ($n, $m) are too large: coordinate count exceeds typemax(Int)"))
  k = Int(q)
  SparseIterate{T, 2}(zeros(T, k), zeros(Int, k), zeros(Int, n, m), 0)
end

Base.length(x::SparseIterate) = length(x.ind2nzval)
Base.size(x::SparseIterate) = size(x.ind2nzval)
Base.IndexStyle(::Type{<:SparseIterate}) = IndexLinear()

Base.@propagate_inbounds Base.getindex(x::SparseIterate, i::Int) = _getstored(x, i)
Base.@propagate_inbounds Base.setindex!(x::SparseIterate, v, i::Int) = _setstored!(x, v, i)

# the coordinates of x, its entries in linear order, as a vector
struct SparseCoordinates{T, N} <: AbstractVector{T}
  x::SparseIterate{T, N}
end

# the error is built from x, so that the view does not escape on the checked path
@noinline _coordinates_boundserror(x::SparseIterate, i) = throw(BoundsError(SparseCoordinates(x), i))

Base.size(c::SparseCoordinates) = (length(c.x),)
Base.IndexStyle(::Type{<:SparseCoordinates}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(c::SparseCoordinates, i::Int)
  @boundscheck (1 <= i <= length(c.x)) || _coordinates_boundserror(c.x, i)
  @inbounds _getstored(c.x, i)
end
Base.@propagate_inbounds function Base.setindex!(c::SparseCoordinates, v, i::Int)
  @boundscheck (1 <= i <= length(c.x)) || _coordinates_boundserror(c.x, i)
  @inbounds _setstored!(c.x, v, i)
  c
end

# aliasing hooks from Base (non-public functions): the view overlaps x, and an aliased
# source is copied first
Base.dataids(c::SparseCoordinates) = Base.dataids(c.x)
Base.unaliascopy(c::SparseCoordinates) = coordinates(copy(c.x))

# the value is converted as for a plain vector of the element type, then written as
# fill!(x, v) writes it: a zero goes to the stored entries only
Base.fill!(c::SparseCoordinates{T}, v) where {T} = (fill!(c.x, convert(T, v)); c)

@inline coordinates(x::SparseIterate{T, N}) where {T, N} = SparseCoordinates{T, N}(x)

## similar
#
Base.similar(A::SparseIterate{T}) where {T} = SparseIterate(T, size(A)...)
Base.similar(A::SparseIterate, ::Type{S}) where {S} = SparseIterate(S, size(A)...)


### Conversion
Base.convert(::Type{<:SparseIterate{T}}, ::AbstractArray) where {T<:Complex} = _real_error(:SparseIterate)

function SparseIterate(x::SparseMatrixCSC{T}) where {T}
  n, m = size(x)
  out = SparseIterate(T, n, m)

  I, J, V = findnz(x)
  for i=1:length(I)
    @inbounds out[I[i], J[i]] = V[i]
  end

  out
end

function SparseIterate(x::SparseVector{T}) where {T}
  p = length(x)
  out = SparseIterate(T, p)
  rowval, nzval = findnz(x)
  for i=1:length(nzval)
    out[rowval[i]] = nzval[i]
  end
  out
end

function SparseIterate(x::Array{T}) where {T}
  out = SparseIterate(T, size(x)...)

  @inbounds for i in eachindex(x)
      out[i] = x[i]
  end

  out
end



function Base.Vector(x::SparseIterate{Tv, 1}) where Tv
    r = zeros(Tv, length(x))
    for k in 1:nactive(x)
        i = x.nzval2ind[k]
        v = x.nzval[k]
        r[i] = v
    end
    return r
end

function Base.Matrix(x::SparseIterate{Tv, 2}) where Tv
  n, m = size(x)
  A = zeros(Tv, n, m)
  for k=1:nactive(x)
    i = x.nzval2ind[k]
    v = x.nzval[k]
    A[i] = v
  end
  return A
end

# Array(x) and convert(Array, x) otherwise fall back to copying entry by entry
Base.Array(x::SparseIterate{T, 1}) where {T} = Vector(x)
Base.Array(x::SparseIterate{T, 2}) where {T} = Matrix(x)


# Between iterates of the same size, a state copy. Another shape of the same length is
# a value write in linear order, as from any other array.
function Base.copyto!(x::SparseIterate, y::SparseIterate)
    _checkreal_broadcast(x, y)
    size(x) == size(y) && return _copyto_stored!(x, y)
    _check_iterate_lengths(x, y)
    invoke(copyto!, Tuple{AbstractArray, AbstractArray}, x, y)
end
Base.copy!(x::SparseIterate, y::SparseIterate) = copyto!(x, y)

# A copy between two array iterates of different lengths throws before writing.
_check_iterate_lengths(x, y) = length(x) == length(y) ||
    throw(DimensionMismatch("destination has length $(length(x)), source has length $(length(y))"))

# The check reads the element type and runs before anything is written; a source
# of abstract element type converts entry by entry, as in Base.
function Base.copyto!(x::SparseIterate, y::AbstractArray)
  _checkreal_broadcast(x, y)
  invoke(copyto!, Tuple{AbstractArray, AbstractArray}, x, y)
end
# Resolve ambiguities with SparseArrays: each runs the method that applies without the check.
# SparseArrays' copyto!(A::AbstractMatrix, B::AbstractSparseMatrixCSC), on every supported Julia.
function Base.copyto!(x::SparseIterate{<:Any, 2}, y::SparseArrays.AbstractSparseMatrixCSC)
  _checkreal_broadcast(x, y)
  invoke(copyto!, Tuple{AbstractMatrix, SparseArrays.AbstractSparseMatrixCSC}, x, y)
end
# SparseArrays 1.14 adds copyto!(dest::AbstractMatrix, src) for adjoint and transposed CSC
# matrices, matrix views and transposed sparse vectors; before it, the copyto! from an
# AbstractArray above applies, with the same check and Base's method.
function Base.copyto!(x::SparseIterate{T, 2}, y::Union{Adjoint{<:Any, <:SparseArrays.AbstractSparseMatrixCSC},
                                                       Transpose{<:Any, <:SparseArrays.AbstractSparseMatrixCSC}}) where {T}
  _checkreal_broadcast(x, y)
  invoke(copyto!, Tuple{AbstractMatrix{T}, typeof(y)}, x, y)
end
@static if isdefined(@__MODULE__, :_SparseMatrixCSCOrView)
  function Base.copyto!(x::SparseIterate{T, 2}, y::_SparseMatrixCSCOrView) where {T}
    _checkreal_broadcast(x, y)
    invoke(copyto!, Tuple{AbstractMatrix{T}, typeof(y)}, x, y)
  end
end
@static if isdefined(@__MODULE__, :_AdjOrTransSparseVectorOrView)
  function Base.copyto!(x::SparseIterate{T, 2}, y::_AdjOrTransSparseVectorOrView) where {T}
    _checkreal_broadcast(x, y)
    invoke(copyto!, Tuple{AbstractMatrix{T}, typeof(y)}, x, y)
  end
end
# CHOLMOD's copyto!(dest::AbstractArray, D::Dense); this method is also more specific than
# its methods for a destination of the element type of D, which copy with the same kernel.
@static if isdefined(@__MODULE__, :_CHOLMODDense)
  function Base.copyto!(x::SparseIterate, y::_CHOLMODDense)
    _checkreal_broadcast(x, y)
    invoke(copyto!, Tuple{AbstractArray, _CHOLMODDense}, x, y)
  end
end
# I and orthogonal factors are not arrays, so their element type is checked here.
function Base.copyto!(x::SparseIterate{<:Any, 2}, y::UniformScaling)
  !(eltype(y) <: Complex) || _real_error(:SparseIterate)
  invoke(copyto!, Tuple{AbstractMatrix, UniformScaling}, x, y)
end
# Q is a lazy factorization, so it is materialized into a dense matrix of the element
# type of x first; the validation, the materialization and the conversion all run
# before x is touched, and the result is written as an ordinary array value write.
# (LinearAlgebra's copyto!(dest::AbstractMatrix, Q::AbstractQ) skips the real-only check,
# writes I into x and multiplies it in place, which fails for most factors after x changed.)
function Base.copyto!(x::SparseIterate{T, 2}, y::LinearAlgebra.AbstractQ) where {T}
  !(eltype(y) <: Complex) || _real_error(:SparseIterate)
  size(y) == size(x) || throw(DimensionMismatch(
    "destination has size $(size(x)), orthogonal factor has size $(size(y))"))
  # Matrix(y) can be thin (a tall QR factor), so the full size(y) matrix is built from the identity.
  staged = convert(Matrix{T}, lmul!(y, Matrix{eltype(y)}(I, size(y))))
  invoke(copyto!, Tuple{AbstractArray, AbstractArray}, x, staged)
  return x
end

function Base.copyto!(x::SparseIterate, bc::Broadcast.Broadcasted{Nothing})
  _checkreal_broadcast(x, bc)
  _staged_result(bc) && return _staged_broadcast!(x, bc)
  _broadcast_loop!(x, bc)
end

# A broadcast into an array iterate is the explicit loop over the unaliased
# right-hand side: Base's loop without its shortcut to copyto!(x, y), which would
# make x .= y a state copy when y is an iterate of the same kind. For a matrix the
# loop runs over Cartesian indices, in linear order.
@inline function _broadcast_loop!(x::AbstractArray, bc::Broadcast.Broadcasted{Nothing})
  axes(x) == axes(bc) ||
    throw(DimensionMismatch("destination axes $(axes(x)) do not match the broadcast axes $(axes(bc))"))
  bc′ = Broadcast.preprocess(x, bc)
  @inbounds for I in eachindex(bc′)
    x[I] = bc′[I]
  end
  x
end

function Base.copyto!(x::SparseIterate, bc::Broadcast.Broadcasted{S}) where {S<:Broadcast.DefaultArrayStyle{0}}
  _checkreal_broadcast(x, bc)
  invoke(copyto!, Tuple{AbstractArray, Broadcast.Broadcasted{S}}, x, bc)
end

# fill! is a value write: the value is checked and converted first, and a zero is
# written into the stored entries, which stay stored (dropzeros!(fill!(x, 0))
# removes them)
function Base.fill!(x::SparseIterate{T}, v) where {T}
  _checkreal_broadcast(x, v)
  s = convert(T, v)
  if iszero(s)
    @inbounds for k in 1:x.nnz
      x.nzval[k] = s
    end
  else
    @inbounds for i in eachindex(x)
      x[i] = s
    end
  end
  x
end

# the entries that are not stored are zero and do not count in a p-norm, p ≥ 0
LinearAlgebra.norm(x::SparseIterate, p::Real = 2) =
  p >= 0 ? norm(view(x.nzval, 1:x.nnz), p) : invoke(norm, Tuple{Any, Real}, x, p)


### show and friends

Base.show(io::IO, x::SparseIterate) = show(convert(IOContext, io), x)

function Base.show(io::IO, ::MIME"text/plain", x::SparseIterate{T, 1}) where T
    xnnz = nactive(x)
    print(io, length(x), "-element ", typeof(x), " with ", xnnz,
           " stored ", xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(io, x)
    end
end

function Base.show(io::IOContext, x::SparseIterate{T, 1}) where T
    n = length(x)
    xnnz = nactive(x)
    if xnnz == 0
        return show(io, MIME("text/plain"), x)
    end
    limit::Bool = get(io, :limit, false)
    half_screen_rows = limit ? max(div(displaysize(io)[1] - 8, 2), 1) : typemax(Int)
    pad = ndigits(n)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    # the first and the last stored entries, counted in the order of their indices
    ind = 0
    for k = 1:n
        x.ind2nzval[k] == 0 && continue
        ind += 1
        if ind < half_screen_rows || ind > xnnz - half_screen_rows
            print(io, "  ", '[', rpad(k, pad), "]  =  ")
            show(io, x[k])
            ind != xnnz && println(io)
        elseif ind == half_screen_rows
            println(io, "   ", " "^pad, "   ⋮")
        end
    end
end



function Base.show(io::IO, ::MIME"text/plain", S::SparseIterate{T,2}) where T
    xnnz = nactive(S)
    print(io, size(S, 1), "×", size(S, 2), " ", typeof(S), " with ", xnnz, " stored ",
              xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        print(io, ":")
        show(io, S)
    end
end

function Base.show(io::IOContext, S::SparseIterate{T, 2}) where T
  if nactive(S) == 0
    return show(io, MIME("text/plain"), S)
  end
  _show_entries(io, nactive(S), (pos -> Tuple(CartesianIndices(size(S))[S.nzval2ind[pos]])),
                pos -> S.nzval[pos], maximum(size(S)))
end

# print the stored entries listed by entry(pos) = (row, col) and value(pos) for
# pos in 1:nnz, in column-major order of the entries, eliding the middle ones
function _show_entries(io::IO, xnnz::Int, entry, value, maxdim::Int)
  limit::Bool = get(io, :limit, false)
  half_screen_rows = limit ? max(div(displaysize(io)[1] - 8, 2), 1) : typemax(Int)
  pad = ndigits(maxdim)
  if !haskey(io, :compact)
    io = IOContext(io, :compact => true)
  end
  order = sortperm(1:xnnz, by = pos -> reverse(entry(pos)))
  for (ind, pos) in enumerate(order)
    if ind < half_screen_rows || ind > xnnz - half_screen_rows
      row, col = entry(pos)
      print(io, "\n  ", '[', rpad(row, pad), ", ", lpad(col, pad), "]  =  ")
      show(io, value(pos))
    elseif ind == half_screen_rows
      print(io, "\n  ⋮")
    end
  end
end


## multiplication

# In the products and inner products below, a zero coefficient of the iterate,
# stored or not, contributes nothing, even against ±Inf or NaN in the other
# operand (numerics.md, "Sparse product semantics"): each kernel tests the stored value, before any scaling
# by α, so that an α·c that underflows still multiplies and α = Inf does not
# turn a zero coefficient into NaN.

function *(A::StridedMatrix{Ta}, x::SparseIterate{Tx, 1}) where {Ta,Tx}
  _checkreal(x, A)
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Vector{Ty}(undef, m)
    mul!(y, A, x)
end

LinearAlgebra.mul!(out::StridedVector, A::StridedMatrix, coef::SparseIterate{T, 1}) where {T} =
  mul!(out, A, coef, true, false)
function LinearAlgebra.mul!(out::StridedVector, A::StridedMatrix, coef::SparseIterate{T, 1}, α::Number, β::Number) where {T}
  _check_one_based(out, A)
  _checkreal(coef, out); _checkreal(coef, A); _checkreal(coef, α); _checkreal(coef, β)
  length(coef) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), length of coef is $(length(coef))"))
  length(out) == size(A, 1) || throw(DimensionMismatch("A has first dimension $(size(A,1)), length of out is $(length(out))"))

  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, coef, α)
  @inbounds for inz = 1:nactive(coef)
    v = coef.nzval[inz]
    iszero(v) && continue
    indColumn = coef.nzval2ind[inz]
    c = scale * v
    @simd for i = 1:size(A, 1)
      out[i] += c*A[i, indColumn]
    end
  end
  out
end

# the product with a matrix of coefficients, one column per response
LinearAlgebra.mul!(out::StridedMatrix, A::StridedMatrix, coef::SparseIterate{T, 2}) where {T} =
  mul!(out, A, coef, true, false)
function LinearAlgebra.mul!(out::StridedMatrix, A::StridedMatrix, coef::SparseIterate{T, 2}, α::Number, β::Number) where {T}
  _check_one_based(out, A)
  _checkreal(coef, out); _checkreal(coef, A); _checkreal(coef, α); _checkreal(coef, β)
  size(coef, 1) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), coef has $(size(coef, 1)) rows"))
  size(out) == (size(A, 1), size(coef, 2)) || throw(DimensionMismatch("out has size $(size(out))"))
  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, coef, α)
  # Few coefficients: visit stored entries. Many: sweep columns, reading A in order.
  # No fixed cutoff is best on every CPU; measurements of 1/32 and alternatives
  # are recorded in docs/design/iterate-product-notes-921.md.
  if 32 * nactive(coef) <= length(coef)
    p = size(coef, 1)
    @inbounds for inz = 1:nactive(coef)
      v = coef.nzval[inz]
      iszero(v) && continue
      j, k = mod1(coef.nzval2ind[inz], p), cld(coef.nzval2ind[inz], p)
      c = scale * v
      @simd for i = 1:size(A, 1)
        out[i, k] += c * A[i, j]
      end
    end
  else
    @inbounds for k = 1:size(coef, 2), j = 1:size(coef, 1)
      pos = coef.ind2nzval[j, k]
      pos == 0 && continue
      v = coef.nzval[pos]
      iszero(v) && continue
      c = scale * v
      @simd for i = 1:size(A, 1)
        out[i, k] += c * A[i, j]
      end
    end
  end
  out
end

const _StridedSymmetric = Symmetric{<:Any, <:StridedMatrix}

# Add c times column j, reading only the triangle that Symmetric owns.
function _symmetric_axpy!(out::StridedVector, A::_StridedSymmetric, j::Int, c)
  M = parent(A)
  n = size(A, 1)
  # One-line SIMD loops avoid phantom uncovered `end` lines on Julia 1.10.
  if A.uplo == 'U'
    @inbounds @simd for i in 1:j; out[i] += c * M[i, j]; end
    @inbounds @simd for i in j+1:n; out[i] += c * M[j, i]; end
  else
    @inbounds @simd for i in 1:j-1; out[i] += c * M[j, i]; end
    @inbounds @simd for i in j:n; out[i] += c * M[i, j]; end
  end
  out
end

_scale_product_output!(out, β) = iszero(β) ? fill!(out, zero(eltype(out))) : (out .*= β)
# Widen integer and Bool scales to the common operand precision.
_product_scale(A, x, α) = convert(promote_type(typeof(α), eltype(A), eltype(x)), α)

function *(A::_StridedSymmetric, x::SparseIterate{T, 1}) where {T}
  _checkreal(x, A)
  mul!(Vector{promote_type(eltype(A), T)}(undef, size(A, 1)), A, x)
end
function *(A::_StridedSymmetric, x::SparseIterate{T, 2}) where {T}
  _checkreal(x, A)
  mul!(Matrix{promote_type(eltype(A), T)}(undef, size(A, 1), size(x, 2)), A, x)
end

LinearAlgebra.mul!(out::StridedVector, A::_StridedSymmetric, x::SparseIterate{<:Any, 1}) =
  mul!(out, A, x, true, false)
function LinearAlgebra.mul!(out::StridedVector, A::_StridedSymmetric, x::SparseIterate{<:Any, 1}, α::Number, β::Number)
  _check_one_based(out, A)
  _checkreal(x, out); _checkreal(x, A); _checkreal(x, α); _checkreal(x, β)
  length(x) == size(A, 2) && length(out) == size(A, 1) || throw(DimensionMismatch())
  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, x, α)
  @inbounds for k in 1:x.nnz
    v = x.nzval[k]
    iszero(v) || _symmetric_axpy!(out, A, x.nzval2ind[k], scale * v)
  end
  out
end

LinearAlgebra.mul!(out::StridedMatrix, A::_StridedSymmetric, x::SparseIterate{<:Any, 2}) =
  mul!(out, A, x, true, false)
function LinearAlgebra.mul!(out::StridedMatrix, A::_StridedSymmetric, x::SparseIterate{<:Any, 2}, α::Number, β::Number)
  _check_one_based(out, A)
  _checkreal(x, out); _checkreal(x, A); _checkreal(x, α); _checkreal(x, β)
  size(x, 1) == size(A, 2) && size(out) == (size(A, 1), size(x, 2)) || throw(DimensionMismatch())
  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, x, α)
  p = size(x, 1)
  @inbounds for k in 1:x.nnz
    v = x.nzval[k]
    iszero(v) && continue
    j, col = mod1(x.nzval2ind[k], p), cld(x.nzval2ind[k], p)
    _symmetric_axpy!(view(out, :, col), A, j, scale * v)
  end
  out
end

# A' equals transpose(A) for real A, so both take the sparse path
const TransposedStridedMatrix{T} = Union{Transpose{<:Any,<:StridedMatrix{T}}, Adjoint{<:Real,<:StridedMatrix{T}}}

function *(trans_a::TransposedStridedMatrix{Ta}, x::SparseIterate{Tx, 1}) where {Ta,Tx}
  _checkreal(x, trans_a)
    A = trans_a.parent
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Vector{Ty}(undef, n)
    mul!(y, transpose(A), x)
end

LinearAlgebra.mul!(out::StridedVector, trans_a::TransposedStridedMatrix, coef::SparseIterate{T, 1}) where {T} =
    mul!(out, trans_a, coef, true, false)
function LinearAlgebra.mul!(out::StridedVector, trans_a::TransposedStridedMatrix, coef::SparseIterate{T, 1}, α::Number, β::Number) where {T}
  _check_one_based(out, trans_a)
  _checkreal(coef, out); _checkreal(coef, trans_a); _checkreal(coef, α); _checkreal(coef, β)
    A = trans_a.parent
    length(coef) == size(A, 1) || throw(DimensionMismatch("A has first dimension $(size(A,1)), length of coef is $(length(coef))"))
    length(out) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), length of out is $(length(out))"))

    iszero(α) && return _scale_product_output!(out, β)
    # No nonzero coefficient: the sum is empty, and out becomes β·out as on the strided
    # path, also for an α of Inf or NaN. The test stops at the first nonzero coefficient.
    any(!iszero, view(coef.nzval, 1:nactive(coef))) || return _scale_product_output!(out, β)
    scale = _product_scale(A, coef, α)
    @inbounds for j = 1:size(A, 2)
        s = zero(eltype(out))
        @simd for inz = 1:nactive(coef)
            v = coef.nzval[inz]
            t = A[coef.nzval2ind[inz], j] * v
            s += ifelse(iszero(v), zero(t), t)    # a select, not a branch, keeps the loop vectorized
        end
        out[j] = iszero(β) ? scale * s : scale * s + β * out[j]
    end
    return out
end

# dot(v, x) over the stored entries of x, for any strided v, such as a row or a
# column of a matrix: dot(view(A, i, :), x) is (A * x)[i] in O(nactive(x)). Other
# vectors take the generic method, as SparseArrays has its own for sparse vectors.
function LinearAlgebra.dot(v::StridedVector, coef::SparseIterate{<:Any, 1})
  _check_one_based(v)
  _checkreal(coef, v)
    length(v) == length(coef) || throw(DimensionMismatch())
    s = zero(promote_type(eltype(v), eltype(coef)))
    @inbounds @simd for icoef = 1:nactive(coef)
        c = coef.nzval[icoef]
        t = dot(v[coef.nzval2ind[icoef]], c)
        s += ifelse(iszero(c), zero(t), t)    # a select, not a branch, keeps the loop vectorized
    end
    s
end
LinearAlgebra.dot(coef::SparseIterate{<:Any, 1}, v::StridedVector) = dot(v, coef)

# Both operands are iterates, so a zero of either contributes nothing.
function LinearAlgebra.dot(x::SparseIterate{<:Any, 1}, y::SparseIterate{<:Any, 1})
    length(x) == length(y) || throw(DimensionMismatch())
    nactive(x) > nactive(y) && return dot(y, x)
    s = zero(promote_type(eltype(x), eltype(y)))
    @inbounds for k = 1:nactive(x)
        a, b = x.nzval[k], y[x.nzval2ind[k]]
        t = dot(a, b)
        s += ifelse(iszero(a) | iszero(b), zero(t), t)    # a select: a branch on the data mispredicts
    end
    s
end

# a' X b and the entries of A X B for a sparse matrix X, in O(nactive(X))
function LinearAlgebra.dot(a::StridedVector, X::SparseIterate{<:Any, 2}, b::StridedVector)
  _check_one_based(a, b)
  _checkreal(X, a); _checkreal(X, b)
  (length(a), length(b)) == size(X) || throw(DimensionMismatch())
  m = size(X, 1)
  s = zero(promote_type(eltype(a), eltype(X), eltype(b)))
  @inbounds for k = 1:nactive(X)
    v = X.nzval[k]
    iszero(v) && continue
    i = X.nzval2ind[k]
    s += a[mod1(i, m)] * v * b[cld(i, m)]
  end
  s
end

# Cold three-factor term: classify genuine NaN/Inf/zero inputs first, then form
# finite nonzero products in scaled units. The range and rounding contract is
# documented under "Three-factor products" in docs/src/iterates.md.
function _triple(a::T, x::T, b::T) where {T<:Union{Float32, Float64}}
  (isnan(a) | isnan(x) | isnan(b)) && return T(NaN)
  infinite = isinf(a) | isinf(x) | isinf(b)
  (iszero(a) | iszero(x) | iszero(b)) && return ifelse(infinite, T(NaN), zero(T))
  infinite && return ifelse(signbit(a) ⊻ signbit(x) ⊻ signbit(b), -T(Inf), T(Inf))
  ma, ea = _frexp(a)
  mx, ex = _frexp(x)
  mb, eb = _frexp(b)
  ldexp((ma * mx) * mb, ea + ex + eb)
end

# C[:, j] += ax * b[j]
@inline function _axpy_terms!(C::Matrix{T}, ax::Vector{T}, b::Vector{T}, j::Int) where {T}
  bj = b[j]
  @inbounds @simd for i in eachindex(ax)
    C[i, j] += ax[i] * bj
  end
end

# Adds the terms A[:, r] * x * B[c, :] of one nonzero coefficient x of X, at the coordinate (r, c),
# into C; a, ax and b are buffers of the lengths of the column of A and the row of B. Written so
# that a symmetric iterate method can call it once for each stored coefficient (and, for an off-diagonal
# coordinate, for its transpose).
function _add_coefficient!(C::Matrix{T}, A, B, r::Int, c::Int, x::T,
                           a::Vector{T}, ax::Vector{T}, b::Vector{T}) where {T}
  m, n = size(C)
  lo = floatmin(T)
  exceptional = false
  for i = 1:m                         # the logical column r of A: the ignored triangle is not read
    ai = A[i, r]
    a[i] = ai
    axi = ai * x
    ax[i] = axi
    exceptional |= !(lo <= abs(axi) < T(Inf)) & !iszero(ai)
  end
  for j = 1:n
    b[j] = B[c, j]                    # the logical row c of B
  end
  if exceptional                      # some a * x is not a finite normal number
    for j = 1:n
      bj = b[j]
      for i = 1:m
        axi = ax[i]
        t = (lo <= abs(axi) < T(Inf)) | iszero(a[i]) ? axi * bj : _triple(a[i], x, bj)
        C[i, j] += t
      end
    end
  else                                # a * x is finite and normal, so (a * x) * b is the final rounding
    for j = 1:n
      _axpy_terms!(C, ax, b, j)
    end
  end
  nothing
end

# Recomputes the entry (i, j) of C, which the ordinary sum left non-finite, as the sum of its terms
# scaled by the largest exponent among them, so that a partial sum that overflows before a later
# cancellation (+big, +big, -big) gives the finite value. It covers an entry whose terms all have
# finite factors, whether the ordinary result was Inf, -Inf or NaN (Inf - Inf); a sum that is
# beyond the range of T stays +-Inf, and an exactly zero sum is +0.0. It does not recover tiny terms
# lost against large ones, which is ordinary rounding. If any factor of any nonzero coefficient
# at this entry (a of A, x of X, b of B) is NaN or infinite, the entry is left as the ordinary sum
# made it, so a genuine NaN or Inf is never dropped.
# The stored coefficients of an iterate and the logical coordinates (r, c) of each: the second pair
# is (0, 0) when there is none (an off-diagonal coefficient of a SymmetricSparseIterate stands for
# (r, c) and (c, r)).
@inline _nstored(X::SparseIterate) = nactive(X)
@inline _nstored(X::SymmetricSparseIterate) = X.nnz
@inline function _logical_coordinates(X::SparseIterate{<:Any, 2}, k::Int)
  m = size(X, 1)
  ind = X.nzval2ind[k]
  (mod1(ind, m), cld(ind, m), 0, 0)
end
@inline function _logical_coordinates(X::SymmetricSparseIterate, k::Int)
  r, c = tril_cartesian(X.p, X.nzval2ind[k])
  r == c ? (r, c, 0, 0) : (r, c, c, r)
end

function _rescue_entry!(C::Matrix{T}, A, X::Union{SparseIterate{T, 2}, SymmetricSparseIterate{T}}, B,
                        i::Int, j::Int) where {T}
  S = zero(T)
  emax = 0
  seen = false
  for k = 1:_nstored(X)
    x = X.nzval[k]
    iszero(x) && continue
    r1, c1, r2, c2 = _logical_coordinates(X, k)
    for t = 1:2
      r, c = t == 1 ? (r1, c1) : (r2, c2)
      r == 0 && continue
      a, b = A[i, r], B[c, j]
      isfinite(a) & isfinite(x) & isfinite(b) || return nothing
      (iszero(a) | iszero(b)) && continue
      ma, ea = _frexp(a)
      mx, ex = _frexp(x)
      mb, eb = _frexp(b)
      e = ea + ex + eb
      if !seen || e > emax
        S = ldexp(S, emax - e)        # S is zero before the first term
        emax, seen = e, true
      end
      S += ldexp((ma * mx) * mb, e - emax)
    end
  end
  C[i, j] = ldexp(S, emax)
  nothing
end

# One vectorized scan of C for entries that the ordinary sum left non-finite, each recomputed by
# _rescue_entry!; nothing else is done on the ordinary path.
function _rescue_nonfinite!(C::Matrix{T}, A, X, B) where {T}
  finite = true
  @inbounds @simd for idx in eachindex(C)
    finite &= isfinite(C[idx])
  end
  if !finite
    for j = 1:size(C, 2), i = 1:size(C, 1)
      isfinite(C[i, j]) || _rescue_entry!(C, A, X, B, i, j)
    end
  end
  C
end

"""
    *(A::Symmetric{T}, X::SparseIterate{T, 2}, B::Symmetric{T}) where {T<:Union{Float32, Float64}}

Return the dense matrix `A * X * B` by summing terms for nonzero entries of `X`.
The direct method requires `Symmetric` wrappers of strided matrices, with all
three operands of the same `Float32` or `Float64` element type and compatible
dimensions. For `X` of size `m × n`, it costs O(nactive(X) * m * n) operations
and O(m * n) memory.

A zero coefficient of `X`, stored or absent, contributes nothing even against
`NaN` or infinity in the factors. Nonzero coefficients always contribute;
zeros in `A` and `B` keep ordinary IEEE behavior. The result can therefore
differ from `A * Matrix(X) * B` on non-finite inputs.

Finite three-factor terms avoid spurious intermediate overflow or underflow,
without promising correct rounding. Terms are summed in storage order; a non-finite sum is
recomputed only when all contributing factors are finite. See
[three-factor products](@ref three-factor-products) for the precise rounding,
range-boundary and cancellation guarantees. Mixed and generic product methods
do not inherit that guarantee.
"""
function *(A::Symmetric{T, <:StridedMatrix{T}}, X::SparseIterate{T, 2},
           B::Symmetric{T, <:StridedMatrix{T}}) where {T<:Union{Float32, Float64}}
  _checkreal(X, A); _checkreal(X, B)
  m, n = size(X)
  size(A, 2) == m && size(B, 1) == n || throw(DimensionMismatch())
  C = zeros(T, m, n)
  a, ax, b = Vector{T}(undef, m), Vector{T}(undef, m), Vector{T}(undef, n)
  for k = 1:nactive(X)
    x = X.nzval[k]
    iszero(x) && continue
    ind = X.nzval2ind[k]
    _add_coefficient!(C, A, B, mod1(ind, m), cld(ind, m), x, a, ax, b)
  end
  _rescue_nonfinite!(C, A, X, B)
end

"""
    *(A::Symmetric, X::Union{SymmetricSparseIterate, SparseIterate{T, 2}}, B::Symmetric)

The product `A * X * B` for a sparse iterate `X`, restricted to the rows and
columns where `X` has nonzeros: O(p² k) for nonzeros in k rows, instead of the
O(p³) of dense products. This is the path of matrix iterates of element types other than
`Float32` and `Float64` (and, through `SymmetricSparseIterate`'s own method, of a symmetric iterate of such a
type); the guarantee of the direct methods does not cover it.
"""
function *(A::Symmetric{T}, X::SparseIterate{T, 2}, B::Symmetric{T}) where {T}
  _checkreal(X, A); _checkreal(X, B)
  m, n = size(X)
  size(A, 2) == m && size(B, 1) == n || throw(DimensionMismatch())
  posI, posJ = zeros(Int, m), zeros(Int, n)
  nI = nJ = 0
  for k = 1:nactive(X)
    iszero(X.nzval[k]) && continue
    i, j = mod1(X.nzval2ind[k], m), cld(X.nzval2ind[k], m)
    posI[i] == 0 && (posI[i] = 1; nI += 1)
    posJ[j] == 0 && (posJ[j] = 1; nJ += 1)
    2 * nI > m && 2 * nJ > n && return A * (Matrix(X) * B)
  end

  I, J = findall(!iszero, posI), findall(!iszero, posJ)
  posI[I] = 1:nI
  posJ[J] = 1:nJ
  XIJ = zeros(T, nI, nJ)
  for k = 1:nactive(X)
    iszero(X.nzval[k]) && continue
    i, j = mod1(X.nzval2ind[k], m), cld(X.nzval2ind[k], m)
    XIJ[posI[i], posJ[j]] = X.nzval[k]
  end
  # multiply in the cheaper order
  nI <= nJ ? A[:, I] * (XIJ * B[J, :]) : (A[:, I] * XIJ) * B[J, :]
end


