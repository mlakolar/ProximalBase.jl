"""
    SparseIterate([T = Float64,] n)
    SparseIterate([T = Float64,] n, m)
    SparseIterate(x::Union{SparseVector, SparseMatrixCSC, Array})

A vector or matrix that stores its nonzero entries in a compact list, so that
reading or writing any entry is O(1) and multiplying by it costs O(nnz).
Intended for coordinate-descent iterates that stay sparse.

Setting an entry to zero keeps it in the list; call `dropzeros!` to remove
stored zeros.
"""
mutable struct SparseIterate{T, N} <: AbstractArray{T, N}
    nzval::Vector{T}            # stored nonzero values (1:nnz)
    nzval2ind::Vector{Int}      # Mapping from nzval to indices
    ind2nzval::Array{Int, N}    # Mapping from indices to nzval
    nnz::Int                    # number of stored entries
end

"""
    SymmetricSparseIterate([T = Float64,] p)
    SymmetricSparseIterate(X::SparseMatrixCSC)

A sparse symmetric `p × p` matrix that stores only its lower triangle, with the
same O(1) access as [`SparseIterate`](@ref). `x[r, c]` and `x[c, r]` refer to
the same entry. Linear indices `1:p(p+1)/2` run over the lower triangle column
by column (see [`sub2indLowerTriangular`](@ref)): they are the coordinates that
[`cdprox!`](@ref) updates, and `length(x)` is `p(p+1)/2`. Generic functions that
expect the linear indices of a dense `p × p` matrix, such as `vec` and
`reshape`, give wrong results or errors; convert with `Matrix(x)` first.
"""
mutable struct SymmetricSparseIterate{T} <: AbstractMatrix{T}
    nzval::Vector{T}            # stored nonzero values (1:nnz)
    nzval2ind::Vector{Int}      # Mapping from nzval to indices
    ind2nzval::Vector{Int}      # Mapping from indices to nzval
    nnz::Int                    # number of stored entries
    p::Int                      # dimension of matrix
end


####################################
#
#  Storage shared by SparseIterate and SymmetricSparseIterate
#
####################################

# Both store their entries in the same way: nzval[1:nnz] are the stored values,
# nzval2ind[1:nnz] their linear indices, and ind2nzval[i] is the position of index i
# in nzval, 0 if i is not stored.
const _StoredIterate{T} = Union{SparseIterate{T}, SymmetricSparseIterate{T}}

SparseArrays.nnz(x::_StoredIterate) = x.nnz

"""
    numCoordinates(x)

Number of coordinates of `x` that [`cdprox!`](@ref) can update: all entries of
a [`SparseIterate`](@ref), the lower triangle of a
[`SymmetricSparseIterate`](@ref), and the groups of an [`AtomIterate`](@ref).
"""
function numCoordinates end

"""
    numActive(x)

Number of active coordinates of `x`: the stored entries of a
[`SparseIterate`](@ref) or [`SymmetricSparseIterate`](@ref), and the active
groups of an [`AtomIterate`](@ref). Solvers use them as the active set;
[`activeCoordinate`](@ref) lists them.
"""
function numActive end

"""
    activeCoordinate(x, i)

The `i`-th active coordinate of `x`, for `1 ≤ i ≤ numActive(x)`: an index of
`x` for a [`SparseIterate`](@ref) or [`SymmetricSparseIterate`](@ref), a group
for an [`AtomIterate`](@ref). Coordinates are listed in the order in which they
became active, except that `dropzeros!` moves the last ones into the places of
those it removes.
"""
function activeCoordinate end

"""
    isActive(x, k)

Whether coordinate `k` of `x` is active (see [`numActive`](@ref)).
"""
function isActive end

numCoordinates(x::_StoredIterate) = length(x.ind2nzval)
numActive(x::_StoredIterate) = x.nnz
Base.@propagate_inbounds function activeCoordinate(x::_StoredIterate, i::Int)
  @boundscheck 1 <= i <= x.nnz || throw(BoundsError(Base.OneTo(x.nnz), i))
  @inbounds x.nzval2ind[i]
end
isActive(x::_StoredIterate, k::Int) = x.ind2nzval[k] != 0

# entry i; for a SymmetricSparseIterate, i is a linear index of the lower triangle
function Base.getindex(x::_StoredIterate{T}, i::Int) where {T}
  checkbounds(Bool, x.ind2nzval, i) || throw(BoundsError(x, i))
  x.ind2nzval[i] == 0 ? zero(T) : x.nzval[x.ind2nzval[i]]
end

function Base.setindex!(x::_StoredIterate{T}, v, i::Int) where {T}
  checkbounds(Bool, x.ind2nzval, i) || throw(BoundsError(x, i))
  v = convert(T, v)
  if x.ind2nzval[i] != 0
    x.nzval[x.ind2nzval[i]] = v
  elseif !iszero(v)
    _store!(x, i, v)
  end
  x
end

# store index i, which is not stored yet, with the value v
function _store!(x::_StoredIterate, i::Int, v)
  x.nnz += 1
  x.nzval[x.nnz] = v
  x.nzval2ind[x.nnz] = i
  x.ind2nzval[i] = x.nnz
  x
end

# remove the stored zeros, moving the last stored entry into the place of each
function SparseArrays.dropzeros!(x::_StoredIterate)
  i = 1
  while i <= x.nnz
    if iszero(x.nzval[i])
      x.ind2nzval[x.nzval2ind[i]] = 0
      if i != x.nnz
        x.nzval[i] = x.nzval[x.nnz]
        x.ind2nzval[x.nzval2ind[x.nnz]] = i
        x.nzval2ind[i] = x.nzval2ind[x.nnz]
      end
      x.nnz -= 1
    else
      i += 1
    end
  end
  x
end

# Copy the stored entries of y into x, an iterate of the same size. When both are
# sparse, touch only their stored entries; otherwise copying whole arrays is faster.
function _copyto_stored!(x::_StoredIterate, y::_StoredIterate)
    if 4 * (x.nnz + y.nnz) < length(x.ind2nzval)
        @inbounds for k = 1:x.nnz
            x.ind2nzval[x.nzval2ind[k]] = 0
        end
        @inbounds for k = 1:y.nnz
            i = y.nzval2ind[k]
            x.nzval[k] = y.nzval[k]
            x.nzval2ind[k] = i
            x.ind2nzval[i] = k
        end
    else
        copyto!(x.nzval, y.nzval)
        copyto!(x.nzval2ind, y.nzval2ind)
        copyto!(x.ind2nzval, y.ind2nzval)
    end
    x.nnz = y.nnz
    x
end


####################################
#
#  Sparse Iterate
#
####################################

SparseIterate(n::Int) = SparseIterate(Float64, n)
SparseIterate(::Type{T}, n::Int) where {T} = SparseIterate{T, 1}(zeros(T, n), zeros(Int, n), zeros(Int, n), 0)
SparseIterate(n::Int, m::Int) = SparseIterate(Float64, n, m)
SparseIterate(::Type{T}, n::Int, m::Int) where {T} = SparseIterate{T, 2}(zeros(T, n*m), zeros(Int, n*m), zeros(Int, n, m), 0)

SparseArrays.issparse(x::SparseIterate) = true

Base.length(x::SparseIterate) = length(x.ind2nzval)
Base.size(x::SparseIterate) = size(x.ind2nzval)
Base.IndexStyle(::Type{<:SparseIterate}) = IndexLinear()

## similar
#
Base.similar(A::SparseIterate{T}) where {T} = SparseIterate(T, size(A)...)
Base.similar(A::SparseIterate, ::Type{S}) where {S} = SparseIterate(S, size(A)...)


### Conversion
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
  nzval = SparseArrays.nonzeros(x)
  rowval = SparseArrays.nonzeroinds(x)
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
    for k in 1:nnz(x)
        i = x.nzval2ind[k]
        v = x.nzval[k]
        r[i] = v
    end
    return r
end

function Base.Matrix(x::SparseIterate{Tv, 2}) where Tv
  n, m = size(x)
  A = zeros(Tv, n, m)
  for k=1:nnz(x)
    i = x.nzval2ind[k]
    v = x.nzval[k]
    A[i] = v
  end
  return A
end

# Array(x) and convert(Array, x) otherwise fall back to copying entry by entry
Base.Array(x::SparseIterate{T, 1}) where {T} = Vector(x)
Base.Array(x::SparseIterate{T, 2}) where {T} = Matrix(x)


"""
    activate!(x::SparseIterate, i)
    activate!(x::AtomIterate, k)

Store entry `i` of `x` as an explicit zero if it is not stored already, and
return `x`. Assigning a zero does not store an entry, so this is how a
coordinate joins the stored entries, for example the active set of an
active-set solver. `dropzeros!` removes the entry again while it is zero. For
an [`AtomIterate`](@ref), make group `k` active, keeping its entries.
"""
function activate!(x::SparseIterate{T}, i::Int) where {T}
  checkbounds(x, i)
  x.ind2nzval[i] == 0 && _store!(x, i, zero(T))
  x
end


function Base.copyto!(x::SparseIterate, y::SparseIterate)
    size(x) == size(y) || throw(DimensionMismatch())
    _copyto_stored!(x, y)
end


### show and friends

Base.show(io::IO, x::SparseIterate) = show(convert(IOContext, io), x)

function Base.show(io::IO, ::MIME"text/plain", x::SparseIterate{T, 1}) where T
    xnnz = nnz(x)
    print(io, length(x), "-element ", typeof(x), " with ", xnnz,
           " stored ", xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(io, x)
    end
end

function Base.show(io::IOContext, x::SparseIterate{T, 1}) where T
    n = length(x)
    xnnz = nnz(x)
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
    xnnz = nnz(S)
    print(io, size(S, 1), "×", size(S, 2), " ", typeof(S), " with ", xnnz, " stored ",
              xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        print(io, ":")
        show(io, S)
    end
end

function Base.show(io::IOContext, S::SparseIterate{T, 2}) where T
  if nnz(S) == 0
    return show(io, MIME("text/plain"), S)
  end

  limit::Bool = get(io, :limit, false)
  if limit
    rows = displaysize(io)[1]
    half_screen_rows = max(div(rows - 8, 2), 1)
  else
    half_screen_rows = typemax(Int)
  end
  pad = ndigits(maximum( size(S) ))
  sep = "\n  "
  if !haskey(io, :compact)
    io = IOContext(io, :compact => true)
  end
  ind = 0
  _ci = CartesianIndices(size(S))
  for ifull in eachindex(S)
    if S.ind2nzval[ifull] != 0
      ind += 1
      if ind < half_screen_rows || ind > nnz(S)-half_screen_rows
        row, col = Tuple(_ci[ifull])
        print(io, sep, '[', rpad(row, pad), ", ", lpad(col, pad), "]  =  ")
        show(io, S.nzval[S.ind2nzval[ifull]])
      elseif ind == half_screen_rows
        print(io, sep, '⋮')
      end
    end
  end
end


## multiplication

function *(A::StridedMatrix{Ta}, x::SparseIterate{Tx, 1}) where {Ta,Tx}
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Vector{Ty}(undef, m)
    mul!(y, A, x)
end

function LinearAlgebra.mul!(out::Vector, A::StridedMatrix, coef::SparseIterate{T, 1}) where {T}
  length(coef) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), length of coef is $(length(coef))"))
  length(out) == size(A, 1) || throw(DimensionMismatch("A has first dimension $(size(A,1)), length of out is $(length(out))"))

  fill!(out, zero(eltype(out)))
  @inbounds for inz = 1:nnz(coef)
    indColumn = coef.nzval2ind[inz]
    c = coef.nzval[inz]
    @simd for i = 1:size(A, 1)
      out[i] += c*A[i, indColumn]
    end
  end
  out
end

# A' equals transpose(A) for real A, so both take the sparse path
const TransposedStridedMatrix{T} = Union{Transpose{<:Any,<:StridedMatrix{T}}, Adjoint{<:Real,<:StridedMatrix{T}}}

function *(transA::TransposedStridedMatrix{Ta}, x::SparseIterate{Tx, 1}) where {Ta,Tx}
    A = transA.parent
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Vector{Ty}(undef, n)
    mul!(y, transpose(A), x)
end

function LinearAlgebra.mul!(out::AbstractVector, transA::TransposedStridedMatrix, coef::SparseIterate{T, 1}) where {T}
    A = transA.parent
    length(coef) == size(A, 1) || throw(DimensionMismatch("A has first dimension $(size(A,1)), length of coef is $(length(coef))"))
    length(out) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), length of out is $(length(out))"))

    @inbounds for j = 1:size(A, 2)
        s = zero(eltype(out))
        @simd for inz = 1:nnz(coef)
            s += A[coef.nzval2ind[inz], j] * coef.nzval[inz]
        end
        out[j] = s
    end
    return out
end

LinearAlgebra.dot(coef::SparseIterate{T, 1}, x::Vector{T}) where {T} = dot(x, coef)

function LinearAlgebra.dot(x::Vector{T}, coef::SparseIterate{T, 1}) where {T}
    v = zero(T)
    @inbounds @simd for icoef = 1:nnz(coef)
        v += x[coef.nzval2ind[icoef]]*coef.nzval[icoef]
    end
    v
end


####################################
#
#  Symmetric Sparse Iterate
#
####################################

SymmetricSparseIterate(n::Int) = SymmetricSparseIterate(Float64, n)
SymmetricSparseIterate(::Type{T}, n::Int) where {T} =
  (_t = div(n*(n+1), 2); SymmetricSparseIterate{T}(zeros(T, _t), zeros(Int, _t), zeros(Int, _t), 0, n))

Base.length(x::SymmetricSparseIterate) = div(x.p*(x.p+1), 2)
Base.size(x::SymmetricSparseIterate) = (x.p, x.p)

Base.IndexStyle(::Type{<:SymmetricSparseIterate}) = IndexCartesian()
function Base.getindex(x::SymmetricSparseIterate, r::Int, c::Int)
  checkbounds(x, r, c)
  linInd = r >= c ? sub2indLowerTriangular(x.p, r, c) : sub2indLowerTriangular(x.p, c, r)
  getindex(x, linInd)
end
function Base.setindex!(x::SymmetricSparseIterate, v, r::Int, c::Int)
  checkbounds(x, r, c)
  linInd = r >= c ? sub2indLowerTriangular(x.p, r, c) : sub2indLowerTriangular(x.p, c, r)
  setindex!(x, v, linInd)
end

function SymmetricSparseIterate(X::SparseMatrixCSC{T}) where {T}
  n, m = size(X)
  n == m || throw(ArgumentError("X needs to be square matrix"))
  out = SymmetricSparseIterate(T, n)

  I, J, V = findnz(X)
  @inbounds for i=1:length(I)
    if I[i] >= J[i]
      out[I[i], J[i]] = V[i]
    end
  end

  out
end

function Base.Matrix(S::SymmetricSparseIterate{Tv}) where Tv
  p = S.p
  A = zeros(Tv, p, p)
  for k=1:nnz(S)
    i = S.nzval2ind[k]
    v = S.nzval[k]
    r, c = ind2subLowerTriangular(p, i)
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

Base.similar(A::SymmetricSparseIterate{T}) where {T} = SymmetricSparseIterate(T, size(A, 1))
Base.similar(A::SymmetricSparseIterate, ::Type{S}) where {S} = SymmetricSparseIterate(S, size(A, 1))

function Base.copyto!(x::SymmetricSparseIterate, y::SymmetricSparseIterate)
    size(x) == size(y) || throw(DimensionMismatch())
    _copyto_stored!(x, y)
end

LinearAlgebra.mul!(C::AbstractMatrix{T}, A::SymmetricSparseIterate{T}, B::AbstractMatrix{T}) where T = _mul_symsparse!(C, A, B)
# resolves ambiguity with LinearAlgebra's mul!(::AbstractMatrix, ::AbstractMatrix, ::AbstractTriangular)
LinearAlgebra.mul!(C::AbstractMatrix{T}, A::SymmetricSparseIterate{T}, B::LinearAlgebra.AbstractTriangular{T}) where T = _mul_symsparse!(C, A, B)

function _mul_symsparse!(C::AbstractMatrix{T}, A::SymmetricSparseIterate{T}, B::AbstractMatrix{T}) where T
  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())
  size(B, 2) == size(C, 2) || throw(DimensionMismatch())

  fill!(C, zero(T))
  i = 1
  @inbounds while i <= A.nnz
    v = A.nzval[i]
    ind = A.nzval2ind[i]
    r, c = ind2subLowerTriangular(size(A, 1), ind)
    if r == c
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


# any element types: the generic fallback would read linear indices as those of a dense matrix
function LinearAlgebra.mul!(C::AbstractVector, A::SymmetricSparseIterate, B::AbstractVector)
  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())

  fill!(C, zero(eltype(C)))
  i = 1
  @inbounds while i <= A.nnz
    v = A.nzval[i]
    ind = A.nzval2ind[i]
    r, c = ind2subLowerTriangular(size(A, 1), ind)
    if r == c
        @inbounds C[r] += B[c] * v
    else
        @inbounds C[r] += B[c] * v
        @inbounds C[c] += B[r] * v
    end
    i += 1
  end
  C
end
