mutable struct SparseIterate{T} <: AbstractVector{T}
    nzval::Vector{T}          # nonzero values
    nzval2full::Vector{Int}   # Mapping from indices in nzval to full vector
    full2nzval::Vector{Int}   # Mapping from indices in full vector to indices in nzval
    nnz::Int
end

SparseIterate(n::Int) = SparseIterate{Float64}(zeros(Float64, n), zeros(Int, n), zeros(Int, n), 0)
SparseIterate(::Type{T}, n::Int) where {T} = SparseIterate{T}(zeros(T, n), zeros(Int, n), zeros(Int, n), 0)

Base.length(x::SparseIterate) = length(x.full2nzval)
Base.size(x::SparseIterate) = (length(x.full2nzval),)
Base.nnz(x::SparseIterate) = x.nnz
Base.getindex{T}(x::SparseIterate{T}, ipred::Int) =
    x.full2nzval[ipred] == 0 ? zero(T) : x.nzval[x.full2nzval[ipred]]
Base.iszero(x::SparseIterate) = x.nnz == 0

Base.IndexStyle(::Type{<:SparseIterate}) = IndexLinear()


function Base.convert(::Type{Vector}, x::SparseIterate{Tv}) where Tv
    n = length(x)
    n == 0 && return Vector{Tv}(0)
    r = zeros(Tv, n)
    for k in 1:nnz(x)
        i = x.nzval2full[k]
        v = x.nzval[k]
        r[i] = v
    end
    return r
end
Base.convert(::Type{Array}, x::SparseIterate) = convert(Vector, x)
Base.full(x::SparseIterate) = convert(Array, x)

### show and friends

function Base.show(io::IO, ::MIME"text/plain", x::SparseIterate)
    xnnz = nnz(x)
    print(io, length(x), "-element ", typeof(x), " with ", xnnz,
           " stored ", xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(io, x)
    end
end

Base.show(io::IO, x::SparseIterate) = show(convert(IOContext, io), x)
function Base.show(io::IOContext, x::SparseIterate)
    n = length(x)
    xnnz = nnz(x)
    if xnnz == 0
        return show(io, MIME("text/plain"), x)
    end
    limit::Bool = get(io, :limit, false)
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    pad = ndigits(n)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for k = 1:n
        if k < half_screen_rows || k > xnnz - half_screen_rows
          if x.full2nzval[k] != 0
            print(io, "  ", '[', rpad(k, pad), "]  =  ")
            show(io, x[k])
            k != n && println(io)
          end
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end


##

Base.similar(A::SparseIterate{T}) where {T} = SparseIterate(T, length(A))
Base.similar(A::SparseIterate, ::Type{S}) where {S} = SparseIterate(S, length(A))


function Base.copy!(x::SparseIterate, y::SparseIterate)
    length(x) == length(y) || throw(DimensionMismatch())
    copy!(x.nzval, y.nzval)
    copy!(x.nzval2full, y.nzval2full)
    copy!(x.full2nzval, y.full2nzval)
    x.nnz = y.nnz
    x
end

function Base.setindex!{T}(x::SparseIterate{T}, v::T, ipred::Int)
  if x.full2nzval[ipred] == 0
    if v != zero(T)
      x.nnz += 1
      x.nzval[x.nnz] = v
      x.nzval2full[x.nnz] = ipred
      x.full2nzval[ipred] = x.nnz
    end
  else
    icoef = x.full2nzval[ipred]
    x.nzval[icoef] = v
  end
  x
end

function Base.dropzeros!{T}(x::SparseIterate{T})
  i = 1
  while i <= x.nnz
    if x.nzval[i] == zero(T)
      x.full2nzval[x.nzval2full[i]] = 0
      if i != x.nnz
        x.nzval[i] = x.nzval[x.nnz]
        x.full2nzval[x.nzval2full[x.nnz]] = i
        x.nzval2full[i] = x.nzval2full[x.nnz]
      end
      x.nnz -= 1
      i -= 1
    end
    i += 1
  end
  x
end

## multiplication
function Base.A_mul_B!{T}(out::Vector{T}, A::AbstractMatrix{T}, coef::SparseIterate{T})
    fill!(out, zero(eltype(out)))
    @inbounds for icoef = 1:nnz(coef)
        ipred = coef.nzval2full[icoef]
        c = coef.nzval[icoef]
        @simd for i = 1:size(A, 1)
            out[i] += c*A[i, ipred]
        end
    end
    out
end

function Base.dot{T}(x::Vector{T}, coef::SparseIterate{T})
    v = 0.0
    @inbounds @simd for icoef = 1:nnz(coef)
        v += x[coef.nzval2full[icoef]]*coef.nzval[icoef]
    end
    v
end








####################################
#
#  Sparse Matrix Iterate
#
####################################



mutable struct SparseMatrixIterate{T} <: AbstractMatrix{T}
    nzval::Vector{T}          # nonzero values
    nzval2full::Vector{Int}   # Mapping from indices in nzval to full matrix
    full2nzval::Matrix{Int}   # Mapping from indices in full matrix to indices in nzval
    nnz::Int
    symmetric::Bool
end

SparseMatrixIterate(n::Int, m::Int) =
  SparseMatrixIterate{Float64}(zeros(Float64, n*m), zeros(Int, n*m), zeros(Int, n, m), 0, false)

SparseMatrixIterate(::Type{T}, n::Int, m::Int) where {T} =
  SparseMatrixIterate{T}(zeros(T, n*m), zeros(Int, n*m), zeros(Int, n, m), 0, false)

SparseMatrixIterate(n::Int, symmetric=true) =
  SparseMatrixIterate{Float64}(zeros(Float64, n*n), zeros(Int, n*n), zeros(Int, n, n), 0, symmetric)
SparseMatrixIterate(::Type{T}, n::Int, symmetric=true) where {T} =
  SparseMatrixIterate{T}(zeros(T, n*n), zeros(Int, n*n), zeros(Int, n, n), 0, symmetric)


Base.length(x::SparseMatrixIterate) = length(x.full2nzval)
Base.size(x::SparseMatrixIterate) = size(x.full2nzval)
Base.nnz(x::SparseMatrixIterate) = x.nnz
Base.iszero(x::SparseMatrixIterate) = x.nnz == 0

Base.IndexStyle(::Type{<:SparseMatrixIterate}) = IndexLinear()

function Base.getindex(A::SparseMatrixIterate{T}, i0::Int, i1::Int) where {T}
    inzval = A.full2nzval[i0, i1]
    inzval == 0 ? zero(T) : A.nzval[inzval]
end
function Base.getindex(A::SparseMatrixIterate{T}, i::Int) where {T}
    inzval = A.full2nzval[i]
    inzval == 0 ? zero(T) : A.nzval[inzval]
end


function Base.show(io::IO, ::MIME"text/plain", S::SparseMatrixIterate)
    xnnz = nnz(S)
    print(io, size(S, 1), "×", size(S, 2), " ", typeof(S), " with ", xnnz, " stored ",
              xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        print(io, ":")
        show(io, S)
    end
end

Base.show(io::IO, S::SparseMatrixIterate) = Base.show(convert(IOContext, io), S::SparseMatrixIterate)
function Base.show(io::IOContext, S::SparseMatrixIterate)
    if nnz(S) == 0
        return show(io, MIME("text/plain"), S)
    end

    limit::Bool = get(io, :limit, false)
    if limit
        rows = displaysize(io)[1]
        half_screen_rows = div(rows - 8, 2)
    else
        half_screen_rows = typemax(Int)
    end
    pad = ndigits(maximum( size(S) ))
    sep = "\n  "
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for ind=1:nnz(S)
        if ind < half_screen_rows || ind > nnz(S)-half_screen_rows
            ifull = S.nzval2full[ind]
            row, col = ind2sub(size(S), ifull)
            print(io, sep, '[', rpad(row, pad), ", ", lpad(col, pad), "]  =  ")
            if isassigned(S.nzval, Int(ind))
                show(io, S.nzval[ind])
            else
                print(io, Base.undef_ref_str)
            end
        elseif ind == half_screen_rows
            print(io, sep, '\u22ee')
        end
    end
end


##

function Base.convert(::Type{Matrix}, S::SparseMatrixIterate{Tv}) where Tv
  n, m = size(S)
  A = zeros(Tv, n, m)
  for k=1:nnz(S)
    i = S.nzval2full[k]
    v = S.nzval[k]
    A[i] = v
  end
  return A
end
Base.convert(::Type{Array}, S::SparseMatrixIterate) = convert(Matrix, S)
Base.full(S::SparseMatrixIterate) = convert(Array, S)


function Base.similar(A::SparseMatrixIterate{T}) where {T}
  n, m = size(A)
  if n == m
    return SparseMatrixIterate(T, n, A.symmetric)
  else
    return SparseMatrixIterate(T, n, m)
  end
end

function Base.similar(A::SparseMatrixIterate, ::Type{S}) where {S}
  n, m = size(A)
  if n == m
    return SparseMatrixIterate(S, n, A.symmetric)
  else
    return SparseMatrixIterate(S, n, m)
  end
end

function Base.copy!(x::SparseMatrixIterate, y::SparseMatrixIterate)
    length(x) == length(y) || throw(DimensionMismatch())
    copy!(x.nzval, y.nzval)
    copy!(x.nzval2full, y.nzval2full)
    copy!(x.full2nzval, y.full2nzval)
    x.nnz = y.nnz
    x.symmetric = y.symmetric
    x
end

Base.setindex!(x::SparseMatrixIterate{T}, v::T, i::Integer, j::Integer) where {T} =
  setindex!(x, v, sub2ind(size(x), i, j))

function Base.setindex!(x::SparseMatrixIterate{T}, v::T, i::Integer) where {T}
  if x.full2nzval[i] == 0
    if v != zero(T)
      x.nnz += 1
      x.nzval[x.nnz] = v
      x.nzval2full[x.nnz] = i
      x.full2nzval[i] = x.nnz
    end
  else
    icoef = x.full2nzval[i]
    x.nzval[icoef] = v
  end
  x
end

function Base.dropzeros!(x::SparseMatrixIterate{T}) where {T}
  i = 1
  while i <= x.nnz
    if x.nzval[i] == zero(T)
      x.full2nzval[x.nzval2full[i]] = 0
      if i != x.nnz
        x.nzval[i] = x.nzval[x.nnz]
        x.full2nzval[x.nzval2full[x.nnz]] = i
        x.nzval2full[i] = x.nzval2full[x.nnz]
      end
      x.nnz -= 1
      i -= 1
    end
    i += 1
  end
  x
end



####################################
#
#  Group Sparse Iterate
#
####################################


struct AtomIterate{T, N, V} <: AbstractArray{T, N}
  storage::Array{T, N}
  atoms::Vector{V}                                 # each atom is a view of the storage
end

function AtomIterate(n::Int, numAtom::Int)
  mod(n, numAtom) == 0 || throw(ArgumentError("numAtom must divide n"))
  storage = zeros(n)
  lenAtom = div(n, numAtom)
  b, e = 1, lenAtom
  a1 = view(storage, b:e)
  V = typeof(a1)
  atoms = Vector{V}(numAtom)
  atoms[1] = a1
  for i=2:numAtom
    b += lenAtom
    e += lenAtom
    atoms[i] = view(storage, b:e)
  end
  AtomIterate{Float64, 1, V}(storage, atoms)
end

# if byRow == true then each atom is a group of the form [b:e, :]
# otherwise each atom is a group of the form [:, b:e]
function AtomIterate(dims::Dims{2}, numAtom::Int, byRow::Bool)
  nr = dims[1]
  nc = dims[2]
  if byRow
    mod(nr, numAtom) == 0 || throw(ArgumentError("numAtom must divide dims[1]"))
    lenAtom = div(nr, numAtom)
  else
    mod(nc, numAtom) == 0 || throw(ArgumentError("numAtom must divide dims[1]"))
    lenAtom = div(nc, numAtom)
  end
  storage = zeros(nr, nc)
  b, e = 1, lenAtom
  if byRow
    a1 = view(storage, b:e, :)
  else
    a1 = view(storage, :, b:e)
  end
  V = typeof(a1)
  atoms = Vector{V}(numAtom)
  atoms[1] = a1
  for i=2:numAtom
    b += lenAtom
    e += lenAtom
    if byRow
      atoms[i] = view(storage, b:e, :)
    else
      atoms[i] = view(storage, :, b:e)
    end
  end
  AtomIterate{Float64, 2, V}(storage, atoms)
end


#

function Base.similar(a::AtomIterate{T, N, V}) where {T,N,V}
  storage = zeros(a.storage)
  numAtoms = length(a.atoms)
  atoms = Vector{V}(numAtoms)
  for i=1:numAtoms
    atoms[i] = SubArray(storage, a.atoms[i].indexes)
  end
  AtomIterate{T,N,V}(storage, atoms)
end

Base.copy(a::AtomIterate) = copy!(similar(a), a)
function Base.copy!(dest::AtomIterate, src::AtomIterate)
  copy!(dest.storage, src.storage)
  dest
end


#

Base.length(x::AtomIterate) = length(x.storage)
Base.size(x::AtomIterate) = size(x.storage)
Base.IndexStyle(::Type{<:AtomIterate}) = IndexLinear()

Base.getindex(x::AtomIterate, i::Int) = getindex(x.storage, i)
Base.getindex(x::AtomIterate, I...) = getindex(x.storage, I...)
Base.setindex!(x::AtomIterate, v, i::Int) = setindex!(x.storage, v, i)
Base.setindex!(x::AtomIterate, v, I...) = setindex!(x.storage, v, I...)
