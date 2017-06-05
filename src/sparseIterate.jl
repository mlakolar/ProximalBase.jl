mutable struct SparseIterate{T, N} <: AbstractArray{T, N}
    nzval::Vector{T}            # stored nonzero values (1:nnz)
    nzval2ind::Vector{Int}      # Mapping from nzval to indices
    ind2nzval::Array{Int, N}    # Mapping from indices to nzval
    nnz::Int                    # number of stored zeros
end

SparseIterate(n::Int) = SparseIterate(Float64, n)
SparseIterate(::Type{T}, n::Int) where {T} = SparseIterate{T, 1}(zeros(T, n), zeros(Int, n), zeros(Int, n), 0)
SparseIterate(n::Int, m::Int) = SparseIterate(Float64, n, m)
SparseIterate(::Type{T}, n::Int, m::Int) where {T} = SparseIterate{T, 2}(zeros(T, n*m), zeros(Int, n*m), zeros(Int, n, m), 0)

Base.length(x::SparseIterate) = length(x.ind2nzval)
Base.size(x::SparseIterate) = size(x.ind2nzval)
Base.nnz(x::SparseIterate) = x.nnz
numCoordinates(x::SparseIterate) = length(x.ind2nzval)

Base.IndexStyle(::Type{<:SparseIterate}) = IndexLinear()

Base.getindex{T}(x::SparseIterate{T}, i::Int) = (checkbounds(x, i); x.ind2nzval[i] == 0 ? zero(T) : x.nzval[x.ind2nzval[i]])
function Base.setindex!{T}(x::SparseIterate{T}, v::T, i::Int)
  checkbounds(x, i)
  if x.ind2nzval[i] == 0
    if v != zero(T)
      x.nnz += 1
      x.nzval[x.nnz] = v
      x.nzval2ind[x.nnz] = i
      x.ind2nzval[i] = x.nnz
    end
  else
    x.nzval[x.ind2nzval[i]] = v
  end
  x
end


function Base.convert(::Type{SparseIterate{T, 2}}, x::SparseMatrixCSC{T}) where {T}
  n, m = size(x)
  out = SparseIterate(T, n, m)

  I, J, V = findnz(x)
  for i=1:length(I)
    @inbounds out[I[i], J[i]] = V[i]
  end

  out
end
function Base.convert(::Type{SparseIterate{T, 1}}, x::SparseVector{T}) where {T}
  p = length(x)
  out = SparseIterate(T, p)
  nzval = SparseArrays.nonzeros(x)
  rowval = SparseArrays.nonzeroinds(x)
  for i=1:length(nzval)
    out[rowval[i]] = nzval[i]
  end
  out
end
Base.convert(::Type{SparseIterate}, x::SparseMatrixCSC{T}) where {T} = convert(SparseIterate{T, 2}, x)
Base.convert(::Type{SparseIterate}, x::SparseVector{T}) where {T} = convert(SparseIterate{T, 1}, x)



function Base.convert(::Type{Vector}, x::SparseIterate{Tv, 1}) where Tv
    n = length(x)
    n == 0 && return Vector{Tv}(0)
    r = zeros(Tv, n)
    for k in 1:nnz(x)
        i = x.nzval2ind[k]
        v = x.nzval[k]
        r[i] = v
    end
    return r
end
function Base.convert(::Type{Matrix}, S::SparseIterate{Tv, 2}) where Tv
  n, m = size(S)
  A = zeros(Tv, n, m)
  for k=1:nnz(S)
    i = S.nzval2ind[k]
    v = S.nzval[k]
    A[i] = v
  end
  return A
end
Base.convert(::Type{Array}, x::SparseIterate{T, 1}) where {T} = convert(Vector, x)
Base.convert(::Type{Array}, x::SparseIterate{T, 2}) where {T} = convert(Matrix, x)
Base.full(x::SparseIterate) = convert(Array, x)


Base.similar(A::SparseIterate{T}) where {T} = SparseIterate(T, size(A)...)
Base.similar(A::SparseIterate, ::Type{S}) where {S} = SparseIterate(S, size(A)...)

function Base.copy!(x::SparseIterate, y::SparseIterate)
    size(x) == size(y) || throw(DimensionMismatch())
    copy!(x.nzval, y.nzval)
    copy!(x.nzval2ind, y.nzval2ind)
    copy!(x.ind2nzval, y.ind2nzval)
    x.nnz = y.nnz
    x
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

function Base.show(io::IO, ::MIME"text/plain", S::SparseIterate{T,2}) where T
    xnnz = nnz(S)
    print(io, size(S, 1), "×", size(S, 2), " ", typeof(S), " with ", xnnz, " stored ",
              xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        print(io, ":")
        show(io, S)
    end
end


function Base.show(io::IOContext, x::SparseIterate{T, 1}) where T
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
          if x.ind2nzval[k] != 0
            print(io, "  ", '[', rpad(k, pad), "]  =  ")
            show(io, x[k])
            k != n && println(io)
          end
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end

function Base.show(io::IOContext, S::SparseIterate{T, 2}) where T
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
  ind = 0
  for ifull in eachindex(S)
    if S.ind2nzval[ifull] != 0
      ind += 1
      if ind < half_screen_rows || ind > nnz(S)-half_screen_rows
        row, col = ind2sub(size(S), ifull)
        print(io, sep, '[', rpad(row, pad), ", ", lpad(col, pad), "]  =  ")
        if isassigned(S.nzval, S.ind2nzval[ifull])
          show(io, S.nzval[S.ind2nzval[ifull]])
        else
          print(io, Base.undef_ref_str)
        end
      elseif ind == half_screen_rows
        print(io, sep, '\u22ee')
      end
    end
  end
end


##


function Base.dropzeros!(x::SparseIterate{T}) where T
  i = 1
  while i <= x.nnz
    if x.nzval[i] == zero(T)
      x.ind2nzval[x.nzval2ind[i]] = 0
      if i != x.nnz
        x.nzval[i] = x.nzval[x.nnz]
        x.ind2nzval[x.nzval2ind[x.nnz]] = i
        x.nzval2ind[i] = x.nzval2ind[x.nnz]
      end
      x.nnz -= 1
      i -= 1
    end
    i += 1
  end
  x
end


## multiplication
function Base.A_mul_B!{T}(out::Vector{T}, A::AbstractMatrix{T}, coef::SparseIterate{T, 1})
    fill!(out, zero(eltype(out)))
    @inbounds for icoef = 1:nnz(coef)
        ipred = coef.nzval2ind[icoef]
        c = coef.nzval[icoef]
        @simd for i = 1:size(A, 1)
            out[i] += c*A[i, ipred]
        end
    end
    out
end

Base.dot(coef::SparseIterate{T, 1}, x::Vector{T}) where {T} = dot(x, coef)
function Base.dot{T}(x::Vector{T}, coef::SparseIterate{T, 1})
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

# data are stored in the lower triangle
struct SymmetricSparseIterate{T} <: AbstractMatrix{T}
  data::SparseIterate{T,2}
end

SymmetricSparseIterate(n::Int) = SymmetricSparseIterate(Float64, n)
SymmetricSparseIterate(::Type{T}, n::Int) where {T} = SymmetricSparseIterate{T}(SparseIterate(T, n, n))

Base.length(x::SymmetricSparseIterate) = length(x.data)
Base.size(x::SymmetricSparseIterate) = size(x.data)
Base.nnz(x::SymmetricSparseIterate) = nnz(x.data)

Base.IndexStyle(::Type{<:SymmetricSparseIterate}) = IndexCartesian()
Base.getindex(x::SymmetricSparseIterate, r::Int, c::Int) = r >= c ? x.data[r, c] : x.data[c, r]
function Base.setindex!(x::SymmetricSparseIterate{T}, v::T, r::Int, c::Int) where {T}
  if r >= c
    setindex!(x.data, v, r, c)
  else
    setindex!(x.data, v, c, r)
  end
  x
end

function Base.convert(::Type{SymmetricSparseIterate{T}}, X::SparseMatrixCSC{T}) where {T}
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
Base.convert(::Type{SymmetricSparseIterate}, x::SparseMatrixCSC{T}) where {T} = convert(SymmetricSparseIterate{T}, x)

function Base.convert(::Type{Matrix}, S::SymmetricSparseIterate{Tv}) where Tv
  n, m = size(S)
  A = zeros(Tv, n, m)
  for k=1:nnz(S)
    i = S.nzval2ind[k]
    r, c = ind2sub(S, i)
    v = S.nzval[k]
    A[r, c] = v
  end
  return A
end
Base.convert(::Type{Array}, x::SymmetricSparseIterate) = convert(Matrix, x)
Base.full(x::SymmetricSparseIterate) = convert(Array, x)

Base.similar(A::SymmetricSparseIterate{T}) where {T} = SymmetricSparseIterate(T, size(A, 1))
Base.similar(A::SymmetricSparseIterate, ::Type{S}) where {S} = SymmetricSparseIterate(S, size(A, 1))

function Base.copy!(x::SymmetricSparseIterate, y::SymmetricSparseIterate)
    copy!(x.data, y.data)
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
