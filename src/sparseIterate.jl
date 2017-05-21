mutable struct SparseIterate{T} <: AbstractVector{T}
    nzval::Vector{T}          # nonzero values
    nzval2full::Vector{Int}   # Mapping from indices in nzval to full vector
    full2nzval::Vector{Int}   # Mapping from indices in full vector to indices in nzval
    nnz::Int
end

SparseIterate(n::Int) = SparseIterate{Float64}(zeros(Float64, n), zeros(Int, n), zeros(Int, n), 0)
SparseIterate(::T, n::Int) where {T} = SparseIterate{T}(zeros(T, n), zeros(Int, n), zeros(Int, n), 0)

Base.length(x::SparseIterate) = length(x.full2nzval)
Base.size(x::SparseIterate) = (length(x.full2nzval),)
Base.nnz(x::SparseIterate) = x.nnz
Base.getindex{T}(x::SparseIterate{T}, ipred::Int) =
    x.full2nzval[ipred] == 0 ? zero(T) : x.nzval[x.full2nzval[ipred]]
Base.iszero(x::SparseIterate) = x.nnz == 0



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
    for k = 1:xnnz
        if k < half_screen_rows || k > xnnz - half_screen_rows
            print(io, "  ", '[', rpad(x.nzval2full[k], pad), "]  =  ")
            show(io, x.nzval[k])
            k != xnnz && println(io)
        elseif k == half_screen_rows
            println(io, "   ", " "^pad, "   \u22ee")
        end
    end
end


##

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
  v
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
function Base.A_mul_B!{T}(out::Vector, A::Matrix, coef::SparseIterate{T})
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
SparseMatrixIterate(n::Int, symmetric=true) =
  SparseMatrixIterate{Float64}(zeros(Float64, n*n), zeros(Int, n*n), zeros(Int, n, n), 0, symmetric)


Base.length(x::SparseMatrixIterate) = length(x.full2nzval)
Base.size(x::SparseMatrixIterate) = size(x.full2nzval)
Base.nnz(x::SparseMatrixIterate) = x.nnz
Base.iszero(x::SparseMatrixIterate) = x.nnz == 0

Base.getindex(A::SparseMatrixIterate, I::Tuple{Integer,Integer}) = getindex(A, I[1], I[2])
function Base.getindex(A::SparseMatrixIterate, i0::Integer, i1::Integer) where T
    if !(1 <= i0 <= size(A, 1) && 1 <= i1 <= size(A, 2)); throw(BoundsError()); end
    inzval = A.full2nzval[i0, i1]
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
    k = 0
    sep = "\n  "
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for ind=1:nnz(S)
        if k < half_screen_rows || k > nnz(S)-half_screen_rows
            ifull = S.nzval2full[k]
            row, col = ind2sub(size(S), ifull)
            print(io, sep, '[', rpad(row, pad), ", ", lpad(col, pad), "]  =  ")
            if isassigned(S.nzval, Int(k))
                show(io, S.nzval[k])
            else
                print(io, Base.undef_ref_str)
            end
        elseif k == half_screen_rows
            print(io, sep, '\u22ee')
        end
        k += 1
    end
end


##

function Base.copy!(x::SparseMatrixIterate, y::SparseMatrixIterate)
    length(x) == length(y) || throw(DimensionMismatch())
    copy!(x.nzval, y.nzval)
    copy!(x.nzval2full, y.nzval2full)
    copy!(x.full2nzval, y.full2nzval)
    x.nnz = y.nnz
    x.symmetric = y.symmetric
    x
end

function Base.setindex!(x::SparseMatrixIterate{T}, v::T, i::Integer, j::Integer) where {T}
  if x.full2nzval[i, j] == 0
    if v != zero(T)
      x.nnz += 1
      x.nzval[x.nnz] = v
      x.nzval2full[x.nnz] = sub2ind(size(x), i, j)
      x.full2nzval[i, j] = x.nnz
    end
  else
    icoef = x.full2nzval[i, j]
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
