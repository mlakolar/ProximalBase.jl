mutable struct SparseIterate{T} <: AbstractVector{T}
    nzval::Vector{T}          # nonzero values
    nzval2full::Vector{Int}   # Mapping from indices in nzval to full vector
    full2nzval::Vector{Int}   # Mapping from indices in full vector to indices in nzval
    nnz::Int
end

SparseIterate(n::Int) = SparseIterate{Float64}(zeros(Float64, n), zeros(Int, n), zeros(Int, n), 0)
SparseIterate{T}(::T, n::Int) = SparseIterate{T}(zeros(T, n), zeros(Int, n), zeros(Int, n), 0)

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
