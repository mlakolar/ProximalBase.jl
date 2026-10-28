"""
    SparseIterate([T = Float64,] n)
    SparseIterate([T = Float64,] n, m)
    SparseIterate(x::Union{SparseVector, SparseMatrixCSC, Array})

A vector or matrix that stores its nonzero entries in a compact list, so that
reading or writing any entry is O(1) and multiplying by it costs O(nnz).
Intended for coordinate-descent and active-set iterates that stay sparse. Its
coordinates (see [`ncoordinates`](@ref)) are its entries, in linear order.

Setting an entry to zero keeps it in the list; call `dropzeros!` to remove
stored zeros. The stored entries are the active coordinates
([`activecoordinates`](@ref)).
"""
mutable struct SparseIterate{T, N} <: AbstractArray{T, N}
    nzval::Vector{T}            # stored nonzero values (1:nnz)
    nzval2ind::Vector{Int}      # Mapping from nzval to indices
    ind2nzval::Array{Int, N}    # Mapping from indices to nzval
    nnz::Int                    # number of stored entries
end

"""
    SymmetricSparseIterate([T = Float64,] p)
    SymmetricSparseIterate(X::AbstractMatrix)

A sparse symmetric `p × p` matrix that stores only its lower triangle, with the
same O(1) access as [`SparseIterate`](@ref). It is an ordinary `p × p` matrix:
`x[r, c]` and `x[c, r]` are the same entry, and linear indices, `length(x) = p²`
and iteration follow the dense matrix, so that an off-diagonal entry is visited
twice.

Writing `x[r, c]` writes `x[c, r]` too. Broadcasting into `x`, `copyto!`, `fill!`,
`rmul!`, `lmul!`, `axpy!` and `axpby!` update each entry once, from the lower
triangle of a source that is not symmetric. Other functions that update the
entries of `x` one at a time, such as `map!` or `mul!` into `x`, would update an
off-diagonal entry twice, and those that build a result that is not symmetric in
`similar(x)`, such as `cumsum` or `circshift`, need `Matrix(x)`. `triu`, `tril`
and `sparse` return a `SparseMatrixCSC`.

Its coordinates are the `p(p+1)/2` entries of the lower triangle, numbered
column by column ([`tril_index`](@ref)): [`prox_coordinate!`](@ref) updates
them, [`ncoordinates`](@ref) counts them, [`activecoordinates`](@ref) lists the
stored ones and [`coordinates`](@ref) is a vector view of them.
"""
mutable struct SymmetricSparseIterate{T} <: AbstractMatrix{T}
    nzval::Vector{T}            # stored nonzero values (1:nnz)
    nzval2ind::Vector{Int}      # Mapping from nzval to coordinates
    ind2nzval::Vector{Int}      # Mapping from coordinates to nzval
    nnz::Int                    # number of stored entries
    p::Int                      # dimension of matrix
end

_stored_values(x::Union{SparseIterate, SymmetricSparseIterate}) = view(x.nzval, 1:x.nnz)

function _scaled_sum(x::SymmetricSparseIterate, s, f::F = identity) where {F}
  v = zero(s[1])
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    u = f(_scale_value(x.nzval[k], s))
    v += r == c ? u : 2u
  end
  v
end

function _scaled_dot(a, x::SparseIterate, s)
  _checkparam(a, x)
  v = zero(s[1])
  @inbounds @simd for k in 1:x.nnz
    v += float(_entry(a, x.nzval2ind[k])) * _scale_value(x.nzval[k], s)
  end
  v
end
function _scaled_dot(a, x::SymmetricSparseIterate, s)
  _checkparam(a, x)
  v = zero(s[1])
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    u = _scale_value(x.nzval[k], s)
    ar = float(_entry(a, r + (c - 1) * x.p))
    ac = r == c ? zero(ar) : float(_entry(a, c + (r - 1) * x.p))
    v += (ar + ac) * u
  end
  v
end


####################################
#
#  Storage shared by SparseIterate and SymmetricSparseIterate
#
####################################

# Both store their entries in the same way: nzval[1:nnz] are the stored values,
# nzval2ind[1:nnz] their coordinates, and ind2nzval[i] is the position of
# coordinate i in nzval, 0 if i is not stored.
const _StoredIterate{T} = Union{SparseIterate{T}, SymmetricSparseIterate{T}}

"""
    nnz(x::Union{SparseIterate, SymmetricSparseIterate})

The number of stored coordinates of `x`, including stored zeros. On a
[`SymmetricSparseIterate`](@ref), count the stored coordinates of the lower
triangle, so each off-diagonal pair counts once. Use [`dropzeros!`](@ref) to
remove stored zeros.
"""
SparseArrays.nnz(x::_StoredIterate) = x.nnz

"""
    ncoordinates(x)

The number of coordinates of the iterate `x` that [`prox_coordinate!`](@ref)
updates: the entries of a [`SparseIterate`](@ref), the entries of the lower
triangle of a [`SymmetricSparseIterate`](@ref), the groups of an
[`AtomIterate`](@ref).
"""
function ncoordinates end

"""
    nactive(x)

The number of active coordinates of `x`: the stored entries of a
[`SparseIterate`](@ref) or [`SymmetricSparseIterate`](@ref), the active groups
of an [`AtomIterate`](@ref). Active-set solvers use them as the active set.
"""
function nactive end

"""
    activecoordinates(x)

A vector of the active coordinates of `x` (see [`nactive`](@ref)), in the order
in which they became active, except that `dropzeros!` moves the last ones into
the places of those it removes. The vector is a view of the internal list: it
does not grow when coordinates become active, so call `activecoordinates` again
after changing the active set.
"""
function activecoordinates end

"""
    isactive(x, k)

Whether coordinate `k` of `x` is active (see [`nactive`](@ref)).
"""
function isactive end

"""
    coordinates(x)

A vector view of the coordinates of `x` (see [`ncoordinates`](@ref)): `x`
itself for a [`SparseIterate`](@ref), whose coordinates are its entries in
linear order, and the entries of the lower triangle for a
[`SymmetricSparseIterate`](@ref). Writing through the view writes `x`.
"""
function coordinates end

ncoordinates(x::_StoredIterate) = length(x.ind2nzval)
nactive(x::_StoredIterate) = x.nnz
activecoordinates(x::_StoredIterate) = view(x.nzval2ind, 1:x.nnz)
isactive(x::_StoredIterate, k::Integer) = x.ind2nzval[k] != 0

# the value of coordinate i
Base.@propagate_inbounds function _getstored(x::_StoredIterate{T}, i::Integer) where {T}
  @boundscheck checkbounds(x.ind2nzval, i)
  @inbounds j = x.ind2nzval[i]
  j == 0 ? zero(T) : @inbounds x.nzval[j]
end

# write coordinate i; a zero is stored only if i is stored already
Base.@propagate_inbounds function _setstored!(x::_StoredIterate{T}, v, i::Integer) where {T}
  @boundscheck checkbounds(x.ind2nzval, i)
  v = convert(T, v)
  @inbounds j = x.ind2nzval[i]
  if j != 0
    @inbounds x.nzval[j] = v
  elseif !iszero(v)
    _store!(x, i, v)
  end
  x
end

# store coordinate i, which is not stored yet, with the value v
function _store!(x::_StoredIterate, i::Integer, v)
  x.nnz += 1
  x.nzval[x.nnz] = v
  x.nzval2ind[x.nnz] = i
  x.ind2nzval[i] = x.nnz
  x
end

"""
    activate!(x::SparseIterate, i)
    activate!(x::SymmetricSparseIterate, k)
    activate!(x::AtomIterate, k)

Make coordinate `i` of `x` active, storing it as an explicit zero if it is not
stored already, and return `x`. Assigning a zero does not store an entry, so
this is how a coordinate joins the active set of an active-set solver;
`dropzeros!` removes it again while it is zero. For an [`AtomIterate`](@ref),
make group `k` active, keeping its entries.
"""
function activate!(x::_StoredIterate{T}, i::Integer) where {T}
  checkbounds(x.ind2nzval, i)
  x.ind2nzval[i] == 0 && _store!(x, i, zero(T))
  x
end

"""
    dropzeros!(x::Union{SparseIterate, SymmetricSparseIterate})

Remove stored zero coordinates from `x` and return `x`. On a
[`SymmetricSparseIterate`](@ref), these are coordinates of the lower triangle.
The last stored coordinate moves into the place of each removed one, so the
order of [`activecoordinates`](@ref) can change. The entries of `x` do not
change.
"""
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

# remove all stored entries
function _clear!(x::_StoredIterate)
  @inbounds for k = 1:x.nnz
    x.ind2nzval[x.nzval2ind[k]] = 0
  end
  x.nnz = 0
  x
end

# Multiplying by a number scales the stored entries, in O(nnz), as for a sparse
# matrix. The generic methods visit every index, among them (r, c) and (c, r) of a
# SymmetricSparseIterate, which share storage, and so would scale it twice.
function LinearAlgebra.rmul!(x::_StoredIterate, s::Number)
  @inbounds for k = 1:x.nnz
    x.nzval[k] *= s
  end
  x
end
function LinearAlgebra.lmul!(s::Number, x::_StoredIterate)
  @inbounds for k = 1:x.nnz
    x.nzval[k] = s * x.nzval[k]
  end
  x
end
function LinearAlgebra.rdiv!(x::_StoredIterate, s::Number)
  @inbounds for k = 1:x.nnz
    x.nzval[k] /= s
  end
  x
end
function LinearAlgebra.ldiv!(s::Number, x::_StoredIterate)
  @inbounds for k = 1:x.nnz
    x.nzval[k] = s \ x.nzval[k]
  end
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

Base.@propagate_inbounds Base.getindex(x::SparseIterate, i::Int) = _getstored(x, i)
Base.@propagate_inbounds Base.setindex!(x::SparseIterate, v, i::Int) = _setstored!(x, v, i)

coordinates(x::SparseIterate) = x

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


function Base.copyto!(x::SparseIterate, y::SparseIterate)
    size(x) == size(y) || throw(DimensionMismatch())
    _copyto_stored!(x, y)
end

# fill!(x, 0) removes the stored entries, as zeros are not stored
function Base.fill!(x::SparseIterate, v)
  if iszero(v)
    _clear!(x)
  else
    @inbounds for i in eachindex(x)
      x[i] = v
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
  _show_entries(io, nnz(S), (pos -> Tuple(CartesianIndices(size(S))[S.nzval2ind[pos]])),
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

function *(A::StridedMatrix{Ta}, x::SparseIterate{Tx, 1}) where {Ta,Tx}
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Vector{Ty}(undef, m)
    mul!(y, A, x)
end

LinearAlgebra.mul!(out::StridedVector, A::StridedMatrix, coef::SparseIterate{T, 1}) where {T} =
  mul!(out, A, coef, true, false)
function LinearAlgebra.mul!(out::StridedVector, A::StridedMatrix, coef::SparseIterate{T, 1}, α::Number, β::Number) where {T}
  length(coef) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), length of coef is $(length(coef))"))
  length(out) == size(A, 1) || throw(DimensionMismatch("A has first dimension $(size(A,1)), length of out is $(length(out))"))

  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, coef, α)
  @inbounds for inz = 1:nnz(coef)
    indColumn = coef.nzval2ind[inz]
    c = scale * coef.nzval[inz]
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
  size(coef, 1) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), coef has $(size(coef, 1)) rows"))
  size(out) == (size(A, 1), size(coef, 2)) || throw(DimensionMismatch("out has size $(size(out))"))
  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, coef, α)
  # Stored-entry traversal wins for few coefficients; column order reads A
  # sequentially and matches LinearAlgebra's generic loop for denser coefficients.
  # Measurements on x86-64 place the crossover between 2% and 5%, hence 1/32.
  if 32 * nnz(coef) <= length(coef)
    p = size(coef, 1)
    @inbounds for inz = 1:nnz(coef)
      j, k = mod1(coef.nzval2ind[inz], p), cld(coef.nzval2ind[inz], p)
      c = scale * coef.nzval[inz]
      @simd for i = 1:size(A, 1)
        out[i, k] += c * A[i, j]
      end
    end
  else
    @inbounds for k = 1:size(coef, 2), j = 1:size(coef, 1)
      pos = coef.ind2nzval[j, k]
      pos == 0 && continue
      c = scale * coef.nzval[pos]
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
# Preserve the common component precision without making a real scale complex.
_product_scale(A, x, α) = convert(promote_type(typeof(α), real(eltype(A)), real(eltype(x))), α)

function *(A::_StridedSymmetric, x::SparseIterate{T, 1}) where {T}
  mul!(Vector{promote_type(eltype(A), T)}(undef, size(A, 1)), A, x)
end
function *(A::_StridedSymmetric, x::SparseIterate{T, 2}) where {T}
  mul!(Matrix{promote_type(eltype(A), T)}(undef, size(A, 1), size(x, 2)), A, x)
end

LinearAlgebra.mul!(out::StridedVector, A::_StridedSymmetric, x::SparseIterate{<:Any, 1}) =
  mul!(out, A, x, true, false)
function LinearAlgebra.mul!(out::StridedVector, A::_StridedSymmetric, x::SparseIterate{<:Any, 1}, α::Number, β::Number)
  length(x) == size(A, 2) && length(out) == size(A, 1) || throw(DimensionMismatch())
  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, x, α)
  @inbounds for k in 1:x.nnz
    _symmetric_axpy!(out, A, x.nzval2ind[k], scale * x.nzval[k])
  end
  out
end

LinearAlgebra.mul!(out::StridedMatrix, A::_StridedSymmetric, x::SparseIterate{<:Any, 2}) =
  mul!(out, A, x, true, false)
function LinearAlgebra.mul!(out::StridedMatrix, A::_StridedSymmetric, x::SparseIterate{<:Any, 2}, α::Number, β::Number)
  size(x, 1) == size(A, 2) && size(out) == (size(A, 1), size(x, 2)) || throw(DimensionMismatch())
  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, x, α)
  p = size(x, 1)
  @inbounds for k in 1:x.nnz
    j, col = mod1(x.nzval2ind[k], p), cld(x.nzval2ind[k], p)
    _symmetric_axpy!(view(out, :, col), A, j, scale * x.nzval[k])
  end
  out
end

# A' equals transpose(A) for real A, so both take the sparse path
const TransposedStridedMatrix{T} = Union{Transpose{<:Any,<:StridedMatrix{T}}, Adjoint{<:Real,<:StridedMatrix{T}}}

function *(trans_a::TransposedStridedMatrix{Ta}, x::SparseIterate{Tx, 1}) where {Ta,Tx}
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
    A = trans_a.parent
    length(coef) == size(A, 1) || throw(DimensionMismatch("A has first dimension $(size(A,1)), length of coef is $(length(coef))"))
    length(out) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), length of out is $(length(out))"))

    iszero(α) && return _scale_product_output!(out, β)
    scale = _product_scale(A, coef, α)
    @inbounds for j = 1:size(A, 2)
        s = zero(eltype(out))
        @simd for inz = 1:nnz(coef)
            s += A[coef.nzval2ind[inz], j] * coef.nzval[inz]
        end
        out[j] = iszero(β) ? scale * s : scale * s + β * out[j]
    end
    return out
end

# dot(v, x) over the stored entries of x, for any strided v, such as a row or a
# column of a matrix: dot(view(A, i, :), x) is (A * x)[i] in O(nnz(x)). Other
# vectors take the generic method, as SparseArrays has its own for sparse vectors.
function LinearAlgebra.dot(v::StridedVector, coef::SparseIterate{<:Any, 1})
    length(v) == length(coef) || throw(DimensionMismatch())
    s = zero(promote_type(eltype(v), eltype(coef)))
    @inbounds @simd for icoef = 1:nnz(coef)
        s += dot(v[coef.nzval2ind[icoef]], coef.nzval[icoef])
    end
    s
end
LinearAlgebra.dot(coef::SparseIterate{<:Any, 1}, v::StridedVector) = conj(dot(v, coef))

function LinearAlgebra.dot(x::SparseIterate{<:Any, 1}, y::SparseIterate{<:Any, 1})
    length(x) == length(y) || throw(DimensionMismatch())
    nnz(x) > nnz(y) && return conj(dot(y, x))
    s = zero(promote_type(eltype(x), eltype(y)))
    @inbounds for k = 1:nnz(x)
        s += dot(x.nzval[k], y[x.nzval2ind[k]])
    end
    s
end

# a' X b and the entries of A X B for a sparse matrix X, in O(nnz(X))
function LinearAlgebra.dot(a::StridedVector, X::SparseIterate{<:Any, 2}, b::StridedVector)
  (length(a), length(b)) == size(X) || throw(DimensionMismatch())
  m = size(X, 1)
  s = zero(promote_type(eltype(a), eltype(X), eltype(b)))
  @inbounds for k = 1:nnz(X)
    i = X.nzval2ind[k]
    s += conj(a[mod1(i, m)]) * X.nzval[k] * b[cld(i, m)]
  end
  s
end

"""
    *(A::Symmetric, X::Union{SymmetricSparseIterate, SparseIterate{T, 2}}, B::Symmetric)

The product `A * X * B` for a sparse iterate `X`, restricted to the rows and
columns where `X` has nonzeros: O(p² k) for nonzeros in k rows, instead of the
O(p³) of dense products.
"""
function *(A::Symmetric{T}, X::SparseIterate{T, 2}, B::Symmetric{T}) where {T}
  m, n = size(X)
  size(A, 2) == m && size(B, 1) == n || throw(DimensionMismatch())
  posI, posJ = zeros(Int, m), zeros(Int, n)
  nI = nJ = 0
  for k = 1:nnz(X)
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
  for k = 1:nnz(X)
    iszero(X.nzval[k]) && continue
    i, j = mod1(X.nzval2ind[k], m), cld(X.nzval2ind[k], m)
    XIJ[posI[i], posJ[j]] = X.nzval[k]
  end
  # multiply in the cheaper order
  nI <= nJ ? A[:, I] * (XIJ * B[J, :]) : (A[:, I] * XIJ) * B[J, :]
end


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

Base.size(x::SymmetricSparseIterate) = (x.p, x.p)
Base.IndexStyle(::Type{<:SymmetricSparseIterate}) = IndexCartesian()

# the coordinate of entry (r, c)
@inline _coordinate(x::SymmetricSparseIterate, r::Integer, c::Integer) =
  r >= c ? tril_index(x.p, r, c) : tril_index(x.p, c, r)

Base.@propagate_inbounds function Base.getindex(x::SymmetricSparseIterate, r::Int, c::Int)
  @boundscheck checkbounds(x, r, c)
  @inbounds _getstored(x, _coordinate(x, r, c))
end
Base.@propagate_inbounds function Base.setindex!(x::SymmetricSparseIterate, v, r::Int, c::Int)
  @boundscheck checkbounds(x, r, c)
  @inbounds _setstored!(x, v, _coordinate(x, r, c))
end

# the coordinates of x, the entries of its lower triangle, as a vector
struct SymmetricCoordinates{T} <: AbstractVector{T}
  x::SymmetricSparseIterate{T}
end
Base.size(c::SymmetricCoordinates) = (length(c.x.ind2nzval),)
Base.IndexStyle(::Type{<:SymmetricCoordinates}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(c::SymmetricCoordinates, k::Int) = _getstored(c.x, k)
Base.@propagate_inbounds Base.setindex!(c::SymmetricCoordinates, v, k::Int) = (_setstored!(c.x, v, k); c)

coordinates(x::SymmetricSparseIterate) = SymmetricCoordinates(x)

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

# the lower triangle of a dense matrix
function SymmetricSparseIterate(X::AbstractMatrix{T}) where {T}
  p = checksquare(X)
  out = SymmetricSparseIterate(T, p)
  @inbounds for c = 1:p, r = c:p
    v = X[r, c]
    iszero(v) || _store!(out, tril_index(p, r, c), v)
  end
  out
end

function Base.Matrix(S::SymmetricSparseIterate{Tv}) where Tv
  p = S.p
  A = zeros(Tv, p, p)
  for k=1:nnz(S)
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

Base.similar(A::SymmetricSparseIterate{T}) where {T} = SymmetricSparseIterate(T, size(A, 1))
Base.similar(A::SymmetricSparseIterate, ::Type{S}) where {S} = SymmetricSparseIterate(S, size(A, 1))

function Base.copyto!(x::SymmetricSparseIterate, y::SymmetricSparseIterate)
    size(x) == size(y) || throw(DimensionMismatch())
    _copyto_stored!(x, y)
end

# From a matrix source, its lower triangle.
Base.copyto!(x::SymmetricSparseIterate, y::AbstractMatrix) = _copyto_lower!(x, y)
# Resolve SparseArrays' copyto!(::AbstractMatrix, ::AbstractSparseMatrixCSC).
Base.copyto!(x::SymmetricSparseIterate, y::SparseArrays.AbstractSparseMatrixCSC) = _copyto_lower!(x, y)
# SparseArrays also specializes matrix views and transposed sparse sources.
@static if isdefined(SparseArrays, :SparseMatrixCSCOrView)
  Base.copyto!(x::SymmetricSparseIterate, y::SparseArrays.SparseMatrixCSCOrView) = _copyto_lower!(x, y)
end
Base.copyto!(x::SymmetricSparseIterate, y::Union{Adjoint{<:Any,<:SparseArrays.AbstractSparseMatrixCSC},
                                               Transpose{<:Any,<:SparseArrays.AbstractSparseMatrixCSC}}) = _copyto_lower!(x, y)
@static if isdefined(SparseArrays, :AdjOrTransSparseVectorOrView)
  Base.copyto!(x::SymmetricSparseIterate, y::SparseArrays.AdjOrTransSparseVectorOrView) = _copyto_lower!(x, y)
end
# CHOLMOD defines both an element-type-matched matrix method and an untyped
# array method for its dense matrix wrapper; preserve the same triangle rule.
Base.copyto!(x::SymmetricSparseIterate, y::SparseArrays.CHOLMOD.Dense) = _copyto_lower!(x, y)
Base.copyto!(x::SymmetricSparseIterate{T}, y::SparseArrays.CHOLMOD.Dense{T}) where
    {T<:SparseArrays.CHOLMOD.VTypes} = _copyto_lower!(x, y)
function _copyto_lower!(x::SymmetricSparseIterate, y::AbstractMatrix)
  size(x) == size(y) || throw(DimensionMismatch())
  p = x.p
  @inbounds for c = 1:p, r = c:p
    _setstored!(x, y[r, c], tril_index(p, r, c))
  end
  x
end

# Broadcasting into x computes each entry of the lower triangle once: visiting
# both (r, c) and (c, r) would cost twice as much and, as they share storage,
# apply an in-place update such as x .= 2 .* x twice.
function Base.copyto!(x::SymmetricSparseIterate, bc::Broadcast.Broadcasted{Nothing})
  axes(x) == axes(bc) || throw(DimensionMismatch("destination axes $(axes(x)) do not match $(axes(bc))"))
  bc′ = Broadcast.preprocess(x, bc)
  p = x.p
  @inbounds for c = 1:p, r = c:p
    _setstored!(x, bc′[CartesianIndex(r, c)], tril_index(p, r, c))
  end
  x
end

function Base.fill!(x::SymmetricSparseIterate, v)
  if iszero(v)
    _clear!(x)
  else
    c = coordinates(x)
    @inbounds for k in eachindex(c)
      c[k] = v
    end
  end
  x
end

# axpy! and axpby! update each entry once, as broadcasting does, from the lower
# triangle of x when x is not symmetric
function LinearAlgebra.axpy!(α::Number, x::AbstractArray, y::SymmetricSparseIterate)
  x′ = _reshape_like(x, y)
  iszero(α) && return y
  y .+= α .* x′
end
function LinearAlgebra.axpby!(α::Number, x::AbstractArray, β::Number, y::SymmetricSparseIterate)
  x′ = _reshape_like(x, y)
  iszero(α) && isone(β) && return y
  y .= α .* x′ .+ β .* y
end

# x, which has as many entries as y, in the shape of y
_reshape_like(x::AbstractArray, y::SymmetricSparseIterate) =
  size(x) == size(y) ? x :
  length(x) == length(y) ? reshape(x, size(y)) :
  throw(DimensionMismatch(lazy"x has length $(length(x)), but y has length $(length(y))"))

# The stored entries as a SparseMatrixCSC, in O(nnz): each nonzero (r, c) of the
# lower triangle gives the entries (r, c) and (c, r) for which keep holds. The
# generic sparse visits all p² entries, and the generic triu and tril work in
# similar(x), which, being symmetric, cannot hold a triangle.
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
SparseArrays.sparse(x::SymmetricSparseIterate) = _sparse(x, (i, j) -> true)
LinearAlgebra.triu(x::SymmetricSparseIterate, k::Integer = 0) = _sparse(x, (i, j) -> j - i >= k)
LinearAlgebra.tril(x::SymmetricSparseIterate, k::Integer = 0) = _sparse(x, (i, j) -> j - i <= k)

function Base.show(io::IO, ::MIME"text/plain", S::SymmetricSparseIterate)
  xnnz = nnz(S)
  print(io, S.p, "×", S.p, " ", typeof(S), " with ", xnnz, " stored ",
        xnnz == 1 ? "entry" : "entries", " in the lower triangle")
  xnnz == 0 && return
  print(io, ":")
  _show_entries(io, xnnz, pos -> tril_cartesian(S.p, S.nzval2ind[pos]), pos -> S.nzval[pos], S.p)
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
    r, c = tril_cartesian(size(A, 1), ind)
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


# any element types: the generic fallback would loop over all p² entries
function LinearAlgebra.mul!(C::AbstractVector, A::SymmetricSparseIterate, B::AbstractVector)
  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())

  fill!(C, zero(eltype(C)))
  i = 1
  @inbounds while i <= A.nnz
    v = A.nzval[i]
    ind = A.nzval2ind[i]
    r, c = tril_cartesian(size(A, 1), ind)
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

# a' X b over the stored entries of X: the entry (A X B)[r, c] of symmetric A, B
# is dot(view(A, :, r), X, view(B, :, c))
function LinearAlgebra.dot(a::StridedVector, X::SymmetricSparseIterate, b::StridedVector)
  length(a) == X.p == length(b) || throw(DimensionMismatch())
  s = zero(promote_type(eltype(a), eltype(X), eltype(b)))
  @inbounds for k = 1:X.nnz
    r, c = tril_cartesian(X.p, X.nzval2ind[k])
    v = X.nzval[k]
    s += r == c ? conj(a[r]) * v * b[c] : (conj(a[r]) * b[c] + conj(a[c]) * b[r]) * v
  end
  s
end

# the trace inner product tr(A'X) over the stored entries of X
function LinearAlgebra.dot(A::StridedMatrix, X::SymmetricSparseIterate)
  size(A) == size(X) || throw(DimensionMismatch())
  s = zero(promote_type(eltype(A), eltype(X)))
  @inbounds for k = 1:X.nnz
    r, c = tril_cartesian(X.p, X.nzval2ind[k])
    v = X.nzval[k]
    s += r == c ? dot(A[r, c], v) : dot(A[r, c] + A[c, r], v)
  end
  s
end
LinearAlgebra.dot(X::SymmetricSparseIterate, A::StridedMatrix) = conj(dot(A, X))

function *(A::Symmetric{T}, X::SymmetricSparseIterate{T}, B::Symmetric{T}) where {T}
  # Dense BLAS products cost O(p³) however sparse X is. Restricted to the rows K where
  # X has nonzeros, A[:, K] * X[K, K] * B[K, :] costs O(p² |K|). Once X has nonzeros
  # in more than half of the rows, the dense products are as fast.
  p = X.p
  size(A, 2) == p == size(B, 1) || throw(DimensionMismatch())
  pos = zeros(Int, p)
  nK = 0
  for k = 1:nnz(X)
    iszero(X.nzval[k]) && continue
    for i in tril_cartesian(p, X.nzval2ind[k])
      pos[i] == 0 && (pos[i] = 1; nK += 1)
    end
    2 * nK > p && return A * (Matrix(X) * B)
  end

  K = findall(!iszero, pos)
  pos[K] = 1:nK                 # position of each row within K
  XK = zeros(T, nK, nK)
  for k = 1:nnz(X)
    iszero(X.nzval[k]) && continue
    r, c = tril_cartesian(p, X.nzval2ind[k])
    XK[pos[r], pos[c]] = XK[pos[c], pos[r]] = X.nzval[k]
  end
  A[:, K] * (XK * B[K, :])
end
