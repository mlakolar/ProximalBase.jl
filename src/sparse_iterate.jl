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
  function SparseIterate{T, N}(nzval, nzval2ind, ind2nzval, nnz) where {T, N}
    T <: Real || _real_error(:SparseIterate)
    new{T, N}(nzval, nzval2ind, ind2nzval, nnz)
  end
end
SparseIterate(nzval::Vector{T}, nzval2ind::Vector{Int}, ind2nzval::Array{Int, N}, nnz::Int) where {T, N} =
  SparseIterate{T, N}(nzval, nzval2ind, ind2nzval, nnz)

"""
    SymmetricSparseIterate([T = Float64,] p)
    SymmetricSparseIterate(X::AbstractMatrix)

A sparse symmetric `p × p` matrix that stores only its lower triangle, with the
same O(1) access as [`SparseIterate`](@ref). It is an ordinary `p × p` matrix:
`x[r, c]` and `x[c, r]` are the same entry, and linear indices, `length(x) = p²`
and iteration follow the dense matrix, so that an off-diagonal entry is visited
twice.

Read and write entries through the lower triangle, `r ≥ c`, for example with
`for c in 1:p, r in c:p`. The position `x[c, r]` is the same entry as
`x[r, c]`, so writing either position writes both. Code that writes every
position writes each off-diagonal entry twice, and the second write wins.

Whole-value writes are checked: the constructor, `copyto!`, `copy!`,
broadcasting into `x`, `map!`, `axpy!`, `axpby!`, `transpose!`, `adjoint!`,
`permutedims!` and `mul!`. The checks also cover indexed array assignment of
every position in order (`x[:, :] = A`, `x[:] = v`), and broadcasting and
`fill!` into full, in-order views and reshapes (`x[:, :] .= A`, `vec(x) .= v`).
The covered wrappers are views whose parent is `x`, reshapes whose parent is
`x`, and views of those reshapes. Use `:` or integer vectors matching each
axis in order for a full two-index assignment or a full matrix view. A full
single-index assignment accepts `:`, the integers `1:length(x)` in order, or
`CartesianIndices(x)` with its shape.
`copyto!(x, Q)` from an orthogonal factor is checked too.

Matrix values must have the axes of `x` and be exactly symmetric: entries
`(r, c)` and `(c, r)` are equal, with `NaN` equal to `NaN`, compared before
conversion to the element type of `x`. Linear full writes reshape their values
in column-major order for this check; `x[:] = v` requires `p²` values.
If the axes differ or symmetry fails, the checked writers throw a
`DimensionMismatch` or an `ArgumentError` before writing anything. To store
one triangle of a matrix `A` that is not symmetric, write `Symmetric(A, :L)`
or `Symmetric(A, :U)`. These functions write each entry of the lower triangle
once. A broadcast whose arguments do not show that its value is symmetric is
first evaluated at both positions of each off-diagonal entry, so a function
with side effects runs more often and may give a value that is not symmetric.

With two array operands, `mul!` into `x` checks the unscaled product for exact
symmetry in a buffer promoted from the destination and operand types, applies
scalars and conversion at the final write, and leaves the product out when
α = 0, as LinearAlgebra does. For `mul!` by a number, the array operand must
have the axes of `x`. The seven-argument `mul!(x, tA, tB, A, B, α, β)`,
when available, uses the same checks with flags `'N'`, `'T'` and `'C'` for
unchanged, transposed and adjoint operands; other flags throw before writing.
`fillband!(x, v, l, u)` checks that `v` is real and writes a symmetric band
(`l == -u`) through the lower triangle once. An empty band (`l > u`) returns
`x` unchanged.

Indexed array assignment, broadcasting and `fill!` into part of `x`, such as
a row, a column, a block or a partial mask, throw an `ArgumentError` before
writing. Logical-mask indexed assignment is rejected even for an all-true
mask. `copyto!` with offsets or index ranges (including transpose flags),
`copy_transpose!` and `copy_adjoint!` with index ranges, `copyto!` from a tuple, a
generator or a number, and the indexed `axpy!` also throw before writing.
Out-of-bounds indices still throw a `BoundsError`.

The position-wise writers `reverse!`, `sort!`, `accumulate!`, `kron!` and
`copytrito!` (when available) throw an `ArgumentError` before writing.
This also rejects `reverse(x)` with or without `dims`, `cumsum!`, `cumprod!`,
and allocating `cumsum`, `cumprod` and `accumulate` with `dims`.
`circshift!` and `circshift` of a nonempty `x` by a nonempty shift throw too.
`triu!(x, k)` and `tril!(x, k)` return `x` unchanged only when every position
is kept: `k ≤ 1 - p` for `triu!`, `k ≥ p - 1` for `tril!`, or any `k` when
`p = 0`. Otherwise they throw an `ArgumentError` before writing.

`lmul!`, `rmul!`, `ldiv!` and `rdiv!` by a matrix, a factorization, an
orthogonal factor or a rotation throw an `ArgumentError` before writing, as
does `ldiv!(x, A, B)` for such an operator `A`. Reductions into `x` (`sum!`,
`prod!`, `maximum!`, `minimum!`, `count!`, `all!`, `any!`, `extrema!`,
`findmax!` and `findmin!` with `x` as either destination), in-place
factorizations (`lu!`, `qr!`, `cholesky!` and `bunchkaufman!`), `get!` and
`sortperm!` also throw before writing. So do `fillband!` with a nonempty band
that is not symmetric, and `read!`, `readbytes!` and `hex2bytes!`, which do
not consume their input. Use `Matrix(x)` for these operations, or compute
into a dense matrix and `copyto!` a symmetric result.

`fill!`, and `rmul!`, `lmul!`, `rdiv!` and `ldiv!` by a number or a
`UniformScaling`, keep `x` symmetric. `replace` and `replace!` visit each
entry of the lower triangle once, and their `count` counts those entries,
column by column. `triu`, `tril` and `sparse` return a `SparseMatrixCSC`.
`sort(x; dims)` and `accumulate(op, x)` without `dims` keep their dense
results. Allocating `lu`, `qr`, `cholesky` and `bunchkaufman` use dense copies.

Other functions that write an arbitrary array position by position are not
checked; apply them to `Matrix(x)`. Scalar-write loops over every position
stay unchecked. Other writes through a view or reshape, such as `copyto!`
into a view and scalar writes through a view, stay unchecked too. The checks
do not extend to other nested wrappers, such as `vec(view(x, :, :))`,
including broadcasts and fills through them.
Two-argument `copyto!(x, itr)` from other non-array iterators is unchecked.
A Base function can write part of `x` before reaching a check. `rand!` and
`randn!` give a symmetric matrix but draw each off-diagonal entry twice.
`shuffle!` and `shuffle` give wrong values; use `Matrix(x)` instead.

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
  function SymmetricSparseIterate{T}(nzval, nzval2ind, ind2nzval, nnz, p) where {T}
    T <: Real || _real_error(:SymmetricSparseIterate)
    new{T}(nzval, nzval2ind, ind2nzval, nnz, p)
  end
end
SymmetricSparseIterate(nzval::Vector{T}, nzval2ind::Vector{Int}, ind2nzval::Vector{Int}, nnz::Int, p::Int) where {T} =
  SymmetricSparseIterate{T}(nzval, nzval2ind, ind2nzval, nnz, p)

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

function _split_dot(a::AbstractArray, x::SparseIterate, sa, sx)
  _checksize(a, x)
  R = typeof(sx[1])
  t = floatmax(R) / (2 * max(x.nnz, 1))
  big = eb = small = es = zero(R)
  @inbounds for k in 1:x.nnz
    big, eb, small, es = _split_add(big, eb, small, es, a[x.nzval2ind[k]], x.nzval[k], sa, sx, t)
  end
  big + eb, small + es
end
# each off-diagonal product separately: a[r, c] + a[c, r] can overflow on its own
function _split_dot(a::AbstractArray, x::SymmetricSparseIterate, sa, sx)
  _checksize(a, x)
  R = typeof(sx[1])
  t = floatmax(R) / (4 * max(x.nnz, 1))
  big = eb = small = es = zero(R)
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    u = x.nzval[k]
    big, eb, small, es = _split_add(big, eb, small, es, a[r, c], u, sa, sx, t)
    r == c || ((big, eb, small, es) = _split_add(big, eb, small, es, a[c, r], u, sa, sx, t))
  end
  big + eb, small + es
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
  _checkreal(x, s)
  @inbounds for k = 1:x.nnz
    x.nzval[k] *= s
  end
  x
end
function LinearAlgebra.lmul!(s::Number, x::_StoredIterate)
  _checkreal(x, s)
  @inbounds for k = 1:x.nnz
    x.nzval[k] = s * x.nzval[k]
  end
  x
end
function LinearAlgebra.rdiv!(x::_StoredIterate, s::Number)
  _checkreal(x, s)
  @inbounds for k = 1:x.nnz
    x.nzval[k] /= s
  end
  x
end
function LinearAlgebra.ldiv!(s::Number, x::_StoredIterate)
  _checkreal(x, s)
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

"""
    nonzeros(x::SparseIterate)

The stored values of `x`, including stored zeros, in storage order. Entry `k`
pairs with `activecoordinates(x)[k]`. The returned view aliases `x`: writing
through it changes `x`. The view is invalidated when the stored set changes.
"""
SparseArrays.nonzeros(x::SparseIterate) = view(x.nzval, 1:x.nnz)

"""
    findnz(x::SparseIterate)

The stored indices and values of `x`, including stored zeros, as copies.
Return `(I, V)` sorted by index for a vector, or `(I, J, V)` in column-major
order for a matrix. Changing the returned arrays does not change `x`.
"""
function SparseArrays.findnz(x::SparseIterate{<:Any, 1})
  order = sortperm(activecoordinates(x))
  x.nzval2ind[order], x.nzval[order]
end

function SparseArrays.findnz(x::SparseIterate{T, 2}) where {T}
  order = sortperm(activecoordinates(x))
  I, J, V = Vector{Int}(undef, x.nnz), Vector{Int}(undef, x.nnz), Vector{T}(undef, x.nnz)
  n = size(x, 1)
  @inbounds for (k, p) in enumerate(order)
    j, i = divrem(x.nzval2ind[p] - 1, n)
    I[k], J[k], V[k] = i + 1, j + 1, x.nzval[p]
  end
  I, J, V
end

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

function Base.copyto!(x::SparseIterate, bc::Broadcast.Broadcasted{Nothing})
  _checkreal_broadcast(x, bc)
  invoke(copyto!, Tuple{AbstractArray, Broadcast.Broadcasted{Nothing}}, x, bc)
end

function Base.copyto!(x::_StoredIterate, bc::Broadcast.Broadcasted{S}) where {S<:Broadcast.DefaultArrayStyle{0}}
  _checkreal_broadcast(x, bc)
  invoke(copyto!, Tuple{AbstractArray, Broadcast.Broadcasted{S}}, x, bc)
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
  _checkreal(coef, out); _checkreal(coef, A); _checkreal(coef, α); _checkreal(coef, β)
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
  _checkreal(coef, out); _checkreal(coef, A); _checkreal(coef, α); _checkreal(coef, β)
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
  _checkreal(x, out); _checkreal(x, A); _checkreal(x, α); _checkreal(x, β)
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
  _checkreal(x, out); _checkreal(x, A); _checkreal(x, α); _checkreal(x, β)
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
  _checkreal(coef, out); _checkreal(coef, trans_a); _checkreal(coef, α); _checkreal(coef, β)
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
  _checkreal(coef, v)
    length(v) == length(coef) || throw(DimensionMismatch())
    s = zero(promote_type(eltype(v), eltype(coef)))
    @inbounds @simd for icoef = 1:nnz(coef)
        s += dot(v[coef.nzval2ind[icoef]], coef.nzval[icoef])
    end
    s
end
LinearAlgebra.dot(coef::SparseIterate{<:Any, 1}, v::StridedVector) = dot(v, coef)

function LinearAlgebra.dot(x::SparseIterate{<:Any, 1}, y::SparseIterate{<:Any, 1})
    length(x) == length(y) || throw(DimensionMismatch())
    nnz(x) > nnz(y) && return dot(y, x)
    s = zero(promote_type(eltype(x), eltype(y)))
    @inbounds for k = 1:nnz(x)
        s += dot(x.nzval[k], y[x.nzval2ind[k]])
    end
    s
end

# a' X b and the entries of A X B for a sparse matrix X, in O(nnz(X))
function LinearAlgebra.dot(a::StridedVector, X::SparseIterate{<:Any, 2}, b::StridedVector)
  _checkreal(X, a); _checkreal(X, b)
  (length(a), length(b)) == size(X) || throw(DimensionMismatch())
  m = size(X, 1)
  s = zero(promote_type(eltype(a), eltype(X), eltype(b)))
  @inbounds for k = 1:nnz(X)
    i = X.nzval2ind[k]
    s += a[mod1(i, m)] * X.nzval[k] * b[cld(i, m)]
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
  _checkreal(X, A); _checkreal(X, B)
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
# aliasing hooks from Base (non-public functions)
Base.dataids(c::SymmetricCoordinates) = Base.dataids(c.x)
Base.unaliascopy(c::SymmetricCoordinates) = coordinates(copy(c.x))

Base.size(c::SymmetricCoordinates) = (length(c.x.ind2nzval),)
Base.IndexStyle(::Type{<:SymmetricCoordinates}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(c::SymmetricCoordinates, k::Int) = _getstored(c.x, k)
Base.@propagate_inbounds Base.setindex!(c::SymmetricCoordinates, v, k::Int) = (_setstored!(c.x, v, k); c)

coordinates(x::SymmetricSparseIterate) = SymmetricCoordinates(x)

# A matrix written into x must have the axes of x and be exactly symmetric; the
# checks run before anything is written. Symmetric(A, :L) keeps one triangle of
# a matrix that is not symmetric.

# NaN matches NaN and 0.0 matches -0.0
_symequal(a, b) = isequal(a, b) || (a == b) === true

# sources whose entries (r, c) and (c, r) are equal by construction
_symmetric_by_type(A) = false
_symmetric_by_type(::Union{SymmetricSparseIterate, Symmetric{<:Number}, Hermitian{<:Real},
                           Diagonal{<:Number}, SymTridiagonal{<:Number}}) = true
_symmetric_by_type(A::Union{Adjoint, Transpose}) = _symmetric_by_type(parent(A))

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

_check_symmetric(A::AbstractMatrix) =
  _symmetric_by_type(A) || _issymmetric_exact(A, _symequal) || _throw_asymmetric(A)
@noinline function _throw_asymmetric(A::AbstractMatrix)
  r, c = _asymmetric_pair(A, _symequal)
  _throw_asymmetric(r, c, A[r, c], A[c, r])
end
@noinline _throw_asymmetric(r, c, a, b) = throw(ArgumentError(
  "the value written into a SymmetricSparseIterate must be symmetric, but entry ($r, $c) " *
  "is $a and entry ($c, $r) is $b; use Symmetric(A, :L) to store the lower triangle of A"))

_check_axes(x::SymmetricSparseIterate, A::AbstractArray) = axes(A) == axes(x) || _throw_axes(x, A)
@noinline _throw_axes(x, A) =
  throw(DimensionMismatch("source axes $(axes(A)) do not match destination axes $(axes(x))"))

# B, a product such as U Σ Vᵀ whose triangles agree only up to rounding when its
# input is symmetric: a SymmetricSparseIterate keeps the lower one
_copyto_rounded!(Y, B) = copyto!(Y, B)
_copyto_rounded!(Y::SymmetricSparseIterate, B) = copyto!(Y, Symmetric(B, :L))

# c, a parameter written into y: when y is a SymmetricSparseIterate, it must
# have the axes of y and be symmetric
_check_symmetric_source(y, c) = true
_check_symmetric_source(y::SymmetricSparseIterate, c::AbstractArray) = _check_axes(y, c)
_check_symmetric_source(y::SymmetricSparseIterate, c::AbstractMatrix) =
  _check_axes(y, c) && _check_symmetric(c)

Base.convert(::Type{SymmetricSparseIterate{T}}, ::AbstractArray) where {T<:Complex} = _real_error(:SymmetricSparseIterate)

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

# From a matrix source, which must be symmetric.
Base.copyto!(x::SymmetricSparseIterate, y::AbstractMatrix) = _copyto_lower!(x, y)
# Resolve sparse-source copyto! ambiguities.
@static if isdefined(@__MODULE__, :_AbstractSparseMatrixCSC)
  Base.copyto!(x::SymmetricSparseIterate, y::_AbstractSparseMatrixCSC) = _copyto_lower!(x, y)
  Base.copyto!(x::SymmetricSparseIterate, y::Union{Adjoint{<:Any,<:_AbstractSparseMatrixCSC},
                                                 Transpose{<:Any,<:_AbstractSparseMatrixCSC}}) = _copyto_lower!(x, y)
end
# SparseArrays also specializes matrix views and transposed sparse sources.
@static if isdefined(@__MODULE__, :_SparseMatrixCSCOrView)
  Base.copyto!(x::SymmetricSparseIterate, y::_SparseMatrixCSCOrView) = _copyto_lower!(x, y)
end
@static if isdefined(@__MODULE__, :_AdjOrTransSparseVectorOrView)
  Base.copyto!(x::SymmetricSparseIterate, y::_AdjOrTransSparseVectorOrView) = _copyto_lower!(x, y)
end
# Sparse factorizations define typed matrix and untyped array copy methods.
# Apply the same checks to their dense matrix wrapper.
@static if isdefined(@__MODULE__, :_CHOLMODDense)
  Base.copyto!(x::SymmetricSparseIterate, y::_CHOLMODDense) = _copyto_lower!(x, y)
  @static if isdefined(@__MODULE__, :_CHOLMODVTypes)
    Base.copyto!(x::SymmetricSparseIterate{T}, y::_CHOLMODDense{T}) where
        {T<:_CHOLMODVTypes} = _copyto_lower!(x, y)
  end
end
function _copyto_lower!(x::SymmetricSparseIterate, y::AbstractMatrix)
  _checkreal(x, y)
  _check_axes(x, y)
  _check_symmetric(y)
  y = Base.unalias(x, y)
  p = x.p
  @inbounds for c = 1:p, r = c:p
    _setstored!(x, y[r, c], tril_index(p, r, c))
  end
  x
end
# A source with another number of dimensions cannot have the axes of x.
Base.copyto!(x::SymmetricSparseIterate, y::AbstractArray) = _throw_axes(x, y)

# A partial copy would write one position of an off-diagonal entry and not the
# other, or update it twice. The 3- and 4-argument copyto! forward to the
# 5-argument one.
@noinline _throw_partial(f) = throw(ArgumentError(
  "$f cannot write part of a SymmetricSparseIterate, whose positions (r, c) and (c, r) " *
  "are one entry; write a symmetric matrix with copyto!(x, A) or axpy!(α, A, x)"))
@noinline _throw_positionwise(f, alternative) = throw(ArgumentError(
  "$f would write a SymmetricSparseIterate position by position, and its result is not " *
  "symmetric in general; $alternative"))

Base.copyto!(x::SymmetricSparseIterate, ::Integer, ::AbstractArray, ::Integer, ::Integer) =
  _throw_partial("copyto! with offsets")
Base.copyto!(x::SymmetricSparseIterate, ::CartesianIndices, ::AbstractArray, ::CartesianIndices) =
  _throw_partial("copyto! with index ranges")
Base.copyto!(x::SymmetricSparseIterate, ::AbstractRange{Int}, ::AbstractRange{Int}, ::AbstractVecOrMat,
             ::AbstractRange{Int}, ::AbstractRange{Int}) = _throw_partial("copyto! with index ranges")
# Resolve the ambiguity with SparseArrays' method for a sparse source.
@static if isdefined(@__MODULE__, :_AbstractSparseMatrixCSC)
  Base.copyto!(x::SymmetricSparseIterate,
               ::CartesianIndices{2, <:Tuple{OrdinalRange{Int, Int}, OrdinalRange{Int, Int}}},
               ::_AbstractSparseMatrixCSC,
               ::CartesianIndices{2, <:Tuple{OrdinalRange{Int, Int}, OrdinalRange{Int, Int}}}) =
    _throw_partial("copyto! with index ranges")
end

# Transposed range copies still write individual matrix positions.
Base.copyto!(x::SymmetricSparseIterate, ::AbstractUnitRange{Int}, ::AbstractUnitRange{Int},
             ::AbstractChar, ::AbstractVecOrMat, ::AbstractUnitRange{Int}, ::AbstractUnitRange{Int}) =
  _throw_partial("copyto! with a transpose flag and index ranges")
_copy_transpose!(x::SymmetricSparseIterate, ::AbstractRange{Int}, ::AbstractRange{Int},
                 ::AbstractVecOrMat, ::AbstractRange{Int}, ::AbstractRange{Int}) =
  _throw_partial("copy_transpose! with index ranges")
_copy_transpose!(x::SymmetricSparseIterate, ::AbstractUnitRange{Int}, ::AbstractUnitRange{Int},
                 ::AbstractChar, ::AbstractVecOrMat, ::AbstractUnitRange{Int}, ::AbstractUnitRange{Int}) =
  _throw_partial("copy_transpose! with a transpose flag and index ranges")
@static if isdefined(@__MODULE__, :_copy_adjoint!)
  _copy_adjoint!(x::SymmetricSparseIterate, ::AbstractRange{Int}, ::AbstractRange{Int},
                ::AbstractVecOrMat, ::AbstractRange{Int}, ::AbstractRange{Int}) =
    _throw_partial("copy_adjoint! with index ranges")
end

# Reading or decoding must reject the destination before consuming the input.
Base.read!(io::IO, x::SymmetricSparseIterate) =
  _throw_positionwise("read!", "read into a dense array, then call copyto!(x, A) with a symmetric result")
Base.readbytes!(io::IO, x::SymmetricSparseIterate{UInt8}, nb = length(x)) =
  _throw_positionwise("readbytes!", "read into a dense array, then call copyto!(x, A) with a symmetric result")
Base.hex2bytes!(x::SymmetricSparseIterate{UInt8}, itr) =
  _throw_positionwise("hex2bytes!", "decode into a dense array, then call copyto!(x, A) with a symmetric result")
Base.hex2bytes!(x::SymmetricSparseIterate{UInt8}, itr::String) =
  _throw_positionwise("hex2bytes!", "decode into a dense array, then call copyto!(x, A) with a symmetric result")

# These position-wise writers cannot preserve shared symmetric entries.
Base.reverse!(x::SymmetricSparseIterate; dims = :) =
  _throw_positionwise("reverse!", "use reverse(Matrix(x); dims)")
Base.sort!(x::SymmetricSparseIterate; kws...) =
  _throw_positionwise("sort!", "use sort(Matrix(x); dims)")
Base.accumulate!(op, x::SymmetricSparseIterate, A; kws...) =
  _throw_positionwise("accumulate!", "accumulate into Matrix(x) instead")
Base.kron!(x::SymmetricSparseIterate, A::AbstractVecOrMat, B::AbstractVecOrMat) =
  _throw_positionwise("kron!", "use copyto!(x, kron(A, B))")
Base.kron!(x::SymmetricSparseIterate, A::AbstractVecOrMat, B::Number) =
  _throw_positionwise("kron!", "use copyto!(x, kron(A, B))")
Base.kron!(x::SymmetricSparseIterate, A::Number, B::AbstractVecOrMat) =
  _throw_positionwise("kron!", "use copyto!(x, kron(A, B))")
@static if isdefined(LinearAlgebra, :copytrito!)
  LinearAlgebra.copytrito!(x::SymmetricSparseIterate, A::AbstractMatrix, uplo::AbstractChar) =
    _throw_positionwise("copytrito!", "use copyto!(x, A) with a symmetric matrix")
end

# Structured Kronecker and sparse triangular-copy sources specialize the writer.
for (A, B) in ((AbstractMatrix, Diagonal), (Diagonal, AbstractMatrix), (Diagonal, Diagonal))
  @eval Base.kron!(x::SymmetricSparseIterate, a::$A, b::$B) =
    _throw_positionwise("kron!", "use copyto!(x, kron(A, B))")
end
@static if isdefined(LinearAlgebra, :copytrito!) && isdefined(@__MODULE__, :_AbstractSparseMatrixCSC)
  LinearAlgebra.copytrito!(x::SymmetricSparseIterate, A::_AbstractSparseMatrixCSC, uplo::Char) =
    _throw_positionwise("copytrito!", "use copyto!(x, A) with a symmetric matrix")
end

# Triangular masking is a no-op only when it keeps every position.
function LinearAlgebra.triu!(x::SymmetricSparseIterate, k::Integer = 0)
  (x.p == 0 || k <= 1 - x.p) && return x
  _throw_positionwise("triu!", "use triu(x, k)")
end
function LinearAlgebra.tril!(x::SymmetricSparseIterate, k::Integer = 0)
  (x.p == 0 || k >= x.p - 1) && return x
  _throw_positionwise("tril!", "use tril(x, k)")
end

# Matrix, factorization, orthogonal-factor and rotation operands need a buffer.
const _SymmetricMatrixOperand = Union{AbstractMatrix, Factorization, _AbstractQ, _AbstractRotation}
LinearAlgebra.lmul!(A::_SymmetricMatrixOperand, x::SymmetricSparseIterate; kws...) =
  _throw_positionwise("lmul!", "compute into a dense matrix, then copyto!(x, A) with a symmetric result")
LinearAlgebra.rmul!(x::SymmetricSparseIterate, A::_SymmetricMatrixOperand; kws...) =
  _throw_positionwise("rmul!", "compute into a dense matrix, then copyto!(x, A) with a symmetric result")
LinearAlgebra.ldiv!(A::_SymmetricMatrixOperand, x::SymmetricSparseIterate; kws...) =
  _throw_positionwise("ldiv!", "compute into a dense matrix, then copyto!(x, A) with a symmetric result")
LinearAlgebra.rdiv!(x::SymmetricSparseIterate, A::_SymmetricMatrixOperand; kws...) =
  _throw_positionwise("rdiv!", "compute into a dense matrix, then copyto!(x, A) with a symmetric result")
LinearAlgebra.ldiv!(x::SymmetricSparseIterate, A::_SymmetricMatrixOperand, B::AbstractArray; kws...) =
  _throw_positionwise("ldiv!", "compute into a dense matrix, then copyto!(x, A) with a symmetric result")

# Resolve structured operands before any of their specialized writers run.
for (f, operands) in (
    (:lmul!, (Diagonal, Bidiagonal, LinearAlgebra.AbstractTriangular,
      Union{LowerTriangular{T,S}, UpperTriangular{T,S}, UnitLowerTriangular{T,S},
            UnitUpperTriangular{T,S}} where {T,S<:AbstractMatrix{T}},
      LQ, _QRPackedQ, _AdjointQ{<:Any,<:_QRPackedQ},
      _Rotation, _AdjointRotation{<:Any,<:_Rotation}, LinearAlgebra.Givens)),
    (:rmul!, (Diagonal, Bidiagonal, LinearAlgebra.AbstractTriangular,
      Union{LowerTriangular{T,S}, UpperTriangular{T,S}, UnitLowerTriangular{T,S},
            UnitUpperTriangular{T,S}} where {T,S<:AbstractMatrix{T}},
      _QRPackedQ, _AdjointQ{<:Any,<:_QRPackedQ},
      _Rotation, _AdjointRotation{<:Any,<:_Rotation}, LinearAlgebra.Givens)),
    (:ldiv!, (Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal, UpperHessenberg,
      Union{Adjoint{T,S},Transpose{T,S}} where {T,S<:Bidiagonal},
      Union{Adjoint{T,S},Transpose{T,S}} where {T,S<:UpperHessenberg{T}},
      LinearAlgebra.AbstractTriangular, _AbstractQ,
      LQ, Cholesky, CholeskyPivoted, LU, LU{T,Tridiagonal{T,V}} where {T,V},
      SVD, QR, _QRCompactWY, QRPivoted, Hessenberg, LDLt{<:Any,<:SymTridiagonal},
      _AdjointFactorization{<:Any,<:LU}, _TransposeFactorization{<:Any,<:LU},
      _AdjointFactorization{<:Any,<:LU{T,Tridiagonal{T,V}}} where {T,V},
      _TransposeFactorization{<:Any,<:LU{T,Tridiagonal{T,V}}} where {T,V},
      _AdjointFactorization{<:Any,<:Union{QR,_QRCompactWY,QRPivoted}},
      _AdjointFactorization{<:Any,<:LQ}, _AdjointFactorization{<:Any,<:Hessenberg})),
    (:rdiv!, (Diagonal, Bidiagonal, SymTridiagonal, UpperHessenberg,
      Union{Adjoint{T,S},Transpose{T,S}} where {T,S<:Bidiagonal},
      Union{Adjoint{T,S},Transpose{T,S}} where {T,S<:UpperHessenberg{T}},
      LinearAlgebra.AbstractTriangular, _AbstractQ, Cholesky, CholeskyPivoted,
      Hessenberg, LDLt{<:Any,<:SymTridiagonal}, LU, LU{T,Tridiagonal{T,V}} where {T,V},
      _AdjointFactorization, _TransposeFactorization)))
  for A in operands
    if f === :lmul! || f === :ldiv!
      @eval LinearAlgebra.$f(A::$A, x::SymmetricSparseIterate; kws...) =
        _throw_positionwise($(string(f)), "compute into a dense matrix, then call copyto!(x, A) with a symmetric result")
    else
      @eval LinearAlgebra.$f(x::SymmetricSparseIterate, A::$A; kws...) =
        _throw_positionwise($(string(f)), "compute into a dense matrix, then call copyto!(x, A) with a symmetric result")
    end
  end
end
# Typed factorization methods couple the factor and destination element types.
for A in (QR, _QRCompactWY)
  @eval LinearAlgebra.ldiv!(A::$A{T}, x::SymmetricSparseIterate{T}; kws...) where {T} =
    _throw_positionwise("ldiv!", "solve into a dense matrix, then call copyto!(x, A) with a symmetric result")
end
LinearAlgebra.ldiv!(A::QRPivoted{T,<:StridedMatrix}, x::SymmetricSparseIterate{T}; kws...) where
    {T<:Union{Float32,Float64,ComplexF32,ComplexF64}} =
  _throw_positionwise("ldiv!", "solve into a dense matrix, then call copyto!(x, A) with a symmetric result")
LinearAlgebra.ldiv!(A::BunchKaufman{T}, x::SymmetricSparseIterate{T}; kws...) where
    {R<:Union{AbstractFloat,Rational}, T<:Union{R,Complex{R}}} =
  _throw_positionwise("ldiv!", "solve into a dense matrix, then call copyto!(x, A) with a symmetric result")
LinearAlgebra.ldiv!(A::Hessenberg{<:Complex,<:Any,<:AbstractMatrix{<:Real}},
                    x::SymmetricSparseIterate{<:Complex}; kws...) =
  _throw_positionwise("ldiv!", "solve into a dense matrix, then call copyto!(x, A) with a symmetric result")
LinearAlgebra.rdiv!(x::SymmetricSparseIterate{<:Complex},
                    A::Hessenberg{<:Complex,<:Any,<:AbstractMatrix{<:Real}}; kws...) =
  _throw_positionwise("rdiv!", "solve into a dense matrix, then call copyto!(x, A) with a symmetric result")
# Three-argument solves specialize both the operator and the right-hand side.
for (A, B) in ((AbstractMatrix, AbstractArray), (Factorization, AbstractMatrix),
               (LinearAlgebra.AbstractTriangular, AbstractVecOrMat), (Diagonal, AbstractVecOrMat),
               (Bidiagonal, AbstractVecOrMat), (_AbstractQ, AbstractVecOrMat),
               (Union{Adjoint{T,S},Transpose{T,S}} where {T,S<:Bidiagonal}, AbstractVecOrMat))
  @eval LinearAlgebra.ldiv!(x::SymmetricSparseIterate, A::$A, B::$B; kws...) =
    _throw_positionwise("ldiv!", "solve into a dense matrix, then call copyto!(x, A) with a symmetric result")
end

# Reductions must reject the destination before initializing its entries.
for f in (:sum!, :prod!, :maximum!, :minimum!, :all!, :any!, :extrema!)
  @eval begin
    Base.$f(x::SymmetricSparseIterate, A::AbstractArray; kws...) =
      _throw_positionwise($(string(f)), "reduce into a dense array, then call copyto!(x, A)")
    Base.$f(f::Function, x::SymmetricSparseIterate, A::AbstractArray; kws...) =
      _throw_positionwise($(string(f)), "reduce into a dense array, then call copyto!(x, A)")
  end
end
Base.count!(x::SymmetricSparseIterate, A::Union{_AbstractBroadcasted, AbstractArray}; kws...) =
  _throw_positionwise("count!", "reduce into a dense array, then call copyto!(x, A)")
Base.count!(f, x::SymmetricSparseIterate, A::Union{_AbstractBroadcasted, AbstractArray}; kws...) =
  _throw_positionwise("count!", "reduce into a dense array, then call copyto!(x, A)")
for f in (:findmax!, :findmin!)
  @eval begin
    Base.$f(x::SymmetricSparseIterate, inds::AbstractArray, A::AbstractArray; kws...) =
      _throw_positionwise($(string(f)), "reduce into a dense array, then call copyto!(x, A)")
    Base.$f(vals::AbstractArray, x::SymmetricSparseIterate, A::AbstractArray; kws...) =
      _throw_positionwise($(string(f)), "reduce into dense arrays, then call copyto!(x, A)")
    Base.$f(vals::SymmetricSparseIterate, x::SymmetricSparseIterate, A::AbstractArray; kws...) =
      _throw_positionwise($(string(f)), "reduce into dense arrays, then call copyto!(x, A)")
  end
end

# In-place factorizations overwrite the symmetry shared by the two triangles.
LinearAlgebra.lu!(x::SymmetricSparseIterate; kws...) =
  _throw_positionwise("lu!", "use lu(Matrix(x))")
for P in (Union{NoPivot, RowMaximum, RowNonZero}, Val{true}, Val{false})
  @eval LinearAlgebra.lu!(x::SymmetricSparseIterate, pivot::$P; kws...) =
    _throw_positionwise("lu!", "use lu(Matrix(x))")
end
LinearAlgebra.qr!(x::SymmetricSparseIterate; kws...) =
  _throw_positionwise("qr!", "use qr(Matrix(x))")
for P in (NoPivot, ColumnNorm, Val{true}, Val{false})
  @eval LinearAlgebra.qr!(x::SymmetricSparseIterate, pivot::$P; kws...) =
    _throw_positionwise("qr!", "use qr(Matrix(x))")
end
LinearAlgebra.cholesky!(x::SymmetricSparseIterate; kws...) =
  _throw_positionwise("cholesky!", "use cholesky(Symmetric(Matrix(x)))")
for P in (NoPivot, RowMaximum, Val{true}, Val{false})
  @eval LinearAlgebra.cholesky!(x::SymmetricSparseIterate, pivot::$P; kws...) =
    _throw_positionwise("cholesky!", "use cholesky(Symmetric(Matrix(x)))")
end
LinearAlgebra.bunchkaufman!(x::SymmetricSparseIterate, rook::Bool = false; kws...) =
  _throw_positionwise("bunchkaufman!", "use bunchkaufman(Matrix(x))")

# get! initializes the destination before copying a subset; sortperm! writes indices.
Base.get!(x::SymmetricSparseIterate{T}, A::AbstractArray,
          I::Union{AbstractRange, AbstractVector{Int}}, default::T) where {T} =
  _throw_positionwise("get!", "use get! with a dense destination")
Base.get!(x::SymmetricSparseIterate{T}, A::AbstractArray, I::_RangeVecIntList, default::T) where {T} =
  _throw_positionwise("get!", "use get! with a dense destination")
Base.sortperm!(x::SymmetricSparseIterate{T}, A::AbstractArray; kws...) where {T<:Integer} =
  _throw_positionwise("sortperm!", "use sortperm(A; dims)")

# AbstractQ copying first writes an identity, so form the whole value in a buffer.
function Base.copyto!(x::SymmetricSparseIterate, Q::_AbstractQ)
  A = Matrix{promote_type(eltype(x), eltype(Q))}(I, size(x))
  lmul!(Q, A)
  copyto!(x, A)
end

# Indexed array assignments must cover every position in storage order.
_full_index(::Colon, ax) = true
_full_index(I::AbstractVector{<:Integer}, ax) =
  !(eltype(I) <: Bool) && length(I) == length(ax) && all(isequal(a, b) for (a, b) in zip(I, ax))
_full_index(I, ax) = false
_full_linear(I, x) = _full_index(I, 1:length(x))
_full_linear(I::AbstractArray{CartesianIndex{2}}, x) =
  size(I) == size(x) && all(isequal(a, b) for (a, b) in zip(I, CartesianIndices(x)))

function _setindex_symmetric!(x, X, I...)
  checkbounds(x, Base.to_indices(x, I)...)
  if length(I) == 2
    all(_full_index(I[d], axes(x, d)) for d in 1:2) || _throw_partial("setindex!")
    return copyto!(x, X)
  elseif length(I) == 1 && _full_linear(I[1], x)
    length(X) == length(x) || _throw_axes(x, X)
    return copyto!(x, reshape(X, size(x)))
  end
  _throw_partial("setindex!")
end
function Base.setindex!(x::SymmetricSparseIterate, X::AbstractArray, I...)
  if any(i -> i isa Union{Colon, AbstractArray}, I)
    return _setindex_symmetric!(x, X, I...)
  end
  invoke(setindex!, Tuple{AbstractArray, Any, Vararg{Any}}, x, X, I...)
end
# Preserve scalar assignment (and its conversion error for an array value).
Base.setindex!(x::SymmetricSparseIterate, X::AbstractArray, r::Int, c::Int) =
  invoke(setindex!, Tuple{SymmetricSparseIterate, Any, Int, Int}, x, X, r, c)

# These common iterator sources can fail after a partial write.
Base.copyto!(x::SymmetricSparseIterate, ::Union{Tuple, Base.Generator, Number}) =
  _throw_positionwise("copyto! from an iterator", "collect the values into a symmetric matrix A and call copyto!(x, A)")
Base.copyto!(x::SymmetricSparseIterate, ::Integer, ::AbstractArray) = _throw_partial("copyto! with offsets")
Base.copyto!(x::SymmetricSparseIterate, ::Integer, ::AbstractArray, ::Integer) = _throw_partial("copyto! with offsets")
Base.copyto!(x::SymmetricSparseIterate, ::Integer, src) = _throw_partial("copyto! with offsets")
Base.copyto!(x::SymmetricSparseIterate, ::Integer, src, ::Integer) = _throw_partial("copyto! with offsets")
Base.copyto!(x::SymmetricSparseIterate, ::Integer, src, ::Integer, ::Integer) =
  _throw_partial("copyto! with offsets")

# Broadcasting through a full view or reshape must use the checked matrix writer.
const _SymmetricReshape = Base.ReshapedArray{T, N, P} where {T, N, P<:SymmetricSparseIterate}
const _SymmetricView = SubArray{T, N, P} where {T, N, P<:SymmetricSparseIterate}
const _SymmetricReshapeView = SubArray{T, N, P} where {T, N, P<:_SymmetricReshape}
const _SymmetricBroadcastDest = Union{_SymmetricView, _SymmetricReshape, _SymmetricReshapeView}

function _checked_symmetric_parent(dest::_SymmetricView, f)
  x = parent(dest)
  I = parentindices(dest)
  length(I) == 2 && all(_full_index(I[d], axes(x, d)) for d in 1:2) ||
    _throw_partial("$f into a view")
  x
end
function _checked_symmetric_parent(dest::_SymmetricReshapeView, f)
  R = parent(dest)
  inds = view(LinearIndices(R), parentindices(dest)...)
  length(dest) == length(R) && all(i == k for (i, k) in zip(inds, 1:length(R))) ||
    _throw_partial("$f into a reshaped view")
  parent(R)
end
_checked_symmetric_parent(dest::_SymmetricReshape, f) = parent(dest)
function _broadcast_symmetric!(dest::_SymmetricView, bc)
  copyto!(_checked_symmetric_parent(dest, "broadcast!"), bc)
  dest
end
_broadcast_symmetric!(dest::Union{_SymmetricReshape, _SymmetricReshapeView}, bc) =
  _broadcast_reshaped!(dest, _checked_symmetric_parent(dest, "broadcast!"), bc)
function _broadcast_reshaped!(dest, x, bc)
  axes(dest) == axes(bc) || throw(DimensionMismatch("broadcast! axes do not match destination"))
  # Collect before conversion and before writing, including aliased updates.
  vals = collect(bc)
  copyto!(x, reshape(vals, size(x)))
  dest
end
Base.copyto!(dest::_SymmetricBroadcastDest, bc::Broadcast.Broadcasted{Nothing}) = _broadcast_symmetric!(dest, bc)
# Scalar broadcast fast paths use fill!, so it checks the same full destinations.
function Base.fill!(dest::_SymmetricBroadcastDest, v)
  x = _checked_symmetric_parent(dest, "fill! or broadcasting")
  _checkreal_broadcast(x, v)
  fill!(x, v)
  dest
end

# Transposition and dimension permutation still require exactly symmetric values.
LinearAlgebra.transpose!(x::SymmetricSparseIterate, A::AbstractMatrix) = copyto!(x, transpose(A))
LinearAlgebra.transpose!(x::SymmetricSparseIterate, A::AbstractVector) = copyto!(x, transpose(A))
LinearAlgebra.adjoint!(x::SymmetricSparseIterate, A::AbstractMatrix) = copyto!(x, adjoint(A))
LinearAlgebra.adjoint!(x::SymmetricSparseIterate, A::AbstractVector) = copyto!(x, adjoint(A))
function Base.permutedims!(x::SymmetricSparseIterate, A::AbstractArray, perm)
  ndims(A) == ndims(x) || throw(MethodError(permutedims!, (x, A, perm)))
  length(perm) == ndims(A) && isperm(perm) || _throw_symmetric_permutation()
  copyto!(x, perm[1] == 1 ? A : transpose(A))
end

@noinline _throw_symmetric_permutation() = throw(ArgumentError(
  "permutedims! requires a permutation of two dimensions; use (1, 2) or (2, 1)"))

# Scalar multiplication must not fall back to length-only, position-wise writes.
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::AbstractArray, s::Number) = mul!(x, A, s, true, false)
LinearAlgebra.mul!(x::SymmetricSparseIterate, s::Number, A::AbstractArray) = mul!(x, s, A, true, false)
function LinearAlgebra.mul!(x::SymmetricSparseIterate, A::AbstractArray, s::Number, α::Number, β::Number)
  _check_axes(x, A)
  invoke(mul!, Tuple{AbstractArray, AbstractArray, Number, Number, Number}, x, A, s, α, β)
end
function LinearAlgebra.mul!(x::SymmetricSparseIterate, s::Number, A::AbstractArray, α::Number, β::Number)
  _check_axes(x, A)
  invoke(mul!, Tuple{AbstractArray, Number, AbstractArray, Number, Number}, x, s, A, α, β)
end

# Broadcasting into x computes each entry of the lower triangle once: visiting
# both (r, c) and (c, r) would cost twice as much and, as they share storage,
# apply an in-place update such as x .= 2 .* x twice. When the leaves do not
# show that the value is symmetric, a first pass compares (r, c) with (c, r)
# before anything is written.
function Base.copyto!(x::SymmetricSparseIterate, bc::Broadcast.Broadcasted{Nothing})
  _checkreal_broadcast(x, bc)
  axes(x) == axes(bc) || throw(DimensionMismatch("destination axes $(axes(x)) do not match $(axes(bc))"))
  symmetric = _bc_symmetric(x, bc)    # before preprocess, which wraps the leaves
  bc′ = Broadcast.preprocess(x, bc)
  p = x.p
  if !symmetric
    @inbounds for c = 1:p, r = c+1:p
      a, b = bc′[CartesianIndex(r, c)], bc′[CartesianIndex(c, r)]
      _symequal(a, b) || _throw_asymmetric(r, c, a, b)
    end
  end
  @inbounds for c = 1:p, r = c:p
    _setstored!(x, bc′[CartesianIndex(r, c)], tril_index(p, r, c))
  end
  x
end

_checkreal_broadcast(x, bc::Broadcast.Broadcasted) = _checkreal_args(x, bc.args)
_checkreal_args(x, args::Tuple) = (_checkreal_broadcast(x, first(args)); _checkreal_args(x, Base.tail(args)))
_checkreal_args(x, ::Tuple{}) = true
# Broadcast leaves can be control arguments, such as functions or rounding modes.
_checkreal_broadcast(x, a) = true
_checkreal_broadcast(x, a::Union{AbstractArray, Ref, Number}) =
  !(eltype(a) <: Complex) || _real_error(nameof(typeof(x)))

# Whether the leaves show that a broadcast of a pure function is symmetric: each
# is a scalar, symmetric by type, or a matrix with the axes of x that is
# symmetric under isequal, which keeps -0.0 apart from 0.0, as 1 ./ A does.
_bc_symmetric(x, bc::Broadcast.Broadcasted) = _bc_args_symmetric(x, bc.args)
_bc_args_symmetric(x, args::Tuple) = _bc_symmetric(x, first(args)) && _bc_args_symmetric(x, Base.tail(args))
_bc_args_symmetric(x, ::Tuple{}) = true
_bc_symmetric(x, a) = axes(a) == () || _symmetric_by_type(a) ||
  (a isa AbstractMatrix && axes(a) == axes(x) && _issymmetric_exact(a, isequal))

# map! into x broadcasts from sources with the axes of x.
function Base.map!(f::F, x::SymmetricSparseIterate, A::AbstractArray) where {F}
  _checkreal_broadcast(x, A)
  _check_axes(x, A)
  x .= f.(A)
end

function Base.map!(f::F, x::SymmetricSparseIterate, A::AbstractArray, B::AbstractArray) where {F}
  _checkreal_broadcast(x, A); _checkreal_broadcast(x, B)
  _check_axes(x, A)
  _check_axes(x, B)
  x .= f.(A, B)
end

function Base.map!(f::F, x::SymmetricSparseIterate, A::AbstractArray, B::AbstractArray,
                   C::AbstractArray, Ds::Vararg{AbstractArray, N}) where {F, N}
  _checkreal_broadcast(x, A); _checkreal_broadcast(x, B); _checkreal_broadcast(x, C)
  _checkreal_args(x, Ds)
  _check_axes(x, A)
  _check_axes(x, B)
  _check_axes(x, C)
  foreach(D -> _check_axes(x, D), Ds)
  x .= f.(A, B, C, Ds...)
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

# A symmetric band visits each stored entry once; other nonempty bands cannot.
function _fillband!(x::SymmetricSparseIterate{T}, v, l, u) where {T}
  l > u && return x
  l == -u || _throw_positionwise("fillband!", "fill a band in Matrix(x), then call copyto!(x, A) with a symmetric result")
  _checkreal(x, v)
  value = convert(T, v)
  p = size(x, 1)
  for c in 1:p, r in c:(c + min(u, p - c))
    x[r, c] = value
  end
  x
end

# axpy! and axpby! broadcast, from a source with the axes of y
function LinearAlgebra.axpy!(α::Number, x::AbstractArray, y::SymmetricSparseIterate)
  _checkreal(y, x); _checkreal(y, α)
  _check_axes(y, x)
  iszero(α) && return y
  y .+= α .* x
end
function LinearAlgebra.axpby!(α::Number, x::AbstractArray, β::Number, y::SymmetricSparseIterate)
  _checkreal(y, x); _checkreal(y, α); _checkreal(y, β)
  _check_axes(y, x)
  iszero(α) && isone(β) && return y
  y .= α .* x .+ β .* y
end

# replace and replace!, with a function or with pairs, reach this method, which
# base/set.jl names as the one a new container type implements. Each entry of
# the lower triangle is visited once, and count counts those entries.
function Base._replace!(new::Base.Callable, res::SymmetricSparseIterate, A::SymmetricSparseIterate,
                        count::Int)
  size(res) == size(A) || throw(DimensionMismatch("destination size $(size(res)) does not match $(size(A))"))
  p = A.p
  n = 0
  @inbounds for c = 1:p, r = c:p
    k = tril_index(p, r, c)
    a = _getstored(A, k)
    if n < count
      v = new(a)
      _setstored!(res, v, k)
      n += (a !== v)
    elseif res !== A
      _setstored!(res, a, k)
    end
  end
  res
end
# The indexed axpy! adds twice to an entry whose two positions are both in ry.
LinearAlgebra.axpy!(::Number, ::AbstractArray, ::AbstractArray{<:Integer}, ::SymmetricSparseIterate,
                    ::AbstractArray{<:Integer}) = _throw_partial("the indexed axpy!")

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

# resolve intersections with structured operand methods
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::LinearAlgebra.AbstractTriangular,
                  B::AbstractVector) = _mul_symmetric!(x, A, B, true, false)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::LinearAlgebra.AbstractTriangular,
                  B::AbstractMatrix) = _mul_symmetric!(x, A, B, true, false)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::AbstractMatrix,
                  B::LinearAlgebra.AbstractTriangular) = _mul_symmetric!(x, A, B, true, false)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::LinearAlgebra.AbstractTriangular,
                  B::LinearAlgebra.AbstractTriangular) = _mul_symmetric!(x, A, B, true, false)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::LinearAlgebra.AbstractTriangular,
                  B::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal}) = _mul_symmetric!(x, A, B, true, false)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal},
                  B::LinearAlgebra.AbstractTriangular) = _mul_symmetric!(x, A, B, true, false)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::LinearAlgebra.AbstractTriangular,
                  B::AbstractVector, α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::LinearAlgebra.AbstractTriangular,
                  B::AbstractMatrix, α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::AbstractMatrix,
                  B::LinearAlgebra.AbstractTriangular, α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::LinearAlgebra.AbstractTriangular,
                  B::LinearAlgebra.AbstractTriangular, α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::LinearAlgebra.AbstractTriangular,
                  B::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal}, α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal},
                  B::LinearAlgebra.AbstractTriangular, α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal},
                  B::AbstractVector, α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal},
                  B::AbstractMatrix, α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::AbstractMatrix,
                  B::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal}, α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal},
                  B::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal}, α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate{T}, A::SymmetricSparseIterate{T},
                  B::AbstractMatrix{T}) where {T} = _mul_symmetric!(x, A, B, true, false)
LinearAlgebra.mul!(x::SymmetricSparseIterate{T}, A::SymmetricSparseIterate{T},
                  B::LinearAlgebra.AbstractTriangular{T}) where {T} = _mul_symmetric!(x, A, B, true, false)

# compute and check the product before writing, including aliased operands
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::AbstractVecOrMat, B::AbstractVecOrMat) =
  _mul_symmetric!(x, A, B, true, false)
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::AbstractVecOrMat, B::AbstractVecOrMat,
                  α::Number, β::Number) = _mul_symmetric!(x, A, B, α, β)

# Flagged products use the same buffered symmetry and alias checks as mul!.
@static if hasmethod(mul!, Tuple{AbstractMatrix, Char, Char, AbstractMatrix, AbstractMatrix, Number, Number})
  @noinline _throw_mul_flag() = throw(ArgumentError(
    "mul! into a SymmetricSparseIterate supports only flags 'N', 'T' and 'C'; use those flags or compute into a dense matrix"))
  _symmetric_mul_operand(A, flag) = isequal(flag, 'N') ? A :
    isequal(flag, 'T') ? transpose(A) : isequal(flag, 'C') ? adjoint(A) : _throw_mul_flag()
  function LinearAlgebra.mul!(x::SymmetricSparseIterate, tA, tB,
                             A::AbstractVecOrMat, B::AbstractVecOrMat, α::Number, β::Number)
    mul!(x, _symmetric_mul_operand(A, tA), _symmetric_mul_operand(B, tB), α, β)
  end
end

# uniform scaling operands use a diagonal with the required inner dimension
LinearAlgebra.mul!(x::SymmetricSparseIterate, A::AbstractMatrix, J::UniformScaling,
                  α::Number, β::Number) =
  _mul_symmetric!(x, A, Diagonal(fill(J.λ, size(A, 2))), α, β)
LinearAlgebra.mul!(x::SymmetricSparseIterate, J::UniformScaling, B::AbstractVecOrMat,
                  α::Number, β::Number) =
  _mul_symmetric!(x, Diagonal(fill(J.λ, size(B, 1))), B, α, β)

function _mul_symmetric!(x::SymmetricSparseIterate, A, B, α, β)
  _checkreal(x, A); _checkreal(x, B); _checkreal(x, α); _checkreal(x, β)
  p = size(x, 1)
  size(A, 1) == size(B, 2) == p && size(A, 2) == size(B, 1) ||
    throw(DimensionMismatch("operands have sizes $(size(A)) and $(size(B)); destination has size $(size(x))"))
  if iszero(α)
    iszero(β) && return fill!(x, 0)
    isone(β) && return x
    x .= x .* β
    return x
  end
  T = Base.promote_op(*, eltype(A), eltype(B))
  U = promote_type(eltype(x), Base.promote_op(+, T, T))
  P = Matrix{U}(undef, p, p)
  mul!(P, A, B)
  pair = _asymmetric_pair(P, _symequal)
  if pair !== nothing
    r, c = pair
    throw(ArgumentError(
      "the product written into a SymmetricSparseIterate must be symmetric, but entry ($r, $c) " *
      "is $(P[r, c]) and entry ($c, $r) is $(P[c, r]); " *
      "use copyto!(x, Symmetric(P, :L)) with P = A * B, or (P + transpose(P)) / 2, to store a product that is not symmetric"))
  end
  if iszero(β)
    x .= P .* α
  else
    x .= P .* α .+ x .* β
  end
  x
end

LinearAlgebra.mul!(C::AbstractMatrix{T}, A::SymmetricSparseIterate{T}, B::AbstractMatrix{T}) where T = _mul_symsparse!(C, A, B)
# resolves ambiguity with LinearAlgebra's mul!(::AbstractMatrix, ::AbstractMatrix, ::AbstractTriangular)
LinearAlgebra.mul!(C::AbstractMatrix{T}, A::SymmetricSparseIterate{T}, B::LinearAlgebra.AbstractTriangular{T}) where T = _mul_symsparse!(C, A, B)

function _mul_symsparse!(C::AbstractMatrix{T}, A::SymmetricSparseIterate{T}, B::AbstractMatrix{T}) where T
  _checkreal(A, C); _checkreal(A, B)
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
  _checkreal(A, C); _checkreal(A, B)
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
  _checkreal(X, a); _checkreal(X, b)
  length(a) == X.p == length(b) || throw(DimensionMismatch())
  s = zero(promote_type(eltype(a), eltype(X), eltype(b)))
  @inbounds for k = 1:X.nnz
    r, c = tril_cartesian(X.p, X.nzval2ind[k])
    v = X.nzval[k]
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
    r, c = tril_cartesian(X.p, X.nzval2ind[k])
    v = X.nzval[k]
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
  invoke(norm, Tuple{Any, Real}, x, p)
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


function LinearAlgebra.mul!(C::AbstractMatrix, A::SymmetricSparseIterate, B::AbstractMatrix)
  _checkreal(A, C); _checkreal(A, B)
  invoke(mul!, Tuple{AbstractMatrix, AbstractMatrix, AbstractMatrix}, C, A, B)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::SymmetricSparseIterate, B::LinearAlgebra.AbstractTriangular)
  _checkreal(A, C); _checkreal(A, B)
  invoke(mul!, Tuple{AbstractMatrix, AbstractMatrix, LinearAlgebra.AbstractTriangular}, C, A, B)
end
function LinearAlgebra.mul!(C::AbstractVector, A::SymmetricSparseIterate, B::AbstractVector, α::Number, β::Number)
  _checkreal(A, C); _checkreal(A, B); _checkreal(A, α); _checkreal(A, β)
  invoke(mul!, Tuple{AbstractVector, AbstractMatrix, AbstractVector, Number, Number}, C, A, B, α, β)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::SymmetricSparseIterate, B::AbstractMatrix, α::Number, β::Number)
  _checkreal(A, C); _checkreal(A, B); _checkreal(A, α); _checkreal(A, β)
  invoke(mul!, Tuple{AbstractMatrix, AbstractMatrix, AbstractMatrix, Number, Number}, C, A, B, α, β)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::SymmetricSparseIterate, B::LinearAlgebra.AbstractTriangular, α::Number, β::Number)
  _checkreal(A, C); _checkreal(A, B); _checkreal(A, α); _checkreal(A, β)
  invoke(mul!, Tuple{AbstractMatrix, AbstractMatrix, LinearAlgebra.AbstractTriangular, Number, Number}, C, A, B, α, β)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::SymmetricSparseIterate,
                            B::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal}, α::Number, β::Number)
  _checkreal(A, C); _checkreal(A, B); _checkreal(A, α); _checkreal(A, β)
  invoke(mul!, Tuple{AbstractMatrix, AbstractMatrix, Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal}, Number, Number},
         C, A, B, α, β)
end
LinearAlgebra.mul!(C::SymmetricSparseIterate, A::SymmetricSparseIterate, B::AbstractMatrix, α::Number, β::Number) =
  _mul_symmetric!(C, A, B, α, β)
LinearAlgebra.mul!(C::SymmetricSparseIterate, A::SymmetricSparseIterate, B::LinearAlgebra.AbstractTriangular, α::Number, β::Number) =
  _mul_symmetric!(C, A, B, α, β)
LinearAlgebra.mul!(C::SymmetricSparseIterate, A::SymmetricSparseIterate, B::AbstractMatrix) =
  _mul_symmetric!(C, A, B, true, false)
LinearAlgebra.mul!(C::SymmetricSparseIterate, A::SymmetricSparseIterate, B::LinearAlgebra.AbstractTriangular) =
  _mul_symmetric!(C, A, B, true, false)

# Mixed real operands keep LinearAlgebra's multiplication order.
function *(A::Symmetric, X::Union{SparseIterate{<:Any, 2}, SymmetricSparseIterate}, B::Symmetric)
  _checkreal(X, A); _checkreal(X, B)
  invoke(*, Tuple{AbstractMatrix, AbstractMatrix, AbstractMatrix}, A, X, B)
end

LinearAlgebra.mul!(C::SymmetricSparseIterate, A::SymmetricSparseIterate,
                  B::Union{Bidiagonal, Diagonal, SymTridiagonal, Tridiagonal}, α::Number, β::Number) =
  _mul_symmetric!(C, A, B, α, β)

# Retain the complete stored pattern, including zeros, across wrapper calls.
function _output_buffer!(s::Scratch, y::Y) where {Y<:_StoredIterate}
  b = s.buf
  if !(b isa Y && size(b) == size(y))
    b = similar(y)
    s.buf = b
  end
  copyto!(b::Y, y)
end
_wrapper_output(y::_StoredIterate, z) = z
