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

Its entries are real. The constructors, two-argument `copyto!` from an array
(for a matrix also from a `UniformScaling` or an orthogonal factor), `copy!`,
broadcasting into `x` and `fill!` throw an `ArgumentError` for a source or value
of complex type, even with a zero imaginary part, before anything is written.
For a vector `x`, `vec(x)` and `reshape(x, length(x))` are `x` itself and are
checked too. `copy!` from a source of another size throws Base's error first,
unless the source is a `SparseIterate` or an [`AtomIterate`](@ref): one of the
same length is copied by value in linear order, and one of another length throws
a `DimensionMismatch`. The check reads types: a tuple, a generator or an array of abstract element type
such as `Vector{Any}` passes it, and its values are converted one by one. Other
writes also convert values one by one, as writes into a plain `Array` of the
element type of `x` do: writes through a view, through `coordinates(x)`, through
`vec` of a matrix or a
`reshape` to another shape, indexed assignment such as `x[i] = v` or `x[:] = c`,
`copyto!` with offsets or index ranges, and `map!`. A complex value converted
one by one throws an `InexactError`, possibly after part of `x` has been
written.
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
every position in order (`x[:, :] = A`, `x[:] = v`), and broadcasting,
`fill!`, `copyto!` and `copy!` into full, in-order views and reshapes
(`x[:, :] .= A`, `vec(x) .= v`, `copyto!(view(x, :, :), A)`).
The covered wrappers are views whose parent is `x`, reshapes whose parent is
`x`, and views of those reshapes; Base makes a view of such a view a view of
`x`, and a reshape of a reshape a reshape of `x`. Use `:` or integer vectors
matching each axis in order for a full two-index assignment or a full matrix
view. A full single-index assignment accepts `:`, the integers `1:length(x)`
in order, or `CartesianIndices(x)` with its shape.
`copyto!(x, Q)` from an orthogonal factor is checked too, as is `copyto!`
from one into a full matrix view or reshape.

Matrix values must have the axes of `x` and be exactly symmetric: entries
`(r, c)` and `(c, r)` are equal, with `NaN` equal to `NaN`, compared before
conversion to the element type of `x`. Linear full writes reshape their values
in column-major order for this check; `x[:] = v` requires `p²` values.
`copyto!` into a full view or reshape reads its source in linear order, as
Base does, so the source needs `p²` values and not the axes of `x`:
`copyto!(view(x, :, :), vec(A))` writes the symmetric `A`, while
`copyto!(x, vec(A))` throws.
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
writing. `copyto!` and `copy!` into such a part through a view throw before
writing too.
Logical-mask indexed assignment is rejected even for an all-true mask.
`copyto!` with offsets or index ranges (including transpose flags),
`copy_transpose!` and `copy_adjoint!` with index ranges, `copyto!` from a tuple, a
generator or a number, and the indexed `axpy!` also throw before writing, as
does `copyto!` with offsets into a view or reshape.
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
stay unchecked. Other writes through a view or reshape stay unchecked too:
indexed assignment such as `view(x, :, :)[:, :] = A`, scalar writes through a
view, `copyto!` with index ranges, and `copyto!` from a tuple, a generator, a
number or another non-array iterator.
The checks do not extend to other nested wrappers, such as `vec(view(x, :, :))`
or a view of it, including broadcasts, fills and copies through them.
Two-argument `copyto!(x, itr)` from other non-array iterators is unchecked.
A Base function can write part of `x` before reaching a check. `rand!` and
`randn!` give a symmetric matrix but draw each off-diagonal entry twice.
`shuffle!` and `shuffle` give wrong values; use `Matrix(x)` instead.

Its coordinates are the `p(p+1)/2` entries of the lower triangle, numbered
column by column ([`tril_index`](@ref)): [`prox_coordinate!`](@ref) updates
them, [`ncoordinates`](@ref) counts them, [`activecoordinates`](@ref) lists the
stored ones and [`coordinates`](@ref) is a vector view of them.

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
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
  acc = _split_start(R, x.nnz)
  @inbounds for k in 1:x.nnz
    acc = _split_add(acc, a[x.nzval2ind[k]], x.nzval[k], sa, sx)
  end
  _split_finish(acc)
end
# each off-diagonal product separately: a[r, c] + a[c, r] can overflow on its own
function _split_dot(a::AbstractArray, x::SymmetricSparseIterate, sa, sx)
  _checksize(a, x)
  R = typeof(sx[1])
  acc = _split_start(R, 2 * x.nnz)
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    u = x.nzval[k]
    acc = _split_add(acc, a[r, c], u, sa, sx)
    r == c || (acc = _split_add(acc, a[c, r], u, sa, sx))
  end
  _split_finish(acc)
end

# the stored entries of an iterate: the recoveries of R1 read only these
_stored_maxabs(x::Union{SparseIterate, SymmetricSparseIterate}) = _maximum_abs(view(x.nzval, 1:x.nnz))
_all_finite(x::Union{SparseIterate, SymmetricSparseIterate}) = all(isfinite, view(x.nzval, 1:x.nnz))


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

# Read-only handle on the internal list of active IDs. A writable view would let a caller
# break the agreement between the list and the reverse map of the iterate.
struct _ActiveIDs <: AbstractVector{Int}
  ids::SubArray{Int,1,Vector{Int},Tuple{UnitRange{Int}},true}
end
Base.size(h::_ActiveIDs) = size(h.ids)
Base.IndexStyle(::Type{_ActiveIDs}) = IndexLinear()
Base.dataids(h::_ActiveIDs) = Base.dataids(h.ids)     # as the view it wraps: aliasing checks still see the list
Base.unaliascopy(h::_ActiveIDs) = _ActiveIDs(Base.unaliascopy(h.ids))   # view(A, h) copies an aliasing index in its own type
Base.@propagate_inbounds Base.getindex(h::_ActiveIDs, i::Int) = h.ids[i]
Base.setindex!(::_ActiveIDs, v, i::Int) =
  throw(ArgumentError("activecoordinates(x) is read-only: write values through activevalues(x), or atom(x, k) for an AtomIterate"))

"""
    activecoordinates(x)

A read-only `AbstractVector{Int}` of the active coordinates of `x` (see
[`nactive`](@ref)), in the order in which they became active, except that
`dropzeros!` moves the last ones into the places of those it removes. It is a
handle on the internal list, not a snapshot: nothing is copied, and it does not
grow when coordinates become active, so call `activecoordinates` again after
changing the active set: after an activation, after `dropzeros!`, and after a
whole-state replacement such as `copyto!(x, y)`, which can reorder the list even
when the active set and `nactive(x)` stay the same. Writing through it throws an `ArgumentError`, because
the list and the iterate's reverse map must stay in agreement; write values
through [`activevalues`](@ref) (or `atom(x, k)` for an [`AtomIterate`](@ref)).
"""
function activecoordinates end

"""
    isactive(x, k)

Whether coordinate `k` of `x` is active (see [`nactive`](@ref)).
"""
function isactive end

"""
    coordinates(x)

A vector view of the coordinates of `x` (see [`ncoordinates`](@ref)): the
entries of a [`SparseIterate`](@ref) in linear order, whatever its shape, and the
entries of the lower triangle for a [`SymmetricSparseIterate`](@ref). The view is
an `AbstractVector` of length `ncoordinates(x)` indexed by coordinate, live (a
later write to `x` shows through it) and writable, and building it copies
nothing. It is never `x` itself.

The view is an ordinary vector of the element type of `x`. A write through it is
a value write: it activates the coordinates it makes nonzero and removes none,
and converts the value as a write into a plain vector does, so a complex value
throws an `InexactError`, possibly after part of `x` has been written; the view
has no complex check of its own. Its products and reductions are those of an
ordinary vector (IEEE, `O(length(x))`): for the sparse product use `x` itself, as
in `dot(v, x)`. `copyto!(coordinates(x), coordinates(y))` is a value write, while
`copyto!(x, y)` is the state copy. A `copyto!`, `copy!` or broadcast that takes both `x` and
`coordinates(x)` copies the source once first, so that its result is that of copying
the source before writing; other calls, such as `mul!` with the view as both output and
input, behave as they do on an aliased plain vector.
"""
function coordinates end

"""
    activevalues(x::Union{SparseIterate, SymmetricSparseIterate})

A writable vector view of the values of the active coordinates of `x`, in the
order of [`activecoordinates`](@ref): entry `a` is the value of coordinate
`activecoordinates(x)[a]`. Writing zero through it keeps the coordinate active;
use [`dropzeros!`](@ref) to remove it. The view has the length [`nactive`](@ref)
had when it was taken and does not grow, so, like `activecoordinates`, take it
again after a coordinate becomes active, after `dropzeros!` or after a
whole-state replacement such as `copyto!(x, y)`, each of which can reorder the
active list. Nothing is copied.
"""
function activevalues end

ncoordinates(x::_StoredIterate) = length(x.ind2nzval)
nactive(x::_StoredIterate) = x.nnz
activecoordinates(x::_StoredIterate) = _ActiveIDs(view(x.nzval2ind, 1:x.nnz))
activevalues(x::_StoredIterate) = view(x.nzval, 1:x.nnz)
isactive(x::_StoredIterate, k::Integer) = x.ind2nzval[k] != 0

# aliasing hooks from Base (non-public functions); the owner reports its storage
# arrays, so that activevalues(x), activecoordinates(x) and the views of x are all
# seen to overlap x
Base.dataids(x::SparseIterate) =
  (Base.dataids(x.nzval)..., Base.dataids(x.nzval2ind)..., Base.dataids(x.ind2nzval)...)

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

# Whether converting values of type S to T can throw. It cannot when S is a
# subtype of T, nor between built-in real types where T is a floating-point type
# (values round, and Float64 to Float32 overflows to Inf); any other pair is checked.
const _BuiltinFloat = Union{Float16, Float32, Float64}
const _BuiltinReal = Union{_BuiltinFloat, Bool, Int8, Int16, Int32, Int64, Int128,
                           UInt8, UInt16, UInt32, UInt64, UInt128}
_convert_may_throw(::Type{T}, ::Type{S}) where {T, S} =
    !(S <: T || (isconcretetype(T) && T <: _BuiltinFloat && isconcretetype(S) && S <: _BuiltinReal))

# The stored values of y as the destination's element type. A fallible conversion
# runs once, here, before a copy writes anything, so a value that does not convert
# throws with the destination unchanged; the copy then reads the converted buffer.
_converted_values(::Type{T}, y::_StoredIterate{S}) where {T, S} =
    _convert_may_throw(T, S) ? T[convert(T, y.nzval[k]) for k = 1:y.nnz] : y.nzval

# Copy the stored entries of y into x, an iterate of the same size. When both are
# sparse, touch only their stored entries; otherwise copying whole arrays is faster.
function _copyto_stored!(x::_StoredIterate{T}, y::_StoredIterate) where {T}
    vals = _converted_values(T, y)
    if 4 * (x.nnz + y.nnz) < length(x.ind2nzval)
        @inbounds for k = 1:x.nnz
            x.ind2nzval[x.nzval2ind[k]] = 0
        end
        @inbounds for k = 1:y.nnz
            i = y.nzval2ind[k]
            x.nzval[k] = vals[k]
            x.nzval2ind[k] = i
            x.ind2nzval[i] = k
        end
    else
        # only the stored values: entries past nnz are stale and may not convert
        copyto!(x.nzval, 1, vals, 1, y.nnz)
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

# These fallback methods serve SymmetricSparseIterate and scale stored entries
# in O(nnz). Generic methods would scale mirrored entries twice because they share
# storage. For scalars that map zero to a nonzero value (Inf, NaN, a zero divisor),
# the result depends on storage until #391 changes the owner's scaling.
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


# Scaling a SparseIterate is a value write equal to its broadcast in values,
# working set and slot order. It costs O(nnz) when the scalar maps zero to zero,
# and O(n) otherwise. The operation is assumed to be ordinary, pure arithmetic.
# A failed conversion throws the broadcast's exception type and leaves a consistent
# iterate, but partial writes need not match the broadcast: stored entries are
# visited in slot order. If the unstored value cannot convert, nothing is written.
function _scale!(x::SparseIterate{T}, s::Number, op::F) where {T, F}
  _checkreal(x, s)
  if _may_activate(x, s, op) && x.nnz < length(x.ind2nzval)
    z = convert(T, op(zero(T), s))
    iszero(z) || return _scale_all!(x, s, op, z)
  end
  @inbounds for k = 1:x.nnz
    x.nzval[k] = op(x.nzval[k], s)
  end
  x
end

# Whether the scalar can activate an unstored coordinate, tested before the working
# set is read: the fast path then costs one scalar test (#427). With IEEE entries and
# an IEEE scalar, op(zero(T), s) cannot throw, and a zero result converts to zero.
# Other pairs keep the full test, as op(zero(T), s) may throw where a fully stored
# iterate scales fine: (0//1) / 0 does, and so may a user-defined Real.
_may_activate(::SparseIterate{T}, s::Base.IEEEFloat, op::F) where {T <: Base.IEEEFloat, F} = !iszero(op(zero(T), s))
_may_activate(::SparseIterate, s::Number, op::F) where {F} = true

# The O(n) path, kept out of line: inlined, its stores give every call of _scale!,
# the O(nnz) path included, a GC frame (#427).
@noinline function _scale_all!(x::SparseIterate{T}, s::Number, op::F, z::T) where {T, F}
  @inbounds for k = 1:x.nnz
    x.nzval[k] = op(x.nzval[k], s)
  end
  @inbounds for i = 1:length(x.ind2nzval)
    x.ind2nzval[i] == 0 && _store!(x, i, z)
  end
  x
end

LinearAlgebra.rmul!(x::SparseIterate, s::Number) = _scale!(x, s, *)
LinearAlgebra.lmul!(s::Number, x::SparseIterate) = _scale!(x, s, (v, s) -> s * v)
LinearAlgebra.rdiv!(x::SparseIterate, s::Number) = _scale!(x, s, /)
LinearAlgebra.ldiv!(s::Number, x::SparseIterate) = _scale!(x, s, (v, s) -> s \ v)


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
@static if isdefined(@__MODULE__, :_AbstractSparseMatrixCSC)
  function Base.copyto!(x::SparseIterate{<:Any, 2}, y::_AbstractSparseMatrixCSC)
    _checkreal_broadcast(x, y)
    invoke(copyto!, Tuple{AbstractMatrix, _AbstractSparseMatrixCSC}, x, y)
  end
  function Base.copyto!(x::SparseIterate{T, 2}, y::Union{Adjoint{<:Any, <:_AbstractSparseMatrixCSC},
                                                         Transpose{<:Any, <:_AbstractSparseMatrixCSC}}) where {T}
    _checkreal_broadcast(x, y)
    invoke(copyto!, Tuple{AbstractMatrix{T}, typeof(y)}, x, y)
  end
end
# Newer SparseArrays also specializes matrix views and transposed sparse vectors.
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
@static if isdefined(@__MODULE__, :_CHOLMODDense)
  function Base.copyto!(x::SparseIterate, y::_CHOLMODDense)
    _checkreal_broadcast(x, y)
    invoke(copyto!, Tuple{AbstractArray, _CHOLMODDense}, x, y)
  end
  @static if isdefined(@__MODULE__, :_CHOLMODVTypes)
    function Base.copyto!(x::SparseIterate{T}, y::_CHOLMODDense{T}) where {T<:_CHOLMODVTypes}
      _checkreal_broadcast(x, y)
      invoke(copyto!, Tuple{AbstractArray{T}, _CHOLMODDense{T}}, x, y)
    end
    function Base.copyto!(x::SparseIterate{T, 2}, y::_CHOLMODDense{T}) where {T<:_CHOLMODVTypes}
      _checkreal_broadcast(x, y)
      invoke(copyto!, Tuple{AbstractArray{T, 2}, _CHOLMODDense{T}}, x, y)
    end
  end
end
# I and orthogonal factors are not arrays, so their element type is checked here.
function Base.copyto!(x::SparseIterate{<:Any, 2}, y::UniformScaling)
  !(eltype(y) <: Complex) || _real_error(:SparseIterate)
  invoke(copyto!, Tuple{AbstractMatrix, UniformScaling}, x, y)
end
@static if isdefined(@__MODULE__, :_AbstractQ)
  function Base.copyto!(x::SparseIterate{<:Any, 2}, y::_AbstractQ)
    !(eltype(y) <: Complex) || _real_error(:SparseIterate)
    invoke(copyto!, Tuple{AbstractMatrix, _AbstractQ}, x, y)
  end
end

function Base.copyto!(x::SparseIterate, bc::Broadcast.Broadcasted{Nothing})
  _checkreal_broadcast(x, bc)
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

function Base.copyto!(x::_StoredIterate, bc::Broadcast.Broadcasted{S}) where {S<:Broadcast.DefaultArrayStyle{0}}
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

# In the products and inner products below, a zero coefficient of the iterate,
# stored or not, contributes nothing, even against ±Inf or NaN in the other
# operand (numerics M1): each kernel tests the stored value, before any scaling
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
  _checkreal(coef, out); _checkreal(coef, A); _checkreal(coef, α); _checkreal(coef, β)
  length(coef) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), length of coef is $(length(coef))"))
  length(out) == size(A, 1) || throw(DimensionMismatch("A has first dimension $(size(A,1)), length of out is $(length(out))"))

  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, coef, α)
  @inbounds for inz = 1:nnz(coef)
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
  _checkreal(coef, out); _checkreal(coef, A); _checkreal(coef, α); _checkreal(coef, β)
  size(coef, 1) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), coef has $(size(coef, 1)) rows"))
  size(out) == (size(A, 1), size(coef, 2)) || throw(DimensionMismatch("out has size $(size(out))"))
  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, coef, α)
  # Stored-entry traversal wins for few coefficients; column order reads A
  # sequentially and matches LinearAlgebra's generic loop for denser coefficients.
  # The crossover density falls as columns of A get longer: for Float64 and a
  # random entry order its median is 7-16% for 64 to 512 rows, 3-5% for 1024,
  # about 1.5-2% for 2048 and 1-5% for 4096, on Apple M3, AMD Zen 3 and Intel
  # Skylake-SP (larger for Float32), and it differs by a factor of two or more
  # between those CPUs. No constant is best everywhere: 1/32 costs 2.5-5% of the
  # time on average over densities of 0.5-50%, the best constant (1/16) 1-3%.
  # A cutoff in bytes, min(1/8, 256/(m*sizeof(T))), cost 0.6-1.5%; that rule was
  # not adopted because its gain over the best constant was under 2 points.
  # See "Matrix coefficients" in docs/src/iterates.md.
  if 32 * nnz(coef) <= length(coef)
    p = size(coef, 1)
    @inbounds for inz = 1:nnz(coef)
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
  _checkreal(coef, out); _checkreal(coef, trans_a); _checkreal(coef, α); _checkreal(coef, β)
    A = trans_a.parent
    length(coef) == size(A, 1) || throw(DimensionMismatch("A has first dimension $(size(A,1)), length of coef is $(length(coef))"))
    length(out) == size(A, 2) || throw(DimensionMismatch("A has second dimension $(size(A,2)), length of out is $(length(out))"))

    iszero(α) && return _scale_product_output!(out, β)
    # No nonzero coefficient: the sum is empty, and out becomes β·out as on the strided
    # path, also for an α of Inf or NaN. The test stops at the first nonzero coefficient.
    any(!iszero, view(coef.nzval, 1:nnz(coef))) || return _scale_product_output!(out, β)
    scale = _product_scale(A, coef, α)
    @inbounds for j = 1:size(A, 2)
        s = zero(eltype(out))
        @simd for inz = 1:nnz(coef)
            v = coef.nzval[inz]
            t = A[coef.nzval2ind[inz], j] * v
            s += ifelse(iszero(v), zero(t), t)    # a select, not a branch, keeps the loop vectorized
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
    nnz(x) > nnz(y) && return dot(y, x)
    s = zero(promote_type(eltype(x), eltype(y)))
    @inbounds for k = 1:nnz(x)
        a, b = x.nzval[k], y[x.nzval2ind[k]]
        t = dot(a, b)
        s += ifelse(iszero(a) | iszero(b), zero(t), t)    # a select: a branch on the data mispredicts
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
    v = X.nzval[k]
    iszero(v) && continue
    i = X.nzval2ind[k]
    s += a[mod1(i, m)] * v * b[cld(i, m)]
  end
  s
end

# The scalar three-factor term for the product A * X * B. A NaN factor gives NaN, a zero factor
# against an infinite one gives NaN (the genuine 0 * Inf), a zero factor otherwise gives zero, and an
# infinite factor otherwise gives an infinity of the sign of the product. Three finite nonzero
# factors give a product with the accuracy of an ordinary product of three numbers: the result is
# within one neighbouring floating-point number of the correctly rounded exact product for a normal
# result, and within one spacing for a subnormal one, and a term within that margin of the overflow
# threshold or of half the smallest subnormal can fall on either side of it. Correct rounding is not
# promised. Overflow and underflow are decided by the final value, not by a partial product, beyond
# that margin. Only the exceptional entries of the product kernel reach it.
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
# that an owner method can call it once for each stored coefficient (and, for an off-diagonal
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
@inline _nstored(X::SparseIterate) = nnz(X)
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

The product `C = A * X * B` for a sparse iterate `X`, with `C[i, j]` the sum of
`A[i, k] * X[k, l] * B[l, j]` over the coordinates `(k, l)` where `X` is
mathematically nonzero, in `O(nnz(X) * m * n)` operations and `O(m * n)` memory for
`X` of size `m × n`. A stored zero and an unstored entry are the same: no term is formed,
and an `X` without nonzeros gives the zero matrix without reading `A` or `B`. A nonzero
coefficient of `X`, including `NaN` and `±Inf`, always contributes; the zeros of `A` and
`B` are ordinary zeros, so a nonzero `Inf` of `X` against a zero of `A` or `B` gives `NaN`.
The result need not equal `A * Matrix(X) * B`.

A term with three finite nonzero factors has the accuracy of an ordinary product of three
numbers: it differs from the correctly rounded exact product by at most one neighbouring
floating-point number for a normal result (a relative error below about `2 eps(T)`), and by at
most one spacing for a subnormal one. A term within that error of the overflow threshold or of
half the smallest subnormal can fall on either side of it, the neighbour of `floatmax(T)` being
`Inf` and the neighbour of zero being the smallest subnormal. Correct rounding is not promised.
Beyond that margin the class of a term is decided by its final value and not by the first partial
product, so `2^600 * 2^600 * 2^-600` is `2^600` and its underflow mirror is exact.

Terms are added in the storage order of `X` in the precision `T`, so tiny terms next to large ones
are rounded away, as in any floating-point sum. An entry whose ordinary sum is non-finite is
recomputed from its terms as a scaled sum when every factor of every nonzero coefficient at that
entry is finite, so a sum that overflows before a later cancellation gives its finite value, and a
sum beyond the range of `T` is `±Inf`. If any such factor is `NaN` or infinite the ordinary
IEEE value stands. A sum that is exactly zero is `+0.0`.

The guarantee covers `Float32` and `Float64` with `Symmetric` wrappers of strided matrices
of the same element type as `X`. Other element types, and mixed element types, are outside
the guarantee and take other methods of `*`. See [Numerical contract](@ref numerics-storage).
"""
function *(A::Symmetric{T, <:StridedMatrix{T}}, X::SparseIterate{T, 2},
           B::Symmetric{T, <:StridedMatrix{T}}) where {T<:Union{Float32, Float64}}
  _checkreal(X, A); _checkreal(X, B)
  m, n = size(X)
  size(A, 2) == m && size(B, 1) == n || throw(DimensionMismatch())
  C = zeros(T, m, n)
  a, ax, b = Vector{T}(undef, m), Vector{T}(undef, m), Vector{T}(undef, n)
  for k = 1:nnz(X)
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
`Float32` and `Float64` (and, through `SymmetricSparseIterate`'s own method, of an owner of such a
type); the guarantee of the direct methods does not cover it.
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
# aliasing hooks from Base (non-public functions); the owner reports its storage
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

!!! warning "Non-finite vectors"
    The product skips absent coordinates but multiplies by stored ones, an
    explicit stored zero included. With a non-finite `v`, a stored zero gives
    `0 * Inf = NaN` where an absent coordinate gives `0`, so `dropzeros!(x)` can
    change `mul!(y, matrixview(x), v)` although the matrix does not. This does
    not yet satisfy the rule of the 0.6 contract (the same result, or the same
    error, before and after `dropzeros!`); it is to be settled with the 0.6 owner.

Entries `(r, c)` and `(c, r)` are one variable of `x`; a coordinate of `x` is
reached through [`coordinates`](@ref), not through this view, so
`matrixview(x)[4]` is a matrix linear index.
"""
matrixview(x::SymmetricSparseIterate) = SymmetricMatrixView(x)

Base.size(A::SymmetricMatrixView) = size(A.x)
Base.IndexStyle(::Type{<:SymmetricMatrixView}) = IndexCartesian()
Base.@propagate_inbounds Base.getindex(A::SymmetricMatrixView, r::Int, c::Int) = A.x[r, c]
# aliasing hooks from Base (non-public functions)
Base.dataids(A::SymmetricMatrixView) = Base.dataids(A.x)
Base.unaliascopy(A::SymmetricMatrixView) = SymmetricMatrixView(copy(A.x))
Base.transpose(A::SymmetricMatrixView) = A
Base.adjoint(A::SymmetricMatrixView{<:Real}) = A
Base.Matrix(A::SymmetricMatrixView) = Matrix(A.x)
Base.Array(A::SymmetricMatrixView) = Matrix(A.x)
SparseArrays.sparse(A::SymmetricMatrixView) = sparse(A.x)

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
_symmetric_by_type(::Union{SymmetricSparseIterate, Symmetric{<:Number}, Hermitian{<:Real},
                           Diagonal{<:Number}, SymTridiagonal{<:Number}}) = true
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
      acc += _product3(a, T(S[i, j]), T(Bm[j, k]))
    end
    Cm[i, k] = iszero(β) ? acc : acc + β * Cm[i, k]
  end
  C
end
# a * b * c rounded once (up to the product of the significands) for finite factors
@inline function _product3(a::T, b::T, c::T) where {T<:AbstractFloat}
  (iszero(a) || iszero(b) || iszero(c)) && return zero(T)
  (isfinite(a) && isfinite(b) && isfinite(c)) || return a * b * c
  fa, ea = _frexp(a); fb, eb = _frexp(b); fc, ec = _frexp(c)
  ldexp(fa * fb * fc, _exponent_sum(T, ea, eb, ec))
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
# A spectral function is defined on symmetric matrices, so its proximal map reads the
# symmetric part of a matrix that is not symmetric by type, whatever the output.
_spectral_argument(x) = x
# A device matrix is read on the host, as the decompositions do.
_spectral_argument(x::AbstractMatrix) =
  _symmetric_by_type(x) ? x : _SymmetricPart(_is_device_array(x) ? Array(x) : x)
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

# A check of one pair, such as the coordinate of prox_coordinate!, reads only that
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
  owner = f === nothing ? "the steps γ" : "the array $name of $(nameof(typeof(f)))"
  label = f === nothing ? "γ" : string(name)
  a isa AbstractMatrix && size(a) == (p, p) || (label = "reshape($label, $p, $p)")
  where = " on a SymmetricSparseIterate"
  r == 0 && _throw_asymmetric(reshape(a, p, p), owner, label, where)
  _throw_asymmetric_pair(owner, label, where, r, c, a[(c - 1) * p + r], a[(r - 1) * p + c])
end

# A, which may live on a device and not allow scalar indexing: checked as a host copy
# in its own element type, so that a mismatch cannot vanish in a promotion
_host(A::Union{Matrix, SparseMatrixCSC}) = A
_host(A::AbstractMatrix) = Matrix{eltype(A)}(A)
_check_symmetric_host(A::AbstractMatrix, subject, name) =
  _symmetric_by_type(A) || _check_symmetric(_host(A), subject, name)

_check_axes(x::SymmetricSparseIterate, A::AbstractArray) = axes(A) == axes(x) || _throw_axes(x, A)
@noinline _throw_axes(x, A) =
  throw(DimensionMismatch("source axes $(axes(A)) do not match destination axes $(axes(x))"))

# B, a product such as U Σ Vᵀ whose triangles agree only up to rounding when its
# input is symmetric: a SymmetricSparseIterate keeps the lower one
_copyto_rounded!(Y, B) = copyto!(Y, B)
_copyto_rounded!(Y::SymmetricSparseIterate, B) = copyto!(Y, Symmetric(B, :L))

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
      _QRCompactWYQ, _AdjointQ{<:Any,<:_QRCompactWYQ},
      _LQPackedQ, _AdjointQ{<:Any,<:_LQPackedQ},
      _HessenbergQ, _AdjointQ{<:Any,<:_HessenbergQ},
      _Rotation, _AdjointRotation{<:Any,<:_Rotation}, LinearAlgebra.Givens)),
    (:rmul!, (Diagonal, Bidiagonal, LinearAlgebra.AbstractTriangular,
      Union{LowerTriangular{T,S}, UpperTriangular{T,S}, UnitLowerTriangular{T,S},
            UnitUpperTriangular{T,S}} where {T,S<:AbstractMatrix{T}},
      _QRPackedQ, _AdjointQ{<:Any,<:_QRPackedQ},
      _QRCompactWYQ, _AdjointQ{<:Any,<:_QRCompactWYQ},
      _LQPackedQ, _AdjointQ{<:Any,<:_LQPackedQ},
      _HessenbergQ, _AdjointQ{<:Any,<:_HessenbergQ},
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

# copyto! through a full view or reshape uses the checked matrix writer. As in
# Base, the source is read in linear order and needs the length, not the axes, of
# the destination. A source with the axes of x keeps its own copyto! method.
const _SymmetricArrayDest{T} = Union{_SymmetricView{T}, _SymmetricReshape{T}, _SymmetricReshapeView{T}}
const _SymmetricMatrixDest{T} = Union{_SymmetricView{T, 2}, _SymmetricReshape{T, 2}, _SymmetricReshapeView{T, 2}}
function _copyto_symmetric!(dest, src)
  x = _checked_symmetric_parent(dest, "copyto!")
  length(src) == length(dest) || throw(DimensionMismatch(
    "copyto! source has $(length(src)) values and the destination has $(length(dest))"))
  _copyto_reshaped!(x, src)
  dest
end
_copyto_reshaped!(x, src) = axes(src) == axes(x) ? copyto!(x, src) : copyto!(x, reshape(src, size(x)))
Base.copyto!(dest::_SymmetricBroadcastDest, src::AbstractArray) = _copyto_symmetric!(dest, src)
# Resolve the sparse-source and factorization ambiguities, as for x itself.
@static if isdefined(@__MODULE__, :_AbstractSparseMatrixCSC)
  Base.copyto!(dest::_SymmetricMatrixDest, src::_AbstractSparseMatrixCSC) = _copyto_symmetric!(dest, src)
  Base.copyto!(dest::_SymmetricMatrixDest, src::Union{Adjoint{<:Any,<:_AbstractSparseMatrixCSC},
                                                    Transpose{<:Any,<:_AbstractSparseMatrixCSC}}) =
    _copyto_symmetric!(dest, src)
end
@static if isdefined(@__MODULE__, :_SparseMatrixCSCOrView)
  Base.copyto!(dest::_SymmetricMatrixDest, src::_SparseMatrixCSCOrView) = _copyto_symmetric!(dest, src)
end
@static if isdefined(@__MODULE__, :_AdjOrTransSparseVectorOrView)
  Base.copyto!(dest::_SymmetricMatrixDest, src::_AdjOrTransSparseVectorOrView) = _copyto_symmetric!(dest, src)
end
@static if isdefined(@__MODULE__, :_CHOLMODDense)
  Base.copyto!(dest::_SymmetricBroadcastDest, src::_CHOLMODDense) = _copyto_symmetric!(dest, src)
  @static if isdefined(@__MODULE__, :_CHOLMODVTypes)
    Base.copyto!(dest::_SymmetricArrayDest{T}, src::_CHOLMODDense{T}) where
        {T<:_CHOLMODVTypes} = _copyto_symmetric!(dest, src)
    Base.copyto!(dest::_SymmetricMatrixDest{T}, src::_CHOLMODDense{T}) where
        {T<:_CHOLMODVTypes} = _copyto_symmetric!(dest, src)
  end
end
# LinearAlgebra writes an identity before applying Q, so form the value in a buffer.
function Base.copyto!(dest::_SymmetricMatrixDest, Q::_AbstractQ)
  x = _checked_symmetric_parent(dest, "copyto!")
  A = Matrix{promote_type(eltype(dest), eltype(Q))}(I, size(dest))
  lmul!(Q, A)
  _copyto_reshaped!(x, A)
  dest
end
# Offsets copy part of the values, even when they cover every position.
Base.copyto!(::_SymmetricBroadcastDest, ::Integer, ::AbstractArray) = _throw_partial("copyto! with offsets")
Base.copyto!(::_SymmetricBroadcastDest, ::Integer, ::AbstractArray, ::Integer) = _throw_partial("copyto! with offsets")
Base.copyto!(::_SymmetricBroadcastDest, ::Integer, ::AbstractArray, ::Integer, ::Integer) =
  _throw_partial("copyto! with offsets")
Base.copyto!(::_SymmetricBroadcastDest, ::Integer, src) = _throw_partial("copyto! with offsets")
Base.copyto!(::_SymmetricBroadcastDest, ::Integer, src, ::Integer) = _throw_partial("copyto! with offsets")
Base.copyto!(::_SymmetricBroadcastDest, ::Integer, src, ::Integer, ::Integer) = _throw_partial("copyto! with offsets")

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
      _symequal(a, b) || _throw_asymmetric([bc′[CartesianIndex(i, j)] for i in 1:p, j in 1:p],
                                           "the value written into a SymmetricSparseIterate", "A")
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
  _issymmetric_exact(P, _symequal) ||
    _throw_asymmetric(P, "the product A * B written into a SymmetricSparseIterate", "(A * B)")
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


# checked before C is modified: C may be activevalues(A), or coordinates(A) when p == 1
function _checkalias(C, A::SymmetricSparseIterate, B)
  Base.mightalias(C, B) && throw(ArgumentError("the output of mul! must not be aliased with its input"))
  Base.mightalias(C, A) && throw(ArgumentError("the output of mul! must not be aliased with the matrix, e.g. coordinates(X) or activevalues(X)"))
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
  _mul_stored!(C, A, B, α, β)
end

# C = α A B + β C over the stored entries of A, O(p + nactive(A)) and no allocation.
# As in BLAS, C is not read when β is zero and B is not read when α is zero.
# A stored explicit zero still multiplies B, so with a non-finite B the product
# can depend on whether a zero is stored (see the documentation of matrixview).
function _mul_stored!(C::AbstractVector, A::SymmetricSparseIterate, B::AbstractVector, α::Number, β::Number)
  p = A.p
  length(B) == p || throw(DimensionMismatch("A has $p columns but B has length $(length(B))"))
  length(C) == p || throw(DimensionMismatch("A has $p rows but C has length $(length(C))"))
  _checkalias(C, A, B)
  iszero(β) ? fill!(C, zero(eltype(C))) : isone(β) || rmul!(C, β)
  iszero(α) && return C
  @inbounds for k in 1:A.nnz
    r, c = tril_cartesian(p, A.nzval2ind[k])
    v = α * A.nzval[k]
    C[r] += B[c] * v
    r == c || (C[c] += B[r] * v)
  end
  C
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
`p`. See [Numerical contract](@ref numerics-storage).
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
