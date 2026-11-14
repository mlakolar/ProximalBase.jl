"""
    SparseIterate([T = Float64,] n)
    SparseIterate([T = Float64,] n, m)
    SparseIterate(x::Union{SparseVector, SparseMatrixCSC, Array})

A vector or matrix that stores its nonzero entries in a compact list, so that
reading or writing any entry is O(1), with supported sparse-aware products
scaling with the number of active coordinates, `nactive(x)`.
Intended for coordinate-descent and active-set iterates that stay sparse. Its
coordinates (see `ncoordinates`) are its entries, in linear order.

Setting an entry to zero keeps it in the list; call `dropzeros!` to remove
stored zeros. The stored entries are the active coordinates
(`activecoordinates`).

Its entries are real. The constructors, two-argument `copyto!` from an array
(for a matrix also from a `UniformScaling` or an orthogonal factor), `copy!`,
broadcasting into `x` and `fill!` throw an `ArgumentError` for a source or value
of complex type, even with a zero imaginary part, before anything is written.
A broadcast into `x` checks its leaves and its result: a result known to be
complex from the types is rejected, and a result of another non-real type (a
`Union` with a complex member, `Any`, `Number`) is evaluated before writing and
rejected if an element is complex, with `x` unchanged.
For a vector `x`, `vec(x)` and `reshape(x, length(x))` are `x` itself and are
checked too. `copy!` from a source of another size throws Base's error first,
unless the source is a `SparseIterate` or a `BlockIterate`: one of the
same length is copied by value in linear order, and one of another length throws
a `DimensionMismatch`. The two-argument `copyto!` and `copy!` read types: a tuple, a generator or an
array of abstract element type such as `Vector{Any}` passes the check, and its
values are converted one by one. Other
writes also convert values one by one, as writes into a plain `Array` of the
element type of `x` do: writes through a view, through `coordinates(x)`, through
`vec` of a matrix or a
`reshape` to another shape, indexed assignment such as `x[i] = v` or `x[:] = c`,
`copyto!` with offsets or index ranges, and `map!`. A complex value converted
one by one throws an `InexactError`, possibly after part of `x` has been
written.

For a matrix, `copyto!(x, Q)` from an orthogonal or unitary factor `Q`, such as
`qr(A).Q`, `qr(A, ColumnNorm()).Q`, `lq(A).Q`, `hessenberg(A).Q` (types
`QRCompactWYQ`, `QRPackedQ`, `LQPackedQ`, `HessenbergQ`) or the adjoint of one,
takes `Q` to be a lazy factorization object, not a dense matrix. It materializes
`Q` into a dense temporary of the element type of `x` and then writes that matrix
as an ordinary array value write. A complex factor, a size other than `size(x)`,
a failure to materialize `Q` and a failing conversion (an `InexactError` for a
fractional value written into an integer `x`) throw before anything is written, so
`x` is unchanged. The write is
a value write, not a copy of a working-set state: an active coordinate whose new
value is zero stays active. The dense temporary
and its allocation are deliberate; a caller who needs speed can pass a
precomputed dense matrix of the full size `size(Q)`, such as
`lmul!(Q, Matrix{T}(I, size(Q)))` with `T = eltype(Q)`. `Matrix(Q)` can be thin,
for example for a tall QR factor (`qr(randn(4, 3)).Q` has size 4x4 while `Matrix(Q)`
is 4x3), whereas the write uses the full `size(Q)` matrix.
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

Symmetric matrix storage for the `q = p(p + 1) / 2` variables of a real symmetric `p × p` matrix:
the entries of its lower triangle, stored in a compact list with O(1) access, as
`SparseIterate`. It is not an array. It has `size(x) == (p, p)`,
`eltype(x)`, `ncoordinates``(x) == q`, and the two-index access below; it
has no `length`, `axes`, iteration, broadcasting or one-index access. Read it
through one of its views, or copy it:

  - `coordinates(x)`: a live, writable vector of the `q` variables, in the
    column-major order of the lower triangle (`tril_index`);
  - `matrixview(x)`: a live, read-only `p × p` matrix (`matrixview`);
  - `Matrix(x)`: an independent dense copy.

`x[i, j]` reads and `x[i, j] = v` writes one variable, from either triangle: the
positions `(i, j)` and `(j, i)` name the same variable, and an index outside
`1:p` throws a `BoundsError`. `x[k]` with one integer is an `ArgumentError`,
because `coordinates(x)[k]` and `matrixview(x)[k]` mean different things. A write
converts the value to `T` first, and stores a nonzero converted value (`NaN`
included) as an active coordinate; a zero written to a coordinate that is not
active does not activate it, and a zero written to an active coordinate keeps it
active. Use `activate!` for an explicit active zero and `dropzeros!` to
remove active zeros. `nactive` counts the active coordinates;
`count(!iszero, activevalues(x))` counts those that are nonzero.

  - `copy(x)` is an independent state: values, active zeros and activation order.
  - `similar(x)` and `similar(x, S)` are empty symmetric iterates of the same `p`.
  - `copyto!(y, x)` between symmetric iterates replaces the state of `y` with a copy of that
    of `x`; `copyto!(x, x)` does nothing.
  - `copyto!(x, A)` and `SymmetricSparseIterate(A)` import a matrix `A`: it must
    be real, of size `(p, p)` and exactly symmetric (`NaN` matches `NaN`, `0.0`
    matches `-0.0`), compared before conversion to `T`. The previous active set
    of `x` is replaced by the coordinates whose converted value is nonzero, in
    increasing coordinate order. `copyto!(x, A)` with a source that shares memory
    with `x`, such as `matrixview(x)`, throws an `ArgumentError`. To store one
    triangle of a matrix that is not symmetric, pass `Symmetric(A, :L)`.
  - `fill!(coordinates(x), 0)` writes only the active values; with `dropzeros!(x)`
    it empties `x` in O(`nactive(x)`). There is no `fill!(x, v)`.

Two symmetric iterates are `==` and `isequal` when their matrices are: `isequal(x, y)` is
`isequal(Matrix(x), Matrix(y))`, so an active `-0.0` differs from an absent
coordinate under `isequal` but not under `==`, and paired `NaN`s are `isequal`
but not `==`. Equal matrices have equal hashes whatever the active sets and their
order. There is no comparison with an array and no `isapprox`: use `Matrix(x)` or
`matrixview(x)`.

Its coordinates are numbered column by column (`tril_index`):
Public indexed assignment updates them, `activecoordinates` lists the
active ones and `coordinates` is a vector view of them.

See Symmetric parameters for which arrays must be symmetric and who validates them.
"""
mutable struct SymmetricSparseIterate{T}
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

function _scaled_dot(a, x::SparseIterate, s)
  _checkparam(a, x)
  v = zero(s[1])
  @inbounds @simd for k in 1:x.nnz
    v += float(_entry(a, x.nzval2ind[k])) * _scale_value(x.nzval[k], s)
  end
  v
end

# the stored entries of an iterate: the recoveries of R1 read only these
_stored_maxabs(x::Union{SparseIterate, SymmetricSparseIterate}) = _maximum_abs(view(x.nzval, 1:x.nnz))
_maximum_abs(x::SymmetricSparseIterate) = _maximum_abs(view(x.nzval, 1:x.nnz))
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
    nnz(A)

Re-export of `SparseArrays.nnz` for standard sparse arrays. ProximalBase
does not define an `nnz` method for its iterates. Use `nactive` to
count active coordinates, including explicitly stored zeros; use
`count(!iszero, activevalues(x))` to count nonzero entries of a `SparseIterate`.
"""
nnz

"""
    ncoordinates(x)

The number of independent coordinates of the iterate `x` that a solver
updates: the entries of a `SparseIterate`, the entries of the lower
triangle of a `SymmetricSparseIterate`, the groups of an
`BlockIterate`.
"""
function ncoordinates end

"""
    nactive(x)

The number of active coordinates of `x`: the stored entries of a
`SparseIterate` or `SymmetricSparseIterate`, the active groups
of a `BlockIterate`. Active-set solvers use them as the active set.
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
  throw(ArgumentError("activecoordinates(x) is read-only: write values through activevalues(x), or blockview(x, k) for a BlockIterate"))

"""
    activecoordinates(x)

A read-only `AbstractVector{Int}` of the active coordinates of `x` (see
`nactive`), in the order in which they became active, except that
`dropzeros!` moves the last ones into the places of those it removes. It is a
handle on the internal list, not a snapshot: nothing is copied, and it does not
grow when coordinates become active, so call `activecoordinates` again after
changing the active set: after an activation, after `dropzeros!`, and after a
whole-state replacement such as `copyto!(x, y)`, which can reorder the list even
when the active set and `nactive(x)` stay the same. Writing through it throws an `ArgumentError`, because
the list and the iterate's reverse map must stay in agreement; write values
through `activevalues` (or `blockview(x, k)` for a `BlockIterate`).
"""
function activecoordinates end

"""
    isactive(x, k)

Whether coordinate `k` of `x` is active (see `nactive`).
"""
function isactive end

"""
    coordinates(x)

A vector view of the coordinates of `x` (see `ncoordinates`): the
entries of a `SparseIterate` in linear order, whatever its shape, and the
entries of the lower triangle for a `SymmetricSparseIterate`. The view is
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
order of `activecoordinates`: entry `a` is the value of coordinate
`activecoordinates(x)[a]`. Writing zero through it keeps the coordinate active;
use `dropzeros!` to remove it. The view has the length `nactive`
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

# aliasing hooks from Base (non-public functions); the symmetric iterate reports its storage
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
    activate!(x::BlockIterate, k)

Make coordinate `i` of `x` active, storing it as an explicit zero if it is not
stored already, and return `x`. Assigning a zero does not store an entry, so
this is how a coordinate joins the active set of an active-set solver;
`dropzeros!` removes it again while it is zero. For a `BlockIterate`,
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
`SymmetricSparseIterate`, these are coordinates of the lower triangle.
When every stored value is zero, the working set is emptied. Otherwise the last
stored coordinate moves into the place of each removed one, so the order of
`activecoordinates` can change. The entries of `x` do not change.
"""
function SparseArrays.dropzeros!(x::_StoredIterate)
  nz, idx, pos = x.nzval, x.nzval2ind, x.ind2nzval
  n = x.nnz
  i = 1
  @inbounds while i <= n
    if iszero(nz[i])
      pos[idx[i]] = 0
      # stored zeros at the end leave without a move; the last nonzero takes the place
      while n > i && iszero(nz[n])
        pos[idx[n]] = 0
        n -= 1
      end
      if n > i
        nz[i] = nz[n]
        idx[i] = idx[n]
        pos[idx[i]] = i
        i += 1
      end
      n -= 1
    else
      i += 1
    end
  end
  x.nnz = n
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


