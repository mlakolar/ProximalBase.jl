"""
    SparseIterate([T = Float64,] n)
    SparseIterate([T = Float64,] n, m)
    SparseIterate(x::Union{SparseVector, SparseMatrixCSC, Array})

A real vector or matrix with a working set of explicitly stored coordinates.
Each coordinate is a Julia linear entry index. Reading or writing an entry
costs O(1); reserved storage costs O(length(x)), even when few entries are
active.

A nonzero write activates its coordinate. A zero write preserves existing
activity; [`activate!`](@ref) stores an explicit zero and [`dropzeros!`](@ref)
removes active zeros. [`nactive`](@ref) counts stored coordinates, including
zeros. [`coordinates`](@ref) exposes all entries, while
[`activecoordinates`](@ref) and [`activevalues`](@ref) expose the working set.

# Examples
```jldoctest
julia> x = SparseIterate(4); x[2] = 3; activate!(x, 4);

julia> (ncoordinates(x), nactive(x), x[4])
(4, 2, 0.0)

julia> x[2] = 0; dropzeros!(x); nactive(x)
0
```

# Write semantics
- `copyto!(x, y)` replaces working-set state when `y` is a same-size
  `SparseIterate`; ordinary array copies and broadcasts write values.
- Direct bulk writes reject known complex source types before writing.
  Views, indexed writes and per-element conversions can fail after partial
  writes. See [write validation](@ref iterate-write-validation).
- Only documented product methods traverse active entries; views use ordinary
  array operations.

See the [storage guide](@ref sparse-iterate-guide) for copies, operation costs
and [writing from an orthogonal factor](@ref sparse-orthogonal-copy).
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
    SymmetricSparseIterate(A::AbstractMatrix)

Store the ``p(p+1)/2`` independent variables of a real symmetric ``p × p``
matrix, with O(1) access and O(p²) reserved storage.

This object is not an `AbstractArray`. Read or write a matrix entry with
`x[i,j]`; both triangles address the same variable. Use [`coordinates`](@ref)
for a writable lower-triangle vector in column-major order,
[`matrixview`](@ref) for a read-only live matrix, or `Matrix(x)` for an
independent dense copy. `size(x)` is `(p,p)`; there is no `length`, `axes`, iteration,
broadcasting or one-index `x[k]` access.

Nonzero writes activate a coordinate; zero writes retain existing activity.
[`activate!`](@ref) stores an explicit zero, [`nactive`](@ref) counts active
independent variables, and [`dropzeros!`](@ref) removes active zeros.

# Examples
```jldoctest
julia> x = SymmetricSparseIterate(2); x[2, 1] = 3;

julia> (ncoordinates(x), nactive(x), x[1, 2])
(3, 1, 3.0)

julia> Matrix(x) == [0.0 3.0; 3.0 0.0]
true
```

# Copies and imports
- `copy(x)` and same-kind `copyto!` preserve values, active zeros and order;
  `similar(x)` creates empty storage of the same dimension.
- A matrix import must be real, one-based and exactly symmetric before
  conversion. It replaces activity with converted nonzeros. Use
  `Symmetric(A, :L)` to select one triangle deliberately.
- `copyto!(x, A)` rejects a source sharing memory with `x`. Shape, symmetry
  and alias checks precede writes; a failing conversion can partially write.

See the [symmetric-iterate guide](@ref symmetric-iterate-guide) for comparisons,
matrix geometry and supported operations, and
[`ProximalBase.check_symmetric_parameters`](@ref) for model validation.
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

# Range recovery reads stored entries only; inactive coordinates contribute zero.
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

Make a coordinate active and return `x`, without changing its value. An absent
scalar coordinate is stored as an explicit zero. For a [`BlockIterate`](@ref),
`k` is a group ID and the group's entries are kept. An already active
coordinate is unchanged.

Assigning zero to an absent coordinate does not activate it. Use this operation
to retain zero coordinates for a solver's next sweep; [`dropzeros!`](@ref)
removes them when they are no longer needed.

# Examples
```jldoctest
julia> x = SparseIterate(3); activate!(x, 2);

julia> (x[2], isactive(x, 2), nactive(x))
(0.0, true, 1)
```
"""
function activate!(x::_StoredIterate{T}, i::Integer) where {T}
  checkbounds(x.ind2nzval, i)
  x.ind2nzval[i] == 0 && _store!(x, i, zero(T))
  x
end

"""
    dropzeros!(x::Union{SparseIterate, SymmetricSparseIterate})

Remove active zero coordinates and return `x`, preserving all mathematical
values. For a [`SymmetricSparseIterate`](@ref), coordinates are independent
lower-triangle variables. Signed zeros are removed; `NaN` and infinities are
kept.

Compaction may reorder active slots. Reacquire [`activecoordinates`](@ref) and
[`activevalues`](@ref) afterward; their old handles are not snapshots. The
cost is O(nactive(x)) before compaction.

# Examples
```jldoctest
julia> x = SparseIterate([0.0, 3.0]); activate!(x, 1);

julia> nactive(x)
2

julia> dropzeros!(x); (nactive(x), x[2])
(1, 3.0)
```
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


