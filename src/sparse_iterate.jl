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
A broadcast into `x` checks its leaves and its result: a result known to be
complex from the types is rejected, and a result of another non-real type (a
`Union` with a complex member, `Any`, `Number`) is evaluated before writing and
rejected if an element is complex, with `x` unchanged.
For a vector `x`, `vec(x)` and `reshape(x, length(x))` are `x` itself and are
checked too. `copy!` from a source of another size throws Base's error first,
unless the source is a `SparseIterate` or an [`AtomIterate`](@ref): one of the
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

The owner of the `q = p(p + 1) / 2` variables of a real symmetric `p × p` matrix:
the entries of its lower triangle, stored in a compact list with O(1) access, as
[`SparseIterate`](@ref). It is not an array. It has `size(x) == (p, p)`,
`eltype(x)`, [`ncoordinates`](@ref)`(x) == q`, and the two-index access below; it
has no `length`, `axes`, iteration, broadcasting or one-index access. Read it
through one of its views, or copy it:

  - `coordinates(x)`: a live, writable vector of the `q` variables, in the
    column-major order of the lower triangle ([`tril_index`](@ref));
  - `matrixview(x)`: a live, read-only `p × p` matrix ([`matrixview`](@ref));
  - `Matrix(x)`: an independent dense copy.

`x[i, j]` reads and `x[i, j] = v` writes one variable, from either triangle: the
positions `(i, j)` and `(j, i)` name the same variable, and an index outside
`1:p` throws a `BoundsError`. `x[k]` with one integer is an `ArgumentError`,
because `coordinates(x)[k]` and `matrixview(x)[k]` mean different things. A write
converts the value to `T` first, and stores a nonzero converted value (`NaN`
included) as an active coordinate; a zero written to a coordinate that is not
active does not activate it, and a zero written to an active coordinate keeps it
active. Use [`activate!`](@ref) for an explicit active zero and `dropzeros!` to
remove active zeros. [`nactive`](@ref) counts the active coordinates;
`count(!iszero, activevalues(x))` counts those that are nonzero.

  - `copy(x)` is an independent state: values, active zeros and activation order.
  - `similar(x)` and `similar(x, S)` are empty owners of the same `p`.
  - `copyto!(y, x)` between owners replaces the state of `y` with a copy of that
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

Two owners are `==` and `isequal` when their matrices are: `isequal(x, y)` is
`isequal(Matrix(x), Matrix(y))`, so an active `-0.0` differs from an absent
coordinate under `isequal` but not under `==`, and paired `NaN`s are `isequal`
but not `==`. Equal matrices have equal hashes whatever the active sets and their
order. There is no comparison with an array and no `isapprox`: use `Matrix(x)` or
`matrixview(x)`.

Its coordinates are numbered column by column ([`tril_index`](@ref)):
[`prox_coordinate!`](@ref) updates them, [`activecoordinates`](@ref) lists the
active ones and [`coordinates`](@ref) is a vector view of them.

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
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
    nnz(x::SparseIterate)

The number of stored coordinates of `x`, including stored zeros. Use
[`dropzeros!`](@ref) to remove stored zeros. A [`SymmetricSparseIterate`](@ref)
has no `nnz`: [`nactive`](@ref)`(x)` counts its stored coordinates, including
stored zeros.
"""
SparseArrays.nnz(x::SparseIterate) = x.nnz

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
When every stored value is zero, the working set is emptied. Otherwise the last
stored coordinate moves into the place of each removed one, so the order of
[`activecoordinates`](@ref) can change. The entries of `x` do not change.
"""
SparseArrays.dropzeros!(x::_StoredIterate) = _dropzeros_compact!(x)

function SparseArrays.dropzeros!(x::SparseIterate)
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

function _dropzeros_compact!(x::_StoredIterate)
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
  # Q is a lazy factorization, so it is materialized into a dense matrix of the element
  # type of x first; the validation, the materialization and the conversion all run
  # before x is touched, and the result is written as an ordinary array value write.
  function Base.copyto!(x::SparseIterate{T, 2}, y::_AbstractQ) where {T}
    !(eltype(y) <: Complex) || _real_error(:SparseIterate)
    size(y) == size(x) || throw(DimensionMismatch(
      "destination has size $(size(x)), orthogonal factor has size $(size(y))"))
    # Matrix(y) can be thin (a tall QR factor), so the full size(y) matrix is built from the identity.
    staged = convert(Matrix{T}, lmul!(y, Matrix{eltype(y)}(I, size(y))))
    invoke(copyto!, Tuple{AbstractArray, AbstractArray}, x, staged)
    return x
  end
end

function Base.copyto!(x::SparseIterate, bc::Broadcast.Broadcasted{Nothing})
  _checkreal_broadcast(x, bc)
  _staged_result(bc) && return _staged_broadcast!(x, bc)
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

function Base.copyto!(x::SparseIterate, bc::Broadcast.Broadcasted{S}) where {S<:Broadcast.DefaultArrayStyle{0}}
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

Base.eltype(::Type{SymmetricSparseIterate{T}}) where {T} = T
Base.eltype(::SymmetricSparseIterate{T}) where {T} = T
Base.size(x::SymmetricSparseIterate) = (x.p, x.p)
# the shape of a matrix: later dimensions have size one, as for any matrix
Base.size(x::SymmetricSparseIterate, d::Integer) = d <= 2 ? size(x)[d] : 1

# the coordinate of entry (r, c)
@inline _coordinate(x::SymmetricSparseIterate, r::Integer, c::Integer) =
  r >= c ? tril_index(x.p, r, c) : tril_index(x.p, c, r)

Base.@propagate_inbounds function Base.getindex(x::SymmetricSparseIterate, r::Int, c::Int)
  @boundscheck _checkpair(x, r, c)
  @inbounds _getstored(x, _coordinate(x, r, c))
end
Base.@propagate_inbounds function Base.setindex!(x::SymmetricSparseIterate, v, r::Int, c::Int)
  @boundscheck _checkpair(x, r, c)
  @inbounds _setstored!(x, v, _coordinate(x, r, c))
end
# one unsigned comparison per index, combined without a branch between them
@inline _checkpair(x::SymmetricSparseIterate, r::Int, c::Int) =
  (((r - 1) % UInt < x.p % UInt) & ((c - 1) % UInt < x.p % UInt)) || _throw_pair(x, r, c)
@noinline _throw_pair(x::SymmetricSparseIterate, r, c) = throw(BoundsError(x, (r, c)))

# Other integer types: the bounds are checked in the type of the index, before it is narrowed to
# Int, so that an out-of-range index of any integer type is a BoundsError.
@inline function _int_pair(x::SymmetricSparseIterate, r::Integer, c::Integer)
  (1 <= r <= x.p && 1 <= c <= x.p) || _throw_pair(x, r, c)
  (Int(r), Int(c))
end
Base.@propagate_inbounds function Base.getindex(x::SymmetricSparseIterate, r::Integer, c::Integer)
  i, j = _int_pair(x, r, c)
  @inbounds x[i, j]
end
Base.@propagate_inbounds function Base.setindex!(x::SymmetricSparseIterate, v, r::Integer, c::Integer)
  i, j = _int_pair(x, r, c)
  @inbounds setindex!(x, v, i, j)
end

# x[k] has two natural readings, the coordinate and the matrix linear index
@noinline _throw_linear_index(x::SymmetricSparseIterate, k) = throw(ArgumentError(
  "a SymmetricSparseIterate has no one-index access: x[$k] could mean a coordinate or a matrix " *
  "linear index. Use x[i, j] for a matrix entry, coordinates(x)[$k] for variable $k of the lower " *
  "triangle, or matrixview(x)[$k] for entry $k of the matrix in column-major order"))
Base.getindex(x::SymmetricSparseIterate, k::Integer) = _throw_linear_index(x, k)
Base.setindex!(x::SymmetricSparseIterate, v, k::Integer) = _throw_linear_index(x, k)

# Base derives axes(x) from size(x) for any object that has a size. The owner has no axes,
# so that eachindex, broadcasting and checks of position lists are refused at once; its
# one-index access, length and iteration have no methods.
Base.axes(x::SymmetricSparseIterate) = throw(ArgumentError(
  "a SymmetricSparseIterate has no axes: use size(x), coordinates(x) for its variables, " *
  "or axes(matrixview(x)) for its matrix"))

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

# fill! of the vector of coordinates is a value write like any other (CRD-3): the value is
# converted first, and a zero reaches only the active coordinates, in O(nactive), so that
# fill!(coordinates(x), 0) followed by dropzeros!(x) empties x without a pass over all q
function Base.fill!(c::SymmetricCoordinates{T}, v) where {T}
  x = c.x
  s = convert(T, v)
  if iszero(s)
    @inbounds for k in 1:x.nnz
      x.nzval[k] = s
    end
  else
    for k in 1:length(x.ind2nzval)
      _setstored!(x, s, k)
    end
  end
  c
end

# a read-only p × p view of x: the matrix that x represents, both triangles
struct SymmetricMatrixView{T} <: AbstractMatrix{T}
  x::SymmetricSparseIterate{T}
end

# The matrix that an owner represents, written through by the kernels that address a position of it
# (the groups of a penalty, a sphere, a spectral decomposition) and store each result in the
# coordinate of its pair. Internal: the public view is read-only, and an owner is written through
# the methods that name it.
# A broadcast into it reads each coordinate from its lower position and expects operands that are symmetric:
# an operand that is not is read from its lower position, not averaged.
struct _OwnerMatrix{T} <: AbstractMatrix{T}
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

!!! note "Non-finite vectors"
    A coefficient of `x` that is zero contributes nothing to the product, whether
    it is absent or an explicitly stored zero: a non-finite entry of `v` that meets
    only zero coefficients does not reach the result, and `mul!(y, matrixview(x), v)`
    is the same before and after `dropzeros!(x)`, up to the order of the finite
    summation. A NaN coefficient is not zero and gives NaN.

Entries `(r, c)` and `(c, r)` are one variable of `x`; a coordinate of `x` is
reached through [`coordinates`](@ref), not through this view, so
`matrixview(x)[4]` is a matrix linear index.
"""
matrixview(x::SymmetricSparseIterate) = SymmetricMatrixView(x)

Base.size(A::SymmetricMatrixView) = size(A.x)
Base.IndexStyle(::Type{<:SymmetricMatrixView}) = IndexCartesian()
Base.@propagate_inbounds Base.getindex(A::SymmetricMatrixView, r::Int, c::Int) = A.x[r, c]

Base.size(A::_OwnerMatrix) = size(A.x)
Base.IndexStyle(::Type{<:_OwnerMatrix}) = IndexCartesian()
Base.@propagate_inbounds Base.getindex(A::_OwnerMatrix, r::Int, c::Int) = A.x[r, c]
Base.@propagate_inbounds function Base.setindex!(A::_OwnerMatrix, v, r::Int, c::Int)
  A.x[r, c] = v
  A
end
Base.dataids(A::_OwnerMatrix) = Base.dataids(A.x)
# a zero reaches the active coordinates only, as for fill!(coordinates(x), 0)
Base.fill!(A::_OwnerMatrix, v) = (fill!(coordinates(A.x), v); A)
# A broadcast into the matrix of an owner stores each coordinate once, from its lower position, so that an
# update such as y .= f.(y) is applied once; the operands are symmetric, as the caller has ensured.
function Base.copyto!(A::_OwnerMatrix, bc::Broadcast.Broadcasted{Nothing})
  axes(A) == axes(bc) || throw(DimensionMismatch("destination axes $(axes(A)) do not match $(axes(bc))"))
  bc′ = Broadcast.preprocess(A, bc)
  x = A.x
  p = x.p
  @inbounds for c = 1:p, r = c:p
    _setstored!(x, bc′[CartesianIndex(r, c)], tril_index(p, r, c))
  end
  A
end
# aliasing hooks from Base (non-public functions)
Base.dataids(A::SymmetricMatrixView) = Base.dataids(A.x)
Base.unaliascopy(A::SymmetricMatrixView) = SymmetricMatrixView(copy(A.x))
# A Symmetric or Hermitian wrapper of the view is copied through its parent, as a wrapper of the same
# type and the same triangle: copy of the wrapper has a dense parent, which Base.unalias refuses on
# Julia 1.10, where the copy must have the type of the original
Base.unaliascopy(A::Symmetric{<:Any, <:SymmetricMatrixView}) = Symmetric(Base.unaliascopy(A.data), Symbol(A.uplo))
Base.unaliascopy(A::Hermitian{<:Any, <:SymmetricMatrixView}) = Hermitian(Base.unaliascopy(A.data), Symbol(A.uplo))
# Base.unalias(x, A) of the owner x copies a source that shares memory with it, keeping the type of the
# source. Base does that through unaliascopy of each wrapper, and on Julia 1.10 it has no such copy for a
# Symmetric or Hermitian wrapper (a copy of the wrapper has a dense parent), so a wrapper of a view or a
# reshape of x, or a view of such a wrapper, was refused. The copy here goes through every derived parent,
# the wrapper types of Base and LinearAlgebra that the views of an owner can be in, and keeps the type, the
# triangle and the independence; any other source takes the copy of Base. The method is for the owner as the
# destination, so that it adds nothing to a function of Base for Base's types.
function Base.unalias(dest::SymmetricSparseIterate, A::AbstractArray)
  Base.mightalias(dest, A) ? _unaliascopy(A) : A
end
Base.unalias(::SymmetricSparseIterate, A::AbstractRange) = A
_unaliascopy(A) = Base.unaliascopy(A)
_unaliascopy(A::Symmetric) = Symmetric(_unaliascopy(A.data), Symbol(A.uplo))
_unaliascopy(A::Hermitian) = Hermitian(_unaliascopy(A.data), Symbol(A.uplo))
_unaliascopy(A::SubArray) =
  typeof(A)(_unaliascopy(A.parent), map(Base.unaliascopy, A.indices), A.offset1, A.stride1)
_unaliascopy(A::Base.ReshapedArray) = typeof(A)(_unaliascopy(A.parent), A.dims, A.mi)
_unaliascopy(A::Transpose) = Transpose(_unaliascopy(A.parent))
_unaliascopy(A::Adjoint) = Adjoint(_unaliascopy(A.parent))
Base.transpose(A::SymmetricMatrixView) = A
Base.adjoint(A::SymmetricMatrixView{<:Real}) = A
Base.Matrix(A::SymmetricMatrixView) = Matrix(A.x)
Base.Array(A::SymmetricMatrixView) = Matrix(A.x)

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
_symmetric_by_type(::Union{SymmetricSparseIterate, SymmetricMatrixView, _OwnerMatrix, Symmetric{<:Number},
                           Hermitian{<:Real}, Diagonal{<:Number}, SymTridiagonal{<:Number}}) = true
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
_symmetric_argument(y::_OwnerMatrix, x::AbstractMatrix) = _symmetric_argument(y.x, x)
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

# the axes of the matrix that x represents
_owner_axes(x::SymmetricSparseIterate) = (Base.OneTo(x.p), Base.OneTo(x.p))
_check_axes(x::SymmetricSparseIterate, A::AbstractArray) = axes(A) == _owner_axes(x) || _throw_axes(x, A)
@noinline _throw_axes(x, A) =
  throw(DimensionMismatch("source axes $(axes(A)) do not match destination axes $(_owner_axes(x))"))

# B, a product such as U Σ Vᵀ whose triangles agree only up to rounding when its
# input is symmetric: a SymmetricSparseIterate keeps the lower one
_copyto_rounded!(Y, B) = copyto!(Y, B)
_copyto_rounded!(Y::SymmetricSparseIterate, B) = copyto!(Y, Symmetric(B, :L))
# the lower triangle of B, one coordinate at a time: a proximal value write keeps the coordinates that are active
_copyto_rounded!(Y::_OwnerMatrix, B) = (_store_lower!(Y.x, B); Y)

# The destination, and the argument, as the kernels that write through the matrix of an owner see them: x is
# the destination itself, or its read-only view, when the call is in place, and the matrix of x otherwise.
_owner_point(::_OwnerMatrix, x) = x
_owner_point(w::_OwnerMatrix, x::SymmetricSparseIterate) = x === w.x ? w : matrixview(x)
_owner_point(w::_OwnerMatrix, x::SymmetricMatrixView) = x.x === w.x ? w : x

Base.convert(::Type{SymmetricSparseIterate{T}}, ::Union{AbstractArray, SymmetricSparseIterate}) where {T<:Complex} =
  _real_error(:SymmetricSparseIterate)

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
  for k=1:S.nnz
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

# a parameter array has an entry for every position of the matrix that x represents
_checkparam(w::AbstractArray, x::SymmetricSparseIterate) =
  length(w) == x.p^2 || _parameter_length_mismatch(length(w), x.p^2)

# copies of the state of x: values, active zeros and the order of activation
Base.copy(x::SymmetricSparseIterate) = copyto!(similar(x), x)
# empty owners of the same dimension
Base.similar(x::SymmetricSparseIterate{T}) where {T} = SymmetricSparseIterate(T, x.p)
Base.similar(x::SymmetricSparseIterate, ::Type{S}) where {S} = SymmetricSparseIterate(S, x.p)

# Between owners, a state copy: the working set of y replaces that of x, active zeros and
# order included, and only the active values are converted, before anything is written.
function Base.copyto!(x::SymmetricSparseIterate, y::SymmetricSparseIterate)
  x === y && return x
  x.p == y.p || throw(DimensionMismatch(
    "destination is $(x.p)×$(x.p), source is $(y.p)×$(y.p)"))
  _copyto_stored!(x, y)
end

# Equality of two owners is that of their matrices, so it does not depend on which coordinates are
# active or on their order: isequal(x, y) is isequal(Matrix(x), Matrix(y)), and x == y is
# Matrix(x) == Matrix(y). An absent coordinate is zero(T), so an active -0.0 differs from it
# under isequal and not under ==, and paired NaNs are isequal and not ==. Both comparisons read
# the active coordinates of the two owners and nothing else.
function _owners_agree(eq::F, x::SymmetricSparseIterate{S}, y::SymmetricSparseIterate{T}) where {F, S, T}
  @inbounds for k in 1:x.nnz
    eq(x.nzval[k], _getstored(y, x.nzval2ind[k])) || return false
  end
  @inbounds for k in 1:y.nnz
    x.ind2nzval[y.nzval2ind[k]] == 0 && !eq(zero(S), y.nzval[k]) && return false
  end
  true
end
function Base.isequal(x::SymmetricSparseIterate, y::SymmetricSparseIterate)
  x === y && return true
  x.p == y.p && _owners_agree(isequal, x, y)
end
Base.:(==)(x::SymmetricSparseIterate, y::SymmetricSparseIterate) = x.p == y.p && _owners_agree(==, x, y)
# The sum over the coordinates that are not the zero of an absent coordinate, of the hash of
# the value and the coordinate: independent of the order of the active set. hash of a real
# number does not depend on its type, so owners of different element types that are isequal
# hash alike, and an active -0.0 contributes where an absent coordinate does not.
function Base.hash(x::SymmetricSparseIterate, h::UInt)
  acc = zero(UInt)
  @inbounds for k in 1:x.nnz
    v = x.nzval[k]
    isequal(v, zero(v)) || (acc += hash(v, hash(x.nzval2ind[k])))
  end
  hash(acc, hash(x.p, hash(:SymmetricSparseIterate, h)))
end

# From a matrix, a checked import that replaces the state of x: the matrix is real, of the
# shape of x, does not share memory with x and is exactly symmetric, all before anything is
# written. The working set is then the coordinates of nonzero converted value, in increasing
# order of coordinate.
Base.copyto!(x::SymmetricSparseIterate, y::AbstractMatrix) = _import_matrix!(x, y)
function _import_matrix!(x::SymmetricSparseIterate{T}, y::AbstractMatrix) where {T}
  _checkreal(x, y)
  _check_axes(x, y)
  _aliases(x, y) && throw(ArgumentError(
    "the source of copyto!(x, A) shares memory with x, as matrixview(x) does: copy it first, " *
    "for example with Matrix(A)"))
  _check_symmetric(y)
  _clear!(x)
  p = x.p
  @inbounds for c = 1:p, r = c:p
    v = convert(T, y[r, c])
    iszero(v) || _store!(x, tril_index(p, r, c), v)
  end
  x
end
# Whether y shares memory with the storage of x. An empty array shares nothing, as for
# Base.mightalias: all empty arrays of a type can share one empty buffer. The wrappers of
# LinearAlgebra do not all report the data of their parent, so the parents are followed.
_aliases(x::SymmetricSparseIterate, y) = x.p > 0 && !isempty(y) && _shares_data(Base.dataids(x), y)
function _shares_data(ids, y)
  any(id -> id in ids, Base.dataids(y)) && return true
  q = parent(y)
  q === y ? false : _shares_data(ids, q)
end
# aliasing hooks from Base: an owner is not an array, so Base.unalias copies a view of it
# (through the unaliascopy methods of the views) when it is the destination
Base.mightalias(x::SymmetricSparseIterate, a::AbstractArray) = _aliases(x, a)
Base.mightalias(a::AbstractArray, x::SymmetricSparseIterate) = _aliases(x, a)
# A source with another number of dimensions cannot have the axes of x.
Base.copyto!(x::SymmetricSparseIterate, y::AbstractArray) = _throw_axes(x, y)

# A broadcast that is written into an iterate is real, for the owner's sources and for AtomIterate.
_checkreal_broadcast(x, bc::Broadcast.Broadcasted) = _checkreal_args(x, bc.args)
_checkreal_args(x, args::Tuple) = (_checkreal_broadcast(x, first(args)); _checkreal_args(x, Base.tail(args)))
_checkreal_args(x, ::Tuple{}) = true
# Broadcast leaves can be control arguments, such as functions or rounding modes.
_checkreal_broadcast(x, a) = true
_checkreal_broadcast(x, a::Union{AbstractArray, Ref, Number}) =
  !(eltype(a) <: Complex) || _real_error(nameof(typeof(x)))

# The result of a broadcast into an iterate is real as well. Its element type R is read from
# the types. _staged_result is false when R is Union{} or a subtype of Real (which includes
# Bool, Integer, Rational and BigFloat results): the broadcast then writes directly. The test
# of R is resolved by inference when the function and the argument types are known to the
# compiler, the usual case. Any other R is staged. A complex R is rejected before
# anything is evaluated, and a Union with a complex member, Any or Number is evaluated into a
# temporary first, so that nothing is written when an element is complex or f throws.
@inline function _staged_result(bc::Broadcast.Broadcasted)
  R = Broadcast.combine_eltypes(bc.f, bc.args)
  !(R === Union{} || R <: Real)
end
@noinline function _staged_broadcast!(x::AbstractArray, bc::Broadcast.Broadcasted{Nothing})
  R = Broadcast.combine_eltypes(bc.f, bc.args)
  axes(x) == axes(bc) ||
    throw(DimensionMismatch("destination axes $(axes(x)) do not match the broadcast axes $(axes(bc))"))
  R <: Complex && _real_error(nameof(typeof(x)))
  # the broadcast that bc represents, with the axes it carries (singleton expansion, an empty destination)
  staged = copy(Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{ndims(x)}}(bc.f, bc.args, bc.axes))
  any(v -> v isa Complex, staged) && _real_error(nameof(typeof(x)))
  @inbounds for I in eachindex(staged)
    x[I] = staged[I]
  end
  x
end

# The stored entries as a SparseMatrixCSC, in O(nactive): each nonzero (r, c) of the
# lower triangle gives the entries (r, c) and (c, r) for which keep holds. The generic
# sparse of a matrix visits all p² entries, and the generic triu and tril of the view
# return a dense matrix.
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
# the view and its triangles as sparse matrices; a stored zero is not an entry
SparseArrays.sparse(A::SymmetricMatrixView) = _sparse(A.x, (i, j) -> true)
LinearAlgebra.triu(A::SymmetricMatrixView, k::Integer = 0) = _sparse(A.x, (i, j) -> j - i >= k)
LinearAlgebra.tril(A::SymmetricMatrixView, k::Integer = 0) = _sparse(A.x, (i, j) -> j - i <= k)

# one line: the shape, the type and the number of active coordinates, not the storage arrays
function Base.show(io::IO, S::SymmetricSparseIterate)
  n = nactive(S)
  print(io, S.p, "×", S.p, " ", typeof(S), " with ", n, " stored ", n == 1 ? "entry" : "entries")
end

function Base.show(io::IO, ::MIME"text/plain", S::SymmetricSparseIterate)
  xnnz = nactive(S)
  print(io, S.p, "×", S.p, " ", typeof(S), " with ", xnnz, " stored ",
        xnnz == 1 ? "entry" : "entries", " in the lower triangle")
  xnnz == 0 && return
  print(io, ":")
  _show_entries(io, xnnz, pos -> tril_cartesian(S.p, S.nzval2ind[pos]), pos -> S.nzval[pos], S.p)
end

LinearAlgebra.mul!(C::AbstractMatrix{T}, A::SymmetricSparseIterate{T}, B::AbstractMatrix{T}) where T = _mul_symsparse!(C, A, B)

function _mul_symsparse!(C::AbstractMatrix{T}, A::SymmetricSparseIterate{T}, B::AbstractMatrix{T}) where T
  _checkreal(A, C); _checkreal(A, B)
  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())
  size(B, 2) == size(C, 2) || throw(DimensionMismatch())
  _checkalias(C, A, B)

  fill!(C, zero(T))
  i = 1
  @inbounds while i <= A.nnz
    v = A.nzval[i]
    ind = A.nzval2ind[i]
    r, c = tril_cartesian(size(A, 1), ind)
    if iszero(v)
      # a zero coefficient contributes nothing, even against a non-finite entry of B
    elseif r == c
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
  _aliases(A, C) && throw(ArgumentError("the output of mul! must not be aliased with the matrix, e.g. coordinates(X) or activevalues(X)"))
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
    if iszero(v)
      # a zero coefficient contributes nothing, even against a non-finite entry of B
    elseif r == c
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
function LinearAlgebra.dot(a::AbstractVector, X::SymmetricSparseIterate, b::AbstractVector)
  _checkreal(X, a); _checkreal(X, b)
  Base.require_one_based_indexing(a, b)
  length(a) == X.p == length(b) || throw(DimensionMismatch())
  s = zero(promote_type(eltype(a), eltype(X), eltype(b)))
  @inbounds for k = 1:X.nnz
    v = X.nzval[k]
    iszero(v) && continue     # a zero coefficient contributes nothing, even against a non-finite entry
    r, c = tril_cartesian(X.p, X.nzval2ind[k])
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
    v = X.nzval[k]
    iszero(v) && continue     # a zero coefficient contributes nothing, even against a non-finite entry
    r, c = tril_cartesian(X.p, X.nzval2ind[k])
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
  p == 2 && return _norm2_scaled(x)
  norm(matrixview(x), p)
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
  for k = 1:X.nnz
    iszero(X.nzval[k]) && continue
    for i in tril_cartesian(p, X.nzval2ind[k])
      pos[i] == 0 && (pos[i] = 1; nK += 1)
    end
    2 * nK > p && return A * (Matrix(X) * B)
  end

  K = findall(!iszero, pos)
  pos[K] = 1:nK                 # position of each row within K
  XK = zeros(T, nK, nK)
  for k = 1:X.nnz
    iszero(X.nzval[k]) && continue
    r, c = tril_cartesian(p, X.nzval2ind[k])
    XK[pos[r], pos[c]] = XK[pos[c], pos[r]] = X.nzval[k]
  end
  A[:, K] * (XK * B[K, :])
end


# the other element types: the stored-entry kernel of the five-argument product
function LinearAlgebra.mul!(C::AbstractMatrix, A::SymmetricSparseIterate, B::AbstractMatrix)
  _checkreal(A, C); _checkreal(A, B)
  _mul_stored!(C, A, B, true, false)
end
function LinearAlgebra.mul!(C::AbstractVector, A::SymmetricSparseIterate, B::AbstractVector, α::Number, β::Number)
  _checkreal(A, C); _checkreal(A, B); _checkreal(A, α); _checkreal(A, β)
  _mul_stored!(C, A, B, α, β)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::SymmetricSparseIterate, B::AbstractMatrix, α::Number, β::Number)
  _checkreal(A, C); _checkreal(A, B); _checkreal(A, α); _checkreal(A, β)
  _mul_stored!(C, A, B, α, β)
end

# C = α A B + β C over the stored entries of A, O(p + nactive(A)) and no allocation.
# As in BLAS, C is not read when β is zero and B is not read when α is zero.
# A coefficient that is zero, stored or not, contributes nothing, even against a
# non-finite entry of B or a non-finite α.
function _mul_stored!(C::AbstractVector, A::SymmetricSparseIterate, B::AbstractVector, α::Number, β::Number)
  p = A.p
  length(B) == p || throw(DimensionMismatch("A has $p columns but B has length $(length(B))"))
  length(C) == p || throw(DimensionMismatch("A has $p rows but C has length $(length(C))"))
  _checkalias(C, A, B)
  iszero(β) ? fill!(C, zero(eltype(C))) : isone(β) || rmul!(C, β)
  iszero(α) && return C
  @inbounds for k in 1:A.nnz
    v0 = A.nzval[k]
    iszero(v0) && continue
    r, c = tril_cartesian(p, A.nzval2ind[k])
    v = α * v0
    C[r] += B[c] * v
    r == c || (C[c] += B[r] * v)
  end
  C
end
function _mul_stored!(C::AbstractMatrix, A::SymmetricSparseIterate, B::AbstractMatrix, α::Number, β::Number)
  p = A.p
  Base.require_one_based_indexing(C, B)
  size(B, 1) == p || throw(DimensionMismatch("A has $p columns but B has $(size(B, 1)) rows"))
  size(C, 1) == p || throw(DimensionMismatch("A has $p rows but C has $(size(C, 1))"))
  size(C, 2) == size(B, 2) || throw(DimensionMismatch("B has $(size(B, 2)) columns but C has $(size(C, 2))"))
  _checkalias(C, A, B)
  iszero(β) ? fill!(C, zero(eltype(C))) : isone(β) || rmul!(C, β)
  iszero(α) && return C
  n = size(C, 2)
  @inbounds for k in 1:A.nnz
    v0 = A.nzval[k]
    iszero(v0) && continue
    r, c = tril_cartesian(p, A.nzval2ind[k])
    v = α * v0
    for j in 1:n
      C[r, j] += B[c, j] * v
      r == c || (C[c, j] += B[r, j] * v)
    end
  end
  C
end

# the allocating products: the type of the sum of the products, as for a matrix
_product_eltype(::Type{Ta}, ::Type{Tb}) where {Ta, Tb} =
  (T = Base.promote_op(*, Ta, Tb); Base.promote_op(+, T, T))
function Base.:*(A::SymmetricSparseIterate, x::AbstractVector)
  _checkreal(A, x)
  mul!(Vector{_product_eltype(eltype(A), eltype(x))}(undef, A.p), A, x)
end
function Base.:*(A::SymmetricSparseIterate, B::AbstractMatrix)
  _checkreal(A, B)
  mul!(Matrix{_product_eltype(eltype(A), eltype(B))}(undef, A.p, size(B, 2)), A, B)
end

# norm_diff(A, B, p) of the matrices that the arguments represent: both triangles of an owner
norm_diff(A::SymmetricSparseIterate, B::SymmetricSparseIterate, p::Real = 2) =
  norm_diff(matrixview(A), matrixview(B), p)
norm_diff(A::SymmetricSparseIterate, B::AbstractArray, p::Real = 2) = norm_diff(matrixview(A), B, p)
norm_diff(A::AbstractArray, B::SymmetricSparseIterate, p::Real = 2) = norm_diff(A, matrixview(B), p)

# Mixed real operands keep LinearAlgebra's multiplication order.
function *(A::Symmetric, X::SparseIterate{<:Any, 2}, B::Symmetric)
  _checkreal(X, A); _checkreal(X, B)
  invoke(*, Tuple{AbstractMatrix, AbstractMatrix, AbstractMatrix}, A, X, B)
end
# an owner of another element type than the factors: the dense product of the matrix it represents
function *(A::Symmetric, X::SymmetricSparseIterate, B::Symmetric)
  _checkreal(X, A); _checkreal(X, B)
  size(A, 2) == X.p == size(B, 1) || throw(DimensionMismatch())
  (A * Matrix(X)) * B
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
