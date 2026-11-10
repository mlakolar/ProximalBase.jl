####################################
#
#  Block Iterate
#
####################################

"""
    BlockIterate([T = Float64,] p, groups)
    BlockIterate(x::AbstractVector, groups)
    BlockIterate([T = Float64,] p, num_groups::Integer)
    BlockIterate(x::AbstractVector, num_groups::Integer)

A vector of length `p` whose coordinates, for coordinate descent, are the groups
("blocks") `x.groups[k]`, which partition `1:p`. The groups are ranges such as
`1:5` or vectors of indices such as `[2, 7, 9]`, of any sizes;
`BlockIterate(p, num_groups)` and `BlockIterate(x, num_groups)` split `1:p` into
`num_groups` equal contiguous groups.

As the stored entries of a [`SparseIterate`](@ref) do, a set of active groups
serves as the active set of a solver, and the entries of the other groups are
zero. Writing a nonzero entry activates its group, [`activate!`](@ref) activates
a group, and `dropzeros!` deactivates the active groups whose entries are all
zero; an iterate built from a vector `x` starts with the groups that hold a
nonzero entry. Reading or writing an entry is O(1), and so is activating or
deactivating a group. [`ncoordinates`](@ref) is the number of groups,
[`nactive`](@ref), [`activecoordinates`](@ref) and [`isactive`](@ref) give the
active groups, and [`blockview`](@ref) is a view of `x` at the entries of a group.
Writing through this view follows the same activation rule as writing to `x`.

Its entries are real. The constructors, two-argument `copyto!` from an array,
`copy!`, broadcasting into `x` and `fill!` throw an `ArgumentError` for a source
or value of complex type, even with a zero imaginary part, before anything is
written. A broadcast into `x` checks its leaves and its result: a result known
to be complex from the types is rejected, and a result of another non-real type
(a `Union` with a complex member, `Any`, `Number`) is evaluated before writing
and rejected if an element is complex, with `x` unchanged. `vec(x)` and `reshape(x, length(x))` are `x` itself and are checked
too. `copy!` from a source of another length throws Base's error first, unless
the source is a `BlockIterate` or a [`SparseIterate`](@ref), which throws a
`DimensionMismatch`. The two-argument `copyto!` and `copy!` read types: a tuple, a generator or an
array of abstract element type such as `Vector{Any}` passes the check, and its
values are converted one by one. Other
writes also convert values one by one, as writes into a plain `Vector` of the
element type of `x` do: writes through `blockview(x, k)` or another view, indexed
assignment such as `x[i] = v` or `x[:] = c`, `copyto!` with offsets or index
ranges, and `map!`. A complex value converted one by one throws an
`InexactError`, possibly after part of `x` has been written.
"""
mutable struct BlockIterate{T, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractVector{T}
  storage::Vector{T}       # the entries; those of inactive groups are zero
  groups::G                # a partition of 1:length(storage)
  group_of::Vector{Int}    # group_of[i] = k for i ∈ groups[k]
  # Group arrays must not be mutated after a penalty or iterate is built.
  layout_cache::Any        # an equal group array, or nothing
  active::Vector{Int}      # active[1:nactive] are the active groups, in the order they joined
  position::Vector{Int}    # position[k] is the index of group k in active, 0 if it is inactive
  nactive::Int
  function BlockIterate{T, G}(storage, groups, group_of, layout_cache, active, position, nactive) where {T, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    T <: Real || _real_error(:BlockIterate)
    new{T, G}(storage, groups, group_of, layout_cache, active, position, nactive)
  end
end
BlockIterate(storage::Vector{T}, groups::G, group_of::Vector{Int}, layout_cache, active::Vector{Int}, position::Vector{Int}, nactive::Int) where {T, G<:AbstractVector{<:AbstractVector{<:Integer}}} =
  BlockIterate{T, G}(storage, groups, group_of, layout_cache, active, position, nactive)

_stored_values(x::BlockIterate) = x.storage
_scaled_dot(a, x::BlockIterate, s) = _scaled_dot(a, x.storage, s)

BlockIterate(p::Integer, groups::AbstractVector{<:AbstractVector{<:Integer}}) = BlockIterate(Float64, p, groups)
function BlockIterate(::Type{T}, p::Integer, groups::AbstractVector{<:AbstractVector{<:Integer}}) where {T}
  G = _own_groups(groups)
  _BlockIterate(zeros(T, p), G, _partition_group_of(G, p))
end

function BlockIterate(x::AbstractVector{T}, groups::AbstractVector{<:AbstractVector{<:Integer}}) where {T}
  G = _own_groups(groups)
  _BlockIterate(Vector{T}(x), G, _partition_group_of(G, length(x)))
end

BlockIterate(p::Integer, num_groups::Integer) = BlockIterate(Float64, p, num_groups)
BlockIterate(::Type{T}, p::Integer, num_groups::Integer) where {T} = BlockIterate(T, p, _equal_groups(p, num_groups))
BlockIterate(x::AbstractVector, num_groups::Integer) = BlockIterate(x, _equal_groups(length(x), num_groups))

# 1:p split into num_groups contiguous groups of equal sizes
function _equal_groups(p::Integer, num_groups::Integer)
  num_groups > 0 && p % num_groups == 0 || throw(ArgumentError("num_groups must divide p."))
  m = div(p, num_groups)
  [(m*(k-1)+1):(m*k) for k in 1:num_groups]
end

# the iterate with the given entries; the groups with a nonzero entry are active
function _BlockIterate(storage::Vector{T}, groups::G, group_of::Vector{Int}) where {T, G}
  K = length(groups)
  x = BlockIterate{T, G}(storage, groups, group_of, nothing, zeros(Int, K), zeros(Int, K), 0)
  for k in 1:K
    any(!iszero, view(storage, groups[k])) && activate!(x, k)
  end
  x
end

# an owned copy of the groups, with a concrete element type: ranges are kept and
# other collections of indices become Vector{Int}s, so that loops over a group
# do not dispatch on its type
function _own_groups(groups::AbstractVector{<:AbstractVector{<:Integer}})
  R = eltype(groups)
  isconcretetype(R) && R <: AbstractRange ? Vector{R}(groups) : [Vector{Int}(G) for G in groups]
end

# group_of[i] = k for i ∈ groups[k], after checking that the groups are nonempty,
# disjoint sets of positive indices
function _group_of(groups)
  n = 0
  for G in groups
    isempty(G) && throw(ArgumentError("Groups must not be empty."))
    minimum(G) >= 1 || throw(ArgumentError("Group indices must be positive."))
    n = max(n, maximum(G))
  end
  group_of = zeros(Int, n)
  for k in eachindex(groups), i in groups[k]
    group_of[i] == 0 || throw(ArgumentError("Groups must be disjoint, but index $i is in groups $(group_of[i]) and $k."))
    group_of[i] = k
  end
  group_of
end

# group_of for groups that partition 1:p
_partition_group_of(groups, p::Integer) = _check_partition(_group_of(groups), p)

function _check_partition(group_of::Vector{Int}, p::Integer)
  length(group_of) == p && all(!iszero, group_of) ||
    throw(ArgumentError("The groups must partition 1:$p."))
  group_of
end

@inline function _samelayout(f, x::BlockIterate)
  x.groups === f.groups && return true
  x.layout_cache === f.groups && return true
  x.groups == f.groups || return false
  x.layout_cache = f.groups
  true
end

# Whether two iterates are known to have equal layouts without comparing them: the same
# layout object, one remembered layout that both share, or a remembered layout that is the
# other's own. Each remembered layout was found equal to the groups of its iterate when it
# was stored, and equality of ordered lists is transitive, so a hit is never wrong.
@inline _layouthit(y::BlockIterate, x::BlockIterate) =
  y.groups === x.groups ||
  (y.layout_cache !== nothing && y.layout_cache === x.layout_cache) ||
  y.layout_cache === x.groups || x.layout_cache === y.groups

# Copying must preserve the cached penalty layouts of both iterates, so this comparison
# remembers nothing.
@inline _samelayout(y::BlockIterate, x::BlockIterate) = _layouthit(y, x) || y.groups == x.groups

"""
    ProximalBase.samegroups(a, b)

Whether `a` and `b` have equal groups: the same number of groups, and the same indices in the
same order in every group (`1:2` and `[1, 2]` are equal, `[2, 1]` is not). `a` and `b` are two
[`BlockIterate`](@ref)s, or a `BlockIterate` and a group penalty (any [`ProximalBase.AbstractGroupPenalty`](@ref),
such as [`GroupNormL2`](@ref) or [`GroupMCP`](@ref)) or an [`IndBallGroupL0`](@ref), in either order. It is the
layout check a solver makes before writing a group, and that the four-argument
[`kkt_violation`](@ref)`(f, g, x, k)` makes before it computes (each throws a `DimensionMismatch` when it
fails), that `value` and `prox!` of a group function make to take the active-group paths, and that `copyto!`
between two iterates makes to copy only the active groups.
`samegroups` is public but not exported.

An iterate remembers one group array that it has found equal to its own. The answer takes
constant time when `a` and `b` share their group array, or both remember the same one, or one
remembers the other's. Otherwise the two are compared in full, which takes time proportional to
the total size of the groups. If they are equal, the iterate `x` (`a` when both are iterates)
remembers the group array of the other, so that asking again takes constant time. The one slot is
replaced each time, so alternating between two equal group arrays that are different objects
compares at every call; a comparison that finds the groups unequal remembers nothing and leaves
the remembered array in place. Neither the cache nor the comparison allocates. The groups of
penalties and iterates must not be mutated (see [`BlockIterate`](@ref)). Because the answer is
remembered in the iterate, calling `samegroups` on one iterate from several threads at once is a
data race, as it is for the other operations on an iterate (the stored value is always an equal
layout).
"""
function samegroups end

samegroups(a::BlockIterate, b::BlockIterate) =
  _layouthit(a, b) || (a.groups == b.groups && (a.layout_cache = b.groups; true))

# storage, active and position are owned by x; groups, group_of and the layout
# cache are layout shared with copies and output buffers, so they are not reported
Base.dataids(x::BlockIterate) =
  (Base.dataids(x.storage)..., Base.dataids(x.active)..., Base.dataids(x.position)...)

Base.length(x::BlockIterate) = length(x.storage)
Base.size(x::BlockIterate) = size(x.storage)
Base.IndexStyle(::Type{<:BlockIterate}) = IndexLinear()

Base.@propagate_inbounds Base.getindex(x::BlockIterate, i::Int) = x.storage[i]
Base.@propagate_inbounds function Base.setindex!(x::BlockIterate, v, i::Int)
  s = convert(eltype(x.storage), v)
  x.storage[i] = s
  iszero(s) || activate!(x, x.group_of[i])     # the entries of inactive groups stay zero
  x
end

ncoordinates(x::BlockIterate) = length(x.groups)
nactive(x::BlockIterate) = x.nactive
activecoordinates(x::BlockIterate) = _ActiveIDs(view(x.active, 1:x.nactive))
isactive(x::BlockIterate, k::Integer) = x.position[k] != 0

"""
    blockview(x::BlockIterate, k)

A view of the entries of group `k`. It is a view of `x` itself, so it aliases
`x`. Writing through it follows the rule of `x[i] = v`: a nonzero entry
activates the group and a zero never deactivates it (see [`dropzeros!`](@ref)).
To take a proximal step at a trial point without activating the group first,
compute [`prox_block!`](@ref)`(out, f, u, k, γ)` in a numerical buffer, then
write the result with `copyto!(blockview(x, k), out)`.
`norm` and `dot` of the view read the storage directly (BLAS for contiguous
groups), while other BLAS routines take generic paths for it.
"""
function blockview(x::BlockIterate, k::Integer)
  G = x.groups[k]
  @inbounds view(x, G)    # groups are a validated partition; only k needs a bounds check
end

"""
    blockindices(x::BlockIterate, k::Integer)

The indices of the entries of group `k`, in the order in which `x` stores them, as a
read-only `AbstractVector{<:Integer}`. A group stored as a range is returned as that range,
whatever its index type: a group stored as `UnitRange{Int32}` is returned as a
`UnitRange{Int32}`. A group stored as a vector is returned as a read-only wrapper of the
stored vector whose element type is `Int`. A write through the wrapper (`h[i] = v`, `fill!`,
`copyto!`, `h .= v`, `sort!` and the like) throws an `ArgumentError` and leaves the groups
unchanged.

Constructing the result takes constant time and does not allocate or copy, and it stays valid
for as long as `x`: the groups of an iterate never change, and activating groups,
`dropzeros!`, `copyto!` and the other writes of values do not affect it. `k` must be in
`1:ncoordinates(x)`, or a `BoundsError` is thrown (a `Bool` is not a group index).

The result can index any array: `view(g, blockindices(x, k))` is the part of `g` that belongs to
group `k`. The fields of `x` and of the wrapper (`h.ids`) are internal, and so is the array that
`parentindices(blockview(x, k))` returns: reading or changing them is not supported, and is not blocked.

# Examples
```julia
x = BlockIterate(6, [[1, 4], [2, 5], [3, 6]])
blockindices(x, 2) == [2, 5]                 # a read-only vector of Int
blockindices(BlockIterate(6, 3), 2) === 3:4   # the stored range
```
"""
function blockindices(x::BlockIterate, k::Integer)
  # the error names the range of valid groups: a BoundsError holds the array it was thrown for, and
  # that must not be a group or the layout
  (!(k isa Bool) && checkbounds(Bool, x.groups, k)) ||
    throw(BoundsError(Base.OneTo(length(x.groups)), k))
  @inbounds _group_indices(x.groups[k])
end

_group_indices(G::AbstractRange{<:Integer}) = G
_group_indices(G::AbstractVector{<:Integer}) = _GroupIndices(G)

# Read-only handle on a group array of an iterate. A writable group would let a caller break the
# agreement between the groups and the map from entries to groups (see _ActiveIDs). The contract covers
# the operations on the handle and the exceptions that blockindices and the handle throw; the field `ids` is
# internal like the fields of BlockIterate, and reading it is unsupported (getproperty is not blocked).
struct _GroupIndices{V<:AbstractVector{<:Integer}} <: AbstractVector{Int}
  ids::V
end
Base.size(h::_GroupIndices) = size(h.ids)
Base.IndexStyle(::Type{<:_GroupIndices}) = IndexLinear()
Base.dataids(h::_GroupIndices) = Base.dataids(h.ids)    # as the array it wraps: aliasing checks still see the group
Base.unaliascopy(h::_GroupIndices) = _GroupIndices(Base.unaliascopy(h.ids))   # view(A, h) copies an aliasing index in its own type
Base.@propagate_inbounds function Base.getindex(h::_GroupIndices, i::Int)
  @boundscheck checkbounds(h, i)      # the error names h: it must not hold the group
  @inbounds convert(Int, h.ids[i])
end
Base.setindex!(::_GroupIndices, v, i::Int) =
  throw(ArgumentError("blockindices(x, k) is read-only: the groups of a BlockIterate cannot be changed"))

# raw storage writes leave activation to the caller
function _block_storage(x::BlockIterate, k::Integer)
  G = x.groups[k]
  @inbounds view(x.storage, G)
end

function activate!(x::BlockIterate, k::Integer)
  checkbounds(x.position, k)
  if x.position[k] == 0
    x.nactive += 1
    x.active[x.nactive] = k
    x.position[k] = x.nactive
  end
  x
end

@inline function _anynonzero(storage, G)
  s = false
  @inbounds for i in G
    s |= !iszero(storage[i])
  end
  s
end

@inline function _deactivate!(x::BlockIterate, k::Integer)
  i = x.position[k]
  if i != 0
    last = x.active[x.nactive]
    x.active[i] = last
    x.position[last] = i
    x.position[k] = 0
    x.nactive -= 1
  end
  x
end

"""
    dropzeros!(x::BlockIterate)

Deactivate the active groups of `x` whose entries are all zero and return `x`.
When every entry of every active group is zero, all groups are deactivated.
Otherwise the last active group moves into the place of each removed one, so the
order of [`activecoordinates`](@ref) can change. The entries of `x` do not change.
"""
function SparseArrays.dropzeros!(x::BlockIterate)
  allzero = true
  for a in 1:x.nactive
    if !all(iszero, _block_storage(x, x.active[a]))
      allzero = false
      break
    end
  end
  if allzero
    for a in 1:x.nactive
      x.position[x.active[a]] = 0
    end
    x.nactive = 0
    return x
  end
  i = 1
  while i <= x.nactive
    k = x.active[i]
    if all(iszero, _block_storage(x, k))
      _deactivate!(x, k)
    else
      i += 1
    end
  end
  x
end

# zero the entries of the active groups and deactivate them
function _zero_active!(x::BlockIterate{T}) where {T}
  for a in 1:x.nactive
    k = x.active[a]
    fill!(_block_storage(x, k), zero(T))
    x.position[k] = 0
  end
  x.nactive = 0
  x
end


Base.similar(x::BlockIterate{T}) where {T} = similar(x, T)
function Base.similar(x::BlockIterate, ::Type{S}) where {S}
  K = length(x.groups)
  BlockIterate{S, typeof(x.groups)}(zeros(S, length(x)), x.groups, x.group_of, x.layout_cache, zeros(Int, K), zeros(Int, K), 0)
end

Base.convert(::Type{<:BlockIterate{T}}, ::AbstractArray) where {T<:Complex} = _real_error(:BlockIterate)

Base.copy(x::BlockIterate{T, G}) where {T, G} =
  BlockIterate{T, G}(copy(x.storage), x.groups, x.group_of, x.layout_cache, copy(x.active), copy(x.position), x.nactive)

# with the same groups, only the entries of the active groups are copied
function Base.copyto!(dest::BlockIterate{T}, src::BlockIterate{S}) where {T, S}
  dest === src && return dest
  length(dest) == length(src) || throw(DimensionMismatch())
  _samelayout(dest, src) ||
    return invoke(copyto!, Tuple{AbstractArray, AbstractArray}, dest, src)
  # a fallible conversion runs once, before dest is touched, and the copy reads its result
  bufs = _convert_may_throw(T, S) ?
    [T[convert(T, v) for v in _block_storage(src, src.active[a])] for a in 1:src.nactive] : nothing
  _zero_active!(dest)
  for a in 1:src.nactive
    k = src.active[a]
    activate!(dest, k)
    copyto!(_block_storage(dest, k), bufs === nothing ? _block_storage(src, k) : bufs[a])
  end
  dest
end

Base.copy!(dest::BlockIterate, src::BlockIterate) = copyto!(dest, src)

# The check reads the element type and runs before anything is written; a source
# of abstract element type converts entry by entry, as in Base.
function Base.copyto!(x::BlockIterate, y::AbstractArray)
  _checkreal_broadcast(x, y)
  invoke(copyto!, Tuple{AbstractArray, AbstractArray}, x, y)
end
# Resolve ambiguities with SparseArrays: each runs the method that applies without the check.
@static if isdefined(@__MODULE__, :_CHOLMODDense)
  function Base.copyto!(x::BlockIterate, y::_CHOLMODDense)
    _checkreal_broadcast(x, y)
    invoke(copyto!, Tuple{AbstractArray, _CHOLMODDense}, x, y)
  end
  @static if isdefined(@__MODULE__, :_CHOLMODVTypes)
    function Base.copyto!(x::BlockIterate{T}, y::_CHOLMODDense{T}) where {T<:_CHOLMODVTypes}
      _checkreal_broadcast(x, y)
      invoke(copyto!, Tuple{AbstractArray{T}, _CHOLMODDense{T}}, x, y)
    end
  end
end

# A SparseIterate and a BlockIterate of the same length: a value write in linear
# order, after the real check and the length check.
function Base.copyto!(x::SparseIterate, y::BlockIterate)
  _checkreal_broadcast(x, y)
  _check_iterate_lengths(x, y)
  invoke(copyto!, Tuple{AbstractArray, AbstractArray}, x, y)
end
function Base.copyto!(x::BlockIterate, y::SparseIterate)
  _checkreal_broadcast(x, y)
  _check_iterate_lengths(x, y)
  invoke(copyto!, Tuple{AbstractArray, AbstractArray}, x, y)
end
Base.copy!(x::SparseIterate, y::BlockIterate) = copyto!(x, y)
Base.copy!(x::BlockIterate, y::SparseIterate) = copyto!(x, y)

function Base.copyto!(x::BlockIterate, bc::Broadcast.Broadcasted{Nothing})
  _checkreal_broadcast(x, bc)
  _staged_result(bc) && return _staged_broadcast!(x, bc)
  _broadcast_loop!(x, bc)
end

function Base.copyto!(x::BlockIterate, bc::Broadcast.Broadcasted{S}) where {S<:Broadcast.DefaultArrayStyle{0}}
  _checkreal_broadcast(x, bc)
  invoke(copyto!, Tuple{AbstractArray, Broadcast.Broadcasted{S}}, x, bc)
end

function Base.fill!(x::BlockIterate, v)
  _checkreal_broadcast(x, v)
  s = convert(eltype(x), v)     # converted once, before x changes; activation follows the stored value
  if iszero(s)
    # a value write: the active groups stay active (dropzeros!(fill!(x, 0)) removes them)
    for a in 1:x.nactive
      fill!(_block_storage(x, x.active[a]), s)
    end
  else
    fill!(x.storage, s)
    for k in eachindex(x.groups)
      activate!(x, k)
    end
  end
  x
end

Base.Vector(x::BlockIterate) = copy(x.storage)


## multiplication: loops over the entries of the active groups, which are as fast
## as BLAS on the active columns whether the groups are contiguous or not

function *(A::StridedMatrix{Ta}, x::BlockIterate{Tx}) where {Ta, Tx}
  _checkreal(x, A)
  size(A, 2) == length(x) || throw(DimensionMismatch())
  mul!(Vector{promote_type(Ta, Tx)}(undef, size(A, 1)), A, x)
end

LinearAlgebra.mul!(out::AbstractVector, A::StridedMatrix, x::BlockIterate) =
  mul!(out, A, x, true, false)
function LinearAlgebra.mul!(out::AbstractVector, A::StridedMatrix, x::BlockIterate, α::Number, β::Number)
  _checkreal(x, out); _checkreal(x, A); _checkreal(x, α); _checkreal(x, β)
  size(A, 2) == length(x) || throw(DimensionMismatch("A has second dimension $(size(A, 2)), length of x is $(length(x))"))
  size(A, 1) == length(out) || throw(DimensionMismatch("A has first dimension $(size(A, 1)), length of out is $(length(out))"))
  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, x, α)
  @inbounds for a in 1:x.nactive, j in x.groups[x.active[a]]
    c = x.storage[j]
    iszero(c) && continue
    c = scale * c
    @simd for i in eachindex(out)
      out[i] += c * A[i, j]
    end
  end
  out
end

function *(A::_StridedSymmetric, x::BlockIterate{T}) where {T}
  _checkreal(x, A)
  mul!(Vector{promote_type(eltype(A), T)}(undef, size(A, 1)), A, x)
end
LinearAlgebra.mul!(out::StridedVector, A::_StridedSymmetric, x::BlockIterate) =
  mul!(out, A, x, true, false)
function LinearAlgebra.mul!(out::StridedVector, A::_StridedSymmetric, x::BlockIterate, α::Number, β::Number)
  _checkreal(x, out); _checkreal(x, A); _checkreal(x, α); _checkreal(x, β)
  length(x) == size(A, 2) && length(out) == size(A, 1) || throw(DimensionMismatch())
  _scale_product_output!(out, β)
  iszero(α) && return out
  scale = _product_scale(A, x, α)
  @inbounds for a in 1:x.nactive, j in x.groups[x.active[a]]
    c = x.storage[j]
    iszero(c) || _symmetric_axpy!(out, A, j, scale * c)
  end
  out
end

# dot(v, x) = Σ vⱼ xⱼ over the entries of the active groups of x; other
# vectors than these take the generic method, as SparseArrays has its own for
# sparse vectors. A zero entry of x contributes nothing, even against ±Inf or
# NaN in v (numerics M1), and so does a zero of v when v is an iterate too.
function LinearAlgebra.dot(v::Union{StridedVector, SparseIterate{<:Any, 1}, BlockIterate}, x::BlockIterate)
  _checkreal(x, v)
  length(v) == length(x) || throw(DimensionMismatch())
  s = zero(promote_type(eltype(v), eltype(x)))
  @inbounds for a in 1:x.nactive, j in x.groups[x.active[a]]
    w, c = v[j], x.storage[j]
    t = dot(w, c)
    s += ifelse(iszero(c) | (!(v isa StridedVector) & iszero(w)), zero(t), t)    # a select, not a branch
  end
  s
end
LinearAlgebra.dot(x::BlockIterate, v::Union{StridedVector, SparseIterate{<:Any, 1}}) = dot(v, x)


_atom_view_storage(v::SubArray{<:Real, 1, <:BlockIterate}) =
  @inbounds view(parent(v).storage, parentindices(v)...)
LinearAlgebra.norm(v::SubArray{<:Real, 1, <:BlockIterate}) = norm(_atom_view_storage(v))
LinearAlgebra.norm(v::SubArray{<:Real, 1, <:BlockIterate}, p::Real) = norm(_atom_view_storage(v), p)
LinearAlgebra.dot(v::SubArray{<:Real, 1, <:BlockIterate}, w::StridedVector) = dot(_atom_view_storage(v), w)
LinearAlgebra.dot(v::StridedVector, w::SubArray{<:Real, 1, <:BlockIterate}) = dot(v, _atom_view_storage(w))
LinearAlgebra.dot(v::SubArray{<:Real, 1, <:BlockIterate}, w::SubArray{<:Real, 1, <:BlockIterate}) =
  dot(_atom_view_storage(v), _atom_view_storage(w))


function Base.show(io::IO, ::MIME"text/plain", x::BlockIterate)
  K, na = length(x.groups), x.nactive
  print(io, length(x), "-element ", typeof(x), " with ", na, " of ", K, K == 1 ? " group" : " groups", " active")
  na == 0 && return
  print(io, ":")
  active = sort!(x.active[1:na])
  limit::Bool = get(io, :limit, false)
  half = limit ? max(div(displaysize(io)[1] - 6, 2), 1) : typemax(Int)
  io = IOContext(io, :compact => true)
  for (a, k) in enumerate(active)
    if a <= half || a > na - half
      print(io, "\n  ", k, ": ", x.groups[k], " => ")
      show(io, Vector(_block_storage(x, k)))
    elseif a == half + 1
      print(io, "\n  ⋮")
    end
  end
end

# The cache owns the entries and active state, but shares the immutable layout.
function _output_buffer!(s::Scratch, y::Y) where {Y<:BlockIterate}
  b = s.buf
  if b isa Y && b.groups === y.groups && length(b) == length(y)
    copyto!(b.storage, y.storage)
    copyto!(b.active, y.active)
    copyto!(b.position, y.position)
    b.nactive = y.nactive
    return b
  end
  b = copy(y)
  s.buf = b
  b
end
