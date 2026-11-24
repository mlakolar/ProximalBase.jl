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
    BlockIterate([T = Float64,] p, f::GroupNormL2)
    BlockIterate(x::AbstractVector, f::GroupNormL2)
    BlockIterate([T = Float64,] p, f::GroupNormLinf)
    BlockIterate(x::AbstractVector, f::GroupNormLinf)
    BlockIterate([T = Float64,] p, f::GroupMCP)
    BlockIterate(x::AbstractVector, f::GroupMCP)
    BlockIterate([T = Float64,] p, f::GroupSCAD)
    BlockIterate(x::AbstractVector, f::GroupSCAD)

A real vector whose optimization coordinates are ordered groups of entries.
The nonempty, disjoint groups must partition `1:p`; ranges and reordered
index vectors are supported. An integer group count divides the entries into
equal contiguous groups and must be positive and divide the entry length.

`x[j]` indexes an entry; [`ncoordinates`](@ref) gives the group count,
[`nactive`](@ref) counts active groups, and [`activate!`](@ref),
[`isactive`](@ref) and [`activecoordinates`](@ref) use group IDs. A nonzero
entry activates its group. Zero writes preserve activity;
[`dropzeros!(::BlockIterate)`](@ref) removes all-zero active groups.
[`blockview`](@ref) gives a writable alias in stored group order, and
[`blockindices`](@ref) gives its read-only indices.
Entry access and group activation cost O(1); storage costs O(p + length(groups)).

# Examples
```jldoctest
julia> x = BlockIterate(4, [[3, 1], [4, 2]]);

julia> copyto!(blockview(x, 1), [2.0, 0.0]);

julia> (x[3], ncoordinates(x), nactive(x))
(2.0, 2, 1)
```

# Ownership and writes
- High-level constructors copy values and group lists. `copy` and `similar`
  own mutable value/activity storage but share layout data, which must not be
  mutated.
  Low-level construction requires independent mutable storage.
- Same-layout `copyto!` replaces active state. Ordinary array copies and
  broadcasts never deactivate groups, though nonzero writes can activate them.
- Direct bulk writes preflight known complex source types. Views, indexed
  writes and per-element conversions may fail after partial writes. See
  [write validation](@ref iterate-write-validation).

See the [block guide](@ref block-iterate-guide) for group-penalty constructors,
copying and the [ownership contract](@ref block-iterate-ownership).
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
    _check_one_based_groups(groups)
    _check_one_based(storage, group_of, active, position)
    T <: Real || _real_error(:BlockIterate)
    new{T, G}(storage, groups, group_of, layout_cache, active, position, nactive)
  end
end
# Low-level: storage, active and position must not overlap in memory with each other or
# with those of another iterate (see the docstring); groups and group_of may be shared.
BlockIterate(storage::Vector{T}, groups::G, group_of::Vector{Int}, layout_cache, active::Vector{Int}, position::Vector{Int}, nactive::Int) where {T, G<:AbstractVector{<:AbstractVector{<:Integer}}} =
  BlockIterate{T, G}(storage, groups, group_of, layout_cache, active, position, nactive)

_stored_values(x::BlockIterate) = x.storage

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

# the list of groups and each group have axes that start at 1 (the groups are read by position under @inbounds)
function _check_one_based_groups(groups)
  _check_one_based(groups)
  for G in groups
    _check_one_based(G)
  end
  true
end

# an owned copy of the groups, with a concrete element type: ranges are kept and
# other collections of indices become Vector{Int}s, so that loops over a group
# do not dispatch on its type
function _own_groups(groups::AbstractVector{<:AbstractVector{<:Integer}})
  _check_one_based_groups(groups)
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
    ProximalBase.samegroups(a::BlockIterate, b::BlockIterate)

Whether two block iterates have the same block layout. Group arrays must not
be mutated after construction. A successful comparison may cache the layout
identity in an iterate; concurrent comparisons on the same iterate require
external synchronization. This function is public but not exported.
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

Return a writable view of group `k` in its stored index order. The view aliases
`x`: nonzero writes activate the group, and zero writes preserve activity.
Use [`dropzeros!(::BlockIterate)`](@ref) to remove an all-zero active group.

For a trial proximal step, compute `prox_block!(out, f, u, k, γ)` in independent
numerical buffers, then commit with `copyto!(blockview(x, k), out)`. Copying
into a block view follows ordinary value-copy semantics, including possible
partial writes on conversion failure.

`norm` and `dot` read the underlying storage directly, using BLAS for contiguous
groups; other BLAS routines use generic paths. See [`blockindices`](@ref) and
the [block guide](@ref block-iterate-guide).
"""
function blockview(x::BlockIterate, k::Integer)
  G = x.groups[k]
  @inbounds view(x, G)    # groups are a validated partition; only k needs a bounds check
end

"""
    blockindices(x::BlockIterate, k::Integer)

Return the global entry indices of group `k`, in stored order, as a read-only
`AbstractVector{<:Integer}`. A stored range retains its range and integer types;
a vector group is exposed through a read-only wrapper of element type `Int`.
Use `collect` for an independent writable vector.

The result takes O(1) time without allocation or copying and stays valid for
`x`'s lifetime. Value writes and active-set changes do not change the layout.
`k` must be in `1:ncoordinates(x)`; invalid IDs, including `Bool`, throw
`BoundsError`. Writes through the wrapper throw `ArgumentError`.

Use the result to index other arrays, for example
`view(g, blockindices(x, k))`. Never modify private layout fields, wrapper
fields or `parentindices(blockview(x, k))`; doing so bypasses validation.

# Examples
```jldoctest
julia> x = BlockIterate(6, [[1, 4], [2, 5], [3, 6]]);

julia> blockindices(x, 2) == [2, 5]
true

julia> blockindices(BlockIterate(6, 3), 2) === 3:4
true
```

See [`blockview`](@ref) for values and the [block guide](@ref block-iterate-guide)
for ordered layouts.
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

Deactivate all-zero active groups and return `x`, preserving entry values.
Signed zeros count as zero; a group containing `NaN` or an infinity stays
active. Compaction may reorder active slots, so reacquire
[`activecoordinates`](@ref) afterward.

# Examples
```jldoctest
julia> x = BlockIterate([0.0, 0.0, 2.0, 0.0], 2); activate!(x, 1);

julia> nactive(x)
2

julia> dropzeros!(x); (nactive(x), isactive(x, 2))
(1, true)
```
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
  bufs === nothing && return _copy_active!(dest, src)
  _zero_active!(dest)
  for a in 1:src.nactive
    k = src.active[a]
    activate!(dest, k)
    copyto!(_block_storage(dest, k), bufs[a])
  end
  dest
end

# The layouts and nonthrowing conversion have already been checked. Both iterates
# have independent value/active storage. Groups shared with src will be entirely
# overwritten, so only groups absent from src need their values cleared.
function _copy_active!(dest::BlockIterate{T}, src::BlockIterate) where {T}
  src.nactive == length(src.groups) && _try_copy_all_active!(dest, src) && return dest
  @inbounds for a in 1:dest.nactive
    k = dest.active[a]
    if src.position[k] == 0
      z = zero(T)
      for i in dest.groups[k]
        dest.storage[i] = z
      end
    end
    dest.position[k] = 0
  end
  @inbounds for a in 1:src.nactive
    k = src.active[a]
    for i in src.groups[k]
      dest.storage[i] = src.storage[i]
    end
    dest.active[a] = k
    dest.position[k] = a
  end
  dest.nactive = src.nactive
  dest
end

# Called by _copy_active! when every group of src is active. When active and position of
# both iterates have length K, it copies storage, active and position as whole arrays and
# returns true; otherwise it returns false without writing, and the loops of _copy_active!
# run. The 7-argument constructor accepts longer vectors, whose entries past K the loops
# leave alone. With the groups a partition of the entries and the storage, active and
# position arrays of the two iterates not overlapping in memory, as the ownership
# contract in the docstring requires, the loops write every entry of dest and set all
# of dest.active and dest.position from src, as the copies do. With an inactive group
# in src, the copies can differ from the loops, which keep dest's values in that group,
# such as a -0.0, unless dest has it active and they clear it to +0.0.
# Kept out of line: inlined into _copy_active!, the copies slowed its loop path for
# small, partly active sources in measurements on Julia 1.13.
@noinline function _try_copy_all_active!(dest::BlockIterate, src::BlockIterate)
  K = length(src.groups)
  length(dest.active) == length(dest.position) == length(src.active) == length(src.position) == K ||
    return false
  copyto!(dest.storage, src.storage)
  copyto!(dest.active, src.active)
  copyto!(dest.position, src.position)
  dest.nactive = src.nactive
  true
end

Base.copy!(dest::BlockIterate, src::BlockIterate) = copyto!(dest, src)

# The check reads the element type and runs before anything is written; a source
# of abstract element type converts entry by entry, as in Base.
function Base.copyto!(x::BlockIterate, y::AbstractArray)
  _checkreal_broadcast(x, y)
  invoke(copyto!, Tuple{AbstractArray, AbstractArray}, x, y)
end
# Resolve the ambiguity with CHOLMOD's copyto!(dest::AbstractArray, D::Dense), which runs
# after the check; as for SparseIterate, one method covers every element type of D.
@static if isdefined(@__MODULE__, :_CHOLMODDense)
  function Base.copyto!(x::BlockIterate, y::_CHOLMODDense)
    _checkreal_broadcast(x, y)
    invoke(copyto!, Tuple{AbstractArray, _CHOLMODDense}, x, y)
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
  _check_one_based(out, A)
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
  _check_one_based(out, A)
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
# NaN in v (numerics.md, "Sparse product semantics"), and so does a zero of v when v is an iterate too.
function LinearAlgebra.dot(v::Union{StridedVector, SparseIterate{<:Any, 1}, BlockIterate}, x::BlockIterate)
  _check_one_based(v)
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


_block_view_storage(v::SubArray{<:Real, 1, <:BlockIterate}) =
  @inbounds view(parent(v).storage, parentindices(v)...)
LinearAlgebra.norm(v::SubArray{<:Real, 1, <:BlockIterate}) = norm(_block_view_storage(v))
LinearAlgebra.norm(v::SubArray{<:Real, 1, <:BlockIterate}, p::Real) = norm(_block_view_storage(v), p)
LinearAlgebra.dot(v::SubArray{<:Real, 1, <:BlockIterate}, w::StridedVector) =
  (_check_one_based(v, w); dot(_block_view_storage(v), w))
LinearAlgebra.dot(v::StridedVector, w::SubArray{<:Real, 1, <:BlockIterate}) =
  (_check_one_based(v, w); dot(v, _block_view_storage(w)))
LinearAlgebra.dot(v::SubArray{<:Real, 1, <:BlockIterate}, w::SubArray{<:Real, 1, <:BlockIterate}) =
  (_check_one_based(v, w); dot(_block_view_storage(v), _block_view_storage(w)))


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
