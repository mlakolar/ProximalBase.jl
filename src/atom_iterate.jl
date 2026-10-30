####################################
#
#  Atom Iterate
#
####################################

"""
    AtomIterate([T = Float64,] p, groups)
    AtomIterate(x::AbstractVector, groups)
    AtomIterate([T = Float64,] p, num_groups::Integer)
    AtomIterate(x::AbstractVector, num_groups::Integer)

A vector of length `p` whose coordinates, for coordinate descent, are the groups
("atoms") `x.groups[k]`, which partition `1:p`. The groups are ranges such as
`1:5` or vectors of indices such as `[2, 7, 9]`, of any sizes;
`AtomIterate(p, num_groups)` and `AtomIterate(x, num_groups)` split `1:p` into
`num_groups` equal contiguous groups.

As the stored entries of a [`SparseIterate`](@ref) do, a set of active groups
serves as the active set of a solver, and the entries of the other groups are
zero. Writing a nonzero entry activates its group, [`activate!`](@ref) activates
a group, and `dropzeros!` deactivates the active groups whose entries are all
zero; an iterate built from a vector `x` starts with the groups that hold a
nonzero entry. Reading or writing an entry is O(1), and so is activating or
deactivating a group. [`ncoordinates`](@ref) is the number of groups,
[`nactive`](@ref), [`activecoordinates`](@ref) and [`isactive`](@ref) give the
active groups, and [`atom`](@ref) is a view of the entries of a group.
"""
mutable struct AtomIterate{T, G<:AbstractVector{<:AbstractVector{<:Integer}}} <: AbstractVector{T}
  storage::Vector{T}       # the entries; those of inactive groups are zero
  groups::G                # a partition of 1:length(storage)
  group_of::Vector{Int}    # group_of[i] = k for i ∈ groups[k]
  # Group arrays must not be mutated after a penalty or iterate is built.
  layout_cache::Any        # an equal group array, or nothing
  active::Vector{Int}      # active[1:nactive] are the active groups, in the order they joined
  position::Vector{Int}    # position[k] is the index of group k in active, 0 if it is inactive
  nactive::Int
  function AtomIterate{T, G}(storage, groups, group_of, layout_cache, active, position, nactive) where {T, G<:AbstractVector{<:AbstractVector{<:Integer}}}
    T <: Real || _real_error(:AtomIterate)
    new{T, G}(storage, groups, group_of, layout_cache, active, position, nactive)
  end
end
AtomIterate(storage::Vector{T}, groups::G, group_of::Vector{Int}, layout_cache, active::Vector{Int}, position::Vector{Int}, nactive::Int) where {T, G<:AbstractVector{<:AbstractVector{<:Integer}}} =
  AtomIterate{T, G}(storage, groups, group_of, layout_cache, active, position, nactive)

_stored_values(x::AtomIterate) = x.storage
_scaled_dot(a, x::AtomIterate, s) = _scaled_dot(a, x.storage, s)

AtomIterate(p::Integer, groups::AbstractVector{<:AbstractVector{<:Integer}}) = AtomIterate(Float64, p, groups)
function AtomIterate(::Type{T}, p::Integer, groups::AbstractVector{<:AbstractVector{<:Integer}}) where {T}
  G = _own_groups(groups)
  _AtomIterate(zeros(T, p), G, _partition_group_of(G, p))
end

function AtomIterate(x::AbstractVector{T}, groups::AbstractVector{<:AbstractVector{<:Integer}}) where {T}
  G = _own_groups(groups)
  _AtomIterate(Vector{T}(x), G, _partition_group_of(G, length(x)))
end

AtomIterate(p::Integer, num_groups::Integer) = AtomIterate(Float64, p, num_groups)
AtomIterate(::Type{T}, p::Integer, num_groups::Integer) where {T} = AtomIterate(T, p, _equal_groups(p, num_groups))
AtomIterate(x::AbstractVector, num_groups::Integer) = AtomIterate(x, _equal_groups(length(x), num_groups))

# 1:p split into num_groups contiguous groups of equal sizes
function _equal_groups(p::Integer, num_groups::Integer)
  num_groups > 0 && p % num_groups == 0 || throw(ArgumentError("num_groups must divide p."))
  m = div(p, num_groups)
  [(m*(k-1)+1):(m*k) for k in 1:num_groups]
end

# the iterate with the given entries; the groups with a nonzero entry are active
function _AtomIterate(storage::Vector{T}, groups::G, group_of::Vector{Int}) where {T, G}
  K = length(groups)
  x = AtomIterate{T, G}(storage, groups, group_of, nothing, zeros(Int, K), zeros(Int, K), 0)
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

@inline function _samelayout(f, x::AtomIterate)
  x.groups === f.groups && return true
  x.layout_cache === f.groups && return true
  x.groups == f.groups || return false
  x.layout_cache = f.groups
  true
end

# Copying must preserve the cached penalty layouts of both iterates.
@inline function _samelayout(y::AtomIterate, x::AtomIterate)
  y.groups === x.groups ||
    (y.layout_cache !== nothing && y.layout_cache === x.layout_cache) ||
    y.layout_cache === x.groups || x.layout_cache === y.groups ||
    y.groups == x.groups
end


Base.length(x::AtomIterate) = length(x.storage)
Base.size(x::AtomIterate) = size(x.storage)
Base.IndexStyle(::Type{<:AtomIterate}) = IndexLinear()

Base.@propagate_inbounds Base.getindex(x::AtomIterate, i::Int) = x.storage[i]
Base.@propagate_inbounds function Base.setindex!(x::AtomIterate, v, i::Int)
  x.storage[i] = v
  iszero(v) || activate!(x, x.group_of[i])     # the entries of inactive groups stay zero
  x
end

ncoordinates(x::AtomIterate) = length(x.groups)
nactive(x::AtomIterate) = x.nactive
activecoordinates(x::AtomIterate) = view(x.active, 1:x.nactive)
isactive(x::AtomIterate, k::Integer) = x.position[k] != 0

"""
    atom(x::AtomIterate, k)

A view of the entries of group `k` of `x`. Writing through it does not activate
the group; see [`activate!`](@ref).
"""
atom(x::AtomIterate, k::Integer) = view(x.storage, x.groups[k])

function activate!(x::AtomIterate, k::Integer)
  checkbounds(x.position, k)
  if x.position[k] == 0
    x.nactive += 1
    x.active[x.nactive] = k
    x.position[k] = x.nactive
  end
  x
end

"""
    dropzeros!(x::AtomIterate)

Deactivate the active groups of `x` whose entries are all zero and return `x`.
The last active group moves into the place of each removed one, so the order
of [`activecoordinates`](@ref) can change. The entries of `x` do not change.
"""
function SparseArrays.dropzeros!(x::AtomIterate)
  i = 1
  while i <= x.nactive
    k = x.active[i]
    if all(iszero, atom(x, k))
      last = x.active[x.nactive]
      x.active[i] = last
      x.position[last] = i
      x.position[k] = 0
      x.nactive -= 1
    else
      i += 1
    end
  end
  x
end

# zero the entries of the active groups and deactivate them
function _zero_active!(x::AtomIterate{T}) where {T}
  for a in 1:x.nactive
    k = x.active[a]
    fill!(atom(x, k), zero(T))
    x.position[k] = 0
  end
  x.nactive = 0
  x
end


Base.similar(x::AtomIterate{T}) where {T} = similar(x, T)
function Base.similar(x::AtomIterate, ::Type{S}) where {S}
  K = length(x.groups)
  AtomIterate{S, typeof(x.groups)}(zeros(S, length(x)), x.groups, x.group_of, x.layout_cache, zeros(Int, K), zeros(Int, K), 0)
end

Base.convert(::Type{<:AtomIterate{T}}, ::AbstractArray) where {T<:Complex} = _real_error(:AtomIterate)

Base.copy(x::AtomIterate{T, G}) where {T, G} =
  AtomIterate{T, G}(copy(x.storage), x.groups, x.group_of, x.layout_cache, copy(x.active), copy(x.position), x.nactive)

# with the same groups, only the entries of the active groups are copied
function Base.copyto!(dest::AtomIterate, src::AtomIterate)
  dest === src && return dest
  length(dest) == length(src) || throw(DimensionMismatch())
  _samelayout(dest, src) ||
    return invoke(copyto!, Tuple{AbstractArray, AbstractArray}, dest, src)
  _zero_active!(dest)
  for a in 1:src.nactive
    k = src.active[a]
    copyto!(atom(dest, k), atom(src, k))
    activate!(dest, k)
  end
  dest
end

function Base.copyto!(x::AtomIterate, bc::Broadcast.Broadcasted{Nothing})
  _checkreal_broadcast(x, bc)
  invoke(copyto!, Tuple{AbstractArray, Broadcast.Broadcasted{Nothing}}, x, bc)
end

function Base.copyto!(x::AtomIterate, bc::Broadcast.Broadcasted{S}) where {S<:Broadcast.AbstractArrayStyle{0}}
  _checkreal_broadcast(x, bc)
  invoke(copyto!, Tuple{AbstractArray, Broadcast.Broadcasted{S}}, x, bc)
end

function Base.fill!(x::AtomIterate, v)
  if iszero(v)
    _zero_active!(x)
  else
    fill!(x.storage, v)
    for k in eachindex(x.groups)
      activate!(x, k)
    end
  end
  x
end

Base.Vector(x::AtomIterate) = copy(x.storage)


## multiplication: loops over the entries of the active groups, which are as fast
## as BLAS on the active columns whether the groups are contiguous or not

function *(A::StridedMatrix{Ta}, x::AtomIterate{Tx}) where {Ta, Tx}
  _checkreal(x, A)
  size(A, 2) == length(x) || throw(DimensionMismatch())
  mul!(Vector{promote_type(Ta, Tx)}(undef, size(A, 1)), A, x)
end

LinearAlgebra.mul!(out::AbstractVector, A::StridedMatrix, x::AtomIterate) =
  mul!(out, A, x, true, false)
function LinearAlgebra.mul!(out::AbstractVector, A::StridedMatrix, x::AtomIterate, α::Number, β::Number)
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

function *(A::_StridedSymmetric, x::AtomIterate{T}) where {T}
  _checkreal(x, A)
  mul!(Vector{promote_type(eltype(A), T)}(undef, size(A, 1)), A, x)
end
LinearAlgebra.mul!(out::StridedVector, A::_StridedSymmetric, x::AtomIterate) =
  mul!(out, A, x, true, false)
function LinearAlgebra.mul!(out::StridedVector, A::_StridedSymmetric, x::AtomIterate, α::Number, β::Number)
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
# sparse vectors
function LinearAlgebra.dot(v::Union{StridedVector, SparseIterate{<:Any, 1}, AtomIterate}, x::AtomIterate)
  _checkreal(x, v)
  length(v) == length(x) || throw(DimensionMismatch())
  s = zero(promote_type(eltype(v), eltype(x)))
  @inbounds for a in 1:x.nactive, j in x.groups[x.active[a]]
    s += dot(v[j], x.storage[j])
  end
  s
end
LinearAlgebra.dot(x::AtomIterate, v::Union{StridedVector, SparseIterate{<:Any, 1}}) = dot(v, x)


function Base.show(io::IO, ::MIME"text/plain", x::AtomIterate)
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
      show(io, Vector(atom(x, k)))
    elseif a == half + 1
      print(io, "\n  ⋮")
    end
  end
end

# The cache owns the entries and active state, but shares the immutable layout.
function _output_buffer!(s::Scratch, y::Y) where {Y<:AtomIterate}
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
