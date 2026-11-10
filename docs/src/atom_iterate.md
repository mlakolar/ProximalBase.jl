# AtomIterate

An [`AtomIterate`](@ref) is a vector whose coordinates are groups of entries,
partitioning ``1{:}p``. Its active groups play the role of the stored entries
of a `SparseIterate`, and [`atom`](@ref)`(x, k)` is a view of `x` at group
``k``. Writes through it follow the rule of `x[i] = v`: a nonzero entry
activates the group, and a zero never deactivates it.
Block coordinate descent for the group lasso computes the trial point
``u = x_{G_k} - \nabla_{G_k} f(x)/L_k`` for group ``k``, with
``L_k = \|X_{G_k}\|_2^2`` for least squares. It then takes the block step
`prox_block!(out, g, u, k, 1/L_k)`, then writes
`copyto!(atom(x,k), out)`. Only a nonzero result activates an absent group.
[Block trials](@ref) gives an example. `AtomIterate(p, g)` builds an
iterate that shares the groups of a group penalty `g`.

## The groups of an iterate

[`atomindices`](@ref)`(x, k)` returns the indices of the entries of group `k`, in the order in
which `x` stores them, and `ProximalBase.samegroups(a, b)` tells whether two layouts are equal.
Use them instead of reading the fields of `x`, which are private.

`atomindices(x, k)` is a read-only `AbstractVector{<:Integer}`. A group that `x` stores as a
range comes back as that range, with its own index type: a group stored as a
`UnitRange{Int32}` is a `UnitRange{Int32}`, and `AtomIterate(p, K)` stores `UnitRange{Int}`s.
Every other group is stored as a `Vector{Int}`, and comes back as a read-only wrapper whose
element type is `Int`, so `Int32` vectors, mixed groups and collections of other integer types
all give `Int`. Writing through the wrapper (`h[i] = v`, `fill!`, `copyto!`, `h .= v`, a `sort!`
that moves an entry) throws an `ArgumentError` and leaves the layout unchanged; `copy` and
`collect` give an ordinary array. The call takes constant time, does not allocate, and throws a
`BoundsError` unless `k` is in `1:ncoordinates(x)`. The result stays valid for the lifetime
of `x`: the groups of an iterate do not change, and activating groups, `dropzeros!`, `copyto!`
and writes of values do not affect it. It indexes other arrays like any integer vector, for
instance `view(g, atomindices(x, k))`. A vector group is wrapped, so such a view is not
indexed by a `Vector{Int}`, and its norms and dot products take the generic paths.

`atomindices` is the supported way to inspect a group layout. Its result is
read-only; group membership must not be changed after constructing the iterate.
Create a new iterate when you need a different partition.

`samegroups(a, b)` is `true` when `a` and `b` have the same number of groups and the same
indices in the same order in every group, and `false` otherwise. `a` and `b` are two iterates,
or an iterate and a group penalty or [`IndBallGroupL0`](@ref), in either order. Ranges and vectors
with the same indices are equal, whatever their index types, and reordering the groups or the
indices inside a group gives `false`. A solver checks this before writing a group result.
The four-argument [`kkt_violation`](@ref)`(f, g, x, k)` also checks it before
computing and throws a `DimensionMismatch` when it is `false`. The whole-array oracles do not
require it: they compute the function of the vector for any layout, and use `samegroups` only to
choose the faster active-group paths.

Group layouts are immutable for the lifetime of the iterate. The comparison can reuse
a previously recognized equal layout; the [developer guide](developer/atom_storage.md)
describes the private single-layout cache and its performance implications.


Group arrays and the indices inside them must not be mutated after a penalty
or an iterate is built. This also applies to arrays retained by inner
constructors.
## Example: grouped working set

```@example atom_iterate_guide
using ProximalBase
using SparseArrays: dropzeros!

x = AtomIterate(6, [1:2, 3:4, 5:6])
x[2] = 3.0
activate!(x, 3)                 # keep an initially zero group active
@assert ncoordinates(x) == 3
@assert nactive(x) == 2
@assert collect(atomindices(x, 1)) == [1, 2]
@assert collect(atom(x, 1)) == [0.0, 3.0]

x[2] = 0.0                     # group 1 remains active until compaction
dropzeros!(x)
@assert nactive(x) == 0
```

`AtomIterate` is an ordinary vector in entry space, but its coordinates for a
coordinate-descent algorithm are **groups**. Use [coordinate oracles](iterate_oracles.md)
for block trial updates, and [developer internals](developer/atom_storage.md)
for the group mapping, read-only index wrappers, and layout cache.
