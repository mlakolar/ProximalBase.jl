# BlockIterate: grouped storage implementation

For the supported entry and group operations, begin with the
[user guide](../block_iterate.md); for group trial points and KKT screens see
[coordinate oracles](../iterate_oracles.md). This page describes invariants
needed when changing `src/block_iterate.jl`. The
[iterate-family contract](contracts/iterate-family-0.6.md) is authoritative.

## Two index spaces and one active set

`BlockIterate{T,G} <: AbstractVector{T}` exposes ordinary entry indexing
`x[j]`, but its optimization coordinates are **groups**. The groups must
partition `1:length(x)` into nonempty, disjoint sets. The implementation
validates the partition when an iterate is constructed and creates a
`group_of[j]` reverse map so that an entry write can find its group in
constant time.

The main private data are:

| Field | Role |
|:--|:--|
| `storage` | Dense entry buffer, with zeros in inactive groups |
| `groups` | Immutable-in-use partition, retained as ranges or owned vectors |
| `group_of` | Entry ID to group ID |
| `active`, `position`, `nactive` | Packed active-group list and reverse positions |
| `layout_cache` | One previously verified equal group layout, used for a fast equality hit |

An active group may contain only zeros. A nonzero entry write activates its
group; a zero write does not deactivate it. `dropzeros!` inspects active
groups and removes all-zero groups, possibly swapping a surviving group
into a removed slot. Never store solver history keyed by active slot across
compaction without remapping. Raw writes to `storage` that bypass activation
can leave an inconsistent iterate: do not use them in solver code.

## Group layout and read-only handles

`blockview(x,k)` is a live view of group entries through `x`, so scalar writes
still activate the group. `blockindices(x,k)` gives a read-only index collection.
Stored ranges are returned as ranges; stored vector groups are exposed
through the private `_GroupIndices` wrapper, which rejects writes and
preserves underlying alias identity. A caller must not mutate `groups` or
reach the original group arrays through wrapper fields.

`samegroups(a,b)` is public but not exported. For two iterates, or an
iterate and a group penalty, it compares the *ordered* groups and the
indices within each group. Equal ranges and vectors agree by value,
but reordered groups or reordered members do not.

The `layout_cache` stores at most one verified equal layout. Repeated
comparison against that exact layout is constant time; comparisons against
a different layout may inspect all indices. A positive comparison of two
iterates can benefit from already shared cache identities. This cache
does not change the mathematical groups and must not be used as a reason
to mutate an otherwise shared layout. Copies retain their layout records;
state copying does not overwrite the destination's cache.

## Oracle boundaries and failure behavior

The mathematical trial-point interface is `prox_block!(out,f,u,k,γ)` with
compact numerical buffers. A solver checks `samegroups(x,f)`, constructs the
trial, computes its result, then writes `copyto!(blockview(x,k), out)` and updates
its caches. The caller must provide output independent of iterate storage and caches;
the oracle neither infers ownership nor validates an iterate layout. Writing a tentative group into private `storage` can leave
nonzero entries absent from the active set. Keep mathematical work separate
from the public write, so a trial that shrinks to zero leaves an absent group
inactive. KKT screens remain function methods with their own layout checks.

Ordinary array writes and a same-kind iterate state copy have distinct
preflight and activation semantics. Test both routes, including aliasing
through `blockview(x,k)` and an index permutation. External oracles must
maintain consistency even when they write only a subset of groups.

## Testing and performance

Relevant tests include `test/block_iterate.jl`,
`test/iterate_queries.jl`, `test/iterate_writes.jl`,
`test/active_handle.jl`, `test/symmetric_groups.jl`, and the
family acceptance matrix. Benchmark Float32 and Float64 vector hot paths,
both contiguous and noncontiguous groups, warm and cold layout comparisons.
Do not infer an O(1) equality comparison merely because a cache exists.
