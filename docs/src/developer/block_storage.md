# BlockIterate: grouped storage implementation

See [the public guide](../block_iterate.md) and
[the current family contract](contracts/iterate-family-0.6.md).

## Representation and immutable layouts

Private `storage` is a dense vector; inactive groups contain zeros. `groups`
is an ordered partition of all entries and `group_of[j]` maps entry to group.
`active[1:nactive]` lists active group IDs, and `position[k]` maps group to
slot or zero. Active groups may contain only zeros. Group membership is
immutable in use, including inside copies and output buffers that share it.

Constructors own copies of vector groups. Homogeneous concrete ranges are
retained as ranges; other lists become Vector{Int}. Each group must be nonempty,
positive-indexed, disjoint and the total partition must cover exactly 1:p.
Inner constructors and private-field access are not supported validation escapes.

A scalar nonzero write converts and stores its value, then activates its group
once. A zero write never deactivates. dropzeros! checks active groups and swaps
a surviving last group into removed slots. Active IDs are unsorted live handles;
their read-only interface protects reverse-map integrity.

## Public views and comparison

`blockview(x,k)` is a SubArray of x, so writes go through x's scalar setter.
`blockindices(x,k)` is a read-only range or private vector wrapper. Its mapping
stays valid for x's lifetime. Exposing writable index lists would desynchronize
`groups` from `group_of`. Do not reach indices via wrapper fields or
`parentindices`.

`samegroups` compares two BlockIterates by ordered lists. A private single-layout
identity cache speeds repeated equal comparisons; it can mutate on a successful
query. It is not a mathematical state change, but shared concurrent queries need
synchronization. copy/similar share immutable layouts; state copy preserves
each destination's layout/cache record. Cache identities and field layout are
implementation details, not API guarantees.

## Writes, copies and failures

Same-layout state copy replaces active groups, order and all-zero active groups,
preconverting fallible active values before changes. Equal-length different-layout
copy is an ordinary value write. Neither broadcast nor full array value copy may
silently replace the active set.

Whole-block copy retains Base's prefix/overlap/conversion behavior and writes
through the scalar setter. A failing read/conversion can partially write;
every nonzero written entry still belongs to an active group. Do not claim
atomic whole-block writes or bypass activation in a private raw buffer.

Pure compact-vector `prox_block!` output must be independent of iterate storage.
The solver verifies its custom block layout, constructs a trial, computes the
numerical result, then uses `copyto!(blockview(x,k),out)` and updates its caches.
No retained built-in group penalty or group-prox storage shortcut exists.

## Regression evidence

Relevant retained suites: `test/block_copy.jl`, `iterate_invariants.jl`,
`iterate_queries.jl`, `iterate_writes.jl`, `real_iterates.jl`,
`cache_and_blocks.jl`, `core_functions.jl` and `minimal_math.jl`.
Include arbitrary order, noncontiguous groups, unequal cardinality, aliases,
partial source/conversion failures, explicit zero groups, state-copy atomic
conversion and different-layout copies. Norm/dot forwarding through blockview
must preserve values while enabling contiguous storage paths where available.
Historical writer experiments are evidence at pinned commits, not shipped APIs.
