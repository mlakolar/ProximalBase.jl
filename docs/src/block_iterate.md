# BlockIterate

*Assumes: vector indexing and block-coordinate methods · You will learn: group storage, active blocks and block views.*

Here an [active](glossary.md#Active-coordinate) block is retained in the solver's storage, even if all its entries are zero.

A [`BlockIterate`](@ref) is an `AbstractVector` whose optimization coordinates
are ordered groups of entries. It stores a dense value vector and an active
set of groups. A nonzero entry activates the whole group. An active group may
contain only zeros until `dropzeros!` removes it.

## Construction and partitions

Use `BlockIterate([T,]p,groups)` for empty storage or
`BlockIterate(x::AbstractVector,groups)` to copy entry values. `T` defaults
to `Float64`. A copied input starts with the groups containing a converted
nonzero entry active, in group order. Real element types are required.

Groups must be nonempty disjoint integer lists covering exactly `1:p`.
An out-of-domain index, duplicate, empty group, gap or overlap throws
`ArgumentError`. Noncontiguous and reordered groups are supported:
`[[3,1],[4,2]]` differs from `[[1,3],[4,2]]`. The order inside a group is
the order used by compact block buffers. Vector groups are copied into owned
`Vector{Int}` arrays. Homogeneous concrete range groups keep their range type.

`BlockIterate([T,]p,K)` and `BlockIterate(x,K)` divide the entries into `K`
equal contiguous groups; `K` must be positive and divide the entry length.
A group partition describes how variables are **stored and updated**; it
is not itself a penalty or regularization function. Pass groups as indices,
not as a penalty object. `NormL1` remains an entrywise mathematical function.

```@example block_layout
using ProximalBase
input = [[3,1],[4,2]]
x = BlockIterate(Float32,4,input)
input[1][1] = 2
@assert collect(blockindices(x,1)) == [3,1]
@assert length(x) == 4 && ncoordinates(x) == 2
@assert collect(blockindices(BlockIterate(6,3),2)) == [3,4]
collect(blockindices(x,1))
```

## Two index spaces

`x[j]` reads/writes entry `j`; `activate!(x,k)` and `isactive(x,k)` use a
**group** ID. `nactive` counts groups; `activecoordinates` lists their IDs
in activation order except for later compaction. `coordinates(x)` and
`activevalues(x)` have no `BlockIterate` method. Use `blockview(x,k)` to access
a group's values.

`blockindices(x,k)` returns a read-only vector/range of global entry indices.
It takes constant time, copies no data and stays valid for the lifetime of
`x`. Invalid `k` throws `BoundsError`; Bool is rejected as a group index by
this method. A range keeps its own integer type; other stored groups appear
through a read-only wrapper with element type Int. Mutation through that
wrapper throws `ArgumentError`. `collect` or `copy` gives an independent array.
The private layout, wrapper fields and `parentindices(blockview(...))` must
not be modified.

`ProximalBase.samegroups(a,b)` compares **two BlockIterates** by ordered groups
and ordered members. Ranges/vectors compare by index values, irrespective of
index types. Copies can share immutable-in-use layout data; the function may
cache a successful comparison. Concurrent calls sharing an iterate need
external synchronization. There is no generic comparison with an arbitrary
custom penalty; compare its lists with `blockindices` yourself.

## Writing a block

`blockview(x,k)` is a writable alias of the iterate in the stored group order.
Writes go through entry setters and preserve activation integrity. Nonzero
writes activate a previously absent group once at the end of the active list;
zero writes never deactivate it.

```@example block_layout
v = blockview(x,1)
copyto!(v,Float32[2,0])
@assert Vector(x) == Float32[0,0,2,0] && isactive(x,1)
activate!(x,2)
copyto!(v,zeros(Float32,2))
@assert nactive(x) == 2
saved = copy(x)
dropzeros!(x)
@assert nactive(x) == 0 && nactive(saved) == 2
@assert ProximalBase.samegroups(x,saved)
Vector(saved)
```

`copyto!(blockview(x,k),values)` follows Base value-copy semantics: a shorter
source writes a prefix; an overlong source fails before writing. Aliased
sources are snapshotted as required and can allocate. A failing read or element
conversion may leave a partially written block, with every stored nonzero
still represented by an active group. An independent ordinary same-type copy
needs no extra storage after warm-up. Contiguous, noncontiguous and reordered
groups have the same value semantics.

To compute a trial without activating anything, use independent numerical
buffers. For an entry-separable penalty, call scalar `prox_block` with each
member's **global entry index**, then copy the resulting compact vector into
`blockview`. A custom compact-vector oracle can instead define `prox_block!`.
The [extension example](examples/block_prox_extension.md) checks the layout
explicitly and keeps the numerical and storage writes separate.

## Copies, broadcasting and products

| Operation | Working-set effect |
|:--|:--|
| `copy(x)` | Independent values and active state; layout data shared, immutable in use |
| `similar(x)` / `similar(x,T)` | Empty active set, same length/layout, zero values |
| Same-layout `copyto!(dest,src)` / `copy!` | Replace active groups/order/all-zero active groups; preconvert fallible active values |
| Different-layout copy of same entry length | Ordinary entry value write, keeping destination activity |
| `x .= A`, `copyto!(x,A)` from an ordinary array | Value write; never deactivate an existing group |
| `fill!(x,0)` | Zero values in active groups, preserving activity |
| `dropzeros!(x)` | Remove all-zero active groups; may reorder slots |
| `Vector(x)` | Independent dense value snapshot |

State-copy length mismatch is `DimensionMismatch`. Direct bulk writes reject
known complex source types before writes; generic views, indexed writes,
abstractly typed arrays and conversions can fail after partial writes. See
[array validation](numerical_usage.md).

Dense strided/Symmetric-strided matrix products and dot with supported dense
or iterate vectors traverse active-group entries and skip zero coefficients.
Norm/dot on a `blockview` forward to its underlying storage, using BLAS where
contiguous; other operations can use generic paths. The full domains are in
[the product table](iterates.md#Supported-products). Do not assume block views
are strided buffers or that inactive entries have no allocated memory.

The [developer guide](developer/block_storage.md) records invariants and the
layout cache; the [complete example](examples/block_storage.md) exercises
arbitrary group order, explicit zero groups and scalar penalty indices.
