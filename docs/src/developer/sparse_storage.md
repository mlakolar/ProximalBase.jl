# SparseIterate: storage internals and method audit

See the [user guide](../sparse_iterate.md) for access and costs. This page
explains the current representation; its fields are private implementation
details, not solver accessors.

## Representation invariants

`nzval[1:s]` contains active values; `nzval2ind[1:s]` maps active slot to
coordinate ID; `ind2nzval[k]` is the inverse slot or zero when absent. The
private field `nnz` is s, the count of stored coordinates, not a public
SparseArrays `nnz(x)` method. For every active slot a,
`ind2nzval[nzval2ind[a]]==a`. IDs are unique and in the logical coordinate
range. Every absent coordinate reads `zero(T)`.

Both maps and value capacity cover the full logical extent, including all
entries of a matrix. This O(n) memory/construction permits O(1) lookup and
insertion without resizing. Active traversal/compaction uses O(s) work.
The matrix coordinate ID is the ordinary column-major linear index.

A scalar write checks bounds and converts before changing state. A nonzero
converted value appends an absent coordinate. Zero leaves an absent coordinate
absent and an active one active. NaN/Inf are nonzero for activation. Activation
can explicitly append zero. No mathematical kernel should remove coordinates
as an incidental consequence of thresholding.

## Compaction and copying

`dropzeros!` removes stored signed zeros and updates both maps when moving a
surviving last slot into a hole. Slot order is unstable; coordinate meaning is
stable. Consumers must reacquire active handles and remap any slot-indexed
history. Empty/all-zero working sets must preserve the same invariants.

Same-shape SparseIterate `copyto!` replaces state. It converts fallible **stored**
values before writing; stale capacity beyond s is irrelevant and must not be
converted. Different shapes of equal length copy values in linear order.
Different lengths throw. Ordinary broadcasts and array writes preserve
existing active coordinates and activate newly nonzero coordinates.

`similar` is empty; `copy` is an independent state copy via array integration.
Coordinate views are ordinary vectors, not state-copy handles.
`fill!(x,0); dropzeros!(x)` is the supported clear. Direct array writers have
stronger predictable complex-source preflight than arbitrary derived views;
see [failure behavior](views_aliasing.md).

## Array integration and numerical kernels

The coordinate wrapper reports the owner's backing array identities to Base,
so copying/broadcasting can snapshot aliases before mutation. The private
`dataids` and `unaliascopy` methods are Julia integration hooks, not public PB
APIs. Test them against current Julia when changing wrappers.

Sparse-aware matrix/dot kernels address stored entries and skip zero iterate
coefficients. Generic array fallback and coordinate-view operations may visit
all entries and use IEEE zeros. The direct three-factor FP32/FP64 methods have
specific range recovery, not a universal guarantee for all products.
Scalar lmul!/rmul!/division on SparseIterate matches corresponding broadcasts:
if the scalar maps zero to a nonzero (Inf/NaN arithmetic), absent coordinates
must activate. A failed conversion may partially write, while keeping maps
consistent. See [numerics](../numerics.md).

## Regression evidence and adding a method

`test/iterate_invariants.jl`, `iterate_queries.jl`, `active_handle.jl`,
`iterate_writes.jl`, `real_iterates.jl`, `sparse_iterate_basics.jl`,
`sparse_iterate.jl` and `sparse_iterate_basics_grids.jl` cover this surface.
Tests use independent dense values and explicit active IDs. Preserve both
mathematical values and state semantics; equality alone cannot detect lost
explicit zeros or corrupted reverse maps.

When adding a writer, specify whether it writes values or replaces state,
check the extent and conversion contract, exercise aliases and partial errors,
and verify all invariants after both success and failure. When adding a sparse
kernel, specify exact operand types and zero/non-finite semantics. Measure
normal FP32/FP64 hot paths before claiming allocation/performance improvements.
