# SymmetricSparseIterate: owner implementation

The [public guide](../symmetric_sparse_iterate.md) teaches the two views.
The [current owner contract](contracts/symmetric-sparse-iterate-0.6.md) scopes
the stable requirements. This page describes private implementation details.

## A non-array owner

The owner stores q=p(p+1)/2 independent variables, while its represented matrix
has p² positions. Making it an AbstractMatrix or exposing a one-integer index
would blur those spaces. Pair access maps either triangle to the same
lower-triangle coordinate; coordinates(x) addresses q variables and matrixview(x)
reads p² matrix entries. Its explicit throwing axes method prevents accidental
array dispatch. Do not add array methods to make a missing function oracle work.

The private maps/value arrays share the packed representation of SparseIterate,
with a dimension p. The reverse-map invariant and unique active IDs apply in
independent-coordinate space. Pair writes must update once and activate once.
Storage capacity is O(q), active work O(s). Coordinate IDs do not depend on slots.

## Public matrix view and internal writer

Public SymmetricMatrixView is live and read-only, including derived views and
wrappers. Generic allocating operations return ordinary writable dense arrays.
Private `_OwnerMatrix` is a separate mutable wrapper used by trusted kernels;
it must never become a public writable matrix escape hatch.

A broadcast through the private writer visits each lower-triangle coordinate
once. Reading/writing both positions independently would apply an in-place
operation twice. Alias identities follow all backing arrays; owner products
reject output overlapping the owner or numerical right operand before writes.
Read-only wrapper behavior and alias detection are tested in matrixview.jl.

## Imports, copies and numerical geometry

Owner state copies preserve order and explicit zeros and preconvert fallible
stored values. Matrix import instead checks real type, exact axes, independence
and exact symmetry before clearing old state, then converts/stores nonzero
coordinates in increasing order. Conversion failure during import can leave
partial new state; don't extend the state-copy atomicity promise to import.

Matrix function values and Frobenius distances count an off-diagonal coordinate
twice. The same multiplicity cancels in an entrywise Frobenius prox. An
independent-vector objective/derivative instead needs explicit multiplicities.
KKT uses a symmetric matrix-gradient pair; it is not the doubled independent
coordinate derivative. Scalar hooks need conversion to full entry indices.

Parameter validation belongs at implemented public boundaries, with all/stored/
chosen-pair domains determined by the kernel. Arbitrary custom public overrides
must call check_symmetric_parameters themselves. Non-symmetric prox arguments
are read through their symmetric part, whereas parameters/imports must be
exactly symmetric or explicitly wrapped by the caller.

## Regression evidence

Retained tests include `test/symmetric_iterate_interface.jl`,
`symmetric_iterate_indexing.jl`, `matrixview.jl`, `symmetric_products.jl`,
`real_iterates.jl`, `sparse_iterate_basics.jl`, `core_functions.jl` and
`minimal_math.jl`. Test refusals as well as supported operations: no owner
length/iteration/broadcast/fill/copy!, no one-index ambiguity, no write through
a public matrix view. Include empty/one-dimensional owners, active zeros,
signed zeros/NaNs, failure state, dense import and off-diagonal multiplicity.
