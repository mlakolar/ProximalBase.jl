# SymmetricSparseIterate: implementation

The [public guide](../symmetric_sparse_iterate.md) teaches the two views.
The [current `SymmetricSparseIterate` contract](contracts/symmetric-sparse-iterate-0.6.md) scopes
the stable requirements. This page describes private implementation details.

## A non-array iterate

The symmetric iterate stores q=p(p+1)/2 independent variables, while its represented matrix
has p² positions. Making it an `AbstractMatrix` or exposing a one-integer index
would blur those spaces. Pair access maps either triangle to the same
lower-triangle coordinate; coordinates(x) addresses q variables and `matrixview`(x)
reads p² matrix entries. Its explicit throwing axes method prevents accidental
array dispatch. Do not add array methods to make a missing function oracle work.

The private maps/value arrays share the packed representation of `SparseIterate`,
with a dimension p. The reverse-map invariant and unique active IDs apply in
independent-coordinate space. Pair writes must update once and activate once.
Storage capacity is O(q), active work O(s). Coordinate IDs do not depend on slots.

## Public matrix view

Public SymmetricMatrixView is live and read-only, including derived views and
wrappers. Generic allocating operations return ordinary writable dense arrays.
No writable matrix wrapper of the symmetric iterate exists, and none may be added as a
public escape hatch; the symmetric iterate is written through its coordinates.

Alias identities follow all backing arrays; symmetric-iterate products
reject output overlapping the symmetric iterate or numerical right operand before writes.
Read-only wrapper behavior and alias detection are tested in `matrixview`.jl.

## Imports, copies and numerical geometry

Symmetric-iterate state copies preserve order and explicit zeros and preconvert fallible
stored values. Matrix import instead checks real type, exact axes, independence
and exact symmetry before clearing old state, then converts/stores nonzero
coordinates in increasing order. Conversion failure during import can leave
partial new state; don't extend the state-copy atomicity promise to import.

Matrix function values and Frobenius distances count an off-diagonal coordinate
twice. The same multiplicity cancels in an entrywise Frobenius prox. An
independent-vector objective/derivative instead needs explicit multiplicities.
KKT uses a symmetric matrix-gradient pair; it is not the doubled independent
coordinate derivative. Scalar hooks need conversion to full entry indices.

Symmetry of per-entry function parameters is a precondition of the model. The
numerical oracles assume it and never compare mirrored parameter entries;
`check_symmetric_parameters` is the explicit all-pairs validation, and the caller
or solver runs it at its solve or workspace boundary. The oracles do keep their
structural validation (shapes, parameter extents, element types, aliases,
supported operations). Non-symmetric prox arguments are trial points, read through
their symmetric part; computed gradients are symmetrized. Matrix imports into the
iterate are validated exactly, since they are iterate data, and a caller that wants a
triangle wraps the matrix explicitly.

## Regression evidence

Retained tests include `test/symmetric_iterate_interface.jl`,
`symmetric_iterate_indexing.jl`, `matrixview.jl`, `symmetric_products.jl`,
`real_iterates.jl`, `sparse_iterate_basics.jl`, `core_functions.jl` and
`minimal_math.jl`. Test refusals as well as supported operations: no symmetric iterate
length/iteration/broadcast/fill/copy!, no one-index ambiguity, no write through
a public matrix view. Include empty/one-dimensional symmetric iterates, active zeros,
signed zeros/NaNs, failure state, dense import and off-diagonal multiplicity.
