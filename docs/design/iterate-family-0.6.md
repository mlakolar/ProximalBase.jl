# Iterate-family contract: retained minimal core

**Status:** current documentation of retained public storage behavior after #850.
This replaces the archived full-family design as the live manual contract.
It makes no new implementation requirement or performance acceptance claim.

## Public requirements

- SparseIterate is a real vector/matrix AbstractArray. Its coordinates are
  entries in Julia column-major order. BlockIterate is a real AbstractVector
  with a validated ordered partition, and its coordinates are groups.
- Coordinate reads/writes, membership and first activation are direct-address
  operations. Capacity is proportional to the full coordinate/entry domain;
  sparse traversal is not a compressed-memory guarantee.
- A nonzero converted value activates an absent coordinate/group. Zero does
  not activate an absent coordinate or deactivate an active one. activate!
  explicitly retains a zero coordinate/group. NaN/Inf are nonzero; signed zeros
  are zero for activation/compaction.
- nactive counts stored coordinates, including explicit zeros. Active IDs are
  read-only, unsorted live handles. Sparse activevalues is writable and pairs
  by slot. Reacquire both after activation, compaction or state replacement.
- dropzeros! removes only zero coordinates/all-zero groups, preserving values
  and possibly changing slot order. A stable coordinate ID is not a packed slot.
- coordinates(SparseIterate) is a live writable ordinary vector. BlockIterate
  has blockview/blockindices instead, with no coordinates/activevalues method.
- copy is an independent mutable state copy. similar is zero-valued empty
  storage. Matching same-kind copyto! replaces active state and preconverts
  fallible stored values before writes. Different SparseIterate shapes of equal
  length, or different BlockIterate layouts of equal entry length, copy values.
- Ordinary value writes and broadcasts preserve existing activity. Clearing
  requires a zero fill followed by dropzeros!. Matrix-import owner semantics
  are specified separately in the symmetric contract.
- Direct array-iterate bulk writers preflight predictable complex-type/shape
  failures. Arbitrary views, abstractly typed sources and element conversions
  can fail after partial writes. Invariants must remain consistent.
- Group membership is immutable in use. Vector input groups are copied;
  copies/similar can share internal immutable layout data. Public blockindices
  is read-only and keeps ordered indices for the owner's lifetime.
- samegroups accepts two BlockIterates and compares ordered groups/members.
  It may cache equality and needs synchronization when shared concurrently.
- blockview writes through public entry setters. Complete-block value copy
  follows Base prefix/alias/conversion semantics; it is not an atomic state copy.
- nnz is re-exported for standard SparseArrays. It is not implemented for
  iterates; nactive and explicit nonzero counts have distinct meanings.

## Mathematical operation boundary

Storage does not grant a function operation merely by implementing coordinates.
Scalar prox_block is pure numerical entry work; compact-vector prox_block! is
an extension hook. The caller constructs a trial, computes in independent
scratch, then writes through public storage and updates residuals. No private
metadata access or implicit storage-aware proximal update is supported.

Supported sparse-aware products skip zero iterate coefficients even against
non-finite other operands. Coordinate views instead use ordinary vector
products and can form IEEE 0*Inf. Exact dispatch domains and range guarantees
are in the user product table and numerical contract, not all AbstractArray
fallbacks. No universal no-allocation, atomicity or sparse-complexity promise
applies to arbitrary inherited array methods.

## Current implementation notes

Packed value/ID lists and reverse maps implement SparseIterate; BlockIterate
uses a dense value vector plus group/position maps. Fields, wrapper types,
loop thresholds and equality-cache identities are private. They may change
while preserving the public behavior above.

## Evidence and historical boundary

Retained suites: iterate_invariants, iterate_queries, active_handle,
iterate_writes, real_iterates, block_copy, sparse_iterate_basics,
sparse_iterate_basics_grids, sparse_iterate, symmetric_products and matrixview.
Run the declared test/tiers.toml inventory. Historical full-family acceptance
records include removed mathematical families and old writers; they are at
[the immutable snapshot](https://github.com/mlakolar/ProximalBase.jl/tree/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design),
not a current method inventory or evidence of tests run today.
