# Coordinate views, ownership and failure behavior

## Handle lifetimes

| Handle | Writable | Lifetime / interpretation |
|:--|:--|:--|
| coordinates(`SparseIterate`) | yes | Stable full-entry vector mapping |
| coordinates(symmetric owner) | yes | Stable independent lower-triangle mapping |
| `matrixview`(owner) | no | Stable full-matrix mapping, mirrored entries |
| `blockview`(block,k) | yes | Stable group order, writes through entry setter |
| `blockindices`(block,k) | no | Immutable-in-use ordered global indices |
| `activecoordinates`(x) | no | Old-extent packed slot handle; reacquire after structural change |
| `activevalues`(sparse/owner) | yes | Same slot mapping and reacquisition rule |

Coordinate views are ordinary vectors with ordinary conversion and IEEE
products, not sparse-kernel proxies. Active handles protect IDs while permitting
value updates, including zeros. Mutating private fields/parents to change IDs or
groups is unsupported, even when Julia introspection makes it possible.

## Alias integration: private machinery

Backing arrays report identities through Base.dataids. Coordinate/active/matrix
wrappers preserve those identities; unaliascopy returns an independent source
in the wrapper's own representation when Base needs a copy for overlap.
These are version-dependent internal Julia hooks, not PB public extension APIs.
Creating/reading a view does not itself copy the owner. Actual overlap handling
can allocate O(full coordinate capacity).

Ordinary array copying/broadcasting and `blockview` copying can snapshot aliased
sources. This does **not** make every mathematical oracle overlap-safe:
`prox!(x,f,x,γ)` supports exact identity; shifted/reversed/derived overlapping
views are unsupported and can silently give incorrect values. Built-in gradient
identity is rejected; supply independent scratch for other overlaps too.
Owner matrix products reject output aliases with either operand.

Compact-vector `prox_block!` has a separate caller precondition: independent
output from all iterate/model cache storage. A view's plain-vector parent does
not establish independence. The custom hook checks exact identity or rejects
other buffer overlap; the allocating wrapper does not enforce that for it.

## Validation and observable failure states

| Operation | Preflight before writes | Failure during element work |
|:--|:--|:--|
| Scalar iterate setter | Bounds and conversion | No prior state change if conversion fails |
| Same-kind matching state copy | Extent/layout and fallible active-value conversions | Staged conversion avoids partial destination state |
| Owner matrix import | Real type, axes, alias rejection, exact symmetry | Conversion/read failure can leave cleared/partial new state |
| Direct array-iterate fill | Real-type check and scalar conversion | Valid scalar then written using activity rules |
| Direct array-iterate broadcast | Leaf/result type and shape; uncertain non-real results staged | Real result conversion can still partially write |
| Ordinary view/coordinate/block value writes | Base indexing/extent rules | Partial reads/conversions permitted, storage invariants maintained |
| Built-in function oracle | Predictable type/shape/parameter/symmetry checks in supported method | Numerical/conversion/user-read failure may leave output changed |
| Custom public oracle override | Whatever its author implements | Must document its own boundary |

A typed complex array source for direct array-iterate copying is rejected before
writes. A Vector{Any}, tuple or generator can instead convert values one by one.
A complex element converted by an ordinary setter/view can raise InexactError,
possibly after prior elements changed. Stage a validated typed independent
buffer when failure atomicity is required; don't advertise every `!` method as
transactional.

## Parameter/data ownership and cache invalidation

Built-in mathematical objects usually retain parameter arrays by reference.
Avoid output aliases with data, parameters and scratch. Some elementwise
kernels tolerate specific parameter aliases, but there is no general alias
safety promise. Views and wrappers can share an object without making this
obvious from their concrete type.

`LeastSquaresLoss` caches an SVD and transformed response; `QuadraticFunction`
caches an eigenbasis. Entry changes require `reset_cache!` on the outermost
`Scaled` object before another prox; changing dimensions requires reconstruction.
The hook returns its input. It is a no-op for ordinary functions without caches.
Values/gradients already reuse scratch, so concurrent calls on the same mutable
function need synchronization even before a decomposition is built.

Tests: active_handle, iterate_writes, block_copy, real_iterates, `matrixview`,
iterate_invariants, cache_and_blocks and ridge_precision. Test values and
activity/order independently after both successful and failed calls.
