# Symmetric-owner contract: retained minimal core

**Status:** current documentation of the retained SymmetricSparseIterate
surface after #850. Historical acceptance goals are archived separately.

## Public representation

- The owner is a real non-array object of size (p,p), with q=p(p+1)/2
  independent lower-triangle coordinates numbered down columns.
- x[r,c] and x[c,r] address one variable. Pair bounds precede narrowing of
  other integer types. One-integer access throws ArgumentError as ambiguous.
- coordinates(x) is a live writable length-q vector. matrixview(x) is a live
  read-only p×p matrix. Matrix(x) is an independent dense copy.
- The owner has no ordinary length, iteration, array broadcast, fill!, copy!,
  view or vec interface. axes explicitly throws. Derived matrix-view wrappers
  must not permit a write to owner storage.
- Real type, active-zero lifecycle, direct-address capacity, packed handles and
  compaction follow the retained family contract in independent-coordinate space.
- copy preserves explicit zeros/order independently; similar is empty.
  Owner-to-owner copyto! replaces state and stages fallible stored conversions.
- Matrix construction/import requires real, exact-size/one-based, exactly symmetric
  input, with equality before conversion. Paired NaNs and signed zeros agree.
  A source sharing owner storage is rejected. Imports replace activity with
  converted nonzeros in increasing coordinate order. Conversion failure after
  clearing can leave partial imported state; import is not atomic state copy.
- Two owners compare/hash by represented values independent of activation
  order. NaN/signed-zero distinctions follow == versus isequal. There is no
  owner-array equality-by-values or isapprox owner method.

## Geometry and supported mathematical boundaries

- Matrix values and squared Frobenius distances count an off-diagonal variable
  twice. The coordinate vector norm counts once. Entrywise Frobenius proximal
  thresholds are unchanged because both terms have matching multiplicity.
- Supported mathematical oracles explicitly accept the owner. Current values/
  prox cover SeparableFunction, SqrNormL2, LinearFunction, ZeroFunction and
  supported Scaled leaves. Gradients cover SqrNormL2/LinearFunction and supported
  scaling; ZeroFunction has no owner gradient. Loss/quadratic owner methods
  are absent.
- Owner allocating gradient is a dense symmetric matrix; allocating prox is
  an independent owner. Fused values refer to the input or stored output as
  specified by the operation.
- An owner prox output accepts scalar steps only. Array steps are rejected
  before writes. Supported nonsymmetric matrix arguments are read through
  their symmetric part; exactly symmetric/type-symmetric arguments retain
  their values. Import, unlike prox, requires exact symmetry.
- Per-entry parameter arrays must be exactly symmetric on the pairs the
  kernel reads. Full validation via check_symmetric_parameters checks all
  declared pairs recursively. Raw scalar hooks/restrict have no owner check.
- A scalar hook uses full-entry index (c-1)p+r derived from owner coordinate
  k. Owner KKT checks its chosen pair and uses the symmetric matrix gradient,
  rejecting an opposing infinity pair. It does not double the reported screen.
- Public owner matrix/vector products reject output overlapping either operand
  before writes. Zero coefficients are skipped, whether stored or absent.
  The product/three-factor numerical guarantees remain scoped to their exact
  dispatch domains.

## Implementation details and evidence

The owner currently shares packed list/reverse-map primitives with SparseIterate,
adding p; public/private matrix views are different types. Those details and
internal Base alias hooks are not extension APIs. Test pair writes once,
coordinate mapping, view refusals through wrappers, empty owners, explicit
zeros, conversion/alias failures and multiplicity independently.

Retained suites: symmetric_iterate_interface, symmetric_iterate_indexing,
matrixview, real_iterates, sparse_iterate_basics, symmetric_products,
core_functions and minimal_math. Historical acceptance records remain at
[the preserved snapshot](https://github.com/mlakolar/ProximalBase.jl/blob/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design/symmetric-sparse-iterate-0.6-acceptance.md).
They are not reproduced unchanged as current capabilities or test commands.
