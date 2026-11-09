# Developer documentation and contracts

This section is for contributors implementing or reviewing **storage types, mathematical function kernels, Julia integration methods and numerical behavior** in ProximalBase 0.6. The user-facing guides explain how to call the APIs; these pages explain *which behavior is required*, *why*, and *how it is tested*.

## Contract index

| Contract or record | Authoritative source | Published documentation |
|:--|:--|:--|
| SparseIterate / AtomIterate family | `docs/design/iterate-family-0.6.md` | [Full 0.6 iterate-family contract](contracts/iterate-family-0.6.md) |
| Family acceptance and downstream migration | `docs/design/iterate-family-0.6-acceptance.md` | [Full iterate acceptance plan](contracts/iterate-family-0.6-acceptance.md) |
| SymmetricSparseIterate independent-coordinate owner | `docs/design/symmetric-sparse-iterate-0.6.md` | [Full symmetric owner contract](contracts/symmetric-sparse-iterate-0.6.md) |
| Symmetric owner acceptance and migration | `docs/design/symmetric-sparse-iterate-0.6-acceptance.md` | [Full symmetric owner acceptance plan](contracts/symmetric-sparse-iterate-0.6-acceptance.md) |
| Package-wide numerical behavior | `docs/src/numerics.md` | [Numerical contract](../numerics.md) (already published) |
| SparseIterate source-level audit | `docs/audit/sparse-iterate-06-method-inventory.md` | [Method-by-method source inventory](sparse-iterate-method-inventory.md) |

**Source of truth:** The files in `docs/design/` remain the authoritative versions of the four iterate-family documents. The documentation build stages their exact content into the published site (with a required relative-link rebasing); **do not edit the generated copies**. The numerical contract already lives in `docs/src/numerics.md` and is published directly. The source audit is historical evidence of definitions, not an alternative public API contract.

To inspect the numerical decisions and implementation evidence behind the public numerical contract, use the following canonical design records in GitHub:

- [Numerical-contract design record](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/README.md)
- [Maintainer decisions and rationale](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/decisions.md)
- [Numerical conformance coverage map](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/coverage.md)
- [Step endpoints and infinite-argument design](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/step-infinity-interface-311-469.md)
- [Historical issue impact review](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/impact-on-open-issues.md)

These **design records** are informative history and tracking evidence; they are not automatically up-to-date tests or an independent normative specification. The public numerical contract governs when a historical discussion differs from it.

## Data structures versus mathematical functions

The storage interfaces are defined by what an algorithm can do with its **iterate**: index, write, activate, copy, traverse the active set and materialize a dense value.

| Iterate | Shape / index space | Main accessors | Distinguishing rule |
|:--|:--|:--|:--|
| `SparseIterate` | `AbstractVector` or `AbstractMatrix`; coordinate = array linear index | `x[k]`, `coordinates(x)`, `activecoordinates(x)`, `activevalues(x)` | O(1) scalar access, active-slot traversal O(s) |
| `SymmetricSparseIterate` | Non-array owner; coordinate = independent lower-triangle entry | `X[i,j]`, `coordinates(X)`, `matrixview(X)`, active handles | `X[k]` is forbidden; symmetric entries represent one variable |
| `AtomIterate` | `AbstractVector`; coordinate = group | `x[j]`, `atom(x,k)`, `atomindices(x,k)`, `activecoordinates(x)` | Active sets contain groups, not scalar entries |

See the [SparseIterate guide](../sparse_iterate.md), [symmetric owner guide](../symmetric_sparse_iterate.md), and [coordinate-view implementation](coordinate_views.md).

The following APIs belong to **functions or algorithms**, not to the storage type's minimum contract: `restrict`, `prox_entry`, `prox_coordinate!`, `kkt_violation`, `gradient!`, `prox!`, `quadratic_model` and `value_and_prox!`. They may accept an iterate, but a new iterate does not need to define those oracles itself. Consult [the function interface](../interface.md) and [coordinate descent](../coordinate_descent.md).

## How to read the 0.6 contracts

Each iterate-family contract contains a **requirements section** with stable rule IDs, followed by design decisions and historical evidence. The requirements and any explicit later amendments govern; past proposals and experiments are not promises. For the current SparseIterate decision, **FC-11 is amended**: `nnz`, `nonzeros`, `findnz` and `issparse` are no longer specialized on `SparseIterate`. The underlying independent-coordinate owner also does not provide these aliases, and `dropzeros!` remains in both types.

The numerical contract explicitly separates promised behavior from conformance-suite gaps. Do not infer that a published rule has passed every backend test: consult the coverage map and the relevant CI run or tracker.

## Contributor workflow for a contract change

1. **State the need and scope.** Is this a storage interface, Julia `AbstractArray` compatibility, a kernel optimization, a function-oracle capability or a numerical policy? Avoid moving one into another merely for convenience.
2. **Edit the authoritative requirement** with rule IDs, semantics, input validation, aliasing, explicit-zero behavior, complexity and performance targets. If replacing an earlier decision, state explicitly which rule was amended.
3. **Update the acceptance plan and tests**, including independent references, exception behavior, and the supported Julia versions. For owner changes, test the distinction between coordinate IDs, active slots and full matrix indices.
4. **Update the user-facing guide, API reference, migration guide and source docstrings.** Keep the developer documentation linked to the authoritative record, rather than cloning the contract into another independently editable file.
5. **Audit the downstream solver consumers.** In particular, changes in copy semantics, group layouts, owner indexing or activation may require ProximalSolvers migration and issue #527's release ledger.
6. **Measure ordinary CPU hot paths** on Float32 and Float64 before and after changes, with the pre-registered benchmark method and stated acceptance threshold. Do not silently accept a regression in the name of interface cleanliness.
7. **Require reviewed, green CI** before merging. Generated documentation appearing in the navigation is not evidence that the numerical or iterate conformance gates have passed.

## Documentation generation

`docs/make.jl` creates `docs/src/developer/contracts/` from the original `docs/design/` contracts at build time, just as it generates the example pages. Generated files are ignored by Git. To add a new authoritative iterate contract to the published documentation, add its source filename to the `ITERATE_CONTRACTS` list in `docs/make.jl`, then add it to the Developer documentation navigation.

The source-level method audit is similarly staged from `docs/audit/`. **Never edit the staged artifacts.** Check generated links and the Documenter build whenever a contract is moved or renamed.

Current implementation/release tracking: [iterate-family umbrella #456](https://github.com/mlakolar/ProximalBase.jl/issues/456), [symmetric-owner umbrella #391](https://github.com/mlakolar/ProximalBase.jl/issues/391), [numerical contract #366](https://github.com/mlakolar/ProximalBase.jl/issues/366), and [downstream migration #527](https://github.com/mlakolar/ProximalBase.jl/issues/527).
