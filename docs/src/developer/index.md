# Developer Documentation

This section explains implementation invariants and extension boundaries.
Begin with [function objects](../function_objects.md), [operations and calling conventions](../operations.md)
and [iterates](../iterates.md) for ordinary usage.

| Area | Guide | Contract / evidence |
|:--|:--|:--|
| Architecture | [Source map](../architecture.md) | Current module include graph |
| Entry storage | [SparseIterate internals](sparse_storage.md) | [Iterate-family contract](contracts/iterate-family-0.6.md) |
| Group storage | [BlockIterate internals](block_storage.md) | Same family contract; block_copy tests |
| Symmetric owner | [Owner internals](symmetric_storage.md) | [Symmetric contract](contracts/symmetric-sparse-iterate-0.6.md) |
| Ownership | [Views, aliases and failure behavior](views_aliasing.md) | Retained iterate_writes/matrixview/real_iterates suites |
| Mathematical extension | [Function dispatch](oracle_dispatch.md) | [Pure block contract](contracts/block-prox-0.6.md) |
| Precision and exceptional arithmetic | [Numerical contract](../numerics.md) | ridge_precision, minimal_math and product suites |
| Contribution/validation | [Change process](change_process.md) | Current CI.yml and test/tiers.toml |

## Current versus historical contracts

The three `docs/design` contracts linked above are updated for the post-#850
minimal core and rendered once by `docs/make.jl`. Generated copies are ignored
by Git. Their public requirements are scoped to actual retained methods;
implementation notes and test evidence are labeled separately.

Historical acceptance plans and numerical decisions describe a broader
implementation, pending targets and past measurements. They remain at
[the preserved post-#848 design snapshot](https://github.com/mlakolar/ProximalBase.jl/tree/eff21b6d7d33c622313122f46c6ee71693c101ee/docs/design).
They are not republished as present guarantees. In particular there is no
current historical full conformance harness or automatic test-map generator.
Use the retained test inventory and [coverage matrix](../api/coverage.md).

## Boundaries to maintain

Storage owns values, coordinate interpretation, activation, copying and views.
A function owns mathematical evaluation, supported gradients/prox, parameter
validation and private numerical scratch. The caller owns step selection,
trials, working-set policy, residual updates and convergence tests.

An interface extension must not mutate storage metadata from a mathematical
hook or read private fields from solver code. Any new public operation needs
its supported signatures, shape/precision/alias/failure contract, narrative
documentation, docstring, independent numerical example and regression evidence.
