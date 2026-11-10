# Developer Documentation

This section is for contributors and maintainers of **ProximalBase.jl**: its
representation invariants, oracle dispatch, Julia integration, mathematical
and numerical contracts, performance constraints, and conformance evidence.
For API usage and examples start with [Iterates](../iterates.md),
[The interface](../interface.md), and [Writing your own function](../extending.md).

The main user path now begins with
[Function objects and capabilities](../function_objects.md),
[Operations on functions](../operations.md), and the
[Function library](../function_library.md). Keep this developer section
for invariants, internal dispatch, normative contracts and conformance,
not general API discovery.


## Architecture and implementation

| Area | Read | Scope |
|:--|:--|:--|
| System design | [Package structure](@ref architecture) | Function hierarchy, traits, call paths, composition, storage and device boundaries |
| Sparse entries | [SparseIterate internals](sparse_storage.md) | Direct addressing, compaction, state vs value copy, method inventory |
| Independent symmetric coordinates | [Symmetric owner internals](symmetric_storage.md) | Non-array owner, coordinate mapping, matrix views, multiplicity |
| Group coordinates | [AtomIterate internals](atom_storage.md) | Group partition, active groups, immutable layout and cache |
| Julia integration | [Coordinate views and aliasing](views_aliasing.md) | Read-only handles, `dataids`/`unaliascopy`, handle lifetimes |
| Mathematical kernels | [Function and oracle dispatch](oracle_dispatch.md) | Function traits versus method dispatch, proximal, gradient and KKT layers |
| Engineering | [Conformance and change process](change_process.md) | Source-of-truth rules, testing, docs, CI and downstream migration |

## Authoritative contracts and acceptance criteria

The four iterate-family documents below are rendered **in full** into the
documentation site at build time from `docs/design/`. The generated pages are
not another editable copy of the contract. Links between these contracts are
preserved; maintainer amendments in a later contract take precedence where
explicitly stated.

| Normative design | Companion validation / migration plan |
|:--|:--|
| [Iterate-family 0.6 contract](contracts/iterate-family-0.6.md) | [Family acceptance plan](contracts/iterate-family-0.6-acceptance.md) |
| [Symmetric owner 0.6 contract](contracts/symmetric-sparse-iterate-0.6.md) | [Symmetric-owner acceptance plan](contracts/symmetric-sparse-iterate-0.6-acceptance.md) |
| [Published numerical contract](@ref numerics) | [Numerical conformance evidence](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/coverage.md) |

**Read the status labels.** The design documents record 0.6 targets,
amendments, tests, and evidence at specific historical commits. A signed-off
target is not a claim that every feature has been implemented, that an issue is
closed, or that all conformance tests currently pass. The *published numerical
contract* is the normative numerical source; the discussion and decisions in
`docs/design/numerical-contract/` are supporting records, not a second policy.

The 0.6 iterate-family contract supersedes the earlier FC-11 requirement for
`SparseIterate` specializations of `nnz`, `nonzeros`, `findnz`, and `issparse`:
they were deliberately removed; `dropzeros!` remains supported. For a
method-level inventory see the
[source audit](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/audit/sparse-iterate-06-method-inventory.md),
which records the source at its audit commit rather than rechecking today's method table.

## A separation worth maintaining

- **Storage contracts** determine which optimization coordinates exist, their
  values, activation, copying, and how callers access them.
- **Function contracts** determine values, proximal maps, gradients,
  curvature, restrictions, and KKT screens. A storage type does not acquire
  these methods merely by implementing `coordinates`.
- **Numerical contracts** determine result precision, non-finite input
  handling, overflow, aliasing, and observable exceptions within the supported
  domain.
- **Solver responsibilities** include choosing steps, update schedules,
  stopping criteria, and when to screen or compact a working set.

Do not define a new method as public merely because the implementation uses it.
If a contract needs to change, use the
[contributor process](change_process.md) and update the canonical requirement
rather than making an independent rule in these explanatory pages.

## Where the sources live

| Purpose | Repository location |
|:--|:--|
| Public documentation | `docs/src/` |
| Authoritative iterate contracts and acceptance plans | `docs/design/` |
| Authoritative published numerical contract | `docs/src/numerics.md` |
| Numerical decision history | `docs/design/numerical-contract/` |
| Storage source | `src/sparse_iterate.jl`, `src/atom_iterate.jl` |
| Function/oracle source | `src/interface.jl`, `src/functions/`, `src/calculus.jl`, `src/partial.jl` |
| Conformance/tests | `test/conformance/`, `test/iterate_*.jl`, `test/atom_iterate.jl`, `test/symmetric_*.jl` |

Generated contracts are created by `docs/make.jl`, ignored by Git, and
should never be edited directly.
