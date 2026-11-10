# Contract conformance and contributor workflow

## One source of truth per requirement

For any public function API change, also update
[Operations on functions](../operations.md) and the relevant
[Function library](../function_library.md) entry. Treat the public
operation inventory as a review checklist against the exports in
`src/ProximalBase.jl`, while retaining the existing docstrings as the
detailed method reference.


The current 0.6 contracts live in `docs/design/iterate-family-0.6.md`
and `docs/design/symmetric-sparse-iterate-0.6.md`, with matching
acceptance plans; the canonical numerical specification is
`docs/src/numerics.md`. Their rendered navigation pages are generated
rather than separately maintained.

Distinguish **normative requirements**, explicit amendments, recommended
implementation techniques, historical audit findings, and unimplemented
targets. Do not turn a historical benchmark or audit note into a guaranteed
behavior without a contract change.

The [family acceptance plan](contracts/iterate-family-0.6-acceptance.md),
[owner acceptance plan](contracts/symmetric-sparse-iterate-0.6-acceptance.md),
and [numerical conformance map](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/coverage.md)
are the starting points for checking current evidence. Passing a doctest
does not establish full numerical conformance.

## Procedure for a contract change

1. Classify the proposal: iterate storage, array compatibility,
   coordinate or function oracle, numerical semantics, or an
   implementation-only optimization.
2. Update the **authoritative requirement** and, where needed, record a
   specific amendment to earlier signed-off rule IDs. State shapes,
   index spaces, validation, aliasing, explicit-zero handling, complexity
   and performance boundaries.
3. Update acceptance criteria and tests, using independent reference
   computations where practical, including exceptional inputs and
   supported Julia versions.
4. Update user guides, source docstrings, API reference and migration
   material. Generated contract pages are not an editing target.
5. Audit downstream consumers in ProximalSolvers and other supported
   packages for private-field access, method dispatch, copy semantics
   and changed index spaces.
6. Benchmark normal Float32 and Float64 CPU hot paths with the
   preregistered comparison method, alongside correctness testing;
   report meaningful slowdowns separately.
7. Require CI and review before merging; a rendered page or green
   documentation build is not itself proof of kernel correctness.

## Useful local checks

From the repository root, with an installed Julia version supported
by the package:

```sh
julia scripts/instantiate.jl docs
julia --project=docs docs/make.jl
```

The documentation script regenerates the Literate examples and the four
rendered contract/acceptance pages before calling Documenter.
`Documentation.yml` builds docs on pull requests and merges. In a
documentation-only PR, inspect that job's output for broken `@ref`
targets, missing page files, doctest errors and Markdown parsing warnings.

For focused iterate changes, run the corresponding files through the
package's test harness and then the requested CI tier. Relevant groups
include `test/iterate_queries.jl`, `test/iterate_writes.jl`,
`test/sparse_iterate.jl`, `test/atom_iterate.jl`,
`test/symmetric_iterate_interface.jl`, and `test/matrixview.jl`.
The package-wide conformance suite is in `test/conformance/`.

## Documentation information architecture

The public [Iterates](../iterates.md) section teaches construction,
indexing, copying, performance-relevant usage and coordinate oracles.
The developer section owns representation invariants, private Base
integration, design contracts, source maps and acceptance methodology.
The [API reference](../api/iterates.md) provides method signatures and
docstrings, not an alternative narrative specification.

Keep existing page paths wherever possible: published deep links and
references to `sparse_iterate.md`, `symmetric_sparse_iterate.md`,
`iterates.md`, `numerics.md`, and `architecture.md` should continue to
resolve even when navigation categories change.
