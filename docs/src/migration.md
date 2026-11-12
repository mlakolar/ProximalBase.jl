# Compatibility and migration

For terms used in the current manual, see the [Glossary](glossary.md).

This manual describes the minimal core after merged PR #850, based on
`2b373c54`. The full post-#848 implementation and its documentation are
preserved at [`eff21b6d7d33c622313122f46c6ee71693c101ee`](https://github.com/mlakolar/ProximalBase.jl/tree/eff21b6d7d33c622313122f46c6ee71693c101ee)
and `legacy/full`. Those archives are historical references, not a second live
API. This development reduction is not a release announcement.

## Retained and removed scope

The [function library](function_library.md) is exhaustive. Group/nonconvex
penalties, logistic/Poisson/robust-loss families, constraints and indicators,
spectral/nuclear families, generic composition/sum wrappers, LMO, second-order
frameworks and GPU extensions were removed. They have no replacement capability
implied by this manual. In particular, `BlockIterate` retains grouped **storage**,
not a built-in group-lasso penalty.

Keep two function objects for `f(x)+g(x)`; no generic function-object sum is
retained. Compute mathematical gradient mappings in caller code. Coordinate
curvatures and residual/working-set policies are algorithm responsibilities.
ProximalSolvers migration is separate; this documentation does not change it.

## Names, coordinates and writes

| Historical code | Retained interface |
|:--|:--|
| `AtomIterate`, `atom`, `atomindices` | `BlockIterate`, `blockview`, `blockindices`; no compatibility aliases |
| Scalar `prox_entry` | `prox_block(f,v,i,γ)` |
| `prox_coordinate!`, `prox_entry!`, `cdprox!` storage updates | Compute a pure result, then write through public storage; no retained aliases |
| Function-first mutating calls | Destination-first `prox!(out,f,x,γ)` and `gradient!(out,f,x)` |
| Iterate `nnz`, `nonzeros`, `findnz` assumptions | `nactive`, `activecoordinates`, `activevalues` (latter absent on `BlockIterate`) |
| Treat symmetric owner as `AbstractMatrix` | `coordinates(x)` for variables, read-only `matrixview(x)` for matrix operations |
| Symmetric-owner `x[k]` | `coordinates(x)[k]` or `matrixview(x)[k]`, depending on index space |
| Symmetric-owner broadcast/fill | Write coordinates, or compute a dense symmetric matrix then `copyto!` |
| Symmetric array steps | Owner outputs accept scalar steps only; use dense output for entry-step calculations where supported |
| User separable parameter arrays without metadata | Declare `ProximalBase.entry_parameters` and validate pair symmetry |

The working set survives value writes. `fill!(x,0)` followed by `dropzeros!(x)`
clears array iterates; use `fill!(coordinates(x),0)` for a symmetric owner.
Same-kind matching state copies replace active state; ordinary broadcasts copy
values and keep previous activity. Read the [storage guide](iterates.md) before
assuming conventional dense-array behavior.

## Historical contracts

Current [developer contracts](developer/index.md) are reconciled against retained
methods and tests. Historical acceptance records remain available at the
immutable archive, including pending requirements and measured experiments.
They are not automatically current guarantees. Numerical endpoints, aliasing
and decomposition caches must follow the [retained numerical contract](numerics.md).
