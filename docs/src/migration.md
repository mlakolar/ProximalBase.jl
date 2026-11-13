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

## Restored functions

| Function | Restored from the archive | Differences from the archived version |
|:--|:--|:--|
| `ElasticNet(λ1, λ2, w)` ([#908](https://github.com/mlakolar/ProximalBase.jl/issues/908)) | Constructor, fields `λ1`, `λ2`, `w`, value, `prox_block`, `value_entry`, array proximal operators, KKT screen, `restrict`, `Scaled` folds and zero scaling | Built on the retained `NormL1` threshold and `SqrNormL2` ridge kernels, so ordinary results can differ in the last bits (multiplication by the rounded `1/(1 + γλ2)`). An infinite `λ1` follows the current `NormL1` rule (`+0` for positive weights, also at `γ = 0` and for `NaN`/`±Inf` inputs) where the archive returned `NaN`. An infinite `λ2` is the indicator of `{0}`. A negative or `NaN` scalar step throws `ArgumentError` before writes; the archive applied the arithmetic |

## Names, coordinates and writes

| Historical code | Retained interface |
|:--|:--|
| `AtomIterate`, `atom`, `atomindices` | `BlockIterate`, `blockview`, `blockindices`; no compatibility aliases |
| Scalar `prox_entry` | `prox_block(f,v,i,γ)` |
| `prox_coordinate!`, `prox_entry!`, `cdprox!` storage updates | Compute a pure result, then write through public storage; no retained aliases |
| Function-first mutating calls | Destination-first `prox!(out,f,x,γ)` and `gradient!(out,f,x)` |
| Iterate `nnz`, `nonzeros`, `findnz` assumptions | `nactive`, `activecoordinates`, `activevalues` (latter absent on `BlockIterate`) |
| Treat symmetric iterate as `AbstractMatrix` | `coordinates(x)` for variables, read-only `matrixview(x)` for matrix operations |
| Symmetric-iterate `x[k]` | `coordinates(x)[k]` or `matrixview(x)[k]`, depending on index space |
| Symmetric-iterate broadcast/fill | Write coordinates, or compute a dense symmetric matrix then `copyto!` |
| Symmetric array steps | Symmetric-iterate outputs accept scalar steps only; use dense output for entry-step calculations where supported |
| User separable parameter arrays without metadata | Declare `ProximalBase.entry_parameters`; call `ProximalBase.check_symmetric_parameters` once per model, because the oracles assume symmetric parameters and do not compare them |

The working set survives value writes. `fill!(x,0)` followed by `dropzeros!(x)`
clears array iterates; use `fill!(coordinates(x),0)` for a symmetric iterate.
Same-kind matching state copies replace active state; ordinary broadcasts copy
values and keep previous activity. Read the [storage guide](iterates.md) before
assuming conventional dense-array behavior.

## Changed numerical results

| Call | Before | Now |
|:--|:--|:--|
| `NormL1(Inf, w)` at step `γ = 0` ([#665](https://github.com/mlakolar/ProximalBase.jl/issues/665)) | Input returned unchanged (`shrink(x, 0⋅Inf)` read as a zero threshold) | Projection onto the closure of the domain: `+0` where `w[i] > 0`, including `NaN`/`±Inf` inputs; unchanged where `w[i] = 0`. Unweighted `NormL1(Inf)` gives all `+0` |

Positive steps, finite levels at `γ = 0`, infinite coordinate weights at a
finite level (#853) and finite-factor overflow (#866) are unchanged. See
[NaN, infinities and zeros](numerics.md#NaN,-infinities-and-zeros).

## Changed argument checks

| Call | Before | Now |
|:--|:--|:--|
| `LinearFunction(c, d)` with `c` and `x` of the same length but different shapes, such as a vector `c` and a matrix `x`, or a `2 × 3` `c` and a `3 × 2` `x` | Accepted or refused depending on which `dot` method the array types reached: dense pairs were accepted (the prox paired entries in linear order), sparse pairs were refused by the value but accepted by `gradient!` and `prox!` | `DimensionMismatch` from every oracle, before any output is written. `c` must satisfy `size(c) == size(x)`; on a `SymmetricSparseIterate` it must be `p × p`, including for `kkt_violation` |

Pairs of the same size are unchanged, with the same values. To keep a model
written with a flattened coefficient vector, reshape it once to the shape of the
argument, as in `LinearFunction(reshape(c, size(x)), d)`.

## Historical contracts

Current [developer contracts](developer/index.md) are reconciled against retained
methods and tests. Historical acceptance records remain available at the
immutable archive, including pending requirements and measured experiments.
They are not automatically current guarantees. Numerical endpoints, aliasing
and decomposition caches must follow the [retained numerical contract](numerics.md).
