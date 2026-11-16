# Compatibility and migration

For terms used in the current manual, see the [Glossary](glossary.md).

This manual describes the minimal core after merged PR #850, based on
`2b373c54`. The full post-#848 implementation and its documentation are
preserved at [`eff21b6d7d33c622313122f46c6ee71693c101ee`](https://github.com/mlakolar/ProximalBase.jl/tree/eff21b6d7d33c622313122f46c6ee71693c101ee)
and `legacy/full`. Those archives are historical references, not a second live
API. This development reduction is not a release announcement.

## Retained and removed scope

The [function library](function_library.md) is exhaustive. Norms and group penalties
other than `NormL1`, `NormL2` and `GroupNormL2`, nonconvex penalties, logistic/Poisson/robust-loss families, constraints and indicators,
spectral/nuclear families, generic composition/sum wrappers, LMO, second-order
frameworks and GPU extensions were removed. They have no replacement capability
implied by this manual. `BlockIterate` retains grouped **storage**; the group-lasso
penalty `GroupNormL2` and the norm `NormL2` were restored separately (#900),
without the other group penalties and norms.

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
| `LinearFunction(c, d)(x)` at a matrix `SparseIterate` `x`, with `NaN` or `±Inf` in `c` at a position that `x` does not store or stores as `+0` or `-0` | `NaN`: the value was `dot(c, x) + d`, which formed the product of the coefficient and the zero at each position it visited (all `mn` positions for a dense `c`, the `p` diagonal positions for a `Diagonal`) | The value of the stored nonzero entries plus `d`, with `c` not read at a zero, in `O(nactive(x))` (`O(p)` for a `Diagonal` `c`, which scans the `p` diagonal positions). A nonzero entry of `x` keeps ordinary arithmetic, a `NaN` coefficient there gives `NaN`. The overflow recovery requires the coefficients at the nonzero entries to be finite, not all of `c`. `f(Matrix(x))` and `dot(c, x)` are unchanged and give `NaN` there |

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

## Changed traits and proximal domains

| Call | Before | Now |
|:--|:--|:--|
| `isconvex` of `QuadraticFunction`, its type or an instance, and of `Scaled` of it ([#919](https://github.com/mlakolar/ProximalBase.jl/issues/919)) | `true` | `false`, which means that convexity is not guaranteed for an arbitrary instance; `f` is convex when the symmetric `A` is positive semidefinite |
| `prox!`, `prox` and `value_and_prox!` of `QuadraticFunction`, directly or through `Scaled`, when the smallest eigenvalue `λmin` of the cached decomposition of `A` is negative and the rounded product `γT·(-λmin)` is not `< 1` (a NaN step included), where `γT` is the step after the factors of any `Scaled` are applied, converted to the element type of the function ([#919](https://github.com/mlakolar/ProximalBase.jl/issues/919)) | Unchecked spectral division by `1 + γλi`, which could return non-finite values or finite values that are not the proximal point | `DomainError` before the output is written |

A symmetric `A` need not be positive semidefinite. The check applies to a
singular positive semidefinite `A` whose computed `λmin` is slightly negative
from rounding, at steps with `γT·(-λmin) ≥ 1` (about `γ ≥ 1/|λmin|`); at those
steps the earlier kernel divided by a computed denominator `1 + γT·λmin ≤ 0`.
A matrix with `λmin ≥ 0`, and a negative step, are not checked and behave as
before. For a nonnegative finite `γT` and finite cached eigenvalues, the check
accepts exactly when every computed denominator `1 + γT·λi` is positive. In
exact arithmetic the proximal subproblem has a unique minimizer if and only if
`1 + γ λmin(A) > 0` for the true smallest eigenvalue, so near that boundary
the check can accept a step for which the true `I + γA` is not positive
definite, and reject one for which it is.

A caller that knows `A ⪰ 0` and relies on convexity asserts it at its own model
boundary, and does so again after changing `A`. A successful prox does not
establish convexity.

## Historical contracts

Current [developer contracts](developer/index.md) are reconciled against retained
methods and tests. Historical acceptance records remain available at the
immutable archive, including pending requirements and measured experiments.
They are not automatically current guarantees. Numerical endpoints, aliasing
and decomposition caches must follow the [retained numerical contract](numerics.md).
