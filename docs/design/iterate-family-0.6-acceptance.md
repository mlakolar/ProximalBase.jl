# The iterate family in 0.6: compatibility and acceptance

> **Draft for maintainer review — proposed 0.6 behavior, not a description of an
> installed release.** This plan explains how callers move to the
> [contract](iterate-family-0.6.md) and how a future implementer proves
> conformance. It links to the contract's requirement IDs (FS, FC, FW, FX, FL,
> FN, FA, FP, FF) and decisions (FD-n) and restates none of the rules. Nothing
> in it is implemented. Every code block that uses `atomindices`,
> `ProximalBase.samegroups`, a read-only handle or a 0.6 rule is a *future*
> example; none is a doctest.

# 1. What was audited

**Commits.** ProximalBase `def81ec9d021c39ed1727cfeb9598357291dd60d` (master,
`Project.toml` 0.5.4). ProximalSolvers `3893fb95da9ab7326530c46b24fed801dbac1e74`
(`origin/master`), read with `git show` and `git grep` at that commit and not
modified. The SparseArrays and Base sources are those of Julia 1.13.3, with
the Base routing also checked on Julia 1.10.12.

**Traced in detail** (ProximalSolvers; every line cited was read at the
audited commit):

| path | what it does with an iterate | finding |
|:--|:--|:--|
| `src/cd/groups.jl:19–48`, `:50–80` | group steps on an `AtomIterate` for any penalty: reads `x.groups[k]` (`:20`, `:52`), writes the trial point into `x.storage` (`:32–37`, `:64–69`), then calls the four-argument `prox_coordinate!` (`:38`, `:70`) | FW-1 fails during the call, and for good if it throws (FA-7). Recipe §3.3 |
| `src/cd/multiresponse.jl` | `B` as `AtomIterate` over `vec(B)` with groups `collect(k:p:p*q)` (`:45`, `:47–48`); the penalty's layout check (`:55`); `x.groups[k]` reads (`:75`, `:119`); public writes `x[j] = new` (`:150`); the gradient over `vec(B)` through `reshape(G, p, q)` (`:180–196`) | matches FS-4; the only private use is the two indexed reads |
| `src/cd/solver.jl:190–212`, `:270–278`, `:318–335`, `:494–530` | momentum history keyed by slot, restarted when `nactive` changes (`:518`) and after full passes; `dropzeros!` after each pass (`:276`); active passes over a handle without activation | consistent with FC-7's lifetime |
| `src/cd/solver.jl:540–602` | the momentum values: `x.nzval` for `SparseIterate` and owner, `x.active`, `x.nactive`, `x.groups`, `x.storage` for `AtomIterate` | 16 private-field uses and 4 indexed reads. Recipe §3.4 |
| `src/cd/solver.jl:640–677` (Anderson, `coordinates(x)` at `:648`) | linear reads and writes by coordinate ID; gated off for `AtomIterate` at `:345` | unaffected by FD-5 |
| `src/solvers/active_accelerated_proximal_gradient_descent.jl:125–304` | `dropzeros!` (`:150`); the gradient at `x` (`:151`); a broadcast read into a dense buffer (`:159`); `activate!` (`:191`); scalar reads (`:199`) and writes to active entries (`:203–206`); the whole-layout check (`:264–266`); indexed reads (`:272`, `:283`); a handle read (`:277`); screening of inactive coordinates (`:294–302`) | reads only; no write through a handle; no reliance on `fill!` clearing. FN-5 applies at `:151` only with non-finite data |
| `src/types.jl:210` | `OptimizationResults` constrains the minimizer to `AbstractArray` or blocks | affects only the owner (FD-15c) |
| `src/cd/path.jl:8–10`, `:261`, `:342` | `_zero_like` rebuilds an `AtomIterate` from `x.groups` (`:10`); the continuation starts with `fill!(x, zero(T))` (`:261`) and relies on it clearing the working set; an indexed read (`:342`) | `:261` changes silently under FD-2. Recipes §3.2, §3.5 |
| `src/cd/models.jl:691` (and `:429`, `:473–478`) | whole-layout equality of iterate and penalty; an indexed read; penalties rebuilt from `g.groups` at each path level | `samegroups` (§3.2); `:473–478` read a penalty field |
| `src/workspaces.jl:153` (and `:93–95`, `:140–149`, `:243–246`) | whole-layout equality of two iterates; `reinit!` copies a same-type start with `_copyto!` (a state copy, unchanged); a CD workspace loads a dense start with `fill!(xs, 0)` and a scatter | `:243` changes silently under FD-2 (§3.5) |
| `src/cd/gap.jl:200`, `:719` | `_coordentries(x::AtomIterate, k) = x.groups[k]`; a loop over `x.groups[k]` with public writes; the other `.groups` reads in `gap.jl` are penalty fields | two indexed reads |
| `src/cd/leastsquares.jl:192` (and `:558`, `:586`, `:598`) | `Iterators.flatten(x.groups[k] for k in activecoordinates(x))`; `m.X[:, G]` with a group as index | indexed reads; a wrapped vector group becomes the index at `:586` (benchmark, FP-6) |
| `src/solvers/diagonal_metric.jl:106`, `:114` | `_copyto!(x, z)` in the diagonal-metric proximal step | `x` can be a `SparseIterate` or `AtomIterate` (`_diagonal_metric` takes any floating-point array, `:12`; the owner is rejected at `:23–24`). `z` is `state.xhat`, of the type of `x` (`src/solvers/proximal_gradient_descent.jl:67–77`, `:121`, `:156–157`), so the copy is a same-kind state copy, unchanged by FD-2 |
| `src/solve.jl:4–6`, `src/blocks.jl:23–40` | `_drop!` is `dropzeros!`; `_copy!` is `copy!` (a same-kind state copy); `_zero` is `zero` (empty) | unchanged |

**Mentioned but not traced** (they touch iterates; I read only the lines
cited): ProximalSolvers `src/cd/groupcurvature.jl:124`, `src/cd/quadratic.jl:148,
175`, `src/cd/sparsegroup.jl:108, 118, 125, 181` (indexed reads),
`src/cd/drivers.jl:95, 115, 150` (`coordinates` by ID), `src/cd/quic.jl:138,
203, 214` (`fill!` on an owner, a #391 Phase 2 item), `src/cd/lla.jl`,
`src/cd/glm.jl`, `src/solvers/accelerated_proximal_gradient_descent.jl`,
`src/solvers/monotone_accelerated_proximal_gradient_descent.jl` and
`src/solvers/admm.jl` (copy patterns seen in a search only; ADMM workspaces
take dense arrays, `src/workspaces.jl:82`).

**Not audited.** Other packages that use the iterates (CovarianceSelection,
DiffPrecTest, HighDimensionalModels and others), ProximalSolvers' tests and
examples, and any private package. A search miss there is not evidence of
compatibility.

**Do the data losses check that their data are finite?** Not in general. Of
those checked, `LeastSquaresLoss` (`src/functions/leastsquares.jl:48–59`) and
`QuadraticFunction` (`src/functions/smooth.jl:345–351`) do not; `GLMLoss`
checks only its weights and offset (`src/functions/glm.jl:190–193`);
`KronQuadratic` checks (`src/functions/smooth.jl:492–493`). So FD-9 changes
oracle values, not only direct `dot` and `mul!` calls: at an iterate with a
stored zero against a non-finite design entry, `LeastSquaresLoss` goes from NaN
to finite (FN-5). ProximalSolvers' coordinate-descent models check their own
data (`_check_finite`, ProximalSolvers `src/cd/multiresponse.jl:34`,
`src/cd/leastsquares.jl:148`), so its sweeps do not change.

**Checks run.** Short scripts against the audited source with Julia 1.13.3,
kept outside the repository, gave every "ran" value of the contract's Part 4:
the handles and their corruption; counts; every write path from a working set
with an explicit zero; the worked example of FD-2 and the owner table of FD-3;
copies, imports and length checks; conversion and activation in `Float32`;
complex sources; non-finite writes; empty and malformed layouts; layout sharing,
the one-slot cache and whole-array fallbacks; coordinate-oracle validation,
returns and aliasing; the block trial against private staging; products with a
NaN or `Inf` against stored, unstored and dense zeros, the generic fallbacks and
SparseArrays; a data-loss value; aliasing hooks; retained memory; the
multi-response mapping; the examples of Part 1. A script on Julia 1.10.12 and
1.13.3 traced which Base entry points reach `fill!` or `copyto!`. Not run: the
package test suite, the benchmark suite and any downstream test.

# 2. Reconciliation with the specialized issues

| issue | fixed on master | decided here | remains for that issue |
|:--|:--|:--|:--|
| #137 API additions and 0.6 decisions | `findnz` and `nonzeros` for `SparseIterate` (`src/sparse_iterate.jl:448`, `:457–471`) | the iterate-contract item for S and A: FD-6, FD-10, FD-13; the owner by #388 | the non-iterate items (`iszeropreserving`, tuple calculus, `public` declarations, the validation hook, `Testing` defaults, non-public stdlib names, deprecations) |
| #189 writes through `atom` (closed) | `atom(x, k)` is `view(x, G)` and activates (`src/atom_iterate.jl:163–166`); regression tests `test/atom_iterate.jl:244–269`; the block-trial form exists (`src/interface.jl:469–478`) | kept (FC-5); the five-argument form is the supported trial (FA-7) | nothing; ProximalSolvers' private staging, which #189's decision found, migrates (§3.3) |
| #192 what `nnz` counts | — | FD-10: SparseArrays' meaning on `SparseIterate`, no `nnz` on `AtomIterate`, four named counts (FC-10, FC-11) | the owner part (ACT-2, #391); other packages' `1:nnz` loops over fields move to `activevalues` |
| #341 value oracles reading every entry | — | FN covers products, not these values. Numerics decision 47 rejects non-finite `LinearFunction` coefficients, so FN-1 does not change `LinearFunction`'s value | all four oracles, under #366 G2 |
| #364 views bypass the real check | — | FD-13: the iterate's own bulk writes check first (FX-4); wrappers follow Base (FN-8), with no overrides. The complex `copyto!` partial write and the `vec(xm)` broadcast are recorded on #364 | the `copyto!` check (a 0.5.x bug fix) and the docstrings |
| #366 numerical contract | — | FD-9's amendment of M1, E2, E3, C1 and S4, word for word; FX-4 and FX-5 follow O6.1 and O7 | sign-off of the amendment; G2 implements FN-1; G12 covers FX-4 and FX-5 |
| #367 aliasing audit | — | FX-5: imports from a handle follow Base's `unalias`; FA-5 holds today | the oracle-by-oracle audit |
| #368 owner views unchecked | — | nothing new; for S and A, writes through wrappers follow FN-8 | expected to close with #391 (MAT-2) |
| #391 owner implementation | Phase 0: `matrixview`, `activevalues`, owner `dataids` (#392); the `owner` benchmark group (#401) | FD-3 amends CPY-4, decision 4, ACT-5, T10; FD-9 amends decision 13, PERF-6's default, T13; FD-7 shares ACT-3's handle; FD-4 and FD-5 are consistent with ORC-3, ORC-6, CRD-1 | Phase 1 implements the owner parts of FD-3, FD-7, FD-9; Phase 2 the downstream recipes |
| #342 groups on an owner | — | out of scope: FL applies to `AtomIterate` | as is |
| #355 per-group `Regularize` steps on `AtomIterate` | — | its destination follows FW-8 | as is |
| #405 writable `activecoordinates`, `AtomIterate` `dropzeros!` hang | — | fixed by FD-7 (FC-7) | filed; a 0.5.x bug fix |
| #406 `dataids` misses the handles | — | fixed by FD-14 (FX-5) | filed; a 0.5.x bug fix |

# 3. Migration recipes (0.5.x → 0.6)

## 3.1 The 23 indexed reads of `x.groups[k]`

Each becomes `atomindices(x, k)` (FC-6). All are ProximalSolvers:

| file | lines |
|:--|:--|
| `src/cd/gap.jl` | 200, 719 |
| `src/cd/groups.jl` | 20, 52 (§3.3) |
| `src/cd/leastsquares.jl` | 192, 558, 598 |
| `src/cd/models.jl` | 429 |
| `src/cd/multiresponse.jl` | 75, 119 |
| `src/cd/path.jl` | 342 |
| `src/cd/quadratic.jl` | 148, 175 |
| `src/cd/solver.jl` | 556, 566, 578, 597 (§3.4) |
| `src/cd/sparsegroup.jl` | 108, 118, 125, 181 |
| `src/solvers/active_accelerated_proximal_gradient_descent.jl` | 272, 283 |

```julia
# 0.5.x, src/cd/leastsquares.jl:192
_activeentries(x::AtomIterate) = Iterators.flatten(x.groups[k] for k in activecoordinates(x))
# 0.6 (future)
_activeentries(x::AtomIterate) = Iterators.flatten(atomindices(x, k) for k in activecoordinates(x))
```

## 3.2 The five whole-layout reads

| site (ProximalSolvers) | 0.5.x | 0.6 (future) |
|:--|:--|:--|
| `src/cd/path.jl:10` | `AtomIterate(T, length(x), x.groups)` | `similar(x)` |
| `src/cd/groupcurvature.jl:124` | `maximum(length, x.groups; init = 0)` | `maximum(k -> length(atomindices(x, k)), 1:ncoordinates(x); init = 0)` |
| `src/cd/models.jl:691` | `x.groups === g.groups \|\| x.groups == g.groups` | `ProximalBase.samegroups(x, g)` |
| `src/solvers/active_accelerated_proximal_gradient_descent.jl:265` | the same | `ProximalBase.samegroups(x, g)` |
| `src/workspaces.jl:153` | `x.groups == y.groups` | `ProximalBase.samegroups(x, y)` |

## 3.3 Group trials in `src/cd/groups.jl`

The trial point goes into the model's buffer and through the five-argument
form (FA-7); `x` is written only by `prox_coordinate!`.

```julia
# 0.6 (future); the QuadModel method changes the same way, with old - buf[i] / L
function _groupstep!(m::LSModel{T}, g::AbstractFunction, x::AtomIterate{T}, k::Int, scaled::Bool) where {T}
  G = atomindices(x, k)
  n = length(G)
  buf = _groupbuffer!(m, 2n)
  for (i, j) in enumerate(G)
    a, b = _column_dot(m, j)
    m.a[j] = a
    buf[i] = b
  end
  L = _groupcurvature!(m, x, k)
  for (i, j) in enumerate(G)
    old = x[j]
    buf[n + i] = old
    buf[i] = iszero(L) ? old : old + m.s * buf[i] / L      # the trial point, not written into x
  end
  prox_coordinate!(x, g, k, view(buf, 1:n), iszero(L) ? floatmax(T) : inv(L))
  maxh = zero(T)
  for (i, j) in enumerate(G)
    h = x[j] - buf[n + i]
    iszero(h) && continue
    _residual_axpy!(m, j, h)
    maxh = max(maxh, _scaledchange(m, x, j, h, scaled))
  end
  maxh
end
```

`view(buf, 1:n)` is a strided `Float32`/`Float64` vector, so the group-norm
kernels take the staging loop of `src/interface.jl:484–493`; `buf` is the
model's (`src/cd/leastsquares.jl:636`) and does not alias `x`.

## 3.4 The momentum history in `src/cd/solver.jl:553–600`

```julia
# 0.6 (future): SparseIterate and owner, through the activevalues handle (FC-8)
_values!(v::Vector, x::_StoredValues, n::Int) = copyto!(v, 1, activevalues(x), 1, n)
_setvalues!(x::_StoredValues, v::Vector, n::Int) = copyto!(activevalues(x), 1, v, 1, n)
function _restartdot(s::_Momentum{T}, x::_StoredValues, n::Int) where {T}
  av, d = activevalues(x), zero(T)
  @inbounds for j in 1:n
    d += (av[j] - s.y[j]) * (s.xcur[j] - s.xprev[j])
  end
  d
end

# 0.6 (future): AtomIterate, through activecoordinates and atomindices (FC-9)
function _nvalues(x::AtomIterate)
  n = 0
  for k in activecoordinates(x)
    n += length(atomindices(x, k))
  end
  n
end
function _values!(v::Vector, x::AtomIterate, n::Int)
  i = 0
  @inbounds for k in activecoordinates(x), j in atomindices(x, k)
    i += 1
    v[i] = x[j]
  end
  v
end
function _setvalues!(x::AtomIterate, v::Vector, n::Int)
  i = 0
  @inbounds for k in activecoordinates(x), j in atomindices(x, k)
    i += 1
    x[j] = v[i]          # the group is active, so this never changes the working set
  end
  x
end
# _restartdot(s, x::AtomIterate, n) loops the same way, reading x[j]
```

`x[j] = v` on an active group converts and stores, and its activation test is
a no-op (FW-7), so the values and the working set are those of the private
version. Its extra work per entry is a benchmark case (FP-6).

## 3.5 `fill!(x, 0)` used as a clear

| site (ProximalSolvers) | 0.5.x | 0.6 (future) |
|:--|:--|:--|
| `src/cd/path.jl:261` | `fill!(x, zero(T))` | `dropzeros!(fill!(x, zero(T)))` for S and A; `dropzeros!(fill!(coordinates(x), zero(T)))` for an owner |
| `src/workspaces.jl:243–246` | `fill!(xs, zero(eltype(xs)))`, then `xs[i] = ws.x[i]` for each nonzero | `dropzeros!(copyto!(xs, ws.x))` |
| `src/cd/quic.jl:138, 203, 214` (owner) | `fill!(X, zero(T))` | `dropzeros!(fill!(coordinates(X), zero(T)))` (#391 Phase 2) |

Without the change, the first two start from the caller's previous working
set and raise no error; the owner sites fail loudly, since the 0.6 owner has
no `fill!`.

## 3.6 A replacing import

```julia
fill!(x, 0); copyto!(x, A)          # 0.5.x
dropzeros!(copyto!(x, A))           # 0.6 (FW-6): the working set is the support of A
dropzeros!(copyto!(X, M))           # the owner (FD-3)
```

## 3.7 Code that wrote through `activecoordinates`

No audited site does. Such code used the handle as a mutable list; under FC-7
the write throws. Change the working set with `activate!` and `dropzeros!`;
for a mutable copy write `collect(activecoordinates(x))`.

## 3.8 The other private-field uses

Sixteen in ProximalSolvers, all replaced by §3.3 and §3.4:

| file | lines | fields |
|:--|:--|:--|
| `src/cd/groups.jl` | 32, 64 | `x.storage` |
| `src/cd/solver.jl` | 555, 556, 562, 566 (two), 568, 574, 578 (two), 580, 590, 597 (two), 599 | `x.nactive`, `x.active`, `x.nzval`, `x.storage` |

# 4. Conformance matrix

Key: *VW* value write (FW-2); *SC* state copy (FW-4); *e* entry index; *ID*
coordinate ID; "today" marks a cell where the audited code differs.

| operation | `SparseIterate` vector | `SparseIterate` matrix | `AtomIterate` | owner |
|:--|:--|:--|:--|:--|
| construction | `SparseIterate(T, n)` empty; `SparseIterate(v)` activates the nonzeros in index order; complex `T`: `ArgumentError` (FS-3) | `SparseIterate(T, m, n)`; from `Matrix` or `SparseMatrixCSC` | validates the layout (FL-1) and owns it (FL-2), or shares a penalty's (FX-6); groups with a nonzero active in group order | OWN-1, OWN-5; `SymmetricSparseIterate(M)` as CPY-4 |
| scalar read | e; `O(1)`; absent reads zero, activates nothing | e or `(r, c)` | e | `X[i, j]` (OWN-2) |
| scalar write | e; VW | e or `(r, c)`; VW | e; VW of its group | CRD-3 |
| first activation | `O(1)`, no growth (FP-1) | same | the group joins, `O(1)` | PERF-1 |
| `coordinates` | `=== x` | `vec(x)` (FC-3; today `x` itself) | `MethodError` (FC-4) | CRD-1 |
| `atom`, `atomindices` | `MethodError` | `MethodError` | group ID; a writable view, a read-only index list (FC-5, FC-6; `atomindices` today missing); `BoundsError` | `MethodError` |
| `activecoordinates`, `activevalues` | read-only handle (FC-7; today writable, #405); `activevalues == nonzeros`, writable (FC-8) | same | read-only handle (today writable); `activevalues`: `MethodError` (FC-9) | ACT-3, ACT-4 |
| `activate!` | ID; no-op if active; `BoundsError` (FW-7) | ID | group ID | ACT-6 |
| `dropzeros!` | removes zeros only; keeps NaN, `Inf` (FW-5, FN-7) | same | removes all-zero groups | ACT-7 |
| `fill!(x, c)` | VW; `c = 0` keeps the set (today clears); complex `c`: `ArgumentError` (today `InexactError`, nothing written) | same | VW; converts first (FN-6; today `1e-100` activates every group at `0f0`) | no `fill!` (OWN-3); `fill!(coordinates(X), c)` is a VW |
| `x .= c` | VW (today clears for `c = 0`); complex: `ArgumentError` | same | same | none (OWN-3); `θ .= c` (CRD-5) |
| `x .= v`, `x .= f.(x, v)`, `x .*= 0` | VW, the loop (FW-3) | same | same | `θ .= …` (CRD-5) |
| `x .= y`, `y` same-kind | VW (today SC) | same | same (today SC) | — |
| `copyto!`/`copy!` from a same-kind iterate | SC, element type may differ (FX-1); `copyto!(x, x)` no-op | same | same layout (FL-4): SC | CPY-3 |
| from another shape or layout | same length: VW (today `DimensionMismatch` for a matrix source); other length: `DimensionMismatch` (FX-3) | same | same length: VW (today too); other length: `DimensionMismatch` | other `p`: `DimensionMismatch` (CPY-3) |
| from a plain array | VW in linear order (FX-2); a longer source: `BoundsError`; `copy!` from another length: `MethodError` (`resize!`) | VW; a longer source: `BoundsError`; `copy!` with other axes: `ArgumentError` (Base) | as the vector | checked VW (FD-3): axes `DimensionMismatch`, asymmetry or complex `ArgumentError`, all before writing |
| from an aliased source | VW of a copy of the source (FX-5; today `copyto!(x, activevalues(x))` is wrong, #406) | same | same | `ArgumentError` (CPY-6) |
| from a complex source | `ArgumentError` before writing (FX-4; today `InexactError` after a partial write) | same | same | `ArgumentError` before writing |
| `copy` | SC | SC | SC; shares the layout | CPY-1 |
| `similar` | empty, same size; `similar(x, S)` | same | empty, shares the layout | CPY-2 |
| `nnz`, `nonzeros`, `findnz` | explicit zeros included (FC-11) | `findnz` gives `(I, J, V)` | `nnz`: `MethodError` | ACT-2 |
| `dot`, `mul!`, `*` | strong zeros (FN-1, FN-2; today storage- and order-dependent); fallbacks IEEE (FN-3) | both walks of `mul!`, both paths of `A * X * B`, `dot(a, X, b)`: strong zeros | `mul!` skips zeros (today too); `dot`: strong zeros (today not) | PERF-6 as amended by FD-9; ORC-10 |
| `prox_coordinate!(x, f, k, γ)` | ID; returns the scalar; FA-3, FA-4 | ID | group ID; returns `atom(x, k)`; layout checked (FL-4) | ORC-5 |
| `prox_coordinate!(x, f, k, u, γ)` | scalar `u` | scalar `u` | `u` of the group's length, may alias `x` (FA-5); unchanged on failure (FA-7) | ORC-6 |
| `kkt_violation(f, g, x, k)` | the scalar screen | same | the group screen (FA-6) | ORC-9 |
| `prox!(y, f, x, γ)` into the iterate | FW-8: values, every nonzero active, zeros unspecified | same | same; a different layout takes the generic path (FL-5) | ORC-3 |

**Edge cases** (expected outcomes; "ran" marks today's behavior where it is
the same):

- **Empty.** `SparseIterate(0)`: `length`, `ncoordinates`, `nactive` and
  `length(coordinates(x))` are 0, `dropzeros!` is a no-op, `x[1]` throws
  `BoundsError`, `dot(Float64[], x) == 0.0`, `mul!(zeros(2), zeros(2, 0), x) ==
  [0, 0]` (ran). `SparseIterate(0, 3)`: size `(0, 3)`, no coordinates (ran).
  `AtomIterate(0, UnitRange{Int}[])`: `length`, `ncoordinates`, `nactive` 0,
  `dropzeros!` a no-op, `atom(x, 1)` and `atomindices(x, 1)` throw
  `BoundsError` (ran for `atom`). `AtomIterate(4, UnitRange{Int}[])` throws
  "The groups must partition 1:4." and `AtomIterate(0, 0)` "num_groups must
  divide p." (ran). The owner at `p = 0` is #388's T01.
- **All zero.** `SparseIterate(zeros(4))` and `AtomIterate(zeros(4), 2)` have
  `nactive == 0` (ran).
- **Active zeros.** `activate!` makes one; value writes keep it; a state copy
  and `copy` keep it; only `dropzeros!` removes it (FW-2, FW-4, FW-5).
- **Compaction and handles.** After `dropzeros!`, a handle taken before reads
  stale slots (ran: `[3, 2, 3]` against the new `[3, 2]`); tests use only a
  handle taken after the call and compare sets, not order (FC-7).
- **Aliased inputs.** `copyto!(x, activevalues(x))` copies the source first
  (FX-5); `copyto!(x, view(x, 4:-1:1))` reverses correctly (ran, both types);
  `x .= 2 .* x` doubles; `prox_coordinate!(a, f, 1, atom(a, 2), γ)` copies `u`
  (ran: `[2.5, 3.5]` for `NormL1(0.5)` and `atom(a, 2) == [3, 4]`); `prox!(x, f,
  x, γ)` follows numerics O7.
- **Malformed groups** (`ArgumentError`, ran):
  - `AtomIterate(4, [1:2, 2:4])`: "Groups must be disjoint, but index 2 is in
    groups 1 and 2."
  - `AtomIterate(4, [1:2, 4:4])` and `AtomIterate(4, [1:2, 3:5])`: "The groups
    must partition 1:4."
  - `AtomIterate(4, [1:2, Int[], 3:4])`: "Groups must not be empty."
  - `AtomIterate(4, [0:1, 2:4])`: "Group indices must be positive."
  - `AtomIterate(5, 2)`: "num_groups must divide p."
- **`Float32` and `Float64` mixed.** Writing `1e-100` into a `Float32` iterate
  activates nothing through every write path: `x[i] =`, `fill!`, `x .= c`,
  `x .= v`, `x .= 1 .* v`, `copyto!`, `x[:] =`, offset `copyto!`, `view .=`,
  `vec .=`, `atom .=`, and a `prox_coordinate!` whose result rounds to zero
  (ran; today `fill!` and `x .= c` of an `AtomIterate` activate both groups at
  `0f0`). A `Float64 → Float32` state copy keeps `1e-50` as an active `0f0`
  (ran, FX-1); a value write of `1e-50` from a dense `Float64` array activates
  nothing (ran).
- **Non-finite values.** NaN and `±Inf` activate their coordinate, `−0.0`
  does not, and `dropzeros!` keeps NaN and `Inf` (ran). A NaN or `Inf` of a
  dense operand that meets only zero coefficients of an iterate does not reach
  the result of an FN-2 method; one that meets a nonzero coefficient does
  (FN-1).

# 5. Acceptance tests

Each row is a named future test; none exists. "Reference" says how the
expected value is obtained *independently*: by hand, by an explicit loop
written in the test, or by a dense computation, never by the kernel under
test. "Fails today" marks a test that fails at the audited commit.

| test | requirements | setup | exact expectation or independent reference | likely home |
|:--|:--|:--|:--|:--|
| **FT01 coordinates and index spaces** | FC-1, FC-2, FC-3, FS-1 | `x = SparseIterate(Float64, 2, 3)`; `v = SparseIterate(4)` | `c = coordinates(x)`: `c isa AbstractVector{Float64}`, `length(c) == 6 == ncoordinates(x)`, `axes(c, 1) == 1:6`; `c[4] = 2.0` gives `x[2, 2] == 2.0`, `activecoordinates(x) == [4]` (4 = (2 − 1)·2 + 2 by hand); `activate!(x, 6)` makes `x[2, 3]` an explicit zero; `coordinates(v) === v`; `@allocated coordinates(x) == 0` after warm-up. *Fails today* (`c` is 2 × 3) | `test/iterate_queries.jl` |
| **FT02 groups, entries, multi-response** | FC-4, FC-5, FC-6, FS-4, FP-2 | `p, q = 3, 2`; `B = [1 2; 3 4; 5 6]`; `x = AtomIterate(vec(B), [collect(r:3:6) for r in 1:3])`; `y = AtomIterate(6, 3)`; `Y = [1 0; 0 1]`, `X = [1 2 0; 0 1 1]` | `atomindices(x, 2) == [2, 5]`; `atomindices(y, 2) == 3:4` and is a `UnitRange{Int}`; `atomindices(x, 2)[1] = 9` throws (any exception) and the layout is unchanged; `@allocated atomindices(x, 2) == 0`; `atomindices(x, 0)` and `atomindices(x, 4)` throw `BoundsError`; `atom(x, r) == B[r, :]` and `x[(c − 1)·3 + r] == B[r, c]` for all `r, c`; `coordinates(x)` throws `MethodError`; with `G = X' * (X * B − Y)` computed densely in the test, `vec(G)[atomindices(x, r)] == G[r, :]`; a view `w = atom(x, 2)` taken before `activate!(x, 1)` and `dropzeros!(x)` still reads `B[2, :]`, and `w[1] = 0` writes `x[2]`. *Fails today* (no `atomindices`) | `test/atom_iterate.jl` |
| **FT03 read-only handles** | FC-7, FW-1 | `x = SparseIterate(4)` with `x[2] = 1`, `x[4] = 3`; `a = AtomIterate(6, [1:2, 3:4, 5:6])` with `a[1] = 1`, `a[5] = 2`; an owner with `X[2, 1] = 1` | `activecoordinates(x)[1] = 3`, `activecoordinates(a)[1] = 2` and the owner's write throw (any exception). Afterwards `Vector(x) == [0, 1, 0, 3]`, `activecoordinates(x) == [2, 4]`, `isactive(x, k) == (k in (2, 4))` for `k = 1:4`; `activecoordinates(a) == [1, 3]` and the consistency helper of `test/atom_iterate.jl:15` holds. Call `dropzeros!(a)` only after asserting that the write threw (today it hangs after a successful write). *Fails today* (#405) | `test/iterate_queries.jl` |
| **FT04 handles after compaction** | FC-7, FC-8, FW-5, FN-7 | `x = SparseIterate(5)` with values `1, 2, 3` at `1:3`; then `x[1] = 0`, `dropzeros!(x)`. A second `x` with `x[4] = NaN`, `x[5] = -Inf` and an explicit zero at 1; an `AtomIterate` with an all-zero active group and a NaN group | `Set(activecoordinates(x)) == Set([2, 3])`; `Dict(zip(activecoordinates(x), activevalues(x))) == Dict(2 => 2.0, 3 => 3.0)`, the same through `nonzeros`; no handle from before the call is used. `dropzeros!` keeps 4 and 5 and removes 1; for the `AtomIterate`, it removes only the all-zero group; values unchanged; `@allocated dropzeros!(x) == 0` | `test/sparse_iterate.jl`, `test/atom_iterate.jl` |
| **FT05 value writes never remove** | FW-2, FW-3, FW-7, FX-2 | `x = SparseIterate(4); x[4] = 7; activate!(x, 2)` (old set `[4, 2]`), fresh for each call; `v = [3, 0, 0, 0]`; `y = SparseIterate(4)` with `y[1] = 3` and an explicit zero at 3. The calls: `x[1] = 3`, `x .= v`, `x .= 0`, `x .= 0 .* x`, `x .= 1 .* v`, `x .= y`, `fill!(x, 0)`, `fill!(x, 2)`, `copyto!(x, v)`, `copy!(x, v)`, `x[:] = v`, `copyto!(x, 1, v, 1, 4)`, `view(x, 1:2) .= [3, 0]`, `vec(x) .= v`, `rmul!(x, 0)`, `x .*= 0`, `map!(identity, x, v)`, `ldiv!(x, lu(I₄), v)`, `permute!(x, [4, 1, 2, 3])`, `mul!(x, A, b)`, `sum!(x, M)`. The same for `a = AtomIterate(4, [1:2, 3:4]); a[4] = 7`, plus `atom(a, 1) .= [3, 0]` | Reference: the same call on `Vector(x)`, a plain `Vector`. Expect `Vector(x) == ref`; `activecoordinates(x)[1:2] == [4, 2]` (old slots kept, in order); `Set(activecoordinates(x)) == Set([4, 2]) ∪ Set(findall(!iszero, ref))` (for `x .= y`, `y`'s explicit zero at 3 does not join). For the `AtomIterate`, the same with groups. Every broadcast equals the explicit loop `for i in eachindex(x); x[i] = rhs[i]; end` run on a copy, values and working set. A sweep over `activecoordinates(x)` writing `x[k] = 0` leaves the handle's contents unchanged. Separately: `ldiv!(x, lu(I₄), y)` with a same-kind `y` gives `y`'s working set (FW-2's stated exception). *Fails today* for `x .= 0`, `fill!(x, 0)`, `x .= y`, `sum!` | `test/iterate_writes.jl` (new) |
| **FT06 state copies** | FW-4, FX-1, FX-6 | `y = SparseIterate(4); y[3] = 5; activate!(y, 1)`; `x` with coordinate 2 active; `y64 = SparseIterate(Float64, 3); y64[2] = 1e-50`; two `AtomIterate`s with equal but distinct layouts | `copy(y)`, `copyto!(x, y)`, `copy!(x, y)`: `activecoordinates == [3, 1]`, the explicit zero kept, values equal, `x`'s old 2 gone; `similar(y)` empty with the same size, `similar(y, Float32)` of `Float32`; `zero(y)` empty; `copyto!(y, y)` leaves `y` unchanged. `copyto!(SparseIterate(Float32, 3), y64)` gives values `Float32[0, 0, 0]` and working set `[2]` (by hand). The `AtomIterate`s copy as state; `copy`, `similar` and `AtomIterate(4, f)` share the layout object (the test may check identity on the private field) | `test/sparse_iterate.jl`, `test/atom_iterate.jl` |
| **FT07 imports and lengths** | FX-2, FX-3 | `x = SparseIterate(2, 2)` with `x[1, 1] = 5`; `s = SparseIterate(4)` with `s[4] = 1` and an explicit zero at 2; `d = AtomIterate(4, [[1, 3], [2, 4]])` with `d[2] = 7`; `src = AtomIterate(4, [1:2, 3:4])` with `src[3] = 5` and group 1 active at zero | `copyto!(x, s)`: `Matrix(x) == [0 0; 0 1]`, working set `[1, 4]` (old 1 kept, 4 joins, `s`'s explicit zero does not). `copyto!(SparseIterate(6), SparseIterate(4))` and `copyto!(AtomIterate(6, 3), AtomIterate(4, 2))` throw `DimensionMismatch` with the destination unchanged. `copyto!(d, src)`: `Vector(d) == [0, 0, 5, 0]`, working set `[2, 1]`. All by hand. *Fails today* for `copyto!(x, s)` | `test/iterate_writes.jl` |
| **FT08 replace and clear** | FW-6 | `x` as in FT05; `a` as in FT05; an owner with `X[2, 2] = 7` | `dropzeros!(copyto!(x, v))` has working set `[1]`; `dropzeros!(fill!(x, 0))` has none; the same with groups (`[1]`, none); `dropzeros!(copyto!(X, [3 0; 0 0]))` has `[1]`; `dropzeros!(fill!(coordinates(X), 0))` has none. Reference: `findall(!iszero, ·)` mapped to IDs by hand | `test/iterate_writes.jl` |
| **FT09 activation after conversion** | FN-6, FW-2 | `Float32` iterates of both types, fresh for each path; `u = [1e-100, 1e-100]` | Every path of the edge-case list with `1e-100` leaves `nactive == 0` and values `0f0`; the same write to an *active* coordinate stores `0f0` and keeps it active; NaN activates, `−0.0` does not. *Fails today* for `fill!(a, 1e-100)` and `a .= 1e-100` (2 groups) | `test/precision_mixed.jl` |
| **FT10 oracle destinations** | FW-8 | `y = SparseIterate(4); y[4] = 7`; `prox!(y, NormL2(λ), [3, 0, 0, 0], 1)` for λ = 0, 1, 10; an `AtomIterate` with group 2 active and `GroupNormL2(λ, [1:2, 3:4])`; `prox!(y, IndBallL0(s), SparseIterate([2, 1, 0, 0]), 1)` for `s = 0, 1, 3` | Values from closed forms written in the test: `max(1 − λ/3, 0)·[3, 0, 0, 0]`; for the group, group 1 is `max(1 − λ√2/3, 0)·[3, 0]` (3 − √2 = 1.5857864376269046 at λ = 1); `IndBallL0`: `[0, 0, 0, 0]`, `[2, 0, 0, 0]`, `[2, 1, 0, 0]`. Every coordinate with a nonzero value is active, every inactive one is zero, and after `dropzeros!` the working set equals the support. Which zeros stay active is *not* asserted. (Defect 5 of the contract is allowed by FW-8, so this test passes today) | `test/iterate_writes.jl` |
| **FT11 strong zeros, SparseIterate** | FN-1, FN-2, FN-4 | `s = [1, 0, 0]` stored at 1; `t` with NaN at 2; `A = [NaN 1 0; NaN 1 0]`; `u` with `u[2] = 1` and an explicit zero at 1; `Z = SparseIterate(2, 2)` with `Z[2, 2] = 1` and an explicit zero at `(1, 1)`; `S3 = Symmetric([NaN 0 0; 0 1 0; 0 0 1])`; `X1` with one nonzero at `(2, 2)` and `X2` nonzero in rows and columns 2:3; matrix coefficients `64 × 2` (stored-entry walk) and `3 × 2` (column walk) with an explicit zero against a NaN column | Reference: an explicit loop over the coordinates where the iterate's coefficient is nonzero. `dot(s, t) == dot(t, s) == 0.0`; `A * u == [1.0, 1.0]` before and after `dropzeros!(u)`, also through `mul!(out, A, u, 2, 0)` (`[2, 2]`), `transpose(permutedims(A)) * u` and `dot(A[1, :], u) == 1.0`; `S3 * u == [0, 1, 0]`; `dot([NaN, 1], Z, [1, 1]) == 1.0` before and after `dropzeros!`; `(S3 * X * I₃)[1, :] == [0, 0, 0]` for `X1` (sparse path) and `X2` (dense fallback); both matrix walks give the loop's result. A non-finite value *in* the iterate meeting a dense zero propagates: with `w[1] = Inf`, `dot([0, 1, 0], w)` is NaN. *Fails today* | `test/sparse_iterate.jl`, `test/nonfinite_inputs.jl` |
| **FT12 strong zeros, AtomIterate and owner** | FN-1, FN-2, FN-4 | `a = AtomIterate(4, [1:2, 3:4])` with `a[3] = 1` and group 1 active; `v = [NaN, 0, 1, 0]`; an owner with `X[2, 2] = 1` and an explicit zero at `(3, 1)`; `M` with NaN at `(3, 1)` and `(1, 3)` | `dot(v, a) == dot(a, v) == 1.0` before and after `dropzeros!`; `[NaN 0 1 0] * a == [1.0]`; `mul!(y, matrixview(X), [Inf, 1, 0]) == [0, 1, 0]` before and after `dropzeros!`; `dot(M, X) == M[2, 2]`; `dot(a3, X, b3)` with a NaN at a zero coordinate equals the loop. *Fails today* (NaN, `[0, 1, NaN]`) | `test/atom_iterate.jl`, `test/symmetric_mul.jl` |
| **FT13 the fallback boundary** | FN-3 | `s` with `s[2] = 1`; `A = [NaN 1; NaN 1]`; the `AtomIterate` of FT12 | `sparse(A) * s`, `Diagonal([NaN, 1]) * s`, `dot(sparsevec([1, 2], [NaN, 1]), s)`, `dot(view([NaN, 1, 0], [1, 2]), s)`, `sum(s .* [NaN, 1])` equal (`isequal`) the same calls on `Vector(s)`: `[NaN, NaN]`, `[NaN, 1.0]`, NaN, NaN, NaN; likewise `sparse([NaN 0 1 0]) * a` and `dot(sparsevec([1, 3], [NaN, 1], 4), a)` | `test/nonfinite_inputs.jl` |
| **FT14 a data-loss value** | FN-5 | `f = LeastSquaresLoss([1, 1], [NaN 1; NaN 1])`; `s` with `s[2] = 1` and an explicit zero at 1; `fa = LeastSquaresLoss([1], [NaN 0 1 0])` at the `AtomIterate` of FT12 | `f(s) == 0.0` before and after `dropzeros!`; `isnan(f([0, 1]))`; `fa(a) == 0.0`. Reference: the residual `y − Σ_{k: s_k ≠ 0} X[:, k] s_k`, a loop in the test. *Fails today* (`f(s)` is NaN with the stored zero) | `test/nonfinite_inputs.jl` |
| **FT15 coordinate oracles** | FA-1–FA-6, FL-4, FC-1 | `x = AtomIterate([1, 2, 0, 0], [1:2, 3:4])`; `fbad = GroupNormL2(1, [[1, 3], [2, 4]])`; `g = [3, 4, 0, 0]`; a `SparseIterate` with `x[1] = 0.5` | Return types: a `Float64` for `SparseIterate`, `atom(x, k)` for `AtomIterate`. `prox_coordinate!(x, fbad, 1, 1)`, `prox_coordinate!(x, fbad, 1, [9, 9], 1)` and `kkt_violation(fbad, g, x, 1)` throw `DimensionMismatch`; `k = 3` throws `BoundsError`; `u = [1]` throws `DimensionMismatch`; a complex `u` and a non-separable non-group `f` throw `ArgumentError`; after each, `Vector(x)` and the working set are unchanged. A group step to zero on an absent group leaves it absent; a scalar step to zero on an active coordinate keeps it active. `prox_coordinate!(y, NormL1(0.5), 1, atom(y, 2), 1)` with `atom(y, 2) == [3, 4]` gives `[2.5, 3.5]` (soft threshold by hand). Screens: `kkt_violation(GroupNormL2(1, [1:2, 3:4]), g, x, 1) ≈ 5 − √2` (weight √2 by hand); `kkt_violation(NormL1(1), g, x, 1) == 3.0` (largest of `|3|, |4|` minus 1); for a `SparseIterate`, `kkt_violation(NormL1(1), [3, −0.5], x, k)` is `2.0` and `−0.5` | `test/block_step.jl`, `test/interface.jl` |
| **FT16 block trials** | FA-7, FW-1 | `f = GroupNormL2(1, [1:2, 3:4])`, `x = AtomIterate(4, f)`; also `GroupNormLinf`, `SparseGroupNormL2`, `GroupMCP` and `NormL1`, groups of sizes 1, 8, 64, contiguous and not | `prox_coordinate!(x, f, 1, [3, 4], 1) ≈ (1 − √2/5)·[3, 4]` (by hand, `rtol = 4eps`); for each penalty the five-argument result `==` the result of staging `u` and calling the four-argument form (this package's own test may stage through the private field); with `fbad` the five-argument call throws and `x` is unchanged and consistent | `test/block_step.jl`, `test/block_step_failures.jl` |
| **FT17 layout** | FL-2, FL-4, FL-5, FL-6, FL-7, FX-6 | `gv = [[1, 2], [3, 4]]`, `a = AtomIterate(4, gv)`; `fm = GroupNormL2(1, [[1, 3], [2, 4]])`, `am = AtomIterate([1, 2, 3, 4], [1:2, 3:4])`; the counting index type of `test/atom_iterate.jl:26–31` | After `gv[1][1] = 3`, `atomindices(a, 1) == [1, 2]`. `samegroups(a, GroupNormL2(1, [[1, 2], [3, 4]]))` is `true`; with `[[2, 1], [3, 4]]` `false` (ordered lists); `samegroups(a, copy(a))` `true`; `samegroups(a, AtomIterate(4, [[1, 3], [2, 4]]))` `false`. `fm(am) == fm(Vector(am))` and `prox!(ym, fm, am, 1) == prox(fm, Vector(am), 1)` (the dense calls are the reference). With counted indices: one full comparison when a distinct equal layout is first seen, none on the next call with the same object, one per call when two distinct equal layouts alternate. If FL-3 is adopted, `parentindices(atom(a, 1))[1][1] = 3` throws. *Fails today* (no `samegroups`, no `atomindices`) | `test/atom_iterate.jl` |
| **FT18 counts and capacity** | FC-9, FC-10, FC-11, FP-3 | `y = SparseIterate(5); y[1] = 1; activate!(y, 3)`; `m = SparseIterate(2, 2)` with `m[2, 1] = 3` and an explicit zero at 4; an `AtomIterate` with one nonzero group, one all-zero active group and one inactive group | `nnz(y) == nactive(y) == length(nonzeros(y)) == 2`; `count(!iszero, activevalues(y)) == 1`; `findnz(y) == ([1, 3], [1.0, 0.0])`; `nonzeros(y)[2] = 4` gives `y[3] == 4`; `findnz(m) == ([2, 2], [1, 2], [3.0, 0.0])`; `nnz(a)` and `activevalues(a)` throw `MethodError`; FC-10's entry and group counts equal `count(!iszero, Vector(a))` and the group count by hand. `Base.summarysize(SparseIterate(Float64, n))` within 1 KiB of `24n`; `AtomIterate(Float64, p, K)` with range groups within 1 KiB of `16p + 32K` | `test/iterate_queries.jl` |
| **FT19 aliasing** | FX-5 | `x = SparseIterate(2); x[2] = 1; x[1] = 2`; `a` an `AtomIterate` with one active group; `w = SparseIterate([1, 2, 0, 4])` | `copyto!(x, activevalues(x))` gives `[1, 2]` (the handle reads `[1, 2]` in slot order, by hand); `Base.mightalias` of `activevalues(x)`, `nonzeros(x)` and `activecoordinates(x)` with `x` is `true`, and of `activecoordinates(a)` with `a`; `copyto!(w, view(w, 4:-1:1))` gives `[4, 0, 2, 1]`; `w .= 2 .* w` doubles. *Fails today* (`[1, 1]`; #406) | `test/iterate_writes.jl` |
| **FT20 real-only writes** | FX-4, FN-8, FS-3 | both types with one active entry; `c = ComplexF64[1, 2im, 0, 0]` | `copyto!(x, c)`, `copy!(x, c)`, `x .= c`, `x .= 2im`, `fill!(x, 2im)` throw `ArgumentError` and leave `x` unchanged; `SparseIterate(ComplexF64, 2)` and `AtomIterate(ComplexF64, 4, 2)` throw `ArgumentError`; `view(x, 1:2) .= ComplexF64[1, 2im]` and `vec(xm) .= c` throw (any exception; the test does not require `x` unchanged, FN-8). *Fails today* for `copyto!`, `copy!` (`InexactError` after a write) and `fill!` (`InexactError`) | `test/real_iterates.jl` |
| **FT21 empty and malformed** | FL-1, FS-3 | the edge-case list of §4 | the outcomes listed there, with the exact messages | `test/atom_iterate.jl`, `test/iterate_queries.jl` |
| **FT22 equality and shape** | FS-1, FS-2, FS-5 | `a`, `b` `AtomIterate`s with entries `[1, 0, 0, 0]`, `b` with group 2 also active; `s`, `t` `SparseIterate`s likewise | `a == b`, `isequal(a, b)`, `hash(a) == hash(b)`, `a == [1, 0, 0, 0]`; the same for `s`, `t`; `s isa AbstractArray`, `a isa AbstractVector`; `matrixview(s)` throws `MethodError` | `test/iterate_queries.jl` |
| **FT23 allocation** | FP-1, FP-2, FP-5, FW-5 | warmed-up `Float32` and `Float64` fixtures of §6 | `@allocated` is 0 for `x[i]`, `x[i] = v`, a first activation, `isactive`, `activate!`, `activecoordinates(x)`, `activevalues(x)`, `coordinates(x)`, `atom(x, k)`, `atomindices(x, k)`, `dropzeros!`, `mul!(out, A, x)`, `dot(v, x)`, the five-argument `prox_coordinate!` with a strided `u`, and the loops of §3.4. Growth with `n`, `s` and `p` is checked in §6 | `test/allocations.jl` |
| **FT24 performance gates** | FP-6 | §6 | each FD case within numerics C3 of the 0.5.x baseline | `benchmark/` |
| **FT25 no private fields downstream** | FF-1, FF-2 | ProximalSolvers `src/` | a scan for `.nzval`, `.nzval2ind`, `.ind2nzval`, `.storage`, `.active`, `.nactive`, `.position`, `.group_of`, `.layout_cache` and `.groups` on iterate receivers finds none (an allowlist names the penalty-field reads) | ProximalSolvers `test/` (outside this repository) |
| **FT26 continuation and workspace start** | FW-2, FW-6 | `solve!` with `continuation > 0` from an `x` holding a nonzero and an explicit zero, S and A; a CD workspace with a dense start solved twice | the result equals that from a fresh zero iterate (`1e-10`, cyclic order); the working set is empty right after the start (through a test hook on the start helper); the second workspace solve starts empty. Fails after FD-2 without the §3.5 change | ProximalSolvers `test/` |
| **FT27 momentum through accessors** | FC-6, FC-8, FC-9, FF-2 | CD with momentum on a lasso (S) and group lassos (A, contiguous and strided groups), cyclic order | the objective after every pass equals, to relative `1e-12`, a run recorded at the audited commits; FT25 finds no field use | ProximalSolvers `test/` |
| **FT28 block trials without private writes** | FA-7, FF-2 | `_groupstep!` for `GroupNormLinf`, `SparseGroupNormL2` and `NormL1` on `LSModel` and `QuadModel` | iterates equal, to relative `1e-12`, a run recorded at the audited commits; a step whose `prox_coordinate!` throws leaves no inactive group with a nonzero entry | ProximalSolvers `test/` |

**Requirement coverage.** FS-1 FT01, FT22; FS-2 FT22; FS-3 FT20, FT21; FS-4
FT02; FS-5 FT22. FC-1 FT01, FT15; FC-2 FT01; FC-3 FT01; FC-4 FT02; FC-5 FT02;
FC-6 FT02, FT27; FC-7 FT03, FT04; FC-8 FT04, FT27; FC-9 FT18, FT27; FC-10 FT18;
FC-11 FT18. FW-1 FT03, FT16; FW-2 FT05, FT09, FT26; FW-3 FT05; FW-4 FT06; FW-5
FT04, FT23; FW-6 FT08, FT26; FW-7 FT05; FW-8 FT10. FX-1 FT06; FX-2 FT05, FT07;
FX-3 FT07; FX-4 FT20; FX-5 FT19; FX-6 FT06, FT17. FL-1 FT21; FL-2 FT17; FL-4
FT15, FT17; FL-5 FT17; FL-6 FT17; FL-7 FT17. FN-1, FN-2, FN-4 FT11, FT12; FN-3
FT13; FN-5 FT14; FN-6 FT09; FN-7 FT04; FN-8 FT20. FA-1–FA-6 FT15; FA-7 FT16,
FT28. FP-1 FT23; FP-2 FT02, FT23; FP-3 FT18; FP-5 FT23; FP-6 FT24. FF-1 FT25;
FF-2 FT25, FT27, FT28.
**Non-promises, without a test of their own:** FL-3 and FP-4 (recommendations;
FT17 gains a case if FL-3 is adopted); the "unspecified" part of FW-8 (FT10
deliberately does not assert which zeros remain).

**Tests that fail at the audited commit, by defect** (contract Part 4): the
writable handles, FT03; defect 1, FT11; defect 2, FT11, FT12, FT14; defect 3,
FT11; defect 4, FT05; defect 5, none (allowed by FW-8; FT10 passes); defect 6,
FT05, FT09; defect 7, FT19; defect 8, FT20.

# 6. Benchmark protocol (a future measurement, not a result)

No timing was run for this plan, apart from the one exploratory `coordinates`
sweep the contract quotes.

**Where.** Two new groups, `sparse` and `atom`, in `benchmark/cases.jl`,
added to `CASE_GROUPS` and to the builders. They follow the `owner` group of
#401, which is merged (`ea0e48e7`, after the audited commit): only calls that
exist in both 0.5.x and 0.6, with a new accessor reached through
`isdefined(ProximalBase, :atomindices)` and the 0.5.x spelling otherwise;
every case checked against an independent reference before timing; a case
that changes state uses `prepare` outside the timed region and `evals = 1`;
`zero_alloc` set where FT23 promises zero bytes; `work` = operations per
call; `harness_hash` in the metadata. The baseline goes to
`benchmark/baselines/iterate-family-0.5.md` with #401's identity table. Were
#401 reverted, the same conventions apply and the groups add the
`harness_hash` field themselves. The owner's strong-zero case (FD-9) is added
to the `owner` group.

**Environment to record.** Julia version, `JULIA_NUM_THREADS=1`, BLAS library
and one BLAS thread, CPU model, OS, ProximalBase commit, `source_hash`,
`harness_hash`, `--check-bounds` setting, suite, groups and repetitions.

**Warm-up and timing.** The harness's own: compile and check outside timing,
the minimum and median of the samples, three or more repetitions in fresh
processes, alternating revisions.

**First activation.** Every timed call activates a batch of 256 *distinct*
coordinates (S) or groups (A) that are absent before the call; the restore
runs outside the timed region and asserts that none is active (as #401's
owner case).

**Fixtures.** `Float32` and `Float64`. `SparseIterate` vectors with
`n ∈ {10⁴, 10⁶}` and `s ∈ {10², 10³, 10⁴}` (`s ≤ n/100`), values bounded away
from zero; matrices `1000 × 64` and `256 × 1000` at densities 0.5%, 3% and 20%
(both sides of the 1/32 switch). `AtomIterate` with `p = 8·10⁴` (`8·10⁵`
extended), group sizes 1, 8 and 64, contiguous (ranges) and non-contiguous
(strided, collected, as in the multi-response model), 1% and 10% of groups
active. Dense operands with 256 and 1024 rows.

**Cases.** Each FD on a hot path has one, marked in bold:

- construction, with a retained-size assertion (FT18's formulas);
- scalar reads (active, absent) and writes to active entries;
- first activation;
- a full sweep `for i in 1:ncoordinates(x)` and an active sweep with a
  trivial update;
- **FD-7:** an active sweep through the read-only `activecoordinates`, and the
  same through `activevalues` (S);
- **FD-5:** a read-write sweep through `coordinates(x)` of a matrix against
  the same sweep through `x`;
- `atom` access: `norm` and `dot` of `atom(x, k)` (BLAS for contiguous
  groups);
- **FD-6:** iteration over `atomindices(x, k)` against `x.groups[k]`, and
  `view(g, atomindices(x, k))` with a vector group;
- **FD-9:** `dot(v, x)`, `mul!(out, A, x)` (strided, transposed, `Symmetric`;
  both matrix walks), `A * X * B` (both paths) and the `AtomIterate` `dot`,
  with finite data: the cost of the strong-zero branch;
- **FD-12:** the five-argument block trial against staging plus the
  four-argument call, for `GroupNormL2`, `GroupNormLinf`, `SparseGroupNormL2`
  and `NormL1`, group sizes 1, 8, 64;
- **FD-8:** the momentum value loops of §3.4 against the private-field
  versions;
- **FD-14:** `copyto!(x, v)` from an unrelated `v` (the alias check only),
  `x .= f.(x)`, and `copyto!(x, activevalues(x))` (the `unalias` copy);
- **FD-2:** `fill!(x, 0)` (a value write) against `dropzeros!(fill!(x, 0))`,
  and `x .= y` (a loop) against `copyto!(x, y)` (the state copy);
- **FD-11:** the existing `group_layouts` cases (`benchmark/cases.jl:419–486`)
  plus `samegroups` with shared and distinct layout objects;
- copy and import: `copy`, same-kind `copyto!`, `copyto!` from a dense array;
- compaction: `dropzeros!` with 10% and 50% of the active coordinates zero.

**Targets** (future, from FP):

| case | as `n` or `p` grows, `s` fixed | as `s` grows | allocation after warm-up |
|:--|:--|:--|:--|
| read, write, `isactive`, first activation | flat | flat | 0 B |
| active sweep, handles, `atomindices` loops | flat | linear | 0 B |
| full sweep, `fill!`, `x .= y`, imports | linear | — | 0 B |
| `dot(v, x)` | flat | linear | 0 B |
| `mul!(out, A, x)` | linear in the rows of `A` | linear | 0 B |
| `dropzeros!` | flat | linear | 0 B |
| construction | linear | — | the reserved arrays (FP-3) |
| `copyto!(x, activevalues(x))` | — | linear | one copy of the source |

**Baseline.** Record the baseline at a pinned 0.5.x commit **before any FD
is implemented, additive ones included**: master with the two groups added
and no FD merged. Each FD pull request then reports its cases against that
baseline in the same session, with `benchmark/compare.py`.

**Review thresholds** (numerics C3). A repeatable slowdown above 5% on a
representative case is investigated and reported; a cost above 10% paid only
by narrow types needs the maintainer's approval; machinery that only narrow
types need leaves `Float64` timings within the harness's noise.

**What a benchmark can and cannot show.** A flat curve over a factor of 100 in
`n` is a regression alarm, not a proof of `O(1)`; the proof is the code (a
direct index table and no growth), a review item. Single runs are
exploratory. A benchmark does not show correctness: every case is checked
first, and FT01–FT23 carry the guarantees.

# 7. Migration and dependency plan

Implementation issues are opened only after the maintainer signs off the
decisions. #405 and #406 are already filed as 0.5.x bug reports.

| item | kind | depends on | integrates with | follow-on issue |
|:--|:--|:--|:--|:--|
| 1. `sparse` and `atom` benchmark groups, 0.5.x baseline | harness only | — | #401's conventions; #391 Phase 0 | "Benchmark groups for SparseIterate and AtomIterate, with the 0.5.x baseline" |
| 2. FD-7 read-only `activecoordinates` | 0.5.x bug fix | 1 | ACT-3; the owner in #391 Phase 1a | #405 (filed) |
| 3. FD-14 `dataids` of the handles | 0.5.x bug fix | 1 | #367; #366 G12 | #406 (filed) |
| 4. FD-13 real check in `copyto!` and `copy!` | 0.5.x bug fix | — | #364; #366 G12 (O6.1) | on #364 |
| 5. FN-6 `AtomIterate` `fill!` converts first | 0.5.x bug fix | — | #366 G1 (conversions) | "AtomIterate fill! decides activation before conversion" |
| 6. FD-6 `atomindices` and `ProximalBase.samegroups` | 0.5.x-safe addition | 1 | ProximalSolvers migration | "AtomIterate: atomindices and samegroups" |
| 7. FD-4, FD-10, FD-12: state the rules 0.5.x already meets | 0.5.x-safe addition (documentation) | — | `docs/src/iterates.md` | "Document the iterate-family rules that 0.5.x satisfies" |
| 8. ProximalSolvers: §3.1–§3.4 and §3.8 | downstream, runs on 0.5.x | 6 | #391 Phase 2 | "ProximalSolvers: replace private iterate fields by accessors" |
| 9. ProximalSolvers: §3.5 | downstream, runs on 0.5.x | — | must land before 10 | "ProximalSolvers: clear working sets with dropzeros!" |
| 10. FD-2 one working-set rule | breaking 0.6 | 1, 9 | #391 Phase 1 integration branch | "0.6: one working-set rule for writes" |
| 11. FD-3 owner `copyto!` is a value write | 0.5.x behavior kept; owner 0.6 | sign-off of the CPY-4 amendment | #391 Phase 1a | part of #391 Phase 1a |
| 12. FD-5 `coordinates(x) = vec(x)` | breaking 0.6 | 1 | #391 Phase 1 | "0.6: coordinates of a matrix SparseIterate is vec(x)" |
| 13. FD-9 strong zeros, with the M1 amendment | breaking 0.6 | sign-off of the amendment; 1 | #366 G2; #391 Phase 1c (PERF-6) | "0.6: strong zeros in products of iterates" |
| 14. FL-3 read-only vector groups | breaking 0.6 (recommendation) | 1 and its measurement | — | "AtomIterate: store vector groups read-only" |
| 15. FF-1 and the documentation rewrite | 0.6 documentation | 10, 12, 13 | #391 Phase 1d | part of #391 Phase 1d |

**Order.** Items 1–9 can run in parallel with #391 Phase 0: they are
additive, bug fixes or downstream changes that work on 0.5.x, and item 1 comes
first so that every later item has a baseline. Items 10–15 change 0.6 behavior
and stack with #391 Phase 1 on its integration branch; item 9 must be released
in ProximalSolvers before item 10, or the continuation start changes silently.
