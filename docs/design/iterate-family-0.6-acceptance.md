# The iterate family in 0.6: compatibility and acceptance

> **For the 0.6 design signed off on 2026-11-01 (FD-16 open), not a description
> of an installed release.** This plan explains how callers move to the
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
| `src/cd/glm.jl:552`, `:572`, `:578`, `:735–754`; `src/cd/generic_newton.jl:122–126`, `:165`; `src/cd/lla.jl:211–224`, `:247` | state copies into an iterate (`copyto!(x, …)`, `copyto!(ws.x0, x)`) around inner coordinate-descent runs | every copy runs between two `_cd!` runs, and each run starts new momentum and Anderson histories (`src/cd/solver.jl:464–475`, `:452`, `:478–489`, `:608`, `:642`), so no slot-keyed history is live across a state copy; FC-7's new lifetime case needs no change here (§3.4) |
| `src/solvers/active_accelerated_proximal_gradient_descent.jl:125–304` | `dropzeros!` (`:150`); the gradient at `x` (`:151`); a broadcast read into a dense buffer (`:159`); `activate!` (`:191`); scalar reads (`:199`) and writes to active entries (`:203–206`); the whole-layout check (`:264–266`); indexed reads (`:272`, `:283`); a handle read (`:277`); screening of inactive coordinates (`:294–302`) | reads only; no write through a handle; no reliance on `fill!` clearing. FN-5 applies at `:151` only with non-finite data |
| `src/types.jl:210` | `OptimizationResults` constrains the minimizer to `AbstractArray` or blocks | affects only the owner (FD-15c) |
| `src/cd/path.jl:8–10`, `:261`, `:342` | `_zero_like` rebuilds an `AtomIterate` from `x.groups` (`:10`); the continuation starts with `fill!(x, zero(T))` (`:261`) and relies on it clearing the working set; an indexed read (`:342`) | `:261` changes silently under FD-2. Recipes §3.2, §3.5 |
| `src/cd/models.jl:691` (and `:429`, `:473–478`) | whole-layout equality of iterate and penalty; an indexed read; penalties rebuilt from `g.groups` at each path level | `samegroups` (§3.2); `:473–478` read a penalty field |
| `src/workspaces.jl:153` (and `:93–95`, `:140–149`, `:243–246`) | whole-layout equality of two iterates; `reinit!` copies a same-type start with `_copyto!` (a state copy, unchanged); a CD workspace loads a dense start with `fill!(xs, 0)` and a scatter | `:243` changes silently under FD-2 (§3.5) |
| `src/cd/gap.jl:200`, `:719` | `_coordentries(x::AtomIterate, k) = x.groups[k]`; a loop over `x.groups[k]` with public writes; the other `.groups` reads in `gap.jl` are penalty fields | two indexed reads |
| `src/cd/leastsquares.jl:192` (and `:558`, `:586`, `:598`) | `Iterators.flatten(x.groups[k] for k in activecoordinates(x))`; `m.X[:, G]` with a group as index | indexed reads; a wrapped vector group becomes the index at `:586` (benchmark, FP-6) |
| `src/solvers/diagonal_metric.jl:106`, `:114` | `_copyto!(x, z)` in the diagonal-metric proximal step | `x` can be a `SparseIterate` or `AtomIterate` (`_diagonal_metric` takes any floating-point array, `:12`; the owner is rejected at `:23–24`). `z` is `state.xhat`, of the type of `x` (`src/solvers/proximal_gradient_descent.jl:67–77`, `:121`, `:156–157`), so the copy is a same-kind state copy, unchanged by FD-2 |
| `src/solve.jl:4–6`, `src/blocks.jl:23–40` | `_drop!` is `dropzeros!`; `_copy!` is `copy!` (a same-kind state copy); `_zero` is `zero` (empty) | unchanged |
| `src/cd/quic.jl:77`, `:91`, `:100` | the Newton loop builds the step `D = SymmetricSparseIterate(T, p)` (`:77`), clears it with `fill!(D, zero(T))` at each iteration (`:91`), and stops when the step is empty (`nactive(D) == 0 && break`, `:100`) | relies on the clear; fails loudly in 0.6, where the owner has no `fill!`. Recipe §3.5 |

**Mentioned but not traced** (they touch iterates; I read only the lines
cited): ProximalSolvers `src/cd/groupcurvature.jl:124`, `src/cd/quadratic.jl:148,
175`, `src/cd/sparsegroup.jl:108, 118, 125, 181` (indexed reads),
`src/cd/drivers.jl:95, 115, 150` (`coordinates` by ID), `src/cd/quic.jl:138,
203, 214` (more `fill!` on an owner, a #391 Phase 2 item), `src/cd/lla.jl`,
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
with an explicit zero; the worked example of FD-2 and the owner example of FD-3;
copies, imports and length checks; conversion and activation in `Float32`;
complex sources; non-finite writes; empty and malformed layouts; layout sharing,
the one-slot cache and whole-array fallbacks; coordinate-oracle validation,
returns and aliasing; the block trial against private staging; products with a
NaN or `Inf` against stored, unstored and dense zeros, the generic fallbacks and
SparseArrays; a data-loss value; aliasing hooks; retained memory; the
multi-response mapping; the examples of Part 1. Further scripts gave the slot
order of the replace spellings, scaling by a number, `A * X * B` against its
term-wise sum (including the association example), products of views and
transposed products, `coordinates` against `vec`, the matrix loss at the
`vec(B)` iterate, a trial `u` that throws while it is read, and the ambiguity
of a checked `copyto!`. Later scripts gave the screens of the built-in group
penalties, a failed owner import against the clear-then-import recipe, the
array shorthand with an overlapping source, `vec` and `reshape` of vectors, a
state copy that reorders slots at equal membership, and the three-factor mixed
fixture. The last scripts gave failed state copies on both branches of the
copy kernel, from empty and non-empty destinations, and for `AtomIterate`; the
shorthand with a same-kind source and with a fallible conversion; the staged
replace with each of those sources, an overlapping one and one of another
length; the ProximalSolvers workspace scatter against both spellings; finite
overflow in a two-vector `dot`; and the index types of stored groups. A script on Julia 1.10.12 and 1.13.3 traced which Base
entry points reach `fill!` or `copyto!`. Not run: the package test suite, the
benchmark suite and any downstream test.

**Changes to the design after its independent review.** FD-5 first
recommended `coordinates(x) = vec(x)`; it now recommends a coordinate-vector
view for both shapes, because `vec` makes products and copies depend on the
shape. A term-by-term rule for `A * X * B`, triggered by an output scan, was
added (superseded by the open FD-16 after the maintainer's review). The replace spelling became `copyto!(dropzeros!(fill!(x, 0)), A)`,
which keeps 0.5.x's slot order (FW-6). Scaling by a number was
classified (FN-9), views of an iterate were placed under FN-3, FD-15a gained a
record, and FD-13 gained the ambiguity cost and the O6 and S4 text.

**Changes after the maintainer's review of the draft.** FA-6 now gives each
group penalty's own screen (dual norm and default weight), with a table. FD-3
is withdrawn: the owner keeps #388's CPY-4, a checked replace, because
a clear-then-import recipe loses the state when the import's checks fail; the
array iterates' replace shorthand is limited to an independent, validated
source, with a safe migration for any other (both revised after the second
review, below). Three-factor products moved out
of FD-9 into a separate, open decision, FD-16, and the sign-offs are grouped
(contract §3.1). The rule for views is drawn by the returned object, since
`vec(x)` of a vector is `x`. A whole-state replacement now ends a handle's
lifetime (FC-7, FD-7).

**Changes after the maintainer's second review.** A same-kind state copy
that throws must now leave the destination unchanged (FW-4, FX-1); today it
can corrupt a `SparseIterate` (#416). The cheap replace shorthand is limited
to sources with value-write dispatch, independent of `x` and with infallible
reads and conversions; every other source is staged by value in `similar(x)`
and then state-copied (FW-6), which replaces the earlier unaliasing
migration. FN-4 and the M1 amendment state the strong-zero guarantee for
zeros and non-finite entries only; summation order and finite overflow follow
A1 (FD-9). `atomindices` returns the stored index type (FC-6, FD-6).

**Changes before structural sign-off.** A state copy converts only the
source's active values, so a stale value in unused capacity cannot make a
valid copy throw (FW-4, FX-1; FT06 gains the copy that must succeed). The
failure clause and that domain also cover the owner's CPY-3, a consistent
extension of #388 that settles the open point on the owner's failure
guarantee with option (a); #416 fixes all three types. Both replace recipes
check the length and the real element type before clearing or staging, and
the general recipe traverses source and destination with separate indices
(FW-6). PERF-6's requirement is no longer said to follow from FD-9: it is
pending with FD-9, with two options. FD-3's guarantee for a failed matrix
import is limited to CPY-4's listed pre-write checks; a conversion failure
follows owner §2.10.

**Decisions recorded (2026-11-01).** The maintainer signed off the structural
group (FD-2 to FD-8, FD-10 to FD-15a, the table of contract §3.2) and FD-9
with its numerics amendment, choosing option (i) for PERF-6's requirement.
FD-16 is open in #419.

# 2. Reconciliation with the specialized issues

| issue | fixed on master | decided here | remains for that issue |
|:--|:--|:--|:--|
| #137 API additions and 0.6 decisions | `findnz` and `nonzeros` for `SparseIterate` (`src/sparse_iterate.jl:448`, `:457–471`) | the iterate-contract item for S and A: FD-6, FD-10, FD-13; the owner by #388 | the non-iterate items (`iszeropreserving`, tuple calculus, `public` declarations, the validation hook, `Testing` defaults, non-public stdlib names, deprecations) |
| #189 writes through `atom` (closed) | `atom(x, k)` is `view(x, G)` and activates (`src/atom_iterate.jl:163–166`); regression tests `test/atom_iterate.jl:244–269`; the block-trial form exists (`src/interface.jl:469–478`) | kept (FC-5); the five-argument form is the supported trial (FA-7) | nothing; ProximalSolvers' private staging, which #189's decision found, migrates (§3.3) |
| #192 what `nnz` counts | — | FD-10: SparseArrays' meaning on `SparseIterate`, no `nnz` on `AtomIterate`, four named counts (FC-10, FC-11) | the owner part (ACT-2, #391); other packages' `1:nnz` loops over fields move to `activevalues` |
| #341 value oracles reading every entry | — | FN covers products, not these values. Numerics decision 47 rejects non-finite `LinearFunction` coefficients, so FN-1 does not change `LinearFunction`'s value | all four oracles, under #366 G2 |
| #364 views bypass the real check | — | FD-13: the iterate's own bulk writes check first (FX-4); wrappers follow Base (FN-8), with no overrides, and FD-13 proposes O6 and S4 text that says so. The check needs `copyto!(x::SparseIterate, y::AbstractArray)` and the methods that resolve its ambiguities with SparseArrays. The complex `copyto!` partial write and the `vec(xm)` broadcast are recorded on #364 | the `copyto!` check (a 0.5.x bug fix), its ambiguity-resolving methods, and the docstrings |
| #366 numerical contract | the Phase 0 conformance suite (#387, merged as `aea50cd9`, after the audited commit), with M1 storage rows in `test/conformance/failures_storage.jl` | FD-9's amendment of M1, E2, E3, C1 and S4, and FD-13's O6 and S4 text, word for word; FD-16's proposed M1 paragraph for its option (a), open; FX-4 and FX-5 follow O6.1 and O7 | sign-off of the amendments, FD-16's separately; the M1-amendment rows join the conformance suite as rows marked `broken = (group = :G2, baseline = …)`, the suite's mechanism for known failures, until FD-9 lands; G2 implements FN-1 and FN-9; G12 covers FX-4 and FX-5 |
| #367 aliasing audit | — | FX-5: imports from a handle follow Base's `unalias`; FA-5 holds today | the oracle-by-oracle audit |
| #368 owner views unchecked | — | nothing new; for S and A, writes through wrappers follow FN-8 | expected to close with #391 (MAT-2) |
| #391 owner implementation | Phase 0: `matrixview`, `activevalues`, owner `dataids` (#392); the `owner` benchmark group (#401) | FD-3 keeps CPY-4 as written; FD-9 amends decision 13, PERF-6's default and T13, and PERF-6's requirement (option (i)); FD-7 shares ACT-3's handle and adds the whole-state replacement to the lifetime of ACT-3 and ACT-4 (a consistent extension); FD-16 decides the class of the owner's three-factor product (open); FD-4 and FD-5 are consistent with ORC-3, ORC-6, CRD-1 | Phase 1 implements CPY-4 and the owner parts of FD-7 and FD-9 (and FD-16 once decided); Phase 2 the downstream recipes |
| #342 groups on an owner | — | out of scope: FL applies to `AtomIterate` | as is |
| #355 per-group `Regularize` steps on `AtomIterate` | — | its destination follows FW-8 | as is |
| #405 writable `activecoordinates`, `AtomIterate` `dropzeros!` hang | — | fixed by FD-7 (FC-7) | filed; a 0.5.x bug fix |
| #406 `dataids` misses the handles | — | fixed by FD-14 (FX-5) | filed; a 0.5.x bug fix |
| #416 a failed state copy corrupts the destination | — | fixed by FW-4's clauses (FX-1), for S, A and the owner's CPY-3: only active values are converted, and the destination is unchanged on failure; recommended, a check before writing only when the conversion can fail | filed; a 0.5.x bug fix |

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

The history is keyed by slot, so it must start again after a whole-state
replacement of the iterate, which can reorder the slots at equal `nactive`
(FC-7). The audited solver already meets that: every state copy into an
iterate in `src/cd/glm.jl`, `src/cd/generic_newton.jl` and `src/cd/lla.jl`
runs between two `_cd!` runs, and each run starts a new history
(`_momentum_start!`, `src/cd/solver.jl:464–475`; the first pass is full and
calls `_momentum_reset!`, `:452`). Nothing changes there.

## 3.5 `fill!(x, 0)` used as a clear

| site (ProximalSolvers) | 0.5.x | 0.6 (future) |
|:--|:--|:--|
| `src/cd/path.jl:261` | `fill!(x, zero(T))` | `dropzeros!(fill!(x, zero(T)))` for S and A; `fill!(coordinates(x), zero(T)); dropzeros!(x)` for an owner |
| `src/workspaces.jl:243–246` | `fill!(xs, zero(eltype(xs)))`, then `xs[i] = ws.x[i]` for each nonzero | `copyto!(dropzeros!(fill!(xs, zero(eltype(xs)))), ws.x)` |
| `src/cd/quic.jl:91` (owner `D`, the Newton step) | `fill!(D, zero(T))` | `fill!(coordinates(D), zero(T)); dropzeros!(D)` (#391 Phase 2) |
| `src/cd/quic.jl:138, 203, 214` (owner) | `fill!(X, zero(T))` | `fill!(coordinates(X), zero(T)); dropzeros!(X)` (#391 Phase 2) |

Without the change, the first two start from the caller's previous working
set and raise no error; the owner sites fail loudly, since the 0.6 owner has
no `fill!`. The owner needs two statements, because `fill!` returns the
coordinate view and `dropzeros!` of that view is a `MethodError` (ran).

The workspace site. At 0.5.x, `fill!(xs, 0)` empties the working set and the
loop then writes the nonzeros of `ws.x` in linear order, so the working set is
the support of `ws.x` in increasing ID order for a `SparseIterate`, and for an
`AtomIterate` the groups in the order in which the loop first meets a nonzero
entry of each. The branch runs only when `xs !== ws.x`, that is, when the
caller passed a `Vector{T}` with `T <: AbstractFloat`: `_workspace_iterate`
(ProximalSolvers `src/workspaces.jl:93–95`) returns an iterate unchanged and
otherwise builds `xs` from a copy of the vector (`SparseIterate(x)`,
`src/sparse_iterate.jl:513–521`; `AtomIterate(x, groups)`,
`src/atom_iterate.jl:56–59`), with the same element type. So `ws.x` is never a
same-kind iterate, shares no memory with `xs`, has its length (each `solve!`
checks the shape, `_workspace_shape`, ProximalSolvers `src/workspaces.jl:113–114`,
`:123`), has a real element type, and converts without failure: FW-6's two
checks hold by construction and may be omitted here, the cheap shorthand
applies, and the staged replace is not needed. Ran: from a working set `[1, 4]` and `ws.x = [0, 3, 0, -0.0, NaN, 1]`,
the 0.5.x loop, the shorthand and the staged replace all give `[2, 5, 6]`
with equal values; for groups `[[2, 4], [1, 3]]` and `ws.x = [1, 0, 0, 2]`,
all three give `[2, 1]`, where the constructor gives `[1, 2]`. Keeping the
loop after `dropzeros!(fill!(xs, zero(eltype(xs))))` is equivalent. A cyclic
coordinate descent from either follows the 0.5.x trajectory.

## 3.6 A replacing import

```julia
fill!(x, 0); copyto!(x, A)                    # 0.5.x

# 0.6 (FW-6): both recipes check first, before anything is cleared or staged
length(A) == length(x) || throw(DimensionMismatch("A must have the length of x"))
eltype(A) <: Real || throw(ArgumentError("A must have a real element type"))

copyto!(dropzeros!(fill!(x, 0)), A)           # cheap: A is not a same-kind iterate, shares
                                              # no memory with x, and its reads and
                                              # conversions cannot fail

t = similar(x)                                # general: stage by value, with separate indices ...
for (i, j) in zip(eachindex(t), eachindex(A))
    t[i] = A[j]
end
copyto!(x, t)                                 # ... then state-copy the completed temporary

copyto!(X, M)                                 # the owner: CPY-4's checked replace (FD-3)
```

**Failure guarantee.** Neither recipe touches `x` when a check fails. The
staged replace also leaves `x` unchanged if staging throws, because `x` is
written only by the final state copy, whose element types agree: a failed
read or conversion throws inside the loop (ran: into
`xj = SparseIterate(Int, 2); xj[2] = 5` from `[1.0, 1.5]`, it throws
`InexactError` and `xj` stays `[0, 5]` with working set `[2]`). Its cost is one
`similar(x)`, `O(n)` memory and time. The cheap shorthand has no failure
guarantee past the checks: it clears first, so a failure after the clear
loses the old state (ran: the same `xi` and source leave `[1, 0]` with working
set `[1]`).

**Why the checks and the separate indices.** Without the element-type check
a complex-typed source with zero imaginary parts passes: the staged loop and
the cheap recipe both turn `ComplexF64[1, 2, 3, 4]` into `[1.0, 2.0, 3.0, 4.0]`
today (ran), where FX-4's checked bulk write rejects it. A joint
`eachindex(t, A)` asks for common indices, and on the `IndexCartesian` path
equal axes: for `x = SparseIterate(4)` and
`A = view(reshape(1:4, 2, 2), [2, 1], :)` it throws `DimensionMismatch`,
while the separate traversal gives `[2.0, 1.0, 4.0, 3.0]` with working set
`[1, 2, 3, 4]` (ran). `eachindex(A)` of an `IndexCartesian` source is
column-major, which is linear order. The length check stays explicit,
because `zip` truncates.

**Why the cheap shorthand needs its conditions.** A same-kind source
dispatches to the state copy and brings its explicit zeros: with
`y = SparseIterate(4); y[4] = 2.; activate!(y, 1)` the shorthand gives `[4, 1]`
and the staged replace `[4]` (ran). An overlapping source is read after the
clear: `copyto!(dropzeros!(fill!(x, 0)), view(x, 2:-1:1))` of
`x = SparseIterate([1., 2.])` gives `[0.0, 0.0]`, while the staged replace
gives `[2.0, 1.0]` (ran). The shorter `dropzeros!(copyto!(x, A))` gives the
same set in another slot order (ran: `[3, 2, 9]` against `[2, 3, 9]`), which
changes the trajectory of a cyclic active pass.

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
| `coordinates` | a coordinate-vector view, indexed by ID; writes are VW; products IEEE; aliased imports copy the source; complex input as a plain vector, `InexactError` possibly after a partial write (FC-3; today `x` itself, with `ArgumentError` before writing) | the same view, linear IDs (today the matrix `x`) | `MethodError` (FC-4) | CRD-1 |
| `vec(x)`, `reshape(x, length(x))`, `view(x, :)` | `vec` and the full-length `reshape` return `x` itself, with `x`'s semantics (FN-1, FX-4); `view(x, :)` is a wrapper (FN-3, FN-8) | `vec(x)` is a Base `ReshapedArray`, a wrapper (FN-3, FN-8) | as the vector | not an array (OWN-1) |
| `atom`, `atomindices` | `MethodError` | `MethodError` | group ID; a writable view, a read-only index list (FC-5, FC-6; `atomindices` today missing); `BoundsError` | `MethodError` |
| `activecoordinates`, `activevalues` | read-only handle (FC-7; today writable, #405), valid until the next activation, compaction or whole-state replacement; `activevalues == nonzeros`, writable (FC-8) | same | read-only handle (today writable); `activevalues`: `MethodError` (FC-9) | ACT-3, ACT-4, with FD-7's whole-state replacement |
| `activate!` | ID; no-op if active; `BoundsError` (FW-7) | ID | group ID | ACT-6 |
| `dropzeros!` | removes zeros only; keeps NaN, `Inf` (FW-5, FN-7) | same | removes all-zero groups | ACT-7 |
| `fill!(x, c)` | VW; `c = 0` keeps the set (today clears); complex `c`: `ArgumentError` (today `InexactError`, nothing written) | same | VW; converts first (FN-6; today `1e-100` activates every group at `0f0`) | no `fill!` (OWN-3); `fill!(coordinates(X), c)` is a VW |
| `x .= c` | VW (today clears for `c = 0`); complex: `ArgumentError` | same | same | none (OWN-3); `θ .= c` (CRD-5) |
| `x .= v`, `x .= f.(x, v)`, `x .*= 0` | VW, the loop (FW-3) | same | same | `θ .= …` (CRD-5) |
| `x .= y`, `y` same-kind | VW (today SC) | same | same (today SC) | — |
| `copyto!`/`copy!` from a same-kind iterate | SC, element type may differ (FX-1); takes the source's slot order and ends old handles (FC-7); `copyto!(x, x)` no-op; only active values are converted, and a failed conversion leaves `x` unchanged (FW-4; today it can corrupt `x`, and a stale value in unused capacity can make a valid copy throw, #416) | same | same layout (FL-4): SC; unchanged on failure (today partly updated, #416) | CPY-3, with FW-4's two clauses (#416) |
| from another shape or layout | same length: VW (today `DimensionMismatch` for a matrix source); other length: `DimensionMismatch` (FX-3) | same | same length: VW (today too); other length: `DimensionMismatch` | other `p`: `DimensionMismatch` (CPY-3) |
| from a plain array | VW in linear order (FX-2); a longer source: `BoundsError`; `copy!` from another length: `MethodError` (`resize!`) | VW; a longer source: `BoundsError`; `copy!` with other axes: `ArgumentError` (Base) | as the vector | checked replace (CPY-4, FD-3): axes `DimensionMismatch`, asymmetry or complex `ArgumentError`, all before writing, `X` unchanged when one of these checks fails (a conversion failure follows owner §2.10); then the converted support in increasing ID order (today a value write) |
| from an aliased source | VW of a copy of the source (FX-5; today `copyto!(x, activevalues(x))` is wrong, #406) | same | same | `ArgumentError` (CPY-6) |
| from a complex source | `ArgumentError` before writing (FX-4; today `InexactError` after a partial write) | same | same | `ArgumentError` before writing |
| `copy` | SC | SC | SC; shares the layout | CPY-1 |
| `similar` | empty, same size; `similar(x, S)` | same | empty, shares the layout | CPY-2 |
| `nnz`, `nonzeros`, `findnz` | explicit zeros included (FC-11) | `findnz` gives `(I, J, V)` | `nnz`: `MethodError` | ACT-2 |
| `dot`, `mul!`, `*` | strong zeros (FN-1, FN-2; today storage- and order-dependent); fallbacks and other objects built from `x` IEEE (FN-3) | both walks of `mul!` and `dot(a, X, b)`: strong zeros; `*(A::Symmetric, X, B::Symmetric)`: open (FN-10, FD-16) | `mul!` skips zeros (today too); `dot`: strong zeros (today not); `transpose(A) * x` and `dot` of an `atom` view: IEEE (FN-3) | PERF-6's default and requirement as amended by FD-9 (requirement: option (i)); ORC-10; `A * X * B`: open (FD-16) |
| scaling `rmul!`, `lmul!`, `rdiv!`, `ldiv!` by a number | IEEE VW equal to `x .*= s` (FN-9; today `[Inf, NaN, 0, 0]` for `rmul!(x, Inf)` with an explicit zero) | same | IEEE VW (today too, through LinearAlgebra's generic method) | `rmul!(coordinates(X), s)`, an ordinary-vector operation (CRD-4) |
| `prox_coordinate!(x, f, k, γ)` | ID; returns the scalar; FA-3, FA-4 | ID | group ID; returns `atom(x, k)`; layout checked (FL-4) | ORC-5 |
| `prox_coordinate!(x, f, k, u, γ)` | scalar `u` | scalar `u` | `u` of the group's length, may alias `x` (FA-5); unchanged after a validation error (FA-4); a partial write is possible if reading `u` throws (FA-7, FT16) | ORC-6 |
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
| **FT01 coordinates and index spaces** | FC-1, FC-2, FC-3, FS-1, FX-5 | `x = SparseIterate(Float64, 2, 3)`; `y = SparseIterate(Float64, 2, 3)` with `y[1, 1] = 3`; `v = SparseIterate(4)` with `v[1] = 1`; `w = SparseIterate([1.0, 2.0, 3.0, 4.0])` | `c = coordinates(x)`: `c isa AbstractVector{Float64}`, `length(c) == 6 == ncoordinates(x)`, `axes(c, 1) == 1:6`, `c !== x`; `c[4] = 2.0` gives `x[2, 2] == 2.0` and `activecoordinates(x) == [4]` (4 = (2 − 1)·2 + 2 by hand); `activate!(x, 6)` makes `x[2, 3]` an explicit zero; for the vector, `cv = coordinates(v)` is not `v`, `cv[1] == 1`, and `cv[3] = 5` gives `v[3] == 5`; `Base.dataids(c) == Base.dataids(x)`; `@allocated coordinates(x) == 0` after warm-up. Products through the view are ordinary-vector products: `dot([1, NaN, 0, 0], cv)` and `dot([1, NaN, 0, 0, 0, 0], c)` are NaN (IEEE, FN-3), unlike `dot([1, NaN, 0, 0], v) == 1.0`; `copyto!(coordinates(x), coordinates(y))` is a value write (working set `[4, 6, 1]`, values those of `y`). Aliased imports copy the source first: `copyto!(w, coordinates(w))` leaves `w == [1, 2, 3, 4]`, and `copyto!(w, 2, coordinates(w), 1, 3)` gives `w == [1, 1, 2, 3]` (by hand: the source is read before the destination changes). *Fails today* (`c` is 2 × 3; for the vector `coordinates(v) === v`) | `test/iterate_queries.jl` |
| **FT02 groups, entries, multi-response** | FC-4, FC-5, FC-6, FS-4, FP-2 | `p, q = 3, 2`; `B = [1 2; 3 4; 5 6]`; `x = AtomIterate(vec(B), [collect(r:3:6) for r in 1:3])`; `y = AtomIterate(6, 3)`; and, by hand, the expected groups `[1, 4]`, `[2, 5]`, `[3, 6]` | `atomindices(x, r)` equals the hand-written group `r` for `r = 1:3`; `atomindices(y, 2) == 3:4` and is a `UnitRange{Int}`; `atomindices(x, 2)[1] = 9` throws (any exception) and the layout is unchanged; `@allocated atomindices(x, 2) == 0`; `atomindices(x, 0)` and `atomindices(x, 4)` throw `BoundsError`; `atom(x, r) == B[r, :]` and `x[(c − 1)·3 + r] == B[r, c]` for all `r, c` (hand-written expectations); `coordinates(x)` throws `MethodError`. Index types (FC-6): for `ai = AtomIterate(4, [Int32(1):Int32(2), Int32(3):Int32(4)])`, `atomindices(ai, 1) == 1:2` and is a `UnitRange{Int32}`, the stored range; for `f32 = GroupNormL2(1.0, [Int32(1):Int32(2), Int32(3):Int32(4)])`, `atomindices(AtomIterate(4, f32), 2)` is a `UnitRange{Int32}`; `atomindices(AtomIterate(4, [Int32[1, 3], Int32[2, 4]]), 1)` has element type `Int`. (Were FD-6's option (e) adopted, the first two would be `UnitRange{Int}`.) FS-4's gradient mapping is a convention for model authors and is not asserted; what the matrix losses do with `x` today is not tested. A view `w = atom(x, 2)` taken before `activate!(x, 1)` and `dropzeros!(x)` still reads `B[2, :]`, and `w[1] = 0` writes `x[2]`. *Fails today* (no `atomindices`) | `test/atom_iterate.jl` |
| **FT03 read-only handles** | FC-7, FW-1 | `x = SparseIterate(4)` with `x[2] = 1`, `x[4] = 3`; `a = AtomIterate(6, [1:2, 3:4, 5:6])` with `a[1] = 1`, `a[5] = 2`; an owner with `X[2, 1] = 1` | `activecoordinates(x)[1] = 3`, `activecoordinates(a)[1] = 2` and the owner's write throw (any exception). Afterwards `Vector(x) == [0, 1, 0, 3]`, `activecoordinates(x) == [2, 4]`, `isactive(x, k) == (k in (2, 4))` for `k = 1:4`; `activecoordinates(a) == [1, 3]` and the consistency helper of `test/atom_iterate.jl:15` holds. Call `dropzeros!(a)` only after asserting that the write threw (today it hangs after a successful write). *Fails today* (#405) | `test/iterate_queries.jl` |
| **FT04 handles after compaction and replacement** | FC-7, FC-8, FW-5, FN-7 | `x = SparseIterate(5)` with values `1, 2, 3` at `1:3`; then `x[1] = 0`, `dropzeros!(x)`. A second `x` with `x[4] = NaN`, `x[5] = -Inf` and an explicit zero at 1; an `AtomIterate` with an all-zero active group and a NaN group. For the replacement case: `x = SparseIterate(3)` with `x[1] = 1`, `x[2] = 2` (slot order `[1, 2]`) and `y` with `y[2] = 2`, `y[1] = 1` (order `[2, 1]`, `x == y`); the same with two `AtomIterate`s of one layout | `Set(activecoordinates(x)) == Set([2, 3])`; `Dict(zip(activecoordinates(x), activevalues(x))) == Dict(2 => 2.0, 3 => 3.0)`, the same through `nonzeros`; no handle from before the call is used. `dropzeros!` keeps 4 and 5 and removes 1; for the `AtomIterate`, it removes only the all-zero group; values unchanged; `@allocated dropzeros!(x) == 0`. After `copyto!(x, y)`, with handles reacquired: `nactive(x) == 2`, `activecoordinates(x) == [2, 1]` and `activevalues(x) == [2.0, 1.0]` (by hand: `y`'s order); the `AtomIterate` gives `[2, 1]`; after `copyto!(x, x)` the order is unchanged. The test does not read a handle taken before the copy | `test/sparse_iterate.jl`, `test/atom_iterate.jl` |
| **FT05 value writes never remove** | FW-2, FW-3, FW-7, FX-2, FN-9 | `x = SparseIterate(4); x[4] = 7; activate!(x, 2)` (old set `[4, 2]`), fresh for each call; `v = [3, 0, 0, 0]`; `y = SparseIterate(4)` with `y[1] = 3` and an explicit zero at 3; `A = [0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]` and `b = [0, 1, 0, 0]` for `mul!`; `M = [0 0; 1 1; 0 0; 0 0]` for `sum!`; `F = lu(Matrix(1.0I, 4, 4))`. The calls: `x[1] = 3`, `x .= v`, `x .= 0`, `x .= 0 .* x`, `x .= 1 .* v`, `x .= y`, `fill!(x, 0)`, `fill!(x, 2)`, `copyto!(x, v)`, `copy!(x, v)`, `x[:] = v`, `copyto!(x, 1, v, 1, 4)`, `view(x, 1:2) .= [3, 0]`, `vec(x) .= v`, `rmul!(x, 0)`, `x .*= 0`, `map!(identity, x, v)`, `ldiv!(x, F, v)`, `permute!(x, [4, 1, 2, 3])`, `mul!(x, A, b)`, `sum!(x, M)`. The same for `a = AtomIterate(4, [1:2, 3:4]); a[4] = 7`, plus `atom(a, 1) .= [3, 0]` | Reference: the same call on `Vector(x)`, a plain `Vector`. Expect `Vector(x) == ref`; `activecoordinates(x)[1:2] == [4, 2]` (old slots kept, in order); `Set(activecoordinates(x)) == Set([4, 2]) ∪ Set(findall(!iszero, ref))` (for `x .= y`, `y`'s explicit zero at 3 does not join). For the `AtomIterate`, the same with groups. Every broadcast equals the explicit loop over the unaliased right-hand side, run on a copy, values and working set. A sweep over `activecoordinates(x)` writing `x[k] = 0` leaves the handle's contents unchanged. Separately: `ldiv!(x, F, y)` with a same-kind `y` gives `y`'s working set (FW-2's stated exception). *Fails today* for `x .= 0`, `fill!(x, 0)`, `x .= y`, `sum!` | `test/iterate_writes.jl` (new) |
| **FT06 state copies** | FW-4, FX-1, FX-6, FC-7 | `y = SparseIterate(4); y[3] = 5; activate!(y, 1)`; `x` with coordinate 2 active; `y64 = SparseIterate(Float64, 3); y64[2] = 1e-50`; two `AtomIterate`s with equal but distinct layouts; `x2` and `y2` with equal entries at coordinates 1, 2 and slot orders `[1, 2]` and `[2, 1]` | `copy(y)`, `copyto!(x, y)`, `copy!(x, y)`: `activecoordinates == [3, 1]`, the explicit zero kept, values equal, `x`'s old 2 gone; `similar(y)` empty with the same size, `similar(y, Float32)` of `Float32`; `zero(y)` empty; `copyto!(y, y)` leaves `y` unchanged. `copyto!(SparseIterate(Float32, 3), y64)` gives values `Float32[0, 0, 0]` and working set `[2]` (by hand). The `AtomIterate`s copy as state; `copy`, `similar` and `AtomIterate(4, f)` share the layout object (the test may check identity on the private field). Equal membership, different order: after `copyto!(x2, y2)` and, separately, `copy!(x2, y2)`, the reacquired `activecoordinates(x2) == [2, 1]` with `nactive(x2) == 2`. **A failed state copy** (FW-4's failure clause): with `snap(z) = (collect(activecoordinates(z)), [isactive(z, k) for k in 1:ncoordinates(z)], [z[i] for i in eachindex(z)], Vector(z))` taken before the call, `copyto!(d, s)` throws (any exception) and `snap(d)` is unchanged, for `s` with `s[1] = 1.0`, `s[2] = 1.5` of `d`'s length and these `Int` destinations: empty of length 16 (sparse branch, `4 · (0 + 2) < 16`) and of length 4 (dense branch); holding 7 at coordinate 5 and 3 at coordinate 9, of length 16 (dense branch, `4 · (2 + 2) = 16`) and of length 32 (sparse branch); and an `AtomIterate(Int, 4, [1:2, 3:4])`, empty and holding 7 at entry 4, from a `Float64` source with 1.0 at entry 1 and 1.5 at entry 3. (Were FW-4's alternative adopted, the test would assert instead that the four parts of `snap(d)` agree with one another.) **A copy that must succeed** (FW-4's domain of conversion, separate from the failed copies): `s = SparseIterate(4); s[1] = 2.0; s[2] = 1.5; s[1] = 0; dropzeros!(s); s[2] = 3.0` has the single active value `3.0` at ID 2, while an unused slot still holds `1.5`; with `d = SparseIterate(Int, 4); d[4] = 7` (dense branch, `4 * (1 + 1) < 4` is false), `copyto!(d, s)` returns without error, `Vector(d) == [0, 3, 0, 0]` and `activecoordinates(d) == [2]`. *Fails today* for the failed copies except the empty length-4 destination: the empty length-16 one ends with `d[1] == 1` and `isactive(d, 1)` but no active coordinate; the length-16 one holding entries ends with coordinate 5 at 1; the length-32 one with `e[5] == 0`, `e[9]` reading 0 while `Vector(e)` holds 3, and active `[1, 9]`; both `AtomIterate`s with `[1, 0, 0, 0]` and active `[1, 2]`; and for the copy that must succeed, which throws `InexactError` and leaves `Vector(d) == [0, 0, 0, 3]` (#416) | `test/sparse_iterate.jl`, `test/atom_iterate.jl` |
| **FT07 imports and lengths** | FX-2, FX-3 | `x = SparseIterate(2, 2)` with `x[1, 1] = 5`; `s = SparseIterate(4)` with `s[4] = 1` and an explicit zero at 2; `d = AtomIterate(4, [[1, 3], [2, 4]])` with `d[2] = 7`; `src = AtomIterate(4, [1:2, 3:4])` with `src[3] = 5` and group 1 active at zero | `copyto!(x, s)`: `Matrix(x) == [0 0; 0 1]`, working set `[1, 4]` (old 1 kept, 4 joins, `s`'s explicit zero does not). `copyto!(SparseIterate(6), SparseIterate(4))` and `copyto!(AtomIterate(6, 3), AtomIterate(4, 2))` throw `DimensionMismatch` with the destination unchanged. `copyto!(d, src)`: `Vector(d) == [0, 0, 5, 0]`, working set `[2, 1]`. All by hand. *Fails today* for `copyto!(x, s)` | `test/iterate_writes.jl` |
| **FT08 replace and clear** | FW-6 | `x = SparseIterate(10)` with working set `[5, 2, 9]` and `A` nonzero at 2, 3, 9; `a = AtomIterate(8, 4)` with working set `[4, 2, 1]` and `Aa` nonzero at 3, 5; an `AtomIterate` with groups `[[2, 4], [1, 3]]`; `w = SparseIterate([1., 2.])` and its view `view(w, 2:-1:1)`; an owner (`p = 3`) with working set `[5, 2]` and `M` nonzero at coordinates 2, 3, 5; an owner `Z` with `Z[1, 1] = 7` and the asymmetric `M2 = [1 2; 3 4]` | `copyto!(dropzeros!(fill!(x, 0)), A)` gives exactly `[2, 3, 9]`, in this order; `dropzeros!(fill!(x, 0))` gives none; for `a`, `[2, 3]`; with groups `[[2, 4], [1, 3]]`, an import of `[1, 1, 0, 0]` after the clear gives `[2, 1]` (the order in which the import first meets a nonzero entry of each group), while `AtomIterate([1, 1, 0, 0], [[2, 4], [1, 3]])` gives `[1, 2]` (group-ID order); for the strided multi-response layout with `B = [0 1; 1 0; 0 0]`, a replace gives `[2, 1]` and the constructor `[1, 2]`. **Same-kind source:** with `y = SparseIterate(4); y[4] = 2.; activate!(y, 1)` and `x4 = SparseIterate(4); x4[2] = 9`, the staged replace (the length and element-type checks, `t = similar(x4)`, the loop `for (i, j) in zip(eachindex(t), eachindex(y)); t[i] = y[j]; end`, then `copyto!(x4, t)`) gives `Vector(x4) == [0, 0, 0, 2]` with working set exactly `[4]`; for `AtomIterate`s with groups `[1:2, 3:4]`, from a source holding 2 at entry 3 with group 1 active at zero, it gives working set `[2]`. The test records that the shorthand would give `[4, 1]` and `[2, 1]` (it state-copies the explicit zero) and does not use it. **Overlapping source:** the staged replace from `view(w, 2:-1:1)` gives `Vector(w) == [2.0, 1.0]` with working set `[1, 2]`; the shorthand would give `[0.0, 0.0]`. **Fallible conversion:** with `xj = SparseIterate(Int, 2); xj[2] = 5`, the staged replace from `[1.0, 1.5]` throws and leaves `Vector(xj) == [0, 5]` with working set `[2]`; for `xm = SparseIterate(3); xm[2] = 4`, a source of length 2 makes the length check throw `DimensionMismatch` and leaves `xm` unchanged; the test records that the shorthand from `[1.0, 1.5]` would leave `[1, 0]` with working set `[1]`. **Cartesian source:** for `x5 = SparseIterate(4)` and `A = view(reshape(1:4, 2, 2), [2, 1], :)` (linear order `[2, 1, 4, 3]`), the staged replace gives `Vector(x5) == [2.0, 1.0, 4.0, 3.0]` with working set `[1, 2, 3, 4]` (a joint `eachindex(t, A)` would throw `DimensionMismatch`). **Complex-typed source:** with `xc = SparseIterate(4); xc[3] = 9`, both recipes given `ComplexF64[1, 2, 3, 4]` throw `ArgumentError` from the element-type check and leave `Vector(xc) == [0, 0, 9, 0]` with working set `[3]`. (Without the check, both loops accept that source today, giving `[1.0, 2.0, 3.0, 4.0]`; the recipe's check is what rejects it.) **Owner:** `copyto!(X, M)` gives exactly `[2, 3, 5]` (CPY-4's increasing ID order), and `fill!(coordinates(X), 0); dropzeros!(X)` gives none; `copyto!(Z, M2)` throws `ArgumentError` and leaves `Matrix(Z) == [7 0; 0 0]` with working set `[1]` (state kept). Reference: the support listed in increasing entry order, by hand. (The shorter `dropzeros!(copyto!(x, A))` gives `[3, 2, 9]`; the test does not use it.) *Fails today* for the owner's `copyto!(X, M)`, a value write that keeps the old slots (`dropzeros!(copyto!(X, M))` gives `[5, 2, 3]`) | `test/iterate_writes.jl`, `test/symmetric_mutators.jl` |
| **FT09 activation after conversion** | FN-6, FW-2 | `Float32` iterates of both types, fresh for each path; `u = [1e-100, 1e-100]` | Every path of the edge-case list with `1e-100` leaves `nactive == 0` and values `0f0`; the same write to an *active* coordinate stores `0f0` and keeps it active; NaN activates, `−0.0` does not. *Fails today* for `fill!(a, 1e-100)` and `a .= 1e-100` (2 groups) | `test/precision_mixed.jl` |
| **FT10 oracle destinations** | FW-8 | `y = SparseIterate(4); y[4] = 7`; `prox!(y, NormL2(λ), [3, 0, 0, 0], 1)` for λ = 0, 1, 10; an `AtomIterate` with group 2 active and `GroupNormL2(λ, [1:2, 3:4])`; `prox!(y, IndBallL0(s), SparseIterate([2, 1, 0, 0]), 1)` for `s = 0, 1, 3` | Values from closed forms written in the test: `max(1 − λ/3, 0)·[3, 0, 0, 0]`; for the group, group 1 is `max(1 − λ√2/3, 0)·[3, 0]`, whose first entry at λ = 1 is `3 − √2`, compared with `≈` at relative tolerance `4eps()` (in `Float64`, `3 − √2` is `1.5857864376269049` and the kernel's `(1 − √2/3)·3` is `1.5857864376269046`); `IndBallL0`: `[0, 0, 0, 0]`, `[2, 0, 0, 0]`, `[2, 1, 0, 0]`. Every coordinate with a nonzero value is active, every inactive one is zero, and after `dropzeros!` the working set equals the support. Which zeros stay active is *not* asserted. (Defect 5 of the contract is allowed by FW-8, so this test passes today) | `test/iterate_writes.jl` |
| **FT11 three-factor products (FD-16, open)** | FN-10 | Four cases, each with `Symmetric` `A` and `B`: **(p1)** `X` 4 × 4 nonzero (1) at `(1, 1)`, `(2, 2)`, `(3, 1)`, `A = Symmetric(Am)` with `Am = Matrix(1.0I, 4, 4)`, `Am[1, 2] = Am[2, 1] = Inf`, `B = Symmetric(ones(4, 4) + 3I)` (sparse path); **(p4)** `X` nonzero at `(1, 1)`, `(2, 2)`, `(3, 3)`, `(1, 2)`, `(2, 3)`, `A = Symmetric(Matrix(1.0I, 4, 4) .+ 1)`, `B` the same with `B[4, 1] = B[1, 4] = NaN` (dense fallback); **(p6)** `X` nonzero at `(1, 1)`, `(1, 2)`, `(2, 1)`, `(3, 1)`, `A = Symmetric(Matrix(Diagonal([Inf, 1, 1, 1])))`, `B = Symmetric(Matrix(1.0I, 4, 4))`; **(R5)** `A = Symmetric([NaN 0 0; 0 1 0; 0 0 1])`, `B = Symmetric(Matrix(1.0I, 3, 3))`, `X1` with one nonzero at `(2, 2)` (sparse path) and `X2` nonzero in rows and columns 2:3 (dense fallback); and the **mixed fixture** `u = 2.0^600`, `A = Symmetric([Inf 0; 0 u])`, `B = Symmetric([1 0; 0 1/u])`, `X` with `X[1, 1] = 1`, `X[2, 2] = u` | Under FD-16's option (a). Reference: for a non-finite entry of the ordinary product, the term-wise loop written in the test, `R[i, j]` summing `(A[i, k] * X[k, l]) * B[l, j]` over the `(k, l)` with `X[k, l] ≠ 0` in column-major order of `X`; a finite entry is the ordinary product's. Compared with `isequal` for the class and `≈` (relative `1e-12`) for finite entries: (p1) row 2 `[Inf, Inf, Inf, Inf]`; (p4) column 1 `[9, 8, 7, 6]`; (p6) row 1 `[NaN, NaN, NaN, NaN]` (under option (c), `A * (Matrix(X) * B)`, it would be `[Inf, Inf, NaN, NaN]`); (R5) row 1 `[0, 0, 0]` for `X1` and `X2`; mixed fixture `[Inf NaN; 0.0 u]`, with `(2, 2) == u` exactly (`4.149515568880993e180`), not `Inf`. The expectations change with the option FD-16 adopts; under option (d) the test is dropped. *Fails today* for p1, p4 and R5's `X2`; p6, R5's `X1` and the mixed fixture pass today | the #387 conformance suite, `test/conformance/failures_storage.jl` (M1 rows, each marked `broken = (group = :G2, baseline = …)` until FD-16 is decided and implemented); `test/sparse_iterate.jl` |
| **FT12 strong zeros with one iterate operand** | FN-1, FN-2, FN-4 | `s = [1, 0, 0]` stored at 1; `t` with NaN at 2; `A = [NaN 1 0; NaN 1 0]`; `u` with `u[2] = 1` and an explicit zero at 1; `Z = SparseIterate(2, 2)` with `Z[2, 2] = 1` and an explicit zero at `(1, 1)`; `S3 = Symmetric([NaN 0 0; 0 1 0; 0 0 1])`; matrix coefficients `64 × 2` (stored-entry walk) and `3 × 2` (column walk) with an explicit zero against a NaN column; `a = AtomIterate(4, [1:2, 3:4])` with `a[3] = 1` and group 1 active; `v = [NaN, 0, 1, 0]`; an owner with `X[2, 2] = 1` and an explicit zero at `(3, 1)`; `M` with NaN at `(3, 1)` and `(1, 3)` | Reference: an explicit loop over the coordinates where the iterate's coefficient is nonzero. `dot(s, t) == dot(t, s) == 0.0`; `A * u == [1.0, 1.0]` before and after `dropzeros!(u)`, also through `mul!(out, A, u, 2, 0)` (`[2, 2]`), `transpose(permutedims(A)) * u` and `dot(A[1, :], u) == 1.0`; `S3 * u == [0, 1, 0]`; `dot([NaN, 1], Z, [1, 1]) == 1.0` before and after `dropzeros!`; both matrix walks give the loop's result. A non-finite value *in* the iterate meeting a dense zero propagates: with `w[1] = Inf`, `dot([0, 1, 0], w)` is NaN. `dot(v, a) == dot(a, v) == 1.0` before and after `dropzeros!`; `[NaN 0 1 0] * a == [1.0]`; `mul!(y, matrixview(X), [Inf, 1, 0])` and `matrixview(X) * [Inf, 1, 0]` give `[0, 1, 0]` before and after `dropzeros!`; `dot(M, X) == M[2, 2]`; `dot(a3, X, b3)` with a NaN at a zero coordinate equals the loop. The test asserts the classification for zeros and non-finite entries only; the finite values of these cases are small integers, so every order sums them exactly. It records as allowed to differ (FN-4, A1) the finite-overflow pair `M = floatmax(Float64)`, `x = SparseIterate([M, M, -M])`, `y` with ones activated in the order 1, 3, 2: `dot(x, y) == Inf` and `dot(y, x) == M` today. *Fails today* for every stored-zero and order case (NaN, `[0, 1, NaN]`) | the #387 conformance suite, `test/conformance/failures_storage.jl` (M1 rows, each marked `broken = (group = :G2, baseline = …)` until FD-9 lands); `test/sparse_iterate.jl`, `test/atom_iterate.jl`, `test/symmetric_mul.jl` |
| **FT13 the fallback boundary and objects built from an iterate** | FN-3 | `s` with `s[2] = 1`; `A = [NaN 1; NaN 1]`; the `AtomIterate` `a` of FT12; `a2 = AtomIterate(4, [1:2, 3:4])` with `a2[2] = 1`; `s0 = SparseIterate([0., 1.])`; `b0 = AtomIterate([0., 1.], [1:1, 2:2])`; `m = SparseIterate(2, 2)` with `m[2, 2] = 1` | `sparse(A) * s`, `Diagonal([NaN, 1]) * s`, `dot(sparsevec([1, 2], [NaN, 1]), s)`, `dot(view([NaN, 1, 0], [1, 2]), s)`, `sum(s .* [NaN, 1])` equal (`isequal`) the same calls on `Vector(s)`: `[NaN, NaN]`, `[NaN, 1.0]`, NaN, NaN, NaN; likewise `sparse([NaN 0 1 0]) * a`, `dot(sparsevec([1, 3], [NaN, 1], 4), a)` and `transpose(permutedims([NaN 0 1 0])) * a` (`[NaN]`). By the returned object: `vec(s0) === s0` and `vec(b0) === b0`, so `dot([NaN, 1], vec(s0)) == dot([NaN, 1], s0) == 1.0` and `dot([NaN, 1], vec(b0)) == 1.0` (the iterate's own rule); `dot([NaN, 1], view(s0, :))` is NaN; for the matrix, `dot([NaN, 0, 0, 1], vec(m))` is NaN (a wrapper). Views are ordinary vectors: `dot(atom(a2, 1), [NaN, 1])` is NaN (it equals `dot([0, 1], [NaN, 1])`), and so is a product through `coordinates` (FT01). For an owner with an explicit zero at `(3, 1)`, the matrix–matrix `mul!(C, matrixview(X), [Inf; 1; 0;;])` and `dot(M, matrixview(X))` with a NaN of `M` there equal the same calls on `Matrix(X)` (`isequal`) | `test/nonfinite_inputs.jl` |
| **FT14 a data-loss value** | FN-5 | `f = LeastSquaresLoss([1, 1], [NaN 1; NaN 1])`; `s` with `s[2] = 1` and an explicit zero at 1; `fa = LeastSquaresLoss([1], [NaN 0 1 0])` at the `AtomIterate` of FT12 | `f(s) == 0.0` before and after `dropzeros!`; `isnan(f([0, 1]))`; `fa(a) == 0.0`. Reference: the residual `y − Σ_{k: s_k ≠ 0} X[:, k] s_k`, a loop in the test. *Fails today* (`f(s)` is NaN with the stored zero) | the #387 conformance suite, `test/conformance/failures_storage.jl` (M1 rows, each marked `broken = (group = :G2, baseline = …)` until FD-9 lands) |
| **FT15 coordinate oracles** | FA-1–FA-6, FL-4, FC-1 | `x = AtomIterate([1, 2, 0, 0], [1:2, 3:4])`; `fbad = GroupNormL2(1, [[1, 3], [2, 4]])`; `g = [3, 4, 0, 0]`; `x2 = AtomIterate(2, [1:2])` and `g2 = [5/8, 5/8]`; a `SparseIterate` with `x[1] = 0.5` | Return types: a `Float64` for `SparseIterate`, `atom(x, k)` for `AtomIterate`. `prox_coordinate!(x, fbad, 1, 1)`, `prox_coordinate!(x, fbad, 1, [9, 9], 1)` and `kkt_violation(fbad, g, x, 1)` throw `DimensionMismatch`; `k = 3` throws `BoundsError`; `u = [1]` throws `DimensionMismatch`; a complex `u` and a non-separable non-group `f` throw `ArgumentError`; after each, `Vector(x)` and the working set are unchanged. A group step to zero on an absent group leaves it absent; a scalar step to zero on an active coordinate keeps it active. `prox_coordinate!(y, NormL1(0.5), 1, atom(y, 2), 1)` with `atom(y, 2) == [3, 4]` gives `[2.5, 3.5]` (soft threshold by hand). Screens, each by hand from the penalty's own dual norm and default weight (contract, FA-6's table): `GroupNormL2(1, [1:2, 3:4])` at `g`, `≈ 5 − √2` (ℓ₂ norm 5, weight `√2`); **`GroupNormLinf(1, [1:2])` at `g2`, `== 0.25`** (ℓ₁ norm `5/4`, weight 1: the group must enter), where the ℓ₂ formula would give `−0.11611652351681556`, a screen-out; `GroupNormLinf(1, [1:2, 3:4])` at `g`, `== 6.0` (`7 − 1`); `SparseGroupNormL2(1, 1, [1:2, 3:4])` at `g`, `≈ √13 − √2` (the soft threshold of `[3, 4]` at 1 is `[2, 3]`); `GroupMCP` and `GroupSCAD` at `g`, `≈ 5 − √2`; `Scaled(GroupNormLinf(1, [1:2]), 2)` at `g2`, `== −0.75`; `Regularize(GroupNormL2(1, [1:2]), 1, [1, 1])` at `[3, 4]`, `≈ √13 − √2`; `kkt_violation(NormL1(1), g, x, 1) == 3.0` (largest of `|3|, |4|` minus 1); for a `SparseIterate`, `kkt_violation(NormL1(1), [3, −0.5], x, k)` is `2.0` and `−0.5`. *Fails today:* none (the code already computes each penalty's screen) | `test/block_step.jl`, `test/interface.jl` |
| **FT16 block trials** | FA-7, FW-1 | `f = GroupNormL2(1, [1:2, 3:4])`, `x = AtomIterate(4, f)`; `fl = GroupNormLinf(2, [1:2, 3:4])`, `xl = AtomIterate(4, fl)`; also `SparseGroupNormL2`, `GroupMCP` and `NormL1`, groups of sizes 1, 8, 64, contiguous and not; a non-strided `u` whose `getindex(2)` throws | `prox_coordinate!(x, f, 1, [3, 4], 1) ≈ (1 − √2/5)·[3, 4]` (by hand, `rtol = 4eps`); `prox_coordinate!(xl, fl, 1, [1, 4], 1) == [1, 2]` (by hand: the prox of `2‖·‖∞` clips at τ = 2, where `max(4 − τ, 0) + max(1 − τ, 0) = 2`); for each penalty the five-argument result `==` the result of staging `u` and calling the four-argument form (this package's own test may stage through the private field); with `fbad` the five-argument call throws and `x` is unchanged and consistent; with the throwing `u`, FW-1 holds after the exception (the test does not require `x` unchanged) | `test/block_step.jl`, `test/block_step_failures.jl` |
| **FT17 layout** | FL-2, FL-4, FL-5, FL-6, FL-7, FX-6 | `gv = [[1, 2], [3, 4]]`, `a = AtomIterate(4, gv)`; `fm = GroupNormL2(1, [[1, 3], [2, 4]])`, `am = AtomIterate([1, 2, 3, 4], [1:2, 3:4])`; the counting index type of `test/atom_iterate.jl:26–31` | After `gv[1][1] = 3`, `atomindices(a, 1) == [1, 2]`. `samegroups(a, GroupNormL2(1, [[1, 2], [3, 4]]))` is `true`; with `[[2, 1], [3, 4]]` `false` (ordered lists); `samegroups(a, copy(a))` `true`; `samegroups(a, AtomIterate(4, [[1, 3], [2, 4]]))` `false`. `fm(am) == fm(Vector(am))` and `prox!(ym, fm, am, 1) == prox(fm, Vector(am), 1)` (the dense calls are the reference). With counted indices: one full comparison when a distinct equal layout is first seen, none on the next call with the same object, one per call when two distinct equal layouts alternate. If FL-3 is adopted, `parentindices(atom(a, 1))[1][1] = 3` throws. *Fails today* (no `samegroups`, no `atomindices`) | `test/atom_iterate.jl` |
| **FT18 counts and capacity** | FC-9, FC-10, FC-11, FP-3 | `y = SparseIterate(5); y[1] = 1; activate!(y, 3)`; `m = SparseIterate(2, 2)` with `m[2, 1] = 3` and an explicit zero at 4; an `AtomIterate` with one nonzero group, one all-zero active group and one inactive group | `nnz(y) == nactive(y) == length(nonzeros(y)) == 2`; `count(!iszero, activevalues(y)) == 1`; `findnz(y) == ([1, 3], [1.0, 0.0])`; `nonzeros(y)[2] = 4` gives `y[3] == 4`; `findnz(m) == ([2, 2], [1, 2], [3.0, 0.0])`; `nnz(a)` and `activevalues(a)` throw `MethodError`; FC-10's entry and group counts equal `count(!iszero, Vector(a))` and the group count by hand. `Base.summarysize(SparseIterate(Float64, n))` within 1 KiB of `24n`; `AtomIterate(Float64, p, K)` with range groups within 1 KiB of `16p + 32K` | `test/iterate_queries.jl` |
| **FT19 aliasing** | FX-5 | `x = SparseIterate(2); x[2] = 1; x[1] = 2`; `a` an `AtomIterate` with one active group; `w = SparseIterate([1, 2, 0, 4])` | `copyto!(x, activevalues(x))` gives `[1, 2]` (the handle reads `[1, 2]` in slot order, by hand); `Base.mightalias` of `activevalues(x)`, `nonzeros(x)` and `activecoordinates(x)` with `x` is `true`, and of `activecoordinates(a)` with `a`; `copyto!(w, view(w, 4:-1:1))` gives `[4, 0, 2, 1]`; `w .= 2 .* w` doubles. *Fails today* (`[1, 1]`; #406) | `test/iterate_writes.jl` |
| **FT20 real-only writes and ambiguities** | FX-4, FN-8, FS-3 | both types with one active entry; `c = ComplexF64[1, 2im, 0, 0]`; `xm = SparseIterate(2, 2)`; `s0 = SparseIterate([0., 1.])` and `b0 = AtomIterate([0., 1.], [1:1, 2:2])` | `copyto!(x, c)`, `copy!(x, c)`, `x .= c`, `x .= 2im`, `fill!(x, 2im)` throw `ArgumentError` and leave `x` unchanged; `SparseIterate(ComplexF64, 2)` and `AtomIterate(ComplexF64, 4, 2)` throw `ArgumentError`. By the returned object: `vec(s0) .= ComplexF64[1, 2im]` and `vec(b0) .= ComplexF64[1, 2im]` are writes into the iterates themselves and throw `ArgumentError` with `s0`, `b0` unchanged (as today); `view(x, 1:2) .= ComplexF64[1, 2im]`, `vec(xm) .= c` (today `1.0` written, then `InexactError`) and, through the coordinate view, `coordinates(x) .= c` throw (any exception; the test does not require `x` unchanged, FN-8). For a vector `x` the last one is a change: today it throws `ArgumentError` before writing, because `coordinates(x)` is `x`. The checked `copyto!` takes every sparse source without ambiguity: `copyto!(xm, sparse([1.0 0; 0 2]))`, a view, an adjoint and a transpose of it all give `Matrix(xm) == [1 0; 0 2]`, and the ambiguity check of `test/aqua.jl` (`Aqua.test_all`, which includes `test_ambiguities`) passes. *Fails today* for `copyto!`, `copy!` (`InexactError` after a write) and `fill!` (`InexactError`) | `test/real_iterates.jl`, `test/aqua.jl` |
| **FT21 empty and malformed** | FL-1, FS-3 | the edge-case list of §4 | the outcomes listed there, with the exact messages | `test/atom_iterate.jl`, `test/iterate_queries.jl` |
| **FT22 equality and shape** | FS-1, FS-2, FS-5 | `a`, `b` `AtomIterate`s with entries `[1, 0, 0, 0]`, `b` with group 2 also active; `s`, `t` `SparseIterate`s likewise | `a == b`, `isequal(a, b)`, `hash(a) == hash(b)`, `a == [1, 0, 0, 0]`; the same for `s`, `t`; `s isa AbstractArray`, `a isa AbstractVector`; `matrixview(s)` throws `MethodError` | `test/iterate_queries.jl` |
| **FT23 allocation** | FP-1, FP-2, FP-5, FP-7, FW-5 | warmed-up `Float32` and `Float64` fixtures of §6 | `@allocated` is 0 for `x[i]`, `x[i] = v`, a first activation, `isactive`, `activate!`, `activecoordinates(x)`, `activevalues(x)`, `coordinates(x)`, `atom(x, k)`, `atomindices(x, k)`, `dropzeros!`, `fill!(coordinates(x), 0)` on both coordinate views (FP-7, whose `O(s)` cost §6 measures), `mul!(out, A, x)`, `dot(v, x)`, the five-argument `prox_coordinate!` with a strided `u`, and the loops of §3.4. Growth with `n`, `s` and `p` is checked in §6 | `test/allocations.jl` |
| **FT24 performance gates** | FP-6, FP-7 | §6 | each FD case against the 0.5.x baseline (the three-factor cases only if FD-16 adopts an option that changes them): a repeatable slowdown above 5% on a representative case is investigated and reported on the implementing pull request, and merging it needs the maintainer's recorded approval (numerics C3) | `benchmark/` |
| **FT25 no private fields downstream** | FF-1, FF-2 | ProximalSolvers `src/` | a scan for `.nzval`, `.nzval2ind`, `.ind2nzval`, `.storage`, `.active`, `.nactive`, `.position`, `.group_of`, `.layout_cache` and `.groups` on iterate receivers finds none (an allowlist names the penalty-field reads) | ProximalSolvers `test/` (outside this repository) |
| **FT26 continuation and workspace start** | FW-2, FW-6 | `solve!` with `continuation > 0` from an `x` holding a nonzero and an explicit zero, S and A; a CD workspace with a dense start solved twice | the result equals that from a fresh zero iterate (`1e-10`, cyclic order); the working set is empty right after the start (through a test hook on the start helper); the second workspace solve starts with the dense start's support in the slot order of 0.5.x's fill-and-scatter (the §3.5 recipe; for strided groups this differs from a fresh construction, FW-6). Fails after FD-2 without the §3.5 change | ProximalSolvers `test/` |
| **FT27 momentum through accessors** | FC-6, FC-8, FC-9, FF-2 | CD with momentum on a lasso (S) and group lassos (A, contiguous and strided groups), cyclic order | the objective after every pass equals, to relative `1e-12`, a run recorded at the audited commits; FT25 finds no field use | ProximalSolvers `test/` |
| **FT28 block trials without private writes** | FA-7, FF-2 | `_groupstep!` for `GroupNormLinf`, `SparseGroupNormL2` and `NormL1` on `LSModel` and `QuadModel` | iterates equal, to relative `1e-12`, a run recorded at the audited commits; a step whose `prox_coordinate!` throws leaves no inactive group with a nonzero entry | ProximalSolvers `test/` |
| **FT29 scaling by a number** | FN-9 | `x = SparseIterate(4)` with `x[1] = 1` and an explicit zero at 2; `b = AtomIterate(4, [1:2, 3:4])` with `b[1] = 1` | Reference: `x .*= s` (respectively `x ./= s`) on a copy, values and working set. `rmul!(x, Inf)`, `lmul!(Inf, x)`, `rdiv!(x, 0.0)`, `ldiv!(0.0, x)` give `[Inf, NaN, NaN, NaN]` with working set `[1, 2, 3, 4]` (today `[Inf, NaN, 0, 0]` with `[1, 2]`); `rmul!(b, Inf)` gives `[Inf, NaN, NaN, NaN]` with groups `[1, 2]` (today too); `rmul!(x, 2.0)` gives `[2, 0, 0, 0]`, keeps `[1, 2]` and allocates nothing. *Fails today* for `SparseIterate` | `test/sparse_iterate.jl`, `test/atom_iterate.jl` |

**Requirement coverage.** FS-1 FT01, FT22; FS-2 FT22; FS-3 FT20, FT21; FS-4
FT02; FS-5 FT22. FC-1 FT01, FT15; FC-2 FT01; FC-3 FT01; FC-4 FT02; FC-5 FT02;
FC-6 FT02, FT27; FC-7 FT03, FT04, FT06; FC-8 FT04, FT27; FC-9 FT18, FT27; FC-10 FT18;
FC-11 FT18. FW-1 FT03, FT16; FW-2 FT05, FT09, FT26; FW-3 FT05; FW-4 FT06; FW-5
FT04, FT23; FW-6 FT08, FT26; FW-7 FT05; FW-8 FT10. FX-1 FT06; FX-2 FT05, FT07;
FX-3 FT07; FX-4 FT20; FX-5 FT19; FX-6 FT06, FT17. FL-1 FT21; FL-2 FT17; FL-4
FT15, FT17; FL-5 FT17; FL-6 FT17; FL-7 FT17. FN-1, FN-2, FN-4 FT12; FN-3
FT13; FN-5 FT14; FN-6 FT09; FN-7 FT04; FN-8 FT20; FN-9 FT05, FT29; FN-10 FT11. FA-1–FA-6 FT15; FA-7 FT16,
FT28. FP-1 FT23; FP-2 FT02, FT23; FP-3 FT18; FP-5 FT23; FP-6 FT24; FP-7 FT23, FT24. FF-1 FT25;
FF-2 FT25, FT27, FT28.
**Non-promises, without a test of their own:** FL-3 and FP-4 (recommendations;
FT17 gains a case if FL-3 is adopted); the "unspecified" part of FW-8 (FT10
deliberately does not assert which zeros remain).

**Tests that fail at the audited commit, by defect** (contract Part 4): the
writable handles, FT03; defect 1, FT12; defect 2, FT12, FT14; defect 3,
FT11 (FD-16, open); defect 4, FT05; defect 5, none (allowed by FW-8; FT10 passes); defect 6,
FT05, FT09; defect 7, FT19; defect 8, FT20; defect 9, FT29; defect 10, FT06 (#416).

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
- **FD-5:** indexing through the `coordinates` view (a read-write sweep, for
  N = 1 and N = 2) against the same sweep through `x`, and a product through
  the view (`dot(v, coordinates(x))`, a generic `O(n)` loop) against
  `dot(v, x)`;
- `atom` access: `norm` and `dot` of `atom(x, k)` (BLAS for contiguous
  groups);
- **FD-6:** iteration over `atomindices(x, k)` against `x.groups[k]`, and
  `view(g, atomindices(x, k))` with a vector group;
- **FD-9:** `dot(v, x)`, `mul!(out, A, x)` (strided, transposed, `Symmetric`;
  both matrix walks) and the `AtomIterate` `dot`, with finite data: the cost
  of the strong-zero branch;
- **FD-16** (only if an option other than (d) is adopted): `A * X * B` on
  both paths with finite data (under option (a), the cost of the `isfinite`
  scan of the output) and with one non-finite entry that meets a zero of `X`
  (the term-wise recomputation). The recomputation changes the algorithm for
  those entries, not only a branch, so its case is reported apart;
- **FN-9:** `rmul!(x, s)` with a finite `s` (the `O(s)` loop) and with
  `s = Inf` (every entry), for both types;
- **FP-7:** `fill!(coordinates(x), 0)` on both coordinate views, and the
  owner's clear `fill!(coordinates(X), 0); dropzeros!(X)`, as `s` grows at
  fixed `q` (the case of ProximalSolvers' `src/cd/quic.jl:91`);
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
- **FW-4's failure clause:** a same-kind `copyto!` with equal element types
  and `Float64 → Float32` (no check, so no change allowed), and
  `Float64 → Int` (the `O(s)` check before writing), on both branches of the
  copy kernel;
- **FW-6:** the cheap replace `copyto!(dropzeros!(fill!(x, 0)), A)` against
  the staged replace (`similar(x)`, the value loop, the state copy), which
  costs `O(n)` memory and time;
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
| `A * X * B`, finite (FD-16, option (a)) | as today's paths, plus one scan of the output | as today's paths | today's arrays |
| `rmul!(x, s)`, finite `s` | flat | linear | 0 B |
| `fill!(coordinates(x), 0)` | flat | linear | 0 B |

**Baseline.** Record the baseline at a pinned 0.5.x commit **before any FD
is implemented, additive ones included**: master with the two groups added
and no FD merged. Each FD pull request then reports its cases against that
baseline in the same session, with `benchmark/compare.py`.

**Gate and review thresholds** (numerics C3). A repeatable slowdown above 5%
on a representative case is investigated and reported on the implementing
pull request, and merging it needs the maintainer's recorded approval (FP-6).
A cost above 10% paid only by narrow types needs the maintainer's approval;
machinery that only narrow types need leaves `Float64` timings within the
harness's noise.

**What a benchmark can and cannot show.** A flat curve over a factor of 100 in
`n` is a regression alarm, not a proof of `O(1)`; the proof is the code (a
direct index table and no growth), a review item. Single runs are
exploratory. A benchmark does not show correctness: every case is checked
first, and FT01–FT23 carry the guarantees.

# 7. Migration and dependency plan

Implementation issues are opened only after the maintainer signs off the
decisions; the structural group and FD-9 were signed off on 2026-11-01. #405 and #406 are
fixed on master (#414 and `00c9852c`), #416 has an open fix (#417), #419
tracks FD-16, and the follow-on issues below were filed on 2026-11-01.

| item | kind | depends on | integrates with | follow-on issue |
|:--|:--|:--|:--|:--|
| 1. `sparse` and `atom` benchmark groups, 0.5.x baseline | harness only | — | #401's conventions; #391 Phase 0 | #420 |
| 2. FD-7 read-only `activecoordinates` | 0.5.x bug fix | 1 | ACT-3; the owner in #391 Phase 1a | #405 (fixed in #414) |
| 3. FD-14 `dataids` of the handles | 0.5.x bug fix | 1 | #367; #366 G12 | #406 (fixed in `00c9852c`) |
| 4. FD-13 real check in `copyto!` and `copy!` | 0.5.x bug fix | — | #364; #366 G12 (O6.1) | on #364 |
| 5. FN-6 `AtomIterate` `fill!` converts first | 0.5.x bug fix | — | #366 G1 (conversions) | #421 |
| 6. FD-6 `atomindices` and `ProximalBase.samegroups` | 0.5.x-safe addition | 1 | ProximalSolvers migration | #422 |
| 7. FD-4, FD-10, FD-12: state the rules 0.5.x already meets | 0.5.x-safe addition (documentation) | — | `docs/src/iterates.md` | #423 |
| 8. ProximalSolvers: §3.1–§3.4 and §3.8 | downstream, runs on 0.5.x | 6 | #391 Phase 2 | mlakolar/ProximalSolvers.jl#165 |
| 9. ProximalSolvers: §3.5 | downstream, runs on 0.5.x | — | must land before 10 | mlakolar/ProximalSolvers.jl#164 |
| 10. FD-2 one working-set rule | breaking 0.6 | 1, 9 | #391 Phase 1 integration branch | #424 |
| 11. FD-3 owner `copyto!` is CPY-4's checked replace | breaking 0.6 for the owner, as #391 planned | — (no amendment) | #391 Phase 1a | part of #391 Phase 1a |
| 12. FD-5 a coordinate-vector view for `SparseIterate` | breaking 0.6 | 1 | #391 Phase 1 (the owner's `SymmetricCoordinates` is the model) | #425 |
| 13. FD-9 strong zeros in products with one iterate operand, with the M1 amendment | breaking 0.6 | 1 (FD-9 signed off; PERF-6: option (i)) | #366 G2, with the M1-amendment rows added to the #387 conformance suite (`test/conformance/failures_storage.jl`), marked `broken = (group = :G2, baseline = …)` until it lands; #391 Phase 1c (PERF-6) | #428 |
| 14. FL-3 read-only vector groups | breaking 0.6 (recommendation) | 1 and its measurement | — | #426 |
| 15. FF-1 and the documentation rewrite | 0.6 documentation | 10, 12, 13 | #391 Phase 1d | part of #391 Phase 1d |
| 16. FN-9 scaling by a number follows IEEE | breaking 0.6 for `SparseIterate` | 1 | #366 G2 | #427 |
| 17. FD-13's O6 and S4 text, and FD-16's M1 paragraph if option (a) is adopted | 0.6 documentation | sign-off | #366 (numerics page) | part of item 13's numerics change |
| 18. FD-16 three-factor products | open; breaking 0.6 for options (a)–(c) | its separate sign-off; 1 | #366 G2; #391 Phase 1c (the owner's method) | #419 (filed: the decision, then its implementation) |
| 19. FW-4's clauses: a state copy converts only active values, and a failed one leaves the destination unchanged | 0.5.x bug fix | 1 | #391 Phase 1a: the owner's CPY-3 runs through the same kernel (`src/sparse_iterate.jl:1122–1124`) and is covered | #416 (fix in #417) |

**Order.** Items 1–9 and 19 can run in parallel with #391 Phase 0: they are
additive, bug fixes or downstream changes that work on 0.5.x, and item 1 comes
first so that every later item has a baseline. Items 10–18 change 0.6 behavior
and stack with #391 Phase 1 on its integration branch. Items 10, 11, 12 and
14 need the structural sign-off (contract §3.1) and items 13 and 16 FD-9's
(FD-9's option (e) is the rejected alternative to FN-9), both given on
2026-11-01; item 18 waits for FD-16's (#419), and item 15 documents whatever has been signed; item 9 must be released
in ProximalSolvers before item 10, or the continuation start changes silently.
