# SymmetricSparseIterate in 0.6: migration and acceptance

> **Draft for maintainer review — proposed 0.6 behavior, not a description of an
> installed release.** This guide explains how real callers move to the
> [contract](symmetric-sparse-iterate-0.6.md) and how a future implementer
> proves conformance. It links to the contract's requirement IDs and restates
> none of the rules. Nothing in it is implemented. Every `julia` block that uses
> `matrixview`, `activevalues` or an owner that is not an `AbstractMatrix` is a
> *future* example; none is a doctest.

# 1. What was audited

**Commits.** ProximalBase `abaceda25442bf6cc61601c8c1d3b2a16e1a3492` (master).
ProximalSolvers `3893fb9` (master, equal to `origin/master` after a fetch).
HighDimensionalModels `4503d3a` (master, equal to `origin/master`).
CovarianceSelection and DiffPrecTest: the local checkouts are behind their
`origin/master` (`48f7387` against `1743682`, `6ba332c` against `ff5f432`); I
read the local trees and then re-ran the field-access search with `git grep` on
`origin/master`. The results matched. No downstream repository was modified.

**Traced in detail.**

| path | files |
|:--|:--|
| a coordinate update with a cache | ProximalSolvers `src/cd/kron.jl` (`KronModel`), `src/cd/kronu.jl` (`KronUModel`), `src/cd/models.jl:146,419–420` |
| the sweep and the momentum history | `src/cd/solver.jl:280–335` (full and active passes), `:190–200`, `:540–600` (history by slot) |
| the matrix-oriented path | `src/workspaces.jl:70–99`, `src/solvers/proximal_gradient_descent.jl:94–160`, `src/blocks.jl:26–92`, `src/solvers/diagonal_metric.jl:102` |

**Mentioned but not traced** (they refer to `SymmetricSparseIterate`; I did not
read how): ProximalSolvers `src/cd/gap.jl`, `path.jl`, `lla.jl`, `glm.jl`,
`quic.jl`, `drivers.jl`, `generic_newton.jl`, `solve.jl`, `precompile.jl`;
`_workspace_noalias`, `_size`, `_check_iterate`, `reinit!` and the accelerated
solvers (`src/solvers/*accelerated*`). HighDimensionalModels
`src/coordinate_descent.jl` and `src/losses.jl:236–256` were read only at the
lines cited below. Treat all of these as **unaudited**; a search miss there is
not evidence of compatibility.

**Not accessible / not done.** FunctionalDataAnalysis, FunctionalGraphicalModels
and KernelSmoothing exist beside these checkouts; I did not search them.
No private downstream package was available. I did not run any downstream test
suite.

**Second revision after review (PR #388).** ORC-3 now matches numerics O7 (T11,
T21); T16 asserts `screen == 0` and `r[k] == 2` only; T07's coupled fixture
starts from zero with a known solution; the benchmark protocol times real first
activation and a real sweep against the pre-refactor owner; IDX-4, MAT-2, VAL-4
and sign-off items 7 and 11 were reworded.

**Revision after the first review (PR #388).** Added: screening against stationarity
(T16 and the T07 residual), the asymmetric penalty reaching a real solver
(T17), the `x.p` reads, the shared state type and `_same_support` (§2.3–§2.4,
T18), allocating oracles (T19) and the step scope (T20). I checked the three
source claims in the review (averaging at `src/cd/models.jl:540`; one type
parameter `X` for `x`, `x_previous`, `xhat`, `delta_x` at
`src/solvers/proximal_gradient_descent.jl:67–77`; `_same_support` by
`eachindex` at `:200`) against the audited commit; all three hold.

**Checks I actually ran.**

1. Two throwaway scripts (kept outside the repository) against ProximalBase at
   `abaceda2` with Julia 1.13.3, `julia --project=.`, reporting the values in
   Part 4 of the contract and the table in §2.1 below. They were read-only.
2. The documentation build (`julia --project=docs docs/make.jl`), result in §6.

I did **not** run the package test suite, the benchmark suite, or any timing.
Every "future target" below is labeled as one.

# 2. Migration

## 2.1 `X[k]` cannot be migrated by search-and-replace

At 0.5.x `X[k]` is the matrix entry at *linear* index `k` (since #22). In 0.4
and earlier it was lower-triangle coordinate `k`. Callers that were written
for the old meaning may still read it that way. Identify the intent first
(contract [IDX](symmetric-sparse-iterate-0.6.md#22-index-spaces)):

| what `k` is at that site | 0.6 form |
|:--|:--|
| a coordinate ID, from `1:ncoordinates(x)` or `tril_index` | `coordinates(x)[k]` (hoist `cx = coordinates(x)` out of a loop) |
| a matrix linear index, from `1:p²` or `(c-1)p + r` | `matrixview(x)[k]` |
| a slot in `activecoordinates(x)` | `activecoordinates(x)[a]` for the ID, `activevalues(x)[a]` for the value |
| `r, c` already known | `x[r, c]` (unchanged) |

Verified at the audited commit for `p = 3`: `coordinates(X)[4]` is the `(2,2)`
variable and `X[4]` is the `(1,2)` entry.

`X[k]` itself should throw an `ArgumentError` that names both replacements
(OWN-3), so a missed site fails loudly with its fix in the message.

## 2.2 The five patterns

**(a) A matrix linear index against a CD coordinate.** ProximalSolvers already
reads coordinates through the view and is unchanged:

```julia
# src/cd/kron.jl:104–118, 0.5.x and 0.6 alike
cx  = coordinates(x)
old = cx[k]                      # coordinate k, not matrix entry k
...
cx[k] = new
```

**(b) Updating all independent variables, against sweeping the active set.**

```julia
# 0.5.x: looks like "scale every entry"; quadruples off-diagonal entries
for j in 1:p, i in 1:p
    x[i, j] *= 2
end

# 0.6: every independent variable, once (O(q); an absent coordinate stays absent)
θ = coordinates(x)
θ .*= 2

# 0.6: only the working set (O(s))
v = activevalues(x)
v .*= 2

# 0.6: an active-set sweep with a solver step (ACT-3, ACT-5)
act = activecoordinates(x)
for a in eachindex(act)
    coordinate_step!(model, g, x, act[a])    # never changes `act`
end
```

`src/cd/solver.jl:318` (`_activepass!`) is already this loop; `:280` (`_fullpass!`)
loops `1:ncoordinates(x)`. Neither changes.

**(c) Passing the iterate to a matrix-reading algorithm.**

```julia
# 0.5.x: the iterate is an AbstractMatrix
get_confusion_matrix(Δ, hΔ)          # Δ::AbstractMatrix, hΔ::AbstractMatrix (src/evaluation.jl:25)
eigvals(Symmetric(x))

# 0.6
get_confusion_matrix(Δ, matrixview(hΔ))
eigvals(Symmetric(matrixview(x)))    # generic read, O(p²) assembly
mul!(y, matrixview(x), v)            # O(p + s), PERF-6
```

The signature `get_confusion_matrix(::AbstractMatrix{T}, ::AbstractMatrix{T})`
is a verified break for an owner argument. Whether any caller passes one I could
not tell from the tree.

**(d) A dense writable workspace.**

```julia
# 0.5.x: loops over private storage
fill!(D, 0)
for k = 1:nnz(Δ)
    r, c = tril_cartesian(Δ.p, Δ.nzval2ind[k])
    D[r, c] = D[c, r] = Δ.nzval[k]
end

# 0.6
copyto!(D, matrixview(Δ))        # into a buffer you own
D = Matrix(x)                    # a fresh dense snapshot
W = similar(matrixview(x))       # a Matrix{T}, MAT-4
```

This is CovarianceSelection `src/directDifference.jl:125–131` (`_densify!`).

**(e) Copying or allocating another solver iterate.**

```julia
x2 = copy(x)                      # active zeros and order preserved (CPY-1)
x3 = similar(x)                   # empty (CPY-2)
copyto!(x3, x)                    # state copy (CPY-3), replaces x3's working set
copyto!(x3, M)                    # checked import of a symmetric matrix (CPY-4), replaces
```

`Base.copy!` is not part of the owner's interface; ProximalSolvers'
`_copy!(y, x) = copy!(y, x)` (`src/blocks.jl:39`) must call `copyto!` for owners.

## 2.3 `KronModel` and `KronUModel`

What must survive, with where it lives:

| item | where | 0.6 |
|:--|:--|:--|
| coordinate read and write through the vector view | `kron.jl:111–117`, `kronu.jl:73–82` | unchanged: `cx = coordinates(x)`, `old = cx[k]`, `cx[k] = new` |
| initialization over the working set | `kron.jl:65–76`, `kronu.jl:44–55` | unchanged: `activecoordinates(x)` and `cx[k]`; `size(x) == size(m.M)` needs `size(x) == (p, p)` (OWN-4) |
| multiplicity | `models.jl:420` `_multiplicity(T, r, c) = r == c ? 1 : 2`, passed as `μ` to `_coordinate_minimizer` (`models.jl:146`) | unchanged; the contract's `mₖ` and entrywise step `mₖ/a` (§2.8) are what this code computes |
| joint-direction derivative and curvature | `_symderivative`, `_symcurvature` (`kron.jl:97–101`) | unchanged: `(M[r,c] + M[c,r]) + (C[r,c] + C[c,r])` and `A[r,r]B[c,c] + A[c,c]B[r,r] + 2A[r,c]B[r,c]` |
| one-row against two-row (rank) updates | `kron.jl` `_rank1!` (diagonal), `_rank2!` (off-diagonal); `kronu.jl` `_rowaxpy!` once or twice via `_move!` | unchanged: a diagonal coordinate moves one position, an off-diagonal two |
| the dimension | `x.p` at 28 sites in `drivers.jl`, `gap.jl`, `generic_newton.jl`, `kron.jl` (`:72,106,134,139`), `kronu.jl` (`:51,70,86,91,100,120`), `models.jl`, `path.jl`, `quic.jl` | **changes:** use the promised `size(x, 1)` (OWN-4); `.p` is a field, and the contract does not make it public |
| the entry index passed to the penalty | `models.jl:419` `_entry(p, r, c) = (c-1)p + r` into `prox_entry` | unchanged (IDX-3) |

The mathematics of either model does not change; the dimension reads do (above).
What can break them is elsewhere: the `x` they are given must be an owner that supports
`coordinates`, `activecoordinates` and `size`, which the contract keeps. They must read the dimension as `size(x, 1)`, not `x.p`.

For `KronUModel` the derivative is `O(p)` and a move is `O(p)` for each of one or
two rows of `U = ΘB`; this does not depend on the owner's storage.

## 2.4 Proximal gradient, stage by stage

A workspace signature change is not a migration. The stages that touch the
iterate, from `initial_state` and `update_state!` of `ProximalGradientDescent`:

| stage | today (file:line) | what the contract changes | what to do |
|:--|:--|:--|:--|
| state type | `proximal_gradient_descent.jl:67–77`: `ProximalGradientDescentState{T, X<:…, G<:…, M, Z, P}` gives `x`, `x_previous`, `xhat` **and** `delta_x` the *same* type `X` | owners for `x`, `x_previous` and dense buffers for `xhat`, `delta_x` cannot share one parameter | **change the state type**: a second parameter for the dense trial buffers (and the blocks helpers that build them), not just the allocator. Both `initial_state` (`:94–122`) and the accelerated states are affected; the latter are unaudited |
| debias support | `proximal_gradient_descent.jl:200`: `_same_support(x, y) = all(i -> iszero(x[i]) == iszero(y[i]), eachindex(x, y))`, called at `:209` | `eachindex` and linear indexing do not exist on an owner (OWN-3) | compare **mathematically nonzero coordinates**, not stored slots: an extra stored zero in one iterate must not change the answer. Test T18 |
| workspace | `workspaces.jl:93` `_workspace_iterate(x::SymmetricSparseIterate, g) = x`; `initial_state` allocates `_dense_similar(x0) = Array{T}(undef, size(x))` (`blocks.jl:33`) and four `_similar(x0)` (`proximal_gradient_descent.jl:117–122`) | `size` is kept; `similar(x)` returns an *empty owner* | keep `x`, `x_previous` as owners; make the **trial point `xhat` and the temporaries dense** `Matrix{T}`; keep the dense gradient buffer |
| trial arithmetic | `blocks.jl:47` `_step!(y, x, g, L) = (@. y = x - g / L; y)` with `y` and `x` owners, `g` dense (`proximal_gradient_descent.jl:156`) | an owner has no broadcast (OWN-3) | `@. xhat = $matrixview(x_previous) - G / L` into the dense `xhat`. If `G` is exactly symmetric (ORC-8), `xhat[r,c]` and `xhat[c,r]` are the same floating-point operation on the same operands, so `xhat` is exactly symmetric and ORC-3's projection is a no-op |
| prox destination | `diagonal_metric.jl:102` `value_and_prox!(x, g, xhat, 1/L)` with owner destination, dense source; then `_drop!(state.x)` (`:158`) | ORC-3/ORC-4 under a scalar Frobenius step; ACT-7 for `dropzeros!` | unchanged call. Gated on **#261** for an inexact `xhat`. A diagonal metric is already rejected (`diagonal_metric.jl:23`) |
| copy of the previous point | `_copy!(state.x_previous, state.x)` (`:146`) calls `copy!` | `copy!` is not defined for an owner | `copyto!` (CPY-3) |
| stopping norm | `blocks.jl:92` `_norm_diff(x, y) = norm_diff(x, y, p)` over owners | ORC-10: a matrix reduction, Frobenius, off-diagonal counted twice | keep the call; add the test T07 that the reported step equals the Frobenius norm. `norm(coordinates(x) - coordinates(xprev))` is the wrong number (smaller by `√2` for an off-diagonal change) |
| objective | `_value!(f, x, …)` | ORC-1 | unchanged |
| refit and debias | `_refit_copy(x) = copy(x)`, `copyto!(state.x, xs)` (`:202,220`) | CPY-1, CPY-3 | unchanged |
| accelerated solvers | `blocks.jl:56` `_extrapolate_point!(y, x, xp, ω)` is the same broadcast | same break as trial arithmetic | the extrapolated point is dense; **unaudited** beyond this pattern |
| tuple blocks | CovarianceSelection `src/automaticIHT.jl:163` passes `(S, U)`, `S` an owner; `_step!(::Tuple…)` maps over blocks | each block helper is the per-array rule | the per-block helpers need the same dense-trial treatment for the `S` block; a design item for ProximalSolvers |

After this migration `prox!` still sees an ambient argument `xhat`; the only
new requirement on the solver is that `xhat` be dense and (for the guarantee of
exactness) exactly symmetric.

## 2.5 Private-field access found

Searched `src/` of each repository with `grep` for `.nzval`, `.nzval2ind`,
`.ind2nzval` and `.nnz`. Only hits whose receiver is a `SymmetricSparseIterate`
need this contract; a `SparseIterate` hit is unaffected.

| site | what it does | receiver | smallest supported replacement |
|:--|:--|:--|:--|
| ProximalSolvers `src/cd/solver.jl:562` | `copyto!(v, 1, x.nzval, 1, n)` copies the first `n` values by slot | owner or `SparseIterate` | `copyto!(v, 1, activevalues(x), 1, n)` |
| `solver.jl:574` | `copyto!(x.nzval, 1, v, 1, n)` writes values by slot | owner or `SparseIterate` | `copyto!(activevalues(x), 1, v, 1, n)` |
| `solver.jl:590` | `x.nzval[j]` inside `_restartdot` | owner or `SparseIterate` | `av = activevalues(x)`; `av[j]` |
| CovarianceSelection `src/utils.jl:61–64` (`hard_threshold!`) | `abs.(view(X.nzval, 1:nnz(X)))`, then zeroes entries below a threshold by slot, then `dropzeros!` | `SymmetricSparseIterate` | `av = activevalues(X)`; `_threshold(abs.(av), s)`; `av[i] = zero(T)` for the small ones; `dropzeros!(X)`. Writing zero keeps the coordinate active until `dropzeros!` (ACT-4, ACT-5), as now |
| CovarianceSelection `src/directDifference.jl:127–129` | `Δ.nzval2ind[k]` and `Δ.nzval[k]` into a dense matrix | owner | `copyto!(D, matrixview(Δ))`, §2.2(d) |
| ProximalSolvers `src/cd/*.jl` (28 sites, see §2.3) | `x.p` for the dimension | owner | `size(x, 1)` |
| DiffPrecTest `src/reduced_diff_estim.jl:57–159`, `inv_hessian_estimation_sym_reduced.jl:64–66` | stored arrays by slot | **`SparseIterate`** (the signatures read `Δ::SparseIterate`, `θ::SparseIterate`) | none: not an owner |
| CovarianceSelection `src/clime.jl:221–229` | `.nzval` of `R` | a `SparseMatrixCSC`, as the `colptr`/`rowval` use shows | none |
| HighDimensionalModels `src/utils.jl:306–310` | `_sparsevec` over stored arrays | `SparseIterate{T,1}` | none |

The momentum history in `solver.jl` is the one place that uses a slot as an
index; it restarts when `nactive` changes and after every full pass, which is
exactly the lifetime of ACT-3/ACT-4. DiffPrecTest's owner uses
(`diff_estimation.jl:87–125`, `util.jl:136`) read only `θ[r, c]`, two-index,
and are unchanged by the contract.

*No method is added for an unverified use.* `findnz`, `nonzeros` and a
matrix-view `nnz` are not defined for the owner (ACT-2); no audited caller
needs them.

# 3. Acceptance tests

Each row is a **named future test**; none exists. "Reference" says how the
expected value is derived *independently*, not by calling the kernel under test.
The requirement IDs are the contract's.

| test | requirements | setup | exact expectation or reference | likely home |
|:--|:--|:--|:--|:--|
| **T01 index spaces and edges** | OWN-2, OWN-4, IDX-1, IDX-2, MAT-6 | `p = 0, 1, 3` | `p = 0`: `size(X) == (0,0)`, `coordinates(X)` empty, `Matrix(X)` is `0×0`, `X[1,1]` throws `BoundsError`. `p = 1`: `X[1,1] = 2.0` gives `θ == [2.0]`, `A == [2.0;;]`; `X[2,1]`, `θ[2]` throw. `p = 3`: `X[2,2] = 7` gives `θ[4] == 7`, `A[4] == 0`, `A[5] == 7`; `θ[7]`, `A[10]`, `X[0,1]` throw `BoundsError`; `tril_cartesian(3, 4) == (2,2)`; `X[1,3]` is `X[3,1]`. Expected values by hand from the column-major order `(1,1),(2,1),(3,1),(2,2),(3,2),(3,3)` | `test/symmetric_iterate_indexing.jl` |
| **T02 one variable, two entries, live and copy** | CRD-1, MAT-1–MAT-5, CPY-5 | `p = 3`; `θ = coordinates(X)`, `A = matrixview(X)`; `X[2,1] = 3.0`; `B = Matrix(X)`; `θ[tril_index(3,2,1)] = 5.0` | `A[1,2] == A[2,1] == 3` before, `5` after; `B[1,2] == 3`; `nactive(X) == 1`; `θ == [0,5,0,0,0,0]`. A write `A[1,2] = 1.0`, and the same write through `view(A, 1:2, 1:2)`, `Symmetric(A)` and `reshape(A, 9)`, is rejected (any exception; the test does not name the type); afterwards `X[2,1] == 5` and `nactive(X) == 1`. `similar(A) isa Matrix`; `(A .+ 1)[1,2] == 6` is a writable `Matrix`; `Base.mightalias(A, θ)` is `true` | same |
| **T03 active zeros, compaction, handles** | CRD-3, ACT-3–ACT-8 | `p = 3`; `X[3,1] = 0.0`; `activate!(X, 3)`; `X[2,1] = 4.0`; `X[2,1] = 0.0`; `act = activecoordinates(X)`; sweep `θ[k] = θ[k] + 1` over `act`; then activate coordinate 6 with value `1`; then set `θ[3] = 0` and call `dropzeros!` | `X[3,1] = 0.0` on an absent coordinate leaves `nactive == 0`; after `activate!` it is `1` with value `0`; after `X[2,1] = 4.0` it is `2`, and after `X[2,1] = 0.0` still `2` (value `0`, active). The sweep leaves `act` unchanged. Writing `−0.0` to an absent coordinate does not activate it; writing `NaN` does. Build the order `k = 2 (0), 3 (5), 6 (7)`, then `dropzeros!`: `Set(activecoordinates(X)) == Set([3, 6])`, `Matrix(X)` before and after are equal, `isactive(X, 2) == false`; the *order* is not asserted and the test does not reuse a handle taken before the call | same |
| **T04 weights: local, full, explicit** | VAL-1, VAL-3–VAL-7, ORC-5, ORC-6 | `W = [0.0 1.0; 3.0 0.0]`, `λ = 0.1`, `γ = 1`, off-diagonal trial value `u = 0.5`, `p = 2`, `k = tril_index(2,2,1)` | `prox_coordinate!(X, NormL1(0.1, W), k, 0.5, 1.0)` and `check_symmetric_parameters(NormL1(0.1, W), X)` throw `ArgumentError` naming entries `(2,1) = 3` and `(1,2) = 1`. With `Symmetric(W, :L)` (weight 3): result `≈ 0.2` (`0.5 − 0.1·3·1`). With the explicit mean `[0 2; 2 0]`: `≈ 0.3`. Compare with `isapprox`, `rtol = 4eps`. **Untouched pair:** `p = 3`, `W3 = ones(3,3)` with `W3[2,1] = 3.0`: `prox_coordinate!` at `(3,3)`, `u = 0.5` returns `≈ 0.4` and does not throw; at `(2,1)` it throws; `check_symmetric_parameters` throws regardless. A symmetric `W` holding `−1` passes the symmetry check and is rejected by the penalty's domain check (VAL-1 independence) | `test/symmetric_parameters*.jl` (extend) |
| **T05 derivatives, projection, screening** | ORC-5, ORC-8, ORC-9, CRD-4, §2.8 | `p = 4`; seeded SPD `A, B`, symmetric `C`; `f(Z) = ½tr(ZAZB) + ⟨C, Z⟩` with `KronQuadratic(A, B, C)`; a symmetric `Z`; one diagonal and one off-diagonal coordinate; an ambient `g = Gs + N`, `N` antisymmetric | Reference: central finite difference of the dense formula along `Eₖ`, `h = 1e-6‖Z‖`. Expected: `(f(Z+hEₖ) − f(Z−hEₖ))/2h ≈ mₖ Gs[r,c]` (`m = 1` diagonal, `2` off) with `Gs` from `gradient!`. For a coordinate **at zero**, `kkt_violation(NormL1(λ, W), g, X, k)` equals `\|Gs[r,c]\| − λW[r,c]` for the ambient `g` and for `Gs`; for opposing infinite entries it throws. Separable scalar-step agreement: random symmetric `V`; whole `prox!` into an owner equals per-coordinate `prox_coordinate!(Z, f, k, V[r,c], γ)`; both equal the soft-threshold formula `sign(v)max(\|v\| − γλw, 0)` written in the test. Screening is **not** asserted to certify a solution here; that is T16 | `test/symmetric_oracles.jl` |
| **T06 narrowing and structural equality** | VAL-1, VAL-2, WRT | a `Float64` matrix `M` with `M[2,1] = 1.0`, `M[1,2] = 1.0 + 1e-12` (equal after `Float32`); a pre-populated `Float32` owner `X32` with an explicit zero and a nonzero; a snapshot `X0 = copy(X32)` | `copyto!(X32, M)` throws `ArgumentError`; afterwards `activecoordinates(X32) == activecoordinates(X0)`, `activevalues(X32) == activevalues(X0)` and every `X32[i,j]` is unchanged. Pairs equal in `Float64` import; `NaN`/`NaN` and `0.0`/`−0.0` pairs import; `1`/`1.0` pairs import. A symmetric matrix that violates a *function's* domain imports fine (the owner checks structure only). **Narrowing and activation (CRD-3, CPY-4):** on a `Float32` owner, `X32[2,1] = 1e-100` leaves the coordinate absent (`nactive` unchanged, value `0f0`); the same write to an *active* coordinate stores `0f0` and keeps it active; importing a `Float64` matrix whose symmetric off-diagonal pair is `1e-100` gives a working set without that coordinate, while a pair of `1e-30` (representable) is active | `test/symmetric_writes.jl` |
| **T07 CD sweep and proximal-gradient solve** | ORC-3–ORC-6, ORC-8–ORC-10, §2.8, §2.4 | **(1)** `p = 4`, symmetric `B`, `λ = 0.3`, loss `½‖Θ−B‖²_F` (a `KronQuadratic` with identity `A`, `B`); **(2)** the coupled problem `KronQuadratic(S, S, C)` with `NormL1(λ)`: `S = [2 1; 1 2]`, `C = [-0.5 3.5; 3.5 3.0]`, `λ = 0.5`, started **from zero** (with the default `C = 0` a zero start is already optimal and nothing would be exercised). Known solution `Xstar = [1 -1; -1 0]`: the matrix gradient there is `S·Xstar·S + C = [-0.5 0.5; 0.5 0.0]`, which cancels the L1 subgradient at the positive diagonal and the negative off-diagonal entries and lies strictly inside `[-λ, λ]` at the zero diagonal; `S` is positive definite, so `Xstar` is the unique solution. Both solvers and the references *must reach `Xstar`* and must perform nonzero cache updates on the way | (1) Reference: closed form `soft(B[r,c], λ)` entrywise. A cyclic CD sweep visits `k = 1..q`; after visit `k` that coordinate equals the closed form (the problem is separable), so the trajectory is asserted coordinate by coordinate. PGD converges to the same matrix to `1e-10`. (2) Reference: an **independent** `q`-vector ISTA/FISTA written in the test in orthonormal coordinates `u[k] = √m[k] θ[k]`. **The reference penalty there is `λ Σₖ √m[k] \|u[k]\|`** (since `λ Σᵢⱼ\|Θᵢⱼ\| = λ Σₖ m[k]\|θ[k]\|`), so the vector prox is a soft threshold with the per-coordinate level `γλ√m[k]`; an ordinary unweighted vector threshold would solve a different objective. Run to `1e-12`. Also a dense cyclic CD written in the test with explicit formulas, compared trajectory-for-trajectory for the fixed cyclic order. Checks: objective (relative `1e-9`); **stationarity**, not screening: the entry-normalized residual `r[k] = max(\|G[k]\| − λw[k], 0)` at a zero coordinate and `\|G[k] + λw[k] sign(x[k])\|` at a nonzero one, computed in the test from the dense gradient, with `maximum(r) ≤ 1e-8` (the test states that it checks the *maximum entry residual*; `G[r,c]` is read from the dense gradient at the matrix position of coordinate `k`; `kkt_violation` is not used for this); the cached `M = AΘB` equals `A * Matrix(X) * B` to `1e-10` after every sweep; and the PGD stopping norm: a single off-diagonal change of `0.1` reports `0.1√2`, not `0.1` | ProximalSolvers `test/` (outside this repository) |
| **T08 scaling** | PERF-1–PERF-7 | §4 | §4 | `benchmark/` plus `test/allocations.jl` |
| **T09 interface surface** | OWN-1, OWN-3, OWN-4, ACT-2 | `X = SymmetricSparseIterate(Float64, 3)` | `X isa AbstractArray == false`; `size(X) == (3,3)`, `size(X, 1) == 3`, `eltype(X) == Float64`, `ncoordinates(X) == 6`. `length(X)`, `nnz(X)`, `axes(X)`, `iterate(X)`, `sum(X)`, `X .= 1` throw (`MethodError`); `X[4]` throws `ArgumentError` whose message contains both `coordinates` and `matrixview`; `X[4] = 1.0` throws. Complex `T` throws `ArgumentError` at construction | `test/symmetric_iterate_interface.jl` |
| **T10 copies and import** | CPY-1–CPY-6 | `X` with active set `[3, 2, 6]` at values `5, 0, 7` | `copy(X)`: same values, same `activecoordinates` in the same order, explicit zero kept; writes and `activate!` on the copy leave `X` unchanged. `similar(X)`: `nactive == 0`, all zero, same `p`. `copyto!(Y, X)` makes `Y` a state copy (explicit zero kept), discards `Y`'s old set; different `p` throws `DimensionMismatch` with `Y` untouched; `copyto!(X, X)` is a no-op. `copyto!(Y, M)` with `M` nonzero at one coordinate only: `activecoordinates(Y)` has that coordinate alone, the old explicit zero is gone. `copyto!(X, matrixview(X))` throws `ArgumentError`, `X` unchanged | `test/symmetric_iterate_copies.jl` |
| **T11 aliasing** | §2.10 table, ORC-3 | one fixed symmetric input `V0`; **each comparison starts from a fresh copy of `V0`** (otherwise the second call shrinks an already-shrunk point) | `prox!(X, f, X, γ)` with `X = copy_of(V0)` equals `prox!(Y, f, V0, γ)`, for a separable `f` **and** a spectral `f` (O7 covers every function); `prox!(Y, f, matrixview(Y), γ)` from a fresh `Y` equals the same out-of-place result; a partially overlapping view is not tested as supported; `θ .= 2 .* θ` doubles `Matrix(X)` | `test/symmetric_oracles.jl` |
| **T12 `activevalues`** | ACT-4 | the set of T10 | `length(activevalues(X)) == nactive(X)`; entry `a` equals `coordinates(X)[activecoordinates(X)[a]]`; writing `0` keeps the coordinate active; `count(!iszero, activevalues(X))` equals the number of nonzero coordinates | same as T03 |
| **T13 matrix–vector product** | PERF-6, MAT-4 | seeded `X`, `v`, `y` filled with garbage | `mul!(y, matrixview(X), v) ≈ Matrix(X) * v` (relative `4eps`), `y` fully overwritten; cost in T08. **Non-finite input (PERF-6):** an owner with one stored explicit zero at `(3,1)` and `v = [Inf, 1, 0]`: `mul!(y, matrixview(X), v)` gives the same result, or throws the same `ArgumentError`, before and after `dropzeros!(X)` (the recommended default throws in both) | `test/symmetric_mul.jl` (extend) |
| **T14 matrix reductions** | ORC-10, CRD-4 | one off-diagonal `3`, `p = 3`; an *asymmetric* dense `M` | `norm(X) ≈ 3√2`; `norm(X, 1) == 6`; `norm(X, Inf) == 3`; `norm(coordinates(X)) == 3`; `norm_diff(X, Y)` equals `norm(Matrix(X) − Matrix(Y))`; `dot(M, X) == 3(M[2,1] + M[1,2])` | `test/symmetric_oracles.jl` |
| **T15 multiplicity in `Float32`** | ORC-1, numerics M2 | a `Float32` owner with one off-diagonal `3f0` | `NormL1(1.0)(X) == 6f0` and the returned type is `Float32`; mixed-precision rules are covered by `test/precision_mixed.jl` and not repeated | `test/symmetric_oracles.jl` |
| **T16 screening is not stationarity** | ORC-9, §2.8 | `p = 2`; `X[2,1] = 1.0`; `f = SqrNormL2(1.0)` (matrix gradient `G[2,1] = 1`); penalty `NormL1(1.0)` (`λw = 1`) | **Negative test:** the screen `kkt_violation(NormL1(1.0), g, X, k)` returns `0` (`\|1\| − 1`), and the independent entry residual is `r[k] = \|G[2,1] + λw·sign(x)\| = 2`. The test asserts `screen == 0` and `r[k] == 2` **only**. (The other numbers for this pair are `maximum(r) = 2`, Frobenius `2√2` and its square `8`; the test does not require a solver to report any particular one.) | `test/symmetric_oracles.jl` |
| **T17 asymmetric penalty reaches a real solver** | VAL-4, VAL-5, #142 | the real workspaces: CD (`SolverWorkspace(x, f, NormL1(0.1, W), CDSolver())`) and proximal gradient (`SolverWorkspace(x, f, NormL1(0.1, W), ProximalGradientDescent(), options)`), `p = 2`, loss `½‖Θ−B‖²_F` with `B[2,1] = 3`, `W = [0.0 1.0; 3.0 0.0]` | raw `W` throws `ArgumentError` from **both constructors, before any iterate is touched**, and is not averaged (the audited `_symmetric_penalty` would average it; this test fails there). `Symmetric(W, :L)` (weight 3) constructs and solves to `Θ[2,1] ≈ 2.7` (`3 − 0.1·3`); the explicit mean `[0 2; 2 0]` constructs and solves to `≈ 2.8`; the two differ, as they are different penalties (tolerance `1e-6`) | ProximalSolvers `test/` (outside this repository) |
| **T18 debias support at the solver boundary** | ACT-4, §2.4 | the real proximal-gradient workspace with `debias = 1`, `p = 3`, a loss with a known two-coordinate support; one backtracking step taken through `update_state!`; `x_previous` holds an extra stored explicit zero (`activate!`) that `x` does not | the supports compare equal (`state.stable` increments to `1`): support is the set of mathematically nonzero coordinates, not stored slots. With differing nonzero sets and identical stored sets, they compare unequal and `stable` resets to `0`. The state type holds owners for `x`, `x_previous` and dense buffers for `xhat`, `delta_x` | ProximalSolvers `test/` |
| **T19 allocating oracles** | ORC-11 | seeded owner `X` | `prox(f, X, γ)` is a `SymmetricSparseIterate`, `!== X`, shares no storage (writing to it leaves `X` unchanged) and equals `prox!` into a fresh owner; `gradient(f, X)` is a `Matrix` with `G == transpose(G)` exactly and equals the `Gs` of `gradient!`; `value_and_gradient(f, X)` returns `f(X)` and the same `G` | `test/symmetric_oracles.jl` |
| **T20 step scope** | VAL-5 | `V = [0 3; 3 0]`, `Y` an owner | scalar `γ = 1` works (`prox!` gives `2.5` for `NormL1(0.5)`); an array `γ`, symmetric or not, throws `ArgumentError` **before any write** to `Y`. The behavior lost with respect to the audited code (a symmetric array `γ` accepted) is listed in the release notes | `test/symmetric_oracles.jl` |
| **T21 import is not projection** | CPY-4, ORC-3, ORC-4, §2.10 | `V = [0.0 1.0; 3.0 0.0]`, `f = NormL1(0.5)`, `Y` an owner prepared with a known state (an active nonzero and an explicit zero), `γ = 1` | `copyto!(Y, V)` throws `ArgumentError` and `Y` is **unchanged** (values and active set). `prox!(Y, f, V, 1.0)` puts **`1.5`** in each off-diagonal position (the pair projects to `2`, then soft-threshold by `0.5`); `value_and_prox!(Y, f, V, 1.0)` returns **`1.5`**, the penalty `0.5·(1.5 + 1.5)` at the stored result. A kernel that still reads only the lower triangle gives `2.5`, and the test fails it. (Audited commit: `2.5`; gated on #261) | `test/symmetric_oracles.jl` |

**Requirement coverage.** OWN-1..5 T09, T01; IDX-1..3 T01, T05; IDX-4 T09
(`activecoordinates(X)[a]` is the only slot-to-ID map); CRD-1..5 T01–T03, T11;
MAT-1..6 T01, T02; CPY-1..6 T10, T02; ACT-1..8 T03, T09, T10, T12; ORC-1..10
T04–T07, T11, T14–T16, T19, T21; ORC-9 screening T05, stationarity T07, T16; VAL-1..8
T04, T06, T17, T20 (VAL-6 and VAL-7 by T04's untouched pair and an extension
function in the same file); PERF-1..8 T08, T13.
PERF-8 is a non-promise and has no test.

# 4. Performance protocol (a future measurement, not a result)

No timing was run for this document. The protocol:

- **Environment to record:** Julia version and `JULIA_NUM_THREADS`, CPU model,
  OS, ProximalBase commit, `--check-bounds` setting. The benchmark harness is
  `benchmark/` (see its README); `BenchmarkTools` is not installed in the
  default environment, so use the harness's own timing.
- **Warm-up:** call each case once untimed, then `GC.gc()`; allocation counts
  come from `@allocated` on the second call; time the minimum of at least 30
  repetitions of a batch.
- **Timed first activation.** Warm up on a *different* coordinate or fixture.
  Every timed first activation must begin with an **absent** coordinate;
  otherwise repeated samples measure the already-active path. Restore the state
  (`dropzeros!` of a stored zero, or a fresh owner) *outside* the timed region.
- **A real coordinate sweep.** Include a warmed-up active sweep that calls
  `prox_coordinate!` with a built-in separable `f`, not only a trivial
  assignment, and a `KronModel` sweep for the cached case.
- **Pre-refactor baseline.** Report the **0.5.x `SymmetricSparseIterate`**
  (`abaceda25`) beside the new owner for every case above that both support,
  in the same session. The CSC and dense baselines answer the matrix-product
  question; only the pre-refactor owner answers whether existing coordinate
  descent became slower.
- **Fixed `s = 100`, `p ∈ {500, 1000, 2000, 4000}`.** For each, record:
  construction; `X[i,j]` read and write (cycling through a precomputed list of
  active and absent positions); first activation of an absent coordinate;
  an active sweep `for k in activecoordinates(X)` with a trivial update;
  `mul!(y, matrixview(X), v)`; `Matrix(X)`.
- **Fixed `p = 1000`, `s ∈ {10², 10³, 10⁴, 10⁵}`.** Record the same, with the
  sweep and `mul!` as the cases of interest.
- **Targets (future, from PERF):**

| case | as `p` grows, `s` fixed | as `s` grows, `p` fixed | allocation after warm-up |
|:--|:--|:--|:--|
| coordinate or pair read/write, `isactive` | flat | flat | 0 B |
| first activation | flat | flat | 0 B |
| active sweep | flat | linear in `s` | 0 B |
| `mul!(y, A, v)` | linear in `p` | linear in `s` | 0 B |
| `Matrix(X)` | quadratic in `p` | linear in `s` | one `p²` array |
| construction | quadratic in `p` | — | `O(q)` |

- **Baselines to report beside the owner:** a `SparseMatrixCSC` product of the
  same `s`, and a dense `Symmetric` product at `p ≤ 2000`.
- **What the benchmark can and cannot say.** A flat curve over a factor of 8 in
  `p` is a regression alarm, not a proof of worst-case `O(1)`. The proof is the
  code: a direct table of length `q` and no growth, which is a review item (the
  audited code has it; the contract does not name it).

# 5. Review gate

1. *Which interface do I index?* `X[i,j]` for a matrix position; `coordinates(X)[k]`
   for a variable by ID; `activecoordinates(X)[a]` for an ID by slot; `matrixview(X)[i]`
   for a matrix linear index (§2.1, IDX-1..4).
2. *Does a write change the working set?* A nonzero write to an absent coordinate
   activates it; a zero write to an absent one does not; a write to an active
   one never removes it; only `dropzeros!` removes (CRD-3, ACT-5, ACT-6). T03.
3. *Where does the factor of two enter?* In the objective, never in storage:
   multiplicity `mₖ` multiplies the penalty and the quadratic along `Eₖ`; a
   coordinate model returns joint-direction derivative and curvature; the
   entrywise step is `mₖ/a`; the KKT residual is entry-normalized (contract
   §2.8). T05, T07, T14.
4. *What is actually checked?* Exact pairwise symmetry of caller data, on the
   source values, before conversion; a coordinate call checks its pair, a bulk
   write checks everything it will write before writing; the whole model is
   certified only by `check_symmetric_parameters` on the caller's original
   penalty (VAL-1..8, T04, T06, T17). `kkt_violation` screens a coordinate at
   zero; it certifies no solution (T16).
5. *Can a dense scan or allocation occur?* The cases in PERF-1..7 do not;
   generic matrix operations on the view may (`PERF-8`) and the list of those
   promised sparse-fast is closed.

*Does every important guarantee have an acceptance case?* Yes, by the coverage
line in §3, with two exceptions that are deliberate: PERF-8 (a non-promise) and
the "unsupported, tracked elsewhere" families of ORC (#336, #342), which are
tested by their own issues. *Does the design remove the need for a catalog of
forbidden Julia mutators?* It removes the catalog for the **owner** (nothing
generic reaches it, OWN-1/OWN-3) and for the **view** (read-only, MAT-2). It
does not and cannot protect an explicit double loop over `(i, j)` on `X[i,j]`
that reads and writes a mirrored variable, as the contract says.

# 6. Result of the documentation build

`julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'`
exited 0, with only the usual local warnings (the size thresholds of
`api/functions.md` and the search index, and skipped deployment). The two
documents are under `docs/design/`, outside `docs/src`, so Documenter does not
build them and the build is **not evidence about them**. Their cross-links
and the file-and-line links to the audited commits were checked by hand, and
the line numbers were read from the checkouts named in §1. Before the review
I also ran the fixtures of T04 (the weight cases) and the finite-difference
and `dot`/`norm` expectations of T05 and T14 against the audited commit's
current API; they gave `0.2`, `0.3`, `0.4`, an `ArgumentError` for the raw and
the touched asymmetric pair, finite differences within `1e-8` of `m·Gs[r,c]`,
and `dot(M, X) == 21`.

# 7. Not done here (separate tasks)

A prototype of `matrixview` and `activevalues`; an implementation plan; changes
to `src/`, tests or benchmarks; downstream migration; the 0.6 release; and
issues #261, #336, #342, #366 and #374, which the contract mentions but does not
solve.
