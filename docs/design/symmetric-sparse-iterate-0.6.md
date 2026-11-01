# SymmetricSparseIterate in ProximalBase 0.6: the contract

> **Draft for maintainer review — proposed 0.6 behavior, not a description of an
> installed release.** Nothing here is implemented. This file is the only
> normative document of the pair; the companion
> [migration and acceptance guide](symmetric-sparse-iterate-0.6-acceptance.md)
> links back to it and restates none of its rules.

**Audited commits.** ProximalBase `abaceda25442bf6cc61601c8c1d3b2a16e1a3492`
(master, 2026-11-01; `Project.toml` says 0.5.4). Links below point at that commit.
ProximalSolvers `3893fb9` (master), read as evidence only. The other consumers
and what was not audited are listed in the companion guide.

**How to read it.** Part 1 is for a Julia user and needs no implementation
knowledge. Part 2 is for a package or solver author: requirements carry stable
IDs, *must* marks a public guarantee, and *recommended* marks an implementation
idea that a conforming implementation may replace. Part 3 lists the choices the
maintainer must sign off. Part 4 is informative evidence.

---

# Part 1. The idea in one page

A real symmetric matrix `X` of size p × p has p(p+1)/2 independent numbers: the
lower triangle. `X[2, 1]` and `X[1, 2]` are *one variable*, not two. An
optimization solver wants to see those p(p+1)/2 variables. A matrix routine
wants to see a p × p matrix. If one object offers both faces as a writable
`AbstractMatrix`, an innocent loop can touch one variable twice:

```julia
for j in 1:p, i in 1:p
    X[i, j] *= 2          # 0.5.x: X[2,1] is doubled twice, so it is quadrupled
end
```

(Run at the audited commit: an off-diagonal `1.0` becomes `4.0`.) No list of
patched `Base` functions can make every such loop right. So `SymmetricSparseIterate`
in 0.6 is **not a matrix and not a vector**. It *owns* the variables and hands
out two views of that one state, plus scalar access by two indices:

```julia
X = SymmetricSparseIterate(Float64, 3)
θ = coordinates(X)                   # live and writable; six independent variables
A = matrixview(X)                    # live and read-only; a 3 × 3 matrix

X[2, 1] = 3.0
@assert A[1, 2] == A[2, 1] == 3.0
@assert nactive(X) == 1
@assert sum(abs, θ) == 3.0
@assert NormL1(1.0)(X) == 6.0       # the matrix penalty counts both positions

B = Matrix(X)                        # an independent dense copy
θ[tril_index(3, 2, 1)] = 5.0
@assert A[1, 2] == 5.0 && B[1, 2] == 3.0
```

Why each assertion holds:

- `X[2, 1] = 3.0` writes one variable. Reading `A[1, 2]` or `A[2, 1]` sees it,
  because `A` is a view of `X`, not a copy.
- `nactive(X) == 1`: the working set holds one *coordinate* (one variable), not
  two matrix positions. A zero is not stored; a nonzero write activates its
  coordinate.
- `sum(abs, θ) == 3.0`: `θ` is an ordinary vector of the six variables, so an
  ordinary vector function sees `3.0` once.
- `NormL1(1.0)(X) == 6.0`: the penalty is a function of the *matrix*, whose two
  positions both hold `3.0`. Its argument is the owner, so it uses the matrix
  meaning. (`NormL1(1.0)(θ)` is `3.0`: a vector argument gets the vector
  meaning. The two are different functions on purpose.)
- `Matrix(X)` copies. After `θ[2] = 5.0` the copy `B` still says `3.0` while the
  live view `A` says `5.0`. (`tril_index(3, 2, 1)` is `2`.)

## Three different indices

| index | range | names | durable? |
|:--|:--|:--|:--|
| matrix linear index `i` | `1:p²` | the column-major position of `matrixview(X)[i]` | yes: a mathematical location |
| coordinate ID `k` | `1:q`, `q = p(p+1)/2` | the variable `coordinates(X)[k]`, `k = tril_index(p, r, c)` with `r ≥ c` | yes: a mathematical location |
| active slot `a` | `1:s`, `s = nactive(X)` | the `a`-th entry of the current `activecoordinates(X)` | **no**: a storage position |

Only the first two describe where something is. A slot must never be kept as an
ID, because `dropzeros!` moves coordinates between slots. Confusing the first two
changes the answer. For p = 3 the lower triangle in column-major order is
`(1,1), (2,1), (3,1), (2,2), (3,2), (3,3)`, so

```julia
coordinates(X)[4]    # the variable at (2, 2)
matrixview(X)[4]     # the matrix entry at (1, 2): linear index 4 is row 1, column 2
```

(Both checked at the audited commit, where `X[4]` is the matrix reading.)

## Where the factor of two lives

In a matrix objective an off-diagonal variable stands for two positions. That
factor of two belongs to the *objective* (the Frobenius inner product), not to
the storage. A vector of coordinates has no factor of two, so `coordinates(X)`
is correct for anything linear: you may double every coordinate and double the
matrix. It is *wrong* for matrix norms: `norm(θ)` is `3.0` for the example,
`norm(X)` is `3√2`. A coordinate solver therefore never rescales storage; it
asks the oracles, which know the multiplicity, and it measures curvature along
the matrix direction (§2.8).

## What changes for you

Iterate with `coordinates(X)` or `activecoordinates(X)`. Pass the owner to
functions that mean the matrix. Pass `matrixview(X)` to code that reads a
matrix, and `Matrix(X)` to code that needs a dense writable one. There is no
`X[k]`, no `length(X)`, and no broadcasting on the owner. The companion guide
walks through real callers.

---

# Part 2. Requirements

**Notation.** `p` is the dimension, `q = p(p+1)/2`, `s = nactive(X)`. `X` is an
owner, `θ = coordinates(X)`, `A = matrixview(X)`. `T = eltype(X)` is a real
fixed-width type. `m_k` is the *multiplicity* of coordinate `k`: 1 on the
diagonal, 2 off it.

## 2.1 The owner

| ID | Requirement |
|:--|:--|
| OWN-1 | `SymmetricSparseIterate{T}` *must* keep its name and its constructors `SymmetricSparseIterate(T, p)` and `SymmetricSparseIterate(p)` (`Float64`). It *must not* subtype `AbstractArray`, `AbstractMatrix` or `AbstractVector`. |
| OWN-2 | `X[i, j]` and `X[i, j] = v` *must* be supported for integers `1 ≤ i, j ≤ p`, from either triangle, bounds-checked, naming one variable. Out of range *must* throw `BoundsError`. |
| OWN-3 | `X[k]`, `X[k] = v`, `length(X)`, `eachindex`, `axes`, iteration, `firstindex`/`lastindex`, `map`, `reduce` and broadcasting on the owner *must not* return values. `X[k]` *should* throw an `ArgumentError` that names `coordinates(X)[k]` and `matrixview(X)[k]`; the others may be a `MethodError`. |
| OWN-4 | `size(X)` *must* be `(p, p)` and `size(X, d)` *must* be `p` for `d ∈ {1, 2}`. `eltype(X)` *must* be `T`. `ncoordinates(X)` *must* be `q`. |
| OWN-5 | Complex `T` *must* throw an `ArgumentError` at construction, as today. |

*Recommended default for `length`:* undefined. *Reason:* the two natural answers
(p² and q) are both "right" for some caller, and #22 shows what a wrong one costs.
*Consequence for a caller:* `length(X)` is a `MethodError`; use `ncoordinates(X)`
(q), `size(X, 1)` (p) or `length(A)` (p²).
*Recommended default for `size`:* keep `(p, p)`. *Reason:* the dimension is
unambiguous, and ProximalSolvers checks it as the shape of a model
(`initialize!` of `KronModel`, src/cd/kron.jl:65). *Consequence:* `size` does
not make the owner an array; no generic array code applies to it.

With no `AbstractArray` supertype, `Base` and `LinearAlgebra` have no generic
method that writes the owner. The only writes are the methods named in this
document. That is what removes the need for a catalog of forbidden mutators
(compare the `SymmetricSparseIterate` override list in
[`src/sparse_iterate.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/abaceda25442bf6cc61601c8c1d3b2a16e1a3492/src/sparse_iterate.jl#L896)
and the 1,130 lines of
`test/symmetric_mutators.jl`, `test/symmetric_mutator_errors.jl` and
`test/symmetric_writes.jl`).

**What this does not make safe.** An explicit double loop over `(i, j)` that
*reads and writes* `X[i, j]` still updates a mirrored variable twice. The
owner cannot prevent that and does not claim to. What it removes is the
*implicit* double loop that a `Base` function runs on your behalf.

## 2.2 Index spaces

| ID | Requirement |
|:--|:--|
| IDX-1 | The coordinate ID of the lower-triangle position `(r, c)`, `r ≥ c`, *must* be `tril_index(p, r, c)`; IDs *must* follow column-major order of the lower triangle, as today. `tril_cartesian(p, k)` *must* invert it. |
| IDX-2 | `X[i, j]` with `i < j` *must* denote the variable of `(j, i)`. |
| IDX-3 | `prox_entry(f, v, i, γ)` takes a *matrix* linear index `i = (c − 1)p + r` (it indexes weights and bounds, which are matrices). This is the one place an oracle takes a matrix linear index; every coordinate operation takes an ID `k`. Docstrings *must* say which. |
| IDX-4 | An integer passed to a coordinate API (`coordinates(X)[k]`, `isactive`, `activate!`, `prox_coordinate!`, `kkt_violation`) is **always** interpreted as an ID. A caller holding a slot `a` writes `activecoordinates(X)[a]`. When a slot and a valid ID are the same `Int` the implementation cannot tell the intent, and no new index type is introduced to do so. |

## 2.3 The coordinate view

| ID | Requirement |
|:--|:--|
| CRD-1 | `coordinates(X)` *must* return an `AbstractVector{T}` of length `q`, live (reads and writes reach `X`), writable, `O(1)` to construct, with no copy of data. |
| CRD-2 | Reading an absent coordinate *must* return `zero(T)` and *must not* activate it. |
| CRD-3 | A write first converts `v` to `T`; activation is decided from the **converted stored value**. If `iszero(convert(T, v))` and the coordinate is *absent*, it *must not* be activated: writing `1e-100` to an absent `Float32` coordinate leaves it absent (the audited code already does this). A nonzero converted value (NaN included) *must* activate it. Writing to an *active* coordinate *must* store the converted value, including zero, and *must not* deactivate it. Symmetry of a *mirrored pair* is a different question and is decided on the source values before conversion (VAL-2). |
| CRD-4 | `θ` *must* be an ordinary vector: `sum(abs, θ)`, `norm(θ)`, `θ .= …`, `fill!(θ, 0)`, `rmul!(θ, 2)` mean what they mean for any vector of `q` numbers. In particular a full traversal of `θ` visits `q` coordinates, not `s`. Linear operations on `θ` are therefore correct for the matrix (doubling every coordinate doubles the matrix); norms and inner products are *not* matrix norms (§2.8). |
| CRD-5 | Broadcasting `θ .= g.(…)` *must* obey CRD-3 per coordinate. It *must* give the same result as the explicit loop `for k in 1:q; θ[k] = g(…); end`. |

## 2.4 The matrix view

| ID | Requirement |
|:--|:--|
| MAT-1 | `matrixview(X)` *must* return an `AbstractMatrix{T}` of size `(p, p)` with ordinary matrix indexing (linear and Cartesian) and both triangles, `O(1)` to construct, with no copy of data, live. |
| MAT-2 | `A` *must* be read-only: no mutation through `A` *or any wrapper derived from it* (`view`, `reshape`, `transpose`, `Symmetric(A)`, `vec`). *Recommended:* define no `setindex!` for the view, so the derived wrappers inherit the refusal. What is required is that the write is **rejected and `X` is unchanged**; the exception type is not specified, and no override should be written only to produce a particular exception for an exotic wrapper. |
| MAT-3 | Read-only is not a snapshot and not an immutable owner: `A` *must* reflect later writes to `X`, and `X` stays mutable. |
| MAT-4 | A generic allocating operation (`A .+ 1`, `A * v`, `similar(A)`, `copy(A)`, `collect(A)`) *must* give an ordinary writable dense result and *must not* give a route to `X`. `similar(A)` *must* be a `Matrix{T}`. |
| MAT-5 | `Base.dataids(A)` *should* be `Base.dataids` of the owner's storage so that `Base.mightalias` and `Base.unalias` see that `A` and `X` share memory. |
| MAT-6 | `A` *must* satisfy the `AbstractArray` interface for reading: `A[i] == A[CartesianIndices(A)[i]]`, `length(A) == p²`, iteration yields `p²` entries. (This is #22's rule, kept for the view and removed from the owner.) |

`Matrix(X)` (and `Array(X)`) *must* return a fresh dense symmetric `Matrix{T}`,
`O(p² + s)`, independent of `X`.

## 2.5 Copies and whole-value writes

Three operations that look alike are different: a *state copy* (the working set
travels), a *checked import* (a matrix becomes the value), and a *dense
snapshot* (`Matrix(X)`).

| ID | Operation | Requirement |
|:--|:--|:--|
| CPY-1 | `copy(X)` | *must* create independent solver state: same `p`, same values, same set of active coordinates, **active zeros preserved**, and the activation order of `X`. `O(q)`. |
| CPY-2 | `similar(X)`, `similar(X, S)` | *must* return an empty owner (no active coordinates, all values zero) of the same `p`, with element type `T` or `S`. `O(q)` memory. |
| CPY-3 | `copyto!(X, Y::SymmetricSparseIterate)` | *must* make `X` a state copy of `Y` (CPY-1), discarding the previous working set of `X`. Different `p` *must* throw `DimensionMismatch` before writing. `copyto!(X, X)` *must* be a no-op. |
| CPY-4 | `copyto!(X, M::AbstractMatrix)` and `SymmetricSparseIterate(M)` | *must* be a checked import (§2.10): `M` must be square of size `(p, p)` and exactly symmetric. **Default: replace.** After the call `X` equals the converted `M`, the working set is exactly the coordinates whose **converted stored value** is nonzero, in increasing ID order (CRD-3 rule), and the previous working set (explicit zeros included) is gone. Symmetry is checked on the source values before conversion (VAL-2). `O(q)` time (a dense source is read once). |
| CPY-5 | `Matrix(X)` | dense, independent, writable snapshot (§2.4). |
| CPY-6 | an aliasing source | A source that shares memory with `X` (`matrixview(X)` or a derived view) *must* be rejected with an `ArgumentError` before writing. |

*Default for CPY-4:* replace. *Reason:* the result then depends only on the
argument, not on what `X` held before, so a warm start from a dense matrix has
the support of that matrix. *Consequence for a caller:* a solver that wants to
*keep* its working set while loading values writes `θ .= …` (CRD-5) or loops
over `activecoordinates`; it does not import. An import never leaves an active
coordinate that the new matrix does not make nonzero.

## 2.6 Stored coordinates and the active set

Three counts must not be confused:

| quantity | meaning | how to get it |
|:--|:--|:--|
| stored coordinates | the working set, explicit zeros included | `nactive(X)` (`= s`) |
| mathematically nonzero coordinates | those with value ≠ 0 | `count(!iszero, activevalues(X))` (`O(s)`) |
| nonzero full-matrix positions | both triangles | `count(!iszero, A)` (`O(p²)`, generic) |

| ID | Requirement |
|:--|:--|
| ACT-1 | `ncoordinates`, `nactive`, `activecoordinates`, `isactive`, `activate!` and `dropzeros!` *must* be kept with today's meanings, on IDs. `isactive(X, k)` is `O(1)` and *must not* allocate. |
| ACT-2 | **`nnz(X)` is not retained.** It *must* be undefined for the owner. `nactive(X)` replaces it. A matrix-view `nnz` is *not* added: no audited caller needs one. `SparseArrays.sparse(A)` is the way to ask a matrix question (*recommended:* `O(p + s)`). |
| ACT-3 | `activecoordinates(X)` *must* return a read-only `AbstractVector{Int}` of the IDs of the active coordinates in slot order, of length `s` at the call. It is a **handle, not a snapshot**: its entries are valid IDs, in unchanged slots, for as long as the active set is unchanged. It *must* be reacquired after any activation or compaction. Use of a stale handle after `dropzeros!` is not specified (the slots may have moved) but *must not* be memory-unsafe. |
| ACT-4 | `activevalues(X)` (**new name**) *must* return a writable `AbstractVector{T}` of length `s` whose entry `a` is the value of coordinate `activecoordinates(X)[a]`. Writing zero through it keeps the coordinate active. Same lifetime as ACT-3. It is the supported replacement for reading and writing the first `nnz` stored values by slot. |
| ACT-5 | Writing a value (CRD-3, `X[i,j] = v`, `activevalues`, `prox_coordinate!`) *must never remove* an active coordinate. Only `dropzeros!` removes a single coordinate, and only one whose value is zero. The one exception is **whole-state replacement**: `copyto!(X, Y)` and `copyto!(X, M)` (CPY-3, CPY-4) discard the previous working set by design, and `copy` and state copies *preserve* active zeros. A sweep over `activecoordinates(X)` that only updates already active coordinates *must* leave `activecoordinates(X)` unchanged, whatever values it writes. |
| ACT-6 | `activate!(X, k)` is the explicit-zero operation: it makes `k` active at value zero if it is absent and changes nothing otherwise. After `X[3,1] = 0.0` (absent) `nactive` is unchanged; after `activate!(X, tril_index(3,3,1))` it is one larger. |
| ACT-7 | `dropzeros!(X)` *must* keep every value, every owner/view relation and every ID unchanged. It *may* reorder slots (today: the last active coordinate moves into each freed slot). It *must* run in `O(s)` and not allocate. |
| ACT-8 | What survives compaction: `X`, `θ` and `A` keep denoting the same variables and values; coordinate IDs do not change; slot order and any handle from ACT-3/ACT-4 do not. |

*Recommended extra accessor:* none. A caller that needs `(k, value)` pairs
zips `activecoordinates(X)` and `activevalues(X)`.

## 2.7 Cost of compaction and handles: why a handle and not a snapshot

A snapshot of the active IDs would cost `O(s)` per sweep and an allocation, which
the active-set solvers avoid. ProximalSolvers' momentum history already keys by
*slot* and restarts when the count changes, because `dropzeros!` reorders
([src/cd/solver.jl:190–200](https://github.com/mlakolar/ProximalSolvers.jl/blob/3893fb9/src/cd/solver.jl#L190-L200)).
ACT-3 states that rule as part of the contract instead of leaving it a comment
in one solver.

## 2.8 The mathematics, independent of storage

**The object.** The variable is a real symmetric matrix `Z` with the Frobenius
inner product `⟨U, V⟩ = Σᵢⱼ UᵢⱼVᵢⱼ`. In coordinates

```
‖Z‖²_F = Σₖ mₖ θₖ²,      mₖ = 1 (diagonal), 2 (off-diagonal).
```

Coordinates are *not* rescaled by √2: `X[r, c]` is the matrix entry. A
function of `coordinates(X)` that is an ordinary vector function stays one
(CRD-4). The orthonormal-coordinate alternative `√mₖ θₖ` would make vector
functions right but would make `X[2,1] ≠ coordinates(X)[2]`, so every stored
number would need a conversion at each read.

**A direction.** Moving coordinate `k` by `t` moves `Z` by `t Eₖ`, with `Eₖ = e_r e_cᵀ + e_c e_rᵀ` (`Eₖ = e_r e_rᵀ` on the diagonal), and `‖Eₖ‖²_F = mₖ`.

**Smooth example.** Let `f(Z) = ½‖Z − B‖²_F` with symmetric `B`, `B[2,1] = 3`, p = 2, and `Z[2,1] = t`, the rest zero. Then `f` along the coordinate is `(t − 3)² + const`.

- The *matrix gradient* projected on the symmetric subspace is `G = Z − B`, with `G[2,1] = t − 3`.
- The *raw directional derivative* along coordinate `k` is `⟨G, Eₖ⟩ = mₖ G[2,1] = 2(t − 3)`.
- The *curvature along the joint direction* is `⟨Eₖ, H Eₖ⟩ = 2` (here `H = I`).
- For an *ambient* (not symmetric) gradient `Ĝ` the same directional derivative is `Ĝ[2,1] + Ĝ[1,2]`. If `Ĝ[2,1] = 2` and `Ĝ[1,2] = 0` it is `2`, the same as for its symmetric part (`1` in each position, times `m = 2`). A solver that read `Ĝ[2,1] = 2` as "the entry" and then multiplied by `m` would get `4`.

**Convention for a coordinate model** (the one the audited solvers already
follow, `coordinate_derivatives` and `_symcurvature` in
[src/cd/kron.jl:97–101](https://github.com/mlakolar/ProximalSolvers.jl/blob/3893fb9/src/cd/kron.jl#L97-L101)):
it returns the derivative and curvature *along `Eₖ`*, already combined over the
two positions. A solver *must not* multiply either of them by 2 again.

**Proximal example, with the check you can do by hand.** Minimize
`½‖Z − B‖²_F + λ Σᵢⱼ |Zᵢⱼ|` with symmetric `B`, `B[r,c] = 3`, `λ = 0.5`. On the
coordinate the objective is `(t − 3)² + 2λ|t|`; the doubled quadratic and the
doubled penalty cancel and the minimizer is `3 − λ = 2.5`, not `2.0`. At the
audited commit, `prox!` into an owner, `prox_entry(NormL1(0.5), 3.0, 2, 1.0)`,
`prox_coordinate!(X, NormL1(0.5), k, 3.0, 1.0)` and a joint-direction coordinate
step all give `2.5`.

The step that the entrywise prox sees depends on how the curvature is
normalized. Let `a` be the curvature along `Eₖ` and `a_u = a / mₖ` the curvature
along the *unit-Frobenius* direction `Eₖ/√mₖ`. The scalar problem
`d(t − θ) + (a/2)(t − θ)² + mₖ gᵢ(t)` is minimized at

```
θ⁺ = prox_{(mₖ/a) gᵢ}( θ − d/a ),     entrywise step  mₖ/a = 1/a_u.
```

So **the step handed to `prox_entry`, and the `γ` of the four-argument
`prox_coordinate!`, is the reciprocal of the curvature measured along a
unit-Frobenius direction.** For the example `a = 2`, `mₖ = 2`: step `1`, giving
`2.5`. Using step `1/a = 0.5` (curvature normalized but the penalty not
doubled) gives `2.75`; using step `2` (penalty doubled, curvature entry-normalized
as `1`) gives `2.0`. Both are wrong, silently.

**Screening is not stationarity.** `kkt_violation` answers one question: does
coordinate `k`, *at zero*, violate optimality, so that it should enter the
working set? Its maximum over all coordinates does **not** certify a solution.
Counterexample: at a coordinate with value `x > 0`, projected matrix gradient
`G = +1` and `λw = 1`, the screen is `|1| − 1 = 0`, yet the entry-normalized
derivative of the objective is `G + λw = +2` and the point is not stationary.
The public screening API is kept. The residual of a *returned solution* is a
separate quantity, entry-normalized like the screen:

```
r_k = max(|G_k| − λ w_k, 0)        if x_k = 0
r_k = |G_k + λ w_k sign(x_k)|      if x_k ≠ 0          (weighted L1 penalty)
```

and the Frobenius-weighted total is `Σₖ mₖ r_k²`. These are different numbers: for one off-diagonal pair with `r_k = 2`,
`maximum(r) = 2`, the Frobenius residual `√(Σ mₖ r_k²)` is `2√2`, and its square
is `8`. A solver *states which one it reports*; the stopping policy is the
solver's. Here `G_k` is `G[r, c]` for the lower-triangle coordinate ID `k`
(`k` is not a matrix linear index).

**Public screening API.** `kkt_violation(f, g, X, k)` takes the *matrix gradient*
`g` (its symmetric part is used) and returns the residual in the *entry*
normalization: `|G[r,c]| − λ w[r,c]` for `NormL1`. It divides out `mₖ`, because
both the derivative and the penalty carry it. A solver that holds the
coordinate derivative `dₖ` *must* pass `dₖ / mₖ` (or the matrix gradient),
never `mₖ dₖ`. Optimality of the whole problem in the Frobenius norm weights
residuals by `mₖ`: `Σₖ mₖ rₖ²`.

## 2.9 The oracles

*Which argument decides the space.* A **value** or **gradient** oracle follows
its point argument; a **proximal** oracle follows its **destination**. An owner
(as point or destination) means *real symmetric matrices with the Frobenius
inner product and both positions counted*. A plain matrix means ordinary
matrices. A coordinate vector (`θ`) means ordinary vectors. A function that
accepts `AbstractMatrix` is **not** thereby a symmetric-constrained proximal
operator: only the rows below, for the families they name, are promised.

| ID | Oracle | Space and meaning | Destination; aliasing | Returned scalar |
|:--|:--|:--|:--|:--|
| ORC-1 | `f(X)` | `f` of the matrix `X`, both positions | none | value *at the input* `X` |
| ORC-2 | `f(θ)`, `f(A)` | ordinary function of a vector, of a matrix | none | value at the input |
| ORC-3 | `prox!(Y, f, V, γ)`, `Y` an owner, `V` an owner or matrix | the minimizer over symmetric `Z` of `f(Z) + ‖Z − V‖²_F / (2γ)` | `Y` is the destination. `V === Y` is supported for **every** function that supports the proximal operation, as numerics O7 promises (recovery paths read no entry they have already overwritten); the unsupported case is a *partial overlap* of `V` and `Y` (a shifted view, #318), not a family of functions. A `V` sharing memory with `Y` through `matrixview(Y)` is the same alias as `Y`. | `Y` |
| ORC-4 | `value_and_prox!(Y, f, V, γ)` | as ORC-3 | as ORC-3 | `f(Y)`, the value *at the result* |
| ORC-5 | `prox_coordinate!(X, f, k, γ)` | one coordinate; the scalar `prox_entry(f, θₖ, i(r,c), γ)` with the other coordinates fixed. The multiplicity cancels: `mₖ` multiplies both `f` and the quadratic. | writes `θₖ` in place (CRD-3 for zeros) | the new `θₖ` |
| ORC-6 | `prox_coordinate!(X, f, k, u, γ)` | as ORC-5 for the trial value `u`, which is *not* written first | writes `θₖ`; a coordinate that stays zero stays absent | the new `θₖ` |
| ORC-7 | `prox_entry(f, v, i, γ)` | raw scalar prox of the entry term at matrix linear index `i` (IDX-3) | none; checks no parameters | the scalar result |
| ORC-8 | `gradient!(G, f, X)`, `value_and_gradient!` | the gradient restricted to symmetric matrices, i.e. `Gs` with `⟨Gs, H⟩ = df(X)[H]` for symmetric `H` | `G` a dense matrix or an owner, `G !== X`; the oracle writes a symmetric result | `value_and_gradient!`: `f(X)` at the input |
| ORC-9 | `kkt_violation(f, g, X, k)` | a **screen at zero** (§2.8): whether coordinate `k`, taken at zero, should move; entry normalization; `g` an ambient or symmetric matrix | none | the screening violation for coordinate `k`. It is **not** a stationarity residual of a returned solution |
| ORC-11 | `prox(f, X, γ)`, `gradient(f, X)`, `value_and_gradient(f, X)` | the allocating forms of ORC-3 and ORC-8. Their present `AbstractArray` dispatch no longer applies to an owner, so each *must* be defined explicitly for the supported families | `prox` returns an **independent owner** (never a view of `X`); `gradient` returns a **dense, exactly symmetric `Matrix`** | as the in-place forms |
| ORC-10 | `norm(X, p)`, `norm_diff(X, Y, p)`, `dot(M, X)` | matrix reductions: `‖·‖₁` counts both triangles, `‖·‖₂` is Frobenius | none | reduction of the matrices; *recommended* `O(s)` for owners |

**Supported families.** Separable penalties and losses with symmetric
per-entry parameters; the spectral and `LogDetLoss`/`KronQuadratic` families
where today's code computes them on a symmetric matrix. **Not designed here,
tracked elsewhere:** group penalties, cones, `SliceSum`/`FiberSum` on an owner
(#342); `SphereConstrained{NormL1}` (#336, wrong today); array steps and
diagonal metrics (a **scope choice** of this redesign, not a mathematical claim:
see VAL-5); `restrict` and Moreau envelopes of
nonseparable functions. A caller that needs one of these gets an error, not a
guess.

## 2.10 Validation and numerical boundaries

**Three operations, not one.**

1. **Storing a matrix** (import, constructor, `copyto!` from a matrix): the
   value must be exactly symmetric. Copying an asymmetric matrix is an error;
   it is not a projection.
2. **Declaring model parameters** (weights, bounds, labels, centers, per-entry
   steps, normals, `c` of a linear term): mirrored entries must agree exactly.
   There is no silent averaging and no preferred triangle (#142, amendment of
   2026-10-30). `Symmetric(W, :L)` is the caller's explicit choice of a
   triangle; it is not evidence that the original array was symmetric.
3. **Projecting an ambient quantity inside an oracle.** The gradient `g` given to
   `kkt_violation`, and the gradient an oracle writes, are projected onto the
   symmetric subspace. So is the *argument* `V` of a proximal map **under a
   scalar Frobenius step `γ`**: for symmetric `Z`,
   `‖Z − V‖²_F = ‖Z − (V + Vᵀ)/2‖²_F + ‖(V − Vᵀ)/2‖²_F`, and the last term does
   not depend on `Z`. The identity needs both the Frobenius norm and a single
   scalar step. It does *not* extend by analogy to array steps, to other metrics
   or to wrappers. **Implementation status:** #261 is open. At the audited commit
   `prox!(Y, NormL1(0.5), [0 1; 3 0], 1.0)` into an owner returns `2.5` for the
   pair (the lower-triangle entry `3`, minus `0.5`), while `prox!(Y, SqrNormL2(1.0), …)`
   with the same argument throws. The contract's rule (project, with the symmetric
   part `2` giving `1.5` for the first) is not yet true of the code.

| ID | Requirement |
|:--|:--|
| VAL-1 | Exact symmetry means `_symequal(a, b) = isequal(a, b) \|\| (a == b) === true` ([src/sparse_iterate.jl:932](https://github.com/mlakolar/ProximalBase.jl/blob/abaceda25442bf6cc61601c8c1d3b2a16e1a3492/src/sparse_iterate.jl#L932)): NaN matches NaN, `0.0` matches `−0.0`, `1` matches `1.0`. It is *structural equality* and says nothing about whether a value is *admissible*: a weight `−1` is symmetric and still rejected by the penalty's own domain check. The two checks stay separate. |
| VAL-2 | The comparison of a mirrored pair *must* be made on the **source** values, before conversion to `T`. A pair unequal in `Float64` that rounds equal in `Float32` *must* be rejected. |
| VAL-3 | A coordinate oracle (`prox_coordinate!`, `kkt_violation` for `k`) *may* inspect only the pair it uses. A stored-only operation *may* inspect only the pairs it touches. Neither certifies the model. |
| VAL-4 | The **full validation entry point** is `ProximalBase.check_symmetric_parameters(f, X)`, with `ProximalBase.entry_parameters(f)` as the declaration for extensions. Both *must* keep their roles. A solver calls it once at creation, on the **original caller-supplied penalty** and before any transformation of its parameters. A solver *must not* silently repair an asymmetric parameter (average, or take one triangle), because the library must not choose the caller's interpretation: a half-filled weight matrix may mean "use this triangle", not "average it with zeros". (Averaging does preserve the objective of a *weighted L1* penalty on symmetric matrices; that is not the reason for the rule.) Validating the result of such a transformation is too late. The audited ProximalSolvers does the opposite: `_symmetric_penalty(g, x::SymmetricSparseIterate)` defaults to `average = true` (`src/cd/models.jl:540`, `_symmetric_average` `:561`), and both workspace constructors call it (`src/workspaces.jl:76,99`). That behavior conflicts with the amended #142 rule and must change in ProximalSolvers; this contract does not change it. |
| VAL-5 | **Scope: this redesign promises scalar Frobenius steps `γ` only** for the oracles of §2.9 on an owner. Per-entry step arrays and diagonal metrics on an owner are **out of scope**, and an array `γ` *should* throw an `ArgumentError` there. This is a scope choice, not a claim that a symmetric diagonal metric destroys coordinate independence. **What would be lost:** the audited code accepts an exactly symmetric `γ` matrix in `prox!` into an owner (run: `prox!(Y, NormL1(0.5), [0 3; 3 0], [1 2; 2 1])` gives `2.0`, and an asymmetric `γ` throws), and `Tilt`'s path checks symmetric steps (`src/metric.jl:69,77`); ProximalSolvers already rejects diagonal preconditioning on an owner (`src/solvers/diagonal_metric.jl:23`). Scalar steps are validated by the step contract (numerics P1–P3). |
| VAL-6 | **What invalidates a validation.** The check is a statement about the arrays at the time of the call. Mutating any array reported by `entry_parameters` afterward, in either triangle, invalidates it (arrays symmetric by type, such as `Symmetric(W, :L)` over a parent the caller mutates, stay symmetric). A caller who mutates parameters re-validates. |
| VAL-7 | A custom entry function whose result depends on the index `(i, j)` without a parameter array (so `entry_parameters` cannot see it) *must* itself be symmetric. That responsibility is the extension author's. |
| VAL-8 | No public unchecked API and no validation cache is introduced by this contract. The per-call cost of dense-parameter scans (#374) is left to its own issue. |

**Checked writes.** Before the first mutation of `X`, including its active set,
the following *must* be detected: a shape or axes mismatch; an index out of
bounds; a mirrored pair that is unequal under VAL-1/VAL-2; a destination that is
not supported (complex element type; the read-only view). The following are
*not* promised: that a conversion which throws (`InexactError` into an integer
`T`) or an arbitrary user callback leaves `X` untouched; that a write that
fails in the middle of a *non*-listed way rolls back.

| bulk operation | aliasing |
|:--|:--|
| `copyto!(X, X)` | no-op |
| `copyto!(X, Y)`, `Y` another owner | supported; distinct owners never share storage |
| `copyto!(X, M)`, `M` a matrix sharing memory with `X` | rejected (CPY-6) |
| `θ .= g.(θ)` | supported: same-coordinate broadcast, as for any vector |
| `prox!(X, f, X, γ)` | supported where the package alias rule allows it |
| `prox!(Y, f, V, γ)`, `V` a partial overlap of `Y` | unsupported (#318) |

**Numbers.** `T` is `Float32` or `Float64`. Mixed precision, non-finite values,
return types and overflow follow the package contract in
[`docs/src/numerics.md`](../src/numerics.md) (notably M1 *implicit zeros are
entries*, M2 *multiplicity*, T- and R-rules); #366 tracks it. This contract adds
one thing only: multiplicity enters as in §2.8, and the stored numbers are never
rescaled. **Flagged for reconciliation:** the sentence in
`docs/src/iterates.md` that an owner is "for indexing, length, iteration and
broadcasting … an ordinary p × p matrix" is superseded by this contract and must
be rewritten with it; numerics M2 is consistent with it.

## 2.11 Performance as a contract

Fixed-width CPU scalar types (`Float32`, `Float64`).

| ID | Requirement |
|:--|:--|
| PERF-1 | After construction, lookup and update by ID or by matrix pair, membership (`isactive`), and the **first activation** of a coordinate *must* be worst-case `O(1)`. This is deterministic indexing, not amortized insertion and not an expected hash lookup. |
| PERF-2 | The capacity prerequisite is stated openly: construction reserves room for all `q` coordinates, so activation never grows storage. Construction time and memory are `O(q)`. "Sparse" does not promise `O(s)` total storage. |
| PERF-3 | Traversal of the active set (`activecoordinates`, `activevalues`) *must* cost `O(s)` of traversal work. The derivative, cache update and full KKT screening a solver attaches have their own costs and are not covered. |
| PERF-4 | Constructing `θ`, `A`, `activecoordinates(X)` and `activevalues(X)` *must not* copy `X`'s data. |
| PERF-5 | After warm-up, these *must not* heap-allocate for `Float32`/`Float64` inputs: `X[i,j]`, `X[i,j] = v`, `θ[k]`, `θ[k] = v`, `isactive`, `activate!` (capacity reserved), `prox_coordinate!` for a built-in separable `f` with scalar or `Symmetric`-typed parameters, and a sweep over `activecoordinates(X)` that calls it. |
| PERF-6 | `mul!(y, A, v)` for a dense vector `v` *must* cost `O(p + s)` for finite input, **including initialization of `y`**. A no-copy wrapper that falls to the dense `O(p²)` scan does not meet it. For a non-finite `v` the result *must* be the same, or the same documented error, **before and after `dropzeros!`**: a mathematical product must not depend on whether a zero has a storage slot. (Audited code does not satisfy this: with `v = [Inf, 1, 0]` and one stored explicit zero at `(3,1)` the product has a `NaN` at row 3, and after `dropzeros!` the same entry is `0.0`.) *Recommended default:* one `O(p)` scan of `v` and an `ArgumentError` for any non-finite entry, so the finite case stays `O(p + s)`. Numerics M1 cites *finite* linear coefficients and does not settle `0·Inf`; the rule must be reconciled with the non-finite policy in `docs/src/numerics.md`. |
| PERF-7 | `dropzeros!` *must* be `O(s)`. `Matrix(X)` is `O(p² + s)`; `copy(X)` and `similar(X)` are `O(q)`. |
| PERF-8 | No other generic matrix operation on `A` is promised to be sparse-fast. *Recommended* sparse-aware methods: `norm`, `dot`, `sparse(A)`, `sum`. A reader of the specification must not infer more. |

The contract names no field, no storage type and no hash table. The audited
code uses a direct index table of length `q`, which meets PERF-1 and PERF-2; that
is an implementation, not a promise.

---

# Part 3. Decisions for maintainer sign-off

Each gives a recommended default, one reason and one consequence.

1. **Owner is no longer an `AbstractMatrix`** (OWN-1). *Recommend: yes.* No
   inherited generic mutator can reach the owner. *Consequence:* every
   `Base`-generic call on an owner (`sum(X)`, `cholesky(X)`, `X * v` through
   `AbstractMatrix`, `x .= …`) stops compiling; callers pass `matrixview(X)` or
   `Matrix(X)`.
2. **`size(X) == (p, p)` and no `length`** (OWN-3, OWN-4). *Recommend: keep
   `size`, leave `length` undefined.* `size` is unambiguous and used; `length`
   has two plausible meanings. *Consequence:* `length(X)` is an error.
3. **`nnz(X)` is removed; `nactive(X)` replaces it; no matrix-view `nnz`**
   (ACT-2). *Recommend: remove.* Its three possible counts differ by up to
   a factor of two and an audited caller (CovarianceSelection's
   `hard_threshold!`) indexes `1:nnz` into private storage.
   *Consequence:* code that used `nnz(X)` as a model size writes
   `count(!iszero, activevalues(X))` after `dropzeros!`, or `nactive(X)`.
4. **Matrix import replaces the working set** (CPY-4). *Recommend: replace.*
   The result depends only on the argument. *Consequence:* explicit zeros
   disappear on import; a warm-start that must keep them uses `θ .= …`.
5. **New name `activevalues`** (ACT-4). *Recommend: add.* Two audited packages
   read or write the first `nnz` values of private storage by slot
   (ProximalSolvers' momentum history, CovarianceSelection's `hard_threshold!`);
   this is the smallest supported replacement. *Consequence:* one new exported
   name, with the handle lifetime of ACT-3.
6. **An active handle is a handle, not a snapshot** (ACT-3). *Recommend: handle,
   reacquire after activation or compaction.* A snapshot costs `O(s)` and an
   allocation per sweep. *Consequence:* a solver that activates inside a sweep
   must reacquire; an active-only sweep need not.
7. **Projection of the proximal argument is part of the oracle under a scalar
   Frobenius step only** (§2.10 item 3). *Recommend: yes, and not beyond.* It
   is exact there. *Consequence:* #261 must land for ORC-3 to be true. An
   array step is not accepted on an owner at all (decision 11, VAL-5), so no
   projection of the argument under an array step arises.
8. **`prox_entry` keeps a matrix linear index** (IDX-3). *Recommend: keep.*
   Weights are matrices. *Consequence:* the two index spaces still meet in this
   one function; the docstring must say so.
9. **Public KKT normalization is the entry normalization** (§2.8). *Recommend:
   keep.* It is comparable across diagonal and off-diagonal coordinates.
   *Consequence:* a solver holding a coordinate derivative divides by `mₖ`
   before calling.
10. **A matrix-semantic reduction set on owners** (ORC-10: `norm`, `norm_diff`,
    `dot`). *Recommend: keep, as today.* ProximalSolvers' stopping norm and
    trial arithmetic need it, and `norm(θ)` is the wrong number.
    *Consequence:* three named methods on the owner outside `AbstractArray`.
11. **Scalar Frobenius steps only** (VAL-5). *Recommend: yes.* One scope, stated
    once; no array-step claim to defend. *Consequence:* symmetric per-entry step
    arrays that `prox!` accepts into an owner today are rejected in 0.6.
12. **Solvers validate the caller's original penalty and never repair it**
    (VAL-4). *Recommend: yes.* The library must not choose the caller's
    interpretation: a half-filled matrix may mean "use this triangle", not
    "average it with zeros", even though averaging preserves a weighted L1
    objective on symmetric matrices. *Consequence:* ProximalSolvers' default-averaging
    `_symmetric_penalty` must change; users who want the mean write it.
13. **Non-finite `v` in `mul!(y, A, v)`: error, by default** (PERF-6).
    *Recommend: `ArgumentError` after one `O(p)` scan.* The product then cannot
    depend on whether a zero has a slot. *Consequence:* a `NaN` or `Inf` in `v`
    raises instead of propagating; this needs reconciling with numerics.

# Part 4. Where today's code differs (informative)

Checked at the audited commit; "ran" means a script was run against it.

| claim | at `abaceda2` |
|:--|:--|
| `SymmetricSparseIterate <: AbstractMatrix`, `size == (p, p)` | [src/sparse_iterate.jl:131](https://github.com/mlakolar/ProximalBase.jl/blob/abaceda25442bf6cc61601c8c1d3b2a16e1a3492/src/sparse_iterate.jl#L131), `:896` |
| `coordinates(X)` is a writable `AbstractVector` of length `q`; `length(X) == p²` | `:913–925`; ran |
| no `matrixview`, no `activevalues` | ran (`isdefined`) / grep |
| `X[4]` at p = 3 is matrix entry `(1,2)`; `coordinates(X)[4]` is `(2,2)` | ran |
| `NormL1(1.0)(X) == 6.0` for one off-diagonal `3.0`; `norm(X) = 3√2`, `norm(coordinates(X)) = 3` | ran |
| a `for i, j` loop `X[i,j] *= 2` quadruples an off-diagonal | ran (`1.0 → 4.0`) |
| `nnz(X) == nactive(X)`, counts stored coordinates, explicit zeros included | `:223`; ran |
| `prox!` of a non-symmetric dense argument into an owner is inconsistent across penalties | ran (`NormL1` gives the lower triangle, `SqrNormL2` throws) |
| ProximalSolvers coordinate updates already use `coordinates(x)`, `activecoordinates(x)`, joint-direction derivative and curvature, and multiplicity 2 on the penalty | src/cd/kron.jl:97–120, src/cd/models.jl:146, `:419–420` |
| `mul!(C::AbstractVector, A::SymmetricSparseIterate, B::AbstractVector)` loops over the stored entries | `:1716` (read, not timed) |
| a stored explicit zero changes a product with non-finite `v` (`NaN` against `0.0` after `dropzeros!`) | ran |
| writing `1e-100` to an absent `Float32` coordinate leaves it absent (activation from the converted value) | ran |
| symmetric array `γ` accepted by `prox!` into an owner; asymmetric `γ` throws | ran |
| `prox(f, X, γ)` returns an owner; `gradient(f, X)` a dense `Matrix` | ran (through `AbstractArray` dispatch, `src/interface.jl:134,318`) |
| ProximalSolvers averages asymmetric weights at the solver boundary | `src/cd/models.jl:540,561`; `src/workspaces.jl:76,99` (read) |
| `kkt_violation` is a screen at zero, not a stationarity residual | `src/interface.jl:559` and docstring (read) |

*Released versus audited versus proposed.* I did not check what is published in
the General registry; "0.5.4" is the `Project.toml` of the audited commit.
Everything in Part 2 that differs from the table above is **proposed 0.6
behavior**.
