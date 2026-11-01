# The iterate family in ProximalBase 0.6: the contract

> **Draft for maintainer review — proposed 0.6 behavior, not a description of an
> installed release.** Nothing here is implemented. This file is the only
> normative document of the pair; the companion
> [compatibility and acceptance plan](iterate-family-0.6-acceptance.md) links
> back to it and restates none of its rules.

**Audited commits.** ProximalBase `def81ec9d021c39ed1727cfeb9598357291dd60d`
(master, 2026-11-01; `Project.toml` says 0.5.4). It includes #392
(`matrixview`, `activevalues`, the owner's `dataids`). ProximalSolvers
`3893fb95da9ab7326530c46b24fed801dbac1e74` (master), read as evidence only.
A bare `src/…:line` refers to ProximalBase at the first commit; one prefixed
with "ProximalSolvers" refers to the second. Full links point at the same
commits. Part 4 lists every statement about current behavior with its
evidence; the companion (§1) lists what was traced and what was not.

**How to read it.** Part 1 is for a Julia user. Part 2 is for a package or
solver author: requirements carry stable IDs, *must* marks a public guarantee,
and *recommended* marks an implementation idea that a conforming
implementation may replace. Each requirement has a status: **0.5.x** (already
true at the audited commit), **0.6 target** (proposed, new or breaking),
**recommendation** (an implementation idea) or **open** (awaiting a separate
decision).
The column "types" says which iterates a row covers: **S** for
`SparseIterate`, **A** for `AtomIterate`, **O** for `SymmetricSparseIterate`,
the *owner*. Part 3 holds the decisions for sign-off. Part 4 is informative.

**Relation to the owner contract.** The owner's rules are those of the
[`SymmetricSparseIterate` contract](symmetric-sparse-iterate-0.6.md) (#388:
OWN, IDX, CRD, MAT, CPY, ACT, ORC, VAL, PERF and decisions 1–13). This file
does not restate them. It cites them by ID, adds the rules the whole family
shares, and gives the two array iterates their own. Where it proposes to
change a signed-off owner rule, Part 3 says so in a decision record
([FD-9](#fd-9-zero-coefficients-are-strong-in-products)), and the owner
documents carry a pointer; where it extends one without changing it (FD-7's
handle lifetime, FD-16's three-factor product), the pointer says so. No
signed-off sentence is rewritten, and CPY-4 stands as written
([FD-3](#fd-3-matrix-import-into-the-owner-keeps-cpy-4)).

---

# Part 1. The idea, with examples first

## Three roles, one object

Each iterate plays three roles. They are meanings, not types: one object
carries all three.

| role | `SparseIterate` | `AtomIterate` | owner |
|:--|:--|:--|:--|
| mathematical value and shape | a vector or a matrix | a vector, sometimes `vec(B)` of a matrix `B` | a symmetric p × p matrix |
| coordinates and their IDs | one entry; ID = linear entry index | one group of entries; ID = group number | one lower-triangle variable; ID = `tril_index(p, r, c)` |
| working-set state | the active entries, explicit zeros included | the active groups | the active variables (ACT) |

**Why the owner is not an array and the other two are.** Generic array code
visits every entry once. For a `SparseIterate` an entry is a variable, and
for an `AtomIterate` an entry belongs to exactly one group, so a loop
`for i in eachindex(x); x[i] *= 2; end` doubles every variable once: correct
for values. For the owner, `X[2, 1]` and `X[1, 2]` are one variable, and the
same loop doubles it twice (#388, Part 1). So the two array iterates keep
`AbstractArray` and `AbstractVector` (FS-1). What they lack is a stated rule
for the working set, safe handles, one meaning for products with a zero
coefficient, and public accessors in place of fields. This contract supplies
them.

## Index spaces

| index | `SparseIterate` | `AtomIterate` | owner | durable? |
|:--|:--|:--|:--|:--|
| entry index | `i ∈ 1:length(x)`; `x[i]`, and `x[r, c]` for a matrix | entry ID `j ∈ 1:length(x)`; `x[j]` | matrix position `(i, j)`; `matrixview(X)[i]` (MAT) | yes |
| coordinate ID | `k = i`; for an m × n matrix `k = (c − 1)m + r`; `coordinates(x)[k]` | group ID `k ∈ 1:ncoordinates(x)`; `atom(x, k)`, `atomindices(x, k)` | `k = tril_index(p, r, c)`; `coordinates(X)[k]` (IDX-1) | yes |
| active slot | `a ∈ 1:nactive(x)`; `activecoordinates(x)[a]`, `activevalues(x)[a]` | `a`; `activecoordinates(x)[a]` | `a` (ACT-3) | **no**: `dropzeros!` moves slots |

An `AtomIterate` has two durable indices, and confusing them changes the
answer: `isactive(x, 2)` asks about *group* 2, `x[2]` reads *entry* 2.

## Examples

Lines marked *future* use proposed behavior; none is a doctest. The others
were run at the audited commit.

**1. Value writes keep coordinates, compaction removes them, a state copy
copies them.**

```julia
x = SparseIterate(4)
x[4] = 7.0; activate!(x, 2)          # working set [4, 2]; 2 is an explicit zero
x .= [3.0, 0.0, 0.0, 0.0]            # a value write
activecoordinates(x)                 # [4, 2, 1]: 1 joins, 4 and 2 stay at zero
fill!(x, 0.0)                        # future: a value write, still [4, 2, 1] (0.5.x empties it)
dropzeros!(x); nactive(x)            # 0: compaction removes zeros, and only zeros

y = SparseIterate(4); y[3] = 5.0; activate!(y, 1)    # y's working set [3, 1]
copyto!(x, y); activecoordinates(x)  # [3, 1]: a state copy takes y's working set
z = SparseIterate(4); z[2] = 1.0
z .= y                               # future: a value write, working set [2, 3]
                                     # (0.5.x: Base turns it into the state copy, [3, 1])
```

A write through the array interface never removes a coordinate, however it is
spelled. To clear the working set, write `dropzeros!(fill!(x, 0))`; to replace
it by the support of an array `A` that shares no memory with `x`,
`copyto!(dropzeros!(fill!(x, 0)), A)`, which leaves the slot order of 0.5.x's
clear-then-import. Any other source is unaliased and checked first (FW-6).

**2. A zero coefficient of an iterate contributes nothing to a product.**

```julia
s = SparseIterate(3); s[1] = 1.0
t = SparseIterate(3); t[2] = NaN
dot(s, t), dot(t, s)                 # 0.5.x: (0.0, NaN); future: (0.0, 0.0)

A = [NaN 1.0 0.0; NaN 1.0 0.0]
u = SparseIterate(3); u[2] = 1.0; activate!(u, 1)    # explicit zero at 1
A * u                                # 0.5.x: [NaN, NaN], and [1.0, 1.0] after dropzeros!(u)
                                     # future: [1.0, 1.0] before and after
A * Vector(u)                        # [NaN, NaN]: a dense operand follows IEEE arithmetic
```

The rule (FN-1) is the one Julia already applies to `false * NaN`, which is
`0.0`, and the one this package applies to `α = 0` in `mul!`. It makes the
result independent of whether a zero is stored and of the order of the
arguments.

**3. Group IDs and entry IDs.** ProximalSolvers' multi-response model stores
a p × q coefficient matrix `B` as an `AtomIterate` over `vec(B)` whose groups
are the rows of `B`.

```julia
p, q = 3, 2
B = [1.0 2.0; 3.0 4.0; 5.0 6.0]
x = AtomIterate(vec(B), [collect(r:p:p*q) for r in 1:p])
x[(2 - 1)*p + 3] == B[3, 2]          # entry ID j = (c − 1)p + r
atom(x, 3) == B[3, :]                # group ID r is row r: a writable view
ncoordinates(x), length(x)           # (3, 6): 3 groups of 6 entries
atomindices(x, 2)                    # future: [2, 5], read-only
```

A model author who computes the p × q gradient `G` of the matrix model maps
it the same way: `vec(G)` is the gradient over the entries, and group `r` of
`vec(G)` is row `r` of `G`. This is a convention for model authors; today
ProximalBase's matrix losses do not take this iterate (`gradient!` of
`LeastSquaresLoss(Y, X)` with a matrix `Y` at `x` throws `DimensionMismatch`;
FD-15a).

**4. A group step at a trial point.**

```julia
f = GroupNormL2(1.0, [1:2, 3:4])     # default weights √|G| = √2
x = AtomIterate(4, f)
prox_coordinate!(x, f, 1, [3.0, 4.0], 1.0)   # the trial point is not written first
# [2.151471862576143, 2.868629150101524] = (1 − √2/5) .* [3, 4]; group 1 is active
```

If the call throws a validation error (FA-4), `x` is unchanged; if reading `u`
itself throws, a partial write is possible (FA-7, FT16). Writing the trial point into private
storage and then calling `prox_coordinate!(x, f, 1, 1.0)` gives the same
numbers when nothing throws; when the call throws it leaves an *inactive*
group holding `[3.0, 4.0]`, which the penalty value and a same-layout
`copyto!` then ignore (`copy` keeps it) (FA-7,
[FD-12](#fd-12-one-contract-for-coordinate-oracles)).

## What changes for you

- Ask for counts and handles through accessors (`nactive`, `activecoordinates`,
  `activevalues`, `atomindices`), not fields. `activecoordinates` becomes
  read-only.
- `fill!(x, 0)` and `x .= y` keep the working set; add `dropzeros!` where you
  meant to clear or replace it. `copyto!(x, y)` between iterates of the same
  kind is still the state copy.
- `coordinates(x)` of a `SparseIterate` becomes a coordinate-vector view of
  `x`, for vectors and matrices alike; it is no longer `x` itself.
- Products with an iterate ignore its zero coefficients, also against NaN and
  `Inf`. Views of an iterate are ordinary vectors, and scaling by a number
  follows IEEE arithmetic.
- Stage a group trial with `prox_coordinate!(x, f, k, u, γ)`.

---

# Part 2. Requirements

**Notation.** `x` is an array iterate, `X` an owner. `n = length(x)` for a
`SparseIterate`, `p = length(x)` and `K = ncoordinates(x)` for an
`AtomIterate`, `q = p(p+1)/2` for an owner, `s = nactive(x)`. `T = eltype(x)`.
"Active entries" of an `AtomIterate` are the entries of its active groups.
A *same-kind* iterate is defined in FX-1.

## 2.1 Shape and roles (FS)

| ID | requirement | status | types |
|:--|:--|:--|:--|
| FS-1 | `SparseIterate{T, N} <: AbstractArray{T, N}` (N = 1, 2) and `AtomIterate{T} <: AbstractVector{T}` *must* keep their supertypes and their array interface: `size`, `length`, linear indexing (and Cartesian for N = 2), iteration and broadcasting describe the mathematical value, and a full traversal visits each entry once. The owner is not an array (OWN-1). | 0.5.x | S, A |
| FS-2 | Equality and hashing *must* see values only: two iterates with the same entries are `==` and `isequal` and hash alike whatever their working sets, and an iterate equals the dense array of its entries. | 0.5.x | S, A |
| FS-3 | `T` *must* be real; a complex `T` throws `ArgumentError` at construction. The numeric scope is numerics S1. | 0.5.x | S, A (O: OWN-5) |
| FS-4 | There is no shape-aware grouped iterate in 0.6. A p × q coefficient matrix `B` is an `AtomIterate` over `vec(B)`: entry ID `j = (c − 1)p + r` is `B[r, c]`. With the rows of `B` as groups, group ID `r` has the entries `r:p:p*q` (as a range or the collected vector) and `atom(x, r) == B[r, :]`. A p × q gradient `G` of the matrix model maps by `vec` the same way: group `r` of `vec(G)` is row `r` of `G`. The gradient mapping is a convention for model authors; what ProximalBase's matrix losses do with this iterate today is described in FD-15a and is not part of this requirement. | 0.5.x | A |
| FS-5 | There is no family-wide value-view name. The value view of an array iterate is the iterate itself; `matrixview` stays owner-only (MAT-1). | 0.5.x | S, A |

## 2.2 Coordinates, handles and counts (FC)

| ID | requirement | status | types |
|:--|:--|:--|:--|
| FC-1 | An integer passed to a coordinate API (`isactive`, `activate!`, `atom`, `atomindices`, `prox_coordinate!`, the four-argument `kkt_violation`) *must* be read as a coordinate ID, and one passed to `getindex` or `setindex!` as an entry index (S, A) or a matrix position (O). A slot is never read as an ID (IDX-4, for the family). | 0.5.x | S, A, O |
| FC-2 | The coordinate ID of a `SparseIterate` *must* be its linear entry index: `k = (c − 1)m + r` for an m × n matrix. | 0.5.x | S |
| FC-3 | `coordinates(x::SparseIterate)` *must* return a coordinate-vector view of `x`, an `AbstractVector{T}` wrapper like the owner's `SymmetricCoordinates`, for N = 1 and N = 2 alike; it is never `x` itself. It has length `ncoordinates(x)`, is indexed by coordinate ID, is live and writable, is `O(1)` to build and copies nothing; its `Base.dataids` are those of `x`, and `Base.unaliascopy` of the view returns a view of a copy of `x` (as `SymmetricCoordinates` does, `src/sparse_iterate.jl:943`), so that an aliased import works (FX-5). Writes through it are value writes (FW-2), and its products are ordinary-vector products (FN-3). It has no check of its own for complex input: it behaves as a plain vector of element type `T` holding the same entries (FN-8), as the owner's `SymmetricCoordinates` does today, so a complex value throws `InexactError`, possibly after a partial write. | 0.6 target (breaking: `coordinates(x) === x` no longer holds for a vector, and for a vector `coordinates(x) .= complex` no longer throws `ArgumentError` before writing, as it does today because the view is `x`) | S (O: CRD-1) |
| FC-4 | `coordinates` *must not* be defined for an `AtomIterate`, whose coordinate is a group. | 0.5.x | A |
| FC-5 | `atom(x, k)` *must* return a writable view of the entries of group `k`, a view of `x` (so it aliases `x`), bounds-checked on `k`. It stays valid for the lifetime of `x`: activation and compaction do not invalidate it. Writes through it are value writes (FW-2). | 0.5.x | A |
| FC-6 | `atomindices(x, k)` (**new**) *must* return the entry IDs of group `k` in stored order as a read-only `AbstractVector{Int}`: the stored range itself for a range group, a read-only wrapper for a vector group. `O(1)`, no allocation, `BoundsError` for `k ∉ 1:ncoordinates(x)`, valid for the lifetime of `x`. `parentindices(atom(x, k))` is not a supported route to the layout. | 0.6 target (0.5.x-safe addition) | A |
| FC-7 | `activecoordinates(x)` *must* return a read-only `AbstractVector{Int}` of the active IDs in slot order, of length `nactive(x)` at the call, built in `O(1)` without copy or allocation. A write through it *must* be rejected (any exception) and leave `x` unchanged. It is a handle with ACT-3's lifetime, extended by one case: it is valid until the next activation, compaction or whole-state replacement, that is, a same-kind state copy (`copyto!(x, y)`, `copy!(x, y)`, FX-1) or the owner's CPY-4 import. This holds whatever happens to `nactive` and to membership: a state copy can keep both and still reorder the slots. A self-copy `copyto!(x, x)` is excepted. A stale handle's contents are unspecified but memory-safe, and so is any history a caller keys by slot. The three iterates share one handle type. | 0.6 target (a 0.5.x bug fix for S and A, #405) | S, A, O (ACT-3) |
| FC-8 | `activevalues(x)` *must* return a writable `AbstractVector{T}` of the active values in slot order, with FC-7's lifetime, the whole-state replacement included; writing zero keeps the coordinate active. For a `SparseIterate`, `nonzeros(x)` is the same handle. | 0.5.x (#392) | S, O (ACT-4) |
| FC-9 | `activevalues` *must not* be defined for an `AtomIterate`. The loop over its active values is `for k in activecoordinates(x), j in atomindices(x, k)`, reading and writing `x[j]`, at `O(active entries)`. | 0.5.x | A |
| FC-10 | Four counts *must not* be conflated; the table below gives each one's spelling. | 0.5.x (stated) | S, A, O |
| FC-11 | `nnz`, `nonzeros` and `findnz` *must* stay on `SparseIterate` with SparseArrays' meaning, explicit zeros included: `nnz(x) == nactive(x)`, `nonzeros(x)` is the `activevalues(x)` handle, and `findnz(x)` returns copies sorted by index. An `AtomIterate` has no `nnz` (`MethodError`); the owner follows ACT-2. | 0.5.x | S, A |

The four counts of FC-10:

| count | `SparseIterate` | `AtomIterate` | owner |
|:--|:--|:--|:--|
| coordinates | `ncoordinates(x) == length(x)` | `ncoordinates(x)`, the groups | `ncoordinates(X) == q` (OWN-4) |
| active, explicit zeros included | `nactive(x) == nnz(x)` | `nactive(x)`, the active groups | `nactive(X)` (ACT-2) |
| mathematical nonzeros | `count(!iszero, activevalues(x))`, `O(s)` | entries: `sum(k -> count(!iszero, atom(x, k)), activecoordinates(x); init = 0)`; groups: `count(k -> any(!iszero, atom(x, k)), activecoordinates(x))`; both `O(active entries)` | `count(!iszero, activevalues(X))` (#388 §2.6) |
| storage capacity | `length(x)` slots, all reserved at construction | `length(x)` values and `ncoordinates(x)` group slots, all reserved | `q` slots (PERF-2) |

## 2.3 The working set (FW)

| ID | requirement | status | types |
|:--|:--|:--|:--|
| FW-1 | **Inactive implies zero.** A coordinate that is not active *must* be zero (every entry of an inactive group is zero) after every supported operation, including one that throws. | 0.6 target (0.5.x except through the writable handles, Part 4) | S, A, O |
| FW-2 | **Value writes.** A value write *must* convert each value to `T` first, activate a coordinate whose converted stored value is nonzero (NaN included; `−0.0` and a value that rounds to zero are zero), and never remove a coordinate. Every write through the array interface is a value write: `setindex!`; every broadcast form, `x .= c` and `x .= y` included; `fill!`; `copyto!` and `copy!` from any source that is not a same-kind iterate; offset `copyto!`; writes through `view`, `reshape`, `vec`, `atom`, `coordinates` and `activevalues`; scaling by a number (FN-9); and Base and LinearAlgebra mutators such as `ldiv!`, `permute!`, `map!`, `sum!` and a generic `mul!` into an iterate. A mutator that Base implements as `copyto!(x, y)` with a same-kind `y` (for example `ldiv!(x, F, y)`) inherits the state copy of FW-4. | 0.6 target | S, A; O: CRD-3, ACT-5 |
| FW-3 | **A broadcast is a loop.** Every broadcast into an array iterate *must* give the values and the working set of the explicit loop `for i in eachindex(x); x[i] = rhs[i]; end` over the unaliased right-hand side. An implementation must not let Base's identity shortcut turn `x .= y` into a state copy (table below). | 0.6 target | S, A (O: CRD-5) |
| FW-4 | **State operations** make `x` a copy of a working set: `copy(x)`; `similar(x)` and `similar(x, S)` (empty); `copyto!(x, y)` and `copy!(x, y)` with `y` a same-kind iterate. The copy *must* keep the source's slot order and explicit zeros and convert values to `x`'s element type without deciding activation again. | 0.5.x | S, A (O: CPY-1–CPY-3) |
| FW-5 | **Compaction.** `dropzeros!(x)` *must* be the only operation that removes coordinates. It removes exactly the active coordinates whose value is zero (the active groups whose entries are all zero), keeps every value and ID, may reorder slots, runs in `O(s)` (S) or `O(active entries)` (A), and does not allocate. A whole-state replacement (FX-1's state copy, the owner's CPY-4 import) is not compaction: it replaces the working set and its slot order by the source's, and also ends FC-7's handles. | 0.5.x | S, A (O: ACT-7) |
| FW-6 | **Spellings.** *Array iterates.* Clearing the working set *is* `dropzeros!(fill!(x, 0))`, at `O(n + s)` (S) or `O(p + active entries)` (A). For a source `A` that shares no memory with `x` and passes the checks (the length of `x`, a real element type), replacing the working set by the support of `A` *is* `copyto!(dropzeros!(fill!(x, 0)), A)`, at the same cost. For any other source the shorthand is wrong, because it clears `x` before it reads `A`: with `x = SparseIterate([1., 2.])`, `copyto!(dropzeros!(fill!(x, 0)), view(x, 2:-1:1))` gives `[0.0, 0.0]`. The safe migration validates the length and the element type, then sets `A = Base.unalias(x, A)`, then uses the shorthand. At the audited commit `SparseIterate`'s `dataids` is the object's identity, so a `view` of `x` is detected (its `dataids` are its parent's) and copied, while the handles `activevalues(x)` and `nonzeros(x)` are detected only once FX-5 widens `dataids`. *The owner.* The replace *is* `copyto!(X, M)`, CPY-4's checked, atomic import; the clear *is* `fill!(coordinates(X), 0); dropzeros!(X)`. *The difference.* `copyto!` from a plain array is therefore a checked replace for the owner (CPY-4, FD-3) and a value write for the array iterates (FW-2). The reason is the one that separates the owner from the arrays: generic Base code reaches the array iterates' `copyto!` (§2.3's table) and never the owner's. *Order.* A replace leaves the slot order of 0.5.x's clear-then-import, and CPY-4's for the owner: increasing ID for `SparseIterate` and the owner. For an `AtomIterate`, a value write in linear order after a clear activates the groups in the order in which it first meets a nonzero entry of each (increasing group ID when every entry of group k precedes every entry of group k + 1, as for contiguous groups in order). The constructor instead activates in group-ID order (`src/atom_iterate.jl:76–78`), so for non-contiguous groups a replace and a fresh construction differ: with groups `[[1, 3], [2, 4]]` and nonzeros at entries 2 and 3 a replace gives `[2, 1]`, and for the strided multi-response layout the constructor gives `[1, 2]` where a replace gives `[2, 1]`. Slot order is observable: a cyclic active pass visits slots in order (ProximalSolvers `src/cd/solver.jl:329–331`), and the momentum history is keyed by slot. The shorter `dropzeros!(copyto!(x, A))` gives the same set in another order. | 0.6 target | S, A, O |
| FW-7 | `activate!(x, k)` *must* make coordinate `k` active, at value zero for S and keeping the (zero) entries of group `k` for A, if it is absent; it changes nothing otherwise and throws `BoundsError` out of range. A sweep over `activecoordinates(x)` that writes values only of active coordinates *must* leave `activecoordinates(x)` unchanged. | 0.5.x | S, A (O: ACT-5, ACT-6) |
| FW-8 | **Oracle destinations.** After a ProximalBase oracle (`prox!`, `value_and_prox!`, `gradient!`, `mul!`, `prox_coordinate!`) writes an iterate, its values *must* be the result and its working set *must* contain every coordinate with a nonzero result (FW-1). Which zero coordinates stay active is unspecified; `dropzeros!` gives the support. | 0.5.x (stated) | S, A (O: ORC-3, ORC-6) |

**Where Base sends a bulk write.** For an `AbstractVector` destination that
defines `fill!` and `copyto!(dest, ::AbstractArray)`, identical on Julia
1.10.12 and 1.13.3 (ran):

| call | reaches | Base source, 1.13.3 (1.10.12) |
|:--|:--|:--|
| `d .= v` | `copyto!(d, v)` | `broadcast.jl:993`, the identity shortcut's `return` (`:997`) |
| `d .= 0` | `fill!(d, 0)` | `broadcast.jl:960`, the scalar case's `return` (`:964`) |
| `d .= 0 .* d`, `d[:] = v`, `map!(identity, d, v)`, `mul!(d, A, v)`, `mul!(d, A, v, 1, 0)`, `copyto!(d, 1, v, 1, 3)` | `setindex!` only | |
| `copy!(d, v)` | `copyto!(d, v)`; a source of another length first calls `resize!`, which an iterate lacks (`MethodError`) | `abstractarray.jl:924–931` for vectors, `:933–937` for other arrays (`:918`, `:927`) |
| `ldiv!(d, lu(A), v)` | `copyto!(d, v)` | |
| `permute!(d, p)` | `copyto!(d, v[p])`, `v[p]` a dense `Vector` | `combinatorics.jl:214`, `permute!(v, p) = (v .= v[p])` (`:192`) |
| `sum!(d, A)`, `zero(d)` | `fill!` | |
| `copy(d)` | `copyto!(similar(d), d)` | |

Under FW-2 every row is a value write, except `copy(d)`, which is the state
copy of FW-4, and `zero(d)`, which fills an empty `similar(d)`. Today the
`fill!` rows clear the working set and `d .= y` with a same-kind `y` is a
state copy (Part 4).

## 2.4 Copies and imports (FX)

| ID | requirement | status | types |
|:--|:--|:--|:--|
| FX-1 | **Same kind.** `y` is a same-kind iterate of `x` when both are `SparseIterate`s of the same size, or both are `AtomIterate`s of the same length whose layouts are equal (FL-4); element types may differ. `copyto!(x, y)` and `copy!(x, y)` *must* then be a state copy (FW-4) that discards `x`'s previous working set. A conversion that makes a value zero keeps its coordinate active: `1e-50` copied from `Float64` into `Float32` is an active `0f0`. `copyto!(x, x)` is a no-op. | 0.5.x | S, A (O: CPY-3) |
| FX-2 | **Imports are value writes.** Every other source of a two-argument `copyto!` or `copy!` (a dense array, a `SparseVector`, a `SparseIterate` of another shape and the same length, an `AtomIterate` of another layout and the same length, a view or another wrapper) *must* give a value write (FW-2) with Base's `copyto!` semantics, in linear order: the old working set stays and the coordinates made nonzero join. With FX-1, the same values can end in different working sets: a copy between two `AtomIterate`s is a state copy when their layouts are equal and a value write when they differ (FD-2, option (e)). | 0.6 target (0.5.x for dense sources; another shape throws `DimensionMismatch` today) | S, A |
| FX-3 | `copyto!(x, y)` between two array iterates of different lengths *must* throw `DimensionMismatch` before writing. | 0.5.x | S, A |
| FX-4 | The iterate's own bulk writes (constructors, two-argument `copyto!`, `copy!`, every broadcast into it, `fill!`) *must* reject a complex source or value with `ArgumentError` before writing anything. The check needs a method `copyto!(x::SparseIterate, y::AbstractArray)`, and with it methods that resolve ambiguities with SparseArrays (FD-13). | 0.6 target (0.5.x for broadcasts; `copyto!` writes, then throws `InexactError`) | S, A (O: CPY-4's checks) |
| FX-5 | **Aliasing.** `Base.dataids(x)` *must* cover every storage array behind a public handle (S: values, ID list and reverse map; A: values and active list), so that `Base.mightalias` sees `activevalues(x)`, `nonzeros(x)` and `activecoordinates(x)` sharing memory with `x`. An import whose source shares memory with `x` *must* give the result of copying the source first (Base's `unalias`). | 0.6 target (a 0.5.x bug fix, #406) | S, A (O: CPY-6 rejects instead) |
| FX-6 | `copy(x)` and `similar(x)` of an `AtomIterate` *must* share its layout object, and `AtomIterate(p, f)` shares the penalty `f`'s. This is safe because the layout is immutable (FL-2). | 0.5.x | A |

## 2.5 Group layout (FL)

| ID | requirement | status | types |
|:--|:--|:--|:--|
| FL-1 | Construction *must* validate the layout: groups nonempty, of positive indices, disjoint, partitioning `1:p`, each failure an `ArgumentError` before anything is built. `AtomIterate(p, K)` needs `K` to divide `p`. | 0.5.x | A |
| FL-2 | **Ownership.** The constructor *must* own the layout: a vector of ranges is kept, any other collection becomes `Vector{Int}` copies, so later changes to the caller's arrays do not reach the iterate. No ProximalBase accessor may expose the layout mutably (FC-6). Mutating group arrays obtained any other way (a field, `parentindices`) is unsupported, as `docs/src/iterates.md:440–442` says today. | 0.6 target (ownership is 0.5.x; `parentindices(atom(x, k))` exposes a vector group today) | A |
| FL-3 | *Recommended:* store vector groups read-only, so that `parentindices(atom(x, k))` cannot mutate them either. Measure `atom`, `norm` and `dot` of atom views, and the products, before adopting it. | recommendation | A |
| FL-4 | **Compatibility for coordinate oracles** (`prox_coordinate!`, the four-argument `kkt_violation`): an iterate and a group penalty *must* have equal ordered index lists (the same number of groups, `groups[k] == f.groups[k]` for every `k`), checked before writing; otherwise `DimensionMismatch("f and x must have the same groups.")`. | 0.5.x | A |
| FL-5 | **Whole-array oracles** (`f(x)`, `prox!`, `value_and_prox!`, gradients) *must* accept any layout and compute the function of the vector; they do not reject a different layout. | 0.5.x | A |
| FL-6 | **Cost of FL-4.** `O(1)` when the iterate and the penalty share the layout object. Otherwise one `O(total group size)` comparison when a different object is first seen, then `O(1)` while the iterate remembers it. It remembers one object: alternating two equal but distinct layouts compares each time. | 0.5.x | A |
| FL-7 | `ProximalBase.samegroups(a, b)` (**new**; public, not exported) *must* answer FL-4's question for an iterate and a group penalty, or for two iterates, at FL-6's cost. | 0.6 target (0.5.x-safe addition) | A |

## 2.6 Products and numerics (FN)

| ID | requirement | status | types |
|:--|:--|:--|:--|
| FN-1 | **Strong zeros.** In the products and inner products of FN-2, a zero coefficient of the iterate, stored or not, *must* contribute nothing, even against `±Inf` or NaN. A non-finite entry that meets a nonzero coefficient propagates as numerics E2 and E3 say. A zero of a dense operand is an ordinary IEEE zero. So a product with one iterate operand does not change when stored zeros are dropped and does not depend on the order of the arguments. The rule applies to the iterate object itself and to the `matrixview(X)` products of FN-2 item (5), not to other objects built from it (FN-3). Three-factor products are a separate, open decision (FN-10). | 0.6 target (breaking; FD-9 amends numerics M1) | S, A, O |
| FN-2 | The methods, exactly: (1) `dot(v::StridedVector, x::SparseIterate{T,1})`, `dot(x::SparseIterate{T,1}, v::StridedVector)` and `dot(x::SparseIterate{T,1}, y::SparseIterate{T,1})`; (2) `dot(v, x::AtomIterate)` for `v` a `StridedVector`, a `SparseIterate{T,1}` or an `AtomIterate`, and `dot(x::AtomIterate, v)` for `v` a `StridedVector` or a `SparseIterate{T,1}`; (3) `dot(a::StridedVector, X::SparseIterate{T,2}, b::StridedVector)`; (4) `mul!(out, A, x)`, its five-argument form and `A * x`, for `x::SparseIterate{T,1}` with `A` a `StridedMatrix`, a transposed or adjoint `StridedMatrix`, or a `Symmetric` strided matrix, for `x::SparseIterate{T,2}` with `A` a `StridedMatrix` or a `Symmetric` strided matrix, and for `x::AtomIterate` with `A` a `StridedMatrix` or a `Symmetric` strided matrix; (5) for the owner, in their 0.6 form (#388), `mul!(y, matrixview(X), v)` and its five-argument form, and `matrixview(X) * v`, which reaches that `mul!` (`src/sparse_iterate.jl:996–997`); `dot(a::StridedVector, X, b::StridedVector)`; and ORC-10's `dot(M::StridedMatrix, X)` and `dot(X, M)`. A sparse-aware `dot` on the view, if added as PERF-8 recommends, is a package method and joins this item. | 0.6 target | S, A, O |
| FN-3 | **The boundary.** A product that ProximalBase does not define (`sparse(A) * x`, `Diagonal(d) * x`, `dot(sparsevec, x)`, `dot(view(v, idx), x)` for a non-strided view, `transpose(A) * x` for an `AtomIterate`, `A * X * B` with a plain matrix `A`, `sum(x .* v)`) sees the iterate as the dense array of its entries and follows IEEE arithmetic. The boundary is drawn by the object a call returns. Base returns `x` itself for `vec(x)` of a vector and for the full-length `reshape(x, n)`, so those keep the iterate's own semantics (FN-1, FX-4): with `s = SparseIterate([0., 1.])`, `dot([NaN, 1.], vec(s)) == dot([NaN, 1.], s) == 1.0`, one and the same dispatch. Every other object built from an iterate is an ordinary vector or matrix, as CRD-4 makes the owner's `θ`: `view(x, :)` and other views, `atom(x, k)`, `coordinates(x)`, `activevalues(x)`, `vec(x)` of a matrix (a Base `ReshapedArray`), a `reshape` to another shape, and `matrixview(X)` outside FN-2 item (5). Their products follow IEEE arithmetic, also where ProximalBase defines a faster method for them, such as `dot` of an `atom` view (`src/atom_iterate.jl:354–357`). So do the matrix–matrix `mul!(C, matrixview(X), B)` (LinearAlgebra's generic method) and `dot(M, matrixview(X))` (LinearAlgebra's generic `dot`); the 0.5.x matrix method at `src/sparse_iterate.jl:1928` is on the owner, not on the view. The rule says so; the difference is not a gap. | 0.6 target (the behavior is 0.5.x, except FC-3's new view) | S, A |
| FN-4 | The class of the result of an FN-2 method *must not* depend on whether a zero is stored or on the order of the arguments. `dot` and `mul!` scan no operand for non-finite values. | 0.6 target | S, A, O |
| FN-5 | A value or gradient computed through an FN-2 method at an iterate argument inherits FN-1. Sums that function values take over an iterate's coefficients with linear coefficients (`_scaled_dot` for the normals of linear sets, `src/sparse_iterate.jl:157–176` and `src/atom_iterate.jl:48`; `_split_dot` in `LogDetLoss`, `src/sparse_iterate.jl:178–201`) need no rule of their own, because M1's existing sentence covers them: the coefficients are finite (E4; normals are checked at construction, `src/functions/sets.jl:690`, and `LogDetLoss` takes that path only for a finite `S`, `src/functions/logdet.jl:110`), so skipping a zero is exact. Of the data losses checked, `LeastSquaresLoss`, `GLMLoss` (beyond its weights and offset) and `QuadraticFunction` do not check that their data are finite, and `KronQuadratic` does; so with a non-finite design entry `f(x)` at an iterate can be finite where `f(Vector(x))` is NaN. | 0.6 target (0.5.x for unstored zeros) | S, A |
| FN-6 | **Activation from the converted value** (CRD-3 for the family): every value write, `fill!` included, *must* decide activation after conversion. `fill!(AtomIterate(Float32, 4, 2), 1e-100)` activates nothing. | 0.6 target (bug fix for the `AtomIterate` `fill!`) | S, A, O |
| FN-7 | Non-finite values are values: NaN and `±Inf` activate their coordinate, and `dropzeros!` keeps them. | 0.5.x | S, A |
| FN-8 | **Wrappers.** A write through an object that wraps an array iterate and is not `x` itself (`view(x, …)`, `vec(x)` of a matrix, a `reshape` to another shape, `atom(x, k)`) *must* behave like the same write through the same wrapper of a plain array of element type `T`, plus FW-2's activation; a write through `coordinates(x)` *must* behave as on a plain vector of element type `T` holding the same entries, plus FW-2's activation. Base's conversion applies: a complex value throws `InexactError`, possibly after a partial write (#364). `vec(x)` of a vector and `reshape(x, length(x))` are `x` itself and get FX-4's check before writing. | 0.6 target (stated; the behavior is 0.5.x) | S, A |
| FN-9 | **Scaling by a number.** `rmul!(x, s)`, `lmul!(s, x)`, `rdiv!(x, s)` and `ldiv!(s, x)` *must* be IEEE value writes equal to `x .*= s` (respectively `x ./= s`) in values and in working set; FN-1 does not apply, since the other operand is one scalar. *Recommended:* loop over the active coordinates (`O(s)`, or `O(active entries)` for A) when the scalar is finite and, for a division, nonzero, which is exact because `0 · s == 0` there; otherwise write every entry and activate per FW-2. | 0.6 target (breaking for S: `rmul!(x, Inf)` on `[1, 0, 0, 0]` with an explicit zero at 2 gives `[Inf, NaN, 0, 0]` today, `x .*= Inf` gives `[Inf, NaN, NaN, NaN]`) | S, A |
| FN-10 | **Three-factor products** `*(A::Symmetric, X::SparseIterate{T,2}, B::Symmetric)` and `*(A::Symmetric, X::SymmetricSparseIterate{T}, B::Symmetric)`: their class guarantee is decided separately, in FD-16. Recommended there: every finite entry of the ordinary BLAS path is kept, and only the non-finite entries are recomputed term by term with the zeros of `X` strong; finite partial products that overflow follow A1 (the last sentence of numerics R2). | open (FD-16) | S, O |

## 2.7 Coordinate oracles (FA)

Shared by both forms of `prox_coordinate!` and by the four-argument
`kkt_violation`.

| ID | requirement | status | types |
|:--|:--|:--|:--|
| FA-1 | `k` *must* be a coordinate ID (FC-1). | 0.5.x | S, A, O |
| FA-2 | **Return.** `prox_coordinate!` *must* return the new scalar for S and O, and `atom(x, k)` for A. | 0.5.x | S, A, O |
| FA-3 | **Activation.** A coordinate that was absent and whose result is zero *must* stay absent; no coordinate is removed. | 0.5.x | S, A, O |
| FA-4 | **Validation before writing:** the bounds of `k` (`BoundsError`), the layout (FL-4), a separable function or a group penalty (`ArgumentError`), the parameters, a real `u` (`ArgumentError`) and the length of `u` (`DimensionMismatch`). After any of these errors `x` *must* be unchanged. | 0.5.x | S, A, O |
| FA-5 | **Aliasing.** The trial `u` may share memory with `x`, for example `atom(x, j)`; it is copied first. | 0.5.x | A |
| FA-6 | **Screening.** `kkt_violation(f, g, x, k)` *must* be a screen at zero. For a group penalty at an inactive group `k` of an `AtomIterate` it is the penalty's own: the dual norm of that penalty's group norm at `g[G_k]`, minus `λ₀ w_k` with that penalty's weights (table below). For a separable `f` on a group, it is the largest entry screen over the group. It is not a stationarity residual (#388 §2.8). | 0.5.x | S, A, O (ORC-9) |
| FA-7 | **Block trials.** `prox_coordinate!(x, f, k, u, γ)` *is* the supported way to take a group step at a trial point. It *must* give the result of staging `u` into group `k` and calling the four-argument form when nothing throws, and leave `x` unchanged after any of FA-4's validated errors. If reading `u` throws (a non-strided `u` whose `getindex` fails), FW-1 holds but entries of the group may already be written. The alternative, copying a non-strided `u` first, would make that atomic too, at the cost of an allocation. Staging a trial by writing private storage is unsupported: it breaks FW-1 for the duration of the call, and for good if the call throws. | 0.5.x (stated) | A |

The screens of the built-in group penalties (each checked at the audited
commit; the separable case is the generic method, `src/interface.jl:672–673`):

| penalty | dual norm at `g[G_k]` | default weight `w_k` | screen | source |
|:--|:--|:--|:--|:--|
| `GroupNormL2(λ₀, groups)` | `‖g[G_k]‖₂` | `√\|G_k\|` | `‖g[G_k]‖₂ − λ₀ w_k` | `src/functions/norms.jl:442–443`, `:498`, `:742–743` |
| `GroupNormLinf(λ₀, groups)` | `‖g[G_k]‖₁` | `1` | `‖g[G_k]‖₁ − λ₀ w_k` | `src/functions/norms.jl:444–445`, `:499`, `:742–743` |
| `SparseGroupNormL2(λ₁, λ₀, groups)` | `‖S_{λ₁}(g[G_k])‖₂`, the soft threshold at `λ₁` | `√\|G_k\|` | `‖S_{λ₁}(g[G_k])‖₂ − λ₀ w_k` | `src/functions/sparsegroup.jl:60–63`, `:96–100`, `:318–319` |
| `GroupMCP`, `GroupSCAD` `(λ₀, groups; a)` | `‖g[G_k]‖₂` | `√\|G_k\|` | `‖g[G_k]‖₂ − λ₀ w_k` | `src/functions/nonconvex.jl:362–363`, `:460`; `src/functions/norms.jl:742–743` |
| `Scaled(f, α)` | — | — | `α` times the screen of `f` at `g / α` | `src/calculus.jl:211–214` |
| `Regularize(f, ρ, c)` | — | — | the screen of `f` at `g − ρc` | `src/composition.jl:657`, `:661–663` |
| a separable `f` | — | — | `max` over `i ∈ G_k` of `kkt_violation(f, g, i)` | `src/interface.jl:672–673` |

`IndBallGroupL0` has no screen (`MethodError`). The screens differ in sign
where it matters: at `g = [5/8, 5/8]` with `λ₀ = 1`, `GroupNormLinf`'s screen
is `0.25`, so the group must enter, while the ℓ₂ formula with its unit weight
gives `−0.11611652351681556` (and `GroupNormL2`'s own screen, with weight `√2`,
gives `−0.5303300858899107`).

## 2.8 Performance and capacity (FP)

| ID | requirement | status | types |
|:--|:--|:--|:--|
| FP-1 | After construction, an entry read or write, `isactive`, `activate!` and the first activation of a coordinate *must* be worst-case `O(1)` and *must not* allocate after warm-up. | 0.5.x | S, A (O: PERF-1) |
| FP-2 | Traversal of the working set through `activecoordinates`, `activevalues`, `atom` and `atomindices` *must* cost `O(s)` (S, O) or `O(active entries)` (A). Building these handles and `coordinates(x)` *must not* copy or allocate. | 0.6 target | S, A, O |
| FP-7 | *Recommended:* an `O(s)` method for `fill!` with a zero on both coordinate views (the `SparseIterate` view and the owner's `SymmetricCoordinates`). It is the same value write, with a faster loop: zeros go only to the active coordinates, the absent ones being zero already. Then the owner's clear `fill!(coordinates(X), 0); dropzeros!(X)` costs `O(s)`, as 0.5.x's `fill!(X, 0)` does (`src/sparse_iterate.jl:1574–1584`, through `_clear!`); without it, Base's generic `fill!` (`array.jl:356`) costs `O(q)`. ProximalSolvers clears an owner this way at every Newton iteration (`src/cd/quic.jl:91`). | recommendation | S, O |
| FP-3 | **Capacity is dense-sized.** Construction reserves room for every coordinate, so activation never grows storage, and construction costs `O(n)`, `O(p + K)` or `O(q)`. Retained memory of an empty iterate: S, three arrays of length `n`, 24 bytes per entry for `Float64` (20 for `Float32`); A, the dense values and the entry-to-group map (16 bytes per entry for `Float64`), two `O(K)` maps of the active groups (16 bytes per group) and the layout (16 bytes per range group, or 8 bytes per index plus a vector header per vector group); O, `O(q)`, 24 bytes per coordinate (PERF-2). "Sparse" promises sparse *work*, not sparse *memory*. | 0.5.x (stated) | S, A, O |
| FP-4 | *Recommended:* add another storage backend (hash maps, growing storage) only for a motivating workload, with a benchmark that shows the gain. | recommendation | S, A, O |
| FP-5 | `mul!(out, A, x)` *must* cost `O(m s)` for a `SparseIterate` vector and `A` with m rows (for a matrix `x`, the stored-entry walk or the column walk, switched at density 1/32 as today), `O(m · active entries)` for an `AtomIterate`, and `dot(v, x)` `O(s)` or `O(active entries)`. FN-1 adds no pass. | 0.5.x | S, A |
| FP-6 | Each 0.6 change on a hot path is measured against the 0.5.x baseline: sweeps through the read-only handle; indexing and products through the `coordinates` view; the strong-zero branch of `dot` and `mul!`; the five-argument block trial against staged writes; the momentum loop through `atomindices`; `atomindices` iteration; `unalias` copies; scaling by a number; and, if FD-16 adopts an option that changes the triple product, its cases. A repeatable slowdown above 5% on a representative case *must* be investigated and reported on the implementing pull request, and merging it needs the maintainer's recorded approval (numerics C3). The companion's protocol measures each case. | 0.6 target | S, A, O |

## 2.9 Fields (FF)

| ID | requirement | status | types |
|:--|:--|:--|:--|
| FF-1 | The fields of `SparseIterate` (`nzval`, `nzval2ind`, `ind2nzval`, `nnz`) and of `AtomIterate` (`storage`, `groups`, `group_of`, `layout_cache`, `active`, `position`, `nactive`) are private: a field read or write is not covered by this contract and may break in any release. | 0.6 target (stated) | S, A |
| FF-2 | Every audited downstream use of a private field *must* have a supported replacement: `x.groups[k]` by `atomindices(x, k)`; whole-layout reads by `similar(x)`, a loop over `atomindices` or `ProximalBase.samegroups`; `x.nzval[1:n]` by `activevalues(x)`; `x.active` and `x.nactive` by `activecoordinates(x)` and `nactive(x)`; staging in `x.storage` by FA-7. The companion (§3) gives the recipes. | 0.6 target | S, A |

Not covered: the fields of group penalties (`f.groups`, `f.group_of`, `f.w`),
which ProximalSolvers also reads. They belong to the penalty API (#342 for the
owner).

---

# Part 3. Decisions for maintainer sign-off

## 3.1 The scale of the design

| | minimal | **moderate (recommended)** | ambitious |
|:--|:--|:--|:--|
| contents | bug fixes only: read-only handles, `dataids`, converted `fill!`, the real check in `copyto!`, one product rule | the rules of Part 2; no new types except read-only wrappers | separate value, coordinate and state types; a public `GroupLayout`; a shape-aware `AtomIterate{T, N}`; non-array owners for all three; other storage backends |
| what it fixes | corruption, aliasing, conversion and complex-source defects | those, and the form-dependent working set, branch-dependent destinations, storage-dependent products and private-field use | those, plus layout identity and matrix-shaped groups |
| what breaks | almost nothing | `fill!(x, 0)`, `x .= y`, matrix `coordinates`, the type of `activecoordinates`, NaN against zero coefficients, field access | most iterate code downstream, `AbstractArray` dispatch included |
| cost | none | a few wrappers and rules; benchmark gates (FP-6) | new types on every hot path; a migration of every penalty |

*Why moderate.* The minimal design leaves the working set dependent on the
spelling of a write (Example 1) and products dependent on storage
(Example 2), which are the faults that make the family incoherent. The
ambitious design has no correctness case that the moderate one misses:
ordinary array traversal is already correct for the two array iterates
(Part 1), the layout can be made safe without a new public type (FD-11), the
one audited workload with matrix-shaped groups (ProximalSolvers'
multi-response model) is served by `vec(B)` and its own model (FD-15a), and no
audited workload needs another backend (FP-4).

*How to sign off.* The decisions fall into three groups, each signed on its
own:

1. **The structural decisions:** FD-2 to FD-8, FD-10 to FD-15a, and the
   table of §3.2. Nothing in them depends on FD-9 or FD-16, so they can be
   signed without either.
2. **FD-9**, strong zeros in products with one iterate operand, with its
   numerics amendment.
3. **FD-16**, three-factor products, an open decision with its own options.

## 3.2 Status quo affirmed

These need no change of behavior. One sign-off covers the table. FD-15a
(shape-aware groups) has a record of its own in §3.3.

| item | decision | requirement |
|:--|:--|:--|
| FD-1. Supertypes | `SparseIterate` stays an `AbstractArray` and `AtomIterate` an `AbstractVector`; there is no correctness case for removing them (#400, guardrail 2) | FS-1 |
| FD-8. No group `activevalues` | it stays undefined. Its only audited consumer is the momentum history (ProximalSolvers `src/cd/solver.jl:553–600`), which reads private fields today and can loop over `activecoordinates` and `atomindices` instead at the same `O(active entries)` cost (companion §3.4); the loop is a benchmark case | FC-9 |
| FD-15b. A family-wide value-view name | none; `matrixview` stays owner-only | FS-5 |
| FD-15c. `OptimizationResults` needs an `AbstractArray` minimizer (ProximalSolvers `src/types.jl:210`) | it affects only the owner and joins #391 Phase 2 | — |
| FD-15d. Private fields | listed, with a supported replacement for each audited use: 23 indexed reads of `x.groups[k]`, 5 whole-layout reads, and 16 other field uses in `src/cd/solver.jl:553–600` and `src/cd/groups.jl:32,64` (ProximalSolvers) | FF-1, FF-2 |

*Sign-off:* ☐ maintainer.

## 3.3 Decision records

### FD-2. One working-set rule for every write

*Status: recommended, awaiting maintainer sign-off.*

*Background.* Both array iterates write through the array interface
(`setindex!` at `src/sparse_iterate.jl:478` through `_setstored!`, `:299–309`;
`src/atom_iterate.jl:140–145`). Base sends bulk writes to `fill!` or to
`copyto!` (table in §2.3), and the iterates specialize three of these:
`fill!` with a zero clears the working set
([`src/sparse_iterate.jl:566–576`](https://github.com/mlakolar/ProximalBase.jl/blob/def81ec9d021c39ed1727cfeb9598357291dd60d/src/sparse_iterate.jl#L566-L576),
[`src/atom_iterate.jl:272–282`](https://github.com/mlakolar/ProximalBase.jl/blob/def81ec9d021c39ed1727cfeb9598357291dd60d/src/atom_iterate.jl#L272-L282));
a same-kind `copyto!` is a state copy (`src/sparse_iterate.jl:551–554`,
`src/atom_iterate.jl:248–260`); and broadcasts call Base's generic method after
a real check (`src/sparse_iterate.jl:556–559`, `src/atom_iterate.jl:262–265`),
which takes Base's identity shortcut to `copyto!`.

*What goes wrong today* (ran; Part 4). The working set depends on how a write
is spelled. From two active coordinates, `fill!(x, 0)` and `x .= 0` leave
`nactive == 0`, while `x .= 0 .* x`, `rmul!(x, 0)` and `x .*= 0` leave 2 (both
types). With `x` holding coordinate 1, and `y` holding coordinate 3 and an
explicit zero at 2, `x .= y` gives `y`'s working set `[3, 2]`, while
`x .= 1 .* y`, the explicit loop and `x[:] = y` give `[1, 3]`; an
`AtomIterate` gives `[2, 1]` against `[1, 2]`.
`sum!(y, M)` drops a coordinate through `fill!`; `mul!(y, A, b)` keeps it.

*Options.*

- **(a) An import from an array replaces the working set** (#388 CPY-4,
  extended to the family).
  - *Correctness:* `copyto!(x, A)` alone gives the support of `A`. But Base
    sends `x .= v`, `copy!`, `ldiv!` and `permute!` to `copyto!`, while
    `x[:] = v` and offset `copyto!` stay value writes, so the result depends
    on the spelling again (worked example).
  - *Downstream:* ProximalBase's oracles write their destinations with
    `copyto!(y, x)` at 35 sites in `src/` (for example
    `src/functions/norms.jl:63`); a dense argument would then replace `y`'s
    working set in some branches and not in others (FD-4). The one audited
    replacing import (ProximalSolvers `src/workspaces.jl:243–246`) is written
    as `fill!` plus a scatter and would not use `copyto!` anyway.
  - *Numerics:* none; values are the same in every option.
  - *Performance:* an `O(s)` clear before each import.
- **(b) Value writes everywhere** (recommended): FW-2 to FW-6.
  - *Correctness:* one rule; Base's routing becomes harmless, because every
    entry point is a value write; replace and clear have explicit spellings.
  - *Downstream:* `fill!(x, 0)` no longer clears. ProximalSolvers
    `src/cd/path.jl:261` (the start of a continuation) and
    `src/workspaces.jl:243` (loading a dense start into a CD workspace) rely on
    the clear; without `dropzeros!` they start from the caller's previous
    working set and no error is raised. `x .= y` is no longer a state copy (no
    audited site uses it on iterates). `copyto!(SparseIterate(2, 2),
    SparseIterate(4))` is accepted (today `DimensionMismatch`).
  - *Numerics:* none.
  - *Performance:* `fill!(x, 0)` writes zeros to the `s` active coordinates
    instead of clearing them, `O(s)` both. `x .= y` between two
    `SparseIterate`s becomes an `O(n)` loop instead of the `O(s)` state copy;
    a caller who wants the state copy writes `copyto!(x, y)`. Both are
    benchmark cases.
- **(c) Value writes, except that `fill!` with a zero clears** (today's
  `fill!`).
  - *Correctness:* `x .= 0` (through `fill!`) and `sum!` clear, `x .= 0 .* x`
    does not: the form dependence stays for zeros.
  - *Downstream:* `path.jl:261` and `workspaces.jl:243` unchanged.
  - *Numerics, performance:* as (b).
- **(d) Status quo.** All of the form dependence above.
- **(e) Reject a copy between `AtomIterate`s of different layouts**, with
  `DimensionMismatch`, instead of today's value write.
  - *Correctness:* `copyto!` between two `AtomIterate`s would always be a
    state copy, so the same values could not end in different working sets
    depending on whether the layouts are equal (FX-2).
  - *Downstream:* no audited site copies across layouts; `reinit!` checks the
    partition before it copies (ProximalSolvers `src/workspaces.jl:152–153`).
  - *Numerics:* none.
  - *Performance:* none; FL-6's comparison is made today anyway.
  - *Weighed:* an `AtomIterate` is an `AbstractVector`, and generic code that
    copies one vector into another of the same length would then throw. (b)
    keeps Base's meaning and states the switch in FX-2; a caller who wants the
    state copy checks `ProximalBase.samegroups` first. (b) stays recommended.
- **(f) A named, checked replacement for the array iterates** (an additive
  function, name to be chosen), which validates the source (length, real
  element type) and handles an overlapping source before it clears `x`.
  - *Correctness:* one call that cannot lose `x`'s state, unlike the
    shorthand of FW-6 with an overlapping or invalid source.
  - *Downstream:* none; it adds a name.
  - *Numerics:* none.
  - *Performance:* `O(n + s)`, plus a copy of an overlapping source.
  - Not recommended now: the shorthand with FW-6's safe migration covers the
    audited uses (§3.5 of the companion); the function can be added later
    without a break.

*The worked example.* `x = SparseIterate(4); x[4] = 7; v = [3, 0, 0, 0]`.
The values are `[3, 0, 0, 0]` after each call, except `permute!`, which gives
`[7, 0, 0, 0]`. The working set:

| call | today (ran) | (b), recommended | (a), import replaces |
|:--|:--|:--|:--|
| `copyto!(x, v)` | `[4, 1]` | `[4, 1]` | `[1]` |
| `copy!(x, v)` | `[4, 1]` | `[4, 1]` | `[1]` (Base: `copyto!`) |
| `x .= v` | `[4, 1]` | `[4, 1]` | `[1]` (Base: `copyto!`) |
| `x[:] = v` | `[4, 1]` | `[4, 1]` | `[4, 1]` |
| `copyto!(x, 1, v, 1, 4)` | `[4, 1]` | `[4, 1]` | `[4, 1]` |
| `ldiv!(x, lu(I₄), v)` | `[4, 1]` | `[4, 1]` | `[1]` (Base: `copyto!`) |
| `permute!(x, [4, 1, 2, 3])` | `[4, 1]` | `[4, 1]` | `[1]` (Base: `x .= x[p]`) |
| `fill!(x, 0)`, `x .= 0` | `[]` | `[4]` | `[4]` (a value write in (a) too) |

With `y = AtomIterate(4, [1:2, 3:4]); y[4] = 7` the same calls give `[2, 1]`
today and under (b), and `[1]` under (a); `fill!` gives `[]` today (ran).
Today's behavior *is* option (b) for every import from an array; (b) changes
only the clearing rows, `x .= y` and the other-shape `copyto!`.

*Recommendation and reason.* (b). Its working set does not depend on the
spelling of a write, with one exception that Base's routing creates and FW-2
states: `ldiv!(x, F, y)` with a same-kind iterate `y` reaches `copyto!(x, y)`
and is a state copy (ran: working set `[1]`, against `[4, 1]` for
`ldiv!(x, F, Vector(y))`). It is already today's behavior for every import
from an array, and the two operations that remove coordinates become explicit
at no extra cost.

*Kind:* breaking 0.6.

*Migration.*

| site | 0.5.x | 0.6 |
|:--|:--|:--|
| ProximalSolvers `src/cd/path.jl:261` | `fill!(x, zero(T))` | `dropzeros!(fill!(x, zero(T)))` for S and A; `fill!(coordinates(x), zero(T)); dropzeros!(x)` for the owner |
| ProximalSolvers `src/workspaces.jl:243–246` | `fill!(xs, 0)`, then a scatter of the nonzeros of `ws.x` | `copyto!(dropzeros!(fill!(xs, zero(eltype(xs)))), ws.x)` |
| ProximalSolvers `src/cd/quic.jl:91` (owner) | `fill!(D, zero(T))` | `fill!(coordinates(D), zero(T)); dropzeros!(D)` |
| a state copy written as `x .= y` | `x .= y` | `copyto!(x, y)` |
| a replacing import | `fill!(x, 0); copyto!(x, A)` | `copyto!(dropzeros!(fill!(x, 0)), A)` for a source independent of `x` that passes the checks; otherwise validate, then `A = Base.unalias(x, A)`, then the shorthand (FW-6) |

*Relation to #388/#391:* consistent extension of CRD-3, CRD-5 and ACT-5 to
the family. For the owner, `copyto!` from a matrix stays CPY-4's checked replace (FD-3); FW-6 states the difference and its reason.

*Sign-off:* ☐ maintainer.

### FD-3. Matrix import into the owner keeps CPY-4

*Status: recommended, awaiting maintainer sign-off.*

*Background.* #388 CPY-4 and decision 4 make `copyto!(X, M)` a checked import
that *replaces* the working set. At the audited commit the owner's
`copyto!(X, M)` (`src/sparse_iterate.jl:1128`, `_copyto_lower!` at
`:1151–1161`) checks first (real type at `:1152`, axes, exact symmetry, an
aliasing copy) and then writes each lower-triangle entry with `_setstored!`:
a value write that keeps the old working set (ran: with `X[2, 2] = 7`,
`copyto!(X, [3 0; 0 0])` leaves the working set `[3, 1]`).

*What goes wrong with a value write.* If `copyto!(X, M)` kept the working set,
a replace would have to be spelled as clear-then-import, and the clear runs
before the import's checks. Ran: with `X[1, 1] = 7` and `M = [1 2; 3 4]`, a
checked `copyto!(X, M)` throws `ArgumentError` (the matrix is not symmetric)
and keeps `X` (`[7 0; 0 0]`, working set `[1]`), while
`fill!(coordinates(X), 0); dropzeros!(X); copyto!(X, M)` throws the same error
after emptying `X` (`[0 0; 0 0]`, working set `[]`).

*Options.*

- **(a) Keep CPY-4: a checked, atomic replace** (recommended). The axes,
  exact symmetry on the source values (VAL-1, VAL-2), a real source and CPY-6
  for an overlapping source are all checked before any write; then the
  working set is the converted support of `M` in increasing ID order.
  - *Correctness:* a failed import leaves `X` as it was, and the result
    depends only on `M`.
  - *Downstream:* no audited site imports a matrix into an owner with
    `copyto!` (ProximalSolvers `src/cd/quic.jl:234` copies *into* a dense
    matrix; refits copy owner to owner).
  - *Numerics:* none.
  - *Performance:* `O(q)` for a dense source, read once, as CPY-4 says; the
    previous working set is cleared in `O(s)` after the checks.
- **(b) A checked value write, with replace spelled as clear-then-import**
  (rejected).
  - *Correctness:* one meaning of `copyto!` from a plain array across the
    family, but the replace spelling destroys the state before the checks run
    (the example above).
  - *Downstream:* none; this is today's behavior.
  - *Numerics:* none.
  - *Performance:* `O(q)`; a replace adds `O(s)`.

*Recommendation and reason.* (a): a replace that validates before it writes
is the only one that cannot lose the caller's state, and #391 already plans
it. The family therefore has two meanings of `copyto!` from a plain array, and
FW-6 states why: generic Base code reaches the array iterates' `copyto!` and
never the owner's.

*Kind:* breaking 0.6 for the owner, as #391 already planned (0.5.x's
`copyto!(X, M)` is a value write).

*Migration.* Code that relied on 0.5.x keeping the working set writes the
values through the coordinate view (`θ .= …`, CRD-5) or loops over
`activecoordinates`. The owner's replace is `copyto!(X, M)`; its clear is
`fill!(coordinates(X), 0); dropzeros!(X)`.

*Relation to #388/#391:* unchanged: CPY-4, decision 4, ACT-5, T06, T10 and
the §2.2(e) comment of the owner's acceptance guide stand as written.

*Sign-off:* ☐ maintainer.

### FD-4. Oracle destinations

*Status: recommended, awaiting maintainer sign-off.*

*Background.* ProximalBase's proximal oracles write a destination with
`copyto!(y, x)` (the spelling occurs at 35 sites in `src/`, for example
`src/functions/norms.jl:63, 256, 696, 799`, `src/functions/sparsegroup.jl:271`,
`src/functions/nonconvex.jl:586, 725`), with a broadcast
(`src/functions/norms.jl:74`), with `fill!(y, 0)` (`:77`, `:262`, `:324`), or
by writing stored values (`src/functions/sets.jl:162–190`).

*What goes wrong today* (ran). With `y` holding coordinate 4,
`prox!(y, NormL2(λ), [3, 0, 0, 0], 1)` leaves the working set `[4, 1]`,
`[4, 1]` and `[]` for λ = 0, 1 and 10. With an `AtomIterate` and
`GroupNormL2` it leaves `[2, 1]`, `[2, 1]` and `[2]`. `prox!(y, IndBallL0(0),
x, 1)` keeps `y`'s `[4]`, while `IndBallL0(1)` gives the argument's `[1, 2]`.
The values are right in every case.

*Options.*

- **(a) Unspecified beyond FW-1** (recommended): values are the result, every
  nonzero result is active, `dropzeros!` gives the support.
  - *Correctness:* enough for solvers, which need FW-1 and `dropzeros!`.
  - *Downstream:* ProximalSolvers compacts after proximal steps
    (`_drop!` in `src/solvers/proximal_gradient_descent.jl:158`) and at the
    start of each active-set iteration
    (`src/solvers/active_accelerated_proximal_gradient_descent.jl:150`); no
    audited site depends on which zeros remain.
  - *Numerics:* none. *Performance:* none.
- **(b) The old set plus the new nonzeros.**
  - *Correctness:* deterministic.
  - *Downstream, performance:* each of the 35 `copyto!(y, x)` sites with a
    same-kind `x` would have to become a merging copy (`O(s_x + s_y)`) in
    place of the state copy.
- **(c) Replace: the working set is exactly the support.**
  - *Correctness:* deterministic.
  - *Performance:* one `O(s)` compaction per call (`O(active entries)` for
    A), paid even by callers that compact anyway.

*Recommendation and reason.* (a). The stronger rules buy a guarantee no
audited caller uses, at the cost of code at 35 sites or a pass per call.

*Kind:* 0.5.x behavior, now stated.

*Migration.* None. A caller that needs the support calls `dropzeros!`.

*Relation to #388/#391:* consistent with ORC-3 and ORC-6.

*Sign-off:* ☐ maintainer.

### FD-5. Coordinates of a SparseIterate

*Status: recommended, awaiting maintainer sign-off.*

*Background.* `coordinates(x::SparseIterate) = x` (`src/sparse_iterate.jl:480`);
its docstring calls it "a vector view of the coordinates" (`:262–270`).

*What goes wrong today* (ran). For `SparseIterate(Float64, 2, 3)`,
`coordinates(x)` is the 2 × 3 matrix itself, of size `(2, 3)`, not an
`AbstractVector`. Code written for the owner's coordinate vector (CRD-1: an
`AbstractVector` of length `ncoordinates`) gets a matrix, so a loop
`for k in axes(coordinates(x), 1)` visits 2 coordinates, not 6. Linear
indexing `coordinates(x)[k]` works, which is why the audited callers are
unaffected.

*Options.*

- **(a) A coordinate-vector view for both shapes** (recommended): FC-3. A thin
  `AbstractVector{T}` wrapper of `x`, like the owner's `SymmetricCoordinates`,
  indexed by coordinate ID; never `x` itself.
  - *Correctness:* one behavior for every shape and for the owner's `θ`. The
    view is an ordinary vector (FN-3): writes are value writes (FW-2), and
    products are IEEE products. `copyto!(coordinates(x), coordinates(y))` is a
    value write for both shapes, and `dot(v, coordinates(x))` has one class
    for both.
  - *Downstream:* the callers that can receive a `SparseIterate`
    (ProximalSolvers `src/cd/drivers.jl:95, 115, 150` and
    `src/cd/solver.jl:648`) index with coordinate IDs and are unaffected; the
    other callers take only owners. `coordinates(x) === x` no longer holds for
    a vector; no audited caller relies on it.
  - *Numerics:* the view has no check of its own and follows FN-8, as the
    owner's `SymmetricCoordinates` does today (ran: `coordinates(X) .=
    ComplexF64[1, 2im, 0]` throws `InexactError` and leaves `[1.0, 0, 0]`).
    For a vector this is a change: today `coordinates(xv) .= ComplexF64[1, 2im,
    0, 0]` throws `ArgumentError` before writing, because the view is `xv`.
  - *Aliasing:* with `dataids` those of `x` and an `unaliascopy` that returns
    a view of a copy of `x`, `copyto!(x, coordinates(x))` leaves `x`
    unchanged and `copyto!(x, 2, coordinates(x), 1, 3)` turns `[1, 2, 3, 4]`
    into `[1, 1, 2, 3]` (ran with a stand-in view); without `unaliascopy`
    Base throws `ArgumentError` ("must make a preventative copy").
  - *Performance:* `O(1)` to build, no copy, `Base.dataids` those of `x`.
    Indexing must inline; a product through the view is a generic `O(n)`
    loop, not the sparse method of `x`. Both are benchmark cases.
- **(b) `coordinates(x) = vec(x)`**: the object itself for a vector, Base's
  linear reshape for a matrix.
  - *Correctness:* an `AbstractVector` for every shape, but its behavior
    depends on the shape (all ran). Products: `dot([1, NaN, 0, 0],
    coordinates(xv))` with an unstored zero at 2 is `1.0` through the
    package's method, while the same call on `vec(xm)` is NaN, and
    `norm(vec(xm))` is LinearAlgebra's generic `O(n)` `norm`. Copies:
    `copyto!(coordinates(x), coordinates(y))` is a state copy for N = 1
    (working set `[1]`) and a value write for N = 2 (`[4, 1]`). Complex input:
    `coordinates(xv) .= ComplexF64[1, 2im, 0, 0]` throws `ArgumentError`
    before writing, while `vec(xm) .= ComplexF64[1, 2im, 0, 0]` writes `1.0`
    and then throws `InexactError`.
  - *Downstream:* as (a), and `coordinates(x) === x` stays true for a vector.
  - *Numerics:* after FD-9, `dot(v, coordinates(x))` would have different
    classes for N = 1 and N = 2.
  - *Performance:* `O(1)`, no allocation. One exploratory run of a read-write
    sweep through `vec(m)` on a 300 × 300 iterate took 1.31 ms against
    1.248 ms through `m` (ratio 1.049, 0 bytes allocated).
- **(a′) As (a), with a real-input check before writing on both coordinate
  views** (the `SparseIterate` view and the owner's `SymmetricCoordinates`).
  - *Correctness:* a complex source would throw `ArgumentError` before any
    write, as `x .= complex` does today.
  - *Downstream:* none (no audited code passes complex data).
  - *Performance and cost:* `copyto!` and broadcast methods on both views,
    with the methods that resolve their ambiguities with SparseArrays, as
    FD-13 describes for the iterate's own `copyto!`.
- **(c) Status quo.** The family has two meanings of `coordinates`.

*Recommendation and reason.* (a): one behavior for both shapes, the owner's
model (CRD-1, CRD-4) for the whole family, at the cost of one small type.

*Kind:* breaking 0.6 for both shapes (`coordinates(x) === x` for a vector,
the matrix itself for N = 2).

*Migration.* Code that used `coordinates(x)` as `x` (its sparse methods, or
a matrix) uses `x`. Linear indexing `coordinates(x)[k]` is unchanged.

*Relation to #388/#391:* consistent extension of CRD-1 and CRD-4.

*Sign-off:* ☐ maintainer.

### FD-6. Groups and entries of an AtomIterate

*Status: recommended, awaiting maintainer sign-off.*

*Background.* An `AtomIterate` has no `coordinates` (a coordinate is a group;
ran: no method). Its layout is the private field `groups`
(`src/atom_iterate.jl:32`). `atom(x, k)` is `view(x, groups[k])`
(`:163–166`), so `parentindices(atom(x, k))` returns the stored group.

*What goes wrong today* (ran, read). ProximalSolvers reads `x.groups` 23 times
by index and 5 times as a whole (companion §3.1). The only public route to a
group's indices, `parentindices(atom(x, k))`, returns the stored
`Vector{Int}`: after `G[1] = 3` on it, `x.groups == [[3, 2], [3, 4]]` while the
entry-to-group map still says `[1, 1, 2, 2]`.

*Options.*

- **(a) `atomindices(x, k)` and `ProximalBase.samegroups(a, b)`**
  (recommended): FC-6, FL-7.
  - *Correctness:* read-only, one group at a time; the equality question
    gets a public predicate.
  - *Downstream:* the 23 indexed reads become `atomindices(x, k)`; the 5
    whole reads become `similar(x)` (`src/cd/path.jl:10`), a loop over
    `atomindices` (`src/cd/groupcurvature.jl:124`) and `samegroups`
    (`src/cd/models.jl:691`,
    `src/solvers/active_accelerated_proximal_gradient_descent.jl:265`,
    `src/workspaces.jl:153`), all ProximalSolvers.
  - *Numerics:* none.
  - *Performance:* `O(1)` and no allocation; a range group is returned as
    stored. A vector group is wrapped, so `view(grad, G)` and `X[:, G]`
    (ProximalSolvers `src/cd/models.jl:429`, `src/cd/leastsquares.jl:586`)
    take a non-`Vector` index: a benchmark case.
- **(b) A public `groups(x)` returning the whole layout read-only.**
  - *Correctness:* read-only, but it invites identity tests on the returned
    object and fixes a layout type in the API.
  - *Downstream, performance:* as (a).
- **(c) Declare the field `x.groups` public and read-only by convention.**
  - *Correctness:* group vectors stay mutable (the corruption above).
  - *Downstream:* no change.
- **(d) `parentindices(atom(x, k))`.** It exposes the private vector.

*Recommendation and reason.* (a): it covers every audited use with the
smallest surface and keeps the layout private.

*Kind:* 0.5.x-safe addition; field reads become unsupported in 0.6 (FF-1).

*Migration.* Companion §3.1 and §3.2.

*Relation to #388/#391:* unchanged (the owner has no groups; #342 tracks
groups on an owner).

*Sign-off:* ☐ maintainer.

### FD-7. Read-only active-coordinate handles

*Status: recommended, awaiting maintainer sign-off.*

*Background.* `activecoordinates` returns writable views of internal lists:
`view(x.nzval2ind, 1:x.nnz)` ([`src/sparse_iterate.jl:287`](https://github.com/mlakolar/ProximalBase.jl/blob/def81ec9d021c39ed1727cfeb9598357291dd60d/src/sparse_iterate.jl#L287),
shared with the owner) and `view(x.active, 1:x.nactive)`
(`src/atom_iterate.jl:149`). ACT-3 already requires a read-only handle for the
owner.

*What goes wrong today* (ran). On a `SparseIterate` with active `[2, 4]`,
`activecoordinates(x)[1] = 3` leaves `x[2] == 1.0`, `isactive(x, 2) == true`,
`isactive(x, 3) == false`, and `Vector(x) == [0, 0, 1, 3]`. On an
`AtomIterate`, `activecoordinates(a)[1] = 2` leaves `active = [2, 3, 0]` and
`position = [1, 0, 2]`; `dropzeros!` then never returns
(`src/atom_iterate.jl:211–222` loops, because `_deactivate!` does nothing when
`position[k] == 0`, `:192–202`; a run was stopped by an alarm after 25 s).

*A lifetime ACT-3 does not name.* A state copy can reorder the slots without
an activation, a compaction, a change of membership or a change of
`nactive`. Ran: `x` with slot order `[1, 2]` and `y` with order `[2, 1]` hold
the same entries; after `copyto!(x, y)` the order is `[2, 1]`, `nactive` is
still 2, and a handle taken before the copy now reads coordinate 2 at its
first slot (an `AtomIterate` likewise ends with `[2, 1]`). FC-7 and FC-8
therefore end a handle's lifetime also at a whole-state replacement (a
same-kind `copyto!` or `copy!`, or the owner's CPY-4 import), whatever happens
to `nactive`, a self-copy excepted; FW-5 separates it from compaction.

*Options.*

- **(a) A read-only wrapper, one type for the family** (recommended): FC-7.
  - *Correctness:* no supported accessor can corrupt the lists.
  - *Downstream:* none of the 29 audited ProximalSolvers lines that call
    `activecoordinates` writes through it.
  - *Numerics:* none.
  - *Performance:* `O(1)`, no allocation; `getindex` must inline (benchmark:
    sweeps through the handle).
- **(b) A copy per call.** Safe, but `O(s)` and an allocation per call,
  against FP-2.
- **(c) Writable views, documented.** The corruption above stays.

*Recommendation and reason.* (a): safety at no traversal cost, and the
owner's handle (ACT-3) becomes the family's.

*Kind:* 0.5.x bug fix, filed as #405. The return type changes, but only
writes, which corrupt state, stop working; readers are unaffected. The owner
gets the same handle type with #391 Phase 1a, or earlier.

*Migration.* Code that wrote through the handle uses `activate!` and
`dropzeros!`; code that needs a mutable list writes
`collect(activecoordinates(x))`. Code that keeps a history by slot starts it
again after a state copy into the iterate. In ProximalSolvers no audited path
copies into an iterate while such a history is live: every `copyto!` into an
iterate in `src/cd/glm.jl` (`:552`, `:572`, `:578`, `:737`, `:754`),
`src/cd/generic_newton.jl` (`:122–123`, `:165`) and `src/cd/lla.jl` (`:214`)
runs between two `_cd!` runs, and each run starts a new momentum history
(`_momentum_start!`, `src/cd/solver.jl:464–475`, then a full first pass and
`_momentum_reset!`, `:452`, `:478–489`) and a new Anderson history (`:608`),
which also restarts when the active list changes (`:642`).

*Relation to #388/#391:* **consistent extension** of ACT-3 and ACT-4: the
family shares ACT-3's handle and lifetime, and a whole-state replacement ends
their handles, a case their lifetime did not name.

*Sign-off:* ☐ maintainer.

### FD-9. Zero coefficients are strong in products

*Status: recommended, awaiting maintainer sign-off.*

*Background.* The products of a `SparseIterate` loop over stored
coefficients: they skip unstored zeros and multiply stored ones
(`src/sparse_iterate.jl:687–693`, `:718–736`, `:778–780`, `:823–829`,
`:840–842`, `:851–853`, `:863–866`; for the owner `:1826–1836`,
`:1839–1849`, `:1913–1927`). An `AtomIterate`'s `mul!` skips zero values
(`src/atom_iterate.jl:307`, `:330`); its `dot` does not (`:342–344`).
Three-factor products `A * X * B` are a separate decision (FD-16). Numerics
M1 lets an operation skip an implicit zero "only when
they contribute nothing", and E2 and E3 make `0 · NaN` and `0 · Inf` NaN.
PERF-6 requires the owner's product to be the same, or to raise the same
error, before and after `dropzeros!`, and decision 13 recommends an
`ArgumentError` after an `O(p)` scan.

*What goes wrong today* (ran; Part 4).

- *Argument order:* with `s = [1, 0, 0]` and `t = [0, NaN, 0]`,
  `dot(s, t) == 0.0` and `dot(t, s)` is NaN.
- *Storage:* with `A = [NaN 1 0; NaN 1 0]` and an explicit zero at 1,
  `A * u` is `[NaN, NaN]`, and `[1.0, 1.0]` after `dropzeros!`; likewise
  `dot(A[1, :], u)` (NaN, 1.0) and `dot([NaN, 1], Z, [1, 1])` (NaN, 1.0). For
  an `AtomIterate` with an active zero group, `B * a == [1.0]` but
  `dot(v, a)` is NaN, and 1.0 after `dropzeros!`. The owner's
  `mul!(y, matrixview(X), [Inf, 1, 0])` gives `[0, 1, NaN]`, and `[0, 1, 0]`
  after `dropzeros!`.
- *An oracle:* `LeastSquaresLoss([1, 1], [NaN 1; NaN 1])` is NaN at an
  iterate with an explicit zero at 1, 0.0 when that zero is not stored, and
  NaN at the dense `[0, 1]`.

*Options.*

- **(a) Strong zeros** (recommended): FN-1 to FN-5.
  - *Correctness:* a product with one iterate operand does not change when
    stored zeros are dropped and does not depend on the order of the
    arguments; one rule for every product in FN-2.
  - *Downstream:* ProximalSolvers' CD models check their data
    (`_check_finite`, ProximalSolvers `src/cd/multiresponse.jl:34`;
    `src/cd/leastsquares.jl:148`), so sweeps do not change. ProximalBase
    oracles at an iterate with non-finite data change from NaN to finite
    where a stored zero meets the non-finite entry (FN-5).
  - *Numerics:* a NaN becomes finite in those cases. That departs from E2 and
    E3 as written and is a class change under C1; the amendment is below.
  - *Performance:* no scan: at most a zero test per stored coefficient
    (benchmark case).
- **(b) An error on a non-finite dense operand** (#388 decision 13).
  - *Correctness:* a defined result where the dense operand is a vector
    (`dot`, `mul!(y, matrixview(X), v)`). For `mul!(y, A, x)` the dense
    operand is a matrix, and the scan costs `O(mn)`, as much as a dense
    product, which removes the reason for a sparse iterate.
  - *Downstream:* a data loss with NaN data would throw.
  - *Numerics:* an `ArgumentError` for a NaN argument contradicts E2 ("A NaN
    entry of the argument is an undefined value, propagated …").
  - *Performance:* `O(length(v))` per call, `O(mn)` for a matrix operand.
- **(c) IEEE, as if the iterate were dense.**
  - *Correctness, numerics:* the dense result, as E2 and E3 say.
  - *Performance:* every zero position must be checked against the dense
    operand: `O(n)` for `dot`, `O(mn)` for `mul!`.
- **(d) Status quo.** Results depend on storage, order and density.
- **(e) Strong scaling as well**: let `rmul!`, `lmul!`, `rdiv!` and `ldiv!`
  by a number skip zero coefficients too.
  - *Correctness:* `rmul!(x, Inf)` would then differ from `x .*= Inf`, which
    is IEEE (FN-9), so two spellings of one operation would disagree.
  - *Downstream, numerics:* none audited; the other operand is a single
    scalar, so IEEE scaling costs nothing extra when the scalar is finite.
  - Not recommended: FN-9 keeps scaling IEEE.

*Precedents.*

- This package already treats `α = 0` and `β = 0` as strong zeros in its own
  `mul!` (`src/sparse_iterate.jl:685`, `_scale_product_output!` at `:757`; the
  comment at `:1910`: "As in BLAS, C is not read when β is zero and B is not
  read when α is zero"). Ran: `mul!(out, A, s, 0.0, 1.0)` with a NaN in
  `A` gives `[5.0, 5.0]`; `mul!(out, A, s, 1.0, 0.0)` with NaN in `out` gives
  `[1.0, 0.0]`.
- Base: `false * NaN === 0.0` (ran).
- SparseArrays (Julia 1.13.3): dense matrix times sparse vector, `_spmul!`
  (`sparsevector.jl:1942–1964`), skips every zero coefficient, stored or
  structural (the `_isnotzero` test at `:1957`). But
  `dot(x::AbstractVector, y::SparseVectorUnion)` (`:1812`) multiplies every
  stored value, and a CSC matrix times a dense vector multiplies stored zeros
  (ran: `[NaN, 1.0]` with a stored zero against `[Inf, 1]`, `[1.0, 1.0]` with a
  structural one). SparseArrays is therefore inconsistent; FN-1 takes the
  `_spmul!` side for every method the package owns.
- Dense OpenBLAS propagates NaN (ran: `[NaN 1; NaN 1] * [0, 1] == [NaN, NaN]`),
  the IEEE side that FN-3 keeps for fallbacks.

*Recommendation and reason.* (a). It is the only option that makes a product
with one iterate operand independent of storage and argument order without an
extra pass; it extends a convention the package already applies to `α` and
`β`; and (b) cannot cover `mul!(y, A, x)` without a scan as costly as a dense
product.

*Kind:* breaking 0.6.

*Migration.* A caller that wants IEEE propagation through a zero
coefficient multiplies the dense array, `A * Vector(x)` (FN-3). No audited
ProximalSolvers site changes.

*Proposed numerics amendment, word for word* (to `docs/src/numerics.md`,
implemented under #366 group G2):

> **M1. Implicit zeros are entries; zero coefficients of an iterate are
> strong.** An unstored entry of a sparse iterate is an entry whose value is
> zero, and an operation skips it only when it contributes nothing: a
> centered ridge moves an implicit zero, and `NegLog` is `+Inf` at zero. In
> the products and inner products listed below, a zero coefficient of the
> iterate, stored or unstored, contributes nothing, even against `±Inf` or NaN
> in the other operand, as `false * NaN` is `0.0` in Julia and as BLAS treats
> `α = 0`. A non-finite entry that meets a nonzero coefficient propagates as
> E2 and E3 say, and a zero of a dense operand is an ordinary IEEE zero. The
> methods are exactly these:
>
> 1. `dot(v::StridedVector, x::SparseIterate{T,1})`,
>    `dot(x::SparseIterate{T,1}, v::StridedVector)` and
>    `dot(x::SparseIterate{T,1}, y::SparseIterate{T,1})`;
> 2. `dot(v, x::AtomIterate)` for `v` a `StridedVector`, a
>    `SparseIterate{T,1}` or an `AtomIterate`, and `dot(x::AtomIterate, v)`
>    for `v` a `StridedVector` or a `SparseIterate{T,1}`;
> 3. `dot(a::StridedVector, X::SparseIterate{T,2}, b::StridedVector)`;
> 4. `mul!(out, A, x)`, its five-argument form and `A * x`, for
>    `x::SparseIterate{T,1}` with `A` a `StridedMatrix`, a transposed or
>    adjoint `StridedMatrix`, or a `Symmetric` strided matrix; for
>    `x::SparseIterate{T,2}` with `A` a `StridedMatrix` or a `Symmetric`
>    strided matrix; and for `x::AtomIterate` with `A` a `StridedMatrix` or a
>    `Symmetric` strided matrix;
> 5. for a `SymmetricSparseIterate` `X`: `mul!(y, matrixview(X), v)` and its
>    five-argument form, and `matrixview(X) * v`, which reaches that `mul!`;
>    `dot(a::StridedVector, X, b::StridedVector)`; and
>    `dot(M::StridedMatrix, X)` and `dot(X, M)`. A sparse-aware `dot` on
>    `matrixview(X)`, if one is added, is a method of the package and joins
>    this item.
>
> Their results therefore do not depend on whether a zero is stored or on
> the order of the arguments. A value or gradient that a function computes
> through one of them at an iterate argument, such as `LeastSquaresLoss` at a
> `SparseIterate`, inherits the rule. The rule applies to the iterate object
> itself, which includes `vec(x)` of a vector and `reshape(x, length(x))`,
> since Base returns `x` for those, and to the `matrixview(X)` products of
> item 5. Every other object built from an iterate (`view(x, …)`,
> `atom(x, k)`, `coordinates(x)`, `activevalues(x)`, `vec(x)` of a matrix, a
> `reshape` to another shape, and `matrixview(X)` in any other product) is an
> ordinary vector or matrix and follows IEEE arithmetic, as does a product
> that ProximalBase does not define, such as `sparse(A) * x`,
> `Diagonal(d) * x`, `transpose(A) * x` for an `AtomIterate`, or
> `sum(x .* v)`. Three-factor products `A * X * B` are not among these
> products. Scaling an iterate by a number (`rmul!`,
> `lmul!`, `rdiv!`, `ldiv!`) is not among these products either: it follows
> IEEE arithmetic, as `x .*= s` does.

> **E2**, a new last bullet: *Exception for iterates (M1).* In the products
> and inner products that M1 lists, a zero coefficient of the iterate
> contributes nothing, so a NaN of the other operand that meets only zero
> coefficients does not reach the result.

> **E3**, a new last sentence: In the products and inner products that M1
> lists, `0 · ±Inf` contributes `0` when the zero is a coefficient of the
> iterate, and NaN when the zero belongs to a dense operand.

> **C1**, a new last sentence: M1's strong zeros changed the class of some
> products of iterates in 0.6, where a zero coefficient of an iterate, stored
> or not, met a non-finite entry; classes changed from NaN to finite values or
> to ±Inf. That change was signed off with M1, and C1 protects the classes
> that M1 defines.

> **S4**, a new last sentence: M1 states one boundary of its own: a product of
> an iterate that ProximalBase does not define, a product of an object built
> from an iterate that is not the iterate itself, and scaling by a number
> follow IEEE arithmetic, and these are outside the strong-zero rule, not
> gaps.

*Relation to #388/#391:* **proposed amendment** of decision 13 and of
PERF-6's recommended default: the `ArgumentError` after an `O(p)` scan is
replaced by FN-1. PERF-6's requirement (the same result, or the same error,
before and after `dropzeros!`) is met. T13's parenthesis "the recommended
default throws in both" becomes "gives `[0, 1, 0]` in both". ORC-10's
`dot(M, X)` is in FN-2. The owner's three-factor product is FD-16's.

*Sign-off (separate from the structural decisions, §3.1):* ☐ maintainer
(FN-1 to FN-5) · ☐ numerics amendment (M1, E2, E3, C1, S4).

### FD-10. Counts

*Status: recommended, awaiting maintainer sign-off.*

*Background.* `nnz(x)` counts stored coordinates, explicit zeros included
(`src/sparse_iterate.jl:223`); `nonzeros` (`:448`) and `findnz` (`:457–471`)
include stored zeros too. An `AtomIterate` has no `nnz`. #192 asks what `nnz`
should count.

*What goes wrong today* (ran). Three numbers answer "how many nonzeros?":
with one explicit zero, `nnz(x) == nactive(x) == 2` and
`count(!iszero, x) == 1`. For the owner they differ further (#388 §2.6).

*Options.*

- **(a) Keep SparseArrays' meaning on `SparseIterate`, no `nnz` on
  `AtomIterate`, ACT-2 for the owner, and document four counts** (recommended;
  FC-10, FC-11).
  - *Correctness:* each count has one name and formula.
  - *Downstream:* none; `1:nnz(x)` loops in other packages (#192) keep
    working.
  - *Numerics:* none. *Performance:* `O(1)`.
- **(b) `nnz` counts mathematical nonzeros.** `O(s)`, and it breaks the
  `1:nnz(x)` loops listed in #192.
- **(c) Remove `nnz` from `SparseIterate`.** It breaks SparseArrays-generic
  code, since `issparse(x)` is `true` (`src/sparse_iterate.jl:439`).

*Recommendation and reason.* (a): SparseArrays' meaning for the type that
claims to be sparse, and an explicit formula for everything else.

*Kind:* 0.5.x behavior, now stated.

*Migration.* A model size is `count(!iszero, activevalues(x))`, or `nactive`
after `dropzeros!`.

*Relation to #388/#391:* consistent with ACT-2; resolves #192 for the array
iterates.

*Sign-off:* ☐ maintainer.

### FD-11. Group layout

*Status: recommended, awaiting maintainer sign-off.*

*Background.* The constructor owns the layout (`_own_groups`,
`src/atom_iterate.jl:85–88`); `copy` and `similar` share it (`:236–245`), and
`AtomIterate(p, f)` shares the penalty's (`src/functions/norms.jl:752–760`).
Compatibility is cached in one slot, `layout_cache::Any`
(`src/atom_iterate.jl:35`, `:116–122`). `docs/src/iterates.md:440–442`
forbids mutating group arrays.

*What goes wrong today* (ran). A vector group is reachable mutably through
`parentindices(atom(x, k))` (FD-6). Two equal but distinct layouts evict each
other from the cache, so alternating them compares every time. Whole-array
oracles with a different layout fall back to the generic path and give the
function of the vector (`fm(am) == fm(Vector(am)) == 10.79669127533634`),
while coordinate oracles throw `DimensionMismatch`; a contract that promised
rejection everywhere would break the first.

*Options.*

- **(a) Promises that match the code, `samegroups`, and read-only storage of
  vector groups as a recommendation** (recommended; FL-1 to FL-7).
  - *Correctness:* the layout cannot change through a supported accessor.
  - *Downstream:* only FD-6's replacements.
  - *Numerics:* none.
  - *Performance:* unchanged, except FL-3, which changes the index type of
    `atom` views of vector groups and must be measured.
- **(b) A public `GroupLayout`** owned, immutable, validated once and shared
  by penalties and iterates.
  - *Correctness:* identity is equality, so compatibility is always `O(1)`.
  - *Downstream:* penalty constructors and fields change; ProximalSolvers
    builds penalties from `g.groups` (`_with_level`,
    ProximalSolvers `src/cd/models.jl:473–478`; `src/cd/lla.jl:79`).
  - *Performance:* removes the one `O(total group size)` comparison that each
    path level pays today, since `_with_level` makes a fresh owned copy
    (`src/functions/norms.jl:438`).
- **(c) Status quo**, with no stated promise.

*Recommendation and reason.* (a): it makes the code's actual guarantees the
contract and closes the one mutable route, without a new public type that
the penalties would have to adopt.

*Kind:* 0.5.x-safe addition (`samegroups`); FL-3, if adopted, in 0.6.

*Migration.* FD-6's.

*Relation to #388/#391:* unchanged (#342 for groups on the owner).

*Sign-off:* ☐ maintainer.

### FD-12. One contract for coordinate oracles

*Status: recommended, awaiting maintainer sign-off.*

*Background.* The scalar forms are `src/interface.jl:384–425`, the group
forms `:464–478`, the block-trial staging `_stage_and_step!` `:483–526`, and
`kkt_violation` `:552–570`. The return type differs by iterate (ran:
`Float64` and a `SubArray`). ProximalSolvers stages group trials by writing
`x.storage` and then calls the four-argument form
(ProximalSolvers `src/cd/groups.jl:32–38`, `:64–70`).

*What goes wrong today* (ran). After a private write of `[3, 4]` into an
inactive group and a call that throws (a different layout), the group is
inactive and holds `[3.0, 4.0]`: `GroupNormL2(1.0, [1:2, 3:4])` of the iterate
is `0.0` while that of `Vector(x)` is `7.0710678118654755`, and a state copy
gives `[0, 0, 0, 0]`. The five-argument form with the same arguments throws
the same `DimensionMismatch` and leaves `x` unchanged. When nothing throws,
the two give the same `[2.151471862576143, 2.868629150101524]`.

*Options.*

- **(a) One contract, FA-1 to FA-7, with the five-argument form as the
  block-trial API** (recommended).
  - *Correctness:* FW-1 holds through a failing step.
  - *Downstream:* two functions in ProximalSolvers `src/cd/groups.jl` change
    (companion §3.3).
  - *Numerics:* identical results when nothing throws (ran).
  - *Performance:* for a strided floating-point `u`, the staging loop of
    `_stage_and_step!` (`src/interface.jl:484–493`) is the same work as the
    private write; `u` is copied only when it aliases `x`. A benchmark case.
- **(b) Bless private staging** by documenting `x.storage`. FW-1 then fails
  for good whenever the following call throws.
- **(c) Per-method documentation only.** The contract stays implicit.

*Recommendation and reason.* (a): the API already exists, and it is the
only route that keeps FW-1 when a step fails.

*Kind:* 0.5.x behavior, now stated; a downstream migration.

*Migration.* Companion §3.3.

*Relation to #388/#391:* consistent extension of ORC-5, ORC-6 and ORC-9.

*Sign-off:* ☐ maintainer.

### FD-13. Wrappers and real-only checks

*Status: recommended, awaiting maintainer sign-off.*

*Background.* Broadcasts into an iterate check for complex arguments before
writing (`_checkreal_broadcast`, `src/sparse_iterate.jl:1532–1538`, called at
`:557`, `:562`, `src/atom_iterate.jl:263`, `:268`). A two-argument `copyto!`
from an array goes to Base's generic method unchecked. The owner's
`_copyto_lower!` checks first (`src/sparse_iterate.jl:1152`).

*What goes wrong today* (ran). `copyto!(x, ComplexF64[1, 2im, 0, 0])` writes
`x[1] = 1.0` and throws `InexactError`, for both types; so do `x[:] = c` and
`view(x, 1:2) .= c2`. `x .= c` throws `ArgumentError` before writing. Numerics
S1 and O6.1 ask for validation before writing.

*Options.*

- **(a) The iterate's own bulk writes check first; wrappers follow Base**
  (recommended; FX-4, FN-8).
  - *Correctness:* every entry point the package defines validates before
    writing; a wrapper behaves as the same wrapper of a plain array.
  - *Downstream:* none (no audited code passes complex data).
  - *Performance:* one element-type test per call, resolved at compile time.
  - *Ambiguities:* the check before writing needs a method
    `copyto!(x::SparseIterate, y::AbstractArray)`. With it,
    `copyto!(SparseIterate(2, 2), sparse([1.0 0; 0 2]))` is ambiguous with
    SparseArrays' `copyto!(::AbstractMatrix, ::AbstractSparseMatrixCSC)`
    (`sparsematrix.jl:616`; ran: `MethodError … is ambiguous`), while a dense
    source works. Resolving methods are needed for the known sources:
    SparseArrays CSC matrices, their views, adjoints and transposes, and
    CHOLMOD `Dense`, as the owner already has (`src/sparse_iterate.jl:1129–1150`).
    Aqua's ambiguity test (`test/aqua.jl`, `Aqua.test_all`) and FT20 check
    them. This is a small fixed list for one entry point, not a catalog.
- **(b) Overrides for views, reshapes and `atom` of iterates** (#364's first
  option). It is the catalog of overrides that the owner redesign removes,
  with the ambiguity risk #368 records.
- **(c) Status quo.** A partial write from the package's own `copyto!`.

*Recommendation and reason.* (a): it fixes the package's own entry points
and states Base's rule for the rest, without a catalog.

*Kind:* 0.5.x bug fix (the `copyto!` check, recorded on #364 together with
the `vec(xm)` broadcast); the wrapper rule is stated.

*Migration.* None.

*Proposed numerics text, word for word* (to `docs/src/numerics.md`; with it,
FN-8 does not contradict S4):

> **O6**, a new first paragraph: These guarantees cover the methods that
> ProximalBase defines. A Base or LinearAlgebra function applied to a wrapper
> of an iterate (`view`, `reshape`, `vec`, `atom`) behaves as it does on the
> same wrapper of a plain array of the same element type, and applied to
> `coordinates(x)` as it does on a plain vector of that element type holding
> the same entries: a conversion that throws can leave a partial write.

> **S4**, a new sentence after the first: Base and LinearAlgebra functions
> applied to wrappers of iterates are not functions of the package; O6 states
> what they do.

*Relation to #388/#391:* unchanged; the owner's part of #364 goes away with
MAT-2.

*Sign-off:* ☐ maintainer (FX-4, FN-8) · ☐ numerics text (O6, S4).

### FD-14. Aliasing

*Status: recommended, awaiting maintainer sign-off.*

*Background.* `SparseIterate` defines no `Base.dataids`, so Base's default
(`abstractarray.jl:1611`, the object's identity) applies.
`Base.dataids(x::AtomIterate)` is that of the values only
(`src/atom_iterate.jl:133`). The owner reports all three storage arrays since
#392 (`src/sparse_iterate.jl:940–943`).

*What goes wrong today* (ran). `Base.mightalias(activevalues(x), x)` and
`mightalias(activecoordinates(x), x)` are `false` for a `SparseIterate`; the
second is `false` for an `AtomIterate`. So after `x[2] = 1; x[1] = 2`,
`copyto!(x, activevalues(x))` gives `[1, 1]` instead of `[1, 2]`. Views of
`x` are detected (`mightalias(view(x, 1:2), x)` is `true`), and
`copyto!(x, view(x, 4:-1:1))` is right for both types.

*Options.*

- **(a) Wider `dataids` and Base's `unalias`** (recommended; FX-5).
  - *Correctness:* every public handle is seen; imports from a handle are
    right.
  - *Downstream:* none audited.
  - *Numerics:* consistent with numerics O7.
  - *Performance:* free unless a source shares memory; then one copy of the
    source (an allocation). `prox_coordinate!`'s alias check
    (`src/interface.jl:474–476`) sees more true aliases, only real ones.
- **(b) Reject an aliasing source**, as CPY-6 does. Base's generic methods
  unalias rather than throw, so rejection needs an override on every entry
  point.
- **(c) Status quo.**

*Recommendation and reason.* (a): Base's own protocol, with no new error.

*Kind:* 0.5.x bug fix, filed as #406.

*Migration.* None.

*Relation to #388/#391:* consistent with #392's owner `dataids`; CPY-6
unchanged.

*Sign-off:* ☐ maintainer.

### FD-15a. Shape-aware grouped iterates

*Status: recommended, awaiting maintainer sign-off.*

*Background.* ProximalSolvers' multi-response model is the motivating
workload: a p × q coefficient matrix `B` stored as an `AtomIterate` over
`vec(B)` with the rows of `B` as groups (ProximalSolvers
`src/cd/multiresponse.jl:45–48`), its layout checked against the penalty
(`:55`), and its own gradient over `vec(B)` through `reshape(G, p, q)`
(`:180–196`).

*What goes wrong today* (ran). Nothing in that model: it is served by
`vec(B)` and its own `MLSModel`. But ProximalBase's matrix losses cannot take
that `AtomIterate`: `gradient!` of `LeastSquaresLoss(Y, X)` with a matrix `Y`
at it throws `DimensionMismatch`. That is today's behavior, not a promise, so
acting on the trigger below would not be a break. The mapping between entries,
groups and the gradient (FS-4) is therefore a convention that each model
author applies.

*Options.*

- **(a) Postpone** (recommended): document the `vec(B)` mapping (FS-4).
  - *Correctness:* nothing is wrong today; the convention is stated.
  - *Downstream:* none.
  - *Numerics, performance:* none.
- **(b) A shape-aware `AtomIterate{T, N}`** that is a p × q matrix with row
  (or column) groups.
  - *Correctness:* ProximalBase's matrix losses could take grouped rows
    directly.
  - *Downstream:* `multiresponse.jl` would be rewritten for the new type.
  - *Numerics:* none.
  - *Performance:* a new type on the group hot paths (Cartesian indexing in
    the group loops, new `atom` views), all to be measured.

*Recommendation and reason.* (a): one audited consumer, already served. The
trigger to revisit: a second consumer, or a ProximalBase matrix loss that
needs grouped rows.

*Kind:* no change.

*Migration.* None.

*Relation to #388/#391:* unchanged.

*Sign-off:* ☐ maintainer.

### FD-16. Three-factor products

*Status: open, awaiting a separate decision; option (a) is recommended.*

*Background.* `*(A::Symmetric, X::SparseIterate{T,2}, B::Symmetric)`
(`src/sparse_iterate.jl:877–902`) multiplies a dense sub-block of `X` on its
sparse path (`:894–901`, the association chosen by `nI <= nJ` at `:901`) and
falls back to dense BLAS when the support is dense (`:888`). The owner's
method (`:1867–1892`, dense fallback at `:1881`) works the same way; 0.5.x has
it, and neither owner document mentions it. Mixed element types take a third
path, `*(A::Symmetric, X::Union{SparseIterate{<:Any, 2}, SymmetricSparseIterate}, B::Symmetric)`
(`:1951–1955`), which calls LinearAlgebra's generic three-argument `*`. The
sparse path skips the zero rows and columns of `X` (`:884`, `:896`), but no
path skips every zero of `X`. FD-9 does not cover these methods (FN-10).

*What goes wrong today* (ran; Part 4).

- *Density:* with `A[1, 1] = NaN` and row 1 of `X` zero, `(A * X * B)[1, :]`
  is `[0, 0, 0]` on the sparse path and `[NaN, NaN, NaN]` on the dense
  fallback.
- *Neither path is strong:* with `A[1, 2] = A[2, 1] = Inf` and a zero inside
  the sparse path's block, row 2 is `[NaN, NaN, NaN, NaN]` where the term-wise
  sum gives `[Inf, Inf, Inf, Inf]`; with a NaN of `B` meeting only a zero
  column of `X` on the dense fallback, column 1 is `[NaN, NaN, NaN, NaN]` where
  the term-wise sum gives `[9.0, 8.0, 7.0, 6.0]`.
- *Association:* with `A` the `Symmetric` diagonal matrix of `(Inf, 1, 1, 1)`,
  `B` the identity and `X` nonzero at `(1, 1)`, `(1, 2)`, `(2, 1)`, `(3, 1)`,
  `A * (Matrix(X) * B)` has row 1 `[Inf, Inf, NaN, NaN]`, while
  `(A * Matrix(X)) * B`, the package and the term-wise sum have
  `[NaN, NaN, NaN, NaN]`. The strong `A * X` has `[Inf, Inf, 0, 0]`, so a
  term-wise rule makes `A * X * I` and `A * X` differ in class.
- *Finite overflow:* with `X[1, 1] = 1e200`, `A[1, 1] = 1e200` and
  `B[1, 1] = 1e-200`, the exact entry is `1.0e200`; the package and
  `A * (Matrix(X) * B)` give `1.0e200`, and `(A * Matrix(X)) * B` gives `Inf`.
- *A mixed fixture:* `u = 2.0^600`, `A = Symmetric(diag(Inf, u))`,
  `B = Symmetric(diag(1, 1/u))`, `X = diag(1, u)`. The package gives
  `[Inf NaN; 0.0 4.149515568880993e180]`. The exact `(2, 2)` entry is `u`, and
  `u * (u * (1/u))` gives it, but `(u * u) * (1/u)` is `Inf`. A full term-wise
  recomputation, triggered by the infinite `(1, 1)`, would therefore turn the
  representable `(2, 2)` into `Inf`, and `(2, 1)` into NaN.

*Options.*

- **(a) Keep the ordinary path's finite entries; recompute only the
  non-finite ones term by term** (recommended). Each non-finite entry becomes
  the sum of `(A[i, k] * X[k, l]) * B[l, j]` over the nonzero `X[k, l]`, in
  column-major order of `X`; the zeros of `X` are strong, those of `A` and `B`
  IEEE.
  - *Correctness:* the class guarantee covers the zeros of `X` meeting
    non-finite entries of `A` or `B`: since NaN and `Inf` propagate through
    the BLAS sums, an entry where that happens is non-finite and is
    recomputed. It excludes finite partial products that overflow: matrix
    products follow A1, by the last sentence of numerics R2. On the mixed
    fixture `(2, 2)` stays the BLAS value `u`, and only `(1, 1)` and `(1, 2)`
    are recomputed, giving `[Inf NaN; 0.0 4.149515568880993e180]` (ran).
  - *Downstream:* no audited ProximalSolvers site calls the triple product;
    ProximalBase's deprecated `A_mul_X_mul_B` forwards to both methods
    (`src/deprecated.jl:72` for the owner; `:84–87` through a matrix
    `SparseIterate`).
  - *Numerics:* the class can change from NaN to finite or `±Inf`; the text
    below states the rule.
  - *Performance:* one `isfinite` scan of the m × n output, plus `O(s)`
    recompute work for each non-finite entry.
- **(b) A range-safe term-wise fallback.** As (a), but each term is evaluated
  with scaling, so that a finite partial product cannot overflow; tested
  against an independent exact or high-precision reference on the mixed
  fixture.
  - *Correctness:* the class of every recomputed entry is that of the exact
    term-wise sum, overflow included.
  - *Downstream:* as (a).
  - *Numerics:* as (a), without the A1 exception.
  - *Performance:* code for the scaled evaluation, and a slower slow path.
- **(c) IEEE on the dense `X` with one fixed association**,
  `A * (Matrix(X) * B)`, made independent of the path by an `O(p²)` scan of
  `A` and `B` and a dense fallback when they are not finite.
  - *Correctness:* one class for every path, but the zeros of `X` are IEEE
    zeros, unlike FD-9's rule for products with one iterate.
  - *Downstream:* as (a).
  - *Numerics:* no amendment needed.
  - *Performance:* an `O(p²)` scan of `A` and `B` on every call.
- **(d) Leave the triple product out of 0.6's guarantees**: IEEE as computed,
  as today.
  - *Correctness:* the class keeps depending on density, path and
    association (the cases above); the rule says so.
  - *Downstream, numerics, performance:* none.

*Recommendation and reason.* (a): it keeps the values of the ordinary path,
fixes the class wherever a zero of `X` meets a non-finite entry, for the cost
of one output scan, and leaves overflow where the numerical contract already
puts matrix products (A1).

*Kind:* breaking 0.6 for (a), (b) and (c); none for (d).

*Migration.* None in the audited code.

*Proposed numerics text for option (a), word for word* (to
`docs/src/numerics.md`):

> **M1**, a new last paragraph: For `*(A::Symmetric, X, B::Symmetric)` with a
> sparse iterate `X`, each entry that the ordinary product computes as finite
> is kept, and each entry that it computes as `±Inf` or NaN is recomputed as
> the sum of `(A[i, k] · X[k, l]) · B[l, j]` over the nonzero coefficients
> `X[k, l]`, so that a zero of `X` contributes nothing against a non-finite
> entry of `A` or `B`. Finite partial products that overflow follow A1, as R2
> says for matrix products.

*Relation to #388/#391:* **consistent extension:** the 0.6 owner keeps 0.5.x's
`*(A::Symmetric, X::SymmetricSparseIterate, B::Symmetric)`, and FD-16 decides
its class.

*Sign-off (separate, §3.1):* ☐ option (a) · (b) · (c) · (d) · ☐ numerics text
(for (a) or (b)).

---

# Part 4. Evidence (informative)

"Ran" means a script ran against the audited source with Julia 1.13.3; the
Base routing was also run on Julia 1.10.12. "Read" gives the line.

## Neighbouring defects

Found while auditing; each is a current fault, with the rule that fixes it.

| # | defect (ran) | fixed by | issue |
|:--|:--|:--|:--|
| 0 | writable `activecoordinates` corrupts a `SparseIterate`, and an `AtomIterate`, whose `dropzeros!` then never returns | FC-7 (FD-7) | #405 (0.5.x bug fix) |
| 1 | `dot(x, y)` of two `SparseIterate`s depends on the argument order (`0.0` against NaN) | FN-1, FN-4 (FD-9) | this contract |
| 2 | a stored zero changes products: `A * s`, `dot(a, s)`, `dot(a, X, b)`; `AtomIterate` `dot`; the owner's `mul!(y, matrixview(X), [Inf, 1, 0])` (`[0, 1, NaN]` against `[0, 1, 0]`) | FN-1, FN-4 (FD-9) | this contract; PERF-6 for the owner |
| 3 | `A * X * B` changes class with the density of `X` (`[0, 0, 0]` against `[NaN, NaN, NaN]`); neither path is strong, and the class depends on the association | FN-10 (FD-16, open) | this contract |
| 4 | the working set depends on the spelling of a write (`x .= 0`, `fill!`, `sum!` clear; `x .= y` state-copies) | FW-2, FW-3 (FD-2) | this contract |
| 5 | an oracle's destination working set depends on its branch | FW-8 (FD-4): allowed, now stated | this contract |
| 6 | `fill!(x, 0)` clears, unlike SparseArrays' `fill!` (`sparsevector.jl:2434–2443`, which converts and keeps stored zeros); `AtomIterate`'s `fill!` tests for zero before converting and activates both groups for `1e-100` | FW-2 (FD-2), FN-6 | this contract |
| 7 | `mightalias` misses the handles; `copyto!(x, activevalues(x))` gives `[1, 1]` | FX-5 (FD-14) | #406 (0.5.x bug fix) |
| 8 | `copyto!` from a complex source writes, then throws `InexactError`; `vec(xm) .=` likewise | FX-4, FN-8 (FD-13) | #364 (comment) |
| 9 | `rmul!(x, Inf)` and `rdiv!(x, 0.0)` of a `SparseIterate` depend on storage: `[Inf, NaN, 0, 0]` against `[Inf, NaN, NaN, NaN]` for `x .*= Inf` | FN-9 | this contract |

## Claims

| claim | at `def81ec9` |
|:--|:--|
| `coordinates(x)` of `SparseIterate(Float64, 2, 3)` is the iterate, size `(2, 3)`; `vec(x)` is a `Base.ReshapedArray` | ran; `src/sparse_iterate.jl:480` |
| `activecoordinates` is a writable `SubArray` for all three; `ac[1] = 3` on a `SparseIterate` gives `x[2] == 1.0`, `isactive(x, 2)`, `!isactive(x, 3)`, `Vector(x) == [0, 0, 1, 3]` | ran; `src/sparse_iterate.jl:287` |
| `AtomIterate`: after `ac[1] = 2`, `active = [2, 3, 0]`, `position = [1, 0, 2]`; `dropzeros!` did not return within 25 s | ran; loop `src/atom_iterate.jl:211–222`, `_deactivate!` `:192–202` |
| `nonzeros(x)` and `activevalues(x)` are the same kind of writable view | ran; `src/sparse_iterate.jl:288`, `:448` |
| with an explicit zero: `nnz`, `nactive`, `count(!iszero, ·)` are `(2, 2, 1)`; `findnz` gives `([1, 3], [1.0, 0.0])`; `nnz(AtomIterate)` is a `MethodError` | ran; `src/sparse_iterate.jl:223`, `:457–471` |
| no `coordinates` and no `activevalues` method for `AtomIterate` | ran (`hasmethod`) |
| `fill!(x, 0)` and `x .= 0` empty the working set of both array iterates; `x .= 0 .* x`, `rmul!(x, 0)` and `x .*= 0` keep both coordinates | ran; `src/sparse_iterate.jl:566–576`, `src/atom_iterate.jl:272–282` |
| `x .= y` with `y` a same-kind iterate is a state copy (`[3, 2]`; `AtomIterate` `[2, 1]`); `x .= 1 .* y`, the loop and `x[:] = y` give `[1, 3]` (`AtomIterate` `[1, 2]`) | ran; `src/sparse_iterate.jl:556–559`, `src/atom_iterate.jl:262–265` |
| Base routing table of §2.3, identical on 1.10.12 and 1.13.3; `x[[4, 1, 2, 3]]` of an iterate is a dense `Vector`; `copy!(SparseIterate(4), ones(2))` is a `MethodError` for `resize!` | ran; Base `broadcast.jl:960`, `:993`, `abstractarray.jl:924–931`, `:933–937`, `combinatorics.jl:214` (1.10.12: `:964`, `:997`, `:918`, `:927`, `:192`) |
| slot order of a replace: from a working set `[5, 2, 9]` and `A` nonzero at 2, 3, 9, `dropzeros!(copyto!(x, A))` gives `[3, 2, 9]` and `copyto!(dropzeros!(fill!(x, 0)), A)` gives `[2, 3, 9]`; `AtomIterate` `[3, 2]` against `[2, 3]`; with groups `[[2, 4], [1, 3]]` an import of `[1, 1, 0, 0]` after a clear gives `[2, 1]` while the constructor from the same values gives `[1, 2]`; with groups `[[1, 3], [2, 4]]` and nonzeros at entries 2 and 3 a replace gives `[2, 1]`; for the strided multi-response layout the constructor gives `[1, 2]`, and a replace and 0.5.x's fill-and-scatter both give `[2, 1]`; the owner gives `[5, 2, 3]` against `[2, 3, 5]`; `dropzeros!(fill!(coordinates(X), 0))` is a `MethodError` | ran; constructor order `src/atom_iterate.jl:76–78` |
| ProximalSolvers' cyclic active pass visits slots in order | read; ProximalSolvers `src/cd/solver.jl:329–331` |
| `ldiv!(x, F, Vector(y))` keeps `x`'s coordinate (`[4, 1]`); with a same-kind `y`, `[1]` | ran |
| scaling, `x = [1, 0, 0, 0]` with an explicit zero at 2: `SparseIterate` `rmul!(x, Inf)`, `rdiv!(x, 0.0)`, `ldiv!(0.0, x)` give `[Inf, NaN, 0, 0]` and `lmul!(NaN, x)` `[NaN, NaN, 0, 0]`, working set `[1, 2]`; `x .*= Inf` and `x ./= 0.0` give `[Inf, NaN, NaN, NaN]`, working set `[1, 2, 3, 4]`; dense `rmul!` `[Inf, NaN, NaN, NaN]`; `AtomIterate` `rmul!(b, Inf)` (LinearAlgebra's generic method) gives `[Inf, NaN, NaN, NaN]` and activates group 2; `rmul!(x, 2.0)` gives `[2, 0, 0, 0]` and keeps `[1, 2]` | ran; `src/sparse_iterate.jl:398–425` |
| in `Float64`, `3 − √2` is `1.5857864376269049` and `(1 − √2/3)·3` is `1.5857864376269046`; the five-argument step of `GroupNormLinf(2, [1:2, 3:4])` at `[1, 4]` gives `[1.0, 2.0]` | ran |
| a checked `copyto!(x::SparseIterate, y::AbstractArray)` makes `copyto!(SparseIterate(2, 2), sparse([1.0 0; 0 2]))` ambiguous; a dense source works | ran; SparseArrays `sparsematrix.jl:616` |
| worked example: each import of FD-2's table gives `[4, 1]` (`AtomIterate` `[2, 1]`), `fill!` and `x .= 0` give `[]` | ran |
| `ldiv!(x, lu(I₄), y)` with `y` a `SparseIterate` holding coordinate 1 gives `[1]` (state copy through `copyto!`) | ran |
| `mul!(y, A, b)` into a `SparseIterate` keeps its old coordinate (`[1, 2]`); `sum!(y, M)` drops it (`[2]`) | ran |
| the owner's `copyto!(X, [3 0; 0 0])` after `X[2, 2] = 7` gives working set `[3, 1]`; `copyto!(coordinates(X), v)` reaches Base's generic `copyto!` and gives `[3, 1]`; `fill!(X, 0)` empties it | ran; `src/sparse_iterate.jl:1128`, `:1151–1161`; Base `abstractarray.jl:1073` |
| a failed owner import: with `X[1, 1] = 7` and `M = [1 2; 3 4]`, `copyto!(X, M)` throws `ArgumentError` and keeps `X` (`[7 0; 0 0]`, working set `[1]`); `fill!(coordinates(X), 0); dropzeros!(X); copyto!(X, M)` throws the same error and leaves `[0 0; 0 0]`, working set `[]` | ran |
| the array shorthand with an overlapping source: `x = SparseIterate([1., 2.])`, `copyto!(dropzeros!(fill!(x, 0)), view(x, 2:-1:1))` gives `[0.0, 0.0]`; `Base.mightalias(x, view(x, 2:-1:1))` is `true`, `Base.unalias(x, view(x, 2:-1:1))` is a new view of a copy holding `[2.0, 1.0]`, and the shorthand from it gives `[2.0, 1.0]` with working set `[1, 2]`; `Base.unalias(x, activevalues(x))` returns the handle itself | ran |
| `copy` keeps active zeros and order (`[3, 1]`, both types); `similar` and `zero` are empty; `copyto!(x, x)` is a no-op | ran |
| `copyto!(SparseIterate(2, 2), SparseIterate(4))`, `copyto!(SparseIterate(6), SparseIterate(4))`, `copyto!(AtomIterate(4, 2), AtomIterate(6, 3))` and `copyto!(AtomIterate(6, 3), AtomIterate(4, 2))` throw `DimensionMismatch` | ran; `src/sparse_iterate.jl:552`, `src/atom_iterate.jl:250` |
| `AtomIterate` `copyto!` into the layout `[[1, 3], [2, 4]]` from `[1:2, 3:4]` gives working set `[2, 1]`: the stale all-zero group 2 plus group 1 | ran; generic fallback `src/atom_iterate.jl:251–252` |
| `copyto!(x32, y64)` of `1e-50` gives values `Float32[0, 0, 0]` and working set `[2]` | ran |
| a stale handle after `dropzeros!` reads `[3, 2, 3]`; the new one `[3, 2]` | ran |
| a state copy reorders slots at equal membership: `x` with order `[1, 2]`, `y` with `[2, 1]`, `x == y`; after `copyto!(x, y)` (and after `copy!(x, y)`), `nactive` is 2, the reacquired `activecoordinates` is `[2, 1]` and `activevalues` `[2.0, 1.0]`, and an old handle's first entry reads 2; `copyto!(x, x)` keeps `[1, 2]`; `AtomIterate` likewise `[2, 1]` | ran |
| equality and hashing ignore the working set, both types | ran |
| groups are copied at construction; `copy`, `similar` and `AtomIterate(4, f)` share them | ran; `src/atom_iterate.jl:85–88`, `:236–245`, `src/functions/norms.jl:752–760` |
| `parentindices(atom(a, 1))[1] === a.groups[1]`, a `Vector{Int64}`; after `G[1] = 3`, `a.groups == [[3, 2], [3, 4]]`, `a.group_of == [1, 1, 2, 2]` | ran |
| the layout cache holds one object; two equal distinct layouts evict each other | ran; `src/atom_iterate.jl:116–122` |
| coordinate oracles reject a different layout with `DimensionMismatch("f and x must have the same groups.")`, `x` unchanged; `f(x)` and `prox!` fall back (`10.79669127533634` for both `fm(am)` and `fm(Vector(am))`) | ran; `src/functions/norms.jl:746–747`, `:765` |
| `kkt_violation` with a different layout throws `DimensionMismatch`; bad `k`, a short `u` and a complex `u` throw `BoundsError`, `DimensionMismatch`, `ArgumentError`, with `x` unchanged | ran |
| group `kkt_violation` is `3.585786437626905` (`5 − √2`, default weight `√2`); `NormL1` on a group gives `3.0` | ran; `src/interface.jl:568–570`, `src/functions/norms.jl:442–443` |
| the screens of the built-in group penalties at `g = [3, 4, 0, 0]`, group 1 inactive: `GroupNormL2` `3.585786437626905`, `GroupNormLinf` `6.0`, `SparseGroupNormL2(1, 1, …)` `2.1913377130908938` (`√13 − √2`), `GroupMCP` and `GroupSCAD` `3.585786437626905`; at `g = [5/8, 5/8]`, `GroupNormLinf(1, [1:2])` `0.25`, the ℓ₂ formula with weight 1 `−0.11611652351681556`, `GroupNormL2(1, [1:2])` `−0.5303300858899107`, `Scaled(GroupNormLinf(1, [1:2]), 2)` `−0.75`; `Regularize(GroupNormL2(1, [1:2]), 1, [1, 1])` at `[3, 4]` `2.1913377130908938`; default weights `1` for `GroupNormLinf` and `√2` (`1.4142135623730951`) for the others at `\|G\| = 2`; `IndBallGroupL0` has no `kkt_violation` (`MethodError`) | ran |
| the five-argument group step gives `[2.151471862576143, 2.868629150101524]` = `(1 − √2/5)·(3, 4)`; a step to zero keeps the group absent | ran |
| a trial `u = atom(y, 2)` aliasing `y` gives `[2.5, 3.5]`, the soft threshold of `[3, 4]` by `0.5` | ran; `src/interface.jl:474–476` |
| private staging followed by a throwing call leaves an inactive group holding `[3.0, 4.0]`; group value `0.0` against `7.0710678118654755`; a same-layout `copyto!` gives zeros, while `copy` keeps `[3.0, 4.0, 0.0, 0.0]` | ran; `src/atom_iterate.jl:244–245` |
| a trial `u` whose `getindex(2)` throws: `prox_coordinate!(x, f, 1, u, 1)` throws and leaves `x == [3, 0, 0, 0]` with group 1 active | ran; `src/interface.jl:506–515` |
| `prox_coordinate!` returns `Float64` for a `SparseIterate`, a `SubArray` for an `AtomIterate` | ran |
| malformed layouts: the six `ArgumentError` messages of the companion's edge-case list, also `AtomIterate(4, UnitRange{Int}[])` ("The groups must partition 1:4.") and `AtomIterate(0, 0)` ("num_groups must divide p.") | ran; `src/atom_iterate.jl:66–70`, `:92–114` |
| empty iterates: `SparseIterate(0)` and `AtomIterate(0, UnitRange{Int}[])` have no coordinates, `dropzeros!` is a no-op, `SparseIterate(0)[1]` and `atom(·, 1)` throw `BoundsError`; `SparseIterate(0, 3)` has size `(0, 3)`; `dot` and `mul!` with them give `0.0` and `[0, 0]` | ran |
| `SparseIterate(zeros(4))` and `AtomIterate(zeros(4), 2)` have no active coordinate; constructors from arrays activate the nonzeros in index (group) order: `[2, 3]` for `[0 1; 2 0]` and for `AtomIterate([0, 0, 1, 2, 3, 0], 3)` | ran; `src/sparse_iterate.jl:513–521`, `src/atom_iterate.jl:73–80` |
| complex element types throw `ArgumentError` at construction; `atom` has no method for a `SparseIterate` | ran; `src/sparse_iterate.jl:21`, `src/atom_iterate.jl:40` |
| `copyto!(SparseIterate(4), ones(5))` throws `BoundsError`; `copyto!(SparseIterate(4), [1, 2])` writes the first two entries; a value write of `1e-50` into `Float32` activates nothing | ran |
| `findnz` of a matrix with an explicit zero gives `([2, 2], [1, 2], [3.0, 0.0])` | ran |
| `kkt_violation(NormL1(1), [3, −0.5], SparseIterate(2), k)` is `2.0` for `k = 1` and `−0.5` for `k = 2` | ran |
| `dot(s, t) == 0.0`, `dot(t, s)` NaN; dense reference NaN | ran |
| `A * u` with a stored zero `[NaN, NaN]`, after `dropzeros!` `[1.0, 1.0]`; `dot(A[1, :], ·)` NaN and 1.0; dense `[NaN, NaN]` | ran |
| `AtomIterate`: `B * a == [1.0]`, `dot(v, a)` NaN with an active zero group, 1.0 after `dropzeros!`; dense NaN | ran; `src/atom_iterate.jl:307`, `:342–344` |
| owner `mul!(y, matrixview(X), [Inf, 1, 0])`: `[0, 1, NaN]` with a stored zero, `[0, 1, 0]` after `dropzeros!` | ran; `src/sparse_iterate.jl:1913–1927` |
| `(A * X * B)[1, :]`: `[0, 0, 0]` on the sparse path, `[NaN, NaN, NaN]` on the dense fallback; the term-wise sum gives `[0, 0, 0]` for both | ran; `src/sparse_iterate.jl:888` |
| `A * X * B` against the term-wise sum: sparse path with an `Inf` meeting a block zero, row 2 `[NaN, NaN, NaN, NaN]` against `[Inf, Inf, Inf, Inf]`; dense fallback with a NaN of `B` meeting a zero column, column 1 `[NaN, NaN, NaN, NaN]` against `[9.0, 8.0, 7.0, 6.0]`; association, row 1 `[Inf, Inf, NaN, NaN]` for `A * (Matrix(X) * B)`, `[NaN, NaN, NaN, NaN]` for `(A * Matrix(X)) * B`, the package and the term-wise sum; a finite 5 × 5 case is all finite and within `2.1421476825068971e-16` of the term-wise sum | ran; `src/sparse_iterate.jl:877–902` |
| `A * Xm` for a `StridedMatrix` and a matrix `SparseIterate` reaches the package's `mul!` | ran; `src/sparse_iterate.jl:698` |
| overflow of a finite partial product: with `X[1, 1] = 1e200`, `A[1, 1] = 1e200`, `B[1, 1] = 1e-200`, the exact entry (in `BigFloat`) is `1.0e200`; the package and `A * (Matrix(X) * B)` give `1.0e200`; `(A * Matrix(X)) * B` and `(A[1, 1] * X[1, 1]) * B[1, 1]` give `Inf` | ran |
| `A * X` against `A * X * I`, with `A` the `Symmetric` diagonal of `(Inf, 1, 1, 1)` and `X` nonzero at `(1, 1)`, `(1, 2)`, `(2, 1)`, `(3, 1)`: the package's `A * X` has row 1 `[Inf, Inf, 0, 0]`, the term-wise `A * X * I` `[NaN, NaN, NaN, NaN]` | ran |
| mixed element types reach `*(A::Symmetric, X::Union{…}, B::Symmetric)`, which calls LinearAlgebra's generic three-argument `*` | ran (`which`); `src/sparse_iterate.jl:1951–1955` |
| `matrixview(X) * v` reaches the package's `mul!` for the view and gives `[0, 1, NaN]` for `v = [Inf, 1, 0]` with an explicit zero at `(3, 1)`; the matrix–matrix `mul!(C, matrixview(X), B)` is LinearAlgebra's generic method (`[NaN, NaN, NaN]` for `B = [Inf; 1; 0;;]`); `dot(M, matrixview(X))` is LinearAlgebra's generic `dot` (NaN with a NaN of `M` at the explicit zero); the owner's own 0.5.x `dot(M, X)` also gives NaN there, as it multiplies stored zeros | ran; `src/sparse_iterate.jl:996–997`, `:1839–1849` |
| `fill!(coordinates(X), 0)` is Base's generic `fill!`; the owner's own `fill!(X, 0)` clears through `_clear!` | ran (`which`); Base `array.jl:356`; `src/sparse_iterate.jl:1574–1584` |
| the owner's `coordinates(X) .= ComplexF64[1, 2im, 0]` throws `InexactError` and leaves `[1.0, 0, 0]` | ran |
| a stand-in coordinate view with `dataids` of `x` and no `unaliascopy` makes `copyto!(x, view)` throw `ArgumentError`; with `unaliascopy`, `copyto!(x, view)` leaves `[1, 2, 3, 4]` unchanged and `copyto!(x, 2, view, 1, 3)` gives `[1, 1, 2, 3]`, as `copyto!(x, 2, x, 1, 3)` does today | ran; Base `abstractarray.jl:1145` (offset `copyto!` unaliases) |
| the coefficients that `_scaled_dot` and `_split_dot` meet are finite: normals are checked at construction, and `LogDetLoss` takes the `_split_dot` path only for a finite `S` | read; `src/functions/sets.jl:690`, `src/functions/logdet.jl:110` |
| `dot([NaN, 1], Z, [1, 1])`: NaN with a stored zero, 1.0 after `dropzeros!` | ran; `src/sparse_iterate.jl:858–868` |
| generic fallbacks: `sparse(A) * s` `[NaN, NaN]`; `Diagonal([NaN, 1]) * s` `[NaN, 1.0]`; `dot(sparsevec, s)`, `dot(view(v, idx), s)`, `sum(s .* v)` NaN; the package's `Matrix * s` and `dot` give 1.0; `AtomIterate` likewise; `transpose(permutedims([NaN 0 1 0])) * a` is LinearAlgebra's generic `*` and gives `[NaN]` where `[NaN 0 1 0] * a` gives `[1.0]` | ran |
| `dot(atom(a, 1), [NaN, 1])` with `atom(a, 1) == [0, 1]` is NaN, through the package's view method | ran; `src/atom_iterate.jl:354–357` |
| `vec` against the vector's own `coordinates`: `copyto!(coordinates(x), coordinates(y))` gives `y`'s working set `[1]` for N = 1 and `[4, 1]` through `vec` for N = 2; `dot([1, NaN, 0, 0], ·)` gives `1.0` for N = 1 and NaN through `vec`; `norm(vec(xm))` is LinearAlgebra's generic `norm`; `Base.dataids(vec(xm)) == Base.dataids(xm)` | ran |
| `vec(s) === s`, `vec(a) === a` and `reshape(s, 2) === s` for vectors; `dot([NaN, 1.], vec(s)) == dot([NaN, 1.], s) == 1.0`, and the same for an `AtomIterate`; `vec(s) .= ComplexF64[1, 2im]` and `vec(a) .= …` throw `ArgumentError` before writing; `dot([NaN, 1.], view(s, :))` is NaN; for a matrix, `dot([NaN, 0, 0, 1.], vec(m))` is NaN and `vec(m) .= ComplexF64[1, 2im, 0, 0]` writes `1.0` and then throws `InexactError` | ran |
| the three-factor mixed fixture: `u = 2.0^600`, `A = Symmetric(diag(Inf, u))`, `B = Symmetric(diag(1, 1/u))`, `X = diag(1, u)`; the package gives `[Inf NaN; 0.0 4.149515568880993e180]`; `(u*u)*(1/u)` is `Inf`, `u*(u*(1/u))` is `4.149515568880993e180`, and the exact product is `u`; recomputing only the non-finite entries term by term gives `[Inf NaN; 0.0 4.149515568880993e180]`, while a full left-to-right term-wise recomputation gives `Inf` at `(2, 2)` and NaN at `(2, 1)` | ran |
| no audited ProximalSolvers path copies into an iterate while a slot-keyed history is live: the copies of `src/cd/glm.jl`, `src/cd/generic_newton.jl` and `src/cd/lla.jl` run between `_cd!` runs, each of which starts new momentum and Anderson histories | read; ProximalSolvers `src/cd/solver.jl:464–475`, `:478–489`, `:608`, `src/cd/glm.jl:735`, `src/cd/generic_newton.jl:126`, `src/cd/lla.jl:247` |
| `LeastSquaresLoss([1, 1], [NaN 1; NaN 1])`: NaN at a stored zero, 0.0 at an unstored zero, NaN at a dense point; `AtomIterate`: 0.0 against dense NaN | ran |
| of the data losses checked, `LeastSquaresLoss` (`src/functions/leastsquares.jl:48–59`) and `QuadraticFunction` (`src/functions/smooth.jl:345–351`) do not check their data for finiteness, `GLMLoss` checks weights and offset only (`src/functions/glm.jl:190–193`), and `KronQuadratic` checks (`src/functions/smooth.jl:492–493`); the others were not checked | read |
| `mul!` with `α = 0` ignores NaN in `A` (`[5.0, 5.0]`); with `β = 0` ignores NaN in `out` (`[1.0, 0.0]`); `false * NaN === 0.0` | ran; `src/sparse_iterate.jl:685`, `:757` |
| SparseArrays: dense × sparse vector skips stored zeros (`[1.0, 1.0]`); `dot(dense, sparse)` with a stored zero NaN; CSC × dense `[NaN, 1.0]` with a stored zero; dense OpenBLAS `[NaN, NaN]` | ran; `sparsevector.jl:1812`, `:1942–1964` (Julia 1.13.3) |
| `prox!` destinations by branch: `NormL2` `[4, 1]`, `[4, 1]`, `[]`; `GroupNormL2` on `AtomIterate` `[2, 1]`, `[2, 1]`, `[2]`; `IndBallL0` `s = 0, 1, 3`: `[4]`, `[1, 2]`, `[1, 2]` | ran; `src/functions/norms.jl:59–81`, `src/functions/sets.jl:162–190` |
| 35 sites spell `copyto!(y, x)` in `src/` | read (`grep`) |
| activation from the converted value on every write path, both types, except `AtomIterate` `fill!` and `x .= 1e-100`, which activate both groups at `0f0` | ran; `src/atom_iterate.jl:273` |
| NaN and `±Inf` activate, `−0.0` does not; `dropzeros!` keeps NaN and `Inf` | ran |
| complex sources: `copyto!`, `x[:] =` and `view(x, 1:2) .=` write `1.0` and throw `InexactError`; `x .= c` and `x .= 2im` throw `ArgumentError` before writing; `fill!(x, 2im)` throws `InexactError` before writing | ran |
| `vec(xm) .= c` on a matrix writes, then `InexactError`; `coordinates(xv) .= c` on a vector throws `ArgumentError` first | ran |
| `mightalias(activevalues(x), x)` and `mightalias(activecoordinates(x), x)` `false` (S); `mightalias(activecoordinates(a), a)` `false`, `mightalias(atom(a, 1), a)` `true` (A); both `true` for the owner; `SparseIterate` uses Base's default `dataids` | ran; `src/atom_iterate.jl:133`, `src/sparse_iterate.jl:940–943`, Base `abstractarray.jl:1611` |
| `copyto!(x, activevalues(x))` after `x[2] = 1; x[1] = 2` gives `[1, 1]` | ran |
| retained memory (`Base.summarysize`): `SparseIterate(Float64, 10^6)` 24000152 bytes (24.000152 per entry), `Float32` 20000152, `SparseIterate(Float64, 1000, 1000)` 24000160; `AtomIterate(Float64, 10^6, 10^5)` 19200256 (19.200256 per entry), with vector groups of 10 30400256 (30.400256); owner `p = 1000` 12012160 (24.00031968031968 per coordinate) | ran |
| `coordinates` sweep, 300 × 300: 1.248 ms through `m`, 1.31 ms through `vec(m)`, ratio 1.049, 0 bytes | ran (one exploratory run) |
| multi-response layout: `x[(c − 1)p + r] == B[r, c]`, `atom(x, r) == B[r, :]`, group 2 is `[2, 5]`; `gradient!` of `LeastSquaresLoss(Y, X)` with a matrix `Y` at this `AtomIterate` throws `DimensionMismatch` | ran |
| #189's raw-view failure is fixed: `atom(x, k)` is `view(x, G)`; regression tests | read; `src/atom_iterate.jl:163–166`, `test/atom_iterate.jl:244–269` |
| ProximalSolvers stages group trials in `x.storage` | read; ProximalSolvers `src/cd/groups.jl:32–38`, `:64–70` |
| ProximalSolvers multi-response groups `collect(k:p:p*q)` and layout check | read; ProximalSolvers `src/cd/multiresponse.jl:45`, `:55` |
| ProximalSolvers `OptimizationResults` needs an `AbstractArray` minimizer | read; ProximalSolvers `src/types.jl:210` |
| ProximalSolvers relies on `fill!(x, 0)` clearing | read; ProximalSolvers `src/cd/path.jl:261`, `src/workspaces.jl:243`, and on an owner `src/cd/quic.jl:91` (with `:77`, `:100`) |
| ProximalSolvers reads `x.groups` 23 times by index and 5 times whole, and other private fields 16 times | read (`git grep`; list in the companion §3) |

*Released versus audited versus proposed.* "0.5.4" is the `Project.toml` of
the audited commit; the General registry was not checked. Everything in
Part 2 that differs from this table is proposed 0.6 behavior.
