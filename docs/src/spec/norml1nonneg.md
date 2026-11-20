# [NormL1Nonneg numerical contract](@id spec-norml1nonneg)

This page gathers the existing numerical text without changing its wording.
The guide retains its mathematical introduction and ordinary examples. Clause
conversion and status assignments remain a separate review step; prose without
an explicit status remains unclassified.

Read [How to read the numerical contracts](index.md) for authority, status
meanings and the migration map.

## [Identity](@id spec-norml1nonneg-identity)

This section contains links only. Constructors and API help:
[`NormL1Nonneg`](@ref). The [`NormL1Nonneg` row in the API coverage
matrix](../api/coverage.md) lists the retained interface.

Source: [`src/functions/norml1nonneg.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/norml1nonneg.jl).
The [evidence section](@ref spec-norml1nonneg-evidence) identifies its tests.

## [Definition and domain](@id spec-norml1nonneg-definition)

The mathematical definition and constructor table remain in
[The nonnegative ℓ₁ penalty](../penalties.md#The-nonnegative-l1-penalty).
The following domain text and examples move from that guide verbatim.

```@setup nonneg_l1
using ProximalBase
f = NormL1Nonneg(0.5, [1.0, 2.0])
```

### The domain

The domain is ``x_i\geq0`` exactly: `-0.0` and `+Inf` are feasible, and any negative number,
a subnormal one included, is not. The value is `Inf` where an entry that is read is negative and
`NaN` where one is `NaN`; `NaN` takes precedence over `Inf`. A `+Inf` entry whose positive finite coefficient
``\lambda_0w_i`` rounds to zero also gives `NaN`, as for `NormL1`
([#655](https://github.com/mlakolar/ProximalBase.jl/issues/655)), which then takes precedence over the `Inf` of a negative
entry. A zero level or a zero weight keeps the domain. With finite weights, `NormL1Nonneg(0.0)` is the indicator of the orthant, with value `0` at a
feasible point (`+Inf` entries included), and its proximal operator is the
projection ``\max(v,0)`` onto the orthant, not the identity: `-0.0` and `-Inf` give `+0`, `+Inf` is
kept and `NaN` stays `NaN`. The projection is the result for an entry in these cases:

* a zero weight, at every level and step (an infinite level and an infinite step included);
* a zero level with a finite weight;
* a zero step with a finite level and a finite weight;
* an infinite step with a zero level or a zero weight.

An infinite weight, and an infinite level with a positive weight, follow the rules below instead
(`NormL1Nonneg(0.0, [Inf])([1.0])` is `Inf`, and `prox(NormL1Nonneg(Inf), [3.0], 0.0)` is `[0.0]`). An infinite step with a
positive finite level and a positive finite weight maps finite inputs and `-Inf` to `+0` and `+Inf` and `NaN`
to `NaN`.

```@example nonneg_l1
@assert f([0.0, -0.0]) == 0 && f([1.0, -1e-300]) == Inf
@assert isnan(f([NaN, -1.0]))
@assert NormL1Nonneg(0.0)([Inf, 1.0]) == 0 && NormL1Nonneg(0.0)([-1.0]) == Inf
@assert isequal(prox(NormL1Nonneg(0.0), [-2.0, -0.0, 3.0, Inf, -Inf, NaN], 1.0), [0.0, 0.0, 3.0, Inf, 0.0, NaN])
@assert prox(f, [3.0, 3.0], 0.0) == [3.0, 3.0]
```

At a feasible point the value is the value of `NormL1` with the same level and weights, with its
recovery of terms below `floatmin` (see the [numerical contract](../numerics.md)): bit for bit on a
`SparseIterate`, a `BlockIterate` and a `SymmetricSparseIterate`, and within the error bound of the sum on a
dense array. `value_and_prox!` returns the value at the stored output.

## [Traits](@id spec-norml1nonneg-traits)

This section contains links only. The existing trait description remains in
[The nonnegative ℓ₁ penalty](../penalties.md#The-nonnegative-l1-penalty) and
the [`NormL1Nonneg`](@ref) API entry.

## [Supported arguments](@id spec-norml1nonneg-arguments)

### Weights, shapes and representations

The array `w` has one weight per entry of the argument, paired by Julia's linear index, and is
stored by reference. `NormL1Nonneg` accepts real dense vectors and matrices (views included),
`SparseIterate`, `BlockIterate` and `SymmetricSparseIterate`, with a scalar step or, except as
noted below, an array of per-entry steps. On a `SparseIterate` or a `SymmetricSparseIterate` the
value reads the stored entries only, and so does the proximal operator from such an iterate into an
iterate of the same kind; an entry stored in an iterate destination stays stored, possibly as a zero,
and a zero result does not activate an absent entry. On a `SymmetricSparseIterate` the weights, which may be omitted,
have `p²` entries in full-matrix linear order, read as a `p × p` array at both positions of an off-diagonal
coordinate, and the value counts such a coordinate twice. The weights must then be exactly symmetric, a
precondition that the oracles assume and `check_symmetric_parameters` validates (see
[Symmetric parameters](../symmetric_parameters.md)).

Not supported: an array of steps with a `SymmetricSparseIterate` destination (an `ArgumentError`
before any write), or with a dense destination and a `SymmetricSparseIterate` argument (a
`MethodError`); the vector form of `prox_block`; complex arguments or weights.

## [Preconditions](@id spec-norml1nonneg-preconditions)

This section contains links only. Weight and representation conditions are in
[Supported arguments](@ref spec-norml1nonneg-arguments), and step conditions are
in [Endpoint clauses](@ref spec-norml1nonneg-endpoints).
Shared conditions remain in
[Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation)
and [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement).

## [Results](@id spec-norml1nonneg-results)

This section contains links only. The stored-output value rule remains with
[The domain](@ref spec-norml1nonneg-definition), and the value/projection cases
are in [Endpoint clauses](@ref spec-norml1nonneg-endpoints).
Return signatures remain in the [`NormL1Nonneg`](@ref) API entry.

## [Ordinary-path clauses](@id spec-norml1nonneg-ordinary)

The ordinary proximal formula and introductory example remain in
[The nonnegative ℓ₁ penalty](../penalties.md#The-nonnegative-l1-penalty).
The following KKT, restriction and scaling text moves without assigning
clause identifiers or statuses.

### KKT scores, restriction and scaling

For finite inputs and ``c_i=\lambda_0w_i``, the screen of a coordinate assumed zero is ``-v_i-c_i``, and
`kkt_violation(g,v,x,k)` is ``-v_k-c_k`` at a zero coordinate, ``\lvert v_k+c_k\rvert`` at a
positive one and `Inf` at a negative one, which is outside the domain. The full-point value is the
``\ell_\infty`` distance from ``-v`` to ``\partial g(x)`` for a finite point in the domain, and `Inf` at a
finite point with a negative entry. The rules for `NaN` and infinite entries are in the
[numerical contract](../numerics.md).

```@example nonneg_l1
h = NormL1Nonneg(0.5, [1.0, 2.0, 4.0])
v = [0.1, -4.0, 0.8]
@assert kkt_violation(h, v, 2) ≈ 4.0 - 0.5 * 2.0
@assert kkt_violation(h, v, [0.0, 0.0, 0.25], 1) ≈ -0.1 - 0.5
@assert kkt_violation(h, v, [0.0, 0.0, 0.25], 3) ≈ 0.8 + 0.5 * 4.0
@assert kkt_violation(h, v, [0.0, -1.0, 0.25], 2) == Inf
@assert kkt_violation(h, v, [0.0, 0.0, 0.25]) ≈ 3.0
@assert kkt_violation(h, v, [0.0, -1.0, 0.25]) == Inf
```

`restrict(h, S)` keeps the level and selects the weights. `Scaled` folds a factor into the level where the
product is a level that the type can hold, and otherwise takes the other routes of
[Scaling a function](../calculus.md). For a finite level a zero factor keeps the domain and the infinite weights:
`0 * NormL1Nonneg(λ0, w)` is `NormL1Nonneg(0, w)`. An infinite level with a zero factor throws an `ArgumentError`.

```@example nonneg_l1
r = restrict(h, [3, 1])
@assert r isa NormL1Nonneg && r.w == [4.0, 1.0] && r([1.0, 1.0]) == 0.5 * 5.0
s = Scaled(h, 2.0)
@assert s([1.0, 1.0, 1.0]) == 2.0 * h([1.0, 1.0, 1.0])
```

## [Range clauses](@id spec-norml1nonneg-range)

The recovery referred to as “above” in the preserved text is owned by
[NormL1 range clauses](@ref spec-norml1-range). The table referred to as
“below” is in [Endpoint clauses](@ref spec-norml1nonneg-endpoints).

`NormL1Nonneg` values take the `NormL1` loops with an `Inf` term in place of the
term of a negative entry; the rules for `NaN` and `Inf` entries are in the table below, and at a
feasible point the recovery above applies to the same terms. At such a point the
value of a dense array agrees with that of `NormL1` within the error bound of the sum, and on the
stored entries of `SparseIterate`, `BlockIterate` and `SymmetricSparseIterate` the loops are
sequential and the values are equal bit for bit.
Where a positive finite level and weight have a product that rounds to zero and the entry is
`+Inf`, the value is `NaN`, as for `NormL1` (the retained product forms `0*Inf`,
[#655](https://github.com/mlakolar/ProximalBase.jl/issues/655)).

## [Endpoint clauses](@id spec-norml1nonneg-endpoints)

### Steps and endpoint rules

`prox`, `prox!` and `value_and_prox!` throw an `ArgumentError` for a negative or `NaN` scalar step
before anything is written. The entries of an array of steps are not checked: they must be
nonnegative, a documented precondition. `prox_block` does not check its step, and `Scaled` keeps
its own step rule; a negative step that reaches the function through `Scaled` violates the
precondition ``\gamma\geq0``, and its result can depend on the representation: a sparse iterate and
a dense array disagree only where the negative step gives a strictly negative threshold at an entry that is not stored,
whose proximal point is then positive and is not visited by a loop over the stored entries.

```@example nonneg_l1
y = [7.0, 7.0]
refused = try prox!(y, f, [3.0, -2.0], -1.0); false catch e; e isa ArgumentError end
@assert refused && y == [7.0, 7.0]
```

An infinite level fixes each entry with a positive weight at `+0` at every step ``\gamma\geq0``,
``\gamma=0`` included, for `NaN` and `±Inf` inputs too (the rule of `NormL1`); entries with a zero
weight are projected onto the orthant. An infinite weight at a finite level maps finite inputs and
`-Inf` to `+0`, also at zero level and step, and `+Inf` and `NaN` to `NaN` (see
[#853](https://github.com/mlakolar/ProximalBase.jl/issues/853)); its value is `0` at `0` and `Inf`
at any other number, also at a zero level. A threshold formed from finite factors that overflows
keeps the documented limitation of `NormL1` for an input of `+Inf`
([#866](https://github.com/mlakolar/ProximalBase.jl/issues/866)).

```@example nonneg_l1
@assert prox(NormL1Nonneg(Inf, [1.0, 0.0]), [NaN, -2.0], 0.0) == [0.0, 0.0]
@assert prox(NormL1Nonneg(Inf, [1.0, 0.0]), [NaN, 4.0], 0.0) == [0.0, 4.0]
@assert prox(NormL1Nonneg(1.0, [Inf]), [3.0], 1.0) == [0.0]
@assert isnan(prox(NormL1Nonneg(1.0, [Inf]), [Inf], 1.0)[1])
@assert NormL1Nonneg(1.0, [Inf, 1.0])([0.0, 2.0]) == 2.0 && NormL1Nonneg(1.0, [Inf, 1.0])([1e-300, 2.0]) == Inf
```

### Endpoint table and executed checks

The following rows and example are retained from the common numerical page.

| Case | Retained behavior |
|:--|:--|
| `NormL1Nonneg`, domain | The domain is `x ≥ 0` exactly: `-0.0` and `+Inf` are feasible, any negative number, a subnormal one included, is not. The value is `Inf` where an entry that is read is negative and `NaN` where one is `NaN` (`NaN` takes precedence); a `+Inf` entry whose positive finite coefficient `λ0 w` rounds to zero also gives `NaN`, as for `NormL1` ([#655](https://github.com/mlakolar/ProximalBase.jl/issues/655)), and then takes precedence over the `Inf` of a negative entry (`NormL1Nonneg(5.0e-324,[5.0e-324,1.0])([Inf,-1.0])` is `NaN`); a zero level or weight keeps the domain. The value is `0` at a feasible point, `+Inf` entries included, for a zero weight and for a zero level with a finite weight; an infinite weight at a zero level has the value of its own row below (`NormL1Nonneg(0.0,[Inf])([1.0])` is `Inf`). A proximal point is `+0`, positive or `NaN`: `-0.0` gives `+0.0` |
| `NormL1Nonneg`, zero weight; zero level with a finite weight; zero step with a finite level and weight | The projection `max(v, 0)` onto the orthant, not the identity: `-Inf` and `-0.0` give `+0`, `+Inf` and `NaN` are kept. A zero weight projects at every level and step, an infinite level and an infinite step included. An infinite weight, and an infinite level with a positive weight, follow their own rows |
| `NormL1Nonneg`, infinite level, nonnegative step | Positive-weight entries prox to `+0` at every step, `γ = 0` included, for non-finite inputs too (the `NormL1` rule, #665); zero-weight entries are projected. The value of a positive-weight entry is `0` at `0`, `NaN` at `NaN` and `Inf` otherwise |
| `NormL1Nonneg`, infinite weight, finite level | Finite inputs and `-Inf` map to `+0`, also at a zero level or step; `+Inf` and `NaN` map to `NaN`, as for `NormL1` ([#853](https://github.com/mlakolar/ProximalBase.jl/issues/853)). The value is `0` at `0`, `Inf` at any other number, `NaN` at `NaN`, also at a zero level |
| `NormL1Nonneg`, finite factors whose threshold overflows, or an infinite step with a positive finite level and a positive finite weight | Finite inputs and `-Inf` give `+0`; `+Inf` gives `NaN` (as `NormL1`, deferred #866); `NaN` gives `NaN`. An infinite step with a zero level or a zero weight is the projection: `prox(NormL1Nonneg(0.0),[3.0],Inf)` is `[3.0]` |
| `NormL1Nonneg`, negative or `NaN` scalar step | `prox`, `prox!` and `value_and_prox!` throw `ArgumentError` before any write; the entries of an array of steps are not checked, and `prox_block` and `Scaled` routes do not check. A negative step through `Scaled` violates the precondition `γ ≥ 0`, and its result can depend on the representation |

```@setup norml1nonneg_endpoints
using ProximalBase
```

```@example norml1nonneg_endpoints
@assert isequal(prox(NormL1Nonneg(0.0),[-2.0,-0.0,3.0,Inf,-Inf,NaN],1.0), [0.0,0.0,3.0,Inf,0.0,NaN])  # the projection
@assert prox(NormL1Nonneg(Inf,[1.0,0.0]),[NaN,-2.0],0.0) == [0,0]   # the NormL1 rule (#665), a zero weight projects
@assert NormL1Nonneg(0.0)([Inf,1.0]) == 0 && NormL1Nonneg(0.0)([-1.0]) == Inf
@assert isnan(NormL1Nonneg(2.0)([NaN,-1.0])) && NormL1Nonneg(2.0)([1.0,-1e-300]) == Inf
@assert try prox(NormL1Nonneg(1.0),[1.0],-1.0); false catch e; e isa ArgumentError end
prox(NormL1Nonneg(0.0),[-2.0,-0.0,3.0,Inf,-Inf,NaN],1.0)
```

## [Failure behavior](@id spec-norml1nonneg-failures)

This section contains links only. The moved text records unsupported
representations in [Supported arguments](@ref spec-norml1nonneg-arguments)
and validation before writes in [Endpoint clauses](@ref spec-norml1nonneg-endpoints).
Shared rules remain in
[Predictable failures and partial writes](../numerical_usage.md#Predictable-failures-and-partial-writes)
and [Validation, aliasing and evidence](../numerics.md#Validation,-aliasing-and-evidence).

## [Limitations and non-guarantees](@id spec-norml1nonneg-limitations)

This section contains links only. The retained non-finite-input and finite-factor
overflow limitations are in [Endpoint clauses](@ref spec-norml1nonneg-endpoints).
Representation exclusions remain in [Supported arguments](@ref spec-norml1nonneg-arguments).
The shared scaling limits remain in
[Range guarantees](@ref numerical-value-range) and
[Unsupported cases](../calculus.md#Unsupported-cases).

## [Clauses owned elsewhere](@id spec-norml1nonneg-elsewhere)

This section contains links only to the shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| `NormL1` value recovery | [NormL1 range clauses](@ref spec-norml1-range) |
| Shared range bounds and scaled-level recovery | [Range guarantees](@ref numerical-value-range) |
| Infinite penalties at a zero step | [Infinite penalties at zero step](@ref infinite-penalty-at-zero-step) |
| Shared KKT type, accuracy and non-finite rules | [NaN, infinities and zeros](../numerics.md#NaN,-infinities-and-zeros) |
| Sparse products and signed-zero storage | [Sparse product semantics](@ref Sparse-product-semantics) |
| Ownership, aliases and output writes | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Symmetric parameter preconditions | [Symmetric parameters](@ref symmetric-parameter-guide) |
| Scaling and unsupported wrapper cases | [Scoring optimality](@ref scaled-kkt) and [Unsupported cases](../calculus.md#Unsupported-cases) |

## [Evidence](@id spec-norml1nonneg-evidence)

These files identify tested domains and reference methods, not new guarantees.

| Test file | Reference or coverage relevant to this page |
|:--|:--|
| [`test/norml1nonneg.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/norml1nonneg.jl) | Constructors, traits, representations, validation before writes, stored-output values and restriction; proximal points compared with constrained minima and the one-sided threshold formula in BigFloat |
| [`test/norml1nonneg_extremes.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/norml1nonneg_extremes.jl) | Endpoint truth tables, values against BigFloat sums, point-aware KKT scores, full-point measures and `Scaled` routes |

The moved examples on this page execute during the strict documentation build.

## [History](@id spec-norml1nonneg-history)

The [restoration issue #947](https://github.com/mlakolar/ProximalBase.jl/issues/947)
records the family restoration. The documentation migration is tracked in
[#1019](https://github.com/mlakolar/ProximalBase.jl/issues/1019).
Historical records do not expand the supported behavior.
