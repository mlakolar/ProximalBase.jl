# [NormL1 numerical contract](@id spec-norml1)

This pilot gathers the existing `NormL1` numerical text without changing its
wording. The original guide and common-contract anchors remain as forwarding
links. Clause conversion and status assignments are a separate review step;
prose without an explicit status remains unclassified.

Read [How to read the numerical contracts](index.md) for authority, status
meanings and the current migration map.

## [Identity](@id spec-norml1-identity)

Constructors and API help: [`NormL1`](@ref).
The [`NormL1` row in the API coverage matrix](../api/coverage.md) lists the
retained interface.

The source checked for this pilot is
[`src/functions/norml1.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/7ae86e2093060eb124211e2f21f043c6379f97ae/src/functions/norml1.jl) at
`7ae86e2093060eb124211e2f21f043c6379f97ae`. The [evidence section](@ref spec-norml1-evidence)
maps the numerical text to retained test files.

## [Definition and domain](@id spec-norml1-definition)

The mathematical definition, constructor arguments and ordinary weighted
penalty interpretation remain in
[Mathematical definition and Julia constructor](../penalties.md#Mathematical-definition-and-Julia-constructor).
The endpoint text below retains the distinctions between finite parameters,
infinite weights and an infinite overall level.

## [Traits](@id spec-norml1-traits)

See the [function-family description](../penalties.md#Mathematical-definition-and-Julia-constructor),
the [`NormL1`](@ref) API entry, and
[Function objects and their properties](../function_objects.md) for the trait
interface. This migration adds no trait promise.

## [Supported arguments](@id spec-norml1-arguments)

`NormL1` acts on real vector or matrix entries. When weights are given,
they must have the same total number of entries as the argument, paired
in Julia's linear indexing order. The function also has explicitly
supported sparse and symmetric iterate operations; for symmetric matrix
storage, matching off-diagonal weights must agree: this is a precondition that the
oracles assume and `check_symmetric_parameters` validates. See
[Symmetric parameters](../symmetric_parameters.md).

The complete operation and representation boundaries remain in
[Operations on functions](../operations.md) and the
[`NormL1`](@ref) API entry.

## [Preconditions](@id spec-norml1-preconditions)

Constructor conditions are in the [`NormL1`](@ref) API entry.
Shared array and precision conditions are in
[Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation)
and [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement).
For symmetric storage, follow [Symmetric parameters](@ref symmetric-parameter-guide).

## [Results](@id spec-norml1-results)

`prox!` writes to a supplied destination; `value_and_prox!` writes
the proximal point and returns `g` evaluated at that stored output.
Exact input/output identity is supported, but partially overlapping
views are not. Weights are stored by reference, so modifying them also
changes the function.

## [Ordinary-path clauses](@id spec-norml1-ordinary)

The ordinary formula and executable examples remain in
[Proximal operator: soft thresholding](../penalties.md#Proximal-operator:-soft-thresholding)
and [Other operations](../penalties.md#Other-operations).
This verbatim pilot does not assign clause identifiers to those statements.

## [Range clauses](@id spec-norml1-range)

`NormL1` values also recover terms `λ0*w[i]*abs(x[i])` that round into the
subnormal range, which loses accuracy even when the total is normal
([#373](https://github.com/mlakolar/ProximalBase.jl/issues/373)). Each such
rounding loses at most half the smallest subnormal spacing,
`eps(T)*floatmin(T)/2`, so an ordinary sum `s ≥ n*floatmin(T)` of `n` terms
loses at most `eps(T)/2` relative to `s` and is returned from the unchanged
vectorized loop. A smaller sum, a zero one included, is recomputed: the
products `w[i]*abs(x[i])` are accumulated with their exponents, and `λ0` is
applied before the single conversion to `T`. This holds for dense arrays and
for the stored entries of `SparseIterate`, `BlockIterate` and
`SymmetricSparseIterate`, and for `value_and_prox!` at its stored output.
Zero and infinite levels, infinite weights and NaN/Inf entries keep the rules
below. Under an infinite `Scaled` factor a recovered positive value gives `Inf`,
as the folded level `NormL1(Inf, w)` does; before #373 a value whose terms all
rounded to zero gave `NaN` there. An exactly zero value still gives `0`.

## [Endpoint clauses](@id spec-norml1-endpoints)

Infinite parameters, NaNs, zeros, overflow and sparse active-zero
semantics require extra care. A zero weight leaves an entry unpenalized.
An infinite weight sets the proximal output to zero for a **finite input value**, including at zero
level and zero step. For non-finite inputs (`NaN`, `±Inf`) with finite level,
there is **no fixed-zero guarantee**: ordinary soft-threshold arithmetic can
return `NaN` ([#853](https://github.com/mlakolar/ProximalBase.jl/issues/853)).
A genuinely infinite overall level `λ0 = Inf` separately fixes every
positive-weight entry at `+0` at any nonnegative step, including non-finite
inputs, also at zero step. See the canonical
[infinite-penalty rule](@ref infinite-penalty-at-zero-step) for its interpretation.
Zero-weight entries stay free:

```julia
prox(NormL1(Inf, [1.0, 0.0, 2.0]), [3.0, 4.0, 5.0], 0.0)  # [0.0, 4.0, 0.0]
prox(NormL1(Inf), [1.0, 2.0], 0.0)                         # [0.0, 0.0]
prox(NormL1(1.0), [1.0, 2.0], 0.0)                         # [1.0, 2.0]: finite level
```

The infinite overall level and an infinite coordinate weight are distinct: the
latter, at a finite level, keeps the finite-input restriction above at every step.

| Case | Retained behavior |
|:--|:--|
| Unweighted `NormL1`, zero level | Zero value on dense float arrays, including NaN entries |
| `NormL1`, zero weight | Unpenalized entry at every level |
| `NormL1`, infinite weight | Finite input values map to zero, even with zero level or step. No fixed-zero guarantee for `NaN`/`±Inf` inputs with finite level; `shrink` can return `NaN`. Value is `Inf` away from zero. See [#853](https://github.com/mlakolar/ProximalBase.jl/issues/853). |
| `NormL1`, infinite level, nonnegative step | Positive-weight entries prox to `+0`, including non-finite inputs; zero-weight entries stay free. See [the zero-step rule](@ref infinite-penalty-at-zero-step) |
| `NormL1`, finite factors whose product overflows | Distinguished from genuinely infinite weights/levels; recovery handles representable finite results where supported. With an input of `±Inf`, an overflowed finite threshold can yield `NaN`; [#866](https://github.com/mlakolar/ProximalBase.jl/issues/866) is a documented, deferred limitation, not an authorized hot-loop fix. |

```@example norml1_numerical_endpoints
using ProximalBase
@assert NormL1(0.0)([NaN,Inf]) == 0
@assert prox(NormL1(Inf),[NaN,Inf],1.0) == [0,0]
@assert prox(NormL1(Inf),[1.0,2.0],0.0) == [0,0]  # zero step: projection onto {0} (#665)
@assert isequal(prox(NormL1(Inf,[1.0,0.0,2.0]),[3.0,NaN,Inf],0.0), [0,NaN,0])
@assert prox(NormL1(1.0),[1.0,2.0],0.0) == [1,2]  # finite level: zero threshold
@assert prox(NormL1(0.0,[Inf,0.0]),[2.0,3.0]) == [0,3]
@assert isnan(shrink(NaN,1.0))
@assert prox(NormL1(1.0,[Inf]),[2.0],1.0) == [0.0]  # finite input fixed at zero
@assert prox(NormL1(0.0,[Inf]),[2.0],0.0) == [0.0]  # zero level/step
@assert isnan(prox(NormL1(1.0,[Inf]),[Inf],1.0)[1])  # non-finite limitation (#853)
@assert isnan(prox(NormL1(1e300,[1e10]),[Inf],1.0)[1])  # final factor overflows (#866)
@assert isnan(prox(NormL1(1e10,[1.0]),[Inf],1e300)[1]) # intermediate product overflows (#866)
prox(NormL1(Inf),[NaN,Inf],1.0)
```

**Decision on `NormL1` non-finite inputs (#853 and #866).** A genuinely infinite
coordinate weight maps a *finite input value* to zero, even at zero level and step.
For inputs of `NaN` or `±Inf` with finite level, that fixed-zero guarantee is
intentionally not extended: retained soft-threshold arithmetic may return
`NaN`. Separately, a threshold made from *finite* factors can overflow to
`Inf`; with an input of `±Inf`, the current result can likewise be `NaN` even
though the mathematical threshold is finite. These examples characterize
present behavior, not additional non-finite-output guarantees. A genuinely
infinite **level** at any nonnegative step retains its distinct fixed-zero rule
for positive weights, including non-finite inputs; see [the zero-step rule](@ref infinite-penalty-at-zero-step).

## [Failure behavior](@id spec-norml1-failures)

The common owners are
[Predictable failures and partial writes](../numerical_usage.md#Predictable-failures-and-partial-writes),
[Validation, aliasing and evidence](../numerics.md#Validation,-aliasing-and-evidence),
and [NaN, infinities and zeros](../numerics.md#NaN,-infinities-and-zeros).
Constructor validation is described in the [`NormL1`](@ref) API entry.

## [Limitations and non-guarantees](@id spec-norml1-limitations)

A threshold formed from entirely finite factors may itself overflow to `Inf`.
For an input of `±Inf` this can also return `NaN`, even though the mathematical
threshold is finite. This is a **documented, deferred limitation**
([#866](https://github.com/mlakolar/ProximalBase.jl/issues/866)), not an
approved change to the proximal kernel. The decision preserves performance
of normal `Float32`/`Float64` weighted operations; the measured cost of extra
checks is recorded in the [numerical endpoint history](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-endpoint-history-921.md).
See also [Array and precision conventions](../numerical_usage.md).

## [Clauses owned elsewhere](@id spec-norml1-elsewhere)

| Subject | Owning section |
|:--|:--|
| Supported precision and agreement between oracle routes | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| General range bounds and limits of recovery | [Range guarantees](@ref numerical-value-range) |
| Infinite penalties at a zero step | [Infinite penalties at zero step](@ref infinite-penalty-at-zero-step) |
| Sparse products and signed-zero storage behavior | [Numerical contract](../numerics.md) and [Sparse product semantics](@ref Sparse-product-semantics) |
| Point-aware KKT scores and their endpoint/type rules | [NaN, infinities and zeros](../numerics.md#NaN,-infinities-and-zeros) |
| Ownership, aliases and output writes | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Symmetric parameter preconditions | [Symmetric parameters](@ref symmetric-parameter-guide) |
| `Scaled` wrappers and their limits | [Scaling a function](@ref scaling-guide) |

These shared statements remain at their current owners during the pilot.

## [Evidence](@id spec-norml1-evidence)

The files below were inspected at `7ae86e2093060eb124211e2f21f043c6379f97ae`.
They identify tested domains and reference methods, not new numerical guarantees.

| Test file | Reference or coverage relevant to this page |
|:--|:--|
| [`test/core_functions.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/7ae86e2093060eb124211e2f21f043c6379f97ae/test/core_functions.jl) | Entrywise soft-threshold formulas, array steps, exact input/output identity and fused values at stored outputs |
| [`test/minimal_math.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/7ae86e2093060eb124211e2f21f043c6379f97ae/test/minimal_math.jl) | Direct soft-threshold formulas and dense representations of sparse, block and symmetric iterate calculations |
| [`test/infinite_weights.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/7ae86e2093060eb124211e2f21f043c6379f97ae/test/infinite_weights.jl) | Independently classified endpoint cases and a `BigFloat` threshold reference; guaranteed, characterized and deferred cases are distinguished in the test |
| [`test/numerical_recovery.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/7ae86e2093060eb124211e2f21f043c6379f97ae/test/numerical_recovery.jl) | Direct `BigFloat` weighted absolute-value sums and soft-threshold references, including subnormal products and stored-output values |
| [`test/kkt_point.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/7ae86e2093060eb124211e2f21f043c6379f97ae/test/kkt_point.jl) | Point-aware scores from their mathematical definitions in `BigFloat`, plus explicit non-finite rules |
| [`test/one_based_axes.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/7ae86e2093060eb124211e2f21f043c6379f97ae/test/one_based_axes.jl) | Unsupported axes and public-boundary validation |
| [`test/symmetric_parameters.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/7ae86e2093060eb124211e2f21f043c6379f97ae/test/symmetric_parameters.jl) | Explicit symmetry checks, parameter shapes and constructor validation |

The [endpoint example](@ref spec-norml1-endpoints) is executed by the strict
documentation build.

## [History](@id spec-norml1-history)

The [numerical endpoint history](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-endpoint-history-921.md)
records prior decisions and measurements. The migration is tracked in
[#1019](https://github.com/mlakolar/ProximalBase.jl/issues/1019).
Historical records do not expand the current supported behavior.
