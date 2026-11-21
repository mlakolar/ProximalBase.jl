# [ElasticNet numerical contract](@id spec-elasticnet)

This page gathers the existing `ElasticNet` numerical text without changing its
wording. The mathematical introduction and ordinary examples remain in the
[elastic-net guide](../penalties.md#The-elastic-net-penalty).
Clause conversion and status assignments remain a separate review step;
prose without an explicit status remains unclassified.

Read [How to read the numerical contracts](index.md) for authority, status
meanings and the current migration map.

## [Identity](@id spec-elasticnet-identity)

Links only: constructors and API help are in [`ElasticNet`](@ref), and the
[`ElasticNet` row in the API coverage matrix](../api/coverage.md) lists the
retained interface.

Source: [`src/functions/elasticnet.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/elasticnet.jl).
The [evidence section](@ref spec-elasticnet-evidence) links the retained test files.

## [Definition and domain](@id spec-elasticnet-definition)

Links only: the definition, levels, weights and constructor examples remain in
[The elastic-net penalty](../penalties.md#The-elastic-net-penalty).

## [Traits](@id spec-elasticnet-traits)

Links only: the family description remains in
[The elastic-net penalty](../penalties.md#The-elastic-net-penalty) and the
[`ElasticNet`](@ref) API entry. The trait interface is described in
[Function objects and their properties](../function_objects.md).

## [Supported arguments](@id spec-elasticnet-arguments)

The retained parenthetical “above” refers to the two-part value recomputation
now described in [Range clauses](@ref spec-elasticnet-range).

The array `w` has one weight per entry of the argument, paired by Julia's
linear index, and is stored by reference. `ElasticNet` accepts real dense
vectors and matrices (views included), `SparseIterate`, `BlockIterate` and
`SymmetricSparseIterate`, with a scalar step or, except as noted below, an
array of per-entry steps. On a `SparseIterate` or a
`SymmetricSparseIterate` the value reads the stored entries only, and so
does the proximal operator from such an iterate into an iterate of the same
kind; with a dense destination it writes, and reads the weights of, every
entry. This is the traversal of the entries: a value whose two parts are
recomputed together (above) also reads every entry of a weight array whose
element type is not concrete, such as a `Vector{Real}`, to choose the type of
its accumulator. An entry stored in an iterate destination stays stored, possibly as
a zero, and a zero result does not activate an absent entry. On a
`SymmetricSparseIterate` the weights
are a `p × p` matrix read at both positions of an off-diagonal
coordinate, and the value counts such a coordinate twice, as the matrix
it represents does. The weights must then be exactly symmetric, a
precondition that the oracles assume and `check_symmetric_parameters`
validates (see [Symmetric parameters](../symmetric_parameters.md)).

Not supported: an array of steps with a `SymmetricSparseIterate`
destination (an `ArgumentError` before any write), or with a dense
destination and a `SymmetricSparseIterate` argument (a `MethodError`);
the vector form of `prox_block`; complex arguments or weights.

## [Preconditions](@id spec-elasticnet-preconditions)

Links only: constructor conditions are in [`ElasticNet`](@ref), parameter shapes
and symmetric storage conditions are in [Supported arguments](@ref spec-elasticnet-arguments),
and step conditions are in [Endpoint clauses](@ref spec-elasticnet-endpoints).
Shared conditions remain in
[Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation)
and [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement).

## [Results](@id spec-elasticnet-results)

Links only: the formulas and agreement between ordinary routes remain in
[The elastic-net penalty](../penalties.md#The-elastic-net-penalty).
Storage effects are retained in [Supported arguments](@ref spec-elasticnet-arguments).
Shared mutation and alias rules remain in
[Array and precision conventions](../numerical_usage.md).

## [Ordinary-path clauses](@id spec-elasticnet-ordinary)

Links only: ordinary mathematical formulas, examples and the KKT tutorial remain in
[The elastic-net penalty](../penalties.md#The-elastic-net-penalty).
This verbatim move assigns no clause identifiers or statuses.

## [Range clauses](@id spec-elasticnet-range)

In the retained text below, “the recovery above” refers to the shared
[Range guarantees](@ref numerical-value-range). Existing implementation details
remain alongside the stated bounds; this move does not assign them a clause status.

`ElasticNet` values are the value of the ℓ₁ part, computed as `NormL1`
computes it, plus the value of the ridge part, computed as `SqrNormL2`
computes it, each with the recovery above, then one addition; on sparse
and symmetric iterates both parts read the stored entries only, and on a
`BlockIterate` its storage. On a
dense `Float32` or `Float64` `Array` with parameters of its type, a single
pass over the entries is returned instead when a test on its sum bounds its
relative error by `(n + 2) eps(T)` for `n ≤ 1/(2 eps(T)) - 2` entries; the bound is proved in a
comment next to the code (`src/functions/elasticnet.jl`). With both levels
positive and finite, a sum of the two parts below `2 floatmin(T)` (each part
rounded on its own) is recomputed: the terms of both parts are added in one
exponent-sum accumulator, as the `NormL1` recomputation adds its terms, and
the sum is rounded once. For `N` nonzero terms (`λ1*w[i]*abs(x[i])` and
`λ2*x[i]^2/2`, up to two per entry and three per off-diagonal coordinate of a
symmetric iterate), with `u` half the epsilon of the accumulator (`Float64` for
`Float32` data) and `N u < 1`, the hypothesis of the summation bound used, the
result is within `(γ_3 + (γ_{N-1}^2 + N 2^-1070)(1 + γ_2)) V` of the exact
value `V` plus half a spacing of `T` at the result (`γ_m = m u/(1 - m u)`), as
proved next to the code, which is below the worst case of the separate parts. On Julia 1.10 `ldexp` returns `0`
for a `Float64` result in `[δ/2, δ)`, `δ` the smallest subnormal
([#896](https://github.com/mlakolar/ProximalBase.jl/issues/896)), and the final
term of the bound is a whole spacing;
`ElasticNet(2.0^-538, 1.0)([2.0^-537])` is `2.0^-1074`. Only a value whose
stored entries are all zero keeps its sum. Under `Scaled`, the value of an
`ElasticNet` is recovered as for any function with levels. For a finite
positive factor, where that gives no finite value, or `0` at a point where the
function is not exactly `0`, the scaled values of the ℓ₁ part and of the ridge
part, each rounded on its own, are added; the sum replaces that result if it is
finite after conversion to its type, and the first result is kept otherwise.
Where one part
dominates, the test can pass although the other part alone would be
recomputed (ℓ₁ terms below `floatmin(T)` beside a normal ridge part, or
squares below `floatmin(T)` beside a normal ℓ₁ part). Its proximal point soft thresholds
by the threshold of the `NormL1` kernel and multiplies by the ridge factor
`1/(1 + γλ2)` of the `SqrNormL2` kernel, both formed from the unrounded
factors, so a factor `γλ2` outside the range still acts on a representable
result.

## [Endpoint clauses](@id spec-elasticnet-endpoints)

```@setup elasticnet_contract
using ProximalBase
h = ElasticNet(0.5, 2.0, [1.0, 0.0, 2.0])
x = [2.0, -3.0, 0.25]
```

### [Invalid proximal scalar steps](@id elastic-net.prox.invalid-scalar-step)

`prox`, `prox!` and `value_and_prox!` throw an `ArgumentError` for a
negative or `NaN` scalar step before anything is written.

### Other endpoint rules and examples

The entries of
an array of steps are not checked: they must be nonnegative, a documented
precondition. `prox_block` does not check its step, and `Scaled` keeps its own step rule.

The optimized array-step loop and the general loop store the same results,
bit for bit. See the [implementation notes](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/function-docs-history-921.md#elastic-net-step-traversal)
for branch-selection conditions and the reason array-step validation remains a
caller precondition.

```@example elasticnet_contract
y = [7.0, 7.0, 7.0]
refused = try prox!(y, h, x, -1.0); false catch e; e isa ArgumentError end
@assert refused && y == [7.0, 7.0, 7.0]
```

An infinite ridge level makes `g` the indicator of ``\{0\}``: at every
step ``\gamma\geq0``, ``\gamma=0`` included, each entry of the proximal
point is `+0`, also for `NaN` and `±Inf` inputs; the value is `0` at the
zero vector, `NaN` if an entry is `NaN`, and `Inf` otherwise; and
the screen `kkt_violation(g, v, i)` is `-Inf`, as is `kkt_violation(g, v, x, i)`
at a zero entry of `x` with a finite `v[i]`, while that form is `Inf` at a nonzero
entry of `x` with finite `v[i]`. An infinite `λ1` and infinite weights follow
the [ℓ₁ endpoint rules](@ref l1-endpoints), and a threshold that overflows from
finite factors keeps their documented limitation for infinite inputs.

```@example elasticnet_contract
hard = ElasticNet(1.0, Inf)
@assert prox(hard, [3.0, NaN], 0.0) == [0.0, 0.0]
@assert hard([0.0, 0.0]) == 0 && hard([0.0, 1e-300]) == Inf
@assert kkt_violation(hard, [5.0], 1) == -Inf
@assert kkt_violation(hard, [5.0], [0.0], 1) == -Inf
@assert kkt_violation(hard, [5.0], [1.0], 1) == Inf
```

The `NormL1` rows referenced below are retained in the
[NormL1 endpoint clauses](@ref spec-norml1-endpoints).

| Case | Retained behavior |
|:--|:--|
| `NormL1` | See the [NormL1 endpoint clauses](@ref spec-norml1-endpoints) |
| `ElasticNet`, negative or `NaN` scalar step | `prox`, `prox!` and `value_and_prox!` throw `ArgumentError` before any write. The entries of an array of steps are not checked: nonnegative entries are a caller precondition, a documented exception to [#908](https://github.com/mlakolar/ProximalBase.jl/issues/908)'s preflight rule. `prox_block` and `Scaled` routes do not check. The historical reasoning is in [numerical endpoint history](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-endpoint-history-921.md) |
| `ElasticNet`, infinite ridge level `λ2`, nonnegative step | The indicator of `{0}` (#908): every entry proxes to `+0`, including at `γ = 0` and for non-finite inputs; value `0` at the zero vector, `NaN` at a `NaN` entry, `Inf` otherwise; the screen `kkt_violation(f,v,i)` is `-Inf`, as is `kkt_violation(f,v,x,i)` at a zero entry of `x` with a finite `v[i]`. `SqrNormL2(Inf)` keeps its own rule |
| `ElasticNet`, infinite `λ1` or infinite weight, overflowing finite threshold | The `NormL1` rows above and [zero-step rule](@ref infinite-penalty-at-zero-step) |
| `ElasticNet`, zero `λ2` | Values of `NormL1(λ1, w)`, and at `γ ≥ 0` with one element type its proximal points, bit for bit; a zero `λ2` adds `0` also at `NaN`/`±Inf` entries |

```@setup elasticnet_numerical_endpoints
using ProximalBase
```

```@example elasticnet_numerical_endpoints
@assert prox(ElasticNet(1.0,Inf),[3.0,NaN,-Inf],0.0) == [0,0,0]  # infinite ridge level: {0}
@assert ElasticNet(1.0,Inf)([0.0,0.0]) == 0 && ElasticNet(1.0,Inf)([0.0,1e-300]) == Inf
@assert isnan(ElasticNet(1.0,Inf)([NaN,0.0]))
@assert prox(ElasticNet(Inf,1.0,[1.0,0.0]),[NaN,3.0],0.0) == [0,3]  # the NormL1 rule (#665)
@assert try prox(ElasticNet(1.0,1.0),[1.0],-1.0); false catch e; e isa ArgumentError end
prox(ElasticNet(1.0,Inf),[3.0,NaN,-Inf],0.0)
```

## [Failure behavior](@id spec-elasticnet-failures)

Links only: refused argument combinations are in
[Supported arguments](@ref spec-elasticnet-arguments), and scalar-step failures
are in [Endpoint clauses](@ref spec-elasticnet-endpoints).
Shared owners remain
[Predictable failures and partial writes](../numerical_usage.md#Predictable-failures-and-partial-writes)
and [Validation, aliasing and evidence](../numerics.md#Validation,-aliasing-and-evidence).

## [Limitations and non-guarantees](@id spec-elasticnet-limitations)

The shared comparison of `NormL1`, `NormL1Nonneg` and `ElasticNet` under `Scaled`
remains with its conditions and preceding example in the
[shared range guarantees](@ref numerical-value-range). The dedicated retained limitation follows.

A known limitation ([#567](https://github.com/mlakolar/ProximalBase.jl/issues/567)):
for `Scaled(ElasticNet(λ1, λ2, w), α)` with finite levels, a finite positive `α`
and a positive finite scalar step `γ` whose product `αγ` is above `floatmax`,
`prox`, `prox!`, `prox_block` and `value_and_prox!` can return `0` where the
exact result is representable. A representable result does not need a
representable coefficient. With `λ1 = 0`, `α = γ = 2.0^600` and `x = [2.0^1000]`,
the coefficient `αγλ2` is `2^1300` for `λ2 = 2.0^100`, above `floatmax`,
while the quotient `x/(1 + αγλ2)`, about `2^-300`, is representable, and the
result is `0`. For `λ2 = 2.0^-1000` and `x = [4.0]` the coefficient is `2^200`
and the result is the representable `4/(1 + 2^200)`.
`Scaled(SqrNormL2(λ2), α)` also returns `0` in the first case.
In some cases the same step passed as an array of steps gives the
representable result, for example in the first case and in its `Float32`
analogue (`λ2 = 2.0f0^80`, `α = γ = 2.0f0^80`, `x = [2.0f0^127]`); for
`λ2 = 2.0^880` and `x = [2.0^1023]`, with the same `α` and `γ`, both steps
give `0` where the result is `2^-1057`:

```julia
s = Scaled(ElasticNet(0.0, 2.0^100), 2.0^600)
prox(s, [2.0^1000], 2.0^600)      # [0.0]
prox(s, [2.0^1000], [2.0^600])    # [4.909093465297727e-91], about 2^-300
t = Scaled(ElasticNet(0.0, 2.0^880), 2.0^600)
prox(t, [2.0^1023], 2.0^600)      # [0.0]
prox(t, [2.0^1023], [2.0^600])    # [0.0]; the result is 2^-1057
```

## [Clauses owned elsewhere](@id spec-elasticnet-elsewhere)

This section contains links only; these shared statements keep their current owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and agreement between oracle routes | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| General range bounds and limits of recovery | [Range guarantees](@ref numerical-value-range) |
| Infinite penalties at zero step | [Infinite penalties at zero step](@ref infinite-penalty-at-zero-step) |
| Infinite ℓ₁ levels, infinite weights and finite-threshold limitations | [NormL1 endpoint clauses](@ref spec-norml1-endpoints) |
| Shared point-aware KKT endpoint and element-type rules | [NaN, infinities and zeros](../numerics.md#NaN,-infinities-and-zeros) |
| Sparse products and signed-zero storage behavior | [Sparse product semantics](@ref Sparse-product-semantics) |
| Ownership, aliases and output writes | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Symmetric parameter preconditions | [Symmetric parameters](@ref symmetric-parameter-guide) |
| `Scaled` wrapper exclusions and KKT rules | [Unsupported cases](../calculus.md#Unsupported-cases) and [Scoring optimality](@ref scaled-kkt) |

## [Evidence](@id spec-elasticnet-evidence)

Links only: the retained tests are
[`test/elasticnet.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/elasticnet.jl),
[`test/elasticnet_extremes.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/elasticnet_extremes.jl),
and [`test/kkt_point.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/kkt_point.jl).
Their existing reference-method summary remains in
[Validation, aliasing and evidence](../numerics.md#Validation,-aliasing-and-evidence).

The retained [endpoint examples](@ref spec-elasticnet-endpoints) are executable
documentation blocks. No numerical example or assertion body is changed by the move.

| Clause ID | Status | Evidence |
|:--|:--|:--|
| `elastic-net.prox.invalid-scalar-step` | `guaranteed` | [Allocating array dispatch](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/interface.jl#L139-L143), [Scalar-step checks](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/elasticnet.jl#L384-L390) and [Invalid-step and unchanged-destination tests](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/elasticnet.jl#L489-L517) |

## [History](@id spec-elasticnet-history)

Links only: [numerical endpoint history](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-endpoint-history-921.md)
and [function documentation history](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/function-docs-history-921.md#elastic-net-step-traversal)
retain prior rationale. The migration is tracked in
[#1019](https://github.com/mlakolar/ProximalBase.jl/issues/1019).
