# [GroupNormLinf numerical contract](@id spec-groupnormlinf)

This page gathers the existing numerical rules for `GroupNormLinf`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-groupnormlinf-identity)

**Links only until clause conversion.** Constructors and API help: [`GroupNormLinf`](@ref).

Source: [`src/functions/groupnormlinf.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/groupnormlinf.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-groupnormlinf-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [the function guide](../group_penalty.md#group-linf-guide).

## [Traits](@id spec-groupnormlinf-traits)

**Links only until clause conversion.** Trait declarations are described by [`GroupNormLinf`](@ref) and [the function guide](../group_penalty.md#group-linf-guide).

## [Supported arguments](@id spec-groupnormlinf-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`GroupNormLinf`](@ref) and [the function guide](../group_penalty.md#group-linf-guide).

## [Preconditions](@id spec-groupnormlinf-preconditions)

A `GroupNormLinf` object keeps a buffer for the threshold, with the length of its longest group, which the first group
that needs one allocates. The buffer is left untouched by the parameter exits, by `NaN` and infinite groups, by a zero
fill decided before the threshold (the computed radius reaches the computed sum of magnitudes; a threshold whose level is zero has
already used the buffer) and, in `prox_block!`, when `out` is an `Array` with a floating-point element type, `u` is an
`Array` or a `SubArray` of an `Array`, and `!Base.mightalias(out, u)` (so not when `out === u`): `out` is then the
buffer. The same object must not be used from several tasks at once while its buffer may be in use (`Scaled` wrappers of
one object share it, and so do the groups of a threaded loop of `prox_block!(u, g, u, k, γ)`), and it must not be used
reentrantly, also within one task: a nested call on the same object, from a `getindex` of an argument while the threshold
runs, can overwrite the candidates of the running threshold if it reaches the threshold with the shared buffer. Construct a fresh `GroupNormLinf` for each task, or synchronize
the calls ([ownership](../numerical_usage.md#Ownership-and-aliases)).

## [Results](@id spec-groupnormlinf-results)

**Links only until clause conversion.** Return values are described by [`GroupNormLinf`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-groupnormlinf-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../group_penalty.md#group-linf-guide).

## [Range clauses](@id spec-groupnormlinf-range)

The [shared range guarantees](@ref numerical-value-range) describe the range check referenced here.
The [endpoint table](@ref spec-groupnormlinf-endpoints) and the [ElasticNet range clauses](@ref spec-elasticnet-range) supply the endpoint and Julia 1.10 scaling context.

`GroupNormLinf` values apply the same check to their group terms `λ0*w[k]*max|x[G_k]|`, each formed in floating point
as `(λ0*w[k])*m` with two rounded products when `λ0*w[k]` is a normal number (and by the three-factor kernel otherwise)
and then converted to the output type, and may recompute from the exponents of `w[k]*max|x[G_k]|` with `λ0` applied once.

For `GroupNormLinf`, let `T` be the common type of the input, output, level, weights and step (`Float32` or `Float64`).
For finite data and positive finite parameters in round-to-nearest arithmetic with gradual underflow, each stored entry
is within `8 n eps(T) M + (2n + 1)δ` of the exact proximal entry for `r = γλ0w[k]`. Here `n ≥ 1` is the size of the
group, `M` its largest magnitude and `δ = nextfloat(zero(T))`. The bound assumes `n eps(T) ≤ 1/2`, which the code does
not check. This applies to the direct array and compact block calls. The parameter precedence for zero and infinite
factors is given in the table below.

The exact nonnegative level at radius `r` is the maximum of zero and the levels
`(sum(abs(x[i]) for i in A) - r) / length(A)` over nonempty subsets `A` of the group. Thus it changes by at most the
largest entry perturbation and the absolute radius perturbation. A relative perturbation of the radius by at most `η`
changes the level by at most `η M / (1 - η)`: at the smaller radius a subset with a positive level has radius per entry
at most `M`. The radius uses two products in the working type, or two products of significands followed by exponent
scaling if the normal-range products cannot be used. With `u = eps(T)/2`, the two products contribute at most `4u M`
to the level error and subnormal radius rounding at most `δ`, including the Julia 1.10 `ldexp` allowance above.

If the computed radius is below the computed sum, the shared threshold contributes at most `4.5 n u M + 2nδ` before
accounting for the radius error. If the computed radius reaches the computed sum, that sum is at least
`(1 - (n - 1)u)` times the exact sum. The subset formula then bounds the level for the computed radius by `(n - 1)u M`.
If the rounded radius overflows in these units, the exact radius exceeds `floatmax(T)/(1 + u)^2`. The exact sum is
at most `2 floatmax(T)/3` when its computed value is at most `floatmax(T)/2`, so zero is then the exact result.

If the first sum exceeds `F/2`, where `F = floatmax(T)`, the group is read in units of `2^-k` with
`k = ceil(log2(n)) + 2`. Under the size hypothesis the rounded scaled sum is at most `(F/4 + nδ)/(1 - n u) < F/2`,
so the first scaling suffices. The scaled entries incur at most `δ` of perturbation in those units. Applying the
unscaled bound there and multiplying back gives `(4.5n + 4)u M` plus at most
`2^k [(2n + 2) + (4.5n + 4)u]δ`. Since `2^k < 8n`, the latter term is at most `48 n^2 δ`.
The scaling trigger requires `M > F/(3n)`, and `144 n^2 δ ≤ u F` for `Float32` and `Float64` under the size
hypothesis, so this term is at most `n u M`. Scaling the level back is exact where finite. Capping it at `M`, including
when it overflows, can only decrease the error. Clipping selects an entry or the signed level, both already in `T`,
so same-type storage adds no rounding error. With a different input or output type, entries are read in the floating
type of the destination and the clipped values are converted to its element type on write.

The level comes from the same threshold as
`NormLinf`, applied to the entries of one group, read through its indices without a copy. With finite data and positive
finite parameters of the same type `Float32` or `Float64`, and input and output of that type, the absolute error in
each stored entry is at most ``8n\varepsilon\max_{i\in G_k}\lvert x_i\rvert+(2n+1)\delta`` under the assumptions in the
[numerical contract](../numerics.md#groupnormlinf-clip-accuracy). Here ``n`` is the group size, ``\varepsilon`` is `eps(T)` and
``\delta`` is the smallest positive subnormal of `T`. The radius is formed as
``(\gamma\lambda_0)w_k`` in floating point and then converted to the output
type. When ``\gamma\lambda_0`` is not a normal number, or the product leaves the normal range of the output type, it is
formed from the significands and exponents of the three factors, so that an intermediate overflow or underflow does not
decide it. The accuracy bound includes the case where the computed radius reaches the computed sum and the entire group becomes `+0`.

## [Endpoint clauses](@id spec-groupnormlinf-endpoints)

| Case | Retained behavior |
|:--|:--|
| `GroupNormLinf`, zero weight | Unpenalized group: returned unchanged (`NaN`/`±Inf` included), also at an infinite level or step; zero value |
| `GroupNormLinf`, infinite weight, or infinite level with a positive weight | Group fixed at `+0` at every nonnegative step, `γ = 0` and `γ = Inf` included, also for `NaN`/`±Inf` inputs, and also for an infinite weight with a zero level; value `0` at a zero group, `Inf` at a nonzero group without `NaN`, `NaN` at a `NaN` entry |
| `GroupNormLinf`, zero level or zero step, finite positive weight | After the rows above: the identity (`-0`, `NaN` and `±Inf` entries unchanged), also when `λ0*w[k]` overflows and at `γ = Inf` with a zero level; value `0` at a zero level (`NaN` entries included), `λ0 w[k] max abs(x_G)` at a zero step of a positive level |
| `GroupNormLinf`, positive finite level and weight, infinite step | `+0` for every input, `NaN` and `±Inf` entries included |
| `GroupNormLinf`, positive finite level, weight and step | A `NaN` entry makes the group `NaN` and leaves the other groups to their own rules; a `±Inf` entry (no `NaN`) leaves the group unchanged, also when the finite product `γ*λ0*w[k]` overflows; otherwise the clip of the group at `τ` with `Σ(abs(xᵢ) - τ)₊ = γ λ0 w[k]`, and `+0` when the computed radius reaches the computed sum of magnitudes or overflows while that sum is finite |
| `GroupNormLinf`, negative or `NaN` step | `ArgumentError` before any write, in `prox!`, `value_and_prox!` and `prox_block!`; no proximal operator for a `SymmetricSparseIterate` (`ArgumentError` before any write) |

```@setup group_penalty
using ProximalBase, LinearAlgebra
```

The rules apply to each group in this order; ``\lambda_0``, ``w_k`` and ``\gamma`` are decided separately, before a radius is formed,
so that the proximal operator and the value do not form a product such as ``\lambda_0w_k`` with ``\lambda_0=\infty`` and ``w_k=0``.

| Case | Result for group ``k`` |
|:--|:--|
| ``w_k=0`` | Unpenalized: returned unchanged, `NaN` and `±Inf` included, also at an infinite level or step; value zero |
| ``w_k=\infty``, or ``\lambda_0=\infty`` with ``w_k>0`` | Fixed at `+0` at every nonnegative step, ``\gamma=0`` and ``\gamma=\infty`` included, for `NaN` and `±Inf` inputs too; ``w_k=\infty`` with ``\lambda_0=0`` is included. Value `0` at a zero group, `Inf` at a nonzero group without `NaN`, `NaN` at a `NaN` entry |
| ``\lambda_0=0`` or ``\gamma=0``, finite positive ``w_k`` | The identity, also when ``\lambda_0w_k`` overflows and at ``\gamma=\infty`` with a zero level. Value `0` at a zero level, ``\lambda_0w_k\max\lvert x_{G_k}\rvert`` at a zero step |
| ``\gamma=\infty``, positive finite level and weight | Fixed at `+0` |
| A `NaN` entry, positive finite level, weight and step | Every entry of the group becomes `NaN`; the other groups are unaffected |
| A `±Inf` entry and no `NaN`, positive finite level, weight and step | Returned unchanged, also when the finite product ``\gamma\lambda_0w_k`` overflows |
| Finite entries, the computed radius reaches the computed sum of magnitudes | `+0`, also when the product ``\gamma\lambda_0w_k`` overflows while the sum is finite |
| ``\gamma<0`` or ``\gamma=`` `NaN` | `ArgumentError` before any write |

The value of a group is ``\lambda_0w_k\max\lvert x_{G_k}\rvert``, formed in floating point as
``(\lambda_0w_k)m`` with two rounded products when ``\lambda_0w_k`` is a normal number and by the three-factor
kernel in power-of-two units otherwise, and then converted to the output type; the sum over the groups may be recomputed in power-of-two units, accumulated in at least `Float64`, where it may
have lost accuracy (a sum below `K*floatmin(T)` for `K` groups, whose terms may have rounded into the subnormal range) or
class (at or above `floatmax(T)/2`), as for `GroupNormL2` ([value range limits](@ref numerical-value-range)). A sum of
magnitudes of one group above `floatmax(T)/2` is processed in power-of-two units. With a positive finite level and
weight, `prox!` computes the clip of a group with finite entries whose sum overflows from the entries scaled by a power of two:

```@example group_penalty
M = floatmax(Float64)
huge = GroupNormLinf(M, [1:2], [2.0])        # the radius γλ₀w = M/2 at γ = 0.25 is finite; λ₀w alone overflows
yh = prox(huge, [M, M], 0.25)
@assert yh[1] == yh[2] && isapprox(yh[1], 0.75M; rtol = 1e-14)
tiny = GroupNormLinf(2.0^-100, [1:2], [2.0^1000])   # γλ₀ underflows, γλ₀w = 2^-100 equals ‖x‖₁
@assert prox(tiny, [2.0^-100, 0.0], 2.0^-1000) == [0.0, 0.0]
yh
```

## [Failure behavior](@id spec-groupnormlinf-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-groupnormlinf-arguments), [Preconditions](@ref spec-groupnormlinf-preconditions) and [Endpoint clauses](@ref spec-groupnormlinf-endpoints), together with the failure behavior in [`GroupNormLinf`](@ref).

## [Limitations and non-guarantees](@id spec-groupnormlinf-limitations)

The [group optimality scores](../group_penalty.md#Optimality-scores-2) explain the scaling rules referenced here.

`GroupNormLinf` has no `SymmetricSparseIterate`, `restrict`, per-group step or compact block method under `Scaled`.
`Scaled(GroupNormLinf(...), α)` multiplies the value by `α` and computes the array proximal operator with the step
`αγ`; it has no folded level for the proximal operator, so a step `αγ` that overflows reaches the function as `Inf` (the scores of
`kkt_violation` fold the level of a chain, as above). A group that reaches the
infinite-step exit then gives `+0`, where the exact radius ``\alpha\gamma\lambda_0w_k`` can be below the ℓ₁ norm of the
group; a group with a zero weight or a zero level with a finite weight keeps the identity, because those exits come first.
### [Zero factor in Scaled](@id group-norm-linf.scaled.zero-factor)

A zero `α` throws an `ArgumentError`.

### Scaled step example

```@example group_penalty
scaled = Scaled(GroupNormLinf(2.0^-1020, [1:2]), 2.0^1000)
@assert prox(scaled, [2000.0, 500.0], 2.0^30) == [0.0, 0.0]           # the step 2^1030 overflows
@assert prox(GroupNormLinf(2.0^-20, [1:2]), [2000.0, 500.0], 2.0^30) == [976.0, 500.0]   # the radius 2^10 = 1024 keeps the group
prox(scaled, [2000.0, 500.0], 2.0^30)
```

## [Clauses owned elsewhere](@id spec-groupnormlinf-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |
| Group optimality scores | [Point-aware scores](@ref spec-kkt_violation-endpoints) |

## [Evidence](@id spec-groupnormlinf-evidence)

| Clause ID | Status | Evidence |
|:--|:--|:--|
| `group-norm-linf.scaled.zero-factor` | `guaranteed` | [Scaled constructor](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/scaled.jl#L111-L118), [Zero-scale error](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/scaled.jl#L133-L135), and [GroupNormLinf zero-factor rejection test](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/groupnormlinf.jl#L1638-L1644) |

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/groupnormlinf.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/groupnormlinf.jl)

## [History](@id spec-groupnormlinf-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
