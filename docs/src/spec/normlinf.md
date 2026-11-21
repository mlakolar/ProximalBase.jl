# [NormLinf numerical contract](@id spec-normlinf)

This page gathers the existing numerical rules for `NormLinf`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-normlinf-identity)

**Links only until clause conversion.** Constructors and API help: [`NormLinf`](@ref).

Source: [`src/functions/normlinf.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/normlinf.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-normlinf-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [the function guide](../group_penalty.md#The-NormLinf-norm).

## [Traits](@id spec-normlinf-traits)

**Links only until clause conversion.** Trait declarations are described by [`NormLinf`](@ref) and [the function guide](../group_penalty.md#The-NormLinf-norm).

## [Supported arguments](@id spec-normlinf-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`NormLinf`](@ref) and [the function guide](../group_penalty.md#The-NormLinf-norm).

## [Preconditions](@id spec-normlinf-preconditions)

**Links only until clause conversion.** Use the [shape and mutation rules](../numerical_usage.md#Shapes-and-destination-first-mutation) and [supported precision](../numerics.md#Supported-precision-and-oracle-agreement), together with the family conditions linked in [Supported arguments](@ref spec-normlinf-arguments).

## [Results](@id spec-normlinf-results)

**Links only until clause conversion.** Return values are described by [`NormLinf`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-normlinf-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../group_penalty.md#The-NormLinf-norm).

## [Range clauses](@id spec-normlinf-range)

The [ElasticNet range clauses](@ref spec-elasticnet-range) explain the Julia 1.10 subnormal-scaling allowance cited here.

The clip level `τ` of `NormLinf` has an **absolute** error. On the ordinary finite path, for `n ≥ 1` entries in the
floating type `T` of the output with `n eps(T) ≤ 1/2` (up to `2^22` entries in `Float32` and `2^51` in `Float64`), a
hypothesis that the code does not check, the level returned by the threshold satisfies
`|τ - τ*| ≤ 2.25 n eps(T) max|xᵢ| + 2nδ`, where `τ*` is the root of `Σ(|xᵢ| - τ)₊ = a` for the entries
`abs(convert(T, xᵢ))` and the radius `a` that the prox forms (the product `γλ` rounded in the promoted type of `γ`, `λ`
and `T`, then to `T`), and `δ = nextfloat(zero(T))` (the storage of a subnormal radius or output adds `δ`; `x = [δ]`,
`λ = δ`, `γ = 0.5` gives `[δ]` where the exact point is `δ/2`). The prox clips at the larger of the level and zero,
which is within the same bound of `max(τ*, 0)`. The derivation, in `src/functions/normlinf.jl`, bounds the excess
`Σ(|xᵢ| - τ)₊ - a` for each of the four ways the threshold search ends, and it does not assume that the entries it found
above the level are the entries above the exact level. Unlike the relative bounds of the `NormL2` and `GroupNormL2`
kernels, the bound is absolute: an entry far below `max|xᵢ|` is not clipped to a relative accuracy. A sum of magnitudes
that overflows or exceeds `floatmax(T)/2` is processed in power-of-two units: the statement above then applies first to
the scaled entries and the scaled radius, as they are rounded, and converting back carries the scaling allowances
described here (the radius `γλ` rounded to `T` can itself be infinite there, as for `x = [M, M]`, `λ = 1.5`, `γ = M`
with `M = floatmax(T)`, which is not the reference of the bound). The subnormal losses of the scaling are below the
nominal scale `n eps(T) max|xᵢ|` under the same hypothesis on `n`, and the scaling of an entry or of the radius changes
it by at most one subnormal spacing (the allowance covers the `ldexp` of Julia 1.10 noted above), a second hypothesis of
that argument that the code does not check.
Through `Scaled`, the wrapper's recovery of a radius `γλα` beyond the range has a limit: for
`Scaled(NormLinf(2.0 * 2.0^-1000), 2.0^1000)` at the step `0.75 floatmax(Float64)` on two entries equal to
`floatmax(Float64)`, even the folded level at half the scale is out of range, the wrapper passes an infinite step, and the
output is `+0` where the exact result is about `floatmax/4`. For positive
finite level and step, an infinite entry without `NaN` gives the identity; on the finite clipping branch the value is
`λ τ`, computed from the largest stored magnitude `τ`, and `value_and_prox!` returns the value of the stored output,
computed as `f(y)` computes it. Endpoint precedence is given in the table below.

The error of ``\tau`` is absolute: on the ordinary
finite path, at most ``2.25\,n\,\varepsilon\max_i\lvert x_i\rvert+2n\delta`` for ``n\ge1`` entries with
``n\varepsilon\le 1/2``, relative to the radius ``a`` that the proximal operator forms from ``\gamma`` and ``\lambda``
and the entries read in the output type, with ``\varepsilon`` the machine epsilon and ``\delta`` the smallest subnormal
number of the output type (the storage of a subnormal radius or output adds ``\delta``). The [numerical
contract](../numerics.md#numerical-value-range) states the hypotheses and the scaled units of the exceptional path.

## [Endpoint clauses](@id spec-normlinf-endpoints)

| Case | Retained behavior |
|:--|:--|
| `NormLinf`, negative or `NaN` step | Checked first: `ArgumentError` before any write, ahead of the rows below, also through `Scaled` (a size mismatch is a `DimensionMismatch`); no proximal operator for a `SymmetricSparseIterate` (`ArgumentError` before any write, also through `Scaled`) |
| `NormLinf`, infinite level, nonnegative step | `+0` at every entry, `γ = 0` and `γ = Inf` included (#665 reading, #946), also for `NaN`/`±Inf` inputs; value `0` at the zero vector (an empty one included), `Inf` at any other argument without a `NaN`, `NaN` at a `NaN` entry |
| `NormLinf`, finite level, zero level or zero step | After the rows above: the identity (`-0`, `NaN` and `±Inf` entries unchanged), also at `γ = Inf` with a zero level; value `0` at a zero level, `λ maxᵢ abs(xᵢ)` at a zero step of a positive level |
| `NormLinf`, positive finite level, infinite step | `+0` for every input (decided before the entries are classified), `NaN` and `±Inf` entries included |
| `NormLinf`, positive finite level and positive finite step | A `NaN` entry makes every output `NaN`; a `±Inf` entry (no `NaN`) leaves the argument unchanged, also when the finite product `γλ` overflows; otherwise the clip at `τ` with `Σ(abs(xᵢ) - τ)₊ = γλ`, and `+0` when `γλ` is at least `‖x‖₁` or overflows while `‖x‖₁` is finite; a sum of magnitudes above `floatmax(T)/2` is processed in power-of-two units |

This passage concerns [full-point optimality scores](../group_penalty.md#norm-linf-kkt).

For finite entries, a zero level gives ``\lVert v\rVert_2``, and an infinite level gives `0` at the zero point and
`Inf` at another point. A `NaN` entry of `v` or `x` gives `NaN`, also beside an infinite entry;
otherwise an infinite entry gives `Inf`, also at a zero or an infinite level. The working type is
`float(promote_type(eltype(v), eltype(x)))`. The rules for the levels, the non-finite entries, the
ranges and the accuracy are in the [numerical contract](../numerics.md).

## [Failure behavior](@id spec-normlinf-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-normlinf-arguments), [Preconditions](@ref spec-normlinf-preconditions) and [Endpoint clauses](@ref spec-normlinf-endpoints), together with the failure behavior in [`NormLinf`](@ref).

## [Limitations and non-guarantees](@id spec-normlinf-limitations)

**Links only until clause conversion.** Shared limits on range recovery and arithmetic-order agreement remain in [Range guarantees](@ref numerical-value-range).

## [Clauses owned elsewhere](@id spec-normlinf-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-normlinf-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/normlinf.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/normlinf.jl)
- [`test/kkt_linf.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/kkt_linf.jl)

## [History](@id spec-normlinf-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
