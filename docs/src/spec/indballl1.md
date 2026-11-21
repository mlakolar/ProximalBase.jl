# [IndBallL1 numerical contract](@id spec-indballl1)

## [Identity](@id spec-indballl1-identity)

[`IndBallL1`](@ref IndBallL1) is the ℓ₁-ball indicator. Its constructor and
oracles are implemented in [`src/functions/indballl1.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/indballl1.jl).

## [Definition and domain](@id spec-indballl1-definition)

`IndBallL1(r)` represents the set `sum(abs, x) ≤ r` for a nonnegative radius
`r`, including `Inf`. Matrices use the magnitudes of all entries. The value
is zero at a feasible point and `Inf` outside, with the non-finite rules
below. The default radius is one.

## [Traits](@id spec-indballl1-traits)

The function is convex and proximable. Its projection couples the entries
through a common threshold.

## [Supported arguments](@id spec-indballl1-arguments)

Value and scalar-step projection support real, one-based dense arrays and
views, `SparseIterate` and `BlockIterate`. The tested numerical target is
homogeneous `Float32` and `Float64`. Value visits the stored entries of a
`SparseIterate`. Sparse-to-sparse projection visits stored entries except for
an all-NaN fill, which covers the logical array. A `BlockIterate` uses its full
vector. `SymmetricSparseIterate` inputs and destinations are unsupported.

## [Preconditions](@id spec-indballl1-preconditions)

A mutating projection requires a destination of the same size as the input.
Exact input/output identity is supported. Distinct overlapping views are
outside the supported aliasing contract. Use a separate function object in
each task whose calls may need the object's scratch buffer.

## [Results](@id spec-indballl1-results)

`prox` returns an allocated projection. `prox!` writes the projection and
returns the destination. The projection is independent of the nonnegative
scalar step.

`value_and_prox!` evaluates the stored point. Its value can be `Inf` when
rounding puts that point outside the computed boundary. For example, with
`δ = nextfloat(0.0)`, radius `3δ` projects `[M, -M]` to `[2δ, -2δ]`.

## [Ordinary-path clauses](@id spec-indballl1-ordinary)

For `IndBallL1`, the value and projection compare the computed sum of
magnitudes with the radius rounded to the floating type `T` of the argument.
There is no feasibility tolerance. Rounding can decide a point at the
boundary. For example, the computed sum for `[1.0, 2.0^-54]` equals `1.0`,
and the projection at radius `1.0` copies that point.

A feasible input is copied with its signed zeros. For a finite point outside
a ball of positive finite radius, the projection soft-thresholds all
magnitudes at a common level. Entries at or below that level become `+0`.

## [Range clauses](@id spec-indballl1-range)

**Implementation notes (not guarantees).**
Finite sums above `floatmax(T)/2` are compared in power-of-two units. The
shared ℓ₁-ball threshold from `NormLinf` selects the soft-threshold level.
A tiny radius can be lost when that level rounds to the largest magnitude.
The fallback sorts the magnitudes and forms the output directly from their
gaps at `O(n log n)` cost. The exceptional path is selected when the
radius is at most `64 eps(T)` times the computed sum in the working units
or when a level search is needed at a subnormal radius. A radius tiny
relative to the largest magnitude goes directly to the fallback. The
other cases on this path check the tentative projected mass before writing.
On the sorted path for a subnormal radius, the gaps and remaining radius
are combined in scaled units and rounded back when storing the output.

With `M = floatmax(Float64)` and radius `1`, the projection of `[M, -M]`
is `[0.5, -0.5]`.

For an exact normal threshold, rounding the threshold alone changes the
projected mass by at most `u S/(1-u)`, where `u = eps(T)/2` and `S` is the
exact sum of the magnitudes in the working units. The displacement is at
most `u` times the exact threshold per affected coordinate, and each
affected magnitude exceeds the smaller of the two thresholds.
Outside the mass screen, this contribution relative to the radius is bounded
by `(S/s)/(128(1-u))` for the computed sum `s`. The bound isolates threshold
rounding. The selector's absolute error, sum rounding, and subnormal
rounding retain their own effects.

## [Endpoint clauses](@id spec-indballl1-endpoints)

The projection accepts scalar steps from zero through `Inf`. A zero radius
projects to positive zeros for every input. An infinite radius copies the
input. The value is zero at infinite radius for inputs without NaN.

At a positive finite radius, a single infinite entry with no NaN projects
to the signed radius there and to positive zeros elsewhere. Several
infinite entries or any NaN give an all-NaN projection. An input NaN gives
value `NaN` at any radius. An empty input has value zero and is copied.

## [Failure behavior](@id spec-indballl1-failures)

### [Invalid radius at construction](@id ind-ball-l1.constructor.invalid-radius)

A negative or NaN radius throws `ArgumentError` at construction.

### Invalid steps and arguments

A negative
or NaN scalar step throws `ArgumentError` before writes, also for `α * f` with `α ≥ 0`,
which is `f` itself. Invalid axes, shapes and argument types are checked before
writes as described in [array and precision conventions](@ref numerical-usage).

## [Limitations and non-guarantees](@id spec-indballl1-limitations)

Feasibility near the boundary depends on floating-point summation and
rounding. The threshold bound above describes the mass error from rounding
an exact normal threshold in the working type. The shared selector's
absolute error also applies.

## [Clauses owned elsewhere](@id spec-indballl1-elsewhere)

This section links to the shared owners.

- [Array and precision conventions](@ref numerical-usage)
- [Shared range guarantees](@ref numerical-value-range)
- [The ℓ∞ norm and its threshold](../group_penalty.md#The-NormLinf-norm)
- [Scaling a function](@ref scaling-guide)

## [Evidence](@id spec-indballl1-evidence)

[`test/indballl1.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/indballl1.jl)
checks the projection against an independent BigFloat reference in `Float32`
and `Float64`. It covers feasible boundaries, overflow, tiny radii, stored
results, storage routes and validation before writes. The
[`IndBallL1` API example](@ref IndBallL1) is executed by the strict
documentation build.

| Clause ID | Status | Evidence |
|:--|:--|:--|
| `ind-ball-l1.constructor.invalid-radius` | `guaranteed` | [Radius constructor](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/indballl1.jl#L49-L55), [nonnegative real check](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/utils.jl#L42), and [invalid-radius tests](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/indballl1.jl#L108-L112) |

## [History](@id spec-indballl1-history)

The [restoration issue #955](https://github.com/mlakolar/ProximalBase.jl/issues/955)
records the numerical decisions and test plan.
