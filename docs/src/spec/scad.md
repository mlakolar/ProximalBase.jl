# [SCAD numerical contract](@id spec-scad)

## Identity

[`SCAD`](@ref SCAD) is the separable smoothly clipped absolute deviation
penalty. Its scalar hooks are defined in
[`src/functions/scad.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/scad.jl).

## Definition and domain

`SCAD(λ, a=3.7)` takes a nonnegative real level and a real shape above 2.
Both parameters can be positive infinity. For a positive finite level and
finite shape, the penalty sums the following contribution at each magnitude
``t=|x_i|``.

```math
p(t) = \begin{cases}
\lambda t & 0 \le t \le \lambda, \\
\lambda t - (t-\lambda)^2 / (2(a-1)) & \lambda < t < a\lambda, \\
(a+1)\lambda^2/2 & t \ge a\lambda.
\end{cases}
```

## Traits

The family is separable, proximable and zero-preserving. Its type-level
convexity and smoothness traits are false. Restriction returns the same
function because the parameters are shared across entries.

## Supported arguments

Direct value and proximal oracles use the real, one-based dense array
and iterate routes of [`SeparableFunction`](@ref SeparableFunction).
The numerical targets are homogeneous CPU `Float32` and `Float64`.
Array steps pair with coordinates by linear index on dense arrays,
`SparseIterate` and `BlockIterate`. A `SymmetricSparseIterate` destination
requires a scalar step.

## Preconditions

Exact input/output identity and independent destinations are supported.
Sizes and axes follow the [array conventions](@ref numerical-usage).
Use nonnegative, non-NaN steps with `prox_block` and as the effective
steps passed through `Scaled`.

## Results

`prox` allocates the result and `prox!` returns its destination.
`value_and_prox!` evaluates the value at the stored result, including
destination rounding. Sparse value and same-kind sparse proximal
routes visit stored entries. The symmetric value counts each
off-diagonal entry twice. Block proximal calls visit the full array.

## Ordinary-path clauses

For a positive finite level and step, and a finite shape and input, the
scalar operator minimizes ``p(u) + (u-|v|)^2/(2\gamma)`` over
magnitudes ``u \ge 0``. The middle interval has positive, zero or
negative curvature as the step passes through ``a-1``. Its interior
minimum is eligible when the curvature is positive. In the other cases
its endpoints suffice, with the smaller endpoint representing a flat
interval. The proximal operator selects the smaller magnitude on an
exact tie. A positive magnitude takes the sign of the input.

## Range clauses

The comparisons for large steps use algebraic differences of the complete
objective in magnitude units. This avoids forming the squared distance
and the penalty plateau when either would leave the floating-point range.
Power-of-two scaling protects the comparison with a zero candidate when
the squared cutoff overflows, and the join comparison at tiny magnitudes.
For an overflowing upper-join residual, the middle root can be formed from
the soft-threshold point. The value sums contributions in the result type.

## [Endpoint clauses](@id scad-endpoints)

Apply the first matching proximal rule for a valid step.

| Level and step | Proximal entry |
|:--|:--|
| `λ = 0` | Identity, including at `γ = Inf` |
| `λ = Inf` | Positive zero, including at `γ = 0` |
| `0 < λ < Inf`, `γ = 0` | Identity |
| `0 < λ < Inf`, `γ = Inf` | Positive zero |
| `0 < λ < Inf`, `0 < γ < Inf` | Global minimizer for finite input. NaN and signed infinities pass through |

Identity routes preserve signed zero. A zero minimizer selected by
thresholding or projection is positive zero. At a zero level the value
is zero for every input. At a positive level a NaN gives a NaN value.
At an infinite level the value is zero at either signed zero and infinite
at every nonzero numeric input. At a positive finite level and finite shape,
the value at either infinity is the plateau, which can exceed the result
type's range.

At a positive finite level and infinite shape the value is the ℓ₁ value.
The proximal operator at a positive finite step is soft thresholding.
At a fixed magnitude the middle piece tends to the linear value as the
shape increases. As the level increases, each nonzero fixed magnitude
eventually lies on the first piece and its value diverges. The infinite-level
limit is therefore the indicator of zero.

## Failure behavior

The constructor rejects negative or NaN levels and shapes at or below 2
or equal to NaN with `ArgumentError`. Direct array oracles reject negative
or NaN steps before writes. They check every entry of an array step,
including inactive coordinates. Array steps on symmetric sparse outputs
are rejected before writes. Structural checks follow the array conventions.

## Limitations and non-guarantees

Finite comparisons near a change of minimizer can round. The homogeneous
targets describe the supported precision. Positive `Scaled` factors use the
wrapper's effective step and value rules. A zero factor throws at construction.
An infinite factor at a positive step can give a NaN value at zero output.
An infinite factor at zero step gives an effective step outside the supported
domain.

## Clauses owned elsewhere

This section links to the shared owners.

- [Array and precision conventions](@ref numerical-usage)
- [Shared range guarantees](@ref numerical-value-range)
- [Scaling a function](@ref scaling-guide)

## Evidence

[`test/scad.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/scad.jl)
checks both supported types against a BigFloat objective and a bounded
search. It covers switches and ties, finite range and endpoints, sparse
activity, aliases, pre-write validation and fused values. The
[penalty guide](../penalties.md#SCAD-penalty) includes an executable example.

## History

The [restoration issue #953](https://github.com/mlakolar/ProximalBase.jl/issues/953)
records the numerical decisions and test plan.
