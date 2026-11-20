# [MCP numerical contract](@id mcp-contract)

## Identity

[`MCP`](@ref MCP) is the separable minimax concave penalty. The scalar
hooks and dense kernels are in
[`src/functions/mcp.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/mcp.jl).

## Definition and domain

`MCP(λ, a=3)` takes a nonnegative real level and a strictly positive real
shape. Both admit positive infinity. For positive finite parameters the
penalty sums the contribution below at each magnitude ``t=|x_i|``.

```math
p(t) = \begin{cases}
\lambda t-t^2/(2a) & 0\leq t\leq a\lambda, \\
a\lambda^2/2 & t>a\lambda.
\end{cases}
```

## Traits

MCP is separable, proximable and zero-preserving. Its type-level convexity
and smoothness traits are false. Restriction returns the same function
because the parameters are shared across entries.

## Supported arguments

Direct oracles use one-based real arrays and the supported
[`SeparableFunction`](@ref SeparableFunction) iterate routes. The
numerical targets are homogeneous CPU `Float32` and `Float64`.
Array steps are matched by linear index on dense arrays, sparse iterates
and block iterates. Symmetric sparse iterate proximal oracles use scalar
steps.

## Preconditions

Exact input/output identity and independent destinations follow the
[array and precision conventions](@ref numerical-usage). Use a
nonnegative, non-NaN step with `prox_block` and as the effective step
passed through `Scaled`.

## Results

`prox` allocates the result, `prox!` returns the destination, and
`value_and_prox!` evaluates the value at the stored proximal point.
Same-kind sparse value and scalar-step prox paths visit stored entries.
An off-diagonal symmetric entry contributes twice to the value.

## Ordinary-path clauses

At a positive finite step, level and shape, the scalar proximal operator
minimizes ``p(u)+(u-|v|)^2/(2\gamma)`` over ``u\geq0``. It gives the
minimizing magnitude the sign of the input. For ``\gamma<a`` the map
uses firm thresholding. For ``\gamma\geq a`` it keeps a finite entry
when its squared magnitude exceeds ``a\gamma\lambda^2``. The
comparison uses the exact products of the represented homogeneous inputs.
Mathematical ties give positive zero.

## Range clauses

The hard comparison uses a fixed-width significand comparison when
rounded products cannot determine the result. The firm calculation
scales an underflowing residual, and the plateau is computed in scaled
form when its intermediate leaves the ordinary range. Interior firm
points and values are computed with floating-point rounding. A nonzero
firm or soft optimum that rounds to zero retains its sign. Within a few
ulps of the value's overflow boundary, rounding in the products can
give `floatmax(T)` or `Inf`.

## Endpoint clauses

Apply the first matching proximal rule for a nonnegative, non-NaN step.

| Level and step | Proximal entry |
|:--|:--|
| `λ = 0` | Identity, including at an infinite step |
| `λ = Inf` | Positive zero, including at a zero step |
| `0 < λ < Inf`, `γ = 0` | Identity |
| `0 < λ < Inf`, `γ = Inf` | Positive zero |
| `0 < λ < Inf`, `0 < γ < Inf` | NaN and signed infinities pass through. Finite entries use the threshold rules above. |

The identity preserves a stored negative zero when the destination can
represent it. At a zero level the value is zero, including at nonfinite
inputs. At any positive level a NaN entry gives a NaN value and a zero
entry gives zero. An infinite level gives an infinite value at all other
entries. For finite positive parameters the value at an infinite entry is
the saturated penalty `aλ²/2`, subject to rounding in the working type.
With infinite shape and finite positive level the value is `λ*abs(x)`.
At a finite positive step the infinite-shape map uses soft thresholding.

## Failure behavior

The constructor rejects negative and NaN levels and nonpositive and NaN
shapes. Direct array and iterate oracles reject negative and NaN steps
before writes. Array steps are checked in full, including at inactive
positions. A symmetric sparse iterate output rejects an array step.

## Limitations and non-guarantees

The homogeneous targets describe the supported precision. `Scaled` uses
its effective step and value rules with ordinary positive finite factors.
Its zero-factor constructor rejects MCP. Long homogeneous dense arrays use
vectorized variable-step blocks when the block satisfies the firm or hard
range checks. Mixed blocks and unresolved comparisons use the scalar path.

## Clauses owned elsewhere

This section links to the shared owners.

- [Array and precision conventions](@ref numerical-usage)
- [Shared range guarantees](@ref numerical-value-range)
- [Scaling a function](@ref scaling-guide)

## Evidence

[`test/mcp.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/mcp.jl)
checks the complete objective with a high-precision reference and a bounded
numerical search. It also covers range, endpoints, exact aliases, step
validation and fused values in both supported types. The
[penalty guide](@ref mcp-guide) contains an executable example.

## History

The [feature issue](https://github.com/mlakolar/ProximalBase.jl/issues/952)
and [restoration plan](https://github.com/mlakolar/ProximalBase.jl/issues/1042)
record the numerical decisions and benchmark scope.
