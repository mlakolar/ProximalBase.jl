# [The ℓ₁ norm and soft thresholding](@id l1-guide)

*Assumes: norms and basic convexity · You will learn: the weighted ℓ₁ function, its Julia constructor and soft-thresholding proximal map.*

`NormL1` represents a nonnegative weighted sum of absolute values,
often called a **weighted ℓ₁ function**. It is a mathematical function,
not an optimization algorithm. An algorithm may use
it as a **regularization penalty** to encourage sparse coefficients, but
that is one application of the function, not a different Julia type.

## Mathematical definition and Julia constructor

For ``x\in\mathbb R^p``, a nonnegative level
``\lambda_0\geq0`` and nonnegative weights ``w_i\geq0``, define

```math
g(x)=\lambda_0\sum_{i=1}^{p}w_i|x_i|.
```

The correspondence with Julia is:

| Mathematics | Julia | Meaning |
|:--|:--|:--|
| ``g(x)=\lambda_0\sum_i\lvert x_i\rvert`` | `NormL1(λ0)` | All weights are one |
| ``g(x)=\lambda_0\sum_i w_i\lvert x_i\rvert`` | `NormL1(λ0, w)` | Individual nonnegative weights |
| ``\lambda_0=1,\ w_i=1`` | `NormL1()` | The ordinary ℓ₁ norm |

Here `λ0` is the **overall scale**, `w[i]` is the weight of coordinate
`i`, and `x[i]` is the coordinate being evaluated. The array `w`
must contain one weight per entry of `x`. A scalar second argument is
not a supported way to specify weights.

```@example library_l1
using ProximalBase

x = [2.0, -3.0, 0.25]
λ0 = 0.5
w = [1.0, 0.0, 2.0]
g = NormL1(λ0, w)

# The Julia result agrees with the mathematical definition.
expected = λ0 * sum(w .* abs.(x))
@assert g(x) ≈ expected
@assert expected == 1.25
g(x)
```

A zero weight leaves its coordinate unpenalized. With **finite strictly
positive** weights and a positive overall scale, this function is a norm;
with zero weights it is a seminorm, and infinite weights introduce
extended-valued restrictions.

The function family is convex, separable and proximable, but is not smooth
at zero in any positively weighted coordinate. Consequently,
`gradient(g,x)` is not the right operation there.

## Proximal operator: soft thresholding

The [proximal operator](glossary.md#Proximal-operator) of this function
performs [soft thresholding](glossary.md#Soft-thresholding).

For an ordinary positive step ``\gamma>0``, its proximal operator is

```math
\operatorname{prox}_{\gamma g}(v)
=\operatorname*{argmin}_z
\left\{\lambda_0\sum_i w_i|z_i|
       +\frac{1}{2\gamma}\|z-v\|_2^2\right\}.
```

Separability lets this problem be solved one coordinate at a time:

```math
[\operatorname{prox}_{\gamma g}(v)]_i
=\operatorname{sign}(v_i)
\max\bigl(|v_i|-\gamma\lambda_0w_i,\ 0\bigr).
```

The threshold for entry ``i`` is ``\gamma\lambda_0w_i``. In Julia,
`prox(g,v,γ)` computes the entire vector and
`prox_block(g,v_i,i,γ)` computes one scalar entry:

```@example library_l1
γ = 1.0
v = x
result = prox(g, v, γ)
reference = sign.(v) .* max.(abs.(v) .- γ * λ0 .* w, 0.0)
@assert result ≈ reference
@assert result == [1.5, -3.0, 0.0]
@assert prox_block(g, v[3], 3, γ) == result[3]
result
```

The scalar utility `shrink(v,c)` performs the same soft threshold on
a single value when the threshold ``c`` is supplied directly.
`value_entry(g,v,i)` computes the mathematical contribution
``\lambda_0w_i|v|`` of entry ``i``:

```@example library_l1
@assert value_entry(g, x[3], 3) == λ0 * w[3] * abs(x[3])
@assert shrink(2.0, 0.5) == 1.5
```

## Using the function in an optimization problem

For instance, an ℓ₁-regularized least-squares problem is

```math
\min_{\beta\in\mathbb R^p}
\left\{\frac{1}{2n}\|y-X\beta\|_2^2
+\lambda\|\beta\|_1\right\}.
```

The `NormL1` object represents **only the second term**, not the entire
objective and not an iterative solver:

```julia
f = (1 / size(X, 1)) * LeastSquaresLoss(y, X)
g = NormL1(λ)
objective_value = f(β) + g(β)
```

See [Getting started](getting_started.md) for a complete executable
example with definitions of `X`, `y`, `β` and `λ`.

## Other operations

The same function supports coordinate restriction,
KKT scores, and per-entry proximal steps:

```@example library_l1
# Keep original coordinates 3 and 1, in that order.
small_g = restrict(g, [3, 1])
@assert small_g([1.0, -2.0]) == 2.0

# The screen of a coordinate assumed zero: positive means that the
# zero-coordinate KKT condition is violated.
grad_at_zero = [0.1, -4.0, 0.8]
@assert kkt_violation(g, grad_at_zero, 3) ≈ -0.2

# The score of the actual coordinate of a point: the margin at the zero
# coordinate 1, and a nonnegative mismatch at the nonzero coordinates 2 and 3.
xz = [0.0, -3.0, 0.25]
@assert kkt_violation(g, grad_at_zero, xz, 1) ≈ 0.1 - 0.5 * 1.0
@assert kkt_violation(g, grad_at_zero, xz, 2) ≈ 4.0
@assert kkt_violation(g, grad_at_zero, xz, 3) ≈ 0.8 + 0.5 * 2.0
# The full-point measure: max(0, largest score).
@assert kkt_violation(g, grad_at_zero, xz) ≈ 4.0

# Each coordinate may have a different proximal step.
steps = [1.0, 2.0, 0.5]
@assert prox(g, x, steps) ≈ [1.5, -3.0, 0.0]
```

`kkt_violation(g,v,i)` is the screen of a coordinate **assumed to be zero**;
`kkt_violation(g,v,x,k)` reads the coordinate `x[k]` and gives
``\lvert v_k\rvert-\lambda_0w_k`` at a zero coordinate and
``\lvert v_k+\lambda_0w_k\operatorname{sign}(x_k)\rvert`` at a nonzero one, a
stationarity mismatch; and `kkt_violation(g,v,x)` is the nonnegative
``\max(0,\max_k s_k)`` over the entries, the ``\ell_\infty`` distance from
``-v`` to ``\partial g(x)`` when the inputs are finite. A positive score at
a nonzero coordinate is not an instruction to set it to zero. See
[Coordinate descent](coordinate_descent.md) and
[Operations on functions](operations.md).

## [Supported inputs and numerical considerations](@id l1-endpoints)

The supported arguments, ownership rules and numerical endpoints now live in
the [NormL1 numerical contract](spec/norml1.md).
See its [supported arguments](@ref spec-norml1-arguments),
[results](@ref spec-norml1-results), [endpoint clauses](@ref spec-norml1-endpoints)
and [limitations](@ref spec-norml1-limitations).

## [The elastic-net penalty](@id The-elastic-net-penalty)

`ElasticNet` adds a ridge term to the weighted ℓ₁ function. For levels
``\lambda_1,\lambda_2\geq0`` and nonnegative weights ``w_i\geq0``,

```math
g(x)=\lambda_1\sum_{i}w_i|x_i|+\frac{\lambda_2}{2}\sum_{i}x_i^2 .
```

| Mathematics | Julia | Meaning |
|:--|:--|:--|
| ``\lambda_1\sum_i\lvert x_i\rvert+\frac{\lambda_2}{2}\sum_i x_i^2`` | `ElasticNet(λ1, λ2)` | All weights are one |
| ``\lambda_1\sum_i w_i\lvert x_i\rvert+\frac{\lambda_2}{2}\sum_i x_i^2`` | `ElasticNet(λ1, λ2, w)` | Weights on the ℓ₁ term only |

It is one separable function, not a sum of two function objects: it is
convex, proximable and zero-preserving, and not smooth. For a step
``\gamma>0`` its proximal operator soft thresholds each entry and then
shrinks it by the ridge factor:

```math
[\operatorname{prox}_{\gamma g}(v)]_i
=\frac{\operatorname{sign}(v_i)\max\bigl(|v_i|-\gamma\lambda_1w_i,\ 0\bigr)}{1+\gamma\lambda_2}.
```

```@example library_l1
h = ElasticNet(0.5, 2.0, [1.0, 0.0, 2.0])
x = [2.0, -3.0, 0.25]
@assert h(x) ≈ 0.5 * (2.0 + 0.0 + 0.5) + 2.0 / 2 * (4.0 + 9.0 + 0.0625)
γ = 0.25
reference = sign.(x) .* max.(abs.(x) .- γ * 0.5 .* [1.0, 0.0, 2.0], 0.0) ./ (1 + γ * 2.0)
@assert prox(h, x, γ) ≈ reference
@assert prox_block(h, x[1], 1, γ) == prox(h, x, γ)[1]
@assert value_entry(h, x[3], 3) ≈ 0.5 * 2.0 * 0.25 + 2.0 * 0.25^2 / 2
prox(h, x, γ)
```

The threshold is formed as `NormL1(λ1, w)` forms it and the ridge factor
as `SqrNormL2(λ2)` forms it, each from the unrounded factors. The ridge
kernel multiplies by the rounded factor ``1/(1+\gamma\lambda_2)``, so a
result can differ in its last bits from a division by ``1+\gamma\lambda_2``:

```@example library_l1
@assert prox_block(ElasticNet(1.0, 1.0), 3.0, 1, 0.5) == 1.6666666666666665
@assert 2.5 / 1.5 == 1.6666666666666667
```

At steps ``\gamma\geq0``, with the argument, the parameters and the
destination of one element type, a zero `λ2` gives the proximal points of
`NormL1(λ1, w)` bit for bit, and with a finite `λ2`, an entry with a zero
weight, or with a zero `λ1` and a finite weight, has the proximal points of
the ridge `SqrNormL2(λ2)` bit for bit, except that an input `-0.0` becomes
`+0.0`. A destination of another element type receives these numbers
converted to its type, and a new `SparseIterate` destination does not
store a zero result. A zero `λ2` also gives the values of `NormL1(λ1, w)`
bit for bit.
The value is the value of the ℓ₁ part, computed as `NormL1` computes it,
plus the value of the ridge part, computed as `SqrNormL2` computes it,
each with its recovery of terms that over- or underflow (see the
[numerical contract](numerics.md)). On a dense `Float32` or `Float64`
`Array` with parameters of its type, a single pass over the entries
replaces the two when a test on its sum bounds its relative error by
``(n+2)\,\varepsilon`` for ``n\leq 1/(2\varepsilon)-2`` entries. With both
levels positive and finite, a sum of the two parts below twice `floatmin`
is recomputed with the terms of both parts added in one exponent-sum
accumulator and rounded once: `ElasticNet(2.0^-538, 1.0)([2.0^-537])` is
`2.0^-1074`, although each part, `2.0^-1075`, rounds to zero on its own.

The KKT scores of `ElasticNet` use ``c_i=\lambda_1w_i``: at a zero entry
``\lvert v_i\rvert-c_i`` (`-Inf` for an infinite ridge level, as for the screen), and at
a nonzero entry ``\lvert(v_i+\lambda_2x_i)+c_i\operatorname{sign}(x_i)\rvert``, with the
ridge term left out for `λ2 = 0`. For arrays and parameters of one floating
type, `ElasticNet(λ1, 0)` and `NormL1(λ1)` give the same scores bit for bit.
The full-point value `kkt_violation(h, v, x)` is the ``\ell_\infty`` distance from
``-v`` to ``\partial h(x)`` when the inputs are finite.

```@example library_l1
vh = [0.1, -4.0, 0.8]
xh = [2.0, -3.0, 0.25]
@assert kkt_violation(h, vh, xh, 1) ≈ (0.1 + 2.0 * 2.0) + 0.5 * 1.0
@assert kkt_violation(h, vh, xh, 2) ≈ abs(-4.0 + 2.0 * -3.0)
@assert kkt_violation(h, vh, xh) ≈ 10.0
@assert kkt_violation(h, vh, [0.0, 0.0, 0.0], 3) ≈ 0.8 - 0.5 * 2.0
```

The weights, shapes, supported representations and unsupported combinations now live in
[Supported arguments](@ref spec-elasticnet-arguments).

### [Steps and endpoint rules](@id elastic-net-steps)

The scalar-step validation, endpoint rules and executed examples now live in
[Endpoint clauses](@ref spec-elasticnet-endpoints).

## [The nonnegative ℓ₁ penalty](@id The-nonnegative-l1-penalty)

`NormL1Nonneg` is the weighted ℓ₁ function on the nonnegative orthant. For a level
``\lambda_0\geq0`` and nonnegative weights ``w_i\geq0`` it is the linear function
``\lambda_0\sum_i w_ix_i`` on ``\{x\geq0\}`` and ``+\infty`` outside it:

```math
g(x)=\begin{cases}\lambda_0\sum_{i}w_ix_i, & x_i\geq0\text{ for every }i,\\ +\infty, & \text{otherwise.}\end{cases}
```

| Mathematics | Julia | Meaning |
|:--|:--|:--|
| ``\lambda_0\sum_i x_i`` on ``x\geq0`` | `NormL1Nonneg(λ0)` | All weights are one |
| ``\lambda_0\sum_i w_ix_i`` on ``x\geq0`` | `NormL1Nonneg(λ0, w)` | Weights paired with entries by linear index |

It is a separable function, convex, proximable and zero-preserving, not smooth and not an
indicator. For finite inputs and a positive finite level, weight and step, its proximal operator is a
one-sided soft threshold, with the threshold ``\gamma\lambda_0w_i`` that `NormL1` forms (the endpoint rules are below):

```math
[\operatorname{prox}_{\gamma g}(v)]_i=\max\bigl(v_i-\gamma\lambda_0w_i,\ 0\bigr).
```

```@example nonneg_l1_introduction
using ProximalBase
f = NormL1Nonneg(0.5, [1.0, 2.0])
@assert f([2.0, 1.0]) == 0.5 * (2.0 + 2.0)
@assert f([2.0, -1.0]) == Inf
@assert prox(f, [2.0, -1.0], 1.0) == [1.5, 0.0]
@assert prox_block(f, 2.0, 1, 1.0) == 1.5 && value_entry(f, 1.0, 2) == 1.0
@assert prox(NormL1Nonneg(0.0), [-2.0, 3.0], 1.0) == [0.0, 3.0]
f([2.0, 1.0])
```

The numerical contract covers the [domain](@ref spec-norml1nonneg-definition),
[weights, shapes and representations](@ref spec-norml1nonneg-arguments),
[steps and endpoint rules](@ref spec-norml1nonneg-endpoints), and
[KKT scores, restriction and scaling](@ref spec-norml1nonneg-ordinary).

## [Minimax concave penalty](@id mcp-guide)

[`MCP(λ, a)`](@ref MCP) applies a nonconvex penalty separately to each
entry. For finite positive parameters it grows as
``\lambda |x_i|-x_i^2/(2a)`` up to ``|x_i|=a\lambda``, where it
saturates at ``a\lambda^2/2``. The default shape is `a = 3`.

For a positive step ``\gamma<a``, the proximal map sets entries of
magnitude up to ``\gamma\lambda`` to zero, shrinks entries between
``\gamma\lambda`` and ``a\lambda``, and retains entries of larger
magnitude. When ``\gamma\geq a``, a finite step gives a global
minimizer by retaining entries of magnitude strictly above
``\lambda\sqrt{a\gamma}`` and returning positive zero at and below
the threshold.

```@example mcp
using ProximalBase

f = MCP(0.5, 2.0)
x = [0.25, -0.75, 2.0, -3.0]
@assert prox(f, x, 1.0) == [0.0, -0.5, 2.0, -3.0]
@assert prox(f, x, 8.0) == [0.0, 0.0, 0.0, -3.0]
```

See the [MCP numerical contract](@ref mcp-contract) for endpoint behavior and
supported step routes.

## SCAD penalty

`SCAD(λ, a=3.7)` sums a penalty that increases linearly near zero and
levels off at magnitude `a*λ`. The scalar proximal operator minimizes the
complete objective. For a step below `a-1` it uses the three-piece
thresholding rule. At larger steps it compares the competing magnitudes.

```@example scad
using ProximalBase
f = SCAD(1.0, 3.0)
@assert prox(f, [1.5, 4.0, 4.5], 1.0) == [0.5, 4.0, 4.5]
@assert prox(f, [1.5, 4.0, 4.5], 4.0) == [0.0, 0.0, 4.5]
nothing # hide
```

At step four, magnitudes zero and four have the same objective for an
input of four, and the proximal operator chooses zero. See the
[SCAD numerical contract](@ref spec-scad) for the penalty formula,
supported storage and endpoint rules.

## [Off-diagonal cardinality penalty](@id norml0offdiag-guide)

`NormL0(λ; diagonal = false)` charges `λ` for each nonzero off-diagonal entry of a
matrix and nothing for the diagonal, so a nonzero symmetric pair costs `2λ`.
The keyword changes the penalty and narrows the arguments: it accepts finite
strided `Float32` or `Float64` square matrices that Base can read and write,
under the storage preconditions of the
[numerical contract](@ref spec-norml0offdiag-arguments) (distinct entries
occupy distinct memory; a zero-stride view as the output is unsupported), and
refuses the vectors, sparse
arrays, iterates and array steps that `NormL0(λ)` accepts. `NormL0(λ)` and
`NormL0(λ; diagonal = true)` keep the all-entry penalty with its full set of
arguments. Symmetry under `==` is the caller's precondition for the symmetric
interpretation and is not checked; asymmetry alone causes no refusal, and for
an asymmetric matrix each entry is decided from its own value. The proximal map copies the diagonal and decides each
off-diagonal entry with the rule of [`NormL0`](@ref): for `λ > 0` an entry is
kept when the scalar comparison on the converted operands finds its square
above `2γλ`, and is positive zero otherwise, and at `λ = 0` the map is the
identity. `NormL0` converts
the entry, `λ` and `γ` to a working type before the comparison, which is exact
when `λ` and `γ` convert exactly to a `Float32` or `Float64` working type;
other levels and steps, for example a `Rational` or an `Int` above `2^53`, are
rounded to that type first, and with a `BigFloat` working type the converted
entry, its square and the product `2γλ` can each be rounded. A mathematical tie
can then be kept: a pair of ones at `λ = 1//3` and `γ = 3//2` is kept. The all-entry `NormL0` applies the same rule to
the diagonal, which can set a small diagonal entry to zero.

```@example norml0offdiag
using ProximalBase
V = [0.01 1.5 1.4; 1.5 3.0 -1.5; 1.4 -1.5 0.01]
f = NormL0(2; diagonal = false)
Y = prox(f, V, 0.5)
@assert Y == [0.01 1.5 0.0; 1.5 3.0 -1.5; 0.0 -1.5 0.01]
@assert f(V) == 12.0 && f(Y) == 8.0
@assert value_and_prox!(similar(V), f, V, 0.5) == 8.0
@assert [prox(NormL0(2), V, 0.5)[i, i] for i in 1:3] == [0.0, 3.0, 0.0]
@assert prox(NormL0(2; diagonal = false), [1.0 2.0; 2.0 1.0], 1.0) == [1.0 0.0; 0.0 1.0]
nothing # hide
```

At step one and level two the threshold is four, so an entry of magnitude
two is a tie and gives zero. Every refusal happens before any write. When the
only failing condition is one of the listed storage kinds (a vector, a sparse
or other non-strided matrix, an iterate or an array step), an unsupported
element type, a reinterpreted array that Base cannot both read and write, a
nonpositive or non-finite real scalar step or a non-finite entry, the call
throws `ArgumentError`; when it is a non-square matrix, the call throws
`DimensionMismatch`. When several conditions fail, which exception is thrown is
not specified. Through `Scaled`, the step and the values follow the rules of
`Scaled`, and the exception for an unsupported argument can come from the
preprocessing of `Scaled`, the checks of this function or dispatch. The oracle does not promise that the output
is positive definite. See the
[`NormL0(λ; diagonal = false)` entry](@ref api-norml0-diagonal-free) and the
[numerical contract of `NormL0(λ; diagonal = false)`](@ref spec-norml0offdiag)
for the full list.
