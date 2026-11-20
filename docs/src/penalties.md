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

### Weights, shapes and representations

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
validates (see [Symmetric parameters](symmetric_parameters.md)).

Not supported: an array of steps with a `SymmetricSparseIterate`
destination (an `ArgumentError` before any write), or with a dense
destination and a `SymmetricSparseIterate` argument (a `MethodError`);
the vector form of `prox_block`; complex arguments or weights.

### [Steps and endpoint rules](@id elastic-net-steps)

`prox`, `prox!` and `value_and_prox!` throw an `ArgumentError` for a
negative or `NaN` scalar step before anything is written. The entries of
an array of steps are not checked: they must be nonnegative, a documented
precondition. `prox_block` does not check its step, and `Scaled` keeps its own step rule.

The optimized array-step loop and the general loop store the same results,
bit for bit. See the [implementation notes](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/function-docs-history-921.md#elastic-net-step-traversal)
for branch-selection conditions and the reason array-step validation remains a
caller precondition.

```@example library_l1
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

```@example library_l1
hard = ElasticNet(1.0, Inf)
@assert prox(hard, [3.0, NaN], 0.0) == [0.0, 0.0]
@assert hard([0.0, 0.0]) == 0 && hard([0.0, 1e-300]) == Inf
@assert kkt_violation(hard, [5.0], 1) == -Inf
@assert kkt_violation(hard, [5.0], [0.0], 1) == -Inf
@assert kkt_violation(hard, [5.0], [1.0], 1) == Inf
```

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

```@example nonneg_l1
using ProximalBase
f = NormL1Nonneg(0.5, [1.0, 2.0])
@assert f([2.0, 1.0]) == 0.5 * (2.0 + 2.0)
@assert f([2.0, -1.0]) == Inf
@assert prox(f, [2.0, -1.0], 1.0) == [1.5, 0.0]
@assert prox_block(f, 2.0, 1, 1.0) == 1.5 && value_entry(f, 1.0, 2) == 1.0
@assert prox(NormL1Nonneg(0.0), [-2.0, 3.0], 1.0) == [0.0, 3.0]
f([2.0, 1.0])
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
recovery of terms below `floatmin` (see the [numerical contract](numerics.md)): bit for bit on a
`SparseIterate`, a `BlockIterate` and a `SymmetricSparseIterate`, and within the error bound of the sum on a
dense array. `value_and_prox!` returns the value at the stored output.

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
[Symmetric parameters](symmetric_parameters.md)).

Not supported: an array of steps with a `SymmetricSparseIterate` destination (an `ArgumentError`
before any write), or with a dense destination and a `SymmetricSparseIterate` argument (a
`MethodError`); the vector form of `prox_block`; complex arguments or weights.

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

### KKT scores, restriction and scaling

For finite inputs and ``c_i=\lambda_0w_i``, the screen of a coordinate assumed zero is ``-v_i-c_i``, and
`kkt_violation(g,v,x,k)` is ``-v_k-c_k`` at a zero coordinate, ``\lvert v_k+c_k\rvert`` at a
positive one and `Inf` at a negative one, which is outside the domain. The full-point value is the
``\ell_\infty`` distance from ``-v`` to ``\partial g(x)`` for a finite point in the domain, and `Inf` at a
finite point with a negative entry. The rules for `NaN` and infinite entries are in the
[numerical contract](numerics.md).

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
[Scaling a function](calculus.md). For a finite level a zero factor keeps the domain and the infinite weights:
`0 * NormL1Nonneg(λ0, w)` is `NormL1Nonneg(0, w)`. An infinite level with a zero factor throws an `ArgumentError`.

```@example nonneg_l1
r = restrict(h, [3, 1])
@assert r isa NormL1Nonneg && r.w == [4.0, 1.0] && r([1.0, 1.0]) == 0.5 * 5.0
s = Scaled(h, 2.0)
@assert s([1.0, 1.0, 1.0]) == 2.0 * h([1.0, 1.0, 1.0])
```

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

## Off-diagonal cardinality penalty

`NormL0OffDiag(λ)` charges `λ` for each nonzero off-diagonal entry of a
matrix and nothing for the diagonal, so a nonzero symmetric pair costs `2λ`.
It accepts finite dense `Float32` or `Float64` matrices that are symmetric
under `==`. The proximal map copies the diagonal and decides each
off-diagonal entry with the rule of [`NormL0`](@ref): an entry is kept when
its square exceeds `2γλ` and is positive zero otherwise, with the
comparison exact when `λ` and `γ` convert exactly to the working type of
`NormL0`; other levels and steps, for example a `Rational` or an `Int` above
`2^53`, are rounded to that type first. The all-entry
`NormL0` applies the same rule to the diagonal, which can set a small
diagonal entry to zero.

```@example norml0offdiag
using ProximalBase
V = [0.01 1.5 1.4; 1.5 3.0 -1.5; 1.4 -1.5 0.01]
f = NormL0OffDiag(2)
Y = prox(f, V, 0.5)
@assert Y == [0.01 1.5 0.0; 1.5 3.0 -1.5; 0.0 -1.5 0.01]
@assert f(V) == 12.0 && f(Y) == 8.0
@assert value_and_prox!(similar(V), f, V, 0.5) == 8.0
@assert [prox(NormL0(2), V, 0.5)[i, i] for i in 1:3] == [0.0, 3.0, 0.0]
@assert prox(NormL0OffDiag(2), [1.0 2.0; 2.0 1.0], 1.0) == [1.0 0.0; 0.0 1.0]
nothing # hide
```

At step one and level two the threshold is four, so an entry of magnitude
two is a tie and gives zero. The oracle throws `ArgumentError` before it
writes anything for an asymmetric or non-finite matrix, a matrix type that is
not strided, mixed element types and a step that is not finite and positive.
It does not promise that the output is positive definite. See the
[`NormL0OffDiag`](@ref) entry and the
[cardinality endpoint rules](numerics.md#Off-diagonal-cardinality-penalty)
for the full list.
