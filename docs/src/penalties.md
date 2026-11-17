# The ℓ₁ norm and soft thresholding

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

## Supported inputs and numerical considerations

`NormL1` acts on real vector or matrix entries. When weights are given,
they must have the same total number of entries as the argument, paired
in Julia's linear indexing order. The function also has explicitly
supported sparse and symmetric iterate operations; for symmetric matrix
storage, matching off-diagonal weights must agree: this is a precondition that the
oracles assume and `check_symmetric_parameters` validates. See
[Symmetric parameters](symmetric_parameters.md).

`prox!` writes to a supplied destination; `value_and_prox!` writes
the proximal point and returns `g` evaluated at that stored output.
Exact input/output identity is supported, but partially overlapping
views are not. Weights are stored by reference, so modifying them also
changes the function.

Infinite parameters, NaNs, zeros, overflow and sparse active-zero
semantics require extra care. A zero weight leaves an entry unpenalized.
An infinite weight sets the proximal output to zero for a **finite input value**, including at zero
level and zero step. For non-finite inputs (`NaN`, `±Inf`) with finite level,
there is **no fixed-zero guarantee**: ordinary soft-threshold arithmetic can
return `NaN` ([#853](https://github.com/mlakolar/ProximalBase.jl/issues/853)).
A genuinely infinite overall level `λ0 = Inf` separately fixes every
positive-weight entry at `+0` at any nonnegative step, including non-finite
inputs (decision 49). At `γ = 0` this is the limit `γ → 0+` of the proximal map
of the fixed function, which is the projection onto the closure of its domain,
not soft thresholding by `0⋅Inf`
([#665](https://github.com/mlakolar/ProximalBase.jl/issues/665)). Zero-weight
entries stay free:

```julia
prox(NormL1(Inf, [1.0, 0.0, 2.0]), [3.0, 4.0, 5.0], 0.0)  # [0.0, 4.0, 0.0]
prox(NormL1(Inf), [1.0, 2.0], 0.0)                         # [0.0, 0.0]
prox(NormL1(1.0), [1.0, 2.0], 0.0)                         # [1.0, 2.0]: finite level
```

Before #665 these zero-step calls with an infinite level returned their input.
The infinite overall level and an infinite coordinate weight are distinct: the
latter, at a finite level, keeps the #853 rule above at every step.

A threshold formed from entirely finite factors may itself overflow to `Inf`.
For an input of `±Inf` this can also return `NaN`, even though the mathematical
threshold is finite. This is a **documented, deferred limitation**
([#866](https://github.com/mlakolar/ProximalBase.jl/issues/866)), not an
approved change to the proximal kernel. The decision preserves performance
of normal `Float32`/`Float64` weighted operations; the measured cost of extra
checks is recorded in the [numerical contract](numerics.md).
See also [Numerical usage](numerical_usage.md).

## The elastic-net penalty

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

### Steps and endpoint rules

`prox`, `prox!` and `value_and_prox!` throw an `ArgumentError` for a
negative or `NaN` scalar step before anything is written. The entries of
an array of steps are not checked: they must be nonnegative, a documented
precondition. This is an exception to the requirement of
[#908](https://github.com/mlakolar/ProximalBase.jl/issues/908) that
negative and `NaN` steps fail before writes: a check reads every step,
while such a call reads the steps before its loop only to choose the loop,
up to the first block of 256 steps that holds a step outside the ordinary
branches (below), and a call from a `SparseIterate` into a `SparseIterate`
reads none; the check would add a pass over the steps, or the rest of one
([#874](https://github.com/mlakolar/ProximalBase.jl/pull/874)).
`prox_block` does not check its step, and `Scaled` keeps its own step rule.

Let `T` be `float` of the element type of the entries. A step ``\gamma``
of entry ``i`` takes the ordinary branches when both conditions hold:

- `λ1` is zero, or the product ``\gamma\lambda_1`` is a positive normal
  number of the type `R` in which it is formed. `R` is the promotion of `T`
  with one type for each of ``\gamma``, `λ1` and the weight of entry ``i``:
  `float` of its type, or `Float16` for a `Bool` (an `ElasticNet` without
  weights has the weight `true`).
- For `T` = `Float32` or `Float64`: one of ``\gamma``, `λ2` is zero or not
  finite, or ``\gamma\lambda_2``, formed in the promotion of `T` with `float`
  of the types of ``\gamma`` and `λ2`, lies in ``[2^{-126}, 2^{125}]``
  (`Float32`) or ``[2^{-1022}, 2^{1021}]`` (`Float64`). Other types of `T`
  have no condition on ``\gamma\lambda_2``.

When the argument and the array of steps have concrete element
types, a call other than from a `SparseIterate` into a `SparseIterate` runs
a loop specialized to these branches if every step takes them; one step
outside them, anywhere, sends the whole call to the general loop. Both loops
store the same results, bit for bit.

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
the rules of `NormL1` above (decision 49, #665, #853 and
[#653](https://github.com/mlakolar/ProximalBase.jl/issues/653)), and a
threshold that overflows from finite factors keeps the #866 limitation.

```@example library_l1
hard = ElasticNet(1.0, Inf)
@assert prox(hard, [3.0, NaN], 0.0) == [0.0, 0.0]
@assert hard([0.0, 0.0]) == 0 && hard([0.0, 1e-300]) == Inf
@assert kkt_violation(hard, [5.0], 1) == -Inf
@assert kkt_violation(hard, [5.0], [0.0], 1) == -Inf
@assert kkt_violation(hard, [5.0], [1.0], 1) == Inf
```
