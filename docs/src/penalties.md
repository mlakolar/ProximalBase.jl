# The ℓ₁ norm and soft thresholding

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
a zero-coordinate KKT test, and per-entry proximal steps:

```@example library_l1
# Keep original coordinates 3 and 1, in that order.
small_g = restrict(g, [3, 1])
@assert small_g([1.0, -2.0]) == 2.0

# Positive means the zero-coordinate KKT condition is violated.
grad_at_zero = [0.1, -4.0, 0.8]
@assert kkt_violation(g, grad_at_zero, 3) ≈ -0.2

# Each coordinate may have a different proximal step.
steps = [1.0, 2.0, 0.5]
@assert prox(g, x, steps) ≈ [1.5, -3.0, 0.0]
```

For an active working set, `kkt_violation` is a **screen for coordinates
assumed to be zero**, not an optimality certificate for an arbitrary
nonzero coordinate. See [Coordinate descent](coordinate_descent.md) and
[Operations on functions](operations.md).

## Supported inputs and numerical considerations

`NormL1` acts on real vector or matrix entries. When weights are given,
they must have the same total number of entries as the argument, paired
in Julia's linear indexing order. The function also has explicitly
supported sparse and symmetric iterate operations; for symmetric matrix
storage, matching off-diagonal weights must agree. See
[Symmetric parameters](symmetric_parameters.md).

`prox!` writes to a supplied destination; `value_and_prox!` writes
the proximal point and returns `g` evaluated at that stored output.
Exact input/output identity is supported, but partially overlapping
views are not. Weights are stored by reference, so modifying them also
changes the function.

Infinite parameters, NaNs, zeros, overflow and sparse active-zero
semantics require extra care. A zero weight leaves an entry unpenalized,
and an infinite weight fixes finite trial values to zero, but the behavior
of non-finite trial values is more subtle. The detailed contracts belong
in [Numerical usage](numerical_usage.md) and the
[Numerical contract](numerics.md).
