# Restricting functions to selected coordinates

*Assumes: vector indexing and quadratic objectives · You will learn: exactly what `restrict(f, S)` does and what happens to omitted coordinates.*

This page explains how `restrict(f, S)` constructs a lower-dimensional
function, what happens to coordinates outside `S`, and why the original
function **need not be separable**. We also distinguish setting omitted
coordinates to zero from holding them at their current nonzero values.

**Question:** If an algorithm is optimizing only a subset of variables,
what function should it use for that smaller problem?

The call `restrict(f, S)` constructs a function of the selected coordinates
`S`. **The original function does not have to be
[separable](glossary.md#Separable-function).** However,
not every mathematical function has a `restrict` method; this operation is
implemented for specific families.

## What happens to variables outside S?

Suppose `f` takes a vector with `p` entries. Let
``S=(s_1,\ldots,s_m)`` be the selected indices, and define an embedding
``P_S:\mathbb R^m\to\mathbb R^p`` by placing ``u_j`` at index
``s_j`` and **zero at every coordinate outside S**. For example,

```math
p=3,\quad S=(1,3),\quad
u=\begin{bmatrix}a\\b\end{bmatrix}
\quad\Longrightarrow\quad
P_Su=\begin{bmatrix}a\\0\\b\end{bmatrix}.
```

For most supported functions, `restrict(f,S)` represents

```math
f_S(u)=f(P_Su).
```

This is a **zero-fixed subproblem**: omitted variables are fixed at zero,
not optimized over, and not held at their values in some current iterate.
The returned function acts on the shorter vector `u`. It does not change
`f` or modify any iterate. Use valid, distinct indices in `S` to
represent a coordinate subset.

## A nonseparable example: quadratic functions

Consider

```math
f(x)=\frac12 x^\mathsf{T}Ax+b^\mathsf{T}x+c,
\qquad
A=\begin{bmatrix}2&1\\1&3\end{bmatrix},
\quad b=\begin{bmatrix}1\\2\end{bmatrix},\quad c=4.
```

The cross-term in ``x^\mathsf{T}Ax`` couples ``x_1`` and ``x_2``,
so this quadratic is **not separable**. Nevertheless, if ``S=(1)``,
fixing ``x_2=0`` leaves

```math
f_S(u)=\frac12(2u^2)+u+4=u^2+u+4.
```

In general, the implementation selects the principal submatrix and
matching linear coefficients:

```math
A_S=A[S,S],\quad b_S=b[S],\quad
f_S(u)=\tfrac12u^\mathsf{T}A_Su+b_S^\mathsf{T}u+c.
```

```@example operations_restrict
using ProximalBase
A = [2.0 1.0; 1.0 3.0]
fquad = QuadraticFunction(A, [1.0, 2.0], 4.0)
qsmall = restrict(fquad, [1])
@assert qsmall([3.0]) ≈ fquad([3.0, 0.0])
@assert qsmall([3.0]) ≈ 16.0
qsmall([3.0])
```

## A nonseparable example: least squares

For the squared-error loss

```math
f(\beta)=\tfrac12\|y-X\beta\|_2^2,
```

restricting to the columns indexed by ``S`` gives

```math
f_S(u)=\tfrac12\|y-X_{:,S}u\|_2^2
      = f(P_Su).
```

This is not a separable problem: the selected coefficients still interact
through the design matrix. The response ``y`` stays unchanged; **omitted
coefficients are set to zero**.

```@example operations_restrict
X = [1.0 2.0; 3.0 1.0; 0.0 2.0]
y = [1.0, 2.0, 3.0]
fls = LeastSquaresLoss(y, X)
lsmall = restrict(fls, [1])
@assert lsmall([0.5]) ≈ fls([0.5, 0.0])
lsmall([0.5])
```

This method applies to a vector-response `LeastSquaresLoss`. For more
general coefficient/response shapes, check the supported methods before
using `restrict`.

## Separable functions and an important constant

For a separable function ``g(x)=\sum_i g_i(x_i)``, selecting ``S``
usually keeps just the terms for those entries:

```math
g_S(u)=\sum_{j=1}^m g_{s_j}(u_j).
```

If the omitted terms are zero when their coordinates are zero, this also
equals ``g(P_Su)``. For example, the ``\ell_1`` penalty has that
property.

**Centered squared norms are different.** For

```math
f(x)=\frac{\lambda}{2}\|x-a\|_2^2,
```

the implemented restriction is

```math
f_S(u)=\frac{\lambda}{2}\|u-a_S\|_2^2,
```

while the original function evaluated with omitted coordinates fixed
at zero is

```math
f(P_Su)=f_S(u)+\underbrace{
\frac{\lambda}{2}\sum_{i\notin S}a_i^2
}_{\text{constant from omitted entries}}.
```

Thus the reduced function has the same derivatives and minimizers as the
zero-fixed problem, but its values differ by a constant. If you compare
the reduced objective against the original full objective, add the
constant back.

```@example operations_restrict
fridge = SqrNormL2(2.0, [1.0, 3.0])
rsmall = restrict(fridge, [1])
u = [2.0]
@assert rsmall(u) ≈ 1.0
@assert fridge([2.0, 0.0]) ≈ 10.0
@assert fridge([2.0, 0.0]) ≈ rsmall(u) + 9.0
```

## What if omitted variables are currently nonzero?

Suppose the current iterate is ``x_0``, and you want to update only
coordinates in ``S`` while keeping all other coordinates at their
**current values**. That is a different subproblem:

```math
h(u)=f\bigl(x_{0,S^c}+P_Su\bigr),
```

where ``x_{0,S^c}`` means ``x_0`` with its selected entries set
to zero.

**Do not assume `restrict(f,S)` constructs this function.** In least
squares, for example, the unchanged coefficients contribute to the
residual. To construct this nonzero-fixed subproblem, you need to
adjust the response:

```math
y_{\mathrm{adjusted}}
=y-X_{:,S^c}\beta_{0,S^c},\qquad
h(u)=\tfrac12\|y_{\mathrm{adjusted}}-X_{:,S}u\|_2^2.
```

For a quadratic, fixing omitted coordinates to nonzero values creates
additional linear and constant terms from the cross-coupling. These
terms must also be derived and included by the caller.

## Which functions support restrict?

| Function | What the method keeps | Omitted coordinates |
|:--|:--|:--|
| `NormL1` | Selected entry weights (or the unweighted penalty) | Set to zero; their penalty is zero |
| `ElasticNet` | Both levels and the selected entry weights (or the unweighted penalty) | Set to zero; their penalty is zero |
| `SqrNormL2` | Selected center entries, if centered | Set to zero; omitted-center constant is **dropped** |
| `LinearFunction` | Selected coefficients and the original constant | Set to zero |
| `QuadraticFunction` | Principal submatrix, selected linear term and constant | Set to zero |
| `LeastSquaresLoss` (vector response) | Selected design columns and original response | Coefficients set to zero |
| `ZeroFunction` | Same zero-function object | No effect |
| `Scaled` | Restricts its wrapped function, then preserves scaling | Depends on the wrapped function |

There is **no universal** `restrict` fallback for arbitrary
`AbstractFunction` objects. A new function may be separable and still
lack a `restrict` method until its author defines one. Likewise,
nonseparability is not a reason to reject restriction.

For supported argument types and method signatures, see the
[API reference](api/interface.md).

