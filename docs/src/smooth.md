# Smooth array functions

Each function below has a mathematical definition and a Julia constructor.
**Constructing** a function supplies its fixed coefficients or data;
**calling** the resulting object with `f(x)` supplies the point at which
it is evaluated. The following sections pair each formula with its constructor,
gradient, proximal map and an executable example.

The first five sections describe array functions with value, gradient and
proximal operations on their supported inputs. The final section covers a
**different scalar observation-loss interface**, which has derivatives but
no array proximal operator. For allocating/mutating conventions,
input restrictions, shared storage and numerical limitations, see
[Operations on functions](operations.md) and
[Numerical usage](numerical_usage.md).

## Least-squares loss

```math
f(B)=\frac12\|Y-XB\|_F^2,\qquad
\nabla f(B)=X^\top(XB-Y).
```

The mathematical symbols correspond to Julia as follows: construct
`f = LeastSquaresLoss(Y, X)`, then evaluate `f(B)` or `gradient(f, B)`.
Here `X` is the real ``n\times p`` **design matrix**, `Y` is the observed
response, and `B` is the coefficient vector or matrix we want to evaluate.
`Y` can be a real length-``n`` vector or ``n\times k`` matrix. A vector response
uses length-``p`` coefficients; a matrix response uses ``p\times k``
coefficients and a gradient of that shape. There is no intercept, weight or
normalization keyword. Include an intercept column in `X` if needed, and use
`(1/n)*LeastSquaresLoss(Y,X)` for mean squared error divided by two.

The domain is all coefficient vectors/matrices of the stated dimensions.
The loss is convex, smooth and proximable; it is not declared separable.
For positive finite ``\gamma``,

```math
\operatorname{prox}_{\gamma f}(V)=
(I+\gamma X^\top X)^{-1}(V+\gamma X^\top Y).
```

The implementation uses a thin SVD cached on the first proximal call, including
when ``p>n``. The value and gradient use reusable residual scratch and read
current data entries. Do not call the same loss object concurrently without
synchronization. After changing entries of `X` or `Y`, call `reset_cache!(f)`;
construct a new object if dimensions change. The input data are not copied.

Dense coefficients, vector/matrix `SparseIterate` and vector `BlockIterate` use
supported array products. This loss has no `SymmetricSparseIterate`-owner,
scalar `prox_block`, `value_entry` or KKT-screen method. Converting an owner
to `Matrix` gives an ordinary matrix-coefficient problem if dimensions fit.
`restrict(f,S)` selects design columns only for **vector-response** losses.
`lipschitz(f)` bounds ``\|X\|_2^2``; it uses a full spectral norm for dense
strided data and a one/infinity-norm bound for other matrices.

Constructor row mismatch and operation dimension mismatch throw
`DimensionMismatch`; complex data/arguments throw `ArgumentError`. SVD-based
proximal evaluation is intended for finite data; no general non-finite-data
recovery is promised for the decomposition. Value range recovery is separate
from the ordinary matrix-product gradient arithmetic.

```@example library_ls
using ProximalBase, LinearAlgebra
X = [1.0 0; 0 2; 1 1]
y = [1.0, -1, 2]
f = LeastSquaresLoss(y, X)
x = [0.5, -0.25]
@assert f(x) ≈ sum(abs2, X*x-y)/2
@assert gradient(f,x) ≈ X'*(X*x-y)
γ = 0.3
@assert prox(f,x,γ) ≈ (I+γ*X'*X) \ (x+γ*X'*y)
out = similar(x)
@assert value_and_prox!(out,f,x,γ) ≈ f(out)
(value = f(x), gradient = gradient(f,x), prox = out)
```

## Squared Euclidean norm

```math
f(x)=\frac{\lambda}{2}\|x-y\|^2,\qquad
\nabla f(x)=\lambda(x-y),\qquad
\operatorname{prox}_{\gamma f}(x)=\frac{x+\gamma\lambda y}{1+\gamma\lambda}.
```

In Julia, `f = SqrNormL2(λ, y)` creates this function and `f(x)` evaluates
it at `x`. The argument `λ` is the nonnegative **scale**, while `y` is the
**center**. `SqrNormL2(λ)` uses center zero, and `SqrNormL2()` uses `λ=1`
and center zero. When supplied, `y` is a real array of the **same shape** as `x`. The level must be nonnegative; negative/NaN levels
throw `ArgumentError`, while `Inf` is accepted with exceptional arithmetic
rather than a general projection policy. On a matrix the norm is Frobenius.
The domain is real arrays of matching shape; symmetric owners are explicitly
supported, with exactly symmetric center parameters.

When used as a regularizer, this is the usual squared-ℓ₂ (ridge) term
with a factor of one half. `SqrNormL2(1,y)` is half
the squared distance to `y`. It is smooth, convex, proximable and separable
by trait, but it is not a `SeparableFunction` subtype and has no `value_entry`.
It has scalar `prox_block`, scalar zero-coordinate screens and `restrict`:
the latter slices the center, or returns the same uncentered object. Slicing a
center omits the constant ``(λ/2)\sum_{j\notin S}y_j^2`` from dropped coordinates
fixed at zero. When that omitted contribution is finite, restore it when comparing full
objective values; the restricted minimizers, gradient and prox are unaffected. Restriction does
not check whether the discarded contribution is finite; a non-finite constant
cannot be ignored when reasoning about the original extended-valued objective.
`lipschitz(f)` is `float(λ)`. Array-step prox is not implemented.

The value has the floating type of the argument. Gradient/prox calculations
use the wider floating type of argument and destination; level, center and
step are parameters that do not widen the model. Range recovery avoids losing
a representable result to overflow/underflow in a coefficient or difference.
The center is rounded in the model precision where it fits. See
[numerics](numerics.md) for FP32/FP64 and zero/non-finite cases.
Shape mismatch is `DimensionMismatch`; complex center, argument or output is
`ArgumentError`. Exact gradient aliasing with the input is rejected. Use
independent parameter/output buffers even where a particular kernel can safely
read an aliased center.

`iszeropreserving(f)` is true only when the center is omitted and the level is
finite. Supplying an all-zero center still follows the centered route and
reports false. A zero level makes the value zero, but does not make every
gradient or endpoint operation immune to IEEE `0*Inf`/`0*NaN` arithmetic.

```@example library_ridge
using ProximalBase
f = SqrNormL2(2.0, [1.0, 0])
x = [3.0, -1]
@assert f(x) == 5 && gradient(f,x) == [4, -2]
@assert prox(f,x,0.5) == [2, -0.5]
@assert prox_block(f, x[2], 2, 0.5) == -0.5
@assert !iszeropreserving(f) && iszeropreserving(SqrNormL2(2.0))
prox(f,x,0.5)
```

## Quadratic function

```math
f(x)=\frac12x^\top A x+b^\top x+c,\qquad
\nabla f(x)=Ax+b,\qquad
\operatorname{prox}_{\gamma f}(v)=(I+\gamma A)^{-1}(v-\gamma b).
```

In Julia, `f = QuadraticFunction(A, b, c)` represents the displayed
formula, and `f(x)` evaluates it at the variable `x`. The constructor takes
`A`, the ``p\times p`` quadratic coefficient matrix; `b`, an optional
length-``p`` linear coefficient vector; and `c`, an optional scalar constant.
When omitted, `b` and `c` are zero. The domain is real length-``p`` vectors,
including vector `SparseIterate` and `BlockIterate`; matrix and symmetric-owner
inputs are not supported. All four capability traits except separability are
true. There is no scalar `prox_block`, `value_entry` or KKT screen.

**Caller assumption:** `A` is symmetric positive semidefinite. Construction
checks real types, squareness and `b`'s length, but does not check symmetry
or eigenvalue signs. Values/gradients multiply by the supplied `A`; the prox
uses an eigendecomposition of `Symmetric(Matrix(A))`, which selects the upper
triangle by default. For nonsymmetric data these operations need not describe
one mathematical function. An indefinite matrix is outside the convex/prox
contract even though the trait reports true.

`A` and `b` are retained by reference; `c` is converted to the promoted
floating constructor type. Scratch and the eigenbasis are private. Changing
entries of `A` requires `reset_cache!(f)` before another prox. Reconstruct
for changed dimensions. `restrict(f,S)` selects `A[S,S]`, `b[S]` and retains
`c`, corresponding to omitted variables fixed at zero. Arbitrary nonzero
fixed coordinates require the caller to derive new linear/constant terms.
`lipschitz` bounds ``\|A\|_2``.

Nonsquare `A`, incompatible `b`, point or output dimensions throw
`DimensionMismatch`; non-real element types throw `ArgumentError`. Gradient
input/output identity is rejected. The prox supports exact identity. The
cached eigenvalue computation has no general non-finite-data guarantee.
Gradient and decomposition arithmetic can overflow independently of the
exceptional value-summation recovery.

```@example library_quadratic
using ProximalBase, LinearAlgebra
A = [2.0 0; 0 4]
b = [1.0, -2]
f = QuadraticFunction(A,b,3.0)
x = [2.0, -1]
@assert f(x) == dot(x,A*x)/2 + dot(b,x) + 3
@assert gradient(f,x) == A*x+b
@assert prox(f,x,0.5) ≈ (I+0.5A) \ (x-0.5b)
@assert restrict(f,[2])([-1.0]) == 7
prox(f,x,0.5)
```

## Zero function

The no-argument constructor `f = ZeroFunction()` represents
``f(x)=0`` for any supported real array `x`. Its gradient is
``\nabla f(x)=0``, and its proximal operator is the identity,
``\operatorname{prox}_{\gamma f}(x)=x``. It has all four traits true and is zero preserving.
It needs no parameters or dimensions. `lipschitz(f)==0`; `restrict(f,S)`
returns `f`; scalar `prox_block(f,v,i,γ)` returns `v`; its zero-coordinate
screen is ``|\mathrm{grad}_i|``. It has no `value_entry` method.

Value/prox also accept symmetric owners, but gradients at/into owners are
not implemented. `prox!` into an owner symmetrizes a matrix argument before
writing, as described in [symmetric parameters](symmetric_parameters.md).
For arrays, scalar and array steps are accepted and ignored numerically;
an array step must still have the correct entry count. Prox returns the
destination, combined prox returns zero. Type and shape errors still apply,
even though the function is identically zero. Exact gradient aliasing is
rejected; the value is zero even at NaN/Inf array entries.

**Activity exception:** an array `ZeroFunction` prox copies via `copyto!` when
the destination differs from the input. Between same-shaped SparseIterates or
same-layout BlockIterates this replaces the destination's active state with the
source's, including explicit zeros. A dense-source copy is an ordinary value
write. The symmetric-owner route uses coordinate value writes and preserves
its destination's existing activity. Exact in-place identity leaves state alone.

```@example library_zero
using ProximalBase
f = ZeroFunction()
x = [2.0, -1]
@assert f(x) == 0 && gradient(f,x) == zeros(2)
@assert prox(f,x,0.3) == x && prox(f,x,0.3) !== x
@assert f([NaN, Inf]) == 0
prox(f,x)
```

## Linear function

```math
f(x)=\langle c,x\rangle+d,\qquad
\nabla f(x)=c,\qquad
\operatorname{prox}_{\gamma f}(x)=x-\gamma c.
```

Construct `f = LinearFunction(c, d)` and evaluate `f(x)`. In the formula
above, the Julia array `c` supplies the coefficients and the scalar `d`
is the constant; omitting `d` sets it to zero. The function is affine on
the full real entry domain;
matrices use the Frobenius inner product. Coefficients must have the same
entry count as `x`; ordinary same-shaped arrays are the portable choice.
Value calls based on generic `dot` can accept equal-length differently shaped
arrays, while specialized dot/write methods can impose matching shapes/axes.
Use matching matrices for symmetric-owner coefficients; the pairs must agree
exactly. The gradient is a dense matrix when allocated at an owner.

The function is smooth, convex, proximable and separable by trait, but is
not declared zero preserving and has no `value_entry`. It supports scalar
`prox_block(f,v,i,γ)=v-γ*c[i]`, a zero-coordinate screen
``|\mathrm{grad}_i+c_i|``, and `restrict(f,S)=LinearFunction(c[S],d)`.
`lipschitz(f)` is zero. Array-step prox is not implemented.

The coefficient array is retained by reference, the scalar constant keeps
its constructor type, and there is no decomposition cache. Exact proximal
input/output aliasing is supported. Gradient input/output identity is rejected;
use an independent destination and coefficient array to avoid modifying the
function's parameters. Mismatched dimensions throw `DimensionMismatch`;
non-real inputs throw `ArgumentError`. Non-finite entries follow arithmetic;
finite-data overflowed values have a scaled recovery path. There is no
nonnegative restriction on `c` or `d`.

```@example library_linear
using ProximalBase, LinearAlgebra
f = LinearFunction([1.0, -2], 3.0)
x = [2.0, 1]
@assert f(x) == 3 && gradient(f,x) == [1, -2]
@assert prox(f,x,0.5) == [1.5, 2]
@assert kkt_violation(f, [-1.0, 2], 1) == 0
prox(f,x,0.5)
```

## Scalar squared loss

Scalar prediction losses use a different interface from these array
functions. The former content of this section now lives in the dedicated
[Scalar losses](scalar_losses.md) section, with mathematical definitions,
the complete `ProximalBase.ScalarLoss` interface and a
[worked `Squared()` example](scalar_squared.md).
