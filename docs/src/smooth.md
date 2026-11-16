# [Smooth array functions](@id Smooth-array-functions)

*Assumes: derivatives, matrices and convexity · You will learn: the exact formulas, constructors and supported operations for smooth array functions.*

Each function below has a mathematical definition and a Julia constructor.
**Constructing** a function supplies its fixed coefficients or data;
**calling** the resulting object with `f(x)` supplies the point at which
it is evaluated. The following sections pair each formula with its constructor,
gradient, proximal map and an executable example.

All functions on this page operate on arrays. Each has values, gradients
and [proximal operators](glossary.md#Proximal-operator) for its supported inputs.
For allocating and mutating conventions, input restrictions, shared
storage and numerical limitations, see
[Operations on functions](operations.md) and
[Numerical usage](numerical_usage.md).

## [Least-squares loss](@id Least-squares-loss)

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
when ``p>n``. It stores `X = U*Diagonal(s)*V'` and `Diagonal(s)*U'*Y`;
subsequent proximal calls use two products with `V`, without rebuilding the
factorization when the step changes. The value and gradient use reusable residual scratch and read
current data entries. Do not call `f(β)`, `gradient!`, `value_and_gradient!` or the
proximal operator of the same loss object concurrently without synchronization; the
oracles [at a supplied predictor or residual](#Least-squares-at-a-predictor-or-residual-that-the-caller-holds)
do not write the loss. After changing entries of `X` or `Y`, call `reset_cache!(f)`;
construct a new object if dimensions change. The input data are not copied.

Dense coefficients, vector/matrix `SparseIterate` and vector `BlockIterate` use
supported array products. This loss has no `SymmetricSparseIterate`,
scalar `prox_block`, `value_entry` or KKT-screen method. Converting a symmetric iterate
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

### [Least squares at a predictor or residual that the caller holds](@id Least-squares-at-a-predictor-or-residual-that-the-caller-holds)

A solver that keeps the linear predictor ``z=X\beta`` current (after a coordinate update, say) can
evaluate the loss from it without a product by ``X``. Three qualified public functions, not exported,
serve this for a vector-response `LeastSquaresLoss` or one `Scaled` of it, and a fourth returns the data
of the loss. Here ``\alpha`` is the factor of the `Scaled` wrapper, and ``\alpha=1`` for the loss itself:

```math
v=\frac{\alpha}{2}\|z-Y\|^2,\qquad
d=z-Y,\qquad
g=\alpha\,\mathrm{fl}(X^\top d),\qquad
v=\frac{\alpha}{2}\|r\|^2\ \text{ for a residual } r.
```

- `ProximalBase.value_from_predictor(f, z)` returns ``v`` from ``z``; it reads no entry of ``X``.
- `ProximalBase.value_and_gradient_from_predictor!(g, d, f, z)` writes ``d`` and ``g`` and returns ``v``,
  with one product by ``X^\top`` and none by ``X``.
- `ProximalBase.value_from_residual(f, r)` returns ``v`` from a residual ``r`` of either sign, reading
  neither ``X`` nor ``Y``.
- `ProximalBase.leastsquares_data(f)` returns `(; Y, X, α)`: the stored `Y` and `X` themselves, and
  `α = one(T)` for a `LeastSquaresLoss{T}` or the stored factor of a `Scaled` loss, as it was supplied.

**Domain.** The loss has a vector `Y` and strided `Float32` or `Float64` arrays `Y` and `X` of one type `T`;
`z`, `r`, `d` and `g` are strided vectors of that type. Anything else is a `MethodError`: a matrix response,
a sparse or other non-strided design, another element type, a nested `Scaled`. The accessor takes any
`LeastSquaresLoss` and one `Scaled` of it. `z`, `r` and `d` must have the axes of `Y` (another length, or an offset
view of the same length, is a `DimensionMismatch`). `Y` and `X` must be one-based: an offset-axes `Y` or `X` is an
`ArgumentError` when the loss is constructed. For `value_and_gradient_from_predictor!` an output `g` with offset axes is
an `ArgumentError` before anything is written.

**The scale as stored and as computed.** The accessor returns the factor ``\alpha`` as stored, with its type.
The oracles compute with it as supplied, not rounded to `T` first: a positive finite `Float32` or `Float64`
(a `Float64` factor with `Float32` data included), or an integer `α` with `Float64(α) == α`, which is
converted to `Float64` once. An integer that `Float64` does not hold (`2^53 + 1`), a `Float16`,
`BigFloat` or `Rational`, and `Inf` are an `ArgumentError`, raised before anything is written; the `Scaled`
constructor is not changed. For an integer the factor used is its `Float64` value.

**Value and rounding.** The result has type `T`. The sum of squares ``s`` is formed in `T`, in one vectorizing
loop for up to ``2^{17}`` entries and by Julia's pairwise summation for longer vectors, ``\alpha s/2`` in the
promoted type of ``\alpha`` and `T`, and the product is rounded to `T` at the result (a subnormal result can round
twice). Where the squares overflow or underflow, where the difference ``z-Y`` overflows, and where the ordinary
value ``\alpha s/2`` is above ``\texttt{floatmax}(T)/2``, the value is recomputed from the squares in units of a power of two with
``\alpha`` folded in, summed compensated, the computation that `SqrNormL2` uses for its exceptional values. The
last case keeps the error of the ordinary sum, which grows with ``n``, from deciding between a finite value and
`Inf`. The recomputation gives a finite value where the value is representable with margin (see the range
statement below). Against 256-bit references the tests require
``(n+8)\,\varepsilon_T|v|`` accuracy for finite data and a value that is representable with that margin, plus
two units of the smallest subnormal where ``|v|`` is below `floatmin(T)`.

**Range.** Write ``u=\texttt{floatmax}(T)-\texttt{prevfloat}(\texttt{floatmax}(T))``, so that ``\texttt{floatmax}(T)+u/2``
is the round-to-nearest overflow threshold, and ``w=(n+8)\varepsilon_T`` for the relative error bound above. For finite
`z` and `Y` and an exact value ``v`` with ``|v|(1+w)\le\texttt{floatmax}(T)`` the result is finite and within that
bound. For ``v`` above ``(\texttt{floatmax}(T)+u/2)(1+w)`` the result is `Inf`. Between these two bounds the result
depends on the side of the threshold. Below it, where ``|v|(1+w)>\texttt{floatmax}(T)`` and ``v`` rounds to a finite
number, the result is finite and within ``w|v|`` of ``v``, which can be a number below `floatmax(T)`
(``\alpha=`` `prevfloat(floatmax(T)/2, 4)`, `z = T[2]`, `Y = T[0]` and `X = ones(T, 1, 1)` give
`prevfloat(floatmax(T), 4)`), or it is `Inf`. From the threshold to ``(\texttt{floatmax}(T)+u/2)(1+w)``, where ``v``
rounds to `Inf`, the recomputation that the oracles share with `SqrNormL2` does not decide on which side of the
threshold the exact value lies: the result can be `floatmax(T)` or `Inf`, and a finite result there is within
``w|v|`` of ``v``. Correct overflow classification in these two regions is not guaranteed.
It is the same kind of boundary uncertainty that
[Overflow and underflow of values](numerics.md#Overflow-and-underflow-of-values) states for `NormL2` and
`GroupNormL2`; it permits no materially wrong finite value outside these regions and no finite result from
non-finite data that the value depends on. An exact positive value below the smallest subnormal can round to
`nextfloat(zero(T))` rather than to zero: for `Float64`, ``\alpha=1.5``, `z = [2.0^-537]` and `Y = [0.0]` the exact
value is `0.75*nextfloat(0.0)`, and the result is `nextfloat(0.0)` on Julia 1.13.3 and `0.0` on 1.10.12, both within the
subnormal allowance.

**Gradient.** The product ``X^\top d`` is rounded to `T` and then multiplied by ``\alpha``. An ``X^\top d`` that overflows or underflows before the scaling is not recovered, even where ``\alpha``
would bring the exact gradient back into range: with `X = [2.0^1023;;]`, `Y = [0.0]`, `z = [4.0]` and
``\alpha=2^{-1022}`` the gradient is `Inf` where the exact one is 8. A difference ``z-Y`` that overflows gives
an infinite entry of ``d`` and a non-finite value from `value_and_gradient_from_predictor!`, where
`value_from_predictor` recovers the value. For ``p>0`` the entries of the gradient are then non-finite (with the BLAS
of the tests, also those of a zero column of `X`, where `0*Inf` is `NaN`), and for ``p=0`` the gradient is empty.

**Predictor as given.** Where an entry of ``z`` is not finite the value is not finite, and the oracles do not form
``X\beta`` again to recover it. `f(β)` and `value_and_gradient!` keep their recovery from `β`: where the
product ``X\beta`` overflows although `β`, `X` and `Y` are finite, they attempt to form it again in units of a power
of two (see [Overflow and underflow of values](numerics.md#Overflow-and-underflow-of-values)). The attempt does not
recover every representable value: with ``a=2^{1020}``, ``b=2^{-1020}``, `X = [a a 0; 0 0 a]`, `β = [a, -a, b]` and
`Y = zeros(2)` the exact value is 0.5, and both methods return 0.0 because the common scaling erases `b`.
On the exceptional range the value of `Scaled(f, α)(β)` can differ from `value_from_predictor` at the same
predictor, because the wrapper forms the inner value first (issue #609).

**Buffers, overlap and concurrency.** All checks run before any write, and on an error `g`, `d` and the scratch
of the loss are unchanged. `d` must not overlap `g`, `X`, `Y`, `z` or the scratch of the loss, and `g` must
not overlap `X`, `Y`, `z` or the scratch, as `Base.mightalias` decides for nonempty arrays (an empty array is
never counted as overlapping, so an empty view of its parent is accepted). For nonempty arrays it refuses two views
of one array whose indices overlap, and in the tested cases accepts views of one array whose ranges or columns are
disjoint. It also refuses a view against its parent, or against a `vec` or `reshape` of the same memory, where the
two are disjoint; not every disjoint layout is accepted. Memory shared in ways that `mightalias` does not
detect (`unsafe_wrap`, an array type without `Base.dataids`) is a precondition on the caller. The oracles neither
read nor write the scratch `f.tmp`, so `z` or `r` may be a view of it, and they write no field of the loss: tasks
with distinct `g` and `d` may call them on one loss while `X` and `Y` are not mutated. The arrays that the
accessor returns are borrowed, not copied or protected; do not mutate them while the loss or its oracles are
in use.

```@example library_ls_predictor
using ProximalBase, LinearAlgebra
X = [1.0 0; 0 2; 1 1]
y = [1.0, -1, 2]
f = LeastSquaresLoss(y, X)
x = [0.5, -0.25]
z = X * x                                   # the predictor, formed by the caller
@assert ProximalBase.value_from_predictor(f, z) ≈ f(x)
d, g = similar(y), similar(x)
v = ProximalBase.value_and_gradient_from_predictor!(g, d, f, z)
@assert v ≈ f(x) && g ≈ X' * (z - y) && d == z - y
s = Scaled(f, 0.25)
@assert ProximalBase.value_from_residual(s, d) ≈ 0.25 * f(x)
ProximalBase.leastsquares_data(s).α
```

## [Squared Euclidean norm](@id Squared-Euclidean-norm)

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
The domain is real arrays of matching shape; symmetric iterates are explicitly
supported, with exactly symmetric center parameters (a precondition that the oracles
assume; see [symmetric parameters](symmetric_parameters.md)).

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

For supported homogeneous `Float32` or `Float64` calls, the value and
numerical outputs use that working precision. The implementation may compute
in a wider type for heterogeneous argument/destination calls, but this is
not a supported mixed-precision contract. In the current kernels, the value
uses the argument's precision; gradients and proximal points use the wider of
the argument and destination precisions. Parameters are rounded to that working
precision when representable, and proximal products are formed from their
unrounded factors before rounding. Combined gradient calls return the standalone
input value and gradient; combined proximal calls return the value at the stored
output. These implementation details do not extend the supported configurations.
Range recovery avoids losing a
representable result to overflow/underflow in a coefficient or difference.
For the supported mode, supply the level, center, step and destination in
the same floating-point type as the argument. See
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

## [Quadratic function](@id Quadratic-function)

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
including vector `SparseIterate` and `BlockIterate`; matrix and symmetric iterate
inputs are not supported. `issmooth` and `isproximable` are true;
`isseparable` and `isconvex` are false. There is no scalar `prox_block`,
`value_entry` or KKT screen.

**Caller precondition:** `A` is symmetric; positive semidefiniteness is not
required. Construction checks real types, squareness and `b`'s length, but does
not check or symmetrize `A`. Values and gradients multiply by the supplied `A`
and are valid for a symmetric indefinite `A`; the prox uses an
eigendecomposition of `Symmetric(Matrix(A))`, which selects the upper
triangle by default. For nonsymmetric data these operations need not describe
one mathematical function.

**Step domain of the prox:** let ``\gamma`` be the step after the factors of
any `Scaled` wrappers are applied, converted to the element type ``T`` of the
function, and let ``\lambda_{\min}`` be the smallest eigenvalue of the cached
decomposition of `A`, a computed approximation of the true smallest eigenvalue.
When ``\lambda_{\min}<0`` and the rounded product ``\gamma(-\lambda_{\min})`` is not
``<1`` (a NaN step included), the prox throws `DomainError` before the output
is written; otherwise it applies the formula above, and when
``\lambda_{\min}\ge0`` no step check is made. For a nonnegative finite
``\gamma`` and finite cached eigenvalues, the check accepts exactly when every
computed denominator ``1+\gamma\lambda_i`` is positive.

In exact arithmetic and for ``\gamma>0``, the proximal subproblem has a unique
minimizer if and only if ``1+\gamma\lambda_{\min}(A)>0`` for the true smallest eigenvalue. The check
tests computed values, so near that boundary it can accept a step for which the
true ``I+\gamma A`` is not positive definite, and reject one for which it is. The
second case includes a singular positive semidefinite `A` whose computed
``\lambda_{\min}`` is slightly negative: steps with ``\gamma(-\lambda_{\min})\ge1``
throw for it, which is about ``\gamma\ge1/|\lambda_{\min}|``.

**Convexity:** `isconvex` is `false` for `QuadraticFunction` because convexity
depends on the matrix: for a symmetric `A`, `f` is convex if and only if
``A\succeq0``, which the package does not check. A successful prox, including one at an accepted step on
an indefinite `A`, does not establish that `f` is convex. A solver that needs
convexity validates the model itself, or relies on a documented caller
assertion, and does so again after the data change.

`A` and `b` are retained by reference; `c` is converted to the promoted
floating constructor type. Scratch and the eigenbasis are private. For a nonempty matrix, the first prox
caches the eigendecomposition; subsequent calls use two eigenbasis products.
An empty matrix recomputes its empty decomposition on each call.
Changing entries of `A` requires `reset_cache!(f)` before another prox,
which then uses the new spectrum and step domain. Changes to `b` alone
need no reset: the prox reads `b` on each call. Reconstruct
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

## [Zero function](@id Zero-function)

The no-argument constructor `f = ZeroFunction()` represents
``f(x)=0`` for any supported real array `x`. Its gradient is
``\nabla f(x)=0``, and its proximal operator is the identity,
``\operatorname{prox}_{\gamma f}(x)=x``. It has all four traits true and is zero preserving.
It needs no parameters or dimensions. `lipschitz(f)==0`; `restrict(f,S)`
returns `f`; scalar `prox_block(f,v,i,γ)` returns `v`; its zero-coordinate
screen is ``|\mathrm{grad}_i|``. It has no `value_entry` method.

Value/prox also accept symmetric iterates, but gradients at/into symmetric iterates are
not implemented. `prox!` into a symmetric iterate symmetrizes a matrix argument before
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
write. The symmetric-iterate route uses coordinate value writes and preserves
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

## [Linear function](@id Linear-function)

```math
f(x)=\langle c,x\rangle+d,\qquad
\nabla f(x)=c,\qquad
\operatorname{prox}_{\gamma f}(x)=x-\gamma c.
```

Construct `f = LinearFunction(c, d)` and evaluate `f(x)`. In the formula
above, the Julia array `c` supplies the coefficients and the scalar `d`
is the constant; omitting `d` sets it to zero. The function is affine on
the full real entry domain;
matrices use the Frobenius inner product. The coefficients and the argument
must have the same dimensions, `size(c) == size(x)`: a vector for a vector, an
`m × n` matrix for an `m × n` matrix, in any storage (dense, sparse, a view, an
`Adjoint` or `Transpose`, a `Diagonal`, an iterate). For a symmetric iterate,
`c` is a `p × p` matrix, the shape of the matrix the iterate represents, and its
mirrored pairs must agree exactly. Every value, gradient and proximal oracle
applies this rule and refuses any other pair, including one with the same number
of entries in another shape, with a `DimensionMismatch` before it writes.
The gradient is a dense matrix when allocated at a symmetric iterate.

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
non-real inputs throw `ArgumentError`. Non-finite entries follow arithmetic,
except at a zero of a matrix `SparseIterate` (below); finite-data overflowed values
have a scaled recovery path. There is no nonnegative restriction on `c` or `d`.

At a matrix `SparseIterate` the value `f(x)` is computed from the stored entries of
`x` whose value is nonzero, in `O(nactive(x))` and without forming a zero for an unstored
entry; the order of the summation is not specified. A `Diagonal` coefficient array is the
exception to the cost: its value is summed over the nonzero stored diagonal entries in
`O(p)` (it scans the `p` diagonal positions), and the other entries of `x` are ignored, as
in the diagonal-only `dot` of `Diagonal`. An entry that is not stored, or is stored as
`+0` or `-0`, adds nothing, even when the coefficient at that position is `NaN`,
`Inf` or `-Inf`, and `c` is not read there; a nonzero entry follows ordinary
arithmetic, also when it or its coefficient is not finite. Historical traversal measurements
are recorded in the [implementation notes](https://github.com/mlakolar/ProximalBase.jl/blob/dd8c1a2e9846d434e81fae63903bb667c57c8158/docs/design/function-docs-history-921.md#linearfunction-sparse-value-measurements). The constant `d` is added
after the sum. When the ordinary value is not finite but `d`, the nonzero entries
and the coefficients at them are, the value is recomputed from those entries (a
coefficient at a zero does not prevent it) as exponent-separated terms, each product
formed from its two factors without scaling one before the other acts. The sum is then
collapsed to one value with its own exponent, and `d` is added to that value in its own
units, so that when the compensated sum of the products cancels to zero the value is `d`,
and a small `d` is not absorbed by the larger terms. The accuracy is the absolute bound
`B = 4 n eps(T) (sum |cᵢxᵢ| + |d|)`, not relative to a dot much smaller than its terms, and the
returned value is finite when the exact value plus `B` is within the range of the type. When the
products lie far beyond the range, so that `B` exceeds it, there is no guarantee of a finite
value: extreme cancellation can give an infinity for a finite exact value (see
[Overflow and underflow of values](numerics.md#Overflow-and-underflow-of-values)).

For a dense `c` that is not a `Diagonal`, `f(Matrix(x))` and `dot(c, x) + d` take the
ordinary `dot`, which visits all positions, so a `NaN`
coefficient at a zero gives `NaN` there: the value differs from them on purpose in that
case. When the participating entries of `x`, the coefficients at them, the coefficients
at the positions where `x` is zero, and `d` are all finite and the ordinary products and partial sums remain finite, the
value agrees with them to the accuracy of the sum. When a
participating entry is not finite, the order of summation can change the IEEE class: with
`c = ones(2, 2)`, `d = 0` and `h = 2^exponent(floatmax(T))`, storing `x[2] = h`, `x[3] = h`
and then `x[1] = -Inf`, a sum that adds `h` and `h` before the `-Inf` gives `NaN`, while
`f(Matrix(x))` and `dot(c, x) + d` give `-Inf`; `f(x)` may give either. This rule is for the matrix
`SparseIterate`: a vector `SparseIterate` keeps its own rules, and a `SymmetricSparseIterate`
forms the terms at its stored zeros, so a `NaN` coefficient there gives `NaN`.

```@example library_linear_sparse
using ProximalBase, LinearAlgebra
x = SparseIterate(2, 2)
x[1, 1] = 2.0
c = [1.0 NaN; 1.0 1.0]            # NaN at a position that x does not store
f = LinearFunction(c, 0.25)
@assert f(x) == 2.25
@assert isnan(f(Matrix(x)))
f(x)
```

```@example library_linear
using ProximalBase, LinearAlgebra
f = LinearFunction([1.0, -2], 3.0)
x = [2.0, 1]
@assert f(x) == 3 && gradient(f,x) == [1, -2]
@assert prox(f,x,0.5) == [1.5, 2]
@assert kkt_violation(f, [-1.0, 2], 1) == 0
prox(f,x,0.5)
```
