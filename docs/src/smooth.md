# [Smooth array functions](@id Smooth-array-functions)

*Assumes: derivatives, matrices and convexity · You will learn: the exact formulas, constructors and supported operations for smooth array functions.*

Each function below has a mathematical definition and a Julia constructor.
**Constructing** a function supplies its fixed coefficients or data;
**calling** the resulting object with `f(x)` supplies the point at which
it is evaluated. The following sections pair each formula with its constructor,
gradient, proximal map and an executable example.

All functions on this page operate on arrays. Each section describes its
supported values, gradients and [proximal operators](glossary.md#Proximal-operator).
For allocating and mutating conventions, input restrictions, shared
storage and numerical limitations, see
[Operations on functions](operations.md) and
[Array and precision conventions](numerical_usage.md).

## [Log-determinant loss](@id Log-determinant-loss)

[`LogDetLoss(S, w)`](@ref LogDetLoss) defines
``f(X)=w[\operatorname{tr}(SX)-\log\det(X)]`` for finite, exactly symmetric
positive-definite matrices. The full-matrix Frobenius gradient is
``w(S-X^{-1})``. Off-diagonal derivatives with respect to the independent
coordinates of a symmetric matrix would count both mirrored entries.
The gradient has no finite global Lipschitz bound.

The constructor copies a finite, nonempty, exactly symmetric `S` and takes a
finite positive weight, defaulting to one. The matching argument and output
shapes are square, with one-based axes and a common `Float32` or `Float64`
precision. A [`SymmetricSparseIterate`](@ref) supplies its logical matrix
through [`matrixview`](@ref). Results are dense. Pass a full writable strided
matrix or view to reuse a destination. Exact in-place prox is supported.
Gradient destinations require independent storage.

The value uses Cholesky. It returns `NaN` at a logical NaN entry and `Inf`
for other domain failures, with an all-NaN gradient in both cases. A zero step
is the identity, including outside the domain. A positive finite step accepts
finite inputs, including asymmetric or indefinite matrices. It diagonalizes
``(X+X^\mathsf{T})/2-\gamma wS`` and replaces each eigenvalue ``d`` with the
positive root of ``z^2-dz-\gamma w=0``. Negative, NaN and infinite steps throw
before writing. Non-finite entries at a positive step also throw before
factorization and writes.

At extreme scales the shifted matrix, its spectrum, or the reconstruction can
exceed the working range. The proximal point is positive definite in exact
arithmetic. Rounding can take the stored point outside the numerical domain,
in which case the fused value follows the standalone domain rule.

```@example library_logdet
using ProximalBase, LinearAlgebra
S = [1.0 0.25; 0.25 1.0]
X = [2.0 0.5; 0.5 2.0]
f = LogDetLoss(S, 0.75)
v, g = value_and_gradient(f, X)
@assert v ≈ 0.75 * (tr(S * X) - logdet(Symmetric(X)))
@assert g ≈ 0.75 * (S - inv(X))
Y = similar(X)
@assert value_and_prox!(Y, f, X, 0.5) == f(Y)
@assert isposdef(Y)
Y
```

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
current data entries. Synchronize calls to `f(β)`, `gradient!`, `value_and_gradient!`,
`kkt_violation(f, v, β)` and the proximal operators on the same loss object. The
oracles [at a supplied predictor or residual](#Least-squares-at-a-predictor-or-residual-that-the-caller-holds)
do not write the loss. After changing entries of `X` or `Y`, call `reset_cache!(f)`;
construct a new object if dimensions change. The input data are not copied.

Dense coefficients, vector/matrix `SparseIterate` and vector `BlockIterate` use
supported array products. This loss has no `SymmetricSparseIterate`,
scalar `prox_block` or `value_entry` method. See [smooth stationarity scores](@ref smooth-kkt)
for the full-point KKT form. Converting a symmetric iterate
to `Matrix` gives an ordinary matrix-coefficient problem if dimensions fit.
`restrict(f,S)` selects design columns only for **vector-response** losses.
`lipschitz(f)` bounds ``\|X\|_2^2``; it uses a full spectral norm for dense
strided data and a one/infinity-norm bound for other matrices.

Constructor row mismatch and operation dimension mismatch throw
`DimensionMismatch`; complex data/arguments throw `ArgumentError`. See the [LeastSquaresLoss limitations and non-guarantees](@ref spec-leastsquares-limitations).

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
of the loss. The two predictor oracles also take matrix-response least squares. Here
``\alpha`` is the factor of the `Scaled` wrapper, and ``\alpha=1`` for the loss itself. The value oracles and the accessor serve `SqrtLeastSquaresLoss` as well, which has its
[own section](#Square-root-least-squares-is-not-smooth) below:

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
`z`, `r`, `d` and `g` are strided vectors of that type. For the two predictor oracles, a matrix response requires
`X::Matrix{T}` and `Y::Matrix{T}` and strided matrix buffers of the same type. The predictor and
residual output match the axes of the `n × K` response and the one-based gradient output
has size `p × K` for `p` design columns. The objective sums squared differences over all
`nK` entries. Other layouts, mixed array element types and nested `Scaled` wrappers produce
`MethodError`. `value_from_residual` continues to take vector responses. The accessor takes any
`LeastSquaresLoss` or `SqrtLeastSquaresLoss` and one `Scaled` of it. `z`, `r` and `d` must have the axes of `Y` (another length, or an offset
view of the same length, is a `DimensionMismatch`). `Y` and `X` must be one-based: an offset-axes `Y` or `X` is an
`ArgumentError` when the loss is constructed. For `value_and_gradient_from_predictor!` an output `g` with offset axes is
an `ArgumentError` before anything is written.

**The scale as stored and as computed.** The accessor returns the factor ``\alpha`` as stored, with its type.
The oracles compute with it as supplied, not rounded to `T` first: a positive finite `Float32` or `Float64`
(a `Float64` factor with `Float32` data included), or an integer `α` with `Float64(α) == α`, which is
converted to `Float64` once. An integer that `Float64` does not hold (`2^53 + 1`), a `Float16`,
`BigFloat` or `Rational`, and `Inf` are an `ArgumentError`, raised before anything is written; the `Scaled`
constructor is not changed. For an integer the factor used is its `Float64` value.

**Value and rounding.** The result has type `T`. For a vector response, the sum of squares ``s`` is formed in `T`, in one vectorizing
loop for up to ``2^{17}`` entries and by Julia's pairwise summation for longer vectors, ``\alpha s/2`` in the
promoted type of ``\alpha`` and `T`, and the product is rounded to `T` at the result (a subnormal result can round
twice). For a matrix response, the predictor value uses a pairwise sum of squared
differences. The fused oracle sums the stored residual squares, with the same
length cutoff as the vector implementation, applied to `nK`. Where the squares
overflow or underflow, where the difference ``z-Y`` overflows, and where the ordinary
value ``\alpha s/2`` is above ``\texttt{floatmax}(T)/2``, the value is recomputed from the squares in units of a power of two with
``\alpha`` folded in, summed compensated, the computation that `SqrNormL2` uses for its exceptional values. The
last case keeps the error of the ordinary sum, which depends on the number of entries, from deciding between a finite value and
`Inf`. The recomputation gives a finite value where the value is representable with margin (see the range
statement below). Against 256-bit references the tests require
``(m+8)\,\varepsilon_T|v|`` accuracy for finite data and a value that is representable with that margin, plus
two units of the smallest subnormal where ``|v|`` is below `floatmin(T)`. Here ``m``
is the number of response entries.

**Range.** Write ``u=\texttt{floatmax}(T)-\texttt{prevfloat}(\texttt{floatmax}(T))``, so that ``\texttt{floatmax}(T)+u/2``
is the round-to-nearest overflow threshold, and ``w=(m+8)\varepsilon_T`` for the relative error bound above. For finite
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
`value_from_predictor` can recover a representable value. For ``p>0`` the gradient of
the affected response column is then non-finite (with the BLAS of the tests, also
for a zero column of ``X``, where `0*Inf` is `NaN`). For ``p=0`` the gradient is empty.

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

## Square-root least squares is not smooth

`SqrtLeastSquaresLoss` is described on this page because it stores the data of `LeastSquaresLoss` and is evaluated
by the same value oracles; it is not a smooth function, and it has no gradient and no proximal operator in this package.

```math
f(\beta)=\|Y-X\beta\|_2,\qquad Y\in\mathbb{R}^n,\ X\in\mathbb{R}^{n\times p}.
```

The loss is convex. For ``X\ne 0`` it is not differentiable where ``Y=X\beta`` (for ``X=0`` it is constant in ``\beta``). It has the unnormalized convention: there is no
``1/\sqrt n`` in the type. The normalized square-root lasso objective and its relation to the unnormalized one are

```math
\frac{1}{\sqrt n}\|Y-X\beta\|_2+\lambda_{\text{normalized}}\|\beta\|_1,\qquad
\lambda_{\text{unnormalized}}=\sqrt n\,\lambda_{\text{normalized}},
```

so that the data term of the normalized objective is `Scaled(f, inv(sqrt(n)))` and the penalty is `NormL1(λ)` with
the level of the objective in use. ProximalBase defines no sum of functions: an algorithm adds the two terms itself.

`issmooth(f)` and `isproximable(f)` are `false` and `isconvex(f)` is `true`. `gradient!`, `value_and_gradient!`,
`prox!`, `value_and_prox!`, `prox_block`, `kkt_violation` and `restrict` are a `MethodError`, and `lipschitz(f)` is an
`ArgumentError`. There is no `ProximalBase.value_and_gradient_from_predictor!` for it: a subgradient at the point
of zero residual is a separate decision that this package does not make. `Scaled(f, 0)` is the zero function for finite data.

**Construction.** `SqrtLeastSquaresLoss(Y, X)` takes a vector `Y` and a matrix `X` with real entries and
`size(X, 1) == length(Y)` (a `DimensionMismatch` otherwise), both with axes that start at 1. A matrix `Y` is a
`MethodError`. `T = float(promote_type(eltype(Y), eltype(X)))` is the type of the scratch that `f(β)` uses.

**Values.** `f(β)` forms ``X\beta`` in the scratch of the loss and takes the norm of ``X\beta-Y``. An entry of
``X\beta`` that overflows is not formed again, unlike `LeastSquaresLoss`: the value is then the norm of the computed
``X\beta-Y``, typically `Inf` or `NaN`. The value oracles, with ``\alpha`` the factor of one `Scaled` wrapper and
``\alpha=1`` for the loss, are

```math
v=\alpha\,\|z-Y\|_2\ \text{ for a predictor } z=X\beta,\qquad
v=\alpha\,\|r\|_2\ \text{ for a residual } r=\pm(Y-X\beta).
```

`ProximalBase.value_from_predictor(f, z)` and `ProximalBase.value_from_residual(f, r)` read no entry of ``X``, neither
read nor write the scratch of the loss, do not modify `z` or `r`, and allocate nothing on their ordinary path for a
raw loss or a `Scaled` one with a `Float32`, `Float64` or `Int` factor. The domain, the axes check, the factors that
are accepted and the result type are those of the least-squares oracles above: a vector `Y`, strided `Float32` or
`Float64` arrays of one type `T`, one `Scaled` at most, and an `ArgumentError` for another factor. The factor is
checked before the sum on every call here, so that `Scaled(f, Inf)` is an `ArgumentError` also at an exact fit, where
`Inf * 0` would be `NaN`. `ProximalBase.leastsquares_data(f)` returns `(; Y, X, α)` as for the least-squares loss; the
accessor is shared, and the two losses are told apart by their types.

**Accuracy and range.** The value is never formed as ``\sqrt{2\,v_{\text{LS}}}`` from the least-squares value
``v_{\text{LS}}``, whose squares overflow where the norm does not. For strided `Float32` or `Float64` vectors of one
element type, which covers both oracles, the sum of squares ``s`` is formed in `T` in one vectorizing loop up to
``2^{17}`` entries and by pairwise summation above. The other arguments of `f(β)`, such as mixed element types or
non-strided arrays, take generic sums in one loop at every length. The product ``\alpha\sqrt s`` is kept when ``s`` is not below ``n\,\texttt{floatmin}(T)/\varepsilon_T`` and the
product is below ``\texttt{floatmax}(T)/2`` (a `NaN` or infinite ``s`` is not kept). Otherwise, and for every sum that is not finite, the norm is recomputed
from the squares in units of a power of two, with ``\alpha`` folded in and the squares summed compensated, in
at least `Float64`. For finite entries that `T` represents exactly, ``n\le 2^{40}`` of them, with ``(n+3)\varepsilon_T\le 1/16`` or
strided `Float32` or `Float64` vectors of one element type, and ``v`` the exact value, the result is within ``(n/2+4)\,\varepsilon_T|v|`` of ``v``, plus two units of
the smallest subnormal where ``|v|<\texttt{floatmin}(T)``; with ``u`` and ``w=(n/2+4)\varepsilon_T`` as in the
range statement above, the result is finite where ``|v|(1+w)\le\texttt{floatmax}(T)``, is `Inf` above
``(\texttt{floatmax}(T)+u/2)(1+w)``, and between the two is `Inf` or finite within the bound (`floatmax(T)` is possible
within ``w`` of the threshold, below it as well as above it). The proof is in the comments of `src/functions/sqrt_leastsquares.jl`. `f(β)` returns its value converted to `float(eltype(β))`, so the bound holds for `eltype(β) == T`, and a narrower `β` adds one rounding to its type. An entry that `T` does not represent exactly (an `Int` `Y` above `maxintfloat(T)`, a `Rational` `Y`) is rounded to `T` in the sum, outside the bound. An exact fit gives
`+0`; a `NaN` entry gives `NaN`; an entry that is `±Inf` gives `Inf`, unless it meets an `Inf` of the same sign in `z`
and `Y`, where the difference is `NaN`.

**The wrapper.** `Scaled(f, α)(β)` is the generic wrapper. It receives `f(β)` already rounded to `float(eltype(β))`,
forms the product with ``\alpha`` from that rounded value and returns it converted to `float(eltype(β))`. A `β` narrower
than `T` therefore rounds the inner value to its own type first: with `f = SqrtLeastSquaresLoss([0.0], fill(0.25, 1, 1))`,
`β = Float32[nextfloat(0f0)]` and `α = 4.0`, `f(β)` and `Scaled(f, α)(β)` are `0.0f0`, while the oracle at `X * β` is about
`1.4e-45`. For `eltype(β) == T`, with a `Float64` factor on `Float32` data the result can differ from the oracle in the last bit, and where
the oracle recovers a value that the wrapper's product loses it can differ on the exceptional range (issue #609, as for
the least-squares loss). `Scaled(f, Inf)(β)` is `Inf * f(β)`, which is `NaN` at an exact fit; the oracles refuse the
factor `Inf`.

```@example library_sqrt_ls
using ProximalBase, LinearAlgebra
Y = [3.0, 0.0]
X = Matrix(1.0I, 2, 2)
f = SqrtLeastSquaresLoss(Y, X)
β = [0.0, 4.0]
@assert f(β) == 5.0
@assert ProximalBase.value_from_predictor(f, X * β) == 5.0
@assert ProximalBase.value_from_residual(f, [3.0, -4.0]) == 5.0 == ProximalBase.value_from_residual(f, [-3.0, 4.0])
n = length(Y)
s = Scaled(f, inv(sqrt(n)))                 # the data term of the normalized objective
@assert ProximalBase.value_from_predictor(s, X * β) ≈ 5 / sqrt(2)
@assert isconvex(f) && !issmooth(f) && !isproximable(f)
ProximalBase.value_from_residual(Scaled(f, 0.5), [3.0, -4.0])
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
It has scalar `prox_block`, the three forms of `kkt_violation` and `restrict`:
the latter slices the center, or returns the same uncentered object. Slicing a
center omits the constant ``(λ/2)\sum_{j\notin S}y_j^2`` from dropped coordinates
fixed at zero. When that omitted contribution is finite, restore it when comparing full
objective values; the restricted minimizers, gradient and prox are unaffected. Restriction does
not check whether the discarded contribution is finite; a non-finite constant
cannot be ignored when reasoning about the original extended-valued objective.
`lipschitz(f)` is `float(λ)`. Array-step prox is not implemented.

For `SqrNormL2`, the three-argument screen `kkt_violation(f,v,i)` takes ``x_i=0`` and does not use
the gradient oracle: it is ``\lvert v_i\rvert``, or ``\lvert v_i-\lambda y_i\rvert`` with a center,
so `kkt_violation(SqrNormL2(Inf), [1.0], 1)` is `1.0`. The four-argument form `kkt_violation(f,v,x,k)` and the
full-point form `kkt_violation(f,v,x)` use the entry ``d_i`` of ``\nabla f(x)`` as the ridge-gradient
helpers form it, in the working type of the call (``\lambda x_i``, or ``\lambda(x_i-y_i)`` with a
center). The score of entry ``i`` is ``\lvert v_i+d_i\rvert``, nonnegative at a zero coordinate as well, and
the full-point value is ``\lVert v+\nabla f(x)\rVert_\infty`` for finite `v` and `x` when `gradient(f,x)` is
computed in the same precision. See the [SqrNormL2 endpoint clauses](@ref spec-sqrnorml2-endpoints).

See the [SqrNormL2 range clauses](@ref spec-sqrnorml2-range).

`iszeropreserving(f)` is true only when the center is omitted and the level is
finite. Supplying an all-zero center still follows the centered route and
reports false. See the [SqrNormL2 endpoint clauses](@ref spec-sqrnorml2-endpoints).

```@example library_ridge
using ProximalBase
f = SqrNormL2(2.0, [1.0, 0])
x = [3.0, -1]
@assert f(x) == 5 && gradient(f,x) == [4, -2]
@assert prox(f,x,0.5) == [2, -0.5]
@assert prox_block(f, x[2], 2, 0.5) == -0.5
@assert !iszeropreserving(f) && iszeropreserving(SqrNormL2(2.0))
v = [-4.0, 1.0]            # ∇f(x) = [4, -2]
@assert kkt_violation(f, v, x, 1) ≈ 0.0 && kkt_violation(f, v, x, 2) ≈ 1.0
@assert kkt_violation(f, v, x) ≈ maximum(abs, v + gradient(f, x))
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
`value_entry` method. See [smooth stationarity scores](@ref smooth-kkt) for the full-point KKT form.

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

## [Stationarity scores for least squares and quadratics](@id smooth-kkt)

`kkt_violation(f, v, x)` returns `‖v + ∇f(x)‖∞` for `LeastSquaresLoss` and
`QuadraticFunction`. Supply `v`, typically the gradient of the other objective
term, and the point `x` where both gradients are evaluated. The score is zero
when the computed residual vanishes, including at an admissible empty point.
An indefinite quadratic can attain first-order stationarity at a saddle point
or maximum.

Use real, one-based inputs of the same shape. The quadratic takes vectors of
length `size(A, 2)`. For least squares, use a vector of length `size(X, 2)`
with a vector response, or a coefficient matrix of size
`(size(X, 2), size(Y, 2))` with a matrix response. The supported array products
are those of `gradient!`, including supported views, sparse or structured data,
and iterate representations. The coordinate screen and point-aware coordinate
and group forms throw `ArgumentError` and name the full-point form. A
`SymmetricSparseIterate` point also throws `ArgumentError`.

`gradient!` stores its result in the function's floating type `T`. The residual
sum and maximum use `float(promote_type(eltype(v), eltype(x), T))`. Wider
inputs can widen the residual arithmetic. The gradient still uses the rounding
of the existing matrix products and destination. For homogeneous `Float32`
and `Float64` inputs and function data, finite ordinary scores agree bit for bit
with `maximum(abs, v .+ gradient(f, x))` at a fixed BLAS thread count in the
same process. A `NaN` in either input gives `NaN`. Otherwise an infinite input
entry gives `+Inf`. With finite inputs, ordinary arithmetic in the gradient
and residual can still overflow, underflow or produce `NaN`.

The call reads `v` and `x` without modifying them and uses `Base.mightalias`
on their storage to detect overlap with the scratch it writes. A quadratic uses its existing
scratch for the gradient and makes one matrix product. A least-squares loss
makes two matrix products and uses both its existing residual scratch and a
dense coefficient-shaped output. The first ordinary call that needs this
output allocates it, and later calls reuse it while its type and shape match.
Homogeneous dense `Float32` and `Float64` inputs need no other scratch.
Synchronize scratch-using calls on the same function object.

If you already have an independent gradient destination `d` of the function's
type and the inputs share that type, you can form the same finite ordinary
score without using or allocating a coefficient buffer inside the function.
The least-squares gradient in this recipe still uses its residual scratch.

```julia
gradient!(d, f, x)
d .+= v
kkt_violation(ZeroFunction(), d, x)
```

For a finite positive factor `α`, `Scaled` applies its
[usual rule](@ref scaled-kkt) to the supplied gradient `v`. An infinite
factor or a retained chain whose factors do not fold throws `ArgumentError`.
Scaling a function with finite data by zero through the public constructor yields `ZeroFunction()`.

## [Zero function](@id Zero-function)

The no-argument constructor `f = ZeroFunction()` represents
``f(x)=0`` for any supported real array `x`. Its gradient is
``\nabla f(x)=0``, and its proximal operator is the identity,
``\operatorname{prox}_{\gamma f}(x)=x``. It has all four traits true and is zero preserving.
It needs no parameters or dimensions. `lipschitz(f)==0`; `restrict(f,S)`
returns `f`; scalar `prox_block(f,v,i,γ)` returns `v`; its zero-coordinate
screen `kkt_violation(f,v,i)` is ``|v_i|``, and so is the four-argument score at a
zero or a nonzero coordinate; the full-point value `kkt_violation(f,v,x)` is
``\lVert v\rVert_\infty``. It has no `value_entry` method.

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
`prox_block(f,v,i,γ)=v-γ*c[i]`, the zero-coordinate screen
``|v_i+c_i|`` of `kkt_violation(f,v,i)`, the same score from the four-argument form
at zero and nonzero coordinates, the full-point value ``\lVert v+c\rVert_\infty``, and `restrict(f,S)=LinearFunction(c[S],d)`.
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
are recorded in the [implementation notes](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/function-docs-history-921.md#linearfunction-sparse-value-measurements). The constant `d` is added
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
@assert kkt_violation(f, [-1.0, 3.0], x, 2) == 1 && kkt_violation(f, [-1.0, 3.0], x) == 1
prox(f,x,0.5)
```

## [Logistic regression](@id Logistic-regression-loss)

Construct `f = LogisticLoss(y, X; weights=nothing, offset=nothing)` for an
unnormalized logistic objective. For an ``n\times p`` design and a length-``p``
coefficient vector,

```math
\eta = X\beta + \mathrm{offset},\qquad
f(\beta)=\sum_{i=1}^n w_i[\log(1+e^{\eta_i})-y_i\eta_i],\qquad
\nabla f(\beta)=X^\top d,\quad d_i=w_i[(1+e^{-\eta_i})^{-1}-y_i].
```

Use one-based strided vectors and a strided matrix or `SparseMatrixCSC` design,
all of the same element type, `Float32` or `Float64`. Responses are finite binary
values or fractions in `[0, 1]`, including at zero weight. The archived logistic
constructor did not enforce this range. Supply finite nonnegative weights and
finite offsets as length-``n`` vectors. The defaults are unit weights and zero
offset. Include a column of ones to estimate an intercept.

The loss is convex and smooth. Its traits `isproximable` and `isseparable`
return false. Use `f(β)`, `gradient(f, β)` or `value_and_gradient(f, β)`.
Their mutating gradient counterparts take a caller-owned strided length-``p``
output of the computation type. `gradient!(g, f, β)` returns `g`.
`value_and_gradient!(g, f, β)` returns the scalar value. Use ordinary
`Scaled(f, α)` or `α*f` for normalization or rescaling in coefficient space.

If the caller already holds **the complete linear predictor**
``\eta=X\beta+\mathrm{offset}``, pass it to
`ProximalBase.value_from_predictor(f, η)` or
`ProximalBase.value_and_gradient_from_predictor!(g, d, f, η)`. Both return
the objective. The fused call also writes the weighted observation derivative
to `d` and the coefficient gradient to `g`, using one product with `X'`.
The predictor and both outputs are one-based strided vectors of the data type.
These operations take an unwrapped loss and leave the supplied predictor intact.

The constructor borrows model arrays. Same-shape edits between calls must
preserve the validated data domain. Reconstruct the loss for structural changes
and refresh a caller-held predictor when its defining inputs change.
Ordinary calls share private scratch and require synchronization across tasks.
Predictor calls can share a loss across tasks when model data stay read-only
and the outputs are independent. Before writes, the oracles reject nonempty
output overlaps reported by `Base.mightalias` with the input, other output,
model arrays (including CSC values and indices) or scratch. An input predictor or
coefficient vector must also be independent of the scratch.

The scalar formulas preserve the small tail for saturated responses.
For a response of one half and `0 < abs(η) < 1/1024`, the derivative uses a
centered expression to retain the slope through weighting. A
zero-weight row contributes zero to the value and observation derivative, even
at a nonfinite predictor. At active rows, NaN propagates. Infinite predictors
give the logistic limit: zero loss for a matching binary response, and infinity
otherwise. Standard matrix products may still propagate nonfinite design
entries, including through a zero derivative. For large weights the scalar
loop retains a weighted tail when the unweighted exponential becomes subnormal
or zero. The matrix products, final value sum and outer scaling use ordinary
floating-point arithmetic with their usual rounding and range limits.
The loss supplies first-order oracles. The historical generic scalar-loss,
coordinate and second-order interfaces remain deferred.

```@example logistic_regression
using ProximalBase
X = [1.0 0.0; 1.0 2.0; 1.0 -1.0]
y = [0.0, 1.0, 0.25]
offset = [0.1, -0.2, 0.0]
f = LogisticLoss(y, X; weights=[1.0, 2.0, 1.0], offset)
β = [0.5, 0.25]
η = X * β + offset
g, d = similar(β), similar(y)
v = ProximalBase.value_and_gradient_from_predictor!(g, d, f, η)
@assert v ≈ f(β)
@assert g ≈ gradient(f, β)
(value=v, gradient=g)
```
