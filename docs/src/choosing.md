# [Choose a function or operation](@id choosing)

Start with the quantity you want to compute. ProximalBase provides function
objects and storage for your algorithm; a solver supplies the iteration,
step selection and stopping rule.

## Choose a function

| Your model term | Use | What to check |
|:--|:--|:--|
| Squared regression error, `sum(abs2, X*β - y)/2` | [`LeastSquaresLoss(y, X)`](@ref LeastSquaresLoss) | Response first; use `Scaled(LeastSquaresLoss(y, X), 1/size(X, 1))` for the mean loss |
| Square-root regression error, `norm(y - X*β)` | [`SqrtLeastSquaresLoss(y, X)`](@ref SqrtLeastSquaresLoss) | Convex but not smooth; value and supplied-predictor/residual oracles, **no gradient or prox** in this core |
| Squared distance or ridge penalty, `λ*sum(abs2, x - y)/2` | [`SqrNormL2(λ, y)`](@ref SqrNormL2) | Omit the center for ridge at zero |
| General quadratic, `x' * A * x / 2 + b' * x + c` | [`QuadraticFunction(A, b, c)`](@ref QuadraticFunction) | `A` must be symmetric (not checked); indefinite matrices restrict proximal steps |
| Lasso or weighted absolute values | [`NormL1(λ, w)`](@ref NormL1) | Weights are optional; scalar prox is soft thresholding |
| Lasso plus ridge | [`ElasticNet(λ1, λ2, w)`](@ref ElasticNet) | Weights affect only the ℓ₁ term |
| One Euclidean norm | [`NormL2(λ)`](@ref NormL2) | This is a norm, not its square; a matrix uses its Frobenius norm |
| Maximum absolute entry (ℓ∞ norm) | [`NormLinf(λ)`](@ref NormLinf) | Nonseparable; prox clips entries using an ℓ₁-ball threshold; matrix value is the largest entry magnitude |
| Sum of norms over disjoint groups | [`GroupNormL2(λ, groups, w)`](@ref GroupNormL2) | Unlisted indices are unpenalized; compact vectors use each group's index order; default weights are √group size |
| Affine or zero term | [`LinearFunction(c, d)`](@ref LinearFunction), [`ZeroFunction()`](@ref ZeroFunction) | Choose the coefficient shape for your array argument |
| Nonnegative multiple of an existing function | [`Scaled(f, α)`](@ref Scaled) or `α*f` | Supported operations depend on the inner function |

The [function library](function_library.md) gives formulas and detailed guides.
To combine a smooth loss `f` and a penalty `g`, evaluate `f(x) + g(x)` and use
their operations separately. The minimal core has no general `f + g` object.

## Choose an operation

| You need | Call | Result |
|:--|:--|:--|
| Function value | `f(x)` | Scalar value at `x` |
| New gradient | [`gradient(f, x)`](@ref gradient) | New dense gradient array |
| Value and gradient | [`value_and_gradient(f, x)`](@ref value_and_gradient) | `(value, gradient)` |
| Gradient Lipschitz bound, when available | [`lipschitz(f)`](@ref lipschitz) | Bound for step selection; check for zero before dividing |
| Reuse a gradient buffer | [`value_and_gradient!(out, f, x)`](@ref value_and_gradient!) | Writes the gradient; returns the value at **input** `x` |
| New proximal point | [`prox(g, x, γ)`](@ref prox) | New proximal output |
| Reuse a proximal buffer and evaluate it | [`value_and_prox!(out, g, x, γ)`](@ref value_and_prox!) | Writes the proximal point; returns the value at **stored output** `out` |
| One scalar or compact-group proximal point | [`prox_block`](@ref), [`prox_block!`](@ref) | Numerical result; commit it to iterate storage separately |
| Test whether a zero coordinate should move | [`kkt_violation`](@ref) | Signed screen; a positive result violates the zero condition |
| Refresh caches after changing data in place | [`reset_cache!(f)`](@ref reset_cache!) | After changes to `QuadraticFunction.A` or `LeastSquaresLoss` data `X`/`Y`; no reset for quadratic `b` alone; `SqrtLeastSquaresLoss` has no cache |

Use positive finite steps for ordinary proximal calls. Match shapes and use
independent gradient buffers. A proximal call supports exact in-place input
and output; other overlapping views are unsupported. Read [operations](operations.md)
for storage-specific methods and [array and precision conventions](numerical_usage.md) for
precision, ownership and failure behavior.

## Try the basic calls

```jldoctest
julia> using ProximalBase

julia> f = SqrNormL2(1.0); g = NormL1(0.5); x = [2.0, -1.0];

julia> (f(x), g(x), Tuple(gradient(f, x)))
(2.5, 1.5, (2.0, -1.0))

julia> Tuple(prox(g, x, 1.0))
(1.5, -0.5)

julia> out = similar(x); value = value_and_prox!(out, g, x, 1.0);

julia> (value, Tuple(out))
(1.0, (1.5, -0.5))
```

The [getting-started tutorial](getting_started.md) combines these operations
into a lasso iteration. If you already hold a predictor or residual, the
[supplied-predictor value oracles](smooth.md#Least-squares-at-a-predictor-or-residual-that-the-caller-holds)
serve both regression losses without recomputing `X*β`; the supplied-predictor
gradient hook applies only to `LeastSquaresLoss`. Their buffer and consistency
preconditions still apply.

## Choose storage when you need it

Start with ordinary arrays. Use [`SparseIterate`](@ref) for a coordinate working
set, [`BlockIterate`](@ref) for an ordered partition of a vector, or
[`SymmetricSparseIterate`](@ref) for independent symmetric matrix variables.
The symmetric owner is not an array: explicitly choose [`coordinates`](@ref)
or [`matrixview`](@ref). The [iterate guide](iterates.md) explains active zeros,
copies and live views.

This manual describes the current minimal 0.6 core. The
[migration guide](migration.md) distinguishes it from archived full-package
features. Detailed exceptional inputs belong in the [numerical contract](numerics.md).
