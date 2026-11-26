# Losses (prototype: `LeastSquaresLoss` entry)

*In the implementation this entry opens the new `losses.md`, followed by
`LogisticLoss`, `PoissonLoss`, `SqrtLeastSquaresLoss` and `LogDetLoss`.*

## [Least squares: `LeastSquaresLoss`](@id entry-leastsquaresloss)

The data-fit term of linear regression, ridge regression and the lasso.

```math
f(\beta)=\tfrac12\,\|y-X\beta\|_2^2,\qquad
\nabla f(\beta)=X^\top(X\beta-y).
```

Note the factor ``\tfrac12`` and the absence of ``1/n``. For the mean loss
``\tfrac1{2n}\|y-X\beta\|_2^2`` use `(1/n) * LeastSquaresLoss(y, X)`. With an
``n\times k`` response ``Y``, the coefficients ``B`` are ``p\times k`` and the norm is
Frobenius.

| Constructor | Meaning |
|:--|:--|
| `LeastSquaresLoss(y, X)` | response first, then the ``n\times p`` design matrix |

The rows of `y` and `X` must agree and their entries must be real; this is
checked. `X` may be sparse. Both arrays are kept by reference (see
[Good to know](@ref ls-good-to-know)).

```@example entry-ls
using ProximalBase

X = [1.0 2.0; 3.0 1.0; 0.0 2.0]
y = [1.0, 2.0, 3.0]
f = LeastSquaresLoss(y, X)
β = [0.5, -0.5]
f(β), gradient(f, β)
```

```@example entry-ls
f_mean = (1 / size(X, 1)) * f       # 1/(2n) ‖y - Xβ‖²
f_mean(β), lipschitz(f_mean)
```

### Operations

| Operation | Call | Supported |
|:--|:--|:--|
| value | `f(β)` | ✓ |
| gradient | `gradient(f, β)`, `gradient!(out, f, β)` | ✓ |
| value and gradient | `value_and_gradient(f, β)`, `value_and_gradient!(out, f, β)` | ✓ the value is at the input ``\beta`` |
| step size | `lipschitz(f)` | ✓ ``\|X\|_2^2`` for a dense matrix; an upper bound for other matrices |
| proximal step | `prox(f, β, γ)`, `prox!(out, f, β, γ)` | ✓ scalar step |
| optimality score | `kkt_violation(f, v, β)` | ✓ whole point only |
| restriction | `restrict(f, S)` | ✓ vector response; coordinates outside `S` fixed at zero |
| one coordinate | `prox_block` | — |
| from a held predictor | `ProximalBase.value_from_predictor(f, Xβ)` and related | ✓ [details](@ref Least-squares-at-a-predictor-or-residual-that-the-caller-holds) |

The proximal step solves ``(X^\top X+\gamma^{-1}I)z=X^\top y+\gamma^{-1}v``. The
first call computes a thin SVD of `X` and keeps it, so later calls with any step
are cheap, also when ``p>n``.

### Inputs and storage

✓ vectors (and ``p\times k`` matrices for a matrix response) and strided views ·
✓ `SparseIterate` · ✓ `BlockIterate` ·
— `SymmetricSparseIterate`.
Element types: `Float64` or `Float32`, the same type for data, parameters, step and
output ([precision rules](@ref numerical-usage)).

### [Good to know](@id ls-good-to-know)

- After changing entries of `X` or `y` in place, call `reset_cache!(f)` before the
  next proximal step; values and gradients read the data directly. Construct a new
  loss after changing dimensions.
- The loss keeps scratch arrays. Do not call it from several tasks at once
  without synchronization.
- `gradient!(out, f, β)` needs an `out` distinct from `β`.
- The proximal step is meant for finite data; it does not repair `NaN` or `Inf` in `X`.

**See also:** [`SqrtLeastSquaresLoss`](@ref) (``\|y-X\beta\|_2``, for the square-root
lasso), [`QuadraticFunction`](@ref) (a general quadratic),
[`SqrNormL2`](@ref) (``\tfrac\lambda2\|x-c\|^2``, no design matrix) ·
[guarantees](@ref spec-leastsquares) · [API](@ref LeastSquaresLoss)
