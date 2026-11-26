# Losses (prototype: `LeastSquaresLoss` entry)

*In the implementation this entry opens `smooth.md`, retitled "Losses", followed by
`LogisticLoss`, `PoissonLoss`, `SqrtLeastSquaresLoss` and `LogDetLoss`. The loss
sections keep their current anchors.*

## [Least squares: `LeastSquaresLoss`](@id entry-leastsquaresloss)

The data-fit term of linear regression, ridge regression and the lasso.

```math
f(\beta)=\tfrac12\,\|y-X\beta\|_2^2,\qquad
\nabla f(\beta)=X^\top(X\beta-y).
```

Note the factor ``\tfrac12`` and the absence of ``1/n``. For the mean loss
``\tfrac1{2n}\|y-X\beta\|_2^2`` with ``n>0`` observations, use `(1/n) * LeastSquaresLoss(y, X)`. With an
``n\times k`` response ``Y``, the coefficients ``B`` are ``p\times k`` and the loss is
``\tfrac12\|Y-XB\|_F^2``.

| Constructor | Meaning |
|:--|:--|
| `LeastSquaresLoss(y, X)` | response first, then the ``n\times p`` design matrix |

`y` and `X` must be real, one-based arrays with the same number of rows; this is
checked. `X` may be a sparse matrix. Both arrays are kept by reference (see
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
| step size | `lipschitz(f)` | ✓ ``\|X\|_2^2`` evaluated in floating point for a dense ``X``, a bound on it for other matrices; rounding can make it slightly smaller than the exact value |
| proximal step | `prox(f, β, γ)`, `prox!(out, f, β, γ)` | ✓ scalar step |
| optimality score | `kkt_violation(f, v, β)` | ✓ whole point only |
| restriction | `restrict(f, S)` | ✓ vector response; coordinates outside `S` fixed at zero |
| one coordinate | `prox_block` | — |
| from a held predictor | `ProximalBase.value_from_predictor(f, Xβ)` and related calls | ✓ [details](@ref Least-squares-at-a-predictor-or-residual-that-the-caller-holds) |

For finite data and a finite step ``\gamma>0``, the proximal point is the solution of
``(X^\top X+\gamma^{-1}I)\,z=X^\top y+\gamma^{-1}v``. ProximalBase computes it from a thin
SVD of a nonempty ``X`` (a sparse ``X`` is converted to a dense matrix for it), which
later calls reuse for other steps.

### Inputs and storage

✓ vectors and strided views ·
◐ matrices, with an ``n\times k`` response ·
✓ `SparseIterate` (a vector, or a matrix with an ``n\times k`` response) ·
✓ `BlockIterate` ·
— `SymmetricSparseIterate`.
A sparse *design* `X` and sparse *coefficients* (`SparseIterate`) are independent choices.
Element types: `Float64` or `Float32`, the same type for data, parameters, step and
output ([precision rules](@ref numerical-usage)).

### [Good to know](@id ls-good-to-know)

- After changing entries of `X` or `y` in place, call `reset_cache!(f)` before the
  next proximal step; values and gradients read the data directly. Construct a new
  loss after changing dimensions.
- Values, gradients, proximal steps and the whole-point score use scratch arrays
  stored in the loss: synchronize concurrent calls on one loss. The held-predictor
  calls write no field of the loss.
- `gradient!(out, f, β)` needs an `out` that does not share memory with `β`.
- The proximal step is meant for finite data and positive finite steps; it does not
  repair `NaN` or `Inf` in `X`. Range limits of products and of the SVD are on the
  [guarantee page](@ref spec-leastsquares).

**See also:** [`SqrtLeastSquaresLoss`](@ref) (``\|y-X\beta\|_2``, for the square-root
lasso), [`QuadraticFunction`](@ref) (a general quadratic),
[`SqrNormL2`](@ref) (``\tfrac\lambda2\|x-c\|^2``, no design matrix) ·
[guarantees](@ref spec-leastsquares) · [API](@ref LeastSquaresLoss)
