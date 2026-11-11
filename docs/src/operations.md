# Operations on functions

Use this catalog to select the mathematical operation your algorithm needs.
All mutating mathematical oracles use **destination first**. An exclamation
mark means mutation, not a guarantee of zero allocations.

## Whole-function calls

| Call | Writes | Returns |
|:--|:--|:--|
| `f(x)` | May reuse function-owned scratch | Scalar value at x |
| `gradient(f,x)` | New dense gradient | Gradient array |
| `gradient!(out,f,x)` | out | out |
| `value_and_gradient(f,x)` | New dense gradient | `(f(x), gradient)` |
| `value_and_gradient!(out,f,x)` | out | Scalar f(x) |
| `prox(f,x,γ=1)` | New proximal point | New array/iterate of floating element type |
| `prox!(out,f,x,γ=1)` | out | out |
| `value_and_prox!(out,f,x,γ=1)` | out | Scalar f(out), at the stored result |

There is no allocating `value_and_prox` binding. Gradients may not overwrite
`x`; exact `out === x` is supported for proximal maps. Other overlapping
input/output views are unsupported. The full rules are in
[the interface](interface.md) and [array conventions](numerical_usage.md).

## Entry, block and working-set operations

| Call | Meaning and supported scope |
|:--|:--|
| `value_entry(f,v,i)` | Scalar term fᵢ(v); NormL1 and user SeparableFunction implementations |
| `prox_block(f,v,i,γ)` | Scalar proximal point at an already computed trial v; NormL1, SqrNormL2, LinearFunction, ZeroFunction and supported Scaled leaves |
| `prox_block!(out,f,u,k,γ)` | User extension hook on compact vectors, returning out; no retained built-in group kernel |
| `prox_block(f,u::AbstractVector,k,γ)` | Allocates a plain floating Vector and delegates to the block hook |
| `kkt_violation(f,grad,i)` | Signed zero-coordinate screen; retained scalar methods above and their Scaled wrappers |
| `kkt_violation(f,grad,x,k)` | Screen using x's coordinate space: scalar entry, symmetric independent coordinate, or maximum over a BlockIterate group |
| `restrict(f,S)` | Slice supported parameters to integer indices S; no general restriction fallback |

The block hook changes a numerical output only. A caller subsequently writes
`x[i]`, `coordinates(x)[k]` or `copyto!(blockview(x,k),out)` and updates its
residual/cache. See [coordinate oracles](iterate_oracles.md).

`restrict` supports ZeroFunction, LinearFunction, SqrNormL2, QuadraticFunction,
NormL1, vector-response LeastSquaresLoss, and Scaled when its leaf supports it.
It does not encode a full subproblem offset at an arbitrary current point.
For a quadratic it selects the principal submatrix and matching linear term;
for a vector-response loss it selects columns and retains the response.

## Bounds and lifecycle

| Call | Meaning |
|:--|:--|
| `lipschitz(f)` | Global gradient Lipschitz bound for retained smooth families; unknown/non-smooth default throws ArgumentError |
| `opnorm2(X)` | Lanczos estimate of squared spectral norm, not a certified upper bound |
| `reset_cache!(f)` | Invalidate lazy decomposition after same-shape data changes; returns f, forwards through Scaled |

## Extension validation hooks

These names are public but not exported; use the `ProximalBase.` prefix.

| Hook | Return / responsibility |
|:--|:--|
| `check_parameters(f,x)` | Default nothing; custom validation throws on unsuitable shapes before writes |
| `entry_parameters(f)` | Tuple of `Symbol => parameter` pairs; custom SeparableFunction default is undeclared (`nothing`) |
| `check_symmetric_parameters(f,x)` | true on success; checks all declared mirrored pairs on an owner, including Scaled leaves |

The [extension guide](extending.md) explains the generic defaults. Scalar loss
hooks `deriv`, `deriv2`, `curvature` and `value_and_deriv` are listed with
[`ScalarLoss`](@ref ProximalBase.ScalarLoss) in the
[function reference](api/functions.md).
