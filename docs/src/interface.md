# The interface

We describe the function interface and the mathematics of full proximal
steps. For updates of individual entries or groups, see
[Coordinate descent](@ref). All mutating operations take their output first,
as in `gradient!(grad, f, x)` and `prox!(out, g, x, γ)`.

## Proximal operators

The proximal operator of a function ``g`` with step ``\gamma > 0`` is

```math
\operatorname{prox}_{\gamma g}(x) = \operatorname*{argmin}_z \; g(z) + \frac{1}{2\gamma}\|z - x\|^2 .
```

It balances decreasing ``g`` against staying near ``x``, and ``\gamma`` sets the
balance. Two cases explain the name and its uses.

- **Projection.** When ``g`` is the indicator function of a set ``C`` (``0`` on
  ``C`` and ``+\infty`` outside), the proximal operator is the Euclidean
  projection onto ``C``, for every ``\gamma``.
- **Shrinkage.** When ``g = \lambda\|\cdot\|_1``, the problem separates over the
  coordinates. Each is solved by soft thresholding,
  ``\operatorname{sign}(x_i)\max(|x_i| - \gamma\lambda, 0)``, which sets small
  coordinates exactly to zero.

For a convex ``g`` (closed and proper) the minimizer is unique, and the map is
*firmly nonexpansive*:

```math
\|\operatorname{prox}_{\gamma g}(x) - \operatorname{prox}_{\gamma g}(y)\|^2
\le \langle \operatorname{prox}_{\gamma g}(x) - \operatorname{prox}_{\gamma g}(y),\, x - y \rangle .
```

The fixed points of ``\operatorname{prox}_{\gamma g}`` are the minimizers of
``g``. For a nonconvex ``g``, such as the ℓ0 penalty, MCP and SCAD with large
steps, or a rank constraint, the minimizer may not be unique. ProximalBase
selects one solution when there are ties. The function pages describe the
closed forms and iterative procedures used to compute the operators.

**The Moreau envelope.** ``M_{\gamma g}(x) = \min_z g(z) + \|z - x\|^2/(2\gamma)``
is a smooth approximation of ``g`` for convex ``g``. Its gradient,

```math
\nabla M_{\gamma g}(x) = \frac{x - \operatorname{prox}_{\gamma g}(x)}{\gamma},
```

is Lipschitz with constant ``1/\gamma``.

**Moreau's decomposition.** With the convex conjugate
``g^*(y) = \sup_x \langle x, y\rangle - g(x)``,

```math
x = \operatorname{prox}_{\gamma g}(x) + \gamma\, \operatorname{prox}_{g^*/\gamma}(x/\gamma).
```

Therefore, the proximal operator of a function gives that of its conjugate.
[`Conjugate`](@ref) uses this, and so do the ℓ∞ norms, through the projection
onto the ℓ1 ball. A check:

```@example interface
using ProximalBase, LinearAlgebra
x, γ = [3.0, -0.5, 1.2], 0.7
f = NormL1(1.0)
prox(f, x, γ) + γ * prox(Conjugate(f), x / γ, 1 / γ) ≈ x
```

## Proximal gradient descent and the gradient mapping

For ``F = f + g`` with ``f`` smooth, the proximal gradient step is

```math
x^+ = \operatorname{prox}_{\gamma g}\big(x - \gamma \nabla f(x)\big),
```

and the *gradient mapping* is ``G_\gamma(x) = (x - x^+)/\gamma``, so that
``x^+ = x - \gamma G_\gamma(x)``. When ``g = 0`` it is the gradient. For convex
``f`` and ``g``, the optimality condition ``0 \in \nabla f(x) + \partial g(x)``
is equivalent, for any ``\gamma > 0``, to

```math
x = \operatorname{prox}_{\gamma g}\big(x - \gamma\nabla f(x)\big)
\iff G_\gamma(x) = 0 .
```

Therefore, ``\|G_\gamma(x)\|`` is a stopping rule that needs neither the optimal value nor
the form of ``g``. [`gradient_mapping`](@ref) computes it, and
[`gradient_mapping!`](@ref) does so without allocating.

**Step sizes.** Suppose that ``f`` is convex with an ``L``-Lipschitz gradient,
``g`` is proper, closed and convex, and ``F`` has a minimizer ``x^\star``.
For an initial point with finite objective and a fixed ``0<\gamma\le 1/L``,
the descent lemma and proximal optimality give

```math
F(x^+) \le F(x) - \frac{\gamma}{2}\|G_\gamma(x)\|^2 ,
```

so the objective does not increase. In particular, after ``k\ge1`` proximal
gradient steps,

```math
F(x^k)-F(x^\star)\le\frac{\|x^0-x^\star\|^2}{2\gamma k}.
```

Under the same convexity and smoothness conditions, the accelerated procedure
FISTA has objective error ``O(1/k^2)``. These global objective guarantees do
not extend to arbitrary nonconvex penalties. [`lipschitz`](@ref) returns
``L`` for the smooth functions. For
functions whose gradient is not globally Lipschitz, such as the Gaussian
log-likelihood and the Poisson loss, it returns ``\infty``, and a solver needs
a line search.

## Steps per entry and diagonal metrics

A scalar step is set by the worst-scaled coordinate. In a diagonal metric
``D = \operatorname{diag}(d)``, the proximal operator of a separable ``g`` is

```math
\operatorname*{argmin}_z\; g(z) + \sum_i \frac{d_i}{2}(z_i - x_i)^2 ,
```

the scalar proximal operator with the step ``\gamma_i = 1/d_i`` in every entry.
So [`prox!`](@ref) and [`value_and_prox!`](@ref) of a [`SeparableFunction`](@ref)
take an array of steps of the size of ``x`` as well as a scalar, on arrays and on
the three iterates. A group norm takes a step per group, a metric constant on each
group, which keeps the closed form. `Scaled`, `Tilt`, `Translate` and `Conjugate`
pass the steps on.

The projections onto a hyperplane, a halfspace, a simplex and an ℓ1 ball take an
array of steps as well, as projections in the metric. The hyperplane has the
closed form ``x - \gamma \odot a\,(\langle a, x\rangle - b)/\sum_i \gamma_i a_i^2``.
The simplex and the ℓ1 ball are ``\max(x_i - \gamma_i\nu, 0)`` and the soft
thresholding of ``x_i`` at ``\gamma_i\nu`` for a level ``\nu`` found by Newton's
method from the left, which ends at the exact root. [`Constrained`](@ref) on a
box, a hyperplane, a halfspace or a simplex takes the steps into its multiplier
search, with ``z_i(\nu) = \operatorname{prox}_{\gamma_i f_i}(x_i - \gamma_i\nu a_i)``.

The curvature of a smooth ``f`` along each coordinate, a Lipschitz constant of
``\partial f/\partial x_k``, comes from [`coordinate_lipschitz`](@ref) and
[`coordinate_lipschitz!`](@ref): ``\lVert X_{:,j}\rVert^2`` for least squares,
the diagonal of a quadratic, a bound for a GLM. These are the steps of coordinate
descent. A method that moves every coordinate at once can use them as the metric,
but the diagonal of a Hessian does not majorize it, so the steps
``\gamma_i = t/L_i`` need a scalar ``t`` from a line search, or
``t = 1/\lambda_{\max}(D^{-1/2}\nabla^2 f\, D^{-1/2})``, which standardizing the
columns of ``X`` amounts to.

## Traits and oracles

A function is a subtype of [`AbstractFunction`](@ref). It is callable, `f(x)`
being its value, which is `Inf` outside its domain. Its traits, defined on
types, say which oracles it provides:

| Trait | Oracles | Meaning |
|:------|:--------|:--------|
| [`issmooth`](@ref) | [`gradient!`](@ref), [`value_and_gradient!`](@ref), [`lipschitz`](@ref) | differentiable, with a gradient |
| [`isproximable`](@ref) | [`prox!`](@ref), [`value_and_prox!`](@ref) | a proximal operator that is cheap to compute |
| [`isseparable`](@ref) | [`prox_entry`](@ref), [`prox_coordinate!`](@ref), [`kkt_violation`](@ref) | ``f(x) = \sum_i f_i(x_i)`` |
| [`isconvex`](@ref) | | convex |
| [`hashessian`](@ref) | [`hessian_vec!`](@ref), [`quadratic_model`](@ref) | second derivatives, for Newton steps |

The allocating versions [`prox`](@ref), [`gradient`](@ref),
[`value_and_gradient`](@ref) and [`hessian_vec`](@ref) return new arrays.

## Conventions

Data arrays retained by a function may be shared with the caller. After changing
entries of `X` or `Y` in a `LeastSquaresLoss`, or `A` in a `QuadraticFunction`,
call [`reset_cache!`](@ref) on the outermost function before using its proximal
operator again. Sizes must stay unchanged. Reconstruct `IndAffine` after changing
its data; its factorization type can change with the rank of `A`.

- **Output first.** Mutating functions take their output first, as `mul!` does:
  `prox!(y, f, x, γ)` and `gradient!(g, f, x)`. The output may be the input:
  `prox!(x, f, x, γ)` works for every function.
- **The value at the result.** [`value_and_prox!`](@ref) stores the proximal
  point and returns the function's value there. Many functions get that value
  from the computation itself: the nuclear norm from the singular values it
  thresholds, an indicator function as zero. A solver that tracks the objective
  should call it instead of `prox!` followed by `f(y)`.
- **Steps and types.** Steps may be any positive real. Computations follow the
  element type of the argument, so `Float32` input gives `Float32` results.
- **No allocations.** The hot calls (`f(x)`, `prox!`, `value_and_prox!`,
  `gradient!`, `value_and_gradient!`, `prox_entry`, `prox_coordinate!`)
  allocate nothing once a function has seen its input. The spectral functions
  and [`LogDetLoss`](@ref) own LAPACK workspaces sized for their input. A
  function that owns a workspace must not be shared between threads; give each
  thread a `deepcopy`.

## GPU arrays

The separable functions work on GPU arrays: CUDA.jl's `CuArray`, and the other
arrays built on GPUArrays (AMDGPU.jl, Metal.jl, oneAPI.jl). Loading such a
package loads an extension in which `f(x)`, [`prox!`](@ref) and
[`value_and_prox!`](@ref) are broadcasts and reductions that run as kernels,
instead of the loops over entries of the dense path. This covers
[`NormL1`](@ref), [`NormL1Nonneg`](@ref), [`ElasticNet`](@ref),
[`IndBox`](@ref) and [`IndBallLinf`](@ref), [`IndNonnegative`](@ref),
[`NormL0`](@ref), [`MCP`](@ref), [`SCAD`](@ref) and [`SqrNormL2`](@ref). It also
covers [`NormL2`](@ref), [`IndBallL2`](@ref), [`NormLinf`](@ref),
[`IndBallL1`](@ref) and [`IndSimplex`](@ref). The last three find the level of
their thresholding by Michelot's algorithm, which needs only reductions,
instead of a selection: a few passes over the array.

The group functions [`GroupNormL2`](@ref), [`GroupNormLinf`](@ref),
[`SparseGroupNormL2`](@ref) and [`IndBallGroupL0`](@ref) run there too, by
segmented reductions. The groups of sizes in one class (1, 2, 3–4, 5–8, …) are
the columns of an index matrix, padded to the largest of them; one gather and
one reduction along the columns give the norms of all the groups of the class,
and a vector of group ids carries each group's factor or level back to its
entries. The number of passes grows with the number of size classes, not with
the number of groups. `Adapt.adapt(CuArray, f)` builds this layout on the
device once; a group function left on the host builds it at every call.
`IndBallGroupL0` selects its `s` groups on the host, among the norms of the
groups; with a bound `k` on the entries of a group, its projection runs on the
CPU only.

The losses with a design matrix, [`LeastSquaresLoss`](@ref) and
[`GLMLoss`](@ref) (with [`LogisticLoss`](@ref), [`PoissonLoss`](@ref) and
[`HuberLoss`](@ref)), keep their buffers next to the response.
`Adapt.adapt(CuArray, f)` moves `X`, the response, the weights and the offset
to the device, where `f(β)`, [`gradient!`](@ref) and
[`value_and_gradient!`](@ref) are products by `X` and `X'` and a broadcast over
the observations. The proximal operator of `LeastSquaresLoss` uses the SVD of
`X` and runs on the CPU only.

- **Parameter arrays on the device.** Weights, bounds and centers must live on
  the device with the argument: `Adapt.adapt(CuArray, NormL1(λ, w))` copies
  them there and checks them, as the constructor does.
- **Your own separable functions.** A [`SeparableFunction`](@ref) whose fields
  are numbers works as it is: its [`prox_entry`](@ref) and
  [`value_entry`](@ref) run inside the kernels. One that holds arrays needs an
  `Adapt.adapt_structure` method that rebuilds it with the arrays converted
  and without checks, since inside a kernel its arrays are device arrays.
- **Not yet.** The spectral functions and [`LogDetLoss`](@ref) (which use
  LAPACK), the second-order methods of `GLMLoss`, and the sparse iterates run
  on the CPU only. The GPU extension supports scalar steps; diagonal-step
  proximal methods currently use CPU loops.

Normal CI checks these operations through JLArrays, a reference backend that
executes on the CPU. Optional physical NVIDIA GPU validation uses the separate
[`test/cuda` environment](https://github.com/mlakolar/ProximalBase.jl/tree/master/test/cuda).
It shares the reference-backend cases, disables scalar indexing, synchronizes
device work before numerical comparisons and reports host and device allocations
separately. CUDA remains optional and is not required for ordinary tests.
