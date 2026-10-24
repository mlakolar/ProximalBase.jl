# The interface

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
returns one of them, computed exactly.

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

So the proximal operator of a function gives that of its conjugate.
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

So ``\|G_\gamma(x)\|`` is a stopping rule that needs neither the optimal value nor
the form of ``g``. [`gradient_mapping`](@ref) computes it, and
[`gradient_mapping!`](@ref) does so without allocating.

**Step sizes.** If ``\nabla f`` is Lipschitz with constant ``L`` and
``\gamma \le 1/L``, the descent lemma gives

```math
F(x^+) \le F(x) - \frac{\gamma}{2}\|G_\gamma(x)\|^2 ,
```

so every step decreases the objective. Proximal gradient descent then converges
at the rate ``O(1/k)`` in objective value, and its accelerated version (FISTA)
at ``O(1/k^2)``. [`lipschitz`](@ref) returns ``L`` for the smooth functions. For
functions whose gradient is not globally Lipschitz, such as the Gaussian
log-likelihood and the Poisson loss, it returns ``\infty``, and a solver needs
a line search.

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
[`NormL0`](@ref), [`MCP`](@ref), [`SCAD`](@ref) and [`SqrNormL2`](@ref).

- **Parameter arrays on the device.** Weights, bounds and centers must live on
  the device with the argument: `Adapt.adapt(CuArray, NormL1(λ, w))` copies
  them there and checks them, as the constructor does.
- **Your own separable functions.** A [`SeparableFunction`](@ref) whose fields
  are numbers works as it is: its [`prox_entry`](@ref) and
  [`value_entry`](@ref) run inside the kernels. One that holds arrays needs an
  `Adapt.adapt_structure` method that rebuilds it with the arrays converted
  and without checks, since inside a kernel its arrays are device arrays.
- **Not yet.** The group norms, the projections onto the simplex and the balls,
  the spectral functions and [`LogDetLoss`](@ref) (which use LAPACK), the
  losses of a design matrix, and the sparse iterates run on the CPU only.
