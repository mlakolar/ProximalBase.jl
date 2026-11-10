
# Operations on functions

This is the **user-facing catalog of operations** that ProximalBase exposes
for function objects. Choose an operation by the question your algorithm
needs answered, then check the corresponding function's supported traits,
operand types, and documentation. The [API reference](api/interface.md)
provides complete method docstrings; this page explains the roles and how
the operations fit together.

The central convention is **output first** for mutating oracles: write
`gradient!(out, f, x)` or `prox!(out, f, x, γ)`. Allocating variants, when
available, create and return an output. An exclamation mark alone does not
guarantee zero allocations; see [Array and numerical conventions](numerical_usage.md).

## Function values and first derivatives

| Operation | Inputs / result | Use |
|:--|:--|:--|
| `f(x)` | Scalar value at `x` | Evaluate an objective, loss, penalty or indicator; `Inf` can denote infeasibility |
| [`value_entry(f, v, i)`](@ref) | Entry value | Evaluate entry `i` of a supported separable function, using scalar `v` |
| [`gradient!(out, f, x)`](@ref) | Writes gradient into `out`; returns `out` | Use in a smooth first-order algorithm |
| [`gradient(f, x)`](@ref) | Allocated gradient | Convenience form of `gradient!` |
| [`value_and_gradient!(out, f, x)`](@ref) | Writes gradient; **returns `f(x)`** | Reuse shared work between value and gradient |
| [`value_and_gradient(f, x)`](@ref) | Allocated pair `(f(x), gradient)` | Convenience combined form |
| [`lipschitz(f)`](@ref) | Known gradient-Lipschitz bound, possibly `Inf` | Step-size selection for supported smooth functions |
| [`opnorm2(X)`](@ref) | Operator-norm estimate for a matrix, not a function oracle | Large matrix norm estimation; an estimate is not automatically a valid upper bound |

A true `issmooth(f)` trait provides differentiability and a gradient, but
does **not** promise that `lipschitz(f)` is finite. Use a line search or
appropriate theory when the known bound is infinite or not available.

## Proximal operators and projections

| Operation | Inputs / result | Use |
|:--|:--|:--|
| [`prox!(out, g, x, γ)`](@ref prox!) | Stores proximal point in `out`; returns `out` | Regularization, proximal gradient and projection |
| [`prox(g, x, γ)`](@ref prox) | Allocated proximal point | Convenience form |
| [`value_and_prox!(out, g, x, γ)`](@ref value_and_prox!) | Writes proximal point; **returns the function value at the resulting point** | Avoid separate evaluation when available |
| [`prox_block(g, u, k, γ)`](@ref prox_block) | Scalar entry or allocated compact group proximal point | Coordinate descent with a separable or group penalty |
| [`prox_block!(out, g, u, k, γ)`](@ref prox_block!) | Writes one compact group proximal point into a numerical buffer | Reuse a group buffer before writing an iterate |
| [`shrink(v, c)`](@ref) | Soft-thresholded scalar or supported input | Low-level shrinkage utility, not a generic function oracle |

For a set indicator `g = IndBox(l, u)` or another proximable indicator,
`prox!` computes a projection in the supported domain. For a penalty such
as `NormL1`, it computes shrinkage. Not every function with a proximal map
supports array-valued steps or every backend; consult the specific docstring.
The return value of `value_and_prox!` is not automatically a *feasibility
certificate* for an inexact projection.

```@example function_operations
using ProximalBase

f = SqrNormL2(1.0)
g = NormL1(0.5)
x = [2.0, -1.0]

grad = similar(x)
value_at_x = value_and_gradient!(grad, f, x)
@assert value_at_x ≈ f(x) && grad ≈ gradient(f, x)

out = similar(x)
value_at_prox = value_and_prox!(out, g, x, 1.0)
@assert out ≈ prox(g, x, 1.0)
@assert value_at_prox ≈ g(out)
(value_at_x = value_at_x, grad = grad, prox = out, value_at_prox = value_at_prox)
```

For mathematical derivations and step-size geometry, read
[The interface](interface.md).

## Coordinate descent, restrictions, and optimality

### [Update vocabulary](@id update-vocabulary)

| Term | Meaning |
|:--|:--|
| Entry | One scalar of the full vector or matrix; its linear index addresses a separable function's weights and bounds. |
| Iterate coordinate | One independent storage update: an entry of `SparseIterate`, a lower-triangle variable of `SymmetricSparseIterate`, or a group of `AtomIterate`. |
| Penalty block | One independent mathematical term of a penalty: a scalar entry for a separable penalty, or a compact vector in a group penalty's own order. This is the index space of `prox_block[!]`. |
| Group | An ordered collection of global entry indices. A solver must check that its iterate and penalty use matching ordered groups before sharing a group ID. |
| Active slot | A temporary position in packed working-set handles; its order can change during compaction. It is never a penalty block or coordinate ID. |
| Atom | A compact linear-minimization result, such as `VertexAtom` or `LowRankAtom`, returned by supported `lmo_atom` calls. It does not mean a group of coefficients. |

The existing storage names `AtomIterate`, `atom(x,k)` and `atomindices(x,k)`
refer to grouped storage and group views. They do not imply linear-minimization
atom semantics. Their proposed rename is a separate maintainer decision;
the #844 migration retains these names. `value_entry`, `coordinate_lipschitz`
and iterate-aware `kkt_violation` retain their existing scopes.

| Operation | Use |
|:--|:--|
| [`prox_block(g, u, k, γ)`](@ref prox_block) | Mathematical result for scalar entry or penalty group `k` at trial `u` |
| [`prox_block!(out, g, u, k, γ)`](@ref prox_block!) | Mathematical group result in caller-owned compact numerical storage |
| [`coordinate_lipschitz(f, k)`](@ref) | Coordinate curvature bound when defined by the smooth loss |
| [`coordinate_lipschitz!(out, f)`](@ref) | Fill supported per-coordinate bounds in a caller-provided output |
| [`kkt_violation(g, grad, i)`](@ref kkt_violation) | Screen entry `i` at zero against the smooth-part gradient |
| [`kkt_violation(g, grad, x, k)`](@ref kkt_violation) | Screen coordinate `k` of the supplied iterate; group and symmetric geometry matter |
| [`restrict(f, S)`](@ref) | Construct the supported restriction of a function to selected coordinates |
| [`gradient_mapping(f, g, x, γ)`](@ref gradient_mapping) | Allocate a proximal-gradient optimality residual for `f+g` |
| [`gradient_mapping!(out, f, g, x, γ)`](@ref gradient_mapping!) | Write the residual into `out` |

The curvature, restriction and screening methods are **not** available for
arbitrary functions merely because they subtype `AbstractFunction`.
`gradient_mapping` is for an `AbstractArray` point, not the non-array
symmetric owner. `prox_block` and `prox_block!` use **penalty block indices**;
the solver converts iterate coordinate IDs, validates group layouts and
symmetric parameters, and writes the result through public storage methods.
For the active-set and trial-point rules see
[Coordinate oracles](iterate_oracles.md) and
[Coordinate descent](coordinate_descent.md).

## Second-order and model operations

| Operation | Use |
|:--|:--|
| [`hashessian(f)`](@ref) | Ask whether the second-order interface is provided |
| [`hessian_vec!(out, f, x, v)`](@ref hessian_vec!) | Hessian-vector product, writing into `out` |
| [`hessian_vec(f, x, v)`](@ref hessian_vec) | Allocating Hessian-vector product |
| [`quadratic_model(f, x)`](@ref) | Construct a quadratic model of a supported function around `x` |
| [`quadratic_model!(model, f, x)`](@ref quadratic_model!) | Refresh/update a compatible model |

These are specialized mathematical computations, not generic automatic
differentiation. A Hessian trait is a capability declaration, not a promise
that arbitrary functions or devices have second-order kernels. Consult
[the second-order API reference](api/interface.md).

## Linear minimization and compact atoms

| Operation | Use |
|:--|:--|
| [`haslmo(C)`](@ref) | Check whether a particular set instance has a supported linear minimization oracle |
| [`lmo!(out, C, g)`](@ref lmo!) | Solve the supported linear minimization subproblem over `C`, writing the minimizer |
| [`lmo(C, g)`](@ref lmo) | Allocating linear minimization oracle |
| [`lmo_atom(C, g)`](@ref) | Return a compact atom in the supported cases, rather than a full dense minimizer |
| [`VertexAtom`](@ref), [`LowRankAtom`](@ref) | Represent compact vector and low-rank atoms used by certain linear minimization oracles |

This is useful for Frank–Wolfe / conditional-gradient methods. `haslmo`
depends on the *instance*, for example whether a set has finite bounds.
The atom form is specialized to sets with a meaningful compact
representation; it is not guaranteed for every set supporting `lmo!`.

## Local linear approximation for nonconvex penalties

| Operation | Use |
|:--|:--|
| [`lla_weight(f, v, i)`](@ref) | Scalar reweighting coefficient of a supported penalty |
| [`lla_weights!(out, f, x)`](@ref) | Compute reweighting coefficients into supplied storage |
| [`lla_weights(f, x)`](@ref) | Allocated weights |

These operations support locally linearized or reweighted algorithms for
selected penalties, not an automatic nonconvex solver. See the
[penalty catalog](penalties.md) for supported functions.

## Capability and state queries

| Operation | Meaning |
|:--|:--|
| [`issmooth`](@ref), [`isproximable`](@ref), [`isseparable`](@ref), [`isconvex`](@ref) | Differentiability, proximal capability, separability, convexity |
| [`hashessian`](@ref), [`haslmo`](@ref), [`iszeropreserving`](@ref) | Second-order capability, instance-specific linear minimization, zero-preservation |
| [`reset_cache!(f)`](@ref) | Invalidate supported function-owned cached state after relevant external data mutation |

Traits and their qualifications are described in
[Function objects and capabilities](function_objects.md). Avoid
equating a capability with specialized support for every possible argument
type.

## Where the remaining public API belongs

Other exports do **not** represent mathematical-function oracles.
`SparseIterate`, `SymmetricSparseIterate` and `AtomIterate`, together with
`ncoordinates`, `coordinates`, `nactive`, `activecoordinates`,
`activevalues`, `matrixview`, `isactive`, `activate!`, `atom`,
`atomindices` and `dropzeros!`, belong to the
[Iterates](iterates.md) and [iterate API reference](api/iterates.md).
`nnz` is re-exported for ordinary sparse arrays; it does **not**
define the number of active coordinates of an iterate.

`norm_diff`, `tril_index` and `tril_cartesian` are general or
indexing utilities, described in the API reference. `PrecisionError`
represents a numerical issue, discussed in
[Array and numerical conventions](numerical_usage.md).
`SeparableFunction` is a function subtype for authors of custom
functions; see [Writing your own function](extending.md).

## Find the right documentation

- **What can I call?** Use this catalog and then the
  [API reference](api/interface.md) for method signatures and exceptions.
- **Which function objects exist?** Use the
  [function library](function_library.md).
- **How are proximal/gradient operations defined?** Use
  [The interface](interface.md).
- **How do I build an algorithm from these calls?** Use
  [Building optimization algorithms](optimization_algorithms.md).
- **How do I implement a new function?** Use
  [Writing your own function](extending.md).
