# [QuadraticFunction numerical contract](@id spec-quadratic)

This page gathers the existing numerical rules for `QuadraticFunction`.
The text is preserved from the earlier manual. Clause conversion and status
assignments are a separate review step; unstated status remains unclassified.
Read [How to read the numerical contracts](index.md) for authority and ownership.

## [Identity](@id spec-quadratic-identity)

**Links only until clause conversion.** Constructors and API help: [`QuadraticFunction`](@ref).

Source: [`src/functions/quadratic.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/quadratic.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-quadratic-definition)

**Links only until clause conversion.** The mathematical definition and ordinary examples are in [the function guide](../smooth.md#Quadratic-function).

## [Traits](@id spec-quadratic-traits)

**Links only until clause conversion.** Trait declarations are described by [`QuadraticFunction`](@ref) and [the function guide](../smooth.md#Quadratic-function).

## [Supported arguments](@id spec-quadratic-arguments)

**Links only until clause conversion.** Operation signatures and representations are described by [`QuadraticFunction`](@ref) and [the function guide](../smooth.md#Quadratic-function).

## [Preconditions](@id spec-quadratic-preconditions)

The proximal formula referred to as “above” is in the [quadratic function guide](../smooth.md#Quadratic-function).

**Step domain of the prox:** let ``\gamma`` be the step after the factors of
any `Scaled` wrappers are applied, converted to the element type ``T`` of the
function, and let ``\lambda_{\min}`` be the smallest eigenvalue of the cached
decomposition of `A`, a computed approximation of the true smallest eigenvalue.
When ``\lambda_{\min}<0`` and the rounded product ``\gamma(-\lambda_{\min})`` is not
``<1`` (a NaN step included), the prox throws `DomainError` before the output
is written; otherwise it applies the formula above, and when
``\lambda_{\min}\ge0`` no step check is made.

### [Computed proximal denominators](@id quadratic-function.prox.computed-denominators)

For a nonnegative finite
``\gamma`` and finite cached eigenvalues, the check accepts exactly when every
computed denominator ``1+\gamma\lambda_i`` is positive.

## [Results](@id spec-quadratic-results)

**Links only until clause conversion.** Return values are described by [`QuadraticFunction`](@ref). Shared output and aliasing rules are in [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases).

## [Ordinary-path clauses](@id spec-quadratic-ordinary)

**Links only until clause conversion.** The ordinary mathematical formulas and examples remain in [the function guide](../smooth.md#Quadratic-function).

## [Range clauses](@id spec-quadratic-range)

In exact arithmetic and for ``\gamma>0``, the proximal subproblem has a unique
minimizer if and only if ``1+\gamma\lambda_{\min}(A)>0`` for the true smallest eigenvalue. The check
tests computed values, so near that boundary it can accept a step for which the
true ``I+\gamma A`` is not positive definite, and reject one for which it is. The
second case includes a singular positive semidefinite `A` whose computed
``\lambda_{\min}`` is slightly negative: steps with ``\gamma(-\lambda_{\min})\ge1``
throw for it, which is about ``\gamma\ge1/|\lambda_{\min}|``.

## [Endpoint clauses](@id spec-quadratic-endpoints)

**Links only until clause conversion.** Shared endpoint conventions remain in [NaN, infinities and zeros](../numerics.md#NaN,-infinities-and-zeros).

## [Failure behavior](@id spec-quadratic-failures)

**Links only until clause conversion.** Read [Supported arguments](@ref spec-quadratic-arguments), [Preconditions](@ref spec-quadratic-preconditions) and [Endpoint clauses](@ref spec-quadratic-endpoints), together with the failure behavior in [`QuadraticFunction`](@ref).

## [Limitations and non-guarantees](@id spec-quadratic-limitations)

The
cached eigenvalue computation has no general non-finite-data guarantee.
Gradient and decomposition arithmetic can overflow independently of the
exceptional value-summation recovery.

## [Clauses owned elsewhere](@id spec-quadratic-elsewhere)

**Links only until clause conversion.** The following rules retain their shared owners.

| Subject | Owning section |
|:--|:--|
| Supported precision and oracle agreement | [Supported precision and oracle agreement](../numerics.md#Supported-precision-and-oracle-agreement) |
| Shared range bounds and recovery limits | [Range guarantees](@ref numerical-value-range) |
| Shapes and destination writes | [Shapes and destination-first mutation](../numerical_usage.md#Shapes-and-destination-first-mutation) |
| Ownership and aliasing | [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases) |
| Sparse products and stored zeros | [Sparse product semantics](@ref Sparse-product-semantics) |

## [Evidence](@id spec-quadratic-evidence)

These existing tests contain the numerical checks for this family. Their assertions define the cases covered; listing a test does not establish a broader guarantee.

- [`test/quadratic_domain.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/quadratic_domain.jl)
- [`test/numerical_recovery.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/numerical_recovery.jl)
- [`test/kkt_smooth.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/kkt_smooth.jl)

| Clause ID | Status | Evidence |
|:--|:--|:--|
| `quadratic-function.prox.computed-denominators` | `characterized` | [Computed step check and spectral solve](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/src/functions/quadratic.jl#L216-L244), [accepted and rejected steps](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/quadratic_domain.jl#L102-L145), and [step conversion at the boundary](https://github.com/mlakolar/ProximalBase.jl/blob/3a14f44b7ac11a8bfc7ffbe1e97c75b253120231/test/quadratic_domain.jl#L292-L307) |

## [History](@id spec-quadratic-history)

The [documentation migration](https://github.com/mlakolar/ProximalBase.jl/issues/1019) records the preserved source text and its review. Numerical decisions remain with their linked source, tests and issue records.
