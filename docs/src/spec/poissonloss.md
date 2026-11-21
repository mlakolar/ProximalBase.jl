# [PoissonLoss numerical contract](@id spec-poissonloss)

## [Identity](@id spec-poissonloss-identity)

[`PoissonLoss`](@ref) is the Poisson regression loss with the log link on dense
`Float64` data. Its constructor and oracles are implemented in
[`src/functions/poissonloss.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/poissonloss.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-poissonloss-definition)

`PoissonLoss(y, X)` takes a response vector `y` of length ``n`` and a design
``X`` with ``n`` rows. With ``\eta = X\beta`` the value and gradient are

```math
f(\beta)=\sum_{i=1}^n \bigl(e^{\eta_i}-y_i\eta_i\bigr),\qquad
\nabla f(\beta)=X^\top d,\quad d_i=e^{\eta_i}-y_i.
```

The term ``\log\Gamma(y_i+1)`` of the Poisson log-likelihood does not depend on
``\beta`` and is omitted. Responses are finite and nonnegative, including
noninteger values. A response of `-0.0` is a zero response.

The value is the sum over observations, not the mean. The factor `(1/n) * f`,
that is `Scaled(f, 1/n)`, gives the mean; there is no `f / n`. A sum that
overflows to `Inf` stays `Inf` under `Scaled` even when the mean is finite,
because the loss has no value-recovery path.

The constructor has no weight argument and no offset argument. The predictor
oracles take any predictor of length ``n``. A caller who forms
``\eta = X\beta + \mathrm{offset}`` evaluates that model: the value is
``\sum_i (e^{\eta_i}-y_i\eta_i)``, the observation derivative is
``d_i=e^{\eta_i}-y_i`` and the coefficient gradient is ``X^\top d``. An exposure
``t_i`` enters as the offset ``\log t_i``. Observation weights have no route in
this loss.

## [Traits](@id spec-poissonloss-traits)

The function is convex and smooth. It is neither proximable nor separable.
`lipschitz(f)` throws `ArgumentError`, because the curvature ``e^{\eta_i}`` has
no global bound. The loss has no second-order interface.

## [Supported arguments](@id spec-poissonloss-arguments)

The constructor takes `y::Vector{Float64}` and `X::Matrix{Float64}`. Other
element types, views and sparse designs throw `MethodError`. Coefficients,
gradient outputs, predictors and observation-derivative outputs are one-based
`StridedVector{Float64}`. The loss supports `f(β)`, `gradient!`,
`value_and_gradient!`, `ProximalBase.value_from_predictor(f, η)` and
`ProximalBase.value_and_gradient_from_predictor!(g, d, f, η)`, together with
the allocating `gradient` and `value_and_gradient`. One `Scaled` wrapper is
supported by `f(β)`, `gradient!` and `value_and_gradient!`, and not by the two
predictor oracles.

## [Preconditions](@id spec-poissonloss-preconditions)

The constructor borrows `y` and `X`. An in-place edit of either is seen by the
next call. A same-shape edit must keep the responses finite and nonnegative;
this is not checked after construction. Reconstruct the loss for a structural
change, and refresh a caller-held predictor when its defining inputs change.

The ordinary calls write a private scratch vector and need synchronization
across tasks. The predictor calls write no field of the loss, so one loss can
serve several tasks while `y` and `X` stay read-only and the outputs are
independent. Before any write, the oracles reject a nonempty overlap, as
reported by `Base.mightalias`, of an output with the input, the other output,
`y`, `X` or the scratch. A coefficient vector or predictor that overlaps the
scratch is also rejected. Sharing that `Base.mightalias` does not report
remains the caller's responsibility.

## [Results](@id spec-poissonloss-results)

`f(β)` returns the value at ``\eta = X\beta``. `gradient!` writes
``X^\top d`` to its output and returns it. `value_and_gradient!` writes the
gradient and returns the value. `value_from_predictor` returns the value at
the supplied predictor and reads neither `X` nor the scratch.
`value_and_gradient_from_predictor!` writes ``d_i=e^{\eta_i}-y_i`` to its `d`
output and ``X^\top d`` to its `g` output, and returns the value. It forms one
product with `X'` and none with `X`. The supplied predictor is not modified.

## [Ordinary-path clauses](@id spec-poissonloss-ordinary)

The accuracy statement is relative to the predictor that the scalar loop
reads: the computed `X*β` for `f(β)`, `gradient!` and `value_and_gradient!`, and
the supplied predictor for the predictor oracles. Rounding in the matrix
product that forms `X*β` is not included. Let `eps = eps(Float64)`, and let
``e_i = e^{\eta_i}`` be finite with ``y_i\eta_i`` finite.

- Each computed value term differs from ``e_i-y_i\eta_i`` by at most
  ``4\,\mathrm{eps}\,(e_i+\lvert y_i\eta_i\rvert)+2^{-1074}``. The last
  summand covers a subnormal or zero computed exponential.
- Each computed derivative differs from ``e_i-y_i`` by at most
  ``4\,\mathrm{eps}\,(e_i+y_i)+2^{-1074}``.
- The value adds the error of plain recursive summation of the computed terms
  ``\hat v_i``, at most ``\gamma_{n-1}\sum_i\lvert\hat v_i\rvert`` with
  ``\gamma_k=(k\,\mathrm{eps}/2)/(1-k\,\mathrm{eps}/2)``. There is no
  relative-accuracy statement for a value that cancels, near
  ``\eta_i=\log y_i`` or across observations.
- The gradient output ``g`` differs from ``X^\top d`` of the exact derivative
  by at most
  ``\sum_i\lvert X_{ij}\rvert\,\lvert\Delta d_i\rvert+\gamma_n\sum_i\lvert X_{ij}\rvert\,\lvert\hat d_i\rvert``
  in entry ``j``, for any summation order of the transposed product, where
  ``\lvert\Delta d_i\rvert`` is bounded by the derivative statement above and
  ``\hat d_i`` is the computed derivative. A product ``X_{ij}\hat d_i`` that
  lies in the subnormal range adds an absolute error of at most ``2^{-1075}``
  for that product.

Julia documents no ulp bound for `exp`; the constant four leaves a margin for
the exponential, the product, and the subtraction. The five operations agree
within these statements at the same predictor.

## [Range clauses](@id spec-poissonloss-range)

A term is computed as `±Inf` when `exp(ηᵢ)` overflows, which happens for a
finite predictor above ``\log(\texttt{floatmax(Float64)})\approx 709.7827``,
or when `yᵢ*ηᵢ` overflows while `exp(ηᵢ)` is finite. In the second case the
value term is `-Inf`, even where the exact term is finite. For example, with
`y = 2.6e305` and `η = 709.0` the exact term is about ``-1.02\times10^{308}``
and the computed term is `-Inf`. The loss attempts no recovery of finite terms
beyond these boundaries. The exact value term can be finite for ``\eta`` up to
about 716.4, and the exact derivative for ``\eta`` up to about 710.48.

At ``\eta=\log(\texttt{floatmax(Float64)})`` itself the exponential and the
derivative are finite, and so is the value term when `y*η` is finite. The sum of
the terms, the transposed product and the outer scaling use ordinary
floating-point arithmetic with their usual rounding and range limits.

**Implementation notes (not guarantees).** One loop computes
`e = exp(η)`, `v = ifelse(iszero(y) | (e == Inf), e, e - y*η)` and `d = e - y`.
It uses plain accumulation with no `@simd`, `muladd` or `fma`. The `ifelse`
selects `e` for a zero response and for an infinite exponential, which avoids
`0*Inf` and `Inf - Inf`. Both arms are evaluated and the unselected arm can be
NaN. The loop reads the predictor in index order, and the calls that return a value
sum the same terms in that order.

## [Endpoint clauses](@id spec-poissonloss-endpoints)

For a finite response ``y\geq0``, the computed value term and derivative at
the endpoint predictors are as follows.

| Response and predictor | Value term | Derivative |
|:--|:--|:--|
| `y = 0`, `η = -Inf` | `0` | `0` |
| `y = 0`, `η = +Inf` | `Inf` | `Inf` |
| `y > 0`, `η = +Inf` | `Inf` | `Inf` |
| `y > 0`, `η = -Inf` | `Inf` | `-y` |
| `η` NaN | NaN | NaN |
| finite `η` with `exp(η)` overflowing | `Inf` | `Inf` |

For finite `y` and finite `η`, no term is NaN. A response of `-0.0` follows
the `y = 0` rows. A call at a nonfinite predictor does not throw.

## [Failure behavior](@id spec-poissonloss-failures)

The constructor throws `DimensionMismatch` when `length(y) != size(X, 1)` and
`ArgumentError` for a negative, NaN or infinite response. A `weights` or
`offset` keyword, other element types, views and sparse arrays throw
`MethodError`. A non-one-based `y` or `X` throws `ArgumentError` or
`MethodError`.

A coefficient vector, predictor or output of the wrong length throws
`DimensionMismatch`. A non-one-based argument throws `ArgumentError` or
`MethodError`, and a reported overlap throws `ArgumentError`. These checks
run before the call writes anything, so the outputs and the scratch are
unchanged after a refusal. A `Scaled` loss reaches the same checks.

## [Limitations and non-guarantees](@id spec-poissonloss-limitations)

The loss does not recover a finite term that overflows, as described under
range clauses. Aggregates of finite inputs can be nonfinite. The value is NaN
when one term is `+Inf` and another is `-Inf`. The transposed product gives
NaN where an infinite ``d_i`` meets ``X_{ij}=0``, or where infinite
contributions of both signs meet. `f(β)` with finite `X` and `β` can form an
infinite predictor in the matrix product. A partial sum that overflows is not
recovered: the terms `1e308`, `1e308` and `-1e308` sum to `Inf` although the
exact sum is `1e308`.

The accuracy statement excludes the rounding of `X*β`, and it gives no
relative accuracy for a value that cancels. The loss has no weights, offset
field, global curvature bound or second-order interface.

## [Clauses owned elsewhere](@id spec-poissonloss-elsewhere)

This section links to the shared owners.

- [Array and precision conventions](@ref numerical-usage)
- [Shared range guarantees](@ref numerical-value-range)
- [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases)
- [Scaling a function](@ref scaling-guide)

## [Evidence](@id spec-poissonloss-evidence)

[`test/poissonloss.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/poissonloss.jl)
compares the terms, the derivative, the value and the gradient with a BigFloat
reference at the `Float64` predictor, using the bounds above. It also covers
each endpoint and overflow case, finite differences, validation and aliases
before writes, `Scaled`, borrowed and edited inputs, empty shapes, warmed
allocations and product counts. Listing a test does not establish a broader
guarantee. The [Poisson regression guide](@ref Poisson-regression-loss) contains
an executable example.

## [History](@id spec-poissonloss-history)

The [restoration issue #1130](https://github.com/mlakolar/ProximalBase.jl/issues/1130)
records the numerical decisions and test plan.
