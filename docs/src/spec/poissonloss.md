# [PoissonLoss numerical contract](@id spec-poissonloss)

## [Identity](@id spec-poissonloss-identity)

[`PoissonLoss`](@ref) is the Poisson regression loss with the log link on dense
`Float64` data. Its constructor and oracles are implemented in
[`src/functions/poissonloss.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/src/functions/poissonloss.jl).
The [API coverage matrix](../api/coverage.md#API-coverage-matrix) lists the supported interface.

## [Definition and domain](@id spec-poissonloss-definition)

`PoissonLoss(y, X)` takes a response vector `y` of length ``n`` and a design
``X`` with ``n`` rows. With ``\eta = X\beta``, the mathematical value and
gradient are

```math
f(\beta)=\sum_{i=1}^n \bigl(e^{\eta_i}-y_i\eta_i\bigr),\qquad
\nabla f(\beta)=X^\top d,\quad d_i=e^{\eta_i}-y_i.
```

The constant ``\log\Gamma(y_i+1)`` of the negative Poisson log-likelihood is
omitted. Responses are finite and nonnegative; noninteger responses are
accepted, and `-0.0` is a zero response.

The value is the sum over observations. For ``n>0`` and a finite sum,
`(1/n) * f`, that is `Scaled(f, 1/n)`, gives the mean as the computed sum scaled
by `1/n` with ordinary rounding. For ``n=0``, evaluating
`(1/0) * f` returns `NaN`. A sum that overflows to `Inf` stays `Inf` after
scaling by `1/n`.

The constructor has no weight or offset argument. The predictor oracles
evaluate the loss at a supplied predictor that meets the requirements under
[Supported arguments](@ref spec-poissonloss-arguments) and
[Preconditions](@ref spec-poissonloss-preconditions). At a supplied
``\eta = X\beta + \mathrm{offset}`` they give the value
``\sum_i (e^{\eta_i}-y_i\eta_i)``, the derivative ``d_i=e^{\eta_i}-y_i`` and the
gradient ``X^\top d`` of the model with that offset. An exposure ``t_i`` enters
as the offset ``\log t_i``.

## [Traits](@id spec-poissonloss-traits)

`isconvex` and `issmooth` return true; `isproximable` and `isseparable` return
false. The API supplies no gradient Lipschitz bound: `lipschitz(f)` throws
`ArgumentError`.

## [Supported arguments](@id spec-poissonloss-arguments)

The constructor takes `y::Vector{Float64}` and `X::Matrix{Float64}`.
Coefficients and gradient outputs of length `size(X, 2)`, and predictors and
derivative outputs of length `length(y)`, are one-based `StridedVector{Float64}`. The operations are `f(β)`, `gradient!(g, f, β)`,
`value_and_gradient!(g, f, β)`, `ProximalBase.value_from_predictor(f, η)`,
`ProximalBase.value_and_gradient_from_predictor!(g, d, f, η)` and the allocating
`gradient` and `value_and_gradient`. A loss wrapped in `Scaled` supports `f(β)`,
`gradient!` and `value_and_gradient!`; the two predictor oracles take an
unwrapped loss.

## [Preconditions](@id spec-poissonloss-preconditions)

The constructor borrows `y` and `X`. Each call reads the current entries of `y`,
and each call that uses the design reads the current entries of `X`. An in-place
edit must keep the responses finite and nonnegative; this is not checked after
construction. A change of length needs a new loss.

The calls `f(β)`, `gradient!` and `value_and_gradient!` write a private scratch
vector and need synchronization across tasks. The predictor oracles write no
field of the loss. Concurrent predictor calls on one loss need `y`, `X` and
every shared read buffer to stay unchanged during the calls, and each written
output to be disjoint from the inputs and outputs of every other concurrent call.

Before any write, a call with otherwise valid arguments throws `ArgumentError`
when `Base.mightalias` reports an overlap of two nonempty arrays: an output and
the input, the other output, `y`, `X` or the scratch, or a coefficient vector or
predictor and the scratch. Sharing that `Base.mightalias` does not report is the
caller's responsibility.

## [Results](@id spec-poissonloss-results)

`f(β)` returns the value at ``\eta = X\beta``. `gradient!` writes ``X^\top d`` to
`g` and returns `g`. `value_and_gradient!` writes ``X^\top d`` to `g` and returns
the value. `value_from_predictor` returns the value at the supplied predictor.
`value_and_gradient_from_predictor!` writes ``d_i=e^{\eta_i}-y_i`` to `d` and
``X^\top d`` to `g`, and returns the value. The predictor oracles do not form
`X*β`; the fused oracle forms one transposed product `X'*d` when both dimensions
of `X` are nonzero and none otherwise. The supplied predictor is not modified.

## [Ordinary-path clauses](@id spec-poissonloss-ordinary)

The bounds below are relative to the predictor that the loop reads: the
computed `X*β` for `f(β)`, `gradient!` and `value_and_gradient!`, and the
supplied predictor for the predictor oracles. They do not include the rounding
of `X*β`. They assume round-to-nearest arithmetic with gradual underflow. Let
`eps = eps(Float64)` and let ``e_i = e^{\eta_i}`` be the exact exponential. The
term and derivative bounds hold for an observation whose computed `Float64`
values `exp(ηᵢ)` and `yᵢ*ηᵢ` are both finite.

- Each computed value term differs from ``e_i-y_i\eta_i`` by at most
  ``4\,\mathrm{eps}\,(e_i+\lvert y_i\eta_i\rvert)+2^{-1074}``.
- Each computed derivative ``\hat d_i`` differs from ``d_i=e_i-y_i`` by at most
  ``4\,\mathrm{eps}\,(e_i+y_i)+2^{-1074}``.
- When every computed term ``\hat v_i`` and every partial sum of the value is
  finite, the computed value differs from ``\sum_i\hat v_i`` by at most
  ``\gamma_{n-1}\sum_i\lvert\hat v_i\rvert``, with
  ``\gamma_k=(k\,\mathrm{eps}/2)/(1-k\,\mathrm{eps}/2)``. An empty sum is zero.
- When every ``\hat d_i``, every product ``X_{ij}\hat d_i`` and every partial sum
  of entry ``j`` of the transposed product is finite, entry ``j`` of the gradient
  output differs from ``(X^\top d)_j`` by at most
  ``\sum_i\lvert X_{ij}\rvert\,\lvert\hat d_i-d_i\rvert+\gamma_n\sum_i\lvert X_{ij}\rvert\,\lvert\hat d_i\rvert``
  plus ``2^{-1075}`` for each nonzero exact product ``X_{ij}\hat d_i`` whose
  computed value is subnormal or zero, for any summation order of the
  transposed product.

The bounds are absolute. There is no relative-accuracy statement for a value
term or derivative whose two parts nearly cancel, or for a value whose terms
cancel across observations. Julia documents no ulp bound for `exp`; the
constant four includes a margin for it.

## [Range clauses](@id spec-poissonloss-range)

`exp(ηᵢ)` is finite at ``\eta_i=\log(\texttt{floatmax(Float64)})\approx 709.7827``
and overflows for a finite predictor above it. When `exp(ηᵢ)` overflows, the
value term and the derivative are `Inf`, also when `yᵢ*ηᵢ` overflows. When
`exp(ηᵢ)` is finite and `yᵢ*ηᵢ` overflows, the value term is `-Inf` for
``\eta_i>0`` and `Inf` for ``\eta_i<0``, and the derivative is finite. The loss
does not recover a finite exact term or derivative whose computation
overflows. The exact value term can lie in the finite `Float64` range for
``\eta_i`` up to about 716.4, and the exact derivative for ``\eta_i`` up to
about 710.48. The sum of the terms and the transposed product use ordinary
floating-point arithmetic with their usual rounding and range limits.

**Implementation notes (not guarantees).** One loop computes the requested
quantities from `e = exp(η)`: the value term
`v = ifelse(iszero(y) | (e == Inf), e, e - y*η)` and the derivative
`d = e - y`, with plain accumulation and no `@simd`, `muladd` or `fma`.

## [Endpoint clauses](@id spec-poissonloss-endpoints)

At `η = -Inf`, a zero response gives value term `0` and derivative `0`, and a
positive response gives value term `Inf` and derivative `-y`. At `η = Inf`, the
value term and the derivative are `Inf`. A NaN predictor gives a NaN value term
and a NaN derivative. A response of `-0.0` follows the zero-response cases. At
a finite predictor, no value term or derivative is NaN. Nonfinite entries of a
coefficient vector or predictor are not refused.

## [Failure behavior](@id spec-poissonloss-failures)

The constructor throws `DimensionMismatch` when `length(y) != size(X, 1)` and
`ArgumentError` for a negative, NaN or infinite response. A `weights` or
`offset` keyword, other array or element types, views and sparse designs throw
`MethodError`; a non-one-based `y` or `X` throws `ArgumentError` or
`MethodError`.

An operation throws `DimensionMismatch` for an argument of the wrong length,
`MethodError` or `ArgumentError` for an argument of another element type or
a non-one-based argument, and `ArgumentError` for an overlap under
[Preconditions](@ref spec-poissonloss-preconditions). These checks run
before the call writes, so the outputs and the scratch are unchanged after a
refusal. An argument with several faults gets one of the corresponding
exceptions. A `Scaled` loss runs the checks of the loss before it writes. The
predictor oracles throw `MethodError` for a `Scaled` loss.

## [Limitations and non-guarantees](@id spec-poissonloss-limitations)

The loss does not recover a finite exact term or derivative whose computation
overflows (see [Range clauses](@ref spec-poissonloss-range)). Aggregates of
finite inputs can be nonfinite: the value is NaN when one term is `Inf` and
another `-Inf`; an entry of the transposed product can be NaN where an infinite
``d_i`` meets ``X_{ij}=0`` or where infinite contributions of both signs meet;
`f(β)` with finite `X` and `β` can form an infinite predictor; and a partial sum
of the value or of the transposed product that overflows is not recovered,
also when the exact result is finite.

The loss has no observation weights, no offset field and no second-order
interface, and the API supplies no gradient Lipschitz bound.

## [Clauses owned elsewhere](@id spec-poissonloss-elsewhere)

This section links to the shared owners.

- [Array and precision conventions](@ref numerical-usage)
- [Shared range guarantees](@ref numerical-value-range)
- [Ownership and aliases](../numerical_usage.md#Ownership-and-aliases)
- [Scaling a function](@ref scaling-guide)

## [Evidence](@id spec-poissonloss-evidence)

[`test/poissonloss.jl`](https://github.com/mlakolar/ProximalBase.jl/blob/master/test/poissonloss.jl)
checks the following cases.

- Value terms, derivatives, value and gradient against a 2048-bit BigFloat
  evaluation at the `Float64` predictor, within the bounds above: six sets with
  predictors from `-1000.0` to `700.0` and responses from `0.0` to `1e12`,
  including zero responses at `-745.0` and `-745.2`; the predictor
  ``\log(\texttt{floatmax(Float64)})`` with response `1.0`; a caller-formed
  exposure offset; data edited after construction; and a design entry
  `nextfloat(0.0)` with derivative `0.5`, whose product rounds to zero.
- The subnormal derivative of a zero response at `-745.0`, written to the
  gradient.
- Value terms and derivatives at `η = -Inf` and `Inf` with responses `0.0`,
  `-0.0` and `2.0`; at a NaN predictor with responses `0.0` and `2.0`; at
  `nextfloat(log(floatmax(Float64)))` and `800.0` with response `1.0`; and at
  `(η, y)` equal to `(709.0, 2.6e305)`, `(-2.0, floatmax(Float64))`,
  `(800.0, floatmax(Float64))` and `(800.0, 1e300)`. Each case runs through
  `value_from_predictor`, the fused oracle, `f(β)`, `value_and_gradient` and
  `gradient`.
- A finite value and gradient at ``\log(\texttt{floatmax(Float64)})``.
- Aggregates of finite inputs: a NaN value from terms `Inf` and `-Inf`; a NaN
  gradient entry from an infinite derivative and a zero design entry; a NaN
  gradient entry from infinite derivatives meeting design entries `1.0` and
  `-1.0` in one column; an `Inf` value from an overflowing partial sum; a
  nonfinite gradient entry from overflowing products with a zero exact
  gradient; and an `Inf` value from an overflowing matrix product.
- Central finite differences of `f(β)` against the gradient at one point.
- At one point: equal values from `f(β)`, `value_and_gradient`,
  `value_and_gradient!` and the two predictor oracles; the gradients of
  `gradient`, `value_and_gradient` and the fused oracle within the bounds above;
  and the predictor oracles leaving the scratch and the predictor unchanged.
- Constructor refusals, and acceptance of the responses `-0.0` and `1e300`.
- Refused calls for wrong lengths, element types and axes, and for scratch and
  output overlaps, each followed by a check that the outputs, the scratch, `y`,
  `X`, `β`, `η` and each array argument of the call are unchanged.
- `Scaled` with factors `1.75` and `1/5`, and `MethodError` from the predictor
  oracles for a `Scaled` loss.
- In-place edits of `y` and `X` seen by the next call, and strided views for
  `β`, `g`, `η` and `d`.
- Empty shapes `(0, 3)`, `(2, 0)` and `(0, 0)`.
- Traits and the `lipschitz` refusal.
- Zero warmed allocations of the five operations at one point.
- Product counts of the five operations, and no products from `f(β)`,
  `gradient!`, `value_and_gradient!` and the fused oracle for empty designs.

The [Poisson regression guide](@ref Poisson-regression-loss) contains an
executable example.

## [History](@id spec-poissonloss-history)

The [restoration issue #1130](https://github.com/mlakolar/ProximalBase.jl/issues/1130)
records the objective and scope, and the
[plan comment](https://github.com/mlakolar/ProximalBase.jl/pull/1133#issuecomment-7058634587)
records the numerical decisions and test plan.
