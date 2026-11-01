# [Numerical contract](@id numerics)

!!! note "Status"
    This page states the numerical contract of ProximalBase 0.6. It was
    designed in [pull request #385](https://github.com/mlakolar/ProximalBase.jl/pull/385),
    whose [decision record](https://github.com/mlakolar/ProximalBase.jl/blob/master/docs/design/numerical-contract/decisions.md)
    gives the reasons, the alternatives and the precedents behind each rule.
    Parts are still being implemented; [issue #366](https://github.com/mlakolar/ProximalBase.jl/issues/366)
    tracks the functions that do not yet conform. The changes to earlier
    results are listed in [Compatibility and migration](@ref).

This page states what ProximalBase promises about the numbers it returns:
their types, their accuracy, the decisions it takes on rounded quantities, and
what NaN, `Inf`, zero and infinite parameters and steps mean. It is written
for solver authors, who need to know what a returned `Inf` or `NaN` says, and
for authors of new functions, who are held to the same rules (see
[Writing your own function](@ref)). Each rule has a label, such as T3 or R2,
which tests, issues and the decision record use to refer to it.

## What a mixed-precision call computes

A `Float32` vector with a `Float64` parameter is common: `NormL1(0.1)` applied
to `Float32` data is one. Four cases fix what such a call computes. Powers of
two are written `2^k` and built with `ldexp(one(T), k)`; Julia evaluates the
integer `2^100` as `0`.

| case | call | result | why |
|:---|:---|:---|:---|
| V3 | `prox!(z, SqrNormL2(2^160), Float32[1], 2^-160)`, `Float32` output | `0.5f0` | `γλ = 1` is combined before rounding; rounding `2^160` and `2^-160` separately would give `Inf · 0` |
| V8 | `prox!(z, NormL1(1 - 2^-30), Float32[1], 1.0)`, `Float32` output | `0f0` | the threshold rounds to `1f0`, and the coordinate lies on the threshold of the converted problem |
| V9 | `value_and_prox!` of `SqrNormL2(1.0, [1 + 2^-28])`, `Float32` output | output `1f0`, value `0f0` | the converted center is `1f0`, and the value is the converted function's value at the stored output |
| V10 | `prox!(z, NormL1(1 + 2^-23), Float32[1], 1 - 2^-23)`, `Float32` parameters, `Float64` output | `2^-46` | the result is delivered in `Float64`, so `γλ = 1 - 2^-46` is formed in `Float64` |

Each result therefore has a *model precision*, normally the precision it is
delivered in; a wider argument or wider data raise it (T3). Parameters are
rounded once to it, after being combined, and the rounded values define the
problem that is solved. This is a deliberate choice: one rule, and arithmetic
at the speed of the precision the caller chose. A rounding that would destroy
the *range* a representable result needs is the exception (R3).

## [Scope](@id numerics-scope)

**S1. Types and environment.** Real `Float32` and `Float64` arguments,
outputs, parameters, data and steps, in any mixture, on the CPU with IEEE
round-to-nearest and gradual underflow. `Float16` is rejected in 0.6 (T6).
`BigFloat`, integer arguments and GPU arrays are covered as T7 and the
[device rules](@ref numerics-devices) say. Complex input is rejected (see
[Precision](@ref)). Automatic-differentiation numbers are not covered.

**S2. Per result, not per function.** A guarantee attaches to each result of
an oracle: the value; each entry of a gradient, a Hessian-vector product, a
proximal output, a quadratic model and LLA weights; each value of
[`kkt_violation`](@ref); and each decision (a zero, a tie, membership, an
accepted multiplier, a rank). Values, `kkt_violation` and the weights of a
quadratic model follow the value rule (R1). The array outputs follow R2, or A1
for matrix products.

- *Quadratic models.* A quadratic model is a function the solver evaluates, so
  its guarantee is about the returned model. The rule applies to its
  *effective* coefficients: a finite coefficient is returned even when one of
  its factors overflows, so a Poisson observation with weight `1/8` at
  `η = 710` gives the coefficient `exp(710)/8 ≈ 2.79e307`. A coefficient that
  cannot be represented in the working precision raises the precision
  exception before `q` is written (O6).
- *[`restrict`](@ref).* Where supported, `restrict(f, S)(u) = f(E_S u) + C`,
  with `E_S` embedding by zero and `C` a finite constant. Its gradient is
  `∇f(E_S u)[S]`. Its proximal operator minimizes over the support; it is not
  a slice of the full proximal point. For `f(z) = z'Qz/2` with
  `Q = [2 1; 1 2]`, unit step and argument `(1, 0)`, the full proximal point
  is `(3/8, -1/8)`, and the restricted one is `1/3`.

**S3. Not promised.** Correctly rounded results. Identical bits across
platforms, SIMD widths, `--check-bounds` settings, devices, library versions
or ProximalBase versions (C1). Exact feasibility from an approximate
projection. Accuracy beyond what the conditioning of the problem allows.

**S4. One rule for the package.** Every rule applies to every function,
including data losses, wrappers and functions with a proximal value but no
standalone value. A function that does not meet a rule is a gap, tracked in
#366 and brought under the rule, not a documented limitation. The one
stated limitation is in R1: wrappers around inner functions whose argument
overflows, where no supported transformation exists.

## [Types and precision](@id numerics-types)

Notation, per call, and per block for tuples: `R` is the type of a returned
scalar, `D` the element type of a written array, `W` the working type of the
arithmetic, and `Wacc` the accumulator type of reductions. Stored parameters,
data and caches keep their own types.

**T1. Return and destination types.**

| call | returns | writes |
|:---|:---|:---|
| `f(x)` | `R = float(eltype(x))` | |
| [`value_and_gradient!`](@ref)`(g, f, x)` | `R` of `x` | `eltype(g)` |
| [`gradient!`](@ref)`(g, f, x)` | `g` | `eltype(g)` |
| [`prox!`](@ref)`(y, f, x, γ)` | `y` | `eltype(y)` |
| [`value_and_prox!`](@ref)`(y, f, x, γ)` | `R = float(eltype(y))` | `eltype(y)` |
| `prox`, `gradient`, `value_and_gradient`, `hessian_vec` | new arrays of `float(eltype(x))` | |

For tuples, `R` is the promotion over the blocks, and each written block keeps
its element type. A fixed-width integer argument gives `Float64`, and a
`BigInt` gives `BigFloat`. An integer destination is rejected for an
operation with fractional results.

**T2. Data, parameters and bookkeeping.**

- *Data:* the matrices of the functions that hold data (`LeastSquaresLoss`,
  `GLMLoss` and its variants, [`LogDetLoss`](@ref),
  [`QuadraticFunction`](@ref), `KronQuadratic`, [`IndAffine`](@ref),
  `IndGraph`, `IndPolyhedron`, `LinearImage`), and the vectors in the same
  equations: responses, observation weights, offsets, the linear and constant
  terms of `QuadraticFunction`, and `b` in `IndAffine`'s `Ax = b`.
- *Data also:* the normal and offsets that define a linear set
  ([`IndHyperplane`](@ref), `IndHalfspace`, `IndHyperslab`), its geometry. A
  linear equation then means the same whether it is written with its own
  constructor or as one row of an `IndAffine`.
- *Parameters:* levels, coordinate weights, centers, bounds and steps. A
  `Float64` literal passed as a level is a parameter.
- *Bookkeeping:* internal state such as warm starts and cached bounds. It never
  defines the problem.

Data are never rounded per call: converting a matrix on every call costs
O(np), and BLAS needs one element type. Parameters are rounded once per call
to the model precision (T3). The line is a choice of model, and its effect is
not confined to the last bits. Take the `Float64` normal `(1, 1 + 2^-25, 1)`
with offset `0`, and the `Float32` point `(2^25, -2^25, 0)` with a `Float32`
output. As data, the normal gives the residual `-1`, and the projection's
third coordinate is about `1/3`. Had the normal been rounded to `Float32`, the
point would lie on the rounded plane, and that coordinate would be `0`.

**T3. The model a call solves.** Each result has a *model precision*, the
widest of:

- the precision it is delivered in: `D` for a written array, `R` for a value;
- the precision of the argument it is evaluated at, and for
  [`hessian_vec!`](@ref) also of the direction;
- the precision of the data, for a function that holds data (T2).

The working type `W` of the result is its model precision. The argument and
the data are never rounded below it.

- A parameter is rounded once to the model precision.
- A combined scalar parameter, such as the threshold `γλw`, is formed in
  `promote_type(W, factor types)`, left to right in the order the function
  documents (`γ·λ·w` for separable thresholds), and rounded once to `W`. That
  order, not platform bits, defines ties.
- The rounded values define the problem.
- A wider accumulator or temporary used only for range recovery (R3), or for
  refinement and certification (A3), does not change the problem. A new call
  with a wider destination is a different call, with its own model precision.

The value returned by `value_and_prox!` describes the stored output (O2), so
its model precision is that of `f(y)`: the output's precision, or the data's
if wider.

For a normal result, rounding a parameter once changes it by at most `eps/2`
relative, the smallest change the result's precision can express. This is a
choice of problem, not a claim that the algorithm that follows is backward
stable; that is each function's guarantee (A1). Two calls agree when their
effective parameters convert to the same value, and not necessarily
otherwise. With `γ = 9f0`, the product `0.1·9` rounds to `0.9f0`, but
`0.1f0·9f0` rounds to `nextfloat(0.9f0)`. So at `x = nextfloat(0.9f0)`,
`NormL1(0.1)` returns `2^-24`, and `NormL1(0.1f0)` returns `0`.

A wide argument with a narrow output keeps the argument's digits:
`QuadraticFunction(Float32[1;;], Float32[-1])` at the `Float64` argument
`1 + 2^-28`, written into a `Float32` gradient, gives `2^-28`.

**T4. Wider destination.** When `D` is wider than the argument or the stored
parameters, the result is computed in `D` from the exactly widened inputs,
cached quantities included. Widening a coefficient that was rounded in the
narrower type is not recomputing it; only recomputing in `D` gives the
accuracy of `D`.

**T5. Constructors.** A constructor validates parameters in their own type
before any conversion. It may store them as given or promoted to a common
type, but never narrows them, and never turns a finite positive parameter
into zero or `Inf`.

**T6. Float16.** In 0.6, `Float16` is rejected by role, before anything is
written, whatever the model precision: as an argument, destination, direction
or data. A `Float16` parameter or step, scalar or array, in an otherwise
supported call widens exactly and is accepted. When `Float16` support is
added, a call with `Float16` operands will compute in at least `Float32`
(`W` and `Wacc`) and round at the oracle's boundary. `Float16`'s largest value
is 65504, so a sum of squares of ordinary data overflows in `Float16`
arithmetic.

**T7. BigFloat.** A call computes in the `BigFloat` precision current when it
runs. A cache built for `BigFloat` records `precision(BigFloat)` and is
rebuilt when a later call runs at another precision. The functions that use
LAPACK reject `BigFloat`.

**T8. Bookkeeping that is always `Float64`.** The warm-start multiplier and
slope of a constrained search, and `IndBox`'s cached `bmax`, are seeds or
bookkeeping. A result meets its guarantee whatever their values, including a
warm start shared between calls in different precisions.

## [Overflow and underflow](@id numerics-range)

**R1. Values.** When the value `v` is a normal number in `R`,
`floatmin(R) ≤ |v| ≤ floatmax(R)`, it is returned with the accuracy of the
ordinary computation, even when an intermediate leaves the range of `R`.
Intermediates include a sum, a difference, a square, a product with a fixed
factor such as `1/2`, and a level or weight that is not representable in `R`.

The ordinary expression is evaluated first. Only when its result is out of
range, or a sum of squares could have lost terms to underflow, is the value
recomputed in units of a power of two, with the level applied last. Ordinary
inputs therefore take the ordinary computation, at its speed.

```julia
julia> SqrNormL2(1e-300)([1e200])
5.0e99

julia> (1e-300 / 2) * sum(abs2, [1e200])
Inf
```

The guarantee covers normal values. A subnormal value may be off by about
`floatmin(R)` in absolute terms, for example `0.0` for `nextfloat(0.0)`.
Underflow before a weight that scales the value back into the normal range is
covered: `SqrNormL2(2.0^1000)` at `[2.0^-511; fill(2.0^-538, 32)]` returns its
value to full relative accuracy.

The guarantee has a small cost, accepted by design. The range check after the
ordinary expression adds a few nanoseconds per call: about 2 to 5 ns on
vectors of up to a thousand entries, which is 5 to 35 percent of the cheapest
values, and no measurable cost from a few thousand entries on. The usual
alternative in optimization software is to leave the ordinary expression
unchecked and ask users to scale their data. ProximalBase follows numerical
linear algebra instead, where BLAS `nrm2`, LAPACK and Julia's `norm` pay a
similar fixed cost so that a representable result is never lost to an
intermediate.

*Wrappers.* Two different failures need different remedies.

- *An inner value out of range at a representable input,* as in
  `Scaled(f, α)` with a tiny `α` and an `f` whose value overflows. The
  wrapper recovers its value through an internal scaled-value mechanism, which
  the built-in functions implement: the inner function returns its value as a
  significand and a power-of-two exponent.
- *A transformed argument out of range,* as in [`Translate`](@ref),
  [`Precompose`](@ref) and `LinearImage`.
  `Translate(NormL1(0.25), [-M])([M])`, with `M = floatmax(Float64)`, has
  inner coordinate `2M` and value `M/2`. Once the wrapper has passed `[Inf]`
  on, no format for the returned value can recover it. Recovery needs a
  supported transformation. For norms and other positively homogeneous
  inner functions, `Translate` and `Precompose` scale the argument and use
  homogeneity, so this example returns `M/2`. Other combinations are not
  covered by R1, and #366 lists them.

The scaled-value mechanism is internal in 0.6. A function you write yourself
is covered by R1 when evaluated directly. Inside `Scaled`, it is covered only
while its own value does not overflow.

**R2. Gradients and closed-form proximal outputs.** Each entry of a gradient,
and each entry of a proximal output computed in closed form, follows R1's
rule in `D`. Closed forms include the separable kernels, group and ball
shrinkage, ridge, and closed-form projections. A normal entry is returned with
the accuracy of the ordinary computation, even when an intermediate leaves the
range of `W`. Such an intermediate can be a difference such as `x - c`, a
product, a ratio, a factor such as `1/(1 + γλ)`, or a parameter converted to
`W`. For example, the gradient of centered `SqrNormL2(2^-1022, [-2^1023])` at
`[2^1023]` is `[4.0]`, although `x - c` overflows. Outputs of matrix products,
factorizations and iterative searches follow A1 instead.

**R3. Conversions keep range.** When rounding a parameter, or a combined
expression such as `γλw`, to `W` would overflow, underflow to zero or lose
bits to the subnormal range, while the final result is representable, the
result is computed as if from the unrounded value. V3, `SqrNormL2(2^160)` at
`Float32[2^100]` with a unit step (`2^-60`), and `SqrNormL2(2^800)` with step
`2^800` at `[2^900]` (`2^-700`, all `Float64`) are examples.

- *Ordinary path.* It is taken when no partial operation of the combined
  expression loses range or subnormal bits that a later factor needs. A final
  scalar that is finite and normal is not enough. In weighted `NormL1` with
  `γ = 3·2^-538`, `λ = 2^-538`, `w = 2^1022` at `x = 2^-52`, the exact
  threshold is `3·2^-54`, so the result is `2^-54`. An unchecked `γ·λ` rounds
  `3·2^-1076` up to `2^-1074`; the threshold would then become
  `4·2^-54 = x`, and the coordinate would be set to zero, although both
  thresholds are normal.
- *Recovery path.* It is taken otherwise. For scalar parameters it is chosen
  once per call, before the loop over entries. The method is free: a wider
  scalar type, or a power-of-two scale carried to the last operation. A wider
  native type is not always available: the `2^800` example above is all
  `Float64`.
- *Scope.* A loss of significance that is not a range event follows T3, not
  R3. V8 is ordinary rounding.

**R4. Final range.** An entrywise result of type `T` (`R` for a value, `D`
for a written entry) underflows to a subnormal or to zero only when the
absolute value of its exact value is below `floatmin(T)`. It overflows to
`±Inf` only when its exact value rounds beyond `floatmax(T)` in absolute
value; there is no saturation at `floatmax`. A matrix reconstruction (A5)
follows its normwise bound instead, within which an entry may saturate at
`floatmax`.

## [Accuracy and decisions](@id numerics-accuracy)

**A1. A guarantee per output.** Each function documents three things: the
problem it solves, with parameters converted under T3; the scale of its
error; and its guarantee. Floating-point error, the final rounding to `D` and
iterative termination error are stated separately. Backward stability is
claimed only where it is established. Constants are the same for every
precision; where one grows with the dimension, the function says how.

| output | guarantee |
|:---|:---|
| values | R1, relative error in a scale that accounts for cancellation |
| closed-form gradients and proximal outputs | R2, componentwise relative error for normal entries, or a stated absolute scale where cancellation makes relative error meaningless |
| convex proximal maps by iteration, scalar step | a forward bound in a stated scale, or an optimality or feasibility certificate on the returned point that implies one |
| proximal maps with array steps | the same, in the metric's norm |
| nonconvex proximal maps | which minimizer is returned, and the rule at ties |
| functions that use LAPACK | A5 |
| iterative methods (searches, Dykstra, CG, partial SVD) | the termination tolerance on the returned point, and what happens when it cannot be met |

**A2. One policy for each kind of decision.**

| decision | obligation |
|:---|:---|
| entrywise threshold, clamp or tie on the model's operands | an exact comparison, with the documented tie and no tolerance. Every path (`prox_entry`, `prox_coordinate!`, the whole array, aliased `prox!(x, f, x, γ)`) and every backend decides the same |
| closed form on a computed reduction, such as a group norm | the same exact comparison, on the reduction as computed. Two paths or backends decide the same whenever the margin exceeds the reductions' error bound (A1); within that margin, either result is allowed. This describes the uncertainty, and adds no band to the comparison |
| decision in an iterative search | certified under A3 |
| spectral decision inside its band | returned as computed, under A5 |

At equality the documented branch applies; soft thresholding, for example,
returns zero at `|x| = γλ`. A coupled group is decided as a whole.

**A3. Decisions in searches.** A search, such as the multiplier search of
[`Constrained`](@ref) on a hyperplane, decides when to accept a multiplier,
certify a point, choose a slab bound or fix an entry.

- A decision is certified outside a documented uncertainty band. The band
  rests on the rounding model with no overflow, and on a bound that covers
  every term's error: products, cached normals, sums of fixed entries, the
  offset subtraction and the reduction. What is certified is the stored
  output, or a bound connects it to the searched point.
- Inside the band, the search refines where that is cheap. Otherwise it
  raises the precision exception before writing, so a solver can tell it from
  invalid input. A message that the set misses the domain would be wrong
  there.
- Refinement and certification may widen the arithmetic on the *effective*
  operands of the original call, its parameters as converted under T3. They
  do not change the problem, its range recovery, or the destination being
  certified. In V8, a `Float64` call with the original threshold returns
  `2^-30`; that does not certify the `Float32` problem, whose answer is `0`.
  A solver's retry with a wider destination is a new call.
- Certification covers every exit (warm start, inactive constraint,
  bracketing, refinement, fixing) and applies to `Float64` too. Iterates may
  differ across platforms; certified decisions do not.

**A4. Tolerances.** Four kinds are kept apart: arithmetic uncertainty,
membership tolerance, algorithmic tolerance and the accuracy a user requests.

- *Membership.* The tolerance is `√eps(T)·max(1, |bound|, s)`, where `s` is
  the error scale of the tested quantity: `0` for entries given directly, and
  `‖λ‖∞` for a computed spectrum.
- *Rank.* The threshold is `√eps(T)·σ₁`, relative and scale invariant. It is
  the package's convention for membership-type decisions. Libraries that
  compute a numerical rank use `p·eps·σ₁` instead: `p = min(m, n)` in Julia's
  `rank`, `max(m, n)` in NumPy's `matrix_rank`, and MATLAB's `rank` uses
  `max(size(A))·eps(norm(A))`. `√eps·σ₁` is the larger only while
  `p < 1/√eps`, about 2,896 in `Float32`.
- Each tolerance is evaluated in the type of the quantity it tests, and no
  default tolerance is a fixed number independent of the precision.

**A5. Functions that use LAPACK.** The eigenvalue and singular-value
functions, [`IndStiefel`](@ref) and [`LogDetLoss`](@ref).

- *Error scale.* LAPACK's symmetric eigensolvers and SVD are backward stable.
  Eigenvalues and singular values carry errors of order `p·eps(W)·‖X‖₂`,
  absolute in the scale of the matrix, and eigenvalues closer to a boundary
  than that are not resolved.
- *Three sources of output error,* stated separately: the input perturbation,
  propagated by the map; the evaluation of the scalar map and the
  reconstruction, in the scale of the output; and the storage in `D`. Write
  `Y_W` for the result before storage, `Y_D` for the result delivered, and
  `E_D = Y_D - Y_W`. The storage term is `‖E_D‖_F`, the actual or bounded
  conversion error. `eps(D)·‖Y_W‖_F` is a simplification where it holds, but
  not a bound through gradual underflow: the `Float64` matrix `[3h/4]`, with
  `h = nextfloat(0f0)`, is its own PSD projection, and storing it in
  `Float32` gives `[h]`, an error of `h/4`.
- *Nonexpansive maps.* The projection onto the PSD cone and the proximal maps
  of convex spectral functions are nonexpansive, so the propagated input
  perturbation is at most `c·p·eps(W)·‖X‖_F`. Nonexpansiveness says nothing
  about the other two sources. `LogDetLoss` with `S = 0`, `w = 1` and `γ = 2`
  maps `X = 0` to `√2·I`, which is not representable although `‖X‖ = 0`. The
  bound is `c·p·eps(W)·(‖X‖_F + ‖Y_W‖_F) + ‖E_D‖_F`.
- *Rank truncation.* A projection onto a rank ball is not continuous:
  `diag(1 + δ, 1)` and `diag(1, 1 + δ)` are `δ√2` apart, while their best
  rank-one approximations keep different axes and are about `√2` apart.
  - *Promised:* in the Frobenius norm,
    `‖X - Y_D‖_F ≤ ‖X - X_k‖_F + c·p·eps(W)·‖X‖_F + ‖E_D‖_F`, where `X_k` is
    a best rank-`k` approximation. The rank-one projection of the `Float64`
    matrix `diag(1 + 2^-28, 0)` into a `Float32` output is `diag(1, 0)`: the
    optimal residual is `0`, and the stored residual is `2^-28`.
  - *The stored matrix's rank:*
    `σ_{k+1}(Y_D) ≤ σ_{k+1}(Y_W) + ‖E_D‖₂`. Numerical rank at most `k`, under
    A4's tolerance evaluated in `D`, is promised only when this bound is
    below the cutoff; a bound above it does not prove that the rank exceeds
    `k`. At subnormal magnitudes the storage error is absolute: the rank-one
    `X = h·v·v'`, with `v = (1, 5/8)`, stored in `Float32` can have numerical
    rank two.
  - *The value returned with the projection* (O2) describes the stored
    matrix, with the standalone membership test. When the bound certifies
    membership, the value is `0`, and no second decomposition is needed;
    otherwise the stored matrix's membership is evaluated.
  - *Closeness to a particular `X_k` or subspace* is promised only when the
    gap `σ_k - σ_{k+1}` is well above `eps(W)·‖X‖`.
- *`IndStiefel`.* The stored output is `Y_D = Q + E_W + E_D`, where `Q` is the
  exact polar factor and `E_W` the computation's error. Its Gram error
  `Y_D'Y_D - I` contains the cross terms of `E_W + E_D` with `Q`, and its
  bound includes both. The column `(1, 1)` stored in `Float32` has
  `|y'y - 1|` of order `eps(Float32)`, not `eps(Float64)`. Closeness to the
  exact polar factor depends on the smallest singular value of `X`.
- *Decisions near a boundary* use A4's relative tolerance, never an absolute
  one. The `√eps·max|σ|` zero mask applies only to functions that count
  nonzeros. A decision inside the band is returned as computed, with the
  guarantee stated in the scale of `‖X‖`; this takes precedence over A3.
- *Before LAPACK.* Non-finite entries are detected and never passed to LAPACK.
  The matrix is scaled by a power of two when its spectrum would overflow or
  underflow. LAPACK runs in `Float32` and `Float64` only, and both the
  argument's and the output's element types are checked.

**A6. Reductions.** Sums may be reassociated by SIMD, so their last bits can
change with the SIMD width and the bounds-checking mode. A decision that
depends on a sum follows A2: in a closed form it is an exact comparison on the
computed sum, and in a search it is certified under A3. Compensated or wider
sums are used on recovery paths and where A1 requires them.

## [NaN, infinities and degenerate parameters](@id numerics-special)

**E1. Order of meaning.** First the structure: shapes, types, steps and
required disjoint buffers. A violation raises a `DimensionMismatch` (sizes)
or an `ArgumentError` before anything is written (O6). Then the parameters
that disable or constrain: a zero level, a zero weight, or an infinite weight
where the function admits one (E4). Then the non-finite entries of the
argument (E2, E3). This is an order of precedence, not three passes over the
data.

**E2. NaN.** A NaN entry of the argument is an undefined value, propagated as
IEEE arithmetic propagates it.

- Every result that depends on that entry is NaN: the value, the gradient
  entries, and the proximal entries of its coordinate (separable functions) or
  of its whole coupled block (groups, balls, spectral functions). Results that
  do not depend on it are computed as usual.
- A comparison with NaN never selects a branch. A failed `norm > threshold`
  test must not turn NaN into zeros, and a failed membership test must not turn
  it into `+Inf`. The shortcut "an indicator's value at a projection is `0`"
  does not apply.
- In a sum of terms, NaN takes precedence over `±Inf`.

So `prox(NormL1(1.0), [NaN, 4.0], 1.0)` is `[NaN, 3.0]`, and
`prox(NormL2(1.0), [NaN, 4.0], 1.0)` is `[NaN, NaN]`. Check such results with
`isequal` or `isnan`, since `==` is false whenever NaN is involved. The
functions that use LAPACK detect a NaN entry before decomposing; their value
is NaN, and their output is NaN in every coupled entry.

**E3. Infinite entries.** A `±Inf` entry of the argument is the extended real.
A value takes the function's formula or limit where it is determined: a norm
gives `+Inf`, and an indicator gives `+Inf`, since no set of real vectors
contains a point with an infinite entry. Where it is not determined, it is
NaN: `(x₁ - x₂)²/2` tends to `0` along `(t, t)` and to `1/2` along
`(t, t + 1)`, so at `(Inf, Inf)` it is NaN, as its formula's `Inf - Inf`
gives. Gradients and proximal outputs take the function's documented limit,
and NaN in the dependent entries otherwise. Uncentered ridge, for example,
maps `-Inf` to value `+Inf`, gradient `-Inf` and proximal output `-Inf`.

**E4. Zero and infinite parameters.**

- *Zero level.* A zero level removes the finite part of the function and keeps
  its domain constraints: an indicator's set, infinite coordinate weights, a
  restricted domain. `NormL1(0, [0, Inf])` stays the indicator of
  `{x₂ = 0}`, and `0 * IndBox` stays the box. A function without domain
  constraints becomes the zero function: value `0`, gradient `0`, and a
  proximal step that copies the argument, NaN entries included. Zero scaling
  is rejected when no domain-preserving proximal representation exists:
  `NegLog` at zero scale would be the indicator of the open half-line
  `(0, ∞)`, which has no closest point to `-1`, so `Scaled(NegLog(), 0)`
  throws.
- *Zero coordinate weight.* The coordinate is free.
- *Infinite coordinate weight,* where the function admits one. The coordinate
  is fixed at zero. An infinite level is not an infinite weight unless the
  function says so.
- *NaN levels, weights and bounds* are rejected at construction, and so are
  non-finite coefficients of [`LinearFunction`](@ref) and [`Tilt`](@ref).
- A finite parameter never acquires a zero-mask or hard-weight meaning
  through conversion (R3).

**E5. Exact exceptional results survive precision changes.** An `Inf` or NaN
that is the exact result, such as an indicator outside its set, is kept
through any conversion or recomputation in another precision. A non-finite
intermediate that comes from overflow or cancellation at finite inputs is
recovered under R1–R3.

**E6. Domains and masks.** A finite point outside the domain gives `+Inf` for
the value, as for `NegEntropy` at `-1` or an indicator off its set, and NaN
for the gradient, since no derivative exists there. At a domain boundary the
value is the one-sided limit where it is finite: `x log x` is `0` at `0`.
Small tails stay explicit: the logistic loss at `η = 40` is about `4.25e-18`,
and so is the size of its derivative. A function with observation masks
documents their precedence. `BernoulliLoss` tests a NaN probability before a
zero observation weight.

## [Steps](@id numerics-steps)

**P1. Validity and endpoints.** Negative and NaN steps raise an
`ArgumentError` before anything is written. A zero or infinite step is
supported only where the function names its endpoint, and is rejected before
writing elsewhere. The named endpoints:

- *Separable functions,* entrywise. At `γ = 0` the limit is the projection
  onto the closure of the domain, which is the identity inside it: `NegLog`
  maps `x = -1` to `0`, since its proximal point `(x + √(x² + 4γ))/2` tends
  to `0`. For the nonconvex separable functions with full domain (`MCP`,
  `SCAD`, `NormL0`) it is the identity. At `γ = Inf` the limit, for a convex
  function with a nonempty set of minimizers, is the projection onto that set.
  A function without a finite minimizer, such as `NegLog`, whose proximal
  point at `0` is `√γ`, rejects `γ = Inf`.
- *Scalar steps on indicators,* constrained sets included, where the
  projection is unchanged.
- *Ridge,* at `0` and `Inf`.

Array steps on coupling sets and in the searches of `Constrained` must be
positive and finite in every entry (see
[Steps per entry and diagonal metrics](@ref)).

**P2. Step types.** Steps are validated in their given type. The effective
step, such as `γλ`, is a combined expression under T3 and R3. Every route
treats the step the same way: `prox!`, `value_and_prox!`, [`prox_entry`](@ref)
and the checks of block functions before they write.

**P3. Diagonal metrics that do not fit in `W`.** Some metrics are not
representable in `W` after conversion: the `Float64` steps `[1e-50, 2e-50]`
become zeros in `Float32`, yet define the same projection as `[1, 2]`. Such a
metric is represented as a retained scale times ratios, and the scale enters
the combined expressions (`γλ`, `γνa`). A ratio that underflows `W` is
rejected before writing. Dividing the steps alone would preserve projections
but change penalized maps.

## [Agreement between oracles](@id numerics-oracles)

**O1. The point of `value_and_gradient!`.** It returns what `f(x)` and
`gradient!(g, f, x)` return, each at the argument and in its own model
precision, even when the two differ. With `x = Float32[1]`, a `Float64`
center `1 + 2^-28` and a `Float64` gradient output, the value is `0f0` (the
center rounded to `Float32` is `1f0`) and the gradient is `[-2^-28]`. A value
that overflows does not make a finite gradient invalid: the R2 example has
value `+Inf` and gradient `[4.0]`.

**O2. The point of `value_and_prox!`.** It returns the value at the stored
output, after rounding to `D`, of the problem with parameters converted under
T3. It may reuse intermediates of the proximal computation, and its last bits
may differ from `f(y)` within A1. A false zero, a false feasibility, or a
change of class among finite, `Inf` and NaN is not allowed.

**O3. Named exceptions to O2.**

1. *Approximate projections.* An iterative projection such as
   [`IndIntersection`](@ref) returns `0`, the value at the exact projection.
   That `0` does not certify that the stored point is feasible; test its
   violation separately, with the function's tolerance.
2. *`LogDetLoss`, until 0.6 is complete.* Its value uses the mapped
   eigenvalues, while its stored output can round to a singular matrix, so the
   value can be finite where `f(Y)` is `Inf`. With `S = zeros(2, 2)`, a unit
   step and `X = [0 2^15; 2^15 0]`, the exact proximal point has determinant
   one, but all four entries of a `Float32` output round to `16384f0`. The
   contract's target is the value at the stored output, `+Inf32` there.

**O4. Wrappers that round between stages.** In mixed precision, the array
proximal operators of `Translate` and `Tilt` round to `D` between stages.

- *Lipschitz inner maps.* The difference between `prox!` and entrywise
  `prox_entry` is bounded per wrapper: each stage's rounding error, in that
  stage's scale, is propagated with the stage's amplification factor (one for a
  nonexpansive stage), and the errors add up over the nesting depth.
  `Translate(NormL1(1.0), [1e8])` at `[0.5]` gives `Float32[0]` from `prox!`
  and `1.5` from `prox_entry`, within `eps(Float32)·1e8`.
- *Discontinuous inner maps.* The staged problem and its tie rule are
  promised, not closeness to the unstaged point.
  `Translate(NormL0(0.5), [-2^-26])` at `1.0` with a unit step gives
  `Float32[-2^-26]` with a `Float32` output, since the shifted argument rounds
  to the threshold `1`, which selects zero. With a `Float64` output it gives
  `[1.0]`.

**O5. Identities with conditions.** A restriction may drop an additive
constant, so gradients agree while values differ by that constant. A conjugate
may give a value with its proximal output and have no standalone value.
Neither is a numerical bug.

**O6. Outputs after a failure.** The guarantee depends on the kind of failure.

1. *Validation:* shapes, types, steps and parameters, detected before any
   work. The call raises a `DimensionMismatch` for sizes and an
   `ArgumentError` otherwise, and leaves every caller-owned output unchanged,
   including tuple blocks and iterate storage.
2. *Infeasibility found during a search,* such as a constraint that misses the
   domain. The call raises an `ArgumentError` that names it, before writing.
3. *A result that cannot be certified or represented in the working
   precision:* a search inside its band (A3), a quadratic model whose
   effective coefficients overflow (S2), or a check that a finite input gave a
   non-finite result. The call raises the precision exception before writing.
4. *Failure inside an external library,* such as a LAPACK error. That
   library's exception propagates, and the function documents which outputs
   remain valid. There is no rollback.

A prescribed NaN result is a result, not a failure.

**O7. Aliasing.** `prox!(x, f, x, γ)` works for every function, recovery
paths included; a recovery that needs the original argument does not read
entries it has already overwritten. A partial overlap, such as a shifted view,
is not supported. Gradient and Hessian-product outputs must be separate from
their inputs unless a method documents otherwise (see [Conventions](@ref)).

## [Sparse and symmetric storage](@id numerics-storage)

**M1. Implicit zeros are entries.** An operation on a sparse iterate skips
them only when they contribute nothing. A centered ridge moves an implicit
zero, and `NegLog` is `+Inf` at zero. Since linear coefficients are finite
(E4), a sparse inner product may skip unstored entries.

**M2. Multiplicity.** A `SymmetricSparseIterate` stores each off-diagonal
coordinate once; it counts twice in reductions and is written once. Its
parameters follow the symmetric-matrix rule of [Sparse iterates](@ref).

**M3. Caches.** A cache is computational state, and a result does not depend
on its history. `reset_cache!` and reconstruction follow [Conventions](@ref),
and warm starts are seeds (T8).

## [Devices](@id numerics-devices)

**G1. Same meaning.** On a supported device, an operation has the same output
types and scope of NaN and `Inf` as on the CPU. It makes the same entrywise
decisions, and the same decisions on reductions whenever the margin exceeds
the reductions' error bound (A2).

**G2. Accuracy.** Accuracy is appropriate to the result and the arithmetic
mode. The bits of reductions and fused multiply-adds are not promised to match
across devices.

**G3. Subnormals flushed to zero.** In a device mode that flushes subnormals,
a stored subnormal input may be read as zero, and recovering it is not
promised. An avoidable subnormal intermediate formed from normal operands is
still a gap.

**G4. No fast-math** that lets the compiler ignore NaN or `Inf` in a kernel
that must tell them apart.

**G5. Where it runs.** A risky scalar parameter is handled by a scalar
decision on the host, not by copying the array to the CPU or widening it on
the device. The host fallbacks that exist are listed in [GPU arrays](@ref).

**G6. Unsupported is not inaccurate.** A backend without a method raises an
error. JLArrays checks dispatch and layout, not subnormal behavior, throughput
or synchronization on real hardware.

## [Cost](@id numerics-cost)

**C1. Non-regression.** A change keeps every result within its stated
accuracy (A1), keeps decisions as A2 specifies them, keeps exact values exact
(copies, clamps) and keeps classes (finite, `Inf`, NaN). Last-bit changes
within the accuracy are allowed, and a pull request that changes
all-`Float64` results says so. Comparing bits remains a test technique for a
change meant to leave results alone, not a promise. A function whose
quantitative bound is still pending is checked against independent references
with a stated error budget.

**C2. Ordinary path first.** Recovery runs only on a reliable trigger. A small
fixed cost per call for a guarantee is accepted, as for R1. Extra passes over
the data, allocations and device synchronizations are measured and approved on
their pull request.

**C3. Review thresholds.** A repeatable slowdown above 5% on a representative
workload is investigated and reported. A cost above 10% paid only by narrow
types needs the maintainer's approval. Machinery that only narrow types need
leaves `Float64` timings within the benchmark harness's noise.

**C4. Not used.** A public `robust = true` switch, global precision settings,
custom number types, `BigFloat` in hot loops, and unconditional widening of
whole arrays.

## [What the solver decides](@id numerics-solver)

ProximalBase returns values and points, and says what an exceptional value
means. The stopping and acceptance policy belongs to the solver.

- An `Inf` value at a trial point means the trial is outside the domain, or
  its value genuinely overflows (the R2 example is in the domain, with value
  `+Inf` and gradient `[4.0]`). Neither is evidence that the problem is
  infeasible. A solver that requires finite evaluations tests `isfinite` as
  well as decrease, since `Inf <= Inf` is true.
- A NaN value fails every comparison. Test acceptance positively,
  `f_trial <= bound`, rather than rejecting on `f_trial > bound`, which a NaN
  passes. Bound the retries and report a numerical failure.
- Evaluate a trial in candidate storage, and commit exactly the stored point
  that was checked.
- A positive step that changes no stored entry is stagnation or an exact fixed
  point; no movement alone does not tell them apart. A NaN residual does not
  certify stationarity.

## [Support](@id numerics-support)

The guarantee each family of functions gives, by element type and backend.
"Bound pending" means the family's quantitative bound has not yet been
derived and published; until it is, changes are checked against independent
references with a stated error budget (C1).

| family | guarantee | element types | backends | bound |
|:---|:---|:---|:---|:---|
| values of every function | R1 | `Float32`, `Float64`, `BigFloat` | CPU, GPU where supported | relative, for normal values |
| separable penalties and losses, ridge | R2, A2 entrywise | `Float32`, `Float64`, `BigFloat` | CPU, GPU | pending |
| group norms, balls, closed-form projections | R2, A2 on reductions | `Float32`, `Float64`, `BigFloat` | CPU, GPU | pending |
| linear sets and their searches in `Constrained` | A3 | `Float32`, `Float64` | CPU | pending |
| eigenvalue and singular-value functions, `IndStiefel`, `LogDetLoss` | A5 | `Float32`, `Float64` | CPU; device arrays through the host | pending |
| data losses and quadratic functions | T2, T3, R1, A1 | data and call precision, the wider | CPU; GLM and least squares on GPU | pending |
| wrappers | R1 with the scaled-value mechanism; homogeneity for norms inside `Translate` and `Precompose` | as the inner function | as the inner function | other overflowing-argument combinations not covered |

`Float16` is rejected as an argument, destination, direction or data in 0.6,
and accepted as a parameter or step (T6). Functions that do not yet meet a
rule are listed in [issue #366](https://github.com/mlakolar/ProximalBase.jl/issues/366).

## [How the package checks this](@id numerics-checks)

- *References.* Exact powers of two, built in the intended type with
  `ldexp(one(T), k)`, or `BigFloat` evaluated with a locally scoped precision
  after applying the problem's conversions (T3) and stage boundaries (O4),
  except where R3 requires recovery from the unrounded values. Precision
  starts at 256 bits and is raised until the rounding and classification of
  the reference are stable.
- *Comparisons.* Exact (`===`, or `isequal` where NaN may occur) for exact
  values, zero decisions and ties. ULPs or a justified relative bound for
  smooth, well-conditioned results. A residual or certificate for matrices
  and iterative outputs. Never an absolute `≈` that would accept a zero where
  a small nonzero result is required, as in V10's `2^-46`. Class and type are
  checked separately from the value.
- *Conformance tests* vary the argument, destination, parameter and step
  types, and include signed zeros, subnormals, neighbors of `floatmin` and
  `floatmax`, both infinities, dependent and independent NaNs, and exact and
  adjacent thresholds. `scripts/test.jl` forces `--check-bounds=yes`, so the
  default mode needs its own run. Each `@test_broken` names its issue.
