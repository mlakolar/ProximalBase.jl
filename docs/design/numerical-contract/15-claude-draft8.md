# Numerical contract

*Draft 8, 2026-11-01. The current draft: Draft 7 (`13-claude-draft7.md`)
revised for ChatGPT's review of it (`14-chatgpt-review-of-draft7.md`). Every
recommendation of `05-decisions.md` is adopted; none is approved yet, so every
rule is *proposed*. The changelog and drafting notes do not go into
`docs/src/numerics.md`. Behavior described as current was checked at
`3f90a29`, the code of this branch, except where a check says `d00c7b86`
(master on 2026-11-01), which ran in a separate worktree of master.*

*Changes from Draft 7:*

- *T2 and decision 46: the normals and offsets of linear sets are data
  (option b). `IndHyperplane` and a one-row `IndAffine` define the same
  constraint, which is today's behavior.*
- *A5: the stored matrix's numerical rank is promised only when a bound
  certifies it. The fused value of a rank indicator describes the stored
  matrix, with the standalone membership test (case L5).*
- *R3: the ordinary path is taken only when no partial product loses range or
  subnormal bits, not when the final scalar is normal (case V13).*
- *A3 and T3: refinement and certification may widen the arithmetic, but do
  not change the model.*
- *R1 and decision 45: `Translate` and `Precompose` of homogeneous inner
  functions recover through homogeneity, so W1 returns `M/2`.*
- *S2 and decision 43: the quadratic model's rule applies to its effective
  coefficients. A finite coefficient `exp(710)/8` must be returned (case P3),
  and a mutating call leaves `q` unchanged (case P2m).*
- *Verification: references are evaluated after the model's conversions.*

This page states what ProximalBase promises about the numbers it returns:
their types, their accuracy, the decisions it takes on rounded quantities, and
what NaN, `Inf`, zero and infinite parameters and steps mean. It is the
reference for maintainers, for authors of new functions, and for solver
writers who need to know what a returned `Inf` or `NaN` says.

Each rule has an identifier (T1, R2, E4, ...) for tests, issues and reviews.
A rule states the intended behavior. It has two statuses.

*Decision status.* A rule that rests on a decision of `05-decisions.md` names
it, as "(decision N)". It is **proposed** until the maintainer approves that
decision, and **approved** after. The decisions record at the end lists the
state of each.

*Implementation status,* at the revision above:

- **Holds**: checked.
- **Gap (#N)**: a bug against this page, tracked in issue N. A gap is brought
  under the rule; it is not documented as a limitation.
- **Changes current behavior**: implementing the rule changes a documented or
  observable result; the change ships with a release note.
- **Not tested**: no check has been run.

## What problem does a mixed-precision call solve?

A `Float32` vector and a `Float64` parameter, such as `NormL1(0.1)` applied to
`Float32` data, are common. Four cases fix what such a call computes. Powers
of two are written `2^k` and built with `ldexp(one(T), k)`; Julia evaluates the
integer `2^100` as `0`.

| case | call | result | why |
|:---|:---|:---|:---|
| V3 | `prox!(z, SqrNormL2(2^160), Float32[1], 2^-160)`, `Float32` output | `0.5f0` | combine `γλ = 1` before rounding; rounding `2^160` and `2^-160` separately gives `Inf · 0` |
| V8 | `prox!(z, NormL1(1 - 2^-30), Float32[1], 1.0)`, `Float32` output | `0f0` | the threshold rounds to `1f0`, and the coordinate lies on the threshold of the converted problem |
| V9 | `value_and_prox!` of centered `SqrNormL2(1.0, [1 + 2^-28])`, `Float32` output | output `1f0`, value `0f0` | the converted model's center is `1f0`, and the value is that model's value at the stored output |
| V10 | `prox!(z, NormL1(1 + 2^-23), Float32[1], 1 - 2^-23)` with `Float32` parameters, `Float64` output | `2^-46` | the result's precision is `Float64`, so `γλ = 1 - 2^-46` is formed in `Float64`, not as a `Float32` product |

So each result has a *model precision*, normally the precision it is delivered
in; a wider argument or wider data raise it (T3). Parameters are rounded once
to it, after being combined, and the rounded values define the problem. This is a deliberate choice, for one rule and
native-precision speed. A rounding that would destroy the *range* a
representable result needs is the exception: V4, below, still returns its small
normal result. T3 states the rule.

## Scope

**S1. Covered types and environment.** Real `Float32` and `Float64`
arguments, outputs, parameters, data and steps, in any mixture, on the CPU
with IEEE round-to-nearest and gradual underflow. `Float16` is rejected in
0.6 (T6). `BigFloat`, integer arguments and GPU arrays are covered as T7, the
[device rules](#devices) and the [support matrix](#support-matrix) say.
Complex input is rejected (Precision section of the interface page). Automatic
differentiation numbers are not covered.

**S2. Per result, not per function** (decision 43). A guarantee attaches to
each result of an oracle: the value; each entry of a gradient, a
Hessian-vector product, a proximal output, a quadratic model and LLA weights;
each `kkt_violation`; and each decision (a zero, a tie, membership, an
accepted multiplier, a rank). Values, `kkt_violation` and model weights follow
the value rules (R1); the array outputs follow R2 or, for matrix products,
A1.

- *Quadratic models.* A quadratic model is itself a function the solver
  evaluates, so its guarantee is about the returned model, not only its
  fields (decision 43). The rule applies to its *effective* coefficients,
  not to an intermediate. A coefficient that is finite must be returned, even
  when a factor of it overflows: weight `1/8` at `η = 710` gives
  `exp(710)/8 ≈ 2.79e307` (case P3). When an effective coefficient cannot
  be represented in the working precision, the call raises the dedicated
  exception before writing `q` (O6, cases P2 and P2m).
- *`restrict`.* Where supported, `restrict(f, S)(u) = f(E_S u) + C`, with
  `E_S` embedding by zero and `C` finite and independent of `u`. Its gradient
  is `∇f(E_S u)[S]`. Its proximal operator is the minimizer over the
  support, not a slice of the full proximal point. For `f(z) = z'Qz/2` with
  `Q = [2 1; 1 2]`, unit step and argument `(1, 0)`, the full proximal point
  is `(3/8, -1/8)`, while the restricted one is `1/3` (case Q1). The
  restrictions documented as unsupported throw.

**S3. Not promised.** Correctly rounded results. Identical bits across
platforms, SIMD widths, `--check-bounds` settings, devices, library versions
or ProximalBase versions (decision 38). Exact feasibility from an approximate projection. Accuracy beyond
what the conditioning of the problem allows.

**S4. One rule for the package.** Every rule applies to every function,
including data losses, wrappers and functions with a proximal value but no
standalone value. Functions users write are held to the same rules, and
`ProximalBase.Testing` checks them where it can.

## Types

Notation, per call, and per block for tuples:

- `R`: the scalar return type.
- `D`: the element type of a written array, the destination.
- `W`: the working type of the arithmetic. `Wacc`: the accumulator type of
  reductions.
- Stored parameters, data and caches keep their own types.

**T1. Return and destination types.** Settled by #161, #185 and #238.
**Holds.**

| call | returns | writes |
|:---|:---|:---|
| `f(x)` | `R = float(eltype(x))` | |
| `value_and_gradient!(g, f, x)` | `R` of `x` | `eltype(g)` |
| `gradient!(g, f, x)` | `g` | `eltype(g)` |
| `prox!(y, f, x, γ)` | `y` | `eltype(y)` |
| `value_and_prox!(y, f, x, γ)` | `R = float(eltype(y))` | `eltype(y)` |
| `prox`, `gradient`, `value_and_gradient`, `hessian_vec` | new arrays of `float(eltype(x))` | |

For tuples, `R` is the promotion over the blocks and each written block keeps
its element type. A fixed-width integer argument gives `Float64`; `BigInt`
gives `BigFloat`. An integer destination is rejected for an operation with
fractional results (#366's C7).

**T2. Data, parameters and bookkeeping** (decisions 6 and 46).

- *Data:* the matrices of the functions that hold data (`LeastSquaresLoss`,
  `GLMLoss` and its variants, `LogDetLoss`, `QuadraticFunction`,
  `KronQuadratic`, `IndAffine`, `IndGraph`, `IndPolyhedron`, `LinearImage`),
  and the vectors that enter the same equations: responses, observation
  weights, offsets, the linear and constant terms of `QuadraticFunction`, and
  `b` in `IndAffine`'s `Ax = b`, and the normal and offsets that define a
  linear set (`IndHyperplane`, `IndHalfspace`, `IndHyperslab`), its geometry
  (decision 46). Data are never rounded per call. Converting a matrix on
  every call costs O(np), and BLAS needs one element type. A linear equation
  means the same thing whether it is written with its own constructor or as
  one row of an `IndAffine`.
- *Parameters:* levels, coordinate weights, centers, bounds and steps. They are rounded once to the model precision (T3). A `Float64`
  literal passed as a level is a parameter.
- *Bookkeeping:* internal state such as warm starts and cached bounds (T8).
  It never defines the model.

The line is a choice of model (decision 46), and its effect is not confined to
the last bits; option (b), adopted here, makes the normal data. Take the `Float64` normal `(1, 1 + 2^-25, 1)`, offset `0`, and
the `Float32` point `(2^25, -2^25, 0)` with a `Float32` output.

- As a parameter, the normal rounds to `(1, 1, 1)`. The point lies on the
  rounded plane, and the projection's third coordinate is `0`.
- As data (a one-row `IndAffine`), the residual is `-1`, and the third
  coordinate is about `1/3`.

Today both give `0.3333333` (checked at `3f90a29`). With the normal as data,
both keep that answer, which **holds**. The two algorithms need not give
identical bits. Under T6, a `Float16` normal is data and is rejected. The cost
of the mixed geometry path is to be measured.

*Current gaps.*

- `QuadraticFunction` with `Float32` data and a `Float64` argument computes
  its value through a `Float32` buffer (relative error `7.5e-9`), while its
  gradient is accurate to `Float64`.
- With a `Float32` gradient output it rounds the argument first (V12).

**Gap (new).**

**T3. The model a call solves** (decisions 3, 5 and 6). Each result has a
*model precision*, the widest of:

- the precision it is delivered in: `D` for a written array, `R` for a value;
- the precision of the argument it is evaluated at, and for `hessian_vec!`
  also of the direction;
- the precision of the data, for a function that holds data (T2).

The working type `W` of the result is its model precision. The argument and
the data are never rounded below it.

- A parameter is rounded once to the model precision.
- A combined scalar parameter, such as the threshold `γλw`, is formed in
  `promote_type(W, factor types)`, left to right in the order the family
  documents (`γ·λ·w` for separable thresholds), and rounded once to `W`. That
  order, not platform bits, defines ties.
- The rounded values define the model.
- A wider accumulator or temporary used only for range recovery (R3), or for
  refinement and certification (A3), does not change the model. A new call
  with a wider destination is a different call, with its own model
  precision.

R3 is the only exception: a rounding that loses range.

The scalar of `value_and_prox!` describes the stored output (O2), so its
model precision is that of `f(y)`: the output's precision, or the data's if
wider. It is not the precision of the proximal computation from `x`.

*Why.* This is a deliberate choice of model: one rule, and native-precision
speed. A decimal literal costs nothing extra, since `NormL1(0.1)` runs in
`Float32` on `Float32` data. Rounding a parameter once changes it by at most
`eps/2` relative, the smallest change the result's precision can express.
This is not a claim that the algorithm that follows is backward stable; that
is each family's guarantee (A1).

Two calls agree when their effective parameters convert to the same value.
They need not agree otherwise. With `γ = 9f0`, `0.1·9` rounds to `0.9f0` but
`0.1f0·9f0` rounds to `nextfloat(0.9f0)`, so at `x = nextfloat(0.9f0)`
`NormL1(0.1)` returns `2^-24` and `NormL1(0.1f0)` returns `0` (checked at
`3f90a29`).

*Cases.*

- V8: `0f0`. **Holds.**
- V9: value `0f0`. **Changes current behavior:** master returns `2^-57`.
- V10: `2^-46`. **Changes current behavior:** master returns `0`.
- V11: see O1.
- V12: `QuadraticFunction(Float32[1;;], Float32[-1])` at the `Float64`
  argument `1 + 2^-28` into a `Float32` gradient. The gradient is `2^-28`, a
  normal `Float32`. **Changes current behavior:** master rounds the argument
  first and returns `0`.

*Current gap.* `SqrNormL2` rounds `γ` and `λ` separately, so
`prox!(z, SqrNormL2(2.0^160), Float32[1], 2.0^-160)` gives `NaN` (V3).
**Gap (new).**

**T4. Wider destination.** When `D` is wider than the argument or the stored
parameters, the result is computed in `D` (the model precision under T3) from
the exactly widened inputs (V10),
cached quantities included (decision 5). For example, the hyperplane
coefficient and `‖a‖²` are formed in `D` (#362). Widening a coefficient that
was rounded in the narrower type is not recomputing it: only recomputing in
`D` gives the accuracy of `D`. Functions that hold data follow T2.
**Gap (#362).**

**T5. Constructors.** A constructor validates parameters in their own type
before any conversion (#313). It may store them as given or promoted to a
common type. It never narrows them, and never turns a finite positive
parameter into zero or `Inf`. **Gap (#313, #333).**

**T6. Float16** (decision 7). In 0.6, `Float16` is rejected by role, before
anything is written and independently of the model precision: a `Float16`
argument, destination, direction or data. A `Float16` parameter or step,
scalar or array, in an otherwise supported call widens exactly and is
accepted, subject to the usual parameter checks. The cases: a `Float16`
argument into a `Float64` output is rejected; a `Float32` argument into a
`Float16` output is rejected; `Float32` arrays with a `Float16` level or step
are accepted. The check is on types and costs nothing at run time. Adding
support later is not a breaking change; changing the meaning of existing
support would be. **Changes current behavior:** generic code runs on
`Float16` today without a promise.

*Rule for when support is added.* A call with model precision `Float16`
computes in at least `Float32` (`W` and `Wacc`), and rounds to `Float16` at
the oracle's boundary. `Float16`'s largest value is 65504, so a sum of squares
of ordinary data overflows in `Float16` arithmetic.

**T7. BigFloat.** A call computes in the `BigFloat` precision that is current
when it runs. A cache built for `BigFloat` records `precision(BigFloat)` and
is rebuilt when a later call runs at another precision (#366 D9). The LAPACK
families reject `BigFloat`. **Gap (new).**

**T8. State that is always Float64.** `LinearConstrained`'s warm-start
multiplier and slope and `IndBox`'s `bmax` are seeds or bookkeeping. A result
meets its guarantee whatever their values. A warm start shared between calls
in different precisions is a seed only. Tolerances stored as fixed numbers
fall under A4.

## Finite arithmetic: range

**R1. Values.** Settled by #283. When the value `v` is a normal number in
`R`, `floatmin(R) ≤ |v| ≤ floatmax(R)`, it is returned with the accuracy of
the ordinary computation, even when an intermediate leaves the range of `R`.
Intermediates include a sum, a difference, a square, a product with a fixed
factor such as `1/2`, and a level or weight that is not representable in `R`.

The ordinary expression is evaluated first. Only when its result is out of
range, or a sum of squares could have lost terms to underflow, is the value
recomputed in units of a power of two, with the level applied last. Ordinary
inputs therefore take the ordinary computation, at its speed. A subnormal value
may be off by about `floatmin(R)`. Underflow before a weight that scales the
value back into the normal range is covered.

*Wrappers* (decision 45). There are two different failures, and they need
different remedies.

- *An inner result out of range at a representable input,* as in
  `Scaled(f, α)` with a tiny `α` and an `f` whose value overflows. The
  wrapper recovers its value through a private scaled-value capability: the
  inner function returns its value as a significand and a power-of-two
  exponent. The built-in functions implement it.
- *A transformed argument out of range,* as in `Translate`, `Precompose` and
  `LinearImage`. `Translate(NormL1(0.25), [-M])([M])`, with
  `M = floatmax(Float64)`, has inner coordinate `2M` and value `M/2`. Once
  the wrapper passes `[Inf]`, no return format can recover it, and the inner
  function cannot tell it from a genuine infinity. Recovery needs a
  supported transformation: an algebraic identity, such as homogeneity for
  norms, or an inner method that takes a scaled argument. `Translate` and
  `Precompose` of a positively homogeneous inner function (the norms) recover
  through homogeneity, so W1 returns `M/2` (decision 45, proposed).
  Combinations without a supported transformation are listed as not covered
  in the support matrix, with their issues (#357, #359).

A user-defined inner function is covered through `Scaled` only once the
capability is made public; until then the support matrix lists it.
Meaningful infinities are preserved either way. **Gap (#357, #359).**

Cases V1 and V2 below. **Holds**, with the remaining functions tracked in
#283 and its parts #356–#360. *The cost and the rejected alternative stay as
the interface page states them today.*

**R2. Gradients and closed-form proximal outputs** (decision 11). Each entry of a gradient,
and each entry of a proximal output computed in closed form, follows R1's
rule in `D`. Closed forms include the separable kernels, group and ball
shrinkage, ridge, and closed-form projections. A normal entry is returned
with the accuracy of the ordinary computation, even when an intermediate
leaves the range of `W`. Such an intermediate can be a difference such as
`x - c`, a product, a ratio, a factor such as `1/(1 + γλ)`, or a parameter
converted to `W`. A subnormal entry may be off by about `floatmin(D)`, or be
zero. An entry whose exact value rounds beyond `floatmax(D)` is `±Inf` with
the correct sign. Outputs of matrix products, factorizations and iterative
searches follow A1 instead.

*Examples (all gaps today).*

- V4: `SqrNormL2(2.0^160)`, step 1, at `[ldexp(1f0, 100)]` gives `2^-60`
  (today `0`).
- V5: the gradient of `SqrNormL2(2.0^160)` at `[ldexp(1f0, -80)]` is
  `[ldexp(1f0, 80)]` (today `Inf`).
- V6: `SqrNormL2(2.0^800)`, step `2^800`, at `[2^900]` gives `[2^-700]`, all
  `Float64` (today `0`).
- V7: the gradient of centered `SqrNormL2` with `x - c` overflowing is `[4.0]`
  (today `Inf`).

Also #361: `Berhu`, `HingeLoss`, `ElasticNet`, `NormL1Fused`.
**Gap (#361; SqrNormL2 new).**

*Rejected alternative.* Draft 2's named list of covered families (its "tier
B") would leave every other function without a stated rule. That conflicts
with S4.

**R3. Conversions keep range** (decision 4). When rounding a parameter, or a combined
expression such as `γλw` or `t·a`, to `W` would overflow, underflow to zero,
or lose bits to the subnormal range, while the final result is representable,
the result is computed as if from the unrounded value. This is #366's C3.

- *Ordinary path.* Taken when no partial operation of the combined
  expression loses range or subnormal bits that a later factor needs. A
  final scalar that is finite and normal is not enough. Ordinary left-to-right
  evaluation then defines the effective parameter. An exactly zero factor
  also takes it. The package's `_weight_product` and `_positive_product`
  already check the partial products.
  - *Example* (case V13): weighted `NormL1` with `γ = 3·2^-538`,
    `λ = 2^-538`, `w = 2^1022` at `x = 2^-52`. The exact threshold is
    `3·2^-54`, so the result is `2^-54`. Without the check, `γ·λ` rounds
    `3·2^-1076` up to `2^-1074`, the final threshold becomes `4·2^-54 = x`,
    and the coordinate is set to zero, although both thresholds are normal.
    **Holds** (checked at `3f90a29`).
- *Recovery path.* Taken otherwise. For scalar parameters it is chosen once
  per call, before the loop over entries.
- *Mechanism.* The method is free: a wider scalar type, or a power-of-two
  scale carried to the last operation. V6 shows that a wider native type is
  not always available.
- *Scope.* A loss of significance that is not a range event follows T3, not
  R3 (V8).

Items: #218, #322, #356 (A11, A13, A21, A25), #359 (B09b). PR #369 fixed
#283's `SLOPE` and device group-weight items.
**Gap (listed issues).**

**R4. Final range.** An entrywise result (R1, R2), of type `T` (`R` for a
value, `D` for a written entry), underflows to a subnormal or to zero only
when the absolute value of its exact value is below `floatmin(T)`. It
overflows to `±Inf` only when its exact value rounds beyond `floatmax(T)` in
absolute value.
There is no saturation at `floatmax`. A matrix reconstruction (A5) follows
its normwise bound instead, within which an entry may saturate at
`floatmax`; `LogDetLoss` documents this (decision 39).

## Accuracy and decisions

**A1. A guarantee per output.** Each family documents three things: the
problem it solves, with parameters as converted under T3; the scale of its
error; and its guarantee. Floating-point error, the final rounding to `D` and
iterative termination error are stated separately. Backward stability is
claimed only where it is established. This is #366's C1.

| output | guarantee |
|:---|:---|
| values | R1; relative error in a scale that accounts for cancellation |
| closed-form gradients and proximal outputs | R2; componentwise relative error for normal entries, or a stated absolute scale where cancellation makes relative error meaningless |
| convex proximal maps by iteration, scalar step | a forward bound in a stated scale, or an optimality or feasibility certificate on the returned point that implies one |
| proximal maps with array steps | the same, in the metric's norm |
| nonconvex proximal maps | which minimizer is returned, and the rule at ties |
| LAPACK families | A5 |
| iterative methods (searches, Dykstra, CG, partial SVD) | the termination tolerance on the returned point, and what happens when it cannot be met (#151) |

Constants are the same for every precision. Where a constant grows with `n`,
the family says how.

**A2. One policy per kind of decision** (decisions 14 and 38).

| decision | obligation |
|:---|:---|
| entrywise threshold, clamp or tie on the model's operands | exact comparison, with the documented tie and no tolerance; every path (entrywise, coordinate, whole-array, aliased) and every backend decides the same |
| closed form on a computed reduction (a group norm) | the same exact comparison, on the reduction as computed. Two paths or backends decide the same whenever the margin exceeds the reductions' error bound (A1); within that margin, either result is allowed. This describes the uncertainty, and adds no band to the comparison |
| decision in an iterative search | certified under A3 |
| spectral decision inside its band | returned as computed, under A5 |

A coupled group is decided as a whole. **Holds** for the entrywise cases
checked (V8 on all four paths).

**A3. Decisions on rounded sums and searches.** This is #366's C5.

- *Decisions.* Accepting a multiplier, certifying a point, membership,
  choosing a slab bound, fixing an entry.
- *Certified region.* A decision is certified outside a documented
  uncertainty band. The band rests on the rounding model with no overflow,
  and on a bound that covers every term's error: products, cached normals,
  sums of fixed entries, the offset subtraction and the reduction.
- *Inside the band* (decision 18). Refine where that is cheap. Otherwise
  throw, before writing, an exception of a dedicated type saying that the
  result cannot be certified in the working precision (decision 31), so a
  solver can tell it from invalid input and retry in a wider type.
  Refinement and certification may widen the arithmetic on the *effective*
  operands of the original call: its parameters as converted under T3. They
  do not change the model, its range-recovery semantics or the destination
  being certified. In V8, a `Float64` call with the original threshold
  returns `2^-30`; that does not certify the `Float32` model, whose answer is
  `0`. A solver's retry with a wider destination is a new call, governed by
  its own model precision. A message
  that the set misses the domain would be wrong there.
- *Scope.* Iterative searches. Closed-form decisions follow A2, and in-band
  LAPACK decisions follow A5.
- *What is certified.* The stored output, or a bound connects it to the
  searched point.
- *Coverage.* Every exit: warm start, inactive constraint, bracketing,
  refinement and fixing. It applies to `Float64` too.
- *Reproducibility.* Iterates may differ across platforms; certified
  decisions do not.

The mechanism (always Float64, adaptive widening, or compensated Float32
sums; #366 D1) is chosen by measurement in the implementing PR, not by this
page. **Gap (#151, #257).**

**A4. Tolerances.** Four kinds are kept apart: arithmetic uncertainty,
membership tolerance, algorithmic tolerance and accuracy the user requests.

- *Membership.* The tolerance is `√eps(T)·max(1, |bound|, s)`, where `s` is
  the error scale of the tested quantity: `0` for entries given directly, and
  `‖λ‖∞` for a computed spectrum (#283 part 2a).
- *Rank.* The threshold is `√eps(T)·σ₁`, relative and scale invariant
  (decision 17). This is the package's convention for every
  membership-type decision. Other libraries' numerical-rank thresholds are
  `p·eps·σ₁`: `min(m, n)` in Julia's `rank`, `max(m, n)` in NumPy's
  `matrix_rank`, and `max(size(A))·eps(norm(A))` in MATLAB's `rank`. `√eps·σ₁`
  is the larger only while `p < 1/√eps`, about 2,896 in `Float32`. It answers
  the membership question every indicator answers, whether the matrix lies
  within tolerance of the rank-`k` set, with the same `√eps` as the
  membership tolerance above.
- *Type.* Each tolerance is evaluated in the type of the quantity it tests.
- *Defaults.* No default is a fixed number independent of the precision.

Known gaps: `_largest_eigenvalue(...; rtol = 1e-10)`, `IndIntersection`'s
default `tol = 1e-10`, the absolute `Testing.test_kkt(...; tol = 1e-8)`, and
the absolute eigenvalue tolerances of #307 (possibly fixed by PR #375; not checked).
**Gap (#307, new).**

**A5. LAPACK families.** This is #366's C5b. It covers the eigenvalue and
singular-value families, `IndStiefel` and `LogDetLoss`.

- *Error scale* (decision 19). LAPACK's symmetric eigensolvers and SVD are
  backward stable. Eigenvalues and singular values carry errors of order
  `p·eps(W)·‖X‖₂`, absolute in the scale of the matrix. Eigenvalues closer
  to a boundary than that are not resolved.
- *Three sources of output error,* stated separately:
  1. the input perturbation, propagated by the map;
  2. the evaluation of the scalar map and the reconstruction, in the scale of
     the output `‖Y‖`;
  3. the rounding to `D`.

  For the PSD projection and the proximal maps of convex spectral functions,
  which are nonexpansive, (1) is at most `c·p·eps(W)·‖X‖_F`. Nonexpansiveness
  says nothing about (2) and (3). `LogDetLoss` with `S = 0`, `w = 1`,
  `γ = 2` at `X = 0` maps to `√2·I`, which is not representable even though
  `‖X‖ = 0`. The bound is `c·p·eps(W)·(‖X‖_F + ‖Y‖_F) + eps(D)·‖Y‖_F`.
- *Rank truncation.* A projection onto a rank ball is not continuous:
  `diag(1 + δ, 1)` and `diag(1, 1 + δ)` are `δ√2` apart, while their best
  rank-one approximations keep different axes and are about `√2` apart.
  - *Promised:* in the Frobenius norm,
    `‖X - Y‖_F ≤ ‖X - X_k‖_F + c·p·eps(W)·‖X‖_F + ‖E_D‖_F`, where `X_k` is a
    best rank-`k` approximation and `E_D` is the error of storing the
    result in `D`.
  - *The stored `Y`.* Its first discarded singular value is bounded by the
    pre-write rank defect plus the storage perturbation:
    `σ_{k+1}(Y) ≤ σ_{k+1}(Y_W) + ‖E_D‖₂`, where `Y_W` is the result before
    the write. Numerical rank at most `k`, under A4's tolerance evaluated in
    `D`, is promised only when this bound is below the cutoff. A bound above
    the cutoff does not prove the rank exceeds `k`.
  - *Subnormal example* (case L5): `X = h·v·v'` in `Float64`, with
    `h = nextfloat(0f0)` and `v = (1, 5/8)`, is rank one. Stored in
    `Float32`, it becomes `[2 1; 1 1]·h` (checked at `d00c7b86`), whose
    singular values have ratio `0.146`, far above the cutoff `3.45e-4`. At
    subnormal magnitudes the storage error is absolute, not relative.
    ChatGPT's review computed `[1 1; 1 0]·h` for an exact projection; the
    package's projection differs before the write, and the conclusion is the
    same.
  - *The fused value of a rank indicator* (O2) describes the stored matrix,
    with the standalone membership test. When the bound above certifies
    membership, the value is `0`, so no second decomposition is needed on
    ordinary inputs. Otherwise the stored matrix's membership is evaluated.
    **Gap (new):** at `d00c7b86`, `value_and_prox!` of `IndBallRank(1)` at
    L5 returns `0` while `IndBallRank(1)` of the stored matrix is `Inf`.
  - *A destination-rounding example* (case L3): the rank-one projection of
    the `Float64` matrix `diag(1 + 2^-28, 0)` into a `Float32` output is
    `diag(1, 0)`. The optimal residual is `0`, but the stored residual is
    `2^-28` (checked at `3f90a29`).
  - *Promised only under a gap:* closeness to a particular `X_k` or subspace,
    when `σ_k - σ_{k+1}` is well above `eps(W)·‖X‖`.
- *`IndStiefel`.* The stored output is `Y = Q + E_W + E_D`, where `Q` is the
  exact polar factor, `E_W` the computation's error and `E_D` the storage
  error. Its Gram error `Y'Y - I` contains the cross terms of `E_W + E_D`
  with `Q`, and the bound is stated with both.
  - *Example* (case L4): the column `(1, 1)` stored in `Float32` gives
    `|y'y - 1| = 5.0e-8`, evaluated in higher precision (checked at
    `3f90a29`), of order `eps(Float32)`, not `eps(Float64)`.
  - Closeness to the exact polar factor depends on the smallest singular
    value of `X`.
- *Testing.* The decomposition's quality before the write and the
  certificate for what was written are tested separately. The storage term
  is stated with the actual or bounded norm of the conversion error, which
  avoids a uniform relative-error claim for subnormal entries.
- *Precedence.* A decision inside the band is returned as computed (below).
  This takes precedence over A3's refine-or-throw for these families.
- *Decisions near a boundary.* These use the relative tolerance of A4 (#240,
  PR #303), never an absolute one. The `√eps·max|σ|` zero mask applies only
  to functions that count nonzeros (#283).
- *Before LAPACK.* In the working type, non-finite entries are detected and
  never passed to LAPACK (#241). The matrix is scaled by a power of two when
  its spectrum would overflow or underflow (#165, #196).
- *Types.* LAPACK runs in `Float32` and `Float64` only. A `Float32` decision
  inside the band is returned as computed, with the guarantee stated in the
  scale of `‖X‖` (#366 D10, option a). Recomputing in `Float64` is added only
  for a measured case.
- *Element-type checks.* Both the argument and the output are checked.
  **Gap (new):** the proximal operators check only the output, so a
  `Float16` argument into a `Float64` output is accepted.

**A6. Reductions.** Sums may be reassociated by SIMD, so their last bits can
change with the SIMD width and the bounds-checking mode. A decision that
depends on such a sum follows A2: in a closed form it is an exact comparison
on the computed sum, and in an iterative search it is certified under A3. Compensated or wider sums are
used on recovery paths and where A1 requires them, and are listed in the
table of wider internals.

## Special values

**E1. Order of meaning.**

1. *Structure.* Shapes, types, steps and required disjoint buffers are
   checked first. A violation raises a `DimensionMismatch` (sizes) or an
   `ArgumentError` before anything is written (O6).
2. *Parameters that disable or constrain.* A zero level, a zero weight, or an
   infinite weight where the family admits one (E4).
3. *Non-finite entries of the argument.* E2 and E3.

This is an order of precedence, not three passes over the data.

**E2. NaN** (decision 20).

- *Meaning.* A NaN entry of the argument is an undefined value, propagated as
  IEEE arithmetic propagates it.
- *What propagates.* Every result that depends on that entry is NaN: the
  value, the gradient entries, and the proximal entries of its coordinate
  (separable functions) or of its whole coupled block (groups, balls,
  spectral functions). Results that do not depend on it are computed as
  usual.
- *Comparisons.* A comparison with NaN never selects a branch. A failed
  `norm > threshold` test must not turn NaN into zeros. A failed membership
  test must not turn it into `+Inf`. The indicator shortcut "value at a
  projection is 0" does not apply.
- *Sums.* In a sum of terms, NaN takes precedence over `±Inf`.

Status:

- `isequal(prox(NormL1(1.0), [NaN, 4.0], 1.0), [NaN, 3.0])`. **Holds.** (`==` is
  false whenever NaN is involved, so tests use `isequal` or `isnan`.)
- `prox(NormL2(1.0), [NaN, 4.0], 1.0)` gives `[0, 0]`, with value `0` from
  `value_and_prox!`. The target is `[NaN, NaN]`. **Gap (new).**
- `prox(GroupNormL2(1.0, [1:1, 2:2]), [NaN, 4.0], 1.0)` gives `[0, 3]`. The
  target is `[NaN, 3]`. **Gap (new).**
- The values of indicators and of `NegEntropy` at NaN, and the spectral
  values at NaN entries, change from `Inf` to `NaN`. The same goes for
  `value_and_prox!` of indicators at NaN, which returns `0` today (#349).
  **Changes behavior.**

**E3. Infinite entries.** A `±Inf` entry of the argument is the extended
real.

- *Values* (decision 21). A value takes the function's formula or limit where
  it is determined. A norm gives `+Inf`. An indicator gives `+Inf`, since no
  set of real vectors contains a point with an infinite entry. Where the value
  is not determined it is NaN: `(x₁ - x₂)²/2` tends to `0` along `(t, t)`
  and to `1/2` along `(t, t + 1)`, so at `(Inf, Inf)` it is NaN, as its
  formula's `Inf - Inf` gives.
- *Gradients and proximal outputs.* They take the family's documented limit,
  and NaN in the dependent entries otherwise. For example, uncentered ridge
  with finite positive `λ` and `γ` maps `-Inf` to value `+Inf`, gradient
  `-Inf` and proximal output `-Inf`.
- *Limits.* No convenient limit is chosen where the limit depends on the
  direction of approach.

**E4. Zero and infinite parameters.**

- *Zero level* (decisions 22 and 40). A zero level removes the finite part
  of the function and keeps its domain constraints: an indicator's set,
  infinite coordinate weights, a restricted domain. This is what `Scaled` and
  the calculus page do today, and what the infinite-weight rule of PR #300
  requires. `NormL1(0, [0, Inf])` stays the indicator of `{x₂ = 0}`, and
  `0 * IndBox` stays the box. A function without domain constraints becomes
  the zero function: value `0`, gradient `0`, and a proximal step that copies
  the argument, reading it only to copy it, NaN entries included (S2). Zero
  scaling is rejected when no domain-preserving proximal representation
  exists, as `Scaled` does today. `NegLog` at zero scale would be the
  indicator of the open half-line `(0, ∞)`, which has no closest point to
  `-1`, so `Scaled(NegLog(), 0)` throws (checked at `3f90a29`). That is
  different from P1's limit of `NegLog`'s proximal point as a positive step
  tends to zero.
  - **Gap (new):** `gradient!` of `SqrNormL2(0.0)` at `[NaN]` gives `[NaN]`
    while its value is `0`.
  - **Changes current behavior:** `NuclearNorm(0.0)` at a NaN entry returns
    `Inf` today.
- *Zero coordinate weight.* The coordinate is free.
- *Infinite coordinate weight,* where the family admits one. The coordinate
  is fixed at zero. Case S4. **Holds.**
- *Infinite level.* It is not an infinite weight unless the family says so.
- *NaN levels, weights and bounds.* Rejected at construction.
- *Conversion.* A finite parameter never acquires a zero-mask or hard-weight
  meaning through conversion (R3).

**E5. Exceptional results survive precision changes; failed arithmetic does
not.** This is #366's C4b. An `Inf` or `NaN` that is the exact result, such as
an indicator outside its set, is kept through any conversion or recomputation
in another precision (#350). A non-finite intermediate that comes from
overflow or cancellation at finite inputs is recovered under R1–R3.
**Gap (#350).**

**E6. Domains and masks.**

- *Outside the domain.* A finite point outside the domain gives `+Inf` for
  the value: `NegEntropy` at `-1`, an indicator off its set. Its gradient is
  NaN, since no derivative exists there; `LogDetLoss` already does this
  (decision 44).
- *Boundary.* At a domain boundary, one-sided limits are used where they are
  finite; `x log x` is `0` at `0`.
- *Small tails.* A finite tail stays explicit: the logistic loss at `η = 40`
  is about `4.25e-18`, and so is the size of its derivative. **Holds.**
- *Masks.* A family with observation masks documents its precedence.
  `BernoulliLoss` tests a NaN probability before a zero observation weight.
  At a non-NaN input, a zero weight contributes zero value and gradient and
  leaves the point unchanged in the proximal step.

## Steps

**P1. Validity and limits.** The current interface rule, completed by #311.

- *Limits* (decision 26). A zero or infinite step is supported only where the
  family names its endpoint; elsewhere it is rejected before writing. The
  named endpoints are:
  - for separable functions, entrywise. At `γ = 0` the limit is the
    projection onto the closure of the domain, which is the identity inside
    it: `NegLog` maps `x = -1` to `0`, since its proximal point
    `(x + √(x² + 4γ))/2` tends to `0`. For the nonconvex separable functions
    with full domain (`MCP`, `SCAD`, `NormL0`) it is the identity. At
    `γ = Inf` the limit, for a convex function with a nonempty set of
    minimizers, is the projection onto that set, the closest minimizer. A
    family without a finite minimizer, such as `NegLog`, whose proximal point
    at `0` is `√γ`, rejects `γ = Inf`.
  - for scalar steps on indicators, constrained sets included, the
    projection is unchanged;
  - for ridge, at `0` and `Inf`.
- *Invalid steps.* Negative and NaN steps raise an `ArgumentError` before
  writing. **Gap (#311):** `prox(NormL1(1.0), [0.5, 0.5], -1.0)` gives
  `[1, 1]`.
- *Array steps.* On the coupling sets and in the multiplier searches of
  `Constrained`, every entry must be positive and finite, checked before
  writing (case S7). **Holds.** Degenerate diagonal metrics need not have a
  unique limiting projection. `IndIsotonic` also requires a representable
  ratio of its largest to its smallest step.

**P2. Step types.** Steps are validated in their given type. The effective
step, such as `γλ`, is a combined expression under T3 and R3. Every route
treats the step the same way: `prox!`, `value_and_prox!`, `prox_entry` and
the block pre-write checks (#366 C6).

**P3. Diagonal metrics that do not fit in `W`.** Some metrics are not
representable in `W` after conversion: the `Float64` steps `[1e-50, 2e-50]`
become zeros in `Float32`, yet define the same projection as `[1, 2]`. Such a
metric is represented as a retained scale times ratios. The scale enters the
combined expressions (`γλ`, `γνa`), as #257 does with `μ = max(γ)·ν`. A ratio
that underflows `W` is rejected before writing (#366 D7). Dividing the steps
alone would preserve projections but change penalized maps. **Gap (#257).**

## Agreement between oracles

**O1. The gradient's point** (decision 5). `value_and_gradient!` returns what
`f(x)` and `gradient!(g, f, x)` return, each at the argument and in its own
model precision (T3), even when the two precisions differ. A fused call is only
an optimization.

- V11: `x = Float32[1]`, a `Float64` center `1 + 2^-28` and a `Float64`
  gradient output give the value `0f0` (the `Float32` model's center is `1f0`)
  and the gradient `[-2^-28]`. **Changes current behavior:** master returns
  the value `2^-57`.
- A value that overflows does not make a finite gradient invalid. V7 gives
  value `+Inf` and gradient `[4.0]`. **Gap (new; the gradient half).**

**O2. The proximal value's point.** `value_and_prox!` returns the value at the
stored output, after rounding to `D`. It may reuse intermediates of the
proximal computation, and its last bits may differ from `f(y)` within A1. A
false zero, a false feasibility, or a change of class among finite, `Inf` and
NaN is not allowed. The value is that of the problem with parameters
converted under T3. Case V9: the center `1 + 2^-28` rounds to `1f0`, the
stored output is `1f0`, and the value is `0`. **Changes behavior:** master
returns `2^-57`, the value of the unconverted problem.

**O3. Named exceptions to O2.** There are two. Under E2 the spectral `Inf`
at non-finite inputs is no longer one.

1. *Approximate projections* (decision 29). An iterative projection such as
   `IndIntersection` returns `0`, the value at the exact projection. That `0`
   does not certify that the stored point is feasible; its violation is
   tested separately, with the family's tolerance.
2. *`LogDetLoss`* (decision 39). Its value uses the mapped eigenvalues,
   while its stored output can round to a singular matrix.
   - *Fixture.* `S = zeros(2, 2)`, `w = γ = 1`, `X = [0 a; a 0]` with
     `a = 2^15`, and a `Float32` output. The exact proximal point has
     diagonal `√(a² + 4)/2` and off-diagonal `a/2`. Its determinant is one,
     so its value is `0`. All four entries round to `16384f0`, so the stored
     matrix is singular.
   - *Current behavior* (checked at `3f90a29`): `value_and_prox!` returns `0`
     while `f(Y) == Inf`.
   - *0.6 target* (decision 39, proposed): `+Inf32`, the value at the stored
     output. Its control is the same fixture with a `Float64` output, whose
     stored matrix is positive definite.
   - *Status:* a documented transitional exception until decision 39 is
     approved and implemented, not a verified bug. A small eigenvalue alone
     is not a reason to call a stored matrix singular: `diag(1, 2^-60)` is
     positive definite in `Float64`.

**O4. Wrappers that round between stages** (decision 30). In mixed precision,
the array proximal operators of `Translate` and `Tilt` round to `D` between
stages.

- *Lipschitz inner maps.* The difference between `prox!` and entrywise
  `prox_entry` is bounded per wrapper. Each stage's rounding error, in that
  stage's scale, is propagated with the stage's amplification factor, which is
  one for a nonexpansive stage, and the errors add up over the nesting depth.
  Pinned case: `Translate(NormL1(1.0), [1e8])` at `[0.5]` gives `Float32[0]`
  from `prox!` and `1.5` from `prox_entry`, within `eps(Float32)·1e8`.
- *Discontinuous inner maps.* The staged model and its tie rule are promised,
  not closeness to the unstaged point. `Translate(NormL0(0.5), [-2^-26])` at
  `1.0`, step 1, gives `Float32[-2^-26]` with a `Float32` output: the shifted
  argument rounds to the threshold `1`, which selects zero. It gives `[1.0]`
  with a `Float64` output (checked at `3f90a29`).

**Gap (new):** today the docs claim no bound.

**O5. Identities with conditions.** A restriction may drop an additive
constant, so gradients agree while raw values differ by that constant. A
conjugate may give a value with its proximal output and have no standalone
value. Neither is a numerical bug.

**O6. Outputs after a failure** (decisions 31, 41 and 42). The guarantee
depends on the kind of failure:

1. *Validation:* shapes, types, steps and parameters, detected before any
   work. The call raises a `DimensionMismatch` for sizes and an
   `ArgumentError` otherwise, and leaves every caller-owned output unchanged,
   including tuple blocks and iterate storage. Settled. **Gap (#323, #371;
   #239 may be done).**
2. *Infeasibility found during a search,* such as a constraint that misses the
   domain (#323). The call raises an `ArgumentError` that names the
   infeasibility before writing, and outputs are unchanged.
3. *Certification failure in a search* (A3). The call raises the dedicated
   exception before writing, and outputs are unchanged. Tests include a late
   failure in a composed or in-place call, not only one on the first block.
4. *A result that cannot be represented in the working precision.* This
   covers an arithmetic stopgap, a check that a finite input gave a
   non-finite result (#244's check in `Conjugate`), until R2 and E5 make it
   unnecessary. It also covers a quadratic model whose coefficients overflow
   (decision 43). The call raises the dedicated exception before writing.
5. *Failure inside an external library,* such as a LAPACK error. That
   library's exception propagates, and the family documents which outputs
   remain valid. There is no rollback.

A prescribed NaN result is a result, not a failure.

**O7. Aliasing.** The settled part; #367 owns the rest.

- `prox!(x, f, x, γ)` works for every function, recovery paths included. A
  recovery that needs the original argument must not read entries it has
  already overwritten.
- A partial overlap, such as a shifted view, is not supported (#318).
- Gradient and Hessian-product outputs are separate from their inputs unless
  a method documents otherwise (#321, #324).

## Storage

**M1. Implicit zeros are entries.** An operation on a sparse iterate skips
them only when they contribute nothing. A centered ridge moves an implicit
zero, and `NegLog` is `+Inf` at zero. The coefficients of `LinearFunction` and
`Tilt` must be finite, which is checked at construction (decision 47), so a
sparse dot may skip unstored entries (#341).

**M2. Multiplicity.** A `SymmetricSparseIterate` stores each off-diagonal
coordinate once. It counts twice in reductions and is written once. The
symmetric-parameter rule of #142 applies. **Holds** (`SqrNormL2(1.0)` of
`[0 3; 3 0]` is `9`).

**M3. Caches.** A cache is computational state. A result does not depend on
the history of the cache. `reset_cache!` and reconstruction follow the
Conventions section of the interface page. Warm starts are seeds (T8).

## Devices

**G1. Same semantics.** On a supported device, an operation has the same
output types and scope of NaN and `Inf` as on the CPU. It makes the same
entrywise decisions, and the same decisions on reductions whenever the margin
exceeds the reductions' error bound (A2).

**G2. Accuracy.** Accuracy is appropriate to the result and the arithmetic
mode. Bits of reductions and fused multiply-adds are not promised to match
across devices.

**G3. Subnormals flushed to zero.** In a device mode that flushes
subnormals, a stored subnormal input may be read as zero; recovering it is not
promised. An avoidable subnormal intermediate formed from normal operands is
still a gap.

**G4. No fast-math** that lets the compiler ignore NaN or `Inf` in a kernel
that must distinguish them.

**G5. Where it runs.** A risky scalar parameter is handled by a scalar
decision on the host (#322), not by copying the array to the CPU or widening
it on the device. Host fallbacks that exist are documented (the interface
page's GPU section).

**G6. Unsupported is not inaccurate.** A backend without a method raises an
error; it is not reported as inaccurate. JLArrays checks dispatch and layout.
It does not check subnormal behavior, throughput or synchronization on real
hardware.

## What the solver decides

ProximalBase returns values and points and says what an exceptional value
means. The stopping and acceptance policy belongs to the solver.

- An `Inf` value at a trial point means the trial is outside the domain, or
  its value genuinely overflows. In V7 the point is in the domain, the value is
  `+Inf`, and the gradient is `[4.0]`. Rejecting such a trial is the solver's
  choice, but neither case is evidence that the problem is infeasible. A
  solver that requires finite evaluations tests `isfinite` as well as
  decrease, since `Inf <= Inf` is true.
- A NaN value fails every comparison. Test acceptance positively,
  `f_trial <= bound`, rather than rejecting on `f_trial > bound`, which a NaN
  passes. Bound the retries and report a numerical failure.
- Evaluate a trial in candidate storage and commit exactly the stored point
  that was checked.
- A positive step that changes no stored entry is stagnation or an exact
  fixed point; no movement alone does not distinguish them. A NaN residual does not
  certify stationarity.

## Cost

**C1. Non-regression** (decision 38). A change keeps every result within its
stated accuracy (A1), keeps decisions as A2 specifies them, keeps exact values
exact (copies and clamps), and keeps classes (finite, `Inf`, NaN). A family
whose quantitative bound is still pending is checked against independent
references with a stated error budget, not against a blanket `isapprox`. Last-bit changes
within the accuracy are allowed; a PR that changes all-`Float64` results says
so. Bitwise comparison remains a test technique for a change meant to leave
results alone, not a promise. Machinery that only narrow types need leaves
`Float64` timings within the benchmark harness's noise.

**C2. Ordinary path first.** Recovery runs only on a reliable trigger. A
small fixed per-call cost for a guarantee is accepted, as for R1: about 2 to
5 ns on vectors of up to a thousand entries. Extra passes over the data,
allocations and device synchronizations are measured and approved on their
PR.

**C3. Review trigger.** A repeatable slowdown above 5% on a representative
workload is investigated and reported. A narrow-type cost above 10% needs the
maintainer's sign-off (#366 D2). Measurements follow the benchmark protocol
(`benchmark/README.md`).

**C4. Not used.** A public `robust = true` switch, global precision settings,
custom numeric types, `BigFloat` in hot loops, and unconditional widening of
whole arrays.

## Verification

- **References.** Exact powers of two, built in the intended type with
  `ldexp(one(T), k)`, or `BigFloat` evaluated with a locally scoped precision
  *after* applying the model's conversions (T3) and stage boundaries (O4),
  except where R3 requires range recovery from the unrounded values. A raw
  `BigFloat` evaluation on every stored parameter targets the wrong model in
  V8 and V9. Start at 256 bits and raise the precision
  until the rounding and classification of the reference are stable.
- **Comparisons.** Exact (`===`) for exact values, zero decisions and ties.
  ULPs or a justified relative bound for smooth, well-conditioned results. A
  residual or certificate for matrices and iterative outputs. Never an
  absolute `≈` that would accept a zero where a small nonzero result is
  required, as in V4 (`2^-60`) and V10 (`2^-46`). Class and type are checked
  separately from the value. Where NaN may occur, compare with `isequal` or
  `isnan`.
- **Powers of two.** Build them with `ldexp(one(T), k)`. Julia evaluates the
  integer `2^100` as `0`, and `Float32[2^100]` is `Float32[0]`.
- **Conformance tests.** One file per family: separable penalties; sets and
  projections; searches; wrappers; spectral; losses. Each file varies the
  argument, destination, parameter and step types, and includes signed
  zeros, subnormals, `floatmin` and `floatmax` neighbors, both infinities,
  dependent and independent NaNs, and exact and adjacent thresholds. Large
  matrices run in the extended tier.
- **Bounds modes.** `scripts/test.jl` forces `--check-bounds=yes`, so the
  default mode needs its own run.
- **Broken rows.** Each `@test_broken` names its issue.

## Acceptance cases

Status at `3f90a29`, run locally. "Draft 2" marks where this draft departs
from the ChatGPT draft.

| case | question | target | today |
|:---|:---|:---|:---|
| V1 | `SqrNormL2(2^-1000)` at `[2^600]` | `2^199` (R1) | holds |
| V2 | `SqrNormL2(2^1000)` at `[2^-511; 32 × 2^-538]` | `2^-23 + 2^-72`, within 2 ulp (R1) | holds |
| V3 | ridge prox, `λ = 2^160`, `γ = 2^-160`, `Float32[1]` | `0.5f0` (T3) | `NaN` |
| V4 | ridge prox, `λ = 2^160`, `γ = 1`, `Float32[2^100]` | `2^-60` (R2, R3) | `0` |
| V5 | ridge gradient, `λ = 2^160`, `Float32[2^-80]` | `2^80`; value `0.5f0` (R2) | `Inf`; value holds |
| V6 | ridge prox, `λ = γ = 2^800`, `[2^900]`, Float64 | `2^-700` (R2, R3) | `0` |
| V7 | centered ridge, `x = 2^1023`, `c = -2^1023`, `λ = 2^-1022` | value `+Inf`, gradient `[4.0]` (O1) | gradient `Inf` |
| V8 | `NormL1(1 - 2^-30)`, `Float32[1]`, step 1 | `0f0` (T3); Draft 2 wanted `2^-30` | holds |
| V9 | centered ridge at `c = 1 + 2^-28`, `Float32` output | output `1f0`, value `0` (T3, O2); Draft 2 wanted `2^-57` | `2^-57` |
| V10 | `NormL1(1 + 2^-23)`, step `1 - 2^-23` (both `Float32`), `Float32[1]`, `Float64` output | `2^-46` (T3, T4) | `0` |
| V11 | centered `SqrNormL2(1.0, [1 + 2^-28])`, `value_and_gradient!` at `Float32[1]` into a `Float64` gradient | value `0f0`, gradient `[-2^-28]` (O1) | value `2^-57`; gradient holds |
| V12 | `QuadraticFunction(Float32[1;;], Float32[-1])`, gradient at `Float64[1 + 2^-28]` into a `Float32` output | `2^-28` (T3) | `0` |
| L2 | `LogDetLoss(zeros(2, 2))`, `value_and_prox!` at `[0 2^15; 2^15 0]`, step 1, `Float32` output | stored output singular; value `+Inf32` (decision 39) | value `0`, `f(Y) == Inf` |
| F16a | `Float16` argument into a `Float64` output | `ArgumentError` before writing (T6) | runs without a promise |
| F16b | `Float32` argument into a `Float16` output | `ArgumentError` before writing (T6) | runs without a promise |
| F16c | `Float32` argument and output, `Float16` level or step | accepted (T6) | accepted |
| W1 | `Translate(NormL1(0.25), [-M])([M])`, `M = floatmax(Float64)` | `M/2`, through homogeneity (R1, decision 45) | `Inf` |
| V13 | weighted `NormL1(2^-538, [2^1022])`, step `3·2^-538`, at `[2^-52]` | `2^-54` (R3) | holds |
| L5 | `IndBallRank(1)` of `h·v·v'`, `h = nextfloat(0f0)`, `v = (1, 5/8)`, into `Float32` | the stored matrix's rank claimed only under the bound; the fused value equals `IndBallRank(1)` of the stored matrix (A5, O2) | stored `[2 1; 1 1]·h`; fused `0`, standalone `Inf` (`d00c7b86`) |
| L3 | rank-one projection of `diag(1 + 2^-28, 0)` into `Float32` | stored `diag(1, 0)`; residual `2^-28` within the bound with `‖E_D‖` (A5) | holds |
| L4 | `IndStiefel` projection of `(1, 1)` into `Float32` | `\|y'y - 1\|` within the bound with the storage term (A5) | `5.0e-8` |
| Q1 | `restrict(QuadraticFunction([2.0 1; 1 2]), [1])`, prox at `[1.0]`, step 1 | `1/3`, not the full prox's `3/8` (S2) | holds |
| P2 | `quadratic_model(PoissonLoss([0.0], ones(1, 1)), [710.0])` | the dedicated exception, since the weight `exp(710)` overflows; the working response is `709` (S2, decision 43) | weight `Inf`, working response `NaN` |
| P2m | `quadratic_model!` into a valid `q`, with a later observation whose coefficient overflows | the dedicated exception, `q` unchanged (O6) | not tested |
| P3 | as P2 with observation weight `1/8` | weight `exp(710)/8 ≈ 2.79e307`, returned (R2) | weight `Inf` (`d00c7b86`) |
| S1 | `IndBox(0, 1)` at `2` | value `Inf`; projection `1`; joint value `0` | holds |
| S2 | `SqrNormL2(0.0)` at `[NaN]` | value `0`, gradient `[0]`, prox `[NaN]` (E4) | gradient `[NaN]` |
| S3 | `NormL1(1)`, `NormL2(1)`, `GroupNormL2` at `[NaN, 4]` | `[NaN, 3]`, `[NaN, NaN]`, `[NaN, 3]` (E2) | L1 holds; `[0, 0]`; `[0, 3]` |
| S4 | `NormL1` weights `[0, Inf]` at `[3, 4]` | value `Inf`, prox `[3, 0]`, joint value `0` (E4) | holds |
| S5 | `NegEntropy(1)` at `0`, `-1`, `NaN` | `0`, `Inf`, `NaN` (E2, E6) | `0`, `Inf`, `Inf` |
| S6 | `NuclearNorm(0)` at a NaN entry | value `0`, identity output (E4); Draft 2 kept `Inf` | `Inf` |
| S7 | simplex projection with zero array steps | `ArgumentError`, output unchanged (P1) | holds |
| S8 | V8 aliased, `prox!(x, f, x, γ)` | same branch as the separate output (A2) | holds |
| S9 | `IndBox(-1, 1)` and `IndBallL2(1)`: `value_and_prox!` at `[NaN, 0]` | outputs `[NaN, 0]` and `[NaN, NaN]`, value `NaN` (E2, O2) | outputs hold; value `0` (#349) |
| S10 | negative step on `NormL1` | `ArgumentError` before writing (P1) | expands (#311) |
| L1 | logistic loss, label 1, `η = 40` | value and derivative `≈ ±4.25e-18` (E6) | holds |

## Support matrix

To be filled in the docs PR from #366's inventory, one row per family and
backend: argument types, destination types, data types, the guarantee kind
(A1) with its quantitative bound or "bound pending", and status. The status
separates three things: the approved policy, the examples checked (with the
SHA they were checked at), and the bounds still pending. The page is
not normative until every affected row has a bound or is marked pending. Rows already known:

- `Float16`: rejected as argument, destination, direction or data in 0.6;
  accepted as a parameter or step in a wider call (T6).
- LAPACK families (A5): `Float32` and `Float64` arguments and outputs only.
  `IndStiefel` and `LogDetLoss` are missing from today's docs list.
- Functions that hold data: compute in the wider of the data's and the call's
  precision (T2).
- Constructor storage: as given (`NormL1`, `NormL2`, `IndBox`), converted to a
  common float type (`GroupNormL2`, `IndHyperplane`'s `b`), or promoted
  without `float` (`ElasticNet`, `MCP`, `SCAD`, `FusedLasso`, `Berhu`,
  `NormLq`). All three satisfy T5; none narrows.

Table of wider internals (T2): the recovery paths of R1–R3, and the search
families once A3's mechanism is chosen.

## Decisions record

Each rule above rests on a decision in `05-decisions.md`. This draft adopts
every recommendation there; none is confirmed yet. The ones that change
behavior most:

| decision | question | recommendation adopted | rules |
|:---|:---|:---|:---|
| 3 | the problem a mixed call solves | parameters rounded once to `W` | T3, O2 |
| 4 | rounding that loses range | recompute without it | R3 |
| 7 | Float16 | rejected in 0.6 | T6 |
| 11 | range of gradients and proximal outputs | every closed-form entry | R2 |
| 20 | NaN in the argument | propagates to what depends on it | E2 |
| 6 | functions that hold data | the wider of data and call precision | T2 |
| 22 | zero level | the zero function | E4 |
| 25 | negative and NaN steps | `ArgumentError` before writing | P1 |
| 38 | what "unchanged" means | the same accuracy, not the same bits | C1, R1, R3 |
| 39 | `LogDetLoss` value at a singular stored output | migrate to O2; saturation allowed for matrices | O3, R4 |
| 40 | zero level against domain constraints | a zero level keeps domain constraints, infinite weights included | E4 |
| 41 | exception for sizes | `DimensionMismatch` | E1, O6 |
| 42 | infeasibility found during a search; arithmetic stopgaps | `ArgumentError` before writing; the dedicated exception before writing | O6 |
| 43 | scope of the contract | every oracle; `restrict` as the interface states it; a quadratic model that cannot represent its coefficients raises the dedicated exception | S2, O6 |
| 44 | gradients outside the domain | NaN | E6 |
| 45 | values of wrappers | a private scaled-value capability for inner results; homogeneity for norms inside `Translate` and `Precompose` (W1 = `M/2`); other combinations listed as not covered | R1 |
| 46 | the line between data and parameters | (b): matrices, their equation vectors, and the normals and offsets of linear sets are data; levels, weights, centers, bounds and steps are parameters | T2 |
| 47 | non-finite linear coefficients | rejected at construction | M1 |

The decisions from #366 map as follows:

- D1 (search mechanism) is decision 18, settled by measurement in the
  implementing PR.
- D2 (cost) is decision 35.
- D3 (Float16) is decision 7.
- D4 (accuracy form) is decision 13.
- D5 (wider destination) is decision 5.
- D6 (data-holding losses) is decision 6, revised: compute in the wider of
  the data's and the call's precision.
- D7 (metrics) is decision 27.
- D8 (inside the band) is decision 18.
- D9 (`BigFloat` caches) is decision 8.
- D10 (LAPACK near a boundary) is decision 19.

## Issues this page governs

- Precision and mixed types: #366, the tracker (#151, #218, #257, #313,
  #350, #362, #363).
- Range of values: #283, with parts #356–#360.
- Range of proximal outputs: #361.
- Steps: #311. Aliasing: #367, #318, #321, #324.
- Non-finite inputs: #349, #350, plus the new items in E2 and E4.
- Tolerances: #307.
- Devices: #322.
- Constructors: #313, #333.
