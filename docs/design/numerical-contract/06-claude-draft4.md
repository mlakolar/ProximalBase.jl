# Numerical contract

*Draft 4, 2026-10-31. The current draft: Draft 3 (`03-claude-draft3.md`)
revised for ChatGPT's comparison (`04-chatgpt-comment.md`), with every
recommendation of `05-decisions.md` adopted. A rule that rests on a decision
not yet confirmed by the maintainer names it, as "(decision N)". Proposed
content of `docs/src/numerics.md`, linked from `docs/src/architecture.md`;
the Precision section of `docs/src/interface.md` becomes a summary of it.
Behavior described as current was checked at master `3f90a29`.*

*Changes from Draft 3:*

- *T3 and O2: V9's value is that of the converted problem, `0`. Draft 3
  promised `2^-57` and contradicted its own T3.*
- *T3: the two-word alternative is listed with its measured cost.*
- *T4: widening a rounded coefficient is distinguished from recomputing it.*
- *T6: Float16 is rejected in 0.6, and the Float32-internals rule is kept for
  later.*
- *E2: NaN is defined by IEEE propagation, not as an "unknown real number",
  which would have contradicted "NaN takes precedence over `Inf`".*
- *A4: the rank threshold is documented as deliberately looser than
  numerical rank.*
- *Case table and decisions record updated to match.*

This page states what ProximalBase promises about the numbers it returns:
their types, their accuracy, the decisions it takes on rounded quantities, and
what NaN, `Inf`, zero and infinite parameters and steps mean. It is the
reference for maintainers, for authors of new functions, and for solver
writers who need to know what a returned `Inf` or `NaN` says.

Each rule has an identifier (T1, R2, E4, ...) for tests, issues and reviews.
A rule states the intended behavior. Its status says whether the package meets
it now:

- **Holds**: checked at the revision above.
- **Gap (#N)**: a bug against this page, tracked in issue N. A gap is brought
  under the rule; it is not documented as a limitation.
- **Changes behavior**: the rule is decided and changes a documented or
  observable result; the change ships with a release note.

## Scope

**S1. Covered types and environment.** Real `Float32` and `Float64`
arguments, outputs, parameters, data and steps, in any mixture, on the CPU
with IEEE round-to-nearest and gradual underflow. `Float16` is rejected in
0.6 (T6). `BigFloat`, integer arguments and GPU arrays are covered as T7, the
[device rules](#devices) and the [support matrix](#support-matrix) say.
Complex input is rejected (Precision section of the interface page). Automatic
differentiation numbers are not covered.

**S2. Per result, not per function.** A guarantee attaches to each result of
an oracle: the value, each entry of a gradient, each entry of a proximal
output, and each decision (a zero, a tie, membership, an accepted multiplier,
a rank).

**S3. Not promised.** Correctly rounded results. Identical bits across
platforms, SIMD widths, `--check-bounds` settings, devices or library
versions. Exact feasibility from an approximate projection. Accuracy beyond
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
fractional results (#363).

**T2. Working type.** `W = D` for written arrays and `W = R` for values, with
two exceptions:

- The losses that hold data (`LeastSquaresLoss`, `GLMLoss` and its variants,
  `LogDetLoss`) compute in their data's precision (#238, #366 D6).
- A family whose guarantee (A1) needs a wider `W` lists it in the
  [table of wider internals](#support-matrix), with its scope and measured
  cost.

*Why.* Float32 and GPU work stays at Float32 speed. The rejected alternative,
`W = promote_type(D, stored types)`, would run every Float32 call that has a
Float64 literal, such as `NormL1(0.1)`, in Float64; #238 rejected that.

**T3. The problem a mixed call solves** (decision 3). A stored parameter
enters the computation rounded once to `W`. A product of scalar parameters
and steps, such as the threshold `γλw`, is formed in the promoted type of its
factors and rounded once, never factor by factor. The rounded value defines
the problem that is solved, including the value `value_and_prox!` reports
(O2). R3 is the only exception: a rounding that loses range.

*Why.* The result is exact for parameters perturbed by at most `eps(W)/2`,
the backward-stability standard of LAPACK. Data in `W` cannot resolve finer
differences, since the argument itself is known only to `eps(W)`. A decimal
literal is harmless: `NormL1(0.1)` and `NormL1(0.1f0)` give identical results
on Float32 data.

*Example (V8, documented behavior).* `prox!(z, NormL1(1 - 2.0^-30), Float32[1], 1.0)`
with a `Float32` destination: the threshold `1 - 2^-30` rounds to `1f0`, the
argument lies on it, and `z == Float32[0]`. **Holds.**

*Example (V9).* Centered `SqrNormL2(1.0, [1 + 2^-28])` with a `Float32`
output: the center rounds to `1f0`, the stored output is `1f0`, and
`value_and_prox!` returns `0`. **Changes behavior:** master returns `2^-57`.

*Rejected alternatives* (measurements in `bench-q1-threshold.jl`):

- *Stored parameters exact everywhere* (Draft 2): every entry is computed in
  the stored type whenever a parameter is not representable in `W`, that is,
  for every Float32 call with a decimal Float64 literal. That costs
  2.6–3.0× on soft thresholding, and more on GPUs.
- *Stored parameters exact only in differences between an argument entry and
  a parameter* (a threshold, a center), carried as two `W` numbers (ChatGPT's
  middle option). That costs 1.12–1.24× on mixed calls, and gives
  `V8 = 2^-30` and `V9 = 2^-57`. It is a second rule beside T3, and it helps
  only where the argument side is exact, so it applies to some families and
  not others.

*Current gap.* `SqrNormL2` rounds `γ` and `λ` separately, so
`prox!(z, SqrNormL2(2.0^160), Float32[1], 2.0^-160)` gives `NaN` (V3).
**Gap (new).**

**T4. Wider destination.** When `D` is wider than the argument or the stored
parameters, the result is computed in `D` from the exactly widened inputs,
cached quantities included (decision 5). For example, the hyperplane
coefficient and `‖a‖²` are formed in `D` (#362). Widening a coefficient that
was rounded in the narrower type is not recomputing it: only recomputing in
`D` gives the accuracy of `D`. The data-holding losses follow T2.
**Gap (#362).**

**T5. Constructors.** A constructor validates parameters in their own type
before any conversion (#313). It may store them as given or promoted to a
common type. It never narrows them, and never turns a finite positive
parameter into zero or `Inf`. **Gap (#313, #333).**

**T6. Float16** (decision 7). In 0.6, a `Float16` argument, output, step or
parameter array raises an `ArgumentError` before anything is written. The
check is on types and costs nothing at run time. Adding support later is not
a breaking change; changing the meaning of existing support would be.
**Changes behavior:** generic code runs on `Float16` today without a
promise.

*Rule for when support is added.* A call with `R` or `D` equal to `Float16`
computes in at least `Float32` (`W` and `Wacc`). Parameters convert to that
`W` directly, not through `Float16`, and the result is rounded to `Float16`
at the oracle's boundary. This is how mixed-precision training keeps
reductions, norms and losses in float32, and how GPU tensor cores accumulate
float16 products. `Float16`'s largest value is 65504, so a sum of squares of
ordinary data overflows in `Float16` arithmetic.

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
inputs therefore keep the bits of the ordinary expression. A subnormal value
may be off by about `floatmin(R)`. Underflow before a weight that scales the
value back into the normal range is covered.

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

- V4: `SqrNormL2(2.0^160)`, step 1, `Float32[2^100]` gives `2^-60` (today `0`).
- V5: the gradient of `SqrNormL2(2.0^160)` at `Float32[2^-80]` is
  `Float32[2^80]` (today `Inf`).
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

- *Ordinary path.* Taken when the rounded combined scalar is finite and
  normal, or exactly zero because a factor is zero. Its results are bitwise
  unchanged.
- *Recovery path.* Taken otherwise. For scalar parameters it is chosen once
  per call, before the loop over entries.
- *Mechanism.* The method is free: a wider scalar type, or a power-of-two
  scale carried to the last operation. V6 shows that a wider native type is
  not always available.
- *Scope.* A loss of significance that is not a range event follows T3, not
  R3 (V8).

Items: #218, #322, #356 (A11, A13, A21, A25), #359 (B09b), #283 (SLOPE
weights; group weights not representable in the device type).
**Gap (listed issues).**

**R4. Final range.** A result underflows to a subnormal or to zero only when
its exact value is below `floatmin(D)`. It overflows to `±Inf` only when its
exact value rounds beyond `floatmax(D)`. There is no saturation at
`floatmax`.

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

**A2. Exact decisions in closed forms.** A threshold, a zero, a clamp or a
tie computed in closed form is decided exactly, on the operands as rounded
under T3. At equality the documented branch applies; soft thresholding, for
example, returns zero at `|x| = γλ`. No tolerance widens or narrows a branch.
The entrywise (`prox_entry`), coordinate (`prox_coordinate!`), whole-array
and aliased (`prox!(x, f, x, γ)`) paths of a separable function take the same
branch at the same destination precision. A coupled group is decided as a
whole. **Holds** for the cases checked (V8 on all four paths).

**A3. Decisions on rounded sums and searches.** This is #366's C5.

- *Decisions.* Accepting a multiplier, certifying a point, membership,
  choosing a slab bound, fixing an entry.
- *Certified region.* A decision is certified outside a documented
  uncertainty band. The band rests on the rounding model with no overflow,
  and on a bound that covers every term's error: products, cached normals,
  sums of fixed entries, the offset subtraction and the reduction.
- *Inside the band (#366 D8).* Refine where that is cheap. Otherwise throw an
  `ArgumentError` saying that the result cannot be certified in the working
  precision. A message that the set misses the domain would be wrong there.
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
  (decision 17). It is deliberately looser than the numerical-rank threshold
  `min(m, n)·eps(T)·σ₁` of `LinearAlgebra.rank`, NumPy and MATLAB. It answers
  the membership question every indicator answers, whether the matrix lies
  within tolerance of the rank-`k` set, with the same `√eps` as the
  membership tolerance above.
- *Type.* Each tolerance is evaluated in the type of the quantity it tests.
- *Defaults.* No default is a fixed number independent of the precision.

Known gaps: `_largest_eigenvalue(...; rtol = 1e-10)`, `IndIntersection`'s
default `tol = 1e-10`, the absolute `Testing.test_kkt(...; tol = 1e-8)`, and
the absolute eigenvalue tolerances of #307. **Gap (#307, new).**

**A5. LAPACK families.** This is #366's C5b. It covers the eigenvalue and
singular-value families, `IndStiefel` and `LogDetLoss`.

- *Error scale.* LAPACK's symmetric eigensolvers and SVD are backward stable.
  Eigenvalues and singular values carry errors of order `p·eps(W)·‖X‖₂`,
  absolute in the scale of the matrix. A projection onto the PSD cone or a
  rank ball, or a spectral proximal map, is accurate to `c·p·eps(W)·‖X‖`.
  Eigenvalues closer to a boundary than that are not resolved.
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
depends on such a sum is certified under A3. Compensated or wider sums are
used on recovery paths and where A1 requires them, and are listed in the
table of wider internals.

## Special values

**E1. Order of meaning.**

1. *Structure.* Shapes, types, steps and required disjoint buffers are
   checked first. A violation raises an `ArgumentError` before anything is
   written (O6).
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

- `prox(NormL1(1.0), [NaN, 4.0], 1.0) == [NaN, 3.0]`. **Holds.**
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

- *Values.* A value takes the function's formula or limit where it is
  determined. A norm or a convex quadratic gives `+Inf`. An indicator gives
  `+Inf`, since no set of real vectors contains a point with an infinite
  entry. Where the value is not determined, as in `Inf - Inf`, it is NaN.
- *Gradients and proximal outputs.* They take the family's documented limit,
  and NaN in the dependent entries otherwise. For example, uncentered ridge
  with finite positive `λ` and `γ` maps `-Inf` to value `+Inf`, gradient
  `-Inf` and proximal output `-Inf`.
- *Limits.* No convenient limit is chosen where the limit depends on the
  direction of approach.

**E4. Zero and infinite parameters.**

- *Zero level.* The function is identically zero: value `0`, gradient `0`,
  and a proximal step that copies the argument. It reads the argument only to
  copy it, NaN entries included (S2). Zero scaling of a function with a
  restricted domain keeps the domain, as `Scaled` does today: `0 * IndBox` is
  still the box.
  - **Gap (new):** `gradient!` of `SqrNormL2(0.0)` at `[NaN]` gives `[NaN]`
    while its value is `0`.
  - **Changes behavior:** `NuclearNorm(0.0)` at a NaN entry returns `Inf`
    today.
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
  the value: `NegEntropy` at `-1`, an indicator off its set.
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

- *Limits.* A step takes its limit only where the limit is unique: for
  separable functions entrywise at `γ = 0` (the identity) and `γ = Inf` (the
  entrywise minimizer); for scalar steps on indicators (the projection is
  unchanged); for ridge at `0` and `Inf`. Otherwise steps must be positive and
  finite.
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

**O1. The gradient's point.** `value_and_gradient!` evaluates both results at
the argument, each in its own type (T1). A value that overflows does not make
a finite gradient invalid. Case V7 gives value `+Inf` and gradient `[4.0]`.
**Gap (new; the gradient half).**

**O2. The proximal value's point.** `value_and_prox!` returns the value at the
stored output, after rounding to `D`. It may reuse intermediates of the
proximal computation, and its last bits may differ from `f(y)` within A1. A
false zero, a false feasibility, or a change of class among finite, `Inf` and
NaN is not allowed. The value is that of the problem with parameters
converted under T3. Case V9: the center `1 + 2^-28` rounds to `1f0`, the
stored output is `1f0`, and the value is `0`. **Changes behavior:** master
returns `2^-57`, the value of the unconverted problem.

**O3. Approximate projections.** This is the one named exception to O2. An
iterative projection such as `IndIntersection` returns `0`, the value at the
exact projection. That `0` does not certify that the stored point is feasible;
its violation is tested separately, with the family's tolerance. Under E2 the
spectral `Inf` at non-finite inputs is no longer an exception.

**O4. Wrappers that round between stages.** In mixed precision, the array
proximal operators of `Translate` and `Tilt` round to `D` between stages. The
bound on the difference between `prox!` and entrywise `prox_entry` is stated
per wrapper, in each stage's intermediate scale, and adds up over the nesting
depth (#366 C4). Pinned case: `Translate(NormL1(1.0), [1e8])` at `[0.5]` gives
`Float32[0]` from `prox!` and `1.5` from `prox_entry`, within
`eps(Float32)·1e8`. **Gap (new):** today the docs claim no bound.

**O5. Identities with conditions.** A restriction may drop an additive
constant, so gradients agree while raw values differ by that constant. A
conjugate may give a value with its proximal output and have no standalone
value. Neither is a numerical bug.

**O6. Errors before writes.** Settled. A call that throws an `ArgumentError`
leaves every caller-owned output unchanged, including tuple blocks and
iterate storage. A prescribed NaN result is a result, not an error. After a
failed external factorization or iterative solve there is no rollback
promise; the family documents which outputs remain valid. **Gap (#239,
#323, #371).**

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
zero, and `NegLog` is `+Inf` at zero.

**M2. Multiplicity.** A `SymmetricSparseIterate` stores each off-diagonal
coordinate once. It counts twice in reductions and is written once. The
symmetric-parameter rule of #142 applies. **Holds** (`SqrNormL2(1.0)` of
`[0 3; 3 0]` is `9`).

**M3. Caches.** A cache is computational state. A result does not depend on
the history of the cache. `reset_cache!` and reconstruction follow the
Conventions section of the interface page. Warm starts are seeds (T8).

## Devices

**G1. Same semantics.** On a supported device, an operation has the same
output types, branches, zero and threshold decisions, and scope of NaN and
`Inf` as on the CPU.

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

- An `Inf` value at a trial point means the trial is outside the domain.
  Rejecting it and shortening the step is the solver's job. It is not
  evidence that the problem is infeasible.
- A NaN value fails every comparison. Test acceptance positively,
  `f_trial <= bound`, rather than rejecting on `f_trial > bound`, which a NaN
  passes. Bound the retries and report a numerical failure.
- Evaluate a trial in candidate storage and commit exactly the stored point
  that was checked.
- A positive step that changes no stored entry is stagnation, not
  convergence. A NaN residual does not certify stationarity.

## Cost

**C1. Float64 non-regression.** Calls whose argument, destination,
parameters and steps are all `Float64` keep their bits, except for listed
correctness repairs that apply to `Float64` too (R1–R3, A3), each named in its
PR. Machinery that only narrow types need leaves `Float64` rows within the
benchmark harness's noise (#366 C0).

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
  `ldexp(one(T), k)`, or `BigFloat` evaluated on the stored binary operands
  with a locally scoped precision. Start at 256 bits and raise the precision
  until the rounding and classification of the reference are stable.
- **Comparisons.** Exact (`===`) for exact values, zero decisions and ties.
  ULPs or a justified relative bound for smooth, well-conditioned results. A
  residual or certificate for matrices and iterative outputs. Never an
  absolute `≈` that would accept the zero V8 and V9 test for. Class and type
  are checked separately from the value.
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
| F16 | any oracle with a `Float16` argument or output | `ArgumentError` before writing (T6) | runs without a promise |
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

To be filled in the docs PR from #366's inventory, one row per family:
argument types, destination types, data types, backend, the guarantee kind
(A1), and status. Rows already known:

- `Float16`: rejected everywhere in 0.6 (T6).
- LAPACK families (A5): `Float32` and `Float64` arguments and outputs only.
  `IndStiefel` and `LogDetLoss` are missing from today's docs list.
- Data-holding losses: compute in the data's precision (T2).
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
| 22 | zero level | the zero function | E4 |
| 25 | negative and NaN steps | `ArgumentError` before writing | P1 |

The decisions from #366 map as follows:

- D1 (search mechanism) is decision 18, settled by measurement in the
  implementing PR.
- D2 (cost) is decision 35.
- D3 (Float16) is decision 7.
- D4 (accuracy form) is decision 13.
- D5 (wider destination) is decision 5.
- D6 (data-holding losses) is decision 6.
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
