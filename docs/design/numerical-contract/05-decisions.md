# Decisions for the numerical contract

*Claude, 2026-10-31. Every decision the contract needs, each with
background, an example, options with pros and cons, precedent in other
packages, and a recommendation. Measurements were taken on an Apple M3
with Julia 1.13.3; behavior described as current was run at master
`3f90a29`. Rule identifiers (T3, R2, ...) refer to `03-claude-draft3.md`,
and D1–D10 refer to #366.*

Each decision ends with a **Decided** line, left empty until the maintainer
answers. An answer is recorded by filling that line in a commit.

## A short primer

The contract uses a few ideas from floating-point arithmetic that a
mathematical paper never needs.

- **Rounding.** A `Float32` holds 24 significant bits and a `Float64` holds
  53. Every arithmetic operation rounds its exact result to the nearest
  representable number. The relative spacing near 1 is `eps(T)`: about
  `1.2e-7` for `Float32` and `2.2e-16` for `Float64`.
- **Range.** The largest finite number is `floatmax(T)`: about `3.4e38` for
  `Float32` and `1.8e308` for `Float64`. A larger result *overflows* to `Inf`.
  The smallest *normal* number is `floatmin(T)`: about `1.2e-38` and
  `2.2e-308`. Below it, *subnormal* numbers keep fewer and fewer bits, until
  the result *underflows* to `0`. An intermediate result, such as `x²` inside
  `‖x‖²`, can overflow or underflow even when the final answer fits.
- **NaN and Inf.** `0/0`, `Inf - Inf` and `0 * Inf` give NaN ("not a
  number"). NaN spreads through arithmetic, and every comparison with it is
  false: `NaN > 1` and `NaN <= 1` are both false. So code that branches on a
  comparison silently picks the `else` branch at NaN.
- **Cancellation.** Subtracting two nearly equal numbers is exact in itself,
  but any error already in the operands becomes large relative to the small
  difference.
- **Backward stability.** This is the standard of numerical linear algebra
  (Higham, *Accuracy and Stability of Numerical Algorithms*, 2002; LAPACK).
  A computed result is acceptable if it is the exact result for inputs
  perturbed by a few `eps`. It is the strongest promise that is affordable
  in general, because the inputs themselves usually carry errors of that
  size.
- **Ill-conditioned decisions.** A decision such as "is `|x|` above the
  threshold?" changes discontinuously at the threshold. Near it, a change in
  the last bit of any input flips the answer, so no algorithm can be
  accurate there in the forward sense. What can be promised is that the
  decision is exact for the inputs as the algorithm used them.

## Index

| # | decision | recommendation | changes current behavior? |
|:--|:---|:---|:---|
| 1 | return and output types | keep (#238) | no |
| 2 | working precision | output type, not promotion | no |
| 3 | what problem a mixed call solves | parameters rounded once to the working type | yes: V9 and some ridge cases |
| 4 | parameters whose rounding loses range | recompute without the rounding | yes, fixes V3–V6 |
| 5 | wider destination | compute in the destination type | yes (#362) |
| 6 | functions that hold data | the wider of data and call precision | yes: Float32 data with a Float64 argument becomes Float64-accurate |
| 7 | Float16 | unsupported in 0.6; rule recorded for later | yes: Float16 is rejected |
| 8 | BigFloat and integers | supported; caches keyed by precision | small |
| 9 | constructors | validate in own type; never narrow | yes (#313, #333) |
| 10 | range of values | keep the #283 rule | no |
| 11 | range of gradients and proximal outputs | same rule, every closed-form entry | yes, fixes V4–V7 and #361 |
| 12 | subnormal results | relative accuracy only for normal numbers | no |
| 13 | form of accuracy guarantees | per output | no |
| 14 | thresholds and ties in closed forms | exact, no tolerance | no |
| 15 | sums and reproducibility | allow SIMD reordering; certify decisions | no |
| 16 | membership tolerance | `√eps·max(1, \|bound\|, s)` | no (gaps fixed) |
| 17 | rank and zero-mask threshold | `√eps·σ₁` | no |
| 18 | decisions in searches | certify; refine, else "cannot certify" | yes (#151, #257) |
| 19 | LAPACK results near a boundary | relative band, results as computed | no |
| 20 | NaN in the argument | NaN spreads to what depends on it | yes |
| 21 | infinite entries | formula or limit, else NaN | small |
| 22 | zero level, zero and infinite weights | zero level is the zero function | yes |
| 23 | exceptional values across precisions | kept (#350) | yes |
| 24 | domains and masks | `+Inf` outside the domain | no |
| 25 | negative and NaN steps | `ArgumentError` before writing | yes (#311) |
| 26 | zero and infinite steps | limits only where unique | no |
| 27 | diagonal metrics that do not fit | scale times ratios | yes (#257) |
| 28 | the point of a joint value | the stored output | no |
| 29 | approximate projections | value `0`, not a certificate | no |
| 30 | wrappers that round between stages | keep; document a bound | no |
| 31 | errors before writes | keep | no |
| 32 | aliasing | keep the current rule; #367 audits | no |
| 33 | sparse and symmetric storage | keep (#142) | no |
| 34 | GPU arrays | same meaning as on the CPU | small |
| 35 | cost policy | three separate thresholds | no |
| 36 | what the solver decides | documentation only | no |
| 37 | where this lives and in what order | this PR, then #366 as tracker | — |

## Types and precision

### 1. Return and output types (T1)

**Decide.** What type does each call return, and what type does it write?

**Today** (#161, #185, #238): `f(x)` and the value from
`value_and_gradient!` have the argument's precision,
`R = float(eltype(x))`. The value from `value_and_prox!` has the output's
precision. Written arrays keep their element type. For example,
`NormL1(0.1)(Float32[1, 2])` returns a `Float32`, although `0.1` is a
`Float64` literal.

**Options.**

- *(a) Keep.* A Float32 problem stays Float32 wherever a decimal literal
  appears.
- *(b) Promote,* as Julia arithmetic does: `0.1 * 1f0` is a `Float64`, and
  the value above would be a `Float64`. That is predictable for Julia users,
  but a Float32 solver would receive `Float64` values from every function
  built with a literal, and type-stable solver loops would break.

**Precedent.**

- ProximalOperators.jl promotes: its `NormL1` returns `f.lambda * n1y`, a
  `Float64` when `λ` is.
- Flux's performance tips warn against Float64 literals in Float32 models,
  and `f32` converts a model's parameters.
- PyTorch, JAX and NumPy 2 (NEP 50) treat a Python scalar as "weak", so it
  takes the array's type: `float32 array * 0.1` stays `float32`.

**Recommendation.** (a), keep. It is what array-language users expect, and
it is settled.

**Decided:**

### 2. Working precision (T2)

**Decide.** In what precision is the arithmetic done?

**Background.** Writing a `Float32` result does not say which precision the
work was done in. Julia's natural answer is promotion:
`y .= x .- c` with `Float32` `x` and `y` and a `Float64` `c` subtracts in
`Float64` and rounds once when storing.

**Options.**

- *(a) The output's type,* `W = D`, and `W = R` for values. Exceptions are
  listed: decision 6, and families whose guarantee needs more.
- *(b) The promoted type of everything involved,* as plain Julia code would
  do.

**Measurement** (`bench-q1-threshold.jl` in this folder): soft thresholding
a Float32 vector with the threshold applied in Float64 costs **2.6–3.0×**
the all-Float32 loop, from 1,000 to 10 million entries. The conversions
dominate even when the loop is limited by memory. On GPUs the gap is larger:
consumer cards run Float64 at 1/32 to 1/64 of the Float32 rate.

**Precedent.**

- Under (b), mixed matrix products cannot use BLAS. In Julia, a `Float32`
  matrix times a `Float64` vector took 491 μs, against 57 μs for
  `Float32 * Float32`.
- PyTorch refuses a mixed-dtype matrix product outright.
- ML frameworks compute in the type of the data and parameters.

**Recommendation.** (a). Float32 is chosen for speed and memory, and (b)
spends that choice without asking.

**Decided:**

### 3. What problem a mixed-precision call solves (T3; Q1; cases V8, V9)

**Decide.** When a parameter is stored in a wider type than the working
precision, does the result answer the problem with the stored parameter or
with the parameter rounded to the working precision?

**Example 1** (V8). `NormL1(1 - 2^-30)`, step 1, at `Float32[1]`.

- With the stored threshold: `1 - (1 - 2^-30) = 2^-30`, and the
  coordinate stays nonzero.
- With the threshold rounded to `Float32`, it becomes `1f0`, the coordinate
  lies exactly on it, and the result is `0`. Master returns `0`.

**Example 2** (V9). Centered ridge with center `c = 1 + 2^-28` in
`Float64`, writing its proximal point to a `Float32` output that stores
`1f0`. With the stored center, the value at that point is
`(1 - c)²/2 = 2^-57`. With the center rounded to `Float32` (it rounds to
`1f0`), it is `0`. Master returns `2^-57`. Draft 3 promised `2^-57` and
also stated rule T3, which implies `0`. ChatGPT's comment found this
contradiction, and this decision resolves it.

**Options.**

- **(a) Parameters rounded once to the working type.** Products of
  parameters and steps, such as `γλw`, are formed first in their own wider
  type and rounded once. The rounded value defines the problem. V8 and V9
  are both `0`.
  - *Pros.* One rule with no exceptions, and the cheapest option. It is
    backward stable: the answer is exact for a parameter perturbed by at most
    `eps(W)/2`, the standard LAPACK holds itself to. Float32 data cannot
    distinguish `1 - 2^-30` from `1` anyway: the coordinate `1f0` is itself
    known only to about `6e-8`, so whether it lies above the threshold by
    `1e-9` carries no information. It also makes `NormL1(0.1)` and
    `NormL1(0.1f0)` give identical results on Float32 data, so a Float64
    literal is harmless.
  - *Cons.* Results are not those of the stored parameters. A user who
    checks V8 by hand in exact arithmetic sees a coefficient set to zero that
    "should" be `2^-30`.
- **(b) Stored parameters exact everywhere.** This is Draft 2's version, and
  what plain Julia broadcasting and ProximalOperators.jl do.
  - *Pros.* It matches a pencil-and-paper calculation, and every result
    depends only on the numbers supplied.
  - *Cons.* Each entry is computed in Float64 whenever a parameter is not
    exactly representable in Float32, which means every Float32 call with a
    decimal literal: 2.6–3.0× on the soft-thresholding loop, and worse on
    GPUs. It reverses #238 for no gain beyond the last-bit cases above.
- **(c) The middle option (ChatGPT's recommendation).** Parameters are
  rounded once, except where an argument entry and a parameter are
  subtracted (a threshold `|x| - t`, a center `x - c`); there the parameter
  keeps its full value.
  - *How.* The parameter is carried as two Float32 numbers, `t_hi + t_lo`,
    and the work stays in Float32 (`bench-q1-threshold.jl`, option c). This
    costs **1.12–1.24×** on soft thresholding, and only for mixed calls whose
    parameter is inexact in Float32. When the parameter is exact, `t_lo` is
    zero and the ordinary loop runs.
  - *Pros.* V8 gives `2^-30` and V9 gives `2^-57`, which can be tested
    exactly.
  - *Cons.* A second rule beside (a), with a boundary that needs care. It
    helps only where the argument side is exact: for a group norm or an
    inner product, the argument side is itself rounded, so the parameter's
    extra digits carry no information, and the rule would apply to some
    families and not others. ChatGPT proposed naming the two operations
    (the L1 threshold and the ridge center); a list of named operations is
    the kind of exception this package avoids. Every separable kernel and
    every GPU kernel with a threshold or a center would need the two-number
    form.

**Precedent.**

- (a): PyTorch, JAX and NumPy 2 weak scalars; Flux's `f32`; LAPACK's
  backward-stability guarantees.
- (b): Julia's promotion rules; ProximalOperators.jl
  (`gl = gamma * f.lambda`, then each entry in the promoted type).
- (c): Julia's ranges use the same two-number trick (`Base.TwicePrecision`)
  so that `0.1:0.1:0.3` hits its endpoints, as do DoubleFloats.jl and
  double-word arithmetic in general (Dekker 1971).

**Recommendation.** **(a).** It is one rule, backward stable, the cheapest,
and the convention of array and ML frameworks. The cases where (a) and (c)
differ are decisions at the threshold, which floating-point data cannot
resolve anyway. If the maintainer prefers results determined by the supplied
numbers, (c) is affordable, measured at 12–24% on mixed calls only, and
should be stated as a general rule for entrywise differences, not a list.
(b) is not recommended.

**Changes.** V9 returns `0` instead of `2^-57`. The ridge oracles round the
product `γλ` once (fixes V3). Draft 3's case table and rule O2 are updated to
match.

**Decided:**

### 4. Parameters whose rounding loses range (R3)

**Decide.** Under decision 3, what if rounding a parameter, or a product
like `γλ`, to the working type overflows to `Inf`, or underflows to zero,
while the answer fits?

**Example.** V4: `SqrNormL2(2^160)`, step 1, at `Float32[2^100]`.

- `γλ = 2^160` is `Inf` in `Float32`.
- The factor `1/(1 + γλ)` underflows to `0`, so master returns `0`.
- The answer `2^100/(1 + 2^160) ≈ 2^-60` is an ordinary `Float32`.

V6 is the same with `Float64` throughout (`2^800` times `2^800`), so a
wider type is not always available.

**Options.**

- *(a) Recompute without the rounding* on a recovery path, chosen once per
  call for scalar parameters. One way carries a power-of-two scale to the
  last operation.
- *(b) Document it as a limitation.*

**Pros and cons.** (a) costs nothing on ordinary inputs; the check runs on
one scalar per call. It needs a second code path per affected kernel,
including GPU kernels (#322). (b) leaves results of `0` or `Inf` that no
user can predict.

**Precedent.**

- Julia's `hypot` and `norm`.
- LAPACK's safe scaling (Anderson, "Algorithm 978: Safe scaling in the
  Level 1 BLAS", ACM TOMS 2017, adopted in LAPACK 3.10).
- Julia's complex division (Baudin and Smith's robust algorithm).
- LogExpFunctions' `logsumexp`, which subtracts the maximum before
  exponentiating.
- ProximalBase's own value rule (#283).

**Recommendation.** (a). It is the same principle the package adopted for
values in #283.

**Decided:**

### 5. Wider destination (T4; #366 D5)

**Decide.** With a `Float32` argument and a `Float64` output, is the result
computed in Float64, or computed in Float32 and widened?

**Example** (#362). A hyperplane projection forms a coefficient
`(⟨a,x⟩ - b)/‖a‖²`. Formed in the argument's `Float32` and then widened, it
has `Float32` accuracy in a `Float64` output.

**Options.**

- *(a) Compute in the output type,* from exactly widened inputs, cached
  quantities included.
- *(b) Compute in the inputs' type, then widen.*

**Pros and cons.** (a) is the same rule as decision 2 (`W = D`), applied in
both directions; the user asked for Float64. Its cost is Float64
arithmetic, which the user chose. (b) is what Julia broadcasting does
(`y .= x ./ 3` divides in `Float32`), but the extra digits of the output
are then noise. Widening a rounded coefficient and recomputing it in the
wider type are different operations; only recomputing gives Float64
accuracy.

**Precedent.** BLAS `DSDOT` takes single-precision vectors, accumulates in
double and returns double. LAPACK's mixed-precision solvers do the
refinement in the wider type.

**Recommendation.** (a).

**Decided:**

### 6. Functions that hold data (T2; #366 D6)

*Revised 2026-10-31 after the maintainer's question on #385 about a
`QuadraticFunction` with a `Float32` matrix and a `Float64` argument. The
first version covered only Float64 data with a Float32 argument.*

**Decide.** `LeastSquaresLoss`, `GLMLoss`, `LogDetLoss`,
`QuadraticFunction`, `KronQuadratic`, `IndAffine`, `IndGraph`,
`IndPolyhedron` and `LinearImage` store a data matrix. When the data's
precision and the call's precision differ, in which precision is the
arithmetic done?

**There are two cases, and they differ.**

- *Data wider than the call* (Float64 data, Float32 argument). Computing in
  the call's precision would round the matrix on every call. That is an
  O(np) conversion, or a second copy of the matrix, or generic code. Computing
  in the data's precision only converts the argument vector, at O(p), and
  uses BLAS. Julia's own `*` does this: a Float64 matrix times a Float32 vector
  converts the vector and calls BLAS, at the same cost as all-Float64
  (380 μs, n = 2000).
- *Data narrower than the call* (Float32 data, Float64 argument: the
  maintainer's example). Every Float32 number is exactly a Float64 number, so
  widening the data loses nothing. Computing in the data's precision would
  instead round the *argument* to Float32 and throw away its digits.

**Two kinds of "Float64 accuracy".**

- *Relative to the matrix the user had before storing it as Float32.* This is
  impossible. Those digits were dropped when the function was built, and no
  computation can recover them. A user who needs them builds the function in
  Float64.
- *Relative to the stored matrix.* This is achievable, by computing in Float64
  from the exactly widened data.

It matters even though the matrix is "only Float32". The solver's iterate is
Float64, and the function it minimizes is the stored one. If the value is
computed with Float32 rounding, its noise of about `1e-8` relative stops a
Float64 solver from telling iterates apart below that level: line searches
misbehave, and convergence stalls at Float32 accuracy on a problem that is
well defined in Float64.

**Measurement** (n = 2000, Float32 SPD matrix, Float64 argument, master
`3f90a29`):

| computation | relative error of the value | time |
|:---|:---|:---|
| `QuadraticFunction` today: `A*x` through a Float32 buffer | `7.5e-9` (Float32 level) | 630 μs |
| Float64 arithmetic on the stored Float32 `A` (generic `A*x`) | `2.0e-16` | 439 μs |
| all-Float64 BLAS, for comparison | | 494 μs |
| all-Float32 BLAS, for comparison | | 59 μs |

Today's `gradient!` of the same function writes a Float64 output and is
Float64-accurate (relative error `1.5e-15`), so the value and the gradient
of one function have different accuracies.

**Options.**

- *(a) The data's precision always* (#238; this decision's first
  recommendation). Fast when the data is wider. When the data is narrower,
  it rounds the argument; today that is also slower, because the mixed
  product into a Float32 buffer takes Julia's generic path.
- *(b) The call's precision always,* as decision 2 does for parameters. Right
  when the data is narrower. When the data is wider, it rounds the matrix on
  every call.
- *(c) The wider of the data's precision and the call's precision,* from
  exactly widened inputs. It never rounds the data per call and never throws
  away the argument's digits. Float32 data with a Float64 argument costs
  about what all-Float64 costs (439 μs against 494 μs here), but keeps the
  Float32 storage. A decomposition cached in the data's precision, such as
  `QuadraticFunction`'s eigendecomposition for its proximal operator, is
  rebuilt and cached in the wider type when needed, like other caches
  keyed by precision.

**Precedent.**

- BLAS `DSDOT` takes single-precision data and accumulates and returns in
  double.
- cuBLAS `cublasGemmEx` separates the storage type of its matrices from the
  compute type: half-precision inputs, single-precision arithmetic.
- Julia's `*` converts the narrower vector to the wider matrix's type.
- All three keep the narrow type for storage and do the arithmetic in the
  wider type.

**Recommendation.** **(c)**, for every function that holds data. The value
then has the call's type (decision 1) and is accurate to that type *for the
data as stored*. The data itself is only as accurate as the numbers the
function was built from; the documentation says so, and says to build in
Float64 for Float64 data. Today's sentence in the interface page ("a function
built in `Float32` and applied to a `Float64` argument returns a `Float64`
value with `Float32` accuracy") changes accordingly.

**Changes.** Values of data-holding functions with narrower data and a wider
argument become accurate to the argument's precision. `QuadraticFunction`
values in that case also get faster.

**Decided:**

### 7. Float16 (T6; #366 D3)

**Decide.** What happens with `Float16` arguments or outputs?

**Background.** `Float16` has 11 significant bits and a largest value of
65504, so `256² = 65536` already overflows. A sum of squares of ordinary
data overflows, and the recovery paths of decisions 4 and 11 would run
constantly. Generic code runs on `Float16` today without any promise. The
LAPACK families reject it.

**Options.**

- *(a) Support it, computing in Float32 inside,* rounding to `Float16` at
  the end. This needs an internal "working type" function in every kernel
  (Float16 maps to Float32). Emulated `Float16` on most CPUs already
  converts to Float32 for each operation.
- *(b) Native Float16 arithmetic.* Fastest on hardware with Float16 units,
  but overflow and accuracy problems everywhere.
- *(c) Unsupported in 0.6.* Raise an `ArgumentError` before writing, and
  record (a) as the rule for when support is added.

**Pros and cons.**

- (a) is the right rule, but it touches every family, GPU kernels
  included, and changes results.
- (c) costs nothing, since the check is on types and happens at compile
  time. It is honest, and adding support later is not a breaking change,
  whereas changing the meaning of existing Float16 support later would be.
  Users with Float16 data convert to Float32, which (a) would do internally
  anyway.

**Precedent.** PyTorch's mixed precision (`autocast`) keeps reductions,
norms, `exp`, `log` and losses in float32 even when the model is float16.
GPU tensor cores multiply float16 numbers but accumulate in float32. Both
are option (a).

**Recommendation.** (c) for 0.6, with (a) recorded as the rule for
supporting Float16 later. Float16 is rare in this domain, and a half-done
promise is worse than none.

**Decided:**

### 8. BigFloat, integers and other number types (T1, T7; #366 D9)

**Decide.** What is promised for `BigFloat` and integer inputs?

**Background.** `BigFloat`'s precision is a global setting that can change
between calls. A function that caches a `BigFloat` workspace can then mix
precisions.

**Options for BigFloat.**

- *(a) Supported in generic code,* with caches recording
  `precision(BigFloat)` and rebuilding when it changes.
- *(b) Rejected in cached families.*
- *(c) Unsupported everywhere.*

**Recommendation.** (a). BigFloat is how users and tests check results
independently, and the generic code already runs. The LAPACK families keep
rejecting it.

Integers keep the current rule: a fixed-width integer argument gives
`Float64`, `BigInt` gives `BigFloat`, and an integer destination is rejected
for fractional results (#363).

**Decided:**

### 9. Constructors (T5)

**Decide.** What may a constructor do with its parameters?

**Today** there are three storage patterns:

- stored as given (`NormL1`);
- converted to a common floating type (`GroupNormL2`);
- promoted without `float` (`ElasticNet`).

None narrows. Two bugs exist:

- #313: bounds are promoted before validation, so a `Float16` bound that is
  valid in its own type fails.
- #333: nested `Scaled` multiplies scales at construction, and an
  underflowing product turns the function into `ZeroFunction`.

**Recommendation.** One rule: validate in the parameter's own type before
any conversion; store as given or promoted; never narrow; never let a finite
positive parameter become `0` or `Inf` through arithmetic at construction.
The three patterns can stay, because all satisfy the rule.

**Decided:**

## Range and accuracy

### 10. Range of values (R1)

**Decide.** Keep the #283 rule: a value that is a normal number is returned
with ordinary accuracy, even when an intermediate overflows or underflows.

**Background.** This is settled and already documented, and its cost was
accepted: 2–5 ns per call. Case V1: `SqrNormL2(2^-1000)` at `[2^600]` is
`2^199`, whereas squaring first overflows.

**Precedent.**

- BLAS `nrm2` and LAPACK `lassq` do this.
- Julia's `norm` and `hypot` do this.
- Optimization packages usually do not. ProximalOperators.jl's `SqrNormL2`
  value is `λ/2 * sum(abs2, y)`, which overflows.

**Recommendation.** Keep.

**Decided:**

### 11. Range of gradients and proximal outputs (R2; Q3)

**Decide.** Does the rule of decision 10 extend to each entry of a gradient
and of a proximal output?

**Today** the interface page says the value rule "does not cover proximal
points, gradients or other outputs". Examples that fail:

- V4 and V6: proximal outputs that become `0`.
- V5: a gradient that becomes `Inf`.
- V7: a centered-ridge gradient that becomes `Inf` because `x - c`
  overflows, although the gradient is `4`.
- #361: `Berhu`, `HingeLoss`, `ElasticNet`, `NormL1Fused`.

**Options.**

- *(a) Every entry of every closed-form output:* separable kernels, group
  and ball shrinkage, ridge, closed-form projections. Matrix products,
  factorizations and iterative searches get a per-family guarantee instead
  (decision 13).
- *(b) A named list of families* (Draft 2's "tier B": `SqrNormL2`, and the
  proximal operators of `NormL1`, `NormL2` and `GroupNormL2`).
- *(c) Values only,* as today.

**Pros and cons.**

- (a) is one rule. Its cost is like the value rule's: a check per call on
  scalar parameters, plus a check on outputs where an intermediate such as
  `x - c` can overflow. Each change is measured.
- (b) leaves most functions undefined.
- (c) is free, but a solver receives `Inf` gradients and `0` iterates for
  problems whose answers fit.

**Precedent.** The same as decision 4: safe scaling in BLAS and LAPACK,
`hypot`, robust complex division.

**Recommendation.** (a).

**Decided:**

### 12. Subnormal results (R1, R4)

**Decide.** How accurate must a result below `floatmin` be?

**Options.**

- *(a) Absolute accuracy of about `floatmin(T)`,* with zero allowed. Relative
  accuracy is promised only for normal results.
- *(b) Relative accuracy everywhere.*

**Pros and cons.** (b) is impossible in general, because subnormal numbers
have fewer bits. (a) is the standard model.

**Precedent.**

- Higham's error model adds an absolute underflow term.
- LAPACK's error bounds assume no underflow.
- GPUs in fast-math mode flush subnormals to zero altogether.

**Recommendation.** (a), keep. A subnormal *intermediate* that makes a
*normal* result wrong is still a bug (decisions 4 and 11).

**Decided:**

### 13. The form of accuracy guarantees (A1; #366 D4)

**Decide.** One universal accuracy promise, or a stated guarantee per kind
of output?

**Options.**

- *(a) Per output.* Values get decision 10. Closed-form outputs get
  componentwise accuracy. Iterative maps get a bound or certificate on the
  returned point. LAPACK families get backward stability in the scale of
  `‖X‖`. Nonconvex maps state which minimizer they return and how ties are
  broken.
- *(b) One universal rule,* such as "backward stable everywhere" or
  "correctly rounded".

**Pros and cons.** (b) fits in one sentence but is false or unaffordable:
correct rounding needs extra precision everywhere, and backward stability is
not even defined for constant functions, diagonal-metric projections or
Dykstra's algorithm (#366's critique). (a) can be true and checked, at the
price of more documentation.

**Precedent.**

- LAPACK documents error bounds routine by routine (LAPACK Users' Guide,
  chapter 4).
- Julia's elementary functions aim for an error of about 1 ulp.
- Correctly rounded libraries such as CRlibm exist, but are not used for
  speed reasons.

**Recommendation.** (a).

**Decided:**

### 14. Thresholds and ties in closed forms (A2)

**Decide.** Are closed-form branch decisions exact, or done with a
tolerance?

**Example.** Soft thresholding at `|x| = γλ` exactly returns `0`. A
tolerance `|x| ≤ γλ + δ` would set more coordinates to zero than the
penalty does.

**Options.**

- *(a) Exact,* on the operands as rounded under decision 3, with the same
  branch on the entrywise, coordinate, whole-array and aliased paths.
- *(b) With a tolerance.*

**Precedent.** glmnet and scikit-learn's coordinate descent use exact
`max(|z| - λ, 0)`.

**Recommendation.** (a). It holds today on all four paths for V8.

**Decided:**

### 15. Sums and reproducibility (A6)

**Decide.** May sums be reordered for speed, so that the last bits change
with SIMD width or `--check-bounds`?

**Options.**

- *(a) Allow reordering;* certify any decision taken on a sum (decision 18).
- *(b) Fixed summation order:* reproducible bits, but several times slower
  for simple sums.
- *(c) Reproducible summation algorithms* (ReproBLAS, Demmel and Nguyen):
  same bits everywhere, at a high cost.

**Precedent.**

- Julia's `sum` uses pairwise summation with SIMD, so its bits depend on
  blocking.
- NumPy uses pairwise summation.
- Intel MKL's bit-reproducible mode (CNR) is opt-in and slower.

**Recommendation.** (a). #257's CI failures came from a decision taken on a
reordered sum, and the fix is to certify the decision, not to fix the order.

**Decided:**

## Decisions and tolerances

### 16. Membership tolerance (A4)

**Decide.** How close to a set must a point be for the indicator's value to
be `0`?

**Today** (#283 part 2a, PR #375) the tolerance is
`√eps(T)·max(1, |bound|, s)`, where `s` is the error scale of the tested
quantity: `0` for entries given directly, and `‖λ‖∞` for a computed
spectrum. Remaining gaps are fixed tolerances: `1e-10` in
`_largest_eigenvalue` and in `IndIntersection`, the absolute `1e-8` in
`Testing.test_kkt`, and the absolute eigenvalue tolerances of #307.

**Options.**

- *(a) Keep.*
- *(b) Exact membership.* A point produced by a projection would then often
  read as infeasible.
- *(c) Relative without the floor of 1.* This is scale invariant, but a
  point at `-1e-20` is outside `[0, ∞)` while it is rounding noise of an
  O(1) computation.

**Precedent.**

- Julia's `isapprox` defaults to a relative tolerance of `√eps` (checked:
  `Base.rtoldefault(Float32) == sqrt(eps(Float32))`).
- NumPy's `isclose` adds an absolute floor (`atol = 1e-8`).
- Solvers' feasibility tolerances (SCS, Ipopt, Gurobi: `1e-4` to `1e-6`) are
  algorithmic, user-set and independent of the precision. That is a
  different kind of tolerance, and the contract keeps them apart.

**Recommendation.** (a). Keep, and remove the fixed defaults.

**Decided:**

### 17. Rank and zero-mask threshold (A4, A5)

**Decide.** Below what size is a singular value counted as zero?

**Today** (#283 part 2a) it is `√eps(T)·σ₁`, used for rank constraints and
for functions that count nonzeros.

**Options.**

- *(a) Keep `√eps·σ₁`.*
- *(b) The numerical-rank standard `min(m, n)·eps·σ₁`.*

**Pros and cons.**

- (a) is the same `√eps` as decision 16, so all membership-type decisions
  agree. It is robust to partial decompositions and to errors in upstream
  data. But it is much looser than the standard: in `Float32`, `√eps` is
  `3.5e-4`, so a matrix whose next singular value is `3e-4·σ₁` counts as
  rank `k`.
- (b) is the smallest band that LAPACK's error justifies, but it treats
  noise from upstream computations as rank.

**Precedent.** Julia's `rank` uses `min(m, n)·eps·σ₁` (checked in
`LinearAlgebra/src/generic.jl`), and so do NumPy's `matrix_rank` and
MATLAB's `rank`.

**Recommendation.** (a), keep, and document that it is deliberately looser
than numerical rank. The question it answers is whether the matrix lies
within tolerance of the rank-`k` set, the same question every indicator
answers.

**Decided:**

### 18. Decisions in multiplier searches (A3; #366 D1 and D8)

**Decide.** `Constrained` on a hyperplane, halfspace, hyperslab or simplex
searches for a multiplier, and decides when it has found one. What is
promised about the point returned?

**Example** (#151). In Float32, the search stops on a small residual, but
the returned point can be about 500 `eps` from the true projection, because
a small slope magnifies the residual. #257 adds a decision whose outcome
depends on summation order.

**Options inside the uncertainty band**, where rounding cannot tell the
answer:

- (a) refine, or recompute in a wider type;
- (b) return a documented inexact result;
- (c) throw "cannot certify in this precision".

For the mechanism (#366 D1), the candidates are: always search in Float64;
widen adaptively; or compensated Float32 sums.

**Pros and cons.**

- Refining is accurate but costs extra passes.
- Throwing is honest, but a solver must handle it.
- Returning an inexact point silently is what happens today.

**Precedent.**

- LAPACK's `dsgesv` solves in single precision, refines in double, and falls
  back to a double solve if refinement fails.
- Julia throws exceptions when a factorization fails (`PosDefException`,
  `SingularException`), and `cholesky(...; check = false)` lets the caller
  choose instead.

**Recommendation.** Certify the stored output. Inside the band, refine where
that is cheap (the `dsgesv` pattern), otherwise throw an `ArgumentError` that
names the precision, not "infeasible". Choose the mechanism by measurement
in the implementing PR.

**Decided:**

### 19. LAPACK results near a boundary (A5; #366 D10)

**Decide.** When an eigenvalue lies within LAPACK's error of a boundary
(zero for the PSD cone, the `k`-th singular value for a rank ball), what is
returned?

**Options.**

- *(a) Decide with the relative band of decision 16* and return the result
  as computed, with the guarantee stated in the scale of `‖X‖`.
- *(b) Recompute in Float64* when a Float32 decision falls inside the band.

**Precedent.** LAPACK guarantees normwise backward error. Julia's `eigen`
and `svd` return results as computed; no common library recomputes.

**Recommendation.** (a). Add (b) only for a measured case.

**Decided:**

## Special values

### 20. NaN in the argument (E2; Q2)

**Decide.** What do the value and the outputs give when the argument has a
NaN entry?

**Today** the package mixes three behaviors:

- closed forms give NaN (`NormL1(1.0)([NaN]) == NaN`);
- indicators, `NegEntropy` and spectral functions give `Inf` (a failed
  membership test, or by design in #241 and #343);
- `value_and_prox!` of indicators gives `0` (#349).

`NuclearNorm(0)` at a NaN entry gives `Inf`, while `SqrNormL2(0)` gives `0`.
`prox(NormL2(1.0), [NaN, 4.0], 1.0)` gives `[0, 0]`, because the NaN norm
fails `norm > threshold`.

**Options.**

- **(A) NaN spreads to what depends on it.**
  - Every result that reads the NaN entry is NaN: the value, that
    coordinate's output for separable functions, the whole coupled block for
    groups, balls and spectral functions.
  - Results that do not depend on it are computed as usual, and a zero-level
    function ignores the argument (decision 22).
  - NaN takes precedence over `Inf` in a sum. No comparison with NaN selects
    a branch.
  - *Pros.* One rule. It is IEEE arithmetic's own rule, it matches Julia
    Base, and a solver can tell "trial outside the domain" (`Inf`) apart from
    "something produced a NaN" (NaN).
  - *Cons.* It reverses #343 for NaN entries (LAPACK still never sees
    them), and indicators at NaN change from `Inf` to NaN. Membership tests
    need a NaN check, fused into their existing pass.
- **(B) NaN gives `+Inf` everywhere.** This extends #343 and #349 to every
  value, so `SqrNormL2(1.0)([NaN])` and `ZeroFunction()([NaN])` become `Inf`.
  - *Pros.* Also one rule. A careless line search that rejects only on
    `f > bound` still rejects.
  - *Cons.* It hides the NaN: a solver whose iterate went NaN sees
    "infeasible" and shortens its step until it gives up. Values change for
    every smooth function.
- **(C) Throw on NaN input.**
  - *Pros.* Loud.
  - *Cons.* An O(n) check in every oracle call inside the solver's inner
    loop, and the solver loses the chance to recover.

**Precedent** (checked in their sources):

- IEEE 754: NaN propagates, and the 2019 revision added `maximum` and
  `minimum` operations that propagate NaN.
- Julia Base: `max(NaN, 1)`, `maximum`, `clamp(NaN, -1, 1)` and
  `norm([NaN, 1])` all return NaN.
- LogExpFunctions: `xlogy(0, NaN)` checks explicitly so that NaN survives
  the `0·log` shortcut.
- ProximalOperators.jl: its `NormL2` proximal operator returns `[NaN, NaN]`
  (its `max(0, NaN)` propagates), but its `IndBox` returns `0` at NaN, so it
  calls a NaN point feasible. That is the hazard of branching on
  comparisons.
- LineSearches.jl's `BackTracking` treats NaN and `Inf` alike (it tests
  `isfinite`), halves the step, and throws after a limit. So (A) and (B)
  behave the same there; the difference is only in what a solver can
  report.
- scikit-learn and glmnet reject NaN data when a model is fitted (C), but
  once, at the model layer, not inside every oracle.

**Recommendation.** **(A).** Under any option, NaN must never become zeros
through a failed comparison; the `NormL2` and `GroupNormL2` results are bugs
regardless. This also resolves #349 (NaN) and removes the spectral exception
from rule O3.

**Decided:**

### 21. Infinite entries (E3)

**Decide.** What do `±Inf` entries of the argument give?

**Recommendation.** The function's formula or limit where it is determined,
and NaN where it is not (`Inf - Inf`):

- a norm or convex quadratic gives `+Inf`;
- an indicator gives `+Inf`, since no set of real vectors contains an
  infinite entry;
- ridge maps `-Inf` to value `+Inf`, gradient `-Inf` and proximal output
  `-Inf` (true today);
- no convenient limit is picked where the limit depends on the direction of
  approach.

The alternative is to treat `Inf` like NaN, which loses legitimate answers
such as the indicator's `+Inf`.

**Precedent.** IEEE arithmetic with infinities, and the extended-value
convention of convex analysis (a function is `+∞` outside its domain).

**Decided:**

### 22. Zero level, zero weights and infinite weights (E4)

**Decide.** What does a zero level mean, and how do zero and infinite
coordinate weights behave?

**Example** (S2). `SqrNormL2(0.0)` at `[NaN]` has value `0` today but
gradient `[NaN]`, because `0 * NaN` is NaN.

**Options for a zero level.**

- *(a) The zero function.* Value `0`, gradient `0`, and a proximal step that
  copies the argument, NaN included. Zero scaling of a function with a
  restricted domain keeps the domain: `0 * IndBox` is still the box.
- *(b) Arithmetic,* `0 * f(x)`. This gives NaN at NaN and at points where
  `f` is infinite, and turns indicators into nonsense.

**Precedent.** The package already does this for `Scaled(f, 0)`, which
becomes `ZeroFunction()` for finite functions and keeps domain constraints
(calculus page).

**Recommendation.** (a). A zero coordinate weight is a free coordinate. An
infinite coordinate weight, where a family admits one, fixes the coordinate
at zero (S4, true today). An infinite level is not an infinite weight.
Levels, weights and bounds that are NaN are rejected at construction. A
finite weight never acquires the meaning of a zero or infinite weight
through rounding (decision 4).

**Changes.** The `SqrNormL2(0)` gradient. `NuclearNorm(0)` at NaN gives `0`
instead of `Inf`.

**Decided:**

### 23. Exceptional values across precisions (E5; #350)

**Decide.** Is an `Inf` or NaN that is the exact answer, such as an
indicator outside its set, kept when a wrapper converts or recomputes in
another precision?

**Example.** #350: mixed-precision `Precompose` discards an indicator's
`Inf`.

**Recommendation.** Keep exact exceptional values through conversion.
Recover non-finite intermediates that arise from finite inputs (decisions 4
and 11).

**Precedent.** IEEE conversions preserve `Inf` and NaN.

**Decided:**

### 24. Domains and masks (E6)

**Decide.** What does a finite point outside the domain give, and how do
observation masks work?

**Recommendation.**

- A finite point outside the domain gives a value of `+Inf`: `NegEntropy` at
  `-1`, an indicator off its set.
- At a domain boundary, the value is the one-sided limit where it is finite:
  `x log x` is `0` at `0`.
- Small tails stay explicit: the logistic loss at `η = 40` is `4.25e-18`,
  not `0`, which is true today.
- A family with masks documents their precedence. `BernoulliLoss` tests a
  NaN probability before a zero weight.

**Precedent.** The extended-value convention of convex analysis, as in CVX
and ProximalOperators.jl indicators. Julia's `log(-1)` and LogExpFunctions'
`xlogx(-1)` throw a `DomainError` instead. An optimization oracle owes the
line search `+Inf`, not an exception.

**Decided:**

## Steps

### 25. Negative and NaN steps (P1; #311)

**Decide.** What happens with a negative or NaN step?

**Today** they are not checked:
`prox(NormL1(1.0), [0.5, 0.5], -1.0)` returns `[1, 1]`; the "shrinkage"
expands.

**Options.**

- *(a) `ArgumentError` before writing.*
- *(b) NaN outputs.*
- *(c) Leave unchecked.*

**Pros and cons.** (a) costs O(1) for a scalar step and O(n) for an array
step, which is already read anyway. It catches solver bugs early. (c) is
free but silently wrong.

**Precedent.** Julia throws a `DomainError` for `sqrt(-1.0)`.
ProximalOperators.jl does not check. The package already throws
`ArgumentError` for invalid input everywhere else.

**Recommendation.** (a).

**Decided:**

### 26. Zero and infinite steps (P1)

**Decide.** Are `γ = 0` and `γ = Inf` allowed?

**Today** (interface page) a step takes its limit only where the limit is
unique: separable functions entrywise (identity at `0`, the minimizer at
`Inf`), scalar steps on indicators, ridge. Array steps on coupling sets must
be positive and finite, because the limit depends on how the steps go to
zero (case S7: the limit can be `(½, ½)` or `(⅓, ⅔)`).
`test/infinite_weights.jl` relies on the separable limits.

**Options.**

- *(a) Keep limits where they are unique.*
- *(b) Require finite positive steps everywhere.* Simpler, but it breaks
  uses that are valid today.

**Recommendation.** (a).

**Decided:**

### 27. Diagonal metrics that do not fit the working type (P3; #366 D7)

**Decide.** What happens to array steps that round to zero in the working
type?

**Example.** Float64 steps `[1e-50, 2e-50]` round to zeros in Float32, but
they define the same projection as `[1, 2]`.

**Options.**

- *(a) Keep a scale times ratios,* with the scale entering expressions such
  as `γλ`. #257 does this with `μ = max(γ)·ν`.
- *(b) Reject.*

**Pros and cons.** (a) keeps valid problems working. Dividing by the scale
alone would preserve projections but change penalized maps, since `NormL1`
needs the actual `γλ`. (b) is simple but refuses valid input.

**Precedent.** LAPACK balances and scales matrices by powers of two
(`dgebal`, `dlascl`) and keeps the scale.

**Recommendation.** (a). A ratio that underflows is rejected before writing.

**Decided:**

## Agreement between oracles

### 28. The point a joint value describes (O1, O2)

**Decide.** At which point does a joint call report its value?

**Recommendation.** Keep the current rules:

- `value_and_gradient!` evaluates both results at the argument. An
  overflowing value does not invalidate a finite gradient (V7: `+Inf` and
  `[4.0]`).
- `value_and_prox!` reports the value at the stored output, after rounding
  to the output type. Under decision 3, that is the value of the problem
  with converted parameters.
- Last-bit differences from `f(y)` are allowed. A false zero, a false
  feasibility, or a change between finite, `Inf` and NaN is not.

**Precedent.**

- ProximalOperators.jl's `prox!` returns `f(y)`.
- NLSolversBase's `value_gradient!`, Zygote's `withgradient` and JAX's
  `value_and_grad` all evaluate at the input.

**Decided:**

### 29. Approximate projections (O3)

**Decide.** What value does an iterative projection such as
`IndIntersection` (Dykstra's algorithm) return with its point?

**Options.**

- *(a) Return `0`,* the value at the exact projection, and document that it
  does not certify feasibility.
- *(b) Test membership of the returned point* with the tolerance of
  decision 16, returning `Inf` when the iteration stopped short.

**Pros and cons.** (b) is honest, but it costs a membership test and gives
a solver `Inf` objectives from a correctly working projection. (a) is the
current documented behavior.

**Recommendation.** (a). It becomes the only named exception to "value at
the stored output", now that decision 20 removes the spectral one.

**Decided:**

### 30. Wrappers that round between stages (O4)

**Decide.** In mixed precision, `Translate` and `Tilt` round the shifted
argument to the output type before calling the inner function. Remove the
rounding, or document a bound?

**Example.** `Translate(NormL1(1.0), [1e8])` at `[0.5]` gives `Float32[0]`
from `prox!` and `1.5` from `prox_entry`. In Float32, `0.5 - 1e8` has lost
the `0.5` (the spacing near `1e8` is 8).

**Options.**

- *(a) Keep, and document a bound* in the scale of the intermediate:
  `eps(Float32)·1e8` here, adding up over nesting depth.
- *(b) Carry intermediates in a wider type:* a second buffer, and against
  decision 2.

**Recommendation.** (a). Under decisions 2 and 3, this is backward stable:
the error is in the scale of the shifted quantity the user built.

**Decided:**

### 31. Errors before writes (O6)

**Decide.** Keep the settled rule that a call raising an `ArgumentError`
leaves every caller-owned output unchanged, tuple blocks and iterate storage
included?

**Background.** A prescribed NaN result is a result, not an error. There is
no rollback after a failed external factorization; the family documents
which outputs remain valid. Gaps: #239, #323, #371.

**Precedent.** Julia's `mul!` and `copyto!` check dimensions before writing;
C++ calls this the strong exception guarantee.

**Recommendation.** Keep.

**Decided:**

### 32. Aliasing (O7; #367)

**Decide.** Keep the current rule for outputs that are also inputs?

**Current rule.** `prox!(x, f, x, γ)` works for every function, including on
recovery paths, which must not read entries they have already overwritten.
Partial overlap is unsupported (#318). Gradient and Hessian-product outputs
are separate from their inputs unless documented (#321, #324). #367 audits
every oracle against this rule.

**Precedent.**

- Julia Base's documentation warns that mutating functions may misbehave
  when arguments share memory, and `mul!` forbids aliasing.
- ProximalOperators.jl supports `y === x` by convention and relies on it
  internally.

**Recommendation.** Keep, and let #367 finish the audit.

**Decided:**

### 33. Sparse and symmetric storage (M1, M2)

**Decide.** Keep the rules for implicit zeros and stored multiplicity?

**Current rules.** Implicit zeros of a sparse iterate are entries: a
centered ridge moves them, and `NegLog` is `+Inf` at them. A stored
off-diagonal of a `SymmetricSparseIterate` counts twice in reductions and is
written once. The symmetric-parameter rule of #142 applies.

**Recommendation.** Keep; both are settled.

**Decided:**

## Devices, cost and the solver

### 34. GPU arrays (G1–G6)

**Decide.** What is promised on a GPU?

**Options.**

- *(a) The same meaning as on the CPU:* output types, branches, zero and
  threshold decisions, scope of NaN and `Inf`. Accuracy is appropriate to
  the arithmetic mode, with no promise of identical bits.
- *(b) Best effort.*

**Precedent.**

- CUDA's fast-math flushes subnormals and approximates functions; CUDA.jl
  does not enable it by default.
- PyTorch documents that some GPU kernels are not bit-reproducible and offers
  an opt-in deterministic mode.

**Recommendation.** (a). In addition:

- no fast-math in kernels that must tell NaN from `Inf`;
- a risky scalar parameter is handled by a host-side scalar decision
  (#322), not by moving the array to the CPU;
- a backend without a method raises an error and is not called inaccurate;
- JLArrays tests dispatch, not hardware behavior.

**Decided:**

### 35. Cost policy (C1–C3; #366 D2)

**Decide.** What cost may the guarantees add?

**Recommendation.** Three separate rules, as ChatGPT's comment points out;
they are not competing estimates.

1. **Float64 non-regression.** Calls that are all-`Float64` keep their bits,
   except listed correctness repairs, each named in its PR. Machinery that
   only narrow types need leaves Float64 timings within benchmark noise.
2. **Investigate above 5%.** A repeatable slowdown above 5% on a
   representative workload is investigated and reported on the PR.
3. **Sign-off above 10% for narrow types.** A cost above 10% that only
   narrow types pay needs the maintainer's approval.

A small fixed cost per call for a guarantee is accepted, as for decision 10
(2–5 ns). Not used: a public `robust = true` switch, global precision
settings, custom number types, `BigFloat` in hot loops, or unconditional
widening of whole arrays.

**Precedent.** BLAS and LAPACK pay a fixed cost for safe scaling. Julia's
`norm` does an extra pass to scale.

**Decided:**

### 36. What the solver decides (documentation only)

**Decide.** Not a decision, but the contract should say what is left to
solvers.

**Recommendation.** The page states it:

- an `Inf` trial value means "outside the domain, shorten the step", not
  "infeasible problem";
- test acceptance positively (`f_trial <= bound`), because a NaN passes
  `f_trial > bound` as "not rejected";
- commit exactly the stored point that was checked;
- a step that changes no stored entry is stagnation, not convergence.

**Precedent.** LineSearches.jl halves on non-finite values and throws after
a limit. Optim.jl keeps the last accepted iterate.

**Decided:**

### 37. Where this lives, and in what order

**Decide.** Where does the design live, and in what order does the work
happen?

**Recommendation.**

- This pull request is the design record and the place for discussion. It
  ends by writing `docs/src/numerics.md`, the interface page's summary and
  the architecture link. It changes documentation only.
- #366 is retitled as the tracker for the whole contract (precision, range,
  special values, steps). #283, #311, #361, #367, #349 and #350 are listed
  under it, plus new issues for the untracked gaps:
  - `NormL2` and `GroupNormL2` turn NaN into zeros;
  - `SqrNormL2` cases V3–V7;
  - the `SqrNormL2(0)` gradient at NaN;
  - LAPACK proximal operators check only the output's element type.
- Implementation starts with ChatGPT's reconciliation table: for each
  oracle, the problem after conversion, the promised result, the acceptance
  case, and status. It begins with V3, V8 and V9, one #151 search case and
  one #362 wider-destination projection, before any patch.

**Decided:**

## Evidence used

- Acceptance cases V1–V9, S1–S10 and the logistic tail, run at `3f90a29`;
  results are in `02-claude-review-of-draft2.md` and Draft 3's case table.
- `bench-q1-threshold.jl`: soft thresholding with a rounded threshold (a),
  per-entry Float64 (b) and a two-word threshold (c). Results on the M3: (b)
  2.99×, 2.73× and 2.58×, and (c) 1.24×, 1.16× and 1.12×, at 10³, 10⁵ and
  10⁷ entries.
- Sources read:
  - ProximalOperators.jl: `normL1.jl`, `indBox.jl`, `normL2.jl`,
    `sqrNormL2.jl`, as installed in the local depot;
  - LineSearches.jl: `backtracking.jl` at `823cc4f`;
  - LogExpFunctions.jl: `basicfuns.jl`;
  - Julia 1.13 `LinearAlgebra`: `rank` and `*` for mixed precisions.
