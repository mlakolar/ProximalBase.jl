# Decisions for the numerical contract

*Claude, 2026-10-31. Every decision the contract needs, each with
background, an example, options with pros and cons, precedent in other
packages, and a recommendation. Measurements were taken on an Apple M3
with Julia 1.13.3; behavior described as current was run at master
`3f90a29`. Rule identifiers (T3, R2, ...) refer to `03-claude-draft3.md`,
and D1–D10 refer to #366.*

**Status.** All 47 decisions were approved by the maintainer on 2026-11-01.
Each entry opens with its **Decided** line; the background, options,
precedent, first recommendation and revision notes below it are kept as the
history of the decision.

*The drafts and reviews named in the history below (`01`–`16`) were removed from this folder when the contract was published in `docs/src/numerics.md`. They are at commit [`850e2bd2`](https://github.com/mlakolar/ProximalBase.jl/tree/850e2bd2/docs/design/numerical-contract) on pull request #385.*

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

Every decision was approved by the maintainer on 2026-11-01 ("go with recommendations", on #385). The **Decided** line at the top of each entry governs; the analysis below it is history.

| # | decision | decided (first sentence; see the entry) |
|:--|:---|:---|
| 1 | return and output types | Keep T1 (#161, #185, #238). |
| 2 | working precision | Each result is computed in its *model precision* (T3): the widest of its delivery precision, its argument's precision (and its direction's, for `hessian_vec!`), and its data's precision. |
| 3 | the problem a mixed call solves | Parameters are rounded once to the model precision. |
| 4 | rounding that loses range | R3: a rounding that loses range a representable result needs is recovered. |
| 5 | wider destination | A wider destination is computed in `D` from exactly widened inputs, caches included (#362). |
| 6 | functions that hold data | The model precision includes the data's precision (T2, T3). |
| 7 | Float16 | In 0.6, Float16 is rejected by role before writing: as argument, destination, direction or data. |
| 8 | BigFloat and integers | BigFloat is supported in generic code, with caches keyed by `precision(BigFloat)`; the LAPACK families reject it. |
| 9 | constructors | Parameters are validated in their own type before any conversion, stored as given or promoted, never narrowed, and never turned from finite positive to `0` or `Inf` (#313, #333). |
| 10 | range of values | Keep the #283 rule (R1): a normal value is returned with the ordinary accuracy, even when an intermediate leaves the range. |
| 11 | range of gradients and proximal outputs | R2: every entry of a gradient and of a closed-form proximal output follows R1's rule in `D`. |
| 12 | subnormal results | Relative accuracy is promised only for normal results. |
| 13 | form of accuracy guarantees | Per output (A1): each family states the problem it solves, its error scale and its guarantee. |
| 14 | decisions by kind | A2's table. |
| 15 | sums and reproducibility | Sums may be reordered. |
| 16 | membership tolerance | `√eps(T)·max(1, \|bound\|, s)`, with `s` the error scale of the tested quantity. |
| 17 | rank and zero-mask threshold | `√eps(T)·σ₁`, the package's convention for membership-type decisions. |
| 18 | decisions in searches | Certify the stored output. |
| 19 | LAPACK results near a boundary | In-band results are returned as computed, which takes precedence over 18. |
| 20 | NaN in the argument | NaN propagates to every result that depends on it, and never becomes zeros or `+Inf` through a failed comparison. |
| 21 | infinite entries | The formula or limit where determined: a norm or an indicator gives `+Inf`. |
| 22 | zero level | With 40: a zero level removes the finite part and keeps domain constraints, infinite weights included. |
| 23 | exceptional values across precisions | Exact `Inf` and NaN results survive conversions (#350). |
| 24 | domains and masks | The value outside the domain is `+Inf`; at a boundary, one-sided limits where finite; small tails stay explicit; masks document their precedence. |
| 25 | negative and NaN steps | `ArgumentError` before writing (#311). |
| 26 | zero and infinite steps | Endpoints only where the family names them: at `γ = 0`, the projection onto the closure of the domain; at `γ = Inf`, the projection onto the minimizers, for convex functions that have them; scalar steps on indicators; ridge. |
| 27 | diagonal metrics that do not fit | A retained scale times ratios; a ratio that underflows is rejected before writing (#257). |
| 28 | the point of a joint value | `value_and_gradient!` agrees with its standalone oracles, even across precisions (V11). |
| 29 | approximate projections | Return `0`, documented as not a feasibility certificate. |
| 30 | wrappers that round between stages | Keep the staging. |
| 31 | outputs after a failure | By kind. |
| 32 | aliasing | Keep the current rule; #367 audits. |
| 33 | sparse and symmetric storage | Keep (#142). |
| 34 | GPU arrays | The same meaning as on the CPU under A2's rules. |
| 35 | cost policy | A 5% slowdown triggers investigation; more than 10% paid only by narrow types needs sign-off; narrow-only machinery leaves Float64 within noise; a small fixed cost per call is accepted. |
| 36 | what the solver decides | Documented: `Inf` means outside the domain or genuine overflow; test acceptance positively and test `isfinite`; commit the checked point; no movement does not distinguish stagnation from a fixed point. |
| 37 | where this lives and in what order | PR #385 is the record. |
| 38 | what "unchanged" means | Accuracy, not bits: results stay within their stated accuracy; decisions as A2 specifies them; exact values and classes unchanged. |
| 39 | `LogDetLoss` | The value is that of the stored output, so a singular stored output gives `Inf`; the verification strategy is chosen by measurement. |
| 40 | zero level against domain constraints | Zero scaling keeps domain constraints and infinite weights, and is rejected when no domain-preserving proximal representation exists (`NegLog`). |
| 41 | exception for sizes | `DimensionMismatch`. |
| 42 | infeasibility found during a search; stopgaps | Infeasibility found during a search raises `ArgumentError` before writing. |
| 43 | scope of the contract | Every oracle. |
| 44 | gradients outside the domain | NaN, for the derivative oracles that are supported. |
| 45 | values of wrappers | A private scaled-result capability; homogeneity recovers overflowing transformed arguments of norms (W1 = `M/2`); other combinations are listed as not covered. |
| 46 | the line between data and parameters | (b): matrices, the vectors in their equations, and the normals and offsets of linear sets are data. |
| 47 | non-finite linear coefficients | Rejected at construction. |

## Types and precision

### 1. Return and output types (T1)

**Decided** (maintainer, 2026-11-01). Keep T1 (#161, #185, #238). `f(x)` and the value of `value_and_gradient!` have type `float(eltype(x))`, and the value of `value_and_prox!` has `float(eltype(y))`. Written arrays keep their element type, and the allocating forms use `float(eltype(x))`.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 2. Working precision (T2)

**Decided** (maintainer, 2026-11-01). Each result is computed in its *model precision* (T3): the widest of its delivery precision, its argument's precision (and its direction's, for `hessian_vec!`), and its data's precision. It is not the promotion of every parameter.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 3. What problem a mixed-precision call solves (T3; Q1; cases V8, V9)

**Decided** (maintainer, 2026-11-01). Parameters are rounded once to the model precision. A combined parameter is formed in `promote_type(W, factor types)`, left to right in the family's documented order, and rounded once, and the rounded values define the model: V3 = `0.5f0`, V8 = `0`, V9 = `0`, V10 = `2^-46`. The bound `eps/2` relative applies to ordinary in-range rounding with a normal result.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Revised after ChatGPT's review of Draft 4 (`07-chatgpt-review-of-draft4.md`).* ChatGPT now endorses (a), with `V8 = 0` and `V9 = 0`, and withdraws its earlier preference for (c). It corrects the justification, rightly:

- a stored `1f0` is exactly one, not uncertain by one `eps`;
- rounding a parameter does not make the algorithm that follows backward stable.

The honest reason for (a) is a deliberate choice of model: one rule, native-precision speed, and the same result for `NormL1(0.1)` and `NormL1(0.1f0)` on Float32 data. Rounding a parameter once changes it by at most `eps/2` relative, the smallest change the result's precision can express. The measurements cover soft thresholding on one Apple M3. They do not establish the cost for centered ridge, for GPUs, or for other implementations.

The rule is now stated per result (decision 5). Each result has a *model precision*, the working type of decision 2 for that result. Combined parameters such as `γλ` are formed in a precision no narrower than the model precision or the factors' promoted type, then rounded once (V10, V11).

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).* Two changes, both checked at `3f90a29`.

- *Decimal literals.* The claim that `NormL1(0.1)` and `NormL1(0.1f0)` give identical results on Float32 data was false. With `γ = 9f0`, `0.1·9` rounds to `0.9f0` but `0.1f0·9f0` rounds to `nextfloat(0.9f0)`, so at `x = nextfloat(0.9f0)` one returns `2^-24` and the other `0`. Two calls agree when their effective parameters convert to the same value.
- *Formation of combined parameters.* They are formed in `promote_type(W, factor types)`, left to right in the order the family documents, and that order defines ties.

### 4. Parameters whose rounding loses range (R3)

**Decided** (maintainer, 2026-11-01). R3: a rounding that loses range a representable result needs is recovered. The ordinary path is taken only when no partial operation loses range or subnormal bits that a later factor needs; a normal final scalar is not enough (V13).

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Revised after ChatGPT's review of Draft 7 (`14-chatgpt-review-of-draft7.md`).* The ordinary path is defined by the partial products, not by the final scalar. In weighted `NormL1` with `γ = 3·2^-538`, `λ = 2^-538`, `w = 2^1022` at `x = 2^-52`, an unprotected `γ·λ` rounds `3·2^-1076` up to `2^-1074`. The final threshold becomes `4·2^-54 = x` instead of `3·2^-54`, and the coordinate is set to zero, although both thresholds are normal.

The package's `_weight_product` already checks the partial product, and returns the right `2^-54` (checked at `3f90a29`). R3 now describes that behavior, and the case becomes a public regression test (V13).

### 5. Wider destination (T4; #366 D5)

**Decided** (maintainer, 2026-11-01). A wider destination is computed in `D` from exactly widened inputs, caches included (#362). Widening a coefficient rounded in a narrower type is not recomputing it.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Revised after ChatGPT's review of Draft 4 (`07-chatgpt-review-of-draft4.md`).* Draft 4's T3 formed `γλ` "in the promoted type of its factors", while T4 computed in a wider destination "from exactly widened inputs". The two disagree when the destination is wider than the factors.

- *V10.* `λ = 1 + 2^-23` and `γ = 1 - 2^-23` in Float32, `x = Float32[1]`, Float64 output. The exact product is `1 - 2^-46`. Multiplying the Float32 factors gives `1f0`, and the result `0`. Widening first gives `2^-46`. Master returns `0`, because the separable kernels compute in the argument's type.
  - Recommendation: `2^-46`. Combined parameters are formed in a precision no narrower than the model precision of the result.
- *V11, the consequence for fused calls.* With `x = Float32[1]`, a Float64 center `1 + 2^-28` and a Float64 gradient output, `value_and_gradient!` returns the Float32 value `0f0`, because the Float32 model's center is `1f0`. It writes the Float64 gradient `[-2^-28]`.
  - A fused call agrees with its two standalone oracles even when their model precisions differ, so a fused call is only an optimization.
  - Master returns the value `2^-57`, with the same gradient.

### 6. Functions that hold data (T2; #366 D6)

**Decided** (maintainer, 2026-11-01). The model precision includes the data's precision (T2, T3). Data are never rounded per call, and the argument is never rounded below the model precision (V12).

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).* The rule must also cover a wide argument with a narrow output. `QuadraticFunction(Float32[1;;], Float32[-1])` at the Float64 argument `1 + 2^-28`, with a Float32 gradient output, should give `2^-28`, a normal Float32 number (V12). Master rounds the argument first and returns `0`.

Revised recommendation, for every function, not only those that hold data: the model precision of a result is the widest of its delivery precision, the precision of the argument it is evaluated at (and of the direction, for `hessian_vec!`), and the data's precision (T3). The argument and the data are never rounded below it. The scalar of `value_and_prox!` describes the stored output, so its model precision is that of `f(y)`.

Data, parameters and bookkeeping are now defined in T2 (decision 46). The measurements cover one layout: n = 2000, a wide argument with a wide output. They do not cover every mixed layout, cached factorization or GPU call.

### 7. Float16 (T6; #366 D3)

**Decided** (maintainer, 2026-11-01). In 0.6, Float16 is rejected by role before writing: as argument, destination, direction or data. A Float16 parameter or step in an otherwise supported call is accepted. "Compute in at least Float32" is the rule for when support is added.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).* Float16 is rejected when the model precision would be Float16: a Float16 argument, output, direction or data. A Float16 parameter, scalar or array, in a wider call widens exactly and is accepted. This also removes the collision with #313's Float16 bound tests.

*Revised after ChatGPT's review of Draft 6 (`12-chatgpt-review-of-draft6.md`).* The rule now works by role. In 0.6, a Float16 argument, destination, direction or data is rejected before writing, independently of the model precision. Under T3, a Float16 argument with a Float64 output has Float64 model precision, so "rejected when the model precision would be Float16" did not reject it. A Float16 parameter or step in an otherwise supported call is accepted.

### 8. BigFloat, integers and other number types (T1, T7; #366 D9)

**Decided** (maintainer, 2026-11-01). BigFloat is supported in generic code, with caches keyed by `precision(BigFloat)`; the LAPACK families reject it. A fixed-width integer argument gives Float64, `BigInt` gives BigFloat, and an integer destination is rejected for fractional results.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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
for fractional results (#366's C7; #363, cited here before, is about integer sums that wrap).

### 9. Constructors (T5)

**Decided** (maintainer, 2026-11-01). Parameters are validated in their own type before any conversion, stored as given or promoted, never narrowed, and never turned from finite positive to `0` or `Inf` (#313, #333).

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

## Range and accuracy

### 10. Range of values (R1)

**Decided** (maintainer, 2026-11-01). Keep the #283 rule (R1): a normal value is returned with the ordinary accuracy, even when an intermediate leaves the range.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 11. Range of gradients and proximal outputs (R2; Q3)

**Decided** (maintainer, 2026-11-01). R2: every entry of a gradient and of a closed-form proximal output follows R1's rule in `D`. Matrix products, factorizations and searches follow A1.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 12. Subnormal results (R1, R4)

**Decided** (maintainer, 2026-11-01). Relative accuracy is promised only for normal results. A subnormal result may be off by about `floatmin` or be zero. A subnormal intermediate that makes a normal result wrong is a bug.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 13. The form of accuracy guarantees (A1; #366 D4)

**Decided** (maintainer, 2026-11-01). Per output (A1): each family states the problem it solves, its error scale and its guarantee.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 14. Thresholds and ties in closed forms (A2)

**Decided** (maintainer, 2026-11-01). A2's table. Entrywise decisions on the model's operands are exact and the same on every path and backend. Decisions on computed reductions are exact comparisons that may differ between paths or backends only within the reductions' error. Search decisions are certified (18). In-band spectral decisions follow 19.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Revised after ChatGPT's review of Draft 4 (`07-chatgpt-review-of-draft4.md`).* "Closed form" does not settle whether a decision needs certification: a group-norm threshold compares a computed reduction, `‖x_g‖`, with `γλ`.

Recommendation: closed-form decisions are exact comparisons on the operands as computed, with no tolerance and no certification band. Parameters are rounded under decision 3; reductions are computed to their A1 accuracy. The decision is then exact for an argument perturbed within the reduction's error.

Certification (decision 18) applies only to iterative searches, where the stopping test can be far from the returned point, as in #151.

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).* One policy per kind of decision (A2's table):

- an entrywise decision on the model's operands is exact, the same on every path and backend;
- a closed-form decision on a computed reduction is the same exact comparison, and two paths or backends may differ only within the reductions' error bound;
- a search decision is certified (decision 18);
- a spectral decision inside its band follows decision 19.

A6 and G1 are aligned with this. Draft 5's A6 had sent decisions on reductions to A3, and its G1 had promised identical branches across devices, which reductions cannot guarantee.

### 15. Sums and reproducibility (A6)

**Decided** (maintainer, 2026-11-01). Sums may be reordered. Decisions on sums follow A2: exact comparisons in closed forms, certification in searches.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

## Decisions and tolerances

### 16. Membership tolerance (A4)

**Decided** (maintainer, 2026-11-01). `√eps(T)·max(1, |bound|, s)`, with `s` the error scale of the tested quantity. No default tolerance is a fixed number independent of the precision.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 17. Rank and zero-mask threshold (A4, A5)

**Decided** (maintainer, 2026-11-01). `√eps(T)·σ₁`, the package's convention for membership-type decisions. It is not a universal numerical rank.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Corrected after ChatGPT's review of Draft 4 (`07-chatgpt-review-of-draft4.md`).* The numerical-rank thresholds differ by library:

- Julia's `rank`: `min(m, n)·eps·σ₁`;
- NumPy's `matrix_rank`: `max(M, N)·eps·σ_max`;
- MATLAB's `rank`: `max(size(A))·eps(norm(A))`.

The recommendation is unchanged.

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).* `√eps·σ₁` is not uniformly looser than `p·eps·σ₁`: the order reverses for `p > 1/√eps`, about 2,896 in Float32. The draft now describes the package's convention directly.

### 18. Decisions in multiplier searches (A3; #366 D1 and D8)

**Decided** (maintainer, 2026-11-01). Certify the stored output. Inside the band, refine, widening the arithmetic on the original call's effective operands without changing the model, or raise the dedicated exception before writing. The mechanism is chosen by measurement.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Revised after ChatGPT's review of Draft 7 (`14-chatgpt-review-of-draft7.md`).* Refinement and certification may widen the arithmetic on the *effective* operands of the original call, its parameters as converted under decision 3, but they do not change the model or the destination being certified. In V8, a Float64 call with the original threshold returns `2^-30`, which does not certify the Float32 model's `0`. A solver's retry with a wider destination is a new call, with its own model.

Test references are likewise evaluated after the model's conversions, except where R3 requires recovery.

### 19. LAPACK results near a boundary (A5; #366 D10)

**Decided** (maintainer, 2026-11-01). In-band results are returned as computed, which takes precedence over 18. Error bounds state three sources separately, with the storage term `‖E_D‖_F`, the actual or bounded conversion error, not `eps(D)·‖Y‖` in general. Rank truncation promises near-optimal error and the stored matrix's rank only when a bound certifies it, and a fused rank-indicator value describes the stored matrix. `IndStiefel` has its own row.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Revised after ChatGPT's review of Draft 4 (`07-chatgpt-review-of-draft4.md`).* Draft 4's A5 claimed that a projection onto the PSD cone or a rank ball, or a spectral proximal map, is accurate to `c·p·eps·‖X‖`.

- *Where it holds.* The PSD projection and the proximal maps of convex spectral functions are nonexpansive. A backward error in `X` gives at most the same forward error, in the Frobenius norm.
- *Where it fails.* A rank-ball projection is not continuous. `diag(1 + δ, 1)` and `diag(1, 1 + δ)` are `δ√2` apart, but their best rank-one approximations keep different axes and are about `√2` apart. At `δ = 0` the best approximation is not unique.

Recommendation for rank truncation: promise rank feasibility and near-optimal error, `‖X - Y‖ ≤ ‖X - X_k‖ + c·p·eps·‖X‖`. Promise closeness to a particular best approximation only when the gap `σ_k - σ_{k+1}` is well above `eps·‖X‖`.

Also state that this decision's policy, returning in-band results as computed, takes precedence over decision 18's refine-or-throw for the LAPACK families.

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).*

- *Nonexpansiveness bounds only part of the error.* It bounds the propagated input perturbation, not the evaluation of the scalar map, the reconstruction, or the rounding to `D`. Example: `LogDetLoss` with `S = 0`, `w = 1` and `γ = 2` at `X = 0` gives `√2·I`, which is not representable although `‖X‖ = 0`. A5 now states the three sources separately, with the output scale `‖Y‖`.
- *Rank truncation* names the Frobenius norm and the numerical rank of the *stored* output.
- *`IndStiefel`* gets its own row: orthonormal to within `c·p·eps`, near-optimal, with closeness to the polar factor depending on the smallest singular value.

*Revised after ChatGPT's review of Draft 6 (`12-chatgpt-review-of-draft6.md`).* The rank and `IndStiefel` guarantees are about the stored output, so they include the error of storing in `D`, not only `eps(W)`. Both examples were checked at `3f90a29`.

- The rank-one projection of `diag(1 + 2^-28, 0)` into Float32 stores `diag(1, 0)`. The optimal residual is `0`; the stored one is `2^-28`.
- `IndStiefel` of the column `(1, 1)` into Float32 gives `|y'y - 1| = 5.0e-8`. ChatGPT's `3.42e-8` assumed both entries round alike; the package stores two different Float32 values.

The stored matrix's numerical rank is evaluated in `D`. The decomposition before the write and the certificate for what was written are tested separately.

*Revised after ChatGPT's review of Draft 7 (`14-chatgpt-review-of-draft7.md`).* The stored matrix's numerical rank cannot be promised unconditionally. The bound `σ_{k+1}(Y) ≤ σ_{k+1}(Y_W) + ‖E_D‖₂` is promised instead, with rank at most `k` only when that bound is below the cutoff.

*Example, checked at `d00c7b86` in a worktree of master.* The rank-one `X = h·v·v'`, with `h = nextfloat(0f0)` and `v = (1, 5/8)`, stored in Float32 becomes `[2 1; 1 1]·h`, whose singular-value ratio is `0.146`, far above the cutoff. ChatGPT's exact projection gives `[1 1; 1 0]·h` with ratio `0.382`; the conclusion is the same.

The fused value of `IndBallRank` must then describe the stored matrix (O2). At `d00c7b86` it returns `0` while the standalone test of the stored matrix is `Inf`. At `3f90a29` both were `0`, because the old `max(1, σ₁)` floor, removed by #375, hid the difference.

Recommendation: the value is `0` when the bound certifies membership, which needs no second decomposition on ordinary inputs. Otherwise the stored matrix's membership is evaluated.

## Special values

### 20. NaN in the argument (E2; Q2)

**Decided** (maintainer, 2026-11-01). NaN propagates to every result that depends on it, and never becomes zeros or `+Inf` through a failed comparison. This reverses #343's `Inf` for NaN entries and resolves #349 as NaN.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 21. Infinite entries (E3)

**Decided** (maintainer, 2026-11-01). The formula or limit where determined: a norm or an indicator gives `+Inf`. Where the value is undetermined, including path-dependent limits, it is NaN.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Corrected after ChatGPT's review of Draft 4 (`07-chatgpt-review-of-draft4.md`).* Draft 4 said a convex quadratic gives `+Inf` at infinite entries. That is wrong in general: `(x₁ - x₂)²/2` tends to `0` along `(t, t)` and to `1/2` along `(t, t + 1)`. At `(Inf, Inf)` the value is not determined, so it is NaN, as the formula's `Inf - Inf` gives. The examples are now a norm (`+Inf`) and an indicator (`+Inf`).

### 22. Zero level, zero weights and infinite weights (E4)

**Decided** (maintainer, 2026-11-01). With 40: a zero level removes the finite part and keeps domain constraints, infinite weights included. It is the zero function only when there are none.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 23. Exceptional values across precisions (E5; #350)

**Decided** (maintainer, 2026-11-01). Exact `Inf` and NaN results survive conversions (#350). Non-finite intermediates from finite inputs are recovered (R1–R3).

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** Is an `Inf` or NaN that is the exact answer, such as an
indicator outside its set, kept when a wrapper converts or recomputes in
another precision?

**Example.** #350: mixed-precision `Precompose` discards an indicator's
`Inf`.

**Recommendation.** Keep exact exceptional values through conversion.
Recover non-finite intermediates that arise from finite inputs (decisions 4
and 11).

**Precedent.** IEEE conversions preserve `Inf` and NaN.

### 24. Domains and masks (E6)

**Decided** (maintainer, 2026-11-01). The value outside the domain is `+Inf`; at a boundary, one-sided limits where finite; small tails stay explicit; masks document their precedence.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

## Steps

### 25. Negative and NaN steps (P1; #311)

**Decided** (maintainer, 2026-11-01). `ArgumentError` before writing (#311).

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 26. Zero and infinite steps (P1)

**Decided** (maintainer, 2026-11-01). Endpoints only where the family names them: at `γ = 0`, the projection onto the closure of the domain; at `γ = Inf`, the projection onto the minimizers, for convex functions that have them; scalar steps on indicators; ridge. Other endpoints are rejected before writing.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Corrected after ChatGPT's review of Draft 4 (`07-chatgpt-review-of-draft4.md`).* Draft 4 said the zero-step limit of a separable function is the identity.

- *Convex functions.* The limit is the projection onto the closure of the domain, which is the identity only inside it. For `f(z) = -log z` at `x = -1`, `prox_{γf}(x) = (-1 + √(1 + 4γ))/2`, which tends to `0`.
- *Nonconvex separable functions with full domain* (`MCP`, `SCAD`, `NormL0`). The limit is the identity.

### 27. Diagonal metrics that do not fit the working type (P3; #366 D7)

**Decided** (maintainer, 2026-11-01). A retained scale times ratios; a ratio that underflows is rejected before writing (#257).

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

## Agreement between oracles

### 28. The point a joint value describes (O1, O2)

**Decided** (maintainer, 2026-11-01). `value_and_gradient!` agrees with its standalone oracles, even across precisions (V11). `value_and_prox!` returns the value of the converted model at the stored output.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 29. Approximate projections (O3)

**Decided** (maintainer, 2026-11-01). Return `0`, documented as not a feasibility certificate.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 30. Wrappers that round between stages (O4)

**Decided** (maintainer, 2026-11-01). Keep the staging. Lipschitz stages carry a bound with their amplification factors; discontinuous inner maps promise the staged model and its tie rule.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).* The additive bound holds only for Lipschitz inner maps, with each stage's amplification factor. For a discontinuous inner map the staged model and its tie rule are promised instead. `Translate(NormL0(0.5), [-2^-26])` at `1.0` gives `Float32[-2^-26]` with a Float32 output and `[1.0]` with a Float64 output (checked at `3f90a29`).

### 31. Errors before writes (O6)

**Decided** (maintainer, 2026-11-01). By kind. Validation raises `DimensionMismatch` for sizes, otherwise `ArgumentError`, before writing. Infeasibility found during a search raises `ArgumentError` before writing. Certification failures and unrepresentable results raise the dedicated exception before writing. A failure inside an external library is documented per family.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** Keep the settled rule that a call raising an `ArgumentError`
leaves every caller-owned output unchanged, tuple blocks and iterate storage
included?

**Background.** A prescribed NaN result is a result, not an error. There is
no rollback after a failed external factorization; the family documents
which outputs remain valid. Gaps: #239, #323, #371.

**Precedent.** Julia's `mul!` and `copyto!` check dimensions before writing;
C++ calls this the strong exception guarantee.

**Recommendation.** Keep.

*Revised after ChatGPT's review of Draft 4 (`07-chatgpt-review-of-draft4.md`).* The guarantee on outputs depends on the kind of failure, not on the exception type.

1. *Validation:* shapes, types, steps and parameters, detected before any work. The call raises an `ArgumentError` and leaves outputs unchanged. Settled.
2. *Certification failure in a search* (decision 18), detected during the work. It is raised before the output is written, so outputs are unchanged.
   - Recommendation: a dedicated exception type. It lets a solver tell "cannot certify in this precision; retry in Float64" apart from "your input is invalid".
   - Precedent: LinearAlgebra's `PosDefException`, `SingularException` and `LAPACKException`.
3. *Failure inside an external library,* such as a LAPACK error. That library's exception propagates, and the family documents which outputs remain valid.

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).* Sizes raise `DimensionMismatch`, not `ArgumentError` (decision 41). Infeasibility found during a search and arithmetic stopgaps get their own categories (decision 42). Tests include a late certification failure in a composed or in-place call.

### 32. Aliasing (O7; #367)

**Decided** (maintainer, 2026-11-01). Keep the current rule; #367 audits.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 33. Sparse and symmetric storage (M1, M2)

**Decided** (maintainer, 2026-11-01). Keep (#142).

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** Keep the rules for implicit zeros and stored multiplicity?

**Current rules.** Implicit zeros of a sparse iterate are entries: a
centered ridge moves them, and `NegLog` is `+Inf` at them. A stored
off-diagonal of a `SymmetricSparseIterate` counts twice in reductions and is
written once. The symmetric-parameter rule of #142 applies.

**Recommendation.** Keep; both are settled.

## Devices, cost and the solver

### 34. GPU arrays (G1–G6)

**Decided** (maintainer, 2026-11-01). The same meaning as on the CPU under A2's rules. No bitwise promise, no fast-math where NaN and `Inf` must be told apart, and host-side scalar decisions (#322).

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 35. Cost policy (C1–C3; #366 D2)

**Decided** (maintainer, 2026-11-01). A 5% slowdown triggers investigation; more than 10% paid only by narrow types needs sign-off; narrow-only machinery leaves Float64 within noise; a small fixed cost per call is accepted. Numerical non-regression is decision 38, not bits.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

### 36. What the solver decides (documentation only)

**Decided** (maintainer, 2026-11-01). Documented: `Inf` means outside the domain or genuine overflow; test acceptance positively and test `isfinite`; commit the checked point; no movement does not distinguish stagnation from a fixed point.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

*Corrected after ChatGPT's review of Draft 4 (`07-chatgpt-review-of-draft4.md`).*

- *`Inf` values.* An `Inf` trial value means the trial is outside the domain, *or* its value genuinely overflows. In V7 the point is in the domain, the value is `+Inf`, and the gradient is `[4.0]`. A solver that requires a finite objective rejects both, but should not report infeasibility.
- *No movement.* A step that changes no stored entry is stagnation *or* an exact fixed point; no movement alone proves either.

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).*

- *Finiteness.* A solver that requires finite evaluations tests `isfinite` as well as decrease, since `Inf <= Inf` is true.
- *No movement.* It does not distinguish stagnation from a fixed point.

### 37. Where this lives, and in what order

**Decided** (maintainer, 2026-11-01). PR #385 is the record. `docs/src/numerics.md` carries the contract, with summaries in the interface, architecture, extending and migration pages. #366 becomes the implementation tracker, starting with a reconciliation table.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

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

## Added after Draft 4

### 38. What "unchanged" means: the same bits, or the same accuracy (C1, R1, R3)

**Decided** (maintainer, 2026-11-01). Accuracy, not bits: results stay within their stated accuracy; decisions as A2 specifies them; exact values and classes unchanged. Bitwise comparison is a test technique, not a promise.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

*Added 2026-10-31 after the maintainer's question on #385: "why we insist
of things being bit for bit the same … I think things should be the same
within precision of the type, not necessarily bit for bit."*

**Decide.** When code changes, or when two paths compute the same result, must
results agree bit for bit, or within the stated accuracy?

**Where bits are promised today.**

- The interface page's overflow rule: "Ordinary inputs therefore give the
  same bits as the ordinary expression", and in its cost paragraph, "keeps
  ordinary results bit for bit what they were".
- #366's C0 and Draft 4's C1: all-Float64 calls keep their bits, except
  listed correctness repairs, each named in its PR.
- #238: results with the argument and the parameters in one precision must
  stay bitwise unchanged.
- Six test files compare results bitwise as a regression technique:
  `conjugate_steps.jl`, `fibers.jl`, `spectral_domain.jl`,
  `spectral_overflow.jl`, `precision_mixed.jl` and `stepchecks.jl`.

**Why bits were used.** A bitwise check is the cheapest strong test that a
refactor did not change ordinary results, for example that adding a recovery
path left the fast path alone. It also keeps solver runs identical across
ProximalBase versions on one machine.

**Options.**

- **(a) Bits.** Keep the bitwise promise for ordinary inputs and for
  all-Float64 calls.
  - *Pros.* Trivial to test. It catches any change, including a small loss of
    accuracy that a loose tolerance would miss. Solver runs are identical
    across versions on one machine.
  - *Cons.* It forbids harmless improvements: fused multiply-add
    (`muladd`), pairwise or compensated sums, SIMD reorganization, a more
    accurate formula. Each becomes a listed exception. It cannot hold across
    Julia versions, BLAS libraries (OpenBLAS, MKL) or CPUs anyway, so users
    cannot rely on it. And tests fail on changes that are not bugs.
- **(b) Accuracy.** A result stays within its stated accuracy (decision
  13), decisions and exact values stay exact, and classes (finite, `Inf`,
  NaN) do not change. Last-bit changes within the accuracy are allowed; a PR
  that changes all-Float64 results says so.
  - *Pros.* It promises what users can rely on and allows improvements.
    Tests check against independent references, which catch real bugs.
  - *Cons.* A small loss of accuracy, such as 1 ulp becoming 10, passes
    unless the tolerances are tight, so tests need ulp-based bounds and
    references. In ill-conditioned or nonconvex problems, a last-bit change
    can change a solver's iteration count or path.
- **(c) No rule.**

**What stays exact under (b).** Everything discrete or exact by construction:

- decisions: zero or nonzero, threshold branches, ties, membership, ranks,
  and the branch agreement of the entrywise, coordinate, whole-array and
  aliased paths (decision 14);
- exact values: a clamp returns the bound; an identity step copies the
  argument, including the sign of zero; a projection of a point already in
  the set returns it unchanged;
- classes: finite, `Inf`, NaN.

**Precedent.**

- PyTorch's reproducibility notes say completely reproducible results are not
  guaranteed across releases, commits or platforms.
- Julia's `muladd` may or may not be fused, and its documentation says the
  result can differ across machines, and even on one machine.
- LAPACK promises error bounds, not bits, and OpenBLAS and MKL give different
  bits. MKL's bit-reproducible mode (CNR) is opt-in, as is the reproducibility
  clause of IEEE 754-2008.
- Julia's own tests check elementary functions in ulps, not bits.

**Recommendation.** **(b)**, as the maintainer suggests.

- The fast-path design of decision 10 stays, justified by speed: ordinary
  inputs take the ordinary computation without paying for scaling.
- Bitwise comparison stays available as a test technique for a change meant
  to leave results alone, but a bitwise difference within the accuracy is not
  a regression.
- C1 and the interface page's wording change.
- The existing bitwise tests stay where they test exact values or decisions,
  and become reference comparisons elsewhere.

**Changes.** Documentation and test practice only; no results change.

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).* "Decisions stay exact" is refined by decision 14's table. Entrywise decisions on identical operands stay exact. A decision on a computed reduction may change only within the reduction's error bound. A family whose quantitative bound is still pending is checked against independent references with a stated error budget, not against a blanket `isapprox`.

### 39. `LogDetLoss`: the value at a stored output that rounds to singular, and saturation (O2, O3, R4)

**Decided** (maintainer, 2026-11-01). The value is that of the stored output, so a singular stored output gives `Inf`; the verification strategy is chosen by measurement. Matrix reconstructions may saturate within their normwise bound, and R4 applies to entrywise results. It is a documented transitional exception until implemented.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

*Added after ChatGPT's review of Draft 4 (`07`), which found an existing
exception that Draft 4 omitted.*

**Decide.** `LogDetLoss`'s docstring, at `3f90a29`, says two things that
conflict with Draft 4:

- "The log-determinant uses the mapped roots and the trace uses the stored
  output. That output can round to a singular matrix, so the returned value
  can be finite even when `f(Y)` is `Inf`." This conflicts with O2, under
  which the value is that of the stored output.
- "Entries exceeding `floatmax` within the reconstruction's rounding bound
  saturate." This conflicts with R4, which allows no saturation.

**Options for the value.**

- *(a) Keep it as a named exception to O2.* No cost. But a solver can hold a
  matrix outside the domain with a finite objective.
- *(b) Migrate to O2.* Compute the log-determinant of the stored output, for
  example from its Cholesky factorization, so that a singular stored output
  gives `Inf`.
  - *Cost.* A Cholesky factorization is about `n³/3` flops, small next to
    the eigendecomposition the proximal operator already performs, which is
    several times `n³`. To be measured.
  - *Pros.* The solver knows whether the matrix it holds is in the domain.

**Options for saturation.**

- *(a) Allow it for matrix reconstructions,* within their normwise bound
  (decision 19), and restrict R4's "no saturation" to entrywise results
  (decisions 10 and 11). An entry saturating at `floatmax` lies within the
  reconstruction's normwise error.
- *(b) Return `Inf` in those entries.* Then the whole matrix is unusable.

**Recommendation.**

- *Value:* (b), migrate. Add an acceptance case, an input whose smallest
  mapped root is below `eps·‖Y‖` so that the stored output rounds to
  singular, and measure the cost. Until then, list it as the second named
  exception to O2.
- *Saturation:* (a). R4 applies to entrywise results.

*Revised after ChatGPT's review of Draft 5 (`09-chatgpt-review-of-draft5.md`).* Draft 5's acceptance case ("a mapped root below `eps·‖Y‖`") was wrong: `diag(1, 2^-60)` is positive definite in Float64.

The new fixture, checked at `3f90a29`: `LogDetLoss(zeros(2, 2))`, step 1, at `[0 2^15; 2^15 0]`, with a Float32 output. All four stored entries are `16384f0`, so the stored matrix is singular and `f(Y) == Inf`, while the fused value is `0`. The exact proximal point has determinant one. The control is the Float64 output, whose stored matrix is positive definite.

The draft now states current behavior, the 0.6 target and the approval status separately.

## Added after Draft 5

*Decisions 40–47 answer the gaps in the contract text found by the screen of
open issues (`10-impact-on-open-issues.md`).*

### 40. A zero level against domain constraints (E4)

**Decided** (maintainer, 2026-11-01). Zero scaling keeps domain constraints and infinite weights, and is rejected when no domain-preserving proximal representation exists (`NegLog`).

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** Draft 5 said a zero level makes the function "identically zero".
The calculus page and PR #300 (#141) say instead that an infinite coordinate
weight fixes its entry at zero "at every level", and that zero scaling "keeps
domain constraints". Which takes precedence? #339 measures a weight scan at a
zero level because of this.

**Options.**

- *(a) A zero level removes the finite part and keeps domain constraints:*
  an indicator's set, infinite weights, a restricted domain. A function
  without domain constraints becomes the zero function.
- *(b) A zero level is always the zero function.* This reverses PR #300:
  `NormL1(0, [0, Inf])` would no longer fix its second entry.

**Precedent.** The package's own `Scaled(f, 0)`. It also matches a convention
of the package: an infinite weight is a constraint, not a large penalty.

**Recommendation.** (a). It keeps PR #300, so #339 is unaffected.

*Revised after ChatGPT's review of Draft 6 (`12-chatgpt-review-of-draft6.md`).* The existing qualification stays: zero scaling is rejected when no domain-preserving proximal representation exists. `NegLog` at zero scale would be the indicator of the open half-line, which has no closest point to `-1`. `Scaled(NegLog(), 0)` throws today (checked at `3f90a29`). This differs from the limit of `NegLog`'s proximal point as a positive step tends to zero (decision 26).

### 41. The exception for sizes (E1, O6)

**Decided** (maintainer, 2026-11-01). `DimensionMismatch`.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** Draft 5 said shape violations raise `ArgumentError`. `src/`
throws `DimensionMismatch` for size mismatches in about 150 places
(`src/utils.jl:225, 242`), and #382 proposes it.

**Options.**

- *(a) `DimensionMismatch` for sizes and shapes, `ArgumentError` for other
  invalid input.*
- *(b) `ArgumentError` for everything.*

**Precedent.** LinearAlgebra raises `DimensionMismatch` for mismatched sizes
(`mul!`, `copyto!`).

**Recommendation.** (a). It is the current code and Julia's convention. The
promise that outputs stay unchanged is the same for both.

### 42. Infeasibility found during a search, and arithmetic stopgaps (O6)

**Decided** (maintainer, 2026-11-01). Infeasibility found during a search raises `ArgumentError` before writing. An arithmetic stopgap raises the dedicated exception before writing.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** O6 had no category for two kinds of failure:

- infeasibility a search discovers during its work, such as a constraint that
  misses the domain (#323), which today can write earlier blocks;
- #244's check in `Conjugate` that a finite input gave a non-finite result.

**Options for infeasibility.**

- *(a) An `ArgumentError` that names the infeasibility, before writing.* The
  input is invalid, even though detecting it needed the search.
- *(b) The dedicated certification exception.*

**Options for the stopgap.**

- *(a) The dedicated exception, before writing, until R2 and E5 remove the
  check.* The input is valid, but the working precision failed.
- *(b) An `ArgumentError`,* as today.

**Recommendation.** (a) for both. Infeasibility describes the input, so it is
an `ArgumentError`. The stopgap is a precision failure, so it is the dedicated
exception. In both cases outputs stay unchanged; for block functions that
means deciding before writing any block (#323).

### 43. The scope of the contract (S2)

**Decided** (maintainer, 2026-11-01). Every oracle. `kkt_violation` and model weights follow the value rules. `restrict` as the interface states it: the objective up to a constant, the gradient by its supported entries, the proximal operator over the support. Quadratic models are judged by their effective coefficients: a finite one is returned (P3); an unrepresentable one raises the dedicated exception before writing `q` (P2, P2m).

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** Draft 5's S2 covered values, gradients, proximal outputs and
decisions. It left out `kkt_violation`, `quadratic_model` and
`quadratic_model!`, `hessian_vec!`, `lla_weights!` and `restrict`.
ProximalSolvers' active-set solver relies on `kkt_violation` and `restrict`,
and #317 (Poisson IRLS weights overflowing) depends on this scope.

**Options.**

- *(a) Every oracle.* `kkt_violation` and quadratic-model weights follow the
  value rules. `hessian_vec!` and LLA weights follow R2, or A1 for matrix
  products. `restrict` is structural: the restricted function's oracles equal
  the original's on the support, with values allowed to differ by a constant
  (O5).
- *(b) Leave them undefined.*

**Recommendation.** (a). For #317, this favors that issue's second option:
compute the working response in a finite form, and let a weight that
genuinely overflows be `Inf` (R4), rather than throwing.

*Revised after ChatGPT's review of Draft 6 (`12-chatgpt-review-of-draft6.md`).* Two corrections.

- *`restrict`.* Draft 6 said the restricted function's "oracles equal the original's on the support". The interface's own wording is more precise, and is restored: the objective restricts up to a finite additive constant, and the gradient restricts by taking its supported entries. The restricted proximal operator minimizes over the support; it is not a slice of the full proximal point. For `Q = [2 1; 1 2]` at `(1, 0)`, the full proximal point is `(3/8, -1/8)` and the restricted one is `1/3` (checked at `3f90a29`).
- *Quadratic models.* A quadratic model is a function the solver evaluates. Allowing an IRLS weight to become `Inf` makes the model unusable: for one Poisson observation with response `0` at `η = 710`, the weight `exp(710)` overflows. The model's value at `709.5` is `exp(710)/8 ≈ 2.79e307`, a normal number, and a model holding `Inf` cannot give it.
  - *Today:* weight `Inf` and working response `NaN` (checked at `3f90a29`). The exact working response is `709`.
  - *Options:* (i) return a model that keeps enough scale to evaluate it, a new representation; or (ii) raise the dedicated exception before writing `q` when the model cannot represent its coefficients.
  - *Revised recommendation:* (ii) for 0.6, with the working response computed in a finite form. An intentional infinite observation weight is a different thing from an overflowed curvature. Tests evaluate the returned model, not only its fields. This replaces the first recommendation's lean towards "let the weight be `Inf`".

*Revised after ChatGPT's review of Draft 7 (`14-chatgpt-review-of-draft7.md`).* The rule applies to the *effective* coefficient, not to the first intermediate that overflows. With observation weight `1/8` at `η = 710`, the coefficient `exp(710)/8 ≈ 2.79e307` is finite and must be returned; today the model holds `Inf` (checked at `d00c7b86`), a range bug under R2.

Tests:

- the allocating case P2 (an overflowing coefficient raises the dedicated exception);
- a mutating counterpart P2m, which starts from a valid `q`, overflows a *later* observation, and checks that `q` is unchanged;
- the success control P3.

### 44. Gradients outside the domain (E6)

**Decided** (maintainer, 2026-11-01). NaN, for the derivative oracles that are supported.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** E6 said values outside the domain are `+Inf`, but gave no rule
for gradients there. #226 proposes "`NaN` derivatives outside the domain, as
for `LogDetLoss`".

**Options.**

- *(a) NaN.* No derivative exists there.
- *(b) Throw.* That costs a domain check in every gradient call, and breaks
  line searches that evaluate a gradient at a rejected trial.
- *(c) `Inf`,* which means nothing for a gradient.

**Precedent.** `LogDetLoss` already returns NaN gradients outside the cone.

**Recommendation.** (a).

### 45. Values of wrappers around arbitrary inner functions (R1)

**Decided** (maintainer, 2026-11-01). A private scaled-result capability; homogeneity recovers overflowing transformed arguments of norms (W1 = `M/2`); other combinations are listed as not covered. No public hook in 0.6.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** R1 promises a normal value whatever the intermediates. A wrapper
around an arbitrary inner function, such as `Scaled(f, α)` with tiny `α` and
an `f` whose value overflows, cannot recover a value from an inner `Inf`
(#357, #359). S4 holds user functions to the same rules. PR #384's decisions
A and B face the same question for functions of a spectrum.

**Options.**

- *(a) A public scaled-value hook.* An inner function may return its value as
  a significand and a power-of-two exponent, and wrappers use it. The built-in
  functions implement it; a user function joins R1 inside wrappers by
  implementing it; without it, the wrapper uses the ordinary value, which is
  correct whenever the inner value does not overflow.
- *(b) A private hook,* with user-defined inner functions listed as a named
  exception.
- *(c) A named exception* for wrappers around non-homogeneous functions, as
  #357 proposes.

**Pros and cons.**

- (a) is one rule with a documented extension point. It adds public API,
  which is easiest to do in the breaking 0.6.
- (b) and (c) add no API, but they are the per-function exceptions this
  package avoids.

**Precedent.** LAPACK's `lassq` returns a sum of squares as a scale and a sum
for the same reason. ProximalBase's internal value hooks (#283) already work
this way.

**Recommendation.** (a), documented in "Writing your own function".

*Revised after ChatGPT's review of Draft 6 (`12-chatgpt-review-of-draft6.md`).* The public hook solved only half of the problem.

- *What it can do.* It recovers an inner *result* that is out of range at a representable input: `Scaled(f, α)` with a tiny `α`.
- *What it cannot do.* It cannot recover a transformed *argument* that has already overflowed. `Translate(NormL1(0.25), [-M])([M])`, with `M = floatmax(Float64)`, has inner coordinate `2M` and value `M/2`; once the wrapper passes `[Inf]`, no return format can recover it. Today it returns `Inf` (checked at `3f90a29`).

Revised recommendation:

- *Inner results:* a private scaled-value capability, implemented by the built-in functions.
- *Transformed arguments:* recovery needs a supported transformation, either an algebraic identity (homogeneity, for norms) or an inner method that takes a scaled argument. Combinations without one are listed as not covered in the support matrix, with issues (#357, #359). This is a limitation for now, and it is stated.
- *User-defined inner functions:* covered once the capability is made public, which is deferred until a second use or a concrete third-party need justifies it. Committing third-party authors to a public hook that cannot deliver the advertised guarantee would be worse than a stated gap.

S4's promise for user functions is qualified accordingly in the support matrix.

*Revised after ChatGPT's review of Draft 7 (`14-chatgpt-review-of-draft7.md`).* W1's status was left as "supported or not covered".

Recommendation: supported. `Translate` and `Precompose` of a positively homogeneous inner function, the norms, recover through homogeneity, so W1 returns `M/2`. This is #357's own proposal for homogeneous inner functions. Other combinations are listed as not covered, with their issues.

### 46. The normals and offsets of sets (T2)

**Decided** (maintainer, 2026-11-01). (b): matrices, the vectors in their equations, and the normals and offsets of linear sets are data. Levels, coordinate weights, centers, bounds and steps are parameters.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** Are the normal `a` and offset `b` of `IndHyperplane`,
`IndHalfspace` and `IndHyperslab` parameters, rounded to the model precision
(T3), or data, kept at their own precision (decision 6)? Today the code
computes the residual and the inside/outside decision with unrounded
`Float64` normals (`src/functions/sets.jl:849, 858`), which treats them as
data. PR #353 does the same.

**Options.**

- *(a) Parameters.*
  - *Pros.* The same rule as centers and weights, and native-precision speed.
  - *Cons.* A one-row `IndAffine`, whose matrix is data, and an
    `IndHyperplane` with the same numbers can differ in the last bits of a
    mixed call.
- *(b) Data.*
  - *Pros.* Matches the code today and the one-row `IndAffine`.
  - *Cons.* A second category of vectors that behave like matrices, and wider
    arithmetic on every mixed call.

**Recommendation.** (a). The line between data and parameters is "matrices,
and the vectors in the same equations as a matrix". It is justified by cost:
converting a matrix every call is O(np), and BLAS needs one type. A normal
is a vector of length `n`, like a center. PR #353's coefficient is then
formed in the model precision, with range recovery under R3.

*Revised after ChatGPT's review of Draft 6 (`12-chatgpt-review-of-draft6.md`).* The effect of the line between data and parameters is not confined to the last bits. Take the Float64 normal `(1, 1 + 2^-25, 1)`, offset `0`, and the Float32 point `(2^25, -2^25, 0)` with a Float32 output.

- As a parameter, the normal rounds to `(1, 1, 1)`, the point lies on the plane, and the projection's third coordinate is `0`.
- As data (a one-row `IndAffine`), the residual is `-1`, and the third coordinate is about `1/3`.

Today both `IndHyperplane` and the one-row `IndAffine` give `0.3333333` (checked at `3f90a29`). So option (a) changes `IndHyperplane`'s answer for this input and makes it differ from `IndAffine`'s. This is a choice of model, and the documentation must say so.

A third option was not in the first version:

- *(c) Every array is data; only scalars are parameters.* "Scalars you pass (levels, steps) are rounded to the working precision; arrays you pass (weights, centers, bounds, normals, matrices) are used as stored."
  - *Pros.* Simple to state. `IndHyperplane` and a one-row `IndAffine` agree, as do `LinearFunction(c)` and `QuadraticFunction(0, c)`. The cost argument that made scalars expensive does not carry over to arrays: a stored array must be read and converted entry by entry under either rule, so keeping its entries costs about the same as rounding them. Scalars are where rounding once saves the per-entry wide arithmetic.
  - *Cons.* It reverses part of decision 3: V9 returns `2^-57`, the value with the stored center, at the point the proximal operator, also computed with the stored center, wrote. Weighted kernels with Float64 weights on Float32 data then compute their thresholds per entry in Float64. That cost is unmeasured; it should be small for memory-bound loops but is uncertain on GPUs.

The recommendation stays (a), for agreement with decision 3 as reviewed. But (c) is a coherent alternative that removes the `IndHyperplane`/`IndAffine` difference, and the maintainer may prefer it. If (c) is chosen, V9's target becomes `2^-57`, and decision 3 is restated as "scalars rounded once".

*Revised after ChatGPT's review of Draft 7 (`14-chatgpt-review-of-draft7.md`).* ChatGPT votes for **(b)**: the defining normal and offsets of `IndHyperplane`, `IndHalfspace` and `IndHyperslab` are geometric data. A user choosing between `IndHyperplane(a, b)` and a one-row `IndAffine(A, b)` is choosing how to write the same constraint, and the equation should mean the same in both, though the algorithms need not give identical bits.

ChatGPT rejects (c), every array as data, as a fix for this problem alone:

- it would change centered ridge: V9's value and V11's scalar become `2^-57`;
- it would conflict with T6's acceptance of Float16 parameter arrays.

**Revised recommendation: (b).**

- It keeps today's behavior: `IndHyperplane` and the one-row `IndAffine` both give `0.3333333` (checked at `3f90a29`).
- It agrees with PR #353's arithmetic.
- It keeps V9 and V11 at `0`.

The line between data and parameters becomes: matrices, the vectors in their equations, and the defining geometry of linear sets are data; levels, coordinate weights, centers, bounds and steps are parameters. The cost of the mixed geometry path is to be measured. Under T6, a Float16 normal is data and is rejected.

### 47. Non-finite linear coefficients (M1)

**Decided** (maintainer, 2026-11-01). Rejected at construction.

*The analysis below is the history of this decision. Where it differs from the line above, the line above governs.*

**Decide.** May the coefficients of `LinearFunction` and `Tilt` be `±Inf`? If
they may, a sparse dot cannot skip unstored entries, since `Inf·0` is NaN
(#341).

**Options.**

- *(a) Reject non-finite coefficients at construction.*
- *(b) Allow them, and read every entry.*

**Pros and cons.** (a) costs one check at construction, and keeps sparse dots
at O(nnz). A linear function with an infinite coefficient is not a
real-valued function, so (b) gains nothing. Constructors already reject NaN
levels (decision 22).

**Recommendation.** (a).

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
