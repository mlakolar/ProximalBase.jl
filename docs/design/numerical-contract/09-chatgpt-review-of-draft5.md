# ChatGPT's review of Draft 5

*Received from the maintainer on 2026-11-01 and recorded verbatim. It reviews `08-claude-draft5.md` and `05-decisions.md` at `36d1bddc`.*

---

# Review of Draft 5: the ProximalBase.jl numerical contract

**Reviewed revision:** `36d1bddcfedc156c1fb2aaa1c164d418672b6e3b` · **Disposition:** approve the direction for implementation planning; revise before treating it as the normative public contract. This is a review, not a new contract or a set of changes to the repository.

I reviewed the [pinned draft](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/docs/design/numerical-contract/08-claude-draft5.md), its [decisions record](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/docs/design/numerical-contract/05-decisions.md), and selected source at that revision. The numerical examples below are independent calculations, not executions of the Julia package. The draft’s “today” column describes its authors’ tests at `3f90a29`, not new conformance results from this review.

## What is resolved

Keep the converted-model decision: V8 and V9 return zero for different, clearly specified reasons. V10 closes Draft 4’s wider-destination product ambiguity. V11 makes a fused value/gradient call answer the same questions as its standalone oracles even when the two results have different precisions. The rank-truncation discussion now distinguishes an excellent approximation from a particular unstable choice of singular vectors. `LogDetLoss` is an explicit, rather than accidental, exception. The separate treatment of validation, certification failure and library failure is valuable. I also endorse decision 38: pin exact answers and specified decisions, not incidental arithmetic bit patterns across releases.

## 1. Give a wide-argument/narrow-output answer for matrix-backed functions

**T2/T3, decision 6.** The draft changes the policy from “work in the data’s precision” to “use the wider of data and call precision,” and says not to discard the argument’s digits. But for a written result, call precision elsewhere means `D`. Take the quadratic `f(x) = x²/2 − x` with `A = [1]`, `b = [-1]`, and the constant all stored in `Float32`. Give it the `Float64` argument `1 + 2⁻²⁸` and ask for a `Float32` gradient. The exact gradient is `2⁻²⁸`, a normal `Float32`. Rounding `x` first gives zero.

**Recommended target:** retain `2⁻²⁸`. For these families, define the model precision as at least the widest of stored numerical data, the argument at which this result is evaluated, and its result/destination precision. State whether directions for Hessian products participate analogously. Classify data, controls, and incidental bookkeeping separately. An incidental `Float64` literal must not automatically count as matrix data. If the intended policy instead allows the argument to be rounded to `D`, publish zero for this fixture and remove the promise to preserve the argument’s digits.

For `value_and_prox!`, distinguish the calculation of the point from the subsequent scalar: the scalar refers to the **stored** `y` and agrees with standalone `f(y)` subject to O2/O3, not automatically to a wider value calculation using the original `x`. A wider recovery temporary is not model precision. The opening should say “normally the delivery precision,” since T2 already has data exceptions. Decision 6’s 2,000-dimensional benchmark supports the measured case (wide input and wide output); it does not measure every mixed layout, cached factorization, or GPU call. Compare [T1–T4 and V10](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/docs/design/numerical-contract/08-claude-draft5.md#types) with [decision 6](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/docs/design/numerical-contract/05-decisions.md#6-functions-that-hold-data-t2-366-d6).

## 2. Keep one policy for each kind of decision

**A2/A3/A5/A6/G1/C1.** A2 allows an ordinary group shrinkage to compare against the norm the reduction actually computed. A6 says a decision based on a reduction is certified under A3. G1 promises CPU/GPU branch identity. At a threshold, two accurate reductions can round to `t` and `nextfloat(t)`, respectively: the first drops the group, and the second keeps it. Correct comparison does not create backend agreement.

| Decision | Proposed obligation |
|---|---|
| Entrywise threshold or clamp on identical model operands | Same decision, including the documented tie; do not add a tolerance. |
| Closed form using a computed reduction | Apply the same comparison policy. Require the same branch when the margin exceeds the allowable reduction errors. Within that margin, allow either result consistent with the numerical bound. This describes uncertainty; it does not add a band to the implemented comparison. |
| Search decision covered by A3 | Certify; refine when needed, or signal the dedicated certification failure before modifying caller-owned outputs. |
| Explicit spectral decision inside its band | Follow A5’s stated exception. |

Apply this distinction also to the exact-decision language of decision 38. A3 certification should not become mandatory for every group kernel by implication. Tie the release policy to [A2–A6](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/docs/design/numerical-contract/08-claude-draft5.md#accuracy-and-decisions) and [the device rules](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/docs/design/numerical-contract/08-claude-draft5.md#devices). Document the relaxation of the old bitwise promise; use error budgets against independent references rather than a blanket `isapprox` replacement for tests.

## 3. Limit forward-error bounds to what the map supports

**A5: the spectral error scale.** Nonexpansiveness bounds the change in an *exact* proximal map caused by perturbing the input. It does not account for evaluation of the scalar map, reconstruction or storage of its output. For `LogDetLoss` with `S = 0`, `w = 1`, `γ = 2`, `X = 0`, the exact point is `√2 I`. A bound on total output error proportional only to `eps(W)·‖X‖` is zero. The output is not exactly representable even when both working and destination types are `Float64`. State input perturbation, scalar-map/reconstruction error and destination rounding separately, with a family-relevant output scale. Retain the stronger simple bound where justified; do not extend it solely because a map is nonexpansive. For truncated rank specify the norm, numerical feasibility of the *stored* matrix, and gap dependence only when comparing a particular subspace or matrix. `IndStiefel` also deserves a feasible/near-optimal or conditioning-aware row rather than an implicit general bound. See the [package’s mapped-root formula](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/src/functions/logdet.jl#L8-L40) and [LAPACK’s SVD error analysis](https://www.netlib.org/lapack/lug/node97.html).

**O4: staged rounding of nonconvex wrappers.** A wrapper of a nonexpansive inner map and a wrapper of a discontinuous map cannot have the same additive output-error rule. Set `b = −2⁻²⁶`, `x = 1`, step one, and consider `Translate(NormL0(0.5), [b])`. The threshold is one and equality selects zero. With a `Float32` intermediate, `x − b` rounds to one: the result after shifting back is `b`. In `Float64` the shifted argument is above one: it is retained and the result after shifting back is one. The intermediate rounding is about `1.49e-8`; the output difference is about one. These are the outcomes of the two specified arithmetic routes, not a claim that package tests were run. For Lipschitz stages propagate errors with their amplification factors (additive for nonexpansive stages). At a discontinuity promise the staged model and its tie rule, not closeness to the unstaged point. See [O4](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/docs/design/numerical-contract/08-claude-draft5.md#agreement-between-oracles) and the [hard-threshold definition](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/src/functions/separable.jl#L714-L725).

## 4. Make decision 39 testable

A mapped root smaller than `eps·‖Y‖` does **not** imply a singular stored output: `diag(1, 2⁻⁶⁰)` is representable and positive definite in `Float64`. Test actual stored singularity. Here is a deterministic reference for a mixed call:

- Let `S = zeros(Float64, 2, 2)`, `w = γ = 1`, `a = 2¹⁵` and `X = [0 a; a 0]` in `Float64`. Use a `Float32` destination.
- For `f(Z) = −logdet(Z)`, the exact diagonal entries of the proximal point are `√(a² + 4)/2`; the off-diagonals are `a/2`. Its determinant is exactly one, so its ideal value is zero.
- All four entries round to `16384f0`. The stored matrix is singular. Under the proposed migration the fused value is `+Inf32`; under the documented exception a finite value is possible. Check the stored entries/class separately from the fused value. Pair with a genuinely positive definite stored-output control, then benchmark the ordinary fused path before committing to any extra factorization.

Make **current behavior / 0.6 target / status of maintainer approval** explicit. An intentional transitional exception is not simultaneously a verified implementation bug. Do not use a small eigenvalue alone to reject a stored matrix. Source: [decision 39](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/docs/design/numerical-contract/05-decisions.md#39-logdetloss-the-value-at-a-stored-output-that-rounds-to-singular-and-saturation-o2-o3-r4).

## Small corrections with concrete consequences

- **T3: decimal literals.** An ordinary `Float64` literal need not widen a vector loop, but the package cannot promise identical results for `NormL1(0.1)` and `NormL1(0.1f0)` for every step. Let `γ = 9f0` and `x = nextfloat(0.9f0)`. Combining the binary64 `0.1` with 9 gives a threshold rounding to `0.9f0`; combining the binary32 `0.1f0` with 9 gives `nextfloat(0.9f0)`. Under T3, one prox returns `2⁻²⁴` and the other returns zero. Keep T3; narrow the claim to identical **converted effective parameters**. For multi-factor products, say whether a canonical formation order or a bound rather than cross-platform bit identity defines ties.
- **P1: infinite step.** “The entrywise minimizer” can be ambiguous or nonexistent. For an interval of minimizers the convex limiting prox selects the closest minimizer, where that result applies. For `NegLog` at `x = 0`, the proximal point is `√γ` (unit weight) and there is no finite minimizer. Require a named endpoint behavior or reject unsupported endpoints before writing. Do not assume separability implies existence.
- **E1/O6: exception class.** Keep the nonmutation promise for invalid input without inadvertently replacing `DimensionMismatch` with `ArgumentError`. [The package’s size checks](https://github.com/mlakolar/ProximalBase.jl/blob/36d1bddcfedc156c1fb2aaa1c164d418672b6e3b/src/utils.jl#L221-L243) already distinguish the two. Test a late certification failure in a composed or in-place call, not just a failure on the first block.
- **E2 and solver guidance.** The literal Julia test `result == [NaN, 3.0]` cannot show the stated result; use `isequal` or componentwise `isnan` and ordinary checks. A solver that requires finite evaluations should test finiteness as well as decrease: `Inf <= Inf` is true. The step-stagnation sentence should read “no movement alone proves **neither** conclusion” or “does not distinguish them.”
- **A4: rank comparison.** `√eps` is not uniformly looser than `p·eps`: the ordering reverses for `p > 1/√eps`, about 2,896 in binary32. Describe the package’s convention directly. See [Julia’s documented rank tolerance](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.rank).
- **T6 and the support matrix.** Say whether a *scalar* `Float16` level is rejected, not only `Float16` inputs, outputs, steps and parameter arrays. Label coverage by family and backend. For R4 use the actual result type (`R` for scalar values, `D` for written entries) and absolute magnitude when describing underflow.

## Publication gate

No need for another general redesign. Settle the mixed matrix case; align A2/A3/A6/G1/C1; qualify A5 and O4; and pin the stored-singular fixture. Before calling the document normative, populate at least the affected support and error-bound rows, or mark them pending. On a family with no actual quantitative bound yet, “within the stated accuracy” is an engineering objective, not a usable release test. Keep the short mixed-precision opening and the case table; move the draft changelog out of the eventual user-facing page.
