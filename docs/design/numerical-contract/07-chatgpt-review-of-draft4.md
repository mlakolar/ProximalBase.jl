# ChatGPT's review of Draft 4

*Received from the maintainer on 2026-10-31 and recorded verbatim. It reviews
`06-claude-draft4.md` together with `05-decisions.md`.*

---

**I would adopt Draft 4 as the basis for ProximalBase 0.6, but I would not publish it as the contract yet.** It makes the central mixed-precision decision more clearly than the reader’s edition did. The remaining problems are narrower: two rules can still give different answers for the same call, one spectral guarantee is mathematically too strong, and one existing exception is missing from the migration plan. Those warrant a focused revision, not another rewrite.    

## The decision I would approve: V8 and V9 both return zero

I previously recommended preserving the original `Float64` threshold in V8 and the original `Float64` center in V9. **Having read the alternatives and their costs, I would accept Draft 4’s choice instead.** The package should have one understandable default: ordinary parameters are converted to the call’s designated precision; a conversion that destroys the *range* needed for a representable result is treated separately. For these three cases, that gives a clear answer.  

| Case | Answer I would endorse | Why |
|---|---|---|
| **V3:** A huge ridge weight and tiny step have product exactly one. | `0.5f0` | Combine the parameters before narrowing. Turning them separately into `Inf` and zero destroys their effect. |
| **V8:** An L1 threshold just below one rounds to `1f0`. | `0f0` | This is an ordinary rounding of the effective threshold. The coordinate is exactly on the threshold of the converted problem. |
| **V9:** A ridge center just above one rounds to `1f0`, and the prox stores `1f0`. | Fused value `0f0` | Evaluate the converted model at the stored point. Returning `2⁻⁵⁷` would use the original center for the value but the converted center for the stated problem. |

The distinction is **not** “small answers don’t matter.” V4 still requires recovering a small, normal output even when its shrinkage factor cannot be stored. The distinction is **which model the API promises to evaluate**. That is a defensible choice for a package where `NormL1(0.1)` should work naturally with a `Float32` vector.  

I would change the *justification*, though. A stored `1f0` is exactly one; it is not inherently uncertain by one epsilon. And rounding parameters does not establish backward stability of the subsequent algorithm. The honest rationale is better: **we deliberately choose the converted model for consistency and native-precision performance.** The reported 2.6–3.0× and 1.12–1.24× timings support that tradeoff for two particular soft-threshold kernels on an Apple M3. The benchmark does not establish those costs for centered ridge, GPUs, or every possible implementation.   

## Three changes I would require before sign-off

### 1. T3 and T4 still disagree when the destination is *wider*

T3 says to form a product such as `γλ` in the promoted type **of its factors**. T4 says that, for a wider destination, the calculation uses the wider type **from exactly widened inputs**. Here is a case where that difference changes the active set, without overflow, underflow, or an approximate reference: 

```julia
u = ldexp(1f0, -23)
λ = 1f0 + u
γ = 1f0 - u
x = Float32[1]
z = zeros(Float64, 1)
prox!(z, NormL1(λ), x, γ)
```

The exact product is `(1 + u)(1 − u) = 1 − 2⁻⁴⁶`. Multiplying the two `Float32` factors gives `1f0`, so a literal reading of T3 produces **zero**. Widening the factors first gives the exact product in `Float64`, so T4 produces **`2⁻⁴⁶`**. **I would require `2⁻⁴⁶` and add this as V10.** It is the natural counterpart to V8: a narrow destination permits the narrower model; a wide destination should not inherit an unnecessary narrow multiplication.

The wording I would use is: **“Determine the model precision for each result first. Form combined scalar parameters in a precision no narrower than that model precision or the factors’ promoted precision, with range recovery where necessary. A wider accumulator or temporary used solely for numerical recovery does not redefine the model.”** This is a specification rule, not a requirement to widen an entire vector.

One more test would settle an important consequence. Let `x = Float32[1]`, let the ridge center be `1 + 2⁻²⁸` in `Float64`, and request a `Float64` gradient. Under the natural per-result reading, `value_and_gradient!` returns **`0f0`** but writes **`[-2⁻²⁸]`**: the scalar follows the `Float32` standalone value and the gradient follows the `Float64` standalone gradient. I would endorse that result and say explicitly that **a fused call agrees with its two standalone oracles, even when their designated precisions differ**. Otherwise an implementer may compute both in the wider type and inadvertently change the scalar.  

### 2. A5 promises too much for rank-constrained projections

A5 groups PSD projection, rank-ball projection, and general spectral proximal maps under a forward-error bound of order `p·eps(W)·‖X‖`. A backward-stable SVD is not enough to justify that conclusion for a particular truncated matrix. The distinction matters even in two dimensions.  [Netlib](https://www.netlib.org/lapack/lug/node97.html?utm_source=chatgpt.com)

Take `diag(1 + δ, 1)` and `diag(1, 1 + δ)`, with positive `δ` arbitrarily small. Their distance is of order `δ`, but their best rank-one approximations retain **different coordinate axes** and are order one apart. At an exact tie, the best approximation is not unique. A solver can return an excellent rank-one approximation without returning a matrix close to one particular reference projection.

**I would keep a normwise forward guarantee for PSD projection in the appropriate norm; for rank truncation, promise rank feasibility and near-optimal approximation error.** Promise closeness to a particular truncated matrix or subspace only under a stated spectral-gap assumption. Likewise, give A5 explicit precedence over A3 where the chosen policy is to return an in-band spectral result rather than refine or throw. Otherwise two implementers can legitimately choose opposite behaviors at the same boundary.   [Netlib](https://www.netlib.org/lapack/lug/node97.html?utm_source=chatgpt.com)

### 3. O2/O3 omit a real `LogDetLoss` exception

This is the most important compatibility finding. O2 says the fused value describes the stored output, and O3 names approximate projections as the **only** exception. But the `LogDetLoss` docstring **at the exact commit you linked** says that its stored matrix can round to a singular matrix while its fused value, computed using the mapped eigenvalues, remains finite. Thus `value_and_prox!` can return a finite value while `f(Y) == Inf`. The same docstring allows saturation within the reconstruction error bound, whereas R4 prohibits saturation. These are documented behaviors, not hypothetical edge cases.   

**My preference is to keep O2 as the long-term default:** a solver should know whether the matrix it actually holds is in the log-determinant domain. But make this a deliberate migration. Add an acceptance case with a finite internal value and singular stored output; decide what the fused scalar must be; and measure the cost of detecting the discrepancy. Until then, list the existing exception. Do not tell users there is only one.

## Smaller edits that would prevent implementation mistakes

| Section | Edit I would make |
|---|---|
| **Solver guidance** | Replace “an `Inf` value at a trial point means the trial is outside the domain.” V7 is the draft’s own counterexample: the point is in-domain, the value genuinely overflows, and the gradient is exactly four. A solver requiring a finite objective may reject the trial without diagnosing *why*. Also, a step that does not move can be stagnation **or** an exact fixed point; lack of movement alone proves neither.   |
| **P1: zero steps** | Do not say separability makes the zero-step limit the identity. For `f(z) = −log z` and `x = −1`, the prox is `(−1 + √(1 + 4γ))/2`, which tends to **zero**, not `−1`. Specify which families admit the endpoint and what happens outside their domain.  |
| **E3: infinite inputs** | Qualify “a convex quadratic gives `+Inf`.” For `f(x₁,x₂) = (x₁−x₂)²/2`, approaching `(+Inf,+Inf)` along `(t,t)` gives zero; along `(t,t+1)` it gives one half. The draft’s own rule against choosing a path-dependent limit is the right one.  |
| **A2/A3: exact decisions** | Reserve exact comparison for directly available operands. A group-norm threshold depends on a **reduction**, even though its prox has a closed form. State whether A3 governs that decision; “closed form” alone cannot decide whether certification is required.   |
| **O6: errors before writes** | Distinguish a structural error detectable before work begins from a failure to certify discovered during an iterative solve. A3 prescribes `ArgumentError` for the latter; O6 both promises unchanged outputs on `ArgumentError` and disclaims rollback after failed solves. Specify the output guarantee by **failure category**, not just exception type.   |
| **A4 and verification** | Keep the chosen rank tolerance, but correct the comparison: Julia uses the **smaller** matrix dimension in its default relative tolerance; NumPy uses the **larger**; MATLAB documents `max(size(A))*eps(norm(A))`. Also fix the stale test instruction about not accepting zero in V8/V9—zero is now their target. Use `ldexp` in executable examples such as `Float32[2^100]`, where integer exponentiation otherwise happens before conversion. [Julia Documentation](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)   |

**For the published document, I would keep the rules and move the drafting history out of the reader’s way.** Open with V3, V8, and V9 under the heading *“What problem does a mixed-precision call solve?”* Then present the rules. Keep the rejected alternatives and benchmark discussion in `05-decisions.md`. Finally, separate **decision status** (*proposed* or *approved*) from **implementation status** (*verified*, *known gap*, or *not tested*). Right now the introduction says “Changes behavior” means decided, while the closing register says none of the recommendations is confirmed. The support matrix is also explicitly unfinished.   

**My maintainer vote: approve the converted-model direction, including V8/V9; request changes before adopting the document as normative.** The three things I would want to see in the next revision are the wider-destination V10 test, a defensible rank-projection guarantee, and an explicit decision on `LogDetLoss`. Those would improve the contract more than adding another layer of numerical machinery.
