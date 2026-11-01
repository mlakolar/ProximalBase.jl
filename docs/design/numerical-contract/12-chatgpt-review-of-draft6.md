# ChatGPT's review of Draft 6

*Received from the maintainer on 2026-11-01 and recorded verbatim. First the
summary posted in conversation, then the full review file
(`ProximalBase-draft6-review.md`, with its headings demoted one level to nest
under this file's). Both review `11-claude-draft6.md` and
`05-decisions.md` at `73c00b1e`.*

---

## Summary as posted

**I would approve Draft 6 as the architecture and start implementation planning.** It resolves the main Draft 5 questions: V12 protects a wide argument even when the gradient destination is narrow; A2 now explains when CPU and GPU threshold decisions may differ; O4 no longer promises that rounding through a discontinuous prox produces a nearby answer; and O3 gives `LogDetLoss` a test that distinguishes the ideal matrix from the one the caller actually receives. I would not reopen those decisions.   

**I would not publish it verbatim as the 0.6 contract.** The most important remaining issue is the new public extension point in decision 45: it promises to solve two problems, but its interface supplies the information needed for only one. The other changes below are local enough that they need not hold up the rest of the work.  

:chatgpt-content-reference{index="26"}[Download the full review as repository-ready Markdown](sandbox:/mnt/data/ProximalBase-draft6-review.md)

### 1. Decision 45: a scaled return value cannot recover a lost argument

This is the one architectural decision I would **not** approve as written. For `Scaled(f, α)`, the proposed hook makes sense: `f` receives a valid argument, its value is too large to return as a `Float64`, and returning a significand and exponent lets the wrapper apply `α` before the final rounding. For `Translate` and `Precompose`, however, the quantity that overflows may be the **argument passed into `f`**.  

Issue #357 already contains the decisive example. Let `M = floatmax(Float64)`:

```julia
Translate(NormL1(0.25), [-M])([M])
```

The mathematical inner coordinate is `M - (-M) = 2M`. The requested value is `0.25 × 2M = M/2`, which fits. But once the wrapper passes `[Inf]` to the inner function, **it is too late to fix the calculation by changing what the inner function returns**. In particular, it cannot tell this input from a genuine infinity. 

**My recommendation:** promise scaled **results** for supported value-scaling cases. For overflowing transformed **arguments**, use a known algebraic identity, an explicitly supported scaled-input path, or document the limitation. I would keep the mechanism private for now rather than commit third-party authors to a public hook that does not deliver the advertised guarantee.

### 2. A5: include destination rounding in the rank and Stiefel guarantees

The revised spectral section correctly separates input error, reconstruction error, and rounding to the destination. But its rank-truncation and `IndStiefel` clauses still specify accuracy for the **stored output** using `eps(W)` alone. Two small tests show why `D` matters even if the computation in `W` is perfect. 

| Test | What an ideal computation followed by `Float32` storage gives |
|---|---|
| Rank-one projection of the `Float64` matrix `diag(1 + 2⁻²⁸, 0)` | The input is already rank one, so the optimal residual is **zero**. The stored result is `diag(1, 0)`, so its actual residual is **2⁻²⁸**. |
| Stiefel projection of the `Float64` column `(1, 1)` | The exact unit column is `(1/√2, 1/√2)`. After storing both entries in `Float32`, the orthogonality error `abs(y'y − 1)`, evaluated in higher precision, is **3.42 × 10⁻⁸**. |

I independently checked these numbers; they are not claimed Julia package runs. **The fix is to the bound, not the algorithm:** include the destination-rounding contribution in both stored-output guarantees, and say which precision defines numerical-rank membership for the stored matrix. This is separate from the quality of LAPACK’s decomposition.  [Netlib](https://www.netlib.org/lapack/lug/node97.html?utm_source=chatgpt.com)

### 3. Decision 43: test the returned quadratic model, not just its fields

I agree with extending the contract to second-order oracles. But a `quadratic_model` is itself a function that a solver will evaluate. Correctly reporting that one of its stored weights overflowed does not make the returned function usable. Decision 43 leans toward allowing an IRLS weight to become `Inf`; issue #317 makes this a practical concern.   

Take a single Poisson observation with response zero, unit weight, and predictor `η = 710`. The exact IRLS weight is `exp(710)`, which exceeds `Float64` range; its working response is **709**. Now evaluate the returned weighted least-squares model at **709.5**. Its contribution is

\[
\tfrac12 e^{710}(709.5-709)^2
= e^{710}/8
\approx 2.79249\times10^{307},
\]

a **normal, finite `Float64` result**. A model that retains only an `Inf` weight has lost the information needed to compute it. This example uses the package’s least-squares representation, without an added Taylor constant.  

**I would allow either of two explicit outcomes:** return a model that retains enough scale to evaluate that finite result, or report before mutating `q` that the requested model cannot be represented. For 0.6, I would prefer the documented failure over introducing a new model representation solely for this case. What I would avoid is returning an ordinary-looking quadratic model with no promise about its next evaluation.

### Three short edits before publication

| Clause | Change I would make |
|---|---|
| **T6 · Float16** | Say **which operand roles are rejected**, not “reject when the model precision would be `Float16`.” Under T3, a `Float16` argument and `Float64` destination have `Float64` model precision, yet the acceptance table says to reject the call. The apparent intent is straightforward: reject half-precision arguments, destinations, directions, and numerical data in 0.6; permit half-precision parameters and steps in otherwise supported wider calls.    |
| **S2 · `restrict`** | Restore the precision of the existing interface wording: **the objective restricts up to a finite additive constant, and the gradient restricts by taking its supported entries**. The restricted prox solves a constrained minimization; it is not generally a slice of the full prox. For `Q = [2 1; 1 2]`, `f(z) = z'Qz/2`, unit step, and input `(1,0)`, the full prox has first entry **3/8**, while the prox restricted to the first coordinate is **1/3**.   |
| **E4 · Zero scaling** | Keep the existing qualification: **reject zero scaling when no domain-preserving proximal representation exists**. Preserving the domain of `NegLog` at zero scale yields the indicator of the *open* positive half-line. From `−1`, there is no closest point in that set. That is different from P1’s valid statement that the prox of `NegLog` *tends to* zero as its positive step tends to zero.   |

One wording change also matters for users: T2 says that `IndHyperplane` and a one-row `IndAffine` with the same numbers can differ “in the last bits.” **They can define different converted constraints.** Take the `Float64` normal `(1, 1 + 2⁻²⁵, 1)` and the `Float32` point `(2²⁵, −2²⁵, 0)`, with zero offset. When the normal rounds to `Float32`, the point is already on the plane and its third coordinate stays **zero**. When the normal is retained as `Float64` matrix data, the residual is `−1` and the exact projected third coordinate is approximately **one third**. I would keep the data/parameter distinction if that is the intended cost tradeoff, but describe it as a **model choice**, not merely a last-bit effect.  

**My maintainer vote:** ratify the per-result precision rule, the decision table, and the stored-point policy now. Before freezing the public contract, narrow decision 45, correct the stored-matrix bounds, and settle what happens when a returned quadratic model cannot represent its own coefficients. **None of those decisions requires another general rewrite or a delay to the ridge, threshold, and fused-oracle work.**

---

## Full review file

## Review of ProximalBase.jl numerical contract, Draft 6

**Reviewed:** [`11-claude-draft6.md` at `73c00b1`](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/docs/design/numerical-contract/11-claude-draft6.md). **Disposition:** approve the core architecture for implementation planning; amend the specific promises below before publishing them as normative. The draft itself records proposals, not maintainer approvals or release-wide conformance. This review assesses that pinned text, not the tip of the branch.

### What I would sign off now

- **T2/T3/O1/O2:** choose precision per result from its delivery type, evaluation argument, relevant data and, for Hessian actions, direction. Distinguish parameters from data. In particular the scalar returned by `value_and_prox!` normally describes `f(y)` at the stored `y`; the original input's precision need not govern that scalar. V3, V8–V12 expose the actual choices instead of relying on words like “promote.”
- **A2/A3/A6/G1/C1:** direct comparisons on identical model operands, comparisons of rounded reductions, certified search decisions, and spectral-band decisions now have distinct obligations. Keep the freedom to improve reductions. Do not reinstate a general bitwise-regression requirement.
- **O3/O4/O6:** keep the deterministic `LogDetLoss` fixture and positive-definite control; adopt the proposed stored-point target, with its cost measured. Keep staged rounding and ties for discontinuous wrapper maps. Distinguish invalid input, detected infeasibility, inability to certify, and external-library failure. Test non-mutation for failures covered by that promise, including late failures and aliasing.

These address the principal Draft 5 findings. The new recommendations about extension authors and second-order models need separate scrutiny. [Draft: types](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/docs/design/numerical-contract/11-claude-draft6.md#types) · [agreement between oracles](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/docs/design/numerical-contract/11-claude-draft6.md#agreement-between-oracles).

### 1. Decision 45: distinguish a scaled result from a scaled argument

**Finding.** Returning an inner value as a significand and an exponent can help `Scaled(f, α)` when the input to `f` is representable but its unscaled value is not. The same interface does not recover an argument that `Translate` or `Precompose` has already converted to infinity. A longer-range *return* cannot repair a lost *input*.

**Acceptance example.** Issue #357 already contains one: write `M = floatmax(Float64)` and evaluate `Translate(NormL1(0.25), [-M])([M])`. The inner mathematical coordinate is `2M`; the requested result is `M/2`. If the wrapper passes `[Inf]`, a correct scaled-value method for the inner norm is no longer being asked about `2M`. It cannot distinguish it from another overflowing coordinate or a genuine infinity.

**Recommended wording.** “A scaled-value capability covers out-of-range inner *results* at representable inputs. Recovery for an out-of-range transformed *argument* requires a supported transformation, algebraic identity, or an inner method that can consume a scaled argument. Without such a capability that combination is not covered.” Preserve meaningful infinities. Mark the distinction in S4 and the support table. Keep the mechanism private until the second use case or a concrete third-party extension justifies a public protocol; this need not delay `Scaled` or homogeneous built-ins.

[Decision 45](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/docs/design/numerical-contract/05-decisions.md#45-values-of-wrappers-around-arbitrary-inner-functions-r1) · [issue #357, B05/B06](https://github.com/mlakolar/ProximalBase.jl/issues/357).

### 2. A5: budget destination rounding in *every* stored-matrix guarantee

The revised convex spectral paragraph separates input perturbation, evaluation/reconstruction and destination rounding. The rank and Stiefel paragraphs revert to bounds using `eps(W)` alone. Two tests require no assumptions about implementation quality:

| Fixture | Even an ideal computation followed by storage gives… |
|---|---|
| Rank-one projection of `diag(1 + 2^-28, 0)`, Float64 input, Float32 output | Optimal residual is zero; the stored diagonal is `(1, 0)`, so the Frobenius residual is `2^-28`. |
| Stiefel projection of the Float64 column `(1,1)`, Float32 output | The closest unit column has entries `1/√2`. Rounding both and evaluating their Gram residual in higher precision gives `abs(y'y - 1) ≈ 3.42285418e-8`. |

**Fix.** Add destination storage error to the rank near-optimality inequality and to the Stiefel orthogonality bound. In the latter, if `Y = Q + E`, the Gram error contains `Q'E + E'Q + E'E`. State the norm and which precision governs numerical-rank membership of the stored matrix. Testing should separate decomposition quality before the write from the certificate for what was written. A safe way to express the rounding contribution is in terms of the actual or bounded norm of the conversion error, avoiding a uniform relative-error claim on subnormals. Precise dimension factors can await a family-level derivation and status row.

[Draft A5](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/docs/design/numerical-contract/11-claude-draft6.md#accuracy-and-decisions) · [LAPACK's SVD error analysis](https://www.netlib.org/lapack/lug/node97.html).

### 3. Decision 43: a result that is itself a function needs a usable contract

**Poisson IRLS.** At response zero, unit observation weight, and predictor `η=710`, the exact IRLS curvature is `v=exp(710)`, while the working response is `709`. The curvature exceeds the Float64 range. Nevertheless the weighted least-squares objective returned by this construction, evaluated at a new predictor `709.5`, is `v*(709.5-709)^2/2 = exp(710)/8 ≈ 2.79249346e307`, a normal Float64 number. This is the least-squares representation without an added Taylor constant, as in the package. An `Inf` field by itself has lost the magnitude needed to answer that call. Checking only `q.weights` and `q.y` does not check that a solver can use `q`.

**Recommendation.** Choose either (i) retain enough scale to evaluate the returned model on its stated domain, or (ii) document a construction failure when that model cannot be represented, before mutating `q` in the in-place call. For 0.6, (ii) is reasonable where (i) would force a new representation. Do not confuse overflow of a finite curvature with an intentional infinite model weight. Use the example above to test the *returned function* as well as its fields.

**Restriction.** Define `restrict(f,S)(u) = f(E_S u) + C` where supported, with `E_S` embedding by zero and `C` finite and independent of `u`. Gradients restrict as expected. But the restricted prox is the minimizer over the support, not the slice of the unrestricted prox. With `Q = [2 1; 1 2]`, `f(z)=z'Qz/2`, unit step, and argument `(1,0)`, the full prox is `(3/8, -1/8)` while restriction to the first coordinate gives `1/3`. This exact-arithmetic difference is not numerical error. Qualify S2's “oracles equal” and carry over the already documented unsupported restrictions.

[Decisions 43–45](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/docs/design/numerical-contract/05-decisions.md#43-the-scope-of-the-contract-s2) · [issue #317](https://github.com/mlakolar/ProximalBase.jl/issues/317) · [IRLS model construction](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/src/second_order.jl#L196-L229) · [existing restrict contract](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/src/interface.jl#L527-L554).

### 4. T6: decide by role, not an ambiguous test on model precision

T3 makes `W` the widest relevant precision. Thus a Float16 input with a Float64 destination has `W=Float64`, yet the F16 acceptance row says to reject it. The same issue occurs for a Float32 input with a Float16 destination. The support matrix's “rejected everywhere” also reads as rejecting parameter-only uses that T6 explicitly permits.

**Recommended 0.6 rule:** reject Float16 in the **argument, destination, direction or numerical-data** role, before writes, independently of `W`; permit a Float16 **parameter or step** that widens exactly into an otherwise supported call. Tests: Float16 input → Float64 destination (reject); Float32 input → Float16 destination (reject); Float32 input/destination with a Float16 level or step (accept, subject to ordinary parameter validity). There is no need to implement half-precision vector kernels to make this distinction. [T6 and support matrix](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/docs/design/numerical-contract/11-claude-draft6.md#support-matrix).

### Two user-facing clarifications

**Decision 46, the meaning of data versus parameters.** The distinction is defensible, but the difference need not be confined to the last bits of each coordinate. Take `a = (1, 1+2^-25, 1)` in Float64, `b=0`, and `x=(2^25,-2^25,0)` in Float32 with Float32 delivery. As a rounded hyperplane normal, `a` becomes `(1,1,1)` and `x` is exactly on the plane: its projection's third coordinate is zero. As a Float64 row of an affine matrix the residual is `−1`; the exact projected third coordinate is `1/(2+(1+2^-25)^2) ≈ 1/3`. Global error may be small relative to this large `x`; the effect is not a last-bit change for the third coordinate. These are the two mathematical models under T2/T3, not a claim about current outputs. Say so explicitly. [Decision 46](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/docs/design/numerical-contract/05-decisions.md#46-the-normals-and-offsets-of-sets-t2).

**Decision 40, open domains.** Keep the restriction in `Scaled`'s existing documentation: reject zero scaling if there is no supported domain-preserving proximal representation. For example, preserving `NegLog`'s domain under zero scaling gives the indicator of `(0,∞)`. Its distance minimization from `−1` does not attain a minimum. This differs from P1's legitimate limiting prox *as the positive step tends to zero*, which is zero. Do not silently close the domain or promise the prox exists. [Existing zero-scaling rule](https://github.com/mlakolar/ProximalBase.jl/blob/73c00b1e53546b966593d18e7fa6e280749ab43a/src/calculus.jl#L43-L75).

### Handoff

Do not reopen V3–V12, require blanket widening, or hold unrelated implementation work for the public-hook decision. Fix the clauses above and give each affected family a support row distinguishing approved policy, checked examples, and pending bounds. Three useful economical additions to the core tests are the wide-to-narrow spectral fixture, the restricted quadratic, and a mixed-role Float16 rejection. The transformed-input and Poisson examples should guard the separate decisions on capabilities and representability. Record current results by baseline SHA, not just “today.”

**Evidence boundary.** The numerical examples above are independent mathematical checks, not runs of ProximalBase.jl. I checked the restriction and projection arithmetic with exact rationals, and the storage and exponential examples with binary32 conversion and higher-precision arithmetic. Implementation status and timings quoted by the draft remain the draft authors' evidence.
