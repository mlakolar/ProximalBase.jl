# ChatGPT's sign-off review of Draft 8

*Received from the maintainer on 2026-11-01, with the words "This is final review from gpt. go with recommendations", and recorded verbatim. First the summary posted in conversation, then the full review file (`ProximalBase-draft8-signoff-review.md`, with its headings demoted one level). Both review `15-claude-draft8.md` and `05-decisions.md` at `71aa9594`.*

---

## Summary as posted

**Yes. I would approve Draft 8 as the architecture for 0.6 and move to implementation.** I agree with the **final revised recommendations for decisions 1–47**, including **46(b): the normals and offsets that define linear constraints are data**. I would make one local correction to a spectral error bound and clean up the decision log before handing it to a coding agent. Neither warrants a Draft 9 design cycle.   

:chatgpt-content-reference{index="22"}[Download the sign-off review, covering all 47 decisions](sandbox:/mnt/data/ProximalBase-draft8-signoff-review.md)

### What I would consider settled

| Decision | Why I would approve it |
|---|---|
| **Precision and geometry — 2–6, 46** | A hyperplane and the same equation written as one row of `IndAffine` now mean the same thing. That does **not** require treating every array-valued weight or ridge center as data. Internal refinement uses the original call’s effective operands; it cannot silently certify a different, wider model.   |
| **Range and wrappers — 4, 10–12, 45** | V13 settles why a finite final threshold is not enough: an earlier product can already have lost the information that determines whether a coefficient is zero. W1 now has a definite answer, `M/2`, for the supported homogeneous case. The implementation mechanism stays private rather than becoming a premature third-party API.   |
| **Second-order models — 31, 42, 43** | P2, P2m, and P3 draw the right boundary. An unrepresentable **effective coefficient** is grounds to reject a model without changing the caller’s `q`; an overflowing *intermediate* is not. With observation weight `1/8`, the Poisson coefficient `exp(710)/8` fits and the call should succeed.   |
| **Stored points, accuracy, and cost — 18, 19, 28, 35, 38, 39** | A fused rank-indicator value describes the matrix actually stored. A bound that certifies membership avoids another decomposition; an inconclusive bound is not proof of nonmembership. Likewise, approve the stored-point policy for `LogDetLoss`, but leave the verification strategy to a measured implementation. Preserve accuracy and meaningful decisions, not incidental bits.    |

### One mathematical edit before publication

**Use `‖E_D‖_F` for the destination-rounding term throughout A5**, as Draft 8 already does for rank truncation. The PSD/convex spectral paragraph still writes `eps(D)·‖Y‖_F` without qualification. A one-entry example shows the difference: let `h` be the smallest positive `Float32`. The positive `Float64` matrix `[3h/4]` is its own PSD projection. Storing it in `Float32` gives `[h]`, an error of **`h/4`**—not an error on the scale of `eps(Float32)·h`. Use the relative expression as a simplification where justified, not as the general bound. **This changes the statement, not the algorithm, and adds no runtime check.**  [Oracle Documentation](https://docs.oracle.com/cd/E60778_01/html/E60763/z4000ac020351.html?utm_source=chatgpt.com)

### One handoff edit to `05-decisions.md`

I would approve the **revised outcomes**, but I would not tell an agent to “implement every paragraph labeled *Recommendation*.” Some of those paragraphs still say the opposite of the final decision:

| Entry | Instruction the agent should see first |
|---|---|
| **18 / 31** | An uncertifiable search raises the **dedicated certification exception**, not the older `ArgumentError`.   |
| **22 / 40** | A zero level removes the finite penalty; it **preserves domain constraints and hard weights**. It does not always produce the zero function.   |
| **35 / 38** | Keep the performance gates, but **accuracy and decisions—not bitwise identity—are the numerical non-regression rule**.  |
| **43 / 45 / 46** | Reject unusable quadratic models; keep the scaled-value mechanism **private**; treat defining geometry as **data**. Earlier recommendations for all three remain visible.  |

Put one **Final recommendation** at the top of each entry and mark the older alternatives *superseded*. Keep the history—it explains why the choices were made—but do not make the implementer reconstruct the decision from chronology.

**My stopping point:** approve the revised decisions, make those two documentation edits, and start the family-by-family work. That is **design sign-off, not release sign-off**: the draft still records failing cases, P2m is marked not tested, and the support matrix is unfinished. I checked the rounding arithmetic independently; I did not rerun the Julia suite or benchmarks. The next useful evidence is passing conformance tests and measured costs, not another architecture review.

---

## Full review file

## ProximalBase.jl numerical contract: Draft 8 sign-off review

**Review target:** [`15-claude-draft8.md`][draft] and [`05-decisions.md`][dec], commit `71aa9594c94698da7c2e404871c10824630fc98d`.

**Recommendation:** Adopt the 47 decisions in their **final, revised sense**, with the local amendment to decision 19 below. Start implementation. Do not commission another architecture draft. Record the maintainer’s own approval separately from this recommendation, and keep approval separate from evidence that a package version implements it.

### Why the architectural review can close

| Draft 7 question | Draft 8 resolution | Vote |
|---|---|---|
| Do a hyperplane and the same equation written as a one-row affine set denote the same constraint? | T2 / 46(b): normals and offsets defining linear sets are data. This does not reclassify all parameter arrays. Algorithms can round differently without changing the equation. | Approve. Benchmark mixed geometry rather than promising its cost. |
| Can a normal final threshold hide damage to a partial product? | R3 / V13: use range recovery if a partial operation loses bits needed later. The weighted L1 example retains `2^-54`, not zero. | Approve. Test behavior rather than prescribing an extra scan. |
| Can refinement certify a different model? | T3 / A3: no. Refine the effective operands of the original call; a new call with a wider destination can have a different model. References respect the same conversions and wrapper stages. | Approve. |
| Is a stored rank projection necessarily a member? | A5 / L5: only when the perturbation bound certifies it. Otherwise evaluate membership of the stored point for the fused scalar. Failure to certify is not proof of nonmembership. | Approve. A certificate can avoid an extra decomposition on ordinary inputs. |
| Does the range promise extend to arbitrary wrappers or unusable model objects? | R1 distinguishes scaled inner results from overflowing transformed inputs. W1 is covered through homogeneity, not a new public API. P2 and P2m reject genuinely unrepresentable effective quadratic coefficients without changing the destination; P3 must succeed when the weight brings the effective coefficient back into range. | Approve. |

### One local correction to the mathematical text

**Decision 19 / A5.** Rank truncation already uses `‖E_D‖`, the norm of the actual or a bounded storage perturbation. Use the same term for the stated bound for PSD and convex spectral proximal outputs. Write `Y_W` for the result before storage, `Y_D` for the result delivered, and `E_D = Y_D − Y_W`. The triangle inequality adds `‖E_D‖_F` to a justified working-precision forward-error bound. A relative approximation such as `eps(D)·‖Y_W‖_F` is useful when its assumptions hold, but it is not a uniform bound through gradual underflow. This is a correction to how the promise is stated; it requires neither an extra production pass nor a change in decision 19’s boundary policy.

A one-entry witness: let `h = 2^-149`, the first positive `Float32`. A positive `Float64` matrix `[3h/4]` is its own PSD projection. Its correctly rounded `Float32` entry is `h`, so the storage error is `h/4`, not of the order of `eps(Float32)·h`. For the same reason, qualify T3’s “at most eps/2 relative” as a bound for ordinary in-range rounding with a normal result. See the [primary reference on subnormal rounding][underflow].

**Implementation note for L5.** Test the returned point and the fused value independently. The draft reports a stored matrix `h·[2 1; 1 1]`, whereas direct rounding of the exact rank-one input gives `h·[1 1; 1 0]`. For the reported matrix, the Frobenius residual from the exact input is `h·sqrt(6769)/64 ≈ 1.28553h`. Rounding four entries near this scale accounts for at most `h` in Frobenius norm. Thus, for this fixture the reported result also merits a pre-write accuracy check; correcting the fused `0` to `Inf` alone does not demonstrate that the projection met its residual bound. This is an inference from the stored entries reported in the draft, not a reproduced Julia run.

### Review of the 47 decisions

“Approve” below refers to a **design target**, not to a claim about shipped behavior or a particular benchmark result.

| Decision IDs | Recommendation to ratify |
|---|---|
| **1–9: return types, working precision, construction** | Approve T1–T8. Determine model precision per result from delivery type, evaluation argument, direction where relevant, and effective data roles. Do not widen an entire array merely because a scalar level is a `Float64` literal. Use the effective combined parameter rule, not factor-by-factor narrowing. Preserve role-specific `Float16` support and precision-aware `BigFloat` caches only where supported. |
| **10–13: range and accuracy** | Approve the finite normal-range goal and the distinction between genuinely subnormal final results and recoverable intermediates. Require a per-family scale, test budget or certificate. Do not advertise an unestablished quantitative bound as verified. |
| **14–19: decisions and spectral methods** | Approve the separation of exact comparisons on identical operands, comparisons on rounded reductions, certified iterative decisions, and spectral decisions near a boundary. Keep the documented rank threshold as a convention rather than a universal mathematical rank. Apply the A5 storage-term correction above. Which refinement strategy pays for itself remains an implementation decision. |
| **20–27: exceptional values, domains and steps** | Approve the family-dependent `NaN`, `Inf`, zero-weight and domain conventions. Rule 40 qualifies rule 22. Admit step endpoints only where documented; validate inadmissible metrics before mutation; preserve the scale when normalizing diagonal steps for penalized maps. |
| **28–34: fused results, wrappers, failure, storage, devices** | Approve the stored-point default. Keep the inexact indicator’s ideal zero a specifically documented non-certificate; decision 39 governs the `LogDetLoss` transition. Use staged semantics for discontinuous wrapper maps. Preserve supported aliasing and sparse multiplicity. GPU parity is for advertised modes under the A2 decision rules, not a universal bitwise or in-band branch guarantee. |
| **35–39: cost, solver responsibilities, release, reproducibility, log determinant** | Approve 5% as an investigation trigger and >10% narrow-type overhead as a sign-off trigger. Treat narrow-only cost on `Float64` as a separate check. Adopt 38’s accuracy/decision policy, not 35’s historical bitwise language. Approve stored-point `LogDetLoss` semantics; benchmark the verification strategy rather than prescribing an unconditional second factorization. |
| **40–44: preserved constraints, errors and second-order outputs** | Approve domain-preserving zero scaling (or rejection when no supported representation exists), `DimensionMismatch` for sizes, a dedicated numerical exception for uncertifiable results, and the distinction between an oracle output and a model a caller will reuse. For quadratic models, judge the effective coefficient, not an overflowing intermediate. Outside-domain derivative rules apply to supported derivative oracles. |
| **45–47: wrappers, geometry, finite linear coefficients** | Approve a **private** scaled-result mechanism, supported transformed-input recovery for homogeneous inner functions, option **46(b)** for geometry, and finite `LinearFunction`/`Tilt` coefficients. Do not publish a third-party extension point prematurely. Document any obligations on callers who mutate retained arrays after validation. |

### Make the ledger safe for a coding agent to follow

Put a single **Final recommendation** and, once the maintainer votes, a separate **Decided** line at the top of each entry. Keep prior alternatives and revisions below, explicitly marked historical. Important examples:

- **2/6:** use Draft 8’s per-result model precision; “output type” in the index is not the entire rule.
- **15/18/31:** not every decision on a sum requires certification; an iterative certification failure uses the dedicated exception, not the old `ArgumentError` recommendation. Invalid input and proven infeasibility have separate rules.
- **22/40:** zero removes the finite penalty, not its domain constraints or admitted hard weights.
- **35/38:** preserve documented accuracy, class and applicable decisions, not incidental Float64 bits. The runtime gates remain in force.
- **43/45/46:** unrepresentable effective model coefficients raise before mutation; the scaled-result capability is private; defining geometry is data. Each entry still contains earlier recommendations saying otherwise.

None requires another design decision. They are handoff edits to prevent two implementers reading the same ledger from taking different actions.

### Suggested maintainer sign-off

> I approve decisions 1–47 in their final revised form as reflected in Draft 8, including option 46(b). In A5, use the actual or bounded destination-storage perturbation for convex spectral outputs, with a relative storage bound only where justified. The final decision line supersedes historical recommendations. Proceed with the family-by-family plan, conformance tests and benchmarks; policy approval is not a claim of implementation or release readiness.

**Evidence boundary.** The reported package observations are those in the [acceptance table][cases], tagged there by revision. For this review I independently checked the subnormal storage arithmetic, the L5 residual, the exact V13 threshold and the P3 reference magnitude. I did not run Julia, CI or performance tests. P2m, among others, is explicitly marked not tested in the draft. Populate the family/backend support matrix with evidence and unresolved bounds before representing the contract as delivered.

[draft]: https://github.com/mlakolar/ProximalBase.jl/blob/71aa9594c94698da7c2e404871c10824630fc98d/docs/design/numerical-contract/15-claude-draft8.md
[dec]: https://github.com/mlakolar/ProximalBase.jl/blob/71aa9594c94698da7c2e404871c10824630fc98d/docs/design/numerical-contract/05-decisions.md
[cases]: https://github.com/mlakolar/ProximalBase.jl/blob/71aa9594c94698da7c2e404871c10824630fc98d/docs/design/numerical-contract/15-claude-draft8.md#acceptance-cases
[underflow]: https://docs.oracle.com/cd/E60778_01/html/E60763/z4000ac020351.html
