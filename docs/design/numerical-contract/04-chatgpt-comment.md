# ChatGPT's comparison of Draft 2 with #366

*Received from the maintainer on 2026-10-31, after Draft 3 was written, and
recorded verbatim. The "ProximalBase-numerical-contract…" fragments are
citation markers from the ChatGPT interface, pointing into Draft 2.*

---

**The reader’s edition and issue #366 agree on the core promise, but they are not yet interchangeable.** I would use **#366 to track the architectural decisions**, put the approved contract in `docs/src/numerics.md`, and use the reader’s edition for its explanations and acceptance examples. That matches the maintainer’s amendment: one design document, linked from `architecture.md`, with a short user-facing summary in `interface.md`. The issue still distinguishes proposed policies from settled ones; the reader’s edition should do the same.   ProximalBase-numerical-contract…

The most important difference is **what problem a mixed-precision call is supposed to solve**. I would settle that before giving either document to a coding agent.

## The decision that can change the answer

Consider one coordinate of `NormL1`. The input is `1.0f0`, the step is one, and the penalty weight is `1 − 2⁻³⁰`, stored in `Float64`. The proximal step subtracts the threshold from the coordinate, stopping at zero. There are two possible interpretations:

| Which threshold defines the problem? | Result |
|---|---|
| The `Float64` weight the caller supplied | `1 − (1 − 2⁻³⁰) = 2⁻³⁰`. The coefficient **stays active**. |
| The weight after conversion to `Float32` | The threshold rounds to `1.0f0`. The coefficient **becomes zero**. |

Both answers fit in `Float32`. No number overflows or underflows. The difference is entirely about **when the implementation is allowed to round**. The reader’s edition expressly promises the first answer in case **V8**. Issue #366 says in C2 that conversion to working precision normally defines the problem, following the earlier #238 decision. C3 allows exceptions for loss of significance without a range failure, but only for expressions the family explicitly identifies. So #366 **can accommodate V8, but its general wording does not by itself guarantee V8**. ProximalBase-numerical-contract…  

There is a second case worth deciding at the same time. In **V9**, the ridge center is `1 + 2⁻²⁸` in `Float64`, and the proximal point is written as `1.0f0`. What penalty should `value_and_prox!` report? Evaluating the **original model at the stored point** gives `2⁻⁵⁷`. Evaluating a model whose **center was also rounded to `Float32`** gives zero. Saying “evaluate at the stored point” settles which *point* to use, but not which *model* to use. The reader’s edition deliberately promises `2⁻⁵⁷`. ProximalBase-numerical-contract…

**My recommendation:** retain #366’s native-precision default, and explicitly list the L1 threshold difference and centered-ridge residual as C3 expressions whose stored operands must retain their effect. That preserves these two promises without requiring every `Float32` call with a `Float64` literal to run in `Float64`. It also gives a reviewer an unambiguous acceptance test. ProximalBase-numerical-contract… ProximalBase-numerical-contract…

## Where they agree, and what #366 adds

| Topic | How the two documents fit together |
|---|---|
| **Return types and ordinary range failures** | They agree: `f(x)` and the scalar from `value_and_gradient!` follow the argument; the scalar from `value_and_prox!` follows the destination. An intermediate overflow is not a legitimate excuse for losing a covered normal result. The guide’s ridge example with `γλ = 1` and answer `0.5f0` is an excellent concrete test of C3. ProximalBase-numerical-contract… ProximalBase-numerical-contract…  |
| **Wider destinations and caches** | The guide says that copying a `Float32` result into `Float64` does not restore lost precision. #366 goes further: for covered families, it proposes **recomputing** quantities such as a projection coefficient and its cached denominator in the wider destination type, from exactly widened stored inputs. Data-holding losses are a proposed exception under D6. The guide should say this explicitly: *widening a rounded coefficient* and *recomputing the coefficient* are different operations. ProximalBase-numerical-contract…   |
| **Decisions near a numerical boundary** | #366 is stronger. It asks what certifies **the point actually returned**, how large the uncertainty is, and what happens inside that band. This matters for #151: a multiplier search can have a small residual while the returned point is hundreds of `Float32` epsilons off. “Cannot certify this point at this precision” is not “the constraint is infeasible.” I would import C5 into the guide, not reduce it to “use a suitable tolerance.”  ProximalBase-numerical-contract…  |
| **Float16, BigFloat, and devices** | The guide explicitly makes **no new half-precision promise**. #366 proposes at least `Float32` arithmetic **and accumulation** for supported `Float16` calls, with CPU, GPU, and LAPACK support decided separately. It also raises how a cached `BigFloat` computation should respond to a change of precision. These are additions, not contradictions. They belong in a support table marked *proposed*, *supported*, or *unsupported*—not in an unconditional headline promise. ProximalBase-numerical-contract… ProximalBase-numerical-contract…  |
| **Steps and wrappers** | The guide explains why **genuinely zero** diagonal steps can be invalid. #366 adds the converse: **positive** `Float64` steps such as `[1e-50, 2e-50]` must not be mistaken for zeros merely because the destination is `Float32`. Their ratio can suffice for a projection, but a penalized map also needs their common scale. Likewise, both allow rounding between wrapper stages, but #366 asks for a bound in the scale of those stages, not merely a warning that `prox!` and `prox_entry` may differ. ProximalBase-numerical-contract… ProximalBase-numerical-contract…  |

Conversely, **the reader’s edition contributes behavior that #366 should not leave implicit**. Its cases distinguish a genuine infinite weight from an accidentally overflowed finite weight; show why one NaN invalidates only one coordinate of an L1 prox but the entire coupled L2 block; and require a combined ridge call to return an overflowing scalar **while still writing the correct finite gradient `[4.0]`**. It also names the exceptions to the stored-point rule, including the spectral input sentinel and the fact that a zero indicator value from an approximate projection is not a feasibility certificate. I would keep these as acceptance cases rather than trying to derive them all from a single `Inf`/`NaN` rule. ProximalBase-numerical-contract… ProximalBase-numerical-contract… ProximalBase-numerical-contract…

## What I would give the coding agent

I would **not** hand over the two texts as coequal specifications. I would give the agent one instruction: **build a reconciliation table before proposing patches.** Each row should name the oracle, the mathematical problem after any permitted conversion, the promised result or error bound, the applicable acceptance example, and whether the policy is settled. Start with **V3, V8, and V9**: they distinguish intermediate range, a threshold decision, and the identity of the model used for a fused value. Then add one constrained-search case from #151 and one wider-destination projection from #362, so that ridge does not inadvertently define the architecture of every other method. ProximalBase-numerical-contract…  

I would also reconcile the performance language explicitly: **5% is the guide’s trigger to investigate** an ordinary-path regression; **1.10× is #366’s proposed threshold for maintainer sign-off** on narrow-type costs; machinery needed only for narrow types should leave all-`Float64` workloads within benchmark noise. Those are three different review decisions, not competing estimates of what robustness costs. ProximalBase-numerical-contract… 

**Bottom line:** keep #366’s architecture and decision process; keep the reader’s edition’s examples and explanations. The edit I would make first is to replace the guide’s broad “we work with the numbers you supplied” with a precise distinction: **ordinary parameter conversion is allowed; for a named operation, it is not allowed to erase a distinction that the operation promises to preserve.** Then name those operations. That is what turns two compatible design intentions into one contract.
