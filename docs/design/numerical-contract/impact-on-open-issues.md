# Impact of the #385 decisions on open issues and PRs

*Screened against Draft 5 and decisions 1–39. Checked by Claude before recording: the conflicts in PRs #299, #348 and #353, the stale reference to #283's SLOPE items (fixed by PR #369), and #375's merge were confirmed in the PRs' text and the git history. #307 is still open; whether #375 fixed it was not checked by running its examples. Draft 6 (`11-claude-draft6.md`) and decisions 40–47 answer the "gaps in the contract text" at the end.*

Screened on 2026-10-31 by a Claude subagent: all 76 open issues and the 7 open PRs of mlakolar/ProximalBase.jl other than #384 and #385. #384 was reviewed separately (see the note at the end). Sources:

- `05-decisions.md` (decisions 1–39) and `08-claude-draft5.md` (rules T, R, A, E, P, O, M, G, C);
- issue and PR bodies, plus the latest status and plan comments of the PRs.

"Decision N" means the recommendation in `05-decisions.md`. None of the decisions is approved yet.

**Counts.** Issues: 7 resolved, 3 direction changes, 3 blocked, 27 consistent, 2 unsure, 34 unaffected. PRs: 3 direction changes, 1 blocked, 1 consistent, 2 unaffected.

## 1. Resolved by the contract

| # | title | decisions / rules | consequence |
|:--|:--|:--|:--|
| #257 | LinearConstrained: subnormal steps give NaN or a misleading domain error | 27 (P3), 25/26 (P1) | The issue offers two fixes: "Search for the scaled multiplier `γν` instead of ν … Alternatively, reject steps below `floatmin`". Decision 27 takes the first (scale times ratios, "#257 does this with `μ = max(γ)·ν`"), and P1 rules out the second, because positive subnormal steps are valid. The Float32 summation-order certification that PR #299 hit belongs to decision 18 (see PR #299). |
| #311 | One step contract for every oracle | 25, 26, 22 (P1, P2, E4) | The issue suggests: "a step takes its limit where the limit is unique … Negative and `NaN` steps are rejected everywhere. Array steps on functions that couple entries must be positive and finite". Decisions 25(a) and 26(a) adopt exactly this. Two refinements: decision 26 says the zero-step limit of a convex separable function is the projection onto the closure of its domain (`NegLog` at `-1` gives `0`), and decision 22 settles the zero-level `SparseGroupNormL2` `Inf * 0` case (a zero level is the zero function, so the step copies the argument). |
| #318 | prox! with an argument that partially overlaps the output | 32 (O7) | The issue asks to "Choose one rule … **Either** unalias at the generic entry point … **Or** document that a partial overlap is not supported". O7 keeps "A partial overlap … is not supported (#318)", which is the second option, not the one the issue leans to. Unaliasing stays allowed as an implementation but is not required. |
| #321 | Should gradient outputs be allowed to alias their argument? | 32 (O7) | The issue asks: "Decide whether the convenience is worth these costs". Decision 32 keeps the rule: "Gradient and Hessian-product outputs are separate from their inputs unless a method documents otherwise (#321, #324)". So the answer is no; #367 continues the audit. |
| #367 | Output–input aliasing: audit and settle one package-wide rule | 32 (O7) | The issue asks for "a single documented rule". Decision 32 keeps today's mix: proximal outputs may alias, gradient and Hessian outputs may not, partial overlap is unsupported. That is neither of the comment's "two coherent positions" A and B. The audit, the test sweep and the `Testing` checks remain as work under that rule. |
| #362 | Hyperplane, halfspace and hyperslab: coefficient rounded in the residual's type when the output is wider | 5 (T4) | The issue asks: "Decide the rule: either document that the coefficient is formed in the residual's type, or form `g` and `q` in the output's type when it is wider". Decision 5(a) picks the second option, from exactly widened inputs and caches. The pinned test "wider projection output compatibility" in `test/extreme_normals.jl` (added by PR #353) must flip. |
| #341 | Value oracles that still read every entry of a sparse iterate | 33 (M1, M2) | The issue says: "Each one needs a decision about semantics". M1 and M2 answer it: implicit zeros are entries, skipped "only when they contribute nothing", and stored off-diagonals count twice. So the O(nnz) `NormL2`, `IndSimplex` and `IndBallL0` methods weight off-diagonals twice and treat unstored positions as zeros. *Unsure for `LinearFunction`:* by M1, a non-finite coefficient at an unstored position contributes `c·0 = NaN`, so a sparse `dot` may skip such positions only when the coefficients there are finite, unless the contract rejects non-finite linear coefficients (listed under new gaps). |

## 2. Direction changes

| # | title | decisions / rules | consequence |
|:--|:--|:--|:--|
| #349 | Indicators and indicator conjugates return 0 from value_and_prox! at non-finite inputs | 20 (E2), 28 (O2), 21 (E3) | **NaN entries.** The issue proposes "`Inf` when the input has a non-finite entry, decided on the input before writing". Decision 20 gives NaN for a NaN entry, and E2 says "The indicator shortcut 'value at a projection is 0' does not apply". Case S9 targets the value `NaN` for `IndBox` and `IndBallL2` at `[NaN, 0]`. The issue's side question, whether conjugates that throw should "give `NaN` outputs instead", is answered yes for NaN entries. **±Inf entries.** O2 puts the value at the stored output, not at the input: `IndBox(-1, 1)` at `[Inf, 0]` projects to `[1, 0]`, so the value is `0`, as in S1. *Unsure* where the projection of an infinite entry is itself undetermined (`IndBallL2` at `[Inf, 0]`): decision 20 says only that it "resolves #349 (NaN)". |
| #357 | Value oracles: overflow in transformed arguments (Translate, Precompose, LinearImage, Smoothed) | 10 (R1), S4, status rule | The issue proposes "an optional scaled-argument hook for homogeneous inner functions …, with the remaining inner functions documented as limited to representable arguments". The draft applies R1 to every function, wrappers included (S4), and says a gap "is brought under the rule; it is not documented as a limitation". *Unsure* whether the maintainer means R1 to bind wrappers around arbitrary non-homogeneous inner functions; if not, the contract needs a named exception here (new gaps). |
| #366 | Precision contract: tracker and design | 6, 7, 18, 31, 37, 38 | The design moved to #385, and decision 37 retitles #366 as the implementation tracker. Its text conflicts with the recommendations in five places. (1) C0: "keep their results bit for bit", and C3: "It stays bitwise unchanged"; decision 38 promises accuracy, not bits. (2) D3: "(a) for the generic CPU paths first", plan step 4.5 "Float16 on the CPU (D3)", and the release note "Float16 computed in Float32"; decision 7 rejects Float16 in 0.6. (3) D6: "keep, and state it in C7's table", and C2: "Losses that hold data keep computing in their data's precision"; decision 6 computes in the wider of the data's and the call's precision. (4) C5: "throw an `ArgumentError` that says the result could not be certified"; decision 31 uses a dedicated exception type. (5) C5 still lists "return a documented inexact result"; decision 18 excludes it. The tracker item "#283 items: `SLOPE` …; group weights not representable in the device type" is stale (see stale references). |
| PR #299 | LinearConstrained: subnormal steps give NaN or a misleading domain error (#257) | 31 (O6, A3), 26 (P1), 18, 27 | (1) The approved settings say "a result is returned only when certified, otherwise `ArgumentError`". Decision 31 and A3 want a dedicated "cannot certify" exception, so a solver can tell it from invalid input. (2) The settings also say "steps must be positive and finite (scalar and array), otherwise `ArgumentError`", and the plan lists γ = 0 and γ = Inf as errors. P1 (decision 26) keeps the limits for scalar steps on indicators: `Constrained(IndBox(-1, 1), IndHyperplane(…))` and the simplex are indicators, so steps 0 and Inf should return the projection. (3) The Float32 certification is still blocked on decision 18; the PR is paused for #151. Consistent with the contract: the scaled multiplier `μ = max(γ)·ν` and rejecting a step ratio that underflows (decision 27). |
| PR #348 | Float32 Constrained on a hyperplane (#151) | 38, 18 (A3), 31, 7, C4 | (1) The PR body promises "keeping … Float64 results bitwise unchanged", and the plan's gates require "identical Float64 call sequences" and "the full Float64 bitwise probe on x86". Decision 38 replaces bits with accuracy, and A3's coverage "applies to `Float64` too", so in-band Float64 searches may also refine or throw. (2) Of the options, C ("a keyword or a type parameter") falls under C4 ("Not used. A public `robust = true` switch"), and D ("document only") is ruled out by decision 18 and the status rule. (3) The Float16 rows are moot under decision 7. (4) In-band failures use decision 31's dedicated exception. Options A and B remain, chosen by measurement (decision 18). A costs about 2×, which needs sign-off under C3 (a narrow-type cost above 10%). |
| PR #353 | IndHyperplane and IndHalfspace round the projection coefficient to a narrow output (#218) | 5 (T4); 2, 3 (T2, T3, A2) | (1) The plan adds "Compatibility tests: Float32 inputs into a Float64 `y` equal the base arithmetic (coefficient rounded in Float32, widened), pinned as the documented staged behaviour". Decision 5 requires computing in `D`, so that test must be replaced; doing so also fixes #362 at the same sites. (2) *Unsure, secondary.* The fix forms the coefficient "in the working precision of the residual", which is the promoted type with unrounded Float64 normals and offsets. T2 and T3 put the ordinary path in `W = D` with parameters rounded once, and keep wider arithmetic for R3's range recovery. The results differ only by rounding, except at `IndHalfspace`'s inside/outside branch (`max(dot(a, x) - b, 0)`, `src/functions/sets.jl:858`), which A2 decides on the T3-rounded operands. |

## 3. Blocked until a decision is approved

| # | title | decisions / rules | consequence |
|:--|:--|:--|:--|
| #151 | Float32 Constrained on a hyperplane: the Float32 residual limits accuracy | 18 (A3); also 7, 38, 31 | Paused: "The design choice for this issue is #366's D1", which is decision 18 (certify; refine where cheap, else throw; mechanism by measurement). The proposal also conflicts in three places. "Float64 and Float16: keep Float64 unchanged, bitwise" and the acceptance check "Float64 results are bitwise unchanged" conflict with decision 38 and with A3 applying to Float64. "Decide separately whether Float16 should widen too" is settled by decision 7 (rejected). |
| #255 | Conjugate: compute the proximal operator without dividing by the step | 4 (R3), 3/5 (T3); P1 | Paused with PR #377 ("#255 (draft #377) waits for this contract"). The open questions are whether `γb` and `γr` products that overflow or underflow may become infinite or zero bounds, and the Float32 coefficient narrowing of #218. R3 (decision 4) answers "computed as if from the unrounded value", and T3 says combine before rounding once. The step-free direction itself agrees with P1 (positive tiny steps are valid). |
| #376 | Conjugates of affine sets and polyhedra: project onto γC | 4 (R3), 3 (T3) | Split from #255. It proposes "scaling the right-hand side to `γb`", so it faces the same range question as PR #377 and waits on the same decisions. |
| PR #377 | Conjugates of l1 balls, simplices, halfspaces, hyperplanes and hyperslabs: project onto γC | 4 (R3), 3/5 (T3), 10 (R1) | Paused: "These are precision-contract questions, so #255 resumes after #366". Its open problems are answered by the recommendations. "a γ·b or γ·r product that overflows or underflows is not an infinite or zero bound" is R3. The #218 narrowing on dilated offsets is R3/T3. "value_and_prox! still divides by the step" is a gap under R1 and P1. The PR can resume as planned once decisions 3 and 4 are approved. |

## 4. Proceeds; consistent

| # | title | decisions / rules | consequence |
|:--|:--|:--|:--|
| #142 | Separable operators on SymmetricSparseIterate: symmetric parameters | 33 (M2) | M2 cites the #142 rule and decision 33 keeps it. The remaining work (part D, PR #383) is unaffected. |
| #153 | Benchmark harness: paired comparisons, margin verdict | 35 (C3) | A margin verdict is what applying C3's 5% and 10% thresholds needs. Not a named gap. |
| #216 | Translate and Tilt write their output before the inner function checks the step | 31 (O6), 25 (P1) | Matches O6. Not cited by the draft. PR #304 (merged) added the pre-write step pass for `SeparableSum`, `SliceSum` and `FiberSum`, so probably only the wrappers that #311's comment lists as "Not forwarded" remain. Check whether this issue can close. |
| #218 | IndHyperplane and IndHalfspace round the projection coefficient to a narrow output | 4 (R3) | Listed under R3. Note on the mechanism: the proposal's "working precision of the residual" is the promoted type (see PR #353, item 2). |
| #239 | Remaining pre-write gaps in block oracles | 31 (O6) | Cited as an O6 gap, but probably already done (see stale references). |
| #261 | SymmetricSparseIterate output: prox! from an argument that is not symmetric | 33 (M2), 11 (R2) | The revised proposal (use the symmetric part of the argument) is the #142 rule. Under R2, `(V + V')/2` must be formed without overflow for entries near `floatmax`, as `_SymmetricGradient` already does with `a/2 + b/2`. |
| #283 | Audit the remaining value oracles for overflow, then state the rule once | 10 (R1), 12, 16, 17, 35, 38 | R1 rests on #283 and decision 10 keeps it. Parts 2b (PR #384) and 4 proceed. "Keep ordinary results bitwise" becomes "within accuracy" (decision 38; bitwise stays a test technique). Rows 5 and 9 were done by PR #369. |
| #307 | Spectral(IndBallL0(k)) and wrapped boxes still use absolute eigenvalue tolerances | 16, 17 (A4, A5) | Its proposal is the relative `√eps` rule. Cited as an A4 gap, but probably fixed by PR #375 (see stale references). |
| #313 | IndHyperslab constructor and scaled box membership promote bounds before validating | 9 (T5) | Cited under T5. Its Float16 test cases ("Test with `Float16` … bounds") collide with T6, which rejects Float16 parameter arrays in 0.6; integer bounds still need the fix. T6 is silent on scalar Float16 bounds. |
| #322 | GPU kernels narrow host levels before applying weights | 4 (R3), 3 (T3), 34 (G5) | Cited under R3 and G5. The significand-and-exponent host level, rounded after combining with weight and step, matches T3 and R3. "(bitwise)" for levels that fit becomes "within accuracy" (decision 38). |
| #323 | LinearConstrained: infeasibility the search finds still writes earlier blocks | 31 (O6), 20 (E2), 26 (P1) | Cited as an O6 gap. Rows 1 and 2 stay pre-write work. Row 3 changes. Its input `[NaN, 0]` gets NaN for the whole coupled block under E2, which is a result, not an error. Its scalar `Inf` step on this indicator takes the projection as its limit under P1. So "Leave the `NaN` and infinite-step cases to #311" resolves to "no error". The draft files this issue under validation failures, although the search detects the infeasibility during the work (new gaps). |
| #324 | hessian_vec! with its output aliasing the direction gives wrong values | 32 (O7), 31 (O6) | Cited under O7. Throwing before writing when `Hv === x` or `Hv === v` matches it. |
| #333 | Nested Scaled multiplies scales at construction | 9 (T5) | Cited under T5. The proposal matches "never turns a finite positive parameter into zero or `Inf`". |
| #336 | SphereConstrained{NormL1} on a SymmetricSparseIterate leaves the sphere | 33 (M2), 28 (O2) | A bug against M2 ("counts twice … written once") and O2: the returned value counts one entry while `f(y)` counts both. Not cited. |
| #350 | Mixed-precision Precompose discards an indicator's Inf value | 23 (E5), 20 | Cited under E5. The fix should keep NaN as well as Inf ("keep an indicator's returned value when it is `Inf`" is too narrow). Decision 20 removes the #343 spectral `Inf` at non-finite inputs, so the example's expected `Inf` at `[Inf 0; 0 1]` may change to the value at the stored output (unsure). Both precisions must agree either way. |
| #354 | Pooling adjacent violators overflows at representable projections | 11 (R2), R4 | The scaled recomputation matches R2 if PAVA counts as a closed form. PAVA is an exact finite algorithm, not an iterative search, but the draft does not name it. "bitwise identical" for ordinary inputs becomes a test technique (decision 38). |
| #355 | Per-group steps for Regularize on AtomIterate and SymmetricSparseIterate outputs | 31 (O6) | The feature is structural. Its side note is an O6 gap: the dense scalar-step path "writes into `y` directly", so "output unchanged when `f` throws" fails. |
| #356 | Value oracles: local overflow and underflow in scalar kernels and losses | 10 (R1), 4 (R3), 12 | A gap under R1. A11, A13, A21 and A25 are R3 items. Leaving out A09, A12 and A14 (subnormal values) matches decision 12. "keeping ordinary results bitwise" becomes "within accuracy" (decision 38). |
| #358 | Value oracles: matrix products and quadratic data terms | 10 (R1), 6 (T2) | A gap under R1. The same functions fall under decision 6 (compute in the wider of the data's and the call's precision), so the scaled-product design should be built on T2's working type. B16 (`LogDetLoss` trace) is R1, not decision 39. |
| #359 | Value oracles: sums of function values and calculus wrappers | 10 (R1), 4 (R3) | A gap under R1. B09b is an R3 item, and B12's "effective scalar level" is R3. The draft leaves open the mechanism the issue asks for ("a contract for combining inner values … decided once for all wrappers") (new gaps). |
| #360 | Value oracles: norms rounded before a large weight; signed reductions | 10 (R1), 12 | A gap under R1. Decision 12 says "A subnormal *intermediate* that makes a *normal* result wrong is still a bug". The NaN-to-zeros defect in the same `NormL2`/`GroupNormL2` code is a separate new gap. |
| #361 | Proximal outputs that overflow at finite inputs with a representable result | 11 (R2) | Cited by decision 11 and R2. The power-of-two recovery matches. It can proceed once decision 11 extends the rule to outputs. |
| #363 | Hyperplane and halfspace: integer residual and integer step sums can wrap | 8 (T1, T2) | Forming the reductions in floating point matches "a fixed-width integer argument gives `Float64`". The draft cites #363 for a different rule (see stale references). |
| #371 | Tuple conjugates write earlier output blocks before a later block fails to convert | 31 (O6) | Cited as an O6 gap. |
| #372 | GPU arrays: direct l1-ball projections for the conjugates of NormLinf and GroupNormLinf | 34 (G1, G2) | The device path gives `[0, 0, 0]` where the CPU gives `[1, 0, 0]`, a different zero decision, which G1 forbids. The behaviour #370 pinned in `test/gpu_suite.jl` changes. |
| #373 | Weighted NormL1 value: products that underflow are lost when the total is normal | 10 (R1), 12 | A gap under R1 and decision 12. It touches the same range-flag kernel as #134 and PR #155. |
| #382 | quadratic_model! on SeparableSum does not check the model's block count | 31 (O6) | Pre-write checking matches O6. The issue proposes `DimensionMismatch`, while E1 and O6 say shape violations raise `ArgumentError` (new gaps). |
| PR #383 | SymmetricSparseIterate: documentation of the symmetric-matrix rule | 33 (M2) | It documents the rule decision 33 keeps. No change. |

## Unsure

| # | title | decisions / rules | why unsure |
|:--|:--|:--|:--|
| #317 | GLMLoss quadratic model: Poisson IRLS weights overflow to Inf and working responses to NaN | 11 (R2), R4, 31 (O6) | The issue offers two options: "Throw an `ArgumentError` … when a weight or working response is not finite", or "compute `ηᵢ - ℓ'/ℓ''` in a form that stays finite … and document that weights may be infinite". By analogy, the contract favors the second: R2 recovers a representable entry, R4 makes `exp(710)` legitimately `Inf`, and O6 keeps `ArgumentError` for validation. But `quadratic_model` is outside the contract's stated scope (S2 covers values, gradients, proximal outputs and decisions). |
| #339 | Group penalties classify their weights on every call | 22 (E4), 35 (C2) | E4 says a zero level is "identically zero: value `0`, gradient `0`, and a proximal step that copies the argument". It also says zero scaling "keeps the domain" and that an infinite weight "fixes the coordinate at zero". The current docs (#141, PR #300) fix an infinite-weight entry "at every level", the zero level included. That is why #339 measures a weight scan at a zero level. Read literally, E4 would reverse PR #300 and remove that scan. If infinite weights count as domain, #339 is unaffected. The contract should state the precedence (new gaps). |

## Unaffected

Issues: #134, #136, #137, #146, #157, #158, #159, #160, #164, #187, #192, #225, #226, #227, #264, #269, #296, #312, #315, #316, #325, #327, #340, #342, #344, #345, #346, #364, #365, #368, #374, #379, #380, #381.

PRs: #155, #351.

## A note on "bitwise" acceptance criteria (decision 38)

Several items use bitwise identity as a check:

- "keep ordinary results bitwise": #283, #322, #354, #356;
- a bit-identical base-against-new probe: PR #351, PR #353.

Decision 38 keeps bitwise comparison as a test technique for a change meant to leave results alone, so these need no change of direction. Only where bits are a promise or a gate does decision 38 conflict: #366 (C0, C3), #151 and PR #348 (Float64 invariance).

## Stale or inaccurate references in 05/08

1. **R3 "Items: … #283 (SLOPE weights; group weights not representable in the device type)"**, and the same item in #366's tracker, are stale. PR #369 (merged as `ba507caa`, an ancestor of the draft's base `3f90a29`) fixed SLOPE (row 5) and found that row 9 "no longer reproduces on master; this PR adds a regression test for it".
2. **#239**, cited in O6 and decision 31, is open but probably done. Its last status comment says "What remains of this issue is #274", and #274 was closed on 2026-10-30 by PR #287.
3. **#307**, cited in A4 and decision 16 ("the absolute eigenvalue tolerances of #307"), is open, but PR #375 probably fixed it. #375 merged on 2026-10-31, after the base `3f90a29`. Its run report gives `Spectral(IndBallL0(5))` at an exact rank-5 projection: `Inf` before, `0` after. It applies the relative tolerance "for every indicator `Spectral` admits". Not verified by running #307's reproductions.
4. **#363 is cited in T1 and decision 8** for "an integer destination is rejected for an operation with fractional results". #363 is about integer *argument* and integer *step* reductions that wrap before conversion; it does not mention destinations. The rule may be right, but the citation is wrong.
5. Every other issue cited as a gap is open. The closed issues and PRs cited (#161, #165, #185, #196, #238, #240, #241, #303, #343, #375) appear only as settled history, correctly. #343 is cited as the behaviour decision 20 reverses.

## New gaps without an issue

**Named in the draft (none has its own issue):**

- `NormL2` and `GroupNormL2` turn NaN into zeros (E2).
- `SqrNormL2` cases V3–V7, including the V7 gradient half of O1. #361 covers other families only.
- The `SqrNormL2(0)` gradient at NaN (E4).
- The LAPACK proximal operators check only the output's element type (A5). This is a "New" checklist line inside #366, not an issue.
- The `QuadraticFunction` value goes through a Float32 buffer (T2, decision 6). Related: #321 item 3 ("Its `tmp` buffer has the data's element type … for a Float32 `A` and `x = [1e-50]`, the product is 0").

**Also implied by the draft, with no issue:**

- Decision 6 for the other data-holding functions: `LeastSquaresLoss`, `GLMLoss`, `LogDetLoss`, `KronQuadratic`, `IndAffine`, `IndGraph`, `IndPolyhedron`, `LinearImage`.
- V10: the separable kernels form `γλ` in the argument's type when the destination is wider (T3, T4). #362 covers only the hyperplane family.
- The V9 and V11 value changes (O1, O2).
- T6: reject Float16 in every oracle before writing.
- T7: key `BigFloat` caches by precision.
- The dedicated "cannot certify" exception type (decisions 18 and 31).
- E2: standalone values of indicators, `NegEntropy` and spectral functions at NaN change from `Inf` to NaN (#349 covers only `value_and_prox!`). E4: `NuclearNorm(0)` at NaN.
- Decision 39: migrate `LogDetLoss` to O2, and document saturation.
- O4: document the wrapper bound. A4: remove the fixed tolerances `_largest_eigenvalue` `rtol = 1e-10`, `IndIntersection` `tol = 1e-10` and `Testing.test_kkt` `tol = 1e-8`. Both are only checklist lines in #366.
- Decision 38: reword the interface page ("give the same bits as the ordinary expression", "bit for bit") and C1, and convert the six bitwise test files to reference comparisons where they do not test exact values.

**Gaps in the contract text, found in this screen:**

1. **Zero level against infinite weights (E4, decision 22).** The draft never states which takes precedence. Current docs: an infinite weight fixes the entry "at every level", and zero scaling "keeps domain constraints". E4: a zero level is "identically zero … copies the argument". Affects #339, PR #300 and #141.
2. **Error type for shapes (E1, O6).** The draft says shape violations raise `ArgumentError`, but `src/` throws `DimensionMismatch` in about 150 places, and #382 proposes it.
3. **Failure kinds missing from O6.** (a) Infeasibility that a search finds during the work: #323, which the draft lists under validation. (b) Arithmetic failure detected after the computation: #244's `ArgumentError` in `Conjugate`'s Moreau fallback "where a finite input gives a non-finite result" (#255). Is (b) a stopgap gap under R2 and E5, and with which exception?
4. **Scope (S2).** It omits `kkt_violation`, `quadratic_model`/`quadratic_model!`, the accuracy of `hessian_vec!`, `lla_weights!` and `restrict`. ProximalSolvers' active-set solver relies on `kkt_violation` and `restrict`, and #317 depends on this scope. Possibly related (unsure whether in scope): `kkt_violation` throws `ArgumentError` at an opposing pair of infinite gradient entries (`src/interface.jl:495-496`), where E3 would give NaN.
5. **Gradients outside the domain (E6).** E6 gives values outside the domain (`+Inf`) but not gradients. #226 proposes "`NaN` derivatives outside the domain, as for `LogDetLoss`".
6. **R1 for wrappers around arbitrary inner functions** (#357, #359). The contract needs either an internal protocol (a scaled argument, or a value with an exponent, binding user functions under S4) or a named exception.
7. **T3 and A2 for sets with stored normals.** `IndHyperplane` and `IndHalfspace` compute the residual and the inside/outside decision in the promoted type with unrounded Float64 parameters (`src/functions/sets.jl:849, 858`). That is decision 3's option (b), while T3's cases check only `NormL1` and `SqrNormL2`. Read from the code, not run.
8. **M1 and non-finite linear coefficients** (#341). May `LinearFunction` and `Tilt` coefficients be `±Inf`? If they may, sparse dots cannot skip unstored entries.

## PR #384 (reviewed separately)

PR #384 (#283 part 2b, values of spectral functions at an overflowing spectrum) was put on hold for #385 on 2026-11-01. It depends on the contract in four places:

- **User-defined functions.** Its open decisions A and B list user-defined `f` as a named exception. That conflicts with S4 unless decision 45 (a scaled-value hook that inner functions implement) is adopted, which makes the hook the way a function joins R1 inside wrappers.
- **Bitwise acceptance.** Its acceptance criterion 3 requires "no difference ... in the actual bits of proximal outputs" at ordinary matrices. Under decision 38 that remains a test technique for a change meant to leave results alone, not a promise.
- **Mixed precision.** It promotes "the spectrum type with every level, coefficient and outer scale". Under T3, levels are rounded once to the model precision, and wider temporaries are used only for range recovery (R3). The two differ only in the last bits for representable levels.
- **Non-finite inputs.** Its criterion 4, "non-finite inputs keep their outcome", must follow decision 20, which changes the spectral `Inf` at NaN entries to NaN.
